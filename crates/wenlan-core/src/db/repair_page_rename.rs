// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{
    error::WenlanError,
    post_write::RepairWriteProof,
    repair::{RenamePageTitleRecoveryArtifact, RepairArtifactStore},
};
use std::{fmt::Write as _, path::Path};
use wenlan_types::repair::{
    RepairApplyReceipt, RepairManifest, RepairMutation, RepairRollbackPayloadV2, RepairTarget,
    RepairWriter,
};

#[cfg(test)]
pub(crate) type RenamePageTitleTestCheckpoint =
    Box<dyn FnOnce() -> Result<(), WenlanError> + 'static>;

#[cfg(test)]
#[derive(Default)]
pub(crate) struct RenamePageTitleTestCheckpoints {
    pub(crate) before_projection_write: Option<RenamePageTitleTestCheckpoint>,
    pub(crate) after_target_write: Option<RenamePageTitleTestCheckpoint>,
    pub(crate) before_projection_restore: Option<RenamePageTitleTestCheckpoint>,
    pub(crate) after_target_restore: Option<RenamePageTitleTestCheckpoint>,
    pub(crate) after_commit_before_artifact: Option<RenamePageTitleTestCheckpoint>,
}

#[cfg(test)]
tokio::task_local! {
    static RENAME_PAGE_TITLE_TEST_CHECKPOINTS:
        std::cell::RefCell<RenamePageTitleTestCheckpoints>;
}

#[cfg(test)]
pub(crate) async fn with_rename_page_title_test_checkpoints<T>(
    checkpoints: RenamePageTitleTestCheckpoints,
    future: impl std::future::Future<Output = T>,
) -> T {
    RENAME_PAGE_TITLE_TEST_CHECKPOINTS
        .scope(std::cell::RefCell::new(checkpoints), async move {
            let output = future.await;
            let remaining = RENAME_PAGE_TITLE_TEST_CHECKPOINTS.with(|checkpoints| {
                let checkpoints = checkpoints.borrow();
                [
                    (
                        "before_projection_write",
                        checkpoints.before_projection_write.is_some(),
                    ),
                    (
                        "after_target_write",
                        checkpoints.after_target_write.is_some(),
                    ),
                    (
                        "before_projection_restore",
                        checkpoints.before_projection_restore.is_some(),
                    ),
                    (
                        "after_target_restore",
                        checkpoints.after_target_restore.is_some(),
                    ),
                    (
                        "after_commit_before_artifact",
                        checkpoints.after_commit_before_artifact.is_some(),
                    ),
                ]
                .into_iter()
                .filter_map(|(name, present)| present.then_some(name))
                .collect::<Vec<_>>()
            });
            assert!(
                remaining.is_empty(),
                "unconsumed rename page title test checkpoints: {}",
                remaining.join(", ")
            );
            output
        })
        .await
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum TestCheckpointKind {
    BeforeProjectionWrite,
    AfterTargetWrite,
    BeforeProjectionRestore,
    AfterTargetRestore,
    AfterCommitBeforeArtifact,
}

#[cfg(test)]
fn run_test_checkpoint(kind: TestCheckpointKind) -> Result<(), WenlanError> {
    RENAME_PAGE_TITLE_TEST_CHECKPOINTS
        .try_with(|checkpoints| {
            let hook = {
                let mut checkpoints = checkpoints.borrow_mut();
                match kind {
                    TestCheckpointKind::BeforeProjectionWrite => {
                        &mut checkpoints.before_projection_write
                    }
                    TestCheckpointKind::AfterTargetWrite => &mut checkpoints.after_target_write,
                    TestCheckpointKind::BeforeProjectionRestore => {
                        &mut checkpoints.before_projection_restore
                    }
                    TestCheckpointKind::AfterTargetRestore => &mut checkpoints.after_target_restore,
                    TestCheckpointKind::AfterCommitBeforeArtifact => {
                        &mut checkpoints.after_commit_before_artifact
                    }
                }
                .take()
            };
            hook.map_or(Ok(()), |hook| hook())
        })
        .unwrap_or(Ok(()))
}

pub(crate) async fn rename_page_title_cas_inner<F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &RepairRollbackPayloadV2,
    page_root: &Path,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    let (page_id, scope, before_title, after_title, after_embedding_hex) =
        match (manifest.target(), manifest.mutation(), rollback) {
            (
                RepairTarget::PageProjection { page_id, scope },
                RepairMutation::RenamePageTitle {
                    before_title,
                    after_title,
                    after_embedding_hex,
                },
                RepairRollbackPayloadV2::RenamePageTitle {
                    page_id: rollback_page_id,
                    ..
                },
            ) if manifest.writer() == RepairWriter::RenamePageTitle
                && page_id == rollback_page_id =>
            {
                (
                    page_id,
                    scope,
                    before_title,
                    after_title,
                    after_embedding_hex,
                )
            }
            _ => {
                return Err(WenlanError::Validation(
                    "page title repair target/writer/mutation mismatch".to_string(),
                ))
            }
        };
    let review_binding = manifest.source().review_binding().ok_or_else(|| {
        WenlanError::Validation("page title repair review binding missing".to_string())
    })?;
    if review_binding.owner_ids().len() != 1
        || review_binding.owner_ids()[0].as_str() != page_id.as_str()
    {
        return Err(WenlanError::Validation(
            "page title repair review binding mismatch".to_string(),
        ));
    }
    let embedding_sql = decode_page_title_embedding(after_embedding_hex)?;
    let session = crate::export::knowledge::KnowledgeProjectionWrite::begin_owned_repair_session(
        page_root.to_path_buf(),
        db,
    )?;
    let projection = session.locked();
    let excluded_paths = crate::repair::rename_page_title_excluded_paths(rollback)?;
    // Before the connection lock, because the permit reads the DB through the
    // same mutex the block below holds. Refusing rather than skipping matches
    // `regenerate_page_projection_cas`: this is a receipted repair, and handing
    // back a receipt for a file that was never rewritten is worse than an error.
    let permit = crate::truth_adapter::page_write_permit(db, page_id)
        .await?
        .ok_or_else(|| {
            WenlanError::Conflict(format!(
                "page {page_id} may not be projected where an automatic reader would find \
                 it, so its title cannot be repaired on disk"
            ))
        })?;
    let conn = db.conn.lock().await;
    conn.execute("BEGIN IMMEDIATE", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair begin: {error}")))?;
    let mut projection_written = false;
    let result = async {
        let mut target_rows = conn
            .query(
                "SELECT title,version,status,space
                   FROM pages WHERE id=?1 LIMIT 2",
                libsql::params![page_id.as_str()],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("repair page title target: {error}")))?;
        let target = target_rows
            .next()
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("repair page title target row: {error}"))
            })?
            .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
        let current_title = target.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("repair page title current title: {error}"))
        })?;
        let current_version = target.get::<i64>(1).map_err(|error| {
            WenlanError::VectorDb(format!("repair page title current version: {error}"))
        })?;
        let current_status = target.get::<String>(2).map_err(|error| {
            WenlanError::VectorDb(format!("repair page title current status: {error}"))
        })?;
        let current_scope = target.get::<Option<String>>(3).map_err(|error| {
            WenlanError::VectorDb(format!("repair page title current scope: {error}"))
        })?;
        if target_rows
            .next()
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("repair page title duplicate target: {error}"))
            })?
            .is_some()
            || current_title != *before_title
            || current_version != manifest.expected_state().version().unwrap_or_default()
            || current_status != "active"
            || current_scope.as_deref() != scope.space()
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        drop(target_rows);
        crate::repair::validate_rename_page_title_collision_on_connection(
            &conn,
            page_id,
            before_title,
            after_title,
            scope.space(),
        )
        .await?;
        let occurrence = crate::repair::validate_rename_page_title_review_on_connection(
            &conn,
            review_binding.review_id(),
            page_id,
        )
        .await?;
        if &occurrence != review_binding.occurrence_digest() {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let before =
            crate::repair::capture_rename_page_title_on_connection(&conn, &projection, page_id)
                .await?;
        if &before != rollback {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let before_target_receipt = crate::repair::rename_page_title_receipt(&before)?;
        if &before_target_receipt != manifest.expected_state().canonical_receipt() {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let database_changes_before_target = conn.total_changes();
        let stored_database_guard = crate::repair::effect_guard_receipt(0);
        let before_scan = projection.scan_page_root_controlled(
            true,
            &crate::lint::pages::fs::PageScanControl::with_timeout(std::time::Duration::from_secs(
                30,
            )),
        )?;
        let non_target_before = crate::repair::rename_page_title_non_target_receipt(
            &stored_database_guard,
            before_scan.non_target_digest(&excluded_paths),
            &before,
        )?;

        let affected = conn
            .execute(
                "UPDATE pages
                    SET title=?1,version=version+1,embedding=vector32(?2)
                  WHERE id=?3 AND status='active' AND version=?4 AND title=?5
                    AND space=COALESCE(?6,'00000000-0000-4000-8000-000000000001')
                    AND NOT EXISTS(
                        SELECT 1 FROM pages collision
                         WHERE collision.status='active' AND collision.id<>?3
                           AND collision.space=COALESCE(?6,'00000000-0000-4000-8000-000000000001')
                           AND lower(collision.title)=lower(?1))",
                libsql::params![
                    after_title.as_str(),
                    embedding_sql,
                    page_id.as_str(),
                    current_version,
                    before_title.as_str(),
                    scope.space().map(str::to_string)
                ],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("repair page title update: {error}")))?;
        if affected != 1 {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        // The canonical Page UPDATE fires the Page FTS maintenance triggers.
        // Treat that statement's complete observed delta as the target write,
        // then require every later proof/projection step to remain DB-read-only.
        let database_changes_after_target = conn.total_changes();
        let target_change_delta = database_changes_after_target
            .checked_sub(database_changes_before_target)
            .ok_or_else(|| WenlanError::VectorDb("repair_effect_counter_underflow".to_string()))?;
        if target_change_delta < affected {
            return Err(WenlanError::VectorDb(
                "repair_target_write_unproven".to_string(),
            ));
        }
        let page = page_on_connection(&conn, page_id).await?;
        if page.title != *after_title || page.version != current_version + 1 {
            return Err(WenlanError::VectorDb(
                "repair_target_write_unproven".to_string(),
            ));
        }
        #[cfg(test)]
        run_test_checkpoint(TestCheckpointKind::BeforeProjectionWrite)?;
        projection_written = true;
        #[cfg(test)]
        if RENAME_PAGE_TITLE_TEST_CHECKPOINTS.try_with(|_| ()).is_ok() {
            projection.write_page_with_after_target_write_permitted(&permit, &page, || {
                run_test_checkpoint(TestCheckpointKind::AfterTargetWrite)
            })?;
        } else {
            projection.write_page_permitted(&permit, &page)?;
        }
        #[cfg(not(test))]
        projection.write_page_permitted(&permit, &page)?;

        let after =
            crate::repair::capture_rename_page_title_on_connection(&conn, &projection, page_id)
                .await?;
        let after_target_receipt = crate::repair::rename_page_title_receipt(&after)?;
        if after_target_receipt == before_target_receipt {
            return Err(WenlanError::VectorDb(
                "repair_target_write_unproven".to_string(),
            ));
        }
        if conn.total_changes() != database_changes_after_target {
            return Err(WenlanError::Conflict("repair_effect_escape".to_string()));
        }
        let after_scan = projection.scan_page_root_controlled(
            true,
            &crate::lint::pages::fs::PageScanControl::with_timeout(std::time::Duration::from_secs(
                30,
            )),
        )?;
        let non_target_after = crate::repair::rename_page_title_non_target_receipt(
            &stored_database_guard,
            after_scan.non_target_digest(&excluded_paths),
            &after,
        )?;
        if non_target_after != non_target_before {
            return Err(WenlanError::Conflict("repair_effect_escape".to_string()));
        }
        let post_apply_db_digest = crate::repair::database_content_digest(&conn).await?;
        Ok(RepairWriteProof::from_parts(
            before_target_receipt,
            after_target_receipt,
            non_target_before,
            non_target_after,
            post_apply_db_digest,
        ))
    }
    .await;

    let proof = match result {
        Ok(proof) => proof,
        Err(error) => {
            let projection_error = if projection_written {
                restore_rename_projection(&projection, rollback).err()
            } else {
                None
            };
            let rollback_error = conn.execute("ROLLBACK", ()).await.err();
            return match (projection_error, rollback_error) {
                (None, None) => Err(error),
                _ => Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                )),
            };
        }
    };
    if let Err(error) = before_commit(&proof) {
        let projection_error = restore_rename_projection(&projection, rollback).err();
        let rollback_error = conn.execute("ROLLBACK", ()).await.err();
        return match (projection_error, rollback_error) {
            (None, None) => Err(error),
            _ => Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            )),
        };
    }
    if let Err(error) = conn.execute("COMMIT", ()).await {
        let projection_error = restore_rename_projection(&projection, rollback).err();
        let rollback_error = conn.execute("ROLLBACK", ()).await.err();
        return if projection_error.is_none() && rollback_error.is_none() {
            Err(WenlanError::VectorDb(format!(
                "repair commit failed: {error}"
            )))
        } else {
            Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ))
        };
    }
    Ok(proof)
}

pub(crate) async fn recover_rename_page_title_apply_receipt(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    page_root: &Path,
) -> Result<Option<RepairApplyReceipt>, WenlanError> {
    let (parsed, rollback) = match store.inspect_rename_page_title_recovery_artifact(manifest)? {
        RenamePageTitleRecoveryArtifact::Completed(receipt) => return Ok(Some(receipt)),
        RenamePageTitleRecoveryArtifact::Absent => return Ok(None),
        RenamePageTitleRecoveryArtifact::Pending {
            verified_receipt,
            rollback,
        } => (verified_receipt, rollback),
    };
    let page_id = match manifest.target() {
        RepairTarget::PageProjection { page_id, .. } => page_id,
        _ => {
            return Err(WenlanError::Validation(
                "page title repair target/writer mismatch".to_string(),
            ))
        }
    };
    let session = crate::export::knowledge::KnowledgeProjectionWrite::begin_owned_repair_session(
        page_root.to_path_buf(),
        db,
    )?;
    let projection = session.locked();
    let connection = db.conn.lock().await;
    connection
        .execute("BEGIN IMMEDIATE", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair recovery begin: {error}")))?;
    let current = match crate::repair::capture_rename_page_title_on_connection(
        &connection,
        &projection,
        page_id,
    )
    .await
    {
        Ok(current) => current,
        Err(_) => {
            let _ = connection.execute("ROLLBACK", ()).await;
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
    };
    #[allow(clippy::large_enum_variant)]
    enum Recovery {
        Publish(RepairApplyReceipt),
        Retry,
        RestoreRetry,
    }
    let recovery = if let Some(receipt) = parsed {
        if crate::repair::rename_page_title_receipt(&current)? == *receipt.after_target_receipt() {
            Recovery::Publish(receipt)
        } else if rename_page_database_matches(&current, &rollback)
            && rename_page_projection_matches_pre(&current, &rollback)
        {
            Recovery::Retry
        } else if rename_page_database_matches(&current, &rollback)
            && rename_page_projection_matches_post(&connection, manifest, &rollback, &current)
                .await?
        {
            Recovery::RestoreRetry
        } else {
            return recovery_required_with_rollback(&connection).await;
        }
    } else if rename_page_database_matches(&current, &rollback)
        && rename_page_projection_matches_pre(&current, &rollback)
    {
        Recovery::Retry
    } else if rename_page_database_matches(&current, &rollback)
        && rename_page_projection_matches_post(&connection, manifest, &rollback, &current).await?
    {
        Recovery::RestoreRetry
    } else {
        return recovery_required_with_rollback(&connection).await;
    };
    if matches!(recovery, Recovery::RestoreRetry) {
        let RepairRollbackPayloadV2::RenamePageTitle {
            projection_target_path,
            projection_entries,
            ..
        } = &rollback
        else {
            unreachable!("typed loader returned title rollback")
        };
        if restore_rename_projection_entries(
            &projection,
            projection_target_path,
            projection_entries,
        )
        .is_err()
        {
            let _ = connection.execute("ROLLBACK", ()).await;
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
    }
    connection
        .execute("COMMIT", ())
        .await
        .map_err(|_| WenlanError::Conflict("repair_apply_recovery_required".to_string()))?;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterCommitBeforeArtifact)?;
    match recovery {
        Recovery::Publish(receipt) => {
            store.publish_rename_page_title_pending_apply_receipt(manifest.manifest_id())?;
            Ok(Some(receipt))
        }
        Recovery::Retry | Recovery::RestoreRetry => {
            store.clear_rename_page_title_pending_apply_receipt(manifest.manifest_id())?;
            Ok(None)
        }
    }
}

fn restore_rename_projection(
    projection: &crate::export::knowledge::LockedRepairProjection<'_>,
    rollback: &RepairRollbackPayloadV2,
) -> Result<(), WenlanError> {
    let RepairRollbackPayloadV2::RenamePageTitle {
        projection_target_path,
        projection_entries,
        ..
    } = rollback
    else {
        return Err(WenlanError::Validation(
            "repair_rollback_writer_mismatch".to_string(),
        ));
    };
    restore_rename_projection_entries(projection, projection_target_path, projection_entries)
}

fn restore_rename_projection_entries(
    projection: &crate::export::knowledge::LockedRepairProjection<'_>,
    projection_target_path: &str,
    projection_entries: &[wenlan_types::repair::RepairRollbackFileEntry],
) -> Result<(), WenlanError> {
    #[cfg(test)]
    if RENAME_PAGE_TITLE_TEST_CHECKPOINTS.try_with(|_| ()).is_ok() {
        projection.restore_rename_page_projection_with_checkpoints(
            projection_target_path,
            projection_entries,
            || run_test_checkpoint(TestCheckpointKind::BeforeProjectionRestore),
            || run_test_checkpoint(TestCheckpointKind::AfterTargetRestore),
        )?;
        return Ok(());
    }
    projection.restore_rename_page_projection(projection_target_path, projection_entries)
}

async fn recovery_required_with_rollback<T>(
    connection: &libsql::Connection,
) -> Result<T, WenlanError> {
    let _ = connection.execute("ROLLBACK", ()).await;
    Err(WenlanError::Conflict(
        "repair_apply_recovery_required".to_string(),
    ))
}

fn rename_page_database_matches(
    current: &RepairRollbackPayloadV2,
    expected: &RepairRollbackPayloadV2,
) -> bool {
    matches!(
        (current, expected),
        (
            RepairRollbackPayloadV2::RenamePageTitle {
                page_id,
                page_columns,
                before_page_row,
                ..
            },
            RepairRollbackPayloadV2::RenamePageTitle {
                page_id: expected_page_id,
                page_columns: expected_columns,
                before_page_row: expected_row,
                ..
            }
        ) if page_id == expected_page_id
            && page_columns == expected_columns
            && before_page_row == expected_row
    )
}

fn rename_page_projection_matches_pre(
    current: &RepairRollbackPayloadV2,
    expected: &RepairRollbackPayloadV2,
) -> bool {
    matches!(
        (current, expected),
        (
            RepairRollbackPayloadV2::RenamePageTitle {
                projection_target_path,
                projection_entries,
                ..
            },
            RepairRollbackPayloadV2::RenamePageTitle {
                projection_target_path: expected_path,
                projection_entries: expected_entries,
                ..
            }
        ) if projection_target_path == expected_path
            && projection_entries == expected_entries
    )
}

async fn rename_page_projection_matches_post(
    connection: &libsql::Connection,
    manifest: &RepairManifest,
    rollback: &RepairRollbackPayloadV2,
    current: &RepairRollbackPayloadV2,
) -> Result<bool, WenlanError> {
    let (
        RepairRollbackPayloadV2::RenamePageTitle {
            page_id,
            projection_target_path,
            projection_entries: before_entries,
            ..
        },
        RepairRollbackPayloadV2::RenamePageTitle {
            projection_target_path: current_target_path,
            projection_entries: current_entries,
            ..
        },
        RepairMutation::RenamePageTitle { after_title, .. },
    ) = (rollback, current, manifest.mutation())
    else {
        return Ok(false);
    };
    if projection_target_path != current_target_path
        || before_entries.len() != 2
        || current_entries.len() != 2
    {
        return Ok(false);
    }
    let mut after_page = page_on_connection(connection, page_id).await?;
    after_page.title = after_title.clone();
    after_page.version = after_page.version.saturating_add(1);
    let expected_markdown = crate::export::knowledge::render_markdown_for(&after_page);
    let current_target = current_entries
        .iter()
        .find(|entry| entry.relative_path() == projection_target_path)
        .and_then(|entry| hex::decode(entry.content_hex()).ok());
    if current_target.as_deref() != Some(expected_markdown.as_bytes()) {
        return Ok(false);
    }
    let before_state = before_entries
        .iter()
        .find(|entry| entry.relative_path() == ".wenlan/state.json")
        .and_then(|entry| hex::decode(entry.content_hex()).ok())
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let current_state = current_entries
        .iter()
        .find(|entry| entry.relative_path() == ".wenlan/state.json")
        .and_then(|entry| hex::decode(entry.content_hex()).ok())
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let (Some(mut before_state), Some(mut current_state)) = (before_state, current_state) else {
        return Ok(false);
    };
    let before_page = before_state
        .get_mut("pages")
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|pages| pages.remove(page_id));
    let current_page = current_state
        .get_mut("pages")
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|pages| pages.remove(page_id));
    let (Some(before_page), Some(current_page)) = (before_page, current_page) else {
        return Ok(false);
    };
    let expected_current_page = serde_json::json!({
        "file": projection_target_path,
        "version": after_page.version,
        "last_written": after_page.last_modified,
    });
    Ok(before_state == current_state
        && before_page.get("file")
            == Some(&serde_json::Value::String(projection_target_path.clone()))
        && current_page == expected_current_page)
}

fn decode_page_title_embedding(value: &str) -> Result<String, WenlanError> {
    const EMBEDDING_HEX_LEN: usize = 768 * std::mem::size_of::<f32>() * 2;
    if value.len() != EMBEDDING_HEX_LEN
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err(WenlanError::Validation(
            "repair_page_embedding_invalid".to_string(),
        ));
    }
    let bytes = hex::decode(value)
        .map_err(|_| WenlanError::Validation("repair_page_embedding_invalid".to_string()))?;
    let mut encoded = String::with_capacity(768 * 12);
    encoded.push('[');
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let float = f32::from_le_bytes(chunk.try_into().expect("four-byte chunks"));
        if !float.is_finite() {
            return Err(WenlanError::Validation(
                "repair_page_embedding_invalid".to_string(),
            ));
        }
        if index > 0 {
            encoded.push(',');
        }
        write!(&mut encoded, "{float}")
            .map_err(|error| WenlanError::Validation(error.to_string()))?;
    }
    encoded.push(']');
    Ok(encoded)
}

async fn page_on_connection(
    connection: &libsql::Connection,
    page_id: &str,
) -> Result<crate::pages::Page, WenlanError> {
    let mut rows = connection
        .query(
            "SELECT id, title, summary, content, entity_id, space, source_memory_ids, version,
                    status, created_at, last_compiled, last_modified,
                    COALESCE(sources_updated_count, 0), stale_reason, COALESCE(user_edited, 0),
                    COALESCE(changelog, '[]'), COALESCE(creation_kind, 'distilled'),
                    COALESCE(review_status, 'confirmed'), workspace, citations
               FROM pages WHERE id=?1 LIMIT 2",
            libsql::params![page_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair page title read: {error}")))?;
    let row = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair page title row: {error}")))?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let page = MemoryDB::row_to_page(&row)?;
    if rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair page title rows: {error}")))?
        .is_some()
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(page)
}
