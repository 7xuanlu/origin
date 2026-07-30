// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{error::WenlanError, post_write::RepairWriteProof, repair::StoredRollbackArtifact};
use std::path::Path;
use wenlan_types::repair::{RepairManifest, RepairMutation, RepairTarget, RepairWriter};

#[cfg(test)]
pub(crate) type RegeneratePageTestCheckpoint =
    Box<dyn FnOnce() -> Result<(), WenlanError> + 'static>;

#[cfg(test)]
#[derive(Default)]
pub(crate) struct RegeneratePageTestCheckpoints {
    pub(crate) after_db_snapshot_before_projection_lock: Option<RegeneratePageTestCheckpoint>,
    pub(crate) after_target_write: Option<RegeneratePageTestCheckpoint>,
    pub(crate) before_projection_restore: Option<RegeneratePageTestCheckpoint>,
    pub(crate) after_projection_restore: Option<RegeneratePageTestCheckpoint>,
}

#[cfg(test)]
tokio::task_local! {
    static REGENERATE_PAGE_TEST_CHECKPOINTS:
        std::cell::RefCell<RegeneratePageTestCheckpoints>;
}

#[cfg(test)]
pub(crate) async fn with_regenerate_page_test_checkpoints<T>(
    checkpoints: RegeneratePageTestCheckpoints,
    future: impl std::future::Future<Output = T>,
) -> T {
    REGENERATE_PAGE_TEST_CHECKPOINTS
        .scope(std::cell::RefCell::new(checkpoints), async move {
            let output = future.await;
            let remaining = REGENERATE_PAGE_TEST_CHECKPOINTS.with(|checkpoints| {
                let checkpoints = checkpoints.borrow();
                [
                    (
                        "after_db_snapshot_before_projection_lock",
                        checkpoints
                            .after_db_snapshot_before_projection_lock
                            .is_some(),
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
                        "after_projection_restore",
                        checkpoints.after_projection_restore.is_some(),
                    ),
                ]
                .into_iter()
                .filter_map(|(name, present)| present.then_some(name))
                .collect::<Vec<_>>()
            });
            assert!(
                remaining.is_empty(),
                "unconsumed regenerate page test checkpoints: {}",
                remaining.join(", ")
            );
            output
        })
        .await
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum TestCheckpointKind {
    AfterDbSnapshotBeforeProjectionLock,
    AfterTargetWrite,
    BeforeProjectionRestore,
    AfterProjectionRestore,
}

#[cfg(test)]
fn run_test_checkpoint(kind: TestCheckpointKind) -> Result<(), WenlanError> {
    REGENERATE_PAGE_TEST_CHECKPOINTS
        .try_with(|checkpoints| {
            let hook = {
                let mut checkpoints = checkpoints.borrow_mut();
                match kind {
                    TestCheckpointKind::AfterDbSnapshotBeforeProjectionLock => {
                        &mut checkpoints.after_db_snapshot_before_projection_lock
                    }
                    TestCheckpointKind::AfterTargetWrite => &mut checkpoints.after_target_write,
                    TestCheckpointKind::BeforeProjectionRestore => {
                        &mut checkpoints.before_projection_restore
                    }
                    TestCheckpointKind::AfterProjectionRestore => {
                        &mut checkpoints.after_projection_restore
                    }
                }
                .take()
            };
            hook.map_or(Ok(()), |hook| hook())
        })
        .unwrap_or(Ok(()))
}

#[cfg(test)]
fn restore_projection_snapshot(
    page_root: &Path,
    before: &StoredRollbackArtifact,
) -> Result<(), WenlanError> {
    run_test_checkpoint(TestCheckpointKind::BeforeProjectionRestore)?;
    crate::repair::restore_page_projection_snapshot(page_root, before)?;
    run_test_checkpoint(TestCheckpointKind::AfterProjectionRestore)
}

#[cfg(not(test))]
fn restore_projection_snapshot(
    page_root: &Path,
    before: &StoredRollbackArtifact,
) -> Result<(), WenlanError> {
    crate::repair::restore_page_projection_snapshot(page_root, before)
}

pub(crate) async fn regenerate_page_projection_cas<F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: &Path,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    let (page_id, expected_scope, database_version) =
        match (manifest.target(), manifest.writer(), manifest.mutation()) {
            (
                RepairTarget::PageProjection { page_id, scope },
                RepairWriter::RegeneratePageProjection,
                RepairMutation::RegeneratePageProjection { database_version },
            ) => (page_id.as_str(), scope.space(), *database_version),
            _ => {
                return Err(WenlanError::Validation(
                    "page projection repair target/writer/mutation mismatch".to_string(),
                ))
            }
        };
    let page = db
        .get_page(page_id)
        .await?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    if page.version != database_version
        || page.workspace.as_deref().or(page.space.as_deref()) != expected_scope
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let paths = crate::repair::projection_rollback_paths(rollback)?;
    let target_path = crate::repair::page_projection_target_path(rollback)?;
    let conn = db.conn.lock().await;
    let page_row = crate::repair::projection_page_row_on_connection(&conn, page_id).await?;
    let post_apply_db_digest = crate::repair::database_content_digest(&conn).await?;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterDbSnapshotBeforeProjectionLock)?;
    crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
        page_root.to_path_buf(),
        db,
        |write| {
            let before = crate::repair::capture_page_projection_from_row(
                page_root,
                page_id,
                page_row.clone(),
                &paths,
                &rollback.table,
            )?;
            let before_target_receipt = crate::repair::target_receipt(&before)?;
            if &before_target_receipt != manifest.expected_state().canonical_receipt() {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let scan_control = crate::lint::pages::fs::PageScanControl::with_timeout(
                std::time::Duration::from_secs(30),
            );
            let before_scan =
                crate::lint::pages::fs::scan_page_root_controlled(page_root, true, &scan_control)
                    .map_err(|error| {
                    WenlanError::Validation(format!("repair projection scan: {error}"))
                })?;
            if !crate::lint::pages::state_checks::projection_target_is_exclusive_page_markdown(
                &before_scan,
                page_id,
                &target_path,
            ) {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let non_target_before = crate::repair::page_projection_non_target_receipt(
                before_scan.non_target_digest(&paths),
                &before,
            )?;

            let result = (|| {
                #[cfg(test)]
                if REGENERATE_PAGE_TEST_CHECKPOINTS.try_with(|_| ()).is_ok() {
                    write.write_page_with_after_target_write(&page, || {
                        run_test_checkpoint(TestCheckpointKind::AfterTargetWrite)
                    })?;
                } else {
                    write.write_page(&page)?;
                }
                #[cfg(not(test))]
                write.write_page(&page)?;
                let after_scan = crate::lint::pages::fs::scan_page_root_controlled(
                    page_root,
                    true,
                    &crate::lint::pages::fs::PageScanControl::with_timeout(
                        std::time::Duration::from_secs(30),
                    ),
                )
                .map_err(|error| {
                    WenlanError::Validation(format!("repair projection scan: {error}"))
                })?;
                let after = crate::repair::capture_page_projection_from_row(
                    page_root,
                    page_id,
                    page_row.clone(),
                    &paths,
                    &rollback.table,
                )?;
                let non_target_after = crate::repair::page_projection_non_target_receipt(
                    after_scan.non_target_digest(&paths),
                    &after,
                )?;
                if non_target_after != non_target_before {
                    return Err(WenlanError::Conflict("repair_effect_escape".to_string()));
                }
                Ok((non_target_after, after))
            })();
            let (non_target_after, after) = match result {
                Ok(result) => result,
                Err(error) => {
                    return match restore_projection_snapshot(page_root, &before) {
                        Ok(()) => Err(error),
                        Err(rollback_error) => Err(WenlanError::VectorDb(format!(
                            "{error}; repair projection rollback failed: {rollback_error}"
                        ))),
                    };
                }
            };
            let completion = (|| {
                let after_target_receipt = crate::repair::target_receipt(&after)?;
                if after_target_receipt == before_target_receipt {
                    return Err(WenlanError::VectorDb(
                        "repair_target_write_unproven".to_string(),
                    ));
                }
                let proof = RepairWriteProof::from_parts(
                    before_target_receipt,
                    after_target_receipt,
                    non_target_before,
                    non_target_after,
                    post_apply_db_digest.clone(),
                );
                before_commit(&proof)?;
                Ok(proof)
            })();
            match completion {
                Ok(proof) => Ok(proof),
                Err(error) => match restore_projection_snapshot(page_root, &before) {
                    Ok(()) => Err(error),
                    Err(rollback_error) => Err(WenlanError::VectorDb(format!(
                        "{error}; repair projection rollback failed: {rollback_error}"
                    ))),
                },
            }
        },
    )
}
