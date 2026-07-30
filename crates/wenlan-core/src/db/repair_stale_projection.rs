// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{
    error::WenlanError,
    post_write::RepairWriteProof,
    repair::{
        RepairArtifactStore, StalePageProjectionRecoveryArtifactUpdate,
        StalePageProjectionRecoveryJournal, StoredRollbackArtifact,
    },
};
use std::path::Path;
use wenlan_types::repair::{
    RepairApplyReceipt, RepairManifest, RepairMutation, RepairScope, RepairTarget, RepairWriter,
};

#[cfg(test)]
pub(crate) type StaleProjectionTestCheckpoint =
    Box<dyn FnOnce() -> Result<(), WenlanError> + 'static>;

#[cfg(test)]
#[derive(Default)]
pub(crate) struct StaleProjectionTestCheckpoints {
    pub(crate) after_db_snapshot_before_projection_lock: Option<StaleProjectionTestCheckpoint>,
    pub(crate) before_journal_persist: Option<StaleProjectionTestCheckpoint>,
    pub(crate) after_journal_persist_before_pin: Option<StaleProjectionTestCheckpoint>,
    pub(crate) after_pin_before_link: Option<StaleProjectionTestCheckpoint>,
    pub(crate) before_source_stage: Option<StaleProjectionTestCheckpoint>,
    pub(crate) before_snapshot_restore: Option<StaleProjectionTestCheckpoint>,
    pub(crate) after_snapshot_restore: Option<StaleProjectionTestCheckpoint>,
    pub(crate) after_recovery_state_before_artifact: Option<StaleProjectionTestCheckpoint>,
    pub(crate) after_recovery_artifact: Option<StaleProjectionTestCheckpoint>,
}

#[cfg(test)]
tokio::task_local! {
    static STALE_PROJECTION_TEST_CHECKPOINTS:
        std::cell::RefCell<StaleProjectionTestCheckpoints>;
}

#[cfg(test)]
pub(crate) async fn with_stale_projection_test_checkpoints<T>(
    checkpoints: StaleProjectionTestCheckpoints,
    future: impl std::future::Future<Output = T>,
) -> T {
    STALE_PROJECTION_TEST_CHECKPOINTS
        .scope(std::cell::RefCell::new(checkpoints), async move {
            let output = future.await;
            let remaining = STALE_PROJECTION_TEST_CHECKPOINTS.with(|checkpoints| {
                let checkpoints = checkpoints.borrow();
                [
                    (
                        "after_db_snapshot_before_projection_lock",
                        checkpoints
                            .after_db_snapshot_before_projection_lock
                            .is_some(),
                    ),
                    (
                        "before_journal_persist",
                        checkpoints.before_journal_persist.is_some(),
                    ),
                    (
                        "after_journal_persist_before_pin",
                        checkpoints.after_journal_persist_before_pin.is_some(),
                    ),
                    (
                        "after_pin_before_link",
                        checkpoints.after_pin_before_link.is_some(),
                    ),
                    (
                        "before_source_stage",
                        checkpoints.before_source_stage.is_some(),
                    ),
                    (
                        "before_snapshot_restore",
                        checkpoints.before_snapshot_restore.is_some(),
                    ),
                    (
                        "after_snapshot_restore",
                        checkpoints.after_snapshot_restore.is_some(),
                    ),
                    (
                        "after_recovery_state_before_artifact",
                        checkpoints.after_recovery_state_before_artifact.is_some(),
                    ),
                    (
                        "after_recovery_artifact",
                        checkpoints.after_recovery_artifact.is_some(),
                    ),
                ]
                .into_iter()
                .filter_map(|(name, present)| present.then_some(name))
                .collect::<Vec<_>>()
            });
            assert!(
                remaining.is_empty(),
                "unconsumed stale projection test checkpoints: {}",
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
    BeforeJournalPersist,
    AfterJournalPersistBeforePin,
    AfterPinBeforeLink,
    BeforeSourceStage,
    BeforeSnapshotRestore,
    AfterSnapshotRestore,
    AfterRecoveryStateBeforeArtifact,
    AfterRecoveryArtifact,
}

#[cfg(test)]
fn run_test_checkpoint(kind: TestCheckpointKind) -> Result<(), WenlanError> {
    STALE_PROJECTION_TEST_CHECKPOINTS
        .try_with(|checkpoints| {
            let hook = {
                let mut checkpoints = checkpoints.borrow_mut();
                match kind {
                    TestCheckpointKind::AfterDbSnapshotBeforeProjectionLock => {
                        &mut checkpoints.after_db_snapshot_before_projection_lock
                    }
                    TestCheckpointKind::BeforeJournalPersist => {
                        &mut checkpoints.before_journal_persist
                    }
                    TestCheckpointKind::AfterJournalPersistBeforePin => {
                        &mut checkpoints.after_journal_persist_before_pin
                    }
                    TestCheckpointKind::AfterPinBeforeLink => {
                        &mut checkpoints.after_pin_before_link
                    }
                    TestCheckpointKind::BeforeSourceStage => &mut checkpoints.before_source_stage,
                    TestCheckpointKind::BeforeSnapshotRestore => {
                        &mut checkpoints.before_snapshot_restore
                    }
                    TestCheckpointKind::AfterSnapshotRestore => {
                        &mut checkpoints.after_snapshot_restore
                    }
                    TestCheckpointKind::AfterRecoveryStateBeforeArtifact => {
                        &mut checkpoints.after_recovery_state_before_artifact
                    }
                    TestCheckpointKind::AfterRecoveryArtifact => {
                        &mut checkpoints.after_recovery_artifact
                    }
                }
                .take()
            };
            hook.map_or(Ok(()), |hook| hook())
        })
        .unwrap_or(Ok(()))
}

#[cfg(test)]
fn restore_snapshot(
    pinned: &mut crate::export::knowledge::PinnedStalePageProjection<'_>,
    before: &StoredRollbackArtifact,
) -> Result<(), WenlanError> {
    run_test_checkpoint(TestCheckpointKind::BeforeSnapshotRestore)?;
    pinned.restore_snapshot(before)?;
    run_test_checkpoint(TestCheckpointKind::AfterSnapshotRestore)
}

#[cfg(not(test))]
fn restore_snapshot(
    pinned: &mut crate::export::knowledge::PinnedStalePageProjection<'_>,
    before: &StoredRollbackArtifact,
) -> Result<(), WenlanError> {
    pinned.restore_snapshot(before)
}

#[cfg(test)]
pub(crate) async fn quarantine_stale_page_projection_cas<F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: &Path,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    quarantine_stale_page_projection_cas_with_apply_journal(
        db,
        manifest,
        rollback,
        page_root,
        |_| Ok(()),
        before_commit,
    )
    .await
}

#[cfg(test)]
pub(crate) async fn quarantine_stale_page_projection_cas_with_before_pin<B, F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: &Path,
    before_pin: B,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    B: FnOnce() -> Result<(), WenlanError> + 'static,
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            before_journal_persist: Some(Box::new(before_pin)),
            ..Default::default()
        },
        quarantine_stale_page_projection_cas_with_apply_journal(
            db,
            manifest,
            rollback,
            page_root,
            |_| Ok(()),
            before_commit,
        ),
    )
    .await
}

#[cfg(test)]
pub(crate) async fn quarantine_stale_page_projection_cas_with_after_pin<A, F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: &Path,
    after_pin: A,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    A: FnOnce() -> Result<(), WenlanError> + 'static,
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            after_pin_before_link: Some(Box::new(after_pin)),
            ..Default::default()
        },
        quarantine_stale_page_projection_cas_with_apply_journal(
            db,
            manifest,
            rollback,
            page_root,
            |_| Ok(()),
            before_commit,
        ),
    )
    .await
}

#[cfg(test)]
pub(crate) async fn quarantine_stale_page_projection_cas_with_before_source_stage<S, F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: &Path,
    before_source_stage: S,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    S: FnOnce() -> Result<(), WenlanError> + 'static,
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            before_source_stage: Some(Box::new(before_source_stage)),
            ..Default::default()
        },
        quarantine_stale_page_projection_cas_with_apply_journal(
            db,
            manifest,
            rollback,
            page_root,
            |_| Ok(()),
            before_commit,
        ),
    )
    .await
}

pub(crate) async fn quarantine_stale_page_projection_cas_with_apply_journal<J, F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: &Path,
    persist_apply_journal: J,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    J: FnOnce(&StoredRollbackArtifact) -> Result<(), WenlanError>,
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    let (page_id, source_path, quarantine_path) =
        match (manifest.target(), manifest.writer(), manifest.mutation()) {
            (
                RepairTarget::PageProjection {
                    page_id,
                    scope: RepairScope::Global,
                },
                RepairWriter::QuarantineStalePageProjection,
                RepairMutation::QuarantineStalePageProjection {
                    source_path,
                    quarantine_path,
                },
            ) => (
                page_id.as_str(),
                source_path.as_str(),
                quarantine_path.as_str(),
            ),
            _ => {
                return Err(WenlanError::Validation(
                    "stale page projection repair target/writer/mutation mismatch".to_string(),
                ))
            }
        };
    let (captured_source, captured_quarantine) =
        crate::repair::stale_page_projection_paths(rollback)?;
    if captured_source != source_path || captured_quarantine != quarantine_path {
        return Err(WenlanError::Validation(
            "repair_projection_rollback_invalid".to_string(),
        ));
    }
    let connection = db.conn.lock().await;
    let mut owner = connection
        .query(
            "SELECT 1 FROM pages WHERE id=?1 LIMIT 1",
            libsql::params![page_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair projection owner CAS: {error}")))?;
    if owner
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair projection owner CAS: {error}")))?
        .is_some()
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    drop(owner);
    let post_apply_db_digest = crate::repair::database_content_digest(&connection).await?;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterDbSnapshotBeforeProjectionLock)?;
    crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
        page_root.to_path_buf(),
        db,
        |write| {
            let before = write.capture_stale_page_projection_current(
                page_id,
                source_path,
                quarantine_path,
            )?;
            let before_target_receipt = crate::repair::target_receipt(&before)?;
            if &before_target_receipt != manifest.expected_state().canonical_receipt() {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let excluded = std::collections::BTreeSet::from([
                ".wenlan".to_string(),
                ".wenlan/state.json".to_string(),
                ".wenlan/orphaned".to_string(),
                source_path.to_string(),
                quarantine_path.to_string(),
            ]);
            let scan = write.scan_page_root_controlled(
                true,
                &crate::lint::pages::fs::PageScanControl::with_timeout(
                    std::time::Duration::from_secs(30),
                ),
            )?;
            if !matches!(
                crate::lint::pages::state_checks::stale_projection_ownership(&scan, page_id),
                crate::lint::pages::state_checks::StaleProjectionOwnership::Exact {
                    source_path: ref source,
                    quarantine_path: ref quarantine,
                } if source == source_path && quarantine == quarantine_path
            ) {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let non_target_before = crate::repair::page_projection_non_target_receipt(
                scan.non_target_digest(&excluded),
                &before,
            )?;
            #[cfg(test)]
            run_test_checkpoint(TestCheckpointKind::BeforeJournalPersist)?;
            persist_apply_journal(&before)?;
            #[cfg(test)]
            run_test_checkpoint(TestCheckpointKind::AfterJournalPersistBeforePin)?;
            let mut pinned = write.pin_stale_page_projection(
                page_id,
                source_path,
                quarantine_path,
                &before,
                manifest.manifest_id(),
            )?;
            let result = (|| {
                #[cfg(test)]
                if STALE_PROJECTION_TEST_CHECKPOINTS.try_with(|_| ()).is_ok() {
                    pinned.quarantine_with_hooks(
                        || run_test_checkpoint(TestCheckpointKind::AfterPinBeforeLink),
                        || run_test_checkpoint(TestCheckpointKind::BeforeSourceStage),
                    )?;
                } else {
                    pinned.quarantine()?;
                }
                #[cfg(not(test))]
                pinned.quarantine()?;
                let after_scan = write.scan_page_root_controlled(
                    true,
                    &crate::lint::pages::fs::PageScanControl::with_timeout(
                        std::time::Duration::from_secs(30),
                    ),
                )?;
                let after = write.capture_stale_page_projection_current(
                    page_id,
                    source_path,
                    quarantine_path,
                )?;
                let non_target_after = crate::repair::page_projection_non_target_receipt(
                    after_scan.non_target_digest(&excluded),
                    &after,
                )?;
                if non_target_after != non_target_before {
                    return Err(WenlanError::Conflict("repair_effect_escape".to_string()));
                }
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
                    post_apply_db_digest,
                );
                before_commit(&proof)?;
                Ok(proof)
            })();
            match result {
                Ok(proof) => Ok(proof),
                Err(error) if !pinned.mutation_started() => Err(error),
                Err(error) => match restore_snapshot(&mut pinned, &before) {
                    Ok(()) => Err(error),
                    Err(rollback_error) => Err(WenlanError::VectorDb(format!(
                        "{error}; stale projection rollback failed: {rollback_error}"
                    ))),
                },
            }
        },
    )
}

pub(crate) async fn recover_stale_page_projection_apply_receipt(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
    pending_receipt: Option<RepairApplyReceipt>,
) -> Result<Option<RepairApplyReceipt>, WenlanError> {
    let page_root = page_root.ok_or_else(|| {
        WenlanError::Validation("page projection repair root unavailable".to_string())
    })?;
    let page_id = match manifest.target() {
        RepairTarget::PageProjection { page_id, .. } => page_id,
        _ => {
            return Err(WenlanError::Validation(
                "stale page projection repair target/writer mismatch".to_string(),
            ))
        }
    };
    let connection = db.conn.lock().await;
    let mut owner = connection
        .query(
            "SELECT 1 FROM pages WHERE id=?1 LIMIT 1",
            libsql::params![page_id.as_str()],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair database: {error}")))?;
    if owner
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair database: {error}")))?
        .is_some()
    {
        return Err(WenlanError::Conflict(
            "repair_apply_recovery_required".to_string(),
        ));
    }
    drop(owner);
    let journal = store.inspect_stale_page_projection_recovery_journal(manifest)?;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterDbSnapshotBeforeProjectionLock)?;
    if matches!(journal, StalePageProjectionRecoveryJournal::Absent) {
        let (source_path, quarantine_path) = crate::repair::stale_page_projection_paths(rollback)?;
        if crate::repair::capture_stale_page_projection_current(
            page_root,
            page_id,
            &source_path,
            &quarantine_path,
        )
        .is_ok_and(|current| current == *rollback)
        {
            #[cfg(test)]
            run_test_checkpoint(TestCheckpointKind::AfterRecoveryStateBeforeArtifact)?;
            store.apply_stale_page_projection_recovery_artifact_update(
                manifest.manifest_id(),
                StalePageProjectionRecoveryArtifactUpdate::ClearPendingOnly,
            )?;
            #[cfg(test)]
            run_test_checkpoint(TestCheckpointKind::AfterRecoveryArtifact)?;
            return Ok(None);
        }
    }
    let apply_rollback = match journal {
        StalePageProjectionRecoveryJournal::Present(rollback) => rollback,
        StalePageProjectionRecoveryJournal::Absent => rollback.clone(),
    };
    let restore_post = pending_receipt.is_none();
    let recovery = crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
        page_root.to_path_buf(),
        db,
        |write| {
            write.recover_stale_page_projection(
                &apply_rollback,
                manifest.manifest_id(),
                restore_post,
            )
        },
    )
    .unwrap_or(crate::export::knowledge::StalePageProjectionRecoveryState::Unknown);
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterRecoveryStateBeforeArtifact)?;
    use crate::export::knowledge::StalePageProjectionRecoveryState as Recovery;
    let result = match (pending_receipt, recovery) {
        (Some(receipt), Recovery::Post)
            if crate::repair::stale_page_projection_post_target_receipt(&apply_rollback)?
                == *receipt.after_target_receipt() =>
        {
            store.apply_stale_page_projection_recovery_artifact_update(
                manifest.manifest_id(),
                StalePageProjectionRecoveryArtifactUpdate::PublishPendingAndClearJournal,
            )?;
            Ok(Some(receipt))
        }
        (Some(_), Recovery::Original) | (None, Recovery::Original) => {
            store.apply_stale_page_projection_recovery_artifact_update(
                manifest.manifest_id(),
                StalePageProjectionRecoveryArtifactUpdate::ClearPendingAndJournal,
            )?;
            Ok(None)
        }
        _ => Err(WenlanError::Conflict(
            "repair_apply_recovery_required".to_string(),
        )),
    };
    #[cfg(test)]
    if result.is_ok() {
        run_test_checkpoint(TestCheckpointKind::AfterRecoveryArtifact)?;
    }
    result
}
