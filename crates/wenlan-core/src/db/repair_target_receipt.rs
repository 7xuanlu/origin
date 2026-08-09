// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{
    error::WenlanError,
    repair::{
        capture_page_projection_on_connection, capture_stale_page_projection_current,
        database_error, projection_rollback_paths, stale_page_projection_paths, target_receipt,
        StoredRollbackArtifact,
    },
};
use std::path::Path;
use wenlan_types::repair::{RepairDigest, RepairManifest, RepairTarget, RepairWriter};

#[cfg(test)]
pub(crate) type RepairTargetReceiptTestCheckpoint =
    Box<dyn FnOnce() -> Result<(), WenlanError> + 'static>;

#[cfg(test)]
#[derive(Default)]
pub(crate) struct RepairTargetReceiptTestCheckpoints {
    pub(crate) after_projection_db_lock_before_validation:
        Option<RepairTargetReceiptTestCheckpoint>,
    pub(crate) before_capture: Option<RepairTargetReceiptTestCheckpoint>,
    pub(crate) after_capture_before_return: Option<RepairTargetReceiptTestCheckpoint>,
}

#[cfg(test)]
tokio::task_local! {
    static REPAIR_TARGET_RECEIPT_TEST_CHECKPOINTS:
        std::cell::RefCell<RepairTargetReceiptTestCheckpoints>;
}

#[cfg(test)]
pub(crate) async fn with_repair_target_receipt_test_checkpoints<T>(
    checkpoints: RepairTargetReceiptTestCheckpoints,
    future: impl std::future::Future<Output = T>,
) -> T {
    REPAIR_TARGET_RECEIPT_TEST_CHECKPOINTS
        .scope(std::cell::RefCell::new(checkpoints), async move {
            let output = future.await;
            let remaining = REPAIR_TARGET_RECEIPT_TEST_CHECKPOINTS.with(|checkpoints| {
                let checkpoints = checkpoints.borrow();
                [
                    (
                        "after_projection_db_lock_before_validation",
                        checkpoints
                            .after_projection_db_lock_before_validation
                            .is_some(),
                    ),
                    ("before_capture", checkpoints.before_capture.is_some()),
                    (
                        "after_capture_before_return",
                        checkpoints.after_capture_before_return.is_some(),
                    ),
                ]
                .into_iter()
                .filter_map(|(name, present)| present.then_some(name))
                .collect::<Vec<_>>()
            });
            assert!(
                remaining.is_empty(),
                "unconsumed repair target receipt test checkpoints: {}",
                remaining.join(", ")
            );
            output
        })
        .await
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum TestCheckpointKind {
    AfterProjectionDbLockBeforeValidation,
    BeforeCapture,
    AfterCaptureBeforeReturn,
}

#[cfg(test)]
fn run_test_checkpoint(kind: TestCheckpointKind) -> Result<(), WenlanError> {
    REPAIR_TARGET_RECEIPT_TEST_CHECKPOINTS
        .try_with(|checkpoints| {
            let hook = {
                let mut checkpoints = checkpoints.borrow_mut();
                match kind {
                    TestCheckpointKind::AfterProjectionDbLockBeforeValidation => {
                        &mut checkpoints.after_projection_db_lock_before_validation
                    }
                    TestCheckpointKind::BeforeCapture => &mut checkpoints.before_capture,
                    TestCheckpointKind::AfterCaptureBeforeReturn => {
                        &mut checkpoints.after_capture_before_return
                    }
                }
                .take()
            };
            hook.map_or(Ok(()), |hook| hook())
        })
        .unwrap_or(Ok(()))
}

pub(crate) async fn read_current_repair_target_receipt(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
) -> Result<(RepairDigest, u64), WenlanError> {
    match manifest.target() {
        RepairTarget::PageProjection { page_id, .. }
            if manifest.writer() == RepairWriter::QuarantineStalePageProjection =>
        {
            read_stale_projection_target_receipt(db, page_id, rollback, page_root).await
        }
        RepairTarget::PageProjection { page_id, .. } => {
            read_projection_target_receipt(db, page_id, rollback, page_root).await
        }
        _ => db.read_repair_target_receipt(manifest.target()).await,
    }
}

async fn read_projection_target_receipt(
    db: &MemoryDB,
    page_id: &str,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
) -> Result<(RepairDigest, u64), WenlanError> {
    let connection = db.conn.lock().await;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterProjectionDbLockBeforeValidation)?;
    let page_root = page_root.ok_or_else(|| {
        WenlanError::Validation("page projection repair root unavailable".to_string())
    })?;
    let paths = projection_rollback_paths(rollback)?;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::BeforeCapture)?;
    let current = capture_page_projection_on_connection(
        &connection,
        page_root,
        page_id,
        &paths,
        &rollback.table,
    )
    .await?;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterCaptureBeforeReturn)?;
    Ok((target_receipt(&current)?, 1))
}

async fn read_stale_projection_target_receipt(
    db: &MemoryDB,
    page_id: &str,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
) -> Result<(RepairDigest, u64), WenlanError> {
    let connection = db.conn.lock().await;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterProjectionDbLockBeforeValidation)?;
    let page_root = page_root.ok_or_else(|| {
        WenlanError::Validation("page projection repair root unavailable".to_string())
    })?;
    let mut owner = connection
        .query(
            "SELECT 1 FROM pages WHERE id=?1 LIMIT 1",
            libsql::params![page_id],
        )
        .await
        .map_err(database_error)?;
    if owner.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    drop(owner);
    let (source_path, quarantine_path) = stale_page_projection_paths(rollback)?;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::BeforeCapture)?;
    let current =
        capture_stale_page_projection_current(page_root, page_id, &source_path, &quarantine_path)?;
    #[cfg(test)]
    run_test_checkpoint(TestCheckpointKind::AfterCaptureBeforeReturn)?;
    Ok((target_receipt(&current)?, 0))
}
