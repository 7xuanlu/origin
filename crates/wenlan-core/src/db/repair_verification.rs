// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{
    error::WenlanError,
    export::knowledge::{KnowledgeProjectionWrite, OwnedRepairProjectionSession},
    lint::pages::fs::{scan_page_root_controlled, PageScanControl},
    repair::{
        capture_page_projection_from_row, capture_rename_page_title_on_connection,
        effect_guard_receipt, page_projection_non_target_receipt,
        projection_page_row_on_connection, projection_rollback_paths,
        rename_page_title_excluded_paths, rename_page_title_non_target_receipt,
        rename_page_title_receipt, repair_digest, repair_target_receipt_on_connection,
        stale_page_projection_paths, target_receipt, validate_current_db_receipts,
        validate_current_page_receipts_locked, validate_current_page_receipts_on_repair_projection,
        validate_deterministic_target_resolved, validate_tag_record_set_on_connection,
        RepairArtifactStore, StoredRollbackArtifact,
    },
};
use std::{collections::BTreeSet, path::Path, time::Duration};
use wenlan_types::repair::{
    RepairApplyReceipt, RepairManifest, RepairRollbackPayloadV2, RepairTarget,
    RepairVerificationReceipt, RepairVerificationReceiptDraft, RepairWriter, VerifyRepairRequest,
};

#[cfg(test)]
pub(crate) type RepairVerificationTestCheckpoint = Box<dyn FnOnce() + 'static>;

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum RepairVerificationTestCommitFailure {
    AfterPersistBeforeCommit,
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct RepairVerificationTestControl {
    pub(crate) after_rename_session_acquired: Option<RepairVerificationTestCheckpoint>,
    pub(crate) after_begin: Option<RepairVerificationTestCheckpoint>,
    pub(crate) before_receipt_persist: Option<RepairVerificationTestCheckpoint>,
    pub(crate) after_receipt_persist: Option<RepairVerificationTestCheckpoint>,
    pub(crate) after_commit_before_pending_clear: Option<RepairVerificationTestCheckpoint>,
    pub(crate) commit_failure_after_persist: Option<RepairVerificationTestCommitFailure>,
}

#[cfg(test)]
tokio::task_local! {
    static REPAIR_VERIFICATION_TEST_CONTROL:
        std::cell::RefCell<RepairVerificationTestControl>;
}

#[cfg(test)]
pub(crate) async fn with_repair_verification_test_control<T>(
    control: RepairVerificationTestControl,
    future: impl std::future::Future<Output = T>,
) -> T {
    REPAIR_VERIFICATION_TEST_CONTROL
        .scope(std::cell::RefCell::new(control), async move {
            let output = future.await;
            let remaining = REPAIR_VERIFICATION_TEST_CONTROL.with(|control| {
                let control = control.borrow();
                [
                    (
                        "after_rename_session_acquired",
                        control.after_rename_session_acquired.is_some(),
                    ),
                    ("after_begin", control.after_begin.is_some()),
                    (
                        "before_receipt_persist",
                        control.before_receipt_persist.is_some(),
                    ),
                    (
                        "after_receipt_persist",
                        control.after_receipt_persist.is_some(),
                    ),
                    (
                        "after_commit_before_pending_clear",
                        control.after_commit_before_pending_clear.is_some(),
                    ),
                    (
                        "commit_failure_after_persist",
                        control.commit_failure_after_persist.is_some(),
                    ),
                ]
                .into_iter()
                .filter_map(|(name, present)| present.then_some(name))
                .collect::<Vec<_>>()
            });
            assert!(
                remaining.is_empty(),
                "unconsumed repair verification test control: {}",
                remaining.join(", ")
            );
            output
        })
        .await
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum RepairVerificationTestCheckpointKind {
    AfterRenameSessionAcquired,
    AfterBegin,
    BeforeReceiptPersist,
    AfterReceiptPersist,
    AfterCommitBeforePendingClear,
}

#[cfg(test)]
pub(crate) fn run_repair_verification_test_checkpoint(kind: RepairVerificationTestCheckpointKind) {
    let hook = REPAIR_VERIFICATION_TEST_CONTROL
        .try_with(|control| {
            let mut control = control.borrow_mut();
            match kind {
                RepairVerificationTestCheckpointKind::AfterRenameSessionAcquired => {
                    &mut control.after_rename_session_acquired
                }
                RepairVerificationTestCheckpointKind::AfterBegin => &mut control.after_begin,
                RepairVerificationTestCheckpointKind::BeforeReceiptPersist => {
                    &mut control.before_receipt_persist
                }
                RepairVerificationTestCheckpointKind::AfterReceiptPersist => {
                    &mut control.after_receipt_persist
                }
                RepairVerificationTestCheckpointKind::AfterCommitBeforePendingClear => {
                    &mut control.after_commit_before_pending_clear
                }
            }
            .take()
        })
        .unwrap_or(None);
    if let Some(hook) = hook {
        hook();
    }
}

#[cfg(test)]
fn consume_repair_verification_test_commit_failure() -> bool {
    REPAIR_VERIFICATION_TEST_CONTROL
        .try_with(|control| {
            matches!(
                control.borrow_mut().commit_failure_after_persist.take(),
                Some(RepairVerificationTestCommitFailure::AfterPersistBeforeCommit)
            )
        })
        .unwrap_or(false)
}

pub(crate) struct RepairVerificationAtomicInput<'a> {
    pub(crate) store: &'a RepairArtifactStore,
    pub(crate) manifest: &'a RepairManifest,
    pub(crate) apply_receipt: &'a RepairApplyReceipt,
    pub(crate) request: &'a VerifyRepairRequest,
    pub(crate) prior_verified_tag_targets: &'a [RepairTarget],
    pub(crate) rollback: Option<&'a StoredRollbackArtifact>,
    pub(crate) rename_rollback: Option<&'a RepairRollbackPayloadV2>,
    pub(crate) page_root: Option<&'a Path>,
    pub(crate) verified_at: i64,
    pub(crate) rename_projection_session: Option<&'a OwnedRepairProjectionSession>,
}

pub(crate) async fn record_repair_verification_atomic(
    db: &MemoryDB,
    input: RepairVerificationAtomicInput<'_>,
) -> Result<RepairVerificationReceipt, WenlanError> {
    let RepairVerificationAtomicInput {
        store,
        manifest,
        apply_receipt,
        request,
        prior_verified_tag_targets,
        rollback,
        rename_rollback,
        page_root,
        verified_at,
        rename_projection_session,
    } = input;

    let connection = db.conn.lock().await;
    connection
        .execute("BEGIN IMMEDIATE", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair verify begin: {error}")))?;
    #[cfg(test)]
    run_repair_verification_test_checkpoint(RepairVerificationTestCheckpointKind::AfterBegin);
    let result = async {
        validate_current_db_receipts(db, request.general_report(), request.deep_report()).await?;
        // The durable content-addressed apply receipt records an apply-time
        // effect guard and rejects unequal non_target_before/non_target_after
        // values. Verification binds a fresh report to the current DB snapshot
        // and rechecks the target receipt below. Unrelated writes after that
        // completed apply transaction must not strand an otherwise valid
        // receipt.
        validate_tag_record_set_on_connection(
            &connection,
            manifest,
            prior_verified_tag_targets,
            true,
        )
        .await?;
        validate_deterministic_target_resolved(db, manifest, page_root).await?;
        let (target_now, _) = match manifest.target() {
            RepairTarget::PageProjection { page_id, .. }
                if manifest.writer() == RepairWriter::RenamePageTitle =>
            {
                let rollback = rename_rollback.ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let projection = rename_projection_session
                    .ok_or_else(|| {
                        WenlanError::Validation(
                            "page projection repair root unavailable".to_string(),
                        )
                    })?
                    .locked();
                let current =
                    capture_rename_page_title_on_connection(&connection, &projection, page_id)
                        .await?;
                let scan = projection.scan_page_root_controlled(
                    true,
                    &PageScanControl::with_timeout(Duration::from_secs(30)),
                )?;
                let excluded = rename_page_title_excluded_paths(rollback)?;
                let non_target_now = rename_page_title_non_target_receipt(
                    &effect_guard_receipt(0),
                    scan.non_target_digest(&excluded),
                    &current,
                )?;
                if non_target_now != *apply_receipt.non_target_after() {
                    return Err(WenlanError::Conflict(
                        "repair_non_target_state_changed".to_string(),
                    ));
                }
                (rename_page_title_receipt(&current)?, 1)
            }
            RepairTarget::PageProjection { page_id, .. }
                if manifest.writer() == RepairWriter::QuarantineStalePageProjection =>
            {
                let rollback = rollback.ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let page_root = page_root.ok_or_else(|| {
                    WenlanError::Validation("page projection repair root unavailable".to_string())
                })?;
                let (source_path, quarantine_path) = stale_page_projection_paths(rollback)?;
                let target_now =
                    KnowledgeProjectionWrite::with_projection_lock(page_root, |projection| {
                        let excluded = BTreeSet::from([
                            ".wenlan".to_string(),
                            ".wenlan/state.json".to_string(),
                            ".wenlan/orphaned".to_string(),
                            source_path.clone(),
                            quarantine_path.clone(),
                        ]);
                        let scan = projection.scan_page_root_controlled(
                            true,
                            &PageScanControl::with_timeout(Duration::from_secs(30)),
                        )?;
                        let current = projection.capture_stale_page_projection_current(
                            page_id,
                            &source_path,
                            &quarantine_path,
                        )?;
                        let non_target_now = page_projection_non_target_receipt(
                            scan.non_target_digest(&excluded),
                            &current,
                        )?;
                        if non_target_now != *apply_receipt.non_target_after() {
                            return Err(WenlanError::Conflict(
                                "repair_non_target_state_changed".to_string(),
                            ));
                        }
                        target_receipt(&current)
                    })?;
                (target_now, 0)
            }
            RepairTarget::PageProjection { page_id, .. } => {
                let rollback = rollback.ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let page_root = page_root.ok_or_else(|| {
                    WenlanError::Validation("page projection repair root unavailable".to_string())
                })?;
                let paths = projection_rollback_paths(rollback)?;
                let page_row = projection_page_row_on_connection(&connection, page_id).await?;
                let target_now = KnowledgeProjectionWrite::with_projection_lock(page_root, |_| {
                    let scan = scan_page_root_controlled(
                        page_root,
                        true,
                        &PageScanControl::with_timeout(Duration::from_secs(30)),
                    )
                    .map_err(|error| {
                        WenlanError::Validation(format!("repair projection scan: {error}"))
                    })?;
                    let current = capture_page_projection_from_row(
                        page_root,
                        page_id,
                        page_row,
                        &paths,
                        &rollback.table,
                    )?;
                    let non_target_now = page_projection_non_target_receipt(
                        scan.non_target_digest(&paths),
                        &current,
                    )?;
                    if non_target_now != *apply_receipt.non_target_after() {
                        return Err(WenlanError::Conflict(
                            "repair_non_target_state_changed".to_string(),
                        ));
                    }
                    target_receipt(&current)
                })?;
                (target_now, 1)
            }
            _ => repair_target_receipt_on_connection(&connection, manifest.target()).await?,
        };
        if target_now != *apply_receipt.after_target_receipt() {
            return Err(WenlanError::Conflict(
                "repair_verification_state_changed".to_string(),
            ));
        }
        let draft = match request.deep_report() {
            Some(deep) => RepairVerificationReceiptDraft::try_new(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                verified_at,
                request.general_report().snapshots().clone(),
                deep.snapshots().clone(),
            ),
            None => RepairVerificationReceiptDraft::try_new_general_only(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                verified_at,
                request.general_report().snapshots().clone(),
            ),
        }
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
        let receipt_digest = repair_digest(&draft.canonical_bytes()?);
        let receipt = RepairVerificationReceipt::from_draft(draft, receipt_digest);
        if let Some(session) = rename_projection_session {
            validate_current_page_receipts_on_repair_projection(
                request.general_report(),
                request.deep_report(),
                &session.locked(),
            )?;
            #[cfg(test)]
            run_repair_verification_test_checkpoint(
                RepairVerificationTestCheckpointKind::BeforeReceiptPersist,
            );
            store.persist_verification_receipt(&receipt)?;
            #[cfg(test)]
            run_repair_verification_test_checkpoint(
                RepairVerificationTestCheckpointKind::AfterReceiptPersist,
            );
            Ok(receipt)
        } else if let Some(page_root) = page_root {
            KnowledgeProjectionWrite::with_projection_lock(page_root, |_| {
                validate_current_page_receipts_locked(
                    request.general_report(),
                    request.deep_report(),
                    Some(page_root),
                )?;
                #[cfg(test)]
                run_repair_verification_test_checkpoint(
                    RepairVerificationTestCheckpointKind::BeforeReceiptPersist,
                );
                store.persist_verification_receipt(&receipt)?;
                #[cfg(test)]
                run_repair_verification_test_checkpoint(
                    RepairVerificationTestCheckpointKind::AfterReceiptPersist,
                );
                Ok(receipt)
            })
        } else {
            #[cfg(test)]
            run_repair_verification_test_checkpoint(
                RepairVerificationTestCheckpointKind::BeforeReceiptPersist,
            );
            store.persist_verification_receipt(&receipt)?;
            #[cfg(test)]
            run_repair_verification_test_checkpoint(
                RepairVerificationTestCheckpointKind::AfterReceiptPersist,
            );
            Ok(receipt)
        }
    }
    .await;
    let receipt = match result {
        Ok(receipt) => receipt,
        Err(error) => {
            let _ = connection.execute("ROLLBACK", ()).await;
            return Err(error);
        }
    };
    #[cfg(test)]
    if consume_repair_verification_test_commit_failure() {
        return Err(WenlanError::VectorDb(
            "repair verify commit: injected test failure".to_string(),
        ));
    }
    connection
        .execute("COMMIT", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair verify commit: {error}")))?;
    Ok(receipt)
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct RepairVerificationDbContender {
    connection: std::sync::Arc<tokio::sync::Mutex<libsql::Connection>>,
    pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
    entered: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: std::sync::Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
    completion: std::sync::Arc<std::sync::Mutex<Option<std::sync::mpsc::Receiver<()>>>>,
}

#[cfg(test)]
impl RepairVerificationDbContender {
    pub(crate) fn new(db: &MemoryDB) -> Self {
        Self {
            connection: std::sync::Arc::clone(&db.conn),
            pending: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            entered: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            thread: std::sync::Arc::new(std::sync::Mutex::new(None)),
            completion: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    pub(crate) fn start_and_observe_pending(&self) {
        use std::future::Future as _;

        let mut thread = self.thread.lock().unwrap();
        assert!(thread.is_none(), "DB contender started more than once");
        let connection = std::sync::Arc::clone(&self.connection);
        let pending = std::sync::Arc::clone(&self.pending);
        let entered = std::sync::Arc::clone(&self.entered);
        let (pending_tx, pending_rx) = std::sync::mpsc::sync_channel(0);
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();
        let mut completion = self.completion.lock().unwrap();
        assert!(
            completion.is_none(),
            "DB contender completion receiver already exists"
        );
        *completion = Some(completion_rx);
        drop(completion);
        *thread = Some(std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async move {
                    let mut lock = Box::pin(connection.lock_owned());
                    let mut pending_tx = Some(pending_tx);
                    let guard = std::future::poll_fn(|context| match lock.as_mut().poll(context) {
                        std::task::Poll::Pending => {
                            pending.store(true, std::sync::atomic::Ordering::SeqCst);
                            if let Some(pending_tx) = pending_tx.take() {
                                pending_tx.send(()).unwrap();
                            }
                            std::task::Poll::Pending
                        }
                        std::task::Poll::Ready(guard) => {
                            pending.store(false, std::sync::atomic::Ordering::SeqCst);
                            entered.store(true, std::sync::atomic::Ordering::SeqCst);
                            std::task::Poll::Ready(guard)
                        }
                    })
                    .await;
                    drop(guard);
                    completion_tx.send(()).unwrap();
                });
        }));
        pending_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("DB contender never reached Pending");
        drop(thread);
        self.assert_pending();
    }

    pub(crate) fn assert_pending(&self) {
        assert!(
            self.pending.load(std::sync::atomic::Ordering::SeqCst),
            "DB contender was not observed Pending"
        );
        assert!(
            !self.entered.load(std::sync::atomic::Ordering::SeqCst),
            "DB contender entered before verification released the connection"
        );
    }

    pub(crate) fn assert_entered_after_verification(&self) {
        let completion = self
            .completion
            .lock()
            .unwrap()
            .take()
            .expect("DB contender completion receiver was never created");
        completion
            .recv_timeout(Duration::from_secs(5))
            .expect("DB contender did not complete after verification returned");
        let thread = self
            .thread
            .lock()
            .unwrap()
            .take()
            .expect("DB contender was never started");
        thread.join().unwrap();
        assert!(
            self.entered.load(std::sync::atomic::Ordering::SeqCst),
            "DB contender did not enter after verification returned"
        );
        assert!(!self.pending.load(std::sync::atomic::Ordering::SeqCst));
    }
}

#[cfg(test)]
pub(crate) async fn assert_repair_verification_transaction_reusable(db: &MemoryDB) {
    let connection = db.conn.lock().await;
    connection.execute("BEGIN IMMEDIATE", ()).await.unwrap();
    connection.execute("ROLLBACK", ()).await.unwrap();
}

#[cfg(test)]
pub(crate) async fn rollback_repair_verification_test_transaction(db: &MemoryDB) {
    db.conn.lock().await.execute("ROLLBACK", ()).await.unwrap();
}
