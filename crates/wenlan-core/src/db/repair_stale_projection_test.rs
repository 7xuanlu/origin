// SPDX-License-Identifier: Apache-2.0

use super::{
    repair_stale_projection::{
        quarantine_stale_page_projection_cas_with_apply_journal,
        recover_stale_page_projection_apply_receipt, with_stale_projection_test_checkpoints,
        StaleProjectionTestCheckpoints,
    },
    MemoryDB,
};
use crate::{
    error::WenlanError,
    lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    },
    repair::{RepairArtifactStore, StalePageProjectionRecoveryJournal, StoredRollbackArtifact},
    repair_plan::prepare_repair_plan,
};
use fs2::FileExt;
use std::{
    fs::OpenOptions,
    future::{poll_fn, Future},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::Poll,
    time::{Duration, Instant},
};
use tokio::{sync::Notify, task::JoinHandle, time::timeout};
use wenlan_types::{
    lint::{LintProfile, LintQuery},
    repair::{
        ApplyRepairRequest, RepairApplyReceipt, RepairApplyReceiptDraft, RepairDigest,
        RepairLintScope, RepairManifest, RepairMutation, RepairWriter,
    },
    repair_plan::{RepairPlanRequest, RepairResolution},
};

static STALE_PROJECTION_TEST_MUTEX: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct StaleProjectionFixture {
    db: Arc<MemoryDB>,
    _db_dir: tempfile::TempDir,
    page_root: tempfile::TempDir,
    _repair_root: tempfile::TempDir,
    store: RepairArtifactStore,
    manifest: RepairManifest,
    rollback: StoredRollbackArtifact,
    original_state: Vec<u8>,
    source_bytes: Vec<u8>,
    source_path: String,
    quarantine_path: String,
}

async fn stale_projection_fixture(page_id: &str) -> StaleProjectionFixture {
    let (db, db_dir) = super::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let source_path = format!("{page_id}.md");
    let pages = serde_json::Map::from_iter([
        (
            page_id.to_string(),
            serde_json::json!({
                "file": source_path,
                "version": 1,
            }),
        ),
        (
            "page-other".to_string(),
            serde_json::json!({
                "file": "other.md",
                "version": 1,
            }),
        ),
    ]);
    let original_state = serde_json::to_vec(&serde_json::json!({
        "schema_version": 2,
        "pages": pages,
    }))
    .unwrap();
    std::fs::write(page_root.path().join(".wenlan/state.json"), &original_state).unwrap();
    let source_bytes =
        format!("---\norigin_id: {page_id}\norigin_version: 1\n---\nstale\n").into_bytes();
    std::fs::write(page_root.path().join(&source_path), &source_bytes).unwrap();
    std::fs::write(page_root.path().join("other.md"), b"other").unwrap();

    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let manifest = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::QuarantineStalePageProjection =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .expect("ready stale projection manifest");
    let rollback_path = store
        .manifest_dir(manifest.manifest_id())
        .unwrap()
        .join(manifest.rollback().relative_path());
    let rollback =
        serde_json::from_slice::<StoredRollbackArtifact>(&std::fs::read(rollback_path).unwrap())
            .unwrap();
    let (manifest_source, quarantine_path) = match manifest.mutation() {
        RepairMutation::QuarantineStalePageProjection {
            source_path,
            quarantine_path,
        } => (source_path.clone(), quarantine_path.clone()),
        mutation => panic!("unexpected stale projection mutation: {mutation:?}"),
    };
    assert_eq!(manifest_source, source_path);

    StaleProjectionFixture {
        db: Arc::new(db),
        _db_dir: db_dir,
        page_root,
        _repair_root: repair_root,
        store,
        manifest,
        rollback,
        original_state,
        source_bytes,
        source_path,
        quarantine_path,
    }
}

struct TwoManifestFixture {
    db: Arc<MemoryDB>,
    _db_dir: tempfile::TempDir,
    page_root: tempfile::TempDir,
    _repair_root: tempfile::TempDir,
    store: RepairArtifactStore,
    manifests: Vec<RepairManifest>,
}

async fn two_manifest_fixture() -> TwoManifestFixture {
    let (db, db_dir) = super::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let ids = ["stale-dynamic-a", "stale-dynamic-b"];
    let pages = ids
        .iter()
        .enumerate()
        .map(|(index, page_id)| {
            (
                (*page_id).to_string(),
                serde_json::json!({"file": format!("{index}.md"), "version": 1}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    std::fs::write(
        page_root.path().join(".wenlan/state.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 2,
            "pages": pages,
        }))
        .unwrap(),
    )
    .unwrap();
    for (index, page_id) in ids.iter().enumerate() {
        std::fs::write(
            page_root.path().join(format!("{index}.md")),
            format!("---\norigin_id: {page_id}\norigin_version: 1\n---\n{page_id}\n"),
        )
        .unwrap();
    }
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let manifests = plan
        .entries()
        .iter()
        .filter_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::QuarantineStalePageProjection =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(manifests.len(), 2);
    TwoManifestFixture {
        db: Arc::new(db),
        _db_dir: db_dir,
        page_root,
        _repair_root: repair_root,
        store,
        manifests,
    }
}

impl StaleProjectionFixture {
    fn state_path(&self) -> std::path::PathBuf {
        self.page_root.path().join(".wenlan/state.json")
    }

    fn source_path(&self) -> std::path::PathBuf {
        self.page_root.path().join(&self.source_path)
    }

    fn quarantine_path(&self) -> std::path::PathBuf {
        self.page_root.path().join(&self.quarantine_path)
    }

    fn manifest_dir(&self) -> std::path::PathBuf {
        self.store
            .manifest_dir(self.manifest.manifest_id())
            .unwrap()
    }

    fn pending_path(&self) -> std::path::PathBuf {
        self.manifest_dir().join(".apply-receipt.json.pending")
    }

    fn final_path(&self) -> std::path::PathBuf {
        self.manifest_dir().join("apply-receipt.json")
    }

    fn journal_path(&self) -> std::path::PathBuf {
        self.manifest_dir()
            .join(".stale-page-projection-apply-journal-v1.json")
    }

    fn projection_stage_path(&self) -> std::path::PathBuf {
        self.page_root.path().join(".wenlan").join(
            crate::export::knowledge::projection_unlink_stage_name(self.manifest.manifest_id()),
        )
    }

    fn assert_original_projection(&self) {
        assert_eq!(
            std::fs::read(self.state_path()).unwrap(),
            self.original_state
        );
        assert_eq!(
            std::fs::read(self.source_path()).unwrap(),
            self.source_bytes
        );
        assert!(!self.quarantine_path().exists());
        assert_eq!(
            std::fs::read(self.page_root.path().join("other.md")).unwrap(),
            b"other"
        );
    }

    async fn apply(&self, now_epoch: i64) -> Result<RepairApplyReceipt, WenlanError> {
        crate::repair::apply_repair_with_pages(
            &self.db,
            &self.store,
            exact_request(&self.manifest),
            Some(self.page_root.path()),
            now_epoch,
        )
        .await
    }
}

fn exact_request(manifest: &RepairManifest) -> ApplyRepairRequest {
    ApplyRepairRequest::try_new(
        manifest.manifest_id().to_string(),
        manifest.manifest_digest().clone(),
        format!(
            "apply repair {} {}",
            manifest.manifest_id(),
            manifest.manifest_digest().as_str()
        ),
    )
    .unwrap()
}

struct DbWriterContender {
    start: Arc<Notify>,
    pending: Arc<AtomicBool>,
    entered: Arc<AtomicBool>,
    task: JoinHandle<()>,
}

impl DbWriterContender {
    fn prestarted(db: Arc<MemoryDB>) -> Self {
        let start = Arc::new(Notify::new());
        let pending = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(AtomicBool::new(false));
        let task_start = Arc::clone(&start);
        let task_pending = Arc::clone(&pending);
        let task_entered = Arc::clone(&entered);
        let task = tokio::spawn(async move {
            task_start.notified().await;
            let mut lock = Box::pin(db.conn.lock());
            let guard = poll_fn(|cx| match Future::poll(Pin::as_mut(&mut lock), cx) {
                Poll::Pending => {
                    task_pending.store(true, Ordering::SeqCst);
                    Poll::Pending
                }
                Poll::Ready(guard) => Poll::Ready(guard),
            })
            .await;
            task_entered.store(true, Ordering::SeqCst);
            guard
                .execute("UPDATE pages SET id=id WHERE 0", ())
                .await
                .unwrap();
        });
        Self {
            start,
            pending,
            entered,
            task,
        }
    }

    fn release_and_require_pending(
        &self,
    ) -> Box<dyn FnOnce() -> Result<(), WenlanError> + 'static> {
        let start = Arc::clone(&self.start);
        let pending = Arc::clone(&self.pending);
        let entered = Arc::clone(&self.entered);
        Box::new(move || {
            start.notify_one();
            let deadline = Instant::now() + Duration::from_secs(2);
            while !pending.load(Ordering::SeqCst) {
                if Instant::now() >= deadline {
                    return Err(WenlanError::Conflict(
                        "DB writer contender did not observe Pending".to_string(),
                    ));
                }
                std::thread::yield_now();
            }
            if entered.load(Ordering::SeqCst) {
                return Err(WenlanError::Conflict(
                    "DB writer contender entered during stale projection repair".to_string(),
                ));
            }
            Ok(())
        })
    }

    fn assert_pending(&self) {
        assert!(self.pending.load(Ordering::SeqCst));
        assert!(!self.entered.load(Ordering::SeqCst));
    }

    async fn assert_enters(self) {
        timeout(Duration::from_secs(2), self.task)
            .await
            .expect("DB writer must enter after stale projection operation returns")
            .unwrap();
        assert!(self.entered.load(Ordering::SeqCst));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn normal_apply_pins_dynamic_journal_digest_and_db_contender() {
    let _serial = STALE_PROJECTION_TEST_MUTEX.lock().await;
    let fixture = stale_projection_fixture("stale-normal").await;
    let pinned_digest = {
        let connection = fixture.db.conn.lock().await;
        crate::repair::database_content_digest(&connection)
            .await
            .unwrap()
    };
    let contender = DbWriterContender::prestarted(Arc::clone(&fixture.db));
    let proof_contender = DbWriterContender::prestarted(Arc::clone(&fixture.db));
    let release_proof_contender = proof_contender.release_and_require_pending();
    let entered_after_journal = Arc::clone(&contender.entered);
    let entered_after_pin = Arc::clone(&contender.entered);
    let entered_before_source = Arc::clone(&contender.entered);
    let captured_journal = std::sync::Mutex::new(None::<StoredRollbackArtifact>);
    let proof_seen = Arc::new(AtomicBool::new(false));
    let proof_seen_in_hook = Arc::clone(&proof_seen);
    let proof_db = Arc::clone(&fixture.db);

    let proof = timeout(
        Duration::from_secs(3),
        with_stale_projection_test_checkpoints(
            StaleProjectionTestCheckpoints {
                after_db_snapshot_before_projection_lock: Some(
                    contender.release_and_require_pending(),
                ),
                after_journal_persist_before_pin: Some(Box::new(move || {
                    assert!(!entered_after_journal.load(Ordering::SeqCst));
                    Ok(())
                })),
                after_pin_before_link: Some(Box::new(move || {
                    assert!(!entered_after_pin.load(Ordering::SeqCst));
                    Ok(())
                })),
                before_source_stage: Some(Box::new(move || {
                    assert!(!entered_before_source.load(Ordering::SeqCst));
                    Ok(())
                })),
                ..Default::default()
            },
            quarantine_stale_page_projection_cas_with_apply_journal(
                &fixture.db,
                &fixture.manifest,
                &fixture.rollback,
                fixture.page_root.path(),
                |current| {
                    *captured_journal.lock().unwrap() = Some(current.clone());
                    Ok(())
                },
                |proof| {
                    release_proof_contender()?;
                    assert!(
                        proof_db.conn.try_lock().is_err(),
                        "the proof callback must run while the DB guard is held"
                    );
                    contender.assert_pending();
                    proof_contender.assert_pending();
                    assert_eq!(proof.post_apply_db_digest(), &pinned_digest);
                    proof_seen_in_hook.store(true, Ordering::SeqCst);
                    Ok(())
                },
            ),
        ),
    )
    .await
    .expect("normal stale projection apply must not self-deadlock")
    .unwrap();

    assert_eq!(proof.post_apply_db_digest(), &pinned_digest);
    assert!(proof_seen.load(Ordering::SeqCst));
    assert_eq!(
        captured_journal.into_inner().unwrap().unwrap(),
        fixture.rollback
    );
    assert!(!fixture.source_path().exists());
    assert_eq!(
        std::fs::read(fixture.quarantine_path()).unwrap(),
        fixture.source_bytes
    );
    contender.assert_enters().await;
    proof_contender.assert_enters().await;
}

#[tokio::test]
async fn database_mutex_precedes_projection_lock_for_apply_and_recovery() {
    let _serial = STALE_PROJECTION_TEST_MUTEX.lock().await;
    let fixture = stale_projection_fixture("stale-lock-order").await;
    let lock_path = fixture.page_root.path().join(".wenlan/.projection.lock");
    let file_lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .unwrap();
    file_lock.lock_exclusive().unwrap();

    let apply_observed = Arc::new(AtomicBool::new(false));
    let apply_observed_in_hook = Arc::clone(&apply_observed);
    let apply_db = Arc::clone(&fixture.db);
    let apply = with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            after_db_snapshot_before_projection_lock: Some(Box::new(move || {
                apply_observed_in_hook.store(apply_db.conn.try_lock().is_err(), Ordering::SeqCst);
                Ok(())
            })),
            ..Default::default()
        },
        quarantine_stale_page_projection_cas_with_apply_journal(
            &fixture.db,
            &fixture.manifest,
            &fixture.rollback,
            fixture.page_root.path(),
            |_| Ok(()),
            |_| Ok(()),
        ),
    )
    .await;
    assert!(matches!(
        apply,
        Err(WenlanError::Conflict(message)) if message == "page_projection_locked"
    ));
    assert!(apply_observed.load(Ordering::SeqCst));

    let recovery_observed = Arc::new(AtomicBool::new(false));
    let recovery_observed_in_hook = Arc::clone(&recovery_observed);
    let recovery_db = Arc::clone(&fixture.db);
    let recovery = with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            after_db_snapshot_before_projection_lock: Some(Box::new(move || {
                recovery_observed_in_hook
                    .store(recovery_db.conn.try_lock().is_err(), Ordering::SeqCst);
                Ok(())
            })),
            ..Default::default()
        },
        recover_stale_page_projection_apply_receipt(
            &fixture.db,
            &fixture.store,
            &fixture.manifest,
            &fixture.rollback,
            Some(fixture.page_root.path()),
            None,
        ),
    )
    .await;
    assert!(matches!(
        recovery,
        Err(WenlanError::Conflict(message)) if message == "repair_apply_recovery_required"
    ));
    assert!(recovery_observed.load(Ordering::SeqCst));
    FileExt::unlock(&file_lock).unwrap();
}

#[tokio::test]
async fn owner_appearance_blocks_apply_and_recovery_without_projection_mutation() {
    let _serial = STALE_PROJECTION_TEST_MUTEX.lock().await;
    let fixture = stale_projection_fixture("stale-owner").await;
    fixture
        .db
        .insert_page(
            "stale-owner",
            "Owner",
            None,
            "database owner",
            None,
            None,
            &[],
            "2026-07-17T00:00:00Z",
        )
        .await
        .unwrap();
    let journal_called = Arc::new(AtomicBool::new(false));
    let journal_called_in_hook = Arc::clone(&journal_called);
    let apply = quarantine_stale_page_projection_cas_with_apply_journal(
        &fixture.db,
        &fixture.manifest,
        &fixture.rollback,
        fixture.page_root.path(),
        move |_| {
            journal_called_in_hook.store(true, Ordering::SeqCst);
            Ok(())
        },
        |_| Ok(()),
    )
    .await;
    assert!(matches!(
        apply,
        Err(WenlanError::Conflict(message)) if message == "repair_target_stale"
    ));
    assert!(!journal_called.load(Ordering::SeqCst));

    let recovery = recover_stale_page_projection_apply_receipt(
        &fixture.db,
        &fixture.store,
        &fixture.manifest,
        &fixture.rollback,
        Some(fixture.page_root.path()),
        None,
    )
    .await;
    assert!(matches!(
        recovery,
        Err(WenlanError::Conflict(message)) if message == "repair_apply_recovery_required"
    ));
    fixture.assert_original_projection();
    assert!(!fixture.pending_path().exists());
    assert!(!fixture.journal_path().exists());
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn journal_failure_and_after_publish_crash_have_distinct_artifact_states() {
    let _serial = STALE_PROJECTION_TEST_MUTEX.lock().await;
    let before = stale_projection_fixture("stale-journal-before").await;
    let before_checkpoint = Arc::new(AtomicBool::new(false));
    let before_checkpoint_in_hook = Arc::clone(&before_checkpoint);
    let error = with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            before_journal_persist: Some(Box::new(move || {
                before_checkpoint_in_hook.store(true, Ordering::SeqCst);
                Ok(())
            })),
            ..Default::default()
        },
        quarantine_stale_page_projection_cas_with_apply_journal(
            &before.db,
            &before.manifest,
            &before.rollback,
            before.page_root.path(),
            |_| {
                Err(WenlanError::Validation(
                    "forced_journal_persist_failure".to_string(),
                ))
            },
            |_| Ok(()),
        ),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Validation error: forced_journal_persist_failure"
    );
    assert!(before_checkpoint.load(Ordering::SeqCst));
    before.assert_original_projection();
    assert!(!before.projection_stage_path().exists());
    assert!(!before.pending_path().exists());
    assert!(!before.journal_path().exists());

    let after = stale_projection_fixture("stale-journal-after").await;
    let error = with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            after_journal_persist_before_pin: Some(Box::new(|| {
                Err(WenlanError::Conflict(
                    "forced_after_journal_publish".to_string(),
                ))
            })),
            ..Default::default()
        },
        after.apply(1_721_000_101),
    )
    .await
    .unwrap_err();
    assert_eq!(error.to_string(), "Conflict: forced_after_journal_publish");
    after.assert_original_projection();
    assert!(after.pending_path().is_file());
    assert!(after.journal_path().is_file());

    let receipt = after.apply(1_721_000_102).await.unwrap();
    assert_eq!(
        receipt.writer(),
        RepairWriter::QuarantineStalePageProjection
    );
    assert!(!after.pending_path().exists());
    assert!(!after.journal_path().exists());
    assert!(after.final_path().is_file());
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn two_manifest_dynamic_journal_recovery_preserves_first_repair() {
    let _serial = STALE_PROJECTION_TEST_MUTEX.lock().await;
    let fixture = two_manifest_fixture().await;
    let first = &fixture.manifests[0];
    let second = &fixture.manifests[1];
    crate::repair::apply_repair_with_pages(
        &fixture.db,
        &fixture.store,
        exact_request(first),
        Some(fixture.page_root.path()),
        1_721_000_201,
    )
    .await
    .unwrap();
    let (first_quarantine, second_source, second_quarantine) =
        match (first.mutation(), second.mutation()) {
            (
                RepairMutation::QuarantineStalePageProjection {
                    quarantine_path: first_quarantine,
                    ..
                },
                RepairMutation::QuarantineStalePageProjection {
                    source_path,
                    quarantine_path,
                },
            ) => (
                first_quarantine.clone(),
                source_path.clone(),
                quarantine_path.clone(),
            ),
            _ => unreachable!("two stale projection manifests"),
        };
    let first_quarantine = fixture.page_root.path().join(first_quarantine);
    let first_bytes = std::fs::read(&first_quarantine).unwrap();
    let expected_dynamic = crate::repair::capture_stale_page_projection_current(
        fixture.page_root.path(),
        match second.target() {
            wenlan_types::repair::RepairTarget::PageProjection { page_id, .. } => page_id,
            _ => unreachable!("stale page target"),
        },
        &second_source,
        &second_quarantine,
    )
    .unwrap();
    let second_plan_rollback = {
        let path = fixture
            .store
            .manifest_dir(second.manifest_id())
            .unwrap()
            .join(second.rollback().relative_path());
        serde_json::from_slice::<StoredRollbackArtifact>(&std::fs::read(path).unwrap()).unwrap()
    };
    assert_ne!(expected_dynamic, second_plan_rollback);

    let error = with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            after_journal_persist_before_pin: Some(Box::new(|| {
                Err(WenlanError::Conflict(
                    "forced_second_after_journal".to_string(),
                ))
            })),
            ..Default::default()
        },
        crate::repair::apply_repair_with_pages(
            &fixture.db,
            &fixture.store,
            exact_request(second),
            Some(fixture.page_root.path()),
            1_721_000_203,
        ),
    )
    .await
    .unwrap_err();
    assert_eq!(error.to_string(), "Conflict: forced_second_after_journal");
    let journal = fixture
        .store
        .inspect_stale_page_projection_recovery_journal(second)
        .unwrap();
    assert!(matches!(
        journal,
        StalePageProjectionRecoveryJournal::Present(ref current)
            if current == &expected_dynamic
    ));
    assert!(fixture
        .store
        .manifest_dir(second.manifest_id())
        .unwrap()
        .join(".apply-receipt.json.pending")
        .is_file());
    assert_eq!(std::fs::read(&first_quarantine).unwrap(), first_bytes);

    crate::repair::apply_repair_with_pages(
        &fixture.db,
        &fixture.store,
        exact_request(second),
        Some(fixture.page_root.path()),
        1_721_000_204,
    )
    .await
    .unwrap();
    assert_eq!(std::fs::read(&first_quarantine).unwrap(), first_bytes);
    assert!(!fixture.page_root.path().join(second_source).exists());
    assert!(fixture.page_root.path().join(second_quarantine).is_file());
}

#[tokio::test]
async fn mutation_split_restore_and_restore_failure_mapping_are_exact() {
    let _serial = STALE_PROJECTION_TEST_MUTEX.lock().await;
    let self_cleaned = stale_projection_fixture("stale-self-cleaned").await;
    let raced_root = self_cleaned.page_root.path().to_path_buf();
    let raced_source = self_cleaned.source_path.clone();
    let replacement = b"replacement survives internal cleanup".to_vec();
    let replacement_in_hook = replacement.clone();
    let error = crate::post_write::quarantine_stale_page_projection_cas_with_before_source_stage(
        &self_cleaned.db,
        &self_cleaned.manifest,
        &self_cleaned.rollback,
        self_cleaned.page_root.path(),
        move || {
            let temporary = raced_root.join("replacement.tmp");
            std::fs::write(&temporary, &replacement_in_hook)?;
            std::fs::rename(temporary, raced_root.join(raced_source))?;
            Ok(())
        },
        |_| Ok(()),
    )
    .await
    .unwrap_err();
    assert_eq!(error.to_string(), "Conflict: repair_target_stale");
    assert_eq!(
        std::fs::read(self_cleaned.source_path()).unwrap(),
        replacement
    );
    assert!(!self_cleaned.quarantine_path().exists());

    let restored = stale_projection_fixture("stale-restored").await;
    let before_restore = Arc::new(AtomicBool::new(false));
    let after_restore = Arc::new(AtomicBool::new(false));
    let error = with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            before_snapshot_restore: Some(Box::new({
                let before_restore = Arc::clone(&before_restore);
                move || {
                    before_restore.store(true, Ordering::SeqCst);
                    Ok(())
                }
            })),
            after_snapshot_restore: Some(Box::new({
                let after_restore = Arc::clone(&after_restore);
                move || {
                    after_restore.store(true, Ordering::SeqCst);
                    Ok(())
                }
            })),
            ..Default::default()
        },
        quarantine_stale_page_projection_cas_with_apply_journal(
            &restored.db,
            &restored.manifest,
            &restored.rollback,
            restored.page_root.path(),
            |_| Ok(()),
            |_| Err(WenlanError::Validation("forced_before_commit".to_string())),
        ),
    )
    .await
    .unwrap_err();
    assert_eq!(error.to_string(), "Validation error: forced_before_commit");
    assert!(before_restore.load(Ordering::SeqCst));
    assert!(after_restore.load(Ordering::SeqCst));
    restored.assert_original_projection();

    let failed_restore = stale_projection_fixture("stale-restore-failure").await;
    let error = with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            before_snapshot_restore: Some(Box::new(|| {
                Err(WenlanError::Validation(
                    "forced_snapshot_restore_failure".to_string(),
                ))
            })),
            ..Default::default()
        },
        quarantine_stale_page_projection_cas_with_apply_journal(
            &failed_restore.db,
            &failed_restore.manifest,
            &failed_restore.rollback,
            failed_restore.page_root.path(),
            |_| Ok(()),
            |_| Err(WenlanError::Validation("forced_before_commit".to_string())),
        ),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Vector DB error: Validation error: forced_before_commit; stale projection rollback \
         failed: Validation error: forced_snapshot_restore_failure"
    );
    assert!(!failed_restore.source_path().exists());
    assert!(failed_restore.quarantine_path().is_file());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn recovery_original_post_mismatch_unknown_and_restore_post_are_closed() {
    let _serial = STALE_PROJECTION_TEST_MUTEX.lock().await;

    let original = stale_projection_fixture("stale-recovery-original").await;
    std::fs::write(original.pending_path(), b"incomplete").unwrap();
    let contender = DbWriterContender::prestarted(Arc::clone(&original.db));
    let pending_before_artifact = original.pending_path();
    let pending_after_artifact = original.pending_path();
    let entered_before_artifact = Arc::clone(&contender.entered);
    let entered_after_artifact = Arc::clone(&contender.entered);
    let recovered = with_stale_projection_test_checkpoints(
        StaleProjectionTestCheckpoints {
            after_db_snapshot_before_projection_lock: Some(contender.release_and_require_pending()),
            after_recovery_state_before_artifact: Some(Box::new(move || {
                assert!(pending_before_artifact.is_file());
                assert!(!entered_before_artifact.load(Ordering::SeqCst));
                Ok(())
            })),
            after_recovery_artifact: Some(Box::new(move || {
                assert!(!pending_after_artifact.exists());
                assert!(!entered_after_artifact.load(Ordering::SeqCst));
                Ok(())
            })),
            ..Default::default()
        },
        recover_stale_page_projection_apply_receipt(
            &original.db,
            &original.store,
            &original.manifest,
            &original.rollback,
            Some(original.page_root.path()),
            None,
        ),
    )
    .await
    .unwrap();
    assert!(recovered.is_none());
    original.assert_original_projection();
    assert!(!original.pending_path().exists());
    assert!(!original.journal_path().exists());
    contender.assert_enters().await;

    let post = stale_projection_fixture("stale-recovery-post").await;
    let receipt = post.apply(1_721_000_301).await.unwrap();
    std::fs::rename(post.final_path(), post.pending_path()).unwrap();
    let recovered = recover_stale_page_projection_apply_receipt(
        &post.db,
        &post.store,
        &post.manifest,
        &post.rollback,
        Some(post.page_root.path()),
        Some(receipt.clone()),
    )
    .await
    .unwrap();
    assert_eq!(recovered, Some(receipt));
    assert!(post.final_path().is_file());
    assert!(!post.pending_path().exists());
    assert!(!post.journal_path().exists());
    assert!(!post.source_path().exists());
    assert_eq!(
        std::fs::read(post.quarantine_path()).unwrap(),
        post.source_bytes
    );

    let mismatch = stale_projection_fixture("stale-recovery-mismatch").await;
    let receipt = mismatch.apply(1_721_000_302).await.unwrap();
    std::fs::remove_file(mismatch.final_path()).unwrap();
    let wrong_after =
        RepairDigest::parse("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            .unwrap();
    let draft = RepairApplyReceiptDraft::try_new(
        mismatch.manifest.manifest_id().to_string(),
        mismatch.manifest.manifest_digest().clone(),
        1_721_000_302,
        receipt.before_target_receipt().clone(),
        wrong_after,
        receipt.non_target_before().clone(),
        receipt.non_target_after().clone(),
        receipt.post_apply_db_digest().unwrap().clone(),
        receipt.actual_effects().clone(),
        receipt.writer(),
    )
    .unwrap();
    let wrong_receipt = RepairApplyReceipt::from_draft(
        draft.clone(),
        crate::repair::repair_digest(&draft.canonical_bytes().unwrap()),
    );
    let pending_bytes = serde_json::to_vec_pretty(&wrong_receipt).unwrap();
    std::fs::write(mismatch.pending_path(), &pending_bytes).unwrap();
    let before_state = std::fs::read(mismatch.state_path()).unwrap();
    let before_quarantine = std::fs::read(mismatch.quarantine_path()).unwrap();
    let error = recover_stale_page_projection_apply_receipt(
        &mismatch.db,
        &mismatch.store,
        &mismatch.manifest,
        &mismatch.rollback,
        Some(mismatch.page_root.path()),
        Some(wrong_receipt),
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Conflict: repair_apply_recovery_required"
    );
    assert_eq!(
        std::fs::read(mismatch.pending_path()).unwrap(),
        pending_bytes
    );
    assert!(!mismatch.final_path().exists());
    assert_eq!(std::fs::read(mismatch.state_path()).unwrap(), before_state);
    assert!(!mismatch.source_path().exists());
    assert_eq!(
        std::fs::read(mismatch.quarantine_path()).unwrap(),
        before_quarantine
    );

    let unknown = stale_projection_fixture("stale-recovery-unknown").await;
    std::fs::write(unknown.pending_path(), b"opaque pending").unwrap();
    std::fs::write(unknown.state_path(), b"{\"foreign\":true}\n").unwrap();
    let foreign_state = std::fs::read(unknown.state_path()).unwrap();
    let error = recover_stale_page_projection_apply_receipt(
        &unknown.db,
        &unknown.store,
        &unknown.manifest,
        &unknown.rollback,
        Some(unknown.page_root.path()),
        None,
    )
    .await
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Conflict: repair_apply_recovery_required"
    );
    assert_eq!(
        std::fs::read(unknown.pending_path()).unwrap(),
        b"opaque pending"
    );
    assert_eq!(std::fs::read(unknown.state_path()).unwrap(), foreign_state);
    assert_eq!(
        std::fs::read(unknown.source_path()).unwrap(),
        unknown.source_bytes
    );
    assert!(!unknown.quarantine_path().exists());

    let restore_post = stale_projection_fixture("stale-recovery-restore-post").await;
    restore_post.apply(1_721_000_303).await.unwrap();
    std::fs::remove_file(restore_post.final_path()).unwrap();
    std::fs::write(restore_post.pending_path(), b"incomplete").unwrap();
    let recovered = recover_stale_page_projection_apply_receipt(
        &restore_post.db,
        &restore_post.store,
        &restore_post.manifest,
        &restore_post.rollback,
        Some(restore_post.page_root.path()),
        None,
    )
    .await
    .unwrap();
    assert!(recovered.is_none());
    restore_post.assert_original_projection();
    assert!(!restore_post.pending_path().exists());
    assert!(!restore_post.journal_path().exists());
}

#[test]
fn generic_final_receipt_cleanup_stays_outside_stale_recovery_child() {
    let repair = include_str!("../repair.rs");
    let child = include_str!("repair_stale_projection.rs");
    let recovery = repair
        .find("async fn recover_apply_receipt(")
        .expect("generic recovery declaration");
    let generic_final = repair[recovery..]
        .find("if final_path.exists()")
        .expect("generic final cleanup branch");
    let generic_final_return = repair[recovery..]
        .find("return Ok(Some(receipt));")
        .expect("generic final cleanup return");
    let first_stale_call = repair[recovery..]
        .find("recover_stale_page_projection_apply_receipt(")
        .expect("stale recovery call");
    assert!(
        generic_final < generic_final_return && generic_final_return < first_stale_call,
        "generic final cleanup must complete before stale recovery dispatch"
    );
    for forbidden in [
        "APPLY_RECEIPT_FILE",
        "load_apply_receipt(",
        "clear_pending_apply_receipt(",
        "clear_stale_page_projection_apply_journal(",
    ] {
        assert!(
            !child.contains(forbidden),
            "stale recovery child may not own generic artifact primitive {forbidden}"
        );
    }
}
