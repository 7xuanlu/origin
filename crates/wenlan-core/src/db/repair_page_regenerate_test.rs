// SPDX-License-Identifier: Apache-2.0

use super::{
    repair_page_regenerate::{
        regenerate_page_projection_cas, with_regenerate_page_test_checkpoints,
        RegeneratePageTestCheckpoint, RegeneratePageTestCheckpoints,
    },
    MemoryDB,
};
use crate::{
    error::WenlanError,
    export::knowledge::KnowledgeProjectionWrite,
    lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    },
    post_write::RepairWriteProof,
    repair::{database_content_digest, RepairArtifactStore, StoredRollbackArtifact},
    repair_plan::prepare_repair_plan,
};
use fs2::FileExt;
use std::{
    fs::OpenOptions,
    future::{poll_fn, Future},
    path::Path,
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
    repair::{ApplyRepairRequest, RepairDigest, RepairLintScope, RepairManifest, RepairWriter},
    repair_plan::{RepairPlanRequest, RepairResolution},
};

static REGENERATE_TEST_MUTEX: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Debug, PartialEq, Eq)]
struct ProjectionBytes {
    target: Vec<u8>,
    state: Option<Vec<u8>>,
    unrelated: Vec<u8>,
}

struct RegenerateFixture {
    db: Arc<MemoryDB>,
    _db_dir: tempfile::TempDir,
    page_root: tempfile::TempDir,
    _repair_root: tempfile::TempDir,
    store: RepairArtifactStore,
    manifest: RepairManifest,
    rollback: StoredRollbackArtifact,
    before_projection: ProjectionBytes,
    pinned_database_digest: RepairDigest,
}

async fn regenerate_fixture() -> RegenerateFixture {
    let (db, db_dir) = super::tests::test_db().await;
    db.insert_page(
        "page-regenerate",
        "Regenerate Page",
        Some("summary"),
        "canonical body",
        None,
        Some("work"),
        &[],
        "2026-07-17T00:00:00Z",
    )
    .await
    .unwrap();
    let page_root = tempfile::tempdir().unwrap();
    let page = db.get_page("page-regenerate").await.unwrap().unwrap();
    KnowledgeProjectionWrite::new(page_root.path().to_path_buf(), &db)
        .write_page(&page)
        .unwrap();
    std::fs::write(page_root.path().join("unrelated.txt"), b"preserve me").unwrap();
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET version=version+1 WHERE id='page-regenerate'",
            (),
        )
        .await
        .unwrap();

    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
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
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
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
                if manifest.writer() == RepairWriter::RegeneratePageProjection =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .expect("ready regenerate page manifest");
    let rollback_path = store
        .manifest_dir(manifest.manifest_id())
        .unwrap()
        .join(manifest.rollback().relative_path());
    let rollback =
        serde_json::from_slice::<StoredRollbackArtifact>(&std::fs::read(rollback_path).unwrap())
            .unwrap();
    let before_projection = projection_bytes(page_root.path(), &rollback);
    let pinned_database_digest = {
        let connection = db.conn.lock().await;
        database_content_digest(&connection).await.unwrap()
    };

    RegenerateFixture {
        db: Arc::new(db),
        _db_dir: db_dir,
        page_root,
        _repair_root: repair_root,
        store,
        manifest,
        rollback,
        before_projection,
        pinned_database_digest,
    }
}

fn projection_bytes(page_root: &Path, rollback: &StoredRollbackArtifact) -> ProjectionBytes {
    let target_path = crate::repair::page_projection_target_path(rollback).unwrap();
    ProjectionBytes {
        target: std::fs::read(page_root.join(target_path)).unwrap(),
        state: std::fs::read(page_root.join(".wenlan/state.json")).ok(),
        unrelated: std::fs::read(page_root.join("unrelated.txt")).unwrap(),
    }
}

async fn call_with_checkpoints<F>(
    fixture: &RegenerateFixture,
    checkpoints: RegeneratePageTestCheckpoints,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    with_regenerate_page_test_checkpoints(
        checkpoints,
        regenerate_page_projection_cas(
            &fixture.db,
            &fixture.manifest,
            &fixture.rollback,
            fixture.page_root.path(),
            before_commit,
        ),
    )
    .await
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
                .execute(
                    "UPDATE pages SET summary='contender entered' WHERE id='page-regenerate'",
                    (),
                )
                .await
                .unwrap();
            drop(guard);
        });
        Self {
            start,
            pending,
            entered,
            task,
        }
    }

    fn release_checkpoint(&self) -> RegeneratePageTestCheckpoint {
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
                    "DB writer contender entered during regenerate".to_string(),
                ));
            }
            Ok(())
        })
    }

    fn assert_not_entered_checkpoint(&self) -> RegeneratePageTestCheckpoint {
        let entered = Arc::clone(&self.entered);
        Box::new(move || {
            if entered.load(Ordering::SeqCst) {
                return Err(WenlanError::Conflict(
                    "DB writer contender entered before regenerate returned".to_string(),
                ));
            }
            Ok(())
        })
    }

    fn assert_not_entered(&self) {
        assert!(!self.entered.load(Ordering::SeqCst));
    }

    async fn assert_enters(self) {
        timeout(Duration::from_secs(2), self.task)
            .await
            .expect("DB writer contender must enter after regenerate returns")
            .unwrap();
        assert!(self.entered.load(Ordering::SeqCst));
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn normal_path_pins_snapshot_and_contender_until_operation_exit_without_self_deadlock() {
    let _serial = REGENERATE_TEST_MUTEX.lock().await;
    let fixture = regenerate_fixture().await;
    let contender = DbWriterContender::prestarted(Arc::clone(&fixture.db));
    let target_write_seen = Arc::new(AtomicBool::new(false));
    let proof_hook_seen = Arc::new(AtomicBool::new(false));
    let target_write_seen_in_hook = Arc::clone(&target_write_seen);
    let proof_hook_seen_in_hook = Arc::clone(&proof_hook_seen);
    let assert_contender_pending = contender.assert_not_entered_checkpoint();
    let proof = timeout(
        Duration::from_secs(3),
        call_with_checkpoints(
            &fixture,
            RegeneratePageTestCheckpoints {
                after_db_snapshot_before_projection_lock: Some(contender.release_checkpoint()),
                after_target_write: Some(Box::new(move || {
                    assert_contender_pending()?;
                    target_write_seen_in_hook.store(true, Ordering::SeqCst);
                    Ok(())
                })),
                ..Default::default()
            },
            |proof| {
                contender.assert_not_entered();
                assert!(target_write_seen.load(Ordering::SeqCst));
                assert_eq!(
                    proof.post_apply_db_digest(),
                    &fixture.pinned_database_digest
                );
                proof_hook_seen_in_hook.store(true, Ordering::SeqCst);
                Ok(())
            },
        ),
    )
    .await
    .expect("normal regenerate must not self-deadlock")
    .unwrap();

    assert_eq!(
        proof.post_apply_db_digest(),
        &fixture.pinned_database_digest
    );
    assert!(proof_hook_seen.load(Ordering::SeqCst));
    assert_ne!(
        projection_bytes(fixture.page_root.path(), &fixture.rollback),
        fixture.before_projection
    );
    contender.assert_enters().await;
    assert_eq!(
        fixture
            .db
            .get_page("page-regenerate")
            .await
            .unwrap()
            .unwrap()
            .summary
            .as_deref(),
        Some("contender entered")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn database_mutex_is_acquired_before_the_projection_lock() {
    let _serial = REGENERATE_TEST_MUTEX.lock().await;
    let fixture = regenerate_fixture().await;
    let lock_path = fixture.page_root.path().join(".wenlan/.projection.lock");
    let file_lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .unwrap();
    file_lock.lock_exclusive().unwrap();
    let observed_db_lock = Arc::new(AtomicBool::new(false));
    let observed_inside = Arc::clone(&observed_db_lock);
    let hook_db = Arc::clone(&fixture.db);

    let result = timeout(
        Duration::from_secs(3),
        call_with_checkpoints(
            &fixture,
            RegeneratePageTestCheckpoints {
                after_db_snapshot_before_projection_lock: Some(Box::new(move || {
                    let attempt = hook_db.conn.try_lock();
                    observed_inside.store(attempt.is_err(), Ordering::SeqCst);
                    drop(attempt);
                    Ok(())
                })),
                ..Default::default()
            },
            |_| Ok(()),
        ),
    )
    .await
    .expect("projection contention must fail fast");

    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "page_projection_locked"
    ));
    assert!(observed_db_lock.load(Ordering::SeqCst));
    file_lock.unlock().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn result_branch_restores_exact_bytes_and_maps_restore_failure_exactly() {
    let _serial = REGENERATE_TEST_MUTEX.lock().await;
    let fixture = regenerate_fixture().await;
    let before_restore = Arc::new(AtomicBool::new(false));
    let after_restore = Arc::new(AtomicBool::new(false));
    let result = call_with_checkpoints(
        &fixture,
        RegeneratePageTestCheckpoints {
            after_target_write: Some(Box::new(|| {
                Err(WenlanError::Conflict(
                    "forced_projection_write_failure".to_string(),
                ))
            })),
            before_projection_restore: Some(Box::new({
                let before_restore = Arc::clone(&before_restore);
                move || {
                    before_restore.store(true, Ordering::SeqCst);
                    Ok(())
                }
            })),
            after_projection_restore: Some(Box::new({
                let after_restore = Arc::clone(&after_restore);
                move || {
                    after_restore.store(true, Ordering::SeqCst);
                    Ok(())
                }
            })),
            ..Default::default()
        },
        |_| Ok(()),
    )
    .await;
    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "forced_projection_write_failure"
    ));
    assert!(before_restore.load(Ordering::SeqCst));
    assert!(after_restore.load(Ordering::SeqCst));
    assert_eq!(
        projection_bytes(fixture.page_root.path(), &fixture.rollback),
        fixture.before_projection
    );

    let failed_restore = regenerate_fixture().await;
    let error = call_with_checkpoints(
        &failed_restore,
        RegeneratePageTestCheckpoints {
            after_target_write: Some(Box::new(|| {
                Err(WenlanError::Conflict(
                    "forced_projection_write_failure".to_string(),
                ))
            })),
            before_projection_restore: Some(Box::new(|| {
                Err(WenlanError::Validation(
                    "forced_projection_restore_failure".to_string(),
                ))
            })),
            ..Default::default()
        },
        |_| Ok(()),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        WenlanError::VectorDb(message)
            if message
                == "Conflict: forced_projection_write_failure; repair projection rollback failed: \
                    Validation error: forced_projection_restore_failure"
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completion_branch_restores_exact_bytes_and_maps_restore_failure_exactly() {
    let _serial = REGENERATE_TEST_MUTEX.lock().await;
    let fixture = regenerate_fixture().await;
    let result = call_with_checkpoints(
        &fixture,
        RegeneratePageTestCheckpoints {
            before_projection_restore: Some(Box::new(|| Ok(()))),
            after_projection_restore: Some(Box::new(|| Ok(()))),
            ..Default::default()
        },
        |_| {
            Err(WenlanError::Conflict(
                "forced_projection_proof_failure".to_string(),
            ))
        },
    )
    .await;
    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "forced_projection_proof_failure"
    ));
    assert_eq!(
        projection_bytes(fixture.page_root.path(), &fixture.rollback),
        fixture.before_projection
    );

    let failed_restore = regenerate_fixture().await;
    let error = call_with_checkpoints(
        &failed_restore,
        RegeneratePageTestCheckpoints {
            before_projection_restore: Some(Box::new(|| {
                Err(WenlanError::Validation(
                    "forced_projection_restore_failure".to_string(),
                ))
            })),
            ..Default::default()
        },
        |_| {
            Err(WenlanError::Conflict(
                "forced_projection_proof_failure".to_string(),
            ))
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        WenlanError::VectorDb(message)
            if message
                == "Conflict: forced_projection_proof_failure; repair projection rollback failed: \
                    Validation error: forced_projection_restore_failure"
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn apply_level_projection_failure_aborts_the_pending_receipt() {
    let _serial = REGENERATE_TEST_MUTEX.lock().await;
    let fixture = regenerate_fixture().await;
    let manifest_dir = fixture
        .store
        .manifest_dir(fixture.manifest.manifest_id())
        .unwrap();
    let result = with_regenerate_page_test_checkpoints(
        RegeneratePageTestCheckpoints {
            after_target_write: Some(Box::new(|| {
                Err(WenlanError::Conflict(
                    "forced_apply_projection_failure".to_string(),
                ))
            })),
            before_projection_restore: Some(Box::new(|| {
                Err(WenlanError::Validation(
                    "forced_apply_projection_restore_failure".to_string(),
                ))
            })),
            ..Default::default()
        },
        crate::repair::apply_repair_with_pages(
            &fixture.db,
            &fixture.store,
            exact_request(&fixture.manifest),
            Some(fixture.page_root.path()),
            1_721_000_001,
        ),
    )
    .await;

    assert!(matches!(
        result,
        Err(WenlanError::VectorDb(message))
            if message
                == "Conflict: forced_apply_projection_failure; repair projection rollback failed: \
                    Validation error: forced_apply_projection_restore_failure"
    ));
    assert!(!manifest_dir.join(".apply-receipt.json.pending").exists());
    assert!(!manifest_dir.join("apply-receipt.json").exists());
    assert_ne!(
        projection_bytes(fixture.page_root.path(), &fixture.rollback),
        fixture.before_projection
    );
}
