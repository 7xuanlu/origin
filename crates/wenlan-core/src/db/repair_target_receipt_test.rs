// SPDX-License-Identifier: Apache-2.0

use super::{
    repair_target_receipt::{
        read_current_repair_target_receipt, with_repair_target_receipt_test_checkpoints,
        RepairTargetReceiptTestCheckpoint, RepairTargetReceiptTestCheckpoints,
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
    repair::{RepairArtifactStore, StoredRollbackArtifact},
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
    repair::{RepairLintScope, RepairManifest, RepairWriter},
    repair_plan::{RepairPlanRequest, RepairResolution},
};

static REPAIR_TARGET_RECEIPT_TEST_MUTEX: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct ReceiptFixture {
    db: Arc<MemoryDB>,
    _db_dir: tempfile::TempDir,
    page_root: Option<tempfile::TempDir>,
    _repair_root: tempfile::TempDir,
    manifest: RepairManifest,
    rollback: StoredRollbackArtifact,
}

fn load_rollback(store: &RepairArtifactStore, manifest: &RepairManifest) -> StoredRollbackArtifact {
    let path = store
        .manifest_dir(manifest.manifest_id())
        .unwrap()
        .join(manifest.rollback().relative_path());
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

async fn default_fixture() -> ReceiptFixture {
    let (db, db_dir) = super::tests::test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('row-default','default','memory','mem-default','default',0,1,'text',
                     0,0,'hide','fact','   ',0,0,'new');",
        )
        .await
        .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            None,
            false,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    let manifest = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::NormalizeMemorySourceAgent =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .expect("normalize source-agent manifest");
    let rollback = load_rollback(&store, &manifest);
    ReceiptFixture {
        db: Arc::new(db),
        _db_dir: db_dir,
        page_root: None,
        _repair_root: repair_root,
        manifest,
        rollback,
    }
}

async fn ordinary_projection_fixture() -> ReceiptFixture {
    let (db, db_dir) = super::tests::test_db().await;
    db.insert_page(
        "page-current-receipt",
        "Current Receipt",
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
    let page = db.get_page("page-current-receipt").await.unwrap().unwrap();
    KnowledgeProjectionWrite::new(page_root.path().to_path_buf(), &db)
        .write_page(&page)
        .unwrap();
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE pages SET version=version+1 WHERE id='page-current-receipt'",
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
        .expect("regenerate projection manifest");
    let rollback = load_rollback(&store, &manifest);
    ReceiptFixture {
        db: Arc::new(db),
        _db_dir: db_dir,
        page_root: Some(page_root),
        _repair_root: repair_root,
        manifest,
        rollback,
    }
}

async fn stale_projection_fixture(page_id: &str) -> ReceiptFixture {
    let (db, db_dir) = super::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    let source_path = format!("{page_id}.md");
    std::fs::write(
        page_root.path().join(".wenlan/state.json"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 2,
            "pages": {
                page_id: {
                    "file": source_path,
                    "version": 1
                }
            }
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        page_root.path().join(&source_path),
        format!("---\norigin_id: {page_id}\norigin_version: 1\n---\nstale\n"),
    )
    .unwrap();
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
        .expect("stale projection manifest");
    let rollback = load_rollback(&store, &manifest);
    ReceiptFixture {
        db: Arc::new(db),
        _db_dir: db_dir,
        page_root: Some(page_root),
        _repair_root: repair_root,
        manifest,
        rollback,
    }
}

fn malformed_rollback(rollback: &StoredRollbackArtifact) -> StoredRollbackArtifact {
    let mut malformed = rollback.clone();
    malformed.columns.clear();
    malformed
}

fn unsafe_rollback(rollback: &StoredRollbackArtifact) -> StoredRollbackArtifact {
    let mut unsafe_rollback = rollback.clone();
    let row = unsafe_rollback
        .rows
        .iter_mut()
        .find(|row| row.first().is_some_and(|path| path != "@page_row"))
        .expect("projection path row");
    row[0] = "../escape".to_string();
    unsafe_rollback
}

async fn call_with_checkpoints(
    fixture: &ReceiptFixture,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
    checkpoints: RepairTargetReceiptTestCheckpoints,
) -> Result<(wenlan_types::repair::RepairDigest, u64), WenlanError> {
    with_repair_target_receipt_test_checkpoints(
        checkpoints,
        read_current_repair_target_receipt(&fixture.db, &fixture.manifest, rollback, page_root),
    )
    .await
}

struct DbContender {
    start: Arc<Notify>,
    pending: Arc<AtomicBool>,
    entered: Arc<AtomicBool>,
    task: JoinHandle<()>,
}

impl DbContender {
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
            drop(guard);
        });
        Self {
            start,
            pending,
            entered,
            task,
        }
    }

    fn release_and_require_pending(&self) -> RepairTargetReceiptTestCheckpoint {
        let start = Arc::clone(&self.start);
        let pending = Arc::clone(&self.pending);
        let entered = Arc::clone(&self.entered);
        Box::new(move || {
            start.notify_one();
            let deadline = Instant::now() + Duration::from_secs(2);
            while !pending.load(Ordering::SeqCst) {
                if Instant::now() >= deadline {
                    return Err(WenlanError::Conflict(
                        "DB contender did not observe Pending".to_string(),
                    ));
                }
                std::thread::yield_now();
            }
            if entered.load(Ordering::SeqCst) {
                return Err(WenlanError::Conflict(
                    "DB contender entered before capture".to_string(),
                ));
            }
            Ok(())
        })
    }

    fn require_pending(&self) -> RepairTargetReceiptTestCheckpoint {
        let pending = Arc::clone(&self.pending);
        let entered = Arc::clone(&self.entered);
        Box::new(move || {
            if !pending.load(Ordering::SeqCst) || entered.load(Ordering::SeqCst) {
                return Err(WenlanError::Conflict(
                    "DB contender was not pending across capture".to_string(),
                ));
            }
            Ok(())
        })
    }

    async fn assert_enters(self) {
        timeout(Duration::from_secs(2), self.task)
            .await
            .expect("DB contender must enter after dispatcher return")
            .unwrap();
        assert!(self.entered.load(Ordering::SeqCst));
    }
}

#[tokio::test]
async fn default_branch_matches_r4_17_ignores_projection_inputs_and_does_not_double_lock() {
    let fixture = default_fixture().await;
    let expected = fixture
        .db
        .read_repair_target_receipt(fixture.manifest.target())
        .await
        .unwrap();
    let bogus = malformed_rollback(&fixture.rollback);
    let actual = timeout(
        Duration::from_secs(2),
        read_current_repair_target_receipt(
            &fixture.db,
            &fixture.manifest,
            &bogus,
            Some(Path::new("/definitely/not/a/page/root")),
        ),
    )
    .await
    .expect("default branch must not double-lock")
    .unwrap();
    assert_eq!(actual, expected);

    fixture
        .db
        .conn
        .lock()
        .await
        .execute(
            "UPDATE memories SET space='personal' WHERE source_id='mem-default'",
            (),
        )
        .await
        .unwrap();
    assert!(matches!(
        read_current_repair_target_receipt(
            &fixture.db,
            &fixture.manifest,
            &bogus,
            Some(Path::new("/still/ignored")),
        )
        .await,
        Err(WenlanError::Conflict(message)) if message == "repair_target_stale"
    ));
}

#[tokio::test]
async fn ordinary_projection_digest_count_and_errors_are_exact() {
    let _serial = REPAIR_TARGET_RECEIPT_TEST_MUTEX.lock().await;
    let fixture = ordinary_projection_fixture().await;
    let root = fixture.page_root.as_ref().unwrap().path();
    std::fs::write(
        root.join(crate::repair::page_projection_target_path(&fixture.rollback).unwrap()),
        b"dynamic projection bytes",
    )
    .unwrap();
    let actual = read_current_repair_target_receipt(
        &fixture.db,
        &fixture.manifest,
        &fixture.rollback,
        Some(root),
    )
    .await
    .unwrap();
    let expected = {
        let connection = fixture.db.conn.lock().await;
        let paths = crate::repair::projection_rollback_paths(&fixture.rollback).unwrap();
        let current = crate::repair::capture_page_projection_on_connection(
            &connection,
            root,
            "page-current-receipt",
            &paths,
            &fixture.rollback.table,
        )
        .await
        .unwrap();
        (crate::repair::target_receipt(&current).unwrap(), 1)
    };
    assert_eq!(actual, expected);

    assert!(matches!(
        read_current_repair_target_receipt(
            &fixture.db,
            &fixture.manifest,
            &fixture.rollback,
            None,
        )
        .await,
        Err(WenlanError::Validation(message))
            if message == "page projection repair root unavailable"
    ));
    assert!(matches!(
        read_current_repair_target_receipt(
            &fixture.db,
            &fixture.manifest,
            &malformed_rollback(&fixture.rollback),
            Some(root),
        )
        .await,
        Err(WenlanError::Validation(message))
            if message == "repair_projection_rollback_invalid"
    ));
    assert!(matches!(
        read_current_repair_target_receipt(
            &fixture.db,
            &fixture.manifest,
            &unsafe_rollback(&fixture.rollback),
            Some(root),
        )
        .await,
        Err(WenlanError::Validation(message))
            if message == "repair_projection_rollback_unsafe_path"
    ));
    fixture
        .db
        .conn
        .lock()
        .await
        .execute("DELETE FROM pages WHERE id='page-current-receipt'", ())
        .await
        .unwrap();
    assert!(matches!(
        read_current_repair_target_receipt(
            &fixture.db,
            &fixture.manifest,
            &fixture.rollback,
            Some(root),
        )
        .await,
        Err(WenlanError::Conflict(message)) if message == "repair_target_stale"
    ));
}

#[tokio::test]
async fn projection_validation_runs_with_database_mutex_held() {
    let _serial = REPAIR_TARGET_RECEIPT_TEST_MUTEX.lock().await;
    let ordinary = ordinary_projection_fixture().await;
    let ordinary_db = Arc::clone(&ordinary.db);
    let ordinary_error = call_with_checkpoints(
        &ordinary,
        &malformed_rollback(&ordinary.rollback),
        ordinary.page_root.as_ref().map(|root| root.path()),
        RepairTargetReceiptTestCheckpoints {
            after_projection_db_lock_before_validation: Some(Box::new(move || {
                assert!(ordinary_db.conn.try_lock().is_err());
                Ok(())
            })),
            ..Default::default()
        },
    )
    .await;
    assert!(matches!(
        ordinary_error,
        Err(WenlanError::Validation(message))
            if message == "repair_projection_rollback_invalid"
    ));
    let ordinary_db = Arc::clone(&ordinary.db);
    let ordinary_missing_root = call_with_checkpoints(
        &ordinary,
        &ordinary.rollback,
        None,
        RepairTargetReceiptTestCheckpoints {
            after_projection_db_lock_before_validation: Some(Box::new(move || {
                assert!(ordinary_db.conn.try_lock().is_err());
                Ok(())
            })),
            ..Default::default()
        },
    )
    .await;
    assert!(matches!(
        ordinary_missing_root,
        Err(WenlanError::Validation(message))
            if message == "page projection repair root unavailable"
    ));

    let stale = stale_projection_fixture("stale-validation-order").await;
    let stale_db = Arc::clone(&stale.db);
    let stale_error = call_with_checkpoints(
        &stale,
        &malformed_rollback(&stale.rollback),
        stale.page_root.as_ref().map(|root| root.path()),
        RepairTargetReceiptTestCheckpoints {
            after_projection_db_lock_before_validation: Some(Box::new(move || {
                assert!(stale_db.conn.try_lock().is_err());
                Ok(())
            })),
            ..Default::default()
        },
    )
    .await;
    assert!(matches!(
        stale_error,
        Err(WenlanError::Validation(message))
            if message == "repair_projection_rollback_invalid"
    ));
    let stale_db = Arc::clone(&stale.db);
    let stale_missing_root = call_with_checkpoints(
        &stale,
        &stale.rollback,
        None,
        RepairTargetReceiptTestCheckpoints {
            after_projection_db_lock_before_validation: Some(Box::new(move || {
                assert!(stale_db.conn.try_lock().is_err());
                Ok(())
            })),
            ..Default::default()
        },
    )
    .await;
    assert!(matches!(
        stale_missing_root,
        Err(WenlanError::Validation(message))
            if message == "page projection repair root unavailable"
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ordinary_projection_holds_database_mutex_across_page_and_filesystem_capture() {
    let _serial = REPAIR_TARGET_RECEIPT_TEST_MUTEX.lock().await;
    let fixture = ordinary_projection_fixture().await;
    let contender = DbContender::prestarted(Arc::clone(&fixture.db));
    let result = timeout(
        Duration::from_secs(3),
        call_with_checkpoints(
            &fixture,
            &fixture.rollback,
            fixture.page_root.as_ref().map(|root| root.path()),
            RepairTargetReceiptTestCheckpoints {
                after_projection_db_lock_before_validation: Some(
                    contender.release_and_require_pending(),
                ),
                before_capture: Some(contender.require_pending()),
                after_capture_before_return: Some(contender.require_pending()),
            },
        ),
    )
    .await
    .expect("ordinary receipt read must not self-deadlock")
    .unwrap();
    assert_eq!(result.1, 1);
    contender.assert_enters().await;
}

#[tokio::test]
async fn stale_projection_digest_count_and_errors_are_exact() {
    let _serial = REPAIR_TARGET_RECEIPT_TEST_MUTEX.lock().await;
    let fixture = stale_projection_fixture("stale-current-receipt").await;
    let root = fixture.page_root.as_ref().unwrap().path();
    std::fs::write(
        root.join("stale-current-receipt.md"),
        b"dynamic stale bytes",
    )
    .unwrap();
    let actual = read_current_repair_target_receipt(
        &fixture.db,
        &fixture.manifest,
        &fixture.rollback,
        Some(root),
    )
    .await
    .unwrap();
    let (source_path, quarantine_path) =
        crate::repair::stale_page_projection_paths(&fixture.rollback).unwrap();
    let current = crate::repair::capture_stale_page_projection_current(
        root,
        "stale-current-receipt",
        &source_path,
        &quarantine_path,
    )
    .unwrap();
    assert_eq!(
        actual,
        (crate::repair::target_receipt(&current).unwrap(), 0)
    );

    assert!(matches!(
        read_current_repair_target_receipt(
            &fixture.db,
            &fixture.manifest,
            &fixture.rollback,
            None,
        )
        .await,
        Err(WenlanError::Validation(message))
            if message == "page projection repair root unavailable"
    ));
    assert!(matches!(
        read_current_repair_target_receipt(
            &fixture.db,
            &fixture.manifest,
            &malformed_rollback(&fixture.rollback),
            Some(root),
        )
        .await,
        Err(WenlanError::Validation(message))
            if message == "repair_projection_rollback_invalid"
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_projection_holds_database_mutex_across_owner_check_and_projection_capture() {
    let _serial = REPAIR_TARGET_RECEIPT_TEST_MUTEX.lock().await;
    let fixture = stale_projection_fixture("stale-contender").await;
    let contender = DbContender::prestarted(Arc::clone(&fixture.db));
    let result = timeout(
        Duration::from_secs(3),
        call_with_checkpoints(
            &fixture,
            &fixture.rollback,
            fixture.page_root.as_ref().map(|root| root.path()),
            RepairTargetReceiptTestCheckpoints {
                after_projection_db_lock_before_validation: Some(
                    contender.release_and_require_pending(),
                ),
                before_capture: Some(contender.require_pending()),
                after_capture_before_return: Some(contender.require_pending()),
            },
        ),
    )
    .await
    .expect("stale receipt read must not self-deadlock")
    .unwrap();
    assert_eq!(result.1, 0);
    contender.assert_enters().await;
}

#[tokio::test]
async fn projection_file_lock_priority_is_branch_exact() {
    let _serial = REPAIR_TARGET_RECEIPT_TEST_MUTEX.lock().await;
    let ordinary = ordinary_projection_fixture().await;
    let ordinary_root = ordinary.page_root.as_ref().unwrap().path();
    let ordinary_lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(ordinary_root.join(".wenlan/.projection.lock"))
        .unwrap();
    ordinary_lock.lock_exclusive().unwrap();
    assert_eq!(
        read_current_repair_target_receipt(
            &ordinary.db,
            &ordinary.manifest,
            &ordinary.rollback,
            Some(ordinary_root),
        )
        .await
        .unwrap()
        .1,
        1
    );

    let stale = stale_projection_fixture("stale-lock-priority").await;
    let stale_root = stale.page_root.as_ref().unwrap().path();
    let stale_lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(stale_root.join(".wenlan/.projection.lock"))
        .unwrap();
    stale_lock.lock_exclusive().unwrap();
    stale
        .db
        .insert_page(
            "stale-lock-priority",
            "Owner",
            None,
            "owner",
            None,
            None,
            &[],
            "2026-07-17T00:00:00Z",
        )
        .await
        .unwrap();
    assert!(matches!(
        read_current_repair_target_receipt(
            &stale.db,
            &stale.manifest,
            &stale.rollback,
            Some(stale_root),
        )
        .await,
        Err(WenlanError::Conflict(message)) if message == "repair_target_stale"
    ));
    stale
        .db
        .conn
        .lock()
        .await
        .execute("DELETE FROM pages WHERE id='stale-lock-priority'", ())
        .await
        .unwrap();
    assert!(matches!(
        read_current_repair_target_receipt(
            &stale.db,
            &stale.manifest,
            &stale.rollback,
            Some(stale_root),
        )
        .await,
        Err(WenlanError::Conflict(message)) if message == "page_projection_locked"
    ));
}

#[test]
fn r4_23_source_contract_is_bounded_and_branch_exact() {
    let child = include_str!("repair_target_receipt.rs");
    let repair = include_str!("../repair.rs");
    let declaration = "pub(crate) async fn read_current_repair_target_receipt(";
    assert_eq!(child.matches(declaration).count(), 1);
    assert!(!repair.contains("async fn target_receipt_current("));
    assert_eq!(
        repair
            .matches("read_current_repair_target_receipt(")
            .count(),
        2
    );
    for forbidden in [
        "with_repair_lock",
        "begin_owned_repair_session",
        "RepairArtifactStore",
        "publish_no_replace",
        "remove_file",
        "clear_pending",
        "quarantine_stale_page_projection_cas",
        "recover_stale_page_projection_apply_receipt",
    ] {
        assert!(
            !child.contains(forbidden),
            "dispatcher gained forbidden capability {forbidden}"
        );
    }
    assert!(!child.contains("drop(connection)"));

    let dispatcher_start = child.find(declaration).expect("dispatcher declaration");
    let ordinary_start = child
        .find("async fn read_projection_target_receipt(")
        .expect("ordinary projection branch");
    let stale_start = child
        .find("async fn read_stale_projection_target_receipt(")
        .expect("stale projection branch");
    assert!(dispatcher_start < ordinary_start && ordinary_start < stale_start);
    let dispatcher = &child[dispatcher_start..ordinary_start];
    let ordinary = &child[ordinary_start..stale_start];
    let stale = &child[stale_start..];
    assert!(dispatcher.contains("db.read_repair_target_receipt(manifest.target()).await"));
    for forbidden in [
        ".conn.lock().await",
        "page_root.ok_or_else",
        "projection_rollback_paths(",
        "stale_page_projection_paths(",
        "capture_page_projection_on_connection(",
        "capture_stale_page_projection_current(",
    ] {
        assert!(
            !dispatcher.contains(forbidden),
            "dispatcher must branch before branch-local primitive {forbidden}"
        );
    }

    let ordered = |segment: &str, labels: &[&str]| {
        let mut previous = 0;
        for label in labels {
            let position = segment
                .find(label)
                .unwrap_or_else(|| panic!("branch lost ordered primitive {label}"));
            assert!(
                position >= previous,
                "branch reordered {label}: {position} before {previous}"
            );
            previous = position;
        }
    };
    assert_eq!(ordinary.matches(".conn.lock().await").count(), 1);
    ordered(
        ordinary,
        &[
            ".conn.lock().await",
            "AfterProjectionDbLockBeforeValidation",
            "page_root.ok_or_else",
            "projection_rollback_paths(rollback)",
            "BeforeCapture",
            "capture_page_projection_on_connection(",
            "AfterCaptureBeforeReturn",
            "target_receipt(&current)",
            "?, 1",
        ],
    );
    for forbidden in [
        "with_projection_lock",
        "with_repair_lock",
        "begin_owned_repair_session",
    ] {
        assert!(
            !ordinary.contains(forbidden),
            "ordinary projection gained lock/session {forbidden}"
        );
    }

    assert_eq!(stale.matches(".conn.lock().await").count(), 1);
    ordered(
        stale,
        &[
            ".conn.lock().await",
            "AfterProjectionDbLockBeforeValidation",
            "page_root.ok_or_else",
            "SELECT 1 FROM pages WHERE id=?1 LIMIT 1",
            "owner.next().await.map_err(database_error)?",
            "repair_target_stale",
            "drop(owner)",
            "stale_page_projection_paths(rollback)",
            "BeforeCapture",
            "capture_stale_page_projection_current(",
            "AfterCaptureBeforeReturn",
            "target_receipt(&current)",
            "?, 0",
        ],
    );
    assert_eq!(
        child
            .matches(
                "run_test_checkpoint(TestCheckpointKind::AfterProjectionDbLockBeforeValidation)",
            )
            .count(),
        2
    );
    assert_eq!(
        child
            .matches("run_test_checkpoint(TestCheckpointKind::BeforeCapture)")
            .count(),
        2
    );
    assert_eq!(
        child
            .matches("run_test_checkpoint(TestCheckpointKind::AfterCaptureBeforeReturn)")
            .count(),
        2
    );
    let recovery = repair
        .find("async fn recover_apply_receipt(")
        .expect("generic recovery declaration");
    let recovery = &repair[recovery..];
    let specialized = recovery
        .find("return recover_stale_page_projection_apply_receipt(")
        .expect("specialized stale recovery dispatch");
    let first_child = recovery
        .find("read_current_repair_target_receipt(")
        .expect("first child call");
    assert!(
        specialized < first_child,
        "specialized stale recovery must remain before generic child calls"
    );
}
