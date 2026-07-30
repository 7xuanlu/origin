// SPDX-License-Identifier: Apache-2.0

use super::repair_deterministic::apply_deterministic_repair_cas;
use super::MemoryDB;
use crate::{
    error::WenlanError,
    lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    },
    repair::RepairArtifactStore,
};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Barrier,
    },
};
use wenlan_types::{
    lint::{LintProfile, LintQuery},
    repair::{RepairLintScope, RepairManifest, RepairWriter},
    repair_plan::{RepairPlanRequest, RepairResolution},
};

const ZERO_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";

async fn prepared_manifest(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    writer: RepairWriter,
) -> RepairManifest {
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let plan = crate::repair_plan::prepare_repair_plan(
        db,
        store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    plan.entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest } if manifest.writer() == writer => {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing ready manifest for {writer:?}: {plan:#?}"))
}

async fn prepared_page_manifest(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    page_root: &Path,
    writer: RepairWriter,
) -> RepairManifest {
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root),
            true,
        )
        .await
        .unwrap();
    let plan = crate::repair_plan::prepare_repair_plan(
        db,
        store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        Some(page_root),
        1_721_000_000,
    )
    .await
    .unwrap();
    plan.entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest } if manifest.writer() == writer => {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing ready page manifest for {writer:?}: {plan:#?}"))
}

async fn simple_fixture() -> (
    MemoryDB,
    tempfile::TempDir,
    tempfile::TempDir,
    RepairManifest,
) {
    let (db, db_dir) = crate::db::tests::test_db().await;
    db.conn
        .lock()
        .await
        .execute(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  supersedes,pinned,confirmed,stability)
             VALUES ('row-normalize','exact','memory','mem-normalize','exact',0,1,'text',
                     0,0,'hide','fact','   ',NULL,0,0,'new')",
            (),
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_manifest(&db, &store, RepairWriter::NormalizeMemorySourceAgent).await;
    (db, db_dir, repair_root, manifest)
}

async fn source_agent_state(db: &MemoryDB) -> Option<String> {
    db.conn
        .lock()
        .await
        .query(
            "SELECT source_agent FROM memories
             WHERE source='memory' AND source_id='mem-normalize'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

async fn archive_fixture() -> (
    MemoryDB,
    tempfile::TempDir,
    tempfile::TempDir,
    tempfile::TempDir,
    RepairManifest,
) {
    let (db, db_dir) = crate::db::tests::test_db().await;
    db.insert_page_with_kind(
        "source-target",
        "Source target",
        None,
        " \n",
        None,
        Some("work"),
        &[],
        "2026-07-17T00:00:00Z",
        "source",
        "unconfirmed",
        Some("work"),
        None,
    )
    .await
    .unwrap();
    let page_root = tempfile::tempdir().unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepared_page_manifest(
        &db,
        &store,
        page_root.path(),
        RepairWriter::ArchiveEmptySourcePage,
    )
    .await;
    (db, db_dir, page_root, repair_root, manifest)
}

async fn page_route_state(db: &MemoryDB) -> (String, i64, i64) {
    let connection = db.conn.lock().await;
    let mut rows = connection
        .query(
            "SELECT i.space,i.generation,s.generation
               FROM page_community_route_inputs i
               JOIN community_route_space_inputs s ON s.space=i.space
              WHERE i.page_id='source-target'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    (
        row.get::<String>(0).unwrap(),
        row.get::<i64>(1).unwrap(),
        row.get::<i64>(2).unwrap(),
    )
}

fn assert_db_mutex_held_during_hook(db: &MemoryDB) {
    let barrier = Barrier::new(2);
    let blocked = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let attempt = db.conn.try_lock();
            blocked.store(attempt.is_err(), Ordering::SeqCst);
            drop(attempt);
            barrier.wait();
        });
        barrier.wait();
        assert!(
            blocked.load(Ordering::SeqCst),
            "a concurrent writer must be blocked throughout the pre-commit hook"
        );
    });
}

fn assert_db_mutex_released(db: &MemoryDB) {
    let guard = db
        .conn
        .try_lock()
        .expect("DB mutex must be acquirable after the deterministic CAS returns");
    drop(guard);
}

#[tokio::test]
async fn simple_writer_commits_proof_and_holds_mutex_through_hook() {
    let (db, _db_dir, _repair_root, manifest) = simple_fixture().await;
    let proof = apply_deterministic_repair_cas(&db, &manifest, &[], |proof| {
        assert_eq!(
            proof.before_target_receipt(),
            manifest.expected_state().canonical_receipt()
        );
        assert_ne!(proof.after_target_receipt(), proof.before_target_receipt());
        assert_eq!(proof.non_target_before(), proof.non_target_after());
        assert_ne!(proof.post_apply_db_digest().as_str(), ZERO_DIGEST);
        assert_db_mutex_held_during_hook(&db);
        Ok(())
    })
    .await
    .unwrap();

    assert_eq!(
        proof.before_target_receipt(),
        manifest.expected_state().canonical_receipt()
    );
    assert_eq!(source_agent_state(&db).await, None);
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn proof_hook_failure_rolls_back_simple_writer() {
    let (db, _db_dir, _repair_root, manifest) = simple_fixture().await;
    let result = apply_deterministic_repair_cas(&db, &manifest, &[], |proof| {
        assert_ne!(proof.after_target_receipt(), proof.before_target_receipt());
        Err(WenlanError::Conflict("hook_failure".to_string()))
    })
    .await;

    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "hook_failure"
    ));
    assert_eq!(source_agent_state(&db).await.as_deref(), Some("   "));
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn archive_page_commits_exact_route_invalidation_and_proof() {
    let (db, _db_dir, _page_root, _repair_root, manifest) = archive_fixture().await;
    let route_before = page_route_state(&db).await;

    let proof = apply_deterministic_repair_cas(&db, &manifest, &[], |_| Ok(()))
        .await
        .unwrap();

    let page = db.get_page("source-target").await.unwrap().unwrap();
    let route_after = page_route_state(&db).await;
    assert_eq!(page.status, "archived");
    assert_eq!(route_after.0, route_before.0);
    assert_eq!(route_after.1, route_before.1.saturating_add(1));
    assert_eq!(route_after.2, route_before.2);
    assert_eq!(proof.non_target_before(), proof.non_target_after());
    assert_ne!(proof.after_target_receipt(), proof.before_target_receipt());
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn archive_page_without_invalidation_trigger_fails_exact_and_rolls_back_active() {
    let (db, _db_dir, _page_root, _repair_root, manifest) = archive_fixture().await;
    let route_before = page_route_state(&db).await;
    db.conn
        .lock()
        .await
        .execute("DROP TRIGGER m4_page_community_page_invalidate", ())
        .await
        .unwrap();

    let result = apply_deterministic_repair_cas(&db, &manifest, &[], |_| Ok(())).await;

    assert!(matches!(
        result,
        Err(WenlanError::VectorDb(message))
            if message == "repair target page route invalidation unproven"
    ));
    assert_eq!(
        db.get_page("source-target").await.unwrap().unwrap().status,
        "active"
    );
    assert_eq!(page_route_state(&db).await, route_before);
    assert_db_mutex_released(&db);
}
