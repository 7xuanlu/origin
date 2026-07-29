// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{
    error::WenlanError,
    lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    },
    repair::{
        lint_review_owner_binding_digest, prepare_memory_reclassification, RepairArtifactStore,
    },
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Barrier,
};
use wenlan_types::{
    lint::{LintProfile, LintQuery},
    repair::{
        PrepareRepairRequest, RepairChoice, RepairDigest, RepairLintScope, RepairManifest,
        RepairRollbackPayloadV2, StoredRepairRollbackArtifact,
    },
    MemoryType, RefinementPayload,
};

const ENTITY_OCCURRENCE: &str = "abababababababababababababababababababababababababababababababab";
const ZERO_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";

async fn reclassification_fixture() -> (MemoryDB, tempfile::TempDir, RepairDigest) {
    let (db, db_dir) = crate::db::tests::test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO spaces (id,name,created_at,updated_at)
             VALUES ('space-personal','personal',1,1),('space-work','work',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,space)
             VALUES ('row-target','Target decision','memory','mem-target','target',0,10,
                     'text',0,0,'hide',NULL,'work'),
                    ('row-target-2','Target detail','memory','mem-target','target',1,10,
                     'text',0,0,'hide',NULL,'work'),
                    ('row-other','Other fact','memory','mem-other','other',0,11,
                     'text',0,0,'hide','fact','personal');",
        )
        .await
        .unwrap();
    let receipt = {
        let connection = db.conn.lock().await;
        crate::repair::target_receipt_on_connection(&connection, "mem-target")
            .await
            .unwrap()
            .0
    };
    (db, db_dir, receipt)
}

async fn reclassification_types(db: &MemoryDB) -> Vec<Option<String>> {
    let connection = db.conn.lock().await;
    let mut rows = connection
        .query(
            "SELECT memory_type FROM memories
             WHERE source='memory' AND source_id='mem-target'
             ORDER BY chunk_index,id",
            (),
        )
        .await
        .unwrap();
    let mut memory_types = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        memory_types.push(row.get(0).unwrap());
    }
    memory_types
}

async fn entity_fixture() -> (
    MemoryDB,
    tempfile::TempDir,
    RepairManifest,
    RepairRollbackPayloadV2,
) {
    let (db, db_dir) = crate::db::tests::test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO spaces (id,name,created_at,updated_at)
             VALUES ('space-work','work',1,1),('space-personal','personal',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,space)
             VALUES ('row-entity','Target memory','memory','mem-entity','target',0,10,
                     'text',0,0,'hide','fact','work');
             INSERT INTO entities
                 (id,name,entity_type,space,created_at,updated_at)
             VALUES ('ent-existing','Existing','concept','work',1,1),
                    ('ent-new','New','concept','work',1,1);
             INSERT INTO memory_entities(memory_id,entity_id)
             VALUES ('mem-entity','ent-existing');
             INSERT INTO enrichment_steps
                 (source_id,step_name,status,error,attempts,updated_at)
             VALUES ('mem-entity','entity_extract','failed','transient',2,1721000000);",
        )
        .await
        .unwrap();

    let occurrence = RepairDigest::parse(ENTITY_OCCURRENCE).unwrap();
    let review_id = format!("lint_review_{ENTITY_OCCURRENCE}");
    let source_ids = vec!["mem-entity".to_string()];
    let payload = RefinementPayload::LintRepairReview {
        check_id: "memories.enrichment_failures".to_string(),
        occurrence_digest: occurrence.clone(),
        owner_binding_digest: lint_review_owner_binding_digest(&occurrence, &source_ids).unwrap(),
        issue: "Complete the failed entity extraction.".to_string(),
        choices: vec!["link ent-new".to_string()],
        suggested_research_queries: vec![],
    };
    db.insert_lint_review_if_absent(
        &review_id,
        &source_ids,
        &serde_json::to_string(&payload).unwrap(),
    )
    .await
    .unwrap();

    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let request = PrepareRepairRequest::try_new_with_choice(
        RepairLintScope::global(),
        general,
        None,
        RepairChoice::complete_entity_extraction(
            review_id,
            "mem-entity".to_string(),
            vec!["ent-new".to_string()],
        )
        .unwrap(),
    )
    .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let manifest = prepare_memory_reclassification(&db, &store, request, 1_721_000_001)
        .await
        .unwrap();
    let rollback_bytes = std::fs::read(
        store
            .manifest_dir(manifest.manifest_id())
            .unwrap()
            .join(manifest.rollback().relative_path()),
    )
    .unwrap();
    let StoredRepairRollbackArtifact::V2(stored_rollback) =
        StoredRepairRollbackArtifact::from_slice(&rollback_bytes).unwrap()
    else {
        panic!("entity extraction preparation must write a v2 rollback")
    };
    let rollback = stored_rollback.payload().clone();
    (db, db_dir, manifest, rollback)
}

async fn entity_state(db: &MemoryDB) -> (Vec<String>, (String, Option<String>, i64, i64)) {
    let connection = db.conn.lock().await;
    let mut rows = connection
        .query(
            "SELECT entity_id FROM memory_entities
             WHERE memory_id='mem-entity' ORDER BY entity_id",
            (),
        )
        .await
        .unwrap();
    let mut entity_ids = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        entity_ids.push(row.get::<String>(0).unwrap());
    }
    drop(rows);
    let mut rows = connection
        .query(
            "SELECT status,error,attempts,updated_at FROM enrichment_steps
             WHERE source_id='mem-entity' AND step_name='entity_extract'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    (
        entity_ids,
        (
            row.get::<String>(0).unwrap(),
            row.get::<Option<String>>(1).unwrap(),
            row.get::<i64>(2).unwrap(),
            row.get::<i64>(3).unwrap(),
        ),
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
        .expect("DB mutex must be acquirable after the CAS returns");
    drop(guard);
}

#[tokio::test]
async fn reclassification_stale_target_rolls_back_without_mutation() {
    let (db, _db_dir, _receipt) = reclassification_fixture().await;
    let stale =
        RepairDigest::parse("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            .unwrap();

    let result = db
        .reclassify_memory_repair_cas(
            "mem-target",
            &stale,
            Some("work"),
            None,
            MemoryType::Decision,
            |_| Ok(()),
        )
        .await;

    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "repair_target_stale"
    ));
    assert_eq!(reclassification_types(&db).await, vec![None, None]);
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn reclassification_success_commits_proof_while_mutex_blocks_a_writer() {
    let (db, _db_dir, expected) = reclassification_fixture().await;
    let proof = db
        .reclassify_memory_repair_cas(
            "mem-target",
            &expected,
            Some("work"),
            None,
            MemoryType::Decision,
            |proof| {
                assert_eq!(proof.before_target_receipt(), &expected);
                assert_ne!(proof.after_target_receipt(), proof.before_target_receipt());
                assert_eq!(proof.non_target_before(), proof.non_target_after());
                assert_ne!(proof.post_apply_db_digest().as_str(), ZERO_DIGEST);
                assert_db_mutex_held_during_hook(&db);
                Ok(())
            },
        )
        .await
        .unwrap();

    assert_eq!(proof.before_target_receipt(), &expected);
    assert_eq!(
        reclassification_types(&db).await,
        vec![Some("decision".to_string()), Some("decision".to_string())]
    );
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn reclassification_hook_failure_rolls_back_the_mutation() {
    let (db, _db_dir, expected) = reclassification_fixture().await;
    let result = db
        .reclassify_memory_repair_cas(
            "mem-target",
            &expected,
            Some("work"),
            None,
            MemoryType::Decision,
            |proof| {
                assert_ne!(proof.after_target_receipt(), proof.before_target_receipt());
                Err(WenlanError::Conflict("hook_failure".to_string()))
            },
        )
        .await;

    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "hook_failure"
    ));
    assert_eq!(reclassification_types(&db).await, vec![None, None]);
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn reclassification_sql_failure_rolls_back_the_transaction() {
    let (db, _db_dir, expected) = reclassification_fixture().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "CREATE TRIGGER abort_reclassification
             BEFORE UPDATE OF memory_type ON memories
             WHEN NEW.source_id='mem-target'
             BEGIN SELECT RAISE(ABORT,'forced reclassification failure'); END;",
        )
        .await
        .unwrap();

    let result = db
        .reclassify_memory_repair_cas(
            "mem-target",
            &expected,
            Some("work"),
            None,
            MemoryType::Decision,
            |_| Ok(()),
        )
        .await;

    assert!(matches!(
        result,
        Err(WenlanError::VectorDb(message))
            if message.contains("repair reclassify")
                && message.contains("forced reclassification failure")
    ));
    assert_eq!(reclassification_types(&db).await, vec![None, None]);
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn reclassification_forced_rollback_failure_is_exact_recovery_required() {
    let (db, _db_dir, expected) = reclassification_fixture().await;
    let result = db
        .reclassify_memory_repair_cas_with_forced_rollback_failure(
            "mem-target",
            &expected,
            Some("work"),
            None,
            MemoryType::Decision,
            |_| Err(WenlanError::Conflict("hook_failure".to_string())),
        )
        .await;

    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "repair_apply_recovery_required"
    ));
    db.conn.lock().await.execute("ROLLBACK", ()).await.unwrap();
    assert_eq!(reclassification_types(&db).await, vec![None, None]);
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn entity_extraction_stale_target_rolls_back_without_mutation() {
    let (db, _db_dir, manifest, rollback) = entity_fixture().await;
    db.conn
        .lock()
        .await
        .execute(
            "UPDATE enrichment_steps SET attempts=3
             WHERE source_id='mem-entity' AND step_name='entity_extract'",
            (),
        )
        .await
        .unwrap();
    let before = entity_state(&db).await;

    let result = db
        .complete_entity_extraction_repair_cas(&manifest, &rollback, |_| Ok(()))
        .await;

    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "repair_target_stale"
    ));
    assert_eq!(entity_state(&db).await, before);
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn entity_extraction_success_commits_proof_while_mutex_blocks_a_writer() {
    let (db, _db_dir, manifest, rollback) = entity_fixture().await;
    let proof = db
        .complete_entity_extraction_repair_cas(&manifest, &rollback, |proof| {
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
    assert_eq!(
        entity_state(&db).await,
        (
            vec!["ent-existing".to_string(), "ent-new".to_string()],
            ("ok".to_string(), None, 2, 1_721_000_000),
        )
    );
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn entity_extraction_hook_failure_rolls_back_the_mutation() {
    let (db, _db_dir, manifest, rollback) = entity_fixture().await;
    let before = entity_state(&db).await;

    let result = db
        .complete_entity_extraction_repair_cas(&manifest, &rollback, |proof| {
            assert_ne!(proof.after_target_receipt(), proof.before_target_receipt());
            Err(WenlanError::Conflict("hook_failure".to_string()))
        })
        .await;

    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "hook_failure"
    ));
    assert_eq!(entity_state(&db).await, before);
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn entity_extraction_sql_failure_rolls_back_the_transaction() {
    let (db, _db_dir, manifest, rollback) = entity_fixture().await;
    let before = entity_state(&db).await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "CREATE TRIGGER abort_entity_link
             BEFORE INSERT ON memory_entities
             WHEN NEW.entity_id='ent-new'
             BEGIN SELECT RAISE(ABORT,'forced entity mutation failure'); END;",
        )
        .await
        .unwrap();

    let result = db
        .complete_entity_extraction_repair_cas(&manifest, &rollback, |_| Ok(()))
        .await;

    assert!(matches!(
        result,
        Err(WenlanError::VectorDb(message))
            if message.contains("repair complete entity extraction link")
                && message.contains("forced entity mutation failure")
    ));
    assert_eq!(entity_state(&db).await, before);
    assert_db_mutex_released(&db);
}

#[tokio::test]
async fn entity_extraction_forced_rollback_failure_is_exact_recovery_required() {
    let (db, _db_dir, manifest, rollback) = entity_fixture().await;
    let before = entity_state(&db).await;

    let result = db
        .complete_entity_extraction_repair_cas_with_forced_rollback_failure(
            &manifest,
            &rollback,
            |_| Err(WenlanError::Conflict("hook_failure".to_string())),
        )
        .await;

    assert!(matches!(
        result,
        Err(WenlanError::Conflict(message)) if message == "repair_apply_recovery_required"
    ));
    db.conn.lock().await.execute("ROLLBACK", ()).await.unwrap();
    assert_eq!(entity_state(&db).await, before);
    assert_db_mutex_released(&db);
}
