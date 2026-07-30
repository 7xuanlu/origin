use super::*;
use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::{LintRunner, TestSyncPoint, TestSynchronization};
use std::sync::Arc;
use wenlan_types::lint::{LintOutcome, LintQuery};
use wenlan_types::RawDocument;

async fn run_at(
    db: &crate::db::MemoryDB,
    epoch_seconds: i64,
    features: TestMemoryFeatures,
) -> wenlan_types::lint::LintReport {
    LintRunner::new(LintClock::fixed_at(epoch_seconds), CancellationToken::new())
        .with_test_memory_features(features)
        .run(
            db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap()
}

fn outcome(report: &wenlan_types::lint::LintReport, id: &str) -> LintOutcome {
    report
        .checks()
        .iter()
        .find(|check| check.check_id() == id)
        .unwrap()
        .outcome()
}

#[tokio::test]
async fn source_text_controls_episode_eligibility_in_sweep_and_runner() {
    let (db, _tmp) = test_db().await;
    db.test_primary_session()
        .await
        .execute(
            "INSERT INTO memories (
                id, content, source, source_id, title, chunk_index, last_modified,
                chunk_type, stability, supersede_mode, needs_reembed, memory_type,
                word_count, source_text
             ) VALUES ('source-text-head', 'short', 'memory', 'source-text-head',
                       'source-text-head', 0, 1, 'text', 'new', 'hide', 1, 'fact',
                       1, 'one two three four five six seven eight')",
            (),
        )
        .await
        .unwrap();

    for observed_at in [1_000_000, 1_001_800, 1_003_600] {
        db.record_derived_artifact_sweep_at(observed_at)
            .await
            .unwrap();
    }

    let report = run_at(
        &db,
        1_003_600,
        TestMemoryFeatures {
            episode: true,
            ..TestMemoryFeatures::default()
        },
    )
    .await;
    assert_eq!(outcome(&report, EPISODE_ID), LintOutcome::Finding);
}

#[tokio::test]
async fn archive_predecessor_stays_head_while_evicted_memory_is_excluded() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO memories (
            id, content, source, source_id, title, chunk_index, last_modified,
            chunk_type, stability, supersede_mode, supersedes, needs_reembed,
            memory_type, word_count
         ) VALUES
            ('archive-old', 'old', 'memory', 'archive-old', 'old', 0, 1,
             'text', 'new', 'hide', NULL, 1, 'fact', 1),
            ('archive-new', 'new', 'memory', 'archive-new', 'new', 0, 2,
             'text', 'new', 'archive', 'archive-old', 1, 'fact', 1),
            ('evicted', 'gone', 'memory', 'evicted', 'evicted', 0, 3,
             'text', 'new', 'evicted', NULL, 1, 'fact', 1)",
        (),
    )
    .await
    .unwrap();
    drop(conn);

    let report = run_at(&db, 1_000_000, TestMemoryFeatures::default()).await;
    let embedding = report
        .checks()
        .iter()
        .find(|check| check.check_id() == EMBEDDING_ID)
        .unwrap();
    assert_eq!(embedding.coverage().denominator(), 2);
}

#[tokio::test]
async fn pending_reembed_on_one_chunk_does_not_mask_an_unowned_missing_embedding() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO memories (
            id, content, source, source_id, title, chunk_index, last_modified,
            chunk_type, stability, supersede_mode, needs_reembed, memory_type, word_count
         ) VALUES
            ('embed-0', 'first', 'memory', 'mixed-embedding', 'mixed', 0, 1,
             'text', 'new', 'hide', 1, 'fact', 1),
            ('embed-1', 'second', 'memory', 'mixed-embedding', 'mixed', 1, 1,
             'text', 'new', 'hide', 0, 'fact', 1)",
        (),
    )
    .await
    .unwrap();
    drop(conn);

    let report = run_at(&db, 1_000_000, TestMemoryFeatures::default()).await;
    assert_eq!(outcome(&report, EMBEDDING_ID), LintOutcome::Finding);
}

#[tokio::test]
async fn canonical_episode_cowrite_between_samples_marks_memory_checks_incomplete() {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);
    let (synchronization, control) = TestSynchronization::new(TestSyncPoint::AfterTrackerSample);
    let runner_db = Arc::clone(&db);
    let task = tokio::spawn(async move {
        LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .with_test_memory_features(TestMemoryFeatures {
                episode: true,
                ..TestMemoryFeatures::default()
            })
            .with_test_synchronization(synchronization)
            .run(
                &runner_db,
                &LintQuery {
                    profile: None,
                    space: None,
                },
                None,
                false,
            )
            .await
    });
    control.wait_until_reached().await;
    db.upsert_documents_with_derived_channels_for_test(
        vec![RawDocument {
            source: "memory".to_string(),
            source_id: "cowrite-race".to_string(),
            title: "cowrite-race".to_string(),
            content: "distilled".to_string(),
            source_text: Some("one two three four five six seven eight".to_string()),
            last_modified: 1,
            stability: Some("new".to_string()),
            memory_type: Some("fact".to_string()),
            ..RawDocument::default()
        }],
        true,
        false,
    )
    .await
    .unwrap();
    control.resume().await;

    let report = task.await.unwrap().unwrap();
    assert!(!report.complete());
    assert_eq!(
        outcome(&report, EPISODE_ID),
        LintOutcome::InconsistentSnapshot
    );
}

#[tokio::test]
async fn canonical_fact_cowrite_advances_the_shared_producer_generation() {
    let (db, _tmp) = test_db().await;
    let before = db.derived_artifact_sample();
    db.upsert_documents_with_derived_channels_for_test(
        vec![RawDocument {
            source: "memory".to_string(),
            source_id: "fact-cowrite".to_string(),
            title: "fact-cowrite".to_string(),
            content: "a fact with enough content for a narrative child".to_string(),
            last_modified: 1,
            stability: Some("new".to_string()),
            memory_type: Some("fact".to_string()),
            ..RawDocument::default()
        }],
        false,
        true,
    )
    .await
    .unwrap();
    assert_ne!(before, db.derived_artifact_sample());
}
