use super::*;
use crate::db::tests::test_db;
use crate::derived_artifact_state::DerivedArtifact;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use wenlan_types::lint::{LintMetricCode, LintMetricValue, LintOutcome, LintQuery};

async fn insert_head(db: &crate::db::MemoryDB, id: &str) {
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO memories (
            id, content, source, source_id, title, chunk_index, last_modified,
            chunk_type, stability, supersede_mode, needs_reembed, memory_type, word_count
         ) VALUES (?1, 'one two three four five six seven eight', 'memory', ?1,
                   ?1, 0, 1, 'text', 'new', 'hide', 1, 'fact', 8)",
        libsql::params![id],
    )
    .await
    .unwrap();
}

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

fn check<'a>(
    report: &'a wenlan_types::lint::LintReport,
    id: &str,
) -> &'a wenlan_types::lint::LintCheckResult {
    report
        .checks()
        .iter()
        .find(|result| result.check_id() == id)
        .unwrap()
}

#[tokio::test]
async fn durable_sweeps_drive_runner_readiness_and_active_backfill_suppresses_findings() {
    let (db, _tmp) = test_db().await;
    insert_head(&db, "mem-overdue").await;
    let features = TestMemoryFeatures {
        episode: true,
        ..TestMemoryFeatures::default()
    };

    db.record_derived_artifact_sweep_at(1_000_000)
        .await
        .unwrap();
    assert_eq!(
        check(&run_at(&db, 1_000_000, features).await, EPISODE_ID).outcome(),
        LintOutcome::Pass
    );
    db.record_derived_artifact_sweep_at(1_001_800)
        .await
        .unwrap();
    assert_eq!(
        check(&run_at(&db, 1_001_800, features).await, EPISODE_ID).outcome(),
        LintOutcome::Pass
    );
    db.record_derived_artifact_sweep_at(1_003_600)
        .await
        .unwrap();

    let guard = db.begin_derived_artifact_write(DerivedArtifact::Episode);
    assert_eq!(
        check(&run_at(&db, 1_003_600, features).await, EPISODE_ID).outcome(),
        LintOutcome::Pass
    );
    drop(guard);
    assert_eq!(
        check(&run_at(&db, 1_003_600, features).await, EPISODE_ID).outcome(),
        LintOutcome::Finding
    );
}

#[tokio::test]
async fn fact_liveness_differentiates_rows_from_real_vector_index_visibility() {
    let (db, _tmp) = test_db().await;
    insert_head(&db, "mem-indexed").await;
    insert_head(&db, "mem-null-vector").await;
    let vector = format!(
        "[{}]",
        std::iter::once("1")
            .chain(std::iter::repeat_n("0", 767))
            .collect::<Vec<_>>()
            .join(",")
    );
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO child_vectors (id, parent_kind, parent_id, field, content, embedding)
         VALUES ('child-indexed', 'memory', 'mem-indexed', 'narrative', 'body', vector32(?1)),
                ('child-null', 'memory', 'mem-null-vector', 'narrative', 'body', NULL)",
        libsql::params![vector],
    )
    .await
    .unwrap();
    let mut rows = conn
        .query("SELECT COUNT(*) FROM child_vectors", ())
        .await
        .unwrap();
    assert_eq!(
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        2
    );
    drop(rows);
    drop(conn);

    let report = run_at(
        &db,
        1_000_000,
        TestMemoryFeatures {
            fact: true,
            ..TestMemoryFeatures::default()
        },
    )
    .await;
    let fact = check(&report, FACT_ID);
    assert_eq!(fact.outcome(), LintOutcome::Pass);
    assert!(fact.metrics().iter().any(|metric| {
        metric.code() == LintMetricCode::AffectedRecords
            && metric.value() == &LintMetricValue::Count { value: 1 }
    }));
}
