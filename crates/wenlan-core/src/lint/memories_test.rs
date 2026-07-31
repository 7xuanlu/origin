use super::*;
use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use crate::lint::test_support::DbSemanticFingerprint;
use wenlan_types::lint::{LintApplicability, LintMetricCode, LintMetricValue, LintQuery};

fn record(id: &str) -> MemoryRecord {
    MemoryRecord {
        source_id: id.to_string(),
        lifecycle_valid: true,
        supersedes: None,
        target_exists: true,
        replaced_by_active: false,
        pending_revision: false,
        recap: false,
        evicted: false,
        embedding_valid: true,
        needs_reembed: false,
        failed_steps: 0,
        classified: true,
        event_dated: false,
        episode: false,
        fact: false,
        page_link: false,
        summary: false,
        episode_eligible: true,
        fact_eligible: true,
        summary_eligible: true,
        episode_receipt: None,
        fact_receipt: None,
        summary_receipt: None,
        head: true,
    }
}

#[test]
fn configured_off_derived_substrate_is_expected_empty() {
    let row = record("mem-a");
    let result = derived(
        EPISODE_ID,
        &[&row],
        false,
        |_| true,
        |r| r.episode,
        |_| None,
    );
    assert_eq!(result.applicability, LintApplicability::ExpectedEmpty);
    assert_eq!(result.precondition, LintPrecondition::ConfiguredOff);
    assert!(result
        .levels
        .iter()
        .all(|level| *level == LintSeverity::Info));
}

#[test]
fn missing_derived_artifact_needs_two_sweeps_and_sixty_minutes() {
    let row = record("mem-a");
    for readiness in [
        DerivedReadiness {
            provider_ready: true,
            active_backfill: true,
            completed_sweeps: 2,
            missing_age_seconds: 7_200,
        },
        DerivedReadiness {
            provider_ready: true,
            active_backfill: false,
            completed_sweeps: 1,
            missing_age_seconds: 7_200,
        },
        DerivedReadiness {
            provider_ready: true,
            active_backfill: false,
            completed_sweeps: 2,
            missing_age_seconds: 3_599,
        },
    ] {
        let result = derived(
            EPISODE_ID,
            &[&row],
            true,
            |_| true,
            |r| r.episode,
            |_| Some(readiness),
        );
        assert_eq!(result.applicability, LintApplicability::Inventory);
        assert_eq!(result.levels, vec![LintSeverity::Info]);
    }
    let overdue = DerivedReadiness {
        provider_ready: true,
        active_backfill: false,
        completed_sweeps: 2,
        missing_age_seconds: 3_600,
    };
    let result = derived(
        EPISODE_ID,
        &[&row],
        true,
        |_| true,
        |r| r.episode,
        |_| Some(overdue),
    );
    assert_eq!(result.applicability, LintApplicability::Applicable);
    assert_eq!(result.levels, vec![LintSeverity::Warning]);
}

#[test]
fn structural_failures_are_errors_but_pending_reembed_is_healthy() {
    let mut row = record("mem-a");
    row.embedding_valid = false;
    row.needs_reembed = true;
    row.embedding_valid = true;
    assert_eq!(
        structural(EMBEDDING_ID, &[&row], |r| r.embedding_valid).levels,
        vec![LintSeverity::Info]
    );
    row.needs_reembed = false;
    row.embedding_valid = false;
    assert_eq!(
        structural(EMBEDDING_ID, &[&row], |r| r.embedding_valid).levels,
        vec![LintSeverity::Error]
    );
    row.failed_steps = 1;
    assert_eq!(
        structural(ENRICHMENT_ID, &[row], |r| r.failed_steps == 0).levels,
        vec![LintSeverity::Error]
    );
}

#[test]
fn vector_liveness_enumerates_each_head_instead_of_accepting_nonzero_total() {
    let present = record("present");
    let mut missing = record("missing");
    missing.embedding_valid = false;
    let result = structural(EMBEDDING_ID, &[&present, &missing], |record| {
        record.embedding_valid
    });
    assert_eq!(result.levels, vec![LintSeverity::Info, LintSeverity::Error]);
    assert!(result.metrics.iter().any(|metric| {
        metric.code() == LintMetricCode::AffectedRecords
            && metric.value() == &LintMetricValue::Count { value: 1 }
    }));
}

#[test]
fn revision_orphans_self_links_and_replaced_heads_are_detected() {
    let mut target = record("target");
    let mut revision = record("revision");
    revision.supersedes = Some("target".to_string());
    target.replaced_by_active = true;
    let mut orphan = record("orphan");
    orphan.supersedes = Some("missing".to_string());
    orphan.target_exists = false;
    orphan.pending_revision = true;
    let mut self_link = record("self");
    self_link.supersedes = Some("self".to_string());
    let mut rows = vec![target.clone(), revision, orphan, self_link];
    mark_heads_and_supersession(&mut rows);
    target = rows.remove(0);
    assert!(!target.head);
    assert!(!rows[1].target_exists);
    assert!(
        !structural(SUPERSESSION_ID, &rows, |r| r.supersedes.as_deref()
            != Some(r.source_id.as_str())
            && r.target_exists
            && (!r.pending_revision || r.supersedes.is_some()))
        .levels
        .iter()
        .all(|level| *level == LintSeverity::Info)
    );
}

#[test]
fn partition_counts_use_full_population_not_sample_cap() {
    let rows = (0..101)
        .map(|index| {
            let mut row = record(&format!("mem-{index:03}"));
            row.event_dated = index % 2 == 0;
            row
        })
        .collect::<Vec<_>>();
    let heads = rows.iter().collect::<Vec<_>>();
    let result = partition_assessment(&heads, &rows);
    assert_eq!(result.levels.len(), 101);
    assert!(result
        .metrics
        .iter()
        .any(|metric| metric.code() == LintMetricCode::EligibleRecords
            && metric.value() == &LintMetricValue::Count { value: 101 }));
    assert!(result.metrics.iter().any(|metric| metric.code()
        == LintMetricCode::MemoryEventDatedHeads
        && metric.value() == &LintMetricValue::Count { value: 51 }));
}

#[tokio::test]
async fn selected_scope_anchors_memory_denominators_and_page_off_is_group_local() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    conn.execute("INSERT INTO spaces (id, name, created_at, updated_at) VALUES ('s-a','alpha',1,1),('s-b','beta',1,1)", ()).await.unwrap();
    for (id, space) in [("mem-a", "alpha"), ("mem-b", "beta")] {
        conn.execute("INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, stability, supersede_mode, space, needs_reembed, memory_type) VALUES (?1,'body','memory',?1,?1,0,1,'text','new','hide',?2,1,'fact')", libsql::params![id, space]).await.unwrap();
    }
    drop(conn);
    let before = semantic_fingerprint(&db).await;
    let report = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: Some("alpha".to_string()),
            },
            None,
            false,
        )
        .await
        .unwrap();
    let lifecycle = report
        .checks()
        .iter()
        .find(|check| check.check_id() == LIFECYCLE_ID)
        .unwrap();
    assert_eq!(lifecycle.coverage().denominator(), 1);
    assert_ne!(lifecycle.precondition(), LintPrecondition::ConfiguredOff);
    let page = report
        .checks()
        .iter()
        .find(|check| check.check_id() == "pages.db.partitions")
        .unwrap();
    assert_eq!(page.precondition(), LintPrecondition::ConfiguredOff);
    assert_eq!(before, semantic_fingerprint(&db).await);
}

async fn semantic_fingerprint(db: &crate::db::MemoryDB) -> DbSemanticFingerprint {
    let snapshot = db.open_isolated_lint_snapshot_for_test().await.unwrap();
    let fingerprint = DbSemanticFingerprint::capture(&snapshot).await.unwrap();
    snapshot.finish().await.unwrap();
    fingerprint
}
