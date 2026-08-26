use super::{assess_channel, ChannelAssessment, CHANNEL_FACT_ID, CHANNEL_GRAPH_ID, ROUTE_SCOPE_ID};
use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use wenlan_types::lint::{
    LintActionCode, LintApplicability, LintMetricCode, LintMetricValue, LintOutcome,
    LintPrecondition, LintQuery, LintSummaryCode,
};

#[path = "serving_review_fact_test.rs"]
mod review_fact_tests;
#[path = "serving_review_test.rs"]
mod review_tests;

#[test]
fn enabled_channel_with_dead_eligible_substrate_is_a_finding() {
    assert_eq!(
        assess_channel(CHANNEL_FACT_ID, true, 3, 0),
        ChannelAssessment::Finding { eligible: 3 }
    );
}

#[test]
fn route_scope_finding_depends_on_the_violation_count() {
    assert!(!super::route_scope_finding(0));
    assert!(super::route_scope_finding(1));
}

#[test]
fn disabled_channel_is_expected_empty() {
    let result = super::channel_result(
        &LintClock::fixed(),
        CHANNEL_FACT_ID,
        assess_channel(CHANNEL_FACT_ID, false, 3, 0),
    );
    assert_eq!(result.outcome(), LintOutcome::Pass);
    assert_eq!(result.applicability(), LintApplicability::ExpectedEmpty);
    assert_eq!(result.precondition(), LintPrecondition::ConfiguredOff);
    assert_eq!(result.summary_code(), LintSummaryCode::ExpectedEmpty);
}

// Tracker row 17: the graph channel serves a substrate that model-backed
// background enrichment builds. With no model source chosen, an empty channel
// is the expected state and the fix is a settings choice, not a repair -- so it
// passes, carries the "choose a model source" advice, and does not gate a fresh
// install's exit code.
#[test]
fn graph_channel_without_a_model_source_passes_and_says_to_choose_one() {
    let result = super::channel_result(
        &LintClock::fixed(),
        CHANNEL_GRAPH_ID,
        ChannelAssessment::ModelSourceUnconfigured { eligible: 2 },
    );
    assert_eq!(result.outcome(), LintOutcome::Pass);
    assert_eq!(result.applicability(), LintApplicability::ExpectedEmpty);
    assert_eq!(result.precondition(), LintPrecondition::ExpectedEmpty);
    assert_eq!(result.summary_code(), LintSummaryCode::ExpectedEmpty);
    assert_eq!(result.recommendation_code(), None);
    assert_eq!(
        result.action_code(),
        Some(LintActionCode::ChooseModelSource)
    );
    // No opaque ordinals for a state that is not a defect.
    assert!(result.evidence().is_empty());
    assert!(!result.coverage().truncated());
    assert_eq!(result.coverage().denominator(), 2);
}

// End to end through the runner: the captured model-source pin reaches the
// graph channel, and it is the only channel it reaches. A channel the owner
// switched off keeps saying so rather than nagging about a model source it will
// not use, and dropping the graph finding takes the report's actionable count
// down with it.
#[tokio::test]
async fn the_model_source_pin_reaches_only_the_graph_channel_through_the_runner() {
    let (db, _tmp) = test_db().await;
    insert_memory(&db, "eligible", None).await;

    let unconfigured = run_with_kg_config(&db, kg_config(false)).await;
    let graph = channel(&unconfigured, CHANNEL_GRAPH_ID);
    assert_eq!(graph.outcome(), LintOutcome::Pass);
    assert_eq!(graph.applicability(), LintApplicability::ExpectedEmpty);
    assert_eq!(graph.precondition(), LintPrecondition::ExpectedEmpty);
    assert_eq!(graph.recommendation_code(), None);
    assert_eq!(graph.action_code(), Some(LintActionCode::ChooseModelSource));
    assert!(unconfigured.complete());

    // The fact channel is off in this fixture and stays configured-off.
    let fact = channel(&unconfigured, CHANNEL_FACT_ID);
    assert_eq!(fact.precondition(), LintPrecondition::ConfiguredOff);
    assert_eq!(fact.recommendation_code(), None);
    assert_eq!(fact.action_code(), None);

    // The same store with a model source chosen: the graph channel and the kg
    // liveness check are exactly what the fresh-store report stops gating on.
    let configured = run_with_kg_config(&db, kg_config(true)).await;
    assert_eq!(
        channel(&configured, CHANNEL_GRAPH_ID).outcome(),
        LintOutcome::Finding
    );
    let mut newly_gating = actionable_ids(&configured);
    for id in actionable_ids(&unconfigured) {
        assert!(newly_gating.remove(&id), "{id} lost its finding");
    }
    assert_eq!(
        newly_gating,
        ["kg.substrate_liveness", CHANNEL_GRAPH_ID]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
}

fn actionable_ids(report: &wenlan_types::lint::LintReport) -> std::collections::BTreeSet<String> {
    report
        .checks()
        .iter()
        .filter(|check| {
            check.outcome() == LintOutcome::Finding
                && check.gate_effect() == wenlan_types::lint::LintGateEffect::Actionable
        })
        .map(|check| check.check_id().to_string())
        .collect()
}

fn kg_config(model_source_configured: bool) -> crate::lint::kg::KgRunConfig {
    crate::lint::kg::KgRunConfig::for_test(true, true, true, model_source_configured, 20)
}

async fn run_with_kg_config(
    db: &crate::db::MemoryDB,
    config: crate::lint::kg::KgRunConfig,
) -> wenlan_types::lint::LintReport {
    temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_PAGE_CHANNEL", Some("0")),
            ("WENLAN_ENABLE_EPISODE_CHANNEL", Some("0")),
            ("WENLAN_ENABLE_FACT_CHANNEL", Some("0")),
            ("WENLAN_ENABLE_GLOBAL_PRELUDE", Some("0")),
        ],
        LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .with_test_kg_config(config)
            .run(
                db,
                &LintQuery {
                    profile: None,
                    space: None,
                },
                None,
                false,
            ),
    )
    .await
    .unwrap()
}

fn channel<'a>(
    report: &'a wenlan_types::lint::LintReport,
    id: &str,
) -> &'a wenlan_types::lint::LintCheckResult {
    report
        .checks()
        .iter()
        .find(|check| check.check_id() == id)
        .unwrap()
}

#[tokio::test]
async fn runner_reports_clean_scope_contracts_and_preserves_telemetry() {
    let (db, _tmp) = test_db().await;
    let before = telemetry_counts(&db).await;
    let report = run_with_flags(&db, None, false).await.unwrap();
    let route_scope = report
        .checks()
        .iter()
        .find(|check| check.check_id() == ROUTE_SCOPE_ID)
        .unwrap();
    assert_eq!(route_scope.outcome(), LintOutcome::Pass);
    assert!(route_scope.metrics().iter().any(|metric| {
        metric.code() == LintMetricCode::AffectedRecords
            && metric.value() == &LintMetricValue::Count { value: 0 }
    }));
    assert_eq!(before, telemetry_counts(&db).await);
}

#[tokio::test]
async fn unknown_scope_fails_closed_before_serving_checks() {
    let (db, _tmp) = test_db().await;
    let result = run_with_flags(&db, Some("missing"), false).await;
    assert!(matches!(
        result,
        Err(crate::lint::runner::LintRunError::InvalidScope)
    ));
    assert_eq!(telemetry_counts(&db).await, (0, 0));
}

#[tokio::test]
async fn uncategorized_uses_null_memory_and_page_ownership_axes() {
    let (db, _tmp) = test_db().await;
    insert_memory(&db, "null-memory", None).await;
    insert_memory(&db, "work-memory", Some("work")).await;
    insert_page(&db, "null-page", None).await;
    insert_page(&db, "work-page", Some("work")).await;
    let report = run_with_flags(&db, Some("uncategorized"), true)
        .await
        .unwrap();
    let page = report
        .checks()
        .iter()
        .find(|check| check.check_id() == super::CHANNEL_PAGE_ID)
        .unwrap();
    assert_eq!(page.coverage().denominator(), 1);
    assert!(page.metrics().iter().any(|metric| {
        metric.code() == LintMetricCode::ObservedRecords
            && metric.value() == &LintMetricValue::Count { value: 1 }
    }));
}

#[tokio::test]
async fn cross_scope_canary_cannot_change_scoped_serving_results() {
    let (db, _tmp) = test_db().await;
    db.create_space("work", None, false).await.unwrap();
    insert_memory(&db, "work-memory", Some("work")).await;
    let before = scoped_serving_json(&db, "work").await;
    insert_memory(&db, "other-memory", Some("other")).await;
    assert_eq!(before, scoped_serving_json(&db, "work").await);
}

async fn scoped_serving_json(db: &crate::db::MemoryDB, space: &str) -> serde_json::Value {
    let report = run_with_flags(db, Some(space), false).await.unwrap();
    serde_json::to_value(
        report
            .checks()
            .iter()
            .filter(|check| check.check_id().starts_with("serving."))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

async fn run_with_flags(
    db: &crate::db::MemoryDB,
    space: Option<&str>,
    page: bool,
) -> Result<wenlan_types::lint::LintReport, crate::lint::runner::LintRunError> {
    temp_env::async_with_vars(
        [
            (
                "WENLAN_ENABLE_PAGE_CHANNEL",
                Some(if page { "1" } else { "0" }),
            ),
            ("WENLAN_ENABLE_EPISODE_CHANNEL", Some("0")),
            ("WENLAN_ENABLE_FACT_CHANNEL", Some("0")),
            ("WENLAN_ENABLE_GLOBAL_PRELUDE", Some("0")),
        ],
        LintRunner::new(LintClock::fixed(), CancellationToken::new()).run(
            db,
            &LintQuery {
                profile: None,
                space: space.map(str::to_string),
            },
            None,
            false,
        ),
    )
    .await
}

async fn insert_memory(db: &crate::db::MemoryDB, id: &str, space: Option<&str>) {
    // M3 PR-1 stage e honest columns: memories.space is NOT NULL since
    // migration 91, so a caller who wants "no space" must bind the reserved
    // sentinel id -- binding an explicit NULL now violates the column
    // constraint here too, same as insert_page below has since M1.
    let space = space.unwrap_or(crate::db::UNFILED_SPACE_ID);
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
         last_modified, chunk_type, stability, supersede_mode, needs_reembed, memory_type,
         word_count, space) VALUES (?1, 'one two three four five six seven eight', 'memory',
         ?1, ?1, 0, 1, 'text', 'new', 'hide', 1, 'fact', 8, ?2)",
        libsql::params![id, space],
    )
    .await
    .unwrap();
}

async fn insert_page(db: &crate::db::MemoryDB, id: &str, workspace: Option<&str>) {
    // M1 honest columns: pages.workspace is NOT NULL since migration 80, so a
    // caller who wants "no workspace" must bind the reserved sentinel id --
    // binding an explicit NULL always violates the column constraint here.
    let workspace = workspace.unwrap_or(crate::db::UNFILED_SPACE_ID);
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO pages (id, title, content, source_memory_ids, version, status,
         created_at, last_compiled, last_modified, workspace, creation_kind, review_status)
         VALUES (?1, ?1, 'body', '[]', 1, 'active', 'now', 'now', 'now', ?2,
         'distilled', 'confirmed')",
        libsql::params![id, workspace],
    )
    .await
    .unwrap();
}

async fn telemetry_counts(db: &crate::db::MemoryDB) -> (i64, i64) {
    let conn = db.test_primary_session().await;
    let mut rows = conn
        .query(
            "SELECT (SELECT COUNT(*) FROM access_log), (SELECT COUNT(*) FROM agent_activity)",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    (row.get(0).unwrap(), row.get(1).unwrap())
}
