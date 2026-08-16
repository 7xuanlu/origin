use super::super::{
    routes::{
        route, sensitive_read_routes, Capability, Method, ScopeBinding, SelectionGate,
        SelectorPrecedence, UnknownScopePolicy,
    },
    CHANNEL_EPISODE_ID, CHANNEL_PAGE_ID, OBSERVABILITY_ID, RERANKER_ID, ROUTE_SCOPE_ID,
};
use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use std::collections::BTreeSet;
use wenlan_types::lint::{
    LintApplicability, LintMetricCode, LintMetricValue, LintOutcome, LintQuery,
};

#[test]
fn route_contract_records_observed_scope_and_trust_semantics() {
    let search = route(Method::Post, "/api/search").expect("search row");
    assert_eq!(
        search.selector_precedence,
        SelectorPrecedence::BodyThenHeader
    );
    assert_eq!(search.capability, Capability::CallerAssertedAgentTrust);
    assert_eq!(search.unknown_scope, UnknownScopePolicy::Rejected);
    assert!(!search.scope_contract_violation());

    let page_search = route(Method::Post, "/api/pages/search").expect("page search row");
    assert_eq!(
        page_search.selector_precedence,
        SelectorPrecedence::BodyThenHeader
    );
    assert!(!page_search.scope_contract_violation());

    for (path, gate) in [
        ("/api/memory/{id}/detail", SelectionGate::SingleId404),
        ("/api/memory/by-ids", SelectionGate::BatchFiltered),
        ("/api/memory/{id}/versions", SelectionGate::SingleId404),
        ("/api/chunks/{source_id}", SelectionGate::SingleId404),
    ] {
        let row = route(Method::Get, path).expect("direct read row");
        assert_eq!(row.selection_gate, gate);
        assert!(!row.scope_contract_violation());
    }

    let snapshot =
        route(Method::Get, "/api/snapshots/{id}/captures-with-content").expect("snapshot read row");
    assert_eq!(
        snapshot.selection_gate,
        SelectionGate::ParentCollectionFiltered
    );
    assert!(!snapshot.scope_contract_violation());
}

#[test]
fn route_catalog_freezes_exact_global_and_scoped_keys() {
    const GLOBAL: &[(Method, &str)] = &[
        (Method::Get, "/api/profile"),
        (Method::Get, "/api/agents"),
        (Method::Get, "/api/agents/{name}"),
        (Method::Get, "/api/memory/stats"),
        (Method::Get, "/api/spaces"),
        (Method::Get, "/api/spaces/default"),
        (Method::Get, "/api/sources"),
        (Method::Get, "/api/profile/narrative"),
        (Method::Get, "/api/knowledge/count"),
        (Method::Get, "/api/onboarding/milestones"),
        (Method::Get, "/api/import/state"),
        (Method::Get, "/api/memory/rejections"),
        (Method::Get, "/api/refinery/queue"),
        (Method::Get, "/api/capture-stats"),
        (Method::Get, "/api/decisions/domains"),
        (Method::Get, "/api/snapshots"),
    ];
    const SCOPED: &[(Method, &str)] = &[
        (Method::Post, "/api/search"),
        (Method::Post, "/api/context"),
        (Method::Post, "/api/brief"),
        (Method::Get, "/api/memory/recent"),
        (Method::Get, "/api/memory/unconfirmed"),
        (Method::Post, "/api/memory/search"),
        (Method::Post, "/api/memory/list"),
        (Method::Get, "/api/memory/nurture"),
        (Method::Get, "/api/memory/pinned"),
        (Method::Get, "/api/retrievals/recent"),
        (Method::Get, "/api/home-stats"),
        (Method::Get, "/api/memory/{source_id}/enrichment-status"),
        (Method::Get, "/api/memory/{id}/revisions"),
        (Method::Get, "/api/indexed-files"),
        (Method::Get, "/api/chunks/{source_id}"),
        (Method::Get, "/api/activities"),
        (Method::Get, "/api/tags"),
        (Method::Get, "/api/suggest-tags"),
        (Method::Get, "/api/memory/{id}/detail"),
        (Method::Get, "/api/memory/by-ids"),
        (Method::Get, "/api/memory/{id}/versions"),
        (Method::Get, "/api/decisions"),
        (Method::Get, "/api/briefing"),
        (Method::Get, "/api/memory/pending-revisions"),
        (Method::Get, "/api/memory/pending-revision/{source_id}"),
        (Method::Get, "/api/snapshots/{id}/captures"),
        (Method::Get, "/api/snapshots/{id}/captures-with-content"),
        (Method::Get, "/api/pages/recent"),
        (Method::Get, "/api/pages/recent-changes"),
        (Method::Get, "/api/pages"),
        (Method::Post, "/api/pages/search"),
        (Method::Post, "/api/pages/export"),
        (Method::Get, "/api/pages/orphan-links"),
        (Method::Get, "/api/pages/{id}"),
        (Method::Post, "/api/pages/{id}/export"),
        (Method::Get, "/api/pages/{id}/sources"),
        (Method::Get, "/api/pages/{id}/links"),
        (Method::Get, "/api/pages/{id}/revisions"),
        (Method::Get, "/api/pages/{id}/map"),
        (Method::Post, "/api/memory/entities/list"),
        (Method::Post, "/api/memory/entities/search"),
        (Method::Get, "/api/memory/entities/{entity_id}"),
        (Method::Get, "/api/memory/entity-suggestions"),
        (Method::Get, "/api/memory/graph"),
        (Method::Get, "/api/knowledge/recent-relations"),
        (Method::Get, "/api/communities"),
        (Method::Get, "/api/communities/members"),
        (Method::Get, "/api/communities/page-assignments"),
        (Method::Get, "/api/communities/proposals"),
    ];

    let rows = sensitive_read_routes().collect::<Vec<_>>();
    let keys = rows
        .iter()
        .map(|row| (row.method, row.path))
        .collect::<BTreeSet<_>>();
    let global = rows
        .iter()
        .filter(|row| row.scope_binding == ScopeBinding::Global)
        .map(|row| (row.method, row.path))
        .collect::<BTreeSet<_>>();
    let scoped = rows
        .iter()
        .filter(|row| row.scope_binding != ScopeBinding::Global)
        .map(|row| (row.method, row.path))
        .collect::<BTreeSet<_>>();

    assert_eq!(rows.len(), 65);
    assert_eq!(keys.len(), 65, "duplicate sensitive route key");
    assert_eq!(global, GLOBAL.iter().copied().collect());
    assert_eq!(scoped, SCOPED.iter().copied().collect());
}

#[tokio::test]
async fn route_scope_result_is_derived_from_canonical_contract() {
    let (db, _tmp) = test_db().await;
    let report = run(&db, None).await;
    let result = check(&report, ROUTE_SCOPE_ID);
    let violations = super::super::routes::scope_contract_violations().collect::<Vec<_>>();
    let defects = violations.len() as u64;
    assert!(violations.is_empty());
    assert_eq!(defects, 0);
    assert_eq!(metric(result, LintMetricCode::AffectedRecords), defects);
}

#[tokio::test]
async fn page_serving_uses_retrieval_flag_not_projection_flag() {
    let (db, _tmp) = test_db().await;
    insert_memory(&db, "page-eligible", "work").await;
    let disabled = temp_env::async_with_vars(
        [("WENLAN_ENABLE_PAGE_CHANNEL", Some("0"))],
        run_with_projection(&db, false),
    )
    .await;
    let report = temp_env::async_with_vars(
        [("WENLAN_ENABLE_PAGE_CHANNEL", Some("1"))],
        run_with_projection(&db, false),
    )
    .await;
    let page = check(&report, CHANNEL_PAGE_ID);
    assert_eq!(page.outcome(), LintOutcome::Finding);
    assert_eq!(page.applicability(), LintApplicability::Applicable);
    assert_ne!(disabled.config_fingerprint(), report.config_fingerprint());
}

#[tokio::test]
async fn episode_liveness_uses_episode_specific_eligibility() {
    let (db, _tmp) = test_db().await;
    insert_memory_with_content(&db, "short", "work", "too short", 2).await;
    let conn = db.test_primary_session().await;
    conn.execute(
        "UPDATE memories SET source_text='one two three four five six seven eight' WHERE source_id='short'",
        (),
    )
    .await
    .expect("source text");
    drop(conn);
    let report = temp_env::async_with_vars(
        [("WENLAN_ENABLE_EPISODE_CHANNEL", Some("1"))],
        run(&db, None),
    )
    .await;
    let episode = check(&report, CHANNEL_EPISODE_ID);
    assert_eq!(metric(episode, LintMetricCode::EligibleRecords), 1);
    assert_eq!(episode.outcome(), LintOutcome::Finding);
}

#[tokio::test]
async fn telemetry_and_reranker_inventory_reports_observed_configuration() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO access_log (source_id,accessed_at) VALUES ('m',1)",
        (),
    )
    .await
    .expect("access row");
    conn.execute("INSERT INTO agent_activity (timestamp,agent_name,action,memory_ids,query,detail) VALUES (1,'agent','search','[]',NULL,'one')", ())
        .await
        .expect("activity row");
    drop(conn);
    let report =
        temp_env::async_with_vars([("WENLAN_RERANKER_MODE", Some("lite"))], run(&db, None)).await;
    let telemetry = check(&report, OBSERVABILITY_ID);
    assert_eq!(metric(telemetry, LintMetricCode::AccessTelemetryRows), 1);
    assert_eq!(
        metric(telemetry, LintMetricCode::AgentActivityTelemetryRows),
        1
    );
    assert_eq!(
        metric(telemetry, LintMetricCode::UnattributedServingChannels),
        2
    );
    let reranker = check(&report, RERANKER_ID);
    assert_eq!(metric(reranker, LintMetricCode::RerankerConfiguredPaths), 2);
    assert_eq!(
        metric(
            reranker,
            LintMetricCode::RerankerRuntimeReadinessUnavailable
        ),
        1
    );
}

pub(super) async fn run(
    db: &crate::db::MemoryDB,
    space: Option<&str>,
) -> wenlan_types::lint::LintReport {
    run_with_projection_and_space(db, false, space).await
}

async fn run_with_projection(
    db: &crate::db::MemoryDB,
    projection: bool,
) -> wenlan_types::lint::LintReport {
    run_with_projection_and_space(db, projection, None).await
}

async fn run_with_projection_and_space(
    db: &crate::db::MemoryDB,
    projection: bool,
    space: Option<&str>,
) -> wenlan_types::lint::LintReport {
    LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            db,
            &LintQuery {
                profile: None,
                space: space.map(str::to_string),
            },
            None,
            projection,
        )
        .await
        .expect("lint report")
}

pub(super) fn check<'a>(
    report: &'a wenlan_types::lint::LintReport,
    id: &str,
) -> &'a wenlan_types::lint::LintCheckResult {
    report
        .checks()
        .iter()
        .find(|check| check.check_id() == id)
        .expect("check")
}

pub(super) fn metric(check: &wenlan_types::lint::LintCheckResult, code: LintMetricCode) -> u64 {
    check
        .metrics()
        .iter()
        .find_map(|metric| {
            (metric.code() == code).then(|| match metric.value() {
                LintMetricValue::Count { value } => *value,
                _ => 0,
            })
        })
        .expect("metric")
}

pub(super) async fn insert_memory(db: &crate::db::MemoryDB, id: &str, space: &str) {
    insert_memory_with_content(db, id, space, "one two three four five six seven eight", 8).await;
}

async fn insert_memory_with_content(
    db: &crate::db::MemoryDB,
    id: &str,
    space: &str,
    content: &str,
    words: i64,
) {
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO memories (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,stability,supersede_mode,needs_reembed,memory_type,word_count,space,pending_revision,is_recap) VALUES (?1,?3,'memory',?1,?1,0,1,'text','new','hide',1,'fact',?4,?2,0,0)",
        libsql::params![id, space, content, words],
    ).await.expect("memory");
}
