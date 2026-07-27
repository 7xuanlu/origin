use super::{
    route, sensitive_read_routes, Capability, Method, ScopeBinding, SelectionGate,
    SelectorPrecedence, UnknownScopePolicy,
};
use std::collections::BTreeSet;

#[test]
fn canonical_matrix_is_unique_and_matches_observed_handler_contracts() {
    let rows = sensitive_read_routes().collect::<Vec<_>>();
    let keys = rows
        .iter()
        .map(|row| (row.method, row.path))
        .collect::<BTreeSet<_>>();
    assert_eq!(keys.len(), rows.len(), "duplicate sensitive read row");

    let search = route(Method::Post, "/api/search").expect("search");
    assert_eq!(
        search.selector_precedence,
        SelectorPrecedence::BodyThenHeader
    );
    assert_eq!(search.capability, Capability::CallerAssertedAgentTrust);
    assert_eq!(search.unknown_scope, UnknownScopePolicy::Rejected);
    assert!(!search.scope_contract_violation());

    let page_search = route(Method::Post, "/api/pages/search").expect("page search");
    assert_eq!(
        page_search.selector_precedence,
        SelectorPrecedence::BodyThenHeader
    );
    assert_eq!(page_search.scope_binding, ScopeBinding::PageWorkspace);
    assert_eq!(page_search.unknown_scope, UnknownScopePolicy::Rejected);
    assert!(!page_search.scope_contract_violation());

    let assignments = route(Method::Get, "/api/communities/page-assignments")
        .expect("community page assignments");
    assert_eq!(assignments.scope_binding, ScopeBinding::PageWorkspace);
    assert_eq!(
        assignments.selector_precedence,
        SelectorPrecedence::QueryThenHeader
    );
    assert!(!assignments.scope_contract_violation());

    for path in [
        "/api/home-stats",
        "/api/retrievals/recent",
        "/api/activities",
        "/api/tags",
    ] {
        let row = route(Method::Get, path).expect("derived projection row");
        assert_eq!(row.selector_precedence, SelectorPrecedence::HeaderOnly);
        assert_eq!(row.scope_binding, ScopeBinding::MemorySpace);
        assert_eq!(row.unknown_scope, UnknownScopePolicy::Rejected);
        assert!(!row.scope_contract_violation());
    }

    for (path, gate) in [
        ("/api/memory/{id}/detail", SelectionGate::SingleId404),
        ("/api/memory/by-ids", SelectionGate::BatchFiltered),
        ("/api/memory/{id}/versions", SelectionGate::SingleId404),
        ("/api/chunks/{source_id}", SelectionGate::SingleId404),
        (
            "/api/memory/pending-revision/{source_id}",
            SelectionGate::SingleId404,
        ),
    ] {
        let row = route(Method::Get, path).expect("direct row");
        assert_eq!(row.selection_gate, gate);
        assert!(!row.scope_contract_violation());
    }
    for path in [
        "/api/snapshots/{id}/captures",
        "/api/snapshots/{id}/captures-with-content",
    ] {
        let row = route(Method::Get, path).expect("parent collection row");
        assert_eq!(row.selection_gate, SelectionGate::ParentCollectionFiltered);
        assert!(!row.scope_contract_violation());
    }
}

#[test]
fn canonical_matrix_freezes_exact_global_and_scoped_keys() {
    const GLOBAL: &[(Method, &str)] = &[
        (Method::Get, "/api/profile"),
        (Method::Get, "/api/agents"),
        (Method::Get, "/api/agents/{name}"),
        (Method::Get, "/api/memory/stats"),
        (Method::Get, "/api/spaces"),
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

    assert_eq!(rows.len(), 62);
    assert_eq!(keys.len(), 62, "duplicate sensitive route key");
    assert_eq!(global, GLOBAL.iter().copied().collect());
    assert_eq!(scoped, SCOPED.iter().copied().collect());
    assert_eq!(
        rows.iter()
            .filter(|row| row.scope_contract_violation())
            .count(),
        0
    );
}

#[test]
fn every_canonical_sensitive_row_is_bound_by_router_construction() {
    let state = std::sync::Arc::new(tokio::sync::RwLock::new(
        crate::state::ServerState::default(),
    ));
    let _: crate::router::AppRouter = crate::router::build_router(state);
}
