// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use wenlan_server::sensitive_read_routes::{
    route, Method, ScopeBinding, SelectionGate, SelectorPrecedence, UnknownScopePolicy,
};

#[derive(Debug, Clone, Copy)]
pub struct ExpectedContract {
    pub method: Method,
    pub path: &'static str,
    pub precedence: SelectorPrecedence,
    pub binding: ScopeBinding,
    pub gate: SelectionGate,
}

pub const WAVE_1: &[ExpectedContract] = &[
    body(Method::Post, "/api/search"),
    body(Method::Post, "/api/context"),
    body(Method::Post, "/api/brief"),
    header(Method::Get, "/api/memory/recent"),
    header(Method::Get, "/api/memory/unconfirmed"),
    body(Method::Post, "/api/memory/search"),
    body(Method::Post, "/api/memory/list"),
    query(Method::Get, "/api/memory/nurture"),
    header(Method::Get, "/api/memory/pinned"),
];

pub const WAVE_2_RECORDS: &[ExpectedContract] = &[
    header_gate(
        Method::Get,
        "/api/memory/{source_id}/enrichment-status",
        SelectionGate::SingleId404,
    ),
    header_gate(
        Method::Get,
        "/api/memory/{id}/revisions",
        SelectionGate::SingleId404,
    ),
    header(Method::Get, "/api/indexed-files"),
    header_gate(
        Method::Get,
        "/api/chunks/{source_id}",
        SelectionGate::SingleId404,
    ),
    header_gate(Method::Get, "/api/suggest-tags", SelectionGate::SingleId404),
    header_gate(
        Method::Get,
        "/api/memory/{id}/detail",
        SelectionGate::SingleId404,
    ),
    header_gate(
        Method::Get,
        "/api/memory/by-ids",
        SelectionGate::BatchFiltered,
    ),
    header_gate(
        Method::Get,
        "/api/memory/{id}/versions",
        SelectionGate::SingleId404,
    ),
    query(Method::Get, "/api/decisions"),
    header(Method::Get, "/api/memory/pending-revisions"),
    header_gate(
        Method::Get,
        "/api/memory/pending-revision/{source_id}",
        SelectionGate::SingleId404,
    ),
];

pub const WAVE_3_PAGES: &[ExpectedContract] = &[
    page_header(Method::Get, "/api/pages/recent"),
    page_header(Method::Get, "/api/pages/recent-changes"),
    page_query(Method::Get, "/api/pages"),
    page_body(Method::Post, "/api/pages/search"),
    page_header(Method::Post, "/api/pages/export"),
    page_header(Method::Get, "/api/pages/orphan-links"),
    page_header_gate(Method::Get, "/api/pages/{id}", SelectionGate::SingleId404),
    page_header_gate(
        Method::Post,
        "/api/pages/{id}/export",
        SelectionGate::SingleId404,
    ),
    page_header_gate(
        Method::Get,
        "/api/pages/{id}/sources",
        SelectionGate::ParentCollectionFiltered,
    ),
    page_header_gate(
        Method::Get,
        "/api/pages/{id}/links",
        SelectionGate::ParentCollectionFiltered,
    ),
    page_header_gate(
        Method::Get,
        "/api/pages/{id}/revisions",
        SelectionGate::ParentCollectionFiltered,
    ),
];

pub const WAVE_4_KNOWLEDGE: &[ExpectedContract] = &[
    entity_body(Method::Post, "/api/memory/entities/list"),
    entity_body(Method::Post, "/api/memory/entities/search"),
    entity_header_gate(
        Method::Get,
        "/api/memory/entities/{entity_id}",
        SelectionGate::SingleId404,
    ),
    entity_header(Method::Get, "/api/memory/graph"),
];

const fn body(method: Method, path: &'static str) -> ExpectedContract {
    expected(method, path, SelectorPrecedence::BodyThenHeader)
}

const fn query(method: Method, path: &'static str) -> ExpectedContract {
    expected(method, path, SelectorPrecedence::QueryThenHeader)
}

const fn header(method: Method, path: &'static str) -> ExpectedContract {
    expected(method, path, SelectorPrecedence::HeaderOnly)
}

const fn header_gate(method: Method, path: &'static str, gate: SelectionGate) -> ExpectedContract {
    let mut contract = header(method, path);
    contract.gate = gate;
    contract
}

const fn expected(
    method: Method,
    path: &'static str,
    precedence: SelectorPrecedence,
) -> ExpectedContract {
    ExpectedContract {
        method,
        path,
        precedence,
        binding: ScopeBinding::MemorySpace,
        gate: SelectionGate::NotApplicable,
    }
}

const fn page_body(method: Method, path: &'static str) -> ExpectedContract {
    page_expected(method, path, SelectorPrecedence::BodyThenHeader)
}

const fn page_query(method: Method, path: &'static str) -> ExpectedContract {
    page_expected(method, path, SelectorPrecedence::QueryThenHeader)
}

const fn page_header(method: Method, path: &'static str) -> ExpectedContract {
    page_expected(method, path, SelectorPrecedence::HeaderOnly)
}

const fn page_header_gate(
    method: Method,
    path: &'static str,
    gate: SelectionGate,
) -> ExpectedContract {
    let mut contract = page_header(method, path);
    contract.gate = gate;
    contract
}

const fn page_expected(
    method: Method,
    path: &'static str,
    precedence: SelectorPrecedence,
) -> ExpectedContract {
    ExpectedContract {
        method,
        path,
        precedence,
        binding: ScopeBinding::PageWorkspace,
        gate: SelectionGate::NotApplicable,
    }
}

const fn entity_body(method: Method, path: &'static str) -> ExpectedContract {
    entity_expected(method, path, SelectorPrecedence::BodyThenHeader)
}

const fn entity_header(method: Method, path: &'static str) -> ExpectedContract {
    entity_expected(method, path, SelectorPrecedence::HeaderOnly)
}

const fn entity_header_gate(
    method: Method,
    path: &'static str,
    gate: SelectionGate,
) -> ExpectedContract {
    let mut contract = entity_header(method, path);
    contract.gate = gate;
    contract
}

const fn entity_expected(
    method: Method,
    path: &'static str,
    precedence: SelectorPrecedence,
) -> ExpectedContract {
    ExpectedContract {
        method,
        path,
        precedence,
        binding: ScopeBinding::EntitySpace,
        gate: SelectionGate::NotApplicable,
    }
}

pub fn assert_wave_1_catalog_contract() {
    let keys = WAVE_1
        .iter()
        .map(|case| (case.method, case.path))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        keys.len(),
        9,
        "Wave 1 registry must contain nine unique keys"
    );

    for expected in WAVE_1 {
        let actual = route(expected.method, expected.path).expect("cataloged Wave 1 route");
        assert_eq!(
            actual.selector_precedence, expected.precedence,
            "{}",
            expected.path
        );
        assert_eq!(actual.scope_binding, expected.binding, "{}", expected.path);
        assert_eq!(actual.selection_gate, expected.gate, "{}", expected.path);
        assert_eq!(
            actual.unknown_scope,
            UnknownScopePolicy::Rejected,
            "{}",
            expected.path
        );
        assert!(!actual.scope_contract_violation(), "{}", expected.path);
    }
}

pub fn assert_wave_1_executed_keys(executed: impl IntoIterator<Item = (Method, &'static str)>) {
    let expected = WAVE_1
        .iter()
        .map(|case| (case.method, case.path))
        .collect::<BTreeSet<_>>();
    let executed = executed.into_iter().collect::<Vec<_>>();
    let unique = executed.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(
        unique.len(),
        executed.len(),
        "duplicate executed Wave 1 key"
    );
    assert_eq!(unique, expected, "executed Wave 1 key set drifted");
}

pub fn assert_wave_2_records_catalog_contract() {
    let keys = WAVE_2_RECORDS
        .iter()
        .map(|case| (case.method, case.path))
        .collect::<BTreeSet<_>>();
    assert_eq!(keys.len(), 11, "Wave 2 records must contain 11 unique keys");

    for expected in WAVE_2_RECORDS {
        let actual = route(expected.method, expected.path).expect("cataloged Wave 2 route");
        assert_eq!(
            actual.selector_precedence, expected.precedence,
            "{}",
            expected.path
        );
        assert_eq!(actual.scope_binding, expected.binding, "{}", expected.path);
        assert_eq!(actual.selection_gate, expected.gate, "{}", expected.path);
        assert_eq!(
            actual.unknown_scope,
            UnknownScopePolicy::Rejected,
            "{}",
            expected.path
        );
        assert!(!actual.scope_contract_violation(), "{}", expected.path);
    }
}

pub fn assert_wave_2_records_executed_keys(
    executed: impl IntoIterator<Item = (Method, &'static str)>,
) {
    let expected = WAVE_2_RECORDS
        .iter()
        .map(|case| (case.method, case.path))
        .collect::<BTreeSet<_>>();
    let executed = executed.into_iter().collect::<Vec<_>>();
    let unique = executed.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(unique.len(), executed.len(), "duplicate Wave 2 record key");
    assert_eq!(unique, expected, "executed Wave 2 record key set drifted");
}

pub fn assert_wave_3_pages_catalog_contract() {
    let keys = WAVE_3_PAGES
        .iter()
        .map(|case| (case.method, case.path))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        keys.len(),
        11,
        "Wave 3 Pages must contain eleven unique keys"
    );

    for expected in WAVE_3_PAGES {
        let actual = route(expected.method, expected.path).expect("cataloged Wave 3 Page route");
        assert_eq!(
            actual.selector_precedence, expected.precedence,
            "{}",
            expected.path
        );
        assert_eq!(actual.scope_binding, expected.binding, "{}", expected.path);
        assert_eq!(actual.selection_gate, expected.gate, "{}", expected.path);
        assert_eq!(
            actual.unknown_scope,
            UnknownScopePolicy::Rejected,
            "{}",
            expected.path
        );
        assert!(!actual.scope_contract_violation(), "{}", expected.path);
    }
}

pub fn assert_wave_3_pages_executed_keys(
    executed: impl IntoIterator<Item = (Method, &'static str)>,
) {
    let expected = WAVE_3_PAGES
        .iter()
        .map(|case| (case.method, case.path))
        .collect::<BTreeSet<_>>();
    let executed = executed.into_iter().collect::<Vec<_>>();
    let unique = executed.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(unique.len(), executed.len(), "duplicate Wave 3 Page key");
    assert_eq!(unique, expected, "executed Wave 3 Page key set drifted");
}

pub fn assert_wave_4_knowledge_catalog_contract() {
    let keys = WAVE_4_KNOWLEDGE
        .iter()
        .map(|case| (case.method, case.path))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        keys.len(),
        4,
        "Wave 4 Knowledge must contain four unique keys"
    );

    for expected in WAVE_4_KNOWLEDGE {
        let actual = route(expected.method, expected.path).expect("cataloged Wave 4 route");
        assert_eq!(
            actual.selector_precedence, expected.precedence,
            "{}",
            expected.path
        );
        assert_eq!(actual.scope_binding, expected.binding, "{}", expected.path);
        assert_eq!(actual.selection_gate, expected.gate, "{}", expected.path);
        assert_eq!(
            actual.unknown_scope,
            UnknownScopePolicy::Rejected,
            "{}",
            expected.path
        );
        assert!(!actual.scope_contract_violation(), "{}", expected.path);
    }

    let rows = wenlan_server::sensitive_read_routes::sensitive_read_routes().collect::<Vec<_>>();
    assert_eq!(rows.len(), 62);
    assert_eq!(
        rows.iter()
            .filter(|row| row.scope_binding == ScopeBinding::Global)
            .count(),
        16
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.scope_binding != ScopeBinding::Global)
            .count(),
        46
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.scope_contract_violation())
            .count(),
        0
    );
}

pub fn assert_wave_4_knowledge_executed_keys(
    executed: impl IntoIterator<Item = (Method, &'static str)>,
) {
    let expected = WAVE_4_KNOWLEDGE
        .iter()
        .map(|case| (case.method, case.path))
        .collect::<BTreeSet<_>>();
    let executed = executed.into_iter().collect::<Vec<_>>();
    let unique = executed.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(
        unique.len(),
        executed.len(),
        "duplicate Wave 4 Knowledge key"
    );
    assert_eq!(
        unique, expected,
        "executed Wave 4 Knowledge key set drifted"
    );
}

pub fn assert_global_executed_keys(executed: impl IntoIterator<Item = (Method, &'static str)>) {
    let expected = wenlan_server::sensitive_read_routes::sensitive_read_routes()
        .filter(|row| row.scope_binding == ScopeBinding::Global)
        .map(|row| (row.method, row.path))
        .collect::<BTreeSet<_>>();
    let executed = executed.into_iter().collect::<Vec<_>>();
    let unique = executed.iter().copied().collect::<BTreeSet<_>>();

    assert_eq!(expected.len(), 16, "Global route catalog count drifted");
    assert_eq!(
        unique.len(),
        executed.len(),
        "duplicate Global behavior key"
    );
    assert_eq!(unique, expected, "executed Global behavior key set drifted");
}
