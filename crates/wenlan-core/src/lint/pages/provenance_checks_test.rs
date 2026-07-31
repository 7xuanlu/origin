use super::citations::{assess_citations, load_and_assess_citations};
use super::source::{assess_sources, load_sources, ExtraEvidence, SourceRecord};
use super::*;
use crate::db::test_support::TestDbSession;
use crate::db::tests::test_db;
use crate::lint::context::{
    AppliedScope, CancellationToken, ExecutionGate, LintClock, LintContext,
};
use std::collections::BTreeMap;
use wenlan_types::lint::{
    LintApplicability, LintMetricCode, LintMetricValue, LintOpaqueId, LintOutcome, LintSeverity,
};

fn source(locator: &str, expected: &str, actual: &[&str]) -> SourceRecord {
    SourceRecord {
        locator: locator.to_string(),
        owner_present: true,
        expected_kind: expected.to_string(),
        evidence_kinds: actual.iter().map(|kind| (*kind).to_string()).collect(),
    }
}

#[test]
fn mixed_source_kinds_and_drift_map_to_exact_outcomes() {
    let passing = assess_sources(
        &[
            source("m", "memory", &["memory"]),
            source("f", "external_file", &["external_file"]),
            source("u", "external_url", &["external_url"]),
            source("a", "authored", &["authored"]),
        ],
        &[
            ExtraEvidence {
                source_kind: "memory".to_string(),
                locator_present: true,
            },
            ExtraEvidence {
                source_kind: "authored".to_string(),
                locator_present: false,
            },
        ],
    )
    .result(SOURCE_COVERAGE_ID, 0)
    .unwrap();
    assert_eq!(passing.outcome(), LintOutcome::Pass);
    assert_eq!(passing.applicability(), LintApplicability::Inventory);
    assert_eq!(passing.coverage().denominator(), 4);

    let warning = assess_sources(
        &[
            source("wrong", "memory", &["external_file"]),
            source("multi", "memory", &["memory", "external_url"]),
        ],
        &[],
    )
    .result(SOURCE_COVERAGE_ID, 0)
    .unwrap();
    assert_eq!(warning.outcome(), LintOutcome::Finding);
    assert_eq!(warning.severity(), LintSeverity::Warning);

    let error = assess_sources(
        &[
            source("missing", "memory", &[]),
            source("unknown", "memory", &["future_kind"]),
            SourceRecord {
                locator: "orphan".to_string(),
                owner_present: false,
                expected_kind: "memory".to_string(),
                evidence_kinds: vec!["memory".to_string()],
            },
        ],
        &[],
    )
    .result(SOURCE_COVERAGE_ID, 0)
    .unwrap();
    assert_eq!(error.outcome(), LintOutcome::Finding);
    assert_eq!(error.severity(), LintSeverity::Error);
}

#[test]
fn citation_null_empty_nonempty_and_occurrence_partitions_are_exact() {
    let citations = serde_json::json!([
        {"occurrence":1,"marker":1,"source_kind":"memory","locator":"secret-memory","score":1.0,"status":"verified","scope":"sentence"},
        {"occurrence":2,"marker":2,"source_kind":"external_file","locator":"/secret/file","score":0.5,"status":"unverified","scope":"paragraph"},
        {"occurrence":3,"marker":3,"source_kind":"external_url","locator":"https://secret.example","score":0.7,"status":"verified","scope":"sentence"},
        {"occurrence":4,"marker":4,"source_kind":"authored","locator":"secret-author","score":0.9,"status":"verified","scope":"paragraph"}
    ])
    .to_string();
    let (assessment, partitions) =
        assess_citations(&[None, Some("[]".to_string()), Some(citations)]);
    let result = assessment.result(CITATION_PARTITIONS_ID, 0).unwrap();
    assert_eq!(result.outcome(), LintOutcome::Pass);
    assert_eq!(result.applicability(), LintApplicability::Inventory);
    assert_eq!(result.coverage().denominator(), 3);
    assert_eq!(partitions.page_total(), 3);
    assert_eq!(partitions.occurrences, 4);
    assert!(partitions.partitions_are_exact());
    assert_eq!(metric(&result, LintMetricCode::CitationNullPages), 1);
    assert_eq!(metric(&result, LintMetricCode::CitationEmptyPages), 1);
    assert_eq!(metric(&result, LintMetricCode::CitationNonemptyPages), 1);
    assert_eq!(
        metric(&result, LintMetricCode::CitationVerifiedOccurrences),
        3
    );
    assert_eq!(
        metric(&result, LintMetricCode::CitationUnverifiedOccurrences),
        1
    );
    assert_eq!(
        metric(&result, LintMetricCode::CitationSentenceOccurrences),
        2
    );
    assert_eq!(
        metric(&result, LintMetricCode::CitationParagraphOccurrences),
        2
    );
    assert_eq!(
        metric(&result, LintMetricCode::CitationMemoryOccurrences),
        1
    );
    assert_eq!(
        metric(&result, LintMetricCode::CitationExternalFileOccurrences),
        1
    );
    assert_eq!(
        metric(&result, LintMetricCode::CitationExternalUrlOccurrences),
        1
    );
    assert_eq!(
        metric(&result, LintMetricCode::CitationAuthoredOccurrences),
        1
    );
    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("secret"));
}

#[test]
fn malformed_or_unknown_citation_dimensions_are_errors() {
    for raw in [
        "not-json".to_string(),
        serde_json::json!([{"occurrence":1,"marker":1,"source_kind":"future","locator":"x","score":0.0,"status":"verified","scope":"sentence"}]).to_string(),
        serde_json::json!([{"occurrence":1,"marker":1,"source_kind":"memory","locator":"x","score":0.0,"status":"future","scope":"sentence"}]).to_string(),
        serde_json::json!([{"occurrence":1,"marker":1,"source_kind":"memory","locator":"x","score":0.0,"status":"verified","scope":"future"}]).to_string(),
    ] {
        let (assessment, _) = assess_citations(&[Some(raw)]);
        let result = assessment.result(CITATION_PARTITIONS_ID, 0).unwrap();
        assert_eq!(result.outcome(), LintOutcome::Finding);
        assert_eq!(result.severity(), LintSeverity::Error);
        assert_eq!(result.coverage().denominator(), 1);
    }
}

#[tokio::test]
async fn hundred_thousand_sql_rows_validate_all_and_cap_opaque_samples() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page(&conn, "page-scale", "workspace-scale", None).await;
    conn.execute(
        "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x + 1 FROM n WHERE x < 100000)
         INSERT INTO page_sources (page_id, memory_source_id, linked_at, link_reason)
         SELECT 'page-scale', printf('private-locator-%06d', x), 1, 'scale' FROM n",
        (),
    )
    .await
    .unwrap();
    conn.execute(
        "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x + 1 FROM n WHERE x < 99999)
         INSERT INTO page_evidence (page_id, source_kind, locator, linked_at, link_reason)
         SELECT 'page-scale', CASE WHEN x > 100 AND x <= 200 THEN 'external_file' ELSE 'memory' END,
                printf('private-locator-%06d', x), 1, 'scale' FROM n",
        (),
    )
    .await
    .unwrap();
    let snapshot = db.open_lint_snapshot().await.unwrap();
    let mut source_count = snapshot
        .query(
            "SELECT COUNT(*) FROM page_sources WHERE page_id = 'page-scale'",
            libsql::params::Params::None,
        )
        .await
        .unwrap();
    assert_eq!(
        source_count
            .next()
            .await
            .unwrap()
            .unwrap()
            .get::<i64>(0)
            .unwrap(),
        100_000
    );
    let mut evidence_count = snapshot
        .query(
            "SELECT COUNT(*) FROM page_evidence WHERE page_id = 'page-scale'",
            libsql::params::Params::None,
        )
        .await
        .unwrap();
    assert_eq!(
        evidence_count
            .next()
            .await
            .unwrap()
            .unwrap()
            .get::<i64>(0)
            .unwrap(),
        99_999
    );
    let scope = AppliedScope::global();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let context = LintContext::new(
        &snapshot,
        &scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let result = super::source::load_and_assess_sources(&context)
        .await
        .unwrap()
        .result(SOURCE_COVERAGE_ID, 0)
        .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Error);
    assert_eq!(result.coverage().denominator(), 100_000);
    assert_eq!(result.coverage().evaluated(), 100_000);
    assert_eq!(result.evidence().len(), 100);
    assert!(result.coverage().truncated());
    assert_eq!(
        result.evidence().first(),
        Some(&wenlan_types::lint::LintEvidenceRef::OpaqueId {
            opaque_id: LintOpaqueId::from_sorted_position(0).unwrap(),
        })
    );
    assert!(!serde_json::to_string(&result)
        .unwrap()
        .contains("private-locator"));
}

#[tokio::test]
async fn set_query_matches_writer_precedence_and_pages_workspace_scope() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_memory(
        &conn,
        "row-source-id",
        "file::shared",
        "memory",
        Some("folder"),
    )
    .await;
    insert_memory(&conn, "file::shared", "other", "authored", None).await;
    insert_memory(&conn, "authored-id", "authored-source", "authored", None).await;
    insert_memory(
        &conn,
        "folder-id-locator",
        "folder::canonical",
        "memory",
        Some("folder"),
    )
    .await;
    insert_memory(
        &conn,
        "url-id-locator",
        "https://canonical.example",
        "webpage",
        None,
    )
    .await;
    insert_memory(
        &conn,
        "authored-id-locator",
        "authored-canonical",
        "authored",
        None,
    )
    .await;
    insert_page(&conn, "page-a", "workspace-a", None).await;
    insert_page(&conn, "page-b", "workspace-b", Some("[]")).await;
    for (page, locator, kind) in [
        ("page-a", "file::shared", "external_file"),
        ("page-a", "https://missing.example", "external_url"),
        ("page-a", "authored-source", "authored"),
        ("page-a", "folder-id-locator", "external_file"),
        ("page-a", "url-id-locator", "external_url"),
        ("page-a", "authored-id-locator", "authored"),
        ("page-a", "plain-missing", "memory"),
        ("page-b", "private-canary", "external_file"),
    ] {
        conn.execute(
            "INSERT INTO page_sources (page_id, memory_source_id, linked_at, link_reason) VALUES (?1, ?2, 1, 'test')",
            libsql::params![page, locator],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO page_evidence (page_id, source_kind, locator, linked_at, link_reason) VALUES (?1, ?2, ?3, 1, 'test')",
            libsql::params![page, kind, locator],
        )
        .await
        .unwrap();
    }
    conn.execute(
        "INSERT INTO page_evidence (page_id, source_kind, locator, linked_at, link_reason) VALUES ('page-a', 'memory', NULL, 1, 'inventory')",
        (),
    )
    .await
    .unwrap();

    let requested = vec![
        "file::shared".to_string(),
        "https://missing.example".to_string(),
        "authored-source".to_string(),
        "folder-id-locator".to_string(),
        "url-id-locator".to_string(),
        "authored-id-locator".to_string(),
        "plain-missing".to_string(),
    ];
    let writer = db.resolve_source_kinds(&requested).await.unwrap();
    let snapshot = db.open_lint_snapshot().await.unwrap();
    let scope = AppliedScope::registered(
        LintOpaqueId::from_sorted_position(0).unwrap(),
        "workspace-a".to_string(),
    );
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let context = LintContext::new(
        &snapshot,
        &scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let records = load_sources(&context).await.unwrap();
    assert_eq!(records.len(), 7);
    let lint = records
        .iter()
        .map(|record| (record.locator.as_str(), record.expected_kind.as_str()))
        .collect::<BTreeMap<_, _>>();
    for locator in requested {
        assert_eq!(
            lint.get(locator.as_str()).copied(),
            writer.get(&locator).map(String::as_str)
        );
    }
    assert!(!serde_json::to_string(
        &assess_sources(&records, &[])
            .result(SOURCE_COVERAGE_ID, 0)
            .unwrap()
    )
    .unwrap()
    .contains("private-canary"));
    let selected_citations = load_and_assess_citations(&context)
        .await
        .unwrap()
        .result(CITATION_PARTITIONS_ID, 0)
        .unwrap();
    assert_eq!(selected_citations.coverage().denominator(), 1);

    let global_scope = AppliedScope::global();
    let global_context = LintContext::new(
        &snapshot,
        &global_scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let global_records = load_sources(&global_context).await.unwrap();
    assert_eq!(global_records.len(), 8);
    let global_result = assess_sources(&global_records, &[])
        .result(SOURCE_COVERAGE_ID, 0)
        .unwrap();
    assert_eq!(global_result.outcome(), LintOutcome::Finding);
    assert_eq!(global_result.severity(), LintSeverity::Error);
    let global_citations = load_and_assess_citations(&global_context)
        .await
        .unwrap()
        .result(CITATION_PARTITIONS_ID, 0)
        .unwrap();
    assert_eq!(global_citations.coverage().denominator(), 2);
}

#[tokio::test]
async fn matching_evidence_cannot_mask_missing_logical_or_row_id_owner() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_memory(&conn, "mem-owner-row", "owner-logical-id", "memory", None).await;
    insert_page(&conn, "page-owner-check", "workspace-a", None).await;
    for locator in [
        "owner-logical-id",
        "mem-owner-row",
        "missing-logical-owner",
        "mem-missing-row-owner",
    ] {
        conn.execute(
            "INSERT INTO page_sources (page_id, memory_source_id, linked_at, link_reason)
             VALUES ('page-owner-check', ?1, 1, 'test')",
            libsql::params![locator],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO page_evidence (page_id, source_kind, locator, linked_at, link_reason)
             VALUES ('page-owner-check', 'memory', ?1, 1, 'test')",
            libsql::params![locator],
        )
        .await
        .unwrap();
    }

    let snapshot = db.open_lint_snapshot().await.unwrap();
    let scope = AppliedScope::global();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let context = LintContext::new(
        &snapshot,
        &scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let result = super::source::load_and_assess_sources(&context)
        .await
        .unwrap()
        .result(SOURCE_COVERAGE_ID, 0)
        .unwrap();

    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Error);
    assert_eq!(result.coverage().denominator(), 4);
    assert_eq!(result.coverage().evaluated(), 4);
    assert_eq!(metric(&result, LintMetricCode::AffectedRecords), 2);
    assert_eq!(result.evidence().len(), 2);
}

#[tokio::test]
async fn uncategorized_scope_finds_unfiled_page_and_excludes_registered() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_memory(&conn, "mem-unfiled", "mem-unfiled", "memory", None).await;
    insert_memory(&conn, "mem-alpha", "mem-alpha", "memory", None).await;
    insert_page(
        &conn,
        "page-unfiled",
        crate::db::UNFILED_SPACE_ID,
        Some("[]"),
    )
    .await;
    insert_page(&conn, "page-CANARY-alpha", "alpha", Some("[]")).await;
    for (page, locator) in [
        ("page-unfiled", "mem-unfiled"),
        ("page-CANARY-alpha", "mem-alpha"),
    ] {
        conn.execute(
            "INSERT INTO page_sources (page_id, memory_source_id, linked_at, link_reason) VALUES (?1, ?2, 1, 'test')",
            libsql::params![page, locator],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO page_evidence (page_id, source_kind, locator, linked_at, link_reason) VALUES (?1, 'memory', ?2, 1, 'test')",
            libsql::params![page, locator],
        )
        .await
        .unwrap();
    }

    let snapshot = db.open_lint_snapshot().await.unwrap();
    let scope = AppliedScope::uncategorized();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let context = LintContext::new(
        &snapshot,
        &scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    // Shape-C regression guard: `workspace IS NULL` was permanently dead
    // after migration 80's NOT NULL stamp, so a revert here would make both
    // checks vacuously empty rather than actually excluding page-CANARY-alpha.
    let records = load_sources(&context).await.unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].locator, "mem-unfiled");

    let citations = load_and_assess_citations(&context)
        .await
        .unwrap()
        .result(CITATION_PARTITIONS_ID, 0)
        .unwrap();
    assert_eq!(citations.coverage().denominator(), 1);
}

fn metric(result: &LintCheckResult, code: LintMetricCode) -> u64 {
    result
        .metrics()
        .iter()
        .find(|metric| metric.code() == code)
        .and_then(|metric| match metric.value() {
            LintMetricValue::Count { value } => Some(*value),
            LintMetricValue::Boolean { .. } | LintMetricValue::CatalogCode { .. } => None,
        })
        .unwrap()
}

async fn insert_memory(
    conn: &TestDbSession,
    id: &str,
    source_id: &str,
    source_name: &str,
    source_agent: Option<&str>,
) {
    conn.execute(
        "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, source_agent) VALUES (?1, 'body', ?2, ?3, ?1, 0, 1, 'text', ?4)",
        libsql::params![id, source_name, source_id, source_agent],
    )
    .await
    .unwrap();
}

async fn insert_page(conn: &TestDbSession, id: &str, workspace: &str, citations: Option<&str>) {
    conn.execute(
        "INSERT INTO pages (id, title, content, source_memory_ids, version, status, created_at, last_compiled, last_modified, workspace, citations) VALUES (?1, ?1, 'body', '[]', 1, 'active', 'now', 'now', 'now', ?2, ?3)",
        libsql::params![id, workspace, citations],
    )
    .await
    .unwrap();
}
