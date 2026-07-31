use super::*;
use crate::db::test_support::TestDbSession;
use crate::db::tests::test_db;
use crate::lint::context::{
    AppliedScope, CancellationToken, ExecutionGate, LintClock, LintContext,
};
use wenlan_types::lint::{LintApplicability, LintOpaqueId, LintOutcome, LintSeverity};

fn page(title: &str, status: &str, creation: &str, review: &str) -> PageRow {
    PageRow {
        title_key: title.to_lowercase(),
        effective_scope: None,
        status: status.to_string(),
        creation_kind: creation.to_string(),
        review_status: review.to_string(),
    }
}

#[test]
fn cartesian_partitions_and_all_unconfirmed_kinds_are_exact_inventory() {
    let mut rows = Vec::new();
    for status in STATUSES {
        for creation in CREATION_KINDS {
            for review in REVIEW_STATUSES {
                rows.push(page(
                    &format!("{status}-{creation}-{review}"),
                    status,
                    creation,
                    review,
                ));
            }
        }
    }
    let partitions = Partitions::from_rows(&rows);
    assert!(partitions.is_exact(20));
    assert!(!partitions.is_exact(19));
    let partition_result = assess_partitions(&rows).result(PARTITIONS_ID, 0).unwrap();
    assert_eq!(partition_result.outcome(), LintOutcome::Pass);
    assert_eq!(partition_result.coverage().denominator(), 20);
    let review_result = assess_review(&rows).result(REVIEW_ID, 0).unwrap();
    assert_eq!(review_result.outcome(), LintOutcome::Pass);
    assert_eq!(review_result.applicability(), LintApplicability::Inventory);
    assert_eq!(review_result.coverage().denominator(), 20);
    for creation in CREATION_KINDS {
        let one = [page("private", "active", creation, "unconfirmed")];
        assert_eq!(
            assess_review(&one).result(REVIEW_ID, 0).unwrap().outcome(),
            LintOutcome::Pass
        );
    }
}

#[test]
fn unknown_storage_values_are_error_findings() {
    for row in [
        page("a", "future", "distilled", "confirmed"),
        page("b", "active", "future", "confirmed"),
        page("c", "active", "distilled", "future"),
    ] {
        let partition = assess_partitions(std::slice::from_ref(&row))
            .result(PARTITIONS_ID, 0)
            .unwrap();
        assert_eq!(partition.outcome(), LintOutcome::Finding);
        assert_eq!(partition.severity(), LintSeverity::Error);
    }
}

#[test]
fn active_duplicates_warn_but_archived_rows_remain_inventory() {
    let rows = [
        page("same", "active", "distilled", "confirmed"),
        page("SAME", "active", "authored", "unconfirmed"),
        page("same", "archived", "research", "unconfirmed"),
    ];
    let duplicates = assess_duplicates(&rows)
        .result(DUPLICATE_TITLES_ID, 0)
        .unwrap();
    assert_eq!(duplicates.outcome(), LintOutcome::Finding);
    assert_eq!(duplicates.severity(), LintSeverity::Warning);
    assert_eq!(duplicates.coverage().denominator(), 2);
    let archive = assess_archive(&rows).result(ARCHIVE_ID, 0).unwrap();
    assert_eq!(archive.outcome(), LintOutcome::Pass);
    assert_eq!(archive.applicability(), LintApplicability::Inventory);
    assert_eq!(archive.coverage().denominator(), 1);
}

#[tokio::test]
async fn cross_effective_scope_active_duplicate_titles_pass() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page_with_scope(
        &conn,
        "page-a",
        "Shared Title",
        "active",
        Some("workspace-a"),
    )
    .await;
    insert_page_with_scope(
        &conn,
        "page-b",
        "shared title",
        "active",
        Some("workspace-b"),
    )
    .await;
    let snapshot = db.open_lint_snapshot().await.unwrap();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let global_scope = AppliedScope::global();
    let global_context = LintContext::new(
        &snapshot,
        &global_scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let global = load_rows(&global_context).await.unwrap();
    let global_result = assess_duplicates(&global)
        .result(DUPLICATE_TITLES_ID, 0)
        .unwrap();
    assert_eq!(global_result.outcome(), LintOutcome::Pass);
    assert_eq!(global_result.coverage().denominator(), 2);
}

#[tokio::test]
async fn same_effective_scope_active_duplicate_titles_warn() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page_with_scope(
        &conn,
        "page-a",
        "Shared Title",
        "active",
        Some("workspace-a"),
    )
    .await;
    insert_page_with_scope(
        &conn,
        "page-b",
        "shared title",
        "active",
        Some("workspace-a"),
    )
    .await;
    let snapshot = db.open_lint_snapshot().await.unwrap();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let global_scope = AppliedScope::global();
    let global_context = LintContext::new(
        &snapshot,
        &global_scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let global = load_rows(&global_context).await.unwrap();
    let global_result = assess_duplicates(&global)
        .result(DUPLICATE_TITLES_ID, 0)
        .unwrap();
    assert_eq!(global_result.outcome(), LintOutcome::Finding);
    assert_eq!(global_result.severity(), LintSeverity::Warning);
    assert_eq!(global_result.coverage().denominator(), 2);
}

#[tokio::test]
// M1 honest columns retired the workspace/space fallback this test used
// to probe (renamed from selected_duplicate_scope_uses_legacy_space_fallback);
// the surviving assertion is that Registered/Uncategorized scope filters
// still partition correctly now that both pages carry `space` directly.
async fn selected_duplicate_scope_filters_registered_and_uncategorized() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page_with_scope(
        &conn,
        "page-current",
        "Shared Title",
        "active",
        Some("workspace-a"),
    )
    .await;
    insert_page_with_scope(
        &conn,
        "page-other",
        "shared title",
        "active",
        Some("workspace-a"),
    )
    .await;
    insert_page_with_scope(&conn, "page-uncategorized", "shared title", "active", None).await;
    let snapshot = db.open_lint_snapshot().await.unwrap();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());

    let registered_scope = AppliedScope::registered(
        LintOpaqueId::from_sorted_position(0).unwrap(),
        "workspace-a".to_string(),
    );
    let registered_context = LintContext::new(
        &snapshot,
        &registered_scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let registered = assess_duplicates(&load_rows(&registered_context).await.unwrap())
        .result(DUPLICATE_TITLES_ID, 0)
        .unwrap();
    assert_eq!(registered.outcome(), LintOutcome::Finding);
    assert_eq!(registered.coverage().denominator(), 2);

    let uncategorized_scope = AppliedScope::uncategorized();
    let uncategorized_context = LintContext::new(
        &snapshot,
        &uncategorized_scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let uncategorized = assess_duplicates(&load_rows(&uncategorized_context).await.unwrap())
        .result(DUPLICATE_TITLES_ID, 0)
        .unwrap();
    assert_eq!(uncategorized.outcome(), LintOutcome::Pass);
    assert_eq!(uncategorized.coverage().denominator(), 1);
    assert!(uncategorized.evidence().is_empty());
}

// M1 honest columns: `workspace` and `space` are NOT NULL and always
// mirrored (migration 80), so this seeds one logical scope into both --
// `None` means unfiled/uncategorized, matching the migration's own
// sentinel rather than binding a literal NULL a NOT NULL column rejects.
async fn insert_page_with_scope(
    conn: &TestDbSession,
    id: &str,
    title: &str,
    status: &str,
    scope: Option<&str>,
) {
    let scope = scope.unwrap_or(crate::db::UNFILED_SPACE_ID);
    conn.execute(
        "INSERT INTO pages (id, title, content, source_memory_ids, version, status, created_at, last_compiled, last_modified, workspace, space, creation_kind, review_status) VALUES (?1, ?2, 'body', '[]', 1, ?3, 'now', 'now', 'now', ?4, ?5, 'distilled', 'confirmed')",
        libsql::params![id, title, status, scope, scope],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn source_page_integrity_accepts_any_canonical_or_legacy_provenance_representation() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    conn.execute_batch(
        "INSERT INTO pages
             (id,title,content,source_memory_ids,version,status,created_at,last_compiled,
              last_modified,creation_kind,review_status,user_edited)
         VALUES
             ('source_invalid','invalid','', '[]',1,'active','now','now','now',
              'source','unconfirmed',0),
             ('source_json','json','', '[\"mem-json\"]',1,'active','now','now','now',
              'source','unconfirmed',0),
             ('source_join','join','', '[]',1,'active','now','now','now',
              'source','unconfirmed',0),
             ('source_evidence','evidence','', '[]',1,'active','now','now','now',
              'source','unconfirmed',0),
             ('source_archived','archived','', '[]',1,'archived','now','now','now',
              'source','unconfirmed',0),
             ('distilled_active','distilled','', '[]',1,'active','now','now','now',
              'distilled','unconfirmed',0);
         INSERT INTO page_sources(page_id,memory_source_id,linked_at,link_reason)
         VALUES ('source_join','mem-join',1,'legacy');
         INSERT INTO page_evidence(page_id,source_kind,locator,linked_at,link_reason)
         VALUES ('source_evidence','external_url',NULL,1,'canonical');",
    )
    .await
    .unwrap();
    let snapshot = db.open_lint_snapshot().await.unwrap();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let scope = AppliedScope::global();
    let context = LintContext::new(
        &snapshot,
        &scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );

    let result = load_and_assess_source_integrity(&context)
        .await
        .unwrap()
        .result(SOURCE_INTEGRITY_ID, 0)
        .unwrap();

    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Error);
    assert_eq!(result.coverage().denominator(), 4);
    assert_eq!(result.coverage().evaluated(), 4);
    assert_eq!(result.evidence().len(), 1);
}

#[tokio::test]
async fn source_page_integrity_runs_without_page_projection_or_page_root() {
    let (db, _tmp) = test_db().await;
    db.test_secondary_session()
        .unwrap()
        .execute_batch(
            "INSERT INTO pages
                 (id,title,content,source_memory_ids,version,status,created_at,last_compiled,
                  last_modified,creation_kind,review_status,user_edited)
             VALUES
                 ('source_invalid','invalid','', '[]',1,'active','now','now','now',
                  'source','unconfirmed',0);",
        )
        .await
        .unwrap();

    let report = crate::lint::runner::LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &wenlan_types::lint::LintQuery::new(
                Some(wenlan_types::lint::LintProfile::General),
                None,
            ),
            None,
            false,
        )
        .await
        .unwrap();
    let source = report
        .checks()
        .iter()
        .find(|check| check.check_id() == SOURCE_INTEGRITY_ID)
        .unwrap();
    assert_eq!(source.outcome(), LintOutcome::Finding);
    assert_eq!(source.coverage().denominator(), 1);
}
