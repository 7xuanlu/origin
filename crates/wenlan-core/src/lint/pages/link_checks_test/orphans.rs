use super::*;

#[tokio::test]
async fn missing_same_scope_target_occurrence_is_inventory_not_a_finding() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page(&conn, "page-source", Some("workspace-a"), "active").await;
    insert_page(&conn, "page-other-scope", Some("workspace-b"), "active").await;
    set_page_title(&conn, "page-other-scope", "topic").await;
    insert_orphan(&conn, "page-source", "topic").await;

    let result = global_orphan_result(&db).await;

    assert_eq!(result.outcome(), LintOutcome::Pass);
    assert_eq!(result.applicability(), LintApplicability::Inventory);
    assert_eq!(result.coverage().denominator(), 1);
    assert_eq!(result.coverage().evaluated(), 1);
    assert!(result.evidence().is_empty());
}

#[tokio::test]
async fn unique_but_unbound_occurrence_remains_a_finding() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page(&conn, "page-source", Some("workspace-a"), "active").await;
    insert_page(&conn, "page-target", Some("workspace-a"), "active").await;
    set_page_title(&conn, "page-target", "topic").await;
    insert_orphan(&conn, "page-source", "topic").await;

    let result = global_orphan_result(&db).await;

    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Warning);
    assert_eq!(result.coverage().denominator(), 1);
    assert_eq!(result.evidence().len(), 1);
}

#[tokio::test]
async fn missing_target_occurrence_does_not_add_evidence_to_an_actionable_finding() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page(&conn, "page-missing-source", Some("workspace-a"), "active").await;
    insert_page(&conn, "page-unbound-source", Some("workspace-a"), "active").await;
    insert_page(&conn, "page-target", Some("workspace-a"), "active").await;
    set_page_title(&conn, "page-target", "unique").await;
    insert_orphan(&conn, "page-missing-source", "missing").await;
    insert_orphan(&conn, "page-unbound-source", "unique").await;

    let result = global_orphan_result(&db).await;

    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.coverage().denominator(), 2);
    assert_eq!(result.evidence().len(), 1);
}

#[tokio::test]
async fn ambiguous_unbound_occurrence_remains_a_finding() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page(&conn, "page-source", Some("workspace-a"), "active").await;
    for target in ["page-target-a", "page-target-b"] {
        insert_page(&conn, target, Some("workspace-a"), "active").await;
        set_page_title(&conn, target, "topic").await;
    }
    insert_orphan(&conn, "page-source", "topic").await;

    let result = global_orphan_result(&db).await;

    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Warning);
    assert_eq!(result.coverage().denominator(), 1);
    assert_eq!(result.evidence().len(), 1);
}

#[tokio::test]
async fn coverage_counts_occurrences_while_metric_counts_distinct_labels() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    for source in ["page-source-a", "page-source-b", "page-source-c"] {
        insert_page(&conn, source, Some("workspace-a"), "active").await;
    }
    insert_orphan(&conn, "page-source-a", "repeated").await;
    insert_orphan(&conn, "page-source-b", "repeated").await;
    insert_orphan(&conn, "page-source-c", "other").await;

    let result = global_orphan_result(&db).await;

    assert_eq!(result.coverage().denominator(), 3);
    assert_eq!(result.coverage().evaluated(), 3);
    assert_eq!(metric_value(&result, LintMetricCode::PageOrphanLabels), 2);
}

#[tokio::test]
async fn totals_and_truncation_are_exact_at_zero_hundred_and_101() {
    for (count, truncated) in [(0_u64, false), (100, false), (101, true)] {
        let (db, _tmp) = test_db().await;
        let conn = db.test_secondary_session().unwrap();
        insert_page(&conn, "page-source", Some("workspace-a"), "active").await;
        for ordinal in 0..count {
            let label = format!("private-label-{ordinal:03}");
            insert_page(&conn, &label, Some("workspace-a"), "active").await;
            conn.execute(
                "INSERT INTO page_links (source_page_id, target_page_id, label_key, label) \
                 VALUES ('page-source', NULL, ?1, ?1)",
                libsql::params![label],
            )
            .await
            .unwrap();
        }
        let before = link_row_count(&conn).await;
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
        let result = load_orphans(&context)
            .await
            .unwrap()
            .result(ORPHAN_LABELS_ID, 0)
            .unwrap();
        assert_eq!(result.coverage().denominator(), count);
        assert_eq!(result.coverage().evaluated(), count);
        assert_eq!(result.coverage().truncated(), truncated);
        assert_eq!(
            result.evidence().len(),
            usize::try_from(count.min(100)).unwrap()
        );
        assert_eq!(
            result.outcome(),
            if count == 0 {
                LintOutcome::Pass
            } else {
                LintOutcome::Finding
            }
        );
        if count > 0 {
            assert_eq!(result.severity(), LintSeverity::Warning);
        }
        assert_eq!(
            metric_value(&result, LintMetricCode::PageOrphanLabels),
            count
        );
        assert!(!serde_json::to_string(&result)
            .unwrap()
            .contains("private-label"));
        drop(snapshot);
        assert_eq!(link_row_count(&conn).await, before);
    }
}

#[tokio::test]
async fn scope_is_anchored_to_active_source_workspace() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page(&conn, "page-a", Some("workspace-a"), "active").await;
    insert_page(&conn, "page-b", Some("workspace-b"), "active").await;
    insert_page(&conn, "page-archived", Some("workspace-a"), "archived").await;
    for (page, label) in [
        ("page-a", "private-a"),
        ("page-b", "private-b"),
        ("page-archived", "private-archived"),
    ] {
        conn.execute(
            "INSERT INTO page_links (source_page_id, target_page_id, label_key, label) \
             VALUES (?1, NULL, ?2, ?2)",
            libsql::params![page, label],
        )
        .await
        .unwrap();
    }
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
    let result = load_orphans(&context)
        .await
        .unwrap()
        .result(ORPHAN_LABELS_ID, 0)
        .unwrap();
    assert_eq!(result.coverage().denominator(), 1);
}

#[tokio::test]
async fn selected_orphan_scope_uses_legacy_space_fallback() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page(&conn, "page-legacy-source", None, "active").await;
    set_page_legacy_space(&conn, "page-legacy-source", "workspace-a").await;
    insert_page(&conn, "page-target", Some("workspace-a"), "active").await;
    set_page_title(&conn, "page-target", "topic").await;
    insert_orphan(&conn, "page-legacy-source", "topic").await;

    let registered_scope = AppliedScope::registered(
        LintOpaqueId::from_sorted_position(0).unwrap(),
        "workspace-a".to_string(),
    );
    let registered = scoped_orphan_result(&db, &registered_scope).await;
    assert_eq!(registered.outcome(), LintOutcome::Finding);
    assert_eq!(registered.coverage().denominator(), 1);
    assert_eq!(registered.evidence().len(), 1);
    assert_eq!(
        metric_value(&registered, LintMetricCode::PageOrphanLabels),
        1
    );

    let uncategorized_scope = AppliedScope::uncategorized();
    let uncategorized = scoped_orphan_result(&db, &uncategorized_scope).await;
    assert_eq!(uncategorized.outcome(), LintOutcome::Pass);
    assert_eq!(uncategorized.applicability(), LintApplicability::Inventory);
    assert_eq!(uncategorized.coverage().denominator(), 0);
    assert!(uncategorized.evidence().is_empty());
    assert_eq!(
        metric_value(&uncategorized, LintMetricCode::PageOrphanLabels),
        0
    );
}

#[tokio::test]
async fn query_failure_propagates_without_a_mutating_fallback() {
    let dir = tempfile::tempdir().unwrap();
    let database = libsql::Builder::new_local(dir.path().join("empty.db"))
        .build()
        .await
        .unwrap();
    let snapshot = LintReadSnapshot::open(&database).await.unwrap();
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
    assert!(load_orphans(&context).await.is_err());
}

async fn global_orphan_result(db: &crate::db::MemoryDB) -> LintCheckResult {
    let scope = AppliedScope::global();
    scoped_orphan_result(db, &scope).await
}

async fn scoped_orphan_result(db: &crate::db::MemoryDB, scope: &AppliedScope) -> LintCheckResult {
    let snapshot = db.open_lint_snapshot().await.unwrap();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let context = LintContext::new(
        &snapshot,
        scope,
        None,
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    load_orphans(&context)
        .await
        .unwrap()
        .result(ORPHAN_LABELS_ID, 0)
        .unwrap()
}

async fn set_page_legacy_space(conn: &TestDbSession, page_id: &str, space: &str) {
    conn.execute(
        "UPDATE pages SET space = ?2 WHERE id = ?1",
        libsql::params![page_id, space],
    )
    .await
    .unwrap();
}

async fn insert_orphan(conn: &TestDbSession, source_page_id: &str, label: &str) {
    conn.execute(
        "INSERT INTO page_links (source_page_id, target_page_id, label_key, label) \
         VALUES (?1, NULL, ?2, ?2)",
        libsql::params![source_page_id, label],
    )
    .await
    .unwrap();
}

async fn set_page_title(conn: &TestDbSession, page_id: &str, title: &str) {
    conn.execute(
        "UPDATE pages SET title = ?2 WHERE id = ?1",
        libsql::params![page_id, title],
    )
    .await
    .unwrap();
}
