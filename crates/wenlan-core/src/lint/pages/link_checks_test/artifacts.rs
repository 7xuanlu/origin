use super::*;

#[tokio::test]
async fn optional_artifacts_are_neutral_but_deterministic_broken_targets_warn() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_secondary_session().unwrap();
    insert_page(&conn, "page-active", None, "active").await;
    insert_page(&conn, "page-target", None, "active").await;
    insert_page(&conn, "page-archived-target", None, "archived").await;
    conn.execute(
        "INSERT INTO page_links (source_page_id, target_page_id, label_key, label) \
         VALUES ('page-active', 'page-target', 'resolved', 'resolved'), \
                ('page-active', 'page-missing', 'missing', 'missing'), \
                ('page-active', 'page-archived-target', 'archived', 'archived'), \
                ('page-active', NULL, 'unresolved', 'unresolved')",
        (),
    )
    .await
    .unwrap();
    let root = TempDir::new().unwrap();
    write(root.path(), "purpose.md", b"purpose");
    write(root.path(), "wiki/_index.md", b"index");
    write(root.path(), "wiki/Overview.md", b"overview");
    write(root.path(), "_sources/mem-one.md", b"stub");
    let scan = scan_page_root(root.path()).unwrap();
    let counts = ArtifactCounts::from_scan(&scan);
    assert_eq!(counts.purpose, 1);
    assert_eq!(counts.schema, 0);
    assert_eq!(counts.index, 1);
    assert_eq!(counts.log, 0);
    assert_eq!(counts.overview, 1);
    assert_eq!(counts.source_stubs, 1);

    let snapshot = db.open_lint_snapshot().await.unwrap();
    let scope = AppliedScope::global();
    let clock = LintClock::fixed();
    let gate = ExecutionGate::new(CancellationToken::new());
    let context = LintContext::new(
        &snapshot,
        &scope,
        Some(&scan),
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let result = load_artifacts(&context)
        .await
        .unwrap()
        .result(ARTIFACT_ID, 0)
        .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Warning);
    assert_eq!(
        metric_value(&result, LintMetricCode::ProjectArchiveRecords),
        1
    );
    assert_eq!(
        metric_value(&result, LintMetricCode::ProjectOutboundLinks),
        4
    );
    assert_eq!(
        metric_value(&result, LintMetricCode::ProjectInboundLinks),
        1
    );
    assert_eq!(metric_value(&result, LintMetricCode::ProjectBrokenLinks), 2);
}
