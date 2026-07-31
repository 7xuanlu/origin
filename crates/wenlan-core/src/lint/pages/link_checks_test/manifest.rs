use super::*;

#[test]
fn divergence_is_inventory_but_invalid_json_is_error() {
    let root = TempDir::new().unwrap();
    write(
        root.path(),
        "_sources/.manifest.json",
        br#"{"pages":{"page-private":["mem-expected"]}}"#,
    );
    write(root.path(), "_sources/mem-extra.md", b"stub");
    let scan = scan_page_root(root.path()).unwrap();
    let result = assess_manifest(&scan, false)
        .result(MANIFEST_ID, 0)
        .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Pass);
    assert_eq!(result.applicability(), LintApplicability::Inventory);
    assert_eq!(metric_value(&result, LintMetricCode::PageManifestPages), 1);
    assert_eq!(
        metric_value(&result, LintMetricCode::PageManifestDivergences),
        2
    );
    assert!(!serde_json::to_string(&result).unwrap().contains("private"));

    write(root.path(), "_sources/.manifest.json", b"{");
    let invalid = scan_page_root(root.path()).unwrap();
    let result = assess_manifest(&invalid, false)
        .result(MANIFEST_ID, 0)
        .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Error);
}

#[cfg(unix)]
#[test]
fn source_symlink_escape_is_a_containment_error() {
    use std::os::unix::fs::symlink;

    let root = TempDir::new().unwrap();
    fs::create_dir_all(root.path().join("_sources")).unwrap();
    symlink(
        "../../private-outside",
        root.path().join("_sources/escape.md"),
    )
    .unwrap();
    let scan = scan_page_root(root.path()).unwrap();
    let result = assess_manifest(&scan, false)
        .result(MANIFEST_ID, 0)
        .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Error);
    assert!(!serde_json::to_string(&result)
        .unwrap()
        .contains("private-outside"));
}

#[test]
fn oversized_manifest_is_bounded_and_classified_as_invalid() {
    let root = TempDir::new().unwrap();
    write(
        root.path(),
        "_sources/.manifest.json",
        &vec![b'x'; 1024 * 1024 + 1],
    );
    let scan = scan_page_root(root.path()).unwrap();
    let result = assess_manifest(&scan, false)
        .result(MANIFEST_ID, 0)
        .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Error);
}

#[cfg(unix)]
#[tokio::test]
async fn selected_runner_keeps_global_errors_without_evidence() {
    use std::os::unix::fs::symlink;

    let (db, _tmp) = test_db().await;
    db.test_primary_session()
        .await
        .execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) \
             VALUES ('lint-space', 'alpha', 1, 1)",
            (),
        )
        .await
        .unwrap();
    let root = TempDir::new().unwrap();
    fs::create_dir_all(root.path().join("_sources")).unwrap();
    symlink(
        "../../private-outside",
        root.path().join("_sources/escape.md"),
    )
    .unwrap();
    let report = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: Some("alpha".to_string()),
            },
            Some(root.path()),
            true,
        )
        .await
        .unwrap();
    let manifest = report
        .checks()
        .iter()
        .find(|check| check.check_id() == MANIFEST_ID)
        .unwrap();
    assert_eq!(manifest.outcome(), LintOutcome::Finding);
    assert_eq!(manifest.severity(), LintSeverity::Error);
    assert!(manifest.evidence().is_empty());
}
