use super::*;
use crate::db::tests::test_db;
use crate::lint::context::{AppliedScope, CancellationToken, ExecutionGate, LintClock};
use crate::lint::pages::fs::scan_page_root;
use crate::lint::runner::LintRunner;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn root() -> TempDir {
    tempfile::tempdir().unwrap()
}

fn write(root: &Path, relative: &str, contents: &str) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

fn scan(state: Option<&str>, files: &[(&str, &str)]) -> (TempDir, PageScan) {
    let root = root();
    if let Some(state) = state {
        write(root.path(), ".wenlan/state.json", state);
    }
    for (path, contents) in files {
        write(root.path(), path, contents);
    }
    let scanned = scan_page_root(root.path()).unwrap();
    (root, scanned)
}

fn page(id: &str, status: &str, version: i64) -> DbPage {
    DbPage {
        id: id.to_string(),
        status: status.to_string(),
        version,
    }
}

fn outcome(assessment: Assessment) -> (LintOutcome, LintSeverity, u64, usize) {
    let result = assessment.result("pages.test", 0).unwrap();
    (
        result.outcome(),
        result.severity(),
        result.coverage().denominator(),
        result.evidence().len(),
    )
}

#[test]
fn state_shapes_follow_the_exact_outcome_mapping() {
    let cases = [
        (None, LintOutcome::Pass, LintSeverity::Info),
        (
            Some("{\"schema_version\":0,\"pages\":{}}"),
            LintOutcome::Finding,
            LintSeverity::Warning,
        ),
        (
            Some("{\"concepts\":{}}"),
            LintOutcome::Finding,
            LintSeverity::Warning,
        ),
        (
            Some("{\"pages\":{}}"),
            LintOutcome::Finding,
            LintSeverity::Warning,
        ),
        (
            Some("{\"schema_version\":2,\"pages\":{}}"),
            LintOutcome::Pass,
            LintSeverity::Info,
        ),
        (
            Some("{\"schema_version\":3,\"pages\":{}}"),
            LintOutcome::NotRunPrerequisite,
            LintSeverity::Error,
        ),
        (
            Some("{\"schema_version\":\"2\",\"pages\":{}}"),
            LintOutcome::Finding,
            LintSeverity::Error,
        ),
        (Some("{"), LintOutcome::Finding, LintSeverity::Error),
    ];
    for (raw, expected_outcome, expected_severity) in cases {
        let (_root, scan) = scan(raw, &[]);
        let (actual_outcome, actual_severity, _, _) = outcome(evaluate_state(&scan, None));
        assert_eq!(actual_outcome, expected_outcome);
        assert_eq!(actual_severity, expected_severity);
    }
}

#[test]
fn missing_state_with_recognized_projection_is_error() {
    let (_root, scan) = scan(
        None,
        &[(
            "projected.md",
            "---\norigin_id: page_projected\norigin_version: 1\n---\nbody\n",
        )],
    );
    let (outcome, severity, _, _) = outcome(evaluate_state(&scan, None));
    assert_eq!(outcome, LintOutcome::Finding);
    assert_eq!(severity, LintSeverity::Error);
}

#[test]
fn future_state_schema_blocks_all_state_dependent_checks() {
    let (_root, scan) = scan(Some("{\"schema_version\":99,\"pages\":{}}"), &[]);
    let selected_ids = BTreeSet::new();
    for (check_id, assessment) in evaluate_all(&scan, &[], false, &selected_ids) {
        let result = assessment.result(check_id, 0).unwrap();
        assert_eq!(result.outcome(), LintOutcome::NotRunPrerequisite);
        assert_eq!(result.severity(), LintSeverity::Error);
        assert_eq!(result.applicability(), LintApplicability::NotApplicable);
        assert_eq!(result.coverage().denominator(), 1);
        assert!(!crate::lint::runner::synthetic_report(vec![result]).complete());
    }
}

#[test]
fn invalid_reserved_missing_and_wrong_targets_are_classified() {
    let cases = [
        (
            "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"../escape.md\",\"version\":1}}}",
            Vec::new(),
            LintSeverity::Error,
        ),
        (
            "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"_sources/a.md\",\"version\":1}}}",
            vec![("_sources/a.md", "stub")],
            LintSeverity::Error,
        ),
        (
            "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"missing.md\",\"version\":1}}}",
            Vec::new(),
            LintSeverity::Warning,
        ),
        (
            "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"a.md\",\"version\":1}}}",
            vec![(
                "a.md",
                "---\norigin_id: page_b\norigin_version: 1\n---\nbody\n",
            )],
            LintSeverity::Error,
        ),
    ];
    for (raw, files, expected) in cases {
        let (_root, scan) = scan(Some(raw), &files);
        let (_, severity, _, _) = outcome(evaluate_identity(
            &scan,
            &[page("page_a", "active", 1)],
            false,
        ));
        assert_eq!(severity, expected);
    }
}

#[test]
fn legacy_duplicate_and_active_archived_partitions_map_correctly() {
    let state = "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"a.md\",\"version\":1},\"concept_a\":{\"file\":\"a.md\",\"version\":1}}}";
    let (_root, scan) = scan(
        Some(state),
        &[(
            "a.md",
            "---\norigin_id: page_a\norigin_version: 1\n---\nbody\n",
        )],
    );
    let (outcome, severity, _, _) = outcome(evaluate_identity(
        &scan,
        &[
            page("page_a", "active", 1),
            page("page_archived", "archived", 1),
        ],
        false,
    ));
    assert_eq!(outcome, LintOutcome::Finding);
    assert_eq!(severity, LintSeverity::Warning);
}

#[test]
fn duplicate_state_targets_are_errors() {
    let state = "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"same.md\",\"version\":1},\"page_b\":{\"file\":\"same.md\",\"version\":1}}}";
    let (_root, scan) = scan(
        Some(state),
        &[(
            "same.md",
            "---\norigin_id: page_a\norigin_version: 1\n---\n",
        )],
    );
    let (_, severity, _, _) = outcome(evaluate_identity(
        &scan,
        &[page("page_a", "active", 1), page("page_b", "active", 1)],
        false,
    ));
    assert_eq!(severity, LintSeverity::Error);
}

#[cfg(unix)]
#[test]
fn selected_state_target_crossing_symlink_is_error() {
    use std::os::unix::fs::symlink;

    let page_root = root();
    let outside = root();
    write(
        outside.path(),
        "page.md",
        "---\norigin_id: page_a\norigin_version: 1\n---\n",
    );
    fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    fs::write(
        page_root.path().join(".wenlan/state.json"),
        "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"linked/page.md\",\"version\":1}}}",
    )
    .unwrap();
    symlink(outside.path(), page_root.path().join("linked")).unwrap();
    let scan = scan_page_root(page_root.path()).unwrap();
    let (_, severity, _, _) = outcome(evaluate_identity(
        &scan,
        &[page("page_a", "active", 1)],
        true,
    ));
    assert_eq!(severity, LintSeverity::Error);
}

#[test]
fn duplicate_and_invalid_frontmatter_are_errors_but_missing_is_inventory() {
    let (_root, scan) = scan(
        Some("{\"schema_version\":2,\"pages\":{}}"),
        &[
            ("a.md", "---\norigin_id: page_a\n---\n"),
            ("b.md", "---\norigin_id: page_a\n---\n"),
            ("manual.md", "manual body"),
            ("bad.md", "---\norigin_id: [\n---\n"),
            ("_sources/private.md", "---\norigin_id: page_a\n---\n"),
        ],
    );
    let (_, severity, denominator, _) = outcome(evaluate_identity(
        &scan,
        &[page("page_a", "active", 1)],
        false,
    ));
    assert_eq!(severity, LintSeverity::Error);
    assert!(denominator >= 5);
    assert_eq!(scan.page_markdown().len(), 4);
}

#[test]
fn versions_pass_warn_inventory_and_error() {
    let cases = [
        (1, "1", 1, LintSeverity::Info),
        (1, "2", 1, LintSeverity::Warning),
        (-1, "1", 1, LintSeverity::Error),
        (1, "bad", 1, LintSeverity::Error),
    ];
    for (state_version, file_version, db_version, expected) in cases {
        let state = serde_json::json!({
            "schema_version": 2,
            "pages": {"page_a": {"file": "a.md", "version": state_version}}
        })
        .to_string();
        let markdown =
            format!("---\norigin_id: page_a\norigin_version: {file_version}\n---\nbody\n");
        let (_root, scan) = scan(Some(&state), &[("a.md", markdown.as_str())]);
        let (_, severity, _, _) = outcome(evaluate_versions(
            &scan,
            &[page("page_a", "active", db_version)],
            false,
        ));
        assert_eq!(severity, expected);
    }
}

#[test]
fn missing_file_version_does_not_hide_state_database_drift() {
    let state = serde_json::json!({
        "schema_version": 2,
        "pages": {"page_a": {"file": "a.md", "version": 1}}
    })
    .to_string();
    let (_root, scan) = scan(
        Some(&state),
        &[("a.md", "---\norigin_id: page_a\n---\nbody\n")],
    );

    let (_, severity, _, _) = outcome(evaluate_versions(
        &scan,
        &[page("page_a", "active", 2)],
        false,
    ));

    assert_eq!(severity, LintSeverity::Warning);
}

#[test]
fn selected_scope_excludes_unanchored_state_and_frontmatter_canaries() {
    let state = "{\"schema_version\":2,\"pages\":{\"page_selected\":{\"file\":\"selected.md\",\"version\":1},\"page_CANARY_OTHER\":{\"file\":\"private.md\",\"version\":9}}}";
    let (_root, scan) = scan(
        Some(state),
        &[
            (
                "selected.md",
                "---\norigin_id: page_selected\norigin_version: 1\n---\n",
            ),
            (
                "private.md",
                "---\norigin_id: page_CANARY_OTHER\norigin_version: 8\n---\n",
            ),
        ],
    );
    let selected = [page("page_selected", "active", 1)];
    let identity = evaluate_identity(&scan, &selected, true)
        .result(IDENTITY_ID, 0)
        .unwrap();
    let versions = evaluate_versions(&scan, &selected, true)
        .result(VERSION_ALIGNMENT_ID, 0)
        .unwrap();
    assert_eq!(identity.outcome(), LintOutcome::Pass);
    assert_eq!(versions.outcome(), LintOutcome::Pass);
    let output = serde_json::to_string(&(identity, versions)).unwrap();
    assert!(!output.contains("CANARY_OTHER"));
}

#[test]
fn defect_after_sample_cap_is_detected_and_output_is_deterministic() {
    let mut state_entries = String::new();
    for index in 0..101 {
        if index > 0 {
            state_entries.push(',');
        }
        state_entries.push_str(&format!(
            "\"page_{index:03}\":{{\"file\":\"missing_{index:03}.md\",\"version\":1}}"
        ));
    }
    let raw = format!("{{\"schema_version\":2,\"pages\":{{{state_entries}}}}}");
    let (_root, scan) = scan(Some(&raw), &[]);
    let first = evaluate_identity(&scan, &[], false)
        .result(IDENTITY_ID, 0)
        .unwrap();
    let second = evaluate_identity(&scan, &[], false)
        .result(IDENTITY_ID, 0)
        .unwrap();
    assert_eq!(first.outcome(), LintOutcome::Finding);
    assert_eq!(first.coverage().denominator(), 101);
    assert_eq!(first.coverage().evaluated(), 101);
    assert_eq!(first.evidence().len(), 100);
    assert!(first.coverage().truncated());
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap()
    );
}

#[test]
fn first_hundred_clean_rows_do_not_hide_defect_101() {
    let mut state_entries = serde_json::Map::new();
    let mut files = Vec::new();
    let mut pages = Vec::new();
    for index in 0..101 {
        let id = format!("page_{index:03}");
        let path = format!("page_{index:03}.md");
        state_entries.insert(id.clone(), serde_json::json!({"file": path, "version": 1}));
        files.push((
            path,
            format!(
                "---\norigin_id: {id}\norigin_version: {}\n---\n",
                if index == 100 { 2 } else { 1 }
            ),
        ));
        pages.push(page(&id, "active", 1));
    }
    let raw = serde_json::json!({"schema_version": 2, "pages": state_entries}).to_string();
    let file_refs = files
        .iter()
        .map(|(path, contents)| (path.as_str(), contents.as_str()))
        .collect::<Vec<_>>();
    let (_root, scan) = scan(Some(&raw), &file_refs);
    let result = evaluate_versions(&scan, &pages, false)
        .result(VERSION_ALIGNMENT_ID, 0)
        .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Warning);
    assert_eq!(result.coverage().denominator(), 101);
    assert_eq!(result.coverage().evaluated(), 101);
    assert_eq!(result.evidence().len(), 1);
    assert!(!result.coverage().truncated());
    assert!(serde_json::to_string(&result)
        .unwrap()
        .contains("\"opaque_id\":101"));
}

#[test]
fn swapped_state_paths_are_error_with_exact_population() {
    let state = "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"b.md\",\"version\":1},\"page_b\":{\"file\":\"a.md\",\"version\":1}}}";
    let (_root, scan) = scan(
        Some(state),
        &[
            ("a.md", "---\norigin_id: page_a\norigin_version: 1\n---\n"),
            ("b.md", "---\norigin_id: page_b\norigin_version: 1\n---\n"),
        ],
    );
    let result = evaluate_identity(
        &scan,
        &[page("page_a", "active", 1), page("page_b", "active", 1)],
        false,
    )
    .result(IDENTITY_ID, 0)
    .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(result.severity(), LintSeverity::Error);
    assert_eq!(result.applicability(), LintApplicability::Applicable);
    assert_eq!(result.coverage().denominator(), 6);
    let report = crate::lint::runner::synthetic_report(vec![result]);
    assert!(report.complete());
    assert_eq!(report.totals().findings(), 1);
}

#[test]
fn reserved_and_non_markdown_targets_are_isolated_errors() {
    for (target, file) in [
        (
            ".wenlan/private.md",
            Some((".wenlan/private.md", "control")),
        ),
        ("page.txt", Some(("page.txt", "not markdown"))),
    ] {
        let raw = serde_json::json!({
            "schema_version": 2,
            "pages": {"page_a": {"file": target, "version": 1}}
        })
        .to_string();
        let files = file.into_iter().collect::<Vec<_>>();
        let (_root, scan) = scan(Some(&raw), &files);
        let result = evaluate_identity(&scan, &[page("page_a", "active", 1)], false)
            .result(IDENTITY_ID, 0)
            .unwrap();
        assert_eq!(result.outcome(), LintOutcome::Finding, "{target}");
        assert_eq!(result.severity(), LintSeverity::Error, "{target}");
        assert_eq!(result.coverage().denominator(), 2, "{target}");
    }
}

#[test]
fn source_stub_is_excluded_from_manual_markdown_population_and_output() {
    let state = "{\"schema_version\":2,\"pages\":{\"page_a\":{\"file\":\"a.md\",\"version\":1}}}";
    let (_root, scan) = scan(
        Some(state),
        &[
            ("a.md", "---\norigin_id: page_a\norigin_version: 1\n---\n"),
            (
                "_sources/CANARY_PRIVATE_STUB.md",
                "---\norigin_id: page_other\n---\n",
            ),
        ],
    );
    let result = evaluate_identity(&scan, &[page("page_a", "active", 1)], false)
        .result(IDENTITY_ID, 0)
        .unwrap();
    assert_eq!(result.outcome(), LintOutcome::Pass);
    assert_eq!(result.coverage().denominator(), 3);
    assert_eq!(scan.page_markdown().len(), 1);
    assert!(!serde_json::to_string(&result)
        .unwrap()
        .contains("CANARY_PRIVATE_STUB"));
}

#[tokio::test]
async fn runner_uses_pages_workspace_and_keeps_other_workspace_out_of_output() {
    let (db, _db_dir) = test_db().await;
    let root = root();
    write(
        root.path(),
        ".wenlan/state.json",
        "{\"schema_version\":2,\"pages\":{\"page_selected\":{\"file\":\"selected.md\",\"version\":1},\"page_CANARY_OTHER\":{\"file\":\"private.md\",\"version\":9}}}",
    );
    write(
        root.path(),
        "selected.md",
        "---\norigin_id: page_selected\norigin_version: 1\n---\nbody\n",
    );
    write(
        root.path(),
        "private.md",
        "---\norigin_id: page_CANARY_OTHER\norigin_version: 8\n---\nprivate\n",
    );
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO spaces (id, name, created_at, updated_at) VALUES ('space-alpha', 'alpha', 1, 1), ('space-beta', 'beta', 1, 1)",
        (),
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO pages (id, title, content, source_memory_ids, version, status, created_at, last_compiled, last_modified, workspace) VALUES ('page_selected', 'selected', 'body', '[]', 1, 'active', '1', '1', '1', 'alpha'), ('page_CANARY_OTHER', 'private', 'private', '[]', 9, 'active', '1', '1', '1', 'beta')",
        (),
    )
    .await
    .unwrap();
    drop(conn);

    let report = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &wenlan_types::lint::LintQuery {
                profile: None,
                space: Some("alpha".to_string()),
            },
            Some(root.path()),
            true,
        )
        .await
        .unwrap();
    for check_id in [STATE_CONTRACT_ID, IDENTITY_ID, VERSION_ALIGNMENT_ID] {
        let check = report
            .checks()
            .iter()
            .find(|check| check.check_id() == check_id)
            .unwrap();
        assert_eq!(check.outcome(), LintOutcome::Pass, "{check_id}");
    }
    assert!(
        report.complete(),
        "all catalogued Page checks are conclusive"
    );
    assert!(!serde_json::to_string(&report)
        .unwrap()
        .contains("CANARY_OTHER"));
}

#[tokio::test]
async fn uncategorized_scope_load_pages_finds_unfiled_workspace_and_excludes_registered() {
    let (db, _db_dir) = test_db().await;
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO pages (id, title, content, source_memory_ids, version, status, created_at, last_compiled, last_modified, workspace) VALUES          ('page-unfiled', 'unfiled', 'body', '[]', 1, 'active', '1', '1', '1', '00000000-0000-4000-8000-000000000001'),          ('page-CANARY-alpha', 'alpha', 'body', '[]', 1, 'active', '1', '1', '1', 'alpha')",
        (),
    )
    .await
    .unwrap();
    drop(conn);

    // Shape-C regression guard: `workspace IS NULL` was permanently dead
    // after migration 80's NOT NULL stamp, so a revert here would make
    // `load_pages` vacuously return zero rows under this scope instead of
    // actually excluding page-CANARY-alpha.
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
    let pages = load_pages(&context).await.unwrap();
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].id, "page-unfiled");
}
