// SPDX-License-Identifier: Apache-2.0

use super::brief_files::{
    import_legacy_status_files, parse_legacy_status, project_brief_receipt, render_brief_receipt,
};
use std::sync::Arc;
use wenlan_core::db::MemoryDB;
use wenlan_core::events::NoopEmitter;
use wenlan_types::{BriefItemState, BriefMutation, BriefUpdateRequest};

const LEGACY_STATUS: &str = r#"# Wenlan — Current Status

## Last session (2026-07-27)
- Added daemon-owned Brief storage
- Reviewed the concurrency contract

## Active
<!-- recent work -->
- Ship the route (added 2026-07-26)
- Wait for CI (added 2026-07-25) (gated: checks pass)
- Preserve malformed metadata (added someday)

## Backlog
- Document migration (added 2026-07-20)
"#;

#[test]
fn legacy_parser_preserves_sections_dates_gates_and_malformed_bullets() {
    let parsed = parse_legacy_status(LEGACY_STATUS, 1_900_000_000).unwrap();

    assert_eq!(
        parsed.last_session_summary,
        "Added daemon-owned Brief storage\nReviewed the concurrency contract"
    );
    assert_eq!(parsed.last_handoff_at, Some(1_785_110_400));
    assert_eq!(parsed.items.len(), 4);

    assert_eq!(parsed.items[0].text, "Ship the route");
    assert_eq!(parsed.items[0].state, BriefItemState::Active);
    assert_eq!(parsed.items[0].added_at, 1_785_024_000);
    assert_eq!(parsed.items[0].gate, None);

    assert_eq!(parsed.items[1].text, "Wait for CI");
    assert_eq!(parsed.items[1].gate.as_deref(), Some("checks pass"));

    assert_eq!(
        parsed.items[2].text,
        "Preserve malformed metadata (added someday)"
    );
    assert_eq!(parsed.items[2].added_at, 1_900_000_000);
    assert_eq!(parsed.items[3].state, BriefItemState::Backlog);
}

#[tokio::test]
async fn legacy_import_is_once_only_and_keeps_the_source_file() {
    let root = tempfile::tempdir().unwrap();
    let db_root = tempfile::tempdir().unwrap();
    let status_file = root.path().join("wenlan.md");
    std::fs::write(&status_file, LEGACY_STATUS).unwrap();
    std::fs::write(
        root.path().join("handoff-wenlan.json"),
        r#"{"lastHandoff":"2026-07-27T00:00:00Z"}"#,
    )
    .unwrap();

    let db = MemoryDB::new(db_root.path(), Arc::new(NoopEmitter))
        .await
        .unwrap();
    let first = import_legacy_status_files(&db, root.path()).await.unwrap();
    assert_eq!(first.imported, 1);
    assert_eq!(first.skipped_existing, 0);
    assert_eq!(
        std::fs::read_to_string(&status_file).unwrap(),
        LEGACY_STATUS
    );

    let brief = db.get_brief_by_space_name("wenlan").await.unwrap().unwrap();
    assert_eq!(brief.active.len(), 3);
    assert_eq!(brief.backlog.len(), 1);
    assert_eq!(
        brief.last_session_summary,
        "Added daemon-owned Brief storage\nReviewed the concurrency contract"
    );

    std::fs::write(
        &status_file,
        LEGACY_STATUS.replace("Ship the route", "must not overwrite"),
    )
    .unwrap();
    let second = import_legacy_status_files(&db, root.path()).await.unwrap();
    assert_eq!(second.imported, 0);
    assert_eq!(second.skipped_existing, 1);
    let unchanged = db.get_brief_by_space_name("wenlan").await.unwrap().unwrap();
    assert!(unchanged
        .active
        .iter()
        .any(|item| item.text == "Ship the route"));
    assert!(unchanged
        .active
        .iter()
        .all(|item| item.text != "must not overwrite"));
}

#[tokio::test]
async fn legacy_import_marker_prevents_space_resurrection_after_rename_or_delete() {
    let root = tempfile::tempdir().unwrap();
    let db_root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("alpha.md"), LEGACY_STATUS).unwrap();
    let db = MemoryDB::new(db_root.path(), Arc::new(NoopEmitter))
        .await
        .unwrap();

    assert_eq!(
        import_legacy_status_files(&db, root.path())
            .await
            .unwrap()
            .imported,
        1
    );
    db.update_space("alpha", "beta", None).await.unwrap();

    let after_rename = import_legacy_status_files(&db, root.path()).await.unwrap();
    assert_eq!(after_rename.imported, 0);
    assert!(db.get_space("alpha").await.unwrap().is_none());
    assert!(db.get_brief_by_space_name("beta").await.unwrap().is_some());

    db.delete_space("beta", "keep").await.unwrap();
    let after_delete = import_legacy_status_files(&db, root.path()).await.unwrap();
    assert_eq!(after_delete.imported, 0);
    assert!(db.get_space("alpha").await.unwrap().is_none());
}

#[tokio::test]
async fn receipt_projection_is_generated_human_only_and_replaces_legacy_after_update() {
    let root = tempfile::tempdir().unwrap();
    let db_root = tempfile::tempdir().unwrap();
    let status_file = root.path().join("wenlan.md");
    std::fs::write(&status_file, LEGACY_STATUS).unwrap();
    let db = MemoryDB::new(db_root.path(), Arc::new(NoopEmitter))
        .await
        .unwrap();
    import_legacy_status_files(&db, root.path()).await.unwrap();

    let brief = db.get_brief_by_space_name("wenlan").await.unwrap().unwrap();
    let rendered = render_brief_receipt(&brief, 1_785_196_800);
    assert!(rendered.contains("Generated by Wenlan"));
    assert!(rendered.contains("## Last session"));
    assert!(rendered.contains("## Active"));
    assert!(rendered.contains("## Backlog"));
    assert!(rendered.contains("(gated: checks pass)"));
    assert!(!rendered.contains(&brief.active[0].id));

    let path = project_brief_receipt(root.path(), &brief, 1_785_196_800).unwrap();
    assert_eq!(path, status_file);
    let projected = std::fs::read_to_string(path).unwrap();
    assert!(projected.starts_with("<!-- Generated by Wenlan"));
    assert_ne!(projected, LEGACY_STATUS);
}

#[tokio::test]
async fn generated_receipt_cannot_recreate_a_renamed_space() {
    let root = tempfile::tempdir().unwrap();
    let db_root = tempfile::tempdir().unwrap();
    let db = MemoryDB::new(db_root.path(), Arc::new(NoopEmitter))
        .await
        .unwrap();
    let request = BriefUpdateRequest {
        space: "alpha".into(),
        caller_id: "projection-test".into(),
        operation_id: "create-generated".into(),
        summary: None,
        mutations: vec![BriefMutation::Add {
            text: "current item".into(),
            state: BriefItemState::Active,
            added_at: None,
            gate: None,
        }],
    };
    db.apply_brief_update(&request).await.unwrap();
    let brief = db.get_brief_by_space_name("alpha").await.unwrap().unwrap();
    project_brief_receipt(root.path(), &brief, 1_785_196_800).unwrap();
    db.update_space("alpha", "beta", None).await.unwrap();

    let report = import_legacy_status_files(&db, root.path()).await.unwrap();
    assert_eq!(report.imported, 0);
    assert!(db.get_space("alpha").await.unwrap().is_none());
    assert!(db.get_brief_by_space_name("beta").await.unwrap().is_some());
}

#[tokio::test]
async fn projection_failure_cannot_rollback_committed_brief_state() {
    let db_root = tempfile::tempdir().unwrap();
    let db = MemoryDB::new(db_root.path(), Arc::new(NoopEmitter))
        .await
        .unwrap();
    let request = BriefUpdateRequest {
        space: "wenlan".into(),
        caller_id: "projection-test".into(),
        operation_id: "committed-first".into(),
        summary: None,
        mutations: vec![BriefMutation::Add {
            text: "committed item".into(),
            state: BriefItemState::Active,
            added_at: None,
            gate: None,
        }],
    };
    db.apply_brief_update(&request).await.unwrap();
    let brief = db.get_brief_by_space_name("wenlan").await.unwrap().unwrap();

    let not_a_directory = db_root.path().join("status-root-is-a-file");
    std::fs::write(&not_a_directory, "occupied").unwrap();
    assert!(project_brief_receipt(&not_a_directory, &brief, 1_785_196_800).is_err());

    let stored = db.get_brief_by_space_name("wenlan").await.unwrap().unwrap();
    assert_eq!(stored.active[0].text, "committed item");
}

#[tokio::test]
async fn older_projection_cannot_replace_a_newer_receipt_for_the_same_space() {
    let root = tempfile::tempdir().unwrap();
    let mut newer = wenlan_types::Brief {
        space_id: "space-1".into(),
        space: "wenlan".into(),
        last_session_summary: "newer summary".into(),
        last_handoff_at: Some(20),
        version: 3,
        active: Vec::new(),
        backlog: Vec::new(),
    };
    let mut older = newer.clone();
    older.last_session_summary = "stale summary".into();
    older.version = 2;

    project_brief_receipt(root.path(), &newer, 30).unwrap();
    project_brief_receipt(root.path(), &older, 31).unwrap();

    let projected = std::fs::read_to_string(root.path().join("wenlan.md")).unwrap();
    assert!(projected.contains("newer summary"));
    assert!(!projected.contains("stale summary"));

    newer.space_id = "space-2".into();
    newer.last_session_summary = "replacement Space".into();
    newer.version = 1;
    project_brief_receipt(root.path(), &newer, 32).unwrap();
    let replaced = std::fs::read_to_string(root.path().join("wenlan.md")).unwrap();
    assert!(replaced.contains("replacement Space"));
}
