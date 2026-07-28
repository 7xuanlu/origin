// SPDX-License-Identifier: Apache-2.0

use super::tests::test_db;
use super::{MemoryDB, SCHEMA_VERSION};
use wenlan_types::{
    BriefConflictReason, BriefItemState, BriefMutation, BriefSummaryUpdate, BriefUpdateRequest,
};

#[tokio::test]
async fn brief_migration_creates_fk_backed_tables() {
    let (db, _temp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "DROP TABLE brief_items;
             DROP TABLE briefs;
             PRAGMA user_version=99;",
        )
        .await
        .unwrap();
    }
    db.migrate_100_brief(99).await.unwrap();
    let conn = db.conn.lock().await;

    for table in ["briefs", "brief_items"] {
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
                libsql::params![table],
            )
            .await
            .unwrap();
        assert!(rows.next().await.unwrap().is_some(), "missing {table}");
    }

    let mut version_rows = conn.query("PRAGMA user_version", ()).await.unwrap();
    assert_eq!(
        version_rows
            .next()
            .await
            .unwrap()
            .unwrap()
            .get::<i64>(0)
            .unwrap(),
        i64::from(SCHEMA_VERSION)
    );
}

#[tokio::test]
async fn brief_migration_follows_space_identity_across_rename_and_delete() {
    let (db, _temp) = test_db().await;
    let space = db.create_space("before", None, false).await.unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO briefs
                (space_id, last_session_summary, last_handoff_at, version, updated_at)
             VALUES (?1, 'summary', 10, 1, 10)",
            libsql::params![space.id.as_str()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO brief_items
                (id, space_id, text, state, added_at, gate, version, updated_at)
             VALUES ('item-1', ?1, 'next step', 'active', 10, NULL, 1, 10)",
            libsql::params![space.id.as_str()],
        )
        .await
        .unwrap();
    }

    db.update_space("before", "after", None).await.unwrap();
    assert_eq!(
        brief_space_id(&db).await.as_deref(),
        Some(space.id.as_str())
    );

    db.delete_space("after", "keep").await.unwrap();
    assert!(brief_space_id(&db).await.is_none());
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query("SELECT COUNT(*) FROM brief_items", ())
        .await
        .unwrap();
    assert_eq!(
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        0
    );
}

async fn brief_space_id(db: &MemoryDB) -> Option<String> {
    let conn = db.conn.lock().await;
    let mut rows = conn.query("SELECT space_id FROM briefs", ()).await.unwrap();
    rows.next()
        .await
        .unwrap()
        .map(|row| row.get::<String>(0).unwrap())
}

fn update_request(
    space: &str,
    operation_id: &str,
    mutations: Vec<BriefMutation>,
) -> BriefUpdateRequest {
    BriefUpdateRequest {
        space: space.into(),
        caller_id: "brief-test".into(),
        operation_id: operation_id.into(),
        summary: None,
        mutations,
    }
}

fn add(text: &str, state: BriefItemState) -> BriefMutation {
    BriefMutation::Add {
        text: text.into(),
        state,
        added_at: None,
        gate: None,
    }
}

#[tokio::test]
async fn brief_read_without_stored_brief_is_write_free() {
    let (db, _temp) = test_db().await;
    db.create_space("empty", None, false).await.unwrap();

    assert!(db.get_brief_by_space_name("empty").await.unwrap().is_none());

    let conn = db.conn.lock().await;
    let mut rows = conn.query("SELECT COUNT(*) FROM briefs", ()).await.unwrap();
    assert_eq!(
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        0
    );
}

#[tokio::test]
async fn brief_first_update_creates_brief_and_stable_item_id() {
    let (db, _temp) = test_db().await;
    let receipt = db
        .apply_brief_update(&update_request(
            "new-space",
            "create",
            vec![add("ship the release", BriefItemState::Active)],
        ))
        .await
        .unwrap();

    assert!(db.get_space("new-space").await.unwrap().is_some());
    assert_eq!(receipt.applied.len(), 1);
    let item_id = receipt.applied[0].item_id.as_deref().unwrap();
    assert!(!item_id.is_empty());

    let brief = db
        .get_brief_by_space_name("new-space")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(brief.active.len(), 1);
    assert_eq!(brief.active[0].id, item_id);
    assert_eq!(brief.active[0].version, 1);
}

#[tokio::test]
async fn brief_mutations_require_current_item_version_and_completion_removes_item() {
    let (db, _temp) = test_db().await;
    let created = db
        .apply_brief_update(&update_request(
            "mutations",
            "add",
            vec![add("draft", BriefItemState::Active)],
        ))
        .await
        .unwrap();
    let item_id = created.applied[0].item_id.clone().unwrap();

    let edited = db
        .apply_brief_update(&update_request(
            "mutations",
            "edit",
            vec![BriefMutation::Edit {
                item_id: item_id.clone(),
                expected_version: 1,
                text: "edited".into(),
            }],
        ))
        .await
        .unwrap();
    assert_eq!(edited.applied[0].version, Some(2));

    let moved = db
        .apply_brief_update(&update_request(
            "mutations",
            "move",
            vec![BriefMutation::Move {
                item_id: item_id.clone(),
                expected_version: 2,
                state: BriefItemState::Backlog,
            }],
        ))
        .await
        .unwrap();
    assert_eq!(moved.applied[0].version, Some(3));

    let gated = db
        .apply_brief_update(&update_request(
            "mutations",
            "gate",
            vec![BriefMutation::SetGate {
                item_id: item_id.clone(),
                expected_version: 3,
                gate: Some("CI green".into()),
            }],
        ))
        .await
        .unwrap();
    assert_eq!(gated.applied[0].version, Some(4));

    let completed = db
        .apply_brief_update(&update_request(
            "mutations",
            "complete",
            vec![BriefMutation::Complete {
                item_id,
                expected_version: 4,
            }],
        ))
        .await
        .unwrap();
    assert_eq!(completed.applied.len(), 1);

    let brief = db
        .get_brief_by_space_name("mutations")
        .await
        .unwrap()
        .unwrap();
    assert!(brief.active.is_empty());
    assert!(brief.backlog.is_empty());
}

#[tokio::test]
async fn brief_stale_item_conflict_does_not_block_unrelated_mutation() {
    let (db, _temp) = test_db().await;
    let created = db
        .apply_brief_update(&update_request(
            "concurrency",
            "add-two",
            vec![
                add("first", BriefItemState::Active),
                add("second", BriefItemState::Active),
            ],
        ))
        .await
        .unwrap();
    let first = created.applied[0].item_id.clone().unwrap();
    let second = created.applied[1].item_id.clone().unwrap();

    db.apply_brief_update(&update_request(
        "concurrency",
        "advance-first",
        vec![BriefMutation::Edit {
            item_id: first.clone(),
            expected_version: 1,
            text: "newer first".into(),
        }],
    ))
    .await
    .unwrap();

    let partial = db
        .apply_brief_update(&update_request(
            "concurrency",
            "partial",
            vec![
                BriefMutation::Edit {
                    item_id: first.clone(),
                    expected_version: 1,
                    text: "stale overwrite".into(),
                },
                BriefMutation::Move {
                    item_id: second.clone(),
                    expected_version: 1,
                    state: BriefItemState::Backlog,
                },
            ],
        ))
        .await
        .unwrap();

    assert_eq!(partial.applied.len(), 1);
    assert_eq!(partial.applied[0].item_id.as_deref(), Some(second.as_str()));
    assert_eq!(partial.conflicts.len(), 1);
    assert_eq!(
        partial.conflicts[0].item_id.as_deref(),
        Some(first.as_str())
    );
    assert_eq!(
        partial.conflicts[0].reason,
        BriefConflictReason::VersionMismatch
    );

    let brief = db
        .get_brief_by_space_name("concurrency")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(brief.active.len(), 1);
    assert_eq!(brief.active[0].text, "newer first");
    assert_eq!(brief.backlog.len(), 1);
    assert_eq!(brief.backlog[0].id, second);
}

#[tokio::test]
async fn brief_operation_replay_is_idempotent_and_digest_mismatch_conflicts() {
    let (db, _temp) = test_db().await;
    let request = update_request(
        "replay",
        "same-operation",
        vec![add("only once", BriefItemState::Active)],
    );
    let first = db.apply_brief_update(&request).await.unwrap();
    let replay = db.apply_brief_update(&request).await.unwrap();
    assert_eq!(first, replay);

    let brief = db.get_brief_by_space_name("replay").await.unwrap().unwrap();
    assert_eq!(brief.active.len(), 1);

    let reused = update_request(
        "replay",
        "same-operation",
        vec![add("different request", BriefItemState::Active)],
    );
    let error = db.apply_brief_update(&reused).await.unwrap_err();
    assert!(matches!(error, crate::error::WenlanError::Conflict(_)));
}

#[tokio::test]
async fn brief_summary_uses_brief_version_without_blocking_item_changes() {
    let (db, _temp) = test_db().await;
    db.apply_brief_update(&update_request(
        "summary",
        "create",
        vec![add("keep moving", BriefItemState::Active)],
    ))
    .await
    .unwrap();

    let first_summary = BriefUpdateRequest {
        space: "summary".into(),
        caller_id: "brief-test".into(),
        operation_id: "summary-one".into(),
        summary: Some(BriefSummaryUpdate {
            text: "first summary".into(),
            expected_version: 1,
        }),
        mutations: Vec::new(),
    };
    db.apply_brief_update(&first_summary).await.unwrap();

    let partial = BriefUpdateRequest {
        space: "summary".into(),
        caller_id: "brief-test".into(),
        operation_id: "summary-stale".into(),
        summary: Some(BriefSummaryUpdate {
            text: "stale summary".into(),
            expected_version: 1,
        }),
        mutations: vec![add("safe addition", BriefItemState::Backlog)],
    };
    let receipt = db.apply_brief_update(&partial).await.unwrap();
    assert_eq!(receipt.applied.len(), 1);
    assert_eq!(receipt.conflicts.len(), 1);
    assert_eq!(
        receipt.conflicts[0].reason,
        BriefConflictReason::BriefVersionMismatch
    );

    let brief = db
        .get_brief_by_space_name("summary")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(brief.last_session_summary, "first summary");
    assert_eq!(brief.backlog.len(), 1);
}
