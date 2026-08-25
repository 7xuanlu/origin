// SPDX-License-Identifier: Apache-2.0

use super::tests::test_db;
use super::MemoryDB;
use crate::error::WenlanError;
use crate::pages::{PageDraftDeleteOutcome, PageDraftPublishOutcome, PageDraftUpdateOutcome};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::{Barrier, Notify};

pub(super) mod transaction_test_hooks {
    use super::*;

    struct Pause {
        reached: Arc<Notify>,
        resume: Arc<Notify>,
    }

    static CREATE_AFTER_INSERT: OnceLock<Mutex<HashMap<String, Pause>>> = OnceLock::new();
    static AFTER_SPACE_VALIDATION: OnceLock<Mutex<HashMap<String, Pause>>> = OnceLock::new();
    static AFTER_SPACE_CASCADE: OnceLock<Mutex<HashMap<String, Pause>>> = OnceLock::new();

    fn install_pause(
        pauses: &OnceLock<Mutex<HashMap<String, Pause>>>,
        id: &str,
    ) -> (Arc<Notify>, Arc<Notify>) {
        let reached = Arc::new(Notify::new());
        let resume = Arc::new(Notify::new());
        pauses.get_or_init(Default::default).lock().unwrap().insert(
            id.to_string(),
            Pause {
                reached: Arc::clone(&reached),
                resume: Arc::clone(&resume),
            },
        );
        (reached, resume)
    }

    async fn reach_pause(pauses: &OnceLock<Mutex<HashMap<String, Pause>>>, id: &str) {
        let pause = pauses
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .remove(id);
        if let Some(pause) = pause {
            pause.reached.notify_one();
            pause.resume.notified().await;
        }
    }

    pub(super) fn pause_create_after_insert(id: &str) -> (Arc<Notify>, Arc<Notify>) {
        install_pause(&CREATE_AFTER_INSERT, id)
    }

    pub(crate) async fn after_create_insert(id: &str) {
        reach_pause(&CREATE_AFTER_INSERT, id).await;
    }

    pub(super) fn pause_after_space_validation(id: &str) -> (Arc<Notify>, Arc<Notify>) {
        install_pause(&AFTER_SPACE_VALIDATION, id)
    }

    pub(crate) async fn after_space_validation(id: &str) {
        reach_pause(&AFTER_SPACE_VALIDATION, id).await;
    }

    pub(super) fn pause_after_space_cascade(key: &str) -> (Arc<Notify>, Arc<Notify>) {
        install_pause(&AFTER_SPACE_CASCADE, key)
    }

    pub(crate) async fn after_space_cascade(key: &str) {
        reach_pause(&AFTER_SPACE_CASCADE, key).await;
    }
}

async fn scalar_i64(db: &MemoryDB, sql: &str, id: &str) -> i64 {
    let conn = db.conn.lock().await;
    let mut rows = conn.query(sql, libsql::params![id]).await.unwrap();
    rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap()
}

async fn seed_non_draft_page(
    db: &MemoryDB,
    id: &str,
    status: &str,
    space: &str,
    version: i64,
    last_modified: &str,
) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO pages (
            id, title, content, space, source_memory_ids, version, status,
            created_at, last_compiled, last_modified, creation_kind,
            review_status, workspace
         ) VALUES (?1, ?2, 'body', ?3, '[]', ?4, ?5,
            ?6, ?6, ?6, 'authored', 'unconfirmed', ?3)",
        libsql::params![
            id,
            format!("{status} page"),
            space,
            version,
            status,
            last_modified
        ],
    )
    .await
    .unwrap();
}

async fn page_version_and_modified(db: &MemoryDB, id: &str) -> (i64, String) {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT version, last_modified FROM pages WHERE id=?1",
            libsql::params![id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    (row.get(0).unwrap(), row.get(1).unwrap())
}

#[tokio::test]
async fn create_rejects_empty_but_persists_meaningful_partial_snapshots() {
    let (db, _tmp) = test_db().await;

    for (title, content, space) in [
        ("", "", None),
        (" \t", "\n", None),
        ("", "", Some("work")),
        (
            "",
            "<!-- origin:sources:start -->owned<!-- origin:sources:end -->",
            None,
        ),
    ] {
        assert!(matches!(
            db.create_page_draft(title, content, space, space).await,
            Err(WenlanError::Validation(_))
        ));
    }

    let title_only = db
        .create_page_draft("  Working title  ", "", None, None)
        .await
        .unwrap();
    let body_only = db
        .create_page_draft("", "  Opening paragraph  ", Some("work"), Some("work"))
        .await
        .unwrap();

    assert_eq!(title_only.title, "  Working title  ");
    assert_eq!(title_only.content, "");
    assert_eq!(body_only.title, "");
    assert_eq!(body_only.content, "  Opening paragraph  ");
    for page in [&title_only, &body_only] {
        assert_eq!(page.status, "draft");
        assert_eq!(page.creation_kind, "authored");
        assert_eq!(page.review_status, "unconfirmed");
        assert_eq!(page.source_memory_ids, Vec::<String>::new());
        assert!(page.citations.is_empty());
        assert!(page.entity_id.is_none());
        assert!(page.summary.is_none());
        assert_eq!(page.version, 1);
    }
}

#[tokio::test]
async fn client_uuid_is_validated_idempotent_and_collision_safe() {
    let (db, _tmp) = test_db().await;
    for invalid in [
        "",
        "not-a-page",
        "page_not-a-uuid",
        "page_00000000-0000-0000-0000-000000000000",
        "page_00000000-0000-4000-0000-000000000001",
        "page_00000000000040008000000000000001",
        "page_00000000-0000-4000-8000-000000000001-extra",
    ] {
        assert!(matches!(
            db.create_page_draft_with_id(invalid, "Draft", "Body", None, None)
                .await,
            Err(WenlanError::Validation(_))
        ));
    }

    let id = "page_00000000-0000-4000-8000-000000000001";
    let first = db
        .create_page_draft_with_id(id, "Draft", "Body  \n", Some("work"), Some("work"))
        .await
        .unwrap();
    let replay = db
        .create_page_draft_with_id(id, "Draft", "Body  \n", Some("work"), Some("work"))
        .await
        .unwrap();
    assert_eq!(first.id, id);
    assert_eq!(replay.version, first.version);
    assert_eq!(
        scalar_i64(&db, "SELECT COUNT(*) FROM pages WHERE id=?1", id).await,
        1
    );
    assert!(matches!(
        db.create_page_draft_with_id(id, "Draft", "Different", Some("work"), Some("work"))
            .await,
        Err(WenlanError::PageDraftIdConflict(conflict_id)) if conflict_id == id
    ));

    seed_non_draft_page(
        &db,
        "page_00000000-0000-4000-8000-000000000002",
        "active",
        "work",
        1,
        "2026-01-01T00:00:00Z",
    )
    .await;
    assert!(matches!(
        db.create_page_draft_with_id(
            "page_00000000-0000-4000-8000-000000000002",
            "Draft",
            "Body",
            None,
            None,
        )
        .await,
        Err(WenlanError::PageDraftIdConflict(_))
    ));
}

#[tokio::test]
async fn create_replay_returns_the_server_mutated_scope_after_space_rename() {
    let (db, _tmp) = test_db().await;
    db.create_space("work", None, false).await.unwrap();
    let id = "page_00000000-0000-4000-8000-000000000003";

    let created = db
        .create_page_draft_with_id_in_registered_space(id, "Draft", "Body", Some("work"))
        .await
        .unwrap();
    db.update_space("work", "work-renamed", None).await.unwrap();

    let replay = db
        .create_page_draft_with_id_in_registered_space(id, "Draft", "Body", Some("work"))
        .await
        .unwrap();

    assert_eq!(replay.id, created.id);
    assert_eq!(replay.version, created.version + 1);
    assert_eq!(replay.space.as_deref(), Some("work-renamed"));
    assert_eq!(replay.workspace.as_deref(), Some("work-renamed"));
    assert_eq!(
        scalar_i64(&db, "SELECT COUNT(*) FROM pages WHERE id=?1", id).await,
        1
    );
}

#[tokio::test]
async fn create_replay_rejects_a_different_scope_even_when_authored_content_matches() {
    let (db, _tmp) = test_db().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();
    let id = "page_00000000-0000-4000-8000-000000000004";

    db.create_page_draft_with_id_in_registered_space(id, "Draft", "Body", Some("work"))
        .await
        .unwrap();

    for conflicting_space in [Some("personal"), None, Some("missing")] {
        assert!(matches!(
            db.create_page_draft_with_id_in_registered_space(
                id,
                "Draft",
                "Body",
                conflicting_space,
            )
            .await,
            Err(WenlanError::PageDraftIdConflict(conflict_id)) if conflict_id == id
        ));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simultaneous_same_id_creates_are_idempotent_or_conflict_by_snapshot() {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);

    let identical_id = "page_00000000-0000-4000-8000-000000000003";
    let barrier = Arc::new(Barrier::new(3));
    let mut identical = Vec::new();
    for _ in 0..2 {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        identical.push(tokio::spawn(async move {
            barrier.wait().await;
            db.create_page_draft_with_id(identical_id, "Draft", "Body", None, None)
                .await
        }));
    }
    barrier.wait().await;
    let identical = [
        identical.remove(0).await.unwrap().unwrap(),
        identical.remove(0).await.unwrap().unwrap(),
    ];
    assert_eq!(identical[0].id, identical[1].id);
    assert_eq!(identical[0].version, identical[1].version);
    assert_eq!(
        scalar_i64(&db, "SELECT COUNT(*) FROM pages WHERE id=?1", identical_id,).await,
        1
    );

    let divergent_id = "page_00000000-0000-4000-8000-000000000004";
    let barrier = Arc::new(Barrier::new(3));
    let mut divergent = Vec::new();
    for (title, content) in [("First", "First body"), ("Second", "Second body")] {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        divergent.push(tokio::spawn(async move {
            barrier.wait().await;
            db.create_page_draft_with_id(divergent_id, title, content, None, None)
                .await
        }));
    }
    barrier.wait().await;
    let outcomes = [
        divergent.remove(0).await.unwrap(),
        divergent.remove(0).await.unwrap(),
    ];
    assert_eq!(outcomes.iter().filter(|outcome| outcome.is_ok()).count(), 1);
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(
                outcome,
                Err(WenlanError::PageDraftIdConflict(id)) if id == divergent_id
            ))
            .count(),
        1
    );
    assert_eq!(
        scalar_i64(&db, "SELECT COUNT(*) FROM pages WHERE id=?1", divergent_id,).await,
        1
    );
}

#[tokio::test]
async fn create_preserves_bytes_null_embedding_and_has_no_derived_rows() {
    let (db, _tmp) = test_db().await;
    let content = "\u{feff}\r\n  Before  \r\nAfter\t \r\n\r\n";
    assert_ne!(
        content.trim_end(),
        content,
        "positive control: trimming must change this fixture"
    );
    let page = db
        .create_page_draft("  Draft  ", content, None, None)
        .await
        .unwrap();

    assert_eq!(page.title, "  Draft  ");
    assert_eq!(page.content, content);
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM pages WHERE id=?1 AND embedding IS NULL",
            &page.id,
        )
        .await,
        1
    );
    for table in ["page_sources", "page_evidence", "page_links"] {
        let sql = format!(
            "SELECT COUNT(*) FROM {table} WHERE {}=?1",
            if table == "page_links" {
                "source_page_id"
            } else {
                "page_id"
            }
        );
        assert_eq!(scalar_i64(&db, &sql, &page.id).await, 0);
    }
}

#[tokio::test]
async fn create_and_update_reject_reserved_delimiters_without_mutation() {
    use crate::export::provenance::{SOURCES_BLOCK_END, SOURCES_BLOCK_START};

    let (db, _tmp) = test_db().await;
    let cases = [
        format!("before {SOURCES_BLOCK_START} after"),
        format!("before {SOURCES_BLOCK_END} after"),
        format!("{SOURCES_BLOCK_START}\nowned\n{SOURCES_BLOCK_END}"),
        format!(
            "{SOURCES_BLOCK_START}\none\n{SOURCES_BLOCK_END}\n\
             {SOURCES_BLOCK_START}\ntwo\n{SOURCES_BLOCK_END}"
        ),
        format!("```md\n{SOURCES_BLOCK_START}\n```\nkept prose"),
    ];

    let rejected_id = "page_00000000-0000-4000-8000-000000000099";
    for content in &cases {
        assert!(matches!(
            db.create_page_draft_with_id(rejected_id, "Draft", content, None, None)
                .await,
            Err(WenlanError::Validation(_))
        ));
        assert!(db.get_page(rejected_id).await.unwrap().is_none());
        assert_eq!(
            scalar_i64(
                &db,
                "SELECT COUNT(*) FROM page_draft_create_requests WHERE page_id=?1",
                rejected_id,
            )
            .await,
            0
        );
    }

    let draft = db
        .create_page_draft("Original title", "Original body  \n", None, None)
        .await
        .unwrap();
    for content in &cases {
        assert!(matches!(
            db.update_page_draft(
                &draft.id,
                draft.version,
                "Changed title",
                content,
                None,
                None,
            )
            .await,
            Err(WenlanError::Validation(_))
        ));
        let after = db.get_page(&draft.id).await.unwrap().unwrap();
        assert_eq!(after.title, draft.title);
        assert_eq!(after.content, draft.content);
        assert_eq!(after.version, draft.version);
    }
}

#[tokio::test]
async fn update_supports_exact_retry_and_rejects_stale_active_missing_and_empty() {
    let (db, _tmp) = test_db().await;
    db.create_space("retry-space", None, false).await.unwrap();
    let draft = db
        .create_page_draft("Draft", "Original body", None, None)
        .await
        .unwrap();

    let first = db
        .update_page_draft_in_registered_space(
            &draft.id,
            1,
            "  Revised title  ",
            "Revised body  \n",
            Some(" retry-space "),
        )
        .await
        .unwrap();
    let PageDraftUpdateOutcome::Updated(first) = first else {
        panic!("expected initial update");
    };
    assert_eq!(first.version, 2);
    db.delete_space("retry-space", "keep").await.unwrap();

    let retry = db
        .update_page_draft_in_registered_space(
            &draft.id,
            1,
            "  Revised title  ",
            "Revised body  \n",
            Some(" retry-space "),
        )
        .await
        .unwrap();
    let PageDraftUpdateOutcome::Updated(retry) = retry else {
        panic!("exact retry must return the committed snapshot");
    };
    assert_eq!(retry.version, 2);
    assert_eq!(retry.content, "Revised body  \n");

    assert!(matches!(
        db.update_page_draft(&draft.id, 1, "stale", "different", None, None)
            .await
            .unwrap(),
        PageDraftUpdateOutcome::VersionConflict { current_version: 2 }
    ));
    assert!(matches!(
        db.update_page_draft(
            &draft.id,
            2,
            "",
            "<!-- origin:sources:start -->owned<!-- origin:sources:end -->",
            None,
            None,
        )
        .await,
        Err(WenlanError::Validation(_))
    ));

    seed_non_draft_page(
        &db,
        "page_active_update_guard",
        "active",
        "work",
        1,
        "2026-01-01T00:00:00Z",
    )
    .await;
    // Wire contract: a non-draft row is "no such draft" (structured 404), not
    // a validation error — a queued editor update racing after publish relies
    // on it.
    assert!(matches!(
        db.update_page_draft(
            "page_active_update_guard",
            1,
            "Changed",
            "Changed",
            None,
            None,
        )
        .await,
        Err(WenlanError::NotFound(_))
    ));
    assert!(matches!(
        db.update_page_draft("page_missing", 1, "Title", "Body", None, None)
            .await,
        Err(WenlanError::NotFound(_))
    ));
}

#[tokio::test]
async fn update_replay_matches_after_space_only_divergent_scope() {
    // Regression: `update_page_draft` (validate_space=false) accepts divergent
    // space/workspace, but the write mirrors ONE resolved scope onto both
    // columns via the workspace-wins ladder. A space-only input (Some("work"),
    // None) stores space=workspace="work". The exact retry must replay as
    // Updated, not fall through to a spurious VersionConflict.
    let (db, _tmp) = test_db().await;
    let draft = db
        .create_page_draft("Draft", "Original body", None, None)
        .await
        .unwrap();

    let first = db
        .update_page_draft(&draft.id, 1, "Revised", "Body", Some("work"), None)
        .await
        .unwrap();
    let PageDraftUpdateOutcome::Updated(first) = first else {
        panic!("expected initial divergent update");
    };
    assert_eq!(first.version, 2);
    assert_eq!(first.space.as_deref(), Some("work"));
    assert_eq!(first.workspace.as_deref(), Some("work"));

    let retry = db
        .update_page_draft(&draft.id, 1, "Revised", "Body", Some("work"), None)
        .await
        .unwrap();
    let PageDraftUpdateOutcome::Updated(retry) = retry else {
        panic!("space-only divergent retry must replay as Updated, not VersionConflict");
    };
    assert_eq!(retry.version, 2);
}

#[tokio::test]
async fn update_replay_matches_after_workspace_only_divergent_scope() {
    // Mirror of the space-only case: workspace-only input (None, Some("work"))
    // also mirrors to space=workspace="work"; the exact retry must replay.
    let (db, _tmp) = test_db().await;
    let draft = db
        .create_page_draft("Draft", "Original body", None, None)
        .await
        .unwrap();

    let first = db
        .update_page_draft(&draft.id, 1, "Revised", "Body", None, Some("work"))
        .await
        .unwrap();
    let PageDraftUpdateOutcome::Updated(first) = first else {
        panic!("expected initial divergent update");
    };
    assert_eq!(first.version, 2);
    assert_eq!(first.space.as_deref(), Some("work"));
    assert_eq!(first.workspace.as_deref(), Some("work"));

    let retry = db
        .update_page_draft(&draft.id, 1, "Revised", "Body", None, Some("work"))
        .await
        .unwrap();
    let PageDraftUpdateOutcome::Updated(retry) = retry else {
        panic!("workspace-only divergent retry must replay as Updated, not VersionConflict");
    };
    assert_eq!(retry.version, 2);
}

#[tokio::test]
async fn update_replay_matches_for_unfiled_none_scope() {
    // Guard: the None,None -> 'unfiled' sentinel retry must still replay. The
    // stored 'unfiled' sentinel translates back to None on the wire, so both
    // the current and requested resolved wire scopes are None and must match.
    let (db, _tmp) = test_db().await;
    let draft = db
        .create_page_draft("Draft", "Original body", None, None)
        .await
        .unwrap();

    let first = db
        .update_page_draft(&draft.id, 1, "Revised", "Body", None, None)
        .await
        .unwrap();
    let PageDraftUpdateOutcome::Updated(first) = first else {
        panic!("expected initial unfiled update");
    };
    assert_eq!(first.version, 2);
    assert_eq!(first.space, None);
    assert_eq!(first.workspace, None);

    let retry = db
        .update_page_draft(&draft.id, 1, "Revised", "Body", None, None)
        .await
        .unwrap();
    let PageDraftUpdateOutcome::Updated(retry) = retry else {
        panic!("unfiled None,None retry must replay as Updated");
    };
    assert_eq!(retry.version, 2);
}

#[tokio::test]
async fn delete_is_version_safe_and_rejects_active_and_missing_pages() {
    let (db, _tmp) = test_db().await;
    let draft = db
        .create_page_draft("Draft", "Body", None, None)
        .await
        .unwrap();
    let updated = db
        .update_page_draft(&draft.id, 1, "Revised", "Updated", None, None)
        .await
        .unwrap();
    assert!(matches!(updated, PageDraftUpdateOutcome::Updated(_)));
    assert!(matches!(
        db.delete_page_draft(&draft.id, 1).await.unwrap(),
        PageDraftDeleteOutcome::VersionConflict { current_version: 2 }
    ));
    assert!(matches!(
        db.delete_page_draft(&draft.id, 2).await.unwrap(),
        PageDraftDeleteOutcome::Deleted
    ));
    assert!(db.get_page(&draft.id).await.unwrap().is_none());

    seed_non_draft_page(
        &db,
        "page_active_delete_guard",
        "active",
        "work",
        1,
        "2026-01-01T00:00:00Z",
    )
    .await;
    // Same wire contract as update: non-draft rows discard as "no such draft"
    // — the editor treats the structured 404 as completed cleanup.
    assert!(matches!(
        db.delete_page_draft("page_active_delete_guard", 1).await,
        Err(WenlanError::NotFound(_))
    ));
    assert!(matches!(
        db.delete_page_draft("page_missing", 1).await,
        Err(WenlanError::NotFound(_))
    ));
}

#[tokio::test]
async fn deleted_draft_id_cannot_be_replayed_or_reused() {
    let (db, _tmp) = test_db().await;
    let id = "page_00000000-0000-4000-8000-000000000005";
    let created = db
        .create_page_draft_with_id(id, "Draft", "Body", Some("work"), Some("work"))
        .await
        .unwrap();
    assert!(matches!(
        db.delete_page_draft(id, created.version).await.unwrap(),
        PageDraftDeleteOutcome::Deleted
    ));
    {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT title, content, space, workspace
                   FROM page_draft_create_requests
                  WHERE page_id=?1",
                libsql::params![id],
            )
            .await
            .unwrap();
        let row = rows
            .next()
            .await
            .unwrap()
            .expect("UUID tombstone must remain");
        for column in 0..4 {
            assert!(
                row.get::<Option<String>>(column).unwrap().is_none(),
                "discard must scrub the fingerprint payload and scope"
            );
        }
    }

    for (title, content, space, workspace) in [
        ("Draft", "Body", Some("work"), Some("work")),
        ("Different", "Request", None, None),
    ] {
        assert!(matches!(
            db.create_page_draft_with_id(id, title, content, space, workspace)
                .await,
            Err(WenlanError::PageDraftIdConflict(conflict_id)) if conflict_id == id
        ));
    }
    assert!(db.get_page(id).await.unwrap().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simultaneous_updates_allow_exactly_one_compare_and_swap_winner() {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);
    let draft = db
        .create_page_draft("Draft", "Body", None, None)
        .await
        .unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let mut tasks = Vec::new();
    for (title, body) in [("First", "First body"), ("Second", "Second body")] {
        let db = Arc::clone(&db);
        let barrier = Arc::clone(&barrier);
        let id = draft.id.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            db.update_page_draft(&id, 1, title, body, None, None)
                .await
                .unwrap()
        }));
    }
    barrier.wait().await;
    let outcomes = [
        tasks.remove(0).await.unwrap(),
        tasks.remove(0).await.unwrap(),
    ];
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, PageDraftUpdateOutcome::Updated(_)))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(
                outcome,
                PageDraftUpdateOutcome::VersionConflict { current_version: 2 }
            ))
            .count(),
        1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_create_after_insert_rolls_back_before_retry() {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);
    let id = "page_00000000-0000-4000-8000-000000000006";
    let (reached, _resume) = transaction_test_hooks::pause_create_after_insert(id);
    let task = {
        let db = Arc::clone(&db);
        tokio::spawn(async move {
            db.create_page_draft_with_id(id, "Draft", "Body", None, None)
                .await
        })
    };

    reached.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(
        scalar_i64(&db, "SELECT COUNT(*) FROM pages WHERE id=?1", id).await,
        0
    );
    db.create_page_draft_with_id(id, "Draft", "Body", None, None)
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn registered_space_validation_is_atomic_with_concurrent_rename() {
    let (db, _tmp) = test_db().await;
    assert!(matches!(
        db.create_page_draft_with_id_in_registered_space(
            "page_00000000-0000-4000-8000-000000000008",
            "Draft",
            "Body",
            Some("missing"),
        )
        .await,
        Err(WenlanError::Validation(_))
    ));

    db.create_space("old", None, false).await.unwrap();
    let db = Arc::new(db);
    let id = "page_00000000-0000-4000-8000-000000000007";
    let (reached, resume) = transaction_test_hooks::pause_after_space_validation(id);
    let create = {
        let db = Arc::clone(&db);
        tokio::spawn(async move {
            db.create_page_draft_with_id_in_registered_space(id, "Draft", "Body", Some("old"))
                .await
        })
    };

    reached.notified().await;
    let mut rename = {
        let db = Arc::clone(&db);
        tokio::spawn(async move { db.update_space("old", "renamed", None).await })
    };
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), &mut rename)
            .await
            .is_err()
    );
    resume.notify_one();
    create.await.unwrap().unwrap();
    rename.await.unwrap().unwrap();

    let persisted = db.get_page(id).await.unwrap().unwrap();
    assert_eq!(persisted.space.as_deref(), Some("renamed"));
    assert_eq!(persisted.workspace.as_deref(), Some("renamed"));
}

#[derive(Clone, Copy)]
enum SpacePath {
    Rename,
    DeleteMove,
    Reassign,
}

async fn assert_space_path_moves_all_statuses_but_only_bumps_draft(path: SpacePath) {
    let (db, _tmp) = test_db().await;
    db.create_space("src", None, false).await.unwrap();
    if !matches!(path, SpacePath::Rename) {
        db.create_space("dest", None, false).await.unwrap();
    }
    let draft = db
        .create_page_draft("Draft", "Body", Some("src"), Some("src"))
        .await
        .unwrap();
    seed_non_draft_page(
        &db,
        "page_active_space_control",
        "active",
        "src",
        7,
        "2026-01-01T00:00:00Z",
    )
    .await;
    seed_non_draft_page(
        &db,
        "page_archived_space_control",
        "archived",
        "src",
        9,
        "2026-01-02T00:00:00Z",
    )
    .await;
    let active_before = page_version_and_modified(&db, "page_active_space_control").await;
    let archived_before = page_version_and_modified(&db, "page_archived_space_control").await;

    match path {
        SpacePath::Rename => {
            db.update_space("src", "dest", None).await.unwrap();
        }
        SpacePath::DeleteMove => {
            db.delete_space("src", "move:dest").await.unwrap();
        }
        SpacePath::Reassign => {
            db.reassign_memories_space("src", "dest").await.unwrap();
        }
    }

    let moved_draft = db.get_page(&draft.id).await.unwrap().unwrap();
    assert_eq!(moved_draft.space.as_deref(), Some("dest"));
    assert_eq!(moved_draft.workspace.as_deref(), Some("dest"));
    assert_eq!(moved_draft.version, draft.version + 1);
    assert_ne!(moved_draft.last_modified, draft.last_modified);
    assert!(matches!(
        db.update_page_draft(&draft.id, draft.version, "Stale", "Stale", None, None)
            .await
            .unwrap(),
        PageDraftUpdateOutcome::VersionConflict { current_version }
            if current_version == draft.version + 1
    ));

    for (id, before) in [
        ("page_active_space_control", active_before),
        ("page_archived_space_control", archived_before),
    ] {
        let page = db.get_page(id).await.unwrap().unwrap();
        assert_eq!(page.space.as_deref(), Some("dest"));
        assert_eq!(page.workspace.as_deref(), Some("dest"));
        assert_eq!(page_version_and_modified(&db, id).await, before);
    }
}

async fn assert_cancelled_space_path_rolls_back_and_releases_connection(path: SpacePath) {
    let (source, destination, hook_key, reuse_name) = match path {
        SpacePath::Rename => (
            "abort-rename-src",
            "abort-rename-dest",
            "update_space:abort-rename-src",
            "after-abort-rename",
        ),
        SpacePath::DeleteMove => (
            "abort-delete-src",
            "abort-delete-dest",
            "delete_space:abort-delete-src",
            "after-abort-delete",
        ),
        SpacePath::Reassign => (
            "abort-reassign-src",
            "abort-reassign-dest",
            "reassign_memories_space:abort-reassign-src",
            "after-abort-reassign",
        ),
    };
    let (db, _tmp) = test_db().await;
    db.create_space(source, None, false).await.unwrap();
    if !matches!(path, SpacePath::Rename) {
        db.create_space(destination, None, false).await.unwrap();
    }
    let draft = db
        .create_page_draft("Draft", "Body", Some(source), Some(source))
        .await
        .unwrap();
    let before = page_version_and_modified(&db, &draft.id).await;
    let (reached, _resume) = transaction_test_hooks::pause_after_space_cascade(hook_key);
    let db = Arc::new(db);
    let operation = {
        let db = Arc::clone(&db);
        tokio::spawn(async move {
            match path {
                SpacePath::Rename => db.update_space(source, destination, None).await.map(|_| ()),
                SpacePath::DeleteMove => {
                    db.delete_space(source, &format!("move:{destination}"))
                        .await
                }
                SpacePath::Reassign => db
                    .reassign_memories_space(source, destination)
                    .await
                    .map(|_| ()),
            }
        })
    };

    reached.notified().await;
    operation.abort();
    assert!(operation.await.unwrap_err().is_cancelled());

    let persisted = db.get_page(&draft.id).await.unwrap().unwrap();
    assert_eq!(persisted.space.as_deref(), Some(source));
    assert_eq!(persisted.workspace.as_deref(), Some(source));
    assert_eq!(page_version_and_modified(&db, &draft.id).await, before);
    assert!(db.get_space(source).await.unwrap().is_some());
    if matches!(path, SpacePath::Rename) {
        assert!(db.get_space(destination).await.unwrap().is_none());
    } else {
        assert!(db.get_space(destination).await.unwrap().is_some());
    }
    db.create_space(reuse_name, None, false).await.unwrap();
}

#[tokio::test]
async fn rename_space_bumps_matching_draft_once_only() {
    assert_space_path_moves_all_statuses_but_only_bumps_draft(SpacePath::Rename).await;
}

#[tokio::test]
async fn delete_space_move_bumps_matching_draft_once_only() {
    assert_space_path_moves_all_statuses_but_only_bumps_draft(SpacePath::DeleteMove).await;
}

#[tokio::test]
async fn reassign_space_bumps_matching_draft_once_only() {
    assert_space_path_moves_all_statuses_but_only_bumps_draft(SpacePath::Reassign).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_space_cascades_rolls_back_and_releases_connection() {
    for path in [
        SpacePath::Rename,
        SpacePath::DeleteMove,
        SpacePath::Reassign,
    ] {
        assert_cancelled_space_path_rolls_back_and_releases_connection(path).await;
    }
}

#[tokio::test]
async fn description_delete_keep_and_failed_space_paths_do_not_bump_drafts() {
    let (db, _tmp) = test_db().await;
    db.create_space("src", None, false).await.unwrap();
    db.create_space("dest", None, false).await.unwrap();
    let draft = db
        .create_page_draft("Draft", "Body", Some("src"), Some("src"))
        .await
        .unwrap();
    let before = page_version_and_modified(&db, &draft.id).await;

    db.update_space("src", "src", Some("description"))
        .await
        .unwrap();
    assert_eq!(page_version_and_modified(&db, &draft.id).await, before);

    assert!(matches!(
        db.reassign_memories_space("src", "src").await,
        Err(WenlanError::Validation(_))
    ));
    assert_eq!(page_version_and_modified(&db, &draft.id).await, before);

    db.delete_space("src", "keep").await.unwrap();
    assert_eq!(page_version_and_modified(&db, &draft.id).await, before);

    assert!(db.update_space("missing", "dest", None).await.is_err());
    assert!(db.reassign_memories_space("missing", "dest").await.is_err());
    assert!(db.delete_space("missing", "move:dest").await.is_err());
    assert_eq!(page_version_and_modified(&db, &draft.id).await, before);
}

async fn page_status_kind_and_embedding(db: &MemoryDB, id: &str) -> (String, String, bool) {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT status, kind, embedding IS NOT NULL FROM pages WHERE id=?1",
            libsql::params![id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    (
        row.get(0).unwrap(),
        row.get(1).unwrap(),
        row.get::<i64>(2).unwrap() == 1,
    )
}

#[tokio::test]
async fn publish_flips_draft_to_active_and_replays_idempotently() {
    let (db, _tmp) = test_db().await;
    let draft = db
        .create_page_draft("  Sharding notes  ", "Body text", None, None)
        .await
        .unwrap();

    let published = match db.publish_page_draft(&draft.id, 1).await.unwrap() {
        PageDraftPublishOutcome::Published(page) => page,
        other => panic!("expected Published, got {other:?}"),
    };
    assert_eq!(published.status, "active");
    assert_eq!(published.title, "Sharding notes");
    assert_eq!(published.version, 2);
    assert_eq!(published.creation_kind, "authored");
    assert_eq!(published.review_status, "unconfirmed");
    assert_eq!(published.last_compiled, published.last_modified);
    let (status, kind, has_embedding) = page_status_kind_and_embedding(&db, &draft.id).await;
    assert_eq!(status, "active");
    assert_eq!(kind, "authored");
    assert!(has_embedding, "publish must write the page embedding");

    // Exact retry of the publish that landed replays the active page.
    let replayed = match db.publish_page_draft(&draft.id, 1).await.unwrap() {
        PageDraftPublishOutcome::Published(page) => page,
        other => panic!("expected replay, got {other:?}"),
    };
    assert_eq!(replayed.version, 2);
    assert_eq!(replayed.status, "active");

    // Any other version on the now-active page is a conflict, not "not a draft".
    assert!(matches!(
        db.publish_page_draft(&draft.id, 5).await.unwrap(),
        PageDraftPublishOutcome::VersionConflict { current_version: 2 }
    ));
    // The matching version on an active page is the not-a-draft validation.
    assert!(matches!(
        db.publish_page_draft(&draft.id, 2).await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn publish_rejects_stale_missing_and_incomplete_drafts() {
    let (db, _tmp) = test_db().await;

    assert!(matches!(
        db.publish_page_draft("page_00000000-0000-4000-8000-00000000f001", 1)
            .await,
        Err(WenlanError::NotFound(_))
    ));

    let draft = db
        .create_page_draft("Draft title", "Body", None, None)
        .await
        .unwrap();
    assert!(matches!(
        db.publish_page_draft(&draft.id, 7).await.unwrap(),
        PageDraftPublishOutcome::VersionConflict { current_version: 1 }
    ));

    // Publishing requires BOTH a trimmed title and non-empty content; a
    // title-only or body-only draft (legal to save) cannot publish, and the
    // failed attempt must not mutate the row.
    let title_only = db
        .create_page_draft("Title only", "", None, None)
        .await
        .unwrap();
    let body_only = db
        .create_page_draft("  \t", "Body only", None, None)
        .await
        .unwrap();
    for draft in [&title_only, &body_only] {
        assert!(matches!(
            db.publish_page_draft(&draft.id, 1).await,
            Err(WenlanError::Validation(_))
        ));
        let (status, _, _) = page_status_kind_and_embedding(&db, &draft.id).await;
        assert_eq!(status, "draft");
        assert_eq!(page_version_and_modified(&db, &draft.id).await.0, 1);
    }
}

#[tokio::test]
async fn publish_blocks_on_same_scope_case_insensitive_title_conflict() {
    let (db, _tmp) = test_db().await;
    db.create_space("work", None, false).await.unwrap();
    seed_non_draft_page(
        &db,
        "page_active",
        "active",
        "work",
        3,
        "2026-01-01T00:00:00Z",
    )
    .await;

    // Same scope + case-insensitively equal trimmed title -> conflict, and the
    // draft stays a draft.
    let clash = db
        .create_page_draft("  ACTIVE PAGE ", "Body", Some("work"), Some("work"))
        .await
        .unwrap();
    match db.publish_page_draft(&clash.id, 1).await.unwrap() {
        PageDraftPublishOutcome::TitleConflict {
            existing_page_id,
            existing_page_title,
            scope,
        } => {
            assert_eq!(existing_page_id, "page_active");
            assert_eq!(existing_page_title, "active page");
            assert_eq!(scope, "work");
        }
        other => panic!("expected TitleConflict, got {other:?}"),
    }
    let (status, _, _) = page_status_kind_and_embedding(&db, &clash.id).await;
    assert_eq!(status, "draft");

    // The same title in a different scope (unfiled here) publishes fine.
    let elsewhere = db
        .create_page_draft("Active page", "Body", None, None)
        .await
        .unwrap();
    assert!(matches!(
        db.publish_page_draft(&elsewhere.id, 1).await.unwrap(),
        PageDraftPublishOutcome::Published(_)
    ));

    // An archived page with the same title does not block a publish.
    seed_non_draft_page(
        &db,
        "page_archived",
        "archived",
        "work",
        1,
        "2026-01-01T00:00:00Z",
    )
    .await;
    let vs_archived = db
        .create_page_draft("Archived page", "Body", Some("work"), Some("work"))
        .await
        .unwrap();
    assert!(matches!(
        db.publish_page_draft(&vs_archived.id, 1).await.unwrap(),
        PageDraftPublishOutcome::Published(_)
    ));
}

#[tokio::test]
async fn publish_folds_unicode_titles_when_checking_conflicts() {
    let (db, _tmp) = test_db().await;
    // The two titles only collide under Unicode case folding; SQLite's
    // ASCII-only lower() would treat them as distinct.
    let existing = db
        .create_page_draft("i σχεδιο проект", "Body", None, None)
        .await
        .unwrap();
    match db.publish_page_draft(&existing.id, 1).await.unwrap() {
        PageDraftPublishOutcome::Published(_) => {}
        other => panic!("seed publish failed: {other:?}"),
    }

    let draft = db
        .create_page_draft("  I ΣΧΕΔΙΟ ПРОЕКТ  ", "Body", None, None)
        .await
        .unwrap();
    match db.publish_page_draft(&draft.id, 1).await.unwrap() {
        PageDraftPublishOutcome::TitleConflict {
            existing_page_id,
            existing_page_title,
            scope,
        } => {
            assert_eq!(existing_page_id, existing.id);
            assert_eq!(existing_page_title, "i σχεδιο проект");
            assert_eq!(scope, super::UNFILED_SPACE_ID);
        }
        other => panic!("expected TitleConflict, got {other:?}"),
    }
    let (status, _, _) = page_status_kind_and_embedding(&db, &draft.id).await;
    assert_eq!(
        status, "draft",
        "conflicted publish must not flip the draft"
    );
}

#[tokio::test]
async fn publish_appends_history_and_maintains_wikilinks() {
    let (db, _tmp) = test_db().await;
    // An active page that references [[Launch Checklist]] before any page
    // carries that title: publishing it leaves an orphan page_links row.
    let referrer = db
        .create_page_draft("Referrer", "See [[Launch Checklist]]", None, None)
        .await
        .unwrap();
    match db.publish_page_draft(&referrer.id, 1).await.unwrap() {
        PageDraftPublishOutcome::Published(_) => {}
        other => panic!("referrer publish failed: {other:?}"),
    }
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM page_links \
             WHERE source_page_id=?1 AND target_page_id IS NULL",
            &referrer.id,
        )
        .await,
        1,
        "publishing a body with an unresolved wikilink must record the orphan"
    );

    let draft = db
        .create_page_draft("Launch Checklist", "Links back to [[Referrer]]", None, None)
        .await
        .unwrap();
    let published = match db.publish_page_draft(&draft.id, 1).await.unwrap() {
        PageDraftPublishOutcome::Published(page) => page,
        other => panic!("publish failed: {other:?}"),
    };

    // The version bump left its immutable history row in the same tx.
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM page_history \
             WHERE page_id=?1 AND edited_by='publish' AND version=2",
            &draft.id,
        )
        .await,
        1
    );

    // The published body's resolved [[Referrer]] link minted an active edge
    // (resolved links live in edges, not page_links).
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM edges \
             WHERE edge_type='links' AND src_id=?1 AND valid_until IS NULL",
            &draft.id,
        )
        .await,
        1
    );

    // Publishing the page named by the pre-existing orphan resolved it: the
    // orphan row is gone and the referrer now carries an active links edge.
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM page_links \
             WHERE source_page_id=?1 AND target_page_id IS NULL",
            &referrer.id,
        )
        .await,
        0
    );
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM edges \
             WHERE edge_type='links' AND src_id=?1 AND valid_until IS NULL",
            &referrer.id,
        )
        .await,
        1
    );
    assert_eq!(published.version, 2);
}

/// Link identity and conflict identity share one fold. The bundled SQLite
/// `lower()` is ASCII-only, so a SQL-side fold resolves `[[launch]]` against
/// "Launch" but leaves `[[i σχεδιο проект]]` orphaned next to
/// "I ΣΧΕΔΙΟ ПРОЕКТ" — while the publish conflict check (Rust fold) calls the
/// same pair a duplicate title. Both now run through `page_title_key`.
#[tokio::test]
async fn wikilinks_fold_unicode_titles_like_the_conflict_check() {
    let (db, _tmp) = test_db().await;
    let referrer = db
        .create_page_draft("Referrer", "See [[i σχεδιο проект]]", None, None)
        .await
        .unwrap();
    match db.publish_page_draft(&referrer.id, 1).await.unwrap() {
        PageDraftPublishOutcome::Published(_) => {}
        other => panic!("referrer publish failed: {other:?}"),
    }
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM page_links \
             WHERE source_page_id=?1 AND target_page_id IS NULL",
            &referrer.id,
        )
        .await,
        1,
        "no page carries that title yet, so the link must record as an orphan"
    );

    // Publish the target under different Greek/Cyrillic casing, with a body
    // that links back to the referrer cross-case too.
    let target = db
        .create_page_draft("I ΣΧΕΔΙΟ ПРОЕКТ", "Back to [[rEfErReR]]", None, None)
        .await
        .unwrap();
    let published = match db.publish_page_draft(&target.id, 1).await.unwrap() {
        PageDraftPublishOutcome::Published(page) => page,
        other => panic!("target publish failed: {other:?}"),
    };
    assert_eq!(published.title, "I ΣΧΕΔΙΟ ПРОЕКТ");

    // The outgoing cross-case link resolved at publish time.
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM edges \
             WHERE edge_type='links' AND src_id=?1 AND valid_until IS NULL",
            &target.id,
        )
        .await,
        1,
        "the published body's cross-case [[rEfErReR]] link must mint an edge"
    );

    // The pre-existing Unicode orphan resolved through the same fold: the
    // orphan row is gone and the referrer carries an active links edge.
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM page_links \
             WHERE source_page_id=?1 AND target_page_id IS NULL",
            &referrer.id,
        )
        .await,
        0,
        "the Greek/Cyrillic orphan must resolve once its target publishes"
    );
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT COUNT(*) FROM edges \
             WHERE edge_type='links' AND src_id=?1 AND valid_until IS NULL",
            &referrer.id,
        )
        .await,
        1
    );
}

#[tokio::test]
async fn update_and_discard_after_publish_report_draft_not_found() {
    let (db, _tmp) = test_db().await;
    let draft = db
        .create_page_draft("Ship notes", "Body", None, None)
        .await
        .unwrap();
    match db.publish_page_draft(&draft.id, 1).await.unwrap() {
        PageDraftPublishOutcome::Published(_) => {}
        other => panic!("publish failed: {other:?}"),
    }

    // A queued editor update or discard racing after publish must get the
    // structured "no such draft" answer, never a validation error: discard
    // treats the 404 as completed cleanup.
    assert!(matches!(
        db.update_page_draft(&draft.id, 2, "Ship notes", "Body v2", None, None)
            .await,
        Err(WenlanError::NotFound(_))
    ));
    assert!(matches!(
        db.delete_page_draft(&draft.id, 2).await,
        Err(WenlanError::NotFound(_))
    ));
    let (status, _, _) = page_status_kind_and_embedding(&db, &draft.id).await;
    assert_eq!(status, "active", "the published page must survive both");
}
