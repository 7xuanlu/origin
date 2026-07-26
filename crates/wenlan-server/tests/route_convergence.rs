// SPDX-License-Identifier: Apache-2.0
//! Integration tests for the 4 converged write-path routes.
//!
//! Drives the axum Router via tower::ServiceExt::oneshot — no TCP socket.
//! Tests that the HTTP layer (route registration, JSON extraction, error
//! mapping) stays in sync with the origin-types wire shapes and
//! post_write capability signatures.
//!
//! Authored create-page coverage uses its supported zero-source path, while
//! distilled fixtures seed their source memories directly. This keeps route
//! validation and error mapping covered without a separate TCP server.

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use std::sync::{Arc, OnceLock};
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::db::MemoryDB;
use wenlan_core::events::NoopEmitter;
use wenlan_server::router::{build_router, AppRouter};
use wenlan_server::state::ServerState;
use wenlan_types::requests::CreateConceptRequest;
use wenlan_types::RawDocument;

fn data_dir_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

struct WritableKnowledgeConfig {
    previous: Option<std::ffi::OsString>,
    _tmp: tempfile::TempDir,
}

impl WritableKnowledgeConfig {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let pages = tmp.path().join("pages");
        std::fs::create_dir_all(&pages).unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::json!({ "knowledge_path": pages.to_string_lossy() }).to_string(),
        )
        .unwrap();
        let previous = std::env::var_os("WENLAN_DATA_DIR");
        std::env::set_var("WENLAN_DATA_DIR", tmp.path());
        Self {
            previous,
            _tmp: tmp,
        }
    }

    /// The vault the routes project into while this guard is alive.
    fn pages_dir(&self) -> std::path::PathBuf {
        self._tmp.path().join("pages")
    }
}

impl Drop for WritableKnowledgeConfig {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
            None => std::env::remove_var("WENLAN_DATA_DIR"),
        }
    }
}

async fn test_app() -> (AppRouter, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .unwrap();
    let state = ServerState {
        db: Some(Arc::new(db)),
        ..ServerState::default()
    };
    let router = build_router(Arc::new(RwLock::new(state)));
    (router, dir)
}

async fn json_post(
    app: &AppRouter,
    path: &str,
    agent: Option<&str>,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(a) = agent {
        builder = builder.header("x-agent-name", a);
    }
    let req = builder
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let val: serde_json::Value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, val)
}

// ── handle_create_entity ────────────────────────────────────────────────────

#[tokio::test]
async fn create_entity_happy_path() {
    let (app, _dir) = test_app().await;
    let (status, body) = json_post(
        &app,
        "/api/memory/entities",
        Some("test-agent"),
        serde_json::json!({"name": "TestEntity", "entity_type": "project"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        body["id"].as_str().is_some(),
        "expected id in response, got: {body}"
    );
}

#[tokio::test]
async fn create_entity_empty_name_returns_422() {
    let (app, _dir) = test_app().await;
    let (status, _body) = json_post(
        &app,
        "/api/memory/entities",
        None,
        serde_json::json!({"name": "", "entity_type": "person"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

// ── handle_create_relation ──────────────────────────────────────────────────

#[tokio::test]
async fn create_relation_happy_path() {
    let (app, _dir) = test_app().await;

    let (_, e1) = json_post(
        &app,
        "/api/memory/entities",
        Some("test"),
        serde_json::json!({"name": "Alice", "entity_type": "person"}),
    )
    .await;
    let (_, e2) = json_post(
        &app,
        "/api/memory/entities",
        Some("test"),
        serde_json::json!({"name": "Bob", "entity_type": "person"}),
    )
    .await;
    let from = e1["id"].as_str().expect("e1 missing id");
    let to = e2["id"].as_str().expect("e2 missing id");

    let (status, body) = json_post(
        &app,
        "/api/memory/relations",
        Some("test"),
        serde_json::json!({"from_entity": from, "to_entity": to, "relation_type": "knows"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        body["id"].as_str().is_some(),
        "expected id in response, got: {body}"
    );
}

#[tokio::test]
async fn create_relation_missing_entities_returns_422() {
    let (app, _dir) = test_app().await;
    let (status, _body) = json_post(
        &app,
        "/api/memory/relations",
        None,
        serde_json::json!({
            "from_entity": "no-such-id",
            "to_entity": "also-missing",
            "relation_type": "knows"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

/// M3g fence: `span`/`model_version`/`prompt_version` are daemon-internal (KG
/// extraction) only. An agent hitting `/api/memory/relations` directly must
/// never be able to set them -- the handler strips all three before the
/// core call, so a request that supplies them still writes `payload=NULL`.
#[tokio::test]
async fn create_relation_strips_agent_supplied_span_capture_fields() {
    let (app, dir) = test_app().await;

    let (_, e1) = json_post(
        &app,
        "/api/memory/entities",
        Some("test"),
        serde_json::json!({"name": "Carol", "entity_type": "person"}),
    )
    .await;
    let (_, e2) = json_post(
        &app,
        "/api/memory/entities",
        Some("test"),
        serde_json::json!({"name": "Dave", "entity_type": "person"}),
    )
    .await;
    let from = e1["id"].as_str().expect("e1 missing id");
    let to = e2["id"].as_str().expect("e2 missing id");

    let (status, body) = json_post(
        &app,
        "/api/memory/relations",
        Some("test"),
        serde_json::json!({
            "from_entity": from,
            "to_entity": to,
            "relation_type": "knows",
            "span": "Carol knows Dave",
            "model_version": "forged-model",
            "prompt_version": "forged-prompt",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let db_file = dir.path().join("origin_memory.db");
    let raw_db = libsql::Builder::new_local(db_file.to_str().unwrap())
        .build()
        .await
        .unwrap();
    let mut rows = raw_db
        .connect()
        .unwrap()
        .query(
            "SELECT payload FROM edges WHERE edge_type = 'relates' AND src_id = ?1 AND dst_id = ?2",
            libsql::params![from, to],
        )
        .await
        .unwrap();
    let payload: Option<String> = rows
        .next()
        .await
        .unwrap()
        .expect("relates edge row")
        .get(0)
        .unwrap();
    assert!(
        payload.is_none(),
        "agent-supplied span/model_version/prompt_version must be stripped, got payload: {payload:?}"
    );
}

// ── handle_add_observation ──────────────────────────────────────────────────

#[tokio::test]
async fn add_observation_happy_path() {
    let (app, _dir) = test_app().await;

    let (_, e) = json_post(
        &app,
        "/api/memory/entities",
        Some("test"),
        serde_json::json!({"name": "Subject", "entity_type": "person"}),
    )
    .await;
    let entity_id = e["id"].as_str().expect("entity missing id");

    let (status, body) = json_post(
        &app,
        "/api/memory/observations",
        Some("test"),
        serde_json::json!({"entity_id": entity_id, "content": "Subject likes Rust"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert!(
        body["id"].as_str().is_some(),
        "expected id in response, got: {body}"
    );
}

#[tokio::test]
async fn add_observation_short_content_returns_422() {
    let (app, _dir) = test_app().await;

    let (_, e) = json_post(
        &app,
        "/api/memory/entities",
        Some("test"),
        serde_json::json!({"name": "S", "entity_type": "person"}),
    )
    .await;
    let entity_id = e["id"].as_str().expect("entity missing id");

    let (status, _body) = json_post(
        &app,
        "/api/memory/observations",
        None,
        serde_json::json!({"entity_id": entity_id, "content": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

// ── Revision history endpoints ───────────────────────────────────────────────

async fn test_app_with_db() -> (AppRouter, Arc<MemoryDB>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(
        MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
            .await
            .unwrap(),
    );
    let state = ServerState {
        db: Some(Arc::clone(&db)),
        ..ServerState::default()
    };
    let router = build_router(Arc::new(RwLock::new(state)));
    (router, db, dir)
}

async fn create_distilled_page_fixture(
    db: &MemoryDB,
    title: &str,
    content: &str,
    source_ids: &[&str],
) -> String {
    let req = CreateConceptRequest {
        title: title.to_string(),
        content: content.to_string(),
        summary: None,
        entity_id: None,
        space: None,
        source_memory_ids: source_ids.iter().map(|id| (*id).to_string()).collect(),
        creation_kind: Some("distilled".to_string()),
        workspace: None,
    };
    let result =
        wenlan_core::post_write::create_page_with_floor(db, req, "test", None, source_ids.len())
            .await
            .expect("distilled page fixture must be created through PageWrite");
    db.set_page_review_status(&result.id, "confirmed")
        .await
        .expect("distilled page fixture review status must be confirmed");
    result.id
}

async fn create_authored_page_fixture(db: &MemoryDB, title: &str, content: &str) -> String {
    let req = CreateConceptRequest {
        title: title.to_string(),
        content: content.to_string(),
        summary: None,
        entity_id: None,
        space: None,
        source_memory_ids: Vec::new(),
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };
    let result = wenlan_core::post_write::create_page(db, req, "test", None)
        .await
        .expect("authored page fixture must be created through PageWrite");
    db.set_page_review_status(&result.id, "confirmed")
        .await
        .expect("authored page fixture review status must be confirmed");
    result.id
}

async fn json_get(app: &AppRouter, path: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method(Method::GET)
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let val: serde_json::Value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, val)
}

async fn json_put(
    app: &AppRouter,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method(Method::PUT)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let val: serde_json::Value = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, val)
}

fn reserved_marker_cases() -> Vec<(&'static str, String)> {
    use wenlan_core::export::provenance::{SOURCES_BLOCK_END, SOURCES_BLOCK_START};

    vec![
        (
            "start-only",
            format!("before\n{SOURCES_BLOCK_START}\nafter"),
        ),
        ("end-only", format!("before\n{SOURCES_BLOCK_END}\nafter")),
        (
            "paired",
            format!("before\n{SOURCES_BLOCK_START}\nowned\n{SOURCES_BLOCK_END}\nafter"),
        ),
        (
            "duplicate",
            format!("{SOURCES_BLOCK_START}\n{SOURCES_BLOCK_START}"),
        ),
        (
            "fenced",
            format!("```markdown\n{SOURCES_BLOCK_START}\n```\nordinary prose"),
        ),
    ]
}

fn knowledge_snapshot(root: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    fn visit(root: &std::path::Path, current: &std::path::Path, out: &mut Vec<(String, Vec<u8>)>) {
        let mut entries = std::fs::read_dir(current)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let path = entry.path();
            let relative = path.strip_prefix(root).unwrap().to_string_lossy();
            if path.is_dir() {
                out.push((format!("{relative}/"), Vec::new()));
                visit(root, &path, out);
            } else {
                out.push((relative.into_owned(), std::fs::read(path).unwrap()));
            }
        }
    }

    let mut snapshot = Vec::new();
    visit(root, root, &mut snapshot);
    snapshot
}

#[tokio::test]
async fn authored_create_rejects_reserved_markers_before_projection_or_db_mutation() {
    let _guard = data_dir_lock().lock().await;
    let config = WritableKnowledgeConfig::new();
    let (app, db, _dir) = test_app_with_db().await;
    let projection_before = knowledge_snapshot(&config.pages_dir());

    for (case, content) in reserved_marker_cases() {
        let (status, body) = json_post(
            &app,
            "/api/pages",
            Some("test-agent"),
            serde_json::json!({
                "title": format!("Rejected {case}"),
                "content": content,
                "source_memory_ids": [],
                "creation_kind": "authored"
            }),
        )
        .await;

        assert_eq!(
            status,
            StatusCode::UNPROCESSABLE_ENTITY,
            "{case} must map to HTTP 422; body: {body}"
        );
        assert!(
            db.list_pages("active", 100, 0).await.unwrap().is_empty(),
            "{case} must not create a DB page"
        );
        assert_eq!(
            knowledge_snapshot(&config.pages_dir()),
            projection_before,
            "{case} must be rejected before projection mutation"
        );
    }
}

/// A refresh that does not land must leave no projection claiming a version
/// ahead of the DB.
///
/// The route writes md *before* the gate runs and rolled back only on `Err`.
/// But a gated or no-op write returns `Ok`, so the agent's md — already stamped
/// `version + 1` — stayed in the vault. That is not a cosmetic leak: the page
/// watcher ingests any md whose `origin_version` is >= the DB version as an
/// `fs_edit` (`sources/page_watcher.rs:154`), so a stranded projection is
/// re-ingested AS A HUMAN WRITE on the next tick — laundering content the gate
/// just refused straight back past the gate and flipping `user_edited`.
///
/// Driven here through the deterministic no-op path (identical content and
/// sources => `WriteOutcome::Unchanged`, `post_write.rs:1602`). The gated race
/// that makes this dangerous needs a human edit interleaved between the route's
/// ownership read and the gate's CAS; that window is reachable only through
/// post_write's `#[cfg(test)]` `PRE_WRITE_GATE`, which this crate cannot see.
/// The route-local invariant is the same either way: no write, no projection.
#[tokio::test]
async fn refresh_that_does_not_write_leaves_no_projection_ahead_of_db() {
    let _guard = data_dir_lock().lock().await;
    let config = WritableKnowledgeConfig::new();
    let (app, db, _dir) = test_app_with_db().await;
    let mem_id = "mem_route_refresh_noop";
    let source_content = "Borrow checking rejects aliasing a value while it is mutably borrowed";
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_string(),
        source_id: mem_id.to_string(),
        title: "Borrow checking source".to_string(),
        content: source_content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("test".to_string()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .unwrap();

    let page_id =
        create_distilled_page_fixture(&db, "Borrow Checking", source_content, &[mem_id]).await;

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    assert!(
        !wenlan_core::post_write::page_is_human_owned(&before),
        "precondition: the page must be machine-owned, or the route returns at its \
         early ownership gate and never reaches the md write this test is about"
    );

    // Identical content AND identical source set — the gate reports Unchanged
    // and writes nothing, so nothing may be left behind in the vault either.
    let (status, body) = json_put(
        &app,
        &format!("/api/pages/{page_id}"),
        serde_json::json!({
            "content": source_content,
            "source_memory_ids": [mem_id]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.version, before.version,
        "precondition: a no-op refresh must not bump the DB version"
    );

    for entry in std::fs::read_dir(config.pages_dir()).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let stamp = text.lines().find_map(|line| {
            line.strip_prefix("origin_version:")
                .and_then(|v| v.trim().trim_matches('"').parse::<i64>().ok())
        });
        if let Some(stamp) = stamp {
            assert!(
                stamp <= after.version,
                "projection {:?} claims origin_version {} but the DB is at {} — the \
                 watcher would re-ingest this as a human edit",
                path.file_name(),
                stamp,
                after.version
            );
        }
    }
}

#[tokio::test]
async fn refresh_page_user_edited_page_stages_revision_card_without_overwrite() {
    let _guard = data_dir_lock().lock().await;
    let _config = WritableKnowledgeConfig::new();
    let (app, db, _dir) = test_app_with_db().await;
    let mem_id = "mem_route_pagewrite_owned";
    let source_content = "Rust ownership keeps memory safety rules explicit in systems code";
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_string(),
        source_id: mem_id.to_string(),
        title: "Rust ownership source".to_string(),
        content: source_content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("test".to_string()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .unwrap();

    let page_id =
        create_distilled_page_fixture(&db, "Rust Ownership", source_content, &[mem_id]).await;

    let human_content =
        "Rust ownership keeps memory safety rules explicit in systems code, with human notes";
    db.update_page_content(&page_id, human_content, &[mem_id], "fs_edit")
        .await
        .unwrap();
    let before = db.get_page(&page_id).await.unwrap().unwrap();
    assert!(before.user_edited, "precondition: page is human-owned");

    let proposed_content =
        "Rust ownership lets the compiler enforce memory safety during page refresh";
    let (status, body) = json_put(
        &app,
        &format!("/api/pages/{page_id}"),
        serde_json::json!({
            "content": proposed_content,
            "source_memory_ids": [mem_id]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["gated"], serde_json::json!(true), "body: {body}");
    let revision_card_id = body["revision_card_id"]
        .as_str()
        .expect("gated response must include revision_card_id");

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, before.content,
        "PageWrite refresh must not overwrite human-owned page prose"
    );
    assert_eq!(
        after.version, before.version,
        "gated PageWrite refresh must not bump the protected page version"
    );

    let revisions = db.list_pending_revisions(10).await.unwrap();
    let card = revisions
        .iter()
        .find(|r| r.revision_source_id == revision_card_id)
        .expect("revision card must be visible in pending revisions");
    assert_eq!(card.target_source_id, page_id);
    assert_eq!(card.revision_content, proposed_content);
}

#[tokio::test]
async fn refresh_page_machine_owned_page_goes_through_page_write_changelog() {
    let _guard = data_dir_lock().lock().await;
    let _config = WritableKnowledgeConfig::new();
    let (app, db, _dir) = test_app_with_db().await;
    let mem_id = "mem_route_pagewrite_refresh";
    let source_content = "Rust async refresh content stays grounded in its source memory";
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_string(),
        source_id: mem_id.to_string(),
        title: "Rust async refresh source".to_string(),
        content: source_content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("test".to_string()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .unwrap();

    let page_id =
        create_distilled_page_fixture(&db, "Rust Async Refresh", source_content, &[mem_id]).await;

    let refreshed_content =
        "Rust async refresh content stays grounded in its source memory and records the route write";
    let (status, body) = json_put(
        &app,
        &format!("/api/pages/{page_id}"),
        serde_json::json!({
            "content": refreshed_content,
            "source_memory_ids": [mem_id]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["ok"], serde_json::json!(true), "body: {body}");
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.content, refreshed_content);
    assert!(!page.user_edited);
    let changelog = page.changelog.as_deref().unwrap_or("[]");
    let entries: serde_json::Value = serde_json::from_str(changelog).unwrap();
    let has_agent_refresh_entry = entries.as_array().is_some_and(|items| {
        items.iter().any(|entry| {
            entry.get("edited_by").and_then(|value| value.as_str()) == Some("agent_refresh")
        })
    });
    assert!(
        has_agent_refresh_entry,
        "HTTP refresh must route through PageWrite changelog, got {changelog}"
    );
}

#[tokio::test]
async fn refresh_rejects_reserved_markers_before_card_page_or_projection_mutation() {
    let _guard = data_dir_lock().lock().await;
    let config = WritableKnowledgeConfig::new();
    let (app, db, _dir) = test_app_with_db().await;
    let mem_id = "mem_route_reserved_refresh";
    let source_content =
        "Rust async refresh content stays grounded in its source memory and exact source bytes";
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_string(),
        source_id: mem_id.to_string(),
        title: "Reserved refresh source".to_string(),
        content: source_content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("test".to_string()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .unwrap();

    let authored_id =
        create_authored_page_fixture(&db, "Human-owned exact source", source_content).await;
    let distilled_id =
        create_distilled_page_fixture(&db, "Machine-owned exact source", source_content, &[mem_id])
            .await;
    let projection_before = knowledge_snapshot(&config.pages_dir());
    let pending_before = db.list_pending_revisions(100).await.unwrap();

    for (ownership, page_id) in [("human", authored_id), ("machine", distilled_id)] {
        let page_before = db.get_page(&page_id).await.unwrap().unwrap();
        let history_before = db.list_page_history(&page_id, 100).await.unwrap();

        for (case, marker_content) in reserved_marker_cases() {
            let content = format!("{source_content}\n\n{marker_content}");
            let (status, body) = json_put(
                &app,
                &format!("/api/pages/{page_id}"),
                serde_json::json!({
                    "content": content,
                    "source_memory_ids": [mem_id]
                }),
            )
            .await;

            assert_eq!(
                status,
                StatusCode::UNPROCESSABLE_ENTITY,
                "{ownership}-owned {case} must map to HTTP 422; body: {body}"
            );
            let page_after = db.get_page(&page_id).await.unwrap().unwrap();
            assert_eq!(
                page_after.content, page_before.content,
                "{ownership} {case}"
            );
            assert_eq!(
                page_after.version, page_before.version,
                "{ownership} {case}"
            );
            assert_eq!(
                page_after.source_memory_ids, page_before.source_memory_ids,
                "{ownership} {case}"
            );
            assert_eq!(
                db.list_page_history(&page_id, 100).await.unwrap(),
                history_before,
                "{ownership} {case} must not append page history"
            );
            assert_eq!(
                db.list_pending_revisions(100).await.unwrap().len(),
                pending_before.len(),
                "{ownership} {case} must not stage a revision card"
            );
            assert_eq!(
                knowledge_snapshot(&config.pages_dir()),
                projection_before,
                "{ownership} {case} must not mutate projection files"
            );
        }
    }
}

/// Non-existent source_id uses the same static 404 as a scope mismatch.
#[tokio::test]
async fn memory_revisions_unknown_id_returns_404() {
    let (app, _dir) = test_app().await;
    let (status, body) = json_get(&app, "/api/memory/nonexistent_mem_id/revisions").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body: {body}");
    assert_eq!(body, serde_json::json!({"error": "memory not found"}));
}

/// Non-existent page id returns 404.
#[tokio::test]
async fn page_revisions_unknown_id_returns_404() {
    let (app, _dir) = test_app().await;
    let (status, _body) = json_get(&app, "/api/pages/nonexistent_page_id/revisions").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Known page returns 200 with correct envelope and empty entries list
/// (newly inserted pages have changelog = '[]').
#[tokio::test]
async fn page_revisions_known_page_returns_envelope() {
    let (app, db, _dir) = test_app_with_db().await;

    let page_id = create_authored_page_fixture(
        &db,
        "Test Revision Page",
        "Full content of the test page for revision surfacing.",
    )
    .await;

    let (status, body) = json_get(&app, &format!("/api/pages/{page_id}/revisions")).await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(
        body["page_id"].as_str(),
        Some(page_id.as_str()),
        "envelope page_id mismatch: {body}"
    );
    assert_eq!(
        body["current_version"].as_i64(),
        Some(1),
        "newly inserted page should have version=1: {body}"
    );
    assert_eq!(
        body["user_edited"].as_bool(),
        Some(false),
        "newly inserted page should not be user_edited: {body}"
    );
    assert!(
        body["entries"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(false),
        "fresh page should have empty changelog entries: {body}"
    );
}

// ── POST /api/memory/list with confirmed filter ──────────────────────────────

#[tokio::test]
async fn list_memories_confirmed_false_filters_unconfirmed() {
    use wenlan_core::sources::RawDocument;

    let (app, db, _dir) = test_app_with_db().await;

    // Insert one confirmed and one unconfirmed memory directly via DB.
    let doc_confirmed = RawDocument {
        source: "memory".to_string(),
        source_id: "mem_conf_001".to_string(),
        title: "Confirmed memory".to_string(),
        content: "Some confirmed content about widgets.".to_string(),
        last_modified: 1_000_000,
        ..Default::default()
    };
    let doc_unconfirmed = RawDocument {
        source: "memory".to_string(),
        source_id: "mem_unconf_001".to_string(),
        title: "Unconfirmed memory".to_string(),
        content: "Some unconfirmed content about gears.".to_string(),
        last_modified: 2_000_000,
        ..Default::default()
    };
    db.upsert_documents(vec![doc_confirmed, doc_unconfirmed])
        .await
        .unwrap();
    // Flip the confirmed memory to confirmed=true via the public API
    db.confirm_memory("mem_conf_001").await.unwrap();

    // POST /api/memory/list with confirmed=false
    let (status, body) = json_post(
        &app,
        "/api/memory/list",
        None,
        serde_json::json!({"confirmed": false, "limit": 50}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");

    let memories = body["memories"].as_array().expect("memories array");
    let ids: Vec<&str> = memories
        .iter()
        .filter_map(|m| m["source_id"].as_str())
        .collect();
    assert!(
        ids.contains(&"mem_unconf_001"),
        "unconfirmed memory must appear; got: {ids:?}"
    );
    assert!(
        !ids.contains(&"mem_conf_001"),
        "confirmed memory must not appear; got: {ids:?}"
    );

    // created_at must be present and non-null
    let unconf = memories
        .iter()
        .find(|m| m["source_id"].as_str() == Some("mem_unconf_001"))
        .expect("unconfirmed memory in response");
    assert!(
        unconf["created_at"].is_number(),
        "created_at must be a number; got: {unconf}"
    );
}

// ── POST /api/memory/:id/update-page demotes review_status ──────────────────

#[tokio::test]
async fn manual_page_edit_route_demotes_review_status_to_unconfirmed() {
    let (app, db, _dir) = test_app_with_db().await;
    let page_id = create_authored_page_fixture(&db, "T", "original body").await;
    assert_eq!(
        db.get_page(&page_id).await.unwrap().unwrap().review_status,
        "confirmed",
        "freshly seeded page starts confirmed"
    );
    // Manual edit via the HTTP route.
    let (status, _body) = json_post(
        &app,
        &format!("/api/memory/{page_id}/update-page"),
        Some("tester"),
        serde_json::json!({ "content": "manually edited body" }),
    )
    .await;
    assert_eq!(
        status,
        axum::http::StatusCode::OK,
        "update route should 200"
    );
    // The trust boundary must cover this bypass route.
    assert_eq!(
        db.get_page(&page_id).await.unwrap().unwrap().review_status,
        "unconfirmed",
        "manual edit must demote the page to unconfirmed"
    );
}
