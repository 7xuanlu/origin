use axum::body::{to_bytes, Body};
use axum::http::Request;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_types::requests::CreateConceptRequest;

async fn fixture() -> (crate::router::AppRouter, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
        Arc::new(wenlan_core::events::NoopEmitter);
    let db = Arc::new(
        wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("database"),
    );
    for space in ["work", "personal"] {
        db.create_space(space, None, false).await.expect("space");
        db.upsert_documents(vec![wenlan_core::sources::RawDocument {
            source: "memory".to_string(),
            source_id: format!("{space}-memory"),
            title: format!("{space} memory"),
            content: "scope probe memory content".to_string(),
            last_modified: 1,
            memory_type: Some("fact".to_string()),
            space: Some(space.to_string()),
            source_agent: Some("test".to_string()),
            confirmed: Some(true),
            ..Default::default()
        }])
        .await
        .expect("memory");
        wenlan_core::post_write::create_page(
            &db,
            CreateConceptRequest {
                title: format!("{space} scopeprobe"),
                content: "scopeprobe page content".to_string(),
                summary: None,
                entity_id: None,
                space: (Some(space.to_string())).into(),
                source_memory_ids: Vec::new(),
                creation_kind: Some("authored".to_string()),
                workspace: Some(space.to_string()),
            },
            "test",
            None,
        )
        .await
        .expect("page");
    }
    db.log_accesses(&["work-memory".to_string()])
        .await
        .expect("access log");
    db.set_document_tags("memory", "work-memory", vec!["scope-tag".to_string()])
        .await
        .expect("document tags");
    let state = Arc::new(RwLock::new(crate::state::ServerState {
        db: Some(db),
        ..Default::default()
    }));
    (crate::router::build_router(state), tmp)
}

async fn json(app: crate::router::AppRouter, request: Request<Body>) -> serde_json::Value {
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), 200);
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("json")
}

fn post(uri: &str, header: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-wenlan-space", header)
        .body(Body::from(body.to_string()))
        .expect("request")
}

#[tokio::test]
async fn list_body_precedes_header_and_unknown_body_is_rejected() {
    let (app, _tmp) = fixture().await;
    let scoped = json(
        app.clone(),
        post(
            "/api/memory/list",
            "personal",
            r#"{"space":"work","limit":20}"#,
        ),
    )
    .await;
    let scoped_ids = source_ids(&scoped["memories"]);
    assert_eq!(scoped_ids, vec!["work-memory"]);

    let rejected = app
        .oneshot(post(
            "/api/memory/list",
            "work",
            r#"{"space":"missing","limit":20}"#,
        ))
        .await
        .expect("response");
    assert_eq!(
        rejected.status(),
        axum::http::StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn direct_memory_detail_hides_conflicting_space_owner() {
    let (app, _tmp) = fixture().await;
    let mismatch = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/memory/work-memory/detail")
                .header("x-wenlan-space", "personal")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let missing = app
        .oneshot(
            Request::builder()
                .uri("/api/memory/missing-memory/detail")
                .header("x-wenlan-space", "personal")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(mismatch.status(), axum::http::StatusCode::NOT_FOUND);
    assert_eq!(missing.status(), axum::http::StatusCode::NOT_FOUND);
    let mismatch_body = to_bytes(mismatch.into_body(), usize::MAX)
        .await
        .expect("mismatch body");
    let missing_body = to_bytes(missing.into_body(), usize::MAX)
        .await
        .expect("missing body");
    assert_eq!(mismatch_body, missing_body);
}

#[tokio::test]
async fn page_search_uses_space_header_and_excludes_cross_scope_rows() {
    let (app, _tmp) = fixture().await;
    let payload = json(
        app,
        post(
            "/api/pages/search",
            "work",
            r#"{"query":"scopeprobe","limit":20}"#,
        ),
    )
    .await;
    let workspaces = payload["pages"]
        .as_array()
        .expect("pages")
        .iter()
        .filter_map(|page| page["workspace"].as_str())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(workspaces, std::collections::BTreeSet::from(["work"]));
}

#[tokio::test]
async fn bulk_page_export_filters_workspace_before_materializing_files() {
    let (app, tmp) = fixture().await;
    let vault = tmp.path().join("bulk-export");
    let body = serde_json::json!({ "vault_path": vault }).to_string();

    let payload = json(app, post("/api/pages/export", "work", &body)).await;

    assert_eq!(payload["exported"], 1);
    assert_eq!(payload["failed"], 0);
    assert!(vault.join("work-scopeprobe.md").is_file());
    assert!(!vault.join("personal-scopeprobe.md").exists());
}

#[tokio::test]
async fn single_page_export_hides_conflicting_workspace_owner() {
    let (app, tmp) = fixture().await;
    let personal = json(
        app.clone(),
        Request::builder()
            .uri("/api/pages?space=personal")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    let personal_id = personal["pages"][0]["id"]
        .as_str()
        .expect("personal page id");
    let vault = tmp.path().join("single-export");
    let body = serde_json::json!({ "vault_path": vault }).to_string();

    let mismatch = app
        .clone()
        .oneshot(post(
            &format!("/api/pages/{personal_id}/export"),
            "work",
            &body,
        ))
        .await
        .expect("mismatch response");
    let missing = app
        .oneshot(post("/api/pages/missing-page/export", "work", &body))
        .await
        .expect("missing response");

    assert_eq!(mismatch.status(), axum::http::StatusCode::NOT_FOUND);
    assert_eq!(missing.status(), axum::http::StatusCode::NOT_FOUND);
    let mismatch_body = to_bytes(mismatch.into_body(), usize::MAX)
        .await
        .expect("mismatch body");
    let missing_body = to_bytes(missing.into_body(), usize::MAX)
        .await
        .expect("missing body");
    assert_eq!(mismatch_body, missing_body);
    assert!(!vault.exists());
}

#[tokio::test]
async fn home_stats_and_tags_observably_return_row_level_data() {
    let (app, _tmp) = fixture().await;
    let home = json(
        app.clone(),
        Request::builder()
            .uri("/api/home-stats")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(home["top_memories"][0]["source_id"], "work-memory");
    assert_eq!(home["top_memories"][0]["space"], "work");
    assert_eq!(
        home["top_memories"][0]["content"],
        "scope probe memory content"
    );

    let tags = json(
        app,
        Request::builder()
            .uri("/api/tags")
            .body(Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(
        tags["document_tags"]["memory::work-memory"],
        serde_json::json!(["scope-tag"])
    );
}

fn source_ids(value: &serde_json::Value) -> Vec<&str> {
    let mut ids = value
        .as_array()
        .expect("memories")
        .iter()
        .filter_map(|memory| memory["source_id"].as_str())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}
