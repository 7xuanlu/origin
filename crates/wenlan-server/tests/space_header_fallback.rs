// SPDX-License-Identifier: Apache-2.0
//! Integration tests: X-Origin-Space header fallback in space-aware POST handlers.
//!
//! - When body omits `space`, the header value is used.
//! - When body supplies `space`, the body wins regardless of the header.
//! - An unregistered named header fails closed without mutating data.

mod common;

use axum::body::Body;
use axum::http::{HeaderValue, Request, StatusCode};
use std::sync::OnceLock;
use tower::ServiceExt;
use wenlan_types::responses::{
    CreateEntityResponse, CreatePageResponse, DecisionsResponse, ListMemoriesResponse,
    NurtureCardsResponse, SearchMemoryResponse, SearchResponse, StoreMemoryResponse,
};

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
}

impl Drop for WritableKnowledgeConfig {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
            None => std::env::remove_var("WENLAN_DATA_DIR"),
        }
    }
}

async fn seed_confirmed_memory(
    db: &std::sync::Arc<wenlan_core::db::MemoryDB>,
    source_id: &str,
    content: &str,
    space: Option<&str>,
) {
    seed_memory_with_stability(db, source_id, content, "fact", "confirmed", space).await;
}

async fn seed_memory_with_stability(
    db: &std::sync::Arc<wenlan_core::db::MemoryDB>,
    source_id: &str,
    content: &str,
    memory_type: &str,
    stability: &str,
    space: Option<&str>,
) {
    db.upsert_documents(vec![wenlan_core::sources::RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: format!("title-{source_id}"),
        content: content.to_string(),
        memory_type: Some(memory_type.to_string()),
        space: space.map(str::to_string),
        last_modified: chrono::Utc::now().timestamp(),
        pending_revision: false,
        ..Default::default()
    }])
    .await
    .expect("seed memory must upsert");
    db.set_stability(source_id, stability)
        .await
        .expect("seed memory must set requested stability");
}

async fn body_as_json<T: serde::de::DeserializeOwned>(response: axum::http::Response<Body>) -> T {
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).expect("response body is valid JSON of expected type")
}

// ===== /api/memory/store (handle_store_memory) — Task 2, already done =====

#[tokio::test]
async fn header_used_when_body_omits_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("career", None, false).await.unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/store")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "career")
        .body(Body::from(
            serde_json::json!({
                "content": "header fallback test memory content",
                "memory_type": "fact"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "store must return 200");

    let stored: StoreMemoryResponse = body_as_json(res).await;
    let space = db
        .get_memory_space(&stored.source_id)
        .await
        .expect("get_memory_space must not fail");
    assert_eq!(
        space.as_deref(),
        Some("career"),
        "stored memory must have space=career from header, got: {space:?}"
    );
}

#[tokio::test]
async fn body_space_wins_over_header() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("health", None, false).await.unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/store")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "career")
        .body(Body::from(
            serde_json::json!({
                "content": "body space wins test memory content here",
                "memory_type": "fact",
                "space": "health"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "store must return 200");

    let stored: StoreMemoryResponse = body_as_json(res).await;
    let space = db
        .get_memory_space(&stored.source_id)
        .await
        .expect("get_memory_space must not fail");
    assert_eq!(
        space.as_deref(),
        Some("health"),
        "stored memory must have space=health from body (not career from header), got: {space:?}"
    );
}

#[tokio::test]
async fn unregistered_header_space_is_rejected_without_mutation() {
    let (router, _tmp, db) = common::test_app().await;
    let content = "unregistered header space must fail without storing content";

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/store")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "surprise-topic")
        .body(Body::from(
            serde_json::json!({
                "content": content,
                "memory_type": "fact"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "store must reject an unregistered header Space"
    );
    let error: serde_json::Value = body_as_json(res).await;
    assert!(
        error["error"]
            .as_str()
            .is_some_and(|message| message.contains("not registered")),
        "{error}"
    );
    assert!(
        !db.has_memory_content(content).await.unwrap(),
        "a rejected header Space must not persist the memory"
    );
    assert!(
        db.get_space("surprise-topic").await.unwrap().is_none(),
        "a rejected header Space must not auto-create a spaces row"
    );
}

// ===== /api/memory/search (handle_search_memory) =====

#[tokio::test]
async fn search_memory_header_fallback_returns_200() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("career", None, false).await.unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/search")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "career")
        .body(Body::from(
            serde_json::json!({
                "query": "test query"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "search_memory with header space must return 200"
    );
}

#[tokio::test]
async fn search_memory_unregistered_header_is_rejected() {
    let (router, _tmp, _db) = common::test_app().await;
    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/search")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "not-a-registered-space")
        .body(Body::from(
            serde_json::json!({
                "query": "espresso calibration fallback",
                "limit": 10
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn search_memory_malformed_space_headers_are_rejected() {
    for (preferred, legacy) in [(true, true), (false, true)] {
        let (router, _tmp, _db) = common::test_app().await;
        let mut builder = Request::builder()
            .method("POST")
            .uri("/api/memory/search")
            .header("Content-Type", "application/json");
        if legacy {
            builder = builder.header("X-Origin-Space", "career");
        }
        let mut request = builder
            .body(Body::from(r#"{"query":"malformed header canary"}"#))
            .unwrap();
        let malformed = HeaderValue::from_bytes(&[0xff]).unwrap();
        if preferred {
            request.headers_mut().insert("X-Wenlan-Space", malformed);
        } else {
            request.headers_mut().insert("X-Origin-Space", malformed);
        }

        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body: serde_json::Value = body_as_json(response).await;
        assert_eq!(body["error"], "invalid Space header");
    }
}

#[tokio::test]
async fn search_memory_uncategorized_filter_matches_only_null_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("alpha", None, false).await.unwrap();
    seed_confirmed_memory(
        &db,
        "uncategorized_search_memory",
        "uncategorized sentinel search match from null space",
        None,
    )
    .await;
    seed_confirmed_memory(
        &db,
        "alpha_search_memory",
        "uncategorized sentinel search match from alpha space",
        Some("alpha"),
    )
    .await;

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/search")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "query": "uncategorized sentinel search match",
                "space": "uncategorized",
                "limit": 10
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "search_memory must return 200"
    );
    let body: SearchMemoryResponse = body_as_json(res).await;
    let ids = body
        .results
        .iter()
        .map(|result| result.source_id.as_str())
        .collect::<Vec<_>>();
    assert!(
        ids.contains(&"uncategorized_search_memory"),
        "uncategorized must include NULL-space memory; got {ids:?}"
    );
    assert!(
        !ids.contains(&"alpha_search_memory"),
        "uncategorized must not become unscoped and include registered-space memories; got {ids:?}"
    );
}

// ===== /api/memory/list (handle_list_memories) =====

#[tokio::test]
async fn list_memories_header_fallback_returns_200() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("health", None, false).await.unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/list")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "health")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "list_memories with header space must return 200"
    );
}

#[tokio::test]
async fn list_memories_unregistered_header_is_rejected() {
    let (router, _tmp, _db) = common::test_app().await;
    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/list")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "ghost-list-space")
        .body(Body::from(
            serde_json::json!({
                "limit": 10,
                "confirmed": true
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn list_memories_uncategorized_filter_matches_only_null_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("alpha", None, false).await.unwrap();
    seed_confirmed_memory(
        &db,
        "uncategorized_list_memory",
        "uncategorized list sentinel null space",
        None,
    )
    .await;
    seed_confirmed_memory(
        &db,
        "alpha_list_memory",
        "uncategorized list sentinel alpha space",
        Some("alpha"),
    )
    .await;

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/list")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "space": "uncategorized",
                "limit": 10,
                "confirmed": true
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "list_memories must return 200"
    );
    let body: ListMemoriesResponse = body_as_json(res).await;
    let ids = body
        .memories
        .iter()
        .map(|memory| memory.source_id.as_str())
        .collect::<Vec<_>>();
    assert!(
        ids.contains(&"uncategorized_list_memory"),
        "uncategorized must include NULL-space memories; got {ids:?}"
    );
    assert!(
        !ids.contains(&"alpha_list_memory"),
        "uncategorized must not become unscoped and include registered-space memories; got {ids:?}"
    );
}

// ===== /api/search (handle_search) =====

#[tokio::test]
async fn search_header_fallback_returns_200() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("work", None, false).await.unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/search")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "work")
        .body(Body::from(
            serde_json::json!({
                "query": "test query"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "search with header space must return 200"
    );
}

#[tokio::test]
async fn search_unregistered_header_is_rejected() {
    let (router, _tmp, _db) = common::test_app().await;
    let req = Request::builder()
        .method("POST")
        .uri("/api/search")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "ghost-general-space")
        .body(Body::from(
            serde_json::json!({
                "query": "violin resin fallback",
                "limit": 10
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn search_uncategorized_filter_matches_only_null_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("alpha", None, false).await.unwrap();
    seed_confirmed_memory(
        &db,
        "uncategorized_general_search",
        "uncategorized general sentinel null space",
        None,
    )
    .await;
    seed_confirmed_memory(
        &db,
        "alpha_general_search",
        "uncategorized general sentinel alpha space",
        Some("alpha"),
    )
    .await;

    let req = Request::builder()
        .method("POST")
        .uri("/api/search")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "query": "uncategorized general sentinel",
                "space": "uncategorized",
                "limit": 10
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "search must return 200");
    let body: SearchResponse = body_as_json(res).await;
    let ids = body
        .results
        .iter()
        .map(|result| result.source_id.as_str())
        .collect::<Vec<_>>();
    assert!(
        ids.contains(&"uncategorized_general_search"),
        "uncategorized must include NULL-space memory; got {ids:?}"
    );
    assert!(
        !ids.contains(&"alpha_general_search"),
        "uncategorized must not become unscoped and include registered-space memories; got {ids:?}"
    );
}

// ===== /api/context (handle_context) =====

#[tokio::test]
async fn chat_context_header_fallback_returns_200() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("personal", None, false).await.unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/context")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "personal")
        .body(Body::from(
            serde_json::json!({
                "query": "what do I like?"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "chat_context with header space must return 200"
    );
}

#[tokio::test]
async fn chat_context_unregistered_header_is_rejected() {
    let (router, _tmp, _db) = common::test_app().await;
    let req = Request::builder()
        .method("POST")
        .uri("/api/context")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "ghost-context-space")
        .body(Body::from(
            serde_json::json!({
                "query": "fountain pen ink fallback",
                "max_chunks": 10
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// ===== /api/memory/entities (handle_list_entities) =====

#[tokio::test]
async fn list_entities_header_fallback_returns_200() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("work", None, false).await.unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/entities/list")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "work")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "list_entities with header space must return 200"
    );
}

#[tokio::test]
async fn list_entities_unregistered_header_is_rejected() {
    let (router, _tmp, _db) = common::test_app().await;
    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/entities/list")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "ghost-entity-space")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// ===== GET /api/memory/nurture (handle_get_nurture_cards) =====

#[tokio::test]
async fn nurture_unregistered_query_space_is_rejected() {
    let (router, _tmp, _db) = common::test_app().await;
    let req = Request::builder()
        .method("GET")
        .uri("/api/memory/nurture?space=ghost-nurture-space&limit=10")
        .body(Body::empty())
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn nurture_empty_query_space_falls_back_to_unscoped() {
    let (router, _tmp, db) = common::test_app().await;
    seed_memory_with_stability(
        &db,
        "empty_query_nurture_memory",
        "empty nurture query space should include this lap steel tuning memory",
        "fact",
        "new",
        None,
    )
    .await;

    let req = Request::builder()
        .method("GET")
        .uri("/api/memory/nurture?space=&limit=10")
        .body(Body::empty())
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "nurture must return 200");
    let body: NurtureCardsResponse = body_as_json(res).await;
    assert!(
        body.cards
            .iter()
            .any(|card| card.source_id == "empty_query_nurture_memory"),
        "empty query spaces must not filter out unscoped nurture cards; got {:?}",
        body.cards
            .iter()
            .map(|card| &card.source_id)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn nurture_uncategorized_filter_matches_only_null_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("alpha", None, false).await.unwrap();
    seed_memory_with_stability(
        &db,
        "uncategorized_nurture_memory",
        "uncategorized nurture sentinel null space",
        "fact",
        "new",
        None,
    )
    .await;
    seed_memory_with_stability(
        &db,
        "alpha_nurture_memory",
        "uncategorized nurture sentinel alpha space",
        "fact",
        "new",
        Some("alpha"),
    )
    .await;

    let req = Request::builder()
        .method("GET")
        .uri("/api/memory/nurture?space=uncategorized&limit=10")
        .body(Body::empty())
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "nurture must return 200");
    let body: NurtureCardsResponse = body_as_json(res).await;
    let ids = body
        .cards
        .iter()
        .map(|card| card.source_id.as_str())
        .collect::<Vec<_>>();
    assert!(
        ids.contains(&"uncategorized_nurture_memory"),
        "uncategorized must include NULL-space nurture cards; got {ids:?}"
    );
    assert!(
        !ids.contains(&"alpha_nurture_memory"),
        "uncategorized must not become unscoped and include registered-space nurture cards; got {ids:?}"
    );
}

// ===== GET /api/pages (handle_list_pages) =====

#[tokio::test]
async fn list_pages_uncategorized_filter_matches_only_null_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("alpha", None, false).await.unwrap();
    let uncategorized_id = common::create_page_fixture(
        &db,
        "Uncategorized Page",
        "uncategorized page body",
        Some("decision"),
        &[],
        "authored",
    )
    .await;
    db.set_page_workspace(&uncategorized_id, None)
        .await
        .unwrap();
    let alpha_id = common::create_page_fixture(
        &db,
        "Alpha Page",
        "alpha page body",
        Some("recap"),
        &[],
        "authored",
    )
    .await;
    db.set_page_workspace(&alpha_id, Some("alpha"))
        .await
        .unwrap();
    let literal_id = common::create_page_fixture(
        &db,
        "Literal Uncategorized Workspace Page",
        "literal uncategorized workspace page body",
        Some("decision"),
        &[],
        "authored",
    )
    .await;
    db.set_page_workspace(&literal_id, Some("uncategorized"))
        .await
        .unwrap();

    let req = Request::builder()
        .method("GET")
        .uri("/api/pages?space=uncategorized&limit=10")
        .body(Body::empty())
        .unwrap();

    let res = router.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "list pages must return 200");
    let body: serde_json::Value = body_as_json(res).await;
    let ids = body["pages"]
        .as_array()
        .expect("pages must be an array")
        .iter()
        .filter_map(|page| page["id"].as_str())
        .collect::<Vec<_>>();
    assert!(
        ids.contains(&uncategorized_id.as_str()),
        "uncategorized must include NULL-space pages; got {ids:?}"
    );
    assert!(
        !ids.contains(&alpha_id.as_str()),
        "uncategorized must not become unscoped and include registered-space pages; got {ids:?}"
    );
    assert!(
        !ids.contains(&literal_id.as_str()),
        "the reserved selector must match NULL, not a literal workspace value; got {ids:?}"
    );

    let global = Request::builder()
        .method("GET")
        .uri("/api/pages?limit=10")
        .body(Body::empty())
        .unwrap();
    let global: serde_json::Value = body_as_json(router.oneshot(global).await.unwrap()).await;
    assert!(
        global["pages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|page| page["id"] == literal_id),
        "Global must retain a Page whose literal workspace is uncategorized"
    );
}

// ===== GET /api/decisions (handle_list_decisions) =====

#[tokio::test]
async fn decisions_unregistered_query_space_is_rejected() {
    let (router, _tmp, _db) = common::test_app().await;
    let req = Request::builder()
        .method("GET")
        .uri("/api/decisions?space=ghost-decision-space&limit=10")
        .body(Body::empty())
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn decisions_empty_query_space_falls_back_to_unscoped() {
    let (router, _tmp, db) = common::test_app().await;
    seed_memory_with_stability(
        &db,
        "empty_query_decision_memory",
        "empty decision query space should include this storage adapter decision",
        "decision",
        "confirmed",
        None,
    )
    .await;

    let req = Request::builder()
        .method("GET")
        .uri("/api/decisions?space=&limit=10")
        .body(Body::empty())
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "decisions must return 200");
    let body: DecisionsResponse = body_as_json(res).await;
    assert!(
        body.decisions
            .iter()
            .any(|decision| decision.source_id == "empty_query_decision_memory"),
        "empty query spaces must not filter out unscoped decisions; got {:?}",
        body.decisions
            .iter()
            .map(|decision| &decision.source_id)
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn decisions_uncategorized_filter_matches_only_null_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("alpha", None, false).await.unwrap();
    seed_memory_with_stability(
        &db,
        "uncategorized_decision_memory",
        "uncategorized decision sentinel null space",
        "decision",
        "confirmed",
        None,
    )
    .await;
    seed_memory_with_stability(
        &db,
        "alpha_decision_memory",
        "uncategorized decision sentinel alpha space",
        "decision",
        "confirmed",
        Some("alpha"),
    )
    .await;

    let req = Request::builder()
        .method("GET")
        .uri("/api/decisions?space=uncategorized&limit=10")
        .body(Body::empty())
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "decisions must return 200");
    let body: DecisionsResponse = body_as_json(res).await;
    let ids = body
        .decisions
        .iter()
        .map(|decision| decision.source_id.as_str())
        .collect::<Vec<_>>();
    assert!(
        ids.contains(&"uncategorized_decision_memory"),
        "uncategorized must include NULL-space decisions; got {ids:?}"
    );
    assert!(
        !ids.contains(&"alpha_decision_memory"),
        "uncategorized must not become unscoped and include registered-space decisions; got {ids:?}"
    );
}

#[tokio::test]
async fn list_entities_uncategorized_filter_matches_only_null_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("alpha", None, false).await.unwrap();
    db.store_entity("Uncategorized Entity", "person", None, None, None)
        .await
        .expect("seed uncategorized entity must store");
    db.store_entity("Alpha Entity", "person", Some("alpha"), None, None)
        .await
        .expect("seed alpha entity must store");

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/entities/list")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "space": "uncategorized"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "list_entities must return 200"
    );
    let body: serde_json::Value = body_as_json(res).await;
    let names = body["entities"]
        .as_array()
        .expect("entities must be an array")
        .iter()
        .filter_map(|entity| entity["name"].as_str())
        .collect::<Vec<_>>();
    assert!(
        names.contains(&"Uncategorized Entity"),
        "uncategorized must include NULL-space entities; got {names:?}"
    );
    assert!(
        !names.contains(&"Alpha Entity"),
        "uncategorized must not become unscoped and include registered-space entities; got {names:?}"
    );
}

// ===== POST /api/memory/entities (handle_create_entity) =====

#[tokio::test]
async fn create_entity_uses_header_when_body_omits_space() {
    let (router, _tmp, db) = common::test_app().await;
    db.create_space("career", None, false).await.unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/entities")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "career")
        .body(Body::from(
            serde_json::json!({
                "name": "test_ent_space_header",
                "entity_type": "person"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "create_entity must return 200"
    );

    let created: CreateEntityResponse = body_as_json(res).await;
    let detail = db
        .get_entity_detail(&created.id)
        .await
        .expect("get_entity_detail must not fail");
    assert_eq!(
        detail.entity.space.as_deref(),
        Some("career"),
        "created entity must have space=career from header, got: {:?}",
        detail.entity.space
    );
}

#[tokio::test]
async fn create_entity_unregistered_header_space_is_rejected_without_mutation() {
    let (router, _tmp, db) = common::test_app().await;

    let req = Request::builder()
        .method("POST")
        .uri("/api/memory/entities")
        .header("Content-Type", "application/json")
        .header("X-Origin-Space", "surprise-entity")
        .body(Body::from(
            serde_json::json!({
                "name": "test_ent_unregistered_header",
                "entity_type": "person"
            })
            .to_string(),
        ))
        .unwrap();

    let res = router.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "create_entity must reject an unregistered header Space"
    );
    let error: serde_json::Value = body_as_json(res).await;
    assert!(
        error["error"]
            .as_str()
            .is_some_and(|message| message.contains("not registered")),
        "{error}"
    );
    assert!(
        db.search_entities_by_name("test_ent_unregistered_header")
            .await
            .unwrap()
            .is_empty(),
        "a rejected header Space must not persist the entity"
    );
    assert!(
        db.get_space("surprise-entity").await.unwrap().is_none(),
        "a rejected header Space must not auto-create a spaces row"
    );
}

// ===== POST /api/pages (handle_create_page) =====

#[tokio::test]
async fn create_page_uses_header_when_body_omits_space() {
    let _env_lock = data_dir_lock().lock().await;
    let _config = WritableKnowledgeConfig::new();
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("career", None, false).await.unwrap();

    // Seed a source memory via HTTP so we have a valid source_id to cite.
    let content = "Rust is a systems programming language with memory safety guarantees";
    let store_res = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memory/store")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "content": content,
                        "memory_type": "fact"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        store_res.status(),
        StatusCode::OK,
        "seed memory store must return 200"
    );
    let stored: StoreMemoryResponse = body_as_json(store_res).await;
    let source_id = stored.source_id;

    // Create a page citing that memory, omitting `space` in body — header must fill it.
    let page_content = "Rust provides memory safety through ownership and borrowing.";
    let create_res = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/pages")
                .header("Content-Type", "application/json")
                .header("X-Origin-Space", "career")
                .body(Body::from(
                    serde_json::json!({
                        "title": "Rust Systems Language",
                        "content": page_content,
                        "source_memory_ids": [source_id],
                        "creation_kind": "authored"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        create_res.status(),
        StatusCode::OK,
        "create_page must return 200"
    );

    let created: CreatePageResponse = body_as_json(create_res).await;
    let page = db
        .get_page(&created.id)
        .await
        .expect("get_page must not fail")
        .expect("page must exist after creation");
    assert_eq!(
        page.space.as_deref(),
        Some("career"),
        "created page must have space=career from header, got: {:?}",
        page.space
    );
    assert_eq!(
        page.workspace.as_deref(),
        Some("career"),
        "created page must have workspace=career from header, got: {:?}",
        page.workspace
    );
}

#[tokio::test]
async fn create_page_unregistered_header_space_is_rejected_without_mutation() {
    let _env_lock = data_dir_lock().lock().await;
    let _config = WritableKnowledgeConfig::new();
    let (router, _tmp, db) = common::test_app_no_gate().await;

    let content = "Rust is a systems programming language with memory safety guarantees";
    let store_res = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memory/store")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "content": content,
                        "memory_type": "fact"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        store_res.status(),
        StatusCode::OK,
        "seed memory store must return 200"
    );
    let stored: StoreMemoryResponse = body_as_json(store_res).await;
    let source_id = stored.source_id;

    let page_content = "Rust provides memory safety through ownership and borrowing.";
    let create_res = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/pages")
                .header("Content-Type", "application/json")
                .header("X-Origin-Space", "surprise-page")
                .body(Body::from(
                    serde_json::json!({
                        "title": "Rust Systems Language Unscoped",
                        "content": page_content,
                        "source_memory_ids": [source_id],
                        "creation_kind": "authored"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        create_res.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "create_page must reject an unregistered header Space"
    );
    let error: serde_json::Value = body_as_json(create_res).await;
    assert!(
        error["error"]
            .as_str()
            .is_some_and(|message| message.contains("not registered")),
        "{error}"
    );
    assert!(
        db.list_pages("active", 100, 0).await.unwrap().is_empty(),
        "a rejected header Space must not persist the page"
    );
    assert!(
        db.get_space("surprise-page").await.unwrap().is_none(),
        "a rejected header Space must not auto-create a spaces row"
    );
}

#[tokio::test]
async fn set_document_space_unregistered_space_unassigns_without_auto_create() {
    let (router, _tmp, db) = common::test_app().await;
    seed_confirmed_memory(
        &db,
        "set_doc_space_memory",
        "document space update should not persist unknown space labels",
        None,
    )
    .await;

    let res = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/documents/set_doc_space_memory/space")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "space_name": "ghost-write-space"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "set document space must return 200"
    );

    let space = db
        .get_memory_space("set_doc_space_memory")
        .await
        .expect("get_memory_space must not fail");
    assert_eq!(
        space, None,
        "unregistered document-space writes must fall back to unscoped"
    );
    assert!(
        db.get_space("ghost-write-space").await.unwrap().is_none(),
        "unregistered document-space writes must not auto-create a spaces row"
    );
}

#[tokio::test]
async fn update_memory_unregistered_space_unassigns_without_auto_create() {
    let (router, _tmp, db) = common::test_app().await;
    seed_confirmed_memory(
        &db,
        "update_memory_space_memory",
        "memory update should not persist unknown space labels",
        None,
    )
    .await;

    let res = router
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/memory/update_memory_space_memory/update")
                .header("Content-Type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "space": "ghost-update-space"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "update memory must return 200"
    );

    let space = db
        .get_memory_space("update_memory_space_memory")
        .await
        .expect("get_memory_space must not fail");
    assert_eq!(
        space, None,
        "unregistered update-memory spaces must fall back to unscoped"
    );
    assert!(
        db.get_space("ghost-update-space").await.unwrap().is_none(),
        "unregistered update-memory spaces must not auto-create a spaces row"
    );
}
