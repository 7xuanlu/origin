// SPDX-License-Identifier: Apache-2.0
//! Q1 (stage c): `kind='entity'` dual-write shadow pages must appear on the
//! page browse/search surfaces (list, search, recent, recent-changes,
//! get-by-id, and its sub-resources) while staying excluded from export.
//! Mirrors the core-level fence-lift tests in `wenlan-core/src/db.rs` at the
//! HTTP layer, so a regression that only breaks the handler wiring (not the
//! underlying DB fn) is still caught.
//!
//! #708 narrowed WHICH shadow pages the browse surfaces show: an ESTABLISHED
//! entity appears, a detected or archived one does not, because the detected
//! set is an index the system grows on its own and listing it buries every
//! real page. Every fixture here therefore establishes its entity; that a
//! DETECTED one stays hidden is pinned in
//! `wenlan-core/src/db/entity_lifecycle_test.rs`.
mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde::Deserialize;
use tower::ServiceExt;
use wenlan_types::pages::Page;
use wenlan_types::responses::PageLinksResponse;
use wenlan_types::{
    ExportStats, ListPageRevisionsResponse, PageChange, PageSourceWithMemory, RecentActivityItem,
};

#[derive(Debug, Deserialize)]
struct PagesEnvelope {
    pages: Vec<Page>,
}

#[derive(Debug, Deserialize)]
struct PageEnvelope {
    page: Page,
}

async fn get(router: &common::AppRouter, uri: &str) -> axum::http::Response<Body> {
    router
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn post_json(
    router: &common::AppRouter,
    uri: &str,
    body: serde_json::Value,
) -> axum::http::Response<Body> {
    router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn put_json(
    router: &common::AppRouter,
    uri: &str,
    body: serde_json::Value,
) -> axum::http::Response<Body> {
    router
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn delete(router: &common::AppRouter, uri: &str) -> axum::http::Response<Body> {
    router
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn body_json<T: serde::de::DeserializeOwned>(resp: axum::http::Response<Body>) -> T {
    let bytes = axum::body::to_bytes(resp.into_body(), 256 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).expect("response body must deserialize")
}

/// Find a page's id by title through the public GET /api/pages listing --
/// the lookup path an HTTP client would actually use, rather than reaching
/// into `entity_page_map` directly from this HTTP-level test.
async fn find_page_id_by_title(router: &common::AppRouter, title: &str) -> String {
    let resp = get(router, "/api/pages?limit=200").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let listed: PagesEnvelope = body_json(resp).await;
    listed
        .pages
        .into_iter()
        .find(|p| p.title == title)
        .unwrap_or_else(|| panic!("page titled {title:?} must appear in GET /api/pages"))
        .id
}

#[tokio::test]
async fn list_pages_includes_entity_kind_shadow() {
    let (router, _tmp, db) = common::test_app().await;
    let entity_id = db
        .store_entity("HTTP List Shadow Marker", "person", None, None, None)
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();

    let resp = get(&router, "/api/pages").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let listed: PagesEnvelope = body_json(resp).await;
    assert!(
        listed
            .pages
            .iter()
            .any(|p| p.title == "HTTP List Shadow Marker"),
        "Q1: GET /api/pages must surface a kind='entity' shadow page"
    );
}

#[tokio::test]
async fn search_pages_includes_entity_kind_shadow() {
    let (router, _tmp, db) = common::test_app().await;
    let entity_id = db
        .store_entity("HTTP Search Shadow Marker", "person", None, None, None)
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();

    let resp = post_json(
        &router,
        "/api/pages/search",
        serde_json::json!({ "query": "HTTP Search Shadow Marker" }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let listed: PagesEnvelope = body_json(resp).await;
    assert!(
        listed
            .pages
            .iter()
            .any(|p| p.title == "HTTP Search Shadow Marker"),
        "Q1: POST /api/pages/search must surface a kind='entity' shadow page"
    );
}

#[tokio::test]
async fn recent_pages_includes_entity_kind_shadow() {
    let (router, _tmp, db) = common::test_app().await;
    let entity_id = db
        .store_entity("HTTP Recent Shadow Marker", "person", None, None, None)
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();

    let resp = get(&router, "/api/pages/recent").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let items: Vec<RecentActivityItem> = body_json(resp).await;
    assert!(
        items.iter().any(|i| i.title == "HTTP Recent Shadow Marker"),
        "Q1: GET /api/pages/recent must surface a kind='entity' shadow page"
    );
}

#[tokio::test]
async fn recent_page_changes_includes_entity_kind_shadow() {
    let (router, _tmp, db) = common::test_app().await;
    let entity_id = db
        .store_entity(
            "HTTP Recent Changes Shadow Marker",
            "person",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();

    let resp = get(&router, "/api/pages/recent-changes").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let changes: Vec<PageChange> = body_json(resp).await;
    assert!(
        changes
            .iter()
            .any(|c| c.title == "HTTP Recent Changes Shadow Marker"),
        "Q1: GET /api/pages/recent-changes must surface a kind='entity' shadow page"
    );
}

#[tokio::test]
async fn get_page_by_id_returns_entity_kind_shadow() {
    let (router, _tmp, db) = common::test_app().await;
    let entity_id = db
        .store_entity("HTTP Get By Id Shadow Marker", "person", None, None, None)
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();
    let id = find_page_id_by_title(&router, "HTTP Get By Id Shadow Marker").await;

    let resp = get(&router, &format!("/api/pages/{id}")).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "Q1: GET /api/pages/{{id}} must 200 for a shadow id, not 404"
    );
    let fetched: PageEnvelope = body_json(resp).await;
    assert_eq!(fetched.page.title, "HTTP Get By Id Shadow Marker");
}

#[tokio::test]
async fn shadow_sub_resources_return_empty_not_404() {
    let (router, _tmp, db) = common::test_app().await;
    let entity_id = db
        .store_entity(
            "HTTP Sub Resource Shadow Marker",
            "person",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();
    let id = find_page_id_by_title(&router, "HTTP Sub Resource Shadow Marker").await;

    let sources_resp = get(&router, &format!("/api/pages/{id}/sources")).await;
    assert_eq!(sources_resp.status(), StatusCode::OK);
    let sources: Vec<PageSourceWithMemory> = body_json(sources_resp).await;
    assert!(sources.is_empty(), "a shadow page has no real sources");

    let links_resp = get(&router, &format!("/api/pages/{id}/links")).await;
    assert_eq!(links_resp.status(), StatusCode::OK);
    let links: PageLinksResponse = body_json(links_resp).await;
    assert!(
        links.outbound.is_empty() && links.inbound.is_empty(),
        "a shadow page has no real wikilinks"
    );

    let revisions_resp = get(&router, &format!("/api/pages/{id}/revisions")).await;
    assert_eq!(revisions_resp.status(), StatusCode::OK);
    let revisions: ListPageRevisionsResponse = body_json(revisions_resp).await;
    assert!(
        revisions.entries.is_empty(),
        "a shadow page has no real revision history"
    );
}

#[tokio::test]
async fn export_stays_stub_free() {
    let (router, tmp, db) = common::test_app().await;
    common::create_page_fixture(
        &db,
        "HTTP Export Concept Page",
        "Real page content that must be exported.",
        None,
        &[],
        "authored",
    )
    .await;
    let entity_id = db
        .store_entity("HTTP Export Shadow Marker", "person", None, None, None)
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();
    let stub_id = find_page_id_by_title(&router, "HTTP Export Shadow Marker").await;

    let vault_path = tmp.path().join("vault");
    let resp = post_json(
        &router,
        "/api/pages/export",
        serde_json::json!({ "vault_path": vault_path.to_string_lossy() }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let stats: ExportStats = body_json(resp).await;
    assert_eq!(
        stats.exported, 1,
        "bulk export must export only the real page, not the entity shadow"
    );
    let written: Vec<_> = std::fs::read_dir(&vault_path)
        .expect("export must have created the vault dir")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    assert_eq!(
        written.len(),
        1,
        "exactly one file must land in the vault; got: {written:?}"
    );
    let exported_content = std::fs::read_to_string(vault_path.join(&written[0])).unwrap();
    assert!(
        !exported_content.contains("HTTP Export Shadow Marker"),
        "the shadow page's title must never appear in an exported file"
    );

    let single_resp = post_json(
        &router,
        &format!("/api/pages/{stub_id}/export"),
        serde_json::json!({ "vault_path": vault_path.to_string_lossy() }),
    )
    .await;
    assert_eq!(
        single_resp.status(),
        StatusCode::NOT_FOUND,
        "single-page export of a shadow id must 404, matching pre-lift behavior"
    );
}

// -- Fix wave (sol-review): Q1 makes shadow ids publicly discoverable, so the
// page MUTATION surfaces must fence them. A shadow id obtained through the
// public GET /api/pages listing must be rejected by archive, delete, manual
// update, agent refresh, and page-map node create -- while the read surface
// (GET /api/pages/{id}) stays 200. --

#[tokio::test]
async fn archive_of_shadow_id_is_rejected() {
    let (router, _tmp, db) = common::test_app().await;
    let entity_id = db
        .store_entity("HTTP Archive Fence Marker", "person", None, None, None)
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();
    let id = find_page_id_by_title(&router, "HTTP Archive Fence Marker").await;

    let resp = post_json(
        &router,
        &format!("/api/pages/{id}/archive"),
        serde_json::json!({}),
    )
    .await;
    assert!(
        resp.status().is_client_error(),
        "POST /api/pages/{{stub}}/archive must return a 4xx, got {}",
        resp.status()
    );

    // The stub must still resolve (archive was rejected, not partially applied).
    let get_resp = get(&router, &format!("/api/pages/{id}")).await;
    assert_eq!(
        get_resp.status(),
        StatusCode::OK,
        "the shadow page must still resolve after a rejected archive"
    );
}

#[tokio::test]
async fn delete_of_shadow_id_is_rejected() {
    let (router, _tmp, db) = common::test_app().await;
    let entity_id = db
        .store_entity("HTTP Delete Fence Marker", "person", None, None, None)
        .await
        .unwrap();
    db.confirm_entity(&entity_id, true).await.unwrap();
    let id = find_page_id_by_title(&router, "HTTP Delete Fence Marker").await;

    let resp = delete(&router, &format!("/api/pages/{id}")).await;
    assert!(
        resp.status().is_client_error(),
        "DELETE /api/pages/{{stub}} must return a 4xx, got {}",
        resp.status()
    );

    let get_resp = get(&router, &format!("/api/pages/{id}")).await;
    assert_eq!(
        get_resp.status(),
        StatusCode::OK,
        "the shadow page must still resolve after a rejected delete"
    );
}

#[tokio::test]
async fn write_and_page_map_surfaces_reject_shadow_while_get_stays_200() {
    let (router, _tmp, db) = common::test_app().await;
    db.store_entity("HTTP Write Fence Marker", "person", None, None, None)
        .await
        .unwrap();
    let id = find_page_id_by_title(&router, "HTTP Write Fence Marker").await;

    // Manual update (POST /api/memory/{id}/update-page): fenced get_page →
    // None → 404.
    let update_resp = post_json(
        &router,
        &format!("/api/memory/{id}/update-page"),
        serde_json::json!({ "content": "attempted stub edit" }),
    )
    .await;
    assert_eq!(
        update_resp.status(),
        StatusCode::NOT_FOUND,
        "manual update of a shadow id must 404"
    );

    // Agent refresh (PUT /api/pages/{id}): valid body, fenced get_page → None →
    // 4xx.
    let refresh_resp = put_json(
        &router,
        &format!("/api/pages/{id}"),
        serde_json::json!({
            "content": "attempted stub refresh",
            "source_memory_ids": ["mem_1"],
        }),
    )
    .await;
    assert!(
        refresh_resp.status().is_client_error(),
        "agent refresh of a shadow id must return a 4xx, got {}",
        refresh_resp.status()
    );

    // Page-map node create (POST /api/pages/{id}/map/nodes): ensure_page_is_active
    // → fenced get_page → None → 404.
    let node_resp = post_json(
        &router,
        &format!("/api/pages/{id}/map/nodes"),
        serde_json::json!({ "base_revision": 0 }),
    )
    .await;
    assert_eq!(
        node_resp.status(),
        StatusCode::NOT_FOUND,
        "page-map node create for a shadow id must 404"
    );

    // The read surface stays 200 throughout.
    let get_resp = get(&router, &format!("/api/pages/{id}")).await;
    assert_eq!(
        get_resp.status(),
        StatusCode::OK,
        "GET /api/pages/{{stub}} must stay 200 while the write surfaces reject it"
    );
}
