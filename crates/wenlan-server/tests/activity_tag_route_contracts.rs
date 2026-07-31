// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER};
use wenlan_server::{router::build_router, state::ServerState};
use wenlan_types::requests::SetDocumentTagsRequest;
use wenlan_types::responses::{ActivityResponse, SuccessResponse, TagsResponse};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

fn json_body<T: serde::Serialize>(value: &T) -> Body {
    Body::from(serde_json::to_vec(value).unwrap())
}

async fn request_bytes(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    body: Body,
    marked: bool,
) -> (StatusCode, Vec<u8>) {
    request_bytes_in_space(router, method, uri, body, marked, None).await
}

async fn request_bytes_in_space(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    body: Body,
    marked: bool,
    space: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(space) = space {
        request = request.header("x-wenlan-space", space);
    }
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "activity-tag-route-contract");
    }
    let response = router
        .clone()
        .oneshot(request.body(body).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, bytes)
}

async fn request_typed<T>(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    body: Body,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) = request_bytes(router, method.clone(), uri, body, false).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "{method} {uri} returned a non-contract response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

async fn request_typed_in_space<T>(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    body: Body,
    space: &str,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) =
        request_bytes_in_space(router, method.clone(), uri, body, false, Some(space)).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "{method} {uri} in {space} returned a non-contract response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

async fn assert_marked_refusal(router: &common::AppRouter, method: Method, uri: &str) {
    let (status, body) = request_bytes(router, method.clone(), uri, Body::empty(), true).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "{method} {uri} must retain its None marker shape"
    );
    let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
    assert!(refusal.error.contains("reader-intent marker"));
}

#[tokio::test]
async fn moved_activity_and_tag_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();
    common::insert_memory(
        &db,
        "activity-tag-source",
        "Rust route contracts preserve deterministic tag suggestions.",
        "memory",
        Some("activity-tag-route-contract"),
        None,
        false,
        100,
    )
    .await;
    db.update_memory_space("activity-tag-source", "work")
        .await
        .unwrap();
    common::insert_memory(
        &db,
        "personal-activity-tag-source",
        "A scoped route must not disclose this personal memory.",
        "memory",
        Some("activity-tag-route-contract"),
        None,
        false,
        200,
    )
    .await;
    db.update_memory_space("personal-activity-tag-source", "personal")
        .await
        .unwrap();
    db.log_agent_activity(
        "codex",
        "read",
        &["activity-tag-source".to_string()],
        Some("route contract"),
        "activity tag route detail",
    )
    .await
    .unwrap();
    db.log_agent_activity(
        "codex",
        "read",
        &["personal-activity-tag-source".to_string()],
        Some("personal route contract"),
        "personal activity tag route detail",
    )
    .await
    .unwrap();
    db.set_document_tags(
        "memory",
        "personal-activity-tag-source",
        vec!["personal-only".to_string()],
    )
    .await
    .unwrap();

    let tag_request = SetDocumentTagsRequest {
        source: Some("memory".to_string()),
        tags: vec!["rust".to_string()],
    };
    let routes = [
        (Method::GET, "/api/activities?limit=10"),
        (Method::GET, "/api/tags"),
        (Method::DELETE, "/api/tags/rust"),
        (
            Method::GET,
            "/api/suggest-tags?source=memory&source_id=activity-tag-source&activity_app=Codex",
        ),
        (Method::PUT, "/api/documents/activity-tag-source/tags"),
    ];
    for (method, uri) in routes {
        assert_marked_refusal(&router, method, uri).await;
    }

    let (status, activities): (StatusCode, ActivityResponse) = request_typed_in_space(
        &router,
        Method::GET,
        "/api/activities?limit=10",
        Body::empty(),
        "work",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(activities.activities.iter().any(|activity| {
        activity.action == "read" && activity.detail.as_deref() == Some("activity tag route detail")
    }));
    assert!(!activities.activities.iter().any(|activity| {
        activity.detail.as_deref() == Some("personal activity tag route detail")
    }));

    let (status, tags): (StatusCode, TagsResponse) = request_typed(
        &router,
        Method::PUT,
        "/api/documents/activity-tag-source/tags",
        json_body(&tag_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(tags.tags, vec!["rust"]);

    let (status, tags): (StatusCode, TagsResponse) =
        request_typed_in_space(&router, Method::GET, "/api/tags", Body::empty(), "work").await;
    assert_eq!(status, StatusCode::OK);
    assert!(tags.tags.iter().any(|tag| tag == "rust"));
    assert!(!tags.tags.iter().any(|tag| tag == "personal-only"));
    assert_eq!(
        tags.document_tags.get("memory::activity-tag-source"),
        Some(&vec!["rust".to_string()])
    );
    assert!(!tags
        .document_tags
        .contains_key("memory::personal-activity-tag-source"));

    let (status, suggestions): (StatusCode, TagsResponse) = request_typed_in_space(
        &router,
        Method::GET,
        "/api/suggest-tags?source=memory&source_id=activity-tag-source&activity_app=Codex",
        Body::empty(),
        "work",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(suggestions.tags.iter().any(|tag| tag == "codex"));
    assert!(!suggestions.tags.iter().any(|tag| tag == "rust"));

    let (status, hidden): (StatusCode, ErrorEnvelope) = request_typed_in_space(
        &router,
        Method::GET,
        "/api/suggest-tags?source=memory&source_id=personal-activity-tag-source",
        Body::empty(),
        "work",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(hidden.error, "memory not found");

    let (status, deleted): (StatusCode, SuccessResponse) =
        request_typed(&router, Method::DELETE, "/api/tags/rust", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(deleted.ok);

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    let no_db_cases = [
        (Method::GET, "/api/activities?limit=10", Vec::new()),
        (Method::GET, "/api/tags", Vec::new()),
        (Method::DELETE, "/api/tags/rust", Vec::new()),
        (
            Method::GET,
            "/api/suggest-tags?source=memory&source_id=missing",
            Vec::new(),
        ),
        (
            Method::PUT,
            "/api/documents/missing/tags",
            serde_json::to_vec(&tag_request).unwrap(),
        ),
    ];
    for (method, uri, body) in no_db_cases {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&no_db, method.clone(), uri, Body::from(body)).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{method} {uri} did not preserve its deterministic no-DB error"
        );
        assert_eq!(error.error, "Database not initialized");
    }
}
