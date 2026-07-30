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
use wenlan_types::responses::{PinnedMemoriesResponse, SuccessResponse};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

async fn request_bytes(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    marked: bool,
    space: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder().method(method).uri(uri);
    if let Some(space) = space {
        request = request.header("x-wenlan-space", space);
    }
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "pinned-memory-route-contract");
    }
    let response = router
        .clone()
        .oneshot(request.body(Body::empty()).unwrap())
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
    space: Option<&str>,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) = request_bytes(router, method.clone(), uri, false, space).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "{method} {uri} returned a non-contract response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

#[tokio::test]
async fn moved_pinned_memory_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();
    for (source_id, space, last_modified) in [
        ("work-pinned-source", "work", 100),
        ("personal-pinned-source", "personal", 200),
    ] {
        common::insert_memory(
            &db,
            source_id,
            &format!("Pinned memory contract for {space}."),
            "memory",
            Some("pinned-memory-route-contract"),
            None,
            false,
            last_modified,
        )
        .await;
        db.update_memory_space(source_id, space).await.unwrap();
    }

    for source_id in ["work-pinned-source", "personal-pinned-source"] {
        let uri = format!("/api/memory/{source_id}/pin");
        let (status, response): (StatusCode, SuccessResponse) =
            request_typed(&router, Method::POST, &uri, None).await;
        assert_eq!(status, StatusCode::OK);
        assert!(response.ok);
    }

    let (status, work): (StatusCode, PinnedMemoriesResponse) =
        request_typed(&router, Method::GET, "/api/memory/pinned", Some("work")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(work.memories.len(), 1);
    assert_eq!(work.memories[0].source_id, "work-pinned-source");

    let (status, global): (StatusCode, PinnedMemoriesResponse) =
        request_typed(&router, Method::GET, "/api/memory/pinned", None).await;
    assert_eq!(status, StatusCode::OK);
    let mut source_ids: Vec<&str> = global
        .memories
        .iter()
        .map(|memory| memory.source_id.as_str())
        .collect();
    source_ids.sort_unstable();
    assert_eq!(
        source_ids,
        vec!["personal-pinned-source", "work-pinned-source"]
    );

    let (status, response): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/work-pinned-source/unpin",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(response.ok);
    let (status, work): (StatusCode, PinnedMemoriesResponse) =
        request_typed(&router, Method::GET, "/api/memory/pinned", Some("work")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(work.memories.is_empty());

    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::GET,
        "/api/memory/pinned",
        Some("missing-space"),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(error.error, "unknown Space: missing-space");

    for (method, uri) in [
        (Method::GET, "/api/memory/pinned"),
        (Method::POST, "/api/memory/missing/pin"),
        (Method::POST, "/api/memory/missing/unpin"),
    ] {
        let (status, body) = request_bytes(&router, method, uri, true, None).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
        assert!(refusal.error.contains("reader-intent marker"));
    }

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    for (method, uri) in [
        (Method::GET, "/api/memory/pinned"),
        (Method::POST, "/api/memory/missing/pin"),
        (Method::POST, "/api/memory/missing/unpin"),
    ] {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&no_db, method, uri, None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.error, "Database not initialized");
    }
}
