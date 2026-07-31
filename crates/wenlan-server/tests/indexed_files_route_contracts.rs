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
use wenlan_types::requests::{
    BulkDeleteItem, BulkDeleteRequest, DeleteByTimeRangeRequest, UpdateChunkRequest,
};
use wenlan_types::responses::{
    DeleteCountResponse, IndexedFilesResponse, MemoryDetail, SuccessResponse,
};

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
    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "indexed-files-route-contract");
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

async fn assert_marked_refusal(router: &common::AppRouter, uri: &str) {
    let (status, body) = request_bytes(router, Method::GET, uri, Body::empty(), true).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "GET {uri} must retain its None marker shape"
    );
    let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
    assert!(refusal.error.contains("reader-intent marker"));
}

#[tokio::test]
async fn moved_indexed_file_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    common::insert_memory(
        &db,
        "indexed-route-source",
        "The original indexed route content.",
        "memory",
        Some("indexed-files-route-contract"),
        None,
        false,
        100,
    )
    .await;

    assert_marked_refusal(&router, "/api/indexed-files").await;
    let (status, indexed): (StatusCode, IndexedFilesResponse) =
        request_typed(&router, Method::GET, "/api/indexed-files", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(indexed
        .files
        .iter()
        .any(|file| file.source_id == "indexed-route-source"));

    assert_marked_refusal(&router, "/api/chunks/indexed-route-source").await;
    let (status, chunks): (StatusCode, Vec<MemoryDetail>) = request_typed(
        &router,
        Method::GET,
        "/api/chunks/indexed-route-source",
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].source_id, "indexed-route-source");

    let (status, error): (StatusCode, ErrorEnvelope) =
        request_typed(&router, Method::GET, "/api/chunks/missing", Body::empty()).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error.error, "memory not found");

    let update_request = UpdateChunkRequest {
        content: "The updated indexed route content.".to_string(),
    };
    let (status, updated): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::PUT,
        "/api/chunks/indexed-route-source/update",
        json_body(&update_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(updated.ok);

    let (status, chunks): (StatusCode, Vec<MemoryDetail>) = request_typed(
        &router,
        Method::GET,
        "/api/chunks/indexed-route-source",
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(chunks[0].content, update_request.content);

    let time_range_request = DeleteByTimeRangeRequest {
        start: i64::MAX - 1,
        end: i64::MAX,
    };
    let (status, deleted): (StatusCode, DeleteCountResponse) = request_typed(
        &router,
        Method::DELETE,
        "/api/chunks/time-range",
        json_body(&time_range_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted.deleted, 0);

    common::insert_memory(
        &db,
        "bulk-delete-route-source",
        "A memory deleted through the indexed route contract.",
        "memory",
        Some("indexed-files-route-contract"),
        None,
        false,
        200,
    )
    .await;
    let bulk_request = BulkDeleteRequest {
        items: vec![BulkDeleteItem {
            source: "memory".to_string(),
            source_id: "bulk-delete-route-source".to_string(),
        }],
    };
    let (status, deleted): (StatusCode, DeleteCountResponse) = request_typed(
        &router,
        Method::POST,
        "/api/chunks/delete-bulk",
        json_body(&bulk_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted.deleted, 1);

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    let no_db_cases = vec![
        (Method::GET, "/api/indexed-files", Vec::new()),
        (Method::GET, "/api/chunks/missing", Vec::new()),
        (
            Method::PUT,
            "/api/chunks/missing/update",
            serde_json::to_vec(&update_request).unwrap(),
        ),
        (
            Method::DELETE,
            "/api/chunks/time-range",
            serde_json::to_vec(&time_range_request).unwrap(),
        ),
        (
            Method::POST,
            "/api/chunks/delete-bulk",
            serde_json::to_vec(&bulk_request).unwrap(),
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
