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
use wenlan_types::responses::{MemoryDetailResponse, PinnedMemoriesResponse};

#[derive(Debug, Deserialize)]
struct CaptureStatsResponse {
    total_chunks: u64,
}

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
            .header("x-agent-name", "memory-detail-route-contract");
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

async fn assert_marked_refusal(router: &common::AppRouter, uri: &str) {
    let (status, body) = request_bytes(router, Method::GET, uri, true, None).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "GET {uri} must retain its None marker shape"
    );
    let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
    assert!(refusal.error.contains("reader-intent marker"));
}

#[tokio::test]
async fn moved_memory_detail_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();
    for (source_id, content, space, last_modified) in [
        (
            "work-detail-source",
            "A work memory used by the detail route contract.",
            "work",
            100,
        ),
        (
            "personal-detail-source",
            "A personal memory hidden from the work scope.",
            "personal",
            200,
        ),
    ] {
        common::insert_memory(
            &db,
            source_id,
            content,
            "memory",
            Some("memory-detail-route-contract"),
            None,
            false,
            last_modified,
        )
        .await;
        db.update_memory_space(source_id, space).await.unwrap();
    }

    for uri in [
        "/api/capture-stats",
        "/api/memory/work-detail-source/detail",
        "/api/memory/by-ids?ids=work-detail-source",
    ] {
        assert_marked_refusal(&router, uri).await;
    }

    let (status, capture): (StatusCode, CaptureStatsResponse) =
        request_typed(&router, Method::GET, "/api/capture-stats", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(capture.total_chunks, 2);
    let (status, selected_capture): (StatusCode, CaptureStatsResponse) = request_typed(
        &router,
        Method::GET,
        "/api/capture-stats",
        Some("missing-space"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(selected_capture.total_chunks, capture.total_chunks);

    let (status, detail): (StatusCode, MemoryDetailResponse) = request_typed(
        &router,
        Method::GET,
        "/api/memory/work-detail-source/detail",
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        detail
            .memory
            .as_ref()
            .map(|memory| memory.source_id.as_str()),
        Some("work-detail-source")
    );

    let (status, hidden): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::GET,
        "/api/memory/personal-detail-source/detail",
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(hidden.error, "memory not found");

    let (status, scoped): (StatusCode, PinnedMemoriesResponse) = request_typed(
        &router,
        Method::GET,
        "/api/memory/by-ids?ids=personal-detail-source,work-detail-source,missing",
        Some("work"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(scoped.memories.len(), 1);
    assert_eq!(scoped.memories[0].source_id, "work-detail-source");

    let (status, global): (StatusCode, PinnedMemoriesResponse) = request_typed(
        &router,
        Method::GET,
        "/api/memory/by-ids?ids=personal-detail-source,work-detail-source,missing",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let source_ids: Vec<&str> = global
        .memories
        .iter()
        .map(|memory| memory.source_id.as_str())
        .collect();
    assert_eq!(
        source_ids,
        vec!["personal-detail-source", "work-detail-source"]
    );

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    for uri in [
        "/api/capture-stats",
        "/api/memory/missing/detail",
        "/api/memory/by-ids?ids=missing",
    ] {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&no_db, Method::GET, uri, None).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "GET {uri} did not preserve its deterministic no-DB error"
        );
        assert_eq!(error.error, "Database not initialized");
    }
}
