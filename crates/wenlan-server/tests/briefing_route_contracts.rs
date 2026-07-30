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
use wenlan_types::briefing::BriefingResponse;

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

async fn request_bytes(
    router: &common::AppRouter,
    marked: bool,
    space: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder().method(Method::GET).uri("/api/briefing");
    if let Some(space) = space {
        request = request.header("x-wenlan-space", space);
    }
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "briefing-route-contract");
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

async fn request_typed<T>(router: &common::AppRouter, space: Option<&str>) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) = request_bytes(router, false, space).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "GET /api/briefing returned a non-contract response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

#[tokio::test]
async fn moved_briefing_handler_preserves_typed_contract() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();
    let now = chrono::Utc::now().timestamp();
    for (source_id, space, last_modified) in [
        ("work-brief-source", "work", now - 1),
        ("personal-brief-source", "personal", now),
    ] {
        common::insert_memory(
            &db,
            source_id,
            &format!("Briefing contract for {space}."),
            "memory",
            Some("briefing-route-contract"),
            None,
            false,
            last_modified,
        )
        .await;
        db.update_memory_space(source_id, space).await.unwrap();
    }
    db.upsert_briefing_cache("personal-cache-secret", 99)
        .await
        .unwrap();
    let cached_before = db.get_cached_briefing().await.unwrap();

    let (status, briefing): (StatusCode, BriefingResponse) =
        request_typed(&router, Some("work")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(briefing.content.contains("title-work-brief-source"));
    assert!(!briefing.content.contains("title-personal-brief-source"));
    assert!(!briefing.content.contains("personal-cache-secret"));
    assert_eq!(db.get_cached_briefing().await.unwrap(), cached_before);

    let (status, error): (StatusCode, ErrorEnvelope) =
        request_typed(&router, Some("missing-space")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(error.error, "unknown Space: missing-space");

    let (status, body) = request_bytes(&router, true, None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
    assert!(refusal.error.contains("reader-intent marker"));

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(&no_db, None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error.error, "Database not initialized");
}
