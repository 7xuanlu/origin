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
use wenlan_types::narrative::NarrativeResponse;

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

async fn request_bytes(
    router: &common::AppRouter,
    method: Method,
    marked: bool,
    space: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let uri = match method {
        Method::GET => "/api/profile/narrative",
        Method::POST => "/api/profile/narrative/regenerate",
        _ => unreachable!("profile narrative contract only exercises GET and POST"),
    };
    let mut request = Request::builder().method(method).uri(uri);
    if let Some(space) = space {
        request = request.header("x-wenlan-space", space);
    }
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "profile-narrative-route-contract");
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
    space: Option<&str>,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) = request_bytes(router, method.clone(), false, space).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "{method} profile narrative returned a non-contract response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

#[tokio::test]
async fn moved_profile_narrative_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.upsert_narrative_cache("cached profile narrative", 7)
        .await
        .unwrap();

    let (status, cached): (StatusCode, NarrativeResponse) =
        request_typed(&router, Method::GET, Some("missing-space")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cached.content, "cached profile narrative");
    assert_eq!(cached.memory_count, 7);
    assert!(!cached.is_stale);

    let (status, regenerated): (StatusCode, NarrativeResponse) =
        request_typed(&router, Method::POST, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(regenerated.content, "");
    assert_eq!(regenerated.memory_count, 0);
    assert!(!regenerated.is_stale);
    assert_eq!(
        db.get_cached_narrative().await.unwrap().unwrap().0,
        "cached profile narrative",
        "empty regeneration must bypass, not return or replace, the seeded cache"
    );

    for method in [Method::GET, Method::POST] {
        let (status, body) = request_bytes(&router, method, true, None).await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
        assert!(refusal.error.contains("reader-intent marker"));
    }

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    for method in [Method::GET, Method::POST] {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&no_db, method, None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.error, "Database not initialized");
    }
}
