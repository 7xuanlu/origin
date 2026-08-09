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
use wenlan_types::responses::{DecisionDomainsResponse, DecisionsResponse};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

async fn request_bytes(
    router: &common::AppRouter,
    uri: &str,
    marked: bool,
    space: Option<&str>,
) -> (StatusCode, Vec<u8>) {
    let mut request = Request::builder().method(Method::GET).uri(uri);
    if let Some(space) = space {
        request = request.header("x-wenlan-space", space);
    }
    if marked {
        request = request
            .header(INTENT_HEADER, "explicit")
            .header(CONTRACT_HEADER, "1")
            .header("x-agent-name", "decisions-route-contract");
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
    uri: &str,
    space: Option<&str>,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let (status, bytes) = request_bytes(router, uri, false, space).await;
    let decoded = serde_json::from_slice::<T>(&bytes).unwrap_or_else(|error| {
        panic!(
            "GET {uri} returned a non-contract response ({error}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, decoded)
}

async fn assert_marked_refusal(router: &common::AppRouter, uri: &str) {
    let (status, body) = request_bytes(router, uri, true, None).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
    assert!(refusal.error.contains("reader-intent marker"));
}

#[tokio::test]
async fn moved_decision_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("personal", None, false).await.unwrap();
    for (source_id, space, last_modified) in [
        ("work-decision-source", "work", 100),
        ("personal-decision-source", "personal", 200),
    ] {
        common::insert_memory(
            &db,
            source_id,
            &format!("Decision contract for {space}."),
            "memory",
            Some("decisions-route-contract"),
            None,
            false,
            last_modified,
        )
        .await;
        db.update_memory_type(source_id, "decision").await.unwrap();
        db.update_memory_space(source_id, space).await.unwrap();
    }

    for uri in ["/api/decisions", "/api/decisions/domains"] {
        assert_marked_refusal(&router, uri).await;
    }

    let (status, decisions): (StatusCode, DecisionsResponse) = request_typed(
        &router,
        "/api/decisions?space=work&limit=10",
        Some("personal"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(decisions.decisions.len(), 1);
    assert_eq!(decisions.decisions[0].source_id, "work-decision-source");

    let (status, domains): (StatusCode, DecisionDomainsResponse) =
        request_typed(&router, "/api/decisions/domains", Some("missing-space")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(domains.domains.iter().any(|space| space == "work"));
    assert!(domains.domains.iter().any(|space| space == "personal"));

    let (status, error): (StatusCode, ErrorEnvelope) =
        request_typed(&router, "/api/decisions?space=missing-space", None).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(error.error, "unknown Space: missing-space");

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    for uri in ["/api/decisions", "/api/decisions/domains"] {
        let (status, error): (StatusCode, ErrorEnvelope) = request_typed(&no_db, uri, None).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.error, "Database not initialized");
    }
}
