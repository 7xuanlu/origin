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
    CreateSpaceRequest, ReorderSpaceRequest, SetDefaultSpaceRequest, SetDocumentSpaceRequest,
    UpdateSpaceRequest,
};
use wenlan_types::responses::{DefaultSpaceResponse, SuccessResponse};
use wenlan_types::Space;

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

#[derive(Debug, Deserialize)]
struct DeletedSpaceResponse {
    deleted: String,
}

#[derive(Debug, Deserialize)]
struct MovedSpaceResponse {
    affected: usize,
}

#[derive(Debug, Deserialize)]
struct StarredSpaceResponse {
    starred: bool,
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
            .header("x-agent-name", "spaces-route-contract");
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
async fn moved_space_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;

    let (status, spaces): (StatusCode, Vec<Space>) =
        request_typed(&router, Method::GET, "/api/spaces", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(spaces.is_empty());

    let alpha_request = CreateSpaceRequest {
        name: "space-route-alpha".to_string(),
        description: Some("Space route contract source.".to_string()),
    };
    let (status, alpha): (StatusCode, Space) = request_typed(
        &router,
        Method::POST,
        "/api/spaces",
        json_body(&alpha_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    for request in [
        CreateSpaceRequest {
            name: "space-route-beta".to_string(),
            description: Some("Space route contract target.".to_string()),
        },
        CreateSpaceRequest {
            name: "space-route-disposable".to_string(),
            description: None,
        },
    ] {
        let (status, _): (StatusCode, Space) =
            request_typed(&router, Method::POST, "/api/spaces", json_body(&request)).await;
        assert_eq!(status, StatusCode::OK);
    }

    let update_request = UpdateSpaceRequest {
        new_name: Some("space-route-work".to_string()),
        description: Some("Updated by the typed Space route contract.".to_string()),
    };
    let (status, updated): (StatusCode, Space) = request_typed(
        &router,
        Method::PUT,
        "/api/spaces/space-route-alpha",
        json_body(&update_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(updated.name, "space-route-work");

    let (status, default): (StatusCode, DefaultSpaceResponse) = request_typed(
        &router,
        Method::PUT,
        "/api/spaces/default",
        json_body(&SetDefaultSpaceRequest {
            space_id: alpha.id.clone(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        default.space.as_ref().map(|space| space.name.as_str()),
        Some("space-route-work")
    );

    let (status, default): (StatusCode, DefaultSpaceResponse) =
        request_typed(&router, Method::GET, "/api/spaces/default", Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(default.space.is_some_and(|space| space.is_default));

    let (status, body) = request_bytes(
        &router,
        Method::DELETE,
        "/api/spaces/default",
        Body::empty(),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(body.is_empty());

    let (status, pinned): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::POST,
        "/api/spaces/space-route-work/pin",
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(pinned.ok);

    let (status, confirmed): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::POST,
        "/api/spaces/space-route-work/confirm",
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(confirmed.ok);

    let (status, reordered): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::POST,
        "/api/spaces/reorder",
        json_body(&ReorderSpaceRequest {
            name: "space-route-work".to_string(),
            new_order: 7,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(reordered.ok);

    assert_marked_refusal(&router, Method::POST, "/api/spaces/space-route-work/star").await;
    let (status, starred): (StatusCode, StarredSpaceResponse) = request_typed(
        &router,
        Method::POST,
        "/api/spaces/space-route-work/star",
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!starred.starred);

    common::insert_memory(
        &db,
        "space-route-document",
        "A document reassigned through the typed Space route contract.",
        "memory",
        Some("spaces-route-contract"),
        None,
        false,
        1,
    )
    .await;
    let (status, assigned): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::POST,
        "/api/documents/space-route-document/space",
        json_body(&SetDocumentSpaceRequest {
            space_name: "space-route-work".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(assigned.ok);
    assert_eq!(
        db.get_memory_space("space-route-document").await.unwrap(),
        Some("space-route-work".to_string())
    );

    assert_marked_refusal(
        &router,
        Method::POST,
        "/api/spaces/space-route-work/move-to/space-route-beta",
    )
    .await;
    let (status, moved): (StatusCode, MovedSpaceResponse) = request_typed(
        &router,
        Method::POST,
        "/api/spaces/space-route-work/move-to/space-route-beta",
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(moved.affected, 1);
    assert_eq!(
        db.get_memory_space("space-route-document").await.unwrap(),
        Some("space-route-beta".to_string())
    );

    assert_marked_refusal(
        &router,
        Method::DELETE,
        "/api/spaces/space-route-disposable",
    )
    .await;
    let (status, deleted): (StatusCode, DeletedSpaceResponse) = request_typed(
        &router,
        Method::DELETE,
        "/api/spaces/space-route-disposable",
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted.deleted, "space-route-disposable");

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    let no_db_cases = vec![
        (Method::GET, "/api/spaces", Vec::new()),
        (
            Method::POST,
            "/api/spaces",
            serde_json::to_vec(&alpha_request).unwrap(),
        ),
        (Method::GET, "/api/spaces/default", Vec::new()),
        (
            Method::PUT,
            "/api/spaces/default",
            serde_json::to_vec(&SetDefaultSpaceRequest { space_id: alpha.id }).unwrap(),
        ),
        (Method::DELETE, "/api/spaces/default", Vec::new()),
        (
            Method::PUT,
            "/api/spaces/missing",
            serde_json::to_vec(&update_request).unwrap(),
        ),
        (Method::DELETE, "/api/spaces/missing", Vec::new()),
        (
            Method::POST,
            "/api/spaces/missing/move-to/also-missing",
            Vec::new(),
        ),
        (Method::POST, "/api/spaces/missing/pin", Vec::new()),
        (Method::POST, "/api/spaces/missing/confirm", Vec::new()),
        (
            Method::POST,
            "/api/spaces/reorder",
            serde_json::to_vec(&ReorderSpaceRequest {
                name: "missing".to_string(),
                new_order: 0,
            })
            .unwrap(),
        ),
        (Method::POST, "/api/spaces/missing/star", Vec::new()),
        (
            Method::POST,
            "/api/documents/missing/space",
            serde_json::to_vec(&SetDocumentSpaceRequest {
                space_name: "missing".to_string(),
            })
            .unwrap(),
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
