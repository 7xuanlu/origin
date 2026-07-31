// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tower::ServiceExt;
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER};
use wenlan_types::page_map::{
    CreateMapEdgeRequest, CreateMapNodeRequest, DeleteMapEdgeRequest, DeleteMapNodeRequest,
    EdgeMutationResponse, NodeMutationResponse, PageMapNodeLayout, PageMapResponse,
    PageMapViewport, PatchMapEdgeRequest, PatchMapNodeRequest, PutPageMapLayoutRequest,
};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

#[derive(Debug, Deserialize)]
struct ResetMapResponse {
    status: String,
}

fn json_body<T: Serialize>(value: &T) -> Body {
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
            .header("x-agent-name", "page-map-route-contract");
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

#[tokio::test]
async fn moved_page_map_routes_preserve_typed_success_error_and_truth_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate_with_page_root().await;
    common::insert_memory(
        &db,
        "map-ref-a",
        "First page-map reference.",
        "memory",
        None,
        None,
        false,
        1,
    )
    .await;
    common::insert_memory(
        &db,
        "map-ref-b",
        "Second page-map reference.",
        "memory",
        None,
        None,
        false,
        2,
    )
    .await;
    let page_id = common::create_page_fixture(
        &db,
        "Page Map Route Contract",
        "A stable body for the page-map route contract.",
        None,
        &["map-ref-a", "map-ref-b"],
        "distilled",
    )
    .await;

    let map_uri = format!("/api/pages/{page_id}/map");
    let (status, empty): (StatusCode, PageMapResponse) =
        request_typed(&router, Method::GET, &map_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(empty.page_id, page_id);
    assert_eq!(empty.revision, 0);
    assert!(empty.nodes.is_empty());
    assert!(empty.edges.is_empty());

    let (status, body) = request_bytes(&router, Method::GET, &map_uri, Body::empty(), true).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "the authoritative None marker shape must refuse reader intent"
    );
    let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
    assert!(refusal.error.contains("reader-intent marker"));

    let create_a = CreateMapNodeRequest {
        base_revision: 0,
        parent_id: None,
        ref_kind: Some("memory".to_string()),
        ref_id: Some("map-ref-a".to_string()),
        label: None,
        rank: 1.0,
    };
    let (status, node_a): (StatusCode, NodeMutationResponse) = request_typed(
        &router,
        Method::POST,
        &format!("{map_uri}/nodes"),
        json_body(&create_a),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(node_a.revision, 2);

    let create_b = CreateMapNodeRequest {
        base_revision: node_a.revision,
        parent_id: None,
        ref_kind: Some("memory".to_string()),
        ref_id: Some("map-ref-b".to_string()),
        label: Some("Second reference".to_string()),
        rank: 2.0,
    };
    let (status, node_b): (StatusCode, NodeMutationResponse) = request_typed(
        &router,
        Method::POST,
        &format!("{map_uri}/nodes"),
        json_body(&create_b),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(node_b.revision, node_a.revision + 1);

    let layout = PutPageMapLayoutRequest {
        base_revision: node_b.revision,
        viewport: Some(PageMapViewport {
            x: 10.0,
            y: 20.0,
            zoom: 1.25,
        }),
        positions: vec![
            PageMapNodeLayout {
                node_id: node_a.node.id.clone(),
                x: 100.0,
                y: 120.0,
                width: 240.0,
                height: 120.0,
                collapsed: false,
            },
            PageMapNodeLayout {
                node_id: node_b.node.id.clone(),
                x: 420.0,
                y: 120.0,
                width: 240.0,
                height: 120.0,
                collapsed: true,
            },
        ],
    };
    let (status, laid_out): (StatusCode, PageMapResponse) = request_typed(
        &router,
        Method::PUT,
        &format!("{map_uri}/layout"),
        json_body(&layout),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(laid_out.revision, node_b.revision + 1);

    let patch_node = PatchMapNodeRequest {
        base_revision: laid_out.revision,
        label: Some(Some("Pinned reference".to_string())),
        pinned: Some(true),
        status: None,
        rank: None,
        parent_id: None,
    };
    let (status, patched_node): (StatusCode, NodeMutationResponse) = request_typed(
        &router,
        Method::PATCH,
        &format!("{map_uri}/nodes/{}", node_a.node.id),
        json_body(&patch_node),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(patched_node.revision, laid_out.revision + 1);
    assert_eq!(patched_node.node.label.as_deref(), Some("Pinned reference"));
    assert!(patched_node.node.pinned);

    let create_edge = CreateMapEdgeRequest {
        base_revision: patched_node.revision,
        from_node: node_a.node.id.clone(),
        to_node: node_b.node.id.clone(),
        kind: "link".to_string(),
        label: Some("route contract edge".to_string()),
    };
    let (status, edge): (StatusCode, EdgeMutationResponse) = request_typed(
        &router,
        Method::POST,
        &format!("{map_uri}/edges"),
        json_body(&create_edge),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(edge.revision, patched_node.revision + 1);

    let patch_edge = PatchMapEdgeRequest {
        base_revision: edge.revision,
        label: Some(Some("patched edge".to_string())),
        status: None,
    };
    let (status, patched_edge): (StatusCode, EdgeMutationResponse) = request_typed(
        &router,
        Method::PATCH,
        &format!("{map_uri}/edges/{}", edge.edge.id),
        json_body(&patch_edge),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(patched_edge.revision, edge.revision + 1);
    assert_eq!(patched_edge.edge.label.as_deref(), Some("patched edge"));

    let delete_edge = DeleteMapEdgeRequest {
        base_revision: patched_edge.revision,
    };
    let (status, deleted_edge): (StatusCode, EdgeMutationResponse) = request_typed(
        &router,
        Method::DELETE,
        &format!("{map_uri}/edges/{}", edge.edge.id),
        json_body(&delete_edge),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted_edge.revision, patched_edge.revision + 1);
    assert_eq!(deleted_edge.edge.status, "dismissed");

    let delete_node = DeleteMapNodeRequest {
        base_revision: deleted_edge.revision,
    };
    let (status, deleted_node): (StatusCode, NodeMutationResponse) = request_typed(
        &router,
        Method::DELETE,
        &format!("{map_uri}/nodes/{}", node_b.node.id),
        json_body(&delete_node),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(deleted_node.revision, deleted_edge.revision + 1);
    assert_eq!(deleted_node.node.status, "dismissed");

    let (status, reset): (StatusCode, ResetMapResponse) =
        request_typed(&router, Method::DELETE, &map_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reset.status, "reset");

    let (status, improved): (StatusCode, PageMapResponse) = request_typed(
        &router,
        Method::POST,
        &format!("{map_uri}/improve"),
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(improved.page_id, page_id);
    assert_eq!(improved.revision, 3);
    assert_eq!(improved.nodes.len(), 3);

    let missing_map_uri = "/api/pages/missing-page/map";
    let error_cases = vec![
        (Method::GET, missing_map_uri.to_string(), Vec::new()),
        (Method::DELETE, missing_map_uri.to_string(), Vec::new()),
        (
            Method::PUT,
            format!("{missing_map_uri}/layout"),
            serde_json::to_vec(&PutPageMapLayoutRequest {
                base_revision: 0,
                viewport: None,
                positions: vec![],
            })
            .unwrap(),
        ),
        (
            Method::POST,
            format!("{missing_map_uri}/nodes"),
            serde_json::to_vec(&create_a).unwrap(),
        ),
        (
            Method::PATCH,
            format!("{missing_map_uri}/nodes/missing-node"),
            serde_json::to_vec(&PatchMapNodeRequest {
                base_revision: 0,
                label: None,
                pinned: None,
                status: None,
                rank: None,
                parent_id: None,
            })
            .unwrap(),
        ),
        (
            Method::DELETE,
            format!("{missing_map_uri}/nodes/missing-node"),
            serde_json::to_vec(&DeleteMapNodeRequest { base_revision: 0 }).unwrap(),
        ),
        (
            Method::POST,
            format!("{missing_map_uri}/edges"),
            serde_json::to_vec(&CreateMapEdgeRequest {
                base_revision: 0,
                from_node: "missing-a".to_string(),
                to_node: "missing-b".to_string(),
                kind: "link".to_string(),
                label: None,
            })
            .unwrap(),
        ),
        (
            Method::PATCH,
            format!("{missing_map_uri}/edges/missing-edge"),
            serde_json::to_vec(&PatchMapEdgeRequest {
                base_revision: 0,
                label: None,
                status: None,
            })
            .unwrap(),
        ),
        (
            Method::DELETE,
            format!("{missing_map_uri}/edges/missing-edge"),
            serde_json::to_vec(&DeleteMapEdgeRequest { base_revision: 0 }).unwrap(),
        ),
        (
            Method::POST,
            format!("{missing_map_uri}/improve"),
            Vec::new(),
        ),
    ];
    for (method, uri, body) in error_cases {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&router, method.clone(), &uri, Body::from(body)).await;
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "{method} {uri} did not preserve its deterministic missing-page error"
        );
        assert!(error.error.contains("missing-page"));
    }
}
