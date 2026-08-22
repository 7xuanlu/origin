// SPDX-License-Identifier: Apache-2.0
mod common;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceExt;
use wenlan_core::truth_contract::{CONTRACT_HEADER, INTENT_HEADER};
use wenlan_server::{router::build_router, state::ServerState};
use wenlan_types::entities::EntityDetail;
use wenlan_types::requests::{
    AddEntityObservationRequest, AddObservationRequest, ConfirmEntityRequest,
    ConfirmObservationRequest, CreateEntityRequest, CreateRelationRequest, LinkEntityRequest,
    ListEntitiesRequest, UpdateObservationRequest,
};
use wenlan_types::responses::{
    AddObservationResponse, CreateEntityResponse, CreateRelationResponse, ListEntitiesResponse,
    SearchEntitiesResponse, SuccessResponse,
};
use wenlan_types::{WriteOutcome, WriteSpaceSource, WriteSpaceTarget};

#[derive(Debug, Deserialize)]
struct ErrorEnvelope {
    error: String,
}

#[derive(Debug, Deserialize)]
struct LinkEntityResponse {
    linked: bool,
}

#[derive(Debug, Serialize)]
struct SearchEntitiesRequestMirror {
    query: String,
    limit: usize,
    space: Option<String>,
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
            .header("x-agent-name", "entity-graph-route-contract");
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
async fn moved_entity_graph_handlers_preserve_typed_contracts() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    common::insert_memory(
        &db,
        "entity-link-source",
        "A memory linked by the entity route contract.",
        "memory",
        Some("entity-graph-route-contract"),
        None,
        false,
        1,
    )
    .await;

    let alpha_request = CreateEntityRequest {
        name: "Entity Route Alpha".to_string(),
        entity_type: "project".to_string(),
        space: Default::default(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.9),
    };
    let (status, alpha): (StatusCode, CreateEntityResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities",
        json_body(&alpha_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let beta_request = CreateEntityRequest {
        name: "Entity Route Beta".to_string(),
        entity_type: "person".to_string(),
        space: Default::default(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.8),
    };
    let (status, beta): (StatusCode, CreateEntityResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities",
        json_body(&beta_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let relation_request = CreateRelationRequest {
        from_entity: alpha.id.clone(),
        to_entity: beta.id.clone(),
        relation_type: "depends_on".to_string(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.8),
        explanation: Some("Typed route evidence.".to_string()),
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let (status, relation): (StatusCode, CreateRelationResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/relations",
        json_body(&relation_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!relation.id.is_empty());

    let observation_request = AddObservationRequest {
        entity_id: alpha.id.clone(),
        content: "Alpha is covered by a typed route contract.".to_string(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.9),
    };
    let (status, observation): (StatusCode, AddObservationResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/observations",
        json_body(&observation_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!observation.id.is_empty());

    let link_request = LinkEntityRequest {
        source_id: "entity-link-source".to_string(),
        entity_id: alpha.id.clone(),
    };
    let (status, body) = request_bytes(
        &router,
        Method::POST,
        "/api/memory/link-entity",
        json_body(&link_request),
        true,
    )
    .await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "the opaque link response must retain its None marker shape"
    );
    let refusal: ErrorEnvelope = serde_json::from_slice(&body).unwrap();
    assert!(refusal.error.contains("reader-intent marker"));

    let (status, linked): (StatusCode, LinkEntityResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/link-entity",
        json_body(&link_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(linked.linked);

    // Unknown ids must not report a link that touched zero rows.
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/memory/link-entity",
        json_body(&LinkEntityRequest {
            source_id: "no-such-memory".to_string(),
            entity_id: "no-such-entity".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(error.error.contains("no-such-memory"), "{}", error.error);

    let list_request = ListEntitiesRequest {
        entity_type: None,
        space: None,
    };
    let (status, listed): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/list",
        json_body(&list_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(listed.entities.iter().any(|entity| entity.id == alpha.id));
    assert!(listed.entities.iter().any(|entity| entity.id == beta.id));

    let entity_uri = format!("/api/memory/entities/{}", alpha.id);
    let (status, detail): (StatusCode, EntityDetail) =
        request_typed(&router, Method::GET, &entity_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detail.entity.id, alpha.id);

    let search_request = SearchEntitiesRequestMirror {
        query: "Entity Route Alpha".to_string(),
        limit: 10,
        space: None,
    };
    let (status, search): (StatusCode, SearchEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/search",
        json_body(&search_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(search
        .results
        .iter()
        .any(|result| result.entity.id == alpha.id));

    let confirm_entity_uri = format!("/api/memory/entities/{}/confirm", alpha.id);
    let (status, confirmed): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::PUT,
        &confirm_entity_uri,
        json_body(&ConfirmEntityRequest { confirmed: true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(confirmed.ok);

    let add_observation_uri = format!("/api/memory/entities/{}/observations", alpha.id);
    let add_observation_request = AddEntityObservationRequest {
        content: "A second typed observation.".to_string(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.75),
    };
    let (status, added): (StatusCode, AddObservationResponse) = request_typed(
        &router,
        Method::POST,
        &add_observation_uri,
        json_body(&add_observation_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // The entity-scoped route shares `POST /api/memory/observations`'s
    // validity contract: unknown entity, short content and out-of-range
    // confidence are all 422, never a 200 with an orphan row.
    let invalid_observation_cases = [
        (
            "/api/memory/entities/missing/observations".to_string(),
            AddEntityObservationRequest {
                content: "Observation for an entity that does not exist.".to_string(),
                source_agent: None,
                confidence: Some(0.5),
            },
            "does not exist",
        ),
        (
            add_observation_uri.clone(),
            AddEntityObservationRequest {
                content: "x".to_string(),
                source_agent: None,
                confidence: Some(0.5),
            },
            "at least 5 characters",
        ),
        (
            add_observation_uri.clone(),
            AddEntityObservationRequest {
                content: "Confidence far out of range.".to_string(),
                source_agent: None,
                confidence: Some(9.0),
            },
            "out of range",
        ),
    ];
    for (uri, request, expected) in invalid_observation_cases {
        let (status, error): (StatusCode, ErrorEnvelope) =
            request_typed(&router, Method::POST, &uri, json_body(&request)).await;
        assert_eq!(
            status,
            StatusCode::UNPROCESSABLE_ENTITY,
            "{uri}: {}",
            error.error
        );
        assert!(error.error.contains(expected), "{uri}: {}", error.error);
    }

    let observation_uri = format!("/api/memory/observations/{}", added.id);
    let (status, updated): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::PUT,
        &observation_uri,
        json_body(&UpdateObservationRequest {
            content: "The updated typed observation.".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(updated.ok);

    let confirm_observation_uri = format!("{observation_uri}/confirm");
    let (status, confirmed): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::PUT,
        &confirm_observation_uri,
        json_body(&ConfirmObservationRequest { confirmed: true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(confirmed.ok);

    let (status, deleted): (StatusCode, SuccessResponse) =
        request_typed(&router, Method::DELETE, &observation_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(deleted.ok);

    let delete_entity_uri = format!("/api/memory/entities/{}/delete", beta.id);
    let (status, deleted): (StatusCode, SuccessResponse) =
        request_typed(&router, Method::DELETE, &delete_entity_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(deleted.ok);

    let no_db = build_router(Arc::new(RwLock::new(ServerState::default())));
    let no_db_cases = vec![
        (
            Method::POST,
            "/api/memory/entities",
            serde_json::to_vec(&alpha_request).unwrap(),
        ),
        (
            Method::POST,
            "/api/memory/relations",
            serde_json::to_vec(&relation_request).unwrap(),
        ),
        (
            Method::POST,
            "/api/memory/observations",
            serde_json::to_vec(&observation_request).unwrap(),
        ),
        (
            Method::POST,
            "/api/memory/link-entity",
            serde_json::to_vec(&link_request).unwrap(),
        ),
        (
            Method::POST,
            "/api/memory/entities/list",
            serde_json::to_vec(&list_request).unwrap(),
        ),
        (
            Method::POST,
            "/api/memory/entities/search",
            serde_json::to_vec(&search_request).unwrap(),
        ),
        (Method::GET, "/api/memory/entities/missing", Vec::new()),
        (
            Method::PUT,
            "/api/memory/entities/missing/confirm",
            serde_json::to_vec(&ConfirmEntityRequest { confirmed: true }).unwrap(),
        ),
        (
            Method::DELETE,
            "/api/memory/entities/missing/delete",
            Vec::new(),
        ),
        (
            Method::POST,
            "/api/memory/entities/missing/observations",
            serde_json::to_vec(&add_observation_request).unwrap(),
        ),
        (
            Method::PUT,
            "/api/memory/observations/missing",
            serde_json::to_vec(&UpdateObservationRequest {
                content: "missing".to_string(),
            })
            .unwrap(),
        ),
        (
            Method::DELETE,
            "/api/memory/observations/missing",
            Vec::new(),
        ),
        (
            Method::PUT,
            "/api/memory/observations/missing/confirm",
            serde_json::to_vec(&ConfirmObservationRequest { confirmed: true }).unwrap(),
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

/// G6 Stage 1.5b Part 2 pin test: the `entities.space` space-sentinel fold
/// is an internal storage detail -- `POST /api/memory/entities` for an
/// unfiled entity must still return `space: null` and
/// `space_source: "uncategorized"` on the wire, exactly as it did when the
/// column was literal SQL NULL. Locks the wire contract now, ahead of the
/// writer-side fold migration landing later in the same PR.
#[tokio::test]
async fn create_entity_unfiled_reports_null_space_on_wire() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    let request = CreateEntityRequest {
        name: "Wire Contract Co".to_string(),
        entity_type: "org".to_string(),
        space: WriteSpaceTarget::Uncategorized,
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.9),
    };
    let (status, created): (StatusCode, CreateEntityResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities",
        json_body(&request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        created.space, None,
        "an unfiled entity must report space: null on the wire, not the sentinel"
    );
    assert_eq!(
        created.space_source,
        Some(WriteSpaceSource::Uncategorized),
        "an unfiled entity must report space_source: uncategorized"
    );
    assert_eq!(created.write_outcome, Some(WriteOutcome::Created));
}

/// Migration 125 (KG observation identity, PR 1): `idx_observations_identity`
/// makes a duplicate `POST .../observations` idempotent at the route layer
/// too -- the second call still returns 200 with the same id, plus a
/// warning surfacing the duplicate instead of a silently-doubled row.
#[tokio::test]
async fn add_entity_observation_twice_returns_same_id_and_warns() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    let entity_request = CreateEntityRequest {
        name: "Route Contract Dedup Entity".to_string(),
        entity_type: "concept".to_string(),
        space: Default::default(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.9),
    };
    let (status, entity): (StatusCode, CreateEntityResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities",
        json_body(&entity_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let observation_uri = format!("/api/memory/entities/{}/observations", entity.id);
    let observation_request = AddEntityObservationRequest {
        content: "The identical observation, posted twice.".to_string(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.8),
    };

    let (status, first): (StatusCode, AddObservationResponse) = request_typed(
        &router,
        Method::POST,
        &observation_uri,
        json_body(&observation_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(first.warnings.is_empty());

    let (status, second): (StatusCode, AddObservationResponse) = request_typed(
        &router,
        Method::POST,
        &observation_uri,
        json_body(&observation_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        second.id, first.id,
        "the duplicate POST returns the same id"
    );
    assert!(
        !second.warnings.is_empty(),
        "the duplicate POST must carry a warning"
    );
}
