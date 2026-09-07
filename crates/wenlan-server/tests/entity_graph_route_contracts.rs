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
    AddEntityAliasRequest, AddEntityObservationRequest, AddObservationRequest,
    ArchiveEntitiesRequest, ConfirmEntityRequest, ConfirmObservationRequest, CreateEntityRequest,
    CreateRelationRequest, EntitySelection, LinkEntityRequest, ListEntitiesRequest,
    MergeEntityRequest, RestoreEntitiesRequest, UpdateObservationRequest,
};
use wenlan_types::responses::{
    AddObservationResponse, CreateEntityResponse, CreateRelationResponse, EntityAliasesResponse,
    EntityBulkResponse, ListEntitiesResponse, MergeEntityResponse, SearchEntitiesResponse,
    SuccessResponse,
};
use wenlan_types::EntityStatus;
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

/// Like [`request_typed`], plus an `X-Wenlan-Space` header -- used to pin
/// the header name on the id-addressed entity write routes (#576).
async fn request_typed_with_space<T>(
    router: &common::AppRouter,
    method: Method,
    uri: &str,
    space: &str,
    body: Body,
) -> (StatusCode, T)
where
    T: DeserializeOwned,
{
    let request = Request::builder()
        .method(method.clone())
        .uri(uri)
        .header("content-type", "application/json")
        .header("X-Wenlan-Space", space)
        .body(body)
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 256 * 1024)
        .await
        .unwrap()
        .to_vec();
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

    let list_request = ListEntitiesRequest::default();
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

    let entity = create_test_entity(&router, "Route Contract Dedup Entity", "concept").await;

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

/// Like [`create_test_entity`], but filed into a named space (which must
/// already be registered via `db.create_space`) instead of Uncategorized.
async fn create_test_entity_in_space(
    router: &common::AppRouter,
    name: &str,
    entity_type: &str,
    space: &str,
) -> CreateEntityResponse {
    let request = CreateEntityRequest {
        name: name.to_string(),
        entity_type: entity_type.to_string(),
        space: WriteSpaceTarget::Named(space.to_string()),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.9),
    };
    let (status, entity): (StatusCode, CreateEntityResponse) = request_typed(
        router,
        Method::POST,
        "/api/memory/entities",
        json_body(&request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    entity
}

async fn create_test_entity(
    router: &common::AppRouter,
    name: &str,
    entity_type: &str,
) -> CreateEntityResponse {
    let request = CreateEntityRequest {
        name: name.to_string(),
        entity_type: entity_type.to_string(),
        space: Default::default(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.9),
    };
    let (status, entity): (StatusCode, CreateEntityResponse) = request_typed(
        router,
        Method::POST,
        "/api/memory/entities",
        json_body(&request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    entity
}

/// A `dry_run` merge previews the counts and alias additions without
/// mutating anything: the loser stays live, nothing is re-pointed.
#[tokio::test]
async fn merge_entity_dry_run_previews_without_mutating() {
    let (router, _tmp, db) = common::test_app_no_gate().await;

    let canonical = create_test_entity(&router, "Merge Canonical", "org").await;
    let loser = create_test_entity(&router, "Merge Loser", "org").await;
    db.add_observation(&loser.id, "A fact about the loser.", None, None)
        .await
        .unwrap();
    db.link_memory_entities("merge-dry-run-memory", &[loser.id.as_str()])
        .await
        .unwrap();

    let merge_uri = format!("/api/memory/entities/{}/merge", loser.id);
    let (status, preview): (StatusCode, MergeEntityResponse) = request_typed(
        &router,
        Method::POST,
        &merge_uri,
        json_body(&MergeEntityRequest {
            into: canonical.id.clone(),
            dry_run: true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!preview.applied, "a dry run must not report applied");
    assert_eq!(preview.canonical_id, canonical.id);
    assert_eq!(preview.loser_id, loser.id);
    assert_eq!(preview.observations, 1);
    assert_eq!(preview.memory_links, 1);
    assert!(preview.aliases_added.contains(&"merge loser".to_string()));

    // A dry run must not delete the loser or mutate anything.
    let loser_uri = format!("/api/memory/entities/{}", loser.id);
    let (status, _detail): (StatusCode, EntityDetail) =
        request_typed(&router, Method::GET, &loser_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK, "dry run must not delete the loser");
}

/// An applied merge deletes the loser's shadow page (404 afterwards),
/// registers the loser's name as a canonical alias, and reports the same
/// counts the preview would.
#[tokio::test]
async fn merge_entity_apply_deletes_loser_and_registers_alias() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    let canonical = create_test_entity(&router, "Apply Canonical", "org").await;
    let loser = create_test_entity(&router, "Apply Loser", "org").await;

    let merge_uri = format!("/api/memory/entities/{}/merge", loser.id);
    let (status, applied): (StatusCode, MergeEntityResponse) = request_typed(
        &router,
        Method::POST,
        &merge_uri,
        json_body(&MergeEntityRequest {
            into: canonical.id.clone(),
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(applied.applied);
    assert_eq!(applied.memory_links, 0);
    assert_eq!(applied.observations, 0);
    assert_eq!(applied.edges, 0);
    assert_eq!(applied.aliases_added, vec!["apply loser".to_string()]);

    let loser_uri = format!("/api/memory/entities/{}", loser.id);
    let (status, _error): (StatusCode, ErrorEnvelope) =
        request_typed(&router, Method::GET, &loser_uri, Body::empty()).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "the loser must be gone after an applied merge"
    );

    let canonical_uri = format!("/api/memory/entities/{}", canonical.id);
    let (status, detail): (StatusCode, EntityDetail) =
        request_typed(&router, Method::GET, &canonical_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        detail.entity.aliases.contains(&"apply loser".to_string()),
        "canonical aliases: {:?}",
        detail.entity.aliases
    );
}

/// Adding an alias is idempotent for its own owner, 409s when another
/// active entity already owns the alias, and 404s for an unknown entity.
#[tokio::test]
async fn add_entity_alias_conflict_and_idempotent() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    let owner_a = create_test_entity(&router, "Bluebird Robotics", "org").await;
    let owner_b = create_test_entity(&router, "Fernwood Bakery", "org").await;

    let alias_uri_a = format!("/api/memory/entities/{}/aliases", owner_a.id);
    let (status, aliases): (StatusCode, EntityAliasesResponse) = request_typed(
        &router,
        Method::POST,
        &alias_uri_a,
        json_body(&AddEntityAliasRequest {
            alias: "Codename".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // The owner's own lowercased name is self-seeded at creation (store_entity),
    // so "codename" joins it rather than being the sole entry.
    assert!(
        aliases.aliases.contains(&"codename".to_string()),
        "{:?}",
        aliases.aliases
    );

    // Re-adding the same alias to its own owner is a no-op, not an error.
    let (status, aliases_again): (StatusCode, EntityAliasesResponse) = request_typed(
        &router,
        Method::POST,
        &alias_uri_a,
        json_body(&AddEntityAliasRequest {
            alias: "Codename".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(aliases_again.aliases, aliases.aliases);

    // A different entity claiming the same alias is a conflict.
    let alias_uri_b = format!("/api/memory/entities/{}/aliases", owner_b.id);
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        &alias_uri_b,
        json_body(&AddEntityAliasRequest {
            alias: "Codename".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(error.error.contains(&owner_a.id), "{}", error.error);

    let (status, _error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/missing/aliases",
        json_body(&AddEntityAliasRequest {
            alias: "Codename".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// An empty (or whitespace-only) alias is a validation error, not a stored
/// blank entry in the entity's alias list.
#[tokio::test]
async fn add_entity_alias_empty_is_422() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;
    let owner = create_test_entity(&router, "Empty Alias Owner", "org").await;

    let detail_uri = format!("/api/memory/entities/{}", owner.id);
    let (status, before): (StatusCode, EntityDetail) =
        request_typed(&router, Method::GET, &detail_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);

    let alias_uri = format!("/api/memory/entities/{}/aliases", owner.id);
    for alias in ["", "   "] {
        let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
            &router,
            Method::POST,
            &alias_uri,
            json_body(&AddEntityAliasRequest {
                alias: alias.to_string(),
            }),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "alias {alias:?}");
        assert!(error.error.contains("empty"), "{}", error.error);
    }

    let (status, after): (StatusCode, EntityDetail) =
        request_typed(&router, Method::GET, &detail_uri, Body::empty()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        after.entity.aliases, before.entity.aliases,
        "a rejected empty alias must not change the alias list"
    );
    assert!(!after.entity.aliases.contains(&"".to_string()));
}

/// Spec acceptance 2: once "Origin" is declared an alias of "wenlan",
/// `POST /api/memory/entities {name:"Origin"}` must resolve to the same
/// canonical id instead of creating a duplicate entity.
#[tokio::test]
async fn create_entity_named_after_alias_resolves_to_canonical() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    let wenlan = create_test_entity(&router, "wenlan", "project").await;

    let alias_uri = format!("/api/memory/entities/{}/aliases", wenlan.id);
    let (status, _aliases): (StatusCode, EntityAliasesResponse) = request_typed(
        &router,
        Method::POST,
        &alias_uri,
        json_body(&AddEntityAliasRequest {
            alias: "Origin".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let resolved = create_test_entity(&router, "Origin", "project").await;
    assert_eq!(
        resolved.id, wenlan.id,
        "creating an entity named after a declared alias must resolve to the canonical"
    );
    assert_eq!(resolved.write_outcome, Some(WriteOutcome::ResolvedExisting));
}

/// A padded alias is trimmed before it is stored and before it is matched --
/// storage and lookup must agree, or a later `POST /api/memory/entities` with
/// the unpadded name silently fails to resolve to the canonical.
#[tokio::test]
async fn add_entity_alias_trims_padding() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    let wenlan = create_test_entity(&router, "wenlan", "project").await;

    let alias_uri = format!("/api/memory/entities/{}/aliases", wenlan.id);
    let (status, aliases): (StatusCode, EntityAliasesResponse) = request_typed(
        &router,
        Method::POST,
        &alias_uri,
        json_body(&AddEntityAliasRequest {
            alias: "  Origin  ".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        aliases.aliases.contains(&"origin".to_string()),
        "{:?}",
        aliases.aliases
    );
    assert!(
        !aliases.aliases.iter().any(|a| a != a.trim()),
        "no padded alias should be stored: {:?}",
        aliases.aliases
    );

    let resolved = create_test_entity(&router, "Origin", "project").await;
    assert_eq!(
        resolved.id, wenlan.id,
        "a trimmed alias must still resolve an unpadded name to the canonical"
    );
    assert_eq!(resolved.write_outcome, Some(WriteOutcome::ResolvedExisting));

    // Re-adding the same padded alias stays idempotent.
    let (status, aliases_again): (StatusCode, EntityAliasesResponse) = request_typed(
        &router,
        Method::POST,
        &alias_uri,
        json_body(&AddEntityAliasRequest {
            alias: "  Origin  ".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(aliases_again.aliases, aliases.aliases);
}

/// Merging an entity into itself is a validation error, not a self-merge
/// no-op.
#[tokio::test]
async fn merge_entity_same_id_is_422() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;
    let entity = create_test_entity(&router, "Merge Same Id", "org").await;

    let merge_uri = format!("/api/memory/entities/{}/merge", entity.id);
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        &merge_uri,
        json_body(&MergeEntityRequest {
            into: entity.id.clone(),
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(error.error.contains("same id"), "{}", error.error);
}

/// An unknown `into` (the canonical) is a 404, not a silent no-op.
#[tokio::test]
async fn merge_entity_unknown_into_is_404() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;
    let loser = create_test_entity(&router, "Merge Unknown Into", "org").await;

    let merge_uri = format!("/api/memory/entities/{}/merge", loser.id);
    let (status, _error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        &merge_uri,
        json_body(&MergeEntityRequest {
            into: "no-such-canonical".to_string(),
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// An unknown loser (`{id}` in the path) is a 404, not a silent no-op.
#[tokio::test]
async fn merge_entity_unknown_loser_is_404() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;
    let canonical = create_test_entity(&router, "Merge Unknown Loser", "org").await;

    let (status, _error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/no-such-loser/merge",
        json_body(&MergeEntityRequest {
            into: canonical.id.clone(),
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Omitting `dry_run` from the request body defaults to an apply (the
/// field's `#[serde(default)]` is `false`), not a safe no-op -- the loser
/// must be gone afterward exactly like an explicit `dry_run: false`.
#[tokio::test]
async fn merge_entity_missing_dry_run_field_applies() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    let canonical = create_test_entity(&router, "Missing DryRun Canonical", "org").await;
    let loser = create_test_entity(&router, "Missing DryRun Loser", "org").await;

    let merge_uri = format!("/api/memory/entities/{}/merge", loser.id);
    let body =
        Body::from(serde_json::to_vec(&serde_json::json!({ "into": canonical.id })).unwrap());
    let (status, applied): (StatusCode, MergeEntityResponse) =
        request_typed(&router, Method::POST, &merge_uri, body).await;
    assert_eq!(status, StatusCode::OK);
    assert!(applied.applied, "omitting dry_run must default to an apply");

    let loser_uri = format!("/api/memory/entities/{}", loser.id);
    let (status, _error): (StatusCode, ErrorEnvelope) =
        request_typed(&router, Method::GET, &loser_uri, Body::empty()).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "the loser must be gone after the default (non-dry-run) apply"
    );
}

/// An applied merge with a real observation, memory link, and edge on the
/// loser reports counts equal to what the transaction actually moved
/// (Findings 2 and 3): the preview's dedup-aware counts and the outcome's
/// own `changes()`-captured counts must agree, not just both be nonzero.
#[tokio::test]
async fn merge_entity_apply_counts_match_what_moved() {
    let (router, _tmp, db) = common::test_app_no_gate().await;

    let canonical = create_test_entity(&router, "Counts Canonical", "org").await;
    let loser = create_test_entity(&router, "Counts Loser", "org").await;
    let other = create_test_entity(&router, "Counts Edge Target", "org").await;

    db.add_observation(
        &loser.id,
        "A fact that must move onto the canonical.",
        None,
        None,
    )
    .await
    .unwrap();
    db.link_memory_entities("merge-counts-memory", &[loser.id.as_str()])
        .await
        .unwrap();

    let relation_request = CreateRelationRequest {
        from_entity: loser.id.clone(),
        to_entity: other.id.clone(),
        relation_type: "depends_on".to_string(),
        source_agent: Some("entity-graph-route-contract".to_string()),
        confidence: Some(0.8),
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let (status, _relation): (StatusCode, CreateRelationResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/relations",
        json_body(&relation_request),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let merge_uri = format!("/api/memory/entities/{}/merge", loser.id);
    let (status, preview): (StatusCode, MergeEntityResponse) = request_typed(
        &router,
        Method::POST,
        &merge_uri,
        json_body(&MergeEntityRequest {
            into: canonical.id.clone(),
            dry_run: true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(preview.memory_links, 1);
    assert_eq!(preview.observations, 1);
    assert_eq!(preview.edges, 1);

    let (status, applied): (StatusCode, MergeEntityResponse) = request_typed(
        &router,
        Method::POST,
        &merge_uri,
        json_body(&MergeEntityRequest {
            into: canonical.id.clone(),
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(applied.applied);
    assert_eq!(applied.memory_links, preview.memory_links);
    assert_eq!(applied.observations, preview.observations);
    assert_eq!(applied.edges, preview.edges);
    assert_eq!(applied.memory_links, 1);
    assert_eq!(applied.observations, 1);
    assert_eq!(applied.edges, 1);
}

/// When the alias conflict's owner is a live entity whose own name equals
/// the alias, the 409 must say so and point at the merge path instead of
/// the generic "already owned by" wording.
#[tokio::test]
async fn add_entity_alias_conflict_with_live_namesake_points_at_merge() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    let origin = create_test_entity(&router, "Origin", "project").await;
    let wenlan = create_test_entity(&router, "wenlan", "project").await;

    let alias_uri = format!("/api/memory/entities/{}/aliases", wenlan.id);
    let (status, error): (StatusCode, ErrorEnvelope) = request_typed(
        &router,
        Method::POST,
        &alias_uri,
        json_body(&AddEntityAliasRequest {
            alias: "Origin".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(error.error.contains(&origin.id), "{}", error.error);
    assert!(
        error.error.contains("/merge"),
        "must point at the merge route: {}",
        error.error
    );
    assert!(
        error.error.contains("wenlan entities merge"),
        "must point at the CLI merge command: {}",
        error.error
    );
}

/// #576: each of the five id-addressed entity write routes honors
/// `X-Wenlan-Space` -- a request carrying the entity's own space succeeds
/// exactly as the header-less path did before scoping landed. Pins the
/// header name itself, not just the underlying scope behavior (covered by
/// `wave_4_knowledge_scopes_entity_writes` in `space_scoping_e2e`).
#[tokio::test]
async fn entity_write_routes_succeed_with_matching_space_header() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();

    let confirm_target =
        create_test_entity_in_space(&router, "Header Scope Confirm", "org", "work").await;
    let (status, confirmed): (StatusCode, SuccessResponse) = request_typed_with_space(
        &router,
        Method::PUT,
        &format!("/api/memory/entities/{}/confirm", confirm_target.id),
        "work",
        json_body(&ConfirmEntityRequest { confirmed: true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(confirmed.ok);

    let observation_target =
        create_test_entity_in_space(&router, "Header Scope Observation", "org", "work").await;
    let (status, added): (StatusCode, AddObservationResponse) = request_typed_with_space(
        &router,
        Method::POST,
        &format!(
            "/api/memory/entities/{}/observations",
            observation_target.id
        ),
        "work",
        json_body(&AddEntityObservationRequest {
            content: "An observation reached through the space header.".to_string(),
            source_agent: None,
            confidence: None,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!added.id.is_empty());

    let alias_target =
        create_test_entity_in_space(&router, "Header Scope Alias", "org", "work").await;
    let (status, aliases): (StatusCode, EntityAliasesResponse) = request_typed_with_space(
        &router,
        Method::POST,
        &format!("/api/memory/entities/{}/aliases", alias_target.id),
        "work",
        json_body(&AddEntityAliasRequest {
            alias: "Header Alias".to_string(),
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(aliases.aliases.contains(&"header alias".to_string()));

    let merge_canonical =
        create_test_entity_in_space(&router, "Header Scope Merge Canonical", "org", "work").await;
    let merge_loser =
        create_test_entity_in_space(&router, "Header Scope Merge Loser", "org", "work").await;
    let (status, merged): (StatusCode, MergeEntityResponse) = request_typed_with_space(
        &router,
        Method::POST,
        &format!("/api/memory/entities/{}/merge", merge_loser.id),
        "work",
        json_body(&MergeEntityRequest {
            into: merge_canonical.id.clone(),
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(merged.applied);

    let delete_target =
        create_test_entity_in_space(&router, "Header Scope Delete", "org", "work").await;
    let (status, deleted): (StatusCode, SuccessResponse) = request_typed_with_space(
        &router,
        Method::DELETE,
        &format!("/api/memory/entities/{}/delete", delete_target.id),
        "work",
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(deleted.ok);
}

/// #708: the Entities view's three routes. `/query` sees every lifecycle state
/// and reports the unpaged total; `/archive` and `/restore` are exact inverses
/// and take either explicit ids or the same filter the view is showing.
#[tokio::test]
async fn entity_graph_routes_query_archive_and_restore_round_trip() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;
    let quiet = create_test_entity(&router, "Zephyr Analytics", "organization").await;
    let loud = create_test_entity(&router, "Borealis Freight", "organization").await;

    // Establishing one of them by hand leaves exactly one detected entity.
    let (status, confirmed): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::PUT,
        &format!("/api/memory/entities/{}/confirm", loud.id),
        json_body(&ConfirmEntityRequest { confirmed: true }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(confirmed.ok);

    let detected_filter = ListEntitiesRequest {
        status: Some(EntityStatus::Detected),
        ..Default::default()
    };
    let (status, detected): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/query",
        json_body(&detected_filter),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(detected.total, 1);
    assert_eq!(detected.entities.len(), 1);
    assert_eq!(detected.entities[0].id, quiet.id);
    assert_eq!(detected.entities[0].status, EntityStatus::Detected);
    assert_eq!(detected.entities[0].memory_count, 0);
    assert_eq!(detected.entities[0].established_by, None);

    let (status, established): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/query",
        json_body(&ListEntitiesRequest {
            status: Some(EntityStatus::Established),
            ..Default::default()
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(established.entities.len(), 1);
    assert_eq!(established.entities[0].id, loud.id);
    assert_eq!(
        established.entities[0].established_by.as_deref(),
        Some("manual")
    );

    // The legacy list route now reports a total alongside its rows.
    let (status, listed): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/list",
        json_body(&ListEntitiesRequest::default()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(listed.total, listed.entities.len() as u64);

    // A dry run answers "how many, and which" without touching anything.
    let by_ids = EntitySelection {
        ids: Some(vec![quiet.id.clone()]),
        filter: None,
    };
    let (status, preview): (StatusCode, EntityBulkResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/archive",
        json_body(&ArchiveEntitiesRequest {
            selection: by_ids.clone(),
            dry_run: true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(preview.count, 1);
    assert!(preview.dry_run);
    assert_eq!(preview.entity_ids, vec![quiet.id.clone()]);

    let (status, still_detected): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/query",
        json_body(&detected_filter),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(still_detected.total, 1, "a dry run must not archive");

    let (status, archived): (StatusCode, EntityBulkResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/archive",
        json_body(&ArchiveEntitiesRequest {
            selection: by_ids.clone(),
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(archived.count, 1);
    assert!(!archived.dry_run);

    let (status, now_archived): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/query",
        json_body(&ListEntitiesRequest {
            status: Some(EntityStatus::Archived),
            ..Default::default()
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(now_archived.entities.len(), 1);
    assert_eq!(now_archived.entities[0].id, quiet.id);

    let (status, restored): (StatusCode, EntityBulkResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/restore",
        json_body(&RestoreEntitiesRequest {
            selection: by_ids,
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(restored.count, 1);

    let (status, back): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/query",
        json_body(&detected_filter),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(back.total, 1);
    assert_eq!(back.entities[0].id, quiet.id);
}

/// A selection must name exactly one of `ids` or `filter`: both is ambiguous,
/// and neither would mean archiving the entire scope.
#[tokio::test]
async fn entity_graph_routes_refuse_an_ambiguous_bulk_selection() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;

    for selection in [
        EntitySelection::default(),
        EntitySelection {
            ids: Some(vec!["some-id".to_string()]),
            filter: Some(ListEntitiesRequest::default()),
        },
    ] {
        for uri in [
            "/api/memory/entities/archive",
            "/api/memory/entities/restore",
        ] {
            let (status, body) = request_bytes(
                &router,
                Method::POST,
                uri,
                json_body(&ArchiveEntitiesRequest {
                    selection: selection.clone(),
                    dry_run: true,
                }),
                false,
            )
            .await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{uri} must refuse an ambiguous selection: {}",
                String::from_utf8_lossy(&body)
            );
        }
    }
}

/// Sol review of #711: `/archive` and `/restore` fold a `filter.space` into
/// the scope exactly as `/query` does. Before this they scoped by header only,
/// so `{filter:{space:"work"}}` with no header read back one Space in the
/// dry run and then archived every Space on apply.
#[tokio::test]
async fn entity_bulk_routes_honor_the_filter_space_like_query() {
    let (router, _tmp, db) = common::test_app_no_gate().await;
    db.create_space("work", None, false).await.unwrap();
    db.create_space("home", None, false).await.unwrap();
    let work = create_test_entity_in_space(&router, "Work Guess", "concept", "work").await;
    let home = create_test_entity_in_space(&router, "Home Guess", "concept", "home").await;

    let work_filter = ListEntitiesRequest {
        status: Some(EntityStatus::Detected),
        space: Some("work".to_string()),
        ..Default::default()
    };
    let (status, seen): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/query",
        json_body(&work_filter),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(seen.total, 1);
    assert_eq!(seen.entities[0].id, work.id);

    let selection = EntitySelection {
        ids: None,
        filter: Some(work_filter.clone()),
    };
    let (status, preview): (StatusCode, EntityBulkResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/archive",
        json_body(&ArchiveEntitiesRequest {
            selection: selection.clone(),
            dry_run: true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        preview.count, 1,
        "the dry run sees the one Space the filter names"
    );
    assert_eq!(preview.entity_ids, vec![work.id.clone()]);

    let (status, applied): (StatusCode, EntityBulkResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/archive",
        json_body(&ArchiveEntitiesRequest {
            selection: selection.clone(),
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        applied.count, 1,
        "the apply touches what the dry run showed"
    );
    assert_eq!(applied.entity_ids, vec![work.id.clone()]);

    // The other Space's entity is untouched.
    let (status, home_seen): (StatusCode, ListEntitiesResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/query",
        json_body(&ListEntitiesRequest {
            status: Some(EntityStatus::Detected),
            space: Some("home".to_string()),
            ..Default::default()
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(home_seen.total, 1);
    assert_eq!(home_seen.entities[0].id, home.id);

    // Restore scopes the same way.
    let (status, restored): (StatusCode, EntityBulkResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/restore",
        json_body(&RestoreEntitiesRequest {
            selection: EntitySelection {
                ids: None,
                filter: Some(ListEntitiesRequest {
                    status: Some(EntityStatus::Archived),
                    space: Some("work".to_string()),
                    ..Default::default()
                }),
            },
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(restored.entity_ids, vec![work.id.clone()]);

    // An unknown Space in the filter is refused, not silently widened.
    let (status, bytes) = request_bytes(
        &router,
        Method::POST,
        "/api/memory/entities/archive",
        json_body(&ArchiveEntitiesRequest {
            selection: EntitySelection {
                ids: None,
                filter: Some(ListEntitiesRequest {
                    space: Some("nowhere".to_string()),
                    ..Default::default()
                }),
            },
            dry_run: true,
        }),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let envelope: ErrorEnvelope = serde_json::from_slice(&bytes).unwrap();
    assert!(
        envelope.error.contains("unknown Space"),
        "{}",
        envelope.error
    );
}

/// Sol review of #711: the Archived tab's "Delete permanently" hits the
/// scoped delete route, which used to gate on a live-only scope check and
/// answer 404 for every archived row.
#[tokio::test]
async fn delete_entity_route_deletes_an_archived_entity() {
    let (router, _tmp, _db) = common::test_app_no_gate().await;
    let entity = create_test_entity(&router, "Quill Harbor", "place").await;
    let (status, archived): (StatusCode, EntityBulkResponse) = request_typed(
        &router,
        Method::POST,
        "/api/memory/entities/archive",
        json_body(&ArchiveEntitiesRequest {
            selection: EntitySelection {
                ids: Some(vec![entity.id.clone()]),
                filter: None,
            },
            dry_run: false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(archived.count, 1);

    let (status, deleted): (StatusCode, SuccessResponse) = request_typed(
        &router,
        Method::DELETE,
        &format!("/api/memory/entities/{}/delete", entity.id),
        Body::empty(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(deleted.ok);

    let (status, _) = request_bytes(
        &router,
        Method::GET,
        &format!("/api/memory/entities/{}", entity.id),
        Body::empty(),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
