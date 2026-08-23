// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::memory_routes::extract_agent_name;
use crate::route_registry::{delete, get, post, put, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_types::requests::{
    AddEntityAliasRequest, AddObservationRequest, CreateEntityRequest, CreateRelationRequest,
    MergeEntityRequest,
};
use wenlan_types::responses::{
    AddObservationResponse, CreateEntityResponse, CreateRelationResponse, EntityAliasesResponse,
    MergeEntityResponse,
};
use wenlan_types::{WriteOutcome, WriteSpaceSource, WriteSpaceTarget};

// ===== Knowledge Graph Types =====

#[derive(Debug, Deserialize)]
pub struct LinkEntityRequest {
    pub source_id: String,
    pub entity_id: String,
}

pub(crate) fn register_writes(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/memory/entities", post(handle_create_entity))
        .route("/api/memory/relations", post(handle_create_relation))
        .route("/api/memory/observations", post(handle_add_observation))
        .route("/api/memory/link-entity", post(handle_link_entity))
}

pub(crate) fn register_reads(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/memory/entities/list", post(handle_list_entities))
        .route("/api/memory/entities/search", post(handle_search_entities))
        .route("/api/memory/graph", get(handle_get_knowledge_graph))
        .route(
            "/api/memory/entities/{entity_id}",
            get(handle_get_entity_detail),
        )
}

pub(crate) fn register_crud(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route(
            "/api/memory/entities/{id}/confirm",
            put(handle_confirm_entity),
        )
        .route(
            "/api/memory/entities/{id}/delete",
            delete(handle_delete_entity),
        )
        .route(
            "/api/memory/entities/{entity_id}/observations",
            post(handle_add_entity_observation),
        )
        .route("/api/memory/entities/{id}/merge", post(handle_merge_entity))
        .route(
            "/api/memory/entities/{id}/aliases",
            post(handle_add_entity_alias),
        )
        .route(
            "/api/memory/observations/{id}",
            put(handle_update_observation).delete(handle_delete_observation),
        )
        .route(
            "/api/memory/observations/{id}/confirm",
            put(handle_confirm_observation),
        )
}

// ===== Knowledge Graph Handlers =====

pub async fn handle_create_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(mut req): Json<CreateEntityRequest>,
) -> Result<Json<CreateEntityResponse>, ServerError> {
    let agent = extract_agent_name(&headers, None);
    let db = {
        let s = state.read().await;
        s.db.as_ref()
            .cloned()
            .ok_or(ServerError::DbNotInitialized)?
    };
    let _space_write_guard = db.lock_space_writes().await;
    let resolved = db
        .resolve_write_space(&req.space, header_space.as_deref())
        .await?;
    req.space = match resolved.space_name.as_ref() {
        Some(name) => WriteSpaceTarget::Named(name.clone()),
        None => WriteSpaceTarget::Uncategorized,
    };
    let result = wenlan_core::post_write::create_entity(&db, req, &agent).await?;
    let persisted_space = db.get_entity_detail(&result.id).await?.entity.space;
    let (space_source, write_outcome) = if result.wrote {
        (
            if persisted_space.is_some() {
                resolved.source
            } else {
                WriteSpaceSource::Uncategorized
            },
            WriteOutcome::Created,
        )
    } else {
        (WriteSpaceSource::Existing, WriteOutcome::ResolvedExisting)
    };
    Ok(Json(CreateEntityResponse {
        id: result.id,
        warnings: result.warnings,
        space: persisted_space,
        space_source: Some(space_source),
        write_outcome: Some(write_outcome),
    }))
}

pub async fn handle_create_relation(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: axum::http::HeaderMap,
    Json(mut req): Json<CreateRelationRequest>,
) -> Result<Json<CreateRelationResponse>, ServerError> {
    let agent = extract_agent_name(&headers, req.source_agent.as_deref());
    let db = {
        let s = state.read().await;
        s.db.as_ref()
            .cloned()
            .ok_or(ServerError::DbNotInitialized)?
    };
    // M3g span capture (span/model_version/prompt_version) is daemon-internal
    // (KG extraction) only -- strip them so an agent-triggered request can
    // never set them, matching the CreateRelationRequest doc comment.
    req.span = None;
    req.model_version = None;
    req.prompt_version = None;
    let result = wenlan_core::post_write::create_relation(&db, req, &agent).await?;
    Ok(Json(CreateRelationResponse {
        id: result.id,
        warnings: result.warnings,
    }))
}

pub async fn handle_add_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    Json(req): Json<AddObservationRequest>,
) -> Result<Json<AddObservationResponse>, ServerError> {
    let agent = extract_agent_name(&headers, req.source_agent.as_deref());
    let db = {
        let s = state.read().await;
        s.db.as_ref()
            .cloned()
            .ok_or(ServerError::DbNotInitialized)?
    };
    let result = wenlan_core::post_write::add_observation(&db, req, &agent).await?;
    Ok(Json(AddObservationResponse {
        id: result.id,
        warnings: result.warnings,
    }))
}

pub async fn handle_link_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<LinkEntityRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref()
            .cloned()
            .ok_or(ServerError::DbNotInitialized)?
    };
    let updated = db
        .update_memory_entity_id(&req.source_id, &req.entity_id)
        .await
        .map_err(|e| ServerError::IngestFailed(e.to_string()))?;
    if updated == 0 {
        return Err(ServerError::NotFound(format!(
            "memory '{}' does not exist",
            req.source_id
        )));
    }
    Ok(Json(serde_json::json!({"linked": true})))
}

// ===== Knowledge Graph Retrieval Handlers =====

#[derive(Debug, Deserialize)]
pub struct ListEntitiesRequest {
    #[serde(default)]
    pub entity_type: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListEntitiesResponse {
    pub entities: Vec<wenlan_core::db::Entity>,
}

pub async fn handle_list_entities(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(req): Json<ListEntitiesRequest>,
) -> Result<Json<ListEntitiesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope =
        crate::read_scope::effective_read_scope(&db, req.space.as_deref(), header_space.as_deref())
            .await?;
    let entities = db
        .list_entities_scoped(req.entity_type.as_deref(), &scope)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(ListEntitiesResponse { entities }))
}

pub async fn handle_get_entity_detail(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(entity_id): Path<String>,
) -> Result<Json<wenlan_core::db::EntityDetail>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let detail = db.get_entity_detail_scoped(&entity_id, &scope).await?;
    Ok(Json(detail))
}

/// GET /api/memory/graph — the whole graph for the requested scope in one
/// read. Space-scoped exactly like `handle_get_entity_detail`: the
/// `SpaceHeader` resolves through `effective_read_scope`.
///
/// # Why `filter_page_refs` and not `filter_pages`
///
/// A `GraphPageNode` is not a `Page`: it carries the page's id and title and
/// nothing else, and it has no field for the two truth axes. `filter_pages`
/// serves an `EntryOnly` page as a reduced entry — id, title, and BOTH axes —
/// and `truth_adapter` is explicit that an entry without its axes is exactly
/// the unearned trust the carve-out exists to prevent. This wire type cannot
/// carry them, so the carve-out is not available to it.
///
/// `filter_page_refs` is the operation for that shape and says so in its own
/// doc: things that hang off a page rather than being one, **map nodes**
/// included. `Full` or gone. At generation 0 every page is `Full`, so this is
/// the identity today, same as every other adapter.
///
/// A dropped page takes its links with it. Removing the node and keeping the
/// edge would either orphan the edge or silently rewire the graph — the exact
/// hazard the reader manifest names in its `/api/pages/{id}/map` demotion note.
pub async fn handle_get_knowledge_graph(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
) -> Result<Json<wenlan_types::KnowledgeGraphResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let mut graph = db
        .get_knowledge_graph_scoped(&scope)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;

    graph.pages = wenlan_core::truth_adapter::filter_page_refs(
        &db,
        &view.grant,
        std::mem::take(&mut graph.pages),
        |page| page.id.as_str(),
    )
    .await
    .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    let visible: std::collections::HashSet<&str> =
        graph.pages.iter().map(|page| page.id.as_str()).collect();
    // Only the `page` endpoints are re-checked: an `entity` or `memory`
    // endpoint is an id into a collection this route already scoped, and
    // neither is a page.
    graph.page_links.retain(|link| {
        let endpoint_visible = |endpoint: &wenlan_types::GraphRef| {
            endpoint.kind != "page" || visible.contains(endpoint.id.as_str())
        };
        endpoint_visible(&link.from) && endpoint_visible(&link.to)
    });
    Ok(Json(graph))
}

#[derive(Debug, Deserialize)]
pub struct SearchEntitiesRequest {
    pub query: String,
    #[serde(default = "default_entity_search_limit")]
    pub limit: usize,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

fn default_entity_search_limit() -> usize {
    20
}

#[derive(Debug, Serialize)]
pub struct SearchEntitiesResponse {
    pub results: Vec<wenlan_core::db::EntitySearchResult>,
}

pub async fn handle_search_entities(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(req): Json<SearchEntitiesRequest>,
) -> Result<Json<SearchEntitiesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope =
        crate::read_scope::effective_read_scope(&db, req.space.as_deref(), header_space.as_deref())
            .await?;
    let results = db
        .search_entities_by_vector_scoped(&req.query, req.limit, &scope)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(SearchEntitiesResponse { results }))
}

// =====================================================================
// Batch 3 — Entity / Observation CRUD
// =====================================================================

/// PUT /api/memory/entities/{id}/confirm
pub async fn handle_confirm_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::ConfirmEntityRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.confirm_entity(&id, req.confirmed)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// DELETE /api/memory/entities/{id}/delete
pub async fn handle_delete_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_entity(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/memory/entities/{entity_id}/observations
pub async fn handle_add_entity_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(entity_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<wenlan_types::requests::AddEntityObservationRequest>,
) -> Result<Json<wenlan_types::responses::AddObservationResponse>, ServerError> {
    let agent = extract_agent_name(&headers, req.source_agent.as_deref());
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // Same validity contract as `POST /api/memory/observations`: entity must
    // exist, content >= 5 chars, confidence in [0, 1].
    let req = AddObservationRequest {
        entity_id,
        content: req.content,
        source_agent: req.source_agent,
        confidence: req.confidence,
    };
    let result = wenlan_core::post_write::add_observation(&db, req, &agent).await?;
    Ok(Json(wenlan_types::responses::AddObservationResponse {
        id: result.id,
        warnings: result.warnings,
    }))
}

/// POST /api/memory/entities/{id}/merge -- merge `{id}` (the loser) into
/// `into` (the canonical). `dry_run` returns the preview without mutating
/// anything; `applied` on the response tells the caller which happened.
pub async fn handle_merge_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<MergeEntityRequest>,
) -> Result<Json<MergeEntityResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    if req.dry_run {
        let preview = db.merge_entities_preview(&req.into, &id).await?;
        return Ok(Json(MergeEntityResponse {
            canonical_id: preview.canonical_id,
            canonical_name: preview.canonical_name,
            loser_id: preview.loser_id,
            loser_name: preview.loser_name,
            memory_links: preview.memory_links,
            observations: preview.observations,
            edges: preview.edges,
            aliases_added: preview.aliases_added,
            applied: false,
        }));
    }
    // One locked pass: the merge validates the ids itself and reports the
    // names and counts it actually moved, so an apply needs no preview.
    let outcome = db.merge_entities(&req.into, &id).await?;
    if !outcome.merged {
        return Err(ServerError::NotFound(format!("entity {id} not found")));
    }
    Ok(Json(MergeEntityResponse {
        canonical_id: req.into,
        canonical_name: outcome.canonical_name,
        loser_id: id,
        loser_name: outcome.loser_name,
        memory_links: outcome.memory_links,
        observations: outcome.observations,
        edges: outcome.edges,
        aliases_added: outcome.aliases_added,
        applied: true,
    }))
}

/// POST /api/memory/entities/{id}/aliases -- declare `alias` as an
/// additional name for entity `{id}`. Idempotent when `{id}` already owns
/// the alias; 409 when another active entity owns it.
pub async fn handle_add_entity_alias(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<AddEntityAliasRequest>,
) -> Result<Json<EntityAliasesResponse>, ServerError> {
    if req.alias.trim().is_empty() {
        return Err(ServerError::ValidationError(
            "alias must not be empty".into(),
        ));
    }
    let alias = req.alias.trim();
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // 404 when `id` does not name a live entity; idempotent when it already
    // owns the alias.
    let Some((_, aliases)) = db.entity_name_and_aliases(&id).await? else {
        return Err(ServerError::NotFound(format!("entity {id} not found")));
    };
    let alias_lower = alias.to_lowercase();
    if aliases.contains(&alias_lower) {
        return Ok(Json(EntityAliasesResponse {
            entity_id: id,
            aliases,
        }));
    }
    if let Some(owner_id) = db.resolve_entity_by_alias(&alias_lower).await? {
        if owner_id != id {
            // `store_entity` self-seeds every live entity's aliases with its
            // own lowercased name, so this conflict is most often "that's
            // someone else's current name" rather than a previously-declared
            // alias -- say so and point at merge.
            let owner_name = db
                .entity_name_and_aliases(&owner_id)
                .await
                .ok()
                .flatten()
                .map(|(name, _)| name);
            if owner_name
                .as_deref()
                .is_some_and(|name| name.to_lowercase() == alias_lower)
            {
                return Err(ServerError::Conflict(format!(
                    "alias {:?} is the name of live entity {owner_id}; use \
                     POST /api/memory/entities/{owner_id}/merge (CLI: wenlan \
                     entities merge ...) instead",
                    alias
                )));
            }
            return Err(ServerError::Conflict(format!(
                "alias {:?} is already owned by entity {owner_id}",
                alias
            )));
        }
    }
    db.add_entity_alias(alias, &id, "api").await?;
    // `add_entity_alias` skips silently when the write-time ownership guard
    // fails: re-read and confirm the alias actually landed.
    let aliases = db
        .entity_name_and_aliases(&id)
        .await?
        .map(|(_, aliases)| aliases)
        .unwrap_or_default();
    if !aliases.contains(&alias_lower) {
        return Err(ServerError::Conflict(format!(
            "alias {:?} was not recorded for entity {id}",
            alias
        )));
    }
    Ok(Json(EntityAliasesResponse {
        entity_id: id,
        aliases,
    }))
}

/// PUT /api/memory/observations/{id}
pub async fn handle_update_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::UpdateObservationRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.update_observation(&id, &req.content)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// DELETE /api/memory/observations/{id}
pub async fn handle_delete_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_observation(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// PUT /api/memory/observations/{id}/confirm
pub async fn handle_confirm_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::ConfirmObservationRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.confirm_observation(&id, req.confirmed)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}
