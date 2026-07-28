// SPDX-License-Identifier: Apache-2.0
//! Page Map (mind-map) v1 HTTP routes. See
//! docs/superpowers/plans/2026-07-18-page-map-api-spec.md for the full
//! contract; this module wires the wenlan-core data layer
//! (`db::page_map`) to HTTP, staying thin per the crate boundary rule (no
//! business logic here). The improve pass itself lives in
//! `wenlan_core::page_map_improve`; `handle_improve_page_map` only invokes it.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::Json;
use serde::Deserialize;
use tokio::sync::RwLock;

use wenlan_core::db::page_map::{
    CreateEdgeOutcome, CreateNodeOutcome, EdgePatch, NodeLayout, NodePatch, PageMapData,
    PageMapEdge as CoreEdge, PageMapNode as CoreNode,
};
use wenlan_core::db::MemoryDB;
use wenlan_types::page_map::{
    CreateMapEdgeRequest, CreateMapNodeRequest, DeleteMapEdgeRequest, DeleteMapNodeRequest,
    EdgeMutationResponse, NodeMutationResponse, PageMapEdge as WireEdge, PageMapNode as WireNode,
    PageMapResponse, PageMapViewport, PatchMapEdgeRequest, PatchMapNodeRequest,
    PutPageMapLayoutRequest, RefState,
};

use crate::error::ServerError;
use crate::state::ServerState;

async fn ensure_page_exists(db: &MemoryDB, page_id: &str) -> Result<(), ServerError> {
    db.get_page(page_id)
        .await?
        .ok_or_else(|| ServerError::NotFound(format!("page {page_id} not found")))?;
    Ok(())
}

async fn ensure_page_is_active(db: &MemoryDB, page_id: &str) -> Result<(), ServerError> {
    let page = db
        .get_page(page_id)
        .await?
        .ok_or_else(|| ServerError::NotFound(format!("page {page_id} not found")))?;
    if page.status != "active" {
        return Err(ServerError::ValidationError(format!(
            "Page {page_id} is not active"
        )));
    }
    Ok(())
}

fn root_node_id(nodes: &[CoreNode]) -> Option<String> {
    nodes
        .iter()
        .find(|n| n.parent_id.is_none())
        .map(|n| n.id.clone())
}

/// `ref_state` is computed at read time, never stored: live iff the node's
/// backing object still resolves. `section` refs additionally check that the
/// heading itself is still present in the page's content (`ref_id` is
/// `"{page_id}#{heading-slug}"`), reusing `wenlan_core::page_map_improve`'s
/// exact heading extraction + slug function so this never duplicates that
/// logic.
async fn compute_ref_state(
    db: &MemoryDB,
    ref_kind: &str,
    ref_id: &str,
) -> Result<RefState, ServerError> {
    let live = match ref_kind {
        "memory" => db.get_memory_type(ref_id).await?.is_some(),
        "entity" => db.get_entity_name_type(ref_id).await?.is_some(),
        "page" => db.get_page(ref_id).await?.is_some(),
        "section" => match ref_id.split_once('#') {
            Some((page_id, heading_slug)) => match db.get_page(page_id).await? {
                Some(page) => wenlan_core::page_map_improve::extract_headings(&page.content)
                    .iter()
                    .any(|h| wenlan_core::export::obsidian::slugify(h) == heading_slug),
                None => false,
            },
            None => false,
        },
        _ => false,
    };
    Ok(if live {
        RefState::Live
    } else {
        RefState::Dangling
    })
}

async fn wire_node(db: &MemoryDB, node: CoreNode) -> Result<WireNode, ServerError> {
    let ref_state = compute_ref_state(db, &node.ref_kind, &node.ref_id).await?;
    Ok(WireNode {
        id: node.id,
        parent_id: node.parent_id,
        rank: node.rank,
        ref_kind: node.ref_kind,
        ref_id: node.ref_id,
        label: node.label,
        status: node.status,
        pinned: node.pinned,
        placed: node.placed,
        collapsed: node.collapsed,
        x: node.x,
        y: node.y,
        width: node.width,
        height: node.height,
        ref_state,
    })
}

fn wire_edge(edge: CoreEdge) -> WireEdge {
    WireEdge {
        id: edge.id,
        from_node: edge.from_node,
        to_node: edge.to_node,
        kind: edge.kind,
        label: edge.label,
        status: edge.status,
    }
}

async fn build_map_response(
    db: &MemoryDB,
    page_id: &str,
    data: PageMapData,
) -> Result<PageMapResponse, ServerError> {
    let viewport = data
        .map
        .viewport
        .as_deref()
        .and_then(|v| serde_json::from_str::<PageMapViewport>(v).ok());
    let mut nodes = Vec::with_capacity(data.nodes.len());
    for node in data.nodes {
        nodes.push(wire_node(db, node).await?);
    }
    let edges = data.edges.into_iter().map(wire_edge).collect();
    Ok(PageMapResponse {
        page_id: page_id.to_string(),
        revision: data.map.revision,
        map_schema: data.map.map_schema,
        viewport,
        nodes,
        edges,
    })
}

#[derive(Debug, Deserialize)]
pub struct MapIncludeQuery {
    #[serde(default)]
    pub include: Option<String>,
}

/// GET /api/pages/{id}/map
///
/// 200 always for an existing page: an absent map synthesizes the empty
/// shape (`revision: 0`, no nodes/edges) rather than 404ing or
/// auto-creating a `page_maps` row. 404 only when the page itself doesn't
/// exist.
pub async fn handle_get_page_map(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(page_id): Path<String>,
    Query(query): Query<MapIncludeQuery>,
) -> Result<Json<PageMapResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_exists(&db, &page_id).await?;

    let include_dismissed = query.include.as_deref() == Some("dismissed");
    let data = db.get_page_map(&page_id, include_dismissed).await?;
    let response = match data {
        Some(data) => build_map_response(&db, &page_id, data).await?,
        None => PageMapResponse {
            page_id,
            revision: 0,
            map_schema: 1,
            viewport: None,
            nodes: vec![],
            edges: vec![],
        },
    };
    Ok(Json(response))
}

/// PUT /api/pages/{id}/map/layout
pub async fn handle_put_page_map_layout(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(page_id): Path<String>,
    Json(req): Json<PutPageMapLayoutRequest>,
) -> Result<Json<PageMapResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    let viewport_json = req
        .viewport
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|e| ServerError::BadRequest(format!("invalid viewport: {e}")))?;
    let positions: Vec<NodeLayout> = req
        .positions
        .into_iter()
        .map(|p| NodeLayout {
            node_id: p.node_id,
            x: p.x,
            y: p.y,
            width: p.width,
            height: p.height,
            collapsed: p.collapsed,
        })
        .collect();

    let data = db
        .put_page_map_layout(
            &page_id,
            req.base_revision,
            viewport_json.as_deref(),
            &positions,
        )
        .await?;
    Ok(Json(build_map_response(&db, &page_id, data).await?))
}

/// POST /api/pages/{id}/map/nodes
///
/// First mutation against a never-initialized map: `init_page_map` runs
/// (idempotent) before the create. When THIS call's `init_page_map` is the
/// one that atomically performed the insert (`created == true`, decided from
/// the INSERT's own `rows_affected`, not a separate pre-check), the client's
/// `base_revision` is necessarily a guess (GET synthesizes `revision: 0` for
/// an absent map — a sentinel, not a real CAS token, since no `page_maps` row
/// ever existed to be stale against), so the freshly initialized revision is
/// used instead of trusting it; otherwise `base_revision` is enforced as a
/// normal conditional write (a stale `base_revision` 409s even though the map
/// already existed before this request). A `parent_id` of `None` attaches
/// under the map's root, resolved here so the very first node can be created
/// without the client already knowing the root's freshly-minted id.
pub async fn handle_create_map_node(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(page_id): Path<String>,
    Json(req): Json<CreateMapNodeRequest>,
) -> Result<Json<NodeMutationResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    let ref_kind = req
        .ref_kind
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("ref_kind is required".to_string()))?;
    let ref_id = req
        .ref_id
        .as_deref()
        .ok_or_else(|| ServerError::BadRequest("ref_id is required".to_string()))?;

    let (map, created) = db.init_page_map(&page_id).await?;
    let parent_id = match req.parent_id.clone() {
        Some(id) => id,
        None => root_node_id(&map.nodes)
            .ok_or_else(|| ServerError::Internal("page map has no root node".to_string()))?,
    };
    let base_revision = if created {
        map.map.revision
    } else {
        req.base_revision
    };

    let outcome = db
        .create_map_node(
            &page_id,
            base_revision,
            &parent_id,
            ref_kind,
            ref_id,
            req.label.as_deref(),
            req.rank,
        )
        .await?;

    match outcome {
        CreateNodeOutcome::Created(node) => Ok(Json(NodeMutationResponse {
            revision: base_revision + 1,
            node: wire_node(&db, node).await?,
        })),
        CreateNodeOutcome::Duplicate(node) => Ok(Json(NodeMutationResponse {
            revision: base_revision,
            node: wire_node(&db, node).await?,
        })),
        CreateNodeOutcome::Tombstoned => Err(ServerError::Conflict(format!(
            "node for {ref_kind}:{ref_id} was previously dismissed under this parent"
        ))),
    }
}

/// PATCH /api/pages/{id}/map/nodes/{node_id}
pub async fn handle_patch_map_node(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path((page_id, node_id)): Path<(String, String)>,
    Json(req): Json<PatchMapNodeRequest>,
) -> Result<Json<NodeMutationResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    let base_revision = req.base_revision;
    let patch = NodePatch {
        label: req.label,
        pinned: req.pinned,
        status: req.status,
        rank: req.rank,
        parent_id: req.parent_id,
    };
    let node = db
        .patch_map_node(&page_id, base_revision, &node_id, patch)
        .await?;
    Ok(Json(NodeMutationResponse {
        revision: base_revision + 1,
        node: wire_node(&db, node).await?,
    }))
}

/// DELETE /api/pages/{id}/map/nodes/{node_id}
pub async fn handle_delete_map_node(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path((page_id, node_id)): Path<(String, String)>,
    Json(req): Json<DeleteMapNodeRequest>,
) -> Result<Json<NodeMutationResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    let node = db
        .delete_map_node(&page_id, req.base_revision, &node_id)
        .await?;
    Ok(Json(NodeMutationResponse {
        revision: req.base_revision + 1,
        node: wire_node(&db, node).await?,
    }))
}

/// POST /api/pages/{id}/map/edges
///
/// Unlike node creation, edges never need init-if-absent: an edge always
/// references existing node ids, which cannot exist unless the map was
/// already initialized (by an earlier node create) — so a genuinely
/// absent map surfaces its natural 404 here.
pub async fn handle_create_map_edge(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(page_id): Path<String>,
    Json(req): Json<CreateMapEdgeRequest>,
) -> Result<Json<EdgeMutationResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    let outcome = db
        .create_map_edge(
            &page_id,
            req.base_revision,
            &req.from_node,
            &req.to_node,
            &req.kind,
            req.label.as_deref(),
        )
        .await?;

    match outcome {
        CreateEdgeOutcome::Created(edge) => Ok(Json(EdgeMutationResponse {
            revision: req.base_revision + 1,
            edge: wire_edge(edge),
        })),
        CreateEdgeOutcome::Duplicate(edge) => Ok(Json(EdgeMutationResponse {
            revision: req.base_revision,
            edge: wire_edge(edge),
        })),
        CreateEdgeOutcome::Tombstoned => Err(ServerError::Conflict(format!(
            "edge {} -> {} ({}) was previously dismissed",
            req.from_node, req.to_node, req.kind
        ))),
    }
}

/// PATCH /api/pages/{id}/map/edges/{edge_id}
pub async fn handle_patch_map_edge(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path((page_id, edge_id)): Path<(String, String)>,
    Json(req): Json<PatchMapEdgeRequest>,
) -> Result<Json<EdgeMutationResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    let base_revision = req.base_revision;
    let patch = EdgePatch {
        label: req.label,
        status: req.status,
    };
    let edge = db
        .patch_map_edge(&page_id, base_revision, &edge_id, patch)
        .await?;
    Ok(Json(EdgeMutationResponse {
        revision: base_revision + 1,
        edge: wire_edge(edge),
    }))
}

/// DELETE /api/pages/{id}/map/edges/{edge_id}
pub async fn handle_delete_map_edge(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path((page_id, edge_id)): Path<(String, String)>,
    Json(req): Json<DeleteMapEdgeRequest>,
) -> Result<Json<EdgeMutationResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    let edge = db
        .delete_map_edge(&page_id, req.base_revision, &edge_id)
        .await?;
    Ok(Json(EdgeMutationResponse {
        revision: req.base_revision + 1,
        edge: wire_edge(edge),
    }))
}

/// DELETE /api/pages/{id}/map (reset)
pub async fn handle_reset_page_map(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(page_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    db.reset_page_map(&page_id).await?;
    Ok(Json(serde_json::json!({"status": "reset"})))
}

/// POST /api/pages/{id}/map/improve
///
/// Synchronous (mirrors `POST /api/steep`): runs the improve pass for this one
/// page inline, then returns the refreshed map. INSERT-ONLY — the pass appends
/// `status = 'suggested'` nodes/edges grounded in the page's real objects
/// (entity, markdown sections, source memories, resolved wikilinks) and never
/// touches an existing row; see `wenlan_core::page_map_improve`. This route is
/// NEVER gated by the `page_map_auto_suggest` config flag — that flag gates only
/// the background scheduler phase (`refinery::Phase::PageMaps`). Returns the
/// full `PageMapResponse` (excluding dismissed rows) so the client reuses the
/// same render path as GET; the freshly-suggested nodes surface with
/// `status = "suggested"`.
pub async fn handle_improve_page_map(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(page_id): Path<String>,
) -> Result<Json<PageMapResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    ensure_page_is_active(&db, &page_id).await?;

    wenlan_core::page_map_improve::improve_page_map(&db, &page_id).await?;

    // The pass always init's the map, so `get_page_map` returns Some here; the
    // None arm is defensive and mirrors GET's empty-map shape.
    let data = db.get_page_map(&page_id, false).await?;
    let response = match data {
        Some(data) => build_map_response(&db, &page_id, data).await?,
        None => PageMapResponse {
            page_id,
            revision: 0,
            map_schema: 1,
            viewport: None,
            nodes: vec![],
            edges: vec![],
        },
    };
    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn new_test_db() -> (Arc<MemoryDB>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = MemoryDB::new(tmp.path(), emitter).await.unwrap();
        (Arc::new(db), tmp)
    }

    async fn seed_test_page(db: &MemoryDB, title: &str) -> String {
        let source_id = format!("src-{title}");
        let source = wenlan_core::sources::RawDocument {
            source: "memory".to_string(),
            source_id: source_id.clone(),
            title: format!("memory-{source_id}"),
            content: format!("{title} body content for testing"),
            memory_type: Some("fact".to_string()),
            space: Some("default".to_string()),
            source_agent: Some("test-agent".to_string()),
            confidence: Some(0.9),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![source]).await.unwrap();
        let result = wenlan_core::post_write::create_page_with_tuning(
            db,
            wenlan_types::requests::CreateConceptRequest {
                title: title.to_string(),
                content: format!("{title} body content for testing"),
                summary: None,
                entity_id: None,
                source_memory_ids: vec![source_id],
                creation_kind: Some("distilled".to_string()),
                space: Some("default".to_string()).into(),
                workspace: Some("default".to_string()),
            },
            "test",
            None,
            1,
            1.1, // forces a new page rather than clustering into an existing one
        )
        .await
        .unwrap();
        result.id
    }

    async fn seed_page_with_content(db: &MemoryDB, title: &str, content: &str) -> String {
        let source_id = format!("src-{title}");
        let source = wenlan_core::sources::RawDocument {
            source: "memory".to_string(),
            source_id: source_id.clone(),
            title: format!("memory-{source_id}"),
            content: content.to_string(),
            memory_type: Some("fact".to_string()),
            space: Some("default".to_string()),
            source_agent: Some("test-agent".to_string()),
            confidence: Some(0.9),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![source]).await.unwrap();
        let result = wenlan_core::post_write::create_page_with_tuning(
            db,
            wenlan_types::requests::CreateConceptRequest {
                title: title.to_string(),
                content: content.to_string(),
                summary: None,
                entity_id: None,
                source_memory_ids: vec![source_id],
                creation_kind: Some("distilled".to_string()),
                space: Some("default".to_string()).into(),
                workspace: Some("default".to_string()),
            },
            "test",
            None,
            1,
            1.1, // forces a new page rather than clustering into an existing one
        )
        .await
        .unwrap();
        result.id
    }

    async fn seed_memory(db: &MemoryDB, source_id: &str) {
        let mem = wenlan_core::sources::RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: format!("memory-{source_id}"),
            content: "test memory content".to_string(),
            memory_type: Some("fact".to_string()),
            space: Some("default".to_string()),
            source_agent: Some("test-agent".to_string()),
            confidence: Some(0.9),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![mem]).await.unwrap();
    }

    fn state_with_db(db: Arc<MemoryDB>) -> Arc<RwLock<ServerState>> {
        Arc::new(RwLock::new(ServerState {
            db: Some(db),
            ..Default::default()
        }))
    }

    fn create_node_request(ref_id: &str, base_revision: i64) -> CreateMapNodeRequest {
        CreateMapNodeRequest {
            base_revision,
            parent_id: None,
            ref_kind: Some("memory".to_string()),
            ref_id: Some(ref_id.to_string()),
            label: None,
            rank: 0.0,
        }
    }

    #[tokio::test]
    async fn get_page_map_returns_empty_shape_for_absent_map() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Absent Map Page").await;
        let state = state_with_db(db);

        let Json(response) = handle_get_page_map(
            State(state),
            Path(page_id.clone()),
            Query(MapIncludeQuery { include: None }),
        )
        .await
        .expect("absent map should return 200, not 404");

        assert_eq!(response.page_id, page_id);
        assert_eq!(response.revision, 0);
        assert!(response.nodes.is_empty());
        assert!(response.edges.is_empty());
    }

    #[tokio::test]
    async fn get_page_map_returns_404_for_unknown_page() {
        let (db, _tmp) = new_test_db().await;
        let state = state_with_db(db);

        let result = handle_get_page_map(
            State(state),
            Path("nonexistent-page".to_string()),
            Query(MapIncludeQuery { include: None }),
        )
        .await;

        match result {
            Err(ServerError::NotFound(_)) => {}
            _ => panic!("expected NotFound for unknown page"),
        }
    }

    #[tokio::test]
    async fn create_map_node_rejects_missing_ref_kind_with_400() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Node 400 Page").await;
        let state = state_with_db(db);

        let result = handle_create_map_node(
            State(state),
            Path(page_id),
            Json(CreateMapNodeRequest {
                base_revision: 0,
                parent_id: None,
                ref_kind: None,
                ref_id: Some("mem-1".to_string()),
                label: None,
                rank: 0.0,
            }),
        )
        .await;

        match result {
            Err(ServerError::BadRequest(_)) => {}
            _ => panic!("expected BadRequest for missing ref_kind"),
        }
    }

    #[tokio::test]
    async fn create_map_node_returns_404_for_unknown_page() {
        let (db, _tmp) = new_test_db().await;
        let state = state_with_db(db);

        let result = handle_create_map_node(
            State(state),
            Path("nonexistent-page".to_string()),
            Json(create_node_request("mem-1", 0)),
        )
        .await;

        match result {
            Err(ServerError::NotFound(_)) => {}
            _ => panic!("expected NotFound for unknown page"),
        }
    }

    #[tokio::test]
    async fn create_map_node_rejects_non_active_pages_without_creating_map_rows() {
        let (db, _tmp) = new_test_db().await;
        let draft = db
            .create_page_draft("Draft", "Body", None, None)
            .await
            .unwrap();
        let archived_id = seed_test_page(&db, "Archived Map Page").await;
        db.archive_page(&archived_id).await.unwrap();
        seed_memory(&db, "mem-1").await;
        let state = state_with_db(db.clone());

        for page_id in [draft.id, archived_id] {
            let result = handle_create_map_node(
                State(state.clone()),
                Path(page_id.clone()),
                Json(create_node_request("mem-1", 0)),
            )
            .await;

            match result {
                Err(ServerError::ValidationError(_)) => {}
                _ => panic!("expected non-active Page to reject Page Map mutation"),
            }
            assert!(
                db.get_page_map(&page_id, true).await.unwrap().is_none(),
                "rejected mutation must not create derived Page Map rows"
            );
        }
    }

    #[tokio::test]
    async fn create_map_node_succeeds_as_first_mutation_against_absent_map() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "First Mutation Page").await;
        seed_memory(&db, "mem-1").await;
        let state = state_with_db(db);

        // The client would have seen revision 0 from a GET on the
        // never-initialized map (the "absent map" contract) and naturally
        // sends that back as base_revision.
        let Json(response) = handle_create_map_node(
            State(state),
            Path(page_id),
            Json(create_node_request("mem-1", 0)),
        )
        .await
        .expect("first node creation against an absent map should succeed, not conflict");

        // init_page_map bumps an absent map to revision 1, then the create bumps to 2.
        assert_eq!(response.revision, 2);
        assert_eq!(response.node.ref_kind, "memory");
        assert_eq!(response.node.ref_state, RefState::Live);
    }

    #[tokio::test]
    async fn create_map_node_duplicate_returns_200_with_existing_row() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Duplicate Page").await;
        seed_memory(&db, "mem-1").await;
        let state = state_with_db(db);

        let Json(first) = handle_create_map_node(
            State(state.clone()),
            Path(page_id.clone()),
            Json(create_node_request("mem-1", 0)),
        )
        .await
        .unwrap();

        let Json(second) = handle_create_map_node(
            State(state),
            Path(page_id),
            Json(create_node_request("mem-1", first.revision)),
        )
        .await
        .expect("duplicate proposal should succeed as 200, not error");

        assert_eq!(second.node.id, first.node.id);
        assert_eq!(
            second.revision, first.revision,
            "duplicate never bumps revision"
        );
    }

    #[tokio::test]
    async fn create_map_node_tombstoned_returns_409() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Tombstone Page").await;
        seed_memory(&db, "mem-1").await;
        let state = state_with_db(db);

        let Json(created) = handle_create_map_node(
            State(state.clone()),
            Path(page_id.clone()),
            Json(create_node_request("mem-1", 0)),
        )
        .await
        .unwrap();

        let Json(deleted) = handle_delete_map_node(
            State(state.clone()),
            Path((page_id.clone(), created.node.id.clone())),
            Json(DeleteMapNodeRequest {
                base_revision: created.revision,
            }),
        )
        .await
        .unwrap();

        let result = handle_create_map_node(
            State(state),
            Path(page_id),
            Json(create_node_request("mem-1", deleted.revision)),
        )
        .await;

        match result {
            Err(ServerError::Conflict(_)) => {}
            _ => panic!("expected Conflict (409) for a tombstoned fingerprint"),
        }
    }

    #[tokio::test]
    async fn patch_map_node_rejects_stale_base_revision_with_409() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Stale Revision Page").await;
        seed_memory(&db, "mem-1").await;
        let state = state_with_db(db);

        let Json(created) = handle_create_map_node(
            State(state.clone()),
            Path(page_id.clone()),
            Json(create_node_request("mem-1", 0)),
        )
        .await
        .unwrap();
        let stale_revision = created.revision;

        let Json(patched) = handle_patch_map_node(
            State(state.clone()),
            Path((page_id.clone(), created.node.id.clone())),
            Json(PatchMapNodeRequest {
                base_revision: stale_revision,
                label: None,
                pinned: None,
                status: None,
                rank: Some(5.0),
                parent_id: None,
            }),
        )
        .await
        .expect("patch with the correct base_revision should succeed");
        assert_eq!(
            patched.revision,
            stale_revision + 1,
            "base_revision round-trips to revision + 1"
        );

        let result = handle_patch_map_node(
            State(state),
            Path((page_id, created.node.id)),
            Json(PatchMapNodeRequest {
                base_revision: stale_revision,
                label: None,
                pinned: None,
                status: None,
                rank: Some(9.0),
                parent_id: None,
            }),
        )
        .await;

        match result {
            Err(ServerError::Conflict(_)) => {}
            _ => panic!("expected Conflict (409) for stale base_revision"),
        }
    }

    #[tokio::test]
    async fn delete_map_node_rejects_root_dismissal_with_422() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Root Protect Page").await;
        let (map, _created) = db.init_page_map(&page_id).await.unwrap();
        let root_id = map.nodes[0].id.clone();
        let base_revision = map.map.revision;
        let state = state_with_db(db);

        let result = handle_delete_map_node(
            State(state),
            Path((page_id, root_id)),
            Json(DeleteMapNodeRequest { base_revision }),
        )
        .await;

        match result {
            Err(ServerError::ValidationError(_)) => {}
            _ => panic!("expected ValidationError (422) for root dismissal"),
        }
    }

    #[tokio::test]
    async fn get_page_map_include_dismissed_toggles_visibility() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Include Dismissed Page").await;
        seed_memory(&db, "mem-1").await;
        let state = state_with_db(db);

        let Json(created) = handle_create_map_node(
            State(state.clone()),
            Path(page_id.clone()),
            Json(create_node_request("mem-1", 0)),
        )
        .await
        .unwrap();
        let _ = handle_delete_map_node(
            State(state.clone()),
            Path((page_id.clone(), created.node.id.clone())),
            Json(DeleteMapNodeRequest {
                base_revision: created.revision,
            }),
        )
        .await
        .unwrap();

        let Json(without) = handle_get_page_map(
            State(state.clone()),
            Path(page_id.clone()),
            Query(MapIncludeQuery { include: None }),
        )
        .await
        .unwrap();
        assert!(
            without.nodes.iter().all(|n| n.id != created.node.id),
            "dismissed node must be excluded by default"
        );

        let Json(with) = handle_get_page_map(
            State(state),
            Path(page_id),
            Query(MapIncludeQuery {
                include: Some("dismissed".to_string()),
            }),
        )
        .await
        .unwrap();
        assert!(
            with.nodes.iter().any(|n| n.id == created.node.id),
            "?include=dismissed must surface the dismissed node"
        );
    }

    #[tokio::test]
    async fn improve_page_map_suggests_and_bumps_revision() {
        let (db, _tmp) = new_test_db().await;
        // seed_test_page attaches one source memory (`src-{title}`), which the
        // improve pass turns into a suggested memory node.
        let page_id = seed_test_page(&db, "Improve Route Page").await;
        let state = state_with_db(db);

        let Json(response) = handle_improve_page_map(State(state), Path(page_id.clone()))
            .await
            .expect("improve should return 200 for an existing page");

        assert_eq!(response.page_id, page_id);
        // init_page_map lifts the absent-map sentinel (0) to revision 1, then
        // the suggested memory node bumps it again.
        assert!(
            response.revision >= 1,
            "improve must bump the revision off the absent-map sentinel, got {}",
            response.revision
        );
        assert!(
            response.nodes.iter().any(|n| n.status == "suggested"),
            "improve must surface at least one suggested node"
        );
    }

    #[tokio::test]
    async fn improve_page_map_404s_for_unknown_page() {
        let (db, _tmp) = new_test_db().await;
        let state = state_with_db(db);

        let result =
            handle_improve_page_map(State(state), Path("nonexistent-page".to_string())).await;

        match result {
            Err(ServerError::NotFound(_)) => {}
            _ => panic!("expected NotFound for unknown page"),
        }
    }

    // Regression guard: the explicit improve route must NEVER be gated by the
    // `page_map_auto_suggest` config flag (that flag gates only the background
    // `refinery::Phase::PageMaps` scheduler phase). We write a config with the
    // flag OFF into a temp data dir, then confirm the handler still produces
    // suggestions. Because the handler does not read config at all, this test
    // is race-safe under cargo's parallel test threads (a concurrently-set
    // `WENLAN_DATA_DIR` cannot change the handler's behavior); its teeth are on
    // a *future* regression that wrongly wires the flag into this route.
    #[tokio::test]
    async fn improve_page_map_runs_when_auto_suggest_disabled() {
        struct DataDirGuard {
            _tmp: tempfile::TempDir,
            previous: Option<std::ffi::OsString>,
        }
        impl DataDirGuard {
            fn new() -> Self {
                let tmp = tempfile::tempdir().unwrap();
                let previous = std::env::var_os("WENLAN_DATA_DIR");
                std::env::set_var("WENLAN_DATA_DIR", tmp.path());
                Self {
                    _tmp: tmp,
                    previous,
                }
            }
        }
        impl Drop for DataDirGuard {
            fn drop(&mut self) {
                match &self.previous {
                    Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                    None => std::env::remove_var("WENLAN_DATA_DIR"),
                }
            }
        }

        let _env = DataDirGuard::new();
        // Empty temp data dir → load_config returns defaults; set the flag OFF
        // explicitly and persist it, so a hypothetical future gating of this
        // route on the flag would read `false`.
        let mut cfg = wenlan_core::config::load_config();
        cfg.page_map_auto_suggest = false;
        wenlan_core::config::save_config(&cfg).unwrap();

        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Disabled Flag Page").await;
        let state = state_with_db(db);

        let Json(response) = handle_improve_page_map(State(state), Path(page_id))
            .await
            .expect("explicit improve must run even with auto_suggest disabled");
        assert!(
            response.nodes.iter().any(|n| n.status == "suggested"),
            "explicit improve route must suggest regardless of the config flag"
        );
    }

    #[tokio::test]
    async fn create_map_node_rejects_stale_zero_base_revision_once_map_exists() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_test_page(&db, "Existing Map Page").await;
        seed_memory(&db, "mem-1").await;

        // Simulate an actor that already initialized the map before this
        // request's own init call runs.
        let (_pre, created) = db.init_page_map(&page_id).await.unwrap();
        assert!(created);

        let state = state_with_db(db);
        let result = handle_create_map_node(
            State(state),
            Path(page_id),
            Json(create_node_request("mem-1", 0)),
        )
        .await;

        match result {
            Err(ServerError::Conflict(_)) => {}
            _ => panic!("expected 409 for stale base_revision=0 once the map already exists"),
        }
    }

    #[tokio::test]
    async fn compute_ref_state_section_dangles_when_heading_removed() {
        let (db, _tmp) = new_test_db().await;
        let page_id = seed_page_with_content(&db, "Section Removal Page", "# Overview\nbody").await;
        let slug = wenlan_core::export::obsidian::slugify("Overview");
        let ref_id = format!("{page_id}#{slug}");

        assert_eq!(
            compute_ref_state(&db, "section", &ref_id).await.unwrap(),
            RefState::Live
        );

        // Heading removed from the page's own content.
        db.update_page_content(&page_id, "no headings here anymore", &[], "test")
            .await
            .unwrap();

        assert_eq!(
            compute_ref_state(&db, "section", &ref_id).await.unwrap(),
            RefState::Dangling
        );
    }
}
