// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::memory_routes::extract_agent_name;
use crate::route_registry::{get, post, put, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    response::Json,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_types::requests::{
    CreateConceptRequest, CreatePageDraftRequest, ExportPagesRequest, PageDraftVersionRequest,
    SearchPagesRequest, UpdatePageDraftRequest,
};
use wenlan_types::responses::{CreatePageResponse, PageDraftResponse};
use wenlan_types::{WriteOutcome, WriteSpaceSource, WriteSpaceTarget};

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route(
            "/api/pages",
            get(handle_list_pages).post(handle_create_page),
        )
        .route("/api/pages/search", post(handle_search_pages))
        .route("/api/pages/drafts", post(handle_create_page_draft))
        .route(
            "/api/pages/drafts/{id}",
            put(handle_update_page_draft).delete(handle_discard_page_draft),
        )
        .route(
            "/api/pages/drafts/{id}/publish",
            post(handle_publish_page_draft),
        )
        .route("/api/pages/export", post(handle_export_pages))
        .route(
            "/api/pages/recent-changes",
            get(crate::routes::handle_recent_page_changes),
        )
        .route("/api/pages/orphan-links", get(handle_list_orphan_links))
        .route("/api/pages/{id}/export", post(handle_export_page))
        .route(
            "/api/pages/{id}",
            get(handle_get_page)
                .put(handle_refresh_page)
                .delete(handle_delete_page),
        )
        .route("/api/pages/{id}/sources", get(handle_get_page_sources))
        .route("/api/pages/{id}/links", get(handle_get_page_links))
        .route("/api/pages/{id}/archive", post(handle_archive_page))
        .route("/api/pages/{id}/revisions", get(handle_get_page_revisions))
        .route("/api/pages/{id}/review", post(handle_review_page))
}

pub(crate) fn register_manual_edit(
    router: TrackedRouter<SharedState>,
) -> TrackedRouter<SharedState> {
    router.route("/api/memory/{id}/update-page", post(handle_update_page))
}

/// GET /api/pages
pub async fn handle_list_pages(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let status = params.get("status").map(|s| s.as_str()).unwrap_or("active");
    let space = params
        .get("space")
        .or_else(|| params.get("domain"))
        .cloned();
    let limit: usize = params
        .get("limit")
        .and_then(|l| l.parse().ok())
        .unwrap_or(50);
    let offset: usize = params
        .get("offset")
        .and_then(|o| o.parse().ok())
        .unwrap_or(0);

    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope =
        crate::read_scope::effective_read_scope(&db, space.as_deref(), header_space.as_deref())
            .await?;
    // Q1 browse surface: `kind='entity'` dual-write shadow pages appear here,
    // so read the unfenced `_browse` twin (export + internal callers keep the
    // fenced `list_pages_scoped`).
    let pages = db
        .list_pages_scoped_browse(status, limit as i64, offset as i64, &scope)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    let pages = wenlan_core::truth_adapter::filter_pages(&db, &view.grant, pages)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({ "pages": pages })))
}

/// GET /api/pages/:id
pub async fn handle_get_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    // Q1 browse surface: a shadow id resolves 200 here, so read the unfenced
    // `_browse` twin (mutation/export by-id reads keep fenced `get_page_scoped`).
    let page = db
        .get_page_scoped_browse(&id, &scope)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    match wenlan_core::truth_adapter::filter_page(&db, &view.grant, page)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?
    {
        Some(page) => Ok(Json(serde_json::json!({ "page": page }))),
        // A hidden page 404s rather than 200-ing empty: for this caller it is
        // not there, and the existing not-found answer is the honest one.
        None => Err(ServerError::NotFound("page not found".to_string())),
    }
}

/// GET /api/pages/{id}/sources
///
/// Returns all source memories linked to a concept via the concept_sources join table,
/// enriched with memory metadata for display.
pub async fn handle_get_page_sources(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    Path(id): Path<String>,
) -> Result<Json<Vec<wenlan_types::PageSourceWithMemory>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let sources = db.get_page_sources_scoped(&id, &scope).await?;
    // The provenance trail hangs off the page rather than being it, so it has
    // no entry form: `reduce_to_entry` strips `source_memory_ids` precisely
    // because handing them out turns a listing into a retrieval plan, and this
    // route hands out the memories themselves on top. Full or nothing.
    let sources =
        wenlan_core::truth_adapter::filter_page_refs(&db, &view.grant, sources, |source| {
            source.page_id.as_str()
        })
        .await?;

    let source_id_strings: Vec<String> =
        sources.iter().map(|s| s.memory_source_id.clone()).collect();
    let memories = match &scope {
        wenlan_core::read_scope::ReadScope::Global => {
            db.get_memories_by_source_ids(&source_id_strings).await
        }
        _ => {
            db.get_memories_by_source_ids_scoped(&source_id_strings, &scope)
                .await
        }
    }
    .map_err(|e| ServerError::SearchFailed(e.to_string()))?;

    let result: Vec<wenlan_types::PageSourceWithMemory> = sources
        .iter()
        .map(|s| {
            let memory = memories
                .iter()
                .find(|m| m.source_id == s.memory_source_id)
                .cloned();
            wenlan_types::PageSourceWithMemory {
                source: s.clone(),
                memory,
            }
        })
        .collect();

    Ok(Json(result))
}

/// POST /api/pages/{id}/archive
pub async fn handle_archive_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // `ServerError::from` maps the entity-shadow fence's WenlanError::Validation
    // to a 4xx (ValidationError), matching the neighboring page-write handlers.
    db.archive_page(&id).await.map_err(ServerError::from)?;
    Ok(Json(serde_json::json!({"status": "archived"})))
}

/// DELETE /api/pages/{id}
///
/// Removes both the DB row and the projected `~/.wenlan/pages/<slug>.md`
/// file. DB-first so a transient md removal failure leaves a stale file
/// (cheap to clean up) rather than a stranded DB row (invisible to the
/// user). The md side failing is logged but not surfaced as an error —
/// the caller's intent (delete the page) succeeded as far as queries
/// are concerned.
pub async fn handle_delete_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref()
            .cloned()
            .ok_or(ServerError::DbNotInitialized)?
    };

    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    let projection =
        wenlan_core::export::knowledge::KnowledgeProjectionWrite::new(knowledge_path, &db);
    // `ServerError::from` maps the entity-shadow fence's WenlanError::Validation
    // to a 4xx (ValidationError), matching the neighboring page-write handlers.
    db.delete_page(&id).await.map_err(ServerError::from)?;

    if let Err(e) = projection.remove_page(&id) {
        tracing::warn!(
            "[page] DB row deleted but md projection cleanup failed for {}: {}",
            id,
            e
        );
    }

    Ok(Json(serde_json::json!({"status": "deleted"})))
}

/// POST /api/pages/search
pub async fn handle_search_pages(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    Json(req): Json<SearchPagesRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope =
        crate::read_scope::effective_read_scope(&db, req.space.as_deref(), header_space.as_deref())
            .await?;
    let results = db
        .search_pages_scoped_browse(
            &req.query,
            req.limit.unwrap_or(20),
            req.page_type.as_deref(),
            &scope,
        )
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    let results = wenlan_core::truth_adapter::filter_pages(&db, &view.grant, results)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({ "pages": results })))
}

/// POST /api/pages
///
/// Atomic md-first + DB-index. The `~/.wenlan/pages/<slug>.md` file is the
/// human-readable canonical form; the DB row is the hybrid index over it.
/// If the DB insert fails after the md write succeeds, the md file is
/// removed so the two stores stay consistent.
pub async fn handle_create_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(mut req): Json<CreateConceptRequest>,
) -> Result<Json<CreatePageResponse>, ServerError> {
    let agent = extract_agent_name(&headers, None);
    let (db, page_root, page_min_cluster_size, page_match_threshold) = {
        let s = state.read().await;
        (
            s.db.as_ref()
                .cloned()
                .ok_or(ServerError::DbNotInitialized)?,
            s.lint_config.page_root().map(std::path::Path::to_path_buf),
            s.tuning.distillation.page_min_cluster_size,
            s.tuning.distillation.page_match_threshold,
        )
    };

    let _space_write_guard = db.lock_space_writes().await;
    let target = normalize_page_write_target(&req.space, req.workspace.as_deref())?;
    let resolved = db
        .resolve_write_space(&target, header_space.as_deref())
        .await?;
    req.space = match resolved.space_name.as_ref() {
        Some(name) => WriteSpaceTarget::Named(name.clone()),
        None => WriteSpaceTarget::Uncategorized,
    };
    req.workspace = resolved.space_name.clone();
    let result = wenlan_core::post_write::create_page_with_tuning(
        &db,
        req,
        &agent,
        page_root.as_deref(),
        page_min_cluster_size,
        page_match_threshold,
    )
    .await?;
    let persisted_page_id = result.attached_to.as_deref().unwrap_or(&result.id);
    let persisted_space = db
        .get_page(persisted_page_id)
        .await?
        .ok_or_else(|| ServerError::Internal("Page disappeared after create".into()))?
        .space;
    let attached_existing = result.attached_to.is_some();
    Ok(Json(CreatePageResponse {
        id: result.id,
        attached_to: result.attached_to,
        warnings: result.warnings,
        space: persisted_space.clone(),
        space_source: Some(if attached_existing {
            WriteSpaceSource::Existing
        } else if persisted_space.is_some() {
            resolved.source
        } else {
            WriteSpaceSource::Uncategorized
        }),
        write_outcome: Some(if attached_existing {
            WriteOutcome::AttachedExisting
        } else {
            WriteOutcome::Created
        }),
    }))
}

/// Map core draft errors onto the editor's wire contract.
///
/// The desktop editor (`src/lib/tauri.ts` `parsePageDraftError`) parses a JSON
/// object with a `code` field out of the error text; the codes and messages
/// here mirror `e2e/tauriMock/runtime.ts`, the executable spec the UI is
/// tested against. Anything uncoded (validation, internal) falls through to
/// the plain `ServerError` mapping.
fn page_draft_error(err: wenlan_core::WenlanError) -> ServerError {
    match err {
        wenlan_core::WenlanError::PageDraftIdConflict(_) => ServerError::Structured {
            status: axum::http::StatusCode::CONFLICT,
            body: serde_json::json!({
                "code": "page_draft_id_conflict",
                "error": "Page draft id already belongs to another Page",
            }),
        },
        wenlan_core::WenlanError::NotFound(_) => draft_not_found(),
        other => ServerError::from(other),
    }
}

fn draft_not_found() -> ServerError {
    ServerError::Structured {
        status: axum::http::StatusCode::NOT_FOUND,
        body: serde_json::json!({
            "code": "page_draft_not_found",
            "error": "Page draft not found",
        }),
    }
}

/// Pass a draft-chain page echo through the caller's truth grant, the same
/// contract as `handle_get_page`: a page this caller may not see is not there,
/// and the structured draft 404 is the honest wire answer. Fresh drafts and
/// publishes are `Unevaluated` and pass in full; this bites only when a
/// publish replay echoes a page since classified unsupported.
async fn filter_draft_echo(
    db: &wenlan_core::db::MemoryDB,
    view: &crate::truth_guard::TruthView,
    page: wenlan_core::pages::Page,
) -> Result<wenlan_core::pages::Page, ServerError> {
    wenlan_core::truth_adapter::filter_page(db, &view.grant, Some(page))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(draft_not_found)
}

/// Decide whether a publish title-conflict 409 may disclose the conflicting
/// page's identity, and if so, which identity. Page ids are reusable, so the
/// pair captured inside the publish transaction can be stale by the time the
/// handler runs. `MemoryDB::conflict_identity_snapshot` makes the row reload,
/// the truth verdict, and the title-fold recheck under one connection guard,
/// so no concurrent rename, delete, or id reuse can split the visibility
/// decision from the disclosed identity. Any failure degrades to omission —
/// the 409 itself is already established and never turns into a 500 here.
async fn visible_conflict_identity(
    db: &wenlan_core::db::MemoryDB,
    view: &crate::truth_guard::TruthView,
    existing_page_id: &str,
    wanted_key: &str,
    expected_scope: &str,
) -> Option<(String, String)> {
    match db
        .conflict_identity_snapshot(&view.grant, existing_page_id, wanted_key, expected_scope)
        .await
    {
        Ok(identity) => identity,
        Err(e) => {
            tracing::warn!(
                "[page] conflict identity check failed for {existing_page_id}; omitting: {e}"
            );
            None
        }
    }
}

fn draft_version_conflict(current_version: i64) -> ServerError {
    ServerError::Structured {
        status: axum::http::StatusCode::CONFLICT,
        body: serde_json::json!({
            "code": "draft_version_conflict",
            "error": "Page draft changed since it was loaded",
            "current_version": current_version,
        }),
    }
}

async fn draft_db(
    state: &Arc<RwLock<ServerState>>,
) -> Result<Arc<wenlan_core::db::MemoryDB>, ServerError> {
    let s = state.read().await;
    s.db.clone().ok_or(ServerError::DbNotInitialized)
}

/// POST /api/pages/drafts
///
/// First durable snapshot of a human-authored Page draft. An omitted `space`
/// key inherits the `X-Wenlan-Space` header; an explicit `null` clears it.
pub async fn handle_create_page_draft(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    Json(req): Json<CreatePageDraftRequest>,
) -> Result<Json<PageDraftResponse>, ServerError> {
    let db = draft_db(&state).await?;
    let space = if req.space_was_provided() {
        req.space.clone()
    } else {
        header_space
    };
    let page = db
        .create_page_draft_with_id_in_registered_space(
            &req.draft_id,
            &req.title,
            &req.content,
            space.as_deref(),
        )
        .await
        .map_err(page_draft_error)?;
    let page = filter_draft_echo(&db, &view, page).await?;
    Ok(Json(PageDraftResponse { page }))
}

/// PUT /api/pages/drafts/{id}
pub async fn handle_update_page_draft(
    State(state): State<Arc<RwLock<ServerState>>>,
    view: crate::truth_guard::TruthView,
    Path(id): Path<String>,
    Json(req): Json<UpdatePageDraftRequest>,
) -> Result<Json<PageDraftResponse>, ServerError> {
    let db = draft_db(&state).await?;
    match db
        .update_page_draft_in_registered_space(
            &id,
            req.expected_version,
            &req.title,
            &req.content,
            req.space.as_deref(),
        )
        .await
        .map_err(page_draft_error)?
    {
        wenlan_core::pages::PageDraftUpdateOutcome::Updated(page) => {
            let page = filter_draft_echo(&db, &view, page).await?;
            Ok(Json(PageDraftResponse { page }))
        }
        wenlan_core::pages::PageDraftUpdateOutcome::VersionConflict { current_version } => {
            Err(draft_version_conflict(current_version))
        }
    }
}

/// POST /api/pages/drafts/{id}/publish
pub async fn handle_publish_page_draft(
    State(state): State<Arc<RwLock<ServerState>>>,
    view: crate::truth_guard::TruthView,
    Path(id): Path<String>,
    Json(req): Json<PageDraftVersionRequest>,
) -> Result<Json<PageDraftResponse>, ServerError> {
    let (db, page_root) = {
        let s = state.read().await;
        (
            s.db.clone().ok_or(ServerError::DbNotInitialized)?,
            s.lint_config.page_root().map(std::path::Path::to_path_buf),
        )
    };
    let page = match db
        .publish_page_draft(&id, req.expected_version)
        .await
        .map_err(page_draft_error)?
    {
        wenlan_core::pages::PageDraftPublishOutcome::Published(page) => page,
        wenlan_core::pages::PageDraftPublishOutcome::VersionConflict { current_version } => {
            return Err(draft_version_conflict(current_version))
        }
        wenlan_core::pages::PageDraftPublishOutcome::TitleConflict {
            existing_page_id,
            existing_page_title,
            scope,
        } => {
            let disclosed = visible_conflict_identity(
                &db,
                &view,
                &existing_page_id,
                &wenlan_core::db::MemoryDB::page_title_key(&existing_page_title),
                &scope,
            )
            .await;
            let mut body = serde_json::json!({
                "code": "page_title_conflict",
                "error": "A Page with this title already exists",
            });
            if let Some((current_id, current_title)) = disclosed {
                body["existing_page_id"] = current_id.into();
                body["existing_page_title"] = current_title.into();
            }
            return Err(ServerError::Structured {
                status: axum::http::StatusCode::CONFLICT,
                body,
            });
        }
    };
    let page = filter_draft_echo(&db, &view, page).await?;
    // First projection write for this page: `reconcile` only repairs pages
    // already in the projection state, so skipping here would leave the
    // published page invisible to `wenlan pages` until another write projects
    // it. Same root as `handle_create_page` (None disables projection);
    // DB-first with a warn on projection failure mirrors `handle_delete_page`:
    // the publish must not be reported as failed once the row flipped.
    if let Some(page_root) = page_root {
        let projection =
            wenlan_core::export::knowledge::KnowledgeProjectionWrite::new(page_root, &db);
        if let Err(e) = projection.write_page_gated(&db, &page).await {
            tracing::warn!(
                "[page] draft {} published but md projection write failed: {}",
                page.id,
                e
            );
        }
    }
    Ok(Json(PageDraftResponse { page }))
}

/// DELETE /api/pages/drafts/{id}
///
/// Drafts are never projected to md, so unlike `handle_delete_page` there is
/// no projection cleanup here.
pub async fn handle_discard_page_draft(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<PageDraftVersionRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = draft_db(&state).await?;
    match db
        .delete_page_draft(&id, req.expected_version)
        .await
        .map_err(page_draft_error)?
    {
        wenlan_core::pages::PageDraftDeleteOutcome::Deleted => {
            Ok(Json(serde_json::json!({"status": "deleted"})))
        }
        wenlan_core::pages::PageDraftDeleteOutcome::VersionConflict { current_version } => {
            Err(draft_version_conflict(current_version))
        }
    }
}

fn normalize_page_write_target(
    space: &WriteSpaceTarget,
    workspace: Option<&str>,
) -> Result<WriteSpaceTarget, ServerError> {
    let workspace = workspace.map(str::trim).filter(|value| !value.is_empty());
    match (space, workspace) {
        (WriteSpaceTarget::Inherit, Some(workspace)) => {
            Ok(WriteSpaceTarget::Named(workspace.to_string()))
        }
        (WriteSpaceTarget::Named(space), Some(workspace)) if space == workspace => {
            Ok(WriteSpaceTarget::Named(space.clone()))
        }
        (WriteSpaceTarget::Named(space), Some(workspace)) => Err(ServerError::ValidationError(
            format!("page space aliases conflict: space='{space}' and workspace='{workspace}'"),
        )),
        (WriteSpaceTarget::Uncategorized, Some(workspace)) => {
            Err(ServerError::ValidationError(format!(
                "page space aliases conflict: explicit Uncategorized and workspace='{workspace}'"
            )))
        }
        (target, None) => Ok(target.clone()),
    }
}

/// POST /api/pages/export
pub async fn handle_export_pages(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(req): Json<ExportPagesRequest>,
) -> Result<Json<wenlan_types::ExportStats>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    // Export must stay stub-free: `kind='entity'` dual-write shadow pages carry
    // no content and are not meant to leave the daemon. The fenced
    // `list_pages_scoped` excludes them in SQL *before* LIMIT, so a burst of
    // fresh stubs can never crowd real pages out of the 1000-row window (the
    // browse surfaces read the `_browse` twin instead).
    let pages = db
        .list_pages_scoped("active", 1000, 0, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    // A write, not a read: this puts page prose into a vault the caller names,
    // and whoever opens that vault later declared no contract and made no
    // gesture -- so the write has to satisfy the automatic reader, which is
    // what `page_write_permit` asks. The projection writer takes its permit
    // through `write_page_gated`; this exporter has no such twin, so the permit
    // is taken here. A declined page is skipped rather than failing the export:
    // refusing to export an unsupported page is the contract working.
    let mut exportable = Vec::with_capacity(pages.len());
    let mut declined = 0usize;
    for page in pages {
        if wenlan_core::truth_adapter::page_write_permit(&db, &page.id)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
            .is_some()
        {
            exportable.push(page);
        } else {
            declined += 1;
        }
    }
    if declined > 0 {
        tracing::info!("[export] skipped {declined} page(s) the automatic reader may not see");
    }
    let vault_path = req
        .vault_path
        .unwrap_or_else(|| "~/obsidian-vault/Wenlan/pages".to_string());
    let expanded = if let Some(rest) = vault_path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}/{}", home, rest)
    } else {
        vault_path
    };
    let exporter =
        wenlan_core::export::obsidian::ObsidianExporter::new(std::path::PathBuf::from(expanded));
    let stats = exporter
        .export_all(&exportable)
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(stats))
}

/// POST /api/pages/{id}/export
pub async fn handle_export_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(page_id): Path<String>,
    Json(req): Json<wenlan_types::requests::ExportPageRequest>,
) -> Result<Json<wenlan_types::responses::ExportPageResponse>, ServerError> {
    // Clone Arc out of the guard so we don't hold the RwLock read guard
    // across the DB await. (AGENTS.md "Repository invariants": never hold
    // tokio::sync::RwLock guards across .await.)
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    // Export must stay stub-free (see handle_export_pages). The fenced
    // `get_page_scoped` resolves a `kind='entity'` shadow id as absent, so a
    // stub 404s here just as it did before the Q1 lift.
    let page = db
        .get_page_scoped(&page_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("page not found".to_string()))?;

    // Same permit as the bulk export, and for the same reason: the vault this
    // writes into is read later by someone who declared no contract. A page the
    // automatic reader may not see is not there for that reader, so the route's
    // existing not-found answer is the honest one.
    if wenlan_core::truth_adapter::page_write_permit(&db, &page.id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .is_none()
    {
        return Err(ServerError::NotFound("page not found".to_string()));
    }

    let expanded = if let Some(rest) = req.vault_path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}/{}", home, rest)
    } else {
        req.vault_path
    };

    let exporter =
        wenlan_core::export::obsidian::ObsidianExporter::new(std::path::PathBuf::from(expanded));
    let result = exporter
        .export(&page)
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(wenlan_types::responses::ExportPageResponse {
        path: result.path,
    }))
}

/// POST /api/memory/{id}/update-page
/// GET /api/pages/{id}/links
///
/// Returns the wikilink graph centered on a page: outbound (labels parsed
/// out of this page's body, with resolved target_page_id when matched) and
/// inbound (every active page whose body links to this title). Used by
/// `/pages` to surface "3 inbound, 2 broken" without the caller having to
/// fetch + parse the full body.
pub async fn handle_get_page_links(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::PageLinksResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    // The whole graph hangs off page `id`: outbound labels are text lifted out
    // of ITS body, and inbound rows exist only because they point at it. A
    // caller who may not see the page in full gets neither -- which is what
    // this route's `NamedPage` marker shape is for. Empty rather than 404,
    // matching the answer an absent page already gets here.
    let named_visible = !wenlan_core::truth_adapter::filter_page_refs(
        &db,
        &view.grant,
        vec![id.clone()],
        |page_id| page_id.as_str(),
    )
    .await?
    .is_empty();
    if !named_visible {
        return Ok(Json(wenlan_types::responses::PageLinksResponse {
            outbound: Vec::new(),
            inbound: Vec::new(),
        }));
    }
    let outbound_raw = db.get_page_outbound_links_scoped(&id, &scope).await?;
    let inbound_raw = db.get_page_inbound_links_scoped(&id, &scope).await?;
    // An inbound row names ANOTHER page and carries the label it links with; a
    // grant for `id` does not cover page B's title in A's links.
    let inbound_raw =
        wenlan_core::truth_adapter::filter_page_refs(&db, &view.grant, inbound_raw, |row| {
            row.0.as_str()
        })
        .await?;
    // Outbound rows are this page's own link text, so they stay -- but a
    // resolved target names another page. A target the caller may not see reads
    // as unresolved, the same answer a dangling link already gets, rather than
    // disclosing that the page exists.
    let resolved_targets: Vec<String> = outbound_raw
        .iter()
        .filter_map(|l| l.target_page_id.clone())
        .collect();
    let visible_targets: std::collections::HashSet<String> =
        wenlan_core::truth_adapter::filter_page_refs(
            &db,
            &view.grant,
            resolved_targets,
            |target| target.as_str(),
        )
        .await?
        .into_iter()
        .collect();
    let outbound = outbound_raw
        .into_iter()
        .map(|l| wenlan_types::responses::PageLinkOutbound {
            label: l.label,
            target_page_id: l.target_page_id.filter(|t| visible_targets.contains(t)),
        })
        .collect();
    let inbound = inbound_raw
        .into_iter()
        .map(|(src, label)| wenlan_types::responses::PageLinkInbound {
            source_page_id: src,
            label,
        })
        .collect();
    Ok(Json(wenlan_types::responses::PageLinksResponse {
        outbound,
        inbound,
    }))
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct OrphanLinksQuery {
    /// Minimum number of distinct source pages reaching for the same label.
    /// Default 2 — single-source orphans are usually typos, not emergence
    /// signal.
    #[serde(default)]
    pub min_count: Option<i64>,
}

/// GET /api/pages/orphan-links
///
/// Group every unresolved wikilink label across the graph by hit count. A
/// label reached for by N independent pages is a topic-discovery signal:
/// emergence can prioritize building a page on it. Capped at 100 rows;
/// callers needing more should raise min_count and re-issue.
pub async fn handle_list_orphan_links(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    axum::extract::Query(q): axum::extract::Query<OrphanLinksQuery>,
) -> Result<Json<wenlan_types::responses::OrphanLinksResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let min_count = q.min_count.unwrap_or(2).max(1);
    // The page an orphan label leaks is the SOURCE, not the target it fails to
    // resolve to: the label is `[[wikilink]]` text lifted out of some page's
    // body, so a caller who may not read that body may not read the text either.
    // Label prose has no entry form, hence `filter_page_refs` -- `Full` or gone.
    //
    // Filtering has to happen on ROWS, which is why this reads the row query
    // rather than the aggregate it replaced: `list_orphan_link_labels_scoped`
    // does the grouping, the `HAVING n >= ?` threshold and the `LIMIT 100`
    // inside SQL, so post-filtering its output would report counts and a
    // truncation point computed over pages this caller cannot see.
    // `fold_orphan_labels` is that same aggregate, applied to what survives.
    let rows = db
        .list_orphan_link_rows_scoped(&scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    let rows = wenlan_core::truth_adapter::filter_page_refs(&db, &view.grant, rows, |row| {
        row.source_page_id.as_str()
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let orphan_labels = wenlan_core::db::MemoryDB::fold_orphan_labels(rows, min_count)
        .into_iter()
        .map(|(label, count)| wenlan_types::responses::OrphanLink { label, count })
        .collect();
    Ok(Json(wenlan_types::responses::OrphanLinksResponse {
        min_count: min_count as usize,
        orphan_labels,
    }))
}

/// POST /api/memory/{id}/update-page
///
/// Manual edit from the app. Goes through the one page-write gate, so a hand
/// edit is treated like every other write: version CAS, changelog entry, and
/// history row. This route does not re-project Markdown; the comment below
/// documents that separate synchronization boundary.
///
/// `expected_version` is optional. Sending it makes the edit a precondition
/// (refused outright if the page moved); omitting it guards on the version the
/// server itself loaded, so the write still cannot clobber an interleaved one.
pub async fn handle_update_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::UpdatePageRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // Preserve the route's 404 shape, but do not copy sources from this
    // advisory read. PageWrite derives them again from the exact generation
    // its CAS updates, so a source attached in between cannot be deleted.
    db.get_page(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound(format!("page {id} not found")))?;

    // knowledge_path=None: this route does not re-project the md, so an
    // in-app edit leaves the vault copy stale until the next re-distill (the
    // watcher sees the daemon ahead and skips rather than reverting, so the
    // DB stays authoritative and nothing is lost). Projecting here needs the
    // knowledge path to come from ServerState — reading global config inside
    // a handler would point tests at the user's real vault. Tracked separately.
    let result = wenlan_core::post_write::update_page_preserving_sources(
        &db,
        &id,
        wenlan_types::requests::UpdatePageRequest {
            content: req.content,
            source_memory_ids: vec![],
            expected_version: req.expected_version,
            caller_id: req.caller_id,
            operation_id: req.operation_id,
        },
        "manual_edit",
        None,
    )
    .await?;

    // A refused write is a conflict, not a success — reporting ok:true would
    // tell the editor its text was saved when the page still holds somebody
    // else's. But "nothing changed" is not a conflict, and this route sends the
    // page's own sources back, so saving a page without editing it lands on the
    // no-change path. Branching on `wrote` alone answered that routine save with
    // "somebody else edited this," which is both wrong and unactionable.
    use wenlan_core::post_write::WriteOutcome;
    match result.outcome {
        WriteOutcome::Wrote | WriteOutcome::Unchanged => {
            Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
        }
        WriteOutcome::Refused => Err(ServerError::Conflict(format!(
            "page {id} changed while this edit was open; reload and reapply"
        ))),
        WriteOutcome::Contended => Err(ServerError::Conflict(format!(
            "page {id} is being written too fast to edit safely; try again"
        ))),
        // Unreachable: gating is for machine writers, and this route is
        // `manual_edit`. Answered honestly rather than folded into a catch-all.
        WriteOutcome::Gated => Err(ServerError::Conflict(format!(
            "edit to page {id} was staged for review instead of applied"
        ))),
    }
}

/// PUT /api/pages/{id}
///
/// Agent-side refresh of a page from its current sources. Replaces content,
/// source list, optional summary; clears `stale_reason` so the refinery's
/// re-distill skips on its next tick (CAS pattern — see refinery
/// `re_distill_stale_pages`). Distinct from POST `/api/memory/{id}/update-page`
/// (manual edit, flips `user_edited`, preserves sources).
///
/// Atomicity mirrors `handle_create_page`: write md first, persist DB index
/// second, roll back md on DB failure. Old md content is held in memory
/// for the rollback path so a failed DB update doesn't leave a stale .md
/// without a matching row.
pub async fn handle_refresh_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::RefreshPageRequest>,
) -> Result<Json<wenlan_types::responses::PageWriteResponse>, ServerError> {
    // Validate the body before touching the filesystem. Empty content would
    // produce an empty md; empty source list would orphan the page from its
    // provenance trail — both contradict the route's documented contract.
    if req.content.trim().is_empty() {
        return Err(ServerError::ValidationError(
            "content must not be empty".into(),
        ));
    }
    if req.source_memory_ids.is_empty() {
        return Err(ServerError::ValidationError(
            "source_memory_ids must not be empty — refresh keeps the page \
             linked to its sources"
                .into(),
        ));
    }
    wenlan_core::export::provenance::validate_canonical_page_content(&req.content)
        .map_err(ServerError::from)?;

    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let existing = db
        .get_page(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::ValidationError(format!("page {} not found", id)))?;

    // Ownership gate (spec §5.1/§5.2): agent refresh is a machine write, so a
    // human-owned page (user_edited=1 OR creation_kind='authored') must never be
    // overwritten in place. Stage a pending revision card instead and return the
    // gating flag; the page prose stays byte-unchanged until the human accepts.
    if wenlan_core::post_write::page_is_human_owned(&existing) {
        let result = wenlan_core::post_write::stage_page_revision_card(
            &db,
            &existing,
            &req.content,
            &req.source_memory_ids,
            "agent_refresh",
            None,
        )
        .await
        .map_err(ServerError::from)?;
        return Ok(Json(wenlan_types::responses::PageWriteResponse {
            ok: true,
            revision_card_id: result.revision_card_id,
            gated: true,
        }));
    }

    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    let writer = wenlan_core::export::knowledge::KnowledgeWriter::new(knowledge_path.clone(), &db);

    // Snapshot the current md content for rollback. If the file is missing
    // we tolerate it — the page may have been created before the projection
    // existed; the rollback then becomes a remove_page.
    let existing_state_file = writer.page_filename(&id);
    let existing_md_content = existing_state_file
        .as_ref()
        .and_then(|f| std::fs::read_to_string(knowledge_path.join(f)).ok());
    let projection = writer.begin_projection_write();

    // Build the refreshed Page for md rendering. Bump version + last_modified
    // mirror what `update_page_content` writes to the DB row.
    //
    // Summary semantics: `None` keeps the existing summary. `Some(s)` where
    // `s` is non-empty replaces it. `Some("")` clears it — empty string maps
    // to NULL in the DB so `IS NULL` filters work (per wenlan-core's NULL
    // semantics rule). The MCP tool description documents this; normalize
    // here so the daemon never stores a literal empty string.
    let now = chrono::Utc::now().to_rfc3339();
    let summary_update: Option<Option<String>> = match &req.summary {
        Some(s) if s.is_empty() => Some(None),
        Some(s) => Some(Some(s.clone())),
        None => None,
    };
    let refreshed_summary = match &summary_update {
        Some(opt) => opt.clone(),
        None => existing.summary.clone(),
    };
    let refreshed_page = wenlan_core::pages::Page {
        id: existing.id.clone(),
        title: existing.title.clone(),
        summary: refreshed_summary.clone(),
        content: req.content.clone(),
        entity_id: existing.entity_id.clone(),
        space: existing.space.clone(),
        source_memory_ids: req.source_memory_ids.clone(),
        version: existing.version + 1,
        status: existing.status.clone(),
        created_at: existing.created_at.clone(),
        last_compiled: now.clone(),
        last_modified: now.clone(),
        sources_updated_count: 0,
        stale_reason: None,
        pending_rebuild: None,
        user_edited: existing.user_edited,
        relevance_score: 0.0,
        last_edited_by: None,
        last_edited_at: None,
        last_delta_summary: None,
        changelog: None,
        creation_kind: existing.creation_kind.clone(),
        review_status: existing.review_status.clone(),
        workspace: existing.workspace.clone(),
        // Content update without fresh citations resets citations to []
        // (Global Constraints: stale claim-maps must not survive a content edit).
        citations: Vec::new(),
        kind: existing.kind.clone(),
        truth: None,
    };

    // 1. md-first
    projection
        .write_page_gated(&db, &refreshed_page)
        .await
        .map_err(|e| ServerError::IngestFailed(format!("write_page: {}", e)))?;

    use wenlan_core::post_write::WriteOutcome;
    let db_result: Result<wenlan_core::post_write::WriteResult, wenlan_core::error::WenlanError> =
        async {
            let result = wenlan_core::post_write::update_page(
                &db,
                &id,
                wenlan_types::requests::UpdatePageRequest {
                    content: req.content.clone(),
                    source_memory_ids: req.source_memory_ids.clone(),
                    expected_version: None,
                    caller_id: None,
                    operation_id: None,
                },
                "agent_refresh",
                false,
                None,
                None,
            )
            .await?;
            // Carry the rest of the refresh only when the DB agrees with what the
            // agent asked for. A Gated/Refused/Contended outcome means a human owns
            // this prose right now — rewriting its summary or clearing its staleness
            // would apply half of a refresh the gate just declined.
            if matches!(
                result.outcome,
                WriteOutcome::Wrote | WriteOutcome::Unchanged
            ) {
                if let Some(opt) = &summary_update {
                    db.update_page_summary(&id, opt.as_deref()).await?;
                }
                db.clear_page_staleness(&id).await?;
            }
            Ok(result)
        }
        .await;

    // Roll back md to the snapshotted content so the two stores stay consistent.
    // If the file existed before and we have its bytes, rewrite them; otherwise
    // drop the projection.
    let roll_back_md = || match (&existing_state_file, &existing_md_content) {
        (Some(filename), Some(prev)) => {
            std::fs::write(knowledge_path.join(filename), prev).map_err(|io| io.to_string())
        }
        _ => projection.remove_page(&id).map_err(|err| err.to_string()),
    };

    let result = match db_result {
        Ok(result) => result,
        Err(e) => {
            if let Err(rb) = roll_back_md() {
                tracing::warn!(
                    "[page] PUT failed and md rollback also failed for {}: db_err={}, rollback_err={}",
                    id,
                    e,
                    rb
                );
            }
            return Err(ServerError::from(e));
        }
    };

    // The md was written before the gate ran, stamped `version + 1`. If the write
    // did not land, that stamp is a lie the page watcher will act on: it ingests
    // any md whose `origin_version` is >= the DB version as an `fs_edit`
    // (sources/page_watcher.rs), which would re-apply this agent content AS A
    // HUMAN WRITE and flip `user_edited` — laundering it straight past the gate
    // that just refused it. Restore the vault to what the DB actually holds.
    if result.outcome != WriteOutcome::Wrote {
        if let Err(rb) = roll_back_md() {
            tracing::warn!(
                "[page] PUT did not write ({:?}) and md rollback failed for {}: {}",
                result.outcome,
                id,
                rb
            );
        }
    }

    Ok(Json(wenlan_types::responses::PageWriteResponse {
        ok: true,
        revision_card_id: result.revision_card_id,
        gated: result.gated,
    }))
}

// ===== Revision history endpoints =====

/// GET /api/pages/{id}/revisions
///
/// Return the changelog entries for a page, plus envelope metadata.
pub async fn handle_get_page_revisions(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::ListPageRevisionsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    // Q1 sub-resource: revisions is a read-only browse surface, so a shadow id
    // resolves 200 (empty changelog) via the unfenced `_browse` twin.
    let page = db.get_page_scoped_browse(&id, &scope).await?;
    // A hidden page 404s here for the same reason it does at `GET /api/pages/
    // {id}`: for this caller it is not there. An `EntryOnly` verdict arrives
    // already reduced, which is what clears the envelope's own `stale_reason`.
    let page = wenlan_core::truth_adapter::filter_page(&db, &view.grant, page)
        .await?
        .ok_or_else(|| ServerError::NotFound("page not found".to_string()))?;
    // A changelog entry is prose about the page -- a delta summary, a citation
    // verdict -- so it rides on `Full`, never on the `EntryOnly` carve-out that
    // lets a listing keep a title. `filter_page_refs` over the named id alone
    // answers "is this page Full for this caller", since the entries have no id
    // of their own to key on; short of Full the changelog is not even read.
    let full = !wenlan_core::truth_adapter::filter_page_refs(
        &db,
        &view.grant,
        vec![id.clone()],
        |page_id| page_id.as_str(),
    )
    .await?
    .is_empty();
    let entries: Vec<wenlan_types::responses::PageChangelogEntry> = if full {
        let changelog_str = db.get_page_changelog_scoped(&id, &scope).await?;
        serde_json::from_str(&changelog_str)
            .map_err(|e| ServerError::Internal(format!("parse changelog: {e}")))?
    } else {
        Vec::new()
    };
    Ok(Json(wenlan_types::responses::ListPageRevisionsResponse {
        page_id: id,
        current_version: page.version,
        user_edited: page.user_edited,
        stale_reason: page.stale_reason.clone(),
        entries,
    }))
}
/// POST /api/pages/{id}/review
///
/// Mark a page human-reviewed, on the strength of a capability the app's Tauri
/// backend minted for one gesture. Binding spec:
/// `docs/plans/2026-07-27-m5-presence-threat-model.md`.
///
/// This handler deliberately holds no protocol logic. It resolves the secret,
/// hands the whole request to `review_page_with_presence`, and turns the answer
/// into a status code — because the order the checks run in is the part that is
/// easy to get backwards, and it belongs in one place rather than in every
/// handler that ever carries presence.
///
/// Nothing here logs the request. The capability's `Debug` redacts its material
/// anyway, but the surest way not to leak it is to never format it (§7).
pub async fn handle_review_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(request): Json<wenlan_types::requests::ReviewPageRequest>,
) -> Result<axum::response::Response, ServerError> {
    use axum::response::IntoResponse;

    let (db, presence_root) = {
        let s = state.read().await;
        (
            s.db.as_ref()
                .cloned()
                .ok_or(ServerError::DbNotInitialized)?,
            s.presence_root.clone(),
        )
    };

    // The root is handed over unresolved. Reading the secret here and refusing
    // early would put `presence_unavailable` in front of the replay lookup, so a
    // client retrying a dropped response after the secret was rotated would be
    // told its write failed when it had already succeeded — the exact T7 case
    // the ordering exists to prevent. No configured root is the same answer as
    // no secret in it, and both are reached at the one place that answers them.
    let now = chrono::Utc::now().timestamp();
    let outcome = db
        .review_page_with_presence(&request.presence, &id, presence_root.as_deref(), now)
        .await
        .map_err(ServerError::from)?;

    Ok(match outcome {
        // A replay returns the stored response verbatim and the same 200 the
        // first execution returned. A retry that succeeded looks like a success.
        wenlan_core::db::ReviewOutcome::Applied(receipt)
        | wenlan_core::db::ReviewOutcome::Replayed(receipt) => Json(receipt).into_response(),
        wenlan_core::db::ReviewOutcome::Refused(refusal) => refusal_response(refusal),
    })
}

/// The coarse wire answer. Carries the code and nothing else — no reason, no
/// path, and no part of the submitted capability (§7).
fn refusal_response(refusal: wenlan_core::presence::PresenceRefusal) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use wenlan_core::presence::PresenceRefusal;

    let status = match refusal {
        PresenceRefusal::Invalid | PresenceRefusal::Expired => StatusCode::FORBIDDEN,
        PresenceRefusal::Replayed | PresenceRefusal::Conflict => StatusCode::CONFLICT,
        PresenceRefusal::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
    };
    (status, Json(serde_json::json!({ "error": refusal.code() }))).into_response()
}

#[cfg(test)]
mod create_page_endpoint_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::json;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;
    use wenlan_types::requests::CreateConceptRequest;

    struct DataDirGuard {
        previous: Option<std::ffi::OsString>,
        _tmp: tempfile::TempDir,
    }

    impl DataDirGuard {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let pages = tmp.path().join("pages");
            std::fs::create_dir_all(&pages).unwrap();
            std::fs::write(
                tmp.path().join("config.json"),
                serde_json::json!({ "knowledge_path": pages.to_string_lossy() }).to_string(),
            )
            .unwrap();
            let previous = std::env::var_os("WENLAN_DATA_DIR");
            std::env::set_var("WENLAN_DATA_DIR", tmp.path());
            Self {
                previous,
                _tmp: tmp,
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

    async fn build_state_with_db(
        page_min_cluster_size: usize,
    ) -> (Arc<RwLock<ServerState>>, tempfile::TempDir) {
        build_state_with_db_and_page_match_threshold(
            page_min_cluster_size,
            wenlan_core::tuning::DistillationConfig::default().page_match_threshold,
        )
        .await
    }

    async fn build_state_with_db_and_page_match_threshold(
        page_min_cluster_size: usize,
        page_match_threshold: f64,
    ) -> (Arc<RwLock<ServerState>>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("failed to create tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new should succeed");
        db.upsert_documents(vec![
            wenlan_core::sources::RawDocument {
                source: "memory".to_string(),
                source_id: "mem-page-floor-a".to_string(),
                title: "mem-page-floor-a".to_string(),
                content: "Rust ownership prevents memory safety bugs".to_string(),
                last_modified: chrono::Utc::now().timestamp(),
                memory_type: Some("fact".to_string()),
                source_agent: Some("test".to_string()),
                confidence: Some(0.9),
                ..Default::default()
            },
            wenlan_core::sources::RawDocument {
                source: "memory".to_string(),
                source_id: "mem-page-floor-b".to_string(),
                title: "mem-page-floor-b".to_string(),
                content: "Rust borrowing validates references at compile time".to_string(),
                last_modified: chrono::Utc::now().timestamp(),
                memory_type: Some("fact".to_string()),
                source_agent: Some("test".to_string()),
                confidence: Some(0.9),
                ..Default::default()
            },
            wenlan_core::sources::RawDocument {
                source: "memory".to_string(),
                source_id: "mem-page-floor-c".to_string(),
                title: "mem-page-floor-c".to_string(),
                content: "Rust lifetimes describe how long references remain valid".to_string(),
                last_modified: chrono::Utc::now().timestamp(),
                memory_type: Some("fact".to_string()),
                source_agent: Some("test".to_string()),
                confidence: Some(0.9),
                ..Default::default()
            },
        ])
        .await
        .unwrap();
        let mut tuning = wenlan_core::tuning::TuningConfig::default();
        tuning.distillation.page_min_cluster_size = page_min_cluster_size;
        tuning.distillation.page_match_threshold = page_match_threshold;
        let server_state = ServerState {
            db: Some(Arc::new(db)),
            tuning,
            ..Default::default()
        };
        (Arc::new(RwLock::new(server_state)), tmp)
    }

    async fn seed_distilled_page(
        db: &wenlan_core::db::MemoryDB,
        title: &str,
        summary: Option<&str>,
        content: &str,
        space: Option<&str>,
        source_ids: &[&str],
    ) -> String {
        db.upsert_documents(
            source_ids
                .iter()
                .map(|source_id| wenlan_core::sources::RawDocument {
                    source: "memory".to_string(),
                    source_id: (*source_id).to_string(),
                    title: (*source_id).to_string(),
                    content: content.to_string(),
                    last_modified: chrono::Utc::now().timestamp(),
                    memory_type: Some("fact".to_string()),
                    source_agent: Some("test".to_string()),
                    confidence: Some(0.9),
                    confirmed: Some(true),
                    ..Default::default()
                })
                .collect(),
        )
        .await
        .unwrap();
        let result = wenlan_core::post_write::create_page_with_floor(
            db,
            CreateConceptRequest {
                title: title.to_string(),
                content: content.to_string(),
                summary: summary.map(str::to_string),
                entity_id: None,
                space: (space.map(str::to_string)).into(),
                source_memory_ids: source_ids.iter().map(|id| (*id).to_string()).collect(),
                creation_kind: Some("distilled".to_string()),
                workspace: space.map(str::to_string),
            },
            "test",
            None,
            source_ids.len(),
        )
        .await
        .unwrap();
        db.set_page_review_status(&result.id, "confirmed")
            .await
            .unwrap();
        result.id
    }

    #[tokio::test]
    async fn create_distilled_page_with_two_sources_returns_422() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let (state, _tmp) = build_state_with_db(3).await;
        let app = crate::router::build_router(state);
        let body = json!({
            "title": "Rust Safety",
            "content": "Rust ownership and borrowing prevent memory safety bugs by validating references",
            "summary": "Rust safety",
            "source_memory_ids": ["mem-page-floor-a", "mem-page-floor-b"],
            "creation_kind": "distilled"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/pages")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = axum::body::to_bytes(response.into_body(), 1_048_576)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["error"], "distilled page requires at least 3 distinct source memories (got 2)",
            "error should explain the exact source floor: {body}"
        );
    }

    #[tokio::test]
    async fn create_distilled_page_uses_configured_source_floor() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let (state, _tmp) = build_state_with_db(4).await;
        let app = crate::router::build_router(state);
        let body = json!({
            "title": "Rust References",
            "content": "Rust ownership, borrowing, and lifetimes keep references memory safe by validating references",
            "summary": "Rust references",
            "source_memory_ids": [
                "mem-page-floor-a",
                "mem-page-floor-b",
                "mem-page-floor-c"
            ],
            "creation_kind": "distilled"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/pages")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = axum::body::to_bytes(response.into_body(), 1_048_576)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["error"], "distilled page requires at least 4 distinct source memories (got 3)",
            "error should reflect ServerState tuning: {body}"
        );
    }

    #[tokio::test]
    async fn create_distilled_near_duplicate_returns_attached_to() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let (state, _tmp) = build_state_with_db(3).await;
        let db = {
            let s = state.read().await;
            s.db.as_ref().cloned().unwrap()
        };
        db.create_space("work", None, false).await.unwrap();
        let existing_id = seed_distilled_page(
            &db,
            "Rust Workspace Operations",
            Some("Rust workspace operations"),
            "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks",
            Some("work"),
            &["mem-page-floor-a", "mem-page-floor-b", "mem-page-floor-c"],
        )
        .await;
        let app = crate::router::build_router(state);
        let body = json!({
            "title": "Rust Workspace Operations",
            "content": "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks",
            "summary": "Rust workspace operations",
            "space": "work",
            "source_memory_ids": [
                "mem-page-floor-a",
                "mem-page-floor-b",
                "mem-page-floor-c"
            ],
            "creation_kind": "distilled"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/pages")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1_048_576)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["id"].as_str(), Some(existing_id.as_str()));
        assert_eq!(
            body["attached_to"].as_str(),
            Some(existing_id.as_str()),
            "create_page response must expose the attach target: {body}"
        );
        let pages = db.list_pages("active", 10, 0).await.unwrap();
        assert_eq!(pages.len(), 1, "HTTP near-duplicate attach must not mint");
    }

    #[tokio::test]
    async fn create_distilled_near_duplicate_uses_configured_page_match_threshold() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let (state, _tmp) = build_state_with_db_and_page_match_threshold(3, 1.1).await;
        let db = {
            let s = state.read().await;
            s.db.as_ref().cloned().unwrap()
        };
        db.create_space("work", None, false).await.unwrap();
        let existing_id = seed_distilled_page(
            &db,
            "Rust Workspace Operations",
            Some("Rust workspace operations"),
            "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks",
            Some("recap"),
            &["mem-page-floor-a", "mem-page-floor-b", "mem-page-floor-c"],
        )
        .await;
        let app = crate::router::build_router(state);
        let body = json!({
            "title": "Rust Workspace Operations",
            "content": "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks",
            "summary": "Rust workspace operations",
            "space": "work",
            "source_memory_ids": [
                "mem-page-floor-a",
                "mem-page-floor-b",
                "mem-page-floor-c"
            ],
            "creation_kind": "distilled"
        });

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/pages")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1_048_576)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_ne!(
            body["id"].as_str(),
            Some(existing_id.as_str()),
            "configured threshold above possible cosine must force a new page: {body}"
        );
        assert!(
            body.get("attached_to").is_none(),
            "new page response must omit attached_to: {body}"
        );
        let pages = db.list_pages("active", 10, 0).await.unwrap();
        assert_eq!(
            pages.len(),
            2,
            "below configured threshold should mint a new page"
        );
    }
}

#[cfg(test)]
mod conflict_identity_tests {
    use super::visible_conflict_identity;
    use crate::truth_guard::TruthView;
    use std::sync::Arc;
    use wenlan_core::db::MemoryDB;

    const PAGE: &str = "page_00000000-0000-4000-8000-0000000000c1";
    /// Stored scope of an unfiled publish: the M1 sentinel space id.
    const SCOPE: &str = "00000000-0000-4000-8000-000000000001";

    async fn test_db() -> (Arc<MemoryDB>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = Arc::new(MemoryDB::new(tmp.path(), emitter).await.unwrap());
        (db, tmp)
    }

    async fn publish(db: &MemoryDB, id: &str, title: &str) -> wenlan_core::pages::Page {
        let draft = db
            .create_page_draft_with_id(id, title, "Body", None, None)
            .await
            .unwrap();
        match db.publish_page_draft(id, draft.version).await.unwrap() {
            wenlan_core::pages::PageDraftPublishOutcome::Published(page) => page,
            _ => panic!("expected a clean publish"),
        }
    }

    #[tokio::test]
    async fn a_visible_conflicting_page_discloses_its_current_identity() {
        let (db, _tmp) = test_db().await;
        let page = publish(&db, PAGE, "Secret Plan").await;
        let wanted = MemoryDB::page_title_key("  secret PLAN ");
        let got =
            visible_conflict_identity(&db, &TruthView::automatic(), PAGE, &wanted, SCOPE).await;
        assert_eq!(got, Some((page.id, page.title)));
    }

    #[tokio::test]
    async fn a_concurrently_deleted_page_degrades_to_omission() {
        let (db, _tmp) = test_db().await;
        publish(&db, PAGE, "Secret Plan").await;
        db.delete_page(PAGE).await.unwrap();
        let wanted = MemoryDB::page_title_key("Secret Plan");
        assert_eq!(
            visible_conflict_identity(&db, &TruthView::automatic(), PAGE, &wanted, SCOPE).await,
            None
        );
    }

    /// Replace the row's title out from under the open `MemoryDB`, the way a
    /// concurrent writer would between the publish transaction and the
    /// handler's reload. Same direct-file seam as the routes tests.
    async fn overwrite_title(tmp: &tempfile::TempDir, id: &str, title: &str) {
        let raw = libsql::Builder::new_local(tmp.path().join("origin_memory.db"))
            .build()
            .await
            .unwrap();
        let conn = raw.connect().unwrap();
        conn.execute(
            "UPDATE pages SET title = ?1 WHERE id = ?2",
            libsql::params![title, id],
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_replaced_row_with_an_unrelated_title_stays_omitted() {
        // Page ids are reusable and rows are mutable: between the publish
        // transaction and this reload the id can hold a different title. The
        // reloaded row no longer folds to the conflicting title, so neither
        // the stale captured title nor the new row's title may be disclosed.
        let (db, tmp) = test_db().await;
        publish(&db, PAGE, "Secret Plan").await;
        overwrite_title(&tmp, PAGE, "Innocuous Note").await;
        let wanted = MemoryDB::page_title_key("Secret Plan");
        assert_eq!(
            visible_conflict_identity(&db, &TruthView::automatic(), PAGE, &wanted, SCOPE).await,
            None
        );
    }

    #[tokio::test]
    async fn a_replaced_row_still_folding_to_the_conflict_discloses_the_reloaded_row() {
        let (db, tmp) = test_db().await;
        let page = publish(&db, PAGE, "Secret Plan").await;
        overwrite_title(&tmp, PAGE, "SECRET plan").await;
        let wanted = MemoryDB::page_title_key("Secret Plan");
        let got =
            visible_conflict_identity(&db, &TruthView::automatic(), PAGE, &wanted, SCOPE).await;
        // The identity comes from the reloaded row, never the stale capture.
        assert_eq!(got, Some((page.id, "SECRET plan".to_string())));
    }

    /// Same direct-file seam as `overwrite_title`, for an arbitrary column.
    async fn overwrite_column(tmp: &tempfile::TempDir, id: &str, column: &str, value: &str) {
        let raw = libsql::Builder::new_local(tmp.path().join("origin_memory.db"))
            .build()
            .await
            .unwrap();
        let conn = raw.connect().unwrap();
        conn.execute(
            &format!("UPDATE pages SET {column} = ?1 WHERE id = ?2"),
            libsql::params![value, id],
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_page_moved_to_another_space_degrades_to_omission() {
        // The publish found the conflict in one scope; if the page moves to a
        // different space before the handler reloads it, it is no longer that
        // conflict and its identity must not be disclosed.
        let (db, tmp) = test_db().await;
        publish(&db, PAGE, "Secret Plan").await;
        overwrite_column(&tmp, PAGE, "space", "another-space").await;
        let wanted = MemoryDB::page_title_key("Secret Plan");
        assert_eq!(
            visible_conflict_identity(&db, &TruthView::automatic(), PAGE, &wanted, SCOPE).await,
            None
        );
    }

    #[tokio::test]
    async fn an_entity_shadow_conflict_stays_omitted() {
        // Publish treats entity shadow pages as conflicts, but conflict
        // responses have never disclosed their identity; the snapshot must
        // keep that fence.
        let (db, tmp) = test_db().await;
        publish(&db, PAGE, "Secret Plan").await;
        overwrite_column(&tmp, PAGE, "kind", "entity").await;
        let wanted = MemoryDB::page_title_key("Secret Plan");
        assert_eq!(
            visible_conflict_identity(&db, &TruthView::automatic(), PAGE, &wanted, SCOPE).await,
            None
        );
    }
}
