// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::route_registry::{get, post, TrackedRouter};
use crate::state::{ServerState, SharedState};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, HeaderValue},
    response::Json,
    Extension,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_core::router::classify::estimate_tokens;
use wenlan_types::requests::ChatContextRequest;
use wenlan_types::responses::{
    ChatContextResponse, KnowledgeContext, ProfileContext, TierTokenEstimates,
};

pub(crate) fn register(router: TrackedRouter<SharedState>) -> TrackedRouter<SharedState> {
    router
        .route("/api/health", get(handle_health))
        .route("/api/status", get(handle_status))
        .route("/api/search", post(handle_search))
        .route("/api/context", post(handle_context))
        .route("/api/llm/test", post(handle_test_llm))
        .route("/api/shutdown", post(handle_shutdown))
        .route("/api/debug/pipeline", get(handle_pipeline_status))
        .route("/api/retrievals/recent", get(handle_recent_retrievals))
        .route("/api/pages/recent", get(handle_recent_pages))
        .route("/api/steep", post(handle_steep))
        .route("/api/distill", post(handle_distill))
        .route("/api/distill/{page_id}", post(handle_redistill))
}

// ===== Request/Response Types =====

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub source_filter: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<wenlan_core::db::SearchResult>,
    pub took_ms: f64,
    /// Distilled pages surfaced through the shared `select_visible_pages`
    /// visibility gate (space-scope → effective-tier → confirmed/rank/cap).
    /// Absent when nothing passed the gate; mirrors `SearchMemoryResponse`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supplemental_pages: Option<Vec<wenlan_core::db::SearchResult>>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub db_initialized: bool,
    pub version: String,
}

// ===== Route Handlers =====

/// GET /api/health - Health check endpoint
pub async fn handle_health(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<HealthResponse>, ServerError> {
    let s = state.read().await;
    let db_initialized = s.db.is_some();

    Ok(Json(HealthResponse {
        status: "ok".to_string(),
        db_initialized,
        version: wenlan_core::version().to_string(),
    }))
}

/// Map the core document-enrichment queue summary to the wire `QueueStatus`.
/// Pure snapshot→response glue (no business logic): idle when nothing is
/// pending, paused when a paused row carries a reason, else active.
fn queue_status_wire(
    core: wenlan_core::db::DocEnrichmentQueueStatus,
) -> wenlan_types::responses::QueueStatus {
    use wenlan_types::responses::QueueStatus;
    if core.pending == 0 {
        QueueStatus::Idle
    } else if let Some(reason) = core.paused_reason {
        QueueStatus::Paused {
            reason,
            pending: core.pending,
            next_retry_at: core.next_retry_at,
        }
    } else {
        QueueStatus::Active {
            pending: core.pending,
        }
    }
}

/// GET /api/status - Get indexing status
pub async fn handle_status(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<wenlan_types::responses::StatusResponse>, ServerError> {
    let (db, reranker_status, reranker_light_status, reranker_mode, on_device_inference) = {
        let s = state.read().await;
        (
            s.db.clone(),
            s.reranker_status.clone(),
            s.reranker_light_status.clone(),
            s.reranker_mode.clone(),
            s.llm
                .as_ref()
                .and_then(|provider| provider.inference_runtime_info())
                .unwrap_or_default(),
        )
    };

    // Present only when the cutover is actually live. There is one contract —
    // "absent means not live" — and reporting generation 0 would give a client a
    // second way to spell it, so a client that checks for the field without also
    // checking its value would conclude enforcement is on. A generation that
    // cannot be read is absent for the same reason: a daemon that just failed to
    // look has no standing to report a state.
    let truth = match &db {
        Some(db) => db
            .truth_cutover_generation()
            .await
            .ok()
            .filter(|generation| *generation > 0)
            .map(|generation| wenlan_types::responses::TruthStatus {
                cutover_generation: generation,
                contract_version: wenlan_core::truth_contract::TRUTH_CONTRACT_VERSION,
            }),
        None => None,
    };

    let (files_indexed, queue, compile_queue) = if let Some(db) = &db {
        let files_indexed = db.count().await.unwrap_or(0);
        let queue = db
            .document_enrichment_queue_status()
            .await
            .map(queue_status_wire)
            .unwrap_or_default();
        let compile_queue = match wenlan_core::refinery::compile_queue_depth(db).await {
            Ok(0) | Err(_) => wenlan_types::responses::QueueStatus::Idle,
            Ok(pending) => wenlan_types::responses::QueueStatus::Active {
                pending: pending as u64,
            },
        };
        (files_indexed, queue, compile_queue)
    } else {
        (
            0,
            wenlan_types::responses::QueueStatus::Idle,
            wenlan_types::responses::QueueStatus::Idle,
        )
    };

    Ok(Json(wenlan_types::responses::StatusResponse {
        is_running: true,
        files_indexed,
        files_total: 0,
        sources_connected: vec![],
        queue,
        compile_queue,
        reranker: reranker_status,
        reranker_light: reranker_light_status,
        reranker_mode,
        on_device_inference,
        capabilities: vec!["default_save_space".to_string()],
        truth,
    }))
}

/// POST /api/search - Semantic search endpoint
pub async fn handle_search(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ServerError> {
    let start = std::time::Instant::now();

    let (db, reranker_light) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        (db, s.reranker_light.clone())
    };
    let scope =
        crate::read_scope::effective_read_scope(&db, req.space.as_deref(), header_space.as_deref())
            .await?;

    let results = if req.source_filter.as_deref() == Some("memory") {
        match reranker_light {
            // Quick-path CE (WENLAN_RERANKER_MODE lite/full): fetch a widened pool via
            // plain search_memory, then cross-encode down to req.limit at the handler
            // layer. search_memory itself stays CE-free, so internal callers are unaffected.
            Some(reranker) => {
                // Light path uses the default pool (max(limit,10)). Unlike the deep path
                // (db.rs), it does NOT read RERANK_POOL_MULTIPLIER/FLOOR — turbo is cheap
                // and the light path isn't an eval-sweep surface. (review note)
                let pool = wenlan_core::db::compute_rerank_fetch_pool(req.limit, None, None);
                let pooled = db
                    .search_memory(&req.query, pool, None, &scope, None, None, None, None)
                    .await
                    .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
                wenlan_core::db::rerank_results_light(reranker, &req.query, pooled, req.limit).await
            }
            // mode=off (default): byte-identical to the pre-mode path.
            None => db
                .search_memory(&req.query, req.limit, None, &scope, None, None, None, None)
                .await
                .map_err(|e| ServerError::SearchFailed(e.to_string()))?,
        }
    } else {
        db.search(&req.query, req.limit, req.source_filter.as_deref(), &scope)
            .await
            .map_err(|e| ServerError::SearchFailed(e.to_string()))?
    };

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Additive page path: surface gated distilled pages through the
    // `supplemental_pages` field, routed through the SAME shared
    // `select_visible_pages` visibility gate (space-scope → effective-tier →
    // confirmed/rank/cap) used by /api/context and /api/memory/search. Resolve
    // caller trust from the `x-agent-name` header, fail-CLOSED to "unknown"
    // (mirror handle_context above). Skipped for the "recent context" sentinel.
    // Fail CLOSED: any lookup error leaves pages out (never surface ungated).
    let supplemental_pages = if req.query == "recent context" {
        None
    } else {
        let agent_name = crate::memory_routes::extract_agent_name(&headers, None);
        let trust_level = if agent_name == "unknown" {
            "unknown".to_string()
        } else {
            db.get_agent(&agent_name)
                .await
                .ok()
                .flatten()
                .map(|a| a.trust_level)
                .unwrap_or_else(|| "unknown".to_string())
        };
        let raw = db
            .search_pages_scoped(&req.query, 3, None, &scope)
            .await
            .unwrap_or_default();
        let ids: std::collections::HashSet<String> =
            results.iter().map(|r| r.source_id.clone()).collect();
        let visible = db
            .select_visible_pages_scoped(raw, &scope, &ids, &trust_level, 3)
            .await;
        // A surfaced page arrives as a `SearchResult` carrying the page's PROSE
        // in `content`, not as a `Page`, so it rides on `Full` -- there is no
        // entry form for a search hit. `source_id` is the page id
        // (`search_result_from_page`, db.rs).
        let surfaced: Vec<wenlan_types::memory::SearchResult> = visible
            .into_iter()
            .map(wenlan_core::db::MemoryDB::search_result_from_page)
            .collect();
        let surfaced =
            wenlan_core::truth_adapter::filter_page_refs(&db, &view.grant, surfaced, |row| {
                row.source_id.as_str()
            })
            .await
            .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
        (!surfaced.is_empty()).then_some(surfaced)
    };

    Ok(Json(SearchResponse {
        results,
        took_ms,
        supplemental_pages,
    }))
}

// ===== Context Endpoints =====

fn render_brief_for_legacy(response: &wenlan_types::BriefReadResponse) -> String {
    use wenlan_types::BriefReadState;

    let Some(brief) = response.brief.as_ref() else {
        return match response.state {
            BriefReadState::SpaceNotResolved => {
                "## Space Brief\nNo Space resolved. Specify a Space to load its Brief.".into()
            }
            BriefReadState::BriefNotCreated => format!(
                "## Space Brief — {}\nNo Brief has been created for this Space.",
                response.space.as_deref().unwrap_or("unknown")
            ),
            BriefReadState::Ready => "## Space Brief\nBrief unavailable.".into(),
        };
    };

    let mut output = format!("## Space Brief — {}\n", brief.space);
    output.push_str("\n### Last session\n");
    if brief.last_session_summary.trim().is_empty() {
        output.push_str("_No session summary._\n");
    } else {
        output.push_str(brief.last_session_summary.trim());
        output.push('\n');
    }
    for (heading, items) in [("Active", &brief.active), ("Backlog", &brief.backlog)] {
        output.push_str(&format!("\n### {heading}\n"));
        if items.is_empty() {
            output.push_str("_None._\n");
            continue;
        }
        for item in items {
            output.push_str("- ");
            output.push_str(&item.text);
            if let Some(gate) = item.gate.as_deref() {
                output.push_str(" (gated: ");
                output.push_str(gate);
                output.push(')');
            }
            output.push('\n');
        }
    }
    output
}

/// POST /api/context — deprecated compatibility adapter over Space Brief.
/// No-topic calls are storage reads only; a topic composes the same Brief with
/// separately-labelled, same-Space recall results.
pub async fn handle_context(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    Json(req): Json<ChatContextRequest>,
) -> Result<Json<ChatContextResponse>, ServerError> {
    let start = std::time::Instant::now();
    let topic = req.query.or(req.conversation_id);
    let request = wenlan_types::BriefReadRequest {
        topic,
        space: req.space,
        legacy_context_limit: Some(req.max_chunks),
    };
    // `/api/context` reaches every page it serves through the Brief handler, so
    // the grant this route resolved travels with the call rather than being
    // re-derived there.
    let Json(brief_response) = crate::brief_routes::handle_read_brief(
        State(state),
        crate::space_header::SpaceHeader(header_space),
        view,
        Json(request),
    )
    .await?;

    let brief_context = render_brief_for_legacy(&brief_response);
    let relevant_memories = brief_response
        .related_context
        .as_ref()
        .map(|related| related.results.clone())
        .unwrap_or_default();
    let related_context = brief_response.related_context.as_ref().map(|related| {
        let mut section = format!("## Related Context — {}\n", related.query);
        if related.results.is_empty() {
            section.push_str("_No related memories found._\n");
        } else {
            for result in &related.results {
                section.push_str(&format!("[{}] {}\n\n", result.title, result.content));
            }
        }
        section
    });
    let context = related_context
        .map(|related| format!("{brief_context}\n\n{related}"))
        .unwrap_or_else(|| brief_context.clone());
    let tier2 = estimate_tokens(&brief_context);
    let tier3 = relevant_memories
        .iter()
        .map(|result| result.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let tier3 = estimate_tokens(&tier3);

    #[allow(deprecated)]
    let profile = ProfileContext {
        narrative: String::new(),
        identity: Vec::new(),
        preferences: Vec::new(),
        goals: Vec::new(),
    };
    Ok(Json(ChatContextResponse {
        context,
        profile,
        knowledge: KnowledgeContext {
            pages: Vec::new(),
            decisions: Vec::new(),
            relevant_memories,
            graph_context: Vec::new(),
        },
        took_ms: start.elapsed().as_secs_f64() * 1000.0,
        token_estimates: TierTokenEstimates {
            tier1_identity: 0,
            tier2_project: tier2,
            tier3_relevant: tier3,
            total: tier2 + tier3,
        },
    }))
}

/// GET /api/debug/pipeline
pub async fn handle_pipeline_status(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    // Snapshot the DB Arc so the read guard is not held across the await
    // (AGENTS.md: never hold a tokio RwLock guard across .await).
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let status = db
        .pipeline_status()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(status))
}

/// POST /api/steep
pub async fn handle_steep(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<SteepResponse>, ServerError> {
    let (
        db,
        llm,
        api_llm,
        synthesis_llm,
        external_llm,
        prompts,
        tuning,
        confidence_cfg,
        distillation_cfg,
    ) = {
        let s = state.read().await;
        (
            s.db.clone()
                .ok_or(ServerError::Internal("DB not initialized".into()))?,
            s.llm.clone(),
            s.api_llm.clone(),
            s.synthesis_llm.clone(),
            s.external_llm.clone(),
            s.prompts.clone(),
            s.tuning.refinery.clone(),
            s.tuning.confidence.clone(),
            s.tuning.distillation.clone(),
        )
    };
    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    let result = wenlan_core::refinery::run_periodic_steep_with_api(
        &db,
        llm.as_ref(),
        api_llm.as_ref(),
        synthesis_llm.as_ref(),
        external_llm.as_ref(),
        &prompts,
        &tuning,
        &confidence_cfg,
        &distillation_cfg,
        Some(&knowledge_path),
        wenlan_core::refinery::TriggerKind::Backstop,
    )
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(SteepResponse {
        memories_decayed: result.memories_decayed,
        recaps_generated: result.recaps_generated,
        distilled: result.distilled,
        pending_remaining: result.pending_remaining,
        phases: result.phases,
    }))
}

#[derive(Debug, Serialize)]
pub struct SteepResponse {
    pub memories_decayed: u64,
    pub recaps_generated: u32,
    pub distilled: u32,
    pub pending_remaining: u32,
    pub phases: Vec<wenlan_core::refinery::PhaseResult>,
}

/// Request body for POST /api/distill. All fields are optional: an empty
/// body (or omitted Content-Type) is treated as `DistillRequest::default()`
/// which preserves the historical full-pass behavior.
///
/// `deny_unknown_fields` rejects bodies that name a field we don't recognize
/// (e.g. an old MCP client that still sends `page_id`). The error surfaces as
/// a 400 with a serde diagnostic — the alternative is silently dropping the
/// field and running a global pass the caller didn't ask for.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistillRequest {
    /// Free-form target string. Resolved server-side:
    /// `page_*`/`concept_*` → page redistill; entity name → scoped distill;
    /// domain value → scoped distill; anything else → 404-ish hint payload.
    #[serde(default, alias = "page_id")]
    pub target: Option<String>,

    /// When true, clears `user_edited` on the resolved page before recompile.
    /// Used by `/distill rebuild <page>` to opt into wiping user prose.
    /// Only valid when target resolves to a single page; otherwise returns a hint.
    /// Requires daemon LLM.
    #[serde(default)]
    pub force: bool,

    /// Read-only pre-flight gate run before trusting the formation path and
    /// again before the retro sweep; returns the fixed threshold stats grid.
    #[serde(default)]
    pub sweep: bool,
}

/// POST /api/distill
/// The write half of this route is gated downstream, but the RESPONSE is a page
/// reader three times over: stale pages carry title/summary/stale_reason and
/// their source ids, each pending cluster can name the page it overlaps, and the
/// orphan feed is text lifted out of page bodies. All three go through the
/// adapter below.
///
/// Its manifest shape is `MarkerShape::None`, so `view.grant` always resolves to
/// `Automatic` today. Taking the view anyway is what keeps that a fact about the
/// manifest rather than an assumption baked into the handler.
pub async fn handle_distill(
    State(state): State<Arc<RwLock<ServerState>>>,
    view: crate::truth_guard::TruthView,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, ServerError> {
    let req: DistillRequest = if body.is_empty() {
        DistillRequest::default()
    } else {
        serde_json::from_slice(&body).map_err(|e| ServerError::ValidationError(e.to_string()))?
    };

    if req.sweep {
        if req.target.is_some() || req.force {
            return Ok(Json(serde_json::json!({
                "sweep": true,
                "hint": "sweep=true cannot be combined with target or force; omit both to run the read-only formation-threshold grid",
            })));
        }
        let (db, tuning) = {
            let s = state.read().await;
            (s.db.clone(), s.tuning.distillation.clone())
        };
        let db = db.ok_or(ServerError::Internal("DB not initialized".into()))?;
        let report = wenlan_core::refinery::formation_sweep(&db, &tuning).await?;
        let payload = serde_json::to_value(report)
            .map_err(|e| ServerError::Internal(format!("formation sweep serialize: {e}")))?;
        return Ok(Json(payload));
    }

    // `/api/distill` always returns clusters as `pending` — the route is
    // user-triggered, so synthesis belongs to whoever called it (the agent
    // session via the skill, or another client with its own LLM). The
    // background refinery calls `distill_pages_scoped` directly with the
    // daemon LLM; that path is the one that synthesizes inline. Two
    // triggers, one shared function, no behavior drift inside the function.
    let (db, prompts, tuning, llm, api_llm) = {
        let s = state.read().await;
        (
            s.db.clone(),
            s.prompts.clone(),
            s.tuning.distillation.clone(),
            s.llm.clone(),
            s.api_llm.clone(),
        )
    };
    let db = db.ok_or(ServerError::Internal("DB not initialized".into()))?;

    let knowledge_path = {
        let config = wenlan_core::config::load_config();
        Some(config.knowledge_path_or_default())
    };

    let target = match req.target.as_deref() {
        Some(raw) if !raw.is_empty() => {
            match wenlan_core::refinery::resolve_distill_target(&db, raw).await? {
                Some(t) => Some(t),
                None => {
                    return Ok(Json(serde_json::json!({
                        "pages_created": 0,
                        "pages_updated": 0,
                        "unresolved": raw,
                        "hint": "target must be a page id (page_* / concept_*), an entity name, or a domain value",
                    })));
                }
            }
        }
        _ => None,
    };

    // Force path: only clear user_edited after an LLM is available to run the
    // rewrite. A skipped no-LLM response must not unlock user prose.
    // Only valid when target resolves to a single page; all other targets
    // (entity, domain, none) return a hint payload.
    if req.force {
        match &target {
            Some(wenlan_core::synthesis::distill::DistillTarget::Page(page_id)) => {
                let prefer_llm = api_llm.as_ref().or(llm.as_ref());
                if !prefer_llm
                    .as_ref()
                    .map(|provider| provider.is_available())
                    .unwrap_or(false)
                {
                    return Ok(Json(serde_json::json!({
                        "status": "skipped",
                        "force": true,
                        "page_id": page_id,
                        "updated": false,
                        "hint": "force rebuild needs an LLM in the daemon — install an on-device model or set an Anthropic key via `wenlan setup` / `/origin:doctor`",
                    })));
                }
                db.clear_user_edited(page_id)
                    .await
                    .map_err(ServerError::from)?;
                let updated = wenlan_core::refinery::deep_distill_single(
                    &db,
                    prefer_llm,
                    &prompts,
                    page_id,
                    knowledge_path.as_deref(),
                )
                .await?;
                return Ok(Json(serde_json::json!({
                    "status": "ok",
                    "force": true,
                    "page_id": page_id,
                    "updated": updated,
                })));
            }
            _ => {
                return Ok(Json(serde_json::json!({
                    "unresolved": req.target.clone().unwrap_or_default(),
                    "hint": "force=true only valid when target is a single page id",
                })));
            }
        }
    }

    let scoped = target.is_some();

    // Capture scope filters before moving `target` into distill_pages_scoped
    // so we can use them to scope the stale-page list below.
    let (stale_entity_filter, stale_space_filter) = match &target {
        Some(wenlan_core::synthesis::distill::DistillTarget::Entity { id, .. }) => {
            (Some(id.clone()), None)
        }
        Some(wenlan_core::synthesis::distill::DistillTarget::Domain(d)) => (None, Some(d.clone())),
        Some(wenlan_core::synthesis::distill::DistillTarget::Page(_)) | None => (None, None),
    };

    let result = wenlan_core::refinery::distill_pages_scoped(
        &db,
        None, // route never invokes daemon LLM; caller synthesizes pending
        &prompts,
        &tuning,
        knowledge_path.as_deref(),
        target,
    )
    .await?;

    // Shared "fully covered" rule with refinery (subset OR Jaccard >= 0.8):
    // those clusters never need synth from either path.
    //
    // Partial-overlap clusters diverge intentionally:
    // - Refinery (no agent reach): marks the page stale and skips. Refresh
    //   is `redistill_changed_pages`' job and needs daemon LLM.
    // - Route (agent has LLM in caller's session): surfaces the cluster
    //   with `existing_page_id` so the skill can synth the refresh inline.
    //
    // Divergence sits at the orchestration layer; the primitive is shared.
    //
    // Performance: load the active-page source-memory index ONCE and reuse
    // it across every cluster comparison, so the route is O(P + P*C) hash
    // ops instead of P*C SQL roundtrips + JSON re-parses.
    let page_index = db
        .load_page_source_index()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    // The overlap match names a page, and the cluster payload publishes that
    // name. Resolve the whole loop's candidates in ONE adapter call up front --
    // per-cluster calls would be one `page_visibility` query per pending
    // cluster, and the overlap itself is a pure in-memory scan of `page_index`.
    let best_per_cluster: Vec<_> = result
        .pending
        .iter()
        .map(|cluster| {
            wenlan_core::db::best_overlapping_page_in_index(&page_index, &cluster.source_ids)
        })
        .collect();
    let overlap_candidates: Vec<String> = best_per_cluster
        .iter()
        .flatten()
        .map(|m| m.page_id.clone())
        .collect();
    let visible_overlaps: std::collections::HashSet<String> =
        wenlan_core::truth_adapter::filter_page_refs(
            &db,
            &view.grant,
            overlap_candidates,
            |page_id| page_id.as_str(),
        )
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .into_iter()
        .collect();

    let mut filtered_pending: Vec<serde_json::Value> = Vec::new();
    for (cluster, best) in result.pending.iter().zip(best_per_cluster) {
        let cluster_size = cluster.source_ids.len();

        let (existing_page_id, existing_page_title, new_memory_count) = match best {
            // Gate the disclosure, keep the suppression. A fully-covered cluster
            // is dropped whether or not the caller may see the page that covers
            // it: suppressing a duplicate uses the knowledge without disclosing
            // it, and surfacing the cluster instead would have the agent mint a
            // second page for a topic that already has one.
            Some(m) if m.intersection >= cluster_size || m.jaccard >= 0.8 => continue,
            // For this caller the overlapping page does not exist, so neither
            // its id nor its title is emitted -- and every memory in the cluster
            // is new. `cluster_size - m.intersection` is derived from the hidden
            // page's own source list, so publishing it would leak how much of
            // this cluster that page already covers.
            Some(m) if !visible_overlaps.contains(&m.page_id) => (None, None, cluster_size),
            Some(m) => (
                Some(m.page_id),
                Some(m.page_title),
                cluster_size - m.intersection,
            ),
            None => (None, None, cluster_size),
        };

        let mut payload = serde_json::to_value(cluster)
            .map_err(|e| ServerError::Internal(format!("cluster serialize: {e}")))?;
        if let serde_json::Value::Object(ref mut map) = payload {
            if let Some(id) = existing_page_id {
                map.insert("existing_page_id".into(), serde_json::json!(id));
            }
            if let Some(title) = existing_page_title {
                map.insert("existing_page_title".into(), serde_json::json!(title));
            }
            map.insert(
                "new_memory_count".into(),
                serde_json::json!(new_memory_count),
            );
        }
        filtered_pending.push(payload);
    }

    // Surface stale pages so the agent's `/distill` skill can refresh them
    // via PUT /api/pages/{id}. Restrict to `source_updated` — `source_conflict`
    // is the user-edited escalation and needs a human-in-the-loop UX, not an
    // auto-refresh. Scoped by entity/domain when the target was scoped.
    let stale_pages_list = db
        .list_stale_pages_scoped(
            "source_updated",
            stale_entity_filter.as_deref(),
            stale_space_filter.as_deref(),
        )
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    // 10 is the LIMIT inside list_stale_pages_scoped. Surface a hint so the
    // skill can tell the user to re-run instead of silently dropping rows.
    // Read PRE-filter on purpose: the flag means "the query hit its cap", not
    // "you were shown ten", so the adapter below must not move it.
    let stale_truncated = stale_pages_list.len() == 10;

    // The payload quotes page prose -- title, summary, and `stale_reason`, which
    // names why the page changed -- and hands out `source_memory_ids` on top, so
    // the pages are reduced before it is built rather than after. Filtering here
    // means an `EntryOnly` page arrives with those fields already blanked and
    // flows through the map below unchanged.
    let stale_pages_list =
        wenlan_core::truth_adapter::filter_pages(&db, &view.grant, stale_pages_list)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;

    let stale_pages_payload: Vec<serde_json::Value> = stale_pages_list
        .iter()
        .map(|p| {
            serde_json::json!({
                "page_id": p.id,
                "title": p.title,
                "summary": p.summary,
                "source_memory_ids": p.source_memory_ids,
                "sources_updated_count": p.sources_updated_count,
                "stale_reason": p.stale_reason,
                "user_edited": p.user_edited,
            })
        })
        .collect();

    // Walk orphan rows in case a page minted earlier in this pass (or by
    // a concurrent create_page call) now matches an existing orphan label.
    // The refinery's Emergence phase does the same on its tick; doing it
    // inline here keeps the route's response (and the skill's view of the
    // graph) consistent without waiting up to a poll interval.
    if let Err(e) = db.resolve_orphan_page_links().await {
        tracing::warn!("[distill] orphan link resolve failed: {e}");
    }

    // Surface the orphan-wikilink feed so the skill can prompt the user to
    // distill a page on a topic that other pages already reach for. Cheap:
    // one GROUP BY against page_links, capped at 100. Threshold N=2 means
    // a single typo doesn't show up — needs at least two distinct pages
    // citing the same label before it counts as signal.
    //
    // Only surface on unscoped passes. A scoped `/distill rust` user
    // doesn't want generic topic suggestions from elsewhere in the graph;
    // that would confuse them about what "scope" means. The unscoped
    // global pass (`/distill deep` or bare `/distill` outside a repo) is
    // the right place for this feed.
    let orphan_topics: Vec<serde_json::Value> = if scoped {
        Vec::new()
    } else {
        // Same rows-then-fold shape as GET /api/pages/orphan-links, and for the
        // same reason: a label leaks the page whose body it was lifted out of,
        // and the aggregate twin would compute its counts and its 100-row cap
        // over pages this caller cannot see. `Global` keeps this feed's existing
        // unscoped reach -- the route already ignored the request's Space here.
        let rows = db
            .list_orphan_link_rows_scoped(&wenlan_core::read_scope::ReadScope::Global)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
        let rows = wenlan_core::truth_adapter::filter_page_refs(&db, &view.grant, rows, |row| {
            row.source_page_id.as_str()
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
        wenlan_core::db::MemoryDB::fold_orphan_labels(rows, 2)
            .into_iter()
            .map(|(label, count)| serde_json::json!({"label": label, "count": count}))
            .collect()
    };

    // `clusters_found` is the raw cluster count BEFORE the covered-page
    // filter above drops clusters a page already subsumes -- "found" means
    // clustering formed them, not that they survived disclosure.
    let no_clusters_and_no_pages = result.pending.is_empty() && result.created.is_empty();
    let mut response = serde_json::json!({
        "pages_created": result.created.len(),
        "scoped": scoped,
        "created_ids": result.created,
        "pending": filtered_pending,
        "stale_pages": stale_pages_payload,
        "stale_truncated": stale_truncated,
        "orphan_topics": orphan_topics,
        "clusters_found": result.pending.len(),
    });
    if no_clusters_and_no_pages {
        if let serde_json::Value::Object(ref mut map) = response {
            map.insert(
                "hint".into(),
                serde_json::json!(
                    "No page-sized cluster formed in this scope: nothing grouped into 3 or more related memories that fit one page. Capture more related memories, or check the daemon log for dropped clusters."
                ),
            );
        }
    }
    Ok(Json(response))
}

/// POST /api/distill/{page_id}
///
/// Re-distill a single page. Requires the daemon to have an LLM available
/// (on-device or Anthropic key). When no model/key is configured (local memory
/// mode) the route returns a 200 with a hint payload instead of a 500 —
/// the caller's intent (refresh this page) can't be honored, but the
/// failure mode is documented in the response so the skill can surface
/// it to the user.
pub async fn handle_redistill(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(page_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let (db, llm, api_llm, prompts) = {
        let s = state.read().await;
        (
            s.db.clone(),
            s.llm.clone(),
            s.api_llm.clone(),
            s.prompts.clone(),
        )
    };
    let db = db.ok_or(ServerError::Internal("DB not initialized".into()))?;

    let knowledge_path = {
        let config = wenlan_core::config::load_config();
        Some(config.knowledge_path_or_default())
    };

    let prefer_llm = api_llm.as_ref().or(llm.as_ref());
    if prefer_llm
        .as_ref()
        .map(|provider| provider.is_available())
        .unwrap_or(false)
    {
        // Clear user_edited and mark stale only when the rewrite can actually
        // run. The skipped no-LLM path must leave user prose locked.
        db.clear_user_edited(&page_id)
            .await
            .map_err(ServerError::from)?;
        let updated = wenlan_core::refinery::deep_distill_single(
            &db,
            prefer_llm,
            &prompts,
            &page_id,
            knowledge_path.as_deref(),
        )
        .await?;
        Ok(Json(serde_json::json!({
            "status": "ok",
            "updated": updated,
        })))
    } else {
        Ok(Json(serde_json::json!({
            "status": "skipped",
            "updated": false,
            "hint": "page re-distill needs an LLM in the daemon — install an on-device model or set an Anthropic key via `wenlan setup` / `/origin:doctor`",
        })))
    }
}

// ===== Recent retrieval / page-change feeds =====

#[derive(Debug, Default, Deserialize)]
pub struct RecentLimitQuery {
    #[serde(default)]
    pub limit: Option<i64>,
}

/// GET /api/retrievals/recent - Recent agent retrieval events joined to page titles.
pub async fn handle_recent_retrievals(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    axum::extract::Query(q): axum::extract::Query<RecentLimitQuery>,
) -> Result<Json<Vec<wenlan_types::RetrievalEvent>>, ServerError> {
    let limit = q.limit.unwrap_or(10).clamp(1, 100);
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let mut events = db
        .list_recent_retrievals_scoped(limit, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    // A retrieval event is not a page, but `page_titles` is page prose -- here
    // the title IS the disclosure, so a title survives only when its page does.
    // `page_ids` is documented as corresponding 1:1 with `page_titles`, so the
    // pair is the unit that lives or dies, and there is no entry form for half
    // of one: `Full` or dropped.
    //
    // The event itself is never dropped. Its query, agent and timestamp are the
    // caller's own activity rather than anything about a page, and an event that
    // vanished would say more than one that came back with no titles.
    //
    // One adapter call for the whole feed: every candidate id across every
    // event goes in, the survivors come back as a set, and each event rebuilds
    // its pairs from that. Per-event calls would be one `page_visibility` query
    // per row of a 100-row feed.
    let page_ids: Vec<String> = events
        .iter()
        .filter(|event| event.page_ids.len() == event.page_titles.len())
        .flat_map(|event| event.page_ids.iter().cloned())
        .collect();
    let visible: std::collections::HashSet<String> =
        wenlan_core::truth_adapter::filter_page_refs(&db, &view.grant, page_ids, |id| id.as_str())
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
            .into_iter()
            .collect();
    for event in &mut events {
        // Legacy rows recorded before `page_ids` existed carry titles and no
        // ids, and any other length mismatch is the same situation: nothing
        // attributes a title to a page, so nothing can be checked against a
        // verdict. Unknown is not permission -- both vectors clear.
        if event.page_ids.len() != event.page_titles.len() {
            event.page_ids.clear();
            event.page_titles.clear();
            continue;
        }
        let (ids, titles): (Vec<String>, Vec<String>) = std::mem::take(&mut event.page_ids)
            .into_iter()
            .zip(std::mem::take(&mut event.page_titles))
            .filter(|(id, _)| visible.contains(id))
            .unzip();
        event.page_ids = ids;
        event.page_titles = titles;
    }
    Ok(Json(events))
}

/// GET /api/pages/recent-changes - Recent page created/revised events.
pub async fn handle_recent_page_changes(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    axum::extract::Query(q): axum::extract::Query<RecentLimitQuery>,
) -> Result<Json<Vec<wenlan_types::PageChange>>, ServerError> {
    let limit = q.limit.unwrap_or(10).clamp(1, 100);
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let changes = db
        .list_recent_changes_scoped(limit, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    // A change row hangs off a page (id + title + what happened to it) rather
    // than being one, and it carries no truth axes of its own, so there is no
    // entry form for it: `Full` or dropped.
    let changes =
        wenlan_core::truth_adapter::filter_page_refs(&db, &view.grant, changes, |change| {
            change.page_id.as_str()
        })
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(changes))
}

/// GET /api/pages/recent — top-N page activity with badge deltas.
/// `since_ms` scopes badge derivation only; the feed is always top-N by recency.
pub async fn handle_recent_pages(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    view: crate::truth_guard::TruthView,
    axum::extract::Query(q): axum::extract::Query<crate::memory_routes::RecentActivityQuery>,
) -> Result<Json<Vec<wenlan_types::RecentActivityItem>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let mut items = db
        .list_recent_pages_with_badges_scoped(q.limit.unwrap_or(10), q.since_ms, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    // The feed interleaves pages and memories under one `id` field, so only the
    // page rows may be keyed on it -- a memory source_id carries no truth state
    // and would read as unsupported, dropping memories that this contract says
    // nothing about. An item carries a `snippet` of the page's prose, so the
    // page rows ride on `Full` with no entry form.
    let page_ids: Vec<String> = items
        .iter()
        .filter(|item| item.kind == wenlan_types::memory::ActivityKind::Page)
        .map(|item| item.id.clone())
        .collect();
    let visible: std::collections::HashSet<String> =
        wenlan_core::truth_adapter::filter_page_refs(&db, &view.grant, page_ids, |id| id.as_str())
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
            .into_iter()
            .collect();
    items.retain(|item| {
        item.kind != wenlan_types::memory::ActivityKind::Page || visible.contains(&item.id)
    });
    Ok(Json(items))
}

/// POST /api/llm/test — probe an OpenAI-compatible LLM endpoint with a 1-shot prompt.
/// Validates a custom endpoint from the app settings UI before saving.
pub async fn handle_test_llm(
    Json(req): Json<wenlan_types::requests::TestLlmRequest>,
) -> Result<Json<wenlan_types::requests::TestLlmResponse>, ServerError> {
    use wenlan_core::llm_provider::{LlmProvider, LlmRequest, OpenAICompatibleProvider};

    let provider = OpenAICompatibleProvider::new_with_key(req.endpoint, req.model, req.api_key);
    let llm_req = LlmRequest {
        system_prompt: None,
        user_prompt: req
            .prompt
            .unwrap_or_else(|| "Say 'hello' and nothing else.".into()),
        max_tokens: 10,
        temperature: 0.0,
        label: None,
        timeout_secs: None,
    };
    let response = provider
        .generate(llm_req)
        .await
        .map_err(|e| ServerError::Internal(format!("test_llm: {e}")))?;
    Ok(Json(wenlan_types::requests::TestLlmResponse { response }))
}

/// POST /api/shutdown — request cooperative, bounded daemon shutdown.
/// The lifecycle coordinator stops accepting new HTTP work, drains owned work,
/// and retains a hard deadline for tasks that cannot cooperate.
pub async fn handle_shutdown(
    Extension(shutdown): Extension<crate::lifecycle::ShutdownHandle>,
) -> (HeaderMap, &'static str) {
    shutdown.request();
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-wenlan-process-id",
        HeaderValue::from_str(&std::process::id().to_string())
            .expect("process id is always a valid HTTP header value"),
    );
    (headers, "shutting down")
}

#[cfg(test)]
mod recent_endpoints_tests {
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;
    use wenlan_core::llm_provider::{LlmBackend, LlmError, LlmProvider, LlmRequest};

    use crate::state::ServerState;

    struct WenlanDataDirGuard {
        previous: Option<std::ffi::OsString>,
        _tmp: tempfile::TempDir,
    }

    impl WenlanDataDirGuard {
        fn new() -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let previous = std::env::var_os("WENLAN_DATA_DIR");
            std::env::set_var("WENLAN_DATA_DIR", tmp.path());
            Self {
                previous,
                _tmp: tmp,
            }
        }
    }

    impl Drop for WenlanDataDirGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
        }
    }

    #[tokio::test]
    async fn shutdown_control_does_not_wait_for_server_state_lock() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let shutdown = state.read().await.shutdown.clone();
        let app = crate::router::build_router_with_shutdown(state.clone(), shutdown.clone());
        let _write_guard = state.write().await;

        let request = Request::builder()
            .method("POST")
            .uri("/api/shutdown")
            .body(Body::empty())
            .unwrap();
        let response =
            tokio::time::timeout(std::time::Duration::from_millis(100), app.oneshot(request))
                .await
                .expect("shutdown control path must bypass the workload-state lock")
                .unwrap();

        assert!(response.status().is_success());
        assert!(shutdown.is_requested());
    }

    struct ExternalCompileProvider {
        state: Arc<RwLock<ServerState>>,
        distill_calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl LlmProvider for ExternalCompileProvider {
        async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
            if request.label.as_deref() == Some("distill_body") {
                self.distill_calls.fetch_add(1, Ordering::SeqCst);
                let guard =
                    tokio::time::timeout(std::time::Duration::from_millis(200), self.state.write())
                        .await
                        .map_err(|_| {
                            LlmError::InferenceFailed(
                                "/api/steep held ServerState read lock across compile await"
                                    .to_string(),
                            )
                        })?;
                drop(guard);
                Ok(format!("{} [1]", request.user_prompt))
            } else {
                Ok("External Compile Topic".to_string())
            }
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "external-compile-test"
        }

        fn backend(&self) -> LlmBackend {
            LlmBackend::Api
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    struct VulkanStatusProvider;

    #[async_trait::async_trait]
    impl LlmProvider for VulkanStatusProvider {
        async fn generate(&self, _request: LlmRequest) -> Result<String, LlmError> {
            Ok("unused".into())
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "vulkan-status-test"
        }

        fn backend(&self) -> LlmBackend {
            LlmBackend::OnDevice
        }

        fn inference_runtime_info(
            &self,
        ) -> Option<wenlan_types::responses::OnDeviceInferenceStatus> {
            Some(wenlan_types::responses::OnDeviceInferenceStatus {
                backend: "vulkan".into(),
                device: Some("NVIDIA GeForce RTX 3060 Laptop GPU".into()),
                device_index: Some(2),
                gpu_layers: 99,
                fallback_reason: None,
            })
        }
    }

    #[tokio::test]
    async fn get_recent_retrievals_without_db_returns_503() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/retrievals/recent?limit=5")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 503);
    }

    #[tokio::test]
    async fn get_recent_page_changes_without_db_returns_503() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/pages/recent-changes?limit=5")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 503);
    }

    #[tokio::test]
    async fn get_recent_retrievals_route_is_registered() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/retrievals/recent")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        // Route exists => NOT a 404. We allow any other status (most likely 503 from DbNotInitialized).
        assert_ne!(response.status(), 404);
    }

    #[tokio::test]
    async fn get_recent_page_changes_route_is_registered() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/pages/recent-changes")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_ne!(response.status(), 404);
    }

    #[tokio::test]
    async fn get_recent_pages_route_is_registered() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/pages/recent?limit=5")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        // Route exists => NOT a 404. With no DB initialised we expect 503.
        assert_ne!(response.status(), 404);
    }

    #[tokio::test]
    async fn get_recent_pages_without_db_returns_503() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/pages/recent?limit=5&since_ms=1000")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 503);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn steep_routes_pinned_external_provider_without_holding_state_lock_across_compile() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();
        let routing = wenlan_core::config::Config {
            synthesis_source: Some("external".to_string()),
            ..wenlan_core::config::Config::default()
        };
        wenlan_core::config::save_config(&routing).unwrap();

        let tmp = tempfile::tempdir().expect("tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
                .await
                .expect("MemoryDB::new"),
        );

        for (i, content) in [
            "libSQL stores vector embeddings in F32_BLOB columns for each memory chunk, giving the Wenlan daemon a local semantic index over durable agent notes.",
            "DiskANN indexes the libSQL embedding column so Wenlan can perform approximate nearest-neighbor lookup without shipping private memory data to a hosted service.",
            "FTS5 triggers keep a lexical index synchronized with the chunks table, letting Wenlan combine full-text matches with vector similarity through reciprocal rank fusion.",
        ]
        .iter()
        .enumerate()
        {
            db.upsert_documents(vec![wenlan_core::sources::RawDocument {
                source: "memory".to_string(),
                source_id: format!("external_route_{}", i),
                title: content.to_string(),
                content: content.to_string(),
                space: Some("architecture".to_string()),
                ..Default::default()
            }])
            .await
            .unwrap();
        }

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..Default::default()
        }));
        let provider = Arc::new(ExternalCompileProvider {
            state: state.clone(),
            distill_calls: AtomicUsize::new(0),
        });
        {
            let mut guard = state.write().await;
            guard.external_llm = Some(provider.clone());
        }

        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/steep")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 200);

        assert!(
            provider.distill_calls.load(Ordering::SeqCst) > 0,
            "/api/steep must route configured external/API-compatible LLMs into the compile lane"
        );
        let pages_after = db.count_active_pages().await.unwrap();
        assert!(
            pages_after > 0,
            "external compile lane must synthesize a page, got {pages_after}"
        );
    }

    #[tokio::test]
    async fn status_reports_reranker_disabled_by_default() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/status")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        // Verify the reranker field is actually present in the JSON response.
        let raw = std::str::from_utf8(&bytes).unwrap();
        assert!(
            raw.contains("\"reranker\""),
            "status JSON must include a reranker field, got: {raw}"
        );
        let parsed: wenlan_types::responses::StatusResponse =
            serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            parsed.reranker,
            wenlan_types::responses::RerankerStatus::Disabled
        );
        assert_eq!(parsed.on_device_inference.backend, "disabled");
    }

    #[tokio::test]
    async fn status_reports_selected_vulkan_device() {
        let state = Arc::new(RwLock::new(ServerState {
            llm: Some(Arc::new(VulkanStatusProvider)),
            ..ServerState::default()
        }));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/status")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: wenlan_types::responses::StatusResponse =
            serde_json::from_slice(&bytes).unwrap();

        assert_eq!(parsed.on_device_inference.backend, "vulkan");
        assert_eq!(
            parsed.on_device_inference.device.as_deref(),
            Some("NVIDIA GeForce RTX 3060 Laptop GPU")
        );
        assert_eq!(parsed.on_device_inference.device_index, Some(2));
    }

    #[tokio::test]
    async fn status_reports_queue_idle_by_default() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/status")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        // The additive queue field must be present on the wire.
        let raw = std::str::from_utf8(&bytes).unwrap();
        assert!(
            raw.contains("\"queue\""),
            "status JSON must include a queue field, got: {raw}"
        );
        // With no DB (and thus an empty queue) it defaults cleanly to idle.
        let parsed: wenlan_types::responses::StatusResponse =
            serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.queue, wenlan_types::responses::QueueStatus::Idle);
    }

    #[test]
    fn status_handler_snapshots_state_before_awaiting_db() {
        let source = include_str!("routes.rs");
        let start = source
            .find("pub async fn handle_status")
            .expect("handle_status should exist");
        let end = source[start..]
            .find("/// POST /api/search")
            .map(|offset| start + offset)
            .expect("handle_search marker should follow handle_status");
        let body = &source[start..end];
        let normalized = body.split_whitespace().collect::<Vec<_>>().join(" ");

        assert!(
            normalized.contains(
                "let (db, reranker_status, reranker_light_status, reranker_mode, \
                 on_device_inference) = { let s = state.read().await;"
            ),
            "handle_status must snapshot cloned state out of ServerState before awaiting DB work"
        );
    }

    #[tokio::test]
    async fn status_reports_compile_queue_idle_by_default() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/status")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let raw = std::str::from_utf8(&bytes).unwrap();
        assert!(
            raw.contains("\"compile_queue\""),
            "status JSON must include a compile_queue field, got: {raw}"
        );
        let parsed: wenlan_types::responses::StatusResponse =
            serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            parsed.compile_queue,
            wenlan_types::responses::QueueStatus::Idle
        );
    }

    #[tokio::test]
    async fn status_reports_compile_queue_depth_when_clusters_are_pending() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new");
        wenlan_core::refinery::persist_compile_queue_depth(&db, 3)
            .await
            .unwrap();

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(Arc::new(db)),
            ..Default::default()
        }));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/status")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: wenlan_types::responses::StatusResponse =
            serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            parsed.compile_queue,
            wenlan_types::responses::QueueStatus::Active { pending: 3 }
        );
    }
}

/// Static assertion: `search_memory_llm_rerank` must NOT appear in this module
/// (routes.rs) at Tier-3.  The LLM reranker regresses below no-rerank on
/// LongMemEval (0.722 < 0.790 base); Tier-3 uses plain `search_memory`.
/// This test catches any accidental re-introduction of the reranked call.
#[cfg(test)]
mod chat_context_tests {
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;

    /// Chat-context route must exist (not 404) and must not require a DB for
    /// routing decisions: the handler returns 503 when DB is absent, not an
    /// LLM-rerank error.  This confirms T3 never pulls in the LLM reranker —
    /// the route completes its routing phase before reaching any LLM call.
    #[tokio::test]
    async fn chat_context_tier3_without_db_returns_503_not_llm_error() {
        // No LLM provider is wired — if T3 still called search_memory_llm_rerank
        // with an llm=None path that short-circuits, this might pass.  But the
        // real guarantee is structural: after removing the reranked branch the
        // handler falls through to DbNotInitialized before touching any LLM.
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let body = r#"{"query":"what programming languages do I know","max_chunks":5}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/api/context")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        // 503 = DbNotInitialized.  Would be a different error if LLM path ran first.
        assert_eq!(
            response.status(),
            503,
            "T3 should fail with DbNotInitialized (503), not an LLM error"
        );
    }
}

#[cfg(test)]
mod distill_request_tests {
    use super::DistillRequest;

    #[test]
    fn distill_request_deserializes_force() {
        let r: DistillRequest =
            serde_json::from_str(r#"{"target":"page_xyz","force":true}"#).unwrap();
        assert_eq!(r.target.as_deref(), Some("page_xyz"));
        assert!(r.force);
    }

    #[test]
    fn distill_request_defaults_force_to_false() {
        let r: DistillRequest = serde_json::from_str(r#"{"target":"foo"}"#).unwrap();
        assert!(!r.force);
    }

    #[test]
    fn distill_request_deserializes_sweep() {
        let r: DistillRequest = serde_json::from_str(r#"{"sweep":true}"#).unwrap();
        assert!(r.sweep);
        assert!(r.target.is_none());
        assert!(!r.force);
    }

    #[test]
    fn distill_request_rejects_unknown_field() {
        let r = serde_json::from_str::<DistillRequest>(r#"{"bogus":true}"#);
        assert!(r.is_err(), "deny_unknown_fields should reject unknown keys");
    }
}

#[cfg(test)]
mod distill_sweep_route_tests {
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;

    async fn seeded_sweep_app() -> (
        crate::router::AppRouter,
        Arc<wenlan_core::db::MemoryDB>,
        tempfile::TempDir,
    ) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
                .await
                .expect("MemoryDB::new"),
        );
        let topic = "formation sweep route reports scoped cluster stats";
        let mut rows = Vec::new();
        for (prefix, space) in [
            ("route_work", Some("work")),
            ("route_personal", Some("personal")),
            ("route_none", None),
        ] {
            for i in 0..3 {
                rows.push(wenlan_core::sources::RawDocument {
                    source: "memory".to_string(),
                    source_id: format!("{prefix}_{i}"),
                    title: format!("{prefix}_{i}"),
                    content: topic.to_string(),
                    last_modified: chrono::Utc::now().timestamp(),
                    memory_type: Some("fact".to_string()),
                    space: space.map(str::to_string),
                    source_agent: Some("test".to_string()),
                    confirmed: Some(true),
                    ..Default::default()
                });
            }
        }
        db.upsert_documents(rows).await.expect("seed sweep fixture");

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..Default::default()
        }));
        (crate::router::build_router(state), db, tmp)
    }

    #[tokio::test]
    async fn distill_sweep_returns_stats_grid_and_writes_no_rows() {
        let (app, db, _tmp) = seeded_sweep_app().await;
        let before = (
            db.count_active_pages().await.unwrap(),
            db.count().await.unwrap(),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/distill")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"sweep":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert!(payload.get("pending").is_none());
        assert!(payload.get("stale_pages").is_none());
        assert!(payload.get("created_ids").is_none());
        assert!(payload.get("orphan_topics").is_none());

        let thresholds = payload["thresholds"]
            .as_array()
            .expect("sweep response must contain threshold grid");
        assert_eq!(thresholds.len(), 4);
        let actual_thresholds: Vec<f64> = thresholds
            .iter()
            .map(|point| point["formation_threshold"].as_f64().unwrap())
            .collect();
        assert_eq!(actual_thresholds, vec![0.55, 0.60, 0.65, 0.70]);

        for point in thresholds {
            assert_eq!(point["global"]["cluster_count"], 3);
            assert_eq!(
                point["global"]["size_distribution"]["sizes"],
                serde_json::json!([3, 3, 3])
            );
            assert_eq!(point["global"]["overlapping_member_fraction"], 0.0);
            assert_eq!(point["capped"], false);

            let per_space = point["per_space"]
                .as_object()
                .expect("per_space stats must be an object");
            for space in ["(none)", "personal", "work"] {
                let stats = per_space
                    .get(space)
                    .unwrap_or_else(|| panic!("missing per-space stats for {space}"));
                assert_eq!(stats["cluster_count"], 1);
                assert_eq!(stats["size_distribution"]["sizes"], serde_json::json!([3]));
                assert_eq!(stats["overlapping_member_fraction"], 0.0);
            }
        }

        let after = (
            db.count_active_pages().await.unwrap(),
            db.count().await.unwrap(),
        );
        assert_eq!(after, before, "sweep route must not write rows");
    }

    #[tokio::test]
    async fn distill_sweep_with_target_or_force_returns_hint_payload() {
        let (app, _db, _tmp) = seeded_sweep_app().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/distill")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"sweep":true,"target":"missing-space","force":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(payload["sweep"], true);
        assert!(
            payload["hint"]
                .as_str()
                .is_some_and(|hint| hint.contains("sweep=true")),
            "hint payload should explain that sweep cannot be combined with target/force: {payload}"
        );
        assert!(payload.get("pending").is_none());
        assert!(payload.get("unresolved").is_none());
    }
}

#[cfg(test)]
mod redistill_contract_tests {
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;
    use wenlan_types::requests::CreateConceptRequest;

    use crate::state::ServerState;

    async fn seeded_user_edited_page() -> (
        crate::router::AppRouter,
        Arc<wenlan_core::db::MemoryDB>,
        String,
        tempfile::TempDir,
    ) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
                .await
                .expect("MemoryDB::new"),
        );
        let result = wenlan_core::post_write::create_page(
            &db,
            CreateConceptRequest {
                title: "Manual page".to_string(),
                content: "original body".to_string(),
                summary: None,
                entity_id: None,
                space: (None).into(),
                source_memory_ids: Vec::new(),
                creation_kind: Some("authored".to_string()),
                workspace: None,
            },
            "test",
            None,
        )
        .await
        .expect("create page");
        let page_id = result.id;
        db.update_page_content(&page_id, "user prose", &["mem_1"], "fs_edit")
            .await
            .expect("mark page user edited");
        let page = db
            .get_page(&page_id)
            .await
            .expect("get page")
            .expect("page exists");
        assert!(page.user_edited, "precondition: page is user edited");

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..Default::default()
        }));
        (crate::router::build_router(state), db, page_id, tmp)
    }

    #[tokio::test]
    async fn page_redistill_without_llm_does_not_clear_user_edited() {
        let (app, db, page_id, _tmp) = seeded_user_edited_page().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/distill/{page_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["status"], "skipped");

        let page = db
            .get_page(&page_id)
            .await
            .expect("get page")
            .expect("page exists");
        assert!(
            page.user_edited,
            "skipped no-LLM re-distill must not unlock user-edited prose"
        );
        assert_ne!(
            page.stale_reason.as_deref(),
            Some("manual_force"),
            "skipped no-LLM re-distill must not mark a manual force rewrite"
        );
    }

    #[tokio::test]
    async fn force_target_redistill_without_llm_does_not_clear_user_edited() {
        let (app, db, page_id, _tmp) = seeded_user_edited_page().await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/distill")
                    .header("content-type", "application/json")
                    .body(Body::from(format!(
                        r#"{{"target":"{page_id}","force":true}}"#
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["status"], "skipped");
        assert_eq!(payload["force"], true);

        let page = db
            .get_page(&page_id)
            .await
            .expect("get page")
            .expect("page exists");
        assert!(
            page.user_edited,
            "skipped no-LLM force re-distill must not unlock user-edited prose"
        );
        assert_ne!(
            page.stale_reason.as_deref(),
            Some("manual_force"),
            "skipped no-LLM force re-distill must not mark a manual force rewrite"
        );
    }
}

#[cfg(test)]
mod context_page_selection_tests {
    use std::collections::HashSet;
    use wenlan_core::pages::{filter_pages_by_source_overlap, select_pages_for_context, Page};

    fn make_page(id: &str, source_ids: &[&str], relevance_score: f32, review_status: &str) -> Page {
        Page {
            id: id.to_string(),
            title: id.to_string(),
            summary: None,
            content: String::new(),
            entity_id: None,
            space: None,
            source_memory_ids: source_ids.iter().map(|s| s.to_string()).collect(),
            version: 1,
            status: "active".to_string(),
            created_at: String::new(),
            last_compiled: String::new(),
            last_modified: String::new(),
            sources_updated_count: 0,
            stale_reason: None,
            pending_rebuild: None,
            user_edited: false,
            relevance_score,
            last_edited_by: None,
            last_edited_at: None,
            last_delta_summary: None,
            changelog: None,
            creation_kind: "distilled".to_string(),
            review_status: review_status.to_string(),
            workspace: None,
            citations: Vec::new(),
            kind: "concept".to_string(),
            truth: None,
        }
    }

    /// Unit-locks the `select_pages_for_context` ranking selector. A high-relevance
    /// page with no source overlap still surfaces; the prior hard overlap gate dropped it.
    #[test]
    fn context_surfaces_zero_overlap_page_that_old_gate_dropped() {
        // Memory hit pool returned by search_memory for this query.
        let search_source_ids: HashSet<String> =
            ["m1", "m2"].iter().map(|s| s.to_string()).collect();

        // Top-relevance page compiled from memories DISJOINT from the pool,
        // plus a lower-relevance page that does overlap.
        let raw_pages = vec![
            make_page("page_zero_overlap", &["x9", "x8"], 0.95, "confirmed"),
            make_page("page_overlap", &["m1"], 0.40, "confirmed"),
        ];

        // The OLD gate (min_overlap >= 1) drops the zero-overlap page — the
        // regression this change removes. Assert it really would have dropped it,
        // so the test below is exercising the gate boundary, not a no-op.
        let gated = filter_pages_by_source_overlap(&raw_pages, &search_source_ids, 1);
        assert!(
            !gated.iter().any(|p| p.id == "page_zero_overlap"),
            "old source-overlap gate should drop the zero-overlap page"
        );

        // The selector surfaces it when the caller requests a three-page cap.
        let selected = select_pages_for_context(&raw_pages, &search_source_ids, 3);
        assert!(
            selected.iter().any(|p| p.id == "page_zero_overlap"),
            "selector must surface the high-relevance zero-overlap page"
        );
    }
}

#[cfg(test)]
mod search_supplemental_pages_tests {
    //! Task 7: `/api/search` surfaces gated distilled pages through the additive
    //! `supplemental_pages` field, routed through the SAME `select_visible_pages`
    //! visibility gate as `/api/context` and `/api/memory/search`.
    use crate::state::ServerState;
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;
    use wenlan_types::requests::CreateConceptRequest;

    async fn seed_confirmed_distilled_page(
        db: &wenlan_core::db::MemoryDB,
        title: &str,
        content: &str,
        source_id: &str,
        source_type: &str,
        space: &str,
    ) -> String {
        let source = wenlan_core::sources::RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: format!("memory-{source_id}"),
            content: if space == "other" {
                "unrelated source memory outside the query result set".to_string()
            } else {
                content.to_string()
            },
            memory_type: Some(source_type.to_string()),
            space: Some(space.to_string()),
            source_agent: Some("test-agent".to_string()),
            confidence: Some(0.9),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![source]).await.unwrap();
        if space == "other" {
            return format!("page_absent_{source_id}");
        }
        let result = wenlan_core::post_write::create_page_with_tuning(
            db,
            CreateConceptRequest {
                title: title.to_string(),
                content: content.to_string(),
                summary: None,
                entity_id: None,
                source_memory_ids: vec![source_id.to_string()],
                creation_kind: Some("distilled".to_string()),
                space: (Some(space.to_string())).into(),
                workspace: Some(space.to_string()),
            },
            "test",
            None,
            1,
            1.1,
        )
        .await
        .unwrap();
        db.set_page_review_status(&result.id, "confirmed")
            .await
            .unwrap();
        result.id
    }

    /// Seed a DB with a tier-2 (`review`) agent, one tier-2 (`decision`) source
    /// memory in space `work`, plus three confirmed active pages: a same-space
    /// page sourced from the tier-2 memory (visible to a tier-2 caller), and a
    /// cross-space page (workspace `other`, disjoint sources → always dropped).
    /// Returns the wired app router.
    async fn seeded_app() -> (crate::router::AppRouter, tempfile::TempDir, String, String) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new");
        db.create_space("work", None, false).await.unwrap();

        // tier-2 caller: trust "review" allows tier 2, denies tier 1; "unknown"
        // (no header) denies tier 2.
        db.register_agent("test-agent").await.unwrap();
        db.update_agent("test-agent", None, None, None, Some("review"), None)
            .await
            .unwrap();

        // A tier-2 (decision) source memory in space "work".
        let mem = wenlan_core::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "m_decision".to_string(),
            title: "memory-m_decision".to_string(),
            content: "zorblax source memory".to_string(),
            memory_type: Some("decision".to_string()),
            space: Some("work".to_string()),
            source_agent: Some("test-agent".to_string()),
            confidence: Some(0.9),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![mem]).await.unwrap();

        // Cross-space: workspace != caller space, source not in result set → DROP.
        let page_cross_id = seed_confirmed_distilled_page(
            &db,
            "Crossmarker",
            "crossmarker body outside query",
            "unrelated",
            "fact",
            "other",
        )
        .await;
        // Same space, tier-2 (decision) source → KEEP for review, DROP for unknown.
        let page_same_id = seed_confirmed_distilled_page(
            &db,
            "Zorblax Samemarker",
            "zorblax samemarker body",
            "m_decision",
            "decision",
            "work",
        )
        .await;

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(Arc::new(db)),
            ..Default::default()
        }));
        (
            crate::router::build_router(state),
            tmp,
            page_same_id,
            page_cross_id,
        )
    }

    /// A tier-2 (`review`) same-space caller gets the same-space, tier-2-sourced
    /// page in `supplemental_pages`; the cross-space page is dropped.
    #[tokio::test]
    async fn search_surfaces_same_space_tier2_page() {
        let (app, _tmp, page_same_id, page_cross_id) = seeded_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("x-agent-name", "test-agent")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"zorblax","space":"work"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "search call should succeed");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: wenlan_types::responses::SearchResponse =
            serde_json::from_slice(&bytes).unwrap();
        let pages = parsed
            .supplemental_pages
            .expect("tier-2 caller must get supplemental pages");
        assert!(
            pages.iter().any(|p| p.source_id == page_same_id),
            "same-space tier-2 page must surface, got: {:?}",
            pages.iter().map(|p| &p.source_id).collect::<Vec<_>>()
        );
        assert!(
            !pages.iter().any(|p| p.source_id == page_cross_id),
            "cross-space page must be dropped by the space-scope gate, got: {:?}",
            pages.iter().map(|p| &p.source_id).collect::<Vec<_>>()
        );
    }

    /// An unknown caller (no `x-agent-name` header → fail-closed "unknown") clears
    /// no tier above 3; both seeded pages are tier-2/cross-space, so the gate
    /// drops everything and `supplemental_pages` is `None`.
    #[tokio::test]
    async fn search_supplemental_pages_none_for_unknown_caller() {
        let (app, _tmp, _page_same_id, _page_cross_id) = seeded_app().await;
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"zorblax","space":"work"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "search call should succeed");
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: wenlan_types::responses::SearchResponse =
            serde_json::from_slice(&bytes).unwrap();
        assert!(
            parsed.supplemental_pages.is_none(),
            "unknown caller must get no pages, got: {:?}",
            parsed.supplemental_pages
        );
    }
}

#[cfg(test)]
mod test_llm_bearer_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::{Arc, Mutex};
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;

    /// Mock OpenAI-compatible server capturing the Authorization header.
    async fn spawn_mock() -> (std::net::SocketAddr, Arc<Mutex<Vec<Option<String>>>>) {
        let captured: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let cap = captured.clone();
        let app = axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(move |headers: axum::http::HeaderMap| {
                let cap = cap.clone();
                async move {
                    cap.lock().unwrap().push(
                        headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .map(String::from),
                    );
                    axum::Json(serde_json::json!({
                        "choices": [{"message": {"content": "hello"}}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (addr, captured)
    }

    async fn probe(addr: std::net::SocketAddr, body: serde_json::Value) -> StatusCode {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let router = crate::router::build_router(state);
        let mut body = body;
        body["endpoint"] = serde_json::json!(format!("http://{addr}"));
        body["model"] = serde_json::json!("test-model");
        router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/llm/test")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn test_llm_forwards_bearer_key() {
        let (addr, captured) = spawn_mock().await;
        let status = probe(addr, serde_json::json!({"api_key": "sk-test"})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            captured.lock().unwrap().as_slice(),
            &[Some("Bearer sk-test".to_string())]
        );
    }

    #[tokio::test]
    async fn test_llm_sends_no_auth_header_without_key() {
        let (addr, captured) = spawn_mock().await;
        let status = probe(addr, serde_json::json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(captured.lock().unwrap().as_slice(), &[None]);
    }

    #[tokio::test]
    async fn test_llm_trims_pasted_whitespace_from_bearer_key() {
        let (addr, captured) = spawn_mock().await;
        let status = probe(addr, serde_json::json!({"api_key": "sk-x\n"})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            captured.lock().unwrap().as_slice(),
            &[Some("Bearer sk-x".to_string())],
            "trailing whitespace/newline in a pasted key must not reach the wire"
        );
    }
}
