// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::state::ServerState;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::Json,
};
use origin_core::router::classify::{classify_query, estimate_tokens, tier_allowed};
use origin_types::requests::ChatContextRequest;
use origin_types::responses::{
    ChatContextResponse, KnowledgeContext, ProfileContext, TierTokenEstimates,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

// ===== Request/Response Types =====

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    pub source_filter: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
}

fn default_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<origin_core::db::SearchResult>,
    pub took_ms: f64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub db_initialized: bool,
    pub version: String,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub is_running: bool,
    pub files_indexed: u64,
    pub files_total: u64,
    pub sources_connected: Vec<String>,
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
        version: origin_core::version().to_string(),
    }))
}

/// GET /api/status - Get indexing status
pub async fn handle_status(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<StatusResponse>, ServerError> {
    let s = state.read().await;

    let files_indexed = if let Some(db) = &s.db {
        db.count().await.unwrap_or(0)
    } else {
        0
    };

    Ok(Json(StatusResponse {
        is_running: true,
        files_indexed,
        files_total: 0,
        sources_connected: vec![],
    }))
}

/// POST /api/search - Semantic search endpoint
pub async fn handle_search(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ServerError> {
    let start = std::time::Instant::now();

    let results = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;

        if req.source_filter.as_deref() == Some("memory") {
            db.search_memory(
                &req.query,
                req.limit,
                None,
                req.domain.as_deref(),
                None,
                None,
                None,
                None,
            )
            .await
            .map_err(|e| ServerError::SearchFailed(e.to_string()))?
        } else {
            db.search(&req.query, req.limit, req.source_filter.as_deref())
                .await
                .map_err(|e| ServerError::SearchFailed(e.to_string()))?
        }
    };

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(SearchResponse { results, took_ms }))
}

// ===== Context Endpoints =====

#[derive(Debug, Deserialize)]
pub struct ContextRequest {
    pub current_file: String,
    pub cursor_prefix: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[derive(Debug, Serialize)]
pub struct ContextSuggestion {
    pub content: String,
    pub score: f32,
    pub source: String,
}

#[derive(Debug, Serialize)]
pub struct ContextResponse {
    pub suggestions: Vec<ContextSuggestion>,
    pub took_ms: f64,
}

/// POST /api/context - Context-aware autocomplete for VS Code
pub async fn handle_context(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<ContextRequest>,
) -> Result<Json<ContextResponse>, ServerError> {
    let start = std::time::Instant::now();

    let file_name = std::path::Path::new(&req.current_file)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&req.current_file);

    let query = format!("{} {}", file_name, req.cursor_prefix);

    let results = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;

        db.search(&query, req.limit, None)
            .await
            .map_err(|e| ServerError::SearchFailed(e.to_string()))?
    };

    let suggestions: Vec<ContextSuggestion> = results
        .into_iter()
        .map(|r| ContextSuggestion {
            content: r.content,
            score: r.score,
            source: r.source_id,
        })
        .collect();

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(ContextResponse {
        suggestions,
        took_ms,
    }))
}

/// POST /api/chat-context - Trust-gated tiered memory retrieval for LLM context injection
pub async fn handle_chat_context(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    Json(req): Json<ChatContextRequest>,
) -> Result<Json<ChatContextResponse>, ServerError> {
    let start = std::time::Instant::now();

    let query = req
        .query
        .as_deref()
        .or(req.conversation_id.as_deref())
        .unwrap_or("recent context");

    let agent_name = headers
        .get("x-agent-name")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown");

    // Snapshot the Arc handles and drop the ServerState read guard BEFORE
    // the multi-second chain of DB + LLM awaits below. Holding the guard
    // across ~15 sequential awaits (load_memories_by_type, search_*,
    // search_memory_reranked, log_accesses, etc.) would block every writer
    // to ServerState — e.g. store_memory's space_store update — for the
    // full duration of a rerank call. See CLAUDE.md locking rules.
    let (db_arc, llm, access_tracker, concept_min_overlap) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        (
            db,
            s.llm.clone(),
            s.access_tracker.clone(),
            s.tuning.distillation.concept_min_overlap,
        )
    }; // guard dropped here
    let db = db_arc.as_ref();

    let agent_trust = if agent_name == "unknown" {
        "unknown".to_string()
    } else {
        db.get_agent(agent_name)
            .await
            .ok()
            .flatten()
            .map(|a| a.trust_level)
            .unwrap_or_else(|| "unknown".to_string())
    };

    let classification = classify_query(query, agent_name, &agent_trust, true);
    let domain_filter = req.domain.as_deref().or(classification.space.as_deref());

    // Tier 1 (identity + preferences)
    let (identity, preferences) = if tier_allowed(&classification.trust_level, 1) {
        let mut id_mems = db
            .load_memories_by_type("identity", 10, domain_filter)
            .await
            .unwrap_or_default();
        let mut pref_mems = db
            .load_memories_by_type("preference", 10, domain_filter)
            .await
            .unwrap_or_default();

        if id_mems.is_empty() && domain_filter.is_some() {
            id_mems = db
                .load_memories_by_type("identity", 5, None)
                .await
                .unwrap_or_default();
        }
        if pref_mems.is_empty() && domain_filter.is_some() {
            pref_mems = db
                .load_memories_by_type("preference", 5, None)
                .await
                .unwrap_or_default();
        }

        (
            id_mems
                .iter()
                .map(|m| m.content.clone())
                .collect::<Vec<_>>(),
            pref_mems
                .iter()
                .map(|m| m.content.clone())
                .collect::<Vec<_>>(),
        )
    } else {
        (Vec::new(), Vec::new())
    };

    // Tier 2 (goals + corrections)
    let goals = if tier_allowed(&classification.trust_level, 2) && req.include_goals {
        db.load_memories_by_type("goal", 5, domain_filter)
            .await
            .unwrap_or_default()
            .iter()
            .map(|m| m.content.clone())
            .collect()
    } else {
        Vec::new()
    };

    let decisions: Vec<String> = if tier_allowed(&classification.trust_level, 2) {
        db.load_memories_by_type("decision", 5, domain_filter)
            .await
            .unwrap_or_default()
            .iter()
            .map(|m| m.content.clone())
            .collect()
    } else {
        Vec::new()
    };

    let corrections = if tier_allowed(&classification.trust_level, 2) && query != "recent context" {
        db.search_corrections_by_topic(query, 5)
            .await
            .unwrap_or_default()
            .iter()
            .map(|r| r.content.clone())
            .collect()
    } else {
        Vec::new()
    };

    // Tier 3 (search)
    let search_results = if classification.use_graph {
        db.search_memory_reranked(
            query,
            req.max_chunks,
            None,
            domain_filter,
            None,
            llm.clone(),
        )
        .await
        .unwrap_or_default()
    } else {
        db.search_memory(
            query,
            req.max_chunks,
            None,
            domain_filter,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap_or_default()
    };

    let threshold = req.relevance_threshold.unwrap_or(0.0) as f32;
    let filtered_search: Vec<_> = search_results
        .into_iter()
        .filter(|r| r.score >= threshold)
        .collect();

    let graph_observations: Vec<String> = filtered_search
        .iter()
        .filter(|r| r.source == "knowledge_graph")
        .map(|r| format!("[{}] {}", r.title, r.content))
        .collect();

    // Source IDs from search results — used to gate concept relevance.
    // A concept is only included if its source memories overlap with the
    // memories that search_memory returned for this query.
    let search_source_ids: std::collections::HashSet<String> = filtered_search
        .iter()
        .map(|r| r.source_id.clone())
        .collect();

    let concept_results: Vec<String> =
        if tier_allowed(&classification.trust_level, 2) && query != "recent context" {
            let raw_concepts = db.search_concepts(query, 3).await.unwrap_or_default();
            let concepts = origin_core::concepts::filter_concepts_by_source_overlap(
                &raw_concepts,
                &search_source_ids,
                concept_min_overlap,
            );
            concepts
                .iter()
                .map(|c| {
                    let summary = c.summary.as_deref().unwrap_or("");
                    format!("**{}**: {}\n{}", c.title, summary, c.content)
                })
                .collect()
        } else {
            Vec::new()
        };

    let narrative = db
        .get_cached_narrative()
        .await
        .ok()
        .flatten()
        .filter(|(content, _, _)| !content.is_empty())
        .map(|(content, _, _)| content)
        .unwrap_or_default();
    let narrative_brief: Option<&str> = if narrative.is_empty() {
        None
    } else {
        Some(&narrative)
    };

    // Token estimates
    let brief_text = narrative_brief.unwrap_or("");
    let tier1_text = format!(
        "{} {} {}",
        brief_text,
        identity.join(" "),
        preferences.join(" ")
    );
    let tier2_text = format!(
        "{} {} {} {}",
        goals.join(" "),
        corrections.join(" "),
        decisions.join(" "),
        concept_results.join(" ")
    );
    let tier3_text = filtered_search
        .iter()
        .map(|r| r.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let t1 = estimate_tokens(&tier1_text);
    let t2 = estimate_tokens(&tier2_text);
    let t3 = estimate_tokens(&tier3_text);
    let token_estimates = TierTokenEstimates {
        tier1_identity: t1,
        tier2_project: t2,
        tier3_relevant: t3,
        total: t1 + t2 + t3,
    };

    // Build combined context string
    let mut sections: Vec<String> = Vec::new();

    if tier_allowed(&classification.trust_level, 1) {
        if let Some(brief) = narrative_brief {
            sections.push(format!("## About the User\n{}\n", brief));
        }
    }

    if !identity.is_empty() {
        let mut s = String::from("## Identity\n");
        for item in &identity {
            s.push_str(&format!("- {}\n", item));
        }
        sections.push(s);
    }

    if !preferences.is_empty() {
        let mut s = String::from("## Preferences\n");
        for item in &preferences {
            s.push_str(&format!("- {}\n", item));
        }
        sections.push(s);
    }

    if !goals.is_empty() {
        let mut s = String::from("## Goals\n");
        for item in &goals {
            s.push_str(&format!("- {}\n", item));
        }
        sections.push(s);
    }

    if !decisions.is_empty() {
        let mut sec = String::from("## Relevant Decisions\n");
        for item in &decisions {
            sec.push_str(&format!("- {}\n", item));
        }
        sections.push(sec);
    }

    if !corrections.is_empty() {
        let mut sec = String::from("## Corrections\n");
        for item in &corrections {
            sec.push_str(&format!("- {}\n", item));
        }
        sections.push(sec);
    }

    if !concept_results.is_empty() {
        let mut sec = String::from("## Compiled Knowledge\n");
        for item in &concept_results {
            sec.push_str(&format!("{}\n\n---\n\n", item));
        }
        sections.push(sec);
    }

    if !graph_observations.is_empty() {
        let mut sec = String::from("## Knowledge Graph\n");
        for item in &graph_observations {
            sec.push_str(&format!("- {}\n", item));
        }
        sections.push(sec);
    }

    if !filtered_search.is_empty() {
        let mut sec = String::from("## Relevant Memories\n");
        for r in &filtered_search {
            sec.push_str(&format!("[{}] {}\n\n", r.title, r.content));
        }
        sections.push(sec);
    }

    let context = sections.join("\n");

    // Track accesses
    let all_source_ids: Vec<String> = filtered_search
        .iter()
        .map(|r| r.source_id.clone())
        .collect();
    if !all_source_ids.is_empty() {
        access_tracker.record_accesses(&all_source_ids);
        if let Err(e) = db.log_accesses(&all_source_ids).await {
            tracing::warn!("Failed to log accesses: {}", e);
        }
    }

    if !all_source_ids.is_empty() {
        let detail = format!("used {} memories", all_source_ids.len());
        if let Err(e) = db
            .log_agent_activity(
                agent_name,
                "read",
                &all_source_ids,
                req.query.as_deref(),
                &detail,
            )
            .await
        {
            tracing::warn!("Failed to log agent read activity: {}", e);
        }
    }

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Fire-once onboarding milestone check (recall side).
    //
    // Only fires when the caller identified itself via `x-agent-name`. The
    // evaluator skips empty-result recalls internally, so we pass the real
    // hit count from `filtered_search` (the post-threshold search results
    // surfaced back to the agent as `knowledge.relevant_memories`). The
    // daemon has no UI to notify, so a fresh `NoopEmitter` is used inline —
    // milestones are still persisted via `record_milestone` and surfaced
    // to the frontend through /api/onboarding/*.
    if agent_name != "unknown" {
        let agent_for_ms = agent_name.to_string();
        let results_count = filtered_search.len();
        // Quote the top-ranked hit so the first-recall toast can show WHAT
        // was just surfaced, not just who asked. Char-safe truncation
        // (UTF-8 rule: never byte-index a Rust string).
        let top_preview_for_ms: Option<String> = filtered_search.first().map(|r| {
            let s = &r.content;
            let truncated: String = s.chars().take(100).collect();
            if truncated.chars().count() < s.chars().count() {
                format!("{}…", truncated.trim_end())
            } else {
                truncated
            }
        });
        let db_for_ms = db_arc.clone();
        let emitter_for_ms: Arc<dyn origin_core::events::EventEmitter> =
            Arc::new(origin_core::events::NoopEmitter);
        tokio::spawn(async move {
            let ev = origin_core::onboarding::MilestoneEvaluator::new(&db_for_ms, emitter_for_ms);
            if let Err(e) = ev
                .check_after_context_call(
                    &agent_for_ms,
                    results_count,
                    top_preview_for_ms.as_deref(),
                )
                .await
            {
                tracing::warn!(?e, "onboarding: check_after_context_call failed");
            }
        });
    }

    Ok(Json(ChatContextResponse {
        context,
        profile: ProfileContext {
            narrative,
            identity,
            preferences,
            goals,
        },
        knowledge: KnowledgeContext {
            concepts: concept_results,
            decisions,
            relevant_memories: filtered_search,
            graph_context: graph_observations,
        },
        took_ms,
        token_estimates,
    }))
}

/// GET /api/ping
pub async fn handle_ping() -> Result<(StatusCode, &'static str), ServerError> {
    Ok((StatusCode::OK, "pong"))
}

/// GET /api/debug/pipeline
pub async fn handle_pipeline_status(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
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
    let s = state.read().await;
    let db =
        s.db.as_ref()
            .ok_or(ServerError::Internal("DB not initialized".into()))?;
    let llm = s.llm.as_ref();
    let api_llm = s.api_llm.as_ref();
    let synthesis_llm = s.synthesis_llm.as_ref();
    let prompts = &s.prompts;
    let tuning = &s.tuning.refinery;
    let confidence_cfg = &s.tuning.confidence;
    let distillation_cfg = &s.tuning.distillation;
    let result = origin_core::refinery::run_periodic_steep_with_api(
        db,
        llm,
        api_llm,
        synthesis_llm,
        prompts,
        tuning,
        confidence_cfg,
        distillation_cfg,
        origin_core::refinery::TriggerKind::Backstop,
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
    pub phases: Vec<origin_core::refinery::PhaseResult>,
}

/// GET /api/eval/distillation
pub async fn handle_distillation_eval(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_core::eval::distillation_quality::DistillationQualityReport>, ServerError> {
    let s = state.read().await;
    let db =
        s.db.as_ref()
            .ok_or(ServerError::Internal("DB not initialized".into()))?;
    let report = origin_core::eval::distillation_quality::evaluate_distillation_quality(db)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(report))
}

/// POST /api/distill
pub async fn handle_distill(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db =
        s.db.as_ref()
            .ok_or(ServerError::Internal("DB not initialized".into()))?;
    let llm = s.llm.as_ref();
    let api_llm = s.api_llm.as_ref();
    let prompts = &s.prompts;
    let tuning = &s.tuning.distillation;

    let prefer_llm = api_llm.or(llm);
    let knowledge_path = {
        let config = origin_core::config::load_config();
        Some(config.knowledge_path_or_default())
    };
    let distilled = origin_core::refinery::distill_concepts(
        db,
        prefer_llm,
        prompts,
        tuning,
        knowledge_path.as_deref(),
    )
    .await?;
    let deep = origin_core::refinery::deep_distill_concepts(
        db,
        prefer_llm,
        prompts,
        tuning,
        knowledge_path.as_deref(),
    )
    .await?;

    Ok(Json(serde_json::json!({
        "concepts_created": distilled,
        "concepts_updated": deep,
    })))
}

/// POST /api/distill/{concept_id}
pub async fn handle_redistill(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(concept_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db =
        s.db.as_ref()
            .ok_or(ServerError::Internal("DB not initialized".into()))?;
    let llm = s.llm.as_ref();
    let api_llm = s.api_llm.as_ref();
    let prompts = &s.prompts;

    let prefer_llm = api_llm.or(llm);
    origin_core::refinery::deep_distill_single(db, prefer_llm, prompts, &concept_id).await?;

    Ok(Json(serde_json::json!({"status": "ok"})))
}

// ===== Recent retrieval / concept-change feeds =====

#[derive(Debug, Default, Deserialize)]
pub struct RecentLimitQuery {
    #[serde(default)]
    pub limit: Option<i64>,
}

/// GET /api/retrievals/recent - Recent agent retrieval events joined to concept titles.
pub async fn handle_recent_retrievals(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(q): axum::extract::Query<RecentLimitQuery>,
) -> Result<Json<Vec<origin_types::RetrievalEvent>>, ServerError> {
    let limit = q.limit.unwrap_or(10).clamp(1, 100);
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let events = db
        .list_recent_retrievals(limit)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(events))
}

/// GET /api/concepts/recent-changes - Recent concept created/revised events.
pub async fn handle_recent_concept_changes(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(q): axum::extract::Query<RecentLimitQuery>,
) -> Result<Json<Vec<origin_types::ConceptChange>>, ServerError> {
    let limit = q.limit.unwrap_or(10).clamp(1, 100);
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let changes = db
        .list_recent_changes(limit)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(changes))
}

/// GET /api/concepts/recent — top-N concept activity with badge deltas.
/// `since_ms` scopes badge derivation only; the feed is always top-N by recency.
pub async fn handle_recent_concepts(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(q): axum::extract::Query<crate::memory_routes::RecentActivityQuery>,
) -> Result<Json<Vec<origin_types::RecentActivityItem>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let items = db
        .list_recent_concepts_with_badges(q.limit.unwrap_or(10), q.since_ms)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(items))
}

/// POST /api/shutdown — exits the daemon process cleanly.
/// Returns 200 OK, then exits 0 after a brief delay so the response is delivered.
pub async fn handle_shutdown() -> &'static str {
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        std::process::exit(0);
    });
    "shutting down"
}

#[cfg(test)]
mod recent_endpoints_tests {
    use axum::body::Body;
    use axum::http::Request;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;

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
    async fn get_recent_concept_changes_without_db_returns_503() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/concepts/recent-changes?limit=5")
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
    async fn get_recent_concept_changes_route_is_registered() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/concepts/recent-changes")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_ne!(response.status(), 404);
    }

    #[tokio::test]
    async fn get_recent_concepts_route_is_registered() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/concepts/recent?limit=5")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        // Route exists => NOT a 404. With no DB initialised we expect 503.
        assert_ne!(response.status(), 404);
    }

    #[tokio::test]
    async fn get_recent_concepts_without_db_returns_503() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/concepts/recent?limit=5&since_ms=1000")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 503);
    }
}
