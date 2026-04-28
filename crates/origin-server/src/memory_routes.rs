// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::state::ServerState;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::Json,
};
use origin_core::sources::compute_effective_confidence;
use origin_types::requests::{
    ConfirmRequest, ListMemoriesRequest, SearchMemoryRequest, StoreMemoryRequest,
};
use origin_types::responses::{
    ConfirmResponse, DeleteResponse, ListMemoriesResponse, SearchMemoryResponse,
    StoreMemoryResponse,
};
use origin_types::sources::{stability_tier, MemoryType, RawDocument, StabilityTier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// ===== Profile Types =====

#[derive(Debug, Serialize)]
pub struct ProfileResponse {
    pub id: String,
    pub name: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
    pub bio: Option<String>,
    pub avatar_path: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Deserialize)]
pub struct UpdateProfileRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub bio: Option<String>,
    #[serde(default)]
    pub avatar_path: Option<String>,
}

// ===== Agent Types =====

#[derive(Debug, Serialize)]
pub struct AgentResponse {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub agent_type: String,
    pub description: Option<String>,
    pub enabled: bool,
    pub trust_level: String,
    pub last_seen_at: Option<i64>,
    pub memory_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Deserialize)]
pub struct UpdateAgentRequest {
    #[serde(default)]
    pub agent_type: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub trust_level: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
}

// ===== Knowledge Graph Types =====

#[derive(Debug, Deserialize)]
pub struct CreateEntityRequest {
    pub name: String,
    pub entity_type: String,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct CreateEntityResponse {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateRelationRequest {
    pub from_entity: String,
    pub to_entity: String,
    pub relation_type: String,
    #[serde(default)]
    pub source_agent: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateRelationResponse {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct AddObservationRequest {
    pub entity_id: String,
    pub content: String,
    #[serde(default)]
    pub source_agent: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct LinkEntityRequest {
    pub source_id: String,
    pub entity_id: String,
}

#[derive(Debug, Serialize)]
pub struct AddObservationResponse {
    pub id: String,
}

// ===== Route Handlers =====

/// Compute schema-validation warnings and the extraction method label,
/// replacing the prior three-branch warnings computation that conflated
/// schema validation with extraction status.
///
/// - branch 1 (LLM extracted): warnings from schema validation of the extracted fields.
/// - branch 2 (agent supplied): warnings from schema validation of the agent fields.
/// - branch 3 (neither): empty warnings, extraction_method = "none".
///
/// Callers should populate `StoreMemoryResponse.extraction_method` with the second tuple element.
fn compute_warnings_and_extraction(
    extracted_fields: Option<&str>,
    agent_fields: Option<&serde_json::Value>,
    memory_type_str: &str,
) -> (Vec<String>, String) {
    use std::collections::HashMap;

    let fields_map =
        |raw: Option<&str>, agent: Option<&serde_json::Value>| -> Option<HashMap<String, String>> {
            if let Some(sf) = raw {
                let value_map: HashMap<String, serde_json::Value> =
                    serde_json::from_str(sf).unwrap_or_default();
                return Some(
                    value_map
                        .into_iter()
                        .map(|(k, v)| {
                            (
                                k,
                                match v {
                                    serde_json::Value::String(s) => s,
                                    other => other.to_string(),
                                },
                            )
                        })
                        .collect(),
                );
            }
            if let Some(agent_sf) = agent {
                let value_map: HashMap<String, serde_json::Value> =
                    serde_json::from_value(agent_sf.clone()).unwrap_or_default();
                return Some(
                    value_map
                        .into_iter()
                        .map(|(k, v)| {
                            (
                                k,
                                match v {
                                    serde_json::Value::String(s) => s,
                                    other => other.to_string(),
                                },
                            )
                        })
                        .collect(),
                );
            }
            None
        };

    match (extracted_fields, agent_fields) {
        (Some(_), _) => {
            let fields = fields_map(extracted_fields, None).unwrap_or_default();
            let schema = origin_core::memory_schema::MemorySchema::for_type(memory_type_str);
            (schema.validate(&fields), "llm".to_string())
        }
        (None, Some(_)) => {
            let fields = fields_map(None, agent_fields).unwrap_or_default();
            let schema = origin_core::memory_schema::MemorySchema::for_type(memory_type_str);
            (schema.validate(&fields), "agent".to_string())
        }
        (None, None) => (Vec::new(), "none".to_string()),
    }
}

/// POST /api/memory/store
pub async fn handle_store_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    Json(req): Json<StoreMemoryRequest>,
) -> Result<Json<StoreMemoryResponse>, ServerError> {
    let trimmed_content = req.content.trim();
    if trimmed_content.len() < 10 {
        return Err(ServerError::ValidationError(
            "Memory content must be at least 10 characters".into(),
        ));
    }

    // Dedup check
    {
        let s = state.read().await;
        if let Some(db) = s.db.as_ref() {
            if db.has_memory_content(&req.content).await.unwrap_or(false) {
                return Err(ServerError::ValidationError(
                    "Duplicate: a memory with this content already exists".into(),
                ));
            }
        }
    }

    // Validate caller-supplied memory_type — parse and keep it as-is. Profile
    // aliases now resolve in the async enrichment path below; previously this
    // block made an LLM call (up to 5s sync) to resolve the subtype, which
    // dominated store-time latency. Caller-supplied alias flows through to the
    // deferred classifier which produces the concrete subtype.
    let caller_supplied_memory_type = !matches!(req.memory_type.as_deref(), None | Some(""));
    let caller_supplied_a_profile_alias = req
        .memory_type
        .as_deref()
        .map(MemoryType::is_profile_alias)
        .unwrap_or(false);
    let validated_memory_type: Option<String> = match req.memory_type.as_deref() {
        None | Some("") => None,
        Some(mt) if MemoryType::is_profile_alias(mt) => {
            // Stored as "identity" as a conservative placeholder; async classify
            // replaces with the actual subtype (identity/preference/goal/fact).
            // Matches the prior fallback when no LLM was available.
            Some("identity".to_string())
        }
        Some(mt) => {
            let parsed: MemoryType = mt.parse().map_err(ServerError::ValidationError)?;
            Some(parsed.to_string())
        }
    };

    let source_id = format!(
        "mem_{}",
        uuid::Uuid::new_v4()
            .to_string()
            .replace('-', "")
            .chars()
            .take(12)
            .collect::<String>()
    );
    let title = req
        .title
        .unwrap_or_else(|| truncate_for_title(&req.content));

    // Phase 1: Agent gating + note whether an LLM is available.
    //
    // Resolve the agent name via `extract_agent_name` (header-canonical)
    // before gating. Previously this read `req.source_agent` (body-only),
    // which meant a caller using the `x-agent-name` header — the canonical
    // channel since the channel-collapse — silently bypassed registration
    // and gating entirely (got `None → "full"` auto-trust).
    //
    // Prompts, classify/extract LLM calls, and classification writebacks all
    // moved to the async enrichment spawn below — the sync path only needs
    // to know whether an LLM is available so it can pick the right response
    // hint and `enrichment` state.
    let resolved_agent = extract_agent_name(&headers, req.source_agent.as_deref());
    let (trust_level, llm_available) = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
        let trust = if resolved_agent == "unknown" {
            // No agent identified at all → local/first-party write, full trust.
            "full".to_string()
        } else {
            db.check_agent_for_write(&resolved_agent)
                .await
                .map_err(|e| ServerError::Internal(e.to_string()))?
        };
        (trust, s.llm.is_some())
    };

    // Placeholder classification. The async enrichment path will replace
    // these via `db.apply_enrichment(...)` and `space_store.set_document_tags(...)`
    // once the LLM has run (typically within a few seconds).
    let memory_type_str = validated_memory_type
        .clone()
        .unwrap_or_else(|| "fact".to_string());
    let classified_domain: Option<String> = None;
    let classified_tags: Vec<String> = Vec::new();
    let classified_quality: Option<String> = None;

    // Structured fields / retrieval_cue are caller-supplied or deferred to the
    // async extractor. Sync path never runs the extract LLM call anymore.
    let extracted_fields: Option<String> = None;
    let extracted_cue: Option<String> = None;

    // Capture before `req.structured_fields` is consumed into the RawDocument —
    // the async enrichment spawn uses this to decide whether to run the LLM
    // extract pass. If the caller already supplied fields, we skip extract.
    let caller_supplied_structured_fields = req.structured_fields.is_some();

    // Phase 2b-validate: split into warnings (schema-validation only) and extraction_method (status label).
    let (warnings, extraction_method) = compute_warnings_and_extraction(
        extracted_fields.as_deref(),
        req.structured_fields.as_ref(),
        &memory_type_str,
    );

    // Phase 2c: Entity resolution
    let resolved_entity_id = if let Some(ref direct_id) = req.entity_id {
        Some(direct_id.clone())
    } else if let Some(ref entity_name) = req.entity {
        let s = state.read().await;
        if let Some(db) = s.db.as_ref() {
            match db.resolve_entity_by_name(entity_name).await {
                Ok(Some(id)) => {
                    tracing::info!("[memory] resolved entity '{}' → {}", entity_name, id);
                    Some(id)
                }
                Ok(None) => {
                    tracing::debug!(
                        "[memory] entity '{}' not found, will be linked post-ingest",
                        entity_name
                    );
                    None
                }
                Err(e) => {
                    tracing::warn!("[memory] entity resolution failed: {e}");
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // --- Topic-match check (pre-batcher) ---
    // Run before building RawDocument. If the incoming memory matches an existing
    // topic (same domain+type, similar embedding), upsert in place and return early,
    // skipping the batcher and avoiding duplicates.
    // When a protected memory is matched we fall through to normal store but flag
    // the new memory as a pending revision so the UI can surface it for review.
    let mut topic_match_protected_id: Option<String> = None;
    {
        let (db_arc, topic_cfg) = {
            let s = state.read().await;
            (s.db.clone(), s.tuning.refinery.topic_match.clone())
        };

        if let Some(ref db) = db_arc {
            // Compute embedding synchronously (CPU-bound, no DB lock needed)
            let embedding_result = db.generate_embeddings(&[trimmed_content.to_string()]);

            if let Ok(ref embeddings) = embedding_result {
                if let Some(content_embedding) = embeddings.first() {
                    let match_result = origin_core::topic_match::find_topic_match(
                        db,
                        &title,
                        validated_memory_type.as_deref(),
                        req.domain.as_deref(),
                        resolved_entity_id.as_deref(),
                        content_embedding,
                        &topic_cfg,
                    )
                    .await;

                    if let Ok(ref result) = match_result {
                        if let Some(ref matched_source_id) = result.matched_source_id {
                            // Trust guardrail: don't overwrite protected memories in place.
                            let is_protected = db
                                .is_memory_protected(matched_source_id)
                                .await
                                .unwrap_or(false);

                            if !is_protected {
                                let upsert_result = db
                                    .upsert_memory_in_place(
                                        matched_source_id,
                                        trimmed_content,
                                        content_embedding,
                                        req.source_agent.as_deref(),
                                        None, // incoming_source_id N/A (no new id yet)
                                        topic_cfg.changelog_cap,
                                    )
                                    .await;

                                if let Ok(()) = upsert_result {
                                    // Spawn async staleness check for linked concepts.
                                    let db_clone = db.clone();
                                    let msid = matched_source_id.clone();
                                    let old_emb = result.old_embedding.clone();
                                    let new_emb = content_embedding.clone();
                                    let handle = tokio::spawn(async move {
                                        let sim = old_emb.as_deref().map(|oe| {
                                            origin_core::topic_match::cosine_similarity(
                                                oe, &new_emb,
                                            )
                                        });
                                        if let Ok(concepts) =
                                            db_clone.get_concepts_for_memory(&msid).await
                                        {
                                            for concept in concepts {
                                                let stale_reason = match sim {
                                                    Some(s) if s > 0.90 => None, // trivial
                                                    Some(s) if s > 0.60 => Some("source_updated"),
                                                    _ => Some("source_conflict"),
                                                };
                                                if let Some(reason) = stale_reason {
                                                    let _ = db_clone
                                                        .set_concept_stale(&concept.id, reason)
                                                        .await;
                                                } else {
                                                    let _ = db_clone
                                                        .increment_concept_sources_updated(
                                                            &concept.id,
                                                        )
                                                        .await;
                                                }
                                            }
                                        }
                                    });
                                    tokio::spawn(async move {
                                        if let Err(e) = handle.await {
                                            tracing::warn!(
                                                "[topic_match] staleness check task panicked: {e}"
                                            );
                                        }
                                    });

                                    tracing::info!(
                                        "[topic_match] upserted in place: source_id={matched_source_id} signals={:?}",
                                        result.signals
                                    );
                                    return Ok(Json(StoreMemoryResponse {
                                        source_id: matched_source_id.clone(),
                                        chunks_created: 1,
                                        memory_type: validated_memory_type
                                            .clone()
                                            .unwrap_or_else(|| "fact".to_string()),
                                        entity_id: resolved_entity_id,
                                        quality: None,
                                        warnings: vec![],
                                        extraction_method: "agent".to_string(),
                                        enrichment: "not_needed".to_string(),
                                        hint: "topic_updated".to_string(),
                                    }));
                                } else if let Err(ref e) = upsert_result {
                                    tracing::warn!(
                                        "[topic_match] upsert_memory_in_place failed, \
                                         falling through to normal store: {e}"
                                    );
                                }
                            } else {
                                tracing::info!(
                                    "[topic_match] matched protected memory {matched_source_id}, \
                                     storing as new pending_revision"
                                );
                                topic_match_protected_id = Some(matched_source_id.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 3: Confidence + auto-confirm + supersede gating
    let memory_type = Some(memory_type_str.clone());
    let tier = stability_tier(memory_type.as_deref());
    let confidence_cfg = {
        let s = state.read().await;
        s.tuning.confidence.clone()
    };
    let effective_confidence = compute_effective_confidence(
        req.confidence,
        memory_type.as_deref(),
        &trust_level,
        classified_quality.as_deref(),
        &confidence_cfg,
    );

    let stability = match (&tier, trust_level.as_str(), classified_quality.as_deref()) {
        (StabilityTier::Protected, _, _) => "new",
        (StabilityTier::Ephemeral, "full", Some("high" | "medium")) => "learned",
        (_, "full", Some("high")) => "learned",
        _ => "new",
    };
    let confirmed = Some(stability == "confirmed");

    // A topic-match against a protected memory also flags this as a pending revision.
    let pending_revision = topic_match_protected_id.is_some();
    // Agent-declared supersedes: trust the caller directly, no auto-detection.
    let final_supersedes = req.supersedes.clone();

    let agent_for_activity = {
        let s = state.read().await;
        extract_agent_name_with_db(&headers, req.source_agent.as_deref(), s.db.as_deref()).await
    };
    let supersedes_for_activity = final_supersedes.clone();

    let supersede_mode = if memory_type_str == "decision" {
        "archive".to_string()
    } else {
        "hide".to_string()
    };

    let final_domain = req.domain.or(classified_domain);
    let structured_fields_for_enrichment = req
        .structured_fields
        .as_ref()
        .map(|v| v.to_string())
        .or_else(|| extracted_fields.clone());
    let doc = RawDocument {
        source: "memory".to_string(),
        source_id: source_id.clone(),
        title,
        summary: None,
        content: req.content.clone(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        event_date: None,
        metadata: HashMap::new(),
        memory_type: Some(memory_type_str.clone()),
        domain: final_domain.clone(),
        source_agent: req.source_agent,
        confidence: Some(effective_confidence),
        confirmed,
        stability: Some(stability.to_string()),
        supersedes: final_supersedes,
        pending_revision,
        entity_id: resolved_entity_id.clone(),
        quality: classified_quality.clone(),
        is_recap: false,
        enrichment_status: "raw".to_string(),
        supersede_mode,
        structured_fields: req
            .structured_fields
            .map(|v| v.to_string())
            .or(extracted_fields),
        retrieval_cue: req.retrieval_cue.clone().or(extracted_cue),
        source_text: None,
    };

    // Pre-chunk locally so we know chunks_created for the response even
    // when the upsert goes through the coalescer (which returns a sum, not
    // per-doc counts). `upsert_documents` applies `redact_pii` BEFORE
    // chunking (db.rs:~4236); PII redaction is not length-preserving
    // (e.g. an email becomes `[REDACTED:EMAIL]` which shifts byte offsets
    // by ±5), so chunking raw content here would drift from what the
    // backend actually stores for any memory near the chunker's 512-char
    // boundary that contains PII. Redact first to keep the counts aligned.
    let chunks_predicted = {
        let redacted = origin_core::privacy::redact_pii(&doc.content);
        let chunker = origin_core::chunker::ChunkingEngine::new();
        chunker
            .chunk(&redacted, &doc.title, &doc.source_id, &doc.metadata)
            .len()
            .max(1)
    };

    // Capture content + agent before doc is moved into the batcher.
    // Needed for rejection logging on the coalesced path where `doc` is
    // consumed by `batcher.submit(...)` before the outcome is known.
    let doc_content_for_log = doc.content.clone();
    let doc_agent_for_log = doc.source_agent.clone();

    // Snapshot out-of-guard: grab the coalescer handle and a DB Arc, then
    // drop the read guard before awaiting on the batched upsert.
    // `tuning` and `quality_gate` are needed for the fallback path (sync gate)
    // and for rejection logging on the coalesced path.
    let (ingest_batcher, db_fallback, tuning, quality_gate) = {
        let s = state.read().await;
        (
            s.ingest_batcher.clone(),
            s.db.clone(),
            s.tuning.clone(),
            s.quality_gate.clone(),
        )
    };

    let chunks_created = if let Some(batcher) = ingest_batcher {
        // Coalesced path: concurrent callers share one FastEmbed call +
        // one libSQL transaction. Gate runs inside the coalescer flush.
        // See `ingest_batcher.rs` for details.
        use crate::ingest_batcher::StoreOutcome;
        match batcher
            .submit(doc, chunks_predicted)
            .await
            .map_err(ServerError::IngestFailed)?
        {
            StoreOutcome::Stored { chunks_created } => chunks_created,
            StoreOutcome::GateRejected {
                reason,
                detail,
                similar_to,
            } => {
                // Rejection logging: the coalescer runs the gate but cannot
                // reach back to per-request handler state; we log here using
                // the tuning/db snapshot taken above.
                if tuning.gate.log_rejections {
                    if let Some(db) = db_fallback.as_ref() {
                        let rej_id = format!(
                            "rej_{}",
                            uuid::Uuid::new_v4()
                                .to_string()
                                .replace('-', "")
                                .chars()
                                .take(12)
                                .collect::<String>()
                        );
                        if let Err(e) = db
                            .log_rejection(
                                &rej_id,
                                &doc_content_for_log,
                                doc_agent_for_log.as_deref(),
                                &reason,
                                Some(&detail),
                                None,
                                similar_to.as_deref(),
                            )
                            .await
                        {
                            tracing::warn!("[quality_gate] failed to log rejection: {e}");
                        }
                    }
                }
                tracing::info!(
                    "[quality_gate] rejected memory from {:?}: {}",
                    doc_agent_for_log.as_deref().unwrap_or("unknown"),
                    detail,
                );
                return Err(ServerError::QualityGateRejected {
                    reason,
                    detail,
                    similar_to,
                });
            }
            StoreOutcome::UpsertFailed(msg) => {
                return Err(ServerError::IngestFailed(msg));
            }
        }
    } else {
        // Fallback when the batcher isn't wired (unit tests, degraded state).
        // Gate runs synchronously here, pre-upsert.
        let db = db_fallback.clone().ok_or(ServerError::DbNotInitialized)?;
        let (gate_result, similar_source_id) = quality_gate
            .evaluate(&doc.content, &db)
            .await
            .unwrap_or_else(|e| {
                tracing::error!("[quality_gate] evaluate failed (fail closed): {e}");
                (
                    origin_core::quality_gate::GateResult {
                        admitted: false,
                        reason: Some(
                            origin_core::quality_gate::RejectionReason::EmbeddingUnavailable(
                                e.to_string(),
                            ),
                        ),
                        scores: origin_core::quality_gate::GateScores {
                            content_type_pass: true,
                            novelty_score: None,
                            word_count: 0,
                            pattern_matched: Some("embedding_unavailable".to_string()),
                            latency_ms: 0,
                        },
                    },
                    None,
                )
            });
        if !gate_result.admitted {
            if let Some(ref reason) = gate_result.reason {
                if tuning.gate.log_rejections {
                    let rej_id = format!(
                        "rej_{}",
                        uuid::Uuid::new_v4()
                            .to_string()
                            .replace('-', "")
                            .chars()
                            .take(12)
                            .collect::<String>()
                    );
                    if let Err(e) = db
                        .log_rejection(
                            &rej_id,
                            &doc.content,
                            doc.source_agent.as_deref(),
                            reason.as_str(),
                            Some(&reason.detail()),
                            gate_result.scores.novelty_score,
                            similar_source_id.as_deref(),
                        )
                        .await
                    {
                        tracing::warn!("[quality_gate] failed to log rejection: {e}");
                    }
                }
                tracing::info!(
                    "[quality_gate] rejected memory from {:?}: {}",
                    doc.source_agent.as_deref().unwrap_or("unknown"),
                    reason.detail()
                );
            }
            let (rej_reason, rej_detail) = gate_result
                .reason
                .map(|r| (r.as_str().to_string(), r.detail()))
                .unwrap_or_else(|| ("unknown".to_string(), "Quality gate rejected".to_string()));
            return Err(ServerError::QualityGateRejected {
                reason: rej_reason,
                detail: rej_detail,
                similar_to: similar_source_id,
            });
        }
        db.upsert_documents(vec![doc])
            .await
            .map_err(|e| ServerError::IngestFailed(e.to_string()))?
    };

    if let Some(ref domain) = final_domain {
        if let Some(db) = db_fallback.as_ref() {
            if let Err(e) = db.auto_create_space_if_needed(domain).await {
                tracing::warn!("[memory] auto-create space failed: {e}");
            }
        }
    }

    if chunks_created == 0 {
        return Err(ServerError::ValidationError(
            "Memory produced no indexable content after processing".into(),
        ));
    }

    // Classified tags are now written in the async enrichment spawn below —
    // `classified_tags` is always empty at this point because classify moved
    // off the sync path. Kept as a no-op branch for the rare caller that
    // pre-supplies tags in the future; can be removed with an API cleanup.
    let _ = classified_tags; // intentionally unused

    // Log agent activity
    {
        let s = state.read().await;
        if let Some(db) = s.db.as_ref() {
            if let Some(ref old_id) = supersedes_for_activity {
                let ids = vec![source_id.clone(), old_id.clone()];
                if let Err(e) = db
                    .log_agent_activity(
                        &agent_for_activity,
                        "refine",
                        &ids,
                        None,
                        "updated with new reasoning",
                    )
                    .await
                {
                    tracing::warn!("Failed to log agent refine activity: {}", e);
                }
            } else {
                let ids = vec![source_id.clone()];
                let detail = format!("stored a {} memory", memory_type_str);
                if let Err(e) = db
                    .log_agent_activity(&agent_for_activity, "store", &ids, None, &detail)
                    .await
                {
                    tracing::warn!("Failed to log agent store activity: {}", e);
                }
            }
        }
    }

    // Record write event for the steep scheduler's burst detection.
    // Must be called BEFORE spawning post-ingest — captures the write
    // timestamp immediately, not after LLM enrichment completes.
    {
        let s = state.read().await;
        s.write_signal.record(&resolved_agent);
    }

    // Deferred enrichment (async, non-blocking).
    //
    // Two phases run here, off the request's critical path:
    //
    //   1. LLM classify + extract + `db.apply_enrichment(...)` + tags write —
    //      replaces the inline LLM calls that used to gate the HTTP response.
    //      Sync path stored placeholder values (memory_type="fact" unless the
    //      caller supplied one; no domain/quality/structured_fields from LLM);
    //      this phase fills them in via a combined UPDATE. Tags go through
    //      `space_store` and require a brief write guard.
    //
    //   2. `run_post_ingest_enrichment(...)` — entity auto-linking, contradiction
    //      candidate queueing, title enrichment, concept growth. Existing
    //      behavior; runs with the enriched fields phase 1 produced.
    //
    // IMPORTANT: extract Arcs/clones from the read guard and drop it BEFORE
    // entering any `.await` on LLM or DB work. Holding a read guard across an
    // await would block writers (e.g., config updates, space edits) for the
    // full duration of enrichment — which can take several seconds per LLM
    // call. See docs/superpowers/specs/2026-04-09-core-app-separation-design.md.
    {
        let state_clone = state.clone();
        let source_id_clone = source_id.clone();
        let content_clone = req.content.clone();
        let entity_id_clone = resolved_entity_id.clone();
        let initial_memory_type = memory_type_str.clone();
        let initial_domain = final_domain.clone();
        // Already moved into the doc above — rebuild from the same formula
        // the RawDocument constructor used.
        let initial_supersede_mode = if memory_type_str == "decision" {
            "archive".to_string()
        } else {
            "hide".to_string()
        };
        let initial_structured_fields = structured_fields_for_enrichment.clone();
        let agent_supplied_memory_type = caller_supplied_memory_type;
        let agent_supplied_profile_alias = caller_supplied_a_profile_alias;
        let agent_supplied_structured_fields = caller_supplied_structured_fields;
        tokio::spawn(async move {
            // Snapshot everything we need, then drop the guard.
            let (db, llm, prompts, refinery, distillation, knowledge_path) = {
                let s = state_clone.read().await;
                let Some(db) = s.db.clone() else {
                    return;
                };
                (
                    db,
                    s.llm.clone(),
                    s.prompts.clone(),
                    s.tuning.refinery.clone(),
                    s.tuning.distillation.clone(),
                    Some(origin_core::config::load_config().knowledge_path_or_default()),
                )
            }; // read guard dropped here — writers may proceed

            // Phase 1: deferred classify + extract + apply_enrichment + tags.
            let mut final_memory_type = initial_memory_type.clone();
            let mut final_domain = initial_domain.clone();
            let mut final_supersede_mode = initial_supersede_mode.clone();
            let mut final_quality: Option<String> = None;
            let mut final_structured_fields: Option<String> = initial_structured_fields.clone();
            let mut final_retrieval_cue: Option<String> = None;
            let mut classified_tags_async: Vec<String> = Vec::new();

            if let Some(ref llm) = llm {
                // Classify. Agent-supplied memory_type stays authoritative
                // unless they passed a profile alias (in which case we take
                // the classifier's concrete subtype). Other fields —
                // domain/quality/tags — come from classify regardless.
                let truncated: String = content_clone.chars().take(1000).collect();
                // 30s matches `OnDeviceProvider::generate`'s own timeout at
                // llm_provider.rs:210. 5s was too aggressive under a burst:
                // when 10 concurrent MCP `remember` calls each fire classify
                // + extract, the 10th call's turn at the single LLM worker
                // lands at ~30s — with a 5s limit, 8 of 10 memories stayed
                // at placeholder "fact" forever. 30s lets the full burst
                // clear; anything beyond that is a real LLM stall and
                // reverts to the placeholder gracefully.
                match tokio::time::timeout(
                    std::time::Duration::from_secs(30),
                    llm.generate(origin_core::llm_provider::LlmRequest {
                        system_prompt: Some(prompts.classify_memory_quality.clone()),
                        user_prompt: truncated,
                        max_tokens: 128,
                        temperature: 0.1,
                        label: None,
                    }),
                )
                .await
                {
                    Ok(Ok(output)) => {
                        if let Some(c) = origin_core::llm_provider::parse_classify_response(&output)
                        {
                            if !agent_supplied_memory_type || agent_supplied_profile_alias {
                                final_memory_type = c.memory_type.clone();
                                // Recompute supersede_mode for the refined type.
                                final_supersede_mode = if final_memory_type == "decision" {
                                    "archive".to_string()
                                } else {
                                    "hide".to_string()
                                };
                            }
                            if final_domain.is_none() {
                                final_domain = c.domain;
                            }
                            final_quality = c.quality;
                            classified_tags_async = c.tags;
                        }
                    }
                    Ok(Err(e)) => tracing::warn!("[store_memory async] classify failed: {e}"),
                    Err(_) => tracing::warn!("[store_memory async] classify timed out after 5s"),
                }

                // Extract structured fields — only if the agent didn't supply them.
                if !agent_supplied_structured_fields {
                    let prompt = origin_core::memory_schema::extraction_prompt_with_template(
                        &final_memory_type,
                        &prompts.extract_structured_fields,
                    );
                    let truncated: String = content_clone.chars().take(1500).collect();
                    // Same reasoning as the classify timeout above — 30s
                    // is the provider's own cap; this is defense-in-depth
                    // so a stalled extract doesn't hang the post_ingest task.
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(30),
                        llm.generate(origin_core::llm_provider::LlmRequest {
                            system_prompt: Some(prompt),
                            user_prompt: truncated,
                            max_tokens: 256,
                            temperature: 0.1,
                            label: None,
                        }),
                    )
                    .await
                    {
                        Ok(Ok(output)) => {
                            let (fields, cue) =
                                origin_core::llm_provider::parse_extraction_response(&output);
                            if let Some(f) = fields {
                                final_structured_fields = Some(f);
                            }
                            if let Some(c) = cue {
                                final_retrieval_cue = Some(c);
                            }
                        }
                        Ok(Err(e)) => {
                            tracing::warn!("[store_memory async] extract failed: {e}")
                        }
                        Err(_) => {
                            tracing::warn!("[store_memory async] extract timed out after 5s")
                        }
                    }
                }

                // Apply the refined classification to the stored memory.
                if let Err(e) = db
                    .apply_enrichment(
                        &source_id_clone,
                        &final_memory_type,
                        final_domain.as_deref(),
                        final_quality.as_deref(),
                        &final_supersede_mode,
                        final_structured_fields.as_deref(),
                        final_retrieval_cue.as_deref(),
                    )
                    .await
                {
                    tracing::warn!("[store_memory async] apply_enrichment failed: {e}");
                }

                // Auto-create a space for the classified domain if one came
                // back from classify and the sync path didn't already see it.
                if let Some(ref domain) = final_domain {
                    if initial_domain.as_deref() != Some(domain.as_str()) {
                        if let Err(e) = db.auto_create_space_if_needed(domain).await {
                            tracing::warn!("[store_memory async] auto-create space failed: {e}");
                        }
                    }
                }

                // Write tags. Brief write guard — synchronous in-memory update.
                if !classified_tags_async.is_empty() {
                    let mut s = state_clone.write().await;
                    s.space_store.set_document_tags(
                        "memory",
                        &source_id_clone,
                        classified_tags_async,
                    );
                    let _ = origin_core::spaces::save_spaces(&s.space_store);
                }
            }

            // Phase 2: existing post-ingest enrichment (entity linking,
            // contradiction queueing, title enrichment, concept growth).
            if let Err(e) = origin_core::post_ingest::run_post_ingest_enrichment(
                &db,
                &source_id_clone,
                &content_clone,
                entity_id_clone.as_deref(),
                Some(final_memory_type.as_str()),
                final_domain.as_deref(),
                final_structured_fields.as_deref(),
                llm.as_ref(),
                &prompts,
                &refinery,
                &distillation,
                knowledge_path.as_deref(),
            )
            .await
            {
                tracing::warn!("[store_memory] post-ingest enrichment failed: {e}");
            }
        });
    }

    // Fire-once onboarding milestone checks (ingest side).
    //
    // Spawned so the HTTP response isn't delayed by these background queries.
    // We snapshot Arc<MemoryDB> out of the read guard BEFORE the spawn so no
    // lock is held across `.await` (per CLAUDE.md). The daemon currently has
    // no UI to notify, so a fresh `NoopEmitter` is used inline — the emit is
    // cosmetic for the HTTP-only path. Milestones are still persisted via
    // `record_milestone` in the DB and surfaced to the UI through the
    // /api/onboarding/* endpoints.
    {
        let db_for_ms = {
            let s = state.read().await;
            s.db.clone()
        };
        if let Some(db_for_ms) = db_for_ms {
            let emitter_for_ms: Arc<dyn origin_core::events::EventEmitter> =
                Arc::new(origin_core::events::NoopEmitter);
            let source_for_ms = agent_for_activity.clone();
            let memory_id_for_ms = source_id.clone();
            tokio::spawn(async move {
                let ev =
                    origin_core::onboarding::MilestoneEvaluator::new(&db_for_ms, emitter_for_ms);
                if let Err(e) = ev
                    .check_after_ingest(&memory_id_for_ms, &source_for_ms)
                    .await
                {
                    tracing::warn!(?e, "onboarding: check_after_ingest failed");
                }
                if let Err(e) = ev.check_after_agent_register(&source_for_ms).await {
                    tracing::warn!(?e, "onboarding: check_after_agent_register failed");
                }
            });
        }
    }

    // Build caller-facing status. Enrichment is pending whenever an LLM is
    // wired — classify + extract + apply_enrichment runs asynchronously and
    // will backfill `memory_type`, `domain`, `quality`, tags, and structured
    // fields within a few seconds. When no LLM is available, the memory
    // stays as caller-supplied — no enrichment will run, and callers should
    // not poll for enriched fields.
    let (enrichment, hint) = if llm_available {
        (
            "pending".to_string(),
            "Stored. Origin is compiling classification + concept links in the \
             background (~2s). Recall will surface the enriched form shortly."
                .to_string(),
        )
    } else {
        ("not_needed".to_string(), String::new())
    };

    Ok(Json(StoreMemoryResponse {
        source_id,
        chunks_created,
        memory_type: memory_type_str,
        entity_id: resolved_entity_id,
        quality: classified_quality,
        warnings,
        extraction_method,
        enrichment,
        hint,
    }))
}

/// POST /api/memory/search
pub async fn handle_search_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    Json(req): Json<SearchMemoryRequest>,
) -> Result<Json<SearchMemoryResponse>, ServerError> {
    let start = std::time::Instant::now();

    let results = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
        db.search_memory(
            &req.query,
            req.limit,
            req.memory_type.as_deref(),
            req.domain.as_deref(),
            req.source_agent.as_deref(),
            None,
            None,
            None,
        )
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?
    };

    let source_ids: Vec<String> = results.iter().map(|r| r.source_id.clone()).collect();
    {
        let s = state.read().await;
        s.access_tracker.record_accesses(&source_ids);
        if let Some(db) = s.db.as_ref() {
            if let Err(e) = db.log_accesses(&source_ids).await {
                tracing::warn!("Failed to log accesses: {}", e);
            }
        }
    }

    {
        let s = state.read().await;
        // Resolve attribution from x-agent-name header, falling back to the
        // deprecated body `source_agent` field. Previously this passed `None`
        // for the body fallback, so requests that sent only body `source_agent`
        // (no header) were logged to `agent_activity` as "unknown", producing
        // the `agent_name="unknown"` rows visible in `/api/retrievals/recent`.
        let agent =
            extract_agent_name_with_db(&headers, req.source_agent.as_deref(), s.db.as_deref())
                .await;
        let detail = format!("found {} results", results.len());
        if let Some(db) = s.db.as_ref() {
            if let Err(e) = db
                .log_agent_activity(&agent, "search", &source_ids, Some(&req.query), &detail)
                .await
            {
                tracing::warn!("Failed to log agent activity: {}", e);
            }
        }
    }

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(Json(SearchMemoryResponse { results, took_ms }))
}

/// POST /api/memory/confirm/{source_id}
pub async fn handle_confirm_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
    body: Option<Json<ConfirmRequest>>,
) -> Result<Json<ConfirmResponse>, ServerError> {
    let confirmed = body.map(|b| b.confirmed).unwrap_or(true);
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    if confirmed {
        db.set_stability(&source_id, "confirmed")
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    } else {
        db.set_stability(&source_id, "new")
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    }
    Ok(Json(ConfirmResponse { confirmed }))
}

/// GET /api/memory/list
pub async fn handle_list_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<ListMemoriesRequest>,
) -> Result<Json<ListMemoriesResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let memories = db
        .list_filtered(
            Some("memory"),
            req.memory_type.as_deref(),
            req.domain.as_deref(),
            req.limit,
        )
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(ListMemoriesResponse { memories }))
}

/// DELETE /api/memory/delete/{source_id}
pub async fn handle_delete_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    Path(source_id): Path<String>,
) -> Result<Json<DeleteResponse>, ServerError> {
    // Snapshot Arc<MemoryDB> + resolve agent name from the RwLock guard,
    // then drop the guard BEFORE any `.await` calls. Follows the pattern
    // established in `handle_store_memory` (see CLAUDE.md: "Never hold a
    // `tokio::sync::RwLock` read or write guard across `.await`.")
    let (db, agent) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let agent = extract_agent_name(&headers, None);
        (db, agent)
    }; // guard dropped here — writers may proceed

    // Capture title before deletion so we can include it in the activity log.
    let title = db
        .get_memory_detail(&source_id)
        .await
        .ok()
        .flatten()
        .map(|m| m.title)
        .filter(|t| !t.is_empty());

    db.delete_by_source_id("memory", &source_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    // Log the forget activity. Memory is gone so title lookup would fail —
    // carry the title in `detail` instead.
    let detail = match title.as_deref() {
        Some(t) => format!("forgot \"{}\"", t),
        None => "forgot a memory".to_string(),
    };
    let ids = vec![source_id.clone()];
    if let Err(e) = db
        .log_agent_activity(&agent, "forget", &ids, None, &detail)
        .await
    {
        tracing::warn!("Failed to log agent forget activity: {}", e);
    }

    Ok(Json(DeleteResponse { deleted: true }))
}

// ===== Knowledge Graph Handlers =====

pub async fn handle_create_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<CreateEntityRequest>,
) -> Result<Json<CreateEntityResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let id = db
        .store_entity(
            &req.name,
            &req.entity_type,
            req.domain.as_deref(),
            req.source_agent.as_deref(),
            req.confidence,
        )
        .await
        .map_err(|e| ServerError::IngestFailed(e.to_string()))?;
    Ok(Json(CreateEntityResponse { id }))
}

pub async fn handle_create_relation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<CreateRelationRequest>,
) -> Result<Json<CreateRelationResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let id = db
        .create_relation(
            &req.from_entity,
            &req.to_entity,
            &req.relation_type,
            req.source_agent.as_deref(),
            None,
            None,
            None,
        )
        .await
        .map_err(|e| ServerError::IngestFailed(e.to_string()))?;
    Ok(Json(CreateRelationResponse { id }))
}

pub async fn handle_add_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<AddObservationRequest>,
) -> Result<Json<AddObservationResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let id = db
        .add_observation(
            &req.entity_id,
            &req.content,
            req.source_agent.as_deref(),
            req.confidence,
        )
        .await
        .map_err(|e| ServerError::IngestFailed(e.to_string()))?;
    Ok(Json(AddObservationResponse { id }))
}

pub async fn handle_link_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<LinkEntityRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.update_memory_entity_id(&req.source_id, &req.entity_id)
        .await
        .map_err(|e| ServerError::IngestFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({"linked": true})))
}

// ===== Profile Handlers =====

pub async fn handle_get_profile(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<ProfileResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let profile = db
        .get_profile()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    match profile {
        Some(p) => Ok(Json(ProfileResponse {
            id: p.id,
            name: p.name,
            display_name: p.display_name,
            email: p.email,
            bio: p.bio,
            avatar_path: p.avatar_path,
            created_at: p.created_at,
            updated_at: p.updated_at,
        })),
        None => Err(ServerError::NotFound("No profile found".to_string())),
    }
}

pub async fn handle_update_profile(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<UpdateProfileRequest>,
) -> Result<Json<ProfileResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let profile = db
        .get_profile()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound("No profile found".to_string()))?;
    db.update_profile(
        &profile.id,
        req.name.as_deref(),
        req.display_name.as_deref(),
        req.email.as_deref(),
        req.bio.as_deref(),
        req.avatar_path.as_deref(),
    )
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let updated = db
        .get_profile()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound("No profile found".to_string()))?;
    Ok(Json(ProfileResponse {
        id: updated.id,
        name: updated.name,
        display_name: updated.display_name,
        email: updated.email,
        bio: updated.bio,
        avatar_path: updated.avatar_path,
        created_at: updated.created_at,
        updated_at: updated.updated_at,
    }))
}

// ===== Agent Handlers =====

pub async fn handle_list_agents(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<Vec<AgentResponse>>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let agents = db
        .list_agents()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(agents.into_iter().map(agent_to_response).collect()))
}

pub async fn handle_get_agent(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<AgentResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let agent = db
        .get_agent(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound(format!("Agent '{}' not found", name)))?;
    Ok(Json(agent_to_response(agent)))
}

pub async fn handle_update_agent(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateAgentRequest>,
) -> Result<Json<AgentResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.get_agent(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound(format!("Agent '{}' not found", name)))?;
    db.update_agent(
        &name,
        req.agent_type.as_deref(),
        req.description.as_deref(),
        req.enabled,
        req.trust_level.as_deref(),
        req.display_name.as_deref(),
    )
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))?;
    let updated = db
        .get_agent(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or(ServerError::NotFound(format!("Agent '{}' not found", name)))?;
    Ok(Json(agent_to_response(updated)))
}

pub async fn handle_delete_agent(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.delete_agent(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "deleted": name })))
}

fn agent_to_response(a: origin_core::db::AgentConnection) -> AgentResponse {
    AgentResponse {
        id: a.id,
        name: a.name,
        display_name: a.display_name,
        agent_type: a.agent_type,
        description: a.description,
        enabled: a.enabled,
        trust_level: a.trust_level,
        last_seen_at: a.last_seen_at,
        memory_count: a.memory_count,
        created_at: a.created_at,
        updated_at: a.updated_at,
    }
}

// ===== Knowledge Graph Retrieval Handlers =====

#[derive(Debug, Deserialize)]
pub struct ListEntitiesRequest {
    #[serde(default)]
    pub entity_type: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ListEntitiesResponse {
    pub entities: Vec<origin_core::db::Entity>,
}

pub async fn handle_list_entities(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<ListEntitiesRequest>,
) -> Result<Json<ListEntitiesResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let entities = db
        .list_entities(req.entity_type.as_deref(), req.domain.as_deref())
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(ListEntitiesResponse { entities }))
}

pub async fn handle_get_entity_detail(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(entity_id): Path<String>,
) -> Result<Json<origin_core::db::EntityDetail>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let detail = db
        .get_entity_detail(&entity_id)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(detail))
}

#[derive(Debug, Deserialize)]
pub struct SearchEntitiesRequest {
    pub query: String,
    #[serde(default = "default_entity_search_limit")]
    pub limit: usize,
}

fn default_entity_search_limit() -> usize {
    20
}

#[derive(Debug, Serialize)]
pub struct SearchEntitiesResponse {
    pub results: Vec<origin_core::db::EntitySearchResult>,
}

pub async fn handle_search_entities(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<SearchEntitiesRequest>,
) -> Result<Json<SearchEntitiesResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let results = db
        .search_entities_by_vector(&req.query, req.limit)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(SearchEntitiesResponse { results }))
}

#[derive(Debug, Serialize)]
pub struct MemoryStatsResponse {
    pub stats: origin_core::db::MemoryStats,
}

pub async fn handle_get_memory_stats(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<MemoryStatsResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let stats = db
        .get_memory_stats()
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(MemoryStatsResponse { stats }))
}

/// GET /api/home-stats
///
/// Aggregate metrics for the homepage dashboard. Combines distillation
/// counts, today/week access_log stats, and the top-N most retrieved
/// memories into a single `HomeStats` payload.
pub async fn handle_get_home_stats(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_types::HomeStats>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let stats = db
        .get_home_stats()
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(stats))
}

// ===== Reclassify Handler =====

#[derive(Debug, Deserialize)]
pub struct ReclassifyMemoryRequest {
    pub memory_type: String,
}

#[derive(Debug, Serialize)]
pub struct ReclassifyMemoryResponse {
    pub source_id: String,
    pub memory_type: String,
}

pub async fn handle_reclassify_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
    Json(req): Json<ReclassifyMemoryRequest>,
) -> Result<Json<ReclassifyMemoryResponse>, ServerError> {
    let parsed: MemoryType = req
        .memory_type
        .parse()
        .map_err(ServerError::ValidationError)?;
    let mt = parsed.to_string();

    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.update_column_by_source_id("memory", &source_id, "memory_type", &mt)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(ReclassifyMemoryResponse {
        source_id,
        memory_type: mt,
    }))
}

// ===== Pending Revision Endpoints =====

pub async fn handle_accept_revision(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.accept_pending_revision(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "accepted": true })))
}

pub async fn handle_dismiss_revision(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.dismiss_pending_revision(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "dismissed": true })))
}

/// POST /api/memory/contradiction/{source_id}/dismiss
///
/// Marks all awaiting-review contradiction flags for this memory as dismissed.
/// Returns 200 OK whether or not any rows were matched (idempotent).
pub async fn handle_dismiss_contradiction(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
) -> Result<StatusCode, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.dismiss_contradiction_for_source(&source_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(StatusCode::OK)
}

// ===== Enrichment Status =====

/// GET /api/memory/{source_id}/enrichment-status
///
/// Returns the enrichment step history and summary for a given memory.
pub async fn handle_get_enrichment_status(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
) -> Result<Json<origin_types::EnrichmentStatusResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let steps = db
        .get_enrichment_steps(&source_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    let summary = db
        .get_enrichment_summary(&source_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::EnrichmentStatusResponse {
        source_id,
        summary,
        steps,
    }))
}

/// POST /api/memory/{source_id}/enrichment-status/retry
///
/// Resets any abandoned enrichment steps so they will be retried.
/// Returns the number of steps reset. Idempotent: returns 0 if nothing was abandoned.
pub async fn handle_retry_enrichment(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let reset = db
        .reset_abandoned_steps(&source_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(
        serde_json::json!({ "source_id": source_id, "steps_reset": reset }),
    ))
}

// ===== Entity Suggestions =====

#[derive(Debug, Serialize)]
pub struct EntitySuggestion {
    pub id: String,
    pub entity_name: Option<String>,
    pub source_ids: Vec<String>,
    pub confidence: f64,
    pub created_at: String,
}

pub async fn handle_get_entity_suggestions(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<Vec<EntitySuggestion>>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let pending = db
        .get_pending_refinements()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    let suggestions: Vec<EntitySuggestion> = pending
        .iter()
        .filter(|p| {
            p.action == "suggest_entity" && (p.status == "pending" || p.status == "awaiting_review")
        })
        .map(|p| EntitySuggestion {
            id: p.id.clone(),
            entity_name: p.payload.clone(),
            source_ids: p.source_ids.clone(),
            confidence: p.confidence,
            created_at: p.created_at.clone(),
        })
        .collect();

    Ok(Json(suggestions))
}

pub async fn handle_approve_entity_suggestion(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;

    let pending = db
        .get_pending_refinements()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    let proposal = pending
        .iter()
        .find(|p| p.id == id && p.action == "suggest_entity")
        .ok_or(ServerError::NotFound(format!("suggestion {}", id)))?;

    let entity_name = proposal
        .payload
        .clone()
        .unwrap_or_else(|| "Unknown".to_string());

    let entity_id = db
        .create_entity(&entity_name, "auto", None)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    let entity_link_distance = s.tuning.refinery.entity_link_distance;
    let linked = origin_core::refinery::reweave_entity_links(db, 20, entity_link_distance)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    db.resolve_refinement(&id, "completed")
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "entity_id": entity_id,
        "entity_name": entity_name,
        "memories_linked": linked,
    })))
}

pub async fn handle_dismiss_entity_suggestion(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.resolve_refinement(&id, "dismissed")
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "dismissed": true })))
}

// ===== Helpers =====

fn truncate_for_title(content: &str) -> String {
    let first_line = content.lines().next().unwrap_or(content);
    if first_line.chars().count() > 80 {
        let truncated: String = first_line.chars().take(77).collect();
        format!("{}...", truncated)
    } else {
        first_line.to_string()
    }
}

/// Resolve the caller's agent name.
///
/// **Single source of truth: the `x-agent-name` HTTP header.** This used to have
/// three channels (header + body `source_agent` + "most-recent-agent-in-last-5-min"
/// heuristic), which produced the same attribution-inconsistency class of bug
/// documented in mem0 issues #3218 / #3998. See the research doc at
/// `docs/superpowers/research/` for the full rationale.
///
/// The `source_agent` parameter is still accepted for backwards compatibility
/// with callers that pass it via request body, but it's **deprecated** — a
/// warning is logged each time it's used without a matching header, so we can
/// remove it cleanly in a follow-up.
///
/// Unknown callers become `"unknown"` (which the frontend filter hides from the
/// user-facing dropdown). Honest-unknown beats guessing.
fn extract_agent_name(headers: &HeaderMap, deprecated_body_agent: Option<&str>) -> String {
    if let Some(agent) = headers
        .get("x-agent-name")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        return agent.to_string();
    }
    if let Some(agent) = deprecated_body_agent
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        tracing::warn!(
            "[agent-attribution] body `source_agent={}` used without x-agent-name header — \
             deprecated, please send the `x-agent-name` HTTP header instead",
            agent
        );
        return agent.to_string();
    }
    "unknown".to_string()
}

/// Thin wrapper kept for call-site ergonomics. Previously this also consulted
/// the DB for a "most-recent-agent-in-last-5-min" fallback — that was a loose-
/// observation footgun (DELETE requests could be attributed to whatever agent
/// last called anything, even a totally unrelated tool). Deleted.
async fn extract_agent_name_with_db(
    headers: &HeaderMap,
    deprecated_body_agent: Option<&str>,
    _db: Option<&origin_core::db::MemoryDB>,
) -> String {
    extract_agent_name(headers, deprecated_body_agent)
}

// ===== Space CRUD Handlers =====

#[derive(Debug, Deserialize)]
pub struct CreateSpaceRequest {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateSpaceRequest {
    pub new_name: Option<String>,
    pub description: Option<String>,
}

pub async fn handle_list_spaces(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<Vec<origin_core::db::Space>>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let spaces = db
        .list_spaces()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(spaces))
}

pub async fn handle_create_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<CreateSpaceRequest>,
) -> Result<Json<origin_core::db::Space>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let space = db
        .create_space(&req.name, req.description.as_deref(), false)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(space))
}

pub async fn handle_update_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
    Json(req): Json<UpdateSpaceRequest>,
) -> Result<Json<origin_core::db::Space>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let new_name = req.new_name.as_deref().unwrap_or(&name);
    let space = db
        .update_space(&name, new_name, req.description.as_deref())
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(space))
}

pub async fn handle_delete_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.delete_space(&name, "keep")
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({"deleted": name})))
}

// ===== Nurture Cards =====

#[derive(Debug, Deserialize)]
pub struct NurtureCardsQuery {
    #[serde(default = "default_nurture_limit")]
    pub limit: usize,
    #[serde(default)]
    pub domain: Option<String>,
}

fn default_nurture_limit() -> usize {
    3
}

#[derive(Debug, Serialize)]
pub struct NurtureCardsResponse {
    pub cards: Vec<origin_core::db::MemoryItem>,
}

pub async fn handle_get_nurture_cards(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(query): axum::extract::Query<NurtureCardsQuery>,
) -> Result<Json<NurtureCardsResponse>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let cards = db
        .get_nurture_cards(query.limit, query.domain.as_deref())
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(NurtureCardsResponse { cards }))
}

/// GET /api/memory/rejections
pub async fn handle_get_rejections(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<Vec<origin_core::db::RejectionRecord>>, ServerError> {
    let limit: usize = params
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50)
        .min(500);
    let reason = params.get("reason").map(|s| s.as_str());

    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let records = db.get_rejections(limit, reason).await?;
    Ok(Json(records))
}

/// GET /api/concepts
pub async fn handle_list_concepts(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let status = params.get("status").map(|s| s.as_str()).unwrap_or("active");
    let domain = params.get("domain").map(|s| s.as_str());
    let limit: usize = params
        .get("limit")
        .and_then(|l| l.parse().ok())
        .unwrap_or(50);
    let offset: usize = params
        .get("offset")
        .and_then(|o| o.parse().ok())
        .unwrap_or(0);

    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let concepts = db
        .list_concepts_by_domain(status, domain, limit, offset)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({ "concepts": concepts })))
}

/// GET /api/concepts/:id
pub async fn handle_get_concept(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    match db.get_concept(&id).await {
        Ok(Some(concept)) => Ok(Json(serde_json::json!({ "concept": concept }))),
        Ok(None) => Err(ServerError::NotFound("concept not found".to_string())),
        Err(e) => Err(ServerError::SearchFailed(e.to_string())),
    }
}

/// GET /api/concepts/{id}/sources
///
/// Returns all source memories linked to a concept via the concept_sources join table,
/// enriched with memory metadata for display.
pub async fn handle_get_concept_sources(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<Vec<origin_types::ConceptSourceWithMemory>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let sources = db
        .get_concept_sources(&id)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;

    let source_id_strings: Vec<String> =
        sources.iter().map(|s| s.memory_source_id.clone()).collect();
    let memories = db
        .get_memories_by_source_ids(&source_id_strings)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;

    let result: Vec<origin_types::ConceptSourceWithMemory> = sources
        .iter()
        .map(|s| {
            let memory = memories
                .iter()
                .find(|m| m.source_id == s.memory_source_id)
                .cloned();
            origin_types::ConceptSourceWithMemory {
                source: s.clone(),
                memory,
            }
        })
        .collect();

    Ok(Json(result))
}

/// POST /api/concepts/{id}/archive
pub async fn handle_archive_concept(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.archive_concept(&id)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({"status": "archived"})))
}

/// DELETE /api/concepts/{id}
pub async fn handle_delete_concept(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.delete_concept(&id)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({"status": "deleted"})))
}

/// POST /api/concepts/search
pub async fn handle_search_concepts(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<SearchConceptsRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let results = db
        .search_concepts(&req.query, req.limit.unwrap_or(20))
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({ "concepts": results })))
}

/// POST /api/concepts
pub async fn handle_create_concept(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<CreateConceptRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    let id = origin_core::concepts::Concept::new_id();
    let now = chrono::Utc::now().to_rfc3339();
    let source_refs: Vec<&str> = req.source_memory_ids.iter().map(|s| s.as_str()).collect();
    db.insert_concept(
        &id,
        &req.title,
        req.summary.as_deref(),
        &req.content,
        req.entity_id.as_deref(),
        req.domain.as_deref(),
        &source_refs,
        &now,
    )
    .await
    .map_err(|e| ServerError::IngestFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({ "id": id })))
}

#[derive(Debug, Deserialize)]
pub struct CreateConceptRequest {
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub entity_id: Option<String>,
    #[serde(default)]
    pub domain: Option<String>,
    #[serde(default)]
    pub source_memory_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct SearchConceptsRequest {
    pub query: String,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ExportConceptsRequest {
    pub vault_path: Option<String>,
}

/// POST /api/concepts/export
pub async fn handle_export_concepts(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<ExportConceptsRequest>,
) -> Result<Json<origin_core::export::ExportStats>, ServerError> {
    let concepts = {
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
        db.list_concepts("active", 1000, 0)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
    };
    let vault_path = req
        .vault_path
        .unwrap_or_else(|| "~/obsidian-vault/Origin/concepts".to_string());
    let expanded = if let Some(rest) = vault_path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}/{}", home, rest)
    } else {
        vault_path
    };
    let exporter =
        origin_core::export::obsidian::ObsidianExporter::new(std::path::PathBuf::from(expanded));
    use origin_core::export::ConceptExporter;
    let stats = exporter
        .export_all(&concepts)
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(stats))
}

/// POST /api/concepts/{id}/export
pub async fn handle_export_concept(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(concept_id): Path<String>,
    Json(req): Json<origin_types::requests::ExportConceptRequest>,
) -> Result<Json<origin_types::responses::ExportConceptResponse>, ServerError> {
    // Clone Arc out of the guard so we don't hold the RwLock read guard
    // across the DB await. (CLAUDE.md: never hold tokio::sync::RwLock guards
    // across .await.)
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let concept = db
        .get_concept(&concept_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound(format!("Concept not found: {}", concept_id)))?;

    let expanded = if let Some(rest) = req.vault_path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}/{}", home, rest)
    } else {
        req.vault_path
    };

    let exporter =
        origin_core::export::obsidian::ObsidianExporter::new(std::path::PathBuf::from(expanded));
    use origin_core::export::ConceptExporter;
    let result = exporter
        .export(&concept)
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(origin_types::responses::ExportConceptResponse {
        path: result.path,
    }))
}

// =====================================================================
// Batch 2 — Indexed files / chunks
// =====================================================================

/// GET /api/indexed-files
pub async fn handle_list_indexed_files(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_types::responses::IndexedFilesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let files = db
        .list_indexed_files()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::IndexedFilesResponse {
        files,
    }))
}

/// GET /api/chunks/{source_id}
pub async fn handle_get_chunks(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
) -> Result<Json<Vec<origin_core::db::MemoryDetail>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // Fetch all chunks for this source_id by iterating chunk indices
    let mut chunks = Vec::new();
    let mut idx = 0i32;
    loop {
        match db.get_memory_details("memory", &source_id, idx).await {
            Ok(Some(detail)) => {
                chunks.push(detail);
                idx += 1;
            }
            Ok(None) => {
                // Also try "file" source if no "memory" chunks found
                if idx == 0 {
                    let mut file_idx = 0i32;
                    while let Ok(Some(detail)) =
                        db.get_memory_details("file", &source_id, file_idx).await
                    {
                        chunks.push(detail);
                        file_idx += 1;
                    }
                }
                break;
            }
            Err(_) => break,
        }
    }
    Ok(Json(chunks))
}

/// PUT /api/chunks/{id}/update
pub async fn handle_update_chunk(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<origin_types::requests::UpdateChunkRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.update_memory(&id, &req.content)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// DELETE /api/chunks/time-range
pub async fn handle_delete_by_time_range(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<origin_types::requests::DeleteByTimeRangeRequest>,
) -> Result<Json<origin_types::responses::DeleteCountResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let deleted = db
        .delete_by_time_range(req.start, req.end)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::DeleteCountResponse {
        deleted,
    }))
}

/// POST /api/chunks/delete-bulk
pub async fn handle_delete_bulk(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<origin_types::requests::BulkDeleteRequest>,
) -> Result<Json<origin_types::responses::DeleteCountResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let mut deleted = 0usize;
    for item in &req.items {
        if db
            .delete_by_source_id(&item.source, &item.source_id)
            .await
            .is_ok()
        {
            deleted += 1;
        }
    }
    Ok(Json(origin_types::responses::DeleteCountResponse {
        deleted,
    }))
}

// =====================================================================
// Batch 3 — Entity / Observation CRUD
// =====================================================================

/// PUT /api/memory/entities/{id}/confirm
pub async fn handle_confirm_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<origin_types::requests::ConfirmEntityRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.confirm_entity(&id, req.confirmed)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// DELETE /api/memory/entities/{id}/delete
pub async fn handle_delete_entity(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_entity(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/memory/entities/{entity_id}/observations
pub async fn handle_add_entity_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(entity_id): Path<String>,
    Json(req): Json<origin_types::requests::AddEntityObservationRequest>,
) -> Result<Json<origin_types::responses::AddObservationResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let id = db
        .add_observation(
            &entity_id,
            &req.content,
            req.source_agent.as_deref(),
            req.confidence,
        )
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::AddObservationResponse { id }))
}

/// PUT /api/memory/observations/{id}
pub async fn handle_update_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<origin_types::requests::UpdateObservationRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.update_observation(&id, &req.content)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// DELETE /api/memory/observations/{id}
pub async fn handle_delete_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_observation(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// PUT /api/memory/observations/{id}/confirm
pub async fn handle_confirm_observation(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<origin_types::requests::ConfirmObservationRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.confirm_observation(&id, req.confirmed)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

// =====================================================================
// Batch 4 — Space CRUD
// =====================================================================

/// POST /api/spaces/{name}/pin — toggle space pinned (starred) state
pub async fn handle_pin_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // pin_space maps to toggle_space_starred in the DB
    db.toggle_space_starred(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/spaces/{name}/confirm
pub async fn handle_confirm_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.confirm_space(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/spaces/reorder
pub async fn handle_reorder_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<origin_types::requests::ReorderSpaceRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.reorder_space(&req.name, req.new_order)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/spaces/{name}/star — toggle starred state
pub async fn handle_toggle_space_starred(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let starred = db
        .toggle_space_starred(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "starred": starred })))
}

/// POST /api/documents/{source_id}/space — assign a document to a space (domain)
pub async fn handle_set_document_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
    Json(req): Json<origin_types::requests::SetDocumentSpaceRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.update_domain(&source_id, &req.space_name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

// =====================================================================
// Batch 5 — Activity, tags, capture stats, memory detail
// =====================================================================

/// GET /api/activities
pub async fn handle_list_activities(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<origin_types::responses::ActivityResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let limit: usize = params
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let agent_name = params.get("agent_name").cloned();
    let since: Option<i64> = params.get("since").and_then(|v| v.parse().ok());
    let activities = db
        .list_agent_activity(limit, agent_name.as_deref(), since)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::ActivityResponse {
        activities,
    }))
}

/// GET /api/tags
pub async fn handle_list_tags(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_types::responses::TagsResponse>, ServerError> {
    let s = state.read().await;
    let tags: Vec<String> = s.space_store.tags.iter().cloned().collect();
    Ok(Json(origin_types::responses::TagsResponse { tags }))
}

/// DELETE /api/tags/{name}
pub async fn handle_delete_tag(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let mut s = state.write().await;
    s.space_store.delete_tag(&name);
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// PUT /api/documents/{source_id}/tags
pub async fn handle_set_document_tags(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
    Json(req): Json<origin_types::requests::SetDocumentTagsRequest>,
) -> Result<Json<origin_types::responses::TagsResponse>, ServerError> {
    let mut s = state.write().await;
    let tags = s
        .space_store
        .set_document_tags("memory", &source_id, req.tags);
    Ok(Json(origin_types::responses::TagsResponse { tags }))
}

#[derive(Debug, Deserialize)]
pub struct SuggestTagsQuery {
    pub source: String,
    pub source_id: String,
    /// Optional caller-side hint — the Tauri app passes the name of the
    /// active application at the document's timestamp (activities are
    /// tracked in-process there, not in the DB).
    #[serde(default)]
    pub activity_app: Option<String>,
}

/// GET /api/suggest-tags?source=...&source_id=...&activity_app=...
///
/// Returns candidate tag names derived from a document's chunked content
/// and title, optionally augmented with a caller-supplied activity app
/// name, and with already-assigned tags filtered out.
pub async fn handle_suggest_tags(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(query): axum::extract::Query<SuggestTagsQuery>,
) -> Result<Json<origin_types::responses::TagsResponse>, ServerError> {
    // Read phase: clone db Arc + snapshot the existing tags vec.
    // Dropping the guard before the awaits keeps writers unblocked.
    let (db, existing) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let existing = s
            .space_store
            .get_document_tags(&query.source, &query.source_id);
        (db, existing)
    };

    let chunks = db
        .get_memories_by_source_id(&query.source, &query.source_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    let chunk_contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let title = chunks.first().map(|c| c.title.clone()).unwrap_or_default();

    let mut tags = origin_core::tags::suggest_tags_for_document(&chunk_contents, &title, &existing);

    // Merge caller-side activity hint (app name), respecting dedup + the
    // "not already assigned" filter.
    if let Some(app) = query.activity_app {
        let normalized = app.trim().to_lowercase();
        if !normalized.is_empty()
            && !existing.iter().any(|t| t == &normalized)
            && !tags.iter().any(|t| t == &normalized)
        {
            tags.push(normalized);
            tags.sort();
        }
    }

    Ok(Json(origin_types::responses::TagsResponse { tags }))
}

/// GET /api/capture-stats
pub async fn handle_capture_stats(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let count = db.count().await.unwrap_or(0);
    Ok(Json(serde_json::json!({
        "total_chunks": count,
    })))
}

/// GET /api/memory/{id}/detail
pub async fn handle_get_memory_detail(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<origin_types::responses::MemoryDetailResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let memory = db
        .get_memory_detail(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::MemoryDetailResponse {
        memory,
    }))
}

/// GET /api/memory/by-ids?ids=mem_a,mem_b,...
///
/// Batch-fetch multiple memories by source_id in a single round trip.
/// The response preserves input order; missing ids are silently omitted.
/// Used by ConceptDetail to load all source memories at once.
pub async fn handle_get_memories_by_ids(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<origin_types::responses::PinnedMemoriesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let ids: Vec<String> = params
        .get("ids")
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let memories = db
        .get_memories_by_source_ids(&ids)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::PinnedMemoriesResponse {
        memories,
    }))
}

/// GET /api/memory/{id}/versions
pub async fn handle_get_version_chain(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<origin_types::responses::VersionChainResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let versions = db
        .get_version_chain(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::VersionChainResponse {
        versions,
    }))
}

/// PUT /api/memory/{id}/update
pub async fn handle_update_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<origin_types::requests::UpdateMemoryRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    if let Some(content) = &req.content {
        db.update_memory(&id, content)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    }
    if let Some(domain) = &req.domain {
        db.update_domain(&id, domain)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    }
    if let Some(true) = req.confirmed {
        db.confirm_memory(&id)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    }
    if let Some(memory_type) = &req.memory_type {
        db.update_memory_type(&id, memory_type)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    }
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// PUT /api/memory/{id}/stability
pub async fn handle_set_stability(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<origin_types::requests::SetStabilityRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.set_stability(&id, &req.stability)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/memory/{id}/correct — apply LLM correction to a memory
pub async fn handle_correct_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<origin_types::requests::CorrectMemoryRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let (db, llm) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let llm = s.llm.clone();
        (db, llm)
    };
    let llm =
        llm.ok_or_else(|| ServerError::Internal("LLM not available for correction".to_string()))?;
    // Get the existing memory content
    let memory = db
        .get_memory_detail(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::Internal(format!("memory {} not found", id)))?;

    // Build a correction prompt for the LLM
    let prompt = format!(
        "Original memory:\n{}\n\nCorrection instruction: {}\n\nProvide the corrected memory content. Output ONLY the corrected text, nothing else.",
        memory.content, req.correction_prompt
    );
    let corrected = llm
        .generate(origin_core::llm_provider::LlmRequest {
            system_prompt: Some("You are a memory correction assistant. Apply the correction to the memory and return only the corrected text.".to_string()),
            user_prompt: prompt,
            max_tokens: 2048,
            temperature: 0.1,
            label: None,
        })
        .await
        .map_err(|e| ServerError::Internal(format!("LLM correction failed: {}", e)))?;

    // Update the memory with corrected content
    db.update_memory(&id, &corrected)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "corrected": corrected,
        "source_id": id,
    })))
}

// =====================================================================
// Batch 6 — Decisions, briefing, working memory, profile narrative, pinned
// =====================================================================

/// GET /api/decisions
pub async fn handle_list_decisions(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<origin_types::responses::DecisionsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let domain = params.get("domain").cloned();
    let limit: usize = params
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let decisions = db
        .list_memories(domain.as_deref(), Some("decision"), None, None, limit)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::DecisionsResponse {
        decisions,
    }))
}

/// GET /api/decisions/domains
pub async fn handle_list_decision_domains(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_types::responses::DecisionDomainsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let domains = db
        .list_decision_domains()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::DecisionDomainsResponse {
        domains,
    }))
}

/// GET /api/briefing
pub async fn handle_get_briefing(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_core::briefing::BriefingResponse>, ServerError> {
    let (db, llm, prompts, tuning) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let llm = s.llm.clone();
        let prompts = s.prompts.clone();
        let tuning = s.tuning.briefing.clone();
        (db, llm, prompts, tuning)
    };
    let briefing = origin_core::briefing::generate_briefing(&db, llm.as_deref(), &prompts, &tuning)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(briefing))
}

/// GET /api/profile/narrative
///
/// Cache-first: returns the cached narrative immediately if present, so the
/// profile page loads instantly instead of waiting on an LLM call every time.
/// Falls through to `generate_narrative` (which writes to cache on success)
/// when the cache is empty. Explicit regeneration still goes through
/// `/api/profile/narrative/regenerate`.
pub async fn handle_get_profile_narrative(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_core::narrative::NarrativeResponse>, ServerError> {
    let (db, llm, prompts, tuning) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let llm = s.llm.clone();
        let prompts = s.prompts.clone();
        let tuning = s.tuning.narrative.clone();
        (db, llm, prompts, tuning)
    };

    // 1. Try the cache first. If we have a stored narrative with content,
    //    return it immediately — no LLM round-trip on page load.
    if let Ok(Some((content, generated_at, memory_count))) = db.get_cached_narrative().await {
        if !content.is_empty() {
            return Ok(Json(origin_core::narrative::NarrativeResponse {
                content,
                generated_at,
                is_stale: false,
                memory_count,
            }));
        }
    }

    // 2. Nothing cached — generate fresh (this call also writes to the cache
    //    so subsequent loads are instant).
    let narrative =
        origin_core::narrative::generate_narrative(&db, llm.as_deref(), &prompts, &tuning)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(narrative))
}

/// POST /api/profile/narrative/regenerate
pub async fn handle_regenerate_narrative(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_core::narrative::NarrativeResponse>, ServerError> {
    // Same as get, but always regenerates (no cache)
    let (db, llm, prompts, tuning) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let llm = s.llm.clone();
        let prompts = s.prompts.clone();
        let tuning = s.tuning.narrative.clone();
        (db, llm, prompts, tuning)
    };
    let narrative =
        origin_core::narrative::generate_narrative(&db, llm.as_deref(), &prompts, &tuning)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(narrative))
}

/// GET /api/memory/pinned
pub async fn handle_list_pinned_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<origin_types::responses::PinnedMemoriesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let memories = db
        .list_pinned_memories()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::PinnedMemoriesResponse {
        memories,
    }))
}

/// POST /api/memory/{id}/pin
pub async fn handle_pin_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.pin_memory(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/memory/{id}/unpin
pub async fn handle_unpin_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.unpin_memory(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

/// GET /api/memory/pending-revision/{source_id}
pub async fn handle_get_pending_revision(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
) -> Result<Json<Option<origin_core::db::PendingRevision>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let revision = db
        .get_pending_revision_for(&source_id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(revision))
}

/// POST /api/snapshots/{id}/delete
pub async fn handle_delete_snapshot(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_snapshot(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

// ===== Session Snapshots =====

#[derive(Debug, Deserialize)]
pub struct SnapshotsQuery {
    #[serde(default = "default_snapshots_limit")]
    pub limit: usize,
}

fn default_snapshots_limit() -> usize {
    10
}

/// GET /api/snapshots?limit=N
///
/// Returns the N most recent session snapshots (default 10).
pub async fn handle_list_snapshots(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(query): axum::extract::Query<SnapshotsQuery>,
) -> Result<Json<Vec<origin_types::SessionSnapshot>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let rows = db
        .get_recent_snapshots(query.limit)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    let snapshots = rows
        .into_iter()
        .map(|r| origin_types::SessionSnapshot {
            id: r.id,
            activity_id: r.activity_id,
            started_at: r.started_at,
            ended_at: r.ended_at,
            primary_apps: r.primary_apps,
            summary: r.summary,
            tags: r.tags,
            capture_count: r.capture_count as u64,
        })
        .collect();
    Ok(Json(snapshots))
}

/// GET /api/snapshots/{id}/captures
///
/// Returns capture metadata (no full text) for a snapshot.
pub async fn handle_get_snapshot_captures(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<Vec<origin_types::SnapshotCapture>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let rows = db
        .get_captures_for_snapshot(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    let captures = rows
        .into_iter()
        .map(|c| origin_types::SnapshotCapture {
            source_id: c.source_id,
            app_name: c.app_name,
            window_title: c.window_title,
            timestamp: c.timestamp,
            source: c.source,
        })
        .collect();
    Ok(Json(captures))
}

/// GET /api/snapshots/{id}/captures-with-content
///
/// Returns captures for a snapshot plus their full chunked text and LLM
/// summary. Mirrors the pre-split `get_snapshot_captures_with_content`
/// Tauri command: for each `capture_refs` row, maps the router-side
/// trigger name (`focus`/`hotkey`/`snip`/`thought`) to the DB-side
/// source name (`focus_capture`/`hotkey_capture`/`snip_capture`/
/// `quick_thought`), fetches chunks, joins their content, and looks up
/// the summary via list_indexed_files.
pub async fn handle_get_snapshot_captures_with_content(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<Vec<origin_types::SnapshotCaptureWithContent>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let captures = db
        .get_captures_for_snapshot(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    // list_indexed_files is a single aggregate scan, so fetch once
    // upfront rather than once per capture.
    let indexed_files = if captures.is_empty() {
        Vec::new()
    } else {
        db.list_indexed_files().await.unwrap_or_default()
    };

    let mut results = Vec::with_capacity(captures.len());
    for c in captures {
        let vdb_source = match c.source.as_str() {
            "focus" => "focus_capture",
            "hotkey" => "hotkey_capture",
            "snip" => "snip_capture",
            "thought" => "quick_thought",
            other => other,
        };

        let chunks = db
            .get_memories_by_source_id(vdb_source, &c.source_id)
            .await
            .unwrap_or_default();
        let content = chunks
            .iter()
            .map(|ch| ch.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        let summary = indexed_files
            .iter()
            .find(|f| f.source == vdb_source && f.source_id == c.source_id)
            .and_then(|f| f.summary.clone());

        results.push(origin_types::SnapshotCaptureWithContent {
            source_id: c.source_id,
            app_name: c.app_name,
            window_title: c.window_title,
            timestamp: c.timestamp,
            source: c.source,
            content,
            summary,
        });
    }

    Ok(Json(results))
}

/// POST /api/memory/{id}/update-concept
pub async fn handle_update_concept(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<origin_types::requests::UpdateConceptRequest>,
) -> Result<Json<origin_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // Preserve existing source_memory_ids — the HTTP request only carries
    // the new content body. Passing &[] here would wipe the concept's
    // source list, causing silent data loss.
    let existing_sources: Vec<String> = db
        .get_concept(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map(|c| c.source_memory_ids)
        .unwrap_or_default();
    let existing_refs: Vec<&str> = existing_sources.iter().map(String::as_str).collect();
    db.update_concept_content(&id, &req.content, &existing_refs, "manual_edit")
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(origin_types::responses::SuccessResponse { ok: true }))
}

// ===== Recent activity feed =====

#[derive(Debug, Default, serde::Deserialize)]
pub struct RecentActivityQuery {
    #[serde(default)]
    pub limit: Option<i64>,
    #[serde(default)]
    pub since_ms: Option<i64>,
}

/// GET /api/memory/recent — top-N memory activity with badge deltas.
/// `since_ms` scopes badge derivation only; the feed is always top-N by recency.
pub async fn handle_recent_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(q): axum::extract::Query<RecentActivityQuery>,
) -> Result<Json<Vec<origin_types::RecentActivityItem>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let items = db
        .list_recent_memories(q.limit.unwrap_or(10), q.since_ms)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(items))
}

/// GET /api/memory/unconfirmed — top-N unconfirmed memories (`confirmed` = 0 or NULL).
///
/// Every item comes back with `badge = NeedsReview`. This feeds Worth-a-glance
/// on the home page so unconfirmed memories always have a way to surface, even
/// when the `lastVisitMs` delta window is too tight to produce a `new` badge.
pub async fn handle_list_unconfirmed_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(q): axum::extract::Query<RecentActivityQuery>,
) -> Result<Json<Vec<origin_types::RecentActivityItem>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let items = db
        .list_unconfirmed_memories(q.limit.unwrap_or(6))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(items))
}

#[cfg(test)]
mod split_tests {
    use super::*;

    // Helper: construct a StoreMemoryResponse from the post-refactor helper and assert shape.
    #[test]
    fn warnings_excludes_extraction_status_when_no_fields_extracted() {
        // Simulates branch-3: no LLM extraction, no agent-supplied fields.
        let (warnings, extraction_method) = compute_warnings_and_extraction(
            /* extracted_fields */ None, /* agent_fields */ None,
            /* memory_type_str */ "fact",
        );
        assert_eq!(extraction_method, "none");
        assert!(
            warnings.is_empty(),
            "warnings must be empty when no fields extracted; got: {:?}",
            warnings
        );
    }

    #[test]
    fn warnings_reports_schema_validation_when_agent_supplies_fields() {
        // Simulates branch-2: agent-supplied structured_fields that fail validation.
        let agent = serde_json::json!({"wrong_field": "value"});
        let (warnings, extraction_method) = compute_warnings_and_extraction(
            /* extracted_fields */ None,
            /* agent_fields */ Some(&agent),
            /* memory_type_str */ "decision",
        );
        assert_eq!(extraction_method, "agent");
        // MemorySchema for "decision" has required fields the agent did not provide,
        // so validation should produce at least one warning.
        assert!(
            !warnings.is_empty(),
            "expected schema-validation warnings for decision missing required fields; got: {:?}",
            warnings
        );
        // None of the warnings should be the old extraction-status string.
        assert!(
            !warnings
                .iter()
                .any(|w| w.contains("no structured fields extracted")),
            "extraction-status leaked into warnings: {:?}",
            warnings
        );
    }

    #[test]
    fn warnings_reports_llm_extraction_when_backend_fills_fields() {
        // Simulates branch-1: LLM extracted, validates cleanly.
        let extracted = serde_json::json!({"claim": "x"}).to_string();
        let (_warnings, extraction_method) = compute_warnings_and_extraction(
            /* extracted_fields */ Some(&extracted),
            /* agent_fields */ None,
            /* memory_type_str */ "fact",
        );
        assert_eq!(extraction_method, "llm");
    }
}

#[cfg(test)]
mod recent_memory_endpoint_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;

    #[tokio::test]
    async fn get_recent_memories_route_is_registered() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/memory/recent?limit=5")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        // Route exists => NOT a 404. With no DB initialised we expect 503.
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_recent_memories_without_db_returns_503() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/memory/recent?limit=5&since_ms=1000")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn get_unconfirmed_memories_route_is_registered() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/memory/unconfirmed?limit=5")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        // Route exists => NOT a 404. With no DB initialised we expect 503.
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_unconfirmed_memories_without_db_returns_503() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let req = Request::builder()
            .method("GET")
            .uri("/api/memory/unconfirmed?limit=5")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}

/// Agent attribution regression tests for `/api/memory/search`.
///
/// Locks in that both the `x-agent-name` header path and the deprecated body
/// `source_agent` fallback correctly write the resolved agent name into
/// `agent_activity.agent_name`. Previously the search handler passed `None`
/// for the body fallback, so requests that sent only body `source_agent` (no
/// header) were logged as `agent_name="unknown"` — producing the "unknown"
/// rows surfaced by `/api/retrievals/recent` in the home-v2 delta feed.
#[cfg(test)]
mod search_agent_attribution_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;

    async fn build_state_with_db() -> (Arc<RwLock<ServerState>>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("failed to create tempdir");
        let emitter: Arc<dyn origin_core::events::EventEmitter> =
            Arc::new(origin_core::events::NoopEmitter);
        let db = origin_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new should succeed");
        let server_state = ServerState {
            db: Some(Arc::new(db)),
            ..Default::default()
        };
        (Arc::new(RwLock::new(server_state)), tmp)
    }

    async fn fetch_activities(app: axum::Router) -> Vec<origin_types::AgentActivityRow> {
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/activities?limit=20")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "activities should succeed");
        let bytes = axum::body::to_bytes(resp.into_body(), 1_048_576)
            .await
            .unwrap();
        let wrapper: origin_types::responses::ActivityResponse =
            serde_json::from_slice(&bytes).expect("parse ActivityResponse");
        wrapper.activities
    }

    #[tokio::test]
    async fn search_with_x_agent_name_header_persists_attribution() {
        let (state, _tmp) = build_state_with_db().await;
        let app = crate::router::build_router(state);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memory/search")
                    .header("x-agent-name", "test-agent")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "search request should succeed, got {}",
            resp.status()
        );

        let activities = fetch_activities(app).await;
        assert!(
            activities
                .iter()
                .any(|a| a.action == "search" && a.agent_name == "test-agent"),
            "expected a search activity attributed to test-agent, got: {:?}",
            activities
                .iter()
                .map(|a| (a.action.clone(), a.agent_name.clone()))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn search_with_body_source_agent_persists_attribution() {
        // Regression: before the fix, the search handler resolved attribution
        // via `extract_agent_name_with_db(&headers, None, ...)` — discarding
        // the body `source_agent` field entirely. Result: requests that sent
        // body `source_agent` but no `x-agent-name` header were attributed to
        // "unknown" in `agent_activity`, masking real callers in
        // `/api/retrievals/recent`.
        let (state, _tmp) = build_state_with_db().await;
        let app = crate::router::build_router(state);

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memory/search")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"query":"hello","source_agent":"body-agent"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "search request should succeed, got {}",
            resp.status()
        );

        let activities = fetch_activities(app).await;
        assert!(
            activities
                .iter()
                .any(|a| a.action == "search" && a.agent_name == "body-agent"),
            "expected a search activity attributed to body-agent (from body `source_agent`), \
             got: {:?}",
            activities
                .iter()
                .map(|a| (a.action.clone(), a.agent_name.clone()))
                .collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
mod dismiss_contradiction_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;

    async fn build_state_with_db() -> (Arc<RwLock<ServerState>>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("failed to create tempdir");
        let emitter: Arc<dyn origin_core::events::EventEmitter> =
            Arc::new(origin_core::events::NoopEmitter);
        let db = origin_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new should succeed");
        let server_state = ServerState {
            db: Some(Arc::new(db)),
            ..Default::default()
        };
        (Arc::new(RwLock::new(server_state)), tmp)
    }

    #[tokio::test]
    async fn dismiss_contradiction_route_is_registered() {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memory/contradiction/mem_nonexistent/dismiss")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Either 2xx (no-op success) or 503 (db not initialized) is OK — just not 404-from-router.
        assert_ne!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "route must be registered; got 404 which means it is missing from the router"
        );
    }

    #[tokio::test]
    async fn dismiss_contradiction_clears_awaiting_review_row() {
        let (state, _tmp) = build_state_with_db().await;

        // Seed: insert a refinement_queue row with action=detect_contradiction status=awaiting_review.
        let source_id = "mem_contradiction_target";
        {
            let s = state.read().await;
            let db = s.db.as_ref().unwrap();
            db.insert_refinement_proposal(
                "ref_contradiction_1",
                "detect_contradiction",
                &[source_id.to_string(), "mem_other".to_string()],
                None,
                0.9,
            )
            .await
            .unwrap();
            // Promote to awaiting_review (default insert status is 'pending').
            db.resolve_refinement("ref_contradiction_1", "awaiting_review")
                .await
                .unwrap();
        }

        // Confirm the memory is flagged before dismissal.
        {
            let s = state.read().await;
            let db = s.db.as_ref().unwrap();
            let flagged = db
                .pending_review_memory_ids(&[source_id.to_string()])
                .await
                .unwrap();
            assert!(
                flagged.contains(source_id),
                "memory should be flagged as needs-review before dismiss"
            );
        }

        let app = crate::router::build_router(state.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/memory/contradiction/{}/dismiss", source_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "dismiss_contradiction should return 200"
        );

        // Verify the row is no longer awaiting_review.
        let s = state.read().await;
        let db = s.db.as_ref().unwrap();
        let flagged = db
            .pending_review_memory_ids(&[source_id.to_string()])
            .await
            .unwrap();
        assert!(
            !flagged.contains(source_id),
            "memory should be cleared from needs-review after dismiss"
        );
    }
}
