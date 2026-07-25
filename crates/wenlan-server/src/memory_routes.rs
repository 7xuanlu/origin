// SPDX-License-Identifier: Apache-2.0
use crate::error::ServerError;
use crate::state::ServerState;
use axum::{
    extract::{Path, State},
    http::HeaderMap,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_core::sources::compute_effective_confidence;
use wenlan_types::requests::{
    AddObservationRequest, ConfirmRequest, CreateConceptRequest, CreateEntityRequest,
    CreateRelationRequest, ExportPagesRequest, ListMemoriesRequest, SearchMemoryRequest,
    SearchPagesRequest, StoreMemoryRequest,
};
use wenlan_types::responses::{
    AddObservationResponse, ConfirmResponse, CreateEntityResponse, CreatePageResponse,
    CreateRelationResponse, DeleteResponse, ListMemoriesResponse, SearchMemoryResponse,
    StoreMemoryResponse,
};
use wenlan_types::sources::{stability_tier, MemoryType, RawDocument, StabilityTier};

pub(crate) async fn registered_request_space(
    db: &wenlan_core::db::MemoryDB,
    requested: &Option<String>,
    context: &str,
) -> Result<Option<String>, ServerError> {
    let proposed = requested
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let registered = db
        .registered_space_or_none(requested.as_deref())
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    if registered.is_none() {
        if let Some(space) = proposed {
            tracing::warn!(
                "[memory] ignoring unregistered space {:?} for {}; using unscoped fallback",
                space,
                context
            );
        }
    }
    Ok(registered)
}

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
pub struct LinkEntityRequest {
    pub source_id: String,
    pub entity_id: String,
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
            let schema = wenlan_core::memory_schema::MemorySchema::for_type(memory_type_str);
            (schema.validate(&fields), "llm".to_string())
        }
        (None, Some(_)) => {
            let fields = fields_map(None, agent_fields).unwrap_or_default();
            let schema = wenlan_core::memory_schema::MemorySchema::for_type(memory_type_str);
            (schema.validate(&fields), "agent".to_string())
        }
        (None, None) => (Vec::new(), "none".to_string()),
    }
}

fn fixed_enrichment_origin(
    caller_supplied_memory_type: bool,
    caller_supplied_profile_alias: bool,
    caller_supplied_structured_fields: bool,
    rejected_explicit_space: bool,
) -> wenlan_core::db::EnrichmentOrigin {
    wenlan_core::db::EnrichmentOrigin {
        memory_type_explicit: caller_supplied_memory_type && !caller_supplied_profile_alias,
        structured_fields_explicit: caller_supplied_structured_fields,
        space_rejected: rejected_explicit_space,
    }
}

/// POST /api/memory/store
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StoreLockTestStage {
    Dedup,
    AgentGate,
    EntityResolution,
    ActivityAgentLookup,
    ActivityLog,
}

#[cfg(test)]
struct StoreLockTestHook {
    stage: StoreLockTestStage,
    reached: tokio::sync::Notify,
    release: tokio::sync::Notify,
}

#[cfg(test)]
fn store_lock_test_hook() -> &'static std::sync::Mutex<Option<Arc<StoreLockTestHook>>> {
    static HOOK: std::sync::OnceLock<std::sync::Mutex<Option<Arc<StoreLockTestHook>>>> =
        std::sync::OnceLock::new();
    HOOK.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(test)]
async fn wait_at_store_lock_test_hook(stage: StoreLockTestStage) {
    let hook = store_lock_test_hook().lock().unwrap().clone();
    if let Some(hook) = hook.filter(|hook| hook.stage == stage) {
        hook.reached.notify_one();
        hook.release.notified().await;
    }
}

#[cfg(test)]
struct StoreLockTestHookRegistration(Arc<StoreLockTestHook>);

#[cfg(test)]
impl StoreLockTestHookRegistration {
    fn install(stage: StoreLockTestStage) -> Self {
        let hook = Arc::new(StoreLockTestHook {
            stage,
            reached: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
        });
        let mut slot = store_lock_test_hook().lock().unwrap();
        if slot.is_some() {
            drop(slot);
            panic!("store lock test hook already installed");
        }
        *slot = Some(hook.clone());
        drop(slot);
        Self(hook)
    }

    fn hook(&self) -> &Arc<StoreLockTestHook> {
        &self.0
    }
}

#[cfg(test)]
impl Drop for StoreLockTestHookRegistration {
    fn drop(&mut self) {
        *store_lock_test_hook().lock().unwrap() = None;
        // notify_one stores a permit if the task cloned the hook but has not
        // parked yet, so cleanup is safe even on a multi-thread test runtime.
        self.0.release.notify_one();
    }
}

pub async fn handle_store_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(mut req): Json<StoreMemoryRequest>,
) -> Result<Json<StoreMemoryResponse>, ServerError> {
    // Apply X-Origin-Space header as fallback only when body omits `space`.
    if req.space.is_none() {
        req.space = header_space;
    }
    let requested_space = req
        .space
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let ignored_unregistered_space = if let Some(space) = requested_space {
        let db = {
            let s = state.read().await;
            s.db.clone().ok_or(ServerError::DbNotInitialized)?
        };
        if let Some(registered_space) = db
            .registered_space_or_none(Some(&space))
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
        {
            req.space = Some(registered_space);
            None
        } else {
            tracing::warn!(
                "[memory] ignoring unregistered space {:?}; storing memory uncategorized",
                space
            );
            req.space = None;
            Some(space)
        }
    } else {
        req.space = None;
        None
    };
    let trimmed_content = req.content.trim();
    if trimmed_content.len() < 10 {
        return Err(ServerError::ValidationError(
            "Memory content must be at least 10 characters".into(),
        ));
    }

    // Dedup check
    let db = {
        let s = state.read().await;
        s.db.clone()
    };
    if let Some(db) = db {
        #[cfg(test)]
        wait_at_store_lock_test_hook(StoreLockTestStage::Dedup).await;
        if db.has_memory_content(&req.content).await.unwrap_or(false) {
            return Err(ServerError::ValidationError(
                "Duplicate: a memory with this content already exists".into(),
            ));
        }
    }

    // Validate caller-supplied memory_type — parse and keep it as-is. Profile
    // aliases now resolve in the ambient classification lane; previously this
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
            // Stored as "identity" as a conservative placeholder; ambient classify
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

    // Phase 1: Agent gating + resolve the same hard-pinned background route
    // the scheduler will use. A loaded slot is capability, not consent.
    //
    // Resolve the agent name via `extract_agent_name` (header-canonical)
    // before gating. Previously this read `req.source_agent` (body-only),
    // which meant a caller using the `x-agent-name` header — the canonical
    // channel since the channel-collapse — silently bypassed registration
    // and gating entirely (got `None → "full"` auto-trust).
    //
    // Prompts, classify/extract LLM calls, and classification writebacks all
    // moved to the ambient scheduler — the sync path only needs
    // to know whether automatic enrichment is authorized and currently
    // available so it can return a truthful response state.
    let resolved_agent = extract_agent_name(&headers, req.source_agent.as_deref());
    let runtime_config = wenlan_core::config::load_config();
    let everyday_pin =
        wenlan_core::refinery::EverydaySource::parse(runtime_config.everyday_source.as_deref());
    let (db, api_llm, external_llm, local_llm) = {
        let s = state.read().await;
        (
            s.db.clone().ok_or(ServerError::DbNotInitialized)?,
            s.api_llm.clone(),
            s.external_llm.clone(),
            s.llm.clone(),
        )
    };
    let trust_level = if resolved_agent == "unknown" {
        // No agent identified at all → local/first-party write, full trust.
        "full".to_string()
    } else {
        #[cfg(test)]
        wait_at_store_lock_test_hook(StoreLockTestStage::AgentGate).await;
        db.check_agent_for_write(&resolved_agent)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?
    };
    let ambient_route_mode = wenlan_core::refinery::resolve_everyday(
        everyday_pin,
        api_llm.as_ref(),
        external_llm.as_ref(),
        local_llm.as_ref(),
    )
    .mode;

    // Placeholder classification. The ambient classification lane may replace
    // these via `db.apply_enrichment(...)` and `db.set_document_tags(...)`
    // after the quiet/cooldown gates admit the memory.
    let memory_type_str = validated_memory_type
        .clone()
        .unwrap_or_else(|| "fact".to_string());
    let classified_tags: Vec<String> = Vec::new();
    let classified_quality: Option<String> = None;

    // Structured fields / retrieval_cue are caller-supplied or deferred to the
    // ambient extractor. Sync path never runs the extract LLM call anymore.
    let extracted_fields: Option<String> = None;
    let extracted_cue: Option<String> = None;

    // Capture before `req.structured_fields` is consumed into the RawDocument —
    // the durable enrichment origin uses this to decide whether the ambient
    // extract pass. If the caller already supplied fields, we skip extract.
    let caller_supplied_structured_fields = req.structured_fields.is_some();
    let enrichment_origin = fixed_enrichment_origin(
        caller_supplied_memory_type,
        caller_supplied_a_profile_alias,
        caller_supplied_structured_fields,
        ignored_unregistered_space.is_some(),
    );

    // Phase 2b-validate: split into warnings (schema-validation only) and extraction_method (status label).
    let (mut warnings, extraction_method) = compute_warnings_and_extraction(
        extracted_fields.as_deref(),
        req.structured_fields.as_ref(),
        &memory_type_str,
    );
    if let Some(space) = ignored_unregistered_space.as_deref() {
        warnings.push(format!(
            "Space '{space}' is not registered; stored uncategorized. Run `wenlan spaces add {space}` before using it."
        ));
    }

    // Phase 2c: Entity resolution
    let resolved_entity_id = if let Some(ref direct_id) = req.entity_id {
        Some(direct_id.clone())
    } else if let Some(ref entity_name) = req.entity {
        let db = {
            let s = state.read().await;
            s.db.clone()
        };
        if let Some(db) = db {
            #[cfg(test)]
            wait_at_store_lock_test_hook(StoreLockTestStage::EntityResolution).await;
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

    let pending_revision = false;
    let final_supersedes = req.supersedes.clone();

    let db = {
        let s = state.read().await;
        s.db.clone()
    };
    #[cfg(test)]
    wait_at_store_lock_test_hook(StoreLockTestStage::ActivityAgentLookup).await;
    let agent_for_activity =
        extract_agent_name_with_db(&headers, req.source_agent.as_deref(), db.as_deref()).await;
    let supersedes_for_activity = final_supersedes.clone();

    let supersede_mode = if memory_type_str == "decision" {
        "archive".to_string()
    } else {
        "hide".to_string()
    };

    let final_domain = req.space;
    let doc = RawDocument {
        source: "memory".to_string(),
        source_id: source_id.clone(),
        title,
        summary: None,
        content: req.content.clone(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        metadata: HashMap::new(),
        memory_type: Some(memory_type_str.clone()),
        space: final_domain.clone(),
        source_agent: req.source_agent,
        confidence: Some(effective_confidence),
        confirmed,
        stability: Some(stability.to_string()),
        supersedes: final_supersedes,
        pending_revision,
        entity_id: resolved_entity_id.clone(),
        quality: classified_quality.clone(),
        importance: None,
        is_recap: false,
        enrichment_status: "raw".to_string(),
        supersede_mode,
        structured_fields: req
            .structured_fields
            .map(|v| v.to_string())
            .or(extracted_fields),
        retrieval_cue: req.retrieval_cue.clone().or(extracted_cue),
        source_text: None,
        content_hash: None,
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
        let redacted = wenlan_core::privacy::redact_pii(&doc.content);
        let chunker = wenlan_core::chunker::ChunkingEngine::new();
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
    let origin_db = db_fallback.clone().ok_or(ServerError::DbNotInitialized)?;
    origin_db
        .upsert_enrichment_origin(&source_id, enrichment_origin)
        .await
        .map_err(|e| ServerError::IngestFailed(e.to_string()))?;

    let chunks_created = if let Some(batcher) = ingest_batcher {
        // Coalesced path: concurrent callers share one FastEmbed call +
        // one libSQL transaction. Gate runs inside the coalescer flush.
        // See `ingest_batcher.rs` for details.
        use crate::ingest_batcher::StoreOutcome;
        let outcome = match batcher.submit(doc, chunks_predicted).await {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = origin_db.delete_enrichment_origin(&source_id).await;
                return Err(ServerError::IngestFailed(error));
            }
        };
        match outcome {
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
                let _ = origin_db.delete_enrichment_origin(&source_id).await;
                return Err(ServerError::QualityGateRejected {
                    reason,
                    detail,
                    similar_to,
                });
            }
            StoreOutcome::UpsertFailed(msg) => {
                let _ = origin_db.delete_enrichment_origin(&source_id).await;
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
                    wenlan_core::quality_gate::GateResult {
                        admitted: false,
                        reason: Some(
                            wenlan_core::quality_gate::RejectionReason::EmbeddingUnavailable(
                                e.to_string(),
                            ),
                        ),
                        scores: wenlan_core::quality_gate::GateScores {
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
            let _ = origin_db.delete_enrichment_origin(&source_id).await;
            return Err(ServerError::QualityGateRejected {
                reason: rej_reason,
                detail: rej_detail,
                similar_to: similar_source_id,
            });
        }
        match db.upsert_documents(vec![doc]).await {
            Ok(chunks) => chunks,
            Err(error) => {
                let _ = origin_db.delete_enrichment_origin(&source_id).await;
                return Err(ServerError::IngestFailed(error.to_string()));
            }
        }
    };

    if chunks_created == 0 {
        let _ = origin_db.delete_enrichment_origin(&source_id).await;
        return Err(ServerError::ValidationError(
            "Memory produced no indexable content after processing".into(),
        ));
    }

    // Classified tags are now written by the ambient classification lane —
    // `classified_tags` is always empty at this point because classify moved
    // off the sync path. Kept as a no-op branch for the rare caller that
    // pre-supplies tags in the future; can be removed with an API cleanup.
    let _ = classified_tags; // intentionally unused

    // Log agent activity
    let db = {
        let s = state.read().await;
        s.db.clone()
    };
    if let Some(db) = db {
        #[cfg(test)]
        wait_at_store_lock_test_hook(StoreLockTestStage::ActivityLog).await;
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

    // Record the write event for steep burst detection and recap batching.
    // Capture the timestamp immediately after the durable store, never after
    // background enrichment.
    {
        let s = state.read().await;
        s.write_signal.record(&resolved_agent);
    }

    // Automatic enrichment is intentionally not spawned from the request path.
    // The ambient scheduler owns admission, ordering, retries, and the one-call
    // budget for every automatic inference stage.

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
        let (db_for_ms, maintenance) = {
            let s = state.read().await;
            (s.db.clone(), s.maintenance_coordinator.clone())
        };
        if let Some(db_for_ms) = db_for_ms {
            let emitter_for_ms: Arc<dyn wenlan_core::events::EventEmitter> =
                Arc::new(wenlan_core::events::NoopEmitter);
            let source_for_ms = agent_for_activity.clone();
            let memory_id_for_ms = source_id.clone();
            tokio::spawn(async move {
                let _maintenance_guard = maintenance.begin_background().await;
                let ev =
                    wenlan_core::onboarding::MilestoneEvaluator::new(&db_for_ms, emitter_for_ms);
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

    // Build caller-facing status from the effective hard-pin route. Recall is
    // available immediately in every state; only a healthy explicit pin may
    // promise that automatic enrichment will eventually run.
    let (enrichment, hint) = match ambient_route_mode {
        wenlan_core::refinery::RouteMode::Pinned => (
            "pending".to_string(),
            "Stored. Recall is available now; Wenlan will quietly enrich \
             classification and page links in the background."
                .to_string(),
        ),
        wenlan_core::refinery::RouteMode::Unconfigured => (
            "paused".to_string(),
            "Stored. Recall is available now; background enrichment is paused \
             until you choose a model source."
                .to_string(),
        ),
        wenlan_core::refinery::RouteMode::PinnedUnavailable => (
            "paused".to_string(),
            "Stored. Recall is available now; background enrichment is paused \
             because the selected model source is unavailable."
                .to_string(),
        ),
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

/// Splits a flat Vec<SearchResult> (as returned by `search_memory_cross_rerank`)
/// into memory rows and page-channel rows.
///
/// Returns `(memory_rows, Some(page_rows))` when any page rows are present,
/// or `(memory_rows, None)` when the slice contains no page rows.
/// Pure function — no I/O, easy to unit-test.
pub(crate) fn partition_search_pages(
    rows: Vec<wenlan_types::memory::SearchResult>,
) -> (
    Vec<wenlan_types::memory::SearchResult>,
    Option<Vec<wenlan_types::memory::SearchResult>>,
) {
    let (pages, memories): (Vec<_>, Vec<_>) = rows.into_iter().partition(|r| r.source == "page");
    let supplemental = if pages.is_empty() { None } else { Some(pages) };
    (memories, supplemental)
}

/// POST /api/memory/search
pub async fn handle_search_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: HeaderMap,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(req): Json<SearchMemoryRequest>,
) -> Result<Json<SearchMemoryResponse>, ServerError> {
    let start = std::time::Instant::now();
    let (db, reranker) = {
        // Snapshot the Arcs we need before any await so we never hold the
        // read guard across the search call (LLM reranker or model load can
        // be slow; see AGENTS.md "Async and locking" guidance).
        let s = state.read().await;
        let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?.clone();
        let reranker = s.reranker.clone();
        (db, reranker)
    };
    let scope =
        crate::read_scope::effective_read_scope(&db, req.space.as_deref(), header_space.as_deref())
            .await?;

    let results = {
        if req.rerank {
            if reranker.is_none() {
                tracing::warn!(
                    "[search] rerank=true requested but no deep reranker wired (set WENLAN_RERANKER_MODE=full, or legacy WENLAN_RERANKER_ENABLED=1); falling back to plain hybrid search"
                );
            }
            db.search_memory_cross_rerank(
                &req.query,
                req.limit,
                req.memory_type.as_deref(),
                &scope,
                req.source_agent.as_deref(),
                reranker,
            )
            .await
            .map_err(|e| ServerError::SearchFailed(e.to_string()))?
        } else {
            db.search_memory(
                &req.query,
                req.limit,
                req.memory_type.as_deref(),
                &scope,
                req.source_agent.as_deref(),
                None,
                None,
                None,
            )
            .await
            .map_err(|e| ServerError::SearchFailed(e.to_string()))?
        }
    };

    // Filter to memory rows ONLY before access_log + agent_activity. Page-channel
    // (PR-B) returns `source="page"` rows whose `source_id` (`page_*`) is NOT a
    // valid memory id; persisting those into `access_log` or `agent_activity`
    // breaks downstream analytics that resolve ids against the `memories` table
    // (e.g. /api/retrievals/recent). See 2026-05-28 adversarial review.
    let memory_source_ids: Vec<String> = results
        .iter()
        .filter(|r| r.source == "memory")
        .map(|r| r.source_id.clone())
        .collect();
    {
        let s = state.read().await;
        s.access_tracker.record_accesses(&memory_source_ids);
        if let Some(db) = s.db.as_ref() {
            if let Err(e) = db.log_accesses(&memory_source_ids).await {
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
                .log_agent_activity(
                    &agent,
                    "search",
                    &memory_source_ids,
                    Some(&req.query),
                    &detail,
                )
                .await
            {
                tracing::warn!("Failed to log agent activity: {}", e);
            }
        }
    }

    let took_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (results, supplemental_pages) = partition_search_pages(results);

    // Additive page path: when the held RRF page-channel supplied nothing
    // (`supplemental_pages.is_none()` — the dedup guard that prevents
    // double-surfacing when the channel is on), surface gated distilled pages
    // through the SAME `supplemental_pages` wire field the MCP `recall` tool
    // already renders. Pages flow through the shared `select_visible_pages`
    // visibility gate (space-scope → effective-tier → confirmed/rank/cap).
    // Fail CLOSED: any lookup error leaves pages out (never surface ungated).
    let supplemental_pages = if supplemental_pages.is_some() || req.query == "recent context" {
        supplemental_pages
    } else {
        let db = {
            let s = state.read().await;
            s.db.clone()
        };
        match db {
            Some(db) => {
                // Resolve caller trust from the `x-agent-name` header
                // (mirror routes.rs handle_context: unknown→"unknown", else
                // db.get_agent(name) → trust_level, default "unknown").
                let agent_name = headers
                    .get("x-agent-name")
                    .and_then(|v| v.to_str().ok())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("unknown");
                let trust_level = if agent_name == "unknown" {
                    "unknown".to_string()
                } else {
                    db.get_agent(agent_name)
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
                if visible.is_empty() {
                    None
                } else {
                    Some(
                        visible
                            .into_iter()
                            .map(wenlan_core::db::MemoryDB::search_result_from_page)
                            .collect(),
                    )
                }
            }
            None => None,
        }
    };

    Ok(Json(SearchMemoryResponse {
        results,
        took_ms,
        supplemental_pages,
    }))
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

/// POST /api/memory/list
pub async fn handle_list_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(req): Json<ListMemoriesRequest>,
) -> Result<Json<ListMemoriesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    }; // guard dropped here
    let scope =
        crate::read_scope::effective_read_scope(&db, req.space.as_deref(), header_space.as_deref())
            .await?;
    let memories = db
        .list_filtered_confirmed_scoped(
            Some("memory"),
            req.memory_type.as_deref(),
            &scope,
            req.confirmed,
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
    headers: HeaderMap,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Json(mut req): Json<CreateEntityRequest>,
) -> Result<Json<CreateEntityResponse>, ServerError> {
    // Apply X-Origin-Space header as fallback only when body omits `space`.
    if req.space.is_none() {
        req.space = header_space;
    }
    let agent = extract_agent_name(&headers, None);
    let db = {
        let s = state.read().await;
        s.db.as_ref()
            .cloned()
            .ok_or(ServerError::DbNotInitialized)?
    };
    req.space = registered_request_space(&db, &req.space, "create_entity").await?;
    let result = wenlan_core::post_write::create_entity(&db, req, &agent).await?;
    Ok(Json(CreateEntityResponse {
        id: result.id,
        warnings: result.warnings,
    }))
}

pub async fn handle_create_relation(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CreateRelationRequest>,
) -> Result<Json<CreateRelationResponse>, ServerError> {
    let agent = extract_agent_name(&headers, req.source_agent.as_deref());
    let db = {
        let s = state.read().await;
        s.db.as_ref()
            .cloned()
            .ok_or(ServerError::DbNotInitialized)?
    };
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

fn agent_to_response(a: wenlan_core::db::AgentConnection) -> AgentResponse {
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

#[derive(Debug, Serialize)]
pub struct MemoryStatsResponse {
    pub stats: wenlan_core::db::MemoryStats,
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
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_types::HomeStats>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let stats = db
        .get_home_stats_scoped(&scope)
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

    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    match wenlan_core::post_write::update_memory(
        &db,
        &source_id,
        wenlan_core::post_write::MemoryUpdate {
            content: None,
            space: None,
            confirm: false,
            memory_type: Some(&mt),
        },
    )
    .await
    {
        Ok(()) | Err(wenlan_core::WenlanError::NotFound(_)) => {}
        Err(error) => return Err(ServerError::from(error)),
    }

    Ok(Json(ReclassifyMemoryResponse {
        source_id,
        memory_type: mt,
    }))
}

// ===== Pending Revision Endpoints =====

pub async fn handle_accept_revision(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: axum::http::HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::RevisionAcceptResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().ok_or(ServerError::DbNotInitialized)?.clone()
    };
    let agent = extract_agent_name(&headers, None);
    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    let result = wenlan_core::post_write::accept_pending_revision_with_knowledge_path(
        &db,
        &id,
        &agent,
        Some(knowledge_path.as_path()),
    )
    .await?;
    Ok(Json(result))
}

pub async fn handle_dismiss_revision(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: axum::http::HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::RevisionDismissResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().ok_or(ServerError::DbNotInitialized)?.clone()
    };
    let agent = extract_agent_name(&headers, None);
    let result = wenlan_core::post_write::dismiss_pending_revision(&db, &id, &agent).await?;
    Ok(Json(result))
}

/// POST /api/memory/contradiction/{source_id}/dismiss
///
/// Marks all awaiting-review contradiction flags for this memory as dismissed.
/// Returns 200 OK whether or not any rows were matched (idempotent).
pub async fn handle_dismiss_contradiction(
    State(state): State<Arc<RwLock<ServerState>>>,
    headers: axum::http::HeaderMap,
    Path(source_id): Path<String>,
) -> Result<Json<wenlan_types::ContradictionDismissResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().ok_or(ServerError::DbNotInitialized)?.clone()
    };
    let agent = extract_agent_name(&headers, None);
    let result = wenlan_core::post_write::dismiss_contradiction(&db, &source_id, &agent).await?;
    Ok(Json(result))
}

// ===== Enrichment Status =====

/// GET /api/memory/{source_id}/enrichment-status
///
/// Returns the enrichment step history and summary for a given memory.
pub async fn handle_get_enrichment_status(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(source_id): Path<String>,
) -> Result<Json<wenlan_types::EnrichmentStatusResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let status = db
        .get_enrichment_status_scoped(&source_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    Ok(Json(status))
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
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<Vec<EntitySuggestion>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let pending = db
        .list_entity_suggestions_scoped(&scope)
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
pub(crate) fn extract_agent_name(
    headers: &HeaderMap,
    deprecated_body_agent: Option<&str>,
) -> String {
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
    _db: Option<&wenlan_core::db::MemoryDB>,
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
) -> Result<Json<Vec<wenlan_core::db::Space>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let spaces = db
        .list_spaces()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(spaces))
}

pub async fn handle_create_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<CreateSpaceRequest>,
) -> Result<Json<wenlan_core::db::Space>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
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
) -> Result<Json<wenlan_core::db::Space>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
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
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_space(&name, "keep")
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({"deleted": name})))
}

pub async fn handle_move_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path((from, to)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let affected = db
        .reassign_memories_space(&from, &to)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(serde_json::json!({"affected": affected})))
}

// ===== Nurture Cards =====

#[derive(Debug, Deserialize)]
pub struct NurtureCardsQuery {
    #[serde(default = "default_nurture_limit")]
    pub limit: usize,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
}

fn default_nurture_limit() -> usize {
    3
}

#[derive(Debug, Serialize)]
pub struct NurtureCardsResponse {
    pub cards: Vec<wenlan_core::db::MemoryItem>,
}

pub async fn handle_get_nurture_cards(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(query): axum::extract::Query<NurtureCardsQuery>,
) -> Result<Json<NurtureCardsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(
        &db,
        query.space.as_deref(),
        header_space.as_deref(),
    )
    .await?;
    let cards = db
        .get_nurture_cards_scoped(query.limit, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(NurtureCardsResponse { cards }))
}

/// GET /api/memory/rejections
pub async fn handle_get_rejections(
    State(state): State<Arc<RwLock<ServerState>>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<Vec<wenlan_core::db::RejectionRecord>>, ServerError> {
    let limit: usize = params
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50)
        .min(500);
    let reason_owned = params.get("reason").cloned();

    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let records = db.get_rejections(limit, reason_owned.as_deref()).await?;
    Ok(Json(records))
}

/// GET /api/pages
pub async fn handle_list_pages(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
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
    let pages = db
        .list_pages_scoped(status, limit as i64, offset as i64, &scope)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({ "pages": pages })))
}

/// GET /api/pages/:id
pub async fn handle_get_page(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    match db.get_page_scoped(&id, &scope).await {
        Ok(Some(page)) => Ok(Json(serde_json::json!({ "page": page }))),
        Ok(None) => Err(ServerError::NotFound("page not found".to_string())),
        Err(e) => Err(ServerError::SearchFailed(e.to_string())),
    }
}

/// GET /api/pages/{id}/sources
///
/// Returns all source memories linked to a concept via the concept_sources join table,
/// enriched with memory metadata for display.
pub async fn handle_get_page_sources(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(id): Path<String>,
) -> Result<Json<Vec<wenlan_types::PageSourceWithMemory>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let sources = db.get_page_sources_scoped(&id, &scope).await?;

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
    let s = state.read().await;
    let db = s.db.as_ref().ok_or(ServerError::DbNotInitialized)?;
    db.archive_page(&id)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({"status": "archived"})))
}

/// DELETE /api/pages/{id}
///
/// Removes both the DB row and the projected `.origin/pages/<slug>.md`
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
    db.delete_page(&id)
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;

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
        .search_pages_scoped(
            &req.query,
            req.limit.unwrap_or(20),
            req.page_type.as_deref(),
            &scope,
        )
        .await
        .map_err(|e| ServerError::SearchFailed(e.to_string()))?;
    Ok(Json(serde_json::json!({ "pages": results })))
}

/// POST /api/pages
///
/// Atomic md-first + DB-index. The `.origin/pages/<slug>.md` file is the
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
    let (db, page_min_cluster_size, page_match_threshold) = {
        let s = state.read().await;
        (
            s.db.as_ref()
                .cloned()
                .ok_or(ServerError::DbNotInitialized)?,
            s.tuning.distillation.page_min_cluster_size,
            s.tuning.distillation.page_match_threshold,
        )
    };

    // Apply X-Origin-Space header as fallback only when body omits `space`.
    if req.space.is_none() {
        req.space = header_space.clone();
    }
    // HTTP/MCP `space` is the legacy scope input; `workspace` is the
    // authoritative P3 axis (spec line 518). Validate each independently
    // against registered spaces, dropping garbage/category values with a
    // warning -- reconciling the two into the honest `pages.space`/
    // `pages.workspace` columns (workspace wins when present) is
    // insert_page_with_kind's job now, not this handler's.
    req.space = registered_request_space(&db, &req.space, "create_page space").await?;
    req.workspace = registered_request_space(&db, &req.workspace, "create_page workspace").await?;
    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    let result = wenlan_core::post_write::create_page_with_tuning(
        &db,
        req,
        &agent,
        Some(knowledge_path.as_path()),
        page_min_cluster_size,
        page_match_threshold,
    )
    .await?;
    Ok(Json(CreatePageResponse {
        id: result.id,
        attached_to: result.attached_to,
        warnings: result.warnings,
    }))
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
    let pages = db
        .list_pages_scoped("active", 1000, 0, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
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
    use wenlan_core::export::PageExporter;
    let stats = exporter
        .export_all(&pages)
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
    // across the DB await. (CLAUDE.md: never hold tokio::sync::RwLock guards
    // across .await.)
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let page = db
        .get_page_scoped(&page_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("page not found".to_string()))?;

    let expanded = if let Some(rest) = req.vault_path.strip_prefix("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{}/{}", home, rest)
    } else {
        req.vault_path
    };

    let exporter =
        wenlan_core::export::obsidian::ObsidianExporter::new(std::path::PathBuf::from(expanded));
    use wenlan_core::export::PageExporter;
    let result = exporter
        .export(&page)
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(wenlan_types::responses::ExportPageResponse {
        path: result.path,
    }))
}

// =====================================================================
// Batch 2 — Indexed files / chunks
// =====================================================================

/// GET /api/indexed-files
pub async fn handle_list_indexed_files(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_types::responses::IndexedFilesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let files = db
        .list_indexed_files_scoped(&scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::IndexedFilesResponse {
        files,
    }))
}

/// GET /api/chunks/{source_id}
pub async fn handle_get_chunks(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(source_id): Path<String>,
) -> Result<Json<Vec<wenlan_core::db::MemoryDetail>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let chunks = db
        .get_chunks_scoped(&source_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    Ok(Json(chunks))
}

/// PUT /api/chunks/{id}/update
pub async fn handle_update_chunk(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::UpdateChunkRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.update_memory(&id, &req.content)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// DELETE /api/chunks/time-range
pub async fn handle_delete_by_time_range(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<wenlan_types::requests::DeleteByTimeRangeRequest>,
) -> Result<Json<wenlan_types::responses::DeleteCountResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let deleted = db
        .delete_by_time_range(req.start, req.end)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::DeleteCountResponse {
        deleted,
    }))
}

/// POST /api/chunks/delete-bulk
pub async fn handle_delete_bulk(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<wenlan_types::requests::BulkDeleteRequest>,
) -> Result<Json<wenlan_types::responses::DeleteCountResponse>, ServerError> {
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
    Ok(Json(wenlan_types::responses::DeleteCountResponse {
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
    Json(req): Json<wenlan_types::requests::AddEntityObservationRequest>,
) -> Result<Json<wenlan_types::responses::AddObservationResponse>, ServerError> {
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
    Ok(Json(wenlan_types::responses::AddObservationResponse {
        id,
        warnings: vec![],
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

// =====================================================================
// Batch 4 — Space CRUD
// =====================================================================

/// POST /api/spaces/{name}/pin — toggle space pinned (starred) state
pub async fn handle_pin_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    // pin_space maps to toggle_space_starred in the DB
    db.toggle_space_starred(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/spaces/{name}/confirm
pub async fn handle_confirm_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.confirm_space(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/spaces/reorder
pub async fn handle_reorder_space(
    State(state): State<Arc<RwLock<ServerState>>>,
    Json(req): Json<wenlan_types::requests::ReorderSpaceRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.reorder_space(&req.name, req.new_order)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
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
    Json(req): Json<wenlan_types::requests::SetDocumentSpaceRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let requested_space = Some(req.space_name);
    let registered_space =
        registered_request_space(&db, &requested_space, "set_document_space").await?;
    db.update_memory_space_opt(&source_id, registered_space.as_deref())
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

// =====================================================================
// Batch 5 — Activity, tags, capture stats, memory detail
// =====================================================================

/// GET /api/activities
pub async fn handle_list_activities(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<wenlan_types::responses::ActivityResponse>, ServerError> {
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
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let activities = db
        .list_agent_activity_scoped(limit, agent_name.as_deref(), since, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::ActivityResponse {
        activities,
    }))
}

/// GET /api/tags
pub async fn handle_list_tags(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_types::responses::TagsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let projection = db
        .list_tags_scoped(&scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::TagsResponse {
        tags: projection.tags,
        document_tags: projection.document_tags,
    }))
}

/// DELETE /api/tags/{name}
pub async fn handle_delete_tag(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(name): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_tag(&name)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// PUT /api/documents/{source_id}/tags
pub async fn handle_set_document_tags(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(source_id): Path<String>,
    Json(req): Json<wenlan_types::requests::SetDocumentTagsRequest>,
) -> Result<Json<wenlan_types::responses::TagsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let source = document_tag_source(&req).to_string();
    let tags = db
        .set_document_tags(&source, &source_id, req.tags)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::TagsResponse {
        tags,
        document_tags: HashMap::new(),
    }))
}

fn document_tag_source(req: &wenlan_types::requests::SetDocumentTagsRequest) -> &str {
    req.source
        .as_deref()
        .map(str::trim)
        .filter(|source| !source.is_empty())
        .unwrap_or("memory")
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
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(query): axum::extract::Query<SuggestTagsQuery>,
) -> Result<Json<wenlan_types::responses::TagsResponse>, ServerError> {
    // Read phase: clone db Arc, drop guard, then fetch existing tags from DB.
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let (existing, chunks) = db
        .get_tag_suggestion_inputs_scoped(&query.source, &query.source_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;

    let chunk_contents: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let title = chunks.first().map(|c| c.title.clone()).unwrap_or_default();

    let mut tags = wenlan_core::tags::suggest_tags_for_document(&chunk_contents, &title, &existing);

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

    Ok(Json(wenlan_types::responses::TagsResponse {
        tags,
        document_tags: HashMap::new(),
    }))
}

#[cfg(test)]
mod tag_route_tests {
    use super::*;

    #[test]
    fn document_tag_source_prefers_request_source() {
        let req = wenlan_types::requests::SetDocumentTagsRequest {
            source: Some("manual".to_string()),
            tags: vec!["rust".to_string()],
        };

        assert_eq!(document_tag_source(&req), "manual");
    }

    #[test]
    fn document_tag_source_defaults_to_memory_for_old_payloads() {
        let req = wenlan_types::requests::SetDocumentTagsRequest {
            source: None,
            tags: vec!["rust".to_string()],
        };

        assert_eq!(document_tag_source(&req), "memory");
    }
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
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::MemoryDetailResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let memory = db
        .get_memory_detail_scoped(&id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    Ok(Json(wenlan_types::responses::MemoryDetailResponse {
        memory: Some(memory),
    }))
}

/// GET /api/memory/by-ids?ids=mem_a,mem_b,...
///
/// Batch-fetch multiple memories by source_id in a single round trip.
/// The response preserves input order; missing ids are silently omitted.
/// Used by ConceptDetail to load all source memories at once.
pub async fn handle_get_memories_by_ids(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<wenlan_types::responses::PinnedMemoriesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
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
        .get_memories_by_source_ids_scoped(&ids, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::PinnedMemoriesResponse {
        memories,
    }))
}

/// GET /api/memory/{id}/versions
pub async fn handle_get_version_chain(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::VersionChainResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let versions = db
        .get_version_chain_scoped(&id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    Ok(Json(wenlan_types::responses::VersionChainResponse {
        versions,
    }))
}

/// PUT /api/memory/{id}/update
pub async fn handle_update_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::UpdateMemoryRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };

    let memory_type = req
        .memory_type
        .as_deref()
        .map(MemoryType::from_str)
        .transpose()
        .map_err(ServerError::BadRequest)?
        .map(|memory_type| memory_type.to_string());
    let registered_space = match &req.space {
        Some(space) => {
            Some(registered_request_space(&db, &Some(space.clone()), "update_memory").await?)
        }
        None => None,
    };

    wenlan_core::post_write::update_memory(
        &db,
        &id,
        wenlan_core::post_write::MemoryUpdate {
            content: req.content.as_deref(),
            space: registered_space.as_ref().map(|space| space.as_deref()),
            confirm: req.confirmed == Some(true),
            memory_type: memory_type.as_deref(),
        },
    )
    .await
    .map_err(ServerError::from)?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

#[cfg(test)]
mod update_memory_endpoint_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;
    use wenlan_types::sources::RawDocument;

    use crate::state::ServerState;

    #[tokio::test]
    async fn reclassify_keeps_response_shape_and_uses_canonical_update() {
        let tmp = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(tmp.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        db.upsert_documents(vec![RawDocument {
            source: "memory".to_string(),
            source_id: "reclassify-canonical".to_string(),
            title: "Reclassify".to_string(),
            content: "This memory changes type through the canonical boundary.".to_string(),
            memory_type: Some("fact".to_string()),
            ..Default::default()
        }])
        .await
        .unwrap();
        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..Default::default()
        }));

        let response = crate::router::build_router(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memory/reclassify/reclassify-canonical")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"memory_type":"decision"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            db.get_memory_detail("reclassify-canonical")
                .await
                .unwrap()
                .unwrap()
                .memory_type
                .as_deref(),
            Some("decision")
        );
    }

    #[tokio::test]
    async fn invalid_memory_type_rejects_the_whole_update_before_mutation() {
        let tmp = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(tmp.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        db.upsert_documents(vec![RawDocument {
            source: "memory".to_string(),
            source_id: "invalid-update".to_string(),
            title: "Original".to_string(),
            content: "Original content must remain unchanged.".to_string(),
            memory_type: Some("fact".to_string()),
            space: Some("work".to_string()),
            confirmed: Some(false),
            ..Default::default()
        }])
        .await
        .unwrap();

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..Default::default()
        }));
        let response = crate::router::build_router(state)
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/memory/invalid-update/update")
                    .header("Content-Type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "content": "Partially applied content must never persist.",
                            "confirmed": true,
                            "memory_type": "not-a-memory-type"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let memory = db
            .get_memory_detail("invalid-update")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(memory.content, "Original content must remain unchanged.");
        assert_eq!(memory.memory_type.as_deref(), Some("fact"));
        assert!(!memory.confirmed);
    }

    #[tokio::test]
    async fn missing_memory_update_returns_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(tmp.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db),
            ..Default::default()
        }));

        let response = crate::router::build_router(state)
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/memory/missing-update-head/update")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"content":"replacement content"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn confirmed_false_keeps_an_already_confirmed_memory_confirmed() {
        let tmp = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(tmp.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        db.upsert_documents(vec![RawDocument {
            source: "memory".to_string(),
            source_id: "confirm-false-noop".to_string(),
            title: "Confirmed".to_string(),
            content: "An already confirmed memory stays confirmed.".to_string(),
            memory_type: Some("fact".to_string()),
            confirmed: Some(true),
            stability: Some("confirmed".to_string()),
            ..Default::default()
        }])
        .await
        .unwrap();

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            ..Default::default()
        }));
        let response = crate::router::build_router(state)
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/memory/confirm-false-noop/update")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"confirmed":false}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            db.get_memory_detail("confirm-false-noop")
                .await
                .unwrap()
                .unwrap()
                .confirmed,
            "confirmed=false remains a no-op"
        );
    }
}

/// PUT /api/memory/{id}/stability
pub async fn handle_set_stability(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::SetStabilityRequest>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.set_stability(&id, &req.stability)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/memory/{id}/correct — apply LLM correction to a memory
pub async fn handle_correct_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    Json(req): Json<wenlan_types::requests::CorrectMemoryRequest>,
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
        .generate(wenlan_core::llm_provider::LlmRequest {
            system_prompt: Some("You are a memory correction assistant. Apply the correction to the memory and return only the corrected text.".to_string()),
            user_prompt: prompt,
            max_tokens: 2048,
            temperature: 0.1,
            label: None,
            timeout_secs: None,
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
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<wenlan_types::responses::DecisionsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let space = params
        .get("space")
        .or_else(|| params.get("domain"))
        .cloned();
    let scope =
        crate::read_scope::effective_read_scope(&db, space.as_deref(), header_space.as_deref())
            .await?;
    let limit: usize = params
        .get("limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);
    let decisions = db
        .list_memories_scoped(&scope, Some("decision"), None, None, limit)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::DecisionsResponse {
        decisions,
    }))
}

/// GET /api/decisions/domains
/// (Path kept as "domains" for back-compat; will rename to "spaces" in PR-A+1.)
pub async fn handle_list_decision_domains(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<wenlan_types::responses::DecisionDomainsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let domains = db
        .list_decision_spaces()
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::DecisionDomainsResponse {
        domains,
    }))
}

/// GET /api/briefing
pub async fn handle_get_briefing(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_core::briefing::BriefingResponse>, ServerError> {
    let (db, llm, prompts, tuning) = {
        let s = state.read().await;
        let db = s.db.clone().ok_or(ServerError::DbNotInitialized)?;
        let llm = s.llm.clone();
        let prompts = s.prompts.clone();
        let tuning = s.tuning.briefing.clone();
        (db, llm, prompts, tuning)
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let briefing = wenlan_core::briefing::generate_briefing_scoped(
        &db,
        llm.as_deref(),
        &prompts,
        &tuning,
        &scope,
    )
    .await?;
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
) -> Result<Json<wenlan_core::narrative::NarrativeResponse>, ServerError> {
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
            return Ok(Json(wenlan_core::narrative::NarrativeResponse {
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
        wenlan_core::narrative::generate_narrative(&db, llm.as_deref(), &prompts, &tuning)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(narrative))
}

/// POST /api/profile/narrative/regenerate
pub async fn handle_regenerate_narrative(
    State(state): State<Arc<RwLock<ServerState>>>,
) -> Result<Json<wenlan_core::narrative::NarrativeResponse>, ServerError> {
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
        wenlan_core::narrative::generate_narrative(&db, llm.as_deref(), &prompts, &tuning)
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(narrative))
}

/// GET /api/memory/pinned
pub async fn handle_list_pinned_memories(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<wenlan_types::responses::PinnedMemoriesResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let memories = db
        .list_pinned_memories_scoped(&scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::PinnedMemoriesResponse {
        memories,
    }))
}

/// POST /api/memory/{id}/pin
pub async fn handle_pin_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.pin_memory(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

/// POST /api/memory/{id}/unpin
pub async fn handle_unpin_memory(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.unpin_memory(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
}

#[derive(Debug, Deserialize)]
pub struct PendingRevisionsQuery {
    #[serde(default = "default_pending_revisions_limit")]
    pub limit: usize,
}

fn default_pending_revisions_limit() -> usize {
    50
}

/// GET /api/memory/pending-revisions?limit=N
pub async fn handle_list_pending_revisions(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(q): axum::extract::Query<PendingRevisionsQuery>,
) -> Result<Json<Vec<wenlan_types::responses::PendingRevisionItem>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let limit = q.limit.clamp(1, 500);
    let items = db
        .list_pending_revisions_scoped(limit, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(items))
}

/// GET /api/memory/pending-revision/{source_id}
pub async fn handle_get_pending_revision(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(source_id): Path<String>,
) -> Result<Json<Option<wenlan_core::db::PendingRevision>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let revision = db
        .get_pending_revision_for_scoped(&source_id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    Ok(Json(Some(revision)))
}

/// POST /api/snapshots/{id}/delete
pub async fn handle_delete_snapshot(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::SuccessResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    db.delete_snapshot(&id)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(wenlan_types::responses::SuccessResponse { ok: true }))
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
) -> Result<Json<Vec<wenlan_types::SessionSnapshot>>, ServerError> {
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
        .map(|r| wenlan_types::SessionSnapshot {
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
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<Vec<wenlan_types::SnapshotCapture>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let rows = db.get_captures_for_snapshot_scoped(&id, &scope).await?;
    let captures = rows
        .into_iter()
        .map(|c| wenlan_types::SnapshotCapture {
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
/// summary.
pub async fn handle_get_snapshot_captures_with_content(
    State(state): State<Arc<RwLock<ServerState>>>,
    Path(id): Path<String>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
) -> Result<Json<Vec<wenlan_types::SnapshotCaptureWithContent>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let captures = db
        .get_snapshot_captures_with_content_scoped(&id, &scope)
        .await?;
    Ok(Json(captures))
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
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::PageLinksResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let outbound_raw = db.get_page_outbound_links_scoped(&id, &scope).await?;
    let inbound_raw = db.get_page_inbound_links_scoped(&id, &scope).await?;
    let outbound = outbound_raw
        .into_iter()
        .map(|l| wenlan_types::responses::PageLinkOutbound {
            label: l.label,
            target_page_id: l.target_page_id,
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
    axum::extract::Query(q): axum::extract::Query<OrphanLinksQuery>,
) -> Result<Json<wenlan_types::responses::OrphanLinksResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let min_count = q.min_count.unwrap_or(2).max(1);
    let labels = db
        .list_orphan_link_labels_scoped(min_count, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    let orphan_labels = labels
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
    };

    // 1. md-first
    projection
        .write_page(&refreshed_page)
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
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(q): axum::extract::Query<RecentActivityQuery>,
) -> Result<Json<Vec<wenlan_types::RecentActivityItem>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let items = db
        .list_recent_memories_scoped(q.limit.unwrap_or(10), q.since_ms, &scope)
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
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    axum::extract::Query(q): axum::extract::Query<RecentActivityQuery>,
) -> Result<Json<Vec<wenlan_types::RecentActivityItem>>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.as_ref().cloned()
    };
    let db = db.ok_or(ServerError::DbNotInitialized)?;
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let items = db
        .list_unconfirmed_memories_scoped(q.limit.unwrap_or(6), &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;
    Ok(Json(items))
}

#[cfg(test)]
mod store_scheduler_handoff_tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::sync::Notify;

    struct DataDirGuard {
        previous: Option<std::ffi::OsString>,
        _tmp: tempfile::TempDir,
    }

    impl DataDirGuard {
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

    impl Drop for DataDirGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
        }
    }

    struct CountingProvider {
        calls: AtomicUsize,
        called: Notify,
    }

    #[async_trait]
    impl wenlan_core::llm_provider::LlmProvider for CountingProvider {
        async fn generate(
            &self,
            _request: wenlan_core::llm_provider::LlmRequest,
        ) -> Result<String, wenlan_core::llm_provider::LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.called.notify_one();
            Ok(
                r#"{"memory_type":"fact","domain":null,"quality":"high","importance":5,"tags":[]}"#
                    .to_string(),
            )
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "store-handoff-test"
        }

        fn backend(&self) -> wenlan_core::llm_provider::LlmBackend {
            wenlan_core::llm_provider::LlmBackend::Api
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    #[tokio::test]
    async fn store_releases_server_state_before_all_db_awaits() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(dir.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        let gate = wenlan_core::quality_gate::QualityGate::new(wenlan_core::tuning::GateConfig {
            enabled: false,
            ..Default::default()
        });
        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db),
            quality_gate: gate,
            ..Default::default()
        }));
        wenlan_core::config::save_config(&wenlan_core::config::Config::default()).unwrap();

        let mut blocked_stages = Vec::new();
        for (index, stage) in [
            StoreLockTestStage::Dedup,
            StoreLockTestStage::AgentGate,
            StoreLockTestStage::EntityResolution,
            StoreLockTestStage::ActivityAgentLookup,
            StoreLockTestStage::ActivityLog,
        ]
        .into_iter()
        .enumerate()
        {
            let registration = StoreLockTestHookRegistration::install(stage);
            let hook = registration.hook().clone();
            let request = StoreMemoryRequest {
                content: format!(
                    "Store lock stage {stage:?} must release ServerState before DB await {index}."
                ),
                memory_type: None,
                space: None,
                source_agent: Some(format!("lock-lifetime-test-agent-{index}")),
                title: None,
                confidence: None,
                supersedes: None,
                entity: (stage == StoreLockTestStage::EntityResolution)
                    .then(|| "missing-lock-test-entity".to_string()),
                entity_id: None,
                structured_fields: None,
                retrieval_cue: None,
            };
            let task_state = state.clone();
            let store_task = tokio::spawn(async move {
                handle_store_memory(
                    State(task_state),
                    HeaderMap::new(),
                    crate::space_header::SpaceHeader(None),
                    Json(request),
                )
                .await
            });
            tokio::time::timeout(std::time::Duration::from_secs(2), hook.reached.notified())
                .await
                .unwrap_or_else(|_| panic!("store route never reached lock stage {stage:?}"));

            let writer =
                tokio::time::timeout(std::time::Duration::from_millis(500), state.write()).await;
            let writer_acquired = writer.is_ok();
            drop(writer);
            hook.release.notify_one();
            let _response = store_task.await.unwrap().unwrap();
            drop(registration);
            if !writer_acquired {
                blocked_stages.push(stage);
            }
        }

        assert!(
            blocked_stages.is_empty(),
            "ServerState read guard remained held across DB awaits at {blocked_stages:?}"
        );
    }

    #[tokio::test]
    async fn store_with_unconfigured_provider_reports_paused_without_calling_provider() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(dir.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        let provider = Arc::new(CountingProvider {
            calls: AtomicUsize::new(0),
            called: Notify::new(),
        });
        let gate = wenlan_core::quality_gate::QualityGate::new(wenlan_core::tuning::GateConfig {
            enabled: false,
            ..Default::default()
        });
        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            llm: Some(provider.clone()),
            quality_gate: gate,
            ..Default::default()
        }));
        let req = StoreMemoryRequest {
            content: "Store must hand enrichment to the ambient scheduler only.".to_string(),
            memory_type: None,
            space: None,
            source_agent: Some("test-agent".to_string()),
            title: None,
            confidence: None,
            supersedes: None,
            entity: None,
            entity_id: None,
            structured_fields: None,
            retrieval_cue: None,
        };

        wenlan_core::config::save_config(&wenlan_core::config::Config::default()).unwrap();
        let response = handle_store_memory(
            State(state),
            HeaderMap::new(),
            crate::space_header::SpaceHeader(None),
            Json(req),
        )
        .await
        .unwrap();

        assert_eq!(response.0.enrichment, "paused");
        assert!(response.0.hint.contains("Recall is available now"));
        assert!(response.0.hint.contains("choose a model source"));
        assert!(
            !response.0.hint.contains("~2s"),
            "ambient work must not promise a foreground-style completion ETA"
        );
        assert!(db.get_classification_candidate(3).await.unwrap().is_some());
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(100),
                provider.called.notified()
            )
            .await
            .is_err(),
            "the HTTP store path must never forward an enrichment inference"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn store_with_healthy_external_pin_reports_pending_without_inline_inference() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(dir.path(), Arc::new(wenlan_core::events::NoopEmitter))
                .await
                .unwrap(),
        );
        let provider = Arc::new(CountingProvider {
            calls: AtomicUsize::new(0),
            called: Notify::new(),
        });
        let gate = wenlan_core::quality_gate::QualityGate::new(wenlan_core::tuning::GateConfig {
            enabled: false,
            ..Default::default()
        });
        let state = Arc::new(RwLock::new(ServerState {
            db: Some(db.clone()),
            external_llm: Some(provider.clone()),
            quality_gate: gate,
            ..Default::default()
        }));
        let req = StoreMemoryRequest {
            content: "A healthy explicit pin should authorize only deferred enrichment."
                .to_string(),
            memory_type: None,
            space: None,
            source_agent: Some("test-agent".to_string()),
            title: None,
            confidence: None,
            supersedes: None,
            entity: None,
            entity_id: None,
            structured_fields: None,
            retrieval_cue: None,
        };

        wenlan_core::config::save_config(&wenlan_core::config::Config {
            everyday_source: Some("external".to_string()),
            ..wenlan_core::config::Config::default()
        })
        .unwrap();
        let response = handle_store_memory(
            State(state),
            HeaderMap::new(),
            crate::space_header::SpaceHeader(None),
            Json(req),
        )
        .await
        .unwrap();

        assert_eq!(response.0.enrichment, "pending");
        assert!(response.0.hint.contains("quietly enrich"));
        assert!(db.get_classification_candidate(3).await.unwrap().is_some());
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(100),
                provider.called.notified()
            )
            .await
            .is_err(),
            "the request path must not spend the authorized provider call"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;

    #[test]
    fn fixed_origin_folds_profile_alias_but_preserves_other_explicit_inputs() {
        assert_eq!(
            fixed_enrichment_origin(true, true, true, true),
            wenlan_core::db::EnrichmentOrigin {
                memory_type_explicit: false,
                structured_fields_explicit: true,
                space_rejected: true,
            }
        );
        assert_eq!(
            fixed_enrichment_origin(true, false, false, false),
            wenlan_core::db::EnrichmentOrigin {
                memory_type_explicit: true,
                structured_fields_explicit: false,
                space_rejected: false,
            }
        );
    }

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
                space: space.map(str::to_string),
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

    use crate::{router::AppRouter, state::ServerState};

    async fn build_state_with_db() -> (Arc<RwLock<ServerState>>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("failed to create tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new should succeed");
        let server_state = ServerState {
            db: Some(Arc::new(db)),
            ..Default::default()
        };
        (Arc::new(RwLock::new(server_state)), tmp)
    }

    async fn fetch_activities(app: AppRouter) -> Vec<wenlan_types::AgentActivityRow> {
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
        let wrapper: wenlan_types::responses::ActivityResponse =
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

/// Wiring tests for the `rerank` flag on `/api/memory/search`.
///
/// These verify that the handler reads `ServerState.reranker` and routes
/// `rerank=true` through `search_memory_cross_rerank`. The `NoopReranker`
/// stand-in keeps these fast and dependency-free — no model weights, no
/// cross-encoder download. The rerank quality itself is covered by
/// `wenlan_core::db::tests::test_search_memory_cross_rerank_*`.
#[cfg(test)]
mod search_rerank_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::{router::AppRouter, state::ServerState};
    use wenlan_core::reranker::{NoopReranker, Reranker};

    async fn build_state(with_reranker: bool) -> (Arc<RwLock<ServerState>>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("failed to create tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new should succeed");
        let reranker: Option<Arc<dyn Reranker>> = if with_reranker {
            Some(Arc::new(NoopReranker))
        } else {
            None
        };
        let server_state = ServerState {
            db: Some(Arc::new(db)),
            reranker,
            ..Default::default()
        };
        (Arc::new(RwLock::new(server_state)), tmp)
    }

    async fn search_response(
        app: AppRouter,
        body: &'static str,
    ) -> wenlan_types::responses::SearchMemoryResponse {
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memory/search")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "search should succeed");
        let bytes = axum::body::to_bytes(resp.into_body(), 1_048_576)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).expect("parse SearchMemoryResponse")
    }

    #[tokio::test]
    async fn rerank_true_with_noop_reranker_returns_same_shape_as_plain_search() {
        // With reranker wired (NoopReranker) and an empty DB, both `rerank=true`
        // and `rerank=false` return the same response shape (empty results,
        // `took_ms` populated). This locks in: the handler doesn't fail when
        // the reranker is consulted, and the response envelope is unchanged.
        let (state, _tmp) = build_state(true).await;
        let app = crate::router::build_router(state);

        let plain = search_response(app.clone(), r#"{"query":"hello"}"#).await;
        let reranked = search_response(app, r#"{"query":"hello","rerank":true}"#).await;

        assert_eq!(plain.results.len(), reranked.results.len());
        assert!(plain.took_ms >= 0.0);
        assert!(reranked.took_ms >= 0.0);
    }

    #[tokio::test]
    async fn rerank_true_without_reranker_falls_back_and_returns_ok() {
        // When `rerank=true` is requested but no reranker is wired, the
        // handler logs a warning and falls back to plain hybrid search
        // rather than failing the request.
        let (state, _tmp) = build_state(false).await;
        let app = crate::router::build_router(state);

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memory/search")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"hello","rerank":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "rerank=true without a wired reranker should fall back, not fail"
        );
    }
}

#[cfg(test)]
mod partition_pages_tests {
    use wenlan_types::memory::SearchResult;

    fn make_result(source: &str, id: &str) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: "body".to_string(),
            source: source.to_string(),
            source_id: id.to_string(),
            title: String::new(),
            url: None,
            chunk_index: 0,
            last_modified: 0,
            score: 1.0,
            chunk_type: None,
            language: None,
            semantic_unit: None,
            memory_type: None,
            space: None,
            source_agent: None,
            confidence: None,
            confirmed: None,
            stability: None,
            supersedes: None,
            summary: None,
            entity_id: None,
            entity_name: None,
            quality: None,
            importance: None,
            event_date: None,
            is_archived: false,
            is_recap: false,
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            content_hash: None,
            raw_score: 0.0,
            version: 0,
            pending_revision: false,
            merged_from: None,
            last_delta_summary: None,
        }
    }

    /// No page rows — supplemental_pages must be None.
    #[test]
    fn no_pages_gives_none() {
        let rows = vec![
            make_result("memory", "mem_1"),
            make_result("memory", "mem_2"),
        ];
        let (mems, pages) = super::partition_search_pages(rows);
        assert_eq!(mems.len(), 2);
        assert!(pages.is_none());
    }

    /// Mixed rows: page rows isolated, memory rows preserved, `results.len() <= limit`.
    #[test]
    fn page_rows_isolated_from_memory_rows() {
        let rows = vec![
            make_result("memory", "mem_1"),
            make_result("page", "page_1"),
            make_result("memory", "mem_2"),
            make_result("page", "page_2"),
        ];
        let (mems, pages) = super::partition_search_pages(rows);
        assert_eq!(mems.len(), 2, "memory rows");
        assert!(mems.iter().all(|r| r.source == "memory"));
        let pages = pages.expect("should have page rows");
        assert_eq!(pages.len(), 2, "page rows");
        assert!(pages.iter().all(|r| r.source == "page"));
    }

    /// All rows are pages — memories empty, supplemental Some.
    #[test]
    fn all_pages_no_memories() {
        let rows = vec![make_result("page", "page_a"), make_result("page", "page_b")];
        let (mems, pages) = super::partition_search_pages(rows);
        assert!(mems.is_empty());
        assert_eq!(pages.unwrap().len(), 2);
    }

    /// Empty input — both empty / None.
    #[test]
    fn empty_input() {
        let (mems, pages) = super::partition_search_pages(vec![]);
        assert!(mems.is_empty());
        assert!(pages.is_none());
    }
}

/// Additive page path on `/api/memory/search` quick path (Task 6).
///
/// When `WENLAN_ENABLE_PAGE_CHANNEL` is unset and `rerank=false`, the RRF
/// page-channel supplies nothing, so `partition_search_pages` leaves
/// `supplemental_pages=None`. The handler then fetches + gates pages through
/// the shared `select_visible_pages` visibility gate (space-scope + effective
/// tier + confirmed/rank/cap) and surfaces them via the existing
/// `supplemental_pages` wire field. Fail-CLOSED: any error leaves pages out.
#[cfg(test)]
mod search_quick_path_page_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;
    use wenlan_types::requests::CreateConceptRequest;

    use crate::{router::AppRouter, state::ServerState};

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
            source_agent: Some("trusted-agent".to_string()),
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
                space: Some(space.to_string()),
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

    /// Seeds a DB with a tier-2 agent ("trusted-agent", trust "review"), a
    /// tier-2 `decision` source memory in space "work", a confirmed same-space
    /// page distilled from that tier-2 memory ("page_work"), and a confirmed
    /// cross-space page whose only source is absent from any result set
    /// ("page_cross"). The tier-2 source makes "page_work" visible to "review"
    /// but invisible to "unknown".
    async fn build_state_with_pages(
    ) -> (Arc<RwLock<ServerState>>, tempfile::TempDir, String, String) {
        let tmp = tempfile::tempdir().expect("tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new");
        db.create_space("work", None, false).await.unwrap();

        // Tier-2 caller: "review" allows tier 2, denies tier 1; "unknown" denies tier 2.
        db.register_agent("trusted-agent").await.unwrap();
        db.update_agent("trusted-agent", None, None, None, Some("review"), None)
            .await
            .unwrap();

        // Source memory: a `decision` (tier 2). Its presence makes the page's
        // effective read tier = 2 (decision/correction → tier 2), so the page is
        // visible to "review" trust but not to "unknown".
        let mem = wenlan_core::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "m_decision".to_string(),
            title: "memory-m_decision".to_string(),
            content: "zorblax source memory".to_string(),
            memory_type: Some("decision".to_string()),
            space: Some("work".to_string()),
            source_agent: Some("trusted-agent".to_string()),
            confidence: Some(0.9),
            confirmed: Some(true),
            ..Default::default()
        };
        db.upsert_documents(vec![mem]).await.unwrap();

        // Same-space page distilled from the tier-2 decision → visible to review.
        let page_work_id = seed_confirmed_distilled_page(
            &db,
            "Zorblax Workmarker",
            "zorblax workmarker body",
            "m_decision",
            "decision",
            "work",
        )
        .await;
        // Cross-space page whose only source is NOT in the memory result set → dropped.
        let page_cross_id = seed_confirmed_distilled_page(
            &db,
            "Crossmarker",
            "crossmarker body outside query",
            "unrelated",
            "fact",
            "other",
        )
        .await;

        let state = Arc::new(RwLock::new(ServerState {
            db: Some(Arc::new(db)),
            ..Default::default()
        }));
        (state, tmp, page_work_id, page_cross_id)
    }

    async fn search(
        app: AppRouter,
        agent: Option<&str>,
    ) -> wenlan_types::responses::SearchMemoryResponse {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/api/memory/search")
            .header("content-type", "application/json");
        if let Some(a) = agent {
            builder = builder.header("x-agent-name", a);
        }
        let resp = app
            .oneshot(
                builder
                    .body(Body::from(
                        r#"{"query":"zorblax","space":"work","rerank":false}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "search should succeed");
        let bytes = axum::body::to_bytes(resp.into_body(), 1_048_576)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).expect("parse SearchMemoryResponse")
    }

    #[tokio::test]
    async fn quick_path_surfaces_gated_page_for_trusted_same_space_caller() {
        let (state, _tmp, page_work_id, page_cross_id) = build_state_with_pages().await;
        let app = crate::router::build_router(state);

        let resp = search(app, Some("trusted-agent")).await;
        let pages = resp
            .supplemental_pages
            .expect("trusted same-space caller should get supplemental pages");
        assert!(
            pages.iter().any(|p| p.source_id == page_work_id),
            "same-space tier-2 page must surface, got: {:?}",
            pages
                .iter()
                .map(|p| p.source_id.clone())
                .collect::<Vec<_>>()
        );
        assert!(
            !pages.iter().any(|p| p.source_id == page_cross_id),
            "cross-space page must be dropped, got: {:?}",
            pages
                .iter()
                .map(|p| p.source_id.clone())
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn quick_path_hides_pages_from_unknown_caller() {
        let (state, _tmp, _page_work_id, _page_cross_id) = build_state_with_pages().await;
        let app = crate::router::build_router(state);

        // No x-agent-name header → trust resolves to "unknown" → the tier-2 page is
        // denied and the cross-space page is dropped → no visible pages at all.
        let resp = search(app, None).await;
        assert!(
            resp.supplemental_pages.is_none(),
            "unknown caller must not receive any pages, got: {:?}",
            resp.supplemental_pages
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
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
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
            db.resolve_refinement_if_open("ref_contradiction_1", "awaiting_review")
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

#[cfg(test)]
mod dismiss_revision_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::ServiceExt;
    use wenlan_types::sources::RawDocument;

    use crate::state::ServerState;

    async fn build_state_with_db() -> (Arc<RwLock<ServerState>>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().expect("failed to create tempdir");
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let db = wenlan_core::db::MemoryDB::new(tmp.path(), emitter)
            .await
            .expect("MemoryDB::new should succeed");
        let server_state = ServerState {
            db: Some(Arc::new(db)),
            ..Default::default()
        };
        (Arc::new(RwLock::new(server_state)), tmp)
    }

    /// HTTP contract for the external wenlan-app caller. `POST
    /// /api/memory/revision/{id}/dismiss` UNSTAGES the revision — it clears the
    /// false `pending_revision` + `supersedes` link and keeps BOTH memories as
    /// independent rows. It must NOT delete. Regression guard against the old
    /// DELETE-on-Dismiss behavior that destroyed a distinct captured memory
    /// whenever the staging was a false positive.
    #[tokio::test]
    async fn dismiss_revision_endpoint_unstages_and_keeps_both() {
        let (state, _tmp) = build_state_with_db().await;
        {
            let s = state.read().await;
            let db = s.db.as_ref().unwrap();
            let target = RawDocument {
                source: "memory".to_string(),
                source_id: "rev_dismiss_target".to_string(),
                title: "Original".to_string(),
                content: "I prefer tabs over spaces".to_string(),
                memory_type: Some("preference".to_string()),
                confirmed: Some(true),
                ..Default::default()
            };
            let revision = RawDocument {
                source: "memory".to_string(),
                source_id: "rev_dismiss_revision".to_string(),
                title: "Revision".to_string(),
                content: "A distinct fact that was falsely staged as a revision".to_string(),
                memory_type: Some("preference".to_string()),
                confirmed: Some(false),
                supersedes: Some("rev_dismiss_target".to_string()),
                pending_revision: true,
                ..Default::default()
            };
            db.upsert_documents(vec![target, revision]).await.unwrap();
        }

        let app = crate::router::build_router(state.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memory/revision/rev_dismiss_revision/dismiss")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "dismiss revision endpoint should return 200"
        );

        // Both memories must SURVIVE — dismiss unstages, it does not delete.
        // get_memory_detail filters pending_revision = 0, so a still-staged or
        // deleted row would return None here.
        let s = state.read().await;
        let db = s.db.as_ref().unwrap();
        let revision = db
            .get_memory_detail("rev_dismiss_revision")
            .await
            .unwrap()
            .expect(
                "dismissed revision must survive as an independent memory (unstage, not delete)",
            );
        assert!(
            revision.supersedes.is_none(),
            "dismiss must clear the false supersedes link"
        );
        assert!(
            db.get_memory_detail("rev_dismiss_target")
                .await
                .unwrap()
                .is_some(),
            "the original target must remain after dismiss"
        );
    }
}

// ===== Revision history endpoints =====

/// GET /api/memory/{id}/revisions
///
/// Walk the supersede chain for a memory and return all revision entries.
pub async fn handle_get_memory_revisions(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::ListMemoryRevisionsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let entries = db
        .walk_supersede_chain_scoped(&id, &scope)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .ok_or_else(|| ServerError::NotFound("memory not found".to_string()))?;
    let chain_depth = entries.last().map(|e| e.depth).unwrap_or(0);
    Ok(Json(wenlan_types::responses::ListMemoryRevisionsResponse {
        current_source_id: id,
        chain_depth,
        entries,
    }))
}

/// GET /api/pages/{id}/revisions
///
/// Return the changelog entries for a page, plus envelope metadata.
pub async fn handle_get_page_revisions(
    State(state): State<Arc<RwLock<ServerState>>>,
    crate::space_header::SpaceHeader(header_space): crate::space_header::SpaceHeader,
    Path(id): Path<String>,
) -> Result<Json<wenlan_types::responses::ListPageRevisionsResponse>, ServerError> {
    let db = {
        let s = state.read().await;
        s.db.clone().ok_or(ServerError::DbNotInitialized)?
    };
    let scope = crate::read_scope::effective_read_scope(&db, None, header_space.as_deref()).await?;
    let page = db
        .get_page_scoped(&id, &scope)
        .await?
        .ok_or_else(|| ServerError::NotFound("page not found".to_string()))?;
    let changelog_str = db.get_page_changelog_scoped(&id, &scope).await?;
    let entries: Vec<wenlan_types::responses::PageChangelogEntry> =
        serde_json::from_str(&changelog_str)
            .map_err(|e| ServerError::Internal(format!("parse changelog: {e}")))?;
    Ok(Json(wenlan_types::responses::ListPageRevisionsResponse {
        page_id: id,
        current_version: page.version,
        user_edited: page.user_edited,
        stale_reason: page.stale_reason.clone(),
        entries,
    }))
}
