//! Canonical post-store ingest enrichment.
//!
//! This module holds the ONE shared enrichment path that runs after a memory's
//! primary chunk is upserted: LLM classify + extract + `apply_enrichment` + tags
//! (Phase 1), entity/title/page enrichment (Phase 2), and dual-pool
//! dedup/contradiction resolution (Phase 3).
//!
//! It was extracted verbatim from the server `handle_store_memory`
//! deferred-reflection task so that the daemon ingest path, the eval seed
//! pipeline, and the importer all enrich through the SAME code. Sharing this
//! eliminates a training-serving skew: a write-time feature added here reaches
//! every consumer (including eval) automatically, instead of silently lagging in
//! a divergent eval shortcut. (Google "Rules of ML", Rule #32: "Re-use code
//! between your training pipeline and your serving pipeline whenever possible.")

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::events::{EventEmitter, NoopEmitter};
use crate::llm_provider::LlmProvider;
use crate::prompts::PromptRegistry;
use crate::tuning::{DistillationConfig, RefineryConfig};

/// Caller-supplied overrides plus the initial sync-path classification that the
/// canonical enrichment refines. These mirror the locals the server
/// `handle_store_memory` sync path computes before dispatching the deferred
/// reflection task.
#[derive(Debug, Clone, Default)]
pub struct EnrichmentOpts {
    /// Placeholder memory_type stored on the sync path (e.g. `"fact"`, or the
    /// caller-supplied subtype). The classifier replaces it unless the caller
    /// supplied a concrete (non-alias) type.
    pub initial_memory_type: String,
    /// Domain/space the sync path already resolved, if any.
    pub initial_domain: Option<String>,
    /// True when the caller explicitly supplied a space/domain but the sync path
    /// rejected it as unregistered and stored the row unscoped.
    pub rejected_explicit_domain: bool,
    /// Supersede mode the sync path computed for `initial_memory_type`.
    pub initial_supersede_mode: String,
    /// Structured fields the caller supplied (skips the extract LLM call).
    pub initial_structured_fields: Option<String>,
    /// True when the caller passed a non-empty `memory_type`.
    pub agent_supplied_memory_type: bool,
    /// True when the caller-supplied `memory_type` was a profile alias (so the
    /// classifier's concrete subtype is taken even though a type was supplied).
    pub agent_supplied_profile_alias: bool,
    /// True when the caller supplied `structured_fields` (skips extract).
    pub agent_supplied_structured_fields: bool,
}

/// The classification the enrichment resolved. The server discards this (the
/// HTTP 200 already returned before the task runs); the eval seed pipeline and
/// future callers consume it for logging / assertions. Every field reflects the
/// value actually written via `apply_enrichment`.
#[derive(Debug, Clone, Default)]
pub struct EnrichmentOutcome {
    pub final_memory_type: String,
    pub final_domain: Option<String>,
    pub final_quality: Option<String>,
    pub final_importance: Option<u8>,
    pub final_structured_fields: Option<String>,
    pub final_retrieval_cue: Option<String>,
    pub final_event_date: Option<i64>,
    pub final_event_end: Option<i64>,
}

/// Phase 1 of the canonical enrichment, in isolation: LLM classify + extract +
/// `apply_enrichment` + tags. Returns the resolved
/// classification (also written to the row via `apply_enrichment`).
///
/// This is the reusable unit. `run_canonical_enrichment` calls it as its Phase 1
/// (when an LLM is present); the eval seed pipeline calls it directly to backfill
/// importance/event_date/quality/structured_fields/retrieval_cue onto
/// already-upserted memories WITHOUT re-running the bulk entity/title/page passes
/// that the eval pipeline owns via its scale-optimized batched variants. Sharing
/// this exact pass is what keeps the eval seed and production in lockstep (Google
/// "Rules of ML", Rule #32) instead of the eval shortcut silently lacking the
/// classify/extract signals.
///
/// `llm` is non-optional here by construction — Phase 1 only ran inside the
/// `if let Some(llm)` guard in the original server task. Callers with no LLM skip
/// this entirely. Log-and-degrade at every step; never propagates an error.
pub async fn run_classification_enrichment(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    opts: &EnrichmentOpts,
) -> EnrichmentOutcome {
    let mut final_memory_type = opts.initial_memory_type.clone();
    let mut final_domain = opts.initial_domain.clone();
    let mut final_supersede_mode = opts.initial_supersede_mode.clone();
    let mut final_quality: Option<String> = None;
    let mut final_importance: Option<u8> = None;
    let mut final_structured_fields: Option<String> = opts.initial_structured_fields.clone();
    let mut final_retrieval_cue: Option<String> = None;
    let mut final_event_date: Option<i64> = None;
    let mut final_event_end: Option<i64> = None;
    let mut classified_tags: Vec<String> = Vec::new();

    // Classify. Agent-supplied memory_type stays authoritative unless they passed
    // a profile alias (in which case we take the classifier's concrete subtype).
    // Other fields — domain/quality/tags — come from classify regardless.
    let truncated: String = content.chars().take(1000).collect();
    // 30s matches `OnDeviceProvider::generate`'s own timeout. 5s was too
    // aggressive under a burst: when 10 concurrent stores each fire classify +
    // extract, the 10th call's turn at the single LLM worker lands at ~30s.
    match tokio::time::timeout(
        std::time::Duration::from_secs(30),
        llm.generate(crate::llm_provider::LlmRequest {
            system_prompt: Some(prompts.classify_memory_quality.clone()),
            user_prompt: truncated,
            max_tokens: 128,
            temperature: 0.1,
            label: Some("classify".into()),
            timeout_secs: None,
        }),
    )
    .await
    {
        Ok(Ok(output)) => {
            if let Some(c) = crate::llm_provider::parse_classify_response(&output) {
                if !opts.agent_supplied_memory_type || opts.agent_supplied_profile_alias {
                    final_memory_type = c.memory_type.clone();
                    // Recompute supersede_mode for the refined type.
                    final_supersede_mode = if final_memory_type == "decision" {
                        "archive".to_string()
                    } else {
                        "hide".to_string()
                    };
                }
                if final_domain.is_none() && !opts.rejected_explicit_domain {
                    let proposed_space =
                        c.space.as_deref().map(str::trim).filter(|s| !s.is_empty());
                    match db.registered_space_or_none(c.space.as_deref()).await {
                        Ok(Some(space)) => final_domain = Some(space),
                        Ok(None) => {
                            if let Some(space) = proposed_space {
                                log::warn!(
                                "[ingest] ignoring unregistered classifier space {:?}; memory remains unscoped",
                                space
                            );
                            }
                        }
                        Err(e) => {
                            log::warn!("[ingest] classifier space lookup failed: {e}")
                        }
                    }
                } else if opts.rejected_explicit_domain {
                    let proposed_space =
                        c.space.as_deref().map(str::trim).filter(|s| !s.is_empty());
                    if let Some(space) = proposed_space {
                        log::warn!(
                            "[ingest] ignoring classifier space {:?}; request space was explicitly rejected",
                            space
                        );
                    }
                }
                final_quality = c.quality;
                final_importance = c.importance;
                classified_tags = c.tags;
            }
        }
        Ok(Err(e)) => log::warn!("[ingest] classify failed: {e}"),
        Err(_) => log::warn!("[ingest] classify timed out after 30s"),
    }

    // Extract structured fields — only if the agent didn't supply them.
    if !opts.agent_supplied_structured_fields {
        let prompt = crate::memory_schema::extraction_prompt_with_template(
            &final_memory_type,
            &prompts.extract_structured_fields,
        );
        let truncated: String = content.chars().take(1500).collect();
        // Same reasoning as the classify timeout above — 30s is the provider's
        // own cap; defense-in-depth so a stalled extract doesn't hang this task.
        match tokio::time::timeout(
            std::time::Duration::from_secs(30),
            llm.generate(crate::llm_provider::LlmRequest {
                system_prompt: Some(prompt),
                user_prompt: truncated,
                max_tokens: 256,
                temperature: 0.1,
                label: None,
                timeout_secs: None,
            }),
        )
        .await
        {
            Ok(Ok(output)) => {
                let extracted = crate::llm_provider::parse_extraction_response(&output);
                final_event_date = extracted.event_date;
                final_event_end = extracted.event_end;
                if let Some(f) = extracted.structured_fields {
                    final_structured_fields = Some(f);
                }
                if let Some(c) = extracted.retrieval_cue {
                    final_retrieval_cue = Some(c);
                }
            }
            Ok(Err(e)) => log::warn!("[ingest] extract failed: {e}"),
            Err(_) => log::warn!("[ingest] extract timed out after 30s"),
        }
    }

    // Apply the refined classification to the stored memory.
    if let Err(e) = db
        .apply_enrichment(
            source_id,
            &final_memory_type,
            final_domain.as_deref(),
            final_quality.as_deref(),
            &final_supersede_mode,
            final_structured_fields.as_deref(),
            final_retrieval_cue.as_deref(),
            final_event_date,
            final_event_end,
            final_importance,
        )
        .await
    {
        log::warn!("[ingest] apply_enrichment failed: {e}");
    }

    // Write tags to MemoryDB.
    if !classified_tags.is_empty() {
        if let Err(e) = db
            .set_document_tags("memory", source_id, classified_tags)
            .await
        {
            log::warn!("[ingest] set_document_tags failed: {e}");
        }
    }

    EnrichmentOutcome {
        final_memory_type,
        final_domain,
        final_quality,
        final_importance,
        final_structured_fields,
        final_retrieval_cue,
        final_event_date,
        final_event_end,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassificationSliceReport {
    pub selected: bool,
    pub committed: bool,
    pub llm_calls: usize,
}

/// Advance one durable memory classification stage with at most one provider
/// call. Field/tag writes and the versioned receipt share one guarded commit.
pub async fn run_classification_enrichment_slice(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
) -> Result<ClassificationSliceReport, WenlanError> {
    const MAX_ATTEMPTS: u32 = 3;
    let Some(input) = db.get_classification_candidate(MAX_ATTEMPTS).await? else {
        return Ok(ClassificationSliceReport::default());
    };

    let request = crate::llm_provider::LlmRequest {
        system_prompt: Some(prompts.classify_memory_quality.clone()),
        user_prompt: input.content.chars().take(1000).collect(),
        max_tokens: 128,
        temperature: 0.1,
        label: Some("classify".into()),
        timeout_secs: None,
    };
    let generated =
        tokio::time::timeout(std::time::Duration::from_secs(30), llm.generate(request)).await;
    let classification = match generated {
        Ok(Ok(output)) => crate::llm_provider::parse_classify_response(&output),
        Ok(Err(error)) => {
            let attempt = input.prior_attempts.saturating_add(1);
            let status = if attempt >= MAX_ATTEMPTS {
                "abandoned"
            } else {
                "needs_retry"
            };
            let committed = db
                .record_enrichment_step_at_version(
                    &input.source_id,
                    "classify",
                    status,
                    Some(&error.to_string()),
                    input.version,
                )
                .await?;
            return Ok(ClassificationSliceReport {
                selected: true,
                committed,
                llm_calls: 1,
            });
        }
        Err(_) => {
            let attempt = input.prior_attempts.saturating_add(1);
            let status = if attempt >= MAX_ATTEMPTS {
                "abandoned"
            } else {
                "needs_retry"
            };
            let committed = db
                .record_enrichment_step_at_version(
                    &input.source_id,
                    "classify",
                    status,
                    Some("classification timed out after 30s"),
                    input.version,
                )
                .await?;
            return Ok(ClassificationSliceReport {
                selected: true,
                committed,
                llm_calls: 1,
            });
        }
    };
    let Some(classification) = classification else {
        let attempt = input.prior_attempts.saturating_add(1);
        let status = if attempt >= MAX_ATTEMPTS {
            "abandoned"
        } else {
            "needs_retry"
        };
        let committed = db
            .record_enrichment_step_at_version(
                &input.source_id,
                "classify",
                status,
                Some("classification response was invalid"),
                input.version,
            )
            .await?;
        return Ok(ClassificationSliceReport {
            selected: true,
            committed,
            llm_calls: 1,
        });
    };

    let derived_memory_type =
        (!input.origin.memory_type_explicit).then_some(classification.memory_type.as_str());
    let derived_supersede_mode = derived_memory_type.map(|memory_type| {
        if memory_type == "decision" {
            "archive"
        } else {
            "hide"
        }
    });
    let derived_space = if input.space.is_none() && !input.origin.space_rejected {
        db.registered_space_or_none(classification.space.as_deref())
            .await?
    } else {
        None
    };
    let committed = db
        .commit_classification_at_version(
            &input.source_id,
            input.version,
            derived_memory_type,
            derived_space.as_deref(),
            classification.quality.as_deref(),
            derived_supersede_mode,
            classification.importance,
            &classification.tags,
        )
        .await?;
    Ok(ClassificationSliceReport {
        selected: true,
        committed,
        llm_calls: 1,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuredExtractSliceReport {
    pub selected: bool,
    pub committed: bool,
    pub llm_calls: usize,
}

/// Advance one durable structured-extraction stage. Explicit structured fields
/// consume no provider budget and receive only a current-version receipt.
pub async fn run_structured_extract_slice(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
) -> Result<StructuredExtractSliceReport, WenlanError> {
    const MAX_ATTEMPTS: u32 = 3;
    let Some(input) = db.get_structured_extract_candidate(MAX_ATTEMPTS).await? else {
        return Ok(StructuredExtractSliceReport::default());
    };
    if input.origin.structured_fields_explicit {
        let committed = db
            .commit_structured_extract_at_version(
                &input.source_id,
                input.version,
                None,
                None,
                None,
                None,
            )
            .await?;
        return Ok(StructuredExtractSliceReport {
            selected: true,
            committed,
            llm_calls: 0,
        });
    }

    let memory_type = input.memory_type.as_deref().unwrap_or("fact");
    let prompt = crate::memory_schema::extraction_prompt_with_template(
        memory_type,
        &prompts.extract_structured_fields,
    );
    let generated = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        llm.generate(crate::llm_provider::LlmRequest {
            system_prompt: Some(prompt),
            user_prompt: input.content.chars().take(1500).collect(),
            max_tokens: 256,
            temperature: 0.1,
            label: Some("structured_extract".into()),
            timeout_secs: None,
        }),
    )
    .await;
    let output = match generated {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            let attempt = input.prior_attempts.saturating_add(1);
            let status = if attempt >= MAX_ATTEMPTS {
                "abandoned"
            } else {
                "needs_retry"
            };
            let committed = db
                .record_enrichment_step_at_version(
                    &input.source_id,
                    "structured_extract",
                    status,
                    Some(&error.to_string()),
                    input.version,
                )
                .await?;
            return Ok(StructuredExtractSliceReport {
                selected: true,
                committed,
                llm_calls: 1,
            });
        }
        Err(_) => {
            let attempt = input.prior_attempts.saturating_add(1);
            let status = if attempt >= MAX_ATTEMPTS {
                "abandoned"
            } else {
                "needs_retry"
            };
            let committed = db
                .record_enrichment_step_at_version(
                    &input.source_id,
                    "structured_extract",
                    status,
                    Some("structured extraction timed out after 30s"),
                    input.version,
                )
                .await?;
            return Ok(StructuredExtractSliceReport {
                selected: true,
                committed,
                llm_calls: 1,
            });
        }
    };
    let extracted = crate::llm_provider::parse_extraction_response(&output);
    let committed = db
        .commit_structured_extract_at_version(
            &input.source_id,
            input.version,
            extracted.structured_fields.as_deref(),
            extracted.retrieval_cue.as_deref(),
            extracted.event_date,
            extracted.event_end,
        )
        .await?;
    Ok(StructuredExtractSliceReport {
        selected: true,
        committed,
        llm_calls: 1,
    })
}

/// Run the canonical post-store enrichment for a single already-upserted memory.
///
/// Behaviour is log-and-degrade at every step (an LLM/DB error warns and the
/// step is skipped, never propagated) — identical to the server task it
/// replaces, which returned `()`. The returned [`EnrichmentOutcome`] reflects
/// the values written.
///
/// Locking: this fn holds NO `RwLock` guard. The caller snapshots `db`, `llm`,
/// `prompts`, `refinery`, `distillation`, and `knowledge_path` out of its state
/// guard and drops that guard BEFORE calling (per the never-hold-a-guard-across-
/// `.await` rule). `cancel` threads the reflection-debounce cooperative-cancel
/// flag; `None` runs every step to completion.
#[allow(clippy::too_many_arguments)]
pub async fn run_canonical_enrichment(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    entity_id: Option<&str>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    refinery: &RefineryConfig,
    distillation: &DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
    opts: &EnrichmentOpts,
    cancel: Option<&AtomicBool>,
) -> EnrichmentOutcome {
    // Phase 1: classify + extract + apply_enrichment + tags. Runs only when an
    // LLM is available; the no-LLM path leaves the classification at the sync
    // placeholder (initial_*). Extracted into `run_classification_enrichment`
    // so the eval seed can reuse exactly this pass — additively, over
    // already-upserted rows — without re-running the bulk entity/title/page
    // passes its scale-optimized pipeline already owns.
    let outcome = match llm {
        Some(llm) => {
            run_classification_enrichment(db, source_id, content, llm, prompts, opts).await
        }
        None => EnrichmentOutcome {
            final_memory_type: opts.initial_memory_type.clone(),
            final_domain: opts.initial_domain.clone(),
            final_quality: None,
            final_importance: None,
            final_structured_fields: opts.initial_structured_fields.clone(),
            final_retrieval_cue: None,
            final_event_date: None,
            final_event_end: None,
        },
    };

    // Phase 2: existing post-ingest enrichment (entity linking, title
    // enrichment, page growth). Runs with the enriched fields Phase 1 produced.
    if let Err(e) = crate::post_ingest::run_post_ingest_enrichment(
        db,
        source_id,
        content,
        entity_id,
        Some(outcome.final_memory_type.as_str()),
        outcome.final_domain.as_deref(),
        outcome.final_structured_fields.as_deref(),
        llm,
        prompts,
        refinery,
        distillation,
        knowledge_path,
        cancel,
        None, // precomputed_kg
    )
    .await
    {
        log::warn!("[ingest] post-ingest enrichment failed: {e}");
    }

    // Phase 3 (T14): dual-pool dedup + contradiction resolution.
    //
    // Behind `WENLAN_ENABLE_DUAL_POOL_RESOLVE` (default OFF -> no-op and the
    // path is byte-identical). Runs LAST so the `event_date` Phase 1 wrote is
    // visible for bidirectional temporal expiry. Best-effort: log-and-degrade.
    if crate::db::dual_pool_resolve_enabled() {
        let resolve_emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
        match crate::synthesis::refinement_queue::resolve_dual_pool(
            db,
            source_id,
            llm,
            prompts,
            &resolve_emitter,
        )
        .await
        {
            Ok(outcome) => {
                if !outcome.invalidated.is_empty()
                    || outcome.expired_incoming
                    || !outcome.flagged_for_review.is_empty()
                    || !outcome.dedup_proposals.is_empty()
                {
                    log::info!(
                        "[ingest] dual-pool resolve: invalidated={:?} expired_incoming={} \
                         flagged={:?} dedup_proposals={}",
                        outcome.invalidated,
                        outcome.expired_incoming,
                        outcome.flagged_for_review,
                        outcome.dedup_proposals.len(),
                    );
                }
            }
            Err(e) => log::warn!("[ingest] dual-pool resolve failed: {e}"),
        }
    }

    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::EnrichmentOrigin;
    use crate::llm_provider::SequencedMockProvider;
    use crate::sources::RawDocument;

    async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDB::new(db_path.as_path(), Arc::new(NoopEmitter))
            .await
            .unwrap();
        (db, dir)
    }

    fn seed_doc(source_id: &str, content: &str) -> RawDocument {
        RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: content.chars().take(40).collect(),
            summary: None,
            content: content.to_string(),
            url: None,
            last_modified: chrono::Utc::now().timestamp(),
            metadata: std::collections::HashMap::new(),
            memory_type: Some("fact".to_string()),
            space: None,
            source_agent: Some("test".to_string()),
            confidence: Some(0.7),
            confirmed: Some(false),
            stability: Some("new".to_string()),
            supersedes: None,
            pending_revision: false,
            entity_id: None,
            quality: None,
            importance: None,
            is_recap: false,
            enrichment_status: "raw".to_string(),
            supersede_mode: "hide".to_string(),
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            content_hash: None,
        }
    }

    fn opts_no_agent_overrides() -> EnrichmentOpts {
        EnrichmentOpts {
            initial_memory_type: "fact".to_string(),
            initial_domain: None,
            rejected_explicit_domain: false,
            initial_supersede_mode: "hide".to_string(),
            initial_structured_fields: None,
            agent_supplied_memory_type: false,
            agent_supplied_profile_alias: false,
            agent_supplied_structured_fields: false,
        }
    }

    #[tokio::test]
    async fn classification_slice_preserves_explicit_type_and_commits_one_receipt() {
        let (db, _dir) = test_db().await;
        db.create_space("work", None, false).await.unwrap();
        let mut doc = seed_doc(
            "mem_classification_slice",
            "The launch decision belongs to the work project and is high priority.",
        );
        doc.memory_type = Some("decision".to_string());
        db.upsert_enrichment_origin(
            "mem_classification_slice",
            EnrichmentOrigin {
                memory_type_explicit: true,
                structured_fields_explicit: false,
                space_rejected: false,
            },
        )
        .await
        .unwrap();
        db.upsert_documents(vec![doc]).await.unwrap();
        let llm: Arc<dyn LlmProvider> = Arc::new(SequencedMockProvider::new(vec![
            r#"{"memory_type":"preference","domain":"work","quality":"high","importance":8,"tags":["launch","priority"]}"#,
        ]));

        let report = run_classification_enrichment_slice(&db, &llm, &PromptRegistry::default())
            .await
            .unwrap();
        assert!(report.selected);
        assert!(report.committed);
        assert_eq!(report.llm_calls, 1);
        let (memory_type, space) = db
            .get_memory_classification("mem_classification_slice")
            .await
            .unwrap();
        assert_eq!(memory_type.as_deref(), Some("decision"));
        assert_eq!(space.as_deref(), Some("work"));
        assert_eq!(
            db.get_document_tags("memory", "mem_classification_slice")
                .await
                .unwrap(),
            vec!["launch".to_string(), "priority".to_string()]
        );
        let steps = db
            .get_enrichment_steps("mem_classification_slice")
            .await
            .unwrap();
        let classify = steps
            .iter()
            .find(|step| step.step == "classify")
            .expect("classification receipt");
        assert_eq!(classify.status, "ok");
        assert_eq!(classify.input_version, Some(1));
    }

    #[tokio::test]
    async fn structured_slice_skips_inference_for_explicit_fields() {
        let (db, _dir) = test_db().await;
        let mut doc = seed_doc(
            "mem_structured_explicit",
            "Explicit structured input must survive every background retry.",
        );
        doc.memory_type = Some("decision".to_string());
        doc.structured_fields = Some(r#"{"claim":"keep me"}"#.to_string());
        db.upsert_enrichment_origin(
            "mem_structured_explicit",
            EnrichmentOrigin {
                memory_type_explicit: true,
                structured_fields_explicit: true,
                space_rejected: false,
            },
        )
        .await
        .unwrap();
        db.upsert_documents(vec![doc]).await.unwrap();
        assert!(
            db.record_enrichment_step_at_version(
                "mem_structured_explicit",
                "classify",
                "ok",
                None,
                1,
            )
            .await
            .unwrap()
        );
        let llm: Arc<dyn LlmProvider> = Arc::new(SequencedMockProvider::new(vec![]));

        let report = run_structured_extract_slice(&db, &llm, &PromptRegistry::default())
            .await
            .unwrap();
        assert!(report.selected);
        assert!(report.committed);
        assert_eq!(report.llm_calls, 0);
        let detail = db
            .get_memory_detail("mem_structured_explicit")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            detail.structured_fields.as_deref(),
            Some(r#"{"claim":"keep me"}"#)
        );
        let steps = db
            .get_enrichment_steps("mem_structured_explicit")
            .await
            .unwrap();
        let structured = steps
            .iter()
            .find(|step| step.step == "structured_extract")
            .expect("structured receipt");
        assert_eq!(structured.status, "ok");
        assert_eq!(structured.input_version, Some(1));
    }

    /// The canonical enrichment must classify + extract via the LLM and persist
    /// the refined classification. This is the property the old eval shortcut
    /// LACKED: it wrote entity + title + page only, never `importance` (T8
    /// salience) or `event_date` (T11/T20 temporal). The outcome carries the
    /// parsed importance + event_date; the DB row carries the refined
    /// memory_type + space (proving `apply_enrichment` ran end-to-end).
    #[tokio::test]
    async fn canonical_enrichment_classifies_and_persists() {
        let (db, _dir) = test_db().await;
        db.create_space("work", None, false).await.unwrap();
        let source_id = "mem_canon_test_1";
        let content =
            "Switched the team standup to 9am on 2026-01-15 because mornings work better.";
        db.upsert_documents(vec![seed_doc(source_id, content)])
            .await
            .unwrap();

        // Call 0 = classify, call 1 = extract (SequencedMockProvider advances).
        let llm: Arc<dyn LlmProvider> = Arc::new(SequencedMockProvider::new(vec![
            r#"{"memory_type":"preference","domain":"work","quality":"high","importance":4,"tags":["standup","schedule"]}"#,
            r#"{"event_date":"2026-01-15","retrieval_cue":"standup time change","topic":"standup"}"#,
        ]));

        let prompts = PromptRegistry::default();
        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();
        let opts = opts_no_agent_overrides();

        let outcome = run_canonical_enrichment(
            &db,
            source_id,
            content,
            None,
            Some(&llm),
            &prompts,
            &refinery,
            &distillation,
            None,
            &opts,
            None,
        )
        .await;

        // Outcome reflects the parsed classify + extract (the write-time signals
        // the eval shortcut never produced).
        assert_eq!(outcome.final_memory_type, "preference");
        assert_eq!(outcome.final_domain.as_deref(), Some("work"));
        assert_eq!(outcome.final_quality.as_deref(), Some("high"));
        assert_eq!(outcome.final_importance, Some(4));
        let expected_event_date = chrono::NaiveDate::from_ymd_opt(2026, 1, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        assert_eq!(outcome.final_event_date, Some(expected_event_date));

        // apply_enrichment persisted the refined type + space to the stored row.
        let (mt, space) = db.get_memory_classification(source_id).await.unwrap();
        assert_eq!(mt.as_deref(), Some("preference"));
        assert_eq!(space.as_deref(), Some("work"));
    }

    #[tokio::test]
    async fn canonical_enrichment_ignores_unregistered_classifier_space() {
        let (db, _dir) = test_db().await;
        let source_id = "mem_canon_unregistered_space";
        let content = "A model-classified memory should not create an origin space again.";
        db.upsert_documents(vec![seed_doc(source_id, content)])
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(SequencedMockProvider::new(vec![
            r#"{"memory_type":"fact","domain":"origin","quality":"high","tags":["naming"]}"#,
            r#"{"retrieval_cue":"origin naming regression"}"#,
        ]));

        let prompts = PromptRegistry::default();
        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();
        let opts = opts_no_agent_overrides();

        let outcome = run_canonical_enrichment(
            &db,
            source_id,
            content,
            None,
            Some(&llm),
            &prompts,
            &refinery,
            &distillation,
            None,
            &opts,
            None,
        )
        .await;

        assert_eq!(
            outcome.final_domain, None,
            "unregistered classifier spaces must be ignored"
        );

        let (_mt, space) = db.get_memory_classification(source_id).await.unwrap();
        assert_eq!(
            space, None,
            "unregistered classifier spaces must not be persisted"
        );
        assert!(
            db.get_space("origin").await.unwrap().is_none(),
            "unregistered classifier spaces must not create a space row"
        );
    }

    #[tokio::test]
    async fn canonical_enrichment_preserves_rejected_explicit_space_as_unscoped() {
        let (db, _dir) = test_db().await;
        db.create_space("work", None, false).await.unwrap();
        let source_id = "mem_canon_rejected_explicit_space";
        let content = "An explicitly rejected space should stay uncategorized after enrichment.";
        db.upsert_documents(vec![seed_doc(source_id, content)])
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(SequencedMockProvider::new(vec![
            r#"{"memory_type":"fact","domain":"work","quality":"high","tags":["scope"]}"#,
            r#"{"retrieval_cue":"scope regression"}"#,
        ]));

        let prompts = PromptRegistry::default();
        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();
        let opts = EnrichmentOpts {
            rejected_explicit_domain: true,
            ..opts_no_agent_overrides()
        };

        let outcome = run_canonical_enrichment(
            &db,
            source_id,
            content,
            None,
            Some(&llm),
            &prompts,
            &refinery,
            &distillation,
            None,
            &opts,
            None,
        )
        .await;

        assert_eq!(
            outcome.final_domain, None,
            "classifier space must not override an explicitly rejected request space"
        );

        let (_mt, space) = db.get_memory_classification(source_id).await.unwrap();
        assert_eq!(
            space, None,
            "memory must remain uncategorized after enrichment"
        );
    }

    /// With no LLM available, Phase 1 is skipped entirely (no classify/extract),
    /// the classification stays at the initial placeholder, and the fn still
    /// returns a well-formed outcome — matching the server's behaviour when
    /// `state.llm` is `None`.
    #[tokio::test]
    async fn canonical_enrichment_no_llm_keeps_placeholder() {
        let (db, _dir) = test_db().await;
        let source_id = "mem_canon_test_2";
        let content = "A plain fact with no enrichment provider available at all.";
        db.upsert_documents(vec![seed_doc(source_id, content)])
            .await
            .unwrap();

        let prompts = PromptRegistry::default();
        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();
        let opts = opts_no_agent_overrides();

        let outcome = run_canonical_enrichment(
            &db,
            source_id,
            content,
            None,
            None,
            &prompts,
            &refinery,
            &distillation,
            None,
            &opts,
            None,
        )
        .await;

        assert_eq!(outcome.final_memory_type, "fact");
        assert_eq!(outcome.final_importance, None);
        assert_eq!(outcome.final_event_date, None);
    }
}
