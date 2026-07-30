// SPDX-License-Identifier: Apache-2.0
//! Post-ingest enrichment pipeline.
//!
//! Legacy one-shot enrichment entrypoint retained for eval and explicit core
//! callers. Production store/import requests only register durable inputs;
//! fixed-stage ambient slices below own background work.
//!
//! Steps:
//! 1. Entity auto-linking (vector search entities > 0.85 distance → set entity_id)
//!    1b. Store-time entity extraction (LLM extract if auto-link found no match)
//! 2. Title enrichment (LLM short title if current looks truncated)
//! 4. (Removed — recaps now handled by event-driven scheduler)
//! 5. Concept growth (update matching page with new memory)
//! 6. (Removed -- enrichment status derived from per-step outcomes in enrichment_steps table)
//!
//! Removed steps (dedup_merge and detect_contradiction proposals):
//! - Dedup check: emitted dedup_merge proposals; accept path is deprecated stale-v1.
//!   Distillation handles dedup. Removed in post-PR #109 cleanup.
//! - Contradiction check: emitted detect_contradiction proposals; accept path calls
//!   flag_memory_for_revision which does not set supersedes IS NOT NULL, so proposals
//!   never surface in /brief. The topic-match-protected path in memory_routes.rs is
//!   the only working contradiction-detection path. Removed in post-PR #109 cleanup.

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::llm_provider::LlmProvider;
use crate::prompts::PromptRegistry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wenlan_types::requests::UpdatePageRequest;

/// Result of the title enrichment step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TitleEnrichResult {
    /// Title was replaced with an LLM-generated short title.
    Enriched,
    /// Title was not truncated (or memory is recap/merged) -- no action needed.
    NotNeeded,
    /// Title IS truncated but LLM output was rejected (too long, generic, etc.).
    LlmRejected,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TitleEnrichmentSliceReport {
    pub selected: bool,
    pub committed: bool,
    pub llm_calls: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PageGrowthSliceReport {
    pub selected: bool,
    pub matched: bool,
    pub terminal_no_match: bool,
    pub committed: bool,
    pub llm_calls: usize,
}

/// Advance one durable title stage. The selector admits only titles that still
/// look automatically truncated; the derived title and receipt commit together
/// against both the memory generation and the title snapshot.
pub async fn run_title_enrichment_slice(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
) -> Result<TitleEnrichmentSliceReport, WenlanError> {
    const MAX_ATTEMPTS: u32 = 3;
    let Some(input) = db.get_title_enrichment_candidate(MAX_ATTEMPTS).await? else {
        return Ok(TitleEnrichmentSliceReport::default());
    };
    let Some(title) = crate::refinery::generate_short_title(llm, &input.content).await else {
        let attempt = input.prior_attempts.saturating_add(1);
        let status = if attempt >= MAX_ATTEMPTS {
            "abandoned"
        } else {
            "needs_retry"
        };
        let committed = db
            .record_enrichment_step_at_version(
                &input.source_id,
                "title_enrich",
                status,
                Some("title generation failed or output was rejected"),
                input.version,
            )
            .await?;
        return Ok(TitleEnrichmentSliceReport {
            selected: true,
            committed,
            llm_calls: 1,
        });
    };
    let committed = db
        .commit_title_at_version(&input.source_id, input.version, &input.title, &title)
        .await?;
    Ok(TitleEnrichmentSliceReport {
        selected: true,
        committed,
        llm_calls: 1,
    })
}

async fn record_page_growth_failure(
    db: &MemoryDB,
    input: &crate::db::PageGrowthInput,
    error: &str,
) -> Result<bool, WenlanError> {
    const MAX_ATTEMPTS: u32 = 3;
    let attempt = input.prior_attempts.saturating_add(1);
    let status = if attempt >= MAX_ATTEMPTS {
        "abandoned"
    } else {
        "needs_retry"
    };
    db.record_enrichment_step_at_version(
        &input.source_id,
        "page_growth",
        status,
        Some(error),
        input.version,
    )
    .await
}

/// Advance one restart-safe Page-growth item. Matching is CPU-only and occurs
/// before admission to the model. A no-match result is terminal for the
/// current memory generation; a match gets exactly one model request and a
/// dual Memory/Page CAS commit.
pub async fn run_page_growth_slice(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    growth_threshold: f64,
    knowledge_path: Option<&std::path::Path>,
) -> Result<PageGrowthSliceReport, WenlanError> {
    const MAX_ATTEMPTS: u32 = 3;
    let Some(input) = db
        .get_page_growth_candidate(MAX_ATTEMPTS, crate::db::entity_sweep_enabled())
        .await?
    else {
        return Ok(PageGrowthSliceReport::default());
    };

    let mem_embedding = match db.generate_embeddings(std::slice::from_ref(&input.content)) {
        Ok(mut embeddings) => embeddings.pop(),
        Err(error) => {
            let committed =
                record_page_growth_failure(db, &input, &format!("memory embedding: {error}"))
                    .await?;
            return Ok(PageGrowthSliceReport {
                selected: true,
                committed,
                ..Default::default()
            });
        }
    };
    let Some(mem_embedding) = mem_embedding else {
        let committed =
            record_page_growth_failure(db, &input, "memory embedding was empty").await?;
        return Ok(PageGrowthSliceReport {
            selected: true,
            committed,
            ..Default::default()
        });
    };

    let Some(page) = db
        .find_matching_page_scoped(
            input.entity_id.as_deref(),
            &mem_embedding,
            growth_threshold,
            input.space.as_deref(),
            false,
        )
        .await?
    else {
        let committed = db
            .record_enrichment_step_at_version(
                &input.source_id,
                "page_growth",
                "ok",
                None,
                input.version,
            )
            .await?;
        return Ok(PageGrowthSliceReport {
            selected: true,
            matched: false,
            terminal_no_match: true,
            committed,
            llm_calls: 0,
        });
    };
    let expected_source_revision = db.get_page_source_revision(&page.id).await?;

    let clean_current = crate::citations::strip_markers(&page.content);
    let evidence = db.get_page_evidence(&page.id).await.unwrap_or_default();
    let mut locators: Vec<String> = evidence
        .iter()
        .filter(|item| item.source_kind == "memory")
        .filter_map(|item| item.locator.clone())
        .filter(|locator| locator != &input.source_id)
        .collect();
    let mut seen = std::collections::HashSet::new();
    locators.retain(|locator| seen.insert(locator.clone()));
    locators.push(input.source_id.clone());

    let memories = db.get_memories_by_source_ids(&locators).await?;
    let by_source: std::collections::HashMap<_, _> = memories
        .into_iter()
        .map(|memory| (memory.source_id.clone(), memory))
        .collect();
    let kinds = db.resolve_source_kinds(&locators).await.unwrap_or_default();
    let numbered: Vec<crate::citations::NumberedSource> = locators
        .iter()
        .filter_map(|locator| by_source.get(locator))
        .enumerate()
        .map(|(index, memory)| crate::citations::NumberedSource {
            index: (index + 1) as u32,
            source_kind: kinds
                .get(&memory.source_id)
                .cloned()
                .unwrap_or_else(|| "memory".to_string()),
            locator: memory.source_id.clone(),
            text: memory.content.chars().take(800).collect(),
        })
        .collect();
    if numbered
        .last()
        .is_none_or(|source| source.locator != input.source_id)
    {
        let committed = record_page_growth_failure(
            db,
            &input,
            "triggering memory could not be resolved for citation",
        )
        .await?;
        return Ok(PageGrowthSliceReport {
            selected: true,
            matched: true,
            terminal_no_match: false,
            committed,
            llm_calls: 0,
        });
    }
    if db.get_page_source_revision(&page.id).await? != expected_source_revision {
        return Ok(PageGrowthSliceReport {
            selected: true,
            matched: true,
            terminal_no_match: false,
            committed: false,
            llm_calls: 0,
        });
    }
    let existing_sources = &numbered[..numbered.len() - 1];
    let user_prompt = format!(
        "## Current Concept\n{}\n\n## Numbered Sources\n{}\n\n## New Memory\n[{}] {}",
        clean_current,
        crate::citations::build_numbered_block(existing_sources),
        numbered.len(),
        input.content,
    );

    let response = match llm
        .generate(crate::llm_provider::LlmRequest {
            system_prompt: Some(prompts.update_page.clone()),
            user_prompt,
            max_tokens: 1024,
            temperature: 0.1,
            label: Some("update_page".to_string()),
            timeout_secs: None,
        })
        .await
    {
        Ok(response) => response,
        Err(error) => {
            let committed =
                record_page_growth_failure(db, &input, &format!("page growth LLM: {error}"))
                    .await?;
            return Ok(PageGrowthSliceReport {
                selected: true,
                matched: true,
                terminal_no_match: false,
                committed,
                llm_calls: 1,
            });
        }
    };
    let updated = crate::llm_provider::strip_think_tags(&response);
    let updated = updated.trim();
    if updated.is_empty() {
        let committed =
            record_page_growth_failure(db, &input, "page growth output was empty").await?;
        return Ok(PageGrowthSliceReport {
            selected: true,
            matched: true,
            terminal_no_match: false,
            committed,
            llm_calls: 1,
        });
    }

    let (updated_body, citations, stats) =
        crate::citations::process_citation_output(updated, &numbered);
    if let Some(threshold) = crate::post_write::merge_shrink_threshold() {
        if !crate::retrieval::integrity::body_shrink_ok(&page.content, &updated_body, threshold) {
            let committed =
                record_page_growth_failure(db, &input, "page growth shrink guard rejected output")
                    .await?;
            return Ok(PageGrowthSliceReport {
                selected: true,
                matched: true,
                terminal_no_match: false,
                committed,
                llm_calls: 1,
            });
        }
    }

    let mut source_ids = page.source_memory_ids.clone();
    if !source_ids.contains(&input.source_id) {
        source_ids.push(input.source_id.clone());
    }
    let citations_json = serde_json::to_string(&citations).unwrap_or_else(|_| "[]".to_string());
    let write = crate::post_write::update_page_growth_at_versions(
        db,
        &page.id,
        UpdatePageRequest {
            content: updated_body,
            source_memory_ids: source_ids,
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        page.version,
        expected_source_revision,
        &input.source_id,
        input.version,
        knowledge_path,
        Some((citations_json, stats.summary())),
    )
    .await?;
    if write.wrote {
        let agent = db
            .get_memory_source_agent(&input.source_id)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(|| "system".to_string());
        let source_ids = vec![input.source_id.clone()];
        if let Err(error) = db
            .log_agent_activity(
                &agent,
                "page_grow",
                &source_ids,
                None,
                &format!("grew \"{}\"", page.title),
            )
            .await
        {
            log::warn!("[page_growth] activity log failed: {error}");
        }
    }

    Ok(PageGrowthSliceReport {
        selected: true,
        matched: true,
        terminal_no_match: false,
        committed: write.wrote,
        llm_calls: 1,
    })
}

/// True iff the caller has signalled cancellation. `None` (the default-OFF
/// path) is never cancelled, so the flag is fully inert unless an operator
/// opts into `WENLAN_ENABLE_REFLECTION_DEBOUNCE` and a newer same-agent write
/// supersedes this one mid-burst. Checked only BETWEEN best-effort steps so a
/// step is never half-applied (clean-boundary cancellation).
fn is_cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.map(|c| c.load(Ordering::SeqCst)).unwrap_or(false)
}

/// Run post-ingest enrichment (async, non-blocking).
/// Called after store_memory fast track returns.
///
/// `cancel` is an opt-in cooperative-cancellation signal (T22 debounced
/// reflection). When `Some` and flipped to `true` by a newer same-agent write,
/// enrichment returns early at the next clean step boundary. `None` (the
/// default) keeps the path byte-identical to pre-T22 behaviour — every step
/// runs to completion.
#[allow(clippy::too_many_arguments)]
pub async fn run_post_ingest_enrichment(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    entity_id: Option<&str>,
    _memory_type: Option<&str>,
    space: Option<&str>,
    _structured_fields: Option<&str>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
    distillation: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
    cancel: Option<&AtomicBool>,
    precomputed_kg: Option<Vec<crate::extract::KgExtractionResult>>,
) -> Result<(), WenlanError> {
    log::info!("[post_ingest] enriching {source_id}");

    // Checkpoint 0: bail before any work if a newer write already superseded us.
    if is_cancelled(cancel) {
        log::info!("[post_ingest] {source_id}: cancelled before first step");
        return Ok(());
    }

    // 1. Entity auto-linking (only if not already linked)
    if entity_id.is_none() {
        match auto_link_entity(db, source_id, content, tuning).await {
            Ok(true) => {
                db.record_enrichment_step(source_id, "entity_link", "ok", None)
                    .await
                    .ok();
            }
            Ok(false) => {
                db.record_enrichment_step(source_id, "entity_link", "ok", None)
                    .await
                    .ok();
            }
            Err(e) => {
                log::warn!("[post_ingest] entity linking failed: {e}");
                db.record_enrichment_step(source_id, "entity_link", "failed", Some(&e.to_string()))
                    .await
                    .ok();
            }
        }
    } else {
        db.record_enrichment_step(source_id, "entity_link", "skipped", None)
            .await
            .ok();
    }

    if is_cancelled(cancel) {
        log::info!("[post_ingest] {source_id}: cancelled after entity_link");
        return Ok(());
    }

    // 2b. Store-time entity extraction with time-windowed batching
    // Re-check entity_id since auto_link_entity may have set it
    let current_entity_id = db
        .get_memory_entity_id(source_id)
        .await
        .unwrap_or(entity_id.map(|s| s.to_string()));
    if current_entity_id.is_none() {
        if let Some(llm_ref) = llm {
            // Look up source_agent from the DB for batch window queries
            let agent = db.get_memory_source_agent(source_id).await.unwrap_or(None);
            let persisted_space = db.get_memory_space(source_id).await.unwrap_or(None);

            // Check for recent memories from the same agent and persisted Space
            // for batched extraction. Space is a provenance boundary for
            // derived writes even though entity identity remains global.
            let batch = match &agent {
                Some(a) => db
                    .find_recent_batch(a, tuning.batch_window_secs, persisted_space.as_deref())
                    .await
                    .unwrap_or_default(),
                None => Vec::new(),
            };

            if batch.len() > 1 {
                // Batch extraction -- combine all recent memories for richer entity/relation extraction
                let combined: String = batch
                    .iter()
                    .enumerate()
                    .map(|(i, (_, c))| {
                        format!("{}. {}", i + 1, c.chars().take(500).collect::<String>())
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                log::info!(
                    "[post_ingest] batch entity extraction: {} memories in window",
                    batch.len()
                );

                // Extract from combined content, then commit against the anchor's
                // own content so span offsets stay char offsets into
                // `memories.content` of the stamped `source_memory_id` (§2.2) --
                // never against the combined batch blob.
                let extraction = crate::refinery::extract_kg(llm_ref, prompts, &combined).await;
                let commit_result = match extraction {
                    Ok(mut kg) => {
                        // A batch-extracted quote may belong to a non-anchor
                        // window member. Re-locate it against the anchor's real
                        // content and drop it on a miss -- never guess, never
                        // fuzzy-match, never keep blob-based offsets.
                        for kg_item in &mut kg {
                            for rel in &mut kg_item.relations {
                                if let Some(quote) = &rel.span {
                                    if crate::extract::locate_span_chars(content, quote).is_none() {
                                        rel.span = None;
                                    }
                                }
                            }
                        }
                        crate::refinery::commit_kg(
                            db,
                            source_id,
                            &kg,
                            content,
                            Some(&llm_ref.model_id()),
                        )
                        .await
                    }
                    Err(e) => Err(e),
                };

                // Link entities to all batch memories
                match commit_result {
                    Ok(Some(eid)) => {
                        // Link all batch memories to the extracted entity
                        for (batch_sid, _) in &batch {
                            if batch_sid != source_id {
                                let _ = db.update_memory_entity_id(batch_sid, &eid).await;
                            }
                        }
                        log::info!(
                            "[post_ingest] batch extraction linked {} memories to entity",
                            batch.len()
                        );
                        db.record_enrichment_step(source_id, "entity_extract", "ok", None)
                            .await
                            .ok();
                    }
                    Ok(None) => {
                        db.record_enrichment_step(source_id, "entity_extract", "ok", None)
                            .await
                            .ok();
                    }
                    Err(e) => {
                        log::warn!("[post_ingest] batch entity extraction failed: {e}");
                        db.record_enrichment_step(
                            source_id,
                            "entity_extract",
                            "failed",
                            Some(&e.to_string()),
                        )
                        .await
                        .ok();
                    }
                }
            } else {
                // Single memory extraction (no batch or source_agent unknown).
                // If a pre-computed KG was supplied by the caller, commit it directly
                // (skipping the inline LLM extract). Otherwise, run the LLM extract
                // as today. Both paths return Result<Option<String>, WenlanError>.
                match match precomputed_kg {
                    Some(kg) => {
                        crate::refinery::commit_kg(
                            db,
                            source_id,
                            &kg,
                            content,
                            Some(&llm_ref.model_id()),
                        )
                        .await
                    }
                    None => {
                        crate::refinery::extract_single_memory_entities(
                            db, llm_ref, prompts, source_id, content,
                        )
                        .await
                    }
                } {
                    Ok(Some(eid)) => {
                        let eid_prefix: String = eid.chars().take(12).collect();
                        log::info!("[post_ingest] {source_id}: extracted entity {eid_prefix}");
                        db.record_enrichment_step(source_id, "entity_extract", "ok", None)
                            .await
                            .ok();
                    }
                    Ok(None) => {
                        db.record_enrichment_step(source_id, "entity_extract", "ok", None)
                            .await
                            .ok();
                    }
                    Err(e) => {
                        log::warn!("[post_ingest] store-time entity extraction failed: {e}");
                        db.record_enrichment_step(
                            source_id,
                            "entity_extract",
                            "failed",
                            Some(&e.to_string()),
                        )
                        .await
                        .ok();
                    }
                }
            }
        }
    } else {
        db.record_enrichment_step(source_id, "entity_extract", "skipped", None)
            .await
            .ok();
    }

    if is_cancelled(cancel) {
        log::info!("[post_ingest] {source_id}: cancelled after entity_extract");
        return Ok(());
    }

    if is_cancelled(cancel) {
        log::info!("[post_ingest] {source_id}: cancelled before title_enrich");
        return Ok(());
    }

    // 5. Title enrichment — generate short topic title if current title is a truncation
    if let Some(llm_ref) = llm {
        match enrich_title(db, source_id, content, llm_ref, false).await {
            Ok(TitleEnrichResult::Enriched) => {
                log::info!("[post_ingest] {source_id}: title enriched");
                db.record_enrichment_step(source_id, "title_enrich", "ok", None)
                    .await
                    .ok();
            }
            Ok(TitleEnrichResult::NotNeeded) => {
                db.record_enrichment_step(source_id, "title_enrich", "ok", None)
                    .await
                    .ok();
            }
            Ok(TitleEnrichResult::LlmRejected) => {
                log::info!("[post_ingest] {source_id}: title LLM-rejected, queuing for retry");
                db.record_enrichment_step(
                    source_id,
                    "title_enrich",
                    "needs_retry",
                    Some("llm_rejected"),
                )
                .await
                .ok();
            }
            Err(e) => {
                log::warn!("[post_ingest] title enrichment failed: {e}");
                db.record_enrichment_step(
                    source_id,
                    "title_enrich",
                    "failed",
                    Some(&e.to_string()),
                )
                .await
                .ok();
            }
        }
    } else {
        db.record_enrichment_step(source_id, "title_enrich", "skipped", None)
            .await
            .ok();
    }

    // 6. Recap trigger — REMOVED. Recaps are now generated by the event-driven
    // scheduler (BurstEnd trigger) at the natural end of agent work sessions,
    // not on every write. generate_recaps_public remains in refinery's public
    // API for standalone core consumers. See 2026-04-12-event-driven-steep-triggers.

    if is_cancelled(cancel) {
        log::info!("[post_ingest] {source_id}: cancelled before page_growth");
        return Ok(());
    }

    // 7. Concept growth — update matching page with new memory
    let projection = knowledge_path.map(|path| {
        crate::export::knowledge::KnowledgeProjectionWrite::new(path.to_path_buf(), db)
    });
    let growth_entity_id = db
        .get_memory_entity_id(source_id)
        .await
        .unwrap_or_else(|_| {
            current_entity_id
                .clone()
                .or_else(|| entity_id.map(str::to_string))
        });
    let growth_space = db
        .get_memory_space(source_id)
        .await
        .unwrap_or_else(|_| space.map(str::to_string));
    match grow_page(
        db,
        source_id,
        content,
        growth_entity_id.as_deref(),
        growth_space.as_deref(),
        llm,
        prompts,
        distillation.page_growth_threshold,
    )
    .await
    {
        Ok(true) => {
            log::info!("[post_ingest] {source_id}: updated matching page");
            db.record_enrichment_step(source_id, "page_growth", "ok", None)
                .await
                .ok();
            if let Some(ref projection) = projection {
                write_grown_page(db, source_id, projection).await;
            }
        }
        Ok(false) => {
            // grow_page returns false when LLM is unavailable — treat as skipped
            if llm.map(|l| l.is_available()).unwrap_or(false) {
                db.record_enrichment_step(source_id, "page_growth", "ok", None)
                    .await
                    .ok();
            } else {
                db.record_enrichment_step(source_id, "page_growth", "skipped", None)
                    .await
                    .ok();
            }
        }
        Err(e) => {
            log::warn!("[post_ingest] page growth failed: {e}");
            db.record_enrichment_step(source_id, "page_growth", "failed", Some(&e.to_string()))
                .await
                .ok();
        }
    }
    drop(projection);

    // 7b. KG quality verification — check entity self-retrieval after all linking/extraction
    let final_entity_id = db
        .get_memory_entity_id(source_id)
        .await
        .unwrap_or(entity_id.map(|s| s.to_string()));
    if let Some(ref eid) = final_entity_id {
        if let Ok(detail) = db.get_entity_detail(eid).await {
            match crate::kg_quality::verify_entity(db, eid, &detail.entity.name).await {
                Ok(ref result) => {
                    for warning in &result.warnings {
                        log::warn!("[post_ingest] {}", warning);
                    }
                }
                Err(e) => log::warn!("[post_ingest] entity verification failed: {e}"),
            }
        }
    }

    Ok(())
}

/// Auto-link memory to an existing entity via vector search.
/// Links to the best matching entity with distance < 0.15 (cosine similarity > 0.85).
pub(crate) async fn auto_link_entity(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    tuning: &crate::tuning::RefineryConfig,
) -> Result<bool, WenlanError> {
    let entities = db.search_entities_by_vector(content, 3).await?;
    for entity in &entities {
        // distance is cosine distance — lower = more similar
        if entity.distance < tuning.entity_link_distance as f32 {
            db.update_memory_entity_id(source_id, &entity.entity.id)
                .await?;
            log::info!(
                "[post_ingest] auto-linked {} to entity '{}' (distance={:.3})",
                source_id,
                entity.entity.name,
                entity.distance,
            );
            return Ok(true);
        }
    }
    Ok(false)
}

/// Generate a short topic title if the current title looks like a content truncation.
///
/// By default, enrichment only fires when the title appears truncated (ends with "...",
/// matches the first content line verbatim, or is 75+ characters long). Eval benchmarks
/// use short synthetic titles that never trigger this heuristic. Set the env var
/// `EVAL_FORCE_TITLE_ENRICHMENT=1` to bypass the truncation check and always enrich.
/// Default behavior (env unset) is identical to before.
pub(crate) async fn enrich_title(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    llm: &Arc<dyn LlmProvider>,
    force: bool,
) -> Result<TitleEnrichResult, WenlanError> {
    // Skip recaps and merged memories — they get titles during generation
    if source_id.starts_with("recap_") || source_id.starts_with("merged_") {
        return Ok(TitleEnrichResult::NotNeeded);
    }

    let force_enrichment = force;

    // Check if current title is a truncation (ends with "..." or matches first line)
    let detail = db.get_memory_detail(source_id).await?;
    let current_title = match &detail {
        Some(d) => &d.title,
        None => return Ok(TitleEnrichResult::NotNeeded),
    };

    if !force_enrichment {
        let first_line = content.lines().next().unwrap_or(content);
        let is_truncated = current_title.ends_with("...")
            || current_title == first_line
            || current_title.len() >= 75;
        if !is_truncated {
            return Ok(TitleEnrichResult::NotNeeded);
        }
    }

    if let Some(short_title) = crate::refinery::generate_short_title(llm, content).await {
        db.update_title(source_id, &short_title).await?;
        Ok(TitleEnrichResult::Enriched)
    } else {
        Ok(TitleEnrichResult::LlmRejected)
    }
}

/// Check if new memory matches an existing page; if so, update it.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn grow_page(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    entity_id: Option<&str>,
    space: Option<&str>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    growth_threshold: f64,
) -> Result<bool, WenlanError> {
    let llm = match llm {
        Some(l) if l.is_available() => l,
        _ => return Ok(false),
    };

    // Get memory embedding for similarity check
    let embeddings = db.generate_embeddings(&[content.to_string()])?;
    let mem_embedding = match embeddings.first() {
        Some(e) => e,
        None => return Ok(false),
    };

    let page = match db
        .find_matching_page_scoped(entity_id, mem_embedding, growth_threshold, space, true)
        .await?
    {
        Some(c) => c,
        None => return Ok(false),
    };

    // (a) Input hygiene — strip stale [N] markers from the current body before
    // it goes into the prompt (⚖ §4). The old markers pointed at a numbered
    // list that no longer exists once we rebuild it below.
    let clean_current = crate::citations::strip_markers(&page.content);

    // (b) Numbered source list = this page's memory-kind evidence (linked
    // order), with the triggering new memory appended last.
    let evidence = db.get_page_evidence(&page.id).await.unwrap_or_default();
    let mut locators: Vec<String> = evidence
        .iter()
        .filter(|e| e.source_kind == "memory")
        .filter_map(|e| e.locator.clone())
        .collect();
    locators.push(source_id.to_string());
    locators.dedup();
    let mems = db.get_memories_by_source_ids(&locators).await?;
    // Resolve each source's typed kind (spec §5.1): the existing evidence is
    // pre-filtered to 'memory', but the newly-triggering `source_id` may be a
    // folder-doc or webpage source, so every entry resolves through the
    // shared resolver rather than hardcoding 'memory'.
    let kinds = db.resolve_source_kinds(&locators).await.unwrap_or_default();
    let numbered: Vec<crate::citations::NumberedSource> = mems
        .iter()
        .enumerate()
        .map(|(i, m)| crate::citations::NumberedSource {
            index: (i + 1) as u32,
            source_kind: kinds
                .get(&m.source_id)
                .cloned()
                .unwrap_or_else(|| "memory".to_string()),
            locator: m.source_id.clone(),
            text: m.content.chars().take(800).collect(),
        })
        .collect();
    if numbered.is_empty() {
        // The triggering memory failed to resolve — nothing to cite against.
        return Ok(false);
    }
    let existing_sources = &numbered[..numbered.len() - 1];

    // LLM: update page with new memory
    let user_prompt = format!(
        "## Current Concept\n{}\n\n## Numbered Sources\n{}\n\n## New Memory\n[{}] {}",
        clean_current,
        crate::citations::build_numbered_block(existing_sources),
        numbered.len(),
        content
    );

    let response = llm
        .generate(crate::llm_provider::LlmRequest {
            system_prompt: Some(prompts.update_page.clone()),
            user_prompt,
            max_tokens: 1024,
            temperature: 0.1,
            label: Some("update_page".to_string()),
            timeout_secs: None,
        })
        .await
        .map_err(|e| WenlanError::Llm(format!("page growth LLM: {e}")))?;

    let updated = crate::llm_provider::strip_think_tags(&response);
    let updated = updated.trim();

    if updated.is_empty() {
        return Ok(false);
    }

    // (c) Verify [N] markers against the numbered sources; out-of-range
    // markers are stripped from the body before it is saved.
    let (updated_body, cites, stats) =
        crate::citations::process_citation_output(updated, &numbered);
    log::info!(
        "[grow_page] page '{}' citations: {}",
        page.title,
        stats.summary()
    );

    // Shrink-guard pre-check (T17): early-exit before calling update_page.
    // update_page has its own shrink-guard backstop, but this early-exit
    // preserves the Ok(false) contract (skipped-growth) rather than Err.
    // Matches the is_empty() early-return contract at line 572.
    if let Some(threshold) = crate::post_write::merge_shrink_threshold() {
        if !crate::retrieval::integrity::body_shrink_ok(&page.content, &updated_body, threshold) {
            log::warn!(
                "[grow_page] shrink-guard skipped growth for page {}: new body ({} chars) < {}% of old ({} chars)",
                page.id,
                updated_body.chars().count(),
                (threshold * 100.0) as u32,
                page.content.chars().count(),
            );
            return Ok(false);
        }
    }

    // Update page with new content + add source memory
    let mut source_ids = page.source_memory_ids.clone();
    if !source_ids.contains(&source_id.to_string()) {
        source_ids.push(source_id.to_string());
    }
    let citations_json = serde_json::to_string(&cites).unwrap_or_else(|_| "[]".to_string());
    let write_result = crate::post_write::update_page(
        db,
        &page.id,
        UpdatePageRequest {
            content: updated_body,
            source_memory_ids: source_ids,
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "page_growth",
        false,
        None,
        Some((citations_json, stats.summary())),
    )
    .await?;
    if write_result.gated || !write_result.wrote {
        return Ok(false);
    }

    // Log activity: attribute to the agent who authored the triggering memory.
    let agent = db
        .get_memory_source_agent(source_id)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| "system".to_string());
    let detail = format!("grew \"{}\"", page.title);
    let ids = vec![source_id.to_string()];
    if let Err(e) = db
        .log_agent_activity(&agent, "page_grow", &ids, None, &detail)
        .await
    {
        log::warn!("[post_ingest] log page_grow activity failed: {e}");
    }

    Ok(true)
}

async fn write_grown_page(
    db: &MemoryDB,
    source_id: &str,
    projection: &crate::export::knowledge::KnowledgeProjectionWrite,
) {
    match db.find_page_by_source_memory(source_id).await {
        Ok(Some(page)) => match projection.write_page_gated(db, &page).await {
            Ok(Some(path)) => log::info!("[post_ingest] wrote page to {path}"),
            // Debug, not info: past the cutover this is the steady state for
            // every provisional page, so at info it would be pure noise. It is
            // still worth a line -- a page that silently stops being projected
            // is exactly the thing someone will come looking for.
            Ok(None) => log::debug!(
                "[post_ingest] not projecting page {} — an automatic reader may not see it",
                page.id
            ),
            Err(e) => log::warn!("[post_ingest] knowledge write failed: {e}"),
        },
        Ok(None) => {}
        Err(e) => log::warn!("[post_ingest] page lookup for knowledge write failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create an in-memory test database.
    async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDB::new(
            db_path.as_path(),
            std::sync::Arc::new(crate::events::NoopEmitter),
        )
        .await
        .unwrap();
        (db, dir)
    }

    /// Helper to build a minimal RawDocument for testing.
    fn make_doc(source_id: &str, content: &str) -> crate::sources::RawDocument {
        crate::sources::RawDocument {
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
            stability: None,
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

    #[tokio::test]
    async fn test_auto_link_entity_no_entities() {
        let (db, _dir) = test_db().await;
        let tuning = crate::tuning::RefineryConfig::default();
        let linked = auto_link_entity(&db, "mem_test", "Some content about nothing", &tuning)
            .await
            .unwrap();
        assert!(!linked, "should not link when no entities exist");
    }

    #[tokio::test]
    async fn title_slice_commits_one_current_result() {
        let (db, _dir) = test_db().await;
        let content = "The ambient scheduler keeps enrichment invisible to users.";
        db.upsert_documents(vec![make_doc("mem_title_slice", content)])
            .await
            .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE memories SET title=?1 WHERE source_id='mem_title_slice'",
                libsql::params![content],
            )
            .await
            .unwrap();
        }
        let llm: Arc<dyn LlmProvider> =
            Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
                "Invisible Ambient Enrichment",
            ]));

        let report = run_title_enrichment_slice(&db, &llm).await.unwrap();

        assert!(report.selected);
        assert!(report.committed);
        assert_eq!(report.llm_calls, 1);
        let detail = db
            .get_memory_detail("mem_title_slice")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(detail.title, "Invisible Ambient Enrichment");
        let receipt = db
            .get_enrichment_steps("mem_title_slice")
            .await
            .unwrap()
            .into_iter()
            .find(|step| step.step == "title_enrich")
            .expect("title receipt");
        assert_eq!(receipt.status, "ok");
        assert_eq!(receipt.input_version, Some(1));
    }

    #[tokio::test]
    async fn test_full_enrichment_pipeline() {
        let (db, _dir) = test_db().await;
        let doc = make_doc("mem_enrich_test", "The capital of France is Paris");
        db.upsert_documents(vec![doc]).await.unwrap();

        // Run enrichment — should complete without error
        run_post_ingest_enrichment(
            &db,
            "mem_enrich_test",
            "The capital of France is Paris",
            None,
            Some("fact"),
            None,
            None,
            None,
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            None, // cancel — T22, inert
            None, // precomputed_kg
        )
        .await
        .unwrap();

        // Verify enrichment_summary is derived from per-step outcomes
        let summary = db.get_enrichment_summary("mem_enrich_test").await.unwrap();
        assert_eq!(summary, "enriched");
    }

    #[tokio::test]
    async fn test_enrichment_records_per_step_outcomes() {
        let (db, _dir) = test_db().await;
        let doc = make_doc("mem_step_record", "The capital of France is Paris");
        db.upsert_documents(vec![doc]).await.unwrap();

        run_post_ingest_enrichment(
            &db,
            "mem_step_record",
            "The capital of France is Paris",
            None,
            Some("fact"),
            None,
            None,
            None,
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            None, // cancel — T22, inert
            None, // precomputed_kg
        )
        .await
        .unwrap();

        let steps = db.get_enrichment_steps("mem_step_record").await.unwrap();
        assert!(!steps.is_empty(), "should have recorded enrichment steps");

        // No provider is deferred work, not a terminal successful skip. Leaving
        // the step absent lets the ambient entity lane pick it up later.
        assert!(
            steps.iter().all(|s| s.step != "entity_extract"),
            "no-provider entity extraction must remain ambient backlog"
        );
        assert!(
            steps
                .iter()
                .all(|step| step.step != "entity_suggestion" && step.step != "page_contradiction"),
            "obsolete stub/heuristic stages must not emit receipts"
        );

        // title_enrich should be skipped (no LLM)
        let title = steps.iter().find(|s| s.step == "title_enrich").unwrap();
        assert_eq!(title.status, "skipped");

        // page_growth should be skipped (no LLM)
        let growth = steps.iter().find(|s| s.step == "page_growth").unwrap();
        assert_eq!(growth.status, "skipped");

        // Summary should be enriched (no failures)
        let summary = db.get_enrichment_summary("mem_step_record").await.unwrap();
        assert_eq!(summary, "enriched");
    }

    /// Regression guard for time-windowed store-time entity-extraction batching.
    ///
    /// `post_ingest::run_post_ingest_enrichment` at lines 62–134 looks for
    /// other recent same-agent memories lacking an `entity_id` and — if the
    /// batch size is >1 — runs one combined entity-extraction LLM call
    /// across all of them, then links the resulting entity to every batch
    /// member. This lets 10 `mcp__origin__remember` calls from the same
    /// agent coalesce into fewer LLM invocations instead of 10 isolated
    /// ones.
    ///
    /// The primitive under test is `MemoryDB::find_recent_batch` at
    /// `db.rs:6786`, which filters by:
    ///   - source = 'memory' AND chunk_index = 0
    ///   - source_agent = <agent>
    ///   - entity_id IS NULL
    ///   - last_modified > now - window_secs
    ///
    /// This test seeds three memories from the same agent through the
    /// normal `upsert_documents` path (so chunking + FTS triggers fire
    /// exactly as in production) and asserts they all show up in the
    /// batch window. Then it confirms that filling in an `entity_id` on
    /// one of them causes it to drop out of the batch — matching the
    /// real-world pattern where a prior enrichment pass has already
    /// linked the memory, so it shouldn't be redundantly re-extracted.
    ///
    /// NOTE: this test requires the FastEmbed model cache. In offline /
    /// cert-failing environments `test_db()` will fail at init — skip at
    /// the harness level, not by gating this test, since the blocker is
    /// shared with ~240 other tests in the crate.
    #[tokio::test]
    async fn find_recent_batch_collects_same_agent_memories_without_entity_id() {
        let (db, _dir) = test_db().await;
        let agent = "mcp-batch-test";
        let mut doc_a = make_doc(
            "mem_batch_a",
            "Alice joined the research team this quarter.",
        );
        doc_a.source_agent = Some(agent.to_string());
        let mut doc_b = make_doc(
            "mem_batch_b",
            "Alice presented the eval results at standup.",
        );
        doc_b.source_agent = Some(agent.to_string());
        let mut doc_c = make_doc("mem_batch_c", "Alice is migrating the daemon to axum 0.8.");
        doc_c.source_agent = Some(agent.to_string());
        db.upsert_documents(vec![doc_a, doc_b, doc_c])
            .await
            .unwrap();

        let window_secs = 600; // 10 min — generous, covers any test timing jitter
        let batch = db
            .find_recent_batch(agent, window_secs, None)
            .await
            .unwrap();
        let ids: Vec<String> = batch.iter().map(|(id, _)| id.clone()).collect();
        assert!(ids.contains(&"mem_batch_a".to_string()));
        assert!(ids.contains(&"mem_batch_b".to_string()));
        assert!(ids.contains(&"mem_batch_c".to_string()));
        assert_eq!(
            batch.len(),
            3,
            "all three same-agent memories must be in the batch window, got: {ids:?}"
        );

        // Flip one memory's entity_id — simulate a prior enrichment pass
        // that already linked it. find_recent_batch must drop it so the
        // next extraction doesn't re-process it.
        db.update_memory_entity_id("mem_batch_b", "entity_alice")
            .await
            .unwrap();
        let batch_after = db
            .find_recent_batch(agent, window_secs, None)
            .await
            .unwrap();
        let ids_after: Vec<String> = batch_after.iter().map(|(id, _)| id.clone()).collect();
        assert!(
            !ids_after.contains(&"mem_batch_b".to_string()),
            "memories with a linked entity_id must be filtered out, got: {ids_after:?}"
        );
        assert_eq!(
            batch_after.len(),
            2,
            "batch should shrink by exactly one after linking one entity"
        );
    }

    /// Second regression guard: agent isolation. Two agents storing at the
    /// same time should each see their own batch; one agent's memories
    /// must never leak into the other's extraction window.
    #[tokio::test]
    async fn find_recent_batch_isolates_by_source_agent() {
        let (db, _dir) = test_db().await;
        let mut doc_claude = make_doc("mem_iso_claude", "Claude Code scaffolded the plugin.");
        doc_claude.source_agent = Some("claude-code".to_string());
        let mut doc_cursor = make_doc("mem_iso_cursor", "Cursor produced a diff for the hook.");
        doc_cursor.source_agent = Some("cursor".to_string());
        db.upsert_documents(vec![doc_claude, doc_cursor])
            .await
            .unwrap();

        let claude_batch = db
            .find_recent_batch("claude-code", 600, None)
            .await
            .unwrap();
        let cursor_batch = db.find_recent_batch("cursor", 600, None).await.unwrap();
        let claude_ids: Vec<String> = claude_batch.iter().map(|(id, _)| id.clone()).collect();
        let cursor_ids: Vec<String> = cursor_batch.iter().map(|(id, _)| id.clone()).collect();

        assert_eq!(claude_ids, vec!["mem_iso_claude"]);
        assert_eq!(cursor_ids, vec!["mem_iso_cursor"]);
    }

    #[tokio::test]
    async fn find_recent_batch_never_mixes_persisted_spaces() {
        let (db, _dir) = test_db().await;
        db.create_space("work", None, false).await.unwrap();
        db.create_space("personal", None, false).await.unwrap();

        let agent = "space-partition-test";
        let mut work_a = make_doc("mem_space_work_a", "Work Space batch item A.");
        work_a.source_agent = Some(agent.to_string());
        work_a.space = Some("work".to_string());
        let mut work_b = make_doc("mem_space_work_b", "Work Space batch item B.");
        work_b.source_agent = Some(agent.to_string());
        work_b.space = Some("work".to_string());
        let mut personal = make_doc("mem_space_personal", "Personal Space batch item.");
        personal.source_agent = Some(agent.to_string());
        personal.space = Some("personal".to_string());
        let mut uncategorized =
            make_doc("mem_space_uncategorized", "Uncategorized Space batch item.");
        uncategorized.source_agent = Some(agent.to_string());

        db.upsert_documents(vec![work_a, work_b, personal, uncategorized])
            .await
            .unwrap();

        let work_batch = db
            .find_recent_batch(agent, 600, Some("work"))
            .await
            .unwrap();
        let work_ids: Vec<&str> = work_batch.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            work_ids,
            vec!["mem_space_work_a", "mem_space_work_b"],
            "derived extraction must stay inside the persisted work Space"
        );

        let uncategorized_batch = db.find_recent_batch(agent, 600, None).await.unwrap();
        let uncategorized_ids: Vec<&str> = uncategorized_batch
            .iter()
            .map(|(id, _)| id.as_str())
            .collect();
        assert_eq!(uncategorized_ids, vec!["mem_space_uncategorized"]);
    }

    // ---- M3g Important-1: batch span base must be the anchor's own content ----

    /// A CannedLlmProvider that always returns one Alice-works_on-Acme
    /// relation carrying `quote` as its span, keyed on the
    /// extract_knowledge_graph prompt (same pattern as `canned_alice` in
    /// `kg/entity_extraction.rs`).
    fn canned_relation_with_span(
        quote: &str,
    ) -> (
        crate::prompts::PromptRegistry,
        std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
    ) {
        use crate::llm_provider::CannedLlmProvider;
        let prompts = crate::prompts::PromptRegistry::default();
        let key_fragment: String = prompts.extract_knowledge_graph.chars().take(30).collect();
        let kg_json = serde_json::json!([{
            "entities": [{"name": "Alice", "type": "person"}, {"name": "Acme", "type": "org"}],
            "observations": [],
            "relations": [{"from": "Alice", "to": "Acme", "type": "works_on", "span": quote}]
        }])
        .to_string();
        let canned: std::sync::Arc<dyn crate::llm_provider::LlmProvider> =
            std::sync::Arc::new(CannedLlmProvider::new("DEFAULT").with(key_fragment, kg_json));
        (prompts, canned)
    }

    /// Fetch the `relates` edge payload between two entities by name.
    async fn relates_payload_between(
        db: &MemoryDB,
        from_name: &str,
        to_name: &str,
    ) -> Option<String> {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT e.payload FROM edges e \
                 JOIN entities fe ON e.src_id = fe.id \
                 JOIN entities te ON e.dst_id = te.id \
                 WHERE e.edge_type = 'relates' AND fe.name = ?1 AND te.name = ?2",
                libsql::params![from_name, to_name],
            )
            .await
            .unwrap();
        rows.next()
            .await
            .unwrap()
            .and_then(|row| row.get::<Option<String>>(0).unwrap_or(None))
    }

    /// A quote that is real text from the combined batch blob but belongs to
    /// the non-anchor window member must be dropped entirely, never located
    /// against the blob and never left as a null-offset placeholder.
    #[tokio::test]
    async fn batch_extraction_span_not_in_anchor_is_dropped() {
        let (db, _dir) = test_db().await;
        let anchor_content = "Alice joined Acme in 2020.";
        let other_content = "Bob left Beta Corp last year.";
        db.upsert_documents(vec![
            make_doc("mem_batch_anchor", anchor_content),
            make_doc("mem_batch_other", other_content),
        ])
        .await
        .unwrap();

        let (prompts, canned) = canned_relation_with_span("Bob left Beta Corp");

        run_post_ingest_enrichment(
            &db,
            "mem_batch_anchor",
            anchor_content,
            None,
            Some("fact"),
            None,
            None,
            Some(&canned),
            &prompts,
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let payload = relates_payload_between(&db, "Alice", "Acme").await.expect(
            "payload must be written -- model_version/prompt_version are always Some on this path",
        );
        let payload: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert!(
            payload["span"].is_null(),
            "a quote that only locates in a non-anchor batch member must be dropped entirely, got {payload}"
        );
    }

    /// A quote that is genuinely the anchor's own text must locate with
    /// correct anchor-based char offsets, exactly as the single-memory path
    /// already does.
    #[tokio::test]
    async fn batch_extraction_span_in_anchor_locates_with_anchor_offsets() {
        let (db, _dir) = test_db().await;
        let anchor_content = "Alice joined Acme in 2020.";
        let other_content = "Bob left Beta Corp last year.";
        db.upsert_documents(vec![
            make_doc("mem_batch_anchor", anchor_content),
            make_doc("mem_batch_other", other_content),
        ])
        .await
        .unwrap();

        let quote = "Alice joined Acme";
        let (prompts, canned) = canned_relation_with_span(quote);

        run_post_ingest_enrichment(
            &db,
            "mem_batch_anchor",
            anchor_content,
            None,
            Some("fact"),
            None,
            None,
            Some(&canned),
            &prompts,
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let payload = relates_payload_between(&db, "Alice", "Acme")
            .await
            .expect("payload must be written");
        let payload: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(payload["span"]["quote"], serde_json::json!(quote));
        assert_eq!(payload["span"]["char_start"], serde_json::json!(0));
        assert_eq!(
            payload["span"]["char_end"],
            serde_json::json!(quote.chars().count())
        );
    }

    // ---- T22: cooperative-cancellation (debounced reflection) ----

    /// Default-OFF guarantee, expressed at the core boundary: passing a cancel
    /// flag that is `false` must be inert — every enrichment step still runs.
    #[tokio::test]
    async fn test_enrichment_runs_all_steps_when_not_cancelled() {
        let (db, _dir) = test_db().await;
        let doc = make_doc("mem_t22_inert", "The capital of France is Paris");
        db.upsert_documents(vec![doc]).await.unwrap();

        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        run_post_ingest_enrichment(
            &db,
            "mem_t22_inert",
            "The capital of France is Paris",
            None,
            Some("fact"),
            None,
            None,
            None,
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            Some(cancel.as_ref()),
            None, // precomputed_kg
        )
        .await
        .unwrap();

        let steps = db.get_enrichment_steps("mem_t22_inert").await.unwrap();
        let names: std::collections::HashSet<&str> =
            steps.iter().map(|s| s.step.as_str()).collect();
        // Same step coverage as the no-cancel path (test_enrichment_records_per_step_outcomes).
        assert!(
            names.contains("entity_link"),
            "entity_link must run when cancel=false"
        );
        assert!(
            !names.contains("entity_extract"),
            "cancel=false must not turn a missing provider into a terminal entity result"
        );
        assert!(
            names.contains("title_enrich"),
            "title_enrich must run when cancel=false"
        );
        assert!(
            names.contains("page_growth"),
            "page_growth must run when cancel=false"
        );
    }

    /// Cancelled before the first step → return Ok(()) with NO steps written.
    #[tokio::test]
    async fn test_enrichment_early_returns_when_cancelled_before_first_step() {
        let (db, _dir) = test_db().await;
        let doc = make_doc("mem_t22_precancel", "The capital of France is Paris");
        db.upsert_documents(vec![doc]).await.unwrap();

        let cancel = std::sync::Arc::new(AtomicBool::new(true));
        run_post_ingest_enrichment(
            &db,
            "mem_t22_precancel",
            "The capital of France is Paris",
            None,
            Some("fact"),
            None,
            None,
            None,
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            Some(cancel.as_ref()),
            None, // precomputed_kg
        )
        .await
        .unwrap();

        let steps = db.get_enrichment_steps("mem_t22_precancel").await.unwrap();
        assert!(
            steps.is_empty(),
            "no enrichment steps must be written when cancelled before the first step, got {steps:?}"
        );
    }

    /// Cancel concurrently with enrichment: whatever steps did run must form a
    /// clean contiguous PREFIX of the canonical step sequence, never a step that
    /// was half-applied or a later step that ran after an earlier one was
    /// skipped. This proves the `is_cancelled` checkpoints cut only at clean
    /// boundaries between whole steps (combined with the cancelled-before-first
    /// test, which is the empty-prefix case, and the not-cancelled test, which
    /// is the full-prefix case).
    ///
    /// The exact cut point depends on the scheduler race (the no-LLM path is
    /// fast), so we assert the invariant that must hold for EVERY cut point
    /// rather than a fixed one. Run on a multi-thread runtime so the flipper and
    /// the enrichment make progress in parallel.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_enrichment_cancel_midway_preserves_committed_steps() {
        // Canonical order recorded by the no-LLM path. Entity extraction is
        // deliberately absent: it remains backlog until a provider exists.
        const CANON: [&str; 3] = ["entity_link", "title_enrich", "page_growth"];

        let (db, _dir) = test_db().await;
        let doc = make_doc("mem_t22_midway", "The capital of France is Paris");
        db.upsert_documents(vec![doc]).await.unwrap();
        let db = std::sync::Arc::new(db);

        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let cancel_task = cancel.clone();
        let db_task = db.clone();
        let handle = tokio::spawn(async move {
            run_post_ingest_enrichment(
                &db_task,
                "mem_t22_midway",
                "The capital of France is Paris",
                None,
                Some("fact"),
                None,
                None,
                None,
                &crate::prompts::PromptRegistry::default(),
                &crate::tuning::RefineryConfig::default(),
                &crate::tuning::DistillationConfig::default(),
                None,
                Some(cancel_task.as_ref()),
                None, // precomputed_kg
            )
            .await
            .unwrap();
        });

        // Flip cancel the moment the first step is observed, then let the rest
        // settle. Bounded spin so a logic bug can't hang the test.
        for _ in 0..2000 {
            let steps = db.get_enrichment_steps("mem_t22_midway").await.unwrap();
            if steps.iter().any(|s| s.step == "entity_link") {
                cancel.store(true, Ordering::SeqCst);
                break;
            }
            tokio::task::yield_now().await;
        }
        handle.await.unwrap();

        let steps = db.get_enrichment_steps("mem_t22_midway").await.unwrap();
        let names: std::collections::HashSet<&str> =
            steps.iter().map(|s| s.step.as_str()).collect();

        // Invariant 1: the recorded steps are exactly the first K of CANON for
        // some K — a contiguous prefix. No later step ever ran while an earlier
        // one was skipped (that would mean a checkpoint failed to cut cleanly).
        let k = CANON.iter().take_while(|s| names.contains(**s)).count();
        for (i, step) in CANON.iter().enumerate() {
            let present = names.contains(*step);
            let expected = i < k;
            assert_eq!(
                present, expected,
                "step {step:?} present={present} but prefix length is {k}: recorded steps must be a contiguous prefix of the canonical order, got {names:?}"
            );
        }
        // Invariant 2: every recorded step is complete (has a non-empty status),
        // i.e. no step was half-written when cancellation hit.
        for st in &steps {
            assert!(
                !st.status.is_empty(),
                "step {:?} was recorded without a status — a half-written step",
                st.step
            );
        }
    }

    /// The memory row stored synchronously before enrichment must remain
    /// retrievable after a cancelled enrichment — cancellation only delays
    /// enrichment, it never drops data.
    #[tokio::test]
    async fn test_memory_row_intact_after_cancel() {
        let (db, _dir) = test_db().await;
        let doc = make_doc("mem_t22_rowintact", "Tokyo is the capital of Japan");
        db.upsert_documents(vec![doc]).await.unwrap();

        let cancel = std::sync::Arc::new(AtomicBool::new(true));
        run_post_ingest_enrichment(
            &db,
            "mem_t22_rowintact",
            "Tokyo is the capital of Japan",
            None,
            Some("fact"),
            None,
            None,
            None,
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            Some(cancel.as_ref()),
            None, // precomputed_kg
        )
        .await
        .unwrap();

        // Row still present and retrievable despite the cancelled enrichment.
        let items = db.list_memories(None, None, None, None, 10).await.unwrap();
        let row = items.iter().find(|i| i.source_id == "mem_t22_rowintact");
        assert!(
            row.is_some(),
            "the stored memory row must survive a cancelled enrichment"
        );
    }

    #[tokio::test]
    async fn test_enrichment_honesty_end_to_end() {
        let (db, _dir) = test_db().await;

        // Store memory A -- full enrichment (no LLM, so LLM steps get skipped)
        let doc_a = make_doc("mem_honest_a", "The Eiffel Tower is in Paris");
        db.upsert_documents(vec![doc_a]).await.unwrap();
        run_post_ingest_enrichment(
            &db,
            "mem_honest_a",
            "The Eiffel Tower is in Paris",
            None,
            Some("fact"),
            None,
            None,
            None,
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            None, // cancel — T22, inert
            None, // precomputed_kg
        )
        .await
        .unwrap();

        // Summary should be enriched (all non-LLM steps ok, LLM steps skipped)
        let summary_a = db.get_enrichment_summary("mem_honest_a").await.unwrap();
        assert_eq!(summary_a, "enriched");

        // list_memories should show enriched
        let items = db.list_memories(None, None, None, None, 10).await.unwrap();
        let item_a = items
            .iter()
            .find(|i| i.source_id == "mem_honest_a")
            .unwrap();
        assert_eq!(item_a.enrichment_status, "enriched");

        // Store memory B -- no enrichment run yet
        let doc_b = make_doc("mem_honest_b", "Tokyo is the capital of Japan");
        db.upsert_documents(vec![doc_b]).await.unwrap();

        // Should be raw (no steps recorded)
        let summary_b = db.get_enrichment_summary("mem_honest_b").await.unwrap();
        assert_eq!(summary_b, "raw");

        let items = db.list_memories(None, None, None, None, 10).await.unwrap();
        let item_b = items
            .iter()
            .find(|i| i.source_id == "mem_honest_b")
            .unwrap();
        assert_eq!(item_b.enrichment_status, "raw");
    }

    /// TDD gate for Task 1.2 — precomputed_kg parameter.
    ///
    /// DB-A: run_post_ingest_enrichment with precomputed_kg = None (canned LLM called for extract).
    /// DB-B: run_post_ingest_enrichment with precomputed_kg = Some(kg) (precomputed KG passed in;
    ///        LLM must NOT be called for extract on this arm).
    ///
    /// Asserts: both link source_id to an entity with the same name, and both record
    /// entity_extract = "ok".
    #[tokio::test]
    async fn precomputed_kg_matches_inline_extract() {
        use crate::llm_provider::{CannedLlmProvider, LlmProvider};
        use std::sync::Arc;

        let prompts = crate::prompts::PromptRegistry::default();

        // Build a CannedLlmProvider that returns a KG with "Alice" as entity.
        // Key = first 30 chars of the extract_knowledge_graph prompt (matches the `sys.contains(key)` check).
        let key_fragment: String = prompts.extract_knowledge_graph.chars().take(30).collect();
        let kg_json =
            r#"[{"entities":[{"name":"Alice","type":"person"}],"observations":[],"relations":[]}]"#;
        let canned: Arc<dyn LlmProvider> =
            Arc::new(CannedLlmProvider::new("DEFAULT").with(key_fragment.clone(), kg_json));

        // Pre-compute the KG using extract_kg (pure, no DB).
        let precomputed = crate::refinery::extract_kg(&canned, &prompts, "Alice joined Acme")
            .await
            .expect("extract_kg failed in test setup");
        assert!(
            !precomputed.is_empty(),
            "precomputed KG must be non-empty for a useful test"
        );

        // ----- DB-A: None arm (inline LLM extract) -----
        let (db_a, _dir_a) = test_db().await;
        let doc_a = make_doc("mem_pkg_a", "Alice joined Acme");
        db_a.upsert_documents(vec![doc_a]).await.unwrap();
        let canned_a: Arc<dyn LlmProvider> =
            Arc::new(CannedLlmProvider::new("DEFAULT").with(key_fragment.clone(), kg_json));
        run_post_ingest_enrichment(
            &db_a,
            "mem_pkg_a",
            "Alice joined Acme",
            None,
            Some("fact"),
            None,
            None,
            Some(&canned_a),
            &prompts,
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            None,
            None, // precomputed_kg = None
        )
        .await
        .unwrap();

        // ----- DB-B: Some(kg) arm (precomputed KG, LLM should NOT be called for extract) -----
        let (db_b, _dir_b) = test_db().await;
        let doc_b = make_doc("mem_pkg_b", "Alice joined Acme");
        db_b.upsert_documents(vec![doc_b]).await.unwrap();
        // Use a CannedLlmProvider that returns DEFAULT (no entities) — if extract_kg is mistakenly
        // called on the B arm, it would extract nothing and the test would fail.
        let canned_b_no_extract: Arc<dyn LlmProvider> = Arc::new(CannedLlmProvider::new("[]"));
        run_post_ingest_enrichment(
            &db_b,
            "mem_pkg_b",
            "Alice joined Acme",
            None,
            Some("fact"),
            None,
            None,
            Some(&canned_b_no_extract),
            &prompts,
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            None,
            Some(precomputed), // precomputed_kg = Some(kg)
        )
        .await
        .unwrap();

        // Both arms must record entity_extract = "ok"
        let steps_a = db_a.get_enrichment_steps("mem_pkg_a").await.unwrap();
        let extract_a = steps_a
            .iter()
            .find(|s| s.step == "entity_extract")
            .expect("entity_extract step missing on arm A");
        assert_eq!(
            extract_a.status, "ok",
            "arm A entity_extract status must be ok"
        );

        let steps_b = db_b.get_enrichment_steps("mem_pkg_b").await.unwrap();
        let extract_b = steps_b
            .iter()
            .find(|s| s.step == "entity_extract")
            .expect("entity_extract step missing on arm B");
        assert_eq!(
            extract_b.status, "ok",
            "arm B entity_extract status must be ok"
        );

        // Both arms must link source_id to an entity, and that entity must be "Alice"
        let eid_a = db_a
            .get_memory_entity_id("mem_pkg_a")
            .await
            .unwrap()
            .expect("arm A must have an entity_id");
        let eid_b = db_b
            .get_memory_entity_id("mem_pkg_b")
            .await
            .unwrap()
            .expect("arm B must have an entity_id");

        // Verify entity name for each arm
        let (name_a, _) = db_a
            .get_entity_name_type(&eid_a)
            .await
            .unwrap()
            .expect("entity from arm A not found");
        assert_eq!(
            name_a, "Alice",
            "arm A entity name must be Alice, got: {name_a}"
        );

        let (name_b, _) = db_b
            .get_entity_name_type(&eid_b)
            .await
            .unwrap()
            .expect("entity from arm B not found");
        assert_eq!(
            name_b, "Alice",
            "arm B entity name must be Alice, got: {name_b}"
        );
    }

    // ── grow_page (Task 6: citation-numbered growth path) ──────────────────

    /// Stub LLM provider that captures the exact prompt it received, so
    /// tests can assert on input hygiene (e.g. that a stale marker never
    /// reached the LLM) as well as the fixed output it returns.
    struct CapturingStubProvider {
        response: String,
        captured_prompt: std::sync::Mutex<Option<String>>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for CapturingStubProvider {
        async fn generate(
            &self,
            request: crate::llm_provider::LlmRequest,
        ) -> Result<String, crate::llm_provider::LlmError> {
            *self.captured_prompt.lock().unwrap() = Some(request.user_prompt.clone());
            Ok(self.response.clone())
        }
        fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "capturing-stub"
        }
        fn backend(&self) -> crate::llm_provider::LlmBackend {
            crate::llm_provider::LlmBackend::OnDevice
        }
    }

    enum GrowthMutation {
        None,
        Memory { source_id: String },
        Page { page_id: String, source_id: String },
    }

    struct MutatingGrowthProvider {
        db: Arc<MemoryDB>,
        response: String,
        mutation: GrowthMutation,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl LlmProvider for MutatingGrowthProvider {
        async fn generate(
            &self,
            _request: crate::llm_provider::LlmRequest,
        ) -> Result<String, crate::llm_provider::LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.mutation {
                GrowthMutation::None => {}
                GrowthMutation::Memory { source_id } => {
                    self.db
                        .apply_memory_update(
                            source_id,
                            Some("newer memory content won the race"),
                            None,
                            false,
                            None,
                            None,
                        )
                        .await
                        .unwrap();
                }
                GrowthMutation::Page { page_id, source_id } => {
                    let page = self.db.get_page(page_id).await.unwrap().unwrap();
                    assert!(self
                        .db
                        .try_update_page_content_with_changelog_at_version(
                            page_id,
                            "human edit won the race",
                            &[source_id.as_str()],
                            "fs_edit",
                            "[]",
                            None,
                            page.version,
                            None,
                        )
                        .await
                        .unwrap());
                }
            }
            Ok(self.response.clone())
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "mutating-growth"
        }

        fn backend(&self) -> crate::llm_provider::LlmBackend {
            crate::llm_provider::LlmBackend::OnDevice
        }
    }

    async fn seed_page_growth_memory(
        db: &MemoryDB,
        source_id: &str,
        content: &str,
        entity_id: Option<&str>,
        space: Option<&str>,
    ) {
        let mut doc = make_doc(source_id, content);
        doc.entity_id = entity_id.map(str::to_string);
        doc.space = space.map(str::to_string);
        db.upsert_documents(vec![doc]).await.unwrap();
        assert!(db
            .record_enrichment_step_at_version(source_id, "entity_extract", "ok", None, 1,)
            .await
            .unwrap());
    }

    async fn insert_growth_page(
        db: &MemoryDB,
        id: &str,
        entity_id: &str,
        scope: &str,
        content: &str,
    ) {
        db.insert_page_with_kind(
            id,
            id,
            None,
            content,
            Some(entity_id),
            Some(scope),
            &[],
            &chrono::Utc::now().to_rfc3339(),
            "research",
            "confirmed",
            Some(scope),
            None,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn page_growth_slice_commits_one_current_result() {
        let (db, _dir) = test_db().await;
        let db = Arc::new(db);
        let entity_id = db
            .create_entity("Ambient Growth", "Topic", Some("work"))
            .await
            .unwrap();
        insert_growth_page(
            &db,
            "ambient-growth-page",
            &entity_id,
            "work",
            "existing machine-owned page body",
        )
        .await;
        seed_page_growth_memory(
            &db,
            "ambient-growth-memory",
            "new evidence for the ambient page",
            Some(&entity_id),
            Some("work"),
        )
        .await;
        let provider = Arc::new(MutatingGrowthProvider {
            db: db.clone(),
            response: "existing machine-owned page body. New evidence.[1]".to_string(),
            mutation: GrowthMutation::None,
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let report = run_page_growth_slice(&db, &llm, &PromptRegistry::default(), 2.0, None)
            .await
            .unwrap();

        assert!(report.selected);
        assert!(report.matched);
        assert!(report.committed);
        assert_eq!(report.llm_calls, 1);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        let page = db.get_page("ambient-growth-page").await.unwrap().unwrap();
        assert!(page
            .source_memory_ids
            .contains(&"ambient-growth-memory".to_string()));
        let receipt = db
            .get_enrichment_steps("ambient-growth-memory")
            .await
            .unwrap()
            .into_iter()
            .find(|step| step.step == "page_growth")
            .expect("page growth receipt");
        assert_eq!(receipt.status, "ok");
        assert_eq!(receipt.input_version, Some(1));
    }

    #[tokio::test]
    async fn page_growth_slice_no_match_is_terminal_without_inference() {
        let (db, _dir) = test_db().await;
        let db = Arc::new(db);
        seed_page_growth_memory(
            &db,
            "ambient-growth-no-match",
            "evidence with no matching page",
            None,
            Some("work"),
        )
        .await;
        let provider = Arc::new(MutatingGrowthProvider {
            db: db.clone(),
            response: "must never be used".to_string(),
            mutation: GrowthMutation::None,
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let first = run_page_growth_slice(&db, &llm, &PromptRegistry::default(), 0.85, None)
            .await
            .unwrap();
        let second = run_page_growth_slice(&db, &llm, &PromptRegistry::default(), 0.85, None)
            .await
            .unwrap();

        assert!(first.selected);
        assert!(!first.matched);
        assert!(first.committed);
        assert!(
            first.terminal_no_match,
            "the scheduler needs an explicit signal before exempting this measured outcome"
        );
        assert_eq!(first.llm_calls, 0);
        assert!(
            !second.selected,
            "terminal no-match receipt must stop re-spins"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn page_growth_slice_rejects_memory_changed_during_inference() {
        let (db, _dir) = test_db().await;
        let db = Arc::new(db);
        let entity_id = db
            .create_entity("Memory CAS", "Topic", Some("work"))
            .await
            .unwrap();
        insert_growth_page(
            &db,
            "memory-cas-page",
            &entity_id,
            "work",
            "page before memory race",
        )
        .await;
        seed_page_growth_memory(
            &db,
            "memory-cas-source",
            "memory before race",
            Some(&entity_id),
            Some("work"),
        )
        .await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MutatingGrowthProvider {
            db: db.clone(),
            response: "stale generated page.[1]".to_string(),
            mutation: GrowthMutation::Memory {
                source_id: "memory-cas-source".to_string(),
            },
            calls: std::sync::atomic::AtomicUsize::new(0),
        });

        let report = run_page_growth_slice(&db, &llm, &PromptRegistry::default(), 2.0, None)
            .await
            .unwrap();

        assert!(report.selected);
        assert!(report.matched);
        assert!(!report.committed);
        assert_eq!(report.llm_calls, 1);
        let page = db.get_page("memory-cas-page").await.unwrap().unwrap();
        assert_eq!(page.content, "page before memory race");
        assert!(db
            .get_enrichment_steps("memory-cas-source")
            .await
            .unwrap()
            .iter()
            .all(|step| step.step != "page_growth"));

        assert!(
            db.record_enrichment_step_at_version(
                "memory-cas-source",
                "entity_extract",
                "ok",
                None,
                2,
            )
            .await
            .unwrap()
        );
        assert!(db
            .get_page_growth_candidate(3, true)
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn page_growth_slice_rejects_linked_source_changed_during_inference() {
        let (db, _dir) = test_db().await;
        let db = Arc::new(db);
        let entity_id = db
            .create_entity("Linked Source CAS", "Topic", Some("work"))
            .await
            .unwrap();
        insert_growth_page(
            &db,
            "linked-source-cas-page",
            &entity_id,
            "work",
            "page before linked source race",
        )
        .await;
        seed_page_growth_memory(
            &db,
            "linked-source-existing",
            "existing linked evidence before race",
            Some(&entity_id),
            Some("work"),
        )
        .await;
        db.link_page_source(
            "linked-source-cas-page",
            "linked-source-existing",
            "test_existing_source",
        )
        .await
        .unwrap();
        seed_page_growth_memory(
            &db,
            "linked-source-trigger",
            "new evidence that triggers page growth",
            Some(&entity_id),
            Some("work"),
        )
        .await;
        let page_version_before = db
            .get_page("linked-source-cas-page")
            .await
            .unwrap()
            .unwrap()
            .version;
        let source_revision_before = db
            .get_page_source_revision("linked-source-cas-page")
            .await
            .unwrap();
        let llm: Arc<dyn LlmProvider> = Arc::new(MutatingGrowthProvider {
            db: db.clone(),
            response: "stale generated page.[1][2]".to_string(),
            mutation: GrowthMutation::Memory {
                source_id: "linked-source-existing".to_string(),
            },
            calls: std::sync::atomic::AtomicUsize::new(0),
        });

        let report = run_page_growth_slice(&db, &llm, &PromptRegistry::default(), 2.0, None)
            .await
            .unwrap();

        assert!(report.selected);
        assert!(report.matched);
        assert!(!report.committed);
        assert_eq!(report.llm_calls, 1);
        let page = db
            .get_page("linked-source-cas-page")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(page.content, "page before linked source race");
        assert_eq!(
            page.version, page_version_before,
            "linked source invalidation must not need a Page content-version bump"
        );
        assert_eq!(
            db.get_page_source_revision("linked-source-cas-page")
                .await
                .unwrap(),
            source_revision_before + 1
        );
        assert!(db
            .get_enrichment_steps("linked-source-trigger")
            .await
            .unwrap()
            .iter()
            .all(|step| step.step != "page_growth"));
    }

    #[tokio::test]
    async fn page_growth_slice_rejects_page_changed_during_inference() {
        let (db, _dir) = test_db().await;
        let db = Arc::new(db);
        let entity_id = db
            .create_entity("Page CAS", "Topic", Some("work"))
            .await
            .unwrap();
        insert_growth_page(
            &db,
            "page-cas-page",
            &entity_id,
            "work",
            "page before human race",
        )
        .await;
        seed_page_growth_memory(
            &db,
            "page-cas-source",
            "memory for page race",
            Some(&entity_id),
            Some("work"),
        )
        .await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MutatingGrowthProvider {
            db: db.clone(),
            response: "stale generated page.[1]".to_string(),
            mutation: GrowthMutation::Page {
                page_id: "page-cas-page".to_string(),
                source_id: "page-cas-source".to_string(),
            },
            calls: std::sync::atomic::AtomicUsize::new(0),
        });

        let report = run_page_growth_slice(&db, &llm, &PromptRegistry::default(), 2.0, None)
            .await
            .unwrap();

        assert!(report.selected);
        assert!(report.matched);
        assert!(!report.committed);
        assert_eq!(report.llm_calls, 1);
        let page = db.get_page("page-cas-page").await.unwrap().unwrap();
        assert_eq!(page.content, "human edit won the race");
        assert!(page.user_edited);
        assert!(db
            .get_enrichment_steps("page-cas-source")
            .await
            .unwrap()
            .iter()
            .all(|step| step.step != "page_growth"));
    }

    #[tokio::test]
    async fn page_growth_mutates_only_the_memory_scope() {
        let (db, _dir) = test_db().await;
        let entity_id = db
            .create_entity("Scoped Growth Entity", "Topic", Some("work"))
            .await
            .unwrap();
        insert_growth_page(
            &db,
            "growth-personal",
            &entity_id,
            "personal",
            "personal page must remain unchanged",
        )
        .await;
        insert_growth_page(
            &db,
            "growth-work",
            &entity_id,
            "work",
            "work page accepts the new evidence",
        )
        .await;

        let source_id = "growth-scope-memory";
        let content = "new scoped evidence belongs only to the work page";
        let mut doc = make_doc(source_id, content);
        doc.space = Some("work".to_string());
        doc.entity_id = Some(entity_id.clone());
        db.upsert_documents(vec![doc]).await.unwrap();

        let stub: Arc<dyn LlmProvider> = Arc::new(CapturingStubProvider {
            response: "work page accepts the new evidence.[1]".to_string(),
            captured_prompt: std::sync::Mutex::new(None),
        });
        // Force the entity-first path. A global entity lookup would see the
        // personal Page first and then fail the scoped match.
        let distillation = crate::tuning::DistillationConfig {
            page_growth_threshold: 2.0,
            ..Default::default()
        };
        run_post_ingest_enrichment(
            &db,
            source_id,
            content,
            Some(&entity_id),
            Some("fact"),
            Some("work"),
            None,
            Some(&stub),
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &distillation,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        let personal = db.get_page("growth-personal").await.unwrap().unwrap();
        let work = db.get_page("growth-work").await.unwrap().unwrap();
        assert!(
            !personal.source_memory_ids.contains(&source_id.to_string()),
            "cross-scope Page must not be grown"
        );
        assert!(
            work.source_memory_ids.contains(&source_id.to_string()),
            "same-scope Page must receive the new source"
        );
    }

    #[tokio::test]
    async fn page_growth_dependency_waits_for_enabled_entity_lane_only() {
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![make_doc(
            "growth-dependency",
            "memory waiting on an enabled entity lane",
        )])
        .await
        .unwrap();

        assert!(db
            .get_page_growth_candidate(3, true)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            db.get_page_growth_candidate(3, false)
                .await
                .unwrap()
                .map(|input| input.source_id),
            Some("growth-dependency".to_string()),
        );
    }

    #[tokio::test]
    async fn page_growth_uses_entity_linked_during_current_enrichment() {
        let (db, _dir) = test_db().await;
        let entity_name = "Current Run Entity";
        let entity_id = db
            .create_entity(entity_name, "Topic", Some("work"))
            .await
            .unwrap();
        insert_growth_page(
            &db,
            "growth-current-entity",
            &entity_id,
            "work",
            "page linked to the entity extracted in this run",
        )
        .await;

        let source_id = "growth-current-entity-memory";
        let content = "this memory is linked to an entity during enrichment";
        let mut doc = make_doc(source_id, content);
        doc.space = Some("work".to_string());
        db.upsert_documents(vec![doc]).await.unwrap();

        let precomputed_kg = vec![crate::extract::KgExtractionResult {
            index: 0,
            entities: vec![crate::extract::ExtractedEntity {
                name: entity_name.to_string(),
                entity_type: "Topic".to_string(),
            }],
            observations: Vec::new(),
            relations: Vec::new(),
        }];
        let stub: Arc<dyn LlmProvider> = Arc::new(CapturingStubProvider {
            response: "page linked to the entity extracted in this run.[1]".to_string(),
            captured_prompt: std::sync::Mutex::new(None),
        });
        let tuning = crate::tuning::RefineryConfig {
            entity_link_distance: -1.0,
            ..Default::default()
        };
        let distillation = crate::tuning::DistillationConfig {
            page_growth_threshold: 2.0,
            ..Default::default()
        };

        run_post_ingest_enrichment(
            &db,
            source_id,
            content,
            None,
            Some("fact"),
            Some("work"),
            None,
            Some(&stub),
            &crate::prompts::PromptRegistry::default(),
            &tuning,
            &distillation,
            None,
            None,
            Some(precomputed_kg),
        )
        .await
        .unwrap();

        assert_eq!(
            db.get_memory_entity_id(source_id).await.unwrap().as_deref(),
            Some(entity_id.as_str()),
            "fixture must link the entity during this enrichment run"
        );
        let page = db.get_page("growth-current-entity").await.unwrap().unwrap();
        assert!(
            page.source_memory_ids.contains(&source_id.to_string()),
            "Page growth must use the entity linked after extraction"
        );
    }

    #[tokio::test]
    async fn grow_page_strips_stale_markers_and_persists_citations() {
        let (db, _dir) = test_db().await;

        // Seed the source memory behind the existing page, then create the
        // page with a STALE [3] marker in its content (pointing at a
        // numbered list that no longer exists).
        let mem_v1 = "mem_grow_v1";
        let v1_content = "Rust is a systems programming language with memory safety guarantees";
        db.upsert_documents(vec![make_doc(mem_v1, v1_content)])
            .await
            .unwrap();

        let page_req = wenlan_types::requests::CreateConceptRequest {
            title: "Rust".to_string(),
            content: format!("{v1_content}.[3]"),
            summary: None,
            entity_id: None,
            space: None.into(),
            source_memory_ids: vec![mem_v1.to_string()],
            // Machine-owned kind on purpose: this test exercises the in-place
            // grow + citation-persistence path. An `authored` (human-owned) page
            // would be intercepted by the PageWrite ownership gate (spec §5.2)
            // into a revision card and never grown in place. `research` is
            // floor-exempt (single source ok) and not human-owned.
            creation_kind: Some("research".to_string()),
            workspace: None,
        };
        let page_id = crate::post_write::create_page(&db, page_req, "test", None)
            .await
            .unwrap()
            .id;
        db.set_page_review_status(&page_id, "confirmed")
            .await
            .unwrap();

        // Seed the NEW memory that triggers growth.
        let mem_new = "mem_grow_new";
        let new_content =
            "Rust also provides zero-cost abstractions for high-performance systems programming";
        db.upsert_documents(vec![make_doc(mem_new, new_content)])
            .await
            .unwrap();

        let stub = std::sync::Arc::new(CapturingStubProvider {
            response: format!(
                "{v1_content}.[1] It also provides zero-cost abstractions for high-performance systems programming.[2]"
            ),
            captured_prompt: std::sync::Mutex::new(None),
        });
        let llm: std::sync::Arc<dyn LlmProvider> = stub.clone();

        let grew = grow_page(
            &db,
            mem_new,
            new_content,
            None,
            None,
            Some(&llm),
            &crate::prompts::PromptRegistry::default(),
            0.0,
        )
        .await
        .unwrap();
        assert!(grew, "grow_page should report growth");

        // Input hygiene: the stale [3] marker must never reach the prompt.
        let captured = stub.captured_prompt.lock().unwrap().clone().unwrap();
        assert!(
            !captured.contains("[3]"),
            "stale marker leaked into prompt: {captured}"
        );

        // Output: page.citations populated, changelog carries a citations_summary.
        let page = db.get_page(&page_id).await.unwrap().unwrap();
        assert!(!page.citations.is_empty(), "citations should be populated");

        let log: Vec<wenlan_types::responses::PageChangelogEntry> =
            serde_json::from_str(&db.get_page_changelog(&page_id).await.unwrap()).unwrap();
        assert!(log.last().unwrap().citations_summary.is_some());
    }

    /// Spec §5.1 (True source kinds): `grow_page`'s numbered-source list must
    /// resolve the triggering memory's typed kind instead of hardcoding
    /// `'memory'`, else a folder-doc source that grows an existing page is
    /// mis-badged as a plain memory citation.
    #[tokio::test]
    async fn grow_page_resolves_external_file_kind_for_folder_doc_trigger() {
        let (db, _dir) = test_db().await;

        let mem_v1 = "mem_grow_kind_v1";
        let v1_content = "Rust is a systems programming language with memory safety guarantees";
        db.upsert_documents(vec![make_doc(mem_v1, v1_content)])
            .await
            .unwrap();

        let page_req = wenlan_types::requests::CreateConceptRequest {
            title: "Rust".to_string(),
            content: v1_content.to_string(),
            summary: None,
            entity_id: None,
            space: None.into(),
            source_memory_ids: vec![mem_v1.to_string()],
            creation_kind: Some("research".to_string()),
            workspace: None,
        };
        let page_id = crate::post_write::create_page(&db, page_req, "test", None)
            .await
            .unwrap()
            .id;
        db.set_page_review_status(&page_id, "confirmed")
            .await
            .unwrap();

        // The NEW triggering source is a folder document: source_agent='folder'
        // + the `{source_id}::{provenance}` id shape stamped by
        // `sources::directory::document_source_id` (directory.rs:372).
        let mem_new = "folder-notes::rust/new.md";
        let new_content =
            "Rust also provides zero-cost abstractions for high-performance systems programming";
        let mut folder_doc = make_doc(mem_new, new_content);
        folder_doc.source_agent = Some("folder".to_string());
        db.upsert_documents(vec![folder_doc]).await.unwrap();

        let stub = std::sync::Arc::new(CapturingStubProvider {
            response: format!(
                "{v1_content}.[1] It also provides zero-cost abstractions for high-performance systems programming.[2]"
            ),
            captured_prompt: std::sync::Mutex::new(None),
        });
        let llm: std::sync::Arc<dyn LlmProvider> = stub.clone();

        let grew = grow_page(
            &db,
            mem_new,
            new_content,
            None,
            None,
            Some(&llm),
            &crate::prompts::PromptRegistry::default(),
            0.0,
        )
        .await
        .unwrap();
        assert!(grew, "grow_page should report growth");

        let page = db.get_page(&page_id).await.unwrap().unwrap();
        let kind_of = page
            .citations
            .iter()
            .find(|c| c.locator == mem_new)
            .map(|c| c.source_kind.clone());
        assert_eq!(
            kind_of.as_deref(),
            Some("external_file"),
            "a folder-doc source growing a page must carry source_kind='external_file', not 'memory'"
        );
    }

    #[tokio::test]
    async fn grow_page_staged_revision_card_reports_skipped_and_does_not_log_growth() {
        let (db, _dir) = test_db().await;

        let mem_v1 = "mem_grow_gate_v1";
        let v1_content = "Rust ownership keeps memory safety rules explicit in systems code";
        db.upsert_documents(vec![make_doc(mem_v1, v1_content)])
            .await
            .unwrap();

        let page_req = wenlan_types::requests::CreateConceptRequest {
            title: "Rust Ownership".to_string(),
            content: v1_content.to_string(),
            summary: None,
            entity_id: None,
            space: None.into(),
            source_memory_ids: vec![mem_v1.to_string()],
            creation_kind: Some("authored".to_string()),
            workspace: None,
        };
        let page_id = crate::post_write::create_page(&db, page_req, "test", None)
            .await
            .unwrap()
            .id;
        db.set_page_review_status(&page_id, "confirmed")
            .await
            .unwrap();
        let before = db.get_page(&page_id).await.unwrap().unwrap();
        assert_eq!(
            before.creation_kind, "authored",
            "precondition: authored pages are human-owned"
        );

        let mem_new = "mem_grow_gate_new";
        let new_content =
            "Rust ownership lets the compiler enforce memory safety during page refresh";
        db.upsert_documents(vec![make_doc(mem_new, new_content)])
            .await
            .unwrap();

        let proposed = format!("{v1_content}.[1] {new_content}.[2]");
        let stub = std::sync::Arc::new(CapturingStubProvider {
            response: proposed.clone(),
            captured_prompt: std::sync::Mutex::new(None),
        });
        let llm: std::sync::Arc<dyn LlmProvider> = stub;

        let grew = grow_page(
            &db,
            mem_new,
            new_content,
            None,
            None,
            Some(&llm),
            &crate::prompts::PromptRegistry::default(),
            0.0,
        )
        .await
        .unwrap();

        assert!(
            !grew,
            "grow_page must report skipped when PageWrite stages a revision card"
        );
        let after = db.get_page(&page_id).await.unwrap().unwrap();
        assert_eq!(
            after.content, before.content,
            "gated page growth must not overwrite human-owned page prose"
        );
        assert_eq!(
            after.version, before.version,
            "gated page growth must not bump the protected page version"
        );

        let revisions = db.list_pending_revisions(10).await.unwrap();
        assert!(
            revisions
                .iter()
                .any(|r| { r.target_source_id == page_id && r.revision_content == proposed }),
            "gated page growth must stage the proposed prose as a revision card"
        );

        let activities = db.list_agent_activity(20, None, None).await.unwrap();
        assert!(
            !activities.iter().any(|a| a.action == "page_grow"),
            "gated page growth must not log a successful page_grow activity"
        );
    }
}
