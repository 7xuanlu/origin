// SPDX-License-Identifier: AGPL-3.0-only
//! Post-ingest enrichment pipeline.
//!
//! Runs asynchronously after `store_memory` returns. Each step is
//! best-effort: failures are logged but do not block the store response
//! or subsequent steps.
//!
//! Steps:
//! 1. Dedup check (vector similarity > 0.92 → queue dedup_merge)
//! 2. Entity auto-linking (vector search entities > 0.85 distance → set entity_id)
//!    2b. Store-time entity extraction (LLM extract if auto-link found no match)
//! 3. Semantic contradiction check (type+domain pre-filter → queue detect_contradiction)
//! 4. Entity creation suggestion (stub — full impl in refinery Task 5)
//! 5. Title enrichment (LLM short title if current looks truncated)
//! 6. (Removed — recaps now handled by event-driven scheduler)
//! 7. Concept growth (update matching concept with new memory)
//! 8. (Removed -- enrichment status derived from per-step outcomes in enrichment_steps table)

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::llm_provider::LlmProvider;
use crate::prompts::PromptRegistry;
use std::sync::Arc;

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

/// Run post-ingest enrichment (async, non-blocking).
/// Called after store_memory fast track returns.
#[allow(clippy::too_many_arguments)]
pub async fn run_post_ingest_enrichment(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    entity_id: Option<&str>,
    memory_type: Option<&str>,
    domain: Option<&str>,
    structured_fields: Option<&str>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
    distillation: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> Result<(), OriginError> {
    log::info!("[post_ingest] enriching {source_id}");

    // 1. Dedup check (safety net — topic matching handles most cases pre-batcher;
    //    this fires only when a duplicate slips through)
    match check_dedup(db, source_id, content, tuning).await {
        Ok(n) if n > 0 => {
            log::warn!(
                "[post_ingest] dedup safety net fired for {source_id}: {n} duplicate candidate(s) queued. \
                 This suggests a gap in topic matching or novelty gate."
            );
            db.record_enrichment_step(source_id, "dedup", "ok", None)
                .await
                .ok();
        }
        Ok(_) => {
            db.record_enrichment_step(source_id, "dedup", "ok", None)
                .await
                .ok();
        }
        Err(e) => {
            log::warn!("[post_ingest] dedup check failed: {e}");
            db.record_enrichment_step(source_id, "dedup", "failed", Some(&e.to_string()))
                .await
                .ok();
        }
    }

    // 2. Entity auto-linking (only if not already linked)
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

            // Check for recent memories from the same agent for batched extraction
            let batch = match &agent {
                Some(a) => db
                    .find_recent_batch(a, tuning.batch_window_secs)
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

                // Extract from combined content, link entities to all batch memories
                match crate::refinery::extract_single_memory_entities(
                    db, llm_ref, prompts, source_id, &combined,
                )
                .await
                {
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
                // Single memory extraction (no batch or source_agent unknown)
                match crate::refinery::extract_single_memory_entities(
                    db, llm_ref, prompts, source_id, content,
                )
                .await
                {
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
        } else {
            db.record_enrichment_step(source_id, "entity_extract", "skipped", None)
                .await
                .ok();
        }
    } else {
        db.record_enrichment_step(source_id, "entity_extract", "skipped", None)
            .await
            .ok();
    }

    // 3. Semantic contradiction check (type+domain pre-filter)
    if let Some(mt) = memory_type {
        match check_contradiction(db, source_id, mt, domain, structured_fields, content).await {
            Ok(n) if n > 0 => {
                log::info!("[post_ingest] {source_id}: {n} contradiction candidate(s) queued");
                db.record_enrichment_step(source_id, "contradiction", "ok", None)
                    .await
                    .ok();
            }
            Ok(_) => {
                db.record_enrichment_step(source_id, "contradiction", "ok", None)
                    .await
                    .ok();
            }
            Err(e) => {
                log::warn!("[post_ingest] contradiction check failed: {e}");
                db.record_enrichment_step(
                    source_id,
                    "contradiction",
                    "failed",
                    Some(&e.to_string()),
                )
                .await
                .ok();
            }
        }
    } else {
        db.record_enrichment_step(source_id, "contradiction", "skipped", None)
            .await
            .ok();
    }

    // 3b. Concept contradiction check — flag related concepts for re-distill if new memory contradicts
    match check_concept_contradiction(db, source_id, content).await {
        Ok(n) if n > 0 => {
            log::info!("[post_ingest] {source_id}: flagged {n} concept(s) for re-distill");
            db.record_enrichment_step(source_id, "concept_contradiction", "ok", None)
                .await
                .ok();
        }
        Ok(_) => {
            db.record_enrichment_step(source_id, "concept_contradiction", "ok", None)
                .await
                .ok();
        }
        Err(e) => {
            log::warn!("[post_ingest] concept contradiction check failed: {e}");
            db.record_enrichment_step(
                source_id,
                "concept_contradiction",
                "failed",
                Some(&e.to_string()),
            )
            .await
            .ok();
        }
    }

    // 4. Entity creation suggestion (stub — full extraction runs in refinery steep)
    match suggest_entity_creation(db, content).await {
        Ok(()) => {
            db.record_enrichment_step(source_id, "entity_suggestion", "ok", None)
                .await
                .ok();
        }
        Err(e) => {
            log::warn!("[post_ingest] entity suggestion failed: {e}");
            db.record_enrichment_step(
                source_id,
                "entity_suggestion",
                "failed",
                Some(&e.to_string()),
            )
            .await
            .ok();
        }
    }

    // 5. Title enrichment — generate short topic title if current title is a truncation
    if let Some(llm_ref) = llm {
        match enrich_title(db, source_id, content, llm_ref).await {
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

    // 7. Concept growth — update matching concept with new memory
    match grow_concept(
        db,
        source_id,
        content,
        entity_id,
        llm,
        prompts,
        distillation.concept_growth_threshold,
    )
    .await
    {
        Ok(true) => {
            log::info!("[post_ingest] {source_id}: updated matching concept");
            db.record_enrichment_step(source_id, "concept_growth", "ok", None)
                .await
                .ok();
            if let Some(kp) = knowledge_path {
                write_grown_concept(db, source_id, kp).await;
            }
        }
        Ok(false) => {
            // grow_concept returns false when LLM is unavailable — treat as skipped
            if llm.map(|l| l.is_available()).unwrap_or(false) {
                db.record_enrichment_step(source_id, "concept_growth", "ok", None)
                    .await
                    .ok();
            } else {
                db.record_enrichment_step(source_id, "concept_growth", "skipped", None)
                    .await
                    .ok();
            }
        }
        Err(e) => {
            log::warn!("[post_ingest] concept growth failed: {e}");
            db.record_enrichment_step(source_id, "concept_growth", "failed", Some(&e.to_string()))
                .await
                .ok();
        }
    }

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

/// Check for semantic contradictions against existing same-type memories.
pub(crate) async fn check_contradiction(
    db: &MemoryDB,
    source_id: &str,
    memory_type: &str,
    domain: Option<&str>,
    structured_fields: Option<&str>,
    _content: &str,
) -> Result<usize, OriginError> {
    if source_id.starts_with("recap_") || memory_type.is_empty() {
        return Ok(0);
    }
    let candidates = db
        .find_same_type_memories(source_id, memory_type, domain, 3)
        .await?;
    if candidates.is_empty() {
        return Ok(0);
    }
    let mut queued = 0;
    for (existing_id, existing_sf, _existing_content) in &candidates {
        let is_candidate = match (structured_fields, existing_sf.as_deref()) {
            (Some(new_sf), Some(old_sf)) => {
                crate::contradiction::fields_may_contradict(memory_type, old_sf, new_sf)
            }
            _ => true, // No structured fields — type+domain overlap is enough signal
        };
        if is_candidate {
            log::info!(
                "[post_ingest] contradiction candidate: {} vs {}",
                source_id,
                existing_id
            );
            let proposal_id = format!("contradiction_{}_{}", source_id, existing_id);
            db.insert_refinement_proposal(
                &proposal_id,
                "detect_contradiction",
                &[source_id.to_string(), existing_id.clone()],
                None,
                0.8,
            )
            .await?;
            queued += 1;
        }
    }
    Ok(queued)
}

/// Check for near-duplicates via vector similarity (cosine > 0.92).
/// Queues `dedup_merge` refinement proposals for matches.
pub(crate) async fn check_dedup(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    tuning: &crate::tuning::RefineryConfig,
) -> Result<u32, OriginError> {
    // Recaps are expected to be similar to their source memories — skip dedup
    if source_id.starts_with("recap_") {
        return Ok(0);
    }

    // Use search() (not search_memory()) to avoid score normalization —
    // dedup threshold comparison needs raw RRF scores.
    let results = db.search(content, 5, Some("memory")).await?;

    let mut queued = 0u32;
    for result in &results {
        if result.source_id == source_id {
            continue;
        }
        if result.score > tuning.dedup_similarity_threshold as f32 {
            let id = uuid::Uuid::new_v4().to_string();
            db.insert_refinement_proposal(
                &id,
                "dedup_merge",
                &[source_id.to_string(), result.source_id.clone()],
                None,
                result.score as f64,
            )
            .await?;
            queued += 1;
        }
    }
    Ok(queued)
}

/// Auto-link memory to an existing entity via vector search.
/// Links to the best matching entity with distance < 0.15 (cosine similarity > 0.85).
pub(crate) async fn auto_link_entity(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    tuning: &crate::tuning::RefineryConfig,
) -> Result<bool, OriginError> {
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

/// Check if new memory content contradicts any related concept.
/// Uses FTS5 search to find related concepts, then checks for negation signals
/// with topic overlap. Flags contradicting concepts for re-distill by adding the
/// new memory to their source list.
pub(crate) async fn check_concept_contradiction(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
) -> Result<usize, OriginError> {
    // Find concepts related to this memory via FTS5 (use first 100 chars as query)
    let query: String = content
        .split_whitespace()
        .take(15)
        .collect::<Vec<_>>()
        .join(" ");
    let concepts = db.search_concepts(&query, 3).await.unwrap_or_default();
    if concepts.is_empty() {
        return Ok(0);
    }

    let mut flagged = 0usize;
    let content_lower = content.to_lowercase();

    for concept in &concepts {
        // Quick heuristic: if the memory contains negation/update signals,
        // it might contradict existing concept content
        let contradiction_signals = [
            "not ",
            "no longer",
            "instead of",
            "rather than",
            "changed from",
            "replaced",
            "deprecated",
            "wrong",
            "incorrect",
            "actually ",
        ];

        let has_signal = contradiction_signals
            .iter()
            .any(|s| content_lower.contains(s));
        if !has_signal {
            continue;
        }

        // Check if memory overlaps with concept topic (bigram jaccard >= 0.15)
        let overlap = crate::contradiction::bigram_jaccard(content, &concept.title);
        if overlap < 0.15 {
            continue;
        }

        // This memory likely contradicts or updates the concept — add it to sources and flag for re-distill
        if !concept.source_memory_ids.contains(&source_id.to_string()) {
            let mut new_sources = concept.source_memory_ids.clone();
            new_sources.push(source_id.to_string());
            let refs: Vec<&str> = new_sources.iter().map(|s| s.as_str()).collect();
            // Update sources without changing content — re-distill will recompile
            let _ = db
                .update_concept_content(&concept.id, &concept.content, &refs, "concept_growth")
                .await;
            log::info!("[post_ingest] concept '{}' flagged for re-distill due to potential contradiction from {}",
                concept.title, source_id);
            flagged += 1;
        }
    }

    Ok(flagged)
}

/// Stub for entity creation suggestion. Full implementation in Task 5 (refinery).
pub(crate) async fn suggest_entity_creation(
    _db: &MemoryDB,
    _content: &str,
) -> Result<(), OriginError> {
    // TODO: Detect entity-like proper nouns in content and queue
    // 'suggest_entity' refinement action if no matching entity exists.
    Ok(())
}

/// Generate a short topic title if the current title looks like a content truncation.
pub(crate) async fn enrich_title(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    llm: &Arc<dyn LlmProvider>,
) -> Result<TitleEnrichResult, OriginError> {
    // Skip recaps and merged memories — they get titles during generation
    if source_id.starts_with("recap_") || source_id.starts_with("merged_") {
        return Ok(TitleEnrichResult::NotNeeded);
    }

    // Check if current title is a truncation (ends with "..." or matches first line)
    let detail = db.get_memory_detail(source_id).await?;
    let current_title = match &detail {
        Some(d) => &d.title,
        None => return Ok(TitleEnrichResult::NotNeeded),
    };
    let first_line = content.lines().next().unwrap_or(content);
    let is_truncated =
        current_title.ends_with("...") || current_title == first_line || current_title.len() >= 75;
    if !is_truncated {
        return Ok(TitleEnrichResult::NotNeeded);
    }

    if let Some(short_title) = crate::refinery::generate_short_title(llm, content).await {
        db.update_title(source_id, &short_title).await?;
        Ok(TitleEnrichResult::Enriched)
    } else {
        Ok(TitleEnrichResult::LlmRejected)
    }
}

/// Check if new memory matches an existing concept; if so, update it.
pub(crate) async fn grow_concept(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    entity_id: Option<&str>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    growth_threshold: f64,
) -> Result<bool, OriginError> {
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

    let concept = match db
        .find_matching_concept(entity_id, mem_embedding, growth_threshold)
        .await?
    {
        Some(c) => c,
        None => return Ok(false),
    };

    // LLM: update concept with new memory
    let user_prompt = format!(
        "## Current Concept\n{}\n\n## New Memory\n[{}] {}",
        concept.content, source_id, content
    );

    let response = llm
        .generate(crate::llm_provider::LlmRequest {
            system_prompt: Some(prompts.update_concept.clone()),
            user_prompt,
            max_tokens: 1024,
            temperature: 0.1,
            label: None,
        })
        .await
        .map_err(|e| OriginError::Llm(format!("concept growth LLM: {e}")))?;

    let updated = crate::llm_provider::strip_think_tags(&response);
    let updated = updated.trim();

    if updated.is_empty() {
        return Ok(false);
    }

    // Update concept with new content + add source memory
    let mut source_ids = concept.source_memory_ids.clone();
    if !source_ids.contains(&source_id.to_string()) {
        source_ids.push(source_id.to_string());
    }
    let source_refs: Vec<&str> = source_ids.iter().map(|s| s.as_str()).collect();
    db.update_concept_content(&concept.id, updated, &source_refs, "concept_growth")
        .await?;

    // Log activity: attribute to the agent who authored the triggering memory.
    let agent = db
        .get_memory_source_agent(source_id)
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| "system".to_string());
    let detail = format!("grew \"{}\"", concept.title);
    let ids = vec![source_id.to_string()];
    if let Err(e) = db
        .log_agent_activity(&agent, "concept_grow", &ids, None, &detail)
        .await
    {
        log::warn!("[post_ingest] log concept_grow activity failed: {e}");
    }

    Ok(true)
}

async fn write_grown_concept(db: &MemoryDB, source_id: &str, knowledge_path: &std::path::Path) {
    match db.find_concept_by_source_memory(source_id).await {
        Ok(Some(concept)) => {
            let writer =
                crate::export::knowledge::KnowledgeWriter::new(knowledge_path.to_path_buf());
            match writer.write_concept(&concept) {
                Ok(path) => log::info!("[post_ingest] wrote concept to {path}"),
                Err(e) => log::warn!("[post_ingest] knowledge write failed: {e}"),
            }
        }
        Ok(None) => {}
        Err(e) => log::warn!("[post_ingest] concept lookup for knowledge write failed: {e}"),
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
            domain: None,
            source_agent: Some("test".to_string()),
            confidence: Some(0.7),
            confirmed: Some(false),
            stability: None,
            supersedes: None,
            pending_revision: false,
            entity_id: None,
            quality: None,
            is_recap: false,
            enrichment_status: "raw".to_string(),
            supersede_mode: "hide".to_string(),
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
        }
    }

    #[tokio::test]
    async fn test_check_dedup_skips_recaps() {
        let (db, _dir) = test_db().await;
        let tuning = crate::tuning::RefineryConfig::default();
        let result = check_dedup(&db, "recap_abc", "some content", &tuning)
            .await
            .unwrap();
        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn test_check_dedup_no_duplicates() {
        let (db, _dir) = test_db().await;
        let doc = make_doc("mem_unique", "Rust is a systems programming language");
        db.upsert_documents(vec![doc]).await.unwrap();
        let tuning = crate::tuning::RefineryConfig::default();
        let result = check_dedup(
            &db,
            "mem_unique",
            "Rust is a systems programming language",
            &tuning,
        )
        .await
        .unwrap();
        // Should not queue self as duplicate
        assert_eq!(result, 0);
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
    async fn test_suggest_entity_creation_stub() {
        let (db, _dir) = test_db().await;
        // Stub should always succeed
        suggest_entity_creation(&db, "Alice uses PostgreSQL")
            .await
            .unwrap();
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
        )
        .await
        .unwrap();

        let steps = db.get_enrichment_steps("mem_step_record").await.unwrap();
        assert!(!steps.is_empty(), "should have recorded enrichment steps");

        // dedup should be ok (no duplicates)
        let dedup = steps.iter().find(|s| s.step == "dedup").unwrap();
        assert_eq!(dedup.status, "ok");

        // entity_extract should be skipped (no LLM)
        let extract = steps.iter().find(|s| s.step == "entity_extract").unwrap();
        assert_eq!(extract.status, "skipped");

        // title_enrich should be skipped (no LLM)
        let title = steps.iter().find(|s| s.step == "title_enrich").unwrap();
        assert_eq!(title.status, "skipped");

        // concept_growth should be skipped (no LLM)
        let growth = steps.iter().find(|s| s.step == "concept_growth").unwrap();
        assert_eq!(growth.status, "skipped");

        // Summary should be enriched (no failures)
        let summary = db.get_enrichment_summary("mem_step_record").await.unwrap();
        assert_eq!(summary, "enriched");
    }

    #[tokio::test]
    async fn test_full_contradiction_flow() {
        let (db, _dir) = test_db().await;

        // Store and confirm an existing preference memory
        let mut existing = make_doc("mem_dark", "I prefer dark mode in editors");
        existing.memory_type = Some("preference".to_string());
        existing.domain = Some("tools".to_string());
        existing.confirmed = Some(true);
        existing.structured_fields =
            Some(r#"{"preference":"dark mode","applies_when":"editors"}"#.to_string());
        db.upsert_documents(vec![existing]).await.unwrap();

        // Store the conflicting memory so it exists in the DB
        let mut new_doc = make_doc("mem_light", "I prefer light mode in editors");
        new_doc.memory_type = Some("preference".to_string());
        new_doc.domain = Some("tools".to_string());
        new_doc.structured_fields =
            Some(r#"{"preference":"light mode","applies_when":"editors"}"#.to_string());
        db.upsert_documents(vec![new_doc]).await.unwrap();

        // Run enrichment on the new conflicting memory
        let new_sf = r#"{"preference":"light mode","applies_when":"editors"}"#;
        run_post_ingest_enrichment(
            &db,
            "mem_light",
            "I prefer light mode in editors",
            None,
            Some("preference"),
            Some("tools"),
            Some(new_sf),
            None,
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
        )
        .await
        .unwrap();

        // Verify a detect_contradiction refinement was queued
        let pending = db.get_pending_refinements().await.unwrap();
        assert!(
            pending.iter().any(|p| p.action == "detect_contradiction"),
            "should have queued a detect_contradiction refinement, got: {:?}",
            pending.iter().map(|p| &p.action).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_no_contradiction_for_different_contexts() {
        let (db, _dir) = test_db().await;

        // Store and confirm an existing preference for editors
        let mut existing = make_doc("mem_dark_ed", "I prefer dark mode in editors");
        existing.memory_type = Some("preference".to_string());
        existing.domain = Some("tools".to_string());
        existing.confirmed = Some(true);
        existing.structured_fields =
            Some(r#"{"preference":"dark mode","applies_when":"editors"}"#.to_string());
        db.upsert_documents(vec![existing]).await.unwrap();

        // Store a memory for a different context (reading documents)
        let mut new_doc = make_doc("mem_light_read", "I prefer light mode for reading");
        new_doc.memory_type = Some("preference".to_string());
        new_doc.domain = Some("tools".to_string());
        new_doc.structured_fields =
            Some(r#"{"preference":"light mode","applies_when":"reading documents"}"#.to_string());
        db.upsert_documents(vec![new_doc]).await.unwrap();

        // Run enrichment — different applies_when context should not trigger contradiction
        let new_sf = r#"{"preference":"light mode","applies_when":"reading documents"}"#;
        run_post_ingest_enrichment(
            &db,
            "mem_light_read",
            "I prefer light mode for reading",
            None,
            Some("preference"),
            Some("tools"),
            Some(new_sf),
            None,
            &crate::prompts::PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
        )
        .await
        .unwrap();

        // Verify no detect_contradiction refinement was queued
        let pending = db.get_pending_refinements().await.unwrap();
        assert!(
            !pending.iter().any(|p| p.action == "detect_contradiction"),
            "should NOT queue contradiction for different contexts, got: {:?}",
            pending.iter().map(|p| &p.action).collect::<Vec<_>>()
        );
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
        let batch = db.find_recent_batch(agent, window_secs).await.unwrap();
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
        let batch_after = db.find_recent_batch(agent, window_secs).await.unwrap();
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

        let claude_batch = db.find_recent_batch("claude-code", 600).await.unwrap();
        let cursor_batch = db.find_recent_batch("cursor", 600).await.unwrap();
        let claude_ids: Vec<String> = claude_batch.iter().map(|(id, _)| id.clone()).collect();
        let cursor_ids: Vec<String> = cursor_batch.iter().map(|(id, _)| id.clone()).collect();

        assert_eq!(claude_ids, vec!["mem_iso_claude"]);
        assert_eq!(cursor_ids, vec!["mem_iso_cursor"]);
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
}
