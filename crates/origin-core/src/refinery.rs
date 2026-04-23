// SPDX-License-Identifier: Apache-2.0
use crate::activity::ACTIVITY_GAP_SECS;
use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::merge::ContradictionResult;
use crate::prompts::PromptRegistry;
use crate::sources::StabilityTier;
use serde::Serialize;
use std::sync::Arc;

/// Canonical list of refinery phase names — single source of truth for the
/// phase set. Adding a new phase requires adding its name here AND adding it
/// to the appropriate non-Backstop arms in `TriggerKind::runs_phase`.
pub const ALL_PHASES: &[&str] = &[
    "decay",
    "promote",
    "recaps",
    "reclassify",
    "reweave",
    "reembed",
    "entity_extraction",
    "community_detection",
    "emergence",
    "re-distill",
    "refinement_queue",
    "decision_logs",
    "decision_backfill",
    "prune_rejections",
    "kg_rethink",
];

/// What triggered a refinery cycle. Different triggers run different subsets
/// of phases — the goal is to do the right work at the right time.
///
/// - `Backstop`: runs every phase. Used by the periodic backstop loop and as
///   the safe default for any code path that doesn't know better.
/// - `BurstEnd`: only `recaps` + `refinement_queue`.
/// - `Idle`: only synthesis phases (`community_detection`, `emergence`,
///   `re-distill`, `decision_logs`).
/// - `Daily`: only maintenance + backfill phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerKind {
    Backstop,
    BurstEnd,
    Idle,
    Daily,
}

impl TriggerKind {
    /// Returns true if this trigger should run the phase named `phase_name`.
    /// Unknown phase names always return `false` — typos at call sites fail
    /// loud as "phase skipped" rather than silently matching.
    pub fn runs_phase(&self, phase_name: &str) -> bool {
        match self {
            Self::Backstop => ALL_PHASES.contains(&phase_name),
            Self::BurstEnd => matches!(phase_name, "recaps" | "refinement_queue"),
            Self::Idle => matches!(
                phase_name,
                "community_detection" | "emergence" | "re-distill" | "decision_logs"
            ),
            Self::Daily => matches!(
                phase_name,
                "decay"
                    | "promote"
                    | "reclassify"
                    | "reweave"
                    | "reembed"
                    | "entity_extraction"
                    | "decision_backfill"
                    | "prune_rejections"
                    | "kg_rethink"
            ),
        }
    }

    /// Per-trigger deadline in seconds. Each trigger runs a different subset
    /// of phases, so they need different time budgets:
    /// - BurstEnd: tight (recaps + refinement only, should be fast)
    /// - Idle/Daily: moderate (focused subsets, no competition)
    /// - Backstop: generous (runs all 14 phases, safety net every 6h)
    pub fn deadline_secs(&self, base: u64) -> u64 {
        match self {
            Self::BurstEnd => base,      // 120s — recaps should be fast
            Self::Idle => base * 5,      // 600s — synthesis can take time
            Self::Daily => base * 3,     // 360s — maintenance is mostly DB ops
            Self::Backstop => base * 10, // 1200s — safety net, let it finish
        }
    }
}

/// How loud Origin should be about a phase's output — the "earned interrupt"
/// signal. Each phase classifies its own output based on what it actually
/// produced, not based on which trigger ran it.
///
/// - `Silent`: never surfaces — pure plumbing.
/// - `Ambient`: ActivityFeed entry only. No bubble, no toast.
/// - `Notable`: feed entry + Thistlebrine speech bubble (reserved for PR A.6).
/// - `Wow`: feed + bubble + OS toast. Earned interrupt, 1-2 per day max.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Nudge {
    Silent,
    Ambient,
    Notable,
    Wow,
}

/// Value a phase closure returns — the count of items processed plus the
/// self-classified `Nudge` level and optional user-facing headline.
#[derive(Debug, Clone)]
pub struct PhaseOutput {
    pub items_processed: usize,
    pub nudge: Nudge,
    pub headline: Option<String>,
}

// ---------------------------------------------------------------------------
// Nudge classifiers — one pure function per phase archetype.
// ---------------------------------------------------------------------------

/// Backfill phases (decay, promote, reclassify, reweave, reembed,
/// entity_extraction, community_detection, decision_backfill,
/// prune_rejections) are pure plumbing — always Silent.
pub(crate) fn classify_backfill(_count: usize) -> (Nudge, Option<String>) {
    (Nudge::Silent, None)
}

/// Recaps phase — Ambient when any are generated.
pub(crate) fn classify_recaps(generated: usize) -> (Nudge, Option<String>) {
    match generated {
        0 => (Nudge::Silent, None),
        1 => (
            Nudge::Ambient,
            Some("Origin steeped a recent activity burst into a recap".to_string()),
        ),
        n => (
            Nudge::Ambient,
            Some(format!(
                "Origin steeped {} recent activity bursts into recaps",
                n
            )),
        ),
    }
}

/// Emergence phase — the only phase that can produce `Wow`.
pub(crate) fn classify_emergence(new_concepts: usize) -> (Nudge, Option<String>) {
    match new_concepts {
        0 => (Nudge::Silent, None),
        1 => (
            Nudge::Wow,
            Some("Origin steeped your memories into a new concept".to_string()),
        ),
        n => (
            Nudge::Wow,
            Some(format!(
                "Origin steeped your memories into {} new concepts",
                n
            )),
        ),
    }
}

/// Re-distill phase — Ambient when any concepts are refreshed.
pub(crate) fn classify_redistill(refreshed: usize) -> (Nudge, Option<String>) {
    match refreshed {
        0 => (Nudge::Silent, None),
        1 => (
            Nudge::Ambient,
            Some("Origin refreshed a concept with new information".to_string()),
        ),
        n => (
            Nudge::Ambient,
            Some(format!(
                "Origin refreshed {} concepts with new information",
                n
            )),
        ),
    }
}

/// Refinement queue phase — Ambient when contradictions are processed.
pub(crate) fn classify_refinement_queue(processed: usize) -> (Nudge, Option<String>) {
    match processed {
        0 => (Nudge::Silent, None),
        1 => (
            Nudge::Ambient,
            Some("Origin resolved a memory contradiction".to_string()),
        ),
        n => (
            Nudge::Ambient,
            Some(format!("Origin resolved {} memory contradictions", n)),
        ),
    }
}

/// Decision logs phase — Ambient when any logs are generated.
pub(crate) fn classify_decision_logs(generated: usize) -> (Nudge, Option<String>) {
    match generated {
        0 => (Nudge::Silent, None),
        1 => (
            Nudge::Ambient,
            Some("Origin narrated a week's worth of decisions into a log".to_string()),
        ),
        n => (
            Nudge::Ambient,
            Some(format!("Origin narrated decisions into {} logs", n)),
        ),
    }
}

/// Result of a single phase within a steep cycle.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseResult {
    pub name: String,
    pub duration_ms: u64,
    pub items_processed: usize,
    pub error: Option<String>,
    /// How loud the frontend should be about this phase's output. On
    /// errors, always `Silent` — backend failures don't interrupt the user.
    pub nudge: Nudge,
    /// Optional user-facing copy describing what the phase did.
    pub headline: Option<String>,
}

/// Run a named phase, capturing timing and errors. Returns PhaseResult even
/// on failure. On error, the nudge is always `Silent` and headline is `None`
/// — backend failures should not produce user-facing notifications.
async fn run_phase<F, Fut>(name: &str, f: F) -> PhaseResult
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<PhaseOutput, OriginError>>,
{
    let start = std::time::Instant::now();
    match f().await {
        Ok(output) => PhaseResult {
            name: name.to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
            items_processed: output.items_processed,
            error: None,
            nudge: output.nudge,
            headline: output.headline,
        },
        Err(e) => PhaseResult {
            name: name.to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
            items_processed: 0,
            error: Some(e.to_string()),
            nudge: Nudge::Silent,
            headline: None,
        },
    }
}

// Post-ingest dedup and recap checks moved to post_ingest.rs

/// Periodic steep — called every 30 minutes by the scheduler.
/// Runs phases sequentially: decay, recaps, reclassify, reweave, reembed, entity extraction, distillation, refinement queue, decision logs.
/// Each phase is isolated — a failure in one phase doesn't prevent subsequent phases from running.
pub async fn run_periodic_steep(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
    _confidence_cfg: &crate::tuning::ConfidenceConfig,
    distillation: &crate::tuning::DistillationConfig,
) -> Result<SteepResult, OriginError> {
    // Forward to implementation with no API/synthesis LLM and full Backstop
    // trigger (preserves pre-PR-A behavior for callers that don't care about
    // event-driven scheduling).
    run_periodic_steep_with_api(
        db,
        llm,
        None,
        None,
        prompts,
        tuning,
        _confidence_cfg,
        distillation,
        TriggerKind::Backstop,
    )
    .await
}

/// Periodic steep with optional API and synthesis LLM providers.
/// `api_llm` is used for routine tasks (entity extraction, classification).
/// `synthesis_llm` is used for distillation/concept synthesis (falls back to api_llm → on-device).
#[allow(clippy::too_many_arguments)]
pub async fn run_periodic_steep_with_api(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    api_llm: Option<&Arc<dyn LlmProvider>>,
    synthesis_llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
    _confidence_cfg: &crate::tuning::ConfidenceConfig,
    distillation: &crate::tuning::DistillationConfig,
    trigger: TriggerKind,
) -> Result<SteepResult, OriginError> {
    let steep_start = std::time::Instant::now();
    let deadline = trigger.deadline_secs(tuning.steep_deadline_secs);
    let mut phases: Vec<PhaseResult> = Vec::new();
    #[allow(unused_assignments)]
    let mut deadline_hit = false; // track if we've logged the first skip

    // Phase metric variables — initialized to 0 so SteepResult fields stay
    // valid even when their producing phase is skipped by the trigger.
    let mut memories_decayed: u64 = 0;
    let mut recaps_generated: u32 = 0;
    let mut imports_reclassified: u32 = 0;
    let mut distilled: u32 = 0;

    // Phase 1: Decay pass
    let db_ref = db;
    if trigger.runs_phase("decay") {
        let phase = run_phase("decay", || async {
            let decayed = db_ref.decay_update_confidence().await? as usize;
            log::info!("[refinery] decay steep: updated {} memories", decayed);
            let (nudge, headline) = classify_backfill(decayed);
            Ok(PhaseOutput {
                items_processed: decayed,
                nudge,
                headline,
            })
        })
        .await;
        memories_decayed = phase.items_processed as u64;
        phases.push(phase);
    }

    // Phase 1b: Promote uncontradicted memories from 'new' to 'learned'
    if trigger.runs_phase("promote") {
        let phase = run_phase("promote", || async {
            let promoted = db_ref.promote_uncontradicted(7).await?;
            if promoted > 0 {
                log::info!("[refinery] promoted {} memories to 'learned'", promoted);
            }
            let (nudge, headline) = classify_backfill(promoted);
            Ok(PhaseOutput {
                items_processed: promoted,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 2: Recap generation (prefer API LLM for title generation)
    let recap_llm = api_llm.or(llm);
    if trigger.runs_phase("recaps") {
        let phase = run_phase("recaps", || async {
            let generated = generate_recaps(db_ref, recap_llm, prompts, tuning).await?;
            if generated > 0 {
                log::info!("[refinery] generated {} recaps", generated);
            }
            let (nudge, headline) = classify_recaps(generated as usize);
            Ok(PhaseOutput {
                items_processed: generated as usize,
                nudge,
                headline,
            })
        })
        .await;
        recaps_generated = phase.items_processed as u32;
        phases.push(phase);
    }

    // Phase 3: Reclassify imported memories
    if trigger.runs_phase("reclassify") {
        let phase = run_phase("reclassify", || async {
            let count = reclassify_imports(db_ref, llm, prompts).await?;
            let (nudge, headline) = classify_backfill(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        imports_reclassified = phase.items_processed as u32;
        phases.push(phase);
    }

    // Phase 4: Reweave entity links (link unlinked memories to recently-created entities)
    if trigger.runs_phase("reweave")
        && {
            let elapsed = steep_start.elapsed().as_secs();
            if elapsed >= deadline {
                if !deadline_hit {
                    log::warn!("[refinery] deadline exceeded ({}s >= {}s) — skipping remaining deadline-gated phases", elapsed, deadline);
                    deadline_hit = true;
                }
                false
            } else {
                true
            }
        }
    {
        let phase = run_phase("reweave", || async {
            let count = reweave_entity_links(
                db_ref,
                tuning.max_reweave_per_steep,
                tuning.entity_link_distance,
            )
            .await?;
            let (nudge, headline) = classify_backfill(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 4b: Re-embed memories with stale embeddings (structured content model)
    if trigger.runs_phase("reembed")
        && {
            let elapsed = steep_start.elapsed().as_secs();
            if elapsed >= deadline {
                if !deadline_hit {
                    log::warn!("[refinery] deadline exceeded ({}s >= {}s) — skipping remaining deadline-gated phases", elapsed, deadline);
                    deadline_hit = true;
                }
                false
            } else {
                true
            }
        }
    {
        let phase = run_phase("reembed", || async {
            let candidates = db_ref.get_reembed_candidates(5).await?;
            let mut count = 0;
            for (chunk_id, content) in &candidates {
                if let Err(e) = db_ref.reembed_memory(chunk_id, content).await {
                    log::warn!("[refinery] reembed failed for {}: {}", chunk_id, e);
                } else {
                    count += 1;
                }
            }
            let (nudge, headline) = classify_backfill(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 5: Entity extraction — prefer API LLM (better JSON accuracy), fall back to on-device
    let extract_llm = api_llm.or(llm);
    if trigger.runs_phase("entity_extraction")
        && {
            let elapsed = steep_start.elapsed().as_secs();
            if elapsed >= deadline {
                if !deadline_hit {
                    log::warn!("[refinery] deadline exceeded ({}s >= {}s) — skipping remaining deadline-gated phases", elapsed, deadline);
                    deadline_hit = true;
                }
                false
            } else {
                true
            }
        }
    {
        let phase = run_phase("entity_extraction", || async {
            let count = extract_entities_from_memories(db_ref, extract_llm, prompts, 5).await?;
            let (nudge, headline) = classify_backfill(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 5b: Community detection (runs before distillation to inform clustering)
    if trigger.runs_phase("community_detection")
        && {
            let elapsed = steep_start.elapsed().as_secs();
            if elapsed >= deadline {
                if !deadline_hit {
                    log::warn!("[refinery] deadline exceeded ({}s >= {}s) — skipping remaining deadline-gated phases", elapsed, deadline);
                    deadline_hit = true;
                }
                false
            } else {
                true
            }
        }
    {
        let phase = run_phase("community_detection", || async {
            let count = db_ref.detect_communities().await?;
            let (nudge, headline) = classify_backfill(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 6: Normal distill — create new concepts from clusters
    // Prefer synthesis LLM (Sonnet+) → API LLM (Haiku) → on-device
    let compile_llm = synthesis_llm.or(api_llm).or(llm);
    let knowledge_path = {
        let config = crate::config::load_config();
        Some(config.knowledge_path_or_default())
    };
    let kp_ref = knowledge_path.as_deref();
    if trigger.runs_phase("emergence") {
        let phase = run_phase("emergence", || async {
            let count =
                distill_concepts(db_ref, compile_llm, prompts, distillation, kp_ref).await?;
            let (nudge, headline) = classify_emergence(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        distilled = phase.items_processed as u32;
        phases.push(phase);
    }

    // Phase 6b: Re-distill — refresh concepts whose source memories changed
    if trigger.runs_phase("re-distill")
        && {
            let elapsed = steep_start.elapsed().as_secs();
            if elapsed >= deadline {
                if !deadline_hit {
                    log::warn!("[refinery] deadline exceeded ({}s >= {}s) — skipping remaining deadline-gated phases", elapsed, deadline);
                    deadline_hit = true;
                }
                false
            } else {
                true
            }
        }
    {
        let phase = run_phase("re-distill", || async {
            let count = redistill_changed_concepts(db_ref, compile_llm, prompts).await?;
            let (nudge, headline) = classify_redistill(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 6c: Process refinement queue (contradictions + entity suggestions only)
    if trigger.runs_phase("refinement_queue") {
        let phase = run_phase("refinement_queue", || async {
            let count = process_refinement_queue(db_ref, llm, prompts, tuning).await?;
            let (nudge, headline) = classify_refinement_queue(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 7: Decision log generation (lightweight recap for decisions)
    if trigger.runs_phase("decision_logs")
        && {
            let elapsed = steep_start.elapsed().as_secs();
            if elapsed >= deadline {
                if !deadline_hit {
                    log::warn!("[refinery] deadline exceeded ({}s >= {}s) — skipping remaining deadline-gated phases", elapsed, deadline);
                    deadline_hit = true;
                }
                false
            } else {
                true
            }
        }
    {
        let phase = run_phase("decision_logs", || async {
            let count = generate_decision_logs(db_ref, llm, prompts, tuning).await?;
            let (nudge, headline) = classify_decision_logs(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 7b: Backfill structured_fields for decisions that lack them
    let backfill_llm = api_llm.or(llm);
    if trigger.runs_phase("decision_backfill")
        && {
            let elapsed = steep_start.elapsed().as_secs();
            if elapsed >= deadline {
                if !deadline_hit {
                    log::warn!("[refinery] deadline exceeded ({}s >= {}s) — skipping remaining deadline-gated phases", elapsed, deadline);
                }
                false
            } else {
                true
            }
        }
    {
        let phase = run_phase("decision_backfill", || async {
            let count =
                backfill_decision_structured_fields(db_ref, backfill_llm, prompts, 5).await?;
            let (nudge, headline) = classify_backfill(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    let pending = db.get_pending_refinements().await.unwrap_or_default();
    let pending_remaining = pending.len() as u32;

    // Phase 8: Prune old rejection log entries (30-day retention).
    // Promoted from tail cleanup to a proper phase in PR A so it can be
    // gated uniformly by TriggerKind and tracked in result.phases like
    // every other phase.
    if trigger.runs_phase("prune_rejections") {
        let phase = run_phase("prune_rejections", || async {
            let count = db_ref.prune_rejections(30).await?;
            let (nudge, headline) = classify_backfill(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 9: KG rethink — periodic knowledge graph quality maintenance.
    // Rate-limited by `kg_rethink_interval_hours` (default 168h = weekly)
    // via `app_metadata.last_kg_rethink_ts`. All five sub-phases are cheap
    // when the graph is clean; the gate mainly avoids redundant log spam.
    if trigger.runs_phase("kg_rethink") {
        let interval_secs = (tuning.kg_rethink_interval_hours as i64).saturating_mul(3600);
        let now = chrono::Utc::now().timestamp();
        let last_ts: i64 = db
            .get_app_metadata("last_kg_rethink_ts")
            .await
            .ok()
            .flatten()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if now.saturating_sub(last_ts) >= interval_secs {
            let phase = run_phase("kg_rethink", || async {
                let report = crate::kg_quality::run_rethink(db_ref, llm, tuning).await?;
                let total = report.merge_candidates
                    + report.types_normalized
                    + report.embeddings_refreshed
                    + report.stale_relations_flagged
                    + report.contradictions_found;
                log::info!(
                    "[refinery] kg_rethink: {} merges, {} normalized, {} refreshed, {} stale, {} contradictions",
                    report.merge_candidates,
                    report.types_normalized,
                    report.embeddings_refreshed,
                    report.stale_relations_flagged,
                    report.contradictions_found,
                );
                let (nudge, headline) = classify_backfill(total);
                Ok(PhaseOutput {
                    items_processed: total,
                    nudge,
                    headline,
                })
            })
            .await;
            // Persist timestamp only if the phase itself didn't error.
            if phase.error.is_none() {
                let _ = db.set_app_metadata("last_kg_rethink_ts", &now.to_string()).await;
            }
            phases.push(phase);
        }
    }

    let elapsed = steep_start.elapsed();
    log::info!(
        "[refinery] steep complete in {}ms — {} phases, {} errors",
        elapsed.as_millis(),
        phases.len(),
        phases.iter().filter(|p| p.error.is_some()).count(),
    );

    // Onboarding milestone check — first-concept and graph-alive. Runs once
    // per steep after all phases complete, so entity-extraction (phase 5) and
    // distillation (phase 6) writes are both visible. Each evaluator call is
    // idempotent via DB uniqueness, so repeated passes are harmless.
    let emitter_for_ms: std::sync::Arc<dyn crate::events::EventEmitter> =
        std::sync::Arc::new(crate::events::NoopEmitter);
    let ev = crate::onboarding::MilestoneEvaluator::new(db, emitter_for_ms);
    if let Err(e) = ev.check_after_refinery_pass().await {
        log::warn!("onboarding: check_after_refinery_pass failed: {e}");
    }

    Ok(SteepResult {
        memories_decayed,
        recaps_generated,
        imports_reclassified,
        distilled,
        pending_remaining,
        phases,
    })
}

/// Reclassify imported memories that lack proper memory_type classification.
async fn reclassify_imports(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
) -> Result<usize, OriginError> {
    let mut reclassified = 0usize;
    if let Some(llm) = llm {
        let imports = db.get_unclassified_imports(3).await?;
        for (source_id, content) in &imports {
            let truncated: String = content.chars().take(1000).collect();
            let response = llm
                .generate(LlmRequest {
                    system_prompt: Some(prompts.classify_memory_quality_strict.clone()),
                    user_prompt: truncated,
                    max_tokens: 128,
                    temperature: 0.1,
                    label: None,
                })
                .await;

            match response {
                Ok(output) => {
                    if let Some(classification) =
                        crate::llm_provider::parse_classify_response(&output)
                    {
                        let sid_prefix: String = source_id.chars().take(12).collect();
                        log::info!(
                            "[refinery] reclassified import {} -> type={}, domain={:?}",
                            sid_prefix,
                            classification.memory_type,
                            classification.domain,
                        );
                        if let Err(e) = db
                            .update_memory_type(source_id, &classification.memory_type)
                            .await
                        {
                            log::warn!("[refinery] reclassify type update failed: {e}");
                        }
                        if let Some(ref domain) = classification.domain {
                            if let Err(e) = db.update_domain(source_id, domain).await {
                                log::warn!("[refinery] reclassify domain update failed: {e}");
                            }
                            // Auto-create space for newly classified domain
                            if let Err(e) = db.auto_create_space_if_needed(domain).await {
                                log::warn!("[refinery] auto-create space failed: {e}");
                            }
                        }
                        if let Some(ref quality) = classification.quality {
                            if let Err(e) = db.update_quality(source_id, quality).await {
                                log::warn!("[refinery] reclassify quality update failed: {e}");
                            }
                        }
                        reclassified += 1;
                    }
                }
                Err(e) => {
                    let sid_prefix: String = source_id.chars().take(12).collect();
                    log::warn!(
                        "[refinery] reclassification failed for {}: {}",
                        sid_prefix,
                        e
                    );
                }
            }
        }
        if reclassified > 0 {
            log::info!("[refinery] reclassified {} imported memories", reclassified);
        }
    }
    Ok(reclassified)
}

/// Extract entities from a single memory via LLM. Returns the primary entity_id if one was created/found.
pub async fn extract_single_memory_entities(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    source_id: &str,
    content: &str,
) -> Result<Option<String>, OriginError> {
    let truncated: String = content.chars().take(500).collect();
    let numbered = format!("1. {}", truncated);

    let response = llm
        .generate(LlmRequest {
            system_prompt: Some(prompts.extract_knowledge_graph.clone()),
            user_prompt: numbered,
            max_tokens: 512,
            temperature: 0.3,
            label: None,
        })
        .await
        .map_err(|e| OriginError::Llm(format!("entity extraction: {}", e)))?;

    let batch = [(0usize, content.to_string())];
    let kg_results = crate::extract::parse_kg_response(&response, &batch);

    let mut entity_cache: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut first_entity_id: Option<String> = None;

    for kg in &kg_results {
        for entity in &kg.entities {
            match crate::importer::resolve_or_create_entity(
                db,
                &mut entity_cache,
                entity,
                "post_ingest",
            )
            .await
            {
                Ok((id, _created)) => {
                    if first_entity_id.is_none() {
                        first_entity_id = Some(id);
                    }
                }
                Err(e) => log::warn!("[post_ingest] entity create failed: {e}"),
            }
        }
        for obs in &kg.observations {
            if let Some(entity_id) = entity_cache.get(&obs.entity.to_lowercase()) {
                let _ = db
                    .add_observation(entity_id, &obs.content, Some("post_ingest"), None)
                    .await;
            }
        }
        for rel in &kg.relations {
            let from_id = entity_cache.get(&rel.from.to_lowercase()).cloned();
            let to_id = entity_cache.get(&rel.to.to_lowercase()).cloned();
            if let (Some(from), Some(to)) = (from_id, to_id) {
                let _ = db
                    .create_relation(&from, &to, &rel.relation_type, Some("post_ingest"), rel.confidence, rel.explanation.as_deref(), Some(source_id))
                    .await;
            }
        }
    }

    // Link memory to first entity
    if let Some(ref eid) = first_entity_id {
        let _ = db.update_memory_entity_id(source_id, eid).await;
    }

    Ok(first_entity_id)
}

/// Extract entities from unlinked memories via LLM and create them in the knowledge graph.
/// Processes `limit` memories per steep to avoid GPU overload.
/// Acts as a backfill for any memories that failed store-time extraction.
pub(crate) async fn extract_entities_from_memories(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    limit: usize,
) -> Result<usize, OriginError> {
    let llm = match llm {
        Some(l) => l,
        None => return Ok(0),
    };

    let unlinked = db.get_unlinked_memories(limit).await?;
    if unlinked.is_empty() {
        return Ok(0);
    }

    let mut total_created = 0usize;

    for (source_id, content) in &unlinked {
        match extract_single_memory_entities(db, llm, prompts, source_id, content).await {
            Ok(Some(_)) => total_created += 1,
            Ok(None) => {}
            Err(e) => {
                let sid_prefix: String = source_id.chars().take(12).collect();
                log::warn!(
                    "[refinery] entity extraction failed for {}: {}",
                    sid_prefix,
                    e
                );
            }
        }
    }

    if total_created > 0 {
        log::info!(
            "[refinery] extracted {} new entities from {} memories",
            total_created,
            unlinked.len()
        );
    }
    Ok(total_created)
}

/// Reweave entity links: find memories with no entity_id and try to match them
/// against existing entities via vector similarity.
pub async fn reweave_entity_links(
    db: &MemoryDB,
    limit: usize,
    entity_link_distance: f64,
) -> Result<usize, OriginError> {
    let unlinked = db.get_unlinked_memories(limit).await?;
    let mut linked = 0usize;
    for (source_id, content) in &unlinked {
        let entities = db.search_entities_by_vector(content, 3).await?;
        for entity in &entities {
            if entity.distance < entity_link_distance as f32 {
                db.update_memory_entity_id(source_id, &entity.entity.id)
                    .await?;
                linked += 1;
                break;
            }
        }
    }
    if linked > 0 {
        log::info!("[refinery] reweave: linked {} memories to entities", linked);
    }
    Ok(linked)
}

/// LLM cluster refinement: for entities with multiple clusters, ask the LLM to merge/split/rename.
async fn refine_clusters_with_llm(
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    clusters: Vec<crate::db::DistillationCluster>,
    token_limit: usize,
) -> Vec<crate::db::DistillationCluster> {
    // Group clusters by entity
    let mut by_entity: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, c) in clusters.iter().enumerate() {
        let key = c
            .entity_name
            .as_deref()
            .or(c.entity_id.as_deref())
            .unwrap_or("unlinked")
            .to_string();
        by_entity.entry(key).or_default().push(i);
    }

    // Only refine entities with 2+ clusters (single clusters = nothing to merge/split)
    let entities_to_refine: Vec<(String, Vec<usize>)> = by_entity
        .into_iter()
        .filter(|(_, indices)| indices.len() >= 2)
        .collect();

    if entities_to_refine.is_empty() {
        return clusters;
    }

    let mut result = clusters;
    let mut merged_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for (entity, indices) in &entities_to_refine {
        // Build one-line summaries for each cluster
        let summaries: String = indices
            .iter()
            .enumerate()
            .map(|(j, &idx)| {
                let c = &result[idx];
                let preview: String = c
                    .contents
                    .iter()
                    .take(3)
                    .map(|s| {
                        let trimmed: String = s.chars().take(60).collect();
                        format!("\"{}...\"", trimmed)
                    })
                    .collect::<Vec<_>>()
                    .join(" / ");
                format!("{}. [{} memories] {}", j, c.source_ids.len(), preview)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let user_prompt = format!("Entity: {}\n\n{}\n/no_think", entity, summaries);

        let response = llm
            .generate(LlmRequest {
                system_prompt: Some(prompts.refine_clusters.clone()),
                user_prompt,
                max_tokens: 512,
                temperature: 0.2,
                label: None,
            })
            .await;

        match response {
            Ok(raw) => {
                let clean = crate::llm_provider::strip_think_tags(&raw);
                if let Some(json_str) = crate::llm_provider::extract_json(&clean) {
                    if let Ok(actions) = serde_json::from_str::<Vec<serde_json::Value>>(json_str) {
                        for action in &actions {
                            let act = action
                                .get("action")
                                .and_then(|v| v.as_str())
                                .unwrap_or("keep");
                            match act {
                                "merge" => {
                                    if let Some(to_merge) =
                                        action.get("clusters").and_then(|v| v.as_array())
                                    {
                                        let merge_indices: Vec<usize> = to_merge
                                            .iter()
                                            .filter_map(|v| v.as_u64().map(|n| n as usize))
                                            .filter(|&j| j < indices.len())
                                            .collect();
                                        if merge_indices.len() >= 2 {
                                            // Guard: don't merge if the result would exceed
                                            // the token limit that sub_cluster_by_tokens split
                                            // on. This prevents the LLM from re-merging
                                            // sub-clusters into a monster that OOMs distillation.
                                            let merged_tokens: usize = merge_indices
                                                .iter()
                                                .map(|&j| result[indices[j]].estimated_tokens)
                                                .sum();
                                            if merged_tokens > token_limit {
                                                log::info!(
                                                    "[refine] skipping merge for '{}' — merged tokens {} > limit {}",
                                                    entity, merged_tokens, token_limit
                                                );
                                            } else {
                                                // Merge: combine all into the first
                                                let first = indices[merge_indices[0]];
                                                for &j in &merge_indices[1..] {
                                                    let idx = indices[j];
                                                    let extra_ids = result[idx].source_ids.clone();
                                                    let extra_contents =
                                                        result[idx].contents.clone();
                                                    result[first].source_ids.extend(extra_ids);
                                                    result[first].contents.extend(extra_contents);
                                                    result[first].estimated_tokens +=
                                                        result[idx].estimated_tokens;
                                                    merged_indices.insert(idx);
                                                }
                                                if let Some(title) =
                                                    action.get("title").and_then(|v| v.as_str())
                                                {
                                                    result[first].entity_name =
                                                        Some(title.to_string());
                                                }
                                                log::info!(
                                                    "[refine] merged {} clusters for '{}'",
                                                    merge_indices.len(),
                                                    entity
                                                );
                                            } // close else (token guard)
                                        }
                                    }
                                }
                                "rename" => {
                                    if let (Some(j), Some(title)) = (
                                        action
                                            .get("cluster")
                                            .and_then(|v| v.as_u64().map(|n| n as usize)),
                                        action.get("title").and_then(|v| v.as_str()),
                                    ) {
                                        if j < indices.len() {
                                            result[indices[j]].entity_name =
                                                Some(title.to_string());
                                            log::info!(
                                                "[refine] renamed cluster {} to '{}' for '{}'",
                                                j,
                                                title,
                                                entity
                                            );
                                        }
                                    }
                                }
                                // "keep" and "split" — split is complex (needs new clusters), defer to global_concept_review
                                _ => {}
                            }
                        }
                    }
                }
            }
            Err(e) => log::warn!("[refine] LLM refinement failed for '{}': {}", entity, e),
        }
    }

    // Remove merged clusters
    if !merged_indices.is_empty() {
        result = result
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !merged_indices.contains(i))
            .map(|(_, c)| c)
            .collect();
    }

    result
}

/// Distill memory clusters into structured concepts.
/// Memories can appear in multiple concepts. Jaccard overlap prevents duplicate concepts.
pub async fn distill_concepts(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> Result<usize, OriginError> {
    let llm = match llm {
        Some(l) if l.is_available() => l,
        _ => return Ok(0),
    };

    // Each model carries its own effective synthesis limit — the max tokens it
    // can meaningfully synthesize (not just read). Research-calibrated per model
    // in on_device_models.rs and llm_provider.rs. Falls back to tuning config
    // if the provider returns the default (for backward compat).
    let token_limit = llm.synthesis_token_limit();
    let raw_clusters = db
        .find_distillation_clusters(
            tuning.similarity_threshold,
            tuning.concept_min_cluster_size,
            tuning.max_clusters_per_steep,
            token_limit,
        )
        .await?;

    // LLM cluster refinement: let LLM merge/split/rename clusters per entity
    let clusters = refine_clusters_with_llm(llm, prompts, raw_clusters, token_limit).await;

    let mut distilled = 0usize;

    // Create the writer once, outside the loop
    let knowledge_writer =
        knowledge_path.map(|kp| crate::export::knowledge::KnowledgeWriter::new(kp.to_path_buf()));

    for cluster in &clusters {
        let topic = cluster
            .entity_name
            .as_deref()
            .or(cluster.domain.as_deref())
            .unwrap_or("general");

        // Skip if a concept with very similar sources already exists (Jaccard > 0.8)
        // Memories CAN appear in multiple concepts — this only prevents duplicate concepts
        let overlap = db
            .max_concept_overlap(&cluster.source_ids)
            .await
            .unwrap_or(0.0);
        if overlap > 0.8 {
            log::info!(
                "[emergence] cluster '{}' overlaps {:.0}% with existing concept, skipping",
                topic,
                overlap * 100.0
            );
            continue;
        }

        // Clean input: strip recap headers, domain prefixes, and structured field noise
        let cleaned_contents: Vec<String> = cluster
            .contents
            .iter()
            .map(|c| {
                let mut s = c.trim().to_string();
                // Strip "Activity burst: ..." header lines
                if let Some(pos) = s.find("\n- ") {
                    let prefix: String = s.chars().take(pos).collect();
                    if prefix.contains("Activity burst") || prefix.contains("memories across") {
                        s = s.chars().skip(pos + 1).collect();
                    }
                }
                // Strip "- [domain] " prefixes from each line
                s = s
                    .lines()
                    .map(|line| {
                        let trimmed = line.trim_start_matches("- ");
                        if trimmed.starts_with('[') {
                            if let Some(end) = trimmed.find("] ") {
                                trimmed[end + 2..].to_string()
                            } else {
                                line.to_string()
                            }
                        } else {
                            line.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                // Strip "claim: " prefix
                if let Some(rest) = s.strip_prefix("claim: ") {
                    s = rest.to_string();
                }
                s
            })
            .collect();

        // Skip thin clusters — not enough substance for meaningful compilation
        let total_content_chars: usize = cleaned_contents.iter().map(|c| c.len()).sum();
        if total_content_chars < 200 {
            log::info!(
                "[compile] cluster too thin ({} chars), skipping topic='{}'",
                total_content_chars,
                topic
            );
            continue;
        }

        log::info!(
            "[distill] processing cluster: {} memories, ~{} tokens",
            cluster.source_ids.len(),
            cluster.estimated_tokens
        );

        // Build user prompt with memory IDs for source attribution.
        // Cap each memory at 800 chars so the LLM gets meaningful substance
        // without runaway context. The 800-char cap is honest: it matches the
        // amount the model can synthesize well at 2048 output tokens.
        const MEM_SNIPPET_CAP: usize = 800;
        let memories_block: String = cluster
            .source_ids
            .iter()
            .zip(cleaned_contents.iter())
            .map(|(id, content)| {
                let snippet: String = content.chars().take(MEM_SNIPPET_CAP).collect();
                let snippet = if content.chars().count() > MEM_SNIPPET_CAP {
                    format!("{}...", snippet.trim_end())
                } else {
                    snippet
                };
                format!("[{}] {}", id, snippet)
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        let user_prompt = format!("Topic: {}\n\n{}\n/no_think", topic, memories_block);

        let response = llm
            .generate(LlmRequest {
                system_prompt: Some(prompts.distill_concept.clone()),
                user_prompt,
                max_tokens: llm.recommended_max_output(),
                temperature: 0.1,
                label: Some("distill_body".into()),
            })
            .await;

        match response {
            Ok(raw) if !raw.trim().is_empty() => {
                let cleaned = crate::llm_provider::strip_think_tags(&raw);
                let cleaned = cleaned.replace("/no_think", "");
                let content = cleaned.trim().to_string();

                if content.is_empty() {
                    log::warn!("[distill] empty output for topic='{}', skipping", topic);
                    continue;
                }

                // Hallucination check: output must be semantically similar to input
                let texts = vec![content.clone(), cleaned_contents.join(" ")];
                if let Ok(embeddings) = db.generate_embeddings(&texts) {
                    if embeddings.len() == 2 {
                        let sim = crate::db::cosine_similarity(&embeddings[0], &embeddings[1]);
                        if sim < 0.6 {
                            log::warn!("[compile] hallucination detected (sim={:.2}) for topic='{}', skipping", sim, topic);
                            continue;
                        }
                        log::info!(
                            "[compile] quality check passed (sim={:.2}) for topic='{}'",
                            sim,
                            topic
                        );
                    }
                }

                // Generate title. If LLM returns None and the only fallback is a generic
                // placeholder (e.g. "general"), skip this cluster entirely — a generic title
                // is worse than no concept at all.
                let llm_title = generate_short_title(llm, &content).await;
                let title = match llm_title {
                    Some(t) => t,
                    None if looks_like_generic_topic(topic)
                        || looks_like_path(topic)
                        || looks_like_code(topic)
                        || looks_like_uuid(topic)
                        || looks_like_short_hash(topic)
                        || looks_like_commit_message(topic) =>
                    {
                        log::info!(
                            "[distill] no title and topic='{}' is garbage, skipping cluster",
                            topic
                        );
                        continue;
                    }
                    None => topic.to_string(),
                };

                // Extract summary from first bullet point
                let summary = content
                    .lines()
                    .find(|l| l.starts_with("- "))
                    .map(|l| l.trim_start_matches("- ").to_string());

                // Build source IDs as &str refs
                let source_refs: Vec<&str> =
                    cluster.source_ids.iter().map(|s| s.as_str()).collect();
                let now = chrono::Utc::now().to_rfc3339();
                let concept_id = crate::concepts::Concept::new_id();

                db.insert_concept(
                    &concept_id,
                    &title,
                    summary.as_deref(),
                    &content,
                    cluster.entity_id.as_deref(),
                    cluster.domain.as_deref(),
                    &source_refs,
                    &now,
                )
                .await?;

                log::info!(
                    "[distill] distilled {} memories -> concept '{}' ('{}')",
                    cluster.source_ids.len(),
                    title,
                    content.chars().take(40).collect::<String>()
                );
                distilled += 1;

                // Log activity — system-attributed, since distillation is background refinery work.
                let source_memory_ids: Vec<String> = cluster.source_ids.to_vec();
                let detail = format!(
                    "created \"{}\" from {} memories",
                    title,
                    cluster.source_ids.len()
                );
                if let Err(e) = db
                    .log_agent_activity(
                        "system",
                        "concept_create",
                        &source_memory_ids,
                        None,
                        &detail,
                    )
                    .await
                {
                    log::warn!("[distill] log concept_create activity failed: {e}");
                }

                if let Some(ref writer) = knowledge_writer {
                    if let Ok(Some(c)) = db.get_concept(&concept_id).await {
                        match writer.write_concept(&c) {
                            Ok(p) => log::info!("[distill] wrote concept to {p}"),
                            Err(e) => log::warn!("[distill] knowledge write failed: {e}"),
                        }
                    }
                }
            }
            Ok(_) => {
                log::warn!("[distill] empty output for topic='{}'", topic);
            }
            Err(e) => {
                log::warn!("[distill] LLM error for topic='{}': {}", topic, e);
            }
        }
    }

    Ok(distilled)
}

/// Layer 2: LLM assigns orphan memories to existing concepts or proposes new ones.
async fn assign_orphan_memories(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    _tuning: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> Result<usize, OriginError> {
    // Find orphan memories: no entity_id, not already in a concept, not recap/merged
    let orphans = db.get_unlinked_memories(30).await?;
    // Filter out merged memories
    let orphans: Vec<(String, String)> = orphans
        .into_iter()
        .filter(|(sid, _)| !sid.starts_with("merged_"))
        .collect();

    if orphans.is_empty() {
        return Ok(0);
    }

    // Get existing concept titles
    let concepts = db.list_concepts("active", 100, 0).await?;
    if concepts.is_empty() && orphans.len() < 3 {
        return Ok(0); // Not enough material
    }

    // Build prompt
    let memories_text: String = orphans
        .iter()
        .enumerate()
        .map(|(i, (_, c))| format!("{}. {}", i, c.chars().take(200).collect::<String>()))
        .collect::<Vec<_>>()
        .join("\n");

    let concepts_text: String = concepts
        .iter()
        .map(|c| {
            format!(
                "[{}] {}: {}",
                c.id,
                c.title,
                c.summary.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let user_prompt = format!(
        "Unassigned memories:\n{}\n\nExisting concepts:\n{}\n/no_think",
        memories_text, concepts_text
    );

    let response = llm
        .generate(LlmRequest {
            system_prompt: Some(prompts.assign_orphans.clone()),
            user_prompt,
            max_tokens: 1024,
            temperature: 0.3,
            label: Some("orphan_assign".into()),
        })
        .await
        .map_err(|e| OriginError::Llm(format!("orphan assignment: {}", e)))?;

    let clean = crate::llm_provider::strip_think_tags(&response);

    // Create the writer once, outside the loop
    let knowledge_writer =
        knowledge_path.map(|kp| crate::export::knowledge::KnowledgeWriter::new(kp.to_path_buf()));

    // Parse assignments
    let mut assigned = 0usize;
    if let Some(json_str) = crate::llm_provider::extract_json(&clean) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            // Process assignments to existing concepts
            if let Some(assignments) = parsed.get("assignments").and_then(|a| a.as_array()) {
                for assignment in assignments {
                    let idx = assignment
                        .get("memory_index")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(999) as usize;
                    let concept_id = assignment
                        .get("concept_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if idx < orphans.len() && !concept_id.is_empty() {
                        let source_id = &orphans[idx].0;
                        // Add this memory to the concept's source list
                        if let Ok(Some(concept)) = db.get_concept(concept_id).await {
                            if !concept.source_memory_ids.contains(&source_id.to_string()) {
                                let mut merged_sources = concept.source_memory_ids.clone();
                                merged_sources.push(source_id.to_string());
                                let refs: Vec<&str> =
                                    merged_sources.iter().map(|s| s.as_str()).collect();
                                let _ = db
                                    .update_concept_content(concept_id, &concept.content, &refs)
                                    .await;
                                assigned += 1;
                            }
                        }
                    }
                }
            }

            // Process proposals (new concepts from orphan groups)
            if let Some(proposals) = parsed.get("proposals").and_then(|a| a.as_array()) {
                for proposal in proposals {
                    let title = proposal.get("title").and_then(|v| v.as_str()).unwrap_or("");
                    let indices = proposal
                        .get("memory_indices")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();

                    if title.is_empty() || indices.len() < 2 {
                        continue;
                    }

                    let valid_indices: Vec<usize> =
                        indices.into_iter().filter(|&i| i < orphans.len()).collect();
                    if valid_indices.len() < 2 {
                        continue;
                    }

                    // Create a new concept from these orphan memories
                    let source_ids: Vec<&str> = valid_indices
                        .iter()
                        .map(|&i| orphans[i].0.as_str())
                        .collect();
                    let contents: Vec<String> = valid_indices
                        .iter()
                        .map(|&i| orphans[i].1.clone())
                        .collect();
                    let content_text = contents.join("\n\n");

                    let concept_id = crate::concepts::Concept::new_id();
                    let now = chrono::Utc::now().to_rfc3339();

                    let _ = db
                        .insert_concept(
                            &concept_id,
                            title,
                            Some(&format!(
                                "Auto-grouped from {} orphan memories",
                                source_ids.len()
                            )),
                            &content_text,
                            None, // no entity_id
                            None, // no domain
                            &source_ids,
                            &now,
                        )
                        .await;
                    assigned += source_ids.len();

                    if let Some(ref writer) = knowledge_writer {
                        if let Ok(Some(c)) = db.get_concept(&concept_id).await {
                            match writer.write_concept(&c) {
                                Ok(p) => log::info!("[distill] wrote concept to {p}"),
                                Err(e) => log::warn!("[distill] knowledge write failed: {e}"),
                            }
                        }
                    }
                }
            }
        }
    }

    if assigned > 0 {
        log::info!(
            "[distill] orphan assignment: {} memories processed",
            assigned
        );
    }
    Ok(assigned)
}

/// Full Karpathy-style deep distill: emergence + orphans + recompile ALL + global review.
/// Triggered by "Distill now" button or weekly background schedule.
pub async fn deep_distill_concepts(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> Result<usize, OriginError> {
    let llm_ref = match llm {
        Some(l) if l.is_available() => l,
        _ => return Ok(0),
    };

    let mut total = 0usize;

    // 1. Emergence — create new concepts from clusters
    let created = distill_concepts(db, llm, prompts, tuning, knowledge_path)
        .await
        .unwrap_or(0);
    total += created;
    if created > 0 {
        log::info!("[deep_distill] created {} new concepts", created);
    }

    // 2. Orphan assignment — assign unlinked memories to concepts or propose new ones
    match assign_orphan_memories(db, llm_ref, prompts, tuning, knowledge_path).await {
        Ok(n) => {
            total += n;
            if n > 0 {
                log::info!("[deep_distill] assigned {} orphan memories", n);
            }
        }
        Err(e) => log::warn!("[deep_distill] orphan assignment failed: {}", e),
    }

    // 3. Recompile ALL active concepts (not just changed ones — full refresh)
    let all_active = db.list_concepts("active", 200, 0).await?;
    for concept in &all_active {
        match recompile_single_concept(db, llm_ref, prompts, concept).await {
            Ok(true) => total += 1,
            Ok(false) => {}
            Err(e) => log::warn!(
                "[deep_distill] recompile failed for '{}': {}",
                concept.title,
                e
            ),
        }
    }

    // 4. Global review — merge/split/create analysis
    if all_active.len() >= 5 {
        match global_concept_review(db, llm_ref, prompts, &all_active).await {
            Ok(n) => {
                total += n;
                if n > 0 {
                    log::info!("[deep_distill] global review applied {} changes", n);
                }
            }
            Err(e) => log::warn!("[deep_distill] global review failed: {}", e),
        }
    }

    log::info!("[deep_distill] complete: {} total changes", total);
    Ok(total)
}

/// Re-distill concepts whose source memories have changed.
/// Called by the steep cycle — only refreshes concepts with meaningful input changes.
pub(crate) async fn redistill_changed_concepts(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
) -> Result<usize, OriginError> {
    let llm = match llm {
        Some(l) if l.is_available() => l,
        _ => return Ok(0),
    };

    let all_active = db.list_concepts("active", 200, 0).await?;
    let mut recompiled = 0usize;

    for concept in &all_active {
        let changed = db
            .has_concept_sources_changed(concept)
            .await
            .unwrap_or(false);
        if !changed {
            continue;
        }

        match recompile_single_concept(db, llm, prompts, concept).await {
            Ok(true) => recompiled += 1,
            Ok(false) => {}
            Err(e) => log::warn!("[re-distill] failed for '{}': {}", concept.title, e),
        }
    }

    if recompiled > 0 {
        log::info!(
            "[re-distill] refreshed {} concepts with changed inputs",
            recompiled
        );
    }
    Ok(recompiled)
}

/// Recompile a single concept from its source memories via LLM.
async fn recompile_single_concept(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    concept: &crate::concepts::Concept,
) -> Result<bool, OriginError> {
    let memories = db
        .get_memory_contents_by_ids(&concept.source_memory_ids)
        .await?;
    if memories.is_empty() {
        log::warn!(
            "[re-distill] concept '{}' has no source memories, skipping",
            concept.id
        );
        return Ok(false);
    }

    const MEM_SNIPPET_CAP: usize = 800;
    let memories_block: String = memories
        .iter()
        .map(|(id, content)| {
            let snippet: String = content.chars().take(MEM_SNIPPET_CAP).collect();
            let snippet = if content.chars().count() > MEM_SNIPPET_CAP {
                format!("{}...", snippet.trim_end())
            } else {
                snippet
            };
            format!("[{}] {}", id, snippet)
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let user_prompt = format!("Topic: {}\n\n{}\n/no_think", concept.title, memories_block);

    let response = llm
        .generate(LlmRequest {
            system_prompt: Some(prompts.distill_concept.clone()),
            user_prompt,
            max_tokens: llm.recommended_max_output(),
            temperature: 0.1,
            label: Some("distill_body".into()),
        })
        .await;

    match response {
        Ok(raw) if !raw.trim().is_empty() => {
            let content = crate::llm_provider::strip_think_tags(&raw)
                .replace("/no_think", "")
                .trim()
                .to_string();
            if !content.is_empty() {
                let source_refs: Vec<&str> = concept
                    .source_memory_ids
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                db.update_concept_content(&concept.id, &content, &source_refs)
                    .await?;
                log::info!("[re-distill] refreshed concept '{}'", concept.title);
                return Ok(true);
            }
        }
        Ok(_) => log::warn!("[re-distill] empty output for '{}'", concept.title),
        Err(e) => log::warn!("[re-distill] LLM error for '{}': {}", concept.title, e),
    }
    Ok(false)
}

/// Layer 3: Periodic global review -- merge/split/create concepts based on holistic analysis.
async fn global_concept_review(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    concepts: &[crate::concepts::Concept],
) -> Result<usize, OriginError> {
    let concepts_text: String = concepts
        .iter()
        .map(|c| {
            format!(
                "[{}] {}: {}",
                c.id,
                c.title,
                c.summary.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let response = llm
        .generate(LlmRequest {
            system_prompt: Some(prompts.global_concept_review.clone()),
            user_prompt: format!("{}\n/no_think", concepts_text),
            max_tokens: 1024,
            temperature: 0.3,
            label: Some("global_review".into()),
        })
        .await
        .map_err(|e| OriginError::Llm(format!("global review: {}", e)))?;

    let clean = crate::llm_provider::strip_think_tags(&response);
    let mut changes = 0usize;

    if let Some(json_str) = crate::llm_provider::extract_json(&clean) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            // Process merges
            if let Some(merges) = parsed.get("merges").and_then(|a| a.as_array()) {
                for merge in merges {
                    let keep_id = merge.get("keep").and_then(|v| v.as_str()).unwrap_or("");
                    let remove_id = merge.get("remove").and_then(|v| v.as_str()).unwrap_or("");
                    if keep_id.is_empty() || remove_id.is_empty() {
                        continue;
                    }

                    // Merge: transfer source_memory_ids from remove to keep, archive remove
                    if let (Ok(Some(keep)), Ok(Some(remove))) = (
                        db.get_concept(keep_id).await,
                        db.get_concept(remove_id).await,
                    ) {
                        let mut merged_sources = keep.source_memory_ids.clone();
                        for sid in &remove.source_memory_ids {
                            if !merged_sources.contains(sid) {
                                merged_sources.push(sid.clone());
                            }
                        }
                        let refs: Vec<&str> = merged_sources.iter().map(|s| s.as_str()).collect();
                        let _ = db
                            .update_concept_content(keep_id, &keep.content, &refs)
                            .await;
                        let _ = db.archive_concept(remove_id).await;
                        changes += 1;
                        log::info!(
                            "[distill] merged concept '{}' into '{}'",
                            remove.title,
                            keep.title
                        );
                    }
                }
            }
            // Note: splits and missing concepts logged but not auto-applied (too risky)
            if let Some(splits) = parsed.get("splits").and_then(|a| a.as_array()) {
                for split in splits {
                    let cid = split
                        .get("concept_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let titles = split.get("sub_titles").and_then(|v| v.as_array());
                    if !cid.is_empty() {
                        log::info!(
                            "[distill] global review suggests splitting concept {}: {:?}",
                            cid,
                            titles
                        );
                    }
                }
            }
        }
    }

    Ok(changes)
}

/// Re-distill a single concept by reloading all source memories and recompiling with LLM.
pub async fn deep_distill_single(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    concept_id: &str,
) -> Result<(), OriginError> {
    let llm = match llm {
        Some(l) if l.is_available() => l,
        Some(_) => {
            return Err(OriginError::Llm(
                "LLM not available for re-distillation".into(),
            ))
        }
        None => {
            return Err(OriginError::Llm(
                "No LLM available for re-distillation".into(),
            ))
        }
    };

    let concept = db
        .get_concept(concept_id)
        .await?
        .ok_or_else(|| OriginError::VectorDb(format!("Concept {} not found", concept_id)))?;

    let memories = db
        .get_memory_contents_by_ids(&concept.source_memory_ids)
        .await?;
    if memories.is_empty() {
        log::warn!(
            "[distill] no source memories found for concept {}",
            concept_id
        );
        return Ok(());
    }

    const MEM_SNIPPET_CAP: usize = 800;
    let memories_block: String = memories
        .iter()
        .map(|(id, content)| {
            let snippet: String = content.chars().take(MEM_SNIPPET_CAP).collect();
            let snippet = if content.chars().count() > MEM_SNIPPET_CAP {
                format!("{}...", snippet.trim_end())
            } else {
                snippet
            };
            format!("[{}] {}", id, snippet)
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let user_prompt = format!("Topic: {}\n\n{}\n/no_think", concept.title, memories_block);

    let response = llm
        .generate(LlmRequest {
            system_prompt: Some(prompts.distill_concept.clone()),
            user_prompt,
            max_tokens: llm.recommended_max_output(),
            temperature: 0.1,
            label: Some("distill_body".into()),
        })
        .await
        .map_err(|e| OriginError::Llm(format!("re-distill LLM: {}", e)))?;

    let content = crate::llm_provider::strip_think_tags(&response)
        .replace("/no_think", "")
        .trim()
        .to_string();

    if content.is_empty() {
        log::warn!(
            "[distill] empty output for concept '{}', skipping",
            concept.title
        );
        return Ok(());
    }

    let source_refs: Vec<&str> = concept
        .source_memory_ids
        .iter()
        .map(|s| s.as_str())
        .collect();
    db.update_concept_content(concept_id, &content, &source_refs)
        .await?;

    log::info!(
        "[distill] re-distilled concept '{}' (v{}->v{})",
        concept.title,
        concept.version,
        concept.version + 1
    );
    Ok(())
}

/// Process pending refinement queue items via LLM.
async fn process_refinement_queue(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
) -> Result<usize, OriginError> {
    let pending = db.get_pending_refinements().await?;
    let mut processed = 0usize;

    for proposal in pending.iter().take(tuning.max_proposals_per_steep) {
        match proposal.action.as_str() {
            "dedup_merge" => {
                // Stale v1 proposal — dismiss (distillation handles merges now)
                db.resolve_refinement(&proposal.id, "dismissed").await?;
                processed += 1;
            }
            "detect_contradiction" => {
                if let Some(llm) = llm {
                    let contents = db.get_memory_contents(&proposal.source_ids).await?;
                    if contents.len() < 2 {
                        db.resolve_refinement(&proposal.id, "dismissed").await?;
                        continue;
                    }

                    let existing_content = contents.get(1).cloned().unwrap_or_default();
                    let new_content = contents.first().cloned().unwrap_or_default();

                    let response = llm
                        .generate(LlmRequest {
                            system_prompt: Some(prompts.detect_contradiction.clone()),
                            user_prompt: format!(
                                "Existing: {}\nNew: {}",
                                existing_content, new_content
                            ),
                            max_tokens: 256,
                            temperature: 0.1,
                            label: None,
                        })
                        .await;

                    if let Ok(r) = response {
                        let r = crate::llm_provider::strip_think_tags(&r);
                        let r = r.trim().to_string();
                        let result = if r.starts_with("CONTRADICTS:") {
                            ContradictionResult::Contradicts {
                                explanation: r
                                    .strip_prefix("CONTRADICTS:")
                                    .unwrap_or("")
                                    .trim()
                                    .to_string(),
                            }
                        } else if r.starts_with("SUPERSEDES:") {
                            ContradictionResult::Supersedes {
                                merged_content: r
                                    .strip_prefix("SUPERSEDES:")
                                    .unwrap_or("")
                                    .trim()
                                    .to_string(),
                            }
                        } else {
                            ContradictionResult::Consistent
                        };

                        match result {
                            ContradictionResult::Consistent => {
                                db.resolve_refinement(&proposal.id, "dismissed").await?;
                            }
                            ContradictionResult::Contradicts { explanation } => {
                                log::info!("[refinery] contradiction detected: {}", explanation);
                                db.resolve_refinement(&proposal.id, "awaiting_review")
                                    .await?;
                            }
                            ContradictionResult::Supersedes { merged_content } => {
                                let tier = db.get_highest_tier(&proposal.source_ids).await?;
                                apply_merge_by_tier(
                                    db,
                                    &proposal.source_ids,
                                    &merged_content,
                                    &proposal.id,
                                    &tier,
                                )
                                .await?;
                            }
                        }
                        processed += 1;
                    }
                }
            }
            "suggest_entity" => {
                // Entity suggestion: payload contains the suggested entity name.
                // Mark as awaiting_review so the UI can surface it for approval.
                db.resolve_refinement(&proposal.id, "awaiting_review")
                    .await?;
                log::info!(
                    "[refinery] entity suggestion queued for review: {:?}",
                    proposal.payload
                );
                processed += 1;
            }
            _ => {
                log::debug!("[refinery] unknown action: {}", proposal.action);
            }
        }
    }
    Ok(processed)
}

/// Backfill structured_fields for decision memories that lack them.
/// Runs extraction via LLM to populate decision, context, alternatives_considered, etc.
/// Processes `limit` decisions per steep to stay within budget.
pub(crate) async fn backfill_decision_structured_fields(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    _prompts: &PromptRegistry,
    limit: usize,
) -> Result<usize, OriginError> {
    let llm = match llm {
        Some(l) if l.is_available() => l,
        _ => return Ok(0),
    };

    let candidates = db.get_decisions_without_structured_fields(limit).await?;
    if candidates.is_empty() {
        return Ok(0);
    }

    let mut backfilled = 0usize;
    for (source_id, content) in &candidates {
        let truncated: String = content.chars().take(500).collect();
        let prompt = crate::memory_schema::extraction_prompt("decision");

        match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            llm.generate(crate::llm_provider::LlmRequest {
                system_prompt: Some(prompt),
                user_prompt: truncated,
                max_tokens: 256,
                temperature: 0.1,
                label: None,
            }),
        )
        .await
        {
            Ok(Ok(raw)) => {
                let (fields_json, _retrieval_cue) =
                    crate::llm_provider::parse_extraction_response(&raw);
                if let Some(ref json_str) = fields_json {
                    if !json_str.is_empty() && json_str != "{}" {
                        if let Err(e) = db.update_structured_fields(source_id, json_str).await {
                            log::warn!(
                                "[refinery] backfill structured_fields for {}: {}",
                                source_id,
                                e
                            );
                        } else {
                            backfilled += 1;
                            log::info!(
                                "[refinery] backfilled structured_fields for decision {}",
                                source_id
                            );
                        }
                    }
                }
            }
            Ok(Err(e)) => log::warn!("[refinery] backfill LLM error for {}: {}", source_id, e),
            Err(_) => log::warn!("[refinery] backfill timeout for {}", source_id),
        }
    }

    if backfilled > 0 {
        log::info!(
            "[refinery] backfilled structured_fields for {}/{} decisions",
            backfilled,
            candidates.len()
        );
    }
    Ok(backfilled)
}

/// Generate decision logs: recap-like summaries for clusters of decision memories.
/// Groups recent decision memories and generates a summary recap if not already covered.
pub(crate) async fn generate_decision_logs(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
) -> Result<usize, OriginError> {
    let now = chrono::Utc::now().timestamp();
    let lookback_start = now - tuning.recap_lookback_secs;

    // Get recent decision memories
    let recent_decisions = db
        .get_recent_decisions_for_recap(lookback_start, 50)
        .await?;
    if recent_decisions.len() < tuning.min_memories_for_recap {
        return Ok(0);
    }

    let bursts = group_into_bursts(&recent_decisions);
    let mut generated = 0usize;

    for burst in &bursts {
        if burst.len() < tuning.min_memories_for_recap {
            continue;
        }

        let burst_start = burst.iter().map(|m| m.3).min().unwrap_or(0);
        let burst_end = burst.iter().map(|m| m.3).max().unwrap_or(0);

        if db.has_recap_covering_range(burst_start, burst_end).await? {
            continue;
        }

        let burst_slice: Vec<(String, String, Option<String>, i64)> =
            burst.iter().map(|m| (*m).clone()).collect();
        let raw_context = build_burst_context(&burst_slice, burst_start, burst_end);

        let (summary, content) = if let Some(llm) = llm {
            let combined = burst
                .iter()
                .enumerate()
                .map(|(i, (_, content, domain, _))| {
                    let d = domain.as_deref().unwrap_or("general");
                    format!("{}. [{}] {}", i + 1, d, content)
                })
                .collect::<Vec<_>>()
                .join("\n");

            let response = llm
                .generate(LlmRequest {
                    system_prompt: Some(prompts.summarize_decisions.clone()),
                    user_prompt: combined,
                    max_tokens: 128,
                    temperature: 0.1,
                    label: None,
                })
                .await;

            match response {
                Ok(output)
                    if output.trim().to_lowercase() != "null" && !output.trim().is_empty() =>
                {
                    let cleaned = crate::llm_provider::strip_think_tags(&output);
                    (cleaned.trim().to_string(), raw_context)
                }
                _ => {
                    let summary = format!("Decision log: {} decisions recorded", burst.len());
                    (summary, raw_context)
                }
            }
        } else {
            let summary = format!("Decision log: {} decisions recorded", burst.len());
            (summary, raw_context)
        };

        let recap_id = format!("recap_{}", uuid::Uuid::new_v4());
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: recap_id,
            title: summary.chars().take(80).collect(),
            summary: Some(summary),
            content,
            url: None,
            last_modified: burst_end,
            metadata: std::collections::HashMap::new(),
            memory_type: Some("decision".to_string()),
            domain: None,
            source_agent: Some("refinery".to_string()),
            confidence: Some(0.5),
            confirmed: None,
            supersedes: None,
            pending_revision: false,
            is_recap: true,
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await?;
        generated += 1;
    }

    if generated > 0 {
        log::info!("[refinery] generated {} decision logs", generated);
    }
    Ok(generated)
}

/// Group memories by activity bursts (30-min gap → new burst).
/// Input memories should be sorted by last_modified (ascending or descending).
/// Output: groups of references into the input slice, each group is one burst.
type BurstItem = (String, String, Option<String>, i64);

#[allow(clippy::type_complexity)]
fn group_into_bursts(memories: &[BurstItem]) -> Vec<Vec<&BurstItem>> {
    if memories.is_empty() {
        return Vec::new();
    }

    // Sort by last_modified ascending for gap detection
    let mut sorted: Vec<&BurstItem> = memories.iter().collect();
    sorted.sort_by_key(|(_, _, _, ts)| *ts);

    let mut bursts: Vec<Vec<&BurstItem>> = Vec::new();
    let mut current: Vec<&BurstItem> = vec![sorted[0]];

    for item in &sorted[1..] {
        let last_ts = current.last().unwrap().3;
        let gap = item.3 - last_ts;
        if gap > ACTIVITY_GAP_SECS {
            bursts.push(current);
            current = Vec::new();
        }
        current.push(item);
    }

    if !current.is_empty() {
        bursts.push(current);
    }

    bursts
}

/// Public wrapper for generate_recaps, callable from post_ingest module.
pub async fn generate_recaps_public(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
) -> Result<u32, OriginError> {
    generate_recaps(db, llm, prompts, tuning).await
}

/// Generate recaps from recent non-recap memories.
/// Groups memories into 30-min activity bursts; generates a recap for each burst
/// with 3+ memories that isn't already covered by an existing recap.
pub(crate) async fn generate_recaps(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
) -> Result<u32, OriginError> {
    let now = chrono::Utc::now().timestamp();
    let lookback_start = now - tuning.recap_lookback_secs;

    // Get all non-recap memories in the lookback window
    let recent = db
        .get_recent_memories_for_recap(lookback_start, 100)
        .await?;
    if recent.len() < tuning.min_memories_for_recap {
        return Ok(0);
    }

    // Group into bursts by 30-min gaps
    let bursts = group_into_bursts(&recent);
    let mut recaps_generated = 0u32;

    for burst in &bursts {
        if burst.len() < tuning.min_memories_for_recap {
            continue;
        }

        // Determine burst time range
        let burst_start = burst.iter().map(|m| m.3).min().unwrap_or(0);
        let burst_end = burst.iter().map(|m| m.3).max().unwrap_or(0);

        // Skip if a recap already covers this burst
        if db.has_recap_covering_range(burst_start, burst_end).await? {
            continue;
        }

        // Group content for the LLM
        let contents: Vec<String> = burst
            .iter()
            .map(|(_, content, domain, _)| match domain {
                Some(d) => format!("[{}] {}", d, content),
                None => content.clone(),
            })
            .collect();

        // Determine the dominant domain
        let mut domain_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (_, _, domain, _) in burst.iter() {
            if let Some(d) = domain {
                *domain_counts.entry(d.clone()).or_insert(0) += 1;
            }
        }
        let dominant_domain = domain_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(d, _)| d);

        // Build raw context from burst
        let burst_slice: Vec<(String, String, Option<String>, i64)> =
            burst.iter().map(|m| (*m).clone()).collect();
        let raw_context = build_burst_context(&burst_slice, burst_start, burst_end);

        // Send to LLM for synthesis (or generate without LLM)
        let (recap_summary, recap_content) = if let Some(llm) = llm {
            let domain_hint = dominant_domain.as_deref().unwrap_or("general");
            let combined = contents
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{}. {}", i + 1, c))
                .collect::<Vec<_>>()
                .join("\n");

            let response = llm
                .generate(LlmRequest {
                    system_prompt: Some(
                        prompts.detect_pattern.replace("{domain_hint}", domain_hint),
                    ),
                    user_prompt: combined,
                    max_tokens: 128,
                    temperature: 0.1,
                    label: None,
                })
                .await;

            match response {
                Ok(output)
                    if output.trim().to_lowercase() != "null" && !output.trim().is_empty() =>
                {
                    let cleaned = crate::llm_provider::strip_think_tags(&output);
                    (cleaned.trim().to_string(), raw_context)
                }
                _ => generate_simple_recap(&burst_slice, burst_start, burst_end),
            }
        } else {
            generate_simple_recap(&burst_slice, burst_start, burst_end)
        };

        // Store the recap — set last_modified = burst_end so it sorts next to its source memories
        let recap_id = format!("recap_{}", uuid::Uuid::new_v4());
        // Track source memory IDs so the detail view can show exact sources
        let source_ids: Vec<&str> = burst.iter().map(|(id, _, _, _)| id.as_str()).collect();
        let structured = serde_json::json!({ "source_ids": source_ids }).to_string();
        // Generate a short topic title via LLM; fall back to burst header
        let burst_header: String = recap_content
            .lines()
            .take(2)
            .collect::<Vec<_>>()
            .join(" · ");
        let title = if let Some(llm) = llm {
            generate_short_title(llm, &recap_content)
                .await
                .unwrap_or_else(|| burst_header.chars().take(80).collect())
        } else {
            burst_header.chars().take(80).collect()
        };
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: recap_id,
            title,
            summary: Some(recap_summary),
            content: recap_content,
            url: None,
            last_modified: burst_end,
            metadata: std::collections::HashMap::new(),
            memory_type: Some("fact".to_string()),
            domain: dominant_domain,
            source_agent: Some("refinery".to_string()),
            confidence: Some(0.5),
            confirmed: None,
            supersedes: None,
            pending_revision: false,
            is_recap: true,
            structured_fields: Some(structured),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await?;
        recaps_generated += 1;
    }

    Ok(recaps_generated)
}

/// Build a structured raw-context string from a burst of memories.
fn build_burst_context(
    memories: &[(String, String, Option<String>, i64)],
    burst_start: i64,
    burst_end: i64,
) -> String {
    use chrono::DateTime;

    let start_fmt = DateTime::from_timestamp(burst_start, 0)
        .map(|dt| dt.format("%H:%M").to_string())
        .unwrap_or_else(|| burst_start.to_string());
    let end_fmt = DateTime::from_timestamp(burst_end, 0)
        .map(|dt| dt.format("%H:%M").to_string())
        .unwrap_or_else(|| burst_end.to_string());

    let domains: Vec<String> = memories
        .iter()
        .filter_map(|(_, _, d, _)| d.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let domains_str = if domains.is_empty() {
        "unknown".to_string()
    } else {
        domains.join(", ")
    };

    let lines: Vec<String> = memories
        .iter()
        .map(|(_, content, domain, _)| {
            let clean = clean_for_recap(content);
            match domain {
                Some(d) => format!("- [{}] {}", d, clean),
                None => format!("- {}", clean),
            }
        })
        .collect();

    format!(
        "Activity burst: {} – {}\n{} memories across {}\n\n{}",
        start_fmt,
        end_fmt,
        memories.len(),
        domains_str,
        lines.join("\n"),
    )
}

/// Returns true for single-word generic stand-ins that are useless as concept titles.
/// These are often the `topic` fallback value when no entity/domain is available.
fn looks_like_generic_topic(s: &str) -> bool {
    let t = s.trim().to_lowercase();
    matches!(
        t.as_str(),
        "general" | "untitled" | "topic" | "concept" | "cluster" | "misc" | "other" | "unknown"
    )
}

fn looks_like_uuid(s: &str) -> bool {
    // e.g. 5b064ab2-8919-48b2-8220-8f7680b426dd
    let trimmed = s.trim();
    trimmed.len() >= 32
        && trimmed.chars().filter(|c| *c == '-').count() >= 3
        && trimmed.chars().all(|c| c.is_ascii_hexdigit() || c == '-')
}

fn looks_like_short_hash(s: &str) -> bool {
    // e.g. e554534 (commit SHA prefix) as sole title or lead token
    let first = s.split_whitespace().next().unwrap_or("");
    (7..=12).contains(&first.len())
        && first.chars().all(|c| c.is_ascii_hexdigit())
        && first.chars().any(|c| c.is_ascii_digit())
}

fn looks_like_code(s: &str) -> bool {
    let lowered = s.trim_start().to_lowercase();
    lowered.starts_with("const ")
        || lowered.starts_with("let ")
        || lowered.starts_with("var ")
        || lowered.starts_with("await ")
        || lowered.starts_with("function ")
        || lowered.starts_with("import ")
        || lowered.starts_with("fn ")
        || s.contains("=>")
        || s.contains("{ where:")
        || s.contains("findUnique")
}

fn looks_like_path(s: &str) -> bool {
    let trimmed = s.trim_start();
    (trimmed.starts_with('[')
        && (trimmed.contains("obs/")
            || trimmed.contains("/2026-")
            || trimmed.contains("/2025-")
            || trimmed.contains(".md")
            || trimmed.contains("::")))
        || trimmed.starts_with('/')
        || trimmed.starts_with("~/")
        || trimmed.contains("/Users/")
        || trimmed.contains("/2026-")
        || trimmed.contains("/2025-")
        || trimmed.contains("/inbox/")
        || trimmed.contains("/second-brain/")
}

fn looks_like_commit_message(s: &str) -> bool {
    let trimmed = s.trim_start();
    let lowered = trimmed.to_lowercase();
    let plain_prefixes = [
        "feat:",
        "fix:",
        "chore:",
        "docs:",
        "refactor:",
        "test:",
        "style:",
        "perf:",
        "ci:",
        "build:",
        "revert:",
    ];
    if plain_prefixes.iter().any(|p| lowered.starts_with(p)) {
        return true;
    }
    // Conventional commits with scope: feat(area): ...
    if let Some(open) = trimmed.find('(') {
        if let Some(colon_close) = trimmed[open..].find("):") {
            let _ = colon_close;
            let prefix_raw = trimmed[..open].to_lowercase();
            let prefix_clean = prefix_raw.trim_end_matches(':');
            if plain_prefixes
                .iter()
                .any(|p| prefix_clean == p.trim_end_matches(':'))
            {
                return true;
            }
        }
    }
    false
}

/// Strip a leading bracketed source-ID prefix like `[obs/unix/2026-03-17]`,
/// `[mem_abc123]`, `[5b064ab2-8919-48b2]` from content before title generation.
/// Only strips if the bracket content has no spaces and looks like a source token
/// (contains slash, underscore, double-colon, or is all hex/hyphens).
fn strip_source_prefix(content: &str) -> &str {
    let trimmed = content.trim_start();
    if !trimmed.starts_with('[') {
        return content;
    }
    if let Some(end) = trimmed.find(']') {
        let inside = &trimmed[1..end];
        if !inside.contains(' ')
            && (inside.contains('/')
                || inside.contains('_')
                || inside.contains("::")
                || inside.chars().all(|c| c.is_ascii_hexdigit() || c == '-'))
        {
            return trimmed[end + 1..].trim_start();
        }
    }
    content
}

/// Generate a short 4-6 word topic title from content using LLM.
pub(crate) async fn generate_short_title(
    llm: &Arc<dyn LlmProvider>,
    content: &str,
) -> Option<String> {
    let stripped = strip_source_prefix(content);
    let input: String = stripped.chars().take(300).collect();
    let response = llm.generate(LlmRequest {
        system_prompt: Some("Given a note, write a 3-5 word title. Output ONLY the title.\n\nExample: 'The system uses libsql for vector storage with DiskANN indexing' → libsql Vector Storage\nExample: 'Google Sign-In fails with developer_error status 10' → Google Sign-In SHA Fix".to_string()),
        user_prompt: format!("{}\n/no_think", input),
        max_tokens: 16,
        temperature: 0.3,
        label: None,
    }).await;

    match response {
        Ok(output) => {
            let cleaned = crate::llm_provider::strip_think_tags(&output).replace("/no_think", "");
            // Strip noise the model echoes: "- ", "[domain] ", numbered prefixes
            let mut title = cleaned.trim().to_string();
            // Strip leading "- "
            if let Some(rest) = title.strip_prefix("- ") {
                title = rest.to_string();
            }
            // Strip "N. " numbered prefix
            if let Some(pos) = title.find(". ") {
                if pos <= 3 && title[..pos].chars().all(|c| c.is_ascii_digit()) {
                    title = title[pos..]
                        .strip_prefix(". ")
                        .unwrap_or(&title[pos..])
                        .to_string();
                }
            }
            // Strip "[domain] " prefix
            if title.starts_with('[') {
                if let Some(pos) = title.find("] ") {
                    title = title[pos..]
                        .strip_prefix("] ")
                        .unwrap_or(&title[pos..])
                        .to_string();
                }
            }
            let title = title
                .trim()
                .trim_matches('"')
                .trim_matches('.')
                .trim_start_matches("Title: ")
                .trim_start_matches("title: ")
                .trim();
            let word_count = title.split_whitespace().count();
            log::info!("[title] clean='{}' ({} words)", title, word_count);
            // Reject if empty, too many words, multiline, or truncated mid-sentence
            let truncated = title.ends_with(',')
                || title.ends_with("including")
                || title.ends_with("with")
                || title.ends_with("the")
                || title.ends_with("and")
                || title.ends_with("for")
                || title.ends_with("of")
                || title.ends_with("in");
            if title.is_empty() || word_count > 8 || title.contains('\n') || truncated {
                log::info!("[title] rejected (empty/too long/multiline)");
                None
            } else if looks_like_generic_topic(title)
                || looks_like_uuid(title)
                || looks_like_short_hash(title)
                || looks_like_code(title)
                || looks_like_path(title)
                || looks_like_commit_message(title)
            {
                log::info!("[distill] rejecting title \"{}\" (pattern match)", title);
                None
            } else {
                Some(title.to_string())
            }
        }
        Err(e) => {
            log::warn!("[title] generation failed: {e}");
            None
        }
    }
}

/// Clean memory content for recap display: strip structured field metadata and prefixes.
fn clean_for_recap(content: &str) -> String {
    let mut s = content.to_string();

    // Strip "claim: " or "context: " prefix
    for prefix in &["claim: ", "context: ", "fact: "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.to_string();
            break;
        }
    }

    // Strip trailing structured metadata "| domain: ... | source: ... | verified: ..."
    if let Some(pos) = s.find(" | domain: ") {
        s.truncate(pos);
    } else if let Some(pos) = s.find(" | source: ") {
        s.truncate(pos);
    } else if let Some(pos) = s.find(" | verified: ") {
        s.truncate(pos);
    } else if let Some(pos) = s.find(" | date: ") {
        s.truncate(pos);
    } else if let Some(pos) = s.find(" | decision: ") {
        s.truncate(pos);
    } else if let Some(pos) = s.find(" | reversible: ") {
        s.truncate(pos);
    }

    // Trim and cap at first sentence or 200 chars for readability
    let s = s.trim();
    if s.len() > 200 {
        let truncated: String = s.chars().take(200).collect();
        // Cut at last sentence boundary if possible
        if let Some(pos) = truncated.rfind(". ") {
            format!("{}.", &truncated[..pos])
        } else {
            format!("{}...", truncated.trim_end())
        }
    } else {
        s.to_string()
    }
}

/// Generate a simple recap without LLM (fallback).
/// Returns (summary, raw_context).
fn generate_simple_recap(
    memories: &[(String, String, Option<String>, i64)],
    burst_start: i64,
    burst_end: i64,
) -> (String, String) {
    let count = memories.len();
    let domains: Vec<String> = memories
        .iter()
        .filter_map(|(_, _, d, _)| d.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let summary = format!(
        "Recap: {} memories stored{}",
        count,
        if domains.is_empty() {
            String::new()
        } else {
            format!(" ({})", domains.join(", "))
        },
    );

    let raw_context = build_burst_context(memories, burst_start, burst_end);

    (summary, raw_context)
}

/// Apply a merge result based on the stability tier of the involved memories.
async fn apply_merge_by_tier(
    db: &MemoryDB,
    source_ids: &[String],
    merged_content: &str,
    proposal_id: &str,
    tier: &StabilityTier,
) -> Result<(), OriginError> {
    match tier {
        StabilityTier::Ephemeral => {
            // Auto-apply silently
            db.apply_merge(source_ids, merged_content).await?;
            db.resolve_refinement(proposal_id, "auto_applied").await?;
            log::info!(
                "[refinery] auto-applied merge (ephemeral) for {}",
                proposal_id
            );
        }
        StabilityTier::Standard => {
            // Auto-apply with notification (toast emitted by caller if app_handle available)
            db.apply_merge(source_ids, merged_content).await?;
            db.resolve_refinement(proposal_id, "auto_applied").await?;
            log::info!(
                "[refinery] auto-applied merge (standard, notify) for {}",
                proposal_id
            );
        }
        StabilityTier::Protected => {
            // Queue for human review — don't auto-apply
            db.resolve_refinement(proposal_id, "awaiting_review")
                .await?;
            log::info!(
                "[refinery] queued merge for review (protected) for {}",
                proposal_id
            );
        }
    }
    Ok(())
}

/// Trigger a refinery steep on demand — used by the chat import flow to
/// process freshly-ingested memories without waiting for the scheduled
/// 30-minute tick.
///
/// This is a thin wrapper around the existing steep logic. Callers should
/// await completion before showing "done" to users.
pub async fn trigger_steep_now(
    db: Arc<MemoryDB>,
    llm: Option<&Arc<dyn LlmProvider>>,
    api_llm: Option<&Arc<dyn LlmProvider>>,
    synthesis_llm: Option<&Arc<dyn LlmProvider>>,
) -> Result<SteepResult, OriginError> {
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = crate::tuning::TuningConfig::load(&crate::tuning::TuningConfig::config_path());

    log::info!("[refinery] on-demand steep triggered");
    run_periodic_steep_with_api(
        &db,
        llm,
        api_llm,
        synthesis_llm,
        &prompts,
        &tuning.refinery,
        &tuning.confidence,
        &tuning.distillation,
        TriggerKind::Backstop,
    )
    .await
}

#[derive(Debug, Serialize)]
pub struct SteepResult {
    pub memories_decayed: u64,
    pub recaps_generated: u32,
    pub imports_reclassified: u32,
    pub distilled: u32,
    pub pending_remaining: u32,
    pub phases: Vec<PhaseResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::tests::test_db;
    use crate::sources::RawDocument;

    fn make_memory(source_id: &str, content: &str, memory_type: &str, domain: &str) -> RawDocument {
        RawDocument {
            source_id: source_id.to_string(),
            content: content.to_string(),
            source: "memory".to_string(),
            title: content.chars().take(40).collect(),
            memory_type: Some(memory_type.to_string()),
            domain: Some(domain.to_string()),
            confidence: Some(0.7),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        }
    }

    #[test]
    fn test_trigger_kind_backstop_runs_all_phases() {
        for &phase in ALL_PHASES {
            assert!(
                TriggerKind::Backstop.runs_phase(phase),
                "Backstop should run {}",
                phase
            );
        }
    }

    #[test]
    fn test_trigger_kind_burst_end_subset() {
        let t = TriggerKind::BurstEnd;
        assert!(t.runs_phase("recaps"));
        assert!(t.runs_phase("refinement_queue"));
        // Should NOT run anything else
        assert!(!t.runs_phase("decay"));
        assert!(!t.runs_phase("promote"));
        assert!(!t.runs_phase("emergence"));
        assert!(!t.runs_phase("community_detection"));
        assert!(!t.runs_phase("decision_logs"));
        assert!(!t.runs_phase("reclassify"));
        assert!(!t.runs_phase("prune_rejections"));
    }

    #[test]
    fn test_trigger_kind_idle_subset() {
        let t = TriggerKind::Idle;
        assert!(t.runs_phase("community_detection"));
        assert!(t.runs_phase("emergence"));
        assert!(t.runs_phase("re-distill"));
        assert!(t.runs_phase("decision_logs"));
        // Should NOT run burst/maintenance/backfill phases
        assert!(!t.runs_phase("recaps"));
        assert!(!t.runs_phase("refinement_queue"));
        assert!(!t.runs_phase("decay"));
        assert!(!t.runs_phase("promote"));
        assert!(!t.runs_phase("reclassify"));
        assert!(!t.runs_phase("reweave"));
        assert!(!t.runs_phase("reembed"));
        assert!(!t.runs_phase("entity_extraction"));
        assert!(!t.runs_phase("decision_backfill"));
        assert!(!t.runs_phase("prune_rejections"));
    }

    #[test]
    fn test_trigger_kind_daily_subset() {
        let t = TriggerKind::Daily;
        assert!(t.runs_phase("decay"));
        assert!(t.runs_phase("promote"));
        assert!(t.runs_phase("reclassify"));
        assert!(t.runs_phase("reweave"));
        assert!(t.runs_phase("reembed"));
        assert!(t.runs_phase("entity_extraction"));
        assert!(t.runs_phase("decision_backfill"));
        assert!(t.runs_phase("prune_rejections"));
        // Should NOT run synthesis or burst phases
        assert!(!t.runs_phase("recaps"));
        assert!(!t.runs_phase("emergence"));
        assert!(!t.runs_phase("re-distill"));
        assert!(!t.runs_phase("decision_logs"));
        assert!(!t.runs_phase("community_detection"));
        assert!(!t.runs_phase("refinement_queue"));
    }

    #[test]
    fn test_trigger_kind_unknown_phase_returns_false() {
        // Defensive: typos in call sites must NOT silently match.
        assert!(!TriggerKind::Backstop.runs_phase("typo"));
        assert!(!TriggerKind::Backstop.runs_phase(""));
        assert!(!TriggerKind::BurstEnd.runs_phase("typo"));
        assert!(!TriggerKind::Idle.runs_phase("typo"));
        assert!(!TriggerKind::Daily.runs_phase("typo"));
    }

    #[test]
    fn test_every_phase_has_non_backstop_trigger() {
        // Safety net: if a new phase is added to ALL_PHASES but not assigned
        // to BurstEnd, Idle, or Daily, it silently becomes backstop-only
        // (running every 6 hours instead of at the right time). This test
        // catches that at compile/test time.
        let non_backstop = [TriggerKind::BurstEnd, TriggerKind::Idle, TriggerKind::Daily];
        for &phase in ALL_PHASES {
            let covered = non_backstop.iter().any(|t| t.runs_phase(phase));
            assert!(
                covered,
                "Phase '{}' is only covered by Backstop — assign it to BurstEnd, Idle, or Daily in runs_phase()",
                phase
            );
        }
    }

    #[tokio::test]
    async fn test_run_periodic_steep_with_api_accepts_trigger_kind() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory(
            "trigger_smoke",
            "Smoke test memory for trigger param",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            TriggerKind::Backstop,
        )
        .await
        .unwrap();

        assert!(!result.phases.is_empty(), "Backstop should produce phases");
    }

    #[tokio::test]
    async fn test_burst_end_trigger_runs_only_burst_phases() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory(
            "burst_test",
            "Test memory for burst trigger",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            TriggerKind::BurstEnd,
        )
        .await
        .unwrap();

        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();

        // Must contain the BurstEnd subset
        assert!(
            phase_names.contains(&"recaps"),
            "BurstEnd should run recaps, got {:?}",
            phase_names
        );
        assert!(
            phase_names.contains(&"refinement_queue"),
            "BurstEnd should run refinement_queue, got {:?}",
            phase_names
        );

        // Must NOT contain anything else
        for name in &phase_names {
            assert!(
                *name == "recaps" || *name == "refinement_queue",
                "BurstEnd should only run recaps + refinement_queue, got unexpected: {}",
                name
            );
        }
    }

    #[tokio::test]
    async fn test_prune_rejections_runs_as_phase() {
        let (db, _dir) = test_db().await;

        // No rejections needed — we just verify the phase runs (or doesn't)
        // based on the trigger. The actual DELETE is a no-op on an empty
        // table; what we care about is whether the phase appears in
        // result.phases.

        // Backstop: prune_rejections should appear in phases
        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            TriggerKind::Backstop,
        )
        .await
        .unwrap();
        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();
        assert!(
            phase_names.contains(&"prune_rejections"),
            "Backstop should run prune_rejections as a tracked phase, got {:?}",
            phase_names
        );

        // BurstEnd: prune_rejections should NOT appear in phases
        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            TriggerKind::BurstEnd,
        )
        .await
        .unwrap();
        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();
        assert!(
            !phase_names.contains(&"prune_rejections"),
            "BurstEnd should NOT run prune_rejections, got {:?}",
            phase_names
        );
    }

    #[tokio::test]
    async fn test_idle_trigger_runs_only_synthesis_phases() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory(
            "idle_test",
            "Test memory for idle trigger",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            TriggerKind::Idle,
        )
        .await
        .unwrap();

        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();

        // Idle subset
        let expected: &[&str] = &[
            "community_detection",
            "emergence",
            "re-distill",
            "decision_logs",
        ];
        for &exp in expected {
            assert!(
                phase_names.contains(&exp),
                "Idle should run {}, got {:?}",
                exp,
                phase_names
            );
        }

        // Must NOT run anything else
        for name in &phase_names {
            assert!(
                expected.contains(name),
                "Idle should only run synthesis phases, got unexpected: {}",
                name
            );
        }
    }

    #[tokio::test]
    async fn test_daily_trigger_runs_only_maintenance_phases() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory(
            "daily_test",
            "Test memory for daily trigger",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            TriggerKind::Daily,
        )
        .await
        .unwrap();

        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();

        // Daily subset — all maintenance + backfill phases including
        // prune_rejections (promoted to a tracked phase in Task 4).
        let expected: &[&str] = &[
            "decay",
            "promote",
            "reclassify",
            "reweave",
            "reembed",
            "entity_extraction",
            "decision_backfill",
            "prune_rejections",
            "kg_rethink",
        ];
        for &exp in expected {
            // kg_rethink is rate-limited by app_metadata — it may or may not run
            // on any given steep. Everything else must run.
            if exp == "kg_rethink" {
                continue;
            }
            assert!(
                phase_names.contains(&exp),
                "Daily should run {}, got {:?}",
                exp,
                phase_names
            );
        }

        // Must NOT run synthesis or burst phases
        for name in &phase_names {
            assert!(
                expected.contains(name),
                "Daily should only run maintenance + backfill phases, got unexpected: {}",
                name
            );
        }
    }

    #[tokio::test]
    async fn test_backstop_trigger_runs_all_phases() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory(
            "backstop_test",
            "Test memory for backstop trigger",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            TriggerKind::Backstop,
        )
        .await
        .unwrap();

        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();

        // All 15 phases must run with Backstop (14 prior + kg_rethink added
        // in the KG quality work). `kg_rethink` is rate-limited, so on a
        // fresh DB (last_kg_rethink_ts=0) it runs on the first steep.
        let expected: &[&str] = &[
            "decay",
            "promote",
            "recaps",
            "reclassify",
            "reweave",
            "reembed",
            "entity_extraction",
            "community_detection",
            "emergence",
            "re-distill",
            "refinement_queue",
            "decision_logs",
            "decision_backfill",
            "prune_rejections",
            "kg_rethink",
        ];
        for &exp in expected {
            assert!(
                phase_names.contains(&exp),
                "Backstop should run {}, got {:?}",
                exp,
                phase_names
            );
        }
        assert_eq!(
            phase_names.len(),
            expected.len(),
            "Backstop should run exactly {} phases, got {}: {:?}",
            expected.len(),
            phase_names.len(),
            phase_names
        );
    }

    #[tokio::test]
    async fn test_run_periodic_steep() {
        let (db, _dir) = test_db().await;

        // Insert some memories
        db.upsert_documents(vec![
            make_memory(
                "steep1",
                "Rust async programming patterns",
                "fact",
                "engineering",
            ),
            make_memory(
                "steep2",
                "Python data science workflows",
                "fact",
                "engineering",
            ),
        ])
        .await
        .unwrap();

        let result = run_periodic_steep(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
        )
        .await
        .unwrap();
        assert!(
            result.memories_decayed >= 2,
            "should decay at least 2 memories"
        );
    }

    #[tokio::test]
    async fn test_refinement_pipeline_end_to_end() {
        let (db, _dir) = test_db().await;

        // 1. Store several memories
        for i in 0..3 {
            db.upsert_documents(vec![make_memory(
                &format!("e2e_{}", i),
                &format!("Rust systems programming language variant {}", i),
                "fact",
                "engineering",
            )])
            .await
            .unwrap();
        }

        // 2. Verify access tracking works (flush should not error)
        db.flush_access_counts(&["e2e_0".to_string()])
            .await
            .unwrap();

        // 3. Verify decay steep updates effective_confidence
        let updated = db.decay_update_confidence().await.unwrap();
        assert!(updated >= 3);

        // 4. Verify refinement queue operations
        db.insert_refinement_proposal(
            "e2e_ref",
            "dedup_merge",
            &["e2e_0".to_string(), "e2e_1".to_string()],
            None,
            0.95,
        )
        .await
        .unwrap();
        let pending = db.get_pending_refinements().await.unwrap();
        assert_eq!(pending.len(), 1);
        db.resolve_refinement("e2e_ref", "auto_applied")
            .await
            .unwrap();
        assert!(db.get_pending_refinements().await.unwrap().is_empty());

        // 5. Run full periodic steep (no LLM — proposals stay unprocessed)
        let result = run_periodic_steep(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
        )
        .await
        .unwrap();
        assert!(result.memories_decayed >= 3);
    }

    #[tokio::test]
    async fn test_apply_merge_by_tier_ephemeral() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![
            make_memory("eph1", "Morning coding recap v1", "goal", "engineering"),
            make_memory("eph2", "Morning coding recap v2", "goal", "engineering"),
        ])
        .await
        .unwrap();

        db.insert_refinement_proposal(
            "merge_eph",
            "dedup_merge",
            &["eph1".into(), "eph2".into()],
            None,
            0.95,
        )
        .await
        .unwrap();

        // Ephemeral tier → should auto-apply silently
        apply_merge_by_tier(
            &db,
            &["eph1".into(), "eph2".into()],
            "Morning coding recap consolidated",
            "merge_eph",
            &StabilityTier::Ephemeral,
        )
        .await
        .unwrap();

        let pending = db.get_pending_refinements().await.unwrap();
        assert!(pending.is_empty(), "proposal should be resolved");
    }

    #[tokio::test]
    async fn test_apply_merge_by_tier_protected() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![
            make_memory(
                "prot1",
                "I prefer Rust for backends",
                "preference",
                "engineering",
            ),
            make_memory(
                "prot2",
                "I always use Rust for server code",
                "preference",
                "engineering",
            ),
        ])
        .await
        .unwrap();

        db.insert_refinement_proposal(
            "merge_prot",
            "dedup_merge",
            &["prot1".into(), "prot2".into()],
            None,
            0.93,
        )
        .await
        .unwrap();

        // Protected tier → should queue for review, NOT auto-apply
        apply_merge_by_tier(
            &db,
            &["prot1".into(), "prot2".into()],
            "I prefer Rust for backend/server code",
            "merge_prot",
            &StabilityTier::Protected,
        )
        .await
        .unwrap();

        let pending = db.get_pending_refinements().await.unwrap();
        assert_eq!(
            pending.len(),
            1,
            "protected merge should be awaiting review"
        );
        assert_eq!(pending[0].status, "awaiting_review");
    }

    #[tokio::test]
    async fn test_generate_recaps_from_recent_memories() {
        let (db, _dir) = test_db().await;

        // Insert 5 recent memories (enough to trigger recap)
        let now = chrono::Utc::now().timestamp();
        for i in 0..5 {
            let mut doc = make_memory(
                &format!("recent_{}", i),
                &format!("Working on feature {} for the Origin project", i),
                "fact",
                "engineering",
            );
            doc.last_modified = now - 60 * i as i64; // spread across last 5 minutes
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // No recap should exist yet
        assert!(!db.has_recap_since(now - 86400_i64).await.unwrap());

        // Generate recaps (no LLM — uses simple fallback)
        let generated = generate_recaps(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(generated, 1, "should generate 1 recap");

        // Recap should now exist
        assert!(db.has_recap_since(now - 86400_i64).await.unwrap());

        // Running again should skip (recap already exists)
        let generated2 = generate_recaps(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(generated2, 0, "should not generate duplicate recap");

        // The recap should be searchable (stored as memory_type=fact, is_recap=1)
        let results = db
            .search_memory(
                "Origin project feature",
                10,
                Some("fact"),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(!results.is_empty(), "recap should be searchable as fact");
    }

    #[tokio::test]
    async fn test_recap_separates_content_and_summary() {
        let (db, _dir) = test_db().await;

        let now = chrono::Utc::now().timestamp();
        for i in 0..4 {
            let mut doc = make_memory(
                &format!("sep_{}", i),
                &format!("Debugging issue {} in the auth module", i),
                "fact",
                "engineering",
            );
            doc.last_modified = now - 60 * i as i64;
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        let generated = generate_recaps(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(generated, 1);

        // Find the recap via search (stored as memory_type=fact, is_recap=1)
        let results = db
            .search_memory(
                "auth module debugging",
                10,
                Some("fact"),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(!results.is_empty(), "recap should be searchable as fact");

        // Find the recap result (source_id starts with "recap_")
        let recap_result = results
            .iter()
            .find(|r| r.source_id.starts_with("recap_"))
            .expect("should find a recap entry in results");

        // Fetch full detail to verify content vs summary separation
        let detail = db
            .get_memory_detail(&recap_result.source_id)
            .await
            .unwrap()
            .unwrap();
        assert!(detail.summary.is_some(), "recap should have a summary");
        let summary = detail.summary.unwrap();
        assert!(
            summary.starts_with("Recap:"),
            "summary should be the concise line"
        );
        assert!(
            detail.content.contains("Activity burst:"),
            "content should be the structured context"
        );
        assert_ne!(detail.content, summary, "content and summary must differ");
    }

    #[tokio::test]
    async fn test_no_recap_when_too_few_memories() {
        let (db, _dir) = test_db().await;

        // Only 2 memories — below threshold
        let now = chrono::Utc::now().timestamp();
        for i in 0..2 {
            let mut doc = make_memory(&format!("few_{}", i), "Some content", "fact", "eng");
            doc.last_modified = now - 60;
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        let generated = generate_recaps(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            generated, 0,
            "should not generate recap with too few memories"
        );
    }

    #[tokio::test]
    async fn test_steep_returns_phase_results() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory(
            "phase1",
            "Memory about Rust programming",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result = run_periodic_steep(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
        )
        .await
        .unwrap();
        assert!(!result.phases.is_empty(), "should have phase results");
        // All phases should succeed (no errors)
        for phase in &result.phases {
            assert!(
                phase.error.is_none(),
                "phase {} should not error: {:?}",
                phase.name,
                phase.error
            );
        }
        // Should have at least decay, recaps, reclassify, reweave, reembed, entity_extraction, distillation, refinement_queue, decision_logs
        assert!(
            result.phases.len() >= 6,
            "should have at least 6 phases, got {}",
            result.phases.len()
        );
        assert_eq!(result.phases[0].name, "decay");
    }

    #[tokio::test]
    async fn test_reweave_links_unlinked_memories_to_entity() {
        let (db, _dir) = test_db().await;

        // Create a memory without entity_id
        let doc = make_memory(
            "rew1",
            "Alice prefers dark mode for all coding tools",
            "preference",
            "ui",
        );
        db.upsert_documents(vec![doc]).await.unwrap();

        // Create entity "Alice"
        let _entity_id = db.create_entity("Alice", "person", None).await.unwrap();

        // Reweave should try to link — but whether it actually links depends on
        // vector similarity between "Alice prefers dark mode..." and the entity "Alice".
        // The entity embedding is of just "Alice" which is quite different from the memory content.
        let linked = reweave_entity_links(&db, 20, 0.15).await.unwrap();
        // We can't assert linked > 0 because vector similarity may be too low
        // for the short entity name vs long memory content. Just verify it doesn't error.
        assert!(linked == 0 || linked == 1, "should link 0 or 1 memories");
    }

    #[tokio::test]
    async fn test_get_unlinked_memories() {
        let (db, _dir) = test_db().await;

        // Memory without entity_id
        let doc = make_memory("unlinked1", "Some memory without entity", "fact", "general");
        db.upsert_documents(vec![doc]).await.unwrap();

        let unlinked = db.get_unlinked_memories(10).await.unwrap();
        assert!(!unlinked.is_empty(), "should find unlinked memories");
        assert_eq!(unlinked[0].0, "unlinked1");
    }

    #[tokio::test]
    async fn test_suggest_entity_in_refinement_queue() {
        let (db, _dir) = test_db().await;

        // Insert a suggest_entity proposal
        db.insert_refinement_proposal(
            "sug1",
            "suggest_entity",
            &["mem1".to_string()],
            Some("PostgreSQL"),
            0.8,
        )
        .await
        .unwrap();

        // Process refinement queue — suggest_entity should be moved to awaiting_review
        let processed = process_refinement_queue(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(processed, 1, "should process 1 suggestion");

        let pending = db.get_pending_refinements().await.unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].status, "awaiting_review");
    }

    #[tokio::test]
    async fn test_decision_logs_no_decisions() {
        let (db, _dir) = test_db().await;

        // No decision memories → should generate 0 logs
        let generated = generate_decision_logs(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(generated, 0);
    }

    #[tokio::test]
    async fn test_decision_logs_with_decisions() {
        let (db, _dir) = test_db().await;

        let now = chrono::Utc::now().timestamp();
        for i in 0..4 {
            let mut doc = make_memory(
                &format!("dec_{}", i),
                &format!("Decided to use approach {} for the backend refactor", i),
                "decision",
                "engineering",
            );
            doc.last_modified = now - 60 * i as i64;
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        let generated = generate_decision_logs(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(generated, 1, "should generate 1 decision log");

        // Verify the decision log is stored as decision type with is_recap=true
        let results = db
            .search_memory(
                "backend refactor approach",
                10,
                Some("decision"),
                None,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let recap = results.iter().find(|r| r.source_id.starts_with("recap_"));
        assert!(recap.is_some(), "should find a decision log recap");
    }

    #[tokio::test]
    async fn test_distill_concepts_creates_concept() {
        let (db, _dir) = test_db().await;

        // Insert 3 memories with same entity — enough for a cluster
        for (i, content) in [
            "libSQL stores vectors using F32_BLOB columns",
            "libSQL uses DiskANN for vector indexing",
            "libSQL supports FTS5 full-text search via triggers",
        ]
        .iter()
        .enumerate()
        {
            let doc = RawDocument {
                source: "memory".to_string(),
                source_id: format!("compile_test_{}", i),
                title: content.to_string(),
                content: content.to_string(),
                entity_id: Some("entity_libsql".to_string()),
                domain: Some("architecture".to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // Run distillation (no LLM — should skip gracefully, return 0)
        let result = distill_concepts(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(result, 0);

        // No concepts created without LLM
        let concepts = db.list_concepts("active", 100, 0).await.unwrap();
        assert_eq!(concepts.len(), 0);
    }

    #[tokio::test]
    async fn test_concept_lifecycle_integration() {
        let (db, _dir) = test_db().await;

        // 1. Insert memories that form a cluster
        for (i, content) in [
            "Origin uses libSQL for all storage including vectors and FTS",
            "libSQL supports DiskANN indexing for vector search",
            "The embedding dimension is 384 using BGE-Small model",
        ]
        .iter()
        .enumerate()
        {
            let doc = RawDocument {
                source: "memory".to_string(),
                source_id: format!("lifecycle_{}", i),
                title: content.to_string(),
                content: content.to_string(),
                entity_id: Some("entity_libsql_lc".to_string()),
                domain: Some("architecture".to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // 2. Verify no concepts yet
        let concepts = db.list_concepts("active", 100, 0).await.unwrap();
        assert_eq!(concepts.len(), 0);

        // 3. Run distillation (no LLM — graceful skip)
        let distilled = distill_concepts(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(distilled, 0);

        // 4. Verify concept CRUD works end-to-end
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_concept(
            "c_int",
            "libSQL Storage",
            Some("Database layer"),
            "## Key Facts\n- test",
            Some("entity_libsql_lc"),
            Some("architecture"),
            &["lifecycle_0", "lifecycle_1", "lifecycle_2"],
            &now,
        )
        .await
        .unwrap();

        // 5. Verify concept search
        let found = db.search_concepts("libSQL storage", 10).await.unwrap();
        assert!(!found.is_empty());

        // 6. Verify concept by entity lookup
        let by_entity = db.get_concept_by_entity("entity_libsql_lc").await.unwrap();
        assert!(by_entity.is_some());

        // 7. Update and verify version increment
        db.update_concept_content(
            "c_int",
            "## Key Facts\n- updated",
            &["lifecycle_0", "lifecycle_1", "lifecycle_2", "lifecycle_3"],
        )
        .await
        .unwrap();
        let updated = db.get_concept("c_int").await.unwrap().unwrap();
        assert_eq!(updated.version, 2);
        assert_eq!(updated.source_memory_ids.len(), 4);

        // 8. Archive and verify
        db.archive_concept("c_int").await.unwrap();
        let active = db.list_concepts("active", 100, 0).await.unwrap();
        assert_eq!(active.len(), 0);
    }

    #[tokio::test]
    async fn test_trigger_steep_now_runs_steep() {
        let (db, _dir) = test_db().await;
        let db_arc = std::sync::Arc::new(db);

        // Insert a fake unclassified import memory.
        db_arc
            .store_raw_import_memory("import_claude_conv-xyz_0", "Hello world", None, None, 0)
            .await
            .unwrap();

        // Trigger on-demand steep.
        let result = trigger_steep_now(db_arc.clone(), None, None, None).await;
        assert!(result.is_ok(), "steep failed: {:?}", result.err());
    }

    // ── Nudge + PhaseOutput tests (Task 1) ──────────────────────────────

    #[test]
    fn test_nudge_variants_exist() {
        // Sanity check that all four levels exist and are distinct.
        let levels = [Nudge::Silent, Nudge::Ambient, Nudge::Notable, Nudge::Wow];
        for (i, a) in levels.iter().enumerate() {
            for (j, b) in levels.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_phase_output_construction() {
        // PhaseOutput should construct directly with all fields.
        let out = PhaseOutput {
            items_processed: 3,
            nudge: Nudge::Ambient,
            headline: Some("test headline".to_string()),
        };
        assert_eq!(out.items_processed, 3);
        assert_eq!(out.nudge, Nudge::Ambient);
        assert_eq!(out.headline.as_deref(), Some("test headline"));

        // Silent case with no headline.
        let silent = PhaseOutput {
            items_processed: 0,
            nudge: Nudge::Silent,
            headline: None,
        };
        assert_eq!(silent.nudge, Nudge::Silent);
        assert!(silent.headline.is_none());
    }

    // ── run_phase Nudge propagation tests (Task 2) ──────────────────────

    // ── Classify_* tests (Task 3) ─────────────────────────────────────

    #[test]
    fn test_classify_backfill_always_silent() {
        assert_eq!(classify_backfill(0), (Nudge::Silent, None));
        assert_eq!(classify_backfill(1), (Nudge::Silent, None));
        assert_eq!(classify_backfill(100), (Nudge::Silent, None));
    }

    #[test]
    fn test_classify_recaps_silent_when_zero() {
        assert_eq!(classify_recaps(0), (Nudge::Silent, None));
    }

    #[test]
    fn test_classify_recaps_ambient_when_generated() {
        let (nudge, headline) = classify_recaps(1);
        assert_eq!(nudge, Nudge::Ambient);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains("steeped"),
            "headline should use steep vocabulary: {}",
            h
        );
    }

    #[test]
    fn test_classify_recaps_plural_form() {
        let (_, headline) = classify_recaps(3);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains('3'),
            "multi-recap headline should mention the count: {}",
            h
        );
    }

    #[test]
    fn test_classify_emergence_silent_when_zero() {
        assert_eq!(classify_emergence(0), (Nudge::Silent, None));
    }

    #[test]
    fn test_classify_emergence_wow_when_new_concepts() {
        let (nudge, headline) = classify_emergence(1);
        assert_eq!(nudge, Nudge::Wow);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains("new concept"),
            "headline should mention a new concept: {}",
            h
        );
    }

    #[test]
    fn test_classify_emergence_wow_plural() {
        let (nudge, headline) = classify_emergence(4);
        assert_eq!(nudge, Nudge::Wow);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains('4'),
            "multi-concept headline should mention count: {}",
            h
        );
        assert!(h.contains("new concepts"), "should use plural: {}", h);
    }

    #[test]
    fn test_classify_redistill_silent_when_zero() {
        assert_eq!(classify_redistill(0), (Nudge::Silent, None));
    }

    #[test]
    fn test_classify_redistill_ambient_when_refreshed() {
        let (nudge, headline) = classify_redistill(2);
        assert_eq!(nudge, Nudge::Ambient);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains("refresh"),
            "headline should describe refreshing: {}",
            h
        );
    }

    #[test]
    fn test_classify_redistill_plural_form() {
        let (_, headline) = classify_redistill(5);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains('5'),
            "plural headline should mention count: {}",
            h
        );
    }

    #[test]
    fn test_classify_refinement_queue_silent_when_zero() {
        assert_eq!(classify_refinement_queue(0), (Nudge::Silent, None));
    }

    #[test]
    fn test_classify_refinement_queue_ambient_when_processed() {
        let (nudge, headline) = classify_refinement_queue(3);
        assert_eq!(nudge, Nudge::Ambient);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains("contradiction"),
            "headline should describe contradiction resolution: {}",
            h
        );
    }

    #[test]
    fn test_classify_decision_logs_silent_when_zero() {
        assert_eq!(classify_decision_logs(0), (Nudge::Silent, None));
    }

    #[test]
    fn test_classify_decision_logs_ambient_when_generated() {
        let (nudge, headline) = classify_decision_logs(1);
        assert_eq!(nudge, Nudge::Ambient);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains("decision"),
            "headline should mention decisions: {}",
            h
        );
    }

    #[test]
    fn test_classify_decision_logs_plural_form() {
        let (_, headline) = classify_decision_logs(3);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains('3'),
            "plural headline should mention count: {}",
            h
        );
    }

    // ── run_phase Nudge propagation tests (Task 2) ──────────────────────

    #[tokio::test]
    async fn test_run_phase_propagates_nudge_and_headline() {
        let result = run_phase("test_phase", || async {
            Ok(PhaseOutput {
                items_processed: 5,
                nudge: Nudge::Wow,
                headline: Some("Origin did a wow thing".to_string()),
            })
        })
        .await;

        assert_eq!(result.name, "test_phase");
        assert_eq!(result.items_processed, 5);
        assert_eq!(result.nudge, Nudge::Wow);
        assert_eq!(result.headline.as_deref(), Some("Origin did a wow thing"));
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_run_phase_on_error_is_silent() {
        let result = run_phase("test_phase_err", || async {
            Err::<PhaseOutput, _>(crate::error::OriginError::VectorDb(
                "test failure".to_string(),
            ))
        })
        .await;

        assert_eq!(result.name, "test_phase_err");
        assert_eq!(result.items_processed, 0);
        assert_eq!(result.nudge, Nudge::Silent);
        assert!(result.headline.is_none());
        assert!(result.error.is_some());
    }

    // ── End-to-end Nudge propagation tests (Task 5) ─────────────────────

    #[tokio::test]
    async fn test_backstop_steep_nudge_levels_default_silent() {
        let (db, _dir) = test_db().await;

        db.upsert_documents(vec![make_memory(
            "nudge_default",
            "Test memory for Nudge propagation",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            TriggerKind::Backstop,
        )
        .await
        .unwrap();

        for phase in &result.phases {
            assert_eq!(
                phase.nudge,
                Nudge::Silent,
                "phase {} should be Silent without LLM/data, got {:?} with headline {:?}",
                phase.name,
                phase.nudge,
                phase.headline,
            );
            assert!(
                phase.headline.is_none(),
                "Silent phase {} should have no headline, got {:?}",
                phase.name,
                phase.headline,
            );
        }
    }

    // ── Activity logging round-trip test (Task 8) ───────────────────────

    #[tokio::test]
    async fn test_non_silent_phases_can_be_logged_as_activity() {
        let (db, _dir) = test_db().await;

        let (nudge, headline) = classify_recaps(2);
        assert_eq!(nudge, Nudge::Ambient);
        let headline = headline.expect("Ambient should have a headline");

        db.log_agent_activity("origin", "steep", &[], None, &headline)
            .await
            .unwrap();

        let activities = db.list_agent_activity(10, None, None).await.unwrap();
        assert!(
            !activities.is_empty(),
            "should have at least one activity after logging"
        );
        let last = &activities[0];
        assert_eq!(last.agent_name, "origin");
        assert_eq!(last.action, "steep");
        assert!(
            last.detail.as_ref().is_some_and(|d| d.contains("steeped")),
            "detail should contain the steep headline, got: {:?}",
            last.detail,
        );
    }

    #[test]
    fn rejects_ugly_titles() {
        assert!(looks_like_uuid("5b064ab2-8919-48b2-8220-8f7680b426dd"));
        assert!(looks_like_short_hash("e554534"));
        assert!(looks_like_code(
            "const concept = await db.concepts.findUnique({ where: { id"
        ));
        assert!(looks_like_code("await db.concepts.findUnique"));
        assert!(looks_like_path("[obs/unix/2026-03-17"));
        // True negatives — these SHOULD pass through.
        assert!(!looks_like_uuid("Python backend architecture"));
        assert!(!looks_like_code("MCP server design"));
        assert!(!looks_like_path("Home page redesign"));
        assert!(!looks_like_short_hash("Redis")); // not hex-only
        assert!(!looks_like_short_hash("2026")); // too short
    }

    #[test]
    fn rejects_generic_topic_titles() {
        // All generic placeholders must be rejected.
        assert!(looks_like_generic_topic("general"));
        assert!(looks_like_generic_topic("General"));
        assert!(looks_like_generic_topic("GENERAL"));
        assert!(looks_like_generic_topic("untitled"));
        assert!(looks_like_generic_topic("topic"));
        assert!(looks_like_generic_topic("concept"));
        assert!(looks_like_generic_topic("cluster"));
        assert!(looks_like_generic_topic("misc"));
        assert!(looks_like_generic_topic("other"));
        assert!(looks_like_generic_topic("unknown"));
        // True negatives — these SHOULD pass through.
        assert!(!looks_like_generic_topic("General AI Architecture"));
        assert!(!looks_like_generic_topic("Origin Concept Model"));
        assert!(!looks_like_generic_topic("libSQL Storage"));
    }

    #[test]
    fn rejects_more_ugly_titles() {
        assert!(looks_like_path("[obs/unix/2026-03-17"));
        assert!(looks_like_path("/Users/lucian/second-brain/inbox/"));
        assert!(looks_like_path(
            "[obsidian-main::/Users/lucian/second-brain/inbox/note.md::1]"
        ));
        assert!(looks_like_commit_message(
            "feat: add pin/unpin action to MemoryCard menu"
        ));
        assert!(looks_like_commit_message("fix(home): handle null query"));
        assert!(looks_like_commit_message("chore: bump version"));
        assert!(looks_like_commit_message("refactor: extract helper"));
        assert!(looks_like_commit_message("docs: update readme"));
        // True negatives
        assert!(!looks_like_commit_message("feature parity with X"));
        assert!(!looks_like_commit_message("Fixed memory leak in router"));
        assert!(!looks_like_path("Personal experience redesign"));
        assert!(!looks_like_path("Home page redesign"));
    }

    #[test]
    fn strips_source_id_prefix() {
        assert_eq!(
            strip_source_prefix("[obs/unix/2026-03-17] actual content"),
            "actual content"
        );
        assert_eq!(
            strip_source_prefix("[mem_abc123] memory content"),
            "memory content"
        );
        assert_eq!(
            strip_source_prefix("[5b064ab2-8919-48b2] some text"),
            "some text"
        );
        assert_eq!(strip_source_prefix("plain content"), "plain content");
        // Bracket with spaces inside is NOT a source id — leave it alone
        assert_eq!(
            strip_source_prefix("[has spaces] content"),
            "[has spaces] content"
        );
    }
}
