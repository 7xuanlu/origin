// SPDX-License-Identifier: Apache-2.0
pub(crate) mod helpers;
pub(crate) use helpers::*;

mod phase;
pub use phase::Phase;

pub(crate) mod summary;

pub use crate::synthesis::distill::{
    deep_distill_single, distill_one_cluster, distill_pages, distill_pages_scoped, formation_sweep,
    resolve_distill_target, DistillTarget,
};

// Re-export KG phase functions from `kg::*` to preserve the public API path
// `wenlan_core::refinery::{extract_single_memory_entities, reweave_entity_links}`.
// External callers: post_ingest.rs, eval/shared.rs, wenlan-server::memory_routes.rs.
pub use crate::kg::entity_extraction::{commit_kg, extract_kg, extract_single_memory_entities};
pub use crate::kg::reweave::reweave_entity_links;

// Internal re-imports for refinery code that still calls into the moved
// distillation helpers (distill_one_cluster + refine_clusters_with_llm).
use crate::synthesis::detect::detect_page_candidates;
use crate::synthesis::distill::{
    automatic_refresh_exceeds_source_cap, distill_pages_scoped_gated, refresh_page, RefreshReason,
    AUTOMATIC_PAGE_REFRESH_SOURCE_CAP,
};
use crate::synthesis::refinement_queue::process_refinement_queue;

use crate::activity::ACTIVITY_GAP_SECS;
use crate::db::{MemoryDB, StalePageCursor};
use crate::error::WenlanError;
use crate::llm_provider::{LlmBackend, LlmProvider, LlmRequest};
use crate::prompts::PromptRegistry;
use serde::Serialize;
use std::sync::Arc;

/// app_metadata key backing the compile queue depth gauge (spec §3.1/§7): the
/// count of clusters the last routed compile left pending because no lane
/// (cloud or healthy on-device) was available to write them. Surfaced on
/// `/api/status`. A gauge, not a counter — each compile tick overwrites it
/// with the pending count discovered THIS tick (the staging pool is
/// re-queried fresh every time, not a persisted queue).
const COMPILE_QUEUE_DEPTH_KEY: &str = "compile_queue_depth_v1";

/// Persist the compile queue depth so `/api/status` can report it without
/// re-running cluster discovery. Called after every routed compile.
pub async fn persist_compile_queue_depth(db: &MemoryDB, depth: usize) -> Result<(), WenlanError> {
    db.set_app_metadata(COMPILE_QUEUE_DEPTH_KEY, &depth.to_string())
        .await
}

/// Read the last-persisted compile queue depth. Defaults to 0 (idle) when
/// never set or unparseable.
pub async fn compile_queue_depth(db: &MemoryDB) -> Result<usize, WenlanError> {
    Ok(db
        .get_app_metadata(COMPILE_QUEUE_DEPTH_KEY)
        .await?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0))
}

/// Whether legacy explicit foreground routing may fall back to on-device
/// synthesis. Background work ignores this preference and instead requires an
/// explicit `synthesis_source` pin.
pub fn on_device_compile_preferred() -> bool {
    std::env::var("WENLAN_PREFER_ON_DEVICE_COMPILE")
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Which slot serves everyday (recap, extraction, bulk-enrich) inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EverydaySource {
    Anthropic,
    External,
    OnDevice,
    Basic,
}

impl EverydaySource {
    /// snake_case label for the resolved-routing endpoint.
    pub fn as_str(self) -> &'static str {
        match self {
            EverydaySource::Anthropic => "anthropic",
            EverydaySource::External => "external",
            EverydaySource::OnDevice => "on_device",
            EverydaySource::Basic => "basic",
        }
    }

    /// Parse a config `everyday_source` pin. `Basic` is not a pinnable source
    /// (it means "no LLM"), so only the three real sources map; anything else
    /// (including `None`) yields no pin.
    pub fn parse(s: Option<&str>) -> Option<Self> {
        match s {
            Some("anthropic") => Some(EverydaySource::Anthropic),
            Some("external") => Some(EverydaySource::External),
            Some("on_device") => Some(EverydaySource::OnDevice),
            _ => None,
        }
    }
}

/// Resolved everyday route: the chosen provider plus the source that named it.
pub struct EverydayRoute<'a> {
    pub source: EverydaySource,
    pub llm: Option<&'a Arc<dyn LlmProvider>>,
}

/// Legacy explicit fallback for foreground callers: Anthropic → external →
/// on-device.
///
/// Background work must call [`resolve_everyday`], which requires an explicit
/// source pin and never crosses sources. This helper remains only for callers
/// whose operation itself is explicit and whose legacy fallback contract is
/// therefore independent of background consent.
pub fn everyday_llm<'a>(
    api_llm: Option<&'a Arc<dyn LlmProvider>>,
    external_llm: Option<&'a Arc<dyn LlmProvider>>,
    on_device: Option<&'a Arc<dyn LlmProvider>>,
) -> EverydayRoute<'a> {
    if let Some(llm) = api_llm {
        EverydayRoute {
            source: EverydaySource::Anthropic,
            llm: Some(llm),
        }
    } else if let Some(llm) = external_llm {
        EverydayRoute {
            source: EverydaySource::External,
            llm: Some(llm),
        }
    } else if let Some(llm) = on_device {
        EverydayRoute {
            source: EverydaySource::OnDevice,
            llm: Some(llm),
        }
    } else {
        EverydayRoute {
            source: EverydaySource::Basic,
            llm: None,
        }
    }
}

/// Which slot serves synthesis (page distillation / emergence) inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesisSource {
    Anthropic,
    External,
    OnDevice,
    None,
}

impl SynthesisSource {
    /// snake_case label for the resolved-routing endpoint.
    pub fn as_str(self) -> &'static str {
        match self {
            SynthesisSource::Anthropic => "anthropic",
            SynthesisSource::External => "external",
            SynthesisSource::OnDevice => "on_device",
            SynthesisSource::None => "none",
        }
    }

    /// Parse a config `synthesis_source` pin. `None` (the "nothing resolved"
    /// label) is not a pinnable source. `on_device` is a first-class pin — it
    /// carries no capability restriction; the legacy compile gate affects only
    /// explicit callers of [`synthesis_route`].
    pub fn parse(s: Option<&str>) -> Option<Self> {
        match s {
            Some("anthropic") => Some(SynthesisSource::Anthropic),
            Some("external") => Some(SynthesisSource::External),
            Some("on_device") => Some(SynthesisSource::OnDevice),
            _ => None,
        }
    }
}

/// Resolved synthesis route: the chosen provider plus the source that named it.
pub struct SynthesisRoute<'a> {
    pub source: SynthesisSource,
    pub llm: Option<&'a Arc<dyn LlmProvider>>,
}

/// Resolve the legacy synthesis fallback for explicit foreground callers.
///
/// It prefers an Api-backed slot in `synthesis → api → external` order, then an
/// available on-device slot only when `WENLAN_PREFER_ON_DEVICE_COMPILE` is set.
/// Background Emergence must call [`resolve_synthesis`], which requires an
/// explicit source pin and never crosses sources.
pub fn synthesis_route<'a>(
    synthesis_llm: Option<&'a Arc<dyn LlmProvider>>,
    api_llm: Option<&'a Arc<dyn LlmProvider>>,
    external_llm: Option<&'a Arc<dyn LlmProvider>>,
    on_device: Option<&'a Arc<dyn LlmProvider>>,
) -> SynthesisRoute<'a> {
    let cloud = [
        (SynthesisSource::Anthropic, synthesis_llm),
        (SynthesisSource::Anthropic, api_llm),
        (SynthesisSource::External, external_llm),
    ]
    .into_iter()
    .find_map(|(source, slot)| {
        slot.filter(|provider| matches!(provider.backend(), LlmBackend::Api))
            .map(|provider| (source, provider))
    });
    if let Some((source, llm)) = cloud {
        return SynthesisRoute {
            source,
            llm: Some(llm),
        };
    }
    if on_device_compile_preferred() {
        if let Some(llm) = on_device.filter(|provider| provider.is_available()) {
            return SynthesisRoute {
                source: SynthesisSource::OnDevice,
                llm: Some(llm),
            };
        }
    }
    SynthesisRoute {
        source: SynthesisSource::None,
        llm: None,
    }
}

/// How a background route was chosen: an explicit ready pin, an explicit pin
/// whose source is unavailable, or no user-authorized source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteMode {
    Pinned,
    PinnedUnavailable,
    Unconfigured,
}

impl RouteMode {
    /// snake_case label for the resolved-routing endpoint.
    pub fn as_str(self) -> &'static str {
        match self {
            RouteMode::Pinned => "pinned",
            RouteMode::PinnedUnavailable => "pinned_unavailable",
            RouteMode::Unconfigured => "unconfigured",
        }
    }
}

/// Resolved everyday route plus how it was chosen.
pub struct EverydayResolution<'a> {
    pub source: EverydaySource,
    pub llm: Option<&'a Arc<dyn LlmProvider>>,
    pub mode: RouteMode,
}

/// Provider health is part of the hard pin boundary, not a fallback signal.
/// A provider may panic while probing a partially initialized runtime; treat
/// that exactly like `false` so automatic work remains pending instead of
/// escaping the scheduler boundary or burning retry receipts.
fn pinned_provider_available(llm: &Arc<dyn LlmProvider>) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| llm.is_available())).unwrap_or(false)
}

/// Resolve everyday routing, honoring an optional per-job source pin.
///
/// - pin set and its slot present + healthy → that source, mode `Pinned`.
/// - pin set but its slot absent/unhealthy → no provider, mode `PinnedUnavailable`.
/// - no pin → no provider, mode `Unconfigured`.
///
/// A slot is present only when that source has been loaded. This is the single
/// entry point the refinery everyday phases and resolved-routing endpoint both
/// call, so behavior and display cannot drift.
pub fn resolve_everyday<'a>(
    pin: Option<EverydaySource>,
    api_llm: Option<&'a Arc<dyn LlmProvider>>,
    external_llm: Option<&'a Arc<dyn LlmProvider>>,
    on_device: Option<&'a Arc<dyn LlmProvider>>,
) -> EverydayResolution<'a> {
    if let Some(pinned) = pin {
        let slot = match pinned {
            EverydaySource::Anthropic => api_llm,
            EverydaySource::External => external_llm,
            EverydaySource::OnDevice => on_device,
            EverydaySource::Basic => None,
        };
        if let Some(llm) = slot.filter(|llm| pinned_provider_available(llm)) {
            return EverydayResolution {
                source: pinned,
                llm: Some(llm),
                mode: RouteMode::Pinned,
            };
        }
        return EverydayResolution {
            source: pinned,
            llm: None,
            mode: RouteMode::PinnedUnavailable,
        };
    }
    EverydayResolution {
        source: EverydaySource::Basic,
        llm: None,
        mode: RouteMode::Unconfigured,
    }
}

/// Resolved synthesis route plus how it was chosen.
pub struct SynthesisResolution<'a> {
    pub source: SynthesisSource,
    pub llm: Option<&'a Arc<dyn LlmProvider>>,
    pub mode: RouteMode,
}

/// Resolve synthesis routing, honoring an optional per-job source pin.
///
/// The `anthropic` pin maps to the dedicated synthesis slot when present, else
/// the routine Anthropic slot (both are the Anthropic source). The `on_device`
/// pin is honored whenever the on-device slot is present — pins carry no
/// capability restriction, symmetric with [`resolve_everyday`]; the compile
/// gate governs only explicit callers of [`synthesis_route`]. A pinned-but-
/// unavailable source remains pinned and pauses instead of crossing providers.
pub fn resolve_synthesis<'a>(
    pin: Option<SynthesisSource>,
    synthesis_llm: Option<&'a Arc<dyn LlmProvider>>,
    api_llm: Option<&'a Arc<dyn LlmProvider>>,
    external_llm: Option<&'a Arc<dyn LlmProvider>>,
    on_device: Option<&'a Arc<dyn LlmProvider>>,
) -> SynthesisResolution<'a> {
    if let Some(pinned) = pin {
        let slot = match pinned {
            SynthesisSource::Anthropic => synthesis_llm.or(api_llm),
            SynthesisSource::External => external_llm,
            SynthesisSource::OnDevice => on_device,
            SynthesisSource::None => None,
        };
        if let Some(llm) = slot.filter(|llm| pinned_provider_available(llm)) {
            return SynthesisResolution {
                source: pinned,
                llm: Some(llm),
                mode: RouteMode::Pinned,
            };
        }
        return SynthesisResolution {
            source: pinned,
            llm: None,
            mode: RouteMode::PinnedUnavailable,
        };
    }
    SynthesisResolution {
        source: SynthesisSource::None,
        llm: None,
        mode: RouteMode::Unconfigured,
    }
}

/// What triggered a refinery cycle. Different triggers run different subsets
/// of phases — the goal is to do the right work at the right time.
///
/// - `Backstop`: runs every phase. Used by the periodic backstop loop and as
///   the safe default for any code path that doesn't know better.
/// - `BurstEnd`: only `recaps` + `refinement_queue`.
/// - `Idle`: only synthesis phases (`community_detection`, `emergence`,
///   `re-distill`, `overview`, `decision_logs`).
/// - `Daily`: only maintenance phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerKind {
    Backstop,
    BurstEnd,
    Idle,
    Daily,
}

impl TriggerKind {
    /// Returns true if this trigger should run `phase`. The compiler-checked
    /// `Phase` enum makes typo'd phase names a compile error — earlier
    /// string-based versions could silently skip phases when names drifted.
    pub fn runs_phase(&self, phase: Phase) -> bool {
        match self {
            Self::Backstop => true,
            Self::BurstEnd => matches!(phase, Phase::Recaps | Phase::RefinementQueue),
            Self::Idle => matches!(
                phase,
                Phase::CommunityDetection
                    | Phase::Detect
                    | Phase::Emergence
                    | Phase::SummaryRollup
                    | Phase::ReDistill
                    | Phase::Overview
                    | Phase::DecisionLogs
                    | Phase::PageMaps
            ),
            Self::Daily => matches!(
                phase,
                Phase::Decay
                    | Phase::Promote
                    | Phase::Reweave
                    | Phase::Reembed
                    | Phase::EntityExtraction
                    | Phase::Overview
                    | Phase::PruneRejections
                    | Phase::Evict
                    | Phase::KgRethink
            ),
        }
    }

    /// Per-trigger deadline in seconds. Each trigger runs a different subset
    /// of phases, so they need different time budgets:
    /// - BurstEnd: tight (recaps + refinement only, should be fast)
    /// - Idle/Daily: moderate (focused subsets, no competition)
    /// - Backstop: generous (runs all phases, safety net every 6h)
    pub fn deadline_secs(&self, base: u64) -> u64 {
        match self {
            Self::BurstEnd => base,      // 120s — recaps should be fast
            Self::Idle => base * 5,      // 600s — synthesis can take time
            Self::Daily => base * 3,     // 360s — maintenance is mostly DB ops
            Self::Backstop => base * 10, // 1200s — safety net, let it finish
        }
    }
}

/// How loud Wenlan should be about a phase's output — the "earned interrupt"
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

/// Backfill phases (decay, promote, reweave, reembed,
/// entity_extraction, community_detection,
/// prune_rejections) are pure plumbing — always Silent.
pub(crate) fn classify_backfill(_count: usize) -> (Nudge, Option<String>) {
    (Nudge::Silent, None)
}

/// Emergence phase — the only phase that can produce `Wow`.
pub(crate) fn classify_emergence(new_concepts: usize) -> (Nudge, Option<String>) {
    match new_concepts {
        0 => (Nudge::Silent, None),
        1 => (
            Nudge::Wow,
            Some("Wenlan steeped your memories into a new page".to_string()),
        ),
        n => (
            Nudge::Wow,
            Some(format!(
                "Wenlan steeped your memories into {} new concepts",
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
            Some("Wenlan refreshed a page with new information".to_string()),
        ),
        n => (
            Nudge::Ambient,
            Some(format!(
                "Wenlan refreshed {} concepts with new information",
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
            Some("Wenlan resolved a memory contradiction".to_string()),
        ),
        n => (
            Nudge::Ambient,
            Some(format!("Wenlan resolved {} memory contradictions", n)),
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

/// Run a typed phase, capturing timing and errors. Returns PhaseResult even
/// on failure. On error, the nudge is always `Silent` and headline is `None`
/// — backend failures should not produce user-facing notifications.
async fn run_phase<F, Fut>(phase: Phase, f: F) -> PhaseResult
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<PhaseOutput, WenlanError>>,
{
    let start = std::time::Instant::now();
    match f().await {
        Ok(output) => PhaseResult {
            name: phase.as_str().to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
            items_processed: output.items_processed,
            error: None,
            nudge: output.nudge,
            headline: output.headline,
        },
        Err(e) => PhaseResult {
            name: phase.as_str().to_string(),
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
/// Runs phases sequentially: decay, recaps, reweave, reembed, entity extraction, distillation, refinement queue, decision logs.
/// Each phase is isolated — a failure in one phase doesn't prevent subsequent phases from running.
pub async fn run_periodic_steep(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
    _confidence_cfg: &crate::tuning::ConfidenceConfig,
    distillation: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> Result<SteepResult, WenlanError> {
    // Forward to implementation with no API/synthesis LLM and full Backstop
    // trigger (preserves pre-PR-A behavior for callers that don't care about
    // event-driven scheduling).
    run_periodic_steep_with_api(
        db,
        llm,
        None,
        None,
        None,
        prompts,
        tuning,
        _confidence_cfg,
        distillation,
        knowledge_path,
        TriggerKind::Backstop,
    )
    .await
}

/// Periodic steep with optional API and synthesis LLM providers.
/// `api_llm` is used for routine tasks (entity extraction, classification).
/// `synthesis_llm` is used for distillation/page synthesis
/// (falls back to api_llm, then external_llm, then on-device).
#[allow(clippy::too_many_arguments)]
pub async fn run_periodic_steep_with_api(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    api_llm: Option<&Arc<dyn LlmProvider>>,
    synthesis_llm: Option<&Arc<dyn LlmProvider>>,
    external_llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
    _confidence_cfg: &crate::tuning::ConfidenceConfig,
    distillation: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
    trigger: TriggerKind,
) -> Result<SteepResult, WenlanError> {
    run_periodic_steep_with_api_scope(
        db,
        llm,
        api_llm,
        synthesis_llm,
        external_llm,
        prompts,
        tuning,
        _confidence_cfg,
        distillation,
        knowledge_path,
        trigger,
        None,
    )
    .await
}

/// Run exactly one trigger-eligible phase and return to the caller. This is the
/// automatic scheduler seam: foreground/explicit callers keep the full-cycle
/// API above, while background turns gain a cooperative yield boundary between
/// phases.
#[derive(Debug)]
pub struct SteepPhaseSliceReport {
    pub result: SteepResult,
    pub selected: bool,
    pub progressed: bool,
    pub more: bool,
    pub retryable: bool,
    pub paused: bool,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_periodic_steep_phase_with_api(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    api_llm: Option<&Arc<dyn LlmProvider>>,
    synthesis_llm: Option<&Arc<dyn LlmProvider>>,
    external_llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
    confidence_cfg: &crate::tuning::ConfidenceConfig,
    distillation: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
    trigger: TriggerKind,
    phase: Phase,
) -> Result<SteepPhaseSliceReport, WenlanError> {
    // Re-distill historically scanned every changed Page and then every stale
    // Page inside one phase. Automatic callers must use the cooperative
    // one-Page seam; the explicit full-cycle API intentionally keeps the
    // legacy batch behavior above.
    if phase == Phase::ReDistill && trigger.runs_phase(phase) {
        let routing_config = crate::config::load_config();
        let synthesis_pin = SynthesisSource::parse(routing_config.synthesis_source.as_deref());
        let compile_llm =
            resolve_synthesis(synthesis_pin, synthesis_llm, api_llm, external_llm, llm).llm;
        let started = std::time::Instant::now();
        return match run_redistill_page_slice(db, compile_llm, prompts, knowledge_path).await {
            Ok(slice) => {
                let refreshed = slice.items_processed;
                let (nudge, headline) = classify_redistill(refreshed);
                Ok(SteepPhaseSliceReport {
                    result: SteepResult {
                        memories_decayed: 0,
                        recaps_generated: 0,
                        distilled: 0,
                        pending_remaining: 0,
                        phases: vec![PhaseResult {
                            name: phase.as_str().to_string(),
                            duration_ms: started.elapsed().as_millis() as u64,
                            items_processed: refreshed,
                            error: None,
                            nudge,
                            headline,
                        }],
                    },
                    selected: slice.selected,
                    progressed: slice.progressed,
                    more: slice.more,
                    retryable: slice.retryable,
                    paused: slice.paused,
                })
            }
            Err(error) => Ok(SteepPhaseSliceReport {
                result: SteepResult {
                    memories_decayed: 0,
                    recaps_generated: 0,
                    distilled: 0,
                    pending_remaining: 0,
                    phases: vec![PhaseResult {
                        name: phase.as_str().to_string(),
                        duration_ms: started.elapsed().as_millis() as u64,
                        items_processed: 0,
                        error: Some(error.to_string()),
                        nudge: Nudge::Silent,
                        headline: None,
                    }],
                },
                selected: false,
                progressed: false,
                more: false,
                retryable: true,
                paused: false,
            }),
        };
    }

    let result = run_periodic_steep_with_api_scope(
        db,
        llm,
        api_llm,
        synthesis_llm,
        external_llm,
        prompts,
        tuning,
        confidence_cfg,
        distillation,
        knowledge_path,
        trigger,
        Some(phase),
    )
    .await?;
    let selected = !result.phases.is_empty();
    let retryable = result.phases.iter().any(|phase| phase.error.is_some());
    Ok(SteepPhaseSliceReport {
        result,
        selected,
        progressed: selected && !retryable,
        more: false,
        retryable,
        paused: false,
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_periodic_steep_with_api_scope(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    api_llm: Option<&Arc<dyn LlmProvider>>,
    synthesis_llm: Option<&Arc<dyn LlmProvider>>,
    external_llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
    _confidence_cfg: &crate::tuning::ConfidenceConfig,
    distillation: &crate::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
    trigger: TriggerKind,
    only_phase: Option<Phase>,
) -> Result<SteepResult, WenlanError> {
    let steep_start = std::time::Instant::now();
    let deadline = trigger.deadline_secs(tuning.steep_deadline_secs);
    // Per-job source pins are read once per cycle. Missing pins authorize no
    // background inference; deterministic phases still run with `None`.
    let routing_config = crate::config::load_config();
    let everyday_pin = EverydaySource::parse(routing_config.everyday_source.as_deref());
    let synthesis_pin = SynthesisSource::parse(routing_config.synthesis_source.as_deref());
    let page_map_auto_suggest = routing_config.page_map_auto_suggest;
    let everyday_route_llm = resolve_everyday(everyday_pin, api_llm, external_llm, llm).llm;
    let runs_phase = |phase: Phase| {
        trigger.runs_phase(phase) && only_phase.is_none_or(|selected| selected == phase)
    };
    let mut phases: Vec<PhaseResult> = Vec::new();
    #[allow(unused_assignments)]
    let mut deadline_hit = false; // track if we've logged the first skip

    // Phase metric variables — initialized to 0 so SteepResult fields stay
    // valid even when their producing phase is skipped by the trigger.
    let mut memories_decayed: u64 = 0;
    let mut recaps_generated: u32 = 0;
    let mut distilled: u32 = 0;

    // Phase 1: Decay pass
    let db_ref = db;
    if runs_phase(Phase::Decay) {
        let phase = run_phase(Phase::Decay, || async {
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
    if runs_phase(Phase::Promote) {
        let phase = run_phase(Phase::Promote, || async {
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

    // Phase 2: Recap generation (everyday job — requires an explicit
    // everyday_source pin; otherwise deterministic no-LLM behavior applies).
    let recap_llm = everyday_route_llm;
    if runs_phase(Phase::Recaps) {
        let phase = run_phase(Phase::Recaps, || async {
            let generated =
                crate::synthesis::recaps::generate_recaps(db_ref, recap_llm, prompts, tuning)
                    .await?;
            if generated > 0 {
                log::info!("[refinery] generated {} recaps", generated);
            }
            let (nudge, headline) = crate::synthesis::recaps::classify_recaps(generated as usize);
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

    // Phase 3: Reweave entity links (link unlinked memories to recently-created entities)
    if runs_phase(Phase::Reweave)
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
        let phase = run_phase(Phase::Reweave, || async {
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
    if runs_phase(Phase::Reembed)
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
        let phase = run_phase(Phase::Reembed, || async {
            let count = crate::migrations::reembed::run(db_ref, 5).await?;
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

    // Phase 5: Entity extraction — everyday job (requires an explicit
    // everyday_source pin; otherwise deterministic no-LLM behavior applies).
    let extract_llm = everyday_route_llm;
    if runs_phase(Phase::EntityExtraction)
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
        let phase = run_phase(Phase::EntityExtraction, || async {
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
    if runs_phase(Phase::CommunityDetection)
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
        let phase = run_phase(Phase::CommunityDetection, || async {
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

    // Phase 5c: DETECT — embedding/cosine-only page candidate pass. It runs
    // before compile/emergence so cheap attach-to-existing-page opportunities
    // are consumed through PageWrite before any LLM synthesis lane is touched.
    if runs_phase(Phase::Detect) {
        let phase = run_phase(Phase::Detect, || async {
            let report = detect_page_candidates(db_ref, distillation).await?;
            if report.candidates_processed > 0
                || report.attached > 0
                || report.skipped_unchanged > 0
            {
                log::info!(
                    "[detect] processed={}, attached={}, skipped_unchanged={}",
                    report.candidates_processed,
                    report.attached,
                    report.skipped_unchanged
                );
            }
            let (nudge, headline) = classify_backfill(report.attached);
            Ok(PhaseOutput {
                items_processed: report.candidates_processed,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 6: Normal distill — create new concepts from clusters. Background
    // synthesis requires an explicit source pin; unavailable pins stay paused.
    let compile_llm =
        resolve_synthesis(synthesis_pin, synthesis_llm, api_llm, external_llm, llm).llm;
    let kp_ref = knowledge_path;
    if runs_phase(Phase::Emergence) {
        let phase = run_phase(Phase::Emergence, || async {
            let run_coherence_gate =
                compile_llm.is_some_and(|provider| matches!(provider.backend(), LlmBackend::Api));
            let result = distill_pages_scoped_gated(
                db_ref,
                compile_llm,
                prompts,
                distillation,
                kp_ref,
                None,
                run_coherence_gate,
            )
            .await?;
            if let Err(e) = persist_compile_queue_depth(db_ref, result.pending.len()).await {
                log::warn!("[emergence] failed to persist compile queue depth: {e}");
            }
            // Re-resolve orphan wikilinks now that distill may have created
            // new pages. Cheap: one SELECT DISTINCT + per-label UPDATE for
            // hits; no LLM. Captures the case where page A linked to
            // [[Topic Z]] before Topic Z existed, and emergence just minted
            // a Topic Z page.
            match db_ref.resolve_orphan_page_links().await {
                Ok(n) if n > 0 => {
                    log::info!("[emergence] resolved {n} orphan wikilink labels");
                }
                Ok(_) => {}
                Err(e) => log::warn!("[emergence] orphan link resolve failed: {e}"),
            }
            let count = result.created.len();
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

    // Phase 6a2: Summary rollup (T18 hierarchical global-context prelude).
    // Ship-dark: the entire build is gated behind `global_prelude_enabled()`
    // so an unset flag means zero build cost. Sequenced after Emergence so the
    // community grouping the buckets key on is fresh. Degrades to a
    // deterministic template when no LLM is available (no silent-zero).
    if runs_phase(Phase::SummaryRollup) && crate::db::global_prelude_enabled() {
        let phase = run_phase(Phase::SummaryRollup, || async {
            let count =
                summary::build_summary_nodes(db_ref, compile_llm.map(|a| a.as_ref())).await?;
            Ok(PhaseOutput {
                items_processed: count,
                nudge: Nudge::Silent,
                headline: None,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 6b: Re-distill — refresh concepts whose source memories changed
    if runs_phase(Phase::ReDistill)
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
        let phase = run_phase(Phase::ReDistill, || async {
            let enqueued = enqueue_changed_pages(db_ref).await?;
            if enqueued > 0 {
                log::info!("[re-distill] queued {enqueued} pages with changed explicit sources");
            }
            // Re-distill every explicitly stale page through the single CAS writer.
            let stale = re_distill_stale_pages(db_ref, compile_llm, prompts, kp_ref).await?;
            let count = stale;
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

    if runs_phase(Phase::Overview)
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
        let phase = run_phase(Phase::Overview, || async {
            let count = maybe_refresh_overview_page(db_ref, compile_llm, prompts, kp_ref).await?;
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
    if runs_phase(Phase::RefinementQueue) {
        let phase = run_phase(Phase::RefinementQueue, || async {
            let count =
                process_refinement_queue(db_ref, everyday_route_llm, prompts, tuning).await?;
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

    // Phase 7: Decision log generation (lightweight recap for decisions).
    // Last deadline-gated phase: omit `deadline_hit = true` — no phase after
    // this reads the flag, so the assignment would be dead code (clippy -D).
    if runs_phase(Phase::DecisionLogs)
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
        let phase = run_phase(Phase::DecisionLogs, || async {
            let count = crate::synthesis::decision_logs::generate_decision_logs(
                db_ref,
                everyday_route_llm,
                prompts,
                tuning,
            )
            .await?;
            let (nudge, headline) = crate::synthesis::decision_logs::classify_decision_logs(count);
            Ok(PhaseOutput {
                items_processed: count,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase: proactive Page-Map suggestions (Idle only; config-gated).
    // Generates `status='suggested'` nodes/edges for recently-changed pages,
    // insert-only. The explicit `POST /api/pages/{id}/map/improve` route runs
    // the SAME pass but is NEVER gated by `page_map_auto_suggest` — only this
    // background phase is. Bounded to 5 pages per pass. Always Silent (no
    // user-facing nudge; suggestions surface in the map UI, not as a toast).
    // `runs_phase` (not `trigger.runs_phase`): a phase-sliced background turn
    // must not leak page_maps into another phase's slice when auto-suggest is on.
    if page_map_auto_suggest && runs_phase(Phase::PageMaps) {
        let phase = run_phase(Phase::PageMaps, || async {
            let improved = crate::page_map_improve::run_proactive_page_maps(db_ref, 5).await?;
            let (nudge, headline) = classify_backfill(improved);
            Ok(PhaseOutput {
                items_processed: improved,
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
    if runs_phase(Phase::PruneRejections) {
        let phase = run_phase(Phase::PruneRejections, || async {
            let count = db_ref.prune_rejections(30).await?;
            // Clean up concept_sources rows whose source memories were deleted.
            match db_ref.cleanup_orphaned_page_sources().await {
                Ok(n) if n > 0 => {
                    log::info!("[refinery] cleaned {} orphaned concept_sources rows", n);
                }
                Ok(_) => {}
                Err(e) => log::warn!("[refinery] orphan concept_sources cleanup failed: {e}"),
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

    // Phase 9: Evict — T21 Stage 1 archive-not-delete soft eviction. Double
    // gated: the trigger must include Evict (Backstop/Daily only) AND the
    // WENLAN_ENABLE_EVICTION env flag must be truthy. Default OFF = no phase,
    // no archiving, byte-identical to current unbounded-append behavior.
    // Placed AFTER the Decay block so Evict reads freshly-decayed
    // effective_confidence within the same cycle. Failures degrade silently
    // (run_phase captures the error into PhaseResult) — they never crash the
    // cycle. Event surfacing flows through PhaseResult nudge/headline; MemoryDB
    // has no emitter.
    if runs_phase(Phase::Evict) && crate::db::eviction_enabled() {
        let phase = run_phase(Phase::Evict, || async {
            let report = db_ref
                .evict_stale(&crate::tuning::EvictionConfig::default())
                .await?;
            log::info!(
                "[refinery] evict: archived {} stale memories",
                report.archived
            );
            let (nudge, headline) = classify_backfill(report.archived);
            Ok(PhaseOutput {
                items_processed: report.archived,
                nudge,
                headline,
            })
        })
        .await;
        phases.push(phase);
    }

    // Phase 10: KG rethink — periodic knowledge graph quality maintenance.
    // Rate-limited by `kg_rethink_interval_hours` (default 168h = weekly)
    // via `app_metadata.last_kg_rethink_ts`. All five sub-phases are cheap
    // when the graph is clean; the gate mainly avoids redundant log spam.
    if runs_phase(Phase::KgRethink) {
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
            let phase = run_phase(Phase::KgRethink, || async {
                let report = crate::kg_quality::run_rethink(db_ref, llm, tuning).await?;
                let total = report.merge_candidates
                    + report.relations_healed
                    + report.entities_healed
                    + report.embeddings_refreshed
                    + report.stale_relations_flagged
                    + report.contradictions_found;
                log::info!(
                    "[refinery] kg_rethink: {} merges, {} healed, {} queued, {} refreshed, {} stale, {} contradictions, {} entities_healed, {} entities_queued",
                    report.merge_candidates,
                    report.relations_healed,
                    report.relations_queued,
                    report.embeddings_refreshed,
                    report.stale_relations_flagged,
                    report.contradictions_found,
                    report.entities_healed,
                    report.entities_queued,
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
                let _ = db
                    .set_app_metadata("last_kg_rethink_ts", &now.to_string())
                    .await;
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

    // Onboarding milestone check — first-page and graph-alive. Runs once
    // per steep after all phases complete, so entity-extraction (phase 5) and
    // distillation (phase 6) writes are both visible. Each evaluator call is
    // idempotent via DB uniqueness, so repeated passes are harmless.
    if only_phase.is_none() {
        let emitter_for_ms: std::sync::Arc<dyn crate::events::EventEmitter> =
            std::sync::Arc::new(crate::events::NoopEmitter);
        let ev = crate::onboarding::MilestoneEvaluator::new(db, emitter_for_ms);
        if let Err(e) = ev.check_after_refinery_pass().await {
            log::warn!("onboarding: check_after_refinery_pass failed: {e}");
        }
    }

    Ok(SteepResult {
        memories_decayed,
        recaps_generated,
        distilled,
        pending_remaining,
        phases,
    })
}

/// Extract entities from unlinked memories via LLM and create them in the knowledge graph.
/// Processes `limit` memories per steep to avoid GPU overload.
/// Acts as a backfill for any memories that failed store-time extraction.
pub(crate) async fn extract_entities_from_memories(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    limit: usize,
) -> Result<usize, WenlanError> {
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

/// Mark pages whose explicit source memories changed. The stale-page sweep is
/// the sole LLM executor, so every refresh reaches the same stale-only CAS.
pub(crate) async fn enqueue_changed_pages(db: &MemoryDB) -> Result<usize, WenlanError> {
    let all_active = db.list_pages("active", 200, 0).await?;
    let mut enqueued = 0usize;

    for page in &all_active {
        let changed = db.has_page_sources_changed(page).await.unwrap_or(false);
        if !changed {
            continue;
        }
        db.set_page_stale(&page.id, "source_updated").await?;
        enqueued += 1;
    }
    Ok(enqueued)
}

/// Re-distill concepts explicitly marked stale by topic-key upserts.
///
/// Distinct from `enqueue_changed_pages` (which checks last_modified timestamps):
/// this targets concepts whose source memories were updated in-place and thus didn't
/// change their last_modified. The `stale_reason` field is set by the topic-match
/// upsert path in handle_store_memory.
///
/// - `source_updated`: LLM re-distills using the join table's current source list.
///   Every stale page (human-owned or not) routes through `refresh_page`, which
///   owns the ownership gate itself: a human-owned page (`user_edited` or
///   `creation_kind = "authored"`) never gets its prose overwritten — the
///   sweep stages a revision card instead (spec §5.1/§5.2) and clears
///   staleness so the card isn't re-proposed every cycle. No-op refreshes
///   stay stale so a later sweep can retry.
pub(crate) async fn re_distill_stale_pages(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    knowledge_path: Option<&std::path::Path>,
) -> Result<usize, WenlanError> {
    let stale = db.list_stale_pages("source_updated").await?;
    if stale.is_empty() {
        return Ok(0);
    }

    let llm_ref = match llm {
        Some(l) if l.is_available() => l,
        _ => {
            log::debug!(
                "[re-distill-stale] no LLM available, skipping {} stale concepts",
                stale.len()
            );
            return Ok(0);
        }
    };

    let mut recompiled = 0usize;
    for page in &stale {
        // Synthesis, citation verification, the fail-closed guard, and the
        // atomic content+citations+changelog write all live in the ONE
        // re-distill op (`refresh_page`); this loop just owns the staleness
        // lifecycle. `refresh_page` reads the page's current sources (join
        // table first) itself, so no per-site source assembly here.
        match refresh_page(
            db,
            llm_ref,
            prompts,
            &page.id,
            RefreshReason::SourceChanged,
            knowledge_path,
        )
        .await
        {
            Ok(outcome) => {
                if outcome.wrote {
                    recompiled += 1;
                    log::info!("[re-distill-stale] refreshed page '{}'", page.title);
                } else if outcome.gated {
                    log::info!(
                        "[re-distill-stale] '{}' human-owned; staged revision card, cleared staleness",
                        page.title
                    );
                } else if outcome.acknowledged {
                    log::info!(
                        "[re-distill-stale] '{}' verified unchanged; acknowledged compile watermark",
                        page.title
                    );
                } else {
                    log::info!(
                        "[re-distill-stale] '{}' yielded no write; keeping stale for retry",
                        page.title
                    );
                }
            }
            Err(e) => log::warn!("[re-distill-stale] refresh error for '{}': {}", page.id, e),
        }
    }

    if recompiled > 0 {
        log::info!("[re-distill-stale] refreshed {} stale concepts", recompiled);
    }
    Ok(recompiled)
}

// v1 stored a numeric OFFSET that cannot be translated after stale-set
// mutation. v2 deliberately starts a fresh keyset pass on upgrade.
const REDISTILL_STALE_CURSOR_KEY: &str = "background_redistill_stale_cursor_v2";
const REDISTILL_SCAN_CURSOR_KEY: &str = "background_redistill_scan_cursor_v1";

/// Outcome of one cooperative background re-distill turn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RedistillSliceReport {
    /// One stale Page was refreshed, or one active Page was checked for change.
    pub selected: bool,
    /// Durable state advanced (content/card/staleness lifecycle or scan cursor).
    pub progressed: bool,
    /// Actual Page refreshes or staged revision cards. Cursor-only progress is
    /// deliberately zero so a cheap unchanged scan cannot emit an Activity
    /// nudge claiming that Wenlan refreshed user data.
    pub items_processed: usize,
    /// Another item is safe to schedule after the global thermal cooldown.
    pub more: bool,
    /// The selected Page remains retryable, but should wait for a later trigger
    /// when it is the only pending item so a poison Page cannot hot-loop.
    pub retryable: bool,
    /// Work exists conceptually but no authorized/available provider can run it.
    pub paused: bool,
}

async fn load_background_cursor(db: &MemoryDB, key: &str) -> usize {
    db.get_app_metadata(key)
        .await
        .ok()
        .flatten()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

async fn persist_background_cursor(
    db: &MemoryDB,
    key: &str,
    cursor: usize,
) -> Result<(), WenlanError> {
    db.set_app_metadata(key, &cursor.to_string()).await
}

async fn load_background_stale_cursor(db: &MemoryDB) -> Option<StalePageCursor> {
    db.get_app_metadata(REDISTILL_STALE_CURSOR_KEY)
        .await
        .ok()
        .flatten()
        .and_then(|value| serde_json::from_str(&value).ok())
}

async fn persist_background_stale_cursor(
    db: &MemoryDB,
    cursor: Option<&StalePageCursor>,
) -> Result<(), WenlanError> {
    let value = cursor
        .map(serde_json::to_string)
        .transpose()?
        .unwrap_or_default();
    db.set_app_metadata(REDISTILL_STALE_CURSOR_KEY, &value)
        .await
}

fn over_cap_redistill_report() -> RedistillSliceReport {
    RedistillSliceReport {
        selected: true,
        progressed: true,
        more: true,
        retryable: true,
        ..RedistillSliceReport::default()
    }
}

/// Refresh at most one stale/changed Page, then return to the scheduler.
///
/// Explicit stale work has priority. A persisted cursor rotates failures so a
/// citation-gated or otherwise retryable Page cannot pin every later Page. When
/// no stale Page exists, exactly one active Page is inspected for legacy
/// timestamp-based change detection; the scan cursor survives daemon restarts.
pub async fn run_redistill_page_slice(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    knowledge_path: Option<&std::path::Path>,
) -> Result<RedistillSliceReport, WenlanError> {
    let Some(provider) = llm.filter(|provider| provider.is_available()) else {
        return Ok(RedistillSliceReport {
            paused: true,
            ..RedistillSliceReport::default()
        });
    };

    let cursor = load_background_stale_cursor(db).await;
    let selected = match db
        .get_stale_page_after("source_updated", cursor.as_ref())
        .await?
    {
        Some(page) => Some(page),
        None if cursor.is_some() => {
            persist_background_stale_cursor(db, None).await?;
            db.get_stale_page_after("source_updated", None).await?
        }
        None => None,
    };
    if let Some(page) = selected {
        // Advance before provider code so a panic cannot pin the same Page.
        // Persisting the selected identity, rather than its mutable row index,
        // keeps the next Page reachable when this one leaves the stale set.
        let selected_cursor = StalePageCursor::for_page(&page);
        persist_background_stale_cursor(db, Some(&selected_cursor)).await?;
        if automatic_refresh_exceeds_source_cap(db, &page).await? {
            log::warn!(
                "[re-distill-slice] page '{}' exceeds the automatic {}-source cap; keeping it stale for explicit refresh",
                page.id,
                AUTOMATIC_PAGE_REFRESH_SOURCE_CAP
            );
            return Ok(over_cap_redistill_report());
        }
        let outcome = refresh_page(
            db,
            provider,
            prompts,
            &page.id,
            RefreshReason::SourceChanged,
            knowledge_path,
        )
        .await;
        return match outcome {
            Ok(outcome) if outcome.wrote || outcome.gated => {
                db.clear_page_staleness(&page.id).await?;
                Ok(RedistillSliceReport {
                    selected: true,
                    progressed: true,
                    items_processed: 1,
                    more: true,
                    ..RedistillSliceReport::default()
                })
            }
            Ok(_) => Ok(RedistillSliceReport {
                selected: true,
                more: true,
                retryable: true,
                ..RedistillSliceReport::default()
            }),
            Err(error) => {
                log::warn!(
                    "[re-distill-slice] refresh error for '{}': {}",
                    page.id,
                    error
                );
                Ok(RedistillSliceReport {
                    selected: true,
                    more: true,
                    retryable: true,
                    ..RedistillSliceReport::default()
                })
            }
        };
    }

    let cursor = load_background_cursor(db, REDISTILL_SCAN_CURSOR_KEY).await;
    let page = db.list_pages("active", 1, cursor as i64).await?;
    let Some(page) = page.first() else {
        persist_background_cursor(db, REDISTILL_SCAN_CURSOR_KEY, 0).await?;
        return Ok(RedistillSliceReport::default());
    };
    if automatic_refresh_exceeds_source_cap(db, page).await? {
        persist_background_cursor(db, REDISTILL_SCAN_CURSOR_KEY, cursor.wrapping_add(1)).await?;
        log::warn!(
            "[re-distill-slice] page '{}' exceeds the automatic {}-source cap; leaving it for explicit refresh",
            page.id,
            AUTOMATIC_PAGE_REFRESH_SOURCE_CAP
        );
        return Ok(over_cap_redistill_report());
    }
    let changed = db.has_page_sources_changed(page).await?;
    if !changed {
        persist_background_cursor(db, REDISTILL_SCAN_CURSOR_KEY, cursor.wrapping_add(1)).await?;
        return Ok(RedistillSliceReport {
            selected: true,
            progressed: true,
            more: true,
            ..RedistillSliceReport::default()
        });
    }

    // The canonical SourceChanged write uses a stale-state CAS. Marking the
    // selected Page first both makes that CAS valid and prevents the historical
    // spend-inference-then-drop loop for timestamp-only changes.
    db.set_page_stale(&page.id, "source_updated").await?;
    let outcome = refresh_page(
        db,
        provider,
        prompts,
        &page.id,
        RefreshReason::SourceChanged,
        knowledge_path,
    )
    .await;
    persist_background_cursor(db, REDISTILL_SCAN_CURSOR_KEY, cursor.wrapping_add(1)).await?;
    match outcome {
        Ok(outcome) if outcome.wrote || outcome.gated => {
            db.clear_page_staleness(&page.id).await?;
            Ok(RedistillSliceReport {
                selected: true,
                progressed: true,
                items_processed: 1,
                more: true,
                ..RedistillSliceReport::default()
            })
        }
        Ok(_) => Ok(RedistillSliceReport {
            selected: true,
            retryable: true,
            ..RedistillSliceReport::default()
        }),
        Err(error) => {
            log::warn!(
                "[re-distill-slice] refresh error for '{}': {}",
                page.id,
                error
            );
            Ok(RedistillSliceReport {
                selected: true,
                retryable: true,
                ..RedistillSliceReport::default()
            })
        }
    }
}

/// Spec §5.3: refresh the reserved, machine-owned Overview page as part of
/// the maintenance re-distill phase — the "wiki is alive" signal. No-op
/// (returns 0, creates nothing) when no LLM lane is available, mirroring the
/// `enqueue_changed_pages` guard above: a steep cycle with no LLM must not
/// create the placeholder row just to leave it unrefreshed.
async fn maybe_refresh_overview_page(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    knowledge_path: Option<&std::path::Path>,
) -> Result<usize, WenlanError> {
    let llm = match llm {
        Some(l) if l.is_available() => l,
        _ => return Ok(0),
    };
    let outcome = crate::synthesis::overview::refresh_overview_page(
        db,
        llm,
        prompts,
        "refinery",
        knowledge_path,
    )
    .await?;
    Ok(if outcome.wrote { 1 } else { 0 })
}

/// Group memories by activity bursts (30-min gap → new burst).
/// Input memories should be sorted by last_modified (ascending or descending).
/// Output: groups of references into the input slice, each group is one burst.
pub(crate) type BurstItem = (String, String, Option<String>, i64);

#[allow(clippy::type_complexity)]
pub(crate) fn group_into_bursts(memories: &[BurstItem]) -> Vec<Vec<&BurstItem>> {
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

/// Clean memory content for recap display: strip structured field metadata and prefixes.
pub(crate) fn clean_for_recap(content: &str) -> String {
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

/// Build a structured raw-context string from a burst of memories.
pub(crate) fn build_burst_context(
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

    let spaces: Vec<String> = memories
        .iter()
        .filter_map(|(_, _, d, _)| d.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let spaces_str = if spaces.is_empty() {
        "unknown".to_string()
    } else {
        spaces.join(", ")
    };

    let lines: Vec<String> = memories
        .iter()
        .map(|(_, content, space, _)| {
            let clean = clean_for_recap(content);
            match space {
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
        spaces_str,
        lines.join("\n"),
    )
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
        user_prompt: input,
        max_tokens: 16,
        temperature: 0.3,
        label: None,
        timeout_secs: None,
    }).await;

    match response {
        Ok(output) => {
            let cleaned = crate::llm_provider::strip_think_tags(&output);
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
            let char_count = title.chars().count();
            log::info!(
                "[title] clean='{}' ({} words, {} chars)",
                title,
                word_count,
                char_count
            );
            // Reject if empty, too many words, too long (CJK content has no
            // whitespace so word_count alone misses runaway CJK strings),
            // multiline, or truncated mid-sentence.
            let truncated = title.ends_with(',')
                || title.ends_with("including")
                || title.ends_with("with")
                || title.ends_with("the")
                || title.ends_with("and")
                || title.ends_with("for")
                || title.ends_with("of")
                || title.ends_with("in");
            if title.is_empty()
                || word_count > 8
                || char_count > 80
                || title.contains('\n')
                || truncated
            {
                log::info!("[title] rejected (empty/too long/multiline)");
                None
            } else if is_all_generic_tokens(title)
                || looks_like_markup_styled(title)
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

#[derive(Debug, Serialize)]
pub struct SteepResult {
    pub memories_decayed: u64,
    pub recaps_generated: u32,
    pub distilled: u32,
    pub pending_remaining: u32,
    pub phases: Vec<PhaseResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::tests::{test_db, EVICT_ENV_LOCK};

    #[tokio::test]
    async fn changed_explicit_source_is_enqueued_without_running_an_llm() {
        let (db, _dir) = test_db().await;
        db.conn
            .lock()
            .await
            .execute(
                "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode)
                 VALUES ('row-source','source','memory','mem_source','source',0,200,
                         'text',0,0,'hide')",
                (),
            )
            .await
            .unwrap();
        db.insert_page(
            "page_changed",
            "Changed",
            None,
            "old content",
            None,
            None,
            &["mem_source"],
            "1970-01-01T00:01:40+00:00",
        )
        .await
        .unwrap();

        assert_eq!(enqueue_changed_pages(&db).await.unwrap(), 1);
        assert_eq!(
            db.get_page_stale_reason("page_changed").await.unwrap(),
            Some("source_updated".to_string())
        );
    }

    /// Minimal provider for routing-resolution tests: reports a fixed backend
    /// and model id; `generate` is never exercised by the pure route helpers.
    struct RouteTestProvider {
        backend: LlmBackend,
        model: &'static str,
        available: bool,
        panic_on_availability: bool,
    }

    impl RouteTestProvider {
        fn arc(backend: LlmBackend, model: &'static str) -> Arc<dyn LlmProvider> {
            Arc::new(Self {
                backend,
                model,
                available: true,
                panic_on_availability: false,
            })
        }

        fn unavailable(backend: LlmBackend, model: &'static str) -> Arc<dyn LlmProvider> {
            Arc::new(Self {
                backend,
                model,
                available: false,
                panic_on_availability: false,
            })
        }

        fn panicking(backend: LlmBackend, model: &'static str) -> Arc<dyn LlmProvider> {
            Arc::new(Self {
                backend,
                model,
                available: true,
                panic_on_availability: true,
            })
        }
    }

    struct SelectiveRedistillProvider {
        seen_prompts: std::sync::Mutex<Vec<String>>,
    }

    impl SelectiveRedistillProvider {
        fn new() -> Self {
            Self {
                seen_prompts: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn saw_good_page(&self) -> bool {
            self.seen_prompts
                .lock()
                .unwrap()
                .iter()
                .any(|prompt| prompt.contains("good-eleventh bounded source"))
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for SelectiveRedistillProvider {
        async fn generate(
            &self,
            request: crate::llm_provider::LlmRequest,
        ) -> Result<String, crate::llm_provider::LlmError> {
            self.seen_prompts
                .lock()
                .unwrap()
                .push(request.user_prompt.clone());
            if request.user_prompt.contains("good-eleventh bounded source") {
                Ok("good-eleventh bounded source [1]".to_string())
            } else {
                Err(crate::llm_provider::LlmError::InferenceFailed(
                    "poison page".to_string(),
                ))
            }
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "selective-redistill"
        }

        fn backend(&self) -> LlmBackend {
            LlmBackend::OnDevice
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for RouteTestProvider {
        async fn generate(
            &self,
            _request: crate::llm_provider::LlmRequest,
        ) -> Result<String, crate::llm_provider::LlmError> {
            unreachable!("route helpers never call generate()")
        }
        fn is_available(&self) -> bool {
            assert!(!self.panic_on_availability, "simulated availability panic");
            self.available
        }
        fn name(&self) -> &str {
            self.model
        }
        fn backend(&self) -> LlmBackend {
            self.backend
        }
        fn model_id(&self) -> String {
            self.model.to_string()
        }
    }

    #[test]
    fn everyday_prefers_anthropic_over_external_and_on_device() {
        let api = RouteTestProvider::arc(LlmBackend::Api, "claude-haiku");
        let ext = RouteTestProvider::arc(LlmBackend::Api, "ollama-llama3");
        let dev = RouteTestProvider::arc(LlmBackend::OnDevice, "qwen3-4b");
        let route = everyday_llm(Some(&api), Some(&ext), Some(&dev));
        assert_eq!(route.source, EverydaySource::Anthropic);
        assert_eq!(route.llm.unwrap().model_id(), "claude-haiku");
    }

    #[test]
    fn everyday_uses_external_when_no_anthropic() {
        // The un-trap: a connected external provider now serves everyday work
        // instead of falling through to the on-device model.
        let ext = RouteTestProvider::arc(LlmBackend::Api, "ollama-llama3");
        let dev = RouteTestProvider::arc(LlmBackend::OnDevice, "qwen3-4b");
        let route = everyday_llm(None, Some(&ext), Some(&dev));
        assert_eq!(route.source, EverydaySource::External);
        assert_eq!(route.llm.unwrap().model_id(), "ollama-llama3");
    }

    #[test]
    fn everyday_falls_to_on_device_then_basic() {
        let dev = RouteTestProvider::arc(LlmBackend::OnDevice, "qwen3-4b");
        let route = everyday_llm(None, None, Some(&dev));
        assert_eq!(route.source, EverydaySource::OnDevice);
        assert_eq!(route.llm.unwrap().model_id(), "qwen3-4b");

        let none = everyday_llm(None, None, None);
        assert_eq!(none.source, EverydaySource::Basic);
        assert!(none.llm.is_none());
    }

    #[test]
    fn synthesis_prefers_anthropic_then_external() {
        let synth = RouteTestProvider::arc(LlmBackend::Api, "claude-sonnet");
        let api = RouteTestProvider::arc(LlmBackend::Api, "claude-haiku");
        let ext = RouteTestProvider::arc(LlmBackend::Api, "ollama-llama3");

        // anthropic-only
        let r = synthesis_route(Some(&synth), Some(&api), None, None);
        assert_eq!(r.source, SynthesisSource::Anthropic);
        assert_eq!(r.llm.unwrap().model_id(), "claude-sonnet");

        // anthropic + external → anthropic (synthesis slot wins)
        let r = synthesis_route(Some(&synth), Some(&api), Some(&ext), None);
        assert_eq!(r.source, SynthesisSource::Anthropic);

        // external-only → external
        let r = synthesis_route(None, None, Some(&ext), None);
        assert_eq!(r.source, SynthesisSource::External);
        assert_eq!(r.llm.unwrap().model_id(), "ollama-llama3");
    }

    #[test]
    fn synthesis_on_device_honors_compile_gate() {
        let dev = RouteTestProvider::arc(LlmBackend::OnDevice, "qwen3-4b");

        // Gate OFF (default): on-device-only synthesis resolves to none.
        let r = temp_env::with_var("WENLAN_PREFER_ON_DEVICE_COMPILE", None::<&str>, || {
            let r = synthesis_route(None, None, None, Some(&dev));
            (r.source, r.llm.map(|l| l.model_id()))
        });
        assert_eq!(r.0, SynthesisSource::None);
        assert!(r.1.is_none());

        // Gate ON: on-device becomes the synthesis provider.
        let r = temp_env::with_var("WENLAN_PREFER_ON_DEVICE_COMPILE", Some("1"), || {
            let r = synthesis_route(None, None, None, Some(&dev));
            (r.source, r.llm.map(|l| l.model_id()))
        });
        assert_eq!(r.0, SynthesisSource::OnDevice);
        assert_eq!(r.1.as_deref(), Some("qwen3-4b"));

        // No cloud, no on-device → none regardless of gate.
        let r = synthesis_route(None, None, None, None);
        assert_eq!(r.source, SynthesisSource::None);
    }

    #[test]
    fn resolve_everyday_pin_on_device_while_anthropic_configured() {
        // THE headline case: everyday pinned to on-device even though an
        // Anthropic key is configured. Pre-pin, api_llm always won and this mix
        // was unreachable; the pin now makes it work.
        let api = RouteTestProvider::arc(LlmBackend::Api, "claude-haiku");
        let dev = RouteTestProvider::arc(LlmBackend::OnDevice, "qwen3-4b");
        let r = resolve_everyday(Some(EverydaySource::OnDevice), Some(&api), None, Some(&dev));
        assert_eq!(r.source, EverydaySource::OnDevice);
        assert_eq!(r.mode, RouteMode::Pinned);
        assert_eq!(r.llm.unwrap().model_id(), "qwen3-4b");
    }

    #[test]
    fn resolve_everyday_pin_absent_source_stays_unavailable_without_fallback() {
        // A pin is a spending/source boundary. Missing external must not cross
        // to the configured Anthropic or on-device providers.
        let api = RouteTestProvider::arc(LlmBackend::Api, "claude-haiku");
        let dev = RouteTestProvider::arc(LlmBackend::OnDevice, "qwen3-4b");
        let r = resolve_everyday(Some(EverydaySource::External), Some(&api), None, Some(&dev));
        assert_eq!(r.source, EverydaySource::External);
        assert_eq!(r.mode.as_str(), "pinned_unavailable");
        assert!(r.llm.is_none());
    }

    #[test]
    fn resolve_everyday_runtime_unavailable_pin_stays_pending_without_fallback() {
        let api = RouteTestProvider::arc(LlmBackend::Api, "claude-haiku");
        let external = RouteTestProvider::unavailable(LlmBackend::Api, "offline-external");
        let r = resolve_everyday(
            Some(EverydaySource::External),
            Some(&api),
            Some(&external),
            None,
        );
        assert_eq!(r.source, EverydaySource::External);
        assert_eq!(r.mode, RouteMode::PinnedUnavailable);
        assert!(r.llm.is_none(), "an unavailable pin must not be invoked");
    }

    #[test]
    fn resolve_everyday_availability_panic_fails_closed() {
        let provider = RouteTestProvider::panicking(LlmBackend::OnDevice, "panicking-local");
        let r = resolve_everyday(Some(EverydaySource::OnDevice), None, None, Some(&provider));
        assert_eq!(r.mode, RouteMode::PinnedUnavailable);
        assert!(r.llm.is_none());
    }

    #[test]
    fn resolve_everyday_no_pin_is_unconfigured_even_with_providers() {
        let api = RouteTestProvider::arc(LlmBackend::Api, "claude-haiku");
        let ext = RouteTestProvider::arc(LlmBackend::Api, "ollama-llama3");
        let r = resolve_everyday(None, Some(&api), Some(&ext), None);
        assert_eq!(r.source, EverydaySource::Basic);
        assert_eq!(r.mode.as_str(), "unconfigured");
        assert!(r.llm.is_none());
    }

    #[test]
    fn resolve_synthesis_pin_external_over_anthropic() {
        let synth = RouteTestProvider::arc(LlmBackend::Api, "claude-sonnet");
        let ext = RouteTestProvider::arc(LlmBackend::Api, "ollama-llama3");
        let r = resolve_synthesis(
            Some(SynthesisSource::External),
            Some(&synth),
            None,
            Some(&ext),
            None,
        );
        assert_eq!(r.source, SynthesisSource::External);
        assert_eq!(r.mode, RouteMode::Pinned);
        assert_eq!(r.llm.unwrap().model_id(), "ollama-llama3");
    }

    #[test]
    fn resolve_synthesis_pin_on_device_while_anthropic_configured() {
        // THE headline: synthesis pinned to on-device even though an Anthropic
        // slot is configured. The pin is honored with no compile gate — pins
        // carry no capability restriction (symmetric with everyday).
        let synth = RouteTestProvider::arc(LlmBackend::Api, "claude-sonnet");
        let api = RouteTestProvider::arc(LlmBackend::Api, "claude-haiku");
        let dev = RouteTestProvider::arc(LlmBackend::OnDevice, "qwen3-4b");
        let r = resolve_synthesis(
            Some(SynthesisSource::OnDevice),
            Some(&synth),
            Some(&api),
            None,
            Some(&dev),
        );
        assert_eq!(r.source, SynthesisSource::OnDevice);
        assert_eq!(r.mode, RouteMode::Pinned);
        assert_eq!(r.llm.unwrap().model_id(), "qwen3-4b");
    }

    #[test]
    fn resolve_synthesis_pin_on_device_absent_stays_unavailable_without_fallback() {
        // Missing on-device must not cross to the configured Anthropic source.
        let synth = RouteTestProvider::arc(LlmBackend::Api, "claude-sonnet");
        let r = resolve_synthesis(
            Some(SynthesisSource::OnDevice),
            Some(&synth),
            None,
            None,
            None,
        );
        assert_eq!(r.source, SynthesisSource::OnDevice);
        assert_eq!(r.mode.as_str(), "pinned_unavailable");
        assert!(r.llm.is_none());
    }

    #[test]
    fn resolve_synthesis_runtime_unavailable_pin_stays_pending_without_fallback() {
        let synth = RouteTestProvider::arc(LlmBackend::Api, "claude-sonnet");
        let external = RouteTestProvider::unavailable(LlmBackend::Api, "offline-external");
        let r = resolve_synthesis(
            Some(SynthesisSource::External),
            Some(&synth),
            None,
            Some(&external),
            None,
        );
        assert_eq!(r.source, SynthesisSource::External);
        assert_eq!(r.mode, RouteMode::PinnedUnavailable);
        assert!(r.llm.is_none(), "an unavailable pin must not be invoked");
    }

    #[test]
    fn resolve_synthesis_no_pin_is_unconfigured_even_with_providers() {
        let synth = RouteTestProvider::arc(LlmBackend::Api, "claude-sonnet");
        let ext = RouteTestProvider::arc(LlmBackend::Api, "ollama-llama3");
        let r = resolve_synthesis(None, Some(&synth), None, Some(&ext), None);
        assert_eq!(r.source, SynthesisSource::None);
        assert_eq!(r.mode.as_str(), "unconfigured");
        assert!(r.llm.is_none());
    }

    #[test]
    fn source_parse_rejects_unknown_and_non_pinnable() {
        assert_eq!(
            EverydaySource::parse(Some("on_device")),
            Some(EverydaySource::OnDevice)
        );
        assert_eq!(EverydaySource::parse(Some("basic")), None); // not pinnable
        assert_eq!(EverydaySource::parse(Some("bogus")), None);
        assert_eq!(EverydaySource::parse(None), None);
        assert_eq!(
            SynthesisSource::parse(Some("external")),
            Some(SynthesisSource::External)
        );
        assert_eq!(
            SynthesisSource::parse(Some("on_device")),
            Some(SynthesisSource::OnDevice)
        );
        assert_eq!(SynthesisSource::parse(Some("none")), None); // not pinnable
    }

    struct RecordingDistillProvider {
        prompts: std::sync::Mutex<Vec<String>>,
    }

    impl RecordingDistillProvider {
        fn new() -> Self {
            Self {
                prompts: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn prompts(&self) -> Vec<String> {
            self.prompts.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for RecordingDistillProvider {
        async fn generate(
            &self,
            request: crate::llm_provider::LlmRequest,
        ) -> Result<String, crate::llm_provider::LlmError> {
            let prompt = request.user_prompt;
            self.prompts.lock().unwrap().push(prompt.clone());
            if prompt.contains("[[Related Page]]") {
                Ok(
                    "Refreshed page prose keeps the valid [[Related Page]] wikilink. [1]"
                        .to_string(),
                )
            } else {
                Ok("Refreshed page prose drops the valid wikilink. [1]".to_string())
            }
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "recording-distill"
        }

        fn backend(&self) -> crate::llm_provider::LlmBackend {
            crate::llm_provider::LlmBackend::OnDevice
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    /// Records the `(system_prompt, label)` of every `generate` call, so a
    /// test driving the FULL `run_periodic_steep_with_api` Emergence phase
    /// (not the `distill_pages_scoped_gated` primitive directly) can assert
    /// whether the coherence-gate system prompt (`prompts.refine_clusters`)
    /// was ever sent — proving the backend-based routing decision at the
    /// Emergence call site, not just the gate parameter it forwards.
    /// Responds compile-plausibly enough to synthesize a real page: echoes
    /// the `distill_body`-labeled user prompt back with a trailing citation
    /// marker (passes the hallucination/faithfulness check trivially) and
    /// returns a short fixed title for any other (title-gen) call.
    struct EmergenceRoutingProvider {
        calls: std::sync::Mutex<Vec<(Option<String>, Option<String>)>>,
        backend: crate::llm_provider::LlmBackend,
    }

    impl EmergenceRoutingProvider {
        fn new(backend: crate::llm_provider::LlmBackend) -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
                backend,
            }
        }

        fn saw_system_prompt(&self, needle: &str) -> bool {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .any(|(sys, _)| sys.as_deref() == Some(needle))
        }

        fn saw_label(&self, needle: &str) -> bool {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .any(|(_, label)| label.as_deref() == Some(needle))
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for EmergenceRoutingProvider {
        async fn generate(
            &self,
            request: crate::llm_provider::LlmRequest,
        ) -> Result<String, crate::llm_provider::LlmError> {
            let label = request.label.clone();
            self.calls
                .lock()
                .unwrap()
                .push((request.system_prompt.clone(), label.clone()));
            if label.as_deref() == Some("distill_body") {
                Ok(format!("{} [1]", request.user_prompt))
            } else {
                Ok("Test Topic".to_string())
            }
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "emergence-routing"
        }

        fn backend(&self) -> crate::llm_provider::LlmBackend {
            self.backend
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }
    use crate::sources::{RawDocument, StabilityTier};
    use crate::synthesis::distill::apply_merge_by_tier;

    fn make_memory(source_id: &str, content: &str, memory_type: &str, space: &str) -> RawDocument {
        RawDocument {
            source_id: source_id.to_string(),
            content: content.to_string(),
            source: "memory".to_string(),
            title: content.chars().take(40).collect(),
            memory_type: Some(memory_type.to_string()),
            space: Some(space.to_string()),
            confidence: Some(0.7),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        }
    }

    #[test]
    fn test_trigger_kind_backstop_runs_all_phases() {
        for &phase in Phase::ALL {
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
        assert!(t.runs_phase(Phase::Recaps));
        assert!(t.runs_phase(Phase::RefinementQueue));
        // Should NOT run anything else
        assert!(!t.runs_phase(Phase::Decay));
        assert!(!t.runs_phase(Phase::Promote));
        assert!(!t.runs_phase(Phase::Emergence));
        assert!(!t.runs_phase(Phase::CommunityDetection));
        assert!(!t.runs_phase(Phase::Detect));
        assert!(!t.runs_phase(Phase::DecisionLogs));
        assert!(!t.runs_phase(Phase::PruneRejections));
        assert!(!t.runs_phase(Phase::Evict));
    }

    #[test]
    fn test_trigger_kind_idle_subset() {
        let t = TriggerKind::Idle;
        assert!(t.runs_phase(Phase::CommunityDetection));
        assert!(t.runs_phase(Phase::Detect));
        assert!(t.runs_phase(Phase::Emergence));
        assert!(t.runs_phase(Phase::ReDistill));
        assert!(t.runs_phase(Phase::Overview));
        assert!(t.runs_phase(Phase::DecisionLogs));
        // Should NOT run burst/maintenance/backfill phases
        assert!(!t.runs_phase(Phase::Recaps));
        assert!(!t.runs_phase(Phase::RefinementQueue));
        assert!(!t.runs_phase(Phase::Decay));
        assert!(!t.runs_phase(Phase::Promote));
        assert!(!t.runs_phase(Phase::Reweave));
        assert!(!t.runs_phase(Phase::Reembed));
        assert!(!t.runs_phase(Phase::EntityExtraction));
        assert!(!t.runs_phase(Phase::PruneRejections));
        assert!(!t.runs_phase(Phase::Evict));
    }

    #[test]
    fn test_trigger_kind_daily_subset() {
        let t = TriggerKind::Daily;
        assert!(t.runs_phase(Phase::Decay));
        assert!(t.runs_phase(Phase::Promote));
        assert!(t.runs_phase(Phase::Reweave));
        assert!(t.runs_phase(Phase::Reembed));
        assert!(t.runs_phase(Phase::EntityExtraction));
        assert!(t.runs_phase(Phase::Overview));
        assert!(t.runs_phase(Phase::PruneRejections));
        assert!(t.runs_phase(Phase::Evict));
        // Should NOT run synthesis or burst phases
        assert!(!t.runs_phase(Phase::Recaps));
        assert!(!t.runs_phase(Phase::Emergence));
        assert!(!t.runs_phase(Phase::Detect));
        assert!(!t.runs_phase(Phase::ReDistill));
        assert!(!t.runs_phase(Phase::DecisionLogs));
        assert!(!t.runs_phase(Phase::CommunityDetection));
        assert!(!t.runs_phase(Phase::RefinementQueue));
    }

    #[test]
    fn test_every_phase_has_non_backstop_trigger() {
        // Safety net: if a new phase is added to `Phase::ALL` but not assigned
        // to BurstEnd, Idle, or Daily, it silently becomes backstop-only
        // (running every 6 hours instead of at the right time). This test
        // catches that at compile/test time.
        let non_backstop = [TriggerKind::BurstEnd, TriggerKind::Idle, TriggerKind::Daily];
        for &phase in Phase::ALL {
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
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            TriggerKind::Backstop,
        )
        .await
        .unwrap();

        assert!(!result.phases.is_empty(), "Backstop should produce phases");
    }

    #[tokio::test]
    async fn background_steep_phase_slice_runs_only_the_requested_phase() {
        let _env_serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();
        db.upsert_documents(vec![make_memory(
            "phase_slice_smoke",
            "A background turn must return after one refinery phase.",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                let config = crate::config::Config {
                    page_map_auto_suggest: true,
                    ..crate::config::Config::default()
                };
                crate::config::save_config(&config).unwrap();

                run_periodic_steep_phase_with_api(
                    &db,
                    None,
                    None,
                    None,
                    None,
                    &PromptRegistry::default(),
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Backstop,
                    Phase::Decay,
                )
                .await
            })
            .await
            .unwrap();

        let phases: Vec<&str> = result
            .result
            .phases
            .iter()
            .map(|phase| phase.name.as_str())
            .collect();
        assert_eq!(phases, vec!["decay"]);
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
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
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
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
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
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
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

    // ── T21 Eviction phase wiring (B1-B3) ────────────────────────────────────

    // `WENLAN_ENABLE_EVICTION` is a process-global env var read synchronously
    // by `crate::db::eviction_enabled()` inside the Evict phase gate.
    // `temp_env::async_with_vars` mutates that global for the duration of a
    // future, but does not serialize against other tests doing the same (or
    // against tests that merely read the ambient value across an `.await`) —
    // two such tests running concurrently can corrupt each other's view of
    // the flag mid-run. `EVICT_ENV_LOCK` (imported above from
    // `crate::db::tests`, where `db.rs`'s own eviction tests also take it) is
    // the ONE process-wide lock every reader AND mutator of this flag shares
    // — not a second, module-local lock that would not exclude the db.rs
    // call sites. Mirrors the `PRF_ENV_LOCK` / `SALIENCE_ENV_LOCK` /
    // `RERANK_BLEND_ENV_LOCK` / `MAGNITUDE_ENV_LOCK` precedent in `db.rs`.

    // B1 — with WENLAN_ENABLE_EVICTION=1, Backstop runs 'evict' as a phase.
    #[tokio::test]
    async fn test_evict_runs_as_phase_when_enabled() {
        let _serial = EVICT_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let result = temp_env::async_with_vars([("WENLAN_ENABLE_EVICTION", Some("1"))], async {
            run_periodic_steep_with_api(
                &db,
                None,
                None,
                None,
                None,
                &PromptRegistry::default(),
                &crate::tuning::RefineryConfig::default(),
                &crate::tuning::ConfidenceConfig::default(),
                &crate::tuning::DistillationConfig::default(),
                None,
                TriggerKind::Backstop,
            )
            .await
            .unwrap()
        })
        .await;
        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();
        assert!(
            phase_names.contains(&"evict"),
            "WENLAN_ENABLE_EVICTION=1 Backstop should run 'evict', got {:?}",
            phase_names
        );
    }

    // B2 — with the flag unset, 'evict' never appears (default-OFF contract,
    // double-gated by trigger membership AND env flag).
    #[tokio::test]
    async fn test_evict_absent_when_flag_unset() {
        let _serial = EVICT_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let result = temp_env::async_with_vars([("WENLAN_ENABLE_EVICTION", None::<&str>)], async {
            run_periodic_steep_with_api(
                &db,
                None,
                None,
                None,
                None,
                &PromptRegistry::default(),
                &crate::tuning::RefineryConfig::default(),
                &crate::tuning::ConfidenceConfig::default(),
                &crate::tuning::DistillationConfig::default(),
                None,
                TriggerKind::Backstop,
            )
            .await
            .unwrap()
        })
        .await;
        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();
        assert!(
            !phase_names.contains(&"evict"),
            "flag unset must keep 'evict' absent even on Backstop, got {:?}",
            phase_names
        );
    }

    // B3 — even with the flag ON, BurstEnd never runs 'evict' (maintenance
    // belongs on Backstop/Daily only, like PruneRejections).
    #[tokio::test]
    async fn test_evict_not_run_on_burstend() {
        let _serial = EVICT_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let result = temp_env::async_with_vars([("WENLAN_ENABLE_EVICTION", Some("1"))], async {
            run_periodic_steep_with_api(
                &db,
                None,
                None,
                None,
                None,
                &PromptRegistry::default(),
                &crate::tuning::RefineryConfig::default(),
                &crate::tuning::ConfidenceConfig::default(),
                &crate::tuning::DistillationConfig::default(),
                None,
                TriggerKind::BurstEnd,
            )
            .await
            .unwrap()
        })
        .await;
        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();
        assert!(
            !phase_names.contains(&"evict"),
            "BurstEnd must NOT run 'evict' even with flag ON, got {:?}",
            phase_names
        );
    }

    #[tokio::test]
    async fn test_idle_trigger_runs_only_synthesis_phases() {
        // The page_maps phase is gated on `config.page_map_auto_suggest`,
        // which the steep reads via `load_config()` — pin WENLAN_DATA_DIR to
        // a temp config so the phase list doesn't depend on the host's
        // config.json (absent on CI → gate off → phase missing).
        let _env_serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();

        db.upsert_documents(vec![make_memory(
            "idle_test",
            "Test memory for idle trigger",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                let config = crate::config::Config {
                    page_map_auto_suggest: true,
                    ..crate::config::Config::default()
                };
                crate::config::save_config(&config).unwrap();

                run_periodic_steep_with_api(
                    &db,
                    None,
                    None,
                    None,
                    None,
                    &PromptRegistry::default(),
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Idle,
                )
                .await
            })
            .await
            .unwrap();

        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();

        // Idle subset
        let expected: &[&str] = &[
            "community_detection",
            "detect",
            "emergence",
            "re-distill",
            "overview",
            "decision_logs",
            "page_maps",
        ];
        for &exp in expected {
            assert!(
                phase_names.contains(&exp),
                "Idle should run {}, got {:?}",
                exp,
                phase_names
            );
        }
        let detect_pos = phase_names
            .iter()
            .position(|name| *name == "detect")
            .expect("detect phase should run on Idle");
        let emergence_pos = phase_names
            .iter()
            .position(|name| *name == "emergence")
            .expect("emergence phase should run on Idle");
        assert!(
            detect_pos < emergence_pos,
            "DETECT must run before compile/emergence, got {:?}",
            phase_names
        );

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
        // Daily runs the Evict phase gate (`TriggerKind::Daily.runs_phase`
        // includes `Phase::Evict`) and asserts an exact maintenance-phase
        // set that excludes 'evict' — it depends on the ambient
        // `WENLAN_ENABLE_EVICTION` staying unset for the test's duration, so
        // it must hold the same lock as the B1-B3 tests that mutate it.
        let _serial = EVICT_ENV_LOCK.lock().await;
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
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            TriggerKind::Daily,
        )
        .await
        .unwrap();

        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();

        // Daily subset — all maintenance phases including
        // prune_rejections (promoted to a tracked phase in Task 4).
        let expected: &[&str] = &[
            "decay",
            "promote",
            "reweave",
            "reembed",
            "entity_extraction",
            "overview",
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
                "Daily should only run maintenance phases, got unexpected: {}",
                name
            );
        }
    }

    #[tokio::test]
    async fn test_backstop_trigger_runs_all_phases() {
        // Backstop always runs the Evict phase gate and this test asserts
        // an EXACT phase count — same ambient-`WENLAN_ENABLE_EVICTION`
        // dependency as the Daily test above, same lock required. The
        // page_maps phase is additionally gated on
        // `config.page_map_auto_suggest` read via `load_config()`, so pin
        // WENLAN_DATA_DIR to a temp config (host config.json is absent on
        // CI → gate off → phase missing).
        let _serial = EVICT_ENV_LOCK.lock().await;
        let _env_serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();

        db.upsert_documents(vec![make_memory(
            "backstop_test",
            "Test memory for backstop trigger",
            "fact",
            "engineering",
        )])
        .await
        .unwrap();

        let result =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                let config = crate::config::Config {
                    page_map_auto_suggest: true,
                    ..crate::config::Config::default()
                };
                crate::config::save_config(&config).unwrap();

                run_periodic_steep_with_api(
                    &db,
                    None,
                    None,
                    None,
                    None,
                    &PromptRegistry::default(),
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Backstop,
                )
                .await
            })
            .await
            .unwrap();

        let phase_names: Vec<&str> = result.phases.iter().map(|p| p.name.as_str()).collect();

        // All phases must run with Backstop. `kg_rethink` is
        // rate-limited, so on a fresh DB (last_kg_rethink_ts=0) it runs
        // on the first steep.
        let expected: &[&str] = &[
            "decay",
            "promote",
            "recaps",
            "reweave",
            "reembed",
            "entity_extraction",
            "community_detection",
            "detect",
            "emergence",
            "re-distill",
            "overview",
            "refinement_queue",
            "decision_logs",
            "page_maps",
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
            None,
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
        db.resolve_refinement_if_open("e2e_ref", "auto_applied")
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
            None,
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
            None,
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
        // Should have at least decay, recaps, reweave, reembed, entity_extraction, distillation, refinement_queue, decision_logs
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
    async fn no_lane_compile_leaves_pending_and_persists_queue_depth() {
        let (db, _dir) = test_db().await;

        // A well-formed, eligible cluster (3+ memories, shared space).
        for (i, content) in [
            "The wenlan daemon persists document chunks in a libSQL table using an F32_BLOB column for the 768-dimension embedding vector, with DiskANN indexing enabling fast approximate nearest-neighbor search across the whole memory store.",
            "libSQL's FTS5 virtual table stays synchronized with the chunks table through SQL triggers, so every insert or update to a chunk automatically refreshes its full-text search index without extra application code.",
            "Hybrid retrieval combines the vector similarity score and the FTS5 rank using reciprocal rank fusion, blending semantic and lexical signals into one ranked result list for each search query.",
        ]
        .iter()
        .enumerate()
        {
            let doc = crate::sources::RawDocument {
                source: "memory".to_string(),
                source_id: format!("nolane_{}", i),
                title: content.to_string(),
                content: content.to_string(),
                space: Some("architecture".to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // No cloud, no agent-lane consumer yet, and the on-device engine is
        // configured but unhealthy (is_available() == false) — the "no lane"
        // case per spec §3.1/§7: clusters must WAIT as pending, nothing
        // half-written.
        let unhealthy: Arc<dyn LlmProvider> =
            Arc::new(crate::llm_provider::MockProvider::unavailable());
        let result = distill_pages_scoped(
            &db,
            Some(&unhealthy),
            &PromptRegistry::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            None,
        )
        .await
        .unwrap();
        assert!(
            result.created.is_empty(),
            "no-lane compile must not write a partial page"
        );
        assert!(
            !result.pending.is_empty(),
            "fixture cluster should be queued as pending, not silently dropped"
        );

        persist_compile_queue_depth(&db, result.pending.len())
            .await
            .unwrap();
        let depth = compile_queue_depth(&db).await.unwrap();
        assert_eq!(
            depth,
            result.pending.len(),
            "/api/status must be able to read back the persisted compile queue depth"
        );
    }

    static COMPILE_ROUTING_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    async fn seed_emergence_routing_fixture(db: &MemoryDB, prefix: &str) {
        for (i, content) in [
            "The daemon persists chunks in a vector-enabled libSQL table and keeps lexical indexes synchronized through database triggers for dependable hybrid retrieval.",
            "Hybrid retrieval combines semantic similarity with lexical rank through reciprocal rank fusion so exact identifiers and paraphrased concepts remain discoverable together.",
            "Durable source evidence lets synthesized pages cite the memories they were compiled from and be rebuilt when those source memories change.",
        ]
        .iter()
        .enumerate()
        {
            db.upsert_documents(vec![crate::sources::RawDocument {
                source: "memory".to_string(),
                source_id: format!("{prefix}_{i}"),
                title: content.to_string(),
                content: content.to_string(),
                space: Some("hard-pin-routing".to_string()),
                ..Default::default()
            }])
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn emergence_tick_without_synthesis_pin_does_not_call_configured_cloud_provider() {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        seed_emergence_routing_fixture(&db, "no_synthesis_pin").await;

        let provider = Arc::new(EmergenceRoutingProvider::new(LlmBackend::Api));
        let synthesis: Arc<dyn LlmProvider> = provider.clone();
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();

        let result =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                crate::config::save_config(&crate::config::Config::default()).unwrap();
                run_periodic_steep_with_api(
                    &db,
                    None,
                    None,
                    Some(&synthesis),
                    None,
                    &PromptRegistry::default(),
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Idle,
                )
                .await
            })
            .await
            .unwrap();

        let emergence = result
            .phases
            .iter()
            .find(|phase| phase.name == "emergence")
            .expect("Idle must run emergence");
        assert!(emergence.error.is_none());
        assert!(
            !provider.saw_label("distill_body"),
            "configured cloud capability without an explicit synthesis pin must not authorize background page synthesis"
        );
        assert!(
            compile_queue_depth(&db).await.unwrap() > 0,
            "eligible page work must remain durable while synthesis consent is unconfigured"
        );
    }

    #[tokio::test]
    async fn idle_tick_without_everyday_pin_does_not_call_ondevice_decision_log_provider() {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().timestamp();
        for i in 0..4 {
            let mut doc = make_memory(
                &format!("hard_pin_decision_{i}"),
                &format!("Decided to use bounded background scheduling approach {i}"),
                "decision",
                "scheduler",
            );
            doc.last_modified = now - 60 * i as i64;
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        let provider = Arc::new(EmergenceRoutingProvider::new(LlmBackend::OnDevice));
        let everyday: Arc<dyn LlmProvider> = provider.clone();
        let prompts = PromptRegistry::default();
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();

        temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
            crate::config::save_config(&crate::config::Config::default()).unwrap();
            run_periodic_steep_with_api(
                &db,
                Some(&everyday),
                None,
                None,
                None,
                &prompts,
                &crate::tuning::RefineryConfig::default(),
                &crate::tuning::ConfidenceConfig::default(),
                &crate::tuning::DistillationConfig::default(),
                None,
                TriggerKind::Idle,
            )
            .await
        })
        .await
        .unwrap();

        assert!(
            !provider.saw_system_prompt(&prompts.summarize_decisions),
            "an installed on-device model without an everyday pin must not authorize background decision-log inference"
        );
    }

    #[tokio::test]
    async fn burst_end_without_everyday_pin_does_not_call_refinement_provider() {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        db.upsert_documents(vec![
            make_memory(
                "hard_pin_existing",
                "The scheduler cooldown is ten minutes.",
                "fact",
                "scheduler",
            ),
            make_memory(
                "hard_pin_incoming",
                "The scheduler cooldown is two minutes.",
                "fact",
                "scheduler",
            ),
        ])
        .await
        .unwrap();
        db.insert_refinement_proposal(
            "hard_pin_refinement",
            "detect_contradiction",
            &["hard_pin_incoming".into(), "hard_pin_existing".into()],
            None,
            0.9,
        )
        .await
        .unwrap();

        let provider = Arc::new(EmergenceRoutingProvider::new(LlmBackend::OnDevice));
        let everyday: Arc<dyn LlmProvider> = provider.clone();
        let prompts = PromptRegistry::default();
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();

        temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
            crate::config::save_config(&crate::config::Config::default()).unwrap();
            run_periodic_steep_with_api(
                &db,
                Some(&everyday),
                None,
                None,
                None,
                &prompts,
                &crate::tuning::RefineryConfig::default(),
                &crate::tuning::ConfidenceConfig::default(),
                &crate::tuning::DistillationConfig::default(),
                None,
                TriggerKind::BurstEnd,
            )
            .await
        })
        .await
        .unwrap();

        assert!(
            !provider.saw_system_prompt(&prompts.detect_contradiction),
            "an installed provider without an everyday pin must not judge background refinement proposals"
        );
    }

    #[tokio::test]
    async fn emergence_tick_without_cloud_defers_healthy_ondevice_engine_to_agent_lane_by_default()
    {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;

        let db_topic: Vec<(&str, &str)> = vec![
            ("topic_db", "The wenlan daemon persists document chunks in a libSQL table using an F32_BLOB column for the 768-dimension embedding vector, with DiskANN indexing enabling fast approximate nearest-neighbor search across the whole memory store."),
            ("topic_db", "libSQL's FTS5 virtual table stays synchronized with the chunks table through SQL triggers, so every insert or update to a chunk automatically refreshes its full-text search index without extra application code."),
            ("topic_db", "Hybrid retrieval combines the vector similarity score and the FTS5 rank using reciprocal rank fusion, blending semantic and lexical signals into one ranked result list for each search query."),
            ("topic_bread", "A sourdough levain needs to be fed with equal parts flour and water twice a day to stay active enough to leaven a loaf, and it should roughly double in volume within four to six hours after each feeding."),
            ("topic_bread", "Increasing the hydration of a bread dough to around eighty percent produces a much more open, irregular crumb structure once it is baked, though it also makes the dough considerably harder to shape by hand."),
            ("topic_bread", "Kneading a wheat dough develops long gluten strands that trap the carbon dioxide bubbles produced by yeast fermentation, which is what allows the loaf to rise instead of collapsing in the oven."),
        ];
        for (i, (space, content)) in db_topic.iter().enumerate() {
            let doc = crate::sources::RawDocument {
                source: "memory".to_string(),
                source_id: format!("emergence_gate_{}", i),
                title: content.to_string(),
                content: content.to_string(),
                space: Some(space.to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        let provider = Arc::new(EmergenceRoutingProvider::new(LlmBackend::OnDevice));
        let llm: Arc<dyn LlmProvider> = provider.clone();
        let prompts = PromptRegistry::default();

        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            Some(&llm),
            None,
            &prompts,
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            TriggerKind::Idle,
        )
        .await
        .unwrap();

        let emergence = result
            .phases
            .iter()
            .find(|p| p.name == "emergence")
            .expect("Idle trigger must run the emergence phase");
        assert!(
            emergence.error.is_none(),
            "emergence phase must not error: {:?}",
            emergence.error
        );
        assert!(
            !provider.saw_label("distill_body"),
            "without cloud, default routing must leave eligible clusters pending for the agent lane instead of invoking the healthy on-device provider"
        );
        // The eligible cluster must NOT have grown a page: check the cluster
        // page by title, not total active-page count. The ReDistill phase's
        // reserved "Overview" page (`ensure_overview_page`) is always minted
        // when an LLM is available, so a raw count is never 0 and would test
        // the wrong thing. Mirrors the sibling default-routing test, which
        // asserts the same "Test Topic" cluster page IS present.
        let cluster_page = db.find_active_page_id_by_title("Test Topic").await.unwrap();
        assert!(
            cluster_page.is_none(),
            "default routing must not synthesize the cluster page via the healthy on-device engine"
        );
        let queue_depth_after = compile_queue_depth(&db).await.unwrap();
        assert!(
            queue_depth_after > 0,
            "default routing must persist queued clusters for /distill, got queue depth {queue_depth_after}"
        );
    }

    #[tokio::test]
    async fn emergence_tick_with_on_device_pin_uses_healthy_ondevice_engine() {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;

        for (i, content) in [
            "The wenlan daemon persists document chunks in a libSQL table using an F32_BLOB column for the 768-dimension embedding vector, with DiskANN indexing enabling fast approximate nearest-neighbor search across the whole memory store.",
            "libSQL's FTS5 virtual table stays synchronized with the chunks table through SQL triggers, so every insert or update to a chunk automatically refreshes its full-text search index without extra application code.",
            "Hybrid retrieval combines the vector similarity score and the FTS5 rank using reciprocal rank fusion, blending semantic and lexical signals into one ranked result list for each search query.",
        ]
        .iter()
        .enumerate()
        {
            let doc = crate::sources::RawDocument {
                source: "memory".to_string(),
                source_id: format!("ondevice_default_{}", i),
                title: content.to_string(),
                content: content.to_string(),
                space: Some("architecture".to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        let provider = Arc::new(EmergenceRoutingProvider::new(LlmBackend::OnDevice));
        let llm: Arc<dyn LlmProvider> = provider.clone();
        let prompts = PromptRegistry::default();
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();

        let result =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                crate::config::save_config(&crate::config::Config {
                    synthesis_source: Some("on_device".to_string()),
                    ..crate::config::Config::default()
                })
                .unwrap();
                run_periodic_steep_with_api(
                    &db,
                    Some(&llm),
                    None,
                    None,
                    None,
                    &prompts,
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Idle,
                )
                .await
            })
            .await
            .unwrap();

        let emergence = result
            .phases
            .iter()
            .find(|p| p.name == "emergence")
            .expect("Idle trigger must run the emergence phase");
        assert!(
            emergence.error.is_none(),
            "emergence phase must not error: {:?}",
            emergence.error
        );
        assert!(
            !provider.saw_system_prompt(&prompts.refine_clusters),
            "an explicitly pinned on-device compile must skip the LLM coherence gate"
        );
        assert!(
            provider.saw_label("distill_body"),
            "an on-device synthesis pin must invoke the healthy on-device provider \
             to compile eligible clusters"
        );
        let cluster_page = db.find_active_page_id_by_title("Test Topic").await.unwrap();
        assert!(
            cluster_page.is_some(),
            "pinned on-device routing must synthesize a page from the eligible cluster \
             via the healthy on-device engine"
        );
        let queue_depth_after = compile_queue_depth(&db).await.unwrap();
        assert_eq!(
            queue_depth_after, 0,
            "pinned on-device routing must resolve eligible clusters through the healthy on-device engine, \
             got queue depth {queue_depth_after}"
        );
    }

    /// Companion to the OnDevice case above: drives the SAME Emergence call
    /// site with no available lane (configured on-device engine that is
    /// unhealthy) and asserts the queue-depth persist call at the end of the
    /// Emergence phase actually ran. Dropping `persist_compile_queue_depth`
    /// from the Emergence phase would leave `compile_queue_depth` at its
    /// default-0 read and make this test fail.
    #[tokio::test]
    async fn emergence_tick_with_unhealthy_engine_persists_pending_queue_depth() {
        let (db, _dir) = test_db().await;

        // A well-formed, eligible cluster (3+ memories, shared space).
        for (i, content) in [
            "libSQL stores vectors using F32_BLOB columns",
            "libSQL uses DiskANN for vector indexing",
            "libSQL supports FTS5 full-text search via triggers",
        ]
        .iter()
        .enumerate()
        {
            let doc = crate::sources::RawDocument {
                source: "memory".to_string(),
                source_id: format!("emergence_nolane_{}", i),
                title: content.to_string(),
                content: content.to_string(),
                space: Some("architecture".to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        assert_eq!(
            compile_queue_depth(&db).await.unwrap(),
            0,
            "queue depth should read back 0 before any compile tick has run"
        );

        let unhealthy: Arc<dyn LlmProvider> =
            Arc::new(crate::llm_provider::MockProvider::unavailable());
        let prompts = PromptRegistry::default();

        let result = run_periodic_steep_with_api(
            &db,
            None,
            None,
            Some(&unhealthy),
            None,
            &prompts,
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
            TriggerKind::Idle,
        )
        .await
        .unwrap();

        let emergence = result
            .phases
            .iter()
            .find(|p| p.name == "emergence")
            .expect("Idle trigger must run the emergence phase");
        assert!(
            emergence.error.is_none(),
            "emergence phase must not error: {:?}",
            emergence.error
        );

        let depth = compile_queue_depth(&db).await.unwrap();
        assert!(
            depth > 0,
            "a full Emergence tick with no available lane must persist a non-zero compile queue depth, got {depth}"
        );
    }

    /// Spec §5.3: the reserved Overview page is a "wiki is alive" signal that
    /// must actually fire in production, not just in `synthesis::overview`'s
    /// own unit tests. Drives the real maintenance entrypoint
    /// (`run_periodic_steep_with_api`, the function the daemon scheduler
    /// calls) with a trigger that runs `Phase::Overview` and an available
    /// LLM, and asserts the reserved "Overview" page exists afterward.
    #[tokio::test]
    async fn test_overview_phase_refreshes_reserved_overview_page() {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let data_dir = tempfile::tempdir().unwrap();
        let knowledge_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();
        let knowledge_path = knowledge_dir.path().to_path_buf();

        // A pre-existing page for the Overview to summarize — mirrors
        // `synthesis::overview`'s own fixture helper.
        let mem_content = "Rust is a systems programming language with memory safety guarantees";
        db.upsert_documents(vec![make_memory(
            "overview_wiring_rust",
            mem_content,
            "fact",
            "engineering",
        )])
        .await
        .unwrap();
        let req = wenlan_types::requests::CreateConceptRequest {
            title: "Rust".to_string(),
            content: mem_content.to_string(),
            summary: None,
            entity_id: None,
            space: None,
            source_memory_ids: vec!["overview_wiring_rust".to_string()],
            creation_kind: Some("research".to_string()),
            workspace: None,
        };
        crate::post_write::create_page(&db, req, "test", None)
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(crate::llm_provider::MockProvider::new(&format!(
            "{mem_content}.[1]"
        )));

        let result =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                let config = crate::config::Config {
                    knowledge_path: Some(knowledge_path),
                    synthesis_source: Some("on_device".to_string()),
                    ..crate::config::Config::default()
                };
                crate::config::save_config(&config).unwrap();

                run_periodic_steep_with_api(
                    &db,
                    Some(&llm),
                    None,
                    None,
                    None,
                    &PromptRegistry::default(),
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Idle,
                )
                .await
            })
            .await
            .unwrap();

        let overview = result
            .phases
            .iter()
            .find(|p| p.name == "overview")
            .expect("Idle trigger must run the overview phase");
        assert!(
            overview.error.is_none(),
            "overview phase must not error: {:?}",
            overview.error
        );

        let overview_id = db.find_active_page_id_by_title("Overview").await.unwrap();
        assert!(
            overview_id.is_some(),
            "the overview phase must create/refresh the reserved Overview page \
             (spec §5.3) — this is the 'wiki is alive' signal and must fire from the real \
             steep cycle, not just from synthesis::overview's own unit tests"
        );
    }

    #[tokio::test]
    async fn test_daily_maintenance_refreshes_reserved_overview_page() {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let data_dir = tempfile::tempdir().unwrap();
        let knowledge_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();
        let knowledge_path = knowledge_dir.path().to_path_buf();

        let mem_content = "Rust ownership lets the compiler enforce aliasing and mutation rules";
        db.upsert_documents(vec![make_memory(
            "overview_daily_rust",
            mem_content,
            "fact",
            "engineering",
        )])
        .await
        .unwrap();
        let req = wenlan_types::requests::CreateConceptRequest {
            title: "Rust Ownership".to_string(),
            content: mem_content.to_string(),
            summary: None,
            entity_id: None,
            space: None,
            source_memory_ids: vec!["overview_daily_rust".to_string()],
            creation_kind: Some("research".to_string()),
            workspace: None,
        };
        crate::post_write::create_page(&db, req, "test", None)
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(crate::llm_provider::MockProvider::new(&format!(
            "{mem_content}.[1]"
        )));

        let result =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                let config = crate::config::Config {
                    knowledge_path: Some(knowledge_path),
                    synthesis_source: Some("on_device".to_string()),
                    ..crate::config::Config::default()
                };
                crate::config::save_config(&config).unwrap();

                run_periodic_steep_with_api(
                    &db,
                    Some(&llm),
                    None,
                    None,
                    None,
                    &PromptRegistry::default(),
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Daily,
                )
                .await
            })
            .await
            .unwrap();

        let overview = result
            .phases
            .iter()
            .find(|phase| phase.name == "overview")
            .unwrap_or_else(|| {
                let phase_names: Vec<&str> =
                    result.phases.iter().map(|phase| phase.name.as_str()).collect();
                panic!(
                    "Daily maintenance must run the reserved Overview refresh phase, got {phase_names:?}"
                )
            });
        assert!(
            overview.error.is_none(),
            "Daily Overview phase must not error: {:?}",
            overview.error
        );

        let overview_id = db.find_active_page_id_by_title("Overview").await.unwrap();
        assert!(
            overview_id.is_some(),
            "the Daily maintenance pass must create/refresh the reserved Overview page \
             (spec §5.3) through the real steep cycle"
        );
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
                space: Some("architecture".to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // Run distillation (no LLM — should skip gracefully, return 0)
        let result = distill_pages(
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
        let concepts = db.list_pages("active", 100, 0).await.unwrap();
        assert_eq!(concepts.len(), 0);
    }

    #[tokio::test]
    async fn test_concept_lifecycle_integration() {
        let (db, _dir) = test_db().await;

        // 1. Insert memories that form a cluster
        for (i, content) in [
            "Wenlan uses libSQL for all storage including vectors and FTS",
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
                space: Some("architecture".to_string()),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // 2. Verify no concepts yet
        let concepts = db.list_pages("active", 100, 0).await.unwrap();
        assert_eq!(concepts.len(), 0);

        // 3. Run distillation (no LLM — graceful skip)
        let distilled = distill_pages(
            &db,
            None,
            &PromptRegistry::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
        )
        .await
        .unwrap();
        assert_eq!(distilled, 0);

        // 4. Verify page CRUD works end-to-end
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
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

        // 5. Verify page search
        let found = db.search_pages("libSQL storage", 10, None).await.unwrap();
        assert!(!found.is_empty());

        // 6. Verify page by entity lookup
        let by_entity = db.get_page_by_entity("entity_libsql_lc").await.unwrap();
        assert!(by_entity.is_some());

        // 7. Update and verify version increment
        db.update_page_content(
            "c_int",
            "## Key Facts\n- updated",
            &["lifecycle_0", "lifecycle_1", "lifecycle_2", "lifecycle_3"],
            "page_growth",
        )
        .await
        .unwrap();
        let updated = db.get_page("c_int").await.unwrap().unwrap();
        assert_eq!(updated.version, 2);
        assert_eq!(updated.source_memory_ids.len(), 4);

        // 8. Archive and verify
        db.archive_page("c_int").await.unwrap();
        let active = db.list_pages("active", 100, 0).await.unwrap();
        assert_eq!(active.len(), 0);
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
    fn test_classify_emergence_silent_when_zero() {
        assert_eq!(classify_emergence(0), (Nudge::Silent, None));
    }

    #[test]
    fn test_classify_emergence_wow_when_new_concepts() {
        let (nudge, headline) = classify_emergence(1);
        assert_eq!(nudge, Nudge::Wow);
        let h = headline.expect("headline should be set");
        assert!(
            h.contains("new page"),
            "headline should mention a new page: {}",
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
            "multi-page headline should mention count: {}",
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

    // ── run_phase Nudge propagation tests (Task 2) ──────────────────────

    #[tokio::test]
    async fn test_run_phase_propagates_nudge_and_headline() {
        let result = run_phase(Phase::Decay, || async {
            Ok(PhaseOutput {
                items_processed: 5,
                nudge: Nudge::Wow,
                headline: Some("Wenlan did a wow thing".to_string()),
            })
        })
        .await;

        assert_eq!(result.name, "decay");
        assert_eq!(result.items_processed, 5);
        assert_eq!(result.nudge, Nudge::Wow);
        assert_eq!(result.headline.as_deref(), Some("Wenlan did a wow thing"));
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_run_phase_on_error_is_silent() {
        let result = run_phase(Phase::Decay, || async {
            Err::<PhaseOutput, _>(crate::error::WenlanError::VectorDb(
                "test failure".to_string(),
            ))
        })
        .await;

        assert_eq!(result.name, "decay");
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
            None,
            &PromptRegistry::default(),
            &crate::tuning::RefineryConfig::default(),
            &crate::tuning::ConfidenceConfig::default(),
            &crate::tuning::DistillationConfig::default(),
            None,
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

        let (nudge, headline) = crate::synthesis::recaps::classify_recaps(2);
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
        // All single-word generic placeholders must still be rejected.
        assert!(is_all_generic_tokens("general"));
        assert!(is_all_generic_tokens("General"));
        assert!(is_all_generic_tokens("GENERAL"));
        assert!(is_all_generic_tokens("untitled"));
        assert!(is_all_generic_tokens("topic"));
        assert!(is_all_generic_tokens("cluster"));
        assert!(is_all_generic_tokens("misc"));
        assert!(is_all_generic_tokens("other"));
        assert!(is_all_generic_tokens("unknown"));
        // Note: "page" no longer in wordlist (false-positive risk on
        // legitimate technical titles); not rejected by is_all_generic_tokens.
        // True negatives — these SHOULD pass through.
        assert!(!is_all_generic_tokens("General AI Architecture"));
        assert!(!is_all_generic_tokens("Wenlan Concept Model"));
        assert!(!is_all_generic_tokens("libSQL Storage"));
    }

    #[test]
    fn is_all_generic_tokens_rejects_multi_word_generic() {
        // Single word generics — preserved behavior
        assert!(is_all_generic_tokens("general"));
        assert!(is_all_generic_tokens("GENERAL"));
        assert!(is_all_generic_tokens("topic"));
        assert!(is_all_generic_tokens("untitled"));

        // Multi-word all-generic — NEW behavior
        assert!(is_all_generic_tokens("General topic"));
        assert!(is_all_generic_tokens("Various Notes"));
        assert!(is_all_generic_tokens("Topic Notes"));
        assert!(is_all_generic_tokens("Misc Things"));
        assert!(is_all_generic_tokens("Misc-things")); // hyphen stripped
        assert!(is_all_generic_tokens("random assorted stuff"));

        // True negatives — must keep
        assert!(!is_all_generic_tokens("Topic and Notes")); // 'and' saves it
        assert!(!is_all_generic_tokens("libsql Vector Storage"));
        assert!(!is_all_generic_tokens("Wenlan Memory Layer"));
        assert!(!is_all_generic_tokens("Wenlan Concept Model")); // wordlist excludes 'page'
        assert!(!is_all_generic_tokens("Notes on Wenlan")); // 'on', 'origin' not generic
        assert!(!is_all_generic_tokens("Content Strategy")); // 'content' not in list, 'strategy' not in list

        // Edge cases
        assert!(!is_all_generic_tokens("")); // empty
        assert!(!is_all_generic_tokens("   ")); // whitespace only
        assert!(!is_all_generic_tokens("!!! ???")); // empty after punctuation strip

        // CJK generics now in wordlist
        assert!(is_all_generic_tokens("杂项"));
        assert!(is_all_generic_tokens("其他"));
        assert!(is_all_generic_tokens("通用"));

        // Mixed-script content with markup is rejected by looks_like_markup_styled,
        // not by is_all_generic_tokens. The latter is intentionally an English-leaning
        // wordlist; markup detection handles the canonical Roland case below.
        assert!(!is_all_generic_tokens("**Roland** — 太正統，d-L 連接快"));
    }

    #[test]
    fn looks_like_markup_styled_catches_canonical_bad_titles() {
        // Canonical Mode A example from the bug report — LLM emitted markdown
        // bold + em-dash + mixed CJK content as a title.
        assert!(looks_like_markup_styled("**Roland** — 太正統，d-L 連接快"));

        // Other markdown-styled hallucinations
        assert!(looks_like_markup_styled("**Important Note**"));
        assert!(looks_like_markup_styled("__bold__"));
        assert!(looks_like_markup_styled("~~strikethrough~~"));
        assert!(looks_like_markup_styled("[[wikilink]]"));
        assert!(looks_like_markup_styled("# Heading"));
        assert!(looks_like_markup_styled("Foo — Bar")); // em-dash separator

        // Clean titles must pass through
        assert!(!looks_like_markup_styled("Wenlan Memory Layer"));
        assert!(!looks_like_markup_styled("libSQL Storage"));
        assert!(!looks_like_markup_styled("Concept Distillation Pipeline"));
        assert!(!looks_like_markup_styled("Range: 1-10")); // ASCII hyphen ok
        assert!(!looks_like_markup_styled("Year 2026–2027")); // en-dash ok
        assert!(!looks_like_markup_styled("")); // empty
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

    #[test]
    fn test_refine_clusters_parses_array_response() {
        let raw = r#"[{"action":"merge","clusters":[0,1]},{"action":"keep","clusters":[2]}]"#;
        let extracted = crate::engine::extract_json_array(raw).unwrap();
        let actions: Vec<serde_json::Value> = serde_json::from_str(&extracted).unwrap();
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0]["action"], "merge");
        assert_eq!(actions[1]["action"], "keep");
    }

    #[tokio::test]
    async fn re_distill_stale_pages_re_projects_md_when_path_passed() {
        use crate::llm_provider::MockProvider;
        use tempfile::TempDir;

        let (db, _db_dir) = test_db().await;
        let knowledge_dir = TempDir::new().unwrap();
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();

        // Seed memory row so the re-distill can read a source. Content shares
        // tokens with the mock body ("refreshed body") so the fail-closed
        // faithfulness gate passes.
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source) \
                 VALUES (?1, ?1, ?1, 'refreshed body reference material', 0, 'text', 'fact', 'test', 'claude-code', ?2, ?2, 1, 'confirmed', 'memory')",
                libsql::params!["mem_seed".to_string(), now_ts],
            )
            .await
            .unwrap();
        }

        db.insert_page(
            "page_stale",
            "Stale Topic",
            None,
            "original body",
            None,
            None,
            &["mem_seed"],
            &now,
        )
        .await
        .unwrap();
        db.set_page_stale("page_stale", "source_updated")
            .await
            .unwrap();

        // The single `refresh_page` op now owns the citation fail-closed gate;
        // a write-path fixture must include a verifiable source marker.
        let llm: Arc<dyn LlmProvider> =
            Arc::new(MockProvider::new("refreshed body reference material [1]"));
        let prompts = PromptRegistry::default();

        let recompiled =
            re_distill_stale_pages(&db, Some(&llm), &prompts, Some(knowledge_dir.path()))
                .await
                .unwrap();
        assert_eq!(recompiled, 1, "stale page should be re-distilled");

        // md projection should land in knowledge_dir.
        let entries: Vec<_> = std::fs::read_dir(knowledge_dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
            .collect();
        assert_eq!(entries.len(), 1, "exactly one md file written");
        let content = std::fs::read_to_string(entries[0].path()).unwrap();
        assert!(
            content.contains("refreshed body"),
            "md body should reflect LLM output, got: {}",
            content
        );
    }

    /// Spec §5.1/§5.2/§6.3: a human-owned page must always get a revision
    /// card, never an in-place write, for EVERY machine write path -- the
    /// daemon staleness sweep included. Pins the gate at the sweep's own
    /// boundary so a future edit to the sweep can't silently reintroduce a
    /// bypass around `refresh_page`'s ownership check.
    #[tokio::test]
    async fn re_distill_stale_pages_user_edited_stages_revision_card_not_source_conflict() {
        use crate::llm_provider::MockProvider;

        let (db, _db_dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();

        // Seed source memory so refresh_page has content to synthesize from.
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source) \
                 VALUES (?1, ?1, ?1, 'refreshed body reference material', 0, 'text', 'fact', 'test', 'claude-code', ?2, ?2, 1, 'confirmed', 'memory')",
                libsql::params!["mem_owned".to_string(), now_ts],
            )
            .await
            .unwrap();
        }

        db.insert_page(
            "page_owned",
            "Owned Topic",
            None,
            "original body",
            None,
            None,
            &["mem_owned"],
            &now,
        )
        .await
        .unwrap();

        // Human edits the page (sets user_edited=1); the source changes again
        // afterward, marking it stale -- exactly the scenario the sweep exists
        // to handle.
        db.update_page_content("page_owned", "human-edited body", &["mem_owned"], "fs_edit")
            .await
            .unwrap();
        db.set_page_stale("page_owned", "source_updated")
            .await
            .unwrap();

        let before = db.get_page("page_owned").await.unwrap().unwrap();
        assert!(before.user_edited, "precondition: page is human-owned");

        // The ownership gate lives after `refresh_page`'s citation verifier, so
        // the fixture must first pass citation verification to reach the gate.
        let llm: Arc<dyn LlmProvider> =
            Arc::new(MockProvider::new("refreshed body reference material [1]"));
        let prompts = PromptRegistry::default();

        let recompiled = re_distill_stale_pages(&db, Some(&llm), &prompts, None)
            .await
            .unwrap();
        assert_eq!(recompiled, 0, "a gated write must not count as a recompile");

        let after = db.get_page("page_owned").await.unwrap().unwrap();
        assert_eq!(
            after.content, before.content,
            "the staleness sweep must never overwrite human-owned page prose"
        );
        assert_eq!(
            after.stale_reason, None,
            "the sweep must clear staleness via the ownership gate outcome, not escalate \
             to a dead-end 'source_conflict' state that no human-facing surface reads"
        );

        let revisions = db.list_pending_revisions(10).await.unwrap();
        assert_eq!(
            revisions.len(),
            1,
            "the staleness sweep must stage a revision card for a human-owned page \
             instead of silently escalating to source_conflict"
        );
        assert_eq!(revisions[0].target_source_id, "page_owned");
    }

    #[tokio::test]
    async fn re_distill_stale_pages_preserves_staleness_when_refresh_noops() {
        use crate::llm_provider::MockProvider;

        let (db, _db_dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();

        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source) \
                 VALUES (?1, ?1, ?1, 'retryable source material', 0, 'text', 'fact', 'test', 'claude-code', ?2, ?2, 1, 'confirmed', 'memory')",
                libsql::params!["mem_retry".to_string(), now_ts],
            )
            .await
            .unwrap();
        }

        db.insert_page(
            "page_retry",
            "Retry Topic",
            None,
            "original body",
            None,
            None,
            &["mem_retry"],
            &now,
        )
        .await
        .unwrap();
        db.set_page_stale("page_retry", "source_updated")
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(""));
        let prompts = PromptRegistry::default();

        let recompiled = re_distill_stale_pages(&db, Some(&llm), &prompts, None)
            .await
            .unwrap();
        assert_eq!(
            recompiled, 0,
            "empty refresh output must not count as a write"
        );

        let after = db.get_page("page_retry").await.unwrap().unwrap();
        assert_eq!(
            after.stale_reason.as_deref(),
            Some("source_updated"),
            "a no-op refresh should stay stale so the next sweep can retry"
        );
    }

    #[tokio::test]
    async fn re_distill_stale_pages_prompt_includes_same_space_existing_titles_hint() {
        let (db, _db_dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();

        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source) \
                 VALUES (?1, ?1, ?1, 'The target topic should retain the valid wikilink to Related Page when refreshed.', 0, 'text', 'fact', 'work', 'claude-code', ?2, ?2, 1, 'confirmed', 'memory')",
                libsql::params!["mem_target".to_string(), now_ts],
            )
            .await
            .unwrap();
        }

        db.insert_page(
            "page_related",
            "Related Page",
            None,
            "A work-space page that should be offered as a real wikilink target.",
            None,
            Some("work"),
            &[],
            &now,
        )
        .await
        .unwrap();
        db.insert_page(
            "page_private",
            "Private Page",
            None,
            "A personal-space page that must not be offered in a work refresh prompt.",
            None,
            Some("personal"),
            &[],
            &now,
        )
        .await
        .unwrap();
        db.insert_page(
            "page_target",
            "Target Page",
            None,
            "Original body links to [[Related Page]].",
            None,
            Some("work"),
            &["mem_target"],
            &now,
        )
        .await
        .unwrap();
        db.set_page_stale("page_target", "source_updated")
            .await
            .unwrap();

        let provider = Arc::new(RecordingDistillProvider::new());
        let llm: Arc<dyn LlmProvider> = provider.clone();
        let prompts = PromptRegistry::default();

        let recompiled = re_distill_stale_pages(&db, Some(&llm), &prompts, None)
            .await
            .unwrap();
        assert_eq!(recompiled, 1, "stale page should be re-distilled");

        let seen = provider.prompts();
        let user_prompt = seen
            .first()
            .expect("stale re-distill must send one compile prompt");
        assert!(
            user_prompt.starts_with("Existing pages you may reference with exact-match wikilinks:"),
            "PageWrite refresh prompt must start with existing-title hint, got:\n{user_prompt}"
        );
        assert!(
            user_prompt.contains("[[Related Page]]"),
            "same-space existing title must be offered as a valid wikilink target, got:\n{user_prompt}"
        );
        assert!(
            !user_prompt.contains("[[Private Page]]"),
            "cross-space existing title must not be offered in a work refresh prompt, got:\n{user_prompt}"
        );

        let page = db.get_page("page_target").await.unwrap().unwrap();
        assert!(
            page.content.contains("[[Related Page]]"),
            "refresh should retain already-valid wikilinks when the prompt exposes the real title, got:\n{}",
            page.content
        );
    }

    /// Atomicity (spec §5.1): re-distill must write the page body and its
    /// per-claim citation map in ONE transaction. The pre-fix site did a
    /// non-atomic two-step — `update_page(.., citations=None)` (content bump,
    /// citations reset to '[]') followed by a SEPARATE `set_page_citations`
    /// call — so a crash between the two commits leaves the page with updated
    /// content but un-updated ('[]') citations.
    ///
    /// The observable proof of atomicity: only
    /// `try_update_page_content_with_changelog` stamps `citations_summary` into
    /// the changelog entry, and it does so in the SAME transaction that writes
    /// the content + citations. The two-step leaves the content-bump changelog
    /// entry WITHOUT a `citations_summary` (set_page_citations touches only the
    /// citations column, no changelog). Asserting the summary is present proves
    /// citations rode the atomic content write, not a second commit.
    #[tokio::test]
    async fn re_distill_persists_citations_atomically_with_content() {
        use crate::llm_provider::MockProvider;

        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();

        // Seed the cited memory. The re-distilled body echoes its content so the
        // [1] marker verifies and yields a non-empty citation map.
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source) \
                 VALUES (?1, ?1, ?1, 'Tokio is an async runtime for Rust', 0, 'text', 'fact', 'test', 'claude-code', ?2, ?2, 1, 'confirmed', 'memory')",
                libsql::params!["mem_seed".to_string(), now_ts],
            )
            .await
            .unwrap();
        }

        db.insert_page(
            "page_stale",
            "Tokio",
            None,
            "original body",
            None,
            None,
            &["mem_seed"],
            &now,
        )
        .await
        .unwrap();
        db.set_page_stale("page_stale", "source_updated")
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> =
            Arc::new(MockProvider::new("Tokio is an async runtime for Rust [1]"));
        let prompts = PromptRegistry::default();

        let recompiled = re_distill_stale_pages(&db, Some(&llm), &prompts, None)
            .await
            .unwrap();
        assert_eq!(recompiled, 1, "stale page should be re-distilled");

        let page = db.get_page("page_stale").await.unwrap().unwrap();
        assert!(
            page.content.contains("Tokio is an async runtime"),
            "content should reflect the re-distilled body, got: {}",
            page.content
        );
        assert!(
            !page.citations.is_empty(),
            "re-distill must persist the per-claim citation map, not leave it empty"
        );

        let changelog_raw = db.get_page_changelog("page_stale").await.unwrap();
        let changelog: Vec<serde_json::Value> = serde_json::from_str(&changelog_raw).unwrap();
        let latest = changelog
            .last()
            .expect("re-distill content write must append a changelog entry");
        assert!(
            latest.get("citations_summary").is_some(),
            "content+citations must commit atomically: the re-distill changelog entry must carry citations_summary (the two-step leaves it absent); changelog={changelog_raw}"
        );
    }

    #[tokio::test]
    async fn background_redistill_slice_processes_exactly_one_page_and_reports_more() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();
        for (memory_id, page_id) in [
            ("redistill_slice_mem_a", "redistill_slice_page_a"),
            ("redistill_slice_mem_b", "redistill_slice_page_b"),
        ] {
            {
                let conn = db.conn.lock().await;
                conn.execute(
                    "INSERT INTO memories
                         (id, source_id, title, content, chunk_index, chunk_type,
                          memory_type, source_agent, created_at, last_modified,
                          confirmed, stability, source)
                     VALUES (?1, ?1, ?1, 'bounded ambient source material', 0,
                             'text', 'fact', 'test', ?2, ?2, 1, 'confirmed', 'memory')",
                    libsql::params![memory_id, now_ts],
                )
                .await
                .unwrap();
            }
            db.insert_page(
                page_id,
                page_id,
                None,
                "old body",
                None,
                None,
                &[memory_id],
                &now,
            )
            .await
            .unwrap();
            db.set_page_stale(page_id, "source_updated").await.unwrap();
        }
        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            "bounded ambient source material [1]",
            "bounded ambient source material [1]",
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let report = run_redistill_page_slice(&db, Some(&llm), &PromptRegistry::default(), None)
            .await
            .unwrap();

        assert!(report.selected);
        assert!(report.progressed);
        assert!(report.more);
        assert!(!report.retryable);
        assert_eq!(provider.call_count(), 1);
        assert_eq!(
            db.list_stale_pages("source_updated").await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn background_redistill_success_does_not_skip_row_shifted_by_stale_removal() {
        let (db, _dir) = test_db().await;
        let now_ts = chrono::Utc::now().timestamp();
        for (suffix, last_modified) in [
            ("a", "2026-07-16T03:00:00+00:00"),
            ("b", "2026-07-16T02:00:00+00:00"),
            ("c", "2026-07-16T01:00:00+00:00"),
        ] {
            let memory_id = format!("redistill_shift_memory_{suffix}");
            let page_id = format!("redistill_shift_page_{suffix}");
            {
                let conn = db.conn.lock().await;
                conn.execute(
                    "INSERT INTO memories
                         (id, source_id, title, content, chunk_index, chunk_type,
                          memory_type, source_agent, created_at, last_modified,
                          confirmed, stability, source)
                     VALUES (?1, ?1, ?1, 'ordered redistill source material', 0,
                             'text', 'fact', 'test', ?2, ?2, 1, 'confirmed', 'memory')",
                    libsql::params![memory_id.as_str(), now_ts],
                )
                .await
                .unwrap();
            }
            db.insert_page(
                &page_id,
                &page_id,
                None,
                "old body",
                None,
                None,
                &[memory_id.as_str()],
                last_modified,
            )
            .await
            .unwrap();
            db.set_page_stale(&page_id, "source_updated").await.unwrap();
        }
        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            "ordered redistill source material [1]",
            "ordered redistill source material [1]",
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        for _ in 0..2 {
            run_redistill_page_slice(&db, Some(&llm), &PromptRegistry::default(), None)
                .await
                .unwrap();
        }

        assert!(
            db.get_page_stale_reason("redistill_shift_page_a")
                .await
                .unwrap()
                .is_none(),
            "the first stale Page should complete"
        );
        assert!(
            db.get_page_stale_reason("redistill_shift_page_b")
                .await
                .unwrap()
                .is_none(),
            "removing the first row must not shift the second Page behind a persisted OFFSET"
        );
        assert_eq!(
            db.get_page_stale_reason("redistill_shift_page_c")
                .await
                .unwrap()
                .as_deref(),
            Some("source_updated"),
            "two slices should leave only the third Page stale"
        );
        assert_eq!(provider.call_count(), 2);
    }

    #[tokio::test]
    async fn background_redistill_over_source_cap_makes_zero_provider_calls_and_stays_stale() {
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();
        let source_ids = (0..65)
            .map(|index| format!("redistill_over_cap_mem_{index:02}"))
            .collect::<Vec<_>>();
        {
            let conn = db.conn.lock().await;
            for source_id in &source_ids {
                conn.execute(
                    "INSERT INTO memories
                         (id, source_id, title, content, chunk_index, chunk_type,
                          memory_type, source_agent, created_at, last_modified,
                          confirmed, stability, source)
                     VALUES (?1, ?1, ?1, 'bounded over cap source', 0,
                             'text', 'fact', 'test', ?2, ?2, 1, 'confirmed', 'memory')",
                    libsql::params![source_id.clone(), now_ts],
                )
                .await
                .unwrap();
            }
        }
        let source_refs = source_ids.iter().map(String::as_str).collect::<Vec<_>>();
        db.insert_page(
            "redistill_over_cap_page",
            "ReDistill over source cap",
            None,
            "old body",
            None,
            None,
            &source_refs,
            &now,
        )
        .await
        .unwrap();
        db.set_page_stale("redistill_over_cap_page", "source_updated")
            .await
            .unwrap();
        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            "bounded over cap source [1]",
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let report = run_redistill_page_slice(&db, Some(&llm), &PromptRegistry::default(), None)
            .await
            .unwrap();

        assert!(report.selected);
        assert!(
            report.progressed,
            "advancing the durable cursor is progress"
        );
        assert!(report.more);
        assert!(report.retryable);
        assert_eq!(report.items_processed, 0);
        assert_eq!(provider.call_count(), 0);
        assert_eq!(
            db.get_page_stale_reason("redistill_over_cap_page")
                .await
                .unwrap()
                .as_deref(),
            Some("source_updated")
        );

        // Legacy Pages can have only the JSON source list. The same cap must
        // apply when the normalized join table has no rows.
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "DELETE FROM page_sources WHERE page_id='redistill_over_cap_page'",
                (),
            )
            .await
            .unwrap();
        }
        persist_background_stale_cursor(&db, None).await.unwrap();
        let legacy_report =
            run_redistill_page_slice(&db, Some(&llm), &PromptRegistry::default(), None)
                .await
                .unwrap();
        assert!(legacy_report.retryable);
        assert_eq!(provider.call_count(), 0);
        assert_eq!(
            db.get_page_stale_reason("redistill_over_cap_page")
                .await
                .unwrap()
                .as_deref(),
            Some("source_updated")
        );
    }

    #[tokio::test]
    async fn background_redistill_phase_uses_one_page_slice_not_legacy_batch() {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        let now_ts = chrono::Utc::now().timestamp();
        for (memory_id, page_id) in [
            ("redistill_phase_mem_a", "redistill_phase_page_a"),
            ("redistill_phase_mem_b", "redistill_phase_page_b"),
        ] {
            {
                let conn = db.conn.lock().await;
                conn.execute(
                    "INSERT INTO memories
                         (id, source_id, title, content, chunk_index, chunk_type,
                          memory_type, source_agent, created_at, last_modified,
                          confirmed, stability, source)
                     VALUES (?1, ?1, ?1, 'bounded phase source material', 0,
                             'text', 'fact', 'test', ?2, ?2, 1, 'confirmed', 'memory')",
                    libsql::params![memory_id, now_ts],
                )
                .await
                .unwrap();
            }
            db.insert_page(
                page_id,
                page_id,
                None,
                "old body",
                None,
                None,
                &[memory_id],
                &now,
            )
            .await
            .unwrap();
            db.set_page_stale(page_id, "source_updated").await.unwrap();
        }
        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            "bounded phase source material [1]",
            "bounded phase source material [1]",
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();

        let report =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                crate::config::save_config(&crate::config::Config {
                    synthesis_source: Some("on_device".to_string()),
                    ..crate::config::Config::default()
                })
                .unwrap();
                run_periodic_steep_phase_with_api(
                    &db,
                    Some(&llm),
                    None,
                    None,
                    None,
                    &PromptRegistry::default(),
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Idle,
                    Phase::ReDistill,
                )
                .await
            })
            .await
            .unwrap();

        assert_eq!(report.result.phases.len(), 1);
        assert_eq!(report.result.phases[0].name, "re-distill");
        assert!(report.selected);
        assert!(report.progressed);
        assert!(report.more);
        assert!(!report.retryable);
        assert!(!report.paused);
        assert_eq!(provider.call_count(), 1);
        assert_eq!(
            db.list_stale_pages("source_updated").await.unwrap().len(),
            1
        );
    }

    #[tokio::test]
    async fn background_redistill_unchanged_scan_progress_is_silent() {
        let _serial = COMPILE_ROUTING_ENV_LOCK.lock().await;
        let (db, _dir) = test_db().await;
        let now_ts = chrono::Utc::now().timestamp();
        let compiled = chrono::Utc::now().to_rfc3339();
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO memories
                     (id, source_id, title, content, chunk_index, chunk_type,
                      memory_type, source_agent, created_at, last_modified,
                      confirmed, stability, source)
                 VALUES ('unchanged_scan_mem', 'unchanged_scan_mem', 'unchanged',
                         'already compiled', 0, 'text', 'fact', 'test', ?1, ?1,
                         1, 'confirmed', 'memory')",
                libsql::params![now_ts],
            )
            .await
            .unwrap();
        }
        db.insert_page(
            "unchanged_scan_page",
            "Unchanged scan page",
            None,
            "already compiled",
            None,
            None,
            &["unchanged_scan_mem"],
            &compiled,
        )
        .await
        .unwrap();
        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![]));
        let llm: Arc<dyn LlmProvider> = provider.clone();
        let data_dir = tempfile::tempdir().unwrap();
        let data_dir_var = data_dir.path().to_string_lossy().to_string();

        let report =
            temp_env::async_with_vars([("WENLAN_DATA_DIR", Some(data_dir_var.as_str()))], async {
                crate::config::save_config(&crate::config::Config {
                    synthesis_source: Some("on_device".to_string()),
                    ..crate::config::Config::default()
                })
                .unwrap();
                run_periodic_steep_phase_with_api(
                    &db,
                    Some(&llm),
                    None,
                    None,
                    None,
                    &PromptRegistry::default(),
                    &crate::tuning::RefineryConfig::default(),
                    &crate::tuning::ConfidenceConfig::default(),
                    &crate::tuning::DistillationConfig::default(),
                    None,
                    TriggerKind::Idle,
                    Phase::ReDistill,
                )
                .await
            })
            .await
            .unwrap();

        assert!(report.selected);
        assert!(
            report.progressed,
            "advancing the durable scan cursor is progress"
        );
        assert_eq!(report.result.phases[0].items_processed, 0);
        assert_eq!(report.result.phases[0].nudge, Nudge::Silent);
        assert_eq!(provider.call_count(), 0);
    }

    #[tokio::test]
    async fn background_redistill_cursor_reaches_page_beyond_ten_retryable_rows() {
        let (db, _dir) = test_db().await;
        let base = chrono::Utc::now();
        for index in 0..11 {
            let is_good = index == 10;
            let memory_id = if is_good {
                "good-eleventh-memory".to_string()
            } else {
                format!("poison-memory-{index:02}")
            };
            let page_id = if is_good {
                "good-eleventh-page".to_string()
            } else {
                format!("poison-page-{index:02}")
            };
            let content = if is_good {
                "good-eleventh bounded source".to_string()
            } else {
                format!("poison bounded source {index:02}")
            };
            let timestamp = (base - chrono::Duration::seconds(index as i64)).timestamp();
            let compiled = (base - chrono::Duration::seconds(index as i64)).to_rfc3339();
            {
                let conn = db.conn.lock().await;
                conn.execute(
                    "INSERT INTO memories
                         (id, source_id, title, content, chunk_index, chunk_type,
                          memory_type, source_agent, created_at, last_modified,
                          confirmed, stability, source)
                     VALUES (?1, ?1, ?1, ?2, 0, 'text', 'fact', 'test', ?3, ?3,
                             1, 'confirmed', 'memory')",
                    libsql::params![memory_id.as_str(), content.as_str(), timestamp],
                )
                .await
                .unwrap();
            }
            db.insert_page(
                &page_id,
                &page_id,
                None,
                "old body",
                None,
                None,
                &[memory_id.as_str()],
                &compiled,
            )
            .await
            .unwrap();
            db.set_page_stale(&page_id, "source_updated").await.unwrap();
        }
        let provider = Arc::new(SelectiveRedistillProvider::new());
        let llm: Arc<dyn LlmProvider> = provider.clone();

        for _ in 0..11 {
            run_redistill_page_slice(&db, Some(&llm), &PromptRegistry::default(), None)
                .await
                .unwrap();
        }

        assert!(
            provider.saw_good_page(),
            "ten retryable rows must not permanently hide the eleventh Page"
        );
        assert!(
            db.get_page("good-eleventh-page")
                .await
                .unwrap()
                .unwrap()
                .stale_reason
                .is_none(),
            "the reachable healthy Page should complete even while ten poison Pages remain"
        );
    }
}
