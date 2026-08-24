// SPDX-License-Identifier: Apache-2.0
//! Operator-facing entry points into the ambient scheduler: a synchronous
//! force-sweep trigger and a status snapshot. Both are read/act surfaces for
//! `crate::ambient_routes`; neither owns any scheduling state of its own —
//! the passive scheduler tick in `scheduler.rs` remains the only place that
//! advances `AmbientSchedule`'s round-robin cursor and thermal cooldown.

use super::*;

/// Outcome of one forced attempt at a single ambient job.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AmbientJobSweepResult {
    pub job: &'static str,
    /// False when the job's own availability gate (provider health, or an
    /// opt-in flag like `citation_backfill_enabled`) parked it — the sweep
    /// still bypasses the idle/resource gate and each job's periodic
    /// due-check, but never bypasses explicit user consent.
    pub attempted: bool,
    pub selected: bool,
    pub llm_calls: usize,
    pub panicked: bool,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AmbientSweepReport {
    pub phases: Vec<AmbientJobSweepResult>,
}

/// Force one bounded, synchronous pass over every ambient job type, bypassing
/// the idle/resource gate and each job's periodic due-check (`ENRICHMENT_
/// SWEEP_INTERVAL` and friends). Each job's own per-call slice bound is
/// unchanged — Document still claims one document, Citation still backfills
/// one page, etc. — only the *scheduling* gate is bypassed, not the batch
/// size.
///
/// This does not touch `AmbientSchedule.next_allowed_at` (the passive
/// scheduler's thermal cooldown clock): that state lives inside the spawned
/// scheduler task's local loop variables, not in `SharedState`, so a forced
/// sweep and the background scheduler are not mutually cooling down — each
/// side still only throttles its own future runs. That is an accepted,
/// documented limitation, not an oversight. They *are* mutually exclusive in
/// time, though: `ServerState::ambient_run_lock` (held by the caller in
/// `crate::ambient_routes::handle_ambient_sweep`, not by this function)
/// keeps a forced sweep and a passive scheduler tick from ever executing
/// ambient jobs concurrently, even though neither one's cooldown clock knows
/// about the other's completed work.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn force_ambient_sweep(
    db: &Arc<wenlan_core::db::MemoryDB>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    api_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    everyday_pin: Option<wenlan_core::refinery::EverydaySource>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    refinery: &wenlan_core::tuning::RefineryConfig,
    distillation: &wenlan_core::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> AmbientSweepReport {
    let provider_available =
        resolve_ambient_provider(everyday_pin, api_llm, external_llm, llm).is_some();
    let availability = AmbientAvailability::for_provider(provider_available);

    let mut phases = Vec::with_capacity(AmbientJob::ALL.len());
    for job in AmbientJob::ALL {
        if !availability.supports(job) {
            phases.push(AmbientJobSweepResult {
                job: job.as_key(),
                attempted: false,
                selected: false,
                llm_calls: 0,
                panicked: false,
                elapsed_ms: 0,
            });
            continue;
        }
        let report = run_ambient_job_safe(
            job,
            db,
            llm,
            api_llm,
            external_llm,
            everyday_pin,
            prompts,
            refinery,
            distillation,
            knowledge_path,
        )
        .await;
        phases.push(AmbientJobSweepResult {
            job: job.as_key(),
            attempted: true,
            selected: report.selected,
            llm_calls: report.llm_calls,
            panicked: report.panicked,
            elapsed_ms: report.elapsed.as_millis(),
        });
    }
    AmbientSweepReport { phases }
}

/// Latest resource/host-activity admission observed by the background
/// scheduler tick. The HTTP status handler reads this instead of sampling
/// CPU/host-activity itself: a freshly constructed `SystemResourceProbe`
/// always reports "warming" until its own first real interval passes, so an
/// independently sampled probe could never agree with the real gate the
/// scheduler actually enforces — it would just be a second, disagreeing
/// opinion.
#[derive(Debug, Clone)]
pub struct AmbientGateSnapshot {
    pub admitted: bool,
    pub blocked_reason: Option<String>,
    pub sampled_at_epoch: i64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AmbientStatusReport {
    pub unchunked_documents: u64,
    pub pages_missing_citations: u64,
    /// `None` only before the scheduler's first post-startup tick has run.
    pub idle_gate_admitted: Option<bool>,
    pub idle_gate_blocked_reason: Option<String>,
    pub idle_gate_sampled_at_epoch: Option<i64>,
    /// Unix-epoch seconds of the last attempt at each job (selected or not),
    /// keyed by `AmbientJob::as_key()`. `None` means never attempted.
    pub phase_last_run_epoch: std::collections::BTreeMap<String, Option<i64>>,
}

/// A missing-citations count above this cap is reported as the cap, not the
/// true total — enough to answer "is this queue draining", not an exact
/// backlog size, and cheap regardless of how large the real backlog is.
const CITATION_STATUS_COUNT_LIMIT: usize = 5_000;

pub(crate) async fn ambient_status(
    db: &wenlan_core::db::MemoryDB,
    gate: Option<AmbientGateSnapshot>,
) -> Result<AmbientStatusReport, wenlan_core::WenlanError> {
    let queue = db.document_enrichment_queue_status().await?;
    let pages_missing_citations = db
        .get_pages_missing_citations(CITATION_STATUS_COUNT_LIMIT)
        .await?
        .len() as u64;

    let mut phase_last_run_epoch = std::collections::BTreeMap::new();
    for job in AmbientJob::ALL {
        let last_run = db
            .get_app_metadata(&ambient_last_run_key(job))
            .await
            .ok()
            .flatten()
            .and_then(|value| value.parse::<i64>().ok());
        phase_last_run_epoch.insert(job.as_key().to_string(), last_run);
    }

    Ok(AmbientStatusReport {
        unchunked_documents: queue.pending,
        pages_missing_citations,
        idle_gate_admitted: gate.as_ref().map(|snapshot| snapshot.admitted),
        idle_gate_blocked_reason: gate
            .as_ref()
            .and_then(|snapshot| snapshot.blocked_reason.clone()),
        idle_gate_sampled_at_epoch: gate.map(|snapshot| snapshot.sampled_at_epoch),
        phase_last_run_epoch,
    })
}
