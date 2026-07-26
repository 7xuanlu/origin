// SPDX-License-Identifier: Apache-2.0
//! Event-driven steep scheduler.
//!
//! Owns all steep scheduling: BurstEnd (per-agent adaptive gap), Idle (automatic
//! recap batching after 10 minutes without Wenlan writes), Daily
//! (first-wake-after-24h), and Backstop (6-hour safety net).
//! Replaces the former steep loop in main.rs.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::host_activity::{sample_host_activity, HostActivitySnapshot};
use crate::state::SharedState;

/// 30-minute ceiling for adaptive gap — matches ACTIVITY_GAP_SECS in wenlan-core.
const BURST_GAP_CEILING: Duration = Duration::from_secs(1800);
/// 5-minute floor — prevents premature firing on fast writers.
const BURST_GAP_FLOOR: Duration = Duration::from_secs(300);
/// Minimum writes to qualify as a recap-worthy burst.
const MIN_BURST_WRITES: usize = 3;
/// Wenlan-write batching threshold for the automatic Idle recap trigger.
/// This is not an OS foreground-idle signal or an ambient-enrichment gate.
const AUTOMATIC_BATCH_IDLE_THRESHOLD: Duration = Duration::from_secs(600);
/// Backstop interval — safety net fires all phases.
const BACKSTOP_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);
/// Daily interval — first quiet turn after 24 hours.
const DAILY_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
/// Poll interval — how often the scheduler checks trigger conditions.
const POLL_INTERVAL: Duration = Duration::from_secs(30);
/// A full two-sample scheduler window without keyboard, mouse, or tablet input.
/// Unlike Wenlan write recency, this is a real OS foreground-activity signal.
const FOREGROUND_INPUT_IDLE_THRESHOLD: Duration = Duration::from_secs(60);
/// Initial delay — lets on-device model warm up before first backstop.
const INITIAL_DELAY: Duration = Duration::from_secs(60);
const DERIVED_RECEIPT_SWEEP_INTERVAL: Duration = Duration::from_secs(30 * 60);
const ENRICHMENT_SWEEP_INTERVAL: Duration = Duration::from_secs(30 * 60);
const RECONCILE_SWEEP_INTERVAL: Duration = Duration::from_secs(30 * 60);
const CITATION_SWEEP_INTERVAL: Duration = Duration::from_secs(30 * 60);
const EDGES_RECONCILE_SWEEP_INTERVAL: Duration = Duration::from_secs(30 * 60);
const ENTITY_PAGE_RECONCILE_SWEEP_INTERVAL: Duration = Duration::from_secs(30 * 60);
const EDGE_GROUNDING_SWEEP_INTERVAL: Duration = Duration::from_secs(30 * 60);
/// Target-Mac evidence keeps short ambient turns below a 5% duty cycle while
/// avoiding the fivefold convergence penalty of the provisional ten-minute
/// hotfix. Automatic recap batching still uses its separate ten-minute window.
const AMBIENT_MIN_RECOVERY: Duration = Duration::from_secs(120);
const GIB: u64 = 1024 * 1024 * 1024;
/// Target-Mac calibration observed up to 1.59 GiB of additional process RSS
/// during one on-device ambient inference. Round up so background work never
/// consumes the ordinary foreground memory reserve.
const ON_DEVICE_INFERENCE_HEADROOM_BYTES: u64 = 2 * GIB;
const AUTOMATIC_STEEP_PHASE_CURSOR_PREFIX: &str = "automatic_steep_phase_cursor_v1";
const AUTOMATIC_MAINTENANCE_STAGE_CURSOR_KEY: &str = "automatic_maintenance_stage_cursor_v1";

#[derive(Debug, Clone, Copy)]
struct ResourceSnapshot {
    cpu_usage_percent: f32,
    available_memory_bytes: u64,
    total_memory_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceBlockReason {
    Warming,
    Unavailable,
    CpuBusy,
    MemoryPressure,
    ForegroundActive,
    HostActivityUnavailable,
    ThermalPressure,
}

#[derive(Debug, Clone, Copy)]
struct ResourcePolicy {
    max_cpu_usage_percent: f32,
    min_available_memory_bytes: u64,
    min_available_memory_percent: u64,
    additional_memory_headroom_bytes: u64,
    idle_samples_required: u8,
}

impl ResourcePolicy {
    const fn conservative() -> Self {
        Self {
            max_cpu_usage_percent: 20.0,
            min_available_memory_bytes: 2 * GIB,
            min_available_memory_percent: 15,
            additional_memory_headroom_bytes: 0,
            idle_samples_required: 2,
        }
    }

    const fn with_additional_memory_headroom(mut self, bytes: u64) -> Self {
        self.additional_memory_headroom_bytes = bytes;
        self
    }

    fn block_reason(self, snapshot: ResourceSnapshot) -> Option<ResourceBlockReason> {
        if !snapshot.cpu_usage_percent.is_finite() || snapshot.total_memory_bytes == 0 {
            return Some(ResourceBlockReason::Unavailable);
        }
        let ratio_floor = snapshot
            .total_memory_bytes
            .saturating_mul(self.min_available_memory_percent)
            / 100;
        let memory_floor = self
            .min_available_memory_bytes
            .max(ratio_floor)
            .saturating_add(self.additional_memory_headroom_bytes);
        if snapshot.available_memory_bytes < memory_floor {
            return Some(ResourceBlockReason::MemoryPressure);
        }
        if snapshot.cpu_usage_percent > self.max_cpu_usage_percent {
            return Some(ResourceBlockReason::CpuBusy);
        }
        None
    }
}

fn host_activity_block_reason(snapshot: HostActivitySnapshot) -> Option<ResourceBlockReason> {
    match snapshot {
        #[cfg(any(not(target_os = "macos"), test))]
        HostActivitySnapshot::Unsupported => None,
        HostActivitySnapshot::Unavailable => Some(ResourceBlockReason::HostActivityUnavailable),
        HostActivitySnapshot::Observed { thermal_state, .. } if thermal_state != 0 => {
            Some(ResourceBlockReason::ThermalPressure)
        }
        HostActivitySnapshot::Observed { idle_for, .. }
            if idle_for < FOREGROUND_INPUT_IDLE_THRESHOLD =>
        {
            Some(ResourceBlockReason::ForegroundActive)
        }
        HostActivitySnapshot::Observed { .. } => None,
    }
}

#[derive(Debug, Default)]
struct ResourceAdmission {
    consecutive_idle_samples: u8,
}

impl ResourceAdmission {
    fn observe(&mut self, snapshot: ResourceSnapshot, policy: ResourcePolicy) -> bool {
        if policy.block_reason(snapshot).is_some() {
            self.consecutive_idle_samples = 0;
            return false;
        }
        self.consecutive_idle_samples = self.consecutive_idle_samples.saturating_add(1);
        self.consecutive_idle_samples >= policy.idle_samples_required
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rb01ProfileBlockReason {
    ResourceUnavailable,
    ThermalUnavailable,
    ThermalPressure,
    CpuBusy,
    MemoryPressure,
    Warming,
}

#[cfg(test)]
const RB01_PROFILE_ADMISSION_MAX_SAMPLES: usize = 4;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rb01ProfileSampleAction {
    Retry,
    Admit,
    Fail(Rb01ProfileBlockReason),
}

#[cfg(test)]
#[derive(Debug, Default)]
struct Rb01ProfileAdmission {
    resources: ResourceAdmission,
}

#[cfg(test)]
impl Rb01ProfileAdmission {
    fn observe(
        &mut self,
        snapshot: Option<ResourceSnapshot>,
        thermal_state: Option<u8>,
        policy: ResourcePolicy,
    ) -> Result<(), Rb01ProfileBlockReason> {
        let Some(snapshot) = snapshot else {
            self.resources = ResourceAdmission::default();
            return Err(Rb01ProfileBlockReason::ResourceUnavailable);
        };
        let Some(thermal_state) = thermal_state else {
            self.resources = ResourceAdmission::default();
            return Err(Rb01ProfileBlockReason::ThermalUnavailable);
        };
        if thermal_state != 0 {
            self.resources = ResourceAdmission::default();
            return Err(Rb01ProfileBlockReason::ThermalPressure);
        }

        if let Some(reason) = policy.block_reason(snapshot) {
            self.resources.observe(snapshot, policy);
            return Err(match reason {
                ResourceBlockReason::CpuBusy => Rb01ProfileBlockReason::CpuBusy,
                ResourceBlockReason::MemoryPressure => Rb01ProfileBlockReason::MemoryPressure,
                ResourceBlockReason::Warming | ResourceBlockReason::Unavailable => {
                    Rb01ProfileBlockReason::ResourceUnavailable
                }
                // Host reasons cannot originate from ResourcePolicy; these
                // arms keep this test-only adapter exhaustive as the shared
                // production reason enum evolves.
                ResourceBlockReason::ForegroundActive => Rb01ProfileBlockReason::Warming,
                ResourceBlockReason::HostActivityUnavailable => {
                    Rb01ProfileBlockReason::ThermalUnavailable
                }
                ResourceBlockReason::ThermalPressure => Rb01ProfileBlockReason::ThermalPressure,
            });
        }

        if self.resources.observe(snapshot, policy) {
            Ok(())
        } else {
            Err(Rb01ProfileBlockReason::Warming)
        }
    }
}

#[cfg(test)]
fn rb01_profile_sample_action(
    admission: &mut Rb01ProfileAdmission,
    snapshot: Option<ResourceSnapshot>,
    thermal_state: Option<u8>,
    policy: ResourcePolicy,
    sample_index: usize,
    max_samples: usize,
) -> Rb01ProfileSampleAction {
    match admission.observe(snapshot, thermal_state, policy) {
        Ok(()) => Rb01ProfileSampleAction::Admit,
        Err(Rb01ProfileBlockReason::CpuBusy | Rb01ProfileBlockReason::Warming)
            if sample_index < max_samples =>
        {
            Rb01ProfileSampleAction::Retry
        }
        Err(reason) => Rb01ProfileSampleAction::Fail(reason),
    }
}

#[cfg(test)]
fn rb01_profile_requested(value: Option<&str>) -> bool {
    value == Some("1")
}

#[derive(Debug, Clone, Copy)]
struct ThermalPolicy {
    minimum_cooldown: Duration,
    recovery_multiplier: u32,
}

impl ThermalPolicy {
    const fn conservative() -> Self {
        Self {
            minimum_cooldown: AMBIENT_MIN_RECOVERY,
            // Work / (work + recovery) <= 5% when this multiplier dominates.
            recovery_multiplier: 19,
        }
    }

    fn cooldown_after(self, elapsed: Duration) -> Duration {
        self.minimum_cooldown
            .max(elapsed.saturating_mul(self.recovery_multiplier))
    }
}

#[derive(Debug, Clone, Copy)]
struct ResourceStatus {
    admitted: bool,
    snapshot: Option<ResourceSnapshot>,
    block_reason: Option<ResourceBlockReason>,
}

fn apply_host_activity(
    status: ResourceStatus,
    host_activity: HostActivitySnapshot,
) -> ResourceStatus {
    let Some(block_reason) = host_activity_block_reason(host_activity) else {
        return status;
    };
    ResourceStatus {
        admitted: false,
        block_reason: Some(block_reason),
        ..status
    }
}

struct SystemResourceProbe {
    system: sysinfo::System,
    last_refresh: Instant,
    admission: ResourceAdmission,
}

impl SystemResourceProbe {
    fn new(now: Instant) -> Self {
        let refreshes = sysinfo::RefreshKind::nothing()
            .with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage())
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram());
        Self {
            system: sysinfo::System::new_with_specifics(refreshes),
            last_refresh: now,
            admission: ResourceAdmission::default(),
        }
    }

    fn sample(&mut self, now: Instant, policy: ResourcePolicy) -> ResourceStatus {
        if now.saturating_duration_since(self.last_refresh) < sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
            return ResourceStatus {
                admitted: false,
                snapshot: None,
                block_reason: Some(ResourceBlockReason::Warming),
            };
        }

        self.system.refresh_cpu_usage();
        self.system.refresh_memory();
        self.last_refresh = now;

        let snapshot = ResourceSnapshot {
            cpu_usage_percent: self.system.global_cpu_usage(),
            available_memory_bytes: self.system.available_memory(),
            total_memory_bytes: self.system.total_memory(),
        };
        let policy_block = policy.block_reason(snapshot);
        let admitted = self.admission.observe(snapshot, policy);
        ResourceStatus {
            admitted,
            snapshot: Some(snapshot),
            block_reason: policy_block
                .or_else(|| (!admitted).then_some(ResourceBlockReason::Warming)),
        }
    }
}

fn observe_deferred_resource_reason(
    previous: &mut Option<ResourceBlockReason>,
    admitted: bool,
    block_reason: Option<ResourceBlockReason>,
) -> Option<ResourceBlockReason> {
    let current = (!admitted).then_some(block_reason).flatten();
    let changed = current.is_some() && current != *previous;
    *previous = current;
    if changed {
        current
    } else {
        None
    }
}

/// Wait until a selected on-device model can be loaded without consuming the
/// scheduler's foreground reserve. The model working set is additive to the
/// normal 2 GiB / 15% floor, and the same two consecutive 30-second CPU samples
/// are required before `spawn_blocking` may touch the model.
pub async fn wait_for_startup_model_admission(
    model_working_set_bytes: u64,
    shutdown: &mut tokio::sync::watch::Receiver<bool>,
) -> bool {
    let policy =
        ResourcePolicy::conservative().with_additional_memory_headroom(model_working_set_bytes);
    let mut probe = SystemResourceProbe::new(Instant::now());
    loop {
        if crate::lifecycle::sleep_or_shutdown(shutdown, POLL_INTERVAL).await {
            return false;
        }
        let status =
            apply_host_activity(probe.sample(Instant::now(), policy), sample_host_activity());
        if status.admitted {
            tracing::info!(
                "[on-device] startup load admitted after two quiet samples; reserved_working_set_mb={}",
                model_working_set_bytes / (1024 * 1024)
            );
            return true;
        }
        tracing::debug!(
            "[on-device] startup load deferred reason={:?} cpu_percent={:?} available_memory_mb={:?} reserved_working_set_mb={}",
            status.block_reason,
            status.snapshot.map(|snapshot| snapshot.cpu_usage_percent),
            status
                .snapshot
                .map(|snapshot| snapshot.available_memory_bytes / (1024 * 1024)),
            model_working_set_bytes / (1024 * 1024),
        );
    }
}

fn startup_model_reservation_blocks_route(
    startup_model_load_reserved: bool,
    route_uses_on_device: bool,
) -> bool {
    startup_model_load_reserved && route_uses_on_device
}

fn background_heavy_resource_admitted(
    resource_status: ResourceStatus,
    startup_model_load_reserved: bool,
    route_uses_on_device: bool,
) -> bool {
    let route_memory_admitted = !route_uses_on_device
        || resource_status.snapshot.is_some_and(|snapshot| {
            ResourcePolicy::conservative()
                .with_additional_memory_headroom(ON_DEVICE_INFERENCE_HEADROOM_BYTES)
                .block_reason(snapshot)
                .is_none()
        });
    resource_status.admitted
        && route_memory_admitted
        && !startup_model_reservation_blocks_route(
            startup_model_load_reserved,
            route_uses_on_device,
        )
}

fn periodic_directory_sync_allowed(resource_admitted: bool) -> bool {
    resource_admitted
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AmbientJob {
    Document,
    Classification,
    StructuredExtract,
    Entity,
    Title,
    PageGrowth,
    Reconcile,
    Citation,
    EdgesReconcile,
    EntityPageReconcile,
    EdgeGroundingPromote,
}

impl AmbientJob {
    const ALL: [Self; 11] = [
        Self::Document,
        Self::Classification,
        Self::StructuredExtract,
        Self::Entity,
        Self::Title,
        Self::PageGrowth,
        Self::Reconcile,
        Self::Citation,
        Self::EdgesReconcile,
        Self::EntityPageReconcile,
        Self::EdgeGroundingPromote,
    ];
}

#[derive(Debug, Clone, Copy)]
struct AmbientAvailability {
    document: bool,
    classification: bool,
    structured_extract: bool,
    entity: bool,
    title: bool,
    page_growth: bool,
    reconcile: bool,
    citation: bool,
    edges_reconcile: bool,
    entity_page_reconcile: bool,
    edge_grounding_promote: bool,
}

impl AmbientAvailability {
    /// Automatic lanes are a consent boundary as well as a capability check.
    /// Deterministic document preparation remains available without a model so
    /// source data becomes searchable; all inference lanes stay parked until
    /// the pinned provider is both authorized and healthy.
    fn for_provider(provider_available: bool) -> Self {
        Self {
            document: true,
            classification: provider_available,
            structured_extract: provider_available,
            entity: provider_available && wenlan_core::db::entity_sweep_enabled(),
            title: provider_available,
            page_growth: provider_available,
            reconcile: provider_available && wenlan_core::db::doc_reconcile_enabled(),
            citation: provider_available && wenlan_core::db::citation_backfill_enabled(),
            // No model or model consent is involved, but the scan still goes
            // through the shared foreground/resource/cooldown controller.
            edges_reconcile: wenlan_core::db::edges_reconcile_enabled(),
            entity_page_reconcile: wenlan_core::db::entity_page_reconcile_enabled(),
            // Provider-gated: the mandatory entailment check spends an LLM call,
            // so the lane parks unless the pinned provider is authorized+healthy
            // AND the opt-in flag is set (mirroring reconcile / citation).
            edge_grounding_promote: provider_available
                && wenlan_core::db::edge_grounding_promote_enabled(),
        }
    }

    const fn supports(self, job: AmbientJob) -> bool {
        match job {
            AmbientJob::Document => self.document,
            AmbientJob::Classification => self.classification,
            AmbientJob::StructuredExtract => self.structured_extract,
            AmbientJob::Entity => self.entity,
            AmbientJob::Title => self.title,
            AmbientJob::PageGrowth => self.page_growth,
            AmbientJob::Reconcile => self.reconcile,
            AmbientJob::Citation => self.citation,
            AmbientJob::EdgesReconcile => self.edges_reconcile,
            AmbientJob::EntityPageReconcile => self.entity_page_reconcile,
            AmbientJob::EdgeGroundingPromote => self.edge_grounding_promote,
        }
    }
}

struct AmbientSchedule {
    cursor: usize,
    next_allowed_at: Instant,
    last_classification: Option<Instant>,
    last_structured_extract: Option<Instant>,
    last_entity: Option<Instant>,
    last_title: Option<Instant>,
    last_page_growth: Option<Instant>,
    last_reconcile: Option<Instant>,
    last_citation: Option<Instant>,
    last_edges_reconcile: Option<Instant>,
    last_entity_page_reconcile: Option<Instant>,
    last_edge_grounding_promote: Option<Instant>,
}

impl AmbientSchedule {
    fn new(now: Instant) -> Self {
        Self {
            cursor: 0,
            next_allowed_at: now,
            last_classification: None,
            last_structured_extract: None,
            last_entity: None,
            last_title: None,
            last_page_growth: None,
            last_reconcile: None,
            last_citation: None,
            last_edges_reconcile: None,
            last_entity_page_reconcile: None,
            last_edge_grounding_promote: None,
        }
    }

    fn select_due(
        &mut self,
        now: Instant,
        availability: AmbientAvailability,
    ) -> Option<AmbientJob> {
        for _ in 0..AmbientJob::ALL.len() {
            let job = AmbientJob::ALL[self.cursor];
            self.cursor = (self.cursor + 1) % AmbientJob::ALL.len();
            if !availability.supports(job) {
                continue;
            }
            let due = match job {
                AmbientJob::Document => true,
                AmbientJob::Classification => self
                    .last_classification
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::StructuredExtract => self
                    .last_structured_extract
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::Entity => self
                    .last_entity
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::Title => self
                    .last_title
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::PageGrowth => self
                    .last_page_growth
                    .is_none_or(|last| now.duration_since(last) >= ENRICHMENT_SWEEP_INTERVAL),
                AmbientJob::Reconcile => self
                    .last_reconcile
                    .is_none_or(|last| now.duration_since(last) >= RECONCILE_SWEEP_INTERVAL),
                AmbientJob::Citation => self
                    .last_citation
                    .is_none_or(|last| now.duration_since(last) >= CITATION_SWEEP_INTERVAL),
                AmbientJob::EdgesReconcile => self
                    .last_edges_reconcile
                    .is_none_or(|last| now.duration_since(last) >= EDGES_RECONCILE_SWEEP_INTERVAL),
                AmbientJob::EntityPageReconcile => {
                    self.last_entity_page_reconcile.is_none_or(|last| {
                        now.duration_since(last) >= ENTITY_PAGE_RECONCILE_SWEEP_INTERVAL
                    })
                }
                AmbientJob::EdgeGroundingPromote => self
                    .last_edge_grounding_promote
                    .is_none_or(|last| now.duration_since(last) >= EDGE_GROUNDING_SWEEP_INTERVAL),
            };
            if !due {
                continue;
            }
            return Some(job);
        }
        None
    }

    /// Back off an empty periodic lane, but leave known backlog due. The global
    /// thermal cooldown still limits actual work; this only prevents a second
    /// 30-minute delay from turning catch-up into a multi-week drain.
    fn note_job_result(&mut self, job: AmbientJob, now: Instant, selected: bool) {
        if selected && job != AmbientJob::EdgesReconcile && job != AmbientJob::EntityPageReconcile {
            return;
        }
        match job {
            AmbientJob::Document => {}
            AmbientJob::Classification => self.last_classification = Some(now),
            AmbientJob::StructuredExtract => self.last_structured_extract = Some(now),
            AmbientJob::Entity => self.last_entity = Some(now),
            AmbientJob::Title => self.last_title = Some(now),
            AmbientJob::PageGrowth => self.last_page_growth = Some(now),
            AmbientJob::Reconcile => self.last_reconcile = Some(now),
            AmbientJob::Citation => self.last_citation = Some(now),
            // One edge reconciliation is a complete full pass, not one item
            // from a backlog. Always enforce its 30-minute interval.
            AmbientJob::EdgesReconcile => self.last_edges_reconcile = Some(now),
            // Same full-pass reasoning as EdgesReconcile.
            AmbientJob::EntityPageReconcile => self.last_entity_page_reconcile = Some(now),
            // Backlog drainer, not a full pass: this arm only runs when the
            // slice made no progress (empty backlog), so stamp the interval to
            // back off. A progressing slice returned early above, staying due.
            AmbientJob::EdgeGroundingPromote => self.last_edge_grounding_promote = Some(now),
        }
    }

    fn note_thermal_work_completion(
        &mut self,
        now: Instant,
        elapsed: Duration,
        policy: ThermalPolicy,
    ) {
        self.next_allowed_at = now + policy.cooldown_after(elapsed);
    }
}

fn ambient_turn_allowed(
    system_resources_idle: bool,
    now: Instant,
    next_allowed_at: Instant,
) -> bool {
    system_resources_idle && now >= next_allowed_at
}

fn automatic_heavy_turn_allowed(
    system_resources_idle: bool,
    ambient_turn_owed: bool,
    now: Instant,
    next_allowed_at: Instant,
) -> bool {
    !ambient_turn_owed && ambient_turn_allowed(system_resources_idle, now, next_allowed_at)
}

fn refresh_last_write_activity(write_signal: &WriteSignal, last_write_activity: &mut Instant) {
    if let Some(latest) = write_signal
        .snapshot()
        .values()
        .flat_map(|timestamps| timestamps.iter().copied())
        .max()
    {
        *last_write_activity = (*last_write_activity).max(latest);
    }
}

fn should_backoff_ambient_lane(selected: bool, llm_calls: usize) -> bool {
    !selected && llm_calls == 0
}

fn ambient_work_consumes_thermal_turn(
    job: AmbientJob,
    selected: bool,
    llm_calls: usize,
    page_growth_terminal_no_match_committed: bool,
) -> bool {
    llm_calls > 0
        || matches!(
            job,
            AmbientJob::EdgesReconcile | AmbientJob::EntityPageReconcile
        )
        || (selected
            && (matches!(job, AmbientJob::Document | AmbientJob::Reconcile)
                || (matches!(job, AmbientJob::PageGrowth)
                    && !page_growth_terminal_no_match_committed)))
}

fn automatic_work_consumes_thermal_turn(selected: bool, llm_calls: usize, panicked: bool) -> bool {
    selected || llm_calls > 0 || panicked
}

/// Ambient-only provider facade that fails closed after forwarding one LLM
/// request. The scheduler is serialized today; this guard keeps the thermal
/// invariant true if a slice later gains a hidden nested call.
struct AmbientBudgetProvider {
    inner: Arc<dyn wenlan_core::llm_provider::LlmProvider>,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl AmbientBudgetProvider {
    fn new(inner: Arc<dyn wenlan_core::llm_provider::LlmProvider>) -> Self {
        Self::with_shared_calls(inner, Arc::new(std::sync::atomic::AtomicUsize::new(0)))
    }

    fn with_shared_calls(
        inner: Arc<dyn wenlan_core::llm_provider::LlmProvider>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        Self { inner, calls }
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl wenlan_core::llm_provider::LlmProvider for AmbientBudgetProvider {
    fn generate<'life0, 'async_trait>(
        &'life0 self,
        request: wenlan_core::llm_provider::LlmRequest,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<String, wenlan_core::llm_provider::LlmError>>
                + Send
                + 'async_trait,
        >,
    >
    where
        'life0: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            if self
                .calls
                .compare_exchange(
                    0,
                    1,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                )
                .is_err()
            {
                return Err(wenlan_core::llm_provider::LlmError::NotAvailable);
            }
            self.inner.generate(request).await
        })
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn backend(&self) -> wenlan_core::llm_provider::LlmBackend {
        self.inner.backend()
    }

    fn synthesis_token_limit(&self) -> usize {
        self.inner.synthesis_token_limit()
    }

    fn recommended_max_output(&self) -> u32 {
        self.inner.recommended_max_output()
    }

    fn context_size(&self) -> u32 {
        self.inner.context_size()
    }

    fn kind(&self) -> &'static str {
        self.inner.kind()
    }

    fn model_id(&self) -> String {
        self.inner.model_id()
    }
}

fn with_shared_automatic_budget(
    provider: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    calls: Arc<std::sync::atomic::AtomicUsize>,
) -> Option<Arc<dyn wenlan_core::llm_provider::LlmProvider>> {
    provider.map(|provider| {
        Arc::new(AmbientBudgetProvider::with_shared_calls(
            provider.clone(),
            calls,
        )) as Arc<dyn wenlan_core::llm_provider::LlmProvider>
    })
}

fn derived_receipt_sweep_due(last: Option<Instant>, now: Instant) -> bool {
    last.is_none_or(|last| now.duration_since(last) >= DERIVED_RECEIPT_SWEEP_INTERVAL)
}

async fn run_derived_receipt_sweep_if_due<F, Fut, E>(
    last: &mut Option<Instant>,
    now: Instant,
    sweep: F,
) -> Result<bool, E>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<(), E>>,
{
    if !derived_receipt_sweep_due(*last, now) {
        return Ok(false);
    }
    let result = sweep().await;
    *last = Some(now);
    result.map(|()| true)
}

/// Lightweight write-event tracker shared between store handlers and the scheduler.
///
/// `handle_store_memory` calls `record()` after each successful store.
/// The scheduler reads snapshots and drains completed bursts via `drain_up_to()`.
#[derive(Clone, Default)]
pub struct WriteSignal {
    inner: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
}

impl WriteSignal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a write event for an agent. The store route calls this after the
    /// durable write so automatic recap batching observes the latest write.
    pub fn record(&self, agent: &str) {
        let mut map = self.inner.lock().unwrap();
        map.entry(agent.to_string())
            .or_default()
            .push(Instant::now());
    }

    /// Record a write with an explicit timestamp (for testing).
    #[cfg(test)]
    pub fn record_at(&self, agent: &str, at: Instant) {
        let mut map = self.inner.lock().unwrap();
        map.entry(agent.to_string()).or_default().push(at);
    }

    /// Snapshot all agents and their timestamps. Does NOT drain.
    pub fn snapshot(&self) -> HashMap<String, Vec<Instant>> {
        self.inner.lock().unwrap().clone()
    }

    /// Atomically drain timestamps <= cutoff for one agent.
    /// Returns the drained timestamps. Timestamps after cutoff remain
    /// for the next burst — prevents TOCTOU race.
    pub fn drain_up_to(&self, agent: &str, cutoff: Instant) -> Vec<Instant> {
        let mut map = self.inner.lock().unwrap();
        if let Some(timestamps) = map.get_mut(agent) {
            let (drained, remaining): (Vec<_>, Vec<_>) =
                timestamps.drain(..).partition(|t| *t <= cutoff);
            if remaining.is_empty() {
                map.remove(agent);
            } else {
                *timestamps = remaining;
            }
            drained
        } else {
            Vec::new()
        }
    }

    /// True if any agent has written since `since`.
    pub fn has_activity_since(&self, since: Instant) -> bool {
        let map = self.inner.lock().unwrap();
        map.values().any(|ts| ts.iter().any(|t| *t > since))
    }
}

/// Compute the adaptive gap for a burst given its write timestamps.
///
/// Formula: `clamp(2 * median_interval, BURST_GAP_FLOOR, BURST_GAP_CEILING)`.
/// With < 2 timestamps (0 intervals), returns `BURST_GAP_CEILING` — a single
/// write naturally times out at the ceiling.
pub fn adaptive_gap(timestamps: &[Instant]) -> Duration {
    if timestamps.len() < 2 {
        return BURST_GAP_CEILING;
    }

    let mut sorted: Vec<Instant> = timestamps.to_vec();
    sorted.sort();

    let mut intervals: Vec<Duration> = Vec::with_capacity(sorted.len() - 1);
    for pair in sorted.windows(2) {
        intervals.push(pair[1].duration_since(pair[0]));
    }

    // Median of intervals
    intervals.sort();
    let median = if intervals.len().is_multiple_of(2) {
        let mid = intervals.len() / 2;
        (intervals[mid - 1] + intervals[mid]) / 2
    } else {
        intervals[intervals.len() / 2]
    };

    let gap = median * 2;
    gap.clamp(BURST_GAP_FLOOR, BURST_GAP_CEILING)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AutomaticTrigger {
    Maintenance,
    BurstEnd {
        agent: String,
        last_write: Instant,
        writes: usize,
        gap: Duration,
    },
    Idle,
    Daily,
    Backstop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaintenanceAdmission {
    None,
    Ready,
    YieldToDueSteep,
}

fn maintenance_admission(
    maintenance_pending: bool,
    maintenance_stage_ran_since_steep: bool,
) -> MaintenanceAdmission {
    match (maintenance_pending, maintenance_stage_ran_since_steep) {
        (false, _) => MaintenanceAdmission::None,
        (true, false) => MaintenanceAdmission::Ready,
        (true, true) => MaintenanceAdmission::YieldToDueSteep,
    }
}

impl AutomaticTrigger {
    fn steep_kind(&self) -> Option<wenlan_core::refinery::TriggerKind> {
        match self {
            Self::Maintenance => None,
            Self::BurstEnd { .. } => Some(wenlan_core::refinery::TriggerKind::BurstEnd),
            Self::Idle => Some(wenlan_core::refinery::TriggerKind::Idle),
            Self::Daily => Some(wenlan_core::refinery::TriggerKind::Daily),
            Self::Backstop => Some(wenlan_core::refinery::TriggerKind::Backstop),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AutomaticPhaseOutcome {
    selected: bool,
    progressed: bool,
    more: bool,
    retryable: bool,
    paused: bool,
    panicked: bool,
}

impl From<&wenlan_core::refinery::SteepPhaseSliceReport> for AutomaticPhaseOutcome {
    fn from(report: &wenlan_core::refinery::SteepPhaseSliceReport) -> Self {
        Self {
            selected: report.selected,
            progressed: report.progressed,
            more: report.more,
            retryable: report.retryable,
            paused: report.paused,
            panicked: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AutomaticMaintenanceOutcome {
    selected: bool,
    progressed: bool,
    more: bool,
    retryable: bool,
    paused: bool,
    panicked: bool,
}

impl From<&wenlan_core::maintenance::MaintenanceSliceReport> for AutomaticMaintenanceOutcome {
    fn from(report: &wenlan_core::maintenance::MaintenanceSliceReport) -> Self {
        Self {
            selected: report.selected,
            progressed: report.progressed,
            more: report.more,
            retryable: report.retryable,
            paused: report.paused,
            panicked: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutomaticBatchDisposition {
    Pending,
    Complete,
}

/// One finite automatic steep round. The trigger's completion marker is not
/// advanced until every eligible phase has yielded once. A phase with proven
/// additional progress moves to the tail, so it cannot monopolize the round.
struct AutomaticSteepBatch {
    trigger: AutomaticTrigger,
    phases: VecDeque<wenlan_core::refinery::Phase>,
}

/// Automatic work is deliberately narrower than the foreground steep API.
/// A phase earns this allowlist only after its scheduler entry point has a
/// durable cursor and a source-proven per-turn work bound.
fn automatic_phase_allowed(phase: wenlan_core::refinery::Phase) -> bool {
    matches!(phase, wenlan_core::refinery::Phase::ReDistill)
}

fn automatic_kind_has_work(kind: wenlan_core::refinery::TriggerKind) -> bool {
    wenlan_core::refinery::Phase::ALL
        .iter()
        .copied()
        .any(|phase| kind.runs_phase(phase) && automatic_phase_allowed(phase))
}

impl AutomaticSteepBatch {
    fn new(trigger: AutomaticTrigger, cursor: Option<wenlan_core::refinery::Phase>) -> Self {
        let kind = trigger
            .steep_kind()
            .expect("maintenance is scheduled through its own stage round");
        let mut phases = wenlan_core::refinery::Phase::ALL
            .iter()
            .copied()
            .filter(|phase| kind.runs_phase(*phase) && automatic_phase_allowed(*phase))
            .collect::<VecDeque<_>>();
        if let Some(cursor) = cursor {
            if let Some(position) = phases.iter().position(|phase| *phase == cursor) {
                phases.rotate_left(position);
            }
        }
        Self { trigger, phases }
    }

    fn next_phase(&self) -> Option<wenlan_core::refinery::Phase> {
        self.phases.front().copied()
    }

    #[cfg(test)]
    fn remaining_phases(&self) -> Vec<wenlan_core::refinery::Phase> {
        self.phases.iter().copied().collect()
    }

    fn complete_phase(
        &mut self,
        attempted: wenlan_core::refinery::Phase,
        outcome: AutomaticPhaseOutcome,
    ) -> AutomaticBatchDisposition {
        let selected = self
            .phases
            .pop_front()
            .expect("automatic batch cannot complete a phase while empty");
        debug_assert_eq!(selected, attempted);
        if outcome.progressed && outcome.more && !outcome.retryable && !outcome.paused {
            self.phases.push_back(attempted);
        }
        if self.phases.is_empty() {
            AutomaticBatchDisposition::Complete
        } else {
            AutomaticBatchDisposition::Pending
        }
    }

    fn cursor_after_attempt(
        &self,
        attempted: wenlan_core::refinery::Phase,
    ) -> wenlan_core::refinery::Phase {
        self.next_phase().unwrap_or_else(|| {
            let kind = self
                .trigger
                .steep_kind()
                .expect("automatic steep batch always has a steep trigger");
            let attempted_index = wenlan_core::refinery::Phase::ALL
                .iter()
                .position(|phase| *phase == attempted)
                .unwrap_or(0);
            (1..=wenlan_core::refinery::Phase::ALL.len())
                .map(|offset| {
                    wenlan_core::refinery::Phase::ALL
                        [(attempted_index + offset) % wenlan_core::refinery::Phase::ALL.len()]
                })
                .find(|phase| kind.runs_phase(*phase) && automatic_phase_allowed(*phase))
                .expect("a constructed automatic steep batch has an allowlisted phase")
        })
    }
}

struct AutomaticMaintenanceRound {
    stages: VecDeque<wenlan_core::maintenance::MaintenanceStage>,
}

impl AutomaticMaintenanceRound {
    fn new(cursor: Option<wenlan_core::maintenance::MaintenanceStage>) -> Self {
        let mut stages = wenlan_core::maintenance::MaintenanceStage::ALL
            .iter()
            .copied()
            .collect::<VecDeque<_>>();
        if let Some(cursor) = cursor {
            if let Some(position) = stages.iter().position(|stage| *stage == cursor) {
                stages.rotate_left(position);
            }
        }
        Self { stages }
    }

    fn next_stage(&self) -> Option<wenlan_core::maintenance::MaintenanceStage> {
        self.stages.front().copied()
    }

    #[cfg(test)]
    fn remaining_stages(&self) -> Vec<wenlan_core::maintenance::MaintenanceStage> {
        self.stages.iter().copied().collect()
    }

    fn complete_stage(
        &mut self,
        attempted: wenlan_core::maintenance::MaintenanceStage,
        outcome: AutomaticMaintenanceOutcome,
    ) -> AutomaticBatchDisposition {
        let selected = self
            .stages
            .pop_front()
            .expect("maintenance round cannot complete a stage while empty");
        debug_assert_eq!(selected, attempted);
        if outcome.progressed && outcome.more && !outcome.retryable && !outcome.paused {
            self.stages.push_back(attempted);
        }
        if self.stages.is_empty() {
            AutomaticBatchDisposition::Complete
        } else {
            AutomaticBatchDisposition::Pending
        }
    }

    fn cursor_after_attempt(
        &self,
        attempted: wenlan_core::maintenance::MaintenanceStage,
    ) -> wenlan_core::maintenance::MaintenanceStage {
        self.next_stage().unwrap_or_else(|| {
            let attempted_index = wenlan_core::maintenance::MaintenanceStage::ALL
                .iter()
                .position(|stage| *stage == attempted)
                .unwrap_or(0);
            wenlan_core::maintenance::MaintenanceStage::ALL
                [(attempted_index + 1) % wenlan_core::maintenance::MaintenanceStage::ALL.len()]
        })
    }
}

async fn load_automatic_maintenance_cursor(
    db: &wenlan_core::db::MemoryDB,
) -> Option<wenlan_core::maintenance::MaintenanceStage> {
    let value = db
        .get_app_metadata(AUTOMATIC_MAINTENANCE_STAGE_CURSOR_KEY)
        .await
        .ok()
        .flatten()?;
    wenlan_core::maintenance::MaintenanceStage::ALL
        .iter()
        .copied()
        .find(|stage| stage.as_str() == value)
}

async fn persist_automatic_maintenance_cursor(
    db: &wenlan_core::db::MemoryDB,
    stage: wenlan_core::maintenance::MaintenanceStage,
) {
    if let Err(error) = db
        .set_app_metadata(AUTOMATIC_MAINTENANCE_STAGE_CURSOR_KEY, stage.as_str())
        .await
    {
        tracing::warn!("[scheduler] failed to persist maintenance stage cursor '{stage}': {error}");
    }
}

fn automatic_phase_cursor_key(trigger: wenlan_core::refinery::TriggerKind) -> String {
    let suffix = match trigger {
        wenlan_core::refinery::TriggerKind::BurstEnd => "burst_end",
        wenlan_core::refinery::TriggerKind::Idle => "idle",
        wenlan_core::refinery::TriggerKind::Daily => "daily",
        wenlan_core::refinery::TriggerKind::Backstop => "backstop",
    };
    format!("{AUTOMATIC_STEEP_PHASE_CURSOR_PREFIX}_{suffix}")
}

async fn load_automatic_phase_cursor(
    db: &wenlan_core::db::MemoryDB,
    trigger: wenlan_core::refinery::TriggerKind,
) -> Option<wenlan_core::refinery::Phase> {
    let value = db
        .get_app_metadata(&automatic_phase_cursor_key(trigger))
        .await
        .ok()
        .flatten()?;
    wenlan_core::refinery::Phase::ALL
        .iter()
        .copied()
        .find(|phase| phase.as_str() == value)
}

async fn persist_automatic_phase_cursor(
    db: &wenlan_core::db::MemoryDB,
    trigger: wenlan_core::refinery::TriggerKind,
    phase: wenlan_core::refinery::Phase,
) {
    if let Err(error) = db
        .set_app_metadata(&automatic_phase_cursor_key(trigger), phase.as_str())
        .await
    {
        tracing::warn!(
            "[scheduler] failed to persist {:?} phase cursor '{}': {error}",
            trigger,
            phase
        );
    }
}

fn queues_maintenance_followup(trigger: &AutomaticTrigger) -> bool {
    matches!(trigger, AutomaticTrigger::Idle | AutomaticTrigger::Backstop)
}

/// Drain completed write bursts that cannot produce any bounded automatic
/// phase. This is bookkeeping only: it must not consume a thermal turn or
/// leave an unsupported BurstEnd trigger resident forever.
fn drain_expired_unactionable_bursts(write_signal: &WriteSignal, now: Instant) -> usize {
    let snapshot = write_signal.snapshot();
    let burst_end_supported = automatic_kind_has_work(wenlan_core::refinery::TriggerKind::BurstEnd);
    let mut drained = 0usize;
    for (agent, timestamps) in snapshot {
        if timestamps.is_empty() {
            continue;
        }
        if timestamps.len() >= MIN_BURST_WRITES && burst_end_supported {
            continue;
        }
        let Some(last_write) = timestamps.iter().copied().max() else {
            continue;
        };
        if now.saturating_duration_since(last_write) > adaptive_gap(&timestamps) {
            drained += write_signal.drain_up_to(&agent, last_write).len();
        }
    }
    drained
}

/// Choose at most one automatic heavy trigger for a scheduler poll. Burst
/// candidates are deterministic so a map iteration cannot accidentally turn
/// one poll into N inference-heavy runs.
fn select_due_automatic_trigger(
    now: Instant,
    snapshot: &HashMap<String, Vec<Instant>>,
    maintenance: MaintenanceAdmission,
    last_write_activity: Instant,
    idle_fired: bool,
    last_daily: Instant,
    last_backstop: Instant,
) -> Option<AutomaticTrigger> {
    if maintenance == MaintenanceAdmission::Ready {
        return Some(AutomaticTrigger::Maintenance);
    }
    let mut bursts = snapshot
        .iter()
        .filter_map(|(agent, timestamps)| {
            if timestamps.len() < MIN_BURST_WRITES {
                return None;
            }
            let last_write = timestamps.iter().copied().max()?;
            let gap = adaptive_gap(timestamps);
            (now.saturating_duration_since(last_write) > gap).then(|| AutomaticTrigger::BurstEnd {
                agent: agent.clone(),
                last_write,
                writes: timestamps.len(),
                gap,
            })
        })
        .collect::<Vec<_>>();
    bursts.sort_by(|left, right| match (left, right) {
        (
            AutomaticTrigger::BurstEnd {
                agent: left_agent, ..
            },
            AutomaticTrigger::BurstEnd {
                agent: right_agent, ..
            },
        ) => left_agent.cmp(right_agent),
        _ => std::cmp::Ordering::Equal,
    });
    if automatic_kind_has_work(wenlan_core::refinery::TriggerKind::BurstEnd) {
        if let Some(burst) = bursts.into_iter().next() {
            return Some(burst);
        }
    }
    if !idle_fired
        && now.saturating_duration_since(last_write_activity) >= AUTOMATIC_BATCH_IDLE_THRESHOLD
        && automatic_kind_has_work(wenlan_core::refinery::TriggerKind::Idle)
    {
        return Some(AutomaticTrigger::Idle);
    }
    if now.saturating_duration_since(last_daily) > DAILY_INTERVAL
        && automatic_kind_has_work(wenlan_core::refinery::TriggerKind::Daily)
    {
        return Some(AutomaticTrigger::Daily);
    }
    if now.saturating_duration_since(last_backstop) > BACKSTOP_INTERVAL
        && automatic_kind_has_work(wenlan_core::refinery::TriggerKind::Backstop)
    {
        return Some(AutomaticTrigger::Backstop);
    }
    (maintenance != MaintenanceAdmission::None).then_some(AutomaticTrigger::Maintenance)
}

#[cfg(test)]
fn idle_due(idle_fired: bool, idle_since: Instant, now: Instant) -> bool {
    !idle_fired && now.duration_since(idle_since) >= AUTOMATIC_BATCH_IDLE_THRESHOLD
}

/// Spawn the event-driven steep scheduler.
///
/// Runs a single tokio task with a 30-second poll loop. All work is awaited
/// inline, and the sticky lifecycle signal is checked at every owned boundary
/// so shutdown finishes the current item without starting another.
pub fn spawn_scheduler(
    shared: SharedState,
    write_signal: WriteSignal,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let resource_policy = ResourcePolicy::conservative();
        let thermal_policy = ThermalPolicy::conservative();
        // Initialize before the built-in startup delay so the first explicit
        // CPU refresh has a valid comparison window without sleeping inside a
        // scheduler poll.
        let mut resource_probe = SystemResourceProbe::new(Instant::now());
        let mut filesystem_resource_probe = SystemResourceProbe::new(Instant::now());
        if crate::lifecycle::sleep_or_shutdown(&mut shutdown, INITIAL_DELAY).await {
            tracing::info!("[scheduler] shutdown before initial delay completed");
            return;
        }

        let mut last_backstop = Instant::now();
        let mut idle_fired = false;
        let mut last_poll_activity = Instant::now();
        let ambient_started_at = Instant::now();
        let mut last_write_activity = write_signal
            .snapshot()
            .values()
            .flat_map(|timestamps| timestamps.iter().copied())
            .max()
            .unwrap_or(ambient_started_at);
        let mut ambient_schedule = AmbientSchedule::new(ambient_started_at);
        let mut last_derived_receipt_sweep = None;
        let mut maintenance_pending = false;
        let mut maintenance_stage_ran_since_steep = false;
        let mut steep_batch: Option<AutomaticSteepBatch> = None;
        let mut maintenance_round: Option<AutomaticMaintenanceRound> = None;
        let mut ambient_turn_owed = false;
        let mut last_deferred_resource_reason = None;

        // Load persisted daily timestamp from DB (survives restarts)
        let last_daily_epoch = load_last_daily(&shared).await;
        let mut last_daily = if last_daily_epoch > 0 {
            // Convert epoch secs to an Instant offset from now.
            let now_epoch = chrono::Utc::now().timestamp();
            let secs_ago = (now_epoch - last_daily_epoch).max(0) as u64;
            Instant::now()
                .checked_sub(Duration::from_secs(secs_ago))
                .unwrap_or_else(Instant::now) // can't go back that far → fire on next eligible poll
        } else {
            // No record → fire Daily on first eligible poll.
            // Offset must exceed 24h so `duration_since(last_daily) > 24h` is true.
            Instant::now()
                .checked_sub(Duration::from_secs(25 * 60 * 60))
                .unwrap_or_else(Instant::now)
        };

        tracing::info!(
            "[scheduler] started — poll every {}s",
            POLL_INTERVAL.as_secs()
        );

        loop {
            if crate::lifecycle::sleep_or_shutdown(&mut shutdown, POLL_INTERVAL).await {
                break;
            }

            let coordinator = {
                let state = shared.read().await;
                state.maintenance_coordinator.clone()
            };
            let Some(_maintenance_guard) = coordinator.try_begin_background() else {
                tracing::debug!("[scheduler] maintenance fence active; skipping poll");
                continue;
            };

            // Reset idle flag if any new activity arrived since last poll
            if write_signal.has_activity_since(last_poll_activity) {
                idle_fired = false;
                if let Some(latest) = write_signal
                    .snapshot()
                    .values()
                    .flat_map(|timestamps| timestamps.iter().copied())
                    .max()
                {
                    last_write_activity = last_write_activity.max(latest);
                }
            }
            last_poll_activity = Instant::now();

            // Snapshot shared state — drop the read guard immediately
            let snapshot = {
                let s = shared.read().await;
                s.db.clone().map(|db| {
                    (
                        db,
                        s.llm.clone(),
                        s.api_llm.clone(),
                        s.synthesis_llm.clone(),
                        s.external_llm.clone(),
                        s.prompts.clone(),
                        s.tuning.refinery.clone(),
                        s.tuning.confidence.clone(),
                        s.tuning.distillation.clone(),
                        s.startup_model_load_reserved
                            .load(std::sync::atomic::Ordering::Acquire),
                    )
                })
            };

            let Some((
                db,
                llm,
                api_llm,
                synthesis_llm,
                external_llm,
                prompts,
                refinery_cfg,
                confidence_cfg,
                distillation_cfg,
                startup_model_load_reserved,
            )) = snapshot
            else {
                tracing::debug!("[scheduler] db not initialized, skipping poll");
                continue;
            };

            // Read routing consent and the knowledge root once per poll. Missing
            // pins authorize no background inference; deterministic work stays
            // available and pinned-but-missing providers never cross sources.
            let runtime_config = wenlan_core::config::load_config();
            let everyday_pin = wenlan_core::refinery::EverydaySource::parse(
                runtime_config.everyday_source.as_deref(),
            );
            let synthesis_pin = wenlan_core::refinery::SynthesisSource::parse(
                runtime_config.synthesis_source.as_deref(),
            );
            if crate::lifecycle::shutdown_requested(&shutdown) {
                break;
            }
            let filesystem_resource_status =
                filesystem_resource_probe.sample(Instant::now(), resource_policy);
            // --- 0. Filesystem page watcher: md → DB ---
            //
            // md is canonical. When the user edits a page in Obsidian / VS
            // Code / etc., reflect the change back into the DB so refinery
            // and search stay aligned with what the user actually wrote.
            // Cheap: a dir scan + frontmatter parse + content compare per
            // file. No LLM, no embedding, no network. Skips files whose
            // origin_version frontmatter trails the DB (daemon wrote
            // last). Runs every poll so freshness ≈ POLL_INTERVAL.
            let knowledge_path = runtime_config.knowledge_path_or_default();
            match wenlan_core::sources::page_watcher::sync_filesystem_edits(&db, &knowledge_path)
                .await
            {
                Ok(stats) if stats.applied > 0 => {
                    tracing::info!(
                        "[scheduler] page-watcher applied {} fs_edit(s)",
                        stats.applied
                    );
                }
                Ok(_) => {}
                Err(e) => tracing::warn!("[scheduler] page-watcher error: {e}"),
            }
            if crate::lifecycle::shutdown_requested(&shutdown) {
                break;
            }

            // --- 0b. Directory sources: cheap mtime/hash sync only (§4). ---
            //
            // Mirrors the page-watcher Step-0 as a cheap per-poll pass: run the
            // SAME sync routine the HTTP handler runs over each registered
            // Directory source (mtime+hash diff, deletion propagation — no LLM),
            // Changed files are queued here; the ambient controller claims at
            // most one bounded document slice after resource/cooldown admission.
            if periodic_directory_sync_allowed(filesystem_resource_status.admitted) {
                sync_directory_sources(&db).await;
            } else {
                tracing::debug!(
                    "[scheduler] directory sync deferred reason={:?}",
                    filesystem_resource_status.block_reason
                );
            }
            if crate::lifecycle::shutdown_requested(&shutdown) {
                break;
            }

            // Filesystem sync can take long enough for fresh writes to arrive;
            // all time comparisons below must use a post-sync clock sample.
            let now = Instant::now();
            let resource_status = apply_host_activity(
                resource_probe.sample(now, resource_policy),
                sample_host_activity(),
            );

            // All automatic heavy work shares the same resource and cooldown
            // gate as the ambient lanes. The Idle recap trigger additionally
            // keeps its Wenlan-write batching window in trigger selection.
            refresh_last_write_activity(&write_signal, &mut last_write_activity);
            let changed_deferred_reason = observe_deferred_resource_reason(
                &mut last_deferred_resource_reason,
                resource_status.admitted,
                resource_status.block_reason,
            );
            if !resource_status.admitted {
                let log_deferred = |reason: ResourceBlockReason| {
                    format!(
                        "[scheduler] heavy work deferred reason={reason:?} cpu_percent={:?} available_memory_mb={:?}",
                        resource_status
                            .snapshot
                            .map(|snapshot| snapshot.cpu_usage_percent),
                        resource_status.snapshot.map(|snapshot| {
                            snapshot.available_memory_bytes / (1024 * 1024)
                        }),
                    )
                };
                if let Some(reason) = changed_deferred_reason {
                    tracing::info!("{}", log_deferred(reason));
                } else if let Some(reason) = resource_status.block_reason {
                    tracing::debug!("{}", log_deferred(reason));
                }
            }
            drain_expired_unactionable_bursts(&write_signal, now);
            let snap = write_signal.snapshot();

            let selected_automatic = automatic_heavy_turn_allowed(
                background_heavy_resource_admitted(
                    resource_status,
                    startup_model_load_reserved,
                    matches!(
                        synthesis_pin,
                        Some(wenlan_core::refinery::SynthesisSource::OnDevice)
                    ),
                ),
                ambient_turn_owed,
                now,
                ambient_schedule.next_allowed_at,
            )
            .then(|| {
                steep_batch
                    .as_ref()
                    .map(|batch| batch.trigger.clone())
                    .or_else(|| {
                        select_due_automatic_trigger(
                            now,
                            &snap,
                            maintenance_admission(
                                maintenance_pending,
                                maintenance_stage_ran_since_steep,
                            ),
                            last_write_activity,
                            idle_fired,
                            last_daily,
                            last_backstop,
                        )
                    })
            })
            .flatten();
            let mut automatic_work_ran = false;

            if let Some(trigger) = selected_automatic {
                let shared_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let budgeted_llm = with_shared_automatic_budget(llm.as_ref(), shared_calls.clone());
                let budgeted_api_llm =
                    with_shared_automatic_budget(api_llm.as_ref(), shared_calls.clone());
                let budgeted_synthesis_llm =
                    with_shared_automatic_budget(synthesis_llm.as_ref(), shared_calls.clone());
                let budgeted_external_llm =
                    with_shared_automatic_budget(external_llm.as_ref(), shared_calls.clone());
                let maintenance_llm = resolve_maintenance_provider(
                    synthesis_pin,
                    budgeted_synthesis_llm.as_ref(),
                    budgeted_api_llm.as_ref(),
                    budgeted_external_llm.as_ref(),
                    budgeted_llm.as_ref(),
                );
                let label = match &trigger {
                    AutomaticTrigger::Maintenance => "Maintenance",
                    AutomaticTrigger::BurstEnd { .. } => "BurstEnd",
                    AutomaticTrigger::Idle => "Idle",
                    AutomaticTrigger::Daily => "Daily",
                    AutomaticTrigger::Backstop => "Backstop",
                };
                // Setup and provider resolution may race with a stop request.
                // Re-check the sticky signal at the final launch boundary so
                // shutdown never starts a new steep/maintenance item.
                if crate::lifecycle::shutdown_requested(&shutdown) {
                    break;
                }

                let (automatic_selected, automatic_panicked) = match trigger {
                    AutomaticTrigger::Maintenance => {
                        if maintenance_round.is_none() {
                            let cursor = load_automatic_maintenance_cursor(&db).await;
                            maintenance_round = Some(AutomaticMaintenanceRound::new(cursor));
                        }
                        let stage = maintenance_round
                            .as_ref()
                            .and_then(AutomaticMaintenanceRound::next_stage)
                            .expect("pending maintenance round has a stage");
                        tracing::info!(
                            "[scheduler] Maintenance stage={stage} — deferred automatic turn"
                        );
                        let outcome = fire_maintenance_stage_safe(
                            db.as_ref(),
                            maintenance_llm.as_ref(),
                            &prompts,
                            &distillation_cfg,
                            Some(knowledge_path.as_path()),
                            stage,
                            label,
                        )
                        .await;
                        let (disposition, cursor) = {
                            let round = maintenance_round
                                .as_mut()
                                .expect("maintenance round survives its stage");
                            let disposition = round.complete_stage(stage, outcome);
                            (disposition, round.cursor_after_attempt(stage))
                        };
                        persist_automatic_maintenance_cursor(&db, cursor).await;
                        if disposition == AutomaticBatchDisposition::Complete {
                            maintenance_round = None;
                            maintenance_pending = false;
                            maintenance_stage_ran_since_steep = false;
                        } else {
                            maintenance_stage_ran_since_steep = true;
                        }
                        (outcome.selected, outcome.panicked)
                    }
                    trigger => {
                        maintenance_stage_ran_since_steep = false;
                        if steep_batch.is_none() {
                            let kind = trigger
                                .steep_kind()
                                .expect("maintenance handled in the previous match arm");
                            let cursor = load_automatic_phase_cursor(&db, kind).await;
                            steep_batch = Some(AutomaticSteepBatch::new(trigger.clone(), cursor));
                        }
                        let (kind, phase) = {
                            let batch = steep_batch
                                .as_ref()
                                .expect("automatic steep batch initialized above");
                            (
                                batch
                                    .trigger
                                    .steep_kind()
                                    .expect("automatic steep batch has a steep trigger"),
                                batch
                                    .next_phase()
                                    .expect("automatic steep batch has an eligible phase"),
                            )
                        };
                        if let AutomaticTrigger::BurstEnd {
                            agent, writes, gap, ..
                        } = &trigger
                        {
                            tracing::info!(
                                "[scheduler] BurstEnd phase={} for agent '{}' — {} writes, gap {:?}",
                                phase,
                                agent,
                                writes,
                                gap
                            );
                        } else {
                            tracing::info!("[scheduler] {label} phase={phase}");
                        }
                        let outcome = fire_steep_phase_safe(
                            &db,
                            budgeted_llm.as_ref(),
                            budgeted_api_llm.as_ref(),
                            budgeted_synthesis_llm.as_ref(),
                            budgeted_external_llm.as_ref(),
                            &prompts,
                            &refinery_cfg,
                            &confidence_cfg,
                            &distillation_cfg,
                            kind,
                            phase,
                            label,
                        )
                        .await;
                        let (disposition, cursor) = {
                            let batch = steep_batch
                                .as_mut()
                                .expect("automatic steep batch survives its phase");
                            let disposition = batch.complete_phase(phase, outcome);
                            (disposition, batch.cursor_after_attempt(phase))
                        };
                        // Persist after every attempt, including captured errors
                        // and panics, so one poison phase cannot pin restarts.
                        persist_automatic_phase_cursor(&db, kind, cursor).await;

                        if disposition == AutomaticBatchDisposition::Complete {
                            let completed = steep_batch
                                .take()
                                .expect("completed automatic steep batch exists")
                                .trigger;
                            if queues_maintenance_followup(&completed) {
                                maintenance_pending = true;
                            }
                            match completed {
                                AutomaticTrigger::BurstEnd {
                                    agent, last_write, ..
                                } => {
                                    write_signal.drain_up_to(&agent, last_write);
                                }
                                AutomaticTrigger::Idle => idle_fired = true,
                                AutomaticTrigger::Daily => {
                                    last_daily = Instant::now();
                                    let epoch = chrono::Utc::now().timestamp().to_string();
                                    if let Err(error) =
                                        db.set_app_metadata("last_daily_steep_ts", &epoch).await
                                    {
                                        tracing::warn!(
                                            "[scheduler] failed to persist last_daily_steep_ts: {error}"
                                        );
                                    }
                                }
                                AutomaticTrigger::Backstop => last_backstop = Instant::now(),
                                AutomaticTrigger::Maintenance => unreachable!(
                                    "maintenance never enters an automatic steep batch"
                                ),
                            }
                        }
                        (outcome.selected, outcome.panicked)
                    }
                };
                automatic_work_ran = true;
                // A multi-phase steep or maintenance round must yield one
                // admitted slot to the ambient round-robin before continuing.
                ambient_turn_owed = true;
                let completion = Instant::now();
                let llm_calls = shared_calls.load(std::sync::atomic::Ordering::SeqCst);
                if automatic_work_consumes_thermal_turn(
                    automatic_selected,
                    llm_calls,
                    automatic_panicked,
                ) {
                    ambient_schedule.note_thermal_work_completion(
                        completion,
                        completion.saturating_duration_since(now),
                        thermal_policy,
                    );
                }
                tracing::info!(
                    "[scheduler] automatic trigger={} selected={} llm_calls={} panicked={} elapsed_ms={} next_eligible_ms={}",
                    label,
                    automatic_selected,
                    llm_calls,
                    automatic_panicked,
                    completion.saturating_duration_since(now).as_millis(),
                    ambient_schedule
                        .next_allowed_at
                        .saturating_duration_since(completion)
                        .as_millis(),
                );
            }
            if crate::lifecycle::shutdown_requested(&shutdown) {
                break;
            }

            if let Err(error) =
                run_derived_receipt_sweep_if_due(&mut last_derived_receipt_sweep, now, || {
                    db.record_derived_artifact_sweep()
                })
                .await
            {
                tracing::warn!("[scheduler] derived receipt sweep error: {error}");
            }
            if crate::lifecycle::shutdown_requested(&shutdown) {
                break;
            }

            // --- 5. Ambient enrichment: one due job, one durable slice, one
            //        LLM request maximum. Never detached. ---
            refresh_last_write_activity(&write_signal, &mut last_write_activity);
            let ambient_now = Instant::now();
            if !automatic_work_ran
                && ambient_turn_allowed(
                    background_heavy_resource_admitted(
                        resource_status,
                        startup_model_load_reserved,
                        matches!(
                            everyday_pin,
                            Some(wenlan_core::refinery::EverydaySource::OnDevice)
                        ),
                    ),
                    ambient_now,
                    ambient_schedule.next_allowed_at,
                )
            {
                let ambient_provider_available = resolve_ambient_provider(
                    everyday_pin,
                    api_llm.as_ref(),
                    external_llm.as_ref(),
                    llm.as_ref(),
                )
                .is_some();
                let availability = AmbientAvailability::for_provider(ambient_provider_available);
                if let Some(job) = ambient_schedule.select_due(ambient_now, availability) {
                    // Availability/selection is intentionally cheap, but may
                    // still race with shutdown. Do not start another ambient
                    // item after the stop signal became sticky.
                    if crate::lifecycle::shutdown_requested(&shutdown) {
                        break;
                    }
                    tracing::info!(
                        "[scheduler] ambient turn started job={:?} cpu_percent={:?} available_memory_mb={:?}",
                        job,
                        resource_status
                            .snapshot
                            .map(|snapshot| snapshot.cpu_usage_percent),
                        resource_status
                            .snapshot
                            .map(|snapshot| snapshot.available_memory_bytes / (1024 * 1024)),
                    );
                    let report = run_ambient_job_safe(
                        job,
                        &db,
                        llm.as_ref(),
                        api_llm.as_ref(),
                        external_llm.as_ref(),
                        everyday_pin,
                        &prompts,
                        &refinery_cfg,
                        &distillation_cfg,
                        Some(knowledge_path.as_path()),
                    )
                    .await;
                    let completion = Instant::now();
                    ambient_schedule.note_job_result(
                        report.job,
                        completion,
                        !should_backoff_ambient_lane(report.selected, report.llm_calls),
                    );
                    // Fresh-document preparation can be CPU-heavy even before
                    // an LLM call, so a selected document also consumes the
                    // conservative thermal turn budget.
                    if report.panicked
                        || ambient_work_consumes_thermal_turn(
                            report.job,
                            report.selected,
                            report.llm_calls,
                            report.page_growth_terminal_no_match_committed,
                        )
                    {
                        ambient_schedule.note_thermal_work_completion(
                            completion,
                            report.elapsed,
                            thermal_policy,
                        );
                    }
                    tracing::info!(
                        "[scheduler] ambient job={:?} selected={} llm_calls={} panicked={} elapsed_ms={} next_eligible_ms={}",
                        report.job,
                        report.selected,
                        report.llm_calls,
                        report.panicked,
                        report.elapsed.as_millis(),
                        ambient_schedule
                            .next_allowed_at
                            .saturating_duration_since(completion)
                            .as_millis(),
                    );
                }
                // The ambient lane received its admission opportunity. Empty
                // work is enough to release the debt; known selected work owns
                // the shared cooldown through `note_thermal_work_completion`.
                ambient_turn_owed = false;
            }
            if crate::lifecycle::shutdown_requested(&shutdown) {
                break;
            }
        }
        tracing::info!("[scheduler] stopped after shutdown request");
    })
}

/// Background polling respects an explicit pause but keeps probing unavailable
/// roots so transient filesystem failures can recover automatically.
fn should_poll_directory_source(source: &wenlan_types::sources::Source) -> bool {
    source.source_type == wenlan_types::sources::SourceType::Directory
        && !matches!(source.status, wenlan_types::sources::SyncStatus::Paused)
}

/// One Directory-source sync + document-enrichment-queue-drive pass (§4).
/// Factored out of the 30s poll loop so it is unit-testable without the timer.
/// Sync every recoverable Directory source via the shared source route. This
/// only discovers changes and updates the durable queue; LLM work is owned by
/// the ambient scheduler below.
async fn sync_directory_sources(db: &Arc<wenlan_core::db::MemoryDB>) {
    let config = wenlan_core::config::load_config();
    for source in config
        .sources
        .iter()
        .filter(|source| should_poll_directory_source(source))
    {
        if let Err(e) =
            crate::source_routes::sync_directory_source(db.clone(), source, &config).await
        {
            tracing::warn!("[scheduler] directory sync '{}' failed: {e}", source.id);
        }
    }
}

#[derive(Debug)]
struct AmbientTurnReport {
    job: AmbientJob,
    selected: bool,
    page_growth_terminal_no_match_committed: bool,
    llm_calls: usize,
    panicked: bool,
    elapsed: Duration,
}

fn resolve_ambient_provider(
    everyday_pin: Option<wenlan_core::refinery::EverydaySource>,
    api_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
) -> Option<Arc<dyn wenlan_core::llm_provider::LlmProvider>> {
    wenlan_core::refinery::resolve_everyday(everyday_pin, api_llm, external_llm, llm)
        .llm
        .cloned()
}

fn resolve_maintenance_provider(
    synthesis_pin: Option<wenlan_core::refinery::SynthesisSource>,
    synthesis_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    api_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
) -> Option<Arc<dyn wenlan_core::llm_provider::LlmProvider>> {
    wenlan_core::refinery::resolve_synthesis(
        synthesis_pin,
        synthesis_llm,
        api_llm,
        external_llm,
        llm,
    )
    .llm
    .cloned()
}

#[allow(clippy::too_many_arguments)]
async fn run_ambient_job_safe(
    job: AmbientJob,
    db: &Arc<wenlan_core::db::MemoryDB>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    api_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    everyday_pin: Option<wenlan_core::refinery::EverydaySource>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    refinery: &wenlan_core::tuning::RefineryConfig,
    distillation: &wenlan_core::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> AmbientTurnReport {
    let started = Instant::now();
    let future = std::panic::AssertUnwindSafe(run_ambient_job(
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
    ));
    match futures::FutureExt::catch_unwind(future).await {
        Ok(report) => report,
        Err(error) => {
            let message = if let Some(message) = error.downcast_ref::<&str>() {
                (*message).to_string()
            } else if let Some(message) = error.downcast_ref::<String>() {
                message.clone()
            } else {
                "unknown panic".to_string()
            };
            tracing::error!(
                "[scheduler] ambient job={job:?} PANICKED — scheduler continues: {message}"
            );
            AmbientTurnReport {
                job,
                selected: true,
                page_growth_terminal_no_match_committed: false,
                llm_calls: 0,
                panicked: true,
                elapsed: started.elapsed(),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_ambient_job(
    job: AmbientJob,
    db: &Arc<wenlan_core::db::MemoryDB>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    api_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    everyday_pin: Option<wenlan_core::refinery::EverydaySource>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    refinery: &wenlan_core::tuning::RefineryConfig,
    distillation: &wenlan_core::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
) -> AmbientTurnReport {
    let started = Instant::now();
    let observed = resolve_ambient_provider(everyday_pin, api_llm, external_llm, llm)
        .map(|provider| Arc::new(AmbientBudgetProvider::new(provider)));
    let provider: Option<Arc<dyn wenlan_core::llm_provider::LlmProvider>> = observed
        .as_ref()
        .map(|provider| provider.clone() as Arc<dyn wenlan_core::llm_provider::LlmProvider>);

    let mut page_growth_terminal_no_match_committed = false;
    let selected = match job {
        AmbientJob::Document => {
            run_document_enrichment_slice_tick(db, provider.as_ref(), prompts).await > 0
        }
        AmbientJob::Classification => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::ingest::run_classification_enrichment_slice(db, provider, prompts)
                .await
            {
                Ok(report) => report.selected,
                Err(error) => {
                    tracing::warn!("[scheduler] classification slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::StructuredExtract => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::ingest::run_structured_extract_slice(db, provider, prompts).await {
                Ok(report) => report.selected,
                Err(error) => {
                    tracing::warn!("[scheduler] structured extraction slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::Entity => {
            let Some(provider) = provider.clone() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            let prompts = prompts.clone();
            match db
                .run_entity_enrichment_slice_with_auto_link(
                    refinery.entity_link_distance as f32,
                    move |content: String| {
                        let provider = provider.clone();
                        let prompts = prompts.clone();
                        async move {
                            wenlan_core::kg::entity_extraction::extract_kg(
                                &provider, &prompts, &content,
                            )
                            .await
                        }
                    },
                )
                .await
            {
                Ok(selected) => selected > 0,
                Err(error) => {
                    tracing::warn!("[scheduler] entity enrichment slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::Title => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::post_ingest::run_title_enrichment_slice(db, provider).await {
                Ok(report) => report.selected,
                Err(error) => {
                    tracing::warn!("[scheduler] title enrichment slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::PageGrowth => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::post_ingest::run_page_growth_slice(
                db,
                provider,
                prompts,
                distillation.page_growth_threshold,
                knowledge_path,
            )
            .await
            {
                Ok(report) => {
                    page_growth_terminal_no_match_committed =
                        report.terminal_no_match && report.committed;
                    report.selected
                }
                Err(error) => {
                    tracing::warn!("[scheduler] page growth slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::Reconcile => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::reconcile::run_reconcile_slice(
                db,
                provider,
                prompts,
                refinery,
                distillation,
            )
            .await
            {
                Ok(report) => report.progressed,
                Err(error) => {
                    tracing::warn!("[scheduler] reconcile slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::Citation => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::citations::run_citation_backfill_slice(db, provider, prompts).await {
                Ok(selected) => selected > 0,
                Err(error) => {
                    tracing::warn!("[scheduler] citation backfill slice error: {error}");
                    false
                }
            }
        }
        AmbientJob::EdgesReconcile => match db.reconcile_edges_parity().await {
            Ok(report) => {
                tracing::info!(
                    "[scheduler] edges parity sweep: drift={} (missing={}, extra={}, corrupt={}) epoch={}",
                    report.drift_count,
                    report.missing_count,
                    report.extra_count,
                    report.corrupt_count,
                    report.epoch
                );
                true
            }
            Err(error) => {
                tracing::warn!("[scheduler] edges parity sweep error: {error}");
                false
            }
        },
        AmbientJob::EntityPageReconcile => match db.reconcile_entity_page_parity().await {
            Ok(report) => {
                tracing::info!(
                    "[scheduler] entity/page parity sweep: drift={} (missing={}, extra={}, corrupt={}) epoch={}",
                    report.drift_count,
                    report.missing_count,
                    report.extra_count,
                    report.corrupt_count,
                    report.epoch
                );
                true
            }
            Err(error) => {
                tracing::warn!("[scheduler] entity/page parity sweep error: {error}");
                false
            }
        },
        AmbientJob::EdgeGroundingPromote => {
            let Some(provider) = provider.as_ref() else {
                return AmbientTurnReport {
                    job,
                    selected: false,
                    page_growth_terminal_no_match_committed: false,
                    llm_calls: 0,
                    panicked: false,
                    elapsed: started.elapsed(),
                };
            };
            match wenlan_core::edge_grounding::run_edge_grounding_slice(db, provider, prompts).await
            {
                Ok(report) => report.progressed,
                Err(error) => {
                    tracing::warn!("[scheduler] edge grounding slice error: {error}");
                    false
                }
            }
        }
    };

    AmbientTurnReport {
        job,
        selected,
        page_growth_terminal_no_match_committed,
        llm_calls: observed
            .as_ref()
            .map_or(0, |provider| provider.call_count()),
        panicked: false,
        elapsed: started.elapsed(),
    }
}

/// Claim at most one document and advance it by at most one LLM request.
/// Paused rows retain their existing backoff through `claim_next_pending`.
async fn run_document_enrichment_slice_tick(
    db: &Arc<wenlan_core::db::MemoryDB>,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    prompts: &wenlan_core::prompts::PromptRegistry,
) -> usize {
    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    match db.claim_next_pending_for_provider(llm.is_some()).await {
        Ok(Some(entry)) => {
            let slice = std::panic::AssertUnwindSafe(
                wenlan_core::document_enrichment::run_document_enrichment_slice(
                    db,
                    &entry,
                    Some(&knowledge_path),
                    llm,
                    prompts,
                ),
            );
            match futures::FutureExt::catch_unwind(slice).await {
                Ok(_) => 1,
                Err(panic) => {
                    wenlan_core::document_enrichment::pause_document_enrichment_after_panic(
                        db, &entry,
                    )
                    .await;
                    std::panic::resume_unwind(panic);
                }
            }
        }
        Ok(None) => 0,
        Err(error) => {
            tracing::warn!("[scheduler] claim_next_pending failed: {error}");
            0
        }
    }
}

async fn fire_maintenance_stage_safe(
    db: &wenlan_core::db::MemoryDB,
    llm: Option<&Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    distillation_cfg: &wenlan_core::tuning::DistillationConfig,
    knowledge_path: Option<&std::path::Path>,
    stage: wenlan_core::maintenance::MaintenanceStage,
    label: &str,
) -> AutomaticMaintenanceOutcome {
    let config = wenlan_core::maintenance::MaintenanceTickConfig {
        page_match_threshold: distillation_cfg.page_match_threshold,
        formation_threshold: distillation_cfg.formation_threshold,
        page_min_cluster_size: distillation_cfg.page_min_cluster_size,
        token_limit: distillation_cfg.ondevice_token_limit,
        max_unlinked_cluster_size: distillation_cfg.max_unlinked_cluster_size,
        max_grouped_cluster_size: distillation_cfg.max_grouped_cluster_size,
        max_per_tick: 5,
    };
    let result =
        std::panic::AssertUnwindSafe(wenlan_core::maintenance::run_maintenance_stage_slice(
            db,
            llm,
            prompts,
            &config,
            knowledge_path,
            stage,
        ));
    match futures::FutureExt::catch_unwind(result).await {
        Ok(Ok(report)) => {
            let result = &report.result;
            tracing::info!(
                "[scheduler] {label} maintenance stage={stage}: selected={}, progressed={}, more={}, retryable={}, paused={}; work pages={}, pairs={}, source_rows={}, raw_seeds={}, eligible_seed_probes={}, ANN_rows={}, fully_filtered_seeds={}, truncated={}; {} merge card(s), {} discovery card(s), {} retro card(s) from {} observed, {} machine refresh(es), {} human card(s), {} orphan label(s), {} overview refresh(es)",
                report.selected,
                report.progressed,
                report.more,
                report.retryable,
                report.paused,
                report.work.pages_examined,
                report.work.pairs_examined,
                report.work.source_rows_examined,
                report.work.seeds_examined,
                report.work.eligible_seeds_probed,
                report.work.neighbor_rows_examined,
                report.work.fully_filtered_seeds,
                report.work.truncated,
                result.merge_cards_emitted,
                result.discovery_cards_emitted,
                result.retro_cards_emitted,
                result.retro_expected_card_volume,
                result.stale_machine_refreshed,
                result.stale_human_cards,
                result.orphan_labels_checked,
                result.overview_refreshed
            );
            AutomaticMaintenanceOutcome::from(&report)
        }
        Ok(Err(error)) => {
            tracing::warn!("[scheduler] {label} maintenance stage={stage} error: {error}");
            AutomaticMaintenanceOutcome {
                retryable: true,
                ..AutomaticMaintenanceOutcome::default()
            }
        }
        Err(error) => {
            let message = if let Some(message) = error.downcast_ref::<&str>() {
                message.to_string()
            } else if let Some(message) = error.downcast_ref::<String>() {
                message.clone()
            } else {
                "unknown panic".to_string()
            };
            tracing::error!(
                "[scheduler] {label} maintenance stage={stage} PANICKED — scheduler continues: {message}"
            );
            AutomaticMaintenanceOutcome {
                panicked: true,
                retryable: true,
                ..AutomaticMaintenanceOutcome::default()
            }
        }
    }
}

/// Load the persisted last_daily_steep_ts from DB. Returns epoch seconds or 0.
async fn load_last_daily(shared: &SharedState) -> i64 {
    let s = shared.read().await;
    if let Some(db) = s.db.as_ref() {
        match db.get_app_metadata("last_daily_steep_ts").await {
            Ok(Some(val)) => val.parse::<i64>().unwrap_or(0),
            _ => 0,
        }
    } else {
        0
    }
}

/// Fire one steep phase with panic isolation. Every outcome returns to the
/// finite batch so its durable cursor advances even after an error or panic.
#[allow(clippy::too_many_arguments)]
async fn fire_steep_phase_safe(
    db: &wenlan_core::db::MemoryDB,
    llm: Option<&std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    api_llm: Option<&std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    synthesis_llm: Option<&std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    refinery_cfg: &wenlan_core::tuning::RefineryConfig,
    confidence_cfg: &wenlan_core::tuning::ConfidenceConfig,
    distillation_cfg: &wenlan_core::tuning::DistillationConfig,
    trigger: wenlan_core::refinery::TriggerKind,
    phase: wenlan_core::refinery::Phase,
    label: &str,
) -> AutomaticPhaseOutcome {
    let result = std::panic::AssertUnwindSafe(fire_steep_phase(
        db,
        llm,
        api_llm,
        synthesis_llm,
        external_llm,
        prompts,
        refinery_cfg,
        confidence_cfg,
        distillation_cfg,
        trigger,
        phase,
        label,
    ));
    match futures::FutureExt::catch_unwind(result).await {
        Ok(outcome) => outcome,
        Err(error) => {
            let message = if let Some(message) = error.downcast_ref::<&str>() {
                message.to_string()
            } else if let Some(message) = error.downcast_ref::<String>() {
                message.clone()
            } else {
                "unknown panic".to_string()
            };
            tracing::error!(
                "[scheduler] {label} phase={phase} PANICKED — scheduler continues: {message}"
            );
            AutomaticPhaseOutcome {
                panicked: true,
                retryable: true,
                ..AutomaticPhaseOutcome::default()
            }
        }
    }
}

/// Fire one phase with the given trigger, log its result, and return scheduler
/// control metadata. Phase errors are captured inside `SteepPhaseSliceReport`.
#[allow(clippy::too_many_arguments)]
async fn fire_steep_phase(
    db: &wenlan_core::db::MemoryDB,
    llm: Option<&std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    api_llm: Option<&std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    synthesis_llm: Option<&std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    external_llm: Option<&std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    refinery_cfg: &wenlan_core::tuning::RefineryConfig,
    confidence_cfg: &wenlan_core::tuning::ConfidenceConfig,
    distillation_cfg: &wenlan_core::tuning::DistillationConfig,
    trigger: wenlan_core::refinery::TriggerKind,
    phase: wenlan_core::refinery::Phase,
    label: &str,
) -> AutomaticPhaseOutcome {
    let started = std::time::Instant::now();
    let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
    match wenlan_core::refinery::run_periodic_steep_phase_with_api(
        db,
        llm,
        api_llm,
        synthesis_llm,
        external_llm,
        prompts,
        refinery_cfg,
        confidence_cfg,
        distillation_cfg,
        Some(&knowledge_path),
        trigger,
        phase,
    )
    .await
    {
        Ok(report) => {
            let errors = report
                .result
                .phases
                .iter()
                .filter(|phase| phase.error.is_some())
                .count();
            tracing::info!(
                "[scheduler] {label} phase={phase} complete in {}ms — {} error(s), selected={}, progressed={}, more={}, retryable={}, paused={}",
                started.elapsed().as_millis(),
                errors,
                report.selected,
                report.progressed,
                report.more,
                report.retryable,
                report.paused,
            );

            for phase_result in &report.result.phases {
                if phase_result.nudge != wenlan_core::refinery::Nudge::Silent {
                    if let Some(ref headline) = phase_result.headline {
                        if let Err(e) = db
                            .log_agent_activity("origin", "steep", &[], None, headline)
                            .await
                        {
                            tracing::warn!(
                                "[scheduler] log activity for phase {} failed: {}",
                                phase_result.name,
                                e
                            );
                        }
                    }
                }
            }
            AutomaticPhaseOutcome::from(&report)
        }
        Err(error) => {
            tracing::warn!("[scheduler] {label} phase={phase} error: {error}");
            AutomaticPhaseOutcome {
                retryable: true,
                ..AutomaticPhaseOutcome::default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_schedule_includes_fixed_memory_stages() {
        let now = Instant::now();
        let mut schedule = AmbientSchedule::new(now);
        let available = AmbientAvailability {
            document: true,
            classification: true,
            structured_extract: true,
            entity: true,
            title: true,
            page_growth: true,
            reconcile: true,
            citation: true,
            edges_reconcile: true,
            entity_page_reconcile: true,
            edge_grounding_promote: true,
        };
        assert_eq!(
            (0..11)
                .filter_map(|_| schedule.select_due(now, available))
                .collect::<Vec<_>>(),
            vec![
                AmbientJob::Document,
                AmbientJob::Classification,
                AmbientJob::StructuredExtract,
                AmbientJob::Entity,
                AmbientJob::Title,
                AmbientJob::PageGrowth,
                AmbientJob::Reconcile,
                AmbientJob::Citation,
                AmbientJob::EdgesReconcile,
                AmbientJob::EntityPageReconcile,
                AmbientJob::EdgeGroundingPromote,
            ]
        );
    }

    #[test]
    fn unconfigured_pin_allows_only_deterministic_document_preparation() {
        let availability = AmbientAvailability::for_provider(false);
        assert!(
            availability.supports(AmbientJob::Document),
            "model consent must not prevent deterministic parse + embedding"
        );
        for job in [
            AmbientJob::Classification,
            AmbientJob::StructuredExtract,
            AmbientJob::Entity,
            AmbientJob::Title,
            AmbientJob::PageGrowth,
            AmbientJob::Reconcile,
            AmbientJob::Citation,
            AmbientJob::EdgeGroundingPromote,
        ] {
            assert!(
                !availability.supports(job),
                "{job:?} must remain pending until an authorized provider is available"
            );
        }
    }

    // Non-vacuity guard for the promotion lane's opt-in flag gate.
    // `unconfigured_pin_allows_only_deterministic_document_preparation` above uses
    // `for_provider(false)`, whose `provider_available && flag` term short-circuits
    // on the provider alone — it never reaches the flag, so it cannot prove the
    // flag gate does anything. This test pins `provider_available = true` and
    // toggles ONLY `WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE`, proving the lane is
    // default-OFF and turns on only when the flag is set (mirroring reconcile /
    // citation gating).
    #[test]
    fn edge_grounding_promote_lane_gated_by_flag_even_with_provider() {
        temp_env::with_var("WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE", None::<&str>, || {
            assert!(
                !AmbientAvailability::for_provider(true)
                    .supports(AmbientJob::EdgeGroundingPromote),
                "promotion lane must stay parked when the opt-in flag is unset, even with a provider"
            );
        });
        temp_env::with_var("WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE", Some("1"), || {
            assert!(
                AmbientAvailability::for_provider(true).supports(AmbientJob::EdgeGroundingPromote),
                "promotion lane must be available with an authorized provider AND the flag ON"
            );
        });
    }

    #[test]
    fn automatic_batch_runs_one_eligible_phase_per_turn_and_completes_only_after_last_phase() {
        let mut batch = AutomaticSteepBatch::new(AutomaticTrigger::Idle, None);
        let expected = wenlan_core::refinery::Phase::ALL
            .iter()
            .copied()
            .filter(|phase| {
                wenlan_core::refinery::TriggerKind::Idle.runs_phase(*phase)
                    && automatic_phase_allowed(*phase)
            })
            .collect::<Vec<_>>();
        assert_eq!(batch.remaining_phases(), expected.as_slice());

        for (index, expected_phase) in expected.iter().copied().enumerate() {
            assert_eq!(batch.next_phase(), Some(expected_phase));
            let disposition = batch.complete_phase(
                expected_phase,
                AutomaticPhaseOutcome {
                    progressed: true,
                    ..AutomaticPhaseOutcome::default()
                },
            );
            if index + 1 == expected.len() {
                assert_eq!(disposition, AutomaticBatchDisposition::Complete);
            } else {
                assert_eq!(disposition, AutomaticBatchDisposition::Pending);
            }
        }
    }

    #[test]
    fn automatic_batch_contains_only_bounded_redistill() {
        for trigger in [AutomaticTrigger::Idle, AutomaticTrigger::Backstop] {
            let batch = AutomaticSteepBatch::new(trigger, None);
            assert_eq!(
                batch.remaining_phases(),
                vec![wenlan_core::refinery::Phase::ReDistill]
            );
        }

        assert!(AutomaticSteepBatch::new(AutomaticTrigger::Daily, None)
            .remaining_phases()
            .is_empty());
    }

    #[test]
    fn automatic_cursor_never_leaves_safe_allowlist() {
        let mut batch = AutomaticSteepBatch::new(
            AutomaticTrigger::Idle,
            Some(wenlan_core::refinery::Phase::ReDistill),
        );
        assert_eq!(
            batch.complete_phase(
                wenlan_core::refinery::Phase::ReDistill,
                AutomaticPhaseOutcome::default(),
            ),
            AutomaticBatchDisposition::Complete
        );
        assert_eq!(
            batch.cursor_after_attempt(wenlan_core::refinery::Phase::ReDistill),
            wenlan_core::refinery::Phase::ReDistill
        );
    }

    #[test]
    fn successful_more_phase_rotates_to_tail() {
        let mut batch = AutomaticSteepBatch::new(
            AutomaticTrigger::Idle,
            Some(wenlan_core::refinery::Phase::ReDistill),
        );
        assert_eq!(
            batch.next_phase(),
            Some(wenlan_core::refinery::Phase::ReDistill)
        );

        assert_eq!(
            batch.complete_phase(
                wenlan_core::refinery::Phase::ReDistill,
                AutomaticPhaseOutcome {
                    progressed: true,
                    more: true,
                    ..AutomaticPhaseOutcome::default()
                },
            ),
            AutomaticBatchDisposition::Pending
        );
        assert_eq!(
            batch.next_phase(),
            Some(wenlan_core::refinery::Phase::ReDistill),
            "the sole bounded phase waits for the next admitted thermal turn"
        );
        assert_eq!(
            batch.remaining_phases().last(),
            Some(&wenlan_core::refinery::Phase::ReDistill)
        );
    }

    #[test]
    fn retryable_or_paused_phase_is_not_requeued_in_current_trigger() {
        for outcome in [
            AutomaticPhaseOutcome {
                progressed: true,
                more: true,
                retryable: true,
                paused: false,
                ..AutomaticPhaseOutcome::default()
            },
            AutomaticPhaseOutcome {
                progressed: true,
                more: true,
                retryable: false,
                paused: true,
                ..AutomaticPhaseOutcome::default()
            },
        ] {
            let mut batch = AutomaticSteepBatch::new(
                AutomaticTrigger::Idle,
                Some(wenlan_core::refinery::Phase::ReDistill),
            );
            batch.complete_phase(wenlan_core::refinery::Phase::ReDistill, outcome);
            assert!(!batch
                .remaining_phases()
                .contains(&wenlan_core::refinery::Phase::ReDistill));
        }
    }

    #[test]
    fn maintenance_round_stays_pending_until_every_stage_attempted() {
        let mut round = AutomaticMaintenanceRound::new(None);
        assert_eq!(
            round.remaining_stages(),
            wenlan_core::maintenance::MaintenanceStage::ALL
        );

        for (index, stage) in wenlan_core::maintenance::MaintenanceStage::ALL
            .iter()
            .copied()
            .enumerate()
        {
            assert_eq!(round.next_stage(), Some(stage));
            let disposition = round.complete_stage(stage, AutomaticMaintenanceOutcome::default());
            if index + 1 == wenlan_core::maintenance::MaintenanceStage::ALL.len() {
                assert_eq!(disposition, AutomaticBatchDisposition::Complete);
            } else {
                assert_eq!(disposition, AutomaticBatchDisposition::Pending);
            }
        }
    }

    #[test]
    fn maintenance_round_cursor_rotates_a_paused_stage_behind_the_rest() {
        let mut round = AutomaticMaintenanceRound::new(Some(
            wenlan_core::maintenance::MaintenanceStage::RetroReview,
        ));
        assert_eq!(
            round.next_stage(),
            Some(wenlan_core::maintenance::MaintenanceStage::RetroReview)
        );
        round.complete_stage(
            wenlan_core::maintenance::MaintenanceStage::RetroReview,
            AutomaticMaintenanceOutcome {
                progressed: true,
                more: true,
                paused: true,
                retryable: false,
                ..AutomaticMaintenanceOutcome::default()
            },
        );
        assert_eq!(
            round.next_stage(),
            Some(wenlan_core::maintenance::MaintenanceStage::NearDuplicate),
            "paused/retryable work waits for a later maintenance round"
        );
    }

    #[test]
    fn maintenance_successful_more_stage_rotates_to_tail() {
        let stage = wenlan_core::maintenance::MaintenanceStage::NearDuplicate;
        let mut round = AutomaticMaintenanceRound::new(Some(stage));

        let disposition = round.complete_stage(
            stage,
            AutomaticMaintenanceOutcome {
                progressed: true,
                more: true,
                retryable: false,
                paused: false,
                ..AutomaticMaintenanceOutcome::default()
            },
        );

        assert_eq!(disposition, AutomaticBatchDisposition::Pending);
        assert_eq!(round.remaining_stages().last().copied(), Some(stage));
        assert_eq!(
            round.remaining_stages().len(),
            wenlan_core::maintenance::MaintenanceStage::ALL.len(),
            "bounded cursor work must stay in the same finite round until EOF"
        );
    }

    #[tokio::test]
    async fn automatic_phase_cursor_persists_after_retryable_attempt() {
        let (db, _db_dir) = new_test_db().await;
        let mut batch = AutomaticSteepBatch::new(
            AutomaticTrigger::Idle,
            Some(wenlan_core::refinery::Phase::ReDistill),
        );
        batch.complete_phase(
            wenlan_core::refinery::Phase::ReDistill,
            AutomaticPhaseOutcome {
                retryable: true,
                ..AutomaticPhaseOutcome::default()
            },
        );
        let cursor = batch.cursor_after_attempt(wenlan_core::refinery::Phase::ReDistill);
        persist_automatic_phase_cursor(&db, wenlan_core::refinery::TriggerKind::Idle, cursor).await;

        assert_eq!(
            load_automatic_phase_cursor(&db, wenlan_core::refinery::TriggerKind::Idle).await,
            Some(wenlan_core::refinery::Phase::ReDistill)
        );
    }

    #[tokio::test]
    async fn maintenance_stage_cursor_persists_after_attempt() {
        let (db, _db_dir) = new_test_db().await;
        let mut round = AutomaticMaintenanceRound::new(Some(
            wenlan_core::maintenance::MaintenanceStage::StalePage,
        ));
        round.complete_stage(
            wenlan_core::maintenance::MaintenanceStage::StalePage,
            AutomaticMaintenanceOutcome::default(),
        );
        let cursor =
            round.cursor_after_attempt(wenlan_core::maintenance::MaintenanceStage::StalePage);
        persist_automatic_maintenance_cursor(&db, cursor).await;

        assert_eq!(
            load_automatic_maintenance_cursor(&db).await,
            Some(wenlan_core::maintenance::MaintenanceStage::Overview)
        );
    }

    #[test]
    fn ambient_schedule_round_robins_all_due_jobs() {
        let now = Instant::now();
        let mut schedule = AmbientSchedule::new(now);
        assert!(
            schedule.last_entity.is_none(),
            "entity is due on first turn"
        );
        assert!(
            schedule.last_reconcile.is_none(),
            "reconcile is due on first turn"
        );
        assert!(
            schedule.last_citation.is_none(),
            "citation is due on first turn"
        );
        let available = AmbientAvailability {
            document: true,
            classification: true,
            structured_extract: true,
            entity: true,
            title: true,
            page_growth: true,
            reconcile: true,
            citation: true,
            edges_reconcile: true,
            entity_page_reconcile: true,
            edge_grounding_promote: true,
        };

        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Document)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Classification)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::StructuredExtract)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Entity)
        );
        assert_eq!(schedule.select_due(now, available), Some(AmbientJob::Title));
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::PageGrowth)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Reconcile)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Citation)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::EdgesReconcile)
        );
    }

    #[test]
    fn selected_backlog_lane_stays_due_after_global_cooldown() {
        let now = Instant::now();
        let mut schedule = AmbientSchedule::new(now);
        let available = AmbientAvailability {
            document: true,
            classification: true,
            structured_extract: true,
            entity: true,
            title: true,
            page_growth: true,
            reconcile: true,
            citation: true,
            edges_reconcile: true,
            entity_page_reconcile: true,
            edge_grounding_promote: true,
        };

        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Document)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Classification)
        );
        schedule.note_job_result(AmbientJob::Classification, now, true);
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::StructuredExtract)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Entity)
        );
        assert_eq!(schedule.select_due(now, available), Some(AmbientJob::Title));
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::PageGrowth)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Reconcile)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Citation)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::EdgesReconcile)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::EntityPageReconcile)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::EdgeGroundingPromote)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Document)
        );
        assert_eq!(
            schedule.select_due(now, available),
            Some(AmbientJob::Classification),
            "known backlog should be paced by the global cooldown, not another 30-minute delay"
        );
    }

    #[test]
    fn attempted_inference_is_not_treated_as_an_empty_lane() {
        assert!(!should_backoff_ambient_lane(false, 1));
        assert!(should_backoff_ambient_lane(false, 0));
        assert!(!should_backoff_ambient_lane(true, 0));
    }

    #[test]
    fn edge_reconcile_full_pass_is_interval_and_thermal_paced_even_on_error() {
        let now = Instant::now();
        let mut schedule = AmbientSchedule::new(now);
        let edge_only = AmbientAvailability {
            document: false,
            classification: false,
            structured_extract: false,
            entity: false,
            title: false,
            page_growth: false,
            reconcile: false,
            citation: false,
            edges_reconcile: true,
            entity_page_reconcile: false,
            edge_grounding_promote: false,
        };

        assert_eq!(
            schedule.select_due(now, edge_only),
            Some(AmbientJob::EdgesReconcile)
        );
        schedule.note_job_result(AmbientJob::EdgesReconcile, now, true);
        assert_eq!(schedule.last_edges_reconcile, Some(now));
        assert_eq!(
            schedule.select_due(
                now + EDGES_RECONCILE_SWEEP_INTERVAL - Duration::from_secs(1),
                edge_only
            ),
            None
        );
        assert_eq!(
            schedule.select_due(now + EDGES_RECONCILE_SWEEP_INTERVAL, edge_only),
            Some(AmbientJob::EdgesReconcile)
        );
        assert!(
            ambient_work_consumes_thermal_turn(AmbientJob::EdgesReconcile, false, 0, false),
            "a failed final watermark write can follow a full scan and must still cool down"
        );
    }

    #[test]
    fn entity_page_reconcile_full_pass_is_interval_and_thermal_paced_even_on_error() {
        let now = Instant::now();
        let mut schedule = AmbientSchedule::new(now);
        let entity_page_only = AmbientAvailability {
            document: false,
            classification: false,
            structured_extract: false,
            entity: false,
            title: false,
            page_growth: false,
            reconcile: false,
            citation: false,
            edges_reconcile: false,
            entity_page_reconcile: true,
            edge_grounding_promote: false,
        };

        assert_eq!(
            schedule.select_due(now, entity_page_only),
            Some(AmbientJob::EntityPageReconcile)
        );
        schedule.note_job_result(AmbientJob::EntityPageReconcile, now, true);
        assert_eq!(schedule.last_entity_page_reconcile, Some(now));
        assert_eq!(
            schedule.select_due(
                now + ENTITY_PAGE_RECONCILE_SWEEP_INTERVAL - Duration::from_secs(1),
                entity_page_only
            ),
            None
        );
        assert_eq!(
            schedule.select_due(now + ENTITY_PAGE_RECONCILE_SWEEP_INTERVAL, entity_page_only),
            Some(AmbientJob::EntityPageReconcile)
        );
        assert!(
            ambient_work_consumes_thermal_turn(AmbientJob::EntityPageReconcile, false, 0, false),
            "a failed final watermark write can follow a full scan and must still cool down"
        );
    }

    #[test]
    fn only_committed_page_growth_terminal_no_match_skips_the_thermal_turn() {
        assert!(!ambient_work_consumes_thermal_turn(
            AmbientJob::PageGrowth,
            true,
            0,
            true,
        ));
        assert!(ambient_work_consumes_thermal_turn(
            AmbientJob::PageGrowth,
            true,
            0,
            false,
        ));
        assert!(ambient_work_consumes_thermal_turn(
            AmbientJob::PageGrowth,
            true,
            1,
            true,
        ));
        assert!(!ambient_work_consumes_thermal_turn(
            AmbientJob::PageGrowth,
            false,
            0,
            false,
        ));
    }

    #[test]
    fn refresh_activity_observes_writes_that_arrive_during_a_poll() {
        let writes = WriteSignal::new();
        let base = Instant::now();
        let fresh = base + Duration::from_secs(5);
        let mut last_write_activity = base;
        writes.record_at("claude", fresh);

        refresh_last_write_activity(&writes, &mut last_write_activity);

        assert_eq!(last_write_activity, fresh);
    }

    #[test]
    fn ambient_turn_uses_resources_and_cooldown_not_global_write_recency() {
        let now = Instant::now();
        assert!(
            ambient_turn_allowed(true, now, now - Duration::from_secs(1),),
            "an unrelated recent write must not hold all ambient backlog"
        );
        assert!(!ambient_turn_allowed(
            true,
            now,
            now + Duration::from_secs(1),
        ));
        assert!(ambient_turn_allowed(true, now, now,));
        assert!(
            !ambient_turn_allowed(false, now, now),
            "ambient work cannot start while the whole machine is busy"
        );
    }

    #[test]
    fn automatic_heavy_turn_uses_resources_and_cooldown_not_global_write_recency() {
        let now = Instant::now();
        assert!(
            automatic_heavy_turn_allowed(true, false, now, now,),
            "trigger-specific batching belongs in trigger selection, not global admission"
        );
    }

    #[test]
    fn pending_automatic_round_yields_one_admission_to_ambient_lane() {
        let now = Instant::now();
        assert!(automatic_heavy_turn_allowed(true, false, now, now));
        assert!(
            !automatic_heavy_turn_allowed(true, true, now, now),
            "an unfinished steep/maintenance round must not monopolize every admitted turn"
        );
        assert!(
            !automatic_heavy_turn_allowed(false, false, now, now),
            "automatic heavy work must defer to foreground system pressure"
        );
    }

    #[test]
    fn resource_policy_rejects_cpu_or_memory_pressure() {
        let policy = ResourcePolicy::conservative();
        let idle = ResourceSnapshot {
            cpu_usage_percent: 8.0,
            available_memory_bytes: 8 * 1024 * 1024 * 1024,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
        };
        assert_eq!(policy.block_reason(idle), None);

        assert_eq!(
            policy.block_reason(ResourceSnapshot {
                cpu_usage_percent: 60.0,
                ..idle
            }),
            Some(ResourceBlockReason::CpuBusy)
        );
        assert_eq!(
            policy.block_reason(ResourceSnapshot {
                available_memory_bytes: 512 * 1024 * 1024,
                ..idle
            }),
            Some(ResourceBlockReason::MemoryPressure)
        );
    }

    #[test]
    fn host_activity_policy_defers_recent_physical_input() {
        assert_eq!(
            host_activity_block_reason(HostActivitySnapshot::Observed {
                thermal_state: 0,
                idle_for: FOREGROUND_INPUT_IDLE_THRESHOLD - Duration::from_millis(1),
            }),
            Some(ResourceBlockReason::ForegroundActive),
        );
        assert_eq!(
            host_activity_block_reason(HostActivitySnapshot::Observed {
                thermal_state: 0,
                idle_for: FOREGROUND_INPUT_IDLE_THRESHOLD,
            }),
            None,
            "one full two-sample window without physical input is admissible",
        );
    }

    #[test]
    fn host_activity_policy_defers_every_non_nominal_thermal_state() {
        for thermal_state in 1..=3 {
            assert_eq!(
                host_activity_block_reason(HostActivitySnapshot::Observed {
                    thermal_state,
                    idle_for: FOREGROUND_INPUT_IDLE_THRESHOLD,
                }),
                Some(ResourceBlockReason::ThermalPressure),
            );
        }
    }

    #[test]
    fn host_activity_policy_fails_closed_only_when_supported_telemetry_is_unavailable() {
        assert_eq!(
            host_activity_block_reason(HostActivitySnapshot::Unavailable),
            Some(ResourceBlockReason::HostActivityUnavailable),
        );
        assert_eq!(
            host_activity_block_reason(HostActivitySnapshot::Unsupported),
            None,
            "non-macOS targets retain the portable CPU/RAM gate",
        );
    }

    #[test]
    fn host_activity_veto_is_applied_after_portable_resource_admission() {
        let admitted = ResourceStatus {
            admitted: true,
            snapshot: Some(ResourceSnapshot {
                cpu_usage_percent: 8.0,
                available_memory_bytes: 8 * GIB,
                total_memory_bytes: 16 * GIB,
            }),
            block_reason: None,
        };

        let blocked = apply_host_activity(
            admitted,
            HostActivitySnapshot::Observed {
                thermal_state: 0,
                idle_for: Duration::ZERO,
            },
        );
        assert!(!blocked.admitted);
        assert_eq!(
            blocked.block_reason,
            Some(ResourceBlockReason::ForegroundActive),
        );
        assert_eq!(blocked.snapshot.unwrap().cpu_usage_percent, 8.0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_host_activity_probe_links_and_returns_a_supported_state() {
        match sample_host_activity() {
            HostActivitySnapshot::Observed { thermal_state, .. } => {
                assert!(thermal_state <= 3);
            }
            HostActivitySnapshot::Unavailable => {}
            HostActivitySnapshot::Unsupported => {
                panic!("the macOS probe must not report an unsupported platform");
            }
        }
    }

    #[test]
    fn deferred_resource_reason_only_escalates_on_transition() {
        let mut last_reason = None;

        assert_eq!(
            observe_deferred_resource_reason(
                &mut last_reason,
                false,
                Some(ResourceBlockReason::CpuBusy),
            ),
            Some(ResourceBlockReason::CpuBusy),
        );
        assert_eq!(
            observe_deferred_resource_reason(
                &mut last_reason,
                false,
                Some(ResourceBlockReason::CpuBusy),
            ),
            None,
            "an unchanged blocker must remain debug-only",
        );
        assert_eq!(
            observe_deferred_resource_reason(
                &mut last_reason,
                false,
                Some(ResourceBlockReason::MemoryPressure),
            ),
            Some(ResourceBlockReason::MemoryPressure),
        );
        assert_eq!(
            observe_deferred_resource_reason(&mut last_reason, true, None),
            None,
        );
        assert_eq!(last_reason, None, "admission resets the transition state");
        assert_eq!(
            observe_deferred_resource_reason(
                &mut last_reason,
                false,
                Some(ResourceBlockReason::CpuBusy),
            ),
            Some(ResourceBlockReason::CpuBusy),
            "a new blocked episode must remain visible at info level",
        );
        assert_eq!(last_reason, Some(ResourceBlockReason::CpuBusy));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_resource_probe_does_not_collapse_available_memory_to_zero() {
        let started = Instant::now();
        let mut probe = SystemResourceProbe::new(started);
        let status = probe.sample(
            started + sysinfo::MINIMUM_CPU_UPDATE_INTERVAL,
            ResourcePolicy::conservative(),
        );
        let snapshot = status
            .snapshot
            .expect("the first eligible refresh must return a resource snapshot");

        assert_ne!(snapshot.total_memory_bytes, 0);
        assert_ne!(
            snapshot.available_memory_bytes, 0,
            "a supported macOS probe must not collapse reclaimable RAM to zero"
        );
    }

    /// Daemon-off target-Mac premise check for the one persistent live soak.
    /// Uses the production probe and cadence without loading either model.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "five-minute target-Mac resource baseline; opt in through WENLAN_RB01_BASELINE=1"]
    fn rb01_daemon_off_resource_baseline_can_open_gate() {
        assert_eq!(
            std::env::var("WENLAN_RB01_BASELINE").as_deref(),
            Ok("1"),
            "explicit baseline opt-in is required",
        );

        const SAMPLE_COUNT: usize = 10;
        let policy = ResourcePolicy::conservative();
        let started = Instant::now();
        let mut probe = SystemResourceProbe::new(started);
        let mut samples = Vec::with_capacity(SAMPLE_COUNT);

        for sample_number in 1..=SAMPLE_COUNT {
            let due = started + POLL_INTERVAL * sample_number as u32;
            std::thread::sleep(due.saturating_duration_since(Instant::now()));
            let status = probe.sample(Instant::now(), policy);
            samples.push(serde_json::json!({
                "sample": sample_number,
                "admitted": status.admitted,
                "block_reason": status.block_reason.map(|reason| format!("{reason:?}")),
                "cpu_percent": status.snapshot.map(|snapshot| snapshot.cpu_usage_percent),
                "available_memory_mb": status.snapshot.map(|snapshot| {
                    snapshot.available_memory_bytes / (1024 * 1024)
                })
            }));
        }

        let cpu_over_limit_count = samples
            .iter()
            .filter(|sample| {
                sample["cpu_percent"]
                    .as_f64()
                    .is_some_and(|cpu| cpu > f64::from(policy.max_cpu_usage_percent))
            })
            .count();
        let memory_pressure_count = samples
            .iter()
            .filter(|sample| sample["block_reason"] == "MemoryPressure")
            .count();
        let first_admitted_sample = samples
            .iter()
            .find(|sample| sample["admitted"] == true)
            .and_then(|sample| sample["sample"].as_u64());

        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "rb01_daemon_off_baseline": {
                    "sample_interval_secs": POLL_INTERVAL.as_secs(),
                    "sample_count": SAMPLE_COUNT,
                    "cpu_limit_percent": policy.max_cpu_usage_percent,
                    "cpu_over_limit_count": cpu_over_limit_count,
                    "memory_pressure_count": memory_pressure_count,
                    "first_admitted_sample": first_admitted_sample,
                    "samples": samples
                }
            }))
            .expect("serialize daemon-off resource baseline")
        );

        assert!(
            cpu_over_limit_count * 2 < SAMPLE_COUNT,
            "the daemon-off host exceeded the production CPU gate in at least half of samples; do not run the persistent soak",
        );
        assert!(
            first_admitted_sample.is_some(),
            "the exact production resource gate never opened while the daemon was off; use the recorded binding reason before changing policy",
        );
    }

    #[test]
    fn scheduler_pins_exact_sysinfo_with_macos_memory_accounting_fix() {
        let manifest = include_str!("../../../Cargo.toml");
        assert!(
            manifest
                .lines()
                .any(|line| line.trim() == "sysinfo = \"=0.38.3\""),
            "the scheduler must retain the reviewed exact sysinfo 0.38.3 pin"
        );

        let lock = include_str!("../../../Cargo.lock");
        let packages: Vec<_> = lock
            .split("[[package]]")
            .filter(|package| {
                package
                    .lines()
                    .any(|line| line.trim() == "name = \"sysinfo\"")
            })
            .collect();
        assert_eq!(
            packages.len(),
            1,
            "Cargo.lock must resolve exactly one sysinfo package"
        );
        let package = packages[0];
        let version = package
            .lines()
            .find_map(|line| line.trim().strip_prefix("version = \""))
            .and_then(|version| version.strip_suffix('"'))
            .expect("sysinfo package must carry a version");
        let mut parts = version
            .split('.')
            .map(|part| part.parse::<u32>().expect("numeric sysinfo version"));
        let version = (
            parts.next().expect("sysinfo major version"),
            parts.next().expect("sysinfo minor version"),
            parts.next().expect("sysinfo patch version"),
        );

        assert_eq!(
            version,
            (0, 38, 3),
            "Cargo.lock must retain the reviewed sysinfo 0.38.3 version"
        );
    }

    #[test]
    fn startup_model_admission_reserves_the_model_above_the_normal_floor() {
        let policy = ResourcePolicy::conservative().with_additional_memory_headroom(3 * GIB);
        let total = 16 * GIB;
        let ratio_floor = total * 15 / 100;
        let required = 3 * GIB + (2 * GIB).max(ratio_floor);

        assert_eq!(
            policy.block_reason(ResourceSnapshot {
                cpu_usage_percent: 8.0,
                available_memory_bytes: required - 1,
                total_memory_bytes: total,
            }),
            Some(ResourceBlockReason::MemoryPressure),
            "startup must not load a model into the ordinary scheduler reserve"
        );
        assert_eq!(
            policy.block_reason(ResourceSnapshot {
                cpu_usage_percent: 8.0,
                available_memory_bytes: required,
                total_memory_bytes: total,
            }),
            None,
            "the exact model working set plus normal reserve is admissible"
        );
    }

    #[test]
    fn startup_model_reservation_blocks_only_on_device_routes() {
        assert!(!startup_model_reservation_blocks_route(false, true));
        assert!(!startup_model_reservation_blocks_route(true, false));
        assert!(startup_model_reservation_blocks_route(true, true));
        let admitted = ResourceStatus {
            admitted: true,
            snapshot: Some(ResourceSnapshot {
                cpu_usage_percent: 8.0,
                available_memory_bytes: 8 * GIB,
                total_memory_bytes: 16 * GIB,
            }),
            block_reason: None,
        };
        let blocked = ResourceStatus {
            admitted: false,
            ..admitted
        };
        assert!(background_heavy_resource_admitted(admitted, true, false));
        assert!(!background_heavy_resource_admitted(admitted, true, true));
        assert!(!background_heavy_resource_admitted(blocked, true, false));
    }

    #[test]
    fn on_device_background_reserves_two_gib_for_inference() {
        let total = 16 * GIB;
        let ordinary_floor = total * 15 / 100;
        let required = ordinary_floor + 2 * GIB;
        let status = |available_memory_bytes| ResourceStatus {
            admitted: true,
            snapshot: Some(ResourceSnapshot {
                cpu_usage_percent: 8.0,
                available_memory_bytes,
                total_memory_bytes: total,
            }),
            block_reason: None,
        };

        assert!(background_heavy_resource_admitted(
            status(required - 1),
            false,
            false,
        ));
        assert!(!background_heavy_resource_admitted(
            status(required - 1),
            false,
            true,
        ));
        assert!(background_heavy_resource_admitted(
            status(required),
            false,
            true,
        ));
    }

    #[test]
    fn periodic_directory_sync_requires_resource_admission() {
        assert!(periodic_directory_sync_allowed(true));
        assert!(!periodic_directory_sync_allowed(false));
    }

    #[test]
    fn resource_admission_requires_two_idle_samples_and_resets_on_pressure() {
        let policy = ResourcePolicy::conservative();
        let idle = ResourceSnapshot {
            cpu_usage_percent: 8.0,
            available_memory_bytes: 8 * 1024 * 1024 * 1024,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
        };
        let busy = ResourceSnapshot {
            cpu_usage_percent: 60.0,
            ..idle
        };
        let mut admission = ResourceAdmission::default();

        assert!(!admission.observe(idle, policy));
        assert!(admission.observe(idle, policy));
        assert!(!admission.observe(busy, policy));
        assert!(!admission.observe(idle, policy));
        assert!(admission.observe(idle, policy));
    }

    #[test]
    fn rb01_profile_admission_fails_closed_and_requires_two_healthy_samples() {
        let policy = ResourcePolicy::conservative();
        let idle = ResourceSnapshot {
            cpu_usage_percent: 8.0,
            available_memory_bytes: 8 * GIB,
            total_memory_bytes: 16 * GIB,
        };
        let mut admission = Rb01ProfileAdmission::default();

        assert_eq!(
            admission.observe(None, Some(0), policy),
            Err(Rb01ProfileBlockReason::ResourceUnavailable)
        );
        assert_eq!(
            admission.observe(Some(idle), None, policy),
            Err(Rb01ProfileBlockReason::ThermalUnavailable)
        );
        assert_eq!(
            admission.observe(Some(idle), Some(1), policy),
            Err(Rb01ProfileBlockReason::ThermalPressure)
        );
        assert_eq!(
            admission.observe(
                Some(ResourceSnapshot {
                    cpu_usage_percent: 21.0,
                    ..idle
                }),
                Some(0),
                policy,
            ),
            Err(Rb01ProfileBlockReason::CpuBusy)
        );
        assert_eq!(
            admission.observe(
                Some(ResourceSnapshot {
                    available_memory_bytes: 2 * GIB - 1,
                    ..idle
                }),
                Some(0),
                policy,
            ),
            Err(Rb01ProfileBlockReason::MemoryPressure)
        );

        assert_eq!(
            admission.observe(Some(idle), Some(0), policy),
            Err(Rb01ProfileBlockReason::Warming)
        );
        assert_eq!(admission.observe(Some(idle), Some(0), policy), Ok(()));

        assert_eq!(
            admission.observe(
                Some(ResourceSnapshot {
                    cpu_usage_percent: 80.0,
                    ..idle
                }),
                Some(0),
                policy,
            ),
            Err(Rb01ProfileBlockReason::CpuBusy)
        );
        assert_eq!(
            admission.observe(Some(idle), Some(0), policy),
            Err(Rb01ProfileBlockReason::Warming)
        );
        assert_eq!(admission.observe(Some(idle), Some(0), policy), Ok(()));
    }

    #[test]
    fn rb01_profile_wait_retries_model_load_cpu_spike_then_requires_quiet_samples() {
        let policy = ResourcePolicy::conservative();
        let idle = ResourceSnapshot {
            cpu_usage_percent: 8.0,
            available_memory_bytes: 8 * GIB,
            total_memory_bytes: 16 * GIB,
        };
        let busy = ResourceSnapshot {
            cpu_usage_percent: 34.0,
            ..idle
        };
        let mut admission = Rb01ProfileAdmission::default();

        assert_eq!(
            rb01_profile_sample_action(&mut admission, Some(busy), Some(0), policy, 1, 4,),
            Rb01ProfileSampleAction::Retry,
            "model-load CPU must settle before the profile is rejected"
        );
        assert_eq!(
            rb01_profile_sample_action(&mut admission, Some(idle), Some(0), policy, 2, 4,),
            Rb01ProfileSampleAction::Retry,
            "one quiet sample is not enough after the load spike"
        );
        assert_eq!(
            rb01_profile_sample_action(&mut admission, Some(idle), Some(0), policy, 3, 4,),
            Rb01ProfileSampleAction::Admit
        );

        let mut admission = Rb01ProfileAdmission::default();
        assert_eq!(
            rb01_profile_sample_action(
                &mut admission,
                Some(ResourceSnapshot {
                    available_memory_bytes: 2 * GIB - 1,
                    ..idle
                }),
                Some(0),
                policy,
                1,
                4,
            ),
            Rb01ProfileSampleAction::Fail(Rb01ProfileBlockReason::MemoryPressure),
            "model residency must not wait through memory pressure"
        );
        assert_eq!(
            rb01_profile_sample_action(
                &mut admission,
                Some(ResourceSnapshot {
                    cpu_usage_percent: 34.0,
                    available_memory_bytes: 2 * GIB - 1,
                    ..idle
                }),
                Some(0),
                policy,
                1,
                4,
            ),
            Rb01ProfileSampleAction::Fail(Rb01ProfileBlockReason::MemoryPressure),
            "memory pressure must win over a simultaneous retryable CPU spike"
        );
        assert_eq!(
            rb01_profile_sample_action(&mut admission, Some(idle), Some(1), policy, 1, 4,),
            Rb01ProfileSampleAction::Fail(Rb01ProfileBlockReason::ThermalPressure),
            "thermal pressure must remain immediately fatal"
        );
        assert_eq!(
            rb01_profile_sample_action(&mut admission, Some(busy), Some(0), policy, 4, 4,),
            Rb01ProfileSampleAction::Fail(Rb01ProfileBlockReason::CpuBusy),
            "CPU retry must remain bounded"
        );
    }

    #[test]
    fn rb01_profile_admission_requires_explicit_opt_in() {
        assert!(!rb01_profile_requested(None));
        assert!(!rb01_profile_requested(Some("0")));
        assert!(!rb01_profile_requested(Some("true")));
        assert!(rb01_profile_requested(Some("1")));
    }

    #[test]
    fn rb01_profile_lane_includes_page_growth_no_match() {
        assert_eq!(
            Rb01ProfileLane::from_env("page-growth").map(Rb01ProfileLane::as_str),
            Some("page-growth")
        );
    }

    #[test]
    fn measured_recovery_floor_is_two_minutes_and_long_turns_extend_it() {
        let policy = ThermalPolicy::conservative();
        assert_eq!(
            policy.cooldown_after(Duration::from_secs(1)),
            Duration::from_secs(120)
        );
        assert!(
            policy.cooldown_after(Duration::from_secs(60)) > Duration::from_secs(120),
            "a long request must earn a longer recovery window than the measured floor"
        );
    }

    #[test]
    fn unsupported_burst_end_does_not_preempt_bounded_idle_work() {
        let now = Instant::now() + DAILY_INTERVAL + Duration::from_secs(1);
        let mut snapshot = HashMap::new();
        snapshot.insert(
            "zeta".to_string(),
            vec![
                now - Duration::from_secs(1_000),
                now - Duration::from_secs(900),
                now - Duration::from_secs(800),
            ],
        );
        snapshot.insert(
            "alpha".to_string(),
            vec![
                now - Duration::from_secs(1_100),
                now - Duration::from_secs(1_000),
                now - Duration::from_secs(900),
            ],
        );

        let selected = select_due_automatic_trigger(
            now,
            &snapshot,
            MaintenanceAdmission::None,
            now - AUTOMATIC_BATCH_IDLE_THRESHOLD,
            false,
            now - DAILY_INTERVAL - Duration::from_secs(1),
            now - BACKSTOP_INTERVAL - Duration::from_secs(1),
        );

        assert_eq!(selected, Some(AutomaticTrigger::Idle));
    }

    #[test]
    fn unsupported_mature_burst_is_drained_without_a_thermal_turn() {
        let now = Instant::now() + DAILY_INTERVAL + Duration::from_secs(1);
        let writes = WriteSignal::new();
        for offset in [1_100, 1_000, 900] {
            writes.record_at("alpha", now - Duration::from_secs(offset));
        }

        assert_eq!(drain_expired_unactionable_bursts(&writes, now), 3);
        assert!(writes.snapshot().is_empty());
    }

    #[test]
    fn automatic_trigger_priority_leaves_later_due_work_for_future_turns() {
        let now = Instant::now() + DAILY_INTERVAL + Duration::from_secs(1);
        let snapshot = HashMap::new();

        assert_eq!(
            select_due_automatic_trigger(
                now,
                &snapshot,
                MaintenanceAdmission::None,
                now - AUTOMATIC_BATCH_IDLE_THRESHOLD,
                false,
                now - DAILY_INTERVAL - Duration::from_secs(1),
                now - BACKSTOP_INTERVAL - Duration::from_secs(1),
            ),
            Some(AutomaticTrigger::Idle)
        );
        assert_eq!(
            select_due_automatic_trigger(
                now,
                &snapshot,
                MaintenanceAdmission::None,
                now - AUTOMATIC_BATCH_IDLE_THRESHOLD,
                true,
                now - DAILY_INTERVAL - Duration::from_secs(1),
                now - BACKSTOP_INTERVAL - Duration::from_secs(1),
            ),
            Some(AutomaticTrigger::Backstop)
        );
    }

    #[test]
    fn pending_maintenance_yields_to_due_steep_after_one_stage() {
        let now = Instant::now() + DAILY_INTERVAL + Duration::from_secs(1);
        let mut snapshot = HashMap::new();
        snapshot.insert(
            "busy-agent".to_string(),
            vec![
                now - Duration::from_secs(1_100),
                now - Duration::from_secs(1_000),
                now - Duration::from_secs(900),
            ],
        );

        assert_eq!(
            select_due_automatic_trigger(
                now,
                &snapshot,
                MaintenanceAdmission::Ready,
                now - AUTOMATIC_BATCH_IDLE_THRESHOLD,
                false,
                now - DAILY_INTERVAL - Duration::from_secs(1),
                now - BACKSTOP_INTERVAL - Duration::from_secs(1),
            ),
            Some(AutomaticTrigger::Maintenance)
        );

        assert_eq!(
            select_due_automatic_trigger(
                now,
                &snapshot,
                MaintenanceAdmission::YieldToDueSteep,
                now - AUTOMATIC_BATCH_IDLE_THRESHOLD,
                false,
                now - DAILY_INTERVAL - Duration::from_secs(1),
                now - BACKSTOP_INTERVAL - Duration::from_secs(1),
            ),
            Some(AutomaticTrigger::Idle)
        );

        let no_bursts = HashMap::new();
        assert_eq!(
            select_due_automatic_trigger(
                now,
                &no_bursts,
                MaintenanceAdmission::YieldToDueSteep,
                now - AUTOMATIC_BATCH_IDLE_THRESHOLD,
                true,
                now - DAILY_INTERVAL - Duration::from_secs(1),
                now - BACKSTOP_INTERVAL - Duration::from_secs(1),
            ),
            Some(AutomaticTrigger::Backstop)
        );
        assert_eq!(
            select_due_automatic_trigger(
                now,
                &no_bursts,
                MaintenanceAdmission::YieldToDueSteep,
                now - AUTOMATIC_BATCH_IDLE_THRESHOLD,
                true,
                now,
                now - BACKSTOP_INTERVAL - Duration::from_secs(1),
            ),
            Some(AutomaticTrigger::Backstop)
        );
    }

    #[test]
    fn idle_and_backstop_enqueue_a_separate_maintenance_turn() {
        assert!(queues_maintenance_followup(&AutomaticTrigger::Idle));
        assert!(queues_maintenance_followup(&AutomaticTrigger::Backstop));
        assert!(!queues_maintenance_followup(&AutomaticTrigger::Daily));
        assert!(!queues_maintenance_followup(&AutomaticTrigger::Maintenance));
    }

    #[tokio::test]
    async fn scheduler_shutdown_interrupts_initial_delay() {
        let shared = Arc::new(tokio::sync::RwLock::new(
            crate::state::ServerState::default(),
        ));
        let writes = WriteSignal::new();
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let task = spawn_scheduler(shared, writes, shutdown_rx);

        shutdown_tx.send_replace(true);
        tokio::time::timeout(Duration::from_millis(250), task)
            .await
            .expect("shutdown must interrupt the scheduler's 60-second initial delay")
            .expect("scheduler task must exit cleanly");
    }

    #[test]
    fn ambient_thermal_work_completion_starts_conservative_cooldown() {
        let now = Instant::now();
        let mut schedule = AmbientSchedule::new(now);
        schedule.note_thermal_work_completion(
            now,
            Duration::from_secs(1),
            ThermalPolicy::conservative(),
        );
        assert_eq!(schedule.next_allowed_at, now + Duration::from_secs(120));
    }

    #[test]
    fn empty_automatic_scan_does_not_consume_a_thermal_turn() {
        assert!(
            !automatic_work_consumes_thermal_turn(false, 0, false),
            "an empty bounded sweep must not delay the first useful ambient item by ten minutes"
        );
    }

    #[test]
    fn selected_inference_or_panic_consumes_an_automatic_thermal_turn() {
        assert!(automatic_work_consumes_thermal_turn(true, 0, false));
        assert!(automatic_work_consumes_thermal_turn(false, 1, false));
        assert!(automatic_work_consumes_thermal_turn(false, 0, true));
    }

    #[tokio::test]
    async fn ambient_provider_hard_caps_forwarded_inference_calls() {
        use wenlan_core::llm_provider::{LlmProvider, LlmRequest};

        let inner = Arc::new(MaintenanceTestProvider {
            body: "response".to_string(),
        });
        let provider = AmbientBudgetProvider::new(inner.clone());
        let request = || LlmRequest {
            system_prompt: None,
            user_prompt: "test".to_string(),
            max_tokens: 8,
            temperature: 0.0,
            label: Some("ambient_budget_test".to_string()),
            timeout_secs: None,
        };

        assert!(provider.generate(request()).await.is_ok());
        assert!(
            provider.generate(request()).await.is_err(),
            "a second inference in one ambient slice must fail closed"
        );
        assert_eq!(provider.call_count(), 1, "telemetry counts forwarded calls");
    }

    #[tokio::test]
    async fn automatic_provider_roles_share_one_poll_inference_budget() {
        use std::sync::atomic::AtomicUsize;
        use wenlan_core::llm_provider::{LlmProvider, LlmRequest};

        let calls = Arc::new(AtomicUsize::new(0));
        let inner = Arc::new(MaintenanceTestProvider {
            body: "response".to_string(),
        });
        let local = AmbientBudgetProvider::with_shared_calls(inner.clone(), calls.clone());
        let synthesis = AmbientBudgetProvider::with_shared_calls(inner.clone(), calls);
        let request = || LlmRequest {
            system_prompt: None,
            user_prompt: "test".to_string(),
            max_tokens: 8,
            temperature: 0.0,
            label: Some("automatic_budget_test".to_string()),
            timeout_secs: None,
        };

        assert!(local.generate(request()).await.is_ok());
        assert!(
            synthesis.generate(request()).await.is_err(),
            "provider roles in one automatic turn must share one inference cap"
        );
        assert_eq!(local.call_count(), 1);
        assert_eq!(synthesis.call_count(), 1);
    }

    #[test]
    fn test_adaptive_gap_empty_returns_ceiling() {
        assert_eq!(adaptive_gap(&[]), BURST_GAP_CEILING);
    }

    #[tokio::test]
    async fn derived_receipt_sweep_dispatches_initially_then_every_thirty_minutes() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let now = Instant::now();
        let mut last = None;
        let calls = AtomicUsize::new(0);
        assert!(run_derived_receipt_sweep_if_due(&mut last, now, || async {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok::<(), ()>(())
        })
        .await
        .unwrap());
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        assert!(!run_derived_receipt_sweep_if_due(
            &mut last,
            now + DERIVED_RECEIPT_SWEEP_INTERVAL - Duration::from_secs(1),
            || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok::<(), ()>(())
            },
        )
        .await
        .unwrap());
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        assert!(run_derived_receipt_sweep_if_due(
            &mut last,
            now + DERIVED_RECEIPT_SWEEP_INTERVAL,
            || async {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok::<(), ()>(())
            },
        )
        .await
        .unwrap());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_adaptive_gap_single_write_returns_ceiling() {
        assert_eq!(adaptive_gap(&[Instant::now()]), BURST_GAP_CEILING);
    }

    #[test]
    fn idle_not_due_until_full_threshold_after_restart() {
        let started = Instant::now();

        assert!(!idle_due(
            false,
            started,
            started + INITIAL_DELAY + POLL_INTERVAL
        ));
        assert!(!idle_due(
            false,
            started,
            started + AUTOMATIC_BATCH_IDLE_THRESHOLD - Duration::from_millis(1)
        ));
        assert!(idle_due(
            false,
            started,
            started + AUTOMATIC_BATCH_IDLE_THRESHOLD
        ));
        assert!(!idle_due(
            true,
            started,
            started + AUTOMATIC_BATCH_IDLE_THRESHOLD
        ));
    }

    #[test]
    fn test_adaptive_gap_fast_writer() {
        // Writes every 30s → median 30s → 2*30s = 60s → clamped to floor (5 min)
        let base = Instant::now();
        let timestamps: Vec<Instant> = (0..10)
            .map(|i| base + Duration::from_secs(30 * i))
            .collect();
        assert_eq!(adaptive_gap(&timestamps), BURST_GAP_FLOOR);
    }

    #[test]
    fn test_adaptive_gap_slow_writer() {
        // Writes every 10 min → median 600s → 2*600s = 1200s (20 min)
        let base = Instant::now();
        let timestamps: Vec<Instant> = (0..5)
            .map(|i| base + Duration::from_secs(600 * i))
            .collect();
        assert_eq!(adaptive_gap(&timestamps), Duration::from_secs(1200));
    }

    #[test]
    fn test_adaptive_gap_very_slow_writer_capped() {
        // Writes every 20 min → median 1200s → 2*1200s = 2400s → capped at 1800s
        let base = Instant::now();
        let timestamps: Vec<Instant> = (0..3)
            .map(|i| base + Duration::from_secs(1200 * i))
            .collect();
        assert_eq!(adaptive_gap(&timestamps), BURST_GAP_CEILING);
    }

    #[test]
    fn test_adaptive_gap_two_writes() {
        // 2 writes 3 min apart → median 180s → 2*180s = 360s (between floor and ceiling)
        let base = Instant::now();
        let timestamps = vec![base, base + Duration::from_secs(180)];
        assert_eq!(adaptive_gap(&timestamps), Duration::from_secs(360));
    }

    #[test]
    fn test_write_signal_record_and_snapshot() {
        let ws = WriteSignal::new();
        let now = Instant::now();
        ws.record_at("claude", now);
        ws.record_at("claude", now + Duration::from_secs(10));
        ws.record_at("obsidian", now);

        let snap = ws.snapshot();
        assert_eq!(snap.get("claude").unwrap().len(), 2);
        assert_eq!(snap.get("obsidian").unwrap().len(), 1);
    }

    #[test]
    fn test_drain_up_to_preserves_later_writes() {
        let ws = WriteSignal::new();
        let t1 = Instant::now();
        let t2 = t1 + Duration::from_secs(10);
        let t3 = t2 + Duration::from_secs(10);

        ws.record_at("claude", t1);
        ws.record_at("claude", t2);
        ws.record_at("claude", t3);

        // Drain up to t2 — t3 should survive
        let drained = ws.drain_up_to("claude", t2);
        assert_eq!(drained.len(), 2);

        let snap = ws.snapshot();
        let remaining = snap.get("claude").unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0], t3);
    }

    #[test]
    fn test_drain_up_to_removes_key_when_empty() {
        let ws = WriteSignal::new();
        let t1 = Instant::now();
        ws.record_at("claude", t1);

        ws.drain_up_to("claude", t1);
        let snap = ws.snapshot();
        assert!(!snap.contains_key("claude"));
    }

    #[test]
    fn test_has_activity_since() {
        let ws = WriteSignal::new();
        let t1 = Instant::now();
        ws.record_at("claude", t1 + Duration::from_secs(5));

        assert!(ws.has_activity_since(t1));
        assert!(!ws.has_activity_since(t1 + Duration::from_secs(10)));
    }

    #[test]
    fn test_adaptive_gap_irregular_pattern() {
        // Mix of fast and slow writes — median should reflect the middle
        let base = Instant::now();
        // 5 writes: gaps of 10s, 10s, 300s, 10s → sorted intervals: 10, 10, 10, 300
        // median of even count = (10 + 10) / 2 = 10s → 2*10 = 20s → clamped to floor (5 min)
        let timestamps = vec![
            base,
            base + Duration::from_secs(10),
            base + Duration::from_secs(20),
            base + Duration::from_secs(320),
            base + Duration::from_secs(330),
        ];
        assert_eq!(adaptive_gap(&timestamps), BURST_GAP_FLOOR);
    }

    #[test]
    fn test_burst_detection_scenario() {
        // Simulate: 5 writes over 2.5 min, then 6 min silence → should be detected as burst end
        let ws = WriteSignal::new();
        let base = Instant::now();
        for i in 0..5 {
            ws.record_at("claude", base + Duration::from_secs(30 * i));
        }
        // Last write at base + 120s. Adaptive gap = floor (5 min) = 300s.
        // At base + 420s (120 + 300), the burst should be detected as ended.
        let gap = adaptive_gap(&ws.snapshot()["claude"]);
        assert_eq!(gap, BURST_GAP_FLOOR); // 5 min

        let last_write = base + Duration::from_secs(120);
        let check_time = last_write + BURST_GAP_FLOOR + Duration::from_secs(1);
        assert!(check_time.duration_since(last_write) > gap);

        // Verify drain preserves nothing (all writes before cutoff)
        let drained = ws.drain_up_to("claude", last_write);
        assert_eq!(drained.len(), 5);
        assert!(ws.snapshot().is_empty());
    }

    #[test]
    fn test_concurrent_agents_independent() {
        // Two agents writing — draining one doesn't affect the other
        let ws = WriteSignal::new();
        let base = Instant::now();

        for i in 0..5 {
            ws.record_at("claude", base + Duration::from_secs(30 * i));
        }
        for i in 0..3 {
            ws.record_at("obsidian", base + Duration::from_secs(60 * i));
        }

        // Drain claude only
        let cutoff = base + Duration::from_secs(120);
        ws.drain_up_to("claude", cutoff);

        let snap = ws.snapshot();
        assert!(!snap.contains_key("claude"));
        assert_eq!(snap["obsidian"].len(), 3);
    }

    // ── §4 Directory-source sync + enrichment-queue-drive tick ───────────────

    /// Isolate `WENLAN_DATA_DIR` (config lives there) to a tempdir for the
    /// duration of a test; restore the prior value on drop.
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

    fn register_directory_source(id: &str, path: &std::path::Path) {
        wenlan_core::config::save_config(&wenlan_core::config::Config {
            sources: vec![wenlan_types::sources::Source {
                id: id.to_string(),
                source_type: wenlan_types::sources::SourceType::Directory,
                path: path.to_path_buf(),
                status: wenlan_types::sources::SyncStatus::Active,
                last_sync: None,
                file_count: 0,
                memory_count: 0,
                last_sync_errors: 0,
                last_sync_error_detail: None,
            }],
            ..wenlan_core::config::Config::default()
        })
        .unwrap();
    }

    #[test]
    fn directory_sync_tick_polls_recoverable_sources_but_not_paused() {
        let mut source = wenlan_types::sources::Source {
            id: "directory-notes".to_string(),
            source_type: wenlan_types::sources::SourceType::Directory,
            path: std::path::PathBuf::from("/tmp/notes"),
            status: wenlan_types::sources::SyncStatus::Active,
            last_sync: None,
            file_count: 0,
            memory_count: 0,
            last_sync_errors: 0,
            last_sync_error_detail: None,
        };

        assert!(should_poll_directory_source(&source));

        source.status =
            wenlan_types::sources::SyncStatus::Unavailable("filesystem stalled".to_string());
        assert!(should_poll_directory_source(&source));

        source.status =
            wenlan_types::sources::SyncStatus::Error("transient file error".to_string());
        assert!(should_poll_directory_source(&source));

        source.status = wenlan_types::sources::SyncStatus::Paused;
        assert!(!should_poll_directory_source(&source));

        source.status = wenlan_types::sources::SyncStatus::Active;
        source.source_type = wenlan_types::sources::SourceType::Obsidian;
        assert!(!should_poll_directory_source(&source));
    }

    async fn new_test_db() -> (Arc<wenlan_core::db::MemoryDB>, tempfile::TempDir) {
        let db_dir = tempfile::tempdir().unwrap();
        let db = Arc::new(
            wenlan_core::db::MemoryDB::new(
                db_dir.path(),
                Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap(),
        );
        (db, db_dir)
    }

    /// One poll tick over a Directory source with a fresh file must enqueue AND
    /// process it into searchable chunks plus a SOURCE page. With no LLM, the
    /// enrichment route still embeds every chunk (searchable) and writes the
    /// deterministic stub SOURCE page — exactly what the page-watcher Step-0
    /// precedent does for its own cheap per-poll pass.
    #[tokio::test]
    async fn directory_sync_and_document_slice_are_separate_steps() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let source_root = tempfile::tempdir().unwrap();
        let file_path = source_root.path().join("note.txt");
        let mut body =
            String::from("Wenlanborg is the code name for the folder ingestion subsystem.\n\n");
        for i in 0..40 {
            body.push_str(&format!(
                "Paragraph {i} describes the document ingestion pipeline in concrete detail so the \
                 chunker splits it into multiple sections rather than a single chunk.\n\n"
            ));
        }
        std::fs::write(&file_path, &body).unwrap();

        let source_id = "directory-notes".to_string();
        register_directory_source(&source_id, source_root.path());

        let (db, _db_dir) = new_test_db().await;
        let prompts = wenlan_core::prompts::PromptRegistry::default();

        sync_directory_sources(&db).await;
        let queued = db
            .get_queue_entry(&source_id, &file_path.to_string_lossy())
            .await
            .unwrap()
            .expect("sync enqueues the file");
        assert_eq!(queued.status, "pending");

        let processed = run_document_enrichment_slice_tick(&db, None, &prompts).await;
        assert_eq!(processed, 1, "the one new file is claimed and processed");

        // The file's chunks are stored + searchable.
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
        let doc_source_id = wenlan_core::sources::directory::document_source_id(
            &source_id,
            &file_path,
            Some(&knowledge_path),
        );
        let chunks = db
            .get_memories_by_source_id("memory", &doc_source_id)
            .await
            .unwrap();
        assert!(
            !chunks.is_empty(),
            "the new file must produce stored chunks"
        );

        let results = db
            .search_memory(
                "Wenlanborg",
                30,
                None,
                &wenlan_core::read_scope::ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(
            results.iter().any(|r| r.source_id == doc_source_id),
            "the new file's chunks must be searchable"
        );

        // A SOURCE page was written for the document.
        let pages = db.list_pages("active", 100, 0).await.unwrap();
        assert!(
            pages.iter().any(|p| p.creation_kind == "source"),
            "a source page must be written for the document"
        );

        // Searchable preparation is durable, while model enrichment waits for
        // an explicit provider pin instead of being falsely marked complete.
        let q = db
            .get_queue_entry(&source_id, &file_path.to_string_lossy())
            .await
            .unwrap()
            .expect("queue entry exists after processing");
        assert_eq!(q.status, "waiting_for_provider");
    }

    /// A paused queue row whose backoff has not elapsed must be SKIPPED by the
    /// tick (backoff auto-resume): `claim_next_pending` never returns it, so it
    /// is not processed and no chunks materialize.
    #[tokio::test]
    async fn directory_sync_tick_skips_paused_queue_with_future_retry() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        // Registered but empty source: sync finds nothing to enqueue.
        let source_root = tempfile::tempdir().unwrap();
        let source_id = "directory-notes".to_string();
        register_directory_source(&source_id, source_root.path());

        let (db, _db_dir) = new_test_db().await;
        let prompts = wenlan_core::prompts::PromptRegistry::default();

        // A paused document whose retry is an hour out.
        let paused_path = source_root
            .path()
            .join("paused.txt")
            .to_string_lossy()
            .to_string();
        db.enqueue_document(&source_id, &paused_path, Some("hash-paused"))
            .await
            .unwrap();
        let future_retry = chrono::Utc::now().timestamp() + 3600;
        db.mark_paused(
            &source_id,
            &paused_path,
            "analysis LLM failed",
            Some(future_retry),
        )
        .await
        .unwrap();

        sync_directory_sources(&db).await;
        let processed = run_document_enrichment_slice_tick(&db, None, &prompts).await;
        assert_eq!(processed, 0, "a paused row with a future retry is skipped");

        let q = db
            .get_queue_entry(&source_id, &paused_path)
            .await
            .unwrap()
            .expect("paused entry remains");
        assert_eq!(q.status, "paused", "the row is still paused, not processed");

        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
        let doc_source_id = wenlan_core::sources::directory::document_source_id(
            &source_id,
            std::path::Path::new(&paused_path),
            Some(&knowledge_path),
        );
        let chunks = db
            .get_memories_by_source_id("memory", &doc_source_id)
            .await
            .unwrap();
        assert!(
            chunks.is_empty(),
            "a skipped paused document must not be processed into chunks"
        );
    }

    #[tokio::test]
    async fn document_provider_panic_pauses_claimed_generation() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();

        let source_root = tempfile::tempdir().unwrap();
        let file_path = source_root.path().join("panic-note.txt");
        std::fs::write(
            &file_path,
            "Wenlan should preserve and retry a claimed document after a provider panic.",
        )
        .unwrap();
        let source_id = "directory-panic";
        register_directory_source(source_id, source_root.path());

        let (db, _db_dir) = new_test_db().await;
        sync_directory_sources(&db).await;
        let before = db
            .get_queue_entry(source_id, &file_path.to_string_lossy())
            .await
            .unwrap()
            .expect("directory sync enqueues the document");
        assert_eq!(before.status, "pending");

        let panicking: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(PanicTestProvider);
        let report = run_ambient_job_safe(
            AmbientJob::Document,
            &db,
            Some(&panicking),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;

        assert!(
            report.panicked,
            "panic remains visible to scheduler accounting"
        );
        let after = db
            .get_queue_entry(source_id, &file_path.to_string_lossy())
            .await
            .unwrap()
            .expect("claimed generation remains queued");
        assert_eq!(after.status, "paused");
        assert_eq!(after.attempt_count, 1);
        assert!(after.next_retry_at.is_some());
        assert!(after
            .error_detail
            .as_deref()
            .is_some_and(|reason| reason.contains("panicked")));
        assert_eq!(after.last_completed_chunk, before.last_completed_chunk);
    }

    struct MaintenanceTestProvider {
        body: String,
    }

    #[async_trait::async_trait]
    impl wenlan_core::llm_provider::LlmProvider for MaintenanceTestProvider {
        async fn generate(
            &self,
            _request: wenlan_core::llm_provider::LlmRequest,
        ) -> Result<String, wenlan_core::llm_provider::LlmError> {
            Ok(self.body.clone())
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "maintenance-test"
        }

        fn backend(&self) -> wenlan_core::llm_provider::LlmBackend {
            wenlan_core::llm_provider::LlmBackend::Api
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    struct AvailabilityTestProvider {
        name: &'static str,
        available: bool,
    }

    #[async_trait::async_trait]
    impl wenlan_core::llm_provider::LlmProvider for AvailabilityTestProvider {
        async fn generate(
            &self,
            _request: wenlan_core::llm_provider::LlmRequest,
        ) -> Result<String, wenlan_core::llm_provider::LlmError> {
            Ok(self.name.to_string())
        }

        fn is_available(&self) -> bool {
            self.available
        }

        fn name(&self) -> &str {
            self.name
        }

        fn backend(&self) -> wenlan_core::llm_provider::LlmBackend {
            wenlan_core::llm_provider::LlmBackend::Api
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    struct PanicTestProvider;

    #[async_trait::async_trait]
    impl wenlan_core::llm_provider::LlmProvider for PanicTestProvider {
        async fn generate(
            &self,
            _request: wenlan_core::llm_provider::LlmRequest,
        ) -> Result<String, wenlan_core::llm_provider::LlmError> {
            panic!("ambient provider panic")
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "panic-test"
        }

        fn backend(&self) -> wenlan_core::llm_provider::LlmBackend {
            wenlan_core::llm_provider::LlmBackend::Api
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    #[test]
    fn ambient_provider_selection_honors_explicit_on_device_pin() {
        let api: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(AvailabilityTestProvider {
                name: "available-api",
                available: true,
            });
        let local: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(AvailabilityTestProvider {
                name: "available-local",
                available: true,
            });

        let selected = resolve_ambient_provider(
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            Some(&api),
            None,
            Some(&local),
        )
        .expect("the on-device pin should select the exact approved source");

        assert_eq!(selected.name(), "available-local");
    }

    #[test]
    fn maintenance_provider_selection_requires_explicit_pin() {
        let api: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(AvailabilityTestProvider {
                name: "available-api",
                available: true,
            });
        let external: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(AvailabilityTestProvider {
                name: "available-external",
                available: true,
            });

        assert!(
            resolve_maintenance_provider(None, None, Some(&api), Some(&external), None).is_none()
        );
    }

    #[test]
    fn maintenance_provider_selection_does_not_fallback_from_missing_pin() {
        let api: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(AvailabilityTestProvider {
                name: "available-api",
                available: true,
            });

        assert!(resolve_maintenance_provider(
            Some(wenlan_core::refinery::SynthesisSource::External),
            None,
            Some(&api),
            None,
            None,
        )
        .is_none());
    }

    #[test]
    fn maintenance_provider_selection_honors_exact_external_pin() {
        let api: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(AvailabilityTestProvider {
                name: "available-api",
                available: true,
            });
        let external: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(AvailabilityTestProvider {
                name: "available-external",
                available: true,
            });

        let selected = resolve_maintenance_provider(
            Some(wenlan_core::refinery::SynthesisSource::External),
            None,
            Some(&api),
            Some(&external),
            None,
        )
        .expect("the explicit external pin should select the external slot");

        assert_eq!(selected.name(), "available-external");
    }

    async fn store_test_memory(db: &wenlan_core::db::MemoryDB, id: &str, content: &str) {
        db.upsert_documents(vec![wenlan_types::RawDocument {
            source: "memory".to_string(),
            source_id: id.to_string(),
            title: id.to_string(),
            content: content.to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".to_string()),
            source_agent: Some("test".to_string()),
            confirmed: Some(true),
            ..Default::default()
        }])
        .await
        .unwrap();
    }

    fn rb01_parse_calibration_load_duties(value: Option<&str>) -> Result<Vec<u8>, String> {
        let Some(value) = value else {
            return Ok(Vec::new());
        };
        if value.trim().is_empty() {
            return Ok(Vec::new());
        }
        let duties = value
            .split(',')
            .map(|part| {
                let duty = part
                    .trim()
                    .parse::<u8>()
                    .map_err(|error| format!("invalid calibration duty {part:?}: {error}"))?;
                if !(1..=100).contains(&duty) {
                    return Err(format!("calibration duty must be 1..=100, got {duty}"));
                }
                Ok(duty)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let total_duty = duties.iter().map(|duty| u16::from(*duty)).sum::<u16>();
        if total_duty > 300 {
            return Err(format!(
                "calibration load is capped at three CPU cores, got {total_duty}%"
            ));
        }
        Ok(duties)
    }

    fn rb01_parse_calibration_cpu_band(value: Option<&str>) -> Result<Option<(f32, f32)>, String> {
        let Some(value) = value else {
            return Ok(None);
        };
        let (minimum, maximum) = value
            .split_once(':')
            .ok_or_else(|| format!("calibration CPU band must be min:max, got {value:?}"))?;
        let minimum = minimum
            .parse::<f32>()
            .map_err(|error| format!("invalid calibration CPU minimum: {error}"))?;
        let maximum = maximum
            .parse::<f32>()
            .map_err(|error| format!("invalid calibration CPU maximum: {error}"))?;
        if !minimum.is_finite() || !maximum.is_finite() || minimum < 0.0 || minimum >= maximum {
            return Err(format!(
                "calibration CPU band must be finite and increasing, got {minimum}:{maximum}"
            ));
        }
        Ok(Some((minimum, maximum)))
    }

    fn rb01_percentile_us(samples: &[u64], percentile: usize) -> u64 {
        assert!(!samples.is_empty(), "latency percentile needs samples");
        assert!((1..=100).contains(&percentile));
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let rank = sorted.len().saturating_mul(percentile).div_ceil(100);
        sorted[rank.saturating_sub(1)]
    }

    struct Rb01SyntheticLoad {
        stop: Arc<std::sync::atomic::AtomicBool>,
        workers: Vec<std::thread::JoinHandle<()>>,
    }

    impl Rb01SyntheticLoad {
        fn start(duties: &[u8]) -> Self {
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let workers = duties
                .iter()
                .copied()
                .enumerate()
                .map(|(index, duty)| {
                    let stop = stop.clone();
                    std::thread::Builder::new()
                        .name(format!("rb01-calibration-load-{index}"))
                        .spawn(move || {
                            let window = Duration::from_millis(20);
                            let busy = window.mul_f64(f64::from(duty) / 100.0);
                            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                                let window_started = Instant::now();
                                while window_started.elapsed() < busy
                                    && !stop.load(std::sync::atomic::Ordering::Relaxed)
                                {
                                    std::hint::spin_loop();
                                }
                                let remaining = window.saturating_sub(window_started.elapsed());
                                if !remaining.is_zero() {
                                    std::thread::sleep(remaining);
                                }
                            }
                        })
                        .expect("spawn bounded RB-01 calibration load")
                })
                .collect();
            Self { stop, workers }
        }
    }

    impl Drop for Rb01SyntheticLoad {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            for worker in self.workers.drain(..) {
                worker.join().expect("RB-01 calibration load stops");
            }
        }
    }

    fn rb01_spawn_latency_probe() -> (
        Arc<std::sync::atomic::AtomicBool>,
        std::thread::JoinHandle<Vec<u64>>,
    ) {
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_stop = stop.clone();
        let worker = std::thread::Builder::new()
            .name("rb01-foreground-latency".to_string())
            .spawn(move || {
                let interval = Duration::from_millis(5);
                let mut next = Instant::now() + interval;
                let mut overshoot_us = Vec::new();
                while !worker_stop.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(next.saturating_duration_since(Instant::now()));
                    let observed = Instant::now();
                    overshoot_us.push(
                        observed
                            .saturating_duration_since(next)
                            .as_micros()
                            .try_into()
                            .unwrap_or(u64::MAX),
                    );
                    next += interval;
                    if next <= observed {
                        next = observed + interval;
                    }
                }
                overshoot_us
            })
            .expect("spawn RB-01 foreground latency probe");
        (stop, worker)
    }

    fn rb01_finish_latency_probe(
        stop: Arc<std::sync::atomic::AtomicBool>,
        worker: std::thread::JoinHandle<Vec<u64>>,
    ) -> Vec<u64> {
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        worker.join().expect("RB-01 foreground latency probe stops")
    }

    fn rb01_latency_summary(samples: &[u64]) -> serde_json::Value {
        serde_json::json!({
            "samples": samples.len(),
            "p50_us": rb01_percentile_us(samples, 50),
            "p95_us": rb01_percentile_us(samples, 95),
            "p99_us": rb01_percentile_us(samples, 99),
            "max_us": samples.iter().copied().max().unwrap_or(0)
        })
    }

    async fn rb01_sample_runtime_until(
        started: Instant,
        stop: Arc<std::sync::atomic::AtomicBool>,
    ) -> Vec<(u64, f32, u64)> {
        let mut system = sysinfo::System::new_all();
        system.refresh_cpu_usage();
        let mut samples = Vec::new();
        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(250)).await;
            system.refresh_cpu_usage();
            system.refresh_memory();
            samples.push((
                started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                system.global_cpu_usage(),
                system.available_memory(),
            ));
        }
        samples
    }

    #[derive(Debug, Clone, Copy)]
    enum Rb01ProfileLane {
        Document,
        Entity,
        PageGrowth,
        Reconcile,
        Citation,
    }

    #[test]
    fn rb01_calibration_load_parser_is_bounded() {
        assert_eq!(
            rb01_parse_calibration_load_duties(Some("100,50")).unwrap(),
            vec![100, 50]
        );
        assert_eq!(
            rb01_parse_calibration_load_duties(Some("100,100,50")).unwrap(),
            vec![100, 100, 50]
        );
        assert_eq!(
            rb01_parse_calibration_load_duties(None).unwrap(),
            Vec::<u8>::new()
        );
        assert!(rb01_parse_calibration_load_duties(Some("0")).is_err());
        assert!(rb01_parse_calibration_load_duties(Some("101")).is_err());
        assert!(rb01_parse_calibration_load_duties(Some("100,100,100,1")).is_err());
    }

    #[test]
    fn rb01_calibration_cpu_band_parser_rejects_inverted_bounds() {
        assert_eq!(
            rb01_parse_calibration_cpu_band(Some("20.1:26.0")).unwrap(),
            Some((20.1, 26.0))
        );
        assert_eq!(rb01_parse_calibration_cpu_band(None).unwrap(), None);
        assert!(rb01_parse_calibration_cpu_band(Some("26:20")).is_err());
        assert!(rb01_parse_calibration_cpu_band(Some("bad")).is_err());
    }

    #[test]
    fn rb01_calibration_recheck_is_fast_but_still_requires_two_samples() {
        assert_eq!(
            rb01_profile_admission_timing(false),
            (POLL_INTERVAL, RB01_PROFILE_ADMISSION_MAX_SAMPLES)
        );
        assert_eq!(
            rb01_profile_admission_timing(true),
            (Duration::from_secs(2), 4)
        );
    }

    #[test]
    fn rb01_latency_percentile_uses_nearest_rank() {
        let samples = (1..=100).collect::<Vec<u64>>();
        assert_eq!(rb01_percentile_us(&samples, 50), 50);
        assert_eq!(rb01_percentile_us(&samples, 95), 95);
        assert_eq!(rb01_percentile_us(&samples, 99), 99);
        assert_eq!(rb01_percentile_us(&[7], 99), 7);
    }

    impl Rb01ProfileLane {
        fn from_env(value: &str) -> Option<Self> {
            match value {
                "document" => Some(Self::Document),
                "entity" => Some(Self::Entity),
                "page-growth" => Some(Self::PageGrowth),
                "reconcile" => Some(Self::Reconcile),
                "citation" => Some(Self::Citation),
                _ => None,
            }
        }

        const fn job(self) -> AmbientJob {
            match self {
                Self::Document => AmbientJob::Document,
                Self::Entity => AmbientJob::Entity,
                Self::PageGrowth => AmbientJob::PageGrowth,
                Self::Reconcile => AmbientJob::Reconcile,
                Self::Citation => AmbientJob::Citation,
            }
        }

        const fn as_str(self) -> &'static str {
            match self {
                Self::Document => "document",
                Self::Entity => "entity",
                Self::PageGrowth => "page-growth",
                Self::Reconcile => "reconcile",
                Self::Citation => "citation",
            }
        }
    }

    struct Rb01ProfileFixture {
        _source_dir: Option<tempfile::TempDir>,
        document_path: Option<String>,
    }

    fn rb01_macos_thermal_state() -> Option<u8> {
        if !cfg!(target_os = "macos") {
            return None;
        }
        let output = if let Some(helper) = std::env::var_os("WENLAN_RB01_THERMAL_HELPER") {
            std::process::Command::new(helper).output().ok()?
        } else {
            std::process::Command::new("/usr/bin/swift")
                .args([
                    "-e",
                    "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
                ])
                .output()
                .ok()?
        };
        if !output.status.success() {
            return None;
        }
        std::str::from_utf8(&output.stdout)
            .ok()?
            .trim()
            .parse()
            .ok()
    }

    async fn rb01_sample_peak_rss(
        pid: sysinfo::Pid,
        baseline_bytes: u64,
        stop: Arc<std::sync::atomic::AtomicBool>,
    ) -> u64 {
        let mut system = sysinfo::System::new();
        let mut peak_bytes = baseline_bytes;
        loop {
            system.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                false,
                sysinfo::ProcessRefreshKind::nothing().with_memory(),
            );
            if let Some(process) = system.process(pid) {
                peak_bytes = peak_bytes.max(process.memory());
            }
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                return peak_bytes;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    fn rb01_profile_admission_timing(calibration: bool) -> (Duration, usize) {
        if calibration {
            (Duration::from_secs(2), 4)
        } else {
            (POLL_INTERVAL, RB01_PROFILE_ADMISSION_MAX_SAMPLES)
        }
    }

    async fn rb01_wait_for_profile_admission(
        calibration: bool,
    ) -> Result<ResourceSnapshot, String> {
        let policy = ResourcePolicy::conservative()
            .with_additional_memory_headroom(ON_DEVICE_INFERENCE_HEADROOM_BYTES);
        let mut probe = SystemResourceProbe::new(Instant::now());
        let mut admission = Rb01ProfileAdmission::default();
        let (sample_interval, max_samples) = rb01_profile_admission_timing(calibration);

        for sample_index in 1..=max_samples {
            tokio::time::sleep(sample_interval).await;
            let status = probe.sample(Instant::now(), policy);
            let thermal = rb01_macos_thermal_state();
            match rb01_profile_sample_action(
                &mut admission,
                status.snapshot,
                thermal,
                policy,
                sample_index,
                max_samples,
            ) {
                Rb01ProfileSampleAction::Retry => {}
                Rb01ProfileSampleAction::Admit => {
                    return status.snapshot.ok_or_else(|| {
                        "profile preflight admitted without a resource snapshot".to_string()
                    });
                }
                Rb01ProfileSampleAction::Fail(reason) => {
                    return Err(format!(
                        "profile preflight sample {sample_index}/{max_samples} \
                         rejected: {reason:?}; \
                         resource={:?}; thermal_state={thermal:?}",
                        status.snapshot
                    ));
                }
            }
        }

        Err("profile preflight exhausted its bounded admission samples".to_string())
    }

    async fn rb01_seed_profile_lane(
        lane: Rb01ProfileLane,
        db: &Arc<wenlan_core::db::MemoryDB>,
    ) -> Rb01ProfileFixture {
        match lane {
            Rb01ProfileLane::Document => {
                use sha2::{Digest, Sha256};

                let source_dir = tempfile::tempdir().unwrap();
                let path = source_dir.path().join("rb01-document.txt");
                let mut body =
                    String::from("Wenlan profiles one bounded document slice at a time.\n\n");
                for index in 0..80 {
                    body.push_str(&format!(
                        "Paragraph {index} explains a separate scheduler invariant with enough \
                         concrete prose to force multiple document chunks during the profile.\n\n"
                    ));
                }
                std::fs::write(&path, body.as_bytes()).unwrap();
                let content_hash = format!("{:x}", Sha256::digest(body.as_bytes()));
                db.enqueue_document(
                    "rb01-directory",
                    &path.to_string_lossy(),
                    Some(&content_hash),
                )
                .await
                .unwrap();
                Rb01ProfileFixture {
                    document_path: Some(path.to_string_lossy().to_string()),
                    _source_dir: Some(source_dir),
                }
            }
            Rb01ProfileLane::Entity => {
                store_test_memory(
                    db,
                    "rb01-entity-memory",
                    "Project Juniper uses Wenlan to keep its scheduler decisions durable.",
                )
                .await;
                Rb01ProfileFixture {
                    _source_dir: None,
                    document_path: None,
                }
            }
            Rb01ProfileLane::PageGrowth => {
                store_test_memory(
                    db,
                    "rb01-page-growth-memory",
                    "A Page Growth no-match slice measures bounded embedding and search work.",
                )
                .await;
                assert!(
                    db.record_enrichment_step_at_version(
                        "rb01-page-growth-memory",
                        "entity_extract",
                        "skipped",
                        None,
                        1,
                    )
                    .await
                    .unwrap(),
                    "Page Growth fixture must satisfy the versioned entity dependency"
                );
                Rb01ProfileFixture {
                    _source_dir: None,
                    document_path: None,
                }
            }
            Rb01ProfileLane::Reconcile => {
                let common = "The Wenlan daemon binds to port 7878 and stores memory locally.";
                db.upsert_documents(vec![
                    wenlan_types::RawDocument {
                        source: "memory".to_string(),
                        source_id: "rb01-doc-a".to_string(),
                        title: "rb01-doc-a".to_string(),
                        content: common.to_string(),
                        last_modified: 1,
                        confirmed: Some(true),
                        source_agent: Some("folder".to_string()),
                        content_hash: Some("rb01-hash-a".to_string()),
                        ..Default::default()
                    },
                    wenlan_types::RawDocument {
                        source: "memory".to_string(),
                        source_id: "rb01-doc-b".to_string(),
                        title: "rb01-doc-b".to_string(),
                        content: common.to_string(),
                        last_modified: 2,
                        confirmed: Some(true),
                        source_agent: Some("folder".to_string()),
                        content_hash: Some("rb01-hash-b".to_string()),
                        ..Default::default()
                    },
                    wenlan_types::RawDocument {
                        source: "memory".to_string(),
                        source_id: "rb01-capture".to_string(),
                        title: "rb01-capture".to_string(),
                        content: common.to_string(),
                        last_modified: 3,
                        confirmed: Some(true),
                        source_agent: Some("claude-code".to_string()),
                        ..Default::default()
                    },
                ])
                .await
                .unwrap();
                Rb01ProfileFixture {
                    _source_dir: None,
                    document_path: None,
                }
            }
            Rb01ProfileLane::Citation => {
                store_test_memory(
                    db,
                    "rb01-citation-source",
                    "The Wenlan daemon binds to port 7878 by default.",
                )
                .await;
                insert_test_page(
                    db,
                    "Wenlan daemon port",
                    "The Wenlan daemon binds to port 7878 by default.",
                    &["rb01-citation-source"],
                    "distilled",
                )
                .await;
                Rb01ProfileFixture {
                    _source_dir: None,
                    document_path: None,
                }
            }
        }
    }

    async fn rb01_lane_progressed(
        lane: Rb01ProfileLane,
        db: &Arc<wenlan_core::db::MemoryDB>,
        fixture: &Rb01ProfileFixture,
    ) -> bool {
        match lane {
            Rb01ProfileLane::Document => match fixture.document_path.as_deref() {
                Some(path) => db
                    .get_queue_entry("rb01-directory", path)
                    .await
                    .ok()
                    .flatten()
                    .is_some_and(|entry| entry.last_completed_chunk >= 0 || entry.status == "done"),
                None => false,
            },
            Rb01ProfileLane::Entity => db
                .get_enrichment_steps("rb01-entity-memory")
                .await
                .unwrap_or_default()
                .iter()
                .any(|step| step.step == "entity_extract"),
            Rb01ProfileLane::PageGrowth => db
                .get_enrichment_steps("rb01-page-growth-memory")
                .await
                .unwrap_or_default()
                .iter()
                .any(|step| step.step == "page_growth"),
            Rb01ProfileLane::Reconcile => {
                db.get_app_metadata("reconcile_frontier_docs")
                    .await
                    .ok()
                    .flatten()
                    .is_some()
                    || db
                        .get_app_metadata("reconcile_frontier_captures")
                        .await
                        .ok()
                        .flatten()
                        .is_some()
            }
            Rb01ProfileLane::Citation => db
                .get_pages_missing_citations(10)
                .await
                .ok()
                .is_some_and(|pages| pages.is_empty()),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "manual RB-01 target-Mac profile; cached qwen3-4b and explicit WENLAN_RB01_PROFILE=1 required"]
    async fn rb01_profile_real_on_device_slice() {
        assert!(
            rb01_profile_requested(std::env::var("WENLAN_RB01_PROFILE").ok().as_deref()),
            "refusing real-model profile without WENLAN_RB01_PROFILE=1"
        );
        if std::env::consts::OS != "macos" {
            panic!("RB-01 real-model profile currently targets supported macOS hardware");
        }
        let lane_value = std::env::var("WENLAN_RB01_LANE")
            .expect("set WENLAN_RB01_LANE=document|entity|page-growth|reconcile|citation");
        let lane = Rb01ProfileLane::from_env(&lane_value)
            .expect("WENLAN_RB01_LANE must be document|entity|page-growth|reconcile|citation");
        let calibration_duties = rb01_parse_calibration_load_duties(
            std::env::var("WENLAN_RB01_CALIBRATION_LOAD_DUTIES")
                .ok()
                .as_deref(),
        )
        .expect("valid WENLAN_RB01_CALIBRATION_LOAD_DUTIES");
        let calibration_cpu_band = rb01_parse_calibration_cpu_band(
            std::env::var("WENLAN_RB01_CALIBRATION_CPU_BAND")
                .ok()
                .as_deref(),
        )
        .expect("valid WENLAN_RB01_CALIBRATION_CPU_BAND");
        assert_eq!(
            calibration_duties.is_empty(),
            calibration_cpu_band.is_none(),
            "calibration load duties and CPU band must be supplied together"
        );
        let model = wenlan_core::on_device_models::get_model("qwen3-4b")
            .expect("qwen3-4b remains in the on-device registry");
        assert!(
            wenlan_core::on_device_models::is_cached(model),
            "refusing to download a model during RB-01; qwen3-4b is not cached"
        );

        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let (db, _db_dir) = new_test_db().await;
        let fixture = rb01_seed_profile_lane(lane, &db).await;

        let mut process_system = sysinfo::System::new_all();
        let pid = sysinfo::get_current_pid().expect("current process id");
        let rss_process_baseline = process_system
            .process(pid)
            .map_or(0, |process| process.memory());
        let available_memory_pre_model = process_system.available_memory();
        let boot_started = Instant::now();
        let provider: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
            wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3-4b"))
                .expect("cached qwen3-4b provider must initialize"),
        );
        let boot_ms = boot_started.elapsed().as_millis();
        process_system.refresh_all();
        let rss_model_loaded = process_system
            .process(pid)
            .map_or(0, |process| process.memory());
        let available_memory_model_loaded = process_system.available_memory();

        let before = match rb01_wait_for_profile_admission(calibration_cpu_band.is_some()).await {
            Ok(snapshot) => snapshot,
            Err(error) if calibration_cpu_band.is_some() => {
                println!(
                    "{}",
                    serde_json::json!({
                        "event": "rb01_calibration_no_inference",
                        "lane": lane.as_str(),
                        "reason": error,
                        "boot_ms": boot_ms,
                        "thermal_after": rb01_macos_thermal_state(),
                        "report_elapsed_ms": 0
                    })
                );
                return;
            }
            Err(error) => panic!("RB-01 profile preflight must remain healthy: {error}"),
        };
        let thermal_before =
            rb01_macos_thermal_state().expect("macOS thermal state must be readable");
        process_system.refresh_all();
        let rss_before = process_system
            .process(pid)
            .map_or(0, |process| process.memory());
        let mut calibration_load = if calibration_duties.is_empty() {
            None
        } else {
            Some(Rb01SyntheticLoad::start(&calibration_duties))
        };
        let (calibration_cpu_before, foreground_latency_baseline) =
            if let Some((minimum_cpu, maximum_cpu)) = calibration_cpu_band {
                process_system.refresh_cpu_usage();
                tokio::time::sleep(Duration::from_secs(3)).await;
                process_system.refresh_cpu_usage();
                let observed_cpu = process_system.global_cpu_usage();

                let (latency_stop, latency_worker) = rb01_spawn_latency_probe();
                tokio::time::sleep(Duration::from_secs(2)).await;
                let latency_samples = rb01_finish_latency_probe(latency_stop, latency_worker);
                let latency_summary = rb01_latency_summary(&latency_samples);

                if observed_cpu < minimum_cpu || observed_cpu > maximum_cpu {
                    calibration_load.take();
                    println!(
                        "{}",
                        serde_json::json!({
                            "event": "rb01_calibration_skipped",
                            "lane": lane.as_str(),
                            "load_duties_percent": calibration_duties,
                            "requested_cpu_band_percent": [minimum_cpu, maximum_cpu],
                            "observed_cpu_percent": observed_cpu,
                            "foreground_timer_baseline": latency_summary,
                            "thermal_after": rb01_macos_thermal_state(),
                            "report_elapsed_ms": 0
                        })
                    );
                    return;
                }
                (Some(observed_cpu), Some(latency_summary))
            } else {
                (None, None)
            };
        let peak_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peak_task = tokio::spawn(rb01_sample_peak_rss(pid, rss_before, peak_stop.clone()));
        tokio::task::yield_now().await;

        let started = Instant::now();
        let runtime_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let runtime_task = tokio::spawn(rb01_sample_runtime_until(started, runtime_stop.clone()));
        let latency_probe = calibration_cpu_band.map(|_| rb01_spawn_latency_probe());
        let report = run_ambient_job_safe(
            lane.job(),
            &db,
            Some(&provider),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;
        let wall_ms = started.elapsed().as_millis();
        let foreground_latency_during = latency_probe.map(|(stop, worker)| {
            let samples = rb01_finish_latency_probe(stop, worker);
            rb01_latency_summary(&samples)
        });
        runtime_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let runtime_samples = runtime_task
            .await
            .expect("RB-01 runtime sampler must remain alive");
        calibration_load.take();
        peak_stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let rss_peak_during_slice = peak_task
            .await
            .expect("RB-01 peak-RSS sampler must remain alive");
        process_system.refresh_all();
        let rss_after = process_system
            .process(pid)
            .map_or(0, |process| process.memory());
        let process_cpu_after = process_system
            .process(pid)
            .map_or(0.0, |process| process.cpu_usage());
        let after = ResourceSnapshot {
            cpu_usage_percent: process_system.global_cpu_usage(),
            available_memory_bytes: process_system.available_memory(),
            total_memory_bytes: process_system.total_memory(),
        };
        let thermal_after =
            rb01_macos_thermal_state().expect("macOS thermal state must remain readable");
        let durable_progress = rb01_lane_progressed(lane, &db, &fixture).await;
        let runtime_peak_cpu_percent = runtime_samples
            .iter()
            .map(|(_, cpu, _)| *cpu)
            .reduce(f32::max);
        let runtime_min_available_memory_bytes =
            runtime_samples.iter().map(|(_, _, memory)| *memory).min();
        let foreground_latency_ok = foreground_latency_baseline
            .as_ref()
            .zip(foreground_latency_during.as_ref())
            .map(|(baseline, during)| {
                let baseline_p99 = baseline["p99_us"].as_u64().unwrap_or(u64::MAX);
                let during_p99 = during["p99_us"].as_u64().unwrap_or(u64::MAX);
                during_p99 <= 10_000 && during_p99 <= baseline_p99.saturating_add(5_000)
            });

        println!(
            "{}",
            serde_json::json!({
                "event": "rb01_profile",
                "lane": lane.as_str(),
                "model": "qwen3-4b",
                "backend": "on_device",
                "boot_ms": boot_ms,
                "selected": report.selected,
                "llm_calls": report.llm_calls,
                "panicked": report.panicked,
                "wall_ms": wall_ms,
                "report_elapsed_ms": report.elapsed.as_millis(),
                "rss_process_baseline_bytes": rss_process_baseline,
                "rss_model_loaded_bytes": rss_model_loaded,
                "rss_model_delta_bytes": rss_model_loaded.saturating_sub(rss_process_baseline),
                "rss_before_bytes": rss_before,
                "rss_peak_during_slice_bytes": rss_peak_during_slice,
                "rss_after_bytes": rss_after,
                "available_memory_pre_model_bytes": available_memory_pre_model,
                "available_memory_model_loaded_bytes": available_memory_model_loaded,
                "system_cpu_before_percent": before.cpu_usage_percent,
                "calibration_load_duties_percent": calibration_duties,
                "calibration_requested_cpu_band_percent": calibration_cpu_band.map(|(min, max)| [min, max]),
                "calibration_cpu_before_percent": calibration_cpu_before,
                "foreground_timer_baseline": foreground_latency_baseline,
                "foreground_timer_during": foreground_latency_during,
                "foreground_timer_acceptance": "during p99 <= 10ms and <= baseline p99 + 5ms",
                "foreground_timer_ok": foreground_latency_ok,
                "runtime_cpu_samples": runtime_samples.iter().map(|(observed_ms, cpu, memory)| serde_json::json!({
                    "observed_ms": observed_ms,
                    "cpu_percent": cpu,
                    "available_memory_bytes": memory
                })).collect::<Vec<_>>(),
                "runtime_peak_cpu_percent": runtime_peak_cpu_percent,
                "runtime_min_available_memory_bytes": runtime_min_available_memory_bytes,
                "system_cpu_after_percent": after.cpu_usage_percent,
                "process_cpu_after_percent": process_cpu_after,
                "available_memory_before_bytes": before.available_memory_bytes,
                "available_memory_after_bytes": after.available_memory_bytes,
                "total_memory_bytes": after.total_memory_bytes,
                "thermal_before": thermal_before,
                "thermal_after": thermal_after,
                "durable_progress": durable_progress,
            })
        );

        assert!(
            report.llm_calls <= 1,
            "one profiled turn forwards at most one request"
        );
        if matches!(lane, Rb01ProfileLane::PageGrowth) {
            assert_eq!(
                report.llm_calls, 0,
                "the Page Growth no-match fixture must measure CPU-only work"
            );
        }
        assert!(
            report.selected,
            "profile fixture must select one durable item"
        );
        assert!(
            durable_progress,
            "profile fixture must leave durable lane progress"
        );
        assert!(!report.panicked, "profiled lane must not panic");
        assert_eq!(
            thermal_after, 0,
            "thermal state left nominal during a single bounded slice"
        );
        std::mem::forget(provider);
    }

    #[tokio::test]
    async fn ambient_provider_panic_isolated_and_next_turn_still_runs() {
        let (db, _db_dir) = new_test_db().await;
        store_test_memory(
            &db,
            "ambient-panic-recovery",
            "The launch decision belongs to the work project.",
        )
        .await;
        db.upsert_enrichment_origin(
            "ambient-panic-recovery",
            wenlan_core::db::EnrichmentOrigin {
                memory_type_explicit: false,
                structured_fields_explicit: false,
                space_rejected: false,
            },
        )
        .await
        .unwrap();

        let panicking: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(PanicTestProvider);
        let first = run_ambient_job_safe(
            AmbientJob::Classification,
            &db,
            Some(&panicking),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;
        assert!(
            first.panicked,
            "the panic is surfaced to scheduler accounting"
        );
        assert!(
            first.selected,
            "a panicked lane stays eligible after the thermal cooldown"
        );

        let healthy: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(MaintenanceTestProvider {
                body: r#"{"memory_type":"decision","domain":null,"quality":"high","importance":8,"tags":["launch"]}"#
                    .to_string(),
            });
        let second = run_ambient_job_safe(
            AmbientJob::Classification,
            &db,
            Some(&healthy),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;

        assert!(!second.panicked);
        assert!(second.selected);
        assert_eq!(second.llm_calls, 1);
        assert!(db
            .get_enrichment_steps("ambient-panic-recovery")
            .await
            .unwrap()
            .iter()
            .any(|step| step.step == "classify" && step.status == "ok"));
    }

    #[tokio::test]
    async fn ambient_classification_turn_forwards_once_and_commits_receipt() {
        let (db, _db_dir) = new_test_db().await;
        store_test_memory(
            &db,
            "ambient-classification",
            "The launch decision belongs to the work project.",
        )
        .await;
        db.upsert_enrichment_origin(
            "ambient-classification",
            wenlan_core::db::EnrichmentOrigin {
                memory_type_explicit: false,
                structured_fields_explicit: false,
                space_rejected: false,
            },
        )
        .await
        .unwrap();
        let provider: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(MaintenanceTestProvider {
                body: r#"{"memory_type":"decision","domain":null,"quality":"high","importance":8,"tags":["launch"]}"#
                    .to_string(),
            });

        let report = run_ambient_job(
            AmbientJob::Classification,
            &db,
            Some(&provider),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;

        assert!(report.selected);
        assert_eq!(report.llm_calls, 1);
        let steps = db
            .get_enrichment_steps("ambient-classification")
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
    async fn ambient_pending_memory_forwards_zero_classification_calls() {
        let (db, _db_dir) = new_test_db().await;
        let mut pending = wenlan_types::RawDocument {
            source: "memory".to_string(),
            source_id: "ambient-pending-classification".to_string(),
            title: "Pending revision".to_string(),
            content: "This revision must not be enriched before approval.".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".to_string()),
            source_agent: Some("test".to_string()),
            confirmed: Some(true),
            ..Default::default()
        };
        pending.pending_revision = true;
        db.upsert_documents(vec![pending]).await.unwrap();
        let provider: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(MaintenanceTestProvider {
                body: "must not be called".to_string(),
            });

        let report = run_ambient_job(
            AmbientJob::Classification,
            &db,
            Some(&provider),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;

        assert!(!report.selected);
        assert_eq!(report.llm_calls, 0);
        assert!(db
            .get_enrichment_steps("ambient-pending-classification")
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn ambient_page_growth_no_match_forwards_zero_calls_and_commits_receipt() {
        let (db, _db_dir) = new_test_db().await;
        store_test_memory(
            &db,
            "ambient-growth-no-match",
            "A standalone memory with no matching Page.",
        )
        .await;
        assert!(db
            .record_enrichment_step_at_version(
                "ambient-growth-no-match",
                "entity_extract",
                "ok",
                None,
                1,
            )
            .await
            .unwrap());
        let provider: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(MaintenanceTestProvider {
                body: "must not be called".to_string(),
            });

        let report = run_ambient_job(
            AmbientJob::PageGrowth,
            &db,
            Some(&provider),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;

        assert!(report.selected);
        assert_eq!(report.llm_calls, 0);
        assert!(
            report.page_growth_terminal_no_match_committed,
            "only a committed terminal no-match may skip the thermal cooldown"
        );
        assert!(!ambient_work_consumes_thermal_turn(
            report.job,
            report.selected,
            report.llm_calls,
            report.page_growth_terminal_no_match_committed,
        ));
        let growth = db
            .get_enrichment_steps("ambient-growth-no-match")
            .await
            .unwrap()
            .into_iter()
            .find(|step| step.step == "page_growth")
            .expect("terminal no-match receipt");
        assert_eq!(growth.status, "ok");
        assert_eq!(growth.input_version, Some(1));
    }

    #[tokio::test]
    async fn ambient_reconcile_backpressure_uses_lane_rescan_backoff() {
        let (db, _db_dir) = new_test_db().await;
        let mut pending = Vec::new();
        for i in 0..=wenlan_core::reconcile::RECONCILE_PENDING_CAP {
            pending.push(wenlan_types::RawDocument {
                source: "memory".to_string(),
                source_id: format!("ambient-reconcile-pending-{i}"),
                title: "Pending reconcile revision".to_string(),
                content: "A pending revision awaiting human review.".to_string(),
                last_modified: chrono::Utc::now().timestamp(),
                source_agent: Some("reconcile".to_string()),
                confirmed: None,
                pending_revision: true,
                supersedes: Some(format!("ambient-reconcile-target-{i}")),
                ..Default::default()
            });
        }
        db.upsert_documents(pending).await.unwrap();
        let provider: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(MaintenanceTestProvider {
                body: "must not be called".to_string(),
            });

        let report = run_ambient_job(
            AmbientJob::Reconcile,
            &db,
            Some(&provider),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;

        assert_eq!(report.llm_calls, 0);
        assert!(
            !report.selected,
            "administrative backpressure is no work and must receive lane backoff"
        );

        let now = Instant::now();
        let mut schedule = AmbientSchedule::new(now);
        schedule.note_job_result(
            report.job,
            now,
            !should_backoff_ambient_lane(report.selected, report.llm_calls),
        );
        assert_eq!(schedule.last_reconcile, Some(now));
        let reconcile_only = AmbientAvailability {
            document: false,
            classification: false,
            structured_extract: false,
            entity: false,
            title: false,
            page_growth: false,
            reconcile: true,
            citation: false,
            edges_reconcile: false,
            entity_page_reconcile: false,
            edge_grounding_promote: false,
        };
        assert_eq!(
            schedule.select_due(
                now + RECONCILE_SWEEP_INTERVAL - Duration::from_secs(1),
                reconcile_only
            ),
            None
        );
        assert_eq!(
            schedule.select_due(now + RECONCILE_SWEEP_INTERVAL, reconcile_only),
            Some(AmbientJob::Reconcile)
        );
    }

    #[tokio::test]
    async fn ambient_reconcile_zero_candidate_progress_stays_due_but_is_thermally_paced() {
        let (db, _db_dir) = new_test_db().await;
        db.upsert_documents(vec![wenlan_types::RawDocument {
            source: "memory".to_string(),
            source_id: "ambient-reconcile-doc-only".to_string(),
            title: "Document-only frontier item".to_string(),
            content: "A folder document with no capture candidate still advances the frontier."
                .to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            source_agent: Some("folder".to_string()),
            confirmed: Some(true),
            content_hash: Some("ambient-reconcile-doc-hash".to_string()),
            ..Default::default()
        }])
        .await
        .unwrap();
        let provider: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(MaintenanceTestProvider {
                body: "must not be called".to_string(),
            });

        let report = run_ambient_job(
            AmbientJob::Reconcile,
            &db,
            Some(&provider),
            None,
            None,
            Some(wenlan_core::refinery::EverydaySource::OnDevice),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::RefineryConfig::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
        )
        .await;

        assert!(
            db.get_app_metadata("reconcile_frontier_docs")
                .await
                .unwrap()
                .is_some(),
            "the zero-candidate item advances the durable frontier"
        );
        assert_eq!(report.llm_calls, 0);
        assert!(
            report.selected,
            "durable frontier progress is real work even without a judge call"
        );

        let now = Instant::now();
        let mut schedule = AmbientSchedule::new(now);
        schedule.note_job_result(
            report.job,
            now,
            !should_backoff_ambient_lane(report.selected, report.llm_calls),
        );
        assert_eq!(
            schedule.last_reconcile, None,
            "known backlog must not receive the 30-minute empty-lane backoff"
        );
        assert!(ambient_work_consumes_thermal_turn(
            report.job,
            report.selected,
            report.llm_calls,
            report.page_growth_terminal_no_match_committed,
        ));
        schedule.note_thermal_work_completion(
            now,
            Duration::from_secs(1),
            ThermalPolicy::conservative(),
        );
        assert_eq!(schedule.next_allowed_at, now + Duration::from_secs(120));
    }

    async fn insert_test_page(
        db: &wenlan_core::db::MemoryDB,
        title: &str,
        content: &str,
        source_ids: &[&str],
        creation_kind: &str,
    ) -> String {
        let source_memory_ids: Vec<String> = if creation_kind == "distilled" {
            source_ids.iter().map(|id| (*id).to_string()).collect()
        } else {
            Vec::new()
        };
        let result = wenlan_core::post_write::create_page_with_tuning(
            db,
            wenlan_types::requests::CreateConceptRequest {
                title: title.to_string(),
                content: content.to_string(),
                summary: None,
                entity_id: None,
                source_memory_ids,
                creation_kind: Some(creation_kind.to_string()),
                space: Some("work".to_string()),
                workspace: Some("work".to_string()),
            },
            "test",
            None,
            source_ids.len().max(1),
            1.1,
        )
        .await
        .unwrap();
        if creation_kind != "distilled" && !source_ids.is_empty() {
            let source_memory_ids: Vec<String> =
                source_ids.iter().map(|id| (*id).to_string()).collect();
            wenlan_core::post_write::page_write(
                db,
                wenlan_core::post_write::PageWrite::Attach {
                    page_id: &result.id,
                    source_memory_ids: &source_memory_ids,
                    link_reason: "test_fixture_attach",
                    agent: "test",
                },
            )
            .await
            .unwrap();
            // These links are the page's already-compiled initial evidence,
            // not a later source addition. Production Attach correctly marks
            // additions stale, so the fixture acknowledges its initial build.
            db.clear_page_staleness(&result.id).await.unwrap();
        }
        db.set_page_review_status(&result.id, "confirmed")
            .await
            .unwrap();
        result.id
    }

    #[tokio::test]
    async fn maintenance_provider_panic_isolated_and_scheduler_state_survives() {
        struct AvailabilityPanicProvider;

        #[async_trait::async_trait]
        impl wenlan_core::llm_provider::LlmProvider for AvailabilityPanicProvider {
            async fn generate(
                &self,
                _request: wenlan_core::llm_provider::LlmRequest,
            ) -> Result<String, wenlan_core::llm_provider::LlmError> {
                unreachable!("availability check must run first")
            }

            fn is_available(&self) -> bool {
                panic!("maintenance availability panic")
            }

            fn name(&self) -> &str {
                "availability-panic-test"
            }

            fn backend(&self) -> wenlan_core::llm_provider::LlmBackend {
                wenlan_core::llm_provider::LlmBackend::Api
            }
        }

        let (db, _db_dir) = new_test_db().await;
        store_test_memory(
            &db,
            "maintenance-panic-source",
            "A source update that requires a machine-page refresh.",
        )
        .await;
        let page_id = insert_test_page(
            &db,
            "Maintenance panic page",
            "Old machine-owned prose.",
            &["maintenance-panic-source"],
            "research",
        )
        .await;
        db.set_page_stale(&page_id, "source_updated").await.unwrap();
        let provider: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            Arc::new(AvailabilityPanicProvider);
        let selected = resolve_maintenance_provider(
            Some(wenlan_core::refinery::SynthesisSource::External),
            None,
            None,
            Some(&provider),
            None,
        );

        fire_maintenance_stage_safe(
            db.as_ref(),
            selected.as_ref(),
            &wenlan_core::prompts::PromptRegistry::default(),
            &wenlan_core::tuning::DistillationConfig::default(),
            None,
            wenlan_core::maintenance::MaintenanceStage::StalePage,
            "panic-test",
        )
        .await;

        assert!(db.get_page(&page_id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn maintenance_tick_detects_page_merge_cards_and_routes_stale_pages() {
        let (db, _db_dir) = new_test_db().await;
        let source = "Rust ownership prevents data races at compile time.";
        for id in [
            "mem_dup_a",
            "mem_dup_b",
            "mem_dup_c",
            "mem_machine",
            "mem_human",
        ] {
            store_test_memory(&db, id, source).await;
        }

        let page_dup_a = insert_test_page(
            &db,
            "Rust ownership",
            "Rust ownership prevents data races at compile time.",
            &["mem_dup_a", "mem_dup_b", "mem_dup_c"],
            "distilled",
        )
        .await;
        let page_dup_b = insert_test_page(
            &db,
            "Rust borrowing",
            "Rust ownership prevents data races at compile time.",
            &["mem_dup_a", "mem_dup_b", "mem_dup_c"],
            "distilled",
        )
        .await;
        let page_machine_stale = insert_test_page(
            &db,
            "Machine stale page",
            "Old machine-owned prose.",
            &["mem_machine"],
            "research",
        )
        .await;
        let page_human_stale = insert_test_page(
            &db,
            "Human stale page",
            "Human-written prose must remain untouched.",
            &["mem_human"],
            "authored",
        )
        .await;
        let _page_orphan_source = insert_test_page(
            &db,
            "Orphan source",
            "This page links to [[Missing Topic]].",
            &["mem_machine"],
            "research",
        )
        .await;
        db.set_page_stale(&page_machine_stale, "source_updated")
            .await
            .unwrap();
        db.set_page_stale(&page_human_stale, "source_updated")
            .await
            .unwrap();

        let llm: std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider> =
            std::sync::Arc::new(MaintenanceTestProvider {
                body: format!("{source} [1]"),
            });
        let prompts = wenlan_core::prompts::PromptRegistry::default();

        let result = wenlan_core::maintenance::run_maintenance_tick(
            &db,
            Some(&llm),
            &prompts,
            &wenlan_core::maintenance::MaintenanceTickConfig {
                page_match_threshold: 0.85,
                formation_threshold: 0.60,
                page_min_cluster_size: 3,
                token_limit: 3500,
                max_unlinked_cluster_size: 20,
                max_grouped_cluster_size: 20,
                max_per_tick: 5,
            },
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.merge_cards_emitted, 1);
        assert_eq!(result.stale_machine_refreshed, 1);
        assert_eq!(result.stale_human_cards, 1);
        assert!(
            result.orphan_labels_checked >= 1,
            "the maintenance tick must run the orphan wikilink check"
        );
        assert_eq!(result.overview_refreshed, 1);

        let proposals = db.get_pending_refinements().await.unwrap();
        let merge_card = proposals
            .iter()
            .find(|p| p.action == "page_merge")
            .expect("near-duplicate pages must emit a page_merge card");
        assert_eq!(merge_card.source_ids.len(), 2);
        assert!(merge_card.source_ids.contains(&page_dup_a));
        assert!(merge_card.source_ids.contains(&page_dup_b));

        let machine = db
            .get_page(&page_machine_stale)
            .await
            .unwrap()
            .expect("machine page remains");
        assert_eq!(machine.stale_reason, None);
        assert!(
            machine
                .content
                .contains("Rust ownership prevents data races"),
            "machine-owned stale page should be refreshed in place"
        );

        let human = db
            .get_page(&page_human_stale)
            .await
            .unwrap()
            .expect("human page remains");
        assert_eq!(human.stale_reason, None);
        assert_eq!(human.content, "Human-written prose must remain untouched.");

        let revisions = db.list_pending_revisions(10).await.unwrap();
        assert!(
            revisions
                .iter()
                .any(|r| r.target_source_id == page_human_stale),
            "human-owned stale page should stage a revision card"
        );

        assert!(
            db.find_active_page_id_by_title("Overview")
                .await
                .unwrap()
                .is_some(),
            "overview refresh must create or update the reserved Overview page"
        );
    }
}
