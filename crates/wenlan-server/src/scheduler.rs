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

mod ambient;
use ambient::*;

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
#[path = "scheduler/scheduler_tests.rs"]
mod tests;
