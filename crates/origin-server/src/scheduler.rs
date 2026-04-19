// SPDX-License-Identifier: Apache-2.0
//! Event-driven steep scheduler.
//!
//! Owns all steep scheduling: BurstEnd (per-agent adaptive gap), Idle (global
//! 10-min quiet), Daily (first-wake-after-24h), and Backstop (6-hour safety net).
//! Replaces the former steep loop in main.rs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::state::SharedState;

/// 30-minute ceiling for adaptive gap — matches ACTIVITY_GAP_SECS in origin-core.
const BURST_GAP_CEILING: Duration = Duration::from_secs(1800);
/// 5-minute floor — prevents premature firing on fast writers.
const BURST_GAP_FLOOR: Duration = Duration::from_secs(300);
/// Minimum writes to qualify as a recap-worthy burst.
const MIN_BURST_WRITES: usize = 3;
/// Global idle threshold — all agents quiet for this long triggers Idle steep.
const IDLE_THRESHOLD: Duration = Duration::from_secs(600);
/// Backstop interval — safety net fires all phases.
const BACKSTOP_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);
/// Poll interval — how often the scheduler checks trigger conditions.
const POLL_INTERVAL: Duration = Duration::from_secs(30);
/// Initial delay — lets on-device model warm up before first backstop.
const INITIAL_DELAY: Duration = Duration::from_secs(60);

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

    /// Record a write event for an agent. Called by handle_store_memory BEFORE
    /// spawning post-ingest enrichment (captures timestamp immediately).
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

/// Spawn the event-driven steep scheduler.
///
/// Runs a single tokio task with a 30-second poll loop that checks four
/// trigger conditions: BurstEnd, Idle, Daily, and Backstop. All `fire()` calls
/// are awaited inline — the poll interval is a minimum, not a hard clock.
pub fn spawn_scheduler(shared: SharedState, write_signal: WriteSignal) {
    tokio::spawn(async move {
        tokio::time::sleep(INITIAL_DELAY).await;

        let mut last_backstop = Instant::now();
        let mut idle_fired = false;
        let mut last_poll_activity = Instant::now();

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
            tokio::time::sleep(POLL_INTERVAL).await;

            // Reset idle flag if any new activity arrived since last poll
            if write_signal.has_activity_since(last_poll_activity) {
                idle_fired = false;
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
                        s.prompts.clone(),
                        s.tuning.refinery.clone(),
                        s.tuning.confidence.clone(),
                        s.tuning.distillation.clone(),
                    )
                })
            };

            let Some((
                db,
                llm,
                api_llm,
                synthesis_llm,
                prompts,
                refinery_cfg,
                confidence_cfg,
                distillation_cfg,
            )) = snapshot
            else {
                tracing::debug!("[scheduler] db not initialized, skipping poll");
                continue;
            };

            let now = Instant::now();

            // --- 1. BurstEnd: per-agent adaptive gap detection ---
            let snap = write_signal.snapshot();
            for (agent, timestamps) in &snap {
                if timestamps.is_empty() {
                    continue;
                }
                let gap = adaptive_gap(timestamps);
                let last_write = *timestamps.iter().max().unwrap();
                if now.duration_since(last_write) > gap {
                    if timestamps.len() >= MIN_BURST_WRITES {
                        tracing::info!(
                            "[scheduler] BurstEnd for agent '{}' — {} writes, gap {:?}",
                            agent,
                            timestamps.len(),
                            gap
                        );
                        fire_steep_safe(
                            &db,
                            llm.as_ref(),
                            api_llm.as_ref(),
                            synthesis_llm.as_ref(),
                            &prompts,
                            &refinery_cfg,
                            &confidence_cfg,
                            &distillation_cfg,
                            origin_core::refinery::TriggerKind::BurstEnd,
                            "BurstEnd",
                        )
                        .await;
                    }
                    write_signal.drain_up_to(agent, last_write);
                }
            }

            // --- 2. Idle: global quiet for IDLE_THRESHOLD ---
            // No need to check write_signal.snapshot().is_empty() — undrained
            // sub-threshold bursts (e.g., a single write waiting for its 30-min
            // ceiling) should not block Idle. Synthesis phases (community_detection,
            // emergence, re-distill, decision_logs) don't overlap with BurstEnd
            // phases (recaps, refinement_queue), so running them concurrently with
            // pending burst timestamps is safe.
            let idle_horizon = now.checked_sub(IDLE_THRESHOLD).unwrap_or_else(Instant::now);
            if !idle_fired && !write_signal.has_activity_since(idle_horizon) {
                tracing::info!(
                    "[scheduler] Idle — all agents quiet for {}s",
                    IDLE_THRESHOLD.as_secs()
                );
                fire_steep_safe(
                    &db,
                    llm.as_ref(),
                    api_llm.as_ref(),
                    synthesis_llm.as_ref(),
                    &prompts,
                    &refinery_cfg,
                    &confidence_cfg,
                    &distillation_cfg,
                    origin_core::refinery::TriggerKind::Idle,
                    "Idle",
                )
                .await;
                idle_fired = true;
            }

            // --- 3. Daily: first-wake-after-24h ---
            if now.duration_since(last_daily) > Duration::from_secs(24 * 60 * 60) {
                tracing::info!("[scheduler] Daily — first fire in >24h");
                fire_steep_safe(
                    &db,
                    llm.as_ref(),
                    api_llm.as_ref(),
                    synthesis_llm.as_ref(),
                    &prompts,
                    &refinery_cfg,
                    &confidence_cfg,
                    &distillation_cfg,
                    origin_core::refinery::TriggerKind::Daily,
                    "Daily",
                )
                .await;
                last_daily = now;
                let epoch = chrono::Utc::now().timestamp().to_string();
                if let Err(e) = db.set_app_metadata("last_daily_steep_ts", &epoch).await {
                    tracing::warn!("[scheduler] failed to persist last_daily_steep_ts: {}", e);
                }
            }

            // --- 4. Backstop: safety net ---
            if now.duration_since(last_backstop) > BACKSTOP_INTERVAL {
                tracing::info!("[scheduler] Backstop — safety net fire");
                fire_steep_safe(
                    &db,
                    llm.as_ref(),
                    api_llm.as_ref(),
                    synthesis_llm.as_ref(),
                    &prompts,
                    &refinery_cfg,
                    &confidence_cfg,
                    &distillation_cfg,
                    origin_core::refinery::TriggerKind::Backstop,
                    "Backstop",
                )
                .await;
                last_backstop = now;
            }
        }
    });
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

/// Fire a steep with panic isolation. If the steep panics (e.g., OOM during
/// distillation of a massive cluster), the scheduler loop continues.
#[allow(clippy::too_many_arguments)]
async fn fire_steep_safe(
    db: &origin_core::db::MemoryDB,
    llm: Option<&std::sync::Arc<dyn origin_core::llm_provider::LlmProvider>>,
    api_llm: Option<&std::sync::Arc<dyn origin_core::llm_provider::LlmProvider>>,
    synthesis_llm: Option<&std::sync::Arc<dyn origin_core::llm_provider::LlmProvider>>,
    prompts: &origin_core::prompts::PromptRegistry,
    refinery_cfg: &origin_core::tuning::RefineryConfig,
    confidence_cfg: &origin_core::tuning::ConfidenceConfig,
    distillation_cfg: &origin_core::tuning::DistillationConfig,
    trigger: origin_core::refinery::TriggerKind,
    label: &str,
) {
    let result = std::panic::AssertUnwindSafe(fire_steep(
        db,
        llm,
        api_llm,
        synthesis_llm,
        prompts,
        refinery_cfg,
        confidence_cfg,
        distillation_cfg,
        trigger,
        label,
    ));
    if let Err(e) = futures::FutureExt::catch_unwind(result).await {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        tracing::error!(
            "[scheduler] {} PANICKED — scheduler continues: {}",
            label,
            msg
        );
    }
}

/// Fire a steep with the given trigger, log result summary and activity entries.
#[allow(clippy::too_many_arguments)]
async fn fire_steep(
    db: &origin_core::db::MemoryDB,
    llm: Option<&std::sync::Arc<dyn origin_core::llm_provider::LlmProvider>>,
    api_llm: Option<&std::sync::Arc<dyn origin_core::llm_provider::LlmProvider>>,
    synthesis_llm: Option<&std::sync::Arc<dyn origin_core::llm_provider::LlmProvider>>,
    prompts: &origin_core::prompts::PromptRegistry,
    refinery_cfg: &origin_core::tuning::RefineryConfig,
    confidence_cfg: &origin_core::tuning::ConfidenceConfig,
    distillation_cfg: &origin_core::tuning::DistillationConfig,
    trigger: origin_core::refinery::TriggerKind,
    label: &str,
) {
    let started = std::time::Instant::now();
    match origin_core::refinery::run_periodic_steep_with_api(
        db,
        llm,
        api_llm,
        synthesis_llm,
        prompts,
        refinery_cfg,
        confidence_cfg,
        distillation_cfg,
        trigger,
    )
    .await
    {
        Ok(result) => {
            let errors = result.phases.iter().filter(|p| p.error.is_some()).count();
            tracing::info!(
                "[scheduler] {} complete in {}ms — {} phases ({} errors), {} decayed, {} recaps, {} distilled, {} pending",
                label,
                started.elapsed().as_millis(),
                result.phases.len(),
                errors,
                result.memories_decayed,
                result.recaps_generated,
                result.distilled,
                result.pending_remaining,
            );

            // Log non-Silent phases as activity entries (same as old steep loop)
            for phase in &result.phases {
                if phase.nudge != origin_core::refinery::Nudge::Silent {
                    if let Some(ref headline) = phase.headline {
                        if let Err(e) = db
                            .log_agent_activity("origin", "steep", &[], None, headline)
                            .await
                        {
                            tracing::warn!(
                                "[scheduler] log activity for phase {} failed: {}",
                                phase.name,
                                e
                            );
                        }
                    }
                }
            }
        }
        Err(e) => tracing::warn!("[scheduler] {} error: {}", label, e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_gap_empty_returns_ceiling() {
        assert_eq!(adaptive_gap(&[]), BURST_GAP_CEILING);
    }

    #[test]
    fn test_adaptive_gap_single_write_returns_ceiling() {
        assert_eq!(adaptive_gap(&[Instant::now()]), BURST_GAP_CEILING);
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
}
