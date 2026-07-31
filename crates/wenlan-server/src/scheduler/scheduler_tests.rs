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
            !AmbientAvailability::for_provider(true).supports(AmbientJob::EdgeGroundingPromote),
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
    let mut round =
        AutomaticMaintenanceRound::new(Some(wenlan_core::maintenance::MaintenanceStage::StalePage));
    round.complete_stage(
        wenlan_core::maintenance::MaintenanceStage::StalePage,
        AutomaticMaintenanceOutcome::default(),
    );
    let cursor = round.cursor_after_attempt(wenlan_core::maintenance::MaintenanceStage::StalePage);
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
    let manifest = include_str!("../../../../Cargo.toml");
    assert!(
        manifest
            .lines()
            .any(|line| line.trim() == "sysinfo = \"=0.38.3\""),
        "the scheduler must retain the reviewed exact sysinfo 0.38.3 pin"
    );

    let lock = include_str!("../../../../Cargo.lock");
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

    source.status = wenlan_types::sources::SyncStatus::Error("transient file error".to_string());
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
        wenlan_core::db::MemoryDB::new(db_dir.path(), Arc::new(wenlan_core::events::NoopEmitter))
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

    let panicking: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(PanicTestProvider);
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
    let api: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(AvailabilityTestProvider {
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
    let api: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(AvailabilityTestProvider {
        name: "available-api",
        available: true,
    });
    let external: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
        Arc::new(AvailabilityTestProvider {
            name: "available-external",
            available: true,
        });

    assert!(resolve_maintenance_provider(None, None, Some(&api), Some(&external), None).is_none());
}

#[test]
fn maintenance_provider_selection_does_not_fallback_from_missing_pin() {
    let api: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(AvailabilityTestProvider {
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
    let api: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(AvailabilityTestProvider {
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

async fn rb01_wait_for_profile_admission(calibration: bool) -> Result<ResourceSnapshot, String> {
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
    let thermal_before = rb01_macos_thermal_state().expect("macOS thermal state must be readable");
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

    let panicking: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(PanicTestProvider);
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
            space: Some("work".to_string()).into(),
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
