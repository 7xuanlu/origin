// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) async fn register_optional_runtime_workers(
    shared: SharedState,
    repair_recovery_pending: bool,
    deep_bgebase_pending: bool,
    reranker_cache_dir: Option<std::path::PathBuf>,
    config: wenlan_core::config::Config,
    db_arc: Arc<wenlan_core::db::MemoryDB>,
) {
    // full mode: load the heavy deep bge-base in the background so startup never
    // blocks on the ~1.1GB download (council fix #3). rerank=true uses plain hybrid
    // until this completes; deep status flips to Active/Failed when the load resolves.
    if optional_runtime_workers_allowed(repair_recovery_pending) && deep_bgebase_pending {
        let shared_for_deep = shared.clone();
        let cache = reranker_cache_dir.clone();
        tokio::spawn(async move {
            use wenlan_types::responses::RerankerStatus;
            tracing::info!(
                "[reranker] full mode: loading deep bge-base in background (~1.1GB first run); \
                 rerank=true uses plain hybrid until ready\u{2026}"
            );
            let loaded = tokio::task::spawn_blocking(move || {
                wenlan_core::reranker::init_cross_encoder_reranker_pick(
                    wenlan_core::reranker::RerankerPick::BgeBase,
                    cache,
                )
            })
            .await;
            match loaded {
                Ok(Ok(r)) => {
                    let model_id = r.model_id().to_string();
                    let mut st = shared_for_deep.write().await;
                    st.reranker_status = RerankerStatus::Active {
                        model_id: model_id.clone(),
                    };
                    st.reranker = Some(r);
                    tracing::info!("[reranker] deep bge-base loaded and active (model={model_id})");
                }
                Ok(Err(e)) => {
                    let mut st = shared_for_deep.write().await;
                    st.reranker_status = RerankerStatus::Failed {
                        reason: e.to_string(),
                    };
                    tracing::warn!(
                        "[reranker] deep bge-base load failed; rerank=true stays on plain hybrid: {e}"
                    );
                }
                Err(e) => {
                    let mut st = shared_for_deep.write().await;
                    st.reranker_status = RerankerStatus::Failed {
                        reason: e.to_string(),
                    };
                    tracing::warn!("[reranker] deep bge-base load task panicked: {e}");
                }
            }
        });
    }

    // Initialize an explicitly selected, already-cached on-device LLM without
    // making daemon restart itself a foreground-heavy event. Selection still
    // supports explicit routes even when background source pins are absent, so
    // we keep preload semantics; the load now waits for two quiet CPU samples
    // and enough free memory for the registry working set *above* the normal
    // scheduler reserve. A reservation also prevents an automatic turn from
    // racing the load after observing the same quiet window.
    //
    // This intentionally does NOT trigger a download — users opt in explicitly
    // via the settings UI (POST /api/on-device-model/download).
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        let selected_model = config
            .on_device_model
            .as_deref()
            .map(|id| wenlan_core::on_device_models::resolve_or_default(Some(id)));
        match selected_model {
            None => tracing::info!(
                "[on-device] no local model selected, skipping init (run `wenlan models install` to enable)"
            ),
            Some(model) if !wenlan_core::on_device_models::is_cached(model) => tracing::info!(
                "[on-device] model {} not cached, skipping init (use settings to download)",
                model.id
            ),
            Some(model) => {
                let shared_for_llm = shared.clone();
                let reservation = {
                    let state = shared.read().await;
                    state
                        .startup_model_load_reserved
                        .store(true, std::sync::atomic::Ordering::Release);
                    state.startup_model_load_reserved.clone()
                };
                let mut load_shutdown = {
                    let state = shared.read().await;
                    state.shutdown.subscribe()
                };
                let working_set_bytes = on_device_model_working_set_bytes(model);
                tokio::spawn(async move {
                    let _reservation = StartupModelLoadReservation(reservation);
                    if !scheduler::wait_for_startup_model_admission(
                        working_set_bytes,
                        &mut load_shutdown,
                    )
                    .await
                    {
                        tracing::info!(
                            "[on-device] shutdown requested before startup load admission"
                        );
                        return;
                    }
                    let model_id = model.id;
                    let result = tokio::task::spawn_blocking(move || {
                        let provider =
                            wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some(
                                model_id,
                            ))?;
                        let arc: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
                            Arc::new(provider);
                        Ok::<_, wenlan_core::error::WenlanError>((
                            arc,
                            model_id.to_string(),
                        ))
                    })
                    .await;

                    match result {
                        Ok(Ok((provider, model_id))) => {
                            let mut state = shared_for_llm.write().await;
                            state.llm = Some(provider);
                            state.loaded_on_device_model = Some(model_id.clone());
                            tracing::info!("[on-device] model {} loaded and available", model_id);
                        }
                        Ok(Err(e)) => tracing::error!("[on-device] init failed: {}", e),
                        Err(e) => tracing::error!("[on-device] init task panicked: {}", e),
                    }
                });
            }
        }
    }

    // Register the LLM-readiness hook so that the `intelligence-ready`
    // onboarding milestone fires the first time any LLM provider successfully
    // serves traffic. `mark_llm_ready` is a one-shot per process, so this hook
    // runs at most once regardless of which provider fires first.
    //
    // The on-device `llm-provider-worker` (`crates/wenlan-core/src/llm_provider.rs:142`)
    // runs on a `std::thread`, not a Tokio task — GPU inference is blocking
    // and would starve the async runtime. When it calls `mark_llm_ready()`
    // from that thread, our hook fires synchronously on a thread with no
    // Tokio reactor in thread-local context. Bare `tokio::spawn(...)` would
    // then panic: "there is no reactor running, must be called from the
    // context of a Tokio 1.x runtime" — exactly what killed the worker on
    // 2026-04-16. Capture a `Handle` here (we are inside `#[tokio::main]`)
    // and use `handle.spawn(...)` from the closure instead.
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        let db_for_ready = db_arc.clone();
        let (maintenance_for_ready, shutdown_for_reconcile) = {
            let state = shared.read().await;
            (
                state.maintenance_coordinator.clone(),
                state.shutdown.clone(),
            )
        };
        let maintenance_for_reconcile = maintenance_for_ready.clone();
        let maintenance_for_genesis = maintenance_for_ready.clone();
        let emitter_for_ready: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let handle = tokio::runtime::Handle::current();
        let hook: wenlan_core::llm_provider::ReadinessHook = Arc::new(move || {
            let db = db_for_ready.clone();
            let emitter = emitter_for_ready.clone();
            let maintenance = maintenance_for_ready.clone();
            handle.spawn(async move {
                let _maintenance_guard = maintenance.begin_background().await;
                let ev = wenlan_core::onboarding::MilestoneEvaluator::new(&db, emitter);
                if let Err(e) = ev.check_after_llm_ready().await {
                    tracing::warn!(?e, "onboarding: check_after_llm_ready failed");
                }
            });
        });
        let _ = wenlan_core::llm_provider::LLM_READINESS_HOOK.set(hook);

        // Startup opens only the fail-closed frontier. Continue both data-sized
        // truth jobs here, after the listener can serve: one bounded enqueue
        // sweep and one bounded reconciliation batch per turn, yielding between
        // turns. The same task also runs at most one page-linked truth-promotion
        // job per turn once an eligible on-device provider appears. Queue rows,
        // leases, and the reconciliation cursor are durable, so a restart
        // resumes instead of losing the tail.
        let db_for_reconcile = db_arc.clone();
        let shared_for_reconcile = shared.clone();
        let mut reconcile_shutdown = shutdown_for_reconcile.subscribe();
        tokio::spawn(async move {
            if lifecycle::sleep_or_shutdown(
                &mut reconcile_shutdown,
                std::time::Duration::from_secs(1),
            )
            .await
            {
                return;
            }

            let mut backlog_complete = false;
            let mut completion_logged = false;
            let mut last_error = None;
            loop {
                // The on-device model may finish loading after this worker has
                // started. Re-snapshot every turn and end the read guard before
                // any database or inference await.
                let truth_provider = {
                    let state = shared_for_reconcile.read().await;
                    state.llm.clone()
                };
                let work = async {
                    let _maintenance_guard = maintenance_for_reconcile.begin_background().await;
                    let enqueued = if backlog_complete {
                        0
                    } else {
                        db_for_reconcile
                            .enqueue_stale_derivation_jobs(startup::SUPPORT_RECONCILE_BATCH as i64)
                            .await?
                    };
                    if enqueued == 0 {
                        backlog_complete = true;
                    }
                    let pass = db_for_reconcile
                        .reconcile_supported_pages(startup::SUPPORT_RECONCILE_BATCH)
                        .await?;
                    let promotion = if let Some(provider) = truth_provider {
                        // Dropping this future on shutdown prevents the worker
                        // from making any later writes. The durable lease is
                        // intentionally left for a later process to reclaim.
                        Some(tokio::select! {
                            biased;
                            _ = lifecycle::wait_for_shutdown(reconcile_shutdown.clone()) => {
                                return Ok::<_, wenlan_core::WenlanError>((enqueued, pass, None));
                            }
                            result = db_for_reconcile.run_page_linked_truth_promotion_turn(
                                provider.as_ref(),
                                "wenlan-server-truth-maintenance",
                            ) => result,
                        }?)
                    } else {
                        None
                    };
                    Ok::<_, wenlan_core::WenlanError>((enqueued, pass, promotion))
                }
                .await;
                let next_delay = match work {
                    Ok((enqueued, pass, promotion)) => {
                        if lifecycle::shutdown_requested(&reconcile_shutdown) {
                            return;
                        }
                        if last_error.take().is_some() {
                            tracing::info!("[truth] background truth maintenance resumed");
                        }
                        if backlog_complete && pass.complete && !completion_logged {
                            tracing::info!(
                                "[truth] background derivation backlog and support reconciliation \
                                 completed; {enqueued} page(s) enqueued and {} page(s) demoted in \
                                 the final batch; truth promotion remains available until shutdown",
                                pass.demoted,
                            );
                            completion_logged = true;
                        } else if !pass.complete {
                            completion_logged = false;
                        }
                        if enqueued > 0 || pass.demoted > 0 {
                            tracing::info!(
                                "[truth] background truth maintenance enqueued {enqueued} page(s) \
                                 and demoted {} page(s)",
                                pass.demoted,
                            );
                        }
                        let promotion_idle = matches!(
                            promotion.as_ref(),
                            None | Some(wenlan_core::db::PromotionTurn::Idle)
                                | Some(wenlan_core::db::PromotionTurn::RefusedProvider)
                        );
                        if let Some(wenlan_core::db::PromotionTurn::Completed {
                            page_id,
                            page_version,
                        }) = promotion
                        {
                            tracing::info!(
                                "[truth] promoted page {page_id} version {page_version}"
                            );
                        }
                        if backlog_complete && pass.complete && promotion_idle {
                            std::time::Duration::from_secs(1)
                        } else {
                            std::time::Duration::from_millis(100)
                        }
                    }
                    Err(error) => {
                        let error = error.to_string();
                        if last_error.as_deref() != Some(error.as_str()) {
                            tracing::warn!(
                                "[truth] background truth maintenance backing off after an error; \
                                 unproved pages remain unevaluated and the durable backlog remains \
                                 resumable: {error}"
                            );
                            last_error = Some(error);
                        }
                        std::time::Duration::from_secs(5)
                    }
                };

                if lifecycle::sleep_or_shutdown(&mut reconcile_shutdown, next_delay).await {
                    return;
                }
            }
        });

        // The M6 genesis shadow (spec §4.1). A second, *sibling* task rather
        // than a stage inside the loop above: the two must be able to fail,
        // back off, and be disabled independently, and folding them together
        // would make one loop's error backoff throttle the other's work.
        //
        // 3s of startup delay rather than M5's 1s so the two do not contend for
        // the single DB connection mutex while the daemon is still opening.
        //
        // **No `ServerState` snapshot appears below, and that is deliberate.**
        // The M5 loop re-reads `state.llm` every turn because it needs a
        // provider; the shadow needs none, so there is no guard to hold across
        // an await and no provider that could be threaded in later without the
        // reviewer noticing a new read appear here. It is the loop-level half
        // of the same structural proof `finalize.rs` makes at the module level.
        //
        // Bounds per turn are the driver's (§4.2): the startup recovery scan
        // once, then at most one of — one space's frontier reconciliation,
        // one candidate prepare, one dry-run finalization. Nothing runs while
        // `genesis_enabled = 0` in the sense that matters: the flag gates
        // publication, and this lane has no publish path at all.
        let db_for_genesis = db_arc.clone();
        let mut genesis_shutdown = shutdown_for_reconcile.subscribe();
        tokio::spawn(async move {
            if lifecycle::sleep_or_shutdown(
                &mut genesis_shutdown,
                std::time::Duration::from_secs(3),
            )
            .await
            {
                return;
            }

            let mut genesis_state = wenlan_core::m6::shadow::ShadowState::default();
            let mut recovery_logged = false;
            let mut last_genesis_error: Option<String> = None;
            loop {
                // Dropping this future on shutdown prevents the turn from
                // making any later writes; its durable lease is left for a
                // later process to reclaim, exactly as the M5 loop does.
                let turn = tokio::select! {
                    biased;
                    _ = lifecycle::wait_for_shutdown(genesis_shutdown.clone()) => return,
                    result = async {
                        let _maintenance_guard = maintenance_for_genesis.begin_background().await;
                        db_for_genesis.run_genesis_shadow_turn(&mut genesis_state).await
                    } => result,
                };

                let next_delay = match turn {
                    Ok(turn) => {
                        if lifecycle::shutdown_requested(&genesis_shutdown) {
                            return;
                        }
                        if last_genesis_error.take().is_some() {
                            tracing::info!("[genesis] shadow loop resumed");
                        }
                        if !recovery_logged {
                            if let Some(report) = genesis_state.recovery_report() {
                                tracing::info!(
                                    "[genesis] startup recovery reaped {} lease(s), staled {} \
                                     candidate(s), left {} on a live lease, and found {} handed-off \
                                     projection(s)",
                                    report.leases_reaped,
                                    report.candidates_staled.len(),
                                    report.candidates_lease_live,
                                    report.projections_handed_off,
                                );
                                recovery_logged = true;
                            }
                        }
                        let did_work = turn.did_work();
                        if did_work {
                            tracing::debug!(?turn, "[genesis] shadow turn");
                        }
                        if did_work {
                            std::time::Duration::from_millis(100)
                        } else {
                            std::time::Duration::from_secs(1)
                        }
                    }
                    Err(error) => {
                        let error = error.to_string();
                        // De-duplicated: the first occurrence and each change,
                        // never every tick. A shadow that logged a warning ten
                        // times a second would be its own outage.
                        if last_genesis_error.as_deref() != Some(error.as_str()) {
                            tracing::warn!(
                                "[genesis] shadow loop backing off after an error; genesis state \
                                 is durable and the next turn resumes where this one stopped: \
                                 {error}"
                            );
                            last_genesis_error = Some(error);
                        }
                        std::time::Duration::from_secs(5)
                    }
                };

                if lifecycle::sleep_or_shutdown(&mut genesis_shutdown, next_delay).await {
                    return;
                }
            }
        });
    }
}

pub(super) async fn serve_and_drain(
    repair_recovery_pending: bool,
    shared: SharedState,
    shutdown: lifecycle::ShutdownHandle,
    scheduler_task: tokio::task::JoinHandle<()>,
    listener: tokio::net::TcpListener,
) -> anyhow::Result<()> {
    if repair_recovery_pending {
        tracing::warn!("repair-only startup: optional runtime workers are disabled");
    } else if wenlan_core::db::entity_sweep_enabled() {
        tracing::info!(
            "Ambient entity enrichment is ON: the shared quiet/cooldown-gated scheduler \
             backfills knowledge-graph links over existing memories. Set \
             WENLAN_ENABLE_ENTITY_SWEEP=0 to disable."
        );
    } else {
        tracing::info!("Ambient entity enrichment is OFF (WENLAN_ENABLE_ENTITY_SWEEP).");
    }

    // Build router
    let app = if repair_recovery_pending {
        router::build_repair_router(shared)
    } else {
        router::build_router_with_shutdown(shared, shutdown.clone())
    };

    // Advertise the bound port before accepting requests.
    // `addr` may be `127.0.0.1:0`; `local_addr()` gives the real ephemeral port.
    let local_addr = listener.local_addr()?;
    tracing::info!("Listening on http://{}", local_addr);

    // Eval harness reads this stdout line to discover the bound port even when
    // WENLAN_BIND_ADDR=127.0.0.1:0. Format MUST stay stable — see
    // crates/wenlan-core/src/eval/http_harness.rs in the P2 plan.
    println!("WENLAN_LISTENING_ON={}", local_addr);
    let _ = std::io::stdout().flush();

    // Alternate signal: write the port to a file if WENLAN_PORT_FILE is set.
    // Eval harness uses this when stdout is captured by tracing-appender.
    if let Ok(port_file) = std::env::var("WENLAN_PORT_FILE") {
        if let Err(e) = std::fs::write(&port_file, local_addr.port().to_string()) {
            tracing::error!("failed to write WENLAN_PORT_FILE={}: {}", port_file, e);
            return Err(anyhow::anyhow!("WENLAN_PORT_FILE write failed: {}", e));
        }
    }

    // Serve until HTTP shutdown or an OS termination signal. Axum stops
    // accepting new connections and drains in-flight requests; the scheduler
    // finishes its currently awaited item without starting another. A hard
    // deadline remains necessary because Tokio cannot cancel arbitrary
    // spawn_blocking work during runtime drop.
    let server = axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(lifecycle::wait_for_shutdown(shutdown.subscribe()))
        .into_future();
    tokio::pin!(server);
    let server_completed = tokio::select! {
        result = &mut server => Some(result),
        _ = lifecycle::wait_for_shutdown(shutdown.subscribe()) => None,
    };

    if let Some(result) = server_completed {
        shutdown.request();
        match tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, scheduler_task).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!("scheduler join failed: {error}"),
            Err(_) => tracing::warn!("scheduler did not stop within the drain deadline"),
        }
        match result {
            Ok(()) => {
                tracing::info!("HTTP server stopped; daemon lifecycle complete");
                exit_daemon(0);
            }
            Err(error) => {
                tracing::error!("HTTP server failed: {error}");
                exit_daemon(1);
            }
        }
    }

    tracing::info!(
        "shutdown requested — draining for at most {}ms",
        SHUTDOWN_DRAIN_TIMEOUT.as_millis()
    );
    let drained = tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, async {
        let server_result = (&mut server).await;
        let scheduler_result = scheduler_task.await;
        (server_result, scheduler_result)
    })
    .await;
    match drained {
        Ok((server_result, scheduler_result)) => {
            if let Err(error) = scheduler_result {
                tracing::warn!("scheduler join failed during shutdown: {error}");
            }
            server_result?;
            // `#[tokio::main]` waits indefinitely for already-started
            // `spawn_blocking` work while dropping the runtime. The HTTP
            // server and scheduler above are the daemon-owned drain boundary;
            // exit explicitly once both have stopped so shutdown remains
            // bounded even if an inference worker is still blocked.
            tracing::info!("graceful shutdown complete");
            exit_daemon(0);
        }
        Err(_) => {
            tracing::warn!(
                "graceful shutdown exceeded {}ms — forcing clean exit",
                SHUTDOWN_DRAIN_TIMEOUT.as_millis()
            );
            exit_daemon(0);
        }
    }
}
