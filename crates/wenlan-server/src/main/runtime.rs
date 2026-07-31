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
        let maintenance_for_ready = {
            let state = shared.read().await;
            state.maintenance_coordinator.clone()
        };
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
