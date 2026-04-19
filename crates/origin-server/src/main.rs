// SPDX-License-Identifier: Apache-2.0
//! Origin headless daemon — runs the memory server without Tauri.

mod config_routes;
mod error;
mod import_routes;
mod ingest_batcher;
mod ingest_routes;
mod knowledge_routes;
mod memory_routes;
mod onboarding_routes;
mod router;
mod routes;
mod scheduler;
mod source_routes;
mod state;
mod websocket;

use clap::{Parser, Subcommand};
use state::{ServerState, SharedState};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Origin memory daemon — headless HTTP server.
#[derive(Parser)]
#[command(name = "origin-server", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Override the data directory (for isolated dev/demo runs).
    /// When set, the daemon reads/writes the DB at `<dir>/memorydb/origin_memory.db`
    /// and config at `<dir>/config.json` instead of the default
    /// `~/Library/Application Support/origin/`. Also honored via `ORIGIN_DATA_DIR` env.
    #[arg(long, global = true)]
    data_dir: Option<std::path::PathBuf>,

    /// Override the HTTP port (default 7878). Useful when running a scratch
    /// daemon alongside the main one. Also honored via `ORIGIN_PORT` env.
    #[arg(long, global = true)]
    port: Option<u16>,
}

#[derive(Subcommand)]
enum Command {
    /// Install as a macOS LaunchAgent (auto-start on login).
    Install,
    /// Uninstall the LaunchAgent.
    Uninstall,
    /// Show daemon status (launchd + health check).
    Status,
}

const PLIST_LABEL: &str = "com.origin.server";
const PLIST_TEMPLATE: &str = include_str!("../resources/com.origin.server.plist");

fn plist_path() -> std::path::PathBuf {
    dirs::home_dir()
        .expect("HOME not set")
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", PLIST_LABEL))
}

fn log_dir() -> std::path::PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("origin")
        .join("logs")
}

fn current_exe_path() -> String {
    std::env::current_exe()
        .expect("cannot determine own path")
        .to_string_lossy()
        .to_string()
}

fn cmd_install() -> anyhow::Result<()> {
    let plist = plist_path();
    let log_path = log_dir();

    // Ensure log directory exists
    std::fs::create_dir_all(&log_path)?;

    // Substitute placeholders
    let content = PLIST_TEMPLATE
        .replace("__ORIGIN_SERVER_PATH__", &current_exe_path())
        .replace("__LOG_PATH__", &log_path.to_string_lossy());

    // Ensure LaunchAgents directory exists
    if let Some(parent) = plist.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Unload first if already installed (ignore errors)
    if plist.exists() {
        let _ = std::process::Command::new("launchctl")
            .args(["unload", &plist.to_string_lossy()])
            .output();
    }

    std::fs::write(&plist, content)?;
    println!("Wrote {}", plist.display());

    let output = std::process::Command::new("launchctl")
        .args(["load", &plist.to_string_lossy()])
        .output()?;

    if output.status.success() {
        println!(
            "Loaded {} — daemon will start automatically on login",
            PLIST_LABEL
        );
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("launchctl load failed: {}", stderr);
    }

    Ok(())
}

fn cmd_uninstall() -> anyhow::Result<()> {
    let plist = plist_path();

    if !plist.exists() {
        println!("{} is not installed", PLIST_LABEL);
        return Ok(());
    }

    let output = std::process::Command::new("launchctl")
        .args(["unload", &plist.to_string_lossy()])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("launchctl unload warning: {}", stderr);
    }

    std::fs::remove_file(&plist)?;
    println!(
        "Removed {} — daemon will no longer auto-start",
        plist.display()
    );

    Ok(())
}

async fn cmd_status() -> anyhow::Result<()> {
    let plist = plist_path();

    // Check plist exists
    if plist.exists() {
        println!("Plist: {} (installed)", plist.display());
    } else {
        println!("Plist: not installed");
    }

    // Check launchctl
    let output = std::process::Command::new("launchctl")
        .args(["list"])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let registered = stdout.lines().any(|line| line.contains(PLIST_LABEL));
    println!(
        "Launchd: {}",
        if registered {
            "registered"
        } else {
            "not registered"
        }
    );

    // Health check
    let port: u16 = std::env::var("ORIGIN_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7878);
    let url = format!("http://127.0.0.1:{}/api/health", port);
    match reqwest::get(&url).await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.text().await.unwrap_or_default();
            println!("Health: ok (port {})\n{}", port, body);
        }
        Ok(resp) => {
            println!("Health: unhealthy (status {})", resp.status());
        }
        Err(e) => {
            println!("Health: not reachable ({})", e);
        }
    }

    Ok(())
}

async fn run_daemon() -> anyhow::Result<()> {
    // Logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,origin_core=info,origin_server=info".into()),
        )
        .init();

    tracing::info!("origin-server v{}", origin_core::version());

    // Port (clap `--port`/`ORIGIN_PORT` → env var set by main(); read here)
    let port: u16 = std::env::var("ORIGIN_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7878);

    // Data directory. `ORIGIN_DATA_DIR` (set by `--data-dir` flag) overrides the
    // default, enabling isolated dev/demo runs (e.g. `--data-dir /tmp/origin-demo`).
    let origin_root = std::env::var_os("ORIGIN_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("origin")
        });
    let data_dir = origin_root.join("memorydb");
    tracing::info!("Origin data root: {}", origin_root.display());

    // Build state
    let mut server_state = ServerState::new();

    // Init MemoryDB
    let emitter: Arc<dyn origin_core::events::EventEmitter> = Arc::new(origin_core::NoopEmitter);
    tracing::info!("Initializing MemoryDB at {}", data_dir.display());
    let db = origin_core::db::MemoryDB::new(&data_dir, emitter).await?;
    let db_arc = Arc::new(db);
    server_state.db = Some(db_arc.clone());

    // One-time backfill: if the knowledge directory is empty but the DB has
    // active concepts, write them all to disk. Handles the case where
    // concepts were created before KnowledgeWriter was wired up, or via a
    // code path that bypasses the writer.
    //
    // We gate on a `.origin/.backfill-attempted` marker file (created on
    // first attempt regardless of outcome) so this block only runs once per
    // daemon install. Without the marker, a persistent write_concept
    // failure — e.g. permission error on the destination directory — would
    // re-trigger a full DB scan + write attempt on every single startup.
    {
        let knowledge_path = origin_core::config::load_config().knowledge_path_or_default();
        let marker_path = knowledge_path.join(".origin").join(".backfill-attempted");

        let already_attempted = marker_path.exists();
        let has_md_files = !already_attempted
            && knowledge_path.exists()
            && std::fs::read_dir(&knowledge_path)
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.path()
                            .extension()
                            .and_then(|s| s.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("md"))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);

        if !already_attempted && !has_md_files {
            match db_arc.list_concepts("active", 10_000, 0).await {
                Ok(concepts) if !concepts.is_empty() => {
                    tracing::info!(
                        "[backfill] knowledge dir empty; writing {} concepts to {}",
                        concepts.len(),
                        knowledge_path.display()
                    );
                    let writer = origin_core::export::knowledge::KnowledgeWriter::new(
                        knowledge_path.clone(),
                    );
                    let mut written = 0usize;
                    let mut failed = 0usize;
                    for concept in &concepts {
                        match writer.write_concept(concept) {
                            Ok(_) => written += 1,
                            Err(e) => {
                                tracing::warn!(
                                    "[backfill] write_concept failed for {}: {}",
                                    concept.id,
                                    e
                                );
                                failed += 1;
                            }
                        }
                    }
                    tracing::info!("[backfill] wrote {} concepts, {} failed", written, failed);

                    // Create the marker file so we don't re-run the
                    // backfill on every subsequent startup — even if every
                    // write_concept above failed. The user can delete
                    // `.origin/.backfill-attempted` to force a retry.
                    if let Some(parent) = marker_path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if let Err(e) = std::fs::write(&marker_path, "") {
                        tracing::warn!(
                            "[backfill] failed to write marker {}: {}",
                            marker_path.display(),
                            e
                        );
                    }
                }
                Ok(_) => {
                    // DB has no concepts yet — nothing to backfill. Don't
                    // create the marker; the next startup after concepts
                    // exist should retry.
                }
                Err(e) => {
                    tracing::warn!("[backfill] list_concepts failed: {}", e);
                }
            }
        }
    }

    // Load intelligence config
    server_state.prompts = origin_core::prompts::PromptRegistry::load(
        &origin_core::prompts::PromptRegistry::override_dir(),
    );
    server_state.tuning =
        origin_core::tuning::TuningConfig::load(&origin_core::tuning::TuningConfig::config_path());
    server_state.quality_gate =
        origin_core::quality_gate::QualityGate::new(server_state.tuning.gate.clone());

    // Load API LLM providers if configured
    let config = origin_core::config::load_config();
    if let Some(ref key) = config.anthropic_api_key {
        if !key.is_empty() {
            let routine_model = config
                .routine_model
                .clone()
                .unwrap_or_else(|| origin_core::llm_provider::DEFAULT_ROUTINE_MODEL.to_string());
            let provider = origin_core::llm_provider::ApiProvider::new(key.clone(), routine_model);
            server_state.api_llm = Some(Arc::new(provider));
            tracing::info!("API LLM provider initialized (routine)");

            let synthesis_model = config
                .synthesis_model
                .clone()
                .unwrap_or_else(|| "claude-sonnet-4-6".to_string());
            let provider =
                origin_core::llm_provider::ApiProvider::new(key.clone(), synthesis_model);
            server_state.synthesis_llm = Some(Arc::new(provider));
            tracing::info!("Synthesis LLM provider initialized");
        }
    }

    // Load external LLM provider if configured
    if let (Some(ref endpoint), Some(ref model)) =
        (&config.external_llm_endpoint, &config.external_llm_model)
    {
        if !endpoint.is_empty() && !model.is_empty() {
            let provider = origin_core::llm_provider::OpenAICompatibleProvider::new(
                endpoint.clone(),
                model.clone(),
            );
            server_state.external_llm = Some(Arc::new(provider));
            tracing::info!("External LLM provider initialized from config");
        }
    }

    // Load space store
    server_state.space_store = origin_core::spaces::load_spaces();

    // Spawn the ingest coalescer. HTTP `/api/memory/store` handlers submit
    // fully-built RawDocuments + pre-computed chunk counts; the coalescer
    // runs the full ingest pipeline (batched quality gate, partition,
    // upsert survivors) per flush window. This amortizes both the FastEmbed
    // invocation (one batched call per flush for gate's novelty check) AND
    // the libSQL transaction (one per flush for the survivors) across
    // concurrent writes.
    //
    // See `crates/origin-server/src/ingest_batcher.rs` for the design and
    // contract tests.
    {
        let db_for_batcher = db_arc.clone();
        let gate_for_batcher = server_state.quality_gate.clone();
        let process: crate::ingest_batcher::BatchProcessFn = Arc::new(
            move |items: Vec<(origin_core::sources::RawDocument, usize)>| {
                let db = db_for_batcher.clone();
                let gate = gate_for_batcher.clone();
                Box::pin(async move { ingest_batch_process(db, gate, items).await })
            },
        );
        server_state.ingest_batcher = Some(crate::ingest_batcher::IngestBatcher::spawn(
            process,
            crate::ingest_batcher::BatcherConfig::default(),
        ));
    }

    let shared: SharedState = Arc::new(RwLock::new(server_state));

    // Initialize on-device LLM in the background if a model is already cached.
    // This intentionally does NOT trigger a download — users opt in explicitly
    // via the settings UI (POST /api/on-device-model/download).
    {
        let shared_for_llm = shared.clone();
        let on_device_id = config.on_device_model.clone();
        tokio::spawn(async move {
            let result = tokio::task::spawn_blocking(move || {
                let model =
                    origin_core::on_device_models::resolve_or_default(on_device_id.as_deref());
                if !origin_core::on_device_models::is_cached(model) {
                    tracing::info!(
                        "[on-device] model {} not cached, skipping init (use settings to download)",
                        model.id
                    );
                    return Ok::<
                        Option<(Arc<dyn origin_core::llm_provider::LlmProvider>, String)>,
                        origin_core::error::OriginError,
                    >(None);
                }
                let provider =
                    origin_core::llm_provider::OnDeviceProvider::new_with_model(Some(model.id))?;
                let arc: Arc<dyn origin_core::llm_provider::LlmProvider> = Arc::new(provider);
                Ok(Some((arc, model.id.to_string())))
            })
            .await;

            match result {
                Ok(Ok(Some((provider, model_id)))) => {
                    let mut state = shared_for_llm.write().await;
                    state.llm = Some(provider);
                    state.loaded_on_device_model = Some(model_id.clone());
                    tracing::info!("[on-device] model {} loaded and available", model_id);
                }
                Ok(Ok(None)) => {} // Not cached — already logged
                Ok(Err(e)) => tracing::error!("[on-device] init failed: {}", e),
                Err(e) => tracing::error!("[on-device] init task panicked: {}", e),
            }
        });
    }

    // Register the LLM-readiness hook so that the `intelligence-ready`
    // onboarding milestone fires the first time any LLM provider successfully
    // serves traffic. `mark_llm_ready` is a one-shot per process, so this hook
    // runs at most once regardless of which provider fires first.
    //
    // The on-device `llm-provider-worker` (`crates/origin-core/src/llm_provider.rs:142`)
    // runs on a `std::thread`, not a Tokio task — GPU inference is blocking
    // and would starve the async runtime. When it calls `mark_llm_ready()`
    // from that thread, our hook fires synchronously on a thread with no
    // Tokio reactor in thread-local context. Bare `tokio::spawn(...)` would
    // then panic: "there is no reactor running, must be called from the
    // context of a Tokio 1.x runtime" — exactly what killed the worker on
    // 2026-04-16. Capture a `Handle` here (we are inside `#[tokio::main]`)
    // and use `handle.spawn(...)` from the closure instead.
    {
        let db_for_ready = db_arc.clone();
        let emitter_for_ready: Arc<dyn origin_core::events::EventEmitter> =
            Arc::new(origin_core::events::NoopEmitter);
        let handle = tokio::runtime::Handle::current();
        let hook: origin_core::llm_provider::ReadinessHook = Arc::new(move || {
            let db = db_for_ready.clone();
            let emitter = emitter_for_ready.clone();
            handle.spawn(async move {
                let ev = origin_core::onboarding::MilestoneEvaluator::new(&db, emitter);
                if let Err(e) = ev.check_after_llm_ready().await {
                    tracing::warn!(?e, "onboarding: check_after_llm_ready failed");
                }
            });
        });
        let _ = origin_core::llm_provider::LLM_READINESS_HOOK.set(hook);
    }

    // Spawn the event-driven steep scheduler.
    // See docs/superpowers/specs/2026-04-12-event-driven-steep-triggers-design.md
    {
        let write_signal = {
            let s = shared.read().await;
            s.write_signal.clone()
        };
        scheduler::spawn_scheduler(shared.clone(), write_signal);
    }

    // Build router
    let app = router::build_router(shared);

    // Bind
    let addr = format!("127.0.0.1:{}", port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => {
            tracing::info!("Listening on http://{}", addr);
            l
        }
        Err(e) => {
            // Check if existing daemon is healthy
            tracing::warn!("Failed to bind {}: {}", addr, e);
            let url = format!("http://127.0.0.1:{}/api/health", port);
            match reqwest::get(&url).await {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("Existing healthy daemon on port {} — exiting cleanly", port);
                    return Ok(());
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Port {} in use and no healthy daemon",
                        port
                    ));
                }
            }
        }
    };

    // Serve
    axum::serve(listener, app).await?;

    Ok(())
}

/// Batch processor invoked by the ingest coalescer per flush. Runs the
/// full per-request ingest pipeline — quality gate evaluate (batched so
/// one FastEmbed call covers every survivor's novelty check) → partition
/// admitted vs rejected → upsert survivors in a single transaction →
/// emit per-doc outcomes in input order.
///
/// Fail-open policy on gate infrastructure failure: if the batched gate
/// evaluator itself returns an error (DB unreachable, embedding panicked
/// inside FastEmbed, etc.), every doc is admitted rather than rejected —
/// matches `QualityGate::evaluate`'s per-doc behavior, which also fails
/// open rather than wedging stores behind the gate.
async fn ingest_batch_process(
    db: std::sync::Arc<origin_core::db::MemoryDB>,
    gate: origin_core::quality_gate::QualityGate,
    items: Vec<(origin_core::sources::RawDocument, usize)>,
) -> Vec<ingest_batcher::StoreOutcome> {
    use ingest_batcher::StoreOutcome;
    use origin_core::quality_gate::{GateResult, GateScores};

    if items.is_empty() {
        return vec![];
    }

    // Batch gate evaluate. One FastEmbed call, N vector queries, one pass.
    let contents: Vec<&str> = items.iter().map(|(d, _)| d.content.as_str()).collect();
    let gate_results = match gate.evaluate_batch(&contents, &db).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("[ingest_batch_process] gate batch evaluate failed, admitting all: {e}");
            contents
                .iter()
                .map(|c| {
                    (
                        GateResult {
                            admitted: true,
                            reason: None,
                            scores: GateScores {
                                content_type_pass: true,
                                novelty_score: None,
                                word_count: c.split_whitespace().count(),
                                pattern_matched: None,
                                latency_ms: 0,
                            },
                        },
                        None,
                    )
                })
                .collect()
        }
    };

    let n = items.len();
    let mut outcomes: Vec<Option<StoreOutcome>> = (0..n).map(|_| None).collect();
    // (original_position, doc, chunks_predicted) for every admitted doc.
    let mut survivors: Vec<(usize, origin_core::sources::RawDocument, usize)> = Vec::new();

    for (i, ((doc, chunks), (gate_result, similar_id))) in
        items.into_iter().zip(gate_results).enumerate()
    {
        if gate_result.admitted {
            survivors.push((i, doc, chunks));
        } else {
            let (reason_str, detail_str) = gate_result
                .reason
                .as_ref()
                .map(|r| (r.as_str().to_string(), r.detail()))
                .unwrap_or_else(|| ("unknown".to_string(), "rejected".to_string()));
            outcomes[i] = Some(StoreOutcome::GateRejected {
                reason: reason_str,
                detail: detail_str,
                similar_to: similar_id,
            });
        }
    }

    if !survivors.is_empty() {
        let docs: Vec<origin_core::sources::RawDocument> =
            survivors.iter().map(|(_, d, _)| d.clone()).collect();
        match db.upsert_documents(docs).await {
            Ok(_total) => {
                for (pos, _, chunks) in &survivors {
                    outcomes[*pos] = Some(StoreOutcome::Stored {
                        chunks_created: *chunks,
                    });
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for (pos, _, _) in &survivors {
                    outcomes[*pos] = Some(StoreOutcome::UpsertFailed(msg.clone()));
                }
            }
        }
    }

    // Any `None` slot means the item was neither admitted nor rejected —
    // shouldn't happen, but backfill defensively.
    outcomes
        .into_iter()
        .map(|o| o.unwrap_or(StoreOutcome::UpsertFailed("missing outcome slot".into())))
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Propagate flags through env vars so both origin-server's own path logic
    // and origin-core's config loader (`origin_core::config::config_path`) see
    // the same values without plumbing a parameter through every call site.
    if let Some(ref dir) = cli.data_dir {
        std::env::set_var("ORIGIN_DATA_DIR", dir);
    }
    if let Some(port) = cli.port {
        std::env::set_var("ORIGIN_PORT", port.to_string());
    }

    match cli.command {
        Some(Command::Install) => cmd_install(),
        Some(Command::Uninstall) => cmd_uninstall(),
        Some(Command::Status) => cmd_status().await,
        None => run_daemon().await,
    }
}
