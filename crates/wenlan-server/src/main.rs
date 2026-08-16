// SPDX-License-Identifier: Apache-2.0
//! Wenlan headless daemon — runs the memory server without Tauri.

mod cmd_backfill;
mod cmd_cutover;
#[path = "main/runtime.rs"]
mod runtime;
#[path = "main/startup.rs"]
mod startup;

struct DaemonDataLock {
    _file: std::fs::File,
}

impl DaemonDataLock {
    fn acquire(root: &std::path::Path, require_existing: bool) -> anyhow::Result<Self> {
        use sha2::Digest as _;

        let absolute_root = if root.is_absolute() {
            root.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|error| anyhow::anyhow!("resolve current directory: {error}"))?
                .join(root)
        };
        if require_existing && !absolute_root.is_dir() {
            anyhow::bail!(
                "repair-only startup requires an existing Wenlan data root: {}",
                root.display()
            );
        }
        if absolute_root.exists() && !absolute_root.is_dir() {
            anyhow::bail!("Wenlan data root is not a directory: {}", root.display());
        }

        let canonical_root = if absolute_root.is_dir() {
            std::fs::canonicalize(&absolute_root).map_err(|error| {
                anyhow::anyhow!("resolve Wenlan data root {}: {error}", root.display())
            })?
        } else {
            let parent = absolute_root.parent().ok_or_else(|| {
                anyhow::anyhow!("Wenlan data root has no parent: {}", root.display())
            })?;
            std::fs::create_dir_all(parent).map_err(|error| {
                anyhow::anyhow!(
                    "create Wenlan data-root parent {}: {error}",
                    parent.display()
                )
            })?;
            let canonical_parent = std::fs::canonicalize(parent).map_err(|error| {
                anyhow::anyhow!(
                    "resolve Wenlan data-root parent {}: {error}",
                    parent.display()
                )
            })?;
            canonical_parent.join(absolute_root.file_name().ok_or_else(|| {
                anyhow::anyhow!("Wenlan data root has no name: {}", root.display())
            })?)
        };
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"wenlan-daemon-data-root-lock-v1\0");
        #[cfg(windows)]
        hasher.update(canonical_root.to_string_lossy().to_lowercase().as_bytes());
        #[cfg(not(windows))]
        hasher.update(canonical_root.as_os_str().as_encoded_bytes());
        let lock_key = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        // Keep operational lock state in the canonical root's stable parent,
        // not in process-dependent TMPDIR and not inside the data being
        // verified. Lock files are intentionally never unlinked: removing one
        // can split contenders across two inodes.
        let lock_path = canonical_root
            .parent()
            .ok_or_else(|| {
                anyhow::anyhow!("Wenlan data root has no lock parent: {}", root.display())
            })?
            .join(format!(".wenlan-daemon-{lock_key}.lock"));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|error| {
                anyhow::anyhow!(
                    "open Wenlan data-root lock {}: {error}",
                    lock_path.display()
                )
            })?;
        fs2::FileExt::try_lock_exclusive(&file).map_err(|error| {
            anyhow::anyhow!(
                "Wenlan data root {} is already owned by another process: {error}",
                canonical_root.display()
            )
        })?;
        Ok(Self { _file: file })
    }
}

/// Resolve the bind address. Honors the `WENLAN_BIND_ADDR` env var when set
/// (e.g. inside Docker where the daemon must listen on `0.0.0.0`). Falls back
/// to the localhost-only address used by the macOS/native install path.
fn resolve_bind_addr(port: u16) -> String {
    wenlan_core::env_compat::var_compat("WENLAN_BIND_ADDR")
        .and_then(|v| v.into_string().ok())
        .unwrap_or_else(|| format!("127.0.0.1:{}", port))
}

fn resolve_startup_bind_addr(port: u16, startup_repair_claimed: bool) -> String {
    if startup_repair_claimed {
        format!("127.0.0.1:{port}")
    } else {
        resolve_bind_addr(port)
    }
}

fn resolve_startup_port(configured_port: u16, startup_repair_claimed: bool) -> anyhow::Result<u16> {
    if startup_repair_claimed && configured_port != 7878 {
        anyhow::bail!("repair-only startup requires canonical port 7878");
    }
    Ok(configured_port)
}

#[cfg(target_os = "macos")]
const SERVER_LOG_MAX_BYTES: usize = 10 * 1024 * 1024;
#[cfg(target_os = "macos")]
const SERVER_LOG_BACKUPS: usize = 5;
#[cfg(any(target_os = "macos", test))]
const BOOTSTRAP_LOG_MAX_BYTES: usize = 256 * 1024;
#[cfg(any(target_os = "macos", test))]
const BOOTSTRAP_LOG_BACKUPS: usize = 1;

fn resolve_wenlan_root() -> std::path::PathBuf {
    wenlan_core::env_compat::var_compat("WENLAN_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("wenlan")
        })
}

fn resolve_brief_status_root(wenlan_root: &std::path::Path) -> std::path::PathBuf {
    if wenlan_core::env_compat::var_compat("WENLAN_DATA_DIR").is_some() {
        return wenlan_root.join("sessions/_status");
    }

    dirs::home_dir()
        .map(|home| home.join(".wenlan/sessions/_status"))
        .unwrap_or_else(|| wenlan_root.join("sessions/_status"))
}

#[cfg(any(target_os = "macos", test))]
fn preflight_rotating_log_path(path: &std::path::Path) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "log path has no parent")
    })?;
    std::fs::create_dir_all(parent)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::other("log path is not a regular file"));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
fn new_server_log_writer(
    wenlan_root: &std::path::Path,
    max_bytes: usize,
    backups: usize,
) -> std::io::Result<file_rotate::FileRotate<file_rotate::suffix::AppendCount>> {
    let path = wenlan_root.join("logs/wenlan-server.log");
    preflight_rotating_log_path(&path)?;
    Ok(file_rotate::FileRotate::new(
        path,
        file_rotate::suffix::AppendCount::new(backups),
        file_rotate::ContentLimit::Bytes(max_bytes),
        file_rotate::compression::Compression::None,
        None,
    ))
}

#[cfg(any(target_os = "macos", test))]
fn new_bootstrap_log_writer(
    wenlan_root: &std::path::Path,
    max_bytes: usize,
    backups: usize,
) -> std::io::Result<file_rotate::FileRotate<file_rotate::suffix::AppendCount>> {
    let path = wenlan_root.join("logs/wenlan-server.bootstrap.log");
    preflight_rotating_log_path(&path)?;
    Ok(file_rotate::FileRotate::new(
        path,
        file_rotate::suffix::AppendCount::new(backups),
        file_rotate::ContentLimit::Bytes(max_bytes),
        file_rotate::compression::Compression::None,
        None,
    ))
}

#[cfg(any(target_os = "macos", test))]
fn write_bootstrap_message(
    wenlan_root: &std::path::Path,
    fallback_root: &std::path::Path,
    message: &str,
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write as _;

    let (mut writer, path) =
        match new_bootstrap_log_writer(wenlan_root, BOOTSTRAP_LOG_MAX_BYTES, BOOTSTRAP_LOG_BACKUPS)
        {
            Ok(writer) => (writer, wenlan_root.join("logs/wenlan-server.bootstrap.log")),
            Err(primary_error) => {
                eprintln!(
                    "Primary bootstrap log unavailable at {}: {primary_error}",
                    wenlan_root.display()
                );
                (
                    new_bootstrap_log_writer(
                        fallback_root,
                        BOOTSTRAP_LOG_MAX_BYTES,
                        BOOTSTRAP_LOG_BACKUPS,
                    )?,
                    fallback_root.join("logs/wenlan-server.bootstrap.log"),
                )
            }
        };
    writeln!(writer, "{message}")?;
    writer.flush()?;
    Ok(path)
}

fn report_bootstrap_error(wenlan_root: &std::path::Path, message: &str) {
    #[cfg(not(target_os = "macos"))]
    let _ = wenlan_root;

    eprintln!("{message}");
    tracing::error!("{message}");

    #[cfg(target_os = "macos")]
    if std::env::var_os("XPC_SERVICE_NAME").is_some() {
        let fallback_root = fallback_log_root();
        let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            write_bootstrap_message(wenlan_root, &fallback_root, message)
        }));
        if let Err(error) = write_result.unwrap_or_else(|_| {
            Err(std::io::Error::other(
                "bootstrap log writer panicked while reporting an error",
            ))
        }) {
            eprintln!("Failed to write bootstrap log: {error}");
        }
    }
}

fn install_bootstrap_panic_hook(wenlan_root: std::path::PathBuf) {
    std::panic::set_hook(Box::new(move |panic| {
        report_bootstrap_error(
            &wenlan_root,
            &format!("panic during daemon bootstrap: {panic}"),
        );
    }));
}

fn new_server_log_rate_limit() -> tracing_throttle::TracingRateLimitLayer {
    tracing_throttle::TracingRateLimitLayer::new()
}

#[cfg(target_os = "macos")]
fn fallback_log_root() -> std::path::PathBuf {
    dirs::home_dir()
        .map(|home| home.join("Library/Logs/com.wenlan.server-fallback"))
        .unwrap_or_else(|| std::env::temp_dir().join("wenlan-server-fallback"))
}

fn init_logging(wenlan_root: &std::path::Path) -> anyhow::Result<()> {
    use tracing_subscriber::prelude::*;

    #[cfg(not(target_os = "macos"))]
    let _ = wenlan_root;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,wenlan_core=info,wenlan_server=info".into());

    #[cfg(target_os = "macos")]
    {
        let writer = match new_server_log_writer(
            wenlan_root,
            SERVER_LOG_MAX_BYTES,
            SERVER_LOG_BACKUPS,
        ) {
            Ok(writer) => writer,
            Err(primary_error) => {
                let fallback_root = fallback_log_root();
                eprintln!(
                    "Primary daemon log unavailable at {}: {primary_error}; using {}",
                    wenlan_root.display(),
                    fallback_root.display()
                );
                new_server_log_writer(
                    &fallback_root,
                    SERVER_LOG_MAX_BYTES,
                    SERVER_LOG_BACKUPS,
                )
                .map_err(|fallback_error| {
                    anyhow::anyhow!(
                        "initialize rotating file logging: primary={primary_error}; fallback={fallback_error}"
                    )
                })?
            }
        };
        let writer = std::sync::Mutex::new(writer);
        let fmt = tracing_subscriber::fmt::layer()
            .with_writer(writer)
            .with_filter(new_server_log_rate_limit());
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt)
            .try_init()
            .map_err(|error| anyhow::anyhow!("initialize rotating file logging: {error}"))
    }

    #[cfg(not(target_os = "macos"))]
    {
        let fmt = tracing_subscriber::fmt::layer().with_filter(new_server_log_rate_limit());
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt)
            .try_init()
            .map_err(|error| anyhow::anyhow!("initialize console logging: {error}"))
    }
}

fn startup_projection_writes_allowed(repair_recovery_pending: bool) -> bool {
    !repair_recovery_pending
}

#[derive(Debug, Clone)]
struct StartupRepairClaim {
    manifest_id: String,
    manifest_digest: wenlan_types::repair::RepairDigest,
}

impl StartupRepairClaim {
    fn try_new(
        manifest_id: Option<String>,
        manifest_digest: Option<String>,
    ) -> anyhow::Result<Option<Self>> {
        match (manifest_id, manifest_digest) {
            (None, None) => Ok(None),
            (Some(manifest_id), Some(manifest_digest)) => {
                let manifest_digest = wenlan_types::repair::RepairDigest::parse(&manifest_digest)
                    .map_err(|error| {
                    anyhow::anyhow!("invalid startup repair digest: {error}")
                })?;
                Ok(Some(Self {
                    manifest_id,
                    manifest_digest,
                }))
            }
            _ => anyhow::bail!(
                "startup repair requires both --repair-manifest-id and --repair-manifest-digest"
            ),
        }
    }

    fn manifest_id(&self) -> &str {
        &self.manifest_id
    }

    fn apply_request(&self) -> anyhow::Result<wenlan_types::repair::ApplyRepairRequest> {
        let approval = format!(
            "apply repair {} {}",
            self.manifest_id,
            self.manifest_digest.as_str()
        );
        wenlan_types::repair::ApplyRepairRequest::try_new(
            self.manifest_id.clone(),
            self.manifest_digest.clone(),
            approval,
        )
        .map_err(|error| anyhow::anyhow!("invalid startup repair claim: {error}"))
    }
}

fn validate_startup_repair_claim(
    store: &wenlan_core::repair::RepairArtifactStore,
    claim: &StartupRepairClaim,
) -> anyhow::Result<()> {
    let manifest = store
        .load_manifest(claim.manifest_id())
        .map_err(|error| anyhow::anyhow!("load startup repair manifest: {error}"))?;
    if manifest.manifest_digest() != &claim.manifest_digest {
        anyhow::bail!("startup repair manifest digest mismatch");
    }
    Ok(())
}

fn stored_repair_apply_request(
    store: &wenlan_core::repair::RepairArtifactStore,
    manifest_id: &str,
) -> anyhow::Result<wenlan_types::repair::ApplyRepairRequest> {
    let manifest = store
        .load_manifest(manifest_id)
        .map_err(|error| anyhow::anyhow!("load pending repair manifest: {error}"))?;
    let digest = manifest.manifest_digest().clone();
    let approval = format!("apply repair {manifest_id} {}", digest.as_str());
    wenlan_types::repair::ApplyRepairRequest::try_new(manifest_id.to_string(), digest, approval)
        .map_err(|error| anyhow::anyhow!("invalid pending repair authority: {error}"))
}

fn select_startup_repair_fence(
    pending_manifest_ids: &[String],
    claim: Option<&StartupRepairClaim>,
) -> anyhow::Result<Option<String>> {
    let mut manifest_ids = std::collections::BTreeSet::new();
    manifest_ids.extend(pending_manifest_ids.iter().cloned());
    if let Some(claim) = claim {
        manifest_ids.insert(claim.manifest_id().to_string());
    }
    match manifest_ids.len() {
        0 => Ok(None),
        1 => Ok(manifest_ids.into_iter().next()),
        _ => anyhow::bail!(
            "multiple pending repairs require operator resolution before startup: {}",
            manifest_ids.into_iter().collect::<Vec<_>>().join(", ")
        ),
    }
}

fn optional_runtime_workers_allowed(startup_repair_claimed: bool) -> bool {
    !startup_repair_claimed
}

fn on_device_model_working_set_bytes(model: &wenlan_core::on_device_models::OnDeviceModel) -> u64 {
    (model.ram_required_gb * 1024.0 * 1024.0 * 1024.0).ceil() as u64
}

struct StartupModelLoadReservation(Arc<std::sync::atomic::AtomicBool>);

impl Drop for StartupModelLoadReservation {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Release);
    }
}

fn existing_daemon_may_satisfy_startup(startup_repair_claimed: bool) -> bool {
    !startup_repair_claimed
}

#[cfg(test)]
#[path = "bind_addr_tests.rs"]
mod bind_addr_tests;

// All other modules live in the library target (src/lib.rs) so that
// integration tests in tests/ can reference them as wenlan_server::<mod>.
use wenlan_server::{
    brief_files, ingest_batcher, lifecycle, router, scheduler,
    state::{ServerState, SharedState},
};

use clap::{Parser, Subcommand};
use std::{future::IntoFuture, io::Write, sync::Arc};
use tokio::sync::RwLock;

const SHUTDOWN_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1_500);

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "_exit"]
    fn unix_process_exit(status: std::ffi::c_int) -> !;
}

fn exit_daemon(code: i32) -> ! {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(unix)]
    {
        // SAFETY: The daemon-owned HTTP and scheduler tasks have already
        // drained. `_exit` terminates the process without running C `atexit`
        // handlers, which could otherwise tear down Metal globals while a
        // deliberately detached blocking inference worker still owns them.
        unsafe { unix_process_exit(code) }
    }
    #[cfg(not(unix))]
    {
        std::process::exit(code)
    }
}

#[cfg(unix)]
struct TerminationSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

#[cfg(unix)]
fn install_termination_signals() -> std::io::Result<TerminationSignals> {
    Ok(TerminationSignals {
        interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?,
        terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
    })
}

#[cfg(unix)]
impl TerminationSignals {
    async fn wait(mut self) {
        tokio::select! {
            _ = self.interrupt.recv() => {}
            _ = self.terminate.recv() => {}
        }
    }
}

#[cfg(windows)]
struct TerminationSignals {
    ctrl_c: tokio::signal::windows::CtrlC,
}

#[cfg(windows)]
fn install_termination_signals() -> std::io::Result<TerminationSignals> {
    Ok(TerminationSignals {
        ctrl_c: tokio::signal::windows::ctrl_c()?,
    })
}

#[cfg(windows)]
impl TerminationSignals {
    async fn wait(mut self) {
        let _ = self.ctrl_c.recv().await;
    }
}

#[cfg(not(any(unix, windows)))]
struct TerminationSignals;

#[cfg(not(any(unix, windows)))]
fn install_termination_signals() -> std::io::Result<TerminationSignals> {
    Ok(TerminationSignals)
}

#[cfg(not(any(unix, windows)))]
impl TerminationSignals {
    async fn wait(self) {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(debug_assertions)]
async fn wait_at_startup_signal_test_barrier() -> anyhow::Result<()> {
    let Some(root) = std::env::var_os("WENLAN_TEST_STARTUP_SIGNAL_BARRIER") else {
        return Ok(());
    };
    let root = std::path::PathBuf::from(root);
    let ready = root.join("ready");
    let release = root.join("release");
    std::fs::write(&ready, b"ready")?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !release.exists() {
        if std::time::Instant::now() >= deadline {
            anyhow::bail!("startup signal test barrier timed out");
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    Ok(())
}

/// Wenlan memory daemon — headless HTTP server.
#[derive(Parser)]
#[command(
    name = "wenlan-server",
    bin_name = "wenlan-server",
    version,
    about = "Wenlan headless HTTP daemon."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Override the data directory (for isolated dev/demo runs).
    /// When set, the daemon reads/writes the DB at `<dir>/memorydb/origin_memory.db`
    /// and config at `<dir>/config.json` instead of the default
    /// the platform data directory under `dirs::data_local_dir().join("wenlan/")`.
    /// macOS: `~/Library/Application Support/wenlan/`. Linux: `~/.local/share/wenlan/`. Windows: `%LOCALAPPDATA%\origin\`. Also honored via `WENLAN_DATA_DIR` env.
    #[arg(long, global = true)]
    data_dir: Option<std::path::PathBuf>,

    /// Override the HTTP port (default 7878). Useful when running a scratch
    /// daemon alongside the main one. Also honored via `WENLAN_PORT` env.
    #[arg(long, global = true)]
    port: Option<u16>,

    /// Internal repair-only startup claim. Both exact fields are required.
    #[arg(long, global = true, hide = true, requires = "repair_manifest_digest")]
    repair_manifest_id: Option<String>,

    /// Approved digest for the exact repair-only startup claim.
    #[arg(long, global = true, hide = true, requires = "repair_manifest_id")]
    repair_manifest_digest: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Internal maintenance: delete archived stale pages. Daemon must be stopped first.
    #[command(name = "backfill-stale-pages", hide = true)]
    BackfillStalePages {
        /// Print candidates without modifying the database.
        #[arg(long)]
        dry_run: bool,
    },
    /// Internal maintenance: advance the M5 truth cutover. Daemon must be stopped
    /// first. Moves judged-unsupported pages' Markdown into `<vault>/archive/`;
    /// `--apply` refuses without a matching dry run.
    #[command(name = "truth-cutover", hide = true)]
    TruthCutover {
        /// Generation to advance to. 1 is the first live generation.
        #[arg(long, default_value_t = 1)]
        generation: i64,
        /// Carry out the plan. Without it this is a dry run and records nothing
        /// but the plan digest.
        #[arg(long)]
        apply: bool,
    },
}

async fn run_daemon(startup_repair_claim: Option<StartupRepairClaim>) -> anyhow::Result<()> {
    let startup_repair_claimed = startup_repair_claim.is_some();
    // Register with the OS before binding or touching durable state. Creating
    // Tokio's platform signal streams installs the handlers synchronously, so
    // SIGTERM/CTRL_C received during startup is retained for the waiter below
    // instead of taking the process down through the platform default path.
    let termination_signals = install_termination_signals()
        .map_err(|error| anyhow::anyhow!("install termination signal handlers: {error}"))?;
    let wenlan_root = resolve_wenlan_root();
    let brief_status_root = resolve_brief_status_root(&wenlan_root);

    // Port (clap `--port`/`WENLAN_PORT` → env var set by main(); read here)
    let configured_port: u16 = wenlan_core::env_compat::var_compat("WENLAN_PORT")
        .and_then(|v| v.into_string().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(7878);
    let port = resolve_startup_port(configured_port, startup_repair_claimed)?;

    // Bind BEFORE touching the data dir. Losing the port race must be free:
    // under launchd KeepAlive, a retry loop that first runs full MemoryDB init
    // (schema/FTS writes + embedder load) hammers the live daemon's SQLite
    // file every ~10s — enough lock/CPU pressure to wedge the daemon that
    // actually owns the port.
    let addr = resolve_startup_bind_addr(port, startup_repair_claimed);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            if !existing_daemon_may_satisfy_startup(startup_repair_claimed) {
                return Err(anyhow::anyhow!(
                    "repair-only daemon failed to acquire {}: {}",
                    addr,
                    e
                ));
            }
            // Check if existing daemon is healthy
            eprintln!("Failed to bind {addr}: {e}");
            let url = format!("http://127.0.0.1:{}/api/health", port);
            // Bounded probe: a mute port-holder (accepts, never responds)
            // must not hang this process forever — under launchd KeepAlive
            // a hung loser also blocks the retry that would recover things.
            let probe = reqwest::Client::new()
                .get(&url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await;
            match probe {
                Ok(resp) if resp.status().is_success() => {
                    // Port already taken by a healthy daemon. If launchd is the
                    // parent (XPC_SERVICE_NAME set), exit non-zero so launchd
                    // retries after ThrottleInterval — otherwise launchd marks
                    // this attempt as a clean exit and refuses to respawn even
                    // after the winning daemon dies (KeepAlive.SuccessfulExit
                    // = false treats exit-0 as success). For sidecar invocation
                    // by the app, exit 0 is the right answer.
                    if std::env::var_os("XPC_SERVICE_NAME").is_some() {
                        report_bootstrap_error(
                            &wenlan_root,
                            &format!(
                                "Existing healthy daemon on port {port} — exiting 75 (launchd retry)"
                            ),
                        );
                        std::process::exit(75);
                    }
                    eprintln!("Existing healthy daemon on port {port} — exiting cleanly");
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

    init_logging(&wenlan_root)?;
    std::panic::set_hook(Box::new(|panic| {
        tracing::error!("panic: {panic}");
    }));
    tracing::info!("wenlan-server v{}", wenlan_core::version());

    #[cfg(debug_assertions)]
    wait_at_startup_signal_test_barrier().await?;
    // Data directory. `WENLAN_DATA_DIR` (set by `--data-dir` flag) overrides the
    // default, enabling isolated dev/demo runs (e.g. `--data-dir /tmp/wenlan-demo`).
    let data_dir = wenlan_root.join("memorydb");
    tracing::info!("Wenlan data root: {}", wenlan_root.display());
    let _data_root_lock = DaemonDataLock::acquire(&wenlan_root, startup_repair_claimed)?;

    let startup::PreparedStartupState {
        server_state,
        db_arc,
        repair_recovery_pending,
        config,
        reranker_cache_dir,
        deep_bgebase_pending,
    } = startup::prepare_startup_state(
        wenlan_root,
        data_dir,
        brief_status_root,
        &startup_repair_claim,
        startup_repair_claimed,
    )
    .await?;

    server_state.maintenance_coordinator.finish_recovery();

    let shared: SharedState = Arc::new(RwLock::new(server_state));

    runtime::register_optional_runtime_workers(
        shared.clone(),
        repair_recovery_pending,
        deep_bgebase_pending,
        reranker_cache_dir,
        config,
        db_arc,
    )
    .await;

    let shutdown = { shared.read().await.shutdown.clone() };
    let signal_shutdown = shutdown.clone();
    tokio::spawn(async move {
        termination_signals.wait().await;
        tracing::info!("termination signal received");
        signal_shutdown.request();
    });

    // Spawn the event-driven steep scheduler.
    // See docs/superpowers/specs/2026-04-12-event-driven-steep-triggers-design.md
    let scheduler_task = if optional_runtime_workers_allowed(repair_recovery_pending) {
        let write_signal = {
            let s = shared.read().await;
            s.write_signal.clone()
        };
        scheduler::spawn_scheduler(shared.clone(), write_signal, shutdown.subscribe())
    } else {
        tokio::spawn(async {})
    };

    runtime::serve_and_drain(
        repair_recovery_pending,
        shared,
        shutdown,
        scheduler_task,
        listener,
    )
    .await
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
    db: std::sync::Arc<wenlan_core::db::MemoryDB>,
    gate: wenlan_core::quality_gate::QualityGate,
    items: Vec<(
        wenlan_core::sources::RawDocument,
        usize,
        Option<wenlan_core::space_context::ResolvedWriteSpace>,
    )>,
) -> Vec<ingest_batcher::StoreOutcome> {
    use ingest_batcher::StoreOutcome;
    use wenlan_core::quality_gate::{GateResult, GateScores};

    type Survivor = (
        usize,
        wenlan_core::sources::RawDocument,
        usize,
        Option<wenlan_core::space_context::ResolvedWriteSpace>,
        Option<(String, f64)>,
    );

    if items.is_empty() {
        return vec![];
    }

    // Batch gate evaluate. One FastEmbed call, N vector queries, one pass.
    let docs: Vec<(&str, Option<&str>)> = items
        .iter()
        .map(|(document, _, _)| (document.content.as_str(), document.supersedes.as_deref()))
        .collect();
    let gate_results = match gate.evaluate_batch(&docs, &db).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("[ingest_batch_process] gate batch evaluate failed (fail closed), rejecting all: {e}");
            docs.iter()
                .map(|c| {
                    (
                        GateResult {
                            admitted: false,
                            reason: Some(
                                wenlan_core::quality_gate::RejectionReason::EmbeddingUnavailable(
                                    e.to_string(),
                                ),
                            ),
                            scores: GateScores {
                                content_type_pass: true,
                                novelty_score: None,
                                word_count: c.0.split_whitespace().count(),
                                pattern_matched: Some("embedding_unavailable".to_string()),
                                latency_ms: 0,
                            },
                            near_duplicate: None,
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
    let mut survivors: Vec<Survivor> = Vec::new();

    for (i, ((doc, chunks, write_space), (gate_result, similar_id))) in
        items.into_iter().zip(gate_results).enumerate()
    {
        if gate_result.admitted {
            survivors.push((i, doc, chunks, write_space, gate_result.near_duplicate));
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
        let docs = survivors
            .iter()
            .map(|(_, document, _, write_space, _)| (document.clone(), write_space.clone()))
            .collect();
        match db.upsert_documents_with_write_spaces(docs).await {
            Ok(_total) => {
                for (pos, _, chunks, _, near_duplicate) in &survivors {
                    outcomes[*pos] = Some(StoreOutcome::Stored {
                        chunks_created: *chunks,
                        near_duplicate: near_duplicate.clone(),
                    });
                }
            }
            Err(e) => {
                let validation = matches!(e, wenlan_core::WenlanError::Validation(_));
                let message = e.to_string();
                for (pos, _, _, _, _) in &survivors {
                    outcomes[*pos] = Some(if validation {
                        StoreOutcome::WriteSpaceInvalid(message.clone())
                    } else {
                        StoreOutcome::UpsertFailed(message.clone())
                    });
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

    // Propagate flags through env vars so both wenlan-server's own path logic
    // and wenlan-core's config loader (`wenlan_core::config::config_path`) see
    // the same values without plumbing a parameter through every call site.
    if let Some(ref dir) = cli.data_dir {
        std::env::set_var("WENLAN_DATA_DIR", dir);
    }
    if let Some(port) = cli.port {
        std::env::set_var("WENLAN_PORT", port.to_string());
    }

    // Resolving the path is read-only. Before rotating tracing is available,
    // a bounded bootstrap file keeps launchd failures and panics observable
    // even though the plist intentionally redirects stdout/stderr to /dev/null.
    let wenlan_root = resolve_wenlan_root();
    install_bootstrap_panic_hook(wenlan_root.clone());

    let result = async {
        let startup_repair_claim = StartupRepairClaim::try_new(
            cli.repair_manifest_id.clone(),
            cli.repair_manifest_digest.clone(),
        )?;

        if cli.command.is_some() && startup_repair_claim.is_some() {
            anyhow::bail!("startup repair claim is only valid when running the daemon");
        }

        match cli.command {
            Some(Command::BackfillStalePages { dry_run }) => cmd_backfill::run(dry_run).await,
            Some(Command::TruthCutover { generation, apply }) => {
                cmd_cutover::run(generation, apply).await
            }
            None => run_daemon(startup_repair_claim).await,
        }
    }
    .await;

    if let Err(error) = &result {
        report_bootstrap_error(
            &wenlan_root,
            &format!("wenlan-server terminated with an error: {error:#}"),
        );
    }
    result
}
