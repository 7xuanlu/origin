// SPDX-License-Identifier: AGPL-3.0-only
use regex::Regex;
use serde::Serialize;
use std::net::TcpListener;
use std::sync::LazyLock;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tokio::time::{sleep, timeout, Duration};

/// Port range for origin-mcp serve (high ports to avoid collisions).
pub const PORT_RANGE_START: u16 = 18080;

/// Relay URL for stable MCP endpoint.
const RELAY_URL: &str = "https://origin-relay.originmemory.workers.dev";

/// Get or create a persistent relay user ID.
/// Stored in ~/.config/origin-mcp/relay_id
fn get_or_create_relay_id() -> String {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".config")
        .join("origin-mcp")
        .join("relay_id");

    if let Ok(id) = std::fs::read_to_string(&path) {
        let id = id.trim().to_string();
        if !id.is_empty() {
            return id;
        }
    }

    // Generate a short random ID
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    let id = format!("u{:x}", hasher.finish())
        .chars()
        .take(12)
        .collect::<String>();

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, &id);
    id
}

/// Register the current tunnel URL with the relay for a stable MCP endpoint.
async fn register_with_relay(tunnel_url: &str) -> Option<String> {
    let relay_id = get_or_create_relay_id();
    let body = serde_json::json!({
        "user_id": relay_id,
        "tunnel_url": tunnel_url,
        "secret": relay_id, // simple shared secret
    });

    match reqwest::Client::new()
        .post(format!("{}/register", RELAY_URL))
        .json(&body)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let relay_mcp_url = format!("{}/{}/mcp", RELAY_URL, relay_id);
            log::warn!("[remote-access] Registered with relay: {}", relay_mcp_url);
            Some(relay_mcp_url)
        }
        Ok(resp) => {
            log::warn!(
                "[remote-access] Relay registration failed: {}",
                resp.status()
            );
            None
        }
        Err(e) => {
            log::warn!("[remote-access] Relay registration error: {}", e);
            None
        }
    }
}

/// Regex to extract cloudflared tunnel URL from stderr output.
static TUNNEL_URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https://[a-z0-9-]+\.trycloudflare\.com").unwrap());

/// Status of the remote access tunnel.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RemoteAccessStatus {
    Off,
    Starting,
    Connected {
        tunnel_url: String,
        token: String,
        /// Stable relay URL (if relay registration succeeded).
        relay_url: Option<String>,
    },
    Error {
        error: String,
    },
}

/// Runtime state for remote access — holds process handles and port.
pub struct RemoteAccessState {
    pub status: RemoteAccessStatus,
    pub mcp_child: Option<tauri_plugin_shell::process::CommandChild>,
    pub tunnel_child: Option<tauri_plugin_shell::process::CommandChild>,
    pub port: Option<u16>,
    /// When Cloudflare returned 429 for our most recent tunnel creation.
    /// Used by `tunnel_health_loop` to enforce a cooldown before burning another quick tunnel.
    pub last_rate_limit_at: Option<std::time::Instant>,
}

impl Default for RemoteAccessState {
    fn default() -> Self {
        Self {
            status: RemoteAccessStatus::Off,
            mcp_child: None,
            tunnel_child: None,
            port: None,
            last_rate_limit_at: None,
        }
    }
}

/// Cooldown after a Cloudflare 429 before we'll try creating another quick tunnel.
const RATE_LIMIT_COOLDOWN: Duration = Duration::from_secs(15 * 60);

/// Kill any orphaned origin-mcp processes on the remote access port range.
/// These accumulate when the Origin app restarts without cleanly shutting down
/// its child processes (the in-memory handles are lost on restart).
pub fn cleanup_orphaned_mcp() {
    let my_pid = std::process::id();
    for port in PORT_RANGE_START..=PORT_RANGE_START + 3 {
        // Use lsof to find the PID holding this port
        let output = std::process::Command::new("lsof")
            .args(["-i", &format!(":{}", port), "-t", "-sTCP:LISTEN"])
            .output();
        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                if let Ok(pid) = line.trim().parse::<u32>() {
                    // Don't kill ourselves
                    if pid == my_pid {
                        continue;
                    }
                    log::warn!(
                        "[remote-access] Killing orphaned process {} on port {}",
                        pid,
                        port
                    );
                    // SIGTERM first, SIGKILL fallback — origin-mcp may ignore SIGTERM
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .output();
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    // Force kill if still alive
                    let _ = std::process::Command::new("kill")
                        .args(["-9", &pid.to_string()])
                        .output();
                }
            }
        }
    }
    // Brief pause to let ports release
    std::thread::sleep(std::time::Duration::from_millis(200));
}

/// Find an available port in the range 18080-18083.
pub fn find_available_port() -> Option<u16> {
    (PORT_RANGE_START..=PORT_RANGE_START + 3)
        .find(|&port| TcpListener::bind(("127.0.0.1", port)).is_ok())
}

/// Parse the tunnel URL from cloudflared's stderr output.
pub fn parse_tunnel_url(stderr: &str) -> Option<String> {
    TUNNEL_URL_RE.find(stderr).map(|m| m.as_str().to_string())
}

/// Token file path for origin-mcp authentication.
/// origin-mcp stores tokens at ~/.config/origin-mcp/token (XDG convention),
/// NOT ~/Library/Application Support/ (macOS convention from dirs::config_dir).
fn token_file_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".config")
        .join("origin-mcp")
        .join("token")
}

/// Read the bearer token from disk.
pub fn read_token() -> Result<String, String> {
    let path = token_file_path();
    std::fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Failed to read token at {}: {}", path.display(), e))
}

/// Start the remote access tunnel (origin-mcp serve + cloudflared).
/// Async — emits `remote-access-status` events as state changes.
/// Called from Tauri command handler — the command returns `Starting`
/// immediately and this runs in a background task.
pub fn toggle_on(
    app_handle: tauri::AppHandle,
    is_retry: bool,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(toggle_on_inner(app_handle, if is_retry { 1 } else { 0 }))
}

/// Start with explicit retry count (used by monitor auto-restart).
fn toggle_on_with_retries(
    app_handle: tauri::AppHandle,
    retry_count: u32,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(toggle_on_inner(app_handle, retry_count))
}

async fn toggle_on_inner(app_handle: tauri::AppHandle, retry_count: u32) {
    use tauri::Emitter;

    // Guard: don't start if already starting or connected
    {
        let state =
            app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
        let app_state = state.read().await;
        let ra = app_state.remote_access.lock().await;
        match &ra.status {
            RemoteAccessStatus::Starting | RemoteAccessStatus::Connected { .. } => {
                log::warn!("[remote-access] Already active, skipping duplicate toggle_on");
                return;
            }
            _ => {}
        }
    }

    let _ = app_handle.emit("remote-access-status", &RemoteAccessStatus::Starting);

    let result = start_tunnel(&app_handle).await;

    match result {
        Ok((tunnel_url, token, mcp_child, tunnel_child, port, mcp_rx, tunnel_rx)) => {
            // Register with relay for a stable URL
            let relay_url = register_with_relay(&tunnel_url).await;

            let status = RemoteAccessStatus::Connected {
                tunnel_url: tunnel_url.clone(),
                token: token.clone(),
                relay_url: relay_url.clone(),
            };
            let _ = app_handle.emit("remote-access-status", &status);

            // Store process handles in state
            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let mut ra = app_state.remote_access.lock().await;
            ra.status = status;
            ra.mcp_child = Some(mcp_child);
            ra.tunnel_child = Some(tunnel_child);
            ra.port = Some(port);
            // We just successfully created a tunnel → we're not currently
            // rate-limited, so clear any stale 429 stamp from an earlier attempt.
            ra.last_rate_limit_at = None;
            drop(ra);
            drop(app_state);

            // Spawn background monitor for crash recovery
            let handle_for_monitor = app_handle.clone();
            tauri::async_runtime::spawn(async move {
                monitor_processes(handle_for_monitor, mcp_rx, tunnel_rx, retry_count).await;
            });

            // Spawn periodic tunnel health check (detects broken tunnels after sleep)
            let handle_for_health = app_handle.clone();
            let health_tunnel_url = tunnel_url.clone();
            let health_retry_count = retry_count;
            tauri::async_runtime::spawn(async move {
                tunnel_health_loop(handle_for_health, health_tunnel_url, health_retry_count).await;
            });
        }
        Err(e) => {
            log::error!("[remote-access] toggle_on failed: {}", e);
            let status = RemoteAccessStatus::Error { error: e.clone() };
            let _ = app_handle.emit("remote-access-status", &status);

            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let mut ra = app_state.remote_access.lock().await;
            ra.status = status;
            // Stamp rate-limit time so `tunnel_health_loop` enforces cooldown before
            // burning another quick tunnel. Only when the error is an actual 429.
            if e.contains("429") || e.contains("rate limit") {
                ra.last_rate_limit_at = Some(std::time::Instant::now());
                log::warn!(
                    "[remote-access] Cloudflare 429 recorded — reconnect cooldown active for {} min",
                    RATE_LIMIT_COOLDOWN.as_secs() / 60
                );
            }
        }
    }
}

/// Read cloudflared event stream until we find a tunnel URL.
/// Returns Ok(url) on success, Err(message) on known errors (e.g. rate limit).
async fn parse_tunnel_url_from_events(
    rx: &mut tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>,
) -> Result<String, Option<String>> {
    let mut accumulated = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            tauri_plugin_shell::process::CommandEvent::Stderr(line) => {
                let text = String::from_utf8_lossy(&line);
                accumulated.push_str(&text);
                // Detect rate limit before waiting for URL
                if accumulated.contains("429") || accumulated.contains("Too Many Requests") {
                    return Err(Some(
                        "Cloudflare rate limit (429) — too many quick tunnels. Will auto-retry."
                            .to_string(),
                    ));
                }
                if accumulated.contains("failed to unmarshal") {
                    return Err(Some(
                        "Cloudflare tunnel creation failed. Will auto-retry.".to_string(),
                    ));
                }
                if let Some(url) = parse_tunnel_url(&accumulated) {
                    return Ok(url);
                }
            }
            tauri_plugin_shell::process::CommandEvent::Stdout(line) => {
                let text = String::from_utf8_lossy(&line);
                accumulated.push_str(&text);
                if let Some(url) = parse_tunnel_url(&accumulated) {
                    return Ok(url);
                }
            }
            _ => {}
        }
    }
    // Channel closed before a URL appeared. Surface what cloudflared printed so
    // future silent-exit cases aren't opaque. (Used to drop `accumulated` on the floor.)
    let trimmed = accumulated.trim();
    if trimmed.is_empty() {
        log::error!(
            "[remote-access] cloudflared exited with no stderr/stdout output — check the sidecar binary is valid and executable"
        );
    } else {
        log::error!(
            "[remote-access] cloudflared exited without producing a URL. Captured output:\n{}",
            trimmed
        );
    }
    Err(None)
}

/// Maximum MCP-only restart attempts before falling back to full restart.
const MAX_MCP_RETRIES: u32 = 3;
/// Maximum full restart attempts (creates new tunnel — costs Cloudflare quota).
const MAX_TUNNEL_RETRIES: u32 = 3;

/// Spawn origin-mcp serve on a given port (without cloudflared).
/// Reusable for both initial start and MCP-only restarts.
async fn spawn_mcp(
    app_handle: &tauri::AppHandle,
    port: u16,
) -> Result<
    (
        tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>,
        tauri_plugin_shell::process::CommandChild,
    ),
    String,
> {
    log::warn!("[remote-access] spawning origin-mcp serve on port {}", port);
    let (mcp_rx, mcp_child) = app_handle
        .shell()
        .sidecar("origin-mcp")
        .map_err(|e| format!("origin-mcp sidecar not found: {}", e))?
        .args([
            "serve",
            "--port",
            &port.to_string(),
            "--no-auth",
            "--agent-name",
            "remote-mcp",
            "--allowed-origins",
            "https://claude.ai,https://chatgpt.com",
        ])
        .spawn()
        .map_err(|e| format!("Failed to spawn origin-mcp serve: {}", e))?;

    let health_url = format!("http://127.0.0.1:{}/health", port);
    let health_ok = timeout(Duration::from_secs(5), async {
        loop {
            if reqwest::get(&health_url)
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                return true;
            }
            sleep(Duration::from_millis(200)).await;
        }
    })
    .await
    .unwrap_or(false);

    if !health_ok {
        let _ = mcp_child.kill();
        return Err("origin-mcp serve failed to start (health check timeout).".to_string());
    }

    Ok((mcp_rx, mcp_child))
}

/// Which sidecar process exited.
enum ExitedProcess {
    Mcp(String),
    Tunnel(String),
}

/// Monitor sidecar processes for unexpected exits.
/// - If origin-mcp exits: respawn only origin-mcp (tunnel stays alive, no Cloudflare cost).
/// - If cloudflared exits: full restart needed (new tunnel URL required).
pub async fn monitor_processes(
    app_handle: tauri::AppHandle,
    mut mcp_rx: tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>,
    mut tunnel_rx: tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>,
    tunnel_retry_count: u32,
) {
    use tauri::Emitter;

    let mut mcp_retries = 0u32;

    loop {
        let exited = tokio::select! {
            event = wait_for_exit(&mut mcp_rx) => ExitedProcess::Mcp(event),
            event = wait_for_exit(&mut tunnel_rx) => ExitedProcess::Tunnel(event),
        };

        match exited {
            ExitedProcess::Mcp(reason) => {
                log::warn!(
                    "[remote-access] origin-mcp exited: {} — attempting MCP-only restart",
                    reason
                );

                // Get port from state, kill old mcp handle
                let port = {
                    let state = app_handle
                        .state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
                    let app_state = state.read().await;
                    let mut ra = app_state.remote_access.lock().await;
                    if let Some(child) = ra.mcp_child.take() {
                        let _ = child.kill();
                    }
                    ra.port
                };

                let Some(port) = port else {
                    // Port cleared by toggle_off (e.g. tunnel_health_loop reconnect) — exit quietly
                    log::info!(
                        "[remote-access] No port in state — another reconnect is handling recovery"
                    );
                    return;
                };

                if mcp_retries >= MAX_MCP_RETRIES {
                    log::warn!("[remote-access] {} MCP-only retries exhausted — falling back to full restart", MAX_MCP_RETRIES);
                    break; // Fall through to full restart below
                }

                let delay = 5u64 * (mcp_retries as u64 + 1); // 5s, 10s, 15s
                log::warn!(
                    "[remote-access] MCP-only restart in {}s (attempt {}/{})",
                    delay,
                    mcp_retries + 1,
                    MAX_MCP_RETRIES
                );
                sleep(Duration::from_secs(delay)).await;

                match spawn_mcp(&app_handle, port).await {
                    Ok((new_rx, new_child)) => {
                        // Store new child in state
                        let state = app_handle
                            .state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
                        let app_state = state.read().await;
                        let mut ra = app_state.remote_access.lock().await;
                        ra.mcp_child = Some(new_child);
                        drop(ra);
                        drop(app_state);

                        mcp_rx = new_rx;
                        mcp_retries = 0; // Reset on success — only count consecutive failures
                        log::warn!("[remote-access] MCP-only restart succeeded — resuming monitor");
                        continue; // Loop back to watch both processes
                    }
                    Err(e) => {
                        log::error!("[remote-access] MCP-only restart failed: {} — falling back to full restart", e);
                        mcp_retries += 1;
                        continue; // Try again if retries remain
                    }
                }
            }
            ExitedProcess::Tunnel(reason) => {
                log::warn!(
                    "[remote-access] cloudflared exited: {} — full restart needed",
                    reason
                );
                break; // Fall through to full restart below
            }
        }
    }

    // Full restart path — kills both processes, creates new tunnel
    toggle_off(&app_handle).await;

    if tunnel_retry_count >= MAX_TUNNEL_RETRIES {
        log::error!(
            "[remote-access] {} full retries exhausted — giving up",
            MAX_TUNNEL_RETRIES
        );
        let status = RemoteAccessStatus::Error {
            error: format!(
                "{} full retries failed — please toggle manually.",
                MAX_TUNNEL_RETRIES
            ),
        };
        let _ = app_handle.emit("remote-access-status", &status);
        let state =
            app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
        let app_state = state.read().await;
        let mut ra = app_state.remote_access.lock().await;
        ra.status = status;
        return;
    }

    // Exponential backoff: 30s, 60s, 120s — gentle on Cloudflare quick tunnel limits
    let delay_secs = 30u64 << tunnel_retry_count;
    let attempt = tunnel_retry_count + 1;

    let status = RemoteAccessStatus::Error {
        error: format!(
            "Full restart in {}s (attempt {}/{})...",
            delay_secs, attempt, MAX_TUNNEL_RETRIES
        ),
    };
    let _ = app_handle.emit("remote-access-status", &status);

    sleep(Duration::from_secs(delay_secs)).await;

    log::warn!(
        "[remote-access] Full restart attempt {}/{}",
        attempt,
        MAX_TUNNEL_RETRIES
    );
    toggle_on_with_retries(app_handle, attempt).await;
}

/// Wait for a process exit or error event on the command event receiver.
async fn wait_for_exit(
    rx: &mut tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>,
) -> String {
    while let Some(event) = rx.recv().await {
        match event {
            tauri_plugin_shell::process::CommandEvent::Terminated(payload) => {
                return format!(
                    "terminated (code: {:?}, signal: {:?})",
                    payload.code, payload.signal
                );
            }
            tauri_plugin_shell::process::CommandEvent::Error(err) => {
                return format!("error: {}", err);
            }
            _ => {} // Ignore stdout/stderr during monitoring
        }
    }
    "channel closed".to_string()
}

/// Periodic tunnel health check + auto-reconnect.
/// Runs every 30s; detects broken tunnels after Mac sleep, ISP outages, etc.
///
/// `reconnect_count` carries across reconnect attempts — shared budget with
/// `monitor_processes` crash recovery, capped by `MAX_TUNNEL_RETRIES`. Each
/// reconnect increments the counter to apply exponential backoff (30/60/120s).
///
/// Safeguards against the 429-storm this used to cause:
/// 1. **Never-healthy tunnels don't trigger reconnect.** A brand-new tunnel that
///    has never returned a successful /health is almost certainly a propagation
///    or routing problem — recreating it burns Cloudflare quota without fixing
///    anything. Wait it out instead.
/// 2. **Exponential backoff between reconnects.** No more "reconnect every 90s".
/// 3. **Respects `last_rate_limit_at` cooldown.** If a prior attempt hit 429,
///    reconnects are deferred for `RATE_LIMIT_COOLDOWN`.
/// 4. **Retry cap.** After `MAX_TUNNEL_RETRIES` health-driven reconnects, the
///    loop gives up and transitions to `Status::Error`. User must re-enable.
async fn tunnel_health_loop(
    app_handle: tauri::AppHandle,
    tunnel_url: String,
    reconnect_count: u32,
) {
    use tauri::Emitter;

    let health_url = format!("{}/health", tunnel_url);
    let mut consecutive_failures = 0u32;
    let mut ever_succeeded = false;

    loop {
        sleep(Duration::from_secs(30)).await;

        // Check if we're still in Connected state
        {
            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let ra = app_state.remote_access.lock().await;
            if !matches!(ra.status, RemoteAccessStatus::Connected { .. }) {
                // No longer connected — stop health checking
                return;
            }
        }

        // Ping the tunnel URL. Log the actual error on failure so we can tell DNS
        // vs TLS vs 5xx vs timeout apart without speculation.
        let check = timeout(Duration::from_secs(5), reqwest::get(&health_url)).await;
        let ok = match check {
            Ok(Ok(resp)) if resp.status().is_success() => true,
            Ok(Ok(resp)) => {
                log::warn!(
                    "[remote-access] Tunnel /health returned non-2xx: {}",
                    resp.status()
                );
                false
            }
            Ok(Err(e)) => {
                // `reqwest::Error`'s Display includes the chain via its error chain,
                // but not always the root cause. Use the debug form too so we see
                // whether this is dns/tls/connect/body.
                log::warn!(
                    "[remote-access] Tunnel /health request error: {} ({:?})",
                    e,
                    e
                );
                false
            }
            Err(_) => {
                log::warn!("[remote-access] Tunnel /health request timed out after 5s");
                false
            }
        };

        if ok {
            if !ever_succeeded {
                log::info!(
                    "[remote-access] Tunnel /health first-success — reconnect tripwire armed"
                );
            }
            consecutive_failures = 0;
            ever_succeeded = true;
            continue;
        }

        consecutive_failures += 1;
        log::warn!(
            "[remote-access] Tunnel health check failed ({}/3)",
            consecutive_failures
        );

        // Fix 3: Don't burn a fresh tunnel on a never-healthy one. If /health has
        // never returned success, this is a propagation or routing problem and
        // recreating the tunnel won't help — it's the same routing. Keep polling
        // and wait for it to come alive (or user intervention).
        if !ever_succeeded {
            if consecutive_failures == 3 {
                log::warn!(
                    "[remote-access] Tunnel has never been reachable after 3 checks. Not triggering reconnect (would just burn Cloudflare quota). Continuing to poll — tunnel may still come up."
                );
            }
            // Reset the counter so we don't spam the warning but keep polling.
            if consecutive_failures >= 3 {
                consecutive_failures = 0;
            }
            continue;
        }

        if consecutive_failures < 3 {
            continue;
        }

        // --- Reconnect decision ---

        // Fix 1a: retry cap.
        if reconnect_count >= MAX_TUNNEL_RETRIES {
            log::error!(
                "[remote-access] Max health-driven reconnects reached ({}). Giving up — re-enable manually.",
                MAX_TUNNEL_RETRIES
            );
            // toggle_off emits `Off` internally. We emit `Error` AFTER it so the
            // frontend's last event is the error message, not Off.
            toggle_off(&app_handle).await;
            let error = format!(
                "Tunnel kept dropping after {} reconnect attempts. Re-enable to try again.",
                MAX_TUNNEL_RETRIES
            );
            let status = RemoteAccessStatus::Error { error };
            let _ = app_handle.emit("remote-access-status", &status);
            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let mut ra = app_state.remote_access.lock().await;
            ra.status = status;
            return;
        }

        // Fix 1b: respect 429 cooldown.
        let in_cooldown = {
            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let ra = app_state.remote_access.lock().await;
            ra.last_rate_limit_at
                .map(|t| t.elapsed() < RATE_LIMIT_COOLDOWN)
                .unwrap_or(false)
        };
        if in_cooldown {
            log::warn!(
                "[remote-access] In 429 cooldown — skipping reconnect, will re-check in 30s"
            );
            // Reset the counter so we don't hammer the cooldown check at max log rate.
            consecutive_failures = 0;
            continue;
        }

        // Fix 1c: exponential backoff matching monitor_processes (30/60/120s).
        let delay_secs = 30u64 << reconnect_count.min(2);
        let next_count = reconnect_count + 1;
        log::warn!(
            "[remote-access] Tunnel unreachable — reconnecting in {}s (attempt {}/{})",
            delay_secs,
            next_count,
            MAX_TUNNEL_RETRIES
        );
        toggle_off(&app_handle).await;
        sleep(Duration::from_secs(delay_secs)).await;
        toggle_on_with_retries(app_handle, next_count).await;
        return; // New toggle_on spawns its own health loop with next_count.
    }
}

async fn start_tunnel(
    app_handle: &tauri::AppHandle,
) -> Result<
    (
        String,                                                                 // tunnel_url
        String,                                                                 // token
        tauri_plugin_shell::process::CommandChild,                              // mcp_child
        tauri_plugin_shell::process::CommandChild,                              // tunnel_child
        u16,                                                                    // port
        tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>, // mcp_rx
        tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>, // tunnel_rx
    ),
    String,
> {
    // 0. Clean up any orphaned origin-mcp processes from previous app sessions
    cleanup_orphaned_mcp();

    // 1. Find available port
    let port = find_available_port()
        .ok_or_else(|| "All remote access ports (18080-18083) are in use.".to_string())?;

    // 2. Ensure token exists
    let token_path = token_file_path();
    log::warn!(
        "[remote-access] token_path={}, exists={}",
        token_path.display(),
        token_path.exists()
    );
    if !token_path.exists() {
        let shell = app_handle.shell();
        log::warn!("[remote-access] generating token via sidecar...");
        let cmd = shell
            .sidecar("origin-mcp")
            .map_err(|e| format!("origin-mcp sidecar not found: {}", e))?;
        log::warn!("[remote-access] sidecar command created, executing token generate...");
        let output = cmd
            .args(["token", "generate"])
            .output()
            .await
            .map_err(|e| format!("Failed to generate token (exec error: {})", e))?;
        log::warn!(
            "[remote-access] token generate exit status: {:?}, stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        if !output.status.success() {
            return Err(format!(
                "Token generation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    let token = read_token()?;

    // 3. Spawn origin-mcp serve (with health check)
    let (mcp_rx, mcp_child) = spawn_mcp(app_handle, port).await?;

    // 4. Spawn cloudflared tunnel
    let (mut tunnel_rx, tunnel_child) = app_handle
        .shell()
        .sidecar("cloudflared")
        .map_err(|e| format!("cloudflared sidecar not found: {}", e))?
        .args(["tunnel", "--url", &format!("http://localhost:{}", port)])
        .spawn()
        .map_err(|e| format!("Failed to spawn cloudflared: {}", e))?;

    // 5. Parse tunnel URL from cloudflared output (it logs to stderr)
    let tunnel_url = match timeout(
        Duration::from_secs(20),
        parse_tunnel_url_from_events(&mut tunnel_rx),
    )
    .await
    {
        Ok(Ok(url)) => url,
        Ok(Err(Some(msg))) => {
            // Known error (rate limit, etc.) — kill mcp and propagate
            let _ = mcp_child.kill();
            let _ = tunnel_child.kill();
            return Err(msg);
        }
        Ok(Err(None)) => {
            let _ = mcp_child.kill();
            return Err("cloudflared exited without producing a tunnel URL.".to_string());
        }
        Err(_) => {
            let _ = mcp_child.kill();
            let _ = tunnel_child.kill();
            return Err("Failed to get tunnel URL from cloudflared (timeout).".to_string());
        }
    };

    // 6. Best-effort tunnel verification — don't kill on failure.
    // Quick tunnels can take 10-20s to become fully reachable after URL is printed.
    let verify_url = format!("{}/health", tunnel_url);
    let verify_ok = timeout(Duration::from_secs(10), async {
        sleep(Duration::from_secs(1)).await;
        for _ in 0..4 {
            if reqwest::get(&verify_url)
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                return true;
            }
            sleep(Duration::from_secs(2)).await;
        }
        false
    })
    .await
    .unwrap_or(false);

    if !verify_ok {
        log::warn!(
            "[remote-access] Tunnel verification didn't complete in time — tunnel may still work"
        );
    }

    Ok((
        tunnel_url,
        token,
        mcp_child,
        tunnel_child,
        port,
        mcp_rx,
        tunnel_rx,
    ))
}

/// Stop the remote access tunnel — kill both processes.
pub async fn toggle_off(app_handle: &tauri::AppHandle) {
    use tauri::Emitter;

    let state = app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
    let app_state = state.read().await;
    let mut ra = app_state.remote_access.lock().await;

    if let Some(child) = ra.tunnel_child.take() {
        let _ = child.kill();
    }
    if let Some(child) = ra.mcp_child.take() {
        let _ = child.kill();
    }

    ra.status = RemoteAccessStatus::Off;
    ra.port = None;
    drop(ra);
    drop(app_state);

    // Sweep for any orphaned processes the handles didn't cover
    cleanup_orphaned_mcp();

    let _ = app_handle.emit("remote-access-status", &RemoteAccessStatus::Off);
}

/// Rotate the bearer token — generate a new token then kill the old origin-mcp.
/// The crash recovery monitor detects the MCP exit and respawns only origin-mcp
/// (tunnel stays alive). The new instance reads the updated token from disk.
pub async fn rotate_token(app_handle: &tauri::AppHandle) -> Result<String, String> {
    // 1. Generate new token first (before killing anything)
    let shell = app_handle.shell();
    let output = shell
        .sidecar("origin-mcp")
        .map_err(|e| format!("origin-mcp sidecar not found: {}", e))?
        .args(["token", "generate"])
        .output()
        .await
        .map_err(|e| format!("Failed to generate token: {}", e))?;
    if !output.status.success() {
        return Err(format!(
            "Token generation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let token = read_token()?;

    // 2. Kill old origin-mcp — crash recovery monitor will detect this,
    //    kill cloudflared, wait 2s, and restart everything with the new token
    //    (already written to disk above).
    {
        let state =
            app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
        let app_state = state.read().await;
        let mut ra = app_state.remote_access.lock().await;
        if let Some(child) = ra.mcp_child.take() {
            let _ = child.kill();
        }
    }

    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_default_is_off() {
        let state = RemoteAccessState::default();
        assert!(matches!(state.status, RemoteAccessStatus::Off));
        assert!(state.port.is_none());
    }

    #[test]
    fn test_status_serializes_correctly() {
        let status = RemoteAccessStatus::Off;
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"off\""));

        let status = RemoteAccessStatus::Starting;
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"starting\""));

        let status = RemoteAccessStatus::Connected {
            tunnel_url: "https://test.trycloudflare.com".to_string(),
            token: "abc123".to_string(),
            relay_url: None,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"connected\""));
        assert!(json.contains("https://test.trycloudflare.com"));

        let status = RemoteAccessStatus::Error {
            error: "something broke".to_string(),
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("something broke"));
    }

    #[test]
    fn test_find_available_port_returns_first_available() {
        let port = find_available_port();
        assert!(port.is_some());
        let p = port.unwrap();
        assert!((PORT_RANGE_START..=PORT_RANGE_START + 3).contains(&p));
    }

    #[test]
    fn test_parse_tunnel_url_from_cloudflared_stderr() {
        let stderr = r#"2026-03-27T10:00:00Z INF +--------------------------------------------------------------------------------------------+
2026-03-27T10:00:00Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2026-03-27T10:00:00Z INF |  https://calm-river-abc123.trycloudflare.com                                               |
2026-03-27T10:00:00Z INF +--------------------------------------------------------------------------------------------+"#;
        let url = parse_tunnel_url(stderr);
        assert_eq!(
            url,
            Some("https://calm-river-abc123.trycloudflare.com".to_string())
        );
    }

    #[test]
    fn test_parse_tunnel_url_no_match() {
        let stderr = "some random log output without a URL";
        let url = parse_tunnel_url(stderr);
        assert!(url.is_none());
    }

    #[test]
    fn test_parse_tunnel_url_real_cloudflared_format() {
        let stderr = "2026-03-27 INF Registered tunnel connection\nhttps://my-tunnel-xyz.trycloudflare.com\n2026-03-27 INF Connection established";
        let url = parse_tunnel_url(stderr);
        assert_eq!(
            url,
            Some("https://my-tunnel-xyz.trycloudflare.com".to_string())
        );
    }

    #[test]
    fn test_parse_tunnel_url_multiple_urls_returns_first() {
        let stderr =
            "https://first-tunnel.trycloudflare.com\nhttps://second-tunnel.trycloudflare.com";
        let url = parse_tunnel_url(stderr);
        assert_eq!(
            url,
            Some("https://first-tunnel.trycloudflare.com".to_string())
        );
    }
}
