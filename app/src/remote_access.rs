// SPDX-License-Identifier: AGPL-3.0-only
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;
use tokio::time::{sleep, timeout, Duration};

const MCP_SIDECAR_NAME: &str = "wenlan-mcp";

/// Port range for wenlan-mcp serve (high ports to avoid collisions).
pub const PORT_RANGE_START: u16 = 18080;
const PORT_RANGE_LEN: u16 = 4;

fn port_range_start() -> u16 {
    #[cfg(debug_assertions)]
    if let Some(port) = std::env::var("WENLAN_DEV_REMOTE_PORT_START")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|port| *port <= u16::MAX - (PORT_RANGE_LEN - 1))
    {
        return port;
    }
    PORT_RANGE_START
}

/// Relay URL for stable MCP endpoint.
// Intentionally still the legacy Origin relay. Do not rename this constant
// until a Wenlan relay endpoint exists and existing relay IDs have a migration
// strategy.
const RELAY_URL: &str = "https://origin-relay.originmemory.workers.dev";

/// Get or create a persistent relay user ID.
/// Stored in ~/.config/wenlan-mcp/relay_id
fn get_or_create_relay_id() -> Result<String, String> {
    let path = relay_id_path();

    match std::fs::read_to_string(&path) {
        Ok(id) => {
            let id = id.trim().to_string();
            if !id.is_empty() {
                if let Some(parent) = path.parent() {
                    restrict_private_dir(parent, "relay_id")?;
                }
                restrict_private_file(&path, "relay_id")?;
                return Ok(id);
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(format!(
                "Failed to read relay ID at {}: {}",
                path.display(),
                e
            ));
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

    write_private_file(&path, id.as_bytes(), "relay_id")?;
    Ok(id)
}

/// Register the current tunnel URL with the relay for a stable MCP endpoint.
async fn register_with_relay(tunnel_url: &str) -> Option<String> {
    let relay_id = match get_or_create_relay_id() {
        Ok(id) => id,
        Err(e) => {
            log::warn!("[remote-access] Relay ID unavailable: {}", e);
            return None;
        }
    };
    let body = serde_json::json!({
        "user_id": &relay_id,
        "tunnel_url": tunnel_url,
        "secret": &relay_id, // simple shared secret
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

#[derive(Debug, Clone, Deserialize, Eq, PartialEq, Serialize)]
struct CloudflaredOwner {
    pid: u32,
    port: u16,
    identity: String,
}

fn cloudflared_owner_authorizes_signal(
    owner: &CloudflaredOwner,
    expected: Option<&CloudflaredOwner>,
    range_start: u16,
    live_identity: Option<&str>,
) -> bool {
    expected.is_none_or(|expected| expected == owner)
        && (range_start..=range_start + (PORT_RANGE_LEN - 1)).contains(&owner.port)
        && live_identity == Some(owner.identity.as_str())
}

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

fn generation_is_current(generation: u64, expected: u64) -> bool {
    generation == expected
}

fn try_begin_start(
    status: &mut RemoteAccessStatus,
    generation: &mut u64,
    expected_generation: Option<u64>,
) -> Option<u64> {
    if matches!(
        status,
        RemoteAccessStatus::Starting | RemoteAccessStatus::Connected { .. }
    ) {
        return None;
    }
    if let Some(expected) = expected_generation {
        if !generation_is_current(*generation, expected) {
            return None;
        }
    } else {
        *generation = generation.wrapping_add(1);
    }
    *status = RemoteAccessStatus::Starting;
    Some(*generation)
}

fn mark_off(status: &mut RemoteAccessStatus, generation: &mut u64) -> u64 {
    *generation = generation.wrapping_add(1);
    *status = RemoteAccessStatus::Off;
    *generation
}

async fn remote_generation_is_current(
    app_handle: &tauri::AppHandle,
    expected_generation: u64,
) -> bool {
    let state = app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
    let app_state = state.read().await;
    let ra = app_state.remote_access.lock().await;
    generation_is_current(ra.generation, expected_generation)
}

async fn wait_for_generation_change(app_handle: &tauri::AppHandle, expected_generation: u64) {
    while remote_generation_is_current(app_handle, expected_generation).await {
        sleep(Duration::from_millis(50)).await;
    }
}

/// Runtime state for remote access — holds process handles and port.
pub struct RemoteAccessState {
    pub status: RemoteAccessStatus,
    pub mcp_child: Option<tauri_plugin_shell::process::CommandChild>,
    pub tunnel_child: Option<tauri_plugin_shell::process::CommandChild>,
    pub port: Option<u16>,
    /// Invalidates stale start/reconnect tasks when the user turns access off.
    pub generation: u64,
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
            generation: 0,
            last_rate_limit_at: None,
        }
    }
}

/// Cooldown after a Cloudflare 429 before we'll try creating another quick tunnel.
const RATE_LIMIT_COOLDOWN: Duration = Duration::from_secs(15 * 60);

/// Kill any orphaned wenlan-mcp processes on the remote access port range.
/// These accumulate when the Wenlan app restarts without cleanly shutting down
/// its child processes (the in-memory handles are lost on restart).
pub fn cleanup_orphaned_mcp() {
    let my_pid = std::process::id();
    let range_start = port_range_start();
    for port in range_start..=range_start + (PORT_RANGE_LEN - 1) {
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
                    let Some(process_identity) = wenlan_mcp_process_identity(pid, port) else {
                        log::warn!(
                            "[remote-access] refusing to kill non-wenlan-mcp listener {} on port {}",
                            pid,
                            port
                        );
                        continue;
                    };
                    log::warn!(
                        "[remote-access] Killing orphaned process {} on port {}",
                        pid,
                        port
                    );
                    // SIGTERM first, SIGKILL fallback — wenlan-mcp may ignore SIGTERM
                    let _ = std::process::Command::new("kill")
                        .arg(pid.to_string())
                        .output();
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    // Force kill only if the PID still identifies the exact
                    // remote-access child we inspected before SIGTERM.
                    if wenlan_mcp_process_identity(pid, port).as_deref()
                        == Some(process_identity.as_str())
                    {
                        let _ = std::process::Command::new("kill")
                            .args(["-9", &pid.to_string()])
                            .output();
                    }
                }
            }
        }
    }
    // Brief pause to let ports release
    std::thread::sleep(std::time::Duration::from_millis(200));
}

fn cloudflared_owner_path() -> PathBuf {
    crate::identity_paths::mcp_config_dir().join("cloudflared-owner.json")
}

fn cloudflared_owner_lock_path(path: &Path) -> PathBuf {
    let mut lock_path = path.as_os_str().to_os_string();
    lock_path.push(".lock");
    PathBuf::from(lock_path)
}

struct CloudflaredOwnerLock<'a> {
    path: &'a Path,
    _file: std::fs::File,
}

impl CloudflaredOwnerLock<'_> {
    fn read(&self) -> Result<Option<CloudflaredOwner>, String> {
        let contents = match std::fs::read(self.path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(format!(
                    "Failed to read cloudflared ownership receipt at {}: {error}",
                    self.path.display()
                ));
            }
        };
        restrict_private_file(self.path, "cloudflared ownership receipt")?;
        serde_json::from_slice(&contents)
            .map(Some)
            .map_err(|error| {
                format!(
                    "Invalid cloudflared ownership receipt at {}: {error}",
                    self.path.display()
                )
            })
    }

    fn write(&self, owner: &CloudflaredOwner) -> Result<(), String> {
        use std::io::Write;

        let contents = serde_json::to_vec(owner).map_err(|error| {
            format!("Failed to serialize cloudflared ownership receipt: {error}")
        })?;
        let parent = self.path.parent().ok_or_else(|| {
            format!(
                "Cloudflared ownership receipt has no parent: {}",
                self.path.display()
            )
        })?;
        let mut staged = tempfile::NamedTempFile::new_in(parent).map_err(|error| {
            format!(
                "Failed to stage cloudflared ownership receipt beside {}: {error}",
                self.path.display()
            )
        })?;
        staged.write_all(&contents).map_err(|error| {
            format!(
                "Failed to write staged cloudflared ownership receipt for {}: {error}",
                self.path.display()
            )
        })?;
        staged.as_file().sync_all().map_err(|error| {
            format!(
                "Failed to sync staged cloudflared ownership receipt for {}: {error}",
                self.path.display()
            )
        })?;
        staged.persist(self.path).map_err(|error| {
            format!(
                "Failed to atomically replace cloudflared ownership receipt at {}: {}",
                self.path.display(),
                error.error
            )
        })?;
        restrict_private_file(self.path, "cloudflared ownership receipt")
    }

    fn remove_if_matches(&self, expected: &CloudflaredOwner) -> Result<(), String> {
        if self.read()?.as_ref() != Some(expected) {
            return Ok(());
        }
        match std::fs::remove_file(self.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!(
                "Failed to remove cloudflared ownership receipt at {}: {error}",
                self.path.display()
            )),
        }
    }
}

fn with_cloudflared_owner_lock<T>(
    path: &Path,
    operation: impl FnOnce(&CloudflaredOwnerLock<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let lock_path = cloudflared_owner_lock_path(path);
    prepare_private_parent(&lock_path, "cloudflared ownership lock")?;
    #[cfg(unix)]
    let lock_file = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&lock_path)
    };
    #[cfg(not(unix))]
    let lock_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path);
    let lock_file = lock_file.map_err(|error| {
        format!(
            "Failed to open cloudflared ownership lock at {}: {error}",
            lock_path.display()
        )
    })?;
    restrict_private_file(&lock_path, "cloudflared ownership lock")?;
    lock_file.lock().map_err(|error| {
        format!(
            "Failed to lock cloudflared ownership receipt at {}: {error}",
            lock_path.display()
        )
    })?;
    operation(&CloudflaredOwnerLock {
        path,
        _file: lock_file,
    })
}

#[cfg(test)]
fn read_cloudflared_owner_at(path: &Path) -> Result<Option<CloudflaredOwner>, String> {
    with_cloudflared_owner_lock(path, |receipt| receipt.read())
}

fn write_cloudflared_owner_at(path: &Path, owner: &CloudflaredOwner) -> Result<(), String> {
    with_cloudflared_owner_lock(path, |receipt| receipt.write(owner))
}

fn record_cloudflared_owner(pid: u32, port: u16) -> Result<CloudflaredOwner, String> {
    let identity = cloudflared_process_identity(pid, port).ok_or_else(|| {
        format!("Spawned cloudflared PID {pid} does not match the expected tunnel for port {port}")
    })?;
    let owner = CloudflaredOwner {
        pid,
        port,
        identity,
    };
    write_cloudflared_owner_at(&cloudflared_owner_path(), &owner)?;
    Ok(owner)
}

fn cleanup_owned_cloudflared(expected: Option<&CloudflaredOwner>) {
    let path = cloudflared_owner_path();
    if let Err(error) = with_cloudflared_owner_lock(&path, |receipt| {
        let owner = match receipt.read()? {
            Some(owner) => owner,
            None => return Ok(()),
        };
        let range_start = port_range_start();
        let live_identity = cloudflared_process_identity(owner.pid, owner.port);
        if !cloudflared_owner_authorizes_signal(
            &owner,
            expected,
            range_start,
            live_identity.as_deref(),
        ) {
            log::warn!(
                "[remote-access] refusing to signal cloudflared PID {} because its ownership receipt, selected range, or live identity does not match",
                owner.pid
            );
            if expected.is_none_or(|expected| expected == &owner) {
                receipt.remove_if_matches(&owner)?;
            }
            return Ok(());
        }
        log::warn!(
            "[remote-access] Killing owned cloudflared process {} for port {}",
            owner.pid,
            owner.port
        );
        let _ = std::process::Command::new("kill")
            .arg(owner.pid.to_string())
            .output();
        std::thread::sleep(std::time::Duration::from_millis(100));
        if cloudflared_process_identity(owner.pid, owner.port).as_deref()
            == Some(owner.identity.as_str())
        {
            let _ = std::process::Command::new("kill")
                .args(["-9", &owner.pid.to_string()])
                .output();
        }
        receipt.remove_if_matches(&owner)
    }) {
        log::warn!("[remote-access] {error}; refusing cloudflared cleanup");
    }
}

fn is_expected_remote_mcp_command(command: &str, port: u16) -> bool {
    let args: Vec<_> = command.split_whitespace().collect();
    let Some(executable) = args.first() else {
        return false;
    };
    let file_name = Path::new(executable)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let expected_executable = matches!(
        file_name,
        "wenlan-mcp"
            | "wenlan-mcp.exe"
            | "wenlan-mcp-aarch64-apple-darwin"
            | "wenlan-mcp-x86_64-unknown-linux-gnu"
            | "wenlan-mcp-aarch64-unknown-linux-gnu"
            | "wenlan-mcp-x86_64-pc-windows-msvc.exe"
    );
    let has_pair = |flag: &str, value: &str| {
        args.windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value)
    };
    expected_executable
        && args.contains(&"serve")
        && has_pair("--port", &port.to_string())
        && has_pair("--agent-name", "remote-mcp")
}

fn wenlan_mcp_process_identity(pid: u32, port: u16) -> Option<String> {
    let pid = pid.to_string();
    let command_output = std::process::Command::new("ps")
        .args(["-ww", "-p", &pid, "-o", "command="])
        .output()
        .ok()?;
    let command = String::from_utf8(command_output.stdout).ok()?;
    if !is_expected_remote_mcp_command(command.trim(), port) {
        return None;
    }
    let started_output = std::process::Command::new("ps")
        .args(["-p", &pid, "-o", "lstart="])
        .output()
        .ok()?;
    let started = String::from_utf8(started_output.stdout).ok()?;
    Some(format!("{}\n{}", started.trim(), command.trim()))
}

fn is_expected_remote_tunnel_command(command: &str, port: u16) -> bool {
    let args: Vec<_> = command.split_whitespace().collect();
    let Some(executable) = args.first() else {
        return false;
    };
    let file_name = Path::new(executable)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let expected_executable = matches!(
        file_name,
        "cloudflared"
            | "cloudflared.exe"
            | "cloudflared-aarch64-apple-darwin"
            | "cloudflared-x86_64-apple-darwin"
            | "cloudflared-x86_64-unknown-linux-gnu"
            | "cloudflared-aarch64-unknown-linux-gnu"
            | "cloudflared-x86_64-pc-windows-msvc.exe"
    );
    let expected_url = format!("http://localhost:{port}");
    expected_executable
        && args.get(1) == Some(&"tunnel")
        && args
            .windows(2)
            .any(|pair| pair[0] == "--url" && pair[1] == expected_url)
}

fn cloudflared_process_identity(pid: u32, port: u16) -> Option<String> {
    let pid = pid.to_string();
    let command_output = std::process::Command::new("ps")
        .args(["-ww", "-p", &pid, "-o", "command="])
        .output()
        .ok()?;
    let command = String::from_utf8(command_output.stdout).ok()?;
    if !is_expected_remote_tunnel_command(command.trim(), port) {
        return None;
    }
    let started_output = std::process::Command::new("ps")
        .args(["-p", &pid, "-o", "lstart="])
        .output()
        .ok()?;
    let started = String::from_utf8(started_output.stdout).ok()?;
    Some(format!("{}\n{}", started.trim(), command.trim()))
}

fn terminate_owned_cloudflared(
    child: tauri_plugin_shell::process::CommandChild,
    owner: &CloudflaredOwner,
) {
    let _ = child.kill();
    cleanup_owned_cloudflared(Some(owner));
}

fn listener_pid_for_port(port: u16) -> Option<u32> {
    let output = std::process::Command::new("lsof")
        .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-t"])
        .output()
        .ok()?;
    String::from_utf8(output.stdout)
        .ok()?
        .lines()
        .find_map(|line| line.trim().parse::<u32>().ok())
}

/// Find an available port in the selected four-port range.
pub fn find_available_port() -> Option<u16> {
    let range_start = port_range_start();
    (range_start..=range_start + (PORT_RANGE_LEN - 1))
        .find(|&port| TcpListener::bind(("127.0.0.1", port)).is_ok())
}

/// Parse the tunnel URL from cloudflared's stderr output.
pub fn parse_tunnel_url(stderr: &str) -> Option<String> {
    TUNNEL_URL_RE.find(stderr).map(|m| m.as_str().to_string())
}

fn create_private_dir(path: &Path, file_name: &str) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|e| {
        format!(
            "Failed to create current {} directory at {}: {}",
            file_name,
            path.display(),
            e
        )
    })?;
    restrict_private_dir(path, file_name)
}

fn restrict_private_dir(path: &Path, file_name: &str) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).map_err(|e| {
            format!(
                "Failed to restrict current {} directory at {}: {}",
                file_name,
                path.display(),
                e
            )
        })?;
    }
    #[cfg(not(unix))]
    {
        let _ = (path, file_name);
    }
    Ok(())
}

fn restrict_private_file(path: &Path, file_name: &str) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    if !path.is_file() {
        return Err(format!(
            "Current {} path is not a file: {}",
            file_name,
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(|e| {
            format!(
                "Failed to restrict current {} file at {}: {}",
                file_name,
                path.display(),
                e
            )
        })?;
    }
    #[cfg(not(unix))]
    {
        let _ = file_name;
    }
    Ok(())
}

fn prepare_private_parent(path: &Path, file_name: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        create_private_dir(parent, file_name)?;
    }
    Ok(())
}

fn write_private_file(path: &Path, contents: &[u8], file_name: &str) -> Result<(), String> {
    prepare_private_parent(path, file_name)?;
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| {
                format!(
                    "Failed to open current {} file at {}: {}",
                    file_name,
                    path.display(),
                    e
                )
            })?;
        file.write_all(contents).map_err(|e| {
            format!(
                "Failed to write current {} file at {}: {}",
                file_name,
                path.display(),
                e
            )
        })?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, contents).map_err(|e| {
            format!(
                "Failed to write current {} file at {}: {}",
                file_name,
                path.display(),
                e
            )
        })?;
    }
    restrict_private_file(path, file_name)
}

fn import_nonempty_legacy_file(
    current_dir: &Path,
    legacy_dir: &Path,
    file_name: &str,
) -> Result<PathBuf, String> {
    let current_path = current_dir.join(file_name);
    if current_path.exists() {
        restrict_private_dir(current_dir, file_name)?;
        restrict_private_file(&current_path, file_name)?;
        return Ok(current_path);
    }

    let legacy_path = legacy_dir.join(file_name);
    let contents = match std::fs::read_to_string(&legacy_path) {
        Ok(contents) => contents,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(current_path),
        Err(e) => {
            log::warn!(
                "[remote-access] Skipping legacy {} import from {}: {}",
                file_name,
                legacy_path.display(),
                e
            );
            return Ok(current_path);
        }
    };

    if contents.trim().is_empty() {
        return Ok(current_path);
    }

    write_private_file(&current_path, contents.as_bytes(), file_name)?;

    Ok(current_path)
}

fn token_file_path_for_dirs(current_dir: &Path, legacy_dir: &Path) -> Result<PathBuf, String> {
    import_nonempty_legacy_file(current_dir, legacy_dir, "token")
}

fn relay_id_path_for_dirs(current_dir: &Path) -> PathBuf {
    current_dir.join("relay_id")
}

/// Token file path for wenlan-mcp authentication.
/// wenlan-mcp stores tokens at ~/.config/wenlan-mcp/token (XDG convention),
/// NOT ~/Library/Application Support/ (macOS convention from dirs::config_dir).
fn token_file_path() -> Result<PathBuf, String> {
    let current = crate::identity_paths::mcp_config_dir();
    if crate::identity_paths::isolated_dev_state_dir().is_some() {
        return token_file_path_for_dirs(&current, &current);
    }
    token_file_path_for_dirs(&current, &crate::identity_paths::legacy_mcp_config_dir())
}

fn relay_id_path() -> PathBuf {
    relay_id_path_for_dirs(&crate::identity_paths::mcp_config_dir())
}

fn token_generate_args(path: &std::path::Path) -> Vec<String> {
    vec![
        "token".to_string(),
        "generate".to_string(),
        "--output".to_string(),
        path.to_string_lossy().into_owned(),
    ]
}

/// Read the bearer token from disk.
pub fn read_token() -> Result<String, String> {
    let path = token_file_path()?;
    std::fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Failed to read token at {}: {}", path.display(), e))
}

/// Start the remote access tunnel (wenlan-mcp serve + cloudflared).
/// Async — emits `remote-access-status` events as state changes.
/// Called from Tauri command handler — the command returns `Starting`
/// immediately and this runs in a background task.
pub fn toggle_on(
    app_handle: tauri::AppHandle,
    is_retry: bool,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(toggle_on_inner(
        app_handle,
        if is_retry { 1 } else { 0 },
        None,
    ))
}

/// Start with explicit retry count (used by monitor auto-restart).
fn toggle_on_with_retries(
    app_handle: tauri::AppHandle,
    retry_count: u32,
    expected_generation: u64,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(toggle_on_inner(
        app_handle,
        retry_count,
        Some(expected_generation),
    ))
}

async fn toggle_on_inner(
    app_handle: tauri::AppHandle,
    retry_count: u32,
    expected_generation: Option<u64>,
) {
    use tauri::Emitter;

    // Guard: don't start if already starting or connected
    let operation_generation = {
        let state =
            app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
        let app_state = state.read().await;
        let mut ra = app_state.remote_access.lock().await;
        let RemoteAccessState {
            status, generation, ..
        } = &mut *ra;
        let Some(operation_generation) = try_begin_start(status, generation, expected_generation)
        else {
            log::warn!("[remote-access] Already active, skipping duplicate toggle_on");
            return;
        };
        let _ = app_handle.emit("remote-access-status", &RemoteAccessStatus::Starting);
        operation_generation
    };

    let result = start_tunnel(&app_handle, operation_generation).await;

    match result {
        Ok((tunnel_url, token, mcp_child, tunnel_child, tunnel_owner, port, mcp_rx, tunnel_rx)) => {
            {
                let state = app_handle
                    .state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
                let app_state = state.read().await;
                let ra = app_state.remote_access.lock().await;
                if !generation_is_current(ra.generation, operation_generation)
                    || !matches!(ra.status, RemoteAccessStatus::Starting)
                {
                    let _ = mcp_child.kill();
                    terminate_owned_cloudflared(tunnel_child, &tunnel_owner);
                    return;
                }
            }

            // Register with relay for a stable URL
            let relay_url = tokio::select! {
                relay_url = register_with_relay(&tunnel_url) => relay_url,
                _ = wait_for_generation_change(&app_handle, operation_generation) => {
                    let _ = mcp_child.kill();
                    terminate_owned_cloudflared(tunnel_child, &tunnel_owner);
                    return;
                }
            };

            let status = RemoteAccessStatus::Connected {
                tunnel_url: tunnel_url.clone(),
                token: token.clone(),
                relay_url: relay_url.clone(),
            };
            // Store process handles in state
            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let mut ra = app_state.remote_access.lock().await;
            if !generation_is_current(ra.generation, operation_generation)
                || !matches!(ra.status, RemoteAccessStatus::Starting)
            {
                drop(ra);
                drop(app_state);
                let _ = mcp_child.kill();
                terminate_owned_cloudflared(tunnel_child, &tunnel_owner);
                return;
            }
            ra.status = status.clone();
            ra.mcp_child = Some(mcp_child);
            ra.tunnel_child = Some(tunnel_child);
            ra.port = Some(port);
            // We just successfully created a tunnel → we're not currently
            // rate-limited, so clear any stale 429 stamp from an earlier attempt.
            ra.last_rate_limit_at = None;
            let _ = app_handle.emit("remote-access-status", &status);
            drop(ra);
            drop(app_state);

            // Spawn background monitor for crash recovery
            let handle_for_monitor = app_handle.clone();
            tauri::async_runtime::spawn(async move {
                monitor_processes(
                    handle_for_monitor,
                    mcp_rx,
                    tunnel_rx,
                    retry_count,
                    operation_generation,
                )
                .await;
            });

            // Spawn periodic tunnel health check (detects broken tunnels after sleep)
            let handle_for_health = app_handle.clone();
            let health_tunnel_url = tunnel_url.clone();
            let health_retry_count = retry_count;
            tauri::async_runtime::spawn(async move {
                tunnel_health_loop(
                    handle_for_health,
                    health_tunnel_url,
                    health_retry_count,
                    operation_generation,
                )
                .await;
            });
        }
        Err(e) => {
            log::error!("[remote-access] toggle_on failed: {}", e);
            let status = RemoteAccessStatus::Error { error: e.clone() };

            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let mut ra = app_state.remote_access.lock().await;
            if !generation_is_current(ra.generation, operation_generation)
                || !matches!(ra.status, RemoteAccessStatus::Starting)
            {
                return;
            }
            ra.status = status.clone();
            // Stamp rate-limit time so `tunnel_health_loop` enforces cooldown before
            // burning another quick tunnel. Only when the error is an actual 429.
            if e.contains("429") || e.contains("rate limit") {
                ra.last_rate_limit_at = Some(std::time::Instant::now());
                log::warn!(
                    "[remote-access] Cloudflare 429 recorded — reconnect cooldown active for {} min",
                    RATE_LIMIT_COOLDOWN.as_secs() / 60
                );
            }
            let _ = app_handle.emit("remote-access-status", &status);
            drop(ra);
            drop(app_state);
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

/// Spawn wenlan-mcp serve on a given port (without cloudflared).
/// Reusable for both initial start and MCP-only restarts.
async fn spawn_mcp(
    app_handle: &tauri::AppHandle,
    port: u16,
    generation: u64,
) -> Result<
    (
        tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>,
        tauri_plugin_shell::process::CommandChild,
    ),
    String,
> {
    log::warn!(
        "[remote-access] spawning {} serve on port {}",
        MCP_SIDECAR_NAME,
        port
    );
    let origin_url = crate::api::WenlanClient::new().base_url().to_string();
    let port_string = port.to_string();
    let (mcp_rx, mcp_child) = app_handle
        .shell()
        .sidecar(MCP_SIDECAR_NAME)
        .map_err(|e| format!("{} sidecar not found: {}", MCP_SIDECAR_NAME, e))?
        .args([
            "--origin-url".to_string(),
            origin_url,
            "serve".to_string(),
            "--port".to_string(),
            port_string,
            "--no-auth".to_string(),
            "--agent-name".to_string(),
            "remote-mcp".to_string(),
            "--allowed-origins".to_string(),
            "https://claude.ai,https://chatgpt.com".to_string(),
        ])
        .spawn()
        .map_err(|e| format!("Failed to spawn {} serve: {}", MCP_SIDECAR_NAME, e))?;

    let child_pid = mcp_child.pid();
    let health_url = format!("http://127.0.0.1:{port}/health");
    let readiness = timeout(Duration::from_secs(5), async {
        loop {
            if !remote_generation_is_current(app_handle, generation).await {
                return Err("Remote access start cancelled.".to_string());
            }
            if reqwest::get(&health_url)
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                match listener_pid_for_port(port) {
                    Some(listener_pid) if listener_pid == child_pid => return Ok(()),
                    Some(listener_pid) => {
                        return Err(format!(
                            "remote access port {port} is owned by PID {listener_pid}, not spawned {} PID {child_pid}",
                            MCP_SIDECAR_NAME
                        ));
                    }
                    None => {}
                }
            }
            sleep(Duration::from_millis(200)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        Err(format!(
            "{} serve failed to start (health check timeout).",
            MCP_SIDECAR_NAME
        ))
    });

    if let Err(error) = readiness {
        let _ = mcp_child.kill();
        return Err(error);
    }

    Ok((mcp_rx, mcp_child))
}

/// Which sidecar process exited.
enum ExitedProcess {
    Mcp(String),
    Tunnel(String),
}

/// Monitor sidecar processes for unexpected exits.
/// - If wenlan-mcp exits: respawn only wenlan-mcp (tunnel stays alive, no Cloudflare cost).
/// - If cloudflared exits: full restart needed (new tunnel URL required).
pub async fn monitor_processes(
    app_handle: tauri::AppHandle,
    mut mcp_rx: tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>,
    mut tunnel_rx: tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>,
    tunnel_retry_count: u32,
    generation: u64,
) {
    use tauri::Emitter;

    let mut mcp_retries = 0u32;

    loop {
        let exited = tokio::select! {
            event = wait_for_exit(&mut mcp_rx) => ExitedProcess::Mcp(event),
            event = wait_for_exit(&mut tunnel_rx) => ExitedProcess::Tunnel(event),
        };

        {
            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let ra = app_state.remote_access.lock().await;
            if !generation_is_current(ra.generation, generation)
                || !matches!(ra.status, RemoteAccessStatus::Connected { .. })
            {
                return;
            }
        }

        match exited {
            ExitedProcess::Mcp(reason) => {
                log::warn!(
                    "[remote-access] {} exited: {} — attempting MCP-only restart",
                    MCP_SIDECAR_NAME,
                    reason
                );

                // Get port from state, kill old mcp handle
                let port = {
                    let state = app_handle
                        .state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
                    let app_state = state.read().await;
                    let mut ra = app_state.remote_access.lock().await;
                    if !generation_is_current(ra.generation, generation)
                        || !matches!(ra.status, RemoteAccessStatus::Connected { .. })
                    {
                        return;
                    }
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

                {
                    let state = app_handle
                        .state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
                    let app_state = state.read().await;
                    let ra = app_state.remote_access.lock().await;
                    if !generation_is_current(ra.generation, generation)
                        || !matches!(ra.status, RemoteAccessStatus::Connected { .. })
                    {
                        return;
                    }
                }

                match spawn_mcp(&app_handle, port, generation).await {
                    Ok((new_rx, new_child)) => {
                        // Store new child in state
                        let state = app_handle
                            .state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
                        let app_state = state.read().await;
                        let mut ra = app_state.remote_access.lock().await;
                        if !generation_is_current(ra.generation, generation)
                            || !matches!(ra.status, RemoteAccessStatus::Connected { .. })
                        {
                            drop(ra);
                            drop(app_state);
                            let _ = new_child.kill();
                            return;
                        }
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
    let Some(restart_generation) = transition_off(&app_handle, Some(generation)).await else {
        return;
    };

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
        let state =
            app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
        let app_state = state.read().await;
        let mut ra = app_state.remote_access.lock().await;
        if !generation_is_current(ra.generation, restart_generation) {
            return;
        }
        ra.status = status.clone();
        let _ = app_handle.emit("remote-access-status", &status);
        drop(ra);
        drop(app_state);
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
    {
        let state =
            app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
        let app_state = state.read().await;
        let mut ra = app_state.remote_access.lock().await;
        if !generation_is_current(ra.generation, restart_generation) {
            return;
        }
        ra.status = status.clone();
        let _ = app_handle.emit("remote-access-status", &status);
    }

    sleep(Duration::from_secs(delay_secs)).await;

    log::warn!(
        "[remote-access] Full restart attempt {}/{}",
        attempt,
        MAX_TUNNEL_RETRIES
    );
    toggle_on_with_retries(app_handle, attempt, restart_generation).await;
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
    generation: u64,
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
            if !generation_is_current(ra.generation, generation)
                || !matches!(ra.status, RemoteAccessStatus::Connected { .. })
            {
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
            let Some(off_generation) = transition_off(&app_handle, Some(generation)).await else {
                return;
            };
            let error = format!(
                "Tunnel kept dropping after {} reconnect attempts. Re-enable to try again.",
                MAX_TUNNEL_RETRIES
            );
            let status = RemoteAccessStatus::Error { error };
            let state =
                app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
            let app_state = state.read().await;
            let mut ra = app_state.remote_access.lock().await;
            if !generation_is_current(ra.generation, off_generation) {
                return;
            }
            ra.status = status.clone();
            let _ = app_handle.emit("remote-access-status", &status);
            drop(ra);
            drop(app_state);
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
        let Some(restart_generation) = transition_off(&app_handle, Some(generation)).await else {
            return;
        };
        sleep(Duration::from_secs(delay_secs)).await;
        toggle_on_with_retries(app_handle, next_count, restart_generation).await;
        return; // New toggle_on spawns its own health loop with next_count.
    }
}

async fn start_tunnel(
    app_handle: &tauri::AppHandle,
    generation: u64,
) -> Result<
    (
        String,                                                                 // tunnel_url
        String,                                                                 // token
        tauri_plugin_shell::process::CommandChild,                              // mcp_child
        tauri_plugin_shell::process::CommandChild,                              // tunnel_child
        CloudflaredOwner, // tunnel owner receipt
        u16,              // port
        tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>, // mcp_rx
        tokio::sync::mpsc::Receiver<tauri_plugin_shell::process::CommandEvent>, // tunnel_rx
    ),
    String,
> {
    // 0. Clean up any orphaned MCP processes from previous app sessions
    cleanup_orphaned_mcp();
    cleanup_owned_cloudflared(None);
    if !remote_generation_is_current(app_handle, generation).await {
        return Err("Remote access start cancelled.".to_string());
    }

    // 1. Find available port
    let range_start = port_range_start();
    let port = find_available_port().ok_or_else(|| {
        format!(
            "All remote access ports ({}-{}) are in use.",
            range_start,
            range_start + (PORT_RANGE_LEN - 1)
        )
    })?;

    // 2. Ensure token exists
    let token_path = token_file_path()?;
    log::warn!(
        "[remote-access] token_path={}, exists={}",
        token_path.display(),
        token_path.exists()
    );
    if !token_path.exists() {
        prepare_private_parent(&token_path, "token")?;
        let shell = app_handle.shell();
        log::warn!("[remote-access] generating token via sidecar...");
        let cmd = shell
            .sidecar(MCP_SIDECAR_NAME)
            .map_err(|e| format!("{} sidecar not found: {}", MCP_SIDECAR_NAME, e))?;
        log::warn!("[remote-access] sidecar command created, executing token generate...");
        let output = cmd
            .args(token_generate_args(&token_path))
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
        restrict_private_file(&token_path, "token")?;
    }
    let token = read_token()?;
    if !remote_generation_is_current(app_handle, generation).await {
        return Err("Remote access start cancelled.".to_string());
    }

    // 3. Spawn wenlan-mcp serve (with health check)
    let (mcp_rx, mcp_child) = spawn_mcp(app_handle, port, generation).await?;

    // 4. Spawn cloudflared tunnel
    let tunnel_command = match app_handle.shell().sidecar("cloudflared") {
        Ok(command) => command,
        Err(error) => {
            let _ = mcp_child.kill();
            return Err(format!("cloudflared sidecar not found: {error}"));
        }
    };
    let (mut tunnel_rx, tunnel_child) = match tunnel_command
        .args(["tunnel", "--url", &format!("http://localhost:{}", port)])
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            let _ = mcp_child.kill();
            return Err(format!("Failed to spawn cloudflared: {error}"));
        }
    };
    let tunnel_owner = match record_cloudflared_owner(tunnel_child.pid(), port) {
        Ok(owner) => owner,
        Err(error) => {
            let _ = mcp_child.kill();
            let _ = tunnel_child.kill();
            return Err(error);
        }
    };
    if !remote_generation_is_current(app_handle, generation).await {
        let _ = mcp_child.kill();
        terminate_owned_cloudflared(tunnel_child, &tunnel_owner);
        return Err("Remote access start cancelled.".to_string());
    }

    // 5. Parse tunnel URL from cloudflared output (it logs to stderr)
    let tunnel_result = tokio::select! {
        result = timeout(
            Duration::from_secs(20),
            parse_tunnel_url_from_events(&mut tunnel_rx),
        ) => result,
        _ = wait_for_generation_change(app_handle, generation) => {
            let _ = mcp_child.kill();
            terminate_owned_cloudflared(tunnel_child, &tunnel_owner);
            return Err("Remote access start cancelled.".to_string());
        }
    };
    let tunnel_url = match tunnel_result {
        Ok(Ok(url)) => url,
        Ok(Err(Some(msg))) => {
            // Known error (rate limit, etc.) — kill mcp and propagate
            let _ = mcp_child.kill();
            terminate_owned_cloudflared(tunnel_child, &tunnel_owner);
            return Err(msg);
        }
        Ok(Err(None)) => {
            let _ = mcp_child.kill();
            terminate_owned_cloudflared(tunnel_child, &tunnel_owner);
            return Err("cloudflared exited without producing a tunnel URL.".to_string());
        }
        Err(_) => {
            let _ = mcp_child.kill();
            terminate_owned_cloudflared(tunnel_child, &tunnel_owner);
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
        tunnel_owner,
        port,
        mcp_rx,
        tunnel_rx,
    ))
}

async fn transition_off(
    app_handle: &tauri::AppHandle,
    expected_generation: Option<u64>,
) -> Option<u64> {
    use tauri::Emitter;

    let state = app_handle.state::<std::sync::Arc<tokio::sync::RwLock<crate::state::AppState>>>();
    let app_state = state.read().await;
    let mut ra = app_state.remote_access.lock().await;
    if expected_generation.is_some_and(|expected| !generation_is_current(ra.generation, expected)) {
        return None;
    }

    let tunnel_child = ra.tunnel_child.take();
    let mcp_child = ra.mcp_child.take();
    let generation = {
        let RemoteAccessState {
            status, generation, ..
        } = &mut *ra;
        mark_off(status, generation)
    };
    ra.port = None;

    if let Some(child) = tunnel_child {
        let _ = child.kill();
    }
    if let Some(child) = mcp_child {
        let _ = child.kill();
    }

    // Sweep for any orphaned processes the handles didn't cover
    cleanup_orphaned_mcp();
    cleanup_owned_cloudflared(None);

    let _ = app_handle.emit("remote-access-status", &RemoteAccessStatus::Off);
    drop(ra);
    drop(app_state);
    Some(generation)
}

/// Stop the remote access tunnel — kill both processes and invalidate stale tasks.
pub async fn toggle_off(app_handle: &tauri::AppHandle) {
    let _ = transition_off(app_handle, None).await;
}

/// Rotate the bearer token — generate a new token then kill the old wenlan-mcp.
/// The crash recovery monitor detects the MCP exit and respawns only wenlan-mcp
/// (tunnel stays alive). The new instance reads the updated token from disk.
pub async fn rotate_token(app_handle: &tauri::AppHandle) -> Result<String, String> {
    // 1. Generate new token first (before killing anything)
    let token_path = token_file_path()?;
    prepare_private_parent(&token_path, "token")?;
    let shell = app_handle.shell();
    let output = shell
        .sidecar(MCP_SIDECAR_NAME)
        .map_err(|e| format!("{} sidecar not found: {}", MCP_SIDECAR_NAME, e))?
        .args(token_generate_args(&token_path))
        .output()
        .await
        .map_err(|e| format!("Failed to generate token: {}", e))?;
    if !output.status.success() {
        return Err(format!(
            "Token generation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    restrict_private_file(&token_path, "token")?;

    let token = read_token()?;

    // 2. Kill old wenlan-mcp — crash recovery monitor will detect this,
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
    use std::ffi::OsString;

    #[cfg(unix)]
    fn file_mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    struct HomeGuard {
        home: Option<OsString>,
        dev_state: Option<OsString>,
        dev_remote_port_start: Option<OsString>,
    }

    impl HomeGuard {
        fn set(path: &Path) -> Self {
            let home = std::env::var_os("HOME");
            let dev_state = std::env::var_os("WENLAN_DEV_STATE_DIR");
            let dev_remote_port_start = std::env::var_os("WENLAN_DEV_REMOTE_PORT_START");
            std::env::set_var("HOME", path);
            std::env::remove_var("WENLAN_DEV_STATE_DIR");
            std::env::remove_var("WENLAN_DEV_REMOTE_PORT_START");
            Self {
                home,
                dev_state,
                dev_remote_port_start,
            }
        }
    }

    impl Drop for HomeGuard {
        fn drop(&mut self) {
            match &self.home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match &self.dev_state {
                Some(value) => std::env::set_var("WENLAN_DEV_STATE_DIR", value),
                None => std::env::remove_var("WENLAN_DEV_STATE_DIR"),
            }
            match &self.dev_remote_port_start {
                Some(value) => std::env::set_var("WENLAN_DEV_REMOTE_PORT_START", value),
                None => std::env::remove_var("WENLAN_DEV_REMOTE_PORT_START"),
            }
        }
    }

    #[test]
    fn test_status_default_is_off() {
        let state = RemoteAccessState::default();
        assert!(matches!(state.status, RemoteAccessStatus::Off));
        assert!(state.port.is_none());
    }

    #[test]
    fn starting_transition_is_atomic_against_duplicate_attempts() {
        let mut status = RemoteAccessStatus::Off;
        let mut generation = 0;

        let start_generation = try_begin_start(&mut status, &mut generation, None).unwrap();
        assert!(matches!(status, RemoteAccessStatus::Starting));
        assert!(try_begin_start(&mut status, &mut generation, Some(start_generation)).is_none());
    }

    #[test]
    fn explicit_off_invalidates_in_flight_start_and_delayed_retry() {
        let mut status = RemoteAccessStatus::Off;
        let mut generation = 0;

        let in_flight = try_begin_start(&mut status, &mut generation, None).unwrap();
        let recovery = mark_off(&mut status, &mut generation);
        assert_ne!(in_flight, recovery);

        let explicit_off = mark_off(&mut status, &mut generation);
        assert_ne!(recovery, explicit_off);
        assert!(try_begin_start(&mut status, &mut generation, Some(recovery)).is_none());
        assert!(!generation_is_current(generation, in_flight));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn listener_identity_resolves_the_process_that_owns_the_port() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();

        assert_eq!(listener_pid_for_port(port), Some(std::process::id()));
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
    fn orphan_cleanup_requires_the_exact_remote_mcp_process_identity() {
        assert!(is_expected_remote_mcp_command(
            "/tmp/wenlan-mcp-aarch64-apple-darwin --origin-url http://127.0.0.1:17777 serve --port 22000 --no-auth --agent-name remote-mcp",
            22000,
        ));
        assert!(!is_expected_remote_mcp_command(
            "/tmp/wenlan-mcp-malicious serve --port 22000 --agent-name remote-mcp",
            22000,
        ));
        assert!(!is_expected_remote_mcp_command(
            "/tmp/wenlan-mcp-aarch64-apple-darwin serve --port 22001 --agent-name remote-mcp",
            22000,
        ));
        assert!(!is_expected_remote_mcp_command(
            "/tmp/wenlan-mcp-aarch64-apple-darwin serve --port 22000 --agent-name another-agent",
            22000,
        ));
    }

    #[test]
    fn orphan_cleanup_requires_the_exact_remote_tunnel_process_identity() {
        assert!(is_expected_remote_tunnel_command(
            "/tmp/cloudflared-aarch64-apple-darwin tunnel --url http://localhost:22000",
            22000,
        ));
        assert!(is_expected_remote_tunnel_command(
            "/tmp/cloudflared tunnel --url http://localhost:22000",
            22000,
        ));
        assert!(!is_expected_remote_tunnel_command(
            "/tmp/cloudflared tunnel --url http://localhost:22001",
            22000,
        ));
        assert!(!is_expected_remote_tunnel_command(
            "/tmp/not-cloudflared tunnel --url http://localhost:22000",
            22000,
        ));
        assert!(!is_expected_remote_tunnel_command(
            "/tmp/cloudflared access tcp --url http://localhost:22000",
            22000,
        ));
    }

    #[test]
    fn cloudflared_signal_requires_the_persisted_owner_and_live_identity() {
        let owner = CloudflaredOwner {
            pid: 42,
            port: 22000,
            identity: "started\ncloudflared tunnel".to_string(),
        };
        let other = CloudflaredOwner {
            pid: 43,
            ..owner.clone()
        };

        assert!(cloudflared_owner_authorizes_signal(
            &owner,
            Some(&owner),
            22000,
            Some(owner.identity.as_str()),
        ));
        assert!(cloudflared_owner_authorizes_signal(
            &owner,
            None,
            22000,
            Some(owner.identity.as_str()),
        ));
        assert!(!cloudflared_owner_authorizes_signal(
            &owner,
            Some(&other),
            22000,
            Some(owner.identity.as_str()),
        ));
        assert!(!cloudflared_owner_authorizes_signal(
            &owner,
            Some(&owner),
            22000,
            Some("different process start"),
        ));
        assert!(!cloudflared_owner_authorizes_signal(
            &owner,
            Some(&owner),
            23000,
            Some(owner.identity.as_str()),
        ));
    }

    #[test]
    fn owner_lock_serializes_receipt_replacement_after_compare_remove() {
        use std::sync::{mpsc, Arc};

        let dir = tempfile::tempdir().unwrap();
        let path = Arc::new(dir.path().join("cloudflared-owner.json"));
        let old_owner = CloudflaredOwner {
            pid: 42,
            port: 22000,
            identity: "old".to_string(),
        };
        let new_owner = CloudflaredOwner {
            pid: 43,
            port: 22001,
            identity: "new".to_string(),
        };
        write_cloudflared_owner_at(&path, &old_owner).unwrap();

        let (validated_tx, validated_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        let cleanup_path = Arc::clone(&path);
        let cleanup_owner = old_owner.clone();
        let cleanup = std::thread::spawn(move || {
            with_cloudflared_owner_lock(&cleanup_path, |receipt| {
                let current = receipt.read()?.unwrap();
                assert_eq!(current, cleanup_owner);
                validated_tx.send(()).unwrap();
                continue_rx.recv().unwrap();
                receipt.remove_if_matches(&cleanup_owner)
            })
            .unwrap();
        });

        validated_rx.recv().unwrap();
        let (replacement_done_tx, replacement_done_rx) = mpsc::channel();
        let replacement_path = Arc::clone(&path);
        let replacement_owner = new_owner.clone();
        let replacement = std::thread::spawn(move || {
            write_cloudflared_owner_at(&replacement_path, &replacement_owner).unwrap();
            replacement_done_tx.send(()).unwrap();
        });

        assert!(replacement_done_rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
        continue_tx.send(()).unwrap();
        cleanup.join().unwrap();
        replacement.join().unwrap();

        assert_eq!(read_cloudflared_owner_at(&path).unwrap(), Some(new_owner));
    }

    #[test]
    #[cfg(unix)]
    fn owner_replacement_swaps_the_receipt_inode_instead_of_truncating_in_place() {
        use std::os::unix::fs::MetadataExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cloudflared-owner.json");
        let old_owner = CloudflaredOwner {
            pid: 42,
            port: 22000,
            identity: "old".to_string(),
        };
        let new_owner = CloudflaredOwner {
            pid: 43,
            port: 22001,
            identity: "new".to_string(),
        };

        write_cloudflared_owner_at(&path, &old_owner).unwrap();
        let old_inode = std::fs::metadata(&path).unwrap().ino();
        write_cloudflared_owner_at(&path, &new_owner).unwrap();
        let new_inode = std::fs::metadata(&path).unwrap().ino();

        assert_ne!(old_inode, new_inode);
        assert_eq!(read_cloudflared_owner_at(&path).unwrap(), Some(new_owner));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[serial_test::serial]
    fn dev_remote_access_uses_its_worktree_port_range() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = HomeGuard::set(tmp.path());
        std::env::set_var("WENLAN_DEV_REMOTE_PORT_START", "23000");

        let port = find_available_port().expect("dev remote access port");

        assert!((23000..=23003).contains(&port));
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

    #[test]
    fn token_path_imports_nonempty_legacy_token_when_current_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let current = tmp.path().join("wenlan-mcp");
        let legacy = tmp.path().join("origin-mcp");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("token"), "legacy-token\n").unwrap();

        let path = token_file_path_for_dirs(&current, &legacy).unwrap();

        assert_eq!(path, current.join("token"));
        assert_eq!(
            std::fs::read_to_string(current.join("token")).unwrap(),
            "legacy-token\n"
        );
    }

    #[test]
    #[cfg(unix)]
    fn token_path_imports_legacy_token_with_private_permissions() {
        let tmp = tempfile::tempdir().unwrap();
        let current = tmp.path().join("wenlan-mcp");
        let legacy = tmp.path().join("origin-mcp");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("token"), "legacy-token\n").unwrap();

        let path = token_file_path_for_dirs(&current, &legacy).unwrap();

        assert_eq!(path, current.join("token"));
        assert_eq!(file_mode(&path), 0o600);
        assert_eq!(file_mode(&current), 0o700);
    }

    #[test]
    fn token_path_keeps_current_token_when_present() {
        let tmp = tempfile::tempdir().unwrap();
        let current = tmp.path().join("wenlan-mcp");
        let legacy = tmp.path().join("origin-mcp");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(current.join("token"), "current-token\n").unwrap();
        std::fs::write(legacy.join("token"), "legacy-token\n").unwrap();

        let path = token_file_path_for_dirs(&current, &legacy).unwrap();

        assert_eq!(path, current.join("token"));
        assert_eq!(
            std::fs::read_to_string(current.join("token")).unwrap(),
            "current-token\n"
        );
    }

    #[test]
    fn token_path_does_not_import_empty_legacy_token() {
        let tmp = tempfile::tempdir().unwrap();
        let current = tmp.path().join("wenlan-mcp");
        let legacy = tmp.path().join("origin-mcp");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("token"), " \n").unwrap();

        let path = token_file_path_for_dirs(&current, &legacy).unwrap();

        assert_eq!(path, current.join("token"));
        assert!(!current.join("token").exists());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[serial_test::serial]
    fn dev_token_path_does_not_import_production_credentials() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = HomeGuard::set(tmp.path());
        let production_legacy = tmp.path().join(".config/origin-mcp");
        std::fs::create_dir_all(&production_legacy).unwrap();
        std::fs::write(production_legacy.join("token"), "production-token\n").unwrap();
        let dev_state = tmp.path().join("dev-state");
        std::env::set_var("WENLAN_DEV_STATE_DIR", &dev_state);

        let path = token_file_path().unwrap();

        assert_eq!(path, dev_state.join("mcp-config/token"));
        assert!(
            !path.exists(),
            "production token must not be copied into dev"
        );
    }

    #[test]
    fn token_path_skips_invalid_legacy_token() {
        let tmp = tempfile::tempdir().unwrap();
        let current = tmp.path().join("wenlan-mcp");
        let legacy = tmp.path().join("origin-mcp");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("token"), [0xff, 0xfe]).unwrap();

        let path = token_file_path_for_dirs(&current, &legacy).unwrap();

        assert_eq!(path, current.join("token"));
        assert!(!current.join("token").exists());
    }

    #[test]
    fn relay_id_path_does_not_import_legacy_relay_id() {
        let tmp = tempfile::tempdir().unwrap();
        let current = tmp.path().join("wenlan-mcp");
        let legacy = tmp.path().join("origin-mcp");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("relay_id"), "stale-relay-id").unwrap();

        let path = relay_id_path_for_dirs(&current);

        assert_eq!(path, current.join("relay_id"));
        assert!(!current.join("relay_id").exists());
    }

    #[test]
    #[cfg(unix)]
    #[serial_test::serial]
    fn relay_id_generation_writes_private_secret_file() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = HomeGuard::set(tmp.path());

        let id = get_or_create_relay_id().unwrap();
        let path = relay_id_path();

        assert!(!id.is_empty());
        assert_eq!(file_mode(&path), 0o600);
        assert_eq!(file_mode(path.parent().unwrap()), 0o700);
    }

    #[test]
    #[serial_test::serial]
    fn relay_id_generation_errors_when_relay_id_path_is_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = HomeGuard::set(tmp.path());
        let path = relay_id_path();
        std::fs::create_dir_all(&path).unwrap();

        assert!(get_or_create_relay_id().is_err());
    }

    #[test]
    #[serial_test::serial]
    fn test_token_generate_args_include_wenlan_output_path() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = HomeGuard::set(tmp.path());
        let path = token_file_path().unwrap();
        let expected_path = tmp.path().join(".config/wenlan-mcp/token");

        assert_eq!(
            token_generate_args(&path),
            vec![
                "token".to_string(),
                "generate".to_string(),
                "--output".to_string(),
                expected_path.to_string_lossy().into_owned()
            ]
        );
    }
}
