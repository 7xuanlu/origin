// SPDX-License-Identifier: AGPL-3.0-only
//! Tauri command surface — thin HTTP client that proxies data operations
//! to the selected daemon (`client.base_url()`; an isolated dev app selects a
//! different port than the installed production runtime).
//!
//! UI-only commands (window positioning, permissions, shortcuts) remain local.
//! Daemon-owned config commands proxy through `state.client`; app-only sensor
//! state mirrors successful daemon config writes where the running process
//! needs an immediate in-memory value.
//! All data/DB commands proxy through `state.client`.

use crate::activity;
use crate::api::percent_encode_path_segment;
use crate::config;
use crate::sources::SourceStatus;
use crate::state::{AppState, IndexStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use wenlan_types::requests;
use wenlan_types::responses;
use wenlan_types::*;

type State = Arc<RwLock<AppState>>;
type WatcherState = Arc<tokio::sync::Mutex<Option<crate::indexer::FileWatcher>>>;

/// Snapshot the daemon client and drop the `AppState` read guard before the
/// caller's HTTP round-trip.
///
/// Holding a `tokio::sync::RwLock` read guard across `.await` is forbidden by
/// AGENTS.md "Repository invariants", and here it can freeze the whole app:
/// tokio's RwLock is write-preferring, so one queued writer — a config write,
/// or `indexer::sync_source` fired by the file watcher — blocks every later
/// reader until the in-flight request finishes, which `api::REQUEST_TIMEOUT`
/// bounds at 600 s. Commands that also need other `AppState` fields use the
/// scoped-block form instead (see `suggest_tags`).
async fn daemon_client(state: &tauri::State<'_, State>) -> crate::api::WenlanClient {
    state.read().await.client.clone()
}

// ── Request types (kept for Tauri IPC deserialization) ─────────────────

#[derive(Debug, Deserialize)]
pub struct QuickCaptureRequest {
    pub title: Option<String>,
    pub content: String,
    pub tags: Option<Vec<String>>,
    pub memory_type: Option<String>,
    pub domain: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoreMemoryRequest {
    pub content: String,
    pub memory_type: Option<String>,
    pub domain: Option<String>,
    pub source_agent: Option<String>,
    pub title: Option<String>,
    pub tags: Option<Vec<String>>,
    pub confidence: Option<f32>,
    pub supersedes: Option<String>,
    pub structured_fields: Option<serde_json::Value>,
    pub retrieval_cue: Option<String>,
}

// ── Response types ────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct StoreMemoryResponse {
    pub source_id: String,
    pub warnings: Vec<String>,
    /// Background-enrichment state from the daemon — `"pending"` when
    /// classify + extract will run asynchronously, `"not_needed"` when
    /// no LLM is available. The frontend uses this to drive live-update
    /// UI (invalidate the stored-memory query once background work lands).
    /// Defaulted so older daemon responses still deserialize cleanly.
    #[serde(default)]
    pub enrichment: String,
    /// Prose nudge for callers — safe to show to the user. Empty when no
    /// enrichment will run.
    #[serde(default)]
    pub hint: String,
}

// ── Window / UI commands (kept as-is) ─────────────────────────────────

#[tauri::command]
pub fn set_traffic_lights_visible(window: tauri::Window, visible: bool) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    #[allow(deprecated)]
    {
        use cocoa::appkit::{NSWindow, NSWindowButton};
        use raw_window_handle::HasWindowHandle;

        let raw_handle = window.window_handle().map_err(|e| e.to_string())?;
        if let raw_window_handle::RawWindowHandle::AppKit(appkit) = raw_handle.as_raw() {
            let ns_view = appkit.ns_view.as_ptr() as cocoa::base::id;
            unsafe {
                let ns_win: cocoa::base::id = objc::msg_send![ns_view, window];
                for button in &[
                    NSWindowButton::NSWindowCloseButton,
                    NSWindowButton::NSWindowMiniaturizeButton,
                    NSWindowButton::NSWindowZoomButton,
                ] {
                    let btn: cocoa::base::id = ns_win.standardWindowButton_(*button);
                    if btn != cocoa::base::nil {
                        let _: () = objc::msg_send![btn, setHidden:!visible];
                    }
                }
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (&window, visible);
    Ok(())
}

/// Hide quick-capture and prevent macOS from auto-activating the main window.
/// Called from the quick-capture webview on Esc / Enter save.
#[tauri::command]
pub async fn dismiss_quick_capture(app: tauri::AppHandle) -> Result<(), String> {
    use tauri::Manager;

    if let Some(qc) = app.get_webview_window("quick-capture") {
        // Use orderOut instead of hide() to remove the window without
        // triggering macOS window promotion (which would show main).
        #[cfg(target_os = "macos")]
        #[allow(deprecated)]
        {
            let qc_for_main_thread = qc.clone();
            qc.run_on_main_thread(move || {
                use cocoa::base::id;
                use raw_window_handle::HasWindowHandle;

                if let Ok(raw_handle) = qc_for_main_thread.window_handle() {
                    if let raw_window_handle::RawWindowHandle::AppKit(appkit) = raw_handle.as_raw()
                    {
                        let ns_view = appkit.ns_view.as_ptr() as id;
                        unsafe {
                            let ns_win: id = objc::msg_send![ns_view, window];
                            let _: () = objc::msg_send![ns_win, orderOut: ns_win];
                        }
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = qc.hide();
        }
    }

    Ok(())
}

#[tauri::command]
pub async fn position_quick_capture(app: tauri::AppHandle) -> Result<(), String> {
    use tauri::Manager;

    let win = app
        .get_webview_window("quick-capture")
        .ok_or("quick-capture window not found")?;
    #[cfg(not(target_os = "macos"))]
    let _ = &win;

    #[cfg(target_os = "macos")]
    #[allow(deprecated)]
    {
        use cocoa::base::id;
        use cocoa::foundation::NSRect;
        use raw_window_handle::HasWindowHandle;
        use tauri::{LogicalPosition, LogicalSize};

        let raw_handle = win.window_handle().map_err(|e| e.to_string())?;
        if let raw_window_handle::RawWindowHandle::AppKit(appkit) = raw_handle.as_raw() {
            let ns_view = appkit.ns_view.as_ptr() as id;

            let (visible, screen_h) = unsafe {
                let ns_win: id = objc::msg_send![ns_view, window];
                if ns_win.is_null() {
                    return Err("NSWindow not attached".into());
                }
                let screen: id = objc::msg_send![ns_win, screen];
                if screen.is_null() {
                    return Err("NSScreen not available".into());
                }
                let visible: NSRect = objc::msg_send![screen, visibleFrame];
                let frame: NSRect = objc::msg_send![screen, frame];
                (visible, frame.size.height)
            };

            let width = 400.0;
            let height = 160.0;
            let padding = 16.0;

            win.set_size(LogicalSize::new(width, height))
                .map_err(|e| e.to_string())?;

            let x = visible.origin.x + visible.size.width - width - padding;
            let y = screen_h - visible.origin.y - padding - height;

            log::debug!("[qc-pos] visible=({:.0},{:.0} {:.0}x{:.0}) screen_h={:.0} → size=({:.0},{:.0}) pos=({:.0},{:.0})",
                visible.origin.x, visible.origin.y, visible.size.width, visible.size.height,
                screen_h, width, height, x, y);

            win.set_position(LogicalPosition::new(x, y))
                .map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

#[tauri::command]
pub async fn get_api_key() -> Result<Option<String>, String> {
    let config = config::load_config();
    Ok(config.anthropic_api_key.map(|key| {
        let chars: Vec<char> = key.chars().collect();
        if chars.len() > 12 {
            let prefix: String = chars[..8].iter().collect();
            let suffix: String = chars[chars.len() - 4..].iter().collect();
            format!("{}...{}", prefix, suffix)
        } else {
            "***".to_string()
        }
    }))
}

#[derive(Debug, Serialize)]
struct AnthropicKeyRequest {
    api_key: String,
}

async fn set_anthropic_key_response(
    client: &crate::api::WenlanClient,
    req: &AnthropicKeyRequest,
) -> Result<responses::SuccessResponse, String> {
    client.put_json("/api/setup/anthropic-key", req).await
}

async fn clear_anthropic_key_response(
    client: &crate::api::WenlanClient,
) -> Result<responses::SuccessResponse, String> {
    client.delete_path("/api/setup/anthropic-key").await
}

#[tauri::command]
pub async fn set_api_key(state: tauri::State<'_, State>, key: String) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    if key.trim().is_empty() {
        let _resp = clear_anthropic_key_response(&client).await?;
    } else {
        let body = AnthropicKeyRequest { api_key: key };
        let _resp = set_anthropic_key_response(&client, &body).await?;
    }
    log::info!("[settings] API key updated");
    Ok(())
}

#[cfg(test)]
mod setup_key_response_tests {
    use super::*;

    #[allow(dead_code)]
    async fn set_anthropic_key_uses_success_response(client: crate::api::WenlanClient) {
        let req = AnthropicKeyRequest {
            api_key: "sk-ant-test".to_string(),
        };
        let _: Result<responses::SuccessResponse, String> =
            set_anthropic_key_response(&client, &req).await;
    }

    #[allow(dead_code)]
    async fn clear_anthropic_key_uses_success_response(client: crate::api::WenlanClient) {
        let _: Result<responses::SuccessResponse, String> =
            clear_anthropic_key_response(&client).await;
    }

    #[allow(dead_code)]
    async fn public_command_keeps_void_surface(state: tauri::State<'_, State>) {
        let _: Result<(), String> = set_api_key(state, String::new()).await;
    }

    #[test]
    fn anthropic_key_request_serializes_daemon_payload() {
        let req = AnthropicKeyRequest {
            api_key: "sk-ant-test".to_string(),
        };
        let value = serde_json::to_value(req).unwrap();
        assert_eq!(value, serde_json::json!({ "api_key": "sk-ant-test" }));
    }
}

#[cfg(test)]
mod ingest_command_tests {
    use super::*;

    #[allow(dead_code)]
    async fn webpage_ingest_command_uses_shared_response(state: tauri::State<'_, State>) {
        let req = requests::IngestWebpageRequest {
            url: "https://example.com/post".to_string(),
            title: "Example Post".to_string(),
            content: "A durable article body.".to_string(),
            metadata: None,
        };
        let _: Result<responses::IngestResponse, String> = ingest_webpage(state, req).await;
    }

    #[test]
    fn webpage_ingest_command_response_type_is_checked() {}
}

#[tauri::command]
pub async fn get_setup_completed(state: tauri::State<'_, State>) -> Result<bool, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    Ok(client.get_setup_status().await?.setup_completed)
}

#[tauri::command]
pub async fn set_setup_completed(
    state: tauri::State<'_, State>,
    completed: bool,
) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.set_setup_completed(completed).await
}

#[tauri::command]
pub async fn should_show_wizard(state: tauri::State<'_, State>) -> Result<bool, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    Ok(!client.get_setup_status().await?.setup_completed)
}

#[tauri::command]
pub async fn detect_mcp_clients_cmd() -> Result<Vec<crate::mcp_config::McpClient>, String> {
    Ok(crate::mcp_config::detect_mcp_clients())
}

#[tauri::command]
pub async fn write_mcp_config(client_type: String) -> Result<(), String> {
    let config_path = crate::mcp_config::client_config_path(&client_type)
        .ok_or(format!("Unknown client type: {}", client_type))?;
    if client_type == "codex_cli" {
        return crate::mcp_config::write_wenlan_entry_toml(&config_path).map_err(|e| e.to_string());
    }
    let is_claude_code = client_type == "claude_code";
    crate::mcp_config::write_wenlan_entry(&config_path, is_claude_code).map_err(|e| e.to_string())
}

/// Removes the raw `wenlan`/legacy `origin` MCP entry from `client_type`'s
/// config file — the fix Diagnostics offers for a double registration (a
/// plugin *and* a raw entry for one client). Symmetric with detection: it
/// clears exactly what `has_configured_entry` recognizes, leaving every
/// sibling server and unrelated key intact. A missing file or absent entry
/// is an `Err` the UI surfaces verbatim.
#[tauri::command]
pub async fn remove_raw_mcp_entry(client_type: String) -> Result<(), String> {
    let config_path = crate::mcp_config::client_config_path(&client_type)
        .ok_or(format!("Unknown client type: {}", client_type))?;
    if client_type == "codex_cli" {
        return crate::mcp_config::remove_wenlan_entry_toml(&config_path)
            .map_err(|e| e.to_string());
    }
    crate::mcp_config::remove_wenlan_entry(&config_path).map_err(|e| e.to_string())
}

/// Removes ONLY the legacy `origin` MCP entry from `client_type`'s config,
/// keeping the live `wenlan` entry — the fix Diagnostics offers for a raw+raw
/// duplicate (both `wenlan` and `origin` in one config, on a no-plugin client
/// like Cursor). Unlike `remove_raw_mcp_entry`, which drops both keys, this
/// leaves the working connection in place. A missing file or absent `origin`
/// entry is an `Err` the UI surfaces verbatim.
#[tauri::command]
pub async fn remove_legacy_mcp_entry(client_type: String) -> Result<(), String> {
    let config_path = crate::mcp_config::client_config_path(&client_type)
        .ok_or(format!("Unknown client type: {}", client_type))?;
    if client_type == "codex_cli" {
        return crate::mcp_config::remove_legacy_origin_entry_toml(&config_path)
            .map_err(|e| e.to_string());
    }
    crate::mcp_config::remove_legacy_origin_entry(&config_path).map_err(|e| e.to_string())
}

/// Returns the current `wenlan` MCP server entry (command + args) that Wenlan
/// uses when writing client configs. Prefers a local binary in dev, falls back
/// to `npx -y wenlan-mcp` otherwise. The frontend uses this to build a
/// copy-pasteable manual-setup JSON snippet with real values instead of
/// `/path/to/wenlan-mcp` placeholder text.
#[tauri::command]
pub async fn get_wenlan_mcp_entry() -> Result<crate::mcp_config::WenlanMcpEntry, String> {
    Ok(crate::mcp_config::wenlan_mcp_entry())
}

/// Installs the Wenlan plugin for `client_type` (`"claude_code"` /
/// `"codex_cli"`) by shelling out to that client's CLI (marketplace add,
/// then plugin install/add) — see `plugin_install::install_client_plugin`.
/// Idempotent: succeeds if the marketplace or plugin is already present.
/// Runs on a blocking thread since the marketplace step can clone over the
/// network.
#[tauri::command]
pub async fn install_client_plugin(client_type: String) -> Result<(), String> {
    tokio::task::spawn_blocking(move || crate::plugin_install::install_client_plugin(&client_type))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

/// The real, resolved wiring of Wenlan on this machine — daemon
/// reachability, the `wenlan-mcp` binary that would actually be written into
/// a client config (with the full candidate trail, missing paths included),
/// and per-client MCP routing. Backs the wizard's "Setting up" step and
/// Settings → Diagnostics. Never rejects on a down daemon — see
/// `wire_state::compute`.
#[tauri::command]
pub async fn wire_state(
    state: tauri::State<'_, State>,
) -> Result<crate::wire_state::WireState, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    Ok(crate::wire_state::compute(&client).await)
}

// ── Activity commands (file-based, local) ─────────────────────────────

#[tauri::command]
pub async fn list_activities(
    state: tauri::State<'_, State>,
) -> Result<Vec<activity::ActivitySummary>, String> {
    let s = state.read().await;
    Ok(s.list_activity_summaries())
}

#[tauri::command]
pub async fn rebuild_activities(state: tauri::State<'_, State>) -> Result<usize, String> {
    // In thin-client mode, we cannot scan the DB for timestamps.
    // Keep the file-based activity rebuild from completed_activities.
    let s = state.read().await;
    Ok(s.completed_activities.len())
}

#[tauri::command]
pub async fn get_capture_stats(
    state: tauri::State<'_, State>,
) -> Result<HashMap<String, u64>, String> {
    let client = daemon_client(&state).await;
    client.get_capture_stats().await
}

#[tauri::command]
pub async fn get_pipeline_status(
    state: tauri::State<'_, State>,
) -> Result<crate::api::PipelineStatusResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.pipeline_status().await
}

#[cfg(test)]
mod pipeline_status_command_type_tests {
    use super::*;

    #[allow(dead_code)]
    async fn get_pipeline_status_uses_typed_response(state: tauri::State<'_, State>) {
        let _: Result<crate::api::PipelineStatusResponse, String> =
            get_pipeline_status(state).await;
    }

    #[test]
    fn pipeline_status_command_response_type_is_checked() {}
}

// ── Remote access commands ────────────────────────────────────────────

#[tauri::command]
pub async fn toggle_remote_access(
    state: tauri::State<'_, State>,
    app_handle: tauri::AppHandle,
    enabled: bool,
) -> Result<crate::remote_access::RemoteAccessStatus, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };

    if enabled {
        client.set_remote_access_enabled(true).await?;

        let handle = app_handle.clone();
        tauri::async_runtime::spawn(async move {
            crate::remote_access::toggle_on(handle, false).await;
        });

        Ok(crate::remote_access::RemoteAccessStatus::Starting)
    } else {
        client.set_remote_access_enabled(false).await?;

        crate::remote_access::toggle_off(&app_handle).await;
        Ok(crate::remote_access::RemoteAccessStatus::Off)
    }
}

#[tauri::command]
pub async fn get_remote_access_status(
    state: tauri::State<'_, State>,
) -> Result<crate::remote_access::RemoteAccessStatus, String> {
    let app_state = state.read().await;
    let ra = app_state.remote_access.lock().await;
    Ok(ra.status.clone())
}

#[derive(serde::Serialize)]
pub struct RemoteConnectionTest {
    pub ok: bool,
    pub latency_ms: Option<u64>,
    pub error: Option<String>,
}

#[tauri::command]
pub async fn test_remote_mcp_connection(
    state: tauri::State<'_, State>,
) -> Result<RemoteConnectionTest, String> {
    // Snapshot out of the lock, then drop the guard.
    //
    // Prefer `relay_url` over `tunnel_url`. The relay domain
    // (origin-relay.originmemory.workers.dev) always resolves via system DNS,
    // while fresh `*.trycloudflare.com` tunnel subdomains can hit ISP DNS
    // cache NXDOMAIN for several minutes — a known Cloudflare quick-tunnel
    // issue. The relay URL also reflects what the user actually hands to
    // Claude.ai / ChatGPT, so probing it is semantically correct.
    let (probe_url, is_relay): (Option<String>, bool) = {
        let app_state = state.read().await;
        let ra = app_state.remote_access.lock().await;
        match &ra.status {
            crate::remote_access::RemoteAccessStatus::Connected {
                tunnel_url,
                relay_url,
                ..
            } => match relay_url {
                Some(url) => (Some(url.clone()), true),
                None => (Some(tunnel_url.clone()), false),
            },
            _ => (None, false),
        }
    };
    let Some(url) = probe_url else {
        return Ok(RemoteConnectionTest {
            ok: false,
            latency_ms: None,
            error: Some("Remote Access not connected".to_string()),
        });
    };
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return Ok(RemoteConnectionTest {
                ok: false,
                latency_ms: None,
                error: Some(format!("http client: {}", e)),
            });
        }
    };
    let start = std::time::Instant::now();
    // Raw tunnel URL: probe `/health` (wenlan-mcp serves it; expect 2xx).
    // Relay URL: probe the URL directly — any HTTP response (even 4xx from
    // method-not-allowed on GET /mcp) proves DNS + TLS + worker reachable;
    // only 5xx / connection errors indicate a real problem.
    let probe = if is_relay {
        url.clone()
    } else {
        format!("{}/health", url.trim_end_matches('/'))
    };
    match client.get(&probe).send().await {
        Ok(resp) => {
            let status = resp.status();
            let latency = Some(start.elapsed().as_millis() as u64);
            let reachable = if is_relay {
                !status.is_server_error()
            } else {
                status.is_success()
            };
            if reachable {
                Ok(RemoteConnectionTest {
                    ok: true,
                    latency_ms: latency,
                    error: None,
                })
            } else {
                Ok(RemoteConnectionTest {
                    ok: false,
                    latency_ms: latency,
                    error: Some(format!("HTTP {}", status)),
                })
            }
        }
        Err(e) => Ok(RemoteConnectionTest {
            ok: false,
            latency_ms: None,
            error: Some(e.to_string()),
        }),
    }
}

// ── File / open commands ──────────────────────────────────────────────

/// What `open_file` and `open_search_result` will hand to the OS: a web page,
/// a local file, or a bare filesystem path. Everything else is refused.
///
/// `open_file` serves the Sources view, where the value is a path the user is
/// browsing in their own configured folder, or the configured folder itself
/// — not attacker-influenced. Search results are the attacker-influenced
/// value: `POST /api/ingest/webpage` and `/api/ingest/memory` store the
/// caller's `url` verbatim (`crates/wenlan-server/src/ingest_routes.rs`), it
/// rides out on every search result, so any agent or page that can put a
/// memory in the store chooses a string the desktop's scheme handlers will
/// act on. That is why clicking a search result calls `open_search_result`,
/// not `open_file` — see its allowlist-based check.
/// `javascript:`, `data:`, `vbscript:` and the long tail of registered
/// application schemes are all reachable via that route. The other opener in
/// the app, the Tauri shell plugin behind citation links, has been
/// scheme-restricted all along by its default scope; these two were not.
/// `file:` is deliberately absent from both. The callers that deal in `file:`
/// URLs strip the prefix before invoking (`openFile` in `src/lib/tauri.ts`;
/// `open_search_result` refuses it outright rather than stripping it), so a
/// local file arrives here as a bare path and goes through the filesystem
/// checks below. Accepting the URL form as well would be a way around them.
const OPENABLE_SCHEMES: [&str; 2] = ["http", "https"];

/// Filename endings this platform treats as something to run rather than
/// something to read. The lists are per-platform on purpose: a `.py` file on
/// macOS opens in an editor and should stay openable, while on Windows the
/// script hosts turn several of these into execution.
#[cfg(target_os = "macos")]
const LAUNCHER_SUFFIXES: [&str; 8] = [
    "app",
    "command",
    "pkg",
    "mpkg",
    "scpt",
    "applescript",
    "workflow",
    "webloc",
];
#[cfg(target_os = "windows")]
const LAUNCHER_SUFFIXES: [&str; 25] = [
    "exe",
    "com",
    "bat",
    "cmd",
    "scr",
    "msi",
    "msp",
    "cpl",
    "hta",
    "pif",
    "ps1",
    "vbs",
    "vbe",
    "js",
    "jse",
    "wsf",
    "wsh",
    "reg",
    "lnk",
    "msc",
    "jar",
    "url",
    "scf",
    "appref-ms",
    "application",
];
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
const LAUNCHER_SUFFIXES: [&str; 1] = ["desktop"];

/// The URL scheme of `target`, lowercased, or `None` when it is a filesystem
/// path.
///
/// A one-character scheme is read as a Windows drive letter (`C:\Users\...`),
/// never as a scheme: RFC 3986 allows a single letter, but nothing registers
/// one, and treating `C:` as a scheme would refuse every Windows path.
fn target_scheme(target: &str) -> Option<String> {
    let (head, _) = target.split_once(':')?;
    if head.len() < 2 || head.contains('/') || head.contains('\\') {
        return None;
    }
    let mut chars = head.chars();
    if !chars.next()?.is_ascii_alphabetic() {
        return None;
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.') {
        return None;
    }
    Some(head.to_ascii_lowercase())
}

/// Whether the last extension of `path` is one this platform launches.
///
/// Read off the string rather than the disk, so it holds for a path that does
/// not exist yet and for a macOS `.app`, which is a directory. Trailing `.`
/// and trailing space are trimmed off the extension before comparing, because
/// Win32 strips both before resolving a filename — `evil.exe   ` opens
/// `evil.exe`.
fn has_launcher_suffix(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.trim_end_matches(['.', ' ']).to_ascii_lowercase())
        .is_some_and(|ext| LAUNCHER_SUFFIXES.contains(&ext.as_str()))
}

/// Whether `path` is a regular file carrying an executable bit. Follows
/// symlinks, so the question is asked of what would actually run. Always
/// `false` off Unix, where the extension carries this instead.
#[cfg(unix)]
fn is_executable_file(path: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}
#[cfg(not(unix))]
fn is_executable_file(_path: &std::path::Path) -> bool {
    false
}

/// `Ok(())` when `open_file` may hand `target` to the OS. This is the whole of
/// the decision, kept out of `open_file` so the tests below exercise the same
/// code the command runs rather than a copy of its reasoning.
///
/// Two gates, because the finding has two shapes. A URL must carry a scheme
/// this app opens. A filesystem path must not be a thing the OS would run:
/// the audit's own example was a memory whose `url` was
/// `/Applications/Utilities/Terminal.app`, which carries no scheme at all and
/// would sail past a scheme check.
///
/// The path is canonicalized before either filesystem check runs, so a
/// symlink whose own name carries no launcher suffix but which resolves to a
/// `.app` bundle (or any other launcher) is judged on what it actually points
/// at, not on its name. Canonicalizing also has the OS resolve a Windows
/// trailing dot or space the way `ShellExecuteExW` would. Canonicalization
/// failing (a path that does not exist, or has just moved) falls back to the
/// raw path rather than refusing outright — this command also serves the
/// Sources view, which can legitimately be handed a path like that.
fn refuse_unopenable_target(target: &str) -> Result<(), String> {
    if target.contains('\0') {
        return Err("Refusing to open a path containing a NUL byte.".to_string());
    }

    if let Some(scheme) = target_scheme(target) {
        if OPENABLE_SCHEMES.contains(&scheme.as_str()) {
            return Ok(());
        }
        return Err(format!(
            "Refusing to open a \"{scheme}:\" link. Wenlan opens web pages and files on this computer."
        ));
    }

    let path = std::path::Path::new(target);
    let canonical = std::fs::canonicalize(path).ok();
    let checked = canonical.as_deref().unwrap_or(path);
    if has_launcher_suffix(checked) || is_executable_file(checked) {
        return Err(format!(
            "Refusing to run \"{target}\". Wenlan opens documents and folders, not programs."
        ));
    }
    Ok(())
}

/// Hand `path` — a filesystem path or a URL, both of which callers pass — to
/// the desktop's default handler via the `open` crate rather than a hand-built
/// per-OS argv: its Windows launcher never routes the path through `cmd.exe`
/// (so a filename containing shell metacharacters cannot inject a second
/// command), and it covers macOS/Linux/BSD the same way. Anything
/// `refuse_unopenable_target` turns down never reaches it.
#[tauri::command]
pub async fn open_file(path: String) -> Result<(), String> {
    refuse_unopenable_target(&path)?;
    open::that_detached(&path).map_err(|e| format!("Failed to open file: {}", e))
}

/// What the daemon's directory ingest actually indexes
/// (`crates/wenlan-core/src/sources/directory.rs`). A search result that
/// points at a local file points at one of these, because that is all that
/// gets indexed — so this can be an allowlist rather than a denylist of
/// everything the OS might run.
const INDEXED_DOCUMENT_SUFFIXES: [&str; 3] = ["md", "txt", "pdf"];

/// `Ok(())` when `open_search_result` may hand `target` to the OS.
///
/// `target` is a search result's `url`, stored verbatim from whatever an
/// ingest call sent (`POST /api/ingest/webpage`, `/api/ingest/memory`) — any
/// agent or page that can write a memory chooses this string. That rules out
/// the denylist `refuse_unopenable_target` uses for the Sources view: a
/// denylist only refuses what it happens to enumerate, and this input is
/// adversarial. So a filesystem path here is judged by an allowlist instead —
/// it must canonicalize to a regular file whose extension is one of the
/// document types Wenlan actually indexes.
///
/// Canonicalizing resolves symlinks and `..` before either filesystem check
/// runs, so a symlink named to look like a document but pointing at an `.app`
/// bundle is judged on what it resolves to. It also fails outright for a
/// trailing Windows dot/space or an embedded NUL, since none of those name a
/// file that exists — a search result points at something Wenlan indexed, so
/// a path that is not there is not openable anyway. The NUL is also checked
/// explicitly first: canonicalize's behaviour on an interior NUL is not
/// something to depend on.
/// Returns the canonical path the decision was made about, so the caller can
/// hand the OS exactly what was judged rather than the string it started
/// from. A `None` means the target is a URL and goes through unchanged.
fn refuse_unopenable_search_result(target: &str) -> Result<Option<std::path::PathBuf>, String> {
    if target.contains('\0') {
        return Err("Refusing to open a path containing a NUL byte.".to_string());
    }

    if let Some(scheme) = target_scheme(target) {
        if OPENABLE_SCHEMES.contains(&scheme.as_str()) {
            return Ok(None);
        }
        return Err(format!(
            "Refusing to open a \"{scheme}:\" link. Wenlan opens web pages and files on this computer."
        ));
    }

    let canonical = std::fs::canonicalize(target)
        .map_err(|_| format!("Refusing to open \"{target}\": no such file."))?;

    let is_file = std::fs::metadata(&canonical)
        .map(|meta| meta.is_file())
        .unwrap_or(false);
    if !is_file {
        return Err(format!(
            "Refusing to open \"{target}\". Wenlan opens documents, not folders or applications."
        ));
    }

    let is_indexed_document = canonical
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .is_some_and(|ext| INDEXED_DOCUMENT_SUFFIXES.contains(&ext.as_str()));
    if !is_indexed_document {
        return Err(format!(
            "Refusing to open \"{target}\". Wenlan opens the document types it indexes."
        ));
    }

    Ok(Some(canonical))
}

/// Hand a search result's `url` to the desktop's default handler.
///
/// This is the attacker-influenced twin of `open_file` — see
/// `refuse_unopenable_search_result` — kept as a separate command so the two
/// call sites carry two different trust levels instead of one shared,
/// necessarily looser check.
///
/// A path is handed to the OS in the canonical form the check ran against,
/// not the string it arrived as. Re-resolving the original would leave a
/// window in which a symlink swapped after the check but before the open
/// sends the OS somewhere the check never saw. A URL goes through unchanged.
#[tauri::command]
pub async fn open_search_result(url: String) -> Result<(), String> {
    let opened = match refuse_unopenable_search_result(&url)? {
        Some(canonical) => open::that_detached(&canonical),
        None => open::that_detached(&url),
    };
    opened.map_err(|e| format!("Failed to open: {e}"))
}

#[cfg(test)]
mod open_target_tests {
    use super::{refuse_unopenable_search_result, refuse_unopenable_target, target_scheme};

    fn allowed(target: &str) -> bool {
        refuse_unopenable_target(target).is_ok()
    }

    fn search_result_allowed(target: &str) -> bool {
        refuse_unopenable_search_result(target).is_ok()
    }

    #[test]
    fn filesystem_paths_have_no_scheme() {
        for path in [
            "/Users/someone/notes/a.md",
            "./relative/file.txt",
            "C:\\Users\\someone\\a.md",
            "\\\\server\\share\\a.md",
            "/Users/someone/notes/12:30 meeting.md",
        ] {
            assert_eq!(target_scheme(path), None, "{path}");
            assert!(allowed(path), "{path}");
        }
    }

    #[test]
    fn web_urls_are_opened() {
        for url in [
            "https://example.com/a",
            "http://example.com/a",
            "HTTPS://Example.com/A",
            "https://example.com/Terminal.app",
        ] {
            assert!(allowed(url), "{url}");
        }
    }

    #[test]
    fn a_memory_url_cannot_reach_another_scheme() {
        for url in [
            "javascript:alert(1)",
            "JavaScript:alert(1)",
            "data:text/html;base64,PHNjcmlwdD4=",
            "vbscript:msgbox(1)",
            "smb://attacker.example/share",
            "ftp://example.com/a",
            "mailto:someone@example.com",
            "x-apple-helpbasic://x",
            "ms-msdt:/id",
            // The one caller strips `file://` before invoking, so a local file
            // arrives as a bare path and gets the filesystem checks below.
            // Letting the URL form through would be a way around them.
            "file:///Applications/Utilities/Terminal.app",
            "file:///Users/someone/a.md",
        ] {
            assert!(!allowed(url), "{url}");
        }
    }

    /// The audit's own example: a memory whose `url` is an application, which
    /// carries no scheme and would sail past a scheme check.
    #[test]
    #[cfg(target_os = "macos")]
    fn a_memory_path_cannot_launch_an_application() {
        for path in [
            "/Applications/Utilities/Terminal.app",
            "/Applications/Utilities/TERMINAL.APP",
            "/Users/someone/Downloads/installer.pkg",
            "/Users/someone/Downloads/run-me.command",
            "/Users/someone/Downloads/thing.workflow",
            "/Users/someone/Downloads/redirect.webloc",
        ] {
            assert!(!allowed(path), "{path}");
        }
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn a_memory_path_cannot_launch_a_program() {
        for path in [
            "C:\\Users\\someone\\Downloads\\evil.exe",
            "C:\\Users\\someone\\Downloads\\EVIL.EXE",
            "C:\\Users\\someone\\Downloads\\evil.bat",
            "C:\\Users\\someone\\Downloads\\evil.ps1",
            "C:\\Users\\someone\\Downloads\\evil.lnk",
        ] {
            assert!(!allowed(path), "{path}");
        }
    }

    /// On Unix the extension carries no authority, so the executable bit does.
    /// A shell script a user could be tricked into clicking is refused; the
    /// notes and folders the Sources view opens are not.
    #[test]
    #[cfg(unix)]
    fn an_executable_file_is_refused_and_a_document_is_not() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let script = tmp.path().join("payload.sh");
        let note = tmp.path().join("note.md");
        std::fs::write(&script, "#!/bin/sh\necho hi\n").unwrap();
        std::fs::write(&note, "# hi\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::set_permissions(&note, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert!(!allowed(script.to_str().unwrap()));
        assert!(allowed(note.to_str().unwrap()));
        // A folder carries the executable bit too, and the Sources view opens
        // folders in the file manager. It must not be caught by this check.
        assert!(allowed(tmp.path().to_str().unwrap()));
    }

    #[test]
    fn a_refused_scheme_says_which_one_and_what_is_allowed() {
        let message = refuse_unopenable_target("JavaScript:alert(1)").unwrap_err();

        assert_eq!(
            message,
            "Refusing to open a \"javascript:\" link. \
             Wenlan opens web pages and files on this computer."
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn a_refused_program_says_which_one() {
        let message = refuse_unopenable_target("/Applications/Utilities/Terminal.app").unwrap_err();

        assert_eq!(
            message,
            "Refusing to run \"/Applications/Utilities/Terminal.app\". \
             Wenlan opens documents and folders, not programs."
        );
    }

    /// The reviewer's exact bypass: a symlink whose own name carries no
    /// launcher suffix, pointing at a directory that is a macOS app bundle.
    /// `fs::metadata` follows the link and sees a directory, so the old
    /// executable-bit check missed it; canonicalizing first fixes that.
    #[test]
    #[cfg(unix)]
    fn a_symlink_disguised_as_a_document_cannot_launch_the_app_bundle_it_points_at() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("Terminal.app");
        std::fs::create_dir(&bundle).unwrap();
        let link = tmp.path().join("Quarterly Notes");
        std::os::unix::fs::symlink(&bundle, &link).unwrap();

        assert!(!allowed(link.to_str().unwrap()));
    }

    #[test]
    fn a_target_containing_a_nul_byte_is_refused() {
        assert!(!allowed("/tmp/notes\0.md"));
    }

    #[test]
    fn a_real_directory_and_a_real_document_stay_openable_after_canonicalization() {
        let tmp = tempfile::tempdir().unwrap();
        let note = tmp.path().join("note.md");
        std::fs::write(&note, "# hi\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&note, std::fs::Permissions::from_mode(0o644)).unwrap();
        }

        assert!(allowed(tmp.path().to_str().unwrap()));
        assert!(allowed(note.to_str().unwrap()));
    }

    // ── refuse_unopenable_search_result ─────────────────────────────────

    #[test]
    fn an_indexed_document_is_allowed() {
        let tmp = tempfile::tempdir().unwrap();
        for name in ["note.md", "note.txt", "note.pdf"] {
            let path = tmp.path().join(name);
            std::fs::write(&path, "hi").unwrap();
            assert!(search_result_allowed(path.to_str().unwrap()), "{name}");
        }
    }

    #[test]
    fn a_search_result_web_url_is_allowed() {
        for url in ["https://example.com/a", "http://example.com/a"] {
            assert!(search_result_allowed(url), "{url}");
        }
    }

    #[test]
    fn a_search_result_cannot_reach_another_scheme() {
        for url in [
            "file:///Users/someone/a.md",
            "javascript:alert(1)",
            "smb://x/y",
        ] {
            assert!(!search_result_allowed(url), "{url}");
        }
    }

    /// Proves the NUL itself is what stops it: the prefix up to the NUL is a
    /// genuinely allowed file.
    #[test]
    fn a_search_result_with_an_embedded_nul_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let note = tmp.path().join("note.md");
        std::fs::write(&note, "hi").unwrap();
        let with_nul = format!("{}\0{}", note.to_str().unwrap(), ".txt");

        assert!(!search_result_allowed(&with_nul));
    }

    #[test]
    #[cfg(unix)]
    fn a_search_result_executable_script_is_refused() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let script = tmp.path().join("payload.sh");
        std::fs::write(&script, "#!/bin/sh\necho hi\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert!(!search_result_allowed(script.to_str().unwrap()));
    }

    /// The allowlist, not the executable bit, is what stops this: a `.exe`
    /// with no execute permission is refused purely for not being an indexed
    /// document type.
    #[test]
    #[cfg(unix)]
    fn a_search_result_windows_executable_is_refused_by_the_allowlist() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let exe = tmp.path().join("evil.exe");
        std::fs::write(&exe, "MZ").unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert!(!search_result_allowed(exe.to_str().unwrap()));
    }

    #[test]
    fn a_search_result_pointing_at_a_directory_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!search_result_allowed(tmp.path().to_str().unwrap()));
    }

    #[test]
    #[cfg(unix)]
    fn a_search_result_symlink_to_an_app_bundle_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let bundle = tmp.path().join("Bundle.app");
        std::fs::create_dir(&bundle).unwrap();
        let link = tmp.path().join("notes");
        std::os::unix::fs::symlink(&bundle, &link).unwrap();

        assert!(!search_result_allowed(link.to_str().unwrap()));
    }

    /// Canonicalization must see through the harmless name: the symlink's own
    /// name ends in `.md`, but it resolves to an executable script.
    #[test]
    #[cfg(unix)]
    fn a_search_result_symlink_disguised_as_markdown_is_refused() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let script = tmp.path().join("payload.sh");
        std::fs::write(&script, "#!/bin/sh\necho hi\n").unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let link = tmp.path().join("notes.md");
        std::os::unix::fs::symlink(&script, &link).unwrap();

        assert!(!search_result_allowed(link.to_str().unwrap()));
    }

    #[test]
    fn a_search_result_pointing_at_a_missing_file_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist.md");
        assert!(!search_result_allowed(missing.to_str().unwrap()));
    }
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirEntryDto {
    name: String,
    is_directory: bool,
}

/// List the immediate entries of a directory for the Sources browser.
///
/// The webview's fs plugin is unscoped (`fs:default`), so a registered
/// source's path isn't readable there on a fresh launch (only paths the user
/// just picked via the dialog are in scope). The Rust side has no such limit,
/// so it reads the directory directly. Names only — never file contents —
/// which is the same trust level the webview already has via `open_file`.
#[tauri::command]
pub async fn read_source_dir(path: String) -> Result<Vec<DirEntryDto>, String> {
    let rd = std::fs::read_dir(&path).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for entry in rd.flatten() {
        // ponytail: 10k-entry ceiling as a payload safety valve; real sources
        // are far smaller. Raise it if a source ever legitimately exceeds it.
        if out.len() >= 10_000 {
            break;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        // Hide dotfiles (.obsidian, .git, .DS_Store) — the daemon skips them at
        // ingest, so they aren't part of the "foundation" the browser shows.
        if name.starts_with('.') {
            continue;
        }
        let is_directory = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        out.push(DirEntryDto { name, is_directory });
    }
    Ok(out)
}

/// Offer the user's real Obsidian vaults as one-tap chips in the connect
/// flow, read from Obsidian's own vault registry. A convenience, never a
/// dependency: any read/parse failure resolves to an empty list rather than
/// an error (see `sources::obsidian::discover_vaults`).
#[tauri::command]
pub async fn detect_obsidian_vaults() -> Result<Vec<crate::sources::obsidian::ObsidianVault>, String>
{
    Ok(crate::sources::obsidian::discover_vaults(
        &crate::sources::obsidian::obsidian_registry_path(),
    ))
}

/// Read a text file's contents for inline preview in the Sources detail pane.
///
/// Same trust level as `open_file`, which already hands the whole file to the
/// native app; the webview's `fs:default` scope can't reach arbitrary
/// registered paths, so the Rust side reads it. The caller gates this to
/// markdown/plain-text extensions — never PDFs or binaries.
#[tauri::command]
pub async fn read_text_file(path: String) -> Result<String, String> {
    // ponytail: 512 KiB ceiling so a stray huge file can't wedge the webview.
    // Real notes are a few KB; raise it if a legit doc ever exceeds it.
    const MAX_BYTES: u64 = 512 * 1024;
    let meta = std::fs::metadata(&path).map_err(|e| e.to_string())?;
    if meta.len() > MAX_BYTES {
        return Err(format!(
            "file is {} KB — too large to preview inline (open it instead)",
            meta.len() / 1024
        ));
    }
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

// ── Index / watch path / source commands (local + config) ─────────────

#[tauri::command]
pub async fn get_index_status(state: tauri::State<'_, State>) -> Result<IndexStatus, String> {
    let (client, local) = {
        let state = state.read().await;
        (state.client.clone(), state.index_status.clone())
    };
    let daemon = client.status().await?;
    Ok(merge_daemon_status(local, daemon))
}

fn merge_daemon_status(mut local: IndexStatus, daemon: responses::StatusResponse) -> IndexStatus {
    local.files_indexed = daemon.files_indexed;
    local.sources_connected = daemon.sources_connected;
    local.reranker = daemon.reranker;
    local.reranker_light = daemon.reranker_light;
    local.reranker_mode = daemon.reranker_mode;
    local
}

#[tauri::command]
pub async fn list_watch_paths(state: tauri::State<'_, State>) -> Result<Vec<String>, String> {
    let state = state.read().await;
    Ok(state
        .watch_paths
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect())
}

#[tauri::command]
pub async fn add_watch_path(
    state: tauri::State<'_, State>,
    watcher: tauri::State<'_, WatcherState>,
    path: String,
) -> Result<(), String> {
    let path = PathBuf::from(&path);
    if !path.exists() {
        return Err(format!("Path does not exist: {}", path.display()));
    }
    if !path.is_dir() {
        return Err(format!("Path is not a directory: {}", path.display()));
    }

    {
        let mut app_state = state.write().await;
        if let Some(source) = app_state.sources.get_mut("local_files") {
            if let Some(local) = source
                .as_any_mut()
                .downcast_mut::<crate::sources::local_files::LocalFilesSource>()
            {
                local.add_watch_path(path.clone());
            }
        }
        if !app_state.watch_paths.contains(&path) {
            app_state.watch_paths.push(path.clone());
        }
    }

    let mut watcher_guard = watcher.lock().await;
    if watcher_guard.is_none() {
        let state_arc = state.inner().clone();
        *watcher_guard =
            Some(crate::indexer::create_file_watcher(state_arc).map_err(|e| e.to_string())?);
    }
    if let Some(w) = watcher_guard.as_mut() {
        crate::indexer::watch_path(w, &path).map_err(|e| e.to_string())?;
    }

    // Persist as a Source entry in config
    {
        let mut cfg = config::load_config();
        if !cfg.sources.iter().any(|s| s.path == path) {
            let dirname = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "dir".to_string());
            let slug = crate::sources::obsidian::slugify(&dirname);
            let id = format!("directory-{}", slug);
            cfg.sources.push(crate::sources::Source {
                id,
                source_type: crate::sources::SourceType::Directory,
                path: path.clone(),
                status: crate::sources::SyncStatus::Active,
                last_sync: None,
                file_count: 0,
                memory_count: 0,
                last_sync_errors: 0,
                last_sync_error_detail: None,
            });
            config::save_config(&cfg).map_err(|e| e.to_string())?;
        }
    }

    // Trigger initial index
    let state_inner = state.inner().clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::indexer::sync_source("local_files", &state_inner).await {
            log::error!("Initial index after add_watch_path failed: {}", e);
        }
    });

    Ok(())
}

#[tauri::command]
pub async fn remove_watch_path(
    state: tauri::State<'_, State>,
    watcher: tauri::State<'_, WatcherState>,
    path: String,
) -> Result<(), String> {
    let path = PathBuf::from(&path);

    {
        let mut app_state = state.write().await;
        if let Some(source) = app_state.sources.get_mut("local_files") {
            if let Some(local) = source
                .as_any_mut()
                .downcast_mut::<crate::sources::local_files::LocalFilesSource>()
            {
                local.remove_watch_path(&path);
            }
        }
        app_state.watch_paths.retain(|p| p != &path);
    }

    let mut watcher_guard = watcher.lock().await;
    if let Some(w) = watcher_guard.as_mut() {
        crate::indexer::unwatch_path(w, &path);
    }

    // Remove from config.sources
    {
        let mut cfg = config::load_config();
        let before = cfg.sources.len();
        cfg.sources.retain(|s| s.path != path);
        if cfg.sources.len() != before {
            config::save_config(&cfg).map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

#[tauri::command]
pub async fn reindex(state: tauri::State<'_, State>) -> Result<(), String> {
    let state_inner = state.inner().clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::indexer::sync_source("local_files", &state_inner).await {
            log::error!("Reindex failed: {}", e);
        }
    });
    Ok(())
}

#[tauri::command]
pub async fn connect_source(
    state: tauri::State<'_, State>,
    source_name: String,
) -> Result<(), String> {
    {
        let mut s = state.write().await;
        let source = s
            .sources
            .get_mut(&source_name)
            .ok_or_else(|| format!("Unknown source: {}", source_name))?;
        source.connect().await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub async fn disconnect_source(
    state: tauri::State<'_, State>,
    source_name: String,
) -> Result<(), String> {
    {
        let mut s = state.write().await;
        let source = s
            .sources
            .get_mut(&source_name)
            .ok_or_else(|| format!("Unknown source: {}", source_name))?;
        source.disconnect().await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
pub async fn sync_source(
    state: tauri::State<'_, State>,
    source_name: String,
) -> Result<(), String> {
    let state_inner = state.inner().clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::indexer::sync_source(&source_name, &state_inner).await {
            log::error!("Sync failed for {}: {}", source_name, e);
        }
    });
    Ok(())
}

#[tauri::command]
pub async fn list_sources(state: tauri::State<'_, State>) -> Result<Vec<SourceStatus>, String> {
    let state = state.read().await;
    Ok(state.list_sources().await)
}

// ═══════════════════════════════════════════════════════════════════════
// DATA COMMANDS — proxied through the daemon via WenlanClient
// ═══════════════════════════════════════════════════════════════════════

// ── Search ────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn search(
    state: tauri::State<'_, State>,
    query: String,
    limit: Option<usize>,
    source_filter: Option<String>,
) -> Result<Vec<SearchResult>, String> {
    let client = daemon_client(&state).await;
    let req = requests::SearchRequest {
        query,
        limit: limit.unwrap_or(10),
        source_filter,
        space: None,
    };
    let resp: responses::SearchResponse = client.post_json("/api/search", &req).await?;
    Ok(search_results_from_response(resp))
}

fn append_supplemental_pages(
    mut results: Vec<SearchResult>,
    supplemental_pages: Option<Vec<SearchResult>>,
) -> Vec<SearchResult> {
    if let Some(mut pages) = supplemental_pages {
        results.append(&mut pages);
    }
    results
}

fn search_results_from_response(resp: responses::SearchResponse) -> Vec<SearchResult> {
    append_supplemental_pages(resp.results, resp.supplemental_pages)
}

// ── Memory CRUD ───────────────────────────────────────────────────────

#[tauri::command]
pub async fn ingest_webpage(
    state: tauri::State<'_, State>,
    req: requests::IngestWebpageRequest,
) -> Result<responses::IngestResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.ingest_webpage(req).await
}

#[tauri::command]
pub async fn distill_review(
    state: tauri::State<'_, State>,
) -> Result<crate::api::DistillReviewResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.distill_review().await
}

#[tauri::command]
pub async fn redistill_page(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<crate::api::PageRedistillResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.redistill_page(&page_id).await
}

#[tauri::command]
pub async fn store_memory(
    state: tauri::State<'_, State>,
    req: StoreMemoryRequest,
) -> Result<StoreMemoryResponse, String> {
    let client = daemon_client(&state).await;
    let daemon_req = requests::StoreMemoryRequest {
        content: req.content,
        memory_type: req.memory_type,
        space: req.domain.into(),
        source_agent: req.source_agent,
        title: req.title,
        confidence: req.confidence,
        supersedes: req.supersedes,
        entity: None,
        entity_id: None,
        structured_fields: req.structured_fields,
        retrieval_cue: req.retrieval_cue,
    };
    let resp: responses::StoreMemoryResponse =
        client.post_json("/api/memory/store", &daemon_req).await?;
    Ok(StoreMemoryResponse {
        source_id: resp.source_id,
        warnings: resp.warnings,
        enrichment: resp.enrichment,
        hint: resp.hint,
    })
}

#[tauri::command]
pub async fn confirm_memory(
    state: tauri::State<'_, State>,
    source_id: String,
    confirmed: bool,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::ConfirmRequest { confirmed };
    let resp: responses::ConfirmResponse = client
        .post_json(
            &format!(
                "/api/memory/confirm/{}",
                percent_encode_path_segment(&source_id)
            ),
            &req,
        )
        .await?;
    // The daemon 404s an unknown id; an older daemon answers 200 without
    // `updated`, which defaults to `true` so that response still reads as
    // success. Branch on the flag so an explicit `false` reports a no-op.
    if !resp.updated {
        return Err(format!("memory {} not found", source_id));
    }
    Ok(())
}

#[tauri::command]
pub async fn set_stability_cmd(
    state: tauri::State<'_, State>,
    source_id: String,
    stability: String,
) -> Result<(), String> {
    // "confirmed" goes through the confirm endpoint (it also flips the confirmed flag);
    // every other stability value is a direct PUT to /api/memory/{id}/stability.
    let client = daemon_client(&state).await;
    let encoded_id = percent_encode_path_segment(&source_id);
    if stability == "confirmed" {
        let req = requests::ConfirmRequest { confirmed: true };
        let resp: responses::ConfirmResponse = client
            .post_json(&format!("/api/memory/confirm/{}", encoded_id), &req)
            .await?;
        if !resp.updated {
            return Err(format!("memory {} not found", source_id));
        }
    } else {
        let req = requests::SetStabilityRequest { stability };
        let _resp: responses::SuccessResponse = client
            .put_json(&format!("/api/memory/{}/stability", encoded_id), &req)
            .await?;
    }
    Ok(())
}

#[tauri::command]
pub async fn delete_memory(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let _resp: responses::DeleteResponse = client
        .delete_path(&format!(
            "/api/memory/delete/{}",
            percent_encode_path_segment(&source_id)
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn reclassify_memory_cmd(
    state: tauri::State<'_, State>,
    source_id: String,
    memory_type: String,
) -> Result<String, String> {
    let client = daemon_client(&state).await;
    let req = requests::ReclassifyMemoryRequest { memory_type };
    let resp: responses::ReclassifyMemoryResponse = client
        .post_json(
            &format!(
                "/api/memory/reclassify/{}",
                percent_encode_path_segment(&source_id)
            ),
            &req,
        )
        .await?;
    Ok(resp.memory_type)
}

// ── Memory detail / list ──────────────────────────────────────────────

#[tauri::command]
pub async fn list_memories_cmd(
    state: tauri::State<'_, State>,
    domain: Option<String>,
    memory_type: Option<String>,
    confirmed: Option<bool>,
    limit: Option<usize>,
) -> Result<Vec<MemoryItem>, String> {
    let client = daemon_client(&state).await;
    let daemon_req = requests::ListMemoriesRequest {
        memory_type,
        space: domain,
        limit: limit.unwrap_or(200),
        confirmed,
    };
    let resp: responses::ListMemoriesResponse =
        client.post_json("/api/memory/list", &daemon_req).await?;

    // The daemon returns IndexedFileInfo; the UI expects MemoryItem. Most
    // fields overlap; extras like entity_id and quality aren't surfaced by
    // the list endpoint and aren't needed for the list view.
    // Keep the client-side filter as a defensive fallback for older daemons.
    let items: Vec<MemoryItem> = resp
        .memories
        .into_iter()
        .filter(|info| match confirmed {
            Some(want) => info.confirmed == Some(want),
            None => true,
        })
        .map(|info| MemoryItem {
            source_id: info.source_id.clone(),
            title: info.title,
            content: info.content,
            summary: info.summary,
            memory_type: info.memory_type,
            space: info.space,
            source_agent: info.source_agent,
            confidence: info.confidence,
            confirmed: info.confirmed.unwrap_or(false),
            stability: info.stability,
            pinned: info.pinned,
            supersedes: None,
            last_modified: info.last_modified,
            chunk_count: info.chunk_count,
            entity_id: None,
            quality: None,
            is_archived: info.is_archived,
            is_recap: info.source_id.starts_with("recap_"),
            enrichment_status: String::from("raw"),
            supersede_mode: String::from("hide"),
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            access_count: 0,
            version: 1,
            changelog: None,
            pending_revision: false,
            merged_from: None,
        })
        .collect();
    Ok(items)
}

#[tauri::command]
pub async fn get_memory_detail(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<Option<MemoryItem>, String> {
    let client = state.read().await.client.clone();
    get_memory_detail_response(&client, &source_id).await
}

async fn get_memory_detail_response(
    client: &crate::api::WenlanClient,
    source_id: &str,
) -> Result<Option<MemoryItem>, String> {
    let response: Option<responses::MemoryDetailResponse> = client
        .get_optional_json(&format!(
            "/api/memory/{}/detail",
            percent_encode_path_segment(source_id)
        ))
        .await?;
    Ok(response.and_then(|response| response.memory))
}

#[tauri::command]
pub async fn get_enrichment_status(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<wenlan_types::EnrichmentStatusResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_enrichment_status(&source_id).await
}

#[tauri::command]
pub async fn get_memory_revisions(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<responses::ListMemoryRevisionsResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_memory_revisions(&source_id).await
}

#[tauri::command]
pub async fn list_memories_by_ids(
    state: tauri::State<'_, State>,
    ids: Vec<String>,
) -> Result<Vec<MemoryItem>, String> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let client = state.read().await.client.clone();
    let ids_param = ids
        .iter()
        .map(|id| percent_encode_path_segment(id))
        .collect::<Vec<_>>()
        .join(",");
    let resp: responses::PinnedMemoriesResponse = client
        .get_json(&format!("/api/memory/by-ids?ids={}", ids_param))
        .await?;
    Ok(resp.memories)
}

#[tauri::command]
pub async fn get_memory_stats_cmd(state: tauri::State<'_, State>) -> Result<MemoryStats, String> {
    let client = daemon_client(&state).await;
    let resp: responses::MemoryStatsResponse = client.get_json("/api/memory/stats").await?;
    Ok(resp.stats)
}

#[tauri::command]
pub async fn get_home_stats(state: tauri::State<'_, State>) -> Result<HomeStats, String> {
    let client = daemon_client(&state).await;
    client.get_json::<HomeStats>("/api/home-stats").await
}

#[tauri::command]
pub async fn update_memory_cmd(
    state: tauri::State<'_, State>,
    source_id: String,
    content: Option<String>,
    domain: Option<String>,
    confirmed: Option<bool>,
    memory_type: Option<String>,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::UpdateMemoryRequest {
        content,
        space: domain,
        confirmed,
        memory_type,
    };
    let _resp: responses::SuccessResponse = client
        .put_json(
            &format!(
                "/api/memory/{}/update",
                percent_encode_path_segment(&source_id)
            ),
            &req,
        )
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn get_version_chain_cmd(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<Vec<MemoryVersionItem>, String> {
    let client = daemon_client(&state).await;
    let resp: responses::VersionChainResponse = client
        .get_json(&format!(
            "/api/memory/{}/versions",
            percent_encode_path_segment(&source_id)
        ))
        .await?;
    Ok(resp.versions)
}

// ── Indexed files / chunks ────────────────────────────────────────────

#[tauri::command]
pub async fn list_indexed_files(
    state: tauri::State<'_, State>,
) -> Result<Vec<IndexedFileInfo>, String> {
    let client = daemon_client(&state).await;
    let resp: responses::IndexedFilesResponse = client.get_json("/api/indexed-files").await?;
    Ok(resp.files)
}

#[tauri::command]
pub async fn get_chunks(
    state: tauri::State<'_, State>,
    _source: String,
    source_id: String,
) -> Result<Vec<MemoryDetail>, String> {
    let client = daemon_client(&state).await;
    let chunks: Vec<MemoryDetail> = client
        .get_json(&format!(
            "/api/chunks/{}",
            percent_encode_path_segment(&source_id)
        ))
        .await?;
    Ok(chunks)
}

#[tauri::command]
pub async fn update_chunk(
    state: tauri::State<'_, State>,
    id: String,
    content: String,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::UpdateChunkRequest { content };
    let _resp: responses::SuccessResponse = client
        .put_json(
            &format!("/api/chunks/{}/update", percent_encode_path_segment(&id)),
            &req,
        )
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn delete_file_chunks(
    state: tauri::State<'_, State>,
    source: String,
    source_id: String,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let _resp: responses::DeleteResponse = client
        .delete_path(&format!(
            "/api/documents/{}/{}",
            percent_encode_path_segment(&source),
            percent_encode_path_segment(&source_id)
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn delete_by_time_range(
    state: tauri::State<'_, State>,
    start: i64,
    end: i64,
) -> Result<(), String> {
    // Remove from in-memory activity list
    {
        let mut s = state.write().await;
        s.completed_activities
            .retain(|a| !(a.started_at <= end && a.ended_at >= start));
        s.save_all_activities();
    }
    let client = daemon_client(&state).await;
    let req = requests::DeleteByTimeRangeRequest { start, end };
    let _resp: responses::DeleteCountResponse =
        client.delete_json("/api/chunks/time-range", &req).await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct BulkDeleteItem {
    pub source: String,
    pub source_id: String,
}

#[tauri::command]
pub async fn delete_bulk(
    state: tauri::State<'_, State>,
    items: Vec<BulkDeleteItem>,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::BulkDeleteRequest {
        items: items
            .into_iter()
            .map(|i| requests::BulkDeleteItem {
                source: i.source,
                source_id: i.source_id,
            })
            .collect(),
    };
    let _resp: responses::DeleteCountResponse =
        client.post_json("/api/chunks/delete-bulk", &req).await?;
    Ok(())
}

// ── Quick capture / ingest ────────────────────────────────────────────

#[tauri::command]
pub async fn quick_capture(
    state: tauri::State<'_, State>,
    req: QuickCaptureRequest,
) -> Result<usize, String> {
    let source_id = format!("manual_{}", chrono::Utc::now().timestamp());

    let title = req.title.unwrap_or_else(|| {
        let first_line = req.content.lines().next().unwrap_or("Untitled");
        if first_line.chars().count() > 60 {
            format!("{}...", first_line.chars().take(60).collect::<String>())
        } else {
            first_line.to_string()
        }
    });

    let mut metadata = HashMap::new();
    if let Some(tags) = &req.tags {
        metadata.insert("tags".to_string(), tags.join(","));
    }
    // IngestMemoryRequest doesn't have typed memory_type/domain fields, so
    // forward them via metadata. The daemon's post-ingest enrichment can
    // pick them up as hints; otherwise they stay as searchable metadata.
    if let Some(ref mt) = req.memory_type {
        metadata.insert("memory_type".to_string(), mt.clone());
    }
    if let Some(ref d) = req.domain {
        metadata.insert("domain".to_string(), d.clone());
    }

    let client = daemon_client(&state).await;
    let ingest_req = requests::IngestMemoryRequest {
        source: "manual".to_string(),
        source_id: source_id.clone(),
        title,
        content: req.content,
        url: None,
        tags: req.tags,
        metadata: Some(metadata),
    };
    let resp: responses::IngestResponse =
        client.post_json("/api/ingest/memory", &ingest_req).await?;
    Ok(resp.chunks_created)
}

#[tauri::command]
pub async fn import_memories_cmd(
    state: tauri::State<'_, State>,
    app_handle: tauri::AppHandle,
    source: String,
    content: String,
    _label: Option<String>,
) -> Result<responses::ImportMemoriesResponse, String> {
    let client = daemon_client(&state).await;
    let req = requests::ImportMemoriesRequest {
        source,
        content,
        label: _label,
        space: Default::default(),
    };
    let result: responses::ImportMemoriesResponse =
        client.post_json("/api/import/memories", &req).await?;

    // Emit event for UI refresh
    use tauri::Emitter;
    let _ = app_handle.emit("import-complete", &result);

    Ok(result)
}

#[tauri::command]
pub async fn import_chat_export(
    state: tauri::State<'_, State>,
    path: String,
) -> Result<wenlan_types::import::ImportChatExportResponse, String> {
    let client = daemon_client(&state).await;
    client.import_chat_export(&path).await
}

#[tauri::command]
pub async fn list_pending_imports(
    state: tauri::State<'_, State>,
) -> Result<Vec<wenlan_types::import::PendingImport>, String> {
    let client = daemon_client(&state).await;
    client.list_pending_imports().await
}

// ── Onboarding milestones ───────────────────────────────────────────

#[tauri::command]
pub async fn list_onboarding_milestones(
    state: tauri::State<'_, State>,
) -> Result<Vec<wenlan_types::onboarding::MilestoneRecord>, String> {
    // Snapshot the client out of the guard so we never hold the RwLock across
    // the HTTP .await — holding it would block all writers for the duration
    // of the request. `WenlanClient` is `Clone` and cheap to clone.
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.list_onboarding_milestones().await
}

#[tauri::command]
pub async fn acknowledge_onboarding_milestone(
    state: tauri::State<'_, State>,
    id: String,
) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.acknowledge_onboarding_milestone(&id).await
}

#[tauri::command]
pub async fn reset_onboarding_milestones(state: tauri::State<'_, State>) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.reset_onboarding_milestones().await
}

#[tauri::command]
pub async fn save_temp_file(bytes: Vec<u8>, filename: String) -> Result<String, String> {
    let dir = std::env::temp_dir().join("origin-chat-import");
    std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir temp: {e}"))?;
    let safe: String = filename
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '.' || *c == '-' || *c == '_')
        .collect();
    // Prevent path traversal: ".." passes the char filter since "." is allowed.
    // Also reject empty filenames (e.g., input was all slashes).
    if safe.is_empty() || safe == "." || safe == ".." {
        return Err("Invalid filename".to_string());
    }
    // Add UUID prefix to prevent overwrites between concurrent imports.
    let unique_name = format!("{}_{}", uuid::Uuid::new_v4(), safe);
    let path = dir.join(&unique_name);
    std::fs::write(&path, &bytes).map_err(|e| format!("write temp: {e}"))?;
    Ok(path.to_string_lossy().into_owned())
}

// ── Knowledge graph / entities ────────────────────────────────────────

#[tauri::command]
pub async fn create_entity_cmd(
    state: tauri::State<'_, State>,
    name: String,
    entity_type: String,
    domain: Option<String>,
) -> Result<String, String> {
    let client = daemon_client(&state).await;
    let req = requests::CreateEntityRequest {
        name,
        entity_type,
        space: domain.into(),
        source_agent: None,
        confidence: None,
    };
    let resp: responses::CreateEntityResponse =
        client.post_json("/api/memory/entities", &req).await?;
    Ok(resp.id)
}

#[tauri::command]
pub async fn list_entities_cmd(
    state: tauri::State<'_, State>,
    entity_type: Option<String>,
    domain: Option<String>,
) -> Result<Vec<Entity>, String> {
    let client = daemon_client(&state).await;
    let req = requests::ListEntitiesRequest {
        entity_type,
        space: domain,
    };
    let resp: responses::ListEntitiesResponse =
        client.post_json("/api/memory/entities/list", &req).await?;
    Ok(resp.entities)
}

#[tauri::command]
pub async fn search_entities_cmd(
    state: tauri::State<'_, State>,
    query: String,
    limit: Option<usize>,
) -> Result<Vec<EntitySearchResult>, String> {
    let client = daemon_client(&state).await;
    let req = requests::SearchEntitiesRequest {
        query,
        limit: limit.unwrap_or(20),
    };
    let resp: responses::SearchEntitiesResponse = client
        .post_json("/api/memory/entities/search", &req)
        .await?;
    Ok(resp.results)
}

#[tauri::command]
pub async fn get_entity_detail_cmd(
    state: tauri::State<'_, State>,
    entity_id: String,
) -> Result<EntityDetail, String> {
    let client = daemon_client(&state).await;
    client
        .get_json(&format!(
            "/api/memory/entities/{}",
            percent_encode_path_segment(&entity_id)
        ))
        .await
}

/// One bulk read of the whole knowledge graph for the current space scope.
///
/// The Graph view used to fan out one `get_entity_detail_cmd` per entity and
/// could only afford the first 20, which drew every other connected entity as
/// an isolate. This returns entities, relations, memory nodes and memory links
/// in a single typed response.
#[tauri::command]
pub async fn get_knowledge_graph_cmd(
    state: tauri::State<'_, State>,
) -> Result<KnowledgeGraphResponse, String> {
    let client = daemon_client(&state).await;
    client.get_json("/api/memory/graph").await
}

#[tauri::command]
pub async fn update_observation_cmd(
    state: tauri::State<'_, State>,
    observation_id: String,
    content: String,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::UpdateObservationRequest { content };
    let _resp: responses::SuccessResponse = client
        .put_json(
            &format!(
                "/api/memory/observations/{}",
                percent_encode_path_segment(&observation_id)
            ),
            &req,
        )
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn delete_observation_cmd(
    state: tauri::State<'_, State>,
    observation_id: String,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let _resp: responses::SuccessResponse = client
        .delete_path(&format!(
            "/api/memory/observations/{}",
            percent_encode_path_segment(&observation_id)
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn delete_entity_cmd(
    state: tauri::State<'_, State>,
    entity_id: String,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let _resp: responses::SuccessResponse = client
        .delete_path(&format!(
            "/api/memory/entities/{}/delete",
            percent_encode_path_segment(&entity_id)
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn confirm_entity_cmd(
    state: tauri::State<'_, State>,
    entity_id: String,
    confirmed: bool,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::ConfirmEntityRequest { confirmed };
    let _resp: responses::SuccessResponse = client
        .put_json(
            &format!(
                "/api/memory/entities/{}/confirm",
                percent_encode_path_segment(&entity_id)
            ),
            &req,
        )
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn confirm_observation_cmd(
    state: tauri::State<'_, State>,
    observation_id: String,
    confirmed: bool,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::ConfirmObservationRequest { confirmed };
    let _resp: responses::SuccessResponse = client
        .put_json(
            &format!(
                "/api/memory/observations/{}/confirm",
                percent_encode_path_segment(&observation_id)
            ),
            &req,
        )
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn add_observation_cmd(
    state: tauri::State<'_, State>,
    entity_id: String,
    content: String,
    source_agent: Option<String>,
    confidence: Option<f32>,
) -> Result<String, String> {
    let client = daemon_client(&state).await;
    let req = requests::AddObservationRequest {
        entity_id,
        content,
        source_agent,
        confidence,
    };
    let resp: responses::AddObservationResponse =
        client.post_json("/api/memory/observations", &req).await?;
    Ok(resp.id)
}

// ── Profile & agents ──────────────────────────────────────────────────

#[tauri::command]
pub async fn get_profile(state: tauri::State<'_, State>) -> Result<Option<Profile>, String> {
    let client = daemon_client(&state).await;
    match client
        .get_json::<responses::ProfileResponse>("/api/profile")
        .await
    {
        Ok(resp) => Ok(Some(Profile {
            id: resp.id,
            name: resp.name,
            display_name: resp.display_name,
            email: resp.email,
            bio: resp.bio,
            avatar_path: resolve_profile_avatar_path(resp.avatar_path),
            created_at: resp.created_at,
            updated_at: resp.updated_at,
        })),
        Err(_) => Ok(None),
    }
}

#[tauri::command]
pub async fn update_profile(
    state: tauri::State<'_, State>,
    _id: String,
    name: Option<String>,
    display_name: Option<String>,
    email: Option<String>,
    bio: Option<String>,
    avatar_path: Option<String>,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::UpdateProfileRequest {
        name,
        display_name,
        email,
        bio,
        avatar_path,
    };
    let _resp: responses::ProfileResponse = client.put_json("/api/profile", &req).await?;
    Ok(())
}

#[tauri::command]
pub async fn list_agents(state: tauri::State<'_, State>) -> Result<Vec<AgentConnection>, String> {
    let client = daemon_client(&state).await;
    let agents: Vec<responses::AgentResponse> = client.get_json("/api/agents").await?;
    Ok(agents
        .into_iter()
        .map(|a| AgentConnection {
            id: a.id,
            name: a.name,
            display_name: a.display_name,
            agent_type: a.agent_type,
            description: a.description,
            enabled: a.enabled,
            trust_level: a.trust_level,
            last_seen_at: a.last_seen_at,
            memory_count: a.memory_count,
            created_at: a.created_at,
            updated_at: a.updated_at,
        })
        .collect())
}

#[tauri::command]
pub async fn get_agent(
    state: tauri::State<'_, State>,
    name: String,
) -> Result<Option<AgentConnection>, String> {
    let client = daemon_client(&state).await;
    match client
        .get_json::<responses::AgentResponse>(&format!(
            "/api/agents/{}",
            percent_encode_path_segment(&name)
        ))
        .await
    {
        Ok(a) => Ok(Some(AgentConnection {
            id: a.id,
            name: a.name,
            display_name: a.display_name,
            agent_type: a.agent_type,
            description: a.description,
            enabled: a.enabled,
            trust_level: a.trust_level,
            last_seen_at: a.last_seen_at,
            memory_count: a.memory_count,
            created_at: a.created_at,
            updated_at: a.updated_at,
        })),
        Err(_) => Ok(None),
    }
}

#[tauri::command]
pub async fn update_agent(
    state: tauri::State<'_, State>,
    name: String,
    agent_type: Option<String>,
    description: Option<String>,
    enabled: Option<bool>,
    trust_level: Option<String>,
    display_name: Option<String>,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::UpdateAgentRequest {
        agent_type,
        description,
        enabled,
        trust_level,
        display_name,
    };
    let _resp: responses::AgentResponse = client
        .put_json(
            &format!("/api/agents/{}", percent_encode_path_segment(&name)),
            &req,
        )
        .await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct DeleteAgentResponse {
    deleted: String,
}

async fn delete_agent_response(
    client: &crate::api::WenlanClient,
    name: &str,
) -> Result<DeleteAgentResponse, String> {
    client
        .delete_path(&format!(
            "/api/agents/{}",
            percent_encode_path_segment(name)
        ))
        .await
}

#[tauri::command]
pub async fn delete_agent(state: tauri::State<'_, State>, name: String) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let DeleteAgentResponse { deleted: _deleted } = delete_agent_response(&client, &name).await?;
    Ok(())
}

// ── Avatar commands ───────────────────────────────────────────────────

fn avatar_storage_dir() -> PathBuf {
    crate::identity_paths::app_data_dir().join("avatars")
}

fn legacy_avatar_storage_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(custom) = std::env::var_os("ORIGIN_DATA_DIR") {
        dirs.push(PathBuf::from(custom).join("avatars"));
    }
    dirs.push(crate::identity_paths::legacy_app_data_dir().join("avatars"));
    dirs
}

fn resolve_profile_avatar_path(avatar_path: Option<String>) -> Option<String> {
    let avatar_path = avatar_path?;
    if avatar_path.is_empty() {
        return None;
    }

    let path = PathBuf::from(&avatar_path);
    if path.exists() {
        return Some(avatar_path);
    }

    let parent = path.parent()?;
    if !legacy_avatar_storage_dirs()
        .iter()
        .any(|legacy_dir| legacy_dir == parent)
    {
        return None;
    }

    let filename = path.file_name()?;
    let migrated = avatar_storage_dir().join(filename);
    if migrated.exists() {
        return Some(migrated.to_string_lossy().to_string());
    }

    None
}

#[tauri::command]
pub async fn set_avatar(
    state: tauri::State<'_, State>,
    source_path: String,
) -> Result<String, String> {
    let source = std::path::Path::new(&source_path);
    if !source.exists() {
        return Err(format!("Source file not found: {}", source_path));
    }

    let ext = source.extension().and_then(|e| e.to_str()).unwrap_or("png");

    let avatars_dir = avatar_storage_dir();
    std::fs::create_dir_all(&avatars_dir)
        .map_err(|e| format!("Failed to create avatars directory: {}", e))?;

    let filename = format!("{}.{}", uuid::Uuid::new_v4(), ext);
    let dest = avatars_dir.join(&filename);

    std::fs::copy(source, &dest).map_err(|e| format!("Failed to copy avatar file: {}", e))?;

    let dest_str = dest.to_string_lossy().to_string();

    // Update profile via daemon
    let client = daemon_client(&state).await;
    let req = requests::UpdateProfileRequest {
        name: None,
        display_name: None,
        email: None,
        bio: None,
        avatar_path: Some(dest_str.clone()),
    };
    let _resp: responses::ProfileResponse = client.put_json("/api/profile", &req).await?;

    Ok(dest_str)
}

#[tauri::command]
pub async fn get_avatar_data_url(state: tauri::State<'_, State>) -> Result<Option<String>, String> {
    let client = daemon_client(&state).await;
    let profile = match client
        .get_json::<responses::ProfileResponse>("/api/profile")
        .await
    {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    let Some(avatar_path) = resolve_profile_avatar_path(profile.avatar_path) else {
        return Ok(None);
    };

    let path = std::path::Path::new(&avatar_path);
    if !path.exists() {
        return Ok(None);
    }

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
    let mime = match ext {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "image/png",
    };

    let bytes = std::fs::read(path).map_err(|e| format!("Failed to read avatar: {}", e))?;
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(Some(format!("data:{};base64,{}", mime, b64)))
}

#[tauri::command]
pub async fn remove_avatar(state: tauri::State<'_, State>) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let profile = match client
        .get_json::<responses::ProfileResponse>("/api/profile")
        .await
    {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };

    if let Some(avatar_path) = resolve_profile_avatar_path(profile.avatar_path) {
        let _ = std::fs::remove_file(avatar_path);
    }

    let req = requests::UpdateProfileRequest {
        name: None,
        display_name: None,
        email: None,
        bio: None,
        avatar_path: Some(String::new()),
    };
    let _resp: responses::ProfileResponse = client.put_json("/api/profile", &req).await?;
    Ok(())
}

// ── Pin/unpin ─────────────────────────────────────────────────────────

#[tauri::command]
pub async fn pin_memory(state: tauri::State<'_, State>, source_id: String) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let _resp: responses::SuccessResponse = client
        .post_empty(&format!(
            "/api/memory/{}/pin",
            percent_encode_path_segment(&source_id)
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn unpin_memory(state: tauri::State<'_, State>, source_id: String) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let _resp: responses::SuccessResponse = client
        .post_empty(&format!(
            "/api/memory/{}/unpin",
            percent_encode_path_segment(&source_id)
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn list_pinned_memories(
    state: tauri::State<'_, State>,
) -> Result<Vec<MemoryItem>, String> {
    let client = daemon_client(&state).await;
    let resp: responses::PinnedMemoriesResponse = client.get_json("/api/memory/pinned").await?;
    Ok(resp.memories)
}

// ── Pending revisions ─────────────────────────────────────────────────

#[tauri::command]
pub async fn accept_pending_revision(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<responses::RevisionAcceptResponse, String> {
    let client = daemon_client(&state).await;
    client
        .post_empty(&format!(
            "/api/memory/revision/{}/accept",
            percent_encode_path_segment(&source_id)
        ))
        .await
}

#[tauri::command]
pub async fn dismiss_pending_revision(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<responses::RevisionDismissResponse, String> {
    let client = daemon_client(&state).await;
    client
        .post_empty(&format!(
            "/api/memory/revision/{}/dismiss",
            percent_encode_path_segment(&source_id)
        ))
        .await
}

// ── Contradiction flags ────────────────────────────────────────────────

#[tauri::command]
pub async fn dismiss_contradiction(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<responses::ContradictionDismissResponse, String> {
    let client = daemon_client(&state).await;
    client
        .post_empty(&format!(
            "/api/memory/contradiction/{}/dismiss",
            percent_encode_path_segment(&source_id)
        ))
        .await
}

// ── Refinery queue ──────────────────────────────────────────────────────

#[tauri::command]
pub async fn list_refinements(
    state: tauri::State<'_, State>,
    limit: Option<usize>,
) -> Result<responses::ListRefinementsResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.list_refinements(limit).await
}

#[tauri::command]
pub async fn accept_refinement(
    state: tauri::State<'_, State>,
    id: String,
) -> Result<responses::AcceptRefinementResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.accept_refinement(&id).await
}

#[tauri::command]
pub async fn reject_refinement(
    state: tauri::State<'_, State>,
    id: String,
) -> Result<responses::RejectRefinementResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.reject_refinement(&id).await
}

#[tauri::command]
pub async fn get_pending_revision(
    state: tauri::State<'_, State>,
    source_id: String,
) -> Result<Option<PendingRevision>, String> {
    let client = daemon_client(&state).await;
    let revision: Option<PendingRevision> = client
        .get_json(&format!(
            "/api/memory/pending-revision/{}",
            percent_encode_path_segment(&source_id)
        ))
        .await?;
    Ok(revision)
}

#[tauri::command]
pub async fn list_pending_revisions(
    state: tauri::State<'_, State>,
    limit: Option<usize>,
) -> Result<Vec<responses::PendingRevisionItem>, String> {
    let client = daemon_client(&state).await;
    let path = match limit {
        Some(limit) => format!("/api/memory/pending-revisions?limit={limit}"),
        None => "/api/memory/pending-revisions".to_string(),
    };
    client.get_json(&path).await
}

// ── Briefing / narrative ──────────────────────────────────────────────

#[tauri::command]
pub async fn get_briefing(state: tauri::State<'_, State>) -> Result<BriefingResponse, String> {
    let client = daemon_client(&state).await;
    let resp: BriefingResponse = client.get_json("/api/briefing").await?;
    Ok(resp)
}

#[tauri::command]
pub async fn get_pending_contradictions(
    _state: tauri::State<'_, State>,
) -> Result<Vec<ContradictionItem>, String> {
    Ok(vec![])
}

#[tauri::command]
pub async fn get_profile_narrative(
    state: tauri::State<'_, State>,
) -> Result<NarrativeResponse, String> {
    let client = daemon_client(&state).await;
    let resp: NarrativeResponse = client.get_json("/api/profile/narrative").await?;
    Ok(resp)
}

#[tauri::command]
pub async fn regenerate_narrative(
    state: tauri::State<'_, State>,
) -> Result<NarrativeResponse, String> {
    let client = daemon_client(&state).await;
    let resp: NarrativeResponse = client
        .post_empty("/api/profile/narrative/regenerate")
        .await?;
    Ok(resp)
}

// ── Agent activity ────────────────────────────────────────────────────

#[tauri::command]
pub async fn list_agent_activity(
    state: tauri::State<'_, State>,
    limit: Option<usize>,
    agent_name: Option<String>,
    since: Option<i64>,
) -> Result<Vec<AgentActivityRow>, String> {
    let client = daemon_client(&state).await;
    let mut path = format!("/api/activities?limit={}", limit.unwrap_or(50));
    if let Some(name) = agent_name {
        path.push_str(&format!(
            "&agent_name={}",
            percent_encode_path_segment(&name)
        ));
    }
    if let Some(since_val) = since {
        path.push_str(&format!("&since={}", since_val));
    }
    let resp: responses::ActivityResponse = client.get_json(&path).await?;
    Ok(resp.activities)
}

// ── Spaces ────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn list_spaces(state: tauri::State<'_, State>) -> Result<Vec<Space>, String> {
    let client = daemon_client(&state).await;
    client.get_json("/api/spaces").await
}

#[tauri::command]
pub async fn get_space(
    state: tauri::State<'_, State>,
    name: String,
) -> Result<Option<Space>, String> {
    // No direct get-by-name endpoint, but we can list and filter
    let client = daemon_client(&state).await;
    let spaces: Vec<Space> = client.get_json("/api/spaces").await?;
    Ok(spaces.into_iter().find(|sp| sp.name == name))
}

#[derive(Debug, Deserialize)]
struct DeleteSpaceResponse {
    deleted: String,
}

#[derive(Debug, Deserialize)]
struct ToggleSpaceStarredResponse {
    starred: bool,
}

async fn delete_space_response(
    client: &crate::api::WenlanClient,
    name: &str,
) -> Result<DeleteSpaceResponse, String> {
    client
        .delete_path(&format!(
            "/api/spaces/{}",
            percent_encode_path_segment(name)
        ))
        .await
}

async fn toggle_space_starred_response(
    client: &crate::api::WenlanClient,
    name: &str,
) -> Result<ToggleSpaceStarredResponse, String> {
    client
        .post_empty(&format!(
            "/api/spaces/{}/star",
            percent_encode_path_segment(name)
        ))
        .await
}

#[tauri::command]
pub async fn create_space(
    state: tauri::State<'_, State>,
    name: String,
    description: Option<String>,
) -> Result<Space, String> {
    let client = daemon_client(&state).await;
    let req = requests::CreateSpaceRequest { name, description };
    client.post_json("/api/spaces", &req).await
}

#[tauri::command]
pub async fn update_space(
    state: tauri::State<'_, State>,
    name: String,
    new_name: String,
    description: Option<String>,
) -> Result<Space, String> {
    let client = daemon_client(&state).await;
    let req = requests::UpdateSpaceRequest {
        new_name: Some(new_name),
        description,
    };
    client
        .put_json(
            &format!("/api/spaces/{}", percent_encode_path_segment(&name)),
            &req,
        )
        .await
}

#[tauri::command]
pub async fn delete_space(state: tauri::State<'_, State>, name: String) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    let DeleteSpaceResponse { deleted: _deleted } = delete_space_response(&client, &name).await?;
    Ok(())
}

#[tauri::command]
pub async fn move_space(
    state: tauri::State<'_, State>,
    from: String,
    to: String,
) -> Result<crate::api::MoveSpaceResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.move_space(&from, &to).await
}

#[tauri::command]
pub async fn confirm_space(state: tauri::State<'_, State>, name: String) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let _resp: responses::SuccessResponse = client
        .post_empty(&format!(
            "/api/spaces/{}/confirm",
            percent_encode_path_segment(&name)
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn reorder_space(
    state: tauri::State<'_, State>,
    name: String,
    new_order: i64,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::ReorderSpaceRequest { name, new_order };
    let _resp: responses::SuccessResponse = client.post_json("/api/spaces/reorder", &req).await?;
    Ok(())
}

#[tauri::command]
pub async fn toggle_space_starred(
    state: tauri::State<'_, State>,
    name: String,
) -> Result<bool, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    let resp = toggle_space_starred_response(&client, &name).await?;
    Ok(resp.starred)
}

// Legacy space commands (local SpaceStore — these are being superseded by daemon spaces)
#[tauri::command]
pub async fn set_document_space(
    state: tauri::State<'_, State>,
    _source: String,
    source_id: String,
    space_id: String,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let req = requests::SetDocumentSpaceRequest {
        space_name: space_id,
    };
    let _resp: responses::SuccessResponse = client
        .post_json(
            &format!(
                "/api/documents/{}/space",
                percent_encode_path_segment(&source_id)
            ),
            &req,
        )
        .await?;
    Ok(())
}

#[cfg(test)]
mod space_command_type_tests {
    use super::*;

    #[allow(dead_code)]
    async fn delete_space_response_uses_typed_deleted_envelope(client: crate::api::WenlanClient) {
        let _: Result<DeleteSpaceResponse, String> = delete_space_response(&client, "space").await;
    }

    #[allow(dead_code)]
    async fn toggle_space_starred_response_uses_typed_starred_envelope(
        client: crate::api::WenlanClient,
    ) {
        let _: Result<ToggleSpaceStarredResponse, String> =
            toggle_space_starred_response(&client, "space").await;
    }

    #[allow(dead_code)]
    async fn move_space_command_uses_typed_affected_envelope(state: tauri::State<'_, State>) {
        let _: Result<crate::api::MoveSpaceResponse, String> =
            move_space(state, "Inbox".to_string(), "Archive".to_string()).await;
    }

    #[allow(dead_code)]
    async fn distill_review_command_uses_typed_review_envelope(state: tauri::State<'_, State>) {
        let _: Result<crate::api::DistillReviewResponse, String> = distill_review(state).await;
    }

    #[allow(dead_code)]
    async fn redistill_page_command_uses_typed_response(state: tauri::State<'_, State>) {
        let _: Result<crate::api::PageRedistillResponse, String> =
            redistill_page(state, "page_1".to_string()).await;
    }

    #[test]
    fn space_response_envelopes_deserialize_daemon_payloads() {
        let deleted: DeleteSpaceResponse = serde_json::from_value(serde_json::json!({
            "deleted": "Engineering"
        }))
        .unwrap();
        assert_eq!(deleted.deleted, "Engineering");

        let starred: ToggleSpaceStarredResponse = serde_json::from_value(serde_json::json!({
            "starred": true
        }))
        .unwrap();
        assert!(starred.starred);
    }
}

// ── Tags ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct TagData {
    pub tags: Vec<String>,
    pub document_tags: HashMap<String, Vec<String>>,
    pub categories: Vec<String>,
    pub document_categories: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
struct SetDocumentTagsRequest {
    source: String,
    tags: Vec<String>,
}

#[tauri::command]
pub async fn list_all_tags(state: tauri::State<'_, State>) -> Result<TagData, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    let inventory = client.list_tag_inventory().await?;
    Ok(tag_data_from_inventory(inventory))
}

fn tag_data_from_inventory(inventory: crate::api::TagInventoryResponse) -> TagData {
    TagData {
        tags: inventory.tags,
        document_tags: inventory.document_tags,
        categories: vec![],
        document_categories: HashMap::new(),
    }
}

#[tauri::command]
pub async fn set_document_tags(
    state: tauri::State<'_, State>,
    source: String,
    source_id: String,
    tags: Vec<String>,
) -> Result<Vec<String>, String> {
    let client = state.read().await.client.clone();
    let req = SetDocumentTagsRequest { source, tags };
    let resp: responses::TagsResponse = client
        .put_json(
            &format!(
                "/api/documents/{}/tags",
                percent_encode_path_segment(&source_id)
            ),
            &req,
        )
        .await?;
    Ok(resp.tags)
}

#[tauri::command]
pub async fn delete_tag(state: tauri::State<'_, State>, name: String) -> Result<(), String> {
    let client = state.read().await.client.clone();
    let _resp: responses::SuccessResponse = client
        .delete_path(&format!("/api/tags/{}", percent_encode_path_segment(&name)))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn suggest_tags(
    state: tauri::State<'_, State>,
    source: String,
    source_id: String,
    last_modified: i64,
) -> Result<Vec<String>, String> {
    // Snapshot everything we need from AppState inside a scoped block so
    // the read guard is dropped before the HTTP call. Holding a
    // `tokio::sync::RwLock` read guard across `.await` would block any
    // writer (config updates, sensor toggles, etc.) for the full
    // duration of the round-trip. See AGENTS.md "Repository invariants".
    //
    // Local signal: the app that was active at the document's timestamp.
    // Activities are tracked in-process by the Tauri app (the daemon has
    // no view of them), so look the app name up here and pass it to the
    // daemon as a merge hint.
    let (client, activity_app): (crate::api::WenlanClient, Option<String>) = {
        let s = state.read().await;
        let activity_app = s
            .list_activity_summaries()
            .into_iter()
            .find(|a| last_modified >= a.started_at && last_modified <= a.ended_at)
            .and_then(|a| a.app_names.first().cloned());
        (s.client.clone(), activity_app)
    }; // guard dropped here

    // Build the query string with percent-encoded values —
    // source/source_id are usually simple ASCII identifiers but may contain
    // spaces or slashes, and the app name commonly has spaces.
    let mut path = String::from("/api/suggest-tags?source=");
    path.push_str(&percent_encode_path_segment(&source));
    path.push_str("&source_id=");
    path.push_str(&percent_encode_path_segment(&source_id));
    if let Some(ref app) = activity_app {
        path.push_str("&activity_app=");
        path.push_str(&percent_encode_path_segment(app));
    }

    let resp: responses::TagsResponse = client.get_json(&path).await?;
    Ok(resp.tags)
}

// ── Sessions ───────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_session_snapshots(
    state: tauri::State<'_, State>,
    limit: Option<usize>,
) -> Result<Vec<wenlan_types::SessionSnapshot>, String> {
    let client = daemon_client(&state).await;
    let path = format!("/api/snapshots?limit={}", limit.unwrap_or(10));
    client.get_json(&path).await
}

#[tauri::command]
pub async fn get_snapshot_captures(
    state: tauri::State<'_, State>,
    snapshot_id: String,
) -> Result<Vec<wenlan_types::SnapshotCapture>, String> {
    let client = daemon_client(&state).await;
    client
        .get_json(&format!(
            "/api/snapshots/{}/captures",
            percent_encode_path_segment(&snapshot_id)
        ))
        .await
}

#[tauri::command]
pub async fn get_snapshot_captures_with_content(
    state: tauri::State<'_, State>,
    snapshot_id: String,
) -> Result<Vec<wenlan_types::SnapshotCaptureWithContent>, String> {
    let client = daemon_client(&state).await;
    client
        .get_json(&format!(
            "/api/snapshots/{}/captures-with-content",
            percent_encode_path_segment(&snapshot_id)
        ))
        .await
}

#[tauri::command]
pub async fn delete_snapshot(
    state: tauri::State<'_, State>,
    snapshot_id: String,
) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let _resp: responses::SuccessResponse = client
        .post_empty(&format!(
            "/api/snapshots/{}/delete",
            percent_encode_path_segment(&snapshot_id)
        ))
        .await?;
    Ok(())
}

// ── Memory nurture ────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_nurture_cards_cmd(
    state: tauri::State<'_, State>,
    _limit: Option<usize>,
    _domain: Option<String>,
) -> Result<Vec<MemoryItem>, String> {
    let client = daemon_client(&state).await;
    let resp: responses::NurtureCardsResponse = client.get_json("/api/memory/nurture").await?;
    Ok(resp.cards)
}

#[tauri::command]
pub async fn correct_memory_cmd(
    state: tauri::State<'_, State>,
    source_id: String,
    correction_prompt: String,
) -> Result<String, String> {
    let client = daemon_client(&state).await;
    let req = requests::CorrectMemoryRequest { correction_prompt };
    let CorrectMemoryResponse {
        corrected,
        source_id: _source_id,
    } = correct_memory_response(&client, &source_id, &req).await?;
    Ok(corrected)
}

#[derive(Debug, Deserialize)]
struct CorrectMemoryResponse {
    corrected: String,
    source_id: String,
}

async fn correct_memory_response(
    client: &crate::api::WenlanClient,
    source_id: &str,
    req: &requests::CorrectMemoryRequest,
) -> Result<CorrectMemoryResponse, String> {
    client
        .post_json(
            &format!(
                "/api/memory/{}/correct",
                percent_encode_path_segment(source_id)
            ),
            req,
        )
        .await
}

#[cfg(test)]
mod remaining_json_command_type_tests {
    use super::*;

    #[allow(dead_code)]
    async fn delete_agent_response_uses_typed_deleted_envelope(client: crate::api::WenlanClient) {
        let _: Result<DeleteAgentResponse, String> = delete_agent_response(&client, "agent").await;
    }

    #[allow(dead_code)]
    async fn correct_memory_response_uses_typed_correction_envelope(
        client: crate::api::WenlanClient,
    ) {
        let req = requests::CorrectMemoryRequest {
            correction_prompt: "fix it".to_string(),
        };
        let _: Result<CorrectMemoryResponse, String> =
            correct_memory_response(&client, "mem", &req).await;
    }

    #[allow(dead_code)]
    async fn public_commands_keep_existing_surfaces(state: tauri::State<'_, State>) {
        let _: Result<(), String> = delete_agent(state.clone(), String::new()).await;
        let _: Result<String, String> =
            correct_memory_cmd(state, String::new(), String::new()).await;
    }

    #[test]
    fn remaining_response_envelopes_deserialize_daemon_payloads() {
        let deleted: DeleteAgentResponse = serde_json::from_value(serde_json::json!({
            "deleted": "agent-name"
        }))
        .unwrap();
        assert_eq!(deleted.deleted, "agent-name");

        let corrected: CorrectMemoryResponse = serde_json::from_value(serde_json::json!({
            "corrected": "updated memory text",
            "source_id": "mem_123"
        }))
        .unwrap();
        assert_eq!(corrected.corrected, "updated memory text");
        assert_eq!(corrected.source_id, "mem_123");
    }
}

// ── Pages ──────────────────────────────────────────────────────────

/// Extract the `page` object from the daemon's `{ "page": {...} }` wrapper.
/// Raw-JSON passthrough: the pinned wenlan-types structs would silently drop
/// fields the daemon added after 0.9.2 (e.g. `citations`); the Rust layer
/// consumes no Page fields on this path, so TypeScript types the response.
fn page_from_wire(mut wire: serde_json::Value) -> Option<serde_json::Value> {
    match wire.get_mut("page") {
        Some(v) if !v.is_null() => Some(v.take()),
        _ => None,
    }
}

#[tauri::command]
pub async fn get_page(
    state: tauri::State<'_, State>,
    id: String,
) -> Result<Option<serde_json::Value>, String> {
    let client = state.read().await.client.clone();
    // The daemon returns 404 when the page doesn't exist, which reqwest
    // turns into an error. Distinguish "not found" from real errors so the
    // frontend sees None for the former and a real error for the latter —
    // rather than the previous silent `Err(_) => Ok(None)` which hid
    // wrapper/deserialization bugs behind a "not found" UI.
    match client
        .get_json::<serde_json::Value>(&format!("/api/pages/{}", percent_encode_path_segment(&id)))
        .await
    {
        Ok(wire) => Ok(page_from_wire(wire)),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("404") || msg.to_lowercase().contains("not found") {
                Ok(None)
            } else {
                Err(format!("get_page failed: {}", msg))
            }
        }
    }
}

#[cfg(test)]
mod get_page_tests {
    use super::*;

    #[test]
    fn page_from_wire_extracts_page_object_with_unknown_fields() {
        let wire = serde_json::json!({
            "page": { "id": "p1", "citations": [{ "occurrence": 1, "marker": 1 }] }
        });
        let page = page_from_wire(wire).expect("page present");
        assert_eq!(page["id"], "p1");
        assert_eq!(page["citations"][0]["occurrence"], 1);
    }

    #[test]
    fn page_from_wire_maps_null_and_missing_page_to_none() {
        assert!(page_from_wire(serde_json::json!({ "page": null })).is_none());
        assert!(page_from_wire(serde_json::json!({})).is_none());
    }
}

fn authored_page_request(
    title: String,
    content: String,
    space: Option<String>,
) -> requests::CreateConceptRequest {
    requests::CreateConceptRequest {
        title: title.trim().to_string(),
        content: content.trim().to_string(),
        summary: None,
        entity_id: None,
        space: space
            .and_then(|value| {
                let normalized = value.trim();
                (!normalized.is_empty()).then(|| normalized.to_string())
            })
            .into(),
        source_memory_ids: Vec::new(),
        creation_kind: Some("authored".to_string()),
        workspace: None,
    }
}

#[tauri::command]
pub async fn create_page(
    state: tauri::State<'_, State>,
    title: String,
    content: String,
    space: Option<String>,
) -> Result<responses::CreatePageResponse, String> {
    let client = state.read().await.client.clone();
    let request = authored_page_request(title, content, space);
    client.post_json("/api/pages", &request).await
}

#[derive(Debug, Serialize)]
struct DraftWriteRequest {
    draft_id: String,
    title: String,
    content: String,
    space: Option<String>,
}

#[derive(Debug, Serialize)]
struct DraftUpdateRequest {
    expected_version: i64,
    title: String,
    content: String,
    space: Option<String>,
}

#[derive(Debug, Serialize)]
struct DraftVersionRequest {
    expected_version: i64,
}

#[derive(Debug, Deserialize)]
struct DraftPageResponse {
    page: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct DraftDiscardResponse {
    status: String,
}

fn normalize_draft_space(space: Option<String>) -> Option<String> {
    space.and_then(|value| {
        let normalized = value.trim();
        (!normalized.is_empty()).then(|| normalized.to_string())
    })
}

async fn create_page_draft_response(
    client: &crate::api::WenlanClient,
    request: DraftWriteRequest,
) -> Result<serde_json::Value, String> {
    let response: DraftPageResponse = client.post_json("/api/pages/drafts", &request).await?;
    Ok(response.page)
}

async fn update_page_draft_response(
    client: &crate::api::WenlanClient,
    id: &str,
    request: DraftUpdateRequest,
) -> Result<serde_json::Value, String> {
    let response: DraftPageResponse = client
        .put_json(
            &format!("/api/pages/drafts/{}", percent_encode_path_segment(id)),
            &request,
        )
        .await?;
    Ok(response.page)
}

async fn publish_page_draft_response(
    client: &crate::api::WenlanClient,
    id: &str,
    request: DraftVersionRequest,
) -> Result<serde_json::Value, String> {
    let response: DraftPageResponse = client
        .post_json(
            &format!(
                "/api/pages/drafts/{}/publish",
                percent_encode_path_segment(id)
            ),
            &request,
        )
        .await?;
    Ok(response.page)
}

async fn discard_page_draft_response(
    client: &crate::api::WenlanClient,
    id: &str,
    request: DraftVersionRequest,
) -> Result<(), String> {
    let response: DraftDiscardResponse = client
        .delete_json(
            &format!("/api/pages/drafts/{}", percent_encode_path_segment(id)),
            &request,
        )
        .await?;
    if response.status == "deleted" {
        Ok(())
    } else {
        Err(format!(
            "discard_page_draft returned unexpected status: {}",
            response.status
        ))
    }
}

#[tauri::command]
pub async fn create_page_draft(
    state: tauri::State<'_, State>,
    client_draft_id: String,
    title: String,
    content: String,
    space: Option<String>,
) -> Result<serde_json::Value, String> {
    let client = state.read().await.client.clone();
    create_page_draft_response(
        &client,
        DraftWriteRequest {
            draft_id: client_draft_id,
            title,
            content,
            space: normalize_draft_space(space),
        },
    )
    .await
}

#[tauri::command]
pub async fn update_page_draft(
    state: tauri::State<'_, State>,
    id: String,
    expected_version: i64,
    title: String,
    content: String,
    space: Option<String>,
) -> Result<serde_json::Value, String> {
    let client = state.read().await.client.clone();
    update_page_draft_response(
        &client,
        &id,
        DraftUpdateRequest {
            expected_version,
            title,
            content,
            space: normalize_draft_space(space),
        },
    )
    .await
}

#[tauri::command]
pub async fn publish_page_draft(
    state: tauri::State<'_, State>,
    id: String,
    expected_version: i64,
) -> Result<serde_json::Value, String> {
    let client = state.read().await.client.clone();
    publish_page_draft_response(&client, &id, DraftVersionRequest { expected_version }).await
}

#[tauri::command]
pub async fn discard_page_draft(
    state: tauri::State<'_, State>,
    id: String,
    expected_version: i64,
) -> Result<(), String> {
    let client = state.read().await.client.clone();
    discard_page_draft_response(&client, &id, DraftVersionRequest { expected_version }).await
}

#[cfg(test)]
mod create_page_command_tests {
    use super::*;

    #[test]
    fn authored_page_request_normalizes_the_optional_space_and_sets_the_contract() {
        let request = authored_page_request(
            "  Durable note  ".to_string(),
            "  Source-backed body  ".to_string(),
            Some("  Wenlan  ".to_string()),
        );
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["title"], "Durable note");
        assert_eq!(value["content"], "Source-backed body");
        assert_eq!(value["space"], "Wenlan");
        assert_eq!(value["creation_kind"], "authored");
        assert_eq!(value["source_memory_ids"], serde_json::json!([]));
        assert!(value["summary"].is_null());
        assert!(value["entity_id"].is_null());
        assert!(value["workspace"].is_null());
    }

    #[test]
    fn authored_page_request_maps_a_blank_space_to_none() {
        let request = authored_page_request(
            "Title".to_string(),
            "Body".to_string(),
            Some("   ".to_string()),
        );
        let value = serde_json::to_value(request).unwrap();

        assert!(value["space"].is_null());
    }
}

#[cfg(test)]
mod page_draft_command_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn serve_once(
        response_body: &'static str,
    ) -> (crate::api::WenlanClient, tokio::task::JoinHandle<String>) {
        serve_status_once("200 OK", response_body).await
    }

    async fn serve_status_once(
        status: &'static str,
        response_body: &'static str,
    ) -> (crate::api::WenlanClient, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = vec![0_u8; 8192];
            let size = stream.read(&mut bytes).await.unwrap();
            let request = String::from_utf8_lossy(&bytes[..size]).to_string();
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body,
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });
        (
            crate::api::WenlanClient::with_base_url(format!("http://{address}")),
            handle,
        )
    }

    #[test]
    fn draft_space_normalization_maps_blank_to_none_without_touching_meaningful_names() {
        assert_eq!(normalize_draft_space(Some("   ".to_string())), None);
        assert_eq!(
            normalize_draft_space(Some("  Wenlan  ".to_string())),
            Some("Wenlan".to_string())
        );
    }

    #[tokio::test]
    async fn create_draft_preserves_whitespace_and_uses_the_drafts_collection() {
        let (client, request) =
            serve_once(r#"{"page":{"id":"draft-1","version":1,"status":"draft"}}"#).await;
        let page = create_page_draft_response(
            &client,
            DraftWriteRequest {
                draft_id: "page_client-1".to_string(),
                title: "  title  ".to_string(),
                content: "  body  \n".to_string(),
                space: None,
            },
        )
        .await
        .unwrap();
        let request = request.await.unwrap();

        assert_eq!(page["id"], "draft-1");
        assert!(request.starts_with("POST /api/pages/drafts HTTP/1.1\r\n"));
        assert!(request.contains(r#""draft_id":"page_client-1""#));
        assert!(request.contains(r#""title":"  title  ""#));
        assert!(request.contains(r#""content":"  body  \n""#));
        assert!(request.contains(r#""space":null"#));
    }

    #[tokio::test]
    async fn update_draft_uses_put_and_snake_case_version_payload() {
        let (client, request) =
            serve_once(r#"{"page":{"id":"draft-1","version":4,"status":"draft"}}"#).await;
        let page = update_page_draft_response(
            &client,
            "draft-1",
            DraftUpdateRequest {
                expected_version: 3,
                title: "Title".to_string(),
                content: "Body".to_string(),
                space: Some("Wenlan".to_string()),
            },
        )
        .await
        .unwrap();
        let request = request.await.unwrap();

        assert_eq!(page["version"], 4);
        assert!(request.starts_with("PUT /api/pages/drafts/draft-1 HTTP/1.1\r\n"));
        assert!(request.contains(r#""expected_version":3"#));
        assert!(request.contains(r#""space":"Wenlan""#));
    }

    #[tokio::test]
    async fn publish_draft_posts_the_version_and_unwraps_the_page() {
        let (client, request) =
            serve_once(r#"{"page":{"id":"draft-1","version":4,"status":"active"}}"#).await;
        let page = publish_page_draft_response(
            &client,
            "draft-1",
            DraftVersionRequest {
                expected_version: 3,
            },
        )
        .await
        .unwrap();
        let request = request.await.unwrap();

        assert_eq!(page["status"], "active");
        assert!(request.starts_with("POST /api/pages/drafts/draft-1/publish HTTP/1.1\r\n"));
        assert!(request.contains(r#""expected_version":3"#));
    }

    #[tokio::test]
    async fn discard_draft_sends_a_versioned_delete_body() {
        let (client, request) = serve_once(r#"{"status":"deleted"}"#).await;
        discard_page_draft_response(
            &client,
            "draft-1",
            DraftVersionRequest {
                expected_version: 4,
            },
        )
        .await
        .unwrap();
        let request = request.await.unwrap();

        assert!(request.starts_with("DELETE /api/pages/drafts/draft-1 HTTP/1.1\r\n"));
        assert!(request.contains(r#""expected_version":4"#));
    }

    #[tokio::test]
    async fn structured_daemon_conflict_json_survives_the_tauri_string_boundary() {
        let body = r#"{"code":"page_title_conflict","error":"exists","existing_page_id":"page-9","existing_page_title":"Existing"}"#;
        let (client, request) = serve_status_once("409 Conflict", body).await;
        let error = publish_page_draft_response(
            &client,
            "draft-1",
            DraftVersionRequest {
                expected_version: 3,
            },
        )
        .await
        .unwrap_err();
        request.await.unwrap();

        assert!(error.contains(body));
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum UpdatePageOutcome {
    Saved,
    UpgradeRequired {
        #[serde(rename = "reportedVersion")]
        reported_version: String,
        #[serde(rename = "requiredFloor")]
        required_floor: String,
    },
    Conflict {
        message: String,
    },
    Failure {
        kind: UpdatePageFailureKind,
        status: u16,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UpdatePageFailureKind {
    NotFound,
    AuthRequired,
    PayloadTooLarge,
    Validation,
    RateLimited,
    Server,
    Other,
}

#[derive(Deserialize)]
struct DaemonErrorEnvelope {
    error: String,
}

fn page_update_error_message(status: u16, body: &str) -> String {
    if let Ok(envelope) = serde_json::from_str::<DaemonErrorEnvelope>(body) {
        if !envelope.error.trim().is_empty() {
            return envelope.error;
        }
    }

    if status == 409 {
        return "Page changed while you were editing.".to_string();
    }

    let body = body.trim();
    if body.is_empty() {
        format!("HTTP {status}")
    } else {
        body.to_string()
    }
}

fn map_page_update_result(
    result: Result<responses::PageWriteResponse, crate::api::PageUpdateRequestError>,
) -> Result<UpdatePageOutcome, String> {
    match result {
        Ok(response) if response.ok && !response.gated => Ok(UpdatePageOutcome::Saved),
        Ok(response) => Ok(UpdatePageOutcome::Failure {
            kind: UpdatePageFailureKind::Other,
            status: 200,
            message: if response.gated {
                "Daemon staged the page update instead of saving it.".to_string()
            } else {
                "Daemon did not confirm the page update.".to_string()
            },
        }),
        Err(crate::api::PageUpdateRequestError::TransportOrDecode(message)) => Err(message),
        Err(crate::api::PageUpdateRequestError::Http { status, body }) => {
            let message = page_update_error_message(status, &body);
            if status == 409 {
                return Ok(UpdatePageOutcome::Conflict { message });
            }

            let kind = match status {
                404 => UpdatePageFailureKind::NotFound,
                401 | 403 => UpdatePageFailureKind::AuthRequired,
                413 => UpdatePageFailureKind::PayloadTooLarge,
                400 | 422 => UpdatePageFailureKind::Validation,
                429 => UpdatePageFailureKind::RateLimited,
                500..=599 => UpdatePageFailureKind::Server,
                _ => UpdatePageFailureKind::Other,
            };
            Ok(UpdatePageOutcome::Failure {
                kind,
                status,
                message,
            })
        }
    }
}

// Stable v0.14.1 is the first released artifact proven to preserve exact page
// source while retaining the v0.14.0 CAS/idempotency contract.
const PAGE_EDIT_DAEMON_FLOOR: &str = "0.14.1";

/// Parse `major.minor.patch`, rejecting any pre-release: a `-rc` build has not
/// shipped the contract yet, so gating on it must not unlock behavior tied to
/// that contract. Build metadata (`+sha`) does not affect precedence and is
/// ignored.
///
/// Hand-rolled rather than pulled from the `semver` crate because adding a
/// dependency means editing `app/Cargo.toml`, which the release-please guard
/// hook blocks. Shared by every daemon-floor gate in this crate — page edits
/// (`daemon_version_supports_page_edit` below) and page reviews
/// (`crate::page_review::daemon_version_supports_review`) both parse the
/// version they read from `/api/health` through this one function.
pub(crate) fn parse_release_version(version: &str) -> Option<(u64, u64, u64)> {
    let core = version.split('+').next()?;
    if core.contains('-') {
        return None;
    }
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

fn daemon_version_supports_page_edit(version: &str) -> bool {
    let Some(candidate) = parse_release_version(version) else {
        return false;
    };
    let floor = parse_release_version(PAGE_EDIT_DAEMON_FLOOR)
        .expect("page editor daemon floor is a static valid release version");

    candidate >= floor
}

fn page_edit_upgrade_required(reported_version: &str) -> Option<UpdatePageOutcome> {
    (!daemon_version_supports_page_edit(reported_version)).then(|| {
        UpdatePageOutcome::UpgradeRequired {
            reported_version: reported_version.to_string(),
            required_floor: PAGE_EDIT_DAEMON_FLOOR.to_string(),
        }
    })
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageEditorDiagnosticEvent {
    DaemonFloorBlocked,
    EditorFallback,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageEditorFallbackReason {
    Load,
    Construction,
}

fn page_editor_diagnostic_message(
    event: PageEditorDiagnosticEvent,
    reported_version: Option<&str>,
    required_floor: Option<&str>,
    reason: Option<PageEditorFallbackReason>,
) -> Result<String, String> {
    match event {
        PageEditorDiagnosticEvent::DaemonFloorBlocked => {
            if reason.is_some() {
                return Err("daemon floor diagnostic cannot contain a fallback reason".to_string());
            }
            let required_floor =
                required_floor.ok_or_else(|| "required floor is missing".to_string())?;
            Ok(format!(
                "[page-editor] daemon_floor_blocked reported_version={} required_floor={required_floor}",
                reported_version.unwrap_or("unavailable"),
            ))
        }
        PageEditorDiagnosticEvent::EditorFallback => {
            if reported_version.is_some() || required_floor.is_some() {
                return Err("editor fallback diagnostic cannot contain daemon fields".to_string());
            }
            let reason = match reason.ok_or_else(|| "fallback reason is missing".to_string())? {
                PageEditorFallbackReason::Load => "load",
                PageEditorFallbackReason::Construction => "construction",
            };
            Ok(format!("[page-editor] editor_fallback reason={reason}"))
        }
    }
}

#[tauri::command]
pub fn record_page_editor_diagnostic(
    event: PageEditorDiagnosticEvent,
    reported_version: Option<String>,
    required_floor: Option<String>,
    reason: Option<PageEditorFallbackReason>,
) -> Result<(), String> {
    let message = page_editor_diagnostic_message(
        event,
        reported_version.as_deref(),
        required_floor.as_deref(),
        reason,
    )?;
    log::warn!("{message}");
    Ok(())
}

#[cfg(test)]
mod update_page_outcome_tests {
    use super::*;
    use crate::api::PageUpdateRequestError;

    #[test]
    fn successful_page_write_maps_to_saved() {
        let result = map_page_update_result(Ok(responses::PageWriteResponse {
            ok: true,
            revision_card_id: Some("revision-1".to_string()),
            gated: false,
        }));

        assert_eq!(result, Ok(UpdatePageOutcome::Saved));
    }

    #[test]
    fn conflict_envelope_maps_to_typed_conflict() {
        let result = map_page_update_result(Err(PageUpdateRequestError::Http {
            status: 409,
            body: r#"{"error":"expected version 7, found 8"}"#.to_string(),
        }));

        assert_eq!(
            result,
            Ok(UpdatePageOutcome::Conflict {
                message: "expected version 7, found 8".to_string(),
            })
        );
    }
}

#[cfg(test)]
mod page_editor_diagnostic_tests {
    use super::*;

    #[test]
    fn daemon_floor_log_contains_versions_but_no_page_or_source_field() {
        let message = page_editor_diagnostic_message(
            PageEditorDiagnosticEvent::DaemonFloorBlocked,
            Some("0.13.9"),
            Some("0.14.0"),
            None,
        )
        .unwrap();

        assert_eq!(
            message,
            "[page-editor] daemon_floor_blocked reported_version=0.13.9 required_floor=0.14.0"
        );
        assert!(!message.contains("page_id"));
        assert!(!message.contains("content"));
    }
}

#[tauri::command]
pub async fn update_page(
    state: tauri::State<'_, State>,
    id: String,
    content: String,
    expected_version: i64,
    caller_id: String,
    operation_id: String,
) -> Result<UpdatePageOutcome, String> {
    if caller_id != "wenlan-app" {
        return Err("Unsupported page-update caller identity".to_string());
    }
    if operation_id.trim().is_empty() {
        return Err("Page-update operation identity is required".to_string());
    }

    let client = state.read().await.client.clone();
    let health = client.health().await?;
    if let Some(outcome) = page_edit_upgrade_required(&health.version) {
        log::warn!(
            "[page-editor] blocked save: daemon_version={} required_floor={}",
            health.version,
            PAGE_EDIT_DAEMON_FLOOR,
        );
        return Ok(outcome);
    }

    let req = requests::UpdatePageRequest {
        content,
        source_memory_ids: Vec::new(),
        expected_version: Some(expected_version),
        caller_id: Some("wenlan-app".to_string()),
        operation_id: Some(operation_id),
    };
    map_page_update_result(client.post_page_update(&id, &req).await)
}

#[tauri::command]
pub async fn archive_page(state: tauri::State<'_, State>, id: String) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let PageStatusResponse { status: _status } = archive_page_response(&client, &id).await?;
    Ok(())
}

#[tauri::command]
pub async fn delete_page(state: tauri::State<'_, State>, id: String) -> Result<(), String> {
    let client = daemon_client(&state).await;
    let PageStatusResponse { status: _status } = delete_page_response(&client, &id).await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct PageStatusResponse {
    status: String,
}

async fn archive_page_response(
    client: &crate::api::WenlanClient,
    id: &str,
) -> Result<PageStatusResponse, String> {
    client
        .post_empty(&format!(
            "/api/pages/{}/archive",
            percent_encode_path_segment(id)
        ))
        .await
}

async fn delete_page_response(
    client: &crate::api::WenlanClient,
    id: &str,
) -> Result<PageStatusResponse, String> {
    client
        .delete_path(&format!("/api/pages/{}", percent_encode_path_segment(id)))
        .await
}

#[cfg(test)]
mod page_status_command_type_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn serve_status_once(
        status: &'static str,
        response_body: &'static str,
    ) -> (crate::api::WenlanClient, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = vec![0_u8; 8192];
            let size = stream.read(&mut bytes).await.unwrap();
            let request = String::from_utf8_lossy(&bytes[..size]).to_string();
            let response = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body,
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });
        (
            crate::api::WenlanClient::with_base_url(format!("http://{address}")),
            handle,
        )
    }

    #[allow(dead_code)]
    async fn archive_page_response_uses_typed_status_envelope(client: crate::api::WenlanClient) {
        let _: Result<PageStatusResponse, String> = archive_page_response(&client, "page").await;
    }

    #[allow(dead_code)]
    async fn delete_page_response_uses_typed_status_envelope(client: crate::api::WenlanClient) {
        let _: Result<PageStatusResponse, String> = delete_page_response(&client, "page").await;
    }

    #[allow(dead_code)]
    async fn page_commands_keep_void_tauri_surface(state: tauri::State<'_, State>) {
        let _: Result<(), String> = archive_page(state.clone(), String::new()).await;
        let _: Result<(), String> = delete_page(state, String::new()).await;
    }

    #[test]
    fn page_status_response_deserializes_daemon_payloads() {
        let archived: PageStatusResponse = serde_json::from_value(serde_json::json!({
            "status": "archived"
        }))
        .unwrap();
        assert_eq!(archived.status, "archived");

        let deleted: PageStatusResponse = serde_json::from_value(serde_json::json!({
            "status": "deleted"
        }))
        .unwrap();
        assert_eq!(deleted.status, "deleted");
    }

    #[tokio::test]
    async fn delete_page_uses_the_permanent_delete_route() {
        let (client, request) = serve_status_once("200 OK", r#"{"status":"deleted"}"#).await;

        let response = delete_page_response(&client, "page-delete-me")
            .await
            .unwrap();
        let request = request.await.unwrap();

        assert_eq!(response.status, "deleted");
        assert!(request.starts_with("DELETE /api/pages/page-delete-me HTTP/1.1\r\n"));
    }

    #[tokio::test]
    async fn delete_page_forwards_daemon_failures() {
        let (client, request) =
            serve_status_once("503 Service Unavailable", r#"{"error":"offline"}"#).await;

        let error = delete_page_response(&client, "page-delete-me")
            .await
            .unwrap_err();
        let request = request.await.unwrap();

        assert!(request.starts_with("DELETE /api/pages/page-delete-me HTTP/1.1\r\n"));
        assert!(error.contains("HTTP DELETE /api/pages/page-delete-me returned 503"));
    }
}

#[cfg(test)]
mod search_response_type_tests {
    use super::*;

    fn search_result(source: &str, source_id: &str) -> SearchResult {
        SearchResult {
            id: format!("{source_id}-hit"),
            content: format!("{source} content"),
            source: source.to_string(),
            source_id: source_id.to_string(),
            title: format!("{source} hit"),
            url: None,
            chunk_index: 0,
            last_modified: 0,
            score: 0.9,
            chunk_type: None,
            language: None,
            semantic_unit: None,
            memory_type: if source == "memory" {
                Some("fact".to_string())
            } else {
                None
            },
            space: None,
            source_agent: None,
            confidence: None,
            confirmed: None,
            stability: None,
            supersedes: None,
            summary: None,
            entity_id: None,
            entity_name: None,
            quality: None,
            importance: None,
            event_date: None,
            is_archived: false,
            is_recap: false,
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            content_hash: None,
            raw_score: 0.0,
            version: 1,
            pending_revision: false,
            merged_from: None,
            last_delta_summary: None,
        }
    }

    #[test]
    fn search_results_include_supplemental_pages() {
        let resp = responses::SearchResponse {
            results: vec![],
            took_ms: 1.0,
            supplemental_pages: Some(vec![search_result("page", "page_1")]),
        };

        let results = search_results_from_response(resp);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "page");
        assert_eq!(results[0].source_id, "page_1");
    }
}

/// Build the `/api/pages` query string. Shared by `list_pages` and
/// `list_pages_explicit_browse`, which differ only in the client verb they
/// dispatch to.
fn pages_query_path(
    status: Option<String>,
    domain: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
) -> String {
    let mut params: Vec<String> = Vec::new();
    if let Some(s) = status {
        params.push(format!("status={}", percent_encode_path_segment(&s)));
    }
    if let Some(d) = domain {
        params.push(format!("domain={}", percent_encode_path_segment(&d)));
    }
    if let Some(l) = limit {
        params.push(format!("limit={}", l));
    }
    if let Some(o) = offset {
        params.push(format!("offset={}", o));
    }
    if params.is_empty() {
        "/api/pages".to_string()
    } else {
        format!("/api/pages?{}", params.join("&"))
    }
}

#[tauri::command]
pub async fn list_pages(
    state: tauri::State<'_, State>,
    status: Option<String>,
    domain: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
) -> Result<Vec<Page>, String> {
    let client = state.read().await.client.clone();
    let path = pages_query_path(status, domain, limit, offset);
    let resp: responses::SearchPagesResponse = client.get_json(&path).await?;
    Ok(resp.pages)
}

#[tauri::command]
pub async fn search_pages(
    state: tauri::State<'_, State>,
    query: String,
    limit: Option<usize>,
) -> Result<Vec<Page>, String> {
    let client = state.read().await.client.clone();
    let req = requests::SearchPagesRequest {
        query,
        limit,
        page_type: None,
        space: None,
    };
    let resp: responses::SearchPagesResponse = client.post_json("/api/pages/search", &req).await?;
    Ok(resp.pages)
}

// ── M5 truth axes: explicit-browse variants ─────────────────────────────
// Same requests as `list_pages`/`search_pages`/`get_page` above, but carrying
// the two-header marker the daemon's truth guard requires before it will
// attach `Page.truth` to a Collection- or NamedPage-shaped response
// (`crates/wenlan-core/src/truth_manifest.rs`). Call these only from a
// human-initiated wiki browse (the pages list, search results, or a page
// detail view a person navigated to) — never from a background poll or an
// agent-driven read, since the daemon durably records every marked call.

#[tauri::command]
pub async fn list_pages_explicit_browse(
    state: tauri::State<'_, State>,
    status: Option<String>,
    domain: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
) -> Result<Vec<Page>, String> {
    let client = state.read().await.client.clone();
    let path = pages_query_path(status, domain, limit, offset);
    let resp: responses::SearchPagesResponse = client.get_json_explicit_browse(&path).await?;
    Ok(resp.pages)
}

#[tauri::command]
pub async fn get_page_explicit_browse(
    state: tauri::State<'_, State>,
    id: String,
) -> Result<Option<serde_json::Value>, String> {
    let client = state.read().await.client.clone();
    match client
        .get_json_explicit_browse::<serde_json::Value>(&format!(
            "/api/pages/{}",
            percent_encode_path_segment(&id)
        ))
        .await
    {
        Ok(wire) => Ok(page_from_wire(wire)),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("404") || msg.to_lowercase().contains("not found") {
                Ok(None)
            } else {
                Err(format!("get_page_explicit_browse failed: {}", msg))
            }
        }
    }
}

#[tauri::command]
pub async fn get_truth_status(
    state: tauri::State<'_, State>,
) -> Result<Option<crate::api::TruthStatus>, String> {
    let client = state.read().await.client.clone();
    client.truth_status().await
}

#[tauri::command]
pub async fn get_page_sources(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<Vec<wenlan_types::PageSourceWithMemory>, String> {
    let client = { state.read().await.client.clone() };
    client.get_page_sources(&page_id).await
}

#[tauri::command]
pub async fn get_page_links(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<responses::PageLinksResponse, String> {
    let client = { state.read().await.client.clone() };
    client.get_page_links(&page_id).await
}

#[tauri::command]
pub async fn get_page_revisions(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<serde_json::Value, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_page_revisions(&page_id).await
}

#[tauri::command]
pub async fn list_orphan_links(
    state: tauri::State<'_, State>,
    min_count: Option<usize>,
) -> Result<responses::OrphanLinksResponse, String> {
    let client = { state.read().await.client.clone() };
    client.list_orphan_links(min_count).await
}

#[tauri::command]
pub async fn get_page_map(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<serde_json::Value, String> {
    let client = { state.read().await.client.clone() };
    client.get_page_map(&page_id).await
}

#[tauri::command]
pub async fn improve_page_map(
    state: tauri::State<'_, State>,
    page_id: String,
) -> Result<serde_json::Value, String> {
    let client = { state.read().await.client.clone() };
    client.improve_page_map(&page_id).await
}

#[tauri::command]
pub async fn create_page_map_node(
    state: tauri::State<'_, State>,
    page_id: String,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let client = { state.read().await.client.clone() };
    client.create_page_map_node(&page_id, body).await
}

#[tauri::command]
pub async fn patch_page_map_node(
    state: tauri::State<'_, State>,
    page_id: String,
    node_id: String,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let client = { state.read().await.client.clone() };
    client.patch_page_map_node(&page_id, &node_id, body).await
}

#[tauri::command]
pub async fn delete_page_map_node(
    state: tauri::State<'_, State>,
    page_id: String,
    node_id: String,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let client = { state.read().await.client.clone() };
    client.delete_page_map_node(&page_id, &node_id, body).await
}

#[tauri::command]
pub async fn put_page_map_layout(
    state: tauri::State<'_, State>,
    page_id: String,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let client = { state.read().await.client.clone() };
    client.put_page_map_layout(&page_id, body).await
}

// ── Communities (M6 cartography) ────────────────────────────────────────

#[tauri::command]
pub async fn list_communities(
    state: tauri::State<'_, State>,
    space: String,
    cursor: Option<String>,
    limit: Option<usize>,
) -> Result<crate::api::CommunityListResponse, String> {
    let client = { state.read().await.client.clone() };
    client
        .list_communities(&space, cursor.as_deref(), limit)
        .await
}

#[tauri::command]
pub async fn list_community_members(
    state: tauri::State<'_, State>,
    space: String,
    cursor: Option<crate::api::CommunityMemberCursor>,
    limit: Option<usize>,
) -> Result<crate::api::CommunityMembersResponse, String> {
    let client = { state.read().await.client.clone() };
    client
        .list_community_members(&space, cursor.as_ref(), limit)
        .await
}

#[tauri::command]
pub async fn export_pages_to_obsidian(
    state: tauri::State<'_, State>,
    vault_path: String,
) -> Result<ExportStats, String> {
    // Delegate bulk export to the daemon (POST /api/pages/export).
    // The daemon has direct FS access and owns the ObsidianExporter.
    let client = state.read().await.client.clone();
    let req = requests::ExportPagesRequest {
        vault_path: Some(vault_path),
    };
    client.post_json("/api/pages/export", &req).await
}

#[tauri::command]
pub async fn export_page_to_obsidian(
    state: tauri::State<'_, State>,
    page_id: String,
    vault_path: String,
) -> Result<responses::ExportPageResponse, String> {
    let client = state.read().await.client.clone();
    let path = format!(
        "/api/pages/{}/export",
        percent_encode_path_segment(&page_id)
    );
    let req = requests::ExportPageRequest { vault_path };
    client.post_json(&path, &req).await
}

#[cfg(test)]
mod export_command_type_tests {
    use super::*;

    #[allow(dead_code)]
    async fn export_page_to_obsidian_uses_daemon_export_response(state: tauri::State<'_, State>) {
        let _: Result<responses::ExportPageResponse, String> =
            export_page_to_obsidian(state, String::new(), String::new()).await;
    }

    #[test]
    fn export_page_to_obsidian_response_type_is_checked() {}
}

#[tauri::command]
pub async fn get_knowledge_path(state: tauri::State<'_, State>) -> Result<String, String> {
    let client = state.read().await.client.clone();
    let resp: responses::KnowledgePathResponse = client.get_json("/api/knowledge/path").await?;
    Ok(resp.path)
}

#[tauri::command]
pub async fn count_knowledge_files(state: tauri::State<'_, State>) -> Result<u64, String> {
    let client = state.read().await.client.clone();
    let resp: responses::KnowledgeCountResponse = client.get_json("/api/knowledge/count").await?;
    Ok(resp.count)
}

// ── Decision log ──────────────────────────────────────────────────────

#[tauri::command]
pub async fn list_decisions_cmd(
    state: tauri::State<'_, State>,
    domain: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<MemoryItem>, String> {
    let client = daemon_client(&state).await;
    let mut path = format!("/api/decisions?limit={}", limit.unwrap_or(200));
    if let Some(d) = domain {
        path.push_str(&format!("&domain={}", percent_encode_path_segment(&d)));
    }
    let resp: responses::DecisionsResponse = client.get_json(&path).await?;
    Ok(resp.decisions)
}

#[tauri::command]
pub async fn list_decision_domains_cmd(
    state: tauri::State<'_, State>,
) -> Result<Vec<String>, String> {
    let client = daemon_client(&state).await;
    let resp: responses::DecisionDomainsResponse =
        client.get_json("/api/decisions/domains").await?;
    Ok(resp.domains)
}

// ── Registered source management ──────────────────────────────────────

pub use crate::sources::sync::SyncStats;

/// The daemon dedupes sources by path; a repeat POST returns this string. The
/// app treats it as success (check-or-ignore), not an error path (§2).
fn already_registered(err: &str) -> bool {
    err.contains("Source already registered")
}

/// Register a directory (folder in place, or the managed uploads dir) with the
/// daemon, which owns ingestion (§1, §6). On repeat registration the daemon
/// returns "Source already registered" — resolve the existing source instead
/// of erroring.
async fn register_directory_source_with_daemon(
    state: &tauri::State<'_, State>,
    path: &std::path::Path,
) -> Result<crate::sources::Source, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    let path_str = path.to_string_lossy().to_string();
    match client.add_source("directory".to_string(), path_str).await {
        Ok(source) => Ok(source),
        Err(e) if already_registered(&e) => client
            .list_sources()
            .await?
            .into_iter()
            .find(|s| s.path == path)
            .ok_or_else(|| "source registered but not returned by daemon".to_string()),
        Err(e) => Err(e),
    }
}

#[tauri::command]
pub async fn add_source(
    state: tauri::State<'_, State>,
    _watcher: tauri::State<'_, WatcherState>,
    source_type: String,
    path: String,
) -> Result<crate::sources::Source, String> {
    let path_buf = PathBuf::from(&path);
    if !path_buf.exists() {
        return Err(format!("Path does not exist: {}", path));
    }
    if !path_buf.is_dir() {
        return Err(format!("Path is not a directory: {}", path));
    }

    match source_type.as_str() {
        "obsidian" => {
            // Accept any folder of markdown files, Obsidian vault or not.
            // Frontend detects .obsidian/ for cosmetic badge purposes.
            // `has_any_markdown` short-circuits on the first match instead
            // of walking the entire vault, so very large vaults don't
            // stall the Tauri executor at registration time.
            if !crate::sources::obsidian::has_any_markdown(&path_buf) {
                return Err(format!("No markdown files found in: {}", path));
            }
            let client = {
                let s = state.read().await;
                s.client.clone()
            };
            client.add_source("obsidian".to_string(), path).await
        }
        "directory" => register_directory_source_with_daemon(&state, &path_buf).await,
        other => Err(format!("Unknown source_type: {}", other)),
    }
}

#[cfg(test)]
mod already_registered_tests {
    #[test]
    fn already_registered_matches_daemon_dedupe_string() {
        assert!(super::already_registered("Source already registered"));
        assert!(super::already_registered(
            "ValidationError: Source already registered for path"
        ));
        assert!(!super::already_registered("Path does not exist"));
        assert!(!super::already_registered("connection refused"));
    }
}

/// Blobs to delete on removal. Only the app-managed uploads dir holds copies;
/// in-place folder sources are never copied, so nothing to clean (§4).
fn managed_blob_paths(
    sources_dir: &std::path::Path,
    source: &crate::sources::Source,
) -> Vec<std::path::PathBuf> {
    if source.path == sources_dir {
        vec![sources_dir.to_path_buf()]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod managed_blob_paths_tests {
    #[test]
    fn managed_blob_paths_targets_only_the_managed_dir() {
        let sources_dir = std::path::Path::new("/home/u/.wenlan/sources");
        let managed = crate::sources::Source {
            id: "directory-sources".into(),
            source_type: crate::sources::SourceType::Directory,
            path: sources_dir.to_path_buf(),
            status: crate::sources::SyncStatus::Active,
            last_sync: None,
            file_count: 0,
            memory_count: 0,
            last_sync_errors: 0,
            last_sync_error_detail: None,
        };
        // The managed dir itself is cleaned; an in-place folder source is not.
        assert_eq!(
            super::managed_blob_paths(sources_dir, &managed),
            vec![sources_dir.to_path_buf()]
        );

        let in_place = crate::sources::Source {
            path: "/home/u/Documents/Books".into(),
            ..managed.clone()
        };
        assert!(super::managed_blob_paths(sources_dir, &in_place).is_empty());
    }
}

#[tauri::command]
pub async fn remove_source(
    state: tauri::State<'_, State>,
    watcher: tauri::State<'_, WatcherState>,
    id: String,
) -> Result<(), String> {
    let local_source = config::load_config()
        .sources
        .iter()
        .find(|s| s.id == id)
        .cloned();

    if let Some(source) = local_source {
        if source.source_type == crate::sources::SourceType::Directory {
            return remove_directory_source(&state, &watcher, &id, source).await;
        }
    }

    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.remove_source(&id).await?;
    if id == "directory-sources" {
        let dir = crate::sources::uploads::sources_dir();
        let _ = std::fs::remove_dir_all(&dir); // managed uploads only, best-effort
    }
    Ok(())
}

async fn remove_directory_source(
    state: &tauri::State<'_, State>,
    watcher: &tauri::State<'_, WatcherState>,
    id: &str,
    source: crate::sources::Source,
) -> Result<(), String> {
    let mut cfg = config::load_config();
    if !cfg.sources.iter().any(|s| s.id == id) {
        return Err(format!("Source not found: {}", id));
    }
    cfg.sources.retain(|s| s.id != id);
    config::save_config(&cfg).map_err(|e| e.to_string())?;

    let sources_dir = crate::sources::uploads::sources_dir();
    for blob in managed_blob_paths(&sources_dir, &source) {
        let _ = std::fs::remove_dir_all(&blob); // best-effort; missing dir is fine
    }

    {
        let mut app_state = state.write().await;
        if let Some(local) = app_state.sources.get_mut("local_files") {
            if let Some(local) = local
                .as_any_mut()
                .downcast_mut::<crate::sources::local_files::LocalFilesSource>()
            {
                local.remove_watch_path(&source.path);
            }
        }
        app_state.watch_paths.retain(|p| p != &source.path);
    }

    // Pruning `watch_paths` does not stop ingestion — the debouncer callback
    // never consults it. The live watcher must be told, or files under a
    // disconnected folder keep flowing into the daemon until the app restarts.
    let mut watcher_guard = watcher.lock().await;
    if let Some(w) = watcher_guard.as_mut() {
        crate::indexer::unwatch_path(w, &source.path);
    }
    Ok(())
}

#[tauri::command]
pub async fn list_registered_sources(
    state: tauri::State<'_, State>,
) -> Result<Vec<crate::sources::Source>, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    let daemon_sources = client.list_sources().await?;
    let local_sources = config::load_config().sources;
    Ok(merge_registered_sources_with_local_directories(
        daemon_sources,
        local_sources,
    ))
}

fn merge_registered_sources_with_local_directories(
    mut daemon_sources: Vec<crate::sources::Source>,
    local_sources: Vec<crate::sources::Source>,
) -> Vec<crate::sources::Source> {
    for source in local_sources
        .into_iter()
        .filter(|s| s.source_type == crate::sources::SourceType::Directory)
    {
        if daemon_sources
            .iter()
            .any(|existing| existing.id == source.id || existing.path == source.path)
        {
            continue;
        }
        daemon_sources.push(source);
    }
    daemon_sources
}

#[tauri::command]
pub async fn sync_registered_source(
    state: tauri::State<'_, State>,
    id: String,
) -> Result<SyncStats, String> {
    let local_source = config::load_config()
        .sources
        .iter()
        .find(|s| s.id == id)
        .cloned();

    if matches!(
        local_source.as_ref().map(|s| &s.source_type),
        Some(crate::sources::SourceType::Directory)
    ) {
        return Err("Only Obsidian sources support manual sync; directory sources use the live file watcher".to_string());
    }

    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    let stats = client.sync_source(&id).await?;
    Ok(SyncStats {
        files_found: stats.files_found,
        ingested: stats.ingested,
        skipped: stats.skipped,
        errors: stats.errors,
        error_detail: None,
    })
}

#[tauri::command]
pub async fn daemon_version(state: tauri::State<'_, State>) -> Result<String, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    Ok(client.health().await?.version)
}

/// Stage a loose file into the managed uploads dir, then ensure that dir is
/// registered with the daemon as a `directory` source (§2, §6).
#[tauri::command]
pub async fn upload_source_file(
    state: tauri::State<'_, State>,
    path: String,
) -> Result<crate::sources::Source, String> {
    let src = std::path::PathBuf::from(&path);
    if !src.is_file() {
        return Err(format!("Not a file: {}", path));
    }
    let dir = crate::sources::uploads::sources_dir();
    crate::sources::uploads::place_upload_file(&dir, &src).map_err(|e| e.to_string())?;
    register_directory_source_with_daemon(&state, &dir).await
}

// ---------------------------------------------------------------------------
// External LLM provider commands (Ollama, LM Studio, etc.)
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_model_choice(
    state: tauri::State<'_, State>,
) -> Result<(Option<String>, Option<String>), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_model_choice().await
}

#[tauri::command]
pub async fn set_model_choice(
    state: tauri::State<'_, State>,
    routine_model: Option<String>,
    synthesis_model: Option<String>,
) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client
        .set_model_choice(routine_model, synthesis_model)
        .await?;
    log::info!("[settings] Model choice updated — restart daemon to apply");
    Ok(())
}

/// Proxy for `GET /api/config/routing` (daemon ≥ PR #357). `None` means the
/// daemon predates the endpoint (404) — the caller renders LEGACY mode.
#[tauri::command]
pub async fn get_resolved_routing(
    state: tauri::State<'_, State>,
) -> Result<Option<crate::api::ResolvedRouting>, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_resolved_routing().await
}

/// Patch the per-job source pins (daemon ≥ PR #357). Each arg: `null` leaves the
/// pin untouched, `""` clears it, a source name pins. Only call once
/// `get_resolved_routing` returned `Some` — never at an old daemon.
#[tauri::command]
pub async fn set_source_pin(
    state: tauri::State<'_, State>,
    everyday_source: Option<String>,
    synthesis_source: Option<String>,
) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client
        .set_source_pin(everyday_source, synthesis_source)
        .await?;
    log::info!("[settings] Source pin updated — restart daemon to apply");
    Ok(())
}

#[tauri::command]
pub async fn get_system_info() -> Result<wenlan_types::system_info::SystemInfo, String> {
    Ok(crate::system_info::detect_system_info())
}

#[tauri::command]
pub async fn get_external_llm(
    state: tauri::State<'_, State>,
) -> Result<(Option<String>, Option<String>), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_external_llm().await
}

#[tauri::command]
pub async fn set_external_llm(
    state: tauri::State<'_, State>,
    endpoint: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.set_external_llm(endpoint, model, api_key).await?;
    log::info!("[settings] External LLM config updated");
    Ok(())
}

#[tauri::command]
pub async fn test_external_llm(
    state: tauri::State<'_, State>,
    endpoint: String,
    model: String,
    api_key: Option<String>,
) -> Result<requests::TestLlmResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.test_llm(endpoint, model, api_key).await
}

#[tauri::command]
pub async fn get_external_llm_key_configured(
    state: tauri::State<'_, State>,
) -> Result<bool, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.external_llm_key_configured().await
}

/// Parse an OpenAI-compatible `GET {endpoint}/models` body into model IDs.
pub(crate) fn parse_models_response(body: &serde_json::Value) -> Vec<String> {
    body.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Model auto-discovery for the Any-provider card (spec §1, §6). Talks to the
/// provider directly (not the daemon) so discovery works before saving.
#[tauri::command]
pub async fn list_external_models(
    endpoint: String,
    api_key: Option<String>,
) -> Result<Vec<String>, String> {
    let base = endpoint.trim_end_matches('/');
    if !(base.starts_with("http://") || base.starts_with("https://")) {
        return Err("Endpoint must start with http:// or https://".to_string());
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut req = client.get(format!("{base}/models"));
    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("{} from {base}/models", resp.status()));
    }
    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(parse_models_response(&body))
}

#[cfg(test)]
mod external_llm_command_type_tests {
    use super::*;

    #[allow(dead_code)]
    async fn test_external_llm_uses_daemon_response_envelope(state: tauri::State<'_, State>) {
        let _: Result<requests::TestLlmResponse, String> =
            test_external_llm(state, String::new(), String::new(), None).await;
    }

    #[test]
    fn test_external_llm_response_type_is_checked() {}

    #[allow(dead_code)]
    async fn get_external_llm_key_configured_uses_typed_response(state: tauri::State<'_, State>) {
        let _: Result<bool, String> = get_external_llm_key_configured(state).await;
    }
}

/// Proxy for `GET /api/on-device-model` — returns per-model cache/load state.
#[tauri::command]
pub async fn get_on_device_model(
    state: tauri::State<'_, State>,
) -> Result<crate::api::OnDeviceModelResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.get_on_device_model().await
}

/// Proxy for `POST /api/on-device-model/download` — triggers download + hot-load.
/// This is a long-running call (minutes for a 2.7GB download).
#[tauri::command]
pub async fn download_on_device_model(
    state: tauri::State<'_, State>,
    model_id: String,
) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.download_on_device_model(model_id).await
}

#[cfg(test)]
mod on_device_model_command_type_tests {
    use super::*;

    #[allow(dead_code)]
    async fn get_on_device_model_uses_typed_response(state: tauri::State<'_, State>) {
        let _: Result<crate::api::OnDeviceModelResponse, String> = get_on_device_model(state).await;
    }

    #[test]
    fn on_device_model_command_response_type_is_checked() {}
}

/// Bytes downloaded so far for an in-flight on-device model download.
///
/// The daemon's `/api/on-device-model/download` is one blocking HTTP call
/// that reports nothing until it finishes, so there is no progress endpoint
/// to poll. But the daemon downloads via hf-hub's sync API, which streams
/// each blob into `<blob-etag>.part` with `OpenOptions::append(true)` and no
/// preallocation, renaming it to `<blob-etag>` only on completion. That
/// means the `.part` file's size on disk is the true number of bytes
/// downloaded so far, even though we don't know the file's final size.
///
/// This walks the whole hub cache rather than resolving the exact repo id
/// for the model being downloaded: `OnDeviceModelEntry` carries no repo_id,
/// and hardcoding the daemon's model registry here would duplicate it. This
/// is safe because exactly one on-device model download is ever in flight
/// during the setup wizard.
fn largest_part_file(hub_dir: &Path) -> Option<u64> {
    let mut largest: Option<u64> = None;
    for model_dir in std::fs::read_dir(hub_dir).ok()?.flatten() {
        let blobs_dir = model_dir.path().join("blobs");
        let Ok(blob_entries) = std::fs::read_dir(&blobs_dir) else {
            continue;
        };
        for blob_entry in blob_entries.flatten() {
            let path = blob_entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("part") {
                continue;
            }
            let Ok(metadata) = blob_entry.metadata() else {
                continue;
            };
            let size = metadata.len();
            largest = Some(largest.map_or(size, |current: u64| current.max(size)));
        }
    }
    largest
}

/// Returns bytes downloaded so far for an in-flight on-device model
/// download, or `None` if no download is in progress (or the hf-hub cache
/// layout has changed). See [`largest_part_file`] for why this is honest
/// about the numerator (bytes so far) but says nothing about a total.
#[tauri::command]
pub fn on_device_model_download_bytes() -> Option<u64> {
    let hub_dir = dirs::home_dir()?.join(".cache/huggingface/hub");
    largest_part_file(&hub_dir)
}

#[cfg(test)]
mod on_device_model_download_bytes_tests {
    use super::*;

    #[test]
    fn returns_none_for_empty_hub_dir() {
        let hub = tempfile::tempdir().unwrap();
        assert_eq!(largest_part_file(hub.path()), None);
    }

    #[test]
    fn returns_none_when_no_part_files_exist() {
        let hub = tempfile::tempdir().unwrap();
        let blobs = hub.path().join("models--org--model").join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("completed-etag"), vec![0u8; 999_999]).unwrap();

        assert_eq!(largest_part_file(hub.path()), None);
    }

    #[test]
    fn returns_size_of_largest_part_file_across_models() {
        let hub = tempfile::tempdir().unwrap();
        let blobs_a = hub.path().join("models--org--model-a").join("blobs");
        let blobs_b = hub.path().join("models--org--model-b").join("blobs");
        std::fs::create_dir_all(&blobs_a).unwrap();
        std::fs::create_dir_all(&blobs_b).unwrap();
        std::fs::write(blobs_a.join("abc123.part"), vec![0u8; 100]).unwrap();
        std::fs::write(blobs_b.join("def456.part"), vec![0u8; 500]).unwrap();
        // A completed blob (no `.part` suffix) must never be counted.
        std::fs::write(blobs_b.join("completed-etag"), vec![0u8; 999_999]).unwrap();

        assert_eq!(largest_part_file(hub.path()), Some(500));
    }

    #[test]
    fn on_device_model_download_bytes_returns_option_u64() {
        let _: Option<u64> = on_device_model_download_bytes();
    }
}

// ── Home delta feed ─────────────────────────────────────────────────────

#[tauri::command]
pub async fn list_recent_retrievals(
    state: tauri::State<'_, State>,
    limit: Option<i64>,
) -> Result<Vec<wenlan_types::RetrievalEvent>, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.list_recent_retrievals(limit.unwrap_or(10)).await
}

#[tauri::command]
pub async fn list_recent_changes(
    state: tauri::State<'_, State>,
    limit: Option<i64>,
) -> Result<Vec<wenlan_types::PageChange>, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.list_recent_changes(limit.unwrap_or(10)).await
}

#[tauri::command]
pub async fn list_recent_memories(
    state: tauri::State<'_, State>,
    limit: Option<i64>,
    since_ms: Option<i64>,
) -> Result<Vec<wenlan_types::RecentActivityItem>, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client
        .list_recent_memories(limit.unwrap_or(10), since_ms)
        .await
}

#[tauri::command]
pub async fn list_unconfirmed_memories(
    state: tauri::State<'_, State>,
    limit: Option<i64>,
) -> Result<Vec<wenlan_types::RecentActivityItem>, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.list_unconfirmed_memories(limit.unwrap_or(6)).await
}

#[tauri::command]
pub async fn list_recent_pages(
    state: tauri::State<'_, State>,
    limit: Option<i64>,
    since_ms: Option<i64>,
) -> Result<Vec<wenlan_types::RecentActivityItem>, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client
        .list_recent_pages(limit.unwrap_or(10), since_ms)
        .await
}

// ── Lifecycle commands ─────────────────────────────────────────────────────

#[tauri::command]
pub async fn is_run_at_login_enabled() -> Result<bool, String> {
    if crate::lifecycle::run_at_login_capability(std::env::consts::OS).is_err() {
        return Ok(false);
    }
    use crate::lifecycle::{is_run_at_login_enabled as inner, SystemLaunchctl};
    Ok(inner(&SystemLaunchctl))
}

#[tauri::command]
pub async fn set_run_at_login(app: tauri::AppHandle, enabled: bool) -> Result<(), String> {
    crate::lifecycle::run_at_login_capability(std::env::consts::OS).map_err(str::to_string)?;
    use crate::lifecycle::{set_run_at_login as inner, SystemLaunchctl};
    let result = inner(enabled, &SystemLaunchctl).await;
    if enabled && result.is_err() {
        // Turning the toggle on stops an app-owned sidecar before the launchd
        // handover; a handover that then fails must not leave the user with
        // no daemon at all, so start one again unless something owns it.
        let outcome =
            crate::daemon_start::start_daemon_if_unowned(&app, &crate::api::WenlanClient::new())
                .await;
        log::warn!("[run-at-login] launchd handover failed; daemon fallback: {outcome:?}");
    }
    result.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn quit_wenlan_full(app_handle: tauri::AppHandle) -> Result<(), String> {
    crate::lifecycle::quit_origin(&app_handle)
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod status_response_tests {
    use super::*;

    #[test]
    fn daemon_status_updates_count_and_reranker_without_forcing_indexing_state() {
        let local = IndexStatus {
            is_running: false,
            files_indexed: 3,
            files_total: 7,
            last_error: Some("local watcher error".to_string()),
            sources_connected: vec!["local".to_string()],
            reranker: wenlan_types::responses::RerankerStatus::Disabled,
            reranker_light: wenlan_types::responses::RerankerStatus::Disabled,
            reranker_mode: "off".to_string(),
        };
        let daemon = responses::StatusResponse {
            is_running: true,
            files_indexed: 42,
            files_total: 0,
            sources_connected: vec!["daemon".to_string()],
            queue: Default::default(),
            compile_queue: Default::default(),
            reranker: wenlan_types::responses::RerankerStatus::Failed {
                reason: "model missing".to_string(),
            },
            reranker_light: wenlan_types::responses::RerankerStatus::Active {
                model_id: "bge-reranker".to_string(),
            },
            reranker_mode: "lite".to_string(),
            on_device_inference: Default::default(),
            capabilities: Vec::new(),
            truth: None,
        };

        let merged = merge_daemon_status(local, daemon);

        assert!(!merged.is_running);
        assert_eq!(merged.files_indexed, 42);
        assert_eq!(merged.files_total, 7);
        assert_eq!(merged.last_error.as_deref(), Some("local watcher error"));
        assert_eq!(merged.sources_connected, vec!["daemon".to_string()]);
        assert_eq!(merged.reranker_mode, "lite");
        assert_eq!(
            merged.reranker,
            wenlan_types::responses::RerankerStatus::Failed {
                reason: "model missing".to_string()
            }
        );
        assert_eq!(
            merged.reranker_light,
            wenlan_types::responses::RerankerStatus::Active {
                model_id: "bge-reranker".to_string()
            }
        );
    }
}

#[cfg(test)]
mod tag_data_tests {
    use super::*;

    #[test]
    fn set_document_tags_request_serializes_source_and_tags() {
        let request = SetDocumentTagsRequest {
            source: "manual".to_string(),
            tags: vec!["rust".to_string()],
        };

        let value = serde_json::to_value(&request).unwrap();

        assert_eq!(value["source"], "manual");
        assert_eq!(value["tags"], serde_json::json!(["rust"]));
    }

    #[test]
    fn tag_data_from_inventory_preserves_document_tags() {
        let mut document_tags = HashMap::new();
        document_tags.insert("memory::mem1".to_string(), vec!["rust".to_string()]);

        let tag_data = tag_data_from_inventory(crate::api::TagInventoryResponse {
            tags: vec!["rust".to_string()],
            document_tags,
        });

        assert_eq!(tag_data.tags, vec!["rust"]);
        assert_eq!(
            tag_data.document_tags.get("memory::mem1"),
            Some(&vec!["rust".to_string()])
        );
        assert!(tag_data.categories.is_empty());
        assert!(tag_data.document_categories.is_empty());
    }
}

#[cfg(test)]
mod avatar_path_tests {
    use super::*;
    use crate::test_env::EnvGuard;

    const AVATAR_ENV_KEYS: &[&str] = &["WENLAN_DATA_DIR", "ORIGIN_DATA_DIR"];

    #[test]
    #[serial_test::serial]
    fn avatar_storage_dir_prefers_wenlan_data_dir() {
        let _env = EnvGuard::capture(AVATAR_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();

        std::env::set_var("WENLAN_DATA_DIR", tmp.path());
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/legacy-origin-avatar-root");

        assert_eq!(avatar_storage_dir(), tmp.path().join("avatars"));
    }

    #[test]
    #[serial_test::serial]
    fn avatar_storage_dir_prefers_wenlan_data_dir_when_both_are_set() {
        let _env = EnvGuard::capture(AVATAR_ENV_KEYS);
        let current = tempfile::tempdir().unwrap();
        let legacy = tempfile::tempdir().unwrap();

        std::env::set_var("WENLAN_DATA_DIR", current.path());
        std::env::set_var("ORIGIN_DATA_DIR", legacy.path());

        assert_eq!(avatar_storage_dir(), current.path().join("avatars"));
    }

    #[test]
    #[serial_test::serial]
    fn avatar_storage_dir_falls_back_to_legacy_origin_data_dir() {
        let _env = EnvGuard::capture(AVATAR_ENV_KEYS);
        let tmp = tempfile::tempdir().unwrap();

        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", tmp.path());

        assert_eq!(avatar_storage_dir(), tmp.path().join("avatars"));
    }

    #[test]
    #[serial_test::serial]
    fn resolves_missing_legacy_avatar_to_wenlan_copy() {
        let _env = EnvGuard::capture(AVATAR_ENV_KEYS);
        let current = tempfile::tempdir().unwrap();
        let legacy = tempfile::tempdir().unwrap();
        let filename = "57515813-4419-4116-bea6-21bc66e1a511.jpg";

        std::env::set_var("WENLAN_DATA_DIR", current.path());
        std::env::set_var("ORIGIN_DATA_DIR", legacy.path());
        std::fs::create_dir_all(current.path().join("avatars")).unwrap();
        std::fs::write(current.path().join("avatars").join(filename), b"avatar").unwrap();

        let legacy_path = legacy.path().join("avatars").join(filename);

        assert_eq!(
            resolve_profile_avatar_path(Some(legacy_path.to_string_lossy().to_string())),
            Some(
                current
                    .path()
                    .join("avatars")
                    .join(filename)
                    .to_string_lossy()
                    .to_string()
            )
        );
    }

    #[test]
    #[serial_test::serial]
    fn does_not_resolve_arbitrary_missing_path_to_avatar_copy() {
        let _env = EnvGuard::capture(AVATAR_ENV_KEYS);
        let current = tempfile::tempdir().unwrap();
        let filename = "same-name.jpg";

        std::env::set_var("WENLAN_DATA_DIR", current.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        std::fs::create_dir_all(current.path().join("avatars")).unwrap();
        std::fs::write(current.path().join("avatars").join(filename), b"avatar").unwrap();

        let arbitrary_path = current.path().join("downloads").join(filename);

        assert_eq!(
            resolve_profile_avatar_path(Some(arbitrary_path.to_string_lossy().to_string())),
            None
        );
    }

    #[test]
    #[serial_test::serial]
    fn does_not_resolve_non_origin_avatar_dir_to_wenlan_copy() {
        let _env = EnvGuard::capture(AVATAR_ENV_KEYS);
        let current = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let filename = "same-name.jpg";

        std::env::set_var("WENLAN_DATA_DIR", current.path());
        std::env::remove_var("ORIGIN_DATA_DIR");
        std::fs::create_dir_all(current.path().join("avatars")).unwrap();
        std::fs::write(current.path().join("avatars").join(filename), b"avatar").unwrap();

        let non_origin_avatar_path = other
            .path()
            .join("not-origin")
            .join("avatars")
            .join(filename);

        assert_eq!(
            resolve_profile_avatar_path(Some(non_origin_avatar_path.to_string_lossy().to_string())),
            None
        );
    }
}

#[cfg(test)]
mod registered_source_tests {
    use super::*;

    fn source(
        id: &str,
        source_type: crate::sources::SourceType,
        path: &str,
    ) -> crate::sources::Source {
        crate::sources::Source {
            id: id.to_string(),
            source_type,
            path: PathBuf::from(path),
            status: crate::sources::SyncStatus::Active,
            last_sync: None,
            file_count: 0,
            memory_count: 0,
            last_sync_errors: 0,
            last_sync_error_detail: None,
        }
    }

    #[test]
    fn registered_source_listing_keeps_local_directory_sources_only() {
        let daemon_sources = vec![source(
            "obsidian-daemon",
            crate::sources::SourceType::Obsidian,
            "/Users/test/vault",
        )];
        let local_sources = vec![
            source(
                "directory-local",
                crate::sources::SourceType::Directory,
                "/Users/test/docs",
            ),
            source(
                "obsidian-stale-local",
                crate::sources::SourceType::Obsidian,
                "/Users/test/old-vault",
            ),
        ];

        let merged = merge_registered_sources_with_local_directories(daemon_sources, local_sources);

        assert_eq!(merged.len(), 2);
        assert!(merged.iter().any(|s| s.id == "obsidian-daemon"));
        assert!(merged.iter().any(|s| s.id == "directory-local"));
        assert!(!merged.iter().any(|s| s.id == "obsidian-stale-local"));
    }
}

#[cfg(test)]
mod list_external_models_tests {
    use super::*;

    #[test]
    fn parses_openai_models_shape() {
        let body = serde_json::json!({
            "object": "list",
            "data": [
                {"id": "llama3.2:3b", "object": "model"},
                {"id": "qwen2.5-coder", "object": "model"}
            ]
        });
        assert_eq!(
            parse_models_response(&body),
            vec!["llama3.2:3b".to_string(), "qwen2.5-coder".to_string()]
        );
    }

    #[test]
    fn missing_or_malformed_data_yields_empty() {
        assert!(parse_models_response(&serde_json::json!({})).is_empty());
        assert!(parse_models_response(&serde_json::json!({"data": "nope"})).is_empty());
        assert!(
            parse_models_response(&serde_json::json!({"data": [{"name": "no-id"}]})).is_empty()
        );
    }
}
