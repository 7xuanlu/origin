// SPDX-License-Identifier: AGPL-3.0-only
#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

#[cfg(feature = "review-fixtures")]
mod review;
#[cfg(feature = "review-fixtures")]
pub use review::run_review;

// ── App-specific modules (Tauri, sensors, UI) ──
pub mod activity;
pub mod api;
pub mod config;
mod daemon_start;
pub mod error;
pub mod events;
mod identity_paths;
mod indexer;
mod lifecycle;
pub mod mcp_config;
pub mod plugin_install;
pub mod privacy;
pub mod remote_access;
mod search;
pub mod sources;
pub mod state;
pub mod system_info;
// Public surface consumed by tray_menu (Task 15); suppress dead_code until then.
#[allow(dead_code)]
pub(crate) mod tray_health;
mod updater;
pub mod wire_state;

use state::AppState;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(target_os = "macos")]
fn activation_policy_for_main_window_visible(_visible: bool) -> tauri::ActivationPolicy {
    tauri::ActivationPolicy::Regular
}

#[cfg(target_os = "macos")]
fn set_main_window_dock_visibility<R: tauri::Runtime>(app: &tauri::AppHandle<R>, visible: bool) {
    // Dock tile comes from the app bundle via IconServices, which applies the
    // standard macOS rounded-rect (squircle) mask. Do NOT re-assert it with
    // setApplicationIconImage_ — a raw bitmap bypasses that mask and renders a
    // square tile while running (round only when parked).
    let _ = app.set_activation_policy(activation_policy_for_main_window_visible(visible));
}

#[cfg(not(target_os = "macos"))]
fn set_main_window_dock_visibility<R: tauri::Runtime>(_app: &tauri::AppHandle<R>, _visible: bool) {}

fn app_log_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .map(|h| h.join("Library/Logs/com.wenlan.desktop"))
        .unwrap_or_else(std::env::temp_dir)
}

fn app_log_file_name() -> &'static str {
    "wenlan.log"
}

static QUIT_GUARD_PENDING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuardedQuitAction {
    RequestFrontendGuard,
    ForceShutdown,
}

fn guarded_quit_action(pending: &std::sync::atomic::AtomicBool) -> GuardedQuitAction {
    if pending.swap(true, std::sync::atomic::Ordering::AcqRel) {
        GuardedQuitAction::ForceShutdown
    } else {
        GuardedQuitAction::RequestFrontendGuard
    }
}

fn cancel_guarded_quit(pending: &std::sync::atomic::AtomicBool) {
    pending.store(false, std::sync::atomic::Ordering::Release);
}

#[cfg(not(feature = "review-fixtures"))]
#[tauri::command]
fn cancel_guarded_quit_request() {
    cancel_guarded_quit(&QUIT_GUARD_PENDING);
}

#[cfg(not(feature = "review-fixtures"))]
fn force_full_quit(app: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        if let Err(e) = crate::lifecycle::quit_origin(&app).await {
            log::error!("[app] forced quit failed: {e}");
            app.exit(1);
        }
    });
}

#[cfg(not(feature = "review-fixtures"))]
fn request_full_quit(app: &tauri::AppHandle) -> Result<(), tauri::Error> {
    use tauri::Emitter;
    match guarded_quit_action(&QUIT_GUARD_PENDING) {
        GuardedQuitAction::RequestFrontendGuard => {
            if let Err(error) = app.emit("quit-requested", ()) {
                cancel_guarded_quit(&QUIT_GUARD_PENDING);
                return Err(error);
            }
        }
        GuardedQuitAction::ForceShutdown => force_full_quit(app.clone()),
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn startup_reveal_fallback_delay() -> std::time::Duration {
    std::time::Duration::from_millis(1200)
}

#[cfg(target_os = "macos")]
fn startup_reveal_fallback_needed(ready: bool, visible: bool) -> bool {
    !ready || !visible
}

#[cfg(not(feature = "review-fixtures"))]
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Log sinks: stderr (for terminal launches, `pnpm tauri dev`) AND a
    // file at ~/Library/Logs/com.wenlan.desktop/wenlan.log.
    // GUI launches send stderr to /dev/null, so without the file sink any
    // setup() error — e.g. a sidecar spawn ENOENT — is silent. That is
    // exactly how the origin-server spawn regression hid for ~15 minutes
    // of live debugging before the culprit was found. Keep both sinks.
    use tracing_subscriber::prelude::*;

    let log_dir = app_log_dir();
    let _ = std::fs::create_dir_all(&log_dir);
    let file_appender = tracing_appender::rolling::never(&log_dir, app_log_file_name());
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    // The guard flushes the background worker on drop. The app lives for
    // the full process, so leaking it is correct — we never want the
    // writer to stop flushing before exit.
    std::mem::forget(guard);

    let env_filter = || {
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            // Was "warn" plus info-level targets for wenlan_lib::{trigger,router,sensor} —
            // all three modules are gone, so those directives named nothing and the
            // filter was already just "warn" in effect.
            tracing_subscriber::EnvFilter::new("warn")
        })
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_ansi(true)
                .with_writer(std::io::stderr)
                .with_filter(env_filter()),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_ansi(false)
                .with_writer(file_writer)
                .with_filter(env_filter()),
        )
        .init();

    tracing::info!(
        log_file = ?log_dir.join(app_log_file_name()),
        "wenlan app starting; logs tee'd to file"
    );

    let app_state = AppState::new();

    let builder =
        tauri::Builder::default().plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            use tauri::Manager;
            if let Some(window) = app.get_webview_window("main") {
                set_main_window_dock_visibility(app, true);
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }));

    #[cfg(debug_assertions)]
    let builder = builder.plugin(tauri_plugin_mcp::init_with_config(
        tauri_plugin_mcp::PluginConfig::new("wenlan".to_string())
            .start_socket_server(true)
            .socket_path("/tmp/tauri-mcp.sock".into()),
    ));

    builder
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_clipboard_x::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .manage(Arc::new(RwLock::new(app_state)))
        .manage(Arc::new(tokio::sync::Mutex::new(
            None::<indexer::FileWatcher>,
        )))
        .setup(|app| {
            let handle = app.handle().clone();

            // Keep the app LaunchServices-friendly: the UI process is a normal
            // Dock app from startup, while close/hide only affects the window.
            #[cfg(target_os = "macos")]
            {
                app.set_activation_policy(activation_policy_for_main_window_visible(false));
            }

            // Tray-app pattern: red-X on the main window hides instead of closing.
            // Without this handler the default Tauri close-button behavior destroys
            // the window, after which the tray "Show" menu's get_webview_window
            // returns None and silently no-ops — leaving a tray icon with no way
            // to bring the window back. prevent_close + hide() keeps the window
            // alive (cheap — it's just hidden), so subsequent show()+set_focus()
            // calls from the tray work.
            {
                use tauri::{Manager, WindowEvent};
                if let Some(main_window) = app.get_webview_window("main") {
                    let win = main_window.clone();
                    let app_for_close = handle.clone();
                    main_window.on_window_event(move |event| {
                        if let WindowEvent::CloseRequested { api, .. } = event {
                            api.prevent_close();
                            let _ = win.hide();
                            set_main_window_dock_visibility(&app_for_close, false);
                        }
                    });
                }
            }

            // Repair a stale server plist before daemon selection. The full
            // first-run install can stay async, but an already-running daemon
            // with the wrong data root must not win the port before repair.
            let daemon_startup_preflight_ok = {
                use tauri::Emitter;
                let launchctl = crate::lifecycle::SystemLaunchctl;
                match crate::lifecycle::prepare_server_plist_for_startup(&launchctl) {
                    Ok(()) => true,
                    Err(e) => {
                        log::warn!("[startup] server plist data-dir preflight failed: {e}");
                        let _ = handle.emit("origin-fallback-mode", ());
                        false
                    }
                }
            };
            // Carry the outcome to the on-demand "Start Wenlan" command, which
            // must not re-run the mutating preflight from a user click.
            crate::daemon_start::set_startup_preflight_ok(daemon_startup_preflight_ok);

            // First-run silent install — H6: run on a blocking task so we
            // don't block setup() (which delays Tauri start by hundreds of ms).
            {
                use tauri::Emitter;
                let install_handle = handle.clone();
                tauri::async_runtime::spawn(async move {
                    let result = tauri::async_runtime::spawn_blocking(|| {
                        let launchctl = crate::lifecycle::SystemLaunchctl;
                        crate::lifecycle::first_run_install_if_needed(&launchctl)
                    })
                    .await;
                    match result {
                        Ok(Ok(())) => {
                            log::info!("[first-run] plist install ok or unnecessary");
                        }
                        Ok(Err(e)) => {
                            log::warn!(
                                "[first-run] plist install failed, fallback mode: {e}"
                            );
                            let _ = install_handle.emit("origin-fallback-mode", ());
                        }
                        Err(e) => {
                            log::warn!("[first-run] install task join error: {e}");
                            let _ = install_handle.emit("origin-fallback-mode", ());
                        }
                    }
                });
            }

            // Configure macOS window: rounded corners, hide traffic lights, set bg color
            #[cfg(target_os = "macos")]
            #[allow(deprecated, unexpected_cfgs)]
            {
                use cocoa::appkit::{NSColor, NSWindow};
                use cocoa::base::{id, nil};
                use raw_window_handle::HasWindowHandle;
                use tauri::Manager;

                if let Some(win) = app.get_webview_window("main") {
                    // Size first because AppKit can recalculate titlebar control
                    // frames while the window geometry changes.
                    let _ = win.set_size(tauri::Size::Logical(tauri::LogicalSize::new(
                        1100.0, 720.0,
                    )));
                    let _ = win.center();

                    if let Ok(raw_handle) = win.window_handle() {
                        if let raw_window_handle::RawWindowHandle::AppKit(appkit) =
                            raw_handle.as_raw()
                        {
                            let ns_view = appkit.ns_view.as_ptr() as id;
                            unsafe {
                                let ns_win: id = objc::msg_send![ns_view, window];
                                let bg = NSColor::colorWithRed_green_blue_alpha_(
                                    nil,
                                    22.0 / 255.0,
                                    33.0 / 255.0,
                                    62.0 / 255.0,
                                    1.0,
                                );
                                ns_win.setBackgroundColor_(bg);
                            }
                        }
                    }
                    // Size the window and keep app-ready as a focus/activation
                    // refinement. The main window is visible from config so launch
                    // cannot depend on a frontend event to appear.
                    {
                        use tauri::Listener;
                        let app_ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
                        let win_for_ready = win.clone();
                        let app_for_ready = handle.clone();
                        let app_ready_for_event = app_ready.clone();
                        // Listen for the frontend "app-ready" event
                        handle.listen("app-ready", move |_| {
                            app_ready_for_event
                                .store(true, std::sync::atomic::Ordering::SeqCst);
                            set_main_window_dock_visibility(&app_for_ready, true);
                            let _ = win_for_ready.show();
                            let _ = win_for_ready.unminimize();
                            let _ = win_for_ready.set_focus();
                        });

                        let win_for_fallback = win.clone();
                        let app_for_fallback = handle.clone();
                        tauri::async_runtime::spawn(async move {
                            tokio::time::sleep(startup_reveal_fallback_delay()).await;
                            let ready = app_ready.load(std::sync::atomic::Ordering::SeqCst);
                            let visible = win_for_fallback.is_visible().unwrap_or(false);
                            if startup_reveal_fallback_needed(ready, visible) {
                                log::warn!(
                                    "[startup] app-ready did not reveal the main window; showing fallback"
                                );
                                set_main_window_dock_visibility(&app_for_fallback, true);
                                let _ = win_for_fallback.show();
                                let _ = win_for_fallback.unminimize();
                                let _ = win_for_fallback.set_focus();
                            }
                        });
                    }
                }
            }

            // Create transparent toast overlay window
            {
                use tauri::{WebviewUrl, WebviewWindowBuilder};

                let toast_win = WebviewWindowBuilder::new(
                    app,
                    "toast",
                    WebviewUrl::App("index.html#toast".into()),
                )
                .title("")
                .inner_size(340.0, 200.0)
                .decorations(false)
                .transparent(true)
                .always_on_top(true)
                .skip_taskbar(true)
                .resizable(false)
                .focused(false)
                .visible(false)
                .build()?;

                toast_win.set_ignore_cursor_events(true)?;

                #[cfg(target_os = "macos")]
                #[allow(deprecated)]
                {
                    use cocoa::appkit::NSColor;
                    use cocoa::base::{id, nil, NO};
                    use raw_window_handle::HasWindowHandle;

                    if let Ok(raw_handle) = toast_win.window_handle() {
                        if let raw_window_handle::RawWindowHandle::AppKit(appkit) =
                            raw_handle.as_raw()
                        {
                            let ns_view = appkit.ns_view.as_ptr() as id;
                            unsafe {
                                let ns_win: id = objc::msg_send![ns_view, window];
                                let clear = NSColor::clearColor(nil);
                                let _: () = msg_send![ns_win, setBackgroundColor: clear];
                                let _: () = msg_send![ns_win, setOpaque: NO];
                                let _: () = msg_send![ns_win, setHasShadow: NO];
                                let style_mask: u64 = msg_send![ns_win, styleMask];
                                let _: () =
                                    msg_send![ns_win, setStyleMask: style_mask | (1u64 << 7)];
                                let _: () = msg_send![ns_win, setLevel: 25_i64];
                            }
                        }
                    }
                }
            }

            // Create quick-capture popup window
            {
                use tauri::{WebviewUrl, WebviewWindowBuilder};

                let qc_win = WebviewWindowBuilder::new(
                    app,
                    "quick-capture",
                    WebviewUrl::App("index.html#quick-capture".into()),
                )
                .title("Quick Capture")
                .inner_size(400.0, 160.0)
                .decorations(false)
                .transparent(true)
                .skip_taskbar(true)
                .resizable(false)
                .visible(false)
                .build()?;

                #[cfg(target_os = "macos")]
                #[allow(deprecated)]
                {
                    use cocoa::appkit::NSColor;
                    use cocoa::base::{id, nil, NO};
                    use raw_window_handle::HasWindowHandle;

                    if let Ok(raw_handle) = qc_win.window_handle() {
                        if let raw_window_handle::RawWindowHandle::AppKit(appkit) =
                            raw_handle.as_raw()
                        {
                            let ns_view = appkit.ns_view.as_ptr() as id;
                            unsafe {
                                let ns_win: id = objc::msg_send![ns_view, window];
                                let clear = NSColor::clearColor(nil);
                                let _: () = msg_send![ns_win, setBackgroundColor: clear];
                                let _: () = msg_send![ns_win, setOpaque: NO];
                                let _: () = msg_send![ns_win, setHasShadow: NO];
                            }
                        }
                    }
                }
            }

            // Register global shortcuts
            use tauri::Manager;
            use tauri_plugin_global_shortcut::GlobalShortcutExt;

            let toggle_shortcut = "CmdOrCtrl+K"
                .parse::<tauri_plugin_global_shortcut::Shortcut>()
                .unwrap();
            let spotlight_shortcut = "CmdOrCtrl+Shift+K"
                .parse::<tauri_plugin_global_shortcut::Shortcut>()
                .unwrap();
            let quick_capture_shortcut = "CmdOrCtrl+Shift+N"
                .parse::<tauri_plugin_global_shortcut::Shortcut>()
                .unwrap();

            let state: tauri::State<Arc<RwLock<AppState>>> = app.state();
            let state_clone = state.inner().clone();
            let watcher_state: tauri::State<Arc<tokio::sync::Mutex<Option<indexer::FileWatcher>>>> =
                app.state();
            let watcher_clone = watcher_state.inner().clone();

            // Save app_handle
            {
                let state_for_handle = state_clone.clone();
                let app_handle = handle.clone();
                tauri::async_runtime::block_on(async {
                    let mut s = state_for_handle.write().await;
                    s.app_handle = Some(app_handle);
                });
            }

            // Register all global shortcuts
            {
                let handle_for_shortcuts = handle.clone();
                let toggle = toggle_shortcut;
                let spotlight = spotlight_shortcut;
                let quick_capture = quick_capture_shortcut;
                app.global_shortcut().on_shortcuts(
                    [toggle, spotlight, quick_capture],
                    move |_app, shortcut, event| {
                        use tauri::Emitter;
                        use tauri_plugin_global_shortcut::ShortcutState;
                        if event.state != ShortcutState::Pressed {
                            return;
                        }
                        if *shortcut == toggle {
                            let _ = handle_for_shortcuts.emit("toggle-spotlight", ());
                        } else if *shortcut == spotlight {
                            if let Some(window) = handle_for_shortcuts.get_webview_window("main") {
                                if window.is_visible().unwrap_or(false) {
                                    let _ = window.hide();
                                    set_main_window_dock_visibility(&handle_for_shortcuts, false);
                                } else {
                                    set_main_window_dock_visibility(&handle_for_shortcuts, true);
                                    let _ = window.show();
                                    let _ = window.set_focus();
                                    let _ = handle_for_shortcuts.emit("show-memory", ());
                                }
                            }
                        } else if *shortcut == quick_capture {
                            if let Some(window) =
                                handle_for_shortcuts.get_webview_window("quick-capture")
                            {
                                #[cfg(target_os = "macos")]
                                #[allow(deprecated)]
                                {
                                    use cocoa::base::id;
                                    use raw_window_handle::HasWindowHandle;
                                    if let Ok(raw_handle) = window.window_handle() {
                                        if let raw_window_handle::RawWindowHandle::AppKit(appkit) = raw_handle.as_raw() {
                                            let ns_view = appkit.ns_view.as_ptr() as id;
                                            unsafe {
                                                let ns_win: id = objc::msg_send![ns_view, window];
                                                let visible: bool = objc::msg_send![ns_win, isVisible];
                                                if visible {
                                                    // orderOut removes the window without
                                                    // triggering macOS window promotion
                                                    let _: () = objc::msg_send![ns_win, orderOut: ns_win];
                                                } else {
                                                    // makeKeyAndOrderFront shows + focuses
                                                    // without activating the app (main stays put)
                                                    let _: () = objc::msg_send![ns_win, setLevel: 3_i64]; // NSFloatingWindowLevel
                                                    let _: () = objc::msg_send![ns_win, makeKeyAndOrderFront: ns_win];
                                                    tauri::async_runtime::spawn({
                                                        let h = handle_for_shortcuts.clone();
                                                        async move {
                                                            let _ = crate::search::position_quick_capture(h).await;
                                                        }
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                                #[cfg(not(target_os = "macos"))]
                                {
                                    if window.is_visible().unwrap_or(false) {
                                        let _ = window.hide();
                                    } else {
                                        let _ = window.show();
                                        let _ = window.set_focus();
                                    }
                                }
                            }
                        }
                    },
                )?;
            }

            // Tray icon: left-click toggles window, right-click menu with Show / Status / Quit
            {
                use tauri::menu::{MenuBuilder, MenuItemBuilder};
                use tauri::tray::TrayIconEvent;
                use tauri::Manager;

                let show_item = MenuItemBuilder::with_id("show", "Show Wenlan").build(app)?;
                let status_item = MenuItemBuilder::with_id("status", "Status: Starting…")
                    .enabled(false)
                    .build(app)?;
                let quit_item = MenuItemBuilder::with_id("quit", "Quit Wenlan").build(app)?;
                let tray_menu = MenuBuilder::new(app)
                    .item(&show_item)
                    .separator()
                    .item(&status_item)
                    .separator()
                    .item(&quit_item)
                    .build()?;

                let tray = app
                    .tray_by_id("main")
                    .or_else(|| app.tray_by_id("default"));
                if let Some(tray) = tray {
                    let _ = tray.set_menu(Some(tray_menu));

                    // Spawn health poller; it updates the icon as state changes.
                    let signal = crate::tray_health::spawn_poller(handle.clone());

                    // Periodically refresh the status label from the signal.
                    {
                        let status_item = status_item.clone();
                        let sig = signal.clone();
                        tauri::async_runtime::spawn(async move {
                            loop {
                                let label = match sig.current() {
                                    crate::tray_health::DaemonState::Up => {
                                        "Status: Up".to_string()
                                    }
                                    crate::tray_health::DaemonState::Starting => {
                                        "Status: Starting…".to_string()
                                    }
                                    crate::tray_health::DaemonState::Down => {
                                        let n = sig.consecutive_down_count();
                                        if n >= 3 {
                                            format!("Status: Down ({}s)", n as u32 * 5)
                                        } else {
                                            "Status: Down".to_string()
                                        }
                                    }
                                };
                                let _ = status_item.set_text(&label);
                                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            }
                        });
                    }

                    let handle_for_tray = handle.clone();
                    tray.on_tray_icon_event(move |_tray, event| {
                        if let TrayIconEvent::Click { button_state, .. } = event {
                            if button_state == tauri::tray::MouseButtonState::Up {
                                if let Some(win) = handle_for_tray.get_webview_window("main") {
                                    if win.is_visible().unwrap_or(false) {
                                        let _ = win.hide();
                                        set_main_window_dock_visibility(&handle_for_tray, false);
                                    } else {
                                        set_main_window_dock_visibility(&handle_for_tray, true);
                                        let _ = win.show();
                                        let _ = win.set_focus();
                                    }
                                }
                            }
                        }
                    });

                    let handle_for_menu = handle.clone();
                    tray.on_menu_event(move |_tray, event| match event.id().as_ref() {
                        "show" => {
                            if let Some(win) = handle_for_menu.get_webview_window("main") {
                                set_main_window_dock_visibility(&handle_for_menu, true);
                                let _ = win.show();
                                let _ = win.set_focus();
                            }
                        }
                        "quit" => {
                            if let Err(e) = request_full_quit(&handle_for_menu) {
                                log::error!("[tray] failed to request guarded quit: {e}");
                                force_full_quit(handle_for_menu.clone());
                            }
                        }
                        _ => {}
                    });
                }
            }

            // Launch wenlan-server daemon as a sidecar process.
            // If a daemon is already running on the port, the sidecar exits cleanly.
            // The shell plugin kills the child when the Tauri app exits.
            //
            // Skip the sidecar only when the current Wenlan launchd service
            // already targets this app-selected data root. A stale launchd
            // plist can exist during migration and first-run repair, but it
            // must not suppress the selected-data-dir sidecar fallback.
            if !daemon_startup_preflight_ok {
                log::warn!(
                    "[init] skipping daemon sidecar because server plist preflight failed"
                );
            } else {
                let launchd_managed =
                    crate::lifecycle::current_server_plist_matches_selected_data_dir();
                if launchd_managed {
                    log::info!(
                        "[init] launchd-managed daemon detected, skipping sidecar spawn"
                    );
                } else if let Err(e) = crate::daemon_start::spawn_daemon_sidecar(app.handle()) {
                    log::error!("[init] {e}. Run: xattr -cr /Applications/Origin.app or /Applications/Wenlan.app");
                }
            }

            // Wait for daemon health, then initialize local state + file watcher
            if !daemon_startup_preflight_ok {
                log::warn!(
                    "[init] skipping daemon health/config hydration because server plist preflight failed"
                );
            } else {
                let init_state = state_clone.clone();
                let remote_handle = handle.clone();
                tauri::async_runtime::spawn(async move {
                // Health check the daemon with exponential backoff
                let client = {
                    let s = init_state.read().await;
                    s.client.clone()
                };
                for i in 0..10u32 {
                    match client.health().await {
                        Ok(health) => {
                            log::info!("[init] Daemon healthy (v{})", health.version);
                            // The daemon binary comes from a separate install
                            // path (LaunchAgent, sidecar, or dev checkout) —
                            // a stale one can hold the port and answer health
                            // while breaking newer API calls.
                            if health.version != env!("CARGO_PKG_VERSION") {
                                log::warn!(
                                    "[init] Daemon version mismatch: daemon v{}, app v{} — a stale daemon may be holding port 7878; restart it (e.g. `wenlan restart`)",
                                    health.version,
                                    env!("CARGO_PKG_VERSION")
                                );
                            }
                            break;
                        }
                        Err(e) => {
                            if i == 9 {
                                log::error!("[init] Daemon not reachable after retries: {}", e);
                                return;
                            }
                            let delay = std::time::Duration::from_millis(200 * (1 << i));
                            log::warn!(
                                "[init] Daemon not ready (attempt {}): {} — retrying in {:?}",
                                i + 1,
                                e,
                                delay
                            );
                            tokio::time::sleep(delay).await;
                        }
                    }
                }

                let daemon_config = match client.get_config().await {
                    Ok(config) => Some(config),
                    Err(e) => {
                        log::warn!(
                            "[init] Daemon config unavailable after health check, falling back to app-local bootstrap config: {}",
                            e
                        );
                        None
                    }
                };

                // Initialize local state (activities, config, file sources)
                let paths = {
                    let mut state = init_state.write().await;
                    match state.initialize_local().await {
                        Ok(paths) => paths,
                        Err(e) => {
                            log::error!("Failed to initialize local state: {}", e);
                            return;
                        }
                    }
                };

                // Set up file watcher for configured paths
                if !paths.is_empty() {
                    let mut watcher_guard = watcher_clone.lock().await;
                    if watcher_guard.is_none() {
                        match indexer::create_file_watcher(init_state.clone()) {
                            Ok(w) => *watcher_guard = Some(w),
                            Err(e) => {
                                log::error!("Failed to create file watcher: {}", e);
                                return;
                            }
                        }
                    }
                    if let Some(w) = watcher_guard.as_mut() {
                        for path in &paths {
                            if let Err(e) = indexer::watch_path(w, path) {
                                log::error!("Failed to watch path {}: {}", path.display(), e);
                            }
                        }
                    }
                    drop(watcher_guard);
                }

                // Run initial sync
                if let Err(e) = indexer::sync_source("local_files", &init_state).await {
                    log::error!("Startup sync failed: {}", e);
                }

                let remote_access_enabled = daemon_config
                    .as_ref()
                    .map(|config| config.remote_access_enabled)
                    .unwrap_or_else(|| config::load_config().remote_access_enabled);
                if remote_access_enabled {
                    tauri::async_runtime::spawn(async move {
                        log::info!("[remote-access] Auto-starting tunnel (config enabled)");
                        crate::remote_access::toggle_on(remote_handle, false).await;
                    });
                }
                });
            }

            // Check for updates on startup; prompt user if one is available
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                crate::updater::check_and_prompt(handle).await;
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            search::search,
            search::get_index_status,
            search::add_watch_path,
            search::remove_watch_path,
            search::reindex,
            search::connect_source,
            search::disconnect_source,
            search::sync_source,
            search::list_sources,
            search::add_source,
            search::remove_source,
            search::list_registered_sources,
            search::sync_registered_source,
            search::daemon_version,
            search::upload_source_file,
            search::list_watch_paths,
            search::list_indexed_files,
            search::delete_file_chunks,
            search::delete_by_time_range,
            search::delete_bulk,
            search::open_file,
            search::read_source_dir,
            search::detect_obsidian_vaults,
            search::read_text_file,
            search::quick_capture,
            search::ingest_clipboard,
            search::ingest_webpage,
            search::distill_review,
            search::redistill_page,
            search::get_api_key,
            search::set_api_key,
            search::get_chunks,
            search::update_chunk,
            search::list_activities,
            search::rebuild_activities,
            search::get_capture_stats,
            search::get_pipeline_status,
            search::list_all_tags,
            search::set_document_tags,
            search::delete_tag,
            search::suggest_tags,
            search::dismiss_quick_capture,
            search::position_quick_capture,
            search::get_session_snapshots,
            search::get_snapshot_captures,
            search::get_snapshot_captures_with_content,
            search::delete_snapshot,
            search::list_spaces,
            search::get_space,
            search::create_space,
            search::update_space,
            search::delete_space,
            search::move_space,
            search::confirm_space,
            search::reorder_space,
            search::toggle_space_starred,
            search::set_document_space,
            search::add_space,
            search::remove_space,
            search::rename_space,
            search::pin_space,
            search::set_traffic_lights_visible,
            // Memory layer commands
            search::store_memory,
            search::import_memories_cmd,
            search::import_chat_export,
            search::list_pending_imports,
            search::list_onboarding_milestones,
            search::acknowledge_onboarding_milestone,
            search::reset_onboarding_milestones,
            search::save_temp_file,
            search::search_memory,
            search::confirm_memory,
            search::set_stability_cmd,
            search::list_memories,
            search::delete_memory,
            search::create_entity_cmd,
            search::list_entities_cmd,
            search::search_entities_cmd,
            search::get_entity_detail_cmd,
            search::update_observation_cmd,
            search::delete_observation_cmd,
            search::delete_entity_cmd,
            search::confirm_entity_cmd,
            search::confirm_observation_cmd,
            search::list_memories_cmd,
            search::get_memory_detail,
            search::get_enrichment_status,
            search::list_memories_by_ids,
            search::get_memory_stats_cmd,
            search::get_home_stats,
            search::update_memory_cmd,
            search::get_version_chain_cmd,
            search::get_memory_revisions,
            search::reclassify_memory_cmd,
            search::add_observation_cmd,
            // Pending revision commands
            search::accept_pending_revision,
            search::dismiss_pending_revision,
            search::get_pending_revision,
            search::list_pending_revisions,
            // Contradiction flag commands
            search::dismiss_contradiction,
            // Profile & agent management commands
            search::get_profile,
            search::update_profile,
            search::list_agents,
            search::get_agent,
            search::update_agent,
            search::delete_agent,
            // Pin/unpin & avatar commands
            search::pin_memory,
            search::unpin_memory,
            search::list_pinned_memories,
            search::set_avatar,
            search::get_avatar_data_url,
            search::remove_avatar,
            // Briefing commands
            search::get_briefing,
            search::get_pending_contradictions,
            // Refinery queue commands
            search::list_refinements,
            search::accept_refinement,
            search::reject_refinement,
            // Narrative commands
            search::get_profile_narrative,
            search::regenerate_narrative,
            // Activity feed command
            search::list_agent_activity,
            // Setup wizard commands
            search::get_setup_status,
            search::get_setup_completed,
            search::set_setup_completed,
            search::should_show_wizard,
            search::detect_mcp_clients_cmd,
            search::write_mcp_config,
            search::remove_raw_mcp_entry,
            search::remove_legacy_mcp_entry,
            search::get_wenlan_mcp_entry,
            search::install_client_plugin,
            search::wire_state,
            // Entity suggestion commands
            search::get_entity_suggestions_cmd,
            search::approve_entity_suggestion_cmd,
            search::dismiss_entity_suggestion_cmd,
            // Remote access commands
            search::toggle_remote_access,
            search::get_remote_access_status,
            search::rotate_remote_token,
            search::test_remote_mcp_connection,
            // Memory nurture commands
            search::get_nurture_cards_cmd,
            search::correct_memory_cmd,
            // Quality gate commands
            search::get_rejection_log,
            // Page commands
            search::get_page,
            search::create_page,
            search::create_page_draft,
            search::update_page_draft,
            search::publish_page_draft,
            search::discard_page_draft,
            search::get_page_sources,
            search::get_page_links,
            search::get_page_revisions,
            search::list_orphan_links,
            search::update_page,
            search::archive_page,
            search::delete_page,
            search::list_pages,
            search::search_pages,
            // Home delta feed commands
            search::list_recent_retrievals,
            search::list_recent_changes,
            search::list_recent_memories,
            search::list_unconfirmed_memories,
            search::list_recent_pages,
            search::list_recent_relations,
            search::export_pages_to_obsidian,
            search::export_page_to_obsidian,
            search::get_knowledge_path,
            search::count_knowledge_files,
            // Decision log commands
            search::list_decisions_cmd,
            search::list_decision_domains_cmd,
            // Model choice + system info commands
            search::get_model_choice,
            search::set_model_choice,
            search::get_resolved_routing,
            search::set_source_pin,
            search::get_system_info,
            // External LLM provider commands
            search::get_external_llm,
            search::set_external_llm,
            search::test_external_llm,
            search::list_external_models,
            search::get_external_llm_key_configured,
            // On-device model commands
            search::get_on_device_model,
            search::download_on_device_model,
            search::on_device_model_download_bytes,
            // Lifecycle commands
            search::is_run_at_login_enabled,
            search::set_run_at_login,
            search::quit_wenlan_full,
            search::quit_origin_full,
            cancel_guarded_quit_request,
            daemon_start::start_daemon_sidecar,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| match event {
            tauri::RunEvent::ExitRequested {
                code: None,
                api,
                ..
            } if !lifecycle::is_quitting() => {
                match request_full_quit(app) {
                    Ok(()) => api.prevent_exit(),
                    Err(e) => log::error!("[app] failed to request guarded quit: {e}"),
                }
            }
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Reopen {
                has_visible_windows,
                ..
            } => {
                    use tauri::Emitter;
                    use tauri::Manager;
                    if let Some(window) = app.get_webview_window("main") {
                        if !has_visible_windows {
                            let _ = app.emit("show-memory", ());
                        }
                        set_main_window_dock_visibility(app, true);
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
            }
            _ => {}
        });
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn visible_main_window_uses_regular_activation_policy() {
        assert!(matches!(
            activation_policy_for_main_window_visible(true),
            tauri::ActivationPolicy::Regular
        ));
    }

    #[test]
    fn hidden_main_window_stays_regular_activation_policy() {
        assert!(matches!(
            activation_policy_for_main_window_visible(false),
            tauri::ActivationPolicy::Regular
        ));
    }

    #[test]
    fn info_plist_does_not_make_main_app_an_agent() {
        let info_plist = include_str!("../Info.plist");

        assert!(!info_plist.contains("<key>LSUIElement</key>\n    <true/>"));
    }

    #[test]
    fn app_log_identity_uses_wenlan() {
        assert!(app_log_dir().ends_with("Library/Logs/com.wenlan.desktop"));
        assert_eq!(app_log_file_name(), "wenlan.log");
    }

    #[test]
    fn startup_reveal_fallback_is_short_but_not_immediate() {
        let delay = startup_reveal_fallback_delay();

        assert!(delay >= std::time::Duration::from_millis(500));
        assert!(delay <= std::time::Duration::from_secs(2));
    }

    #[test]
    fn startup_fallback_only_reveals_when_ready_or_visibility_is_missing() {
        assert!(!startup_reveal_fallback_needed(true, true));
        assert!(startup_reveal_fallback_needed(false, true));
        assert!(startup_reveal_fallback_needed(true, false));
        assert!(startup_reveal_fallback_needed(false, false));
    }

    #[test]
    fn repeated_guarded_quit_forces_shutdown_until_the_frontend_cancels() {
        let pending = std::sync::atomic::AtomicBool::new(false);

        assert_eq!(
            guarded_quit_action(&pending),
            GuardedQuitAction::RequestFrontendGuard
        );
        assert_eq!(
            guarded_quit_action(&pending),
            GuardedQuitAction::ForceShutdown
        );

        cancel_guarded_quit(&pending);
        assert_eq!(
            guarded_quit_action(&pending),
            GuardedQuitAction::RequestFrontendGuard
        );
    }
}
