// SPDX-License-Identifier: AGPL-3.0-only
#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

// ── App-specific modules (Tauri, sensors, UI) ──
mod ambient;
pub mod api;
pub mod events;
mod indexer;
mod lifecycle;
pub mod mcp_config;
pub mod remote_access;
mod router;
mod search;
mod sensor;
pub mod sources;
pub mod state;
// Public surface consumed by tray_menu (Task 15); suppress dead_code until then.
#[allow(dead_code)]
pub(crate) mod tray_health;
mod trigger;
mod updater;

// ── Re-export origin-core modules ──
// These re-exports let existing `crate::module::Type` paths work unchanged
// throughout the app (search.rs, state.rs, router/, etc.)
pub use origin_core::access_tracker;
pub use origin_core::activity;
pub use origin_core::briefing;
pub use origin_core::chunker;
pub use origin_core::classify;
pub use origin_core::concepts;
pub use origin_core::config;
pub use origin_core::context_packager;
pub use origin_core::contradiction;
pub use origin_core::db as memory_db;
pub use origin_core::decay;
pub use origin_core::engine;
pub use origin_core::error;
pub use origin_core::eval;
pub use origin_core::export;
pub use origin_core::extract;
pub use origin_core::importer;
pub use origin_core::llm_classifier;
pub use origin_core::llm_provider;
pub use origin_core::memory_schema;
pub use origin_core::merge;
pub use origin_core::narrative;
pub use origin_core::post_ingest;
pub use origin_core::privacy;
pub use origin_core::prompts;
pub use origin_core::quality_gate;
pub use origin_core::refinery;
pub use origin_core::rerank;
pub use origin_core::schema;
// sources is a local module (has app-specific sync logic)
pub use origin_core::spaces;
pub use origin_core::tags;
pub use origin_core::tuning;
pub use origin_core::working_memory;

use state::AppState;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Log sinks: stderr (for terminal launches, `pnpm tauri dev`) AND a
    // daily-rotating file at ~/Library/Logs/com.origin.desktop/origin.log.
    // GUI launches send stderr to /dev/null, so without the file sink any
    // setup() error — e.g. a sidecar spawn ENOENT — is silent. That is
    // exactly how the origin-server spawn regression hid for ~15 minutes
    // of live debugging before the culprit was found. Keep both sinks.
    use tracing_subscriber::prelude::*;

    let log_dir = dirs::home_dir()
        .map(|h| h.join("Library/Logs/com.origin.desktop"))
        .unwrap_or_else(std::env::temp_dir);
    let _ = std::fs::create_dir_all(&log_dir);
    let file_appender = tracing_appender::rolling::daily(&log_dir, "origin.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    // The guard flushes the background worker on drop. The app lives for
    // the full process, so leaking it is correct — we never want the
    // writer to stop flushing before exit.
    std::mem::forget(guard);

    let env_filter = || {
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new(
                "warn,origin_lib::trigger=info,origin_lib::router=info,origin_lib::sensor=info",
            )
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
        log_file = ?log_dir.join("origin.log"),
        "origin app starting; logs tee'd to file"
    );

    let app_state = AppState::new();

    let builder =
        tauri::Builder::default().plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            use tauri::Manager;
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }));

    #[cfg(debug_assertions)]
    let builder = builder.plugin(tauri_plugin_mcp::init_with_config(
        tauri_plugin_mcp::PluginConfig::new("origin".to_string())
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

            // Hide Dock icon at runtime (belt-and-suspenders with Info.plist LSUIElement).
            // Info.plist LSUIElement=true suppresses the icon before the process starts;
            // this call ensures it stays hidden even if Tauri ever resets the policy.
            #[cfg(target_os = "macos")]
            {
                app.set_activation_policy(tauri::ActivationPolicy::Accessory);
            }

            // First-run silent install — gated by opt-out flag.
            {
                use crate::lifecycle::{
                    app_plist_path, install_app_plist, install_server_plist_via_subprocess,
                    server_plist_path, user_opted_out, SystemLaunchctl,
                };

                if !user_opted_out() {
                    // Detect: missing plists OR plist path mismatch (e.g., app moved)
                    let exe_canonical = std::env::current_exe()
                        .ok()
                        .and_then(|p| std::fs::canonicalize(&p).ok());

                    let app_plist_stale = app_plist_path()
                        .ok()
                        .and_then(|p| std::fs::read_to_string(&p).ok())
                        .map(|content| {
                            if let Some(exe) = &exe_canonical {
                                !content.contains(exe.to_string_lossy().as_ref())
                            } else {
                                false
                            }
                        })
                        .unwrap_or(true); // missing plist = stale

                    let server_plist_stale = server_plist_path()
                        .ok()
                        .and_then(|p| std::fs::read_to_string(&p).ok())
                        .map(|content| {
                            if let Some(exe) = &exe_canonical {
                                let expected_server = exe.parent()
                                    .map(|p| p.join("origin-server").to_string_lossy().to_string())
                                    .unwrap_or_default();
                                !content.contains(&expected_server)
                            } else {
                                false
                            }
                        })
                        .unwrap_or(true);

                    if app_plist_stale || server_plist_stale {
                        let launchctl = SystemLaunchctl;
                        // Best-effort; failures fall through to fallback mode (existing
                        // sidecar spawn at the bottom of setup()).
                        if let Err(e) = install_server_plist_via_subprocess() {
                            log::warn!("[first-run] origin-server install failed: {e}");
                        }
                        if let Err(e) = install_app_plist(&launchctl) {
                            log::warn!("[first-run] install_app_plist failed: {e}");
                        } else {
                            log::info!("[first-run] LaunchAgents installed");
                        }
                    }
                }
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
                    // Size the window but keep it hidden until the frontend signals
                    // ready, preventing a white flash on startup.
                    let _ = win.set_size(tauri::Size::Logical(tauri::LogicalSize::new(1100.0, 720.0)));
                    let _ = win.center();
                    {
                        use tauri::Listener;
                        let win_for_ready = win.clone();
                        // Listen for the frontend "app-ready" event
                        handle.listen("app-ready", move |_| {
                            let _ = win_for_ready.show();
                            let _ = win_for_ready.set_focus();
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

            // Create transparent snip overlay window
            {
                use tauri::{WebviewUrl, WebviewWindowBuilder};

                let snip_win = WebviewWindowBuilder::new(
                    app,
                    "snip",
                    WebviewUrl::App("index.html#snip".into()),
                )
                .title("")
                .decorations(false)
                .transparent(true)
                .always_on_top(true)
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

                    if let Ok(raw_handle) = snip_win.window_handle() {
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
                                let _: () = msg_send![ns_win, setLevel: 1000_i64];
                                let screen: id = msg_send![
                                    objc::runtime::Class::get("NSScreen").unwrap(),
                                    mainScreen
                                ];
                                let frame: cocoa::foundation::NSRect = msg_send![screen, frame];
                                let _: () = msg_send![ns_win, setFrame: frame display: NO];
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

            // Create transparent ambient overlay window
            {
                use tauri::{WebviewUrl, WebviewWindowBuilder};

                let ambient_win = WebviewWindowBuilder::new(
                    app,
                    "ambient",
                    WebviewUrl::App("index.html#ambient".into()),
                )
                .title("")
                .inner_size(360.0, 400.0)
                .decorations(false)
                .transparent(true)
                .always_on_top(true)
                .skip_taskbar(true)
                .resizable(false)
                .focused(false)
                .visible(false)
                .build()?;

                #[cfg(target_os = "macos")]
                #[allow(deprecated)]
                {
                    use cocoa::appkit::NSColor;
                    use cocoa::base::{id, nil, NO};
                    use raw_window_handle::HasWindowHandle;

                    if let Ok(raw_handle) = ambient_win.window_handle() {
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

            // Create transparent icon trigger window
            {
                use tauri::{WebviewUrl, WebviewWindowBuilder};

                let icon_win = WebviewWindowBuilder::new(
                    app,
                    "icon",
                    WebviewUrl::App("index.html#icon".into()),
                )
                .title("")
                .inner_size(60.0, 44.0)
                .decorations(false)
                .transparent(true)
                .always_on_top(true)
                .skip_taskbar(true)
                .resizable(false)
                .focused(false)
                .visible(false)
                .build()?;

                #[cfg(target_os = "macos")]
                #[allow(deprecated)]
                {
                    use cocoa::appkit::NSColor;
                    use cocoa::base::{id, nil, NO};
                    use raw_window_handle::HasWindowHandle;

                    if let Ok(raw_handle) = icon_win.window_handle() {
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

                #[cfg(not(target_os = "macos"))]
                let _ = icon_win;
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
            let capture_shortcut = "CmdOrCtrl+Shift+M"
                .parse::<tauri_plugin_global_shortcut::Shortcut>()
                .unwrap();
            let snip_shortcut = "CmdOrCtrl+Shift+S"
                .parse::<tauri_plugin_global_shortcut::Shortcut>()
                .unwrap();
            let quick_capture_shortcut = "CmdOrCtrl+Shift+N"
                .parse::<tauri_plugin_global_shortcut::Shortcut>()
                .unwrap();
            let ambient_shortcut = "CmdOrCtrl+Shift+O"
                .parse::<tauri_plugin_global_shortcut::Shortcut>()
                .unwrap();

            let state: tauri::State<Arc<RwLock<AppState>>> = app.state();
            let state_clone = state.inner().clone();
            let watcher_state: tauri::State<Arc<tokio::sync::Mutex<Option<indexer::FileWatcher>>>> =
                app.state();
            let watcher_clone = watcher_state.inner().clone();

            // ── Unified trigger channel ──
            let (trigger_tx, trigger_rx) =
                tokio::sync::mpsc::channel::<trigger::types::TriggerEvent>(32);
            let (bundle_tx, bundle_rx) =
                tokio::sync::mpsc::channel::<router::bundle::ContextBundle>(8);

            // Smart router (tokio task)
            let router_state = state_clone.clone();
            tauri::async_runtime::spawn(router::intent::run_router(
                trigger_rx,
                bundle_tx,
                router_state,
            ));

            // Context consumer (tokio task)
            let consumer_state = state_clone.clone();
            tauri::async_runtime::spawn(router::intent::run_context_consumer(
                bundle_rx,
                consumer_state,
            ));

            // Save trigger_tx and app_handle
            let trigger_tx_for_state = trigger_tx.clone();
            {
                let state_for_handle = state_clone.clone();
                let app_handle = handle.clone();
                tauri::async_runtime::block_on(async {
                    let mut s = state_for_handle.write().await;
                    s.app_handle = Some(app_handle);
                    s.trigger_tx = Some(trigger_tx_for_state);
                });
            }

            // Ambient context monitor
            let ambient_state = state_clone.clone();
            tauri::async_runtime::spawn(crate::ambient::monitor::run_ambient_monitor(
                ambient_state,
            ));

            // Spawn selection sensor if enabled
            {
                let sel_enabled = crate::config::load_config().selection_capture_enabled;
                if sel_enabled {
                    let sel_tx = trigger_tx.clone();
                    let stop_flag = tauri::async_runtime::block_on(async {
                        state_clone.read().await.selection_stop.clone()
                    });
                    crate::trigger::selection::spawn_selection_sensor(sel_tx, stop_flag);
                }
            }

            // Register all global shortcuts
            {
                let handle_for_shortcuts = handle.clone();
                let trigger_tx_shortcut = trigger_tx;
                let toggle = toggle_shortcut;
                let spotlight = spotlight_shortcut;
                let capture = capture_shortcut;
                let snip = snip_shortcut;
                let quick_capture = quick_capture_shortcut;
                let ambient = ambient_shortcut;
                app.global_shortcut().on_shortcuts(
                    [toggle, spotlight, capture, snip, quick_capture, ambient],
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
                                } else {
                                    let _ = window.show();
                                    let _ = window.set_focus();
                                    let _ = handle_for_shortcuts.emit("show-memory", ());
                                }
                            }
                        } else if *shortcut == capture {
                            if crate::config::load_config().screen_capture_enabled {
                                let _ = trigger_tx_shortcut
                                    .try_send(trigger::types::TriggerEvent::ManualHotkey);
                            }
                        } else if *shortcut == snip {
                            if crate::config::load_config().screen_capture_enabled {
                                if let Some(window) =
                                    handle_for_shortcuts.get_webview_window("snip")
                                {
                                    let _ = window.show();
                                    let _ = window.set_focus();
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
                        } else if *shortcut == ambient {
                            if let Some(win) = handle_for_shortcuts.get_webview_window("ambient") {
                                if win.is_visible().unwrap_or(false) {
                                    let _ = win.hide();
                                } else {
                                    let _ = win.show();
                                    let h = handle_for_shortcuts.clone();
                                    tauri::async_runtime::spawn(async move {
                                        let _ = search::position_ambient_bottom_right(
                                            h.clone(),
                                            400.0,
                                            200.0,
                                        )
                                        .await;
                                        let _ = h.emit("trigger-ambient", ());
                                    });
                                }
                            }
                        }
                    },
                )?;
            }

            // Tray icon: left-click toggles window, right-click menu with Show/Quit
            {
                use tauri::Manager;
                use tauri::menu::{MenuBuilder, MenuItemBuilder};
                use tauri::tray::TrayIconEvent;

                let show_item = MenuItemBuilder::with_id("show", "Show Origin").build(app)?;
                let quit_item = MenuItemBuilder::with_id("quit", "Quit Origin").build(app)?;
                let tray_menu = MenuBuilder::new(app)
                    .item(&show_item)
                    .separator()
                    .item(&quit_item)
                    .build()?;

                let tray = app.tray_by_id("main").or_else(|| {
                    app.tray_by_id("default")
                });
                if let Some(tray) = tray {
                    let _ = tray.set_menu(Some(tray_menu));

                    let handle_for_tray = handle.clone();
                    tray.on_tray_icon_event(move |_tray, event| {
                        if let TrayIconEvent::Click { button_state, .. } = event {
                            if button_state == tauri::tray::MouseButtonState::Up {
                                if let Some(win) = handle_for_tray.get_webview_window("main") {
                                    if win.is_visible().unwrap_or(false) {
                                        let _ = win.hide();
                                    } else {
                                        let _ = win.show();
                                        let _ = win.set_focus();
                                    }
                                }
                            }
                        }
                    });

                    let handle_for_menu = handle.clone();
                    tray.on_menu_event(move |_tray, event| {
                        match event.id().as_ref() {
                            "show" => {
                                if let Some(win) = handle_for_menu.get_webview_window("main") {
                                    let _ = win.show();
                                    let _ = win.set_focus();
                                }
                            }
                            "quit" => {
                                std::process::exit(0);
                            }
                            _ => {}
                        }
                    });
                }
            }

            // Launch origin-server daemon as a sidecar process.
            // If a daemon is already running on the port, the sidecar exits cleanly.
            // The shell plugin kills the child when the Tauri app exits.
            {
                use tauri_plugin_shell::ShellExt;
                match app.shell().sidecar("origin-server") {
                    Ok(sidecar) => match sidecar.spawn() {
                        Ok((mut rx, _child)) => {
                            log::info!(
                                "[init] Spawned origin-server daemon (pid {})",
                                _child.pid()
                            );
                            tauri::async_runtime::spawn(async move {
                                use tauri_plugin_shell::process::CommandEvent;
                                while let Some(event) = rx.recv().await {
                                    match event {
                                        CommandEvent::Stdout(line) => {
                                            log::info!(
                                                "[daemon] {}",
                                                String::from_utf8_lossy(&line)
                                            );
                                        }
                                        CommandEvent::Stderr(line) => {
                                            log::warn!(
                                                "[daemon] {}",
                                                String::from_utf8_lossy(&line)
                                            );
                                        }
                                        CommandEvent::Terminated(status) => {
                                            log::warn!("[daemon] exited: {:?}", status);
                                            break;
                                        }
                                        _ => {}
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            log::error!("[init] Failed to spawn origin-server sidecar: {}. Run: xattr -cr /Applications/Origin.app", e);
                        }
                    },
                    Err(e) => {
                        log::error!("[init] Failed to create origin-server sidecar command: {}", e);
                    }
                }
            }

            // Wait for daemon health, then initialize local state + file watcher
            let init_state = state_clone.clone();
            tauri::async_runtime::spawn(async move {
                // Health check the daemon with exponential backoff
                let client = {
                    let s = init_state.read().await;
                    s.client.clone()
                };
                for i in 0..10u32 {
                    match client.health().await {
                        Ok(_) => {
                            log::info!("[init] Daemon healthy");
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
            });

            // Auto-start remote access tunnel if previously enabled
            {
                let config = config::load_config();
                if config.remote_access_enabled {
                    let handle = app.handle().clone();
                    tauri::async_runtime::spawn(async move {
                        log::info!("[remote-access] Auto-starting tunnel (config enabled)");
                        crate::remote_access::toggle_on(handle, false).await;
                    });
                }
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
            search::list_watch_paths,
            search::list_indexed_files,
            search::delete_file_chunks,
            search::delete_by_time_range,
            search::delete_bulk,
            search::open_file,
            search::quick_capture,
            search::ingest_clipboard,
            search::get_clipboard_enabled,
            search::set_clipboard_enabled,
            search::get_api_key,
            search::set_api_key,
            search::get_chunks,
            search::update_chunk,
            search::list_activities,
            search::rebuild_activities,
            search::trigger_manual_capture,
            search::get_capture_stats,
            search::capture_region,
            search::list_all_tags,
            search::set_document_tags,
            search::delete_tag,
            search::suggest_tags,
            search::dismiss_quick_capture,
            search::position_quick_capture,
            search::position_ambient_bottom_right,
            search::get_working_memory,
            search::get_session_snapshots,
            search::get_snapshot_captures,
            search::get_snapshot_captures_with_content,
            search::delete_snapshot,
            search::get_skip_apps,
            search::set_skip_apps,
            search::get_skip_title_patterns,
            search::set_skip_title_patterns,
            search::get_private_browsing_detection,
            search::set_private_browsing_detection,
            search::list_spaces,
            search::get_space,
            search::create_space,
            search::update_space,
            search::delete_space,
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
            search::list_memories_by_ids,
            search::get_memory_stats_cmd,
            search::get_home_stats,
            search::update_memory_cmd,
            search::get_version_chain_cmd,
            search::reclassify_memory_cmd,
            search::add_observation_cmd,
            // Pending revision commands
            search::accept_pending_revision,
            search::dismiss_pending_revision,
            search::get_pending_revision,
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
            search::record_eval_signal,
            search::set_avatar,
            search::get_avatar_data_url,
            search::remove_avatar,
            // Briefing commands
            search::get_briefing,
            search::get_pending_contradictions,
            // Narrative commands
            search::get_profile_narrative,
            search::regenerate_narrative,
            // Activity feed command
            search::list_agent_activity,
            // Setup wizard commands
            search::get_setup_completed,
            search::set_setup_completed,
            search::should_show_wizard,
            search::detect_mcp_clients_cmd,
            search::write_mcp_config,
            search::get_origin_mcp_entry,
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
            // Concept commands
            search::get_concept,
            search::get_concept_sources,
            search::update_concept,
            search::archive_concept,
            search::delete_concept,
            search::list_concepts,
            search::search_concepts,
            // Home delta feed commands
            search::list_recent_retrievals,
            search::list_recent_changes,
            search::list_recent_memories,
            search::list_unconfirmed_memories,
            search::list_recent_concepts,
            search::list_recent_relations,
            search::export_concepts_to_obsidian,
            search::export_concept_to_obsidian,
            search::get_knowledge_path,
            search::count_knowledge_files,
            // Distillation trigger commands
            search::trigger_distillation,
            search::redistill_concept,
            // Ambient overlay commands
            search::dismiss_ambient_card,
            search::trigger_ambient,
            search::get_ambient_mode,
            search::set_ambient_mode,
            search::get_screen_capture_enabled,
            search::set_screen_capture_enabled,
            search::check_screen_permission,
            search::request_screen_permission,
            // Selection trigger commands
            search::get_selection_capture_enabled,
            search::set_selection_capture_enabled,
            search::check_accessibility_permission,
            search::request_accessibility_permission,
            search::trigger_icon_click,
            // Decision log commands
            search::list_decisions_cmd,
            search::list_decision_domains_cmd,
            // Model choice + system info commands
            search::get_model_choice,
            search::set_model_choice,
            search::get_system_info,
            // External LLM provider commands
            search::get_external_llm,
            search::set_external_llm,
            search::test_external_llm,
            // On-device model commands
            search::get_on_device_model,
            search::download_on_device_model,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            #[cfg(target_os = "macos")]
            if let tauri::RunEvent::Reopen {
                has_visible_windows,
                ..
            } = event
            {
                use tauri::Emitter;
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    if !has_visible_windows {
                        let _ = app.emit("show-memory", ());
                    }
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        });
}
