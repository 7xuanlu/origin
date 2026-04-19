// SPDX-License-Identifier: AGPL-3.0-only

/// Selection trigger — polls AXSelectedText from the focused app every 150ms.
/// When highlighted text is ≥ MIN_SELECTION_CHARS and stable for DEBOUNCE_MS,
/// emits TriggerEvent::TextSelected.
///
/// Requires macOS Accessibility permission (AXIsProcessTrusted).
/// If permission is not granted, the sensor loop does not start (silent no-op).
use crate::trigger::types::{SelectionEvent, TriggerEvent};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

/// How often to poll AXSelectedText (milliseconds).
const POLL_INTERVAL_MS: u64 = 150;

/// Emit only after selection is stable for this many ms.
const DEBOUNCE_MS: u64 = 300;

/// Minimum selection length to bother searching.
const MIN_SELECTION_CHARS: usize = 15;

// ─── Debounce state machine ───────────────────────────────────────────────────

/// State for the debounce logic (testable without AX dependency).
pub struct DebounceState {
    /// The last selection we successfully sent (prevents re-firing for same text).
    pub last_sent: String,
    /// Text we're currently waiting to confirm via debounce.
    pub pending_text: String,
    /// When debounce started for `pending_text`.
    pub debounce_start: Option<std::time::Instant>,
}

/// What the debounce state machine wants the caller to do.
pub enum DebounceAction {
    /// Fire: emit this text as a trigger event.
    Fire { text: String },
    /// Wait: debounce in progress, try again later.
    Wait,
    /// Skip: text too short, empty, or same as last sent.
    Skip,
}

/// Process one poll result through the debounce state machine.
///
/// `new_text` — the freshly-read selected text (None = nothing selected).
/// `state`    — mutable debounce state (persists between calls).
/// `debounce_ms` / `min_chars` — thresholds (injectable for tests).
pub fn process_poll(
    new_text: Option<&str>,
    state: &mut DebounceState,
    debounce_ms: u64,
    min_chars: usize,
) -> DebounceAction {
    let text = match new_text {
        Some(t) if t.chars().count() >= min_chars => t,
        _ => {
            // Nothing selected or too short — reset all state so the same
            // text can fire again on the next selection.
            state.debounce_start = None;
            state.pending_text = String::new();
            state.last_sent = String::new();
            return DebounceAction::Skip;
        }
    };

    // Don't re-fire for the same text we just sent
    if text == state.last_sent {
        return DebounceAction::Skip;
    }

    if text == state.pending_text {
        // Same as pending — check if debounce has elapsed
        match state.debounce_start {
            Some(start) if start.elapsed().as_millis() >= debounce_ms as u128 => {
                // Debounce elapsed — fire
                let fired_text = state.pending_text.clone();
                state.last_sent = fired_text.clone();
                state.debounce_start = None;
                DebounceAction::Fire { text: fired_text }
            }
            Some(_) => DebounceAction::Wait,
            None => {
                // debounce_start was cleared (e.g. selection briefly dropped below
                // min_chars and returned) — restart the timer
                state.debounce_start = Some(std::time::Instant::now());
                DebounceAction::Wait
            }
        }
    } else {
        // New text — start/reset debounce
        state.pending_text = text.to_string();
        state.debounce_start = Some(std::time::Instant::now());
        DebounceAction::Wait
    }
}

// ─── Thread entry point ───────────────────────────────────────────────────────

/// Spawn the selection sensor on a dedicated std::thread.
/// The thread stops when `stop` is set to true.
pub fn spawn_selection_sensor(trigger_tx: Sender<TriggerEvent>, stop: Arc<AtomicBool>) {
    std::thread::Builder::new()
        .name("selection-sensor".to_string())
        .spawn(move || run_selection_sensor(trigger_tx, stop))
        .expect("failed to spawn selection-sensor thread");
}

fn run_selection_sensor(trigger_tx: Sender<TriggerEvent>, stop: Arc<AtomicBool>) {
    #[cfg(target_os = "macos")]
    {
        if !ax_is_trusted() {
            log::warn!("[selection] Accessibility not granted — sensor not starting");
            return;
        }

        // Create shared dedup state
        let last_fired: Arc<std::sync::Mutex<String>> =
            Arc::new(std::sync::Mutex::new(String::new()));

        // Thread 1: CGEventTap — fires on mouse-up after drag (works for Chrome, Electron, Terminal)
        {
            let tx = trigger_tx.clone();
            let stop2 = stop.clone();
            let lf = last_fired.clone();
            std::thread::Builder::new()
                .name("selection-tap".to_string())
                .spawn(move || event_tap::run_event_tap(tx, stop2, lf))
                .expect("failed to spawn selection-tap thread");
        }

        // Thread 2 (this thread): slow AX poll — catches keyboard selections (Shift+Arrow)
        // in native macOS apps where AX works. Runs at 300ms to reduce CPU overhead.
        // The CGEventTap handles mouse-drag selections; this handles keyboard-only selections.
        log::info!("[selection] AX poll started (300ms, keyboard selections)");

        let mut state = DebounceState {
            last_sent: String::new(),
            pending_text: String::new(),
            debounce_start: None,
        };

        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(300));

            let selected = read_selected_text_ax();
            let action = process_poll(
                selected.as_deref(),
                &mut state,
                DEBOUNCE_MS,
                MIN_SELECTION_CHARS,
            );
            if let DebounceAction::Fire { text } = action {
                // Dedup: skip if same text was just fired by CGEventTap
                {
                    let mut last = last_fired.lock().unwrap_or_else(|e| e.into_inner());
                    if *last == text {
                        continue;
                    }
                    *last = text.clone();
                }
                let (cursor_x, cursor_y) = cursor_position();
                log::debug!("[selection] ax-poll fire: {} chars", text.len());
                let _ = trigger_tx.try_send(TriggerEvent::TextSelected(SelectionEvent {
                    text,
                    cursor_x,
                    cursor_y,
                }));
            }
        }

        log::info!("[selection] AX poll stopped");
    }

    #[cfg(not(target_os = "macos"))]
    {
        log::info!("[selection] sensor not started (non-macOS)");
        let _ = (trigger_tx, stop);
    }
}

// ─── macOS AX API ─────────────────────────────────────────────────────────────

/// Check if this process has Accessibility permission.
pub fn ax_is_trusted() -> bool {
    #[cfg(target_os = "macos")]
    {
        #[link(name = "ApplicationServices", kind = "framework")]
        extern "C" {
            fn AXIsProcessTrusted() -> bool;
        }
        unsafe { AXIsProcessTrusted() }
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}

/// Read the currently selected text via the macOS Accessibility API.
/// Three-step chain: system_wide → AXFocusedApplication → AXFocusedUIElement → AXSelectedText.
/// Returns None if nothing is selected, Origin is frontmost, or any AX call fails.
/// Does NOT fall back to clipboard — callers handle that.
#[cfg(target_os = "macos")]
pub(crate) fn read_selected_text_ax() -> Option<String> {
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_void};

    type AXUIElementRef = *mut c_void;
    type CFStringRef = *const c_void;
    type AXError = i32;

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXUIElementCreateSystemWide() -> AXUIElementRef;
        fn AXUIElementCopyAttributeValue(
            element: AXUIElementRef,
            attribute: CFStringRef,
            value: *mut AXUIElementRef,
        ) -> AXError;
        fn AXUIElementGetPid(element: AXUIElementRef, pid: *mut i32) -> AXError;
        fn CFGetTypeID(cf: *mut c_void) -> usize;
        fn CFStringGetTypeID() -> usize;
        fn CFStringGetCString(
            the_string: CFStringRef,
            buffer: *mut c_char,
            buffer_size: i64,
            encoding: u32,
        ) -> bool;
        fn CFStringCreateWithCString(
            alloc: *mut c_void,
            cstr: *const c_char,
            encoding: u32,
        ) -> CFStringRef;
        fn CFRelease(cf: *mut c_void);
    }

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;

    unsafe {
        let system_wide = AXUIElementCreateSystemWide();
        if system_wide.is_null() {
            return None;
        }

        // Step 1: frontmost application element
        let app_attr_cstr = CString::new("AXFocusedApplication").ok()?;
        let app_attr_cf = CFStringCreateWithCString(
            std::ptr::null_mut(),
            app_attr_cstr.as_ptr(),
            K_CF_STRING_ENCODING_UTF8,
        );
        if app_attr_cf.is_null() {
            CFRelease(system_wide);
            return None;
        }

        let mut app_element: AXUIElementRef = std::ptr::null_mut();
        let app_err = AXUIElementCopyAttributeValue(system_wide, app_attr_cf, &mut app_element);
        CFRelease(system_wide);
        CFRelease(app_attr_cf as *mut c_void);
        if app_err != 0 || app_element.is_null() {
            return None;
        }

        // Skip if Origin itself is frontmost (webview always returns AXSelectedText="")
        let mut app_pid: i32 = 0;
        if AXUIElementGetPid(app_element, &mut app_pid) == 0 && app_pid == std::process::id() as i32
        {
            CFRelease(app_element);
            return None;
        }

        // Step 2: focused UI element within that app
        let focused_attr_cstr = CString::new("AXFocusedUIElement").ok()?;
        let focused_attr_cf = CFStringCreateWithCString(
            std::ptr::null_mut(),
            focused_attr_cstr.as_ptr(),
            K_CF_STRING_ENCODING_UTF8,
        );
        if focused_attr_cf.is_null() {
            CFRelease(app_element);
            return None;
        }

        let mut focused: AXUIElementRef = std::ptr::null_mut();
        let focused_err = AXUIElementCopyAttributeValue(app_element, focused_attr_cf, &mut focused);
        CFRelease(app_element);
        CFRelease(focused_attr_cf as *mut c_void);
        if focused_err != 0 || focused.is_null() {
            return None;
        }

        // Step 3: AXSelectedText from the focused element
        let sel_attr_cstr = CString::new("AXSelectedText").ok()?;
        let sel_attr_cf = CFStringCreateWithCString(
            std::ptr::null_mut(),
            sel_attr_cstr.as_ptr(),
            K_CF_STRING_ENCODING_UTF8,
        );
        if sel_attr_cf.is_null() {
            CFRelease(focused);
            return None;
        }

        let mut value: AXUIElementRef = std::ptr::null_mut();
        let sel_err = AXUIElementCopyAttributeValue(focused, sel_attr_cf, &mut value);
        CFRelease(focused);
        CFRelease(sel_attr_cf as *mut c_void);
        if sel_err != 0 || value.is_null() {
            return None;
        }

        // Must be a CFString (not CFAttributedString)
        if CFGetTypeID(value) != CFStringGetTypeID() {
            CFRelease(value);
            return None;
        }

        let mut buf: Vec<c_char> = vec![0; 4096];
        let ok = CFStringGetCString(
            value as CFStringRef,
            buf.as_mut_ptr(),
            buf.len() as i64,
            K_CF_STRING_ENCODING_UTF8,
        );
        CFRelease(value);
        if !ok {
            return None;
        }

        let text = CStr::from_ptr(buf.as_ptr()).to_str().ok()?.to_string();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

/// Read the currently selected text via clipboard simulation (universal fallback).
///
/// **Call-site contract**: This function sends `Cmd+C` into the frontmost application.
/// It MUST only be called once per user-initiated selection gesture (e.g., from a
/// CGEventTap mouse-up callback), never from a polling loop. Calling it in a loop
/// would inject spurious Cmd+C keystrokes into the user's workflow.
///
/// How it works:
///   1. Saves the current clipboard contents and change count
///   2. Sends Cmd+C via System Events (reliable from background processes)
///   3. Waits 100ms for the target app to process the copy
///   4. Reads the new clipboard; if changeCount is unchanged, nothing was selected
///   5. Restores the original clipboard
///
/// Returns None if nothing was selected (clipboard unchanged) or on error.
#[cfg(target_os = "macos")]
pub(crate) fn read_selected_text_clipboard() -> Option<String> {
    // language=AppleScript
    const SCRIPT: &str = r#"
use framework "AppKit"
use scripting additions

set pb to current application's NSPasteboard's generalPasteboard()
set origCount to pb's changeCount()
set origContents to (get the clipboard)

tell application "System Events"
    keystroke "c" using {command down}
end tell

delay 0.1

set newCount to pb's changeCount()
if newCount = origCount then
    return ""
end if

set newContents to (get the clipboard)
set the clipboard to origContents
return newContents
"#;

    let output = std::process::Command::new("osascript")
        .arg("-e")
        .arg(SCRIPT)
        .output()
        .ok()?;

    if !output.status.success() {
        log::warn!(
            "[selection] osascript failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    let text = text.trim_end_matches('\n').to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Dispatcher: try AX first, fall back to clipboard.
/// Exported for use in the CGEventTap callback.
///
/// **Important**: The clipboard fallback sends Cmd+C into the frontmost app.
/// Call this only once per user-initiated gesture (from a CGEventTap callback),
/// never from a polling loop.
#[cfg(target_os = "macos")]
pub(crate) fn get_selected_text() -> Option<String> {
    get_selected_text_with_readers(read_selected_text_ax, read_selected_text_clipboard)
}

/// Testable dispatcher — accepts injected reader functions.
/// `ax_reader`        — returns selected text via AX API (returns None if nothing selected).
/// `clipboard_reader` — returns selected text via clipboard sim (fallback).
pub(crate) fn get_selected_text_with_readers<F, G>(
    ax_reader: F,
    clipboard_reader: G,
) -> Option<String>
where
    F: FnOnce() -> Option<String>,
    G: FnOnce() -> Option<String>,
{
    ax_reader().or_else(clipboard_reader)
}

/// Get the current mouse cursor position in macOS logical screen coordinates.
/// macOS origin is bottom-left; Tauri's LogicalPosition uses top-left.
/// The caller (frontend) handles the Y-flip.
#[cfg(target_os = "macos")]
fn cursor_position() -> (f64, f64) {
    use std::os::raw::c_void;

    #[repr(C)]
    #[derive(Default)]
    struct CGPoint {
        x: f64,
        y: f64,
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventCreate(source: *mut c_void) -> *mut c_void;
        fn CGEventGetLocation(event: *mut c_void) -> CGPoint;
        fn CFRelease(cf: *mut c_void);
    }

    unsafe {
        let event = CGEventCreate(std::ptr::null_mut());
        if event.is_null() {
            return (0.0, 0.0);
        }
        let pt = CGEventGetLocation(event);
        CFRelease(event);
        (pt.x, pt.y)
    }
}

// NOTE: AXBoundsForRange was attempted for pixel-perfect selection positioning
// but is broken in Chrome/Electron (Electron issues #34755, #47393 — closed "not planned").
// All shipping macOS selection tools (PopClip, Selected, OpenFire) use mouse-up position.

/// CGEventTap-based sensor: fires on kCGEventLeftMouseUp, detects drag-selections,
/// reads selected text via AX then clipboard fallback.
#[cfg(target_os = "macos")]
mod event_tap {
    use crate::trigger::types::TriggerEvent;
    use std::os::raw::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tokio::sync::mpsc::Sender;

    // CGEvent types
    type CGEventRef = *mut c_void;
    type CGEventTapRef = *mut c_void;
    type CFMachPortRef = *mut c_void;
    type CFRunLoopSourceRef = *mut c_void;
    type CFRunLoopRef = *mut c_void;
    type CGEventType = u32;
    type CGEventMask = u64;
    type CGEventTapLocation = u32;
    type CGEventTapPlacement = u32;
    type CGEventTapOptions = u32;

    const K_CG_EVENT_LEFT_MOUSE_UP: CGEventType = 2;
    const K_CG_EVENT_LEFT_MOUSE_DOWN: CGEventType = 1;
    const K_CG_HID_EVENT_TAP: CGEventTapLocation = 0;
    const K_CG_HEAD_INSERT_EVENT_TAP: CGEventTapPlacement = 0;
    const K_CG_EVENT_TAP_OPTION_LISTEN_ONLY: CGEventTapOptions = 1;

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct CGPoint {
        pub x: f64,
        pub y: f64,
    }

    // Callback type for CGEventTapCreate
    type CGEventTapCallBack = unsafe extern "C" fn(
        proxy: *mut c_void,
        event_type: CGEventType,
        event: CGEventRef,
        user_info: *mut c_void,
    ) -> CGEventRef;

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventTapCreate(
            tap: CGEventTapLocation,
            place: CGEventTapPlacement,
            options: CGEventTapOptions,
            events_of_interest: CGEventMask,
            callback: CGEventTapCallBack,
            user_info: *mut c_void,
        ) -> CFMachPortRef;
        fn CGEventGetLocation(event: CGEventRef) -> CGPoint;
        fn CFMachPortCreateRunLoopSource(
            allocator: *mut c_void,
            port: CFMachPortRef,
            order: isize,
        ) -> CFRunLoopSourceRef;
        fn CFRunLoopGetCurrent() -> CFRunLoopRef;
        fn CFRunLoopAddSource(rl: CFRunLoopRef, source: CFRunLoopSourceRef, mode: *const c_void);
        fn CFRelease(cf: *mut c_void);
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        static kCFRunLoopDefaultMode: *const c_void;
    }

    /// Shared state passed as user_info pointer to the CGEventTap callback.
    pub struct TapState {
        pub trigger_tx: Sender<TriggerEvent>,
        pub stop: Arc<AtomicBool>,
        /// Position of the last mouse-down event (for drag detection).
        pub mouse_down_pos: std::sync::Mutex<CGPoint>,
        /// Last text that was fired (shared with AX-poll to prevent double-firing).
        pub last_fired: Arc<std::sync::Mutex<String>>,
    }

    /// CGEventTap callback. Called on every left-mouse-down and left-mouse-up.
    /// On mouse-down: records position for drag detection.
    /// On mouse-up after a drag (>3px): reads text via AX + clipboard fallback.
    pub unsafe extern "C" fn tap_callback(
        _proxy: *mut c_void,
        event_type: CGEventType,
        event: CGEventRef,
        user_info: *mut c_void,
    ) -> CGEventRef {
        let state = &*(user_info as *const TapState);

        if event_type == K_CG_EVENT_LEFT_MOUSE_DOWN {
            let pos = CGEventGetLocation(event);
            if let Ok(mut down) = state.mouse_down_pos.lock() {
                *down = pos;
            }
            return event;
        }

        if event_type != K_CG_EVENT_LEFT_MOUSE_UP {
            return event;
        }

        // Get mouse-up position and compare with mouse-down
        let up_pos = CGEventGetLocation(event);
        let min_drag_px = 3.0_f64;
        let (is_drag, _down_y) = {
            if let Ok(down) = state.mouse_down_pos.lock() {
                let dx = up_pos.x - down.x;
                let dy = up_pos.y - down.y;
                ((dx * dx + dy * dy).sqrt() > min_drag_px, down.y)
            } else {
                (false, up_pos.y)
            }
        };

        if !is_drag {
            // Non-drag click = deselection or app switch — clear dedup and hide icon
            if let Ok(mut last) = state.last_fired.lock() {
                last.clear();
            }
            let _ = state.trigger_tx.try_send(TriggerEvent::HideIcon);
            return event;
        }

        // Brief pause for the app to finish updating its selection state
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Read selected text: AX first, clipboard fallback
        let selected = super::get_selected_text();

        if let Some(text) = selected {
            if text.chars().count() >= super::MIN_SELECTION_CHARS {
                // Dedup: skip if same text was just fired by another path
                {
                    let mut last = state.last_fired.lock().unwrap_or_else(|e| e.into_inner());
                    if *last == text {
                        return event;
                    }
                    *last = text.clone();
                }
                // Mouse-up position is the only reliable cross-app signal on macOS.
                // (AXBoundsForRange is broken in Chrome/Electron — confirmed upstream.)
                log::info!(
                    "[selection] tap fire: {} chars at ({:.0},{:.0})",
                    text.len(),
                    up_pos.x,
                    up_pos.y
                );
                let _ = state.trigger_tx.try_send(TriggerEvent::TextIcon {
                    text,
                    x: up_pos.x,
                    y: up_pos.y,
                });
            }
        }

        event
    }

    /// Install the CGEventTap and run the current thread's CFRunLoop until `stop` is set.
    /// This function blocks — it must be called from a dedicated `std::thread`.
    pub fn run_event_tap(
        trigger_tx: Sender<TriggerEvent>,
        stop: Arc<AtomicBool>,
        last_fired: Arc<std::sync::Mutex<String>>,
    ) {
        // Allocate TapState on the heap; pointer lives as long as the run loop runs.
        let state = Box::new(TapState {
            trigger_tx,
            stop: stop.clone(),
            mouse_down_pos: std::sync::Mutex::new(CGPoint::default()),
            last_fired,
        });
        let state_ptr = Box::into_raw(state) as *mut c_void;

        // Listen to both mouse-down (to record position) and mouse-up (to detect drag)
        let mask: CGEventMask =
            (1u64 << K_CG_EVENT_LEFT_MOUSE_DOWN) | (1u64 << K_CG_EVENT_LEFT_MOUSE_UP);

        let tap = unsafe {
            CGEventTapCreate(
                K_CG_HID_EVENT_TAP,
                K_CG_HEAD_INSERT_EVENT_TAP,
                K_CG_EVENT_TAP_OPTION_LISTEN_ONLY,
                mask,
                tap_callback,
                state_ptr,
            )
        };

        if tap.is_null() {
            log::warn!(
                "[selection] CGEventTapCreate failed — Accessibility permission may not be granted"
            );
            // Reclaim heap allocation
            unsafe {
                let _ = Box::from_raw(state_ptr as *mut TapState);
            }
            return;
        }

        let run_loop_source =
            unsafe { CFMachPortCreateRunLoopSource(std::ptr::null_mut(), tap, 0) };

        unsafe {
            let rl = CFRunLoopGetCurrent();
            CFRunLoopAddSource(rl, run_loop_source, kCFRunLoopDefaultMode);
        }

        log::info!("[selection] CGEventTap installed — listening for drag-selections");

        // Spin in 100ms ticks so we can check the stop flag
        loop {
            unsafe {
                cg_run_loop_run_in_mode_for(0.1);
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }
        }

        unsafe {
            CFRelease(tap);
            CFRelease(run_loop_source);
            let _ = Box::from_raw(state_ptr as *mut TapState);
        }
        log::info!("[selection] CGEventTap stopped");
    }

    /// Run the current thread's CFRunLoop for `seconds` seconds.
    unsafe fn cg_run_loop_run_in_mode_for(seconds: f64) {
        #[link(name = "CoreFoundation", kind = "framework")]
        extern "C" {
            fn CFRunLoopRunInMode(
                mode: *const c_void,
                seconds: f64,
                return_after_source_handled: bool,
            ) -> i32;
        }
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn make_state() -> DebounceState {
        DebounceState {
            last_sent: String::new(),
            pending_text: String::new(),
            debounce_start: None,
        }
    }

    #[test]
    fn short_text_is_skipped() {
        let mut s = make_state();
        let action = process_poll(Some("hi"), &mut s, 300, 15);
        assert!(matches!(action, DebounceAction::Skip));
    }

    #[test]
    fn empty_text_is_skipped() {
        let mut s = make_state();
        let action = process_poll(None, &mut s, 300, 15);
        assert!(matches!(action, DebounceAction::Skip));
    }

    #[test]
    fn new_text_starts_debounce() {
        let mut s = make_state();
        let action = process_poll(Some("hello world this is test selection"), &mut s, 300, 15);
        assert!(matches!(action, DebounceAction::Wait));
        assert!(s.debounce_start.is_some());
        assert_eq!(s.pending_text, "hello world this is test selection");
    }

    #[test]
    fn same_pending_text_fires_after_debounce() {
        let mut s = make_state();
        let text = "hello world this is test selection";
        // First poll: start debounce
        process_poll(Some(text), &mut s, 300, 15);
        // Backdate debounce start to simulate 400ms elapsed
        s.debounce_start = Some(Instant::now() - Duration::from_millis(400));
        // Second poll with same text: should fire
        let action = process_poll(Some(text), &mut s, 300, 15);
        assert!(matches!(action, DebounceAction::Fire { .. }));
        assert_eq!(s.last_sent, text);
        assert!(s.debounce_start.is_none());
    }

    #[test]
    fn same_text_as_last_sent_does_not_re_fire() {
        let mut s = make_state();
        let text = "hello world this is test selection";
        s.last_sent = text.to_string();
        let action = process_poll(Some(text), &mut s, 300, 15);
        assert!(matches!(action, DebounceAction::Skip));
    }

    #[test]
    fn changing_text_resets_debounce() {
        let mut s = make_state();
        process_poll(Some("hello world this is test selection"), &mut s, 300, 15);
        // Different text arrives before debounce fires
        let action = process_poll(
            Some("different text that is also long enough"),
            &mut s,
            300,
            15,
        );
        assert!(matches!(action, DebounceAction::Wait));
        assert_eq!(s.pending_text, "different text that is also long enough");
    }

    #[test]
    fn debounce_restarts_after_brief_drop_below_min_chars() {
        let mut s = make_state();
        let text = "hello world this is test selection";
        // Start debounce
        process_poll(Some(text), &mut s, 300, 15);
        assert!(s.debounce_start.is_some());
        // Selection briefly drops below min_chars
        process_poll(Some("hi"), &mut s, 300, 15);
        assert!(s.debounce_start.is_none()); // timer cleared
                                             // Same text reappears — timer must restart
        let action = process_poll(Some(text), &mut s, 300, 15);
        assert!(matches!(action, DebounceAction::Wait));
        assert!(s.debounce_start.is_some()); // timer restarted
                                             // Simulate debounce elapsed
        s.debounce_start = Some(std::time::Instant::now() - std::time::Duration::from_millis(400));
        let action = process_poll(Some(text), &mut s, 300, 15);
        assert!(matches!(action, DebounceAction::Fire { .. }));
    }

    #[test]
    fn dispatcher_prefers_ax_result() {
        // When AX returns text, clipboard fallback is not invoked.
        // We verify by calling get_selected_text_with_readers with a mock AX that returns text
        // and a mock clipboard that panics if called.
        let result = get_selected_text_with_readers(
            || Some("selected via ax".to_string()),
            || panic!("clipboard should not be called when AX succeeds"),
        );
        assert_eq!(result, Some("selected via ax".to_string()));
    }

    #[test]
    fn dispatcher_falls_back_to_clipboard_when_ax_empty() {
        // When AX returns None, clipboard fallback is invoked.
        let result =
            get_selected_text_with_readers(|| None, || Some("selected via clipboard".to_string()));
        assert_eq!(result, Some("selected via clipboard".to_string()));
    }

    #[test]
    fn dedup_state_prevents_double_fire() {
        use std::sync::{Arc, Mutex};
        let last_fired: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let text = "hello world this is a test selection";

        // First fire should succeed (text differs from last_fired)
        let should_fire = {
            let last = last_fired.lock().unwrap();
            *last != text
        };
        assert!(should_fire);
        *last_fired.lock().unwrap() = text.to_string();

        // Second fire with same text should be suppressed
        let should_fire = {
            let last = last_fired.lock().unwrap();
            *last != text
        };
        assert!(!should_fire);
    }
}
