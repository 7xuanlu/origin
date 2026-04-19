// SPDX-License-Identifier: AGPL-3.0-only
/// User idle detection via macOS CGEventSource.
///
/// Uses CGEventSourceSecondsSinceLastEventType to query the HID system
/// for time since last keyboard/mouse event.
/// AFK threshold in seconds. Skip all capture when exceeded.
#[allow(dead_code)]
pub const AFK_THRESHOLD_SECS: f64 = 60.0;

/// Returns seconds since the user last interacted (keyboard or mouse).
/// Returns 0.0 on non-macOS or on FFI failure (fail-open: assume active).
#[cfg(target_os = "macos")]
#[allow(dead_code)]
pub fn user_idle_seconds() -> f64 {
    use std::sync::atomic::{AtomicBool, Ordering};

    // kCGEventSourceStateHIDSystemState = 1 (reads directly from HID system)
    const K_CG_EVENT_SOURCE_STATE_HID: i32 = 1;
    // kCGAnyInputEventType = ~0 (0xFFFFFFFF) — matches any HID event
    const K_CG_ANY_INPUT_EVENT_TYPE: u32 = u32::MAX;

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventSourceSecondsSinceLastEventType(sourceStateID: i32, eventType: u32) -> f64;
    }

    static LOGGED_ONCE: AtomicBool = AtomicBool::new(false);

    let secs = unsafe {
        CGEventSourceSecondsSinceLastEventType(
            K_CG_EVENT_SOURCE_STATE_HID,
            K_CG_ANY_INPUT_EVENT_TYPE,
        )
    };

    // Log the first value so we can diagnose issues
    if !LOGGED_ONCE.swap(true, Ordering::Relaxed) {
        log::info!(
            "[idle] CGEventSourceSecondsSinceLastEventType returned: {:.1}s",
            secs
        );
    }

    // Sanity check: if the function returns negative or absurdly large values
    // (e.g. no permissions, or no events ever recorded), fail-open (assume active)
    if !(0.0..=86400.0).contains(&secs) {
        return 0.0;
    }

    secs
}

#[cfg(not(target_os = "macos"))]
pub fn user_idle_seconds() -> f64 {
    0.0 // Fail-open: assume active on non-macOS
}
