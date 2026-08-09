// SPDX-License-Identifier: Apache-2.0
//! Low-overhead host signals used only to veto new background-heavy work.

use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) enum HostActivitySnapshot {
    /// The current platform has no production probe; portable CPU/RAM admission
    /// remains authoritative.
    #[cfg(any(not(target_os = "macos"), test))]
    Unsupported,
    /// A supported platform failed to produce trustworthy telemetry.
    Unavailable,
    Observed {
        thermal_state: u8,
        idle_for: Duration,
    },
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn wenlan_macos_thermal_state() -> i32;
    fn wenlan_macos_seconds_since_last_input() -> f64;
}

#[cfg(target_os = "macos")]
pub(crate) fn sample_host_activity() -> HostActivitySnapshot {
    // SAFETY: Both functions are zero-argument wrappers compiled into this
    // crate. They synchronously return scalar values from public macOS APIs and
    // retain no Rust pointers or callbacks.
    let (thermal_state, seconds_since_input) = unsafe {
        (
            wenlan_macos_thermal_state(),
            wenlan_macos_seconds_since_last_input(),
        )
    };
    let Ok(thermal_state) = u8::try_from(thermal_state) else {
        return HostActivitySnapshot::Unavailable;
    };
    if thermal_state > 3 {
        return HostActivitySnapshot::Unavailable;
    }
    let Ok(idle_for) = Duration::try_from_secs_f64(seconds_since_input) else {
        return HostActivitySnapshot::Unavailable;
    };

    HostActivitySnapshot::Observed {
        thermal_state,
        idle_for,
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn sample_host_activity() -> HostActivitySnapshot {
    HostActivitySnapshot::Unsupported
}
