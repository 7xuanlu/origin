// SPDX-License-Identifier: AGPL-3.0-only
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{image::Image, AppHandle, Emitter};

const ACTIVE_1X: &[u8] = include_bytes!("../icons/tray-icon.png");
const DIM_1X: &[u8] = include_bytes!("../icons/tray-icon-dim.png");

/// Spec M2: how long to remain in `Starting` while waiting for the first
/// successful poll before transitioning to `Down`.
pub const STARTING_GRACE: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DaemonState {
    Starting = 0,
    Up = 1,
    Down = 2,
}

impl DaemonState {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Up,
            2 => Self::Down,
            _ => Self::Starting,
        }
    }
}

#[derive(Clone)]
pub struct HealthSignal {
    state: Arc<AtomicU8>,
    consecutive_down: Arc<AtomicU8>,
}

impl HealthSignal {
    pub fn new() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(DaemonState::Starting as u8)),
            consecutive_down: Arc::new(AtomicU8::new(0)),
        }
    }

    pub fn current(&self) -> DaemonState {
        DaemonState::from_u8(self.state.load(Ordering::Acquire))
    }

    #[allow(dead_code)]
    pub fn consecutive_down_count(&self) -> u8 {
        self.consecutive_down.load(Ordering::Acquire)
    }

    fn store(&self, s: DaemonState) {
        self.state.store(s as u8, Ordering::Release);
    }
}

impl Default for HealthSignal {
    fn default() -> Self {
        Self::new()
    }
}

/// Pure decision function — given the current state, the latest poll result,
/// and how long we've been running, return the next state.
///
/// Spec M2: stay `Starting` while waiting for the first successful poll OR up
/// to `STARTING_GRACE` elapsed, whichever comes first. Only transition to
/// `Down` once we've either been `Up` at least once OR the grace window has
/// passed.
pub(crate) fn next_state(
    prev: DaemonState,
    poll_ok: bool,
    elapsed_since_start: Duration,
    ever_up: bool,
) -> DaemonState {
    if poll_ok {
        return DaemonState::Up;
    }
    match prev {
        DaemonState::Starting if !ever_up && elapsed_since_start < STARTING_GRACE => {
            DaemonState::Starting
        }
        _ => DaemonState::Down,
    }
}

/// Spawn the poll loop. Returns a HealthSignal that the tray menu can read.
pub fn spawn_poller(app_handle: AppHandle) -> HealthSignal {
    spawn_poller_at(Instant::now(), app_handle)
}

/// Test-friendly entry point that allows injecting the start instant.
pub fn spawn_poller_at(start_instant: Instant, app_handle: AppHandle) -> HealthSignal {
    let signal = HealthSignal::new();
    let signal_clone = signal.clone();
    let handle = app_handle.clone();

    tauri::async_runtime::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(1500))
            .build()
            .expect("reqwest client");

        let interval = Duration::from_secs(5);
        let mut prev_state = DaemonState::Starting;
        let mut ever_up = false;

        loop {
            let result = client.get("http://127.0.0.1:7878/api/health").send().await;
            let poll_ok = matches!(&result, Ok(r) if r.status().is_success());

            let new_state = next_state(prev_state, poll_ok, start_instant.elapsed(), ever_up);

            if new_state == DaemonState::Up {
                ever_up = true;
            }

            if new_state == DaemonState::Down {
                signal_clone.consecutive_down.fetch_add(1, Ordering::AcqRel);
            } else {
                signal_clone.consecutive_down.store(0, Ordering::Release);
            }

            if new_state != prev_state {
                signal_clone.store(new_state);
                let icon_bytes = match new_state {
                    DaemonState::Up => ACTIVE_1X,
                    _ => DIM_1X,
                };
                if let Some(tray) = handle
                    .tray_by_id("main")
                    .or_else(|| handle.tray_by_id("default"))
                {
                    if let Ok(img) = Image::from_bytes(icon_bytes) {
                        let _ = tray.set_icon(Some(img));
                    }
                }
                let _ = handle.emit("tray-state-changed", new_state as u8);
                prev_state = new_state;
            }

            tokio::time::sleep(interval).await;
        }
    });

    signal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_default_is_starting() {
        let s = HealthSignal::new();
        assert_eq!(s.current(), DaemonState::Starting);
        assert_eq!(s.consecutive_down_count(), 0);
    }

    #[test]
    fn first_failed_poll_within_grace_stays_starting() {
        // Spec M2: Starting until first successful poll OR 30s elapsed.
        // A failed first poll at ~5s must NOT transition to Down.
        let next = next_state(
            DaemonState::Starting,
            false,
            Duration::from_secs(5),
            /* ever_up */ false,
        );
        assert_eq!(next, DaemonState::Starting);
    }

    #[test]
    fn first_successful_poll_transitions_to_up() {
        let next = next_state(DaemonState::Starting, true, Duration::from_secs(5), false);
        assert_eq!(next, DaemonState::Up);
    }

    #[test]
    fn failed_poll_after_grace_transitions_to_down() {
        let next = next_state(DaemonState::Starting, false, Duration::from_secs(31), false);
        assert_eq!(next, DaemonState::Down);
    }

    #[test]
    fn failed_poll_after_ever_up_transitions_to_down_immediately() {
        // Once we've been Up at least once, a failed poll → Down regardless
        // of elapsed time.
        let next = next_state(
            DaemonState::Up,
            false,
            Duration::from_secs(2),
            /* ever_up */ true,
        );
        assert_eq!(next, DaemonState::Down);
    }

    #[test]
    fn starting_stays_at_grace_boundary_minus_epsilon() {
        // < 30s should still hold Starting on a fail.
        let next = next_state(DaemonState::Starting, false, Duration::from_secs(29), false);
        assert_eq!(next, DaemonState::Starting);
    }

    #[test]
    fn down_stays_down_on_subsequent_fail() {
        let next = next_state(DaemonState::Down, false, Duration::from_secs(60), true);
        assert_eq!(next, DaemonState::Down);
    }

    #[test]
    fn down_recovers_to_up_on_successful_poll() {
        let next = next_state(DaemonState::Down, true, Duration::from_secs(60), true);
        assert_eq!(next, DaemonState::Up);
    }
}
