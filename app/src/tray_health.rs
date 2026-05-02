// SPDX-License-Identifier: AGPL-3.0-only
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{image::Image, AppHandle, Emitter};

const ACTIVE_1X: &[u8] = include_bytes!("../icons/tray-icon.png");
const DIM_1X: &[u8] = include_bytes!("../icons/tray-icon-dim.png");

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

/// Spawn the poll loop. Returns a HealthSignal that the tray menu can read.
pub fn spawn_poller(app_handle: AppHandle) -> HealthSignal {
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

        loop {
            let result = client.get("http://127.0.0.1:7878/api/health").send().await;

            let new_state = match result {
                Ok(r) if r.status().is_success() => DaemonState::Up,
                _ => DaemonState::Down,
            };

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
}
