// SPDX-License-Identifier: AGPL-3.0-only
//! Tauri-specific [`EventEmitter`] implementation.
//!
//! Bridges the core crate's `EventEmitter` trait to `tauri::Emitter` so that
//! core modules (MemoryDB, refinery, etc.) can push events to the Tauri
//! frontend without depending on Tauri directly.

use origin_core::events::EventEmitter;
use tauri::Emitter;

/// Wraps a `tauri::AppHandle` and forwards events to the Tauri frontend.
pub struct TauriEmitter {
    handle: tauri::AppHandle,
}

impl TauriEmitter {
    pub fn new(handle: tauri::AppHandle) -> Self {
        Self { handle }
    }
}

impl EventEmitter for TauriEmitter {
    fn emit(&self, event: &str, payload: &str) -> anyhow::Result<()> {
        self.handle
            .emit(event, payload.to_string())
            .map_err(|e| anyhow::anyhow!("Tauri emit failed: {e}"))
    }
}
