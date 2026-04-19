// SPDX-License-Identifier: Apache-2.0
//! Event emission trait — replaces `Option<tauri::AppHandle>` in MemoryDB.
//!
//! Core modules emit events through this abstraction so the core crate stays
//! free of Tauri or other UI-layer dependencies. The Tauri app supplies its
//! own implementation that wraps `tauri::Emitter`; tests and headless
//! operation use `NoopEmitter`.

use anyhow::Result;

/// Trait for emitting events from core to external consumers.
///
/// Replaces `Option<tauri::AppHandle>` in MemoryDB and other modules that
/// previously needed to push updates to the Tauri frontend.
pub trait EventEmitter: Send + Sync {
    /// Emit a named event with a string payload (typically JSON).
    fn emit(&self, event: &str, payload: &str) -> Result<()>;
}

/// No-op emitter for testing and headless operation.
///
/// Always returns `Ok(())` without doing anything.
pub struct NoopEmitter;

impl EventEmitter for NoopEmitter {
    fn emit(&self, _event: &str, _payload: &str) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_emitter_always_ok() {
        let emitter = NoopEmitter;
        assert!(emitter.emit("test-event", "{}").is_ok());
        assert!(emitter.emit("", "").is_ok());
    }

    #[test]
    fn noop_emitter_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoopEmitter>();
    }
}
