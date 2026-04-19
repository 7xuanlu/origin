// SPDX-License-Identifier: AGPL-3.0-only
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Buffers memory access events and flushes to DB every 60 seconds.
/// Keeps the search read path free of write contention.
///
/// `Clone` gives you another handle to the same underlying buffer — the
/// `Arc<Mutex<_>>` makes this safe and cheap. Clone to hand a tracker out
/// of a `RwLock` guard without blocking writers across long operations.
#[derive(Clone)]
pub struct AccessTracker {
    buffer: Arc<Mutex<HashSet<String>>>,
}

impl Default for AccessTracker {
    fn default() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(HashSet::new())),
        }
    }
}

impl AccessTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a memory was accessed (by source_id).
    pub fn record_access(&self, source_id: &str) {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.insert(source_id.to_string());
        }
    }

    /// Record multiple accesses at once.
    pub fn record_accesses(&self, source_ids: &[String]) {
        if let Ok(mut buf) = self.buffer.lock() {
            for id in source_ids {
                buf.insert(id.clone());
            }
        }
    }

    /// Drain the buffer and return all source_ids that need flushing.
    pub fn drain(&self) -> Vec<String> {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.drain().collect()
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_drain() {
        let tracker = AccessTracker::new();
        tracker.record_access("mem_1");
        tracker.record_access("mem_2");
        tracker.record_access("mem_1"); // duplicate — should dedup

        let drained = tracker.drain();
        assert_eq!(drained.len(), 2);
        assert!(drained.contains(&"mem_1".to_string()));
        assert!(drained.contains(&"mem_2".to_string()));

        // Buffer should be empty after drain
        assert!(tracker.drain().is_empty());
    }

    #[test]
    fn test_record_accesses_batch() {
        let tracker = AccessTracker::new();
        tracker.record_accesses(&["a".into(), "b".into(), "c".into()]);
        assert_eq!(tracker.drain().len(), 3);
    }

    #[test]
    fn test_drain_empty() {
        let tracker = AccessTracker::new();
        assert!(tracker.drain().is_empty());
    }
}
