// SPDX-License-Identifier: AGPL-3.0-only
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Rolling buffer retention: 15 minutes.
const DEFAULT_MAX_AGE_SECS: i64 = 900;

/// Maximum characters in a text snippet.
pub const MAX_SNIPPET_CHARS: usize = 300;

/// A single entry in the working memory buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryEntry {
    pub timestamp: i64,
    pub source: String,
    pub app_name: String,
    pub window_title: String,
    pub text_snippet: String,
    pub source_id: String,
}

/// In-memory rolling buffer of recent captures for zero-query Spotlight.
pub struct WorkingMemory {
    entries: VecDeque<WorkingMemoryEntry>,
    max_age_secs: i64,
}

impl WorkingMemory {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            max_age_secs: DEFAULT_MAX_AGE_SECS,
        }
    }

    /// Push a new entry to the front. Deduplicates by source_id (replaces existing).
    /// Prunes expired entries afterward.
    pub fn push(&mut self, entry: WorkingMemoryEntry) {
        // Remove existing entry with same source_id
        self.entries.retain(|e| e.source_id != entry.source_id);
        self.entries.push_front(entry);
        self.prune();
    }

    /// Update the timestamp of an existing entry by source_id (for dedup path).
    /// Moves the entry to the front since it's the most recent activity.
    pub fn touch(&mut self, source_id: &str, new_timestamp: i64) {
        if let Some(pos) = self.entries.iter().position(|e| e.source_id == source_id) {
            if let Some(mut entry) = self.entries.remove(pos) {
                entry.timestamp = new_timestamp;
                self.entries.push_front(entry);
            }
        }
    }

    /// Return a clone of all non-expired entries, newest first.
    /// Prunes expired entries as a side effect.
    pub fn get_recent(&mut self) -> Vec<WorkingMemoryEntry> {
        self.prune();
        self.entries.iter().cloned().collect()
    }

    /// Remove entries older than max_age_secs.
    fn prune(&mut self) {
        let cutoff = chrono::Utc::now().timestamp() - self.max_age_secs;
        self.entries.retain(|e| e.timestamp >= cutoff);
    }
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> i64 {
        chrono::Utc::now().timestamp()
    }

    fn make_entry(source_id: &str, app: &str, ts: i64) -> WorkingMemoryEntry {
        WorkingMemoryEntry {
            timestamp: ts,
            source: "ambient".to_string(),
            app_name: app.to_string(),
            window_title: format!("{} Window", app),
            text_snippet: format!("Text from {}", app),
            source_id: source_id.to_string(),
        }
    }

    #[test]
    fn test_push_and_get_recent_newest_first() {
        let mut wm = WorkingMemory::new();
        let ts = now();
        wm.push(make_entry("a", "VS Code", ts - 10));
        wm.push(make_entry("b", "Chrome", ts - 5));
        wm.push(make_entry("c", "Terminal", ts));

        let entries = wm.get_recent();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].source_id, "c"); // newest
        assert_eq!(entries[1].source_id, "b");
        assert_eq!(entries[2].source_id, "a"); // oldest
    }

    #[test]
    fn test_expired_entries_pruned() {
        let mut wm = WorkingMemory::new();
        let ts = now();
        wm.push(make_entry("old", "Old App", ts - 1000)); // >15 min ago
        wm.push(make_entry("new", "New App", ts));

        let entries = wm.get_recent();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source_id, "new");
    }

    #[test]
    fn test_touch_updates_timestamp_and_moves_to_front() {
        let mut wm = WorkingMemory::new();
        let ts = now();
        wm.push(make_entry("a", "VS Code", ts - 10));
        wm.push(make_entry("b", "Chrome", ts - 5));

        // Touch "a" to make it the most recent
        wm.touch("a", ts);

        let entries = wm.get_recent();
        assert_eq!(entries[0].source_id, "a");
        assert_eq!(entries[0].timestamp, ts);
        assert_eq!(entries[1].source_id, "b");
    }

    #[test]
    fn test_push_deduplicates_by_source_id() {
        let mut wm = WorkingMemory::new();
        let ts = now();
        wm.push(make_entry("a", "VS Code", ts - 10));
        wm.push(make_entry("b", "Chrome", ts - 5));
        wm.push(make_entry("a", "VS Code Updated", ts)); // replace "a"

        let entries = wm.get_recent();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].source_id, "a");
        assert_eq!(entries[0].app_name, "VS Code Updated"); // replaced
        assert_eq!(entries[1].source_id, "b");
    }

    #[test]
    fn test_touch_nonexistent_is_noop() {
        let mut wm = WorkingMemory::new();
        let ts = now();
        wm.push(make_entry("a", "VS Code", ts));
        wm.touch("nonexistent", ts);

        let entries = wm.get_recent();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].source_id, "a");
    }

    #[test]
    fn test_empty_get_recent() {
        let mut wm = WorkingMemory::new();
        let entries = wm.get_recent();
        assert!(entries.is_empty());
    }
}
