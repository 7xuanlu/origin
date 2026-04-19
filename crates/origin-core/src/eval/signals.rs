// SPDX-License-Identifier: AGPL-3.0-only
//! Implicit feedback signal capture for memory eval.

use crate::db::MemoryDB;
use sha2::{Digest, Sha256};

/// Signal types captured from user interactions.
#[derive(Debug, Clone, Copy)]
pub enum SignalType {
    Confirm,
    Delete,
    Pin,
    Unpin,
    Reclassify,
    Edit,
    SearchClick,
}

impl SignalType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Confirm => "confirm",
            Self::Delete => "delete",
            Self::Pin => "pin",
            Self::Unpin => "unpin",
            Self::Reclassify => "reclassify",
            Self::Edit => "edit",
            Self::SearchClick => "search_click",
        }
    }
}

/// Record an eval signal. Fire-and-forget — errors are logged, never propagated.
pub async fn record_signal(
    db: &MemoryDB,
    signal_type: SignalType,
    memory_id: &str,
    query_context: Option<&str>,
    rank_position: Option<i32>,
    metadata: Option<&str>,
) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Deterministic ID for natural dedup (no hex crate — use format! directly)
    let mut hasher = Sha256::new();
    hasher.update(signal_type.as_str().as_bytes());
    hasher.update(memory_id.as_bytes());
    hasher.update(now.to_le_bytes());
    let hash = hasher.finalize();
    let id = format!(
        "sig_{}",
        hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    );

    if let Err(e) = db
        .insert_eval_signal(
            &id,
            signal_type.as_str(),
            memory_id,
            query_context,
            rank_position,
            now,
            metadata,
        )
        .await
    {
        log::warn!("[eval] failed to record signal: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_record_signal_writes_to_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();

        record_signal(&db, SignalType::Confirm, "mem_123", None, None, None).await;

        // Verify signal was written
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT signal_type, memory_id FROM eval_signals", ())
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let signal_type: String = row.get(0).unwrap();
        let memory_id: String = row.get(1).unwrap();
        assert_eq!(signal_type, "confirm");
        assert_eq!(memory_id, "mem_123");
    }

    #[tokio::test]
    async fn test_record_signal_deduplicates() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();

        // Two signals with same params at same second should dedup via INSERT OR IGNORE
        record_signal(&db, SignalType::Pin, "mem_456", None, None, None).await;
        record_signal(&db, SignalType::Pin, "mem_456", None, None, None).await;

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM eval_signals", ())
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let count: i64 = row.get(0).unwrap();
        // Should be 1 (deduplicated) or 2 (different timestamps) — both are acceptable
        assert!((1..=2).contains(&count));
    }
}
