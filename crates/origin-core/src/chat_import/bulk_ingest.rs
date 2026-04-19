// SPDX-License-Identifier: Apache-2.0
//! Bulk ingest helper for chat imports.
//!
//! Takes a batch of `ParsedConversation`s and stores them as raw memories
//! tagged for the existing `reclassify_imports` pipeline to pick up. Does
//! NOT run classification, extraction, or distillation itself — that work
//! happens in the refinery steep that runs after ingest finishes.

use crate::chat_import::types::{ParsedConversation, Vendor};
use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::events::EventEmitter;
use chrono::{DateTime, Utc};
use std::sync::Arc;

/// Tracks the state of an in-progress (or completed) bulk chat-history import.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImportState {
    pub id: String,
    pub vendor: Vendor,
    pub source_path: String,
    pub total_conversations: Option<i64>,
    pub processed_conversations: i64,
    pub stage: ImportStage,
    pub error_message: Option<String>,
    pub started_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Discrete stages of a bulk import pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportStage {
    Parsing,
    StageA,
    StageB,
    Done,
    Error,
}

impl ImportStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Parsing => "parsing",
            Self::StageA => "stage_a",
            Self::StageB => "stage_b",
            Self::Done => "done",
            Self::Error => "error",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "parsing" => Some(Self::Parsing),
            "stage_a" => Some(Self::StageA),
            "stage_b" => Some(Self::StageB),
            "done" => Some(Self::Done),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

/// Result of a bulk import pass.
pub struct BulkImportResult {
    /// Number of conversations actually ingested (after dedup).
    pub conversations_ingested: usize,
    /// Number of conversations skipped because they already existed.
    pub conversations_skipped_existing: usize,
    /// Number of individual memories stored.
    pub memories_stored: usize,
}

/// Store all memories from a batch of parsed conversations, skipping any
/// conversation whose `external_id` is already in the database.
///
/// Each memory is stored with `source = 'memory'`, `source_id` set to the
/// conversation-level import key (e.g. `import_claude_{conv_external_id}`),
/// and `memory_type = NULL`. All messages in the same conversation share
/// the same `source_id`, with `chunk_index` distinguishing ordinal position.
///
/// This matches the existing `import_%` pattern that
/// `refinery::reclassify_imports` picks up automatically.
pub async fn bulk_import_conversations(
    db: Arc<MemoryDB>,
    batch: &[ParsedConversation],
    emitter: Arc<dyn EventEmitter>,
    import_id: &str,
) -> Result<BulkImportResult, OriginError> {
    if batch.is_empty() {
        return Ok(BulkImportResult {
            conversations_ingested: 0,
            conversations_skipped_existing: 0,
            memories_stored: 0,
        });
    }

    // 1. Compute candidate source_ids for dedup.
    let candidates: Vec<String> = batch
        .iter()
        .map(|c| c.vendor.build_source_id(&c.external_id))
        .collect();
    let existing = db.check_existing_import_source_ids(&candidates).await?;

    // 2. Filter to new conversations only.
    let new_conversations: Vec<&ParsedConversation> = batch
        .iter()
        .filter(|c| {
            let sid = c.vendor.build_source_id(&c.external_id);
            !existing.contains(&sid)
        })
        .collect();

    let skipped = batch.len() - new_conversations.len();

    // 3. Collect all memories into a batch for transactional insert.
    //    Emit progress events every 10 memories accumulated (conversation boundary).
    let total_estimate: usize = new_conversations.iter().map(|c| c.messages.len()).sum();
    #[allow(clippy::type_complexity)]
    let mut entries: Vec<(
        String,
        String,
        Option<String>,
        Option<chrono::DateTime<chrono::Utc>>,
        i64,
    )> = Vec::new();
    let mut last_emitted_count: usize = 0;

    for conv in &new_conversations {
        let source_id = conv.vendor.build_source_id(&conv.external_id);
        for (ordinal, msg) in conv.messages.iter().enumerate() {
            entries.push((
                source_id.clone(),
                msg.content.clone(),
                conv.title.clone(),
                msg.created_at,
                ordinal as i64,
            ));
        }

        // Emit progress if we crossed a multiple-of-10 boundary since last emission.
        let current_count = entries.len();
        if current_count / 10 > last_emitted_count / 10 {
            let payload = serde_json::json!({
                "import_id": import_id,
                "stage": "stage_a",
                "memories_processed": current_count,
                "memories_total": total_estimate,
                "entity_counts": {"people": 0, "projects": 0, "concepts": 0, "decisions": 0, "tools": 0},
                "concepts_written": 0,
                "concepts_total": 0,
                "latest_concept_titles": []
            });
            let _ = emitter.emit("chat-import-progress", &payload.to_string());
            last_emitted_count = current_count;
        }
    }

    let memories_stored = db.store_raw_import_memories_batch(&entries).await?;

    // Final progress emission — always fire so the frontend sees the
    // definitive 100% signal regardless of whether the count landed on an
    // exact multiple of 10 during the loop above.
    if !entries.is_empty() {
        let payload = serde_json::json!({
            "import_id": import_id,
            "stage": "stage_a",
            "memories_processed": memories_stored,
            "memories_total": total_estimate,
            "entity_counts": {"people": 0, "projects": 0, "concepts": 0, "decisions": 0, "tools": 0},
            "concepts_written": 0,
            "concepts_total": 0,
            "latest_concept_titles": []
        });
        let _ = emitter.emit("chat-import-progress", &payload.to_string());
    }

    Ok(BulkImportResult {
        conversations_ingested: new_conversations.len(),
        conversations_skipped_existing: skipped,
        memories_stored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_import::types::{MessageRole, ParsedMessage, Vendor};
    use crate::events::NoopEmitter;

    fn make_convo(external_id: &str, contents: &[&str]) -> ParsedConversation {
        ParsedConversation {
            external_id: external_id.into(),
            vendor: Vendor::Claude,
            title: Some("Test".into()),
            created_at: None,
            summary: None,
            messages: contents
                .iter()
                .map(|c| ParsedMessage {
                    role: MessageRole::Assistant,
                    content: c.to_string(),
                    created_at: None,
                })
                .collect(),
        }
    }

    fn noop_emitter() -> Arc<dyn EventEmitter> {
        Arc::new(NoopEmitter)
    }

    #[tokio::test]
    async fn bulk_import_stores_all_memories() {
        let (db, _tmp) = crate::db::tests::test_db().await;
        let db_arc = Arc::new(db);

        let batch = vec![
            make_convo("c1", &["m1", "m2", "m3"]),
            make_convo("c2", &["m4"]),
        ];
        let result = bulk_import_conversations(db_arc.clone(), &batch, noop_emitter(), "imp_test")
            .await
            .unwrap();
        assert_eq!(result.conversations_ingested, 2);
        assert_eq!(result.conversations_skipped_existing, 0);
        assert_eq!(result.memories_stored, 4);
    }

    #[tokio::test]
    async fn bulk_import_skips_existing_conversations() {
        let (db, _tmp) = crate::db::tests::test_db().await;
        let db_arc = Arc::new(db);

        let first = vec![make_convo("c1", &["m1"])];
        bulk_import_conversations(db_arc.clone(), &first, noop_emitter(), "imp_test")
            .await
            .unwrap();

        // Re-import the same conversation plus a new one.
        let second = vec![make_convo("c1", &["m1"]), make_convo("c2", &["m2"])];
        let result = bulk_import_conversations(db_arc.clone(), &second, noop_emitter(), "imp_test")
            .await
            .unwrap();
        assert_eq!(result.conversations_ingested, 1);
        assert_eq!(result.conversations_skipped_existing, 1);
        assert_eq!(result.memories_stored, 1);
    }

    #[tokio::test]
    async fn bulk_import_empty_batch_returns_zeros() {
        let (db, _tmp) = crate::db::tests::test_db().await;
        let db_arc = Arc::new(db);

        let result = bulk_import_conversations(db_arc.clone(), &[], noop_emitter(), "imp_test")
            .await
            .unwrap();
        assert_eq!(result.conversations_ingested, 0);
        assert_eq!(result.conversations_skipped_existing, 0);
        assert_eq!(result.memories_stored, 0);
    }

    #[tokio::test]
    async fn import_state_roundtrip() {
        use super::ImportStage;

        let (db, _tmp) = crate::db::tests::test_db().await;
        let db_arc = Arc::new(db);

        let id = "imp_xyz".to_string();
        db_arc
            .start_import_state(&id, Vendor::Claude, "/tmp/test.zip")
            .await
            .unwrap();

        let loaded = db_arc.load_import_state(&id).await.unwrap().expect("row");
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.vendor, Vendor::Claude);
        assert_eq!(loaded.stage, ImportStage::Parsing);

        db_arc
            .update_import_state_stage(&id, ImportStage::StageA, Some(42), Some(5))
            .await
            .unwrap();
        let loaded = db_arc.load_import_state(&id).await.unwrap().expect("row");
        assert_eq!(loaded.stage, ImportStage::StageA);
        assert_eq!(loaded.total_conversations, Some(42));
        assert_eq!(loaded.processed_conversations, 5);

        db_arc
            .update_import_state_stage(&id, ImportStage::Done, None, None)
            .await
            .unwrap();
        let loaded = db_arc.load_import_state(&id).await.unwrap().expect("row");
        assert_eq!(loaded.stage, ImportStage::Done);
        // total_conversations should be preserved from prior update
        assert_eq!(loaded.total_conversations, Some(42));
    }
}
