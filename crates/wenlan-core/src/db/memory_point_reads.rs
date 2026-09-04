// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;
use std::collections::HashMap;

pub(crate) struct PendingMemoryRevisionPayload {
    pub(crate) revision_id: String,
    pub(crate) supersedes: String,
    pub(crate) content: String,
    pub(crate) structured_fields: Option<String>,
}

impl MemoryDB {
    /// Checks an active, non-episode row by its physical `memories.id`.
    ///
    /// Error prefixes intentionally retain their former caller context so this
    /// movement-only extraction does not change observable diagnostics.
    pub(crate) async fn has_active_non_episode_memory_id(
        &self,
        id: &str,
    ) -> Result<bool, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT 1 FROM memories WHERE id = ?1 AND pending_revision = 0 AND source != 'episode' LIMIT 1",
                libsql::params![id],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("source page chunk lookup: {e}")))?;
        rows.next()
            .await
            .map(|row| row.is_some())
            .map_err(|e| WenlanError::VectorDb(format!("source page chunk lookup row: {e}")))
    }

    /// Returns the pending memory revision selected for a physical memory id.
    ///
    /// An exact `source_id` match wins over the legacy `supersedes` fallback;
    /// otherwise the newest eligible legacy row wins.
    ///
    /// The returned `content` comes from `source_text`, and only the
    /// `chunk_index = 0` row is eligible. Both halves matter. A staged
    /// card is written through the ordinary document upsert, so a body over
    /// roughly 1,500 characters becomes several `memories` rows sharing one
    /// `source_id` and one `last_modified` -- the ordering above cannot break
    /// that tie, so the old unrestricted `LIMIT 1` returned an arbitrary
    /// fragment. The accept path writes what this read returns as the page's
    /// complete new body, which destroyed two real pages on 2026-08-30
    /// (21,038 characters down to 1,814, and 9,142 down to 1,428; issue
    /// #650). `content` on any single row is one chunk by construction;
    /// `source_text` carries the whole staged body on every chunk row,
    /// because `stage_page_revision_card` passes the full string in both and
    /// the chunk loop copies `source_text` verbatim onto each row. Pinning
    /// chunk zero also makes the row choice deterministic and agrees with
    /// `list_pending_revisions_scoped`, which already pins it, so the queue a
    /// human reviews and the body an accept writes come from the same row.
    /// `COALESCE` keeps a row whose `source_text` is NULL readable; the only
    /// cards that predate the column also predate source-revision fencing,
    /// and `accept_page_revision_card` refuses those outright.
    pub(crate) async fn pending_memory_revision_payload(
        &self,
        id: &str,
    ) -> Result<Option<PendingMemoryRevisionPayload>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, supersedes, COALESCE(source_text, content), structured_fields \
                 FROM memories \
                 WHERE pending_revision = 1 \
                   AND source = 'memory' \
                   AND chunk_index = 0 \
                   AND (source_id = ?1 OR supersedes = ?1) \
                 ORDER BY CASE WHEN source_id = ?1 THEN 0 ELSE 1 END, last_modified DESC \
                 LIMIT 1",
                libsql::params![id.to_string()],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("resolve_page_revision_card: {e}")))?;

        let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("resolve_page_revision_card row: {e}")))?
        else {
            return Ok(None);
        };
        Ok(Some(PendingMemoryRevisionPayload {
            revision_id: row
                .get::<String>(0)
                .map_err(|e| WenlanError::VectorDb(format!("revision source_id: {e}")))?,
            supersedes: row
                .get::<String>(1)
                .map_err(|e| WenlanError::VectorDb(format!("revision supersedes: {e}")))?,
            content: row
                .get::<String>(2)
                .map_err(|e| WenlanError::VectorDb(format!("revision content: {e}")))?,
            structured_fields: row.get::<Option<String>>(3).unwrap_or(None),
        }))
    }

    /// Loads chunk-zero content hashes for memory rows keyed by source id.
    ///
    /// Missing rows are omitted and SQL `NULL` hashes remain `None`.
    pub(crate) async fn memory_content_hashes_for_source_ids(
        &self,
        source_ids: &[String],
    ) -> Result<HashMap<String, Option<String>>, WenlanError> {
        if source_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = source_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT source_id, content_hash FROM memories WHERE source = 'memory' AND chunk_index = 0 AND source_id IN ({placeholders})"
        );
        let params: Vec<libsql::Value> = source_ids
            .iter()
            .cloned()
            .map(libsql::Value::Text)
            .collect();
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(params))
            .await
            .map_err(|e| WenlanError::VectorDb(format!("distill content_hash fetch: {e}")))?;

        let mut hashes = HashMap::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("distill content_hash row: {e}")))?
        {
            let source_id = row.get::<String>(0).unwrap_or_default();
            let content_hash = row.get::<Option<String>>(1).unwrap_or(None);
            hashes.insert(source_id, content_hash);
        }
        Ok(hashes)
    }
}
