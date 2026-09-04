// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;
use std::collections::HashMap;

pub(crate) struct PendingMemoryRevisionPayload {
    pub(crate) revision_id: String,
    pub(crate) supersedes: String,
    /// One chunk of the staged text, exactly as `memories` stores it. This is
    /// what a memory-target revision accept has always used, and it stays that
    /// way: for a memory card `source_text` is the pre-distillation prose, not
    /// the body an accept writes.
    pub(crate) content: String,
    /// The whole staged body, for a card whose `content` is only part of it:
    /// several chunk rows and a `source_text` to recover the rest from. `None`
    /// whenever this row's own `content` is already the whole body, so a
    /// caller can take `full_body` when it is there and `content` otherwise.
    ///
    /// A page card needs this and `content` cannot supply it: past the
    /// chunker's budget the staged body becomes several rows and no single one
    /// holds all of it (issue #650).
    ///
    /// Two things make the raw column wrong to hand straight to a caller, and
    /// both are silent:
    ///
    /// - `upsert_documents` redacts `content` before chunking (`redact_pii`)
    ///   and grounds relative dates in it, but copies `source_text` in
    ///   verbatim, so the raw column would write PII into a page and into its
    ///   exported markdown. `rehydrate_staged_body` re-applies both.
    /// - `apply_memory_update` rewrites chunk zero's `content` and deletes the
    ///   rest, never touching `source_text`. An edited card is therefore one
    ///   row whose `source_text` is the body before the edit, so `full_body`
    ///   is `None` there and the edit is what an accept writes.
    pub(crate) full_body: Option<String>,
    pub(crate) structured_fields: Option<String>,
}

/// Re-applies the write-time text transforms that `upsert_documents` runs on
/// `content` before chunking, so a body recovered from `source_text` matches
/// what the chunk rows hold.
///
/// Kept beside the read rather than at the call site: every consumer of a
/// recovered body needs both transforms, and a caller that forgot the first
/// one would publish PII.
///
/// `observed_at` must be the row's `created_at`, not its `last_modified`.
/// `upsert_documents` grounds against the document's `last_modified` and stores
/// that same value in both columns, so at insert the two agree. They stop
/// agreeing afterwards: `apply_memory_update` bumps `last_modified` on every
/// call, including a metadata-only one that changes no text and deletes no
/// sibling chunks -- confirming a card, or moving it to another space, is
/// enough. Anchoring on `last_modified` would then ground "yesterday" against
/// the date of that unrelated edit instead of the date the card was staged.
pub(super) fn rehydrate_staged_body(raw: &str, observed_at: i64) -> String {
    let redacted = crate::privacy::redact_pii(raw);
    if crate::db::temporal_grounding_enabled() {
        let anchor =
            chrono::DateTime::from_timestamp(observed_at, 0).unwrap_or_else(chrono::Utc::now);
        crate::temporal_query::ground_relative_dates(&redacted, anchor)
    } else {
        redacted
    }
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
    /// Only the `chunk_index = 0` row is eligible. When sibling chunks exist
    /// its `content` is a fragment, so the whole staged body comes back
    /// alongside it as `full_body`; with no siblings `content` is already the
    /// whole body and `full_body` is `None`.
    ///
    /// A staged card goes through the ordinary document upsert, so a body past
    /// the chunker's size budget becomes several `memories` rows sharing one
    /// `source_id` and one `last_modified`. The ordering below cannot break
    /// that tie, so an unrestricted `LIMIT 1` returned an arbitrary fragment,
    /// and the page accept path wrote that fragment as a page's complete new
    /// body -- 21,038 characters down to 1,814 on a real page (issue #650).
    /// Pinning chunk zero makes the row deterministic, which is what the
    /// `full_body`-is-`None` fallback rests on, and matches
    /// `list_pending_revisions_scoped`, so the queue a human reviews and the
    /// body an accept writes come from one row.
    ///
    /// The budget that decides whether a card chunks at all is not one number:
    /// the markdown splitter caps a chunk at 1,500 characters, the plain-text
    /// splitter at 512, and with a tokenizer loaded the cap is 510 BGE tokens.
    /// A few hundred characters of unstructured prose is already enough.
    pub(crate) async fn pending_memory_revision_payload(
        &self,
        id: &str,
    ) -> Result<Option<PendingMemoryRevisionPayload>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, supersedes, content, structured_fields, source_text, \
                        COALESCE(created_at, last_modified), \
                        EXISTS (SELECT 1 FROM memories sibling \
                                WHERE sibling.source_id = memories.source_id \
                                  AND sibling.source = 'memory' \
                                  AND sibling.chunk_index > 0) \
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
            full_body: (row.get::<i64>(6).unwrap_or(0) != 0)
                .then(|| row.get::<Option<String>>(4).unwrap_or(None))
                .flatten()
                .map(|raw| rehydrate_staged_body(&raw, row.get::<i64>(5).unwrap_or(0))),
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
