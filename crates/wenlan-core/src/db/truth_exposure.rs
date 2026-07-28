// SPDX-License-Identifier: Apache-2.0
//! The durable half of the M5 exposure contract: the marker audit log, and the
//! cutover generation that decides whether any of it is live yet.
//!
//! Two small things, both here because they are properties of the database
//! rather than of a request.
//!
//! **The audit log** is the *only* compensating control for the one attack D3
//! concedes. `Collection` and `NamedPage` compose: a caller willing to forge the
//! intent marker can list unsupported page IDs and then fetch them one at a
//! time. Nothing at cooperative tier prevents that -- the daemon is loopback and
//! unauthenticated, and cannot tell a forged gesture from a real one. What it
//! can do is leave a trail, so page-at-a-time extraction is a visible pattern
//! instead of an invisible one. An untested audit row is a sentence, not a
//! control, which is why `truth_guard` has a test that goes RED when the write
//! is removed.
//!
//! **The cutover generation** is why PR-B changes no behavior. Every adapter is
//! installed and mutation-tested against a generation the tests advance
//! themselves; in production it stays 0 and every adapter is pass-through. PR-C
//! advances it once, through the fenced ceremony -- the same shape as the M4
//! reader cutover, deliberately.

use super::MemoryDB;
use crate::truth_contract::{visible_at, MarkerOutcome, TruthGrant, Visibility};
use crate::WenlanError;
use std::collections::HashMap;

/// `app_metadata` key holding the durable cutover generation. Absent or `0`
/// means every truth adapter is inert.
pub const TRUTH_CUTOVER_GENERATION_KEY: &str = "truth_cutover_generation";

/// The two truth axes for one page.
///
/// Independent by construction: neither is inferred from the other. A page can
/// be machine-supported and unreviewed, or human-reviewed and unsupported, and
/// both facts have to reach the reader for a `collection` entry to be legal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageTruth {
    pub supported: bool,
    pub human_reviewed: bool,
}

/// One recorded marked call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruthMarkerAudit {
    pub marked_at: i64,
    /// Who called, from `x-agent-name`. `unknown` when the caller did not say --
    /// honest-unknown beats guessing, same as the rest of agent attribution.
    pub caller: String,
    pub method: String,
    /// The route *template*, not the concrete URI: `/api/pages/{id}`. The IDs
    /// live in `page_ids`, where they can be counted.
    pub path: String,
    pub outcome: String,
    /// The page IDs this call named, in path order.
    pub page_ids: Vec<String>,
}

impl MemoryDB {
    /// Migration 101 DDL. Additive and idempotent.
    pub(super) async fn ensure_truth_exposure_tables(
        tx: &libsql::Transaction,
    ) -> Result<(), WenlanError> {
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS truth_marker_audit (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                marked_at  INTEGER NOT NULL,
                caller     TEXT    NOT NULL,
                method     TEXT    NOT NULL,
                path       TEXT    NOT NULL,
                outcome    TEXT    NOT NULL CHECK (outcome IN (
                               'refused', 'automatic',
                               'granted_collection', 'granted_named_page')),
                -- JSON array, empty for a collection call. Stored as text
                -- because the interesting query is 'which IDs did this caller
                -- name over the last hour', which reads the whole row anyway.
                page_ids   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_truth_marker_audit_recent
                ON truth_marker_audit(marked_at DESC);
            CREATE INDEX IF NOT EXISTS idx_truth_marker_audit_caller
                ON truth_marker_audit(caller, marked_at DESC);
            ",
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m101 truth exposure tables: {error}")))?;
        Ok(())
    }

    /// Record one marked call.
    ///
    /// Called for every request carrying the intent marker, granted or refused.
    /// A refusal is the signature of a misconfigured integration and costs one
    /// row to keep.
    pub async fn record_truth_marker(
        &self,
        caller: &str,
        method: &str,
        path: &str,
        outcome: MarkerOutcome,
        page_ids: &[String],
    ) -> Result<(), WenlanError> {
        let ids = serde_json::to_string(page_ids)
            .map_err(|e| WenlanError::VectorDb(format!("record_truth_marker page_ids: {e}")))?;
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO truth_marker_audit (marked_at, caller, method, path, outcome, page_ids)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            libsql::params![
                chrono::Utc::now().timestamp(),
                caller,
                method,
                path,
                outcome.as_str(),
                ids
            ],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("record_truth_marker: {e}")))?;
        Ok(())
    }

    /// The most recent marked calls, newest first.
    pub async fn recent_truth_markers(
        &self,
        limit: usize,
    ) -> Result<Vec<TruthMarkerAudit>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT marked_at, caller, method, path, outcome, page_ids
                   FROM truth_marker_audit
                  ORDER BY marked_at DESC, id DESC
                  LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("recent_truth_markers: {e}")))?;

        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("recent_truth_markers next: {e}")))?
        {
            let ids: String = row.get(5).unwrap_or_default();
            out.push(TruthMarkerAudit {
                marked_at: row.get(0).unwrap_or_default(),
                caller: row.get(1).unwrap_or_default(),
                method: row.get(2).unwrap_or_default(),
                path: row.get(3).unwrap_or_default(),
                outcome: row.get(4).unwrap_or_default(),
                page_ids: serde_json::from_str(&ids).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// The durable cutover generation. `0` -- the production value throughout
    /// PR-B -- means every truth adapter is pass-through.
    ///
    /// ponytail: one indexed `app_metadata` read per request. Cache it on
    /// `MemoryDB` behind an atomic if it ever shows up in a profile; a
    /// single-user loopback daemon is nowhere near needing that today.
    pub async fn truth_cutover_generation(&self) -> Result<i64, WenlanError> {
        Ok(self
            .get_app_metadata(TRUTH_CUTOVER_GENERATION_KEY)
            .await?
            .and_then(|raw| raw.trim().parse::<i64>().ok())
            .unwrap_or(0))
    }

    /// Both truth axes for a batch of pages, in one query.
    ///
    /// A page with no `page_truth_state` row reads as unsupported and
    /// unreviewed. The absence of a support record is not evidence of support --
    /// and post-migration that absence is the normal case, not an anomaly.
    pub async fn page_truth_states(
        &self,
        page_ids: &[String],
    ) -> Result<HashMap<String, PageTruth>, WenlanError> {
        let mut out = HashMap::new();
        if page_ids.is_empty() {
            return Ok(out);
        }
        let conn = self.conn.lock().await;
        // Chunked because SQLite caps bound parameters, and a page list comes
        // from a response payload whose length nobody here controls.
        for chunk in page_ids.chunks(400) {
            let placeholders = (1..=chunk.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT page_id, support_status, human_reviewed
                   FROM page_truth_state
                  WHERE page_id IN ({placeholders})"
            );
            let params = chunk
                .iter()
                .map(|id| libsql::Value::from(id.as_str()))
                .collect::<Vec<_>>();
            let mut rows = conn
                .query(&sql, params)
                .await
                .map_err(|e| WenlanError::VectorDb(format!("page_truth_states: {e}")))?;
            while let Some(row) = rows
                .next()
                .await
                .map_err(|e| WenlanError::VectorDb(format!("page_truth_states next: {e}")))?
            {
                let page_id: String = row.get(0).unwrap_or_default();
                let status: String = row.get(1).unwrap_or_default();
                let reviewed: i64 = row.get(2).unwrap_or(0);
                out.insert(
                    page_id,
                    PageTruth {
                        supported: status == "supported",
                        human_reviewed: reviewed == 1,
                    },
                );
            }
        }
        Ok(out)
    }

    /// One call's verdict for a batch of pages -- **the** entry point every
    /// adapter uses.
    ///
    /// At generation 0 this is a single `app_metadata` read and every page comes
    /// back `Full`, so an inert adapter costs one cheap lookup and never touches
    /// `page_truth_state` at all. That is what makes installing 79 of them
    /// before the cutover affordable.
    ///
    /// A page ID with no entry in the result was not asked about; adapters
    /// should treat a missing key the same as `Hidden` rather than as
    /// permission, but the map is total over `page_ids` by construction.
    pub async fn page_visibility(
        &self,
        grant: &TruthGrant,
        page_ids: &[String],
    ) -> Result<HashMap<String, Visibility>, WenlanError> {
        let generation = self.truth_cutover_generation().await?;
        if generation == 0 {
            return Ok(page_ids
                .iter()
                .map(|id| (id.clone(), Visibility::Full))
                .collect());
        }
        let states = self.page_truth_states(page_ids).await?;
        Ok(page_ids
            .iter()
            .map(|id| {
                let supported = states.get(id).is_some_and(|s| s.supported);
                (id.clone(), visible_at(generation, grant, id, supported))
            })
            .collect())
    }

    /// Advance (or roll back) the cutover generation.
    ///
    /// PR-B calls this only from tests. PR-C owns the production advance, which
    /// is a two-phase fenced ceremony this setter is merely the last step of --
    /// so nothing here should be read as permission to flip it.
    pub async fn set_truth_cutover_generation(&self, generation: i64) -> Result<(), WenlanError> {
        self.set_app_metadata(TRUTH_CUTOVER_GENERATION_KEY, &generation.to_string())
            .await
    }
}

#[cfg(test)]
#[path = "truth_exposure_test.rs"]
mod tests;
