// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;
use std::str::FromStr;

impl MemoryDB {
    /// Record a milestone. Returns `Some(record)` if newly fired, `None` if
    /// already fired. Race-safe against concurrent callers via
    /// `ON CONFLICT ... DO NOTHING RETURNING` — two simultaneous
    /// evaluator checks cannot both fire the same milestone.
    pub async fn record_milestone(
        &self,
        id: crate::onboarding::MilestoneId,
        payload: Option<serde_json::Value>,
    ) -> Result<Option<crate::onboarding::MilestoneRecord>, WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let payload_str =
            match &payload {
                Some(v) => Some(serde_json::to_string(v).map_err(|e| {
                    WenlanError::Generic(format!("record_milestone payload: {}", e))
                })?),
                None => None,
            };
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "INSERT INTO onboarding_milestones (id, first_triggered_at, payload) \
                 VALUES (?1, ?2, ?3) \
                 ON CONFLICT(id) DO NOTHING \
                 RETURNING id, first_triggered_at, acknowledged_at, payload",
                libsql::params![id.as_str(), now, payload_str],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("record_milestone insert: {}", e)))?;
        match rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("record_milestone next: {}", e)))?
        {
            Some(row) => {
                let id_str: String = row.get(0).map_err(|e| {
                    WenlanError::VectorDb(format!("record_milestone row.id: {}", e))
                })?;
                let parsed_id = crate::onboarding::MilestoneId::from_str(&id_str)
                    .map_err(WenlanError::Generic)?;
                let first_triggered_at: i64 = row.get(1).map_err(|e| {
                    WenlanError::VectorDb(format!("record_milestone row.first_triggered_at: {}", e))
                })?;
                let acknowledged_at: Option<i64> = row.get(2).map_err(|e| {
                    WenlanError::VectorDb(format!("record_milestone row.acknowledged_at: {}", e))
                })?;
                let payload_str_out: Option<String> = row.get(3).map_err(|e| {
                    WenlanError::VectorDb(format!("record_milestone row.payload: {}", e))
                })?;
                let payload_val = payload_str_out.and_then(|s| serde_json::from_str(&s).ok());
                Ok(Some(crate::onboarding::MilestoneRecord {
                    id: parsed_id,
                    first_triggered_at,
                    acknowledged_at,
                    payload: payload_val,
                }))
            }
            None => Ok(None),
        }
    }

    /// Return all milestone rows ordered by trigger time ascending. Called
    /// by the `/api/onboarding/milestones` endpoint and the cold-start
    /// toast replay logic in `MilestoneToaster`.
    pub async fn list_milestones(
        &self,
    ) -> Result<Vec<crate::onboarding::MilestoneRecord>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, first_triggered_at, acknowledged_at, payload \
                 FROM onboarding_milestones ORDER BY first_triggered_at ASC",
                (),
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("list_milestones: {}", e)))?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("list_milestones next: {}", e)))?
        {
            let id_str: String = row
                .get(0)
                .map_err(|e| WenlanError::VectorDb(format!("list_milestones row.id: {}", e)))?;
            let parsed_id = match crate::onboarding::MilestoneId::from_str(&id_str) {
                Ok(id) => id,
                Err(e) => {
                    log::warn!(
                        "list_milestones: skipping unknown milestone id '{}': {}",
                        id_str,
                        e
                    );
                    continue;
                }
            };
            let first_triggered_at: i64 = row.get(1).map_err(|e| {
                WenlanError::VectorDb(format!("list_milestones row.first_triggered_at: {}", e))
            })?;
            let acknowledged_at: Option<i64> = row.get(2).map_err(|e| {
                WenlanError::VectorDb(format!("list_milestones row.acknowledged_at: {}", e))
            })?;
            let payload_str: Option<String> = row.get(3).map_err(|e| {
                WenlanError::VectorDb(format!("list_milestones row.payload: {}", e))
            })?;
            let payload_val = payload_str.and_then(|s| serde_json::from_str(&s).ok());
            out.push(crate::onboarding::MilestoneRecord {
                id: parsed_id,
                first_triggered_at,
                acknowledged_at,
                payload: payload_val,
            });
        }
        Ok(out)
    }

    /// Set `acknowledged_at = now()` for the given milestone, but only if
    /// it has not already been acknowledged (so we don't overwrite the
    /// original ack timestamp on repeated clicks).
    pub async fn acknowledge_milestone(
        &self,
        id: crate::onboarding::MilestoneId,
    ) -> Result<(), WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE onboarding_milestones SET acknowledged_at = ?1 \
             WHERE id = ?2 AND acknowledged_at IS NULL",
            libsql::params![now, id.as_str()],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("acknowledge_milestone: {}", e)))?;
        Ok(())
    }

    /// Update the `payload.shown_count` counter for a milestone. Used by
    /// `FirstConceptModal` to track non-acknowledging dismissals so the
    /// modal self-retires after 3 shows (see spec §3.2). Returns the new
    /// count. If the milestone is not yet recorded, this is a no-op and
    /// returns 0.
    pub async fn increment_milestone_shown_count(
        &self,
        id: crate::onboarding::MilestoneId,
    ) -> Result<u32, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT payload FROM onboarding_milestones WHERE id = ?1",
                libsql::params![id.as_str()],
            )
            .await
            .map_err(|e| {
                WenlanError::VectorDb(format!("increment_milestone_shown_count select: {}", e))
            })?;
        let current: Option<String> = match rows.next().await.map_err(|e| {
            WenlanError::VectorDb(format!("increment_milestone_shown_count next: {}", e))
        })? {
            Some(r) => r.get(0).map_err(|e| {
                WenlanError::VectorDb(format!("increment_milestone_shown_count row: {}", e))
            })?,
            None => return Ok(0),
        };
        let mut payload_val: serde_json::Value = current
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_else(|| serde_json::json!({}));
        let count = payload_val
            .get("shown_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
            + 1;
        payload_val["shown_count"] = serde_json::json!(count);
        let new_str = serde_json::to_string(&payload_val).map_err(|e| {
            WenlanError::Generic(format!("increment_milestone_shown_count serialize: {}", e))
        })?;
        conn.execute(
            "UPDATE onboarding_milestones SET payload = ?1 WHERE id = ?2",
            libsql::params![new_str, id.as_str()],
        )
        .await
        .map_err(|e| {
            WenlanError::VectorDb(format!("increment_milestone_shown_count update: {}", e))
        })?;
        Ok(count)
    }

    /// Clear all milestone rows. Dev/demo-only — exposed via
    /// `POST /api/onboarding/reset` and gated to `import.meta.env.DEV` on
    /// the frontend side.
    pub async fn reset_onboarding_milestones(&self) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM onboarding_milestones", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("reset_onboarding_milestones: {}", e)))?;
        Ok(())
    }
}
