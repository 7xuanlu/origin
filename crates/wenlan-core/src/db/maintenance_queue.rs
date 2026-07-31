// SPDX-License-Identifier: Apache-2.0
//! Bounded presence probes over the `refinement_queue` table.

use super::MemoryDB;
use crate::error::WenlanError;

impl MemoryDB {
    /// Checks whether retro maintenance is blocked by an open review card.
    pub(crate) async fn has_pending_retro_review(&self) -> Result<bool, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT 1 FROM refinement_queue \
                 WHERE status IN ('pending', 'awaiting_review') \
                   AND action IN ('page_merge', 'page_keep_or_archive') \
                 LIMIT 1",
                (),
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("pending retro probe: {error}")))?;
        rows.next()
            .await
            .map(|row| row.is_some())
            .map_err(|error| WenlanError::VectorDb(format!("pending retro probe row: {error}")))
    }

    /// Checks whether cross-space discovery already has an open review card.
    pub(crate) async fn has_pending_cross_space_discovery(&self) -> Result<bool, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT 1 FROM refinement_queue \
                 WHERE status IN ('pending', 'awaiting_review') \
                   AND action = 'cross_space_discovery' \
                 LIMIT 1",
                (),
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("pending cross-space discovery probe: {error}"))
            })?;
        rows.next().await.map(|row| row.is_some()).map_err(|error| {
            WenlanError::VectorDb(format!("pending cross-space discovery probe row: {error}"))
        })
    }
}
