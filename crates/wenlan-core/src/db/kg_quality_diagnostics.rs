// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

#[derive(Debug)]
pub(crate) struct ContradictionObservationCount {
    pub(crate) entity_name: String,
    pub(crate) observation_count: i64,
}

impl MemoryDB {
    /// Count live `relates` edges whose payload `source_memory_id` no longer
    /// resolves to a memory row. Reads `edges` (the sole live producer of
    /// relations since G6 Stage 2 PR 2b), not the frozen legacy `relations`
    /// table.
    pub(crate) async fn count_stale_relation_sources(&self) -> Result<usize, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM edges
                 WHERE edge_type = 'relates' AND valid_until IS NULL
                 AND json_extract(payload, '$.source_memory_id') IS NOT NULL
                 AND json_extract(payload, '$.source_memory_id') NOT IN (SELECT DISTINCT source_id FROM memories WHERE source_id IS NOT NULL)",
                (),
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("stale relations: {}", e)))?;

        let count: i64 = match rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("stale relations row: {}", e)))?
        {
            Some(row) => row.get::<i64>(0).unwrap_or(0),
            None => 0,
        };
        Ok(count as usize)
    }

    pub(crate) async fn list_contradiction_observation_counts(
        &self,
    ) -> Result<Vec<ContradictionObservationCount>, WenlanError> {
        // G6 Stage 1.5a: entity-name hydration moved onto the `kind='entity'`
        // shadow page via `entity_page_map`; the observations content read
        // itself is unchanged (out of scope -- see spec target #17).
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT p.title, COUNT(o.id) as obs_count
                 FROM entity_page_map epm
                 JOIN pages p ON p.id = epm.page_id AND p.kind = 'entity' AND p.status = 'active'
                 JOIN observations o ON o.entity_id = epm.entity_id
                 GROUP BY epm.entity_id, p.title HAVING obs_count >= 10
                 ORDER BY obs_count DESC LIMIT 20",
                (),
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("contradictions scan: {}", e)))?;

        let mut counts = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("contradictions row: {}", e)))?
        {
            counts.push(ContradictionObservationCount {
                entity_name: row.get::<String>(0).unwrap_or_default(),
                observation_count: row.get::<i64>(1).unwrap_or(0),
            });
        }
        Ok(counts)
    }
}
