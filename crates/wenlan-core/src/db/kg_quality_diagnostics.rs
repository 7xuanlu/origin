// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

#[derive(Debug)]
pub(crate) struct ContradictionObservationCount {
    pub(crate) entity_name: String,
    pub(crate) observation_count: i64,
}

impl MemoryDB {
    pub(crate) async fn count_stale_relation_sources(&self) -> Result<usize, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM relations
                 WHERE source_memory_id IS NOT NULL
                 AND source_memory_id NOT IN (SELECT DISTINCT source_id FROM memories WHERE source_id IS NOT NULL)",
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
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT e.name, COUNT(o.id) as obs_count
                 FROM entities e JOIN observations o ON o.entity_id = e.id
                 GROUP BY e.id, e.name HAVING obs_count >= 10
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
