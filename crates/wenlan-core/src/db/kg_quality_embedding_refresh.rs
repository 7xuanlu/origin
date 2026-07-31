// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

#[derive(Debug)]
pub(crate) struct StaleEntityEmbeddingCandidate {
    pub(crate) entity_id: String,
    pub(crate) entity_name: String,
}

impl MemoryDB {
    pub(crate) async fn stale_entity_embedding_candidates_for_refresh(
        &self,
    ) -> Result<Vec<StaleEntityEmbeddingCandidate>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT e.id, e.name
                 FROM entities e
                 LEFT JOIN observations o ON o.entity_id = e.id
                    AND o.created_at > COALESCE(e.embedding_updated_at, 0)
                 GROUP BY e.id, e.name
                 HAVING COUNT(o.id) >= 5",
                (),
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("stale entity query: {}", e)))?;

        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("stale entity row: {}", e)))?
        {
            out.push(StaleEntityEmbeddingCandidate {
                entity_id: row.get::<String>(0).unwrap_or_default(),
                entity_name: row.get::<String>(1).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    pub(crate) async fn recent_entity_observation_contents_for_embedding_refresh(
        &self,
        entity_id: &str,
    ) -> Result<Vec<String>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut obs_rows = conn
            .query(
                "SELECT content FROM observations WHERE entity_id = ?1 ORDER BY created_at DESC LIMIT 10",
                libsql::params![entity_id.to_string()],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("obs fetch: {}", e)))?;
        let mut contents = Vec::new();
        while let Some(row) = obs_rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("obs row: {}", e)))?
        {
            contents.push(row.get::<String>(0).unwrap_or_default());
        }
        Ok(contents)
    }
}
