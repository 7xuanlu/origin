// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

#[derive(Debug)]
pub(crate) struct StaleEntityEmbeddingCandidate {
    pub(crate) entity_id: String,
    pub(crate) entity_name: String,
}

impl MemoryDB {
    /// G6 Stage 1.5b Part 3: reads the entity's `kind='entity'` shadow page
    /// via `entity_page_map`, unconditional hard cutover (same program
    /// contract as `MemoryDB::list_entities`). `embedding_updated_at` is
    /// mirrored 1:1 onto `pages` by
    /// `insert_entity_shadow_page`/`update_entity_shadow_page`.
    pub(crate) async fn stale_entity_embedding_candidates_for_refresh(
        &self,
    ) -> Result<Vec<StaleEntityEmbeddingCandidate>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT epm.entity_id, p.title
                 FROM entity_page_map epm
                 JOIN pages p ON p.id = epm.page_id
                    AND p.kind = 'entity' AND p.status = 'active'
                 LEFT JOIN observations o ON o.entity_id = epm.entity_id
                    AND o.created_at > COALESCE(p.embedding_updated_at, 0)
                 GROUP BY epm.entity_id, p.title
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
