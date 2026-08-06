// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

impl MemoryDB {
    // G6 Stage 2 PR 2b sweep instance 4: ported alongside `fold_relation_type`
    // (db.rs) in the same PR, closing the writer-coupling hazard the prior
    // carryover comment (2026-08-05, G6 Stage 1.2) flagged -- the healer's
    // writer now enumerates live `relates` edges too, so discovery and repair
    // agree on the source of truth again. Distinct `relation_type` over live
    // relates edges.
    pub(crate) async fn distinct_relation_types_for_vocabulary_heal(
        &self,
    ) -> Result<Vec<String>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT DISTINCT semantic_type FROM edges \
                 WHERE edge_type = 'relates' AND valid_until IS NULL",
                (),
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("distinct rel types: {}", e)))?;
        let mut types = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("rel type row: {}", e)))?
        {
            types.push(row.get::<String>(0).unwrap_or_default());
        }
        Ok(types)
    }

    pub(crate) async fn distinct_entity_types_for_vocabulary_heal(
        &self,
    ) -> Result<Vec<String>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT DISTINCT entity_type FROM entities", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("distinct entity types: {}", e)))?;
        let mut types = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("entity type row: {}", e)))?
        {
            types.push(row.get::<String>(0).unwrap_or_default());
        }
        Ok(types)
    }
}
