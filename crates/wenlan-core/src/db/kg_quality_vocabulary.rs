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
            // NULL and empty-string are distinct signals -- NULL means no type
            // was ever recorded (not a vocabulary value), empty string means a
            // row explicitly stored one. Collapsing NULL into "" via
            // `unwrap_or_default` would misreport an absent type as the
            // vocabulary heal's "keep empty" case, so skip NULL rather than
            // coerce it.
            if let Some(t) = row.get::<Option<String>>(0).unwrap_or(None) {
                types.push(t);
            }
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
            // `entities.entity_type` is `TEXT NOT NULL` -- unlike relates
            // edges' `semantic_type`, a NULL row is impossible by schema, so
            // `unwrap_or_default` never coerces an absent value; it only
            // ever hands back the stored string.
            types.push(row.get::<String>(0).unwrap_or_default());
        }
        Ok(types)
    }
}
