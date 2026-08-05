// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

impl MemoryDB {
    // 2026-08-05, G6 Stage 1.2 deliberate carryover: stays on `relations`,
    // not `edges`. This is writer coupling, not drift detection -- the
    // healer's writer, `fold_relation_type` (db.rs), enumerates
    // `SELECT ... FROM relations WHERE relation_type = ?1` to do the actual
    // fold. If discovery read `edges` here while repair enumerates
    // `relations`, the healer would discover a type it then folds zero rows
    // for. Migrate this reader only alongside `fold_relation_type` itself
    // (see docs/plans/2026-08-05-g6-stage12-relations-readers-spec.md).
    pub(crate) async fn distinct_relation_types_for_vocabulary_heal(
        &self,
    ) -> Result<Vec<String>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT DISTINCT relation_type FROM relations", ())
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
