// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

impl MemoryDB {
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
