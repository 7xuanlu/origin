// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

#[derive(Debug)]
pub(crate) struct DuplicateNameGroup {
    pub(crate) lower_name: String,
    pub(crate) entity_count: i64,
}

#[derive(Debug)]
pub(crate) struct MinHashEntityInput {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) entity_type: String,
}

impl MemoryDB {
    pub(crate) async fn duplicate_name_groups_for_merge_candidates(
        &self,
    ) -> Result<Vec<DuplicateNameGroup>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT LOWER(name) as lname, COUNT(*) as cnt FROM entities
             GROUP BY LOWER(name) HAVING cnt > 1",
                (),
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("merge candidates query: {}", e)))?;

        let mut groups = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("merge candidates row: {}", e)))?
        {
            groups.push(DuplicateNameGroup {
                lower_name: row.get::<String>(0).unwrap_or_default(),
                entity_count: row.get::<i64>(1).unwrap_or(0),
            });
        }
        Ok(groups)
    }

    pub(crate) async fn entities_for_minhash_merge_candidates(
        &self,
    ) -> Result<Vec<MinHashEntityInput>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT id, name, entity_type FROM entities", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("minhash candidates scan: {e}")))?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("minhash candidates row: {e}")))?
        {
            out.push(MinHashEntityInput {
                id: row.get(0).unwrap_or_default(),
                name: row.get(1).unwrap_or_default(),
                entity_type: row.get(2).unwrap_or_default(),
            });
        }
        Ok(out)
    }
}
