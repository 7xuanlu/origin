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
    // G6 Stage 2 PR 2c sub-step 3 item 1: `entities.name` reads move to the
    // `kind='entity'` shadow page's `title` via `entity_page_map`/`pages`.
    pub(crate) async fn duplicate_name_groups_for_merge_candidates(
        &self,
    ) -> Result<Vec<DuplicateNameGroup>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT LOWER(p.title) as lname, COUNT(*) as cnt FROM entity_page_map epm
             JOIN pages p ON p.id = epm.page_id
             WHERE p.kind = 'entity' AND p.status = 'active'
             GROUP BY LOWER(p.title) HAVING cnt > 1",
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

    // G6 Stage 2 PR 2c sub-step 3 item 1: `entities.name`/`entity_type` reads
    // move to the `kind='entity'` shadow page via `entity_page_map`/`pages`.
    pub(crate) async fn entities_for_minhash_merge_candidates(
        &self,
    ) -> Result<Vec<MinHashEntityInput>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT epm.entity_id, p.title, p.entity_type FROM entity_page_map epm
                 JOIN pages p ON p.id = epm.page_id
                 WHERE p.kind = 'entity' AND p.status = 'active'",
                (),
            )
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
