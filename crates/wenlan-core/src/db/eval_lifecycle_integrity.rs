// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use super::MemoryDB;
use crate::error::WenlanError;

pub(crate) struct EvalLifecycleSupersedesInput {
    pub(crate) source_id: String,
    pub(crate) supersedes: Option<String>,
}

pub(crate) struct EvalLifecycleStateCounts {
    pub(crate) memory_count: usize,
    pub(crate) archived_count: usize,
    pub(crate) entity_count: usize,
    pub(crate) concept_count: usize,
}

pub(crate) struct EvalLifecycleArchivedInput {
    pub(crate) source_id: String,
    pub(crate) content: String,
}

impl MemoryDB {
    pub(crate) async fn eval_lifecycle_supersedes_inputs(
        &self,
    ) -> Result<Vec<EvalLifecycleSupersedesInput>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT DISTINCT source_id, supersedes FROM memories \
                 WHERE source_id LIKE 'merged_%' AND source = 'memory'",
                libsql::params![],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("supersedes scan: {e}")))?;

        let mut inputs = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("supersedes row scan: {e}")))?
        {
            inputs.push(EvalLifecycleSupersedesInput {
                source_id: row.get(0).unwrap_or_default(),
                supersedes: row.get(1).unwrap_or(None),
            });
        }
        Ok(inputs)
    }

    pub(crate) async fn eval_lifecycle_state_counts(
        &self,
    ) -> Result<EvalLifecycleStateCounts, WenlanError> {
        let conn = self.conn.lock().await;

        let memory_count = {
            let mut rows = conn
                .query(
                    "SELECT COUNT(*) FROM memories WHERE source = 'memory'",
                    libsql::params![],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("count memories: {e}")))?;
            if let Ok(Some(row)) = rows.next().await {
                row.get::<i64>(0).unwrap_or(0) as usize
            } else {
                0
            }
        };

        let archived_count = {
            let mut rows = conn
                .query(
                    "SELECT COUNT(*) FROM memories WHERE source = 'memory' AND supersede_mode = 'archive'",
                    libsql::params![],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("count archived: {e}")))?;
            if let Ok(Some(row)) = rows.next().await {
                row.get::<i64>(0).unwrap_or(0) as usize
            } else {
                0
            }
        };

        // G6 Stage 1.5a: counts via `entity_page_map` (1:1 with `entities` by
        // the shadow-page invariant) instead of `entities` directly.
        let entity_count = {
            let mut rows = conn
                .query(
                    "SELECT COUNT(*) FROM entity_page_map epm
                     JOIN pages p ON p.id = epm.page_id
                     WHERE p.kind = 'entity' AND p.status = 'active'",
                    libsql::params![],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("count entities: {e}")))?;
            if let Ok(Some(row)) = rows.next().await {
                row.get::<i64>(0).unwrap_or(0) as usize
            } else {
                0
            }
        };

        let concept_count = {
            let mut rows = conn
                .query(
                    "SELECT COUNT(*) FROM concepts WHERE status = 'active'",
                    libsql::params![],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("count concepts: {e}")))?;
            if let Ok(Some(row)) = rows.next().await {
                row.get::<i64>(0).unwrap_or(0) as usize
            } else {
                0
            }
        };

        Ok(EvalLifecycleStateCounts {
            memory_count,
            archived_count,
            entity_count,
            concept_count,
        })
    }

    pub(crate) async fn eval_lifecycle_merged_ids(&self) -> Result<HashSet<String>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT DISTINCT source_id FROM memories WHERE source_id LIKE 'merged_%' AND source = 'memory'",
                libsql::params![],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("merged ids: {e}")))?;

        let mut ids = HashSet::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("merged ids row scan: {e}")))?
        {
            if let Ok(id) = row.get::<String>(0) {
                ids.insert(id);
            }
        }
        Ok(ids)
    }

    pub(crate) async fn eval_lifecycle_archived_inputs(
        &self,
    ) -> Result<Vec<EvalLifecycleArchivedInput>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT source_id, content FROM memories \
                 WHERE supersede_mode = 'archive' AND source = 'memory' AND chunk_index = 0",
                libsql::params![],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("archive scan: {e}")))?;

        let mut inputs = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("archive row scan: {e}")))?
        {
            inputs.push(EvalLifecycleArchivedInput {
                source_id: row.get(0).unwrap_or_default(),
                content: row.get(1).unwrap_or_default(),
            });
        }
        Ok(inputs)
    }
}
