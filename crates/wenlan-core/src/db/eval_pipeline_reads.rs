// SPDX-License-Identifier: Apache-2.0

use super::{not_self_archived, MemoryDB};
use crate::error::WenlanError;

impl MemoryDB {
    pub(crate) async fn eval_pipeline_corpus_contents(&self) -> Result<Vec<String>, WenlanError> {
        let conn = self.conn.lock().await;
        let live = not_self_archived("memories");
        let mut rows = conn
            .query(
                &format!("SELECT content FROM memories WHERE chunk_index = 0 AND {live}"),
                (),
            )
            .await
            .map_err(|e| WenlanError::Generic(format!("count_corpus: {e}")))?;

        let mut contents = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::Generic(e.to_string()))?
        {
            contents.push(
                row.get(0)
                    .map_err(|e| WenlanError::Generic(e.to_string()))?,
            );
        }
        Ok(contents)
    }

    pub(crate) async fn eval_pipeline_evidence_pairs(
        &self,
    ) -> Result<Vec<(String, String)>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT cs.memory_source_id, m.source_id AS concept_mem_sid \
                 FROM concept_sources cs \
                 JOIN concepts c ON cs.concept_id = c.id \
                 JOIN memories m ON m.source_id LIKE 'merged_%' \
                   AND m.chunk_index = 0 \
                   AND m.entity_id = c.entity_id \
                 WHERE c.status = 'active'",
                (),
            )
            .await
            .map_err(|e| WenlanError::Generic(format!("expand_evidence: {e}")))?;

        let mut pairs = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::Generic(e.to_string()))?
        {
            let memory_source_id = row
                .get(0)
                .map_err(|e| WenlanError::Generic(e.to_string()))?;
            let concept_memory_source_id = row
                .get(1)
                .map_err(|e| WenlanError::Generic(e.to_string()))?;
            pairs.push((memory_source_id, concept_memory_source_id));
        }
        Ok(pairs)
    }
}
