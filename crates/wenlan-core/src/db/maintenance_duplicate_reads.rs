// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

#[derive(Debug)]
pub(crate) struct NearDuplicatePairRead {
    pub(crate) left_id: String,
    pub(crate) right_id: String,
    pub(crate) left_embedding: Vec<f32>,
    pub(crate) right_embedding: Vec<f32>,
    pub(crate) left_fallback_sources: Vec<String>,
    pub(crate) right_fallback_sources: Vec<String>,
    pub(crate) eligible: bool,
}

#[derive(Debug)]
pub(crate) struct PageEmbeddingDistanceRead {
    pub(crate) left_id: String,
    pub(crate) right_id: String,
    pub(crate) distance: f64,
}

pub(crate) struct NearDuplicateSliceReader<'db> {
    conn: tokio::sync::MutexGuard<'db, libsql::Connection>,
}

impl MemoryDB {
    pub(crate) async fn begin_near_duplicate_slice_reader(&self) -> NearDuplicateSliceReader<'_> {
        NearDuplicateSliceReader {
            conn: self.conn.lock().await,
        }
    }

    pub(crate) async fn embedding_near_duplicate_pairs(
        &self,
        threshold: f64,
        limit: Option<usize>,
    ) -> Result<Vec<PageEmbeddingDistanceRead>, WenlanError> {
        let conn = self.conn.lock().await;
        let sql = match limit {
            Some(_) => {
                "SELECT a.id, b.id, vector_distance_cos(a.embedding, b.embedding) AS dist \
                 FROM pages a \
                 JOIN pages b ON a.id < b.id \
                 WHERE a.status = 'active' \
                   AND b.status = 'active' \
                   AND a.embedding IS NOT NULL \
                   AND b.embedding IS NOT NULL \
                   AND COALESCE(a.review_status, 'confirmed') = 'confirmed' \
                   AND COALESCE(b.review_status, 'confirmed') = 'confirmed' \
                   AND a.space = b.space \
                   AND lower(a.title) != 'overview' \
                   AND lower(b.title) != 'overview' \
                   AND vector_distance_cos(a.embedding, b.embedding) <= ?1 \
                 ORDER BY dist ASC \
                 LIMIT ?2"
            }
            None => {
                "SELECT a.id, b.id, vector_distance_cos(a.embedding, b.embedding) AS dist \
                 FROM pages a \
                 JOIN pages b ON a.id < b.id \
                 WHERE a.status = 'active' \
                   AND b.status = 'active' \
                   AND a.embedding IS NOT NULL \
                   AND b.embedding IS NOT NULL \
                   AND COALESCE(a.review_status, 'confirmed') = 'confirmed' \
                   AND COALESCE(b.review_status, 'confirmed') = 'confirmed' \
                   AND a.space = b.space \
                   AND lower(a.title) != 'overview' \
                   AND lower(b.title) != 'overview' \
                   AND vector_distance_cos(a.embedding, b.embedding) <= ?1 \
                 ORDER BY dist ASC"
            }
        };
        let mut rows = match limit {
            Some(limit) => {
                conn.query(sql, libsql::params![threshold, limit as i64])
                    .await
            }
            None => conn.query(sql, libsql::params![threshold]).await,
        }
        .map_err(|e| WenlanError::VectorDb(format!("page near-duplicate query: {e}")))?;

        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("page near-duplicate row: {e}")))?
        {
            let left_id = row
                .get(0)
                .map_err(|e| WenlanError::VectorDb(format!("near-dup left id: {e}")))?;
            let right_id = row
                .get(1)
                .map_err(|e| WenlanError::VectorDb(format!("near-dup right id: {e}")))?;
            out.push(PageEmbeddingDistanceRead {
                left_id,
                right_id,
                distance: row.get(2).unwrap_or(1.0),
            });
        }
        Ok(out)
    }
}

impl NearDuplicateSliceReader<'_> {
    pub(crate) async fn scan_near_duplicate_slice(
        &self,
        cursor: Option<(&str, &str)>,
        pair_read_limit: usize,
    ) -> Result<Vec<NearDuplicatePairRead>, WenlanError> {
        let mut sql = String::from(
            "SELECT a.id, b.id, a.embedding, b.embedding, \
                    a.source_memory_ids, b.source_memory_ids, \
                    CASE WHEN a.status = 'active' \
                               AND b.status = 'active' \
                               AND COALESCE(a.review_status, 'confirmed') = 'confirmed' \
                               AND COALESCE(b.review_status, 'confirmed') = 'confirmed' \
                               AND COALESCE(a.workspace, a.space, '') = COALESCE(b.workspace, b.space, '') \
                               AND lower(a.title) != 'overview' \
                               AND lower(b.title) != 'overview' \
                         THEN 1 ELSE 0 END AS eligible \
             FROM pages a \
             JOIN pages b ON a.id < b.id \
             WHERE 1 = 1",
        );
        let mut bind = Vec::<libsql::Value>::new();
        if let Some((left_id, right_id)) = cursor {
            sql.push_str(" AND (a.id > ? OR (a.id = ? AND b.id > ?))");
            bind.push(libsql::Value::Text(left_id.to_owned()));
            bind.push(libsql::Value::Text(left_id.to_owned()));
            bind.push(libsql::Value::Text(right_id.to_owned()));
        }
        sql.push_str(" ORDER BY a.id, b.id LIMIT ?");
        bind.push(libsql::Value::Integer(pair_read_limit as i64));

        let mut rows = self
            .conn
            .query(&sql, libsql::params_from_iter(bind))
            .await
            .map_err(|error| WenlanError::VectorDb(format!("bounded Page pair scan: {error}")))?;
        let mut pair_rows = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("bounded Page pair row: {error}")))?
        {
            let decode_embedding = |index| {
                row.get::<Vec<u8>>(index)
                    .unwrap_or_default()
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect::<Vec<_>>()
            };
            let decode_sources = |index| {
                row.get::<String>(index)
                    .ok()
                    .and_then(|json| serde_json::from_str::<Vec<String>>(&json).ok())
                    .unwrap_or_default()
            };
            pair_rows.push(NearDuplicatePairRead {
                left_id: row.get::<String>(0).unwrap_or_default(),
                right_id: row.get::<String>(1).unwrap_or_default(),
                left_embedding: decode_embedding(2),
                right_embedding: decode_embedding(3),
                left_fallback_sources: decode_sources(4),
                right_fallback_sources: decode_sources(5),
                eligible: row.get::<i64>(6).unwrap_or(0) != 0,
            });
        }
        drop(rows);

        Ok(pair_rows)
    }

    pub(crate) async fn load_bounded_page_source_ids(
        &self,
        page_id: &str,
        source_read_limit: usize,
    ) -> Result<Vec<String>, WenlanError> {
        let mut source_rows = self
            .conn
            .query(
                "SELECT memory_source_id FROM page_sources \
                 WHERE page_id = ?1 ORDER BY memory_source_id LIMIT ?2",
                libsql::params![page_id, source_read_limit as i64],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("bounded Page sources for '{page_id}': {error}"))
            })?;
        let mut source_ids = Vec::new();
        while let Some(row) = source_rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("bounded Page source row for '{page_id}': {error}"))
        })? {
            source_ids.push(row.get::<String>(0).unwrap_or_default());
        }
        Ok(source_ids)
    }
}
