// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

pub(crate) struct AutomaticRetroPageScan {
    pub(crate) eligible_source_count: Option<usize>,
    pub(crate) next_cursor: Option<String>,
    pub(crate) more: bool,
}

impl MemoryDB {
    /// Examines one Page for the automatic retro stub lane.
    ///
    /// The second Page establishes `more`; normalized and legacy source counts
    /// are capped at the caller's `source_read_limit`.
    pub(crate) async fn scan_automatic_retro_stub_slice(
        &self,
        cursor: Option<&str>,
        source_read_limit: usize,
    ) -> Result<AutomaticRetroPageScan, WenlanError> {
        let conn = self.conn.lock().await;
        let mut sql = String::from(
            "SELECT id, source_memory_ids, \
                    CASE WHEN status = 'active' \
                               AND COALESCE(creation_kind, 'distilled') = 'distilled' \
                               AND COALESCE(user_edited, 0) = 0 \
                               AND lower(title) != 'overview' \
                         THEN 1 ELSE 0 END AS eligible \
             FROM pages WHERE 1 = 1",
        );
        let mut bind = Vec::<libsql::Value>::new();
        if let Some(cursor) = cursor {
            sql.push_str(" AND id > ?");
            bind.push(libsql::Value::Text(cursor.to_string()));
        }
        sql.push_str(" ORDER BY id LIMIT 2");
        let mut rows = conn
            .query(&sql, libsql::params_from_iter(bind))
            .await
            .map_err(|error| WenlanError::VectorDb(format!("bounded retro Page scan: {error}")))?;
        let mut pages = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("bounded retro Page row: {error}")))?
        {
            pages.push((
                row.get::<String>(0).unwrap_or_default(),
                row.get::<String>(1)
                    .ok()
                    .and_then(|json| serde_json::from_str::<Vec<String>>(&json).ok())
                    .unwrap_or_default(),
                row.get::<i64>(2).unwrap_or(0) != 0,
            ));
        }
        drop(rows);
        let more = pages.len() > 1;
        let Some((page_id, fallback_sources, eligible)) = pages.into_iter().next() else {
            return Ok(AutomaticRetroPageScan {
                eligible_source_count: None,
                next_cursor: None,
                more: false,
            });
        };

        if !eligible {
            return Ok(AutomaticRetroPageScan {
                eligible_source_count: None,
                next_cursor: Some(page_id),
                more,
            });
        }

        // The caller-provided cap is sufficient for its stub decision; no
        // exact count or full source-list materialization is needed.
        let mut source_rows = conn
            .query(
                "SELECT memory_source_id FROM page_sources \
                 WHERE page_id = ?1 ORDER BY memory_source_id LIMIT ?2",
                libsql::params![page_id.as_str(), source_read_limit as i64],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("bounded retro sources for '{page_id}': {error}"))
            })?;
        let mut source_count = 0usize;
        while source_rows
            .next()
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("bounded retro source row for '{page_id}': {error}"))
            })?
            .is_some()
        {
            source_count += 1;
        }
        if source_count == 0 {
            source_count = fallback_sources.into_iter().take(source_read_limit).count();
        }
        Ok(AutomaticRetroPageScan {
            eligible_source_count: Some(source_count),
            next_cursor: Some(page_id),
            more,
        })
    }
}
