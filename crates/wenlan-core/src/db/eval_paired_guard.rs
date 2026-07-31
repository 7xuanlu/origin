// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

impl MemoryDB {
    pub(crate) async fn eval_paired_summary_node_count(&self) -> Result<i64, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM summary_nodes", libsql::params![])
            .await
            .map_err(|e| {
                WenlanError::VectorDb(format!("eval_paired_summary_node_count query: {e}"))
            })?;
        let Some(row) = rows.next().await.map_err(|e| {
            WenlanError::VectorDb(format!("eval_paired_summary_node_count next: {e}"))
        })?
        else {
            return Ok(0);
        };
        row.get::<i64>(0)
            .map_err(|e| WenlanError::VectorDb(format!("eval_paired_summary_node_count get: {e}")))
    }
}
