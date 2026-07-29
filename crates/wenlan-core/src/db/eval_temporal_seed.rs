// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvalTemporalSeed {
    pub(crate) source_id: String,
    pub(crate) event_date: i64,
}

impl MemoryDB {
    pub(crate) async fn apply_eval_temporal_seeds(&self, seeds: &[EvalTemporalSeed]) {
        let conn = self.conn.lock().await;
        for seed in seeds {
            let _ = conn
                .execute(
                    "UPDATE memories SET event_date = ? WHERE source_id = ?",
                    libsql::params![seed.event_date, seed.source_id.as_str()],
                )
                .await;
        }
    }
}
