// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;

impl MemoryDB {
    pub(crate) async fn assert_eval_feature_substrate_live(
        &self,
        feature: &str,
    ) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        crate::eval::seed_contract::assert_feature_substrate_live(&conn, feature).await
    }

    pub(crate) async fn assert_eval_migration_substrates_live(&self) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        crate::eval::seed_contract::assert_feature_substrate_live(&conn, "temporal").await?;
        crate::eval::seed_contract::assert_feature_substrate_live(&conn, "graph").await?;
        crate::eval::seed_contract::assert_feature_substrate_live(&conn, "pages").await
    }
}
