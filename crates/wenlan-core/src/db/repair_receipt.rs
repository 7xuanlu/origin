// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;
use wenlan_types::repair::{RepairDigest, RepairTarget};

impl MemoryDB {
    pub(crate) async fn read_repair_target_receipt(
        &self,
        target: &RepairTarget,
    ) -> Result<(RepairDigest, u64), WenlanError> {
        let connection = self.conn.lock().await;
        crate::repair::repair_target_receipt_on_connection(&connection, target).await
    }
}
