// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::lint::{
    observation::LintRunObserver,
    snapshot::{LintReadSnapshot, SnapshotError},
};
use std::sync::Arc;

impl MemoryDB {
    pub async fn open_lint_snapshot(&self) -> Result<LintReadSnapshot<'_>, SnapshotError> {
        LintReadSnapshot::open_with_freshness(&self._db, Arc::clone(&self.lint_freshness)).await
    }

    pub(crate) async fn open_unpinned_lint_snapshot(
        &self,
        observer: Arc<dyn LintRunObserver>,
    ) -> Result<LintReadSnapshot<'_>, SnapshotError> {
        LintReadSnapshot::open_unpinned(&self._db, Arc::clone(&self.lint_freshness), observer).await
    }
}
