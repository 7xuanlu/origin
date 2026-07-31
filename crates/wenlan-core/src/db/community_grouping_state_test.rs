// SPDX-License-Identifier: Apache-2.0
//! Contract tests for DB-owned community grouping state.

use super::CommunityGroupingLeaseCleanup;
use std::sync::Arc;

#[test]
fn lease_cleanup_drop_without_tokio_runtime_falls_back_without_panicking() {
    let directory = tempfile::tempdir().expect("lease cleanup tempdir");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("lease cleanup setup runtime");
    let (_db, connection) = runtime.block_on(async {
        let db = libsql::Builder::new_local(directory.path().join("lease-cleanup.db"))
            .build()
            .await
            .expect("lease cleanup database");
        let connection = db.connect().expect("lease cleanup connection");
        (db, connection)
    });
    drop(runtime);

    let cleanup = CommunityGroupingLeaseCleanup::new(
        Arc::new(tokio::sync::Mutex::new(connection)),
        "no-runtime-space".to_owned(),
        1,
        "no-runtime-token".to_owned(),
    );
    drop(cleanup);
}
