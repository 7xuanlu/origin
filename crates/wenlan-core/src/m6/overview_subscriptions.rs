// SPDX-License-Identifier: Apache-2.0
//! Migration-109 substrate for M6 overview subscriptions.

use crate::WenlanError;

/// Add the overview-subscription tables and exclusion indexes to the caller's
/// migration transaction.
pub(crate) async fn ensure_overview_subscription_tables(
    tx: &libsql::Transaction,
) -> Result<(), WenlanError> {
    tx.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS m6_overview_subscriptions (
            subscription_id  TEXT    NOT NULL PRIMARY KEY,
            scope_kind       TEXT    NOT NULL
                                CHECK (scope_kind IN ('install', 'space', 'community')),
            scope_id         TEXT    NOT NULL,
            page_id          TEXT    NOT NULL,
            space            TEXT,
            state            TEXT    NOT NULL
                                CHECK (state IN ('active', 'detached')),
            created_at       INTEGER NOT NULL,
            detached_at      INTEGER,
            CHECK (
                (scope_kind = 'install' AND space IS NULL) OR
                (scope_kind <> 'install' AND space IS NOT NULL)
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_m6_overview_sub_scope_active
            ON m6_overview_subscriptions(scope_kind, scope_id)
            WHERE state = 'active';

        CREATE UNIQUE INDEX IF NOT EXISTS idx_m6_overview_sub_page_active
            ON m6_overview_subscriptions(page_id)
            WHERE state = 'active';
        ",
    )
    .await
    .map(|_| ())
    .map_err(|error| WenlanError::VectorDb(format!("m109 overview subscriptions: {error}")))
}
