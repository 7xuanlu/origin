// SPDX-License-Identifier: Apache-2.0
//! Attached-object capture and replay for the M5 `edges` rebuild.
//!
//! SQLite cannot alter a `CHECK` constraint, so widening `edges` means dropping
//! and recreating it — and `DROP TABLE` takes every index and trigger attached
//! to it. `docs/plans/2026-07-27-m5-edge-rebuild-matrix.md` §7 originally told
//! the implementer to recreate "the space fence, both twins". The live schema
//! attaches eight triggers. Recreating two would have silently dropped the six
//! M4 triggers that keep community grouping and page-community route inputs
//! invalidated, and nothing downstream fails loudly when that happens: the
//! tables stay correct-looking while their invalidation stops firing.
//!
//! So the rebuild does not carry a list. It reads the `CREATE` statements back
//! out of `sqlite_master` before the drop and replays them afterwards, which
//! stays correct when a later rung attaches a ninth trigger without touching
//! this file.

use super::MemoryDB;
use crate::WenlanError;

impl MemoryDB {
    /// The `CREATE` statements of every index and trigger attached to `table`,
    /// ordered so a replay is deterministic.
    ///
    /// Implicit indexes are skipped by `sql IS NOT NULL`: SQLite records a
    /// NULL `sql` for the index it derives from a `PRIMARY KEY` or `UNIQUE`
    /// declaration, and that index comes back with the `CREATE TABLE` rather
    /// than needing replay. Filtering on the NULL rather than on an
    /// `sqlite_autoindex%` name prefix keys off the property that actually
    /// makes them unreplayable — there is no statement to replay.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "wired into migration 97 with the edges rebuild")
    )]
    pub(super) async fn capture_attached_objects(
        tx: &libsql::Transaction,
        table: &str,
    ) -> Result<Vec<String>, WenlanError> {
        let mut rows = tx
            .query(
                "SELECT sql FROM sqlite_master
                 WHERE tbl_name = ?1
                   AND type IN ('index','trigger')
                   AND sql IS NOT NULL
                 ORDER BY type, name",
                libsql::params![table],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("capture attached objects for {table}: {error}"))
            })?;

        let mut statements = Vec::new();
        while let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("read attached object for {table}: {error}"))
        })? {
            statements.push(row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("decode attached object for {table}: {error}"))
            })?);
        }
        Ok(statements)
    }

    /// Replay captured `CREATE` statements against the rebuilt table.
    ///
    /// Fails loud on the first statement that does not apply. A rebuild that
    /// swallowed a failure here would leave a widened table with part of its
    /// fence missing, which is the "after rename, before triggers" window that
    /// artifact 3 §8 says must refuse to serve rather than accept writes.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "wired into migration 97 with the edges rebuild")
    )]
    pub(super) async fn replay_attached_objects(
        tx: &libsql::Transaction,
        statements: &[String],
    ) -> Result<(), WenlanError> {
        for statement in statements {
            tx.execute(statement, ()).await.map_err(|error| {
                WenlanError::VectorDb(format!("replay attached object [{statement}]: {error}"))
            })?;
        }
        Ok(())
    }
}
