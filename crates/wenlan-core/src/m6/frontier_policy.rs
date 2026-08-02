// SPDX-License-Identifier: Apache-2.0
//! D2/D3 plus J1 frontier-policy substrate for migration 109.
//!
//! This module deliberately owns no transaction. Migration setup and every
//! state transition take a caller-owned transaction so the marker change and
//! the replacement live reason cannot be observed separately.

use std::fmt;

/// A suppression remains live until the reconciler stores its lapse.
/// Wall-clock expiry is only eligibility for that transition.
pub const LIVE_SUPPRESSION_PREDICATE: &str = "lapsed_at IS NULL";

/// A retained card binding remains live until dismissal or expiry closes it.
pub const LIVE_CARD_BINDING_PREDICATE: &str = "closed_at IS NULL";

/// A retained quarantine remains live until an explicit lift stores its marker.
pub const LIVE_QUARANTINE_PREDICATE: &str = "lifted_at IS NULL";

/// Retained frontier histories that the space-rename closure must rewrite.
pub const SPACE_RENAME_TABLES: [&str; 3] = [
    "genesis_suppression",
    "genesis_card_binding",
    "genesis_quarantine",
];

const SUPPRESSION_SECONDS: i64 = 180 * 86_400;

#[derive(Debug)]
pub enum FrontierPolicyError {
    Database(libsql::Error),
    Invariant(String),
    #[cfg(test)]
    Injected(DismissFailpoint),
}

impl fmt::Display for FrontierPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Database(error) => write!(formatter, "frontier policy database error: {error}"),
            Self::Invariant(message) => write!(formatter, "frontier policy invariant: {message}"),
            #[cfg(test)]
            Self::Injected(failpoint) => {
                write!(formatter, "injected dismissal failure: {failpoint:?}")
            }
        }
    }
}

impl std::error::Error for FrontierPolicyError {}

impl From<libsql::Error> for FrontierPolicyError {
    fn from(error: libsql::Error) -> Self {
        Self::Database(error)
    }
}

pub type FrontierPolicyResult<T> = Result<T, FrontierPolicyError>;

/// One frontier group that a shared surfaced card names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CardGroup<'a> {
    pub independence_group_id: &'a str,
    pub page_id: &'a str,
}

impl<'a> CardGroup<'a> {
    pub const fn new(independence_group_id: &'a str, page_id: &'a str) -> Self {
        Self {
            independence_group_id,
            page_id,
        }
    }
}

#[derive(Debug)]
struct OpenCardBinding {
    space: String,
    independence_group_id: String,
    coverage_epoch: i64,
    page_id: String,
    first_seen_at: i64,
}

/// Additive D2/D3/J1 DDL. The migration-109 orchestrator calls this once inside
/// its shared transaction.
pub async fn ensure_frontier_policy_tables(tx: &libsql::Transaction) -> FrontierPolicyResult<()> {
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS genesis_suppression (
             space                  TEXT    NOT NULL,
             independence_group_id  TEXT    NOT NULL,
             coverage_epoch         INTEGER NOT NULL,
             page_id                TEXT    NOT NULL CHECK(length(trim(page_id)) > 0),
             reason                 TEXT    NOT NULL CHECK(length(trim(reason)) > 0),
             first_seen_at          INTEGER NOT NULL,
             suppressed_at          INTEGER NOT NULL,
             expires_at             INTEGER NOT NULL,
             lapsed_at              INTEGER,
             CHECK(expires_at > suppressed_at),
             CHECK(lapsed_at IS NULL OR lapsed_at >= expires_at),
             PRIMARY KEY(space, independence_group_id, coverage_epoch, suppressed_at)
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_genesis_suppression_one_live
             ON genesis_suppression(space, independence_group_id, coverage_epoch)
             WHERE lapsed_at IS NULL;
         CREATE INDEX IF NOT EXISTS idx_genesis_suppression_lapse
             ON genesis_suppression(space, coverage_epoch, expires_at)
             WHERE lapsed_at IS NULL;

         CREATE TABLE IF NOT EXISTS genesis_card_binding (
             space                  TEXT    NOT NULL,
             independence_group_id  TEXT    NOT NULL,
             coverage_epoch         INTEGER NOT NULL,
             card_id                TEXT    NOT NULL CHECK(length(trim(card_id)) > 0),
             page_id                TEXT    NOT NULL CHECK(length(trim(page_id)) > 0),
             first_seen_at          INTEGER NOT NULL,
             created_at             INTEGER NOT NULL,
             closed_at              INTEGER,
             CHECK(closed_at IS NULL OR closed_at >= created_at),
             PRIMARY KEY(space, independence_group_id, coverage_epoch, card_id)
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_genesis_card_binding_one_live
             ON genesis_card_binding(space, independence_group_id, coverage_epoch)
             WHERE closed_at IS NULL;
         CREATE INDEX IF NOT EXISTS idx_genesis_card_binding_open_card
             ON genesis_card_binding(card_id)
             WHERE closed_at IS NULL;

         CREATE TABLE IF NOT EXISTS genesis_quarantine (
             space                  TEXT    NOT NULL,
             independence_group_id  TEXT    NOT NULL,
             coverage_epoch         INTEGER NOT NULL,
             reason                 TEXT    NOT NULL CHECK(length(trim(reason)) > 0),
             first_seen_at          INTEGER NOT NULL,
             quarantined_at         INTEGER NOT NULL,
             lifted_at              INTEGER,
             CHECK(lifted_at IS NULL OR lifted_at >= quarantined_at),
             PRIMARY KEY(space, independence_group_id, coverage_epoch, quarantined_at)
         );
         CREATE UNIQUE INDEX IF NOT EXISTS idx_genesis_quarantine_one_live
             ON genesis_quarantine(space, independence_group_id, coverage_epoch)
             WHERE lifted_at IS NULL;",
    )
    .await?;
    Ok(())
}

/// F7 from the frontier: retire its re-derivable row and append one live
/// suppression. The page ID is the suppression identity; it is not an opaque
/// free-form identity string.
pub async fn suppress_frontier_group(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
    page_id: &str,
    reason: &str,
) -> FrontierPolicyResult<()> {
    let first_seen_at =
        frontier_first_seen_at(tx, space, independence_group_id, coverage_epoch).await?;
    let removed = tx
        .execute(
            "DELETE FROM genesis_frontier
              WHERE space = ?1
                AND independence_group_id = ?2
                AND coverage_epoch = ?3",
            libsql::params![space, independence_group_id, coverage_epoch],
        )
        .await?;
    require_one(
        removed,
        "suppress transition did not retire exactly one frontier row",
    )?;
    insert_suppression(
        tx,
        space,
        independence_group_id,
        coverage_epoch,
        page_id,
        reason,
        first_seen_at,
    )
    .await
}

/// F3: bind all named frontier groups to one retained shared-card history.
/// Each deletion and binding is in the caller's transaction; an error leaves
/// every group in its prior frontier state after rollback.
pub async fn bind_frontier_groups_to_card(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
    card_id: &str,
    groups: &[CardGroup<'_>],
) -> FrontierPolicyResult<u64> {
    for group in groups {
        let first_seen_at =
            frontier_first_seen_at(tx, space, group.independence_group_id, coverage_epoch).await?;
        let removed = tx
            .execute(
                "DELETE FROM genesis_frontier
                  WHERE space = ?1
                    AND independence_group_id = ?2
                    AND coverage_epoch = ?3",
                libsql::params![space, group.independence_group_id, coverage_epoch],
            )
            .await?;
        require_one(
            removed,
            "card transition did not retire exactly one frontier row",
        )?;
        tx.execute(
            "INSERT INTO genesis_card_binding (
                 space, independence_group_id, coverage_epoch, card_id, page_id,
                 first_seen_at, created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, unixepoch())",
            libsql::params![
                space,
                group.independence_group_id,
                coverage_epoch,
                card_id,
                group.page_id,
                first_seen_at
            ],
        )
        .await?;
    }
    Ok(groups.len() as u64)
}

/// F11: first stamp every observed expiry with the stable liveness marker,
/// then restore its frontier row. Both statements share the caller's
/// transaction, so G5 can observe either the suppression or the frontier but
/// never a wall-clock-created gap between them.
pub async fn reconcile_expired_suppressions(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
) -> FrontierPolicyResult<u64> {
    let mut rows = tx
        .query(
            "SELECT independence_group_id, suppressed_at, first_seen_at
               FROM genesis_suppression
              WHERE space = ?1
                AND coverage_epoch = ?2
                AND lapsed_at IS NULL
                AND expires_at <= unixepoch()
              ORDER BY expires_at, independence_group_id, suppressed_at",
            libsql::params![space, coverage_epoch],
        )
        .await?;
    let mut expired = Vec::new();
    while let Some(row) = rows.next().await? {
        expired.push((
            row.get::<String>(0)?,
            row.get::<i64>(1)?,
            row.get::<i64>(2)?,
        ));
    }
    drop(rows);

    for (independence_group_id, suppressed_at, first_seen_at) in &expired {
        let lapsed = tx
            .execute(
                "UPDATE genesis_suppression
                    SET lapsed_at = unixepoch()
                  WHERE space = ?1
                    AND independence_group_id = ?2
                    AND coverage_epoch = ?3
                    AND suppressed_at = ?4
                    AND lapsed_at IS NULL
                    AND expires_at <= unixepoch()",
                libsql::params![
                    space,
                    independence_group_id.as_str(),
                    coverage_epoch,
                    suppressed_at
                ],
            )
            .await?;
        require_one(lapsed, "suppression lapse marker lost its live row")?;
        tx.execute(
            "INSERT INTO genesis_frontier (
                 space, independence_group_id, coverage_epoch,
                 first_seen_at, next_scan_at
             ) VALUES (?1, ?2, ?3, ?4, unixepoch())",
            libsql::params![
                space,
                independence_group_id.as_str(),
                coverage_epoch,
                first_seen_at
            ],
        )
        .await?;
    }
    Ok(expired.len() as u64)
}

/// F9 from the frontier: retire the frontier row and append an explicit live
/// quarantine. Fast lift/reactivation in one SQLite second remains repeatable:
/// the stored identity clock advances to at least the preceding identity + 1.
pub async fn quarantine_frontier_group(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
    reason: &str,
) -> FrontierPolicyResult<()> {
    let first_seen_at =
        frontier_first_seen_at(tx, space, independence_group_id, coverage_epoch).await?;
    let removed = tx
        .execute(
            "DELETE FROM genesis_frontier
              WHERE space = ?1
                AND independence_group_id = ?2
                AND coverage_epoch = ?3",
            libsql::params![space, independence_group_id, coverage_epoch],
        )
        .await?;
    require_one(
        removed,
        "quarantine transition did not retire exactly one frontier row",
    )?;
    tx.execute(
        "INSERT INTO genesis_quarantine (
             space, independence_group_id, coverage_epoch, reason,
             first_seen_at, quarantined_at
         ) VALUES (
             ?1, ?2, ?3, ?4, ?5,
             MAX(
                 unixepoch(),
                 COALESCE((
                     SELECT MAX(quarantined_at) + 1
                       FROM genesis_quarantine
                      WHERE space = ?1
                        AND independence_group_id = ?2
                        AND coverage_epoch = ?3
                 ), unixepoch())
             )
         )",
        libsql::params![
            space,
            independence_group_id,
            coverage_epoch,
            reason,
            first_seen_at
        ],
    )
    .await?;
    Ok(())
}

/// F12: stamp the stable lift marker before restoring frontier in the same
/// caller-owned transaction. The retained row remains queryable forever.
pub async fn lift_quarantine_to_frontier(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
) -> FrontierPolicyResult<u64> {
    let mut rows = tx
        .query(
            "SELECT quarantined_at, first_seen_at
               FROM genesis_quarantine
              WHERE space = ?1
                AND independence_group_id = ?2
                AND coverage_epoch = ?3
                AND lifted_at IS NULL",
            libsql::params![space, independence_group_id, coverage_epoch],
        )
        .await?;
    let row = rows.next().await?.ok_or_else(|| {
        FrontierPolicyError::Invariant(format!(
            "group {independence_group_id:?} in {space:?} epoch {coverage_epoch} has no live quarantine"
        ))
    })?;
    let quarantined_at = row.get::<i64>(0)?;
    let first_seen_at = row.get::<i64>(1)?;
    drop(rows);

    let lifted = tx
        .execute(
            "UPDATE genesis_quarantine
                SET lifted_at = MAX(unixepoch(), quarantined_at)
              WHERE space = ?1
                AND independence_group_id = ?2
                AND coverage_epoch = ?3
                AND quarantined_at = ?4
                AND lifted_at IS NULL",
            libsql::params![space, independence_group_id, coverage_epoch, quarantined_at],
        )
        .await?;
    require_one(lifted, "quarantine lift marker lost its live row")?;
    tx.execute(
        "INSERT INTO genesis_frontier (
             space, independence_group_id, coverage_epoch,
             first_seen_at, next_scan_at
         ) VALUES (?1, ?2, ?3, ?4, unixepoch())",
        libsql::params![space, independence_group_id, coverage_epoch, first_seen_at],
    )
    .await?;
    Ok(1)
}

/// F7 from a shared card: close every live per-group binding for `card_id`
/// first, then append each group's suppression reason. The ordering is
/// deliberate and the entire operation remains one caller-owned transaction.
pub async fn dismiss_card_to_suppression(
    tx: &libsql::Transaction,
    card_id: &str,
    reason: &str,
) -> FrontierPolicyResult<u64> {
    dismiss_card_to_suppression_impl(tx, card_id, reason, None).await
}

async fn dismiss_card_to_suppression_impl(
    tx: &libsql::Transaction,
    card_id: &str,
    reason: &str,
    #[cfg(test)] failpoint: Option<DismissFailpoint>,
    #[cfg(not(test))] _failpoint: Option<()>,
) -> FrontierPolicyResult<u64> {
    let bindings = open_card_bindings(tx, card_id).await?;
    if bindings.is_empty() {
        return Ok(0);
    }

    let closed = tx
        .execute(
            "UPDATE genesis_card_binding
                SET closed_at = unixepoch()
              WHERE card_id = ?1 AND closed_at IS NULL",
            libsql::params![card_id],
        )
        .await?;
    if closed != bindings.len() as u64 {
        return Err(FrontierPolicyError::Invariant(format!(
            "card {card_id:?} selected {} open bindings but closed {closed}",
            bindings.len()
        )));
    }

    #[cfg(test)]
    if failpoint == Some(DismissFailpoint::AfterBindingsClosed) {
        return Err(FrontierPolicyError::Injected(
            DismissFailpoint::AfterBindingsClosed,
        ));
    }

    #[cfg(test)]
    let mut completed = 0usize;
    for binding in &bindings {
        insert_suppression(
            tx,
            &binding.space,
            &binding.independence_group_id,
            binding.coverage_epoch,
            &binding.page_id,
            reason,
            binding.first_seen_at,
        )
        .await?;
        #[cfg(test)]
        {
            completed += 1;
        }
        #[cfg(test)]
        if failpoint == Some(DismissFailpoint::AfterSuppression(completed)) {
            return Err(FrontierPolicyError::Injected(
                DismissFailpoint::AfterSuppression(completed),
            ));
        }
    }
    Ok(bindings.len() as u64)
}

async fn insert_suppression(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
    page_id: &str,
    reason: &str,
    first_seen_at: i64,
) -> FrontierPolicyResult<()> {
    tx.execute(
        "INSERT INTO genesis_suppression (
             space, independence_group_id, coverage_epoch, page_id, reason,
             first_seen_at, suppressed_at, expires_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6,
                   unixepoch(), unixepoch() + ?7)",
        libsql::params![
            space,
            independence_group_id,
            coverage_epoch,
            page_id,
            reason,
            first_seen_at,
            SUPPRESSION_SECONDS
        ],
    )
    .await?;
    Ok(())
}

async fn frontier_first_seen_at(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
) -> FrontierPolicyResult<i64> {
    let mut rows = tx
        .query(
            "SELECT first_seen_at FROM genesis_frontier
              WHERE space = ?1
                AND independence_group_id = ?2
                AND coverage_epoch = ?3",
            libsql::params![space, independence_group_id, coverage_epoch],
        )
        .await?;
    let row = rows.next().await?.ok_or_else(|| {
        FrontierPolicyError::Invariant(format!(
            "group {independence_group_id:?} in {space:?} epoch {coverage_epoch} is not in frontier"
        ))
    })?;
    Ok(row.get::<i64>(0)?)
}

async fn open_card_bindings(
    tx: &libsql::Transaction,
    card_id: &str,
) -> FrontierPolicyResult<Vec<OpenCardBinding>> {
    let mut rows = tx
        .query(
            "SELECT space, independence_group_id, coverage_epoch, page_id, first_seen_at
               FROM genesis_card_binding
              WHERE card_id = ?1 AND closed_at IS NULL
              ORDER BY space, coverage_epoch, independence_group_id",
            libsql::params![card_id],
        )
        .await?;
    let mut bindings = Vec::new();
    while let Some(row) = rows.next().await? {
        bindings.push(OpenCardBinding {
            space: row.get(0)?,
            independence_group_id: row.get(1)?,
            coverage_epoch: row.get(2)?,
            page_id: row.get(3)?,
            first_seen_at: row.get(4)?,
        });
    }
    Ok(bindings)
}

fn require_one(affected: u64, message: &str) -> FrontierPolicyResult<()> {
    if affected == 1 {
        Ok(())
    } else {
        Err(FrontierPolicyError::Invariant(format!(
            "{message}; affected {affected} rows"
        )))
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DismissFailpoint {
    AfterBindingsClosed,
    AfterSuppression(usize),
}

#[cfg(test)]
pub async fn dismiss_card_to_suppression_with_failpoint(
    tx: &libsql::Transaction,
    card_id: &str,
    reason: &str,
    failpoint: DismissFailpoint,
) -> FrontierPolicyResult<u64> {
    dismiss_card_to_suppression_impl(tx, card_id, reason, Some(failpoint)).await
}
