//! S0-162: renaming a space closes over every space-keyed row M6 genesis can
//! reach.
//!
//! `update_space` cascades over `memories`, `entities`, and `pages`/`workspace`
//! and historically stopped there, while the M4 community substrate and the M6
//! genesis substrate key on the same mutable string. A stale name is not
//! self-healing: the scheduler claims dirty spaces with
//! `SELECT space FROM space_graph_state WHERE dirty = 1` and no join to a live
//! `spaces` row (`db/community_grouping_state.rs:84`), so an orphaned row can be
//! selected indefinitely.
//!
//! The postcondition is S0-162's, stated over rows rather than over statements:
//! after a rename, no M6-reachable row names a space that does not resolve.
//! Every row keyed by the old name is either updated to the new name or retired.
//!
//! One space-keyed table is deliberately excluded and the exclusion is the scope
//! line: `page_draft_create_requests` (`db.rs:7783`). The old name it keeps is
//! not stale data, it is replay history, and its immutability is the table's
//! purpose — migration 76 says so at `db.rs:7771`, the create path matches a
//! replay against the stored request before validating the space against the
//! live row (`db/page_drafts.rs:246`), and
//! `create_replay_returns_the_server_mutated_scope_after_space_rename` locks it
//! down. Rewriting those rows would break delayed-create replay.

use crate::WenlanError;

/// The closure, in **dependency order**, each table paired with whether `space`
/// sits inside its PRIMARY KEY.
///
/// The order is load-bearing and is not alphabetical. `community_members` and
/// `page_community_assignments` both carry a BEFORE UPDATE fence that re-checks
/// the row's community against `communities.space` (`db.rs:11204`, `db.rs:10978`),
/// so `communities` has to be carrying the new name before either is rewritten
/// or the fence aborts the rename. `edges` carries the matching AFTER UPDATE
/// fence against its endpoint's space (`db.rs:8932`), which is why the caller
/// invokes this only after the memories/entities/pages cascade has run.
///
/// A `true` in the second field means a row already keyed to the destination
/// name would collide on the PRIMARY KEY. Re-derivable community control rows
/// are retired before the rewrite. This is not defensive: the ORDINARY rename
/// hits it. The pre-existing `pages` cascade fires
/// `m4_page_community_page_invalidate`, which does
/// `INSERT OR IGNORE INTO community_route_space_inputs (NEW.space, 0)`
/// (`db.rs:11014`) — so by the time this runs, a row under the destination name
/// already exists, created by this very rename.
///
/// Retiring it and moving the old row over is also the right direction rather
/// than merely the convenient one: the trigger's row starts at generation 0
/// while the old-name row carries the accumulated routing generation, and
/// keeping the fresh row would walk that counter backwards.
///
/// Rows that predate the rename are orphans by construction. `spaces.name` is
/// UNIQUE, so a rename to a live space fails before the cascade runs and any
/// surviving community row under that name names no live space. The genesis
/// rows are different: they carry permanent provenance/coverage and may never
/// be retired merely because their space name does not currently resolve. The
/// destination fence below therefore refuses the entire rename when any such
/// row exists.
const CLOSURE: &[(&str, bool)] = &[
    ("communities", false),
    ("community_members", true),
    ("page_community_assignments", false),
    ("page_community_route_inputs", false),
    ("community_route_space_inputs", true),
    ("space_graph_state", true),
    ("grouping_leases", true),
    // `community_reader_parity` is NOT here, and its absence is a correction to
    // S0-162 rather than an omission. The artifact's closed enumeration lists it
    // as the eleventh table and cites its DDL, but that DDL (`db.rs:10926`) is
    // undone by the later "retire local-only community parity model" migration,
    // which does `DROP TABLE IF EXISTS community_reader_parity` (`db.rs:11659`)
    // and replaces the per-space table with the global singleton
    // `community_parity_input_state` — which has no `space` column at all. The
    // table does not exist in a migrated database, so closing over it raises
    // `no such table`. Ten of the eleven are real; this one is a citation to a
    // statement a later migration takes back.
    ("community_reader_space_proof", true),
    ("community_publication_receipts", true),
    ("edges", false),
    // PR-A's own genesis substrate (migration 108). Empty in this build, but
    // the postcondition is stated over M6-reachable rows and these are the most
    // M6-reachable rows there are; leaving them out would mean the rung that
    // starts writing them has to remember to come back.
    ("genesis_candidates", false),
    ("genesis_coverage_state", true),
    ("genesis_group_coverage", true),
    ("genesis_frontier", true),
    // Migration 109 completes the M6 substrate. Relevance rows and refresh
    // jobs are re-derivable; human decisions, readiness evidence, exact
    // dependency snapshots, retained subscriptions, and monotone counters are
    // permanent and are fenced below before any destination cleanup occurs.
    ("m6_pair_stats", true),
    ("m6_adjacency", true),
    ("genesis_refresh_jobs", false),
    ("m6_refresh_dependencies", true),
    ("m6_readiness", true),
    ("m6_readiness_soak_receipts", true),
    ("genesis_suppression", true),
    ("genesis_card_binding", true),
    ("genesis_quarantine", true),
    ("m6_overview_subscriptions", false),
    ("m6_counters", false),
];

const PERMANENT_GENESIS_TABLES: &[&str] = &[
    "genesis_candidates",
    "genesis_coverage_state",
    "genesis_group_coverage",
    "genesis_frontier",
    "m6_refresh_dependencies",
    "m6_readiness",
    "m6_readiness_soak_receipts",
    "genesis_suppression",
    "genesis_card_binding",
    "genesis_quarantine",
    "m6_overview_subscriptions",
    "m6_counters",
];

/// Re-derivable rows whose `space` is not part of their primary key. They do
/// not collide mechanically with the renamed source row, but leaving an orphan
/// under the destination name would make unrelated stale work live when the
/// space rename creates that name.
const DESTINATION_ORPHANS_TO_RETIRE: &[&str] = &["genesis_refresh_jobs"];

/// Permanent M6 rows whose identity is `spaces.id`, not the name. Migration
/// 109's identity triggers (`m6/remaining_substrate.rs`) refuse an `UPDATE …
/// SET space` to a name whose `spaces.id` differs from the row's `space_id`,
/// and refuse every `DELETE`, so a merge into a live space can neither move
/// nor retire them. A rename keeps the id and passes; a merge leaves them
/// under the source name — the retained-proof posture `delete_space(_,
/// "keep")` already has, locked by
/// `delete_keep_cannot_authorize_proof_parking_and_zero_reset`.
const SPACE_ID_BOUND_TABLES: &[&str] = &["genesis_coverage_state", "m6_counters"];

/// Rewrite every space-keyed row from `old_name` to `new_name` inside the
/// caller's transaction. The caller must already have renamed the `spaces` row
/// and cascaded `memories`, `entities`, and `pages` (see the `edges` fence
/// above).
pub(super) async fn cascade_space_rename(
    tx: &libsql::Transaction,
    old_name: &str,
    new_name: &str,
) -> Result<(), WenlanError> {
    // Table names come from `CLOSURE`, never from a caller; the space names are
    // parameterized.
    for table in PERMANENT_GENESIS_TABLES {
        let mut rows = tx
            .query(
                &format!("SELECT 1 FROM {table} WHERE space = ?1 LIMIT 1"),
                libsql::params![new_name],
            )
            .await
            .map_err(|e| {
                WenlanError::VectorDb(format!(
                    "update_space inspect destination genesis {table}: {e}"
                ))
            })?;
        if rows
            .next()
            .await
            .map_err(|e| {
                WenlanError::VectorDb(format!(
                    "update_space inspect destination genesis {table}: {e}"
                ))
            })?
            .is_some()
        {
            return Err(WenlanError::Validation(format!(
                "destination genesis table {table} already has rows for space {new_name:?}"
            )));
        }
    }

    for table in DESTINATION_ORPHANS_TO_RETIRE {
        tx.execute(
            &format!("DELETE FROM {table} WHERE space = ?1"),
            libsql::params![new_name],
        )
        .await
        .map_err(|e| {
            WenlanError::VectorDb(format!("update_space retire orphan {table} rows: {e}"))
        })?;
    }

    for (table, space_in_primary_key) in CLOSURE {
        if !space_in_primary_key {
            continue;
        }
        tx.execute(
            &format!("DELETE FROM {table} WHERE space = ?1"),
            libsql::params![new_name],
        )
        .await
        .map_err(|e| {
            WenlanError::VectorDb(format!("update_space retire orphan {table} rows: {e}"))
        })?;
    }
    for (table, _) in CLOSURE {
        tx.execute(
            &format!("UPDATE {table} SET space = ?1 WHERE space = ?2"),
            libsql::params![new_name, old_name],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("update_space cascade {table}: {e}")))?;
    }
    Ok(())
}

/// Fold every space-keyed row from `old_name` into the LIVE space
/// `target_name` inside the caller's transaction, after the memories/pages
/// cascade (see the `edges` fence above). This is a merge, not a rename: the
/// destination has rows of its own that must survive, so nothing under
/// `target_name` is retired. Over the same `CLOSURE`, per table: `edges` are
/// re-keyed to the target; `communities` are retired the way
/// `finalize_community_grouping` retires a superseded generation; re-derivable
/// community/genesis control rows are dropped; permanent genesis rows move
/// only when the target holds none (a merge must not splice two spaces'
/// histories, so both sides carrying rows refuses the whole operation before
/// anything is written); `SPACE_ID_BOUND_TABLES` stay put. Finally the target
/// is marked dirty so grouping re-runs over the merged input.
pub(super) async fn cascade_space_merge(
    tx: &libsql::Transaction,
    old_name: &str,
    target_name: &str,
) -> Result<(), WenlanError> {
    for table in PERMANENT_GENESIS_TABLES {
        if SPACE_ID_BOUND_TABLES.contains(table) {
            continue;
        }
        if has_rows_naming(tx, table, old_name).await?
            && has_rows_naming(tx, table, target_name).await?
        {
            return Err(WenlanError::Validation(format!(
                "both {old_name:?} and {target_name:?} carry permanent genesis rows in {table}; \
                 refusing to merge their histories"
            )));
        }
    }

    // `edges` goes first, out of CLOSURE order: `m4_grouping_edge_update`
    // (db.rs `m4_grouping_edge_update`) re-materialises the `space_graph_state`
    // row of OLD.space when a grounded edge changes space, so that row can only
    // be retired once the edge rewrite is done.
    tx.execute(
        "UPDATE edges SET space = ?1 WHERE space = ?2",
        libsql::params![target_name, old_name],
    )
    .await
    .map_err(|e| WenlanError::VectorDb(format!("space merge cascade edges: {e}")))?;

    let now = chrono::Utc::now().timestamp();
    for (table, _) in CLOSURE {
        match *table {
            "edges" => {}
            "communities" => {
                tx.execute(
                    "UPDATE communities SET retired_at = ?2, updated_at = ?2 \
                     WHERE space = ?1 AND retired_at IS NULL",
                    libsql::params![old_name, now],
                )
                .await
                .map_err(|e| {
                    WenlanError::VectorDb(format!("space merge retire communities: {e}"))
                })?;
            }
            table if SPACE_ID_BOUND_TABLES.contains(&table) => {}
            table if PERMANENT_GENESIS_TABLES.contains(&table) => {
                tx.execute(
                    &format!("UPDATE {table} SET space = ?1 WHERE space = ?2"),
                    libsql::params![target_name, old_name],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("space merge cascade {table}: {e}")))?;
            }
            table => {
                tx.execute(
                    &format!("DELETE FROM {table} WHERE space = ?1"),
                    libsql::params![old_name],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("space merge retire {table}: {e}")))?;
            }
        }
    }

    // Same shape as `m4_grouping_entity_update`'s destination upsert, so a
    // merge that moved no entity page still invalidates the target's grouping.
    tx.execute(
        "INSERT INTO space_graph_state
             (space, graph_generation, grouping_generation, published_generation, dirty)
         VALUES (?1, 0, 1, NULL, 1)
         ON CONFLICT(space) DO UPDATE SET
             grouping_generation = grouping_generation + 1,
             dirty = 1",
        libsql::params![target_name],
    )
    .await
    .map_err(|e| WenlanError::VectorDb(format!("space merge dirty target: {e}")))?;
    Ok(())
}

async fn has_rows_naming(
    tx: &libsql::Transaction,
    table: &str,
    space: &str,
) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(
            &format!("SELECT 1 FROM {table} WHERE space = ?1 LIMIT 1"),
            libsql::params![space],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("space merge inspect {table}: {e}")))?;
    Ok(rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("space merge inspect {table}: {e}")))?
        .is_some())
}

/// Every table the closure covers, for the teeth.
#[cfg(test)]
pub(super) fn closed_tables() -> Vec<&'static str> {
    CLOSURE.iter().map(|(table, _)| *table).collect()
}

/// Permanent M6 rows for which a destination-name orphan refuses the rename
/// instead of being silently discarded.
#[cfg(test)]
pub(super) fn permanent_tables() -> Vec<&'static str> {
    PERMANENT_GENESIS_TABLES.to_vec()
}

/// Permanent M6 rows a merge leaves under the source name.
#[cfg(test)]
pub(super) fn space_id_bound_tables() -> Vec<&'static str> {
    SPACE_ID_BOUND_TABLES.to_vec()
}
