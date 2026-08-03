// SPDX-License-Identifier: Apache-2.0
//! M6's seam onto M4's community durable-reader gate.
//!
//! Two of the four D2 signals — `evidence_cluster` and `community_overview` —
//! read the community partition, so they must know whether the durable M4
//! substrate (`communities` / `community_members`) has been proven for every
//! relevant space, or whether the legacy `entities.community_id` is still the
//! only trustworthy source. That question already has exactly one right
//! answer, and `db::community_reader_durable_gate_sql` is it.
//!
//! **What is single-sourced here is the SQL predicate, not the plumbing.**
//! This module rebuilds the six lines that run a `SELECT` and read column 0,
//! because M4's own `community_reader_uses_durable` takes a `libsql::Connection`
//! while every M6 read runs inside a `libsql::Transaction`. Copying the
//! predicate itself would be the bug — one gate drifting from the other is
//! precisely the failure the central gate exists to prevent — so the predicate
//! crosses the boundary and nothing else does.
//!
//! Fail-closed, mirroring M4: an unknown consumer makes the gate SQL `0=1`, and
//! a query error reads as "not durable" rather than propagating. There is no
//! state in which uncertainty is reported as a durable partition.

use crate::db::{community_reader_durable_gate_sql, COMMUNITY_GENESIS_CANDIDATES_CONSUMER};

/// Is M4's durable community partition proven for M6's genesis reader?
///
/// The value PR-B's signals take as `community_partition_durable`. It is a
/// single global bool rather than a per-space one on purpose: the gate SQL
/// takes no space parameter — space appears in it only inside universally
/// quantified subqueries ("every relevant space has a clean, current receipt")
/// — so the predicate is already all-or-nothing across the store, and a
/// per-space signature would imply a discrimination the gate cannot make.
pub(crate) async fn partition_is_durable(tx: &libsql::Transaction) -> bool {
    let sql = format!(
        "SELECT {}",
        community_reader_durable_gate_sql(COMMUNITY_GENESIS_CANDIDATES_CONSUMER)
    );
    let result = async {
        let mut rows = tx.query(&sql, ()).await?;
        Ok::<_, libsql::Error>(rows.next().await?.and_then(|row| row.get::<i64>(0).ok()) == Some(1))
    }
    .await;
    matches!(result, Ok(true))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::m6::genesis_test_support::GenesisDb;

    /// The seam's contract when the gate cannot be evaluated. The M6 fixture
    /// installs the genesis and community substrates but none of the gate's
    /// control-plane tables (`community_reader_cutover` and friends), so the
    /// query errors — and an error must read as "not durable", never as durable
    /// and never as a panic that takes the turn down.
    ///
    /// Deliberately runs on the shared fixture rather than opening its own
    /// libSQL handle: an ad-hoc handle would be a tracked R4 standalone origin,
    /// and it would prove nothing this does not.
    #[tokio::test]
    async fn an_unreadable_gate_is_not_a_durable_partition() {
        let db = GenesisDb::new().await;
        let tx = db.tx().await;

        assert!(
            !partition_is_durable(&tx).await,
            "a gate that cannot be evaluated must report the partition as not durable"
        );
    }
}
