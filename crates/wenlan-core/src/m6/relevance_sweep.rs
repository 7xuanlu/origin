// SPDX-License-Identifier: Apache-2.0
//! The bounded relevance sweep (PR-C slice C1) and its decay reference.
//!
//! # Why the decay reference exists
//!
//! The estimator cells are decayed sums, not counts:
//!
//! ```text
//! contribution(g) = hub_weight(g) * 0.5 ^ (age_days(g) / 180)
//! ```
//!
//! with `age_days` measured from the group's most recent contributing root's
//! `provenance_roots.created_at`. Decay is therefore a function of *(stored
//! root timestamp, evaluation time)* — and nothing in the schema pinned the
//! evaluation time. S0-91 states the oracle as byte equality of a normalized
//! snapshot "at a fixed relevance generation" and S0-92 fixes accumulation
//! order and 9-dp rounding to make that equality exact, but neither pins the
//! *instant*. An incremental update that decays a touched pair to `unixepoch()`
//! at T₁ and a full recompute at T₂ > T₁ produce different cells for every
//! untouched pair, so the oracle would fail continuously for a reason that is
//! not a bug — and fail *later*, looking exactly like an incremental defect.
//!
//! `space_graph_state` cannot supply the pin: it carries no timestamp.
//!
//! **Q1's adjudication: a per-space monotone counter in `m6_counters`, and no
//! migration.** That table is `(space_id, space, name, value >= 0)` with a
//! monotone-on-update trigger, a monotone-on-insert-replace trigger, two
//! identity guards, and a no-delete trigger. A `unixepoch()` stored there is a
//! monotone integer that may never decrease — exactly the safety property a
//! decay reference needs, because a reference that moved backwards would make
//! a pair's decayed weight *increase*.
//!
//! Every incremental update decays to the **stored** reference, never to
//! `unixepoch()`. The reference advances only in a full re-reference pass. The
//! oracle then becomes exact by construction: every row at a given reference
//! decays to the same instant, so incremental and full agree with no tolerance.
//!
//! S0-11 is honored — `unixepoch()` is still evaluated in-statement; it is
//! stored once rather than read per row.

use super::evidence::read_counter;
use crate::WenlanError;

/// Q1's per-space decay reference: the `unixepoch()` at which this space's pair
/// table was last re-referenced.
pub const COUNTER_RELEVANCE_DECAY_REFERENCE: &str = "relevance_decay_reference";

/// The half-life of a root's contribution, in days.
pub const DECAY_HALF_LIFE_DAYS: f64 = 180.0;

const SECONDS_PER_DAY: f64 = 86_400.0;

/// This space's stored decay reference, or `None` before the first pass.
pub async fn decay_reference(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<Option<i64>, WenlanError> {
    read_counter(tx, space_id, COUNTER_RELEVANCE_DECAY_REFERENCE).await
}

/// Advance the reference to now, returning the value in force afterwards.
///
/// `unixepoch()` is evaluated in-statement (S0-11). Two layers keep the
/// reference monotone and they do different jobs: the table's trigger is what
/// makes lowering *impossible* (dropping the `WHERE` guard raises `M6 counter
/// cannot decrease`, verified as a RED control), while the guard is what makes
/// a backwards clock a *no-op instead of an error* — so a stepped clock costs
/// the sweep one idle turn rather than a failed one. Callers must invoke this only from a
/// full re-reference pass that rewrites every pair row for the space in the
/// same transaction; advancing it without that rewrite silently re-dates rows
/// that were never recomputed.
pub async fn advance_decay_reference(
    tx: &libsql::Transaction,
    space_id: &str,
    space: &str,
) -> Result<i64, WenlanError> {
    if decay_reference(tx, space_id).await?.is_none() {
        tx.execute(
            "INSERT INTO m6_counters (space_id, space, name, value)
             VALUES (?1, ?2, ?3, unixepoch())",
            libsql::params![space_id, space, COUNTER_RELEVANCE_DECAY_REFERENCE],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 decay reference create: {error}")))?;
    } else {
        tx.execute(
            "UPDATE m6_counters SET value = unixepoch()
              WHERE space_id = ?1 AND name = ?2 AND unixepoch() > value",
            libsql::params![space_id, COUNTER_RELEVANCE_DECAY_REFERENCE],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 decay reference advance: {error}")))?;
    }

    decay_reference(tx, space_id).await?.ok_or_else(|| {
        WenlanError::VectorDb("m6 decay reference missing after advance".to_string())
    })
}

/// One group's decayed contribution at a fixed reference.
///
/// A root newer than the reference contributes at full weight rather than
/// amplified: a negative age would make `0.5^(age/180)` exceed 1 and inflate
/// the cell above its own hub weight, which no later retraction could undo
/// (`apply_group_eligibility_change` refuses to drive a cell negative). Roots
/// created after the last re-reference are exactly the rows an incremental
/// update is about, so this clamp is on the hot path, not an edge case.
pub fn decayed_contribution(hub_weight: f64, root_created_at: i64, reference: i64) -> f64 {
    if !hub_weight.is_finite() || hub_weight < 0.0 {
        return 0.0;
    }
    let age_days = ((reference - root_created_at) as f64 / SECONDS_PER_DAY).max(0.0);
    hub_weight * 0.5_f64.powf(age_days / DECAY_HALF_LIFE_DAYS)
}
