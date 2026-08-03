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
use super::independence::ELIGIBLE_EVIDENCE_PREDICATE;
use super::relevance::{PairStatsValues, MAX_NEIGHBORS_PER_ENDPOINT};
use crate::WenlanError;
use std::collections::BTreeMap;

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

/// `R-HUB-WEIGHT`: `min(1, 64/d)`, where `d` is the pages the group touches.
///
/// Reference points from the contract's §3 table: `d=32 → 1.0`, `d=64 → 1.0`,
/// `d=128 → 0.5`, `d=5000 → 0.0128`. The last is the one G6.10 gates — a hub
/// that touches everything must not contribute like a group that touches two
/// things, or one over-connected document dominates every pair in the space.
///
/// `d` here is the **true** degree, counted before the top-64 cut. Weighting by
/// the post-cut count would make every hub weigh `64/64 = 1.0` — capping the
/// selection and the weight with one number, which is exactly the G6.7-vs-G6.10
/// split: those are two caps and each needs its own.
pub fn hub_weight(pages_touched: i64) -> f64 {
    if pages_touched <= 0 {
        // A group touching nothing forms no pairs, so this weight is never
        // multiplied into a cell. Zero rather than one so that if it ever is,
        // it adds nothing instead of a full unit of phantom support.
        return 0.0;
    }
    if pages_touched <= MAX_NEIGHBORS_PER_ENDPOINT as i64 {
        return 1.0;
    }
    MAX_NEIGHBORS_PER_ENDPOINT as f64 / pages_touched as f64
}

/// One independence group's bounded support within a space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupSupport {
    pub group_id: String,
    /// Pages the group touches, counted **before** the top-64 cut, so
    /// [`hub_weight`] sees the real degree.
    pub degree: i64,
    /// The group's most recent contributing root's `created_at`, which is what
    /// §2.1 measures `age_days` from.
    pub newest_root_created_at: i64,
    /// At most 64 pages, ordered `(support_recency DESC, page_id ASC)` per
    /// S0-89. Deduplicated by the `GROUP BY`, so a page supported by several
    /// roots of the group appears once.
    pub pages: Vec<String>,
}

impl GroupSupport {
    /// Whether the top-64 cut dropped pages this group actually touches.
    ///
    /// S0-90's rule for candidate sets — a truncated set is recorded as
    /// truncated — read across to hub selection: `degree > 64` means the pair
    /// set is a deterministic sample, not the whole group.
    pub fn truncated(&self) -> bool {
        self.degree > MAX_NEIGHBORS_PER_ENDPOINT as i64
    }
}

/// Read one group's top-64 supported pages, its true degree, and its decay age.
///
/// One statement, at most 64 decoded rows. `COUNT(*) OVER ()` and
/// `MAX(…) OVER ()` are evaluated over the full grouped set *before* the
/// `LIMIT`, which is what lets a single query return a bounded page list beside
/// an unbounded-degree count. Computing the degree from the returned rows
/// instead would silently read every hub as `d=64` — the G6.10 break.
///
/// The eligibility predicate is [`ELIGIBLE_EVIDENCE_PREDICATE`], shared verbatim
/// with D1's count, so a group cannot be independent enough to clear the floor
/// while contributing nothing here.
pub async fn group_support(
    tx: &libsql::Transaction,
    space: &str,
    group_id: &str,
) -> Result<Option<GroupSupport>, WenlanError> {
    // An edge with a page on both ends supports both, so the page column is a
    // UNION ALL over the two endpoint positions rather than a CASE, which would
    // silently keep only the source side.
    let sql = format!(
        "WITH page_edges AS (
             SELECT e.*, e.src_id AS page_id FROM edges e WHERE e.src_kind = 'page'
             UNION ALL
             SELECT e.*, e.dst_id AS page_id FROM edges e WHERE e.dst_kind = 'page'
         ),
         support AS (
             SELECT e.page_id AS page_id, MAX(r.created_at) AS support_recency
               FROM page_edges e
               JOIN provenance_roots r ON r.root_id = e.root_id
              WHERE e.space = ?1
                AND r.independence_group_id = ?2
                AND {ELIGIBLE_EVIDENCE_PREDICATE}
              GROUP BY e.page_id
         )
         SELECT page_id,
                COUNT(*) OVER ()            AS degree,
                MAX(support_recency) OVER () AS newest_root_created_at
           FROM support
          ORDER BY support_recency DESC, page_id ASC
          LIMIT {MAX_NEIGHBORS_PER_ENDPOINT}"
    );

    let mut rows = tx
        .query(&sql, libsql::params![space, group_id])
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 group support: {error}")))?;

    let mut pages = Vec::new();
    let mut degree = 0i64;
    let mut newest_root_created_at = 0i64;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 group support row: {error}")))?
    {
        pages.push(
            row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("m6 group support page: {error}"))
            })?,
        );
        degree = row
            .get::<i64>(1)
            .map_err(|error| WenlanError::VectorDb(format!("m6 group support degree: {error}")))?;
        newest_root_created_at = row
            .get::<i64>(2)
            .map_err(|error| WenlanError::VectorDb(format!("m6 group support age: {error}")))?;
    }

    if pages.is_empty() {
        return Ok(None);
    }
    Ok(Some(GroupSupport {
        group_id: group_id.to_string(),
        degree,
        newest_root_created_at,
        pages,
    }))
}

/// Every eligible group in the space, each with its bounded page set, ordered
/// `independence_group_id ASC`.
///
/// The order is S0-92's, and it is produced by the query rather than sorted
/// afterwards so there is one place it can be wrong. `ROW_NUMBER()` applies
/// S0-89's top-64 per group while `COUNT(*) OVER (PARTITION BY …)` reports each
/// group's true degree, the same split [`group_support`] makes for one group.
///
/// This is the **full** side of the oracle and is deliberately unbounded in
/// group count: it is the re-reference pass, not the bounded slice. The ≤512
/// materialization budget applies to a route evaluation.
pub async fn eligible_groups(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<Vec<GroupSupport>, WenlanError> {
    let sql = format!(
        "WITH page_edges AS (
             SELECT e.*, e.src_id AS page_id FROM edges e WHERE e.src_kind = 'page'
             UNION ALL
             SELECT e.*, e.dst_id AS page_id FROM edges e WHERE e.dst_kind = 'page'
         ),
         support AS (
             SELECT r.independence_group_id AS gid,
                    e.page_id                AS page_id,
                    MAX(r.created_at)        AS support_recency
               FROM page_edges e
               JOIN provenance_roots r ON r.root_id = e.root_id
              WHERE e.space = ?1
                AND {ELIGIBLE_EVIDENCE_PREDICATE}
              GROUP BY gid, e.page_id
         ),
         ranked AS (
             SELECT gid, page_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY gid ORDER BY support_recency DESC, page_id ASC
                    )                                        AS rank_in_group,
                    COUNT(*)   OVER (PARTITION BY gid)       AS degree,
                    MAX(support_recency) OVER (PARTITION BY gid) AS newest
               FROM support
         )
         SELECT gid, page_id, degree, newest
           FROM ranked
          WHERE rank_in_group <= {MAX_NEIGHBORS_PER_ENDPOINT}
          ORDER BY gid ASC, rank_in_group ASC"
    );

    let mut rows = tx
        .query(&sql, libsql::params![space])
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 eligible groups: {error}")))?;

    let mut groups: Vec<GroupSupport> = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 eligible groups row: {error}")))?
    {
        let gid = row
            .get::<String>(0)
            .map_err(|error| WenlanError::VectorDb(format!("m6 eligible groups gid: {error}")))?;
        let page = row
            .get::<String>(1)
            .map_err(|error| WenlanError::VectorDb(format!("m6 eligible groups page: {error}")))?;
        let degree = row.get::<i64>(2).map_err(|error| {
            WenlanError::VectorDb(format!("m6 eligible groups degree: {error}"))
        })?;
        let newest = row
            .get::<i64>(3)
            .map_err(|error| WenlanError::VectorDb(format!("m6 eligible groups age: {error}")))?;

        match groups.last_mut() {
            Some(last) if last.group_id == gid => last.pages.push(page),
            _ => groups.push(GroupSupport {
                group_id: gid,
                degree,
                newest_root_created_at: newest,
                pages: vec![page],
            }),
        }
    }
    Ok(groups)
}

/// The whole space's pair table at a fixed decay reference — the **full** side
/// of S0-91's oracle.
///
/// # Why `n00` is derived rather than accumulated
///
/// The artifacts define `n00` as "groups supporting neither" and store it as a
/// column, but never say how it is *maintained*. Accumulating it per pair makes
/// the incremental path unbounded: retracting one group changes "supports
/// neither" for **every** pair in the space, not only the pairs that group
/// touches — so an implementation that stores `n00` as an independent sum must
/// rewrite the entire table on every eligibility change, and S0-91's negative
/// control (the incremental path stayed within its row-visit bound) could never
/// pass.
///
/// It does not have to be independent. §2.2 uses `n00` only through
/// `Ñ = ñ11 + ñ10 + ñ01 + ñ00 = N + 2.0`, and `N` — the total decayed mass of
/// the space's eligible groups — is a **per-space** scalar, identical for every
/// pair. Every eligible group falls in exactly one of the four cells, so
///
/// ```text
/// n00(A,B) = N - n11(A,B) - n10(A,B) - n01(A,B)
/// ```
///
/// is an identity, not an approximation. `n00` is therefore derived data that
/// happens to be materialized, which keeps the incremental path bounded to the
/// pairs a group actually forms. Deriving it here — in the one function every
/// consumer goes through — is what makes the stored column a cache rather than
/// a second source of truth.
///
/// Accumulation is in `independence_group_id ASC` (S0-92), which `groups`
/// already carries from [`eligible_groups`]; `n10`/`n01`/`n00` are then single
/// subtractions off deterministic sums, so the whole table is reproducible.
pub fn recompute_pair_stats(
    groups: &[GroupSupport],
    reference: i64,
) -> BTreeMap<(String, String), PairStatsValues> {
    let mut total_mass = 0.0_f64;
    let mut page_mass: BTreeMap<&str, f64> = BTreeMap::new();
    // n11 and the undecayed distinct-group count, accumulated together so a
    // pair can never gain decayed co-support without gaining a group.
    let mut co_support: BTreeMap<(&str, &str), (f64, i64)> = BTreeMap::new();

    for group in groups {
        let contribution = decayed_contribution(
            hub_weight(group.degree),
            group.newest_root_created_at,
            reference,
        );
        total_mass += contribution;
        for page in &group.pages {
            *page_mass.entry(page.as_str()).or_insert(0.0) += contribution;
        }

        // `page_a < page_b` is the table's CHECK, so the pair key is built from
        // a lexicographically sorted copy rather than the recency order the
        // selection returned.
        let mut ordered: Vec<&str> = group.pages.iter().map(String::as_str).collect();
        ordered.sort_unstable();
        for (index, page_a) in ordered.iter().enumerate() {
            for page_b in &ordered[index + 1..] {
                let cell = co_support.entry((page_a, page_b)).or_insert((0.0, 0));
                cell.0 += contribution;
                cell.1 += 1;
            }
        }
    }

    co_support
        .into_iter()
        .map(|((page_a, page_b), (n11, distinct_group_count))| {
            let mass_a = page_mass.get(page_a).copied().unwrap_or(0.0);
            let mass_b = page_mass.get(page_b).copied().unwrap_or(0.0);
            // Each cell is clamped at zero: the table's CHECKs forbid a
            // negative, and the only way to reach one here is float error in
            // the last bits of a subtraction that is exact in real arithmetic.
            let values = PairStatsValues {
                n11,
                n10: (mass_a - n11).max(0.0),
                n01: (mass_b - n11).max(0.0),
                n00: (total_mass - mass_a - mass_b + n11).max(0.0),
                distinct_group_count,
            };
            ((page_a.to_string(), page_b.to_string()), values)
        })
        .collect()
}
