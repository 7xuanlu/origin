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

/// `R-BUDGET-QUERIES` (S0-95 instrument 2): statements per evaluation.
pub const MAX_QUERIES_PER_EVALUATION: u32 = 4;

/// `R-BUDGET-ROWS` (S0-95 instrument 1): rows *materialized* per evaluation.
/// D9's word is "materializes", so this counts rows decoded from a cursor.
pub const MAX_ROWS_PER_EVALUATION: u32 = 512;

/// How many stale endpoints one turn *looks at* before packing.
///
/// Not how many it sweeps: the turn takes the longest prefix of this window
/// whose projected fan-out fits what is left of `R-BUDGET-ROWS`. A fixed
/// swept-count cannot hold that bound — groups-per-page has a tail, and the
/// benchmark found a breach at turn 102 of the frozen corpus with a slice of
/// four. The bound has to be enforced by construction or it is not a bound.
///
/// **Not** [`super::relevance::MAX_CANDIDATE_ENDPOINTS`]. That constant is
/// `R-CAND-CAP`, D9's cap
/// on the *route evaluation's* query (c) — a read over `m6_adjacency` whose
/// materialization is 32 rows by the precomputed table's shape (S0-96: "the
/// route evaluation never touches `edges` directly"). This sweep is the
/// adjacency *rebuild*, which does read `edges`, and there one endpoint drags
/// in every group touching it (~19 on the S0-97 corpus) and every page those
/// groups touch. Borrowing 32 here spends the same `R-BUDGET-ROWS` budget on
/// roughly sixty times the rows.
///
/// Set from the benchmark, never the other way round: S0-99 forbids answering
/// a budget breach by tuning a constant, and 512 is the frozen one. This is
/// the parameter the contract left free, so this is the one that moves.
pub const SWEEP_CANDIDATE_WINDOW: usize = 8;

/// The two CI-gated budget instruments, enforced at the read rather than
/// measured after it.
///
/// S0-95 is explicit that these are runtime counters or they are nothing: a
/// textual `LIMIT 512` on an unindexed predicate visits the whole table and
/// still reports 512 rows, and `EXPLAIN QUERY PLAN` reports the plan, not the
/// work. The other two instruments (`FULLSCAN_STEP`, `SQLITE_SCANSTAT_NVISIT`)
/// need a build this crate does not produce; §8.7 records them as declared
/// deferrals, discharged by the local bench receipt.
///
/// Exceeding a bound is an error, not a truncation — a sweep that silently did
/// less work would look identical to one that stayed in budget.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct QueryBudget {
    pub queries: u32,
    pub rows: u32,
}

impl QueryBudget {
    /// Charge one statement, before it is issued.
    pub fn charge_query(&mut self) -> Result<(), WenlanError> {
        self.queries += 1;
        if self.queries > MAX_QUERIES_PER_EVALUATION {
            return Err(WenlanError::VectorDb(format!(
                "M6 relevance evaluation issued {} queries; cap is {MAX_QUERIES_PER_EVALUATION}",
                self.queries
            )));
        }
        Ok(())
    }

    /// Charge one decoded row, as it is decoded.
    pub fn charge_row(&mut self) -> Result<(), WenlanError> {
        self.rows += 1;
        if self.rows > MAX_ROWS_PER_EVALUATION {
            return Err(WenlanError::VectorDb(format!(
                "M6 relevance evaluation materialized {} rows; cap is {MAX_ROWS_PER_EVALUATION}",
                self.rows
            )));
        }
        Ok(())
    }
}

/// This space's stored decay reference, or `None` before the first pass.
/// A stale endpoint, with the number of rows sweeping it would materialize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateEndpoint {
    pub page_id: String,
    /// The `(group, page)` rows `groups_touching` decodes for this endpoint
    /// alone, computed by the same top-64 cap that query applies.
    pub projected_rows: i64,
}

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

/// Pages this sweep should re-reference, capped at 32 **at the query** (S0-90).
///
/// Staleness is `m6_pair_stats.updated_generation` behind the current grouping
/// generation, which is what that column is for. The test is **any** row naming
/// the page being behind, not the absence of a current one: a turn writes every
/// pair incident to its endpoints, so a page picks up current rows as a *co*-
/// page of somebody else's sweep long before its own remaining pairs are
/// re-derived. Reading one current row as "this page is done" retires it early
/// and strands the rest of its pairs at the old generation forever — a silent
/// loss, because the table still looks fully populated.
///
/// A page with no rows at all is therefore not a candidate. That is the same
/// rule, not an exception: nothing about it is behind. Pages first acquire rows
/// in the full re-reference pass, and a page with no eligible support never
/// acquires any — which is what stops the sweep spinning on pages that can
/// never produce a pair.
///
/// The cap is in the `LIMIT`, not a `Vec::truncate` afterwards: G6.8's break is
/// exactly a post-hoc truncate, which leaves the returned count passing while
/// the query still reads the whole table. One extra row is requested so
/// truncation is *observable* without a second counting query — S0-90 requires
/// a truncated candidate set to be recorded as truncated, and a set that
/// silently filled its cap is a decision whose inputs cannot be reconstructed.
pub async fn candidate_endpoints(
    tx: &libsql::Transaction,
    space: &str,
    grouping_generation: i64,
    budget: &mut QueryBudget,
) -> Result<(Vec<CandidateEndpoint>, bool), WenlanError> {
    budget.charge_query()?;
    let sql = format!(
        "WITH page_edges AS (
             SELECT e.*, e.src_id AS page_id FROM edges e WHERE e.src_kind = 'page'
             UNION ALL
             SELECT e.*, e.dst_id AS page_id FROM edges e WHERE e.dst_kind = 'page'
         ),
         support AS (
             SELECT r.independence_group_id AS gid, e.page_id AS page_id
               FROM page_edges e
               JOIN provenance_roots r ON r.root_id = e.root_id
              WHERE e.space = ?1
                AND {ELIGIBLE_EVIDENCE_PREDICATE}
              GROUP BY gid, e.page_id
         ),
         group_size AS (
             -- Capped exactly where `groups_touching` caps, so the projection
             -- is that query's row count and not an estimate of it.
             SELECT gid, MIN(COUNT(*), {MAX_NEIGHBORS_PER_ENDPOINT}) AS pages
               FROM support GROUP BY gid
         ),
         stale AS (
             SELECT p.id AS id FROM pages p
              WHERE p.space = ?1
                AND lower(p.title) <> 'overview'
                AND EXISTS (
                    SELECT 1 FROM m6_pair_stats s
                     WHERE s.space = ?1 AND s.updated_generation < ?2
                       AND (s.page_a = p.id OR s.page_b = p.id)
                )
              ORDER BY p.id ASC
              LIMIT ?3
         )
         SELECT st.id, COALESCE(SUM(gs.pages), 0)
           FROM stale st
           LEFT JOIN support s  ON s.page_id = st.id
           LEFT JOIN group_size gs ON gs.gid = s.gid
          GROUP BY st.id
          ORDER BY st.id ASC"
    );
    let mut rows = tx
        .query(
            &sql,
            libsql::params![
                space,
                grouping_generation,
                SWEEP_CANDIDATE_WINDOW as i64 + 1
            ],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 candidate endpoints: {error}")))?;

    let mut endpoints = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 candidate endpoints row: {error}")))?
    {
        budget.charge_row()?;
        endpoints.push(CandidateEndpoint {
            page_id: row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("m6 candidate endpoints decode: {error}"))
            })?,
            projected_rows: row.get::<i64>(1).map_err(|error| {
                WenlanError::VectorDb(format!("m6 candidate projection decode: {error}"))
            })?,
        });
    }

    let truncated = endpoints.len() > SWEEP_CANDIDATE_WINDOW;
    endpoints.truncate(SWEEP_CANDIDATE_WINDOW);
    Ok((endpoints, truncated))
}

/// The longest prefix of `candidates` whose projected cost fits `budget`.
///
/// Cost is counted twice per projected row: once for the `groups_touching`
/// row itself, once for the `pinned_space_mass` row that row's page can add.
/// The union of two endpoints' groups is at most the sum, and the distinct
/// pages are at most the rows, so this can only over-reserve — which is the
/// direction a budget may err in.
///
/// Always returns at least one endpoint when any exist: a turn that reserves
/// its way to zero work makes no progress and the sweep never converges.
pub fn pack_within_budget(candidates: &[CandidateEndpoint], budget: &QueryBudget) -> Vec<String> {
    let remaining = i64::from(MAX_ROWS_PER_EVALUATION) - i64::from(budget.rows);
    let mut chosen = Vec::new();
    let mut reserved = 0i64;
    for candidate in candidates {
        let cost = candidate.projected_rows.saturating_mul(2);
        if !chosen.is_empty() && reserved + cost > remaining {
            break;
        }
        chosen.push(candidate.page_id.clone());
        reserved += cost;
    }
    chosen
}

/// The eligible groups supporting **any** of `endpoints`, each with its bounded
/// page set, ordered `independence_group_id ASC` (S0-92).
///
/// This is the bounded counterpart of [`eligible_groups`]: same shape, same
/// shared eligibility predicate, restricted to the groups a turn can affect.
pub async fn groups_touching(
    tx: &libsql::Transaction,
    space: &str,
    endpoints: &[String],
    budget: &mut QueryBudget,
) -> Result<Vec<GroupSupport>, WenlanError> {
    if endpoints.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = (2..2 + endpoints.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
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
         touched AS (
             SELECT DISTINCT gid FROM support WHERE page_id IN ({placeholders})
         ),
         ranked AS (
             SELECT s.gid, s.page_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.gid ORDER BY s.support_recency DESC, s.page_id ASC
                    )                                              AS rank_in_group,
                    COUNT(*) OVER (PARTITION BY s.gid)             AS degree,
                    MAX(s.support_recency) OVER (PARTITION BY s.gid) AS newest
               FROM support s
               JOIN touched t ON t.gid = s.gid
         )
         SELECT gid, page_id, degree, newest
           FROM ranked
          WHERE rank_in_group <= {MAX_NEIGHBORS_PER_ENDPOINT}
          ORDER BY gid ASC, rank_in_group ASC"
    );

    let mut params: Vec<libsql::Value> = Vec::with_capacity(1 + endpoints.len());
    params.push(libsql::Value::Text(space.to_string()));
    params.extend(
        endpoints
            .iter()
            .map(|page| libsql::Value::Text(page.clone())),
    );

    budget.charge_query()?;
    let mut rows = tx
        .query(&sql, libsql::params_from_iter(params))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 groups touching: {error}")))?;

    let mut groups: Vec<GroupSupport> = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 groups touching row: {error}")))?
    {
        budget.charge_row()?;
        let gid = row
            .get::<String>(0)
            .map_err(|error| WenlanError::VectorDb(format!("m6 groups touching gid: {error}")))?;
        let page = row
            .get::<String>(1)
            .map_err(|error| WenlanError::VectorDb(format!("m6 groups touching page: {error}")))?;
        let degree = row.get::<i64>(2).map_err(|error| {
            WenlanError::VectorDb(format!("m6 groups touching degree: {error}"))
        })?;
        let newest = row
            .get::<i64>(3)
            .map_err(|error| WenlanError::VectorDb(format!("m6 groups touching age: {error}")))?;
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

/// The decayed marginals a pair table's cells are measured against: `total` is
/// `N`, the space's whole eligible mass, and `per_page` each page's share of it.
///
/// # Why `N` is pinned rather than recomputed
///
/// `n00(A,B) = N - mass(A) - mass(B) + n11(A,B)`, so `N` reaches every cell in
/// the space. Activating one root moves `N`, which moves `n00` for **every**
/// stored pair — including every pair that root does not touch. An
/// implementation that keeps `n00` exact against a live `N` therefore has to
/// rewrite the whole table on each eligibility change, which S0-91's negative
/// control (the incremental path stayed within its row-visit bound) exists to
/// forbid. Exact-and-live and bounded-incremental cannot both hold; that is a
/// property of the definition, not of any particular implementation.
///
/// The resolution is the one Q1 already applies to the decay reference: `N` is
/// **part of the same pin**. It is established by the full re-reference pass
/// (`RelevanceTurn::Rereferenced`) and held constant until the next one, so a
/// bounded turn's rows agree exactly with a full recomputation *at that pin*.
/// Between passes `N` is stale by a bounded, queryable amount, exactly as the
/// decay is, and for the same reason.
///
/// The marginals live in `m6_space_mass` and `m6_page_mass` rather than being
/// recovered from a pair row's cells. They *are* recoverable — `N` is
/// `n11 + n10 + n01 + n00` and `mass(A)` is `n11 + n10`, the identity gated by
/// `the_four_cells_always_sum_to_the_spaces_total_mass` — and C1 read them
/// back that way at first. Dedicated tables replaced it because `n00` is not
/// stored any more: deriving it needs the marginals, so recovering the
/// marginals from it would be circular. The identity still holds over what a
/// reader assembles, which is why that test still gates it.
///
/// The spec left this open — Q1 pins the *instant* and never says `N` belongs
/// to the same pin — so this is C1 adjudicating it, and the oracle is what
/// holds the adjudication honest.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SpaceMass {
    pub total: f64,
    pub per_page: BTreeMap<String, f64>,
}

impl SpaceMass {
    /// Sum the marginals from a group set. Correct **only** when `groups` is
    /// every eligible group in the space, which is why a bounded turn calls
    /// [`pinned_space_mass`] instead of reusing the groups it scanned.
    pub fn from_groups(groups: &[GroupSupport], reference: i64) -> SpaceMass {
        let mut mass = SpaceMass::default();
        for group in groups {
            let contribution = decayed_contribution(
                hub_weight(group.degree),
                group.newest_root_created_at,
                reference,
            );
            mass.total += contribution;
            for page in &group.pages {
                *mass.per_page.entry(page.clone()).or_insert(0.0) += contribution;
            }
        }
        mass
    }

    /// A page with no eligible support has no mass, so an absent key reads as
    /// zero rather than erroring — the same value the sums would have produced.
    pub fn page(&self, page: &str) -> f64 {
        self.per_page.get(page).copied().unwrap_or(0.0)
    }
}

/// Read the pinned `N` and the named pages' marginals out of `m6_space_mass`
/// and `m6_page_mass`.
///
/// One statement for as many pages as were asked for, plus the space total as
/// a correlated scalar. An empty result means the space has no mass rows,
/// which is the state the full re-reference pass handles — a bounded turn
/// reaching it has nothing to measure against and writes nothing.
pub async fn pinned_space_mass(
    tx: &libsql::Transaction,
    space: &str,
    pages: &[String],
    budget: &mut QueryBudget,
) -> Result<SpaceMass, WenlanError> {
    if pages.is_empty() {
        return Ok(SpaceMass::default());
    }
    budget.charge_query()?;

    let placeholders = (2..2 + pages.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    // One row per requested page plus the space total, so the whole marginal
    // set costs a single query and the reader never has to reconstruct `N`
    // from a pair row's four cells.
    let sql = format!(
        "SELECT m.page_id, m.mass, (SELECT total FROM m6_space_mass WHERE space = ?1)
           FROM m6_page_mass m
          WHERE m.space = ?1
            AND m.page_id IN ({placeholders})
          ORDER BY m.page_id"
    );

    let mut params = Vec::with_capacity(1 + pages.len());
    params.push(libsql::Value::Text(space.to_string()));
    params.extend(pages.iter().map(|page| libsql::Value::Text(page.clone())));

    let mut rows = tx
        .query(&sql, libsql::params_from_iter(params))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pinned mass: {error}")))?;

    let mut mass = SpaceMass::default();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pinned mass row: {error}")))?
    {
        budget.charge_row()?;
        let page = row
            .get::<String>(0)
            .map_err(|error| WenlanError::VectorDb(format!("m6 pinned mass page: {error}")))?;
        let page_mass = row
            .get::<f64>(1)
            .map_err(|error| WenlanError::VectorDb(format!("m6 pinned mass value: {error}")))?;
        mass.total = row
            .get::<f64>(2)
            .map_err(|error| WenlanError::VectorDb(format!("m6 pinned mass total: {error}")))?;
        mass.per_page.insert(page, page_mass);
    }
    Ok(mass)
}

/// Persist the pinned marginals the pair table no longer carries.
///
/// Written only by the full re-reference pass and by a mutation, both of
/// which know the whole space's mass. A bounded sweep turn reads these and
/// never writes them: it sees only the groups touching its own candidates, so
/// any marginal it computed would be a fragment of the real one.
pub async fn write_space_mass(
    tx: &libsql::Transaction,
    space: &str,
    mass: &SpaceMass,
) -> Result<usize, WenlanError> {
    tx.execute(
        "INSERT INTO m6_space_mass (space, total) VALUES (?1, ?2)
         ON CONFLICT(space) DO UPDATE SET total = excluded.total",
        libsql::params![space, mass.total],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 space mass write: {error}")))?;

    for (page, page_mass) in &mass.per_page {
        tx.execute(
            "INSERT INTO m6_page_mass (space, page_id, mass) VALUES (?1, ?2, ?3)
             ON CONFLICT(space, page_id) DO UPDATE SET mass = excluded.mass",
            libsql::params![space, page.clone(), *page_mass],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 page mass write: {error}")))?;
    }
    Ok(mass.per_page.len())
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
///
/// # Why the marginals arrive as an argument
///
/// Only `n11` and the distinct-group count are properties of the *pair*. A
/// group contributes to them exactly when it supports both pages, so the groups
/// touching one endpoint are all the evidence they need. `mass(A)`, `mass(B)`
/// and `N` are not pair-local — they range over every eligible group in the
/// space — and a bounded turn scans only the groups touching its ≤32 candidate
/// endpoints. Summing them over that subset gives any page whose groups were
/// only partly scanned too small a marginal, and the error lands in
/// `n10`/`n01`/`n00` where S0-91's digest catches it.
///
/// So [`SpaceMass`] is supplied by the caller, and each path gets it the way it
/// can afford: the full re-reference pass sums the groups it already holds
/// ([`SpaceMass::from_groups`]), and a bounded turn reads the pinned values back
/// out of the table ([`pinned_space_mass`]). That both routes agree is exactly
/// what the oracle tests, which is why the marginals are not quietly re-derived
/// here from whatever `groups` happens to contain.
pub fn recompute_pair_stats(
    groups: &[GroupSupport],
    reference: i64,
    mass: &SpaceMass,
) -> BTreeMap<(String, String), PairStatsValues> {
    let total_mass = mass.total;
    // n11 and the undecayed distinct-group count, accumulated together so a
    // pair can never gain decayed co-support without gaining a group.
    let mut co_support: BTreeMap<(&str, &str), (f64, i64)> = BTreeMap::new();

    for group in groups {
        let contribution = decayed_contribution(
            hub_weight(group.degree),
            group.newest_root_created_at,
            reference,
        );

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
            let mass_a = mass.page(page_a);
            let mass_b = mass.page(page_b);
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

/// What the full re-reference pass did. No [`QueryBudget`]: this is the pass
/// the incremental bound explicitly does not apply to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevanceRereference {
    pub groups_read: usize,
    pub pages_referenced: usize,
    pub pairs_written: usize,
    pub adjacency_rows_written: usize,
    pub decay_reference: i64,
}

/// What one relevance turn did, in the shape the maintenance driver reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelevanceTurn {
    /// The full re-reference pass, which establishes the pin every bounded turn
    /// afterwards measures against. Its own variant rather than a flag on
    /// [`RelevanceSweep`], so the incremental row bound cannot be asserted
    /// against it by accident — it is the one turn the bound does not describe.
    Rereferenced(RelevanceRereference),
    /// Nothing was stale; the sweep found no work.
    Idle,
    /// Another holder owns `(relevance, space, grouping_generation)` — C2's
    /// `LeaseHeld` signal, and **not** work.
    LeaseHeld,
    /// A slice ran. Carries S0-90's truncation record and S0-95's counters.
    Swept(RelevanceSweep),
}

/// The receipt for one swept slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevanceSweep {
    pub endpoints_considered: usize,
    /// S0-90: a candidate set that hit its cap is recorded as truncated, so an
    /// attachment decision made over it can be reconstructed.
    pub endpoints_truncated: bool,
    pub pairs_written: usize,
    /// Pairs incident to a swept endpoint that no longer have any supporting
    /// group. Reported rather than silent: this is the only write in the turn
    /// that removes information.
    pub pairs_removed: usize,
    pub adjacency_rows_written: usize,
    pub decay_reference: i64,
    pub budget: QueryBudget,
}

/// One bounded relevance slice (spec §5.1.2).
///
/// Ordering is the spec's: acquire, select, read, recompute, write, release —
/// with the release in the same transaction as the last write, so a slice that
/// dies mid-write cannot leave the lease released over a half-written table.
///
/// The caller owns the transaction and the commit. Nothing here calls a model:
/// the sweep is four indexed reads and two writes.
pub async fn run_relevance_sweep(
    tx: &libsql::Transaction,
    space_id: &str,
    space: &str,
    grouping_generation: i64,
    attempt: i64,
) -> Result<RelevanceTurn, WenlanError> {
    let token = match super::leases::acquire(
        tx,
        super::leases::LeasePhase::Relevance,
        space,
        grouping_generation,
        attempt,
    )
    .await?
    {
        Some(token) => token,
        None => return Ok(RelevanceTurn::LeaseHeld),
    };

    // A space with no pin yet gets the full re-reference pass, which is the
    // only thing allowed to establish one. Everything after it is bounded.
    let Some(reference) = decay_reference(tx, space_id).await? else {
        let outcome = run_full_rereference(tx, space_id, space, grouping_generation).await?;
        super::leases::release(
            tx,
            super::leases::LeasePhase::Relevance,
            space,
            grouping_generation,
            &token,
        )
        .await?;
        return Ok(outcome);
    };

    let mut budget = QueryBudget::default();
    let (candidates, endpoints_truncated) =
        candidate_endpoints(tx, space, grouping_generation, &mut budget).await?;
    // The window says what is stale; the packing says what fits. Doing this
    // after the candidate query is what lets the reservation be exact rather
    // than a guess made before anything was read.
    let endpoints = pack_within_budget(&candidates, &budget);
    if endpoints.is_empty() {
        super::leases::release(
            tx,
            super::leases::LeasePhase::Relevance,
            space,
            grouping_generation,
            &token,
        )
        .await?;
        return Ok(RelevanceTurn::Idle);
    }

    let groups = groups_touching(tx, space, &endpoints, &mut budget).await?;
    // Every page the scanned groups touch, not just the candidates: a pair is
    // written for each co-occurring page, and its marginal has to be the pinned
    // space-wide one even when that page was not itself a candidate.
    let involved: Vec<String> = groups
        .iter()
        .flat_map(|group| group.pages.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut mass = pinned_space_mass(tx, space, &involved, &mut budget).await?;
    // A support change moves the *page's* marginal but not the space total:
    // `total` sums each eligible group once, so a group that gains or loses a
    // page is still exactly one eligible group. A group leaving the universe
    // altogether is the other case, and it goes through `apply_group_mutation`,
    // which owns the total. That split is what lets a bounded turn refresh
    // marginals at all -- it read every group touching its own endpoints, so
    // those it can compute exactly, and every other page's stays pinned until
    // that page is itself swept.
    refresh_endpoint_mass(tx, space, &endpoints, &groups, reference, &mut mass).await?;
    let stats = recompute_pair_stats(&groups, reference, &mass);

    // Only the pairs this turn computed *completely*. A candidate endpoint's
    // groups are all in `groups`, so every group supporting both pages of a
    // candidate-incident pair was scanned and its cells are exact. A pair whose
    // two pages are merely co-members of one scanned group is not: the groups
    // supporting it through some *other* page were never read, so writing it
    // would store a partial count — and stamp it current, which is worse than
    // leaving it stale.
    let endpoint_set: std::collections::BTreeSet<&str> =
        endpoints.iter().map(String::as_str).collect();
    let complete: BTreeMap<(String, String), PairStatsValues> = stats
        .into_iter()
        .filter(|((page_a, page_b), _)| {
            endpoint_set.contains(page_a.as_str()) || endpoint_set.contains(page_b.as_str())
        })
        .collect();

    // A pair vanishes when its last co-supporting group does, and an upsert
    // cannot say that. The row would keep its stale cells *and* its stale
    // generation, which leaves its endpoints permanently stale: the sweep
    // re-selects them every turn, never advances them, and never reaches
    // `Idle`. That is a livelock, not a wrong number.
    //
    // Delete-then-insert over exactly the pairs this turn owns, the same shape
    // `rewrite_adjacency` uses, and sound for the same reason `complete` is:
    // every group touching a swept endpoint was read this turn, so a stored
    // pair incident to one and absent from `complete` has no support left.
    let pairs_removed = delete_pairs_incident_to(tx, space, &endpoints).await?;
    let pairs_written = write_pair_stats(tx, space, grouping_generation, &complete).await?;
    let adjacency_rows_written = rewrite_adjacency(tx, space, &endpoints, &groups).await?;

    // Same transaction as the last write (§5.1.2 step 6).
    super::leases::release(
        tx,
        super::leases::LeasePhase::Relevance,
        space,
        grouping_generation,
        &token,
    )
    .await?;

    Ok(RelevanceTurn::Swept(RelevanceSweep {
        endpoints_considered: endpoints.len(),
        endpoints_truncated,
        pairs_written,
        pairs_removed,
        adjacency_rows_written,
        decay_reference: reference,
        budget,
    }))
}

/// The full re-reference pass: read every eligible group, advance the decay
/// reference, and rewrite the whole space's pair table against it.
///
/// This is the one deliberately unbounded read in the module. Q1 sanctions it —
/// "the reference advances only in a full re-reference pass, which rewrites
/// every pair row for the space in one transaction" — and it is what makes the
/// pin exist for every bounded turn afterwards. It carries no [`QueryBudget`]
/// for that reason: charging the incremental bound against the pass that
/// establishes the baseline would only invite the bound to be loosened until a
/// full scan fit inside it, which is the failure S0-99 names.
async fn run_full_rereference(
    tx: &libsql::Transaction,
    space_id: &str,
    space: &str,
    grouping_generation: i64,
) -> Result<RelevanceTurn, WenlanError> {
    let groups = eligible_groups(tx, space).await?;
    if groups.is_empty() {
        // No pin is established: a space with no eligible support has no mass
        // to reference, and pinning zero would make the next turn measure real
        // groups against an `N` of nothing.
        return Ok(RelevanceTurn::Idle);
    }

    let reference = advance_decay_reference(tx, space_id, space).await?;
    let mass = SpaceMass::from_groups(&groups, reference);
    write_space_mass(tx, space, &mass).await?;
    let stats = recompute_pair_stats(&groups, reference, &mass);
    let pairs_written = write_pair_stats(tx, space, grouping_generation, &stats).await?;

    let endpoints: Vec<String> = mass.per_page.keys().cloned().collect();
    let adjacency_rows_written = rewrite_adjacency(tx, space, &endpoints, &groups).await?;

    Ok(RelevanceTurn::Rereferenced(RelevanceRereference {
        groups_read: groups.len(),
        pages_referenced: endpoints.len(),
        pairs_written,
        adjacency_rows_written,
        decay_reference: reference,
    }))
}

/// UPSERT the computed cells, stamping the generation they were computed under.
/// Recompute this turn's endpoints' marginals from the groups it read, store
/// them, and fold them into `mass` so the pairs written below use the fresh
/// value rather than the pinned one.
///
/// Exact, not approximate: [`groups_touching`] returns every eligible group
/// touching a swept endpoint, and `mass(p)` is the sum of those groups'
/// contributions. The top-64 page cap applies identically here and in
/// [`SpaceMass::from_groups`], so the two paths truncate the same groups.
async fn refresh_endpoint_mass(
    tx: &libsql::Transaction,
    space: &str,
    endpoints: &[String],
    groups: &[GroupSupport],
    reference: i64,
    mass: &mut SpaceMass,
) -> Result<(), WenlanError> {
    for endpoint in endpoints {
        let refreshed: f64 = groups
            .iter()
            .filter(|group| group.pages.iter().any(|page| page == endpoint))
            .map(|group| {
                decayed_contribution(
                    hub_weight(group.degree),
                    group.newest_root_created_at,
                    reference,
                )
            })
            .sum();
        tx.execute(
            "INSERT INTO m6_page_mass (space, page_id, mass) VALUES (?1, ?2, MAX(0.0, ?3))
             ON CONFLICT(space, page_id) DO UPDATE SET mass = excluded.mass",
            libsql::params![space, endpoint.as_str(), refreshed],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 endpoint mass write: {error}")))?;
        mass.per_page.insert(endpoint.clone(), refreshed);
    }
    Ok(())
}

/// Drop every stored pair incident to one of this turn's endpoints.
///
/// Paired with the `write_pair_stats` that follows it: together they replace
/// the turn's owned slice of the table rather than merging into it. A write,
/// so it is not charged against [`QueryBudget`] -- the budget's two instruments
/// count statements that *decode rows* and the rows they decode, and this
/// statement returns none.
async fn delete_pairs_incident_to(
    tx: &libsql::Transaction,
    space: &str,
    endpoints: &[String],
) -> Result<usize, WenlanError> {
    if endpoints.is_empty() {
        return Ok(0);
    }
    let placeholders = (2..2 + endpoints.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "DELETE FROM m6_pair_stats
          WHERE space = ?1
            AND (page_a IN ({placeholders}) OR page_b IN ({placeholders}))"
    );

    let mut params = Vec::with_capacity(1 + endpoints.len());
    params.push(libsql::Value::Text(space.to_string()));
    params.extend(
        endpoints
            .iter()
            .map(|endpoint| libsql::Value::Text(endpoint.clone())),
    );

    let removed = tx
        .execute(&sql, libsql::params_from_iter(params))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pair stats delete: {error}")))?;
    Ok(removed as usize)
}

async fn write_pair_stats(
    tx: &libsql::Transaction,
    space: &str,
    grouping_generation: i64,
    stats: &BTreeMap<(String, String), PairStatsValues>,
) -> Result<usize, WenlanError> {
    let mut pairs_written = 0usize;
    for ((page_a, page_b), values) in stats {
        tx.execute(
            "INSERT INTO m6_pair_stats
                 (space, page_a, page_b, n11, n10, n01, n00,
                  distinct_group_count, updated_generation)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(space, page_a, page_b) DO UPDATE SET
                 n11 = excluded.n11, n10 = excluded.n10,
                 n01 = excluded.n01, n00 = excluded.n00,
                 distinct_group_count = excluded.distinct_group_count,
                 updated_generation   = excluded.updated_generation",
            libsql::params![
                space,
                page_a.as_str(),
                page_b.as_str(),
                values.n11,
                values.n10,
                values.n01,
                values.n00,
                values.distinct_group_count,
                grouping_generation
            ],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pair stats upsert: {error}")))?;
        pairs_written += 1;
    }
    Ok(pairs_written)
}

/// Rewrite each swept endpoint's `m6_adjacency` rows: ranks 1..=64 ordered
/// `(support_recency DESC, page_id ASC)` per S0-89.
///
/// Delete-then-insert rather than upsert, because rank is part of the primary
/// key: a neighbour that moved from rank 3 to rank 5 would otherwise leave its
/// old row behind and the endpoint would carry two rows for one neighbour,
/// which the `UNIQUE(space, endpoint_kind, endpoint_id, neighbor_id)` index
/// would reject only on the second insert — after the first had already
/// corrupted the ranking.
async fn rewrite_adjacency(
    tx: &libsql::Transaction,
    space: &str,
    endpoints: &[String],
    groups: &[GroupSupport],
) -> Result<usize, WenlanError> {
    let mut written = 0usize;
    for endpoint in endpoints {
        // Neighbours are the pages co-supported with this endpoint, each
        // carrying the most recent group recency that connects them — the same
        // ordering key the top-64 selection uses.
        let mut recency: BTreeMap<&str, i64> = BTreeMap::new();
        for group in groups {
            if !group.pages.iter().any(|page| page == endpoint) {
                continue;
            }
            for page in &group.pages {
                if page == endpoint {
                    continue;
                }
                let entry = recency.entry(page.as_str()).or_insert(i64::MIN);
                *entry = (*entry).max(group.newest_root_created_at);
            }
        }

        tx.execute(
            "DELETE FROM m6_adjacency
              WHERE space = ?1 AND endpoint_kind = 'page' AND endpoint_id = ?2",
            libsql::params![space, endpoint.as_str()],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 adjacency clear: {error}")))?;

        let mut ranked: Vec<(&str, i64)> = recency.into_iter().collect();
        ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
        for (rank, (neighbor, _)) in ranked.iter().take(MAX_NEIGHBORS_PER_ENDPOINT).enumerate() {
            tx.execute(
                "INSERT INTO m6_adjacency
                     (space, endpoint_kind, endpoint_id, neighbor_id, rank)
                 VALUES (?1, 'page', ?2, ?3, ?4)",
                libsql::params![space, endpoint.as_str(), *neighbor, rank as i64 + 1],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m6 adjacency insert: {error}")))?;
            written += 1;
        }
    }
    Ok(written)
}

/// Read the pair table as S0-91's normalized snapshot, deriving the marginals.
///
/// `n11` and `distinct_group_count` come from `m6_pair_stats`, which is where
/// they belong: they are properties of the pair. `n10`, `n01` and `n00` are
/// computed here from the pinned marginals, because they are not — they range
/// over the space's whole eligible group universe (contract line 236). Storing
/// them per pair is what made one root activation a full-table rewrite.
///
/// This is the reader the digest must go through. A caller that read the
/// stored cells directly would be reading a cache that a bounded mutation
/// deliberately does not refresh, and would see a difference the estimator
/// does not have.
pub async fn pair_stats_snapshot(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<Vec<(String, String, PairStatsValues)>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT s.page_a, s.page_b, s.n11, s.distinct_group_count,
                    COALESCE(ma.mass, 0.0), COALESCE(mb.mass, 0.0),
                    COALESCE((SELECT total FROM m6_space_mass WHERE space = ?1), 0.0)
               FROM m6_pair_stats s
               LEFT JOIN m6_page_mass ma ON ma.space = ?1 AND ma.page_id = s.page_a
               LEFT JOIN m6_page_mass mb ON mb.space = ?1 AND mb.page_id = s.page_b
              WHERE s.space = ?1
              ORDER BY s.page_a ASC, s.page_b ASC",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pair snapshot: {error}")))?;

    let mut snapshot = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pair snapshot row: {error}")))?
    {
        let decode = |index: i32, what: &str| -> Result<f64, WenlanError> {
            row.get::<f64>(index)
                .map_err(|error| WenlanError::VectorDb(format!("m6 pair snapshot {what}: {error}")))
        };
        let page_a = row
            .get::<String>(0)
            .map_err(|error| WenlanError::VectorDb(format!("m6 pair snapshot a: {error}")))?;
        let page_b = row
            .get::<String>(1)
            .map_err(|error| WenlanError::VectorDb(format!("m6 pair snapshot b: {error}")))?;
        let n11 = decode(2, "n11")?;
        let distinct_group_count = row
            .get::<i64>(3)
            .map_err(|error| WenlanError::VectorDb(format!("m6 pair snapshot count: {error}")))?;
        let mass_a = decode(4, "mass a")?;
        let mass_b = decode(5, "mass b")?;
        let total = decode(6, "total")?;
        snapshot.push((
            page_a,
            page_b,
            PairStatsValues {
                n11,
                n10: (mass_a - n11).max(0.0),
                n01: (mass_b - n11).max(0.0),
                n00: (total - mass_a - mass_b + n11).max(0.0),
                distinct_group_count,
            },
        ));
    }
    Ok(snapshot)
}

/// What one group-eligibility mutation touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutationReceipt {
    pub pages_touched: usize,
    pub pairs_touched: usize,
}

/// Apply one group's eligibility transition incrementally.
///
/// The whole point of deriving the marginals: this touches `1` space-total
/// row, at most `R-HUB-TOPK` (64) page-mass rows, and at most
/// `R-HUB-MAXPAIRS` (2016) pair rows — never the rest of the table, however
/// many pairs the space has. That is the bound S0-91's negative control
/// asserts, and it is only reachable because `n00` is not stored per pair.
pub async fn apply_group_mutation(
    tx: &libsql::Transaction,
    space: &str,
    group: &GroupSupport,
    reference: i64,
    change: super::relevance::EligibilityChange,
) -> Result<MutationReceipt, WenlanError> {
    use super::relevance::{apply_group_eligibility_change, EligibilityChange};

    let contribution = decayed_contribution(
        hub_weight(group.degree),
        group.newest_root_created_at,
        reference,
    );

    let mut ordered: Vec<&str> = group.pages.iter().map(String::as_str).collect();
    ordered.sort_unstable();
    ordered.dedup();
    ordered.truncate(MAX_NEIGHBORS_PER_ENDPOINT);

    let signed = match change {
        EligibilityChange::Retracted => -contribution,
        EligibilityChange::Reactivated => contribution,
    };

    // The space total moves once, not once per pair.
    tx.execute(
        "UPDATE m6_space_mass SET total = MAX(0.0, total + ?2) WHERE space = ?1",
        libsql::params![space, signed],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 mutation total: {error}")))?;

    for page in &ordered {
        tx.execute(
            "INSERT INTO m6_page_mass (space, page_id, mass) VALUES (?1, ?2, MAX(0.0, ?3))
             ON CONFLICT(space, page_id) DO UPDATE SET mass = MAX(0.0, mass + ?3)",
            libsql::params![space, *page, signed],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 mutation page mass: {error}")))?;
    }

    let mut pairs_touched = 0usize;
    for (index, page_a) in ordered.iter().enumerate() {
        for page_b in &ordered[index + 1..] {
            let mut values = read_pair_cells(tx, space, page_a, page_b).await?;
            apply_group_eligibility_change(&mut values, contribution, change).map_err(
                |reason| {
                    WenlanError::VectorDb(format!("m6 mutation pair {page_a}/{page_b}: {reason}"))
                },
            )?;
            if values.distinct_group_count == 0 {
                // The pair's last supporting group just left. A full
                // recomputation would not emit this pair at all, so leaving a
                // zeroed row behind would make the incremental digest differ
                // by exactly the rows nobody supports any more.
                tx.execute(
                    "DELETE FROM m6_pair_stats
                      WHERE space = ?1 AND page_a = ?2 AND page_b = ?3",
                    libsql::params![space, *page_a, *page_b],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("m6 mutation pair delete: {error}"))
                })?;
            } else {
                tx.execute(
                    "INSERT INTO m6_pair_stats
                         (space, page_a, page_b, n11, n10, n01, n00,
                          distinct_group_count, updated_generation)
                     VALUES (?1, ?2, ?3, ?4, 0, 0, 0, ?5, 0)
                     ON CONFLICT(space, page_a, page_b) DO UPDATE
                         SET n11 = excluded.n11,
                             distinct_group_count = excluded.distinct_group_count",
                    libsql::params![
                        space,
                        *page_a,
                        *page_b,
                        values.n11,
                        values.distinct_group_count
                    ],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("m6 mutation pair write: {error}"))
                })?;
            }
            pairs_touched += 1;
        }
    }

    Ok(MutationReceipt {
        pages_touched: ordered.len(),
        pairs_touched,
    })
}

/// The two authoritative cells of one pair, defaulting to an absent pair.
async fn read_pair_cells(
    tx: &libsql::Transaction,
    space: &str,
    page_a: &str,
    page_b: &str,
) -> Result<PairStatsValues, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT n11, distinct_group_count FROM m6_pair_stats
              WHERE space = ?1 AND page_a = ?2 AND page_b = ?3",
            libsql::params![space, page_a, page_b],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pair cells: {error}")))?;
    let row = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pair cells row: {error}")))?;
    match row {
        Some(row) => Ok(PairStatsValues {
            n11: row
                .get::<f64>(0)
                .map_err(|error| WenlanError::VectorDb(format!("m6 pair cells n11: {error}")))?,
            n10: 0.0,
            n01: 0.0,
            n00: 0.0,
            distinct_group_count: row
                .get::<i64>(1)
                .map_err(|error| WenlanError::VectorDb(format!("m6 pair cells count: {error}")))?,
        }),
        None => Ok(PairStatsValues {
            n11: 0.0,
            n10: 0.0,
            n01: 0.0,
            n00: 0.0,
            distinct_group_count: 0,
        }),
    }
}
