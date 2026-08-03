// SPDX-License-Identifier: Apache-2.0
//! M6 relevance substrate (D1 and D6).
//!
//! The tables are additive and inert until a later rung wires producers and
//! readers. This module nevertheless owns the invariants those later writers
//! must not be able to weaken: the raw currently-eligible co-supporting group
//! count, deterministic pair-stat normalization, bounded adjacency slots, and
//! neighbor uniqueness within an endpoint kind.

use crate::m6::digest::m6_digest_owned;
use crate::WenlanError;

/// D1's undecayed, unsmoothed distinct-group floor (S0-88).
///
/// Its reader is the §2.2 estimator, which C1 does not build — C1 produces the
/// table the estimator will read. The allow is on this item rather than on the
/// module (§12) so that the next unused item in `relevance` fails the build
/// instead of being absorbed by a blanket.
#[allow(dead_code)]
pub const QUALIFIED_CO_CITATION_GROUP_FLOOR: i64 = 3;

/// S0-91's frozen normalized-snapshot domain.
///
/// Transitively test-only: `normalized_pair_stats_digest` is the oracle the
/// mutation tests compare through, and nothing in production digests the pair
/// table. Same targeted-allow reasoning as the floor above.
#[allow(dead_code)]
pub const PAIR_STATS_DIGEST_DOMAIN: &str = "m6-pairstats-v1";

/// D9's hard caps for query (c).
pub const MAX_CANDIDATE_ENDPOINTS: usize = 32;
pub const MAX_NEIGHBORS_PER_ENDPOINT: usize = 64;

/// Install the D1/D6 relevance tables and their fixed indexes.
///
/// This is deliberately a transaction-scoped entrypoint: migration 109's
/// root-owned orchestrator composes it with the other approved D1-D8 homes and
/// advances `user_version` only after the whole additive schema commits.
pub async fn ensure_relevance_tables(tx: &libsql::Transaction) -> Result<(), WenlanError> {
    tx.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS m6_pair_stats (
            space                 TEXT    NOT NULL,
            page_a                TEXT    NOT NULL,
            page_b                TEXT    NOT NULL,
            n11                   REAL    NOT NULL CHECK (n11 >= 0),
            n10                   REAL    NOT NULL CHECK (n10 >= 0),
            n01                   REAL    NOT NULL CHECK (n01 >= 0),
            n00                   REAL    NOT NULL CHECK (n00 >= 0),
            distinct_group_count  INTEGER NOT NULL CHECK (distinct_group_count >= 0),
            updated_generation    INTEGER NOT NULL CHECK (updated_generation >= 0),
            PRIMARY KEY(space, page_a, page_b),
            CHECK (page_a < page_b)
        );
        CREATE INDEX IF NOT EXISTS idx_m6_pair_stats_space_page_a_generation
            ON m6_pair_stats(space, page_a, updated_generation);

        CREATE TABLE IF NOT EXISTS m6_adjacency (
            space          TEXT    NOT NULL,
            endpoint_kind  TEXT    NOT NULL,
            endpoint_id    TEXT    NOT NULL,
            neighbor_id    TEXT    NOT NULL,
            rank            INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 64),
            PRIMARY KEY(space, endpoint_kind, endpoint_id, rank),
            UNIQUE(space, endpoint_kind, endpoint_id, neighbor_id)
        );
        CREATE INDEX IF NOT EXISTS idx_m6_adjacency_space_neighbor
            ON m6_adjacency(space, neighbor_id);

        -- The marginals `n10`/`n01`/`n00` are derived from, not stored in,
        -- `m6_pair_stats`. They range over every eligible group in the space,
        -- so maintaining them per pair would make one root activation rewrite
        -- the whole table -- which is what `R-HUB-MAXPAIRS` (2016 pairs per
        -- group) exists to forbid. Only `n11` and `distinct_group_count` are
        -- properties of a pair; the rest of the estimator lives here.
        CREATE TABLE IF NOT EXISTS m6_page_mass (
            space   TEXT NOT NULL,
            page_id TEXT NOT NULL,
            mass    REAL NOT NULL CHECK (mass >= 0),
            PRIMARY KEY(space, page_id)
        );
        CREATE TABLE IF NOT EXISTS m6_space_mass (
            space TEXT PRIMARY KEY,
            total REAL NOT NULL CHECK (total >= 0)
        );
        ",
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m109 relevance tables: {error}")))?;
    Ok(())
}

/// The four decayed estimator cells plus D1's raw co-support count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairStatsValues {
    pub n11: f64,
    pub n10: f64,
    pub n01: f64,
    pub n00: f64,
    /// Currently eligible distinct groups in `n11`; never a historical count.
    pub distinct_group_count: i64,
}

/// One normalized pair-stat row at a fixed relevance generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairStatsSnapshotRow<'a> {
    pub page_a: &'a str,
    pub page_b: &'a str,
    pub values: PairStatsValues,
}

/// A group-level eligibility transition after all roots in the group collapse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EligibilityChange {
    Retracted,
    Reactivated,
}

/// Derive D1's floor at read time from the raw count.
pub fn qualified_co_citation(values: &PairStatsValues) -> bool {
    values.distinct_group_count >= QUALIFIED_CO_CITATION_GROUP_FLOOR
}

/// Apply one collapsed group's eligibility transition to an in-memory row.
///
/// Callers must invoke this only when a group as a whole becomes ineligible or
/// eligible, not for each root: several roots in one independence group still
/// count once.
///
/// Only `n11` and the raw count move. This function once took a `PairCell`
/// naming which of the four estimator cells to charge, and that parameter is
/// gone rather than unused: `n10`/`n01`/`n00` are *derived* from
/// `m6_page_mass`/`m6_space_mass` at read time (see
/// [`super::relevance_sweep::pair_stats_snapshot`]), so charging a marginal
/// here would count the same transition twice — once in the stored cell and
/// again in the mass the reader subtracts. The mutation applier updates mass
/// separately; a cell selector would only offer a way to get that wrong.
pub fn apply_group_eligibility_change(
    values: &mut PairStatsValues,
    decayed_contribution: f64,
    change: EligibilityChange,
) -> Result<(), &'static str> {
    if !decayed_contribution.is_finite() || decayed_contribution < 0.0 {
        return Err("decayed contribution must be finite and non-negative");
    }

    match change {
        EligibilityChange::Retracted => {
            if values.n11 < decayed_contribution {
                return Err("retraction would make a decayed cell negative");
            }
            if values.distinct_group_count == 0 {
                return Err("retraction would make the distinct-group count negative");
            }
            values.n11 -= decayed_contribution;
            values.distinct_group_count -= 1;
        }
        EligibilityChange::Reactivated => {
            if values.distinct_group_count == i64::MAX {
                return Err("reactivation would overflow the distinct-group count");
            }
            let reactivated = values.n11 + decayed_contribution;
            if !reactivated.is_finite() {
                return Err("reactivation would make a decayed cell non-finite");
            }
            values.n11 = reactivated;
            values.distinct_group_count += 1;
        }
    }
    Ok(())
}

/// Digest S0-91's normalized pair table, including D1's raw count.
///
/// Rows are encoded as length-framed tuples, byte-sorted, deduplicated as the
/// frozen `sorted_set` construction requires, then hashed under
/// `m6-pairstats-v1`. Decayed cells are rounded to exactly nine decimal places
/// before encoding (S0-92). `updated_generation` is absent because the caller
/// snapshots at one fixed generation.
pub fn normalized_pair_stats_digest(
    rows: &[PairStatsSnapshotRow<'_>],
) -> Result<String, &'static str> {
    let mut encoded_rows = Vec::with_capacity(rows.len());
    for row in rows {
        validate_snapshot_row(row)?;
        let fields = [
            row.page_a.as_bytes().to_vec(),
            row.page_b.as_bytes().to_vec(),
            rounded_cell_part(row.values.n11),
            rounded_cell_part(row.values.n10),
            rounded_cell_part(row.values.n01),
            rounded_cell_part(row.values.n00),
            row.values.distinct_group_count.to_string().into_bytes(),
        ];
        encoded_rows.push(length_frame_tuple(&fields));
    }
    encoded_rows.sort_unstable();
    encoded_rows.dedup();

    let mut parts = Vec::with_capacity(encoded_rows.len() + 1);
    parts.push(encoded_rows.len().to_string().into_bytes());
    parts.extend(encoded_rows);
    Ok(m6_digest_owned(PAIR_STATS_DIGEST_DOMAIN, &parts))
}

fn validate_snapshot_row(row: &PairStatsSnapshotRow<'_>) -> Result<(), &'static str> {
    if row.page_a >= row.page_b {
        return Err("pair-stat rows require page_a < page_b");
    }
    if row.values.distinct_group_count < 0 {
        return Err("distinct-group count must be non-negative");
    }
    for cell in [
        row.values.n11,
        row.values.n10,
        row.values.n01,
        row.values.n00,
    ] {
        if !cell.is_finite() || cell < 0.0 {
            return Err("pair-stat cells must be finite and non-negative");
        }
    }
    Ok(())
}

fn rounded_cell_part(value: f64) -> Vec<u8> {
    // Fixed-precision formatting performs the specified 9-dp rounding without
    // multiplying first (which could overflow for an otherwise finite value).
    let normalized_zero = if value == 0.0 { 0.0 } else { value };
    format!("{normalized_zero:.9}").into_bytes()
}

fn length_frame_tuple(fields: &[Vec<u8>]) -> Vec<u8> {
    let mut encoded = Vec::new();
    for field in fields {
        encoded.extend_from_slice(&(field.len() as u64).to_le_bytes());
        encoded.extend_from_slice(field);
    }
    encoded
}

/// Query (c): aggregate common neighbors for all candidate endpoints in one
/// statement while keeping endpoint kinds isolated.
///
/// The `endpoint_kind` predicate and grouping key are both intentional. IDs
/// can be shared across kinds; omitting either makes a page's count depend on
/// entity adjacency with the same opaque ID.
pub async fn common_neighbor_counts<C, N>(
    connection: &libsql::Connection,
    space: &str,
    endpoint_kind: &str,
    candidate_endpoint_ids: &[C],
    source_neighbor_ids: &[N],
) -> Result<Vec<(String, i64)>, WenlanError>
where
    C: AsRef<str>,
    N: AsRef<str>,
{
    if candidate_endpoint_ids.len() > MAX_CANDIDATE_ENDPOINTS {
        return Err(WenlanError::VectorDb(format!(
            "M6 relevance query has {} candidates; cap is {MAX_CANDIDATE_ENDPOINTS}",
            candidate_endpoint_ids.len()
        )));
    }
    if source_neighbor_ids.len() > MAX_NEIGHBORS_PER_ENDPOINT {
        return Err(WenlanError::VectorDb(format!(
            "M6 relevance query has {} source neighbors; cap is {MAX_NEIGHBORS_PER_ENDPOINT}",
            source_neighbor_ids.len()
        )));
    }
    if candidate_endpoint_ids.is_empty() || source_neighbor_ids.is_empty() {
        return Ok(Vec::new());
    }

    let candidate_placeholders = positional_placeholders(3, candidate_endpoint_ids.len());
    let neighbor_start = 3 + candidate_endpoint_ids.len();
    let neighbor_placeholders = positional_placeholders(neighbor_start, source_neighbor_ids.len());
    let sql = format!(
        "SELECT endpoint_kind, endpoint_id, COUNT(*)
           FROM m6_adjacency
          WHERE space = ?1
            AND endpoint_kind = ?2
            AND endpoint_id IN ({candidate_placeholders})
            AND neighbor_id IN ({neighbor_placeholders})
          GROUP BY endpoint_kind, endpoint_id
          ORDER BY endpoint_kind, endpoint_id"
    );

    let mut params =
        Vec::with_capacity(2 + candidate_endpoint_ids.len() + source_neighbor_ids.len());
    params.push(libsql::Value::Text(space.to_string()));
    params.push(libsql::Value::Text(endpoint_kind.to_string()));
    params.extend(
        candidate_endpoint_ids
            .iter()
            .map(|id| libsql::Value::Text(id.as_ref().to_string())),
    );
    params.extend(
        source_neighbor_ids
            .iter()
            .map(|id| libsql::Value::Text(id.as_ref().to_string())),
    );

    let mut rows = connection
        .query(&sql, libsql::params_from_iter(params))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("M6 query (c): {error}")))?;
    let mut counts = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("M6 query (c) row: {error}")))?
    {
        let returned_kind = row
            .get::<String>(0)
            .map_err(|error| WenlanError::VectorDb(format!("M6 query (c) kind: {error}")))?;
        if returned_kind != endpoint_kind {
            return Err(WenlanError::VectorDb(format!(
                "M6 query (c) returned endpoint kind {returned_kind:?} for {endpoint_kind:?}"
            )));
        }
        let endpoint_id = row
            .get::<String>(1)
            .map_err(|error| WenlanError::VectorDb(format!("M6 query (c) endpoint: {error}")))?;
        let count = row
            .get::<i64>(2)
            .map_err(|error| WenlanError::VectorDb(format!("M6 query (c) count: {error}")))?;
        counts.push((endpoint_id, count));
    }
    Ok(counts)
}

fn positional_placeholders(start: usize, count: usize) -> String {
    (start..start + count)
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}
