// SPDX-License-Identifier: Apache-2.0
//! D7 — the durable frontier (artifact 5, machine F; spec §3.5, §4.3).
//!
//! The frontier is **derived, not maintained**. [`reason_counts`] recomputes
//! every eligible group's accounting from primary evidence on every call, and
//! [`reconcile_space`] repairs what it finds. That one choice is what makes
//! most of artifact 5 §8's sixteen parking paths impossible rather than
//! guarded: a lost cursor, a dropped event, a crashed worker, and a
//! never-delivered notification all cost a rescan, and none can lose a group,
//! because nothing about a group's presence in the eligible set depends on
//! anything M6 wrote (§1.4).
//!
//! Two failure shapes, two meanings (S0-43):
//!
//! * `reason_count = 0` — eligible and unaccounted for. This is silent parking,
//!   and repairing it by inserting a frontier row (F1) is the frontier's whole
//!   job.
//! * `reason_count > 1` — double-booked. **No automatic repair.** Two live
//!   reasons means two transitions committed that were meant to be mutually
//!   exclusive; auto-picking a winner would destroy the only trace of which one
//!   was wrong. [`reconcile_space`] refuses before writing anything.
//!
//! PR-B2 adds no driver: nothing here is called from `scheduler.rs`,
//! `refinery/`, or any HTTP route.

use super::constants::{
    BELOW_FLOOR_SURFACING_SECONDS, CARD_ID_PREFIX, DOMAIN_CARD_SPACE, FRONTIER_SCAN_SLICE_ROWS,
    NEXT_SCAN_MAX_AHEAD_SECONDS,
};
use super::digest::m6_digest_owned;
use super::frontier_policy::{bind_frontier_groups_to_card, CardGroup};
use super::identity;
use crate::WenlanError;

/// The six durable homes of machine F's six states (S0-42). Named here because
/// [`reason_counts`] must join all six and a missing join reads as a zero —
/// which is indistinguishable from real parking.
pub const REASON_TABLES: [&str; 6] = [
    "genesis_group_coverage",
    "genesis_candidate_roots",
    "genesis_frontier",
    "genesis_card_binding",
    "genesis_suppression",
    "genesis_quarantine",
];

/// One eligible group and how many live durable reasons account for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupAccounting {
    pub independence_group_id: String,
    /// Exactly `1` is the invariant (I-1). See the module doc for what `0` and
    /// `> 1` each mean.
    pub reason_count: i64,
}

/// Artifact 5 §1.2 + §1.3 verbatim in structure: the eligible set, left-joined
/// against all six durable homes, one row per eligible group.
///
/// `live_reservations` uses the **stored** `released_at IS NULL` bit rather
/// than the candidate's state (Q8 adjudication): that is the definition the two
/// partial unique indexes enforce, and the window where the two definitions
/// disagree — a candidate gone terminal before the recovery scan released its
/// claims — is left observable rather than papered over by a second definition
/// that no index backs.
///
/// Only `claim_role = 'concept'` counts. A witness row is deliberately not
/// coverage (I-3), and counting one would let an overview candidate hide a
/// group from the frontier forever.
pub async fn reason_counts(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
) -> Result<Vec<GroupAccounting>, WenlanError> {
    let mut rows = tx
        .query(
            "WITH eligible AS (
                 SELECT DISTINCT r.independence_group_id AS gid
                   FROM edges e
                   JOIN provenance_roots r ON r.root_id = e.root_id
                  WHERE e.space = ?1
                    AND e.grounded = 1
                    AND e.valid_until IS NULL
                    AND r.status = 'active'
                    AND r.root_kind <> 'generated'
             ),
             live_reservations AS (
                 SELECT DISTINCT gr.independence_group_id AS gid
                   FROM genesis_candidate_roots gr
                   JOIN genesis_candidates c ON c.candidate_id = gr.candidate_id
                  WHERE c.space = ?1
                    AND gr.coverage_epoch = ?2
                    AND gr.claim_role = 'concept'
                    AND gr.released_at IS NULL
             )
             SELECT e.gid,
                    (cov.independence_group_id IS NOT NULL)
                  + (clm.gid IS NOT NULL)
                  + (fro.independence_group_id IS NOT NULL)
                  + (crd.independence_group_id IS NOT NULL)
                  + (sup.independence_group_id IS NOT NULL)
                  + (qua.independence_group_id IS NOT NULL) AS reason_count
               FROM eligible e
               LEFT JOIN genesis_group_coverage cov
                      ON cov.independence_group_id = e.gid
                     AND cov.space = ?1 AND cov.coverage_epoch = ?2
               LEFT JOIN live_reservations clm
                      ON clm.gid = e.gid
               LEFT JOIN genesis_frontier fro
                      ON fro.independence_group_id = e.gid
                     AND fro.space = ?1 AND fro.coverage_epoch = ?2
               LEFT JOIN genesis_card_binding crd
                      ON crd.independence_group_id = e.gid
                     AND crd.space = ?1 AND crd.coverage_epoch = ?2
                     AND crd.closed_at IS NULL
               LEFT JOIN genesis_suppression sup
                      ON sup.independence_group_id = e.gid
                     AND sup.space = ?1 AND sup.coverage_epoch = ?2
                     AND sup.lapsed_at IS NULL
               LEFT JOIN genesis_quarantine qua
                      ON qua.independence_group_id = e.gid
                     AND qua.space = ?1 AND qua.coverage_epoch = ?2
                     AND qua.lifted_at IS NULL
              ORDER BY e.gid",
            libsql::params![space, coverage_epoch],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 frontier differential: {error}")))?;

    let mut accounting = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 frontier differential row: {error}")))?
    {
        let decode = |error: libsql::Error| {
            WenlanError::VectorDb(format!("m6 frontier differential decode: {error}"))
        };
        accounting.push(GroupAccounting {
            independence_group_id: row.get(0).map_err(decode)?,
            reason_count: row.get(1).map_err(decode)?,
        });
    }
    Ok(accounting)
}

/// What one reconciliation pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReconcileOutcome {
    /// Groups that read `0` and got an F1 frontier row. The **count of
    /// repairs is the signal**: a healthy space repairs nothing, and a
    /// repair rate that does not fall is a leak upstream of the frontier.
    pub repaired: Vec<String>,
    /// Suppressions that reached `expires_at` and were lapsed back to the
    /// frontier in this pass (F11).
    pub lapsed_suppressions: u64,
    /// Every eligible group examined, repaired or not.
    pub examined: usize,
}

/// The differential reconciliation for one space and epoch (F1 + F11).
///
/// Order matters. Expired suppressions lapse **first**, because lapse and
/// frontier restoration are one transaction inside
/// `frontier_policy::reconcile_expired_suppressions`; counting first would read
/// those groups as suppressed and then immediately change their reason, so the
/// returned accounting would describe a state that no longer exists.
///
/// Double-booked groups are collected and refused **before any repair is
/// written**, so a refusal leaves the caller's transaction with nothing of this
/// function's in it to roll back.
pub async fn reconcile_space(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
) -> Result<ReconcileOutcome, WenlanError> {
    let lapsed_suppressions =
        super::frontier_policy::reconcile_expired_suppressions(tx, space, coverage_epoch)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m6 suppression lapse: {error}")))?;

    let accounting = reason_counts(tx, space, coverage_epoch).await?;
    let double_booked: Vec<&GroupAccounting> = accounting
        .iter()
        .filter(|group| group.reason_count > 1)
        .collect();
    if !double_booked.is_empty() {
        let detail = double_booked
            .iter()
            .map(|group| format!("{}={}", group.independence_group_id, group.reason_count))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(WenlanError::Conflict(format!(
            "m6 frontier: {} group(s) in {space:?} epoch {coverage_epoch} have more than one live durable reason ({detail})",
            double_booked.len()
        )));
    }

    let mut repaired = Vec::new();
    for group in accounting.iter().filter(|group| group.reason_count == 0) {
        insert_frontier_row(tx, space, &group.independence_group_id, coverage_epoch).await?;
        repaired.push(group.independence_group_id.clone());
    }
    Ok(ReconcileOutcome {
        repaired,
        lapsed_suppressions,
        examined: accounting.len(),
    })
}

/// F1 — the genuinely-first insertion for a group in this epoch.
///
/// This is the only writer that sets `first_seen_at = unixepoch()`; every
/// re-entry path goes through [`restore_frontier_row`] and preserves the
/// earlier value (S0-50). `ON CONFLICT DO NOTHING` rather than `OR REPLACE`
/// (S0-48): an insert racing an existing row must leave the older
/// `first_seen_at` standing, since replacing it is exactly the clock reset that
/// makes a repeatedly-claimed group never reach seven days.
pub async fn insert_frontier_row(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
) -> Result<u64, WenlanError> {
    tx.execute(
        "INSERT INTO genesis_frontier
             (space, independence_group_id, coverage_epoch, first_seen_at, next_scan_at)
         VALUES (?1, ?2, ?3, unixepoch(), unixepoch())
         ON CONFLICT(space, independence_group_id, coverage_epoch) DO NOTHING",
        libsql::params![space, independence_group_id, coverage_epoch],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 frontier insert: {error}")))
}

/// F2 — a concept claim became the group's durable reason, so its frontier row
/// retires.
///
/// Returns the number of rows removed, and **does not require exactly one**.
/// A claim can legally precede the group's first reconciliation, in which case
/// there is no frontier row to retire and nothing is wrong: D4's exclusion is
/// the partial unique index on the claim, never the absence of a frontier row.
pub async fn retire_frontier_row(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
) -> Result<u64, WenlanError> {
    tx.execute(
        "DELETE FROM genesis_frontier
          WHERE space = ?1 AND independence_group_id = ?2 AND coverage_epoch = ?3",
        libsql::params![space, independence_group_id, coverage_epoch],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 frontier retire: {error}")))
}

/// F5 — a reservation was released; the group returns to `waiting_frontier`.
///
/// **Disclosed deviation from S0-50, and the one place PR-B2 could not satisfy
/// a Stage-0 decision with the shipped schema.** S0-50 requires `first_seen_at`
/// to survive every re-entry within an epoch, and names F5 as one of the four
/// re-entry paths. But F2 *deleted* the frontier row when the claim was made,
/// and `genesis_candidate_roots` has no `first_seen_at` column to have carried
/// the value across — unlike `genesis_card_binding`, `genesis_suppression`, and
/// `genesis_quarantine`, each of which does carry one, which is why F8/F11/F12
/// preserve it exactly.
///
/// The closure used here needs no migration: restore from
/// `MIN(genesis_candidate_roots.created_at)` over **every** claim this group
/// ever held in this epoch, released or not, falling back to `unixepoch()` for
/// a group with no claim history. That defeats the failure S0-50 actually
/// targets — P13, claim/release churn resetting the clock — because the
/// minimum over all claims does not move when a group is claimed and released
/// repeatedly. What it loses is the interval between a group first becoming
/// eligible and its first claim, so the seven-day timer starts at first claim
/// rather than at first eligibility. Closing that gap needs a `first_seen_at`
/// column on `genesis_candidate_roots`, i.e. a migration, which this slice is
/// forbidden from adding.
pub async fn restore_frontier_row(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
) -> Result<u64, WenlanError> {
    tx.execute(
        "INSERT INTO genesis_frontier
             (space, independence_group_id, coverage_epoch, first_seen_at, next_scan_at)
         SELECT ?1, ?2, ?3,
                COALESCE((
                    SELECT MIN(gr.created_at)
                      FROM genesis_candidate_roots gr
                      JOIN genesis_candidates c ON c.candidate_id = gr.candidate_id
                     WHERE c.space = ?1
                       AND gr.independence_group_id = ?2
                       AND gr.coverage_epoch = ?3
                ), unixepoch()),
                unixepoch()
          WHERE TRUE
         ON CONFLICT(space, independence_group_id, coverage_epoch) DO NOTHING",
        libsql::params![space, independence_group_id, coverage_epoch],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 frontier restore: {error}")))
}

/// F8 — a surfaced card expired; close its live binding and return the group to
/// the frontier, in the caller's transaction.
///
/// The binding carries `first_seen_at`, so this path preserves the clock
/// exactly as S0-50 requires — no approximation, unlike [`restore_frontier_row`].
/// Returns `false` when the group had no open binding, which is the normal case
/// for a candidate that never reached `review_required`.
pub async fn expire_card_binding_to_frontier(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT card_id, first_seen_at
               FROM genesis_card_binding
              WHERE space = ?1 AND independence_group_id = ?2
                AND coverage_epoch = ?3 AND closed_at IS NULL",
            libsql::params![space, independence_group_id, coverage_epoch],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 card binding read: {error}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 card binding row: {error}")))?
    else {
        return Ok(false);
    };
    let decode =
        |error: libsql::Error| WenlanError::VectorDb(format!("m6 card binding decode: {error}"));
    let card_id: String = row.get(0).map_err(decode)?;
    let first_seen_at: i64 = row.get(1).map_err(decode)?;
    drop(rows);

    let closed = tx
        .execute(
            "UPDATE genesis_card_binding
                SET closed_at = unixepoch()
              WHERE space = ?1 AND independence_group_id = ?2
                AND coverage_epoch = ?3 AND card_id = ?4 AND closed_at IS NULL",
            libsql::params![space, independence_group_id, coverage_epoch, card_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 card binding close: {error}")))?;
    if closed != 1 {
        return Err(WenlanError::Conflict(format!(
            "m6 card expiry closed {closed} bindings for {independence_group_id:?}, expected 1"
        )));
    }
    tx.execute(
        "INSERT INTO genesis_frontier
             (space, independence_group_id, coverage_epoch, first_seen_at, next_scan_at)
         VALUES (?1, ?2, ?3, ?4, unixepoch())
         ON CONFLICT(space, independence_group_id, coverage_epoch) DO NOTHING",
        libsql::params![space, independence_group_id, coverage_epoch, first_seen_at],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 card expiry restore: {error}")))?;
    Ok(true)
}

/// Defer a frontier row's next examination, clamped to 24h ahead (S0-46).
///
/// The clamp is in the statement, not in Rust, so there is no path — including a
/// caller that computes its own delay — that can write a `next_scan_at` past
/// the ceiling. Unbounded deferral is a parking mechanism: push the value
/// forward every pass and the seven-day timer is never evaluated (P3).
pub async fn defer_scan(
    tx: &libsql::Transaction,
    space: &str,
    independence_group_id: &str,
    coverage_epoch: i64,
    delay_seconds: i64,
) -> Result<u64, WenlanError> {
    tx.execute(
        "UPDATE genesis_frontier
            SET next_scan_at = unixepoch() + MIN(MAX(?4, 0), ?5)
          WHERE space = ?1 AND independence_group_id = ?2 AND coverage_epoch = ?3",
        libsql::params![
            space,
            independence_group_id,
            coverage_epoch,
            delay_seconds,
            NEXT_SCAN_MAX_AHEAD_SECONDS
        ],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 frontier defer: {error}")))
}

/// The largest `next_scan_at - unixepoch()` currently stored for a space; `0`
/// when the space has no frontier rows. G5's S0-46 assertion reads this.
pub async fn max_next_scan_lead_seconds(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<i64, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT COALESCE(MAX(next_scan_at - unixepoch()), 0)
               FROM genesis_frontier WHERE space = ?1",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 next_scan lead: {error}")))?;
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 next_scan lead row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 next_scan lead decode: {error}")))?
        .unwrap_or(0))
}

// ---------------------------------------------------------------------------
// The scan cursor (S0-44, S0-45 as amended by Q7)
// ---------------------------------------------------------------------------

/// One frontier row, in D7's ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontierRow {
    pub independence_group_id: String,
    pub first_seen_at: i64,
    pub next_scan_at: i64,
}

/// A resume position within one pass over one space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanCursor {
    pub next_scan_at: i64,
    pub first_seen_at: i64,
    pub independence_group_id: String,
}

/// One slice of a pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScanSlice {
    pub rows: Vec<FrontierRow>,
    /// `None` means the pass wrapped — the cursor is cleared and the next pass
    /// starts over. A wrap is **not** a checkpoint and asserts nothing (§2.2):
    /// a group inserted at an already-passed position is simply seen next pass,
    /// and it was accounted for the whole time by its frontier row. Unscanned
    /// and unaccounted-for are different conditions; only the second is parking.
    pub next_cursor: Option<ScanCursor>,
}

/// The `app_metadata` key prefix for the per-space cursor.
///
/// **Amends S0-45** per the Q7 adjudication. S0-45 specified one global key
/// ordered `next_scan_at, first_seen_at, space, gid`, but the shipped
/// `idx_genesis_frontier_scan` is `(space, next_scan_at, first_seen_at,
/// independence_group_id)` — space **first** — so a global cursor would either
/// sort or scan every space. The index is shipped; the cursor is a pure
/// optimization (S0-44) whose loss costs one extra pass. Amending the cursor is
/// the cheap side of that trade, and round-robin across spaces gives the same
/// starvation-freedom the global ordering was reaching for.
pub const FRONTIER_CURSOR_KEY_PREFIX: &str = "m6_frontier_cursor_v1:";

fn cursor_key(space: &str) -> String {
    format!("{FRONTIER_CURSOR_KEY_PREFIX}{space}")
}

/// Read a space's resume position. An absent key and the empty string both mean
/// "start from the beginning" — the tree's cursor idiom, and the reason a
/// cleared cursor needs no separate "wrapped" flag.
pub async fn read_cursor(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<Option<ScanCursor>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT value FROM app_metadata WHERE key = ?1",
            libsql::params![cursor_key(space)],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 frontier cursor read: {error}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 frontier cursor row: {error}")))?
    else {
        return Ok(None);
    };
    let value: String = row
        .get(0)
        .map_err(|error| WenlanError::VectorDb(format!("m6 frontier cursor decode: {error}")))?;
    Ok(parse_cursor(&value))
}

/// `"<next_scan_at>|<first_seen_at>|<gid>"`, or `None` for the cleared cursor.
///
/// A malformed value returns `None` rather than erroring, and that is
/// deliberate: S0-44 makes the cursor a pure optimization, so a corrupted one
/// must cost a rescan, never a failed scan.
fn parse_cursor(value: &str) -> Option<ScanCursor> {
    let mut parts = value.splitn(3, '|');
    let next_scan_at = parts.next()?.parse().ok()?;
    let first_seen_at = parts.next()?.parse().ok()?;
    let independence_group_id = parts.next()?;
    if independence_group_id.is_empty() {
        return None;
    }
    Some(ScanCursor {
        next_scan_at,
        first_seen_at,
        independence_group_id: independence_group_id.to_string(),
    })
}

/// Store a space's resume position, or clear it on wrap.
pub async fn write_cursor(
    tx: &libsql::Transaction,
    space: &str,
    cursor: Option<&ScanCursor>,
) -> Result<(), WenlanError> {
    let value = match cursor {
        Some(cursor) => format!(
            "{}|{}|{}",
            cursor.next_scan_at, cursor.first_seen_at, cursor.independence_group_id
        ),
        None => String::new(),
    };
    tx.execute(
        "INSERT INTO app_metadata (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        libsql::params![cursor_key(space), value],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 frontier cursor write: {error}")))?;
    Ok(())
}

/// Walk one slice of a space's frontier in D7's ordering, resuming from
/// `cursor`. Read-only; the caller decides whether to persist the returned
/// position.
///
/// The row cap bounds work per turn, never what stays accounted for (S0-54):
/// rows past the cap keep their frontier rows and are seen on the next slice.
pub async fn scan_slice(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
    cursor: Option<&ScanCursor>,
) -> Result<ScanSlice, WenlanError> {
    let limit = FRONTIER_SCAN_SLICE_ROWS as i64;
    // One statement for both arms, with the cursor bound as a sentinel rather
    // than branching the SQL: the row-value comparison is the index walk, and
    // two spellings of it could drift apart in exactly the ordering the whole
    // cursor depends on.
    let mut rows = tx
        .query(
            "SELECT independence_group_id, first_seen_at, next_scan_at
               FROM genesis_frontier
              WHERE space = ?1 AND coverage_epoch = ?2
                AND (?3 = 0
                     OR (next_scan_at, first_seen_at, independence_group_id) > (?4, ?5, ?6))
              ORDER BY next_scan_at, first_seen_at, independence_group_id
              LIMIT ?7",
            libsql::params![
                space,
                coverage_epoch,
                i64::from(cursor.is_some()),
                cursor.map(|cursor| cursor.next_scan_at).unwrap_or_default(),
                cursor
                    .map(|cursor| cursor.first_seen_at)
                    .unwrap_or_default(),
                cursor
                    .map(|cursor| cursor.independence_group_id.as_str())
                    .unwrap_or_default(),
                limit
            ],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 frontier scan: {error}")))?;
    let mut slice = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 frontier scan row: {error}")))?
    {
        let decode = |error: libsql::Error| {
            WenlanError::VectorDb(format!("m6 frontier scan decode: {error}"))
        };
        slice.push(FrontierRow {
            independence_group_id: row.get(0).map_err(decode)?,
            first_seen_at: row.get(1).map_err(decode)?,
            next_scan_at: row.get(2).map_err(decode)?,
        });
    }
    let next_cursor = (slice.len() as i64 == limit).then(|| {
        let last = slice.last().expect("a full slice has a last row");
        ScanCursor {
            next_scan_at: last.next_scan_at,
            first_seen_at: last.first_seen_at,
            independence_group_id: last.independence_group_id.clone(),
        }
    });
    Ok(ScanSlice {
        rows: slice,
        next_cursor,
    })
}

// ---------------------------------------------------------------------------
// F3 — the coalesced unformed-topic card
// ---------------------------------------------------------------------------

/// `m6_unformed_topic_<space-digest>_<epoch>` (artifact 5 §3.2). One card per
/// `(space, coverage_epoch)` (S0-47), and the space appears only as a digest so
/// a space name never enters a durable ID.
pub fn card_id(space_id: &str, coverage_epoch: i64) -> String {
    let digest = m6_digest_owned(DOMAIN_CARD_SPACE, &[space_id.as_bytes().to_vec()]);
    format!("{CARD_ID_PREFIX}{digest}_{coverage_epoch}")
}

/// Frontier groups whose seven-day below-floor timer has fired.
///
/// The observable is the frontier row's own age: a group that has held a
/// frontier row for seven days is a group nothing claimed, which is exactly
/// what "below the independence floor for seven days" looks like from here.
/// The card fires regardless of how far below the floor the group is — that is
/// P10, the positive control for a permanently small space.
pub async fn below_floor_groups(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
) -> Result<Vec<String>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT independence_group_id
               FROM genesis_frontier
              WHERE space = ?1 AND coverage_epoch = ?2
                AND first_seen_at <= unixepoch() - ?3
              ORDER BY first_seen_at, independence_group_id",
            libsql::params![space, coverage_epoch, BELOW_FLOOR_SURFACING_SECONDS],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 below-floor scan: {error}")))?;
    let mut groups = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 below-floor row: {error}")))?
    {
        groups.push(
            row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("m6 below-floor decode: {error}"))
            })?,
        );
    }
    Ok(groups)
}

/// What F3 bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Surfacing {
    pub card_id: String,
    pub groups: Vec<String>,
}

/// F3 — bind every timed-out below-floor group in a space to the one coalesced
/// card, in the caller's transaction.
///
/// **This writes the binding, not the card.** The `refinement_queue` row that a
/// human would see is deliberately not written here: the Q11 adjudication makes
/// "epochs opened for every space with `genesis_enabled = 0` and no page,
/// support, or **card** row written" a positive control of the shadow, so
/// projecting a user-visible card is PR-B3's, under the flag. A driver that
/// calls this without also projecting the card would be parking every group it
/// binds — the binding must be written in the same transaction as the
/// projection, and S0-48 fixes that projection as `INSERT OR IGNORE` with an
/// explicit status so a dismissed card is never resurrected.
///
/// The binding's `page_id` is the page this group *would* form: the
/// evidence-cluster slot over the single group, run through the same derivation
/// a real candidate uses. It is an identity, not a row — nothing is created in
/// `pages` — and it is what makes a later suppression's identity (§4, durable
/// forever) refer to something stable rather than to a group ID that a new
/// epoch re-keys.
pub async fn surface_below_floor_groups(
    tx: &libsql::Transaction,
    space: &str,
    space_id: &str,
    coverage_epoch: i64,
) -> Result<Option<Surfacing>, WenlanError> {
    let groups = below_floor_groups(tx, space, coverage_epoch).await?;
    if groups.is_empty() {
        return Ok(None);
    }
    let card = card_id(space_id, coverage_epoch);
    let page_ids: Vec<String> = groups
        .iter()
        .map(|group| {
            identity::page_id(&identity::slot_id_evidence_cluster(
                space_id,
                std::slice::from_ref(group),
            ))
        })
        .collect();
    let card_groups: Vec<CardGroup<'_>> = groups
        .iter()
        .zip(page_ids.iter())
        .map(|(group, page_id)| CardGroup::new(group.as_str(), page_id.as_str()))
        .collect();
    bind_frontier_groups_to_card(tx, space, coverage_epoch, &card, &card_groups)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 below-floor binding: {error}")))?;
    Ok(Some(Surfacing {
        card_id: card,
        groups,
    }))
}
