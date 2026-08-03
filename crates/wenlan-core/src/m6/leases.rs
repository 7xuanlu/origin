// SPDX-License-Identifier: Apache-2.0
//! D6 — one durable lease registry (spec §3.4, machine C).
//!
//! This module has **no storage of its own**. It is a facade over the M4
//! `grouping_leases` table (`crates/wenlan-core/src/db.rs`, DDL in
//! `ensure community substrate tables`), extended only by new values in the
//! existing `phase` column. Creating a parallel lease system is a contract
//! violation (I-6), and the cheapest way to notice one would be a second table
//! — so there is deliberately nothing here that could become one.
//!
//! Three properties are load-bearing and each is a gate row:
//!
//! * **`phase` is parameterised in the reap as well as the insert.** M4's reap
//!   scopes itself with the literal `'community'` (artifact 2 §2.1 fact 1); a
//!   copy that forgot to change it would reap a live lease belonging to another
//!   phase. Every statement below binds `phase`.
//! * **Acquisition is `INSERT ... ON CONFLICT DO NOTHING`, and zero affected
//!   rows is the "someone else holds it" signal** (C2). That is already the CAS
//!   D6 wants, so a manual call and a scheduler call for the same
//!   `(phase, space, input_generation)` join the same operation structurally —
//!   the primary key *is* the operation identity. No extra coordination exists,
//!   and none may be added.
//! * **There is no renewal edge** (S0-4). A lease either completes inside its
//!   TTL or is reaped; a renewal timer would have to hold lease state across the
//!   model call, which is the one thing D3 forbids spanning.
//!
//! Every clock is SQLite `unixepoch()` evaluated in-statement (S0-11) — no
//! Rust-side `now` is passed as a parameter anywhere in this module.

use super::constants::{
    FRONTIER_LEASE_TTL_SECONDS, GENESIS_LEASE_TTL_SECONDS, RELEVANCE_LEASE_TTL_SECONDS,
};
use crate::WenlanError;

/// The M6 phases PR-B2 and PR-C drive. `refresh` is still reserved by S0-3 and
/// lands with C2; it is absent here rather than declared unused, because a
/// phase value with no acquirer is a value nothing checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeasePhase {
    /// Candidate prepare through finalize (900s).
    Genesis,
    /// Frontier reconciliation scan (120s).
    Frontier,
    /// Bounded relevance sweep (300s).
    Relevance,
}

impl LeasePhase {
    /// Every phase M6 owns, and therefore the only rows M6 may reap.
    /// `grouping_leases` is shared with M4's `community` phase.
    pub const ALL: [LeasePhase; 3] = [
        LeasePhase::Genesis,
        LeasePhase::Frontier,
        LeasePhase::Relevance,
    ];

    /// The stored `grouping_leases.phase` value.
    pub fn as_str(self) -> &'static str {
        match self {
            LeasePhase::Genesis => "genesis",
            LeasePhase::Frontier => "frontier",
            LeasePhase::Relevance => "relevance",
        }
    }

    /// TTL in seconds (S0-3).
    pub fn ttl_seconds(self) -> i64 {
        match self {
            LeasePhase::Genesis => GENESIS_LEASE_TTL_SECONDS,
            LeasePhase::Frontier => FRONTIER_LEASE_TTL_SECONDS,
            LeasePhase::Relevance => RELEVANCE_LEASE_TTL_SECONDS,
        }
    }
}

/// C1/C2/C7/C9 — reap this phase's stale rows for the space, then attempt the
/// acquire. Returns the fresh token on success and `None` when another holder
/// owns the operation (C2, the `LeaseHeld` signal).
///
/// The reap arm is M4's, with `phase` bound rather than spelled: an expired row
/// (C7) or one for a superseded `input_generation` (C9) is deleted, and a live
/// row for the same generation survives to make the insert affect zero rows.
pub async fn acquire(
    tx: &libsql::Transaction,
    phase: LeasePhase,
    space: &str,
    input_generation: i64,
    attempt: i64,
) -> Result<Option<String>, WenlanError> {
    tx.execute(
        "DELETE FROM grouping_leases
          WHERE phase = ?1 AND space = ?2
            AND (expires_at <= unixepoch() OR input_generation <> ?3)",
        libsql::params![phase.as_str(), space, input_generation],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 lease reap: {error}")))?;

    let token = uuid::Uuid::new_v4().to_string();
    let acquired = tx
        .execute(
            "INSERT INTO grouping_leases
                 (phase, space, input_generation, token, expires_at, attempt)
             VALUES (?1, ?2, ?3, ?4, unixepoch() + ?5, ?6)
             ON CONFLICT(phase, space, input_generation) DO NOTHING",
            libsql::params![
                phase.as_str(),
                space,
                input_generation,
                token.clone(),
                phase.ttl_seconds(),
                attempt
            ],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 lease acquire: {error}")))?;

    Ok((acquired == 1).then_some(token))
}

/// Gate E-1's ownership CAS: is `token` still the live holder?
///
/// Mirrors M4's finalize-time check, which reads the lease *before* any write
/// for the reason artifact 2 §2.1 fact 3 records — a finalize that has lost its
/// lease must write nothing, including nothing expensive. An expired-but-present
/// row is **not** ownership: `expires_at > unixepoch()` is part of the predicate,
/// because a row in that condition is still physically there and still visible
/// to any query that forgets the time (machine C's `expired` state).
pub async fn owns(
    tx: &libsql::Transaction,
    phase: LeasePhase,
    space: &str,
    input_generation: i64,
    token: &str,
) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT 1 FROM grouping_leases
              WHERE phase = ?1 AND space = ?2
                AND input_generation = ?3 AND token = ?4
                AND expires_at > unixepoch()
              LIMIT 1",
            libsql::params![phase.as_str(), space, input_generation, token],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 lease ownership: {error}")))?;
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 lease ownership row: {error}")))?
        .is_some())
}

/// C4/C5 — release the lease this worker holds. Token-scoped, so a release
/// arriving after a takeover cannot delete the new holder's row.
pub async fn release(
    tx: &libsql::Transaction,
    phase: LeasePhase,
    space: &str,
    input_generation: i64,
    token: &str,
) -> Result<u64, WenlanError> {
    tx.execute(
        "DELETE FROM grouping_leases
          WHERE phase = ?1 AND space = ?2
            AND input_generation = ?3 AND token = ?4",
        libsql::params![phase.as_str(), space, input_generation, token],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 lease release: {error}")))
}

/// C8 — the startup recovery scan's step 1 (S0-5): delete every expired lease
/// row belonging to a phase M6 owns.
///
/// Broader than the acquire-time reap, which is scoped to one space so it
/// cannot touch another space's live row; this one runs before the first turn,
/// when M6 holds nothing it should keep. M4's lazy reap-at-acquire is not
/// sufficient for M6, because M6 reads *reservations* to decide whether a group
/// has a durable reason — a reservation orphaned by a killed process would
/// otherwise hide its group from the frontier forever.
///
/// **Scoped to [`LeasePhase::ALL`], amending S0-5's "across all phases".**
/// `grouping_leases` is shared with M4, whose four delete statements are every
/// one of them scoped `WHERE phase = 'community'`. The justification above
/// reaches only M6's own reservations; M6 never reads a community-phase lease,
/// so reaping one buys nothing. It is harmless against M4 *as written today*
/// — M4's finalize ignores the delete's rowcount and its prepare reaps expired
/// rows unconditionally anyway — but that is a fact about M4's current code
/// rather than a contract, and a shadow subsystem should not silently depend on
/// it. One predicate removes the coupling.
pub async fn reap_expired_m6_phases(tx: &libsql::Transaction) -> Result<u64, WenlanError> {
    // Placeholders are generated digits, never phase text, so the phase values
    // stay bound parameters.
    let placeholders = (1..=LeasePhase::ALL.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let phases = LeasePhase::ALL
        .iter()
        .map(|phase| libsql::Value::from(phase.as_str()))
        .collect::<Vec<_>>();
    tx.execute(
        &format!(
            "DELETE FROM grouping_leases
              WHERE expires_at <= unixepoch() AND phase IN ({placeholders})"
        ),
        phases,
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 lease startup reap: {error}")))
}
