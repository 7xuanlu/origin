// SPDX-License-Identifier: Apache-2.0
//! One hermetic in-memory fixture shared by every PR-B2 gate module.
//!
//! The genesis substrate is the **shipped** DDL: migration 108's
//! `db::GENESIS_SUBSTRATE_DDL`, M4's `MemoryDB::COMMUNITY_SUBSTRATE_DDL` for
//! the one `grouping_leases` registry, and migration 109's
//! `frontier_policy::ensure_frontier_policy_tables` +
//! `remaining_substrate::ensure_remaining_substrate`. The two partial unique
//! indexes migration 108 creates are D4's entire exclusion mechanism, so a
//! hand-copied DDL would let the gates pass against a substrate production
//! never runs. R4 forbids `db` from handing a `libsql` handle across the
//! module boundary, so the shipped SQL arrives here as text and this fixture
//! brings its own connection.
//!
//! The primary-evidence tables (`edges`, `provenance_roots`, `pages`,
//! `page_links`, the claim-anchor trio) are fixture shapes carrying only the
//! columns the M6 readers touch, matching the PR-B1 test modules.

use crate::db::{MemoryDB, GENESIS_SUBSTRATE_DDL};
use crate::m6::candidates;
use crate::m6::frontier_policy::ensure_frontier_policy_tables;
use crate::m6::refresh_readiness::ensure_refresh_readiness_tables;
use crate::m6::remaining_substrate::ensure_remaining_substrate;

pub(super) struct GenesisDb {
    _database: libsql::Database,
    pub connection: libsql::Connection,
}

impl GenesisDb {
    pub async fn new() -> Self {
        let database = libsql::Builder::new_local(":memory:")
            .build()
            .await
            .expect("build in-memory database");
        let connection = database.connect().expect("connect in-memory database");
        connection
            .execute_batch(
                "CREATE TABLE spaces (
                     id   TEXT PRIMARY KEY,
                     name TEXT NOT NULL UNIQUE
                 );
                 CREATE TABLE app_metadata (
                     key   TEXT PRIMARY KEY,
                     value TEXT NOT NULL
                 );
                 CREATE TABLE provenance_roots (
                     root_id                TEXT PRIMARY KEY,
                     root_kind              TEXT NOT NULL,
                     independence_group_id  TEXT NOT NULL,
                     status                 TEXT NOT NULL
                 );
                 CREATE TABLE edges (
                     edge_id      TEXT PRIMARY KEY,
                     src_kind     TEXT NOT NULL,
                     src_id       TEXT NOT NULL,
                     dst_kind     TEXT NOT NULL,
                     dst_id       TEXT NOT NULL,
                     edge_type    TEXT NOT NULL DEFAULT 'relates',
                     grounded     INTEGER NOT NULL,
                     root_id      TEXT,
                     space        TEXT NOT NULL,
                     valid_until  INTEGER
                 );
                 CREATE TABLE pages (
                     id      TEXT PRIMARY KEY,
                     title   TEXT NOT NULL,
                     content TEXT NOT NULL DEFAULT '',
                     version INTEGER NOT NULL DEFAULT 1,
                     space   TEXT NOT NULL DEFAULT '',
                     status  TEXT NOT NULL DEFAULT 'active'
                 );
                 CREATE TABLE page_links (
                     source_page_id TEXT NOT NULL,
                     target_page_id TEXT,
                     label_key      TEXT NOT NULL,
                     label          TEXT NOT NULL,
                     PRIMARY KEY (source_page_id, label_key)
                 );
                 CREATE TABLE page_truth_state (
                     page_id        TEXT NOT NULL,
                     page_version   INTEGER NOT NULL,
                     support_status TEXT NOT NULL
                 );
                 CREATE TABLE page_version_claims (
                     page_id           TEXT NOT NULL,
                     page_version      INTEGER NOT NULL,
                     claim_revision_id TEXT NOT NULL,
                     ordinal           INTEGER NOT NULL,
                     PRIMARY KEY (page_id, page_version, claim_revision_id)
                 );
                 CREATE TABLE claim_anchors (
                     claim_revision_id TEXT NOT NULL,
                     source_doc_id     TEXT NOT NULL,
                     source_version    INTEGER NOT NULL,
                     span_start        INTEGER NOT NULL,
                     span_end          INTEGER NOT NULL,
                     span_digest       TEXT NOT NULL,
                     created_at        INTEGER NOT NULL
                 );
                 CREATE TABLE m6_overview_subscriptions (
                     subscription_id TEXT PRIMARY KEY,
                     scope_kind      TEXT NOT NULL,
                     scope_id        TEXT NOT NULL,
                     state           TEXT NOT NULL
                 );",
            )
            .await
            .expect("create primary-evidence fixture schema");

        connection
            .execute_batch(MemoryDB::COMMUNITY_SUBSTRATE_DDL)
            .await
            .expect("install the shipped grouping_leases registry");
        connection
            .execute_batch(GENESIS_SUBSTRATE_DDL)
            .await
            .expect("install the shipped migration-108 genesis substrate");

        let tx = connection
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .expect("begin schema transaction");
        ensure_frontier_policy_tables(&tx)
            .await
            .expect("install the migration-109 frontier policy substrate");
        // Same order the migration itself uses (`db.rs`, migration 109), so the
        // fixture cannot pass against a table layout production never builds.
        // Added in PR-B3: `evidence::note_shadow_readiness` writes `m6_readiness`.
        ensure_refresh_readiness_tables(&tx)
            .await
            .expect("install the migration-109 refresh readiness substrate");
        ensure_remaining_substrate(&tx)
            .await
            .expect("install the rest of migration 109");
        tx.commit().await.expect("commit schema transaction");

        Self {
            _database: database,
            connection,
        }
    }

    pub async fn tx(&self) -> libsql::Transaction {
        self.connection
            .transaction_with_behavior(libsql::TransactionBehavior::Deferred)
            .await
            .expect("begin fixture transaction")
    }

    pub async fn exec(&self, sql: &str, params: impl libsql::params::IntoParams) {
        self.connection
            .execute(sql, params)
            .await
            .unwrap_or_else(|error| panic!("fixture statement failed: {error}\n{sql}"));
    }

    pub async fn scalar(&self, sql: &str, params: impl libsql::params::IntoParams) -> i64 {
        let mut rows = self
            .connection
            .query(sql, params)
            .await
            .unwrap_or_else(|error| panic!("fixture query failed: {error}\n{sql}"));
        rows.next()
            .await
            .expect("read fixture scalar row")
            .expect("fixture scalar row present")
            .get(0)
            .expect("decode fixture scalar")
    }

    pub async fn seed_space(&self, id: &str, name: &str) {
        self.exec(
            "INSERT INTO spaces (id, name) VALUES (?1, ?2)",
            libsql::params![id, name],
        )
        .await;
    }

    /// One grounded edge and its active, non-generated provenance root — the
    /// shape artifact 5 §1.2's eligible set selects.
    pub async fn seed_evidence(&self, root_id: &str, space: &str, group: &str, entity_id: &str) {
        self.exec(
            "INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status)
             VALUES (?1, 'document_ingest', ?2, 'active')",
            libsql::params![root_id, group],
        )
        .await;
        self.exec(
            "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id,
                                grounded, root_id, space, valid_until)
             VALUES (?1, 'entity', ?2, 'entity', 'peer', 1, ?1, ?3, NULL)",
            libsql::params![root_id, entity_id, space],
        )
        .await;
    }

    /// D1 in its own transaction, returning the epoch candidates key on.
    pub async fn open_epoch(&self, space_id: &str, space: &str) -> i64 {
        let tx = self.tx().await;
        let state = candidates::open_coverage_epoch(&tx, space_id, space)
            .await
            .expect("open coverage epoch");
        tx.commit().await.expect("commit coverage epoch");
        state.coverage_epoch
    }

    /// Drop the space's held leases, as the turn driver's release would (C4).
    /// PR-B2 has no driver, so a test that wants a second prepare in the same
    /// generation says so explicitly rather than pretending the lease vanished.
    pub async fn release_leases(&self) {
        self.exec("DELETE FROM grouping_leases", ()).await;
    }

    /// Move a stored clock backwards. S0-11's whole point: a test advances time
    /// only by moving a stored value, never by sleeping or by injecting a
    /// Rust-side `now`.
    pub async fn age_candidate(&self, candidate_id: &str, seconds: i64) {
        self.exec(
            "UPDATE genesis_candidates
                SET created_at = created_at - ?2,
                    updated_at = updated_at - ?2,
                    next_attempt_at = next_attempt_at - ?2
              WHERE candidate_id = ?1",
            libsql::params![candidate_id, seconds],
        )
        .await;
    }

    pub async fn age_frontier(&self, group: &str, seconds: i64) {
        self.exec(
            "UPDATE genesis_frontier
                SET first_seen_at = first_seen_at - ?2
              WHERE independence_group_id = ?1",
            libsql::params![group, seconds],
        )
        .await;
    }

    pub async fn candidate_state(&self, candidate_id: &str) -> String {
        let mut rows = self
            .connection
            .query(
                "SELECT state FROM genesis_candidates WHERE candidate_id = ?1",
                libsql::params![candidate_id],
            )
            .await
            .expect("query candidate state");
        rows.next()
            .await
            .expect("read candidate row")
            .expect("candidate row present")
            .get(0)
            .expect("decode candidate state")
    }

    pub async fn candidate_reason(&self, candidate_id: &str) -> Option<String> {
        let mut rows = self
            .connection
            .query(
                "SELECT reason FROM genesis_candidates WHERE candidate_id = ?1",
                libsql::params![candidate_id],
            )
            .await
            .expect("query candidate reason");
        rows.next()
            .await
            .expect("read candidate row")
            .expect("candidate row present")
            .get(0)
            .expect("decode candidate reason")
    }
}

/// A candidate row in `observed`, committed, with its claims left to the
/// caller's `prepare_candidate`.
pub(super) async fn observe(
    db: &GenesisDb,
    candidate_id: &str,
    space: &str,
    signal_kind: crate::m6::identity::SignalKind,
    coverage_epoch: i64,
    input_generation: i64,
) {
    let tx = db.tx().await;
    candidates::observe_candidate(
        &tx,
        &candidates::ObservedCandidate {
            candidate_id,
            slot_id: &format!("slot-{candidate_id}"),
            page_id: &format!("m6p_{candidate_id}"),
            space,
            signal_kind,
            coverage_epoch,
            input_generation,
            active_root_digest: "digest-0",
        },
    )
    .await
    .expect("observe candidate");
    tx.commit().await.expect("commit observed candidate");
}

/// `[ClaimReservation]` over one group, one root per group.
pub(super) fn claims(pairs: &[(&str, &str)]) -> Vec<candidates::ClaimReservation> {
    pairs
        .iter()
        .map(|(root_id, group)| candidates::ClaimReservation {
            root_id: (*root_id).to_string(),
            independence_group_id: (*group).to_string(),
        })
        .collect()
}
