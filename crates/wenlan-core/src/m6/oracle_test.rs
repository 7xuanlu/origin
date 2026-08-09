// SPDX-License-Identifier: Apache-2.0
//! `recompute_full` and `diff` (spec §6.1, §9.1: "the field census; a
//! deliberately divergent snapshot is detected").

use super::identity::SignalKind;
use super::oracle::{diff, recompute_full, GenesisSnapshot};
use super::signals::CandidateProposal;

struct TestDb {
    _database: libsql::Database,
    connection: libsql::Connection,
}

impl TestDb {
    async fn new() -> Self {
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
                 CREATE TABLE m6_overview_subscriptions (
                     subscription_id TEXT PRIMARY KEY,
                     scope_kind      TEXT NOT NULL,
                     scope_id        TEXT NOT NULL,
                     state           TEXT NOT NULL
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
                 CREATE TABLE communities (
                     community_id TEXT PRIMARY KEY,
                     space        TEXT NOT NULL,
                     retired_at   INTEGER
                 );
                 CREATE TABLE community_members (
                     space        TEXT NOT NULL,
                     node_id      TEXT NOT NULL,
                     community_id TEXT NOT NULL,
                     attachment   TEXT NOT NULL,
                     PRIMARY KEY (space, node_id)
                 );",
            )
            .await
            .expect("create fixture schema");
        Self {
            _database: database,
            connection,
        }
    }

    async fn tx(&self) -> libsql::Transaction {
        self.connection
            .transaction_with_behavior(libsql::TransactionBehavior::Deferred)
            .await
            .expect("begin fixture transaction")
    }

    async fn seed_space(&self, id: &str, name: &str) {
        self.connection
            .execute(
                "INSERT INTO spaces (id, name) VALUES (?1, ?2)",
                libsql::params![id, name],
            )
            .await
            .expect("seed space");
    }

    async fn seed_entity_evidence(&self, id: &str, space: &str, group_id: &str, entity_id: &str) {
        self.connection
            .execute(
                "INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status)
                 VALUES (?1, 'document_ingest', ?2, 'active')",
                libsql::params![id, group_id],
            )
            .await
            .expect("seed root");
        self.connection
            .execute(
                "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id, grounded, root_id, space, valid_until)
                 VALUES (?1, 'entity', ?2, 'entity', 'peer', 1, ?1, ?3, NULL)",
                libsql::params![id, entity_id, space],
            )
            .await
            .expect("seed edge");
    }

    async fn seed_community(&self, community_id: &str, space: &str) {
        self.connection
            .execute(
                "INSERT INTO communities (community_id, space, retired_at) VALUES (?1, ?2, NULL)",
                libsql::params![community_id, space],
            )
            .await
            .expect("seed community");
    }

    async fn seed_community_member(&self, space: &str, node_id: &str, community_id: &str) {
        self.connection
            .execute(
                "INSERT INTO community_members (space, node_id, community_id, attachment)
                 VALUES (?1, ?2, ?3, 'core')",
                libsql::params![space, node_id, community_id],
            )
            .await
            .expect("seed community member");
    }
}

async fn seed_three_admitting_groups(db: &TestDb, space: &str) {
    for (i, group) in ["group-a", "group-b", "group-c"].into_iter().enumerate() {
        db.seed_entity_evidence(
            &format!("root-{i}-0"),
            space,
            group,
            &format!("entity-{i}-0"),
        )
        .await;
        db.seed_entity_evidence(
            &format!("root-{i}-1"),
            space,
            group,
            &format!("entity-{i}-1"),
        )
        .await;
    }
}

fn sample_proposal(group_count: i64) -> CandidateProposal {
    CandidateProposal {
        slot_id: "slot-a".to_string(),
        page_id: "m6p_a".to_string(),
        signal_kind: SignalKind::SpaceOverview,
        root_ids: vec!["root-0".to_string(), "root-1".to_string()],
        group_count,
    }
}

/// Vacuous case: an empty install recomputes to an empty snapshot, even with
/// the durability gate open — there is simply nothing behind it.
#[tokio::test]
async fn empty_install_recomputes_empty() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    let tx = db.tx().await;
    let snapshot = recompute_full(&tx, "space-id-a", 1, true).await.unwrap();
    assert!(snapshot.proposals.is_empty());
    assert!(diff(&snapshot, &GenesisSnapshot::default()).is_empty());
}

/// `recompute_full` reflects exactly what a direct signal call would
/// produce — it is a reader over the same primary evidence, not a second
/// implementation of the count.
#[tokio::test]
async fn recompute_full_matches_the_signal_reader() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_three_admitting_groups(&db, "space-a").await;

    let tx = db.tx().await;
    let snapshot = recompute_full(&tx, "space-id-a", 1, false).await.unwrap();
    assert_eq!(snapshot.proposals.len(), 1);
    let proposal = &snapshot.proposals[0];
    assert_eq!(proposal.signal_kind, SignalKind::SpaceOverview);
    assert_eq!(proposal.group_count, 3);
}

/// A durable partition wires evidence-cluster and community-overview into
/// the full snapshot alongside space-overview; an undurable one keeps them
/// out of the exact same fixture — proving the flag actually gates the
/// two, not just the direct signal calls.
#[tokio::test]
async fn recompute_full_wires_community_signals_only_when_durable() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a").await;
    for (i, group) in ["group-a", "group-b", "group-c"].into_iter().enumerate() {
        for j in 0..2 {
            let entity = format!("entity-{i}-{j}");
            db.seed_community_member("space-a", &entity, "comm-1").await;
            db.seed_entity_evidence(&format!("root-{i}-{j}"), "space-a", group, &entity)
                .await;
        }
    }

    let tx = db.tx().await;
    let undurable = recompute_full(&tx, "space-id-a", 1, false).await.unwrap();
    let kinds: Vec<_> = undurable.proposals.iter().map(|p| p.signal_kind).collect();
    assert_eq!(
        kinds,
        vec![SignalKind::SpaceOverview],
        "undurable must exclude evidence-cluster and community-overview \
         even though the underlying evidence otherwise qualifies"
    );

    let durable = recompute_full(&tx, "space-id-a", 1, true).await.unwrap();
    let mut kinds: Vec<_> = durable.proposals.iter().map(|p| p.signal_kind).collect();
    kinds.sort_by_key(|k| format!("{k:?}"));
    assert_eq!(
        kinds,
        vec![
            SignalKind::CommunityOverview,
            SignalKind::EvidenceCluster,
            SignalKind::SpaceOverview,
        ],
        "durable must admit all three signals this fixture qualifies for"
    );
}

#[tokio::test]
async fn diff_is_empty_for_identical_snapshots() {
    let a = GenesisSnapshot {
        proposals: vec![sample_proposal(3)],
        ..Default::default()
    };
    let b = a.clone();
    assert!(diff(&a, &b).is_empty());
}

/// The field census: a divergence in a field other than the key
/// (`group_count`) is caught, not just presence/absence.
#[tokio::test]
async fn diff_detects_field_level_divergence() {
    let a = GenesisSnapshot {
        proposals: vec![sample_proposal(3)],
        ..Default::default()
    };
    let b = GenesisSnapshot {
        proposals: vec![sample_proposal(4)],
        ..Default::default()
    };
    let divergences = diff(&a, &b);
    assert_eq!(divergences.len(), 1);
    assert!(
        divergences[0].detail.contains("group_count"),
        "{:?}",
        divergences[0]
    );
}

#[tokio::test]
async fn diff_detects_added_and_removed_proposals() {
    let mut extra = sample_proposal(3);
    extra.slot_id = "slot-b".to_string();

    let a = GenesisSnapshot {
        proposals: vec![sample_proposal(3)],
        ..Default::default()
    };
    let b = GenesisSnapshot {
        proposals: vec![sample_proposal(3), extra],
        ..Default::default()
    };
    let divergences = diff(&a, &b);
    assert_eq!(divergences.len(), 1);
    assert_eq!(divergences[0].key, "slot-b");
    assert!(divergences[0].detail.contains("absent in a"));
}
