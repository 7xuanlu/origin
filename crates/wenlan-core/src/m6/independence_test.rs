// SPDX-License-Identifier: Apache-2.0
//! G2 — the independence floor counts groups, not rows (spec §8, artifact-12
//! rows G2.1-G2.8).

use super::independence::{
    admits, distinct_group_count, GroupCountScope, INDEPENDENCE_GROUP_FLOOR,
};

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
                "CREATE TABLE provenance_roots (
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
                     grounded     INTEGER NOT NULL,
                     root_id      TEXT,
                     valid_until  INTEGER
                 );
                 CREATE TABLE pages (
                     id    TEXT PRIMARY KEY,
                     title TEXT NOT NULL
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

    /// Seed one root and one live, grounded edge naming it, pointed at a
    /// generic entity-to-entity `relates` shape (no page endpoint).
    async fn seed_grounded(&self, edge_id: &str, root_id: &str, group_id: &str) {
        self.seed_root(root_id, group_id, "document_ingest", "active")
            .await;
        self.seed_edge(edge_id, root_id, true, None, "entity", "entity")
            .await;
    }

    async fn seed_root(&self, root_id: &str, group_id: &str, kind: &str, status: &str) {
        self.connection
            .execute(
                "INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status)
                 VALUES (?1, ?2, ?3, ?4)",
                libsql::params![root_id, kind, group_id, status],
            )
            .await
            .expect("seed root");
    }

    #[allow(clippy::too_many_arguments)]
    async fn seed_edge(
        &self,
        edge_id: &str,
        root_id: &str,
        grounded: bool,
        valid_until: Option<i64>,
        src_kind: &str,
        dst_kind: &str,
    ) {
        self.connection
            .execute(
                "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id, grounded, root_id, valid_until)
                 VALUES (?1, ?2, 'x', ?3, 'y', ?4, ?5, ?6)",
                libsql::params![
                    edge_id,
                    src_kind,
                    dst_kind,
                    grounded,
                    root_id,
                    valid_until,
                ],
            )
            .await
            .expect("seed edge");
    }

    async fn seed_page(&self, id: &str, title: &str) {
        self.connection
            .execute(
                "INSERT INTO pages (id, title) VALUES (?1, ?2)",
                libsql::params![id, title],
            )
            .await
            .expect("seed page");
    }
}

/// G2.1 positive control: ten chunks (edges/roots) of one document collapse
/// into one group, not ten — `COUNT(DISTINCT ...)`, never `COUNT(*)`.
#[tokio::test]
async fn counts_groups_not_rows() {
    let db = TestDb::new().await;
    for i in 0..10 {
        db.seed_grounded(&format!("edge-{i}"), &format!("root-{i}"), "group-a")
            .await;
    }
    let tx = db.tx().await;
    let count = distinct_group_count(&tx, &GroupCountScope::unscoped())
        .await
        .expect("count");
    assert_eq!(
        count, 1,
        "ten chunks of one document must collapse to one group"
    );
    assert!(!admits(count), "one group never clears the floor of 3");
}

/// G2.2 positive control: a retracted root's group stops counting.
#[tokio::test]
async fn retracted_root_group_excluded() {
    let db = TestDb::new().await;
    db.seed_grounded("edge-a", "root-a", "group-a").await;
    db.seed_grounded("edge-b", "root-b", "group-b").await;
    let tx = db.tx().await;
    assert_eq!(
        distinct_group_count(&tx, &GroupCountScope::unscoped())
            .await
            .unwrap(),
        2
    );
    tx.execute(
        "UPDATE provenance_roots SET status = 'failed' WHERE root_id = 'root-a'",
        (),
    )
    .await
    .expect("retract root-a");
    assert_eq!(
        distinct_group_count(&tx, &GroupCountScope::unscoped())
            .await
            .unwrap(),
        1,
        "a retracted root's group must stop counting"
    );
}

/// G2.3 positive control: an ungrounded edge contributes nothing.
#[tokio::test]
async fn ungrounded_edge_excluded() {
    let db = TestDb::new().await;
    db.seed_grounded("edge-a", "root-a", "group-a").await;
    db.seed_root("root-b", "group-b", "document_ingest", "active")
        .await;
    db.seed_edge("edge-b", "root-b", false, None, "entity", "entity")
        .await;
    let tx = db.tx().await;
    assert_eq!(
        distinct_group_count(&tx, &GroupCountScope::unscoped())
            .await
            .unwrap(),
        1,
        "an ungrounded edge must not contribute its group"
    );
}

/// G2.4 positive control: a retracted (valid_until set) edge contributes
/// nothing, even though its root is still active.
#[tokio::test]
async fn retracted_edge_excluded() {
    let db = TestDb::new().await;
    db.seed_grounded("edge-a", "root-a", "group-a").await;
    db.seed_root("root-b", "group-b", "document_ingest", "active")
        .await;
    db.seed_edge(
        "edge-b",
        "root-b",
        true,
        Some(1_700_000_000),
        "entity",
        "entity",
    )
    .await;
    let tx = db.tx().await;
    assert_eq!(
        distinct_group_count(&tx, &GroupCountScope::unscoped())
            .await
            .unwrap(),
        1,
        "a retracted edge must not contribute its group"
    );
}

/// G2.5 positive control: a generated root contributes nothing — no
/// self-bootstrapping.
#[tokio::test]
async fn generated_root_excluded() {
    let db = TestDb::new().await;
    db.seed_grounded("edge-a", "root-a", "group-a").await;
    db.seed_root("root-b", "group-b", "generated", "active")
        .await;
    db.seed_edge("edge-b", "root-b", true, None, "entity", "entity")
        .await;
    let tx = db.tx().await;
    assert_eq!(
        distinct_group_count(&tx, &GroupCountScope::unscoped())
            .await
            .unwrap(),
        1,
        "a generated root must never justify more M6 output"
    );
}

/// G2.6 positive control: the floor is 3, so a 2-group cluster is refused
/// and a 3-group cluster admits.
#[tokio::test]
async fn floor_boundary() {
    assert_eq!(INDEPENDENCE_GROUP_FLOOR, 3);
    assert!(!admits(2), "a 2-group cluster must be refused");
    assert!(admits(3), "a 3-group cluster must admit");
}

/// G2.7 positive control: an Overview page contributes nothing, keyed on
/// `lower(title)`, per S0-164 (never `pages.kind`, which `drift_guard`
/// teeth #16 still fences).
#[tokio::test]
async fn overview_page_excluded() {
    let db = TestDb::new().await;
    db.seed_grounded("edge-a", "root-a", "group-a").await;

    db.seed_page("page-overview", "Overview").await;
    db.seed_root("root-b", "group-b", "document_ingest", "active")
        .await;
    db.seed_edge("edge-b", "root-b", true, None, "page", "entity")
        .await;
    db.connection
        .execute(
            "UPDATE edges SET src_id = 'page-overview' WHERE edge_id = 'edge-b'",
            (),
        )
        .await
        .expect("point edge-b at the Overview page");

    let tx = db.tx().await;
    assert_eq!(
        distinct_group_count(&tx, &GroupCountScope::unscoped())
            .await
            .unwrap(),
        1,
        "evidence mediated through an Overview page must not contribute"
    );
}

/// G2.8 positive control: all human authorship collapses into the single
/// group `human:local` (artifact 3 §7.2, boundary case B28), so three human
/// deltas can never clear the 3-group floor on their own.
#[tokio::test]
async fn human_capture_collapses_to_one_group() {
    let db = TestDb::new().await;
    for i in 0..3 {
        db.seed_root(
            &format!("root-{i}"),
            "human:local",
            "human_capture",
            "active",
        )
        .await;
        db.seed_edge(
            &format!("edge-{i}"),
            &format!("root-{i}"),
            true,
            None,
            "entity",
            "entity",
        )
        .await;
    }
    let tx = db.tx().await;
    let count = distinct_group_count(&tx, &GroupCountScope::unscoped())
        .await
        .unwrap();
    assert_eq!(count, 1, "human authorship must collapse into one group");
    assert!(
        !admits(count),
        "three human deltas alone must not clear the floor"
    );
}

/// Vacuous case: an install with zero grounded edges admits zero
/// candidates.
#[tokio::test]
async fn empty_install_counts_zero() {
    let db = TestDb::new().await;
    let tx = db.tx().await;
    let count = distinct_group_count(&tx, &GroupCountScope::unscoped())
        .await
        .unwrap();
    assert_eq!(count, 0);
    assert!(!admits(count));
}

/// A caller-supplied scope narrows the count without relaxing R1-R5: a
/// scoped-out group with a retracted root still drops out.
#[tokio::test]
async fn scope_narrows_without_relaxing_shared_conjuncts() {
    let db = TestDb::new().await;
    db.seed_grounded("edge-a", "root-a", "group-a").await;
    db.seed_grounded("edge-b", "root-b", "group-b").await;
    db.seed_grounded("edge-c", "root-c", "group-c").await;
    let tx = db.tx().await;

    let scope = GroupCountScope {
        predicate: "r.independence_group_id IN (?1, ?2)",
        params: vec![
            libsql::Value::Text("group-a".into()),
            libsql::Value::Text("group-b".into()),
        ],
    };
    assert_eq!(distinct_group_count(&tx, &scope).await.unwrap(), 2);

    tx.execute(
        "UPDATE provenance_roots SET status = 'failed' WHERE root_id = 'root-a'",
        (),
    )
    .await
    .expect("retract root-a");
    assert_eq!(
        distinct_group_count(&tx, &scope).await.unwrap(),
        1,
        "R2 (active roots only) must still apply inside a narrowed scope"
    );
}
