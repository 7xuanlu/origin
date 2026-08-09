// SPDX-License-Identifier: Apache-2.0
//! The four D2 signals (spec §9.1: "each of the four signals at
//! threshold - 1, threshold, threshold + 1").

use super::identity::{self, page_id, slot_id_space_overview, SignalKind};
use super::signals::{community_overview, evidence_cluster, orphan_wikilink, space_overview};

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

    /// One grounded, live root/edge pair naming a distinct entity, in one
    /// space and one independence group.
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

    /// Seed a `supports` edge naming `revision_id` and `root_id`, written
    /// with `grounded = 0` — matching production (the supports-edge writer
    /// never asserts grounding, only inherits it via `root_id`).
    #[allow(clippy::too_many_arguments)]
    async fn seed_supports_edge(
        &self,
        edge_id: &str,
        revision_id: &str,
        root_id: &str,
        space: &str,
    ) {
        self.connection
            .execute(
                "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id, edge_type, grounded, root_id, space, valid_until)
                 VALUES (?1, 'claim_revision', ?2, 'memory', 'mem-x', 'supports', 0, ?3, ?4, NULL)",
                libsql::params![edge_id, revision_id, root_id, space],
            )
            .await
            .expect("seed supports edge");
    }

    /// Seed a page carrying `content` at `version`, `supported` in
    /// `page_truth_state` at that same version.
    async fn seed_page(&self, page_id: &str, space: &str, content: &str, version: i64) {
        self.connection
            .execute(
                "INSERT INTO pages (id, title, content, version, space, status)
                 VALUES (?1, 'Note', ?2, ?3, ?4, 'active')",
                libsql::params![page_id, content, version, space],
            )
            .await
            .expect("seed page");
        self.connection
            .execute(
                "INSERT INTO page_truth_state (page_id, page_version, support_status)
                 VALUES (?1, ?2, 'supported')",
                libsql::params![page_id, version],
            )
            .await
            .expect("seed page_truth_state");
    }

    /// Seed one orphan `page_links` row for `page_id` naming `label`.
    async fn seed_orphan_link(&self, page_id: &str, label: &str) {
        let label_key =
            super::label_key::normalize_label_key(label).expect("label normalizes for the fixture");
        self.connection
            .execute(
                "INSERT INTO page_links (source_page_id, target_page_id, label_key, label)
                 VALUES (?1, NULL, ?2, ?3)",
                libsql::params![page_id, label_key, label],
            )
            .await
            .expect("seed orphan link");
    }

    /// Seed a claim revision anchored to `content[span_start..span_end]` at
    /// `page_id`'s `version`. The digest is computed for real from the
    /// fixture body, so a gate exercises the same verification production
    /// performs rather than a stand-in that would pass regardless.
    #[allow(clippy::too_many_arguments)]
    async fn seed_claim_anchor(
        &self,
        page_id: &str,
        version: i64,
        revision_id: &str,
        content: &str,
        span: std::ops::Range<usize>,
        ordinal: i64,
    ) {
        let digest = crate::provenance::revision_content_digest(&content[span.clone()]);
        self.connection
            .execute(
                "INSERT INTO page_version_claims (page_id, page_version, claim_revision_id, ordinal)
                 VALUES (?1, ?2, ?3, ?4)",
                libsql::params![page_id, version, revision_id, ordinal],
            )
            .await
            .expect("seed page_version_claims");
        self.connection
            .execute(
                "INSERT INTO claim_anchors (claim_revision_id, source_doc_id, source_version, span_start, span_end, span_digest, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
                libsql::params![
                    revision_id,
                    page_id,
                    version,
                    span.start as i64,
                    span.end as i64,
                    digest,
                ],
            )
            .await
            .expect("seed claim_anchors");
    }

    async fn seed_community(&self, community_id: &str, space: &str, retired_at: Option<i64>) {
        self.connection
            .execute(
                "INSERT INTO communities (community_id, space, retired_at) VALUES (?1, ?2, ?3)",
                libsql::params![community_id, space, retired_at],
            )
            .await
            .expect("seed community");
    }

    async fn seed_community_member(
        &self,
        space: &str,
        node_id: &str,
        community_id: &str,
        attachment: &str,
    ) {
        self.connection
            .execute(
                "INSERT INTO community_members (space, node_id, community_id, attachment)
                 VALUES (?1, ?2, ?3, ?4)",
                libsql::params![space, node_id, community_id, attachment],
            )
            .await
            .expect("seed community member");
    }

    async fn seed_subscription(&self, scope_kind: &str, scope_id: &str, state: &str) {
        self.connection
            .execute(
                "INSERT INTO m6_overview_subscriptions (subscription_id, scope_kind, scope_id, state)
                 VALUES (?1, ?2, ?3, ?4)",
                libsql::params![format!("sub-{scope_kind}-{scope_id}"), scope_kind, scope_id, state],
            )
            .await
            .expect("seed subscription");
    }
}

/// Three groups, six distinct entities: clears both the D1 floor and the
/// 5-grounded-node floor.
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

#[tokio::test]
async fn admits_when_floor_and_node_count_clear() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_three_admitting_groups(&db, "space-a").await;

    let tx = db.tx().await;
    let proposal = space_overview(&tx, "space-id-a")
        .await
        .expect("query succeeds")
        .expect("candidate admitted");

    assert_eq!(proposal.signal_kind, SignalKind::SpaceOverview);
    assert_eq!(proposal.group_count, 3);
    assert_eq!(proposal.root_ids.len(), 6);
    assert_eq!(proposal.slot_id, slot_id_space_overview("space-id-a"));
    assert_eq!(proposal.page_id, page_id(&proposal.slot_id));
}

#[tokio::test]
async fn rejects_below_group_floor() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_entity_evidence("root-0", "space-a", "group-a", "entity-0")
        .await;
    db.seed_entity_evidence("root-1", "space-a", "group-b", "entity-1")
        .await;

    let tx = db.tx().await;
    assert!(
        space_overview(&tx, "space-id-a").await.unwrap().is_none(),
        "two groups must not admit"
    );
}

#[tokio::test]
async fn rejects_below_grounded_node_floor() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    // Three groups (clears D1) but only two distinct entities (misses the
    // signal-4 "≥5 grounded nodes" extra).
    db.seed_entity_evidence("root-0", "space-a", "group-a", "entity-0")
        .await;
    db.seed_entity_evidence("root-1", "space-a", "group-b", "entity-0")
        .await;
    db.seed_entity_evidence("root-2", "space-a", "group-c", "entity-1")
        .await;

    let tx = db.tx().await;
    assert!(
        space_overview(&tx, "space-id-a").await.unwrap().is_none(),
        "three groups over two nodes must not admit signal 4's extra floor"
    );
}

#[tokio::test]
async fn rejects_with_active_space_subscription() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_three_admitting_groups(&db, "space-a").await;
    db.seed_subscription("space", "space-a", "active").await;

    let tx = db.tx().await;
    assert!(
        space_overview(&tx, "space-id-a").await.unwrap().is_none(),
        "an active space-overview subscription must block a second candidate"
    );
}

#[tokio::test]
async fn admits_with_only_a_detached_subscription() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_three_admitting_groups(&db, "space-a").await;
    db.seed_subscription("space", "space-a", "detached").await;

    let tx = db.tx().await;
    assert!(
        space_overview(&tx, "space-id-a").await.unwrap().is_some(),
        "a detached subscription must not block re-subscription"
    );
}

#[tokio::test]
async fn evidence_from_another_space_does_not_count() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_space("space-id-b", "space-b").await;
    seed_three_admitting_groups(&db, "space-b").await;

    let tx = db.tx().await;
    assert!(
        space_overview(&tx, "space-id-a").await.unwrap().is_none(),
        "another space's evidence must not admit this space's candidate"
    );
}

// ---------------------------------------------------------------------------
// Signal 1 — evidence-cluster
// ---------------------------------------------------------------------------

/// Three groups, six distinct entities, all `attachment = 'core'` members of
/// `community_id` in `space` — clears both the D1 floor and (incidentally)
/// signal 3's 5-grounded-node floor.
async fn seed_three_admitting_groups_in_community(db: &TestDb, space: &str, community_id: &str) {
    for (i, group) in ["group-a", "group-b", "group-c"].into_iter().enumerate() {
        for j in 0..2 {
            let entity = format!("entity-{i}-{j}");
            db.seed_community_member(space, &entity, community_id, "core")
                .await;
            db.seed_entity_evidence(&format!("root-{i}-{j}"), space, group, &entity)
                .await;
        }
    }
}

#[tokio::test]
async fn evidence_cluster_admits_when_durable_and_core_members_clear_floor() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    seed_three_admitting_groups_in_community(&db, "space-a", "comm-1").await;

    let tx = db.tx().await;
    let proposals = evidence_cluster(&tx, "space-id-a", true)
        .await
        .expect("query succeeds");
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].group_count, 3);
    assert_eq!(proposals[0].signal_kind, SignalKind::EvidenceCluster);
    let expected_slot = identity::slot_id_evidence_cluster(
        "space-id-a",
        &[
            "group-a".to_string(),
            "group-b".to_string(),
            "group-c".to_string(),
        ],
    );
    assert_eq!(proposals[0].slot_id, expected_slot);
    assert_eq!(proposals[0].page_id, page_id(&proposals[0].slot_id));
}

/// Positive control for the durability gate: the identical fixture above
/// admits nothing when `community_partition_durable` is false, proving the
/// admission above was for real rather than the gate being a no-op.
#[tokio::test]
async fn evidence_cluster_rejects_when_partition_not_durable() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    seed_three_admitting_groups_in_community(&db, "space-a", "comm-1").await;

    let tx = db.tx().await;
    assert!(
        evidence_cluster(&tx, "space-id-a", false)
            .await
            .unwrap()
            .is_empty(),
        "an undurable partition must admit nothing, even with otherwise-qualifying evidence"
    );
}

#[tokio::test]
async fn evidence_cluster_ignores_retired_community() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-retired", "space-a", Some(1)).await;
    db.seed_community("comm-live", "space-a", None).await;
    seed_three_admitting_groups_in_community(&db, "space-a", "comm-retired").await;

    let tx = db.tx().await;
    assert!(
        evidence_cluster(&tx, "space-id-a", true)
            .await
            .unwrap()
            .is_empty(),
        "a retired community must not produce a candidate, even a clearly-qualifying one"
    );

    // Positive control: the identical membership under a non-retired
    // community does admit — proves the rejection above was for real, not a
    // fixture mistake. `community_members`' primary key is `(space,
    // node_id)` (one community per node), so this moves the rows rather
    // than duplicating them.
    tx.execute(
        "UPDATE community_members SET community_id = 'comm-live' WHERE community_id = 'comm-retired'",
        (),
    )
    .await
    .expect("move membership onto the live community");
    let proposals = evidence_cluster(&tx, "space-id-a", true)
        .await
        .expect("query succeeds");
    assert_eq!(proposals.len(), 1, "the live community must admit");
}

/// G2-style discriminator for the `attachment = 'core'` scope: a peripheral
/// member's evidence must not count toward the floor. Positive control:
/// promoting the same entity's evidence to a core member admits.
#[tokio::test]
async fn evidence_cluster_peripheral_members_do_not_count() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    db.seed_community_member("space-a", "entity-core-0", "comm-1", "core")
        .await;
    db.seed_entity_evidence("root-0", "space-a", "group-a", "entity-core-0")
        .await;
    db.seed_community_member("space-a", "entity-core-1", "comm-1", "core")
        .await;
    db.seed_entity_evidence("root-1", "space-a", "group-b", "entity-core-1")
        .await;
    db.seed_community_member("space-a", "entity-peripheral", "comm-1", "peripheral")
        .await;
    db.seed_entity_evidence("root-2", "space-a", "group-c", "entity-peripheral")
        .await;

    let tx = db.tx().await;
    assert!(
        evidence_cluster(&tx, "space-id-a", true)
            .await
            .unwrap()
            .is_empty(),
        "a peripheral member's evidence must not count toward signal 1's floor"
    );

    tx.execute(
        "INSERT INTO community_members (space, node_id, community_id, attachment)
         VALUES ('space-a', 'entity-core-2', 'comm-1', 'core')",
        (),
    )
    .await
    .expect("promote a third core member");
    tx.execute(
        "INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status)
         VALUES ('root-3', 'document_ingest', 'group-c', 'active')",
        (),
    )
    .await
    .expect("seed root-3");
    tx.execute(
        "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id, grounded, root_id, space, valid_until)
         VALUES ('root-3', 'entity', 'entity-core-2', 'entity', 'peer', 1, 'root-3', 'space-a', NULL)",
        (),
    )
    .await
    .expect("seed edge for root-3");
    assert_eq!(
        evidence_cluster(&tx, "space-id-a", true)
            .await
            .unwrap()
            .len(),
        1,
        "a third group's evidence via a *core* member must admit — proves \
         the rejection above was for real"
    );
}

// ---------------------------------------------------------------------------
// Signal 3 — community-overview
// ---------------------------------------------------------------------------

#[tokio::test]
async fn community_overview_admits_when_durable_and_unblocked() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    seed_three_admitting_groups_in_community(&db, "space-a", "comm-1").await;

    let tx = db.tx().await;
    let proposals = community_overview(&tx, "space-id-a", true)
        .await
        .expect("query succeeds");
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].group_count, 3);
    assert_eq!(proposals[0].signal_kind, SignalKind::CommunityOverview);
    assert_eq!(
        proposals[0].slot_id,
        identity::slot_id_community_overview("space-id-a", "comm-1")
    );
    assert_eq!(proposals[0].page_id, page_id(&proposals[0].slot_id));
}

#[tokio::test]
async fn community_overview_rejects_when_partition_not_durable() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    seed_three_admitting_groups_in_community(&db, "space-a", "comm-1").await;

    let tx = db.tx().await;
    assert!(
        community_overview(&tx, "space-id-a", false)
            .await
            .unwrap()
            .is_empty(),
        "an undurable partition must admit nothing, even with otherwise-qualifying evidence"
    );
}

#[tokio::test]
async fn community_overview_rejects_with_active_subscription() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    seed_three_admitting_groups_in_community(&db, "space-a", "comm-1").await;
    db.seed_subscription("community", "comm-1", "active").await;

    let tx = db.tx().await;
    assert!(
        community_overview(&tx, "space-id-a", true)
            .await
            .unwrap()
            .is_empty(),
        "an active community-overview subscription must block a second candidate"
    );
}

#[tokio::test]
async fn community_overview_admits_with_only_a_detached_subscription() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    seed_three_admitting_groups_in_community(&db, "space-a", "comm-1").await;
    db.seed_subscription("community", "comm-1", "detached")
        .await;

    let tx = db.tx().await;
    assert!(
        !community_overview(&tx, "space-id-a", true)
            .await
            .unwrap()
            .is_empty(),
        "a detached subscription must not block re-subscription"
    );
}

#[tokio::test]
async fn community_overview_rejects_below_grounded_node_floor() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    db.seed_community_member("space-a", "entity-0", "comm-1", "core")
        .await;
    db.seed_community_member("space-a", "entity-1", "comm-1", "core")
        .await;
    db.seed_entity_evidence("root-0", "space-a", "group-a", "entity-0")
        .await;
    db.seed_entity_evidence("root-1", "space-a", "group-b", "entity-0")
        .await;
    db.seed_entity_evidence("root-2", "space-a", "group-c", "entity-1")
        .await;

    let tx = db.tx().await;
    assert!(
        community_overview(&tx, "space-id-a", true)
            .await
            .unwrap()
            .is_empty(),
        "three groups over two nodes must not admit signal 3's extra floor"
    );
}

/// Discriminator against signal 1's attachment filter leaking into signal 3:
/// a group whose only member is `peripheral` must still count here.
#[tokio::test]
async fn community_overview_includes_peripheral_members() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_community("comm-1", "space-a", None).await;
    for (i, group) in ["group-a", "group-b", "group-c"].into_iter().enumerate() {
        let attachment = if i == 2 { "peripheral" } else { "core" };
        for j in 0..2 {
            let entity = format!("entity-{i}-{j}");
            db.seed_community_member("space-a", &entity, "comm-1", attachment)
                .await;
            db.seed_entity_evidence(&format!("root-{i}-{j}"), "space-a", group, &entity)
                .await;
        }
    }

    let tx = db.tx().await;
    let proposals = community_overview(&tx, "space-id-a", true)
        .await
        .expect("query succeeds");
    assert_eq!(
        proposals.len(),
        1,
        "unlike signal 1, a peripheral member's evidence must count toward signal 3's floor"
    );
    assert_eq!(proposals[0].group_count, 3);
}

// ---------------------------------------------------------------------------
// Signal 2 — orphan-wikilink (G-A1..G-A7, docs/plans/2026-08-03-m6-pr-a-followup-2-scope.md §5)
// ---------------------------------------------------------------------------

/// Build page content with one wikilink to `label`, returning the content
/// and the target capture's exact byte range — the same shape
/// `claim_anchors.span_start/span_end` carries, so a fixture can anchor a
/// revision directly over (or around) the occurrence.
fn orphan_body(prefix: &str, label: &str, suffix: &str) -> (String, std::ops::Range<usize>) {
    let content = format!("{prefix}[[{label}]]{suffix}");
    let target_start = prefix.len() + 2; // past "[["
    let target_end = target_start + label.len();
    (content, target_start..target_end)
}

/// Baseline: two referring pages for `label` in `space`, three
/// independence groups total (one via page-a's revision, two via page-b's)
/// — clears both the >= 2-referring-page precondition and the D1 floor.
async fn seed_admitting_orphan_baseline(db: &TestDb, space: &str, label: &str) {
    let (content_a, span_a) = orphan_body("Intro sentence has ", label, " in it.");
    db.seed_page("page-a", space, &content_a, 1).await;
    db.seed_orphan_link("page-a", label).await;
    db.seed_claim_anchor("page-a", 1, "rev-a1", &content_a, span_a, 0)
        .await;
    db.seed_entity_evidence("root-a1", space, "group-a", "entity-a1")
        .await;
    db.seed_supports_edge("supp-a1", "rev-a1", "root-a1", space)
        .await;

    let (content_b, span_b) = orphan_body("Another mention of ", label, " over here.");
    db.seed_page("page-b", space, &content_b, 1).await;
    db.seed_orphan_link("page-b", label).await;
    db.seed_claim_anchor("page-b", 1, "rev-b1", &content_b, span_b, 0)
        .await;
    db.seed_entity_evidence("root-b1", space, "group-b", "entity-b1")
        .await;
    db.seed_supports_edge("supp-b1", "rev-b1", "root-b1", space)
        .await;
    db.seed_entity_evidence("root-b2", space, "group-c", "entity-b2")
        .await;
    db.seed_supports_edge("supp-b2", "rev-b1", "root-b2", space)
        .await;
}

/// G-A1 positive control: a page with two revisions, the link in revision 1
/// only, revision 2 carrying two additional independence groups — the count
/// must see only revision 1's group. Widening the scope to every revision
/// of the referring page would read 5, not 3.
#[tokio::test]
async fn m6_orphan_link_scoped_to_containing_revision() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;

    let (content_a, span_a1) = orphan_body(
        "Linked sentence has ",
        "Rust",
        " in it. Unrelated tail sentence follows.",
    );
    let tail_start = content_a.find("Unrelated").expect("tail marker present");
    let span_a2 = tail_start..content_a.len();
    db.seed_page("page-a", "space-a", &content_a, 1).await;
    db.seed_orphan_link("page-a", "Rust").await;
    db.seed_claim_anchor("page-a", 1, "rev-a1", &content_a, span_a1, 0)
        .await;
    db.seed_claim_anchor("page-a", 1, "rev-a2", &content_a, span_a2, 1)
        .await;
    db.seed_entity_evidence("root-a1", "space-a", "group-a", "entity-a1")
        .await;
    db.seed_supports_edge("supp-a1", "rev-a1", "root-a1", "space-a")
        .await;
    db.seed_entity_evidence("root-a2", "space-a", "group-d", "entity-a2")
        .await;
    db.seed_supports_edge("supp-a2", "rev-a2", "root-a2", "space-a")
        .await;
    db.seed_entity_evidence("root-a3", "space-a", "group-e", "entity-a3")
        .await;
    db.seed_supports_edge("supp-a3", "rev-a2", "root-a3", "space-a")
        .await;

    let (content_b, span_b) = orphan_body("Second mention of ", "Rust", " right here.");
    db.seed_page("page-b", "space-a", &content_b, 1).await;
    db.seed_orphan_link("page-b", "Rust").await;
    db.seed_claim_anchor("page-b", 1, "rev-b1", &content_b, span_b, 0)
        .await;
    db.seed_entity_evidence("root-b1", "space-a", "group-b", "entity-b1")
        .await;
    db.seed_supports_edge("supp-b1", "rev-b1", "root-b1", "space-a")
        .await;
    db.seed_entity_evidence("root-b2", "space-a", "group-c", "entity-b2")
        .await;
    db.seed_supports_edge("supp-b2", "rev-b1", "root-b2", "space-a")
        .await;

    let tx = db.tx().await;
    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert_eq!(proposals.len(), 1);
    assert_eq!(
        proposals[0].group_count, 3,
        "rev-a2's groups (d, e) must not count: the link is only in rev-a1"
    );
}

/// G-A2: the R8 scope (bridging through `supports.root_id`, not matching
/// `supports` edges as `e` directly) admits a genuinely 3-group candidate.
/// Positive control: removing one group's only supports edge drops the
/// count to 2 and the candidate must reject, proving the count moves rather
/// than being vacuously satisfied.
#[tokio::test]
async fn m6_orphan_link_scope_is_not_structurally_empty() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_admitting_orphan_baseline(&db, "space-a", "Rust").await;

    let tx = db.tx().await;
    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert_eq!(
        proposals.len(),
        1,
        "the naive `supports`-as-`e` scope would \
        always read zero, since supports edges are written grounded = 0 in \
        production; this admits because the scope bridges through root_id"
    );
    assert_eq!(proposals[0].group_count, 3);

    tx.execute("DELETE FROM edges WHERE edge_id = 'supp-b2'", ())
        .await
        .expect("drop one supports edge");
    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert!(
        proposals.is_empty(),
        "two groups must not admit — proves the 3-group admission above was real"
    );
}

/// G-A3: a revision whose stored `span_digest` no longer matches the live
/// body contributes zero groups.
#[tokio::test]
async fn m6_orphan_link_binding_verifies_anchor_digest() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_admitting_orphan_baseline(&db, "space-a", "Rust").await;

    // Change the bytes at the anchored span itself — same length, still a
    // valid (differently-cased) wikilink so the occurrence is still found
    // and still normalizes to "rust" — so this exercises the digest check
    // specifically, not the earlier occurrences-empty short-circuit that a
    // wholesale body replacement would trip instead.
    let tx = db.tx().await;
    tx.execute(
        "UPDATE pages SET content = 'Another mention of [[RUST]] over here.' WHERE id = 'page-b'",
        (),
    )
    .await
    .expect("corrupt page-b body at the anchored span after the anchor was minted");

    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert!(
        proposals.is_empty(),
        "page-b's revision must be dropped (stale digest), leaving only \
         page-a's one group — below the floor"
    );
}

/// G-A4: a fullwidth-form target and a plain-ASCII target must collapse
/// onto the same D8-normalized candidate, and the candidate is keyed on the
/// normalized form, not the raw text.
#[tokio::test]
async fn m6_orphan_link_binding_uses_d8_key() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;

    let (content_a, span_a) = orphan_body("See ", "\u{FF32}ust", " over here."); // fullwidth 'R'
    db.seed_page("page-a", "space-a", &content_a, 1).await;
    db.seed_orphan_link("page-a", "\u{FF32}ust").await;
    db.seed_claim_anchor("page-a", 1, "rev-a1", &content_a, span_a, 0)
        .await;
    db.seed_entity_evidence("root-a1", "space-a", "group-a", "entity-a1")
        .await;
    db.seed_supports_edge("supp-a1", "rev-a1", "root-a1", "space-a")
        .await;

    let (content_b, span_b) = orphan_body("Also ", "rust", " mentioned.");
    db.seed_page("page-b", "space-a", &content_b, 1).await;
    db.seed_orphan_link("page-b", "rust").await;
    db.seed_claim_anchor("page-b", 1, "rev-b1", &content_b, span_b, 0)
        .await;
    db.seed_entity_evidence("root-b1", "space-a", "group-b", "entity-b1")
        .await;
    db.seed_supports_edge("supp-b1", "rev-b1", "root-b1", "space-a")
        .await;
    db.seed_entity_evidence("root-b2", "space-a", "group-c", "entity-b2")
        .await;
    db.seed_supports_edge("supp-b2", "rev-b1", "root-b2", "space-a")
        .await;

    let tx = db.tx().await;
    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert_eq!(
        proposals.len(),
        1,
        "the fullwidth and plain forms must collapse into one candidate, not two"
    );
    assert_eq!(proposals[0].group_count, 3, "both pages' groups combine");

    let normalized_slot = identity::slot_id_orphan_wikilink("space-id-a", "rust");
    let raw_slot = identity::slot_id_orphan_wikilink("space-id-a", "\u{FF32}ust");
    assert_eq!(proposals[0].slot_id, normalized_slot);
    assert_ne!(
        proposals[0].slot_id, raw_slot,
        "binding on the raw target text would produce a different slot_id"
    );
}

/// G-A5: alias (`[[Label|display]]`) and fragment (`[[Label#heading]]`)
/// forms both bind to the bare target. Binding on capture group 0 (the
/// full `[[...]]` match) instead of the target group would make
/// `normalize_label_key` reject every occurrence (R1, structural chars),
/// so this admission is itself the discriminator for that mutation.
#[tokio::test]
async fn m6_orphan_link_binding_reads_alias_and_fragment_forms() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;

    let content_a = "See [[Rust|display text]] over here.".to_string();
    let start_a = content_a.find("Rust").expect("target present");
    db.seed_page("page-a", "space-a", &content_a, 1).await;
    db.seed_orphan_link("page-a", "Rust").await;
    db.seed_claim_anchor(
        "page-a",
        1,
        "rev-a1",
        &content_a,
        start_a..start_a + "Rust".len(),
        0,
    )
    .await;
    db.seed_entity_evidence("root-a1", "space-a", "group-a", "entity-a1")
        .await;
    db.seed_supports_edge("supp-a1", "rev-a1", "root-a1", "space-a")
        .await;

    let content_b = "Also [[Rust#heading]] mentioned.".to_string();
    let start_b = content_b.find("Rust").expect("target present");
    db.seed_page("page-b", "space-a", &content_b, 1).await;
    db.seed_orphan_link("page-b", "Rust").await;
    db.seed_claim_anchor(
        "page-b",
        1,
        "rev-b1",
        &content_b,
        start_b..start_b + "Rust".len(),
        0,
    )
    .await;
    db.seed_entity_evidence("root-b1", "space-a", "group-b", "entity-b1")
        .await;
    db.seed_supports_edge("supp-b1", "rev-b1", "root-b1", "space-a")
        .await;
    db.seed_entity_evidence("root-b2", "space-a", "group-c", "entity-b2")
        .await;
    db.seed_supports_edge("supp-b2", "rev-b1", "root-b2", "space-a")
        .await;

    let tx = db.tx().await;
    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].group_count, 3);
}

/// G-A6: a revision present at an earlier page version but absent from
/// `page_version_claims` at the page's current version must not
/// contribute, even though its span still verifies against the live body.
#[tokio::test]
async fn m6_orphan_link_ignores_superseded_revision() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;

    let (content_a, span_a) = orphan_body("See ", "Rust", " here.");
    db.seed_page("page-a", "space-a", &content_a, 2).await;
    db.seed_orphan_link("page-a", "Rust").await;
    db.seed_claim_anchor("page-a", 2, "rev-a-current", &content_a, span_a.clone(), 0)
        .await;
    db.seed_claim_anchor("page-a", 1, "rev-a-superseded", &content_a, span_a, 0)
        .await;
    db.seed_entity_evidence("root-a-current", "space-a", "group-a", "entity-a1")
        .await;
    db.seed_supports_edge(
        "supp-a-current",
        "rev-a-current",
        "root-a-current",
        "space-a",
    )
    .await;
    db.seed_entity_evidence("root-a-old", "space-a", "group-d", "entity-a2")
        .await;
    db.seed_supports_edge("supp-a-old", "rev-a-superseded", "root-a-old", "space-a")
        .await;

    let (content_b, span_b) = orphan_body("Also ", "Rust", " mentioned.");
    db.seed_page("page-b", "space-a", &content_b, 1).await;
    db.seed_orphan_link("page-b", "Rust").await;
    db.seed_claim_anchor("page-b", 1, "rev-b1", &content_b, span_b, 0)
        .await;
    db.seed_entity_evidence("root-b1", "space-a", "group-b", "entity-b1")
        .await;
    db.seed_supports_edge("supp-b1", "rev-b1", "root-b1", "space-a")
        .await;
    db.seed_entity_evidence("root-b2", "space-a", "group-c", "entity-b2")
        .await;
    db.seed_supports_edge("supp-b2", "rev-b1", "root-b2", "space-a")
        .await;

    let tx = db.tx().await;
    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert_eq!(proposals.len(), 1);
    assert_eq!(
        proposals[0].group_count, 3,
        "the superseded revision's group (d) must not count — dropping \
         the page_version = pages.version join condition would read 4"
    );
}

/// G-A7: a page whose `page_truth_state.support_status = 'supported'`
/// names an older `page_version` than the page's current version is not an
/// eligible referring page. Positive control: catching the stamp up to the
/// current version restores eligibility and admission.
#[tokio::test]
async fn m6_orphan_link_supported_check_is_version_bound() {
    let db = TestDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_admitting_orphan_baseline(&db, "space-a", "Rust").await;

    let tx = db.tx().await;
    tx.execute(
        "UPDATE page_truth_state SET page_version = 0 WHERE page_id = 'page-b'",
        (),
    )
    .await
    .expect("stale-stamp page-b's support");
    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert!(
        proposals.is_empty(),
        "page-b's stale support stamp must drop it from the referring-page \
         count, leaving only page-a — below the 2-page floor"
    );

    tx.execute(
        "UPDATE page_truth_state SET page_version = 1 WHERE page_id = 'page-b'",
        (),
    )
    .await
    .expect("catch up the support stamp");
    let proposals = orphan_wikilink(&tx, "space-id-a")
        .await
        .expect("query succeeds");
    assert_eq!(
        proposals.len(),
        1,
        "a current-version support stamp restores eligibility"
    );
}
