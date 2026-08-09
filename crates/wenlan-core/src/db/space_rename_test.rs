//! Teeth for S0-162's rename closure — mutation-catalog gates `G8.11a`-`c`.
//!
//! One scenario asserted several ways: rename a space that carries community
//! and genesis substrate, then check that identity is unchanged, that the new
//! name resolves, and that no row still names the old space.

use super::space_rename::{closed_tables, permanent_tables};
use crate::db::tests::test_db;
use crate::db::MemoryDB;

const OLD: &str = "Research Notes";
const NEW: &str = "Field Notes";

/// Tables that carry a `space` column and are closed over by the cascade that
/// already existed before S0-162 (`update_space`, `db.rs:19327`-`:19360`).
const PRE_EXISTING_CASCADE: &[&str] = &["memories", "entities", "pages"];

/// Tables that carry a `space` column and are deliberately left alone.
///
/// `page_draft_create_requests` is S0-162's one recorded exclusion: its stored
/// name is replay history, not stale data (`db.rs:7771`).
const DELIBERATE_EXCLUSIONS: &[&str] = &["page_draft_create_requests"];

#[test]
fn permanent_destination_fence_covers_retained_migration_109_history() {
    for table in [
        "genesis_suppression",
        "genesis_card_binding",
        "genesis_quarantine",
        "m6_overview_subscriptions",
        "m6_refresh_dependencies",
        "m6_readiness",
        "m6_readiness_soak_receipts",
        "m6_counters",
    ] {
        assert!(
            permanent_tables().contains(&table),
            "retained M6 history in {table} would be deleted on a destination-name collision"
        );
    }
}

async fn seed_substrate(conn: &libsql::Connection, space: &str) {
    let stmts: Vec<(&str, Vec<libsql::Value>)> = vec![
        // G6 Stage 3: the `entities` row 'ent-1' that used to head this list
        // is gone with the table (migration 123). Its space column was one of
        // the rename targets; the shadow page below now carries the entity's
        // space, and it is already in the list.
        //
        // G6 Stage 2 item 1 (db.rs:9219): the ported `edges_space_fence`/
        // `edges_space_fence_update` triggers resolve an entity endpoint's
        // space via its shadow page, not `entities.space` directly, and
        // fail closed (abort) when no shadow page exists. The non-legacy
        // edge seeded below needs one for 'ent-1' before it can insert.
        // Same shape as `insert_entity_shadow_page`/`m4_community_gates.rs`'s
        // `seed_entity_shadow_pages`.
        (
            "INSERT INTO pages (
                 id, title, summary, content, kind, entity_type, confidence, entity_confirmed,
                 embedding, space, workspace, source_memory_ids, version, status,
                 created_at, last_compiled, last_modified, creation_kind, review_status
             )
             VALUES ('shadow-ent-1', 'Anchor', NULL, '', 'entity', 'concept', NULL, 0,
                     NULL, ?1, ?1, '[]', 1, 'active',
                     '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z',
                     'entity', 'unconfirmed')",
            vec![space.into()],
        ),
        (
            "INSERT INTO entity_page_map (entity_id, page_id, created_at)
             VALUES ('ent-1', 'shadow-ent-1', '2024-01-01T00:00:00Z')",
            vec![],
        ),
        // `m4_page_community_page_insert_input` (db.rs:11483) fires on every
        // `pages` insert and mints a `page_community_route_inputs` row keyed
        // by page_id, so the shadow page above added a second one this
        // closure never expected -- 'page-1' is the fixture's one
        // community-routable page. Undo the trigger's side effect for the
        // shadow page only; it carries no separate routing meaning here.
        (
            "DELETE FROM page_community_route_inputs WHERE page_id = 'shadow-ent-1'",
            vec![],
        ),
        (
            "INSERT INTO pages (id, title, content, space, created_at, last_compiled, last_modified)
             VALUES ('page-1', 'Anchor', 'body', ?1, '2024-01-01T00:00:00Z',
                     '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z')",
            vec![space.into()],
        ),
        (
            "INSERT INTO communities
                 (community_id, space, algo_version, projection_version, created_at, updated_at)
             VALUES ('com-1', ?1, 'leiden-v1', 'proj-v1', 1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO community_members
                 (space, node_id, node_kind, community_id, published_generation, attachment)
             VALUES (?1, 'ent-1', 'entity', 'com-1', 1, 'core')",
            vec![space.into()],
        ),
        (
            "INSERT INTO page_community_assignments
                 (page_id, space, community_id, state, score, page_version,
                  routing_input_generation, routing_space_generation,
                  community_published_generation, route_version, updated_at)
             VALUES ('page-1', ?1, 'com-1', 'assigned', 0.9, 1, 0, 0, 1, 'route-v1', 1)",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO page_community_route_inputs (page_id, space, generation)
             VALUES ('page-1', ?1, 0)",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO community_route_space_inputs (space, generation) VALUES (?1, 0)",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO space_graph_state (space, dirty) VALUES (?1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO grouping_leases (phase, space, input_generation, token, expires_at)
             VALUES ('grouping', ?1, 1, 'tok-1', 9999999999)",
            vec![space.into()],
        ),
        // No `community_reader_parity` row: the table is dropped by the
        // "retire local-only community parity model" migration (`db.rs:11659`),
        // which is why it is not in the closure either.
        (
            "INSERT OR REPLACE INTO community_reader_space_proof
                 (consumer, space, proven_published_generation, proven_membership_digest)
             VALUES ('scoped_pages', ?1, 1, 'digest-1')",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO community_publication_receipts
                 (space, published_generation, membership_digest, algo_version,
                  projection_version, published_at)
             VALUES (?1, 1, 'digest-1', 'leiden-v1', 'proj-v1', 1)",
            vec![space.into()],
        ),
        // A NON-legacy edge, so `edges_space_fence_update` (db.rs:8932) is live
        // on the rewrite. Its endpoint is the seeded entity, which the
        // pre-existing cascade renames first — this row is what proves the
        // ordering rather than merely assuming it.
        (
            "INSERT INTO edges
                 (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
                  grounded, space, created_at)
             VALUES ('edge-1', 'ent-1', 'entity', 'ent-1', 'entity', 'relates',
                     'assertion', 0, ?1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO genesis_candidates
                 (candidate_id, slot_id, page_id, space, signal_kind, coverage_epoch,
                  input_generation, active_root_digest, state, created_at, updated_at)
             VALUES ('cand-1', 'slot-1', 'page-1', ?1, 'evidence-cluster', 1, 1,
                     'rootdigest', 'observed', 1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO genesis_coverage_state (space, opened_at, space_id)
             SELECT name, 1, id FROM spaces WHERE name = ?1",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO genesis_group_coverage
                 (space, independence_group_id, coverage_epoch, page_id, candidate_id, covered_at)
             VALUES (?1, 'grp-1', 1, 'page-1', 'cand-1', 1)",
            vec![space.into()],
        ),
        (
            "INSERT OR REPLACE INTO genesis_frontier
                 (space, independence_group_id, coverage_epoch, first_seen_at, next_scan_at)
             VALUES (?1, 'grp-1', 1, 1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO m6_pair_stats
                 (space, page_a, page_b, n11, n10, n01, n00,
                  distinct_group_count, updated_generation)
             VALUES (?1, 'page-a', 'page-b', 1.0, 0.0, 0.0, 1.0, 1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO m6_adjacency
                 (space, endpoint_kind, endpoint_id, neighbor_id, rank)
             VALUES (?1, 'page', 'page-a', 'neighbor-a', 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO genesis_refresh_jobs
                 (job_id, page_id, base_page_version, base_source_revision, space,
                  dependency_generation, active_root_digest, state, attempt,
                  readiness_epoch, schema_version, reason, created_at, updated_at)
             VALUES (1, 'page-refresh', 1, 1, ?1, 1, 'roots', 'finalized', 0,
                     0, 109, 'source_updated', 1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO m6_refresh_dependencies
                 (space, page_id, page_version, claim_revision_id, root_id, created_at)
             VALUES (?1, 'page-refresh', 1, 'claim-1', 'root-1', 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO m6_readiness
                 (space, stage, signal, epoch, phase, operation_state, updated_at)
             VALUES (?1, 'A', '-', 0, 'off', 'active', 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO m6_readiness_soak_receipts
                 (space, stage, signal, readiness_epoch, window_started_at,
                  window_ended_at, observed_turns, daemon_starts,
                  old_writer_mutations, work_bound_violations,
                  support_regressions, recorded_at)
             VALUES (?1, 'A', '-', 0, 0, 259200, 20, 1, 0, 0, 0, 259200)",
            vec![space.into()],
        ),
        (
            "INSERT INTO genesis_suppression
                 (space, independence_group_id, coverage_epoch, page_id, reason,
                  first_seen_at, suppressed_at, expires_at)
             VALUES (?1, 'grp-suppressed', 1, 'page-suppressed', 'human choice',
                     1, 1, 2)",
            vec![space.into()],
        ),
        (
            "INSERT INTO genesis_card_binding
                 (space, independence_group_id, coverage_epoch, card_id, page_id,
                  first_seen_at, created_at)
             VALUES (?1, 'grp-card', 1, 'card-1', 'page-card', 1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO genesis_quarantine
                 (space, independence_group_id, coverage_epoch, reason,
                  first_seen_at, quarantined_at)
             VALUES (?1, 'grp-quarantine', 1, 'needs review', 1, 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO m6_overview_subscriptions
                 (subscription_id, scope_kind, scope_id, page_id, space, state,
                  created_at)
             VALUES ('subscription-1', 'community', 'com-1', 'page-overview',
                     ?1, 'active', 1)",
            vec![space.into()],
        ),
        (
            "INSERT INTO m6_counters(space_id, space, name, value)
             SELECT id, name, 'relevance_generation', 1 FROM spaces WHERE name = ?1",
            vec![space.into()],
        ),
    ];
    for (sql, params) in stmts {
        conn.execute(sql, params)
            .await
            .unwrap_or_else(|e| panic!("seed failed: {sql}\n{e}"));
    }
}

async fn seed_destination_genesis(conn: &libsql::Connection) {
    let stmts = [
        "INSERT INTO genesis_candidates
             (candidate_id, slot_id, page_id, space, signal_kind, coverage_epoch,
              input_generation, active_root_digest, state, created_at, updated_at)
         VALUES ('cand-destination', 'slot-destination', 'page-destination', ?1,
                 'evidence-cluster', 7, 11, 'destination-root', 'published', 2, 3)",
        "INSERT INTO genesis_coverage_state
             (space, coverage_epoch, epoch_state, opened_at, genesis_enabled, space_id)
         SELECT name, 7, 'active', 2, 1, id FROM spaces WHERE name = ?1",
        "INSERT INTO genesis_group_coverage
             (space, independence_group_id, coverage_epoch, page_id, candidate_id, covered_at)
         VALUES (?1, 'grp-destination', 7, 'page-destination', 'cand-destination', 4)",
        "INSERT INTO genesis_frontier
             (space, independence_group_id, coverage_epoch, first_seen_at, next_scan_at)
         VALUES (?1, 'grp-destination', 7, 5, 6)",
    ];
    for sql in stmts {
        conn.execute(sql, libsql::params![NEW])
            .await
            .unwrap_or_else(|e| panic!("seed destination genesis failed: {sql}\n{e}"));
    }
}

async fn seed_source_evidence(conn: &libsql::Connection) {
    conn.execute(
        "INSERT INTO genesis_candidate_roots
             (candidate_id, root_id, independence_group_id, coverage_epoch,
              claim_role, consumed, retained, created_at)
         VALUES ('cand-1', 'root-source', 'grp-source', 1, 'concept', 1, 1, 7)",
        (),
    )
    .await
    .expect("seed source genesis evidence");
}

async fn rows_naming(conn: &libsql::Connection, table: &str, space: &str) -> i64 {
    let mut rows = conn
        .query(
            &format!("SELECT COUNT(*) FROM {table} WHERE space = ?1"),
            libsql::params![space],
        )
        .await
        .unwrap_or_else(|e| panic!("count {table}: {e}"));
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

async fn seeded_db() -> (MemoryDB, tempfile::TempDir) {
    let (db, dir) = test_db().await;
    db.create_space(OLD, None, false).await.unwrap();
    {
        let conn = db.conn.lock().await;
        seed_substrate(&conn, OLD).await;
    }
    (db, dir)
}

#[tokio::test]
async fn every_space_keyed_row_follows_the_rename() {
    let (db, _dir) = seeded_db().await;
    db.update_space(OLD, NEW, None).await.unwrap();

    let conn = db.conn.lock().await;
    for table in closed_tables() {
        assert_eq!(
            rows_naming(&conn, table, NEW).await,
            1,
            "{table} should carry exactly one row under the new name"
        );
    }
}

#[tokio::test]
async fn no_substrate_row_still_names_the_old_space() {
    // G8.11c stated once over the whole closure: a row keyed to the old name
    // must not survive the rename and stay claimable.
    let (db, _dir) = seeded_db().await;
    db.update_space(OLD, NEW, None).await.unwrap();

    let conn = db.conn.lock().await;
    for table in closed_tables() {
        assert_eq!(
            rows_naming(&conn, table, OLD).await,
            0,
            "{table} still names the old space after the rename"
        );
    }
}

#[tokio::test]
async fn the_page_draft_ledger_keeps_the_old_name() {
    // The one deliberate exclusion. Rewriting it would break delayed-create
    // replay, which is the failure the ledger exists to prevent.
    let (db, _dir) = seeded_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO page_draft_create_requests (page_id, title, content, space)
             VALUES ('draft-1', 'Draft', 'body', ?1)",
            libsql::params![OLD],
        )
        .await
        .unwrap();
    }

    db.update_space(OLD, NEW, None).await.unwrap();

    let conn = db.conn.lock().await;
    assert_eq!(
        rows_naming(&conn, "page_draft_create_requests", OLD).await,
        1,
        "the create-request ledger must keep the name the request was made under"
    );
}

#[tokio::test]
async fn an_orphan_row_under_the_new_name_is_retired() {
    // `spaces.name` is UNIQUE, so a row already keyed to the destination name
    // names no live space. It must not block the rename.
    let (db, _dir) = seeded_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT OR REPLACE INTO space_graph_state (space, graph_generation, dirty) VALUES (?1, 77, 0)",
            libsql::params![NEW],
        )
        .await
        .unwrap();
    }

    db.update_space(OLD, NEW, None).await.unwrap();

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*), MAX(graph_generation), MAX(dirty) FROM space_graph_state
             WHERE space = ?1",
            libsql::params![NEW],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    let count: i64 = row.get(0).unwrap();
    let generation: i64 = row.get(1).unwrap();
    let dirty: i64 = row.get(2).unwrap();
    assert_eq!(count, 1, "the orphan must be retired, not kept alongside");
    assert_eq!(generation, 0, "the renamed row must be the survivor");
    assert_eq!(dirty, 1, "the renamed row must be the survivor");
}

#[tokio::test]
async fn an_orphan_refresh_job_does_not_become_live_under_the_renamed_space() {
    let (db, _dir) = seeded_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO genesis_refresh_jobs
                 (job_id, page_id, base_page_version, base_source_revision, space,
                  dependency_generation, active_root_digest, state, attempt,
                  readiness_epoch, schema_version, reason, created_at, updated_at)
             VALUES (2, 'orphan-page', 1, 1, ?1, 1, 'orphan-roots', 'finalized', 0,
                     0, 109, 'source_updated', 1, 1)",
            libsql::params![NEW],
        )
        .await
        .unwrap();
    }

    db.update_space(OLD, NEW, None).await.unwrap();

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*), MIN(job_id), MAX(job_id)
               FROM genesis_refresh_jobs WHERE space = ?1",
            libsql::params![NEW],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(1).unwrap(), 1);
    assert_eq!(row.get::<i64>(2).unwrap(), 1);
}

#[tokio::test]
async fn delete_keep_cannot_authorize_proof_parking_and_zero_reset() {
    let (db, _dir) = seeded_db().await;
    let old_space_id = db.get_space(OLD).await.unwrap().unwrap().id;
    db.create_space(NEW, None, false).await.unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE genesis_coverage_state SET m6_mutation_count = 5 WHERE space = ?1",
            libsql::params![OLD],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE m6_counters SET value = 5
              WHERE space = ?1 AND name = 'relevance_generation'",
            libsql::params![OLD],
        )
        .await
        .unwrap();
    }

    db.delete_space(OLD, "keep").await.unwrap();

    let conn = db.conn.lock().await;
    let genesis_move = conn
        .execute(
            "UPDATE OR REPLACE genesis_coverage_state
                SET space = ?1, m6_mutation_count = 6
              WHERE space = ?2",
            libsql::params![NEW, OLD],
        )
        .await;
    let genesis_reset = conn
        .execute(
            "INSERT INTO genesis_coverage_state
                 (space, opened_at, m6_mutation_count, space_id)
             VALUES (?1, 100, 0, ?2)",
            libsql::params![OLD, old_space_id.clone()],
        )
        .await;
    let counter_move = conn
        .execute(
            "UPDATE OR REPLACE m6_counters
                SET space = ?1, value = 6
              WHERE space = ?2 AND name = 'relevance_generation'",
            libsql::params![NEW, OLD],
        )
        .await;
    let counter_reset = conn
        .execute(
            "INSERT INTO m6_counters(space_id, space, name, value)
             VALUES (?1, ?2, 'relevance_generation', 0)",
            libsql::params![old_space_id.clone(), OLD],
        )
        .await;

    assert!(
        genesis_move.is_err() && genesis_reset.is_err(),
        "delete-keep must not turn an unrelated live space into proof-move authorization"
    );
    assert!(
        counter_move.is_err() && counter_reset.is_err(),
        "delete-keep must not permit a named counter to park and reset"
    );

    let mut rows = conn
        .query(
            "SELECT space, space_id, m6_mutation_count FROM genesis_coverage_state",
            (),
        )
        .await
        .unwrap();
    let genesis = rows.next().await.unwrap().unwrap();
    assert_eq!(genesis.get::<String>(0).unwrap(), OLD);
    assert_eq!(genesis.get::<String>(1).unwrap(), old_space_id);
    assert_eq!(genesis.get::<i64>(2).unwrap(), 5);
    assert!(rows.next().await.unwrap().is_none());

    let mut rows = conn
        .query("SELECT space_id, space, name, value FROM m6_counters", ())
        .await
        .unwrap();
    let counter = rows.next().await.unwrap().unwrap();
    assert_eq!(counter.get::<String>(0).unwrap(), old_space_id);
    assert_eq!(counter.get::<String>(1).unwrap(), OLD);
    assert_eq!(counter.get::<String>(2).unwrap(), "relevance_generation");
    assert_eq!(counter.get::<i64>(3).unwrap(), 5);
    assert!(rows.next().await.unwrap().is_none());
}

#[tokio::test]
async fn recreated_space_name_cannot_adopt_retained_proof_identity() {
    let (db, _dir) = seeded_db().await;
    let old_space_id = db.get_space(OLD).await.unwrap().unwrap().id;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE genesis_coverage_state SET m6_mutation_count = 5 WHERE space = ?1",
            libsql::params![OLD],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE m6_counters SET value = 5
              WHERE space = ?1 AND name = 'relevance_generation'",
            libsql::params![OLD],
        )
        .await
        .unwrap();
    }
    db.delete_space(OLD, "keep").await.unwrap();
    let recreated = db.create_space(OLD, None, false).await.unwrap();
    assert_ne!(recreated.id, old_space_id);

    let conn = db.conn.lock().await;
    let genesis_replace = conn
        .execute(
            "INSERT OR REPLACE INTO genesis_coverage_state
                 (space, opened_at, m6_mutation_count, space_id)
             VALUES (?1, 100, 0, ?2)",
            libsql::params![OLD, recreated.id.clone()],
        )
        .await;
    let counter_replace = conn
        .execute(
            "INSERT OR REPLACE INTO m6_counters(space_id, space, name, value)
             VALUES (?1, ?2, 'relevance_generation', 0)",
            libsql::params![recreated.id, OLD],
        )
        .await;
    assert!(
        genesis_replace.is_err() && counter_replace.is_err(),
        "recreating a name with a new spaces.id cannot adopt or reset retained proofs"
    );

    let mut rows = conn
        .query(
            "SELECT space_id, m6_mutation_count FROM genesis_coverage_state",
            (),
        )
        .await
        .unwrap();
    let genesis = rows.next().await.unwrap().unwrap();
    assert_eq!(genesis.get::<String>(0).unwrap(), old_space_id);
    assert_eq!(genesis.get::<i64>(1).unwrap(), 5);
    let mut rows = conn
        .query("SELECT space_id, value FROM m6_counters", ())
        .await
        .unwrap();
    let counter = rows.next().await.unwrap().unwrap();
    assert_eq!(counter.get::<String>(0).unwrap(), old_space_id);
    assert_eq!(counter.get::<i64>(1).unwrap(), 5);
}

#[tokio::test]
async fn destination_genesis_rows_refuse_the_rename_without_mutating_either_side() {
    let (db, _dir) = seeded_db().await;
    db.create_space(NEW, None, false).await.unwrap();
    {
        let conn = db.conn.lock().await;
        seed_source_evidence(&conn).await;
        seed_destination_genesis(&conn).await;
    }
    db.delete_space(NEW, "keep").await.unwrap();

    let error = db
        .update_space(OLD, NEW, Some("renamed"))
        .await
        .expect_err("permanent destination genesis rows must refuse the rename");
    assert!(
        error.to_string().contains("destination genesis"),
        "rename was refused for the wrong reason: {error}"
    );

    assert!(db.get_space(OLD).await.unwrap().is_some());
    assert!(db.get_space(NEW).await.unwrap().is_none());

    let conn = db.conn.lock().await;
    for table in closed_tables() {
        assert_eq!(
            rows_naming(&conn, table, OLD).await,
            1,
            "source row in {table} was not rolled back"
        );
    }
    for table in [
        "genesis_candidates",
        "genesis_coverage_state",
        "genesis_group_coverage",
        "genesis_frontier",
    ] {
        assert_eq!(
            rows_naming(&conn, table, NEW).await,
            1,
            "destination row in {table} changed during the refused rename"
        );
    }

    let mut rows = conn
        .query(
            "SELECT candidate_id, coverage_epoch, input_generation, active_root_digest, state
             FROM genesis_candidates WHERE space = ?1",
            libsql::params![NEW],
        )
        .await
        .unwrap();
    let candidate = rows.next().await.unwrap().unwrap();
    assert_eq!(candidate.get::<String>(0).unwrap(), "cand-destination");
    assert_eq!(candidate.get::<i64>(1).unwrap(), 7);
    assert_eq!(candidate.get::<i64>(2).unwrap(), 11);
    assert_eq!(candidate.get::<String>(3).unwrap(), "destination-root");
    assert_eq!(candidate.get::<String>(4).unwrap(), "published");

    let mut rows = conn
        .query(
            "SELECT root_id, independence_group_id, consumed, retained
             FROM genesis_candidate_roots WHERE candidate_id = 'cand-1'",
            (),
        )
        .await
        .unwrap();
    let evidence = rows.next().await.unwrap().unwrap();
    assert_eq!(evidence.get::<String>(0).unwrap(), "root-source");
    assert_eq!(evidence.get::<String>(1).unwrap(), "grp-source");
    assert_eq!(evidence.get::<i64>(2).unwrap(), 1);
    assert_eq!(evidence.get::<i64>(3).unwrap(), 1);
}

#[tokio::test]
async fn a_rename_does_not_change_the_space_id() {
    // G8.11a. M6 identity digests `spaces.id`, so a rename that leaves the id
    // alone leaves `slot_id`, `page_id`, and candidate identity alone with it.
    let (db, _dir) = seeded_db().await;
    let before = db.get_space(OLD).await.unwrap().unwrap();
    db.update_space(OLD, NEW, None).await.unwrap();
    let after = db.get_space(NEW).await.unwrap().unwrap();

    assert_eq!(before.id, after.id, "a rename must not re-key the space");

    let slot_before = crate::m6::identity::slot_id_space_overview(&before.id);
    let slot_after = crate::m6::identity::slot_id_space_overview(&after.id);
    assert_eq!(slot_before, slot_after);
    assert_eq!(
        crate::m6::identity::page_id(&slot_before),
        crate::m6::identity::page_id(&slot_after)
    );
    assert_eq!(
        crate::m6::identity::candidate_id(&slot_before, 1),
        crate::m6::identity::candidate_id(&slot_after, 1)
    );
}

#[tokio::test]
async fn the_new_name_resolves_and_the_old_one_does_not() {
    // G8.11b's reachable half. S0-161 refuses a mint whose stored space does
    // not resolve; a routine rename must not be what makes it refuse.
    let (db, _dir) = seeded_db().await;
    db.update_space(OLD, NEW, None).await.unwrap();

    assert!(db.get_space(NEW).await.unwrap().is_some());
    assert!(db.get_space(OLD).await.unwrap().is_none());

    // Every substrate row now names a space that resolves.
    let conn = db.conn.lock().await;
    for table in closed_tables() {
        let mut rows = conn
            .query(
                &format!(
                    "SELECT COUNT(*) FROM {table} t
                     WHERE NOT EXISTS (SELECT 1 FROM spaces s WHERE s.name = t.space)"
                ),
                (),
            )
            .await
            .unwrap();
        let unresolvable: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(unresolvable, 0, "{table} holds a row naming no live space");
    }
}

#[tokio::test]
async fn renaming_a_space_with_no_substrate_is_a_no_op() {
    // The catalog's suite-level control.
    let (db, _dir) = test_db().await;
    db.create_space(OLD, None, false).await.unwrap();
    db.update_space(OLD, NEW, None).await.unwrap();

    let conn = db.conn.lock().await;
    for table in closed_tables() {
        assert_eq!(rows_naming(&conn, table, NEW).await, 0, "{table}");
        assert_eq!(rows_naming(&conn, table, OLD).await, 0, "{table}");
    }
}

#[tokio::test]
async fn every_space_keyed_table_is_accounted_for() {
    // The enumeration S0-162 calls closed is only closed if something checks
    // it. A new table carrying a `space` column must land in the closure, in
    // the pre-existing cascade, or in the recorded exclusions — never silently
    // outside all three.
    let (db, _dir) = test_db().await;
    let conn = db.conn.lock().await;

    let mut tables: Vec<String> = Vec::new();
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
             ORDER BY name",
            (),
        )
        .await
        .unwrap();
    while let Some(row) = rows.next().await.unwrap() {
        tables.push(row.get(0).unwrap());
    }

    let mut space_keyed: Vec<String> = Vec::new();
    for table in tables {
        let mut cols = conn
            .query(&format!("PRAGMA table_info({table})"), ())
            .await
            .unwrap();
        while let Some(col) = cols.next().await.unwrap() {
            let name: String = col.get(1).unwrap();
            if name == "space" {
                space_keyed.push(table.clone());
                break;
            }
        }
    }

    let accounted: Vec<&str> = closed_tables()
        .into_iter()
        .chain(PRE_EXISTING_CASCADE.iter().copied())
        .chain(DELIBERATE_EXCLUSIONS.iter().copied())
        .collect();

    let unaccounted: Vec<&String> = space_keyed
        .iter()
        .filter(|t| !accounted.contains(&t.as_str()))
        .collect();
    assert!(
        unaccounted.is_empty(),
        "these tables carry a `space` column and are in no S0-162 bucket: {unaccounted:?}"
    );

    for table in closed_tables() {
        assert!(
            space_keyed.iter().any(|t| t == table),
            "{table} is in the closure but carries no `space` column"
        );
    }
}
