//! Teeth for S0-162's rename closure — mutation-catalog gates `G8.11a`-`c`.
//!
//! One scenario asserted several ways: rename a space that carries community
//! and genesis substrate, then check that identity is unchanged, that the new
//! name resolves, and that no row still names the old space.

use super::space_rename::closed_tables;
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

async fn seed_substrate(conn: &libsql::Connection, space: &str) {
    let stmts: Vec<(&str, Vec<libsql::Value>)> = vec![
        (
            "INSERT INTO entities (id, name, entity_type, space, created_at, updated_at)
             VALUES ('ent-1', 'Anchor', 'concept', ?1, 1, 1)",
            vec![space.into()],
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
            "INSERT OR REPLACE INTO genesis_coverage_state (space, opened_at) VALUES (?1, 1)",
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
    ];
    for (sql, params) in stmts {
        conn.execute(sql, params)
            .await
            .unwrap_or_else(|e| panic!("seed failed: {sql}\n{e}"));
    }
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
