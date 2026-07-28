// SPDX-License-Identifier: Apache-2.0
//! Pre-conditions for the M5 `edges` rebuild (artifact 3 §7).
//!
//! SQLite cannot alter a CHECK constraint, so widening `edges` for the
//! `claim_revision`/`root` endpoint kinds and the `attests` edge type requires
//! dropping and recreating the table. Everything attached to it — every index
//! and every trigger — goes with it.
//!
//! These tests record what is actually attached, read from `sqlite_master` on a
//! freshly migrated database rather than from a hand-written list. The rebuild
//! must restore exactly this set.

use super::tests::test_db;

async fn schema_objects(kind: &str) -> Vec<String> {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    // `sqlite_autoindex_edges_1` is excluded: it is SQLite's implicit index for
    // `edge_id TEXT PRIMARY KEY`, created by the PK declaration itself, so the
    // rebuild gets it back from `CREATE TABLE` rather than restoring it.
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master
             WHERE type = ?1 AND tbl_name = 'edges'
               AND name NOT LIKE 'sqlite_autoindex%'
             ORDER BY name",
            libsql::params![kind],
        )
        .await
        .unwrap();
    let mut names = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        names.push(row.get::<String>(0).unwrap());
    }
    names
}

/// Weakening: recreate only the triggers a spec paragraph happens to name.
///
/// `docs/plans/2026-07-27-m5-edge-rebuild-matrix.md` §7 step 6 says "recreate
/// triggers (space fence, both twins)" — two names. The live schema carries
/// eight. A rebuild that trusts the prose silently drops the six M4 triggers
/// that keep community grouping and page-community route inputs invalidated,
/// and nothing downstream would fail loudly: the tables stay correct-looking
/// while their invalidation stops firing.
///
/// The rebuild must diff against `sqlite_master`, not against a list.
#[tokio::test]
async fn every_trigger_on_edges_is_recorded_for_the_rebuild() {
    assert_eq!(
        schema_objects("trigger").await,
        [
            "edges_space_fence",
            "edges_space_fence_update",
            "m4_grouping_edge_delete",
            "m4_grouping_edge_insert",
            "m4_grouping_edge_update",
            "m4_page_community_edge_delete_invalidate",
            "m4_page_community_edge_insert_invalidate",
            "m4_page_community_edge_update_invalidate",
        ],
        "the edges rebuild must restore every one of these; update this list \
         and the rebuild together, never one alone"
    );
}

/// Same weakening, index half. Dropping `edges` drops all of these too.
#[tokio::test]
async fn every_index_on_edges_is_recorded_for_the_rebuild() {
    assert_eq!(
        schema_objects("index").await,
        [
            "idx_edges_active_grounded_space_type",
            "idx_edges_dst",
            "idx_edges_operation",
            "idx_edges_root",
            "idx_edges_src",
            "idx_edges_superseded",
        ],
        "the edges rebuild must restore every one of these, plus the four \
         partial support/attest indexes it adds (artifact 3 §6)"
    );
}
