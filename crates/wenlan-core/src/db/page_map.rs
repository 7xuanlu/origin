// SPDX-License-Identifier: Apache-2.0
//
// Page Map (mind-map) v1 — db-layer types + accessors over migration 75's
// `page_maps` / `page_map_nodes` / `page_map_edges` tables. See
// docs/superpowers/plans/2026-07-18-page-map-api-spec.md for the full
// contract (data model, identity/tombstone rule, state machine, graph
// invariants). This module implements only the data layer: routes and wire
// DTOs are a later stage and live outside `wenlan-core`.
//
// Declared `pub mod page_map` (unlike the private `mod count` /
// `scoped_pages` siblings) because stage 2 needs `PageMapNode` /
// `PageMapEdge` / the create outcomes as real cross-crate types, not just
// methods on `MemoryDB`.

use super::{commit_or_rollback, MemoryDB};
use crate::WenlanError;

/// A `page_maps` row.
#[derive(Debug, Clone, PartialEq)]
pub struct PageMap {
    pub page_id: String,
    pub revision: i64,
    pub map_schema: i64,
    pub viewport: Option<String>,
    pub generated_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A `page_map_nodes` row.
#[derive(Debug, Clone, PartialEq)]
pub struct PageMapNode {
    pub id: String,
    pub page_id: String,
    /// `None` only for the root.
    pub parent_id: Option<String>,
    pub rank: f64,
    pub ref_kind: String,
    pub ref_id: String,
    /// Map-local display override; `None` = render from the backing object.
    pub label: Option<String>,
    pub status: String,
    pub pinned: bool,
    pub placed: bool,
    pub collapsed: bool,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub width: Option<f64>,
    pub height: Option<f64>,
    pub fingerprint: String,
    pub provenance: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// A `page_map_edges` row.
#[derive(Debug, Clone, PartialEq)]
pub struct PageMapEdge {
    pub id: String,
    pub page_id: String,
    pub from_node: String,
    pub to_node: String,
    pub kind: String,
    pub label: Option<String>,
    pub status: String,
    pub provenance: Option<String>,
    pub created_at: String,
}

/// Full map read (`get_page_map` / `init_page_map` / any mutation that
/// returns the whole map).
#[derive(Debug, Clone, PartialEq)]
pub struct PageMapData {
    pub map: PageMap,
    pub nodes: Vec<PageMapNode>,
    pub edges: Vec<PageMapEdge>,
}

/// Fields a node PATCH may touch. `None` = leave unchanged. `label` is a
/// nested `Option` so a patch can distinguish "don't touch the label" from
/// "clear the override back to NULL" (render from the backing object).
#[derive(Debug, Clone, Default)]
pub struct NodePatch {
    pub label: Option<Option<String>>,
    pub pinned: Option<bool>,
    pub status: Option<String>,
    pub rank: Option<f64>,
    /// Re-parent target node id.
    pub parent_id: Option<String>,
}

/// Fields an edge PATCH may touch (accept/dismiss/relabel per the spec's
/// route table — no from/to/kind changes).
#[derive(Debug, Clone, Default)]
pub struct EdgePatch {
    pub label: Option<Option<String>>,
    pub status: Option<String>,
}

/// One placed node in a layout write.
#[derive(Debug, Clone)]
pub struct NodeLayout {
    pub node_id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub collapsed: bool,
}

/// Outcome of `create_map_node`. The fingerprint unique index is both the
/// dedup AND tombstone key: a conflicting row with `status = 'dismissed'`
/// means the fingerprint was previously dismissed (a fresh uuid cannot
/// bypass that tombstone — nothing is inserted), while a conflicting row in
/// any other status is a plain live duplicate.
#[derive(Debug, Clone, PartialEq)]
pub enum CreateNodeOutcome {
    Created(PageMapNode),
    Duplicate(PageMapNode),
    Tombstoned,
}

/// Outcome of `create_map_edge`, mirroring `CreateNodeOutcome` for the
/// `UNIQUE(page_id, from_node, to_node, kind)` key.
#[derive(Debug, Clone, PartialEq)]
pub enum CreateEdgeOutcome {
    Created(PageMapEdge),
    Duplicate(PageMapEdge),
    Tombstoned,
}

const NODE_COLUMNS: &str = "id, page_id, parent_id, rank, ref_kind, ref_id, label, status, \
     pinned, placed, collapsed, x, y, width, height, fingerprint, provenance, created_at, updated_at";
const EDGE_COLUMNS: &str =
    "id, page_id, from_node, to_node, kind, label, status, provenance, created_at";

fn row_to_page_map(row: &libsql::Row) -> Result<PageMap, WenlanError> {
    Ok(PageMap {
        page_id: row
            .get(0)
            .map_err(|e| WenlanError::VectorDb(format!("page_map.page_id: {e}")))?,
        revision: row
            .get(1)
            .map_err(|e| WenlanError::VectorDb(format!("page_map.revision: {e}")))?,
        map_schema: row
            .get(2)
            .map_err(|e| WenlanError::VectorDb(format!("page_map.map_schema: {e}")))?,
        viewport: row
            .get(3)
            .map_err(|e| WenlanError::VectorDb(format!("page_map.viewport: {e}")))?,
        generated_at: row
            .get(4)
            .map_err(|e| WenlanError::VectorDb(format!("page_map.generated_at: {e}")))?,
        created_at: row
            .get(5)
            .map_err(|e| WenlanError::VectorDb(format!("page_map.created_at: {e}")))?,
        updated_at: row
            .get(6)
            .map_err(|e| WenlanError::VectorDb(format!("page_map.updated_at: {e}")))?,
    })
}

fn row_to_map_node(row: &libsql::Row) -> Result<PageMapNode, WenlanError> {
    Ok(PageMapNode {
        id: row
            .get(0)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.id: {e}")))?,
        page_id: row
            .get(1)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.page_id: {e}")))?,
        parent_id: row
            .get(2)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.parent_id: {e}")))?,
        rank: row
            .get(3)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.rank: {e}")))?,
        ref_kind: row
            .get(4)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.ref_kind: {e}")))?,
        ref_id: row
            .get(5)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.ref_id: {e}")))?,
        label: row
            .get(6)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.label: {e}")))?,
        status: row
            .get(7)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.status: {e}")))?,
        pinned: row
            .get::<i64>(8)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.pinned: {e}")))?
            != 0,
        placed: row
            .get::<i64>(9)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.placed: {e}")))?
            != 0,
        collapsed: row
            .get::<i64>(10)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.collapsed: {e}")))?
            != 0,
        x: row
            .get(11)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.x: {e}")))?,
        y: row
            .get(12)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.y: {e}")))?,
        width: row
            .get(13)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.width: {e}")))?,
        height: row
            .get(14)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.height: {e}")))?,
        fingerprint: row
            .get(15)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.fingerprint: {e}")))?,
        provenance: row
            .get(16)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.provenance: {e}")))?,
        created_at: row
            .get(17)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.created_at: {e}")))?,
        updated_at: row
            .get(18)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_node.updated_at: {e}")))?,
    })
}

fn row_to_map_edge(row: &libsql::Row) -> Result<PageMapEdge, WenlanError> {
    Ok(PageMapEdge {
        id: row
            .get(0)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.id: {e}")))?,
        page_id: row
            .get(1)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.page_id: {e}")))?,
        from_node: row
            .get(2)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.from_node: {e}")))?,
        to_node: row
            .get(3)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.to_node: {e}")))?,
        kind: row
            .get(4)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.kind: {e}")))?,
        label: row
            .get(5)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.label: {e}")))?,
        status: row
            .get(6)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.status: {e}")))?,
        provenance: row
            .get(7)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.provenance: {e}")))?,
        created_at: row
            .get(8)
            .map_err(|e| WenlanError::VectorDb(format!("page_map_edge.created_at: {e}")))?,
    })
}

/// ASCII Unit Separator — joins fingerprint components unambiguously, since
/// `validate_ref_component` rejects it from ever appearing inside a
/// `ref_kind`/`ref_id`. Replaces the old `"{kind}:{id}@{parent}"` encoding,
/// where a ref_id containing `:` or `@` could inject a fake component
/// boundary and collide with an unrelated node's fingerprint.
const FINGERPRINT_SEP: char = '\u{1f}';

/// `fingerprint = "{ref_kind}<SEP>{ref_id}<SEP>{parent_ref}"` (spec: Identity
/// & tombstones). The root's own fingerprint is `fingerprint_for("page",
/// page_id, "~")`.
fn fingerprint_for(ref_kind: &str, ref_id: &str, parent_ref: &str) -> String {
    format!("{ref_kind}{FINGERPRINT_SEP}{ref_id}{FINGERPRINT_SEP}{parent_ref}")
}

/// A node's own `"{ref_kind}<SEP>{ref_id}"` token as used in a *child's*
/// fingerprint — `"~"` when the node itself is the root.
fn parent_ref_token(node: &PageMapNode) -> String {
    if node.parent_id.is_none() {
        "~".to_string()
    } else {
        format!("{}{FINGERPRINT_SEP}{}", node.ref_kind, node.ref_id)
    }
}

/// Rejects a `ref_kind`/`ref_id` containing the fingerprint separator — it
/// would otherwise let a crafted ref inject a fake component boundary into
/// `fingerprint_for`'s encoding.
fn validate_ref_component(ref_kind: &str, ref_id: &str) -> Result<(), WenlanError> {
    if ref_kind.contains(FINGERPRINT_SEP) || ref_id.contains(FINGERPRINT_SEP) {
        return Err(WenlanError::Validation(
            "ref_kind/ref_id must not contain the fingerprint separator".to_string(),
        ));
    }
    Ok(())
}

/// Count of non-dismissed direct children of `node_id` — used to block a
/// dismissal (or delete, which dismisses) that would orphan live descendants
/// (spec: dismissal cannot orphan).
async fn count_live_children(
    conn: &libsql::Connection,
    page_id: &str,
    node_id: &str,
) -> Result<i64, WenlanError> {
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM page_map_nodes \
             WHERE page_id = ?1 AND parent_id = ?2 AND status != 'dismissed'",
            libsql::params![page_id, node_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("count_live_children: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("count_live_children row: {e}")))?
    {
        Some(row) => row
            .get(0)
            .map_err(|e| WenlanError::VectorDb(format!("count_live_children col: {e}"))),
        None => Ok(0),
    }
}

fn revision_conflict(page_id: &str, current: i64, expected: i64) -> WenlanError {
    WenlanError::Conflict(format!(
        "page_map for {page_id} is at revision {current}, expected {expected}"
    ))
}

async fn read_page_map_revision(
    conn: &libsql::Connection,
    page_id: &str,
) -> Result<Option<i64>, WenlanError> {
    let mut rows = conn
        .query(
            "SELECT revision FROM page_maps WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_page_map_revision: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_page_map_revision row: {e}")))?
    {
        Some(row) => Ok(Some(row.get(0).map_err(|e| {
            WenlanError::VectorDb(format!("read_page_map_revision col: {e}"))
        })?)),
        None => Ok(None),
    }
}

async fn read_page_map(
    conn: &libsql::Connection,
    page_id: &str,
) -> Result<Option<PageMap>, WenlanError> {
    let mut rows = conn
        .query(
            "SELECT page_id, revision, map_schema, viewport, generated_at, created_at, updated_at \
             FROM page_maps WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_page_map: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_page_map row: {e}")))?
    {
        Some(row) => Ok(Some(row_to_page_map(&row)?)),
        None => Ok(None),
    }
}

async fn read_map_nodes(
    conn: &libsql::Connection,
    page_id: &str,
    include_dismissed: bool,
) -> Result<Vec<PageMapNode>, WenlanError> {
    let sql = if include_dismissed {
        format!("SELECT {NODE_COLUMNS} FROM page_map_nodes WHERE page_id = ?1 ORDER BY rank")
    } else {
        format!(
            "SELECT {NODE_COLUMNS} FROM page_map_nodes WHERE page_id = ?1 AND status != 'dismissed' ORDER BY rank"
        )
    };
    let mut rows = conn
        .query(&sql, libsql::params![page_id])
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_nodes: {e}")))?;
    let mut out = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_nodes row: {e}")))?
    {
        out.push(row_to_map_node(&row)?);
    }
    Ok(out)
}

async fn read_map_edges(
    conn: &libsql::Connection,
    page_id: &str,
    include_dismissed: bool,
) -> Result<Vec<PageMapEdge>, WenlanError> {
    let sql = if include_dismissed {
        format!("SELECT {EDGE_COLUMNS} FROM page_map_edges WHERE page_id = ?1 ORDER BY created_at")
    } else {
        format!(
            "SELECT {EDGE_COLUMNS} FROM page_map_edges WHERE page_id = ?1 AND status != 'dismissed' ORDER BY created_at"
        )
    };
    let mut rows = conn
        .query(&sql, libsql::params![page_id])
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_edges: {e}")))?;
    let mut out = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_edges row: {e}")))?
    {
        out.push(row_to_map_edge(&row)?);
    }
    Ok(out)
}

async fn load_page_map_data(
    conn: &libsql::Connection,
    page_id: &str,
    include_dismissed: bool,
) -> Result<Option<PageMapData>, WenlanError> {
    let map = match read_page_map(conn, page_id).await? {
        Some(m) => m,
        None => return Ok(None),
    };
    let nodes = read_map_nodes(conn, page_id, include_dismissed).await?;
    let edges = read_map_edges(conn, page_id, include_dismissed).await?;
    Ok(Some(PageMapData { map, nodes, edges }))
}

async fn read_map_node(
    conn: &libsql::Connection,
    page_id: &str,
    node_id: &str,
) -> Result<Option<PageMapNode>, WenlanError> {
    let sql = format!("SELECT {NODE_COLUMNS} FROM page_map_nodes WHERE page_id = ?1 AND id = ?2");
    let mut rows = conn
        .query(&sql, libsql::params![page_id, node_id])
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_node: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_node row: {e}")))?
    {
        Some(row) => Ok(Some(row_to_map_node(&row)?)),
        None => Ok(None),
    }
}

async fn read_map_node_by_fingerprint(
    conn: &libsql::Connection,
    page_id: &str,
    fingerprint: &str,
) -> Result<Option<PageMapNode>, WenlanError> {
    let sql = format!(
        "SELECT {NODE_COLUMNS} FROM page_map_nodes WHERE page_id = ?1 AND fingerprint = ?2"
    );
    let mut rows = conn
        .query(&sql, libsql::params![page_id, fingerprint])
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_node_by_fingerprint: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_node_by_fingerprint row: {e}")))?
    {
        Some(row) => Ok(Some(row_to_map_node(&row)?)),
        None => Ok(None),
    }
}

async fn read_map_edge(
    conn: &libsql::Connection,
    page_id: &str,
    edge_id: &str,
) -> Result<Option<PageMapEdge>, WenlanError> {
    let sql = format!("SELECT {EDGE_COLUMNS} FROM page_map_edges WHERE page_id = ?1 AND id = ?2");
    let mut rows = conn
        .query(&sql, libsql::params![page_id, edge_id])
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_edge: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_edge row: {e}")))?
    {
        Some(row) => Ok(Some(row_to_map_edge(&row)?)),
        None => Ok(None),
    }
}

async fn read_map_edge_by_key(
    conn: &libsql::Connection,
    page_id: &str,
    from_node: &str,
    to_node: &str,
    kind: &str,
) -> Result<Option<PageMapEdge>, WenlanError> {
    let sql = format!(
        "SELECT {EDGE_COLUMNS} FROM page_map_edges \
         WHERE page_id = ?1 AND from_node = ?2 AND to_node = ?3 AND kind = ?4"
    );
    let mut rows = conn
        .query(&sql, libsql::params![page_id, from_node, to_node, kind])
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_edge_by_key: {e}")))?;
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("read_map_edge_by_key row: {e}")))?
    {
        Some(row) => Ok(Some(row_to_map_edge(&row)?)),
        None => Ok(None),
    }
}

/// Distinguishes "doesn't exist anywhere" (404 NotFound) from "exists, but
/// in a different page's map" (422 graph-invariant violation — spec: parent
/// / edge endpoints must resolve within the same map) for a referenced node
/// id that wasn't found scoped to `page_id`.
async fn missing_node_error(
    conn: &libsql::Connection,
    node_id: &str,
) -> Result<WenlanError, WenlanError> {
    let exists_elsewhere = conn
        .query(
            "SELECT 1 FROM page_map_nodes WHERE id = ?1 LIMIT 1",
            libsql::params![node_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("missing_node_error lookup: {e}")))?
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("missing_node_error row: {e}")))?
        .is_some();
    Ok(if exists_elsewhere {
        WenlanError::Validation(format!("node {node_id} is not part of this page's map"))
    } else {
        WenlanError::NotFound(format!("node {node_id} not found"))
    })
}

/// Walk-to-root cycle check (spec: maps are small, so a straight walk is
/// sufficient): true if re-parenting `node_id` under `new_parent_id` would
/// make `node_id` its own ancestor. A depth cap guards a corrupted spine
/// from looping forever; a runaway walk is conservatively treated as a
/// cycle.
async fn node_would_cycle(
    conn: &libsql::Connection,
    page_id: &str,
    node_id: &str,
    new_parent_id: &str,
) -> Result<bool, WenlanError> {
    let mut current = new_parent_id.to_string();
    for _ in 0..10_000 {
        if current == node_id {
            return Ok(true);
        }
        match read_map_node(conn, page_id, &current).await? {
            Some(n) => match n.parent_id {
                Some(p) => current = p,
                None => return Ok(false), // reached the root
            },
            None => return Ok(false),
        }
    }
    Ok(true)
}

/// The bump-and-check helper: atomically bumps `page_maps.revision` from
/// `base_revision` to `base_revision + 1`. `rows_affected == 0` means
/// `base_revision` was stale (someone else wrote first) → `Conflict`.
async fn bump_revision(
    conn: &libsql::Connection,
    page_id: &str,
    base_revision: i64,
) -> Result<i64, WenlanError> {
    let rows = conn
        .execute(
            "UPDATE page_maps SET revision = revision + 1, updated_at = datetime('now') \
             WHERE page_id = ?1 AND revision = ?2",
            libsql::params![page_id, base_revision],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("bump_revision: {e}")))?;
    if rows == 0 {
        return Err(WenlanError::Conflict(format!(
            "page_map for {page_id} changed concurrently (expected revision {base_revision})"
        )));
    }
    Ok(base_revision + 1)
}

async fn ensure_page_map_owner_active(
    conn: &libsql::Connection,
    page_id: &str,
) -> Result<(), WenlanError> {
    let mut rows = conn
        .query(
            "SELECT status FROM pages WHERE id=?1 LIMIT 1",
            libsql::params![page_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("load Page Map owner status: {e}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("load Page Map owner status row: {e}")))?
    else {
        return Err(WenlanError::NotFound(format!("Page {page_id}")));
    };
    let status: String = row
        .get(0)
        .map_err(|e| WenlanError::VectorDb(format!("Page Map owner status: {e}")))?;
    if status != "active" {
        return Err(WenlanError::Validation(format!(
            "Page {page_id} is not active"
        )));
    }
    Ok(())
}

async fn read_active_page_map_revision(
    conn: &libsql::Connection,
    page_id: &str,
) -> Result<Option<i64>, WenlanError> {
    ensure_page_map_owner_active(conn, page_id).await?;
    read_page_map_revision(conn, page_id).await
}

impl MemoryDB {
    /// Full map read. Non-dismissed nodes/edges by default; pass
    /// `include_dismissed = true` for the `?include=dismissed` audit view.
    /// Returns `None` when no `page_maps` row exists yet — the API layer's
    /// "revision 0, empty map" response is synthesized there, never stored.
    pub async fn get_page_map(
        &self,
        page_id: &str,
        include_dismissed: bool,
    ) -> Result<Option<PageMapData>, WenlanError> {
        let conn = self.conn.lock().await;
        load_page_map_data(&conn, page_id, include_dismissed).await
    }

    /// Reads `page_maps.generated_at` — the server-internal watermark stamped
    /// by the improve pass (the last time suggestions were generated for this
    /// page). `None` when no map row exists or it has never been generated.
    /// Not part of the client wire shape; the proactive scheduler uses it to
    /// skip pages whose `last_modified` hasn't advanced past it.
    pub async fn page_map_generated_at(
        &self,
        page_id: &str,
    ) -> Result<Option<String>, WenlanError> {
        let conn = self.conn.lock().await;
        Ok(read_page_map(&conn, page_id)
            .await?
            .and_then(|m| m.generated_at))
    }

    /// Stamps `page_maps.generated_at` to `generated_at` WITHOUT bumping the
    /// revision — it is a server-internal watermark, not a client-visible map
    /// change, so bumping would raise spurious 409s against in-flight client
    /// edits. No-op when no map row exists.
    ///
    /// The caller passes `page.last_modified` captured at the START of an
    /// improve pass, not `now()` — otherwise an edit made DURING the pass
    /// (bumping `last_modified` after the watermark would be stamped) could
    /// land between capture and stamp and be silently missed by the next
    /// proactive tick. Same clock and format as `pages.last_modified` (see
    /// `post_write::create_page_with_tuning`): an RFC3339 UTC string. This
    /// matters because SQLite's `datetime('now')` ("YYYY-MM-DD HH:MM:SS")
    /// both truncates to whole seconds and uses a space separator, so a
    /// lexical or same-second compare against RFC3339 `last_modified` would
    /// be wrong. One format on both sides keeps the compare honest.
    pub async fn stamp_page_map_generated_at(
        &self,
        page_id: &str,
        generated_at: &str,
    ) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        ensure_page_map_owner_active(&conn, page_id).await?;
        conn.execute(
            "UPDATE page_maps SET generated_at = ?2 WHERE page_id = ?1",
            libsql::params![page_id, generated_at],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("stamp_page_map_generated_at: {e}")))?;
        Ok(())
    }

    /// Idempotent: creates the `page_maps` row + root node
    /// (`ref_kind='page'`, `ref_id=page_id`, `parent_id=NULL`, fingerprint
    /// `fingerprint_for("page", page_id, "~")`) in one transaction when no
    /// map exists yet, then returns the current map state either way. Root
    /// creation counts as the first write, so a freshly-initialized map's
    /// `revision` is 1 — 0 is reserved for "no `page_maps` row at all"
    /// (spec: whole-map lifecycle).
    ///
    /// The returned `bool` is `true` only when THIS call performed the
    /// insert, decided atomically from the INSERT's own `rows_affected`
    /// (`ON CONFLICT(page_id) DO NOTHING`) rather than from a separate,
    /// non-atomic pre-check — a caller (e.g. `handle_create_map_node`) must
    /// use this flag, not an earlier `get_page_map` read, to decide whether
    /// to substitute the freshly-initialized revision for a client's
    /// `base_revision`; substituting on a stale earlier read let a genuinely
    /// stale `base_revision` silently succeed instead of 409ing.
    pub async fn init_page_map(&self, page_id: &str) -> Result<(PageMapData, bool), WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("init_page_map begin: {e}")))?;

        let result: Result<bool, WenlanError> = async {
            ensure_page_map_owner_active(&conn, page_id).await?;
            if load_page_map_data(&conn, page_id, true).await?.is_some() {
                return Ok(false);
            }
            let root_id = uuid::Uuid::new_v4().to_string();
            let fingerprint = fingerprint_for("page", page_id, "~");
            let inserted = conn
                .execute(
                    "INSERT INTO page_maps (page_id, revision) VALUES (?1, 0) \
                     ON CONFLICT(page_id) DO NOTHING",
                    libsql::params![page_id],
                )
                .await
                .map_err(|e| {
                    WenlanError::VectorDb(format!("init_page_map insert page_maps: {e}"))
                })?;
            if inserted == 0 {
                // Someone else's insert won; nothing left for us to do.
                return Ok(false);
            }
            conn.execute(
                "INSERT INTO page_map_nodes (id, page_id, parent_id, rank, ref_kind, ref_id, status, fingerprint) \
                 VALUES (?1, ?2, NULL, 0, 'page', ?2, 'active', ?3)",
                libsql::params![root_id, page_id, fingerprint],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("init_page_map insert root: {e}")))?;
            conn.execute(
                "UPDATE page_maps SET revision = 1, updated_at = datetime('now') WHERE page_id = ?1",
                libsql::params![page_id],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("init_page_map bump: {e}")))?;
            Ok(true)
        }
        .await;

        let created = match result {
            Ok(created) => created,
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(e);
            }
        };
        commit_or_rollback(&conn)
            .await
            .map_err(|e| WenlanError::VectorDb(format!("init_page_map commit: {e}")))?;

        let data = load_page_map_data(&conn, page_id, true)
            .await?
            .ok_or_else(|| {
                WenlanError::VectorDb(format!("init_page_map: {page_id} missing after insert"))
            })?;
        Ok((data, created))
    }

    /// Creates a node under `parent_id`. Requires an existing `page_maps`
    /// row (call `init_page_map` first on an uninitialized page — the root
    /// node's id becomes the natural first `parent_id`). Computes
    /// `fingerprint` and relies on the unique index for dedup/tombstone
    /// detection: see `CreateNodeOutcome`.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_map_node(
        &self,
        page_id: &str,
        base_revision: i64,
        parent_id: &str,
        ref_kind: &str,
        ref_id: &str,
        label: Option<&str>,
        rank: f64,
    ) -> Result<CreateNodeOutcome, WenlanError> {
        if !matches!(ref_kind, "memory" | "entity" | "page" | "section") {
            return Err(WenlanError::Validation(format!(
                "unknown ref_kind '{ref_kind}'"
            )));
        }
        validate_ref_component(ref_kind, ref_id)?;
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("create_map_node begin: {e}")))?;

        let result = async {
            let current_revision = read_active_page_map_revision(&conn, page_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("no page_map for page {page_id}")))?;
            if current_revision != base_revision {
                return Err(revision_conflict(page_id, current_revision, base_revision));
            }

            let parent = match read_map_node(&conn, page_id, parent_id).await? {
                Some(p) => p,
                None => return Err(missing_node_error(&conn, parent_id).await?),
            };
            if parent.status == "dismissed" {
                return Err(WenlanError::Validation(format!(
                    "parent {parent_id} is dismissed"
                )));
            }
            let parent_ref = parent_ref_token(&parent);
            let fingerprint = fingerprint_for(ref_kind, ref_id, &parent_ref);

            let node_id = uuid::Uuid::new_v4().to_string();
            let inserted = conn
                .execute(
                    "INSERT INTO page_map_nodes \
                     (id, page_id, parent_id, rank, ref_kind, ref_id, label, status, fingerprint) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'active', ?8) \
                     ON CONFLICT(page_id, fingerprint) DO NOTHING",
                    libsql::params![
                        node_id.clone(),
                        page_id,
                        parent_id,
                        rank,
                        ref_kind,
                        ref_id,
                        label,
                        fingerprint.clone()
                    ],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("create_map_node insert: {e}")))?;

            if inserted == 0 {
                let existing = read_map_node_by_fingerprint(&conn, page_id, &fingerprint)
                    .await?
                    .ok_or_else(|| {
                        WenlanError::VectorDb(
                            "create_map_node: conflicting row vanished".to_string(),
                        )
                    })?;
                return Ok(if existing.status == "dismissed" {
                    CreateNodeOutcome::Tombstoned
                } else {
                    CreateNodeOutcome::Duplicate(existing)
                });
            }

            bump_revision(&conn, page_id, current_revision).await?;
            let created = read_map_node(&conn, page_id, &node_id)
                .await?
                .ok_or_else(|| {
                    WenlanError::VectorDb("create_map_node: row missing after insert".to_string())
                })?;
            Ok(CreateNodeOutcome::Created(created))
        }
        .await;

        match result {
            Ok(outcome) => {
                commit_or_rollback(&conn)
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("create_map_node commit: {e}")))?;
                Ok(outcome)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(e)
            }
        }
    }

    /// Suggestion-path sibling of [`MemoryDB::create_map_node`]: inserts a node
    /// with `status = 'suggested'` and a `provenance` stamp, used by the
    /// improve pass. INSERT-ONLY by construction — the same `ON CONFLICT(page_id,
    /// fingerprint) DO NOTHING` makes re-proposing an existing OR dismissed
    /// fingerprint a no-op (`Duplicate` / `Tombstoned`), so a suggestion pass
    /// can never modify, resurrect, or overwrite a pinned/active/dismissed row.
    /// Kept a separate accessor (rather than parameterizing `create_map_node`)
    /// so the tested active-node write path stays byte-for-byte untouched.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_suggested_map_node(
        &self,
        page_id: &str,
        base_revision: i64,
        parent_id: &str,
        ref_kind: &str,
        ref_id: &str,
        label: Option<&str>,
        rank: f64,
        provenance: &str,
    ) -> Result<CreateNodeOutcome, WenlanError> {
        if !matches!(ref_kind, "memory" | "entity" | "page" | "section") {
            return Err(WenlanError::Validation(format!(
                "unknown ref_kind '{ref_kind}'"
            )));
        }
        validate_ref_component(ref_kind, ref_id)?;
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("create_suggested_map_node begin: {e}")))?;

        let result = async {
            let current_revision = read_active_page_map_revision(&conn, page_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("no page_map for page {page_id}")))?;
            if current_revision != base_revision {
                return Err(revision_conflict(page_id, current_revision, base_revision));
            }

            let parent = match read_map_node(&conn, page_id, parent_id).await? {
                Some(p) => p,
                None => return Err(missing_node_error(&conn, parent_id).await?),
            };
            if parent.status == "dismissed" {
                return Err(WenlanError::Validation(format!(
                    "parent {parent_id} is dismissed"
                )));
            }
            let parent_ref = parent_ref_token(&parent);
            let fingerprint = fingerprint_for(ref_kind, ref_id, &parent_ref);

            let node_id = uuid::Uuid::new_v4().to_string();
            let inserted = conn
                .execute(
                    "INSERT INTO page_map_nodes \
                     (id, page_id, parent_id, rank, ref_kind, ref_id, label, status, fingerprint, provenance) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'suggested', ?8, ?9) \
                     ON CONFLICT(page_id, fingerprint) DO NOTHING",
                    libsql::params![
                        node_id.clone(),
                        page_id,
                        parent_id,
                        rank,
                        ref_kind,
                        ref_id,
                        label,
                        fingerprint.clone(),
                        provenance
                    ],
                )
                .await
                .map_err(|e| {
                    WenlanError::VectorDb(format!("create_suggested_map_node insert: {e}"))
                })?;

            if inserted == 0 {
                let existing = read_map_node_by_fingerprint(&conn, page_id, &fingerprint)
                    .await?
                    .ok_or_else(|| {
                        WenlanError::VectorDb(
                            "create_suggested_map_node: conflicting row vanished".to_string(),
                        )
                    })?;
                return Ok(if existing.status == "dismissed" {
                    CreateNodeOutcome::Tombstoned
                } else {
                    CreateNodeOutcome::Duplicate(existing)
                });
            }

            bump_revision(&conn, page_id, current_revision).await?;
            let created = read_map_node(&conn, page_id, &node_id)
                .await?
                .ok_or_else(|| {
                    WenlanError::VectorDb(
                        "create_suggested_map_node: row missing after insert".to_string(),
                    )
                })?;
            Ok(CreateNodeOutcome::Created(created))
        }
        .await;

        match result {
            Ok(outcome) => {
                commit_or_rollback(&conn).await.map_err(|e| {
                    WenlanError::VectorDb(format!("create_suggested_map_node commit: {e}"))
                })?;
                Ok(outcome)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(e)
            }
        }
    }

    /// Applies `patch` to a node. Any touched field (`label`, `status`,
    /// `rank`, `parent_id`) sets `pinned = 1` unless the same patch also
    /// gives `pinned` explicitly. Status transitions are restricted to
    /// `suggested->active`, `suggested->dismissed`, `active->dismissed`.
    /// Root (`parent_id IS NULL`) cannot be dismissed or re-parented.
    /// Re-parenting rejects cycles (walk-to-root) and recomputes
    /// `fingerprint`, which must not collide with another node's.
    pub async fn patch_map_node(
        &self,
        page_id: &str,
        base_revision: i64,
        node_id: &str,
        patch: NodePatch,
    ) -> Result<PageMapNode, WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("patch_map_node begin: {e}")))?;

        let result = async {
            let current_revision = read_active_page_map_revision(&conn, page_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("no page_map for page {page_id}")))?;
            if current_revision != base_revision {
                return Err(revision_conflict(page_id, current_revision, base_revision));
            }
            let node = read_map_node(&conn, page_id, node_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("node {node_id} not found")))?;
            if node.status == "dismissed" {
                return Err(WenlanError::Validation(
                    "node is dismissed and cannot be modified".to_string(),
                ));
            }
            let is_root = node.parent_id.is_none();

            if let Some(new_status) = &patch.status {
                if is_root && new_status == "dismissed" {
                    return Err(WenlanError::Validation(
                        "root node cannot be dismissed".to_string(),
                    ));
                }
                let allowed = matches!(
                    (node.status.as_str(), new_status.as_str()),
                    ("suggested", "active") | ("suggested", "dismissed") | ("active", "dismissed")
                );
                if !allowed {
                    return Err(WenlanError::Validation(format!(
                        "invalid status transition {} -> {new_status}",
                        node.status
                    )));
                }
                if new_status == "dismissed" {
                    let live_children = count_live_children(&conn, page_id, node_id).await?;
                    if live_children > 0 {
                        return Err(WenlanError::Validation(
                            "node has live children and cannot be dismissed".to_string(),
                        ));
                    }
                }
            }
            if patch.parent_id.is_some() && is_root {
                return Err(WenlanError::Validation(
                    "root node cannot be re-parented".to_string(),
                ));
            }

            let mut new_fingerprint = node.fingerprint.clone();
            let mut new_parent_id = node.parent_id.clone();
            if let Some(target_parent_id) = &patch.parent_id {
                if target_parent_id == node_id {
                    return Err(WenlanError::Validation(
                        "node cannot be its own parent".to_string(),
                    ));
                }
                let target_parent = match read_map_node(&conn, page_id, target_parent_id).await? {
                    Some(p) => p,
                    None => return Err(missing_node_error(&conn, target_parent_id).await?),
                };
                if target_parent.status == "dismissed" {
                    return Err(WenlanError::Validation(format!(
                        "parent {target_parent_id} is dismissed"
                    )));
                }
                if node_would_cycle(&conn, page_id, node_id, target_parent_id).await? {
                    return Err(WenlanError::Validation(
                        "re-parent would create a cycle".to_string(),
                    ));
                }
                let parent_ref = parent_ref_token(&target_parent);
                new_fingerprint = fingerprint_for(&node.ref_kind, &node.ref_id, &parent_ref);
                if new_fingerprint != node.fingerprint {
                    if let Some(collision) =
                        read_map_node_by_fingerprint(&conn, page_id, &new_fingerprint).await?
                    {
                        if collision.id != node.id {
                            return Err(WenlanError::Validation(
                                "re-parent collides with an existing node under the new parent"
                                    .to_string(),
                            ));
                        }
                    }
                }
                new_parent_id = Some(target_parent_id.clone());
            }

            let new_label = match &patch.label {
                Some(inner) => inner.clone(),
                None => node.label.clone(),
            };
            let new_rank = patch.rank.unwrap_or(node.rank);
            let new_status = patch.status.clone().unwrap_or_else(|| node.status.clone());
            let touched_other_field = patch.label.is_some()
                || patch.status.is_some()
                || patch.rank.is_some()
                || patch.parent_id.is_some();
            let new_pinned = patch.pinned.unwrap_or(touched_other_field || node.pinned);

            conn.execute(
                "UPDATE page_map_nodes \
                 SET parent_id = ?1, rank = ?2, label = ?3, status = ?4, pinned = ?5, \
                     fingerprint = ?6, updated_at = datetime('now') \
                 WHERE page_id = ?7 AND id = ?8",
                libsql::params![
                    new_parent_id,
                    new_rank,
                    new_label,
                    new_status,
                    new_pinned as i64,
                    new_fingerprint,
                    page_id,
                    node_id
                ],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("patch_map_node update: {e}")))?;

            bump_revision(&conn, page_id, current_revision).await?;
            read_map_node(&conn, page_id, node_id)
                .await?
                .ok_or_else(|| {
                    WenlanError::VectorDb("patch_map_node: row missing after update".to_string())
                })
        }
        .await;

        match result {
            Ok(node) => {
                commit_or_rollback(&conn)
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("patch_map_node commit: {e}")))?;
                Ok(node)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(e)
            }
        }
    }

    /// Tombstones a node (`status = 'dismissed'`) — the DELETE route's
    /// semantics. Reuses `patch_map_node` so root protection and the status
    /// state machine stay in one place.
    pub async fn delete_map_node(
        &self,
        page_id: &str,
        base_revision: i64,
        node_id: &str,
    ) -> Result<PageMapNode, WenlanError> {
        self.patch_map_node(
            page_id,
            base_revision,
            node_id,
            NodePatch {
                status: Some("dismissed".to_string()),
                ..Default::default()
            },
        )
        .await
    }

    /// Creates a user cross-link or suggested edge between two nodes in the
    /// same map. Same fingerprint-style dedup/tombstone handling as
    /// `create_map_node`, keyed on `UNIQUE(page_id, from_node, to_node,
    /// kind)`.
    pub async fn create_map_edge(
        &self,
        page_id: &str,
        base_revision: i64,
        from_node: &str,
        to_node: &str,
        kind: &str,
        label: Option<&str>,
    ) -> Result<CreateEdgeOutcome, WenlanError> {
        if !matches!(kind, "link" | "suggested") {
            return Err(WenlanError::Validation(format!(
                "unknown edge kind '{kind}'"
            )));
        }
        if from_node == to_node {
            return Err(WenlanError::Validation(
                "edge cannot connect a node to itself".to_string(),
            ));
        }
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("create_map_edge begin: {e}")))?;

        let result = async {
            let current_revision = read_active_page_map_revision(&conn, page_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("no page_map for page {page_id}")))?;
            if current_revision != base_revision {
                return Err(revision_conflict(page_id, current_revision, base_revision));
            }
            let from = match read_map_node(&conn, page_id, from_node).await? {
                Some(n) => n,
                None => return Err(missing_node_error(&conn, from_node).await?),
            };
            if from.status == "dismissed" {
                return Err(WenlanError::Validation(format!(
                    "node {from_node} is dismissed"
                )));
            }
            let to = match read_map_node(&conn, page_id, to_node).await? {
                Some(n) => n,
                None => return Err(missing_node_error(&conn, to_node).await?),
            };
            if to.status == "dismissed" {
                return Err(WenlanError::Validation(format!(
                    "node {to_node} is dismissed"
                )));
            }

            let edge_id = uuid::Uuid::new_v4().to_string();
            let inserted = conn
                .execute(
                    "INSERT INTO page_map_edges (id, page_id, from_node, to_node, kind, label, status) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'active') \
                     ON CONFLICT(page_id, from_node, to_node, kind) DO NOTHING",
                    libsql::params![edge_id.clone(), page_id, from_node, to_node, kind, label],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("create_map_edge insert: {e}")))?;

            if inserted == 0 {
                let existing = read_map_edge_by_key(&conn, page_id, from_node, to_node, kind)
                    .await?
                    .ok_or_else(|| {
                        WenlanError::VectorDb(
                            "create_map_edge: conflicting row vanished".to_string(),
                        )
                    })?;
                return Ok(if existing.status == "dismissed" {
                    CreateEdgeOutcome::Tombstoned
                } else {
                    CreateEdgeOutcome::Duplicate(existing)
                });
            }

            bump_revision(&conn, page_id, current_revision).await?;
            let created = read_map_edge(&conn, page_id, &edge_id).await?.ok_or_else(|| {
                WenlanError::VectorDb("create_map_edge: row missing after insert".to_string())
            })?;
            Ok(CreateEdgeOutcome::Created(created))
        }
        .await;

        match result {
            Ok(outcome) => {
                commit_or_rollback(&conn)
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("create_map_edge commit: {e}")))?;
                Ok(outcome)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(e)
            }
        }
    }

    /// Suggestion-path sibling of [`MemoryDB::create_map_edge`]: inserts an
    /// edge with `status = 'suggested'` and a `provenance` stamp. INSERT-ONLY on
    /// the `UNIQUE(page_id, from_node, to_node, kind)` key — re-proposing an
    /// existing or dismissed edge is a no-op (`Duplicate` / `Tombstoned`),
    /// mirroring the node sibling. Existing `create_map_edge` stays untouched.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_suggested_map_edge(
        &self,
        page_id: &str,
        base_revision: i64,
        from_node: &str,
        to_node: &str,
        kind: &str,
        label: Option<&str>,
        provenance: &str,
    ) -> Result<CreateEdgeOutcome, WenlanError> {
        if !matches!(kind, "link" | "suggested") {
            return Err(WenlanError::Validation(format!(
                "unknown edge kind '{kind}'"
            )));
        }
        if from_node == to_node {
            return Err(WenlanError::Validation(
                "edge cannot connect a node to itself".to_string(),
            ));
        }
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("create_suggested_map_edge begin: {e}")))?;

        let result = async {
            let current_revision = read_active_page_map_revision(&conn, page_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("no page_map for page {page_id}")))?;
            if current_revision != base_revision {
                return Err(revision_conflict(page_id, current_revision, base_revision));
            }
            let from = match read_map_node(&conn, page_id, from_node).await? {
                Some(n) => n,
                None => return Err(missing_node_error(&conn, from_node).await?),
            };
            if from.status == "dismissed" {
                return Err(WenlanError::Validation(format!(
                    "node {from_node} is dismissed"
                )));
            }
            let to = match read_map_node(&conn, page_id, to_node).await? {
                Some(n) => n,
                None => return Err(missing_node_error(&conn, to_node).await?),
            };
            if to.status == "dismissed" {
                return Err(WenlanError::Validation(format!(
                    "node {to_node} is dismissed"
                )));
            }

            let edge_id = uuid::Uuid::new_v4().to_string();
            let inserted = conn
                .execute(
                    "INSERT INTO page_map_edges \
                     (id, page_id, from_node, to_node, kind, label, status, provenance) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'suggested', ?7) \
                     ON CONFLICT(page_id, from_node, to_node, kind) DO NOTHING",
                    libsql::params![
                        edge_id.clone(),
                        page_id,
                        from_node,
                        to_node,
                        kind,
                        label,
                        provenance
                    ],
                )
                .await
                .map_err(|e| {
                    WenlanError::VectorDb(format!("create_suggested_map_edge insert: {e}"))
                })?;

            if inserted == 0 {
                let existing = read_map_edge_by_key(&conn, page_id, from_node, to_node, kind)
                    .await?
                    .ok_or_else(|| {
                        WenlanError::VectorDb(
                            "create_suggested_map_edge: conflicting row vanished".to_string(),
                        )
                    })?;
                return Ok(if existing.status == "dismissed" {
                    CreateEdgeOutcome::Tombstoned
                } else {
                    CreateEdgeOutcome::Duplicate(existing)
                });
            }

            bump_revision(&conn, page_id, current_revision).await?;
            let created = read_map_edge(&conn, page_id, &edge_id)
                .await?
                .ok_or_else(|| {
                    WenlanError::VectorDb(
                        "create_suggested_map_edge: row missing after insert".to_string(),
                    )
                })?;
            Ok(CreateEdgeOutcome::Created(created))
        }
        .await;

        match result {
            Ok(outcome) => {
                commit_or_rollback(&conn).await.map_err(|e| {
                    WenlanError::VectorDb(format!("create_suggested_map_edge commit: {e}"))
                })?;
                Ok(outcome)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(e)
            }
        }
    }

    /// Applies `patch` (status/label only) to an edge. Same status state
    /// machine as nodes; edges have no root-equivalent invariant.
    pub async fn patch_map_edge(
        &self,
        page_id: &str,
        base_revision: i64,
        edge_id: &str,
        patch: EdgePatch,
    ) -> Result<PageMapEdge, WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("patch_map_edge begin: {e}")))?;

        let result = async {
            let current_revision = read_active_page_map_revision(&conn, page_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("no page_map for page {page_id}")))?;
            if current_revision != base_revision {
                return Err(revision_conflict(page_id, current_revision, base_revision));
            }
            let edge = read_map_edge(&conn, page_id, edge_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("edge {edge_id} not found")))?;
            if edge.status == "dismissed" {
                return Err(WenlanError::Validation(
                    "edge is dismissed and cannot be modified".to_string(),
                ));
            }

            if let Some(new_status) = &patch.status {
                let allowed = matches!(
                    (edge.status.as_str(), new_status.as_str()),
                    ("suggested", "active") | ("suggested", "dismissed") | ("active", "dismissed")
                );
                if !allowed {
                    return Err(WenlanError::Validation(format!(
                        "invalid status transition {} -> {new_status}",
                        edge.status
                    )));
                }
            }

            let new_label = match &patch.label {
                Some(inner) => inner.clone(),
                None => edge.label.clone(),
            };
            let new_status = patch.status.clone().unwrap_or_else(|| edge.status.clone());

            conn.execute(
                "UPDATE page_map_edges SET label = ?1, status = ?2 WHERE page_id = ?3 AND id = ?4",
                libsql::params![new_label, new_status, page_id, edge_id],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("patch_map_edge update: {e}")))?;

            bump_revision(&conn, page_id, current_revision).await?;
            read_map_edge(&conn, page_id, edge_id)
                .await?
                .ok_or_else(|| {
                    WenlanError::VectorDb("patch_map_edge: row missing after update".to_string())
                })
        }
        .await;

        match result {
            Ok(edge) => {
                commit_or_rollback(&conn)
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("patch_map_edge commit: {e}")))?;
                Ok(edge)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(e)
            }
        }
    }

    /// Tombstones an edge (`status = 'dismissed'`) — the DELETE route's
    /// semantics. Reuses `patch_map_edge`.
    pub async fn delete_map_edge(
        &self,
        page_id: &str,
        base_revision: i64,
        edge_id: &str,
    ) -> Result<PageMapEdge, WenlanError> {
        self.patch_map_edge(
            page_id,
            base_revision,
            edge_id,
            EdgePatch {
                status: Some("dismissed".to_string()),
                ..Default::default()
            },
        )
        .await
    }

    /// Writes viewport + per-node positions/sizes/collapsed state in one
    /// transaction, setting `placed = 1` AND `pinned = 1` on every positioned
    /// node (spec: moving a node with placement is a user mutation, so it
    /// pins the same way any other user edit does). Every `node_id` in
    /// `positions` must already belong to this map.
    pub async fn put_page_map_layout(
        &self,
        page_id: &str,
        base_revision: i64,
        viewport: Option<&str>,
        positions: &[NodeLayout],
    ) -> Result<PageMapData, WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("put_page_map_layout begin: {e}")))?;

        let result = async {
            let current_revision = read_active_page_map_revision(&conn, page_id)
                .await?
                .ok_or_else(|| WenlanError::NotFound(format!("no page_map for page {page_id}")))?;
            if current_revision != base_revision {
                return Err(revision_conflict(page_id, current_revision, base_revision));
            }

            for pos in positions {
                let rows = conn
                    .execute(
                        "UPDATE page_map_nodes \
                         SET x = ?1, y = ?2, width = ?3, height = ?4, collapsed = ?5, placed = 1, \
                             pinned = 1, updated_at = datetime('now') \
                         WHERE page_id = ?6 AND id = ?7",
                        libsql::params![
                            pos.x,
                            pos.y,
                            pos.width,
                            pos.height,
                            pos.collapsed as i64,
                            page_id,
                            pos.node_id.clone()
                        ],
                    )
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("put_page_map_layout node: {e}")))?;
                if rows == 0 {
                    return Err(missing_node_error(&conn, &pos.node_id).await?);
                }
            }
            conn.execute(
                "UPDATE page_maps SET viewport = ?1, updated_at = datetime('now') WHERE page_id = ?2",
                libsql::params![viewport, page_id],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("put_page_map_layout viewport: {e}")))?;

            bump_revision(&conn, page_id, current_revision).await?;
            load_page_map_data(&conn, page_id, true).await?.ok_or_else(|| {
                WenlanError::VectorDb("put_page_map_layout: map missing after update".to_string())
            })
        }
        .await;

        match result {
            Ok(data) => {
                commit_or_rollback(&conn).await.map_err(|e| {
                    WenlanError::VectorDb(format!("put_page_map_layout commit: {e}"))
                })?;
                Ok(data)
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                Err(e)
            }
        }
    }

    /// Full reset: deletes the `page_maps` row; `ON DELETE CASCADE` clears
    /// `page_map_nodes` / `page_map_edges` INCLUDING tombstones (spec:
    /// reset means "start clean"). No `base_revision` — this is the escape
    /// hatch, not a conditional write. Idempotent: a page with no map row
    /// is a no-op.
    pub async fn reset_page_map(&self, page_id: &str) -> Result<(), WenlanError> {
        let conn = self.conn.lock().await;
        ensure_page_map_owner_active(&conn, page_id).await?;
        conn.execute(
            "DELETE FROM page_maps WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("reset_page_map: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "page_map_test.rs"]
mod tests;
