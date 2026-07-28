// SPDX-License-Identifier: Apache-2.0
//! Attached-object capture and replay for the M5 `edges` rebuild.
//!
//! SQLite cannot alter a `CHECK` constraint, so widening `edges` means dropping
//! and recreating it — and `DROP TABLE` takes every index and trigger attached
//! to it. `docs/plans/2026-07-27-m5-edge-rebuild-matrix.md` §7 originally told
//! the implementer to recreate "the space fence, both twins". The live schema
//! attaches eight triggers. Recreating two would have silently dropped the six
//! M4 triggers that keep community grouping and page-community route inputs
//! invalidated, and nothing downstream fails loudly when that happens: the
//! tables stay correct-looking while their invalidation stops firing.
//!
//! So the rebuild does not carry a list. It reads the `CREATE` statements back
//! out of `sqlite_master` before the drop and replays them afterwards, which
//! stays correct when a later rung attaches a ninth trigger without touching
//! this file.

use super::MemoryDB;
use crate::WenlanError;

/// Every column of `edges`, in declaration order (`db.rs:8916`).
///
/// Named explicitly on both sides of the copy rather than using `SELECT *`, so
/// a column added to one shape and not the other is a SQL error instead of a
/// silent positional shift.
pub(super) const EDGE_COLUMNS: &str =
    "edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage, \
                            grounded, root_id, space, weight, payload, provenance, operation_id, \
                            created_at, superseded_by, valid_until";

/// The widened table. Three CHECKs are new, and each one is a storage-layer
/// tooth the artifacts asked for by name.
///
/// The **lineage tooth** (artifact 3 §4) exists because the space fence skips
/// its whole body on `lineage='legacy'`. A `claim_revision → memory` row
/// written as legacy would bypass the fence entirely and land straight in the
/// trusted support set, with nothing but writer discipline in the way. A CHECK
/// has no such exemption, so the storage layer refuses the row.
///
/// The **attests** and **supports** shape CHECKs encode §3 (attestation is
/// never grounded — human presence is provenance, not grounding) and §5
/// (`attests.root_id` equals `src_id`, so an attestation cannot claim one
/// root's authority while pointing at another's identity).
///
/// The full allowed-tuple table stays a writer-side guard, per §2. These three
/// are here because §4 says explicitly that the rebuild is the one chance to
/// add them.
pub(super) const EDGES_WIDENED_DDL: &str = "CREATE TABLE edges_new (
    edge_id TEXT PRIMARY KEY,
    src_id TEXT NOT NULL,
    src_kind TEXT NOT NULL CHECK(src_kind IN ('page','memory','entity','external','claim_revision','root')),
    dst_id TEXT NOT NULL,
    dst_kind TEXT NOT NULL CHECK(dst_kind IN ('page','memory','entity','external','claim_revision','root')),
    edge_type TEXT NOT NULL CHECK(edge_type IN ('mentions','relates','cites','supports','links','attests')),
    lineage TEXT NOT NULL CHECK(lineage IN ('assertion','evidence','synthesis','legacy')),
    grounded INTEGER NOT NULL CHECK(grounded IN (0,1)),
    root_id TEXT REFERENCES provenance_roots(root_id),
    space TEXT NOT NULL,
    weight REAL,
    payload TEXT,
    provenance TEXT,
    operation_id TEXT,
    created_at INTEGER NOT NULL,
    superseded_by TEXT REFERENCES edges(edge_id),
    valid_until INTEGER,
    CHECK (
        lineage <> 'legacy'
        OR (src_kind NOT IN ('claim_revision','root')
            AND dst_kind NOT IN ('claim_revision','root'))
    ),
    CHECK (
        edge_type <> 'attests'
        OR (src_kind = 'root' AND dst_kind = 'claim_revision'
            AND lineage = 'assertion' AND grounded = 0
            AND root_id IS NOT NULL AND root_id = src_id)
    ),
    CHECK (
        edge_type <> 'supports'
        OR (src_kind = 'claim_revision' AND dst_kind = 'memory'
            AND lineage = 'evidence' AND root_id IS NOT NULL)
    ),
    CHECK (
        src_kind NOT IN ('claim_revision','root')
        OR edge_type IN ('supports','attests')
    ),
    CHECK (dst_kind <> 'root'),
    CHECK (dst_kind <> 'claim_revision' OR edge_type = 'attests')
)";

/// The space fence, extended for the two new endpoint kinds (artifact 3 §4).
///
/// `claim_revision` resolves through its claim to the owning page's space —
/// claims are page-scoped, so this is exact. `root` has no space at all
/// (`provenance_roots` carries no such column), so it is **source-side
/// exempt**, scoped as narrowly as the existing destination-side exemption for
/// `cites → external`: only `edge_type='attests' AND src_kind='root'`. The
/// destination `claim_revision` stays fenced, so a root cannot attest into a
/// foreign space.
///
/// Written as one string used for both twins so the INSERT and UPDATE bodies
/// cannot drift apart — the failure the UPDATE twin exists to prevent.
const FENCE_BODY: &str = "
    SELECT RAISE(ABORT, 'edges_space_fence: cross-space edge rejected')
    WHERE (
        NOT (NEW.edge_type = 'attests' AND NEW.src_kind = 'root')
        AND (
            CASE NEW.src_kind
                WHEN 'page' THEN (SELECT space FROM pages WHERE id = NEW.src_id)
                WHEN 'memory' THEN (SELECT space FROM memories WHERE source_id = NEW.src_id)
                WHEN 'entity' THEN (SELECT space FROM entities WHERE id = NEW.src_id)
                WHEN 'claim_revision' THEN (
                    SELECT p.space FROM claim_revisions cr
                      JOIN claims c ON c.claim_id = cr.claim_id
                      JOIN pages p ON p.id = c.page_id
                     WHERE cr.claim_revision_id = NEW.src_id
                )
                ELSE NULL
            END
        ) IS NOT NEW.space
    )
    OR (
        NOT (NEW.edge_type = 'cites' AND NEW.dst_kind = 'external')
        AND (
            CASE NEW.dst_kind
                WHEN 'page' THEN (SELECT space FROM pages WHERE id = NEW.dst_id)
                WHEN 'memory' THEN (SELECT space FROM memories WHERE source_id = NEW.dst_id)
                WHEN 'entity' THEN (SELECT space FROM entities WHERE id = NEW.dst_id)
                WHEN 'claim_revision' THEN (
                    SELECT p.space FROM claim_revisions cr
                      JOIN claims c ON c.claim_id = cr.claim_id
                      JOIN pages p ON p.id = c.page_id
                     WHERE cr.claim_revision_id = NEW.dst_id
                )
                ELSE NULL
            END
        ) IS NOT NEW.space
    );";

/// The four partial indexes from artifact 3 §6.
///
/// All four are active-only and **all four carry the lineage predicate**. An
/// index that filtered lineage while its query did not would still reach the
/// bypassed row by scan — the tooth would look present and not be. The
/// predicate is written here once and the M5 read paths use the same one.
/// `IF NOT EXISTS` because a re-run replays these from the capture first: after
/// one successful rebuild they are attached to `edges` like any other index, so
/// the census picks them up and recreates them before this loop is reached. The
/// post-rebuild set assertion is what proves they are present, so tolerating
/// the second create loses no tooth and keeps the migration re-runnable the way
/// this codebase's repair paths expect.
const M5_INDEXES: [(&str, &str); 4] = [
    (
        "idx_edges_supports_fwd",
        "CREATE INDEX IF NOT EXISTS idx_edges_supports_fwd ON edges(src_id, dst_id)
         WHERE edge_type='supports' AND src_kind='claim_revision'
           AND lineage='evidence' AND valid_until IS NULL",
    ),
    (
        "idx_edges_supports_rev",
        "CREATE INDEX IF NOT EXISTS idx_edges_supports_rev ON edges(dst_id)
         WHERE edge_type='supports' AND src_kind='claim_revision'
           AND lineage='evidence' AND valid_until IS NULL",
    ),
    (
        "idx_edges_attests_fwd",
        "CREATE INDEX IF NOT EXISTS idx_edges_attests_fwd ON edges(dst_id)
         WHERE edge_type='attests' AND src_kind='root'
           AND lineage='assertion' AND valid_until IS NULL",
    ),
    (
        "idx_edges_attests_rev",
        "CREATE INDEX IF NOT EXISTS idx_edges_attests_rev ON edges(src_id, dst_id)
         WHERE edge_type='attests' AND src_kind='root'
           AND lineage='assertion' AND valid_until IS NULL",
    ),
];

/// The one §5 rule the CHECKs cannot express: **"a `generated` root can never
/// attest"**.
///
/// The six table CHECKs constrain `attests` to `src_kind='root'`,
/// `lineage='assertion'`, `grounded=0`, and `root_id = src_id`. None of them can
/// see whether the root at that id is a *human* one, because a SQLite CHECK may
/// not contain a subquery — it is confined to the row being written. So the
/// rule that decides whether a machine can manufacture human presence lives
/// here, in a trigger, which can.
///
/// Fail-closed on both arms of `NOT EXISTS`: an `attests` edge naming a root
/// that is `generated`, and one naming a root that does not exist at all, are
/// both refused. The second matters as much as the first — "no such root" must
/// never read as "nothing to object to".
///
/// Two twins for the same reason the space fence has two: an UPDATE can flip
/// `edge_type` to `attests` or repoint `src_id` at a machine root, and a rule
/// enforced only on INSERT is a rule with a documented way around it.
const M5_TRIGGERS: [(&str, &str); 2] = [
    (
        "edges_attests_requires_human_root",
        "CREATE TRIGGER IF NOT EXISTS edges_attests_requires_human_root
         BEFORE INSERT ON edges
         WHEN NEW.edge_type = 'attests'
         BEGIN
             SELECT RAISE(ABORT, 'attests requires a human provenance root')
             WHERE NOT EXISTS (
                 SELECT 1 FROM provenance_roots
                  WHERE root_id = NEW.src_id
                    AND root_kind IN ('human_capture','human_edit_delta')
             );
         END",
    ),
    (
        "edges_attests_requires_human_root_update",
        "CREATE TRIGGER IF NOT EXISTS edges_attests_requires_human_root_update
         BEFORE UPDATE ON edges
         WHEN NEW.edge_type = 'attests'
         BEGIN
             SELECT RAISE(ABORT, 'attests requires a human provenance root')
             WHERE NOT EXISTS (
                 SELECT 1 FROM provenance_roots
                  WHERE root_id = NEW.src_id
                    AND root_kind IN ('human_capture','human_edit_delta')
             );
         END",
    ),
];

impl MemoryDB {
    /// The `CREATE` statements of every index and trigger attached to `table`,
    /// ordered so a replay is deterministic.
    ///
    /// Implicit indexes are skipped by `sql IS NOT NULL`: SQLite records a
    /// NULL `sql` for the index it derives from a `PRIMARY KEY` or `UNIQUE`
    /// declaration, and that index comes back with the `CREATE TABLE` rather
    /// than needing replay. Filtering on the NULL rather than on an
    /// `sqlite_autoindex%` name prefix keys off the property that actually
    /// makes them unreplayable — there is no statement to replay.
    pub(super) async fn capture_attached_objects(
        tx: &libsql::Transaction,
        table: &str,
    ) -> Result<Vec<String>, WenlanError> {
        let mut rows = tx
            .query(
                "SELECT sql FROM sqlite_master
                 WHERE tbl_name = ?1
                   AND type IN ('index','trigger')
                   AND sql IS NOT NULL
                 ORDER BY type, name",
                libsql::params![table],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("capture attached objects for {table}: {error}"))
            })?;

        let mut statements = Vec::new();
        while let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("read attached object for {table}: {error}"))
        })? {
            statements.push(row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("decode attached object for {table}: {error}"))
            })?);
        }
        Ok(statements)
    }

    /// Replay captured `CREATE` statements against the rebuilt table.
    ///
    /// Fails loud on the first statement that does not apply. A rebuild that
    /// swallowed a failure here would leave a widened table with part of its
    /// fence missing, which is the "after rename, before triggers" window that
    /// artifact 3 §8 says must refuse to serve rather than accept writes.
    pub(super) async fn replay_attached_objects(
        tx: &libsql::Transaction,
        statements: &[String],
    ) -> Result<(), WenlanError> {
        for statement in statements {
            tx.execute(statement, ()).await.map_err(|error| {
                WenlanError::VectorDb(format!("replay attached object [{statement}]: {error}"))
            })?;
        }
        Ok(())
    }

    /// The names of every index and trigger attached to `table`, sorted.
    ///
    /// The post-rebuild assertion compares NAMES, not statements: two fence
    /// bodies change deliberately, so statement equality would fail on the
    /// intended edit and teach the next reader to loosen the check.
    async fn attached_object_names(
        tx: &libsql::Transaction,
        table: &str,
    ) -> Result<Vec<String>, WenlanError> {
        let mut rows = tx
            .query(
                "SELECT name FROM sqlite_master
                 WHERE tbl_name = ?1
                   AND type IN ('index','trigger')
                   AND sql IS NOT NULL
                 ORDER BY name",
                libsql::params![table],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("attached object names for {table}: {error}"))
            })?;
        let mut names = Vec::new();
        while let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("read attached object name for {table}: {error}"))
        })? {
            names.push(row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("decode attached object name for {table}: {error}"))
            })?);
        }
        Ok(names)
    }

    /// Rebuild `edges` with the M5 widened CHECKs, extended fence, and the four
    /// support/attestation indexes (artifact 3 §7).
    ///
    /// Runs inside the caller's transaction. The caller is responsible for
    /// `PRAGMA foreign_keys = OFF` around it and `PRAGMA foreign_key_check`
    /// before committing — `edges.superseded_by` self-references `edges`, so
    /// the drop is only legal with enforcement suspended.
    ///
    /// Row-for-row and byte-for-byte: no row is added, dropped, or rewritten.
    /// A pre-M5 `supports` row would abort the copy on the new shape CHECK,
    /// which is the fail-closed halt artifact 3 §2 asks for — a `supports` edge
    /// of unknown provenance is exactly the shape M5 reads as truth, so it is
    /// never waved through.
    pub(super) async fn rebuild_edges_widened(tx: &libsql::Transaction) -> Result<(), WenlanError> {
        let captured = Self::capture_attached_objects(tx, "edges").await?;
        // A set, not a list: on a re-run the four M5 indexes are already
        // attached, so they arrive through the census as well as through the
        // loop below, and a list would double-count them.
        let mut expected: std::collections::BTreeSet<String> =
            Self::attached_object_names(tx, "edges")
                .await?
                .into_iter()
                .collect();

        tx.execute(EDGES_WIDENED_DDL, ())
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m97 create edges_new: {error}")))?;

        tx.execute(
            &format!(
                "INSERT INTO edges_new ({EDGE_COLUMNS})
                 SELECT {EDGE_COLUMNS} FROM edges ORDER BY edge_id"
            ),
            (),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m97 copy edges: {error}")))?;

        Self::assert_copy_is_faithful(tx).await?;

        // From here to the last CREATE below, `edges` is absent or unfenced.
        // The whole sequence is inside one transaction precisely so no other
        // connection ever observes that window (artifact 3 §8).
        tx.execute("DROP TABLE edges", ())
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m97 drop edges: {error}")))?;
        Self::rename_into_place(tx).await?;

        Self::replay_attached_objects(tx, &captured).await?;

        // The captured fence is the OLD one: it resolves an unknown kind to
        // NULL and `IS NOT` is null-safe, so it aborts every claim_revision and
        // root endpoint. Replaying it restores parity; these two statements
        // then widen it. Replace-after-replay rather than filtering the capture,
        // so the capture stays a census with no special cases in it.
        tx.execute_batch(&format!(
            "DROP TRIGGER edges_space_fence;
             DROP TRIGGER edges_space_fence_update;
             CREATE TRIGGER edges_space_fence
             AFTER INSERT ON edges
             WHEN NEW.lineage != 'legacy'
             BEGIN{FENCE_BODY}END;
             CREATE TRIGGER edges_space_fence_update
             AFTER UPDATE ON edges
             WHEN NEW.lineage != 'legacy' AND NEW.valid_until IS NULL
             BEGIN{FENCE_BODY}END;"
        ))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m97 widen space fence: {error}")))?;

        for (name, statement) in M5_INDEXES {
            tx.execute(statement, ()).await.map_err(|error| {
                WenlanError::VectorDb(format!("m97 create M5 index {name}: {error}"))
            })?;
            expected.insert(name.to_string());
        }

        for (name, statement) in M5_TRIGGERS {
            tx.execute(statement, ()).await.map_err(|error| {
                WenlanError::VectorDb(format!("m97 create M5 trigger {name}: {error}"))
            })?;
            expected.insert(name.to_string());
        }

        let actual: std::collections::BTreeSet<String> = Self::attached_object_names(tx, "edges")
            .await?
            .into_iter()
            .collect();
        if actual != expected {
            return Err(WenlanError::VectorDb(format!(
                "m97 attached-object set changed across the rebuild: expected {expected:?}, \
                 found {actual:?}"
            )));
        }
        Ok(())
    }

    /// `ALTER TABLE edges_new RENAME TO edges`, with the schema-wide reparse
    /// suppressed.
    ///
    /// Measured, not assumed: without the pragma this statement fails with
    /// ``error in trigger m4_page_community_memory_insert_invalidate: no such
    /// table: main.edges``. That trigger, and its delete twin (`db.rs:11162`,
    /// `db.rs:11180`), are attached to **`memories`**, not to `edges`, so
    /// `DROP TABLE edges` does not take them — they survive the drop naming a
    /// table that, for the length of one statement, does not exist. Since
    /// SQLite 3.25 a rename reparses every trigger and view in the schema in
    /// order to rewrite references to the renamed object, and that reparse
    /// resolves table names, so it trips over them.
    ///
    /// `legacy_alter_table` is the documented way to ask for the pre-3.25
    /// behavior: rename the entry, rewrite nothing. Nothing references
    /// `edges_new` — it was created moments earlier inside this transaction —
    /// so there is no reference for the reparse to fix, and suppressing it
    /// costs nothing.
    ///
    /// Dropping the two `memories` triggers and recreating them would also
    /// work, and is rejected: it is a hand-maintained list of everything that
    /// happens to mention `edges` today, which is the same defect the
    /// attached-object census exists to remove. This way a future trigger
    /// referencing `edges` needs no change here.
    ///
    /// Renaming the OLD table out of the way instead would be actively unsafe.
    /// The reparse is not merely an obstacle — it *rewrites bodies*, so
    /// `edges` → `edges_old` would silently repoint both `memories` triggers at
    /// `edges_old` and leave them there after the new table took the name.
    ///
    /// The pragma is connection state rather than transaction state, so it is
    /// restored on the failure path too.
    async fn rename_into_place(tx: &libsql::Transaction) -> Result<(), WenlanError> {
        tx.execute("PRAGMA legacy_alter_table = ON", ())
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("m97 legacy_alter_table on: {error}"))
            })?;
        let renamed = tx
            .execute("ALTER TABLE edges_new RENAME TO edges", ())
            .await;
        tx.execute("PRAGMA legacy_alter_table = OFF", ())
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("m97 legacy_alter_table off: {error}"))
            })?;
        renamed.map_err(|error| WenlanError::VectorDb(format!("m97 rename edges_new: {error}")))?;
        Ok(())
    }

    /// Both tables hold exactly the same rows, checked by set difference in
    /// both directions rather than by a digest.
    ///
    /// `EXCEPT` is exact where a checksum is merely unlikely to collide, and it
    /// names the offending row instead of reporting that some byte moved. It
    /// materializes both sides, so on a very large `edges` this is the
    /// migration's memory ceiling; batching the comparison by `edge_id` range
    /// is the upgrade path if that ever binds.
    pub(super) async fn assert_copy_is_faithful(
        tx: &libsql::Transaction,
    ) -> Result<(), WenlanError> {
        for (from, to) in [("edges", "edges_new"), ("edges_new", "edges")] {
            let mut rows = tx
                .query(
                    &format!(
                        "SELECT count(*) FROM (
                             SELECT {EDGE_COLUMNS} FROM {from}
                             EXCEPT
                             SELECT {EDGE_COLUMNS} FROM {to}
                         )"
                    ),
                    (),
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("m97 verify {from} minus {to}: {error}"))
                })?;
            let missing: i64 = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("m97 verify read: {error}")))?
                .ok_or_else(|| WenlanError::VectorDb("m97 verify returned no row".into()))?
                .get(0)
                .map_err(|error| WenlanError::VectorDb(format!("m97 verify decode: {error}")))?;
            if missing != 0 {
                return Err(WenlanError::VectorDb(format!(
                    "m97 edges copy is not faithful: {missing} row(s) in {from} absent from {to}"
                )));
            }
        }
        Ok(())
    }
}
