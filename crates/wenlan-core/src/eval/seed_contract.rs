// SPDX-License-Identifier: Apache-2.0
//! Seed-completeness contract (Fowler `ContractTest`).
//!
//! A cached eval "seed" DB is only trustworthy if it is (a) free of duplicate
//! memory rows, (b) fully enriched (every memory classified), and (c) the
//! fixture it claims to be. Historically none of this was checked — only
//! schema-migration replay was.
//!
//! Cautionary note (2026-06-08): an eyeballed analysis of the `lme_v1` seed
//! mis-read multi-chunk storage as duplication (counting `chunk_index>0`
//! continuation rows of long memories) and the NULL importance on those
//! continuation rows as "half-classified". This contract — scoped to memory
//! heads (`chunk_index=0`) — corrected that on its first run: `lme_v1` actually
//! has zero `(source, source_id)` duplicates and is essentially fully
//! classified. That is the whole point of putting the rule in code: a runnable
//! check beats prose. The real risks it still guards: true `(source, source_id)`
//! dupes, incomplete classification, and fixture-identity confusion (the
//! existing `lme_v1` seed is the ORACLE fixture, not the LME-S haystack).
//!
//! This module makes the integrity rule **code, not prose**: a pure-SQL check
//! that runs against any seed DB and fails LOUD on a violation. The cheap
//! pure-SQL nature means it runs in CI (no GPU, no embedder), so a dirty seed
//! fails the build independent of any agent or runbook discipline.
//!
//! It is the gate inside the (forthcoming) atomic seed-build pipeline: build to
//! a temp dir → `assert_seed_contract` → only then atomic-publish to canonical.
//! A broken seed can never become canonical because the gate sits before the swap.

use crate::error::WenlanError;

/// What a given seed variant promises. Thresholds are intentionally explicit so
/// the contract is auditable and so a lenient cue floor (per-type cue coverage
/// is not 100% by nature) does not become a flaky blocker.
#[derive(Debug, Clone)]
pub struct SeedExpectations {
    /// Human label for messages and the manifest, e.g. `"lme_oracle"` / `"lme_s"`.
    pub variant: String,
    /// Require zero duplicate `(source, source_id)` memory rows.
    pub require_no_dupes: bool,
    /// Require 100% of memory rows to be classified (`importance IS NOT NULL`).
    pub require_full_classification: bool,
    /// Minimum fraction of memory rows with a non-empty `retrieval_cue`.
    /// `0.0` = report-only (does not fail) — the council-recommended lenient
    /// default until the per-type cue distribution is established.
    pub min_cue_coverage: f64,
    /// Minimum fraction with a non-null `event_date`. `0.0` = report-only.
    pub min_event_date_coverage: f64,
    /// Require at least one `memory_entities` link to exist (graph-stream
    /// substrate is non-empty). `false` = report-only. This is a *presence*
    /// check, not a percentage: a coverage floor rots (AGENTS.md), but zero
    /// links means the graph channel is dead, which is the recurring bug a
    /// re-seed must fail loud on. Set by the seed orchestrator after it runs
    /// the entity-linking step.
    pub require_graph_links: bool,
    /// Require at least one non-null `event_date` (temporal substrate is
    /// non-empty). `false` = report-only. Presence check, same rationale as
    /// `require_graph_links`: LME turn text carries no dates, so a re-seed that
    /// forgot the `event_date` injection ships the temporal channel starved.
    pub require_event_dates: bool,
    /// Require at least one `status = 'active'` row in `pages` (page-channel
    /// substrate is non-empty). `false` = report-only. Presence check, same
    /// rationale as the other floors: the 2026-06-09 re-seed's distill step
    /// produced ~4 pages that did not persist, so the page channel shipped
    /// inert (pages=0) and every page-channel A/B measured nothing — exactly
    /// the dead-channel lie this contract exists to fail loud on.
    pub require_pages: bool,
    /// If set, the seed's recorded manifest fixture sha256 must equal this.
    /// `None` skips the identity check (manifest stamping lands with C4b).
    pub expect_fixture_sha256: Option<String>,
}

impl SeedExpectations {
    /// Strict profile: no dupes, fully classified, cue/date report-only, no
    /// manifest assertion yet. The floor every clean seed must clear.
    pub fn strict(variant: impl Into<String>) -> Self {
        Self {
            variant: variant.into(),
            require_no_dupes: true,
            require_full_classification: true,
            min_cue_coverage: 0.0,
            min_event_date_coverage: 0.0,
            require_graph_links: false,
            require_event_dates: false,
            require_pages: false,
            expect_fixture_sha256: None,
        }
    }

    /// Substrate-liveness profile: no-dupes PLUS the graph + temporal + page
    /// substrate presence checks. This is what the seed orchestrator asserts
    /// after running every enrichment step, and what an eval runner asserts
    /// before measuring a channel — so a starved substrate fails the SEED or is
    /// refused by the EVAL, never silently reported as "the channel doesn't help".
    ///
    /// `require_full_classification` is deliberately OFF here: `complete()` gates
    /// the channels that ship at *zero* (graph/temporal — the recurring lie),
    /// which are presence checks that never rot. Classification is a near-100%
    /// *coverage* concern, not a starved-channel one — a single trivial turn
    /// ("give me 6 more") legitimately yields no importance, so demanding exactly
    /// 100% would block substrate verification on noise. Classification coverage
    /// is reported (and validated separately by `strict()` + the real-data test,
    /// which accepts `>0.99`).
    pub fn complete(variant: impl Into<String>) -> Self {
        Self {
            require_full_classification: false,
            require_graph_links: true,
            require_event_dates: true,
            require_pages: true,
            ..Self::strict(variant)
        }
    }
}

/// Measured facts about a seed DB plus any contract violations found.
#[derive(Debug, Clone)]
pub struct SeedContractReport {
    /// Memory rows in scope (`source='memory' AND chunk_index=0 AND is_recap=0`).
    pub rows: i64,
    /// Distinct `(source, source_id)` pairs in scope. Equals `rows` iff no dupes.
    pub distinct_keys: i64,
    /// Rows with `importance IS NOT NULL`.
    pub classified: i64,
    /// Rows with a non-empty `retrieval_cue`.
    pub cue_nonempty: i64,
    /// Rows with a non-null `event_date`.
    pub event_date_nonempty: i64,
    /// Total `memory_entities` link rows (graph-stream substrate). `0` means the
    /// graph channel has nothing to surface. Tolerant of the table being absent
    /// on pre-graph seeds (reported as `0`).
    pub graph_links: i64,
    /// Active pages (`status = 'active'`, matching `MemoryDB::count_active_pages`).
    /// `0` means the page channel has nothing to surface. Tolerant of the table
    /// being absent on pre-pages seeds (reported as `0`).
    pub pages: i64,
    /// Recorded manifest fixture sha256, if the seed stamped one.
    pub manifest_fixture_sha256: Option<String>,
    /// One human-readable string per failed expectation. Empty = contract holds.
    pub violations: Vec<String>,
}

impl SeedContractReport {
    pub fn holds(&self) -> bool {
        self.violations.is_empty()
    }
    pub fn cue_coverage(&self) -> f64 {
        if self.rows == 0 {
            0.0
        } else {
            self.cue_nonempty as f64 / self.rows as f64
        }
    }
    pub fn event_date_coverage(&self) -> f64 {
        if self.rows == 0 {
            0.0
        } else {
            self.event_date_nonempty as f64 / self.rows as f64
        }
    }
}

/// Scope shared by every count: one head-row per memory, matching the scope of
/// `get_memories_needing_classification` so the classification ratio is exact.
const SCOPE: &str = "source = 'memory' AND chunk_index = 0 AND is_recap = 0";

async fn count(conn: &libsql::Connection, where_extra: &str) -> Result<i64, WenlanError> {
    let sql = format!("SELECT COUNT(*) FROM memories WHERE {SCOPE}{where_extra}");
    let mut rows = conn
        .query(&sql, ())
        .await
        .map_err(|e| WenlanError::VectorDb(format!("seed_contract count: {e}")))?;
    let row = rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("seed_contract next: {e}")))?
        .ok_or_else(|| WenlanError::Generic("seed_contract: empty COUNT result".into()))?;
    row.get::<i64>(0)
        .map_err(|e| WenlanError::VectorDb(format!("seed_contract get: {e}")))
}

/// Measure a seed DB against its expectations. Pure SQL — no embedder, no GPU.
/// `conn` is any libSQL connection to the seed (open read-only with
/// `libsql::Builder::new_local(path)` for the CI/standalone case).
pub async fn check_seed_contract(
    conn: &libsql::Connection,
    expect: &SeedExpectations,
) -> Result<SeedContractReport, WenlanError> {
    let rows = count(conn, "").await?;

    // distinct (source, source_id) — the CORRECT dedup key (idempotency is on
    // the tuple, not source_id alone). A subquery keeps it one round-trip.
    let distinct_keys = {
        let sql = format!(
            "SELECT COUNT(*) FROM (SELECT 1 FROM memories WHERE {SCOPE} \
             GROUP BY source, source_id)"
        );
        let mut r = conn
            .query(&sql, ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("seed_contract distinct: {e}")))?;
        r.next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("seed_contract distinct next: {e}")))?
            .ok_or_else(|| WenlanError::Generic("seed_contract: empty distinct result".into()))?
            .get::<i64>(0)
            .map_err(|e| WenlanError::VectorDb(format!("seed_contract distinct get: {e}")))?
    };

    let classified = count(conn, " AND importance IS NOT NULL").await?;
    let cue_nonempty = count(
        conn,
        " AND retrieval_cue IS NOT NULL AND retrieval_cue <> ''",
    )
    .await?;
    let event_date_nonempty = count(conn, " AND event_date IS NOT NULL").await?;
    let graph_links = count_memory_entities(conn).await?;
    let pages = count_active_pages(conn).await?;

    let manifest_fixture_sha256 = read_manifest_value(conn, "seed_fixture_sha256").await?;

    let mut violations = Vec::new();

    if expect.require_no_dupes && rows != distinct_keys {
        violations.push(format!(
            "duplicate rows: {rows} rows but {distinct_keys} distinct (source, source_id) \
             ({} extra copies)",
            rows - distinct_keys
        ));
    }
    if expect.require_full_classification && classified != rows {
        violations.push(format!(
            "incomplete classification: {classified}/{rows} rows have importance ({} unclassified)",
            rows - classified
        ));
    }
    let cue_cov = if rows == 0 {
        0.0
    } else {
        cue_nonempty as f64 / rows as f64
    };
    if expect.min_cue_coverage > 0.0 && cue_cov < expect.min_cue_coverage {
        violations.push(format!(
            "retrieval_cue coverage {:.1}% below floor {:.1}%",
            cue_cov * 100.0,
            expect.min_cue_coverage * 100.0
        ));
    }
    let date_cov = if rows == 0 {
        0.0
    } else {
        event_date_nonempty as f64 / rows as f64
    };
    if expect.min_event_date_coverage > 0.0 && date_cov < expect.min_event_date_coverage {
        violations.push(format!(
            "event_date coverage {:.1}% below floor {:.1}%",
            date_cov * 100.0,
            expect.min_event_date_coverage * 100.0
        ));
    }
    // Presence checks (no percentage → no rot): a dead channel is exactly zero.
    if expect.require_graph_links && graph_links == 0 {
        violations.push(
            "graph substrate empty: 0 memory_entities links (graph channel is dead — \
             run the entity-linking seed step before measuring graph)"
                .to_string(),
        );
    }
    if expect.require_event_dates && event_date_nonempty == 0 {
        violations.push(
            "temporal substrate empty: 0 event_date rows (temporal channel is dead — \
             run the event_date injection seed step before measuring temporal)"
                .to_string(),
        );
    }
    if expect.require_pages && pages == 0 {
        violations.push(
            "page substrate empty: 0 active pages (page channel is dead — \
             run the distill seed step and verify it persists before measuring pages)"
                .to_string(),
        );
    }
    if let Some(want) = &expect.expect_fixture_sha256 {
        match &manifest_fixture_sha256 {
            Some(got) if got == want => {}
            Some(got) => violations.push(format!(
                "fixture mismatch: manifest sha256 {got} != expected {want}"
            )),
            None => violations.push(format!(
                "missing manifest: expected fixture sha256 {want}, seed has none"
            )),
        }
    }

    Ok(SeedContractReport {
        rows,
        distinct_keys,
        classified,
        cue_nonempty,
        event_date_nonempty,
        graph_links,
        pages,
        manifest_fixture_sha256,
        violations,
    })
}

/// Count `memory_entities` link rows. Tolerant of the table being absent
/// (pre-graph seeds) — returns `0` rather than erroring, mirroring
/// `read_manifest_value`'s table-missing handling.
async fn count_memory_entities(conn: &libsql::Connection) -> Result<i64, WenlanError> {
    let mut rows = match conn.query("SELECT COUNT(*) FROM memory_entities", ()).await {
        Ok(r) => r,
        Err(_) => return Ok(0), // table missing on pre-graph seeds
    };
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("memory_entities next: {e}")))?
    {
        Some(row) => row
            .get::<i64>(0)
            .map_err(|e| WenlanError::VectorDb(format!("memory_entities get: {e}"))),
        None => Ok(0),
    }
}

/// Count `status = 'active'` rows in `pages` (page-channel substrate). The
/// WHERE clause mirrors `MemoryDB::count_active_pages` so the contract and the
/// runner-side sanity gate agree on what counts as a live page. Tolerant of
/// the table being absent (pre-pages seeds) — returns `0` rather than erroring.
async fn count_active_pages(conn: &libsql::Connection) -> Result<i64, WenlanError> {
    let mut rows = match conn
        .query(
            // G6: `kind='entity'` rows in `pages` are entity shadow pages, not
            // distilled pages. Counting them makes this liveness gate report a
            // live page channel on a store with zero distilled pages — the
            // exact "measuring against a dead substrate" lie the contract
            // exists to prevent. Same fence as lint's `load_pages` (G6 Stage 2
            // PR 2c item 6). Defect predates Stage 3: entity shadow pages have
            // existed since Stage 2 made the entity side canonical.
            "SELECT COUNT(*) FROM pages \
             WHERE status = 'active' AND COALESCE(kind, 'concept') <> 'entity'",
            (),
        )
        .await
    {
        Ok(r) => r,
        Err(_) => return Ok(0), // table missing on pre-pages seeds
    };
    match rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("pages count next: {e}")))?
    {
        Some(row) => row
            .get::<i64>(0)
            .map_err(|e| WenlanError::VectorDb(format!("pages count get: {e}"))),
        None => Ok(0),
    }
}

/// Fail LOUD if the seed violates its contract. This is the pipeline gate and
/// the CI assertion. Returns the report on success for logging.
pub async fn assert_seed_contract(
    conn: &libsql::Connection,
    expect: &SeedExpectations,
) -> Result<SeedContractReport, WenlanError> {
    let report = check_seed_contract(conn, expect).await?;
    if !report.holds() {
        return Err(WenlanError::Generic(format!(
            "seed contract [{}] VIOLATED ({} rows): {}",
            expect.variant,
            report.rows,
            report.violations.join("; ")
        )));
    }
    Ok(report)
}

/// Eval-side no-drift gate. Refuse to measure a channel whose substrate is
/// empty: a graph/temporal A/B over a starved DB produces an uninterpretable
/// null that gets misread as "the channel doesn't help". The eval runner calls
/// this at entry; the seed orchestrator asserts the producing side
/// (`SeedExpectations::complete`). Same contract at both ends — no drift, no lie.
///
/// Feature keys match by substring so the many A/B labels map without an enum:
/// any feature containing `graph` requires `memory_entities`; any containing
/// `temp` requires `event_date`; any containing `page` requires active `pages`.
/// A feature the gate doesn't model imposes no requirement (returns `Ok`) —
/// the gate never blocks a channel it can't check.
pub async fn assert_feature_substrate_live(
    conn: &libsql::Connection,
    feature: &str,
) -> Result<(), WenlanError> {
    let f = feature.to_ascii_lowercase();
    if f.contains("graph") && count_memory_entities(conn).await? == 0 {
        return Err(WenlanError::Generic(format!(
            "EVAL REFUSED [{feature}]: graph substrate empty (0 memory_entities links). \
             A graph A/B here measures noise, not graph. Re-seed via \
             seed_scenario_dbs_complete before measuring."
        )));
    }
    if f.contains("temp") && count(conn, " AND event_date IS NOT NULL").await? == 0 {
        return Err(WenlanError::Generic(format!(
            "EVAL REFUSED [{feature}]: temporal substrate empty (0 event_date rows). \
             A temporal A/B here measures noise. Re-seed via \
             seed_scenario_dbs_complete before measuring."
        )));
    }
    if f.contains("page") && count_active_pages(conn).await? == 0 {
        return Err(WenlanError::Generic(format!(
            "EVAL REFUSED [{feature}]: page substrate empty (0 active pages). \
             A page-channel A/B here measures noise, not pages. Re-seed via \
             seed_scenario_dbs_complete (with a persisting distill step) before measuring."
        )));
    }
    Ok(())
}

/// Read a manifest value from `app_metadata`. Returns `Ok(None)` if the key is
/// absent or the table does not exist (older seeds predate the manifest).
async fn read_manifest_value(
    conn: &libsql::Connection,
    key: &str,
) -> Result<Option<String>, WenlanError> {
    let mut rows = match conn
        .query(
            "SELECT value FROM app_metadata WHERE key = ?1",
            libsql::params![key],
        )
        .await
    {
        Ok(r) => r,
        Err(_) => return Ok(None), // table missing on pre-manifest seeds
    };
    let next = rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("manifest read: {e}")))?;
    match next {
        Some(row) => Ok(row.get::<String>(0).ok()),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal in-memory seed table. CI-runnable: no embedder, no GPU, no files.
    async fn mem_conn() -> libsql::Connection {
        let db = libsql::Builder::new_local(":memory:")
            .build()
            .await
            .expect("open in-memory libsql");
        let conn = db.connect().expect("connect in-memory libsql");
        conn.execute_batch(
            "CREATE TABLE memories (
                id TEXT, source TEXT, source_id TEXT, chunk_index INTEGER DEFAULT 0,
                is_recap INTEGER DEFAULT 0, importance REAL,
                retrieval_cue TEXT, event_date INTEGER, content TEXT
             );
             CREATE TABLE memory_entities (memory_id TEXT, entity_id TEXT);
             -- `kind` mirrors the real `pages` schema. It is not decoration:
             -- `count_active_pages` fences out `kind='entity'` shadow pages,
             -- and a fixture table without the column makes that query fail to
             -- compile, which the counter's table-missing tolerance then
             -- reports as zero active pages.
             CREATE TABLE pages (id TEXT, status TEXT, kind TEXT);
             CREATE TABLE app_metadata (key TEXT PRIMARY KEY, value TEXT);",
        )
        .await
        .expect("create schema");
        conn
    }

    async fn insert(
        conn: &libsql::Connection,
        sid: &str,
        importance: Option<f64>,
        cue: Option<&str>,
    ) {
        conn.execute(
            "INSERT INTO memories (id, source, source_id, chunk_index, is_recap, importance, retrieval_cue, event_date)
             VALUES (?1, 'memory', ?1, 0, 0, ?2, ?3, NULL)",
            libsql::params![sid, importance, cue],
        )
        .await
        .expect("insert row");
    }

    /// Set `event_date` for a memory head (temporal substrate).
    async fn set_event_date(conn: &libsql::Connection, sid: &str, date: i64) {
        conn.execute(
            "UPDATE memories SET event_date = ?2 WHERE source_id = ?1",
            libsql::params![sid, date],
        )
        .await
        .expect("set event_date");
    }

    /// Link a memory head to an entity (graph substrate).
    async fn link_entity(conn: &libsql::Connection, sid: &str, entity: &str) {
        conn.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?1, ?2)",
            libsql::params![sid, entity],
        )
        .await
        .expect("link entity");
    }

    /// Insert a page row (page-channel substrate).
    /// A distilled page, the substrate the page channel is measured over.
    /// `kind` is left NULL, which is what a page predating the column carries;
    /// `count_active_pages` folds that to `'concept'`.
    async fn insert_page(conn: &libsql::Connection, pid: &str, status: &str) {
        conn.execute(
            "INSERT INTO pages (id, status) VALUES (?1, ?2)",
            libsql::params![pid, status],
        )
        .await
        .expect("insert page");
    }

    /// An entity's `kind='entity'` shadow page. Since G6 Stage 2 made the
    /// entity side canonical these share the `pages` table with distilled
    /// pages, and they are exactly what the page channel is NOT measured over.
    async fn insert_entity_shadow_page(conn: &libsql::Connection, pid: &str, status: &str) {
        conn.execute(
            "INSERT INTO pages (id, status, kind) VALUES (?1, ?2, 'entity')",
            libsql::params![pid, status],
        )
        .await
        .expect("insert entity shadow page");
    }

    #[tokio::test]
    async fn clean_seed_passes() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue a")).await;
        insert(&conn, "b", Some(0.5), Some("cue b")).await;
        let r = assert_seed_contract(&conn, &SeedExpectations::strict("test"))
            .await
            .expect("clean seed should pass");
        assert_eq!(r.rows, 2);
        assert_eq!(r.distinct_keys, 2);
        assert!(r.holds());
    }

    #[tokio::test]
    async fn duplicate_rows_fail() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        insert(&conn, "a", Some(0.5), Some("cue")).await; // same (source, source_id)
        let r = check_seed_contract(&conn, &SeedExpectations::strict("test"))
            .await
            .unwrap();
        assert_eq!(r.rows, 2);
        assert_eq!(r.distinct_keys, 1);
        assert!(!r.holds(), "duplicate (source,source_id) must violate");
        assert!(r.violations.iter().any(|v| v.contains("duplicate")));
    }

    #[tokio::test]
    async fn partial_classification_fails() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        insert(&conn, "b", None, None).await; // unclassified tail row
        let r = check_seed_contract(&conn, &SeedExpectations::strict("test"))
            .await
            .unwrap();
        assert_eq!(r.classified, 1);
        assert_eq!(r.rows, 2);
        assert!(!r.holds());
        assert!(r.violations.iter().any(|v| v.contains("classification")));
    }

    #[tokio::test]
    async fn manifest_mismatch_fails() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        conn.execute(
            "INSERT INTO app_metadata (key, value) VALUES ('seed_fixture_sha256', 'deadbeef')",
            (),
        )
        .await
        .unwrap();
        let mut expect = SeedExpectations::strict("test");
        expect.expect_fixture_sha256 = Some("cafebabe".to_string());
        let r = check_seed_contract(&conn, &expect).await.unwrap();
        assert_eq!(r.manifest_fixture_sha256.as_deref(), Some("deadbeef"));
        assert!(r.violations.iter().any(|v| v.contains("fixture mismatch")));
    }

    #[tokio::test]
    async fn cue_floor_is_lenient_by_default() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), None).await; // no cue
        insert(&conn, "b", Some(0.5), None).await; // no cue
                                                   // strict() leaves min_cue_coverage = 0.0 → report-only, must still pass.
        let r = check_seed_contract(&conn, &SeedExpectations::strict("test"))
            .await
            .unwrap();
        assert_eq!(r.cue_nonempty, 0);
        assert!(
            r.holds(),
            "cue floor 0.0 must not fail a fully-classified seed"
        );
    }

    #[tokio::test]
    async fn complete_profile_fails_on_empty_graph_substrate() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        set_event_date(&conn, "a", 1_700_000_000).await; // temporal OK
                                                         // no memory_entities links → graph dead
        let r = check_seed_contract(&conn, &SeedExpectations::complete("test"))
            .await
            .unwrap();
        assert_eq!(r.graph_links, 0);
        assert!(
            !r.holds(),
            "complete() must fail when graph substrate empty"
        );
        assert!(
            r.violations
                .iter()
                .any(|v| v.contains("graph substrate empty")),
            "expected graph-substrate violation, got {:?}",
            r.violations
        );
    }

    #[tokio::test]
    async fn complete_profile_fails_on_empty_temporal_substrate() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        link_entity(&conn, "a", "ent_1").await; // graph OK
                                                // no event_date → temporal dead
        let r = check_seed_contract(&conn, &SeedExpectations::complete("test"))
            .await
            .unwrap();
        assert_eq!(r.event_date_nonempty, 0);
        assert!(
            !r.holds(),
            "complete() must fail when temporal substrate empty"
        );
        assert!(
            r.violations
                .iter()
                .any(|v| v.contains("temporal substrate empty")),
            "expected temporal-substrate violation, got {:?}",
            r.violations
        );
    }

    #[tokio::test]
    async fn complete_profile_passes_with_full_substrate() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue a")).await;
        insert(&conn, "b", Some(0.5), Some("cue b")).await;
        set_event_date(&conn, "a", 1_700_000_000).await;
        link_entity(&conn, "a", "ent_1").await;
        insert_page(&conn, "p1", "active").await;
        let r = assert_seed_contract(&conn, &SeedExpectations::complete("test"))
            .await
            .expect("full substrate should pass complete()");
        assert!(r.graph_links >= 1);
        assert!(r.event_date_nonempty >= 1);
        assert!(r.pages >= 1);
        assert!(r.holds());
    }

    #[tokio::test]
    async fn complete_profile_fails_on_empty_page_substrate() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        set_event_date(&conn, "a", 1_700_000_000).await; // temporal OK
        link_entity(&conn, "a", "ent_1").await; // graph OK
                                                // no pages → page channel dead
        let r = check_seed_contract(&conn, &SeedExpectations::complete("test"))
            .await
            .unwrap();
        assert_eq!(r.pages, 0);
        assert!(!r.holds(), "complete() must fail when page substrate empty");
        assert!(
            r.violations
                .iter()
                .any(|v| v.contains("page substrate empty")),
            "expected page-substrate violation, got {:?}",
            r.violations
        );
    }

    #[tokio::test]
    async fn non_active_pages_do_not_satisfy_the_page_floor() {
        // An archived/draft page is not a live channel surface; only
        // status='active' counts (mirrors MemoryDB::count_active_pages).
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        set_event_date(&conn, "a", 1_700_000_000).await;
        link_entity(&conn, "a", "ent_1").await;
        insert_page(&conn, "p1", "archived").await;
        let r = check_seed_contract(&conn, &SeedExpectations::complete("test"))
            .await
            .unwrap();
        assert_eq!(r.pages, 0, "archived pages must not count");
        assert!(!r.holds());
        assert!(r
            .violations
            .iter()
            .any(|v| v.contains("page substrate empty")));
    }

    #[tokio::test]
    async fn strict_profile_ignores_graph_and_temporal_substrate() {
        // Regression guard: strict() must stay lenient on graph/temporal/pages
        // so the generic CI check + minimal seeds don't break. Only complete()
        // has teeth.
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        // no links, no event_date, no pages
        let r = check_seed_contract(&conn, &SeedExpectations::strict("test"))
            .await
            .unwrap();
        assert_eq!(r.graph_links, 0);
        assert_eq!(r.event_date_nonempty, 0);
        assert_eq!(r.pages, 0);
        assert!(
            r.holds(),
            "strict() must not fail on empty graph/temporal/page substrate"
        );
    }

    #[tokio::test]
    async fn eval_gate_refuses_graph_on_empty_substrate() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        set_event_date(&conn, "a", 1_700_000_000).await; // temporal fine
                                                         // no links → graph dead
        let err = assert_feature_substrate_live(&conn, "graph_stream")
            .await
            .expect_err("graph A/B on empty substrate must be refused");
        assert!(format!("{err}").contains("graph substrate empty"));
        // temporal feature must still pass (event_date present)
        assert_feature_substrate_live(&conn, "temporal")
            .await
            .expect("temporal substrate present → allowed");
    }

    #[tokio::test]
    async fn eval_gate_refuses_temporal_on_empty_substrate() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        link_entity(&conn, "a", "ent_1").await; // graph fine
                                                // no event_date → temporal dead
        let err = assert_feature_substrate_live(&conn, "expand_temp")
            .await
            .expect_err("temporal A/B on empty substrate must be refused");
        assert!(format!("{err}").contains("temporal substrate empty"));
        assert_feature_substrate_live(&conn, "graph")
            .await
            .expect("graph substrate present → allowed");
    }

    #[tokio::test]
    async fn eval_gate_refuses_page_on_empty_substrate() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        let err = assert_feature_substrate_live(&conn, "page_channel")
            .await
            .expect_err("page A/B on empty substrate must be refused");
        assert!(format!("{err}").contains("page substrate empty"));
        insert_page(&conn, "p1", "active").await;
        assert_feature_substrate_live(&conn, "page_channel")
            .await
            .expect("page substrate present → allowed");
    }

    /// The fence itself, which nothing else covers. A store whose only active
    /// `pages` rows are entity shadow pages has a DEAD page channel, and the
    /// contract exists to refuse exactly that rather than measure over it.
    ///
    /// Mutation control: drop `AND COALESCE(kind, 'concept') <> 'entity'` from
    /// `count_active_pages` and the first refusal below stops arriving.
    #[tokio::test]
    async fn entity_shadow_pages_are_not_page_channel_substrate() {
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), Some("cue")).await;
        insert_entity_shadow_page(&conn, "ent-page", "active").await;

        let err = assert_feature_substrate_live(&conn, "page_channel")
            .await
            .expect_err("a store with only entity shadow pages has no page channel");
        assert!(format!("{err}").contains("page substrate empty"));

        // The control: one genuine distilled page and the same call proceeds,
        // so the refusal above is the fence and not an unrelated failure.
        insert_page(&conn, "p1", "active").await;
        assert_feature_substrate_live(&conn, "page_channel")
            .await
            .expect("one distilled page revives the channel");
    }

    #[tokio::test]
    async fn eval_gate_allows_unmodeled_feature() {
        // A feature the gate doesn't model (no graph/temp/page substrate need)
        // must never be blocked, even on an otherwise-empty DB.
        let conn = mem_conn().await;
        insert(&conn, "a", Some(0.5), None).await;
        assert_feature_substrate_live(&conn, "session_diversity")
            .await
            .expect("unmodeled feature must pass");
        assert_feature_substrate_live(&conn, "magnitude_fusion")
            .await
            .expect("unmodeled feature must pass");
    }

    #[tokio::test]
    async fn graph_links_count_tolerates_missing_table() {
        // Pre-graph seeds have no memory_entities table; count must report 0,
        // not error. (Mirrors read_manifest_value's table-missing tolerance.)
        let db = libsql::Builder::new_local(":memory:")
            .build()
            .await
            .unwrap();
        let conn = db.connect().unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (
                id TEXT, source TEXT, source_id TEXT, chunk_index INTEGER DEFAULT 0,
                is_recap INTEGER DEFAULT 0, importance REAL,
                retrieval_cue TEXT, event_date INTEGER, content TEXT
             );",
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO memories (id, source, source_id, importance) VALUES ('a','memory','a',0.5)",
            (),
        )
        .await
        .unwrap();
        // strict() does not require links/pages → must pass despite both the
        // memory_entities and pages tables being absent from this schema.
        let r = check_seed_contract(&conn, &SeedExpectations::strict("test"))
            .await
            .expect("missing memory_entities/pages tables must not error");
        assert_eq!(r.graph_links, 0);
        assert_eq!(r.pages, 0);
        assert!(r.holds());
    }

    /// Characterize REAL data: scoped to memory heads, the `lme_v1` seed has
    /// zero `(source, source_id)` duplicates and is essentially fully classified
    /// (an earlier eyeballed analysis wrongly called it double-seeded +
    /// half-classified by counting multi-chunk continuation rows). Guards against
    /// a regression that would reintroduce real dupes. Pure SQL, no GPU. Resolves
    /// the seed like `cached_scenario_db_check.rs`:
    /// `SCENARIO_DB_ROOT` > `EVAL_BASELINES_DIR/scenario_seeded` > `~/.cache/origin-eval/scenario_seeded`.
    #[tokio::test]
    #[ignore = "needs cached scenario DB (scripts/seed-scenario-dbs.sh); pure SQL, no GPU"]
    async fn real_lme_v1_has_no_dupes_and_is_classified() {
        let root = std::env::var("SCENARIO_DB_ROOT")
            .map(std::path::PathBuf::from)
            .or_else(|_| {
                std::env::var("EVAL_BASELINES_DIR")
                    .map(|p| std::path::PathBuf::from(p).join("scenario_seeded"))
            })
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                std::path::PathBuf::from(home)
                    .join(".cache")
                    .join("origin-eval")
                    .join("scenario_seeded")
            });
        let db_file = root.join("lme_v1").join("origin_memory.db");
        if !db_file.exists() {
            eprintln!("SKIP: {} not found", db_file.display());
            return;
        }
        let db = libsql::Builder::new_local(db_file.to_str().unwrap())
            .build()
            .await
            .expect("open lme_v1");
        let conn = db.connect().expect("connect lme_v1");
        let r = check_seed_contract(&conn, &SeedExpectations::strict("lme_v1"))
            .await
            .expect("contract check");
        eprintln!(
            "[seed_contract] lme_v1: rows={} distinct={} classified={} cue={} ({:.1}%) event_date={} ({:.1}%)",
            r.rows,
            r.distinct_keys,
            r.classified,
            r.cue_nonempty,
            r.cue_coverage() * 100.0,
            r.event_date_nonempty,
            r.event_date_coverage() * 100.0,
        );
        eprintln!("[seed_contract] violations: {:#?}", r.violations);
        // CORRECTED 2026-06-08: lme_v1 is NOT double-seeded. At memory-head scope
        // there are zero duplicates; the contract caught the misdiagnosis. Real
        // residual issues are orthogonal to this assert: a stray unclassified row
        // and the fixture being ORACLE (not LME-S).
        assert_eq!(r.rows, r.distinct_keys, "no duplicate memory heads");
        assert!(
            !r.violations.iter().any(|v| v.contains("duplicate")),
            "must NOT report duplicates: {:?}",
            r.violations
        );
        assert!(
            r.classified as f64 / r.rows.max(1) as f64 > 0.99,
            "essentially fully classified (got {}/{})",
            r.classified,
            r.rows
        );
    }
}
