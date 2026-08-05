use super::provenance_checks::SOURCE_COVERAGE_ID;
use super::state_checks::VERSION_ALIGNMENT_ID;
use crate::lint::context::{
    AppliedScope, CancellationToken, ExecutionGate, LintClock, LintContext,
};
use crate::lint::pages::fs::scan_page_root;
use crate::lint::snapshot::{LintReadSnapshot, LintRows};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;
use wenlan_types::lint::{LintEvidenceRef, LintOpaqueId, LintOutcome, LINT_MAX_EVIDENCE_PER_CHECK};

const FIXTURE_ENV: &str = "WENLAN_LINT_SCALE_FIXTURE";
const PAGE_COUNT: usize = 10_000;
const EVIDENCE_COUNT: u64 = 100_000;

fn fixture_root() -> PathBuf {
    std::env::var_os(FIXTURE_ENV)
        .map(PathBuf::from)
        .expect("WENLAN_LINT_SCALE_FIXTURE must name the prebuilt fixture root")
}

#[tokio::test]
#[ignore = "fixture construction runs separately from the measured gate"]
async fn generate_diagnostic_scale_fixture() {
    generate_fixture(&fixture_root()).await;
}

#[tokio::test]
#[ignore = "run through scripts/lint-scale-gate.sh or the Windows CI step"]
async fn production_page_group_scale_gate() {
    run_fixture(&fixture_root()).await;
}

async fn generate_fixture(root: &Path) {
    assert!(!root.exists(), "scale fixture root must not already exist");
    let page_root = root.join("pages");
    std::fs::create_dir_all(page_root.join(".wenlan")).expect("create Page state directory");
    std::fs::create_dir_all(page_root.join("_sources")).expect("create Page source directory");
    std::fs::write(page_root.join("_sources/.manifest.json"), b"{\"pages\":{}}")
        .expect("write empty source manifest");

    let mut state = String::from("{\"schema_version\":2,\"pages\":{");
    for index in 1..=PAGE_COUNT {
        let separator = if index == 1 { "" } else { "," };
        write!(
            state,
            "{separator}\"page-{index:05}\":{{\"file\":\"page-{index:05}.md\",\"version\":1}}"
        )
        .expect("append deterministic state entry");
        let origin_version = if index == PAGE_COUNT { 2 } else { 1 };
        std::fs::write(
            page_root.join(format!("page-{index:05}.md")),
            format!(
                "---\norigin_id: page-{index:05}\norigin_version: {origin_version}\n---\nbody\n"
            ),
        )
        .expect("write deterministic Page fixture");
    }
    state.push_str("}}");
    std::fs::write(page_root.join(".wenlan/state.json"), state)
        .expect("write deterministic Page state");

    let database = libsql::Builder::new_local(root.join("lint-scale.db"))
        .build()
        .await
        .expect("open scale database");
    let connection = database.connect().expect("connect scale database");
    connection
        .execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE pages (
               id TEXT PRIMARY KEY,
               title TEXT NOT NULL,
               version INTEGER NOT NULL,
               status TEXT NOT NULL,
               creation_kind TEXT NOT NULL,
               review_status TEXT NOT NULL,
               workspace TEXT,
               space TEXT,
               source_memory_ids TEXT,
               citations TEXT
             );
             CREATE INDEX idx_pages_status ON pages(status);
             CREATE TABLE memories (
               id TEXT PRIMARY KEY,
               source TEXT,
               source_id TEXT,
               source_agent TEXT
             );
             CREATE INDEX idx_memories_source_id ON memories(source_id);
             CREATE TABLE page_sources (
               page_id TEXT NOT NULL,
               memory_source_id TEXT NOT NULL,
               linked_at INTEGER NOT NULL,
               link_reason TEXT,
               PRIMARY KEY (page_id, memory_source_id)
             );
             CREATE INDEX idx_page_sources_memory ON page_sources(memory_source_id);
             CREATE TABLE page_evidence (
               page_id TEXT NOT NULL,
               source_kind TEXT NOT NULL,
               locator TEXT,
               linked_at INTEGER NOT NULL,
               link_reason TEXT,
               PRIMARY KEY (page_id, source_kind, locator)
             );
             CREATE INDEX idx_page_evidence_locator ON page_evidence(locator)
               WHERE locator IS NOT NULL;
             CREATE TABLE page_links (
               source_page_id TEXT NOT NULL,
               target_page_id TEXT,
               label_key TEXT NOT NULL,
               label TEXT NOT NULL,
               PRIMARY KEY (source_page_id, label_key)
             );
             CREATE TABLE edges (
               edge_id TEXT PRIMARY KEY,
               src_id TEXT NOT NULL,
               src_kind TEXT NOT NULL CHECK(src_kind IN ('page','memory','entity','external')),
               dst_id TEXT NOT NULL,
               dst_kind TEXT NOT NULL CHECK(dst_kind IN ('page','memory','entity','external')),
               edge_type TEXT NOT NULL CHECK(edge_type IN ('mentions','relates','cites','supports','links')),
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
               semantic_type TEXT
             );
             CREATE INDEX idx_edges_src ON edges(src_kind, src_id);
             CREATE INDEX idx_edges_dst ON edges(dst_kind, dst_id);
             CREATE INDEX idx_edges_root ON edges(root_id) WHERE root_id IS NOT NULL;
             CREATE INDEX idx_edges_operation ON edges(operation_id) WHERE operation_id IS NOT NULL;
             CREATE INDEX idx_edges_superseded ON edges(superseded_by) WHERE superseded_by IS NOT NULL;
             CREATE INDEX idx_edges_active_grounded_space_type
               ON edges(space, edge_type) WHERE valid_until IS NULL AND grounded = 1;
             CREATE INDEX idx_page_links_target ON page_links(target_page_id)
               WHERE target_page_id IS NOT NULL;
             CREATE INDEX idx_page_links_orphan ON page_links(label_key)
               WHERE target_page_id IS NULL;",
        )
        .await
        .expect("create production-shaped lint tables");
    connection
        .execute_batch(
            "WITH RECURSIVE n(x) AS (
               VALUES(1) UNION ALL SELECT x + 1 FROM n WHERE x < 10000
             )
             INSERT INTO pages
               (id, title, version, status, creation_kind, review_status, workspace, citations)
             SELECT printf('page-%05d', x), printf('Page %05d', x), 1, 'active',
                    'distilled', 'confirmed', NULL, NULL
               FROM n;
             WITH RECURSIVE n(x) AS (
               VALUES(1) UNION ALL SELECT x + 1 FROM n WHERE x < 100000
             )
             INSERT INTO memories (id, source, source_id, source_agent)
             SELECT printf('memory-%06d', x), 'text',
                    printf('source-%05d-%02d', ((x - 1) / 10) + 1, ((x - 1) % 10) + 1),
                    NULL
               FROM n;
             WITH RECURSIVE n(x) AS (
               VALUES(1) UNION ALL SELECT x + 1 FROM n WHERE x < 100000
             )
             INSERT INTO page_sources (page_id, memory_source_id, linked_at, link_reason)
             SELECT printf('page-%05d', ((x - 1) / 10) + 1),
                    printf('source-%05d-%02d', ((x - 1) / 10) + 1, ((x - 1) % 10) + 1),
                    1, 'scale'
               FROM n;
             WITH RECURSIVE n(x) AS (
               VALUES(1) UNION ALL SELECT x + 1 FROM n WHERE x < 100000
             )
             INSERT INTO page_evidence (page_id, source_kind, locator, linked_at, link_reason)
             SELECT printf('page-%05d', ((x - 1) / 10) + 1),
                    CASE WHEN x = 100000 THEN 'external_file' ELSE 'memory' END,
                    printf('source-%05d-%02d', ((x - 1) / 10) + 1, ((x - 1) % 10) + 1),
                    1, 'scale'
               FROM n;",
        )
        .await
        .expect("populate deterministic scale database");
    println!(
        "FIXTURE_READY pages={PAGE_COUNT} state_entries={PAGE_COUNT} owner_rows={EVIDENCE_COUNT} evidence_rows={EVIDENCE_COUNT} projection_defect_file={PAGE_COUNT} evidence_defect_row={EVIDENCE_COUNT}"
    );
}

async fn run_fixture(root: &Path) {
    let database = libsql::Builder::new_local(root.join("lint-scale.db"))
        .build()
        .await
        .expect("open prebuilt scale database");
    let page_root = root.join("pages");
    let measured = Instant::now();
    let snapshot = LintReadSnapshot::open(&database)
        .await
        .expect("open shared production lint snapshot");
    let scan = scan_page_root(&page_root).expect("scan production Page projection");
    assert_eq!(scan.page_markdown().len(), PAGE_COUNT);
    assert_eq!(scan.raw_state.edges.len(), PAGE_COUNT);
    assert_eq!(
        scalar_count(&snapshot, "SELECT COUNT(*) FROM pages").await,
        PAGE_COUNT as u64
    );
    assert_eq!(
        scalar_count(&snapshot, "SELECT COUNT(*) FROM page_evidence").await,
        EVIDENCE_COUNT
    );
    assert_eq!(
        scalar_count(&snapshot, "SELECT COUNT(*) FROM memories").await,
        EVIDENCE_COUNT
    );

    let scope = AppliedScope::global();
    let clock = LintClock::capture();
    let gate = ExecutionGate::new(CancellationToken::new());
    let context = LintContext::new(
        &snapshot,
        &scope,
        Some(&scan),
        &clock,
        &gate,
        wenlan_types::lint::LintProfile::General,
    );
    let results = super::run(&context, true).await;
    let failed_checks = results
        .iter()
        .filter(|result| result.outcome() == LintOutcome::FailedToRun)
        .map(|result| result.check_id())
        .collect::<Vec<_>>();
    assert!(
        failed_checks.is_empty(),
        "production Page group must execute every check; failed checks: {failed_checks:?}"
    );
    assert!(results.iter().all(|result| {
        result.coverage().evidence_cap() == LINT_MAX_EVIDENCE_PER_CHECK
            && result.evidence().len() <= usize::from(LINT_MAX_EVIDENCE_PER_CHECK)
    }));

    let projection = result(&results, VERSION_ALIGNMENT_ID);
    assert_eq!(projection.outcome(), LintOutcome::Finding);
    assert_eq!(projection.coverage().denominator(), PAGE_COUNT as u64);
    assert_eq!(projection.coverage().evaluated(), PAGE_COUNT as u64);
    assert_eq!(
        projection.evidence(),
        &[LintEvidenceRef::OpaqueId {
            opaque_id: LintOpaqueId::from_sorted_position(PAGE_COUNT - 1).unwrap(),
        }]
    );

    let evidence = result(&results, SOURCE_COVERAGE_ID);
    assert_eq!(evidence.outcome(), LintOutcome::Finding);
    assert_eq!(evidence.coverage().denominator(), EVIDENCE_COUNT);
    assert_eq!(evidence.coverage().evaluated(), EVIDENCE_COUNT);
    assert_eq!(
        evidence.evidence(),
        &[LintEvidenceRef::OpaqueId {
            opaque_id: LintOpaqueId::from_sorted_position(EVIDENCE_COUNT as usize - 1).unwrap(),
        }]
    );
    drop(context);
    let receipt = snapshot
        .finish()
        .await
        .expect("finish shared lint snapshot");
    assert!(receipt.is_consistent());
    println!(
        "SCALE_GATE_PASS pages={PAGE_COUNT} state_entries={PAGE_COUNT} owner_rows={EVIDENCE_COUNT} evidence_rows={EVIDENCE_COUNT} projection_defect_file={PAGE_COUNT} evidence_defect_row={EVIDENCE_COUNT} sample_cap={LINT_MAX_EVIDENCE_PER_CHECK} measured_ms={} measured_region=shared_snapshot+page_scan+exact_population_assertions+production_page_group+snapshot_finish",
        measured.elapsed().as_millis()
    );
}

async fn scalar_count(snapshot: &LintReadSnapshot<'_>, sql: &str) -> u64 {
    let mut rows: LintRows<'_> = snapshot
        .query(sql, libsql::params::Params::None)
        .await
        .expect("count fixture rows");
    u64::try_from(
        rows.next()
            .await
            .expect("read fixture count")
            .expect("fixture count row")
            .get::<i64>(0)
            .expect("integer fixture count"),
    )
    .expect("non-negative fixture count")
}

fn result<'a>(
    results: &'a [wenlan_types::lint::LintCheckResult],
    check_id: &str,
) -> &'a wenlan_types::lint::LintCheckResult {
    results
        .iter()
        .find(|result| result.check_id() == check_id)
        .expect("production Page check result")
}
