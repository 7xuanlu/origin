// SPDX-License-Identifier: Apache-2.0
//! P3 provenance integration tests — Task 4 through Task 6.

use async_trait::async_trait;
use std::sync::Arc;
use wenlan_core::db::{DistillationCluster, MemoryDB};
use wenlan_core::llm_provider::{LlmBackend, LlmError, LlmProvider, LlmRequest};
use wenlan_core::prompts::PromptRegistry;
use wenlan_core::read_scope::ReadScope;
use wenlan_core::refinery::distill_one_cluster;
use wenlan_core::reranker::Reranker;
use wenlan_core::sources::RawDocument;
use wenlan_core::WenlanError;
use wenlan_core::{EventEmitter, NoopEmitter};
use wenlan_types::requests::CreateConceptRequest;

// ---------------------------------------------------------------------------
// Task 6 env guard: serializes tests that mutate WENLAN_ENABLE_PAGE_CHANNEL
// so they don't clobber each other when cargo runs them in parallel.
// ---------------------------------------------------------------------------

static PAGE_CHANNEL_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct PageChannelEnvGuard {
    prev: Option<String>,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl PageChannelEnvGuard {
    fn set(value: &str) -> Self {
        let guard = PAGE_CHANNEL_ENV_LOCK.lock().unwrap();
        let prev = std::env::var("WENLAN_ENABLE_PAGE_CHANNEL").ok();
        std::env::set_var("WENLAN_ENABLE_PAGE_CHANNEL", value);
        Self {
            prev,
            _guard: guard,
        }
    }
}

impl Drop for PageChannelEnvGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var("WENLAN_ENABLE_PAGE_CHANNEL", v),
            None => std::env::remove_var("WENLAN_ENABLE_PAGE_CHANNEL"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers (mirroring distillation_quality.rs / provenance_p2.rs)
// ---------------------------------------------------------------------------

async fn make_db() -> (Arc<MemoryDB>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("test.db");
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let db = MemoryDB::new(&db_path, emitter)
        .await
        .expect("MemoryDB::new");
    (Arc::new(db), dir)
}

/// Seed a raw memory via upsert.  `confirmed = None` ⇒ NULL in DB, which
/// passes the clustering filter `(confirmed = 0 OR confirmed IS NULL)`.
async fn seed_memory(db: &MemoryDB, source_id: &str, content: &str) {
    let doc = RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        summary: None,
        content: content.to_string(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        // Put all memories in the same space so they cluster under the same
        // topic label and the thin-cluster guard (200 chars) is computed
        // over all three together.
        space: Some("technology".to_string()),
        source_agent: Some("test-agent".to_string()),
        confidence: None,
        confirmed: None, // unconfirmed → eligible for clustering
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };
    db.upsert_documents(vec![doc]).await.expect("seed memory");
}

struct PageFixture<'a> {
    title: &'a str,
    summary: Option<&'a str>,
    content: &'a str,
    space: Option<&'a str>,
    source_ids: &'a [&'a str],
    creation_kind: &'a str,
    workspace: Option<&'a str>,
}

async fn create_page_fixture(db: &MemoryDB, fixture: PageFixture<'_>) -> String {
    let PageFixture {
        title,
        summary,
        content,
        space,
        source_ids,
        creation_kind,
        workspace,
    } = fixture;
    if !source_ids.is_empty() {
        db.upsert_documents(
            source_ids
                .iter()
                .map(|source_id| RawDocument {
                    source: "memory".to_string(),
                    source_id: (*source_id).to_string(),
                    title: (*source_id).to_string(),
                    content: content.to_string(),
                    last_modified: chrono::Utc::now().timestamp(),
                    memory_type: Some("fact".to_string()),
                    space: space.map(str::to_string),
                    source_agent: Some("test-agent".to_string()),
                    confirmed: Some(true),
                    ..Default::default()
                })
                .collect(),
        )
        .await
        .expect("seed page sources");
    }

    let req = CreateConceptRequest {
        title: title.to_string(),
        content: content.to_string(),
        summary: summary.map(str::to_string),
        entity_id: None,
        space: (space.map(str::to_string)).into(),
        source_memory_ids: source_ids.iter().map(|id| (*id).to_string()).collect(),
        creation_kind: Some(creation_kind.to_string()),
        workspace: workspace.map(str::to_string),
    };
    let result = if creation_kind == "distilled" {
        wenlan_core::post_write::create_page_with_tuning(
            db,
            req,
            "test",
            None,
            source_ids.len().max(1),
            1.1,
        )
        .await
        .expect("create distilled page fixture")
    } else {
        wenlan_core::post_write::create_page(db, req, "test", None)
            .await
            .expect("create page fixture")
    };
    db.set_page_review_status(&result.id, "confirmed")
        .await
        .expect("confirm page fixture");
    result.id
}

/// Read `last_distilled_at` for a memory directly from SQL.
/// Returns `None` if the memory doesn't exist or the column is NULL.
async fn last_distilled_at_of(db: &Arc<MemoryDB>, source_id: &str) -> Option<i64> {
    db.read_last_distilled_at(source_id)
        .await
        .expect("read_last_distilled_at")
}

// ---------------------------------------------------------------------------
// Stub LLM — always available; returns content rich enough to pass the
// 0.6 cosine-similarity hallucination gate when seeded with Rust async content.
// ---------------------------------------------------------------------------

/// A deterministic LLM stub that returns either a rich distillation body or
/// a short title, based on which prompt system hint is present.
struct DistillStubLlm {
    /// Fixed body text echoed back for distillation calls.
    body: String,
}

impl DistillStubLlm {
    fn new(body: impl Into<String>) -> Self {
        Self { body: body.into() }
    }
}

#[async_trait]
impl LlmProvider for DistillStubLlm {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
        let sys = request.system_prompt.as_deref().unwrap_or("");
        // "3-5 word" matches generate_short_title's system prompt
        // (refinery/mod.rs) — the title call every successful distill makes.
        if sys.contains("3-5 word") {
            Ok("Rust Async Runtime Internals".to_string())
        } else {
            // Distillation body — echo our pre-built content.
            Ok(self.body.clone())
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "distill-stub"
    }

    fn backend(&self) -> LlmBackend {
        LlmBackend::OnDevice
    }
}

// ---------------------------------------------------------------------------
// Task 4: normal distill path stamps last_distilled_at
// ---------------------------------------------------------------------------

/// Seeds three thematically-similar Rust-async memories that form a single
/// cluster at the test's 0.6 similarity threshold (their pairwise cosines
/// hover near the 0.73 production default — too knife-edge to rely on).
///
/// Total content is ~640 chars which clears the 200-char thin-cluster guard.
async fn seed_rust_async_cluster(db: &MemoryDB) {
    seed_memory(
        db,
        "mem_p3_a",
        "Rust async runtime tokio executor drives futures to completion \
         using a multi-threaded work-stealing scheduler. Each thread runs \
         a local run-queue and steals tasks from siblings when idle. \
         The reactor polls I/O readiness via epoll on Linux and kqueue on macOS.",
    )
    .await;

    seed_memory(
        db,
        "mem_p3_b",
        "Tokio async tasks are spawned via tokio::spawn and scheduled onto \
         the multi-thread runtime. The runtime splits CPU-bound and I/O-bound \
         work: blocking tasks run in a dedicated thread pool to avoid starving \
         async I/O futures on the reactor threads.",
    )
    .await;

    seed_memory(
        db,
        "mem_p3_c",
        "Async Rust futures are zero-cost abstractions: a future that has never \
         been polled allocates nothing on the heap in the common case. The state \
         machine generated by the compiler stores only the locals alive across \
         await points, keeping stack usage bounded in deep async call chains.",
    )
    .await;
}

#[tokio::test]
async fn normal_distill_stamps_last_distilled_at() {
    let (db, _dir) = make_db().await;

    // Seed 3 thematically-similar memories.  They are unconfirmed (confirmed=NULL)
    // and share the same space ("technology"), so they land in the unlinked pool
    // and cluster by cosine similarity.
    seed_rust_async_cluster(&db).await;

    // Build a distillation body whose embedding will be similar to the source
    // memories (> 0.6 cosine similarity).  The content closely paraphrases the
    // seeded memories, so the hallucination gate passes.
    let body = "- TLDR: Rust async internals\n\n\
        Tokio's async runtime uses a work-stealing multi-threaded scheduler where \
        each thread maintains a local task queue and steals from sibling threads \
        when idle.  The I/O reactor polls epoll on Linux and kqueue on macOS for \
        readiness events.  Blocking tasks run on a dedicated thread pool so they \
        never starve the async futures on the reactor threads.  Async Rust futures \
        are zero-cost state machines compiled from async fn bodies; they only \
        allocate on the heap when polled, keeping memory usage proportional to \
        the depth of concurrent work.";

    let llm: Arc<dyn LlmProvider> = Arc::new(DistillStubLlm::new(body));

    let prompts = wenlan_core::prompts::PromptRegistry::default();
    // similarity_threshold 0.6 instead of the 0.73 default: the three seeds'
    // pairwise cosines sit near 0.73, and with exactly min_cluster_size
    // memories one flipped edge from run-to-run ONNX embedding jitter kills
    // the cluster (observed flaking 4/23 runs at the default). 0.6 gives
    // margin while still exercising the real clustering code.
    let tuning = wenlan_core::tuning::DistillationConfig {
        similarity_threshold: 0.6,
        ..Default::default()
    };

    let result = wenlan_core::refinery::distill_pages(
        &db,
        Some(&llm),
        &prompts,
        &tuning,
        None, // no knowledge-writer path
    )
    .await
    .expect("distill_pages must succeed");

    assert!(
        result > 0,
        "distill_pages must have created at least one page from the cluster (got 0); \
         check that all 3 memories are unconfirmed and share the 'technology' space"
    );

    // Every source memory must now carry a non-NULL last_distilled_at.
    for sid in ["mem_p3_a", "mem_p3_b", "mem_p3_c"] {
        let ld = last_distilled_at_of(&db, sid).await;
        assert!(
            ld.is_some(),
            "{sid} must carry a non-NULL last_distilled_at after normal distill"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 5: scoped-match attach path stamps last_distilled_at for attached ids
// ---------------------------------------------------------------------------

/// Verifies that when `distill_one_cluster` takes the scoped-match ATTACH path
/// (find_matching_page_scoped fires, no new page is synthesised) it stamps
/// `last_distilled_at` only on the source_ids whose `link_page_source` call
/// succeeded, not on every id in the cluster.
#[tokio::test]
async fn scoped_match_attach_stamps_only_attached_ids() {
    let (db, _dir) = make_db().await;

    // Seed two memories into the memories table so stamp_last_distilled_at has
    // rows to UPDATE.  Content length and topic are arbitrary — the cluster is
    // built manually below; we are not going through the clustering code path.
    seed_memory(
        &db,
        "mem_s3",
        "Tokio async runtime work-stealing scheduler reactor Rust executor tasks futures \
         epoll kqueue multi-threaded blocking thread pool I/O readiness",
    )
    .await;
    seed_memory(
        &db,
        "mem_s4",
        "Rust async futures zero-cost state machine poll heap allocation await point \
         tokio spawn runtime scheduler work-stealing thread local queue",
    )
    .await;

    // We construct the centroid by embedding the SAME title+summary text so the
    // cosine between centroid and page embedding is effectively 1.0, guaranteeing
    // find_matching_page_scoped fires at the 0.85 threshold regardless of the
    // unrelated body.
    let page_title = "Tokio Async Runtime";
    let page_summary = "Tokio async runtime work-stealing scheduler reactor Rust executor";
    let page_body = "Tokio is the de-facto async runtime for Rust. It runs futures on a \
        multi-threaded work-stealing scheduler and uses the OS event queue (epoll / kqueue) \
        as its I/O reactor. Blocking tasks run on a dedicated thread pool so they never \
        starve async I/O futures.";

    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: page_title,
            summary: Some(page_summary),
            content: page_body,
            space: None,
            source_ids: &["mem_scoped_page_seed"],
            creation_kind: "distilled",
            workspace: None,
        },
    )
    .await;

    // Build the centroid embedding from title+summary (same as the page embed
    // prefix) so the cosine is near 1.0.
    let centroid_text = format!("{page_title} {page_summary}");
    let centroid = db
        .generate_embeddings(&[centroid_text])
        .expect("generate_embeddings must succeed")
        .remove(0);

    // PRECONDITION: confirm the scoped matcher finds the seed page at 0.85.
    // If this assertion fails, tune the centroid text to be more similar to
    // page_title + page_summary + page_body (the full embed text).
    {
        let probe = db
            .find_matching_page_scoped(None, &centroid, 0.85, None, false)
            .await
            .expect("find_matching_page_scoped must not error");
        assert_eq!(
            probe.as_ref().map(|p| p.id.as_str()),
            Some(page_id.as_str()),
            "precondition: scoped matcher must find seeded PageWrite page at threshold 0.85 \
             (tune centroid_text if this fails)"
        );
    }

    // Build a cluster with non-overlapping source_ids (mem_s3, mem_s4).
    // find_best_overlapping_page will return None for these (no page cites them),
    // so distill_one_cluster falls through to find_matching_page_scoped, which
    // then fires and attaches.
    let cluster = DistillationCluster {
        source_ids: vec!["mem_s3".to_string(), "mem_s4".to_string()],
        contents: vec![
            "Tokio async runtime work-stealing scheduler reactor Rust executor tasks futures"
                .to_string(),
            "Rust async futures zero-cost state machine poll heap allocation await point"
                .to_string(),
        ],
        entity_id: None,
        entity_name: Some("Tokio".to_string()),
        space: None,
        estimated_tokens: 50,
        centroid_embedding: Some(centroid),
    };

    // Use the same DistillStubLlm as Task 4 (its body is never invoked on the
    // scoped-match path because the function returns before synthesis).
    let stub_body = "Tokio async runtime work-stealing scheduler reactor.";
    let llm: Arc<dyn LlmProvider> = Arc::new(DistillStubLlm::new(stub_body));
    let prompts = PromptRegistry::default();

    let result = distill_one_cluster(&db, &llm, &prompts, &cluster, None)
        .await
        .expect("distill_one_cluster must not error");

    assert!(
        result.is_none(),
        "scoped-match attach must return Ok(None) (no new page created)"
    );

    // Both source memories must now carry a non-NULL last_distilled_at because
    // link_page_source succeeded for both (no pre-existing page_sources rows).
    for sid in ["mem_s3", "mem_s4"] {
        let ld = last_distilled_at_of(&db, sid).await;
        assert!(
            ld.is_some(),
            "{sid} must carry a non-NULL last_distilled_at after scoped-match attach"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 6 helpers
// ---------------------------------------------------------------------------

/// Seed a raw memory with an explicit space via upsert.
async fn seed_memory_in_space(db: &MemoryDB, source_id: &str, content: &str, space: &str) {
    let doc = RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        summary: None,
        content: content.to_string(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        space: Some(space.to_string()),
        source_agent: Some("test-agent".to_string()),
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };
    db.upsert_documents(vec![doc])
        .await
        .expect("seed_memory_in_space");
}

// ---------------------------------------------------------------------------
// Task 6 — Step 1: source-less page respects workspace gate
// ---------------------------------------------------------------------------

/// A source-less authored page bound to workspace="work" MUST surface under
/// work-scoped recall and MUST NOT leak into personal-scoped recall.
///
/// This verifies the "workspace axis" branch of the security gate:
///   page.workspace == space_filter => pass, regardless of sources.
/// Passing `None` for `reranker` degrades gracefully (no CE model needed in CI).
#[tokio::test]
async fn source_less_page_passes_own_workspace_no_personal_leak() {
    // Serialize env mutation so parallel test runs don't clobber each other.
    let _env = PageChannelEnvGuard::set("1");

    let (db, _dir) = make_db().await;

    // Seed anchor memories so both scoped pools are non-empty (prevents the
    // filter branch from short-circuiting due to empty memory_results).
    seed_memory_in_space(
        &db,
        "mem_work_anchor",
        "quarterly planning cadence review meeting cycle",
        "work",
    )
    .await;
    seed_memory_in_space(
        &db,
        "mem_pers_anchor",
        "quarterly planning cadence personal goals review",
        "personal",
    )
    .await;

    // Source-less authored page, confirmed (NOT unconfirmed — a passing test on
    // an unconfirmed page would bless a trust-model bypass). Zero source ids.
    let page_id = create_page_fixture(
        &db,
        PageFixture {
            title: "Quarterly Planning Cadence",
            summary: Some("how we plan quarters"),
            content: "Quarterly planning cadence prose. We review goals every quarter.",
            space: None,
            source_ids: &[],
            creation_kind: "authored",
            workspace: Some("work"),
        },
    )
    .await;

    // work-scoped recall: the page MUST surface (workspace match).
    let work = db
        .search_memory_cross_rerank(
            "quarterly planning cadence",
            10,
            None,                                  // memory_type filter
            &ReadScope::Space("work".to_string()), // space filter = "work"
            None,                                  // source_agent filter
            None,                                  // reranker (None = no CE model required in CI)
        )
        .await
        .expect("work-scoped recall must not error");

    assert!(
        work.iter()
            .any(|r| r.id == page_id || r.source_id == page_id),
        "source-less page in workspace=work must surface under work-scoped recall; \
         got results: {:?}",
        work.iter()
            .map(|r| (r.source.as_str(), r.id.as_str()))
            .collect::<Vec<_>>()
    );

    // personal-scoped recall: the same page MUST NOT appear.
    let personal = db
        .search_memory_cross_rerank(
            "quarterly planning cadence",
            10,
            None,                                      // memory_type filter
            &ReadScope::Space("personal".to_string()), // space filter = "personal"
            None,                                      // source_agent filter
            None,                                      // reranker
        )
        .await
        .expect("personal-scoped recall must not error");

    assert!(
        !personal
            .iter()
            .any(|r| r.id == page_id || r.source_id == page_id),
        "work-workspace page LEAKED into personal-scoped recall (cross-workspace disclosure); \
         got results: {:?}",
        personal
            .iter()
            .map(|r| (r.source.as_str(), r.id.as_str()))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Task 6 — Step 4 production write path test
// ---------------------------------------------------------------------------

/// create_page with workspace=Some("work") MUST persist pages.workspace='work'.
/// This verifies the production write path added in Step 4: insert_page_with_kind
/// receives workspace and writes it to the SQL row atomically.
#[tokio::test]
async fn create_page_persists_workspace() {
    use wenlan_core::post_write::create_page;
    use wenlan_types::requests::CreateConceptRequest;

    let (db, _dir) = make_db().await;

    // Seed a source memory so distilled creation_kind validation passes
    // (authored kind skips the source requirement, but we test authored here).
    let req = CreateConceptRequest {
        title: "Work Planning Notes".to_string(),
        content: "These are authored work planning notes for our quarterly review cycle."
            .to_string(),
        summary: Some("quarterly review planning".to_string()),
        entity_id: None,
        space: (None).into(), // space is the category column; workspace is the new axis
        source_memory_ids: vec![],
        creation_kind: Some("authored".to_string()),
        workspace: Some("work".to_string()),
    };

    let result = create_page(&db, req, "test-agent", None)
        .await
        .expect("create_page must succeed for authored source-less page");

    let page_id = result.id;

    // Verify pages.workspace = 'work' in the DB by reading back via get_page.
    let page = db
        .get_page(&page_id)
        .await
        .expect("get_page must not error")
        .expect("page must exist after create_page");

    assert_eq!(
        page.workspace.as_deref(),
        Some("work"),
        "pages.workspace must be 'work' after create_page with workspace=Some(\"work\")"
    );
}

// ---------------------------------------------------------------------------
// Task 7 — consolidation-demotion multiplier: distilled memory ranks below
// an equal-relevance control for the maturation window.
// ---------------------------------------------------------------------------

/// Seeds a memory with a specific content body for the demotion test.
async fn seed_memory_for_demotion(db: &MemoryDB, source_id: &str, content: &str) {
    let doc = RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        summary: None,
        content: content.to_string(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        space: Some("technology".to_string()),
        source_agent: Some("test-agent".to_string()),
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    };
    db.upsert_documents(vec![doc])
        .await
        .expect("seed memory for demotion");
}

/// A just-distilled memory must be demoted in ranking relative to an
/// equal-relevance undistilled control, and must remain reachable (rank-move
/// only, not eviction).
///
/// Uses `WENLAN_ENABLE_PAGE_CHANNEL=1` so the cross-rerank path exercises the
/// full code; `reranker=None` so no CE model is required in CI.
///
/// Seeds 4 memories total: the distilled + control pair share the same
/// query-matching topic; two extra memories on an unrelated topic bulk the
/// corpus so the vector index has enough neighbours and the content-dedup
/// logic (bigram Jaccard) never hits a near-empty pool edge case.
#[tokio::test]
async fn distilled_memory_demoted_but_reachable() {
    let _env = PageChannelEnvGuard::set("1");

    let (db, _dir) = make_db().await;

    // Two extra padding memories on an unrelated topic — give the vector
    // index a realistic corpus so both tokio memories rank in the top results.
    seed_memory_for_demotion(
        &db,
        "mem_pad_1",
        "Photosynthesis converts light energy into chemical energy stored in glucose. \
         Chlorophyll absorbs red and blue wavelengths, reflecting green. \
         ATP and NADPH produced in the light reactions power the Calvin cycle.",
    )
    .await;
    seed_memory_for_demotion(
        &db,
        "mem_pad_2",
        "Continental drift reshapes ocean basins over millions of years. \
         Subduction zones recycle oceanic crust into the mantle while \
         mid-ocean ridges create new crust from magma upwelling.",
    )
    .await;

    // The distilled/control pair: same Rust-async topic so they draw
    // comparable cosine scores against the query.
    let content_a = "Rust async runtime tokio executor drives futures to completion \
        using a multi-threaded work-stealing scheduler. Each thread runs a local \
        run-queue and steals tasks from siblings when idle. The reactor polls I/O \
        readiness via epoll on Linux and kqueue on macOS. Blocking tasks run on a \
        dedicated thread pool so they never starve the async futures.";
    let content_b = "Tokio async runtime executor futures Rust work-stealing scheduler \
        drives tasks to completion. Local run-queues per thread steal work from \
        siblings when idle. The I/O reactor uses epoll on Linux and kqueue on macOS. \
        Blocking operations run on a separate thread pool to avoid starving \
        the async reactor.";

    seed_memory_for_demotion(&db, "mem_distilled", content_a).await;
    seed_memory_for_demotion(&db, "mem_control", content_b).await;

    // Stamp mem_distilled as just-distilled (age = 0 → DEMOTE_FLOOR = 0.7).
    let now = chrono::Utc::now().timestamp();
    db.stamp_last_distilled_at(&["mem_distilled".to_string()], now)
        .await
        .expect("stamp_last_distilled_at must succeed");

    // Search without scoping, no CE reranker needed.
    let results = db
        .search_memory_cross_rerank(
            "rust async runtime tokio executor work-stealing scheduler",
            10,
            None,
            &ReadScope::Global,
            None,
            None, // reranker = None (no CE model in CI)
        )
        .await
        .expect("search_memory_cross_rerank must not error");

    // mem_distilled must still be present (demotion is rank-move, not eviction).
    assert!(
        results.iter().any(|r| r.source_id == "mem_distilled"),
        "mem_distilled must remain reachable after demotion (got: {:?})",
        results
            .iter()
            .map(|r| r.source_id.as_str())
            .collect::<Vec<_>>()
    );

    // mem_control must also be present.
    assert!(
        results.iter().any(|r| r.source_id == "mem_control"),
        "mem_control must be present in results (got: {:?})",
        results
            .iter()
            .map(|r| r.source_id.as_str())
            .collect::<Vec<_>>()
    );

    // Score-based assertion: mem_distilled's score < mem_control's score.
    // This is robust to positional ties (embedding jitter can swap equal-score rows).
    let score_distilled = results
        .iter()
        .find(|r| r.source_id == "mem_distilled")
        .map(|r| r.score)
        .expect("mem_distilled in results");
    let score_control = results
        .iter()
        .find(|r| r.source_id == "mem_control")
        .map(|r| r.score)
        .expect("mem_control in results");

    assert!(
        score_distilled < score_control,
        "distilled memory (score={score_distilled}) must rank below equal-relevance \
         control (score={score_control}) during the demotion window"
    );
}

// ---------------------------------------------------------------------------
// C1 regression: demotion must survive the CE rerank overwrite.
//
// A stub reranker that returns EQUAL scores for every candidate is the
// tightest possible probe: if demotion is applied pre-CE (the bug) the CE
// overwrites both scores to the same value and the assertion below fails; if
// demotion is applied post-CE (the fix) the demotion multiplier fires AFTER
// the equal CE scores land, making score_distilled < score_control.
// ---------------------------------------------------------------------------

/// Stub reranker that returns the same constant score (0.5) for every
/// candidate.  This guarantees the CE overwrite fires and produces equal
/// scores for all candidates, so demotion is the ONLY thing that can
/// differentiate mem_distilled from mem_control.
struct EqualScoreReranker;

impl Reranker for EqualScoreReranker {
    fn rerank(
        &self,
        _query: &str,
        candidates: &[(String, String)],
    ) -> Result<Vec<(String, f32)>, WenlanError> {
        Ok(candidates
            .iter()
            .map(|(id, _)| (id.clone(), 0.5_f32))
            .collect())
    }

    fn model_id(&self) -> &str {
        "equal-score-stub"
    }
}

/// Demotion must be rank-effective even when a CE reranker overwrites all
/// scores to the same value first.
///
/// Setup is identical to `distilled_memory_demoted_but_reachable` except
/// `reranker = Some(EqualScoreReranker)` so the CE overwrite path fires.
/// After the CE pass every candidate carries score=0.5.  The demotion
/// multiplier (DEMOTE_FLOOR=0.7 for age=0) must run POST-CE and bring
/// mem_distilled below mem_control.
#[tokio::test]
async fn demotion_survives_ce_rerank() {
    let _env = PageChannelEnvGuard::set("1");

    let (db, _dir) = make_db().await;

    // Same padding + topic pair as `distilled_memory_demoted_but_reachable`.
    seed_memory_for_demotion(
        &db,
        "mem_pad_ce1",
        "Photosynthesis converts light energy into chemical energy stored in glucose. \
         Chlorophyll absorbs red and blue wavelengths, reflecting green. \
         ATP and NADPH produced in the light reactions power the Calvin cycle.",
    )
    .await;
    seed_memory_for_demotion(
        &db,
        "mem_pad_ce2",
        "Continental drift reshapes ocean basins over millions of years. \
         Subduction zones recycle oceanic crust into the mantle while \
         mid-ocean ridges create new crust from magma upwelling.",
    )
    .await;

    let content_a = "Rust async runtime tokio executor drives futures to completion \
        using a multi-threaded work-stealing scheduler. Each thread runs a local \
        run-queue and steals tasks from siblings when idle. The reactor polls I/O \
        readiness via epoll on Linux and kqueue on macOS. Blocking tasks run on a \
        dedicated thread pool so they never starve the async futures.";
    let content_b = "Tokio async runtime executor futures Rust work-stealing scheduler \
        drives tasks to completion. Local run-queues per thread steal work from \
        siblings when idle. The I/O reactor uses epoll on Linux and kqueue on macOS. \
        Blocking operations run on a separate thread pool to avoid starving \
        the async reactor.";

    seed_memory_for_demotion(&db, "mem_distilled_ce", content_a).await;
    seed_memory_for_demotion(&db, "mem_control_ce", content_b).await;

    // Stamp mem_distilled_ce as just-distilled (age=0 → DEMOTE_FLOOR=0.7).
    let now = chrono::Utc::now().timestamp();
    db.stamp_last_distilled_at(&["mem_distilled_ce".to_string()], now)
        .await
        .expect("stamp_last_distilled_at must succeed");

    let reranker: Arc<dyn Reranker> = Arc::new(EqualScoreReranker);

    let results = db
        .search_memory_cross_rerank(
            "rust async runtime tokio executor work-stealing scheduler",
            10,
            None,
            &ReadScope::Global,
            None,
            Some(reranker),
        )
        .await
        .expect("search_memory_cross_rerank must not error");

    // Both must be reachable.
    assert!(
        results.iter().any(|r| r.source_id == "mem_distilled_ce"),
        "mem_distilled_ce must remain reachable after CE-path demotion (got: {:?})",
        results
            .iter()
            .map(|r| r.source_id.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        results.iter().any(|r| r.source_id == "mem_control_ce"),
        "mem_control_ce must be present in results (got: {:?})",
        results
            .iter()
            .map(|r| r.source_id.as_str())
            .collect::<Vec<_>>()
    );

    let score_distilled = results
        .iter()
        .find(|r| r.source_id == "mem_distilled_ce")
        .map(|r| r.score)
        .expect("mem_distilled_ce in results");
    let score_control = results
        .iter()
        .find(|r| r.source_id == "mem_control_ce")
        .map(|r| r.score)
        .expect("mem_control_ce in results");

    // After the equal-score CE pass, only the post-CE demotion multiplier can
    // differentiate these two.  If this assertion fails, the demotion block is
    // still pre-CE (bug C1 from the adversarial review).
    assert!(
        score_distilled < score_control,
        "demotion must survive CE overwrite: distilled score ({score_distilled}) must be \
         < control score ({score_control}); if equal, demotion is being erased by the CE \
         overwrite (adversarial finding C1)"
    );
}
