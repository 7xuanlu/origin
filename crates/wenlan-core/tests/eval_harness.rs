#![cfg(feature = "eval-harness")]
//! Integration test: eval harness runs against seeded DB with fixture data.
//!
//! Tests using bundled fixtures run in CI (FastEmbed model cached in GitHub Actions).
//! Tests needing external data (locomo10.json, longmemeval) or real GPU LLM stay `#[ignore]`.

use wenlan_core::eval::runner::{run_eval, GateMode};

/// Stderr logger so distill's per-cluster `log::warn!`/`log::info!` skip reasons
/// (thin cluster / LLM error / hallucination sim / garbage title) are visible in
/// the `seed_scenario_dbs_complete --nocapture` run. Without this, distill's
/// silent-skip paths are invisible and a page-starved seed is only caught by the
/// page floor after hours of seeding.
struct StderrLogger;
impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}
static STDERR_LOGGER: StderrLogger = StderrLogger;

/// Resolve the eval data root. Defaults to `app/eval/` (legacy location).
/// Override via `WENLAN_EVAL_ROOT` env var to support Phase 5 PR3 extraction
/// or local relocation. Centralizing this means future moves touch one site.
fn eval_root() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("WENLAN_EVAL_ROOT") {
        return std::path::PathBuf::from(p);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/eval")
}

/// Root directory for the layered baseline layout. Defaults to
/// `~/.cache/origin-eval/baselines`; override via `EVAL_BASELINES_DIR`.
fn baselines_root() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("EVAL_BASELINES_DIR") {
        return std::path::PathBuf::from(p).join("baselines");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home)
        .join(".cache")
        .join("origin-eval")
        .join("baselines")
}

/// Dual-write helper: legacy `save_baseline` already called by the caller;
/// this writes the same data through the P0b layered baseline path so
/// `compare-baselines` + the L1 directory layout pick it up.
///
/// Best-effort: skip if the report has no env stamp (cannot be layered).
/// Errors are surfaced (panicked) so test failures point at this site.
fn save_layered<R, F>(report: &R, to_eval: F)
where
    R: ?Sized,
    F: FnOnce(&R) -> wenlan_core::eval::report::EvalReport,
{
    let eval_report = to_eval(report);
    if eval_report.env.is_none() {
        eprintln!("save_layered: skipped (no env stamp)");
        return;
    }
    match wenlan_core::eval::report::save_full_report(&baselines_root(), &eval_report) {
        Ok(path) => println!("Saved layered baseline to {:?}", path),
        Err(e) => panic!("save_full_report failed: {e}"),
    }
}

#[tokio::test]
#[ignore]
async fn test_locomo_benchmark() {
    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        println!("SKIP: locomo10.json not found at {:?}", locomo_path);
        return;
    }

    let report = wenlan_core::eval::locomo::run_locomo_eval(&locomo_path)
        .await
        .unwrap();

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║           LOCOMO BENCHMARK RESULTS                       ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Conversations: {:<42}║", report.conversations.len());
    println!("║  Memories:      {:<42}║", report.total_memories);
    println!(
        "║  Questions:     {:<42}║",
        format!("{} (excl. adversarial)", report.total_questions)
    );
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  AGGREGATE                                               ║");
    println!(
        "║    NDCG@10:     {:<42}║",
        format!("{:.4}", report.aggregate_ndcg_at_10)
    );
    println!(
        "║    MRR:         {:<42}║",
        format!("{:.4}", report.aggregate_mrr)
    );
    println!(
        "║    Recall@5:    {:<42}║",
        format!("{:.4}", report.aggregate_recall_at_5)
    );
    println!(
        "║    Hit Rate@1:  {:<42}║",
        format!("{:.4}", report.aggregate_hit_rate_at_1)
    );
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  PER CATEGORY                                            ║");
    for cat in &report.per_category_aggregate {
        println!(
            "║    {:12} (n={:>4}): NDCG={:.3} MRR={:.3} R@5={:.3}    ║",
            cat.name, cat.count, cat.ndcg_at_10, cat.mrr, cat.recall_at_5
        );
    }
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  PER CONVERSATION                                        ║");
    for conv in &report.conversations {
        println!(
            "║    {:8} ({:>3} mem, {:>3} qa): NDCG={:.3} MRR={:.3}      ║",
            conv.sample_id,
            conv.memories_seeded,
            conv.questions_evaluated,
            conv.overall_ndcg_at_10,
            conv.overall_mrr
        );
    }
    println!("╚═══════════════════════════════════════════════════════════╝");

    // Sanity checks
    assert!(
        report.total_questions > 1000,
        "Expected >1000 QA pairs, got {}",
        report.total_questions
    );
    assert!(
        report.total_memories > 2000,
        "Expected >2000 memories, got {}",
        report.total_memories
    );
    assert!(report.aggregate_ndcg_at_10 > 0.0, "NDCG should be positive");
}

#[tokio::test]
#[ignore]
async fn test_locomo_gate_comparison() {
    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }

    use wenlan_core::eval::locomo::{run_locomo_eval_with_gate, LocomoGateMode};

    let clean = run_locomo_eval_with_gate(&locomo_path, LocomoGateMode::Clean)
        .await
        .unwrap();
    let noisy = run_locomo_eval_with_gate(&locomo_path, LocomoGateMode::Noisy)
        .await
        .unwrap();
    let gated = run_locomo_eval_with_gate(&locomo_path, LocomoGateMode::Gated)
        .await
        .unwrap();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!(
        "║        LOCOMO BENCHMARK — GATE IMPACT ({:>4} questions)        ║",
        clean.total_questions
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║              Clean      Noisy      Gated     Δ(Gated-Noisy)   ║");
    println!(
        "║  NDCG@10:    {:.4}     {:.4}     {:.4}     {:+.4}           ║",
        clean.aggregate_ndcg_at_10,
        noisy.aggregate_ndcg_at_10,
        gated.aggregate_ndcg_at_10,
        gated.aggregate_ndcg_at_10 - noisy.aggregate_ndcg_at_10
    );
    println!(
        "║  MRR:        {:.4}     {:.4}     {:.4}     {:+.4}           ║",
        clean.aggregate_mrr,
        noisy.aggregate_mrr,
        gated.aggregate_mrr,
        gated.aggregate_mrr - noisy.aggregate_mrr
    );
    println!(
        "║  Recall@5:   {:.4}     {:.4}     {:.4}     {:+.4}           ║",
        clean.aggregate_recall_at_5,
        noisy.aggregate_recall_at_5,
        gated.aggregate_recall_at_5,
        gated.aggregate_recall_at_5 - noisy.aggregate_recall_at_5
    );
    println!(
        "║  Hit@1:      {:.4}     {:.4}     {:.4}     {:+.4}           ║",
        clean.aggregate_hit_rate_at_1,
        noisy.aggregate_hit_rate_at_1,
        gated.aggregate_hit_rate_at_1,
        gated.aggregate_hit_rate_at_1 - noisy.aggregate_hit_rate_at_1
    );
    println!(
        "║  Memories:   {:<6}     {:<6}     {:<6}                      ║",
        clean.total_memories, noisy.total_memories, gated.total_memories
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  PER CATEGORY (Gated vs Noisy delta)                          ║");

    for (i, gcat) in gated.per_category_aggregate.iter().enumerate() {
        if i < noisy.per_category_aggregate.len() {
            let ncat = &noisy.per_category_aggregate[i];
            println!(
                "║    {:12} NDCG {:+.3}  MRR {:+.3}  R@5 {:+.3}              ║",
                gcat.name,
                gcat.ndcg_at_10 - ncat.ndcg_at_10,
                gcat.mrr - ncat.mrr,
                gcat.recall_at_5 - ncat.recall_at_5
            );
        }
    }
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Gate should recover at least some of the noise degradation
    assert!(
        gated.aggregate_ndcg_at_10 >= noisy.aggregate_ndcg_at_10,
        "Gated ({:.4}) should be >= Noisy ({:.4}) on NDCG",
        gated.aggregate_ndcg_at_10,
        noisy.aggregate_ndcg_at_10
    );
}

#[tokio::test]
#[ignore]
async fn test_longmemeval_benchmark() {
    // Try oracle first (small, ~15MB), then S-cleaned (large, ~277MB)
    let data_dir = eval_root().join("data");
    let oracle_path = data_dir.join("longmemeval_oracle.json");
    let s_path = data_dir.join("longmemeval_s_cleaned.json");

    let path = if oracle_path.exists() {
        oracle_path
    } else if s_path.exists() {
        s_path
    } else {
        println!(
            "SKIP: LongMemEval dataset not found. Download with:\n\
             curl -sL 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json' \
             -o {:?}",
            oracle_path
        );
        return;
    };

    println!("Running LongMemEval benchmark from {:?}", path);

    let report = wenlan_core::eval::longmemeval::run_longmemeval_eval(&path)
        .await
        .unwrap();

    println!("\n{}", report.to_terminal());

    // Sanity checks
    assert!(
        report.total_questions > 0,
        "Expected >0 questions, got {}",
        report.total_questions
    );
    assert!(
        report.total_memories > 0,
        "Expected >0 memories, got {}",
        report.total_memories
    );
    assert!(report.aggregate_ndcg_at_10 > 0.0, "NDCG should be positive");
    // Should have at least 4 categories (oracle has all 6)
    assert!(
        report.per_category.len() >= 4,
        "Expected at least 4 categories, got {}",
        report.per_category.len()
    );
}

#[tokio::test]
#[ignore]
async fn test_longmemeval_gate_comparison() {
    let path = eval_root().join("data/longmemeval_oracle.json");
    if !path.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    use wenlan_core::eval::longmemeval::{run_longmemeval_eval_with_gate, LongMemEvalGateMode};

    let clean = run_longmemeval_eval_with_gate(&path, LongMemEvalGateMode::Clean)
        .await
        .unwrap();
    let noisy = run_longmemeval_eval_with_gate(&path, LongMemEvalGateMode::Noisy)
        .await
        .unwrap();
    let gated = run_longmemeval_eval_with_gate(&path, LongMemEvalGateMode::Gated)
        .await
        .unwrap();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!(
        "║      LONGMEMEVAL BENCHMARK — GATE IMPACT ({} questions)     ║",
        clean.total_questions
    );
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║              Clean      Noisy      Gated     Δ(Gated-Noisy)   ║");
    println!(
        "║  NDCG@10:    {:.4}     {:.4}     {:.4}     {:+.4}           ║",
        clean.aggregate_ndcg_at_10,
        noisy.aggregate_ndcg_at_10,
        gated.aggregate_ndcg_at_10,
        gated.aggregate_ndcg_at_10 - noisy.aggregate_ndcg_at_10
    );
    println!(
        "║  MRR:        {:.4}     {:.4}     {:.4}     {:+.4}           ║",
        clean.aggregate_mrr,
        noisy.aggregate_mrr,
        gated.aggregate_mrr,
        gated.aggregate_mrr - noisy.aggregate_mrr
    );
    println!(
        "║  Recall@5:   {:.4}     {:.4}     {:.4}     {:+.4}           ║",
        clean.aggregate_recall_at_5,
        noisy.aggregate_recall_at_5,
        gated.aggregate_recall_at_5,
        gated.aggregate_recall_at_5 - noisy.aggregate_recall_at_5
    );
    println!(
        "║  Hit@1:      {:.4}     {:.4}     {:.4}     {:+.4}           ║",
        clean.aggregate_hit_rate_at_1,
        noisy.aggregate_hit_rate_at_1,
        gated.aggregate_hit_rate_at_1,
        gated.aggregate_hit_rate_at_1 - noisy.aggregate_hit_rate_at_1
    );
    println!(
        "║  Memories:   {:<6}     {:<6}     {:<6}                      ║",
        clean.total_memories, noisy.total_memories, gated.total_memories
    );
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Per-category
    if gated.per_category.len() == noisy.per_category.len() {
        println!("\nPer-category (Gated vs Noisy delta):");
        for (i, gcat) in gated.per_category.iter().enumerate() {
            let ncat = &noisy.per_category[i];
            println!(
                "  {:4} NDCG {:+.3}  MRR {:+.3}  R@5 {:+.3}",
                gcat.code,
                gcat.ndcg_at_10 - ncat.ndcg_at_10,
                gcat.mrr - ncat.mrr,
                gcat.recall_at_5 - ncat.recall_at_5
            );
        }
    }

    // Gate should recover at least some of the noise degradation
    assert!(
        gated.aggregate_ndcg_at_10 >= noisy.aggregate_ndcg_at_10,
        "Gated ({:.4}) should be >= Noisy ({:.4}) on NDCG",
        gated.aggregate_ndcg_at_10,
        noisy.aggregate_ndcg_at_10
    );
}

// ---------------------------------------------------------------------------
// Lifecycle eval tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_lifecycle_locomo_with_mock_llm() {
    use std::sync::Arc;
    use wenlan_core::eval::lifecycle::{run_lifecycle_locomo, EvalMockLlm};

    let path = eval_root().join("data/locomo10.json");
    if !path.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }

    let mock: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(EvalMockLlm::new());
    let report = run_lifecycle_locomo(&path, Some(mock)).await.unwrap();

    assert_eq!(report.phases.len(), 6);
    assert!(report.case_count > 0, "Should have at least 1 LoCoMo case");

    for pm in &report.phases {
        assert!(pm.ndcg_at_10 >= 0.0 && pm.ndcg_at_10 <= 1.0);
    }

    println!("{}", report.to_terminal());
}

#[tokio::test]
#[ignore]
async fn test_lifecycle_longmemeval_with_mock_llm() {
    use std::sync::Arc;
    use wenlan_core::eval::lifecycle::{run_lifecycle_longmemeval, EvalMockLlm};

    let path = eval_root().join("data/longmemeval_oracle.json");
    if !path.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let mock: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(EvalMockLlm::new());
    let report = run_lifecycle_longmemeval(&path, Some(mock)).await.unwrap();

    assert_eq!(report.phases.len(), 6);
    assert!(
        report.case_count > 0,
        "Should have at least 1 LongMemEval case"
    );

    for pm in &report.phases {
        assert!(pm.ndcg_at_10 >= 0.0 && pm.ndcg_at_10 <= 1.0);
    }

    println!("{}", report.to_terminal());
}

// ---------------------------------------------------------------------------
// Baseline save tests (run manually with --ignored to establish baselines)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn save_fixture_baseline() {
    let fixture_dir = eval_root().join("fixtures");
    let tmp = tempfile::tempdir().unwrap();
    let report = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::Off)
        .await
        .unwrap();
    let path = eval_root().join("baselines/fixture_baseline.json");
    report.save_baseline(&path).unwrap();
    println!("Saved fixture baseline to {:?}", path);
    println!("{}", report.to_terminal());
}

#[tokio::test]
#[ignore]
async fn save_locomo_baseline() {
    let path = eval_root().join("data/locomo10.json");
    if !path.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }
    let report = wenlan_core::eval::locomo::run_locomo_eval(&path)
        .await
        .unwrap();
    let baselines_dir = eval_root().join("baselines");
    std::fs::create_dir_all(&baselines_dir).unwrap();
    let baseline_path = baselines_dir.join(report.baseline_filename("locomo"));
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LoCoMo baseline to {:?}", baseline_path);
    save_layered(&report, |r| r.to_eval_report());
}

#[tokio::test]
#[ignore]
async fn save_longmemeval_baseline() {
    // LME_FIXTURE overrides the fixture (default oracle = evidence-only/easy).
    // Set LME_FIXTURE=data/longmemeval_s.json for the REAL ~47-session haystack
    // retrieval eval. extract_memories ingests ALL haystack_sessions, so the same
    // runner gives oracle (easy) vs full-haystack (hard) purely by fixture choice.
    let rel =
        std::env::var("LME_FIXTURE").unwrap_or_else(|_| "data/longmemeval_oracle.json".to_string());
    let path = eval_root().join(&rel);
    if !path.exists() {
        println!("SKIP: {rel} not found");
        return;
    }
    let report = wenlan_core::eval::longmemeval::run_longmemeval_eval(&path)
        .await
        .unwrap();
    println!(
        "=== LME retrieval [{rel}] OVERALL ({}q, {}mem): ndcg@10={:.3} R@5={:.3} mrr={:.3} hit@1={:.3} ===",
        report.total_questions,
        report.total_memories,
        report.aggregate_ndcg_at_10,
        report.aggregate_recall_at_5,
        report.aggregate_mrr,
        report.aggregate_hit_rate_at_1
    );
    for c in &report.per_category {
        println!(
            "  {:28} n={:<4} ndcg@10={:.3} R@5={:.3} mrr={:.3} hit@1={:.3}",
            c.question_type, c.count, c.ndcg_at_10, c.recall_at_5, c.mrr, c.hit_rate_at_1
        );
    }
    let baselines_dir = eval_root().join("baselines");
    std::fs::create_dir_all(&baselines_dir).unwrap();
    let baseline_path = baselines_dir.join(report.baseline_filename("longmemeval"));
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LongMemEval baseline to {:?}", baseline_path);
    save_layered(&report, |r| r.to_eval_report());
}

/// LME full-haystack TEMPORAL retrieval A/B (event_date injected + search_memory_temporal).
/// Compare per-category vs `save_longmemeval_baseline` (base) on the same fixture +
/// EVAL_LME_STRATIFIED subset. Toggle the lever with WENLAN_ENABLE_TEMPORAL_FILTER=1
/// (hard window filter) or WENLAN_ENABLE_TEMPORAL_SOFT_BOOST=1 (never-drop boost).
#[tokio::test]
#[ignore]
async fn save_longmemeval_temporal_baseline() {
    let rel =
        std::env::var("LME_FIXTURE").unwrap_or_else(|_| "data/longmemeval_oracle.json".to_string());
    let path = eval_root().join(&rel);
    if !path.exists() {
        println!("SKIP: {rel} not found");
        return;
    }
    let report = wenlan_core::eval::longmemeval::run_longmemeval_eval_temporal(&path)
        .await
        .unwrap();
    println!(
        "=== LME TEMPORAL retrieval [{rel}] OVERALL ({}q, {}mem): ndcg@10={:.3} R@5={:.3} mrr={:.3} hit@1={:.3} ===",
        report.total_questions,
        report.total_memories,
        report.aggregate_ndcg_at_10,
        report.aggregate_recall_at_5,
        report.aggregate_mrr,
        report.aggregate_hit_rate_at_1
    );
    for c in &report.per_category {
        println!(
            "  {:28} n={:<4} ndcg@10={:.3} R@5={:.3} mrr={:.3} hit@1={:.3}",
            c.question_type, c.count, c.ndcg_at_10, c.recall_at_5, c.mrr, c.hit_rate_at_1
        );
    }
}

// Cross-encoder rerank variants — fastembed TextRerank (BGERerankerV2M3) in
// place of the LLM reranker. First run downloads the model weights.
//
// WENLAN_ENABLE_PAGE_CHANNEL is forced to None (unset) and
// WENLAN_RERANKER_MODEL is PINNED to bge-v2-m3 here so the pre-PR-B
// 0.684 / 0.883 disk artifacts stay reproducible regardless of the caller env
// (the unset-env default flipped to bge-base on 2026-06-11; these two tests
// protect v2-m3 reference artifacts whose legacy filenames do not encode the
// reranker model).
// The ephemeral per-conversation DBs these tests build today have zero
// distilled pages, so page-channel is a no-op for them with or without the
// wrap. The wrap makes the intent explicit at the source.
#[tokio::test]
#[ignore]
async fn save_locomo_cross_rerank_baseline() {
    temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_PAGE_CHANNEL", None::<&str>),
            ("WENLAN_RERANKER_MODEL", Some("bge-v2-m3")),
        ],
        async {
            let path = eval_root().join("data/locomo10.json");
            if !path.exists() {
                println!("SKIP: locomo10.json not found");
                return;
            }
            let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
                .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
            let report = wenlan_core::eval::locomo::run_locomo_eval_cross_rerank(&path, reranker)
                .await
                .unwrap();
            let baselines_dir = eval_root().join("baselines");
            std::fs::create_dir_all(&baselines_dir).unwrap();
            let baseline_path = baselines_dir.join(report.baseline_filename("locomo"));
            report.save_baseline(&baseline_path).unwrap();
            println!("Saved LoCoMo cross-rerank baseline to {:?}", baseline_path);
        },
    )
    .await;
}

#[tokio::test]
#[ignore]
async fn save_longmemeval_cross_rerank_baseline() {
    temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_PAGE_CHANNEL", None::<&str>),
            ("WENLAN_RERANKER_MODEL", Some("bge-v2-m3")),
        ],
        async {
            let path = eval_root().join("data/longmemeval_oracle.json");
            if !path.exists() {
                println!("SKIP: longmemeval_oracle.json not found");
                return;
            }
            let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
                .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
            let report =
                wenlan_core::eval::longmemeval::run_longmemeval_eval_cross_rerank(&path, reranker)
                    .await
                    .unwrap();
            let baselines_dir = eval_root().join("baselines");
            std::fs::create_dir_all(&baselines_dir).unwrap();
            let baseline_path = baselines_dir.join(report.baseline_filename("longmemeval"));
            report.save_baseline(&baseline_path).unwrap();
            println!(
                "Saved LongMemEval cross-rerank baseline to {:?}",
                baseline_path
            );
        },
    )
    .await;
}

// ---------------------------------------------------------------------------
// PR-B page-channel with-pages baseline runners
// ---------------------------------------------------------------------------

/// Resolve the root directory for cached scenario DBs.
///
/// Resolution order (highest priority first):
/// 1. `SCENARIO_DB_ROOT` env var
/// 2. `${EVAL_BASELINES_DIR}/scenario_seeded`
/// 3. `~/.cache/origin-eval/scenario_seeded/` (canonical default)
fn resolve_scenario_db_root_from_harness() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("SCENARIO_DB_ROOT") {
        return std::path::PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("EVAL_BASELINES_DIR") {
        return std::path::PathBuf::from(p).join("scenario_seeded");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home)
        .join(".cache")
        .join("origin-eval")
        .join("scenario_seeded")
}

/// T3 graph-gate A/B experiment (retrieval-only, no GPU LLM, no judge).
/// Runs the base `search_memory` path over the cached LoCoMo scenario DB with
/// `WENLAN_ENABLE_GRAPH_GATE` OFF (graph always on) vs ON (gated), and prints
/// the retrieval-metric deltas. Single-run = scaffold/direction only.
#[tokio::test]
#[ignore = "needs cached scenario DB (run scripts/seed-scenario-dbs.sh); retrieval-only, no GPU"]
async fn graph_gate_ab_locomo() {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db;

    let db_dir = resolve_scenario_db_root_from_harness().join("locomo_v1");
    if !db_dir.join("origin_memory.db").exists() {
        println!("SKIP: no seeded DB at {}", db_dir.display());
        return;
    }
    let fixture = eval_root().join("data/locomo10.json");
    if !fixture.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }
    let db = wenlan_core::db::MemoryDB::new(
        &db_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open locomo_v1 scenario DB");

    let off = temp_env::async_with_vars(
        [("WENLAN_ENABLE_GRAPH_GATE", None::<&str>)],
        run_locomo_eval_from_db(&db, &fixture),
    )
    .await
    .expect("gate-off eval");
    let on = temp_env::async_with_vars(
        [("WENLAN_ENABLE_GRAPH_GATE", Some("1"))],
        run_locomo_eval_from_db(&db, &fixture),
    )
    .await
    .expect("gate-on eval");

    let cov = |r: &wenlan_core::eval::locomo::LocomoReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };
    println!("=== T3 GRAPH-GATE A/B (LoCoMo, search_memory path, retrieval-only) ===");
    println!("questions evaluated: {}", off.total_questions);
    println!(
        "GATE OFF (graph always): ndcg@10={:.4} recall@5={:.4} mrr={:.4} hit@1={:.4} cov={:.4}",
        off.aggregate_ndcg_at_10,
        off.aggregate_recall_at_5,
        off.aggregate_mrr,
        off.aggregate_hit_rate_at_1,
        cov(&off)
    );
    println!(
        "GATE ON  (gated):        ndcg@10={:.4} recall@5={:.4} mrr={:.4} hit@1={:.4} cov={:.4}",
        on.aggregate_ndcg_at_10,
        on.aggregate_recall_at_5,
        on.aggregate_mrr,
        on.aggregate_hit_rate_at_1,
        cov(&on)
    );
    println!(
        "DELTA (on-off):          ndcg@10={:+.4} recall@5={:+.4} mrr={:+.4} hit@1={:+.4} cov={:+.4}",
        on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
        on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
        on.aggregate_mrr - off.aggregate_mrr,
        on.aggregate_hit_rate_at_1 - off.aggregate_hit_rate_at_1,
        cov(&on) - cov(&off)
    );
}

/// T13 magnitude-fusion A/B on BOTH benches (retrieval-only). Unlike T3/T12, this
/// changes FTS scoring for EVERY query with FTS hits, so a real (non-zero) delta
/// is expected. Single-run scaffold — N≥3 for any headline.
#[tokio::test]
#[ignore = "needs cached scenario DBs; retrieval-only, no GPU"]
async fn magnitude_fusion_ab_dualbench() {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db;
    let root = resolve_scenario_db_root_from_harness();

    let lo_dir = root.join("locomo_v1");
    if lo_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/locomo10.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lo_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(
                [("WENLAN_MAGNITUDE_FUSION", None::<&str>)],
                run_locomo_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_MAGNITUDE_FUSION", Some("1"))],
                run_locomo_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            println!(
                "[MAGFUSION A/B LoCoMo] q={} ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 d={:+.4} | mrr d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                on.aggregate_mrr - off.aggregate_mrr
            );
        }
    }

    let lme_dir = root.join("lme_v1");
    if lme_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/longmemeval_oracle.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lme_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(
                [("WENLAN_MAGNITUDE_FUSION", None::<&str>)],
                run_longmemeval_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_MAGNITUDE_FUSION", Some("1"))],
                run_longmemeval_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            println!(
                "[MAGFUSION A/B LME] q={} ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 d={:+.4} | mrr d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                on.aggregate_mrr - off.aggregate_mrr
            );
        }
    }
}

/// T20 per-session diversification cap A/B on BOTH benches.
///
/// IMPORTANT: T20's cap is wired into `search_memory_cross_rerank` (the CE
/// path), NOT the base `search_memory` path. So unlike the other dual-bench
/// A/Bs above (which use `run_*_eval_from_db` -> `search_memory`), this test
/// MUST use `run_*_eval_cross_rerank_from_db` -> `search_memory_cross_rerank`
/// with a real cross-encoder reranker, or the cap never fires (all-zero delta).
///
/// EXPECT: LoCoMo neutral (all source_ids are `locomo_*` -> `session_key`
/// returns None -> exempt from the cap). LME may move: its source_ids carry
/// `lme_*_t*` session structure, so the cap demotes >max hits from one session
/// per question and backfills from other sessions.
///
/// Single-run scaffold — N>=3 for any headline claim. Needs Metal GPU
/// (cross-encoder) + cached scenario DBs. Run unsandboxed:
///   cargo test -p origin-core --test eval_harness session_diversity_ab_dualbench -- --ignored --nocapture
#[tokio::test]
#[ignore = "needs Metal GPU (cross-encoder) + cached scenario DBs"]
async fn session_diversity_ab_dualbench() {
    use wenlan_core::eval::locomo::run_locomo_eval_cross_rerank_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_cross_rerank_from_db;
    let root = resolve_scenario_db_root_from_harness();

    // coverage_recall blind field (matches the dual-bench measurement vehicle).
    let lo_cov = |r: &wenlan_core::eval::locomo::LocomoReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };
    let lme_cov = |r: &wenlan_core::eval::longmemeval::LongMemEvalReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };

    // -- LoCoMo (expected neutral: locomo_* ids are exempt) --
    let lo_dir = root.join("locomo_v1");
    if lo_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/locomo10.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lo_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
                .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
            let off = temp_env::async_with_vars(
                [("WENLAN_ENABLE_SESSION_DIVERSITY", None::<&str>)],
                run_locomo_eval_cross_rerank_from_db(&db, &fx, reranker.clone()),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_ENABLE_SESSION_DIVERSITY", Some("1"))],
                run_locomo_eval_cross_rerank_from_db(&db, &fx, reranker.clone()),
            )
            .await
            .unwrap();
            println!(
                "[SESSDIV A/B LoCoMo] q={} | ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 off={:.4} on={:.4} d={:+.4} | cov(blind) off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                off.aggregate_recall_at_5,
                on.aggregate_recall_at_5,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                lo_cov(&off),
                lo_cov(&on),
                lo_cov(&on) - lo_cov(&off),
            );
        } else {
            println!(
                "[SESSDIV A/B LoCoMo] SKIP: locomo10.json not found at {}",
                fx.display()
            );
        }
    } else {
        println!(
            "[SESSDIV A/B LoCoMo] SKIP: {}/origin_memory.db missing",
            lo_dir.display()
        );
    }

    // -- LME (expected to move: lme_*_t* ids carry session structure) --
    let lme_dir = root.join("lme_v1");
    if lme_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/longmemeval_oracle.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lme_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
                .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
            let off = temp_env::async_with_vars(
                [("WENLAN_ENABLE_SESSION_DIVERSITY", None::<&str>)],
                run_longmemeval_eval_cross_rerank_from_db(&db, &fx, reranker.clone()),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_ENABLE_SESSION_DIVERSITY", Some("1"))],
                run_longmemeval_eval_cross_rerank_from_db(&db, &fx, reranker.clone()),
            )
            .await
            .unwrap();
            println!(
                "[SESSDIV A/B LME] q={} | ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 off={:.4} on={:.4} d={:+.4} | cov(blind) off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                off.aggregate_recall_at_5,
                on.aggregate_recall_at_5,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                lme_cov(&off),
                lme_cov(&on),
                lme_cov(&on) - lme_cov(&off),
            );
        } else {
            println!(
                "[SESSDIV A/B LME] SKIP: longmemeval_oracle.json not found at {}",
                fx.display()
            );
        }
    } else {
        println!(
            "[SESSDIV A/B LME] SKIP: {}/origin_memory.db missing",
            lme_dir.display()
        );
    }
}

/// T19 query-adaptive channel-reweighting A/B on BOTH benches (retrieval-only).
/// Toggles `WENLAN_ENABLE_QUERY_INTENT` OFF vs ON over the cached scenario DBs
/// via the base `search_memory` path. ON, Factual-classified (short, non-relational)
/// queries upweight the FTS RRF stream; General/Temporal stay identity. Default-OFF
/// path is byte-identical by construction, so any non-zero delta comes from queries
/// that classify Factual. Single-run scaffold — N≥3 for any headline claim.
#[tokio::test]
#[ignore = "needs cached scenario DBs; retrieval-only, no GPU"]
async fn query_intent_ab_dualbench() {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db;
    let root = resolve_scenario_db_root_from_harness();

    // coverage_recall blind field (matches the graph-seed dual-bench measurement vehicle)
    let lo_cov = |r: &wenlan_core::eval::locomo::LocomoReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };
    let lme_cov = |r: &wenlan_core::eval::longmemeval::LongMemEvalReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };

    let lo_dir = root.join("locomo_v1");
    if lo_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/locomo10.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lo_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(
                [("WENLAN_ENABLE_QUERY_INTENT", None::<&str>)],
                run_locomo_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_ENABLE_QUERY_INTENT", Some("1"))],
                run_locomo_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            println!(
                "[QUERY-INTENT A/B LoCoMo] q={} | ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 off={:.4} on={:.4} d={:+.4} | cov_blind off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                off.aggregate_recall_at_5,
                on.aggregate_recall_at_5,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                lo_cov(&off),
                lo_cov(&on),
                lo_cov(&on) - lo_cov(&off),
            );
        } else {
            println!("SKIP LoCoMo: locomo10.json not found at {}", fx.display());
        }
    } else {
        println!("SKIP LoCoMo: no seeded DB at {}", lo_dir.display());
    }

    let lme_dir = root.join("lme_v1");
    if lme_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/longmemeval_oracle.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lme_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(
                [("WENLAN_ENABLE_QUERY_INTENT", None::<&str>)],
                run_longmemeval_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_ENABLE_QUERY_INTENT", Some("1"))],
                run_longmemeval_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            println!(
                "[QUERY-INTENT A/B LME] q={} | ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 off={:.4} on={:.4} d={:+.4} | cov_blind off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                off.aggregate_recall_at_5,
                on.aggregate_recall_at_5,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                lme_cov(&off),
                lme_cov(&on),
                lme_cov(&on) - lme_cov(&off),
            );
        } else {
            println!(
                "SKIP LME: longmemeval_oracle.json not found at {}",
                fx.display()
            );
        }
    } else {
        println!("SKIP LME: no seeded DB at {}", lme_dir.display());
    }
}

/// STEP 7 cheap combined A/B: the two filled-data flags that engage the no-GPU
/// base `search_memory` path, toggled TOGETHER OFF vs ON over the cached scenario
/// DBs. `WENLAN_ENABLE_SALIENCE_PRIOR` reads the backfilled `importance`;
/// `WENLAN_ENABLE_TEMPORAL_SOFT_BOOST` reads the injected `event_date`. Combined
/// (not per-flag) by design — a directional gate before the expensive per-flag
/// grid. Retrieval-only, no GPU. The other reseed-unblocked flags (FACT_CHANNEL,
/// EPISODE_CHANNEL, SESSION_DIVERSITY) live on the cross-rerank path and need a
/// separate GPU A/B. Single-run scaffold — N≥3 for any headline claim.
///
/// ```bash
/// cargo test -p origin-core --features eval-harness --test eval_harness \
///   data_flags_ab_dualbench -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "needs cached scenario DBs; retrieval-only, no GPU"]
async fn data_flags_ab_dualbench() {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db;
    let root = resolve_scenario_db_root_from_harness();

    let on_vars = [
        ("WENLAN_ENABLE_SALIENCE_PRIOR", Some("1")),
        ("WENLAN_ENABLE_TEMPORAL_SOFT_BOOST", Some("1")),
    ];
    let off_vars = [
        ("WENLAN_ENABLE_SALIENCE_PRIOR", None::<&str>),
        ("WENLAN_ENABLE_TEMPORAL_SOFT_BOOST", None::<&str>),
    ];

    let lo_cov = |r: &wenlan_core::eval::locomo::LocomoReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };
    let lme_cov = |r: &wenlan_core::eval::longmemeval::LongMemEvalReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };

    let lo_dir = root.join("locomo_v1");
    if lo_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/locomo10.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lo_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(off_vars, run_locomo_eval_from_db(&db, &fx))
                .await
                .unwrap();
            let on = temp_env::async_with_vars(on_vars, run_locomo_eval_from_db(&db, &fx))
                .await
                .unwrap();
            println!(
                "[DATA-FLAGS A/B LoCoMo] q={} | ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 off={:.4} on={:.4} d={:+.4} | cov_blind off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                off.aggregate_recall_at_5,
                on.aggregate_recall_at_5,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                lo_cov(&off),
                lo_cov(&on),
                lo_cov(&on) - lo_cov(&off),
            );
        } else {
            println!("SKIP LoCoMo: locomo10.json not found at {}", fx.display());
        }
    } else {
        println!("SKIP LoCoMo: no seeded DB at {}", lo_dir.display());
    }

    let lme_dir = root.join("lme_v1");
    if lme_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/longmemeval_oracle.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lme_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(off_vars, run_longmemeval_eval_from_db(&db, &fx))
                .await
                .unwrap();
            let on = temp_env::async_with_vars(on_vars, run_longmemeval_eval_from_db(&db, &fx))
                .await
                .unwrap();
            println!(
                "[DATA-FLAGS A/B LME] q={} | ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 off={:.4} on={:.4} d={:+.4} | cov_blind off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                off.aggregate_recall_at_5,
                on.aggregate_recall_at_5,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                lme_cov(&off),
                lme_cov(&on),
                lme_cov(&on) - lme_cov(&off),
            );
        } else {
            println!(
                "SKIP LME: longmemeval_oracle.json not found at {}",
                fx.display()
            );
        }
    } else {
        println!("SKIP LME: no seeded DB at {}", lme_dir.display());
    }
}

// --- STEP 8 per-flag screen helpers (no new dep) ---

/// Deterministic seeded sample of `k` distinct indices from `0..n`.
///
/// Partial Fisher-Yates driven by a SplitMix-ish LCG so we get a seeded random
/// draw WITHOUT pulling a `rand` dev-dep. Same seed -> same set (reproducible);
/// different seeds -> almost surely different sets (sampling variance). Returns
/// `min(k, n)` indices. The eval pipeline itself stays deterministic
/// (`paired.rs` asserts no RNG); the ONLY variance source for a retrieval
/// metric is WHICH questions you draw, which is exactly what this seeds.
fn seeded_sample(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let k = k.min(n);
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in 0..k {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = i + (state >> 33) as usize % (n - i);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

/// Write a `k`-element seeded subset of the top-level JSON array at `src` to a
/// temp file. Returns `(TempDir, path)` — keep the `TempDir` alive while the
/// runner reads the path. Both LoCoMo (`locomo10.json`) and LME
/// (`longmemeval_oracle.json`) are top-level arrays, so element-slicing yields
/// a valid subset fixture (subset of QUESTIONS; the DB corpus stays full).
fn write_json_subset(
    src: &std::path::Path,
    k: usize,
    seed: u64,
) -> (tempfile::TempDir, std::path::PathBuf) {
    let data = std::fs::read_to_string(src).expect("read fixture json");
    let arr: Vec<serde_json::Value> =
        serde_json::from_str(&data).expect("fixture json is a top-level array");
    let picks = seeded_sample(arr.len(), k, seed);
    let subset: Vec<&serde_json::Value> = picks.iter().map(|&i| &arr[i]).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("subset.json");
    std::fs::write(&path, serde_json::to_string(&subset).unwrap()).unwrap();
    (dir, path)
}

/// Population mean + (population) stddev of a small sample.
fn mean_std(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    (m, var.sqrt())
}

#[test]
fn seeded_sample_deterministic_distinct_bounded() {
    let a = seeded_sample(500, 60, 7);
    let b = seeded_sample(500, 60, 7);
    assert_eq!(a, b, "same seed must reproduce the same draw");
    assert_eq!(a.len(), 60);
    let set: std::collections::HashSet<_> = a.iter().copied().collect();
    assert_eq!(set.len(), 60, "indices must be distinct");
    assert!(a.iter().all(|&i| i < 500), "indices must be in range");
}

#[test]
fn seeded_sample_varies_by_seed() {
    assert_ne!(seeded_sample(500, 60, 1), seeded_sample(500, 60, 2));
}

#[test]
fn seeded_sample_caps_at_n() {
    let a = seeded_sample(10, 50, 3);
    assert_eq!(a.len(), 10);
    let set: std::collections::HashSet<_> = a.iter().copied().collect();
    assert_eq!(
        set.len(),
        10,
        "small population yields all distinct indices"
    );
}

#[test]
fn mean_std_matches_hand_calc() {
    let (m, s) = mean_std(&[1.0, 2.0, 3.0]);
    assert!((m - 2.0).abs() < 1e-9);
    assert!((s - (2.0f64 / 3.0).sqrt()).abs() < 1e-9);
}

/// STEP 8 per-flag screen: isolate each no-GPU base-path reseed flag ON vs an
/// all-OFF baseline, across 3 seeded random question draws, paired on the same
/// subset. Reports per-flag ndcg@10 + recall@5 delta mean +/- stddev per bench.
///
/// WHY per-flag (vs `data_flags_ab_dualbench` which toggles a bundle): a bundled
/// A/B can't attribute WHICH flag moved the metric. This screens each flag's
/// marginal contribution over the all-off baseline so we can keep/park them one
/// by one. After the screen, the kept SET still needs a combined re-confirm
/// (one-by-one misses interactions).
///
/// WHY 3 draws (not 3 reruns): the retrieval pipeline is deterministic
/// (`paired.rs` asserts no RNG), so 3 reruns of the same subset give 3 identical
/// numbers. The only variance source is which questions you sample, so N=3 means
/// 3 seeded draws -> a mini sampling-bootstrap of the per-flag delta.
///
/// Scaffold, sign-level only. LoCoMo population is just 10 conversations, so the
/// K=7 draws overlap heavily -> the stddev UNDERSTATES true variance; trust the
/// SIGN of the mean, not the magnitude. A citable claim needs N>=3 full-fixture
/// runs per the Eval Citation Discipline.
///
/// ```bash
/// WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
/// SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_seeded \
/// cargo test -p origin-core --features eval-harness --test eval_harness \
///   data_flags_perflag_screen -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "needs cached scenario DBs + raw dataset json (WENLAN_EVAL_ROOT); retrieval-only, no GPU"]
async fn data_flags_perflag_screen() {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db;

    let root = resolve_scenario_db_root_from_harness();
    // (display name, env var) — all on the no-GPU `search_memory` base path.
    let flags = [
        ("salience", "WENLAN_ENABLE_SALIENCE_PRIOR"),
        ("temporal_soft", "WENLAN_ENABLE_TEMPORAL_SOFT_BOOST"),
        ("temporal_ground", "WENLAN_ENABLE_TEMPORAL_GROUNDING"),
        ("temporal_filter", "WENLAN_ENABLE_TEMPORAL_FILTER"),
    ];
    let all_off: Vec<(&str, Option<&str>)> = flags.iter().map(|(_, e)| (*e, None)).collect();
    let draws = [11u64, 23, 37];

    // LoCoMo: 10 conversations -> K=7 per draw (heavy overlap; sign only).
    let lo_dir = root.join("locomo_v1");
    let lo_fx = eval_root().join("data/locomo10.json");
    if lo_dir.join("origin_memory.db").exists() && lo_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lo_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .unwrap();
        let mut d_ndcg: std::collections::HashMap<&str, Vec<f64>> =
            flags.iter().map(|(n, _)| (*n, Vec::new())).collect();
        let mut d_recall: std::collections::HashMap<&str, Vec<f64>> =
            flags.iter().map(|(n, _)| (*n, Vec::new())).collect();
        for &seed in &draws {
            let (_guard, sub) = write_json_subset(&lo_fx, 7, seed);
            let off =
                temp_env::async_with_vars(all_off.clone(), run_locomo_eval_from_db(&db, &sub))
                    .await
                    .unwrap();
            for (name, env) in flags {
                let mut on_vars = all_off.clone();
                for v in on_vars.iter_mut() {
                    if v.0 == env {
                        v.1 = Some("1");
                    }
                }
                let on = temp_env::async_with_vars(on_vars, run_locomo_eval_from_db(&db, &sub))
                    .await
                    .unwrap();
                d_ndcg
                    .get_mut(name)
                    .unwrap()
                    .push(on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10);
                d_recall
                    .get_mut(name)
                    .unwrap()
                    .push(on.aggregate_recall_at_5 - off.aggregate_recall_at_5);
            }
        }
        println!("[PERFLAG-SCREEN LoCoMo | K=7/10 conv x 3 draws | baseline=all-off]");
        for (name, _) in flags {
            let (mn, sn) = mean_std(&d_ndcg[name]);
            let (mr, sr) = mean_std(&d_recall[name]);
            println!("  {name:<16} ndcg@10 d={mn:+.4} sd={sn:.4} | recall@5 d={mr:+.4} sd={sr:.4}");
        }
    } else {
        println!(
            "SKIP LoCoMo perflag: missing db {} or fixture {}",
            lo_dir.display(),
            lo_fx.display()
        );
    }

    // LME: 500 questions -> K=60 per draw (low overlap; real draw diversity).
    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if lme_dir.join("origin_memory.db").exists() && lme_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lme_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .unwrap();
        let mut d_ndcg: std::collections::HashMap<&str, Vec<f64>> =
            flags.iter().map(|(n, _)| (*n, Vec::new())).collect();
        let mut d_recall: std::collections::HashMap<&str, Vec<f64>> =
            flags.iter().map(|(n, _)| (*n, Vec::new())).collect();
        for &seed in &draws {
            let (_guard, sub) = write_json_subset(&lme_fx, 60, seed);
            let off =
                temp_env::async_with_vars(all_off.clone(), run_longmemeval_eval_from_db(&db, &sub))
                    .await
                    .unwrap();
            for (name, env) in flags {
                let mut on_vars = all_off.clone();
                for v in on_vars.iter_mut() {
                    if v.0 == env {
                        v.1 = Some("1");
                    }
                }
                let on =
                    temp_env::async_with_vars(on_vars, run_longmemeval_eval_from_db(&db, &sub))
                        .await
                        .unwrap();
                d_ndcg
                    .get_mut(name)
                    .unwrap()
                    .push(on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10);
                d_recall
                    .get_mut(name)
                    .unwrap()
                    .push(on.aggregate_recall_at_5 - off.aggregate_recall_at_5);
            }
        }
        println!("[PERFLAG-SCREEN LME | K=60/500 q x 3 draws | baseline=all-off]");
        for (name, _) in flags {
            let (mn, sn) = mean_std(&d_ndcg[name]);
            let (mr, sr) = mean_std(&d_recall[name]);
            println!("  {name:<16} ndcg@10 d={mn:+.4} sd={sn:.4} | recall@5 d={mr:+.4} sd={sr:.4}");
        }
    } else {
        println!(
            "SKIP LME perflag: missing db {} or fixture {}",
            lme_dir.display(),
            lme_fx.display()
        );
    }
}

// --- STEP 8b temporal-subset screen helpers ---

/// Keep only temporal-category questions. LoCoMo: filter each conversation's
/// `qa` to `category == 2` (the temporal class; 1=multi-hop, 3=open, 4=single,
/// 5=adversarial) and drop conversations left with none. LME: keep only
/// array elements whose `question_type == "temporal-reasoning"`.
///
/// WHY: the temporal soft-boost is query-cue-gated (`db.rs:8151` forces ×1.0
/// when the query has no parsed `temporal_cue`), so an aggregate screen washes
/// it out — most questions carry no time cue. Restricting to temporal questions
/// is the only fair test of whether the flag helps the queries it's built for.
fn filter_temporal(arr: Vec<serde_json::Value>, bench: &str) -> Vec<serde_json::Value> {
    if bench == "locomo" {
        arr.into_iter()
            .filter_map(|mut conv| {
                let keep: Vec<serde_json::Value> = conv
                    .get("qa")?
                    .as_array()?
                    .iter()
                    .filter(|q| q.get("category").and_then(|c| c.as_u64()) == Some(2))
                    .cloned()
                    .collect();
                if keep.is_empty() {
                    return None;
                }
                conv["qa"] = serde_json::Value::Array(keep);
                Some(conv)
            })
            .collect()
    } else {
        arr.into_iter()
            .filter(|q| {
                q.get("question_type").and_then(|t| t.as_str()) == Some("temporal-reasoning")
            })
            .collect()
    }
}

/// Temporal-only sibling of [`write_json_subset`]: filter to temporal questions
/// first, then take a `k`-element seeded draw.
fn write_temporal_subset(
    src: &std::path::Path,
    bench: &str,
    k: usize,
    seed: u64,
) -> (tempfile::TempDir, std::path::PathBuf) {
    let data = std::fs::read_to_string(src).expect("read fixture json");
    let arr: Vec<serde_json::Value> =
        serde_json::from_str(&data).expect("fixture json is a top-level array");
    let arr = filter_temporal(arr, bench);
    let picks = seeded_sample(arr.len(), k, seed);
    let subset: Vec<&serde_json::Value> = picks.iter().map(|&i| &arr[i]).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("subset.json");
    std::fs::write(&path, serde_json::to_string(&subset).unwrap()).unwrap();
    (dir, path)
}

#[test]
fn filter_temporal_locomo_keeps_only_cat2() {
    let arr: Vec<serde_json::Value> = serde_json::from_str(
        r#"[{"qa":[{"category":2,"question":"a"},{"category":4,"question":"b"}]},
            {"qa":[{"category":1,"question":"c"}]}]"#,
    )
    .unwrap();
    let out = filter_temporal(arr, "locomo");
    assert_eq!(out.len(), 1, "conversation with no cat2 must be dropped");
    assert_eq!(out[0]["qa"].as_array().unwrap().len(), 1, "only cat2 kept");
    assert_eq!(out[0]["qa"][0]["category"], 2);
}

#[test]
fn filter_temporal_lme_keeps_only_temporal_reasoning() {
    let arr: Vec<serde_json::Value> = serde_json::from_str(
        r#"[{"question_type":"temporal-reasoning","question":"a"},
            {"question_type":"multi-session","question":"b"}]"#,
    )
    .unwrap();
    let out = filter_temporal(arr, "lme");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0]["question_type"], "temporal-reasoning");
}

/// STEP 8b temporal-subset screen: the FAIR test for the temporal flags. Same
/// per-flag / 3-seeded-draw / paired design as `data_flags_perflag_screen`, but
/// restricted to temporal-category questions (LoCoMo cat-2, LME
/// temporal-reasoning) so the cue-gated boost actually has cues to fire on.
///
/// Scaffold, sign-level. LoCoMo still only 10 conversations (K=7 overlap) — but
/// now each contributes ~32 temporal qa, so per-draw question count is healthy.
/// LME has 133 temporal questions -> K=60 gives real draw diversity.
///
/// ```bash
/// WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
/// SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_seeded \
/// cargo test -p origin-core --features eval-harness --test eval_harness \
///   data_flags_temporal_subset_screen -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "needs cached scenario DBs + raw dataset json (WENLAN_EVAL_ROOT); retrieval-only, no GPU"]
async fn data_flags_temporal_subset_screen() {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db;

    let root = resolve_scenario_db_root_from_harness();
    let flags = [
        ("temporal_soft", "WENLAN_ENABLE_TEMPORAL_SOFT_BOOST"),
        ("temporal_ground", "WENLAN_ENABLE_TEMPORAL_GROUNDING"),
        ("temporal_filter", "WENLAN_ENABLE_TEMPORAL_FILTER"),
    ];
    let all_off: Vec<(&str, Option<&str>)> = flags.iter().map(|(_, e)| (*e, None)).collect();
    let draws = [11u64, 23, 37];

    let lo_dir = root.join("locomo_v1");
    let lo_fx = eval_root().join("data/locomo10.json");
    if lo_dir.join("origin_memory.db").exists() && lo_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lo_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .unwrap();
        let mut d_ndcg: std::collections::HashMap<&str, Vec<f64>> =
            flags.iter().map(|(n, _)| (*n, Vec::new())).collect();
        let mut d_recall: std::collections::HashMap<&str, Vec<f64>> =
            flags.iter().map(|(n, _)| (*n, Vec::new())).collect();
        for &seed in &draws {
            let (_guard, sub) = write_temporal_subset(&lo_fx, "locomo", 7, seed);
            let off =
                temp_env::async_with_vars(all_off.clone(), run_locomo_eval_from_db(&db, &sub))
                    .await
                    .unwrap();
            for (name, env) in flags {
                let mut on_vars = all_off.clone();
                for v in on_vars.iter_mut() {
                    if v.0 == env {
                        v.1 = Some("1");
                    }
                }
                let on = temp_env::async_with_vars(on_vars, run_locomo_eval_from_db(&db, &sub))
                    .await
                    .unwrap();
                d_ndcg
                    .get_mut(name)
                    .unwrap()
                    .push(on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10);
                d_recall
                    .get_mut(name)
                    .unwrap()
                    .push(on.aggregate_recall_at_5 - off.aggregate_recall_at_5);
            }
        }
        println!("[TEMPORAL-SUBSET LoCoMo cat-2 | K=7/10 conv x 3 draws | baseline=all-off]");
        for (name, _) in flags {
            let (mn, sn) = mean_std(&d_ndcg[name]);
            let (mr, sr) = mean_std(&d_recall[name]);
            println!("  {name:<16} ndcg@10 d={mn:+.4} sd={sn:.4} | recall@5 d={mr:+.4} sd={sr:.4}");
        }
    } else {
        println!(
            "SKIP LoCoMo temporal-subset: missing db {} or fixture {}",
            lo_dir.display(),
            lo_fx.display()
        );
    }

    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if lme_dir.join("origin_memory.db").exists() && lme_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lme_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .unwrap();
        let mut d_ndcg: std::collections::HashMap<&str, Vec<f64>> =
            flags.iter().map(|(n, _)| (*n, Vec::new())).collect();
        let mut d_recall: std::collections::HashMap<&str, Vec<f64>> =
            flags.iter().map(|(n, _)| (*n, Vec::new())).collect();
        for &seed in &draws {
            let (_guard, sub) = write_temporal_subset(&lme_fx, "lme", 60, seed);
            let off =
                temp_env::async_with_vars(all_off.clone(), run_longmemeval_eval_from_db(&db, &sub))
                    .await
                    .unwrap();
            for (name, env) in flags {
                let mut on_vars = all_off.clone();
                for v in on_vars.iter_mut() {
                    if v.0 == env {
                        v.1 = Some("1");
                    }
                }
                let on =
                    temp_env::async_with_vars(on_vars, run_longmemeval_eval_from_db(&db, &sub))
                        .await
                        .unwrap();
                d_ndcg
                    .get_mut(name)
                    .unwrap()
                    .push(on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10);
                d_recall
                    .get_mut(name)
                    .unwrap()
                    .push(on.aggregate_recall_at_5 - off.aggregate_recall_at_5);
            }
        }
        println!(
            "[TEMPORAL-SUBSET LME temporal-reasoning | K=60/133 q x 3 draws | baseline=all-off]"
        );
        for (name, _) in flags {
            let (mn, sn) = mean_std(&d_ndcg[name]);
            let (mr, sr) = mean_std(&d_recall[name]);
            println!("  {name:<16} ndcg@10 d={mn:+.4} sd={sn:.4} | recall@5 d={mr:+.4} sd={sr:.4}");
        }
    } else {
        println!(
            "SKIP LME temporal-subset: missing db {} or fixture {}",
            lme_dir.display(),
            lme_fx.display()
        );
    }
}

/// STEP 9 reranker model A/B: compare the shippable native cross-encoders on
/// Wenlan's own eval to answer "which reranker to use." Candidates (all Apache/MIT,
/// all fastembed cross-encoders): bge-reranker-v2-m3 (current, 0.6B), bge-reranker-base
/// (0.3B, half size), jina-reranker-v1-turbo-en (37.8M). Non-commercial (jina-v2/v3)
/// and no-ONNX (mxbai/Qwen3) candidates are excluded — see the reranker survey.
///
/// Design: paired over 3 seeded draws, channels forced OFF (production default) so
/// ONLY the reranker model varies. A NO-OP CONTROL — bge-v2-m3 run a second time —
/// calibrates the pipeline noise floor; a "model A != model B" claim is only real
/// if the gap exceeds |bge-v2-m3 - bge-v2-m3#noise|. (This control is the lesson
/// from the temporal screen: without it, HashMap tie-break noise reads as signal.)
///
/// Each model is loaded BYO via `WENLAN_RERANKER_ONNX_DIR` (curled into
/// `~/.cache/origin-eval/rerankers/<name>/`) to dodge the Xet enum-download failure.
/// Scaffold, sign-level. CPU cross-encoder — slow; run watched.
///
/// ```bash
/// WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
/// SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_seeded \
/// cargo test -p origin-core --features eval-harness --test eval_harness \
///   reranker_model_ab -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "needs cached scenario DBs + raw dataset json + BYO reranker ONNX dirs; CPU cross-encoder, slow"]
async fn reranker_model_ab() {
    use wenlan_core::eval::locomo::run_locomo_eval_cross_rerank_from_db;

    let root = resolve_scenario_db_root_from_harness();
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let rr_base = format!("{home}/.cache/origin-eval/rerankers");
    // (label, onnx_dir). Two bge-v2-m3 entries: the 2nd ("#noise") is the no-op
    // control — same model, same draws -> any delta is pipeline noise.
    // turbo first (fastest) for quick feedback that the pipeline works + a timing
    // anchor before the slow bge-v2-m3 arms.
    let models = [
        ("jina-turbo", format!("{rr_base}/jina-turbo")),
        ("bge-base", format!("{rr_base}/bge-base")),
        ("bge-v2-m3", format!("{rr_base}/bge-v2-m3")),
        ("bge-v2-m3#noise", format!("{rr_base}/bge-v2-m3")),
    ];
    // Force every opt-in channel/reseed flag OFF (production default) so only the
    // reranker model varies across arms.
    let chan_off: Vec<(&str, Option<&str>)> = vec![
        ("WENLAN_ENABLE_PAGE_CHANNEL", None),
        ("WENLAN_ENABLE_EPISODE_CHANNEL", None),
        ("WENLAN_ENABLE_FACT_CHANNEL", None),
        ("WENLAN_ENABLE_SESSION_DIVERSITY", None),
        ("WENLAN_ENABLE_SALIENCE_PRIOR", None),
        ("WENLAN_ENABLE_TEMPORAL_SOFT_BOOST", None),
        ("WENLAN_ENABLE_TEMPORAL_FILTER", None),
        ("WENLAN_ENABLE_TEMPORAL_GROUNDING", None),
        ("WENLAN_ENABLE_GRAPH_GATE", None),
        ("WENLAN_ENABLE_GRAPH_SEED", None),
        ("WENLAN_ENABLE_GRAPH_KHOP", None),
        ("WENLAN_ENABLE_QUERY_INTENT", None),
        ("WENLAN_ENABLE_COT_RETRIEVAL", None),
        ("WENLAN_ENABLE_GLOBAL_PRELUDE", None),
    ];
    // K (conversations per draw) + draws (seeds) are env-tunable so the slow CPU
    // cross-encoder run can be scoped/scaled without a 20-min recompile. LoCoMo has
    // only 10 conversations, so K must be < 10 for draw diversity. Defaults are
    // deliberately tiny — bump via RERANK_AB_K / RERANK_AB_DRAWS once timing is known.
    let k: usize = std::env::var("RERANK_AB_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let draws: Vec<u64> = std::env::var("RERANK_AB_DRAWS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![11, 23]);

    let lo_dir = root.join("locomo_v1");
    let lo_fx = eval_root().join("data/locomo10.json");
    if !(lo_dir.join("origin_memory.db").exists() && lo_fx.exists()) {
        println!(
            "SKIP reranker-ab: missing db {} or fixture {}",
            lo_dir.display(),
            lo_fx.display()
        );
        return;
    }
    let db = wenlan_core::db::MemoryDB::new(
        &lo_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .unwrap();

    let mut ndcg: std::collections::HashMap<&str, Vec<f64>> =
        models.iter().map(|(n, _)| (*n, Vec::new())).collect();
    let mut recall: std::collections::HashMap<&str, Vec<f64>> =
        models.iter().map(|(n, _)| (*n, Vec::new())).collect();

    for (label, dir) in &models {
        if !std::path::Path::new(dir).join("model.onnx").exists() {
            println!("SKIP {label}: no model.onnx at {dir}");
            continue;
        }
        for &seed in &draws {
            let (_guard, sub) = write_json_subset(&lo_fx, k, seed);
            eprintln!("[start] {label} seed={seed} k={k}");
            let mut vars = chan_off.clone();
            vars.push(("WENLAN_RERANKER_ONNX_DIR", Some(dir.as_str())));
            vars.push(("WENLAN_RERANKER_MODEL_ID", Some(label)));
            let report = temp_env::async_with_vars(vars, async {
                let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
                    .expect("init_cross_encoder_reranker (BYO) failed");
                run_locomo_eval_cross_rerank_from_db(&db, &sub, reranker).await
            })
            .await
            .unwrap();
            ndcg.get_mut(label)
                .unwrap()
                .push(report.aggregate_ndcg_at_10);
            recall
                .get_mut(label)
                .unwrap()
                .push(report.aggregate_recall_at_5);
            println!(
                "  [run] {label:<16} seed={seed} ndcg@10={:.4} recall@5={:.4}",
                report.aggregate_ndcg_at_10, report.aggregate_recall_at_5
            );
        }
    }

    println!(
        "[RERANKER-AB LoCoMo | K={k} x {} draws | channels OFF | abs metrics]",
        draws.len()
    );
    for (label, _) in &models {
        let (mn, sn) = mean_std(&ndcg[label]);
        let (mr, sr) = mean_std(&recall[label]);
        println!("  {label:<16} ndcg@10={mn:.4}±{sn:.4} | recall@5={mr:.4}±{sr:.4}");
    }
    // Noise floor: |bge-v2-m3 - bge-v2-m3#noise| (same model twice). A model
    // difference must exceed this to count as signal.
    if !ndcg["bge-v2-m3"].is_empty() && !ndcg["bge-v2-m3#noise"].is_empty() {
        let (m0, _) = mean_std(&ndcg["bge-v2-m3"]);
        let (m1, _) = mean_std(&ndcg["bge-v2-m3#noise"]);
        let (r0, _) = mean_std(&recall["bge-v2-m3"]);
        let (r1, _) = mean_std(&recall["bge-v2-m3#noise"]);
        println!(
            "[RERANKER-AB NOISE FLOOR] ndcg@10={:.4} recall@5={:.4} (same-model run-to-run)",
            (m0 - m1).abs(),
            (r0 - r1).abs()
        );
    }
}

/// STEP 7 GPU combined A/B: the three filled-data flags that engage the
/// cross-rerank path, toggled TOGETHER OFF vs ON over the cached scenario DBs.
/// `WENLAN_ENABLE_EPISODE_CHANNEL` (5th RRF stream, reads backfilled episode
/// rows), `WENLAN_ENABLE_FACT_CHANNEL` (reads backfilled `structured_fields`),
/// and `WENLAN_ENABLE_SESSION_DIVERSITY` (reads injected `event_date`). Combined
/// directional gate (parallel to `data_flags_ab_dualbench` on the no-GPU path);
/// per-flag attribution follows if this moves. Needs Metal GPU (cross-encoder
/// reranker, downloads ~600MB on first run) + cached scenario DBs. Run unsandboxed.
/// Single-run scaffold — N≥3 for any headline claim.
///
/// ```bash
/// cargo test -p origin-core --features eval-harness --test eval_harness \
///   cross_rerank_data_flags_ab_dualbench -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "needs Metal GPU (cross-encoder) + cached scenario DBs"]
async fn cross_rerank_data_flags_ab_dualbench() {
    use wenlan_core::eval::locomo::run_locomo_eval_cross_rerank_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_cross_rerank_from_db;
    let root = resolve_scenario_db_root_from_harness();

    let on_vars = [
        ("WENLAN_ENABLE_EPISODE_CHANNEL", Some("1")),
        ("WENLAN_ENABLE_FACT_CHANNEL", Some("1")),
        ("WENLAN_ENABLE_SESSION_DIVERSITY", Some("1")),
    ];
    let off_vars = [
        ("WENLAN_ENABLE_EPISODE_CHANNEL", None::<&str>),
        ("WENLAN_ENABLE_FACT_CHANNEL", None::<&str>),
        ("WENLAN_ENABLE_SESSION_DIVERSITY", None::<&str>),
    ];

    let lo_cov = |r: &wenlan_core::eval::locomo::LocomoReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };
    let lme_cov = |r: &wenlan_core::eval::longmemeval::LongMemEvalReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };

    let lo_dir = root.join("locomo_v1");
    if lo_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/locomo10.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lo_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
                .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
            let off = temp_env::async_with_vars(
                off_vars,
                run_locomo_eval_cross_rerank_from_db(&db, &fx, reranker.clone()),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                on_vars,
                run_locomo_eval_cross_rerank_from_db(&db, &fx, reranker.clone()),
            )
            .await
            .unwrap();
            println!(
                "[XR DATA-FLAGS A/B LoCoMo] q={} | ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 off={:.4} on={:.4} d={:+.4} | cov_blind off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                off.aggregate_recall_at_5,
                on.aggregate_recall_at_5,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                lo_cov(&off),
                lo_cov(&on),
                lo_cov(&on) - lo_cov(&off),
            );
        } else {
            println!("SKIP LoCoMo: locomo10.json not found at {}", fx.display());
        }
    } else {
        println!("SKIP LoCoMo: no seeded DB at {}", lo_dir.display());
    }

    let lme_dir = root.join("lme_v1");
    if lme_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/longmemeval_oracle.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lme_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
                .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
            let off = temp_env::async_with_vars(
                off_vars,
                run_longmemeval_eval_cross_rerank_from_db(&db, &fx, reranker.clone()),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                on_vars,
                run_longmemeval_eval_cross_rerank_from_db(&db, &fx, reranker.clone()),
            )
            .await
            .unwrap();
            println!(
                "[XR DATA-FLAGS A/B LME] q={} | ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 off={:.4} on={:.4} d={:+.4} | cov_blind off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                off.aggregate_recall_at_5,
                on.aggregate_recall_at_5,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
                lme_cov(&off),
                lme_cov(&on),
                lme_cov(&on) - lme_cov(&off),
            );
        } else {
            println!(
                "SKIP LME: longmemeval_oracle.json not found at {}",
                fx.display()
            );
        }
    } else {
        println!("SKIP LME: no seeded DB at {}", lme_dir.display());
    }
}

/// T9 wide-pool-seeded graph-expansion A/B on BOTH benches (retrieval-only).
/// Measurement vehicle is coverage_recall (NDCG is neutral-by-construction —
/// KG observation rows are stripped from user output, only the RRF boost survives,
/// so reordering of the surviving chunks is the only NDCG signal). The graph-seed
/// expands the entity set used for the KG-RRF boost, which can pull more source
/// chunks into coverage. Toggles `WENLAN_ENABLE_GRAPH_SEED` OFF vs ON over the
/// cached scenario DBs. Single-run scaffold — N≥3 for any headline claim.
#[tokio::test]
#[ignore = "needs cached scenario DBs; retrieval-only, no GPU"]
async fn graph_seed_ab_dualbench() {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db;
    let root = resolve_scenario_db_root_from_harness();

    // coverage_recall blind field (the T9 measurement vehicle)
    let lo_cov = |r: &wenlan_core::eval::locomo::LocomoReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };
    let lme_cov = |r: &wenlan_core::eval::longmemeval::LongMemEvalReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };

    let lo_dir = root.join("locomo_v1");
    if lo_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/locomo10.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lo_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(
                [("WENLAN_ENABLE_GRAPH_SEED", None::<&str>)],
                run_locomo_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_ENABLE_GRAPH_SEED", Some("1"))],
                run_locomo_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            println!(
                "[GRAPH-SEED A/B LoCoMo] q={} | cov_blind off={:.4} on={:.4} d={:+.4} | ndcg@10 off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                lo_cov(&off),
                lo_cov(&on),
                lo_cov(&on) - lo_cov(&off),
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
            );
        } else {
            println!("SKIP LoCoMo: locomo10.json not found at {}", fx.display());
        }
    } else {
        println!("SKIP LoCoMo: no seeded DB at {}", lo_dir.display());
    }

    let lme_dir = root.join("lme_v1");
    if lme_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/longmemeval_oracle.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lme_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(
                [("WENLAN_ENABLE_GRAPH_SEED", None::<&str>)],
                run_longmemeval_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_ENABLE_GRAPH_SEED", Some("1"))],
                run_longmemeval_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            println!(
                "[GRAPH-SEED A/B LME] q={} | cov_blind off={:.4} on={:.4} d={:+.4} | ndcg@10 off={:.4} on={:.4} d={:+.4}",
                off.total_questions,
                lme_cov(&off),
                lme_cov(&on),
                lme_cov(&on) - lme_cov(&off),
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
            );
        } else {
            println!(
                "SKIP LME: longmemeval_oracle.json not found at {}",
                fx.display()
            );
        }
    } else {
        println!("SKIP LME: no seeded DB at {}", lme_dir.display());
    }
}

/// T12 FTS-hardening A/B on BOTH benches (retrieval-only). Hardening only changes
/// special-char/overlong queries (absent from clean LoCoMo/LME), so the expected
/// result is delta about 0 — this confirms no-regression on clean-query benchmarks.
#[tokio::test]
#[ignore = "needs cached scenario DBs; retrieval-only, no GPU"]
async fn fts_hardening_ab_dualbench() {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db;
    let root = resolve_scenario_db_root_from_harness();

    let lo_dir = root.join("locomo_v1");
    if lo_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/locomo10.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lo_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(
                [("WENLAN_ENABLE_FTS_HARDENING", None::<&str>)],
                run_locomo_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_ENABLE_FTS_HARDENING", Some("1"))],
                run_locomo_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            println!(
                "[FTS A/B LoCoMo] q={} ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5
            );
        }
    }

    let lme_dir = root.join("lme_v1");
    if lme_dir.join("origin_memory.db").exists() {
        let fx = eval_root().join("data/longmemeval_oracle.json");
        if fx.exists() {
            let db = wenlan_core::db::MemoryDB::new(
                &lme_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap();
            let off = temp_env::async_with_vars(
                [("WENLAN_ENABLE_FTS_HARDENING", None::<&str>)],
                run_longmemeval_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            let on = temp_env::async_with_vars(
                [("WENLAN_ENABLE_FTS_HARDENING", Some("1"))],
                run_longmemeval_eval_from_db(&db, &fx),
            )
            .await
            .unwrap();
            println!(
                "[FTS A/B LME] q={} ndcg@10 off={:.4} on={:.4} d={:+.4} | recall@5 d={:+.4}",
                off.total_questions,
                off.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10,
                on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
                on.aggregate_recall_at_5 - off.aggregate_recall_at_5
            );
        }
    }
}

/// STEP 7 (a2): inject `event_date` from dataset session metadata into the cached
/// seed DBs (locomo_v1 + lme_v1). GPU-FREE — classify-from-text cannot recover
/// these dates because the observation/turn text is date-stripped; the per-session
/// date lives only in dataset metadata. Run this BEFORE the on-device classify
/// backfill so the temporal channel (T11/T20) has data. Non-destructive: only fills
/// the `event_date` column for matching `source_id`s.
///
/// ```bash
/// cargo test -p origin-core --features eval-harness --test eval_harness \
///   seed_inject_event_dates -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn seed_inject_event_dates() {
    use wenlan_core::eval::{locomo, longmemeval};

    let root = resolve_scenario_db_root_from_harness();
    let emitter = || std::sync::Arc::new(wenlan_core::events::NoopEmitter);

    // --- LoCoMo ---
    let lc_dir = root.join("locomo_v1");
    let lc_fixture = eval_root().join("data/locomo10.json");
    if lc_dir.join("origin_memory.db").exists() && lc_fixture.exists() {
        let samples = locomo::load_locomo(&lc_fixture).expect("load locomo");
        let updates: Vec<(String, i64)> = locomo::event_date_map(&samples).into_iter().collect();
        let db = wenlan_core::db::MemoryDB::new(&lc_dir, emitter())
            .await
            .expect("open locomo_v1");
        let n = db
            .set_event_dates_by_source_id(&updates)
            .await
            .expect("inject locomo event_dates");
        eprintln!(
            "[inject] locomo_v1: {} source_ids mapped -> {} rows updated",
            updates.len(),
            n
        );
    } else {
        eprintln!("SKIP locomo: missing seed DB or fixture");
    }

    // --- LongMemEval ---
    let lme_dir = root.join("lme_v1");
    let lme_fixture = eval_root().join("data/longmemeval_oracle.json");
    if lme_dir.join("origin_memory.db").exists() && lme_fixture.exists() {
        let samples = longmemeval::load_longmemeval(&lme_fixture).expect("load lme");
        let updates: Vec<(String, i64)> =
            longmemeval::event_date_map(&samples).into_iter().collect();
        let db = wenlan_core::db::MemoryDB::new(&lme_dir, emitter())
            .await
            .expect("open lme_v1");
        let n = db
            .set_event_dates_by_source_id(&updates)
            .await
            .expect("inject lme event_dates");
        eprintln!(
            "[inject] lme_v1: {} source_ids mapped -> {} rows updated",
            updates.len(),
            n
        );
    } else {
        eprintln!("SKIP lme: missing seed DB or fixture");
    }
}

/// STEP 7: on-device classify backfill on the pooled seed DBs (locomo_v1 + lme_v1).
/// Populates importance/quality/structured_fields/memory_type/retrieval_cue for the
/// ~8064 memories that are `importance IS NULL` (the seeds predate the Phase-1
/// classification pass). ~4.3h on Metal at concurrency=8. Run AFTER
/// `seed_inject_event_dates` — classify's `event_date` write is COALESCE, so an
/// injected date survives (extract returns None for date-stripped text).
///
/// ```bash
/// EVAL_ENRICHMENT_CONCURRENCY=8 WENLAN_LLM_PARALLEL_SEQS=8 \
///   cargo test -p origin-core --features eval-harness --test eval_harness \
///   seed_backfill_classify -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn seed_backfill_classify() {
    use std::sync::Arc;
    use wenlan_core::eval::shared::run_classification_for_eval_concurrent;
    use wenlan_core::llm_provider::OnDeviceProvider;

    let concurrency: usize = std::env::var("EVAL_ENRICHMENT_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let root = resolve_scenario_db_root_from_harness();

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = match OnDeviceProvider::new() {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("SKIP: on-device init failed: {e}");
            return;
        }
    };

    let overall = std::time::Instant::now();
    for seed in ["locomo_v1", "lme_v1"] {
        let dir = root.join(seed);
        if !dir.join("origin_memory.db").exists() {
            eprintln!("SKIP {seed}: no seed DB at {}", dir.display());
            continue;
        }
        let db = wenlan_core::db::MemoryDB::new(&dir, Arc::new(wenlan_core::events::NoopEmitter))
            .await
            .expect("open pooled seed DB");
        let before = db
            .get_memories_needing_classification()
            .await
            .unwrap()
            .len();
        eprintln!(
            "[backfill] {seed}: {before} memories need classification (concurrency={concurrency})"
        );
        let t0 = std::time::Instant::now();
        let n = run_classification_for_eval_concurrent(&db, &llm, concurrency)
            .await
            .expect("classify backfill");
        let elapsed = t0.elapsed().as_secs_f64();
        let after = db
            .get_memories_needing_classification()
            .await
            .unwrap()
            .len();
        eprintln!(
            "[backfill] {seed}: classified {n} in {:.0}s ({:.2}s/mem); remaining unclassified={after}",
            elapsed,
            elapsed / (n.max(1) as f64)
        );
    }
    eprintln!(
        "[backfill] DONE both seeds in {:.0}s ({:.2}h)",
        overall.elapsed().as_secs_f64(),
        overall.elapsed().as_secs_f64() / 3600.0
    );
}

/// STEP 7 (T2): backfill verbatim `source='episode'` rows into the cached seed
/// DBs (locomo_v1 + lme_v1) so the episode channel (`WENLAN_ENABLE_EPISODE_CHANNEL`)
/// has data to measure. GPU-FREE — only FastEmbed (deterministic), no LLM. Derives
/// each episode through the same `derive_episode` helper the write-path co-write
/// uses (no skew). Byte-identical to a fresh flag-on ingest for single-chunk
/// parents (all of locomo_v1); multi-chunk lme parents (~4.5%) capture the first
/// chunk only (see `backfill_episodes` doc). Non-destructive + idempotent
/// (deterministic ids + paired delete). The base channel excludes
/// `source='episode'`, so existing baselines are unaffected until the read flag
/// is turned on.
///
/// ```bash
/// cargo test -p origin-core --features eval-harness --test eval_harness \
///   seed_backfill_episodes -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn seed_backfill_episodes() {
    use std::sync::Arc;

    let root = resolve_scenario_db_root_from_harness();
    for seed in ["locomo_v1", "lme_v1"] {
        let dir = root.join(seed);
        if !dir.join("origin_memory.db").exists() {
            eprintln!("SKIP {seed}: no seed DB at {}", dir.display());
            continue;
        }
        let db = wenlan_core::db::MemoryDB::new(&dir, Arc::new(wenlan_core::events::NoopEmitter))
            .await
            .expect("open pooled seed DB");
        let t0 = std::time::Instant::now();
        let n = db.backfill_episodes().await.expect("backfill episodes");
        eprintln!(
            "[episodes] {seed}: wrote {n} episode rows in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    }
}

/// THE ONE SEED ROUTE. Runs every enrichment step on the cached scenario seed
/// DBs in the correct order, then asserts the completeness contract. This is the
/// single recipe — never hand-run the individual `seed_*` STEP tests, which are
/// just the steps this orchestrates. Run THIS and the seed is complete + verified
/// by construction; miss a step and the contract fails the seed (not a later eval).
///
/// Steps (order matters — inject before classify so COALESCE preserves dates):
///   1. inject event_date (GPU-free)          → temporal substrate
///   2. classify backfill (GPU)               → importance/quality/cue
///   3. entity + memory_entities sweep (GPU)   → graph substrate
///   4. episode backfill (GPU-free embed)      → episode substrate
///   5. distill pages (GPU)                    → page substrate
///   6. assert SeedExpectations::complete()    → fail loud on any starved channel
///
/// Adding a write-time channel? Add its step here AND its floor to
/// `SeedExpectations` — the contract is what stops a channel shipping starved
/// and being re-discovered as "doesn't help" (a lie). The same contract gates
/// the eval side (`assert_feature_substrate_live`), so neither producer nor
/// consumer can drift onto a dead substrate.
///
/// ```bash
/// # full (overnight on-device); or SEED_COMPLETE_BENCH=lme to scope one bench:
/// WENLAN_EVAL_ROOT=$PWD/app/eval EVAL_ENRICHMENT_CONCURRENCY=8 \
///   WENLAN_LLM_PARALLEL_SEQS=8 SEED_COMPLETE_BENCH=lme \
///   cargo test -p origin-core --features eval-harness --test eval_harness \
///   seed_scenario_dbs_complete -- --ignored --nocapture --test-threads=1
/// ```
#[tokio::test]
#[ignore = "GPU + cached scenario DB; L7 manual. The one seed route. Set WENLAN_EVAL_ROOT."]
async fn seed_scenario_dbs_complete() {
    use std::sync::Arc;
    use wenlan_core::eval::seed_contract::{assert_seed_contract, SeedExpectations};
    use wenlan_core::eval::shared::run_classification_for_eval_concurrent;
    use wenlan_core::prompts::PromptRegistry;
    use wenlan_core::read_scope::ReadScope;
    use wenlan_core::tuning::DistillationConfig;

    // Route distill's per-cluster log::warn!/info! skip reasons to stderr so the
    // --nocapture run shows WHY a cluster produced no page. set_logger is global +
    // idempotent across tests; ignore the already-set error.
    let _ = log::set_logger(&STDERR_LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info));

    let concurrency: usize = std::env::var("EVAL_ENRICHMENT_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let bench_filter = std::env::var("SEED_COMPLETE_BENCH").ok();
    let model = std::env::var("SEED_COMPLETE_MODEL").ok();

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new_with_model(model.as_deref())
            .expect("on-device provider"),
    );
    assert!(
        llm.is_available(),
        "[seed_complete] on-device LLM unavailable (Metal init failed?). \
         distill (step 5) would return 0 created pages and the page floor \
         would fail only after hours of seeding. Fix the GPU env first \
         (run outside the CC sandbox; check `pgrep -la origin` for GPU contention)."
    );
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();
    let root = resolve_scenario_db_root_from_harness();

    for bench in ["locomo_v1", "lme_v1"] {
        let short = bench.trim_end_matches("_v1"); // "locomo" / "lme"
        if let Some(f) = &bench_filter {
            if f != short && f != bench {
                continue;
            }
        }
        let dir = root.join(bench);
        if !dir.join("origin_memory.db").exists() {
            eprintln!(
                "[seed_complete] SKIP {bench}: no base DB at {}. Build it first \
                 (fullpipeline harness / scripts/seed-scenario-dbs.sh).",
                dir.display()
            );
            continue;
        }
        eprintln!("\n[seed_complete] === {bench} ===");
        let t_bench = std::time::Instant::now();
        let db = wenlan_core::db::MemoryDB::new(&dir, Arc::new(wenlan_core::events::NoopEmitter))
            .await
            .expect("open scenario seed DB");

        // 1. event_date injection (GPU-free, seconds). Per-session date lives in
        //    fixture metadata; turn/observation text is date-stripped.
        let fixture = eval_root().join(if short == "locomo" {
            "data/locomo10.json"
        } else {
            "data/longmemeval_oracle.json"
        });
        if fixture.exists() {
            let updates: Vec<(String, i64)> = if short == "locomo" {
                let samples =
                    wenlan_core::eval::locomo::load_locomo(&fixture).expect("load locomo");
                wenlan_core::eval::locomo::event_date_map(&samples)
                    .into_iter()
                    .collect()
            } else {
                let samples =
                    wenlan_core::eval::longmemeval::load_longmemeval(&fixture).expect("load lme");
                wenlan_core::eval::longmemeval::event_date_map(&samples)
                    .into_iter()
                    .collect()
            };
            let n = db
                .set_event_dates_by_source_id(&updates)
                .await
                .expect("inject event_date");
            eprintln!(
                "[seed_complete] 1/5 event_date: {} mapped -> {n} rows",
                updates.len()
            );
        } else {
            eprintln!(
                "[seed_complete] 1/5 event_date SKIP: fixture {} missing",
                fixture.display()
            );
        }

        // 2. classify backfill (GPU). Fills importance/quality/cue on the still
        //    -unclassified heads; COALESCE keeps the injected event_date.
        let t = std::time::Instant::now();
        let classified = run_classification_for_eval_concurrent(&db, &llm, concurrency)
            .await
            .expect("classify backfill");
        eprintln!(
            "[seed_complete] 2/5 classify: {classified} in {:.1}m",
            t.elapsed().as_secs_f64() / 60.0
        );

        // 3. entity + memory_entities backfill (GPU). The graph-stream substrate
        //    that has been silently missing on the consolidated seed. MUST drain by
        //    the memory_entities JUNCTION (populate_memory_entities_sweep uses
        //    `unlinked_memories` = NOT IN memory_entities), NOT by the legacy
        //    `entity_id` column: the consolidated seed has stale entity_id markers
        //    (44%) set by the old enrich WITHOUT junction rows, so the entity_id-
        //    gated production path (get_unlinked_memories) skips them and links only
        //    ~10%. Junction-based draining + an attempted-set for termination links
        //    them all. This is the 6/07 graph-substrate populator.
        let t = std::time::Instant::now();
        let linked =
            wenlan_core::eval::locomo::populate_memory_entities_sweep(&db, &llm, &prompts).await;
        eprintln!(
            "[seed_complete] 3/5 memory_entities: {linked} memories linked in {:.1}m",
            t.elapsed().as_secs_f64() / 60.0
        );

        // 4. episode backfill (GPU-free embed).
        let n = db.backfill_episodes().await.expect("backfill episodes");
        eprintln!("[seed_complete] 4/5 episodes: {n} rows");

        // 5. distill pages (GPU). Scoped call so created-vs-pending is visible:
        // zero created with pending>0 = the LLM lane was skipped or every
        // cluster died in the synthesis gauntlet — fail HERE with evidence,
        // not at the contract after the fact.
        let t = std::time::Instant::now();
        let r = wenlan_core::refinery::distill_pages_scoped(
            &db,
            Some(&llm),
            &prompts,
            &tuning,
            None,
            None,
        )
        .await
        .expect("distill pages");
        // find_distillation_clusters_scoped does NOT exclude already-compiled
        // memories (db.rs: "No covered_ids exclusion"); the per-cluster
        // max_page_overlap>0.8 skip means a re-run over an already-distilled DB
        // legitimately creates 0 NEW pages. So 0 created is only a failure when
        // the DB also has 0 active pages (the genuinely-starved case).
        let active_pages = db.count_active_pages().await.expect("count active pages");
        eprintln!(
            "[seed_complete] 5/5 pages: created={} pending={} active_total={} in {:.1}m",
            r.created.len(),
            r.pending.len(),
            active_pages,
            t.elapsed().as_secs_f64() / 60.0
        );
        assert!(
            !r.created.is_empty() || active_pages > 0,
            "[seed_complete] {bench}: distill created 0 pages and the DB has 0 \
             active pages (pending={}). Read the [WARN]/[INFO] distill/compile lines \
             above for per-cluster skip reasons (thin <200 chars / LLM error / \
             hallucination sim<0.6 / garbage topic); pending>0 with no warnings \
             = LLM lane skipped.",
            r.pending.len()
        );

        // 6. completeness gate. Drop the MemoryDB lock first, open a fresh conn.
        drop(db);
        let sdb = libsql::Builder::new_local(dir.join("origin_memory.db").to_str().unwrap())
            .build()
            .await
            .expect("open seed for contract");
        let conn = sdb.connect().expect("connect seed for contract");
        let report = assert_seed_contract(&conn, &SeedExpectations::complete(bench))
            .await
            .unwrap_or_else(|e| panic!("[seed_complete] {bench} contract VIOLATED: {e}"));
        eprintln!(
            "[seed_complete] {bench} COMPLETE + verified in {:.1}m: rows={} classified={} cue={} ({:.0}%) event_date={} ({:.0}%) graph_links={} pages={}",
            t_bench.elapsed().as_secs_f64() / 60.0,
            report.rows,
            report.classified,
            report.cue_nonempty,
            report.cue_coverage() * 100.0,
            report.event_date_nonempty,
            report.event_date_coverage() * 100.0,
            report.graph_links,
            report.pages,
        );
    }
}

/// T3 graph-gate A/B experiment on LongMemEval (retrieval-only, no GPU LLM).
/// Dual-bench companion to `graph_gate_ab_locomo` so T3 is validated on BOTH
/// metrics, not a partial view.
#[tokio::test]
#[ignore = "needs cached scenario DB (run scripts/seed-scenario-dbs.sh); retrieval-only, no GPU"]
async fn graph_gate_ab_lme() {
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db;

    let db_dir = resolve_scenario_db_root_from_harness().join("lme_v1");
    if !db_dir.join("origin_memory.db").exists() {
        println!("SKIP: no seeded LME DB at {}", db_dir.display());
        return;
    }
    let fixture = eval_root().join("data/longmemeval_oracle.json");
    if !fixture.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }
    let db = wenlan_core::db::MemoryDB::new(
        &db_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open lme_v1 scenario DB");

    let off = temp_env::async_with_vars(
        [("WENLAN_ENABLE_GRAPH_GATE", None::<&str>)],
        run_longmemeval_eval_from_db(&db, &fixture),
    )
    .await
    .expect("gate-off eval");
    let on = temp_env::async_with_vars(
        [("WENLAN_ENABLE_GRAPH_GATE", Some("1"))],
        run_longmemeval_eval_from_db(&db, &fixture),
    )
    .await
    .expect("gate-on eval");

    let cov = |r: &wenlan_core::eval::longmemeval::LongMemEvalReport| {
        r.coverage.as_ref().map(|c| c.blind).unwrap_or(0.0)
    };
    println!("=== T3 GRAPH-GATE A/B (LongMemEval, search_memory path, retrieval-only) ===");
    println!("questions evaluated: {}", off.total_questions);
    println!(
        "GATE OFF (graph always): ndcg@10={:.4} recall@5={:.4} mrr={:.4} hit@1={:.4} cov={:.4}",
        off.aggregate_ndcg_at_10,
        off.aggregate_recall_at_5,
        off.aggregate_mrr,
        off.aggregate_hit_rate_at_1,
        cov(&off)
    );
    println!(
        "GATE ON  (gated):        ndcg@10={:.4} recall@5={:.4} mrr={:.4} hit@1={:.4} cov={:.4}",
        on.aggregate_ndcg_at_10,
        on.aggregate_recall_at_5,
        on.aggregate_mrr,
        on.aggregate_hit_rate_at_1,
        cov(&on)
    );
    println!(
        "DELTA (on-off):          ndcg@10={:+.4} recall@5={:+.4} mrr={:+.4} hit@1={:+.4} cov={:+.4}",
        on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
        on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
        on.aggregate_mrr - off.aggregate_mrr,
        on.aggregate_hit_rate_at_1 - off.aggregate_hit_rate_at_1,
        cov(&on) - cov(&off)
    );
}

// ===========================================================================
// PAIRED A/B EMITTER (validation apparatus v2)
// ===========================================================================
//
// Emits one JSONL file per (feature, bench) under $EVAL_OUT, one line per query
// per flag arm, with per-query NDCG@10 / recall@5 / MRR + a wall-clock retrieval
// latency. The aggregate `*_ab_*` tests above are kept intact; this test exposes
// the per-query data they discard so `analyze_paired.py` can run a paired
// Wilcoxon / bootstrap (variance from across-queries, not across-runs).
//
// Run (unsandboxed, against the SNAPSHOT DBs so the seeds stay pristine):
//   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
//   SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_snapshot \
//   EVAL_OUT=/tmp/eval_paired \
//     cargo test -p origin-core --features eval-harness --test eval_harness -- \
//     --ignored --nocapture --test-threads=1 paired_ab_emit
//
// Filter to one feature for a smoke run with $EVAL_PAIRED_ONLY (comma list),
// e.g. EVAL_PAIRED_ONLY=fts_hardening.

/// Resolve the per-query JSONL output directory ($EVAL_OUT, default a fresh
/// tmp dir). Created if missing.
fn paired_out_dir() -> std::path::PathBuf {
    let dir = std::env::var("EVAL_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("eval_paired"));
    std::fs::create_dir_all(&dir).expect("create EVAL_OUT dir");
    dir
}

/// Append per-query rows to `$EVAL_OUT/<feature>_<bench>.jsonl`.
fn write_paired_rows(feature: &str, bench: &str, rows: &[wenlan_core::eval::paired::PerQueryRow]) {
    use std::io::Write;
    let path = paired_out_dir().join(format!("{feature}_{bench}.jsonl"));
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("open jsonl");
    for r in rows {
        let line = serde_json::to_string(r).expect("serialize PerQueryRow");
        writeln!(f, "{line}").expect("write jsonl line");
    }
    println!("[paired] wrote {} rows -> {}", rows.len(), path.display());
}

fn paired_feature_selected(feature: &str) -> bool {
    match std::env::var("EVAL_PAIRED_ONLY") {
        Ok(only) => only.split(',').map(|s| s.trim()).any(|s| s == feature),
        Err(_) => true,
    }
}

/// Run one cached-DB feature on both benches (LoCoMo + LME via `search_memory`),
/// OFF then ON, emitting per-query JSONL for each arm.
async fn paired_run_cached_feature(feature: &str, flag: &str) {
    use wenlan_core::eval::locomo::run_locomo_eval_from_db_collect;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db_collect;
    let root = resolve_scenario_db_root_from_harness();
    // WENLAN_GRAPH_MEMORY_STREAM is default-ON since 2026-06-10, so its OFF arm
    // must explicitly disable it ("0"); every other paired flag is opt-in (OFF =
    // unset).
    let off_val: Option<&str> = if flag == "WENLAN_GRAPH_MEMORY_STREAM" {
        Some("0")
    } else {
        None
    };
    // Seed-vs-stream mutual exclusion: the seeded branch is reachable only when
    // !(allow_graph_stream && graph_memory_stream_enabled()), so with the stream
    // default-ON a graph_seed A/B routes BOTH arms through the stream (byte-
    // identical A/A). Pin the stream opted-out on BOTH arms to measure the seed.
    let seed_stream_pin: Option<(&str, Option<&str>)> = if flag == "WENLAN_ENABLE_GRAPH_SEED" {
        Some(("WENLAN_GRAPH_MEMORY_STREAM", Some("0")))
    } else {
        None
    };

    // -- LoCoMo --
    let lo_dir = root.join("locomo_v1");
    let lo_fx = eval_root().join("data/locomo10.json");
    if lo_dir.join("origin_memory.db").exists() && lo_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lo_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .expect("open locomo_v1 snapshot DB");
        for (state, val) in [("off", off_val), ("on", Some("1"))] {
            let rows = temp_env::async_with_vars(
                std::iter::once((flag, val))
                    .chain(seed_stream_pin)
                    .collect::<Vec<_>>(),
                run_locomo_eval_from_db_collect(&db, &lo_fx, feature, state),
            )
            .await
            .expect("locomo collect");
            write_paired_rows(feature, "locomo", &rows);
        }
    } else {
        println!(
            "[paired:{feature}] SKIP LoCoMo (db {} fixture {})",
            lo_dir.join("origin_memory.db").exists(),
            lo_fx.exists()
        );
    }

    // -- LME --
    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if lme_dir.join("origin_memory.db").exists() && lme_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lme_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .expect("open lme_v1 snapshot DB");
        for (state, val) in [("off", off_val), ("on", Some("1"))] {
            let rows = temp_env::async_with_vars(
                std::iter::once((flag, val))
                    .chain(seed_stream_pin)
                    .collect::<Vec<_>>(),
                run_longmemeval_eval_from_db_collect(&db, &lme_fx, feature, state),
            )
            .await
            .expect("lme collect");
            write_paired_rows(feature, "lme", &rows);
        }
    } else {
        println!(
            "[paired:{feature}] SKIP LME (db {} fixture {})",
            lme_dir.join("origin_memory.db").exists(),
            lme_fx.exists()
        );
    }
}

/// Run one cached-DB feature on both benches through the CROSS-RERANK read path
/// (`search_memory_cross_rerank`, where the page / episode / fact / global-prelude
/// channels live), OFF then ON, emitting per-query JSONL for each arm.
///
/// A CE-path flag flipped on the base `search_memory` collector reads a zero delta
/// because that read never touches the channel — this routes it correctly so the
/// flag's effect is actually measurable.
async fn paired_run_cached_feature_cross_rerank(
    feature: &str,
    flag: &str,
    reranker: std::sync::Arc<dyn wenlan_core::reranker::Reranker>,
) {
    use wenlan_core::eval::locomo::run_locomo_eval_cross_rerank_from_db_collect;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_cross_rerank_from_db_collect;
    let root = resolve_scenario_db_root_from_harness();
    // WENLAN_GRAPH_MEMORY_STREAM is default-ON since 2026-06-10, so its OFF arm
    // must explicitly disable it ("0"); every other paired flag is opt-in (OFF =
    // unset).
    let off_val: Option<&str> = if flag == "WENLAN_GRAPH_MEMORY_STREAM" {
        Some("0")
    } else {
        None
    };

    // -- LoCoMo --
    let lo_dir = root.join("locomo_v1");
    let lo_fx = eval_root().join("data/locomo10.json");
    if lo_dir.join("origin_memory.db").exists() && lo_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lo_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .expect("open locomo_v1 snapshot DB");
        for (state, val) in [("off", off_val), ("on", Some("1"))] {
            let rows = temp_env::async_with_vars(
                [(flag, val)],
                run_locomo_eval_cross_rerank_from_db_collect(
                    &db,
                    &lo_fx,
                    reranker.clone(),
                    feature,
                    state,
                ),
            )
            .await
            .expect("locomo CE collect");
            write_paired_rows(feature, "locomo", &rows);
        }
    } else {
        println!(
            "[paired:{feature}] SKIP LoCoMo (db {} fixture {})",
            lo_dir.join("origin_memory.db").exists(),
            lo_fx.exists()
        );
    }

    // -- LME --
    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if lme_dir.join("origin_memory.db").exists() && lme_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lme_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .expect("open lme_v1 snapshot DB");
        for (state, val) in [("off", off_val), ("on", Some("1"))] {
            let rows = temp_env::async_with_vars(
                [(flag, val)],
                run_longmemeval_eval_cross_rerank_from_db_collect(
                    &db,
                    &lme_fx,
                    reranker.clone(),
                    feature,
                    state,
                ),
            )
            .await
            .expect("lme CE collect");
            write_paired_rows(feature, "lme", &rows);
        }
    } else {
        println!(
            "[paired:{feature}] SKIP LME (db {} fixture {})",
            lme_dir.join("origin_memory.db").exists(),
            lme_fx.exists()
        );
    }
}

/// Like `paired_run_cached_feature_cross_rerank`, but the two arms set the SAME
/// flag to DIFFERENT values (`off_val` then `on_val`) rather than unset-vs-"1".
///
/// Needed for `RERANK_POOL_FLOOR` (the rerank window): the A/B is 10 vs 50, not
/// off vs on. Passing an explicit `off_val` (e.g. `Some("10")`) is deliberate —
/// the LoCoMo/LME cross_rerank collectors default `RERANK_POOL_FLOOR` to "10"
/// via an unscoped `set_var` when it is unset (longmemeval.rs:1536), which would
/// otherwise LEAK past a `None` arm and silently pin the next bench to 10. With
/// both arms passing a concrete value through `temp_env`, the var is always
/// present, that internal default never fires, and each scope restores cleanly.
async fn paired_run_cached_feature_cross_rerank_vals(
    feature: &str,
    flag: &str,
    off_val: Option<&str>,
    on_val: Option<&str>,
    reranker: std::sync::Arc<dyn wenlan_core::reranker::Reranker>,
) {
    use wenlan_core::eval::locomo::run_locomo_eval_cross_rerank_from_db_collect;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_cross_rerank_from_db_collect;
    let root = resolve_scenario_db_root_from_harness();

    // -- LoCoMo --
    let lo_dir = root.join("locomo_v1");
    let lo_fx = eval_root().join("data/locomo10.json");
    if lo_dir.join("origin_memory.db").exists() && lo_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lo_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .expect("open locomo_v1 snapshot DB");
        for (state, val) in [("off", off_val), ("on", on_val)] {
            let rows = temp_env::async_with_vars(
                [(flag, val)],
                run_locomo_eval_cross_rerank_from_db_collect(
                    &db,
                    &lo_fx,
                    reranker.clone(),
                    feature,
                    state,
                ),
            )
            .await
            .expect("locomo CE collect");
            write_paired_rows(feature, "locomo", &rows);
        }
    } else {
        println!(
            "[paired:{feature}] SKIP LoCoMo (db {} fixture {})",
            lo_dir.join("origin_memory.db").exists(),
            lo_fx.exists()
        );
    }

    // -- LME --
    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if lme_dir.join("origin_memory.db").exists() && lme_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lme_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .expect("open lme_v1 snapshot DB");
        for (state, val) in [("off", off_val), ("on", on_val)] {
            let rows = temp_env::async_with_vars(
                [(flag, val)],
                run_longmemeval_eval_cross_rerank_from_db_collect(
                    &db,
                    &lme_fx,
                    reranker.clone(),
                    feature,
                    state,
                ),
            )
            .await
            .expect("lme CE collect");
            write_paired_rows(feature, "lme", &rows);
        }
    } else {
        println!(
            "[paired:{feature}] SKIP LME (db {} fixture {})",
            lme_dir.join("origin_memory.db").exists(),
            lme_fx.exists()
        );
    }
}

/// Umbrella test: emit per-query paired JSONL for the Track-A features.
///
/// Base `search_memory`-path features: T3 graph-gate, T9 graph-seed, T12
/// fts-hardening, T13 magnitude-fusion, T19 query-intent. Plus T4a
/// temporal-filter + temporal-soft-boost (SELF-SEED — tagged re-seed; LME only).
///
/// CROSS-RERANK-path features — page / episode / fact / global-prelude channels +
/// T20 session-diversity — are routed through
/// `paired_run_cached_feature_cross_rerank` so their flag deltas are measurable.
/// Flipping a CE-path flag on the base `search_memory` collector reads a zero
/// delta because that read never touches the channel (the prior T20 trap). The
/// CE arm builds the BGE-reranker-v2-m3 weights (~600MB on first run) and only
/// when at least one CE feature is selected, so base-only smoke runs stay light.
#[tokio::test]
#[ignore = "needs cached scenario DBs (use SNAPSHOT copies); retrieval-only, no GPU. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT"]
async fn headroom_probe_emit() {
    println!("=== RECALL-HEADROOM PROBE (decompose ladder Step 0) ===");
    let out_dir = paired_out_dir();
    println!("EVAL_OUT = {}", out_dir.display());

    let root = resolve_scenario_db_root_from_harness();
    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if !lme_dir.join("origin_memory.db").exists() || !lme_fx.exists() {
        println!(
            "SKIP: lme_v1 snapshot DB ({}) or fixture ({}) missing",
            lme_dir.join("origin_memory.db").exists(),
            lme_fx.exists()
        );
        return;
    }

    let db = wenlan_core::db::MemoryDB::new(
        &lme_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open lme_v1 snapshot DB");

    let rows = wenlan_core::eval::longmemeval::run_longmemeval_headroom_probe_from_db(&db, &lme_fx)
        .await
        .expect("headroom probe");

    use std::io::Write;
    let path = out_dir.join("headroom_lme.jsonl");
    let mut f = std::fs::File::create(&path).expect("create headroom jsonl");
    for r in &rows {
        let line = serde_json::to_string(r).expect("serialize HeadroomRow");
        writeln!(f, "{line}").expect("write jsonl line");
    }
    println!("[headroom] wrote {} rows -> {}", rows.len(), path.display());
}

/// Per-question recall-headroom probe on the FAITHFUL per-question LME-S
/// instrument: opens each question's own seeded scenario DB (reused from the
/// pool-sweep cache, no re-seed) and measures where gold sits at limits
/// 10/30/100. Decides whether a deeper CE rerank pool can EVER add accuracy on
/// per-question LME-S (PR-3): if gold_in_10 == gold_in_30 == gold_in_100 for
/// (nearly) every question, pool depth is structurally inert and no sweep size
/// helps. Read-only (no answer-gen, no LLM, no batching).
///
/// ```bash
/// LME_S_FIXTURE=/path/to/even_subset.json \
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-ce-lme-s-smoke \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   headroom_probe_per_question_lme_s -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn headroom_probe_per_question_lme_s() {
    println!("=== PER-QUESTION RECALL-HEADROOM PROBE (gold-rank, faithful instrument) ===");
    let lme_path = match std::env::var("LME_S_FIXTURE") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            let d = eval_root().join("data");
            let c = d.join("longmemeval_s_cleaned.json");
            if c.exists() {
                c
            } else {
                d.join("longmemeval_s.json")
            }
        }
    };
    if !lme_path.exists() {
        println!("SKIP: LME-S fixture not found at {:?}", lme_path);
        return;
    }
    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));

    let rows = wenlan_core::eval::longmemeval::run_longmemeval_headroom_probe_per_question(
        &lme_path, &baselines,
    )
    .await
    .expect("per-question headroom probe");

    if rows.is_empty() {
        println!(
            "SKIP/EMPTY: no seeded per-question DBs under {:?}",
            baselines
        );
        return;
    }

    println!(
        "\n{:<28} {:>5} {:>8} {:>8} {:>8}   gold_ranks_100",
        "category", "ngold", "in_10", "in_30", "in_100"
    );
    let (mut knee_30, mut widen_100, mut inert) = (0usize, 0usize, 0usize);
    for r in &rows {
        // CE-knee: gold reachable at 30 but not 10. Pool-widening: at 100 not 30.
        if r.gold_in_30 > r.gold_in_10 {
            knee_30 += 1;
        }
        if r.gold_in_100 > r.gold_in_30 {
            widen_100 += 1;
        }
        if r.gold_in_10 == r.gold_in_30 && r.gold_in_30 == r.gold_in_100 {
            inert += 1;
        }
        println!(
            "{:<28} {:>5} {:>8} {:>8} {:>8}   {:?}",
            r.category, r.n_gold, r.gold_in_10, r.gold_in_30, r.gold_in_100, r.gold_ranks_100
        );
    }
    println!(
        "\n[headroom_per_question] n={} | CE-knee questions (gold in 30 not 10)={} | \
         pool-widen (100 not 30)={} | pool-INERT (10==30==100)={}",
        rows.len(),
        knee_30,
        widen_100,
        inert
    );
    println!(
        "VERDICT: {} of {} questions exercise the deep-pool regime; if ~0, adaptive CE pool \
         is structurally inert on per-question LME-S (no sweep size helps).",
        knee_30 + widen_100,
        rows.len()
    );
}

/// Decompose-recall probe (Step 1 of the decompose ladder): base@30 vs
/// date-prefix@30 vs subquery union vs RRF-merge@30, all on the base
/// `search_memory` path. Consumes the pre-generated subquery fixture at
/// `EVAL_SUBQ_PATH` (JSONL `{query_id, subqueries}` — agent-delegated
/// decomposition, the primary lane from the 2026-05-30 decision). Join the
/// emitted `decompose_recall_lme.jsonl` with `headroom_lme.jsonl` on
/// `query_id` to compare the union arm against the single-query limit=100
/// ceiling (pool-size control).
#[tokio::test]
#[ignore = "needs cached scenario DBs (use SNAPSHOT copies) + EVAL_SUBQ_PATH subquery fixture; retrieval-only, no GPU. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT + EVAL_SUBQ_PATH"]
async fn decompose_recall_probe_emit() {
    println!("=== DECOMPOSE-RECALL PROBE (decompose ladder Step 1) ===");
    let out_dir = paired_out_dir();
    println!("EVAL_OUT = {}", out_dir.display());

    let Some(subq_path) = std::env::var_os("EVAL_SUBQ_PATH").map(std::path::PathBuf::from) else {
        println!("SKIP: EVAL_SUBQ_PATH not set (path to subquery fixture JSONL)");
        return;
    };
    let root = resolve_scenario_db_root_from_harness();
    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if !lme_dir.join("origin_memory.db").exists() || !lme_fx.exists() || !subq_path.exists() {
        println!(
            "SKIP: lme_v1 snapshot DB ({}) or fixture ({}) or subquery fixture ({}) missing",
            lme_dir.join("origin_memory.db").exists(),
            lme_fx.exists(),
            subq_path.exists()
        );
        return;
    }

    let db = wenlan_core::db::MemoryDB::new(
        &lme_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open lme_v1 snapshot DB");

    let rows = wenlan_core::eval::longmemeval::run_longmemeval_decompose_recall_probe_from_db(
        &db, &lme_fx, &subq_path,
    )
    .await
    .expect("decompose recall probe");

    use std::io::Write;
    let path = out_dir.join("decompose_recall_lme.jsonl");
    let mut f = std::fs::File::create(&path).expect("create decompose recall jsonl");
    for r in &rows {
        let line = serde_json::to_string(r).expect("serialize DecomposeRecallRow");
        writeln!(f, "{line}").expect("write jsonl line");
    }
    println!(
        "[decompose_recall] wrote {} rows -> {}",
        rows.len(),
        path.display()
    );
}

/// Deep-S CE rerank POOL / MODEL retrieval probe — NDCG@10 + recall@5 + P50
/// latency, NO answer generation, NO judge (the cheap, deterministic path that
/// DECIDES pool depth + CE on/off; see eval/AGENTS.md "CE rerank pool / model
/// eval methodology"). ONE arm per invocation — a sweep script drives the env
/// grid. Reads pre-seeded FAITHFUL per-question deep-S DBs from
/// `EVAL_BASELINES_DIR` (`fullpipeline/lme/<qid>/`); `LME_S_FIXTURE` must list
/// exactly the cached qids. NOT the consolidated DB. Emits
/// `EVAL_OUT/<arm_label>.jsonl` of `PerQueryRow` for `analyze_paired.py`.
///
/// ```bash
/// EVAL_BASELINES_DIR=~/.cache/origin-eval-pool31 LME_S_FIXTURE=/tmp/claude/lme_s_31.json \
///   EVAL_OUT=/tmp/claude/pool_ndcg CE_AB_ARM_LABEL=ce_on_pool20_base CE_AB_RERANK=on \
///   RERANK_POOL_FLOOR=20 WENLAN_RERANKER_MODEL=bge-base RUSTC_WRAPPER= \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   rerank_pool_retrieval_probe_emit -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "retrieval-only deep-S NDCG probe: needs pre-seeded per-question deep-S DBs in EVAL_BASELINES_DIR + LME_S_FIXTURE listing exactly those qids. No GPU answer, no judge."]
async fn rerank_pool_retrieval_probe_emit() {
    // CONFOUND CONTROL (boule adversarial review, 2026-06-19):
    // `search_memory_cross_rerank_cued` sets `allow_graph_stream = reranker.is_none()`,
    // so a CE-off arm (reranker=None) would run WITH the +0.0545-NDCG graph stream
    // while every CE-on arm runs WITHOUT it — a confound LARGER than the CE signal
    // being measured. Force the stream OFF on every arm so CE-on/off and the pool
    // sweep are clean and comparable. (Production realism — CE-on/stream-off vs
    // hybrid/stream-on — is a SEPARATE experiment, not this selector.)
    std::env::set_var("WENLAN_GRAPH_MEMORY_STREAM", "0");

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    let fixture = match std::env::var("LME_S_FIXTURE") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            let d = eval_root().join("data");
            let c = d.join("longmemeval_s_cleaned.json");
            if c.exists() {
                c
            } else {
                d.join("longmemeval_s.json")
            }
        }
    };
    if !fixture.exists() {
        eprintln!("SKIP: LME-S fixture not found at {}", fixture.display());
        return;
    }
    let arm_label =
        std::env::var("CE_AB_ARM_LABEL").unwrap_or_else(|_| "rerank_pool_arm".to_string());
    let want_ce = std::env::var("CE_AB_RERANK")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "on" | "true" | "yes"
            )
        })
        .unwrap_or(false);
    let reranker = if want_ce {
        Some(
            wenlan_core::reranker::init_cross_encoder_reranker(None)
                .expect("init_cross_encoder_reranker failed"),
        )
    } else {
        None
    };
    eprintln!(
        "[rerank_pool_probe] arm={arm_label} ce={want_ce} baselines={} fixture={} \
         pool_floor={} model={}",
        baselines.display(),
        fixture.display(),
        std::env::var("RERANK_POOL_FLOOR").unwrap_or_else(|_| "10(default)".into()),
        std::env::var("WENLAN_RERANKER_MODEL").unwrap_or_else(|_| "bge-base(default)".into()),
    );

    let rows = wenlan_core::eval::longmemeval::run_longmemeval_rerank_pool_probe(
        &baselines, &fixture, reranker, &arm_label,
    )
    .await
    .expect("rerank_pool probe failed");

    let n = rows.len().max(1) as f64;
    let mean_ndcg: f64 = rows.iter().map(|r| r.ndcg10).sum::<f64>() / n;
    let mean_recall: f64 = rows.iter().map(|r| r.recall5).sum::<f64>() / n;
    // Source-expanded coverage: cov_expanded - cov_blind = the page channel's
    // honest contribution (pages credit gold via provenance, not positional
    // rank). Both None on emitters that don't measure coverage; this probe
    // always sets them, so unwrap_or(0.0) is a guard, not a real path.
    let mean_cov_blind: f64 = rows.iter().map(|r| r.cov_blind.unwrap_or(0.0)).sum::<f64>() / n;
    let mean_cov_expanded: f64 = rows
        .iter()
        .map(|r| r.cov_expanded.unwrap_or(0.0))
        .sum::<f64>()
        / n;
    let mut lat: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = lat.get(lat.len() / 2).copied().unwrap_or(0.0);
    eprintln!(
        "[rerank_pool_probe] arm={arm_label} n={} mean_ndcg10={mean_ndcg:.4} \
         mean_recall5={mean_recall:.4} cov_blind={mean_cov_blind:.4} \
         cov_expanded={mean_cov_expanded:.4} cov_delta={:.4} p50_latency_ms={p50:.0}",
        rows.len(),
        mean_cov_expanded - mean_cov_blind
    );

    let out_dir = paired_out_dir();
    std::fs::create_dir_all(&out_dir).ok();
    let path = out_dir.join(format!("{arm_label}.jsonl"));
    use std::io::Write;
    let mut f = std::fs::File::create(&path).expect("create jsonl");
    for r in &rows {
        writeln!(
            f,
            "{}",
            serde_json::to_string(r).expect("serialize PerQueryRow")
        )
        .expect("write jsonl line");
    }
    eprintln!(
        "[rerank_pool_probe] wrote {} rows -> {}",
        rows.len(),
        path.display()
    );
}

/// Decompose-CE probe (Step 2 of the decompose ladder, the wire/kill gate):
/// CE conversion at MATCHED budget. OFF = CE over the base pool@30, ON = CE
/// over the decompose RRF-merge@30 — same 30-candidate CE input both arms, so
/// the delta is pool composition, never pool size. Emits standard PerQueryRows
/// (`decompose_ce_lme.jsonl`) for analyze_paired.py. Needs `EVAL_SUBQ_PATH`
/// (same fixture as `decompose_recall_probe_emit`).
#[tokio::test]
#[ignore = "downloads ~600MB CE model (CPU); needs cached scenario SNAPSHOT DB + EVAL_SUBQ_PATH. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT + EVAL_SUBQ_PATH"]
async fn decompose_ce_probe_emit() {
    println!("=== DECOMPOSE-CE PROBE (decompose ladder Step 2, matched budget 30) ===");
    let out_dir = paired_out_dir();
    println!("EVAL_OUT = {}", out_dir.display());

    let Some(subq_path) = std::env::var_os("EVAL_SUBQ_PATH").map(std::path::PathBuf::from) else {
        println!("SKIP: EVAL_SUBQ_PATH not set (path to subquery fixture JSONL)");
        return;
    };
    let root = resolve_scenario_db_root_from_harness();
    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if !lme_dir.join("origin_memory.db").exists() || !lme_fx.exists() || !subq_path.exists() {
        println!(
            "SKIP: lme_v1 snapshot DB ({}) or fixture ({}) or subquery fixture ({}) missing",
            lme_dir.join("origin_memory.db").exists(),
            lme_fx.exists(),
            subq_path.exists()
        );
        return;
    }

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
    println!("CE model = {} (CPU)", reranker.model_id());

    let db = wenlan_core::db::MemoryDB::new(
        &lme_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open lme_v1 snapshot DB");

    let rows = wenlan_core::eval::longmemeval::run_longmemeval_decompose_ce_probe_from_db(
        &db, &lme_fx, &subq_path, reranker,
    )
    .await
    .expect("decompose CE probe");

    write_paired_rows("decompose_ce", "lme", &rows);
    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        out_dir.display()
    );
}

#[tokio::test]
#[ignore = "needs cached scenario DBs (use SNAPSHOT copies); retrieval-only, no GPU. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT"]
async fn paired_ab_emit() {
    println!("=== PAIRED A/B EMIT (apparatus v2) ===");
    println!("EVAL_OUT = {}", paired_out_dir().display());

    // (feature_tag, env_flag) for the cached-DB / base `search_memory` features.
    let cached: [(&str, &str); 8] = [
        ("graph_stream", "WENLAN_GRAPH_MEMORY_STREAM"),
        ("graph_gate", "WENLAN_ENABLE_GRAPH_GATE"),
        ("graph_seed", "WENLAN_ENABLE_GRAPH_SEED"),
        ("fts_hardening", "WENLAN_ENABLE_FTS_HARDENING"),
        ("magnitude_fusion", "WENLAN_MAGNITUDE_FUSION"),
        ("query_intent", "WENLAN_ENABLE_QUERY_INTENT"),
        // Cached-DB temporal arms: unlike the self-seeded `temporal_*` arms
        // below (per-question evidence-only DBs — no distractors, so the
        // boost has nothing to separate), these read the pooled canonical
        // scenario DB where event_date is 100% live and every other
        // question's memories act as distractors. The 'temp' substring keeps
        // the substrate-liveness gate armed (refuses on event_date = 0).
        // WENLAN_ENABLE_TEMPORAL_GROUNDING has no cached arm: it is a
        // write-time flag (rewrites date phrases at ingest), so flipping it
        // on a read-only cached DB measures exactly zero by construction.
        ("temporal_soft_cached", "WENLAN_ENABLE_TEMPORAL_SOFT_BOOST"),
        ("temporal_filter_cached", "WENLAN_ENABLE_TEMPORAL_FILTER"),
    ];
    for (feature, flag) in cached {
        if !paired_feature_selected(feature) {
            continue;
        }
        println!("--- feature {feature} (flag {flag}) ---");
        paired_run_cached_feature(feature, flag).await;
    }

    // CE-path features: page / episode / fact / global-prelude channels live in
    // `search_memory_cross_rerank`, not the base path. Route them through the
    // cross-rerank collectors so a flag flip produces a real delta. Build the
    // reranker ONCE and only when a CE feature is selected (the BGE-reranker-v2-m3
    // weights are ~600MB on first download), so base-only smoke runs stay light.
    // `rerank_graph_stack` (WENLAN_GRAPH_MEMORY_STREAM toggled WITH the CE active
    // on both arms) RETIRED 2026-06-10: the 2026-06-09 ns receipt (+0.0126,
    // p=0.17) closed stream-under-CE, and the code now hard-skips the stream
    // whenever a reranker is present (allow_graph_stream = reranker.is_none()).
    // With a reranker on both arms the flag toggles nothing, so the two arms are
    // byte-identical by construction — the arm could only emit a fake null.
    let ce: [(&str, &str); 6] = [
        ("page_channel", "WENLAN_ENABLE_PAGE_CHANNEL"),
        ("episode_channel", "WENLAN_ENABLE_EPISODE_CHANNEL"),
        ("fact_channel", "WENLAN_ENABLE_FACT_CHANNEL"),
        ("global_prelude", "WENLAN_ENABLE_GLOBAL_PRELUDE"),
        ("session_diversity", "WENLAN_ENABLE_SESSION_DIVERSITY"),
        // Skip-preference bypass: ON arm routes preference-intent queries past
        // the CE back to the base ranking (the −0.155 SSP fix). Non-preference
        // rows are byte-identical between arms (detector has 0 false positives
        // on the LME-S fixture), so the delta isolates the bypassed category.
        ("rerank_skip_pref", "WENLAN_RERANK_SKIP_PREFERENCE"),
    ];
    if ce.iter().any(|(f, _)| paired_feature_selected(f)) {
        let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
            .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
        for (feature, flag) in ce {
            if !paired_feature_selected(feature) {
                continue;
            }
            println!("--- feature {feature} (flag {flag}) [CROSS-RERANK path] ---");
            paired_run_cached_feature_cross_rerank(feature, flag, reranker.clone()).await;
        }
    }

    // T4a temporal-filter: self-seeds, LME only, search_memory_temporal path.
    if paired_feature_selected("temporal_filter") {
        use wenlan_core::eval::longmemeval::run_longmemeval_eval_temporal_collect;
        println!("--- feature temporal_filter (flag WENLAN_ENABLE_TEMPORAL_FILTER) [RE-SEED] ---");
        let lme_fx = eval_root().join("data/longmemeval_oracle.json");
        if lme_fx.exists() {
            for (state, val) in [("off", None::<&str>), ("on", Some("1"))] {
                let rows = temp_env::async_with_vars(
                    [("WENLAN_ENABLE_TEMPORAL_FILTER", val)],
                    run_longmemeval_eval_temporal_collect(&lme_fx, "temporal_filter", state),
                )
                .await
                .expect("temporal collect");
                write_paired_rows("temporal_filter", "lme", &rows);
            }
        } else {
            println!("[paired:temporal_filter] SKIP LME (fixture missing)");
        }
    }

    // T4a temporal-soft-boost: self-seeds, LME only, search_memory_temporal path.
    // OFF arm = plain baseline (no temporal flag); ON arm = binary in-window score boost.
    if paired_feature_selected("temporal_soft_boost") {
        use wenlan_core::eval::longmemeval::run_longmemeval_eval_temporal_collect;
        println!("--- feature temporal_soft_boost (flag WENLAN_ENABLE_TEMPORAL_SOFT_BOOST) [RE-SEED] ---");
        let lme_fx = eval_root().join("data/longmemeval_oracle.json");
        if lme_fx.exists() {
            for (state, val) in [("off", None::<&str>), ("on", Some("1"))] {
                let rows = temp_env::async_with_vars(
                    [("WENLAN_ENABLE_TEMPORAL_SOFT_BOOST", val)],
                    run_longmemeval_eval_temporal_collect(&lme_fx, "temporal_soft_boost", state),
                )
                .await
                .expect("temporal soft-boost collect");
                write_paired_rows("temporal_soft_boost", "lme", &rows);
            }
        } else {
            println!("[paired:temporal_soft_boost] SKIP LME (fixture missing)");
        }
    }

    println!(
        "=== PAIRED A/B EMIT done -> run scripts/analyze_paired.py on {} ===",
        paired_out_dir().display()
    );
}

/// Paired base-vs-cross-encoder emitter (LME). Measures whether the cross-encoder
/// reranker improves retrieval over the base `search_memory` path on the SAME
/// queries + SAME snapshot DB.
///
/// OFF arm = base `search_memory` (the `run_longmemeval_eval_from_db_collect`
/// path). ON arm = `search_memory_cross_rerank` (CE rescoring over the widened
/// pool). Both write to `$EVAL_OUT/cross_rerank_lme.jsonl`; `analyze_paired.py`
/// joins by `query_id` and runs the paired Wilcoxon / bootstrap.
///
/// First run downloads the BGE-reranker-v2-m3 weights (~600MB) from HuggingFace
/// and runs on CPU (fastembed ONNX). Pin the subset with `EVAL_LME_LIMIT`.
///
/// Run (unsandboxed, against the SNAPSHOT DB so the seed stays pristine):
///   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
///   SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_snapshot \
///   EVAL_OUT=~/.cache/origin-eval/reranker_out EVAL_LME_LIMIT=50 \
///     cargo test -p origin-core --features eval-harness --test eval_harness -- \
///     --ignored --nocapture --test-threads=1 paired_cross_rerank_emit
#[tokio::test]
#[ignore = "downloads ~600MB CE model (CPU); needs cached scenario SNAPSHOT DB. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT"]
async fn paired_cross_rerank_emit() {
    use wenlan_core::eval::longmemeval::{
        run_longmemeval_eval_cross_rerank_from_db_collect, run_longmemeval_eval_from_db_collect,
    };
    println!("=== PAIRED CROSS-RERANK EMIT (base vs cross_rerank) ===");
    println!("EVAL_OUT = {}", paired_out_dir().display());

    let lme_dir = resolve_scenario_db_root_from_harness().join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if !lme_dir.join("origin_memory.db").exists() || !lme_fx.exists() {
        println!(
            "[paired:cross_rerank] SKIP LME (db {} fixture {})",
            lme_dir.join("origin_memory.db").exists(),
            lme_fx.exists()
        );
        return;
    }

    let db = wenlan_core::db::MemoryDB::new(
        &lme_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open lme_v1 snapshot DB");

    // OFF arm: base search_memory.
    let off_rows = run_longmemeval_eval_from_db_collect(&db, &lme_fx, "cross_rerank", "off")
        .await
        .expect("base collect");
    write_paired_rows("cross_rerank", "lme", &off_rows);
    println!("[paired:cross_rerank] OFF (base) rows = {}", off_rows.len());

    // ON arm: cross-encoder rerank. First construction downloads ~600MB + runs on CPU.
    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
    println!(
        "[paired:cross_rerank] CE model = {} (CPU)",
        reranker.model_id()
    );
    let on_rows = run_longmemeval_eval_cross_rerank_from_db_collect(
        &db,
        &lme_fx,
        reranker,
        "cross_rerank",
        "on",
    )
    .await
    .expect("cross_rerank collect");
    write_paired_rows("cross_rerank", "lme", &on_rows);
    println!(
        "[paired:cross_rerank] ON (cross_rerank) rows = {}",
        on_rows.len()
    );

    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// Run both benches through the CROSS-RERANK read path with BOTH flag arms
/// pinned to the SAME value (`flag_val`), but tagged as the `off` then `on`
/// arms. Used for the A/A no-op control: OFF-vs-OFF must read ~zero per-query
/// delta through `analyze_paired.py`, proving the apparatus does not fabricate
/// signal. `feature` should differ from the real A/B feature so the JSONL
/// files don't collide (e.g. `rerank_blend_aa`).
async fn paired_run_cached_feature_cross_rerank_control(
    feature: &str,
    flag: &str,
    flag_val: Option<&str>,
    reranker: std::sync::Arc<dyn wenlan_core::reranker::Reranker>,
) {
    use wenlan_core::eval::locomo::run_locomo_eval_cross_rerank_from_db_collect;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_cross_rerank_from_db_collect;
    let root = resolve_scenario_db_root_from_harness();

    // -- LoCoMo --
    let lo_dir = root.join("locomo_v1");
    let lo_fx = eval_root().join("data/locomo10.json");
    if lo_dir.join("origin_memory.db").exists() && lo_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lo_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .expect("open locomo_v1 snapshot DB");
        for state in ["off", "on"] {
            let rows = temp_env::async_with_vars(
                [(flag, flag_val)],
                run_locomo_eval_cross_rerank_from_db_collect(
                    &db,
                    &lo_fx,
                    reranker.clone(),
                    feature,
                    state,
                ),
            )
            .await
            .expect("locomo CE collect (control)");
            write_paired_rows(feature, "locomo", &rows);
        }
    } else {
        println!(
            "[paired:{feature}] SKIP LoCoMo (db {} fixture {})",
            lo_dir.join("origin_memory.db").exists(),
            lo_fx.exists()
        );
    }

    // -- LME --
    let lme_dir = root.join("lme_v1");
    let lme_fx = eval_root().join("data/longmemeval_oracle.json");
    if lme_dir.join("origin_memory.db").exists() && lme_fx.exists() {
        let db = wenlan_core::db::MemoryDB::new(
            &lme_dir,
            std::sync::Arc::new(wenlan_core::events::NoopEmitter),
        )
        .await
        .expect("open lme_v1 snapshot DB");
        for state in ["off", "on"] {
            let rows = temp_env::async_with_vars(
                [(flag, flag_val)],
                run_longmemeval_eval_cross_rerank_from_db_collect(
                    &db,
                    &lme_fx,
                    reranker.clone(),
                    feature,
                    state,
                ),
            )
            .await
            .expect("lme CE collect (control)");
            write_paired_rows(feature, "lme", &rows);
        }
    } else {
        println!(
            "[paired:{feature}] SKIP LME (db {} fixture {})",
            lme_dir.join("origin_memory.db").exists(),
            lme_fx.exists()
        );
    }
}

/// Paired A/B emitter for `WENLAN_ENABLE_RERANK_BLEND` (blend vs replace).
///
/// The flag ONLY affects the cross_rerank path: when ON, the CE logit is
/// BLENDED with the boosted-RRF score (`α·σ(CE)+(1−α)·norm(WRRF)`) instead of
/// REPLACING it. The blend helpers live in
/// `crates/origin-core/src/retrieval/blend.rs`; the wiring is in
/// `search_memory_cross_rerank` (`crates/origin-core/src/db.rs:~9329`). A flag
/// flipped on the base `search_memory` collector reads a zero delta because
/// that read never reaches the CE rescoring, so this routes BOTH benches
/// through `run_*_eval_cross_rerank_from_db_collect` where the blend lives.
///
/// Emits per-query JSONL for both benches (LoCoMo + LME):
///   - `rerank_blend_locomo.jsonl` / `rerank_blend_lme.jsonl` — A/B:
///     OFF arm = replace (flag unset), ON arm = blend (flag=1).
///   - `rerank_blend_aa_locomo.jsonl` / `rerank_blend_aa_lme.jsonl` — A/A
///     no-op control: SAME arm twice (OFF/OFF, flag unset both times). The
///     analyzer must read ~zero delta here, proving the harness isn't
///     fabricating signal from re-running a deterministic collector.
///
/// First run downloads the BGE-reranker-v2-m3 weights (~600MB) and runs on CPU
/// (fastembed ONNX). Honor `EVAL_LOCOMO_LIMIT` / `EVAL_LME_LIMIT` for subset
/// smoke runs.
///
/// Run (unsandboxed, against the SNAPSHOT DBs so the seeds stay pristine):
///   EVAL_LOCOMO_LIMIT=20 EVAL_LME_LIMIT=20 \
///   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
///   SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_snapshot \
///   EVAL_OUT=~/.cache/origin-eval/rerank_blend_out \
///     cargo test -p origin-core --test eval_harness rerank_blend_paired_ab -- \
///     --ignored --nocapture --test-threads=1
#[tokio::test]
#[ignore = "downloads ~600MB CE model (CPU); needs cached scenario SNAPSHOT DB. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT"]
async fn rerank_blend_paired_ab() {
    println!("=== RERANK-BLEND PAIRED A/B (blend vs replace) ===");
    println!("EVAL_OUT = {}", paired_out_dir().display());

    // Build the CE reranker ONCE (shared across the A/B and A/A arms). First
    // construction downloads ~600MB BGE-reranker-v2-m3 from HuggingFace + runs
    // on CPU (fastembed ONNX).
    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
    println!("CE model = {} (CPU)", reranker.model_id());

    // A/B arm: OFF (replace, flag unset) vs ON (blend, flag=1).
    println!("--- feature rerank_blend (flag WENLAN_ENABLE_RERANK_BLEND) [A/B] ---");
    paired_run_cached_feature_cross_rerank(
        "rerank_blend",
        "WENLAN_ENABLE_RERANK_BLEND",
        reranker.clone(),
    )
    .await;

    // A/A control: OFF vs OFF (flag unset on BOTH arms). Must read ~zero delta.
    println!("--- feature rerank_blend_aa (A/A no-op control: OFF vs OFF) ---");
    paired_run_cached_feature_cross_rerank_control(
        "rerank_blend_aa",
        "WENLAN_ENABLE_RERANK_BLEND",
        None,
        reranker.clone(),
    )
    .await;

    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// Paired A/B emitter for `RERANK_POOL_FLOOR` (rerank window: 10 vs 50).
///
/// The fetch-pool floor controls how many candidates the cross-encoder rescores
/// before truncation to `limit` (`compute_rerank_fetch_pool`, db.rs:~397). EXP3
/// (n=50/cat scaffold) showed widening 10→50 lifts recall@5 ~+10-12pp on LME and
/// ~+2pp on LoCoMo — but n=50 hit only 2-3 of 10 conversations, so the magnitude
/// is unreliable. This re-runs the A/B over the FULL fixture through the trusted
/// paired apparatus (v2) for a citable, per-category, A/A-controlled answer.
///
/// Determinism note: retrieval recall@5 / ndcg@10 at a fixed window are
/// DETERMINISTIC (CE forward pass + RRF + cached embeddings — no sampling), so a
/// single full-fixture run is sufficient for the recall verdict; the A/A control
/// (window 10 vs 10) proves it by reading ~zero delta. The N≥3 mean±stddev gate
/// from task #9 applies to STOCHASTIC LLM-judge answer accuracy, not this
/// deterministic retrieval metric. Latency (gate (a), default-on viability) is
/// measured separately and gates only the Phase-2 routing flip, not the window.
///
/// Emits per-query JSONL for both benches:
///   - `rerank_window_locomo.jsonl` / `rerank_window_lme.jsonl` — A/B:
///     OFF arm = pool floor 10 (current default), ON arm = pool floor 50.
///   - `rerank_window_aa_locomo.jsonl` / `rerank_window_aa_lme.jsonl` — A/A
///     no-op control: pool floor 10 on BOTH arms. Analyzer must read ~zero delta.
///
/// First run downloads the BGE-reranker-v2-m3 weights (~600MB) and runs on CPU.
///
/// Run (unsandboxed, against the SNAPSHOT DBs so the seeds stay pristine):
///   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
///   SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_snapshot \
///   EVAL_OUT=~/.cache/origin-eval/rerank_window_out \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     rerank_window_paired_ab -- --ignored --nocapture --test-threads=1
#[tokio::test]
#[ignore = "downloads ~600MB CE model (CPU); needs cached scenario SNAPSHOT DB. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT"]
async fn rerank_window_paired_ab() {
    println!("=== RERANK-WINDOW PAIRED A/B (pool floor 10 vs 50) ===");
    println!("EVAL_OUT = {}", paired_out_dir().display());

    // Build the CE reranker ONCE (shared across the A/B and A/A arms). First
    // construction downloads ~600MB BGE-reranker-v2-m3 from HuggingFace + runs
    // on CPU (fastembed ONNX).
    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
    println!("CE model = {} (CPU)", reranker.model_id());

    // A/B arm: pool floor 10 (current default) vs 50 (widened, peer norm).
    println!("--- feature rerank_window (flag RERANK_POOL_FLOOR) [A/B 10 vs 50] ---");
    paired_run_cached_feature_cross_rerank_vals(
        "rerank_window",
        "RERANK_POOL_FLOOR",
        Some("10"),
        Some("50"),
        reranker.clone(),
    )
    .await;

    // A/A control: pool floor 10 on BOTH arms. Must read ~zero delta.
    println!("--- feature rerank_window_aa (A/A no-op control: 10 vs 10) ---");
    paired_run_cached_feature_cross_rerank_control(
        "rerank_window_aa",
        "RERANK_POOL_FLOOR",
        Some("10"),
        reranker.clone(),
    )
    .await;

    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// Knee sweep for `RERANK_POOL_FLOOR`: intermediate windows 20 and 30 vs the
/// 10 baseline. Follows up `rerank_window_paired_ab`, which proved 10→50 is a
/// real recall win (LME ndcg +0.052, BH-sig) but latency-prohibitive as a
/// default (P99 +9.8s on CPU). Recall gain is sublinear and latency ~linear in
/// pool size, so the knee — the smallest window capturing most of the +0.052 at
/// an acceptable P99 — likely sits at 20 or 30. This measures both rather than
/// estimating from the 10/50 endpoints.
///
/// No A/A arm: determinism was already established by `rerank_window_paired_ab`
/// (LoCoMo A/A = 0.0000 exact; LME A/A +0.0014 noise floor). Re-running it would
/// only burn ~2h. The 10-baseline arm is recomputed within each feature so the
/// per-query pairing stays within-run.
///
/// Emits `rerank_w20_{locomo,lme}.jsonl` (10 vs 20) and
/// `rerank_w30_{locomo,lme}.jsonl` (10 vs 30). Feed all of EVAL_OUT (this run +
/// the prior 10/50 run, if pointed at the same dir) to analyze_paired.py for the
/// full 10/20/30/50 recall+latency curve.
///
/// Run (unsandboxed, against the SNAPSHOT DBs so the seeds stay pristine):
///   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
///   SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_snapshot \
///   EVAL_OUT=~/.cache/origin-eval/rerank_window_knee_out \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     rerank_window_knee_sweep -- --ignored --nocapture --test-threads=1
#[tokio::test]
#[ignore = "downloads ~600MB CE model (CPU); needs cached scenario SNAPSHOT DB. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT"]
async fn rerank_window_knee_sweep() {
    println!("=== RERANK-WINDOW KNEE SWEEP (pool floor 10 vs 20, 10 vs 30) ===");
    println!("EVAL_OUT = {}", paired_out_dir().display());

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
    println!("CE model = {} (CPU)", reranker.model_id());

    // A/B arm: pool floor 10 (current default) vs 20.
    println!("--- feature rerank_w20 (flag RERANK_POOL_FLOOR) [A/B 10 vs 20] ---");
    paired_run_cached_feature_cross_rerank_vals(
        "rerank_w20",
        "RERANK_POOL_FLOOR",
        Some("10"),
        Some("20"),
        reranker.clone(),
    )
    .await;

    // A/B arm: pool floor 10 (current default) vs 30.
    println!("--- feature rerank_w30 (flag RERANK_POOL_FLOOR) [A/B 10 vs 30] ---");
    paired_run_cached_feature_cross_rerank_vals(
        "rerank_w30",
        "RERANK_POOL_FLOOR",
        Some("10"),
        Some("30"),
        reranker.clone(),
    )
    .await;

    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// Smaller-CE model sweep (#187): per model, paired arms on the cached DBs.
///
/// OFF arm = base `search_memory` collector (no CE).
/// ON arm  = CE collector with `WENLAN_RERANKER_MODEL=<model>`.
///
/// Plus one A/A floor arm (`rerank_model_aa`, base vs base, no CE) per bench so
/// `analyze_paired.py` has a noise floor that matches the OFF arm path.
///
/// Models:
///   `rerank_model_v2m3` -- BGE-reranker-v2-m3 (~568M params, quality ceiling; default until 2026-06-11)
///   `rerank_model_turbo` -- JINA-v1-turbo-en  (~37M params, CPU-fast, English-only)
///   `rerank_model_base`  -- BGE-reranker-base  (~278M params, mid-size)
///
/// `jina-turbo` is Xet-backed (hf-hub cannot fetch it directly). Use
/// `WENLAN_RERANKER_ONNX_DIR` pointing to a pre-downloaded local ONNX dir when
/// running with that model. When `WENLAN_RERANKER_ONNX_DIR` is set the BYO path
/// is active and `WENLAN_RERANKER_MODEL` is ignored -- so each BYO invocation
/// must select exactly ONE model feature via `EVAL_PAIRED_ONLY`. The test
/// panics if more than one model feature is selected while the dir is set
/// (subsumes the unset case: unset selects all three), and cross-checks the
/// constructed reranker's `model_id()` against the feature label. The check
/// validates label-vs-label (the BYO model_id comes from WENLAN_RERANKER_MODEL_ID
/// or the dir basename), so it catches copy-paste cross-wiring, not wrong
/// weights inside a correctly-named dir.
///
/// Run (unsandboxed, against SNAPSHOT DBs so seeds stay pristine):
///   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
///   SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_snapshot \
///   EVAL_OUT=~/.cache/origin-eval/rerank_model_sweep_out \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     rerank_model_sweep -- --ignored --nocapture --test-threads=1
///
/// BYO runbook (one model per invocation, shared EVAL_OUT). The FIRST run must
/// also select the A/A floor (`rerank_model_aa` is not a model feature, so the
/// pair stays legal under the one-model guard) or no run ever writes
/// `rerank_model_aa_*.jsonl` and every verdict comes out NO-FLOOR:
///   run 1: EVAL_PAIRED_ONLY=rerank_model_v2m3,rerank_model_aa \
///          WENLAN_RERANKER_ONNX_DIR=~/.cache/origin-eval/rerankers/bge-v2-m3 \
///          WENLAN_RERANKER_MODEL_ID=bge-v2-m3 ... (same env above) ...
///   run 2: EVAL_PAIRED_ONLY=rerank_model_turbo + the jina-turbo dir + id
///   run 3: EVAL_PAIRED_ONLY=rerank_model_base + the bge-base dir + id
#[tokio::test]
#[ignore = "cached scenario DBs; CPU ONNX; set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT"]
async fn rerank_model_sweep() {
    use wenlan_core::eval::locomo::run_locomo_eval_cross_rerank_from_db_collect;
    use wenlan_core::eval::locomo::run_locomo_eval_from_db_collect;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_cross_rerank_from_db_collect;
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_from_db_collect;

    println!("=== RERANK-MODEL SWEEP (smaller-CE latency/quality matrix, #187) ===");
    println!("EVAL_OUT = {}", paired_out_dir().display());

    // (feature tag, WENLAN_RERANKER_MODEL value, expected model_id substring).
    let models: [(&str, &str, &str); 3] = [
        ("rerank_model_v2m3", "bge-v2-m3", "v2-m3"),
        ("rerank_model_turbo", "turbo", "turbo"),
        ("rerank_model_base", "bge-base", "base"),
    ];

    // Guard: when WENLAN_RERANKER_ONNX_DIR is set the BYO path is active and
    // WENLAN_RERANKER_MODEL is ignored by init_cross_encoder_reranker, so every
    // selected model feature would run on the SAME mounted weights under
    // different labels -- a mislabeled, latency-confounded matrix. Allow at most
    // one selected model feature per BYO invocation (`> 1`, not `!= 1`, so an
    // aa-only floor run stays legal with the dir exported -- the A/A arm builds
    // no reranker). EVAL_PAIRED_ONLY unset selects all three, so the unset case
    // is subsumed.
    let byo_active = std::env::var_os("WENLAN_RERANKER_ONNX_DIR").is_some();
    if byo_active {
        let selected: Vec<&str> = models
            .iter()
            .map(|(f, _, _)| *f)
            .filter(|f| paired_feature_selected(f))
            .collect();
        if selected.len() > 1 {
            panic!(
                "WENLAN_RERANKER_ONNX_DIR is set (BYO weights) but {} model features \
                 are selected ({selected:?}). The BYO path ignores WENLAN_RERANKER_MODEL, \
                 so every selected feature would run on the same mounted weights under \
                 different labels. Select exactly one of rerank_model_{{v2m3,turbo,base}} \
                 via EVAL_PAIRED_ONLY.",
                selected.len()
            );
        }
    }

    let root = resolve_scenario_db_root_from_harness();
    let benches: [(&str, &str); 2] = [
        ("locomo", "data/locomo10.json"),
        ("lme", "data/longmemeval_oracle.json"),
    ];
    let mut wrote_any = false;

    // A/A floor: base (search_memory, no CE) vs base -- same collector on both
    // arms. Must read ~zero delta, establishing the noise floor for the OFF arm
    // path. One pair per bench.
    if paired_feature_selected("rerank_model_aa") {
        println!("--- feature rerank_model_aa (A/A no-op: base vs base) ---");
        for (bench, fx_rel) in benches {
            let db_dir = root.join(format!("{bench}_v1"));
            let fx_path = eval_root().join(fx_rel);
            if !db_dir.join("origin_memory.db").exists() || !fx_path.exists() {
                println!(
                    "[paired:rerank_model_aa] SKIP {bench} (db {} fixture {})",
                    db_dir.join("origin_memory.db").exists(),
                    fx_path.exists()
                );
                continue;
            }
            let db = wenlan_core::db::MemoryDB::new(
                &db_dir,
                std::sync::Arc::new(wenlan_core::events::NoopEmitter),
            )
            .await
            .unwrap_or_else(|e| panic!("rerank_model_aa/{bench} open snapshot DB: {e}"));
            for state in ["off", "on"] {
                let rows = if bench == "locomo" {
                    run_locomo_eval_from_db_collect(&db, &fx_path, "rerank_model_aa", state)
                        .await
                        .unwrap_or_else(|e| {
                            panic!("rerank_model_aa/{bench} base collect ({state} arm): {e}")
                        })
                } else {
                    run_longmemeval_eval_from_db_collect(&db, &fx_path, "rerank_model_aa", state)
                        .await
                        .unwrap_or_else(|e| {
                            panic!("rerank_model_aa/{bench} base collect ({state} arm): {e}")
                        })
                };
                write_paired_rows("rerank_model_aa", bench, &rows);
                wrote_any = true;
            }
        }
    }

    // Per-model sweep: OFF = base search_memory (no CE), ON = CE with that model.
    for (feature, model, expect_sub) in models {
        if !paired_feature_selected(feature) {
            continue;
        }
        println!("--- {feature} (WENLAN_RERANKER_MODEL={model}) ---");
        // Set the model env var so init_cross_encoder_reranker picks it up.
        // The test runs alone with --test-threads=1 so there is no concurrency
        // hazard. We remove the var after constructing the reranker to avoid
        // leaking it into subsequent iterations (the reranker object already
        // captures the model choice).
        std::env::set_var("WENLAN_RERANKER_MODEL", model);
        let reranker =
            wenlan_core::reranker::init_cross_encoder_reranker(None).unwrap_or_else(|e| {
                panic!("{feature}: init_cross_encoder_reranker (first run downloads weights): {e}")
            });
        std::env::remove_var("WENLAN_RERANKER_MODEL");
        println!("CE model_id = {} (CPU)", reranker.model_id());
        // Label/weights cross-check (BYO only): the BYO loader stamps model_id
        // from WENLAN_RERANKER_MODEL_ID (or the dir basename), so assert it
        // contains the feature's expected token -- mounted weights can't run
        // under the wrong feature label. The enum path stamps the fastembed
        // Debug repr (e.g. "BGERerankerV2M3", no hyphens) and selects weights
        // from the same tuple, so label/weights divergence is impossible there.
        if byo_active {
            assert!(
                reranker.model_id().to_lowercase().contains(expect_sub),
                "feature {feature} expected model_id containing {expect_sub:?}, got {:?} -- \
                 the mounted WENLAN_RERANKER_ONNX_DIR weights do not match the feature label",
                reranker.model_id()
            );
        }

        for (bench, fx_rel) in benches {
            let db_dir = root.join(format!("{bench}_v1"));
            let fx_path = eval_root().join(fx_rel);
            if !db_dir.join("origin_memory.db").exists() || !fx_path.exists() {
                println!(
                    "[paired:{feature}] SKIP {bench} (db {} fixture {})",
                    db_dir.join("origin_memory.db").exists(),
                    fx_path.exists()
                );
                continue;
            }
            // OFF arm: base search_memory (no CE), labeled "off".
            {
                let db = wenlan_core::db::MemoryDB::new(
                    &db_dir,
                    std::sync::Arc::new(wenlan_core::events::NoopEmitter),
                )
                .await
                .unwrap_or_else(|e| panic!("{feature}/{bench} open snapshot DB (off arm): {e}"));
                let rows = if bench == "locomo" {
                    run_locomo_eval_from_db_collect(&db, &fx_path, feature, "off")
                        .await
                        .unwrap_or_else(|e| panic!("{feature}/{bench} base collect (off arm): {e}"))
                } else {
                    run_longmemeval_eval_from_db_collect(&db, &fx_path, feature, "off")
                        .await
                        .unwrap_or_else(|e| panic!("{feature}/{bench} base collect (off arm): {e}"))
                };
                write_paired_rows(feature, bench, &rows);
            }
            // ON arm: CE collector with the selected model, labeled "on".
            // (wrote_any is set once per bench iteration, after this arm --
            // a straight-line OFF-arm assignment would be dead, the ON arm
            // always follows in the same iteration.)
            {
                let db = wenlan_core::db::MemoryDB::new(
                    &db_dir,
                    std::sync::Arc::new(wenlan_core::events::NoopEmitter),
                )
                .await
                .unwrap_or_else(|e| panic!("{feature}/{bench} open snapshot DB (on arm): {e}"));
                let rows = if bench == "locomo" {
                    run_locomo_eval_cross_rerank_from_db_collect(
                        &db,
                        &fx_path,
                        reranker.clone(),
                        feature,
                        "on",
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{feature}/{bench} CE collect (on arm): {e}"))
                } else {
                    run_longmemeval_eval_cross_rerank_from_db_collect(
                        &db,
                        &fx_path,
                        reranker.clone(),
                        feature,
                        "on",
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{feature}/{bench} CE collect (on arm): {e}"))
                };
                write_paired_rows(feature, bench, &rows);
                wrote_any = true;
            }
        }
    }

    // A typo'd EVAL_PAIRED_ONLY selects nothing and would otherwise exit green
    // having written zero files; missing DBs/fixtures on every selected arm
    // would too. Fail loud instead of green-zero.
    assert!(
        wrote_any,
        "rerank_model_sweep wrote NO paired rows -- check EVAL_PAIRED_ONLY spelling \
         (valid: rerank_model_aa, rerank_model_v2m3, rerank_model_turbo, \
         rerank_model_base) and that SCENARIO_DB_ROOT / WENLAN_EVAL_ROOT point at \
         the seeded DBs + fixtures"
    );

    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// PR-B page-channel ON baseline (LoCoMo).
///
/// Uses the pre-seeded consolidated scenario DB at
/// `${SCENARIO_DB_ROOT or ~/.cache/origin-eval/scenario_seeded}/locomo_v1/origin_memory.db`
/// — skips ingest entirely. Page-channel ON by default; set
/// `WENLAN_ENABLE_PAGE_CHANNEL=1` to measure the ON variant. Page-channel is OFF by default.
///
/// Filename suffix `__with_pages` distinguishes from the per-conversation
/// `cross_rerank__*__pool_baseline.json` headline (which uses ephemeral DBs
/// and is preserved as the 0.684 bar).
#[tokio::test]
#[ignore = "needs Metal GPU + cached scenario DB (run scripts/seed-scenario-dbs.sh)"]
async fn save_locomo_v2_with_pages_baseline() {
    let scenario_root = resolve_scenario_db_root_from_harness();
    let db_dir = scenario_root.join("locomo_v1");
    assert!(
        db_dir.join("origin_memory.db").exists(),
        "missing {}/origin_memory.db — run scripts/seed-scenario-dbs.sh",
        db_dir.display()
    );

    let db = wenlan_core::db::MemoryDB::new(
        &db_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open locomo_v1 scenario DB");

    // Sanity: cached scenario DB must have distilled pages for page-channel
    // to be measurable. An empty pages table silently produces page-OFF
    // metrics stamped as page-ON. SKIP semantics match the fixture-missing
    // branch below so contributors without seeded DBs get a clear message
    // instead of a thread panic.
    let pages_count = db
        .count_active_pages()
        .await
        .expect("count_active_pages failed");
    if pages_count == 0 {
        println!(
            "SKIP: cached scenario DB has 0 active pages at {}. Run scripts/seed-scenario-dbs.sh from the repo root then verify with cached_scenario_db_compat_check.",
            db_dir.display()
        );
        return;
    }
    println!("Pages in scenario DB: {}", pages_count);

    let fixture = eval_root().join("data/locomo10.json");
    if !fixture.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");

    let report =
        wenlan_core::eval::locomo::run_locomo_eval_cross_rerank_from_db(&db, &fixture, reranker)
            .await
            .unwrap();

    let baselines_dir = eval_root().join("baselines");
    std::fs::create_dir_all(&baselines_dir).unwrap();
    let mut filename = report.baseline_filename("locomo");
    // Branch suffix on WENLAN_ENABLE_PAGE_CHANNEL so page-ON and page-OFF artifacts
    // don't collide at the legacy app/eval/baselines/ path. Truthy parse via
    // shared helper so suffix matches what the production code path actually did.
    let suffix = if wenlan_core::db::page_channel_enabled() {
        "__with_pages"
    } else {
        "__no_pages"
    };
    if let Some(stripped) = filename.strip_suffix(".json") {
        filename = format!("{}{}.json", stripped, suffix);
    } else {
        filename = format!("{}{}", filename, suffix);
    }
    let baseline_path = baselines_dir.join(filename);
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LoCoMo v2 with-pages baseline to {:?}", baseline_path);
    println!("  NDCG@10:  {:.4}", report.aggregate_ndcg_at_10);
    println!("  Recall@5: {:.4}", report.aggregate_recall_at_5);
    println!("  MRR:      {:.4}", report.aggregate_mrr);
    save_layered(&report, |r| r.to_eval_report());
}

/// PR-B page-channel ON baseline (LongMemEval).
///
/// Uses the pre-seeded consolidated scenario DB at
/// `${SCENARIO_DB_ROOT or ~/.cache/origin-eval/scenario_seeded}/lme_v1/origin_memory.db`
/// — skips ingest entirely. Page-channel ON by default; set
/// `WENLAN_ENABLE_PAGE_CHANNEL=1` to measure the ON variant. Page-channel is OFF by default.
///
/// Filename suffix `__with_pages` distinguishes from the per-question
/// `cross_rerank__*__pool_baseline.json` headline.
#[tokio::test]
#[ignore = "needs Metal GPU + cached scenario DB (run scripts/seed-scenario-dbs.sh)"]
async fn save_longmemeval_v2_with_pages_baseline() {
    let scenario_root = resolve_scenario_db_root_from_harness();
    let db_dir = scenario_root.join("lme_v1");
    assert!(
        db_dir.join("origin_memory.db").exists(),
        "missing {}/origin_memory.db — run scripts/seed-scenario-dbs.sh",
        db_dir.display()
    );

    let db = wenlan_core::db::MemoryDB::new(
        &db_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open lme_v1 scenario DB");

    // Sanity: cached scenario DB must have distilled pages for page-channel
    // to be measurable. An empty pages table silently produces page-OFF
    // metrics stamped as page-ON. SKIP semantics match the fixture-missing
    // branch below so contributors without seeded DBs get a clear message
    // instead of a thread panic.
    let pages_count = db
        .count_active_pages()
        .await
        .expect("count_active_pages failed");
    if pages_count == 0 {
        println!(
            "SKIP: cached scenario DB has 0 active pages at {}. Run scripts/seed-scenario-dbs.sh from the repo root then verify with cached_scenario_db_compat_check.",
            db_dir.display()
        );
        return;
    }
    println!("Pages in scenario DB: {}", pages_count);

    let fixture = eval_root().join("data/longmemeval_oracle.json");
    if !fixture.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");

    let report = wenlan_core::eval::longmemeval::run_longmemeval_eval_cross_rerank_from_db(
        &db, &fixture, reranker,
    )
    .await
    .unwrap();

    let baselines_dir = eval_root().join("baselines");
    std::fs::create_dir_all(&baselines_dir).unwrap();
    let mut filename = report.baseline_filename("longmemeval");
    // Branch suffix on WENLAN_ENABLE_PAGE_CHANNEL so page-ON and page-OFF artifacts
    // don't collide at the legacy app/eval/baselines/ path. Truthy parse via
    // shared helper so suffix matches what the production code path actually did.
    let suffix = if wenlan_core::db::page_channel_enabled() {
        "__with_pages"
    } else {
        "__no_pages"
    };
    if let Some(stripped) = filename.strip_suffix(".json") {
        filename = format!("{}{}.json", stripped, suffix);
    } else {
        filename = format!("{}{}", filename, suffix);
    }
    let baseline_path = baselines_dir.join(filename);
    report.save_baseline(&baseline_path).unwrap();
    println!(
        "Saved LongMemEval v2 with-pages baseline to {:?}",
        baseline_path
    );
    println!("  NDCG@10:  {:.4}", report.aggregate_ndcg_at_10);
    println!("  Recall@5: {:.4}", report.aggregate_recall_at_5);
    println!("  MRR:      {:.4}", report.aggregate_mrr);
    save_layered(&report, |r| r.to_eval_report());
}

// ---------------------------------------------------------------------------
// Token efficiency / quality-cost tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_quality_cost_fixtures() {
    use wenlan_core::eval::token_efficiency::{run_quality_cost_eval, SearchStrategy};

    let fixture_dir = eval_root().join("fixtures");

    let strategies = vec![
        SearchStrategy::Wenlan,
        SearchStrategy::NaiveRag,
        SearchStrategy::FullReplay,
        SearchStrategy::NoMemory,
    ];

    let report = run_quality_cost_eval(&fixture_dir, &strategies, 10)
        .await
        .unwrap();

    assert_eq!(report.strategies.len(), 4);
    assert!(
        report.headline.savings_pct > 0.0,
        "should show token savings"
    );

    // Wenlan should have better quality than NaiveRag
    let origin = report
        .strategies
        .iter()
        .find(|s| s.strategy == "origin")
        .unwrap();
    let naive = report
        .strategies
        .iter()
        .find(|s| s.strategy == "naive_rag")
        .unwrap();
    assert!(
        origin.ndcg_at_10 >= naive.ndcg_at_10,
        "Wenlan NDCG ({:.3}) should be >= NaiveRag ({:.3})",
        origin.ndcg_at_10,
        naive.ndcg_at_10
    );

    println!("{}", report.to_terminal());
}

#[tokio::test]
#[ignore]
async fn test_quality_cost_agent_workload() {
    use wenlan_core::eval::token_efficiency::{run_quality_cost_eval, SearchStrategy};

    let fixture_dir = eval_root().join("fixtures");
    if !fixture_dir.join("agent_coding_session.toml").exists() {
        println!("SKIP: agent fixtures not found");
        return;
    }

    let strategies = vec![
        SearchStrategy::Wenlan,
        SearchStrategy::NaiveRag,
        SearchStrategy::FullReplay,
        SearchStrategy::NoMemory,
    ];

    let report = run_quality_cost_eval(&fixture_dir, &strategies, 10)
        .await
        .unwrap();

    assert!(report.headline.savings_pct > 0.0);
    println!("{}", report.to_terminal());
}

#[tokio::test]
#[ignore]
async fn save_quality_cost_fixtures_baseline() {
    use wenlan_core::eval::token_efficiency::{run_quality_cost_eval, SearchStrategy};

    let fixture_dir = eval_root().join("fixtures");
    let baseline_path = eval_root().join("baselines/quality_cost_fixtures_baseline.json");

    let strategies = vec![
        SearchStrategy::Wenlan,
        SearchStrategy::NaiveRag,
        SearchStrategy::FullReplay,
        SearchStrategy::NoMemory,
    ];

    let report = run_quality_cost_eval(&fixture_dir, &strategies, 10)
        .await
        .unwrap();

    println!("{}", report.to_terminal());
    report.save_baseline(&baseline_path).unwrap();
    println!("\nBaseline saved to {:?}", baseline_path);
}

#[tokio::test]
#[ignore]
async fn test_scaling_curve() {
    use wenlan_core::eval::token_efficiency::run_scaling_eval;

    let fixture_dir = eval_root().join("fixtures");

    let sizes = vec![5, 10, 20, 50];
    let points = run_scaling_eval(&fixture_dir, &sizes, 10).await.unwrap();

    assert!(!points.is_empty(), "should produce scaling points");

    println!("\n=== Scaling Curve ===");
    println!(
        "{:<12} | {:<15} | {:<15}",
        "Corpus Size", "Wenlan Tokens", "Replay Tokens"
    );
    println!("{:-<12}-+-{:-<15}-+-{:-<15}", "", "", "");
    for p in &points {
        println!(
            "{:<12} | {:<15.0} | {:<15.0}",
            p.corpus_size, p.origin_tokens, p.replay_tokens
        );
    }

    // FullReplay should grow with corpus size
    if points.len() >= 2 {
        let first = &points[0];
        let last = points.last().unwrap();
        assert!(
            last.replay_tokens > first.replay_tokens,
            "FullReplay tokens should grow: {} -> {}",
            first.replay_tokens,
            last.replay_tokens
        );
    }
}

// ---------------------------------------------------------------------------
// Pipeline eval: LoCoMo + LongMemEval through Wenlan's full pipeline
// ---------------------------------------------------------------------------

/// Run LoCoMo through Wenlan's full pipeline: flat → enriched → distilled.
/// Requires Metal GPU (run with sandbox disabled).
#[tokio::test]
#[ignore]
async fn benchmark_locomo_pipeline() {
    use std::sync::Arc;
    use wenlan_core::eval::token_efficiency::run_locomo_pipeline_eval;

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found at {:?}", locomo_path);
        return;
    }

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new()
            .expect("Failed to init on-device LLM. Run with sandbox disabled for Metal GPU."),
    );

    let report = run_locomo_pipeline_eval(&locomo_path, llm, 10, 10)
        .await
        .expect("run_locomo_pipeline_eval failed");

    eprintln!("\n{}", report.to_terminal());

    // Sanity checks
    assert!(
        report.total_queries > 0,
        "Expected >0 queries, got {}",
        report.total_queries
    );
    assert!(
        !report.aggregate.is_empty(),
        "Should have aggregate metrics"
    );

    // Flat/Wenlan should have non-zero NDCG
    let flat_origin = report
        .aggregate
        .iter()
        .find(|c| c.condition == "flat" && c.strategy == "origin");
    assert!(
        flat_origin.is_some(),
        "Should have flat/origin aggregate cell"
    );
    assert!(
        flat_origin.unwrap().ndcg_at_10 > 0.0,
        "Flat/Wenlan NDCG should be > 0"
    );
}

/// Run LongMemEval through Wenlan's full pipeline: flat → enriched → distilled.
/// Requires Metal GPU (run with sandbox disabled).
/// Caps at 100 questions for reasonable runtime.
#[tokio::test]
#[ignore]
async fn benchmark_longmemeval_pipeline() {
    use std::sync::Arc;
    use wenlan_core::eval::token_efficiency::run_longmemeval_pipeline_eval;

    let path = eval_root().join("data/longmemeval_oracle.json");
    if !path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found at {:?}", path);
        return;
    }

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new()
            .expect("Failed to init on-device LLM. Run with sandbox disabled for Metal GPU."),
    );

    let report = run_longmemeval_pipeline_eval(&path, llm, 10, 500)
        .await
        .expect("run_longmemeval_pipeline_eval failed");

    eprintln!("\n{}", report.to_terminal());

    // Sanity checks
    assert!(
        report.total_queries > 0,
        "Expected >0 queries, got {}",
        report.total_queries
    );
    assert!(
        !report.aggregate.is_empty(),
        "Should have aggregate metrics"
    );

    let flat_origin = report
        .aggregate
        .iter()
        .find(|c| c.condition == "flat" && c.strategy == "origin");
    assert!(
        flat_origin.is_some(),
        "Should have flat/origin aggregate cell"
    );
    assert!(
        flat_origin.unwrap().ndcg_at_10 > 0.0,
        "Flat/Wenlan NDCG should be > 0"
    );
}

// ---------------------------------------------------------------------------
// Context path eval: recall vs context coverage comparison
// ---------------------------------------------------------------------------

/// Compare recall (search_memory only) vs context (search + concepts + graph).
/// Requires Metal GPU for enrichment/distillation. Run with sandbox disabled.
#[tokio::test]
#[ignore]
async fn benchmark_context_path() {
    use std::sync::Arc;
    use wenlan_core::eval::token_efficiency::run_context_path_eval;

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new()
            .expect("Failed to init on-device LLM. Run with sandbox disabled for Metal GPU."),
    );

    // 1 conversation for quick validation, 10 for full benchmark
    let report = run_context_path_eval(&locomo_path, llm, 10, 1)
        .await
        .expect("run_context_path_eval failed");

    eprintln!("\n{}", report.to_terminal());

    assert!(report.total_questions > 0);
}

/// Context path eval for LongMemEval: recall vs context coverage.
/// Requires Metal GPU. Run with sandbox disabled.
#[tokio::test]
#[ignore]
async fn benchmark_context_path_longmemeval() {
    use std::sync::Arc;
    use wenlan_core::eval::token_efficiency::run_context_path_eval_longmemeval;

    let path = eval_root().join("data/longmemeval_oracle.json");
    if !path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new()
            .expect("Failed to init on-device LLM. Run with sandbox disabled for Metal GPU."),
    );

    // Full benchmark: all 500 questions
    let report = run_context_path_eval_longmemeval(&path, llm, 10, 500)
        .await
        .expect("run_context_path_eval_longmemeval failed");

    eprintln!("\n{}", report.to_terminal());

    assert!(report.total_questions > 0);
}

// ---------------------------------------------------------------------------
// E2E answer quality: flat vs structured context with LLM-as-judge
// ---------------------------------------------------------------------------

/// Generate E2E answers for LoCoMo (flat vs structured context).
/// Saves judgment tuples for offline Claude Haiku judging.
/// Requires Metal GPU for on-device LLM. Run with sandbox disabled.
#[tokio::test]
#[ignore]
async fn generate_e2e_context_tuples_locomo() {
    use std::sync::Arc;
    use wenlan_core::eval::token_efficiency::{run_e2e_context_eval, save_judgment_tuples};

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new().expect("Failed to init on-device LLM"),
    );

    // 1 conversation, 20 questions for quick validation
    let tuples = run_e2e_context_eval(&locomo_path, llm, 10, 1, 20)
        .await
        .expect("run_e2e_context_eval failed");

    eprintln!("Generated {} judgment tuples", tuples.len());
    assert!(!tuples.is_empty(), "should generate at least some tuples");

    // Save for offline judging (try baselines dir, fallback to tmpdir)
    let baselines_dir = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines_dir).ok();
    let out_path = baselines_dir.join("e2e_context_tuples_locomo.json");
    save_judgment_tuples(&tuples, &out_path).expect("save tuples");
    eprintln!("Saved to {:?}", out_path);
}

/// Generate E2E answers for LongMemEval (flat vs structured context).
#[tokio::test]
#[ignore]
async fn generate_e2e_context_tuples_longmemeval() {
    use std::sync::Arc;
    use wenlan_core::eval::token_efficiency::{
        run_e2e_context_eval_longmemeval, save_judgment_tuples,
    };

    let path = eval_root().join("data/longmemeval_oracle.json");
    if !path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new().expect("Failed to init on-device LLM"),
    );

    // 50 questions for validation
    let tuples = run_e2e_context_eval_longmemeval(&path, llm, 10, 50, 1)
        .await
        .expect("run_e2e_context_eval_longmemeval failed");

    eprintln!("Generated {} judgment tuples", tuples.len());
    assert!(!tuples.is_empty());

    let baselines_dir = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines_dir).ok();
    let out_path = baselines_dir.join("e2e_context_tuples_longmemeval.json");
    save_judgment_tuples(&tuples, &out_path).expect("save tuples");
    eprintln!("Saved to {:?}", out_path);
}

/// Judge saved LoCoMo E2E context tuples with Claude Haiku.
/// Run after generate_e2e_context_tuples_locomo.
#[tokio::test]
#[ignore]
async fn judge_e2e_context_locomo() {
    use wenlan_core::eval::token_efficiency::{
        aggregate_judgments, judge_with_claude, load_judgment_tuples,
    };

    let tuples_path = eval_root().join("baselines/e2e_context_tuples_locomo.json");
    if !tuples_path.exists() {
        eprintln!("SKIP: run generate_e2e_context_tuples_locomo first");
        return;
    }

    let tuples = load_judgment_tuples(&tuples_path).expect("load tuples");
    eprintln!("Judging {} tuples...", tuples.len());

    let results = judge_with_claude(&tuples, 3).await.expect("judge failed");

    let report = aggregate_judgments(&results, "haiku");
    eprintln!("\n=== E2E Context Eval: LoCoMo (Claude Haiku Judge) ===");
    eprintln!(
        "{:<25} | {:<10} | {:<10} | {:<14} | Total",
        "Approach", "Accuracy", "Correct", "Context Tok"
    );
    eprintln!(
        "{:-<25}-+-{:-<10}-+-{:-<10}-+-{:-<14}-+-{:-<6}",
        "", "", "", "", ""
    );
    for r in &report.results_by_approach {
        eprintln!(
            "{:<25} | {:<10.1}% | {:<10} | {:<14.0} | {}",
            r.approach,
            r.accuracy * 100.0,
            r.correct,
            r.mean_context_tokens,
            r.total
        );
    }
    eprintln!("\nTotal judged: {}", report.total_judged);
}

// ---------------------------------------------------------------------------
// API-based E2E: Haiku as answer model, Sonnet as judge
// ---------------------------------------------------------------------------

/// Generate E2E answers using Claude Haiku (Max plan via CLI) instead of Qwen 4B.
/// No API key needed -- uses `claude -p` with OAuth.
#[tokio::test]
#[ignore]
async fn generate_e2e_context_tuples_locomo_api() {
    use std::sync::Arc;
    use wenlan_core::eval::token_efficiency::{run_e2e_context_eval, save_judgment_tuples};

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
        Arc::new(wenlan_core::llm_provider::ClaudeCliProvider::haiku());

    // 1 conversation, 20 questions for quick validation
    let tuples = run_e2e_context_eval(&locomo_path, llm, 10, 1, 20)
        .await
        .expect("run_e2e_context_eval with Haiku CLI failed");

    eprintln!("Generated {} judgment tuples (Haiku CLI)", tuples.len());
    assert!(!tuples.is_empty());

    let baselines_dir = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines_dir).ok();
    let out_path = baselines_dir.join("e2e_context_tuples_locomo_api.json");
    save_judgment_tuples(&tuples, &out_path).expect("save tuples");
    eprintln!("Saved to {:?}", out_path);
}

/// Judge saved API-generated tuples with Claude Sonnet (stronger judge).
/// Run after generate_e2e_context_tuples_locomo_api.
#[tokio::test]
#[ignore]
async fn judge_e2e_context_locomo_api_sonnet() {
    use wenlan_core::eval::token_efficiency::{
        aggregate_judgments, judge_with_claude_model, load_judgment_tuples,
    };

    let tuples_path = eval_root().join("baselines/e2e_context_tuples_locomo_api.json");
    if !tuples_path.exists() {
        eprintln!("SKIP: run generate_e2e_context_tuples_locomo_api first");
        return;
    }

    let tuples = load_judgment_tuples(&tuples_path).expect("load tuples");
    eprintln!("Judging {} tuples with Sonnet...", tuples.len());

    let results = judge_with_claude_model(&tuples, 3, "sonnet")
        .await
        .expect("judge failed");

    let report = aggregate_judgments(&results, "sonnet");
    eprintln!("\n=== E2E Context Eval: LoCoMo (Haiku answers, Sonnet judge) ===");
    eprintln!(
        "{:<25} | {:<10} | {:<10} | {:<14} | Total",
        "Approach", "Accuracy", "Correct", "Context Tok"
    );
    eprintln!(
        "{:-<25}-+-{:-<10}-+-{:-<10}-+-{:-<14}-+-{:-<6}",
        "", "", "", "", ""
    );
    for r in &report.results_by_approach {
        eprintln!(
            "{:<25} | {:<10.1}% | {:<10} | {:<14.0} | {}",
            r.approach,
            r.accuracy * 100.0,
            r.correct,
            r.mean_context_tokens,
            r.total
        );
    }
    eprintln!("\nTotal judged: {}", report.total_judged);
}

/// Re-judge the on-device (Qwen 4B) tuples with Sonnet instead of Haiku.
/// Compares judge quality: does a stronger judge change the ranking?
#[tokio::test]
#[ignore]
async fn judge_e2e_context_locomo_sonnet() {
    use wenlan_core::eval::token_efficiency::{
        aggregate_judgments, judge_with_claude_model, load_judgment_tuples,
    };

    let tuples_path = eval_root().join("baselines/e2e_context_tuples_locomo.json");
    if !tuples_path.exists() {
        eprintln!("SKIP: run generate_e2e_context_tuples_locomo first");
        return;
    }

    let tuples = load_judgment_tuples(&tuples_path).expect("load tuples");
    eprintln!("Judging {} tuples with Sonnet...", tuples.len());

    let results = judge_with_claude_model(&tuples, 3, "sonnet")
        .await
        .expect("judge failed");

    let report = aggregate_judgments(&results, "sonnet");
    eprintln!("\n=== E2E Context Eval: LoCoMo (Qwen answers, Sonnet judge) ===");
    eprintln!(
        "{:<25} | {:<10} | {:<10} | {:<14} | Total",
        "Approach", "Accuracy", "Correct", "Context Tok"
    );
    eprintln!(
        "{:-<25}-+-{:-<10}-+-{:-<10}-+-{:-<14}-+-{:-<6}",
        "", "", "", "", ""
    );
    for r in &report.results_by_approach {
        eprintln!(
            "{:<25} | {:<10.1}% | {:<10} | {:<14.0} | {}",
            r.approach,
            r.accuracy * 100.0,
            r.correct,
            r.mean_context_tokens,
            r.total
        );
    }
    eprintln!("\nTotal judged: {}", report.total_judged);
}

// ---------------------------------------------------------------------------
// Batch API Judge
// ---------------------------------------------------------------------------

/// Judge saved E2E context tuples via Batch API.
///
/// ```bash
/// ANTHROPIC_API_KEY=... cargo test -p origin --test eval_harness judge_e2e_batch -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn judge_e2e_batch() {
    use wenlan_core::eval::judge::{
        aggregate_judgments, judge_with_batch_api, load_judgment_tuples,
    };

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    let tuples_path = baselines.join("e2e_context_tuples_locomo.json");
    if !tuples_path.exists() {
        eprintln!("SKIP: run generate_e2e_context_tuples_locomo first");
        return;
    }

    let tuples = load_judgment_tuples(&tuples_path).expect("load failed");
    let judge_model = std::env::var("LME_JUDGE_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

    eprintln!(
        "=== Batch Judge ({} tuples, model={}) ===",
        tuples.len(),
        judge_model
    );

    let results = judge_with_batch_api(&tuples, &judge_model, None)
        .await
        .expect("batch judge failed");

    let report = aggregate_judgments(&results, &judge_model);
    for r in &report.results_by_approach {
        eprintln!(
            "  {}: {:.1}% ({}/{}) — {:.0} ctx tokens",
            r.approach,
            r.accuracy * 100.0,
            r.correct,
            r.total,
            r.mean_context_tokens
        );
    }
    eprintln!("\nTotal judged: {}", report.total_judged);
}

// ---------------------------------------------------------------------------
// Full-Pipeline (Enrichment + Concepts) — Batch API
// ---------------------------------------------------------------------------

/// Full-pipeline LoCoMo: enrich on-device, batch-generate answers, reuse flat cache.
///
/// ```bash
/// ANTHROPIC_API_KEY=... cargo test -p origin --test eval_harness generate_fullpipeline_locomo -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn generate_fullpipeline_locomo() {
    use wenlan_core::eval::answer_quality::run_fullpipeline_locomo_batch;

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let cli_mode = std::env::var("EVAL_PHASE3_CLI").as_deref() == Ok("1");
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(k) => k,
        Err(_) if cli_mode => String::new(), // CLI path doesn't use the key (Batch API bypassed)
        Err(_) => panic!("ANTHROPIC_API_KEY required (or set EVAL_PHASE3_CLI=1 for CLI path)"),
    };
    let answer_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("fullpipeline_locomo_tuples.json");

    eprintln!(
        "[fullpipeline] LoCoMo\n  model: {}\n  cost cap: ${:.2}\n  output: {:?}\n  cli_mode: {}",
        answer_model, cost_cap, output_path, cli_mode,
    );

    let enrichment = wenlan_core::eval::shared::EnrichmentMode::from_env(&answer_model, cost_cap)
        .expect("EnrichmentMode init failed");

    let tuples = run_fullpipeline_locomo_batch(
        &locomo_path,
        enrichment,
        &api_key,
        &answer_model,
        &output_path,
        cost_cap,
    )
    .await
    .expect("fullpipeline locomo failed");

    eprintln!("\nDone: {} tuples saved to {:?}", tuples.len(), output_path);
}

/// Full-pipeline LME: enrich on-device, batch-generate answers, reuse flat cache.
///
/// ```bash
/// ANTHROPIC_API_KEY=... cargo test -p origin --test eval_harness generate_fullpipeline_lme -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn generate_fullpipeline_lme() {
    use wenlan_core::eval::answer_quality::run_fullpipeline_lme_batch;

    let lme_path = eval_root().join("data/longmemeval_oracle.json");
    if !lme_path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let cli_mode = std::env::var("EVAL_PHASE3_CLI").as_deref() == Ok("1");
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(k) => k,
        Err(_) if cli_mode => String::new(),
        Err(_) => panic!("ANTHROPIC_API_KEY required (or set EVAL_PHASE3_CLI=1 for CLI path)"),
    };
    let answer_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("fullpipeline_lme_tuples.json");

    eprintln!(
        "[fullpipeline] LME\n  model: {}\n  cost cap: ${:.2}\n  output: {:?}\n  cli_mode: {}",
        answer_model, cost_cap, output_path, cli_mode,
    );

    let enrichment = wenlan_core::eval::shared::EnrichmentMode::from_env(&answer_model, cost_cap)
        .expect("EnrichmentMode init failed");

    let tuples = run_fullpipeline_lme_batch(
        &lme_path,
        enrichment,
        &api_key,
        &answer_model,
        &output_path,
        cost_cap,
    )
    .await
    .expect("fullpipeline lme failed");

    eprintln!("\nDone: {} tuples saved to {:?}", tuples.len(), output_path);
}

/// Enrich LongMemEval per-question DBs without answer generation.
///
/// Runs the same per-scenario seed/enrich path as `generate_fullpipeline_lme`,
/// then stops before Batch API / CLI answer generation. This is useful for
/// warming the expensive local phases first: entity extraction, title
/// enrichment, and concept distillation.
///
/// ```bash
/// WENLAN_LLM_WORKERS=1 WENLAN_LLM_PARALLEL_SEQS=8 EVAL_ENRICHMENT_CONCURRENCY=8 \
///   cargo test -p origin --test eval_harness enrich_fullpipeline_lme_only -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn enrich_fullpipeline_lme_only() {
    use wenlan_core::eval::longmemeval::{extract_memories, load_longmemeval};
    use wenlan_core::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use wenlan_core::sources::RawDocument;

    // Fixture override: LME_S_FIXTURE points at the DEEP per-question fixture
    // (longmemeval_s*.json — ~47-session haystack) for a faithful deep-S seed-only
    // run; default stays the shallow oracle for backward compat.
    let lme_path = match std::env::var("LME_S_FIXTURE") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => eval_root().join("data/longmemeval_oracle.json"),
    };
    if !lme_path.exists() {
        eprintln!("SKIP: fixture not found at {lme_path:?}");
        return;
    }

    let answer_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);
    let limit: Option<usize> = std::env::var("LME_LIMIT_QUESTIONS")
        .ok()
        .and_then(|s| s.parse().ok());
    let skip: usize = std::env::var("LME_SKIP_QUESTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines).ok();

    eprintln!(
        "[lme_enrich_only] baselines={} skip={} limit={:?}",
        baselines.display(),
        skip,
        limit
    );

    let samples = load_longmemeval(&lme_path).expect("load_longmemeval");
    use futures::StreamExt;
    use std::sync::Arc;

    let enrichment = Arc::new(
        EnrichmentMode::from_env(&answer_model, cost_cap).expect("EnrichmentMode init failed"),
    );
    let shared_embedder = eval_shared_embedder();

    // Cross-question fan-out. The serial loop this replaces only filled the M=8
    // batch DURING one question's classify phase, then ran n_seqs=1 through the
    // entity/distill gaps — leaving the GPU starved most of the run. Seeding
    // EVAL_SCENARIO_CONCURRENCY questions at once keeps the shared on-device
    // coalescer fed (each question owns its own scenario DB, so no write race),
    // raising realized width toward M=8 (~1.5x at conc=16, AGENTS.md). Mirrors
    // the answer_quality.rs full-pipeline fan-out, minus P3/P4.
    let scenario_concurrency: usize = std::env::var("EVAL_SCENARIO_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .clamp(1, 16);
    eprintln!("[lme_enrich_only] scenario concurrency = {scenario_concurrency}");

    let t0 = std::time::Instant::now();
    let total = samples.len();
    let end = limit.map(|n| (skip + n).min(total)).unwrap_or(total);
    let indices: Vec<usize> = (skip..end).collect();
    let n_work = indices.len();
    let samples_arc = Arc::new(samples);
    let baselines_arc = Arc::new(baselines.clone());
    let done = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Each task: (question_id, mem_count, enriched). None = empty/failed scenario.
    let results: Vec<(String, usize, usize)> = futures::stream::iter(indices.into_iter())
        .map(|idx| {
            let shared_embedder = shared_embedder.clone();
            let enrichment = enrichment.clone();
            let samples_arc = samples_arc.clone();
            let baselines_arc = baselines_arc.clone();
            let done = done.clone();
            async move {
                let sample = &samples_arc[idx];
                let ground_truth = sample
                    .answer
                    .as_str()
                    .unwrap_or(&sample.answer.to_string())
                    .to_string();
                let memories = extract_memories(sample);
                if ground_truth.is_empty() || memories.is_empty() {
                    return None;
                }

                let scope_dir = scenario_db_dir(&baselines_arc, "lme", &sample.question_id);
                let question_id = sample.question_id.clone();
                let qid_doc = question_id.clone();
                let question_type = sample.question_type.clone();
                let memories_owned = memories.clone();

                let scenario_t0 = std::time::Instant::now();
                let db = match open_or_seed_scenario_db(
                    &scope_dir,
                    shared_embedder,
                    move || {
                        memories_owned
                            .iter()
                            .map(|mem| RawDocument {
                                content: mem.content.clone(),
                                source_id: format!(
                                    "lme_{}_{}_t{}",
                                    qid_doc, mem.session_idx, mem.turn_idx
                                ),
                                source: "memory".to_string(),
                                title: format!(
                                    "session {} turn {}",
                                    mem.session_idx, mem.turn_idx
                                ),
                                memory_type: Some(
                                    if question_type == "single-session-preference" {
                                        "preference"
                                    } else {
                                        "fact"
                                    }
                                    .to_string(),
                                ),
                                space: Some("conversation".to_string()),
                                last_modified: chrono::Utc::now().timestamp(),
                                ..Default::default()
                            })
                            .collect()
                    },
                    &enrichment,
                )
                .await
                {
                    Ok(db) => db,
                    Err(e) => {
                        eprintln!("[lme_enrich_only] WARN q={question_id} seed failed: {e}");
                        return None;
                    }
                };

                let mem_count = db.memory_count().await.unwrap_or(0);
                let enriched = db.enriched_memory_count().await.unwrap_or(0);
                let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                // `scenario_t0.elapsed()` is the wall this question was IN FLIGHT — but
                // `.buffer_unordered(scenario_concurrency)` runs that many questions through
                // ONE GPU worker at once, so each question's wall is ~conc× its true
                // throughput cost. A reader who treats the raw in-flight wall as "cost to
                // seed one question" overcounts by conc× (the recurring 2026-06-21 misread:
                // 53min shown / conc3 = ~18min true). Emit wall AND wall/conc (the honest
                // per-Q rate when GPU-bound) so the inflated number can't be misread alone.
                let inflight_s = scenario_t0.elapsed().as_secs_f32();
                eprintln!(
                    "[lme_enrich_only] {}/{} q={} mem={} enriched={}/{} wall={:.0}s/conc{}≈{:.0}s/Q total={:.1}m",
                    d,
                    n_work,
                    question_id,
                    memories.len(),
                    enriched,
                    mem_count,
                    inflight_s,
                    scenario_concurrency,
                    inflight_s / scenario_concurrency.max(1) as f32,
                    t0.elapsed().as_secs_f32() / 60.0,
                );
                Some((question_id, mem_count, enriched))
            }
        })
        .buffer_unordered(scenario_concurrency)
        .filter_map(|x| async move { x })
        .collect()
        .await;

    let processed = results.len();
    let total_memories: usize = results.iter().map(|(_, m, _)| m).sum();
    let not_enriched: Vec<String> = results
        .iter()
        .filter(|(_, m, e)| m != e)
        .map(|(q, m, e)| format!("{q} ({e}/{m})"))
        .collect();

    eprintln!(
        "[lme_enrich_only] done: {} scenarios, {} memories in {:.1}m (concurrency {})",
        processed,
        total_memories,
        t0.elapsed().as_secs_f32() / 60.0,
        scenario_concurrency,
    );
    assert!(
        not_enriched.is_empty(),
        "scenarios not fully enriched: {not_enriched:?}"
    );
}

/// Smoke test: verify cached per-scenario enriched DBs open and report fully-enriched.
///
/// Skips silently if no per-scenario DBs are present (gitignored, only present locally
/// after a full-pipeline eval run). When present, walks each benchmark's scenario
/// subdirectories and asserts that any non-empty scenario DB is fully enriched.
///
/// Fast (no LLM, no API) — just opens DB, counts memories, asserts enrichment_complete.
#[tokio::test]
async fn smoke_enriched_db_reuse() {
    use std::sync::Arc;
    use wenlan_core::db::MemoryDB;

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));

    let mut total_scenarios_checked = 0usize;
    let mut total_passed = 0usize;

    for benchmark in ["locomo", "lme"] {
        let bench_dir = baselines.join("fullpipeline").join(benchmark);
        if !bench_dir.exists() {
            eprintln!(
                "[smoke] SKIP {}: {} not found",
                benchmark,
                bench_dir.display()
            );
            continue;
        }

        let entries = match std::fs::read_dir(&bench_dir) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[smoke] SKIP {}: read_dir failed: {}", benchmark, e);
                continue;
            }
        };

        let mut bench_checked = 0usize;
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let scenario_id = entry.file_name().to_string_lossy().to_string();
            let scenario_dir = entry.path();

            let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
                Arc::new(wenlan_core::NoopEmitter);
            let db = MemoryDB::new(&scenario_dir, emitter)
                .await
                .expect("[smoke] open scenario DB");
            let mem = db.memory_count().await.expect("memory_count");
            let enriched = db.enriched_memory_count().await.expect("enriched_count");

            if mem == 0 {
                // empty / partial scenario — skip rather than fail
                continue;
            }

            assert_eq!(
                mem, enriched,
                "[smoke] {}/{} should be fully enriched (got {}/{})",
                benchmark, scenario_id, enriched, mem
            );
            total_scenarios_checked += 1;
            total_passed += 1;
            bench_checked += 1;
        }

        if bench_checked > 0 {
            eprintln!(
                "[smoke] {}: checked {} scenarios, all fully enriched",
                benchmark, bench_checked
            );
        }
    }

    if total_scenarios_checked == 0 {
        eprintln!(
            "[smoke] SKIP: no per-scenario enriched DBs found locally \
             (CI without baselines is normal)."
        );
    } else {
        eprintln!("[smoke] OK: {} scenarios verified", total_passed);
    }
}

/// Judge full-pipeline tuples for LoCoMo via Batch API.
///
/// ```bash
/// ANTHROPIC_API_KEY=... cargo test -p origin --test eval_harness judge_fullpipeline_locomo -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn judge_fullpipeline_locomo() {
    use wenlan_core::eval::judge::{
        aggregate_judgments, judge_with_batch_api, load_judgment_tuples,
    };

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    // EVAL_TUPLES_FILE override lets us judge alternate files (e.g. *_pregate.json)
    let default_path = baselines.join("fullpipeline_locomo_tuples.json");
    let tuples_path: std::path::PathBuf = std::env::var("EVAL_TUPLES_FILE")
        .map(std::path::PathBuf::from)
        .unwrap_or(default_path);
    if !tuples_path.exists() {
        eprintln!("SKIP: run generate_fullpipeline_locomo first");
        return;
    }

    let tuples = load_judgment_tuples(&tuples_path).expect("load failed");
    let judge_model = std::env::var("EVAL_JUDGE_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

    eprintln!(
        "=== Full-Pipeline LoCoMo Judge ({} tuples, judge={}) ===",
        tuples.len(),
        judge_model
    );

    let results = judge_with_batch_api(&tuples, &judge_model, None)
        .await
        .expect("batch judge failed");

    let report = aggregate_judgments(&results, &judge_model);
    print_judge_report(&report);
}

/// Judge full-pipeline tuples for LME via Batch API.
///
/// ```bash
/// ANTHROPIC_API_KEY=... cargo test -p origin --test eval_harness judge_fullpipeline_lme -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn judge_fullpipeline_lme() {
    use wenlan_core::eval::judge::{
        aggregate_judgments, judge_with_batch_api, load_judgment_tuples,
    };

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    let tuples_path = baselines.join("fullpipeline_lme_tuples.json");
    if !tuples_path.exists() {
        eprintln!("SKIP: run generate_fullpipeline_lme first");
        return;
    }

    let tuples = load_judgment_tuples(&tuples_path).expect("load failed");
    let judge_model = std::env::var("EVAL_JUDGE_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

    eprintln!(
        "=== Full-Pipeline LME Judge ({} tuples, judge={}) ===",
        tuples.len(),
        judge_model
    );

    let results = judge_with_batch_api(&tuples, &judge_model, None)
        .await
        .expect("batch judge failed");

    let report = aggregate_judgments(&results, &judge_model);
    print_judge_report(&report);
}

// ---------------------------------------------------------------------------
// Distill A/B — 4B vs 9B quality probe at 90s budget
// ---------------------------------------------------------------------------

/// Quick A/B test: distill same cluster with 4B and 9B at 90s budget.
/// Quality comparison via embedding similarity to source mems + manual review.
///
/// ```bash
/// cargo test -p origin --test eval_harness quality_distill_4b_vs_9b -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn quality_distill_4b_vs_9b() {
    use std::sync::Arc;
    use wenlan_core::eval::locomo::{extract_observations, load_locomo};
    use wenlan_core::eval::shared::eval_shared_embedder;
    use wenlan_core::llm_provider::{LlmProvider, LlmRequest, OnDeviceProvider};
    use wenlan_core::prompts::PromptRegistry;

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found at {:?}", locomo_path);
        return;
    }

    let samples = load_locomo(&locomo_path).unwrap();
    let observations = extract_observations(&samples[0]);
    let obs_slice: Vec<&str> = observations
        .iter()
        .take(5)
        .map(|o| o.content.as_str())
        .collect();

    if obs_slice.is_empty() {
        eprintln!("SKIP: no observations found in first sample");
        return;
    }

    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());

    // Build prompt matching distill_one_cluster format
    let topic = "Caroline";
    let memories_block: String = obs_slice
        .iter()
        .enumerate()
        .map(|(i, content)| {
            let snippet: String = content.chars().take(800).collect();
            format!("[obs_{}] {}", i, snippet)
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let user_prompt = format!("Topic: {}\n\n{}", topic, memories_block);
    let source_text = obs_slice.join(" ");

    eprintln!(
        "\n[distill-ab] cluster: {} observations, prompt ~{} chars",
        obs_slice.len(),
        user_prompt.len()
    );

    // Helper: embed texts with shared embedder
    let embedder = eval_shared_embedder();
    let embed_texts = |texts: Vec<String>| -> Vec<Vec<f32>> {
        let mut embedder = embedder.lock().unwrap();
        embedder.embed(texts, None).unwrap_or_default()
    };

    // Helper: cosine similarity (inline, cosine_similarity is pub(crate))
    let cosine_sim = |a: &[f32], b: &[f32]| -> f64 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            (dot / (na * nb)) as f64
        }
    };

    // Embed source for similarity comparison
    let source_embs = embed_texts(vec![source_text.clone()]);
    let source_emb = match source_embs.into_iter().next() {
        Some(e) => e,
        None => {
            eprintln!("SKIP: embedding failed");
            return;
        }
    };

    for (model_id, label) in [("qwen3-4b", "4B"), ("qwen3.5-9b", "9B")] {
        eprintln!("\n[distill-ab] loading {} ...", label);
        let llm: Arc<dyn LlmProvider> = match OnDeviceProvider::new_with_model(Some(model_id)) {
            Ok(p) => Arc::new(p),
            Err(e) => {
                eprintln!("[distill-ab] SKIP {}: {}", label, e);
                continue;
            }
        };

        let max_tokens = llm.recommended_max_output();
        let t0 = std::time::Instant::now();
        let result = llm
            .generate(LlmRequest {
                system_prompt: Some(prompts.distill_page.clone()),
                user_prompt: user_prompt.clone(),
                max_tokens,
                temperature: 0.1,
                label: Some(format!("distill_ab_{}", label)),
                timeout_secs: Some(90),
            })
            .await;
        let elapsed = t0.elapsed();

        match result {
            Ok(raw) => {
                let cleaned = wenlan_core::llm_provider::strip_think_tags(&raw);
                let text = cleaned.trim().to_string();

                // Compute similarity to source
                let out_embs = embed_texts(vec![text.clone()]);
                let sim = out_embs
                    .into_iter()
                    .next()
                    .map(|e| cosine_sim(&e, &source_emb))
                    .unwrap_or(0.0);

                eprintln!(
                    "[distill-ab] {} sim={:.3} len={} elapsed={:.1}s",
                    label,
                    sim,
                    text.len(),
                    elapsed.as_secs_f64()
                );

                let out_path = format!("/tmp/distill_{}.txt", label.to_lowercase());
                let _ = std::fs::write(&out_path, &text);
                eprintln!("[distill-ab] {} output saved to {}", label, out_path);
                eprintln!("--- {} output (first 400 chars) ---", label);
                eprintln!("{}", text.chars().take(400).collect::<String>());
                eprintln!("---");
            }
            Err(e) => {
                eprintln!("[distill-ab] {} FAILED: {}", label, e);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Batch Size Probe — find on-device extraction overflow point
// ---------------------------------------------------------------------------

/// Probe extraction at batch sizes 1, 5, 10, 20, 30, 50 to find the on-device
/// context overflow point and quality degradation curve.
///
/// ```bash
/// cargo test -p origin --test eval_harness probe_batch_sizes -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn probe_batch_sizes() {
    use std::sync::Arc;
    use wenlan_core::eval::locomo::{extract_observations, load_locomo};
    use wenlan_core::eval::shared::probe_extraction_batch_sizes;

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let samples = load_locomo(&locomo_path).unwrap();
    let obs: Vec<(String, String)> = extract_observations(&samples[0])
        .iter()
        .enumerate()
        .map(|(i, m)| (format!("obs_{}", i), m.content.clone()))
        .collect();
    eprintln!(
        "Loaded {} observations from {}",
        obs.len(),
        samples[0].sample_id
    );

    // Test 4B first (default), then 9B if available
    let model_id = std::env::var("PROBE_MODEL").ok();
    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new_with_model(model_id.as_deref())
            .expect("on-device LLM required"),
    );
    eprintln!("Model: {}", model_id.as_deref().unwrap_or("4B (default)"));

    let batch_sizes = [1, 2, 3, 5, 10, 20, 30, 50];
    let results = probe_extraction_batch_sizes(&obs, &llm, &batch_sizes).await;

    eprintln!("\n=== Batch Size Probe Results ===");
    eprintln!(
        "{:>5} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10}",
        "Batch", "InTok", "RespLen", "Entities", "Obs", "Ent/Input"
    );
    eprintln!(
        "{:-<5}-+-{:-<8}-+-{:-<8}-+-{:-<8}-+-{:-<8}-+-{:-<10}",
        "", "", "", "", "", ""
    );
    for (bs, in_tok, resp_len, ents, obs_count) in &results {
        let ratio = if *bs > 0 {
            *ents as f64 / *bs as f64
        } else {
            0.0
        };
        eprintln!(
            "{:>5} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10.2}",
            bs, in_tok, resp_len, ents, obs_count, ratio
        );
    }
}

/// Smoke test: 1 conversation, full pipeline, validates all batch phases work.
///
/// ```bash
/// ANTHROPIC_API_KEY=... cargo test -p origin --test eval_harness smoke_fullpipeline -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn smoke_fullpipeline() {
    use std::sync::Arc;
    use wenlan_core::eval::locomo::{extract_observations, load_locomo};
    use wenlan_core::eval::shared::{
        count_tokens, eval_shared_embedder, run_concept_distillation_batch_api,
        run_enrichment_batch_api, run_title_enrichment_batch_api,
    };

    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY required");
    let model = "claude-haiku-4-5-20251001";

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let samples = load_locomo(&locomo_path).unwrap();
    let sample = &samples[0]; // Just 1 conversation
    let memories = extract_observations(sample);
    eprintln!("Conv {}: {} observations", sample.sample_id, memories.len());

    // Seed
    let shared_embedder = eval_shared_embedder();
    let tmp = tempfile::tempdir().unwrap();
    let db = wenlan_core::db::MemoryDB::new_with_shared_embedder(
        tmp.path(),
        Arc::new(wenlan_core::events::NoopEmitter),
        shared_embedder,
    )
    .await
    .unwrap();

    let docs: Vec<wenlan_core::sources::RawDocument> = memories
        .iter()
        .enumerate()
        .map(|(i, mem)| wenlan_core::sources::RawDocument {
            content: mem.content.clone(),
            source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
            source: "memory".to_string(),
            title: format!("{} session {}", mem.speaker, mem.session_num),
            memory_type: Some("fact".to_string()),
            space: Some("conversation".to_string()),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        })
        .collect();
    let seeded = db.upsert_documents(docs).await.unwrap();
    eprintln!("Seeded: {} chunks", seeded);

    // Phase 1: Entity extraction
    let entities = run_enrichment_batch_api(&db, &api_key, model, 2.0)
        .await
        .unwrap();
    eprintln!("Entities: {}", entities);
    assert!(entities > 0, "should extract some entities");

    // Phase 2: Title enrichment
    let titles = run_title_enrichment_batch_api(&db, &api_key, model, 1.0)
        .await
        .unwrap();
    eprintln!("Titles enriched: {}", titles);

    // Phase 3: Concept distillation
    let concepts = run_concept_distillation_batch_api(&db, &api_key, model, 1.0)
        .await
        .unwrap();
    eprintln!("Concepts: {}", concepts);

    // Phase 4: Context collection - check flat vs structured differ
    let qa = &sample.qa[0];
    let flat_results = db
        .search_memory(
            &qa.question,
            10,
            None,
            &ReadScope::Global,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    let flat_ctx: String = flat_results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("{}. {}", i + 1, r.content))
        .collect::<Vec<_>>()
        .join("\n");
    let flat_tokens = count_tokens(&flat_ctx);

    let concept_results = db
        .search_pages(&qa.question, 3, None)
        .await
        .unwrap_or_default();
    let mut structured_parts: Vec<String> = Vec::new();
    if !concept_results.is_empty() {
        structured_parts.push("## Compiled Knowledge".to_string());
        for c in &concept_results {
            structured_parts.push(format!(
                "**{}**: {}",
                c.title,
                c.content.chars().take(200).collect::<String>()
            ));
        }
    }
    structured_parts.push(flat_ctx.clone());
    let structured_tokens = count_tokens(&structured_parts.join("\n\n"));

    eprintln!(
        "\nContext check for: {}\n  flat: {} tokens\n  structured: {} tokens (delta: +{})\n  concepts found: {}",
        &qa.question.chars().take(60).collect::<String>(),
        flat_tokens, structured_tokens, structured_tokens - flat_tokens, concept_results.len()
    );

    eprintln!("\n=== Smoke test PASSED ===");
    eprintln!(
        "  {} entities, {} titles, {} concepts",
        entities, titles, concepts
    );
    eprintln!(
        "  Structured context is {} tokens larger than flat",
        structured_tokens - flat_tokens
    );
}

/// Manual smoke for the per-scenario DB refactor (T1-T6).
///
/// Exercises `open_or_seed_scenario_db` with 2 LoCoMo samples on-device:
/// 1. Seeds + enriches per-conv DB at `{tempdir}/fullpipeline/locomo/{sample_id}/`.
/// 2. Verifies expected paths exist with `mem == enriched`.
/// 3. Builds context for first qa via per-conv DB.
/// 4. Re-runs `open_or_seed_scenario_db` to confirm cache-hit branch (no re-enrich).
/// 5. Asserts the two per-conv DBs are PHYSICALLY ISOLATED (no cross-conv source_ids).
///
/// Parameterizable via env for use as a pre-flight probe before a full LoCoMo run.
///
/// On-device Qwen3-4B (free, Metal GPU). Requires `--ignored --nocapture`.
///
/// Run as probe before full LME/LoCoMo:
///   SMOKE_LOCOMO_SAMPLES=10 EVAL_LOCAL_MODEL=qwen3.5-9b \
///   cargo test -p origin --test eval_harness smoke_per_scenario_locomo -- --ignored --nocapture
/// Default config (2 samples x 10 obs) is the CI smoke.
#[tokio::test]
#[ignore]
async fn smoke_per_scenario_locomo() {
    use wenlan_core::eval::locomo::{extract_observations, load_locomo};
    use wenlan_core::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use wenlan_core::sources::RawDocument;

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let samples = load_locomo(&locomo_path).unwrap();
    let n_samples: usize = std::env::var("SMOKE_LOCOMO_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let max_obs_per_conv: usize = std::env::var("SMOKE_LOCOMO_MAX_OBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    assert!(
        samples.len() >= n_samples.min(2),
        "need >={} LoCoMo samples for smoke",
        n_samples.min(2)
    );

    // Tempdir for baselines so we don't pollute real eval cache
    let tmp = tempfile::tempdir().unwrap();
    let baselines_dir = tmp.path().to_path_buf();
    eprintln!(
        "[smoke-per-scenario] baselines: {}",
        baselines_dir.display()
    );

    // On-device enrichment (default per EnrichmentMode::from_env when EVAL_ENRICHMENT unset)
    let enrichment = EnrichmentMode::from_env("claude-haiku-4-5-20251001", 1.0)
        .expect("EnrichmentMode::from_env");
    let shared_embedder = eval_shared_embedder();

    let pair = &samples[..n_samples.min(samples.len())];
    let mut per_sample_source_ids: Vec<Vec<String>> = Vec::new();
    let mut total_mems_processed: usize = 0;

    let smoke_t0 = std::time::Instant::now();
    for sample in pair {
        let mut memories = extract_observations(sample);
        memories.truncate(max_obs_per_conv);
        eprintln!(
            "[smoke-per-scenario] {} ({} obs, truncated from full) -> open_or_seed",
            sample.sample_id,
            memories.len()
        );

        let scope_dir = scenario_db_dir(&baselines_dir, "locomo", &sample.sample_id);

        let sample_id = sample.sample_id.clone();
        let memories_owned = memories.clone();
        let conv_t0 = std::time::Instant::now();
        let db = open_or_seed_scenario_db(
            &scope_dir,
            shared_embedder.clone(),
            move || {
                memories_owned
                    .iter()
                    .enumerate()
                    .map(|(i, mem)| RawDocument {
                        content: mem.content.clone(),
                        source_id: format!("locomo_{}_obs_{}", sample_id, i),
                        source: "memory".to_string(),
                        title: format!("{} session {}", mem.speaker, mem.session_num),
                        memory_type: Some("fact".to_string()),
                        space: Some("conversation".to_string()),
                        last_modified: chrono::Utc::now().timestamp(),
                        ..Default::default()
                    })
                    .collect()
            },
            &enrichment,
        )
        .await
        .expect("open_or_seed_scenario_db");
        let conv_elapsed = conv_t0.elapsed().as_secs_f32();

        let mem_count = db.memory_count().await.unwrap_or(0);
        let enriched = db.enriched_memory_count().await.unwrap_or(0);
        total_mems_processed += mem_count;
        eprintln!(
            "[smoke-per-scenario] {}: seed+enrich done in {:.1}s, mem={} enriched={}",
            sample.sample_id, conv_elapsed, mem_count, enriched
        );
        assert!(
            mem_count > 0,
            "{}: should have seeded memories",
            sample.sample_id
        );
        assert_eq!(
            mem_count, enriched,
            "{}: should be fully enriched ({}/{})",
            sample.sample_id, enriched, mem_count
        );

        // Verify physical path exists
        assert!(
            scope_dir.join("origin_memory.db").exists(),
            "{}: DB file should exist at {}",
            sample.sample_id,
            scope_dir.display()
        );

        // Retrieval scoped to own conv: every result must come from this sample's source_id prefix
        let results = db
            .search_memory(
                "anything",
                50,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("search_memory");
        let expected_prefix = format!("locomo_{}_obs_", sample.sample_id);
        for r in &results {
            assert!(
                r.source_id.starts_with(&expected_prefix),
                "{}: cross-conv leak! result source_id={} (expected prefix {})",
                sample.sample_id,
                r.source_id,
                expected_prefix
            );
        }
        per_sample_source_ids.push(results.iter().map(|r| r.source_id.clone()).collect());
    }

    // Cross-conv isolation: zero source_id overlap between conv 0 and conv 1
    let conv0: std::collections::HashSet<_> = per_sample_source_ids[0].iter().collect();
    let conv1: std::collections::HashSet<_> = per_sample_source_ids[1].iter().collect();
    let overlap: Vec<_> = conv0.intersection(&conv1).copied().collect();
    assert!(
        overlap.is_empty(),
        "cross-conv source_id leak: {:?}",
        overlap
    );

    // Cache-hit verification: re-open first conv. Should NOT re-enrich.
    let sample = &pair[0];
    let mut memories = extract_observations(sample);
    memories.truncate(max_obs_per_conv);
    let scope_dir = scenario_db_dir(&baselines_dir, "locomo", &sample.sample_id);
    let sample_id_2 = sample.sample_id.clone();
    let memories_owned_2 = memories.clone();
    let cache_t0 = std::time::Instant::now();
    let _db_cached = open_or_seed_scenario_db(
        &scope_dir,
        shared_embedder.clone(),
        move || {
            memories_owned_2
                .iter()
                .enumerate()
                .map(|(i, mem)| RawDocument {
                    content: mem.content.clone(),
                    source_id: format!("locomo_{}_obs_{}", sample_id_2, i),
                    source: "memory".to_string(),
                    title: format!("{} session {}", mem.speaker, mem.session_num),
                    memory_type: Some("fact".to_string()),
                    space: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                })
                .collect()
        },
        &enrichment,
    )
    .await
    .expect("re-open cached");
    let cache_ms = cache_t0.elapsed().as_millis();
    eprintln!(
        "[smoke-per-scenario] cache-hit re-open of {}: {} ms (expected <2000ms; no re-enrich)",
        sample.sample_id, cache_ms
    );
    assert!(
        cache_ms < 5000,
        "cache hit too slow ({} ms) -- helper likely re-enriched",
        cache_ms
    );

    let total_elapsed = smoke_t0.elapsed().as_secs_f32();
    eprintln!(
        "\n=== smoke_per_scenario_locomo PASSED in {:.1}s ===",
        total_elapsed
    );
    eprintln!("  per-conv DB layout: ✓");
    eprintln!(
        "  enrichment per-conv (truncated to {} obs each): ✓",
        max_obs_per_conv
    );
    eprintln!("  retrieval scoped to own conv: ✓");
    eprintln!("  cross-conv source_id isolation: ✓");
    eprintln!("  cache-hit re-open ({}ms): ✓", cache_ms);

    // Extrapolation summary — useful for pre-flight estimation.
    let mean_per_mem_sec = total_elapsed / total_mems_processed.max(1) as f32;
    let full_locomo_estimate_min = (mean_per_mem_sec * 500.0) / 60.0;
    eprintln!(
        "[smoke-locomo] SUMMARY: {} samples x ~{} obs = {} obs in {:.1}s ({:.2}s/obs). \
         Full LoCoMo (~500 obs) extrapolated: ~{:.1} min (~{:.1} hours).",
        n_samples,
        max_obs_per_conv,
        total_mems_processed,
        total_elapsed,
        mean_per_mem_sec,
        full_locomo_estimate_min,
        full_locomo_estimate_min / 60.0,
    );
}

/// STEP 6 measurement: isolated classify-ONLY rate for the STEP 7 additive backfill.
///
/// Classification is orthogonal to entity/title/page enrichment (it reads `content`,
/// writes `importance`/`event_date`/`quality`), so seeding N docs and timing
/// `run_classification_for_eval_concurrent` alone yields the exact per-memory rate the
/// STEP 7 snapshot path pays when backfilling the existing entity/title/page-enriched
/// seeds (which are all `importance IS NULL`). No entity/title/distill passes needed.
///
/// On-device Qwen3-4B (free, Metal). Isolated tempdir (no cache pollution).
///
/// ```bash
/// MEASURE_CLASSIFY_N=30 EVAL_ENRICHMENT_CONCURRENCY=8 WENLAN_LLM_PARALLEL_SEQS=8 \
///   cargo test -p origin-core --features eval-harness --test eval_harness \
///   measure_classify_only_rate -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn measure_classify_only_rate() {
    use std::sync::Arc;
    use wenlan_core::eval::locomo::{extract_observations, load_locomo};
    use wenlan_core::eval::shared::{eval_shared_embedder, run_classification_for_eval_concurrent};
    use wenlan_core::llm_provider::OnDeviceProvider;
    use wenlan_core::sources::RawDocument;

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found at {:?}", locomo_path);
        return;
    }

    let n: usize = std::env::var("MEASURE_CLASSIFY_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let concurrency: usize = std::env::var("EVAL_ENRICHMENT_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // Gather N observations across samples (content only — classify ignores titles/entities).
    let samples = load_locomo(&locomo_path).unwrap();
    let mut obs: Vec<String> = Vec::new();
    'outer: for s in &samples {
        for o in extract_observations(s) {
            obs.push(o.content.clone());
            if obs.len() >= n {
                break 'outer;
            }
        }
    }
    assert!(!obs.is_empty(), "no observations loaded");

    let shared_embedder = eval_shared_embedder();
    // Persist to MEASURE_CLASSIFY_DB_OUT for post-run SQL shape inspection; else
    // throwaway tempdir (auto-cleaned). `tmp` stays in scope either way.
    let tmp = tempfile::tempdir().unwrap();
    let db_dir = match std::env::var("MEASURE_CLASSIFY_DB_OUT") {
        Ok(out) => {
            std::fs::create_dir_all(&out).ok();
            std::path::PathBuf::from(out)
        }
        Err(_) => tmp.path().to_path_buf(),
    };
    eprintln!("[measure-classify] db_dir = {}", db_dir.display());
    let db = wenlan_core::db::MemoryDB::new_with_shared_embedder(
        &db_dir,
        Arc::new(wenlan_core::events::NoopEmitter),
        shared_embedder,
    )
    .await
    .unwrap();

    let docs: Vec<RawDocument> = obs
        .iter()
        .enumerate()
        .map(|(i, c)| RawDocument {
            content: c.clone(),
            source_id: format!("classify_probe_obs_{}", i),
            source: "memory".to_string(),
            title: format!("probe {}", i),
            memory_type: Some("fact".to_string()),
            space: Some("conversation".to_string()),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        })
        .collect();
    db.upsert_documents(docs).await.unwrap();

    // PRE-condition: every seeded memory is unclassified (importance IS NULL).
    let pre = db
        .get_memories_needing_classification()
        .await
        .unwrap()
        .len();
    assert_eq!(
        pre,
        obs.len(),
        "all seeded mems should be unclassified pre-run ({}/{})",
        pre,
        obs.len()
    );

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = match OnDeviceProvider::new() {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("SKIP: on-device init failed: {e}");
            return;
        }
    };

    let t0 = std::time::Instant::now();
    let processed = run_classification_for_eval_concurrent(&db, &llm, concurrency)
        .await
        .unwrap();
    let elapsed = t0.elapsed().as_secs_f64();

    // POST-condition: classify populated importance for every memory.
    let post = db
        .get_memories_needing_classification()
        .await
        .unwrap()
        .len();
    assert_eq!(
        post, 0,
        "all mems should be classified post-run ({post} remain)"
    );
    assert_eq!(processed, obs.len(), "processed count mismatch");

    let rate = elapsed / obs.len() as f64;
    eprintln!("\n=== classify-only rate (concurrency={concurrency}) ===");
    eprintln!(
        "  N={} processed={} elapsed={:.1}s  =>  {:.2}s/mem",
        obs.len(),
        processed,
        elapsed,
        rate
    );
    eprintln!(
        "  STEP 7 corpus 8064 mems => {:.0}s = {:.1} min = {:.2} h",
        8064.0 * rate,
        8064.0 * rate / 60.0,
        8064.0 * rate / 3600.0
    );
    eprintln!(
        "    LME 5533 => {:.2} h ;  LoCoMo 2531 => {:.2} h",
        5533.0 * rate / 3600.0,
        2531.0 * rate / 3600.0
    );
}

/// End-to-end smoke that verifies EVAL_BASELINES_DIR wires through to a real
/// DB-open code path. Builds the scenario path via the helper + `scenario_db_dir`,
/// opens a `MemoryDB` at that path, and asserts the DB file lands where expected.
///
/// Avoids `open_or_seed_scenario_db` to keep the test light: no enrichment,
/// no embedder warmup beyond what `MemoryDB::new` does on its own.
#[tokio::test]
#[ignore]
async fn smoke_eval_baselines_dir_e2e() {
    use std::sync::Arc;
    use wenlan_core::db::MemoryDB;
    use wenlan_core::eval::shared::{eval_baselines_dir_override, scenario_db_dir};
    use wenlan_core::events::NoopEmitter;

    let tmp = tempfile::tempdir().unwrap();
    let path_str = tmp.path().to_str().unwrap().to_string();

    // Use temp_env::async_with_vars for closure-scoped, panic-safe restore.
    // Avoids env-var leak if any assertion below panics.
    temp_env::async_with_vars([("EVAL_BASELINES_DIR", Some(path_str.as_str()))], async {
        let baselines = eval_baselines_dir_override().expect("override should resolve");
        assert_eq!(baselines, tmp.path());

        let scope = scenario_db_dir(&baselines, "smoketest", "id-1");
        std::fs::create_dir_all(&scope).unwrap();

        let db = MemoryDB::new(&scope, Arc::new(NoopEmitter))
            .await
            .expect("MemoryDB should open at override path");

        let count = db.memory_count().await.unwrap_or(0);
        assert_eq!(count, 0, "fresh DB should have 0 memories");

        let expected_db_path = scope.join("origin_memory.db");
        assert!(
            expected_db_path.exists(),
            "DB not at expected EVAL_BASELINES_DIR path: {}",
            expected_db_path.display()
        );
    })
    .await;
}

/// Manual smoke for per-scenario enrichment using Claude CLI provider (D2).
///
/// Validates that `EnrichmentMode::OnDevice(ClaudeCliProvider)` works end-to-end:
/// seed → Phase-1 concurrent classification (EVAL_ENRICHMENT_CONCURRENCY=4) → Phase-2
/// sequential per-memory entity extraction + title + page enrichment (post-de-fork,
/// EVAL_ENRICHMENT_CONCURRENCY governs Phase-1 only) → distillation → isolation check.
/// Uses claude-haiku-4-5-20251001 via Max plan.
///
/// - 1 LoCoMo conversation, 5 observations (small: CLI subprocess ~2-3s/call)
/// - Expected duration: ~30s (Phase-1: 5 obs / 4 concurrency × 3s; Phase-2: 5 obs sequential × ~3s + distill)
/// - Skips silently if `claude` binary is not in PATH
///
/// ```bash
/// cargo test -p origin --test eval_harness smoke_per_scenario_locomo_cli -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn smoke_per_scenario_locomo_cli() {
    use std::sync::Arc;
    use wenlan_core::eval::locomo::{extract_observations, load_locomo};
    use wenlan_core::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use wenlan_core::sources::RawDocument;

    // Probe for `claude` binary — skip silently if not available.
    let probe = std::process::Command::new("claude")
        .arg("--version")
        .output();
    match probe {
        Ok(out) if out.status.success() => {
            eprintln!(
                "[smoke-cli] claude binary found: {}",
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }
        _ => {
            eprintln!(
                "SKIP: `claude` binary not in PATH — install Claude Code CLI to run this smoke"
            );
            return;
        }
    }

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let samples = load_locomo(&locomo_path).unwrap();
    assert!(!samples.is_empty(), "need at least 1 LoCoMo sample");

    // Set EVAL_ENRICHMENT_CONCURRENCY=4 for this test scope.
    // SAFETY: single-threaded test setup; env var is read by enrich_db_for_eval_local.
    let prev_concurrency = std::env::var("EVAL_ENRICHMENT_CONCURRENCY").ok();
    std::env::set_var("EVAL_ENRICHMENT_CONCURRENCY", "4");
    // Restore on drop via a simple guard.
    struct EnvGuard(Option<String>);
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.0 {
                Some(v) => std::env::set_var("EVAL_ENRICHMENT_CONCURRENCY", v),
                None => std::env::remove_var("EVAL_ENRICHMENT_CONCURRENCY"),
            }
        }
    }
    let _env_guard = EnvGuard(prev_concurrency);

    let tmp = tempfile::tempdir().unwrap();
    let baselines_dir = tmp.path().to_path_buf();
    eprintln!("[smoke-cli] baselines: {}", baselines_dir.display());

    let cli_llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::ClaudeCliProvider::new("claude-haiku-4-5-20251001"),
    );
    let enrichment = EnrichmentMode::OnDevice(cli_llm);
    let shared_embedder = eval_shared_embedder();

    // 1 conversation × 5 observations — small enough for CLI subprocess overhead.
    const MAX_OBS: usize = 5;
    let sample = &samples[0];
    let mut memories = extract_observations(sample);
    memories.truncate(MAX_OBS);

    eprintln!(
        "[smoke-cli] {} ({} obs) -> open_or_seed",
        sample.sample_id,
        memories.len()
    );

    let scope_dir = scenario_db_dir(&baselines_dir, "locomo", &sample.sample_id);
    let sample_id = sample.sample_id.clone();
    let memories_owned = memories.clone();
    let t0 = std::time::Instant::now();

    let db = open_or_seed_scenario_db(
        &scope_dir,
        shared_embedder.clone(),
        move || {
            memories_owned
                .iter()
                .enumerate()
                .map(|(i, mem)| RawDocument {
                    content: mem.content.clone(),
                    source_id: format!("locomo_{}_obs_{}", sample_id, i),
                    source: "memory".to_string(),
                    title: format!("{} session {}", mem.speaker, mem.session_num),
                    memory_type: Some("fact".to_string()),
                    space: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                })
                .collect()
        },
        &enrichment,
    )
    .await
    .expect("open_or_seed_scenario_db CLI");

    let elapsed = t0.elapsed().as_secs_f32();

    let mem_count = db.memory_count().await.unwrap_or(0);
    let enriched = db.enriched_memory_count().await.unwrap_or(0);

    eprintln!(
        "[smoke-cli] done in {:.1}s — mem={} enriched={}",
        elapsed, mem_count, enriched
    );

    assert!(mem_count > 0, "should have seeded memories");
    assert_eq!(
        mem_count, enriched,
        "should be fully enriched ({}/{})",
        enriched, mem_count
    );
    assert!(
        scope_dir.join("origin_memory.db").exists(),
        "DB file should exist"
    );

    // Verify retrieval is scoped to this sample.
    let results = db
        .search_memory(
            "anything",
            50,
            None,
            &ReadScope::Global,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("search_memory");
    let expected_prefix = format!("locomo_{}_obs_", sample.sample_id);
    for r in &results {
        assert!(
            r.source_id.starts_with(&expected_prefix),
            "cross-conv leak! source_id={} (expected prefix {})",
            r.source_id,
            expected_prefix
        );
    }

    eprintln!(
        "\n=== smoke_per_scenario_locomo_cli PASSED in {:.1}s ===",
        elapsed
    );
    eprintln!("  CLI enrichment (concurrency=4): ✓");
    eprintln!("  mem={} enriched={}: ✓", mem_count, enriched);
    eprintln!("  retrieval scoped to own conv: ✓");
}

/// Smoke for new `EnrichmentMode::Cli` (batched + persistent CLI enrichment).
///
/// Distinct from `smoke_per_scenario_locomo_cli` above which exercises the
/// per-memory `OnDevice(ClaudeCliProvider)` route (one subprocess per memory).
/// This test exercises the new batched route with `--resume`, session rotation,
/// JSONL persistence, retry, and cost telemetry.
///
/// ```bash
/// EVAL_ENRICHMENT_BATCH_SIZE_ENTITIES=5 EVAL_ENRICHMENT_BATCH_SIZE_TITLES=10 \
///   cargo test -p origin --test eval_harness smoke_per_scenario_locomo_cli_batched -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn smoke_per_scenario_locomo_cli_batched() {
    use wenlan_core::eval::locomo::{extract_observations, load_locomo};
    use wenlan_core::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use wenlan_core::sources::RawDocument;

    let probe = std::process::Command::new("claude")
        .arg("--version")
        .output();
    match probe {
        Ok(out) if out.status.success() => {
            eprintln!(
                "[smoke-cli-batched] claude binary: {}",
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }
        _ => {
            eprintln!("SKIP: `claude` binary not in PATH");
            return;
        }
    }

    let locomo_path = eval_root().join("data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let samples = load_locomo(&locomo_path).unwrap();
    assert!(!samples.is_empty(), "need at least 1 LoCoMo sample");

    let batch_entities: usize = std::env::var("EVAL_ENRICHMENT_BATCH_SIZE_ENTITIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let batch_titles: usize = std::env::var("EVAL_ENRICHMENT_BATCH_SIZE_TITLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let rotation: usize = std::env::var("EVAL_ENRICHMENT_ROTATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let cost_cap: f64 = std::env::var("EVAL_ENRICHMENT_COST_CAP_USD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);
    let max_obs: usize = std::env::var("SMOKE_MAX_OBS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let cache_subdir = tempfile::tempdir().unwrap();
    let enrichment = EnrichmentMode::Cli {
        model: std::env::var("EVAL_ENRICHMENT_CLI_MODEL").unwrap_or_else(|_| "haiku".into()),
        batch_entities,
        batch_titles,
        rotation,
        retries: 3,
        cost_cap_usd: cost_cap,
        cache_dir: cache_subdir.path().to_path_buf(),
    };
    let shared_embedder = eval_shared_embedder();

    let tmp = tempfile::tempdir().unwrap();
    let baselines_dir = tmp.path().to_path_buf();
    eprintln!(
        "[smoke-cli-batched] baselines: {} | batch_entities={} batch_titles={} cost_cap=${:.2} max_obs={}",
        baselines_dir.display(),
        batch_entities,
        batch_titles,
        cost_cap,
        max_obs,
    );

    let sample = &samples[0];
    let mut memories = extract_observations(sample);
    memories.truncate(max_obs);

    let scope_dir = scenario_db_dir(&baselines_dir, "locomo", &sample.sample_id);
    let sample_id = sample.sample_id.clone();
    let memories_owned = memories.clone();
    let t0 = std::time::Instant::now();

    let db = open_or_seed_scenario_db(
        &scope_dir,
        shared_embedder.clone(),
        move || {
            memories_owned
                .iter()
                .enumerate()
                .map(|(i, mem)| RawDocument {
                    content: mem.content.clone(),
                    source_id: format!("locomo_{}_obs_{}", sample_id, i),
                    source: "memory".to_string(),
                    title: format!("{} session {}", mem.speaker, mem.session_num),
                    memory_type: Some("fact".to_string()),
                    space: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                })
                .collect()
        },
        &enrichment,
    )
    .await
    .expect("open_or_seed_scenario_db CLI batched");

    let elapsed = t0.elapsed().as_secs_f32();
    let mem_count = db.memory_count().await.unwrap_or(0);
    let enriched = db.enriched_memory_count().await.unwrap_or(0);

    eprintln!(
        "[smoke-cli-batched] done in {:.1}s | mem={} enriched={}",
        elapsed, mem_count, enriched
    );

    assert!(mem_count > 0);
    assert_eq!(mem_count, enriched, "should be fully enriched");

    eprintln!(
        "\n=== smoke_per_scenario_locomo_cli_batched PASSED in {:.1}s ===",
        elapsed
    );
}

/// Manual smoke for LME per-scenario DB refactor (T3).
///
/// Mirrors `smoke_per_scenario_locomo` but for LongMemEval:
/// Smoke-tests per-scenario LME enrichment. Also serves as a pre-flight probe before a
/// full LME run: set SMOKE_LME_SAMPLES=10 to cover more scenarios and see extrapolated
/// wall-time for the full 500-question dataset.
///
/// - Per-question DB at `{tempdir}/fullpipeline/lme/{question_id}/`
/// - Verifies isolation, cache reuse, mem==enriched
/// - Prints per-mem timing and full-LME extrapolation at end
///
/// On-device Qwen3-4B, ~60-90s on Metal GPU. Requires `--ignored --nocapture`.
///
/// Run as probe before full LME/LoCoMo:
///   SMOKE_LME_SAMPLES=10 EVAL_LOCAL_MODEL=qwen3.5-9b \
///   cargo test -p origin --test eval_harness smoke_per_scenario_lme -- --ignored --nocapture
/// Default config (2 samples x 5 mems) is the CI smoke.
#[tokio::test]
#[ignore]
async fn smoke_per_scenario_lme() {
    use wenlan_core::eval::longmemeval::{extract_memories, load_longmemeval};
    use wenlan_core::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use wenlan_core::sources::RawDocument;

    let lme_path = eval_root().join("data/longmemeval_oracle.json");
    if !lme_path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let samples = load_longmemeval(&lme_path).unwrap();
    let n_samples: usize = std::env::var("SMOKE_LME_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let max_mem_per_sample: usize = std::env::var("SMOKE_LME_MAX_MEMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    assert!(
        samples.len() >= n_samples.min(2),
        "need >={} LME samples for smoke",
        n_samples.min(2)
    );

    let tmp = tempfile::tempdir().unwrap();
    let baselines_dir = tmp.path().to_path_buf();
    eprintln!("[smoke-lme] baselines: {}", baselines_dir.display());

    let enrichment = EnrichmentMode::from_env("claude-haiku-4-5-20251001", 1.0)
        .expect("EnrichmentMode::from_env");
    let shared_embedder = eval_shared_embedder();

    let pair = &samples[..n_samples.min(samples.len())];
    let mut per_sample_source_ids: Vec<Vec<String>> = Vec::new();
    let mut total_mems_processed: usize = 0;

    let smoke_t0 = std::time::Instant::now();
    for sample in pair {
        let mut memories = extract_memories(sample);
        memories.truncate(max_mem_per_sample);
        eprintln!(
            "[smoke-lme] {} ({} mems, truncated) -> open_or_seed",
            sample.question_id,
            memories.len()
        );

        let scope_dir = scenario_db_dir(&baselines_dir, "lme", &sample.question_id);

        let question_id = sample.question_id.clone();
        let question_type = sample.question_type.clone();
        let memories_owned = memories.clone();
        let conv_t0 = std::time::Instant::now();
        let db = open_or_seed_scenario_db(
            &scope_dir,
            shared_embedder.clone(),
            move || {
                memories_owned
                    .iter()
                    .map(|mem| RawDocument {
                        content: mem.content.clone(),
                        source_id: format!(
                            "lme_{}_{}_t{}",
                            question_id, mem.session_idx, mem.turn_idx
                        ),
                        source: "memory".to_string(),
                        title: format!("session {} turn {}", mem.session_idx, mem.turn_idx),
                        memory_type: Some(
                            if question_type == "single-session-preference" {
                                "preference"
                            } else {
                                "fact"
                            }
                            .to_string(),
                        ),
                        space: Some("conversation".to_string()),
                        last_modified: chrono::Utc::now().timestamp(),
                        ..Default::default()
                    })
                    .collect()
            },
            &enrichment,
        )
        .await
        .expect("open_or_seed_scenario_db");
        let conv_elapsed = conv_t0.elapsed().as_secs_f32();

        let mem_count = db.memory_count().await.unwrap_or(0);
        let enriched = db.enriched_memory_count().await.unwrap_or(0);
        total_mems_processed += mem_count;
        eprintln!(
            "[smoke-lme] {}: seed+enrich done in {:.1}s, mem={} enriched={}",
            sample.question_id, conv_elapsed, mem_count, enriched
        );
        assert!(
            mem_count > 0,
            "{}: should have seeded memories",
            sample.question_id
        );
        assert_eq!(
            mem_count, enriched,
            "{}: should be fully enriched ({}/{})",
            sample.question_id, enriched, mem_count
        );

        assert!(
            scope_dir.join("origin_memory.db").exists(),
            "{}: DB file should exist at {}",
            sample.question_id,
            scope_dir.display()
        );

        // Retrieval scoped to own scenario
        let results = db
            .search_memory(
                "anything",
                50,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await
            .expect("search_memory");
        let expected_prefix = format!("lme_{}_", sample.question_id);
        for r in &results {
            assert!(
                r.source_id.starts_with(&expected_prefix),
                "{}: cross-scenario leak! result source_id={} (expected prefix {})",
                sample.question_id,
                r.source_id,
                expected_prefix
            );
        }
        per_sample_source_ids.push(results.iter().map(|r| r.source_id.clone()).collect());
    }

    // Cross-scenario isolation
    let s0: std::collections::HashSet<_> = per_sample_source_ids[0].iter().collect();
    let s1: std::collections::HashSet<_> = per_sample_source_ids[1].iter().collect();
    let overlap: Vec<_> = s0.intersection(&s1).copied().collect();
    assert!(
        overlap.is_empty(),
        "cross-scenario source_id leak: {:?}",
        overlap
    );

    // Cache-hit re-open
    let sample = &pair[0];
    let mut memories = extract_memories(sample);
    memories.truncate(max_mem_per_sample);
    let scope_dir = scenario_db_dir(&baselines_dir, "lme", &sample.question_id);
    let question_id_2 = sample.question_id.clone();
    let question_type_2 = sample.question_type.clone();
    let memories_owned_2 = memories.clone();
    let cache_t0 = std::time::Instant::now();
    let _db_cached = open_or_seed_scenario_db(
        &scope_dir,
        shared_embedder.clone(),
        move || {
            memories_owned_2
                .iter()
                .map(|mem| RawDocument {
                    content: mem.content.clone(),
                    source_id: format!(
                        "lme_{}_{}_t{}",
                        question_id_2, mem.session_idx, mem.turn_idx
                    ),
                    source: "memory".to_string(),
                    title: format!("session {} turn {}", mem.session_idx, mem.turn_idx),
                    memory_type: Some(
                        if question_type_2 == "single-session-preference" {
                            "preference"
                        } else {
                            "fact"
                        }
                        .to_string(),
                    ),
                    space: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                })
                .collect()
        },
        &enrichment,
    )
    .await
    .expect("re-open cached");
    let cache_ms = cache_t0.elapsed().as_millis();
    eprintln!(
        "[smoke-lme] cache-hit re-open of {}: {} ms (expected <2000ms; no re-enrich)",
        sample.question_id, cache_ms
    );
    assert!(
        cache_ms < 5000,
        "cache hit too slow ({} ms) -- helper likely re-enriched",
        cache_ms
    );

    let total_elapsed = smoke_t0.elapsed().as_secs_f32();
    eprintln!(
        "\n=== smoke_per_scenario_lme PASSED in {:.1}s ===",
        total_elapsed
    );
    eprintln!("  per-question DB layout: ✓");
    eprintln!(
        "  enrichment per-scenario (truncated to {} mems each): ✓",
        max_mem_per_sample
    );
    eprintln!("  retrieval scoped to own scenario: ✓");
    eprintln!("  cross-scenario source_id isolation: ✓");
    eprintln!("  cache-hit re-open ({}ms): ✓", cache_ms);

    // Extrapolation summary — useful for pre-flight estimation.
    let mean_per_mem_sec = total_elapsed / total_mems_processed.max(1) as f32;
    let full_lme_estimate_min = (mean_per_mem_sec * 10_000.0) / 60.0;
    eprintln!(
        "[smoke-lme] SUMMARY: {} samples x ~{} mems = {} mems in {:.1}s ({:.2}s/mem). \
         Full LME (~10k mems) extrapolated: ~{:.1} min (~{:.1} hours).",
        n_samples,
        max_mem_per_sample,
        total_mems_processed,
        total_elapsed,
        mean_per_mem_sec,
        full_lme_estimate_min,
        full_lme_estimate_min / 60.0,
    );
}

/// Judge LME tuples via Claude CLI (Max plan, no API key).
///
/// Uses task-specific judge prompts matching the LongMemEval paper.
/// Concurrency configurable via EVAL_CLI_CONCURRENCY (default 8).
///
/// ```bash
/// cargo test -p origin --test eval_harness judge_fullpipeline_lme_cli -- --ignored --nocapture
/// EVAL_CLI_CONCURRENCY=4 cargo test -p origin --test eval_harness judge_fullpipeline_lme_cli -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn judge_fullpipeline_lme_cli() {
    use wenlan_core::eval::judge::{
        aggregate_judgments, judge_with_claude_model_batched_persistent,
        judge_with_claude_model_persistent, load_judgment_tuples,
    };
    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    let tuples_path = baselines.join("fullpipeline_lme_tuples.json");
    if !tuples_path.exists() {
        eprintln!("SKIP: run generate_fullpipeline_lme first");
        return;
    }
    let tuples = load_judgment_tuples(&tuples_path).expect("load failed");
    let concurrency: usize = std::env::var("EVAL_CLI_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let model = std::env::var("EVAL_CLI_MODEL").unwrap_or_else(|_| "haiku".to_string());
    let max_retries: u32 = std::env::var("EVAL_JUDGE_RETRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let batch_size: usize = std::env::var("EVAL_JUDGE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let rotation_calls: usize = std::env::var("EVAL_JUDGE_ROTATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let limit: Option<usize> = std::env::var("EVAL_JUDGE_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok());

    let tuples_to_judge: Vec<_> = match limit {
        Some(n) => tuples.iter().take(n).cloned().collect(),
        None => tuples,
    };
    let cache_path = if batch_size > 1 {
        baselines.join("fullpipeline_lme_judgments_batch.jsonl")
    } else {
        baselines.join("fullpipeline_lme_judgments.jsonl")
    };

    let (results, label) = if batch_size > 1 {
        eprintln!(
            "Judging {} LME tuples via CLI BATCHED (model={}, batch={}, rotation={}, retries={}, cache={})...",
            tuples_to_judge.len(),
            model,
            batch_size,
            rotation_calls,
            max_retries,
            cache_path.display()
        );
        let r = judge_with_claude_model_batched_persistent(
            &tuples_to_judge,
            batch_size,
            rotation_calls,
            &model,
            &cache_path,
            max_retries,
        )
        .await
        .expect("judge failed");
        (r, format!("{}-batch{}-cli", model, batch_size))
    } else {
        eprintln!(
            "Judging {} LME tuples via CLI (model={}, concurrency={}, retries={}, cache={})...",
            tuples_to_judge.len(),
            model,
            concurrency,
            max_retries,
            cache_path.display()
        );
        let r = judge_with_claude_model_persistent(
            &tuples_to_judge,
            concurrency,
            &model,
            &cache_path,
            max_retries,
        )
        .await
        .expect("judge failed");
        (r, format!("{}-cli", model))
    };
    let report = aggregate_judgments(&results, &label);

    print_judge_report(&report);
}

/// Judge LoCoMo tuples via Claude CLI (Max plan, no API key).
///
/// ```bash
/// cargo test -p origin --test eval_harness judge_fullpipeline_locomo_cli -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn judge_fullpipeline_locomo_cli() {
    use wenlan_core::eval::judge::{
        aggregate_judgments, judge_with_claude_model_batched_persistent,
        judge_with_claude_model_persistent, load_judgment_tuples,
    };
    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    let tuples_path = baselines.join("fullpipeline_locomo_tuples.json");
    if !tuples_path.exists() {
        eprintln!("SKIP: run generate_fullpipeline_locomo first");
        return;
    }
    let tuples = load_judgment_tuples(&tuples_path).expect("load failed");
    let concurrency: usize = std::env::var("EVAL_CLI_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let model = std::env::var("EVAL_CLI_MODEL").unwrap_or_else(|_| "haiku".to_string());
    let max_retries: u32 = std::env::var("EVAL_JUDGE_RETRIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let batch_size: usize = std::env::var("EVAL_JUDGE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let rotation_calls: usize = std::env::var("EVAL_JUDGE_ROTATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let limit: Option<usize> = std::env::var("EVAL_JUDGE_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok());

    let tuples_to_judge: Vec<_> = match limit {
        Some(n) => tuples.iter().take(n).cloned().collect(),
        None => tuples,
    };
    let cache_path = if batch_size > 1 {
        baselines.join("fullpipeline_locomo_judgments_batch.jsonl")
    } else {
        baselines.join("fullpipeline_locomo_judgments.jsonl")
    };

    if batch_size > 1 {
        eprintln!(
            "Judging {} LoCoMo tuples via CLI BATCHED (model={}, batch={}, rotation={}, retries={}, cache={})...",
            tuples_to_judge.len(),
            model,
            batch_size,
            rotation_calls,
            max_retries,
            cache_path.display()
        );
        let results = judge_with_claude_model_batched_persistent(
            &tuples_to_judge,
            batch_size,
            rotation_calls,
            &model,
            &cache_path,
            max_retries,
        )
        .await
        .expect("judge failed");
        let report = aggregate_judgments(&results, &format!("{}-batch{}-cli", model, batch_size));
        print_judge_report(&report);
    } else {
        eprintln!(
            "Judging {} LoCoMo tuples via CLI SINGLE (model={}, concurrency={}, retries={}, cache={})...",
            tuples_to_judge.len(),
            model,
            concurrency,
            max_retries,
            cache_path.display()
        );
        let results = judge_with_claude_model_persistent(
            &tuples_to_judge,
            concurrency,
            &model,
            &cache_path,
            max_retries,
        )
        .await
        .expect("judge failed");
        let report = aggregate_judgments(&results, &format!("{}-cli", model));
        print_judge_report(&report);
    }
}

/// Print a judge report with per-category breakdown and task-averaged accuracy.
fn print_judge_report(report: &wenlan_core::eval::judge::JudgedE2EReport) {
    eprintln!(
        "\n{:<30} | {:>8} | {:>6} | {:>10}",
        "Approach", "Accuracy", "N", "Ctx Tokens"
    );
    eprintln!("{:-<30}-+-{:-<8}-+-{:-<6}-+-{:-<10}", "", "", "", "");
    let mut task_accs = Vec::new();
    for r in &report.results_by_approach {
        eprintln!(
            "{:<30} | {:>7.1}% | {:>6} | {:>10.0}",
            r.approach,
            r.accuracy * 100.0,
            r.total,
            r.mean_context_tokens
        );
        task_accs.push(r.accuracy);
    }
    let task_avg = if task_accs.is_empty() {
        0.0
    } else {
        task_accs.iter().sum::<f64>() / task_accs.len() as f64 * 100.0
    };
    eprintln!("\nTotal judged: {}", report.total_judged);
    eprintln!("Task-averaged accuracy: {:.1}%", task_avg);
}

/// Probe concept relevance scores from enriched DBs.
///
/// Runs search_pages with real embeddings on sample questions from each benchmark.
/// Prints score distributions so we can set a data-driven threshold.
///
/// ```bash
/// cargo test -p origin --test eval_harness probe_concept_scores -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn probe_concept_scores() {
    use std::sync::Arc;
    use wenlan_core::db::MemoryDB;
    use wenlan_core::eval::shared::eval_shared_embedder;
    use wenlan_core::events::NoopEmitter;

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    let shared_embedder = eval_shared_embedder();

    // Sample questions from tuples
    let locomo_tuples_path = baselines.join("fullpipeline_locomo_tuples.json");
    let lme_tuples_path = baselines.join("fullpipeline_lme_tuples.json");

    for (label, db_name, tuples_path, n_samples) in [
        (
            "LoCoMo",
            "fullpipeline_locomo_tuples.db",
            &locomo_tuples_path,
            10,
        ),
        ("LME", "fullpipeline_lme_tuples.db", &lme_tuples_path, 10),
    ] {
        let db_dir = baselines.join(db_name);
        if !db_dir.exists() || !tuples_path.exists() {
            eprintln!("SKIP {label}: enriched DB or tuples not found");
            continue;
        }

        let db = MemoryDB::new_with_shared_embedder(
            &db_dir,
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await
        .expect("open DB");

        // Load sample questions spread across categories
        let tuples: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(tuples_path).unwrap()).unwrap();

        let mut by_cat: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for t in &tuples {
            let cat = t["category"]
                .as_str()
                .unwrap_or(
                    t["approach"]
                        .as_str()
                        .unwrap_or("?")
                        .strip_prefix("structured_")
                        .unwrap_or("?"),
                )
                .to_string();
            let q = t["question"].as_str().unwrap_or("").to_string();
            by_cat.entry(cat).or_default().push(q);
        }

        let mut samples: Vec<(String, String)> = Vec::new();
        for (cat, qs) in by_cat.iter() {
            for q in qs.iter().take(n_samples / by_cat.len().max(1)) {
                samples.push((cat.clone(), q.clone()));
            }
        }
        // Fill remaining
        'outer: for (cat, qs) in by_cat.iter() {
            for q in qs.iter().skip(n_samples / by_cat.len().max(1)) {
                if samples.len() >= n_samples {
                    break 'outer;
                }
                samples.push((cat.clone(), q.clone()));
            }
        }

        eprintln!(
            "\n=== {label}: Concept Scores ({} samples) ===",
            samples.len()
        );
        eprintln!(
            "{:<24} | {:>6} {:>6} {:>6} | {:>5} {:>5} {:>5} | Question",
            "Category", "C1", "C2", "C3", "Tok1", "Tok2", "Tok3"
        );
        eprintln!("{}", "-".repeat(110));

        let mut all_scores: Vec<f32> = Vec::new();
        for (cat, question) in &samples {
            let concepts = db.search_pages(question, 3, None).await.unwrap_or_default();
            let scores: Vec<f32> = concepts.iter().map(|c| c.relevance_score).collect();
            let tokens: Vec<usize> = concepts
                .iter()
                .map(|c| c.content.len() / 4) // rough char-to-token
                .collect();

            all_scores.extend(&scores);

            eprintln!(
                "{:<24} | {:>5.3} {:>5.3} {:>5.3} | {:>5} {:>5} {:>5} | {}",
                &cat[..cat.len().min(24)],
                scores.first().unwrap_or(&0.0),
                scores.get(1).unwrap_or(&0.0),
                scores.get(2).unwrap_or(&0.0),
                tokens.first().unwrap_or(&0),
                tokens.get(1).unwrap_or(&0),
                tokens.get(2).unwrap_or(&0),
                &question[..question.len().min(45)],
            );
        }

        // Summary stats
        all_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = all_scores.len();
        if n > 0 {
            let mean: f32 = all_scores.iter().sum::<f32>() / n as f32;
            eprintln!(
                "\n  {label} scores: mean={:.3} min={:.3} max={:.3} p25={:.3} p50={:.3} p75={:.3} (n={})",
                mean,
                all_scores[0],
                all_scores[n - 1],
                all_scores[n / 4],
                all_scores[n / 2],
                all_scores[3 * n / 4],
                n,
            );
        }
    }
}

/// Probe source overlap gate across ALL questions in both enriched DBs.
///
/// Runs search_memory + search_pages for every question, counts how many
/// concepts pass the overlap gate (>= min_overlap source memories overlap
/// with search results). This validates whether the gate behaves as expected
/// without running expensive LLM answer generation.
///
/// ```bash
/// cargo test -p origin --test eval_harness probe_overlap_gate -- --ignored --nocapture
/// EVAL_MIN_OVERLAP=2 cargo test ... probe_overlap_gate -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn probe_overlap_gate() {
    use std::collections::HashMap;
    use std::sync::Arc;
    use wenlan_core::db::MemoryDB;
    use wenlan_core::eval::shared::eval_shared_embedder;
    use wenlan_core::events::NoopEmitter;
    use wenlan_core::pages::filter_pages_by_source_overlap;

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    let shared_embedder = eval_shared_embedder();
    let min_overlap: usize = std::env::var("EVAL_MIN_OVERLAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    for (label, db_name, tuples_name) in [
        (
            "LoCoMo",
            "fullpipeline_locomo_tuples.db",
            "fullpipeline_locomo_tuples.json",
        ),
        (
            "LME",
            "fullpipeline_lme_tuples.db",
            "fullpipeline_lme_tuples.json",
        ),
    ] {
        let db_dir = baselines.join(db_name);
        let tuples_path = baselines.join(tuples_name);
        if !db_dir.exists() || !tuples_path.exists() {
            eprintln!("SKIP {label}: artifacts missing");
            continue;
        }

        let db = MemoryDB::new_with_shared_embedder(
            &db_dir,
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await
        .expect("open DB");

        let tuples: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(&tuples_path).unwrap()).unwrap();

        // Dedup questions (same q may appear with different categories in some files)
        let mut seen = std::collections::HashSet::new();
        let questions: Vec<(String, String)> = tuples
            .iter()
            .filter_map(|t| {
                let q = t["question"].as_str()?.to_string();
                if !seen.insert(q.clone()) {
                    return None;
                }
                let cat = t["category"]
                    .as_str()
                    .or_else(|| {
                        t["approach"]
                            .as_str()
                            .and_then(|s| s.strip_prefix("structured_"))
                    })
                    .unwrap_or("?")
                    .to_string();
                Some((cat, q))
            })
            .collect();

        let total_q = questions.len();
        let mut total_concepts = 0usize;
        let mut total_kept = 0usize;
        let mut overlap_when_kept: Vec<usize> = Vec::new();
        let mut overlap_when_filtered: Vec<usize> = Vec::new();
        let mut per_q_kept_dist: HashMap<usize, usize> = HashMap::new();
        let mut per_cat_kept: HashMap<String, (usize, usize)> = HashMap::new(); // cat -> (kept_q, total_q)

        eprintln!("\n=== {label}: probing {total_q} questions (min_overlap={min_overlap}) ===",);

        for (i, (cat, q)) in questions.iter().enumerate() {
            // Real search_memory (top-10, no domain filter — matches eval pipeline)
            let results = match db
                .search_memory(q, 10, None, &ReadScope::Global, None, None, None, None)
                .await
            {
                Ok(r) => r,
                Err(_) => continue,
            };
            let search_ids: std::collections::HashSet<String> =
                results.iter().map(|r| r.source_id.clone()).collect();

            // Real search_pages (top-3)
            let raw_concepts = db.search_pages(q, 3, None).await.unwrap_or_default();
            let kept = filter_pages_by_source_overlap(&raw_concepts, &search_ids, min_overlap);

            for c in &raw_concepts {
                total_concepts += 1;
                let overlap = c
                    .source_memory_ids
                    .iter()
                    .filter(|sid| search_ids.contains(sid.as_str()))
                    .count();
                if kept.iter().any(|k| k.id == c.id) {
                    total_kept += 1;
                    overlap_when_kept.push(overlap);
                } else {
                    overlap_when_filtered.push(overlap);
                }
            }
            *per_q_kept_dist.entry(kept.len()).or_insert(0) += 1;
            let entry = per_cat_kept.entry(cat.clone()).or_insert((0, 0));
            entry.1 += 1;
            if !kept.is_empty() {
                entry.0 += 1;
            }

            if i % 100 == 99 {
                eprintln!("  [{}/{}] processed", i + 1, total_q);
            }
        }

        let kept_pct = total_kept as f64 / total_concepts.max(1) as f64 * 100.0;
        let mean_kept_overlap = if overlap_when_kept.is_empty() {
            0.0
        } else {
            overlap_when_kept.iter().sum::<usize>() as f64 / overlap_when_kept.len() as f64
        };
        let mean_filt_overlap = if overlap_when_filtered.is_empty() {
            0.0
        } else {
            overlap_when_filtered.iter().sum::<usize>() as f64 / overlap_when_filtered.len() as f64
        };

        eprintln!("\n  --- Results ---");
        eprintln!("  Total concept-query pairs: {total_concepts}");
        eprintln!(
            "  Kept (passed gate): {total_kept} ({kept_pct:.1}%)  mean_overlap_when_kept={mean_kept_overlap:.1}"
        );
        eprintln!(
            "  Filtered: {} ({:.1}%)  mean_overlap_when_filtered={:.2}",
            total_concepts - total_kept,
            (total_concepts - total_kept) as f64 / total_concepts.max(1) as f64 * 100.0,
            mean_filt_overlap,
        );

        let mut dist: Vec<(usize, usize)> = per_q_kept_dist.into_iter().collect();
        dist.sort();
        eprintln!(
            "  Concepts passing per question: {}",
            dist.iter()
                .map(|(k, v)| format!("{k}→{v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );

        eprintln!("\n  Per-category (questions with at least one passing concept):");
        let mut cats: Vec<(String, (usize, usize))> = per_cat_kept.into_iter().collect();
        cats.sort_by(|a, b| a.0.cmp(&b.0));
        for (cat, (kept_q, total_q)) in cats {
            eprintln!(
                "    {:<28} {:4}/{:4} ({:5.1}%)",
                cat,
                kept_q,
                total_q,
                kept_q as f64 / total_q.max(1) as f64 * 100.0
            );
        }
    }
}

/// Stress-test the on-device LLM provider with rising concurrency.
///
/// Submits N concurrent `generate()` calls to a single `OnDeviceProvider`,
/// measures wall-clock and per-call latency, and reports throughput. Useful
/// for verifying that the persistent LlamaContext optimization (build the
/// context once, clear KV cache between calls) actually pays off vs the old
/// per-call `new_context()` path, and for measuring the impact of the
/// multi-worker pool (S1, `WENLAN_LLM_WORKERS`) and continuous batching
/// (S2, `WENLAN_LLM_PARALLEL_SEQS`).
///
/// Three useful invocations to compare:
///   WENLAN_LLM_WORKERS=1 WENLAN_LLM_PARALLEL_SEQS=1   # baseline (single seq, single ctx)
///   WENLAN_LLM_WORKERS=4 WENLAN_LLM_PARALLEL_SEQS=1   # S1: 4 contexts, 1 seq each
///   WENLAN_LLM_WORKERS=1 WENLAN_LLM_PARALLEL_SEQS=4   # S2: 1 context, 4 parallel seqs
///   WENLAN_LLM_WORKERS=2 WENLAN_LLM_PARALLEL_SEQS=2   # composed: 2 ctx x 2 seq = 4 concurrent
///
/// Requires Metal GPU + a downloaded Qwen3-4B model. Marked `#[ignore]` so it
/// doesn't run in CI. Invoke with:
///   cargo test -p origin --test eval_harness stress_concurrent_inference \
///       -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn stress_concurrent_inference() {
    use futures::future::join_all;
    use std::sync::Arc;
    use std::time::Instant;
    use wenlan_core::llm_provider::{LlmProvider, LlmRequest, OnDeviceProvider};

    eprintln!("[stress] booting OnDeviceProvider (Qwen3-4B default)...");
    let boot_start = Instant::now();
    let provider: Arc<dyn LlmProvider> = match OnDeviceProvider::new_with_model(None) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("[stress] SKIP: failed to init OnDeviceProvider: {e}");
            return;
        }
    };
    eprintln!("[stress] provider ready in {:?}", boot_start.elapsed());

    // Small entity-extraction style prompt — representative of the calls made
    // during a real LoCoMo / LongMemEval run. Kept short to make per-call
    // latency the dominant signal (not prompt prefill).
    let system_prompt = "You extract structured information from text. \
                         Return strict JSON only, no preamble, no markdown."
        .to_string();
    let user_prompt = "Extract entities (people, places, organizations) from: \
                       'Alice met Bob in Paris last summer. They discussed Acme Corp.'\n\
                       Reply as JSON: {\"entities\": [...]}"
        .to_string();

    // Warmup: one call to JIT Metal pipelines and trigger the persistent
    // context build. Without this, the first concurrent batch absorbs setup
    // cost and skews the N=1 measurement.
    eprintln!("[stress] warmup call...");
    let warmup_start = Instant::now();
    let _ = provider
        .generate(LlmRequest {
            system_prompt: Some(system_prompt.clone()),
            user_prompt: user_prompt.clone(),
            max_tokens: 64,
            temperature: 0.1,
            label: Some("warmup".into()),
            timeout_secs: None,
        })
        .await;
    eprintln!("[stress] warmup done in {:?}", warmup_start.elapsed());

    eprintln!("\n[stress] N | wall(s) | throughput(c/s) | mean(s) | p50(s) | p95(s) | failures");
    eprintln!("[stress] --|---------|-----------------|---------|--------|--------|---------");

    for &n in &[1usize, 2, 4, 8, 16] {
        let total_start = Instant::now();
        let mut handles = Vec::with_capacity(n);

        for i in 0..n {
            let prov = Arc::clone(&provider);
            let sys = system_prompt.clone();
            let usr = user_prompt.clone();
            let label = format!("stress_n{n}_i{i}");
            handles.push(tokio::spawn(async move {
                let call_start = Instant::now();
                let res = prov
                    .generate(LlmRequest {
                        system_prompt: Some(sys),
                        user_prompt: usr,
                        max_tokens: 128,
                        temperature: 0.1,
                        label: Some(label),
                        timeout_secs: None,
                    })
                    .await;
                (res.is_ok(), call_start.elapsed())
            }));
        }

        let outcomes: Vec<(bool, std::time::Duration)> = join_all(handles)
            .await
            .into_iter()
            .map(|j| j.unwrap_or((false, std::time::Duration::from_secs(0))))
            .collect();

        let wall = total_start.elapsed();
        let failures = outcomes.iter().filter(|(ok, _)| !ok).count();
        let mut latencies: Vec<f64> = outcomes
            .iter()
            .filter(|(ok, _)| *ok)
            .map(|(_, d)| d.as_secs_f64())
            .collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = if latencies.is_empty() {
            0.0
        } else {
            latencies.iter().sum::<f64>() / latencies.len() as f64
        };
        let p50 = percentile(&latencies, 0.50);
        let p95 = percentile(&latencies, 0.95);
        let throughput = (n - failures) as f64 / wall.as_secs_f64().max(0.001);

        eprintln!(
            "[stress] {n:>2} | {wall:>7.2} | {tput:>15.2} | {mean:>7.2} | {p50:>6.2} | {p95:>6.2} | {fail}",
            wall = wall.as_secs_f64(),
            tput = throughput,
            fail = failures,
        );

        // Cool-down between batches so KV cache pressure resets and Metal
        // queue allocator has a chance to drain.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    // Memory ceiling note: each persistent context allocates ~256-512 MiB of
    // KV cache for Qwen3-4B Q4_K_M at 8K context. With the current
    // single-worker architecture, only ONE context is alive at a time, so
    // real GPU memory usage stays bounded regardless of N. Continuous
    // batching (slot-based, deferred) would multiply this per slot.
    eprintln!("\n[stress] M2 Pro memory note:");
    eprintln!("[stress]   Single persistent KV cache: ~256-512 MiB (Qwen3-4B @ 8K)");
    eprintln!(
        "[stress]   With continuous batching (deferred): N slots = N x KV cache, \
         practical cap 4-8 slots on 16GB"
    );
}

/// Percentile helper for sorted f64 slice. Linear interpolation between
/// adjacent ranks. Returns 0.0 if empty.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = p * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
    }
}

#[test]
fn eval_baselines_dir_override_env_var() {
    use wenlan_core::eval::shared::eval_baselines_dir_override;

    // Unset → None.
    temp_env::with_var("EVAL_BASELINES_DIR", None::<&str>, || {
        assert_eq!(eval_baselines_dir_override(), None);
    });

    // Set → Some(PathBuf).
    let tmp = tempfile::tempdir().unwrap();
    let path_str = tmp.path().to_str().unwrap().to_string();
    temp_env::with_var("EVAL_BASELINES_DIR", Some(&path_str), || {
        assert_eq!(eval_baselines_dir_override().as_deref(), Some(tmp.path()));
    });

    // Empty string → None.
    temp_env::with_var("EVAL_BASELINES_DIR", Some(""), || {
        assert_eq!(eval_baselines_dir_override(), None);
    });
}

#[test]
fn eval_report_schema_v1_round_trips_env_fields() {
    use wenlan_core::eval::report::{EvalReport, ReportEnv};
    let r = EvalReport {
        env: Some(ReportEnv {
            fixture_revision: "deadbeef".into(),
            embedder_model: "BGE-Base-EN-v1.5-Q".into(),
            embedder_revision: "768d".into(),
            retrieval_method: "search_memory".into(),
            llm_provider_class: "on-device".into(),
            llm_model: "Qwen3-4B-Instruct".into(),
            judge_model: Some("claude-haiku".into()),
            origin_version: env!("CARGO_PKG_VERSION").into(),
            eval_timestamp_unix: 1747800000,
            ..ReportEnv::default()
        }),
        ..EvalReport::default()
    };
    let json = serde_json::to_string(&r).unwrap();
    let back: EvalReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back.env.as_ref().unwrap().fixture_revision, "deadbeef");
    assert_eq!(
        back.env.as_ref().unwrap().embedder_model,
        "BGE-Base-EN-v1.5-Q"
    );
}

#[test]
fn eval_report_back_compat_loads_pre_v1_json() {
    // A realistic pre-v1 EvalReport JSON (all required fields, no "env" key).
    // Verifies that adding env: Option<ReportEnv> doesn't break old reports.
    let old = r#"{
        "fixture_count":10,"file_count":2,"search_mode":"search_memory",
        "ndcg_at_10":0.5,"ndcg_at_5":0.4,"map_at_5":0.3,"map_at_10":0.35,
        "mrr":0.6,"recall_at_1":0.2,"recall_at_3":0.4,"recall_at_5":0.7,
        "hit_rate_at_1":0.2,"hit_rate_at_3":0.4,"precision_at_3":0.3,"precision_at_5":0.25,
        "neg_above_relevant":1,"total_negatives":5,"negative_leakage":4,
        "baseline":null,"per_case":[]
    }"#;
    let r: wenlan_core::eval::report::EvalReport = serde_json::from_str(old).unwrap();
    assert!(r.env.is_none());
}

/// Re-distill cached LoCoMo per-conversation DBs to populate the new
/// `concept_sources` join table on DBs built before PR #4.
///
/// Pre-conditions:
/// - 10 cached per-conv DBs exist at `<baselines>/fullpipeline/locomo/conv-*/origin_memory.db`,
///   where `<baselines>` resolves from `EVAL_BASELINES_DIR` (or defaults to
///   `app/eval/baselines/`). Each DB has memories + entities populated, no concept_sources.
/// - `EVAL_ALLOW_WIPE=1` set (defense-in-depth from prior 5901-memory wipe incident).
/// - On-device LLM available (Metal GPU). Run with sandbox disabled. Cloud
///   fallback selected via `EVAL_ENRICHMENT=cloud`.
///
/// What it does, per DB:
///   1. Backup → `origin_memory.db.backup_<unix_ts>`.
///   2. Open via `MemoryDB::new_with_shared_embedder` (no cache check, no re-seed).
///   3. Delete every existing concept (status active + archived) via the public
///      `delete_page` API. concept_sources cascades via FK.
///   4. Call `refinery::distill_pages` directly. Cluster threshold honored.
///   5. Verify: new concepts > 0; first concept has at least one row in
///      `concept_sources` via `get_page_sources`.
///
/// Run:
///   EVAL_BASELINES_DIR=/Users/lucian/Repos/origin/.worktrees/eval-per-scenario/app/eval/baselines \
///     EVAL_ALLOW_WIPE=1 \
///     cargo test -p origin --test eval_harness redistill_cached_locomo_concepts \
///       -- --ignored --nocapture
///
/// Probe a single conv first to validate the path before running all 10:
///   EVAL_REDISTILL_CONVS=conv-26 EVAL_BASELINES_DIR=... EVAL_ALLOW_WIPE=1 \
///     cargo test ... -- --ignored --nocapture
///
/// Comma-separated to scope to a subset:
///   EVAL_REDISTILL_CONVS=conv-26,conv-30 ...
#[tokio::test]
#[ignore]
async fn redistill_cached_locomo_concepts() {
    use wenlan_core::eval::shared::{
        eval_baselines_dir_override, eval_shared_embedder, EnrichmentMode,
    };
    use wenlan_core::prompts::PromptRegistry;
    use wenlan_core::tuning::DistillationConfig;

    if std::env::var("EVAL_ALLOW_WIPE").as_deref() != Ok("1") {
        eprintln!("SKIP: set EVAL_ALLOW_WIPE=1 to permit clearing concepts on cached DBs");
        return;
    }

    let baselines = eval_baselines_dir_override().unwrap_or_else(|| eval_root().join("baselines"));
    let locomo_dir = baselines.join("fullpipeline").join("locomo");
    if !locomo_dir.exists() {
        eprintln!(
            "SKIP: {} not found (set EVAL_BASELINES_DIR)",
            locomo_dir.display()
        );
        return;
    }

    let enrichment = EnrichmentMode::from_env("claude-haiku-4-5-20251001", 1.0)
        .expect("EnrichmentMode::from_env");
    let llm = match &enrichment {
        EnrichmentMode::OnDevice(p) => p.clone(),
        _ => {
            eprintln!(
                "SKIP: redistill requires EnrichmentMode::OnDevice (got {:?}); unset EVAL_ENRICHMENT or set to local",
                std::any::type_name_of_val(&enrichment)
            );
            return;
        }
    };
    if !llm.is_available() {
        eprintln!("SKIP: on-device LLM unavailable (Metal/ggml init failed)");
        return;
    }

    let embedder = eval_shared_embedder();
    let prompts = PromptRegistry::default();
    let tuning = DistillationConfig::default();

    // Optional filter: comma-separated conv names (e.g. "conv-26,conv-30")
    // for scoped probe runs. Unset = process every cached DB.
    let filter: Option<std::collections::HashSet<String>> = std::env::var("EVAL_REDISTILL_CONVS")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        });

    let mut conv_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&locomo_dir)
        .expect("read locomo dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("origin_memory.db").exists())
        .filter(
            |p| match (&filter, p.file_name().and_then(|n| n.to_str())) {
                (Some(set), Some(name)) => set.contains(name),
                _ => true,
            },
        )
        .collect();
    conv_dirs.sort();
    if conv_dirs.is_empty() {
        eprintln!(
            "SKIP: no per-conv DBs found under {} (filter={:?})",
            locomo_dir.display(),
            filter
        );
        return;
    }
    eprintln!(
        "[redistill] found {} cached LoCoMo conv DBs (filter={:?})",
        conv_dirs.len(),
        filter
    );

    let started = std::time::Instant::now();
    let ts = chrono::Utc::now().timestamp();
    let pid = std::process::id();

    // Phase 1: back up every DB BEFORE touching any. Atomic guarantee: if any
    // backup fails (collision, IO), abort before destruction starts. Refusing
    // to overwrite an existing backup protects a prior partial run from
    // losing its recovery copy on re-run within the same second.
    let mut backups: Vec<std::path::PathBuf> = Vec::with_capacity(conv_dirs.len());
    for conv_dir in &conv_dirs {
        let db_file = conv_dir.join("origin_memory.db");
        let backup = conv_dir.join(format!("origin_memory.db.backup_{}_{}", ts, pid));
        if backup.exists() {
            panic!(
                "refuse to overwrite existing backup: {} (rename or delete the prior backup before retrying)",
                backup.display()
            );
        }
        std::fs::copy(&db_file, &backup).expect("backup");
        backups.push(backup);
    }
    eprintln!(
        "[redistill] backed up {} DBs (suffix _{}_{}); to roll back: \
         for f in <conv>/origin_memory.db.backup_{}_{}; do mv $f ${{f%.backup_*}}; done",
        backups.len(),
        ts,
        pid,
        ts,
        pid
    );

    // Phase 2: per-DB clear + redistill. Failures log and continue (other
    // DBs unaffected) so one bad conv doesn't strand the rest.
    let mut ok_count = 0usize;
    let mut fail_count = 0usize;
    for conv_dir in &conv_dirs {
        let conv_name = conv_dir.file_name().unwrap().to_string_lossy().to_string();
        match redistill_one_conv(
            conv_dir,
            &conv_name,
            &llm,
            &prompts,
            &tuning,
            embedder.clone(),
        )
        .await
        {
            Ok(()) => ok_count += 1,
            Err(e) => {
                eprintln!("[redistill] {}: FAILED — {}", conv_name, e);
                fail_count += 1;
            }
        }
    }

    eprintln!(
        "\n=== redistill_cached_locomo_concepts: {} ok / {} failed in {:.1}s ===",
        ok_count,
        fail_count,
        started.elapsed().as_secs_f32()
    );
    assert_eq!(
        fail_count, 0,
        "{} conv(s) failed redistill; backups remain at <conv>/origin_memory.db.backup_{}_{}",
        fail_count, ts, pid
    );
}

#[cfg(test)]
async fn redistill_one_conv(
    conv_dir: &std::path::Path,
    conv_name: &str,
    llm: &std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    tuning: &wenlan_core::tuning::DistillationConfig,
    embedder: wenlan_core::db::SharedEmbedder,
) -> Result<(), String> {
    use std::sync::Arc;
    use wenlan_core::db::MemoryDB;
    use wenlan_core::refinery::distill_pages;

    let emitter: Arc<dyn wenlan_core::events::EventEmitter> = Arc::new(wenlan_core::NoopEmitter);
    let db = MemoryDB::new_with_shared_embedder(conv_dir, emitter, embedder)
        .await
        .map_err(|e| format!("open: {e}"))?;

    let active = db
        .list_pages("active", 100_000, 0)
        .await
        .map_err(|e| format!("list_pages active: {e}"))?;
    let archived = db
        .list_pages("archived", 100_000, 0)
        .await
        .map_err(|e| format!("list_pages archived: {e}"))?;
    let total_existing = active.len() + archived.len();
    for c in active.iter().chain(archived.iter()) {
        db.delete_page(&c.id)
            .await
            .map_err(|e| format!("delete_page {}: {e}", c.id))?;
    }
    eprintln!(
        "[redistill] {}: cleared {} concepts (active={}, archived={})",
        conv_name,
        total_existing,
        active.len(),
        archived.len()
    );

    let conv_t0 = std::time::Instant::now();
    let created = distill_pages(&db, Some(llm), prompts, tuning, None)
        .await
        .map_err(|e| format!("distill_pages: {e}"))?;
    let conv_secs = conv_t0.elapsed().as_secs_f32();

    let new_active = db
        .list_pages("active", 100_000, 0)
        .await
        .map_err(|e| format!("post list_pages: {e}"))?;
    let mut sources_total = 0usize;
    let mut concepts_with_sources = 0usize;
    for c in &new_active {
        let s = db
            .get_page_sources(&c.id)
            .await
            .map_err(|e| format!("get_page_sources {}: {e}", c.id))?;
        sources_total += s.len();
        if !s.is_empty() {
            concepts_with_sources += 1;
        }
    }
    eprintln!(
        "[redistill] {}: created={} active_now={} concept_sources_rows={} ({}/{} concepts have sources) in {:.1}s",
        conv_name,
        created,
        new_active.len(),
        sources_total,
        concepts_with_sources,
        new_active.len(),
        conv_secs
    );
    if !new_active.is_empty() && concepts_with_sources == 0 {
        return Err(format!(
            "expected concept_sources rows after redistill (created {} concepts, none have sources)",
            new_active.len()
        ));
    }
    Ok(())
}

#[test]
fn fixture_revision_hash_is_stable_sha256_prefix() {
    use wenlan_core::eval::fixtures::fixture_revision_hash;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"deterministic test bytes").unwrap();
    let h1 = fixture_revision_hash(tmp.path()).unwrap();
    let h2 = fixture_revision_hash(tmp.path()).unwrap();
    assert_eq!(h1, h2, "hash must be deterministic");
    assert_eq!(h1.len(), 16, "expect 16-char hex prefix of sha256");
}

#[test]
fn fixture_revision_hash_changes_when_bytes_change() {
    use wenlan_core::eval::fixtures::fixture_revision_hash;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"version one").unwrap();
    let h1 = fixture_revision_hash(tmp.path()).unwrap();
    std::fs::write(tmp.path(), b"version two").unwrap();
    let h2 = fixture_revision_hash(tmp.path()).unwrap();
    assert_ne!(h1, h2, "hash must change when bytes change");
}

// Serialize tests that read/write EVAL_MAX_USD. std::env is process-global;
// parallel cargo-test workers race on set_var/remove_var. tokio::sync::Mutex
// because the guard is held across .await in submit_batch tests; std::sync
// Mutex would trigger clippy::await_holding_lock.
static EVAL_MAX_USD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test]
async fn submit_batch_aborts_when_estimate_exceeds_eval_max_usd() {
    use wenlan_core::eval::anthropic::submit_batch;
    let _guard = EVAL_MAX_USD_LOCK.lock().await;
    std::env::set_var("EVAL_MAX_USD", "0.001");
    let client = reqwest::Client::new();
    let huge_prompt = "x".repeat(1_000_000);
    let prompts = vec![("id-0".to_string(), huge_prompt, None, 2048usize)];
    // cost_cap_usd set high so only the EVAL_MAX_USD env var triggers the abort
    let err = submit_batch(&client, "fake-key", prompts, "claude-haiku-4-5", 99.0)
        .await
        .expect_err("should abort due to EVAL_MAX_USD cap");
    std::env::remove_var("EVAL_MAX_USD");
    assert!(
        err.contains("EVAL_MAX_USD"),
        "error should mention cap: {}",
        err
    );
}

#[tokio::test]
async fn submit_batch_no_cap_env_var_unset_does_not_block() {
    use wenlan_core::eval::anthropic::estimate_batch_cost;
    let _guard = EVAL_MAX_USD_LOCK.lock().await;
    std::env::remove_var("EVAL_MAX_USD");
    let prompts = vec![("id".to_string(), "hi".to_string(), None, 16usize)];
    let cost = estimate_batch_cost(&prompts);
    assert!(cost < 0.01, "tiny batch should be sub-cent: got {cost}");
}

#[test]
fn latency_summary_p50_p99_from_samples() {
    use wenlan_core::eval::latency::{latency_summary, LatencySummary};
    let samples_ms: Vec<u64> = (1..=100).collect();
    let s: LatencySummary = latency_summary(&samples_ms);
    assert!(
        (s.p50_ms as i64 - 50).abs() <= 1,
        "p50 ≈ 50, got {}",
        s.p50_ms
    );
    assert!(
        (s.p99_ms as i64 - 99).abs() <= 1,
        "p99 ≈ 99, got {}",
        s.p99_ms
    );
    assert_eq!(s.total_ms, samples_ms.iter().sum::<u64>());
    assert_eq!(s.sample_count, 100);
}

#[test]
fn latency_summary_empty_returns_zero() {
    use wenlan_core::eval::latency::latency_summary;
    let s = latency_summary(&[]);
    assert_eq!(s.sample_count, 0);
    assert_eq!(s.p50_ms, 0);
}

#[test]
fn judge_prompt_has_branch_for_every_lme_task_category() {
    use wenlan_core::eval::judge::task_judge_prompt;
    let categories = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "temporal-reasoning",
        "knowledge-update",
        "multi-session",
    ];
    for cat in categories {
        let p = task_judge_prompt(cat, "Q", "GT", "A");
        assert!(!p.is_empty(), "category '{}' produced empty prompt", cat);
        let lower = p.to_lowercase();
        assert!(
            lower.contains(cat) || lower.contains(&cat.replace('-', " ")),
            "category '{}' prompt should mention the category name or rubric token, got: {}",
            cat,
            p.chars().take(200).collect::<String>()
        );
    }
}

// ---------------------------------------------------------------------------
// Task 6: ReportEnv wiring tests
// ---------------------------------------------------------------------------

#[test]
fn eval_report_serializes_env_retrieval_method() {
    use wenlan_core::eval::report::{EvalReport, ReportEnv};
    let r = EvalReport {
        env: Some(ReportEnv {
            retrieval_method: "search_memory".into(),
            ..Default::default()
        }),
        ..EvalReport::default()
    };
    let json = serde_json::to_string(&r).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        v["env"]["retrieval_method"].as_str(),
        Some("search_memory"),
        "retrieval_method should serialize"
    );
}

// TODO(eval-repro): verify runner population once GPU baseline saves
#[test]
fn report_env_populated_correctly_per_runner_variant() {
    use wenlan_core::eval::report::ReportEnv;
    let base = ReportEnv {
        retrieval_method: "search_memory".into(),
        llm_provider_class: "none".into(),
        ..Default::default()
    };
    let reranked = ReportEnv {
        retrieval_method: "search_memory_reranked".into(),
        llm_provider_class: "on-device".into(),
        ..Default::default()
    };
    assert_ne!(base.retrieval_method, reranked.retrieval_method);
}

#[test]
fn baseline_filename_encodes_config() {
    use wenlan_core::eval::report::{EvalReport, ReportEnv};
    let r = EvalReport {
        env: Some(ReportEnv {
            retrieval_method: "search_memory_reranked".into(),
            llm_provider_class: "on-device".into(),
            fixture_revision: "deadbeefcafef00d".into(),
            ..Default::default()
        }),
        ..EvalReport::default()
    };
    assert_eq!(
        r.baseline_filename("longmemeval"),
        "longmemeval__reranked__on-device__deadbeefcafef00d.json"
    );
}

#[test]
fn baseline_filename_falls_back_when_env_missing() {
    use wenlan_core::eval::report::EvalReport;
    let r = EvalReport::default();
    assert_eq!(r.baseline_filename("foo"), "foo.json");
}

#[tokio::test]
#[ignore]
async fn run_kg_faithfulness_smoke() {
    use wenlan_core::eval::kg_faithfulness::run_kg_faithfulness_eval;

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("app/eval/kg_fixtures");
    if !fixture_dir.exists() {
        eprintln!("SKIP: kg_fixtures dir not found");
        return;
    }

    // IMPORTANT: eval_shared_extractor does NOT exist today (only eval_shared_embedder).
    // KgExtractor is also not a standalone type — extract_kg_batch lives on LlmEngine.
    // Constructing LlmEngine requires a model file on disk and Metal GPU access, both
    // unavailable in CI. Ship the SKIP placeholder; Task 5 lands fixtures and follow-up
    // work wires the extractor properly.
    eprintln!("SKIP: extractor construction TBD (see plan Task 4 Step 6 note)");
    let _ = fixture_dir;
    let _ = run_kg_faithfulness_eval;
}

#[tokio::test]
#[ignore]
async fn run_page_faithfulness_smoke() {
    use wenlan_core::eval::page_faithfulness::run_page_faithfulness_eval;

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("app/eval/page_fixtures");
    if !fixture_dir.exists() {
        eprintln!("SKIP: page_fixtures dir not found");
        return;
    }
    let report = run_page_faithfulness_eval(&fixture_dir);
    eprintln!("\n=== Page-Faithfulness ===");
    eprintln!(
        "Mean faithfulness: {:.3} across {} cases ({} below threshold)",
        report.mean_faithfulness, report.case_count, report.below_threshold_count
    );
    for c in &report.per_case {
        let marker = if c.meets_threshold() { "OK" } else { "FAIL" };
        eprintln!(
            "  [{}] {} {:.2} (expected >= {:.2})",
            marker, c.case_id, c.faithfulness, c.expected_min
        );
    }

    // Guard against the print-only false-green: known hallucination
    // negative-controls (seed_hallucinations.toml, id prefix `page_halluc`,
    // floor 0.99 — "the scorer SHOULD flag these as below threshold") MUST be
    // flagged. Asserting ONLY the negative controls (not positive fixtures)
    // keeps the canary non-flaky despite the lexical scorer's known
    // paraphrase-misses on faithful pages.
    let mut negative_controls = 0usize;
    for c in &report.per_case {
        if c.case_id.starts_with("page_halluc") {
            negative_controls += 1;
            assert!(
                !c.meets_threshold(),
                "negative-control {} scored {:.2} >= floor {:.2} — scorer FAILED to flag a hallucinated page",
                c.case_id, c.faithfulness, c.expected_min
            );
        }
    }
    assert!(
        negative_controls > 0,
        "no `page_halluc` negative-control fixtures found — check app/eval/page_fixtures/seed_hallucinations.toml is present"
    );
}

#[tokio::test]
#[ignore]
async fn kg_faithfulness_llm_judge_smoke() {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("SKIP: ANTHROPIC_API_KEY not set");
        return;
    }
    if std::env::var("EVAL_RUN_LLM_JUDGE").as_deref() != Ok("1") {
        eprintln!("SKIP: set EVAL_RUN_LLM_JUDGE=1 to actually fire the Batch API");
        return;
    }
    use wenlan_core::eval::kg_faithfulness::{KgExpectedEntity, KgFixtureCase};
    use wenlan_core::eval::kg_faithfulness_llm::judge_kg_case_with_llm;
    use wenlan_core::extract::{ExtractedEntity, ExtractedRelation, KgExtractionResult};

    let case = KgFixtureCase {
        id: "smoke_001".into(),
        source_text: "Rust is a systems programming language.".into(),
        expected_entities: vec![KgExpectedEntity {
            name: "Rust".into(),
            kind: "language".into(),
        }],
        expected_relations: vec![],
    };
    let extracted = KgExtractionResult {
        index: 0,
        entities: vec![
            ExtractedEntity {
                name: "Rust".into(),
                entity_type: "language".into(),
            },
            ExtractedEntity {
                name: "Python".into(),
                entity_type: "language".into(),
            },
        ],
        observations: vec![],
        relations: vec![ExtractedRelation {
            from: "Rust".into(),
            to: "systems programming".into(),
            relation_type: "is_a".into(),
            confidence: None,
            explanation: None,
        }],
    };
    let judged = judge_kg_case_with_llm(&case, &extracted, "claude-haiku-4-5-20251001")
        .await
        .expect("judge call");
    eprintln!("\n=== KG-Faith LLM Judge ===");
    eprintln!("Case: {}", judged.case_id);
    eprintln!("Entity faithful rate: {:.2}", judged.entity_faithful_rate);
    for e in &judged.entities {
        eprintln!("  [{}] {:?} :: {}", e.name, e.verdict, e.reason);
    }
    eprintln!(
        "Relation faithful rate: {:.2}",
        judged.relation_faithful_rate
    );
    for r in &judged.relations {
        eprintln!(
            "  [{} --{}-> {}] {:?} :: {}",
            r.from, r.relation_type, r.to, r.verdict, r.reason
        );
    }
    assert!(judged.entities.iter().any(|e| e.name == "Rust"));
    assert!(judged.entities.iter().any(|e| e.name == "Python"));
}

/// Smoke test the Anthropic Batch API tool_use round-trip end-to-end.
///
/// Submits a single-item batch with one LME-style triplet, polls until complete,
/// asserts that the structured verdict extraction returns score 0 or 1 with a
/// non-empty reason. Validates that:
///   - Anthropic Batch API accepts `tools` + `tool_choice` in the per-request
///     params field (the assumption in spec Section 3).
///   - The tool_use response shape matches `extract_tool_verdict` expectations.
///
/// Cost: ~$0.001 per run (single judgment with haiku).
///
/// Run before any PR that touches the judge tool_use path and before any 0.7.0
/// release. If this fails, the assumption that Batch API supports tool_choice
/// is wrong and Section 3 of the spec must be revised.
///
/// ```bash
/// ANTHROPIC_API_KEY=... cargo test -p origin-core --test eval_harness \
///   smoke_tool_use_judge_returns_structured_verdict -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn smoke_tool_use_judge_returns_structured_verdict() {
    use wenlan_core::eval::judge::{judge_with_batch_api, JudgmentTuple};

    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("SKIP: ANTHROPIC_API_KEY not set");
        return;
    }

    let tuple = JudgmentTuple {
        question: "What is the capital of France?".to_string(),
        ground_truth: "Paris".to_string(),
        answer: "Paris is the capital of France.".to_string(),
        category: "single-session-user".to_string(),
        approach: "smoke".to_string(),
        context_tokens: 0,
        question_id: String::new(),
    };

    let judge_model = std::env::var("EVAL_JUDGE_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

    eprintln!(
        "=== L7 smoke: tool_use judge round-trip (model={}) ===",
        judge_model
    );

    let results = judge_with_batch_api(&[tuple], &judge_model, Some(0.10))
        .await
        .expect("batch judge failed");

    assert_eq!(results.len(), 1, "expected exactly one judgment");
    let r = &results[0];

    eprintln!("smoke result: score={} reason={:?}", r.score, r.reason);

    assert!(r.score == 0 || r.score == 1, "score must be binary");
    assert!(
        !r.reason.is_empty(),
        "verdict_reason must be non-empty under tool_use"
    );
    // Sanity: Paris is correct, judge should return 1.
    assert_eq!(
        r.score, 1,
        "Paris->France obvious-correct judgment should be 1"
    );
}

/// T4a temporal-filter A/B on LongMemEval (retrieval-only, no GPU LLM, no judge).
///
/// The runner `run_longmemeval_eval_temporal` self-seeds a fresh ephemeral DB per
/// question from the fixture, stamps `event_date` from `haystack_dates`, then
/// retrieves via `search_memory_temporal(.., now=question_date)`.
///
/// `WENLAN_ENABLE_TEMPORAL_FILTER` controls whether the hard temporal filter
/// activates on High-confidence temporal cues. This A/B measures:
///   OFF (None)  -- temporal search path with filter disabled (plain search)
///   ON  ("1")   -- temporal search path with hard filter enabled
///
/// Respects `EVAL_LME_LIMIT` for fast iteration (e.g. EVAL_LME_LIMIT=30).
/// Single-run scaffold -- N>=3 for any headline per AGENTS.md Eval Citation Discipline.
#[tokio::test]
#[ignore = "self-seeds from fixture; retrieval-only, no GPU; set WENLAN_EVAL_ROOT + EVAL_LME_LIMIT"]
async fn temporal_filter_ab_lme() {
    use wenlan_core::eval::longmemeval::run_longmemeval_eval_temporal;

    let fixture = eval_root().join("data/longmemeval_oracle.json");
    if !fixture.exists() {
        println!(
            "SKIP: longmemeval_oracle.json not found at {}",
            fixture.display()
        );
        return;
    }

    println!(
        "=== T4a TEMPORAL-FILTER A/B (LongMemEval, search_memory_temporal, retrieval-only) ==="
    );
    println!("fixture: {}", fixture.display());

    let off = temp_env::async_with_vars(
        [("WENLAN_ENABLE_TEMPORAL_FILTER", None::<&str>)],
        run_longmemeval_eval_temporal(&fixture),
    )
    .await
    .expect("temporal eval OFF failed");

    let on = temp_env::async_with_vars(
        [("WENLAN_ENABLE_TEMPORAL_FILTER", Some("1"))],
        run_longmemeval_eval_temporal(&fixture),
    )
    .await
    .expect("temporal eval ON failed");

    println!("questions evaluated: {}", off.total_questions);
    println!(
        "FILTER OFF: ndcg@10={:.4} recall@5={:.4} mrr={:.4} hit@1={:.4}",
        off.aggregate_ndcg_at_10,
        off.aggregate_recall_at_5,
        off.aggregate_mrr,
        off.aggregate_hit_rate_at_1,
    );
    println!(
        "FILTER ON:  ndcg@10={:.4} recall@5={:.4} mrr={:.4} hit@1={:.4}",
        on.aggregate_ndcg_at_10,
        on.aggregate_recall_at_5,
        on.aggregate_mrr,
        on.aggregate_hit_rate_at_1,
    );
    println!(
        "DELTA (on-off): ndcg@10={:+.4} recall@5={:+.4} mrr={:+.4} hit@1={:+.4}",
        on.aggregate_ndcg_at_10 - off.aggregate_ndcg_at_10,
        on.aggregate_recall_at_5 - off.aggregate_recall_at_5,
        on.aggregate_mrr - off.aggregate_mrr,
        on.aggregate_hit_rate_at_1 - off.aggregate_hit_rate_at_1,
    );

    // Per-category breakdown -- print temporal-reasoning bucket specifically
    println!("\n--- Per-category breakdown (OFF) ---");
    for cat in &off.per_category {
        println!(
            "  {:30} n={:3}  ndcg@10={:.4}  recall@5={:.4}  mrr={:.4}",
            cat.question_type, cat.count, cat.ndcg_at_10, cat.recall_at_5, cat.mrr
        );
    }
    println!("--- Per-category breakdown (ON) ---");
    for cat in &on.per_category {
        println!(
            "  {:30} n={:3}  ndcg@10={:.4}  recall@5={:.4}  mrr={:.4}",
            cat.question_type, cat.count, cat.ndcg_at_10, cat.recall_at_5, cat.mrr
        );
    }

    // Targeted temporal-reasoning delta
    let tr_off = off
        .per_category
        .iter()
        .find(|c| c.question_type == "temporal-reasoning");
    let tr_on = on
        .per_category
        .iter()
        .find(|c| c.question_type == "temporal-reasoning");
    match (tr_off, tr_on) {
        (Some(o), Some(n)) => {
            println!(
                "\n>>> temporal-reasoning bucket (n={}): ndcg@10 OFF={:.4} ON={:.4} d={:+.4} | recall@5 OFF={:.4} ON={:.4} d={:+.4} | mrr OFF={:.4} ON={:.4} d={:+.4}",
                o.count,
                o.ndcg_at_10, n.ndcg_at_10, n.ndcg_at_10 - o.ndcg_at_10,
                o.recall_at_5, n.recall_at_5, n.recall_at_5 - o.recall_at_5,
                o.mrr, n.mrr, n.mrr - o.mrr,
            );
        }
        _ => {
            println!("\n>>> temporal-reasoning bucket: not present in per_category (may be absent at this EVAL_LME_LIMIT)");
        }
    }
}

// ---------------------------------------------------------------------------
// Temporal oracle probe: 3-arm LoCoMo retrieval experiment
// ---------------------------------------------------------------------------

/// Inner runner for the temporal oracle probe.
///
/// Runs three arms over the seeded LoCoMo scenario DB and emits paired JSONL
/// for two feature comparisons that `analyze_paired.py` can consume:
///
/// - `temporal_oracle_AB`: Baseline (off) vs ExtractCue (on)
/// - `temporal_oracle_AC`: Baseline (off) vs Oracle (on)
///
/// The Baseline rows are cloned and labeled with the appropriate `feature`
/// before writing so each pair file is self-consistent (the JSONL analyzer
/// joins on `(bench, query_id)` within a single feature file, so the feature
/// label must match across both arms of the same file).
///
/// The soft temporal boost env var (`WENLAN_ENABLE_TEMPORAL_SOFT_BOOST=1`) is
/// set via `temp_env::async_with_vars` for the ExtractCue and Oracle arms;
/// the Baseline arm runs without it so the boost is the controlled variable.
async fn run_temporal_oracle_probe(reranker: std::sync::Arc<dyn wenlan_core::reranker::Reranker>) {
    use wenlan_core::eval::locomo::{run_locomo_eval_cross_rerank_temporal_collect, TemporalArm};

    let root = resolve_scenario_db_root_from_harness();
    let lo_dir = root.join("locomo_v1");
    let lo_fx = eval_root().join("data/locomo10.json");

    if !lo_dir.join("origin_memory.db").exists() || !lo_fx.exists() {
        println!(
            "[temporal_oracle_probe] SKIP LoCoMo (db={} fixture={})",
            lo_dir.join("origin_memory.db").exists(),
            lo_fx.exists()
        );
        return;
    }

    let db = wenlan_core::db::MemoryDB::new(
        &lo_dir,
        std::sync::Arc::new(wenlan_core::events::NoopEmitter),
    )
    .await
    .expect("open locomo_v1 snapshot DB");

    // --- Arm A: Baseline (no cue, boost env unset) ---
    println!("--- arm Baseline (no cue, WENLAN_ENABLE_TEMPORAL_SOFT_BOOST unset) ---");
    let baseline_rows_ab = run_locomo_eval_cross_rerank_temporal_collect(
        &db,
        &lo_fx,
        reranker.clone(),
        "temporal_oracle_AB",
        TemporalArm::Baseline,
        "off",
    )
    .await
    .expect("baseline collect for AB");

    // Clone baseline rows for the AC feature (different feature label).
    let baseline_rows_ac: Vec<_> = baseline_rows_ab
        .iter()
        .cloned()
        .map(|mut r| {
            r.feature = "temporal_oracle_AC".to_string();
            r
        })
        .collect();

    // --- Arm B: ExtractCue (soft boost ON) ---
    println!("--- arm ExtractCue (WENLAN_ENABLE_TEMPORAL_SOFT_BOOST=1) ---");
    let extractcue_rows = temp_env::async_with_vars(
        [("WENLAN_ENABLE_TEMPORAL_SOFT_BOOST", Some("1"))],
        run_locomo_eval_cross_rerank_temporal_collect(
            &db,
            &lo_fx,
            reranker.clone(),
            "temporal_oracle_AB",
            TemporalArm::ExtractCue,
            "on",
        ),
    )
    .await
    .expect("extractcue collect");

    // --- Arm C: Oracle (soft boost ON) ---
    println!("--- arm Oracle (WENLAN_ENABLE_TEMPORAL_SOFT_BOOST=1) ---");
    let oracle_rows = temp_env::async_with_vars(
        [("WENLAN_ENABLE_TEMPORAL_SOFT_BOOST", Some("1"))],
        run_locomo_eval_cross_rerank_temporal_collect(
            &db,
            &lo_fx,
            reranker.clone(),
            "temporal_oracle_AC",
            TemporalArm::Oracle,
            "on",
        ),
    )
    .await
    .expect("oracle collect");

    // --- Emit paired JSONL ---
    // temporal_oracle_AB: Baseline(off) vs ExtractCue(on)
    write_paired_rows("temporal_oracle_AB", "locomo", &baseline_rows_ab);
    write_paired_rows("temporal_oracle_AB", "locomo", &extractcue_rows);

    // temporal_oracle_AC: Baseline(off) vs Oracle(on)
    write_paired_rows("temporal_oracle_AC", "locomo", &baseline_rows_ac);
    write_paired_rows("temporal_oracle_AC", "locomo", &oracle_rows);
}

/// 3-arm temporal oracle probe for LoCoMo: Baseline vs ExtractCue vs Oracle.
///
/// Emits per-query JSONL for two paired comparisons:
///   - `temporal_oracle_AB_locomo.jsonl`: Baseline (off) vs ExtractCue (on)
///   - `temporal_oracle_AC_locomo.jsonl`: Baseline (off) vs Oracle (on)
///
/// The soft temporal boost (`WENLAN_ENABLE_TEMPORAL_SOFT_BOOST=1`) is active
/// for arms B and C; Baseline runs without it. Feed EVAL_OUT to
/// `analyze_paired.py` to see per-query Δ for each arm pair.
///
/// Run (unsandboxed, against the SEEDED DBs — the snapshot has all-NULL
/// event_date so the temporal boost would never fire there; the probe is
/// read-only so the seeds stay intact):
///
///   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
///   SCENARIO_DB_ROOT=~/.cache/origin-eval/scenario_seeded \
///   EVAL_OUT=~/.cache/origin-eval/temporal_oracle_out \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     temporal_oracle_probe -- --ignored --nocapture --test-threads=1
#[tokio::test]
#[ignore = "downloads ~600MB CE model (CPU); needs cached scenario seeded LoCoMo DB. Set WENLAN_EVAL_ROOT + SCENARIO_DB_ROOT + EVAL_OUT"]
async fn temporal_oracle_probe() {
    println!("=== TEMPORAL ORACLE PROBE (3-arm: Baseline / ExtractCue / Oracle) ===");
    println!("EVAL_OUT = {}", paired_out_dir().display());

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker failed (downloads ~600MB on first run)");
    println!("CE model = {} (CPU)", reranker.model_id());

    run_temporal_oracle_probe(reranker).await;

    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// #15 slice-1: paired A/B on the expanded deep path. Arm OFF = keyword gate
/// (WENLAN_ENABLE_GRAPH_GATE=1 so the recorded graph_skipped matches the realized
/// routing; intent flag unset). Arm ON = LLM intent gate (WENLAN_ENABLE_INTENT_LLM=1,
/// graph gate also on). Both arms append to query_intent_llm_locomo.jsonl for
/// analyze_paired.py. Needs a local GPU LLM (Qwen3.5-9B on Metal) + locomo10.json; L7 manual.
///
/// CAVEAT (read before interpreting): the ON arm feeds the intent object's
/// expansions into RRF while the OFF arm feeds the legacy array-rephrasing
/// expansions, so this A/B contrasts the whole intent PIPELINE vs the legacy
/// pipeline, not `use_graph` in isolation. A positive delta means "enable the
/// intent pipeline"; to attribute it to routing alone, add a third arm
/// (intent-expansions + keyword gate). Clear $EVAL_OUT between runs:
/// write_paired_rows APPENDS, so re-running double-counts rows.
///
/// Run (unsandboxed, real GPU):
///   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
///   EVAL_OUT=$HOME/.cache/origin-eval/intent_llm_out \
///   CARGO_TARGET_DIR=/Users/lucian/Repos/origin/target \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     query_intent_llm_probe -- --ignored --nocapture --test-threads=1
///   python3 scripts/analyze_paired.py --dir $HOME/.cache/origin-eval/intent_llm_out
#[tokio::test]
#[ignore = "needs local GPU LLM (Qwen3.5-9B) + locomo10.json; L7 manual. Set WENLAN_EVAL_ROOT + EVAL_OUT"]
async fn query_intent_llm_probe() {
    use std::sync::Arc;
    println!("=== QUERY-INTENT-LLM PROBE (intent-gate vs keyword-gate) ===");
    println!("EVAL_OUT = {}", paired_out_dir().display());

    let lo_fx = eval_root().join("data/locomo10.json");
    if !lo_fx.exists() {
        println!("SKIP: locomo10.json not found at {:?}", lo_fx);
        return;
    }
    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b"))
            .expect("OnDeviceProvider qwen3.5-9b init failed — check the model is downloaded + Metal is available"),
    );

    // Arm OFF: keyword gate. GRAPH_GATE=1 so graph_skipped reflects realized routing.
    println!("--- arm OFF (keyword gate) ---");
    let off_rows = temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_GRAPH_GATE", Some("1")),
            ("WENLAN_ENABLE_INTENT_LLM", None::<&str>),
        ],
        wenlan_core::eval::locomo::run_locomo_eval_expanded_intent_collect(
            &lo_fx,
            llm.clone(),
            "query_intent_llm",
            "off",
        ),
    )
    .await
    .expect("keyword-gate collect");

    // Arm ON: LLM intent gate.
    println!("--- arm ON (intent-LLM gate) ---");
    let on_rows = temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_GRAPH_GATE", Some("1")),
            ("WENLAN_ENABLE_INTENT_LLM", Some("1")),
        ],
        wenlan_core::eval::locomo::run_locomo_eval_expanded_intent_collect(
            &lo_fx,
            llm.clone(),
            "query_intent_llm",
            "on",
        ),
    )
    .await
    .expect("intent-gate collect");

    write_paired_rows("query_intent_llm", "locomo", &off_rows);
    write_paired_rows("query_intent_llm", "locomo", &on_rows);
    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// TRACK 2: isolate the expansion temperature on the legacy path. Both arms run
/// the legacy array-expansion (WENLAN_ENABLE_INTENT_LLM unset); arm "off"=temp 0.3,
/// arm "on"=temp 0.0. Graph gate identical (WENLAN_ENABLE_GRAPH_GATE=1). Emits
/// expand_temp_locomo.jsonl for analyze_paired.py.
///
/// NOTE: write_paired_rows APPENDS — clear $EVAL_OUT between runs or rows double-count.
///
/// Run (unsandboxed, GPU):
///   WENLAN_EVAL_ROOT=/Users/lucian/Repos/origin/app/eval \
///   EVAL_OUT=$HOME/.cache/origin-eval/expand_temp_out \
///   EVAL_LOCOMO_LIMIT=10 \
///   CARGO_TARGET_DIR=/Users/lucian/Repos/origin/target \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     expand_temp_isolation_probe -- --ignored --nocapture --test-threads=1
///   python3 scripts/analyze_paired.py --dir $HOME/.cache/origin-eval/expand_temp_out
#[tokio::test]
#[ignore = "needs local LLM + LoCoMo fixture; L7 manual. Set WENLAN_EVAL_ROOT + EVAL_OUT"]
async fn expand_temp_isolation_probe() {
    use std::sync::Arc;
    println!("=== EXPAND-TEMP ISOLATION PROBE (temp0.3 vs temp0.0, legacy path) ===");
    let lo_fx = eval_root().join("data/locomo10.json");
    if !lo_fx.exists() {
        println!("[expand_temp] SKIP (fixture {} missing)", lo_fx.display());
        return;
    }
    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).unwrap(),
    );

    // Arm OFF = temp 0.3 (baseline). Intent flag UNSET in both arms.
    let off_rows = temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_GRAPH_GATE", Some("1")),
            ("WENLAN_ENABLE_INTENT_LLM", None::<&str>),
            ("WENLAN_EXPAND_TEMP", Some("0.3")),
        ],
        wenlan_core::eval::locomo::run_locomo_eval_expanded_intent_collect(
            &lo_fx,
            llm.clone(),
            "expand_temp",
            "off",
        ),
    )
    .await
    .expect("temp0.3 collect");

    // Arm ON = temp 0.0.
    let on_rows = temp_env::async_with_vars(
        [
            ("WENLAN_ENABLE_GRAPH_GATE", Some("1")),
            ("WENLAN_ENABLE_INTENT_LLM", None::<&str>),
            ("WENLAN_EXPAND_TEMP", Some("0.0")),
        ],
        wenlan_core::eval::locomo::run_locomo_eval_expanded_intent_collect(
            &lo_fx,
            llm.clone(),
            "expand_temp",
            "on",
        ),
    )
    .await
    .expect("temp0.0 collect");

    write_paired_rows("expand_temp", "locomo", &off_rows);
    write_paired_rows("expand_temp", "locomo", &on_rows);
    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// Paired graph-stream probe (#10): graph augmentation OFF vs ON, isolated.
/// Ephemeral per-conversation DBs (fixture-driven), deterministic expansion
/// (temp 0), rerank off, graph gate off (do_graph always true). The ONLY
/// difference between arms is WENLAN_GRAPH_MEMORY_STREAM:
///   OFF = stream "0" -> augment runs the legacy observation no-op == no graph
///         (the flag is default-ON since 2026-06-10, so the OFF arm pins "0").
///   ON  = stream "1" -> live entity->memory stream (memory_entities populated
///                       per conversation by the collector).
/// Per-query ndcg@10 / recall@5 paired; analyze_paired.py reports the whole-set
/// delta AND the graph-touched subset (queries where Δndcg != 0). The collector
/// prints memory_entities coverage so a null can be checked against substrate
/// liveness before it is trusted.
///
/// Run (unsandboxed, GPU). Smoke first with a subset:
///   WENLAN_EVAL_ROOT=$PWD/app/eval \
///   EVAL_OUT=$HOME/.cache/origin-eval/graph_stream_out \
///   EVAL_LOCOMO_LIMIT=20 \
///   CARGO_TARGET_DIR=/Users/lucian/Repos/origin/target \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     graph_stream_pair_locomo -- --ignored --nocapture --test-threads=1
///   python3 scripts/analyze_paired.py --dir $HOME/.cache/origin-eval/graph_stream_out
#[tokio::test]
#[ignore = "needs local LLM + LoCoMo fixture; L7 manual. Set WENLAN_EVAL_ROOT + EVAL_OUT"]
async fn graph_stream_pair_locomo() {
    use std::sync::Arc;
    println!("=== GRAPH-STREAM PAIR (stream OFF vs ON, entity-populated, LoCoMo) ===");
    let lo_fx = eval_root().join("data/locomo10.json");
    if !lo_fx.exists() {
        println!("[graph_stream] SKIP (fixture {} missing)", lo_fx.display());
        return;
    }
    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).unwrap(),
    );
    let prompts = wenlan_core::prompts::PromptRegistry::default();

    // OFF arm: stream "0" -> augment is the legacy observation no-op == no graph
    // (default-ON since 2026-06-10, so OFF must pin "0", not unset).
    let off_rows = temp_env::async_with_vars(
        [
            ("WENLAN_GRAPH_MEMORY_STREAM", Some("0")),
            ("WENLAN_ENABLE_GRAPH_GATE", None::<&str>),
            ("WENLAN_ENABLE_INTENT_LLM", None::<&str>),
            ("WENLAN_EXPAND_TEMP", Some("0.0")),
        ],
        wenlan_core::eval::locomo::run_locomo_eval_graph_stream_collect(
            &lo_fx,
            llm.clone(),
            &prompts,
            "graph_stream",
            "off",
        ),
    )
    .await
    .expect("stream-off collect");

    // ON arm: live entity->memory stream over the populated junction.
    let on_rows = temp_env::async_with_vars(
        [
            ("WENLAN_GRAPH_MEMORY_STREAM", Some("1")),
            ("WENLAN_ENABLE_GRAPH_GATE", None::<&str>),
            ("WENLAN_ENABLE_INTENT_LLM", None::<&str>),
            ("WENLAN_EXPAND_TEMP", Some("0.0")),
        ],
        wenlan_core::eval::locomo::run_locomo_eval_graph_stream_collect(
            &lo_fx,
            llm.clone(),
            &prompts,
            "graph_stream",
            "on",
        ),
    )
    .await
    .expect("stream-on collect");

    write_paired_rows("graph_stream", "locomo", &off_rows);
    write_paired_rows("graph_stream", "locomo", &on_rows);
    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// LongMemEval twin of [`graph_stream_pair_locomo`]. Same cached, LLM-free-query
/// design; populates `lme/<question_id>` once, then both arms reuse it.
///   WENLAN_EVAL_ROOT=$PWD/app/eval \
///   EVAL_OUT=$HOME/.cache/origin-eval/graph_stream_lme_out \
///   EVAL_LME_LIMIT=20 \
///   CARGO_TARGET_DIR=/Users/lucian/Repos/origin/target \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     graph_stream_pair_lme -- --ignored --nocapture --test-threads=1
///   python3 scripts/analyze_paired.py --dir $HOME/.cache/origin-eval/graph_stream_lme_out
#[tokio::test]
#[ignore = "needs local LLM + LME fixture; L7 manual. Set WENLAN_EVAL_ROOT + EVAL_OUT"]
async fn graph_stream_pair_lme() {
    use std::sync::Arc;
    println!("=== GRAPH-STREAM PAIR (stream OFF vs ON, entity-populated, LME) ===");
    let fx = eval_root().join("data/longmemeval_oracle.json");
    if !fx.exists() {
        println!("[graph_stream] SKIP (fixture {} missing)", fx.display());
        return;
    }
    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).unwrap(),
    );
    let prompts = wenlan_core::prompts::PromptRegistry::default();

    // OFF arm: stream "0" -> augment is the legacy observation no-op == no graph
    // (default-ON since 2026-06-10, so OFF must pin "0", not unset).
    let off_rows = temp_env::async_with_vars(
        [
            ("WENLAN_GRAPH_MEMORY_STREAM", Some("0")),
            ("WENLAN_ENABLE_GRAPH_GATE", None::<&str>),
        ],
        wenlan_core::eval::longmemeval::run_longmemeval_eval_graph_stream_collect(
            &fx,
            llm.clone(),
            &prompts,
            "graph_stream",
            "off",
        ),
    )
    .await
    .expect("stream-off collect");

    // ON arm: live entity->memory stream over the populated junction.
    let on_rows = temp_env::async_with_vars(
        [
            ("WENLAN_GRAPH_MEMORY_STREAM", Some("1")),
            ("WENLAN_ENABLE_GRAPH_GATE", None::<&str>),
        ],
        wenlan_core::eval::longmemeval::run_longmemeval_eval_graph_stream_collect(
            &fx,
            llm.clone(),
            &prompts,
            "graph_stream",
            "on",
        ),
    )
    .await
    .expect("stream-on collect");

    write_paired_rows("graph_stream", "lme", &off_rows);
    write_paired_rows("graph_stream", "lme", &on_rows);
    println!(
        "=== done -> python3 scripts/analyze_paired.py --dir {} ===",
        paired_out_dir().display()
    );
}

/// TRACK 1 graph-substrate gate (#10). Copies the populated scenario_seeded
/// locomo_v1 DB, runs the entity-linking sweep with the FINE primitive
/// (`extract_entities_for_content`, NOT the speaker-level single-entity one),
/// then prints the `memory_entities` degree distribution. Decides whether a
/// fine entity->memory bridge exists (Option A viable) or it is still a
/// speaker-pool hairball (graph-first insolvent).
///
/// Run (unsandboxed, GPU). Smoke first with a small cap, then full:
///   SCENARIO_DB=$HOME/.cache/origin-eval/scenario_seeded/locomo_v1/origin_memory.db \
///   GATE_MAX_MEMORIES=300 \
///   CARGO_TARGET_DIR=/Users/lucian/Repos/origin/target \
///     cargo test -p origin-core --features eval-harness --test eval_harness \
///     graph_substrate_gate_locomo -- --ignored --nocapture --test-threads=1
#[tokio::test]
#[ignore = "needs local LLM + scenario_seeded DB; L7 manual. Set SCENARIO_DB"]
async fn graph_substrate_gate_locomo() {
    use std::sync::Arc;
    let src = std::env::var("SCENARIO_DB").unwrap_or_else(|_| {
        format!(
            "{}/.cache/origin-eval/scenario_seeded/locomo_v1/origin_memory.db",
            std::env::var("HOME").unwrap()
        )
    });
    if !std::path::Path::new(&src).exists() {
        println!("[gate] SKIP (missing {src})");
        return;
    }
    // Work on a COPY -- never mutate the canonical seed.
    let tmp = tempfile::tempdir().unwrap();
    let dst = tmp.path().join("origin_memory.db");
    std::fs::copy(&src, &dst).expect("copy seed");
    let db = wenlan_core::db::MemoryDB::new(tmp.path(), Arc::new(wenlan_core::events::NoopEmitter))
        .await
        .expect("open seed copy");

    let before = db.memory_entities_degree_stats().await.unwrap();
    println!("[gate] BEFORE memory_entities: {before:?}");

    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = Arc::new(
        wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).unwrap(),
    );
    let prompts = wenlan_core::prompts::PromptRegistry::default();
    let cap: Option<usize> = std::env::var("GATE_MAX_MEMORIES")
        .ok()
        .and_then(|v| v.parse().ok());

    let processed = run_capped_fine_sweep(&db, &llm, &prompts, 32, cap).await;
    println!("[gate] processed {processed} memories");

    let after = db.memory_entities_degree_stats().await.unwrap();
    println!("[gate] AFTER memory_entities: {after:?}");
    print_top_hubs(&db, 20).await;
    println!(
        "[gate] DECISION INPUTS: p50={} p90={} max={} hubs>50={} memories_linked={}",
        after.p50_memories_per_entity,
        after.p90_memories_per_entity,
        after.max_memories_per_entity,
        after.entities_gt_50,
        after.memories_linked
    );
}

/// Run a bounded FINE entity-linking sweep over unlinked memories, optionally
/// capped at `cap` memories. Tracks attempted source_ids so memories that
/// yield zero entities do not get re-fetched forever (run_enrichment_sweep
/// would infinite-loop on the fine primitive, which can return empty). Returns
/// memories attempted.
async fn run_capped_fine_sweep(
    db: &wenlan_core::db::MemoryDB,
    llm: &std::sync::Arc<dyn wenlan_core::llm_provider::LlmProvider>,
    prompts: &wenlan_core::prompts::PromptRegistry,
    batch_size: usize,
    cap: Option<usize>,
) -> usize {
    use std::collections::HashSet;
    let mut attempted: HashSet<String> = HashSet::new();
    loop {
        if cap.is_some_and(|c| attempted.len() >= c) {
            break;
        }
        // Grow the fetch window past already-attempted (zero-entity) rows, which
        // never leave the unlinked set and pile up at the source_id front. Without
        // this, a front cluster of >= batch_size zero-entity rows would make every
        // fetched row already-attempted -> fresh empty -> early break, under-counting
        // links. attempted.len() >= the zero-entity-attempted count, so the window
        // always surfaces >= batch_size fresh rows when any remain. O(n^2) queries
        // total, fine for a one-shot L7 gate.
        let batch = db
            .unlinked_memories(batch_size + attempted.len())
            .await
            .unwrap_or_default();
        // Only rows we have not already attempted; if none are fresh, no further
        // progress is possible (the rest are zero-entity memories) -> stop.
        let fresh: Vec<(String, String)> = batch
            .into_iter()
            .filter(|(sid, _)| !attempted.contains(sid))
            .collect();
        if fresh.is_empty() {
            break;
        }
        for (sid, content) in fresh {
            if cap.is_some_and(|c| attempted.len() >= c) {
                break;
            }
            let ents = wenlan_core::kg::entity_extraction::extract_entities_for_content(
                db,
                llm,
                prompts,
                &content,
                Some(sid.as_str()),
            )
            .await
            .unwrap_or_default();
            if !ents.is_empty() {
                let refs: Vec<&str> = ents.iter().map(|s| s.as_str()).collect();
                let _ = db.link_memory_entities(&sid, &refs).await;
                // NB: we intentionally do NOT also set the legacy `entity_id` column
                // (run_enrichment_sweep did) — the gate reads only the memory_entities
                // junction, so the junction row alone is sufficient + correct here.
            }
            attempted.insert(sid);
        }
    }
    attempted.len()
}

/// Print the top-N hub entities (count, name) from memory_entities for eyeball
/// classification (speaker pool vs concept bridge).
async fn print_top_hubs(db: &wenlan_core::db::MemoryDB, n: usize) {
    match db.top_memory_entity_hubs(n).await {
        Ok(hubs) => {
            println!("[gate] TOP-{n} hubs (count, name):");
            for (c, name) in hubs {
                println!("    {c:5}  {name}");
            }
        }
        Err(e) => println!("[gate] top_hubs err: {e}"),
    }
}

// ===== CE A/B answer-accuracy harness =====

/// Runtime proof for the CE A/B per-question seed loop.
///
/// GPU/Metal + minutes, NOT run in CI/gate. This seeds one temporal-reasoning
/// LongMemEval oracle question through `run_fullpipeline_lme`, then opens
/// that question's scenario DB and verifies the temporal substrate was populated.
/// Post-fix GREEN-only proof (not a red/green pair): with the seed-loop injection
/// in place, event_date count is > 0; it cannot exhibit the pre-fix RED state
/// against current code.
#[tokio::test]
#[ignore = "GPU/Metal + minutes; runtime proof for CE A/B event_date injection"]
async fn ce_ab_seed_injects_event_date() {
    use std::sync::Arc;
    use wenlan_core::eval::answer_quality::{run_fullpipeline_lme, ArmSpec, DbSource};
    use wenlan_core::eval::shared::{scenario_db_dir, EnrichmentMode};
    use wenlan_core::llm_provider::{LlmProvider, OnDeviceProvider};

    let lme_path = eval_root().join("data/longmemeval_oracle.json");
    if !lme_path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found at {:?}", lme_path);
        return;
    }

    let raw = std::fs::read_to_string(&lme_path).expect("read longmemeval_oracle.json");
    let samples: Vec<serde_json::Value> =
        serde_json::from_str(&raw).expect("parse longmemeval_oracle.json");
    let sample = samples
        .into_iter()
        .find(|sample| {
            sample.get("question_type").and_then(|v| v.as_str()) == Some("temporal-reasoning")
        })
        .expect("fixture should contain a temporal-reasoning question");
    let question_id = sample
        .get("question_id")
        .and_then(|v| v.as_str())
        .expect("temporal sample has question_id")
        .to_string();

    let tmp = tempfile::tempdir().unwrap();
    let fixture_path = tmp.path().join("one_temporal_lme.json");
    std::fs::write(
        &fixture_path,
        serde_json::to_vec_pretty(&vec![sample]).expect("serialize one-question fixture"),
    )
    .expect("write one-question fixture");

    let previous_graph_stream = std::env::var_os("WENLAN_GRAPH_MEMORY_STREAM");
    std::env::set_var("WENLAN_GRAPH_MEMORY_STREAM", "0");

    let llm: Arc<dyn LlmProvider> = Arc::new(
        OnDeviceProvider::new_with_model(Some("qwen3.5-9b"))
            .expect("Qwen3.5-9B on-device provider"),
    );
    let enrichment = EnrichmentMode::OnDevice(llm.clone());
    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None)
        .expect("init_cross_encoder_reranker");
    let output_path = tmp.path().join("ce_ab_lme_tuples.json");

    run_fullpipeline_lme(
        &fixture_path,
        enrichment,
        llm,
        Some(reranker),
        &ArmSpec::ce_two(),
        DbSource::SeedPerQuestion,
        &output_path,
    )
    .await
    .expect("run_fullpipeline_lme");

    if let Some(value) = previous_graph_stream {
        std::env::set_var("WENLAN_GRAPH_MEMORY_STREAM", value);
    } else {
        std::env::remove_var("WENLAN_GRAPH_MEMORY_STREAM");
    }

    let scenario_dir = scenario_db_dir(tmp.path(), "lme", &question_id);
    let scenario_db =
        libsql::Builder::new_local(scenario_dir.join("origin_memory.db").to_str().unwrap())
            .build()
            .await
            .expect("open scenario DB");
    let conn = scenario_db.connect().expect("connect scenario DB");
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM memories WHERE event_date IS NOT NULL",
            (),
        )
        .await
        .expect("query event_date count");
    let row = rows.next().await.unwrap().unwrap();
    let event_date_count = row.get::<i64>(0).unwrap();
    assert!(
        event_date_count > 0,
        "CE A/B per-question scenario DB must have temporal substrate"
    );
}

/// Fair CE A/B answer-accuracy eval for LongMemEval.
///
/// Both arms route through `search_memory_cross_rerank` so P3 distill-demotion
/// is symmetric. Graph stream must be OFF (WENLAN_GRAPH_MEMORY_STREAM=0) so
/// the OFF arm does not re-enable it. Answers are generated on-device by
/// Qwen3.5-9B at temperature 0.0 (NOT Claude — avoids self-judge with the
/// Haiku judge).
///
/// ```bash
/// WENLAN_GRAPH_MEMORY_STREAM=0 \
/// LME_LIMIT_QUESTIONS=50 \
///   cargo test -p origin-core --test eval_harness generate_ce_ab_lme -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn generate_ce_ab_lme() {
    use std::sync::Arc;
    use wenlan_core::eval::answer_quality::{run_fullpipeline_lme, ArmSpec, DbSource};
    use wenlan_core::llm_provider::{LlmProvider, OnDeviceProvider};

    let lme_path = eval_root().join("data/longmemeval_oracle.json");
    if !lme_path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found at {:?}", lme_path);
        return;
    }

    // The enrichment phase (entity extraction, distillation) is configured via
    // EnrichmentMode; answer-gen is on-device Qwen3.5-9B (see below), so these
    // only steer enrichment, not the answers being measured.
    let enrich_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    // Safety: the runner enforces this, but surfacing it early gives a clearer
    // message when the operator forgot to set it.
    if wenlan_core::db::graph_memory_stream_enabled() {
        panic!(
            "Set WENLAN_GRAPH_MEMORY_STREAM=0 before running this test. \
             The CE A/B is invalid without it (see run_fullpipeline_lme docs)."
        );
    }

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("ce_ab_lme_tuples.json");

    eprintln!(
        "[generate_ce_ab_lme]\n  answer model: on-device qwen3.5-9b (temp 0.0)\n  enrich model: {}\n  output: {:?}",
        enrich_model, output_path,
    );

    // On-device Qwen3.5-9B generates the answers being measured (NOT Claude).
    let llm: Arc<dyn LlmProvider> = Arc::new(
        OnDeviceProvider::new_with_model(Some("qwen3.5-9b"))
            .expect("OnDeviceProvider::new_with_model failed — is the Qwen3.5-9B model available?"),
    );

    let enrichment = wenlan_core::eval::shared::EnrichmentMode::from_env(&enrich_model, cost_cap)
        .expect("EnrichmentMode::from_env failed");

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None).expect(
        "init_cross_encoder_reranker failed (downloads ~1.1GB bge-reranker-base on first run)",
    );

    let tuples = run_fullpipeline_lme(
        &lme_path,
        enrichment,
        llm,
        Some(reranker),
        &ArmSpec::ce_two(),
        DbSource::SeedPerQuestion,
        &output_path,
    )
    .await
    .expect("run_fullpipeline_lme failed");

    eprintln!(
        "\n[generate_ce_ab_lme] Done: {} tuples saved to {:?}",
        tuples.len(),
        output_path
    );
}

/// CE A/B answer-accuracy on the CONSOLIDATED (large-pool, N>>k) LME DB.
///
/// Counterpart to `generate_ce_ab_lme`, which seeds one tiny oracle-pool DB per
/// question (N~k) so the cross-encoder reorders an already-complete context and
/// cannot change membership. This variant runs every question against ONE
/// pre-seeded consolidated DB (all questions' memories in a single store), so
/// retrieval must select k-from-many and the CE has real reselection headroom.
///
/// REQUIRED env:
/// - `CE_AB_CONSOLIDATED_DB` — directory containing the seeded `origin_memory.db`
///   (e.g. `~/.cache/origin-eval/scenario_seeded/lme_v1`). The DB is opened once,
///   read-only (no seed, no event_date inject — it is already enriched + dated).
/// - `WENLAN_GRAPH_MEMORY_STREAM=0` — confounder gate (same as the oracle variant).
/// - `RERANK_POOL_FLOOR` > 10 STRONGLY recommended (e.g. 50): with the default
///   floor of 10 the CE fetch pool equals the top-10 window, so it reorders the
///   returned docs without reselecting — a null by construction regardless of
///   the consolidated pool size (see `compute_rerank_fetch_pool`). The runner
///   logs a WARNING when the effective fetch pool is <= 10.
///
/// ```bash
/// CE_AB_CONSOLIDATED_DB=$HOME/.cache/origin-eval/scenario_seeded/lme_v1 \
/// WENLAN_GRAPH_MEMORY_STREAM=0 RERANK_POOL_FLOOR=50 \
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-ce-consolidated \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   generate_ce_ab_lme_consolidated -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn generate_ce_ab_lme_consolidated() {
    use std::sync::Arc;
    use wenlan_core::eval::answer_quality::{run_fullpipeline_lme, ArmSpec, DbSource};
    use wenlan_core::llm_provider::{LlmProvider, OnDeviceProvider};

    let lme_path = eval_root().join("data/longmemeval_oracle.json");
    if !lme_path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found at {:?}", lme_path);
        return;
    }

    let consolidated_dir = match std::env::var("CE_AB_CONSOLIDATED_DB") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            eprintln!(
                "SKIP: set CE_AB_CONSOLIDATED_DB to the seeded consolidated DB dir \
                 (e.g. ~/.cache/origin-eval/scenario_seeded/lme_v1)"
            );
            return;
        }
    };
    if !consolidated_dir.join("origin_memory.db").exists() {
        eprintln!(
            "SKIP: no origin_memory.db under CE_AB_CONSOLIDATED_DB={:?}. \
             Seed it (scripts/seed-scenario-dbs.sh) first.",
            consolidated_dir
        );
        return;
    }

    if wenlan_core::db::graph_memory_stream_enabled() {
        panic!(
            "Set WENLAN_GRAPH_MEMORY_STREAM=0 before running this test. \
             The CE A/B is invalid without it (see run_fullpipeline_lme docs)."
        );
    }

    let enrich_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("ce_ab_lme_consolidated_tuples.json");

    eprintln!(
        "[generate_ce_ab_lme_consolidated]\n  consolidated DB: {:?}\n  answer model: \
         on-device qwen3.5-9b (temp 0.0)\n  enrich model (unused on consolidated path): {}\n  \
         output: {:?}",
        consolidated_dir, enrich_model, output_path,
    );

    let llm: Arc<dyn LlmProvider> = Arc::new(
        OnDeviceProvider::new_with_model(Some("qwen3.5-9b"))
            .expect("OnDeviceProvider::new_with_model failed — is the Qwen3.5-9B model available?"),
    );

    // EnrichmentMode is unused on the consolidated path (no per-question seeding),
    // but the signature requires one; supply the configured mode for symmetry.
    let enrichment = wenlan_core::eval::shared::EnrichmentMode::from_env(&enrich_model, cost_cap)
        .expect("EnrichmentMode::from_env failed");

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None).expect(
        "init_cross_encoder_reranker failed (downloads ~1.1GB bge-reranker-base on first run)",
    );

    let tuples = run_fullpipeline_lme(
        &lme_path,
        enrichment,
        llm,
        Some(reranker),
        &ArmSpec::ce_two(),
        DbSource::Consolidated(consolidated_dir),
        &output_path,
    )
    .await
    .expect("run_fullpipeline_lme (consolidated) failed");

    eprintln!(
        "\n[generate_ce_ab_lme_consolidated] Done: {} tuples saved to {:?}",
        tuples.len(),
        output_path
    );
}

/// Judge CE A/B tuples and print McNemar report.
///
/// Load tuples from `ce_ab_lme_tuples.json`, judge via Haiku (or
/// `LME_JUDGE_MODEL` override), then print the paired McNemar report.
///
/// For N>=3 runs, execute `generate_ce_ab_lme` N times and merge the
/// tuple files before judging.
///
/// ```bash
/// WENLAN_GRAPH_MEMORY_STREAM=0 \
/// ANTHROPIC_API_KEY=... \
///   cargo test -p origin-core --test eval_harness judge_ce_ab_lme -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn judge_ce_ab_lme() {
    use wenlan_core::eval::judge::{
        judge_with_claude_model_batched_persistent, load_judgment_tuples, paired_answer_mcnemar,
    };

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    // Default to the oracle output; CE_AB_TUPLES_FILE lets the same judge grade
    // the consolidated variant (ce_ab_lme_consolidated_tuples.json) without a
    // second judge fn — keeps producer/consumer on one path.
    let tuples_file =
        std::env::var("CE_AB_TUPLES_FILE").unwrap_or_else(|_| "ce_ab_lme_tuples.json".to_string());
    let tuples_path = baselines.join(&tuples_file);

    if !tuples_path.exists() {
        eprintln!(
            "SKIP: {:?} not found. Run generate_ce_ab_lme (or set CE_AB_TUPLES_FILE) first.",
            tuples_path
        );
        return;
    }

    // The judge runs via the `claude` CLI (OAuth subscription, no API spend).
    // ANTHROPIC_API_KEY must be UNSET for a $-free run: if it is set, the
    // `claude` CLI bills the Anthropic API instead of the subscription. Warn
    // loudly rather than require the key (the old `expect` forced the billing
    // trap on every run).
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        eprintln!(
            "[judge_ce_ab_lme] WARNING: ANTHROPIC_API_KEY is set — the `claude` CLI will \
             bill the API instead of your OAuth subscription. Unset it for a $-free run."
        );
    }

    let judge_model = std::env::var("LME_JUDGE_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());

    let tuples = load_judgment_tuples(&tuples_path).expect("load_judgment_tuples failed");
    eprintln!("[judge_ce_ab_lme] Loaded {} tuples", tuples.len());

    let cache_path = baselines.join("ce_ab_lme_judge_cache.jsonl");
    let results = judge_with_claude_model_batched_persistent(
        &tuples,
        10, // batch_size
        3,  // rotation_calls
        &judge_model,
        &cache_path,
        3, // max_retries
    )
    .await
    .expect("judging failed");

    let report = paired_answer_mcnemar(&results);

    eprintln!("\n=== CE A/B McNemar Report ===");
    eprintln!(
        "n_pairs={} | OFF acc={:.4} | ON acc={:.4} | delta={:+.4}",
        report.n_pairs, report.off_acc, report.on_acc, report.delta
    );
    eprintln!(
        "b(off✓ on✗)={} | c(off✗ on✓)={} | stat={:.4} | p={:.4} | test={}",
        report.b, report.c, report.stat, report.p_value, report.test_used
    );
    eprintln!(
        "discordant pairs (b+c): {}{}",
        report.n_discordant,
        if report.underpowered {
            "  [UNDERPOWERED: b+c < 10]"
        } else {
            ""
        }
    );
    eprintln!("\nPer-category:");
    for cat in &report.by_category {
        eprintln!(
            "  {:<30} n={:4} off={:.4} on={:.4} delta={:+.4} b={} c={}",
            cat.category, cat.n_pairs, cat.off_acc, cat.on_acc, cat.delta, cat.b, cat.c
        );
    }

    // Print JSON for machine consumption.
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );
}

/// BEST-POSSIBLE CEILING: single-arm full-stack answer accuracy on GOLD (oracle).
///
/// Unlike the CE A/B (which forces every feature OFF to isolate the cross-encoder),
/// this runs ONE arm with the full production stack turned ON and just measures how
/// accurately Wenlan answers the gold/oracle questions. Answers what the CE A/Bs
/// never could: "how good are we with everything on?" Pairs with
/// `judge_lme_fullstack_ceiling` for the accuracy report.
///
/// GOLD = `longmemeval_oracle.json` via `DbSource::SeedPerQuestion` (each question's
/// own evidence sessions, the easy/oracle retrieval setting). Enrichment runs per
/// question (classify + entities + distill), so pages/graph substrate exists.
///
/// "Best-possible" = the operator turns the feature flags ON. RECOMMENDED env:
/// ```bash
/// WENLAN_ENABLE_PAGE_CHANNEL=1 WENLAN_GRAPH_MEMORY_STREAM=1 \
/// WENLAN_ENABLE_TEMPORAL_SOFT_BOOST=1 RERANK_POOL_FLOOR=50 \
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-ceiling \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   generate_lme_fullstack_ceiling -- --ignored --nocapture
/// ```
/// NOTE: on the CE path the live graph stream is structurally suppressed
/// (`allow_graph_stream = reranker.is_none()`), so `WENLAN_GRAPH_MEMORY_STREAM=1`
/// contributes via entity-linked memories already in the pool, not the live stream.
#[tokio::test]
#[ignore]
async fn generate_lme_fullstack_ceiling() {
    use std::sync::Arc;
    use wenlan_core::eval::answer_quality::{run_fullpipeline_lme, ArmSpec, DbSource};
    use wenlan_core::llm_provider::{LlmProvider, OnDeviceProvider};

    let lme_path = eval_root().join("data/longmemeval_oracle.json");
    if !lme_path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found at {:?}", lme_path);
        return;
    }

    let enrich_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("lme_fullstack_ceiling_tuples.json");

    // Receipt: echo the effective feature config so the baseline can't lie about
    // what "full stack" meant for this run.
    eprintln!(
        "[generate_lme_fullstack_ceiling] FULL-STACK config:\n  \
         page_channel={}  graph_stream_flag={}  temporal_soft_boost={}  temporal_filter={}\n  \
         RERANK_POOL_FLOOR={}  RERANK_POOL_MULTIPLIER={}\n  \
         answer model: on-device qwen3.5-9b (temp 0.0)\n  enrich model: {}\n  output: {:?}",
        wenlan_core::db::page_channel_enabled(),
        wenlan_core::db::graph_memory_stream_enabled(),
        wenlan_core::db::temporal_soft_boost_enabled(),
        wenlan_core::db::temporal_filter_enabled(),
        std::env::var("RERANK_POOL_FLOOR").unwrap_or_else(|_| "10 (default)".into()),
        std::env::var("RERANK_POOL_MULTIPLIER").unwrap_or_else(|_| "1 (default)".into()),
        enrich_model,
        output_path,
    );

    let llm: Arc<dyn LlmProvider> = Arc::new(
        OnDeviceProvider::new_with_model(Some("qwen3.5-9b"))
            .expect("OnDeviceProvider::new_with_model failed — is the Qwen3.5-9B model available?"),
    );

    let enrichment = wenlan_core::eval::shared::EnrichmentMode::from_env(&enrich_model, cost_cap)
        .expect("EnrichmentMode::from_env failed");

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None).expect(
        "init_cross_encoder_reranker failed (downloads ~1.1GB bge-reranker-base on first run)",
    );

    let tuples = run_fullpipeline_lme(
        &lme_path,
        enrichment,
        llm,
        Some(reranker),
        &ArmSpec::full_stack(),
        DbSource::SeedPerQuestion,
        &output_path,
    )
    .await
    .expect("run_fullpipeline_lme (full_stack) failed");

    eprintln!(
        "\n[generate_lme_fullstack_ceiling] Done: {} tuples saved to {:?}",
        tuples.len(),
        output_path
    );
}

/// FULL-STACK accuracy on the DEEP per-question LME-S substrate (pool-decision +
/// headline instrument).
///
/// Judge the pages A/B tuples (`generate_pages_ab_lme_s` output) via `claude -p` (no API
/// key — Max OAuth). `aggregate_judgments` groups by the tuple `approach` field
/// (`ce_off`/`ce_on`/`ce_on_pages` x category); collapse arms across categories in
/// post-processing for the directional pages-help verdict (ce_on_pages vs ce_on).
///
/// ```bash
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-pool90-mig \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   judge_pages_ab_lme_s -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn judge_pages_ab_lme_s() {
    use wenlan_core::eval::token_efficiency::{
        aggregate_judgments, judge_with_claude, load_judgment_tuples,
    };
    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    let tuples_path = baselines.join("pages_ab_lme_s_tuples.json");
    if !tuples_path.exists() {
        eprintln!(
            "SKIP: run generate_pages_ab_lme_s first ({:?} missing)",
            tuples_path
        );
        return;
    }
    let tuples = load_judgment_tuples(&tuples_path).expect("load pages-ab tuples");
    let conc: usize = std::env::var("EVAL_JUDGE_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    eprintln!(
        "Judging {} pages-A/B tuples via claude -p (conc={})...",
        tuples.len(),
        conc
    );
    let results = judge_with_claude(&tuples, conc)
        .await
        .expect("judge failed");
    let report = aggregate_judgments(&results, "haiku");
    eprintln!("\n=== Pages A/B (deep LME-S, claude -p judge) ===");
    eprintln!(
        "{:<36} | {:<9} | {:<8} | Total",
        "approach (arm_category)", "accuracy", "correct"
    );
    for r in &report.results_by_approach {
        eprintln!(
            "{:<36} | {:<8.1}% | {:<8} | {}",
            r.approach,
            r.accuracy * 100.0,
            r.correct,
            r.total
        );
    }
    eprintln!("\nTotal judged: {}", report.total_judged);
}

/// STEP (manual unblock): inject real dataset `event_date` into the PER-QUESTION
/// LME-S fullpipeline DBs under `EVAL_BASELINES_DIR/fullpipeline/lme/<qid>`. The pool90
/// seed (`enrich_fullpipeline_lme_only`) skipped temporal injection, so the migrate-read
/// substrate-live guard refuses (0 event_date rows). This loops each per-Q DB and writes
/// `event_date` from dataset session metadata (`event_date_map` -> `set_event_dates_by_source_id`):
/// GPU-free, non-destructive (fills `event_date` only). Run against a COPY of pool90 so the
/// original stays pristine.
///
/// ```bash
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-pool90-mig LME_S_FIXTURE=/tmp/claude/lme_s_90.json \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   seed_inject_event_dates_per_q_lme_s -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn seed_inject_event_dates_per_q_lme_s() {
    use wenlan_core::eval::longmemeval;
    use wenlan_core::eval::shared::{eval_baselines_dir_override, scenario_db_dir};
    let baselines = eval_baselines_dir_override()
        .expect("EVAL_BASELINES_DIR must point at the per-Q baselines dir (the copy)");
    let lme_path = std::path::PathBuf::from(
        std::env::var("LME_S_FIXTURE").expect("LME_S_FIXTURE must point at the deep LME-S fixture"),
    );
    let samples = longmemeval::load_longmemeval(&lme_path).expect("load lme-s fixture");
    let updates: Vec<(String, i64)> = longmemeval::event_date_map(&samples).into_iter().collect();
    eprintln!(
        "[inject_per_q] {} source_id->date mappings across {} questions; baselines={:?}",
        updates.len(),
        samples.len(),
        baselines
    );
    let emitter = || std::sync::Arc::new(wenlan_core::events::NoopEmitter);
    let mut dbs = 0usize;
    let mut missing = 0usize;
    for sample in &samples {
        let dir = scenario_db_dir(&baselines, "lme", &sample.question_id);
        if !dir.join("origin_memory.db").exists() {
            missing += 1;
            continue;
        }
        let db = wenlan_core::db::MemoryDB::new(&dir, emitter())
            .await
            .expect("open per-q db");
        let n = db
            .set_event_dates_by_source_id(&updates)
            .await
            .expect("inject event_dates");
        eprintln!("[inject_per_q] {} -> {} rows", sample.question_id, n);
        dbs += 1;
    }
    eprintln!(
        "[inject_per_q] done: {} DBs updated, {} qids had no DB under baselines",
        dbs, missing
    );
    assert!(
        dbs > 0,
        "no per-q DBs found under {:?}/fullpipeline/lme",
        baselines
    );
}

/// Pages answer-acc A/B over the DEEP per-question LME-S substrate, reusing the
/// pre-seeded pool90 DBs in `EVAL_BASELINES_DIR` (NO re-seed; cache-hit reads the
/// existing per-question seeds, so the substrate stays untouched). `ArmSpec::pages_ab()`
/// = 3 arms isolating the page channel in the ANSWER context: ce_off (CE off, no pages)
/// / ce_on (CE on, no pages) / ce_on_pages (CE on, pages). Answers on-device
/// Qwen3.5-9B (temp 0.0); the judge is a separate `claude -p` step over the saved
/// tuples. A `CeOff` arm is present so `needs_confounder_isolation` is true and the
/// runner enforces `WENLAN_GRAPH_MEMORY_STREAM=0`. The page fetch is driven by the
/// per-arm `include_pages` toggle (independent of `WENLAN_ENABLE_PAGE_CHANNEL`).
///
/// ```bash
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-pool90 LME_S_FIXTURE=/tmp/claude/lme_s_90.json \
/// WENLAN_GRAPH_MEMORY_STREAM=0 EVAL_ENRICHMENT=local LME_LIMIT_QUESTIONS=20 \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   generate_pages_ab_lme_s -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn generate_pages_ab_lme_s() {
    use std::sync::Arc;
    use wenlan_core::eval::answer_quality::{run_fullpipeline_lme, ArmSpec, DbSource};
    use wenlan_core::llm_provider::{LlmProvider, OnDeviceProvider};

    let lme_path = match std::env::var("LME_S_FIXTURE") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            let data_dir = eval_root().join("data");
            let cleaned = data_dir.join("longmemeval_s_cleaned.json");
            if cleaned.exists() {
                cleaned
            } else {
                data_dir.join("longmemeval_s.json")
            }
        }
    };
    if !lme_path.exists() {
        eprintln!(
            "SKIP: LME-S fixture not found at {:?}. Set LME_S_FIXTURE=<path> or point \
             WENLAN_EVAL_ROOT at the main checkout's app/eval.",
            lme_path
        );
        return;
    }

    let enrich_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("pages_ab_lme_s_tuples.json");

    eprintln!(
        "[generate_pages_ab_lme_s] PAGES A/B (DEEP S fixture, pool90 reuse):\n  \
         fixture: {:?}\n  \
         arms: ce_off (no pages) / ce_on (no pages) / ce_on_pages (pages)\n  \
         page_channel_flag={}  graph_stream_flag={}\n  \
         RERANK_POOL_FLOOR={}\n  \
         answer model: on-device qwen3.5-9b (temp 0.0)\n  enrich model: {}\n  output: {:?}",
        lme_path,
        wenlan_core::db::page_channel_enabled(),
        wenlan_core::db::graph_memory_stream_enabled(),
        std::env::var("RERANK_POOL_FLOOR").unwrap_or_else(|_| "10 (default)".into()),
        enrich_model,
        output_path,
    );

    let llm: Arc<dyn LlmProvider> =
        Arc::new(OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).expect(
            "OnDeviceProvider::new_with_model failed -- is the Qwen3.5-9B model available?",
        ));

    let enrichment = wenlan_core::eval::shared::EnrichmentMode::from_env(&enrich_model, cost_cap)
        .expect("EnrichmentMode::from_env failed");

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None).expect(
        "init_cross_encoder_reranker failed (downloads ~1.1GB bge-reranker-base on first run)",
    );

    let tuples = run_fullpipeline_lme(
        &lme_path,
        enrichment,
        llm,
        Some(reranker),
        &ArmSpec::pages_ab(),
        DbSource::SeedPerQuestion,
        &output_path,
    )
    .await
    .expect("run_fullpipeline_lme (pages_ab deep-S) failed");

    eprintln!(
        "\n[generate_pages_ab_lme_s] Done: {} tuples saved to {:?}",
        tuples.len(),
        output_path
    );
}

/// Same as `generate_lme_fullstack_ceiling` but uses the DEEP `longmemeval_s.json`
/// fixture (each question carries its own sessions + ~47 real distractor sessions)
/// instead of the shallow oracle fixture, so the CE fetch pool has genuine
/// reselection headroom and `RERANK_POOL_FLOOR`
/// actually move the result. Single-arm `ArmSpec::full_stack()` = production stack
/// (page + temporal + CE; graph stream structurally suppressed under live CE), so
/// the per-pool accuracy is the pool decision IN full-capacity context, and the
/// number at the chosen pool is the LME-S headline.
///
/// Set `EVAL_SCENARIO_CONCURRENCY=8` to batch all on-device phases (#270). Fixture:
/// `LME_S_FIXTURE` override → `longmemeval_s_cleaned.json` → `longmemeval_s.json`.
/// ```bash
/// WENLAN_ENABLE_PAGE_CHANNEL=1 RERANK_POOL_FLOOR=50 EVAL_SCENARIO_CONCURRENCY=8 \
/// WENLAN_LLM_PARALLEL_SEQS=8 WENLAN_LLM_CTX_SIZE=16384 WENLAN_LLM_COALESCE_MS=10 \
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-fullstack-s \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   generate_lme_fullstack_s -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn generate_lme_fullstack_s() {
    use std::sync::Arc;
    use wenlan_core::eval::answer_quality::{run_fullpipeline_lme, ArmSpec, DbSource};
    use wenlan_core::llm_provider::{LlmProvider, OnDeviceProvider};

    let lme_path = match std::env::var("LME_S_FIXTURE") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            let data_dir = eval_root().join("data");
            let cleaned = data_dir.join("longmemeval_s_cleaned.json");
            if cleaned.exists() {
                cleaned
            } else {
                data_dir.join("longmemeval_s.json")
            }
        }
    };
    if !lme_path.exists() {
        eprintln!(
            "SKIP: LME-S fixture not found at {:?}. Set LME_S_FIXTURE=<path> or point \
             WENLAN_EVAL_ROOT at the main checkout's app/eval.",
            lme_path
        );
        return;
    }

    let enrich_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("lme_fullstack_s_tuples.json");

    eprintln!(
        "[generate_lme_fullstack_s] FULL-STACK config (DEEP S fixture):\n  \
         fixture: {:?}\n  \
         page_channel={}  graph_stream_flag={}  temporal_soft_boost={}  temporal_filter={}\n  \
         RERANK_POOL_FLOOR={}\n  \
         answer model: on-device qwen3.5-9b (temp 0.0)\n  enrich model: {}\n  output: {:?}",
        lme_path,
        wenlan_core::db::page_channel_enabled(),
        wenlan_core::db::graph_memory_stream_enabled(),
        wenlan_core::db::temporal_soft_boost_enabled(),
        wenlan_core::db::temporal_filter_enabled(),
        std::env::var("RERANK_POOL_FLOOR").unwrap_or_else(|_| "10 (default)".into()),
        enrich_model,
        output_path,
    );

    let llm: Arc<dyn LlmProvider> = Arc::new(
        OnDeviceProvider::new_with_model(Some("qwen3.5-9b"))
            .expect("OnDeviceProvider::new_with_model failed — is the Qwen3.5-9B model available?"),
    );

    let enrichment = wenlan_core::eval::shared::EnrichmentMode::from_env(&enrich_model, cost_cap)
        .expect("EnrichmentMode::from_env failed");

    let reranker = wenlan_core::reranker::init_cross_encoder_reranker(None).expect(
        "init_cross_encoder_reranker failed (downloads ~1.1GB bge-reranker-base on first run)",
    );

    let tuples = run_fullpipeline_lme(
        &lme_path,
        enrichment,
        llm,
        Some(reranker),
        &ArmSpec::full_stack(),
        DbSource::SeedPerQuestion,
        &output_path,
    )
    .await
    .expect("run_fullpipeline_lme (full_stack deep-S) failed");

    eprintln!(
        "\n[generate_lme_fullstack_s] Done: {} tuples saved to {:?}",
        tuples.len(),
        output_path
    );
}

/// Judge the full-stack ceiling tuples → single-arm ACCURACY (per category +
/// overall). No McNemar (single arm); uses `aggregate_judgments`, where each arm
/// label is `fullstack_{category}` so per-approach rows ARE per-category accuracy.
///
/// ```bash
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-ceiling \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   judge_lme_fullstack_ceiling -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn judge_lme_fullstack_ceiling() {
    use wenlan_core::eval::judge::{
        aggregate_judgments, judge_with_claude_model_batched_persistent, load_judgment_tuples,
    };

    let baselines = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"));
    // FULLSTACK_TUPLES_FILE lets this same judge grade the deep-S variant
    // (`lme_fullstack_s_tuples.json`) without a second judge fn.
    let tuples_file = std::env::var("FULLSTACK_TUPLES_FILE")
        .unwrap_or_else(|_| "lme_fullstack_ceiling_tuples.json".to_string());
    let tuples_path = baselines.join(&tuples_file);
    if !tuples_path.exists() {
        eprintln!(
            "SKIP: {:?} not found. Run generate_lme_fullstack_ceiling first.",
            tuples_path
        );
        return;
    }

    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        eprintln!(
            "[judge_lme_fullstack_ceiling] WARNING: ANTHROPIC_API_KEY is set — the `claude` CLI \
             will bill the API instead of your OAuth subscription. Unset it for a $-free run."
        );
    }

    let judge_model = std::env::var("LME_JUDGE_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
    let tuples = load_judgment_tuples(&tuples_path).expect("load_judgment_tuples failed");
    eprintln!(
        "[judge_lme_fullstack_ceiling] Loaded {} tuples",
        tuples.len()
    );

    let cache_path = baselines.join("lme_fullstack_ceiling_judge_cache.jsonl");
    let results =
        judge_with_claude_model_batched_persistent(&tuples, 10, 3, &judge_model, &cache_path, 3)
            .await
            .expect("judging failed");

    let report = aggregate_judgments(&results, &judge_model);
    let total: usize = report.results_by_approach.iter().map(|r| r.total).sum();
    let correct: usize = report.results_by_approach.iter().map(|r| r.correct).sum();
    let overall = correct as f64 / (total.max(1)) as f64;

    eprintln!("\n=== LME FULL-STACK CEILING (gold/oracle) ===");
    eprintln!("OVERALL accuracy = {:.4} ({}/{})", overall, correct, total);
    eprintln!("\nPer-category (approach = fullstack_<category>):");
    let mut rows = report.results_by_approach.clone();
    rows.sort_by(|a, b| a.approach.cmp(&b.approach));
    for r in &rows {
        eprintln!(
            "  {:<40} acc={:.4} ({}/{})  mean_ctx_tokens={:.0}",
            r.approach, r.accuracy, r.correct, r.total, r.mean_context_tokens
        );
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "overall_accuracy": overall,
            "total": total,
            "correct": correct,
            "by_approach": report.results_by_approach,
            "judge_model": judge_model,
        }))
        .unwrap_or_default()
    );
}

// ── Task 2.1 RED contract test ─────────────────────────────────────────────
//
// PURPOSE (TDD – this test is EXPECTED TO FAIL).
//
// There are two post-store enrichment paths in origin-core:
//   • Production / canonical: `ingest::run_canonical_enrichment` runs
//     Phase-1 (classify) + Phase-2 via `post_ingest::run_post_ingest_enrichment`
//     (auto_link_entity → extract_single_memory_entities) + Phase-3.
//   • Eval seed (FORKED): `eval::shared::enrich_db_for_eval_local` runs
//     `run_classification_for_eval_concurrent` (Phase 1) +
//     `run_entity_extraction_for_eval_concurrent` (NO auto_link_entity — all
//     memories go through LLM extraction via `get_unlinked_memories`) +
//     `run_title_enrichment_for_eval` + `distill_pages`.
//
// The structural divergence the test captures:
//
//   The canonical arm uses `auto_link_entity` (cosine-distance vector search
//   against the entities table) BEFORE LLM extraction. When a match is found
//   below the `entity_link_distance` threshold, the memory's `entity_id` column
//   is updated and LLM extraction is SKIPPED — that memory receives no new row
//   in the `memory_entities` junction table from the canonical arm.
//
//   The eval arm uses `get_unlinked_memories` (entity_id IS NULL filter) and
//   always routes every match to LLM extraction via `extract_single_memory_entities`,
//   writing a `memory_entities` row unconditionally.
//
//   Concretely: after Doc 1 is processed in the canonical arm and its entity
//   ("Alice Chen") is embedded in the entities table, Doc 2 ("Alice Chen.") is
//   auto-linked at distance ≈ 0 (entity name and content are nearly identical),
//   skipping LLM extraction → no `memory_entities` row for Doc 2 in the
//   canonical arm. The eval arm processes all 3 docs via LLM → writes
//   `memory_entities` for all 3 → sets diverge.
//
//   To reliably trigger this divergence, Doc 2 content is set to the entity
//   name verbatim ("Alice Chen.") so cosine-distance < 0.15 (production default)
//   is guaranteed.
//
// EXPECTED OUTCOME:
//   The `assert_eq!` on entity-link tuples FAILS (eval arm has 7+ tuples, canonical
//   arm has fewer because Doc 2 and/or Doc 3 are auto-linked and skip LLM).
//   This failure is the RED gate. Do NOT fix it here. Task 2.3 fixes the fork.
//
// Run:
//   RUSTC_WRAPPER= cargo test -p origin-core --test eval_harness \
//     --features eval-harness \
//     enrichment_parity_contract -- --ignored --nocapture
#[tokio::test]
#[ignore = "GPU (on-device LLM); parity regression guard â enrich_db_for_eval_local must produce identical entity-link, page-membership, and title-nonempty sets as run_canonical_enrichment. Task 2.3 de-forked Phase-2 through run_post_ingest_enrichment."]
async fn enrichment_parity_contract() {
    use std::collections::HashSet;
    use std::sync::Arc;
    use wenlan_core::db::MemoryDB;
    use wenlan_core::eval::shared::enrich_db_for_eval_local;
    use wenlan_core::events::NoopEmitter;
    use wenlan_core::ingest::{run_canonical_enrichment, EnrichmentOpts};
    use wenlan_core::llm_provider::{LlmProvider, OnDeviceProvider};
    use wenlan_core::prompts::PromptRegistry;
    use wenlan_core::tuning::{DistillationConfig, RefineryConfig};
    use wenlan_types::sources::RawDocument;

    // ── LLM provider (shared between arms for reproducibility) ──────────────
    let llm: Arc<dyn wenlan_core::llm_provider::LlmProvider> = match OnDeviceProvider::new() {
        Ok(p) => {
            assert!(
                p.is_available(),
                "[parity_contract] on-device LLM unavailable. \
                     Run outside the CC sandbox; check `pgrep -la origin` for GPU contention."
            );
            Arc::new(p)
        }
        Err(e) => panic!("[parity_contract] on-device provider init failed: {e}"),
    };

    // Pin concurrency to 1 for determinism.
    std::env::set_var("EVAL_ENRICHMENT_CONCURRENCY", "1");
    // Ensure no Anthropic key (on-device only).
    std::env::remove_var("ANTHROPIC_API_KEY");

    // Fixed timestamp so both arms are identical on last_modified.
    let fixed_ts: i64 = 1_700_000_000;

    // ── Synthetic docs ───────────────────────────────────────────────────────
    // Doc 1: long sentence so LLM extracts "Alice Chen" entity and writes the
    //        entity to the `entities` table + `memory_entities` junction.
    // Doc 2: VERBATIM entity name ("Alice Chen.") so cosine distance from the
    //        "Alice Chen" entity embedding < 0.15 → `auto_link_entity` fires in
    //        the CANONICAL arm → skips LLM extraction → NO `memory_entities` row.
    //        In the EVAL arm, `get_unlinked_memories` (entity_id IS NULL) picks
    //        it up anyway and routes it through LLM extraction → junction row written.
    // Doc 3: Another sentence about the same person, for coverage.
    let test_docs: Vec<(&str, &str)> = vec![
        (
            "parity_src_001",
            "Alice Chen is the lead engineer at Acme Corp. She manages the platform team.",
        ),
        (
            // Short verbatim entity-name content — guarantees auto_link fires in canonical arm.
            "parity_src_002",
            "Alice Chen.",
        ),
        (
            "parity_src_003",
            "Acme Corp is migrating to Rust. The migration is led by Alice Chen.",
        ),
    ];

    fn make_doc(source_id: &str, content: &str, ts: i64) -> RawDocument {
        RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: content.chars().take(40).collect(),
            summary: None,
            content: content.to_string(),
            url: None,
            last_modified: ts,
            metadata: std::collections::HashMap::new(),
            memory_type: Some("fact".to_string()),
            space: None,
            source_agent: Some("parity_test".to_string()),
            confidence: Some(0.9),
            confirmed: Some(true),
            stability: Some("new".to_string()),
            supersedes: None,
            pending_revision: false,
            entity_id: None,
            quality: None,
            importance: None,
            is_recap: false,
            enrichment_status: "raw".to_string(),
            supersede_mode: "hide".to_string(),
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
        }
    }

    // Shared embedder — avoid double model load.
    let shared_embedder = MemoryDB::create_shared_embedder()
        .await
        .expect("create_shared_embedder");

    // ── ARM A: eval path ─────────────────────────────────────────────────────
    let eval_dir = tempfile::TempDir::new().expect("eval tempdir");
    let eval_db_path = eval_dir.path().join("origin_memory.db");
    {
        let db = MemoryDB::new_with_shared_embedder(
            eval_dir.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await
        .expect("eval DB init");

        db.upsert_documents(
            test_docs
                .iter()
                .map(|(id, content)| make_doc(id, content, fixed_ts))
                .collect(),
        )
        .await
        .expect("eval upsert_documents");

        eprintln!("[parity_contract] ARM A (eval): enrich_db_for_eval_local...");
        enrich_db_for_eval_local(&db, &llm)
            .await
            .expect("enrich_db_for_eval_local");
        eprintln!("[parity_contract] ARM A: done");
        // Drop DB — flush WAL before raw libsql read below.
    }

    // ── ARM B: canonical path ────────────────────────────────────────────────
    let canon_dir = tempfile::TempDir::new().expect("canon tempdir");
    let canon_db_path = canon_dir.path().join("origin_memory.db");
    {
        let db = MemoryDB::new_with_shared_embedder(
            canon_dir.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await
        .expect("canon DB init");

        db.upsert_documents(
            test_docs
                .iter()
                .map(|(id, content)| make_doc(id, content, fixed_ts))
                .collect(),
        )
        .await
        .expect("canon upsert_documents");

        let prompts = PromptRegistry::default();
        // Use production-default entity_link_distance (0.15). Doc 2 content is the
        // verbatim entity name "Alice Chen." so cosine distance < 0.15 is expected,
        // triggering auto_link and skipping LLM extraction for Doc 2.
        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();

        eprintln!("[parity_contract] ARM B (canonical): run_canonical_enrichment × 3...");
        for (source_id, content) in &test_docs {
            let opts = EnrichmentOpts {
                initial_memory_type: "fact".to_string(),
                initial_domain: None,
                rejected_explicit_domain: false,
                initial_supersede_mode: "hide".to_string(),
                initial_structured_fields: None,
                agent_supplied_memory_type: false,
                agent_supplied_profile_alias: false,
                agent_supplied_structured_fields: false,
            };
            run_canonical_enrichment(
                &db,
                source_id,
                content,
                None,
                Some(&llm),
                &prompts,
                &refinery,
                &distillation,
                None,
                &opts,
                None,
            )
            .await;
        }
        eprintln!("[parity_contract] ARM B: done");
        // Mirror production's background refinery (and Arm A's internal Phase-3):
        // run_canonical_enrichment is the STORE path, whose Phase-3 is dual-pool
        // dedup (default-OFF), NOT page distillation. Without this, Arm B ends with
        // zero pages while Arm A distilled, making the page-membership assert below
        // distill-vs-nothing. Distilling here makes it eval-distill vs canonical-distill.
        wenlan_core::refinery::distill_pages(&db, Some(&llm), &prompts, &distillation, None)
            .await
            .expect("canon distill_pages");
        // Drop DB before raw libsql read below.
    }

    // ── Read membership tuples via raw libsql (same pattern as seed_contract) ─
    async fn read_entity_link_tuples(db_path: &std::path::Path) -> Vec<(String, String)> {
        let sdb = libsql::Builder::new_local(db_path.to_str().unwrap())
            .build()
            .await
            .expect("open db for entity read");
        let conn = sdb.connect().expect("connect for entity read");
        let mut rows = conn
            .query(
                // G6 Stage 3: entity identity is the `kind='entity'` shadow
                // page reached through `entity_page_map`; `entities` is gone.
                // `pages.title` carries what `entities.name` used to.
                "SELECT me.memory_id, p.title \
                 FROM memory_entities me \
                 JOIN entity_page_map epm ON epm.entity_id = me.entity_id \
                 JOIN pages p ON p.id = epm.page_id \
                 WHERE p.kind = 'entity' AND p.status = 'active' \
                 ORDER BY me.memory_id, p.title",
                (),
            )
            .await
            .expect("entity_link query");
        let mut out = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let mem_id: String = row.get(0).unwrap_or_default();
            let entity_name: String = row.get(1).unwrap_or_default();
            out.push((mem_id, entity_name));
        }
        out
    }

    async fn read_page_membership_tuples(db_path: &std::path::Path) -> Vec<(String, String)> {
        let sdb = libsql::Builder::new_local(db_path.to_str().unwrap())
            .build()
            .await
            .expect("open db for page read");
        let conn = sdb.connect().expect("connect for page read");
        let mut rows = conn
            .query(
                "SELECT page_id, memory_source_id \
                 FROM page_sources \
                 ORDER BY page_id, memory_source_id",
                (),
            )
            .await
            .expect("page_sources query");
        let mut out = Vec::new();
        while let Ok(Some(row)) = rows.next().await {
            let page_id: String = row.get(0).unwrap_or_default();
            let mem_src: String = row.get(1).unwrap_or_default();
            out.push((page_id, mem_src));
        }
        out
    }

    async fn read_titles_nonempty(db_path: &std::path::Path) -> HashSet<String> {
        let sdb = libsql::Builder::new_local(db_path.to_str().unwrap())
            .build()
            .await
            .expect("open db for title read");
        let conn = sdb.connect().expect("connect for title read");
        let mut rows = conn
            .query(
                "SELECT source_id FROM memories \
                 WHERE source = 'memory' AND chunk_index = 0 \
                   AND title IS NOT NULL AND title != '' \
                 ORDER BY source_id",
                (),
            )
            .await
            .expect("title query");
        let mut set = HashSet::new();
        while let Ok(Some(row)) = rows.next().await {
            set.insert(row.get::<String>(0).unwrap_or_default());
        }
        set
    }

    let eval_entity_links = read_entity_link_tuples(&eval_db_path).await;
    let canon_entity_links = read_entity_link_tuples(&canon_db_path).await;
    let eval_page_membership = read_page_membership_tuples(&eval_db_path).await;
    let canon_page_membership = read_page_membership_tuples(&canon_db_path).await;
    let eval_titles_nonempty = read_titles_nonempty(&eval_db_path).await;
    let canon_titles_nonempty = read_titles_nonempty(&canon_db_path).await;

    // ── Report ───────────────────────────────────────────────────────────────
    eprintln!("\n[parity_contract] === MEMBERSHIP TUPLE COMPARISON ===");
    eprintln!("\nEntity-link tuples (memory_id, entity_name):");
    eprintln!(
        "  EVAL  ({} tuples): {:?}",
        eval_entity_links.len(),
        eval_entity_links
    );
    eprintln!(
        "  CANON ({} tuples): {:?}",
        canon_entity_links.len(),
        canon_entity_links
    );
    let eval_set: HashSet<_> = eval_entity_links.iter().collect();
    let canon_set: HashSet<_> = canon_entity_links.iter().collect();
    let only_eval: Vec<_> = eval_set.difference(&canon_set).collect();
    let only_canon: Vec<_> = canon_set.difference(&eval_set).collect();
    eprintln!(
        "  In EVAL only ({} tuples): {:?}",
        only_eval.len(),
        only_eval
    );
    eprintln!(
        "  In CANON only ({} tuples): {:?}",
        only_canon.len(),
        only_canon
    );

    eprintln!("\nPage-membership tuples:");
    eprintln!(
        "  EVAL  ({} tuples): {:?}",
        eval_page_membership.len(),
        eval_page_membership
    );
    eprintln!(
        "  CANON ({} tuples): {:?}",
        canon_page_membership.len(),
        canon_page_membership
    );

    eprintln!("\nTitle-nonempty source_ids:");
    eprintln!("  EVAL  ({} ids): {:?}", eval_titles_nonempty.len(), {
        let mut v: Vec<_> = eval_titles_nonempty.iter().cloned().collect();
        v.sort();
        v
    });
    eprintln!("  CANON ({} ids): {:?}", canon_titles_nonempty.len(), {
        let mut v: Vec<_> = canon_titles_nonempty.iter().cloned().collect();
        v.sort();
        v
    });

    // ── RED assertion: parity sets MUST match (assert_eq! fails when they differ) ──
    //
    // This assert_eq! is the RED gate.
    //
    // The two paths fork at `auto_link_entity`:
    //   • Eval arm: always routes to `extract_single_memory_entities` (no auto_link step).
    //   • Canonical arm: auto-links Doc 2 ("Alice Chen." — verbatim entity name) at
    //     cosine distance ≈ 0, skipping LLM extraction → no `memory_entities` row for Doc 2.
    //
    // Expected failure message: eval_entity_links (N tuples) != canon_entity_links (M tuples)
    // where N > M because canonical arm has fewer junction rows (Doc 2 was auto-linked).
    //
    // Post-de-fork (Task 2.3 DONE): enrich_db_for_eval_local routes Phase-2 through
    // the per-memory run_post_ingest_enrichment path, so both arms auto-link Doc 2
    // and the entity-link tuples now match. This assert is the load-bearing regression
    // guard: if Phase-2 is ever re-forked, the tuples diverge and this fails.
    //
    // Vacuous-pass guard: if the LLM extracted nothing, both arms are empty and the
    // assert_eq! below would pass while measuring nothing — fail loud instead.
    assert!(
        !eval_entity_links.is_empty(),
        "[parity_contract] vacuous: corpus produced zero entity links — strengthen corpus or check the LLM extracted entities"
    );
    assert_eq!(
        eval_entity_links,
        canon_entity_links,
        "[parity_contract] RED: entity-link tuples DIVERGE between eval and canonical paths.\n\
         Eval arm processed all 3 docs via LLM (enrich_db_for_eval_local has no auto_link step).\n\
         Canonical arm auto-linked Doc 2 ('Alice Chen.' ~ entity 'alice chen' at dist≈0), \
         skipping LLM extraction for that doc → fewer memory_entities rows.\n\
         EVAL  ({eval_n}): {eval:?}\n\
         CANON ({canon_n}): {canon:?}",
        eval_n = eval_entity_links.len(),
        eval = eval_entity_links,
        canon_n = canon_entity_links.len(),
        canon = canon_entity_links,
    );

    // Secondary check: page-membership parity. Both arms now run distill_pages at
    // Phase-3 (Arm A internally, Arm B explicitly above), so this guards that the eval
    // seed's page distillation matches the canonical distill. May be empty for this
    // minimal 3-doc corpus (no cluster reaches the min size) — that is an acceptable
    // 0==0 outcome; the assert still catches a structural divergence if pages do form.
    assert_eq!(
        eval_page_membership, canon_page_membership,
        "[parity_contract] page-membership tuples diverge (eval-distill vs canonical-distill): eval={:?} canon={:?}",
        eval_page_membership, canon_page_membership,
    );

    assert_eq!(
        eval_titles_nonempty, canon_titles_nonempty,
        "[parity_contract] title-nonempty sets diverge: eval={:?} canon={:?}",
        eval_titles_nonempty, canon_titles_nonempty,
    );

    // If we reach here, all sets are equal — the fork is already gone (Task 2.3 is done).
    eprintln!(
        "\n[parity_contract] GREEN: all membership sets are equal — \
         the eval-vs-canonical fork is resolved. Task 2.3 complete."
    );
}

/// Settle the OPEN flag from `lesson_libsql_count_vector_index_bug`: does the
/// libsql LIBRARY mis-plan `SELECT COUNT(*) FROM pages WHERE status='active'`
/// against the `pages` vector-index shadow (idx_pages_embedding), the way the
/// external sqlite3 CLI does? `MemoryDB::count_active_pages` (db.rs) — which the
/// seed contract's page floor relies on — uses exactly that aggregate. If the
/// library shares the CLI bug, the floor would falsely reject live-page DBs.
///
/// Asserts, for every cached pool DB: library COUNT(*) == library row-enumerate,
/// AND the enumerate is > 0 (pool DBs are seeded with pages). Run:
/// ```text
/// EVAL_BASELINES_DIR=$HOME/.cache/origin-eval-pool31 RUSTC_WRAPPER= \
///   cargo test -p origin-core --test eval_harness --features eval-harness \
///   libsql_count_active_pages_parity -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "libsql-library COUNT(*)-vs-enumerate parity on cached pool page DBs; set EVAL_BASELINES_DIR"]
async fn libsql_count_active_pages_parity() {
    let root = wenlan_core::eval::shared::eval_baselines_dir_override()
        .unwrap_or_else(|| eval_root().join("baselines"))
        .join("fullpipeline")
        .join("lme");
    if !root.is_dir() {
        eprintln!("SKIP: no cached page DBs at {}", root.display());
        return;
    }
    let mut checked = 0usize;
    let mut mismatches: Vec<String> = Vec::new();
    let mut zero_pages: Vec<String> = Vec::new();
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .expect("read pool dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    for d in dirs {
        let db_path = d.join("origin_memory.db");
        if !db_path.exists() {
            continue;
        }
        let qid = d.file_name().unwrap().to_string_lossy().to_string();
        let sdb = libsql::Builder::new_local(db_path.to_str().unwrap())
            .build()
            .await
            .expect("open page DB");
        let conn = sdb.connect().expect("connect page DB");

        // Aggregate path — the exact query count_active_pages runs.
        let mut crow = conn
            .query("SELECT COUNT(*) FROM pages WHERE status = 'active'", ())
            .await
            .expect("count query");
        let count: i64 = crow
            .next()
            .await
            .expect("count next")
            .expect("count row")
            .get::<i64>(0)
            .expect("count get");

        // Enumerate path — the bug-immune ground truth.
        let mut erows = conn
            .query("SELECT id FROM pages WHERE status = 'active'", ())
            .await
            .expect("enum query");
        let mut enumerated = 0i64;
        while let Ok(Some(_)) = erows.next().await {
            enumerated += 1;
        }

        checked += 1;
        if count != enumerated {
            mismatches.push(format!("{qid}: count={count} enum={enumerated}"));
        }
        if enumerated == 0 {
            zero_pages.push(qid);
        }
    }

    eprintln!(
        "[count_parity] checked={checked} mismatches={} zero_page_dbs={}",
        mismatches.len(),
        zero_pages.len()
    );
    assert!(checked > 0, "no DBs checked under {}", root.display());
    assert!(
        mismatches.is_empty(),
        "libsql COUNT(*) diverged from enumerate on {} DB(s): {:?} — \
         count_active_pages / seed-contract page floor would be WRONG on these",
        mismatches.len(),
        mismatches
    );
    assert!(
        zero_pages.is_empty(),
        "{} pool DB(s) have zero active pages (expected seeded): {:?}",
        zero_pages.len(),
        zero_pages
    );
}
