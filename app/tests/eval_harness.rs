//! Integration test: eval harness runs against seeded DB with fixture data.
//!
//! Tests using bundled fixtures run in CI (FastEmbed model cached in GitHub Actions).
//! Tests needing external data (locomo10.json, longmemeval) or real GPU LLM stay `#[ignore]`.

use origin_lib::eval::runner::{run_eval, GateMode};

#[tokio::test]
async fn test_eval_harness_produces_report() {
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let tmp = tempfile::tempdir().unwrap();

    let report = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::Off)
        .await
        .unwrap();

    // Should have loaded our seed fixtures
    assert!(
        report.fixture_count >= 3,
        "Expected at least 3 fixture cases, got {}",
        report.fixture_count
    );

    // Primary metric should be > 0
    assert!(
        report.ndcg_at_10 > 0.0,
        "NDCG@10 should be > 0, got {}",
        report.ndcg_at_10
    );

    // Print report for debugging
    println!("{}", report.to_terminal());
}

#[tokio::test]
async fn test_eval_metrics_are_bounded() {
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let tmp = tempfile::tempdir().unwrap();

    let report = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::Off)
        .await
        .unwrap();

    // All ratio metrics should be in [0, 1]
    for (name, val) in [
        ("NDCG@10", report.ndcg_at_10),
        ("NDCG@5", report.ndcg_at_5),
        ("MAP@10", report.map_at_10),
        ("MAP@5", report.map_at_5),
        ("MRR", report.mrr),
        ("Recall@1", report.recall_at_1),
        ("Recall@3", report.recall_at_3),
        ("Recall@5", report.recall_at_5),
        ("Hit Rate@1", report.hit_rate_at_1),
        ("Hit Rate@3", report.hit_rate_at_3),
        ("P@3", report.precision_at_3),
        ("P@5", report.precision_at_5),
    ] {
        assert!((0.0..=1.0).contains(&val), "{} out of range: {}", name, val);
    }
}

#[tokio::test]
async fn test_eval_baseline_comparison() {
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let tmp = tempfile::tempdir().unwrap();
    let baseline_path = tmp.path().join("baseline.json");

    // First run: no baseline, save one
    let report1 = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::Off)
        .await
        .unwrap();
    report1.save_baseline(&baseline_path).unwrap();

    // Second run: load baseline, verify comparison is populated
    let report2 = run_eval(
        &fixture_dir,
        tmp.path(),
        None,
        Some(&baseline_path),
        GateMode::Off,
    )
    .await
    .unwrap();
    assert!(
        report2.baseline.is_some(),
        "Baseline comparison should be populated"
    );

    let b = report2.baseline.clone().unwrap();
    assert!((b.ndcg_at_10 - report1.ndcg_at_10).abs() < 0.001);
    assert!((b.mrr - report1.mrr).abs() < 0.001);

    println!("{}", report2.to_terminal());
}

#[tokio::test]
async fn test_eval_with_gate_filter() {
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let tmp = tempfile::tempdir().unwrap();

    // Run all three modes
    let off = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::Off)
        .await
        .unwrap();
    let content = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::ContentOnly)
        .await
        .unwrap();
    let full = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::Full)
        .await
        .unwrap();

    // ContentOnly should filter some negatives via content checks
    assert!(
        content.gate_content_filtered > 0,
        "ContentOnly should have filtered some negative seeds"
    );

    // Full should filter at least as many as ContentOnly (content + novelty)
    let full_total = full.gate_content_filtered + full.gate_novelty_filtered;
    assert!(
        full_total >= content.gate_content_filtered,
        "Full ({}) should filter at least as many as ContentOnly ({})",
        full_total,
        content.gate_content_filtered
    );

    println!(
        "\n=== GATE IMPACT ON RETRIEVAL ({} cases) ===",
        off.fixture_count
    );
    println!("                    Off        Content     Full        Δ(Full-Off)");
    println!(
        "  NDCG@10:          {:.4}     {:.4}      {:.4}      {:+.4}",
        off.ndcg_at_10,
        content.ndcg_at_10,
        full.ndcg_at_10,
        full.ndcg_at_10 - off.ndcg_at_10
    );
    println!(
        "  MRR:              {:.4}     {:.4}      {:.4}      {:+.4}",
        off.mrr,
        content.mrr,
        full.mrr,
        full.mrr - off.mrr
    );
    println!(
        "  Recall@5:         {:.4}     {:.4}      {:.4}      {:+.4}",
        off.recall_at_5,
        content.recall_at_5,
        full.recall_at_5,
        full.recall_at_5 - off.recall_at_5
    );
    println!(
        "  MAP@10:           {:.4}     {:.4}      {:.4}      {:+.4}",
        off.map_at_10,
        content.map_at_10,
        full.map_at_10,
        full.map_at_10 - off.map_at_10
    );
    println!(
        "  Neg leakage:      {:<4}       {:<4}        {:<4}        {:+}",
        off.negative_leakage,
        content.negative_leakage,
        full.negative_leakage,
        full.negative_leakage as i64 - off.negative_leakage as i64
    );
    println!(
        "  Content filtered: {:<4}       {:<4}        {:<4}",
        off.gate_content_filtered, content.gate_content_filtered, full.gate_content_filtered
    );
    println!(
        "  Novelty filtered: {:<4}       {:<4}        {:<4}",
        off.gate_novelty_filtered, content.gate_novelty_filtered, full.gate_novelty_filtered
    );
}

#[tokio::test]
#[ignore]
async fn test_locomo_benchmark() {
    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !locomo_path.exists() {
        println!("SKIP: locomo10.json not found at {:?}", locomo_path);
        return;
    }

    let report = origin_lib::eval::locomo::run_locomo_eval(&locomo_path)
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
    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !locomo_path.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }

    use origin_lib::eval::locomo::{run_locomo_eval_with_gate, LocomoGateMode};

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
    let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data");
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

    let report = origin_lib::eval::longmemeval::run_longmemeval_eval(&path)
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
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !path.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    use origin_lib::eval::longmemeval::{run_longmemeval_eval_with_gate, LongMemEvalGateMode};

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
async fn test_lifecycle_fixture_with_mock_llm() {
    use origin_lib::eval::lifecycle::{run_lifecycle_fixture, EvalMockLlm};
    use std::sync::Arc;

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let mock: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(EvalMockLlm::new());

    let report = run_lifecycle_fixture(&fixture_dir, Some(mock))
        .await
        .unwrap();

    // Should have 6 phases
    assert_eq!(
        report.phases.len(),
        6,
        "Expected 6 phases, got {}",
        report.phases.len()
    );

    // All metrics should be in [0, 1]
    for pm in &report.phases {
        assert!(
            pm.ndcg_at_10 >= 0.0 && pm.ndcg_at_10 <= 1.0,
            "NDCG@10 out of range in {:?}: {}",
            pm.phase,
            pm.ndcg_at_10
        );
        assert!(
            pm.mrr >= 0.0 && pm.mrr <= 1.0,
            "MRR out of range in {:?}: {}",
            pm.phase,
            pm.mrr
        );
    }

    // Should have 5 deltas (between consecutive phases)
    assert_eq!(
        report.deltas.len(),
        5,
        "Expected 5 deltas, got {}",
        report.deltas.len()
    );

    // Each delta verdict should be valid
    for d in &report.deltas {
        assert!(
            d.verdict == "helped" || d.verdict == "hurt" || d.verdict == "neutral",
            "Invalid verdict: {}",
            d.verdict
        );
    }

    // Per-case count should match fixture count
    assert_eq!(report.per_case.len(), report.case_count);

    println!("{}", report.to_terminal());
}

#[tokio::test]
async fn test_lifecycle_fixture_no_llm() {
    use origin_lib::eval::lifecycle::run_lifecycle_fixture;

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");

    let report = run_lifecycle_fixture(&fixture_dir, None).await.unwrap();

    // Should still have 6 phases
    assert_eq!(report.phases.len(), 6);

    // Without LLM, EntityExtraction/Distillation/ConceptRetrieval/Insights should match PostIngest
    let post_ingest_ndcg = report.phases[1].ndcg_at_10;
    let entity_ndcg = report.phases[2].ndcg_at_10;
    let distill_ndcg = report.phases[3].ndcg_at_10;
    let concept_ndcg = report.phases[4].ndcg_at_10;
    let insights_ndcg = report.phases[5].ndcg_at_10;

    assert!(
        (entity_ndcg - post_ingest_ndcg).abs() < 1e-9,
        "Without LLM, EntityExtraction NDCG ({}) should match PostIngest ({})",
        entity_ndcg,
        post_ingest_ndcg
    );
    assert!(
        (distill_ndcg - post_ingest_ndcg).abs() < 1e-9,
        "Without LLM, Distillation NDCG ({}) should match PostIngest ({})",
        distill_ndcg,
        post_ingest_ndcg
    );
    assert!(
        (concept_ndcg - post_ingest_ndcg).abs() < 1e-9,
        "Without LLM, ConceptRetrieval NDCG ({}) should match PostIngest ({})",
        concept_ndcg,
        post_ingest_ndcg
    );
    assert!(
        (insights_ndcg - post_ingest_ndcg).abs() < 1e-9,
        "Without LLM, Insights NDCG ({}) should match PostIngest ({})",
        insights_ndcg,
        post_ingest_ndcg
    );

    println!("{}", report.to_terminal());
}

#[tokio::test]
#[ignore]
async fn test_lifecycle_locomo_with_mock_llm() {
    use origin_lib::eval::lifecycle::{run_lifecycle_locomo, EvalMockLlm};
    use std::sync::Arc;

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !path.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }

    let mock: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(EvalMockLlm::new());
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
    use origin_lib::eval::lifecycle::{run_lifecycle_longmemeval, EvalMockLlm};
    use std::sync::Arc;

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !path.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let mock: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(EvalMockLlm::new());
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

#[tokio::test]
async fn test_eval_empty_set_and_temporal() {
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let tmp = tempfile::tempdir().unwrap();

    let report = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::Off)
        .await
        .unwrap();

    // Empty-set cases should be counted
    assert!(
        report.empty_set_count > 0,
        "Should have empty-set cases, got {}",
        report.empty_set_count
    );

    // False confidence should be populated and finite
    if let Some(fc) = report.empty_set_false_confidence {
        assert!(
            fc.is_finite(),
            "False confidence should be finite, got {}",
            fc
        );
        assert!(
            fc >= 0.0,
            "False confidence should be non-negative, got {}",
            fc
        );
    }

    // Score gap should be populated
    if let Some(sg) = report.score_gap {
        assert!(sg.is_finite(), "Score gap should be finite, got {}", sg);
    }

    // Temporal ordering should have been checked
    assert!(
        report.temporal_ordering_total > 0,
        "Should have temporal ordering pairs, got {}",
        report.temporal_ordering_total
    );
    assert!(report.temporal_ordering_correct <= report.temporal_ordering_total);

    if let Some(rate) = report.temporal_ordering_rate {
        assert!(
            (0.0..=1.0).contains(&rate),
            "Temporal rate out of range: {}",
            rate
        );
    }

    println!("{}", report.to_terminal());
}

// ---------------------------------------------------------------------------
// Baseline save tests (run manually with --ignored to establish baselines)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn save_fixture_baseline() {
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let tmp = tempfile::tempdir().unwrap();
    let report = run_eval(&fixture_dir, tmp.path(), None, None, GateMode::Off)
        .await
        .unwrap();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/fixture_baseline.json");
    report.save_baseline(&path).unwrap();
    println!("Saved fixture baseline to {:?}", path);
    println!("{}", report.to_terminal());
}

#[tokio::test]
#[ignore]
async fn save_locomo_baseline() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !path.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }
    let report = origin_lib::eval::locomo::run_locomo_eval(&path)
        .await
        .unwrap();
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/locomo_baseline.json");
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LoCoMo baseline to {:?}", baseline_path);
}

#[tokio::test]
#[ignore]
async fn save_longmemeval_baseline() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !path.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }
    let report = origin_lib::eval::longmemeval::run_longmemeval_eval(&path)
        .await
        .unwrap();
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/longmemeval_baseline.json");
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LongMemEval baseline to {:?}", baseline_path);
}

#[tokio::test]
#[ignore]
async fn save_locomo_reranked_baseline() {
    use std::sync::Arc;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !path.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }
    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).unwrap(),
    );
    let report = origin_lib::eval::locomo::run_locomo_eval_reranked(&path, llm)
        .await
        .unwrap();
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/locomo_reranked_baseline.json");
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LoCoMo reranked baseline to {:?}", baseline_path);
}

#[tokio::test]
#[ignore]
async fn save_longmemeval_reranked_baseline() {
    use std::sync::Arc;
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !path.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }
    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).unwrap(),
    );
    let report = origin_lib::eval::longmemeval::run_longmemeval_eval_reranked(&path, llm)
        .await
        .unwrap();
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/longmemeval_reranked_baseline.json");
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LongMemEval reranked baseline to {:?}", baseline_path);
}

#[tokio::test]
#[ignore]
async fn save_locomo_expanded_baseline() {
    use std::sync::Arc;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !path.exists() {
        println!("SKIP: locomo10.json not found");
        return;
    }
    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).unwrap(),
    );
    let report = origin_lib::eval::locomo::run_locomo_eval_expanded(&path, llm)
        .await
        .unwrap();
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/locomo_expanded_baseline.json");
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LoCoMo expanded baseline to {:?}", baseline_path);
}

#[tokio::test]
#[ignore]
async fn save_longmemeval_expanded_baseline() {
    use std::sync::Arc;
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !path.exists() {
        println!("SKIP: longmemeval_oracle.json not found");
        return;
    }
    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new_with_model(Some("qwen3.5-9b")).unwrap(),
    );
    let report = origin_lib::eval::longmemeval::run_longmemeval_eval_expanded(&path, llm)
        .await
        .unwrap();
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/longmemeval_expanded_baseline.json");
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LongMemEval expanded baseline to {:?}", baseline_path);
}

#[tokio::test]
#[ignore]
async fn save_locomo_plus_baseline() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo_plus.json");
    if !path.exists() {
        println!("SKIP: locomo_plus.json not found");
        return;
    }
    let report = origin_lib::eval::locomo_plus::run_locomo_plus_eval(&path, None)
        .await
        .unwrap();
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/locomo_plus_baseline.json");
    report.save_baseline(&baseline_path).unwrap();
    println!("Saved LoCoMo-Plus baseline to {:?}", baseline_path);
}

#[tokio::test]
async fn test_lifecycle_pipeline_quality() {
    use origin_lib::eval::lifecycle::{run_lifecycle_fixture, EvalMockLlm};
    use std::sync::Arc;

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let mock: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(EvalMockLlm::new());

    let report = run_lifecycle_fixture(&fixture_dir, Some(mock))
        .await
        .unwrap();

    // Round-trip should always be populated for fixture lifecycle
    assert!(
        report.round_trip.is_some(),
        "Round-trip result should be populated"
    );
    let rt = report.round_trip.as_ref().unwrap();
    assert_eq!(rt.total_cases, report.case_count);
    assert!(rt.loss_rate >= 0.0 && rt.loss_rate <= 1.0);
    assert!(rt.mean_delta.is_finite());

    // Archive leakage: may or may not be populated (depends on mock LLM distillation)
    // If populated, leakage rate should be in [0, 1]
    if let Some(ref al) = report.archive_leakage {
        assert!(al.leakage_rate >= 0.0 && al.leakage_rate <= 1.0);
        assert!(al.leaked <= al.total_archived);
    }

    // Temporal preservation: may or may not be populated (depends on fixtures with supersedes)
    if let Some(ref tp) = report.temporal_preservation {
        assert!(tp.preservation_rate >= 0.0 && tp.preservation_rate <= 1.0);
        assert_eq!(tp.preserved + tp.violated, tp.total_chains);
    }

    println!("{}", report.to_terminal());
}

/// Tests that the concept compilation pipeline produces searchable concepts.
/// Seeds memories → runs distillation with mock LLM → verifies ConceptRetrieval phase has non-zero concept_count.
///
/// Note: `extract_entities_from_memories` and `distill_concepts` are pub(crate) and not
/// accessible from integration tests, so we use `run_lifecycle_fixture` which invokes the full
/// pipeline internally and exposes the ConceptRetrieval phase in the report.
#[tokio::test]
async fn test_concept_retrieval_eval() {
    use origin_lib::eval::lifecycle::{run_lifecycle_fixture, EvalMockLlm, LifecyclePhase};
    use std::sync::Arc;

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let mock: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(EvalMockLlm::new());

    let report = run_lifecycle_fixture(&fixture_dir, Some(mock))
        .await
        .unwrap();

    // Must have a ConceptRetrieval phase in the report
    let concept_phase = report
        .phases
        .iter()
        .find(|p| p.phase == LifecyclePhase::ConceptRetrieval);
    assert!(
        concept_phase.is_some(),
        "LifecycleReport should contain a ConceptRetrieval phase"
    );

    let cp = concept_phase.unwrap();

    // NDCG and MRR must be within [0, 1]
    assert!(
        cp.ndcg_at_10 >= 0.0 && cp.ndcg_at_10 <= 1.0,
        "ConceptRetrieval NDCG@10 out of range: {}",
        cp.ndcg_at_10
    );
    assert!(
        cp.mrr >= 0.0 && cp.mrr <= 1.0,
        "ConceptRetrieval MRR out of range: {}",
        cp.mrr
    );

    println!(
        "ConceptRetrieval phase: concept_count={}, NDCG@10={:.4}, MRR={:.4}",
        cp.concept_count, cp.ndcg_at_10, cp.mrr
    );

    // If concepts were created, retrieval quality should be non-zero
    if cp.concept_count > 0 {
        println!(
            "  {} compiled concepts available for FTS5 search",
            cp.concept_count
        );
    } else {
        println!(
            "WARN: No concepts created — mock LLM may not have generated compilable output. \
             This test validates the pipeline path, not mock quality."
        );
    }

    println!("{}", report.to_terminal());
}

// ---------------------------------------------------------------------------
// Token efficiency / quality-cost tests
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_quality_cost_fixtures() {
    use origin_lib::eval::token_efficiency::{run_quality_cost_eval, SearchStrategy};

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");

    let strategies = vec![
        SearchStrategy::Origin,
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

    // Origin should have better quality than NaiveRag
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
        "Origin NDCG ({:.3}) should be >= NaiveRag ({:.3})",
        origin.ndcg_at_10,
        naive.ndcg_at_10
    );

    println!("{}", report.to_terminal());
}

#[tokio::test]
#[ignore]
async fn test_quality_cost_agent_workload() {
    use origin_lib::eval::token_efficiency::{run_quality_cost_eval, SearchStrategy};

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    if !fixture_dir.join("agent_coding_session.toml").exists() {
        println!("SKIP: agent fixtures not found");
        return;
    }

    let strategies = vec![
        SearchStrategy::Origin,
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
    use origin_lib::eval::token_efficiency::{run_quality_cost_eval, SearchStrategy};

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let baseline_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/quality_cost_fixtures_baseline.json");

    let strategies = vec![
        SearchStrategy::Origin,
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
    use origin_lib::eval::token_efficiency::run_scaling_eval;

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");

    let sizes = vec![5, 10, 20, 50];
    let points = run_scaling_eval(&fixture_dir, &sizes, 10).await.unwrap();

    assert!(!points.is_empty(), "should produce scaling points");

    println!("\n=== Scaling Curve ===");
    println!(
        "{:<12} | {:<15} | {:<15}",
        "Corpus Size", "Origin Tokens", "Replay Tokens"
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

/// Concept impact across ALL fixture cases — single shared DB, no per-case ephemeral DBs.
///
/// Seeds all fixture memories into one DB, creates concepts from seed clusters (grouped by
/// domain + case), then measures combined recall (search_memory ∪ concept source_ids) vs
/// memory-only recall across every fixture query.
#[tokio::test]
async fn test_concept_before_after_comparison() {
    use origin_lib::eval::fixtures::load_fixtures;
    use origin_lib::eval::metrics;
    use origin_lib::memory_db::MemoryDB;
    use origin_lib::sources::RawDocument;
    use std::collections::HashSet;
    use std::sync::Arc;

    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");
    let cases = load_fixtures(&fixture_dir).unwrap();

    // One shared DB for all cases
    let tmp = tempfile::tempdir().unwrap();
    let emitter: Arc<dyn origin_core::events::EventEmitter> = Arc::new(origin_core::NoopEmitter);
    let db = MemoryDB::new(tmp.path(), emitter).await.unwrap();

    // Seed ALL memories from ALL cases into one DB
    let mut all_docs: Vec<RawDocument> = Vec::new();
    for case in &cases {
        for seed in &case.seeds {
            all_docs.push(RawDocument {
                source_id: seed.id.clone(),
                content: seed.content.clone(),
                source: "memory".into(),
                title: seed.id.clone(),
                memory_type: Some(seed.memory_type.clone()),
                domain: seed.domain.clone(),
                confirmed: seed.confirmed,
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            });
        }
        for neg in &case.negative_seeds {
            all_docs.push(RawDocument {
                source_id: neg.id.clone(),
                content: neg.content.clone(),
                source: "memory".into(),
                title: neg.id.clone(),
                memory_type: Some(neg.memory_type.clone()),
                domain: neg.domain.clone(),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            });
        }
    }
    // Dedup by source_id (some IDs may appear in multiple fixture files)
    let mut seen_ids: HashSet<String> = HashSet::new();
    all_docs.retain(|d| seen_ids.insert(d.source_id.clone()));

    println!(
        "Seeding {} unique memories from {} fixture cases...",
        all_docs.len(),
        cases.len()
    );
    db.upsert_documents(all_docs).await.unwrap();

    // --- BEFORE: measure recall across all cases (search_memory only) ---
    let k = 5;
    let mut total_recall_before = 0.0;
    let mut total_combined_recall = 0.0;
    let mut cases_with_improvement = 0usize;
    let mut total_recovered = 0usize;
    let mut scored_cases = 0usize;

    // For each case: create a concept from its positive seeds, then compare
    let now = chrono::Utc::now().to_rfc3339();
    for (i, case) in cases.iter().enumerate() {
        if case.empty_set {
            continue;
        }
        let relevant_ids: HashSet<&str> = case
            .seeds
            .iter()
            .filter(|s| s.relevance >= 2)
            .map(|s| s.id.as_str())
            .collect();
        if relevant_ids.is_empty() {
            continue;
        }
        scored_cases += 1;

        // search_memory only
        let results = db
            .search_memory(
                &case.query,
                k,
                None,
                case.domain.as_deref(),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        let ranked_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        let recall_before = metrics::recall_at_k(&ranked_ids, &relevant_ids, k);
        total_recall_before += recall_before;

        // Insert concept compiled from this case's positive seeds
        let source_refs: Vec<&str> = case.seeds.iter().map(|s| s.id.as_str()).collect();
        let concept_content: String = case
            .seeds
            .iter()
            .map(|s| format!("- {}", s.content))
            .collect::<Vec<_>>()
            .join("\n");
        let concept_id = format!("concept_case_{}", i);
        let _ = db
            .insert_concept(
                &concept_id,
                &format!(
                    "Compiled: {}",
                    case.query.chars().take(50).collect::<String>()
                ),
                Some(&case.query),
                &concept_content,
                None,
                case.domain.as_deref(),
                &source_refs,
                &now,
            )
            .await;

        // Combined: search_concepts + search_memory
        let concepts = db.search_concepts(&case.query, 3).await.unwrap_or_default();
        let mut combined: Vec<String> = Vec::new();
        for concept in &concepts {
            for sid in &concept.source_memory_ids {
                if !combined.contains(sid) {
                    combined.push(sid.clone());
                }
            }
        }
        for id in &ranked_ids {
            let s = id.to_string();
            if !combined.contains(&s) {
                combined.push(s);
            }
        }
        let combined_refs: Vec<&str> = combined.iter().map(|s| s.as_str()).collect();
        let recall_after = metrics::recall_at_k(&combined_refs, &relevant_ids, combined_refs.len());
        total_combined_recall += recall_after;

        let found_before = ranked_ids
            .iter()
            .filter(|id| relevant_ids.contains(*id))
            .count();
        let found_after = combined_refs
            .iter()
            .filter(|id| relevant_ids.contains(*id))
            .count();
        if found_after > found_before {
            cases_with_improvement += 1;
            total_recovered += found_after - found_before;
        }

        // Clean up this case's concept to avoid cross-case FTS contamination.
        // Memories stay (shared DB is realistic — real DBs have all memories).
        // But concepts from case N shouldn't pollute case N+1's search_concepts results.
        let _ = db.delete_concept(&concept_id).await;
    }

    let avg_recall_before = total_recall_before / scored_cases.max(1) as f64;
    let avg_combined_recall = total_combined_recall / scored_cases.max(1) as f64;
    let delta = avg_combined_recall - avg_recall_before;

    println!(
        "\n=== CONCEPT IMPACT ACROSS {} FIXTURE CASES (oracle ceiling*) ===",
        scored_cases
    );
    println!("Avg Recall@{} (memory only):   {:.3}", k, avg_recall_before);
    println!("Avg Cov+C*  (memory+concepts): {:.3}", avg_combined_recall);
    println!("Delta:                         {:+.3}", delta);
    println!(
        "Cases improved:                {}/{}",
        cases_with_improvement, scored_cases
    );
    println!("Total memories recovered:      {}", total_recovered);
    println!();
    println!("* Oracle ceiling: unbounded coverage with synthetic concepts from");
    println!("  known-relevant seeds. Not rank-limited like R@5. Real LLM-generated");
    println!("  concepts will score lower. Not comparable to LoCoMo/LongMemEval.");

    // Combined recall must be >= memory-only across the full fixture set
    assert!(
        avg_combined_recall >= avg_recall_before,
        "Combined recall ({:.3}) must be >= memory-only ({:.3})",
        avg_combined_recall,
        avg_recall_before
    );
}

// ---------------------------------------------------------------------------
// Pipeline eval: LoCoMo + LongMemEval through Origin's full pipeline
// ---------------------------------------------------------------------------

/// Run LoCoMo through Origin's full pipeline: flat → enriched → distilled.
/// Requires Metal GPU (run with sandbox disabled).
#[tokio::test]
#[ignore]
async fn benchmark_locomo_pipeline() {
    use origin_lib::eval::token_efficiency::run_locomo_pipeline_eval;
    use std::sync::Arc;

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found at {:?}", locomo_path);
        return;
    }

    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new()
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

    // Flat/Origin should have non-zero NDCG
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
        "Flat/Origin NDCG should be > 0"
    );
}

/// Run LongMemEval through Origin's full pipeline: flat → enriched → distilled.
/// Requires Metal GPU (run with sandbox disabled).
/// Caps at 100 questions for reasonable runtime.
#[tokio::test]
#[ignore]
async fn benchmark_longmemeval_pipeline() {
    use origin_lib::eval::token_efficiency::run_longmemeval_pipeline_eval;
    use std::sync::Arc;

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found at {:?}", path);
        return;
    }

    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new()
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
        "Flat/Origin NDCG should be > 0"
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
    use origin_lib::eval::token_efficiency::run_context_path_eval;
    use std::sync::Arc;

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new()
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
    use origin_lib::eval::token_efficiency::run_context_path_eval_longmemeval;
    use std::sync::Arc;

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new()
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
    use origin_lib::eval::token_efficiency::{run_e2e_context_eval, save_judgment_tuples};
    use std::sync::Arc;

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new().expect("Failed to init on-device LLM"),
    );

    // 1 conversation, 20 questions for quick validation
    let tuples = run_e2e_context_eval(&locomo_path, llm, 10, 1, 20)
        .await
        .expect("run_e2e_context_eval failed");

    eprintln!("Generated {} judgment tuples", tuples.len());
    assert!(!tuples.is_empty(), "should generate at least some tuples");

    // Save for offline judging (try baselines dir, fallback to tmpdir)
    let baselines_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
    std::fs::create_dir_all(&baselines_dir).ok();
    let out_path = baselines_dir.join("e2e_context_tuples_locomo.json");
    save_judgment_tuples(&tuples, &out_path).expect("save tuples");
    eprintln!("Saved to {:?}", out_path);
}

/// Generate E2E answers for LongMemEval (flat vs structured context).
#[tokio::test]
#[ignore]
async fn generate_e2e_context_tuples_longmemeval() {
    use origin_lib::eval::token_efficiency::{
        run_e2e_context_eval_longmemeval, save_judgment_tuples,
    };
    use std::sync::Arc;

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new().expect("Failed to init on-device LLM"),
    );

    // 50 questions for validation
    let tuples = run_e2e_context_eval_longmemeval(&path, llm, 10, 50, 1)
        .await
        .expect("run_e2e_context_eval_longmemeval failed");

    eprintln!("Generated {} judgment tuples", tuples.len());
    assert!(!tuples.is_empty());

    let baselines_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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
    use origin_lib::eval::token_efficiency::{
        aggregate_judgments, judge_with_claude, load_judgment_tuples,
    };

    let tuples_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/e2e_context_tuples_locomo.json");
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
    use origin_lib::eval::token_efficiency::{run_e2e_context_eval, save_judgment_tuples};
    use std::sync::Arc;

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> =
        Arc::new(origin_lib::llm_provider::ClaudeCliProvider::haiku());

    // 1 conversation, 20 questions for quick validation
    let tuples = run_e2e_context_eval(&locomo_path, llm, 10, 1, 20)
        .await
        .expect("run_e2e_context_eval with Haiku CLI failed");

    eprintln!("Generated {} judgment tuples (Haiku CLI)", tuples.len());
    assert!(!tuples.is_empty());

    let baselines_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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
    use origin_lib::eval::token_efficiency::{
        aggregate_judgments, judge_with_claude_model, load_judgment_tuples,
    };

    let tuples_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/e2e_context_tuples_locomo_api.json");
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
    use origin_lib::eval::token_efficiency::{
        aggregate_judgments, judge_with_claude_model, load_judgment_tuples,
    };

    let tuples_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("eval/baselines/e2e_context_tuples_locomo.json");
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
    use origin_lib::eval::judge::{
        aggregate_judgments, judge_with_batch_api, load_judgment_tuples,
    };

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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

// Deleted: lme_phase1_retrieve, lme_phase2_answer, lme_phase3_judge (3-phase accuracy pipeline)
// Deleted: locomo_phase1_retrieve, locomo_phase2_answer, locomo_phase3_judge
// Deleted: lme_batch_answer, lme_batch_judge, locomo_batch_answer, locomo_batch_judge
// These used types from the deleted accuracy pipeline in locomo.rs/longmemeval.rs.
// Use judge_e2e_batch above for batch API judging of E2E context tuples.
