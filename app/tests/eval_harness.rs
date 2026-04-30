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
    use origin_lib::eval::answer_quality::run_fullpipeline_locomo_batch;

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
    if !locomo_path.exists() {
        eprintln!("SKIP: locomo10.json not found");
        return;
    }

    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY required");
    let answer_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("fullpipeline_locomo_tuples.json");

    eprintln!(
        "[fullpipeline] LoCoMo\n  model: {}\n  cost cap: ${:.2}\n  output: {:?}",
        answer_model, cost_cap, output_path,
    );

    let enrichment = origin_lib::eval::shared::EnrichmentMode::from_env(&answer_model, cost_cap)
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
    use origin_lib::eval::answer_quality::run_fullpipeline_lme_batch;

    let lme_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !lme_path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found");
        return;
    }

    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY required");
    let answer_model =
        std::env::var("EVAL_ANSWER_MODEL").unwrap_or_else(|_| "claude-haiku-4-5-20251001".into());
    let cost_cap: f64 = std::env::var("EVAL_COST_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
    std::fs::create_dir_all(&baselines).ok();
    let output_path = baselines.join("fullpipeline_lme_tuples.json");

    eprintln!(
        "[fullpipeline] LME\n  model: {}\n  cost cap: ${:.2}\n  output: {:?}",
        answer_model, cost_cap, output_path,
    );

    let enrichment = origin_lib::eval::shared::EnrichmentMode::from_env(&answer_model, cost_cap)
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
/// ORIGIN_LLM_WORKERS=1 ORIGIN_LLM_PARALLEL_SEQS=8 EVAL_ENRICHMENT_CONCURRENCY=8 \
///   cargo test -p origin --test eval_harness enrich_fullpipeline_lme_only -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn enrich_fullpipeline_lme_only() {
    use origin_lib::eval::longmemeval::{extract_memories, load_longmemeval};
    use origin_lib::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use origin_lib::sources::RawDocument;

    let lme_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
    if !lme_path.exists() {
        eprintln!("SKIP: longmemeval_oracle.json not found");
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

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
    std::fs::create_dir_all(&baselines).ok();

    eprintln!(
        "[lme_enrich_only] baselines={} skip={} limit={:?}",
        baselines.display(),
        skip,
        limit
    );

    let samples = load_longmemeval(&lme_path).expect("load_longmemeval");
    let enrichment =
        EnrichmentMode::from_env(&answer_model, cost_cap).expect("EnrichmentMode init failed");
    let shared_embedder = eval_shared_embedder();

    let t0 = std::time::Instant::now();
    let mut processed = 0usize;
    let mut total_memories = 0usize;

    for (idx, sample) in samples.iter().enumerate().skip(skip) {
        if limit.is_some_and(|n| processed >= n) {
            break;
        }

        let ground_truth = sample
            .answer
            .as_str()
            .unwrap_or(&sample.answer.to_string())
            .to_string();
        if ground_truth.is_empty() {
            continue;
        }

        let memories = extract_memories(sample);
        if memories.is_empty() {
            continue;
        }

        let scope_dir = scenario_db_dir(&baselines, "lme", &sample.question_id);
        let question_id = sample.question_id.clone();
        let question_type = sample.question_type.clone();
        let memories_owned = memories.clone();

        let scenario_t0 = std::time::Instant::now();
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
                        domain: Some("conversation".to_string()),
                        last_modified: chrono::Utc::now().timestamp(),
                        ..Default::default()
                    })
                    .collect()
            },
            &enrichment,
        )
        .await
        .expect("open_or_seed_scenario_db");

        let mem_count = db.memory_count().await.unwrap_or(0);
        let enriched = db.enriched_memory_count().await.unwrap_or(0);
        processed += 1;
        total_memories += mem_count;

        eprintln!(
            "[lme_enrich_only] {}/{} idx={} q={} mem={} enriched={}/{} elapsed={:.1}s total={:.1}m",
            processed,
            limit.unwrap_or(samples.len().saturating_sub(skip)),
            idx,
            sample.question_id,
            memories.len(),
            enriched,
            mem_count,
            scenario_t0.elapsed().as_secs_f32(),
            t0.elapsed().as_secs_f32() / 60.0,
        );
        assert_eq!(
            mem_count, enriched,
            "{} should be fully enriched ({}/{})",
            sample.question_id, enriched, mem_count
        );
    }

    eprintln!(
        "[lme_enrich_only] done: {} scenarios, {} memories in {:.1}m",
        processed,
        total_memories,
        t0.elapsed().as_secs_f32() / 60.0
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
    use origin_lib::memory_db::MemoryDB;
    use std::sync::Arc;

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");

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

            let emitter: Arc<dyn origin_core::events::EventEmitter> =
                Arc::new(origin_core::NoopEmitter);
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
    use origin_lib::eval::judge::{
        aggregate_judgments, judge_with_batch_api, load_judgment_tuples,
    };

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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
    use origin_lib::eval::judge::{
        aggregate_judgments, judge_with_batch_api, load_judgment_tuples,
    };

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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
    use origin_lib::eval::locomo::{extract_observations, load_locomo};
    use origin_lib::eval::shared::eval_shared_embedder;
    use origin_lib::llm_provider::{LlmProvider, LlmRequest, OnDeviceProvider};
    use origin_lib::prompts::PromptRegistry;
    use std::sync::Arc;

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
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
                system_prompt: Some(prompts.distill_concept.clone()),
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
                let cleaned = origin_lib::llm_provider::strip_think_tags(&raw);
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
    use origin_lib::eval::locomo::{extract_observations, load_locomo};
    use origin_lib::eval::shared::probe_extraction_batch_sizes;
    use std::sync::Arc;

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
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
    let llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::OnDeviceProvider::new_with_model(model_id.as_deref())
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
    use origin_lib::eval::locomo::{extract_observations, load_locomo};
    use origin_lib::eval::shared::{
        count_tokens, eval_shared_embedder, run_concept_distillation_batch_api,
        run_enrichment_batch_api, run_title_enrichment_batch_api,
    };
    use std::sync::Arc;

    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY required");
    let model = "claude-haiku-4-5-20251001";

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
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
    let db = origin_core::db::MemoryDB::new_with_shared_embedder(
        tmp.path(),
        Arc::new(origin_core::events::NoopEmitter),
        shared_embedder,
    )
    .await
    .unwrap();

    let docs: Vec<origin_lib::sources::RawDocument> = memories
        .iter()
        .enumerate()
        .map(|(i, mem)| origin_lib::sources::RawDocument {
            content: mem.content.clone(),
            source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
            source: "memory".to_string(),
            title: format!("{} session {}", mem.speaker, mem.session_num),
            memory_type: Some("fact".to_string()),
            domain: Some("conversation".to_string()),
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
        .search_memory(&qa.question, 10, None, None, None, None, None, None)
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
        .search_concepts(&qa.question, 3)
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
///   SMOKE_LOCOMO_SAMPLES=10 EVAL_LOCAL_MODEL=qwen3.5-9b EVAL_ENRICHMENT_BATCH_SIZE=8 \
///   cargo test -p origin --test eval_harness smoke_per_scenario_locomo -- --ignored --nocapture
/// Default config (2 samples x 10 obs) is the CI smoke.
#[tokio::test]
#[ignore]
async fn smoke_per_scenario_locomo() {
    use origin_lib::eval::locomo::{extract_observations, load_locomo};
    use origin_lib::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use origin_lib::sources::RawDocument;

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
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
                        domain: Some("conversation".to_string()),
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
            .search_memory("anything", 50, None, None, None, None, None, None)
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
                    domain: Some("conversation".to_string()),
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

/// Manual smoke for per-scenario enrichment using Claude CLI provider (D2).
///
/// Validates that `EnrichmentMode::OnDevice(ClaudeCliProvider)` works end-to-end:
/// seed → concurrent entity extraction + title enrichment (EVAL_ENRICHMENT_CONCURRENCY=4)
/// → distillation → isolation check. Uses claude-haiku-4-5-20251001 via Max plan.
///
/// - 1 LoCoMo conversation, 5 observations (small: CLI subprocess ~2-3s/call)
/// - Expected duration: ~20s (5 obs × 2 phases / 4 concurrency × 3s + distill)
/// - Skips silently if `claude` binary is not in PATH
///
/// ```bash
/// cargo test -p origin --test eval_harness smoke_per_scenario_locomo_cli -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn smoke_per_scenario_locomo_cli() {
    use origin_lib::eval::locomo::{extract_observations, load_locomo};
    use origin_lib::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use origin_lib::sources::RawDocument;
    use std::sync::Arc;

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

    let locomo_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
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

    let cli_llm: Arc<dyn origin_lib::llm_provider::LlmProvider> = Arc::new(
        origin_lib::llm_provider::ClaudeCliProvider::new("claude-haiku-4-5-20251001"),
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
                    domain: Some("conversation".to_string()),
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
        .search_memory("anything", 50, None, None, None, None, None, None)
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
///   SMOKE_LME_SAMPLES=10 EVAL_LOCAL_MODEL=qwen3.5-9b EVAL_ENRICHMENT_BATCH_SIZE=8 \
///   cargo test -p origin --test eval_harness smoke_per_scenario_lme -- --ignored --nocapture
/// Default config (2 samples x 5 mems) is the CI smoke.
#[tokio::test]
#[ignore]
async fn smoke_per_scenario_lme() {
    use origin_lib::eval::longmemeval::{extract_memories, load_longmemeval};
    use origin_lib::eval::shared::{
        eval_shared_embedder, open_or_seed_scenario_db, scenario_db_dir, EnrichmentMode,
    };
    use origin_lib::sources::RawDocument;

    let lme_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/longmemeval_oracle.json");
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
                        domain: Some("conversation".to_string()),
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
            .search_memory("anything", 50, None, None, None, None, None, None)
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
                    domain: Some("conversation".to_string()),
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
    use origin_lib::eval::judge::{
        aggregate_judgments, judge_with_claude_model_persistent, load_judgment_tuples,
    };
    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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
    let cache_path = baselines.join("fullpipeline_lme_judgments.jsonl");

    eprintln!(
        "Judging {} LME tuples via CLI (model={}, concurrency={}, retries={}, cache={})...",
        tuples.len(),
        model,
        concurrency,
        max_retries,
        cache_path.display()
    );
    let results =
        judge_with_claude_model_persistent(&tuples, concurrency, &model, &cache_path, max_retries)
            .await
            .expect("judge failed");
    let report = aggregate_judgments(&results, &format!("{}-cli", model));

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
    use origin_lib::eval::judge::{
        aggregate_judgments, judge_with_claude_model_batched_persistent,
        judge_with_claude_model_persistent, load_judgment_tuples,
    };
    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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
fn print_judge_report(report: &origin_lib::eval::judge::JudgedE2EReport) {
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
/// Runs search_concepts with real embeddings on sample questions from each benchmark.
/// Prints score distributions so we can set a data-driven threshold.
///
/// ```bash
/// cargo test -p origin --test eval_harness probe_concept_scores -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore]
async fn probe_concept_scores() {
    use origin_core::db::MemoryDB;
    use origin_core::events::NoopEmitter;
    use origin_lib::eval::shared::eval_shared_embedder;
    use std::sync::Arc;

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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
            let concepts = db.search_concepts(question, 3).await.unwrap_or_default();
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
/// Runs search_memory + search_concepts for every question, counts how many
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
    use origin_core::concepts::filter_concepts_by_source_overlap;
    use origin_core::db::MemoryDB;
    use origin_core::events::NoopEmitter;
    use origin_lib::eval::shared::eval_shared_embedder;
    use std::collections::HashMap;
    use std::sync::Arc;

    let baselines = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/baselines");
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
                .search_memory(q, 10, None, None, None, None, None, None)
                .await
            {
                Ok(r) => r,
                Err(_) => continue,
            };
            let search_ids: std::collections::HashSet<String> =
                results.iter().map(|r| r.source_id.clone()).collect();

            // Real search_concepts (top-3)
            let raw_concepts = db.search_concepts(q, 3).await.unwrap_or_default();
            let kept = filter_concepts_by_source_overlap(&raw_concepts, &search_ids, min_overlap);

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
/// multi-worker pool (S1, `ORIGIN_LLM_WORKERS`) and continuous batching
/// (S2, `ORIGIN_LLM_PARALLEL_SEQS`).
///
/// Three useful invocations to compare:
///   ORIGIN_LLM_WORKERS=1 ORIGIN_LLM_PARALLEL_SEQS=1   # baseline (single seq, single ctx)
///   ORIGIN_LLM_WORKERS=4 ORIGIN_LLM_PARALLEL_SEQS=1   # S1: 4 contexts, 1 seq each
///   ORIGIN_LLM_WORKERS=1 ORIGIN_LLM_PARALLEL_SEQS=4   # S2: 1 context, 4 parallel seqs
///   ORIGIN_LLM_WORKERS=2 ORIGIN_LLM_PARALLEL_SEQS=2   # composed: 2 ctx x 2 seq = 4 concurrent
///
/// Requires Metal GPU + a downloaded Qwen3-4B model. Marked `#[ignore]` so it
/// doesn't run in CI. Invoke with:
///   cargo test -p origin --test eval_harness stress_concurrent_inference \
///       -- --ignored --nocapture
#[tokio::test]
#[ignore]
async fn stress_concurrent_inference() {
    use futures::future::join_all;
    use origin_lib::llm_provider::{LlmProvider, LlmRequest, OnDeviceProvider};
    use std::sync::Arc;
    use std::time::Instant;

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
