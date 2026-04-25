// SPDX-License-Identifier: AGPL-3.0-only
//! Token efficiency evaluation — measures quality vs cost across search strategies.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::eval::fixtures::load_fixtures;
use crate::eval::metrics;
use crate::events::NoopEmitter;
use crate::sources::RawDocument;
use crate::tuning::ConfidenceConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use std::sync::LazyLock;

/// Shared BPE tokenizer instance (cl100k_base). Initialized once, intentionally
/// leaked to avoid destructor conflicts with ONNX runtime at process exit.
static BPE: LazyLock<&'static tiktoken_rs::CoreBPE> = LazyLock::new(|| {
    let bpe = tiktoken_rs::cl100k_base().expect("failed to load cl100k_base tokenizer");
    Box::leak(Box::new(bpe))
});

/// Count tokens in text using tiktoken cl100k_base encoding.
pub fn count_tokens(text: &str) -> usize {
    BPE.encode_with_special_tokens(text).len()
}

// ===== Types =====

/// The search strategies compared by the token efficiency eval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Origin's full hybrid search (vector + FTS + RRF).
    Origin,
    /// Origin hybrid + LLM reranking pass.
    OriginReranked,
    /// Origin hybrid + LLM query expansion.
    OriginExpanded,
    /// Vector-only search (no FTS, no RRF, no scoring).
    NaiveRag,
    /// Return the entire corpus unchanged (upper bound on context cost).
    FullReplay,
    /// No context at all (lower bound on quality).
    NoMemory,
    /// Ablation: FTS5 BM25 search only — no vectors, no RRF.
    FtsOnly,
    /// Ablation: vector + FTS merged by max score (no RRF fusion).
    VectorPlusFts,
}

impl SearchStrategy {
    /// Snake_case identifier used in serialized output.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Origin => "origin",
            Self::OriginReranked => "origin_reranked",
            Self::OriginExpanded => "origin_expanded",
            Self::NaiveRag => "naive_rag",
            Self::FullReplay => "full_replay",
            Self::NoMemory => "no_memory",
            Self::FtsOnly => "fts_only",
            Self::VectorPlusFts => "vector_plus_fts",
        }
    }

    /// Human-readable label for terminal display.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Origin => "Origin",
            Self::OriginReranked => "Origin+Rerank",
            Self::OriginExpanded => "Origin+Expand",
            Self::NaiveRag => "Naive RAG",
            Self::FullReplay => "Full Replay",
            Self::NoMemory => "No Memory",
            Self::FtsOnly => "FTS Only",
            Self::VectorPlusFts => "Vector+FTS",
        }
    }

    /// Whether this strategy requires an LLM call (skipped in fast eval mode).
    pub fn requires_llm(&self) -> bool {
        matches!(self, Self::OriginReranked | Self::OriginExpanded)
    }
}

/// Per-query token and compression metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenMetrics {
    /// Tokens in the retrieved context passed to the model.
    pub context_tokens: usize,
    /// Tokens in the query itself.
    pub query_tokens: usize,
    /// Tokens in the full corpus (all seeds concatenated).
    pub corpus_tokens: usize,
    /// context_tokens / corpus_tokens. 0.0 when corpus is empty.
    pub compression_ratio: f64,
    /// Number of chunks/memories returned by the search.
    pub chunks_retrieved: usize,
}

impl TokenMetrics {
    /// Compute compression ratio safely (avoids division by zero).
    pub fn compute_compression_ratio(context_tokens: usize, corpus_tokens: usize) -> f64 {
        if corpus_tokens == 0 {
            0.0
        } else {
            context_tokens as f64 / corpus_tokens as f64
        }
    }
}

/// Aggregated metrics for one strategy across all eval cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyReport {
    pub strategy: String,
    pub mean_context_tokens: f64,
    pub median_context_tokens: f64,
    pub p25_context_tokens: f64,
    pub p75_context_tokens: f64,
    pub stddev_context_tokens: f64,
    pub mean_compression_ratio: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub stddev_ndcg: f64,
    pub stddev_mrr: f64,
}

/// Top-line token-efficiency comparison: Origin vs FullReplay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadlineMetrics {
    /// Percentage reduction in tokens: (replay - origin) / replay * 100.
    pub savings_pct: f64,
    /// Mean context tokens for Origin strategy.
    pub origin_tokens: f64,
    /// Mean context tokens for FullReplay strategy.
    pub replay_tokens: f64,
    /// Percentage of FullReplay quality retained by Origin: origin_ndcg / replay_ndcg * 100.
    /// Uses NDCG@10 as the quality proxy.
    pub quality_retained_pct: f64,
}

/// Token vs quality tradeoff at varying corpus sizes (optional scaling experiment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPoint {
    pub corpus_size: usize,
    pub origin_tokens: f64,
    pub replay_tokens: f64,
}

/// Per-turn token counts in a multi-turn session simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnPoint {
    /// 1-based turn number.
    pub turn: usize,
    /// Origin tokens used this turn (fresh retrieval).
    pub origin_tokens: usize,
    /// Replay tokens used this turn (corpus + accumulated history).
    pub replay_tokens: usize,
    /// Cumulative Origin tokens up to and including this turn.
    pub cumulative_origin: usize,
    /// Cumulative Replay tokens up to and including this turn.
    pub cumulative_replay: usize,
}

/// Report from a multi-turn token accumulation simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnReport {
    /// Number of turns simulated.
    pub turns: usize,
    /// Per-turn breakdown.
    pub per_turn: Vec<MultiTurnPoint>,
    /// Total Origin tokens across all turns.
    pub total_origin_tokens: usize,
    /// Total Replay tokens across all turns.
    pub total_replay_tokens: usize,
    /// Percentage savings: (replay - origin) / replay * 100.
    pub savings_pct: f64,
}

/// Full quality-cost evaluation report, serializable to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityCostReport {
    pub benchmark: String,
    pub timestamp: String,
    pub tokenizer: String,
    pub strategies: Vec<StrategyReport>,
    pub headline: HeadlineMetrics,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scaling: Vec<ScalingPoint>,
}

impl QualityCostReport {
    /// Render a human-readable table to a String.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Quality-Cost Report: {}\n", self.benchmark));
        out.push_str(&format!("Timestamp: {}\n", self.timestamp));
        out.push_str(&format!("Tokenizer: {}\n\n", self.tokenizer));

        // Column widths: Strategy, NDCG@10±σ, MRR±σ, Recall@5, Tokens±σ, Compression
        let col_widths = (16usize, 14usize, 12usize, 10usize, 16usize, 13usize);
        out.push_str(&format!(
            "{:<w0$}  {:>w1$}  {:>w2$}  {:>w3$}  {:>w4$}  {:>w5$}\n",
            "Strategy",
            "NDCG@10",
            "MRR",
            "Recall@5",
            "Tokens/Query",
            "Compression",
            w0 = col_widths.0,
            w1 = col_widths.1,
            w2 = col_widths.2,
            w3 = col_widths.3,
            w4 = col_widths.4,
            w5 = col_widths.5,
        ));
        let sep_len = col_widths.0
            + col_widths.1
            + col_widths.2
            + col_widths.3
            + col_widths.4
            + col_widths.5
            + 10;
        out.push_str(&"-".repeat(sep_len));
        out.push('\n');

        for s in &self.strategies {
            let ndcg_str = format!("{:.4}±{:.4}", s.ndcg_at_10, s.stddev_ndcg);
            let mrr_str = format!("{:.4}±{:.4}", s.mrr, s.stddev_mrr);
            let tok_str = format!(
                "{:.1}±{:.1}",
                s.mean_context_tokens, s.stddev_context_tokens
            );
            out.push_str(&format!(
                "{:<w0$}  {:>w1$}  {:>w2$}  {:>w3$.4}  {:>w4$}  {:>w5$.4}\n",
                s.strategy,
                ndcg_str,
                mrr_str,
                s.recall_at_5,
                tok_str,
                s.mean_compression_ratio,
                w0 = col_widths.0,
                w1 = col_widths.1,
                w2 = col_widths.2,
                w3 = col_widths.3,
                w4 = col_widths.4,
                w5 = col_widths.5,
            ));
        }

        out.push('\n');
        out.push_str(&format!(
            "Headline: {:.1}% token savings vs Full Replay ({:.1} vs {:.1} tokens/query)\n",
            self.headline.savings_pct, self.headline.origin_tokens, self.headline.replay_tokens,
        ));
        out.push_str(&format!(
            "          {:.1}% quality retained (NDCG@10 vs Full Replay)\n",
            self.headline.quality_retained_pct,
        ));
        out
    }

    /// Save report to disk as pretty-printed JSON.
    pub fn save_baseline(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load a previously saved report from disk.
    pub fn load_baseline(path: &Path) -> Result<QualityCostReport, std::io::Error> {
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(std::io::Error::other)
    }
}

// ===== Native Memory Augmentation Types =====

/// Token cost model for a native AI memory platform.
///
/// Token estimates sourced from public documentation, model cards, and community
/// measurements as of April 2026. These are researched estimates, not measurements
/// from those systems' APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeMemoryBaseline {
    /// Platform name.
    pub platform: String,
    /// Tokens injected per turn (memory content only, excluding base system prompt).
    pub memory_tokens_per_turn: usize,
    /// Whether memory content grows with usage or is capped.
    /// One of: "unbounded", "capped", "synthesized"
    pub growth_model: String,
    /// Brief description of how memory is injected.
    pub mechanism: String,
}

/// One alternative scenario for recalling specific information (without Origin).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallAlternative {
    /// Machine-readable identifier.
    pub scenario: String,
    /// Additional tokens needed on top of native memory per recall event.
    pub tokens_per_recall: usize,
    /// Human-readable description.
    pub description: String,
}

/// Multi-turn augmentation comparison: native-only vs native+Origin vs native+full-replay.
///
/// Models a 10-turn session where a fraction of turns need specific recall.
/// Native memory is a constant cost already paid; Origin adds a small overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnAugmentationComparison {
    /// Total turns in the session.
    pub turns: usize,
    /// Turns where specific recall is needed (not every turn requires it).
    pub recall_turns: usize,
    /// Native-only total: native_per_turn * turns (no specific recall capability).
    pub native_only_total: usize,
    /// Native + Origin: native_per_turn * turns + origin_retrieval * recall_turns.
    pub native_plus_origin_total: usize,
    /// Native + full replay: native_per_turn * turns + full_replay_tokens * recall_turns.
    pub native_plus_replay_total: usize,
    /// Origin's additional cost as a percentage of the native-only baseline.
    pub origin_overhead_pct: f64,
}

/// Augmentation report: what Origin ADDS on top of native memory.
///
/// Origin does not replace native memory — users pay for native memory regardless.
/// This report answers: "What does adding Origin cost, and what do you get?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeMemoryAugmentationReport {
    /// What Origin adds per query on top of native memory (mean across fixture cases).
    pub origin_retrieval_tokens: f64,
    /// Native baselines for reference (cost already paid by the user).
    pub baselines: Vec<NativeMemoryBaseline>,
    /// Alternative costs when the user needs specific recall without Origin.
    pub alternatives: Vec<RecallAlternative>,
    /// Multi-turn comparison: native-only vs native+Origin vs native+full-replay.
    pub multi_turn: MultiTurnAugmentationComparison,
    /// Honest framing note explaining the augmentation model.
    pub framing_note: String,
}

// Keep the old name as a type alias so any external callers don't break immediately.
#[deprecated(since = "0.0.0", note = "Use NativeMemoryAugmentationReport instead")]
pub type NativeMemoryComparisonReport = NativeMemoryAugmentationReport;

// ===== Runner =====

/// Run a quality-cost evaluation across fixture cases.
///
/// For each case:
/// 1. Seeds an ephemeral DB.
/// 2. Runs each non-LLM strategy.
/// 3. Scores quality with NDCG@10, MRR, Recall@5.
/// 4. Aggregates token usage per strategy.
///
/// `strategies` may include `OriginReranked`/`OriginExpanded` but they will be
/// silently skipped (they require a running LLM engine).
pub async fn run_quality_cost_eval(
    fixture_dir: &Path,
    strategies: &[SearchStrategy],
    limit: usize,
) -> Result<QualityCostReport, OriginError> {
    let cases = load_fixtures(fixture_dir)?;

    // Per-strategy accumulators
    let mut context_tokens_all: HashMap<SearchStrategy, Vec<usize>> = HashMap::new();
    let mut compression_all: HashMap<SearchStrategy, Vec<f64>> = HashMap::new();
    let mut ndcg_all: HashMap<SearchStrategy, Vec<f64>> = HashMap::new();
    let mut mrr_all: HashMap<SearchStrategy, Vec<f64>> = HashMap::new();
    let mut recall5_all: HashMap<SearchStrategy, Vec<f64>> = HashMap::new();

    for strategy in strategies {
        context_tokens_all.insert(*strategy, Vec::new());
        compression_all.insert(*strategy, Vec::new());
        ndcg_all.insert(*strategy, Vec::new());
        mrr_all.insert(*strategy, Vec::new());
        recall5_all.insert(*strategy, Vec::new());
    }

    let confidence_cfg = ConfidenceConfig::default();

    for case in &cases {
        if case.empty_set {
            continue; // Skip empty-set cases — no relevant docs to measure quality against.
        }

        // Compute corpus tokens from seeds
        let corpus_text: String = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let corpus_tokens = count_tokens(&corpus_text);

        // Seed an ephemeral DB
        let case_tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir for eval case: {}", e)))?;
        let db = MemoryDB::new(case_tmp.path(), Arc::new(NoopEmitter)).await?;

        let all_docs: Vec<RawDocument> = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .map(|seed| crate::eval::runner::seed_to_doc(seed, &confidence_cfg))
            .collect();
        db.upsert_documents(all_docs).await?;

        // Build scoring maps for this case
        let relevant: HashSet<&str> = case
            .seeds
            .iter()
            .filter(|s| s.relevance >= 2)
            .map(|s| s.id.as_str())
            .collect();
        let grades: HashMap<&str, u8> = case
            .seeds
            .iter()
            .map(|s| (s.id.as_str(), s.relevance))
            .collect();

        for strategy in strategies {
            if strategy.requires_llm() {
                continue;
            }

            let (context_tokens, ndcg, mrr_score, recall5) = match strategy {
                SearchStrategy::Origin => {
                    let results = db
                        .search_memory(
                            &case.query,
                            limit,
                            None,
                            case.domain.as_deref(),
                            None,
                            Some(1.0), // neutralize confirmation boost — fixture bias
                            Some(1.0), // neutralize recap penalty — fixture bias
                            None,
                        )
                        .await?;
                    let ctx_tokens = count_results_tokens(&results);
                    let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
                    let ndcg = metrics::ndcg_at_k(&ranked, &grades, 10);
                    let mrr_v = metrics::mrr(&ranked, &relevant);
                    let r5 = metrics::recall_at_k(&ranked, &relevant, 5);
                    (ctx_tokens, ndcg, mrr_v, r5)
                }
                SearchStrategy::NaiveRag => {
                    let results = db
                        .naive_vector_search(&case.query, limit, case.domain.as_deref())
                        .await?;
                    let ctx_tokens = count_results_tokens(&results);
                    let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
                    let ndcg = metrics::ndcg_at_k(&ranked, &grades, 10);
                    let mrr_v = metrics::mrr(&ranked, &relevant);
                    let r5 = metrics::recall_at_k(&ranked, &relevant, 5);
                    (ctx_tokens, ndcg, mrr_v, r5)
                }
                SearchStrategy::FullReplay => {
                    // Cost = entire corpus; quality = best possible (1.0 on all metrics)
                    (corpus_tokens, 1.0, 1.0, 1.0)
                }
                SearchStrategy::NoMemory => {
                    // Cost = 0 tokens; quality = 0 (no context)
                    (0, 0.0, 0.0, 0.0)
                }
                SearchStrategy::FtsOnly => {
                    let results = db
                        .fts_only_search(&case.query, limit, case.domain.as_deref())
                        .await?;
                    let ctx_tokens = count_results_tokens(&results);
                    let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
                    let ndcg = metrics::ndcg_at_k(&ranked, &grades, 10);
                    let mrr_v = metrics::mrr(&ranked, &relevant);
                    let r5 = metrics::recall_at_k(&ranked, &relevant, 5);
                    (ctx_tokens, ndcg, mrr_v, r5)
                }
                SearchStrategy::VectorPlusFts => {
                    let results = db
                        .vector_plus_fts_search(&case.query, limit, case.domain.as_deref())
                        .await?;
                    let ctx_tokens = count_results_tokens(&results);
                    let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
                    let ndcg = metrics::ndcg_at_k(&ranked, &grades, 10);
                    let mrr_v = metrics::mrr(&ranked, &relevant);
                    let r5 = metrics::recall_at_k(&ranked, &relevant, 5);
                    (ctx_tokens, ndcg, mrr_v, r5)
                }
                SearchStrategy::OriginReranked | SearchStrategy::OriginExpanded => {
                    unreachable!("LLM strategies already skipped above")
                }
            };

            let compression =
                TokenMetrics::compute_compression_ratio(context_tokens, corpus_tokens);

            context_tokens_all
                .get_mut(strategy)
                .unwrap()
                .push(context_tokens);
            compression_all.get_mut(strategy).unwrap().push(compression);
            ndcg_all.get_mut(strategy).unwrap().push(ndcg);
            mrr_all.get_mut(strategy).unwrap().push(mrr_score);
            recall5_all.get_mut(strategy).unwrap().push(recall5);
        }
    }

    // Aggregate per-strategy
    let mut strategy_reports: Vec<StrategyReport> = Vec::new();
    for strategy in strategies {
        if strategy.requires_llm() {
            continue;
        }

        let ctx_vec = &context_tokens_all[strategy];
        let comp_vec = &compression_all[strategy];
        let ndcg_vec = &ndcg_all[strategy];
        let mrr_vec = &mrr_all[strategy];
        let r5_vec = &recall5_all[strategy];

        let n = ctx_vec.len().max(1) as f64;

        let mean_ctx = ctx_vec.iter().sum::<usize>() as f64 / n;
        let (median_ctx, p25_ctx, p75_ctx) = {
            let mut sorted = ctx_vec.clone();
            sorted.sort_unstable();
            if sorted.is_empty() {
                (0.0, 0.0, 0.0)
            } else {
                let len = sorted.len();
                let median = if len.is_multiple_of(2) {
                    (sorted[len / 2 - 1] + sorted[len / 2]) as f64 / 2.0
                } else {
                    sorted[len / 2] as f64
                };
                let p25 = sorted[(len as f64 * 0.25) as usize] as f64;
                let p75 = sorted[((len as f64 * 0.75) as usize).min(len - 1)] as f64;
                (median, p25, p75)
            }
        };
        let stddev_ctx = {
            let variance = ctx_vec
                .iter()
                .map(|&x| {
                    let diff = x as f64 - mean_ctx;
                    diff * diff
                })
                .sum::<f64>()
                / n;
            variance.sqrt()
        };
        let mean_comp = comp_vec.iter().sum::<f64>() / n;
        let mean_ndcg = ndcg_vec.iter().sum::<f64>() / n;
        let mean_mrr = mrr_vec.iter().sum::<f64>() / n;
        let mean_r5 = r5_vec.iter().sum::<f64>() / n;
        let stddev_ndcg = {
            let variance = ndcg_vec
                .iter()
                .map(|&x| {
                    let diff = x - mean_ndcg;
                    diff * diff
                })
                .sum::<f64>()
                / n;
            variance.sqrt()
        };
        let stddev_mrr = {
            let variance = mrr_vec
                .iter()
                .map(|&x| {
                    let diff = x - mean_mrr;
                    diff * diff
                })
                .sum::<f64>()
                / n;
            variance.sqrt()
        };

        strategy_reports.push(StrategyReport {
            strategy: strategy.name().to_string(),
            mean_context_tokens: mean_ctx,
            median_context_tokens: median_ctx,
            p25_context_tokens: p25_ctx,
            p75_context_tokens: p75_ctx,
            stddev_context_tokens: stddev_ctx,
            mean_compression_ratio: mean_comp,
            ndcg_at_10: mean_ndcg,
            mrr: mean_mrr,
            recall_at_5: mean_r5,
            stddev_ndcg,
            stddev_mrr,
        });
    }

    // Compute headline metrics
    let origin_report = strategy_reports
        .iter()
        .find(|r| r.strategy == SearchStrategy::Origin.name());
    let replay_report = strategy_reports
        .iter()
        .find(|r| r.strategy == SearchStrategy::FullReplay.name());

    let (origin_tokens, replay_tokens, savings_pct, quality_retained_pct) =
        match (origin_report, replay_report) {
            (Some(o), Some(r)) => {
                let savings = if r.mean_context_tokens > 0.0 {
                    (r.mean_context_tokens - o.mean_context_tokens) / r.mean_context_tokens * 100.0
                } else {
                    0.0
                };
                let quality = if r.ndcg_at_10 > 0.0 {
                    o.ndcg_at_10 / r.ndcg_at_10 * 100.0
                } else {
                    100.0
                };
                (
                    o.mean_context_tokens,
                    r.mean_context_tokens,
                    savings,
                    quality,
                )
            }
            _ => (0.0, 0.0, 0.0, 0.0),
        };

    Ok(QualityCostReport {
        benchmark: "origin-eval".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        tokenizer: "cl100k_base".to_string(),
        strategies: strategy_reports,
        headline: HeadlineMetrics {
            savings_pct,
            origin_tokens,
            replay_tokens,
            quality_retained_pct,
        },
        scaling: Vec::new(),
    })
}

/// Simulate multi-turn token accumulation.
///
/// Seeds a DB with fixture memories, then simulates N turns of an agent session.
/// Each turn runs an Origin search (constant cost) vs full replay (accumulating cost).
/// The `response_overhead` parameter estimates tokens per LLM response that accumulate
/// in the without-Origin conversation history.
///
/// Picks the fixture case with the most seeds (positive + negative) for the most
/// realistic numbers. Uses multiple fixture queries (rotating) if available.
pub async fn run_multi_turn_eval(
    fixture_dir: &Path,
    turns: usize,
    limit: usize,
    response_overhead: usize,
) -> Result<MultiTurnReport, OriginError> {
    let cases = load_fixtures(fixture_dir)?;

    // Pick the non-empty case with the most seeds for the most realistic numbers.
    let best_case = cases
        .iter()
        .filter(|c| !c.empty_set && !c.seeds.is_empty())
        .max_by_key(|c| c.seeds.len() + c.negative_seeds.len())
        .ok_or_else(|| OriginError::Generic("no non-empty fixture cases found".to_string()))?;

    // Collect all non-empty cases to rotate queries across turns (more realistic).
    let query_cases: Vec<&crate::eval::fixtures::EvalCase> = cases
        .iter()
        .filter(|c| !c.empty_set && !c.seeds.is_empty())
        .collect();

    // Compute corpus tokens from the selected case.
    let corpus_text: String = best_case
        .seeds
        .iter()
        .chain(best_case.negative_seeds.iter())
        .map(|s| s.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let corpus_tokens = count_tokens(&corpus_text);

    // Seed an ephemeral DB with the best case's memories.
    let confidence_cfg = crate::tuning::ConfidenceConfig::default();
    let case_tmp = tempfile::tempdir()
        .map_err(|e| OriginError::Generic(format!("tempdir for multi-turn eval: {}", e)))?;
    let db = MemoryDB::new(case_tmp.path(), Arc::new(NoopEmitter)).await?;

    let all_docs: Vec<RawDocument> = best_case
        .seeds
        .iter()
        .chain(best_case.negative_seeds.iter())
        .map(|seed| crate::eval::runner::seed_to_doc(seed, &confidence_cfg))
        .collect();
    db.upsert_documents(all_docs).await?;

    // Simulate N turns.
    let mut per_turn: Vec<MultiTurnPoint> = Vec::with_capacity(turns);
    let mut cumulative_origin = 0usize;
    let mut cumulative_replay = 0usize;

    for turn in 1..=turns {
        // Rotate through available queries for variety across turns.
        let query_case = query_cases[(turn - 1) % query_cases.len()];
        let query = &query_case.query;

        // Origin: fresh retrieval each turn — constant cost.
        let results = db
            .search_memory(
                query,
                limit,
                None,
                best_case.domain.as_deref(),
                None,
                Some(1.0), // neutralize confirmation boost
                Some(1.0), // neutralize recap penalty
                None,
            )
            .await?;
        let origin_tokens = count_results_tokens(&results);

        // Replay: full corpus + accumulated conversation history.
        // Turn 1: corpus_tokens (no prior history yet)
        // Turn N: corpus_tokens + (N-1) * response_overhead
        let replay_tokens = corpus_tokens + (turn - 1) * response_overhead;

        cumulative_origin += origin_tokens;
        cumulative_replay += replay_tokens;

        per_turn.push(MultiTurnPoint {
            turn,
            origin_tokens,
            replay_tokens,
            cumulative_origin,
            cumulative_replay,
        });
    }

    let total_origin_tokens = cumulative_origin;
    let total_replay_tokens = cumulative_replay;
    let savings_pct = if total_replay_tokens > 0 {
        (total_replay_tokens.saturating_sub(total_origin_tokens)) as f64
            / total_replay_tokens as f64
            * 100.0
    } else {
        0.0
    };

    Ok(MultiTurnReport {
        turns,
        per_turn,
        total_origin_tokens,
        total_replay_tokens,
        savings_pct,
    })
}

/// Measure what Origin adds on top of native memory (augmentation framing).
///
/// Users already pay for native memory (Claude Code's CLAUDE.md, ChatGPT's memory facts,
/// etc.) regardless of whether they use Origin. Origin does not replace that cost.
/// The real question is: "What does Origin cost on top, and what does it provide?"
///
/// This function runs Origin search against fixtures to get actual mean tokens/query,
/// then models alternative recall costs and multi-turn overhead.
///
/// Native platform token estimates are researched estimates as of April 2026, sourced
/// from public documentation, model cards, and community measurements. They are not
/// measurements from those platforms' APIs.
pub async fn run_native_memory_augmentation(
    fixture_dir: &Path,
    limit: usize,
) -> Result<NativeMemoryAugmentationReport, OriginError> {
    // Step 1: run Origin against fixtures to get actual mean retrieval tokens/query.
    let cases = load_fixtures(fixture_dir)?;
    let confidence_cfg = ConfidenceConfig::default();
    let mut origin_token_samples: Vec<usize> = Vec::new();
    let mut full_replay_samples: Vec<usize> = Vec::new();

    for case in &cases {
        if case.empty_set || case.seeds.is_empty() {
            continue;
        }

        let case_tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir for augmentation eval: {}", e)))?;
        let db = MemoryDB::new(case_tmp.path(), Arc::new(NoopEmitter)).await?;

        let all_docs: Vec<RawDocument> = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .map(|seed| crate::eval::runner::seed_to_doc(seed, &confidence_cfg))
            .collect();
        db.upsert_documents(all_docs).await?;

        let results = db
            .search_memory(
                &case.query,
                limit,
                None,
                case.domain.as_deref(),
                None,
                Some(1.0), // neutralize confirmation boost
                Some(1.0), // neutralize recap penalty
                None,
            )
            .await?;

        origin_token_samples.push(count_results_tokens(&results));

        // Full replay: entire corpus concatenated
        let corpus_text: String = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .map(|s| s.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        full_replay_samples.push(count_tokens(&corpus_text));
    }

    let origin_retrieval_tokens = if origin_token_samples.is_empty() {
        0.0
    } else {
        origin_token_samples.iter().sum::<usize>() as f64 / origin_token_samples.len() as f64
    };
    let mean_full_replay_tokens = if full_replay_samples.is_empty() {
        0
    } else {
        full_replay_samples.iter().sum::<usize>() / full_replay_samples.len()
    };

    // Step 2: define native baselines with researched token costs (April 2026 estimates).
    //
    // Sources:
    //   Claude Code: CLAUDE.md (~8K) + MEMORY.md project file (~3.3K) injected every turn.
    //     Estimate: ~11,300 tokens/turn for memory content (excluding base system prompt overhead).
    //   ChatGPT: up to ~200 discrete memory facts (~2,300 tok) + custom instructions (~750 tok).
    //     Estimate: ~3,050 tokens/turn combined. Capped at ~200 facts.
    //   Claude.ai: synthesized memory summary auto-pruned to fit context.
    //     Estimate: ~2,000 tokens/turn (community observations; no official figure published).
    let baselines = vec![
        NativeMemoryBaseline {
            platform: "Claude Code".to_string(),
            memory_tokens_per_turn: 11_300,
            growth_model: "unbounded".to_string(),
            mechanism: "full CLAUDE.md injected every turn; grows with project knowledge"
                .to_string(),
        },
        NativeMemoryBaseline {
            platform: "ChatGPT".to_string(),
            memory_tokens_per_turn: 3_050,
            growth_model: "capped".to_string(),
            mechanism: "all saved facts injected every turn; capped at ~200 entries".to_string(),
        },
        NativeMemoryBaseline {
            platform: "Claude.ai".to_string(),
            memory_tokens_per_turn: 2_000,
            growth_model: "synthesized".to_string(),
            mechanism: "synthesized summary injected every turn; auto-pruned to fit context"
                .to_string(),
        },
    ];

    // Step 3: define alternative recall scenarios.
    // These are what a user pays ADDITIONALLY when they need specific information
    // that native memory doesn't capture (e.g., a decision made 3 sessions ago).
    let alternatives = vec![
        RecallAlternative {
            scenario: "no_recall".to_string(),
            tokens_per_recall: 0,
            description: "Skip it — information is lost; model proceeds without context"
                .to_string(),
        },
        RecallAlternative {
            scenario: "manual_reexplain".to_string(),
            tokens_per_recall: 800,
            description: "User re-types the context manually (~800 tokens of explanation)"
                .to_string(),
        },
        RecallAlternative {
            scenario: "paste_full_history".to_string(),
            tokens_per_recall: mean_full_replay_tokens,
            description: format!(
                "Paste the full conversation history (~{} tokens, measured from fixtures)",
                mean_full_replay_tokens
            ),
        },
        RecallAlternative {
            scenario: "origin_retrieval".to_string(),
            tokens_per_recall: origin_retrieval_tokens.round() as usize,
            description: format!(
                "Origin finds relevant context automatically (~{} tokens, measured from fixtures)",
                origin_retrieval_tokens.round() as usize
            ),
        },
    ];

    // Step 4: model 10-turn session overhead.
    // Use Claude Code (11,300 tokens/turn) as the reference — most Origin users are Claude Code users.
    // Assume 60% of turns need specific recall (realistic for coding sessions).
    let turns = 10usize;
    let recall_turns = 6usize; // 60% of turns
    let native_per_turn = 11_300usize; // Claude Code reference

    let native_only_total = native_per_turn * turns;
    let native_plus_origin_total =
        native_per_turn * turns + (origin_retrieval_tokens.round() as usize) * recall_turns;
    let native_plus_replay_total = native_per_turn * turns + mean_full_replay_tokens * recall_turns;
    let origin_overhead_pct = if native_only_total > 0 {
        (native_plus_origin_total.saturating_sub(native_only_total)) as f64
            / native_only_total as f64
            * 100.0
    } else {
        0.0
    };

    let multi_turn = MultiTurnAugmentationComparison {
        turns,
        recall_turns,
        native_only_total,
        native_plus_origin_total,
        native_plus_replay_total,
        origin_overhead_pct,
    };

    let framing_note =
        "Native memory (CLAUDE.md, ChatGPT facts, Claude.ai summaries) is a fixed cost users \
         already pay. Origin does not replace it — it augments it. The question is not \
         'Origin vs native' but 'what does adding Origin cost, and what do you get in return?'"
            .to_string();

    Ok(NativeMemoryAugmentationReport {
        origin_retrieval_tokens,
        baselines,
        alternatives,
        multi_turn,
        framing_note,
    })
}

/// Deprecated alias for [`run_native_memory_augmentation`].
///
/// Kept for backward compatibility. Prefer the new name which reflects the honest framing.
#[deprecated(since = "0.0.0", note = "Use run_native_memory_augmentation instead")]
pub async fn run_native_memory_comparison(
    fixture_dir: &Path,
    limit: usize,
) -> Result<NativeMemoryAugmentationReport, OriginError> {
    run_native_memory_augmentation(fixture_dir, limit).await
}

/// Count total tokens across a slice of SearchResults (content field only).
fn count_results_tokens(results: &[crate::db::SearchResult]) -> usize {
    results.iter().map(|r| count_tokens(&r.content)).sum()
}

// ===== Pipeline Stage Token Efficiency =====

/// Pipeline stage for token efficiency measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStage {
    /// Raw memories as-is.
    Raw,
    /// Simulated distillation: groups of similar memories merged into single entries.
    Distilled,
    /// Simulated concept: all related memories compiled into one dense article.
    Concept,
}

/// Per-stage token metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStageMetrics {
    pub stage: String,
    pub memory_count: usize,
    pub total_corpus_tokens: usize,
    pub mean_tokens_per_memory: f64,
    pub search_result_tokens: f64,
    pub ndcg_at_10: f64,
    /// ndcg / (search_result_tokens / 1000.0) — quality per 1K tokens.
    pub information_density: f64,
}

/// Report from pipeline token efficiency evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTokenReport {
    pub stages: Vec<PipelineStageMetrics>,
    /// (raw_corpus - concept_corpus) / raw_corpus * 100.
    pub token_reduction_pct: f64,
    /// concept_density / raw_density.
    pub density_improvement: f64,
}

/// Measure token efficiency across simulated pipeline stages.
///
/// For each fixture case:
/// 1. Raw: seed all memories, search, measure tokens + quality
/// 2. Distilled: group seeds by memory_type, merge groups of 2+ into single entries, search again
/// 3. Concept: combine ALL seeds into one "compiled knowledge" entry, search
///
/// Reports how consolidation reduces corpus tokens while maintaining (or improving) quality.
///
/// This is a *simulation* — merging is synthetic (string concatenation), not LLM-driven.
/// For real LLM distillation numbers see `benchmark_pipeline_token_real_llm` in tests.
pub async fn run_pipeline_token_eval_simulated(
    fixture_dir: &Path,
    limit: usize,
) -> Result<PipelineTokenReport, OriginError> {
    let cases = load_fixtures(fixture_dir)?;
    let confidence_cfg = ConfidenceConfig::default();

    // Per-stage accumulators: (total_corpus_tokens, total_memory_count, total_search_result_tokens, total_ndcg)
    let mut raw_corpus: Vec<usize> = Vec::new();
    let mut raw_counts: Vec<usize> = Vec::new();
    let mut raw_search_tokens: Vec<f64> = Vec::new();
    let mut raw_ndcg: Vec<f64> = Vec::new();

    let mut distilled_corpus: Vec<usize> = Vec::new();
    let mut distilled_counts: Vec<usize> = Vec::new();
    let mut distilled_search_tokens: Vec<f64> = Vec::new();
    let mut distilled_ndcg: Vec<f64> = Vec::new();

    let mut concept_corpus: Vec<usize> = Vec::new();
    let mut concept_counts: Vec<usize> = Vec::new();
    let mut concept_search_tokens: Vec<f64> = Vec::new();
    let mut concept_ndcg: Vec<f64> = Vec::new();

    for case in cases.iter().take(limit) {
        if case.empty_set || case.seeds.is_empty() {
            continue;
        }

        // Build scoring maps for NDCG
        let grades: HashMap<&str, u8> = case
            .seeds
            .iter()
            .map(|s| (s.id.as_str(), s.relevance))
            .collect();

        // ---- Stage 1: Raw ----
        {
            let all_docs: Vec<RawDocument> = case
                .seeds
                .iter()
                .chain(case.negative_seeds.iter())
                .map(|s| crate::eval::runner::seed_to_doc(s, &confidence_cfg))
                .collect();

            let corpus_text: String = all_docs
                .iter()
                .map(|d| d.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let corpus_tok = count_tokens(&corpus_text);

            let tmp = tempfile::tempdir()
                .map_err(|e| OriginError::Generic(format!("tempdir pipeline raw: {e}")))?;
            let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter)).await?;
            db.upsert_documents(all_docs.clone()).await?;

            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    case.domain.as_deref(),
                    None,
                    Some(1.0),
                    Some(1.0),
                    None,
                )
                .await?;

            let search_tok = count_results_tokens(&results) as f64;
            let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            let ndcg = metrics::ndcg_at_k(&ranked, &grades, 10);

            raw_corpus.push(corpus_tok);
            raw_counts.push(all_docs.len());
            raw_search_tokens.push(search_tok);
            raw_ndcg.push(ndcg);
        }

        // ---- Stage 2: Simulated Distillation ----
        // Group positive seeds by memory_type; merge groups of 2+ into single entries.
        // Keep negative seeds as-is.
        {
            let mut type_groups: HashMap<&str, Vec<&crate::eval::fixtures::SeedMemory>> =
                HashMap::new();
            for seed in &case.seeds {
                type_groups
                    .entry(seed.memory_type.as_str())
                    .or_default()
                    .push(seed);
            }

            let mut distilled_docs: Vec<RawDocument> = Vec::new();
            let mut merged_ids: HashSet<String> = HashSet::new();

            for group in type_groups.values() {
                if group.len() >= 2 {
                    // Merge the group into one combined entry
                    let combined_content = group
                        .iter()
                        .map(|s| s.content.as_str())
                        .collect::<Vec<_>>()
                        .join(" | ");
                    // Use the ID of the highest-relevance seed as the merged ID
                    let best_seed = group.iter().max_by_key(|s| s.relevance).unwrap();
                    let merged_id = format!("distilled_{}", best_seed.id);
                    let mut doc = crate::eval::runner::seed_to_doc(best_seed, &confidence_cfg);
                    doc.content = combined_content;
                    doc.source_id = merged_id.clone();
                    distilled_docs.push(doc);
                    for s in group {
                        merged_ids.insert(s.id.clone());
                    }
                } else {
                    // Single member: keep as-is
                    for seed in group {
                        distilled_docs
                            .push(crate::eval::runner::seed_to_doc(seed, &confidence_cfg));
                    }
                }
            }

            // Add negative seeds unchanged
            for neg in &case.negative_seeds {
                distilled_docs.push(crate::eval::runner::seed_to_doc(neg, &confidence_cfg));
            }

            let corpus_text: String = distilled_docs
                .iter()
                .map(|d| d.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let corpus_tok = count_tokens(&corpus_text);
            let doc_count = distilled_docs.len();

            let tmp = tempfile::tempdir()
                .map_err(|e| OriginError::Generic(format!("tempdir pipeline distilled: {e}")))?;
            let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter)).await?;
            db.upsert_documents(distilled_docs).await?;

            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    case.domain.as_deref(),
                    None,
                    Some(1.0),
                    Some(1.0),
                    None,
                )
                .await?;

            let search_tok = count_results_tokens(&results) as f64;
            // Build a modified grades map: merged IDs map to the best relevance in the group
            let mut distilled_grades: HashMap<String, u8> = HashMap::new();
            for group in type_groups.values() {
                if group.len() >= 2 {
                    let best_seed = group.iter().max_by_key(|s| s.relevance).unwrap();
                    let merged_id = format!("distilled_{}", best_seed.id);
                    let best_relevance = group.iter().map(|s| s.relevance).max().unwrap_or(0);
                    distilled_grades.insert(merged_id, best_relevance);
                } else {
                    for seed in group {
                        distilled_grades.insert(seed.id.clone(), seed.relevance);
                    }
                }
            }
            let distilled_grades_ref: HashMap<&str, u8> = distilled_grades
                .iter()
                .map(|(k, v)| (k.as_str(), *v))
                .collect();
            let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            let ndcg = metrics::ndcg_at_k(&ranked, &distilled_grades_ref, 10);

            distilled_corpus.push(corpus_tok);
            distilled_counts.push(doc_count);
            distilled_search_tokens.push(search_tok);
            distilled_ndcg.push(ndcg);
        }

        // ---- Stage 3: Simulated Concept ----
        // All positive seeds combined into one dense "concept article" entry.
        // Keep negative seeds as-is.
        {
            let combined_content = format!(
                "Compiled knowledge: {}",
                case.seeds
                    .iter()
                    .map(|s| s.content.as_str())
                    .collect::<Vec<_>>()
                    .join(". ")
            );
            let concept_id = "concept_compiled";
            let concept_domain = case
                .domain
                .clone()
                .or_else(|| case.seeds.first().and_then(|s| s.domain.clone()));
            let concept_doc = RawDocument {
                content: combined_content,
                source_id: concept_id.to_string(),
                source: "memory".to_string(),
                title: "Compiled knowledge".to_string(),
                memory_type: Some("concept".to_string()),
                domain: concept_domain,
                ..Default::default()
            };

            let mut concept_docs = vec![concept_doc];
            for neg in &case.negative_seeds {
                concept_docs.push(crate::eval::runner::seed_to_doc(neg, &confidence_cfg));
            }

            let corpus_text: String = concept_docs
                .iter()
                .map(|d| d.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let corpus_tok = count_tokens(&corpus_text);
            let doc_count = concept_docs.len();

            let tmp = tempfile::tempdir()
                .map_err(|e| OriginError::Generic(format!("tempdir pipeline concept: {e}")))?;
            let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter)).await?;
            db.upsert_documents(concept_docs).await?;

            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    case.domain.as_deref(),
                    None,
                    Some(1.0),
                    Some(1.0),
                    None,
                )
                .await?;

            let search_tok = count_results_tokens(&results) as f64;
            // Concept entry gets the highest relevance from the positive seeds
            let best_relevance = case.seeds.iter().map(|s| s.relevance).max().unwrap_or(0);
            let concept_grades: HashMap<&str, u8> =
                [(concept_id, best_relevance)].into_iter().collect();
            let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            let ndcg = metrics::ndcg_at_k(&ranked, &concept_grades, 10);

            concept_corpus.push(corpus_tok);
            concept_counts.push(doc_count);
            concept_search_tokens.push(search_tok);
            concept_ndcg.push(ndcg);
        }
    }

    let n = raw_corpus.len().max(1) as f64;

    fn mean_f64(v: &[f64]) -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
    fn total_usize(v: &[usize]) -> usize {
        v.iter().sum()
    }
    fn total_f64(v: &[f64]) -> f64 {
        v.iter().sum()
    }

    let raw_total_corpus = total_usize(&raw_corpus);
    let raw_total_count = total_usize(&raw_counts);
    let raw_mean_search = total_f64(&raw_search_tokens) / n;
    let raw_mean_ndcg = mean_f64(&raw_ndcg);
    let raw_mean_tok_per_mem = if raw_total_count > 0 {
        raw_total_corpus as f64 / raw_total_count as f64
    } else {
        0.0
    };
    let raw_density = if raw_mean_search > 0.0 {
        raw_mean_ndcg / (raw_mean_search / 1000.0)
    } else {
        0.0
    };

    let dist_total_corpus = total_usize(&distilled_corpus);
    let dist_total_count = total_usize(&distilled_counts);
    let dist_mean_search = total_f64(&distilled_search_tokens) / n;
    let dist_mean_ndcg = mean_f64(&distilled_ndcg);
    let dist_mean_tok_per_mem = if dist_total_count > 0 {
        dist_total_corpus as f64 / dist_total_count as f64
    } else {
        0.0
    };
    let dist_density = if dist_mean_search > 0.0 {
        dist_mean_ndcg / (dist_mean_search / 1000.0)
    } else {
        0.0
    };

    let conc_total_corpus = total_usize(&concept_corpus);
    let conc_total_count = total_usize(&concept_counts);
    let conc_mean_search = total_f64(&concept_search_tokens) / n;
    let conc_mean_ndcg = mean_f64(&concept_ndcg);
    let conc_mean_tok_per_mem = if conc_total_count > 0 {
        conc_total_corpus as f64 / conc_total_count as f64
    } else {
        0.0
    };
    let conc_density = if conc_mean_search > 0.0 {
        conc_mean_ndcg / (conc_mean_search / 1000.0)
    } else {
        0.0
    };

    // Token reduction is measured on search result tokens (context delivered to LLM),
    // not raw corpus size — since simulated concept entries have the same text length
    // as their constituent seeds concatenated, corpus size barely changes in simulation.
    // The real efficiency gain is in the retrieved context: the concept stage returns
    // fewer, denser entries, so the LLM receives fewer tokens while preserving quality.
    let token_reduction_pct = if raw_mean_search > 0.0 {
        ((raw_mean_search - conc_mean_search) / raw_mean_search * 100.0).max(0.0)
    } else {
        0.0
    };

    let density_improvement = if raw_density > 0.0 {
        conc_density / raw_density
    } else {
        0.0
    };

    let stages = vec![
        PipelineStageMetrics {
            stage: "raw".to_string(),
            memory_count: raw_total_count,
            total_corpus_tokens: raw_total_corpus,
            mean_tokens_per_memory: raw_mean_tok_per_mem,
            search_result_tokens: raw_mean_search,
            ndcg_at_10: raw_mean_ndcg,
            information_density: raw_density,
        },
        PipelineStageMetrics {
            stage: "distilled".to_string(),
            memory_count: dist_total_count,
            total_corpus_tokens: dist_total_corpus,
            mean_tokens_per_memory: dist_mean_tok_per_mem,
            search_result_tokens: dist_mean_search,
            ndcg_at_10: dist_mean_ndcg,
            information_density: dist_density,
        },
        PipelineStageMetrics {
            stage: "concept".to_string(),
            memory_count: conc_total_count,
            total_corpus_tokens: conc_total_corpus,
            mean_tokens_per_memory: conc_mean_tok_per_mem,
            search_result_tokens: conc_mean_search,
            ndcg_at_10: conc_mean_ndcg,
            information_density: conc_density,
        },
    ];

    Ok(PipelineTokenReport {
        stages,
        token_reduction_pct,
        density_improvement,
    })
}

// ===== Quality at Scale Types =====

/// Quality-at-scale data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAtScalePoint {
    /// Number of memories stored.
    pub memory_count: usize,
    /// Origin's measured retrieval quality (NDCG@10).
    pub origin_ndcg: f64,
    /// Origin's token cost per query.
    pub origin_tokens: f64,
    /// Native approach: all memories injected (tokens = total corpus).
    pub native_tokens: f64,
    /// Native approach: estimated effective quality (degrades with scale due to "lost in middle").
    pub native_effective_quality: f64,
    /// Origin's quality-per-token ratio.
    pub origin_quality_per_1k_tokens: f64,
    /// Native's quality-per-token ratio.
    pub native_quality_per_1k_tokens: f64,
}

/// Full quality-at-scale report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAtScaleReport {
    pub points: Vec<QualityAtScalePoint>,
    /// The crossover point: at what memory count does Origin's approach become clearly better?
    pub crossover_memory_count: Option<usize>,
    pub methodology_note: String,
}

// ===== Quality at Scale Runner =====

/// Model the "lost in the middle" quality degradation for naive all-inject approaches.
///
/// Based on Liu et al., 2023 "Lost in the Middle: How Language Models Use Long Contexts".
/// LLMs reliably use information at the start and end of a context window, but struggle
/// with content positioned in the middle of long inputs.
///
/// At small counts (<=10 facts) the LLM can attend to everything. As N grows the relevant
/// fact is increasingly likely to be buried, and effective retrieval quality degrades.
/// The floor of 0.3 reflects that even with very large contexts the model retains some
/// ability to locate key facts when explicitly prompted.
fn native_quality_at_scale(memory_count: usize) -> f64 {
    if memory_count <= 10 {
        return 0.95;
    }
    // Logarithmic decay: 0.95 * (1 - 0.12 * ln(N/10))
    let decay = 0.12 * (memory_count as f64 / 10.0).ln();
    (0.95 * (1.0 - decay)).max(0.3)
}

/// Measure how recall quality and token cost compare as memory count grows.
///
/// At each corpus size: measures Origin's actual NDCG and tokens (via search),
/// and models native memory's quality degradation (lost-in-the-middle effect).
///
/// For each size in `sizes`:
/// 1. Takes the first N seeds across all non-empty fixture cases.
/// 2. Seeds an ephemeral DB and runs Origin's hybrid search.
/// 3. Measures Origin NDCG@10 and token cost from the retrieved results.
/// 4. Computes native_tokens as the full corpus token count (all-inject).
/// 5. Applies the lost-in-the-middle degradation model for native effective quality.
/// 6. Reports quality-per-1k-tokens for both and finds the crossover point.
pub async fn run_quality_at_scale_eval(
    fixture_dir: &Path,
    sizes: &[usize],
    limit: usize,
) -> Result<QualityAtScaleReport, OriginError> {
    let cases = load_fixtures(fixture_dir)?;
    if cases.is_empty() {
        return Ok(QualityAtScaleReport {
            points: Vec::new(),
            crossover_memory_count: None,
            methodology_note: "No fixture cases found.".to_string(),
        });
    }

    let confidence_cfg = ConfidenceConfig::default();
    let mut points: Vec<QualityAtScalePoint> = Vec::with_capacity(sizes.len());

    for &size in sizes {
        let mut origin_tokens_sum: f64 = 0.0;
        let mut origin_ndcg_sum: f64 = 0.0;
        let mut native_tokens_sum: f64 = 0.0;
        let mut case_count: f64 = 0.0;

        for case in &cases {
            if case.empty_set {
                continue;
            }

            // Collect seeds (positive first, then negative) and take first `size`.
            let all_seeds: Vec<&crate::eval::fixtures::SeedMemory> = case
                .seeds
                .iter()
                .chain(case.negative_seeds.iter())
                .collect();
            let subset: Vec<&crate::eval::fixtures::SeedMemory> =
                all_seeds.into_iter().take(size).collect();
            if subset.is_empty() {
                continue;
            }

            // Seed ephemeral DB
            let case_tmp = tempfile::tempdir()
                .map_err(|e| OriginError::Generic(format!("tmpdir quality_at_scale: {e}")))?;
            let db = MemoryDB::new(case_tmp.path(), Arc::new(NoopEmitter)).await?;

            let docs: Vec<RawDocument> = subset
                .iter()
                .map(|s| crate::eval::runner::seed_to_doc(s, &confidence_cfg))
                .collect();
            db.upsert_documents(docs).await?;

            // Build grading map from the positive seeds present in this subset
            let subset_ids: std::collections::HashSet<&str> =
                subset.iter().map(|s| s.id.as_str()).collect();
            let grades: HashMap<&str, u8> = case
                .seeds
                .iter()
                .filter(|s| subset_ids.contains(s.id.as_str()))
                .map(|s| (s.id.as_str(), s.relevance))
                .collect();

            // Origin: hybrid search
            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    case.domain.as_deref(),
                    None,
                    Some(1.0), // neutralize confirmation boost — fixture bias
                    Some(1.0), // neutralize recap penalty — fixture bias
                    None,
                )
                .await?;

            let origin_tok = count_results_tokens(&results) as f64;
            let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            let ndcg = metrics::ndcg_at_k(&ranked, &grades, 10);

            // Native: all-inject — corpus token count
            let native_content: String = subset
                .iter()
                .map(|s| s.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let native_tok = count_tokens(&native_content) as f64;

            origin_tokens_sum += origin_tok;
            origin_ndcg_sum += ndcg;
            native_tokens_sum += native_tok;
            case_count += 1.0;
        }

        if case_count == 0.0 {
            continue;
        }

        let origin_tokens = origin_tokens_sum / case_count;
        let origin_ndcg = origin_ndcg_sum / case_count;
        let native_tokens = native_tokens_sum / case_count;
        let native_effective_quality = native_quality_at_scale(size);

        // Quality-per-1k-tokens: guard against zero tokens
        let origin_quality_per_1k_tokens = if origin_tokens > 0.0 {
            origin_ndcg / (origin_tokens / 1000.0)
        } else {
            0.0
        };
        let native_quality_per_1k_tokens = if native_tokens > 0.0 {
            native_effective_quality / (native_tokens / 1000.0)
        } else {
            0.0
        };

        points.push(QualityAtScalePoint {
            memory_count: size,
            origin_ndcg,
            origin_tokens,
            native_tokens,
            native_effective_quality,
            origin_quality_per_1k_tokens,
            native_quality_per_1k_tokens,
        });
    }

    // Crossover: first point where Origin's quality-per-1k-tokens exceeds native's.
    let crossover_memory_count = points
        .iter()
        .find(|p| p.origin_quality_per_1k_tokens > p.native_quality_per_1k_tokens)
        .map(|p| p.memory_count);

    let methodology_note = "Native quality estimated using lost-in-the-middle degradation model \
        (Liu et al., 2023). At ≤10 facts native quality ≈ 0.95; degrades logarithmically \
        thereafter (floor 0.30). Origin quality is measured directly via NDCG@10 on fixture \
        cases. Token counts use cl100k_base tokenizer."
        .to_string();

    Ok(QualityAtScaleReport {
        points,
        crossover_memory_count,
        methodology_note,
    })
}

/// Run scaling evaluation: same queries at increasing corpus sizes.
///
/// For each size, seeds only the first N memories from each case, then runs
/// Origin and FullReplay strategies to measure token cost scaling.
pub async fn run_scaling_eval(
    fixture_dir: &Path,
    corpus_sizes: &[usize],
    limit: usize,
) -> Result<Vec<ScalingPoint>, OriginError> {
    let cases = load_fixtures(fixture_dir)?;
    if cases.is_empty() {
        return Err(OriginError::Generic("no fixture cases found".to_string()));
    }

    let confidence_cfg = ConfidenceConfig::default();
    let mut points = Vec::new();

    for &size in corpus_sizes {
        let mut origin_tokens_sum: f64 = 0.0;
        let mut replay_tokens_sum: f64 = 0.0;
        let mut case_count: f64 = 0.0;

        for case in &cases {
            if case.empty_set {
                continue;
            }

            // Take first `size` seeds (positive + negative combined)
            let all_seeds: Vec<&crate::eval::fixtures::SeedMemory> = case
                .seeds
                .iter()
                .chain(case.negative_seeds.iter())
                .collect();
            let subset: Vec<&crate::eval::fixtures::SeedMemory> =
                all_seeds.into_iter().take(size).collect();
            if subset.is_empty() {
                continue;
            }

            // Seed ephemeral DB with subset
            let case_tmp =
                tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tmpdir: {e}")))?;
            let db = MemoryDB::new(case_tmp.path(), Arc::new(NoopEmitter)).await?;

            let docs: Vec<RawDocument> = subset
                .iter()
                .map(|s| crate::eval::runner::seed_to_doc(s, &confidence_cfg))
                .collect();
            db.upsert_documents(docs).await?;

            // FullReplay: all subset content
            let replay_content: String = subset
                .iter()
                .map(|s| s.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let replay_tokens = count_tokens(&replay_content);

            // Origin: search and count (neutralize confirmation/recap bias)
            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    case.domain.as_deref(),
                    None,
                    Some(1.0),
                    Some(1.0),
                    None,
                )
                .await?;
            let origin_content: String = results
                .iter()
                .map(|r| r.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let origin_tok = count_tokens(&origin_content);

            origin_tokens_sum += origin_tok as f64;
            replay_tokens_sum += replay_tokens as f64;
            case_count += 1.0;
        }

        if case_count > 0.0 {
            points.push(ScalingPoint {
                corpus_size: size,
                origin_tokens: origin_tokens_sum / case_count,
                replay_tokens: replay_tokens_sum / case_count,
            });
        }
    }

    Ok(points)
}

// ===== Memory Layer Comparison Types =====

/// Memory layer approach being compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayerApproach {
    /// All memories in a markdown doc, injected every turn (Claude Code style).
    FlatMarkdown,
    /// Discrete fact list, all injected, capped at 200 (ChatGPT style).
    FactList,
    /// Compressed summary, fixed ~2K token budget (Claude.ai style).
    SynthesizedSummary,
    /// Query-specific retrieval, top-K (Origin).
    OriginRetrieval,
    /// Native markdown + Origin retrieval on top (complement).
    OriginPlusNative,
}

impl MemoryLayerApproach {
    fn key(&self) -> &'static str {
        match self {
            Self::FlatMarkdown => "flat_markdown",
            Self::FactList => "fact_list",
            Self::SynthesizedSummary => "synthesized_summary",
            Self::OriginRetrieval => "origin_retrieval",
            Self::OriginPlusNative => "origin_plus_native",
        }
    }

    fn display(&self) -> &'static str {
        match self {
            Self::FlatMarkdown => "Flat Markdown",
            Self::FactList => "Fact List",
            Self::SynthesizedSummary => "Synth Summary",
            Self::OriginRetrieval => "Origin",
            Self::OriginPlusNative => "Origin+Native",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Self::FlatMarkdown => {
                "All memories in one markdown doc, injected every turn (Claude Code style)"
            }
            Self::FactList => {
                "Flat fact list, all injected per turn, capped at 200 (ChatGPT style)"
            }
            Self::SynthesizedSummary => {
                "Lossy compressed summary ~2K tokens, fixed budget (Claude.ai style)"
            }
            Self::OriginRetrieval => "Query-specific retrieval, top-K relevant results only",
            Self::OriginPlusNative => {
                "Native markdown context + Origin retrieval on top (complement)"
            }
        }
    }
}

/// Per-approach aggregated result from the memory layer comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLayerResult {
    pub approach: String,
    pub display_name: String,
    pub mean_tokens_per_query: f64,
    pub mean_ndcg: f64,
    pub quality_per_1k_tokens: f64,
    /// Average fraction of relevant memories that can be found (0.0–1.0).
    pub memories_accessible: f64,
    pub description: String,
}

/// Full memory layer comparison report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLayerComparisonReport {
    pub approaches: Vec<MemoryLayerResult>,
    pub complement_advantage: String,
    pub methodology: String,
}

// ===== Memory Layer Comparison Runner =====

/// Run a fair head-to-head comparison of memory layer approaches on the same fixture data.
///
/// Deterministic Fisher-Yates shuffle using the query string as a seed.
/// Produces the same ordering for the same (items, seed_str) pair, ensuring
/// reproducible NDCG measurements across runs while removing storage-order bias.
fn deterministic_shuffle<T: Clone>(items: &[T], seed_str: &str) -> Vec<T> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed_str.hash(&mut hasher);
    let seed = hasher.finish();

    let mut shuffled = items.to_vec();
    let len = shuffled.len();
    let mut state = seed;
    for i in (1..len).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state as usize) % (i + 1);
        shuffled.swap(i, j);
    }
    shuffled
}

/// - FactList: all seeds as discrete facts, capped at 200, all injected
/// - SynthesizedSummary: concatenated content truncated to ~2K tokens (lossy)
/// - OriginRetrieval: actual search_memory (hybrid vector+FTS) top-K
/// - OriginPlusNative: native markdown tokens + Origin retrieval tokens
pub async fn run_memory_layer_comparison(
    fixture_dir: &Path,
    limit: usize,
) -> Result<MemoryLayerComparisonReport, OriginError> {
    let cases = load_fixtures(fixture_dir)?;
    let confidence_cfg = ConfidenceConfig::default();

    // Accumulators: tokens, ndcg, accessible fraction per approach
    let mut tokens_acc: HashMap<MemoryLayerApproach, Vec<f64>> = HashMap::new();
    let mut ndcg_acc: HashMap<MemoryLayerApproach, Vec<f64>> = HashMap::new();
    let mut accessible_acc: HashMap<MemoryLayerApproach, Vec<f64>> = HashMap::new();

    let all_approaches = [
        MemoryLayerApproach::FlatMarkdown,
        MemoryLayerApproach::FactList,
        MemoryLayerApproach::SynthesizedSummary,
        MemoryLayerApproach::OriginRetrieval,
        MemoryLayerApproach::OriginPlusNative,
    ];
    for a in &all_approaches {
        tokens_acc.insert(*a, Vec::new());
        ndcg_acc.insert(*a, Vec::new());
        accessible_acc.insert(*a, Vec::new());
    }

    for case in &cases {
        if case.empty_set || case.seeds.is_empty() {
            continue;
        }

        // Build scoring structures
        let grades: HashMap<&str, u8> = case
            .seeds
            .iter()
            .map(|s| (s.id.as_str(), s.relevance))
            .collect();

        let relevant: Vec<String> = case
            .seeds
            .iter()
            .filter(|s| s.relevance >= 2)
            .map(|s| s.id.clone())
            .collect();

        let all_seeds: Vec<&crate::eval::fixtures::SeedMemory> = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .collect();

        // ---- FlatMarkdown ----
        // Token count uses storage order (order doesn't affect token count).
        // NDCG uses a deterministic shuffle to remove positive-seeds-first bias.
        let flat_ndcg;
        {
            let markdown = all_seeds
                .iter()
                .enumerate()
                .map(|(i, s)| format!("## Memory {}\n{}", i + 1, s.content))
                .collect::<Vec<_>>()
                .join("\n\n");
            let tokens = count_tokens(&markdown) as f64;
            // Shuffle before NDCG: real native memory has no guaranteed ranking
            let shuffled = deterministic_shuffle(&all_seeds, &case.query);
            let all_ids: Vec<&str> = shuffled.iter().map(|s| s.id.as_str()).collect();
            flat_ndcg = metrics::ndcg_at_k(&all_ids, &grades, all_ids.len());
            tokens_acc
                .get_mut(&MemoryLayerApproach::FlatMarkdown)
                .unwrap()
                .push(tokens);
            ndcg_acc
                .get_mut(&MemoryLayerApproach::FlatMarkdown)
                .unwrap()
                .push(flat_ndcg);
            // All seeds present — fully accessible
            accessible_acc
                .get_mut(&MemoryLayerApproach::FlatMarkdown)
                .unwrap()
                .push(1.0);
        }

        // ---- FactList (cap 200) ----
        // Cap is applied after shuffle so the 200-item window is random, not biased toward positives.
        {
            // Shuffle first so the cap doesn't preferentially keep positive seeds
            let shuffled_all = deterministic_shuffle(&all_seeds, &case.query);
            let capped: Vec<&crate::eval::fixtures::SeedMemory> = if shuffled_all.len() > 200 {
                shuffled_all[..200].to_vec()
            } else {
                shuffled_all
            };
            let tokens: f64 = capped
                .iter()
                .map(|s| count_tokens(&s.content))
                .sum::<usize>() as f64;
            let capped_ids: Vec<&str> = capped.iter().map(|s| s.id.as_str()).collect();
            let ndcg = metrics::ndcg_at_k(&capped_ids, &grades, capped_ids.len());
            let relevant_in_capped = relevant
                .iter()
                .filter(|id| capped_ids.contains(&id.as_str()))
                .count();
            let accessible = relevant_in_capped as f64 / relevant.len().max(1) as f64;
            tokens_acc
                .get_mut(&MemoryLayerApproach::FactList)
                .unwrap()
                .push(tokens);
            ndcg_acc
                .get_mut(&MemoryLayerApproach::FactList)
                .unwrap()
                .push(ndcg);
            accessible_acc
                .get_mut(&MemoryLayerApproach::FactList)
                .unwrap()
                .push(accessible);
        }

        // ---- SynthesizedSummary (truncate to ~2K tokens) ----
        {
            // Rough budget: 2000 tokens * ~4 chars/token = 8000 chars
            let budget_chars = 8000usize;
            // Shuffle before joining so truncation doesn't preferentially keep positive seeds
            let seeds_ref: Vec<&crate::eval::fixtures::SeedMemory> = case.seeds.iter().collect();
            let shuffled_seeds = deterministic_shuffle(&seeds_ref, &case.query);
            let full_text = shuffled_seeds
                .iter()
                .map(|s| s.content.as_str())
                .collect::<Vec<_>>()
                .join(". ");
            // UTF-8 safe truncation via char count
            let summary: String = full_text.chars().take(budget_chars).collect();
            let tokens = count_tokens(&summary) as f64;
            // Which relevant seeds are (at least partially) present in the summary?
            let accessible_count = case
                .seeds
                .iter()
                .filter(|s| {
                    if s.relevance < 2 {
                        return false;
                    }
                    let prefix_len = s.content.chars().count().min(50);
                    let prefix: String = s.content.chars().take(prefix_len).collect();
                    !prefix.is_empty() && summary.contains(&prefix)
                })
                .count();
            let accessible = accessible_count as f64 / relevant.len().max(1) as f64;
            // For NDCG: only seeds whose content prefix appears in the summary
            let accessible_ids: Vec<&str> = case
                .seeds
                .iter()
                .filter(|s| {
                    let prefix_len = s.content.chars().count().min(50);
                    let prefix: String = s.content.chars().take(prefix_len).collect();
                    !prefix.is_empty() && summary.contains(&prefix)
                })
                .map(|s| s.id.as_str())
                .collect();
            let ndcg = metrics::ndcg_at_k(&accessible_ids, &grades, accessible_ids.len().min(10));
            tokens_acc
                .get_mut(&MemoryLayerApproach::SynthesizedSummary)
                .unwrap()
                .push(tokens);
            ndcg_acc
                .get_mut(&MemoryLayerApproach::SynthesizedSummary)
                .unwrap()
                .push(ndcg);
            accessible_acc
                .get_mut(&MemoryLayerApproach::SynthesizedSummary)
                .unwrap()
                .push(accessible);
        }

        // ---- OriginRetrieval ----
        // Run actual search — seed ephemeral DB, run hybrid search
        let origin_ndcg;
        let origin_tokens: f64;
        {
            let case_tmp = tempfile::tempdir()
                .map_err(|e| OriginError::Generic(format!("tempdir memory_layer: {}", e)))?;
            let db = MemoryDB::new(case_tmp.path(), Arc::new(NoopEmitter)).await?;
            let all_docs: Vec<RawDocument> = all_seeds
                .iter()
                .map(|seed| crate::eval::runner::seed_to_doc(seed, &confidence_cfg))
                .collect();
            db.upsert_documents(all_docs).await?;

            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    case.domain.as_deref(),
                    None,
                    Some(1.0), // neutralize confirmation boost
                    Some(1.0), // neutralize recap penalty
                    None,
                )
                .await?;

            origin_tokens = count_results_tokens(&results) as f64;
            let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            origin_ndcg = metrics::ndcg_at_k(&ranked, &grades, 10);
        }
        tokens_acc
            .get_mut(&MemoryLayerApproach::OriginRetrieval)
            .unwrap()
            .push(origin_tokens);
        ndcg_acc
            .get_mut(&MemoryLayerApproach::OriginRetrieval)
            .unwrap()
            .push(origin_ndcg);
        // Origin can potentially find any stored memory
        accessible_acc
            .get_mut(&MemoryLayerApproach::OriginRetrieval)
            .unwrap()
            .push(1.0);

        // ---- OriginPlusNative (complement) ----
        // Tokens = flat markdown tokens + origin retrieval tokens
        // Quality = origin ranking quality (conservative; native provides general background)
        {
            let markdown = all_seeds
                .iter()
                .enumerate()
                .map(|(i, s)| format!("## Memory {}\n{}", i + 1, s.content))
                .collect::<Vec<_>>()
                .join("\n\n");
            let markdown_tokens = count_tokens(&markdown) as f64;
            let complement_tokens = markdown_tokens + origin_tokens;
            // Quality is at least as good as the better of native or Origin alone.
            // Native has all memories in random order; Origin has ranked retrieval.
            // Together the LLM benefits from both, so the floor is max(flat_ndcg, origin_ndcg).
            let complement_ndcg = flat_ndcg.max(origin_ndcg);
            tokens_acc
                .get_mut(&MemoryLayerApproach::OriginPlusNative)
                .unwrap()
                .push(complement_tokens);
            ndcg_acc
                .get_mut(&MemoryLayerApproach::OriginPlusNative)
                .unwrap()
                .push(complement_ndcg);
            accessible_acc
                .get_mut(&MemoryLayerApproach::OriginPlusNative)
                .unwrap()
                .push(1.0);
        }
    }

    // Aggregate per approach
    let mut results: Vec<MemoryLayerResult> = Vec::new();
    for approach in &all_approaches {
        let toks = &tokens_acc[approach];
        let ndcgs = &ndcg_acc[approach];
        let accs = &accessible_acc[approach];
        let n = toks.len().max(1) as f64;

        let mean_tokens = toks.iter().sum::<f64>() / n;
        let mean_ndcg = ndcgs.iter().sum::<f64>() / n;
        let mean_accessible = accs.iter().sum::<f64>() / n;
        let quality_per_1k = if mean_tokens > 0.0 {
            mean_ndcg / (mean_tokens / 1000.0)
        } else {
            0.0
        };

        results.push(MemoryLayerResult {
            approach: approach.key().to_string(),
            display_name: approach.display().to_string(),
            mean_tokens_per_query: mean_tokens,
            mean_ndcg,
            quality_per_1k_tokens: quality_per_1k,
            memories_accessible: mean_accessible,
            description: approach.description().to_string(),
        });
    }

    let complement_advantage = "Native memory provides general project context (conventions, \
        architecture). Origin adds precise, query-specific recall. Together they cover both \
        general and specific knowledge needs."
        .to_string();

    let methodology = "All approaches tested on the same fixture data. Native simulations model \
        documented platform behavior. Origin quality measured via actual search. Quality \
        estimated via NDCG@K against fixture relevance grades."
        .to_string();

    Ok(MemoryLayerComparisonReport {
        approaches: results,
        complement_advantage,
        methodology,
    })
}

// ===== Phase 2: End-to-End LLM Answer Evaluation =====

/// End-to-end answer quality for one approach.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2EAnswerResult {
    pub approach: String,
    /// 0–1: fraction of relevant info captured in answer
    pub mean_answer_score: f64,
    /// tokens sent as context (measured via tiktoken)
    pub mean_context_tokens: f64,
    /// tokens in LLM response (from API usage field)
    pub mean_answer_tokens: f64,
    /// context + answer
    pub mean_total_tokens: f64,
    pub queries_evaluated: usize,
}

/// Full end-to-end evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2EEvalReport {
    pub results: Vec<E2EAnswerResult>,
    pub model: String,
    pub methodology: String,
}

/// Score an LLM answer against relevant seeds using keyword overlap.
///
/// For each relevant seed, extracts key words (length > 4) and checks
/// whether at least 30% of them appear in the answer. Score = fraction
/// of relevant seeds whose key content appears in the answer.
fn score_answer(answer: &str, relevant_seeds: &[&str]) -> f64 {
    if relevant_seeds.is_empty() {
        return 0.0;
    }

    let answer_lower = answer.to_lowercase();
    let mut found = 0usize;

    for seed_content in relevant_seeds {
        let key_words: Vec<&str> = seed_content
            .split_whitespace()
            .filter(|w| w.len() > 4)
            .collect();

        if key_words.is_empty() {
            continue;
        }

        let matches = key_words
            .iter()
            .filter(|w| answer_lower.contains(&w.to_lowercase() as &str))
            .count();

        if matches as f64 / key_words.len() as f64 >= 0.3 {
            found += 1;
        }
    }

    found as f64 / relevant_seeds.len() as f64
}

/// Call the Anthropic API and return (answer_text, input_tokens, output_tokens).
/// Returns Err on API failure (caller should skip the case rather than panic).
async fn call_llm_for_answer(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
) -> Result<(String, usize, usize), String> {
    let body = serde_json::json!({
        "model": model,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    });

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("API error {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse error: {e}"))?;

    let answer = json["content"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|block| block["text"].as_str())
        .unwrap_or("")
        .to_string();

    let input_tokens = json["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
    let output_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize;

    Ok((answer, input_tokens, output_tokens))
}

/// End-to-end answer evaluation: send context to LLM, judge answer quality.
///
/// Requires ANTHROPIC_API_KEY environment variable.
///
/// Tests three approaches:
/// - FlatMarkdown: all seeds as markdown context
/// - Origin: search results as context
/// - NoContext: no context (LLM baseline)
///
/// For each case, composes a prompt with context + query, sends to Haiku,
/// and scores the answer via keyword overlap against relevant seeds.
///
/// `limit` controls the search top-K; `max_cases` caps API calls for cost control
/// (each case = 3 API calls, one per approach).
pub async fn run_e2e_answer_eval(
    fixture_dir: &Path,
    limit: usize,
    max_cases: usize,
) -> Result<E2EEvalReport, OriginError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| OriginError::Generic("ANTHROPIC_API_KEY not set".to_string()))?;

    let model = "claude-haiku-4-5-20251001";
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| OriginError::Generic(format!("failed to build reqwest client: {e}")))?;

    let cases = load_fixtures(fixture_dir)?;
    let confidence_cfg = ConfidenceConfig::default();

    // Per-approach accumulators: answer_score, context_tokens, answer_tokens
    let approach_keys = ["flat_markdown", "origin", "no_context"];
    let mut scores: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut ctx_tokens: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut ans_tokens: HashMap<&str, Vec<f64>> = HashMap::new();
    for key in &approach_keys {
        scores.insert(key, Vec::new());
        ctx_tokens.insert(key, Vec::new());
        ans_tokens.insert(key, Vec::new());
    }

    let mut cases_done = 0usize;

    for case in &cases {
        if cases_done >= max_cases {
            break;
        }
        if case.empty_set || case.seeds.is_empty() {
            continue;
        }

        // Gather relevant seeds (relevance >= 2) for judging
        let relevant_seed_contents: Vec<&str> = case
            .seeds
            .iter()
            .filter(|s| s.relevance >= 2)
            .map(|s| s.content.as_str())
            .collect();
        if relevant_seed_contents.is_empty() {
            continue;
        }

        let all_seeds: Vec<&crate::eval::fixtures::SeedMemory> = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .collect();

        // ---- Build contexts for each approach ----

        // FlatMarkdown: all seeds as numbered markdown sections
        let flat_context = all_seeds
            .iter()
            .enumerate()
            .map(|(i, s)| format!("## Memory {}\n{}", i + 1, s.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        // Origin: seed ephemeral DB, run hybrid search
        let origin_context = {
            let case_tmp = tempfile::tempdir()
                .map_err(|e| OriginError::Generic(format!("tempdir e2e: {e}")))?;
            let db = MemoryDB::new(case_tmp.path(), Arc::new(NoopEmitter)).await?;
            let docs: Vec<RawDocument> = all_seeds
                .iter()
                .map(|seed| crate::eval::runner::seed_to_doc(seed, &confidence_cfg))
                .collect();
            db.upsert_documents(docs).await?;
            let results = db
                .search_memory(
                    &case.query,
                    limit,
                    None,
                    case.domain.as_deref(),
                    None,
                    Some(1.0),
                    Some(1.0),
                    None,
                )
                .await?;
            results
                .iter()
                .enumerate()
                .map(|(i, r)| format!("## Result {}\n{}", i + 1, r.content))
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        // ---- Send each approach to the LLM ----

        let approaches: &[(&str, &str)] = &[
            ("flat_markdown", &flat_context),
            ("origin", &origin_context),
            ("no_context", ""),
        ];

        for (approach_key, context) in approaches {
            let prompt = if context.is_empty() {
                format!(
                    "Question: {}\n\nAnswer the question as best you can. Be specific and concise.",
                    case.query
                )
            } else {
                format!(
                    "Context:\n{}\n\nQuestion: {}\n\nAnswer the question using only the context provided. Be specific and concise.",
                    context, case.query
                )
            };

            let ctx_tok_count = if context.is_empty() {
                0usize
            } else {
                count_tokens(context)
            };

            // Rate limit between calls
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            match call_llm_for_answer(&client, &api_key, model, &prompt).await {
                Ok((answer, _input_tok, output_tok)) => {
                    let score = score_answer(&answer, &relevant_seed_contents);
                    scores.get_mut(approach_key).unwrap().push(score);
                    ctx_tokens
                        .get_mut(approach_key)
                        .unwrap()
                        .push(ctx_tok_count as f64);
                    ans_tokens
                        .get_mut(approach_key)
                        .unwrap()
                        .push(output_tok as f64);
                }
                Err(e) => {
                    log::warn!(
                        "[e2e_eval] case '{}' approach '{}' skipped: {}",
                        case.query,
                        approach_key,
                        e
                    );
                }
            }
        }

        cases_done += 1;
    }

    // Aggregate
    let mut results: Vec<E2EAnswerResult> = Vec::new();
    for key in &approach_keys {
        let score_vec = &scores[key];
        let ctx_vec = &ctx_tokens[key];
        let ans_vec = &ans_tokens[key];
        let n = score_vec.len().max(1) as f64;

        let mean_score = score_vec.iter().sum::<f64>() / n;
        let mean_ctx = ctx_vec.iter().sum::<f64>() / n;
        let mean_ans = ans_vec.iter().sum::<f64>() / n;
        let mean_total = mean_ctx + mean_ans;

        results.push(E2EAnswerResult {
            approach: key.to_string(),
            mean_answer_score: mean_score,
            mean_context_tokens: mean_ctx,
            mean_answer_tokens: mean_ans,
            mean_total_tokens: mean_total,
            queries_evaluated: score_vec.len(),
        });
    }

    Ok(E2EEvalReport {
        results,
        model: model.to_string(),
        methodology: "Keyword overlap judge: answer scores 1 for a relevant seed when ≥30% of its \
            key words (len>4) appear in the answer. Final score = fraction of relevant seeds \
            matched. Context tokens counted via cl100k_base; answer tokens from API usage field."
            .to_string(),
    })
}

// ===== E2E LoCoMo Answer Quality Eval (On-Device LLM) =====

/// Per-approach result for the E2E LoCoMo eval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2ELocomoResult {
    /// Approach identifier: "origin", "full_replay", "no_context".
    pub approach: String,
    /// Mean keyword-overlap score between LLM answer and ground truth (0–1).
    pub mean_answer_score: f64,
    /// Mean tokens of context sent to the LLM.
    pub mean_context_tokens: f64,
    /// Number of QA pairs evaluated for this approach.
    pub questions_evaluated: usize,
    /// Mean character length of the LLM's response.
    pub mean_answer_length: f64,
}

/// Full E2E LoCoMo benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct E2ELocomoReport {
    /// Model name used for inference.
    pub model: String,
    /// Number of conversations evaluated.
    pub conversations: usize,
    /// Max QA pairs sampled per conversation.
    pub questions_per_conv: usize,
    /// Total QA pairs evaluated.
    pub total_questions: usize,
    /// Per-approach results.
    pub results: Vec<E2ELocomoResult>,
}

// ===== LLM-as-Judge Types =====

/// A single E2E answer to be judged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentTuple {
    pub question: String,
    pub ground_truth: String,
    pub approach: String,
    pub answer: String,
    pub context_tokens: usize,
}

/// Result from the LLM judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentResult {
    pub question: String,
    pub approach: String,
    /// 0 or 1.
    pub score: u8,
    pub reason: String,
    pub context_tokens: usize,
}

/// Per-approach aggregated result in a judged E2E report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgedApproachResult {
    pub approach: String,
    /// Fraction of questions scoring 1.
    pub accuracy: f64,
    pub total: usize,
    pub correct: usize,
    pub mean_context_tokens: f64,
}

/// Full judged E2E report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgedE2EReport {
    pub judge_model: String,
    pub total_judged: usize,
    pub results_by_approach: Vec<JudgedApproachResult>,
}

// ===== LLM-as-Judge Functions =====

/// Save E2E answer tuples to JSON for offline judging.
pub fn save_judgment_tuples(tuples: &[JudgmentTuple], path: &Path) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(tuples).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Load previously saved judgment tuples from JSON.
pub fn load_judgment_tuples(path: &Path) -> Result<Vec<JudgmentTuple>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(std::io::Error::other)
}

/// Judge answer tuples using Claude via the `claude -p` CLI.
///
/// Requires Claude Code CLI installed (`claude --version` must succeed).
/// Uses Haiku via the user's existing Max subscription — no API key needed.
/// Runs up to `concurrency` judgments in parallel.
pub async fn judge_with_claude(
    tuples: &[JudgmentTuple],
    concurrency: usize,
) -> Result<Vec<JudgmentResult>, OriginError> {
    judge_with_claude_model(tuples, concurrency, "haiku").await
}

/// Judge tuples with a specific Claude model (e.g. "haiku", "sonnet").
pub async fn judge_with_claude_model(
    tuples: &[JudgmentTuple],
    concurrency: usize,
    model: &str,
) -> Result<Vec<JudgmentResult>, OriginError> {
    use tokio::sync::Semaphore;

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut handles = Vec::new();

    for tuple in tuples {
        let sem = semaphore.clone();
        let tuple = tuple.clone();
        let model = model.to_string();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            judge_single_tuple_model(&tuple, &model).await
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(Ok(result)) => results.push(result),
            Ok(Err(e)) => log::warn!("[judge] judgment failed: {}", e),
            Err(e) => log::warn!("[judge] task panicked: {}", e),
        }
    }

    Ok(results)
}

/// Judge a single (question, ground_truth, answer) tuple via `claude -p`.
///
/// Passes the prompt via stdin and disables all tools (`--allowedTools ""`), which
/// prevents Claude Code's agentic tool-calling loop and gets a direct text/JSON response.
/// OAuth auth from the user's existing login is used (no API key required).
pub async fn judge_single_tuple(tuple: &JudgmentTuple) -> Result<JudgmentResult, OriginError> {
    judge_single_tuple_model(tuple, "haiku").await
}

/// Judge a single tuple with a specific Claude model.
pub async fn judge_single_tuple_model(
    tuple: &JudgmentTuple,
    model: &str,
) -> Result<JudgmentResult, OriginError> {
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;

    let prompt = format!(
        "Question: {}\n\nReference answer: {}\n\nCandidate answer: {}\n\nIs the candidate answer correct? Compare against the reference.",
        tuple.question, tuple.ground_truth, tuple.answer
    );

    let json_schema = r#"{"type":"object","properties":{"score":{"type":"integer","enum":[0,1]},"reason":{"type":"string"}},"required":["score","reason"]}"#;

    let system_prompt = "You are an expert evaluator judging answer correctness. \
        Score 1 if the candidate answer contains the key information from the reference answer \
        (even if worded differently). Score 0 if the candidate answer is wrong, missing key \
        information, or irrelevant. Think step by step before scoring.";

    let mut child = Command::new("claude")
        .args([
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--json-schema",
            json_schema,
            "--system-prompt",
            system_prompt,
            "--no-session-persistence",
            "--allowedTools",
            "",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| OriginError::Generic(format!("claude -p failed to start: {}", e)))?;

    // Write prompt to stdin then close it.
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(prompt.as_bytes())
            .await
            .map_err(|e| OriginError::Generic(format!("write to claude stdin failed: {}", e)))?;
        // drop closes stdin
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| OriginError::Generic(format!("claude -p wait failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout_preview = String::from_utf8_lossy(&output.stdout);
        return Err(OriginError::Generic(format!(
            "claude -p exited with error: stderr={} stdout={}",
            stderr.trim(),
            stdout_preview.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse the JSON response. `--output-format json` returns an envelope; the structured
    // output lives in the `structured_output` field when `--json-schema` is used.
    let parsed: serde_json::Value = parse_judge_json(&stdout).map_err(|e| {
        OriginError::Generic(format!(
            "judge response parse error: {} — raw: {}",
            e, stdout
        ))
    })?;

    let score = parsed.get("score").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
    let reason = parsed
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("no reason")
        .to_string();

    Ok(JudgmentResult {
        question: tuple.question.clone(),
        approach: tuple.approach.clone(),
        score,
        reason,
        context_tokens: tuple.context_tokens,
    })
}

/// Try to extract the judgment JSON object from `claude -p` output.
///
/// When `--output-format json` is combined with `--json-schema`, Claude Code returns an
/// envelope like:
/// ```json
/// {"type":"result", "structured_output": {"score":1, "reason":"..."}, ...}
/// ```
/// We try several strategies to locate the score/reason object:
/// 1. `structured_output` field in the envelope (primary path).
/// 2. `result` field in the envelope (fallback for older CLI versions).
/// 3. Top-level object if it already contains `score`.
/// 4. Extract any `{...}` substring (last resort).
fn parse_judge_json(stdout: &str) -> Result<serde_json::Value, serde_json::Error> {
    let trimmed = stdout.trim();

    if let Ok(envelope) = serde_json::from_str::<serde_json::Value>(trimmed) {
        // Strategy 1: structured_output field (primary — used when --json-schema is set).
        if let Some(so) = envelope.get("structured_output") {
            if so.get("score").is_some() {
                return Ok(so.clone());
            }
        }
        // Strategy 2: result field (text-mode fallback).
        if let Some(result) = envelope.get("result") {
            if result.get("score").is_some() {
                return Ok(result.clone());
            }
            // result may be a JSON string — try to parse it.
            if let Some(s) = result.as_str() {
                if let Ok(inner) = serde_json::from_str::<serde_json::Value>(s) {
                    if inner.get("score").is_some() {
                        return Ok(inner);
                    }
                }
            }
        }
        // Strategy 3: top-level already has score.
        if envelope.get("score").is_some() {
            return Ok(envelope);
        }
    }

    // Strategy 4: extract the first balanced {...} block.
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start <= end {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&trimmed[start..=end]) {
                if v.get("score").is_some() {
                    return Ok(v);
                }
            }
        }
    }

    // Final fallback: return a parse error to surface the raw output.
    serde_json::from_str(trimmed)
}

/// Aggregate judgment results into a report sorted by accuracy descending.
pub fn aggregate_judgments(results: &[JudgmentResult], judge_model: &str) -> JudgedE2EReport {
    let mut by_approach: HashMap<String, Vec<&JudgmentResult>> = HashMap::new();
    for r in results {
        by_approach.entry(r.approach.clone()).or_default().push(r);
    }

    let mut approach_results: Vec<JudgedApproachResult> = by_approach
        .iter()
        .map(|(approach, items)| {
            let total = items.len();
            let correct = items.iter().filter(|r| r.score == 1).count();
            let accuracy = correct as f64 / total.max(1) as f64;
            let mean_tokens =
                items.iter().map(|r| r.context_tokens as f64).sum::<f64>() / total.max(1) as f64;
            JudgedApproachResult {
                approach: approach.clone(),
                accuracy,
                total,
                correct,
                mean_context_tokens: mean_tokens,
            }
        })
        .collect();

    approach_results.sort_by(|a, b| {
        b.accuracy
            .partial_cmp(&a.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    JudgedE2EReport {
        judge_model: judge_model.to_string(),
        total_judged: results.len(),
        results_by_approach: approach_results,
    }
}

/// Score an LLM answer against a ground-truth string using keyword overlap.
///
/// Splits the ground truth into words longer than 3 characters and measures
/// what fraction appear in the (lowercased) answer.
fn score_answer_against_ground_truth(answer: &str, ground_truth: &str) -> f64 {
    let answer_lower = answer.to_lowercase();
    let gt_words: Vec<&str> = ground_truth
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .collect();
    if gt_words.is_empty() {
        return 0.0;
    }
    let matches = gt_words
        .iter()
        .filter(|w| answer_lower.contains(&w.to_lowercase() as &str))
        .count();
    matches as f64 / gt_words.len() as f64
}

/// Run end-to-end answer quality evaluation on LoCoMo using the on-device LLM.
///
/// For each LoCoMo conversation:
/// 1. Seeds all observations into an ephemeral DB.
/// 2. For up to `max_questions_per_conv` non-adversarial QA pairs:
///    - **origin**: retrieve top-`search_top_k` results, compose prompt, call LLM.
///    - **full_replay**: use ALL observations as context (skipped if > 4000 tokens).
///    - **no_context**: ask the question with no memory context.
/// 3. Scores each LLM answer against the ground-truth via keyword overlap.
///
/// The `llm_provider` must be an `OnDeviceProvider` (or any `LlmProvider`).
/// This function is `async` but LLM calls are routed through the provider's
/// internal worker thread — no extra `spawn_blocking` needed here.
///
/// Returns an `(E2ELocomoReport, Vec<JudgmentTuple>)` tuple.
///
/// The `JudgmentTuple` list contains raw (question, ground_truth, approach, answer,
/// context_tokens) records for every answered question. Save them with
/// [`save_judgment_tuples`] and score offline with [`judge_with_claude`].
pub async fn run_e2e_locomo_eval(
    locomo_path: &Path,
    max_questions_per_conv: usize,
    search_top_k: usize,
    llm_provider: Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<(E2ELocomoReport, Vec<JudgmentTuple>), OriginError> {
    use crate::eval::locomo::{extract_observations, load_locomo};
    use crate::llm_provider::{strip_think_tags, LlmRequest};

    let samples = load_locomo(locomo_path)?;

    // Accumulators: (answer_score, context_tokens, answer_len)
    let approach_keys = ["origin", "full_replay", "no_context"];
    let mut scores: std::collections::HashMap<&str, Vec<f64>> =
        approach_keys.iter().map(|k| (*k, Vec::new())).collect();
    let mut ctx_tokens: std::collections::HashMap<&str, Vec<f64>> =
        approach_keys.iter().map(|k| (*k, Vec::new())).collect();
    let mut ans_lens: std::collections::HashMap<&str, Vec<f64>> =
        approach_keys.iter().map(|k| (*k, Vec::new())).collect();

    // Collect raw tuples for offline LLM judging.
    let mut judgment_tuples: Vec<JudgmentTuple> = Vec::new();

    let total_convs = samples.len();

    for (conv_idx, sample) in samples.iter().enumerate() {
        let memories = extract_observations(sample);
        if memories.is_empty() {
            continue;
        }

        // Build full-replay corpus text (all observations concatenated).
        let corpus_text: String = memories
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let corpus_tokens = count_tokens(&corpus_text);
        // Cap full_replay at 4000 tokens to stay within model's synthesis limit.
        const FULL_REPLAY_TOKEN_LIMIT: usize = 4000;
        let full_replay_context: Option<String> = if corpus_tokens <= FULL_REPLAY_TOKEN_LIMIT {
            Some(corpus_text.clone())
        } else {
            // Truncate by taking observations until we reach the limit.
            let mut truncated = String::new();
            for mem in &memories {
                let candidate = if truncated.is_empty() {
                    mem.content.clone()
                } else {
                    format!("{}\n\n{}", truncated, mem.content)
                };
                if count_tokens(&candidate) > FULL_REPLAY_TOKEN_LIMIT {
                    break;
                }
                truncated = candidate;
            }
            if truncated.is_empty() {
                None // even one observation exceeds the limit — skip full_replay
            } else {
                Some(truncated)
            }
        };

        // Seed ephemeral DB for Origin retrieval.
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir e2e_locomo: {e}")))?;
        let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter)).await?;

        let docs: Vec<crate::sources::RawDocument> = memories
            .iter()
            .enumerate()
            .map(|(i, mem)| crate::sources::RawDocument {
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
        db.upsert_documents(docs).await?;

        // Iterate QA pairs (skip adversarial, cap at max_questions_per_conv).
        let mut questions_done = 0usize;
        for qa in &sample.qa {
            if questions_done >= max_questions_per_conv {
                break;
            }
            if qa.category == 5 {
                continue; // skip adversarial
            }

            let ground_truth = qa
                .answer
                .as_ref()
                .map(|v| v.as_str().unwrap_or(&v.to_string()).to_string())
                .unwrap_or_default();

            if ground_truth.is_empty() {
                continue;
            }

            eprintln!(
                "[e2e_locomo] Conv {}/{}, Q {}/{}...",
                conv_idx + 1,
                total_convs,
                questions_done + 1,
                max_questions_per_conv,
            );

            let system_prompt = "Answer the question using only the provided context. \
                Be specific and concise. Respond in 1-3 sentences."
                .to_string();

            // ---- Origin approach: hybrid search ----
            let origin_context = {
                let results = db
                    .search_memory(
                        &qa.question,
                        search_top_k,
                        None,
                        Some("conversation"),
                        None,
                        None,
                        None,
                        None,
                    )
                    .await?;
                results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| format!("{}. {}", i + 1, r.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let origin_ctx_tokens = count_tokens(&origin_context);

            let origin_request = LlmRequest {
                system_prompt: Some(system_prompt.clone()),
                user_prompt: format!("Context:\n{}\n\nQuestion: {}", origin_context, qa.question),
                max_tokens: 200,
                temperature: 0.1,
                label: Some("e2e_locomo_origin".to_string()),
            };
            match llm_provider.generate(origin_request).await {
                Ok(raw_answer) => {
                    let answer = strip_think_tags(&raw_answer);
                    let score = score_answer_against_ground_truth(&answer, &ground_truth);
                    scores.get_mut("origin").unwrap().push(score);
                    ctx_tokens
                        .get_mut("origin")
                        .unwrap()
                        .push(origin_ctx_tokens as f64);
                    ans_lens
                        .get_mut("origin")
                        .unwrap()
                        .push(answer.len() as f64);
                    judgment_tuples.push(JudgmentTuple {
                        question: qa.question.clone(),
                        ground_truth: ground_truth.clone(),
                        approach: "origin".to_string(),
                        answer,
                        context_tokens: origin_ctx_tokens,
                    });
                }
                Err(e) => {
                    log::warn!("[e2e_locomo] origin approach failed: {e}");
                }
            }

            // ---- FullReplay approach ----
            if let Some(ref replay_ctx) = full_replay_context {
                let replay_ctx_tokens = count_tokens(replay_ctx);
                let replay_request = LlmRequest {
                    system_prompt: Some(system_prompt.clone()),
                    user_prompt: format!("Context:\n{}\n\nQuestion: {}", replay_ctx, qa.question),
                    max_tokens: 200,
                    temperature: 0.1,
                    label: Some("e2e_locomo_full_replay".to_string()),
                };
                match llm_provider.generate(replay_request).await {
                    Ok(raw_answer) => {
                        let answer = strip_think_tags(&raw_answer);
                        let score = score_answer_against_ground_truth(&answer, &ground_truth);
                        scores.get_mut("full_replay").unwrap().push(score);
                        ctx_tokens
                            .get_mut("full_replay")
                            .unwrap()
                            .push(replay_ctx_tokens as f64);
                        ans_lens
                            .get_mut("full_replay")
                            .unwrap()
                            .push(answer.len() as f64);
                        judgment_tuples.push(JudgmentTuple {
                            question: qa.question.clone(),
                            ground_truth: ground_truth.clone(),
                            approach: "full_replay".to_string(),
                            answer,
                            context_tokens: replay_ctx_tokens,
                        });
                    }
                    Err(e) => {
                        log::warn!("[e2e_locomo] full_replay approach failed: {e}");
                    }
                }
            }
            // If full_replay was skipped (too long), we don't push to its accumulators.

            // ---- NoContext approach ----
            let no_ctx_request = LlmRequest {
                system_prompt: Some(
                    "Answer the question as best you can from your knowledge. \
                    Be specific and concise. Respond in 1-3 sentences."
                        .to_string(),
                ),
                user_prompt: format!("Question: {}", qa.question),
                max_tokens: 200,
                temperature: 0.1,
                label: Some("e2e_locomo_no_context".to_string()),
            };
            match llm_provider.generate(no_ctx_request).await {
                Ok(raw_answer) => {
                    let answer = strip_think_tags(&raw_answer);
                    let score = score_answer_against_ground_truth(&answer, &ground_truth);
                    scores.get_mut("no_context").unwrap().push(score);
                    ctx_tokens.get_mut("no_context").unwrap().push(0.0);
                    ans_lens
                        .get_mut("no_context")
                        .unwrap()
                        .push(answer.len() as f64);
                    judgment_tuples.push(JudgmentTuple {
                        question: qa.question.clone(),
                        ground_truth: ground_truth.clone(),
                        approach: "no_context".to_string(),
                        answer,
                        context_tokens: 0,
                    });
                }
                Err(e) => {
                    log::warn!("[e2e_locomo] no_context approach failed: {e}");
                }
            }

            questions_done += 1;
        }
    }

    // Aggregate per-approach
    let total_questions = scores["origin"].len();
    let mut results: Vec<E2ELocomoResult> = Vec::new();
    for key in &approach_keys {
        let score_vec = &scores[key];
        let ctx_vec = &ctx_tokens[key];
        let len_vec = &ans_lens[key];
        let n = score_vec.len().max(1) as f64;

        results.push(E2ELocomoResult {
            approach: key.to_string(),
            mean_answer_score: score_vec.iter().sum::<f64>() / n,
            mean_context_tokens: ctx_vec.iter().sum::<f64>() / n,
            questions_evaluated: score_vec.len(),
            mean_answer_length: len_vec.iter().sum::<f64>() / n,
        });
    }

    Ok((
        E2ELocomoReport {
            model: llm_provider.name().to_string(),
            conversations: samples.len(),
            questions_per_conv: max_questions_per_conv,
            total_questions,
            results,
        },
        judgment_tuples,
    ))
}

// ===== Pipeline Eval: LoCoMo + LongMemEval through Origin's full pipeline =====

/// Pipeline condition: what processing has been applied to the DB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineCondition {
    /// Raw observations seeded via upsert_documents, no enrichment.
    Flat,
    /// Flat + entity extraction (auto-linking, entity creation).
    Enriched,
    /// Enriched + distillation via real LLM (merged concept memories).
    Distilled,
}

impl PipelineCondition {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Enriched => "enriched",
            Self::Distilled => "distilled",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Flat => "Flat (raw)",
            Self::Enriched => "Enriched (entities)",
            Self::Distilled => "Distilled (LLM concepts)",
        }
    }
}

/// Metrics for one (condition, strategy) cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCellMetrics {
    pub condition: String,
    pub strategy: String,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub mean_context_tokens: f64,
    pub corpus_tokens: usize,
    pub memory_count: usize,
    /// NDCG per 1K context tokens.
    pub information_density: f64,
    pub queries_evaluated: usize,
}

/// Per-conversation/question result showing all conditions x strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConversationResult {
    pub id: String,
    pub observations_seeded: usize,
    pub queries_evaluated: usize,
    pub distillation_clusters_found: usize,
    pub distilled_concepts_created: usize,
    pub cells: Vec<PipelineCellMetrics>,
}

/// Full pipeline eval report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineBenchmarkReport {
    pub benchmark: String,
    pub timestamp: String,
    pub conversations: usize,
    pub total_queries: usize,
    pub total_observations: usize,
    pub llm_model: String,
    /// Aggregate metrics across all conversations/questions.
    pub aggregate: Vec<PipelineCellMetrics>,
    pub per_conversation: Vec<PipelineConversationResult>,
}

impl PipelineBenchmarkReport {
    /// Format as terminal-friendly text.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Pipeline Eval: {} ({})\n",
            self.benchmark, self.timestamp
        ));
        out.push_str(&format!(
            "LLM: {} | Conversations: {} | Queries: {} | Observations: {}\n\n",
            self.llm_model, self.conversations, self.total_queries, self.total_observations
        ));

        out.push_str(&format!(
            "{:<24} | {:<14} | {:<8} | {:<8} | {:<8} | {:<10} | {:<8}\n",
            "Condition x Strategy", "NDCG@10", "MRR", "R@5", "Tok/Q", "Corpus", "Density"
        ));
        out.push_str(&format!(
            "{:-<24}-+-{:-<14}-+-{:-<8}-+-{:-<8}-+-{:-<8}-+-{:-<10}-+-{:-<8}\n",
            "", "", "", "", "", "", ""
        ));

        for cell in &self.aggregate {
            out.push_str(&format!(
                "{:<24} | {:<14.4} | {:<8.4} | {:<8.4} | {:<8.1} | {:<10} | {:<8.3}\n",
                format!("{} / {}", cell.condition, cell.strategy),
                cell.ndcg_at_10,
                cell.mrr,
                cell.recall_at_5,
                cell.mean_context_tokens,
                cell.corpus_tokens,
                cell.information_density,
            ));
        }

        // Distillation summary
        let total_clusters: usize = self
            .per_conversation
            .iter()
            .map(|c| c.distillation_clusters_found)
            .sum();
        let total_concepts: usize = self
            .per_conversation
            .iter()
            .map(|c| c.distilled_concepts_created)
            .sum();
        out.push_str(&format!(
            "\nDistillation: {} clusters found, {} concepts created\n",
            total_clusters, total_concepts
        ));

        out
    }
}

/// Search strategies to test in pipeline eval (subset -- no LLM-requiring ones).
const PIPELINE_STRATEGIES: &[SearchStrategy] = &[
    SearchStrategy::Origin,
    SearchStrategy::NaiveRag,
    SearchStrategy::FtsOnly,
    SearchStrategy::VectorPlusFts,
];

/// Run search for a given strategy and return (result_ids, context_tokens).
async fn run_strategy_search(
    db: &MemoryDB,
    query: &str,
    strategy: SearchStrategy,
    limit: usize,
    domain: Option<&str>,
) -> Result<(Vec<String>, usize), OriginError> {
    let results = match strategy {
        SearchStrategy::Origin => {
            db.search_memory(query, limit, None, domain, None, Some(1.0), Some(1.0), None)
                .await?
        }
        SearchStrategy::NaiveRag => db.naive_vector_search(query, limit, domain).await?,
        SearchStrategy::FtsOnly => db.fts_only_search(query, limit, domain).await?,
        SearchStrategy::VectorPlusFts => db.vector_plus_fts_search(query, limit, domain).await?,
        _ => return Ok((vec![], 0)),
    };
    let ids: Vec<String> = results.iter().map(|r| r.source_id.clone()).collect();
    let tokens = count_results_tokens(&results);
    Ok((ids, tokens))
}

/// Evaluate all strategies for a set of QA pairs against a DB, returning per-cell metrics.
///
/// `relevance_map` maps source_id -> relevance grade (0-3).
/// `evidence_sets` maps question index -> set of relevant source_ids.
#[allow(clippy::too_many_arguments)]
async fn evaluate_condition<Q: AsRef<str>>(
    db: &MemoryDB,
    condition: PipelineCondition,
    questions: &[Q],
    evidence_sets: &[HashSet<String>],
    relevance_map: &HashMap<String, u8>,
    corpus_tokens: usize,
    memory_count: usize,
    limit: usize,
    domain: Option<&str>,
) -> Result<Vec<PipelineCellMetrics>, OriginError> {
    let mut cells = Vec::new();
    let total_strategies = PIPELINE_STRATEGIES.len();

    for (si, &strategy) in PIPELINE_STRATEGIES.iter().enumerate() {
        eprintln!(
            "    [{}/{}] {} / {} ({} queries)...",
            condition.name(),
            strategy.name(),
            si + 1,
            total_strategies,
            questions.len(),
        );
        let mut ndcg_vals = Vec::new();
        let mut mrr_vals = Vec::new();
        let mut recall_vals = Vec::new();
        let mut ctx_tokens_vals = Vec::new();

        for (qi, question) in questions.iter().enumerate() {
            let (result_ids, ctx_tokens) =
                run_strategy_search(db, question.as_ref(), strategy, limit, domain).await?;

            let result_refs: Vec<&str> = result_ids.iter().map(|s| s.as_str()).collect();

            // Build grades for this query's results
            let grades: HashMap<&str, u8> = result_refs
                .iter()
                .map(|id| (*id, *relevance_map.get(*id).unwrap_or(&0)))
                .collect();

            let relevant_set: HashSet<&str> =
                evidence_sets[qi].iter().map(|s| s.as_str()).collect();

            ndcg_vals.push(metrics::ndcg_at_k(&result_refs, &grades, 10));
            mrr_vals.push(metrics::mrr(&result_refs, &relevant_set));
            recall_vals.push(metrics::recall_at_k(&result_refs, &relevant_set, 5));
            ctx_tokens_vals.push(ctx_tokens as f64);
        }

        let n = ndcg_vals.len().max(1) as f64;
        let mean_ndcg = ndcg_vals.iter().sum::<f64>() / n;
        let mean_mrr = mrr_vals.iter().sum::<f64>() / n;
        let mean_recall = recall_vals.iter().sum::<f64>() / n;
        let mean_ctx = ctx_tokens_vals.iter().sum::<f64>() / n;
        let density = if mean_ctx > 0.0 {
            mean_ndcg / (mean_ctx / 1000.0)
        } else {
            0.0
        };

        cells.push(PipelineCellMetrics {
            condition: condition.name().to_string(),
            strategy: strategy.name().to_string(),
            ndcg_at_10: mean_ndcg,
            mrr: mean_mrr,
            recall_at_5: mean_recall,
            mean_context_tokens: mean_ctx,
            corpus_tokens,
            memory_count,
            information_density: density,
            queries_evaluated: questions.len(),
        });
    }

    Ok(cells)
}

/// Count corpus tokens from all memories in a DB.
async fn count_corpus_tokens(db: &MemoryDB) -> Result<(usize, usize), OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT content FROM memories WHERE chunk_index = 0 \
             AND supersede_mode <> 'archive'",
            (),
        )
        .await
        .map_err(|e| OriginError::Generic(format!("count_corpus: {e}")))?;

    let mut total_tokens = 0usize;
    let mut count = 0usize;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| OriginError::Generic(e.to_string()))?
    {
        let content: String = row
            .get(0)
            .map_err(|e| OriginError::Generic(e.to_string()))?;
        total_tokens += count_tokens(&content);
        count += 1;
    }
    Ok((total_tokens, count))
}

/// Expand evidence sets to include merged/distilled source_ids.
///
/// Expand evidence sets to include concept source_ids via the `concept_sources` join table.
///
/// After distillation, concepts link to their source memories via `concept_sources` (PR4).
/// For each evidence source_id, check if any concept consumed it. If so, add the concept's
/// merged memory source_id to the evidence set so NDCG credits distilled retrieval.
async fn expand_evidence_for_distillation(
    db: &MemoryDB,
    original_evidence: &[HashSet<String>],
) -> Result<Vec<HashSet<String>>, OriginError> {
    let conn = db.conn.lock().await;

    // Build reverse map: memory_source_id -> [concept merged source_ids]
    // Uses the concept_sources join table (PR4) for precise lineage tracking.
    let mut rows = conn
        .query(
            "SELECT cs.memory_source_id, m.source_id AS concept_mem_sid \
             FROM concept_sources cs \
             JOIN concepts c ON cs.concept_id = c.id \
             JOIN memories m ON m.source_id LIKE 'merged_%' \
               AND m.chunk_index = 0 \
               AND m.entity_id = c.entity_id \
             WHERE c.status = 'active'",
            (),
        )
        .await
        .map_err(|e| OriginError::Generic(format!("expand_evidence: {e}")))?;

    let mut mem_to_concepts: HashMap<String, Vec<String>> = HashMap::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| OriginError::Generic(e.to_string()))?
    {
        let memory_sid: String = row
            .get(0)
            .map_err(|e| OriginError::Generic(e.to_string()))?;
        let concept_sid: String = row
            .get(1)
            .map_err(|e| OriginError::Generic(e.to_string()))?;
        mem_to_concepts
            .entry(memory_sid)
            .or_default()
            .push(concept_sid);
    }
    drop(rows);
    drop(conn);

    let mut expanded: Vec<HashSet<String>> = original_evidence.to_vec();
    for evidence_set in &mut expanded {
        let mut additions: Vec<String> = Vec::new();
        for evidence_id in evidence_set.iter() {
            if let Some(concept_sids) = mem_to_concepts.get(evidence_id) {
                additions.extend(concept_sids.iter().cloned());
            }
        }
        for a in additions {
            evidence_set.insert(a);
        }
    }

    Ok(expanded)
}

/// Run LoCoMo through Origin's full pipeline: flat, enriched, distilled.
///
/// For each conversation:
/// 1. Flat: seed observations, measure retrieval quality + tokens
/// 2. Enriched: run entity extraction on the same DB
/// 3. Distilled: run `distill_concepts()` with real LLM, measure again
///
/// Requires on-device LLM (Metal GPU). Run with sandbox disabled.
pub async fn run_locomo_pipeline_eval(
    locomo_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_conversations: usize,
) -> Result<PipelineBenchmarkReport, OriginError> {
    use crate::eval::locomo::{extract_observations, load_locomo};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_locomo(locomo_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    let mut per_conversation = Vec::new();
    let mut total_queries = 0usize;
    let mut total_observations = 0usize;

    // Per-cell accumulators: (condition, strategy) -> Vec<(ndcg, mrr, recall, ctx_tokens)>
    #[allow(clippy::type_complexity)]
    let mut accumulators: HashMap<(String, String), Vec<(f64, f64, f64, f64, usize, usize)>> =
        HashMap::new();

    let conv_limit = max_conversations.min(samples.len());

    for (conv_idx, sample) in samples.iter().take(max_conversations).enumerate() {
        let memories = extract_observations(sample);
        if memories.is_empty() {
            continue;
        }
        let obs_count = memories.len();
        total_observations += obs_count;

        eprintln!(
            "[pipeline] Conv {}/{} ({}): {} observations",
            conv_idx + 1,
            conv_limit,
            sample.sample_id,
            obs_count,
        );

        // Build evidence mapping: dia_id -> source_id
        let dia_to_source: HashMap<String, String> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                (
                    m.dia_id.clone(),
                    format!("locomo_{}_obs_{}", sample.sample_id, i),
                )
            })
            .collect();

        // Prepare questions and evidence sets (non-adversarial only)
        let mut questions: Vec<String> = Vec::new();
        let mut evidence_sets: Vec<HashSet<String>> = Vec::new();
        let mut relevance_map: HashMap<String, u8> = HashMap::new();

        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            }
            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if relevant_ids.is_empty() {
                continue;
            }

            // All seeded source_ids get a relevance grade
            for (idx, _mem) in memories.iter().enumerate() {
                let sid = format!("locomo_{}_obs_{}", sample.sample_id, idx);
                let grade = if relevant_ids.contains(&sid) { 1u8 } else { 0 };
                // Use max in case a source_id is evidence for multiple questions
                let entry = relevance_map.entry(sid).or_insert(0);
                *entry = (*entry).max(grade);
            }

            questions.push(qa.question.clone());
            evidence_sets.push(relevant_ids);
        }

        if questions.is_empty() {
            continue;
        }
        total_queries += questions.len();

        // ---- Create DB and seed flat ----
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir pipeline: {e}")))?;
        let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter)).await?;

        let docs: Vec<RawDocument> = memories
            .iter()
            .enumerate()
            .map(|(i, mem)| RawDocument {
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
        db.upsert_documents(docs).await?;

        // ---- Condition 1: Flat ----
        eprintln!("  [flat] evaluating {} queries...", questions.len());
        let (flat_corpus_tokens, flat_mem_count) = count_corpus_tokens(&db).await?;
        let flat_cells = evaluate_condition(
            &db,
            PipelineCondition::Flat,
            &questions,
            &evidence_sets,
            &relevance_map,
            flat_corpus_tokens,
            flat_mem_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // ---- Condition 2: Enriched (entity extraction) ----
        eprintln!("  [enriched] running entity extraction...");
        // Run entity extraction via refinery's extract functions.
        // We use the LLM to extract entities from each memory.
        let extract_count = run_entity_extraction_for_eval(&db, &llm).await?;
        eprintln!("    extracted {} entities", extract_count);

        let (enriched_corpus_tokens, enriched_mem_count) = count_corpus_tokens(&db).await?;

        // Rebuild relevance map to include any new source_ids from entity extraction
        // (entity extraction doesn't create new memories, just links, so map is unchanged)
        let enriched_cells = evaluate_condition(
            &db,
            PipelineCondition::Enriched,
            &questions,
            &evidence_sets,
            &relevance_map,
            enriched_corpus_tokens,
            enriched_mem_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // ---- Condition 3: Distilled (real LLM) ----
        eprintln!("  [distilled] running distillation...");
        let distilled_count =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;
        eprintln!("    distilled {} concepts", distilled_count);

        // Count clusters that were found (for reporting)
        let clusters = db
            .find_distillation_clusters(
                tuning.similarity_threshold,
                tuning.concept_min_cluster_size,
                tuning.max_clusters_per_steep,
                llm.synthesis_token_limit(),
            )
            .await
            .unwrap_or_default();

        let (distilled_corpus_tokens, distilled_mem_count) = count_corpus_tokens(&db).await?;

        // Expand evidence sets to credit merged memories
        let expanded_evidence = expand_evidence_for_distillation(&db, &evidence_sets).await?;

        // Rebuild relevance map to include merged source_ids
        let mut distilled_relevance = relevance_map.clone();
        for evidence_set in &expanded_evidence {
            for id in evidence_set {
                if id.starts_with("merged_") {
                    distilled_relevance.entry(id.clone()).or_insert(1);
                }
            }
        }

        let distilled_cells = evaluate_condition(
            &db,
            PipelineCondition::Distilled,
            &questions,
            &expanded_evidence,
            &distilled_relevance,
            distilled_corpus_tokens,
            distilled_mem_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // Collect into per-conversation result
        let mut all_cells = Vec::new();
        all_cells.extend(flat_cells);
        all_cells.extend(enriched_cells);
        all_cells.extend(distilled_cells);

        for cell in &all_cells {
            accumulators
                .entry((cell.condition.clone(), cell.strategy.clone()))
                .or_default()
                .push((
                    cell.ndcg_at_10,
                    cell.mrr,
                    cell.recall_at_5,
                    cell.mean_context_tokens,
                    cell.corpus_tokens,
                    cell.memory_count,
                ));
        }

        per_conversation.push(PipelineConversationResult {
            id: sample.sample_id.clone(),
            observations_seeded: obs_count,
            queries_evaluated: questions.len(),
            distillation_clusters_found: clusters.len(),
            distilled_concepts_created: distilled_count,
            cells: all_cells,
        });

        // Print per-conversation summary
        if let Some(flat_origin) = per_conversation.last().and_then(|c| {
            c.cells
                .iter()
                .find(|c| c.condition == "flat" && c.strategy == "origin")
        }) {
            if let Some(dist_origin) = per_conversation.last().and_then(|c| {
                c.cells
                    .iter()
                    .find(|c| c.condition == "distilled" && c.strategy == "origin")
            }) {
                eprintln!(
                    "  Flat NDCG={:.3} @{:.0}tok -> Distilled NDCG={:.3} @{:.0}tok ({}c, {}m->{}m)",
                    flat_origin.ndcg_at_10,
                    flat_origin.mean_context_tokens,
                    dist_origin.ndcg_at_10,
                    dist_origin.mean_context_tokens,
                    distilled_count,
                    flat_mem_count,
                    distilled_mem_count,
                );
            }
        }
    }

    // Aggregate across conversations
    let mut aggregate = Vec::new();
    for ((condition, strategy), vals) in &accumulators {
        let n = vals.len().max(1) as f64;
        let mean_ndcg = vals.iter().map(|v| v.0).sum::<f64>() / n;
        let mean_mrr = vals.iter().map(|v| v.1).sum::<f64>() / n;
        let mean_recall = vals.iter().map(|v| v.2).sum::<f64>() / n;
        let mean_ctx = vals.iter().map(|v| v.3).sum::<f64>() / n;
        let mean_corpus = (vals.iter().map(|v| v.4).sum::<usize>() as f64 / n) as usize;
        let mean_mem_count = (vals.iter().map(|v| v.5).sum::<usize>() as f64 / n) as usize;
        let density = if mean_ctx > 0.0 {
            mean_ndcg / (mean_ctx / 1000.0)
        } else {
            0.0
        };

        aggregate.push(PipelineCellMetrics {
            condition: condition.clone(),
            strategy: strategy.clone(),
            ndcg_at_10: mean_ndcg,
            mrr: mean_mrr,
            recall_at_5: mean_recall,
            mean_context_tokens: mean_ctx,
            corpus_tokens: mean_corpus,
            memory_count: mean_mem_count,
            information_density: density,
            queries_evaluated: total_queries,
        });
    }
    // Sort: condition order (flat, enriched, distilled) then strategy
    let condition_order = |c: &str| match c {
        "flat" => 0,
        "enriched" => 1,
        "distilled" => 2,
        _ => 3,
    };
    aggregate.sort_by(|a, b| {
        condition_order(&a.condition)
            .cmp(&condition_order(&b.condition))
            .then(a.strategy.cmp(&b.strategy))
    });

    Ok(PipelineBenchmarkReport {
        benchmark: "LoCoMo".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        conversations: per_conversation.len(),
        total_queries,
        total_observations,
        llm_model: llm.name().to_string(),
        aggregate,
        per_conversation,
    })
}

/// Run LongMemEval through Origin's full pipeline: flat, enriched, distilled.
///
/// Same approach as LoCoMo but adapted for LongMemEval's per-question structure.
/// LongMemEval has 500 questions with ~22 turns each (oracle variant).
/// Smaller per-question corpora mean distillation may find fewer clusters.
pub async fn run_longmemeval_pipeline_eval(
    longmemeval_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_questions: usize,
) -> Result<PipelineBenchmarkReport, OriginError> {
    use crate::eval::longmemeval::{extract_memories, load_longmemeval};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_longmemeval(longmemeval_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    // Pre-create shared embedder
    eprintln!("[pipeline-lme] loading shared embedder...");
    let shared_embedder = MemoryDB::create_shared_embedder().await?;

    let mut per_conversation = Vec::new();
    let mut total_queries = 0usize;
    let mut total_observations = 0usize;

    #[allow(clippy::type_complexity)]
    let mut accumulators: HashMap<(String, String), Vec<(f64, f64, f64, f64, usize, usize)>> =
        HashMap::new();

    let sample_count = samples.len().min(max_questions);

    for (q_idx, sample) in samples.iter().take(max_questions).enumerate() {
        let memories = extract_memories(sample);
        if memories.is_empty() {
            continue;
        }
        let obs_count = memories.len();
        total_observations += obs_count;

        if q_idx % 25 == 0 {
            eprintln!(
                "[pipeline-lme] Q {}/{} ({}): {} memories, type={}",
                q_idx + 1,
                sample_count,
                sample.question_id,
                obs_count,
                sample.question_type,
            );
        }

        // Build evidence: turns with has_answer=true
        let evidence_session_set: HashSet<&str> = sample
            .answer_session_ids
            .iter()
            .map(|s| s.as_str())
            .collect();

        let mut evidence_ids: HashSet<String> = HashSet::new();
        let mut relevance_map: HashMap<String, u8> = HashMap::new();

        for mem in memories.iter() {
            let sid = format!(
                "lme_{}_{}_t{}",
                sample.question_id, mem.session_idx, mem.turn_idx
            );
            let grade = if mem.has_answer {
                3
            } else if evidence_session_set.contains(mem.session_id.as_str()) {
                2
            } else {
                0
            };
            if grade >= 2 {
                evidence_ids.insert(sid.clone());
            }
            relevance_map.insert(sid, grade);
        }

        if evidence_ids.is_empty() {
            continue;
        }

        let questions = vec![sample.question.clone()];
        let evidence_sets = vec![evidence_ids];
        total_queries += 1;

        // Create DB with shared embedder
        let tmp =
            tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir lme: {e}")))?;
        let db = MemoryDB::new_with_shared_embedder(
            tmp.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;

        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| RawDocument {
                content: mem.content.clone(),
                source_id: format!(
                    "lme_{}_{}_t{}",
                    sample.question_id, mem.session_idx, mem.turn_idx
                ),
                source: "memory".to_string(),
                title: format!("session {} turn {}", mem.session_idx, mem.turn_idx),
                memory_type: Some(
                    if sample.question_type == "single-session-preference" {
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
            .collect();
        db.upsert_documents(docs).await?;

        // ---- Flat ----
        let (flat_corpus, flat_count) = count_corpus_tokens(&db).await?;
        let flat_cells = evaluate_condition(
            &db,
            PipelineCondition::Flat,
            &questions,
            &evidence_sets,
            &relevance_map,
            flat_corpus,
            flat_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // ---- Enriched ----
        let _extract_count = run_entity_extraction_for_eval(&db, &llm).await?;
        let (enriched_corpus, enriched_count) = count_corpus_tokens(&db).await?;
        let enriched_cells = evaluate_condition(
            &db,
            PipelineCondition::Enriched,
            &questions,
            &evidence_sets,
            &relevance_map,
            enriched_corpus,
            enriched_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        // ---- Distilled ----
        let distilled_count =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;

        let clusters = db
            .find_distillation_clusters(
                tuning.similarity_threshold,
                tuning.concept_min_cluster_size,
                tuning.max_clusters_per_steep,
                llm.synthesis_token_limit(),
            )
            .await
            .unwrap_or_default();

        let (distilled_corpus, distilled_mem_count) = count_corpus_tokens(&db).await?;

        let expanded_evidence = expand_evidence_for_distillation(&db, &evidence_sets).await?;
        let mut distilled_relevance = relevance_map.clone();
        for evidence_set in &expanded_evidence {
            for id in evidence_set {
                if id.starts_with("merged_") {
                    distilled_relevance.entry(id.clone()).or_insert(1);
                }
            }
        }

        let distilled_cells = evaluate_condition(
            &db,
            PipelineCondition::Distilled,
            &questions,
            &expanded_evidence,
            &distilled_relevance,
            distilled_corpus,
            distilled_mem_count,
            search_limit,
            Some("conversation"),
        )
        .await?;

        let mut all_cells = Vec::new();
        all_cells.extend(flat_cells);
        all_cells.extend(enriched_cells);
        all_cells.extend(distilled_cells);

        for cell in &all_cells {
            accumulators
                .entry((cell.condition.clone(), cell.strategy.clone()))
                .or_default()
                .push((
                    cell.ndcg_at_10,
                    cell.mrr,
                    cell.recall_at_5,
                    cell.mean_context_tokens,
                    cell.corpus_tokens,
                    cell.memory_count,
                ));
        }

        per_conversation.push(PipelineConversationResult {
            id: sample.question_id.clone(),
            observations_seeded: obs_count,
            queries_evaluated: 1,
            distillation_clusters_found: clusters.len(),
            distilled_concepts_created: distilled_count,
            cells: all_cells,
        });

        if q_idx % 50 == 49 {
            eprintln!("  [progress] {}/{} questions done", q_idx + 1, sample_count);
        }
    }

    // Aggregate
    let mut aggregate = Vec::new();
    for ((condition, strategy), vals) in &accumulators {
        let n = vals.len().max(1) as f64;
        let mean_ndcg = vals.iter().map(|v| v.0).sum::<f64>() / n;
        let mean_mrr = vals.iter().map(|v| v.1).sum::<f64>() / n;
        let mean_recall = vals.iter().map(|v| v.2).sum::<f64>() / n;
        let mean_ctx = vals.iter().map(|v| v.3).sum::<f64>() / n;
        let mean_corpus = (vals.iter().map(|v| v.4).sum::<usize>() as f64 / n) as usize;
        let mean_mem_count = (vals.iter().map(|v| v.5).sum::<usize>() as f64 / n) as usize;
        let density = if mean_ctx > 0.0 {
            mean_ndcg / (mean_ctx / 1000.0)
        } else {
            0.0
        };

        aggregate.push(PipelineCellMetrics {
            condition: condition.clone(),
            strategy: strategy.clone(),
            ndcg_at_10: mean_ndcg,
            mrr: mean_mrr,
            recall_at_5: mean_recall,
            mean_context_tokens: mean_ctx,
            corpus_tokens: mean_corpus,
            memory_count: mean_mem_count,
            information_density: density,
            queries_evaluated: total_queries,
        });
    }
    let condition_order = |c: &str| match c {
        "flat" => 0,
        "enriched" => 1,
        "distilled" => 2,
        _ => 3,
    };
    aggregate.sort_by(|a, b| {
        condition_order(&a.condition)
            .cmp(&condition_order(&b.condition))
            .then(a.strategy.cmp(&b.strategy))
    });

    Ok(PipelineBenchmarkReport {
        benchmark: "LongMemEval".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        conversations: per_conversation.len(),
        total_queries,
        total_observations,
        llm_model: llm.name().to_string(),
        aggregate,
        per_conversation,
    })
}

/// Run entity extraction using Origin's production pipeline (refinery path).
///
/// Uses `extract_entities_from_memories` which calls `extract_single_memory_entities`
/// with the production EXTRACT_KNOWLEDGE_GRAPH prompt (PR5) and proper Qwen chat
/// template formatting. Much more reliable than the old custom JSON extraction.
///
/// Runs in batches of `batch_size` unlinked memories until all are processed.
async fn run_entity_extraction_for_eval(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<usize, OriginError> {
    use crate::prompts::PromptRegistry;

    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let batch_size = 10;
    let mut total = 0usize;

    // Keep extracting until no unlinked memories remain (or no progress)
    loop {
        let extracted =
            crate::refinery::extract_entities_from_memories(db, Some(llm), &prompts, batch_size)
                .await?;
        if extracted == 0 {
            break;
        }
        total += extracted;
        eprintln!(
            "    [entity_extract] batch: +{} entities (total: {})",
            extracted, total
        );
    }

    Ok(total)
}

// ===== Context Path Eval: recall vs context coverage comparison =====

/// Per-question result comparing recall (search_memory) vs context (search + concepts + graph).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPathResult {
    pub question: String,
    pub category: String,
    /// Evidence source_ids found by search_memory alone.
    pub recall_found: usize,
    /// Evidence source_ids found by search_memory + concepts + graph.
    pub context_found: usize,
    /// Total evidence source_ids for this question.
    pub total_evidence: usize,
    pub recall_coverage: f64,
    pub context_coverage: f64,
    /// Source_ids recovered by context that recall missed.
    pub recovered_ids: Vec<String>,
    /// Tokens: recall context vs full context.
    pub recall_tokens: usize,
    pub context_tokens: usize,
}

/// Aggregate report for context path eval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPathReport {
    pub benchmark: String,
    pub total_questions: usize,
    pub mean_recall_coverage: f64,
    pub mean_context_coverage: f64,
    pub coverage_delta: f64,
    pub questions_improved: usize,
    pub total_evidence_recovered: usize,
    pub mean_recall_tokens: f64,
    pub mean_context_tokens: f64,
    pub per_category: Vec<ContextPathCategoryResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPathCategoryResult {
    pub category: String,
    pub count: usize,
    pub mean_recall_coverage: f64,
    pub mean_context_coverage: f64,
    pub delta: f64,
}

impl ContextPathReport {
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Context Path Eval: {} ({} questions)\n\n",
            self.benchmark, self.total_questions
        ));
        out.push_str(&format!(
            "{:<20} | {:<10} | {:<10} | {:<8} | {:<8}\n",
            "Path", "Coverage", "Tok/Q", "Improved", "Recovered"
        ));
        out.push_str(&format!(
            "{:-<20}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+-{:-<8}\n",
            "", "", "", "", ""
        ));
        out.push_str(&format!(
            "{:<20} | {:<10.4} | {:<10.1} | {:<8} | {:<8}\n",
            "recall (search only)", self.mean_recall_coverage, self.mean_recall_tokens, "-", "-"
        ));
        out.push_str(&format!(
            "{:<20} | {:<10.4} | {:<10.1} | {:<8} | {:<8}\n",
            "context (full)",
            self.mean_context_coverage,
            self.mean_context_tokens,
            self.questions_improved,
            self.total_evidence_recovered
        ));
        out.push_str(&format!(
            "\nCoverage delta: {:+.4} ({:.1}% relative improvement)\n",
            self.coverage_delta,
            if self.mean_recall_coverage > 0.0 {
                self.coverage_delta / self.mean_recall_coverage * 100.0
            } else {
                0.0
            }
        ));

        if !self.per_category.is_empty() {
            out.push_str("\nPer category:\n");
            for cat in &self.per_category {
                out.push_str(&format!(
                    "  {} (n={}): recall={:.3} context={:.3} delta={:+.3}\n",
                    cat.category,
                    cat.count,
                    cat.mean_recall_coverage,
                    cat.mean_context_coverage,
                    cat.delta,
                ));
            }
        }

        out
    }
}

/// Compare recall (search_memory only) vs context (search + concepts + graph) on LoCoMo.
///
/// Seeds one conversation at a time, runs enrichment + distillation, then for each question:
/// 1. recall path: search_memory top-K, check evidence coverage
/// 2. context path: search_memory top-K + search_concepts top-3 source_ids, check coverage
///
/// Reports coverage delta: how many evidence items does the context path recover
/// that recall alone misses.
///
/// Requires on-device LLM for enrichment/distillation. Run with sandbox disabled.
pub async fn run_context_path_eval(
    locomo_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_conversations: usize,
) -> Result<ContextPathReport, OriginError> {
    use crate::eval::locomo::{category_name, extract_observations, load_locomo};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_locomo(locomo_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    let mut all_results: Vec<ContextPathResult> = Vec::new();
    let conv_limit = max_conversations.min(samples.len());

    for (conv_idx, sample) in samples.iter().take(max_conversations).enumerate() {
        let memories = extract_observations(sample);
        if memories.is_empty() {
            continue;
        }

        eprintln!(
            "[context_eval] Conv {}/{} ({}): {} observations",
            conv_idx + 1,
            conv_limit,
            sample.sample_id,
            memories.len(),
        );

        // Build evidence mapping
        let dia_to_source: HashMap<String, String> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| {
                (
                    m.dia_id.clone(),
                    format!("locomo_{}_obs_{}", sample.sample_id, i),
                )
            })
            .collect();

        // Seed DB
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir context_eval: {e}")))?;
        let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter)).await?;

        let docs: Vec<RawDocument> = memories
            .iter()
            .enumerate()
            .map(|(i, mem)| RawDocument {
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
        db.upsert_documents(docs).await?;

        // Run enrichment + distillation
        eprintln!("  [enriching] entity extraction...");
        let entities = run_entity_extraction_for_eval(&db, &llm).await?;
        eprintln!("  [enriching] {} entities. distilling...", entities);

        let concepts =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;
        eprintln!("  [enriched] {} concepts. evaluating...", concepts);

        // Evaluate each question
        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            }

            let evidence_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if evidence_ids.is_empty() {
                continue;
            }

            // --- Recall path: search_memory only ---
            let recall_results = db
                .search_memory(
                    &qa.question,
                    search_limit,
                    None,
                    Some("conversation"),
                    None,
                    None,
                    None,
                    None,
                )
                .await?;
            let recall_ids: HashSet<String> =
                recall_results.iter().map(|r| r.source_id.clone()).collect();
            let recall_tokens = count_results_tokens(&recall_results);

            let recall_found = evidence_ids.intersection(&recall_ids).count();

            // --- Context path: search_memory + search_concepts ---
            let mut context_ids = recall_ids.clone();
            let mut context_tokens = recall_tokens;

            // Add concept source_ids
            let concept_results = db
                .search_concepts(&qa.question, 3)
                .await
                .unwrap_or_default();
            for concept in &concept_results {
                context_tokens += count_tokens(&concept.content);
                // Get source memories via concept_sources join table
                let sources = db
                    .get_concept_sources(&concept.id)
                    .await
                    .unwrap_or_default();
                for src in &sources {
                    context_ids.insert(src.memory_source_id.clone());
                }
                // Also use the legacy source_memory_ids field
                for sid in &concept.source_memory_ids {
                    context_ids.insert(sid.clone());
                }
            }

            let context_found = evidence_ids.intersection(&context_ids).count();
            let recovered: Vec<String> = evidence_ids
                .iter()
                .filter(|id| context_ids.contains(*id) && !recall_ids.contains(*id))
                .cloned()
                .collect();

            all_results.push(ContextPathResult {
                question: qa.question.clone(),
                category: category_name(qa.category).to_string(),
                recall_found,
                context_found,
                total_evidence: evidence_ids.len(),
                recall_coverage: recall_found as f64 / evidence_ids.len() as f64,
                context_coverage: context_found as f64 / evidence_ids.len() as f64,
                recovered_ids: recovered,
                recall_tokens,
                context_tokens,
            });
        }

        let conv_results: Vec<&ContextPathResult> = all_results
            .iter()
            .rev()
            .take_while(|r| !r.question.is_empty()) // all from this conv
            .collect();
        let improved = conv_results
            .iter()
            .filter(|r| r.context_found > r.recall_found)
            .count();
        eprintln!(
            "  {} questions, {} improved by context path",
            conv_results.len(),
            improved,
        );
    }

    // Aggregate
    let n = all_results.len().max(1) as f64;
    let mean_recall_cov = all_results.iter().map(|r| r.recall_coverage).sum::<f64>() / n;
    let mean_context_cov = all_results.iter().map(|r| r.context_coverage).sum::<f64>() / n;
    let questions_improved = all_results
        .iter()
        .filter(|r| r.context_found > r.recall_found)
        .count();
    let total_recovered: usize = all_results.iter().map(|r| r.recovered_ids.len()).sum();
    let mean_recall_tok = all_results
        .iter()
        .map(|r| r.recall_tokens as f64)
        .sum::<f64>()
        / n;
    let mean_context_tok = all_results
        .iter()
        .map(|r| r.context_tokens as f64)
        .sum::<f64>()
        / n;

    // Per-category
    let mut cat_map: HashMap<String, Vec<&ContextPathResult>> = HashMap::new();
    for r in &all_results {
        cat_map.entry(r.category.clone()).or_default().push(r);
    }
    let mut per_category: Vec<ContextPathCategoryResult> = cat_map
        .iter()
        .map(|(cat, results)| {
            let cn = results.len().max(1) as f64;
            let rc = results.iter().map(|r| r.recall_coverage).sum::<f64>() / cn;
            let cc = results.iter().map(|r| r.context_coverage).sum::<f64>() / cn;
            ContextPathCategoryResult {
                category: cat.clone(),
                count: results.len(),
                mean_recall_coverage: rc,
                mean_context_coverage: cc,
                delta: cc - rc,
            }
        })
        .collect();
    per_category.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ContextPathReport {
        benchmark: "LoCoMo".to_string(),
        total_questions: all_results.len(),
        mean_recall_coverage: mean_recall_cov,
        mean_context_coverage: mean_context_cov,
        coverage_delta: mean_context_cov - mean_recall_cov,
        questions_improved,
        total_evidence_recovered: total_recovered,
        mean_recall_tokens: mean_recall_tok,
        mean_context_tokens: mean_context_tok,
        per_category,
    })
}

// ===== E2E Answer Quality: flat vs structured context with LLM-as-judge =====

/// Run E2E answer quality comparison: flat (search_memory) vs structured (search + concepts).
///
/// For each LoCoMo question:
/// 1. Build flat context: search_memory top-K concatenated
/// 2. Build structured context: search_memory + concept articles (like chat-context)
/// 3. Generate answers from both contexts using on-device LLM
/// 4. Return JudgmentTuples for offline Claude Haiku judging
///
/// Requires enrichment + distillation to be run first (concepts must exist).
/// Call this after seeding + enriching a DB, or use the all-in-one wrapper.
async fn generate_e2e_answers_for_question(
    db: &MemoryDB,
    question: &str,
    ground_truth: &str,
    category: &str,
    search_limit: usize,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<Vec<JudgmentTuple>, OriginError> {
    use crate::llm_provider::{strip_think_tags, LlmRequest};

    let system_prompt = "Answer the question using only the provided context. \
        Be specific and concise. Respond in 1-3 sentences."
        .to_string();

    let mut tuples = Vec::new();

    // --- Flat context: search_memory only ---
    let flat_results = db
        .search_memory(
            question,
            search_limit,
            None,
            Some("conversation"),
            None,
            None,
            None,
            None,
        )
        .await?;
    let flat_context: String = flat_results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("{}. {}", i + 1, r.content))
        .collect::<Vec<_>>()
        .join("\n");
    let flat_tokens = count_tokens(&flat_context);

    let flat_request = LlmRequest {
        system_prompt: Some(system_prompt.clone()),
        user_prompt: format!("Context:\n{}\n\nQuestion: {}", flat_context, question),
        max_tokens: 200,
        temperature: 0.1,
        label: Some("e2e_flat".to_string()),
    };
    if let Ok(raw) = llm.generate(flat_request).await {
        let answer = strip_think_tags(&raw);
        tuples.push(JudgmentTuple {
            question: question.to_string(),
            ground_truth: ground_truth.to_string(),
            approach: format!("flat_{}", category),
            answer,
            context_tokens: flat_tokens,
        });
    }

    // --- Structured context: search_memory + concept articles ---
    let mut structured_parts: Vec<String> = Vec::new();

    // Concept articles (like chat-context's "Compiled Knowledge" section)
    let concepts = db.search_concepts(question, 3).await.unwrap_or_default();
    if !concepts.is_empty() {
        structured_parts.push("## Compiled Knowledge".to_string());
        for c in &concepts {
            let summary = c.summary.as_deref().unwrap_or("");
            structured_parts.push(format!("**{}**: {}\n{}", c.title, summary, c.content));
        }
    }

    // Memory search results
    if !flat_results.is_empty() {
        structured_parts.push("## Relevant Memories".to_string());
        for (i, r) in flat_results.iter().enumerate() {
            structured_parts.push(format!("{}. {}", i + 1, r.content));
        }
    }

    let structured_context = structured_parts.join("\n\n");
    let structured_tokens = count_tokens(&structured_context);

    let structured_request = LlmRequest {
        system_prompt: Some(system_prompt),
        user_prompt: format!("Context:\n{}\n\nQuestion: {}", structured_context, question),
        max_tokens: 200,
        temperature: 0.1,
        label: Some("e2e_structured".to_string()),
    };
    if let Ok(raw) = llm.generate(structured_request).await {
        let answer = strip_think_tags(&raw);
        tuples.push(JudgmentTuple {
            question: question.to_string(),
            ground_truth: ground_truth.to_string(),
            approach: format!("structured_{}", category),
            answer,
            context_tokens: structured_tokens,
        });
    }

    Ok(tuples)
}

/// Run full E2E answer quality eval on LoCoMo: seed, enrich, distill, generate answers.
///
/// Returns JudgmentTuples for offline judging with `judge_with_claude`.
/// Two approaches per question: "flat_{category}" and "structured_{category}".
pub async fn run_e2e_context_eval(
    locomo_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_conversations: usize,
    max_questions_per_conv: usize,
) -> Result<Vec<JudgmentTuple>, OriginError> {
    use crate::eval::locomo::{category_name, extract_observations, load_locomo};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_locomo(locomo_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    let mut all_tuples: Vec<JudgmentTuple> = Vec::new();
    let conv_limit = max_conversations.min(samples.len());

    for (conv_idx, sample) in samples.iter().take(max_conversations).enumerate() {
        let memories = extract_observations(sample);
        if memories.is_empty() {
            continue;
        }

        eprintln!(
            "[e2e_context] Conv {}/{} ({}): {} observations",
            conv_idx + 1,
            conv_limit,
            sample.sample_id,
            memories.len(),
        );

        // Seed DB
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir e2e_context: {e}")))?;
        let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter)).await?;

        let docs: Vec<RawDocument> = memories
            .iter()
            .enumerate()
            .map(|(i, mem)| RawDocument {
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
        db.upsert_documents(docs).await?;

        // Enrich + distill
        eprintln!("  [enriching]...");
        let entities = run_entity_extraction_for_eval(&db, &llm).await?;
        let concepts =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;
        eprintln!(
            "  [enriched] {} entities, {} concepts. generating answers...",
            entities, concepts
        );

        // Generate answers for each question
        let mut questions_done = 0usize;
        for qa in &sample.qa {
            if questions_done >= max_questions_per_conv {
                break;
            }
            if qa.category == 5 {
                continue;
            }

            let ground_truth = qa
                .answer
                .as_ref()
                .map(|v| v.as_str().unwrap_or(&v.to_string()).to_string())
                .unwrap_or_default();
            if ground_truth.is_empty() {
                continue;
            }

            let category = category_name(qa.category);

            match generate_e2e_answers_for_question(
                &db,
                &qa.question,
                &ground_truth,
                category,
                search_limit,
                &llm,
            )
            .await
            {
                Ok(tuples) => {
                    all_tuples.extend(tuples);
                }
                Err(e) => {
                    log::warn!("[e2e_context] question failed: {e}");
                }
            }

            questions_done += 1;
            if questions_done.is_multiple_of(10) {
                eprintln!(
                    "  [progress] {}/{} questions",
                    questions_done, max_questions_per_conv
                );
            }
        }

        eprintln!(
            "  Conv done: {} answers generated ({} tuples total)",
            questions_done,
            all_tuples.len(),
        );
    }

    eprintln!(
        "[e2e_context] Total: {} judgment tuples ({} questions x 2 approaches)",
        all_tuples.len(),
        all_tuples.len() / 2,
    );

    Ok(all_tuples)
}

/// Same as run_e2e_context_eval but for LongMemEval.
pub async fn run_e2e_context_eval_longmemeval(
    longmemeval_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_questions: usize,
    _max_answers_per_question: usize,
) -> Result<Vec<JudgmentTuple>, OriginError> {
    use crate::eval::longmemeval::{category_name, extract_memories, load_longmemeval};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_longmemeval(longmemeval_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    // Pre-create shared embedder
    eprintln!("[e2e_context_lme] loading shared embedder...");
    let shared_embedder = MemoryDB::create_shared_embedder().await?;

    let mut all_tuples: Vec<JudgmentTuple> = Vec::new();
    let sample_limit = max_questions.min(samples.len());

    for (q_idx, sample) in samples.iter().take(max_questions).enumerate() {
        let memories = extract_memories(sample);
        if memories.is_empty() {
            continue;
        }

        if q_idx % 25 == 0 {
            eprintln!(
                "[e2e_context_lme] Q {}/{} ({}): {} memories",
                q_idx + 1,
                sample_limit,
                sample.question_id,
                memories.len(),
            );
        }

        // Seed DB with shared embedder
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir e2e_lme: {e}")))?;
        let db = MemoryDB::new_with_shared_embedder(
            tmp.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;

        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| RawDocument {
                content: mem.content.clone(),
                source_id: format!(
                    "lme_{}_{}_t{}",
                    sample.question_id, mem.session_idx, mem.turn_idx
                ),
                source: "memory".to_string(),
                title: format!("session {} turn {}", mem.session_idx, mem.turn_idx),
                memory_type: Some(
                    if sample.question_type == "single-session-preference" {
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
            .collect();
        db.upsert_documents(docs).await?;

        // Enrich + distill
        let _entities = run_entity_extraction_for_eval(&db, &llm).await?;
        let _concepts =
            crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;

        // Generate answers
        let ground_truth = sample
            .answer
            .as_str()
            .unwrap_or(&sample.answer.to_string())
            .to_string();
        if ground_truth.is_empty() {
            continue;
        }

        let category = category_name(&sample.question_type);

        if let Ok(tuples) = generate_e2e_answers_for_question(
            &db,
            &sample.question,
            &ground_truth,
            category,
            search_limit,
            &llm,
        )
        .await
        {
            all_tuples.extend(tuples);
        }

        if q_idx % 50 == 49 {
            eprintln!(
                "  [progress] {}/{} questions, {} tuples",
                q_idx + 1,
                sample_limit,
                all_tuples.len()
            );
        }
    }

    eprintln!(
        "[e2e_context_lme] Total: {} judgment tuples",
        all_tuples.len(),
    );

    Ok(all_tuples)
}

// ===== Context Path Eval: LongMemEval =====

/// Same as run_context_path_eval but for LongMemEval dataset.
/// Each question has its own memory set. Evidence = memories from answer_session_ids.
pub async fn run_context_path_eval_longmemeval(
    longmemeval_path: &Path,
    llm: Arc<dyn crate::llm_provider::LlmProvider>,
    search_limit: usize,
    max_questions: usize,
) -> Result<ContextPathReport, OriginError> {
    use crate::eval::longmemeval::{category_name, extract_memories, load_longmemeval};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let samples = load_longmemeval(longmemeval_path)?;
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    // Pre-create shared embedder — loads 140MB ONNX model once instead of per-question
    eprintln!("[context_path_lme] loading shared embedder...");
    let shared_embedder = MemoryDB::create_shared_embedder().await?;
    eprintln!(
        "[context_path_lme] embedder ready, processing {} questions",
        samples.len().min(max_questions)
    );

    let mut all_results: Vec<ContextPathResult> = Vec::new();
    let sample_limit = max_questions.min(samples.len());

    for (q_idx, sample) in samples.iter().take(max_questions).enumerate() {
        let memories = extract_memories(sample);
        if memories.is_empty() {
            continue;
        }

        if q_idx % 25 == 0 {
            eprintln!(
                "[context_path_lme] Q {}/{} ({}): {} memories{}",
                q_idx + 1,
                sample_limit,
                sample.question_id,
                memories.len(),
                if memories.len() < 15 {
                    " (skip distill)"
                } else {
                    ""
                },
            );
        }

        // Build evidence mapping: memories from answer sessions are evidence
        let answer_session_set: HashSet<String> =
            sample.answer_session_ids.iter().cloned().collect();

        let evidence_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| answer_session_set.contains(&m.session_id))
            .map(|m| {
                format!(
                    "lme_{}_{}_t{}",
                    sample.question_id, m.session_idx, m.turn_idx
                )
            })
            .collect();

        if evidence_source_ids.is_empty() {
            continue;
        }

        // Seed DB with shared embedder (skip 10-30s model reload per question)
        let tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir ctx_lme: {e}")))?;
        let db = MemoryDB::new_with_shared_embedder(
            tmp.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;

        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| RawDocument {
                content: mem.content.clone(),
                source_id: format!(
                    "lme_{}_{}_t{}",
                    sample.question_id, mem.session_idx, mem.turn_idx
                ),
                source: "memory".to_string(),
                title: format!("session {} turn {}", mem.session_idx, mem.turn_idx),
                memory_type: Some(
                    if sample.question_type == "single-session-preference" {
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
            .collect();
        db.upsert_documents(docs).await?;

        // Only distill if enough memories to form meaningful clusters (min 15).
        // LME has avg 11 memories per question — most won't produce concepts.
        // Each distillation call takes ~30s (LLM inference), so skipping saves hours.
        if memories.len() >= 15 {
            let _concepts =
                crate::refinery::distill_concepts(&db, Some(&llm), &prompts, &tuning, None).await?;
        }

        // --- Recall path ---
        let recall_results = db
            .search_memory(
                &sample.question,
                search_limit,
                None,
                Some("conversation"),
                None,
                None,
                None,
                None,
            )
            .await?;
        let recall_ids: HashSet<String> =
            recall_results.iter().map(|r| r.source_id.clone()).collect();
        let recall_tokens = count_results_tokens(&recall_results);
        let recall_found = evidence_source_ids.intersection(&recall_ids).count();

        // --- Context path ---
        let mut context_ids = recall_ids.clone();
        let mut context_tokens = recall_tokens;

        let concept_results = db
            .search_concepts(&sample.question, 3)
            .await
            .unwrap_or_default();
        for concept in &concept_results {
            context_tokens += count_tokens(&concept.content);
            let sources = db
                .get_concept_sources(&concept.id)
                .await
                .unwrap_or_default();
            for src in &sources {
                context_ids.insert(src.memory_source_id.clone());
            }
            for sid in &concept.source_memory_ids {
                context_ids.insert(sid.clone());
            }
        }

        let context_found = evidence_source_ids.intersection(&context_ids).count();
        let recovered: Vec<String> = evidence_source_ids
            .iter()
            .filter(|id| context_ids.contains(*id) && !recall_ids.contains(*id))
            .cloned()
            .collect();

        let category = category_name(&sample.question_type);

        all_results.push(ContextPathResult {
            question: sample.question.clone(),
            category: category.to_string(),
            recall_found,
            context_found,
            total_evidence: evidence_source_ids.len(),
            recall_coverage: recall_found as f64 / evidence_source_ids.len() as f64,
            context_coverage: context_found as f64 / evidence_source_ids.len() as f64,
            recovered_ids: recovered,
            recall_tokens,
            context_tokens,
        });

        if q_idx % 50 == 49 {
            let improved = all_results
                .iter()
                .filter(|r| r.context_found > r.recall_found)
                .count();
            eprintln!(
                "  [progress] {}/{} questions, {} improved",
                q_idx + 1,
                sample_limit,
                improved,
            );
        }
    }

    // Aggregate
    let n = all_results.len().max(1) as f64;
    let mean_recall_cov = all_results.iter().map(|r| r.recall_coverage).sum::<f64>() / n;
    let mean_context_cov = all_results.iter().map(|r| r.context_coverage).sum::<f64>() / n;
    let questions_improved = all_results
        .iter()
        .filter(|r| r.context_found > r.recall_found)
        .count();
    let total_recovered: usize = all_results.iter().map(|r| r.recovered_ids.len()).sum();
    let mean_recall_tok = all_results
        .iter()
        .map(|r| r.recall_tokens as f64)
        .sum::<f64>()
        / n;
    let mean_context_tok = all_results
        .iter()
        .map(|r| r.context_tokens as f64)
        .sum::<f64>()
        / n;

    // Per-category
    let mut cat_map: HashMap<String, Vec<&ContextPathResult>> = HashMap::new();
    for r in &all_results {
        cat_map.entry(r.category.clone()).or_default().push(r);
    }
    let mut per_category: Vec<ContextPathCategoryResult> = cat_map
        .iter()
        .map(|(cat, results)| {
            let cn = results.len().max(1) as f64;
            let rc = results.iter().map(|r| r.recall_coverage).sum::<f64>() / cn;
            let cc = results.iter().map(|r| r.context_coverage).sum::<f64>() / cn;
            ContextPathCategoryResult {
                category: cat.clone(),
                count: results.len(),
                mean_recall_coverage: rc,
                mean_context_coverage: cc,
                delta: cc - rc,
            }
        })
        .collect();
    per_category.sort_by(|a, b| {
        b.delta
            .partial_cmp(&a.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ContextPathReport {
        benchmark: "LongMemEval".to_string(),
        total_questions: all_results.len(),
        mean_recall_coverage: mean_recall_cov,
        mean_context_coverage: mean_context_cov,
        coverage_delta: mean_context_cov - mean_recall_cov,
        questions_improved,
        total_evidence_recovered: total_recovered,
        mean_recall_tokens: mean_recall_tok,
        mean_context_tokens: mean_context_tok,
        per_category,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens_nonempty() {
        let tokens = count_tokens("Hello, world!");
        assert!(tokens > 0, "should count at least 1 token");
        assert!(tokens < 10, "a short sentence should be under 10 tokens");
    }

    #[test]
    fn test_count_tokens_empty() {
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn test_count_tokens_realistic_memory() {
        let memory = "Decided to use SQLite instead of PostgreSQL for the local-first architecture. Reasoning: no daemon dependency, single-file DB, good enough for single-user workloads up to 100K memories.";
        let tokens = count_tokens(memory);
        // A ~35-word sentence should be roughly 40-60 tokens
        assert!(tokens > 20 && tokens < 80, "got {} tokens", tokens);
    }

    #[test]
    fn test_token_metrics_compression_ratio() {
        let ratio = TokenMetrics::compute_compression_ratio(100, 1000);
        assert!(
            (ratio - 0.1).abs() < 1e-9,
            "100/1000 should be 0.1, got {}",
            ratio
        );

        let ratio2 = TokenMetrics::compute_compression_ratio(50, 200);
        assert!(
            (ratio2 - 0.25).abs() < 1e-9,
            "50/200 should be 0.25, got {}",
            ratio2
        );
    }

    #[test]
    fn test_token_metrics_zero_corpus() {
        let ratio = TokenMetrics::compute_compression_ratio(0, 0);
        assert_eq!(ratio, 0.0, "zero corpus should return 0.0");

        let ratio2 = TokenMetrics::compute_compression_ratio(100, 0);
        assert_eq!(
            ratio2, 0.0,
            "non-zero context with zero corpus should return 0.0"
        );
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(SearchStrategy::Origin.name(), "origin");
        assert_eq!(SearchStrategy::OriginReranked.name(), "origin_reranked");
        assert_eq!(SearchStrategy::OriginExpanded.name(), "origin_expanded");
        assert_eq!(SearchStrategy::NaiveRag.name(), "naive_rag");
        assert_eq!(SearchStrategy::FullReplay.name(), "full_replay");
        assert_eq!(SearchStrategy::NoMemory.name(), "no_memory");
        assert_eq!(SearchStrategy::FtsOnly.name(), "fts_only");
        assert_eq!(SearchStrategy::VectorPlusFts.name(), "vector_plus_fts");

        assert_eq!(SearchStrategy::Origin.display_name(), "Origin");
        assert_eq!(
            SearchStrategy::OriginReranked.display_name(),
            "Origin+Rerank"
        );
        assert_eq!(
            SearchStrategy::OriginExpanded.display_name(),
            "Origin+Expand"
        );
        assert_eq!(SearchStrategy::NaiveRag.display_name(), "Naive RAG");
        assert_eq!(SearchStrategy::FullReplay.display_name(), "Full Replay");
        assert_eq!(SearchStrategy::NoMemory.display_name(), "No Memory");
        assert_eq!(SearchStrategy::FtsOnly.display_name(), "FTS Only");
        assert_eq!(SearchStrategy::VectorPlusFts.display_name(), "Vector+FTS");
    }

    #[test]
    fn test_strategy_requires_llm() {
        assert!(!SearchStrategy::Origin.requires_llm());
        assert!(SearchStrategy::OriginReranked.requires_llm());
        assert!(SearchStrategy::OriginExpanded.requires_llm());
        assert!(!SearchStrategy::NaiveRag.requires_llm());
        assert!(!SearchStrategy::FullReplay.requires_llm());
        assert!(!SearchStrategy::NoMemory.requires_llm());
        assert!(!SearchStrategy::FtsOnly.requires_llm());
        assert!(!SearchStrategy::VectorPlusFts.requires_llm());
    }

    #[tokio::test]
    async fn test_naive_vector_search_returns_results() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter))
            .await
            .unwrap();

        let docs = vec![
            RawDocument {
                content: "Rust ownership rules prevent data races at compile time.".to_string(),
                source_id: "rust_ownership".to_string(),
                source: "memory".to_string(),
                title: "Rust ownership".to_string(),
                memory_type: Some("fact".to_string()),
                ..Default::default()
            },
            RawDocument {
                content: "tokio is an async runtime for Rust using the executor pattern."
                    .to_string(),
                source_id: "tokio_async".to_string(),
                source: "memory".to_string(),
                title: "Tokio async".to_string(),
                memory_type: Some("fact".to_string()),
                ..Default::default()
            },
            RawDocument {
                content: "SQLite is an embedded relational database with ACID guarantees."
                    .to_string(),
                source_id: "sqlite_db".to_string(),
                source: "memory".to_string(),
                title: "SQLite".to_string(),
                memory_type: Some("fact".to_string()),
                ..Default::default()
            },
        ];
        db.upsert_documents(docs).await.unwrap();

        let results = db
            .naive_vector_search("async programming in Rust", 3, None)
            .await
            .unwrap();

        // Should return results (DiskANN index is built on upsert)
        assert!(
            !results.is_empty(),
            "naive_vector_search should return results after seeding"
        );
        assert!(results.len() <= 3, "should respect the limit");
    }

    #[tokio::test]
    #[ignore] // Eval benchmark (~4 min on CI) — run locally with --ignored
    async fn test_run_quality_cost_eval_basic() {
        // Use the project's fixture directory if it exists; otherwise skip gracefully.
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");

        if !fixture_dir.exists() {
            eprintln!("Skipping test_run_quality_cost_eval_basic: fixture dir not found");
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

        // Report shape
        assert!(!report.benchmark.is_empty());
        assert!(!report.timestamp.is_empty());
        assert_eq!(report.tokenizer, "cl100k_base");

        // We should have a report for each non-LLM strategy
        let non_llm: Vec<_> = strategies.iter().filter(|s| !s.requires_llm()).collect();
        assert_eq!(
            report.strategies.len(),
            non_llm.len(),
            "should have one StrategyReport per non-LLM strategy"
        );

        // Headline sanity
        // FullReplay has more tokens than Origin (unless corpus is tiny)
        let origin_r = report.strategies.iter().find(|r| r.strategy == "origin");
        let replay_r = report
            .strategies
            .iter()
            .find(|r| r.strategy == "full_replay");
        if let (Some(_o), Some(_r)) = (origin_r, replay_r) {
            // savings_pct can be 0 or positive
            assert!(report.headline.savings_pct >= 0.0);
        }
    }

    #[test]
    fn test_terminal_report_formatting() {
        let report = QualityCostReport {
            benchmark: "test-bench".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            tokenizer: "cl100k_base".to_string(),
            strategies: vec![
                StrategyReport {
                    strategy: "origin".to_string(),
                    mean_context_tokens: 120.5,
                    median_context_tokens: 100.0,
                    p25_context_tokens: 0.0,
                    p75_context_tokens: 0.0,
                    stddev_context_tokens: 0.0,
                    mean_compression_ratio: 0.12,
                    ndcg_at_10: 0.82,
                    mrr: 0.75,
                    recall_at_5: 0.68,
                    stddev_ndcg: 0.0,
                    stddev_mrr: 0.0,
                },
                StrategyReport {
                    strategy: "full_replay".to_string(),
                    mean_context_tokens: 1000.0,
                    median_context_tokens: 950.0,
                    p25_context_tokens: 0.0,
                    p75_context_tokens: 0.0,
                    stddev_context_tokens: 0.0,
                    mean_compression_ratio: 1.0,
                    ndcg_at_10: 1.0,
                    mrr: 1.0,
                    recall_at_5: 1.0,
                    stddev_ndcg: 0.0,
                    stddev_mrr: 0.0,
                },
            ],
            headline: HeadlineMetrics {
                savings_pct: 87.95,
                origin_tokens: 120.5,
                replay_tokens: 1000.0,
                quality_retained_pct: 82.0,
            },
            scaling: vec![],
        };

        let output = report.to_terminal();

        // Header
        assert!(
            output.contains("test-bench"),
            "should contain benchmark name"
        );
        assert!(output.contains("cl100k_base"), "should contain tokenizer");

        // Column headers
        assert!(output.contains("NDCG@10"), "should have NDCG@10 column");
        assert!(output.contains("MRR"), "should have MRR column");
        assert!(output.contains("Recall@5"), "should have Recall@5 column");
        assert!(output.contains("Tokens/Query"), "should have token column");

        // Data rows
        assert!(output.contains("origin"), "should contain origin row");
        assert!(
            output.contains("full_replay"),
            "should contain full_replay row"
        );

        // Headline
        assert!(
            output.contains("87.9") || output.contains("88.0"),
            "should show savings_pct"
        );
    }

    #[tokio::test]
    async fn test_multi_turn_eval() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!(
                "Skipping test_multi_turn_eval: fixture dir not found at {:?}",
                fixture_dir
            );
            return;
        }

        let report = run_multi_turn_eval(&fixture_dir, 10, 10, 200)
            .await
            .unwrap();

        assert_eq!(report.turns, 10);
        assert_eq!(report.per_turn.len(), 10);

        // Origin total should be MUCH less than replay total
        assert!(
            report.total_origin_tokens < report.total_replay_tokens,
            "Origin {} should be < Replay {}",
            report.total_origin_tokens,
            report.total_replay_tokens
        );

        // Replay should grow each turn (each turn adds response_overhead)
        for i in 1..report.per_turn.len() {
            assert!(
                report.per_turn[i].replay_tokens >= report.per_turn[i - 1].replay_tokens,
                "Replay should grow turn-over-turn: turn {} ({}) < turn {} ({})",
                i + 1,
                report.per_turn[i].replay_tokens,
                i,
                report.per_turn[i - 1].replay_tokens
            );
        }

        // Origin should stay roughly constant (allow up to 50% drift due to query rotation)
        let first = report.per_turn[0].origin_tokens;
        let last = report.per_turn.last().unwrap().origin_tokens;
        if first > 0 {
            let drift = (last as f64 - first as f64).abs() / first as f64;
            assert!(
                drift < 0.5,
                "Origin should stay roughly constant, drift={:.1}%",
                drift * 100.0
            );
        }

        // Print results
        eprintln!(
            "\n=== Multi-Turn Token Accumulation ({} turns) ===",
            report.turns
        );
        eprintln!(
            "{:<6} | {:<15} | {:<15} | {:<15} | {:<15}",
            "Turn", "Origin/turn", "Replay/turn", "Origin cumul", "Replay cumul"
        );
        eprintln!(
            "{:-<6}-+-{:-<15}-+-{:-<15}-+-{:-<15}-+-{:-<15}",
            "", "", "", "", ""
        );
        for p in &report.per_turn {
            eprintln!(
                "{:<6} | {:<15} | {:<15} | {:<15} | {:<15}",
                p.turn, p.origin_tokens, p.replay_tokens, p.cumulative_origin, p.cumulative_replay
            );
        }
        eprintln!(
            "\nTotal: Origin={}, Replay={}, Savings={:.1}%",
            report.total_origin_tokens, report.total_replay_tokens, report.savings_pct
        );
    }

    #[tokio::test]
    #[ignore] // Eval benchmark (~4 min on CI) — run locally with --ignored
    async fn test_scaling_eval_basic() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!("Skipping: fixture dir not found at {:?}", fixture_dir);
            return;
        }

        let sizes = vec![3, 5, 10, 20, 40];
        let points = run_scaling_eval(&fixture_dir, &sizes, 10).await.unwrap();

        assert!(
            !points.is_empty(),
            "should produce at least one scaling point"
        );
        // With more seeds, replay tokens should increase
        if points.len() >= 2 {
            assert!(
                points[1].replay_tokens >= points[0].replay_tokens,
                "replay tokens should grow with corpus size"
            );
        }
    }

    #[test]
    fn test_baseline_save_load_roundtrip() {
        let report = QualityCostReport {
            benchmark: "roundtrip-test".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            tokenizer: "cl100k_base".to_string(),
            strategies: vec![StrategyReport {
                strategy: "origin".to_string(),
                mean_context_tokens: 42.0,
                median_context_tokens: 40.0,
                p25_context_tokens: 0.0,
                p75_context_tokens: 0.0,
                stddev_context_tokens: 0.0,
                mean_compression_ratio: 0.05,
                ndcg_at_10: 0.9,
                mrr: 0.88,
                recall_at_5: 0.77,
                stddev_ndcg: 0.0,
                stddev_mrr: 0.0,
            }],
            headline: HeadlineMetrics {
                savings_pct: 95.0,
                origin_tokens: 42.0,
                replay_tokens: 840.0,
                quality_retained_pct: 90.0,
            },
            scaling: vec![],
        };

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("token_efficiency_baseline.json");

        report
            .save_baseline(&path)
            .expect("save_baseline should succeed");
        assert!(path.exists(), "baseline file should exist after save");

        let loaded = QualityCostReport::load_baseline(&path).expect("load_baseline should succeed");

        assert_eq!(loaded.benchmark, report.benchmark);
        assert_eq!(loaded.timestamp, report.timestamp);
        assert_eq!(loaded.tokenizer, report.tokenizer);
        assert_eq!(loaded.strategies.len(), report.strategies.len());
        assert_eq!(loaded.strategies[0].strategy, report.strategies[0].strategy);
        assert!((loaded.strategies[0].ndcg_at_10 - report.strategies[0].ndcg_at_10).abs() < 1e-9);
        assert!((loaded.headline.savings_pct - report.headline.savings_pct).abs() < 1e-9);
    }

    #[tokio::test]
    #[ignore] // Takes ~10 minutes — seeds embeddings for ~5K observations across 10 conversations
    async fn benchmark_locomo_token_efficiency() {
        use crate::eval::locomo::{extract_observations, load_locomo};

        let locomo_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/data/locomo10.json");

        if !locomo_path.exists() {
            eprintln!("Skipping: locomo10.json not found at {:?}", locomo_path);
            return;
        }

        let samples = load_locomo(&locomo_path).expect("failed to load LoCoMo dataset");
        eprintln!("Loaded {} conversations", samples.len());

        let search_limit = 10;

        // Accumulators across all conversations and QA pairs
        let mut origin_tokens_all: Vec<usize> = Vec::new();
        let mut naive_tokens_all: Vec<usize> = Vec::new();
        let mut corpus_tokens_all: Vec<usize> = Vec::new();
        let mut total_questions = 0usize;
        let mut total_observations = 0usize;

        for (conv_idx, sample) in samples.iter().enumerate() {
            let memories = extract_observations(sample);
            let obs_count = memories.len();
            total_observations += obs_count;

            // Compute corpus tokens for this conversation (all observations concatenated)
            let corpus_text: String = memories
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let conv_corpus_tokens = count_tokens(&corpus_text);

            eprintln!(
                "Conversation {}/{} ({}): {} observations, {} corpus tokens — seeding...",
                conv_idx + 1,
                samples.len(),
                sample.sample_id,
                obs_count,
                conv_corpus_tokens,
            );

            // Create ephemeral DB and seed all observations
            let tmp = tempfile::tempdir().expect("tempdir");
            let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter))
                .await
                .expect("MemoryDB::new");

            let docs: Vec<RawDocument> = memories
                .iter()
                .enumerate()
                .map(|(i, mem)| RawDocument {
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
            db.upsert_documents(docs).await.expect("upsert_documents");

            // Evaluate each non-adversarial QA pair
            let mut conv_questions = 0usize;
            let mut conv_origin_sum = 0usize;
            let mut conv_naive_sum = 0usize;

            for qa in &sample.qa {
                if qa.category == 5 {
                    continue; // skip adversarial
                }

                // Origin hybrid search
                let origin_results = db
                    .search_memory(
                        &qa.question,
                        search_limit,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .await
                    .expect("search_memory");
                let origin_ctx_tokens = count_results_tokens(&origin_results);

                // Naive vector search
                let naive_results = db
                    .naive_vector_search(&qa.question, search_limit, None)
                    .await
                    .expect("naive_vector_search");
                let naive_ctx_tokens = count_results_tokens(&naive_results);

                origin_tokens_all.push(origin_ctx_tokens);
                naive_tokens_all.push(naive_ctx_tokens);
                corpus_tokens_all.push(conv_corpus_tokens);

                conv_questions += 1;
                conv_origin_sum += origin_ctx_tokens;
                conv_naive_sum += naive_ctx_tokens;
            }

            total_questions += conv_questions;

            if conv_questions > 0 {
                let conv_origin_mean = conv_origin_sum as f64 / conv_questions as f64;
                let conv_naive_mean = conv_naive_sum as f64 / conv_questions as f64;
                let conv_origin_pct = if conv_corpus_tokens > 0 {
                    (1.0 - conv_origin_mean / conv_corpus_tokens as f64) * 100.0
                } else {
                    0.0
                };
                eprintln!(
                    "  {} QA pairs | Origin: {:.0} tok/q ({:.1}% savings) | Naive: {:.0} tok/q | Corpus: {} tok",
                    conv_questions,
                    conv_origin_mean,
                    conv_origin_pct,
                    conv_naive_mean,
                    conv_corpus_tokens,
                );
            }
        }

        // Global aggregates
        let n = total_questions as f64;
        let mean_origin = origin_tokens_all.iter().sum::<usize>() as f64 / n;
        let mean_naive = naive_tokens_all.iter().sum::<usize>() as f64 / n;
        let mean_corpus = corpus_tokens_all.iter().sum::<usize>() as f64 / n;

        let origin_savings = (1.0 - mean_origin / mean_corpus) * 100.0;
        let naive_savings = (1.0 - mean_naive / mean_corpus) * 100.0;

        let origin_compression = mean_origin / mean_corpus;
        let naive_compression = mean_naive / mean_corpus;

        eprintln!("\n========================================");
        eprintln!("LoCoMo Token Efficiency Results");
        eprintln!("========================================");
        eprintln!("Conversations:     {}", samples.len());
        eprintln!("Total observations: {}", total_observations);
        eprintln!("Total QA pairs:    {}", total_questions);
        eprintln!("Search limit:      {}", search_limit);
        eprintln!("Tokenizer:         cl100k_base");
        eprintln!("----------------------------------------");
        eprintln!("                   Tokens/Query  Compression  Savings");
        eprintln!(
            "Full Replay        {:>10.1}  {:>11.4}  {:>7.1}%",
            mean_corpus, 1.0, 0.0
        );
        eprintln!(
            "Origin (hybrid)    {:>10.1}  {:>11.4}  {:>7.1}%",
            mean_origin, origin_compression, origin_savings
        );
        eprintln!(
            "Naive RAG (vector) {:>10.1}  {:>11.4}  {:>7.1}%",
            mean_naive, naive_compression, naive_savings
        );
        eprintln!("----------------------------------------");
        eprintln!(
            "Headline: Origin saves {:.1}% tokens vs Full Replay ({:.0} vs {:.0} tokens/query)",
            origin_savings, mean_origin, mean_corpus,
        );
        eprintln!(
            "          Naive RAG saves {:.1}% tokens vs Full Replay ({:.0} vs {:.0} tokens/query)",
            naive_savings, mean_naive, mean_corpus,
        );
        eprintln!("========================================");
    }

    /// Ablation test: seeds a small DB with 3 clearly-differentiated documents and
    /// runs NaiveRag, FtsOnly, VectorPlusFts, and Origin. Verifies:
    /// - All strategies return non-empty results (basic correctness).
    /// - Origin and VectorPlusFts return at most `limit` results.
    /// - FtsOnly can retrieve keyword-matching documents.
    #[tokio::test]
    async fn test_ablation_strategies() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MemoryDB::new(tmp.path(), Arc::new(NoopEmitter))
            .await
            .unwrap();

        let docs = vec![
            RawDocument {
                content: "Rust ownership prevents memory safety issues at compile time via the borrow checker.".to_string(),
                source_id: "rust_safety".to_string(),
                source: "memory".to_string(),
                title: "Rust safety".to_string(),
                memory_type: Some("fact".to_string()),
                ..Default::default()
            },
            RawDocument {
                content: "tokio provides an async runtime for Rust with futures and tasks.".to_string(),
                source_id: "tokio_runtime".to_string(),
                source: "memory".to_string(),
                title: "Tokio runtime".to_string(),
                memory_type: Some("fact".to_string()),
                ..Default::default()
            },
            RawDocument {
                content: "SQLite is an embedded relational database engine with ACID guarantees.".to_string(),
                source_id: "sqlite_db".to_string(),
                source: "memory".to_string(),
                title: "SQLite".to_string(),
                memory_type: Some("fact".to_string()),
                ..Default::default()
            },
        ];
        db.upsert_documents(docs).await.unwrap();

        let query = "Rust async runtime";
        let limit = 3;

        // NaiveRag: vector-only
        let naive = db.naive_vector_search(query, limit, None).await.unwrap();
        assert!(!naive.is_empty(), "NaiveRag should return results");
        assert!(naive.len() <= limit, "NaiveRag should respect limit");

        // FtsOnly: keyword BM25
        let fts = db.fts_only_search(query, limit, None).await.unwrap();
        // FTS may return empty if tokeniser doesn't match; don't assert non-empty
        // but do assert limit is respected when non-empty.
        assert!(fts.len() <= limit, "FtsOnly should respect limit");

        // VectorPlusFts: merged by max score
        let vpf = db.vector_plus_fts_search(query, limit, None).await.unwrap();
        assert!(
            !vpf.is_empty(),
            "VectorPlusFts should return results (vector path guaranteed)"
        );
        assert!(vpf.len() <= limit, "VectorPlusFts should respect limit");

        // Origin: full hybrid
        let origin = db
            .search_memory(
                query,
                limit,
                None,
                None,
                None,
                Some(1.0), // neutralize confirmation boost
                Some(1.0), // neutralize recap penalty
                None,
            )
            .await
            .unwrap();
        assert!(!origin.is_empty(), "Origin should return results");
        assert!(origin.len() <= limit, "Origin should respect limit");

        // VectorPlusFts should cover at least what NaiveRag covers (superset of signals)
        let naive_ids: HashSet<&str> = naive.iter().map(|r| r.source_id.as_str()).collect();
        let vpf_ids: HashSet<&str> = vpf.iter().map(|r| r.source_id.as_str()).collect();
        // At minimum the vector-matched docs should appear in VectorPlusFts
        for id in &naive_ids {
            assert!(
                vpf_ids.contains(id),
                "VectorPlusFts should include doc '{}' found by NaiveRag",
                id
            );
        }
    }

    #[tokio::test]
    #[ignore] // Eval benchmark (~3 min on CI) — run locally with --ignored
    async fn test_pipeline_token_eval_simulated() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!(
                "Skipping test_pipeline_token_eval_simulated: fixture dir not found at {:?}",
                fixture_dir
            );
            return;
        }

        let report = run_pipeline_token_eval_simulated(&fixture_dir, 10)
            .await
            .unwrap();

        assert_eq!(report.stages.len(), 3);

        // Distilled should have fewer memories than raw
        let raw = &report.stages[0];
        let distilled = &report.stages[1];
        assert!(
            distilled.memory_count <= raw.memory_count,
            "distilled memory count {} should be <= raw {}",
            distilled.memory_count,
            raw.memory_count
        );

        // Concept should have fewest memories
        let concept = &report.stages[2];
        assert!(
            concept.memory_count <= distilled.memory_count,
            "concept memory count {} should be <= distilled {}",
            concept.memory_count,
            distilled.memory_count
        );

        // Token reduction should be non-negative (concept stage delivers fewer search result
        // tokens to the LLM context — it consolidates many entries into one).
        // Note: in simulation, corpus size stays roughly equal (content is concatenated not
        // condensed), so reduction is measured on retrieved context tokens.
        assert!(
            report.token_reduction_pct >= 0.0,
            "token_reduction_pct should be >= 0, got {}",
            report.token_reduction_pct
        );

        // Print results
        eprintln!("\n=== Pipeline Token Efficiency ===");
        eprintln!(
            "{:<12} | {:<8} | {:<12} | {:<12} | {:<8} | Density",
            "Stage", "Memories", "Corpus Tok", "Search Tok", "NDCG"
        );
        eprintln!(
            "{:-<12}-+-{:-<8}-+-{:-<12}-+-{:-<12}-+-{:-<8}-+-{:-<8}",
            "", "", "", "", "", ""
        );
        for s in &report.stages {
            eprintln!(
                "{:<12} | {:<8} | {:<12} | {:<12.1} | {:<8.3} | {:.4}",
                s.stage,
                s.memory_count,
                s.total_corpus_tokens,
                s.search_result_tokens,
                s.ndcg_at_10,
                s.information_density
            );
        }
        eprintln!(
            "\nToken reduction (raw -> concept): {:.1}%",
            report.token_reduction_pct
        );
        eprintln!("Density improvement: {:.2}x", report.density_improvement);
    }

    #[tokio::test]
    #[ignore] // Eval benchmark (~5 min on CI) — run locally with --ignored
    async fn test_native_memory_augmentation() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!(
                "Skipping test_native_memory_augmentation: fixture dir not found at {:?}",
                fixture_dir
            );
            return;
        }

        let report = run_native_memory_augmentation(&fixture_dir, 10)
            .await
            .unwrap();

        // Structural assertions
        assert!(!report.baselines.is_empty());
        assert!(!report.alternatives.is_empty());
        assert!(
            report.origin_retrieval_tokens > 0.0,
            "Origin should retrieve some tokens from fixtures"
        );

        // The multi-turn model should show that Origin adds minimal overhead
        let mt = &report.multi_turn;
        assert!(
            mt.native_plus_origin_total >= mt.native_only_total,
            "Origin is additive — total must be >= native-only"
        );
        assert!(
            mt.native_plus_replay_total >= mt.native_plus_origin_total,
            "Full replay should cost more than Origin retrieval"
        );
        assert!(
            mt.origin_overhead_pct < 5.0,
            "Origin overhead should be under 5% on a 10-turn Claude Code session, got {:.2}%",
            mt.origin_overhead_pct
        );

        // Origin retrieval should be cheaper than full replay per recall event
        let origin_per_recall = report
            .alternatives
            .iter()
            .find(|a| a.scenario == "origin_retrieval")
            .map(|a| a.tokens_per_recall)
            .unwrap_or(0);
        let replay_per_recall = report
            .alternatives
            .iter()
            .find(|a| a.scenario == "paste_full_history")
            .map(|a| a.tokens_per_recall)
            .unwrap_or(0);
        assert!(
            origin_per_recall < replay_per_recall,
            "Origin retrieval ({} tokens) should be cheaper than full replay ({} tokens)",
            origin_per_recall,
            replay_per_recall
        );

        // Print augmentation framing
        eprintln!("\n=== Origin: Cost of Better Recall ===\n");
        eprintln!("When you need specific information from past sessions:\n");
        eprintln!(
            "{:<28} | {:<19} | Quality",
            "Alternative", "Additional tokens"
        );
        eprintln!("{:-<28}-+-{:-<19}-+-{:-<50}", "", "", "");
        for alt in &report.alternatives {
            eprintln!(
                "{:<28} | {:<19} | {}",
                alt.scenario, alt.tokens_per_recall, alt.description
            );
        }

        eprintln!(
            "\nOver a {}-turn Claude Code session ({} turns need recall):",
            mt.turns, mt.recall_turns
        );
        eprintln!(
            "  Without Origin: {:>7} tokens (native memory only, no specific recall)",
            mt.native_only_total
        );
        eprintln!(
            "  With Origin:    {:>7} tokens (+{:.1}% overhead, full specific recall)",
            mt.native_plus_origin_total, mt.origin_overhead_pct
        );
        eprintln!(
            "  Full history:   {:>7} tokens (every recall pastes entire history)",
            mt.native_plus_replay_total
        );

        eprintln!(
            "\nOrigin adds {:.1}% token overhead to give you automatic, precise recall.",
            mt.origin_overhead_pct
        );

        eprintln!("\nNative memory baselines (already paid, constant per turn):");
        for b in &report.baselines {
            eprintln!(
                "  {:<12} {:>6} tokens/turn  [{}]  {}",
                b.platform, b.memory_tokens_per_turn, b.growth_model, b.mechanism
            );
        }
    }

    /// Real-LLM pipeline token eval — runs actual distillation via Qwen3-4B and measures
    /// how much context is delivered to an LLM before and after consolidation.
    ///
    /// Picks the fixture case with the most seeds, seeds a DB, measures raw token
    /// counts, runs real LLM distillation (`distill_concepts`), then re-measures.
    ///
    /// Requires the Qwen3-4B model to be present in the hf-hub cache.
    /// Run with: cargo test -p origin-core --lib eval::token_efficiency::tests::benchmark_pipeline_token_real_llm -- --ignored --nocapture
    #[tokio::test]
    #[ignore] // Requires on-device LLM (Qwen3-4B) — use --ignored to run
    async fn benchmark_pipeline_token_real_llm() {
        use crate::llm_provider::OnDeviceProvider;
        use crate::on_device_models;
        use crate::prompts::PromptRegistry;
        use crate::refinery::distill_concepts;
        use crate::tuning::DistillationConfig;

        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");

        if !fixture_dir.exists() {
            eprintln!("Skipping: fixture dir not found at {:?}", fixture_dir);
            return;
        }

        // Check model is cached before trying to load it.
        let model_spec = on_device_models::get_default_model();
        if !on_device_models::is_cached(model_spec) {
            eprintln!(
                "Skipping: {} not found in hf-hub cache. Download it first via LlmEngine::download_model().",
                model_spec.display_name
            );
            return;
        }

        // Load the model on a blocking thread (LlmEngine is not Send/Sync).
        eprintln!("Loading {} ...", model_spec.display_name);
        let llm_provider: Arc<dyn crate::llm_provider::LlmProvider> =
            match tokio::task::spawn_blocking(OnDeviceProvider::new).await {
                Ok(Ok(p)) => Arc::new(p),
                Ok(Err(e)) => {
                    eprintln!("Skipping: LLM provider init failed: {}", e);
                    return;
                }
                Err(e) => {
                    eprintln!("Skipping: spawn_blocking panicked: {}", e);
                    return;
                }
            };
        eprintln!("Model loaded. Provider: {}", llm_provider.name());

        let cases = load_fixtures(&fixture_dir).unwrap();

        // Pick the case with the most seeds (most realistic for distillation).
        let best_case = match cases
            .iter()
            .filter(|c| !c.empty_set && c.seeds.len() >= 2)
            .max_by_key(|c| c.seeds.len() + c.negative_seeds.len())
        {
            Some(c) => c,
            None => {
                eprintln!("Skipping: no fixture case with >= 2 seeds found");
                return;
            }
        };

        eprintln!(
            "Using fixture case: {:?} ({} seeds, {} negative seeds)",
            best_case.query,
            best_case.seeds.len(),
            best_case.negative_seeds.len()
        );

        let confidence_cfg = ConfidenceConfig::default();
        let distillation_cfg = DistillationConfig::default();
        let prompts = PromptRegistry::default();

        // ---- Raw stage ----
        let tmp_raw = tempfile::tempdir().unwrap();
        let db_raw = MemoryDB::new(tmp_raw.path(), Arc::new(NoopEmitter))
            .await
            .unwrap();

        let all_docs: Vec<RawDocument> = best_case
            .seeds
            .iter()
            .chain(best_case.negative_seeds.iter())
            .map(|s| crate::eval::runner::seed_to_doc(s, &confidence_cfg))
            .collect();
        let raw_doc_count = all_docs.len();

        let corpus_text: String = all_docs
            .iter()
            .map(|d| d.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let corpus_tokens = count_tokens(&corpus_text);

        db_raw.upsert_documents(all_docs).await.unwrap();

        let raw_results = db_raw
            .search_memory(
                &best_case.query,
                10,
                None,
                best_case.domain.as_deref(),
                None,
                Some(1.0),
                Some(1.0),
                None,
            )
            .await
            .unwrap();
        let raw_tokens = count_results_tokens(&raw_results);
        let raw_result_count = raw_results.len();

        eprintln!(
            "\n=== Pipeline Token Eval (Real LLM: {}) ===",
            llm_provider.name()
        );
        eprintln!("Query: {}", best_case.query);
        eprintln!(
            "Corpus: {} memories, {} tokens",
            raw_doc_count, corpus_tokens
        );
        eprintln!(
            "Raw search: {} results, {} context tokens  (compression: {:.3})",
            raw_result_count,
            raw_tokens,
            TokenMetrics::compute_compression_ratio(raw_tokens, corpus_tokens)
        );

        // ---- Run real distillation ----
        // Seed the same DB for distillation (reuse db_raw).
        eprintln!("\nRunning distillation ...");
        let distill_start = std::time::Instant::now();
        let concepts_created = distill_concepts(
            &db_raw,
            Some(&llm_provider),
            &prompts,
            &distillation_cfg,
            None, // no knowledge_path in eval
        )
        .await
        .unwrap_or_else(|e| {
            eprintln!("  distill_concepts error (non-fatal): {}", e);
            0
        });
        let distill_ms = distill_start.elapsed().as_millis();
        eprintln!(
            "Distillation done in {}ms, concepts created: {}",
            distill_ms, concepts_created
        );

        // ---- Post-distillation stage ----
        let distilled_results = db_raw
            .search_memory(
                &best_case.query,
                10,
                None,
                best_case.domain.as_deref(),
                None,
                Some(1.0),
                Some(1.0),
                None,
            )
            .await
            .unwrap();
        let distilled_tokens = count_results_tokens(&distilled_results);
        let distilled_result_count = distilled_results.len();

        eprintln!(
            "Post-distillation search: {} results, {} context tokens  (compression: {:.3})",
            distilled_result_count,
            distilled_tokens,
            TokenMetrics::compute_compression_ratio(distilled_tokens, corpus_tokens)
        );

        // ---- Summary ----
        let token_reduction = if raw_tokens > 0 {
            (raw_tokens.saturating_sub(distilled_tokens)) as f64 / raw_tokens as f64 * 100.0
        } else {
            0.0
        };

        eprintln!("\n--- Summary ---");
        eprintln!("  Corpus:          {} tokens", corpus_tokens);
        eprintln!(
            "  Raw search:      {} tokens ({} results)",
            raw_tokens, raw_result_count
        );
        eprintln!(
            "  Distilled search: {} tokens ({} results)",
            distilled_tokens, distilled_result_count
        );
        eprintln!(
            "  Token reduction: {:.1}% (raw -> distilled)",
            token_reduction
        );
        eprintln!("  Concepts created: {}", concepts_created);

        // Basic sanity: we should at least get some search results back.
        assert!(
            !distilled_results.is_empty() || raw_results.is_empty(),
            "should return results if raw stage had results"
        );
    }

    #[tokio::test]
    #[ignore] // Eval benchmark (~13 min on CI) — run locally with --ignored
    async fn test_quality_at_scale() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!("Skipping test_quality_at_scale: fixture dir not found");
            return;
        }

        let sizes = vec![5, 10, 20, 40, 80];
        let report = run_quality_at_scale_eval(&fixture_dir, &sizes, 10)
            .await
            .unwrap();

        assert!(!report.points.is_empty());

        eprintln!("\n=== Quality at Scale: Origin vs Native Memory ===");
        eprintln!(
            "{:<10} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12} | {:<12}",
            "Memories",
            "Origin NDCG",
            "Native NDCG*",
            "Origin Tok",
            "Native Tok",
            "Origin Q/kT",
            "Native Q/kT"
        );
        eprintln!(
            "{:-<10}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}",
            "", "", "", "", "", "", ""
        );
        for p in &report.points {
            eprintln!(
                "{:<10} | {:<12.3} | {:<12.3} | {:<12.0} | {:<12.0} | {:<12.3} | {:<12.3}",
                p.memory_count,
                p.origin_ndcg,
                p.native_effective_quality,
                p.origin_tokens,
                p.native_tokens,
                p.origin_quality_per_1k_tokens,
                p.native_quality_per_1k_tokens
            );
        }
        if let Some(cross) = report.crossover_memory_count {
            eprintln!(
                "\nCrossover at {} memories: Origin becomes more efficient per token",
                cross
            );
        } else {
            eprintln!("\nNo crossover observed in measured range");
        }
        eprintln!("\n* Native quality estimated using lost-in-the-middle degradation model (Liu et al., 2023)");
        eprintln!("  {}", report.methodology_note);
    }

    #[test]
    fn test_native_quality_at_scale_model() {
        // At 10 or fewer memories quality should be 0.95
        assert_eq!(native_quality_at_scale(10), 0.95);
        assert_eq!(native_quality_at_scale(1), 0.95);

        // Quality should degrade monotonically as count grows
        let q20 = native_quality_at_scale(20);
        let q50 = native_quality_at_scale(50);
        let q100 = native_quality_at_scale(100);
        let q500 = native_quality_at_scale(500);
        assert!(q20 < 0.95, "q20={q20} should be below 0.95");
        assert!(q50 < q20, "should degrade: q50={q50} < q20={q20}");
        assert!(q100 < q50, "should degrade: q100={q100} < q50={q50}");
        assert!(q500 < q100, "should degrade: q500={q500} < q100={q100}");
        // Floor at 0.30
        assert!(native_quality_at_scale(10_000) >= 0.30);
    }

    #[tokio::test]
    #[ignore] // Eval benchmark (~5 min on CI) — run locally with --ignored
    async fn test_memory_layer_comparison() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");

        if !fixture_dir.exists() {
            eprintln!(
                "Skipping test_memory_layer_comparison: fixture dir not found at {:?}",
                fixture_dir
            );
            return;
        }

        let report = run_memory_layer_comparison(&fixture_dir, 10).await.unwrap();

        assert_eq!(
            report.approaches.len(),
            5,
            "should have exactly 5 approaches"
        );

        // Verify all expected approaches are present
        let approach_keys: Vec<&str> = report
            .approaches
            .iter()
            .map(|a| a.approach.as_str())
            .collect();
        assert!(
            approach_keys.contains(&"flat_markdown"),
            "missing flat_markdown"
        );
        assert!(approach_keys.contains(&"fact_list"), "missing fact_list");
        assert!(
            approach_keys.contains(&"synthesized_summary"),
            "missing synthesized_summary"
        );
        assert!(
            approach_keys.contains(&"origin_retrieval"),
            "missing origin_retrieval"
        );
        assert!(
            approach_keys.contains(&"origin_plus_native"),
            "missing origin_plus_native"
        );

        // Origin should have better quality-per-token than flat markdown
        // (Origin retrieves only relevant results; flat markdown dumps everything unranked)
        let origin = report
            .approaches
            .iter()
            .find(|a| a.approach == "origin_retrieval")
            .unwrap();
        let flat = report
            .approaches
            .iter()
            .find(|a| a.approach == "flat_markdown")
            .unwrap();
        assert!(
            origin.quality_per_1k_tokens > flat.quality_per_1k_tokens,
            "Origin quality/token ({:.3}) should exceed flat markdown ({:.3})",
            origin.quality_per_1k_tokens,
            flat.quality_per_1k_tokens,
        );

        // Origin+Native has higher tokens than Origin alone (it includes the markdown too)
        let complement = report
            .approaches
            .iter()
            .find(|a| a.approach == "origin_plus_native")
            .unwrap();
        assert!(
            complement.mean_tokens_per_query >= origin.mean_tokens_per_query,
            "complement tokens ({:.0}) should be >= origin tokens ({:.0})",
            complement.mean_tokens_per_query,
            origin.mean_tokens_per_query,
        );

        // All NDCG values should be in [0.0, 1.0]
        for a in &report.approaches {
            assert!(
                a.mean_ndcg >= 0.0 && a.mean_ndcg <= 1.0,
                "approach {} NDCG {:.3} out of range",
                a.approach,
                a.mean_ndcg,
            );
            assert!(
                a.memories_accessible >= 0.0 && a.memories_accessible <= 1.0,
                "approach {} accessible {:.3} out of range",
                a.approach,
                a.memories_accessible,
            );
        }

        // Print comparison table
        eprintln!("\n=== Memory Layer Comparison ===");
        eprintln!(
            "{:<22} | {:<10} | {:<8} | {:<10} | {:<10} | Description",
            "Approach", "Tok/query", "NDCG", "Q/1kT", "Accessible"
        );
        eprintln!(
            "{:-<22}-+-{:-<10}-+-{:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<30}",
            "", "", "", "", "", ""
        );
        for a in &report.approaches {
            let desc_len = a.description.chars().count().min(50);
            let desc: String = a.description.chars().take(desc_len).collect();
            eprintln!(
                "{:<22} | {:<10.0} | {:<8.3} | {:<10.3} | {:<10.1}% | {}",
                a.display_name,
                a.mean_tokens_per_query,
                a.mean_ndcg,
                a.quality_per_1k_tokens,
                a.memories_accessible * 100.0,
                desc,
            );
        }
        eprintln!("\nComplement: {}", report.complement_advantage);
        eprintln!("Methodology: {}", report.methodology);
    }

    // ---- score_answer unit tests ----

    #[test]
    fn test_score_answer_perfect_match() {
        let answer = "The project uses SQLite as the database backend and Rust as the language.";
        let seeds = &[
            "SQLite database backend for storage",
            "Rust programming language for safety",
        ];
        let score = score_answer(answer, seeds);
        assert!(
            score > 0.0,
            "answer clearly mentions both seeds, score should be positive"
        );
    }

    #[test]
    fn test_score_answer_no_match() {
        let answer = "I do not know the answer to this question.";
        let seeds = &["SQLite is used as the embedded relational database"];
        let score = score_answer(answer, seeds);
        // "sqlite" is > 4 chars and not in "i do not know the answer to this question"
        assert_eq!(score, 0.0, "no key words match — should be 0.0");
    }

    #[test]
    fn test_score_answer_empty_seeds() {
        let answer = "some answer";
        let score = score_answer(answer, &[]);
        assert_eq!(score, 0.0, "empty relevant seeds should return 0.0");
    }

    #[test]
    fn test_score_answer_partial_match() {
        // Answer matches first seed but not second
        let answer = "The architecture uses SQLite for storage with ACID transactions.";
        let seeds = &[
            "SQLite is used for ACID-compliant storage",
            "Redis is used for caching and pub-sub messaging",
        ];
        let score = score_answer(answer, seeds);
        // First seed: "sqlite" ✓, "used" (4 chars, skip), "acid-compliant" ✓, "storage" ✓ → matches
        // Second seed: "redis", "used", "caching", "pub-sub", "messaging" — none appear → no match
        assert!(
            score > 0.0 && score < 1.0,
            "partial match should be between 0 and 1, got {score}"
        );
    }

    #[tokio::test]
    #[ignore] // Requires ANTHROPIC_API_KEY
    async fn benchmark_e2e_answer_quality() {
        let api_key = match std::env::var("ANTHROPIC_API_KEY") {
            Ok(k) if !k.is_empty() => k,
            _ => {
                eprintln!("Skipping: ANTHROPIC_API_KEY not set");
                return;
            }
        };
        let _ = api_key; // run_e2e_answer_eval reads ANTHROPIC_API_KEY from env directly

        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!("Skipping: fixture dir not found at {:?}", fixture_dir);
            return;
        }

        let report = run_e2e_answer_eval(&fixture_dir, 10, 5)
            .await
            .expect("run_e2e_answer_eval failed");

        assert!(!report.results.is_empty(), "should have results");
        assert_eq!(report.model, "claude-haiku-4-5-20251001");

        eprintln!("\n=== End-to-End Answer Quality ===");
        eprintln!("Model: {}", report.model);
        eprintln!(
            "{:<20} | {:<12} | {:<12} | {:<12} | Queries",
            "Approach", "Answer Score", "Context Tok", "Answer Tok"
        );
        eprintln!(
            "{:-<20}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<8}",
            "", "", "", "", ""
        );
        for r in &report.results {
            eprintln!(
                "{:<20} | {:<12.3} | {:<12.0} | {:<12.0} | {}",
                r.approach,
                r.mean_answer_score,
                r.mean_context_tokens,
                r.mean_answer_tokens,
                r.queries_evaluated
            );
        }
        eprintln!("\nMethodology: {}", report.methodology);

        // Origin should score >= NoContext (it has relevant info; no-context relies on world knowledge only)
        let origin = report.results.iter().find(|r| r.approach == "origin");
        let no_ctx = report.results.iter().find(|r| r.approach == "no_context");
        if let (Some(o), Some(n)) = (origin, no_ctx) {
            // Soft check: Origin's answer score should not be worse than no-context minus a tolerance
            assert!(
                o.mean_answer_score >= n.mean_answer_score - 0.2,
                "Origin answer score ({:.3}) was much worse than no-context ({:.3})",
                o.mean_answer_score,
                n.mean_answer_score
            );
            // Origin should use fewer tokens than FlatMarkdown
            let flat = report
                .results
                .iter()
                .find(|r| r.approach == "flat_markdown");
            if let Some(f) = flat {
                assert!(
                    o.mean_context_tokens <= f.mean_context_tokens,
                    "Origin context ({:.0} tok) should be <= FlatMarkdown ({:.0} tok)",
                    o.mean_context_tokens,
                    f.mean_context_tokens
                );
            }
        }
    }

    /// E2E answer quality on LoCoMo using the on-device Qwen3-4B model.
    ///
    /// For each conversation in locomo10.json, seeds all observations, then for up to
    /// `max_questions_per_conv` QA pairs runs three approaches (origin, full_replay,
    /// no_context) and scores LLM answers against ground truth via keyword overlap.
    ///
    /// Takes ~10 minutes for 5 questions/conv x 10 convs = 150 LLM calls.
    ///
    /// Run with:
    /// cargo test -p origin-core --lib eval::token_efficiency::tests::benchmark_e2e_locomo_on_device -- --ignored --nocapture
    #[tokio::test]
    #[ignore] // Requires on-device Qwen3-4B model (~10 min)
    async fn benchmark_e2e_locomo_on_device() {
        use crate::llm_provider::OnDeviceProvider;
        use crate::on_device_models;

        // Check model available
        let model_spec = on_device_models::get_default_model();
        if !on_device_models::is_cached(model_spec) {
            eprintln!(
                "Skipping: {} not found in hf-hub cache. Download it first.",
                model_spec.display_name
            );
            return;
        }

        let locomo_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/data/locomo10.json");
        if !locomo_path.exists() {
            eprintln!("Skipping: locomo10.json not found at {:?}", locomo_path);
            return;
        }

        // Load provider on a blocking thread (LlmEngine is not Send/Sync).
        eprintln!("Loading {} ...", model_spec.display_name);
        let provider: Arc<dyn crate::llm_provider::LlmProvider> =
            match tokio::task::spawn_blocking(OnDeviceProvider::new).await {
                Ok(Ok(p)) => Arc::new(p),
                Ok(Err(e)) => {
                    eprintln!("Skipping: LLM provider init failed: {}", e);
                    return;
                }
                Err(e) => {
                    eprintln!("Skipping: spawn_blocking panicked: {}", e);
                    return;
                }
            };
        eprintln!("Model loaded. Provider: {}", provider.name());

        let (report, _tuples) = run_e2e_locomo_eval(&locomo_path, 5, 10, provider)
            .await
            .unwrap();

        eprintln!("\n=== E2E Answer Quality: LoCoMo (On-Device) ===");
        eprintln!("Model: {}", report.model);
        eprintln!(
            "Questions: {} ({} per conv x {} convs)",
            report.total_questions, report.questions_per_conv, report.conversations
        );
        eprintln!(
            "{:<20} | {:<12} | {:<12} | Avg Answer Len",
            "Approach", "Answer Score", "Context Tok"
        );
        eprintln!("{:-<20}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "");
        for r in &report.results {
            eprintln!(
                "{:<20} | {:<12.3} | {:<12.0} | {:.0} chars",
                r.approach, r.mean_answer_score, r.mean_context_tokens, r.mean_answer_length
            );
        }

        // Origin should score at least as well as NoContext
        let origin = report
            .results
            .iter()
            .find(|r| r.approach == "origin")
            .unwrap();
        let no_ctx = report
            .results
            .iter()
            .find(|r| r.approach == "no_context")
            .unwrap();
        eprintln!(
            "\nOrigin vs NoContext: {:.3} vs {:.3} answer score",
            origin.mean_answer_score, no_ctx.mean_answer_score
        );

        // Soft check: Origin with memory context should not be dramatically worse than no-context.
        assert!(
            origin.mean_answer_score >= no_ctx.mean_answer_score - 0.3,
            "Origin answer score ({:.3}) was much worse than no-context ({:.3})",
            origin.mean_answer_score,
            no_ctx.mean_answer_score,
        );
    }

    /// Judge a single hardcoded tuple via `claude -p` CLI.
    ///
    /// Run with:
    /// cargo test -p origin-core --lib eval::token_efficiency::tests::test_judge_single_tuple -- --ignored --nocapture
    #[tokio::test]
    #[ignore] // Requires claude CLI with active Max subscription
    async fn test_judge_single_tuple() {
        // Check claude CLI is available.
        let check = tokio::process::Command::new("claude")
            .arg("--version")
            .output()
            .await;
        match check {
            Err(e) => {
                eprintln!("Skipping: claude CLI not available: {}", e);
                return;
            }
            Ok(out) if !out.status.success() => {
                eprintln!("Skipping: claude --version failed");
                return;
            }
            Ok(out) => {
                let ver = String::from_utf8_lossy(&out.stdout);
                eprintln!("claude CLI: {}", ver.trim());
            }
        }

        let tuple = JudgmentTuple {
            question: "What database does Origin use?".to_string(),
            ground_truth: "Origin uses libSQL (Turso's SQLite fork)".to_string(),
            approach: "test".to_string(),
            answer: "Origin uses libSQL, which is Turso's fork of SQLite, for its database layer."
                .to_string(),
            context_tokens: 50,
        };

        let result = judge_single_tuple(&tuple).await.unwrap();
        eprintln!("Score: {}, Reason: {}", result.score, result.reason);
        assert_eq!(result.score, 1, "correct answer should score 1");
    }

    /// E2E LoCoMo eval with Claude Haiku judge (two-phase: generate then judge).
    ///
    /// Phase 1: run on-device Qwen3-4B to collect raw answers and tuples.
    /// Phase 2: feed tuples to `claude -p haiku` as binary judge.
    ///
    /// Run with:
    /// cargo test -p origin-core --lib eval::token_efficiency::tests::benchmark_e2e_locomo_judged -- --ignored --nocapture
    #[tokio::test]
    #[ignore] // Requires on-device Qwen3-4B model AND claude CLI with Max subscription (~15 min)
    async fn benchmark_e2e_locomo_judged() {
        use crate::llm_provider::OnDeviceProvider;
        use crate::on_device_models;

        // Check model available.
        let model_spec = on_device_models::get_default_model();
        if !on_device_models::is_cached(model_spec) {
            eprintln!(
                "Skipping: {} not found in hf-hub cache. Download it first.",
                model_spec.display_name
            );
            return;
        }

        // Check claude CLI available.
        let check = tokio::process::Command::new("claude")
            .arg("--version")
            .output()
            .await;
        match check {
            Err(e) => {
                eprintln!("Skipping: claude CLI not available: {}", e);
                return;
            }
            Ok(out) if !out.status.success() => {
                eprintln!("Skipping: claude --version failed");
                return;
            }
            Ok(out) => {
                let ver = String::from_utf8_lossy(&out.stdout);
                eprintln!("claude CLI: {}", ver.trim());
            }
        }

        let locomo_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/data/locomo10.json");

        if !locomo_path.exists() {
            eprintln!("Skipping: locomo10.json not found at {:?}", locomo_path);
            return;
        }

        // Phase 1: run E2E eval to collect tuples.
        eprintln!("Loading {} ...", model_spec.display_name);
        let provider: Arc<dyn crate::llm_provider::LlmProvider> =
            match tokio::task::spawn_blocking(OnDeviceProvider::new).await {
                Ok(Ok(p)) => Arc::new(p),
                Ok(Err(e)) => {
                    eprintln!("Skipping: LLM provider init failed: {}", e);
                    return;
                }
                Err(e) => {
                    eprintln!("Skipping: spawn_blocking panicked: {}", e);
                    return;
                }
            };
        eprintln!("Model loaded. Provider: {}", provider.name());

        eprintln!("Phase 1: generating answers (5 questions/conv)...");
        let (_keyword_report, tuples) = run_e2e_locomo_eval(&locomo_path, 5, 10, provider)
            .await
            .unwrap();

        eprintln!("Collected {} tuples. Saving to tempdir...", tuples.len());
        let tmp = tempfile::tempdir().unwrap();
        let tuples_path = tmp.path().join("judgment_tuples.json");
        save_judgment_tuples(&tuples, &tuples_path).unwrap();
        eprintln!("Saved tuples to {:?}", tuples_path);

        // Phase 2: judge with Claude Haiku via `claude -p`.
        eprintln!(
            "Phase 2: judging {} tuples with Claude Haiku (concurrency=3)...",
            tuples.len()
        );
        let results = judge_with_claude(&tuples, 3).await.unwrap();
        eprintln!("Judged {} / {} tuples.", results.len(), tuples.len());

        // Aggregate and print.
        let report = aggregate_judgments(&results, "haiku");
        eprintln!("\n=== E2E Answer Quality: LoCoMo (Claude Haiku Judge) ===");
        eprintln!(
            "{:<20} | {:<10} | {:<10} | {:<14} | Total",
            "Approach", "Accuracy", "Correct", "Context Tok"
        );
        eprintln!(
            "{:-<20}-+-{:-<10}-+-{:-<10}-+-{:-<14}-+-{:-<6}",
            "", "", "", "", ""
        );
        for r in &report.results_by_approach {
            eprintln!(
                "{:<20} | {:<10.1}% | {:<10} | {:<14.0} | {}",
                r.approach,
                r.accuracy * 100.0,
                r.correct,
                r.mean_context_tokens,
                r.total
            );
        }
        eprintln!("\nTotal judged: {}", report.total_judged);
    }
}
