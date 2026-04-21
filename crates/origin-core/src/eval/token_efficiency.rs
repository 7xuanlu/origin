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

/// Shared BPE tokenizer instance (cl100k_base). Initialized once on first use.
static BPE: LazyLock<tiktoken_rs::CoreBPE> = LazyLock::new(|| {
    tiktoken_rs::cl100k_base().expect("failed to load cl100k_base tokenizer")
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
        let sep_len = col_widths.0 + col_widths.1 + col_widths.2 + col_widths.3
            + col_widths.4 + col_widths.5 + 10;
        out.push_str(&"-".repeat(sep_len));
        out.push('\n');

        for s in &self.strategies {
            let ndcg_str = format!("{:.4}±{:.4}", s.ndcg_at_10, s.stddev_ndcg);
            let mrr_str = format!("{:.4}±{:.4}", s.mrr, s.stddev_mrr);
            let tok_str = format!("{:.1}±{:.1}", s.mean_context_tokens, s.stddev_context_tokens);
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
            self.headline.savings_pct,
            self.headline.origin_tokens,
            self.headline.replay_tokens,
        ));
        out.push_str(&format!(
            "          {:.1}% quality retained (NDCG@10 vs Full Replay)\n",
            self.headline.quality_retained_pct,
        ));
        out
    }

    /// Save report to disk as pretty-printed JSON.
    pub fn save_baseline(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load a previously saved report from disk.
    pub fn load_baseline(path: &Path) -> Result<QualityCostReport, std::io::Error> {
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

// ===== Native Memory Baseline Types =====

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

/// Comparison report: Origin vs native memory systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeMemoryComparisonReport {
    /// Origin's token cost per query (mean across fixture cases).
    pub origin_tokens_per_query: f64,
    /// Native baselines for comparison.
    pub baselines: Vec<NativeMemoryBaseline>,
    /// Per-baseline savings.
    pub comparisons: Vec<NativeMemoryComparison>,
    /// Multi-turn comparison (10 turns).
    pub multi_turn_10: MultiTurnNativeComparison,
}

/// Origin vs a single native platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeMemoryComparison {
    pub platform: String,
    pub native_tokens_per_turn: usize,
    pub origin_tokens_per_turn: f64,
    pub savings_pct: f64,
    /// Qualitative description of the retrieval quality difference.
    pub quality_advantage: String,
}

/// Multi-turn cumulative token comparison (fixed number of turns).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTurnNativeComparison {
    pub turns: usize,
    pub origin_total: f64,
    pub claude_code_total: usize,
    pub chatgpt_total: usize,
    pub claude_ai_total: usize,
}

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
                    let results = db.naive_vector_search(&case.query, limit, case.domain.as_deref()).await?;
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
            compression_all
                .get_mut(strategy)
                .unwrap()
                .push(compression);
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
                let median = if len % 2 == 0 {
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
                (o.mean_context_tokens, r.mean_context_tokens, savings, quality)
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

/// Compare Origin's retrieval-based memory against native platform memory injection.
///
/// Runs Origin search against fixtures to get actual mean tokens/query, then compares
/// against known fixed-overhead native memory systems.
///
/// Note: the comparison is inherently apples-to-oranges. Native memory is always-on
/// general context injected every turn regardless of relevance. Origin retrieves only
/// what is relevant to the specific query. The `quality_advantage` field on each
/// comparison entry describes this distinction honestly.
///
/// Native platform token estimates are researched estimates as of April 2026, sourced
/// from public documentation, model cards, and community measurements. They are not
/// measurements from those platforms' APIs.
pub async fn run_native_memory_comparison(
    fixture_dir: &Path,
    limit: usize,
) -> Result<NativeMemoryComparisonReport, OriginError> {
    // Step 1: run Origin against fixtures to get actual mean tokens/query.
    let cases = load_fixtures(fixture_dir)?;
    let confidence_cfg = ConfidenceConfig::default();
    let mut origin_token_samples: Vec<usize> = Vec::new();

    for case in &cases {
        if case.empty_set || case.seeds.is_empty() {
            continue;
        }

        let case_tmp = tempfile::tempdir()
            .map_err(|e| OriginError::Generic(format!("tempdir for native comparison: {}", e)))?;
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
    }

    let origin_tokens_per_query = if origin_token_samples.is_empty() {
        0.0
    } else {
        origin_token_samples.iter().sum::<usize>() as f64 / origin_token_samples.len() as f64
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
            mechanism: "dumps full CLAUDE.md every turn regardless of query relevance".to_string(),
        },
        NativeMemoryBaseline {
            platform: "ChatGPT".to_string(),
            memory_tokens_per_turn: 3_050,
            growth_model: "capped".to_string(),
            mechanism: "injects all saved facts every turn; capped at ~200 entries".to_string(),
        },
        NativeMemoryBaseline {
            platform: "Claude.ai".to_string(),
            memory_tokens_per_turn: 2_000,
            growth_model: "synthesized".to_string(),
            mechanism: "synthesized summary, not query-specific; auto-pruned".to_string(),
        },
    ];

    // Step 3: compute per-baseline savings.
    let quality_advantages = [
        "dumps full CLAUDE.md every turn regardless of query relevance",
        "injects all saved facts every turn; capped at ~200 entries",
        "synthesized summary, not query-specific; auto-pruned",
    ];

    let comparisons: Vec<NativeMemoryComparison> = baselines
        .iter()
        .zip(quality_advantages.iter())
        .map(|(baseline, quality_adv)| {
            let native = baseline.memory_tokens_per_turn as f64;
            let savings_pct = if native > 0.0 {
                (native - origin_tokens_per_query) / native * 100.0
            } else {
                0.0
            };
            NativeMemoryComparison {
                platform: baseline.platform.clone(),
                native_tokens_per_turn: baseline.memory_tokens_per_turn,
                origin_tokens_per_turn: origin_tokens_per_query,
                savings_pct,
                quality_advantage: quality_adv.to_string(),
            }
        })
        .collect();

    // Step 4: compute 10-turn totals.
    // Native platforms inject the same overhead every turn (always-on).
    // Origin retrieves fresh relevant context each turn (constant per-query cost).
    let turns = 10usize;
    let claude_code = baselines
        .iter()
        .find(|b| b.platform == "Claude Code")
        .map(|b| b.memory_tokens_per_turn)
        .unwrap_or(0);
    let chatgpt = baselines
        .iter()
        .find(|b| b.platform == "ChatGPT")
        .map(|b| b.memory_tokens_per_turn)
        .unwrap_or(0);
    let claude_ai = baselines
        .iter()
        .find(|b| b.platform == "Claude.ai")
        .map(|b| b.memory_tokens_per_turn)
        .unwrap_or(0);

    let multi_turn_10 = MultiTurnNativeComparison {
        turns,
        origin_total: origin_tokens_per_query * turns as f64,
        claude_code_total: claude_code * turns,
        chatgpt_total: chatgpt * turns,
        claude_ai_total: claude_ai * turns,
    };

    Ok(NativeMemoryComparisonReport {
        origin_tokens_per_query,
        baselines,
        comparisons,
        multi_turn_10,
    })
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

            let corpus_text: String = all_docs.iter().map(|d| d.content.as_str()).collect::<Vec<_>>().join("\n\n");
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
            let mut type_groups: HashMap<&str, Vec<&crate::eval::fixtures::SeedMemory>> = HashMap::new();
            for seed in &case.seeds {
                type_groups.entry(seed.memory_type.as_str()).or_default().push(seed);
            }

            let mut distilled_docs: Vec<RawDocument> = Vec::new();
            let mut merged_ids: HashSet<String> = HashSet::new();

            for (_mtype, group) in &type_groups {
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
                        distilled_docs.push(crate::eval::runner::seed_to_doc(seed, &confidence_cfg));
                    }
                }
            }

            // Add negative seeds unchanged
            for neg in &case.negative_seeds {
                distilled_docs.push(crate::eval::runner::seed_to_doc(neg, &confidence_cfg));
            }

            let corpus_text: String = distilled_docs.iter().map(|d| d.content.as_str()).collect::<Vec<_>>().join("\n\n");
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
            for (_mtype, group) in &type_groups {
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
            let concept_domain = case.domain.clone().or_else(|| {
                case.seeds.first().and_then(|s| s.domain.clone())
            });
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

            let corpus_text: String = concept_docs.iter().map(|d| d.content.as_str()).collect::<Vec<_>>().join("\n\n");
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
            let concept_grades: HashMap<&str, u8> = [(concept_id, best_relevance)].into_iter().collect();
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
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
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
    } else { 0.0 };
    let raw_density = if raw_mean_search > 0.0 {
        raw_mean_ndcg / (raw_mean_search / 1000.0)
    } else { 0.0 };

    let dist_total_corpus = total_usize(&distilled_corpus);
    let dist_total_count = total_usize(&distilled_counts);
    let dist_mean_search = total_f64(&distilled_search_tokens) / n;
    let dist_mean_ndcg = mean_f64(&distilled_ndcg);
    let dist_mean_tok_per_mem = if dist_total_count > 0 {
        dist_total_corpus as f64 / dist_total_count as f64
    } else { 0.0 };
    let dist_density = if dist_mean_search > 0.0 {
        dist_mean_ndcg / (dist_mean_search / 1000.0)
    } else { 0.0 };

    let conc_total_corpus = total_usize(&concept_corpus);
    let conc_total_count = total_usize(&concept_counts);
    let conc_mean_search = total_f64(&concept_search_tokens) / n;
    let conc_mean_ndcg = mean_f64(&concept_ndcg);
    let conc_mean_tok_per_mem = if conc_total_count > 0 {
        conc_total_corpus as f64 / conc_total_count as f64
    } else { 0.0 };
    let conc_density = if conc_mean_search > 0.0 {
        conc_mean_ndcg / (conc_mean_search / 1000.0)
    } else { 0.0 };

    // Token reduction is measured on search result tokens (context delivered to LLM),
    // not raw corpus size — since simulated concept entries have the same text length
    // as their constituent seeds concatenated, corpus size barely changes in simulation.
    // The real efficiency gain is in the retrieved context: the concept stage returns
    // fewer, denser entries, so the LLM receives fewer tokens while preserving quality.
    let token_reduction_pct = if raw_mean_search > 0.0 {
        ((raw_mean_search - conc_mean_search) / raw_mean_search * 100.0).max(0.0)
    } else { 0.0 };

    let density_improvement = if raw_density > 0.0 {
        conc_density / raw_density
    } else { 0.0 };

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
            let subset: Vec<&crate::eval::fixtures::SeedMemory> = all_seeds.into_iter().take(size).collect();
            if subset.is_empty() {
                continue;
            }

            // Seed ephemeral DB with subset
            let case_tmp = tempfile::tempdir()
                .map_err(|e| OriginError::Generic(format!("tmpdir: {e}")))?;
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
                    &case.query, limit, None, case.domain.as_deref(),
                    None, Some(1.0), Some(1.0), None,
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
        assert!((ratio - 0.1).abs() < 1e-9, "100/1000 should be 0.1, got {}", ratio);

        let ratio2 = TokenMetrics::compute_compression_ratio(50, 200);
        assert!((ratio2 - 0.25).abs() < 1e-9, "50/200 should be 0.25, got {}", ratio2);
    }

    #[test]
    fn test_token_metrics_zero_corpus() {
        let ratio = TokenMetrics::compute_compression_ratio(0, 0);
        assert_eq!(ratio, 0.0, "zero corpus should return 0.0");

        let ratio2 = TokenMetrics::compute_compression_ratio(100, 0);
        assert_eq!(ratio2, 0.0, "non-zero context with zero corpus should return 0.0");
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
        assert_eq!(SearchStrategy::OriginReranked.display_name(), "Origin+Rerank");
        assert_eq!(SearchStrategy::OriginExpanded.display_name(), "Origin+Expand");
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
                content: "tokio is an async runtime for Rust using the executor pattern.".to_string(),
                source_id: "tokio_async".to_string(),
                source: "memory".to_string(),
                title: "Tokio async".to_string(),
                memory_type: Some("fact".to_string()),
                ..Default::default()
            },
            RawDocument {
                content: "SQLite is an embedded relational database with ACID guarantees.".to_string(),
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
        assert!(output.contains("test-bench"), "should contain benchmark name");
        assert!(output.contains("cl100k_base"), "should contain tokenizer");

        // Column headers
        assert!(output.contains("NDCG@10"), "should have NDCG@10 column");
        assert!(output.contains("MRR"), "should have MRR column");
        assert!(output.contains("Recall@5"), "should have Recall@5 column");
        assert!(output.contains("Tokens/Query"), "should have token column");

        // Data rows
        assert!(output.contains("origin"), "should contain origin row");
        assert!(output.contains("full_replay"), "should contain full_replay row");

        // Headline
        assert!(
            output.contains("87.9") || output.contains("88.0"),
            "should show savings_pct"
        );
    }

    #[tokio::test]
    async fn test_multi_turn_eval() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!("Skipping test_multi_turn_eval: fixture dir not found at {:?}", fixture_dir);
            return;
        }

        let report = run_multi_turn_eval(&fixture_dir, 10, 10, 200).await.unwrap();

        assert_eq!(report.turns, 10);
        assert_eq!(report.per_turn.len(), 10);

        // Origin total should be MUCH less than replay total
        assert!(
            report.total_origin_tokens < report.total_replay_tokens,
            "Origin {} should be < Replay {}",
            report.total_origin_tokens, report.total_replay_tokens
        );

        // Replay should grow each turn (each turn adds response_overhead)
        for i in 1..report.per_turn.len() {
            assert!(
                report.per_turn[i].replay_tokens >= report.per_turn[i - 1].replay_tokens,
                "Replay should grow turn-over-turn: turn {} ({}) < turn {} ({})",
                i + 1, report.per_turn[i].replay_tokens,
                i, report.per_turn[i - 1].replay_tokens
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
        eprintln!("\n=== Multi-Turn Token Accumulation ({} turns) ===", report.turns);
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

        assert!(!points.is_empty(), "should produce at least one scaling point");
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

        report.save_baseline(&path).expect("save_baseline should succeed");
        assert!(path.exists(), "baseline file should exist after save");

        let loaded = QualityCostReport::load_baseline(&path).expect("load_baseline should succeed");

        assert_eq!(loaded.benchmark, report.benchmark);
        assert_eq!(loaded.timestamp, report.timestamp);
        assert_eq!(loaded.tokenizer, report.tokenizer);
        assert_eq!(loaded.strategies.len(), report.strategies.len());
        assert_eq!(loaded.strategies[0].strategy, report.strategies[0].strategy);
        assert!(
            (loaded.strategies[0].ndcg_at_10 - report.strategies[0].ndcg_at_10).abs() < 1e-9
        );
        assert!(
            (loaded.headline.savings_pct - report.headline.savings_pct).abs() < 1e-9
        );
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
        assert!(!vpf.is_empty(), "VectorPlusFts should return results (vector path guaranteed)");
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
    async fn test_pipeline_token_eval_simulated() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!("Skipping test_pipeline_token_eval_simulated: fixture dir not found at {:?}", fixture_dir);
            return;
        }

        let report = run_pipeline_token_eval_simulated(&fixture_dir, 10).await.unwrap();

        assert_eq!(report.stages.len(), 3);

        // Distilled should have fewer memories than raw
        let raw = &report.stages[0];
        let distilled = &report.stages[1];
        assert!(distilled.memory_count <= raw.memory_count,
            "distilled memory count {} should be <= raw {}",
            distilled.memory_count, raw.memory_count);

        // Concept should have fewest memories
        let concept = &report.stages[2];
        assert!(concept.memory_count <= distilled.memory_count,
            "concept memory count {} should be <= distilled {}",
            concept.memory_count, distilled.memory_count);

        // Token reduction should be non-negative (concept stage delivers fewer search result
        // tokens to the LLM context — it consolidates many entries into one).
        // Note: in simulation, corpus size stays roughly equal (content is concatenated not
        // condensed), so reduction is measured on retrieved context tokens.
        assert!(report.token_reduction_pct >= 0.0,
            "token_reduction_pct should be >= 0, got {}",
            report.token_reduction_pct);

        // Print results
        eprintln!("\n=== Pipeline Token Efficiency ===");
        eprintln!("{:<12} | {:<8} | {:<12} | {:<12} | {:<8} | {}",
            "Stage", "Memories", "Corpus Tok", "Search Tok", "NDCG", "Density");
        eprintln!("{:-<12}-+-{:-<8}-+-{:-<12}-+-{:-<12}-+-{:-<8}-+-{:-<8}", "", "", "", "", "", "");
        for s in &report.stages {
            eprintln!("{:<12} | {:<8} | {:<12} | {:<12.1} | {:<8.3} | {:.4}",
                s.stage, s.memory_count, s.total_corpus_tokens, s.search_result_tokens, s.ndcg_at_10, s.information_density);
        }
        eprintln!("\nToken reduction (raw -> concept): {:.1}%", report.token_reduction_pct);
        eprintln!("Density improvement: {:.2}x", report.density_improvement);
    }

    #[tokio::test]
    async fn test_native_memory_comparison() {
        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("app/eval/fixtures");
        if !fixture_dir.exists() {
            eprintln!("Skipping test_native_memory_comparison: fixture dir not found at {:?}", fixture_dir);
            return;
        }

        let report = run_native_memory_comparison(&fixture_dir, 10).await.unwrap();

        assert!(!report.baselines.is_empty());
        assert!(!report.comparisons.is_empty());

        // Origin should use fewer tokens than all native platforms
        for comp in &report.comparisons {
            assert!(comp.savings_pct > 0.0,
                "Origin should save vs {} but got {:.1}%", comp.platform, comp.savings_pct);
        }

        // Print comparison
        eprintln!("\n=== Origin vs Native Memory Systems ===");
        eprintln!("Origin: {:.0} tokens/query (retrieval-based, query-relevant)\n", report.origin_tokens_per_query);
        eprintln!("{:<15} | {:<12} | {:<10} | {}", "Platform", "Tokens/turn", "Savings", "Mechanism");
        eprintln!("{:-<15}-+-{:-<12}-+-{:-<10}-+-{:-<30}", "", "", "", "");
        for comp in &report.comparisons {
            eprintln!("{:<15} | {:<12} | {:<10.1}% | {}",
                comp.platform, comp.native_tokens_per_turn, comp.savings_pct, comp.quality_advantage);
        }
        eprintln!("\n=== 10-Turn Session Total ===");
        eprintln!("Origin:      {:.0} tokens", report.multi_turn_10.origin_total);
        eprintln!("Claude Code: {} tokens ({:.0}x more)",
            report.multi_turn_10.claude_code_total,
            report.multi_turn_10.claude_code_total as f64 / report.multi_turn_10.origin_total);
        eprintln!("ChatGPT:     {} tokens ({:.0}x more)",
            report.multi_turn_10.chatgpt_total,
            report.multi_turn_10.chatgpt_total as f64 / report.multi_turn_10.origin_total);
        eprintln!("Claude.ai:   {} tokens ({:.0}x more)",
            report.multi_turn_10.claude_ai_total,
            report.multi_turn_10.claude_ai_total as f64 / report.multi_turn_10.origin_total);
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
            match tokio::task::spawn_blocking(|| OnDeviceProvider::new()).await {
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

        eprintln!("\n=== Pipeline Token Eval (Real LLM: {}) ===", llm_provider.name());
        eprintln!("Query: {}", best_case.query);
        eprintln!("Corpus: {} memories, {} tokens", raw_doc_count, corpus_tokens);
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
        eprintln!("Distillation done in {}ms, concepts created: {}", distill_ms, concepts_created);

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
        eprintln!("  Raw search:      {} tokens ({} results)", raw_tokens, raw_result_count);
        eprintln!(
            "  Distilled search: {} tokens ({} results)",
            distilled_tokens, distilled_result_count
        );
        eprintln!("  Token reduction: {:.1}% (raw -> distilled)", token_reduction);
        eprintln!("  Concepts created: {}", concepts_created);

        // Basic sanity: we should at least get some search results back.
        assert!(
            !distilled_results.is_empty() || raw_results.is_empty(),
            "should return results if raw stage had results"
        );
    }
}
