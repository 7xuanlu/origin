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
    pub mean_compression_ratio: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
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

        let col_widths = (16usize, 10usize, 8usize, 10usize, 13usize, 13usize);
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
            out.push_str(&format!(
                "{:<w0$}  {:>w1$.4}  {:>w2$.4}  {:>w3$.4}  {:>w4$.1}  {:>w5$.4}\n",
                s.strategy,
                s.ndcg_at_10,
                s.mrr,
                s.recall_at_5,
                s.mean_context_tokens,
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
                            None,
                            None,
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
        let median_ctx = {
            let mut sorted = ctx_vec.clone();
            sorted.sort_unstable();
            if sorted.is_empty() {
                0.0
            } else if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) as f64 / 2.0
            } else {
                sorted[sorted.len() / 2] as f64
            }
        };
        let mean_comp = comp_vec.iter().sum::<f64>() / n;
        let mean_ndcg = ndcg_vec.iter().sum::<f64>() / n;
        let mean_mrr = mrr_vec.iter().sum::<f64>() / n;
        let mean_r5 = r5_vec.iter().sum::<f64>() / n;

        strategy_reports.push(StrategyReport {
            strategy: strategy.name().to_string(),
            mean_context_tokens: mean_ctx,
            median_context_tokens: median_ctx,
            mean_compression_ratio: mean_comp,
            ndcg_at_10: mean_ndcg,
            mrr: mean_mrr,
            recall_at_5: mean_r5,
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

/// Count total tokens across a slice of SearchResults (content field only).
fn count_results_tokens(results: &[crate::db::SearchResult]) -> usize {
    results.iter().map(|r| count_tokens(&r.content)).sum()
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

            // Origin: search and count
            let results = db
                .search_memory(
                    &case.query, limit, None, case.domain.as_deref(),
                    None, None, None, None,
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

        assert_eq!(SearchStrategy::Origin.display_name(), "Origin");
        assert_eq!(SearchStrategy::OriginReranked.display_name(), "Origin+Rerank");
        assert_eq!(SearchStrategy::OriginExpanded.display_name(), "Origin+Expand");
        assert_eq!(SearchStrategy::NaiveRag.display_name(), "Naive RAG");
        assert_eq!(SearchStrategy::FullReplay.display_name(), "Full Replay");
        assert_eq!(SearchStrategy::NoMemory.display_name(), "No Memory");
    }

    #[test]
    fn test_strategy_requires_llm() {
        assert!(!SearchStrategy::Origin.requires_llm());
        assert!(SearchStrategy::OriginReranked.requires_llm());
        assert!(SearchStrategy::OriginExpanded.requires_llm());
        assert!(!SearchStrategy::NaiveRag.requires_llm());
        assert!(!SearchStrategy::FullReplay.requires_llm());
        assert!(!SearchStrategy::NoMemory.requires_llm());
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
                    mean_compression_ratio: 0.12,
                    ndcg_at_10: 0.82,
                    mrr: 0.75,
                    recall_at_5: 0.68,
                },
                StrategyReport {
                    strategy: "full_replay".to_string(),
                    mean_context_tokens: 1000.0,
                    median_context_tokens: 950.0,
                    mean_compression_ratio: 1.0,
                    ndcg_at_10: 1.0,
                    mrr: 1.0,
                    recall_at_5: 1.0,
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
                mean_compression_ratio: 0.05,
                ndcg_at_10: 0.9,
                mrr: 0.88,
                recall_at_5: 0.77,
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
}
