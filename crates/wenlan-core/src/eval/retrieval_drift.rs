// SPDX-License-Identifier: Apache-2.0
//! Differential / equivalence oracle for retrieval ranking (drift detector).
//! Freezes retrieval rankings over the eval fixtures as a committed golden, then
//! asserts the live ranking still agrees (top-weighted RBO) — catching silent
//! reorderings the noisy labeled NDCG eval (`eval::retrieval`) misses.
//! HONEST FRAMING: detects DRIFT from a frozen reference, NOT absolute correctness.
//! The golden is trustworthy only because the labeled NDCG eval vouches each baseline.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::events::NoopEmitter;
use crate::read_scope::ReadScope;

#[allow(dead_code)]
const TOP_K: usize = 10;
#[allow(dead_code)]
const RBO_P: f64 = 0.9; // persistence: top-2 swap ~0.85, deep swap ~0.99 — top weighted ~10× depth-10
#[allow(dead_code)]
const CURRENT_RBO_THRESHOLD: f64 = 0.95; // 0.95: above a benign deep swap (~0.99), below a top-2 swap (~0.85)
#[allow(dead_code)]
const ANCHOR_RBO_THRESHOLD: f64 = 0.85; // 0.85: looser floor — full-reverse lands ~0.25, well below this

/// Per-query frozen ranking snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRanking {
    /// Stable per-case identity (query + space + sorted seed ids). Used to match
    /// golden↔live — NOT the query string, which is non-unique across fixture cases
    /// (distinct corpora can share a natural-language query), and `load_fixtures`
    /// walks the dir in non-deterministic order, so a query-keyed match mispairs.
    pub key: String,
    pub query: String,
    pub ranked_ids: Vec<String>,
    /// Stored for DIAGNOSIS only — never asserted exactly (cross-arch f32 ULP drift).
    pub scores: Vec<f32>,
}

/// Stable identity for a fixture case: query + space + sorted seed ids. Distinct
/// corpora that share a query string get distinct keys; order-independent so the
/// non-deterministic fixture load order can't mispair golden↔live.
fn case_key(case: &crate::eval::fixtures::EvalCase) -> String {
    let mut ids: Vec<&str> = case.seeds.iter().map(|s| s.id.as_str()).collect();
    ids.sort_unstable();
    format!(
        "{}\u{241f}{}\u{241f}{}",
        case.query,
        case.space.as_deref().unwrap_or(""),
        ids.join(",")
    )
}

/// Full golden snapshot for a fixture set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingGolden {
    pub generated_env: String,
    pub fixture_set_hash: String,
    pub top_k: usize,
    pub queries: Vec<QueryRanking>,
}

/// Serialize a [`RankingGolden`] to pretty-printed JSON at `path`.
pub fn save_golden(golden: &RankingGolden, path: &Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(golden).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Deserialize a [`RankingGolden`] from a JSON file at `path`.
pub fn load_golden(path: &Path) -> std::io::Result<RankingGolden> {
    let raw = std::fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(std::io::Error::other)
}

/// Capture current retrieval rankings over the eval fixtures.
///
/// Mirrors the labeled eval's `Wenlan` strategy seeding + search EXACTLY (see
/// `eval::retrieval::run_quality_cost_eval`) so the golden measures the same
/// ranking the labeled NDCG eval grades — the layering only holds if both rank
/// identically. Skips `empty_set` / seedless cases (no meaningful ranking).
///
/// Deterministic: FastEmbed (BGE-Base-Q) + the reranker have no thread/seed
/// config, so the same fixture text yields the same ranking on a given arch.
/// Cross-arch f32 ULP is absorbed by the RBO tolerance at the gate, not here.
pub async fn capture_rankings(
    fixture_dir: &Path,
    top_k: usize,
) -> Result<RankingGolden, WenlanError> {
    let cases = crate::eval::fixtures::load_fixtures(fixture_dir)?;
    let shared_embedder = crate::eval::shared::eval_shared_embedder();
    let confidence_cfg = crate::tuning::ConfidenceConfig::default();
    let fixture_set_hash = crate::eval::fixtures::fixture_set_hash(fixture_dir)
        .map_err(|e| WenlanError::Generic(format!("fixture_set_hash: {e}")))?;

    let mut queries = Vec::new();
    for case in &cases {
        if case.empty_set || case.seeds.is_empty() {
            continue;
        }
        let case_tmp = tempfile::tempdir()
            .map_err(|e| WenlanError::Generic(format!("tempdir for drift capture: {e}")))?;
        let db = MemoryDB::new_with_shared_embedder(
            case_tmp.path(),
            Arc::new(NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;
        let all_docs: Vec<crate::sources::RawDocument> = case
            .seeds
            .iter()
            .chain(case.negative_seeds.iter())
            .map(|seed| crate::eval::runner::seed_to_doc(seed, &confidence_cfg))
            .collect();
        db.upsert_documents(all_docs).await?;
        // EXACT mirror of the `Wenlan` strategy call in run_quality_cost_eval:
        let scope = match case.space.as_deref() {
            None => ReadScope::Global,
            Some("uncategorized") => ReadScope::Uncategorized,
            Some(space) => ReadScope::Space(space.to_string()),
        };
        let results = db
            .search_memory(
                &case.query,
                top_k,
                None,
                &scope,
                None,
                Some(1.0), // neutralize confirmation boost — fixture bias
                Some(1.0), // neutralize recap penalty — fixture bias
                None,
            )
            .await?;
        queries.push(QueryRanking {
            key: case_key(case),
            query: case.query.clone(),
            ranked_ids: results.iter().map(|r| r.source_id.clone()).collect(),
            scores: results.iter().map(|r| r.score).collect(),
        });
    }

    Ok(RankingGolden {
        generated_env: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        fixture_set_hash,
        top_k,
        queries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("m{i}")).collect()
    }

    #[test]
    fn injected_reverse_top2_is_red() {
        let g = ids(10);
        let mut bad = g.clone();
        bad.swap(0, 1);
        let s = crate::eval::rank_overlap::rbo(&g, &bad, super::RBO_P);
        // Concrete expected value: top-2 swap with p=0.9, n=10 → ~0.8465.
        // This assertion fails if rbo is wrong, independent of the threshold constants.
        assert!(
            (s - 0.846466_f64).abs() < 1e-4,
            "top-2 swap RBO must be ~0.8465 (got {s:.6})"
        );
        assert!(
            s < super::CURRENT_RBO_THRESHOLD,
            "reverse-top-2 must trip current floor (got {s:.4})"
        );
    }
    #[test]
    fn injected_drop_best_is_red() {
        let g = ids(10);
        let mut bad = g[1..].to_vec();
        bad.push("mX".to_string());
        let s = crate::eval::rank_overlap::rbo(&g, &bad, super::RBO_P);
        // Concrete expected value: drop-best with p=0.9, n=10 → ~0.6386.
        // This assertion fails if rbo is wrong, independent of the threshold constants.
        assert!(
            (s - 0.638556_f64).abs() < 1e-4,
            "drop-best RBO must be ~0.6386 (got {s:.6})"
        );
        assert!(
            s < super::CURRENT_RBO_THRESHOLD,
            "drop-best must trip current floor (got {s:.4})"
        );
    }
    #[test]
    fn injected_full_reverse_is_red_even_at_anchor() {
        let g = ids(10);
        let mut bad = g.clone();
        bad.reverse();
        let s = crate::eval::rank_overlap::rbo(&g, &bad, super::RBO_P);
        // Concrete expected value: full reverse with p=0.9, n=10 → ~0.2502.
        // This assertion fails if rbo is wrong, independent of the threshold constants.
        assert!(
            (s - 0.250152_f64).abs() < 1e-4,
            "full-reverse RBO must be ~0.2502 (got {s:.6})"
        );
        assert!(
            s < super::ANCHOR_RBO_THRESHOLD,
            "full reverse must trip even the looser anchor floor (got {s:.4})"
        );
    }
    #[test]
    fn identical_is_green() {
        let g = ids(10);
        let s = crate::eval::rank_overlap::rbo(&g, &g, super::RBO_P);
        // Concrete expected value: identical lists → exactly 1.0.
        // This assertion fails if rbo is wrong, independent of the threshold constants.
        assert!(
            (s - 1.0_f64).abs() < 1e-9,
            "identical RBO must be exactly 1.0 (got {s:.9})"
        );
        assert!(
            s >= super::CURRENT_RBO_THRESHOLD,
            "identical must pass (got {s:.4})"
        );
    }
    #[test]
    fn benign_deep_neartie_swap_is_green() {
        let g = ids(10);
        let mut benign = g.clone();
        benign.swap(7, 8);
        let s = crate::eval::rank_overlap::rbo(&g, &benign, super::RBO_P);
        // Concrete expected value: swap positions 7/8 with p=0.9, n=10 → ~0.9908.
        // This assertion fails if rbo is wrong, independent of the threshold constants.
        assert!(
            (s - 0.990821_f64).abs() < 1e-4,
            "deep near-tie swap RBO must be ~0.9908 (got {s:.6})"
        );
        assert!(
            s >= super::CURRENT_RBO_THRESHOLD,
            "deep near-tie swap must stay green (got {s:.4})"
        );
    }

    #[test]
    fn golden_json_roundtrips() {
        let g = RankingGolden {
            generated_env: "linux-x86_64".to_string(),
            fixture_set_hash: "abc123".to_string(),
            top_k: 10,
            queries: vec![QueryRanking {
                key: "q\u{241f}\u{241f}m1,m2".to_string(),
                query: "q".to_string(),
                ranked_ids: vec!["m1".to_string(), "m2".to_string()],
                scores: vec![0.91, 0.42],
            }],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.json");
        save_golden(&g, &path).unwrap();
        let back = load_golden(&path).unwrap();
        assert_eq!(back.queries.len(), 1);
        assert_eq!(back.queries[0].ranked_ids, vec!["m1", "m2"]);
        assert_eq!(back.top_k, 10);
    }

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("app/eval/fixtures")
    }

    fn goldens_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/eval/goldens")
    }

    /// Regenerate the goldens. Gated by `EVAL_GOLDEN_BLESS=1` so it never runs by
    /// accident. Run ONLY when the labeled eval (`eval::retrieval`) is green AND the
    /// ranking change is intentional (the second-oracle + human-intent refresh gate).
    ///   EVAL_GOLDEN_BLESS=1 cargo test -p wenlan-core --lib \
    ///     eval::retrieval_drift::tests::bless_goldens -- --ignored --nocapture
    #[tokio::test]
    #[ignore] // bless/capture — manual, gated by EVAL_GOLDEN_BLESS=1; needs FastEmbed model
    async fn bless_goldens() {
        if std::env::var("EVAL_GOLDEN_BLESS").as_deref() != Ok("1") {
            eprintln!("SKIP bless_goldens: set EVAL_GOLDEN_BLESS=1 to regenerate goldens");
            return;
        }
        let dir = fixture_dir();
        assert!(dir.exists(), "fixture dir not found: {}", dir.display());
        let golden = super::capture_rankings(&dir, super::TOP_K).await.unwrap();
        assert!(!golden.queries.is_empty(), "captured zero queries");
        let gdir = goldens_dir();
        std::fs::create_dir_all(&gdir).unwrap();
        super::save_golden(&golden, &gdir.join("retrieval_ranking.current.json")).unwrap();
        // Anchor is written only if absent — it drifts slowly, by explicit human
        // decision (delete the file to re-anchor), NOT on every bless.
        let anchor = gdir.join("retrieval_ranking.anchor.json");
        if !anchor.exists() {
            super::save_golden(&golden, &anchor).unwrap();
        }
        eprintln!(
            "blessed {} queries ({}) -> {}",
            golden.queries.len(),
            golden.generated_env,
            gdir.display()
        );
    }

    /// The differential drift gate (L6, push to `main`). Captures live rankings and
    /// asserts top-weighted RBO vs both the current and the old-anchor goldens.
    /// `#[ignore]`d (needs the FastEmbed model); the dedicated main-canary workflow
    /// selects this exact full test name alongside the labeled quality/cost eval.
    #[tokio::test]
    #[ignore] // L6 — needs FastEmbed model; exact-selected on relevant main pushes
    async fn ranking_drift_vs_golden() {
        let dir = fixture_dir();
        if !dir.exists() {
            eprintln!("SKIP ranking_drift_vs_golden: fixture dir not found");
            return;
        }
        let gdir = goldens_dir();
        let current = super::load_golden(&gdir.join("retrieval_ranking.current.json"))
            .expect("current golden missing — run bless_goldens with EVAL_GOLDEN_BLESS=1");
        let anchor = super::load_golden(&gdir.join("retrieval_ranking.anchor.json"))
            .expect("anchor golden missing — run bless_goldens with EVAL_GOLDEN_BLESS=1");
        let live = super::capture_rankings(&dir, super::TOP_K).await.unwrap();

        if live.fixture_set_hash != current.fixture_set_hash {
            eprintln!(
                "NOTE: fixture_set_hash changed (golden {} vs live {}); compared on overlapping \
                 queries — re-bless if the fixture change was intentional.",
                current.fixture_set_hash, live.fixture_set_hash
            );
        }

        // Match by stable per-case key, NOT query string (non-unique across cases).
        use std::collections::HashMap;
        let cur: HashMap<&str, &super::QueryRanking> = current
            .queries
            .iter()
            .map(|q| (q.key.as_str(), q))
            .collect();
        let anc: HashMap<&str, &super::QueryRanking> =
            anchor.queries.iter().map(|q| (q.key.as_str(), q)).collect();

        let mut failures: Vec<String> = Vec::new();
        for q in &live.queries {
            if let Some(g) = cur.get(q.key.as_str()) {
                let s = crate::eval::rank_overlap::rbo(&g.ranked_ids, &q.ranked_ids, super::RBO_P);
                if s < super::CURRENT_RBO_THRESHOLD {
                    failures.push(format!(
                        "[current RBO {s:.4} < {:.4}] {:?}\n  golden: {:?}\n  live:   {:?}\n  golden_scores: {:?}\n  live_scores:   {:?}",
                        super::CURRENT_RBO_THRESHOLD, q.query, g.ranked_ids, q.ranked_ids, g.scores, q.scores
                    ));
                }
            }
            if let Some(g) = anc.get(q.key.as_str()) {
                let s = crate::eval::rank_overlap::rbo(&g.ranked_ids, &q.ranked_ids, super::RBO_P);
                if s < super::ANCHOR_RBO_THRESHOLD {
                    failures.push(format!(
                        "[anchor RBO {s:.4} < {:.4}] {:?}\n  anchor: {:?}\n  live:   {:?}",
                        super::ANCHOR_RBO_THRESHOLD,
                        q.query,
                        g.ranked_ids,
                        q.ranked_ids
                    ));
                }
            }
        }

        assert!(
            failures.is_empty(),
            "RANKING DRIFT DETECTED ({} violation(s)) — a ranking change reached the retrieval \
             order. If intentional + the labeled eval (eval::retrieval) is still green, re-bless \
             (EVAL_GOLDEN_BLESS=1). Otherwise this is a silent regression:\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
}
