// SPDX-License-Identifier: Apache-2.0
//! Per-query emission for paired A/B retrieval evals (validation apparatus v2).
//!
//! The aggregate `run_*_eval_from_db` runners are *deterministic* per
//! `(DB, fixture, flag)` — re-running them gives stddev ≈ 0, which makes a
//! "run N times, take stddev" protocol vacuous. The statistically valid source
//! of variance is **across queries**: a paired per-query test (Wilcoxon signed
//! rank / bootstrap over the per-query Δ) on a single deterministic run.
//!
//! These collector runners mirror the scoring loop of their aggregate siblings
//! but emit one [`PerQueryRow`] per query (with a wall-clock retrieval latency)
//! instead of only the aggregate. The downstream `analyze_paired.py` joins the
//! OFF and ON JSONL arms by `(bench, query_id)` and runs the paired stats.
//!
//! Scaffolding only — not part of the shipped product surface.

use serde::{Deserialize, Serialize};

/// One query's retrieval metrics for a single flag arm.
///
/// Emitted as a JSONL line. The analyzer joins OFF and ON files on
/// `(bench, query_id)` to form per-query Δ pairs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerQueryRow {
    /// Feature under test, e.g. `"fts_hardening"`.
    pub feature: String,
    /// Bench, e.g. `"locomo"` or `"lme"`.
    pub bench: String,
    /// Flag arm: `"off"` or `"on"`.
    pub flag_state: String,
    /// Stable per-query id. LoCoMo: `"<sample_id>#q<idx>"`. LME: native `question_id`.
    pub query_id: String,
    /// Category bucket (LoCoMo numeric category as string, LME `question_type`).
    pub category: String,
    pub ndcg10: f64,
    pub recall5: f64,
    pub mrr: f64,
    /// Wall-clock latency of the retrieval call for this query, milliseconds.
    pub latency_ms: f64,
    /// T3 graph-gate only: whether the graph was skipped for this query.
    /// `None` for features that don't gate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_skipped: Option<bool>,
    /// T4a temporal-filter only: whether a high-confidence temporal cue fired for
    /// this query (i.e. the temporal hard filter actually engaged). `None` for
    /// features whose delta is not temporal-cue-gated. Lets the analyzer report a
    /// per-touched-subset delta instead of a corpus-averaged ~96.6% no-op figure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_touched: Option<bool>,
    /// Generic G3 attribution flag: did the feature's channel actually touch
    /// this query? Feature-specific predicate — see the `channel_touch` probes
    /// in eval/shared.rs (landing in a later commit of this branch). Takes
    /// precedence over `temporal_touched`/`graph_skipped` in analyze_paired.py.
    /// `None` = no honest probe wired for this feature; the analyzer then
    /// stars the verdict (vacuous attribution) on purpose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel_touched: Option<bool>,
    /// Set-based coverage recall over the result bundle WITHOUT page provenance
    /// expansion: pages contribute only their own id (which never matches a
    /// memory-keyed gold id). Equals the share of gold leaf ids retrieved
    /// directly. `None` for emitters that don't measure coverage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cov_blind: Option<f64>,
    /// Set-based coverage recall WITH page provenance expansion: a `source="page"`
    /// (or `"summary"`) result is credited via the memory source ids it was
    /// distilled from. `cov_expanded - cov_blind` is the honest page-channel
    /// contribution — the metric pages move that positional NDCG/recall cannot
    /// see. `None` for emitters that don't measure coverage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cov_expanded: Option<f64>,
}

/// Threat-2 guard for the cross-rerank paired runners: assert the
/// `summary_nodes` table is empty before scoring.
///
/// The global-context prelude (db.rs:9268) PREPENDs `summary_nodes` rows to the
/// search output. They carry no gold leaf id, so a populated table would push
/// gold memories down the result list and silently depress recall/ndcg without
/// any flag flip. The table is empty unless `WENLAN_ENABLE_GLOBAL_PRELUDE` plus
/// a populate step ran, so this is a no-op on the default path — but it fails
/// loud if a future change accidentally seeds it.
///
/// A missing `summary_nodes` table reads as zero (the query errors, which we
/// treat as "absent, fine"); only a positive count trips the assert.
pub async fn assert_summary_nodes_empty(db: &crate::db::MemoryDB) {
    // Missing table (or any read error) -> treat as empty.
    let count = db.eval_paired_summary_node_count().await.unwrap_or(0);
    assert_eq!(
        count, 0,
        "summary_nodes is non-empty ({count} rows): the global-prelude prepend \
         would silently demote gold memories below the prelude rows; refusing to \
         emit a misleading paired baseline"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_channel_touched_serde_roundtrip_and_omission() {
        let mut row = PerQueryRow {
            feature: "x".into(),
            bench: "lme".into(),
            flag_state: "on".into(),
            query_id: "q1".into(),
            category: "TR".into(),
            ndcg10: 0.5,
            recall5: 0.5,
            mrr: 0.5,
            latency_ms: 1.0,
            graph_skipped: None,
            temporal_touched: None,
            channel_touched: None,
            cov_blind: None,
            cov_expanded: None,
        };
        // None must be OMITTED from JSON (old analyzers unaffected)
        let js = serde_json::to_string(&row).unwrap();
        assert!(
            !js.contains("channel_touched"),
            "None must serialize to absent: {js}"
        );
        // Some(true) round-trips
        row.channel_touched = Some(true);
        let js = serde_json::to_string(&row).unwrap();
        let back: PerQueryRow = serde_json::from_str(&js).unwrap();
        assert_eq!(back.channel_touched, Some(true));
        // old JSONL without the field still deserializes (default None)
        let old = js.replace(",\"channel_touched\":true", "");
        let back: PerQueryRow = serde_json::from_str(&old).unwrap();
        assert_eq!(back.channel_touched, None);
    }
}
