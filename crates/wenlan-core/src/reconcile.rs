// SPDX-License-Identifier: Apache-2.0
//! Doc-grounded revisions (L3 reconcile).
//!
//! A 30-min scheduler sweep detects direct factual contradictions between
//! ingested documents (`source_agent='folder'`) and agent captures, and stages
//! human-gated rewrite+cite revisions through the existing pending-revisions
//! surface. Doc rows are NEVER written; the capture is untouched until human
//! accept. Design: docs/superpowers/specs/2026-07-02-doc-grounded-revisions-design.md.

use std::sync::Arc;

use crate::db::MemoryDB;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::prompts::PromptRegistry;
use crate::tuning::{DistillationConfig, RefineryConfig};
use wenlan_types::RawDocument;

/// Minimum cosine SIMILARITY for a frontier item / candidate pair to reach the
/// LLM judge. Known recall ceiling (contradictions need not be embedding-near);
/// measured post-ship per spec §7.
pub const RECONCILE_COSINE_GATE: f64 = 0.70;
/// Hard cap on LLM judge calls per tick across both frontiers (GPU contention).
pub const RECONCILE_JUDGE_CALLS_PER_TICK: usize = 25;
/// Back-pressure: sweep holds entirely while this many doc-grounded revisions
/// await human review.
pub const RECONCILE_PENDING_CAP: usize = 20;
/// Max frontier rows fetched per frontier per tick.
pub const RECONCILE_BATCH_PER_FRONTIER: usize = 50;
/// Vector top-k candidates per frontier item.
pub(crate) const RECONCILE_TOP_K: usize = 5;
/// Consecutive failed ticks on the same head item before poison-pill ejection.
pub(crate) const RECONCILE_POISON_TICKS: u32 = 3;

/// Inputs for staging one doc-grounded revision row.
#[derive(Debug, Clone)]
pub struct RevisionInput<'a> {
    pub capture_source_id: &'a str,
    pub capture_space: Option<&'a str>,
    /// File-level doc source_id ("{source_id}::{path}", shared by all chunks).
    pub doc_file_source_id: &'a str,
    pub doc_chunk_index: i64,
    pub doc_hash: &'a str,
    pub revised_content: &'a str,
}

/// Stage ONE revision row via the canonical store + enrichment path (ingest
/// parity: embedding at store time, Phase-1 classify/tags when an LLM is
/// available). pending_revision=1 keeps it out of Phase-3 pools; the human
/// gate owns activation. Returns the new revision source_id.
pub async fn write_revision(
    db: &MemoryDB,
    input: RevisionInput<'_>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    refinery: &RefineryConfig,
    distillation: &DistillationConfig,
) -> Result<String, crate::WenlanError> {
    let source_id = format!(
        "mem_{}",
        uuid::Uuid::new_v4()
            .to_string()
            .replace('-', "")
            .chars()
            .take(12)
            .collect::<String>()
    );
    let structured = serde_json::json!({
        "revises": input.capture_source_id,
        "grounded_in": input.doc_file_source_id,
        "grounded_chunk": input.doc_chunk_index,
        "doc_hash": input.doc_hash,
    })
    .to_string();

    let row = RawDocument {
        source: "memory".to_string(),
        source_id: source_id.clone(),
        title: input.revised_content.chars().take(80).collect(),
        content: input.revised_content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        space: input.capture_space.map(str::to_string),
        source_agent: Some("reconcile".to_string()),
        confirmed: None,
        supersedes: Some(input.capture_source_id.to_string()),
        pending_revision: true,
        structured_fields: Some(structured.clone()),
        ..Default::default()
    };
    db.upsert_documents(vec![row]).await?;
    db.upsert_enrichment_origin(
        &source_id,
        crate::db::EnrichmentOrigin {
            // These are reconcile placeholders/provenance, not user-pinned
            // semantic choices. Durable fixed lanes may refine them after the
            // human accepts this generation.
            memory_type_explicit: false,
            structured_fields_explicit: false,
            space_rejected: false,
        },
    )
    .await?;

    let opts = crate::ingest::EnrichmentOpts {
        initial_memory_type: "identity".to_string(),
        initial_domain: input.capture_space.map(str::to_string),
        rejected_explicit_domain: false,
        initial_supersede_mode: "hide".to_string(),
        initial_structured_fields: Some(structured),
        agent_supplied_memory_type: false,
        agent_supplied_profile_alias: false,
        // Protect the provenance JSON from Phase-1 overwrite.
        agent_supplied_structured_fields: true,
    };
    crate::ingest::run_canonical_enrichment(
        db,
        &source_id,
        input.revised_content,
        None,
        llm,
        prompts,
        refinery,
        distillation,
        None,
        &opts,
        None,
    )
    .await;
    Ok(source_id)
}

/// A frontier row awaiting reconciliation (doc chunk or capture).
#[derive(Debug, Clone, PartialEq)]
pub struct ReconcileItem {
    pub source_id: String,
    pub chunk_index: i64,
    pub content: String,
    pub space: Option<String>,
    pub last_modified: i64,
    /// Per-file SHA-256 (docs only; None for captures).
    pub content_hash: Option<String>,
}

/// A vector-matched candidate on the opposite side of a frontier item.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconcileCandidate {
    pub source_id: String,
    pub chunk_index: i64,
    pub content: String,
    pub last_modified: i64,
    pub created_at: i64,
    pub content_hash: Option<String>,
    pub source_agent: Option<String>,
    pub cosine: f64,
}

/// One judge-confirmed conflict: candidate index + the corrected capture text.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConflictProposal {
    pub idx: usize,
    pub revised_content: String,
}

/// Defensively parse the judge's response. Mirrors `parse_dual_pool`'s
/// silent-zero guard: ANY parse failure returns empty (the sweep must never
/// act on garbage). Out-of-range indices and blank rewrites are dropped.
pub(crate) fn parse_doc_reconcile(raw: &str, total_len: usize) -> Vec<ConflictProposal> {
    let stripped = crate::llm_provider::strip_think_tags(raw);
    let (start, end) = match (stripped.find('{'), stripped.rfind('}')) {
        (Some(s), Some(e)) if e >= s => (s, e),
        _ => return Vec::new(),
    };
    #[derive(serde::Deserialize)]
    struct RawConflict {
        idx: i64,
        #[serde(default)]
        revised_content: String,
    }
    #[derive(serde::Deserialize)]
    struct Raw {
        #[serde(default)]
        conflicts: Vec<RawConflict>,
    }
    let parsed: Raw = match serde_json::from_str(&stripped[start..=end]) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    parsed
        .conflicts
        .into_iter()
        .filter_map(|c| {
            let idx = usize::try_from(c.idx).ok()?;
            if idx >= total_len || c.revised_content.trim().is_empty() {
                return None;
            }
            Some(ConflictProposal {
                idx,
                revised_content: c.revised_content,
            })
        })
        .collect()
}

fn date_label(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|d| d.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| ts.to_string())
}

/// Render the judge user-prompt: focus text + numbered candidates, both sides
/// dated so a stale doc can be weighed against a newer confirmed capture.
pub(crate) fn build_reconcile_prompt(
    focus_is_doc: bool,
    focus_content: &str,
    focus_date: i64,
    candidates: &[ReconcileCandidate],
) -> String {
    let (focus_label, cand_label) = if focus_is_doc {
        ("DOCUMENT", "CAPTURE")
    } else {
        ("CAPTURE", "DOCUMENT")
    };
    let mut p = format!(
        "FOCUS ({focus_label}, dated {}):\n{}\n\nCANDIDATES ({cand_label} side):\n",
        date_label(focus_date),
        focus_content
    );
    for (i, c) in candidates.iter().enumerate() {
        p.push_str(&format!(
            "[{i}] (dated {}) {}\n",
            date_label(c.last_modified),
            c.content
        ));
    }
    // Direction-explicit rewrite target: small judges fumble the "CAPTURE
    // side" indirection and echo the capture text back (seen live).
    p.push_str(if focus_is_doc {
        "\nRewrite target: the flagged candidate's text (the CAPTURE), updated with the FOCUS document's facts. revised_content must differ from the candidate's current text.\n"
    } else {
        "\nRewrite target: the FOCUS text (the CAPTURE), updated with the flagged candidate document's facts. revised_content must differ from the FOCUS's current text.\n"
    });
    p
}

/// Persisted per-frontier cursor + poison-pill state. Stored as JSON in
/// app_metadata (keys below); a missing/corrupt value degrades to default =
/// full-corpus sweep from zero (bounded by batch + judge caps).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FrontierState {
    pub ts: i64,
    pub id: String,
    pub chunk: i64,
    pub stuck_id: Option<String>,
    pub failures: u32,
}

pub(crate) const FRONTIER_DOCS_KEY: &str = "reconcile_frontier_docs";
pub(crate) const FRONTIER_CAPTURES_KEY: &str = "reconcile_frontier_captures";
const RECONCILE_SLICE_NEXT_KEY: &str = "reconcile_slice_next_frontier";

/// Record an LLM failure on the frontier head item. Returns true when the item
/// has failed RECONCILE_POISON_TICKS consecutive ticks and must be ejected
/// (caller advances the watermark past it with a warn! log).
pub(crate) fn note_failure(st: &mut FrontierState, item_key: &str) -> bool {
    if st.stuck_id.as_deref() == Some(item_key) {
        st.failures += 1;
        if st.failures >= RECONCILE_POISON_TICKS {
            st.stuck_id = None;
            st.failures = 0;
            return true;
        }
    } else {
        st.stuck_id = Some(item_key.to_string());
        st.failures = 1;
    }
    false
}

fn advance(st: &mut FrontierState, item: &ReconcileItem) {
    st.ts = item.last_modified;
    st.id = item.source_id.clone();
    st.chunk = item.chunk_index;
}

async fn load_state(db: &MemoryDB, key: &str) -> FrontierState {
    match db.get_app_metadata(key).await {
        Ok(Some(v)) => serde_json::from_str(&v).unwrap_or_default(),
        _ => FrontierState::default(),
    }
}

async fn save_state(db: &MemoryDB, key: &str, st: &FrontierState) {
    let json = serde_json::to_string(st).unwrap_or_default();
    if let Err(e) = db.set_app_metadata(key, &json).await {
        log::warn!("[reconcile] failed to persist {key}: {e}");
    }
}

/// Outcome of one sweep tick, for scheduler logging.
#[derive(Debug, Default, PartialEq)]
pub struct ReconcileReport {
    /// At least one frontier watermark advanced, including a zero-candidate
    /// item that required no judge call.
    pub progressed: bool,
    pub judged: usize,
    pub proposed: usize,
    pub skipped_backpressure: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frontier {
    Docs,
    Captures,
}

impl Frontier {
    const fn other(self) -> Self {
        match self {
            Self::Docs => Self::Captures,
            Self::Captures => Self::Docs,
        }
    }

    const fn as_metadata(self) -> &'static str {
        match self {
            Self::Docs => "docs",
            Self::Captures => "captures",
        }
    }
}

/// One sweep tick: both frontiers, shared judge budget, watermark semantics per
/// spec §3.2 (advance only past judged / zero-candidate items; retry on LLM
/// failure; 3-tick poison-pill ejection; back-pressure hold at the pending cap).
pub async fn run_reconcile_tick(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    refinery: &RefineryConfig,
    distillation: &DistillationConfig,
) -> Result<ReconcileReport, crate::WenlanError> {
    run_reconcile_with_budget(
        db,
        llm,
        prompts,
        refinery,
        distillation,
        RECONCILE_JUDGE_CALLS_PER_TICK,
        [Frontier::Docs, Frontier::Captures],
        Some(llm),
    )
    .await
}

/// Advance the reconcile frontiers with at most one LLM judge request.
pub async fn run_reconcile_slice(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    refinery: &RefineryConfig,
    distillation: &DistillationConfig,
) -> Result<ReconcileReport, crate::WenlanError> {
    let first = match db.get_app_metadata(RECONCILE_SLICE_NEXT_KEY).await? {
        Some(value) if value == Frontier::Captures.as_metadata() => Frontier::Captures,
        _ => Frontier::Docs,
    };
    let report = run_reconcile_with_budget(
        db,
        llm,
        prompts,
        refinery,
        distillation,
        1,
        [first, first.other()],
        // The judge already consumes this slice's one inference. Passing its
        // provider into write_revision would hide classify/extract/title/page
        // calls behind the budget. The pending row still receives its content,
        // provenance, embedding, and deterministic no-provider post-ingest work.
        None,
    )
    .await?;
    db.set_app_metadata(RECONCILE_SLICE_NEXT_KEY, first.other().as_metadata())
        .await?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
async fn run_reconcile_with_budget(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    refinery: &RefineryConfig,
    distillation: &DistillationConfig,
    mut budget: usize,
    frontier_order: [Frontier; 2],
    revision_llm: Option<&Arc<dyn LlmProvider>>,
) -> Result<ReconcileReport, crate::WenlanError> {
    let mut report = ReconcileReport::default();
    // Back-pressure: proposals drip at human curation pace. Watermarks hold.
    if db.pending_reconcile_at_cap(RECONCILE_PENDING_CAP).await? {
        report.skipped_backpressure = true;
        return Ok(report);
    }
    for frontier in frontier_order {
        run_frontier(
            db,
            llm,
            prompts,
            refinery,
            distillation,
            frontier,
            revision_llm,
            &mut budget,
            &mut report,
        )
        .await?;
    }
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
async fn run_frontier(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    refinery: &RefineryConfig,
    distillation: &DistillationConfig,
    which: Frontier,
    revision_llm: Option<&Arc<dyn LlmProvider>>,
    budget: &mut usize,
    report: &mut ReconcileReport,
) -> Result<(), crate::WenlanError> {
    if *budget == 0 {
        return Ok(());
    }
    let key = match which {
        Frontier::Docs => FRONTIER_DOCS_KEY,
        Frontier::Captures => FRONTIER_CAPTURES_KEY,
    };
    let mut st = load_state(db, key).await;
    let items = match which {
        Frontier::Docs => {
            db.reconcile_doc_frontier(st.ts, &st.id, st.chunk, RECONCILE_BATCH_PER_FRONTIER)
                .await?
        }
        Frontier::Captures => {
            db.reconcile_capture_frontier(st.ts, &st.id, st.chunk, RECONCILE_BATCH_PER_FRONTIER)
                .await?
        }
    };

    for item in items {
        if *budget == 0 {
            break; // unjudged items stay ahead of the watermark for next tick
        }
        let item_key = format!("{}#{}", item.source_id, item.chunk_index);
        let focus_is_doc = matches!(which, Frontier::Docs);
        // Candidates live on the OPPOSITE side of the frontier item.
        let candidates = db
            .reconcile_candidates(
                &item.source_id,
                item.chunk_index,
                item.space.as_deref(),
                !focus_is_doc, // captures frontier looks toward docs
                RECONCILE_TOP_K,
                RECONCILE_COSINE_GATE,
            )
            .await?;
        if candidates.is_empty() {
            advance(&mut st, &item);
            report.progressed = true;
            continue;
        }

        // Budget counts LLM attempts (a failing provider gets no free
        // retries within a tick); judged counts only successful calls.
        *budget -= 1;
        let user_prompt =
            build_reconcile_prompt(focus_is_doc, &item.content, item.last_modified, &candidates);
        let raw = match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            llm.generate(LlmRequest {
                system_prompt: Some(prompts.doc_reconcile.clone()),
                user_prompt,
                max_tokens: 1024,
                temperature: 0.1,
                label: Some("doc_reconcile".to_string()),
                timeout_secs: None,
            }),
        )
        .await
        {
            Ok(Ok(r)) => r,
            _ => {
                if note_failure(&mut st, &item_key) {
                    log::warn!(
                        "[reconcile] ejecting poison item {item_key} after {RECONCILE_POISON_TICKS} failed ticks"
                    );
                    advance(&mut st, &item);
                    report.progressed = true;
                    continue;
                }
                break; // hold watermark; retry this head item next tick
            }
        };
        report.judged += 1;
        // Successful judge call on the previously stuck item: clear the strike.
        if st.stuck_id.as_deref() == Some(item_key.as_str()) {
            st.stuck_id = None;
            st.failures = 0;
        }

        for p in parse_doc_reconcile(&raw, candidates.len()) {
            let cand = &candidates[p.idx];
            // Map frontier roles onto (capture, doc) for the revision row.
            let (capture_id, doc_file, doc_chunk, doc_hash, doc_agent, capture_agent) =
                if focus_is_doc {
                    (
                        cand.source_id.as_str(),
                        item.source_id.as_str(),
                        item.chunk_index,
                        item.content_hash.as_deref(),
                        Some("folder"),
                        cand.source_agent.as_deref(),
                    )
                } else {
                    (
                        item.source_id.as_str(),
                        cand.source_id.as_str(),
                        cand.chunk_index,
                        cand.content_hash.as_deref(),
                        cand.source_agent.as_deref(),
                        None,
                    )
                };
            // Direction assert: only a strictly-higher-precedence source may
            // ground a revision. Defense-in-depth only — the SQL side
            // predicates already pin the doc side to 'folder' and exclude it
            // from the capture side, so this cannot fire today. The capture
            // side is conservatively ranked as a plain capture (frontier rows
            // don't carry source_agent), which those predicates make accurate.
            use crate::retrieval::resolve::source_precedence;
            if source_precedence(doc_agent) <= source_precedence(capture_agent) {
                continue;
            }
            // A "revision" that leaves the capture's text unchanged is a
            // no-op — small judges sometimes echo the capture instead of
            // rewriting it toward the document. Never queue it for review.
            let capture_content = if focus_is_doc {
                cand.content.as_str()
            } else {
                item.content.as_str()
            };
            if p.revised_content.trim() == capture_content.trim() {
                log::info!(
                    "[reconcile] judge echoed capture text for {capture_id}; skipping no-op proposal"
                );
                continue;
            }
            let Some(doc_hash) = doc_hash else {
                log::warn!("[reconcile] doc {doc_file} has no content_hash; skipping proposal");
                continue;
            };
            if db
                .reconcile_pair_exists(capture_id, doc_file, doc_hash)
                .await?
                || db.capture_has_pending_revision(capture_id).await?
            {
                continue;
            }
            let input = RevisionInput {
                capture_source_id: capture_id,
                capture_space: item.space.as_deref(),
                doc_file_source_id: doc_file,
                doc_chunk_index: doc_chunk,
                doc_hash,
                revised_content: &p.revised_content,
            };
            match write_revision(db, input, revision_llm, prompts, refinery, distillation).await {
                Ok(_) => report.proposed += 1,
                Err(e) => log::warn!("[reconcile] write_revision for {capture_id} failed: {e}"),
            }
        }
        advance(&mut st, &item);
        report.progressed = true;
    }
    save_state(db, key, &st).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, content: &str) -> ReconcileCandidate {
        ReconcileCandidate {
            source_id: id.to_string(),
            chunk_index: 0,
            content: content.to_string(),
            last_modified: 1_700_000_000,
            created_at: 1_700_000_000,
            content_hash: None,
            source_agent: None,
            cosine: 0.9,
        }
    }

    #[test]
    fn parse_well_formed_conflicts() {
        let raw = r#"{"conflicts":[{"idx":0,"revised_content":"Port is 7878."},{"idx":2,"revised_content":"Uses libSQL."}]}"#;
        let out = parse_doc_reconcile(raw, 3);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].idx, 0);
        assert_eq!(out[0].revised_content, "Port is 7878.");
        assert_eq!(out[1].idx, 2);
    }

    #[test]
    fn parse_tolerates_think_tags_and_fences() {
        let raw = "<think>hmm</think>\n```json\n{\"conflicts\":[{\"idx\":1,\"revised_content\":\"x\"}]}\n```";
        assert_eq!(parse_doc_reconcile(raw, 2).len(), 1);
    }

    #[test]
    fn parse_garbage_and_no_conflicts_return_empty() {
        assert!(parse_doc_reconcile("not json", 3).is_empty());
        assert!(parse_doc_reconcile(r#"{"conflicts":[]}"#, 3).is_empty());
        assert!(parse_doc_reconcile(r#"{"other":1}"#, 3).is_empty());
    }

    #[test]
    fn parse_drops_out_of_range_negative_and_empty_content() {
        let raw = r#"{"conflicts":[{"idx":9,"revised_content":"x"},{"idx":-1,"revised_content":"x"},{"idx":0,"revised_content":"  "},{"idx":1,"revised_content":"keep"}]}"#;
        let out = parse_doc_reconcile(raw, 3);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].idx, 1);
    }

    #[test]
    fn prompt_numbers_candidates_and_carries_dates_and_roles() {
        let cands = vec![cand("a", "claim A"), cand("b", "claim B")];
        let p = build_reconcile_prompt(true, "doc text", 1_700_000_000, &cands);
        assert!(p.contains("FOCUS (DOCUMENT"));
        assert!(p.contains("CANDIDATES (CAPTURE side)"));
        assert!(p.contains("[0]") && p.contains("[1]"));
        assert!(
            p.contains("2023-11-14"),
            "epoch 1700000000 renders as its UTC date"
        );
        assert!(
            p.contains("Rewrite target: the flagged candidate's text (the CAPTURE)"),
            "doc-focus prompt names the candidate as rewrite target"
        );
        let p2 = build_reconcile_prompt(false, "capture text", 1_700_000_000, &cands);
        assert!(p2.contains("FOCUS (CAPTURE"));
        assert!(p2.contains("CANDIDATES (DOCUMENT side)"));
        assert!(
            p2.contains("Rewrite target: the FOCUS text (the CAPTURE)"),
            "capture-focus prompt names the focus as rewrite target"
        );
    }

    #[test]
    fn registry_carries_doc_reconcile_default() {
        let reg = crate::prompts::PromptRegistry::default();
        assert!(reg.doc_reconcile.contains("mutually exclusive"));
        assert!(reg.doc_reconcile.contains("conflicts"));
    }

    #[test]
    fn note_failure_tracks_and_ejects_after_three_ticks() {
        let mut st = FrontierState::default();
        assert!(!note_failure(&mut st, "a#0"), "tick 1: hold");
        assert_eq!(st.failures, 1);
        assert!(!note_failure(&mut st, "a#0"), "tick 2: hold");
        assert!(note_failure(&mut st, "a#0"), "tick 3: eject");
        assert_eq!(st.failures, 0);
        assert!(st.stuck_id.is_none(), "state reset after ejection");
    }

    #[test]
    fn note_failure_resets_when_head_item_changes() {
        let mut st = FrontierState::default();
        note_failure(&mut st, "a#0");
        note_failure(&mut st, "a#0");
        assert!(
            !note_failure(&mut st, "b#0"),
            "different head: counter restarts"
        );
        assert_eq!(st.failures, 1);
        assert_eq!(st.stuck_id.as_deref(), Some("b#0"));
    }

    #[test]
    fn frontier_state_round_trips_json() {
        let st = FrontierState {
            ts: 42,
            id: "mem_x".into(),
            chunk: 3,
            stuck_id: Some("mem_y#0".into()),
            failures: 2,
        };
        let json = serde_json::to_string(&st).unwrap();
        assert_eq!(serde_json::from_str::<FrontierState>(&json).unwrap(), st);
        // Empty/corrupt value degrades to a full-corpus-from-zero default.
        assert_eq!(serde_json::from_str::<FrontierState>("garbage").ok(), None);
    }

    #[tokio::test]
    async fn ambient_reconcile_slice_uses_one_judge_call() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let common = "The Wenlan daemon binds to port 7878 and stores memory locally.";
        let mut docs = Vec::new();
        for (source_id, source_agent, confirmed, content_hash, ts) in [
            ("doc_a", "folder", true, Some("hash_a"), 1_i64),
            ("doc_b", "folder", true, Some("hash_b"), 2_i64),
            ("capture_a", "claude-code", true, None, 3_i64),
        ] {
            docs.push(RawDocument {
                source: "memory".to_string(),
                source_id: source_id.to_string(),
                title: source_id.to_string(),
                content: common.to_string(),
                last_modified: ts,
                confirmed: Some(confirmed),
                source_agent: Some(source_agent.to_string()),
                content_hash: content_hash.map(str::to_string),
                ..Default::default()
            });
        }
        db.upsert_documents(docs).await.unwrap();

        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            r#"{"conflicts":[{"idx":0,"revised_content":"The Wenlan daemon now binds to port 7979 and stores memory locally."}]}"#,
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let report = run_reconcile_slice(
            &db,
            &llm,
            &PromptRegistry::default(),
            &RefineryConfig::default(),
            &DistillationConfig::default(),
        )
        .await
        .unwrap();

        assert_eq!(report.judged, 1);
        assert_eq!(
            report.proposed, 1,
            "the judge response should stage a revision"
        );
        assert_eq!(
            provider.call_count(),
            1,
            "staging the revision must not hide extra ambient requests"
        );
        let pending = db.list_pending_revisions(10).await.unwrap();
        let revision_id = &pending
            .iter()
            .find(|revision| revision.target_source_id == "capture_a")
            .expect("staged reconcile revision")
            .revision_source_id;
        let origin = db.resolve_enrichment_origin(revision_id).await.unwrap();
        assert!(!origin.memory_type_explicit);
        assert!(!origin.structured_fields_explicit);
        assert!(!origin.space_rejected);
    }

    #[tokio::test]
    async fn accepted_revision_converges_all_fixed_enrichment_dependencies() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (db, _dir) = crate::db::tests::test_db().await;
        db.upsert_documents(vec![RawDocument {
            source: "memory".to_string(),
            source_id: "revision_dependency_target".to_string(),
            title: "Old scheduler behavior".to_string(),
            content: "The old scheduler runs continuously.".to_string(),
            last_modified: 1,
            memory_type: Some("fact".to_string()),
            source_agent: Some("claude-code".to_string()),
            ..Default::default()
        }])
        .await
        .unwrap();
        db.upsert_documents(vec![RawDocument {
            source: "memory".to_string(),
            source_id: "older_unrelated_enrichment_backlog".to_string(),
            title: "Older unrelated backlog".to_string(),
            content: "An older unrelated memory is still waiting for enrichment.".to_string(),
            last_modified: 0,
            source_agent: Some("claude-code".to_string()),
            ..Default::default()
        }])
        .await
        .unwrap();

        let revised_content = "The ambient scheduler now advances one bounded unit at a time, preserves durable dependency receipts, and yields between background turns so enrichment remains invisible to the user.";
        let revision_id = write_revision(
            &db,
            RevisionInput {
                capture_source_id: "revision_dependency_target",
                capture_space: None,
                doc_file_source_id: "docs/scheduler.md",
                doc_chunk_index: 3,
                doc_hash: "sha256:revision",
                revised_content,
            },
            None,
            &PromptRegistry::default(),
            &RefineryConfig::default(),
            &DistillationConfig::default(),
        )
        .await
        .unwrap();
        db.accept_pending_revision(&revision_id).await.unwrap();
        let revision_version = db
            .get_memory_detail(&revision_id)
            .await
            .unwrap()
            .unwrap()
            .version;

        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            r#"{"memory_type":"fact","quality":"high","importance":5,"tags":["scheduler"]}"#,
            r#"{"topic":"ambient scheduler","retrieval_cue":"how background enrichment stays cool"}"#,
            "Invisible Ambient Scheduler",
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let classification = crate::ingest::run_classification_enrichment_slice(
            &db,
            &llm,
            &PromptRegistry::default(),
        )
        .await
        .unwrap();
        assert!(classification.selected && classification.committed);
        assert!(classification.llm_calls <= 1);

        let structured =
            crate::ingest::run_structured_extract_slice(&db, &llm, &PromptRegistry::default())
                .await
                .unwrap();
        assert!(structured.selected && structured.committed);
        assert!(structured.llm_calls <= 1);

        let entity_calls = Arc::new(AtomicUsize::new(0));
        let calls = entity_calls.clone();
        let selected = db
            .run_entity_enrichment_slice(move |_content| async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok::<Vec<crate::extract::KgExtractionResult>, crate::WenlanError>(Vec::new())
            })
            .await
            .unwrap();
        assert_eq!(selected, 1);
        assert_eq!(entity_calls.load(Ordering::SeqCst), 1);

        let title = crate::post_ingest::run_title_enrichment_slice(&db, &llm)
            .await
            .unwrap();
        assert!(title.selected && title.committed);
        assert!(title.llm_calls <= 1);

        let growth = crate::post_ingest::run_page_growth_slice(
            &db,
            &llm,
            &PromptRegistry::default(),
            2.0,
            None,
        )
        .await
        .unwrap();
        assert!(growth.selected && growth.committed);
        assert!(!growth.matched);
        assert_eq!(growth.llm_calls, 0);
        assert_eq!(provider.call_count(), 3);

        let steps = db.get_enrichment_steps(&revision_id).await.unwrap();
        for step_name in [
            "classify",
            "structured_extract",
            "entity_extract",
            "title_enrich",
            "page_growth",
        ] {
            let receipt = steps
                .iter()
                .find(|receipt| receipt.step == step_name)
                .unwrap_or_else(|| panic!("missing {step_name} receipt"));
            assert_eq!(receipt.status, "ok", "{step_name} must converge");
            assert_eq!(
                receipt.input_version,
                Some(revision_version),
                "{step_name} must depend on the accepted generation"
            );
        }

        let detail = db.get_memory_detail(&revision_id).await.unwrap().unwrap();
        let fields: serde_json::Value =
            serde_json::from_str(detail.structured_fields.as_deref().unwrap()).unwrap();
        assert_eq!(fields["revises"], "revision_dependency_target");
        assert_eq!(fields["grounded_in"], "docs/scheduler.md");
        assert_eq!(fields["doc_hash"], "sha256:revision");
        assert_eq!(fields["topic"], "ambient scheduler");
        assert_eq!(detail.memory_type.as_deref(), Some("fact"));
        assert_eq!(detail.quality.as_deref(), Some("high"));
        assert_eq!(detail.title, "Invisible Ambient Scheduler");
        assert_eq!(
            detail.retrieval_cue.as_deref(),
            Some("how background enrichment stays cool")
        );
        assert_eq!(
            db.get_document_tags("memory", &revision_id).await.unwrap(),
            vec!["scheduler".to_string()]
        );
        assert_eq!(
            detail.supersedes.as_deref(),
            Some("revision_dependency_target")
        );
        assert!(!detail.pending_revision);
    }

    #[tokio::test]
    async fn ambient_reconcile_slice_alternates_frontier_priority() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let common = "The Wenlan daemon binds to port 7878 and stores memory locally.";
        let mut docs = Vec::new();
        for (source_id, source_agent, content_hash, ts) in [
            ("doc_fair_a", "folder", Some("hash_fair_a"), 1_i64),
            ("doc_fair_b", "folder", Some("hash_fair_b"), 2_i64),
            ("capture_fair", "claude-code", None, 3_i64),
        ] {
            docs.push(RawDocument {
                source: "memory".to_string(),
                source_id: source_id.to_string(),
                title: source_id.to_string(),
                content: common.to_string(),
                last_modified: ts,
                confirmed: Some(true),
                source_agent: Some(source_agent.to_string()),
                content_hash: content_hash.map(str::to_string),
                ..Default::default()
            });
        }
        db.upsert_documents(docs).await.unwrap();
        let provider = Arc::new(crate::llm_provider::SequencedMockProvider::new(vec![
            r#"{"conflicts":[]}"#,
        ]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let first = run_reconcile_slice(
            &db,
            &llm,
            &PromptRegistry::default(),
            &RefineryConfig::default(),
            &DistillationConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(first.judged, 1);
        assert_eq!(provider.call_count(), 1, "one slice gets one judge call");
        assert_eq!(
            db.get_app_metadata(RECONCILE_SLICE_NEXT_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some(Frontier::Captures.as_metadata()),
            "the next slice must prefer the opposite frontier"
        );

        let calls_before_second = provider.call_count();
        let second = run_reconcile_slice(
            &db,
            &llm,
            &PromptRegistry::default(),
            &RefineryConfig::default(),
            &DistillationConfig::default(),
        )
        .await
        .unwrap();
        assert!(second.judged <= 1);
        assert_eq!(
            provider.call_count() - calls_before_second,
            second.judged,
            "the second slice may also spend at most one judge call"
        );
        assert_eq!(
            db.get_app_metadata(RECONCILE_SLICE_NEXT_KEY)
                .await
                .unwrap()
                .as_deref(),
            Some(Frontier::Docs.as_metadata()),
            "frontier priority must alternate back after the second slice"
        );
    }
}
