// SPDX-License-Identifier: Apache-2.0
//! Per-claim citation numbering, marker parsing, and union-calibrated
//! verification (pure functions), plus the annotate-only backfill sweep that
//! drains legacy (`citations IS NULL`) pages. See
//! `docs/superpowers/specs/2026-07-03-per-claim-citations-design.md`.

use std::sync::Arc;

use wenlan_types::pages::PageCitation;

use crate::db::MemoryDB;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::prompts::PromptRegistry;
use crate::WenlanError;

/// Cap on source text length embedded in the numbered block, matching
/// `MEM_SNIPPET_CAP` in `synthesis/distill.rs`.
const SOURCE_TEXT_CAP: usize = 800;

/// One numbered source available for citation at distill time.
pub struct NumberedSource {
    pub index: u32,
    pub source_kind: String,
    pub locator: String,
    pub text: String,
}

/// Resolve the typed `page_evidence.source_kind` for a source row shape.
///
/// Maps a citing source row (`source` / `source_agent` / `source_id` columns)
/// to the `page_evidence.source_kind` CHECK domain
/// (`'memory' | 'external_url' | 'external_file' | 'authored'`, db.rs:5873).
/// Replaces the ten hardcoded `'memory'` literals at the page-evidence
/// emitters; PageWrite (the atomic-citations task) is the one wiring site.
///
/// Precedence (first match wins):
/// 1. `authored` — `source` or `source_agent` == `"authored"` (human-owned
///    content promoted into evidence).
/// 2. `external_url` — a URL-shaped `source_id` (webpage captures set
///    `source_id = url`, ingest_routes.rs:118).
/// 3. `external_file` — a folder document: `source_agent == "folder"` with the
///    `{source_id}::{provenance}` id shape stamped by
///    `sources::directory::document_source_id` (directory.rs:372).
/// 4. `memory` — everything else (plain agent captures).
///
/// Intentionally pure so PageWrite can call it while holding no DB lock and
/// doing no I/O.
pub fn resolve_page_evidence_source_kind(
    source: &str,
    source_agent: Option<&str>,
    source_id: &str,
) -> &'static str {
    if source.eq_ignore_ascii_case("authored")
        || source_agent.is_some_and(|agent| agent.eq_ignore_ascii_case("authored"))
    {
        return "authored";
    }

    if source_id.starts_with("http://") || source_id.starts_with("https://") {
        return "external_url";
    }

    if source_agent.is_some_and(|agent| agent.eq_ignore_ascii_case("folder"))
        && source_id.contains("::")
    {
        return "external_file";
    }

    "memory"
}

/// Render the numbered source block fed to the LLM prompt: `"[1] text\n\n[2] text"`.
/// Source text is capped at `SOURCE_TEXT_CAP` chars (char-safe).
pub fn build_numbered_block(sources: &[NumberedSource]) -> String {
    sources
        .iter()
        .map(|s| {
            let capped: String = s.text.chars().take(SOURCE_TEXT_CAP).collect();
            format!("[{}] {}", s.index, capped)
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Remove every `[N]` marker from body prose, collapsing the resulting
/// doubled whitespace.
pub fn strip_markers(body: &str) -> String {
    let marker_re = regex::Regex::new(r"\[\d+\]").expect("static regex");
    let stripped = marker_re.replace_all(body, "");
    let space_re = regex::Regex::new(r" {2,}").expect("static regex");
    space_re.replace_all(&stripped, " ").trim().to_string()
}

/// Per-body citation counts.
pub struct CitationStats {
    pub verified: usize,
    pub unverified: usize,
    pub stripped: usize,
}

impl CitationStats {
    pub fn summary(&self) -> String {
        format!(
            "{} verified, {} unverified, {} stripped",
            self.verified, self.unverified, self.stripped
        )
    }
}

/// Bidirectional lexical support between a claim span and source text: the
/// max of (a) fraction of the SPAN's content tokens found in the source —
/// right direction for terse claims — and (b) fraction of the SOURCE's
/// content tokens found in the span — right direction for verbose synthesis,
/// where elaboration vocabulary dilutes direction (a) below the floor on
/// clearly-supported paragraphs (live smoke 2026-07-03: true claims at
/// 0.11-0.44 one run, 0.65-0.71 the next, purely on output verbosity).
/// A claim citing an unrelated source fails BOTH directions. Uses the shared
/// `faithfulness::overlap_fraction` scorer in both directions — the bench
/// itself is untouched.
fn bidirectional_support(span: &str, source: &str) -> f64 {
    crate::faithfulness::overlap_fraction(span, source)
        .max(crate::faithfulness::overlap_fraction(source, span))
}

/// Normalize raw LLM marker output: `[ 1 ]` -> `[1]`, `[1,3]` -> `[1][3]`.
fn normalize_markers(body: &str) -> String {
    let spaced_re = regex::Regex::new(r"\[\s*(\d+)\s*\]").expect("static regex");
    let normalized = spaced_re.replace_all(body, "[$1]");

    let comma_re = regex::Regex::new(r"\[(\d+(?:\s*,\s*\d+)+)\]").expect("static regex");
    comma_re
        .replace_all(&normalized, |caps: &regex::Captures| {
            caps[1]
                .split(',')
                .map(|n| format!("[{}]", n.trim()))
                .collect::<String>()
        })
        .into_owned()
}

/// Strip out-of-range markers (index 0 or > sources.len()), counting each
/// removal into `stripped`. Returns the cleaned body.
fn strip_out_of_range(body: &str, num_sources: usize, stripped: &mut usize) -> String {
    let marker_re = regex::Regex::new(r"\[(\d+)\]").expect("static regex");
    let mut out = String::with_capacity(body.len());
    let mut last_end = 0;
    for cap in marker_re.captures_iter(body) {
        let m = cap.get(0).expect("group 0 always present");
        let n: usize = cap[1].parse().unwrap_or(0);
        out.push_str(&body[last_end..m.start()]);
        if n >= 1 && n <= num_sources {
            out.push_str(m.as_str());
        } else {
            *stripped += 1;
        }
        last_end = m.end();
    }
    out.push_str(&body[last_end..]);
    out
}

/// Normalize markers, strip out-of-range ones, then score every remaining
/// marker occurrence per sentence against the union of its claim's cited
/// sources. Returns the (possibly marker-stripped) body, the per-occurrence
/// citation records in body order, and aggregate stats.
///
/// Sentence boundaries are computed on a marker-free "bare" copy of the
/// body: `split_sentences` requires the terminal punctuation to be directly
/// followed by whitespace, but a marker sits between them (`"claim.[1] Next"`).
/// Removing the marker restores that adjacency (`"claim. Next"`) while each
/// marker's removal position (recorded before it is dropped) still tells us
/// which sentence it belonged to.
pub fn process_citation_output(
    body: &str,
    sources: &[NumberedSource],
) -> (String, Vec<PageCitation>, CitationStats) {
    let normalized = normalize_markers(body);
    let mut stripped = 0usize;
    let clean_body = strip_out_of_range(&normalized, sources.len(), &mut stripped);

    let marker_re = regex::Regex::new(r"\[(\d+)\]").expect("static regex");
    let mut bare_body = String::with_capacity(clean_body.len());
    let mut marker_positions: Vec<(u32, usize)> = Vec::new();
    let mut last_end = 0;
    for cap in marker_re.captures_iter(&clean_body) {
        let m = cap.get(0).expect("group 0 always present");
        let n: u32 = cap[1].parse().unwrap_or(0);
        bare_body.push_str(&clean_body[last_end..m.start()]);
        marker_positions.push((n, bare_body.len()));
        last_end = m.end();
    }
    bare_body.push_str(&clean_body[last_end..]);

    // Sentence spans over the bare body, using the same delimiter
    // `faithfulness::split_sentences` splits on.
    let delim_re = regex::Regex::new(r"(?m)[.!?]+\s+").expect("static regex");
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut prev = 0;
    for m in delim_re.find_iter(&bare_body) {
        spans.push((prev, m.start()));
        prev = m.end();
    }
    spans.push((prev, bare_body.len()));

    // Paragraph spans (blank-line delimited) for the fallback scope: small
    // on-device models attach markers to a paragraph's closing elaboration
    // sentence rather than the fact sentence, so a sentence-only check badges
    // true claims (live smoke 2026-07-03: 2/3 supported claims scored 0.0).
    // A claim that fails at sentence scope retries against its enclosing
    // paragraph; the record's `scope` field keeps the weaker guarantee visible.
    let para_re = regex::Regex::new(r"\n\s*\n").expect("static regex");
    let mut para_spans: Vec<(usize, usize)> = Vec::new();
    let mut pprev = 0;
    for m in para_re.find_iter(&bare_body) {
        para_spans.push((pprev, m.start()));
        pprev = m.end();
    }
    para_spans.push((pprev, bare_body.len()));

    let mut citations = Vec::new();
    let mut occurrence = 0u32;
    let mut verified = 0usize;
    let mut unverified = 0usize;

    let mut i = 0;
    while i < marker_positions.len() {
        let span_idx = spans
            .iter()
            .rposition(|s| s.0 <= marker_positions[i].1)
            .unwrap_or(0);
        let mut group = vec![marker_positions[i]];
        let mut j = i + 1;
        while j < marker_positions.len() {
            let next_span_idx = spans
                .iter()
                .rposition(|s| s.0 <= marker_positions[j].1)
                .unwrap_or(0);
            if next_span_idx != span_idx {
                break;
            }
            group.push(marker_positions[j]);
            j += 1;
        }

        let (span_start, span_end) = spans[span_idx];
        let sentence = bare_body[span_start..span_end].trim();
        let union: String = group
            .iter()
            .filter_map(|&(n, _)| sources.get((n - 1) as usize))
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        // Tier 1: the marker's own sentence. Tier 2 (fallback): the
        // enclosing paragraph — see the para_spans comment above.
        let marker_pos = group[0].1;
        let sentence_verified = bidirectional_support(sentence, &union) >= 0.5;
        let (claim_verified, scope, claim_text) = if sentence_verified {
            (true, "sentence", sentence)
        } else {
            let (p_start, p_end) = para_spans
                .iter()
                .rev()
                .find(|p| p.0 <= marker_pos)
                .copied()
                .unwrap_or((0, bare_body.len()));
            let paragraph = bare_body[p_start..p_end].trim();
            let para_verified = bidirectional_support(paragraph, &union) >= 0.5;
            (para_verified, "paragraph", paragraph)
        };
        if claim_verified {
            verified += group.len();
        } else {
            unverified += group.len();
        }

        for &(n, _) in &group {
            occurrence += 1;
            if let Some(src) = sources.get((n - 1) as usize) {
                // Audit score at the scope that decided the status.
                let score = bidirectional_support(claim_text, &src.text);
                citations.push(PageCitation {
                    occurrence,
                    marker: n,
                    source_kind: src.source_kind.clone(),
                    locator: src.locator.clone(),
                    score,
                    status: if claim_verified {
                        "verified"
                    } else {
                        "unverified"
                    }
                    .to_string(),
                    scope: scope.to_string(),
                });
            }
        }

        i = j;
    }

    (
        clean_body,
        citations,
        CitationStats {
            verified,
            unverified,
            stripped,
        },
    )
}

/// Max legacy pages annotated per sweep tick.
const BACKFILL_BATCH_SIZE: usize = 5;
/// Consecutive failed annotation calls before a page is poison-pilled
/// (`citations = '[]'`, changelog notes the giveup).
const MAX_ANNOTATE_ATTEMPTS: i64 = 3;
/// Changelog cap, matching `post_write.rs`'s `DEFAULT_CHANGELOG_CAP`.
const CHANGELOG_CAP: usize = 20;

/// `app_metadata` key tracking consecutive failed annotation calls for one
/// exact Page generation. A stale inference can only touch its old-version
/// budget, never consume retries for the current Page.
fn attempt_key(page_id: &str, page_version: i64) -> String {
    format!("citation_backfill_attempts:{page_id}:v{page_version}")
}

/// Collapse all whitespace runs to a single space and trim. Used by the
/// annotate-only guard to compare the model output against the input body
/// independent of incidental whitespace reflow.
fn normalize_ws(s: &str) -> String {
    let re = regex::Regex::new(r"\s+").expect("static regex");
    re.replace_all(s.trim(), " ").to_string()
}

/// Build a changelog entry for the annotate-only sweep and append it to the
/// page's existing changelog (best-effort: a read/parse failure falls back to
/// a single-entry array rather than losing the write).
async fn build_backfill_changelog(
    db: &MemoryDB,
    page_id: &str,
    version: i64,
    citations_summary: &str,
) -> String {
    let entry = serde_json::json!({
        "version": version,
        "at": chrono::Utc::now().timestamp(),
        "edited_by": "citation_backfill",
        "citations_summary": citations_summary,
    });
    let existing = db
        .get_page_changelog(page_id)
        .await
        .unwrap_or_else(|_| "[]".to_string());
    crate::db::append_changelog_entry(&existing, entry, CHANGELOG_CAP)
        .unwrap_or_else(|_| "[]".to_string())
}

/// Record a failed annotate attempt (provider error, guard rejection, or zero
/// markers) against the page's attempt counter. On the 3rd consecutive failure,
/// poison-pills the page (`citations = '[]'`, changelog notes the giveup with
/// `giveup_reason`) and clears the counter; otherwise bumps it.
async fn record_annotate_failure(
    db: &MemoryDB,
    page_id: &str,
    page_version: i64,
    giveup_reason: &str,
) -> Result<(), WenlanError> {
    let key = attempt_key(page_id, page_version);
    let attempts = db
        .get_app_metadata(&key)
        .await?
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(0)
        + 1;
    if attempts >= MAX_ANNOTATE_ATTEMPTS {
        let changelog = build_backfill_changelog(db, page_id, page_version, giveup_reason).await;
        let _ = db
            .set_page_citations_with_changelog_at_version(
                page_id,
                Some("[]"),
                &changelog,
                page_version,
            )
            .await;
        let _ = db.set_app_metadata(&key, "0").await;
    } else {
        let _ = db.set_app_metadata(&key, &attempts.to_string()).await;
        log::info!(
            "[citation_backfill] page {page_id} annotate attempt failed (attempt {attempts})"
        );
    }
    Ok(())
}

/// Annotate-only backfill sweep: pick up to `BACKFILL_BATCH_SIZE` active pages
/// with `citations IS NULL`, insert `[N]` markers against their source
/// evidence, and save. A deterministic prose guard (marker-stripped output
/// must whitespace-normalize-equal the input body) rejects any output that
/// rewrites text; 3 consecutive rejections poison-pill the page to
/// `citations = '[]'` so the sweep doesn't retry it forever (it regains
/// citations naturally at its next growth re-distill).
pub async fn run_citation_backfill_tick(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
) -> Result<(), WenlanError> {
    run_citation_backfill_with_page_limit(db, llm, prompts, BACKFILL_BATCH_SIZE)
        .await
        .map(|_| ())
}

/// Advance citation backfill by one selected page. A page without source
/// evidence may finish without an LLM call; a page with evidence performs at
/// most one annotate request.
pub async fn run_citation_backfill_slice(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
) -> Result<usize, WenlanError> {
    run_citation_backfill_with_page_limit(db, llm, prompts, 1).await
}

async fn run_citation_backfill_with_page_limit(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    page_limit: usize,
) -> Result<usize, WenlanError> {
    let page_ids = db.get_pages_missing_citations(page_limit).await?;
    let selected = page_ids.len();
    for page_id in page_ids {
        let Some(page) = db.get_page(&page_id).await? else {
            continue;
        };

        let evidence = db.get_page_evidence(&page_id).await.unwrap_or_default();
        let evidence_sources: Vec<(String, String)> = evidence
            .iter()
            .filter_map(|e| {
                e.locator
                    .clone()
                    .map(|locator| (locator, e.source_kind.clone()))
            })
            .collect();
        let locators: Vec<String> = evidence_sources
            .iter()
            .map(|(locator, _)| locator.clone())
            .collect();
        if locators.is_empty() {
            let changelog = build_backfill_changelog(
                db,
                &page_id,
                page.version,
                "citation backfill gave up: no source evidence",
            )
            .await;
            let _ = db
                .set_page_citations_with_changelog_at_version(
                    &page_id,
                    Some("[]"),
                    &changelog,
                    page.version,
                )
                .await;
            continue;
        }

        let mems = db.get_memories_by_source_ids(&locators).await?;
        let source_kinds: std::collections::HashMap<&str, &str> = evidence_sources
            .iter()
            .map(|(locator, source_kind)| (locator.as_str(), source_kind.as_str()))
            .collect();
        let numbered: Vec<NumberedSource> = mems
            .iter()
            .enumerate()
            .map(|(i, m)| NumberedSource {
                index: (i + 1) as u32,
                source_kind: source_kinds
                    .get(m.source_id.as_str())
                    .copied()
                    .unwrap_or("memory")
                    .to_string(),
                locator: m.source_id.clone(),
                text: m.content.chars().take(SOURCE_TEXT_CAP).collect(),
            })
            .collect();

        let user_prompt = format!(
            "## Page Body\n{}\n\n## Numbered Sources\n{}",
            page.content,
            build_numbered_block(&numbered)
        );
        let raw = match llm
            .generate(LlmRequest {
                system_prompt: Some(prompts.annotate_citations.clone()),
                user_prompt,
                max_tokens: llm.recommended_max_output(),
                temperature: 0.0,
                label: Some("citation_annotate".to_string()),
                timeout_secs: None,
            })
            .await
        {
            Ok(raw) => raw,
            Err(error) => {
                log::warn!("[citation_backfill] page {page_id} provider error: {error}");
                record_annotate_failure(
                    db,
                    &page_id,
                    page.version,
                    "citation backfill gave up: provider error after 3 attempts",
                )
                .await?;
                continue;
            }
        };
        let out = crate::llm_provider::strip_think_tags(&raw)
            .trim()
            .to_string();

        // ⚖ deterministic prose guard: markers-stripped output must equal the
        // input body (whitespace-normalized) — legacy prose is never changed
        // by this sweep.
        let same =
            normalize_ws(&strip_markers(&out)) == normalize_ws(&strip_markers(&page.content));
        if same {
            let (body, cites, stats) = process_citation_output(&out, &numbered);
            if cites.is_empty() {
                // Zero markers is a failed attempt per spec §6 ("guard
                // rejections OR zero markers") — retry up to
                // MAX_ANNOTATE_ATTEMPTS instead of draining the page on the
                // first pass.
                record_annotate_failure(
                    db,
                    &page_id,
                    page.version,
                    "citation backfill gave up: zero markers after 3 attempts",
                )
                .await?;
            } else {
                let json = serde_json::to_string(&cites).unwrap_or_else(|_| "[]".into());
                let changelog =
                    build_backfill_changelog(db, &page_id, page.version + 1, &stats.summary())
                        .await;
                let existing_sources: Vec<&str> =
                    page.source_memory_ids.iter().map(String::as_str).collect();
                let committed = db
                    .try_update_page_content_with_changelog_at_version(
                        &page_id,
                        &body,
                        &existing_sources,
                        "citation_backfill",
                        &changelog,
                        Some(&json),
                        page.version,
                        None,
                    )
                    .await?;
                if committed {
                    let _ = db
                        .set_app_metadata(&attempt_key(&page_id, page.version), "0")
                        .await;
                }
            }
        } else {
            record_annotate_failure(
                db,
                &page_id,
                page.version,
                "citation backfill gave up: annotate guard rejected 3x",
            )
            .await?;
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srcs() -> Vec<NumberedSource> {
        vec![
            NumberedSource {
                index: 1,
                source_kind: "memory".into(),
                locator: "mem_a".into(),
                text: "The daemon binds to port 7878 by default".into(),
            },
            NumberedSource {
                index: 2,
                source_kind: "memory".into(),
                locator: "mem_b".into(),
                text: "FastEmbed uses BGE-Base embeddings with 768 dimensions".into(),
            },
        ]
    }

    #[test]
    fn resolves_page_evidence_source_kind_from_source_row_shape() {
        let cases = [
            (
                "folder document",
                "memory",
                Some("folder"),
                "notes::/Users/lucian/Notes/report.pdf",
                "external_file",
            ),
            (
                "webpage capture",
                "webpage",
                None,
                "https://example.com/research",
                "external_url",
            ),
            (
                "authored source",
                "authored",
                None,
                "page_manual_summary",
                "authored",
            ),
            (
                "plain memory",
                "memory",
                Some("claude-code"),
                "mem_plain",
                "memory",
            ),
        ];

        for (label, source, source_agent, source_id, expected) in cases {
            assert_eq!(
                resolve_page_evidence_source_kind(source, source_agent, source_id),
                expected,
                "{label}"
            );
        }
    }

    #[test]
    fn numbered_block_format() {
        let b = build_numbered_block(&srcs());
        assert!(b.starts_with("[1] The daemon"));
        assert!(b.contains("\n\n[2] FastEmbed"));
    }

    #[test]
    fn verified_claim_gets_citation() {
        let body = "The daemon binds to port 7878 by default.[1] Unrelated hallucinated claim about quantum computing.[2]";
        let (out, cites, stats) = process_citation_output(body, &srcs());
        assert_eq!(out, body); // in-range markers stay in the body
        assert_eq!(cites.len(), 2);
        assert_eq!(cites[0].status, "verified");
        assert_eq!(cites[0].locator, "mem_a");
        assert_eq!(cites[1].status, "unverified");
        assert_eq!(stats.verified, 1);
        assert_eq!(stats.unverified, 1);
    }

    #[test]
    fn out_of_range_marker_stripped() {
        let body = "A claim.[7] Another about the daemon port 7878 binding default.[1]";
        let (out, cites, stats) = process_citation_output(body, &srcs());
        assert!(!out.contains("[7]"));
        assert!(out.contains("[1]"));
        assert_eq!(cites.len(), 1);
        assert_eq!(stats.stripped, 1);
    }

    #[test]
    fn malformed_markers_normalized() {
        let body = "The daemon binds port 7878 default.[ 1 ] Embeddings use BGE-Base 768 dimensions FastEmbed.[1,2]";
        let (out, cites, _s) = process_citation_output(body, &srcs());
        assert!(out.contains("default.[1]"));
        assert!(out.contains("[1][2]"));
        assert_eq!(cites.len(), 3);
    }

    #[test]
    fn reused_marker_gets_per_occurrence_status() {
        // Separate paragraphs so the second occurrence cannot inherit the
        // first paragraph's support via the paragraph-scope fallback.
        let body =
            "The daemon binds to port 7878 by default.[1]\n\nCompletely unrelated quantum claim.[1]";
        let (_o, cites, _s) = process_citation_output(body, &srcs());
        assert_eq!(cites.len(), 2);
        assert_eq!((cites[0].occurrence, &cites[0].status[..]), (1, "verified"));
        assert_eq!(cites[0].scope, "sentence");
        assert_eq!(
            (cites[1].occurrence, &cites[1].status[..]),
            (2, "unverified")
        );
        assert_eq!(cites[1].scope, "paragraph"); // both tiers tried, both failed
    }

    #[test]
    fn verbose_claim_verifies_via_source_coverage() {
        // Verbose synthesis: the sentence contains the WHOLE source fact plus
        // elaboration vocabulary that dilutes the claim-token direction below
        // the floor. The source-coverage direction (all source tokens present
        // in the span) verifies it.
        let body = "Specifically, the daemon binds to port 7878 by default, \
                    which reviewers consider a sensible hardening choice overall.[1]";
        let (_o, cites, _s) = process_citation_output(body, &srcs());
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].status, "verified");
        assert_eq!(cites[0].scope, "sentence");
        assert!(cites[0].score >= 0.5);
    }

    #[test]
    fn elaboration_sentence_verifies_at_paragraph_scope() {
        // Small models attach the marker to a paragraph's closing
        // elaboration sentence; the fact lives in the preceding sentence.
        // Sentence scope fails, the enclosing paragraph clears the floor.
        let body = "The daemon binds to port 7878 by default. \
                    This binding reduces exposure.[1]";
        let (_o, cites, _s) = process_citation_output(body, &srcs());
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].status, "verified");
        assert_eq!(cites[0].scope, "paragraph");
        assert!(cites[0].score >= 0.5);
    }

    #[test]
    fn multi_marker_claim_verified_against_union() {
        // Claim draws half its tokens from each source: the claim-token
        // direction fails each source alone but passes the union.
        let body = "The daemon port 7878 uses BGE-Base embeddings with 768 dimensions.[1][2]";
        let (_o, cites, _s) = process_citation_output(body, &srcs());
        assert!(cites.iter().all(|c| c.status == "verified"));
        assert!(cites.iter().all(|c| c.score > 0.0)); // per-source audit scores populated
    }

    #[test]
    fn strip_markers_removes_all() {
        assert_eq!(
            strip_markers("Claim one.[1] Claim two.[12]"),
            "Claim one. Claim two."
        );
        assert_eq!(strip_markers("No markers here."), "No markers here.");
    }

    #[test]
    fn zero_markers_yields_empty_records() {
        let (out, cites, stats) = process_citation_output("Plain body.", &srcs());
        assert_eq!(out, "Plain body.");
        assert!(cites.is_empty());
        assert_eq!(stats.verified + stats.unverified + stats.stripped, 0);
    }

    // -- Task 7: annotate-only backfill tick --

    use crate::llm_provider::{LlmBackend, LlmError, LlmProvider, MockProvider};
    use crate::prompts::PromptRegistry;
    use std::sync::Arc;
    use tokio::sync::Notify;

    struct BlockingCitationProvider {
        entered: Arc<Notify>,
        release: Arc<Notify>,
        response: String,
    }

    #[async_trait::async_trait]
    impl LlmProvider for BlockingCitationProvider {
        async fn generate(&self, _request: LlmRequest) -> Result<String, LlmError> {
            self.entered.notify_one();
            self.release.notified().await;
            Ok(self.response.clone())
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "blocking-citation"
        }

        fn backend(&self) -> LlmBackend {
            LlmBackend::OnDevice
        }

        fn kind(&self) -> &'static str {
            "test"
        }
    }

    /// Insert a bare `memories` row so `get_memories_by_source_ids` can find it.
    /// Mirrors the raw-insert pattern used by `synthesis::distill` tests.
    async fn insert_test_memory(db: &crate::db::MemoryDB, source_id: &str, content: &str) {
        insert_test_memory_with_agent(db, source_id, content, "claude-code").await;
    }

    async fn insert_test_memory_with_agent(
        db: &crate::db::MemoryDB,
        source_id: &str,
        content: &str,
        source_agent: &str,
    ) {
        let now_ts = chrono::Utc::now().timestamp();
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, source_agent, created_at, last_modified, confirmed, stability, source) \
             VALUES (?1, ?1, ?1, ?2, 0, 'text', 'fact', ?3, ?4, ?4, 1, 'confirmed', 'memory')",
            libsql::params![
                source_id.to_string(),
                content.to_string(),
                source_agent.to_string(),
                now_ts
            ],
        )
        .await
        .unwrap();
    }

    const BACKFILL_BODY: &str = "The daemon binds to port 7878 by default.";
    const BACKFILL_MEM_CONTENT: &str = "The daemon binds to port 7878 by default";

    async fn seed_backfill_page(db: &crate::db::MemoryDB, page_id: &str, with_evidence: bool) {
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(page_id, "T", None, BACKFILL_BODY, None, None, &[], &now)
            .await
            .unwrap();
        if with_evidence {
            insert_test_memory(db, "mem_a", BACKFILL_MEM_CONTENT).await;
            db.link_page_evidence(page_id, "memory", Some("mem_a"), None, "test")
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn backfill_happy_path_saves_citations_body_unchanged_prose() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_happy", true).await;

        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let prompts = PromptRegistry::default();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        let page = db.get_page("p_happy").await.unwrap().unwrap();
        assert_eq!(page.content, annotated, "annotated body should be saved");
        assert_eq!(page.citations.len(), 1, "citations: {:?}", page.citations);
        assert_eq!(page.citations[0].status, "verified");
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_happy".to_string()),
            "page should no longer be citations-missing"
        );
    }

    #[tokio::test]
    async fn citation_result_for_old_page_version_is_dropped() {
        let (db, _dir) = crate::db::tests::test_db().await;
        insert_test_memory(&db, "mem_citation_old", BACKFILL_MEM_CONTENT).await;
        insert_test_memory(&db, "mem_citation_new", "The current replacement source").await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "p_citation_cas",
            "T",
            None,
            BACKFILL_BODY,
            None,
            None,
            &["mem_citation_old"],
            &now,
        )
        .await
        .unwrap();

        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let llm: Arc<dyn LlmProvider> = Arc::new(BlockingCitationProvider {
            entered: entered.clone(),
            release: release.clone(),
            response: format!("{BACKFILL_BODY}[1]"),
        });
        let db = Arc::new(db);
        let task = {
            let db = db.clone();
            tokio::spawn(async move {
                run_citation_backfill_slice(&db, &llm, &PromptRegistry::default()).await
            })
        };

        entered.notified().await;
        db.update_page_content(
            "p_citation_cas",
            "The user replaced this Page while citation inference was running.",
            &["mem_citation_new"],
            "manual_edit",
        )
        .await
        .unwrap();
        release.notify_one();
        task.await.unwrap().unwrap();

        let page = db.get_page("p_citation_cas").await.unwrap().unwrap();
        assert_eq!(page.version, 2, "stale citation output must not commit");
        assert_eq!(
            page.content,
            "The user replaced this Page while citation inference was running."
        );
        assert_eq!(
            page.source_memory_ids,
            vec!["mem_citation_new".to_string()],
            "stale evidence must not be restored"
        );
    }

    #[tokio::test]
    async fn citation_result_is_dropped_when_page_is_archived() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_citation_archived", true).await;
        let initial_version = db
            .get_page("p_citation_archived")
            .await
            .unwrap()
            .unwrap()
            .version;

        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let llm: Arc<dyn LlmProvider> = Arc::new(BlockingCitationProvider {
            entered: entered.clone(),
            release: release.clone(),
            response: format!("{BACKFILL_BODY}[1]"),
        });
        let db = Arc::new(db);
        let task = {
            let db = db.clone();
            tokio::spawn(async move {
                run_citation_backfill_slice(&db, &llm, &PromptRegistry::default()).await
            })
        };

        entered.notified().await;
        db.archive_page("p_citation_archived").await.unwrap();
        release.notify_one();
        task.await.unwrap().unwrap();

        let page = db.get_page("p_citation_archived").await.unwrap().unwrap();
        assert_eq!(page.status, "archived");
        assert_eq!(
            page.version,
            initial_version + 1,
            "archive must advance the Page generation"
        );
        assert_eq!(
            page.content, BACKFILL_BODY,
            "in-flight citation output must not rewrite an archived Page"
        );
        assert!(page.citations.is_empty());
    }

    #[tokio::test]
    async fn ambient_backfill_slice_processes_at_most_one_page() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        insert_test_memory(&db, "mem_slice", BACKFILL_MEM_CONTENT).await;
        for page_id in ["p_slice_a", "p_slice_b"] {
            db.insert_page(page_id, "T", None, BACKFILL_BODY, None, None, &[], &now)
                .await
                .unwrap();
            db.link_page_evidence(page_id, "memory", Some("mem_slice"), None, "test")
                .await
                .unwrap();
        }

        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let prompts = PromptRegistry::default();

        let processed = run_citation_backfill_slice(&db, &llm, &prompts)
            .await
            .unwrap();
        assert_eq!(processed, 1);

        let remaining = db.get_pages_missing_citations(10).await.unwrap();
        assert_eq!(remaining.len(), 1, "one ambient turn processes one page");
    }

    #[tokio::test]
    async fn ambient_backfill_slice_reports_empty_backlog() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new("unused"));

        let processed = run_citation_backfill_slice(&db, &llm, &PromptRegistry::default())
            .await
            .unwrap();

        assert_eq!(processed, 0);
    }

    #[tokio::test]
    async fn backfill_preserves_external_file_source_kind() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let page_id = "p_external_file";
        let source_id = "folder-notes::backfill.md";
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(page_id, "T", None, BACKFILL_BODY, None, None, &[], &now)
            .await
            .unwrap();
        insert_test_memory_with_agent(&db, source_id, BACKFILL_MEM_CONTENT, "folder").await;
        db.link_page_evidence(page_id, "external_file", Some(source_id), None, "test")
            .await
            .unwrap();

        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let prompts = PromptRegistry::default();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        let page = db.get_page(page_id).await.unwrap().unwrap();
        assert_eq!(page.citations.len(), 1, "citations: {:?}", page.citations);
        assert_eq!(page.citations[0].source_kind, "external_file");
        assert_eq!(page.citations[0].locator, source_id);
    }

    #[tokio::test]
    async fn backfill_guard_rejects_rewritten_prose_and_records_attempt() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_guard", true).await;

        let rewritten = "A completely different sentence about something else.[1]";
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(rewritten));
        let prompts = PromptRegistry::default();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        let page = db.get_page("p_guard").await.unwrap().unwrap();
        assert_eq!(page.content, BACKFILL_BODY, "prose must never be rewritten");
        assert!(page.citations.is_empty());
        assert!(
            db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_guard".to_string()),
            "citations should still be NULL (not processed)"
        );
        let version = db.get_page("p_guard").await.unwrap().unwrap().version;
        let attempts = db
            .get_app_metadata(&attempt_key("p_guard", version))
            .await
            .unwrap();
        assert_eq!(attempts.as_deref(), Some("1"));
    }

    #[tokio::test]
    async fn backfill_poison_pill_after_three_guard_rejections() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_poison", true).await;

        let rewritten = "A completely different sentence about something else.[1]";
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(rewritten));
        let prompts = PromptRegistry::default();

        for _ in 0..3 {
            run_citation_backfill_tick(&db, &llm, &prompts)
                .await
                .unwrap();
        }

        let page = db.get_page("p_poison").await.unwrap().unwrap();
        assert_eq!(page.content, BACKFILL_BODY, "prose must never be rewritten");
        assert!(page.citations.is_empty());
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_poison".to_string()),
            "citations should be '[]' (gave up), not NULL"
        );
        let log = db.get_page_changelog("p_poison").await.unwrap();
        assert!(
            log.contains("citation backfill gave up"),
            "changelog: {log}"
        );
        let version = db.get_page("p_poison").await.unwrap().unwrap().version;
        let attempts = db
            .get_app_metadata(&attempt_key("p_poison", version))
            .await
            .unwrap();
        assert_eq!(
            attempts.as_deref(),
            Some("0"),
            "attempt key must be cleared"
        );
    }

    #[tokio::test]
    async fn backfill_provider_error_at_attempt_cap_is_recorded_and_terminal() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_provider_error", true).await;
        let version = db
            .get_page("p_provider_error")
            .await
            .unwrap()
            .unwrap()
            .version;
        let key = attempt_key("p_provider_error", version);
        db.set_app_metadata(&key, "2").await.unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::unavailable());
        let selected = run_citation_backfill_slice(&db, &llm, &PromptRegistry::default())
            .await
            .expect("a provider failure must advance durable retry state");

        assert_eq!(selected, 1);
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_provider_error".to_string()),
            "the third provider failure must terminally drain this Page generation"
        );
        assert_eq!(
            db.get_app_metadata(&key).await.unwrap().as_deref(),
            Some("0"),
            "the terminal attempt must clear the generation-scoped counter"
        );
        let log = db.get_page_changelog("p_provider_error").await.unwrap();
        assert!(
            log.contains("provider error after 3 attempts"),
            "changelog: {log}"
        );
    }

    #[tokio::test]
    async fn backfill_happy_path_preserves_source_memory_ids() {
        // Regression: the annotate-success path must not clobber
        // `source_memory_ids` with an empty array (it broke
        // page refresh / `max_source_trust_tier` / page-growth append, which
        // all read the page's linked sources back).
        let (db, _dir) = crate::db::tests::test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "p_sources",
            "T",
            None,
            BACKFILL_BODY,
            None,
            None,
            &["mem_a"],
            &now,
        )
        .await
        .unwrap();
        insert_test_memory(&db, "mem_a", BACKFILL_MEM_CONTENT).await;
        db.link_page_evidence("p_sources", "memory", Some("mem_a"), None, "test")
            .await
            .unwrap();

        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let prompts = PromptRegistry::default();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        let page = db.get_page("p_sources").await.unwrap().unwrap();
        assert_eq!(
            page.source_memory_ids,
            vec!["mem_a".to_string()],
            "source_memory_ids must survive the annotate-only save"
        );
    }

    #[tokio::test]
    async fn backfill_zero_markers_retries_before_poison_pill() {
        // Regression: a guard-passing output with zero [N] markers must count
        // toward the 3-attempt poison-pill (spec §6: "guard rejections OR
        // zero markers"), not drain the page on the first tick.
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_zero", true).await;

        // Guard-passing (unchanged prose), but no [N] markers inserted.
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(BACKFILL_BODY));
        let prompts = PromptRegistry::default();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        assert!(
            db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_zero".to_string()),
            "citations should still be NULL after one zero-marker attempt (retry, not drain)"
        );
        let version = db.get_page("p_zero").await.unwrap().unwrap().version;
        let attempts = db
            .get_app_metadata(&attempt_key("p_zero", version))
            .await
            .unwrap();
        assert_eq!(attempts.as_deref(), Some("1"));

        // Two more zero-marker ticks trigger the poison-pill.
        for _ in 0..2 {
            run_citation_backfill_tick(&db, &llm, &prompts)
                .await
                .unwrap();
        }
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_zero".to_string()),
            "citations should be '[]' after 3 zero-marker attempts (gave up)"
        );
        let log = db.get_page_changelog("p_zero").await.unwrap();
        assert!(
            log.contains("citation backfill gave up"),
            "changelog: {log}"
        );
    }

    #[tokio::test]
    async fn backfill_no_evidence_page_gives_up_without_llm_call() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_noevidence", false).await;

        // An unavailable provider errors on every call; if the tick tried to
        // call it, the whole tick would return Err and this test would fail.
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::unavailable());
        let prompts = PromptRegistry::default();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        let page = db.get_page("p_noevidence").await.unwrap().unwrap();
        assert!(page.citations.is_empty());
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_noevidence".to_string()),
            "citations should be '[]' (gave up), not NULL"
        );
        let log = db.get_page_changelog("p_noevidence").await.unwrap();
        assert!(
            log.contains("citation backfill gave up"),
            "changelog: {log}"
        );
    }
}
