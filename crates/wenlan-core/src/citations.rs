// SPDX-License-Identifier: Apache-2.0
//! Per-claim citation numbering, marker parsing, and union-calibrated
//! verification (pure functions), plus the annotate-only backfill sweep that
//! drains legacy (`citations IS NULL`) pages. See
//! `docs/superpowers/specs/2026-07-03-per-claim-citations-design.md`.

use std::sync::Arc;
use std::sync::LazyLock;

use regex::Regex;
use wenlan_types::pages::PageCitation;

use crate::db::MemoryDB;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::prompts::PromptRegistry;
use crate::WenlanError;

/// Cap on source text length embedded in the numbered block, matching
/// `MEM_SNIPPET_CAP` in `synthesis/distill.rs`.
const SOURCE_TEXT_CAP: usize = 800;

// Module-level `LazyLock` statics so each pattern compiles once, matching the
// idiom at `temporal_query.rs:38-54`.
static RE_MARKERS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[\d+\]").unwrap());
static RE_DOUBLE_SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r" {2,}").unwrap());
static RE_SPACED_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\s*(\d+)\s*\]").unwrap());
static RE_COMMA_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+(?:\s*,\s*\d+)+)\]").unwrap());
/// Shared by `strip_out_of_range` and `process_citation_output`.
static RE_MARKER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[(\d+)\]").unwrap());
static RE_PARAGRAPH_BREAK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\s*\n").unwrap());
static RE_WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

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
    let stripped = RE_MARKERS.replace_all(body, "");
    RE_DOUBLE_SPACE
        .replace_all(&stripped, " ")
        .trim()
        .to_string()
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
    let normalized = RE_SPACED_MARKER.replace_all(body, "[$1]");

    RE_COMMA_MARKER
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
    let mut out = String::with_capacity(body.len());
    let mut last_end = 0;
    for cap in RE_MARKER.captures_iter(body) {
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
/// body: `faithfulness::sentence_spans` requires the terminal punctuation to
/// be directly followed by whitespace, but a marker sits between them
/// (`"claim.[1] Next"`).
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

    let mut bare_body = String::with_capacity(clean_body.len());
    let mut marker_positions: Vec<(u32, usize)> = Vec::new();
    let mut last_end = 0;
    for cap in RE_MARKER.captures_iter(&clean_body) {
        let m = cap.get(0).expect("group 0 always present");
        let n: u32 = cap[1].parse().unwrap_or(0);
        bare_body.push_str(&clean_body[last_end..m.start()]);
        marker_positions.push((n, bare_body.len()));
        last_end = m.end();
    }
    bare_body.push_str(&clean_body[last_end..]);

    // Sentence spans over the bare body. The boundary rule lives in
    // `faithfulness::sentence_spans` and only there — this path needs the
    // offsets (to attribute each marker to its sentence), which is why it
    // takes the span form rather than `split_sentences`.
    let spans = crate::faithfulness::sentence_spans(&bare_body);

    // Paragraph spans (blank-line delimited) for the fallback scope: small
    // on-device models attach markers to a paragraph's closing elaboration
    // sentence rather than the fact sentence, so a sentence-only check badges
    // true claims (live smoke 2026-07-03: 2/3 supported claims scored 0.0).
    // A claim that fails at sentence scope retries against its enclosing
    // paragraph; the record's `scope` field keeps the weaker guarantee visible.
    let mut para_spans: Vec<(usize, usize)> = Vec::new();
    let mut pprev = 0;
    for m in RE_PARAGRAPH_BREAK.find_iter(&bare_body) {
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

/// `app_metadata` key tracking consecutive failed annotation calls for a
/// page — exactly ONE row per page, ever. Which Page generation the stored
/// count belongs to is encoded in the VALUE (`attempt_generation`), not the
/// key, so this can be looked up and cleaned up with a single row instead
/// of accumulating one dead key per generation forever.
pub(crate) fn attempt_key(page_id: &str) -> String {
    format!("citation_backfill_attempts:{page_id}")
}

/// The value prefix a stored attempt count must start with to belong to
/// this exact Page generation. Keyed on BOTH `version` and
/// `source_revision`: a source attach (`link_page_source`) bumps only
/// `source_revision`, leaving `version` unchanged, so matching on `version`
/// alone would let failed attempts against the OLD evidence set carry over
/// and poison-pill a page that has since gained fresh, never-tried
/// evidence.
fn attempt_generation(page_version: i64, source_revision: i64) -> String {
    format!("v{page_version}:s{source_revision}:")
}

/// Parse the attempt count out of a stored `app_metadata` value, but only
/// when it was recorded against `generation` — a value from a different
/// (older or newer) generation is stale for this read and counts as zero,
/// exactly as if no row existed yet.
fn parse_attempts(value: &str, generation: &str) -> i64 {
    value
        .strip_prefix(generation)
        .and_then(|rest| rest.parse::<i64>().ok())
        .unwrap_or(0)
}

/// Collapse all whitespace runs to a single space and trim. Used by the
/// annotate-only guard to compare the model output against the input body
/// independent of incidental whitespace reflow.
fn normalize_ws(s: &str) -> String {
    RE_WHITESPACE.replace_all(s.trim(), " ").to_string()
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
    expected_source_revision: i64,
    giveup_reason: &str,
) -> Result<(), WenlanError> {
    let key = attempt_key(page_id);
    let generation = attempt_generation(page_version, expected_source_revision);
    let attempts = db
        .get_app_metadata(&key)
        .await?
        .map(|v| parse_attempts(&v, &generation))
        .unwrap_or(0)
        + 1;
    if attempts >= MAX_ANNOTATE_ATTEMPTS {
        let changelog = build_backfill_changelog(db, page_id, page_version, giveup_reason).await;
        let landed = db
            .set_page_citations_with_changelog_at_version(
                page_id,
                Some("[]"),
                &changelog,
                page_version,
                expected_source_revision,
            )
            .await?;
        if landed {
            // Terminal write: `citations` is now poison-pilled to `[]` for
            // this generation, so no future write will ever key on it
            // again. Delete the counter only because our own write landed —
            // a rejected CAS means the generation already moved on, and the
            // stale value left behind will read as zero attempts against
            // whatever generation comes next (`parse_attempts`), so leaving
            // it is harmless while deleting it unconditionally here could
            // race a re-arm's own counter (see the helper's doc comment).
            let _ = db
                .delete_app_metadata_if_value_starts_with(&key, &generation)
                .await;
        }
    } else {
        let _ = db
            .set_app_metadata(&key, &format!("{generation}{attempts}"))
            .await;
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
        // Read the fence counter FIRST, then the fields it protects. A
        // concurrent source attach resets `citations` to NULL and bumps
        // `source_revision` to re-arm this page for backfill, but never
        // bumps `version` -- threading `source_revision` into every CAS
        // write below closes the window where a stale result computed from
        // evidence read before that attach would otherwise still match on
        // `version` alone and overwrite the freshly-armed page. Reading it
        // in this order (fence, then the fields it protects) also matters:
        // an attach that lands AFTER this read bumps the counter past what
        // we captured, so the CAS below catches it; an attach that lands
        // BEFORE this read is already reflected in the `get_page` that
        // follows, so its source list is never stale. Reading `get_page`
        // first would leave a window where an attach landing between the
        // two reads shows up in `source_revision` but not in
        // `page.source_memory_ids` -- both fences would then pass on a
        // write that still carries the old source list and drops the
        // freshly attached source. Use the tolerant read here: a page
        // deleted between selection (`get_pages_missing_citations`, above)
        // and this read must skip like the `get_page` miss right below it,
        // not abort the whole slice with `get_page_source_revision`'s
        // does-not-exist error.
        let Some(source_revision) = db.try_get_page_source_revision(&page_id).await? else {
            continue;
        };
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
            let landed = db
                .set_page_citations_with_changelog_at_version(
                    &page_id,
                    Some("[]"),
                    &changelog,
                    page.version,
                    source_revision,
                )
                .await?;
            if landed {
                // Terminal write (poison-pilled to `[]`): clean up any
                // attempt row a prior guard-rejection/provider-error left
                // behind for this generation before evidence dropped to
                // zero, so app_metadata doesn't keep a dead row this
                // give-up itself never wrote. Only when our own write
                // landed -- a rejected CAS means the generation moved on
                // and the row here already belongs to something else.
                let _ = db
                    .delete_app_metadata_if_value_starts_with(
                        &attempt_key(&page_id),
                        &attempt_generation(page.version, source_revision),
                    )
                    .await;
            }
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

        if numbered.is_empty() {
            // Non-empty evidence locators that all failed to resolve to
            // content is a provenance data bug (e.g. a genuinely pruned
            // chunk), not an annotate failure -- it must never spend an LLM
            // call or count toward the 3-attempt poison-pill. But leaving
            // `citations` NULL is its own bug: `get_pages_missing_citations`
            // orders NULL pages oldest-`last_modified`-first, and this
            // give-up touches neither citations nor last_modified, so the
            // exact same page is re-selected on every later tick -- the
            // same starvation class PR #584 fixed, now starving every page
            // behind this one forever instead of a bounded 3 attempts.
            // Treat it exactly like the "no source evidence" give-up right
            // above: leave the missing-citations selection via `citations =
            // '[]'`, no attempt recorded. Recovery is unchanged -- any
            // source-set change (`link_page_source` / `replace_page_sources`)
            // resets `citations` back to NULL.
            log::warn!(
                "[citation_backfill] page {page_id} has {} source evidence locator(s) that \
                 resolved to zero content rows; giving up without spending an attempt",
                locators.len()
            );
            let changelog = build_backfill_changelog(
                db,
                &page_id,
                page.version,
                &format!(
                    "citation backfill gave up: {} evidence locator(s) resolve to no content",
                    locators.len()
                ),
            )
            .await;
            let landed = db
                .set_page_citations_with_changelog_at_version(
                    &page_id,
                    Some("[]"),
                    &changelog,
                    page.version,
                    source_revision,
                )
                .await?;
            if landed {
                // Terminal write; same landed-gated cleanup as the "no
                // source evidence" give-up above.
                let _ = db
                    .delete_app_metadata_if_value_starts_with(
                        &attempt_key(&page_id),
                        &attempt_generation(page.version, source_revision),
                    )
                    .await;
            }
            continue;
        }

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
                    source_revision,
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
                    source_revision,
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
                // Fenced on BOTH version and source_revision: a concurrent
                // source attach (`link_page_source`) re-arms this page
                // (citations -> NULL, source_revision bumped) without
                // touching `version`, so a version-only CAS here would still
                // match and overwrite the freshly-armed page with a stale
                // annotate result computed from evidence read before that
                // attach. `source_revision` was captured before `page` (see
                // the read-order note above), before the evidence read.
                let committed = db
                    .try_update_page_content_with_changelog_at_versions(
                        &page_id,
                        &body,
                        &existing_sources,
                        &changelog,
                        Some(&json),
                        page.version,
                        source_revision,
                        None,
                    )
                    .await?;
                if committed {
                    // Terminal write (citations now populated): delete
                    // rather than reset the counter so app_metadata doesn't
                    // keep a dead row per generation forever. Only because
                    // our own write landed -- see the give-up branches
                    // above for why an unlanded CAS must leave the row.
                    let _ = db
                        .delete_app_metadata_if_value_starts_with(
                            &attempt_key(&page_id),
                            &attempt_generation(page.version, source_revision),
                        )
                        .await;
                }
            }
        } else {
            record_annotate_failure(
                db,
                &page_id,
                page.version,
                source_revision,
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
        let conn = db.test_primary_session().await;
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

    /// Round-2 doc-citation-locator fix, widened (option 1, `citation_backfill`
    /// combined-fence caller): the annotate-success write now fences on BOTH
    /// `version` and `source_revision`
    /// (`try_update_page_content_with_changelog_at_versions`), closing the
    /// window a version-only CAS left open -- `link_page_source` re-arms a
    /// page (citations back to NULL, `source_revision` bumped) WITHOUT
    /// touching `version`, so a version-only CAS would still match and
    /// clobber the freshly re-armed page with a stale annotate result
    /// computed from evidence read before the attach.
    ///
    /// Unlike the give-up paths
    /// (`backfill_page_rearmed_by_link_page_source_after_giveup_is_annotated_
    /// next_slice`, which never call the LLM and so have no interposable
    /// await), the annotate-success path calls `LlmProvider::generate`,
    /// giving `BlockingCitationProvider` a real seam to race a live
    /// `link_page_source` attach against the in-flight write.
    #[tokio::test]
    async fn citation_result_for_stale_source_revision_is_dropped() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_stale_source_rev", true).await;

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
        insert_test_memory(&db, "mem_stale_new", "A second source attached mid-flight").await;
        db.link_page_source("p_stale_source_rev", "mem_stale_new", "test")
            .await
            .unwrap();
        release.notify_one();
        task.await.unwrap().unwrap();

        let page = db.get_page("p_stale_source_rev").await.unwrap().unwrap();
        assert_eq!(
            page.content, BACKFILL_BODY,
            "stale annotate output computed before the source attach must not commit"
        );
        assert!(
            db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_stale_source_rev".to_string()),
            "the source attach must win: citations stay NULL, not overwritten by the stale write"
        );
        assert_eq!(
            page.source_memory_ids.len(),
            2,
            "the concurrently attached source must be preserved: {:?}",
            page.source_memory_ids
        );
        assert!(page.source_memory_ids.contains(&"mem_a".to_string()));
        assert!(page
            .source_memory_ids
            .contains(&"mem_stale_new".to_string()));

        // The very next slice must pick the page back up and annotate with
        // the new source visible -- proving the fence drops the stale write
        // without starving the page.
        let annotated = format!("{BACKFILL_BODY}[1][2]");
        let llm2: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let selected2 = run_citation_backfill_slice(&db, &llm2, &PromptRegistry::default())
            .await
            .unwrap();
        assert_eq!(selected2, 1, "the next slice must select the re-armed page");
        let resolved = db.get_page("p_stale_source_rev").await.unwrap().unwrap();
        assert_eq!(
            resolved.citations.len(),
            2,
            "both sources must be cited once the page is annotated cleanly: {:?}",
            resolved.citations
        );
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
        let source_revision = db.get_page_source_revision("p_guard").await.unwrap();
        let attempts = db.get_app_metadata(&attempt_key("p_guard")).await.unwrap();
        assert_eq!(
            attempts.as_deref(),
            Some(format!("{}1", attempt_generation(version, source_revision)).as_str())
        );
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
        let attempts = db.get_app_metadata(&attempt_key("p_poison")).await.unwrap();
        assert_eq!(
            attempts, None,
            "a landed terminal write must delete the attempt row, not reset it to \"0\", so \
             app_metadata doesn't keep a dead row per drained generation"
        );
    }

    /// Whether an attempt-counter row exists for a page under the
    /// one-row-per-page scheme (`attempt_key`). Named for continuity with
    /// the per-generation-prefix COUNT this replaces; under the new scheme
    /// there is at most one row per page, so this can only ever be 0 or 1.
    async fn count_attempt_rows(db: &crate::db::MemoryDB, page_id: &str) -> i64 {
        if db
            .get_app_metadata(&attempt_key(page_id))
            .await
            .unwrap()
            .is_some()
        {
            1
        } else {
            0
        }
    }

    /// Round-4 finding (LOW): every terminal citation write (annotate
    /// success, attempts-exhausted `[]`, give-up `[]`) used to leave the
    /// attempt-key row behind forever -- success and the poison-pill wrote
    /// `"0"` instead of deleting, and the give-up paths never touched the
    /// key at all even when an earlier failed attempt on the same
    /// generation had left one. Round 5 replaced the per-generation-key
    /// scheme with a single per-page key whose VALUE encodes the
    /// generation, cleaned up via
    /// `delete_app_metadata_if_value_starts_with` gated on the terminal
    /// write's own CAS having landed.
    #[tokio::test]
    async fn backfill_terminal_writes_delete_the_attempt_counter() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let prompts = PromptRegistry::default();

        // An unrelated page's key must survive every cleanup below.
        db.set_app_metadata(&attempt_key("unrelated_page"), "v1:s1:2")
            .await
            .unwrap();

        // -- Success case -- built manually (not `seed_backfill_page`,
        // which hardcodes evidence memory id "mem_a" and would collide with
        // the poison-pill page's own seeding below).
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "p_cleanup_success",
            "T",
            None,
            BACKFILL_BODY,
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        insert_test_memory(&db, "mem_cleanup_success", BACKFILL_MEM_CONTENT).await;
        db.link_page_evidence(
            "p_cleanup_success",
            "memory",
            Some("mem_cleanup_success"),
            None,
            "test",
        )
        .await
        .unwrap();
        let version = db
            .get_page("p_cleanup_success")
            .await
            .unwrap()
            .unwrap()
            .version;
        let source_revision = db
            .get_page_source_revision("p_cleanup_success")
            .await
            .unwrap();
        // A stray row from an earlier failed attempt on this exact
        // generation, proving the delete removes something real rather
        // than deleting nothing.
        db.set_app_metadata(
            &attempt_key("p_cleanup_success"),
            &format!("{}1", attempt_generation(version, source_revision)),
        )
        .await
        .unwrap();

        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        run_citation_backfill_slice(&db, &llm, &prompts)
            .await
            .unwrap();
        let page = db.get_page("p_cleanup_success").await.unwrap().unwrap();
        assert_eq!(
            page.citations.len(),
            1,
            "sanity: p_cleanup_success must have actually succeeded"
        );
        assert_eq!(
            count_attempt_rows(&db, "p_cleanup_success").await,
            0,
            "the annotate-success terminal write must delete every attempt row for this page"
        );

        // -- Poison-pill case --
        db.insert_page(
            "p_cleanup_poison",
            "T",
            None,
            BACKFILL_BODY,
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        insert_test_memory(&db, "mem_cleanup_poison", BACKFILL_MEM_CONTENT).await;
        db.link_page_evidence(
            "p_cleanup_poison",
            "memory",
            Some("mem_cleanup_poison"),
            None,
            "test",
        )
        .await
        .unwrap();
        let rewritten = "A completely different sentence about something else.[1]";
        let llm_poison: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(rewritten));
        for _ in 0..3 {
            run_citation_backfill_slice(&db, &llm_poison, &prompts)
                .await
                .unwrap();
        }
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_cleanup_poison".to_string()),
            "sanity: p_cleanup_poison must have actually poison-pilled"
        );
        assert_eq!(
            count_attempt_rows(&db, "p_cleanup_poison").await,
            0,
            "the attempts-exhausted terminal write must delete every attempt row for this page"
        );

        assert_eq!(
            db.get_app_metadata(&attempt_key("unrelated_page"))
                .await
                .unwrap()
                .as_deref(),
            Some("v1:s1:2"),
            "an unrelated page's key must survive both cleanups above"
        );
    }

    /// Round-5 finding F2/F3 (LOW): the poison-pill's terminal write
    /// (`set_page_citations_with_changelog_at_version`) is itself CAS-gated
    /// on `(version, source_revision)`. If that CAS is rejected -- the page
    /// moved on since the caller's own stale read -- the attempt counter
    /// must be left exactly as it was: a rejected write is not a terminal
    /// write for the generation that rejected it, and deleting the row
    /// anyway would either lose real attempt history or (worse) wipe a
    /// counter a fresh generation had already started writing to.
    #[tokio::test]
    async fn record_annotate_failure_leaves_the_counter_when_the_poison_cas_is_rejected() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "p_poison_cas_rejected",
            "T",
            None,
            BACKFILL_BODY,
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        let page = db.get_page("p_poison_cas_rejected").await.unwrap().unwrap();
        let version = page.version;
        let stale_source_revision = db
            .get_page_source_revision("p_poison_cas_rejected")
            .await
            .unwrap();

        // Seed 2 prior failures against the (version, stale_source_revision)
        // generation -- the call below is the 3rd, reaching the poison
        // threshold and attempting the terminal CAS write.
        let key = attempt_key("p_poison_cas_rejected");
        let generation = attempt_generation(version, stale_source_revision);
        db.set_app_metadata(&key, &format!("{generation}2"))
            .await
            .unwrap();

        // Advance the ACTUAL page past `stale_source_revision` -- the CAS
        // inside the poison branch below will target the now-superseded
        // value and must be rejected.
        insert_test_memory(&db, "mem_poison_cas_rejected", BACKFILL_MEM_CONTENT).await;
        db.link_page_source("p_poison_cas_rejected", "mem_poison_cas_rejected", "test")
            .await
            .unwrap();
        let current_source_revision = db
            .get_page_source_revision("p_poison_cas_rejected")
            .await
            .unwrap();
        assert_ne!(
            current_source_revision, stale_source_revision,
            "sanity: the attach must advance source_revision past the stale value"
        );

        record_annotate_failure(
            &db,
            "p_poison_cas_rejected",
            version,
            stale_source_revision,
            "citation backfill gave up: test",
        )
        .await
        .unwrap();

        assert!(
            db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_poison_cas_rejected".to_string()),
            "citations must still be NULL -- a rejected CAS must not have poison-pilled the page"
        );
        assert_eq!(
            db.get_app_metadata(&key).await.unwrap().as_deref(),
            Some(format!("{generation}2").as_str()),
            "a rejected terminal CAS must leave the counter exactly as seeded, not delete it"
        );
    }

    /// Round-5 finding F2/F3: the two "give up without evidence" branches
    /// discard the attempt counter unconditionally on the old scheme, so a
    /// stray row left by an earlier failed attempt on this exact generation
    /// must still be cleaned up by a give-up whose own terminal CAS lands
    /// (this give-up never itself wrote the row, but is still responsible
    /// for the generation's cleanup once the row's generation matches its
    /// own terminal write).
    #[tokio::test]
    async fn give_up_branches_delete_the_counter_only_when_their_write_lands() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_giveup_cleanup", false).await;
        let page = db.get_page("p_giveup_cleanup").await.unwrap().unwrap();
        let source_revision = db
            .get_page_source_revision("p_giveup_cleanup")
            .await
            .unwrap();
        db.set_app_metadata(
            &attempt_key("p_giveup_cleanup"),
            &format!("{}1", attempt_generation(page.version, source_revision)),
        )
        .await
        .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::unavailable());
        run_citation_backfill_slice(&db, &llm, &PromptRegistry::default())
            .await
            .unwrap();

        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_giveup_cleanup".to_string()),
            "sanity: the give-up (no source evidence) must have poison-pilled to '[]'"
        );
        assert_eq!(
            count_attempt_rows(&db, "p_giveup_cleanup").await,
            0,
            "the give-up's own landed terminal write must delete the pre-existing attempt row \
             even though this give-up never itself wrote it"
        );
    }

    /// Round-5 finding F2/F3: under the single-row-per-page scheme, a
    /// generation rollover must overwrite the row in place rather than
    /// leaving the old generation's row behind under a different key --
    /// there is only ever ONE `app_metadata` row for a page's attempt
    /// counter, for its whole lifetime.
    #[tokio::test]
    async fn generation_rollover_keeps_exactly_one_attempt_row() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_rollover", true).await;

        let rewritten = "A completely different sentence about something else.[1]";
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(rewritten));
        let prompts = PromptRegistry::default();
        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        let version = db.get_page("p_rollover").await.unwrap().unwrap().version;

        insert_test_memory(&db, "mem_rollover_new", "A newly attached source").await;
        db.link_page_source("p_rollover", "mem_rollover_new", "test")
            .await
            .unwrap();
        let new_source_revision = db.get_page_source_revision("p_rollover").await.unwrap();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        assert_eq!(
            count_attempt_rows(&db, "p_rollover").await,
            1,
            "exactly one attempt row must ever exist for a page"
        );
        assert_eq!(
            db.get_app_metadata(&attempt_key("p_rollover"))
                .await
                .unwrap()
                .as_deref(),
            Some(format!("{}1", attempt_generation(version, new_source_revision)).as_str()),
            "the rolled-over generation's count must start fresh at 1, overwriting the old \
             generation's value in place"
        );
    }

    /// Round-5 finding F2/F3, citations-level: the same guarantee
    /// `delete_app_metadata_if_value_starts_with_only_deletes_a_matching_value`
    /// proves at the db level (db/main_tests.rs), exercised through the
    /// citation-backfill key/generation helpers this feature actually uses.
    #[tokio::test]
    async fn terminal_cleanup_does_not_touch_a_newer_generation() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let key = attempt_key("p_generation_race");
        let older = attempt_generation(1, 1);
        let newer = attempt_generation(1, 2);
        db.set_app_metadata(&key, &format!("{newer}1"))
            .await
            .unwrap();

        let deleted_stale = db
            .delete_app_metadata_if_value_starts_with(&key, &older)
            .await
            .unwrap();
        assert!(
            !deleted_stale,
            "an older generation's prefix must not delete a row a newer generation already wrote"
        );
        assert_eq!(
            db.get_app_metadata(&key).await.unwrap().as_deref(),
            Some(format!("{newer}1").as_str()),
            "the newer generation's row must be untouched"
        );

        let deleted_matching = db
            .delete_app_metadata_if_value_starts_with(&key, &newer)
            .await
            .unwrap();
        assert!(
            deleted_matching,
            "the matching generation's prefix must delete the row"
        );
        assert_eq!(db.get_app_metadata(&key).await.unwrap(), None);
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
        let source_revision = db
            .get_page_source_revision("p_provider_error")
            .await
            .unwrap();
        let key = attempt_key("p_provider_error");
        db.set_app_metadata(
            &key,
            &format!("{}2", attempt_generation(version, source_revision)),
        )
        .await
        .unwrap();

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
            db.get_app_metadata(&key).await.unwrap(),
            None,
            "the terminal attempt must delete the generation-scoped counter, not reset it to \
             \"0\""
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
        let source_revision = db.get_page_source_revision("p_zero").await.unwrap();
        let attempts = db.get_app_metadata(&attempt_key("p_zero")).await.unwrap();
        assert_eq!(
            attempts.as_deref(),
            Some(format!("{}1", attempt_generation(version, source_revision)).as_str())
        );

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

    /// Round-3 finding (LOW): the attempt budget is matched on BOTH
    /// `version` and `source_revision`, not `version` alone -- round 5 moved
    /// that match from the `app_metadata` KEY to a generation prefix in the
    /// VALUE (`attempt_generation`/`parse_attempts`), but the guarantee is
    /// the same. A source attach (`link_page_source`) bumps only
    /// `source_revision`, so two failed attempts against the OLD evidence
    /// set followed by an attach followed by one failure on the NEW
    /// evidence must count as attempt 1 of a fresh budget, not attempt 3 of
    /// the old one -- the new evidence has never actually been tried 3
    /// times, so poison-pilling it now would throw away citations the model
    /// was never even given a fair shot at.
    #[tokio::test]
    async fn attempt_budget_resets_when_source_revision_advances_mid_generation() {
        let (db, _dir) = crate::db::tests::test_db().await;
        seed_backfill_page(&db, "p_budget_reset", true).await;

        let rewritten = "A completely different sentence about something else.[1]";
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(rewritten));
        let prompts = PromptRegistry::default();

        // Two guard rejections against the ORIGINAL source_revision
        // generation.
        for _ in 0..2 {
            run_citation_backfill_tick(&db, &llm, &prompts)
                .await
                .unwrap();
        }
        let version = db
            .get_page("p_budget_reset")
            .await
            .unwrap()
            .unwrap()
            .version;
        let original_source_revision = db.get_page_source_revision("p_budget_reset").await.unwrap();
        assert_eq!(
            db.get_app_metadata(&attempt_key("p_budget_reset"))
                .await
                .unwrap()
                .as_deref(),
            Some(format!("{}2", attempt_generation(version, original_source_revision)).as_str()),
            "sanity: two consecutive rejections must have recorded 2 attempts \
             against the original generation"
        );

        // A concurrent source attach advances the generation: bumps
        // `source_revision` (and resets `citations`, already NULL) without
        // touching `version`.
        insert_test_memory(
            &db,
            "mem_budget_new",
            "A newly attached source mid-generation",
        )
        .await;
        db.link_page_source("p_budget_reset", "mem_budget_new", "test")
            .await
            .unwrap();
        let new_source_revision = db.get_page_source_revision("p_budget_reset").await.unwrap();
        assert_ne!(
            new_source_revision, original_source_revision,
            "sanity: the attach must advance source_revision"
        );

        // One more rejection, now against the NEW generation.
        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        assert!(
            db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_budget_reset".to_string()),
            "one failure on a fresh generation must not poison-pill it -- the \
             old generation's 2 failures must not carry over just because \
             `version` is unchanged"
        );
        let attempts_after = db
            .get_app_metadata(&attempt_key("p_budget_reset"))
            .await
            .unwrap();
        assert_eq!(
            attempts_after.as_deref(),
            Some(format!("{}1", attempt_generation(version, new_source_revision)).as_str()),
            "the new generation's attempt count must start fresh at 1, overwriting the old \
             generation's value in place -- exactly one app_metadata row for this page, ever"
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

    /// Guard B: evidence LOCATORS are non-empty (unlike
    /// `backfill_no_evidence_page_gives_up_without_llm_call`, which has no
    /// evidence at all), but every locator fails to resolve to a content row
    /// -- a provenance data bug (e.g. a genuinely pruned chunk), not an
    /// annotate failure. `MockProvider::unavailable()` errors on any call,
    /// so if the guard failed to fire, the slice would either fall through
    /// to `record_annotate_failure` (setting the attempt key) or return Err.
    ///
    /// The give-up must still leave the missing-citations selection
    /// (`citations = '[]'`, exactly like the "no source evidence" and
    /// "3 failed attempts" give-ups right above/below): leaving `citations`
    /// NULL would re-select this exact page on every later slice forever
    /// (it's always the oldest -- nothing here ever touches
    /// `last_modified`), starving every page queued behind it. The second
    /// half of this test is the starvation regression: a second, resolvable
    /// page queued behind p_orphan must get its turn on the very next
    /// slice.
    #[tokio::test]
    async fn backfill_empty_numbered_skips_without_attempt_or_llm_call() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page("p_orphan", "T", None, BACKFILL_BODY, None, None, &[], &now)
            .await
            .unwrap();
        // Evidence points at a locator with no matching `memories` row at
        // all -- not inserted here on purpose.
        db.link_page_evidence("p_orphan", "memory", Some("orphan_locator"), None, "test")
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::unavailable());
        let prompts = PromptRegistry::default();

        let selected = run_citation_backfill_slice(&db, &llm, &prompts)
            .await
            .expect("the guard must fire before ever calling the (unavailable) LLM");
        assert_eq!(selected, 1, "the slice must select p_orphan");

        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_orphan".to_string()),
            "citations must become '[]' (not stay NULL) so the page leaves \
             the missing-citations selection; NULL would re-select the same \
             page forever and starve every page behind it"
        );
        let attempts = db.get_app_metadata(&attempt_key("p_orphan")).await.unwrap();
        assert_eq!(
            attempts, None,
            "a provenance data bug must not spend an attempt -- an attempt \
             key here would mean the LLM was actually called \
             (MockProvider::unavailable would error and record one)"
        );
        let log = db.get_page_changelog("p_orphan").await.unwrap();
        assert!(
            log.contains("citation backfill gave up: 1 evidence locator(s) resolve to no content"),
            "changelog: {log}"
        );

        // Starvation regression: a second, resolvable page queued behind
        // p_orphan must get annotated on the very next slice, proving the
        // give-up actually freed the queue instead of re-selecting p_orphan
        // forever.
        seed_backfill_page(&db, "p_resolvable", true).await;
        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm2: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let selected2 = run_citation_backfill_slice(&db, &llm2, &prompts)
            .await
            .unwrap();
        assert_eq!(selected2, 1, "the second slice must select p_resolvable");
        let resolved = db.get_page("p_resolvable").await.unwrap().unwrap();
        assert_eq!(
            resolved.citations.len(),
            1,
            "p_resolvable must get annotated on the very next slice, not \
             starved behind p_orphan: {:?}",
            resolved.citations
        );
    }

    /// Round-4 finding (LOW): `get_page_source_revision` errors with
    /// `Validation("page '...' does not exist")` on a missing row. The
    /// backfill loop's first per-page read used to be that call, so a page
    /// deleted between selection (`get_pages_missing_citations`, which
    /// materializes the whole batch up front) and its own turn in the loop
    /// aborted the ENTIRE slice with `Err`, discarding whatever pages ahead
    /// of it in the batch had already been annotated. Fixed by
    /// `try_get_page_source_revision` (`Ok(None)` on a missing row) plus a
    /// `continue`, mirroring the existing `get_page` miss right below it.
    ///
    /// There is no live interposable await between `get_pages_missing_citations`
    /// and the first page's read (same reasoning as the read-order tests
    /// above), so this races the SECOND selected page's read against the
    /// FIRST page's blocking LLM call instead: two pages selected in one
    /// batch, the second deleted while the first is still mid-flight (proof
    /// the batch really did contain both before either was fully
    /// processed), first finishes and gets annotated normally, the loop
    /// then reaches the now-missing second page.
    #[tokio::test]
    async fn backfill_slice_skips_a_page_deleted_after_selection_instead_of_erroring() {
        let (db, _dir) = crate::db::tests::test_db().await;
        // Explicit timestamps: `get_pages_missing_citations` orders
        // `last_modified ASC`, so p_first is selected/processed before
        // p_deleted_mid_slice.
        db.insert_page(
            "p_first",
            "T",
            None,
            BACKFILL_BODY,
            None,
            None,
            &[],
            "2020-01-01T00:00:00Z",
        )
        .await
        .unwrap();
        insert_test_memory(&db, "mem_first", BACKFILL_MEM_CONTENT).await;
        db.link_page_evidence("p_first", "memory", Some("mem_first"), None, "test")
            .await
            .unwrap();

        let later = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "p_deleted_mid_slice",
            "T",
            None,
            BACKFILL_BODY,
            None,
            None,
            &[],
            &later,
        )
        .await
        .unwrap();
        insert_test_memory(&db, "mem_deleted", BACKFILL_MEM_CONTENT).await;
        db.link_page_evidence(
            "p_deleted_mid_slice",
            "memory",
            Some("mem_deleted"),
            None,
            "test",
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
            let llm = llm.clone();
            tokio::spawn(async move {
                run_citation_backfill_with_page_limit(&db, &llm, &PromptRegistry::default(), 2)
                    .await
            })
        };

        // p_first has entered its (blocked) LLM call -- both pages were
        // already selected into the batch by this point.
        entered.notified().await;
        db.delete_page("p_deleted_mid_slice").await.unwrap();
        release.notify_one();

        let selected = task
            .await
            .unwrap()
            .expect("a page deleted mid-slice must be skipped, not error the whole slice");
        assert_eq!(
            selected, 2,
            "the slice must still report both pages as selected"
        );

        let page = db.get_page("p_first").await.unwrap().unwrap();
        assert_eq!(
            page.citations.len(),
            1,
            "p_first, selected ahead of the deleted page, must still be annotated: {:?}",
            page.citations
        );
    }

    /// Round-2 doc-citation-locator fix: the give-up write's terminal CAS
    /// (`set_page_citations_with_changelog_at_version`) now also fences on
    /// `source_revision`, closing the window where a concurrent source
    /// attach (`link_page_source`) re-arms a page (citations back to NULL,
    /// source_revision bumped, `version` untouched) and a stale give-up
    /// write computed from evidence read before that attach would otherwise
    /// still match on `version` and clobber the re-armed page.
    ///
    /// The give-up paths never call the LLM, so unlike the annotate-success
    /// path (`citation_result_for_old_page_version_is_dropped`'s
    /// `BlockingCitationProvider` seam) there is no deterministic,
    /// test-interposable await between a give-up's evidence read and its
    /// terminal write to race a concurrent attach against directly -- see
    /// the round-2 report for that gap. This proves the fix's actual
    /// guarantee end-to-end instead: a page the sweep gave up on is re-armed
    /// by `link_page_source`, and the very next slice picks it up and
    /// annotates it, i.e. the new source_revision fence does not starve the
    /// page it exists to protect.
    #[tokio::test]
    async fn backfill_page_rearmed_by_link_page_source_after_giveup_is_annotated_next_slice() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page("p_rearmed", "T", None, BACKFILL_BODY, None, None, &[], &now)
            .await
            .unwrap();
        // Evidence points at a locator with no matching `memories` row --
        // resolves to zero content, hitting the `numbered.is_empty()`
        // give-up (round-1's fix).
        db.link_page_evidence("p_rearmed", "memory", Some("orphan_locator"), None, "test")
            .await
            .unwrap();

        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::unavailable());
        let prompts = PromptRegistry::default();
        let selected = run_citation_backfill_slice(&db, &llm, &prompts)
            .await
            .expect("the guard must skip before ever calling the (unavailable) LLM");
        assert_eq!(selected, 1, "the slice must select p_rearmed");
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_rearmed".to_string()),
            "sanity: the give-up write must leave citations = '[]', not NULL"
        );

        // Re-arm the SAME page via link_page_source: resets citations to
        // NULL and bumps source_revision without touching version -- the
        // exact concurrent-attach shape the fix guards against, replayed
        // here sequentially (after the give-up commits) as a recovery
        // proof rather than a live interleaving.
        insert_test_memory(&db, "mem_rearmed", BACKFILL_MEM_CONTENT).await;
        db.link_page_source("p_rearmed", "mem_rearmed", "test")
            .await
            .unwrap();
        assert!(
            db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_rearmed".to_string()),
            "sanity: link_page_source must re-arm the page (citations back to NULL)"
        );

        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm2: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let selected2 = run_citation_backfill_slice(&db, &llm2, &prompts)
            .await
            .unwrap();
        assert_eq!(selected2, 1, "the next slice must select p_rearmed again");
        let page = db.get_page("p_rearmed").await.unwrap().unwrap();
        assert_eq!(
            page.citations.len(),
            1,
            "the source_revision fence must not starve the page it protects -- \
             p_rearmed must be annotated on the very next slice: {:?}",
            page.citations
        );
    }

    /// Change A end-to-end through the real backfill sweep: a document
    /// source page's evidence locator is a chunk `id` (document_enrichment.
    /// rs:297), not a `source_id`. Before the read-side fix this locator
    /// resolved to zero content rows (LOCATOR MISMATCH), building an empty
    /// numbered block; after the fix it resolves and the page annotates
    /// normally.
    #[tokio::test]
    async fn backfill_id_shaped_locator_resolves_and_annotates() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page(
            "p_doc_chunk",
            "T",
            None,
            BACKFILL_BODY,
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        // A document chunk row: `id` distinct from `source_id`, mirroring
        // the chunk id document_enrichment.rs:297 mints.
        {
            let now_ts = chrono::Utc::now().timestamp();
            let conn = db.test_primary_session().await;
            conn.execute(
                "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, source_agent, created_at, last_modified, confirmed, stability, source) \
                 VALUES ('chunk_id_0', 'doc_source_id', 'chunk_id_0', ?1, 0, 'text', 'fact', 'folder', ?2, ?2, 1, 'confirmed', 'memory')",
                libsql::params![BACKFILL_MEM_CONTENT.to_string(), now_ts],
            )
            .await
            .unwrap();
        }
        // Evidence locator is the chunk's id, not doc_source_id -- the exact
        // shape a document source page's evidence stores.
        db.link_page_evidence(
            "p_doc_chunk",
            "external_file",
            Some("chunk_id_0"),
            None,
            "test",
        )
        .await
        .unwrap();

        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let prompts = PromptRegistry::default();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        let page = db.get_page("p_doc_chunk").await.unwrap().unwrap();
        assert_eq!(
            page.citations.len(),
            1,
            "the id-shaped locator must resolve to content, producing a \
             non-empty numbered block and a real citation: {:?}",
            page.citations
        );
        assert_eq!(page.citations[0].locator, "chunk_id_0");
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&"p_doc_chunk".to_string()),
            "page should no longer be citations-missing"
        );
    }

    /// Relocated from `tests/page_citations_e2e.rs` (G6 Stage 2 PR 2b):
    /// `link_page_evidence` closed to `#[cfg(test)]` (Q2 ruling), and
    /// `#[cfg(test)]` is invisible across the `tests/` integration-binary
    /// boundary, so this helper's manual evidence attach no longer compiled
    /// there. Moved into the crate's own unit-test suite, where it does.
    /// Exercises the real `create_page` API (not the lower-level
    /// `insert_page`), unlike this module's other `seed_backfill_page`.
    async fn seed_backfill_page_via_create_page(
        db: &crate::db::MemoryDB,
        body: &str,
        mem_id: &str,
        mem_content: &str,
    ) -> String {
        insert_test_memory(db, mem_id, mem_content).await;
        let result = crate::post_write::create_page(
            db,
            wenlan_types::requests::CreateConceptRequest {
                title: "T".to_string(),
                content: body.to_string(),
                summary: None,
                entity_id: None,
                space: None.into(),
                source_memory_ids: vec![],
                creation_kind: Some("authored".to_string()),
                workspace: None,
            },
            "test",
            None,
        )
        .await
        .unwrap();
        db.link_page_evidence(&result.id, "memory", Some(mem_id), None, "test")
            .await
            .unwrap();
        result.id
    }

    /// Relocated from `tests/page_citations_e2e.rs::backfill_annotates_legacy_page`.
    /// Assertions unchanged.
    #[tokio::test]
    async fn backfill_annotates_legacy_page_via_create_page() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let page_id =
            seed_backfill_page_via_create_page(&db, BACKFILL_BODY, "mem_a", BACKFILL_MEM_CONTENT)
                .await;
        assert!(db
            .get_pages_missing_citations(10)
            .await
            .unwrap()
            .contains(&page_id));

        let annotated = format!("{BACKFILL_BODY}[1]");
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&annotated));
        let prompts = PromptRegistry::default();

        run_citation_backfill_tick(&db, &llm, &prompts)
            .await
            .unwrap();

        let page = db.get_page(&page_id).await.unwrap().unwrap();
        assert_eq!(
            page.content, annotated,
            "prose stays byte-identical modulo the inserted marker"
        );
        assert_eq!(page.citations.len(), 1, "citations: {:?}", page.citations);
        assert_eq!(page.citations[0].status, "verified");
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&page_id),
            "page should no longer be citations-missing"
        );
        let changelog = db.get_page_changelog(&page_id).await.unwrap();
        assert!(
            changelog.contains("citation_backfill"),
            "changelog: {changelog}"
        );
    }

    /// Relocated from `tests/page_citations_e2e.rs::backfill_guard_rejects_rewrite_then_poison_pills`.
    /// Assertions unchanged.
    #[tokio::test]
    async fn backfill_guard_rejects_rewrite_then_poison_pills_via_create_page() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let page_id =
            seed_backfill_page_via_create_page(&db, BACKFILL_BODY, "mem_a", BACKFILL_MEM_CONTENT)
                .await;

        let rewritten = "A completely different sentence about something else entirely.[1]";
        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(rewritten));
        let prompts = PromptRegistry::default();

        // 3 consecutive rejected ticks poison-pill the page.
        for _ in 0..3 {
            run_citation_backfill_tick(&db, &llm, &prompts)
                .await
                .unwrap();
        }

        let page = db.get_page(&page_id).await.unwrap().unwrap();
        assert_eq!(page.content, BACKFILL_BODY, "prose must never be rewritten");
        assert!(page.citations.is_empty());
        assert!(
            !db.get_pages_missing_citations(10)
                .await
                .unwrap()
                .contains(&page_id),
            "citations should be '[]' (gave up), not NULL"
        );
        let changelog = db.get_page_changelog(&page_id).await.unwrap();
        assert!(
            changelog.contains("citation backfill gave up"),
            "changelog: {changelog}"
        );
    }
}
