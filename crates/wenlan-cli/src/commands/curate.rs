// SPDX-License-Identifier: Apache-2.0
//! `wenlan curate` — walk pending revisions (conflicts / merges) from the daemon.
//!
//! CLI-first replacement for the `/curate` skill's deferred-MCP round-trips: the
//! skill Bashes `wenlan --format json curate` to list, then
//! `wenlan curate accept|dismiss <revision_source_id>` to act, so it pays no
//! ToolSearch hop. Scope is revisions only (the attention-gated surface that
//! actually needs a human); pending captures stay on the rare opt-in MCP path.

use anyhow::Result;
use serde::Serialize;
use wenlan_types::responses::PendingRevisionItem;

use crate::client::WenlanClient;
use crate::output::{print_json, ResolvedFormat};

/// One logical revision: all per-chunk rows sharing a `revision_source_id`, joined.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GroupedRevision {
    /// The memory this revision would replace (shown for context).
    pub target_source_id: String,
    /// The staged revision row id — the accept/dismiss action key, so the
    /// *named* revision is acted on even when several compete for one target.
    pub revision_source_id: String,
    /// Full revision text, chunks joined in order.
    pub content: String,
    pub source_agent: Option<String>,
    /// Current text of the memory this revision would replace, fetched via
    /// `/api/memory/{id}/detail`. `None` if the original could not be fetched.
    pub original: Option<String>,
    /// A labeled `OLD:` / `NEW:` preview of original -> revision, for the card to
    /// drop straight into an `AskUserQuestion` option preview. `None` when the
    /// original is unavailable (the card falls back to showing `content`).
    pub diff: Option<String>,
    /// Doc file source_id that grounds a doc-grounded revision (L3); None for
    /// other revision producers.
    pub grounded_in: Option<String>,
}

#[derive(clap::Subcommand, Debug)]
pub enum CurateAction {
    /// List pending revisions (the default when no action is given).
    Revisions {
        /// Max revisions to fetch.
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
    },
    /// Accept a revision: replace the original memory with the revised text.
    Accept {
        /// `revision_source_id` of the revision (from the list output).
        revision_source_id: String,
    },
    /// Dismiss a revision: drop it, keep the original memory.
    Dismiss {
        /// `revision_source_id` of the revision (from the list output).
        revision_source_id: String,
    },
}

const DEFAULT_LIMIT: usize = 20;

pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    action: Option<CurateAction>,
) -> Result<()> {
    match action.unwrap_or(CurateAction::Revisions {
        limit: DEFAULT_LIMIT,
    }) {
        CurateAction::Revisions { limit } => list_revisions(client, format, quiet, limit).await,
        CurateAction::Accept { revision_source_id } => {
            let resp = client.accept_revision(&revision_source_id).await?;
            if quiet {
                return Ok(());
            }
            match format {
                ResolvedFormat::Json => print_json(&resp)?,
                ResolvedFormat::Table => println!(
                    "{} revision for {}",
                    if resp.wrote { "accepted" } else { "no-op" },
                    resp.target_source_id
                ),
            }
            Ok(())
        }
        CurateAction::Dismiss { revision_source_id } => {
            let resp = client.dismiss_revision(&revision_source_id).await?;
            if quiet {
                return Ok(());
            }
            match format {
                ResolvedFormat::Json => print_json(&resp)?,
                ResolvedFormat::Table => println!(
                    "{} revision for {}",
                    if resp.wrote { "dismissed" } else { "no-op" },
                    resp.target_source_id
                ),
            }
            Ok(())
        }
    }
}

async fn list_revisions(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    limit: usize,
) -> Result<()> {
    let items = client.list_pending_revisions(limit).await?;
    let mut groups = group_revisions(items);
    enrich_with_diffs(client, &mut groups).await;
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&groups)?,
        ResolvedFormat::Table => print_table(&groups),
    }
    Ok(())
}

/// Fetch each revision's ORIGINAL memory (via `/api/memory/{id}/detail`) and
/// build an OLD/NEW preview so the card shows what it would replace. One extra
/// local HTTP round-trip per revision (~ms); a fetch miss leaves `original` /
/// `diff` as `None` and the card falls back to showing the revised text.
async fn enrich_with_diffs(client: &WenlanClient, groups: &mut [GroupedRevision]) {
    for g in groups.iter_mut() {
        if let Ok(detail) = client.get_memory_detail(&g.target_source_id).await {
            if let Some(mem) = detail.memory {
                g.diff = Some(revision_preview(&mem.content, &g.content));
                g.original = Some(mem.content);
            }
        }
    }
}

fn print_table(groups: &[GroupedRevision]) {
    if groups.is_empty() {
        println!("(no pending revisions — nothing needs you)");
        return;
    }
    println!(
        "{} pending revision{}",
        groups.len(),
        if groups.len() == 1 { "" } else { "s" }
    );
    for g in groups {
        // Prefer the OLD/NEW preview (shows what it replaces) over the raw revision text.
        let body = g.diff.as_deref().unwrap_or(g.content.as_str());
        let snippet = if body.chars().count() > 160 {
            format!("{}...", body.chars().take(157).collect::<String>())
        } else {
            body.to_string()
        };
        let agent = g.source_agent.as_deref().unwrap_or("daemon");
        println!("  {}  ·  ({})  {}", g.revision_source_id, agent, snippet);
        if let Some(g) = &g.grounded_in {
            // "{source_id}::{path}" -> show the file path; fall back to raw.
            let path = g.split_once("::").map(|(_, p)| p).unwrap_or(g);
            println!("      grounded in {path}");
        }
    }
    println!("accept/dismiss: wenlan curate accept|dismiss <revision_source_id>");
}

/// Group per-chunk `PendingRevisionItem` rows into one logical revision each.
///
/// `list_pending_revisions` returns one row per chunk; a long revision spans
/// several rows sharing the same `revision_source_id`. Join their content in the
/// order received (the daemon sorts by `last_modified DESC`) so each card is ONE
/// logical revision, not a mid-sentence fragment.
fn group_revisions(items: Vec<PendingRevisionItem>) -> Vec<GroupedRevision> {
    use std::collections::HashMap;
    let mut order: Vec<String> = Vec::new();
    let mut by_rev: HashMap<String, GroupedRevision> = HashMap::new();
    for it in items {
        let piece = it.revision_content.trim();
        match by_rev.get_mut(&it.revision_source_id) {
            Some(g) => {
                if !piece.is_empty() {
                    if !g.content.is_empty() {
                        g.content.push(' ');
                    }
                    g.content.push_str(piece);
                }
            }
            None => {
                order.push(it.revision_source_id.clone());
                by_rev.insert(
                    it.revision_source_id.clone(),
                    GroupedRevision {
                        target_source_id: it.target_source_id,
                        revision_source_id: it.revision_source_id,
                        content: piece.to_string(),
                        source_agent: it.source_agent,
                        original: None,
                        diff: None,
                        grounded_in: it.grounded_in,
                    },
                );
            }
        }
    }
    order
        .into_iter()
        .filter_map(|k| by_rev.remove(&k))
        .collect()
}

/// Max chars per OLD/NEW block. The card preview box collapses to ~4 lines, and a
/// long OLD block wraps past the fold and hides NEW's label — so clip each block
/// short enough that both labels stay visible. The full text lives in the memory.
const BLOCK_CHARS: usize = 100;

/// Card-ready preview of `original` -> `revision`: two labeled blocks, `OLD:` then
/// a blank line then `NEW:`. A word-level inline diff was tried first but read as
/// unreadable confetti for these revisions (the dual-pool dedup matches a *new*
/// memory to an unrelated old one, so almost every word differs). Plain labeled
/// old/new, blank-line separated so OLD's wrapped tail can't blur into NEW, is what
/// a human can actually scan. Bounded so the model echoes little and the picker
/// stays fast.
fn revision_preview(original: &str, revision: &str) -> String {
    format!(
        "OLD: {}\n\nNEW: {}",
        clip(original, BLOCK_CHARS),
        clip(revision, BLOCK_CHARS)
    )
}

/// Flatten whitespace to single spaces and clip to `max` chars (char-safe).
fn clip(s: &str, max: usize) -> String {
    let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        format!("{}…", flat.chars().take(max).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(target: &str, rev: &str, content: &str) -> PendingRevisionItem {
        PendingRevisionItem {
            target_source_id: target.to_string(),
            revision_source_id: rev.to_string(),
            revision_content: content.to_string(),
            source_agent: None,
            last_modified: 0,
            target_kind: wenlan_types::responses::RevisionTargetKind::Memory,
            grounded_in: None,
        }
    }

    #[test]
    fn empty_input_yields_no_groups() {
        assert!(group_revisions(vec![]).is_empty());
    }

    #[test]
    fn single_chunk_is_one_group_trimmed() {
        let groups = group_revisions(vec![item("mem_a", "rev_1", "  hello world  ")]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].target_source_id, "mem_a");
        assert_eq!(groups[0].content, "hello world");
    }

    #[test]
    fn multi_chunk_same_revision_joins_in_order() {
        let groups = group_revisions(vec![
            item("mem_a", "rev_1", "first part"),
            item("mem_a", "rev_1", "second part"),
        ]);
        assert_eq!(groups.len(), 1, "two chunks of one revision = one group");
        assert_eq!(groups[0].content, "first part second part");
    }

    #[test]
    fn distinct_revisions_stay_separate_in_first_seen_order() {
        let groups = group_revisions(vec![
            item("mem_b", "rev_2", "bee"),
            item("mem_a", "rev_1", "ay"),
        ]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].revision_source_id, "rev_2");
        assert_eq!(groups[1].revision_source_id, "rev_1");
    }

    #[test]
    fn preview_labels_old_and_new_no_word_diff_markers() {
        let p = revision_preview("the original text here", "the revised text now");
        assert!(
            p.contains("OLD: the original text here"),
            "old labeled: {p}"
        );
        assert!(p.contains("NEW: the revised text now"), "new labeled: {p}");
        assert!(
            !p.contains("[-") && !p.contains("{+"),
            "no word-diff confetti: {p}"
        );
    }

    #[test]
    fn preview_separates_old_and_new_with_blank_line() {
        // The card preview box collapses to a few lines; a blank line between the
        // blocks keeps OLD's wrapped tail from blurring into NEW (the screenshot
        // "old and new is hard to see / not good formatted" complaint).
        let p = revision_preview("alpha", "beta");
        let lines: Vec<&str> = p.lines().collect();
        let old_idx = lines.iter().position(|l| l.starts_with("OLD:")).unwrap();
        let new_idx = lines.iter().position(|l| l.starts_with("NEW:")).unwrap();
        assert!(new_idx > old_idx + 1, "blank line between blocks: {p:?}");
        assert_eq!(lines[old_idx + 1], "", "separator line is blank: {p:?}");
    }

    #[test]
    fn preview_flattens_whitespace_and_clips_long_text() {
        let long_original = "word ".repeat(80); // 400 chars, whitespace-heavy
        let p = revision_preview(&long_original, "a short revision");
        assert!(p.contains("NEW: a short revision"), "new shown: {p}");
        assert!(p.contains('…'), "long original clipped: {p}");
        let old_line = p.lines().find(|l| l.starts_with("OLD:")).unwrap();
        assert!(
            !old_line.contains("  "),
            "whitespace flattened (no double spaces): {old_line}"
        );
        assert!(
            old_line.chars().count() <= "OLD: ".len() + BLOCK_CHARS + 1,
            "old line bounded: {old_line}"
        );
    }
}
