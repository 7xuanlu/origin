// SPDX-License-Identifier: Apache-2.0
//! Wikilink graph for distilled pages.
//!
//! Pages are written with `[[Label]]` link syntax. The distill prompt emits
//! them, users write them by hand. The daemon's job is to:
//!   1. Pull `[[Label]]` occurrences out of a page body.
//!   2. Resolve each label against current page titles (case-insensitive,
//!      trimmed). No fuzzy matching in v1 — the LLM's job is to use exact
//!      titles, and the orphan-by-label feed surfaces drift.
//!   3. Persist resolved/unresolved pairs into the `page_links` table so the
//!      refinery can re-resolve later (when a target page is created) and
//!      emergence can mine repeated orphan labels as topic candidates.
//!
//! Parsing delegates to `sources::obsidian::extract_wikilinks` — that module
//! already handles code-block exclusion, aliased links, and heading anchors
//! correctly. Here we strip the structural details (heading / display /
//! embed flag) down to the target label, since the link's identity for
//! graph purposes is the target, not the display text.

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::sources::obsidian;
use std::collections::HashSet;

/// A wikilink reference stored in or read from the `page_links` table.
/// `target_page_id` is `None` when the resolver couldn't find a matching
/// page title — those rows persist with NULL target so the refinery's
/// orphan-resolve phase can pick them up later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wikilink {
    pub label: String,
    pub target_page_id: Option<String>,
}

/// Pull every `[[Label]]` reference out of `content`, returning labels in
/// document order, deduplicated case-insensitively. Skips embeds (`![[...]]`,
/// reserved for transclusion) and links inside fenced code blocks. The
/// obsidian regex naturally rejects targets containing `]`, `|`, or `#`
/// so malformed input like `[[[[foo]]]]` or `[[foo]bar]]` produces nothing.
pub fn extract_wikilinks(content: &str) -> Vec<String> {
    let parsed = obsidian::extract_wikilinks(content);
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for link in parsed {
        if link.is_embed {
            continue;
        }
        let label = link.target.trim().to_string();
        if label.is_empty() {
            continue;
        }
        // Obsidian's regex accepts `[` inside the target since `[^\]|#]+`
        // doesn't exclude it — so `[[[[foo]]]]` captures `[[foo`. Reject
        // labels that carry stray brackets; they'd poison page_links and
        // the orphan-by-count feed.
        if label.contains('[') || label.contains(']') {
            continue;
        }
        let key = label.to_lowercase();
        if seen.insert(key) {
            out.push(label);
        }
    }
    out
}

/// Resolve each extracted label against the current `pages` table.
/// Case-insensitive exact match on title, restricted to `status='active'`.
/// Unresolved labels keep `target_page_id = None` so the caller can persist
/// them as orphan rows.
pub async fn resolve_against_pages(
    db: &MemoryDB,
    labels: &[String],
    scope: Option<&str>,
) -> Result<Vec<Wikilink>, WenlanError> {
    if labels.is_empty() {
        return Ok(Vec::new());
    }
    // One scope scan for the whole label set — the fold and the ambiguity
    // contract live in `folded_title_owners_scoped`.
    let owners = db.folded_title_owners_scoped(scope).await?;
    Ok(labels
        .iter()
        .map(|label| Wikilink {
            label: label.clone(),
            target_page_id: owners
                .get(&MemoryDB::page_title_key(label))
                .cloned()
                .flatten(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_single_link() {
        let v = extract_wikilinks("hello [[Rust Ownership]] world");
        assert_eq!(v, vec!["Rust Ownership"]);
    }

    #[test]
    fn extract_multiple_links_preserve_order() {
        let v = extract_wikilinks("[[Alpha]] then [[Beta]] then [[Gamma]]");
        assert_eq!(v, vec!["Alpha", "Beta", "Gamma"]);
    }

    #[test]
    fn extract_dedups_case_insensitively() {
        let v = extract_wikilinks("[[Wenlan]] and [[wenlan]] and [[WENLAN]]");
        // Keep first-seen casing.
        assert_eq!(v, vec!["Wenlan"]);
    }

    #[test]
    fn extract_handles_aliased_links() {
        // Obsidian regex captures `target` as group 2; display text is
        // stripped here.
        let v = extract_wikilinks("see [[Rust Ownership|borrowing rules]] for more");
        assert_eq!(v, vec!["Rust Ownership"]);
    }

    #[test]
    fn extract_strips_heading_anchor() {
        // `[[Page#Section]]` reduces to the target page.
        let v = extract_wikilinks("see [[Rust Ownership#borrowing]] below");
        assert_eq!(v, vec!["Rust Ownership"]);
    }

    #[test]
    fn extract_skips_embeds() {
        // Embeds (`![[...]]`) are for transclusion — out of scope for the
        // graph. Plain `[[...]]` is the citation link.
        let v = extract_wikilinks("![[Diagram]] then [[Real Link]]");
        assert_eq!(v, vec!["Real Link"]);
    }

    #[test]
    fn extract_skips_code_block_wikilinks() {
        let v = extract_wikilinks("real [[A]]\n\n```\n[[NotALink]]\n```\n\nmore [[B]]");
        assert_eq!(v, vec!["A", "B"]);
    }

    #[test]
    fn extract_handles_utf8_payload() {
        let v = extract_wikilinks("café [[Naïve Bayes]] résumé");
        assert_eq!(v, vec!["Naïve Bayes"]);
    }

    #[test]
    fn extract_returns_empty_for_no_links() {
        let v = extract_wikilinks("plain prose with [no brackets] only");
        assert!(v.is_empty());
    }

    #[test]
    fn extract_rejects_nested_brackets() {
        // `[[[[foo]]]]` — obsidian regex captures target `[[foo` (it allows
        // `[` inside the target group). The bracket post-filter drops it.
        let v = extract_wikilinks("noise [[[[foo]]]] more");
        assert!(v.is_empty());
    }

    #[test]
    fn extract_rejects_stray_closing_bracket() {
        // `[[foo]bar]]` — obsidian regex fails to find a complete
        // `[[...]]` since target stops at the inner `]` but `\]\]` isn't
        // there immediately after. Nothing captured.
        let v = extract_wikilinks("noise [[foo]bar]] more");
        assert!(v.is_empty());
    }

    #[test]
    fn extract_keeps_well_formed_after_garbage() {
        // Bracket post-filter drops the captured `[[foo`; real link survives.
        let v = extract_wikilinks("[[[[foo]]]] and [[Real Link]] keep going");
        assert_eq!(v, vec!["Real Link"]);
    }
}
