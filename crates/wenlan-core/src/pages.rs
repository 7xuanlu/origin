// SPDX-License-Identifier: Apache-2.0
//! Knowledge compilation: pages are synthesized wiki entries distilled from memory clusters.

// Re-export the wire type from wenlan-types so existing consumers keep working.
pub use wenlan_types::pages::Page;

/// Result of replacing one complete Page draft snapshot.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum PageDraftUpdateOutcome {
    Updated(Page),
    VersionConflict { current_version: i64 },
}

/// Result of discarding a Page draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageDraftDeleteOutcome {
    Deleted,
    VersionConflict { current_version: i64 },
}

/// Generate a new unique page ID.
///
/// Replaces the former `Page::new_id()` associated function now that `Page`
/// is defined in `wenlan-types` and `impl` blocks on foreign types are
/// disallowed.
pub fn new_page_id() -> String {
    format!("page_{}", uuid::Uuid::new_v4())
}

/// Build the text embedded for page-level matching.
///
/// This is shared by page writes and PageWrite dedup so the stored page vector
/// and the candidate vector cannot drift apart.
pub(crate) fn page_embedding_text(title: &str, summary: Option<&str>, content: &str) -> String {
    const PAGE_EMBED_CONTENT_CAP: usize = 1500;
    let capped_body: String = content.chars().take(PAGE_EMBED_CONTENT_CAP).collect();
    match summary {
        Some(summary) => format!("{title} {summary} {capped_body}"),
        None => format!("{title} {capped_body}"),
    }
}

/// Maps a source memory's `memory_type` to the read-trust tier it sits behind.
pub fn trust_tier_for_memory_type(memory_type: Option<&str>) -> u8 {
    match memory_type {
        Some("identity") | Some("preference") => 1,
        Some("decision") | Some("correction") => 2,
        _ => 3,
    }
}

/// Filter pages by source overlap with search results.
///
/// A page is contextually relevant if the memories it was compiled from
/// overlap with the memories that search_memory returned for this query.
/// This is the strongest relevance signal: it answers "is this page about
/// the thing I'm searching for?" rather than relying on embedding similarity
/// (which we proved doesn't discriminate between good and garbage pages).
///
/// `min_overlap`: minimum number of search result source_ids that must appear
/// in the page's `source_memory_ids`. Recommended: 2 (filters noise while
/// keeping pages with genuine topical overlap).
pub fn filter_pages_by_source_overlap(
    pages: &[Page],
    search_result_source_ids: &std::collections::HashSet<String>,
    min_overlap: usize,
) -> Vec<Page> {
    pages
        .iter()
        .filter(|c| {
            let overlap = c
                .source_memory_ids
                .iter()
                .filter(|sid| search_result_source_ids.contains(sid.as_str()))
                .count();
            overlap >= min_overlap
        })
        .cloned()
        .collect()
}

/// Select trusted pages for the assembled context block.
///
/// `review_status == "confirmed"` means the page passed the distillation
/// faithfulness gate, not that it was manually reviewed.
pub fn select_pages_for_context(
    pages: &[Page],
    search_result_source_ids: &std::collections::HashSet<String>,
    cap: usize,
) -> Vec<Page> {
    let mut selected: Vec<Page> = pages
        .iter()
        .filter(|page| page.review_status == "confirmed")
        .cloned()
        .collect();

    selected.sort_by(|left, right| {
        if (left.relevance_score - right.relevance_score).abs() <= 1e-6 {
            let left_overlap = left
                .source_memory_ids
                .iter()
                .filter(|sid| search_result_source_ids.contains(sid.as_str()))
                .count();
            let right_overlap = right
                .source_memory_ids
                .iter()
                .filter(|sid| search_result_source_ids.contains(sid.as_str()))
                .count();
            return right_overlap.cmp(&left_overlap);
        }

        right
            .relevance_score
            .partial_cmp(&left.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    selected.truncate(cap);
    selected
}

/// Hard workspace scope for the additive Page path. Source-memory overlap is
/// a relevance signal only and can never override the Page's workspace.
pub fn scope_filter_pages(
    pages: Vec<Page>,
    caller_space: Option<&str>,
    _memory_source_ids: &std::collections::HashSet<String>,
) -> Vec<Page> {
    let Some(space) = caller_space else {
        return pages;
    };
    pages
        .into_iter()
        .filter(|page| page.workspace.as_deref() == Some(space))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_page(id: &str, source_ids: &[&str]) -> Page {
        make_page_with(id, source_ids, 0.5, "confirmed")
    }

    fn make_page_with(
        id: &str,
        source_ids: &[&str],
        relevance_score: f32,
        review_status: &str,
    ) -> Page {
        Page {
            id: id.to_string(),
            title: id.to_string(),
            summary: None,
            content: String::new(),
            entity_id: None,
            space: None,
            source_memory_ids: source_ids.iter().map(|s| s.to_string()).collect(),
            version: 1,
            status: "active".to_string(),
            created_at: String::new(),
            last_compiled: String::new(),
            last_modified: String::new(),
            sources_updated_count: 0,
            stale_reason: None,
            pending_rebuild: None,
            user_edited: false,
            relevance_score,
            last_edited_by: None,
            last_edited_at: None,
            last_delta_summary: None,
            changelog: None,
            creation_kind: "distilled".to_string(),
            review_status: review_status.to_string(),
            workspace: None,
            citations: Vec::new(),
            kind: "concept".to_string(),
        }
    }

    #[test]
    fn tier_map_identity_and_preference_are_tier1() {
        assert_eq!(trust_tier_for_memory_type(Some("identity")), 1);
        assert_eq!(trust_tier_for_memory_type(Some("preference")), 1);
    }

    #[test]
    fn tier_map_decision_correction_tier2_else_tier3() {
        assert_eq!(trust_tier_for_memory_type(Some("decision")), 2);
        assert_eq!(trust_tier_for_memory_type(Some("correction")), 2);
        assert_eq!(trust_tier_for_memory_type(Some("fact")), 3);
        assert_eq!(trust_tier_for_memory_type(None), 3);
    }

    #[test]
    fn test_overlap_keeps_matching_concept() {
        let pages = vec![make_page("c1", &["m1", "m2", "m3"])];
        let search_ids: HashSet<String> = ["m1", "m2"].iter().map(|s| s.to_string()).collect();
        let kept = filter_pages_by_source_overlap(&pages, &search_ids, 2);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn test_overlap_filters_low_overlap() {
        let pages = vec![make_page("c1", &["m1", "m2", "m3"])];
        let search_ids: HashSet<String> = ["m1", "m99"].iter().map(|s| s.to_string()).collect();
        let kept = filter_pages_by_source_overlap(&pages, &search_ids, 2);
        assert_eq!(kept.len(), 0); // only 1 overlap, need 2
    }

    #[test]
    fn test_overlap_empty_concept_sources() {
        let pages = vec![make_page("c1", &[])];
        let search_ids: HashSet<String> = ["m1"].iter().map(|s| s.to_string()).collect();
        let kept = filter_pages_by_source_overlap(&pages, &search_ids, 1);
        assert_eq!(kept.len(), 0);
    }

    #[test]
    fn test_overlap_empty_search_results() {
        let pages = vec![make_page("c1", &["m1", "m2"])];
        let search_ids: HashSet<String> = HashSet::new();
        let kept = filter_pages_by_source_overlap(&pages, &search_ids, 1);
        assert_eq!(kept.len(), 0);
    }

    #[test]
    fn test_overlap_zero_threshold_keeps_all() {
        let pages = vec![make_page("c1", &["m1"]), make_page("c2", &["m99"])];
        let search_ids: HashSet<String> = ["m1"].iter().map(|s| s.to_string()).collect();
        let kept = filter_pages_by_source_overlap(&pages, &search_ids, 0);
        assert_eq!(kept.len(), 2); // min_overlap=0 keeps everything
    }

    #[test]
    fn page_embedding_text_concats_title_summary_and_capped_body() {
        // With a summary the slots join as "{title} {summary} {body}".
        assert_eq!(
            page_embedding_text("Title", Some("Summary"), "Body"),
            "Title Summary Body"
        );
        // Without a summary the summary slot is dropped entirely.
        assert_eq!(page_embedding_text("Title", None, "Body"), "Title Body");
        // The body is capped so a long page can't blow the embedder window;
        // the cap counts characters, not bytes.
        let long_body = "x".repeat(2000);
        let text = page_embedding_text("T", None, &long_body);
        let body_part = text.strip_prefix("T ").expect("title prefix present");
        assert_eq!(
            body_part.chars().count(),
            1500,
            "body must be capped at 1500 chars"
        );
    }

    #[test]
    fn page_embedding_text_is_the_single_source_of_truth() {
        // The write side (`insert_page_with_kind`) and the read/dedup side
        // (`create_page_impl`) must build the page embedding vector from the
        // SAME helper. If either re-inlines the formula or the cap, the stored
        // page vector and the dedup candidate vector silently drift and
        // near-duplicate pages stop matching — the exact bug this dedup path
        // exists to prevent. Guard it at compile-inlined source level.
        let db_source = include_str!("db.rs");
        let post_write_source = include_str!("post_write.rs");

        // Both call sites route through the shared helper.
        assert!(
            db_source.contains("page_embedding_text("),
            "insert_page_with_kind must call the shared page_embedding_text helper"
        );
        assert!(
            post_write_source.contains("page_embedding_text("),
            "PageWrite dedup must call the shared page_embedding_text helper"
        );
        // Neither call site redefines the helper or its cap locally.
        assert!(
            !post_write_source.contains("fn page_embedding_text"),
            "PageWrite must not carry its own page_embedding_text definition"
        );
        assert!(
            !db_source.contains("PAGE_EMBED_CONTENT_CAP"),
            "insert_page_with_kind must not carry a private page embedding cap"
        );
        assert!(
            !post_write_source.contains("PAGE_EMBED_CONTENT_CAP"),
            "PageWrite dedup must not carry a private page embedding cap"
        );
    }

    #[test]
    fn test_overlap_mixed_keeps_and_filters() {
        let pages = vec![
            make_page("good", &["m1", "m2", "m3", "m4", "m5"]),
            make_page("noise", &["m90", "m91", "m92"]),
            make_page("edge", &["m1", "m90"]),
        ];
        let search_ids: HashSet<String> =
            ["m1", "m2", "m3"].iter().map(|s| s.to_string()).collect();
        let kept = filter_pages_by_source_overlap(&pages, &search_ids, 2);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].id, "good"); // 3 overlap
                                        // "noise" has 0 overlap, "edge" has 1 overlap — both filtered at min_overlap=2
    }

    #[test]
    fn search_pages_scope_does_not_allow_source_overlap_to_override_workspace() {
        let mut personal = make_page("personal", &["work-memory"]);
        personal.workspace = Some("personal".to_string());
        let work_sources = HashSet::from(["work-memory".to_string()]);

        let kept = scope_filter_pages(vec![personal], Some("work"), &work_sources);

        assert!(
            kept.is_empty(),
            "a source-memory overlap must not expose a Page from another workspace"
        );
    }

    #[test]
    fn select_keeps_zero_overlap_page_when_score_high() {
        let pages = vec![
            make_page_with("low_overlap", &["m1"], 0.25, "confirmed"),
            make_page_with("high_zero_overlap", &["m90"], 0.95, "confirmed"),
        ];
        let search_ids: HashSet<String> = ["m1"].iter().map(|s| s.to_string()).collect();

        let selected = select_pages_for_context(&pages, &search_ids, 2);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].id, "high_zero_overlap");
    }

    #[test]
    fn select_drops_unconfirmed() {
        let pages = vec![
            make_page_with("unconfirmed", &["m1"], 0.99, "unconfirmed"),
            make_page_with("confirmed", &["m2"], 0.10, "confirmed"),
        ];
        let search_ids: HashSet<String> = ["m1", "m2"].iter().map(|s| s.to_string()).collect();

        let selected = select_pages_for_context(&pages, &search_ids, 3);

        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "confirmed");
    }

    #[test]
    fn select_ranks_by_score_desc() {
        let pages = vec![
            make_page_with("low_with_overlap", &["m1"], 0.20, "confirmed"),
            make_page_with("high_without_overlap", &["m90"], 0.80, "confirmed"),
            make_page_with("mid_with_overlap", &["m2"], 0.50, "confirmed"),
        ];
        let search_ids: HashSet<String> = ["m1", "m2"].iter().map(|s| s.to_string()).collect();

        let selected = select_pages_for_context(&pages, &search_ids, 3);

        let ids: Vec<_> = selected.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(
            ids,
            vec![
                "high_without_overlap",
                "mid_with_overlap",
                "low_with_overlap"
            ]
        );
    }

    #[test]
    fn select_overlap_breaks_ties() {
        let pages = vec![
            make_page_with("zero_overlap", &["m90"], 0.75, "confirmed"),
            make_page_with("with_overlap", &["m1"], 0.75, "confirmed"),
        ];
        let search_ids: HashSet<String> = ["m1"].iter().map(|s| s.to_string()).collect();

        let selected = select_pages_for_context(&pages, &search_ids, 2);

        assert_eq!(selected[0].id, "with_overlap");
        assert_eq!(selected[1].id, "zero_overlap");
    }

    #[test]
    fn select_respects_cap() {
        let pages = vec![
            make_page_with("one", &["m1"], 0.90, "confirmed"),
            make_page_with("two", &["m2"], 0.80, "confirmed"),
            make_page_with("three", &["m3"], 0.70, "confirmed"),
        ];
        let search_ids: HashSet<String> =
            ["m1", "m2", "m3"].iter().map(|s| s.to_string()).collect();

        let selected = select_pages_for_context(&pages, &search_ids, 2);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].id, "one");
        assert_eq!(selected[1].id, "two");
    }

    #[test]
    fn scope_requires_matching_workspace_even_with_source_overlap() {
        let mut p_other = make_page("p_other", &["x1"]); // workspace None, sources disjoint
        p_other.workspace = Some("personal".to_string());
        let mut p_match = make_page("p_match", &["x2"]);
        p_match.workspace = Some("work".to_string());
        let p_overlap = make_page("p_overlap", &["m1"]); // workspace None but source in result set
        let ids: HashSet<String> = ["m1"].iter().map(|s| s.to_string()).collect();
        let kept = scope_filter_pages(vec![p_other, p_match, p_overlap], Some("work"), &ids);
        let kept_ids: Vec<_> = kept.iter().map(|p| p.id.as_str()).collect();
        assert!(kept_ids.contains(&"p_match")); // workspace == caller space
        assert!(!kept_ids.contains(&"p_overlap")); // source overlap cannot override workspace
        assert!(!kept_ids.contains(&"p_other")); // cross-space, no overlap → dropped
    }

    #[test]
    fn scope_noop_when_no_space_filter() {
        let ids: HashSet<String> = HashSet::new();
        let pages = vec![make_page("a", &["z"])];
        assert_eq!(scope_filter_pages(pages, None, &ids).len(), 1); // unscoped recall keeps all
    }
}
