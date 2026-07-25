// SPDX-License-Identifier: Apache-2.0
//! Wire types for compiled knowledge pages.

use serde::{Deserialize, Serialize};

pub const PAGE_PENDING_REBUILD_MESSAGE: &str = "evidence updated, prose rebuild pending";

/// A compiled knowledge page — structured, cross-referenced, backed by source memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub id: String,
    pub title: String,
    pub summary: Option<String>,
    pub content: String,
    pub entity_id: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    /// Kept for dual-write transition; prefer concept_sources join table for new reads.
    pub source_memory_ids: Vec<String>,
    pub version: i64,
    pub status: String,
    pub created_at: String,
    pub last_compiled: String,
    pub last_modified: String,
    /// How many source memories were updated since last distillation.
    pub sources_updated_count: i64,
    /// Why this page is stale: "source_updated" | "source_conflict" | None.
    pub stale_reason: Option<String>,
    /// Human-facing compile state for stale pages whose evidence has already
    /// changed but whose prose is waiting for the next compile lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_rebuild: Option<String>,
    /// True if a human has edited this page's content directly.
    pub user_edited: bool,
    /// Relevance score from search (0.0-1.0). Only populated by `search_pages`;
    /// zero for persisted/non-search contexts.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_edited_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_edited_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_delta_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changelog: Option<String>,
    /// Scope axis (P3). Distinct from `space` (category column). Set at creation
    /// time from `CreateConceptRequest.workspace` or the `X-Origin-Space` header.
    /// NULL = no workspace constraint. Enforced only by the scoped-recall page gate
    /// on the cross-rerank search path; direct page lookups (search_pages MCP tool,
    /// exports) do not filter by workspace.
    #[serde(default)]
    pub workspace: Option<String>,
    /// Routing metadata: which mechanism created this page.
    /// One of: "distilled" | "authored" | "research" | "imported".
    /// NOT a trust signal (see `review_status` for that).
    #[serde(default = "default_creation_kind")]
    pub creation_kind: String,
    /// Trust boundary: whether this page has been confirmed as accurate.
    /// One of: "unconfirmed" | "confirmed".
    /// Every newly-created page is born "unconfirmed" regardless of
    /// `creation_kind` (spec 3.1/5.2). `default_review_status()` below is a
    /// wire/deserialization fallback for rows predating this column (backfilled
    /// "confirmed" by migration 62), not the birth value for new pages.
    #[serde(default = "default_review_status")]
    pub review_status: String,
    /// Per-occurrence [N] citation records for this page's body (spec §3).
    /// Empty for pages never citation-distilled or citation-backfilled.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<PageCitation>,
    /// Unified page-kind discriminator (migration 89). One of:
    /// "entity" | "concept" | "source" | "overview" | "authored".
    /// `kind = "entity"` marks the M3 dual-write shadow pages that mirror
    /// `entities` rows -- write-only in PR-1 (spec's fail-closed visibility
    /// fence excludes them at `select_visible_pages` and every page-search/
    /// list surface that bypasses it), never surfaced to a reader yet.
    ///
    /// `skip_serializing` freezes the wire shape (spec M3 D4): `kind` is an
    /// internal discriminator the daemon uses for the visibility fence, but it
    /// is deliberately absent from every serialized `Page` so the app-visible
    /// contract is byte-identical to pre-M3. `default = "default_page_kind"`
    /// stays so inbound JSON (which never carries `kind`) still deserializes.
    #[serde(default = "default_page_kind", skip_serializing)]
    pub kind: String,
}

fn default_creation_kind() -> String {
    "distilled".to_string()
}

fn default_page_kind() -> String {
    "concept".to_string()
}

fn default_review_status() -> String {
    "confirmed".to_string()
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

pub fn pending_rebuild_for_stale_reason(stale_reason: Option<&str>) -> Option<String> {
    match stale_reason {
        Some("source_updated") => Some(PAGE_PENDING_REBUILD_MESSAGE.to_string()),
        _ => None,
    }
}

/// Typed provenance link for a page (P2 successor to `PageSource`).
/// Backed by the `page_evidence` SQL table (migration 60).
/// Additive — `PageSource` / `page_sources` are NOT removed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PageEvidence {
    pub page_id: String,
    pub source_kind: String, // memory | external_url | external_file | authored
    pub locator: Option<String>,
    pub title: Option<String>,
    pub linked_at: i64,
    pub link_reason: Option<String>,
}

/// One [N] marker occurrence in a page body, in body order (per-occurrence,
/// NOT per marker number: the same [1] reused in two sentences gets two
/// records with independent statuses). Renderer join rule: the k-th [N]
/// instance scanning the body left-to-right joins the record with
/// occurrence == k.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageCitation {
    pub occurrence: u32,
    pub marker: u32,
    pub source_kind: String,
    pub locator: String,
    /// Claim-vs-this-source-alone content-token overlap (0..=1) at the
    /// recorded `scope` granularity, audit only.
    pub score: f64,
    /// "verified" | "unverified" — decided claim-level vs the UNION of the
    /// claim's cited sources (spec §5).
    pub status: String,
    /// Granularity that decided the status: "sentence" (the marker's own
    /// sentence cleared the floor) or "paragraph" (sentence failed; the
    /// enclosing paragraph cleared it — small on-device models cite at
    /// paragraph ends, so elaboration sentences would otherwise badge true
    /// claims). The recorded scope keeps the weaker guarantee visible to
    /// renderers. Unverified records carry "paragraph" (both tiers tried).
    #[serde(default = "default_citation_scope")]
    pub scope: String,
}

fn default_citation_scope() -> String {
    "sentence".to_string()
}

#[cfg(test)]
mod citation_tests {
    use super::*;

    #[test]
    fn old_page_json_deserializes_with_empty_citations() {
        let json = r#"{"id":"p1","title":"T","content":"body","source_memory_ids":[],
            "version":1,"status":"active","created_at":"x","last_compiled":"x",
            "last_modified":"x","sources_updated_count":0,"user_edited":false}"#;
        let p: Page = serde_json::from_str(json).unwrap();
        assert!(p.citations.is_empty());
    }

    #[test]
    fn citation_roundtrip() {
        let c = PageCitation {
            occurrence: 2,
            marker: 1,
            source_kind: "memory".into(),
            locator: "mem_a".into(),
            score: 0.31,
            status: "unverified".into(),
            scope: "paragraph".into(),
        };
        let s = serde_json::to_string(&vec![c]).unwrap();
        let back: Vec<PageCitation> = serde_json::from_str(&s).unwrap();
        assert_eq!(back[0].occurrence, 2);
        assert_eq!(back[0].status, "unverified");
        assert_eq!(back[0].scope, "paragraph");
        // pre-scope records (old rows) default to "sentence"
        let legacy: PageCitation = serde_json::from_str(
            r#"{"occurrence":1,"marker":1,"source_kind":"memory","locator":"m","score":1.0,"status":"verified"}"#,
        )
        .unwrap();
        assert_eq!(legacy.scope, "sentence");
    }

    /// Wire freeze (spec M3 D4): the `kind` discriminator must never appear in a
    /// serialized `Page`, so the app-visible contract stays byte-identical to
    /// pre-M3. A shadow page (`kind = "entity"`) is fenced from readers, but even
    /// a normal page must not leak the field.
    #[test]
    fn kind_is_never_serialized_onto_the_wire() {
        let mut page: Page = serde_json::from_str(
            r#"{"id":"p1","title":"T","content":"body","source_memory_ids":[],
                "version":1,"status":"active","created_at":"x","last_compiled":"x",
                "last_modified":"x","sources_updated_count":0,"user_edited":false}"#,
        )
        .unwrap();
        // Deserialization still fills `kind` from its default when absent.
        assert_eq!(page.kind, "concept");
        page.kind = "entity".to_string();

        let value: serde_json::Value = serde_json::to_value(&page).unwrap();
        assert!(
            value.get("kind").is_none(),
            "serialized Page must not carry a `kind` key (wire freeze D4); got {value}"
        );
    }
}
