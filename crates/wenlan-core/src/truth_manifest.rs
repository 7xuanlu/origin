// SPDX-License-Identifier: Apache-2.0
//! M5 reader manifest -- the executable enumeration of every registered reader.
//!
//! Generated from `docs/plans/2026-07-27-m5-reader-manifest-inventory.md`, which
//! is itself generated from the tree rather than hand-written. Three earlier
//! hand-enumerations were wrong in three different ways (151, 163, and a route
//! counted inside a `#[cfg(test)]` module), which is why neither this table nor
//! that document is trusted alone: the tests assert the two agree with each
//! other AND with a fresh scan of `crates/wenlan-server/src`.
//!
//! This table is the enforcement surface -- `TrackedRouter` looks routes up here
//! and refuses to register an unclassified path. The document is the reviewable
//! artifact. Neither may drift from the tree.
//!
//! Binding spec: `docs/plans/2026-07-27-m5-reader-manifest.md`.

/// Every HTTP method the router can register, not just the two the
/// scope-sensitivity table needs.
///
/// `lint::serving::routes::Method` carries only `Get`/`Post` because scope
/// sensitivity is a read-path concern. Truth classification must be **total**
/// over the router, so it needs the mutating methods too -- a `PUT` returning a
/// page title is page-bearing whatever else it does to the row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReaderMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
}

/// Which router instance a call site lands in.
///
/// Identity is `(builder, method, path)`, never `(method, path)`. `/api/health`
/// and `/api/status` register once per builder, and two call sites
/// (`lint_routes::register`, `repair_routes::register_execution`) land in both,
/// so a set keyed on path alone reports a phantom drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Builder {
    Main,
    Repair,
    MainAndRepair,
}

impl Builder {
    /// Does this call site install into `builder`?
    pub const fn installs_into(self, builder: Builder) -> bool {
        matches!(
            (self, builder),
            (Builder::Main, Builder::Main)
                | (Builder::Repair, Builder::Repair)
                | (Builder::MainAndRepair, Builder::Main)
                | (Builder::MainAndRepair, Builder::Repair)
        )
    }
}

/// Whether the reader can put page prose -- or an opaque payload that might
/// contain it -- in front of a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageBearing {
    Yes,
    No,
}

/// The reader's truth class.
///
/// There is deliberately no `Explicit` variant. Reader intent is a property of
/// the **call**, not the route -- `GET /api/pages` serves both a human who
/// clicked and a sidebar polling it every 10s -- so no route may earn
/// `explicit` from its path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthClass {
    Automatic,
    NotApplicable,
}

/// What a per-call human-intent marker may do at this reader.
///
/// Fail-closed by construction: a reader added tomorrow is `None` until someone
/// deliberately gives it a shape. `None` **refuses** a marker rather than
/// ignoring it -- an ignored marker is a wiring mistake that behaves correctly
/// today and silently wrong after a refactor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerShape {
    /// Nothing. The request is refused, not downgraded.
    None,
    /// Provisional entries: page id + title + both axes per item, never prose.
    /// Only for item types that can carry a page identity *and* both axes.
    Collection,
    /// Full prose for the page named in the path, both axes.
    NamedPage,
    /// Both, for a single reader that lists and opens. `wenlan pages` is the
    /// only one: bare it lists, with a query it opens. Its enforcement is the
    /// projection-directory invariant (binding spec section 5), not a wire gate
    /// -- it reads Markdown off disk, where negotiation cannot reach it.
    CollectionAndNamedPage,
}

/// One registered HTTP reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpReader {
    pub method: ReaderMethod,
    pub path: &'static str,
    pub builder: Builder,
    pub page_bearing: PageBearing,
    pub class: TruthClass,
    pub marker_shape: MarkerShape,
    /// The enforcement address -- must resolve to a real function.
    pub adapter: &'static str,
    /// What decided `page_bearing`. Reviewable, and re-checked by the field scan.
    pub evidence: &'static str,
}

/// One MCP tool. MCP may **never** transmit an intent marker: there is no human
/// gesture behind a tool call, the agent is the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McpReader {
    pub tool: &'static str,
    pub page_bearing: PageBearing,
    pub class: TruthClass,
    pub marker_shape: MarkerShape,
    pub adapter: &'static str,
}

/// One top-level CLI subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliReader {
    pub subcommand: &'static str,
    pub page_bearing: PageBearing,
    pub class: TruthClass,
    pub marker_shape: MarkerShape,
    pub adapter: &'static str,
}

/// All 165 registered `(method, path, handler)` triples.
///
/// 47 page-bearing, 118 not. Expands to 169 `(builder, method, path)`
/// runtime entries: 163 in `main`, 6 in `repair`.
#[rustfmt::skip]
pub const HTTP_READERS: &[HttpReader] = &[
    HttpReader { method: ReaderMethod::Get, path: "/api/activities", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_list_activities", evidence: "AgentActivityRow.detail = title={page.title}" },
    HttpReader { method: ReaderMethod::Get, path: "/api/agents", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/agents/{name}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_delete_agent", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Get, path: "/api/agents/{name}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Put, path: "/api/agents/{name}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/briefing", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/capture-stats", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_capture_stats", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Post, path: "/api/chunks/delete-bulk", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/chunks/time-range", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/chunks/{id}/update", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/chunks/{source_id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_get_chunks", evidence: "MemoryDetail.title/content via revision card" },
    HttpReader { method: ReaderMethod::Get, path: "/api/communities", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/communities/members", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/communities/page-assignments", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/communities/proposals", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/communities/proposals/{id}/accept", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/communities/proposals/{id}/reject", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/config", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Put, path: "/api/config", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/config/routing", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/config/skip-apps", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/config/skip-apps", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/context", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_context", evidence: "ChatContextResponse.context, KnowledgeContext.graph_context, Searc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/debug/pipeline", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_pipeline_status", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Get, path: "/api/decisions", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/decisions/domains", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/distill", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_distill", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Post, path: "/api/distill/{page_id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_redistill", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Post, path: "/api/documents/{source_id}/space", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/documents/{source_id}/tags", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/documents/{source}/{source_id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/health", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/health", builder: Builder::Repair, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/home-stats", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/import/chat-export", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/import/memories", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/import/state", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/indexed-files", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/ingest/memory", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/ingest/text", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/ingest/webpage", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/knowledge/count", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/knowledge/path", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/knowledge/recent-relations", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/lint", builder: Builder::MainAndRepair, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_lint", evidence: "LintAgentRecord.excerpt, LintAgentRecord.source_excerpt, LintCheck" },
    HttpReader { method: ReaderMethod::Post, path: "/api/lint", builder: Builder::MainAndRepair, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_lint_submission", evidence: "LintAgentRecord.excerpt, LintAgentRecord.source_excerpt, LintCheck" },
    HttpReader { method: ReaderMethod::Post, path: "/api/llm/test", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/by-ids", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/confirm/{source_id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/contradiction/{source_id}/dismiss", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/memory/delete/{source_id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/entities", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/entities/list", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/entities/search", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/entities/{entity_id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_get_entity_detail", evidence: "Entity.name = pages.title (M3)" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/entities/{entity_id}/observations", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/memory/entities/{id}/confirm", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/memory/entities/{id}/delete", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/entity-suggestions", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/link-entity", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_link_entity", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/list", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/nurture", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/observations", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/memory/observations/{id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/memory/observations/{id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/memory/observations/{id}/confirm", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/pending-revision/{source_id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/pending-revisions", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/pinned", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/recent", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/reclassify/{source_id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/rejections", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/relations", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/revision/{id}/accept", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/revision/{id}/dismiss", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/search", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_search_memory", evidence: "SearchResult.content, SearchResult.content_hash, SearchResult.last" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/stats", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/store", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/unconfirmed", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/{id}/correct", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_correct_memory", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/{id}/detail", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/{id}/pin", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/{id}/revisions", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Put, path: "/api/memory/{id}/stability", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/{id}/unpin", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/memory/{id}/update", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/memory/{id}/update-page", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/{id}/versions", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/memory/{source_id}/enrichment-status", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/on-device-model", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/on-device-model/download", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/onboarding/milestones", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/onboarding/milestones/{id}/acknowledge", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/onboarding/reset", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::Collection, adapter: "handle_list_pages", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Post, path: "/api/pages", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/pages/export", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_export_pages", evidence: "EFFECT: writes page prose to the requested vault" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages/orphan-links", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_list_orphan_links", evidence: "OrphanLink.label, OrphanLinksResponse.orphan_labels" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages/recent", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::Collection, adapter: "handle_recent_pages", evidence: "RecentActivityItem.snippet, RecentActivityItem.title" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages/recent-changes", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::Collection, adapter: "handle_recent_page_changes", evidence: "PageChange.title" },
    HttpReader { method: ReaderMethod::Post, path: "/api/pages/search", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::Collection, adapter: "handle_search_pages", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/pages/{id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_delete_page", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages/{id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::NamedPage, adapter: "handle_get_page", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Put, path: "/api/pages/{id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/pages/{id}/archive", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_archive_page", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Post, path: "/api/pages/{id}/export", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_export_page", evidence: "EFFECT: writes page prose to the requested vault" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages/{id}/links", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::NamedPage, adapter: "handle_get_page_links", evidence: "PageLinkInbound.label, PageLinkOutbound.label" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/pages/{id}/map", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_reset_page_map", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages/{id}/map", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::NamedPage, adapter: "handle_get_page_map", evidence: "PageMapEdge.label, PageMapNode.label" },
    HttpReader { method: ReaderMethod::Post, path: "/api/pages/{id}/map/edges", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_create_map_edge", evidence: "PageMapEdge.label" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/pages/{id}/map/edges/{edge_id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_delete_map_edge", evidence: "PageMapEdge.label" },
    HttpReader { method: ReaderMethod::Patch, path: "/api/pages/{id}/map/edges/{edge_id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_patch_map_edge", evidence: "PageMapEdge.label" },
    HttpReader { method: ReaderMethod::Post, path: "/api/pages/{id}/map/improve", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_improve_page_map", evidence: "PageMapEdge.label, PageMapNode.label" },
    HttpReader { method: ReaderMethod::Put, path: "/api/pages/{id}/map/layout", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_put_page_map_layout", evidence: "PageMapEdge.label, PageMapNode.label" },
    HttpReader { method: ReaderMethod::Post, path: "/api/pages/{id}/map/nodes", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_create_map_node", evidence: "PageMapNode.label" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/pages/{id}/map/nodes/{node_id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_delete_map_node", evidence: "PageMapNode.label" },
    HttpReader { method: ReaderMethod::Patch, path: "/api/pages/{id}/map/nodes/{node_id}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_patch_map_node", evidence: "PageMapNode.label" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages/{id}/revisions", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::NamedPage, adapter: "handle_get_page_revisions", evidence: "PageChangelogEntry.citations_summary, PageChangelogEntry.delta_sum" },
    HttpReader { method: ReaderMethod::Get, path: "/api/pages/{id}/sources", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::NamedPage, adapter: "handle_get_page_sources", evidence: "MemoryItem.content, MemoryItem.source_text, MemoryItem.summary, Me" },
    HttpReader { method: ReaderMethod::Get, path: "/api/ping", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_ping", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Get, path: "/api/profile", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/profile", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/profile/narrative", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/profile/narrative/regenerate", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/refinery/queue", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/refinery/queue/{id}/accept", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/refinery/queue/{id}/reject", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/repairs/apply", builder: Builder::MainAndRepair, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_apply", evidence: "RepairTarget.label_key" },
    HttpReader { method: ReaderMethod::Post, path: "/api/repairs/plan", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/repairs/plan/entries", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_plan_entries", evidence: "RepairMutation.after_title, RepairMutation.before_title, RepairSys" },
    HttpReader { method: ReaderMethod::Post, path: "/api/repairs/prepare", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_prepare", evidence: "RepairMutation.after_title, RepairMutation.before_title, RepairTar" },
    HttpReader { method: ReaderMethod::Post, path: "/api/repairs/verify", builder: Builder::MainAndRepair, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/retrievals/recent", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_recent_retrievals", evidence: "RetrievalEvent.memory_snippets, RetrievalEvent.page_titles" },
    HttpReader { method: ReaderMethod::Post, path: "/api/search", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_search", evidence: "SearchResult.content, SearchResult.content_hash, SearchResult.last" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/setup/anthropic-key", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Put, path: "/api/setup/anthropic-key", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/setup/status", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/shutdown", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_shutdown", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Get, path: "/api/snapshots", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/snapshots/{id}/captures", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Get, path: "/api/snapshots/{id}/captures-with-content", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/snapshots/{id}/delete", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/sources", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/sources", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/sources/{id}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/sources/{id}/sync", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/spaces", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/spaces", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/spaces/default", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/spaces/default", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Put, path: "/api/spaces/default", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/spaces/reorder", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/spaces/{from}/move-to/{to}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_move_space", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/spaces/{name}", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_delete_space", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Put, path: "/api/spaces/{name}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
    HttpReader { method: ReaderMethod::Post, path: "/api/spaces/{name}/confirm", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/spaces/{name}/pin", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/spaces/{name}/star", builder: Builder::Main, page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "handle_toggle_space_starred", evidence: "opaque response type — fail-closed" },
    HttpReader { method: ReaderMethod::Get, path: "/api/status", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/status", builder: Builder::Repair, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Post, path: "/api/steep", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/suggest-tags", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/api/tags", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Delete, path: "/api/tags/{name}", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "no prose fields" },
    HttpReader { method: ReaderMethod::Get, path: "/ws/updates", builder: Builder::Main, page_bearing: PageBearing::No, class: TruthClass::NotApplicable, marker_shape: MarkerShape::None, adapter: "—", evidence: "DEMOTED — proof in the inventory doc" },
];

/// All 29 `#[tool(` declarations in `crates/wenlan-mcp/src/tools.rs`.
#[rustfmt::skip]
pub const MCP_READERS: &[McpReader] = &[
    McpReader { tool: "accept_refinement", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "accept_revision", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "apply_lint_repair", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "capture", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "confirm_memory", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "context", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "create_entity", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "create_relation", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "delete_page", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "dismiss_revision", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "distill", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "forget", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "get_lint_agent_work_page", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "get_lint_repair_plan_entries", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "get_memory_revisions", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "get_page_revisions", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "get_page_sources", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "lint", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "list_pending", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "list_pending_imports", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "list_pending_revisions", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "list_refinements", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "list_rejections", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "prepare_lint_repair", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "prepare_lint_repair_plan", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "recall", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "reject_refinement", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "verify_lint_repair", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
    McpReader { tool: "write_page", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "tool handler" },
];

/// All 19 top-level `Commands` variants in `crates/wenlan-cli/src/main.rs`.
#[rustfmt::skip]
pub const CLI_READERS: &[CliReader] = &[
    CliReader { subcommand: "wenlan status", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan setup", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan background", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan restart", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan doctor", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan lint", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan models", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan keys", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan enrichment", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan connect", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan search", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan recall", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan pages", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::CollectionAndNamedPage, adapter: "enforce_projection_directory_invariant" },
    CliReader { subcommand: "wenlan sources", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan capture", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan memories", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan curate", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan agents", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
    CliReader { subcommand: "wenlan spaces", page_bearing: PageBearing::Yes, class: TruthClass::Automatic, marker_shape: MarkerShape::None, adapter: "subcommand renderer" },
];

/// Look up a registered HTTP reader by its runtime identity.
///
/// `builder` matters: a call site marked `MainAndRepair` answers for both.
pub fn http_reader(
    builder: Builder,
    method: ReaderMethod,
    path: &str,
) -> Option<&'static HttpReader> {
    HTTP_READERS
        .iter()
        .find(|row| row.method == method && row.path == path && row.builder.installs_into(builder))
}

/// Every `(builder, method, path)` actually installed -- the `MainAndRepair`
/// rows expanded into one entry per builder. 169 entries from 165 call sites.
pub fn runtime_entries() -> impl Iterator<Item = (Builder, ReaderMethod, &'static str)> {
    HTTP_READERS.iter().flat_map(|row| {
        [Builder::Main, Builder::Repair]
            .into_iter()
            .filter(move |b| row.builder.installs_into(*b))
            .map(move |b| (b, row.method, row.path))
    })
}

#[cfg(test)]
#[path = "truth_manifest_test.rs"]
mod tests;
