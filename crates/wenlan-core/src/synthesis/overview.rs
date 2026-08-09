// SPDX-License-Identifier: Apache-2.0
//! Spec §5.3: the reserved machine-owned Overview page (nashsu `overview.md`
//! parity). One well-known title, never duplicated; the maintenance pass
//! syncs its evidence to the currently most-active pages and refreshes it in
//! place through the ONE re-distill op -- no new write primitive, no new
//! table: creation goes through `post_write::create_page` (floor-exempt
//! `research` kind, same guard as any machine-owned page) and the refresh IS
//! `refresh_page` (via `refresh_page_with_prompt`, parameterized on the
//! dedicated `overview_summary` prompt -- the "+ a summary prompt" the spec
//! calls for -- instead of the generic deep-dive `distill_page` template).

use std::path::Path;
use std::sync::Arc;

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::llm_provider::LlmProvider;
use crate::prompts::PromptRegistry;
use crate::synthesis::distill::{refresh_page_with_prompt, RefreshOutcome, RefreshReason};

/// Reserved, well-known title for the single machine-owned overview page.
/// Looked up case-insensitively via `find_active_page_id_by_title` so the
/// row is created once and refreshed in place forever after -- never
/// duplicated.
pub const OVERVIEW_PAGE_TITLE: &str = "Overview";

/// How many of the most recently touched active pages feed the overview's
/// evidence set. Small and fixed on purpose -- spec says "no new machinery".
const OVERVIEW_TOP_PAGES: i64 = 5;

/// Placeholder body for the reserved row before its first refresh populates
/// real content. `create_page` requires non-empty content; `refresh_page`
/// overwrites this on the first maintenance tick.
const OVERVIEW_PLACEHOLDER_CONTENT: &str =
    "This page is refreshed automatically to summarize the wiki's current top pages.";

/// Source memory ids of the current top pages, for the overview's own
/// evidence set. `exclude_page_id` keeps the overview page from citing its
/// own prior summary as if it were a memory.
async fn top_page_source_ids(
    db: &MemoryDB,
    exclude_page_id: Option<&str>,
) -> Result<Vec<String>, WenlanError> {
    let pages = db.list_pages("active", OVERVIEW_TOP_PAGES + 1, 0).await?;
    let mut ids = Vec::new();
    let mut included = 0i64;
    for page in pages {
        if included >= OVERVIEW_TOP_PAGES {
            break;
        }
        if Some(page.id.as_str()) == exclude_page_id
            || page.title.eq_ignore_ascii_case(OVERVIEW_PAGE_TITLE)
        {
            continue;
        }
        let sources = db.get_page_sources(&page.id).await?;
        if !sources.is_empty() {
            ids.extend(sources.into_iter().map(|s| s.memory_source_id));
        } else {
            ids.extend(page.source_memory_ids.clone());
        }
        included += 1;
    }
    Ok(ids)
}

/// Looks up the reserved overview row by title, creating a floor-exempt
/// placeholder if it doesn't exist yet. Returns the page id. Idempotent: the
/// title lookup means a second call never creates a second row.
async fn ensure_overview_page(
    db: &MemoryDB,
    agent: &str,
    knowledge_path: Option<&Path>,
) -> Result<String, WenlanError> {
    if let Some(id) = db.find_active_page_id_by_title(OVERVIEW_PAGE_TITLE).await? {
        return Ok(id);
    }
    let req = wenlan_types::requests::CreateConceptRequest {
        title: OVERVIEW_PAGE_TITLE.to_string(),
        content: OVERVIEW_PLACEHOLDER_CONTENT.to_string(),
        summary: None,
        entity_id: None,
        space: None.into(),
        source_memory_ids: Vec::new(),
        // "research" is machine-owned (never `user_edited`/"authored") and
        // floor-exempt (spec §5.1: only `distilled` requires >=
        // page_min_cluster_size sources) -- the reserved row can exist with
        // zero sources until the first refresh populates it.
        creation_kind: Some("research".to_string()),
        workspace: None,
    };
    let result = crate::post_write::create_page(db, req, agent, knowledge_path).await?;
    Ok(result.id)
}

/// Spec §5.3: refresh the reserved overview page in place. Called by the
/// maintenance pass. Ensures the reserved row exists, REPLACES its evidence
/// with the current top pages' sources (`replace_page_sources` -- prunes
/// anything no longer top-ranked, so the set tracks "the current top pages"
/// instead of accumulating the union of every page ever top-ranked over the
/// wiki's lifetime), then goes through the same stale-mark / `refresh_page`
/// sequence as `refinery::re_distill_stale_pages`. The refresh write or exact
/// unchanged-result CAS owns staleness acknowledgement atomically.
pub async fn refresh_overview_page(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    agent: &str,
    knowledge_path: Option<&Path>,
) -> Result<RefreshOutcome, WenlanError> {
    let page_id = ensure_overview_page(db, agent, knowledge_path).await?;

    let top_sources = top_page_source_ids(db, Some(&page_id)).await?;
    let top_source_refs: Vec<&str> = top_sources.iter().map(String::as_str).collect();
    db.replace_page_sources(&page_id, &top_source_refs, "overview_sync")
        .await?;

    db.set_page_stale(&page_id, "overview_sync").await?;
    // The Overview needs a "table of contents" style summary, not the
    // deep-dive `distill_page` prompt every other page refresh uses -- so it
    // goes through `refresh_page_with_prompt` (the same re-distill op,
    // parameterized on the system prompt) with the dedicated
    // `overview_summary` template.
    let outcome = refresh_page_with_prompt(
        db,
        llm,
        &prompts.overview_summary,
        &page_id,
        RefreshReason::SourceChanged,
        knowledge_path,
    )
    .await?;
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::MemoryDB;
    use crate::events::NoopEmitter;
    use crate::llm_provider::{LlmBackend, LlmError, LlmProvider, LlmRequest, MockProvider};
    use crate::prompts::PromptRegistry;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;

    /// Records the system prompt of every `generate` call, so a test can
    /// assert the Overview refresh used the dedicated `overview_summary`
    /// prompt rather than the generic `distill_page` prompt every other page
    /// refresh uses.
    struct RecordingProvider {
        response: String,
        system_prompt: StdMutex<Option<String>>,
    }

    impl RecordingProvider {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                system_prompt: StdMutex::new(None),
            }
        }

        fn captured_system_prompt(&self) -> Option<String> {
            self.system_prompt.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for RecordingProvider {
        async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
            *self.system_prompt.lock().unwrap() = request.system_prompt.clone();
            Ok(self.response.clone())
        }

        fn is_available(&self) -> bool {
            true
        }

        fn name(&self) -> &str {
            "overview-recording"
        }

        fn backend(&self) -> LlmBackend {
            LlmBackend::OnDevice
        }

        fn kind(&self) -> &'static str {
            "mock"
        }
    }

    async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = MemoryDB::new(&path, Arc::new(NoopEmitter)).await.unwrap();
        (db, dir)
    }

    fn make_doc(source_id: &str, content: &str) -> crate::sources::RawDocument {
        crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: content.chars().take(40).collect(),
            summary: None,
            content: content.to_string(),
            url: None,
            last_modified: chrono::Utc::now().timestamp(),
            metadata: std::collections::HashMap::new(),
            memory_type: Some("fact".to_string()),
            space: None,
            source_agent: Some("test".to_string()),
            confidence: Some(0.7),
            confirmed: Some(false),
            stability: None,
            supersedes: None,
            pending_revision: false,
            entity_id: None,
            quality: None,
            importance: None,
            is_recap: false,
            enrichment_status: "raw".to_string(),
            supersede_mode: "hide".to_string(),
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            content_hash: None,
        }
    }

    async fn create_research_page(db: &MemoryDB, title: &str, mem_id: &str, content: &str) {
        db.upsert_documents(vec![make_doc(mem_id, content)])
            .await
            .unwrap();
        let req = wenlan_types::requests::CreateConceptRequest {
            title: title.to_string(),
            content: content.to_string(),
            summary: None,
            entity_id: None,
            space: None.into(),
            source_memory_ids: vec![mem_id.to_string()],
            creation_kind: Some("research".to_string()),
            workspace: None,
        };
        crate::post_write::create_page(db, req, "test", None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn refresh_overview_page_creates_reserved_row_summarizing_top_pages() {
        let (db, _dir) = test_db().await;

        let mem_content = "Rust is a systems programming language with memory safety guarantees";
        create_research_page(&db, "Rust", "mem_overview_rust", mem_content).await;

        let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&format!("{mem_content}.[1]")));
        let prompts = PromptRegistry::default();

        let outcome = refresh_overview_page(&db, &llm, &prompts, "test", None)
            .await
            .unwrap();
        assert!(outcome.wrote, "overview refresh should write in place");
        assert!(
            !outcome.gated,
            "overview page must be machine-owned, never gated to a revision card"
        );

        let overview_id = db
            .find_active_page_id_by_title(OVERVIEW_PAGE_TITLE)
            .await
            .unwrap()
            .expect("reserved overview page must exist after refresh");
        let page = db.get_page(&overview_id).await.unwrap().unwrap();
        assert!(!page.user_edited, "overview page must be machine-owned");
        assert_ne!(
            page.creation_kind, "authored",
            "overview page must never be human-owned (would gate refreshes into revision cards)"
        );
        assert!(
            page.content.contains("Rust"),
            "overview should summarize the current top page's evidence, got: {}",
            page.content
        );

        // A second maintenance tick, with a NEW top page in play, must refresh
        // the SAME reserved row in place -- never a second "Overview" page.
        let mem_content2 =
            "Python is a dynamically typed programming language emphasizing readability";
        create_research_page(&db, "Python", "mem_overview_python", mem_content2).await;
        let llm2: Arc<dyn LlmProvider> =
            Arc::new(MockProvider::new(&format!("{mem_content2}.[1]")));

        let outcome2 = refresh_overview_page(&db, &llm2, &prompts, "test", None)
            .await
            .unwrap();
        assert!(
            outcome2.wrote,
            "second maintenance tick should refresh again"
        );

        let all_active = db.list_pages("active", 100, 0).await.unwrap();
        let overview_pages: Vec<_> = all_active
            .iter()
            .filter(|p| p.title.eq_ignore_ascii_case(OVERVIEW_PAGE_TITLE))
            .collect();
        assert_eq!(
            overview_pages.len(),
            1,
            "overview page must never duplicate across maintenance ticks"
        );
        assert_eq!(
            overview_pages[0].id, overview_id,
            "the SAME reserved row must be refreshed in place"
        );
    }

    #[tokio::test]
    async fn refresh_overview_page_bounds_evidence_across_many_cycling_top_pages() {
        let (db, _dir) = test_db().await;
        let prompts = PromptRegistry::default();

        // Cycle through 7 top pages -- more than OVERVIEW_TOP_PAGES (5) -- to
        // prove the overview's evidence set tracks the CURRENT top pages
        // instead of accumulating the union of every page that was ever
        // top-ranked.
        let mut mem_ids = Vec::new();
        for i in 1..=7 {
            let title = format!("Topic{i}");
            let mem_id = format!("mem_overview_cycle_{i}");
            let content =
                format!("Topic{i} is a specific programming concept with unique details.");
            create_research_page(&db, &title, &mem_id, &content).await;
            mem_ids.push(mem_id);

            let llm: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(&format!("{content}[1]")));
            let outcome = refresh_overview_page(&db, &llm, &prompts, "test", None)
                .await
                .unwrap();
            assert!(outcome.wrote, "tick {i} should refresh the overview body");
        }

        let overview_id = db
            .find_active_page_id_by_title(OVERVIEW_PAGE_TITLE)
            .await
            .unwrap()
            .expect("reserved overview page must exist");

        let evidence = db.get_page_sources(&overview_id).await.unwrap();
        assert_eq!(
            evidence.len(),
            OVERVIEW_TOP_PAGES as usize,
            "overview evidence set must stay bounded to OVERVIEW_TOP_PAGES after {} cycling top pages, got {:?}",
            mem_ids.len(),
            evidence.iter().map(|s| &s.memory_source_id).collect::<Vec<_>>()
        );

        // The earliest cycling pages must have been pruned once they dropped
        // out of the current top-N -- proves replace semantics, not
        // additive-forever accumulation.
        let linked_ids: Vec<&str> = evidence
            .iter()
            .map(|s| s.memory_source_id.as_str())
            .collect();
        assert!(
            !linked_ids.contains(&mem_ids[0].as_str()),
            "earliest cycling page's source must be pruned once no longer top-ranked, evidence: {:?}",
            linked_ids
        );
        assert!(
            !linked_ids.contains(&mem_ids[1].as_str()),
            "second-earliest cycling page's source must also be pruned, evidence: {:?}",
            linked_ids
        );
    }

    /// Fix wave (stage c review, Critical-1 leak site 2): a fresh
    /// kind='entity' dual-write shadow (stamped `now`, so it would sort into
    /// the top of a `last_modified DESC` window ahead of every real page)
    /// must never consume one of the OVERVIEW_TOP_PAGES evidence slots or
    /// contribute its (always-empty) source_memory_ids.
    #[tokio::test]
    async fn top_page_source_ids_excludes_entity_kind_shadow() {
        let (db, _dir) = test_db().await;

        let mut mem_ids = Vec::new();
        for i in 1..=OVERVIEW_TOP_PAGES {
            let mem_id = format!("mem_overview_shadow_guard_{i}");
            let content =
                format!("Topic{i} is a specific programming concept with unique details.");
            create_research_page(&db, &format!("Topic{i}"), &mem_id, &content).await;
            mem_ids.push(mem_id);
        }
        // Seeded after the real pages, so store_entity's `now` timestamp
        // sorts it first without the fence.
        db.store_entity("Overview Shadow Marker", "person", None, None, None)
            .await
            .unwrap();

        let ids = top_page_source_ids(&db, None).await.unwrap();
        for mem_id in &mem_ids {
            assert!(
                ids.contains(mem_id),
                "real page source {mem_id} must not be displaced by the entity shadow, got: {ids:?}"
            );
        }
        assert_eq!(
            ids.len(),
            mem_ids.len(),
            "the shadow must contribute zero source ids of its own, got: {ids:?}"
        );
    }

    /// The Overview refresh must use its OWN dedicated summary prompt, not
    /// the generic `distill_page` prompt every other page refresh uses, and
    /// its synthesis input must span ALL of the wiki's current top pages at
    /// once (not just the most recently touched one).
    #[tokio::test]
    async fn refresh_overview_page_uses_dedicated_summary_prompt_and_spans_all_top_pages() {
        let (db, _dir) = test_db().await;
        let prompts = PromptRegistry::default();

        let rust_content = "Rust is a systems programming language with memory safety guarantees";
        create_research_page(&db, "Rust", "mem_overview_prompt_rust", rust_content).await;
        let python_content =
            "Python is a dynamically typed programming language emphasizing readability";
        create_research_page(&db, "Python", "mem_overview_prompt_python", python_content).await;

        // Both memories' own sentences, verbatim -- trivially grounded
        // against the faithfulness gate regardless of which page cites which.
        let response = format!("{rust_content}. [1] {python_content}. [2]");
        let recorder = Arc::new(RecordingProvider::new(&response));
        let llm: Arc<dyn LlmProvider> = recorder.clone();

        let outcome = refresh_overview_page(&db, &llm, &prompts, "test", None)
            .await
            .unwrap();
        assert!(outcome.wrote, "overview refresh should write in place");

        assert_eq!(
            recorder.captured_system_prompt(),
            Some(prompts.overview_summary.clone()),
            "the Overview refresh must use the dedicated overview_summary prompt"
        );
        assert_ne!(
            recorder.captured_system_prompt(),
            Some(prompts.distill_page.clone()),
            "the Overview refresh must NOT use the generic distill_page prompt"
        );

        let overview_id = db
            .find_active_page_id_by_title(OVERVIEW_PAGE_TITLE)
            .await
            .unwrap()
            .expect("reserved overview page must exist");
        let page = db.get_page(&overview_id).await.unwrap().unwrap();
        assert!(
            page.content.contains("Rust") && page.content.contains("Python"),
            "with two current top pages, the overview must summarize BOTH, not just one: {}",
            page.content
        );

        // Both pages' member memories must be in the evidence set at once --
        // proving the synthesis input spans every current top page, not just
        // whichever was touched most recently.
        let evidence = db.get_page_sources(&overview_id).await.unwrap();
        let linked_ids: Vec<&str> = evidence
            .iter()
            .map(|s| s.memory_source_id.as_str())
            .collect();
        assert!(
            linked_ids.contains(&"mem_overview_prompt_rust"),
            "evidence must include the Rust page's source, got {:?}",
            linked_ids
        );
        assert!(
            linked_ids.contains(&"mem_overview_prompt_python"),
            "evidence must include the Python page's source, got {:?}",
            linked_ids
        );
    }
}
