// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use wenlan_core::db::MemoryDB;
use wenlan_core::events::NoopEmitter;
use wenlan_core::llm_provider::{LlmBackend, LlmError, LlmProvider, LlmRequest};
use wenlan_core::maintenance::{
    run_maintenance_stage_slice, MaintenanceStage, MaintenanceTickConfig,
};
use wenlan_core::post_write::{create_page, update_page};
use wenlan_core::prompts::PromptRegistry;
use wenlan_core::synthesis::distill::distill_pages_scoped;
use wenlan_core::synthesis::refinement_queue::apply_refinement;
use wenlan_core::tuning::DistillationConfig;
use wenlan_types::requests::{CreateConceptRequest, UpdatePageRequest};
use wenlan_types::RawDocument;

async fn temp_db() -> (tempfile::TempDir, MemoryDB) {
    let dir = tempfile::tempdir().expect("temp dir");
    let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("open temp MemoryDB");
    (dir, db)
}

struct DistillStub {
    body: &'static str,
    title: &'static str,
}

#[async_trait::async_trait]
impl LlmProvider for DistillStub {
    async fn generate(&self, req: LlmRequest) -> Result<String, LlmError> {
        if req.label.as_deref() == Some("distill_body") {
            Ok(self.body.to_string())
        } else {
            Ok(self.title.to_string())
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "distill-redesign-stub"
    }

    fn backend(&self) -> LlmBackend {
        LlmBackend::Api
    }
}

async fn store_staging_memory(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    space: &str,
    last_modified: i64,
) {
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: content.chars().take(40).collect(),
        content: content.to_string(),
        last_modified,
        memory_type: Some("fact".to_string()),
        space: Some(space.to_string()),
        source_agent: Some("codex-test".to_string()),
        confidence: Some(0.9),
        confirmed: Some(true),
        stability: Some("learned".to_string()),
        quality: Some("high".to_string()),
        ..Default::default()
    }])
    .await
    .expect("store staging memory");
}

async fn store_staging_raw(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    space: &str,
    source_agent: &str,
    last_modified: i64,
) {
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: content.chars().take(40).collect(),
        content: content.to_string(),
        last_modified,
        memory_type: Some("fact".to_string()),
        space: Some(space.to_string()),
        source_agent: Some(source_agent.to_string()),
        confidence: Some(0.9),
        confirmed: Some(true),
        stability: Some("learned".to_string()),
        quality: Some("high".to_string()),
        content_hash: Some(format!("hash-{source_id}")),
        ..Default::default()
    }])
    .await
    .expect("store staging raw row");
}

fn distill_config() -> DistillationConfig {
    DistillationConfig {
        formation_threshold: 0.60,
        page_min_cluster_size: 3,
        page_match_threshold: 0.85,
        max_clusters_per_steep: 4,
        max_unlinked_cluster_size: 20,
        max_grouped_cluster_size: 20,
        ..Default::default()
    }
}

fn permissive_distill_config() -> DistillationConfig {
    DistillationConfig {
        formation_threshold: 0.0,
        page_match_threshold: 0.0,
        ..distill_config()
    }
}

fn page_req(
    title: &str,
    content: &str,
    source_ids: &[&str],
    creation_kind: &str,
    space: &str,
) -> CreateConceptRequest {
    CreateConceptRequest {
        title: title.to_string(),
        content: content.to_string(),
        summary: Some(title.to_string()),
        entity_id: None,
        space: (Some(space.to_string())).into(),
        source_memory_ids: source_ids.iter().map(|id| id.to_string()).collect(),
        creation_kind: Some(creation_kind.to_string()),
        workspace: Some(space.to_string()),
    }
}

fn maintenance_config() -> MaintenanceTickConfig {
    MaintenanceTickConfig {
        page_match_threshold: 0.85,
        formation_threshold: 0.60,
        page_min_cluster_size: 3,
        token_limit: 3_500,
        max_unlinked_cluster_size: 20,
        max_grouped_cluster_size: 20,
    }
}

#[tokio::test]
async fn full_flow_capture_staging_detect_compile_creates_unconfirmed_page_and_keep_card() {
    let (_dir, db) = temp_db().await;
    let now = chrono::Utc::now().timestamp();
    let memories = [
        (
            "distill_redesign_full_a",
            "The quartz scheduler keeps a bounded staging queue for page formation, grouping confirmed agent captures by shared intent before any compile lane runs.",
        ),
        (
            "distill_redesign_full_b",
            "The quartz scheduler uses the staging queue to detect related captures first, then routes page compilation through the canonical PageWrite path.",
        ),
        (
            "distill_redesign_full_c",
            "The quartz scheduler leaves every newly compiled page unconfirmed so the curation queue remains the backstop for page quality.",
        ),
    ];
    for (source_id, content) in memories {
        store_staging_memory(&db, source_id, content, "work", now).await;
    }

    let llm: Arc<dyn LlmProvider> = Arc::new(DistillStub {
        body: "The quartz scheduler detects related captures in staging before routing compilation through PageWrite.[1]\n\nNewly compiled pages stay unconfirmed for curation review.[3]",
        title: "Quartz Scheduler Page Formation",
    });
    let prompts = PromptRegistry::default();

    let result = distill_pages_scoped(&db, Some(&llm), &prompts, &distill_config(), None, None)
        .await
        .expect("distill full-flow cluster");
    assert_eq!(result.created.len(), 1, "one page should be compiled");
    assert!(
        result.pending.is_empty(),
        "stub LLM should finish the cluster"
    );

    let page = db
        .get_page(&result.created[0])
        .await
        .expect("read page")
        .expect("page exists");
    assert_eq!(page.review_status, "unconfirmed");

    let discovery = run_maintenance_stage_slice(
        &db,
        None,
        &prompts,
        &maintenance_config(),
        None,
        MaintenanceStage::CrossSpaceDiscovery,
    )
    .await
    .expect("run cross-space discovery slice");
    assert_eq!(
        discovery.result.discovery_cards_emitted, 0,
        "same-space compile must not become a cross-space card"
    );

    // The retro slice audits one Page per call; this database has exactly one.
    run_maintenance_stage_slice(
        &db,
        None,
        &prompts,
        &maintenance_config(),
        None,
        MaintenanceStage::RetroReview,
    )
    .await
    .expect("run retro review slice");

    let keep_cards: Vec<_> = db
        .get_pending_refinements()
        .await
        .expect("list refinement cards")
        .into_iter()
        .filter(|proposal| {
            proposal.action == "page_keep_or_archive"
                && proposal.source_ids.iter().any(|id| id == &page.id)
        })
        .collect();
    assert_eq!(
        keep_cards.len(),
        1,
        "new unconfirmed distilled page must have a keep/archive card"
    );
}

#[tokio::test]
async fn near_duplicate_cluster_attaches_to_existing_page_without_second_birth() {
    let (_dir, db) = temp_db().await;
    let now = chrono::Utc::now().timestamp();
    let existing_sources = [
        (
            "distill_redesign_attach_existing_a",
            "Rust workspaces share a single Cargo lockfile across related crates for stable builds.",
        ),
        (
            "distill_redesign_attach_existing_b",
            "Rust workspace members inherit shared package metadata from the root manifest.",
        ),
        (
            "distill_redesign_attach_existing_c",
            "Rust workspace checks validate every member crate together in one command.",
        ),
    ];
    for (source_id, content) in existing_sources {
        store_staging_memory(&db, source_id, content, "work", now).await;
    }
    let existing = create_page(
        &db,
        page_req(
            "Rust Workspace Operations",
            "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks.",
            &[
                "distill_redesign_attach_existing_a",
                "distill_redesign_attach_existing_b",
                "distill_redesign_attach_existing_c",
            ],
            "distilled",
            "work",
        ),
        "test",
        None,
    )
    .await
    .expect("seed existing page");
    db.set_page_review_status(&existing.id, "confirmed")
        .await
        .expect("confirm existing page");

    let candidate_sources = [
        (
            "distill_redesign_attach_candidate_a",
            "Rust workspaces share one Cargo lockfile for related crates during local builds.",
        ),
        (
            "distill_redesign_attach_candidate_b",
            "Rust workspace package metadata can be inherited from the root manifest.",
        ),
        (
            "distill_redesign_attach_candidate_c",
            "Rust workspace checks can validate all member crates together.",
        ),
    ];
    for (offset, (source_id, content)) in candidate_sources.into_iter().enumerate() {
        store_staging_memory(&db, source_id, content, "work", now + 10 + offset as i64).await;
    }

    let llm: Arc<dyn LlmProvider> = Arc::new(DistillStub {
        body: "Rust workspaces keep a shared Cargo lockfile, inherited package metadata, and all-crate checks for related crates.[1][2][3]",
        title: "Rust Workspace Operations",
    });
    let prompts = PromptRegistry::default();

    let result = distill_pages_scoped(
        &db,
        Some(&llm),
        &prompts,
        &permissive_distill_config(),
        None,
        None,
    )
    .await
    .expect("distill near-duplicate cluster");

    assert!(
        result.created.is_empty(),
        "near-duplicate cluster should attach, not create a new page"
    );
    let pages = db
        .list_pages("active", 10, 0)
        .await
        .expect("list active pages");
    assert_eq!(pages.len(), 1, "attach must leave one active page");

    let sources = db
        .get_page_sources(&existing.id)
        .await
        .expect("read attached sources");
    let locators: std::collections::BTreeSet<_> = sources
        .iter()
        .map(|source| source.memory_source_id.as_str())
        .collect();
    for expected in [
        "distill_redesign_attach_existing_a",
        "distill_redesign_attach_existing_b",
        "distill_redesign_attach_existing_c",
        "distill_redesign_attach_candidate_a",
        "distill_redesign_attach_candidate_b",
        "distill_redesign_attach_candidate_c",
    ] {
        assert!(
            locators.contains(expected),
            "attached page sources should include {expected}; got {locators:?}"
        );
    }
}

#[tokio::test]
async fn merge_card_accept_moves_data_now_and_defers_prose_recompile_atomically() {
    let (_dir, db) = temp_db().await;
    let now = chrono::Utc::now().timestamp();
    for (source_id, content) in [
        (
            "distill_redesign_merge_shared_a",
            "The blue migration board tracks adapter rollout owners and acceptance notes.",
        ),
        (
            "distill_redesign_merge_shared_b",
            "The blue migration board keeps adapter rollout risks visible during review.",
        ),
        (
            "distill_redesign_merge_survivor_c",
            "The blue migration board stores post-merge validation notes for the data lane.",
        ),
        (
            "distill_redesign_merge_absorbed_d",
            "The blue migration board records prose refresh follow-up after data moves.",
        ),
    ] {
        store_staging_memory(&db, source_id, content, "work", now).await;
    }

    let survivor_seed = create_page(
        &db,
        page_req(
            "Blue Migration Board",
            "The blue migration board tracks adapter rollout owners, acceptance notes, and post-merge validation notes.",
            &[
                "distill_redesign_merge_shared_a",
                "distill_redesign_merge_shared_b",
                "distill_redesign_merge_survivor_c",
            ],
            "research",
            "work",
        ),
        "test",
        None,
    )
    .await
    .expect("seed first merge page");
    db.set_page_review_status(&survivor_seed.id, "confirmed")
        .await
        .expect("confirm first merge page");
    let absorbed_seed = create_page(
        &db,
        page_req(
            "Blue Migration Board Notes",
            "The blue migration board tracks adapter rollout owners, acceptance notes, and prose refresh follow-up.",
            &[
                "distill_redesign_merge_shared_a",
                "distill_redesign_merge_shared_b",
                "distill_redesign_merge_absorbed_d",
            ],
            "research",
            "work",
        ),
        "test",
        None,
    )
    .await
    .expect("seed second merge page");
    db.set_page_review_status(&absorbed_seed.id, "confirmed")
        .await
        .expect("confirm second merge page");

    let prompts = PromptRegistry::default();
    let maintenance = run_maintenance_stage_slice(
        &db,
        None,
        &prompts,
        &maintenance_config(),
        None,
        MaintenanceStage::NearDuplicate,
    )
    .await
    .expect("detect merge card");
    assert_eq!(
        maintenance.result.merge_cards_emitted, 1,
        "near-duplicate confirmed pages should mint one merge card"
    );

    let merge_card = db
        .get_pending_refinements()
        .await
        .expect("list cards")
        .into_iter()
        .find(|proposal| proposal.action == "page_merge")
        .expect("page_merge card exists");
    let survivor_id = merge_card.source_ids[0].clone();
    let absorbed_id = merge_card.source_ids[1].clone();
    let survivor_before = db
        .get_page(&survivor_id)
        .await
        .expect("read survivor")
        .expect("survivor exists");

    let outcome = apply_refinement(&db, &merge_card.id, "test-agent")
        .await
        .expect("accept merge card");
    assert_eq!(outcome.action_applied, "page_merge");

    let survivor_after = db
        .get_page(&survivor_id)
        .await
        .expect("read survivor after")
        .expect("survivor exists after");
    let absorbed_after = db
        .get_page(&absorbed_id)
        .await
        .expect("read absorbed after")
        .expect("absorbed exists after");
    assert_eq!(absorbed_after.status, "archived");
    assert_eq!(
        survivor_after.content, survivor_before.content,
        "merge accept moves evidence now but leaves prose for later recompile"
    );
    assert_eq!(
        survivor_after.stale_reason.as_deref(),
        Some("source_updated")
    );
    assert!(survivor_after.pending_rebuild.is_some());

    let survivor_sources = db
        .get_page_sources(&survivor_id)
        .await
        .expect("read merged sources");
    let locators: std::collections::BTreeSet<_> = survivor_sources
        .iter()
        .map(|source| source.memory_source_id.as_str())
        .collect();
    for expected in [
        "distill_redesign_merge_shared_a",
        "distill_redesign_merge_shared_b",
        "distill_redesign_merge_survivor_c",
        "distill_redesign_merge_absorbed_d",
    ] {
        assert!(
            locators.contains(expected),
            "merged survivor should have source {expected}; got {locators:?}"
        );
    }
    let resolved_card = db
        .get_refinement_proposal(&merge_card.id)
        .await
        .expect("read resolved card")
        .expect("card remains for audit");
    assert_eq!(resolved_card.status, "resolved");
    assert!(
        !(absorbed_after.status == "active" && survivor_after.stale_reason.is_some()),
        "accept return must not expose data-moved/prose-stale state while absorbed page remains active"
    );
}

#[tokio::test]
async fn refresh_split_updates_machine_page_and_stages_human_revision_card() {
    let (_dir, db) = temp_db().await;
    let now = chrono::Utc::now().timestamp();
    let source_ids = [
        "distill_redesign_refresh_a",
        "distill_redesign_refresh_b",
        "distill_redesign_refresh_c",
    ];
    for (source_id, content) in [
        (
            source_ids[0],
            "The amber release lane refreshes machine-owned pages in place.",
        ),
        (
            source_ids[1],
            "The amber release lane stages revision cards for human-owned pages.",
        ),
        (
            source_ids[2],
            "The amber release lane keeps source ids attached during refresh.",
        ),
    ] {
        store_staging_memory(&db, source_id, content, "work", now).await;
    }

    let machine = create_page(
        &db,
        page_req(
            "Amber Release Lane",
            "The amber release lane starts with machine-owned page prose.",
            &source_ids,
            "distilled",
            "work",
        ),
        "test",
        None,
    )
    .await
    .expect("create machine page");
    db.set_page_review_status(&machine.id, "confirmed")
        .await
        .expect("confirm machine page");
    let machine_update = update_page(
        &db,
        &machine.id,
        UpdatePageRequest {
            content: "The amber release lane refreshes machine-owned pages in place while keeping source ids attached.".to_string(),
            source_memory_ids: source_ids.iter().map(|id| id.to_string()).collect(),
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "agent_refresh",
        false,
        None,
        None,
    )
    .await
    .expect("refresh machine page");
    assert!(machine_update.wrote);
    assert!(!machine_update.gated);
    let machine_page = db
        .get_page(&machine.id)
        .await
        .expect("read machine page")
        .expect("machine page exists");
    assert!(machine_page
        .content
        .contains("refreshes machine-owned pages in place"));
    assert!(!machine_page.user_edited);
    assert_eq!(
        machine_page.last_edited_by.as_deref(),
        Some("agent_refresh")
    );

    let human = create_page(
        &db,
        page_req(
            "Amber Human Lane",
            "The amber release lane starts with human-owned page prose.",
            &source_ids,
            "distilled",
            "work",
        ),
        "test",
        None,
    )
    .await
    .expect("create human page");
    let manual_content = "A human rewrote the amber release lane page and owns this prose.";
    update_page(
        &db,
        &human.id,
        UpdatePageRequest {
            content: manual_content.to_string(),
            source_memory_ids: source_ids.iter().map(|id| id.to_string()).collect(),
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "manual_edit",
        false,
        None,
        None,
    )
    .await
    .expect("mark human owned");

    let staged = update_page(
        &db,
        &human.id,
        UpdatePageRequest {
            content: "The amber release lane agent refresh should become a pending revision card."
                .to_string(),
            source_memory_ids: source_ids.iter().map(|id| id.to_string()).collect(),
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "agent_refresh",
        false,
        None,
        None,
    )
    .await
    .expect("refresh human-owned page");
    assert!(staged.gated);
    assert!(!staged.wrote);
    let revision_id = staged
        .revision_card_id
        .as_deref()
        .expect("revision card id returned");
    let human_page = db
        .get_page(&human.id)
        .await
        .expect("read human page")
        .expect("human page exists");
    assert_eq!(human_page.content, manual_content);
    assert!(human_page.user_edited);
    let revisions = db
        .list_pending_revisions(10)
        .await
        .expect("list pending revisions");
    assert!(
        revisions.iter().any(|revision| {
            revision.target_source_id == human.id && revision.revision_source_id == revision_id
        }),
        "human-owned refresh should stage a pending revision card"
    );
    let activity = db
        .list_agent_activity(10, Some("agent-refresh"), None)
        .await
        .expect("list agent_refresh activity");
    assert!(
        activity
            .iter()
            .any(|row| row.agent_name == "agent-refresh" && row.action == "page_revision_card"),
        "agent_refresh should log a revision-card activity"
    );
}

#[tokio::test]
async fn non_memory_citation_kinds_survive_initial_compile_and_readback() {
    let (_dir, db) = temp_db().await;
    let now = chrono::Utc::now().timestamp();
    for (source_id, content) in [
        (
            "distill_redesign_citation_plain_a",
            "Rust ownership rules prevent data races at compile time in shared systems.",
        ),
        (
            "distill_redesign_citation_plain_b",
            "Rust lifetimes explain when borrowed references remain valid across functions.",
        ),
        (
            "distill_redesign_citation_plain_c",
            "Rust move semantics transfer ownership between functions without a garbage collector.",
        ),
    ] {
        store_staging_memory(&db, source_id, content, "work", now).await;
    }
    store_staging_raw(
        &db,
        "folder-notes::rust/ownership.md",
        "The borrow checker enforces ownership rules statically by tracking lifetimes across function boundaries.",
        "work",
        "folder",
        now + 20,
    )
    .await;

    let llm: Arc<dyn LlmProvider> = Arc::new(DistillStub {
        body: "Rust ownership prevents data races at compile time.[1]\n\nRust lifetimes keep borrowed references valid across functions.[2]\n\nRust move semantics transfer ownership without a garbage collector.[3]\n\nThe borrow checker enforces ownership rules statically across function boundaries.[4]",
        title: "Rust Ownership Rules",
    });
    let prompts = PromptRegistry::default();
    let result = distill_pages_scoped(
        &db,
        Some(&llm),
        &prompts,
        &permissive_distill_config(),
        None,
        None,
    )
    .await
    .expect("distill citation-kind cluster");
    assert_eq!(result.created.len(), 1, "cluster should compile one page");

    let page = db
        .get_page(&result.created[0])
        .await
        .expect("read page")
        .expect("page exists");
    assert!(
        page.citations.iter().any(
            |citation| citation.locator == "folder-notes::rust/ownership.md"
                && citation.source_kind == "external_file"
        ),
        "folder-ingested source should read back as an external_file citation"
    );
    let evidence = db
        .get_page_evidence(&page.id)
        .await
        .expect("read page evidence");
    assert!(
        evidence.iter().any(|item| item.locator.as_deref()
            == Some("folder-notes::rust/ownership.md")
            && item.source_kind == "external_file"),
        "page_evidence should preserve the external_file kind"
    );
}

#[tokio::test]
async fn cross_space_cluster_mints_discovery_card_not_page() {
    let (_dir, db) = temp_db().await;
    let now = chrono::Utc::now().timestamp();
    for (source_id, space, content) in [
        (
            "distill_redesign_cross_space_work_a",
            "work",
            "The lantern planning thread links product launch notes to weekend travel logistics.",
        ),
        (
            "distill_redesign_cross_space_personal_b",
            "personal",
            "The lantern planning thread links weekend travel logistics to product launch notes.",
        ),
        (
            "distill_redesign_cross_space_work_c",
            "work",
            "The lantern planning thread keeps product launch notes and travel logistics together.",
        ),
    ] {
        store_staging_memory(&db, source_id, content, space, now).await;
    }

    let prompts = PromptRegistry::default();
    let config = MaintenanceTickConfig {
        formation_threshold: 0.0,
        ..maintenance_config()
    };
    let maintenance = run_maintenance_stage_slice(
        &db,
        None,
        &prompts,
        &config,
        None,
        MaintenanceStage::CrossSpaceDiscovery,
    )
    .await
    .expect("run cross-space discovery pass");
    assert_eq!(maintenance.result.discovery_cards_emitted, 1);
    let active_pages = db
        .list_pages("active", 10, 0)
        .await
        .expect("list active pages");
    assert!(
        active_pages.is_empty(),
        "cross-space cluster should mint a discovery card instead of compiling a page"
    );

    let card = db
        .get_pending_refinements()
        .await
        .expect("list cards")
        .into_iter()
        .find(|proposal| proposal.action == "cross_space_discovery")
        .expect("cross-space discovery card exists");
    assert_eq!(card.source_ids.len(), 3);
    let payload = card.payload.as_deref().unwrap_or_default();
    assert!(payload.contains("\"spaces\":[\"personal\",\"work\"]"));
    assert!(payload.contains("\"pick_space\""));
}
