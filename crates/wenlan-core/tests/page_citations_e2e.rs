// SPDX-License-Identifier: Apache-2.0
//! Per-claim citations — full-pipeline e2e (distill + annotate-only backfill).
//! Style of doc_reconcile_e2e: in-process MemoryDB, no server, no network LLM.

use std::sync::Arc;

use wenlan_core::db::{DistillationCluster, MemoryDB};
use wenlan_core::events::NoopEmitter;
use wenlan_core::llm_provider::{LlmBackend, LlmError, LlmProvider, LlmRequest};
use wenlan_core::post_write::create_page;
use wenlan_core::prompts::PromptRegistry;
use wenlan_core::synthesis::distill::distill_one_cluster;
use wenlan_types::requests::CreateConceptRequest;
use wenlan_types::RawDocument;

async fn temp_db() -> (tempfile::TempDir, MemoryDB) {
    let dir = tempfile::tempdir().unwrap();
    let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("open temp MemoryDB");
    (dir, db)
}

/// Stub for the distill path: returns a fixed marked body for the
/// `distill_body` call, and a short title for every other call (title gen).
struct DistillStub {
    body: &'static str,
}

#[async_trait::async_trait]
impl LlmProvider for DistillStub {
    async fn generate(&self, req: LlmRequest) -> Result<String, LlmError> {
        if req.label.as_deref() == Some("distill_body") {
            Ok(self.body.to_string())
        } else {
            Ok("Wenlan Daemon Notes".to_string())
        }
    }
    fn is_available(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "distill-stub"
    }
    fn backend(&self) -> LlmBackend {
        LlmBackend::Api
    }
}

/// Stub for the annotate-only backfill path: returns a fixed body for every
/// `citation_annotate` call.
struct AnnotateStub {
    body: String,
}

#[async_trait::async_trait]
impl LlmProvider for AnnotateStub {
    async fn generate(&self, _req: LlmRequest) -> Result<String, LlmError> {
        Ok(self.body.clone())
    }
    fn is_available(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "annotate-stub"
    }
    fn backend(&self) -> LlmBackend {
        LlmBackend::Api
    }
}

fn distill_cluster() -> DistillationCluster {
    DistillationCluster {
        source_ids: vec!["mem_daemon".into(), "mem_embed".into(), "mem_local".into()],
        contents: vec![
            "The Wenlan daemon binds to port 7878 by default on localhost, providing \
             the HTTP API surface used by the CLI and the MCP bridge for all downstream \
             tools that talk to the local memory store."
                .to_string(),
            "FastEmbed uses the BGE-Base-EN embeddings model with 768 dimensions for \
             vector search across every stored memory and page in the local database, \
             combined with FTS5 for hybrid retrieval."
                .to_string(),
            "Wenlan keeps personal agent memory local-first in a libSQL database so \
             tools can recall durable facts without sending every source artifact to \
             an external service."
                .to_string(),
        ],
        entity_id: None,
        entity_name: Some("Wenlan daemon".into()),
        space: None,
        estimated_tokens: 120,
        centroid_embedding: None,
    }
}

async fn seed_distill_sources(db: &MemoryDB, cluster: &DistillationCluster) {
    db.upsert_documents(
        cluster
            .source_ids
            .iter()
            .zip(cluster.contents.iter())
            .map(|(source_id, content)| RawDocument {
                source: "memory".to_string(),
                source_id: source_id.clone(),
                title: source_id.clone(),
                content: content.clone(),
                last_modified: chrono::Utc::now().timestamp(),
                confirmed: Some(true),
                ..Default::default()
            })
            .collect(),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn distill_emits_verified_citations() {
    let (_dir, db) = temp_db().await;
    let llm: Arc<dyn LlmProvider> = Arc::new(DistillStub {
        body: "The Wenlan daemon binds to port 7878 by default.[1]\n\n\
               FastEmbed uses BGE-Base embeddings with 768 dimensions.[2]",
    });
    let prompts = PromptRegistry::default();
    let cluster = distill_cluster();
    seed_distill_sources(&db, &cluster).await;

    let page_id = distill_one_cluster(&db, &llm, &prompts, &cluster, None)
        .await
        .unwrap()
        .expect("cluster should synthesize a page");

    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.citations.len(), 2, "citations: {:?}", page.citations);
    assert!(
        page.citations.iter().all(|c| c.status == "verified"),
        "both claims closely paraphrase their cited source: {:?}",
        page.citations
    );
    assert!(page.content.contains("[1]"));
    assert!(page.content.contains("[2]"));
}

#[tokio::test]
async fn out_of_range_marker_stripped_and_counted() {
    let (_dir, db) = temp_db().await;
    let llm: Arc<dyn LlmProvider> = Arc::new(DistillStub {
        body: "The Wenlan daemon binds to port 7878 by default.[1]\n\n\
               A hallucinated claim citing a source that does not exist.[9]",
    });
    let prompts = PromptRegistry::default();
    let cluster = distill_cluster();
    seed_distill_sources(&db, &cluster).await;

    let page_id = distill_one_cluster(&db, &llm, &prompts, &cluster, None)
        .await
        .unwrap()
        .expect("cluster should synthesize a page");

    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert!(
        !page.content.contains("[9]"),
        "out-of-range marker must be stripped from body: {}",
        page.content
    );
    assert!(page.content.contains("[1]"));
    assert_eq!(
        page.citations.len(),
        1,
        "only the in-range marker gets a citation record: {:?}",
        page.citations
    );
    assert_eq!(page.citations[0].marker, 1);
}

#[tokio::test]
async fn unmatched_claim_unverified() {
    let (_dir, db) = temp_db().await;
    // [2] correctly cites the embeddings source (verified); [1] cites the
    // daemon source but the claim text has nothing to do with it (unverified).
    let llm: Arc<dyn LlmProvider> = Arc::new(DistillStub {
        body: "FastEmbed uses BGE-Base embeddings with 768 dimensions for vector search.[2]\n\n\
               The daemon supports encrypted peer-to-peer video conferencing.[1]",
    });
    let prompts = PromptRegistry::default();
    let cluster = distill_cluster();
    seed_distill_sources(&db, &cluster).await;

    let page_id = distill_one_cluster(&db, &llm, &prompts, &cluster, None)
        .await
        .unwrap()
        .expect("cluster should synthesize a page");

    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.citations.len(), 2, "citations: {:?}", page.citations);
    let verified = page
        .citations
        .iter()
        .find(|c| c.marker == 2)
        .expect("marker 2 citation present");
    assert_eq!(verified.status, "verified");
    let unverified = page
        .citations
        .iter()
        .find(|c| c.marker == 1)
        .expect("marker 1 citation present");
    assert_eq!(unverified.status, "unverified");
    // The unverified claim's text stays in the body — badge, never a rewrite.
    assert!(page
        .content
        .contains("encrypted peer-to-peer video conferencing"));
}

/// Seed a legacy page (citations NULL) with one memory-kind evidence link,
/// mirroring the annotate-only backfill's target shape.
async fn seed_backfill_page(db: &MemoryDB, body: &str, mem_id: &str, mem_content: &str) -> String {
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_string(),
        source_id: mem_id.to_string(),
        title: mem_content.chars().take(40).collect(),
        content: mem_content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .unwrap();

    let result = create_page(
        db,
        CreateConceptRequest {
            title: "T".to_string(),
            content: body.to_string(),
            summary: None,
            entity_id: None,
            space: (None).into(),
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

const BACKFILL_BODY: &str = "The daemon binds to port 7878 by default.";
const BACKFILL_MEM_CONTENT: &str = "The daemon binds to port 7878 by default";

#[tokio::test]
async fn backfill_annotates_legacy_page() {
    let (_dir, db) = temp_db().await;
    let page_id = seed_backfill_page(&db, BACKFILL_BODY, "mem_a", BACKFILL_MEM_CONTENT).await;
    assert!(db
        .get_pages_missing_citations(10)
        .await
        .unwrap()
        .contains(&page_id));

    let annotated = format!("{BACKFILL_BODY}[1]");
    let llm: Arc<dyn LlmProvider> = Arc::new(AnnotateStub {
        body: annotated.clone(),
    });
    let prompts = PromptRegistry::default();

    wenlan_core::citations::run_citation_backfill_tick(&db, &llm, &prompts)
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

#[tokio::test]
async fn backfill_guard_rejects_rewrite_then_poison_pills() {
    let (_dir, db) = temp_db().await;
    let page_id = seed_backfill_page(&db, BACKFILL_BODY, "mem_a", BACKFILL_MEM_CONTENT).await;

    let rewritten = "A completely different sentence about something else entirely.[1]";
    let llm: Arc<dyn LlmProvider> = Arc::new(AnnotateStub {
        body: rewritten.to_string(),
    });
    let prompts = PromptRegistry::default();

    // 3 consecutive rejected ticks poison-pill the page.
    for _ in 0..3 {
        wenlan_core::citations::run_citation_backfill_tick(&db, &llm, &prompts)
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

#[tokio::test]
async fn old_page_wire_compat() {
    let (_dir, db) = temp_db().await;
    let result = create_page(
        &db,
        CreateConceptRequest {
            title: "Old Page".to_string(),
            content: "Some old body.".to_string(),
            summary: None,
            entity_id: None,
            space: (None).into(),
            source_memory_ids: vec![],
            creation_kind: Some("authored".to_string()),
            workspace: None,
        },
        "test",
        None,
    )
    .await
    .unwrap();

    let page = db.get_page(&result.id).await.unwrap().unwrap();
    assert!(
        page.citations.is_empty(),
        "a never citation-processed page deserializes with an empty citations vec"
    );
}
