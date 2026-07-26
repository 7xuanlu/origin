// SPDX-License-Identifier: Apache-2.0
//! Entity extraction phase: extract entities/relations from a single memory's content.

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::prompts::PromptRegistry;
use std::sync::Arc;

/// Extract KG entities from `content` via LLM, create/upsert them in the DB,
/// and return the list of entity ids. Does **not** write to `memory_entities` —
/// the caller is responsible for linking them to a specific memory.
///
/// This is the extraction-only primitive used by `run_enrichment_sweep` (and any
/// other caller that controls when/how linkage happens).
pub async fn extract_entities_for_content(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    content: &str,
) -> Result<Vec<String>, WenlanError> {
    let truncated: String = content.chars().take(500).collect();
    let numbered = format!("1. {}", truncated);

    let response = llm
        .generate(LlmRequest {
            system_prompt: Some(prompts.extract_knowledge_graph.clone()),
            user_prompt: numbered,
            max_tokens: 512,
            temperature: 0.3,
            label: Some("extract".into()),
            timeout_secs: None,
        })
        .await
        .map_err(|e| WenlanError::Llm(format!("entity extraction: {}", e)))?;

    let batch = [(0usize, content.to_string())];
    let kg_results = crate::extract::parse_kg_response(&response, &batch);

    let mut entity_cache: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for kg in &kg_results {
        for entity in &kg.entities {
            let name_lower = entity.name.to_lowercase();
            if entity_cache.contains_key(&name_lower) {
                continue;
            }
            let req = wenlan_types::requests::CreateEntityRequest {
                name: entity.name.clone(),
                entity_type: entity.entity_type.clone(),
                space: None,
                source_agent: Some("post_ingest".to_string()),
                confidence: None,
            };
            match crate::post_write::create_entity(db, req, "post_ingest").await {
                Ok(result) => {
                    entity_cache.insert(name_lower, result.id);
                }
                Err(e) => log::warn!("[extract_entities_for_content] entity create failed: {e}"),
            }
        }
        for obs in &kg.observations {
            if let Some(entity_id) = entity_cache.get(&obs.entity.to_lowercase()) {
                let req = wenlan_types::requests::AddObservationRequest {
                    entity_id: entity_id.clone(),
                    content: obs.content.clone(),
                    source_agent: Some("post_ingest".to_string()),
                    confidence: None,
                };
                if let Err(e) = crate::post_write::add_observation(db, req, "post_ingest").await {
                    log::warn!("[extract_entities_for_content] add_observation failed: {e}");
                }
            }
        }
        for rel in &kg.relations {
            let from_id = entity_cache.get(&rel.from.to_lowercase()).cloned();
            let to_id = entity_cache.get(&rel.to.to_lowercase()).cloned();
            if let (Some(from), Some(to)) = (from_id, to_id) {
                let req = wenlan_types::requests::CreateRelationRequest {
                    from_entity: from,
                    to_entity: to,
                    relation_type: rel.relation_type.clone(),
                    source_agent: Some("post_ingest".to_string()),
                    confidence: rel.confidence,
                    explanation: rel.explanation.clone(),
                    source_memory_id: None,
                    span: rel.span.clone(),
                    model_version: Some(llm.model_id()),
                    prompt_version: Some(
                        crate::extract::EXTRACT_KNOWLEDGE_GRAPH_PROMPT_VERSION.to_string(),
                    ),
                };
                if let Err(e) = crate::post_write::create_relation_with_span(
                    db,
                    req,
                    "post_ingest",
                    Some(content),
                )
                .await
                {
                    log::warn!("[extract_entities_for_content] create_relation failed: {e}");
                }
            }
        }
    }

    Ok(entity_cache.into_values().collect())
}

/// Pure half: call the LLM and parse the KG response. No DB access.
/// Returns the raw parse result to be committed by `commit_kg`.
pub async fn extract_kg(
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    content: &str,
) -> Result<Vec<crate::extract::KgExtractionResult>, WenlanError> {
    let truncated: String = content.chars().take(500).collect();
    let numbered = format!("1. {}", truncated);

    let response = llm
        .generate(LlmRequest {
            system_prompt: Some(prompts.extract_knowledge_graph.clone()),
            user_prompt: numbered,
            max_tokens: 512,
            temperature: 0.3,
            label: Some("extract".into()),
            timeout_secs: None,
        })
        .await
        .map_err(|e| WenlanError::Llm(format!("entity extraction: {}", e)))?;

    let batch = [(0usize, content.to_string())];
    Ok(crate::extract::parse_kg_response(&response, &batch))
}

/// Serial DB-write half: commit a parsed KG result set to the DB and link
/// the memory row identified by `source_id`. Returns the primary entity id.
/// `content` is the exact text `kg` was extracted from -- threaded through
/// (never re-fetched) so M3g span capture (§2.3) locates each relation's
/// quote against the string the model actually saw. `model_version` is the
/// extraction LLM's provenance for the same capture (§6.6).
pub async fn commit_kg(
    db: &MemoryDB,
    source_id: &str,
    kg: &[crate::extract::KgExtractionResult],
    content: &str,
    model_version: Option<&str>,
) -> Result<Option<String>, WenlanError> {
    let mut entity_cache: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut first_entity_id: Option<String> = None;

    for kg_item in kg {
        for entity in &kg_item.entities {
            let name_lower = entity.name.to_lowercase();
            // In-batch cache (per-extraction-pass dedup; capability fn doesn't take a cache)
            if let Some(id) = entity_cache.get(&name_lower) {
                if first_entity_id.is_none() {
                    first_entity_id = Some(id.clone());
                }
                continue;
            }
            let req = wenlan_types::requests::CreateEntityRequest {
                name: entity.name.clone(),
                entity_type: entity.entity_type.clone(),
                space: None,
                source_agent: Some("post_ingest".to_string()),
                confidence: None,
            };
            match crate::post_write::create_entity(db, req, "post_ingest").await {
                Ok(result) => {
                    entity_cache.insert(name_lower, result.id.clone());
                    if first_entity_id.is_none() {
                        first_entity_id = Some(result.id);
                    }
                }
                Err(e) => log::warn!("[post_ingest] entity create failed: {e}"),
            }
        }
        for obs in &kg_item.observations {
            if let Some(entity_id) = entity_cache.get(&obs.entity.to_lowercase()) {
                let req = wenlan_types::requests::AddObservationRequest {
                    entity_id: entity_id.clone(),
                    content: obs.content.clone(),
                    source_agent: Some("post_ingest".to_string()),
                    confidence: None,
                };
                if let Err(e) = crate::post_write::add_observation(db, req, "post_ingest").await {
                    log::warn!("[post_ingest] add_observation failed: {e}");
                }
            }
        }
        for rel in &kg_item.relations {
            let from_id = entity_cache.get(&rel.from.to_lowercase()).cloned();
            let to_id = entity_cache.get(&rel.to.to_lowercase()).cloned();
            if let (Some(from), Some(to)) = (from_id, to_id) {
                let req = wenlan_types::requests::CreateRelationRequest {
                    from_entity: from,
                    to_entity: to,
                    relation_type: rel.relation_type.clone(),
                    source_agent: Some("post_ingest".to_string()),
                    confidence: rel.confidence,
                    explanation: rel.explanation.clone(),
                    source_memory_id: Some(source_id.to_string()),
                    span: rel.span.clone(),
                    model_version: model_version.map(|s| s.to_string()),
                    prompt_version: Some(
                        crate::extract::EXTRACT_KNOWLEDGE_GRAPH_PROMPT_VERSION.to_string(),
                    ),
                };
                if let Err(e) = crate::post_write::create_relation_with_span(
                    db,
                    req,
                    "post_ingest",
                    Some(content),
                )
                .await
                {
                    log::warn!("[post_ingest] create_relation failed: {e}");
                }
            }
        }
    }

    // Link memory to first entity (1-1 legacy column).
    if let Some(ref eid) = first_entity_id {
        let _ = db.update_memory_entity_id(source_id, eid).await;
    }

    // Write all extracted entities into the many-to-many junction table.
    if !entity_cache.is_empty() {
        let ids: Vec<&str> = entity_cache.values().map(|s| s.as_str()).collect();
        if let Err(e) = db.link_memory_entities(source_id, &ids).await {
            log::warn!("[post_ingest] link_memory_entities failed: {e}");
        }
    }

    Ok(first_entity_id)
}

/// Extract entities from a single memory via LLM. Returns the primary entity_id if one was created/found.
pub async fn extract_single_memory_entities(
    db: &MemoryDB,
    llm: &Arc<dyn LlmProvider>,
    prompts: &PromptRegistry,
    source_id: &str,
    content: &str,
) -> Result<Option<String>, WenlanError> {
    let kg = extract_kg(llm, prompts, content).await?;
    commit_kg(db, source_id, &kg, content, Some(&llm.model_id())).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_provider::CannedLlmProvider;
    use crate::sources::RawDocument;
    use std::sync::Arc;

    /// Build a test prompt registry using defaults (no override dir needed).
    fn test_prompts() -> PromptRegistry {
        crate::prompts::PromptRegistry::default()
    }

    /// Build a CannedLlmProvider that returns a minimal KG JSON containing
    /// "Alice" as an entity when the extract_knowledge_graph prompt is used.
    fn canned_alice() -> Arc<CannedLlmProvider> {
        let prompts = test_prompts();
        // The key must appear in (or equal) the system_prompt sent by extract_kg.
        // We use a short substring of the prompt as the key.
        let key_fragment = prompts
            .extract_knowledge_graph
            .chars()
            .take(30)
            .collect::<String>();
        let kg_json =
            r#"[{"entities":[{"name":"Alice","type":"person"}],"observations":[],"relations":[]}]"#;
        Arc::new(CannedLlmProvider::new("DEFAULT").with(key_fragment, kg_json))
    }

    #[tokio::test]
    async fn extract_kg_returns_parsed_kg_no_db() {
        let prompts = test_prompts();
        let canned = canned_alice();
        let result = extract_kg(
            &(canned as Arc<dyn LlmProvider>),
            &prompts,
            "Alice joined Acme",
        )
        .await
        .expect("extract_kg should succeed");
        assert!(
            !result.is_empty(),
            "should return at least one KgExtractionResult"
        );
        let first = &result[0];
        assert!(
            first.entities.iter().any(|e| e.name == "Alice"),
            "expected entity 'Alice' in result, got: {:?}",
            first.entities
        );
    }

    #[tokio::test]
    async fn extract_then_commit_creates_and_links() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = crate::db::MemoryDB::new(dir.path(), Arc::new(crate::events::NoopEmitter))
            .await
            .expect("MemoryDB::new failed");
        // Seed a memory row so commit_kg's update_memory_entity_id / link_memory_entities have a target.
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: "m1".to_string(),
            title: "Test memory".to_string(),
            content: "Alice joined Acme".to_string(),
            ..Default::default()
        };
        db.upsert_documents(vec![doc])
            .await
            .expect("upsert_documents failed");

        let prompts = test_prompts();
        let canned = canned_alice();
        let kg = extract_kg(
            &(canned as Arc<dyn LlmProvider>),
            &prompts,
            "Alice joined Acme",
        )
        .await
        .expect("extract_kg failed");

        let eid = commit_kg(&db, "m1", &kg, "Alice joined Acme", None)
            .await
            .expect("commit_kg failed");
        assert!(
            eid.is_some(),
            "expected a primary entity id after commit_kg"
        );

        // Verify the memory is linked to the entity via the legacy 1-1 column.
        let stored_eid = db
            .get_memory_entity_id("m1")
            .await
            .expect("get_memory_entity_id failed");
        assert_eq!(
            stored_eid, eid,
            "stored entity_id should match the one returned by commit_kg"
        );
    }
}
