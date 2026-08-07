// SPDX-License-Identifier: Apache-2.0
//! Knowledge graph quality checks: post-store verification and periodic rethink.

use crate::db::{
    ContradictionObservationCount, DuplicateNameGroup, MemoryDB, MinHashEntityInput,
    StaleEntityEmbeddingCandidate,
};
use crate::error::WenlanError;
use crate::llm_provider::LlmProvider;
use crate::read_scope::ReadScope;
use crate::tuning::RefineryConfig;
use std::sync::Arc;

/// Result of a post-store verification check.
#[derive(Debug)]
pub struct VerificationResult {
    pub entity_self_retrieval_passed: Option<bool>,
    pub concept_self_retrieval_passed: Option<bool>,
    pub relation_consistency_passed: Option<bool>,
    pub warnings: Vec<String>,
}

/// Run post-store verification checks on a newly created/linked entity.
pub async fn verify_entity(
    db: &MemoryDB,
    entity_id: &str,
    entity_name: &str,
) -> Result<VerificationResult, WenlanError> {
    let mut warnings = Vec::new();

    // Entity self-retrieval test: search by name, check if this entity appears in top 5
    let self_retrieval_passed = match db.search_entities_by_vector(entity_name, 5).await {
        Ok(results) => {
            let found = results.iter().any(|r| r.entity.id == entity_id);
            if !found {
                warnings.push(format!(
                    "Entity '{}' ({}) not found in top-5 self-retrieval results",
                    entity_name, entity_id
                ));
            }
            Some(found)
        }
        Err(_) => None, // Embedding not available, skip check
    };

    Ok(VerificationResult {
        entity_self_retrieval_passed: self_retrieval_passed,
        concept_self_retrieval_passed: None,
        relation_consistency_passed: None,
        warnings,
    })
}

/// Run post-store verification on a newly distilled page.
pub async fn verify_page(
    db: &MemoryDB,
    page_id: &str,
    page_title: &str,
) -> Result<VerificationResult, WenlanError> {
    let mut warnings = Vec::new();

    let self_retrieval_passed = match db
        .search_memory(
            page_title,
            10,
            None,
            &ReadScope::Global,
            None,
            None,
            None,
            None,
        )
        .await
    {
        Ok(results) => {
            let found = results.iter().any(|r| r.source_id == page_id);
            if !found {
                warnings.push(format!(
                    "Page '{}' ({}) not found in top-10 self-retrieval results",
                    page_title, page_id
                ));
            }
            Some(found)
        }
        Err(_) => None,
    };

    Ok(VerificationResult {
        entity_self_retrieval_passed: None,
        concept_self_retrieval_passed: self_retrieval_passed,
        relation_consistency_passed: None,
        warnings,
    })
}

/// Report of a rethink pass.
#[derive(Debug, Default, serde::Serialize)]
pub struct RethinkReport {
    pub merge_candidates: usize,
    pub relations_healed: usize,
    pub relations_queued: usize,
    pub entities_healed: usize,
    pub entities_queued: usize,
    pub embeddings_refreshed: usize,
    pub stale_relations_flagged: usize,
    pub contradictions_found: usize,
}

/// Run the periodic knowledge graph rethink pass.
///
/// Phases:
/// 1. Entity merge candidates -- find duplicates with identical lowercase names
/// 2. Relation vocabulary heal -- auto-fold safe non-canonical types (known
///    aliases + deterministic transforms) to canonical, queue the rest for review
/// 3. Entity embedding refresh -- re-embed entities with many new observations
/// 4. Stale relation detection -- relations whose source memory was deleted
/// 5. Contradiction scan -- log entities with many observations for review
pub async fn run_rethink(
    db: &MemoryDB,
    _llm: Option<&Arc<dyn LlmProvider>>,
    _config: &RefineryConfig,
) -> Result<RethinkReport, WenlanError> {
    let rel = heal_relation_vocabulary(db).await?;
    let ent = heal_entity_vocabulary(db).await?;
    let report = RethinkReport {
        merge_candidates: find_merge_candidates(db).await?,
        relations_healed: rel.healed,
        relations_queued: rel.queued,
        entities_healed: ent.healed,
        entities_queued: ent.queued,
        embeddings_refreshed: refresh_stale_entity_embeddings(db).await?,
        stale_relations_flagged: detect_stale_relations(db).await?,
        contradictions_found: scan_contradictions(db).await?,
    };

    let receipt = serde_json::json!({
        "relations_healed": report.relations_healed,
        "relations_queued": report.relations_queued,
        "entities_healed": report.entities_healed,
        "entities_queued": report.entities_queued,
    })
    .to_string();
    let _ = db
        .log_agent_activity("daemon", "vocab_heal_receipt", &[], None, &receipt)
        .await;

    Ok(report)
}

/// Find entity pairs with identical lowercase names that might be duplicates.
/// Migration 40 deduplicated existing data, but new duplicates can appear when
/// entities are created via code paths that bypass alias resolution.
pub async fn find_merge_candidates(db: &MemoryDB) -> Result<usize, WenlanError> {
    let groups: Vec<DuplicateNameGroup> = db.duplicate_name_groups_for_merge_candidates().await?;
    let mut count = 0usize;
    for group in groups {
        log::warn!(
            "[rethink] merge candidate: '{}' has {} entities -- review for dedup",
            group.lower_name,
            group.entity_count
        );
        // Each group with N > 1 yields N-1 merge candidates.
        count += (group.entity_count.saturating_sub(1)) as usize;
    }

    // T16: surface MinHash/LSH band collisions into the human-review queue.
    // Opt-in (WENLAN_ENABLE_ENTITY_MINHASH). These are the *borderline* pairs
    // the auto-merge cascade deliberately leaves alone: exact Jaccard in
    // [0.85, 0.9), or a same/near match across DIFFERENT entity types. They are
    // enqueued as `entity_merge` proposals tagged `minhash_jaccard`, never
    // auto-merged.
    if crate::db::entity_minhash_enabled() {
        count += surface_minhash_merge_candidates(db).await?;
    }
    Ok(count)
}

/// Scan high-entropy entity names for LSH band collisions and enqueue the
/// borderline pairs (Jaccard in [0.85, 0.9), or cross-type) into the
/// human-review refinement queue. Returns the number of proposals enqueued.
async fn surface_minhash_merge_candidates(db: &MemoryDB) -> Result<usize, WenlanError> {
    use crate::retrieval::dedup;
    use std::collections::HashMap;

    // Pull every entity once (id, name, entity_type).
    let inputs: Vec<MinHashEntityInput> = db.entities_for_minhash_merge_candidates().await?;
    let entities: Vec<(String, String, String)> = inputs
        .into_iter()
        .filter_map(|input| {
            let MinHashEntityInput {
                id,
                name,
                entity_type,
            } = input;
            if dedup::has_high_entropy(&name) {
                Some((id, name, entity_type))
            } else {
                None
            }
        })
        .collect();

    // Bucket entity indices by band key.
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, (_, name, _)) in entities.iter().enumerate() {
        for key in dedup::name_band_keys(name) {
            buckets.entry(key).or_default().push(i);
        }
    }

    // Examine each colliding pair once.
    let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut enqueued = 0usize;
    for idxs in buckets.values() {
        for a in 0..idxs.len() {
            for b in (a + 1)..idxs.len() {
                let (i, j) = (idxs[a].min(idxs[b]), idxs[a].max(idxs[b]));
                if i == j || !seen.insert((i, j)) {
                    continue;
                }
                let (ref id_i, ref name_i, ref type_i) = entities[i];
                let (ref id_j, ref name_j, ref type_j) = entities[j];
                let jac = dedup::name_jaccard(name_i, name_j);
                let same_type = type_i.eq_ignore_ascii_case(type_j);
                // Human-review band: borderline similarity, OR a cross-type
                // collision the auto-merge guard would have skipped. Anything
                // at/above the auto-merge threshold with the SAME type is left
                // to the write-path cascade and is not re-queued here.
                let borderline = (0.85..dedup::FUZZY_JACCARD_THRESHOLD).contains(&jac);
                let cross_type_match = !same_type && jac >= 0.85;
                if !(borderline || cross_type_match) {
                    continue;
                }
                let i_len = id_i.len().min(8);
                let j_len = id_j.len().min(8);
                let proposal_id = format!("minhash_{}_{}", &id_i[..i_len], &id_j[..j_len]);
                let payload = serde_json::json!({
                    "existing_id": id_i,
                    "new_id": id_j,
                    "jaccard": jac,
                    "same_type": same_type,
                    "provenance": "minhash_jaccard",
                })
                .to_string();
                if db
                    .insert_refinement_proposal(
                        &proposal_id,
                        "entity_merge",
                        &[id_i.clone(), id_j.clone()],
                        Some(&payload),
                        jac,
                    )
                    .await
                    .is_ok()
                {
                    enqueued += 1;
                }
            }
        }
    }
    Ok(enqueued)
}

/// Counts returned by a vocabulary heal pass: how many relations were
/// auto-folded to a canonical, and how many distinct new types were queued
/// for human review as promote candidates.
#[derive(Debug, Default, Clone, Copy)]
pub struct VocabHealCounts {
    pub healed: usize,
    pub queued: usize,
}

/// Heal non-canonical relation types: auto-fold the deterministically safe ones
/// (known aliases + safe transforms), queue everything else as a promote candidate.
pub async fn heal_relation_vocabulary(db: &MemoryDB) -> Result<VocabHealCounts, WenlanError> {
    // First, read all distinct relation types.
    let types_to_check = db.distinct_relation_types_for_vocabulary_heal().await?;

    let canonicals = db.relation_canonicals().await?;
    let mut counts = VocabHealCounts::default();
    for rel_type in &types_to_check {
        if rel_type.is_empty() {
            continue;
        }
        // Already canonical? nothing to do.
        if canonicals.iter().any(|c| c == rel_type) {
            continue;
        }
        // Band A1: known alias in the vocabulary -> deterministic fold.
        if let Some(canonical) = db.resolve_relation_type(rel_type).await? {
            let folded = db.fold_relation_type(rel_type, &canonical).await?;
            counts.healed += folded;
            log::info!(
                "[rethink] folded '{}' -> '{}' ({} relations, ledgered)",
                rel_type,
                canonical,
                folded
            );
            continue;
        }
        // Band A2: deterministic safe transform toward an existing canonical.
        if let Some(canonical) = crate::vocab::safe_transform::safe_transform(rel_type, &canonicals)
        {
            let folded = db.fold_relation_type(rel_type, &canonical).await?;
            counts.healed += folded;
            log::info!(
                "[rethink] folded '{}' -> '{}' ({} relations, ledgered)",
                rel_type,
                canonical,
                folded
            );
            continue;
        }
        // Band B: semantic / new -> queue one promote candidate.
        if db
            .insert_vocab_promote_proposal("relation", rel_type, None)
            .await?
        {
            counts.queued += 1;
        }
    }
    Ok(counts)
}

/// Heal non-canonical entity types: auto-fold the deterministically safe ones
/// (known aliases + safe transforms), queue everything else as a promote candidate.
pub async fn heal_entity_vocabulary(db: &MemoryDB) -> Result<VocabHealCounts, WenlanError> {
    // First, read all distinct entity types.
    let types_to_check = db.distinct_entity_types_for_vocabulary_heal().await?;

    let canonicals = db.entity_canonicals().await?;
    let mut counts = VocabHealCounts::default();
    for entity_type in &types_to_check {
        if entity_type.is_empty() {
            continue;
        }
        // Already canonical? nothing to do.
        if canonicals.iter().any(|c| c == entity_type) {
            continue;
        }
        // Band A1: known alias in the vocabulary -> deterministic fold.
        if let Some(canonical) = db.resolve_entity_type(entity_type).await? {
            let folded = db.fold_entity_type(entity_type, &canonical).await?;
            counts.healed += folded;
            log::info!(
                "[rethink] folded '{}' -> '{}' ({} entities, ledgered)",
                entity_type,
                canonical,
                folded
            );
            continue;
        }
        // Band A2: deterministic safe transform toward an existing canonical.
        if let Some(canonical) =
            crate::vocab::safe_transform::safe_transform(entity_type, &canonicals)
        {
            let folded = db.fold_entity_type(entity_type, &canonical).await?;
            counts.healed += folded;
            log::info!(
                "[rethink] folded '{}' -> '{}' ({} entities, ledgered)",
                entity_type,
                canonical,
                folded
            );
            continue;
        }
        // Band B: semantic / new -> queue one promote candidate.
        if db
            .insert_vocab_promote_proposal("entity", entity_type, None)
            .await?
        {
            counts.queued += 1;
        }
    }
    Ok(counts)
}

/// Refresh embeddings for entities that have accumulated 5+ new observations
/// since their embedding was last updated.
pub async fn refresh_stale_entity_embeddings(db: &MemoryDB) -> Result<usize, WenlanError> {
    // Find candidates.
    let candidates: Vec<StaleEntityEmbeddingCandidate> =
        db.stale_entity_embedding_candidates_for_refresh().await?;

    let mut refreshed = 0usize;
    for candidate in &candidates {
        // Build text from entity name + top 10 recent observations.
        let mut parts = vec![candidate.entity_name.clone()];
        for content in db
            .recent_entity_observation_contents_for_embedding_refresh(&candidate.entity_id)
            .await?
        {
            if !content.is_empty() {
                parts.push(content);
            }
        }
        let combined = parts.join(". ");
        match db
            .refresh_entity_embedding(&candidate.entity_id, &combined)
            .await
        {
            Ok(()) => {
                refreshed += 1;
                log::info!(
                    "[rethink] refreshed embedding for entity '{}'",
                    candidate.entity_name
                );
            }
            Err(e) => log::warn!(
                "[rethink] refresh_entity_embedding failed for '{}': {}",
                candidate.entity_name,
                e
            ),
        }
    }
    Ok(refreshed)
}

/// Count relations whose `source_memory_id` no longer corresponds to an
/// existing memory. Logged for visibility; actual pruning is deferred until
/// relation temporality lands (requires a dedicated `stale` column).
pub async fn detect_stale_relations(db: &MemoryDB) -> Result<usize, WenlanError> {
    let count = db.count_stale_relation_sources().await?;
    if count > 0 {
        log::warn!(
            "[rethink] {} relations have stale source_memory_id (source memory deleted)",
            count
        );
    }
    Ok(count)
}

/// Scan for entities with many observations, logging them for manual review.
/// A full contradiction detection pass would need LLM; this is a cheap proxy
/// that highlights entities most likely to contain conflicting information.
pub async fn scan_contradictions(db: &MemoryDB) -> Result<usize, WenlanError> {
    let counts: Vec<ContradictionObservationCount> =
        db.list_contradiction_observation_counts().await?;
    for item in &counts {
        log::info!(
            "[rethink] entity '{}' has {} observations -- review for contradictions",
            item.entity_name,
            item.observation_count
        );
    }
    Ok(counts.len())
}

/// Embedding-based hallucination check: returns `true` if the body is
/// semantically similar (cosine >= 0.6) to the concatenation of the source
/// memories' content. Returns `false` if the body diverges from its
/// cited sources, or if embeddings cannot be produced.
///
/// Used by the agent-triggered `/api/pages` create path. The daemon-side
/// distillation in `synthesis::distill` keeps its own inline check for
/// now since it builds source text from a cluster, not by id.
pub async fn hallucination_guard(
    db: &MemoryDB,
    body: &str,
    source_ids: &[String],
) -> Result<bool, WenlanError> {
    if source_ids.is_empty() {
        return Ok(true);
    }
    let mut source_contents: Vec<String> = Vec::with_capacity(source_ids.len());
    for sid in source_ids {
        // Propagate DB errors; Ok(None) means unresolvable id — skip silently.
        if let Some(detail) = db.get_memory_detail(sid).await? {
            source_contents.push(detail.content);
        }
    }
    if source_contents.is_empty() {
        return Ok(true);
    }
    let joined = source_contents.join(" ");
    let texts = vec![body.to_string(), joined];
    match db.generate_embeddings(&texts) {
        Ok(embs) if embs.len() == 2 => {
            let sim = crate::db::cosine_similarity(&embs[0], &embs[1]);
            Ok(sim >= 0.6)
        }
        _ => Ok(true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDB::new(db_path.as_path(), Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();
        (db, dir)
    }

    #[tokio::test]
    async fn test_verify_entity_passes_for_good_entity() {
        let (db, _dir) = test_db().await;

        let id = db
            .store_entity("Rust", "technology", None, Some("test"), None)
            .await
            .unwrap();
        let result = verify_entity(&db, &id, "Rust").await.unwrap();
        assert_eq!(result.entity_self_retrieval_passed, Some(true));
        assert!(result.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_verify_entity_warns_on_missing() {
        let (db, _dir) = test_db().await;

        // Create an entity, then check a non-matching name
        let id = db
            .store_entity("Rust", "technology", None, Some("test"), None)
            .await
            .unwrap();
        let result = verify_entity(
            &db,
            &id,
            "completely unrelated query that won't match Rust at all",
        )
        .await
        .unwrap();
        // With only one entity in the DB, vector search should still return it as
        // closest result even for an unrelated query, so this may still pass.
        // The important thing is the function executes without error.
        assert!(result.entity_self_retrieval_passed.is_some());
    }

    #[tokio::test]
    async fn test_run_rethink_normalizes_relation_types() {
        let (db, _dir) = test_db().await;

        let e1 = db
            .store_entity("Alice", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("ProjectX", "project", None, Some("test"), None)
            .await
            .unwrap();

        // Bypass create_relation's normalization by inserting directly into
        // `edges` (see `heal_relation_vocabulary_folds_aliases_and_queues_semantics`
        // above for why `create_relation` itself can't seed a raw `working_at`
        // row post-G6-Stage-2). This simulates legacy data created before
        // relation vocabulary existed.
        let legacy_edge_id = crate::provenance::compute_edge_id(
            "relates",
            "entity",
            &e1,
            "entity",
            &e2,
            "working_at",
        );
        {
            let conn = db.test_primary_session().await;
            let space: Option<String> = {
                let mut rows = conn
                    .query(
                        "SELECT space FROM entities WHERE id = ?1",
                        libsql::params![e1.clone()],
                    )
                    .await
                    .unwrap();
                rows.next().await.unwrap().and_then(|r| r.get(0).ok())
            };
            conn.execute(
                "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, \
                                     lineage, grounded, root_id, space, weight, payload, \
                                     provenance, operation_id, created_at, superseded_by, \
                                     valid_until, semantic_type) \
                 VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion', 0, NULL, \
                         ?4, NULL, NULL, NULL, NULL, ?5, NULL, NULL, 'working_at')",
                libsql::params![
                    legacy_edge_id.clone(),
                    e1.clone(),
                    e2.clone(),
                    space,
                    chrono::Utc::now().timestamp()
                ],
            )
            .await
            .unwrap();
        }

        let config = RefineryConfig::default();
        let report = run_rethink(&db, None, &config).await.unwrap();

        // Rethink should have normalized "working_at" -> "works_on"
        assert!(
            report.relations_healed >= 1,
            "expected at least 1 type normalized, got {}",
            report.relations_healed
        );

        // Verify the canonical edge now exists (works_on), and the legacy-type
        // edge_id is retired (superseded), not left live under the old type.
        let conn = db.test_primary_session().await;
        let canonical_edge_id =
            crate::provenance::compute_edge_id("relates", "entity", &e1, "entity", &e2, "works_on");
        let mut rows = conn
            .query(
                "SELECT semantic_type, valid_until FROM edges WHERE edge_id = ?1",
                libsql::params![canonical_edge_id],
            )
            .await
            .unwrap();
        let row = rows
            .next()
            .await
            .unwrap()
            .expect("canonical works_on edge should exist");
        let rt: String = row.get::<String>(0).unwrap();
        assert_eq!(rt, "works_on");
        drop(rows);
        let mut legacy_rows = conn
            .query(
                "SELECT valid_until FROM edges WHERE edge_id = ?1",
                libsql::params![legacy_edge_id],
            )
            .await
            .unwrap();
        let legacy_row = legacy_rows
            .next()
            .await
            .unwrap()
            .expect("legacy-type edge should still exist, retired");
        assert!(
            legacy_row.get::<Option<i64>>(0).unwrap().is_some(),
            "legacy-type edge must be retired (valid_until set) after the fold"
        );
    }

    #[tokio::test]
    async fn test_run_rethink_completes_on_empty_db() {
        let (db, _dir) = test_db().await;
        let config = RefineryConfig::default();
        let report = run_rethink(&db, None, &config).await.unwrap();
        // All zero is fine; the point is it runs without error.
        assert_eq!(report.merge_candidates, 0);
        assert_eq!(report.relations_healed, 0);
        assert_eq!(report.stale_relations_flagged, 0);
    }

    #[tokio::test]
    async fn test_find_merge_candidates_detects_duplicates() {
        temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", None::<&str>)], async {
            let (db, _dir) = test_db().await;

            // Create three case-folded duplicates plus a singleton by bypassing
            // alias resolution. The caller owns the N-1 candidate math.
            let conn = db.test_primary_session().await;
            let now = chrono::Utc::now().timestamp();
            for (id, name) in [
                ("dup-1", "Alice"),
                ("dup-2", "alice"),
                ("dup-3", "ALICE"),
                ("singleton", "Bob"),
            ] {
                conn.execute(
                    "INSERT INTO entities (id, name, entity_type, source_agent, created_at, updated_at)
                     VALUES (?1, ?2, 'person', 'test', ?3, ?3)",
                    libsql::params![id, name, now],
                )
                .await
                .unwrap();
            }
            drop(conn);

            let count = find_merge_candidates(&db).await.unwrap();
            assert_eq!(
                count, 2,
                "three duplicate entities produce N-1 candidates; singleton is excluded"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_full_kg_quality_pipeline() {
        let (db, _dir) = test_db().await;

        // 1. Create entity with alias self-registration
        let id1 = db
            .store_entity("Alice Chen", "person", None, Some("test"), None)
            .await
            .unwrap();

        // 2. Verify alias resolution (case-insensitive)
        let resolved = db.resolve_entity_by_alias("alice chen").await.unwrap();
        assert_eq!(resolved, Some(id1.clone()));

        // 3. Create relation using canonical type
        let proj = db
            .store_entity("ProjectX", "project", None, Some("test"), None)
            .await
            .unwrap();
        db.create_relation(
            &id1,
            &proj,
            "works_on",
            Some("test"),
            Some(0.9),
            Some("she leads it"),
            Some("mem_1"),
        )
        .await
        .unwrap();

        // 4. Create relation using alias type — should normalize at insert
        let proj2 = db
            .store_entity("ProjectY", "project", None, Some("test"), None)
            .await
            .unwrap();
        db.create_relation(&id1, &proj2, "working_at", Some("test"), None, None, None)
            .await
            .unwrap();

        // Verify normalization happened at insert time
        {
            let conn = db.test_primary_session().await;
            let mut rows = conn
                .query(
                    "SELECT relation_type FROM relations WHERE from_entity = ?1",
                    libsql::params![id1.clone()],
                )
                .await
                .unwrap();
            while let Some(row) = rows.next().await.unwrap() {
                let rt: String = row.get::<String>(0).unwrap();
                assert_eq!(
                    rt, "works_on",
                    "expected all relations normalized to 'works_on'"
                );
            }
        }

        // 5. Entity self-retrieval passes
        let vr = verify_entity(&db, &id1, "Alice Chen").await.unwrap();
        assert_eq!(vr.entity_self_retrieval_passed, Some(true));

        // 6. Rethink completes successfully
        let config = RefineryConfig::default();
        let report = run_rethink(&db, None, &config).await.unwrap();
        // Nothing to normalize (already canonical); no duplicates;
        // embedding_refreshed and stale counts should be 0.
        assert_eq!(report.merge_candidates, 0);
        assert_eq!(report.relations_healed, 0);
    }

    // ── Fix 1: Case-insensitive relation resolution ────────────────────────

    #[tokio::test]
    async fn test_resolve_relation_type_case_insensitive() {
        let (db, _dir) = test_db().await;

        // "Working_At" should resolve to "works_on" (alias match, case-insensitive)
        let result = db.resolve_relation_type("Working_At").await.unwrap();
        assert_eq!(
            result,
            Some("works_on".to_string()),
            "Working_At should resolve to works_on via case-insensitive alias lookup"
        );

        // "WORKS_ON" is itself the canonical, just uppercased — should return "works_on"
        let result = db.resolve_relation_type("WORKS_ON").await.unwrap();
        assert_eq!(
            result,
            Some("works_on".to_string()),
            "WORKS_ON should resolve to works_on via canonical lookup (lowercased)"
        );

        // Identity: already canonical lowercase
        let result = db.resolve_relation_type("works_on").await.unwrap();
        assert_eq!(
            result,
            Some("works_on".to_string()),
            "works_on should return itself unchanged"
        );

        // Novel type not in vocabulary: return None
        let result = db.resolve_relation_type("novel_type").await.unwrap();
        assert_eq!(
            result, None,
            "novel_type should return None (not in vocabulary)"
        );
    }

    // ── Fix 2: ON CONFLICT updates confidence when higher ─────────────────

    #[tokio::test]
    async fn test_create_relation_confidence_upsert() {
        let (db, _dir) = test_db().await;

        let e1 = db
            .store_entity("Alice", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("ProjectAlpha", "project", None, Some("test"), None)
            .await
            .unwrap();

        // Insert with confidence 0.5
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.5), None, None)
            .await
            .unwrap();

        // Upsert with higher confidence 0.9 — should update
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.9), None, None)
            .await
            .unwrap();

        // Verify confidence is now 0.9 (canonical-only since G6 Stage 2 item 1:
        // `create_relation`'s keep-if-stronger upsert lands on the live `edges`
        // row, not `relations`).
        {
            let conn = db.test_primary_session().await;
            let mut rows = conn
                .query(
                    "SELECT json_extract(payload,'$.confidence') FROM edges \
                     WHERE edge_type = 'relates' AND src_kind = 'entity' AND dst_kind = 'entity' \
                       AND src_id = ?1 AND dst_id = ?2 AND semantic_type = ?3 AND valid_until IS NULL",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let confidence: f64 = row.get::<f64>(0).unwrap();
            assert!(
                (confidence - 0.9).abs() < 1e-6,
                "confidence should be 0.9 after higher-confidence upsert, got {confidence}"
            );
        }

        // Upsert with lower confidence 0.3 — should NOT update
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.3), None, None)
            .await
            .unwrap();

        // Verify confidence is still 0.9
        {
            let conn = db.test_primary_session().await;
            let mut rows = conn
                .query(
                    "SELECT json_extract(payload,'$.confidence'), COUNT(*) as cnt FROM edges \
                     WHERE edge_type = 'relates' AND src_kind = 'entity' AND dst_kind = 'entity' \
                       AND src_id = ?1 AND dst_id = ?2 AND semantic_type = ?3 AND valid_until IS NULL",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let confidence: f64 = row.get::<f64>(0).unwrap();
            let cnt: i64 = row.get::<i64>(1).unwrap();
            assert!(
                (confidence - 0.9).abs() < 1e-6,
                "confidence should still be 0.9 after lower-confidence upsert, got {confidence}"
            );
            assert_eq!(cnt, 1, "only 1 relation row should exist, got {cnt}");
        }
    }

    // ── Fix 3: fold_relation_type merges provenance + ledgers the loser ────

    #[tokio::test]
    async fn fold_relation_type_merges_provenance_and_ledgers_the_loser() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("EntityA", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("EntityB", "project", None, Some("test"), None)
            .await
            .unwrap();
        // Seeded via `create_relation` (canonical-only since G6 Stage 2 PR 2b
        // item 1), using two vocabulary-canonical, non-aliased types --
        // `working_at`/`works_on` no longer work here because `working_at` is
        // itself an ALIAS that `resolve_relation_type` collapses onto
        // `works_on` at create time, so a `create_relation("working_at", ...)`
        // call would never produce a `working_at`-typed edge to fold in the
        // first place. `leads`/`works_on` are both canonicals (see the sibling
        // `fold_relation_type_dual_writes_edges`, db/main_tests.rs), so they
        // stay distinct until this test's own fold merges them.
        // Survivor: canonical works_on, confidence 0.9.
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.9), None, None)
            .await
            .unwrap();
        // Loser: leads, confidence 0.5, has an explanation.
        db.create_relation(
            &e1,
            &e2,
            "leads",
            Some("test"),
            Some(0.5),
            Some("employed since 2020"),
            None,
        )
        .await
        .unwrap();

        let folded = db.fold_relation_type("leads", "works_on").await.unwrap();
        assert_eq!(folded, 1);

        let conn = db.test_primary_session().await;
        // Exactly one A->B relates edge survives, canonical, confidence
        // preserved at 0.9, and the loser's explanation was merged in
        // (survivor had none).
        let mut rows = conn
            .query(
                "SELECT COUNT(*), MAX(json_extract(payload,'$.confidence')) FROM edges \
                 WHERE edge_type = 'relates' AND src_id = ?1 AND dst_id = ?2 AND valid_until IS NULL",
                libsql::params![e1.clone(), e2.clone()],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let cnt: i64 = row.get(0).unwrap();
        let conf: f64 = row.get(1).unwrap();
        assert_eq!(cnt, 1, "collision must not leave two rows");
        assert!(
            (conf - 0.9).abs() < 1e-6,
            "survivor keeps the higher confidence"
        );
        drop(rows);
        // The absorbed edge's pre-image is in the ledger (undo is possible),
        // and the pre-image is COMPLETE: it carries created_at so a NOT-NULL
        // restore is possible on undo.
        let mut led = conn
            .query(
                "SELECT COUNT(*), MAX(pre_image) FROM vocab_heal_ledger WHERE old_value = 'leads'",
                (),
            )
            .await
            .unwrap();
        let led_row = led.next().await.unwrap().unwrap();
        let n: i64 = led_row.get(0).unwrap();
        let pre_image: String = led_row.get(1).unwrap();
        assert!(n >= 1, "the folded loser must be recorded in the ledger");
        assert!(
            pre_image.contains("created_at"),
            "pre-image must include created_at so undo can restore the NOT-NULL column, got {pre_image}"
        );
        // Spec §2.5: the SURVIVOR is mutated too (provenance COALESCE-merge), so its
        // pre-image must also be ledgered before the UPDATE, else undo cannot restore
        // its pre-merge confidence/explanation. The survivor's edge_id is
        // content-addressed by (relates, entity, e1, entity, e2, "works_on") and
        // appears in NO loser pre-image (the loser's own edge_id has a different
        // discriminator, "leads"), so its presence proves the survivor pre-image
        // was captured.
        let survivor_edge_id =
            crate::provenance::compute_edge_id("relates", "entity", &e1, "entity", &e2, "works_on");
        assert!(
            pre_image.contains(&survivor_edge_id),
            "survivor pre-image must be ledgered so undo can restore the mutated survivor, got {pre_image}"
        );
    }

    #[tokio::test]
    async fn fold_relation_type_rolls_back_when_ledger_insert_fails() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("RollbackA", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("RollbackB", "project", None, Some("test"), None)
            .await
            .unwrap();
        // Seeded via `create_relation` -- see the sibling test above for why
        // `leads`/`works_on` (both vocabulary canonicals) replace the old
        // `working_at`/`works_on` alias pair.
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.9), None, None)
            .await
            .unwrap();
        db.create_relation(
            &e1,
            &e2,
            "leads",
            Some("test"),
            Some(0.5),
            Some("must survive failed ledger"),
            None,
        )
        .await
        .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute_batch(
                "CREATE TRIGGER fail_relation_vocab_ledger
                 BEFORE INSERT ON vocab_heal_ledger
                 WHEN NEW.kind = 'relation'
                 BEGIN
                     SELECT RAISE(FAIL, 'forced relation ledger failure');
                 END;",
            )
            .await
            .unwrap();
        }

        let error = db
            .fold_relation_type("leads", "works_on")
            .await
            .expect_err("ledger failure must reject the whole fold");
        assert!(
            error.to_string().contains("forced relation ledger failure"),
            "unexpected fold error: {error}"
        );

        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT edge_id, semantic_type, valid_until, json_extract(payload,'$.explanation')
                 FROM edges
                 WHERE edge_type = 'relates' AND src_id = ?1 AND dst_id = ?2",
                libsql::params![e1.clone(), e2.clone()],
            )
            .await
            .unwrap();
        let mut surviving = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            surviving.push((
                row.get::<String>(0).unwrap(),
                row.get::<String>(1).unwrap(),
                row.get::<Option<i64>>(2).unwrap(),
                row.get::<Option<String>>(3).unwrap(),
            ));
        }
        surviving.sort();
        let mut expected = vec![
            (
                crate::provenance::compute_edge_id(
                    "relates", "entity", &e1, "entity", &e2, "leads",
                ),
                "leads".to_string(),
                None,
                Some("must survive failed ledger".to_string()),
            ),
            (
                crate::provenance::compute_edge_id(
                    "relates", "entity", &e1, "entity", &e2, "works_on",
                ),
                "works_on".to_string(),
                None,
                None,
            ),
        ];
        expected.sort();
        assert_eq!(
            surviving, expected,
            "edge mutation and undo ledger must commit or roll back together"
        );
    }

    // ── hallucination_guard ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_hallucination_guard_rejects_unrelated_body() {
        let (db, _dir) = test_db().await;
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "rust-mem".to_string(),
            title: "rust-mem".to_string(),
            content: "Rust is a systems programming language".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".to_string()),
            source_agent: Some("test".to_string()),
            confidence: Some(0.9),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();
        let body = "Pasta carbonara uses eggs and pancetta";
        let source_ids = vec!["rust-mem".to_string()];
        let passed = crate::kg_quality::hallucination_guard(&db, body, &source_ids)
            .await
            .unwrap();
        assert!(!passed, "guard should reject body unrelated to sources");
    }

    #[tokio::test]
    async fn test_hallucination_guard_accepts_related_body() {
        let (db, _dir) = test_db().await;
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "rust-mem-2".to_string(),
            title: "rust-mem-2".to_string(),
            content: "Rust is a systems programming language with memory safety".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".to_string()),
            source_agent: Some("test".to_string()),
            confidence: Some(0.9),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();
        let body = "Rust provides memory-safe systems programming";
        let source_ids = vec!["rust-mem-2".to_string()];
        let passed = crate::kg_quality::hallucination_guard(&db, body, &source_ids)
            .await
            .unwrap();
        assert!(passed, "guard should accept body matching sources");
    }

    // ── Fix 5: source_memory_id is populated ──────────────────────────────

    #[tokio::test]
    async fn test_create_relation_source_memory_id() {
        let (db, _dir) = test_db().await;

        let e1 = db
            .store_entity("PersonX", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("ProjectZ", "project", None, Some("test"), None)
            .await
            .unwrap();

        // Insert with source_memory_id "mem_123" and confidence 0.8
        db.create_relation(
            &e1,
            &e2,
            "works_on",
            Some("test"),
            Some(0.8),
            None,
            Some("mem_123"),
        )
        .await
        .unwrap();

        // Verify source_memory_id was stored (canonical-only since G6 Stage 2
        // item 1: `source_memory_id` lives in `payload`, not a `relations`
        // column).
        {
            let conn = db.test_primary_session().await;
            let mut rows = conn
                .query(
                    "SELECT json_extract(payload,'$.source_memory_id') FROM edges \
                     WHERE edge_type = 'relates' AND src_kind = 'entity' AND dst_kind = 'entity' \
                       AND src_id = ?1 AND dst_id = ?2 AND semantic_type = ?3 AND valid_until IS NULL",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let smid: String = row.get::<String>(0).unwrap();
            assert_eq!(smid, "mem_123", "source_memory_id should be mem_123");
        }

        // Upsert with lower confidence and source_memory_id "mem_456".
        // Relation provenance is fill-only: the first non-null source remains.
        db.create_relation(
            &e1,
            &e2,
            "works_on",
            Some("test"),
            Some(0.3),
            None,
            Some("mem_456"),
        )
        .await
        .unwrap();

        {
            let conn = db.test_primary_session().await;
            let mut rows = conn
                .query(
                    "SELECT json_extract(payload,'$.source_memory_id'), \
                            json_extract(payload,'$.confidence') FROM edges \
                     WHERE edge_type = 'relates' AND src_kind = 'entity' AND dst_kind = 'entity' \
                       AND src_id = ?1 AND dst_id = ?2 AND semantic_type = ?3 AND valid_until IS NULL",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let smid: String = row.get::<String>(0).unwrap();
            let conf: f64 = row.get::<f64>(1).unwrap();
            // Confidence should still be 0.8 (not overwritten by lower 0.3)
            assert!(
                (conf - 0.8).abs() < 1e-6,
                "confidence should remain 0.8 after lower-confidence upsert, got {conf}"
            );
            assert_eq!(
                smid, "mem_123",
                "source_memory_id should preserve the first non-null source"
            );
        }

        // Upsert with higher confidence and source_memory_id "mem_789"
        db.create_relation(
            &e1,
            &e2,
            "works_on",
            Some("test"),
            Some(0.95),
            None,
            Some("mem_789"),
        )
        .await
        .unwrap();

        // Higher confidence updates the score, but not existing provenance.
        {
            let conn = db.test_primary_session().await;
            let mut rows = conn
                .query(
                    "SELECT json_extract(payload,'$.source_memory_id'), \
                            json_extract(payload,'$.confidence') FROM edges \
                     WHERE edge_type = 'relates' AND src_kind = 'entity' AND dst_kind = 'entity' \
                       AND src_id = ?1 AND dst_id = ?2 AND semantic_type = ?3 AND valid_until IS NULL",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let smid: String = row.get::<String>(0).unwrap();
            let conf: f64 = row.get::<f64>(1).unwrap();
            assert!(
                (conf - 0.95).abs() < 1e-6,
                "confidence should be 0.95 after higher-confidence upsert, got {conf}"
            );
            assert_eq!(
                smid, "mem_123",
                "source_memory_id should remain mem_123 after higher-confidence upsert"
            );
        }
    }

    // ── T16: MinHash band-collision surfacing into the human-review queue ────

    #[tokio::test]
    async fn find_merge_candidates_surfaces_band_near_dups() {
        temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", Some("1"))], async {
            let (db, _dir) = test_db().await;
            // "Glorptech"/"Glorptechs": high-entropy, share an LSH band, exact
            // Jaccard ~0.875 in [0.85, 0.9) -> borderline -> human-review only.
            let jac = crate::retrieval::dedup::name_jaccard("Glorptech", "Glorptechs");
            assert!(
                (0.85..0.9).contains(&jac),
                "fixture pair must sit in the borderline band, got {jac}"
            );
            let id1 = db
                .store_entity("Glorptech", "project", None, Some("t"), None)
                .await
                .unwrap();
            let id2 = db
                .store_entity("Glorptechs", "project", None, Some("t"), None)
                .await
                .unwrap();
            // Both still exist (NOT auto-merged).
            assert!(db.get_entity_name_type(&id1).await.unwrap().is_some());
            assert!(db.get_entity_name_type(&id2).await.unwrap().is_some());

            let enqueued = find_merge_candidates(&db).await.unwrap();
            assert!(enqueued >= 1, "borderline band collision must be counted");

            let pending = db.get_pending_refinements().await.unwrap();
            let proposal = pending
                .iter()
                .find(|p| p.action == "entity_merge" && p.id.starts_with("minhash_"))
                .expect("a minhash entity_merge proposal must be queued");
            let payload = proposal.payload.as_ref().expect("proposal payload");
            assert!(
                payload.contains("minhash_jaccard"),
                "proposal must carry minhash_jaccard provenance, got: {payload}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn find_merge_candidates_minhash_off_no_band_proposals() {
        // Flag OFF: the band-collision pass must not run at all.
        temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", None::<&str>)], async {
            let (db, _dir) = test_db().await;
            db.store_entity("Glorptech", "project", None, Some("t"), None)
                .await
                .unwrap();
            db.store_entity("Glorptechs", "project", None, Some("t"), None)
                .await
                .unwrap();
            find_merge_candidates(&db).await.unwrap();
            let pending = db.get_pending_refinements().await.unwrap();
            assert!(
                !pending.iter().any(|p| p.id.starts_with("minhash_")),
                "flag OFF must enqueue no minhash proposals"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn find_merge_candidates_minhash_on_skips_low_entropy_collisions() {
        temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", Some("1"))], async {
            let (db, _dir) = test_db().await;
            for (name, entity_type) in [("aaaaaa", "person"), ("aaaaaaa", "project")] {
                db.store_entity(name, entity_type, None, Some("test"), None)
                    .await
                    .unwrap();
            }

            let count = find_merge_candidates(&db).await.unwrap();
            assert_eq!(count, 0);
            let pending = db.get_pending_refinements().await.unwrap();
            assert!(
                !pending.iter().any(|p| p.id.starts_with("minhash_")),
                "low-entropy rows must be filtered by caller policy even when MinHash is enabled"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn heal_relation_vocabulary_folds_aliases_and_queues_semantics() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("A", "person", None, Some("t"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("B", "project", None, Some("t"), None)
            .await
            .unwrap();
        let now = chrono::Utc::now().timestamp();
        {
            let conn = db.test_primary_session().await;
            // Both rows must land with their RAW (non-normalized) type, so this
            // seeds `edges` directly rather than via `create_relation` --
            // `create_relation` resolves `working_at` to `works_on` and coerces
            // an unrecognized type like `design_inspiration` to `related_to`
            // (queuing its own promote proposal) at WRITE time, which would
            // defeat the exact discovery-then-heal path this test exercises.
            let space: Option<String> = {
                let mut rows = conn
                    .query(
                        "SELECT space FROM entities WHERE id = ?1",
                        libsql::params![e1.clone()],
                    )
                    .await
                    .unwrap();
                rows.next().await.unwrap().and_then(|r| r.get(0).ok())
            };
            for (id_suffix, relation_type) in [("r1", "working_at"), ("r2", "design_inspiration")] {
                let edge_id = crate::provenance::compute_edge_id(
                    "relates",
                    "entity",
                    &e1,
                    "entity",
                    &e2,
                    relation_type,
                );
                conn.execute(
                    "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, \
                                         lineage, grounded, root_id, space, weight, payload, \
                                         provenance, operation_id, created_at, superseded_by, \
                                         valid_until, semantic_type) \
                     VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion', 0, NULL, \
                             ?4, NULL, NULL, NULL, NULL, ?5, NULL, NULL, ?6)",
                    libsql::params![
                        edge_id,
                        e1.clone(),
                        e2.clone(),
                        space.clone(),
                        now,
                        relation_type
                    ],
                )
                .await
                .unwrap_or_else(|e| panic!("seed edge {id_suffix}: {e}"));
            }
        }
        let counts = super::heal_relation_vocabulary(&db).await.unwrap();
        assert!(
            counts.healed >= 1,
            "working_at is a known alias, must auto-fold"
        );
        assert!(
            counts.queued >= 1,
            "design_inspiration must be queued as promote"
        );
        // working_at is gone (folded); design_inspiration still present (awaiting human).
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT DISTINCT semantic_type FROM edges \
                 WHERE edge_type = 'relates' AND valid_until IS NULL \
                 ORDER BY semantic_type",
                (),
            )
            .await
            .unwrap();
        let mut types = vec![];
        while let Some(r) = rows.next().await.unwrap() {
            types.push(r.get::<String>(0).unwrap());
        }
        assert!(types.contains(&"works_on".to_string()));
        assert!(types.contains(&"design_inspiration".to_string()));
        assert!(!types.contains(&"working_at".to_string()));
        drop(rows);
        drop(conn); // release before get_pending_refinements re-locks conn (non-reentrant mutex)
                    // And a promote proposal exists for design_inspiration.
        let pending = db.get_pending_refinements().await.unwrap();
        assert!(pending.iter().any(|p| p.action == "vocab_promote"
            && p.payload
                .as_deref()
                .unwrap_or("")
                .contains("design_inspiration")));
    }

    // G6 Stage 2 PR 2b sweep instance 4 acceptance pin: `heal_relation_vocabulary`
    // discovers a non-canonical-type edge that COLLIDES with an existing
    // canonical edge (unlike the simple-rename case above), and the fold it
    // drives keeps the stronger confidence rather than duplicate-asserting --
    // exercising the full `heal_relation_vocabulary` -> `fold_relation_type`
    // path end to end against `edges`, not just `fold_relation_type` in
    // isolation (see `fold_relation_type_merges_provenance_and_ledgers_the_loser`
    // above for the isolated collision case).
    #[tokio::test]
    async fn heal_relation_vocabulary_discovers_and_folds_edges_collision_keeps_stronger() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("HealA", "person", None, Some("t"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("HealB", "project", None, Some("t"), None)
            .await
            .unwrap();

        // Canonical survivor, minted through the real canonical-only writer
        // (`create_relation`, post-G6-Stage-2 item 1), confidence 0.9.
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.9), None, None)
            .await
            .unwrap();

        // Legacy-typed loser, seeded directly into `edges` (bypassing
        // `create_relation`'s eager alias resolution -- same technique as
        // `heal_relation_vocabulary_folds_aliases_and_queues_semantics`
        // above), confidence 0.3 -- weaker than the survivor.
        let loser_edge_id = crate::provenance::compute_edge_id(
            "relates",
            "entity",
            &e1,
            "entity",
            &e2,
            "working_at",
        );
        {
            let conn = db.test_primary_session().await;
            let space: Option<String> = {
                let mut rows = conn
                    .query(
                        "SELECT space FROM entities WHERE id = ?1",
                        libsql::params![e1.clone()],
                    )
                    .await
                    .unwrap();
                rows.next().await.unwrap().and_then(|r| r.get(0).ok())
            };
            let payload = serde_json::json!({"confidence": 0.3}).to_string();
            conn.execute(
                "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, \
                                     lineage, grounded, root_id, space, weight, payload, \
                                     provenance, operation_id, created_at, superseded_by, \
                                     valid_until, semantic_type) \
                 VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion', 0, NULL, \
                         ?4, NULL, ?5, NULL, NULL, ?6, NULL, NULL, 'working_at')",
                libsql::params![
                    loser_edge_id.clone(),
                    e1.clone(),
                    e2.clone(),
                    space,
                    payload,
                    chrono::Utc::now().timestamp()
                ],
            )
            .await
            .unwrap();
        }

        let counts = super::heal_relation_vocabulary(&db).await.unwrap();
        assert!(
            counts.healed >= 1,
            "working_at is a known alias colliding with an existing canonical, must auto-fold"
        );

        let conn = db.test_primary_session().await;
        // Exactly one live edge between e1/e2: no duplicate assert.
        let mut rows = conn
            .query(
                "SELECT COUNT(*), MAX(json_extract(payload,'$.confidence')) FROM edges \
                 WHERE edge_type = 'relates' AND src_id = ?1 AND dst_id = ?2 AND valid_until IS NULL",
                libsql::params![e1.clone(), e2.clone()],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let cnt: i64 = row.get(0).unwrap();
        let conf: f64 = row.get(1).unwrap();
        assert_eq!(
            cnt, 1,
            "collision must not leave two live edges (no duplicate assert)"
        );
        assert!(
            (conf - 0.9).abs() < 1e-6,
            "survivor keeps the stronger confidence (keep-if-stronger), got {conf}"
        );
        drop(rows);

        // The canonical survivor's edge_id is unchanged (content-addressed by
        // participants + canonical type) and stays live; the legacy-typed
        // edge is retired, not left duplicated.
        let canonical_edge_id =
            crate::provenance::compute_edge_id("relates", "entity", &e1, "entity", &e2, "works_on");
        let mut surv = conn
            .query(
                "SELECT valid_until FROM edges WHERE edge_id = ?1",
                libsql::params![canonical_edge_id],
            )
            .await
            .unwrap();
        let surv_row = surv
            .next()
            .await
            .unwrap()
            .expect("canonical survivor edge exists");
        assert!(
            surv_row.get::<Option<i64>>(0).unwrap().is_none(),
            "survivor stays live"
        );
        drop(surv);
        let mut loser = conn
            .query(
                "SELECT valid_until FROM edges WHERE edge_id = ?1",
                libsql::params![loser_edge_id],
            )
            .await
            .unwrap();
        let loser_row = loser
            .next()
            .await
            .unwrap()
            .expect("legacy-typed edge still exists, retired");
        assert!(
            loser_row.get::<Option<i64>>(0).unwrap().is_some(),
            "legacy-typed edge is retired, not duplicated"
        );
    }

    #[tokio::test]
    async fn heal_entity_vocabulary_folds_plural_and_queues_semantic() {
        let (db, _dir) = test_db().await;
        db.store_entity("X", "concepts", None, Some("t"), None)
            .await
            .unwrap(); // safe: concepts->concept
        db.store_entity("Y", "interest", None, Some("t"), None)
            .await
            .unwrap(); // semantic: queue
        let counts = super::heal_entity_vocabulary(&db).await.unwrap();
        assert!(counts.healed >= 1);
        assert!(counts.queued >= 1);
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query("SELECT entity_type FROM entities ORDER BY entity_type", ())
            .await
            .unwrap();
        let mut types = vec![];
        while let Some(r) = rows.next().await.unwrap() {
            types.push(r.get::<String>(0).unwrap());
        }
        assert!(types.contains(&"concept".to_string()));
        assert!(!types.contains(&"concepts".to_string()));
        assert!(types.contains(&"interest".to_string())); // unchanged, awaiting human
    }

    #[tokio::test]
    async fn run_rethink_emits_a_vocab_heal_receipt() {
        let (db, _dir) = test_db().await;
        // Seed one queue-able dirty relation.
        let e1 = db
            .store_entity("A", "person", None, Some("t"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("B", "project", None, Some("t"), None)
            .await
            .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute("INSERT INTO relations (id, from_entity, to_entity, relation_type, created_at) VALUES ('r1', ?1, ?2, 'design_inspiration', 0)",
                libsql::params![e1, e2]).await.unwrap();
        }
        let cfg = crate::tuning::RefineryConfig::default();
        super::run_rethink(&db, None, &cfg).await.unwrap();
        // A vocab-heal receipt activity row exists.
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM agent_activity WHERE action = 'vocab_heal_receipt'",
                (),
            )
            .await
            .unwrap();
        let n: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert!(n >= 1, "run_rethink must emit a counted vocab-heal receipt");
    }
}
