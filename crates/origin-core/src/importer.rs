use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use chrono::NaiveDate;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::classify::ClassificationResult;
use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::extract::{ExtractedEntity, KgExtractionResult};
use crate::llm_provider::LlmProvider;
use crate::sources::{compute_effective_confidence, RawDocument};

/// Pre-compiled regexes for prefix stripping (compiled once).
/// Matches: [2025-06-15] [type] - ... or [2025-06-15] - ...
static DATE_PREFIX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[(\d{4}-\d{2}-\d{2})\]\s*(?:\[(\w+)\]\s*)?-\s*").unwrap());
/// Matches: [unknown] [type] - ... or [unknown] - ...
static UNKNOWN_PREFIX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[unknown\]\s*(?:\[(\w+)\]\s*)?-\s*").unwrap());
static NUMBERED_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d+\.\s+").unwrap());
/// Standalone type tag at start of line: [type] - ...
static TYPE_PREFIX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[(\w+)\]\s*-\s*").unwrap());

/// Maximum raw input size in bytes (512 KB).
pub const MAX_INPUT_BYTES: usize = 512_000;

/// Maximum number of parsed memories allowed per import.
pub const MAX_MEMORIES: usize = 500;

/// Minimum character count for a memory to be kept (filters noise).
const MIN_CONTENT_CHARS: usize = 20;

/// ChatGPT memory export section headers — not actual memories.
/// Claude and ChatGPT memory export section headers — not actual memories.
const SECTION_HEADERS: &[&str] = &[
    // Claude section headers
    "earlier context",
    "recent months",
    "brief history",
    "top of mind",
    "personal context",
    "work context",
    "communication style",
    "communication & interaction style",
    "communication & interaction",
    "communication & interaction preferences",
    "behavioral patterns",
    "general context",
    "professional context",
    "learning & interests",
    "preferences & style",
    "key relationships",
    "background",
    // ChatGPT section headers
    "about me",
    "helpful response style",
    "what i do",
    "my preferences",
];

/// Valid memory type tags that can appear in import format.
/// Includes legacy types (correction, custom, recap) for backward compat — they map to "fact".
const VALID_TYPES: &[&str] = &[
    "identity",
    "preference",
    "decision",
    "fact",
    "goal",
    "correction",
    "custom",
    "recap", // legacy — accepted but stored as "fact"
];

/// A single memory extracted from raw import text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedMemory {
    pub content: String,
    /// Unix timestamp extracted from date prefix, if present.
    pub extracted_date: Option<i64>,
    /// Memory type extracted from `[type]` tag, if present.
    pub memory_type: Option<String>,
}

/// Validate that raw input does not exceed the size limit.
pub fn validate_input(input: &str) -> Result<(), OriginError> {
    if input.len() > MAX_INPUT_BYTES {
        return Err(OriginError::Generic(format!(
            "Input size ({} bytes) exceeds maximum allowed ({} bytes)",
            input.len(),
            MAX_INPUT_BYTES
        )));
    }
    Ok(())
}

/// Validate that the parsed memory count does not exceed the limit.
pub fn validate_parsed_count(count: usize) -> Result<(), OriginError> {
    if count > MAX_MEMORIES {
        return Err(OriginError::Generic(format!(
            "Parsed {} memories, which exceeds the maximum of {}",
            count, MAX_MEMORIES
        )));
    }
    Ok(())
}

/// Returns true if a line is a visual separator (e.g. `---`, `===`, `***`).
fn is_separator(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }
    // A separator is a line composed entirely of repeated `-`, `=`, or `*`
    // with at least 3 characters.
    if trimmed.chars().count() >= 3 {
        let first = trimmed.chars().next().unwrap();
        if matches!(first, '-' | '=' | '*') && trimmed.chars().all(|c| c == first) {
            return true;
        }
    }
    false
}

/// Detect whether paragraph mode should be used.
///
/// Paragraph mode is active when:
/// 1. The text contains at least one `\n\n` boundary, AND
/// 2. The majority of double-newline-separated blocks contain internal
///    single newlines (i.e., they are multi-line paragraphs, not just
///    single lines separated by blank lines).
fn is_paragraph_mode(text: &str) -> bool {
    if !text.contains("\n\n") {
        return false;
    }
    let blocks: Vec<&str> = text.split("\n\n").collect();
    if blocks.len() < 2 {
        return false;
    }
    let multi_line_blocks = blocks
        .iter()
        .filter(|b| {
            let trimmed = b.trim();
            !trimmed.is_empty() && trimmed.contains('\n')
        })
        .count();
    // "most blocks span multiple lines" — majority threshold
    let non_empty_blocks = blocks.iter().filter(|b| !b.trim().is_empty()).count();
    non_empty_blocks > 0 && multi_line_blocks * 2 >= non_empty_blocks
}

/// Strip common list/date prefixes from a line, returning the cleaned
/// content, an optional extracted date, and an optional memory type.
fn strip_prefix(line: &str) -> (String, Option<i64>, Option<String>) {
    let trimmed = line.trim();

    // Date prefix with optional type tag: [2025-06-15] [type] - ... or [2025-06-15] - ...
    if let Some(caps) = DATE_PREFIX_RE.captures(trimmed) {
        let date_str = &caps[1];
        let rest = &trimmed[caps.get(0).unwrap().end()..];
        let timestamp = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .ok()
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|dt| dt.and_utc().timestamp());
        let mem_type = caps
            .get(2)
            .map(|m| m.as_str().to_lowercase())
            .filter(|t| VALID_TYPES.contains(&t.as_str()));
        return (rest.trim().to_string(), timestamp, mem_type);
    }

    // [unknown] with optional type tag: [unknown] [type] - ...
    if let Some(caps) = UNKNOWN_PREFIX_RE.captures(trimmed) {
        let rest = &trimmed[caps.get(0).unwrap().end()..];
        let mem_type = caps
            .get(1)
            .map(|m| m.as_str().to_lowercase())
            .filter(|t| VALID_TYPES.contains(&t.as_str()));
        return (rest.trim().to_string(), None, mem_type);
    }

    // Standalone type tag: [type] - ...
    if let Some(caps) = TYPE_PREFIX_RE.captures(trimmed) {
        let tag = caps[1].to_lowercase();
        if VALID_TYPES.contains(&tag.as_str()) {
            let rest = &trimmed[caps.get(0).unwrap().end()..];
            return (rest.trim().to_string(), None, Some(tag));
        }
    }

    // Numbered list: 1. , 2. , etc.
    if let Some(m) = NUMBERED_PREFIX_RE.find(trimmed) {
        return (trimmed[m.end()..].trim().to_string(), None, None);
    }

    // Bullet prefixes: - , * , bullet char
    if let Some(rest) = trimmed.strip_prefix("- ") {
        return (rest.trim().to_string(), None, None);
    }
    if let Some(rest) = trimmed.strip_prefix("* ") {
        return (rest.trim().to_string(), None, None);
    }
    if let Some(rest) = trimmed.strip_prefix('\u{2022}') {
        return (rest.trim().to_string(), None, None);
    }

    (trimmed.to_string(), None, None)
}

/// Parse raw text into a list of individual memories.
///
/// Supports multiple formats:
/// - Plain lines (one memory per line)
/// - Paragraph mode (double-newline separated blocks)
/// - ChatGPT-style date prefixes: `[2025-06-15] - ...`
/// - `[unknown] - ...` prefixes
/// - Bullet lists (`- `, `* `, `\u{2022} `)
/// - Numbered lists (`1. `, `2. `, etc.)
///
/// Skips empty lines, separators (`---`, `===`, `***`), and entries shorter
/// than 20 characters (filters section headers). Deduplicates exact matches within the batch.
pub fn parse_memories(raw_text: &str) -> Vec<ParsedMemory> {
    let paragraph_mode = is_paragraph_mode(raw_text);

    let raw_blocks: Vec<String> = if paragraph_mode {
        // Split on double newlines; collapse internal newlines to spaces.
        raw_text
            .split("\n\n")
            .map(|block| {
                block
                    .lines()
                    .map(|l| l.trim())
                    .filter(|l| !l.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    } else {
        // One entry per line.
        raw_text.lines().map(|l| l.to_string()).collect()
    };

    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for block in &raw_blocks {
        let trimmed = block.trim();

        // Skip empty and separator lines
        if is_separator(trimmed) {
            continue;
        }

        let (content, extracted_date, memory_type) = strip_prefix(trimmed);

        // Skip entries shorter than MIN_CONTENT_CHARS
        if content.chars().count() < MIN_CONTENT_CHARS {
            continue;
        }

        // Skip known section headers (e.g. "Earlier context", "Top of mind")
        if SECTION_HEADERS.contains(&content.to_lowercase().as_str()) {
            continue;
        }

        // Deduplicate exact matches
        if !seen.insert(content.clone()) {
            continue;
        }

        result.push(ParsedMemory {
            content,
            extracted_date,
            memory_type,
        });
    }

    result
}

/// Number of memories per LLM batch.
const BATCH_SIZE: usize = 8;

/// Result of an import operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    pub imported: usize,
    pub skipped: usize,
    pub breakdown: HashMap<String, usize>,
    pub entities_created: usize,
    pub observations_added: usize,
    pub relations_created: usize,
    pub batch_id: String,
}

/// Find indices of memories that already exist in the DB by exact content match.
/// Uses content-based matching instead of vector similarity because contextual
/// enrichment in upsert_documents changes the stored embedding prefix.
pub async fn find_duplicates(db: &MemoryDB, memories: &[ParsedMemory]) -> HashSet<usize> {
    let mut duplicates = HashSet::new();
    for (i, memory) in memories.iter().enumerate() {
        if let Ok(true) = db.has_memory_content(&memory.content).await {
            duplicates.insert(i);
        }
    }
    duplicates
}

/// Returns (entity_id, was_newly_created).
pub async fn resolve_or_create_entity(
    db: &MemoryDB,
    entity_cache: &mut HashMap<String, String>,
    entity: &ExtractedEntity,
    source: &str,
) -> Result<(String, bool), OriginError> {
    let name_lower = entity.name.to_lowercase();

    // Step 0: In-batch cache (case-insensitive)
    if let Some(id) = entity_cache.get(&name_lower) {
        return Ok((id.clone(), false));
    }

    // Step 1: Alias lookup (exact, case-insensitive)
    if let Some(id) = db.resolve_entity_by_alias(&name_lower).await? {
        entity_cache.insert(name_lower, id.clone());
        return Ok((id, false));
    }

    // Step 2: Entity name lookup (exact, case-insensitive)
    if let Ok(results) = db.search_entities_by_name(&entity.name).await {
        if let Some(existing) = results.first() {
            entity_cache.insert(name_lower.clone(), existing.id.clone());
            db.add_entity_alias(&name_lower, &existing.id, "auto")
                .await
                .ok();
            return Ok((existing.id.clone(), false));
        }
    }

    // Step 3: Vector similarity (distance < 0.1)
    if let Ok(results) = db.search_entities_by_vector(&entity.name, 1).await {
        if let Some(result) = results.first() {
            if result.distance < 0.1 {
                entity_cache.insert(name_lower.clone(), result.entity.id.clone());
                db.add_entity_alias(&name_lower, &result.entity.id, "auto")
                    .await
                    .ok();
                return Ok((result.entity.id.clone(), false));
            }
        }
    }

    // Step 4: Create new entity + self-alias (store_entity auto-creates alias now)
    let id = db
        .store_entity(&entity.name, &entity.entity_type, None, Some(source), None)
        .await?;
    entity_cache.insert(name_lower, id.clone());
    Ok((id, true))
}

/// Import memories without LLM enrichment (for testing and fast-path).
/// Fast import path: stores memories immediately with default classification ("fact").
/// No LLM inference — no GPU heat. Batches all documents into a single upsert call
/// for efficient embedding generation. LLM classification can run later via refinery.
pub async fn import_memories_no_llm(
    db: &MemoryDB,
    raw_text: &str,
    source: &str,
    _label: Option<&str>,
    confidence_cfg: &crate::tuning::ConfidenceConfig,
) -> Result<ImportResult, OriginError> {
    validate_input(raw_text)?;
    let memories = parse_memories(raw_text);
    validate_parsed_count(memories.len())?;

    let duplicates = find_duplicates(db, &memories).await;

    let batch_id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().timestamp();
    let mut imported = 0usize;
    let mut skipped = 0usize;
    let mut breakdown: HashMap<String, usize> = HashMap::new();

    // Collect all non-duplicate docs for a single batch upsert
    let mut docs: Vec<RawDocument> = Vec::new();

    for (i, memory) in memories.iter().enumerate() {
        if duplicates.contains(&i) {
            skipped += 1;
            continue;
        }

        // memory_type starts as None (unclassified) — refinery reclassifies in background
        let confidence = compute_effective_confidence(None, None, "review", None, confidence_cfg);

        let title: String = memory.content.chars().take(80).collect();
        let source_id = format!("import_{}_{}", batch_id, i);

        docs.push(RawDocument {
            source: "memory".to_string(),
            source_id,
            title,
            summary: None,
            content: memory.content.clone(),
            url: None,
            last_modified: memory.extracted_date.unwrap_or(now),
            metadata: HashMap::from([
                ("import_source".to_string(), source.to_string()),
                ("import_batch".to_string(), batch_id.clone()),
            ]),
            memory_type: None,
            domain: None,
            source_agent: Some(source.to_string()),
            confidence: Some(confidence),
            confirmed: None,
            supersedes: None,
            pending_revision: false,
            ..Default::default()
        });

        imported += 1;
        *breakdown.entry("unclassified".to_string()).or_insert(0) += 1;
    }

    // Single batch upsert — one embedding generation pass for all docs
    if !docs.is_empty() {
        db.upsert_documents(docs).await?;
    }

    Ok(ImportResult {
        imported,
        skipped,
        breakdown,
        entities_created: 0,
        observations_added: 0,
        relations_created: 0,
        batch_id,
    })
}

/// Intermediate state between the DB-dependent and LLM-dependent import phases.
/// Produced by `import_phase1_prepare`, consumed by `import_phase3_store`.
pub struct ImportPrepared {
    pub memories: Vec<ParsedMemory>,
    pub texts: Vec<String>,
    pub duplicates: HashSet<usize>,
}

/// Phase 1: Parse, embed, and deduplicate. Requires DB access (short).
pub async fn import_phase1_prepare(
    db: &MemoryDB,
    raw_text: &str,
) -> Result<ImportPrepared, OriginError> {
    validate_input(raw_text)?;
    let memories = parse_memories(raw_text);
    validate_parsed_count(memories.len())?;

    let texts: Vec<String> = memories.iter().map(|m| m.content.clone()).collect();
    let duplicates = find_duplicates(db, &memories).await;

    Ok(ImportPrepared {
        memories,
        texts,
        duplicates,
    })
}

/// Phase 2: LLM classification + KG extraction. Does NOT require DB access.
/// This is the slow phase (45-60s with LLM inference) and should run
/// outside any RwLock guard to avoid blocking writers.
pub async fn import_phase2_llm(
    llm: Option<&Arc<dyn LlmProvider>>,
    prepared: &ImportPrepared,
    prompts: &crate::prompts::PromptRegistry,
) -> (Vec<ClassificationResult>, Vec<KgExtractionResult>) {
    let ImportPrepared {
        texts,
        duplicates,
        memories,
        ..
    } = prepared;

    // Collect non-duplicate texts for LLM processing
    let non_dup_texts: Vec<String> = texts
        .iter()
        .enumerate()
        .filter(|(i, _)| !duplicates.contains(i))
        .map(|(_, t)| t.clone())
        .collect();
    let non_dup_indices: Vec<usize> = (0..texts.len())
        .filter(|i| !duplicates.contains(i))
        .collect();

    // Batch classification via LLM (if available)
    let non_dup_classifications: Vec<ClassificationResult> = if let Some(llm) = llm {
        let mut results = Vec::with_capacity(non_dup_texts.len());
        for batch in non_dup_texts.chunks(BATCH_SIZE) {
            let numbered = batch
                .iter()
                .enumerate()
                .map(|(i, m)| format!("{}. {}", i + 1, m))
                .collect::<Vec<_>>()
                .join("\n");

            let response = llm
                .generate(crate::llm_provider::LlmRequest {
                    system_prompt: Some(prompts.batch_classify.clone()),
                    user_prompt: numbered,
                    max_tokens: 2048,
                    temperature: 0.3,
                    label: None,
                    timeout_secs: None,
                })
                .await;

            match response {
                Ok(output) => {
                    let batch_results =
                        crate::classify::parse_classification_response(&output, batch.len());
                    results.extend(batch_results);
                }
                Err(e) => {
                    log::warn!("[importer] LLM classify batch failed: {e}");
                    results.extend(vec![ClassificationResult::default(); batch.len()]);
                }
            }
        }
        results
    } else {
        non_dup_texts
            .iter()
            .map(|_| ClassificationResult::default())
            .collect()
    };

    // Map non-dup classifications back to full-index classifications
    let mut classifications: Vec<ClassificationResult> =
        vec![ClassificationResult::default(); texts.len()];
    for (slot, &orig_idx) in non_dup_indices.iter().enumerate() {
        if let Some(c) = non_dup_classifications.get(slot) {
            classifications[orig_idx] = c.clone();
        }
    }

    // KG extraction via LLM (only for identity/preference/fact)
    let kg_eligible: Vec<(usize, String)> = memories
        .iter()
        .enumerate()
        .filter(|(i, _)| !duplicates.contains(i))
        .filter(|(i, _)| {
            let mt = classifications
                .get(*i)
                .map(|c| c.memory_type.as_str())
                .unwrap_or("fact");
            matches!(mt, "identity" | "preference" | "fact")
        })
        .map(|(i, m)| (i, m.content.clone()))
        .collect();

    let kg_results: Vec<KgExtractionResult> = if let Some(llm) = llm {
        if !kg_eligible.is_empty() {
            let mut results = Vec::new();
            for batch in kg_eligible.chunks(BATCH_SIZE) {
                let numbered = batch
                    .iter()
                    .enumerate()
                    .map(|(i, (_, m))| format!("{}. {}", i + 1, m))
                    .collect::<Vec<_>>()
                    .join("\n");

                let response = llm
                    .generate(crate::llm_provider::LlmRequest {
                        system_prompt: Some(prompts.extract_knowledge_graph.clone()),
                        user_prompt: numbered,
                        max_tokens: 2048,
                        temperature: 0.3,
                        label: None,
                        timeout_secs: None,
                    })
                    .await;

                match response {
                    Ok(output) => {
                        let batch_results = crate::extract::parse_kg_response(&output, batch);
                        results.extend(batch_results);
                    }
                    Err(e) => {
                        log::warn!("[importer] LLM KG extraction failed: {e}");
                    }
                }
            }
            results
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    (classifications, kg_results)
}

/// Phase 3: Store documents and KG results in the database. Requires DB access.
pub async fn import_phase3_store(
    db: &MemoryDB,
    prepared: &ImportPrepared,
    classifications: &[ClassificationResult],
    kg_results: &[KgExtractionResult],
    source: &str,
    confidence_cfg: &crate::tuning::ConfidenceConfig,
) -> Result<ImportResult, OriginError> {
    let batch_id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().timestamp();
    let mut imported = 0usize;
    let mut skipped = 0usize;
    let mut breakdown: HashMap<String, usize> = HashMap::new();
    let mut entities_created = 0usize;
    let mut observations_added = 0usize;
    let mut relations_created = 0usize;
    let mut entity_cache: HashMap<String, String> = HashMap::new();

    for (i, memory) in prepared.memories.iter().enumerate() {
        if prepared.duplicates.contains(&i) {
            skipped += 1;
            continue;
        }

        let classification = classifications.get(i).cloned().unwrap_or_default();
        let memory_type = &classification.memory_type;
        let domain = classification.domain.as_deref();
        let confidence =
            compute_effective_confidence(None, Some(memory_type), "review", None, confidence_cfg);

        let title: String = memory.content.chars().take(80).collect();
        let source_id = format!("import_{}_{}", batch_id, i);

        let doc = RawDocument {
            source: "memory".to_string(),
            source_id,
            title,
            summary: None,
            content: memory.content.clone(),
            url: None,
            last_modified: memory.extracted_date.unwrap_or(now),
            metadata: HashMap::from([
                ("import_source".to_string(), source.to_string()),
                ("import_batch".to_string(), batch_id.clone()),
            ]),
            memory_type: Some(memory_type.to_string()),
            domain: domain.map(|s| s.to_string()),
            source_agent: Some(source.to_string()),
            confidence: Some(confidence),
            confirmed: None,
            supersedes: None,
            pending_revision: false,
            ..Default::default()
        };

        db.upsert_documents(vec![doc]).await?;
        imported += 1;
        *breakdown.entry(memory_type.to_string()).or_insert(0) += 1;
    }

    // Store KG results
    for kg in kg_results {
        for entity in &kg.entities {
            if let Ok((_id, true)) =
                resolve_or_create_entity(db, &mut entity_cache, entity, source).await
            {
                entities_created += 1;
            }
        }
        for obs in &kg.observations {
            if let Some(entity_id) = entity_cache.get(&obs.entity.to_lowercase()) {
                if db
                    .add_observation(entity_id, &obs.content, Some(source), None)
                    .await
                    .is_ok()
                {
                    observations_added += 1;
                }
            }
        }
        for rel in &kg.relations {
            let from_id = entity_cache.get(&rel.from.to_lowercase()).cloned();
            let to_id = entity_cache.get(&rel.to.to_lowercase()).cloned();
            if let (Some(from), Some(to)) = (from_id, to_id) {
                let mem_source_id = format!("import_{}_{}", batch_id, kg.index);
                if db
                    .create_relation(
                        &from,
                        &to,
                        &rel.relation_type,
                        Some(source),
                        rel.confidence,
                        rel.explanation.as_deref(),
                        Some(&mem_source_id),
                    )
                    .await
                    .is_ok()
                {
                    relations_created += 1;
                }
            }
        }
    }

    Ok(ImportResult {
        imported,
        skipped,
        breakdown,
        entities_created,
        observations_added,
        relations_created,
        batch_id,
    })
}

/// Import memories with full LLM enrichment (classification + knowledge graph extraction).
///
/// This is a convenience wrapper that runs all three phases sequentially.
/// For callers that need to minimize lock scope (e.g., server/Tauri handlers),
/// use the phased API: `import_phase1_prepare` → `import_phase2_llm` → `import_phase3_store`.
pub async fn import_memories(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    raw_text: &str,
    source: &str,
    _label: Option<&str>,
    prompts: &crate::prompts::PromptRegistry,
) -> Result<ImportResult, OriginError> {
    let prepared = import_phase1_prepare(db, raw_text).await?;
    let (classifications, kg_results) = import_phase2_llm(llm, &prepared, prompts).await;
    let confidence_cfg = crate::tuning::ConfidenceConfig::default();
    import_phase3_store(
        db,
        &prepared,
        &classifications,
        &kg_results,
        source,
        &confidence_cfg,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_memories tests ───────────────────────────────────────────

    #[test]
    fn test_parse_single_line_memories() {
        let input = "User is a software engineer\nPrefers dark mode in editors\nLikes Rust programming language";
        let result = parse_memories(input);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, "User is a software engineer");
        assert_eq!(result[1].content, "Prefers dark mode in editors");
        assert_eq!(result[2].content, "Likes Rust programming language");
        assert!(result[0].extracted_date.is_none());
    }

    #[test]
    fn test_parse_chatgpt_date_prefix() {
        let input =
            "[2025-06-15] - User works at a fintech startup\n[2025-01-20] - Working on Q2 launch";
        let result = parse_memories(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "User works at a fintech startup");
        assert!(result[0].extracted_date.is_some());
        assert_eq!(result[1].content, "Working on Q2 launch");
    }

    #[test]
    fn test_parse_unknown_date_prefix() {
        let input = "[unknown] - Prefers dark mode in editors";
        let result = parse_memories(input);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].content, "Prefers dark mode in editors");
        assert!(result[0].extracted_date.is_none());
    }

    #[test]
    fn test_parse_bulleted_list() {
        let input = "- User is a software engineer\n\u{2022} Prefers dark mode in editors\n* Likes Rust programming language";
        let result = parse_memories(input);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, "User is a software engineer");
        assert_eq!(result[1].content, "Prefers dark mode in editors");
        assert_eq!(result[2].content, "Likes Rust programming language");
    }

    #[test]
    fn test_parse_numbered_list() {
        let input = "1. User is a software engineer\n2. Prefers dark mode in editors\n3. Likes Rust programming language";
        let result = parse_memories(input);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].content, "User is a software engineer");
    }

    #[test]
    fn test_parse_type_tags() {
        let input = "[2025-03-01] [identity] - Lives in San Francisco\n[unknown] [preference] - Prefers concise responses always\n[2025-06-15] [decision] - Chose Rust over Electron\n[unknown] [correction] - Never use emojis in code";
        let result = parse_memories(input);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].memory_type.as_deref(), Some("identity"));
        assert_eq!(result[1].memory_type.as_deref(), Some("preference"));
        assert_eq!(result[2].memory_type.as_deref(), Some("decision"));
        assert_eq!(result[3].memory_type.as_deref(), Some("correction"));
        assert_eq!(result[0].content, "Lives in San Francisco");
        assert!(result[0].extracted_date.is_some());
        assert!(result[1].extracted_date.is_none());
    }

    #[test]
    fn test_parse_without_type_tags_defaults_none() {
        let input = "[2025-01-01] - User is a backend engineer\nPrefers dark mode in all editors";
        let result = parse_memories(input);
        assert_eq!(result.len(), 2);
        assert!(result[0].memory_type.is_none());
        assert!(result[1].memory_type.is_none());
    }

    #[test]
    fn test_skip_empty_and_separators() {
        let input = "User is a software engineer\n\n---\n\n===\n***\nPrefers dark mode in editors";
        let result = parse_memories(input);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_skip_short_lines() {
        // Short words and section-header-style lines (< 20 chars) are filtered
        let input = "User is a software engineer\nhi\nok\nPersonal context\nTop of mind\nRecent months\nPrefers dark mode in editors";
        let result = parse_memories(input);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_skip_chatgpt_section_headers() {
        let input = "Earlier context\n[2025-01-01] - User is a backend engineer\nTop of mind\nWork context\n[2025-06-01] - Prefers dark mode in editors\nPersonal context";
        let result = parse_memories(input);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].content, "User is a backend engineer");
        assert_eq!(result[1].content, "Prefers dark mode in editors");
    }

    #[test]
    fn test_deduplicate_exact_matches() {
        let input = "User is a software engineer\nUser is a software engineer\nPrefers dark mode in editors";
        let result = parse_memories(input);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_paragraph_mode() {
        let input = "User is a software engineer\nwho works at a fintech startup\n\nPrefers dark mode\nin all applications";
        let result = parse_memories(input);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].content,
            "User is a software engineer who works at a fintech startup"
        );
        assert_eq!(result[1].content, "Prefers dark mode in all applications");
    }

    // ── validation tests ───────────────────────────────────────────────

    #[test]
    fn test_validate_input_size_ok() {
        let input = "a".repeat(100);
        assert!(validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_too_large() {
        let input = "a".repeat(MAX_INPUT_BYTES + 1);
        let err = validate_input(&input).unwrap_err();
        assert!(err.to_string().contains("exceeds"));
    }

    #[test]
    fn test_validate_too_many_memories() {
        let input = (0..501)
            .map(|i| format!("This is a test memory number {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let parsed = parse_memories(&input);
        assert!(parsed.len() > MAX_MEMORIES);
        let err = validate_parsed_count(parsed.len()).unwrap_err();
        assert!(err.to_string().contains("500"));
    }

    // ── Task 5: Vector dedup tests ────────────────────────────────────

    #[tokio::test]
    async fn test_find_duplicates_empty_db() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let memories = vec![ParsedMemory {
            content: "User is an engineer".to_string(),
            extracted_date: None,
            memory_type: None,
        }];
        let dupes = find_duplicates(&db, &memories).await;
        assert!(dupes.is_empty());
    }

    // ── Task 6: Entity resolution tests ───────────────────────────────

    #[tokio::test]
    async fn test_resolve_or_create_entity_new() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let mut cache = HashMap::new();
        let entity = crate::extract::ExtractedEntity {
            name: "user".to_string(),
            entity_type: "person".to_string(),
        };
        let (id, created) = resolve_or_create_entity(&db, &mut cache, &entity, "chatgpt")
            .await
            .unwrap();
        assert!(!id.is_empty());
        assert!(created);
        assert_eq!(cache.get("user"), Some(&id));
    }

    #[tokio::test]
    async fn test_resolve_or_create_entity_cached() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let mut cache = HashMap::new();
        cache.insert("user".to_string(), "existing_id".to_string());
        let entity = crate::extract::ExtractedEntity {
            name: "user".to_string(),
            entity_type: "person".to_string(),
        };
        let (id, created) = resolve_or_create_entity(&db, &mut cache, &entity, "chatgpt")
            .await
            .unwrap();
        assert_eq!(id, "existing_id");
        assert!(!created);
    }

    // ── Task 4: Alias-based entity resolution tests ───────────────────

    #[tokio::test]
    async fn test_resolve_or_create_entity_alias_resolution() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let mut cache = std::collections::HashMap::new();

        let entity = crate::extract::ExtractedEntity {
            name: "Alice Chen".to_string(),
            entity_type: "person".to_string(),
        };

        // First call: creates entity + alias
        let (id1, created1) = resolve_or_create_entity(&db, &mut cache, &entity, "test")
            .await
            .unwrap();
        assert!(created1);

        // Clear cache to force alias lookup
        cache.clear();

        // Second call with same case: should resolve via alias, not create
        let (id2, created2) = resolve_or_create_entity(&db, &mut cache, &entity, "test")
            .await
            .unwrap();
        assert!(!created2);
        assert_eq!(id1, id2);

        // Clear cache again
        cache.clear();

        // Third call with different case: should also resolve via alias
        let entity_lower = crate::extract::ExtractedEntity {
            name: "alice chen".to_string(),
            entity_type: "person".to_string(),
        };
        let (id3, created3) = resolve_or_create_entity(&db, &mut cache, &entity_lower, "test")
            .await
            .unwrap();
        assert!(!created3);
        assert_eq!(id1, id3);
    }

    // ── Task 7: Import orchestration tests ────────────────────────────

    #[tokio::test]
    async fn test_import_memories_basic() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let input = "User is a software engineer\nPrefers dark mode in editors\nLikes Rust programming language";
        let result = import_memories_no_llm(
            &db,
            input,
            "chatgpt",
            None,
            &crate::tuning::ConfidenceConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(result.imported, 3);
        assert_eq!(result.skipped, 0);
        assert!(!result.batch_id.is_empty());
    }

    #[tokio::test]
    async fn test_import_memories_dedup() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let input = "User is a software engineer\nPrefers dark mode in editors";
        let result1 = import_memories_no_llm(
            &db,
            input,
            "chatgpt",
            None,
            &crate::tuning::ConfidenceConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(result1.imported, 2);
        let result2 = import_memories_no_llm(
            &db,
            input,
            "chatgpt",
            None,
            &crate::tuning::ConfidenceConfig::default(),
        )
        .await
        .unwrap();
        assert_eq!(result2.imported, 0);
        assert_eq!(result2.skipped, 2);
    }

    #[tokio::test]
    async fn test_import_memories_input_too_large() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let input = "a".repeat(crate::importer::MAX_INPUT_BYTES + 1);
        let result = import_memories_no_llm(
            &db,
            &input,
            "chatgpt",
            None,
            &crate::tuning::ConfidenceConfig::default(),
        )
        .await;
        assert!(result.is_err());
    }
}
