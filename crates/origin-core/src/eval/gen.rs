// SPDX-License-Identifier: AGPL-3.0-only
//! Fixture generation — adversarial and blind test case creation.

use crate::eval::fixtures::{EvalCase, FixtureFile, SeedMemory};
use crate::llm_provider::{LlmProvider, LlmRequest};
use std::path::Path;
use std::sync::Arc;

/// Known pipeline failure modes for regression fixtures.
const REGRESSION_PATTERNS: &[RegressionPattern] = &[
    RegressionPattern {
        name: "cross_domain_semantic_overlap",
        description: "Technical query where identity/personal memories share keywords with project-specific decisions",
        generator_prompt: "Create a memory eval test case about a technical decision (e.g., choosing a library, database, or algorithm). \
            Generate:\n\
            1. A search query (5-8 words) asking about the decision\n\
            2. Two relevant seed memories: one decision (relevance=3, confirmed=true) and one supporting fact (relevance=2)\n\
            3. Two negative seeds: identity/personal memories that share 1-2 keywords with the query but are about a person's background, not the technical decision\n\
            All seeds need domain. Relevant seeds: domain=\"projectA\". Negatives: domain=\"personal\", memory_type=\"identity\".",
    },
    RegressionPattern {
        name: "recap_original_competition",
        description: "Original memory competing with its recap summary that has high keyword overlap",
        generator_prompt: "Create a memory eval test case where an original memory competes with recap summaries.\n\
            Generate:\n\
            1. A search query about a specific technical topic\n\
            2. One relevant seed: detailed original memory (relevance=3, confirmed=true, memory_type=\"decision\" or \"fact\")\n\
            3. Two negative seeds: recap-style memories (memory_type=\"recap\") that mention the same topic but are session summaries like \"Session recap: worked on X, fixed Y, tested Z\"\n\
            The recaps should share keywords with the query but contain less specific information.",
    },
    RegressionPattern {
        name: "fts_and_failure",
        description: "Multi-word query where relevant memory doesn't contain all query terms, testing FTS OR fallback",
        generator_prompt: "Create a memory eval test case where the relevant memory uses different vocabulary than the query.\n\
            Generate:\n\
            1. A 4-6 word search query using specific terminology\n\
            2. Two relevant seeds that answer the query but use synonyms/related terms, NOT the exact query words (relevance=3 confirmed=true, relevance=2)\n\
            3. One negative seed that contains more of the query's exact words but is about an unrelated topic\n\
            Example: query \"vector embedding model selection\" but relevant content says \"chose bge-small-en-v1.5 via FastEmbed for on-device inference\".",
    },
    RegressionPattern {
        name: "confirmed_vs_unconfirmed",
        description: "Confirmed relevant memory competing with unconfirmed negative at similar embedding distance",
        generator_prompt: "Create a memory eval test case where confirmation status should differentiate results.\n\
            Generate:\n\
            1. A search query about a specific topic\n\
            2. One relevant seed: confirmed=true, relevance=3, specific and authoritative content\n\
            3. One relevant seed: relevance=2, unconfirmed, less specific\n\
            4. One negative seed: unconfirmed, semantically similar content but from a different domain\n\
            The negative should be close enough in meaning that embedding distance alone won't separate it.",
    },
    RegressionPattern {
        name: "graph_observation_vs_memory",
        description: "Knowledge graph observations competing with direct memories for the same query",
        generator_prompt: "Create a memory eval test case with both direct memories and knowledge graph entities.\n\
            Generate:\n\
            1. A search query about a technology or concept\n\
            2. Two relevant seed memories about the topic (relevance=3 confirmed=true, relevance=2)\n\
            3. One entity with name, type, domain, and 3 observations (one graded relevance=2, two ungraded)\n\
            4. One negative seed from a different domain\n\
            The entity observations should complement the direct memories, not duplicate them.",
    },
    RegressionPattern {
        name: "structured_vs_prose",
        description: "Memory with structured_fields vs plain prose memory for the same query",
        generator_prompt: "Create a memory eval test case comparing structured and prose memories.\n\
            Generate:\n\
            1. A search query about a preference or decision\n\
            2. One relevant seed with structured_fields JSON (relevance=3, confirmed=true). Include a structured_fields string like '{\"preference\":\"X\",\"applies_when\":\"Y\"}'\n\
            3. One relevant seed without structured_fields, plain prose (relevance=2)\n\
            4. One negative seed that matches the query topic but is about a different context",
    },
];

struct RegressionPattern {
    name: &'static str,
    description: &'static str,
    generator_prompt: &'static str,
}

/// Generate regression fixtures targeting known pipeline weaknesses.
pub async fn generate_regression(
    llm: &Arc<dyn LlmProvider>,
    count: usize,
    out_dir: &Path,
) -> Result<usize, crate::error::OriginError> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| crate::error::OriginError::Generic(format!("create gen dir: {e}")))?;

    let mut generated = 0;

    for (i, pattern) in REGRESSION_PATTERNS.iter().enumerate() {
        if generated >= count {
            break;
        }

        let raw_case = match generate_case(llm, pattern.generator_prompt).await {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[fixture_gen] generate failed for {}: {e}", pattern.name);
                continue;
            }
        };
        let mut graded_case = match grade_case(llm, &raw_case).await {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[fixture_gen] grading failed for {}: {e}", pattern.name);
                raw_case // Use ungraded if grading fails
            }
        };

        if !validate_case(&mut graded_case) {
            log::warn!(
                "[fixture_gen] skipping invalid case for pattern {}",
                pattern.name
            );
            continue;
        }

        let fixture = FixtureFile {
            cases: vec![graded_case],
        };
        let toml_str = toml::to_string_pretty(&fixture)
            .map_err(|e| crate::error::OriginError::Generic(format!("serialize fixture: {e}")))?;

        let filename = format!("reg_{:03}_{}.toml", i + 1, pattern.name);
        let path = out_dir.join(&filename);
        std::fs::write(&path, &toml_str).map_err(|e| {
            crate::error::OriginError::Generic(format!("write fixture {filename}: {e}"))
        })?;

        generated += 1;
        log::info!("[fixture_gen] wrote {filename} ({})", pattern.description);
    }

    Ok(generated)
}

/// Generate blind fixtures with no knowledge of pipeline internals.
pub async fn generate_blind(
    llm: &Arc<dyn LlmProvider>,
    count: usize,
    out_dir: &Path,
) -> Result<usize, crate::error::OriginError> {
    std::fs::create_dir_all(out_dir)
        .map_err(|e| crate::error::OriginError::Generic(format!("create gen dir: {e}")))?;

    let mut generated = 0;
    let mut seen_queries: std::collections::HashSet<String> = std::collections::HashSet::new();
    let topics = [
        "cooking recipe",
        "work project deadline",
        "fitness routine",
        "travel plans",
        "book recommendation",
        "software architecture decision",
        "health appointment",
        "financial goal",
        "language learning",
        "home renovation",
        "team meeting notes",
        "API design choice",
        "database migration",
        "debugging session",
        "code review feedback",
        "product launch",
        "customer feedback",
        "hiring decision",
        "tool preference",
        "conference talk",
        "research finding",
        "performance optimization",
    ];

    for i in 0..count.max(topics.len()) {
        if generated >= count {
            break;
        }
        let topic = topics[i % topics.len()];
        let blind_prompt = format!(
            "You are generating test data for a personal memory system. The system stores memories with these fields:\n\
            - content: the memory text\n\
            - memory_type: one of \"identity\", \"preference\", \"decision\", \"fact\", \"goal\"\n\
            - domain: a project or topic name (e.g., \"work\", \"cooking\", \"fitness\", \"projectX\")\n\
            - confirmed: whether the user has verified this memory (true/false)\n\n\
            Create a realistic search scenario about {topic_hint} (scenario #{num}):\n\
            1. A natural search query a person would type to find a memory (5-10 words)\n\
            2. 2-3 seed memories that SHOULD be found (mark the best match relevance=3, others relevance=2 or 1)\n\
            3. 1-2 negative memories that should NOT be found — they're about different topics but might share some words\n\n\
            Make the scenario realistic — use specific details, names, dates, technical terms. Avoid generic placeholder content.\n\
            The highest-relevance seed should have confirmed=true.\n\n\
            Respond with ONLY valid JSON matching this schema:\n\
            {{\"query\": \"...\", \"domain\": \"...\" or null, \"seeds\": [{{\"id\": \"mem_...\", \"content\": \"...\", \"memory_type\": \"...\", \"domain\": \"...\", \"relevance\": 3, \"confirmed\": true}}], \"negative_seeds\": [{{\"id\": \"neg_...\", \"content\": \"...\", \"memory_type\": \"...\", \"domain\": \"...\"}}]}}",
            num = i + 1,
            topic_hint = topic
        );

        let raw_case = match generate_case(llm, &blind_prompt).await {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[fixture_gen] blind generate failed #{}: {e}", i + 1);
                continue;
            }
        };
        let mut graded_case = match grade_case(llm, &raw_case).await {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[fixture_gen] blind grading failed #{}: {e}", i + 1);
                raw_case
            }
        };

        if !validate_case(&mut graded_case) {
            log::warn!("[fixture_gen] skipping invalid blind case #{}", i + 1);
            continue;
        }

        // Dedup: skip if we've seen this query before (model repetition)
        if !seen_queries.insert(graded_case.query.clone()) {
            log::warn!(
                "[fixture_gen] skipping duplicate query: {}",
                graded_case.query
            );
            continue;
        }

        let fixture = FixtureFile {
            cases: vec![graded_case],
        };
        let toml_str = toml::to_string_pretty(&fixture)
            .map_err(|e| crate::error::OriginError::Generic(format!("serialize fixture: {e}")))?;

        let filename = format!("blind_{:03}.toml", generated + 1);
        let path = out_dir.join(&filename);
        std::fs::write(&path, &toml_str).map_err(|e| {
            crate::error::OriginError::Generic(format!("write fixture {filename}: {e}"))
        })?;

        generated += 1;
        log::info!("[fixture_gen] wrote {filename}");
    }

    Ok(generated)
}

/// Call the LLM to generate a single eval case from a prompt.
async fn generate_case(
    llm: &Arc<dyn LlmProvider>,
    prompt: &str,
) -> Result<EvalCase, crate::error::OriginError> {
    let system = "You generate eval test cases for a memory search system. \
        Respond with ONLY valid JSON matching this schema:\n\
        {\"query\": \"...\", \"domain\": \"...\" or null, \
        \"seeds\": [{\"id\": \"mem_...\", \"content\": \"...\", \"memory_type\": \"...\", \"domain\": \"...\", \"relevance\": 3, \"confirmed\": true}], \
        \"negative_seeds\": [{\"id\": \"neg_...\", \"content\": \"...\", \"memory_type\": \"...\", \"domain\": \"...\"}]}\n\
        No markdown, no explanation — ONLY the JSON object.";

    let response = llm
        .generate(LlmRequest {
            system_prompt: Some(system.to_string()),
            user_prompt: prompt.to_string(),
            max_tokens: 1024,
            temperature: 0.9, // Higher temperature for diverse generation; grader uses 0.1
            label: None,
        })
        .await
        .map_err(|e| crate::error::OriginError::Generic(format!("LLM generate: {e}")))?;

    parse_case_json(&response)
}

/// Call the LLM to independently grade/verify relevance of a generated case.
/// This is a SEPARATE call from generation — the grader doesn't see the generator prompt.
async fn grade_case(
    llm: &Arc<dyn LlmProvider>,
    case: &EvalCase,
) -> Result<EvalCase, crate::error::OriginError> {
    let mut seeds_desc = String::new();
    for (i, s) in case.seeds.iter().enumerate() {
        seeds_desc.push_str(&format!(
            "Seed {}: [{}] {}\n",
            i + 1,
            s.memory_type,
            s.content
        ));
    }
    for (i, n) in case.negative_seeds.iter().enumerate() {
        seeds_desc.push_str(&format!(
            "Neg {}: [{}] {}\n",
            i + 1,
            n.memory_type,
            n.content
        ));
    }

    let grader_prompt = format!(
        "Query: \"{}\"\n\nMemories:\n{}\n\
        For each memory, rate relevance to the query on a 0-3 scale:\n\
        3 = directly answers the query\n\
        2 = relevant supporting information\n\
        1 = tangentially related\n\
        0 = not relevant\n\n\
        Respond with ONLY a JSON array of integers, one per memory in order. Example: [3, 2, 0, 0]",
        case.query, seeds_desc
    );

    let response = llm.generate(LlmRequest {
        system_prompt: Some("You are a relevance grader. Rate each memory's relevance to the query. Respond with ONLY a JSON array of integers.".to_string()),
        user_prompt: grader_prompt,
        max_tokens: 64,
        temperature: 0.1,
        label: None,
    }).await.map_err(|e| crate::error::OriginError::Generic(format!("LLM grade: {e}")))?;

    let grades = parse_grades(&response, case.seeds.len() + case.negative_seeds.len());

    let mut graded = case.clone();
    for (i, seed) in graded.seeds.iter_mut().enumerate() {
        if let Some(&grade) = grades.get(i) {
            seed.relevance = grade;
        }
    }
    for (i, neg) in graded.negative_seeds.iter_mut().enumerate() {
        if let Some(&grade) = grades.get(i + case.seeds.len()) {
            neg.relevance = grade;
        }
    }

    Ok(graded)
}

const VALID_MEMORY_TYPES: &[&str] = &[
    "identity",
    "preference",
    "decision",
    "fact",
    "goal",
    "recap",
];

/// Normalize invalid memory_type to the closest valid type.
fn normalize_memory_type(raw: &str) -> String {
    let lower = raw.to_lowercase();
    if VALID_MEMORY_TYPES.contains(&lower.as_str()) {
        return lower;
    }
    // Map common LLM hallucinations to valid types
    match lower.as_str() {
        "hypothesis" | "observation" | "direct" | "learning" => "fact".to_string(),
        "structured_fields" | "structured" | "prose" => "fact".to_string(),
        "embedding_model" | "model" | "technical" => "decision".to_string(),
        "entity" | "concept" | "knowledge" => "fact".to_string(),
        "personal" | "bio" | "background" => "identity".to_string(),
        "wish" | "aspiration" | "plan" => "goal".to_string(),
        "like" | "dislike" | "favorite" => "preference".to_string(),
        _ => "fact".to_string(), // safe default
    }
}

/// Validate a generated case: non-empty, valid types, at least 1 seed.
fn validate_case(case: &mut EvalCase) -> bool {
    if case.query.is_empty() || case.seeds.is_empty() {
        return false;
    }
    // Normalize all memory_types
    for seed in &mut case.seeds {
        seed.memory_type = normalize_memory_type(&seed.memory_type);
    }
    for neg in &mut case.negative_seeds {
        neg.memory_type = normalize_memory_type(&neg.memory_type);
    }
    true
}

/// Parse JSON case from LLM output. Handles noisy output by finding matching braces.
fn parse_case_json(response: &str) -> Result<EvalCase, crate::error::OriginError> {
    let start = response.find('{').ok_or_else(|| {
        crate::error::OriginError::Generic("no JSON object in LLM response".into())
    })?;

    // Find matching closing brace (handles nested objects)
    let mut depth = 0i32;
    let mut end = start;
    let mut in_string = false;
    let mut escape_next = false;
    for (i, ch) in response[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    end = start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(crate::error::OriginError::Generic(
            "unmatched braces in LLM response".into(),
        ));
    }

    let json_str = &response[start..=end];
    let val: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| crate::error::OriginError::Generic(format!("parse LLM JSON: {e}")))?;

    let query = val["query"]
        .as_str()
        .ok_or_else(|| crate::error::OriginError::Generic("missing 'query' field".into()))?
        .to_string();
    let domain = val["domain"].as_str().map(|s| s.to_string());

    let seeds = parse_seed_array(&val["seeds"])?;
    let negatives = parse_seed_array(&val["negative_seeds"])?;

    Ok(EvalCase {
        query,
        domain,
        seeds,
        negative_seeds: negatives,
        entities: vec![],
        empty_set: false,
    })
}

fn parse_seed_array(val: &serde_json::Value) -> Result<Vec<SeedMemory>, crate::error::OriginError> {
    let arr = val
        .as_array()
        .ok_or_else(|| crate::error::OriginError::Generic("expected array for seeds".into()))?;

    let mut seeds = Vec::new();
    for item in arr {
        seeds.push(SeedMemory {
            id: item["id"].as_str().unwrap_or("mem_unknown").to_string(),
            content: item["content"].as_str().unwrap_or("").to_string(),
            memory_type: item["memory_type"].as_str().unwrap_or("fact").to_string(),
            domain: item["domain"].as_str().map(|s| s.to_string()),
            relevance: item["relevance"].as_u64().unwrap_or(0) as u8,
            structured_fields: item["structured_fields"].as_str().map(|s| s.to_string()),
            confidence: None,
            confirmed: item["confirmed"].as_bool(),
            quality: None,
            is_recap: None,
            source_agent: None,
            age_days: None,
            supersedes: None,
        });
    }
    Ok(seeds)
}

/// Parse a JSON array of integer grades from LLM output.
fn parse_grades(response: &str, expected: usize) -> Vec<u8> {
    let start = response.find('[');
    let end = response.rfind(']');
    if let (Some(s), Some(e)) = (start, end) {
        if e > s {
            if let Ok(vals) = serde_json::from_str::<Vec<serde_json::Value>>(&response[s..=e]) {
                return vals
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0).min(3) as u8)
                    .collect();
            }
        }
    }
    log::warn!("[fixture_gen] failed to parse grades, using defaults");
    vec![0; expected]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_provider::{LlmError, LlmProvider, LlmRequest};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock LLM that returns canned JSON responses for generate and grade calls.
    struct MockLlm {
        call_count: AtomicUsize,
        responses: Vec<String>,
    }

    impl MockLlm {
        fn new(responses: Vec<&str>) -> Arc<Self> {
            Arc::new(Self {
                call_count: AtomicUsize::new(0),
                responses: responses.into_iter().map(|s| s.to_string()).collect(),
            })
        }
    }

    #[async_trait]
    impl LlmProvider for MockLlm {
        async fn generate(&self, _request: LlmRequest) -> Result<String, LlmError> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            self.responses
                .get(idx)
                .cloned()
                .ok_or(LlmError::InferenceFailed("no more mock responses".into()))
        }
        fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "mock"
        }
        fn backend(&self) -> crate::llm_provider::LlmBackend {
            crate::llm_provider::LlmBackend::OnDevice
        }
    }

    const MOCK_CASE_JSON: &str = r#"{"query": "database migration strategy", "domain": "backend", "seeds": [{"id": "mem_1", "content": "Chose flyway over liquibase for SQL migrations due to simpler config", "memory_type": "decision", "domain": "backend", "relevance": 3, "confirmed": true}, {"id": "mem_2", "content": "Migrations run on startup in dev, manually in prod via CI pipeline", "memory_type": "fact", "domain": "backend", "relevance": 2}], "negative_seeds": [{"id": "neg_1", "content": "Prefers PostgreSQL over MySQL for ACID compliance", "memory_type": "preference", "domain": "backend"}]}"#;

    const MOCK_GRADES: &str = "[3, 2, 0]";

    #[test]
    fn test_parse_case_json_clean() {
        let json = r#"{"query": "database choice", "domain": "project", "seeds": [{"id": "mem_1", "content": "chose postgres", "memory_type": "decision", "domain": "project", "relevance": 3, "confirmed": true}], "negative_seeds": [{"id": "neg_1", "content": "unrelated", "memory_type": "fact", "domain": "other"}]}"#;
        let case = parse_case_json(json).unwrap();
        assert_eq!(case.query, "database choice");
        assert_eq!(case.seeds.len(), 1);
        assert_eq!(case.seeds[0].relevance, 3);
        assert_eq!(case.negative_seeds.len(), 1);
    }

    #[test]
    fn test_parse_case_json_with_noise() {
        let response = "Here's the test case:\n```json\n{\"query\": \"test\", \"domain\": null, \"seeds\": [], \"negative_seeds\": []}\n```";
        let case = parse_case_json(response).unwrap();
        assert_eq!(case.query, "test");
    }

    #[test]
    fn test_parse_grades_clean() {
        let grades = parse_grades("[3, 2, 0, 1]", 4);
        assert_eq!(grades, vec![3, 2, 0, 1]);
    }

    #[test]
    fn test_parse_grades_with_noise() {
        let grades = parse_grades("The relevance scores are: [3, 2, 0]", 3);
        assert_eq!(grades, vec![3, 2, 0]);
    }

    #[test]
    fn test_parse_grades_fallback() {
        let grades = parse_grades("not valid json", 3);
        assert_eq!(grades, vec![0, 0, 0]);
    }

    #[test]
    fn test_normalize_valid_types() {
        assert_eq!(normalize_memory_type("fact"), "fact");
        assert_eq!(normalize_memory_type("Decision"), "decision");
        assert_eq!(normalize_memory_type("IDENTITY"), "identity");
        assert_eq!(normalize_memory_type("recap"), "recap");
    }

    #[test]
    fn test_normalize_hallucinated_types() {
        assert_eq!(normalize_memory_type("embedding_model"), "decision");
        assert_eq!(normalize_memory_type("hypothesis"), "fact");
        assert_eq!(normalize_memory_type("structured_fields"), "fact");
        assert_eq!(normalize_memory_type("prose"), "fact");
        assert_eq!(normalize_memory_type("direct"), "fact");
        assert_eq!(normalize_memory_type("totally_unknown"), "fact");
    }

    #[test]
    fn test_validate_case_normalizes_types() {
        let mut case = EvalCase {
            query: "test".into(),
            domain: None,
            seeds: vec![SeedMemory {
                id: "m1".into(),
                content: "content".into(),
                memory_type: "embedding_model".into(),
                domain: None,
                relevance: 3,
                structured_fields: None,
                confidence: None,
                confirmed: None,
                quality: None,
                is_recap: None,
                source_agent: None,
                age_days: None,
                supersedes: None,
            }],
            negative_seeds: vec![],
            entities: vec![],
            empty_set: false,
        };
        assert!(validate_case(&mut case));
        assert_eq!(case.seeds[0].memory_type, "decision");
    }

    #[test]
    fn test_validate_case_rejects_empty() {
        let mut case = EvalCase {
            query: "".into(),
            domain: None,
            seeds: vec![],
            negative_seeds: vec![],
            entities: vec![],
            empty_set: false,
        };
        assert!(!validate_case(&mut case));
    }

    #[tokio::test]
    async fn test_generate_case_with_mock() {
        let llm = MockLlm::new(vec![MOCK_CASE_JSON]);
        let case = generate_case(&(llm as Arc<dyn LlmProvider>), "test prompt")
            .await
            .unwrap();
        assert_eq!(case.query, "database migration strategy");
        assert_eq!(case.seeds.len(), 2);
        assert_eq!(case.seeds[0].memory_type, "decision");
        assert_eq!(case.negative_seeds.len(), 1);
    }

    #[tokio::test]
    async fn test_grade_case_with_mock() {
        let case = parse_case_json(MOCK_CASE_JSON).unwrap();
        let llm = MockLlm::new(vec![MOCK_GRADES]);
        let graded = grade_case(&(llm as Arc<dyn LlmProvider>), &case)
            .await
            .unwrap();
        assert_eq!(graded.seeds[0].relevance, 3);
        assert_eq!(graded.seeds[1].relevance, 2);
        assert_eq!(graded.negative_seeds[0].relevance, 0);
    }

    #[tokio::test]
    async fn test_generate_regression_with_mock() {
        // 6 patterns × 2 calls each (generate + grade) = 12 responses needed
        let responses: Vec<&str> = (0..12)
            .map(|i| {
                if i % 2 == 0 {
                    MOCK_CASE_JSON
                } else {
                    MOCK_GRADES
                }
            })
            .collect();
        let llm = MockLlm::new(responses);

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("regression");
        let count = generate_regression(&(llm as Arc<dyn LlmProvider>), 6, &out)
            .await
            .unwrap();

        assert_eq!(count, 6);
        // Verify files exist
        let files: Vec<_> = std::fs::read_dir(&out)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
            .collect();
        assert_eq!(files.len(), 6);

        // Verify first file is valid TOML
        let content = std::fs::read_to_string(files[0].path()).unwrap();
        let fixture: crate::eval::fixtures::FixtureFile = toml::from_str(&content).unwrap();
        assert_eq!(fixture.cases.len(), 1);
        assert!(!fixture.cases[0].query.is_empty());
    }

    #[tokio::test]
    async fn test_generate_blind_with_mock() {
        // 3 cases × 2 calls each = 6 responses, but need unique queries for dedup
        let case1 = r#"{"query": "best pasta recipe for weeknight dinner", "domain": "cooking", "seeds": [{"id": "mem_1", "content": "Quick garlic pasta: boil spaghetti, saute garlic in olive oil, toss with parmesan", "memory_type": "fact", "domain": "cooking", "relevance": 3, "confirmed": true}], "negative_seeds": [{"id": "neg_1", "content": "Bought a new cast iron skillet last month", "memory_type": "fact", "domain": "kitchen"}]}"#;
        let case2 = r#"{"query": "kubernetes deployment configuration", "domain": "devops", "seeds": [{"id": "mem_2", "content": "Using helm charts with values override per environment", "memory_type": "decision", "domain": "devops", "relevance": 3, "confirmed": true}], "negative_seeds": [{"id": "neg_2", "content": "Docker desktop uses too much memory on Mac", "memory_type": "fact", "domain": "tools"}]}"#;
        let case3 = r#"{"query": "morning exercise routine schedule", "domain": "fitness", "seeds": [{"id": "mem_3", "content": "Run 5k at 6am Monday Wednesday Friday, yoga Tuesday Thursday", "memory_type": "preference", "domain": "fitness", "relevance": 3, "confirmed": true}], "negative_seeds": [{"id": "neg_3", "content": "Signed up for gym membership in January", "memory_type": "fact", "domain": "fitness"}]}"#;

        let responses = vec![case1, MOCK_GRADES, case2, MOCK_GRADES, case3, MOCK_GRADES];
        let llm = MockLlm::new(responses);

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("blind");
        let count = generate_blind(&(llm as Arc<dyn LlmProvider>), 3, &out)
            .await
            .unwrap();

        assert_eq!(count, 3);
        let files: Vec<_> = std::fs::read_dir(&out)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
            .collect();
        assert_eq!(files.len(), 3);
    }

    #[tokio::test]
    async fn test_generate_blind_dedup_skips_duplicates() {
        // All 3 calls return the same query — dedup should keep only 1
        let responses = vec![
            MOCK_CASE_JSON,
            MOCK_GRADES,
            MOCK_CASE_JSON,
            MOCK_GRADES,
            MOCK_CASE_JSON,
            MOCK_GRADES,
        ];
        let llm = MockLlm::new(responses);

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("blind");
        let count = generate_blind(&(llm as Arc<dyn LlmProvider>), 3, &out)
            .await
            .unwrap();

        assert_eq!(count, 1); // Only 1 unique query
    }

    #[tokio::test]
    async fn test_generate_regression_skips_failures() {
        // First call returns bad JSON, second pair succeeds
        let responses = vec![
            "not valid json at all", // generate call 1 — fails
            MOCK_CASE_JSON,
            MOCK_GRADES, // generate + grade call 2 — succeeds
            MOCK_CASE_JSON,
            MOCK_GRADES, // call 3
            MOCK_CASE_JSON,
            MOCK_GRADES, // call 4
            MOCK_CASE_JSON,
            MOCK_GRADES, // call 5
            MOCK_CASE_JSON,
            MOCK_GRADES, // call 6
        ];
        let llm = MockLlm::new(responses);

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("regression");
        let count = generate_regression(&(llm as Arc<dyn LlmProvider>), 6, &out)
            .await
            .unwrap();

        assert_eq!(count, 5); // 6 patterns, 1 failed = 5 generated
    }
}
