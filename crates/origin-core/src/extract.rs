// SPDX-License-Identifier: AGPL-3.0-only
//! Knowledge-graph extraction via the on-device LLM engine.
//!
//! Extracts entities, observations, and relations from batches of memory
//! content, returning structured [`KgExtractionResult`] records. Uses
//! [`crate::engine::LlmEngine::run_inference`] to drive generation and
//! [`crate::engine::extract_json_array`] for lenient JSON parsing.

use crate::engine::{extract_json_array, LlmEngine, CTX_SIZE};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    #[serde(alias = "type")]
    pub entity_type: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtractedObservation {
    pub entity: String,
    pub content: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtractedRelation {
    pub from: String,
    pub to: String,
    #[serde(alias = "type")]
    pub relation_type: String,
    pub confidence: Option<f64>,
    pub explanation: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KgExtractionResult {
    pub index: usize,
    pub entities: Vec<ExtractedEntity>,
    pub observations: Vec<ExtractedObservation>,
    pub relations: Vec<ExtractedRelation>,
}

/// Parse a batch KG extraction response from LLM output.
/// Extracts JSON array, validates each entry, falls back to empty defaults for invalid entries,
/// and pads with empty defaults if the array is shorter than expected.
pub fn parse_kg_response(raw: &str, memories: &[(usize, String)]) -> Vec<KgExtractionResult> {
    let expected_count = memories.len();

    let empty_result = |idx: usize| KgExtractionResult {
        index: idx,
        entities: Vec::new(),
        observations: Vec::new(),
        relations: Vec::new(),
    };

    let json_str = match extract_json_array(raw) {
        Some(s) => s,
        None => return memories.iter().map(|(idx, _)| empty_result(*idx)).collect(),
    };

    let entries: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return memories.iter().map(|(idx, _)| empty_result(*idx)).collect(),
    };

    let mut results: Vec<KgExtractionResult> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let index = memories.get(i).map(|(idx, _)| *idx).unwrap_or(i);

            let entities = entry
                .get("entities")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|e| serde_json::from_value::<ExtractedEntity>(e.clone()).ok())
                        .collect()
                })
                .unwrap_or_default();

            let observations = entry
                .get("observations")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|e| {
                            serde_json::from_value::<ExtractedObservation>(e.clone()).ok()
                        })
                        .collect()
                })
                .unwrap_or_default();

            let relations = entry
                .get("relations")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|e| serde_json::from_value::<ExtractedRelation>(e.clone()).ok())
                        .collect()
                })
                .unwrap_or_default();

            KgExtractionResult {
                index,
                entities,
                observations,
                relations,
            }
        })
        .collect();

    // Pad with empty defaults if fewer results than expected
    while results.len() < expected_count {
        let idx = memories
            .get(results.len())
            .map(|(idx, _)| *idx)
            .unwrap_or(results.len());
        results.push(empty_result(idx));
    }

    results
}

#[allow(dead_code)] // Wired via OnDeviceProvider as part of the refinery pipeline
impl LlmEngine {
    /// Extract knowledge graph entities, observations, and relations from a batch of memories.
    /// Returns one `KgExtractionResult` per input memory. Falls back to empty results on failure.
    pub fn extract_kg_batch(&self, memories: &[(usize, String)]) -> Vec<KgExtractionResult> {
        if memories.is_empty() {
            return Vec::new();
        }

        let numbered = memories
            .iter()
            .enumerate()
            .map(|(i, (_, m))| format!("{}. {}", i + 1, m))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {numbered}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
            sys = self.prompts().extract_knowledge_graph,
        );

        let empty_result = |idx: usize| KgExtractionResult {
            index: idx,
            entities: Vec::new(),
            observations: Vec::new(),
            relations: Vec::new(),
        };

        match self.run_inference(&prompt, 2048, 0.3, CTX_SIZE, Some("extract")) {
            Some(response) => parse_kg_response(&response, memories),
            None => memories.iter().map(|(idx, _)| empty_result(*idx)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kg_response_valid() {
        let json = r#"[{"i":1,"entities":[{"name":"user","type":"person"}],"observations":[{"entity":"user","content":"is a software engineer"}],"relations":[]}]"#;
        let results = parse_kg_response(json, &[(0, "test".to_string())]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entities.len(), 1);
        assert_eq!(results[0].entities[0].name, "user");
        assert_eq!(results[0].observations.len(), 1);
    }

    #[test]
    fn test_parse_kg_response_malformed() {
        let json = "not json";
        let results = parse_kg_response(json, &[(0, "a".to_string()), (1, "b".to_string())]);
        assert_eq!(results.len(), 2);
        assert!(results[0].entities.is_empty());
    }

    #[test]
    fn test_parse_kg_response_partial() {
        let json = r#"[{"i":1,"entities":[{"name":"user","type":"person"}],"observations":[],"relations":[]}]"#;
        let results = parse_kg_response(json, &[(0, "a".to_string()), (1, "b".to_string())]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entities.len(), 1);
        assert!(results[1].entities.is_empty());
    }

    #[test]
    fn test_parse_kg_response_with_confidence_and_explanation() {
        let raw = r#"[{"i": 0, "entities": [{"name": "Alice", "type": "person"}], "observations": [{"entity": "Alice", "content": "leads backend team"}], "relations": [{"from": "Alice", "to": "Backend Team", "type": "leads", "confidence": 0.9, "explanation": "Alice is the tech lead for the backend team"}]}]"#;
        let memories = vec![(0, "Alice leads the backend team".to_string())];
        let results = parse_kg_response(raw, &memories);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].relations.len(), 1);
        assert_eq!(results[0].relations[0].confidence, Some(0.9));
        assert_eq!(
            results[0].relations[0].explanation.as_deref(),
            Some("Alice is the tech lead for the backend team")
        );
    }

    #[test]
    fn test_parse_kg_response_missing_confidence_defaults_none() {
        let raw = r#"[{"i": 0, "entities": [], "observations": [], "relations": [{"from": "A", "to": "B", "type": "uses"}]}]"#;
        let memories = vec![(0, "test".to_string())];
        let results = parse_kg_response(raw, &memories);
        assert_eq!(results[0].relations[0].confidence, None);
        assert_eq!(results[0].relations[0].explanation, None);
    }
}
