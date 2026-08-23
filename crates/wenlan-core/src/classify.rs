// SPDX-License-Identifier: Apache-2.0
//! Memory classification response parsing.
//!
//! Parses LLM classification output into [`ClassificationResult`].

use crate::engine::extract_json_array;

use wenlan_types::MemoryType;

/// Result of classifying a single memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassificationResult {
    pub memory_type: String,
    pub space: Option<String>,
    pub tags: Vec<String>,
    /// Quality signal: "low", "medium", "high", or None (default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// T8 salience prior: per-memory importance rating 1-10 (LLM-assigned at
    /// write time), or None when absent/malformed. NEVER defaults to a number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<u8>,
}

impl Default for ClassificationResult {
    fn default() -> Self {
        Self {
            memory_type: "fact".to_string(),
            space: None,
            tags: Vec::new(),
            quality: None,
            importance: None,
        }
    }
}

/// Parse a batch classification response from LLM output.
/// Extracts JSON array, validates each entry, falls back to defaults for invalid entries,
/// and pads with defaults if the array is shorter than expected.
pub fn parse_classification_response(
    raw: &str,
    expected_count: usize,
) -> Vec<ClassificationResult> {
    let json_str = match extract_json_array(raw) {
        Some(s) => s,
        None => return vec![ClassificationResult::default(); expected_count],
    };

    let entries: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return vec![ClassificationResult::default(); expected_count],
    };

    let mut results: Vec<ClassificationResult> = entries
        .iter()
        .map(|entry| {
            let memory_type = entry
                .get("type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_lowercase())
                .filter(|s| MemoryType::all_values().contains(&s.as_str()))
                .unwrap_or_else(|| "fact".to_string());

            let space = entry
                .get("domain")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());

            let tags = entry
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            ClassificationResult {
                memory_type,
                space,
                tags,
                quality: None,
                importance: None,
            }
        })
        .collect();

    // Pad with defaults if fewer results than expected
    while results.len() < expected_count {
        results.push(ClassificationResult::default());
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_classification_response_valid() {
        let json = r#"[{"i":1,"type":"identity","domain":"work","tags":["engineer"]},{"i":2,"type":"preference","domain":"technology","tags":["dark mode"]}]"#;
        let results = parse_classification_response(json, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].memory_type, "identity");
        assert_eq!(results[0].space, Some("work".to_string()));
        assert_eq!(results[1].memory_type, "preference");
    }

    #[test]
    fn test_parse_classification_response_malformed_entry() {
        let json =
            r#"[{"i":1,"type":"identity","domain":"work","tags":[]},{"i":2,"type":"INVALID"}]"#;
        let results = parse_classification_response(json, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].memory_type, "identity");
        assert_eq!(results[1].memory_type, "fact"); // invalid type falls back
    }

    #[test]
    fn test_parse_classification_response_total_garbage() {
        let json = "this is not json at all";
        let results = parse_classification_response(json, 3);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.memory_type == "fact"));
    }

    #[test]
    fn test_parse_classification_response_wrong_count() {
        let json = r#"[{"i":1,"type":"identity","domain":"work","tags":[]}]"#;
        let results = parse_classification_response(json, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].memory_type, "identity");
        assert_eq!(results[1].memory_type, "fact"); // padded
    }

    #[test]
    fn test_parse_classification_response_with_surrounding_text() {
        let json = r#"Here are the results: [{"i":1,"type":"fact","domain":"work","tags":[]}] Hope that helps!"#;
        let results = parse_classification_response(json, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_type, "fact");
    }

    #[test]
    fn test_classify_prompt_contains_all_types() {
        // Drift guard: the compiled-in classify prompt must list every
        // canonical type so the LLM never emits a value our parser rejects.
        // Derive the haystack from the actual prompt const rather than a
        // hand-typed string, otherwise the test rots in parallel with the
        // prompt itself.
        use crate::prompts::defaults::CLASSIFY_MEMORY_QUALITY;
        for val in MemoryType::all_values() {
            assert!(
                CLASSIFY_MEMORY_QUALITY.contains(val),
                "CLASSIFY_MEMORY_QUALITY prompt must include canonical type \"{val}\"",
            );
        }
    }

    #[test]
    fn test_classify_prompts_omit_legacy_goal() {
        // "goal" is folded to Identity by MemoryType::FromStr — it must not
        // appear in any LLM-facing prompt or the model will keep emitting it.
        use crate::prompts::defaults::{CLASSIFY_MEMORY, CLASSIFY_MEMORY_QUALITY};
        for (label, prompt) in [
            ("CLASSIFY_MEMORY", CLASSIFY_MEMORY),
            ("CLASSIFY_MEMORY_QUALITY", CLASSIFY_MEMORY_QUALITY),
        ] {
            let has_goal = prompt
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .any(|tok| tok == "goal");
            assert!(
                !has_goal,
                "{label} prompt still advertises legacy \"goal\" token",
            );
        }
    }
}
