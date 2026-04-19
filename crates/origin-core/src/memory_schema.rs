// SPDX-License-Identifier: AGPL-3.0-only

use serde_json;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MemorySchema {
    pub memory_type: String,
    pub required: Vec<&'static str>,
    pub optional: Vec<&'static str>,
    pub retrieval_cue_template: String,
}

impl MemorySchema {
    pub fn for_type(memory_type: &str) -> Self {
        match memory_type {
            "identity" => Self {
                memory_type: "identity".into(),
                required: vec!["claim"],
                optional: vec!["evidence", "since"],
                retrieval_cue_template: "Who is the user in terms of {claim}?".into(),
            },
            "preference" => Self {
                memory_type: "preference".into(),
                required: vec!["preference", "applies_when"],
                optional: vec!["strength", "alternatives_rejected"],
                retrieval_cue_template: "What does the user prefer regarding {preference}?".into(),
            },
            "decision" => Self {
                memory_type: "decision".into(),
                required: vec!["decision", "context"],
                optional: vec!["alternatives_considered", "date", "reversible"],
                retrieval_cue_template: "What was decided about {decision} and why?".into(),
            },
            "goal" => Self {
                memory_type: "goal".into(),
                required: vec!["objective"],
                optional: vec!["target_date", "milestones", "status", "blockers"],
                retrieval_cue_template: "What is the goal for {objective} and its status?".into(),
            },
            // Wildcard must be last — catches "fact" and unknown types
            _ => Self {
                memory_type: "fact".into(),
                required: vec!["claim"],
                optional: vec!["source", "verified", "domain"],
                retrieval_cue_template: "What do I know about {claim}?".into(),
            },
        }
    }

    pub fn validate(&self, fields: &HashMap<String, String>) -> Vec<String> {
        self.required
            .iter()
            .filter(|f| !fields.contains_key(**f) || fields[**f].is_empty())
            .map(|f| {
                format!(
                    "missing required field '{}' for {} memory",
                    f, self.memory_type
                )
            })
            .collect()
    }

    pub fn generate_retrieval_cue(&self, fields: &HashMap<String, String>) -> Option<String> {
        let mut cue = self.retrieval_cue_template.clone();
        for (key, value) in fields {
            let placeholder = format!("{{{}}}", key);
            cue = cue.replace(&placeholder, value);
        }
        // Check for unresolved schema placeholders specifically (not arbitrary braces in values)
        let has_unresolved = self
            .required
            .iter()
            .chain(self.optional.iter())
            .any(|f| cue.contains(&format!("{{{}}}", f)));
        if has_unresolved {
            None
        } else {
            Some(cue)
        }
    }
}

/// Build an LLM prompt that extracts structured fields for a given memory type.
/// Uses `prompts.extract_structured_fields` template if provided, otherwise uses compiled default.
pub fn extraction_prompt(memory_type: &str) -> String {
    extraction_prompt_with_template(
        memory_type,
        crate::prompts::defaults::EXTRACT_STRUCTURED_FIELDS,
    )
}

/// Build extraction prompt using a specific template string.
/// Template placeholders: {memory_type}, {fields_json}, {required}, {optional}
pub fn extraction_prompt_with_template(memory_type: &str, template: &str) -> String {
    let schema = MemorySchema::for_type(memory_type);
    let all_fields: Vec<&str> = schema
        .required
        .iter()
        .chain(schema.optional.iter())
        .copied()
        .collect();

    let fields_json: Vec<String> = all_fields
        .iter()
        .map(|f| format!("  \"{}\": \"...\"", f))
        .collect();

    template
        .replace("{memory_type}", memory_type)
        .replace("{fields_json}", &fields_json.join(",\n"))
        .replace("{required}", &schema.required.join(", "))
        .replace("{optional}", &schema.optional.join(", "))
}

/// Convert structured_fields JSON into a deterministic pipe-delimited string.
/// Returns None if JSON is invalid or empty. Keys are sorted for determinism.
pub fn flatten_structured_fields(json_str: &str) -> Option<String> {
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json_str).ok()?;
    if map.is_empty() {
        return None;
    }
    let mut pairs: Vec<(String, String)> = map
        .into_iter()
        .filter_map(|(k, v)| {
            let val = match v {
                serde_json::Value::String(s) if !s.is_empty() => s,
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Number(n) => n.to_string(),
                _ => return None,
            };
            Some((k, val))
        })
        .collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    if pairs.is_empty() {
        return None;
    }
    Some(
        pairs
            .into_iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect::<Vec<_>>()
            .join(" | "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_schema_required_fields() {
        let schema = MemorySchema::for_type("identity");
        assert_eq!(schema.required, vec!["claim"]);
    }

    #[test]
    fn test_all_types_have_schemas() {
        for t in ["identity", "preference", "decision", "fact", "goal"] {
            let schema = MemorySchema::for_type(t);
            assert!(!schema.required.is_empty(), "{} has no required fields", t);
            assert!(
                !schema.retrieval_cue_template.is_empty(),
                "{} has no retrieval cue",
                t
            );
        }
    }

    #[test]
    fn test_validate_complete_identity() {
        let schema = MemorySchema::for_type("identity");
        let mut fields = std::collections::HashMap::new();
        fields.insert("claim".to_string(), "I am a Rust developer".to_string());
        let warnings = schema.validate(&fields);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_missing_required_returns_warning() {
        let schema = MemorySchema::for_type("decision");
        let mut fields = std::collections::HashMap::new();
        fields.insert("decision".to_string(), "Use libSQL".to_string());
        let warnings = schema.validate(&fields);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("context"));
    }

    #[test]
    fn test_generate_retrieval_cue() {
        let schema = MemorySchema::for_type("fact");
        let mut fields = std::collections::HashMap::new();
        fields.insert(
            "claim".to_string(),
            "libsql Connection is Send but not Sync".to_string(),
        );
        let cue = schema.generate_retrieval_cue(&fields);
        assert!(cue.is_some());
        assert!(cue.unwrap().contains("libsql"));
    }

    #[test]
    fn test_unknown_type_falls_back_to_fact() {
        let schema = MemorySchema::for_type("nonexistent");
        assert_eq!(schema.memory_type, "fact");
    }

    #[test]
    fn test_extraction_prompt_for_identity() {
        let prompt = extraction_prompt("identity");
        assert!(prompt.contains("claim"));
        assert!(prompt.contains("evidence"));
        assert!(prompt.contains("retrieval_cue"));
    }

    #[test]
    fn test_extraction_prompt_for_decision() {
        let prompt = extraction_prompt("decision");
        assert!(prompt.contains("decision"));
        assert!(prompt.contains("context"));
        assert!(prompt.contains("alternatives_considered"));
    }

    #[test]
    fn test_flatten_structured_fields_preference() {
        let json =
            r#"{"preference":"dark mode","applies_when":"editors, terminals","strength":"strong"}"#;
        let result = flatten_structured_fields(json);
        assert!(result.is_some());
        let flat = result.unwrap();
        assert!(flat.contains("preference: dark mode"));
        assert!(flat.contains("applies_when: editors, terminals"));
        assert!(flat.contains("strength: strong"));
    }

    #[test]
    fn test_flatten_structured_fields_identity() {
        let json = r#"{"claim":"I am a Rust developer","evidence":"10 years experience"}"#;
        let result = flatten_structured_fields(json);
        assert!(result.is_some());
        assert!(result.unwrap().contains("claim: I am a Rust developer"));
    }

    #[test]
    fn test_flatten_structured_fields_empty_json() {
        let result = flatten_structured_fields("{}");
        assert!(result.is_none(), "empty object should return None");
    }

    #[test]
    fn test_flatten_structured_fields_invalid_json() {
        let result = flatten_structured_fields("not json");
        assert!(result.is_none());
    }

    #[test]
    fn test_flatten_structured_fields_deterministic_order() {
        let json = r#"{"preference":"dark mode","applies_when":"editors"}"#;
        let a = flatten_structured_fields(json).unwrap();
        let b = flatten_structured_fields(json).unwrap();
        assert_eq!(a, b, "flattening must be deterministic");
    }

    #[test]
    fn test_flatten_structured_fields_with_pipe_in_value() {
        let json = r#"{"claim":"A | B are both valid","source":"docs"}"#;
        let result = flatten_structured_fields(json).unwrap();
        assert!(result.contains("claim: A | B are both valid"));
    }
}
