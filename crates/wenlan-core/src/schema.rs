// SPDX-License-Identifier: Apache-2.0

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
                optional: vec!["evidence", "since", "event_date", "event_end"],
                retrieval_cue_template: "Who is the user in terms of {claim}?".into(),
            },
            "preference" => Self {
                memory_type: "preference".into(),
                required: vec!["preference", "applies_when"],
                optional: vec![
                    "strength",
                    "alternatives_rejected",
                    "event_date",
                    "event_end",
                ],
                retrieval_cue_template: "What does the user prefer regarding {preference}?".into(),
            },
            "decision" => Self {
                memory_type: "decision".into(),
                required: vec!["decision", "context"],
                optional: vec![
                    "alternatives_considered",
                    "date",
                    "reversible",
                    "event_date",
                    "event_end",
                ],
                retrieval_cue_template: "What was decided about {decision} and why?".into(),
            },
            // Wildcard must be last — catches "fact", legacy "goal", and unknown types
            _ => Self {
                memory_type: "fact".into(),
                required: vec!["claim"],
                optional: vec!["source", "verified", "domain", "event_date", "event_end"],
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
    let parts = split_structured_fields_to_facts(json_str);
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" | "))
    }
}

/// Split structured_fields JSON into one "key: value" string per scalar field
/// (T15a fact-channel producer). Sibling of [`flatten_structured_fields`]: same
/// serde parse, same scalar-coercion filter (non-empty String / Bool / Number),
/// same key-sort for determinism, but emits ONE child string per field instead
/// of a single pipe-joined line.
///
/// Returns `vec![]` (never a panic, never `vec![""]`) when the JSON is invalid,
/// empty, or carries no coercible scalar fields. Empty-string values are
/// filtered exactly like `flatten_structured_fields` (the PR-#147 silent-zero
/// guard class).
pub fn split_structured_fields_to_facts(json_str: &str) -> Vec<String> {
    let map: serde_json::Map<String, serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(m) => m,
        Err(_) => return Vec::new(),
    };
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
    pairs
        .into_iter()
        .map(|(k, v)| format!("{}: {}", k, v))
        .collect()
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
        for t in [
            "identity",
            "preference",
            "decision",
            "lesson",
            "gotcha",
            "fact",
        ] {
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

    #[test]
    fn every_type_has_event_date_and_event_end_in_optional() {
        // "fact", "lesson", "gotcha" all route through the _ wildcard arm
        for ty in [
            "identity",
            "preference",
            "decision",
            "fact",
            "lesson",
            "gotcha",
        ] {
            let s = MemorySchema::for_type(ty);
            assert!(
                s.optional.contains(&"event_date"),
                "{ty} missing event_date in optional"
            );
            assert!(
                s.optional.contains(&"event_end"),
                "{ty} missing event_end in optional"
            );
        }
    }

    #[test]
    fn extract_prompt_contains_event_date_for_every_type() {
        // Drift-guard: ensures that event_date and event_end survive the full
        // render pipeline. The schema test above only checks the optional Vec;
        // this test catches a dropped {optional} placeholder in the template.
        for ty in [
            "identity",
            "preference",
            "decision",
            "fact",
            "lesson",
            "gotcha",
        ] {
            let rendered = extraction_prompt(ty);
            assert!(
                rendered.contains("event_date"),
                "{ty} rendered EXTRACT prompt missing event_date"
            );
            assert!(
                rendered.contains("event_end"),
                "{ty} rendered EXTRACT prompt missing event_end"
            );
        }
    }

    #[test]
    fn for_type_decision_retains_existing_optional_fields() {
        let s = MemorySchema::for_type("decision");
        assert!(s.optional.contains(&"alternatives_considered"));
        assert!(s.optional.contains(&"date"));
        assert!(s.optional.contains(&"reversible"));
    }

    // ── T15a: split_structured_fields_to_facts ────────────────────────────────

    #[test]
    fn split_structured_fields_empty_returns_empty() {
        assert_eq!(split_structured_fields_to_facts("{}"), Vec::<String>::new());
        assert_eq!(
            split_structured_fields_to_facts("not json"),
            Vec::<String>::new()
        );
        assert_eq!(split_structured_fields_to_facts(""), Vec::<String>::new());
        // An object with only non-coercible values yields no children.
        assert_eq!(
            split_structured_fields_to_facts("{\"a\":null,\"b\":[1,2]}"),
            Vec::<String>::new()
        );
    }

    #[test]
    fn split_structured_fields_one_per_field() {
        // Two scalar fields -> exactly two children, key-sorted.
        let out = split_structured_fields_to_facts("{\"date\":\"2024-03-01\",\"city\":\"Tokyo\"}");
        assert_eq!(
            out,
            vec!["city: Tokyo".to_string(), "date: 2024-03-01".to_string()]
        );
    }

    #[test]
    fn split_structured_fields_stringifies_non_string_scalars() {
        // Bool + Number coerced like flatten_structured_fields.
        let out = split_structured_fields_to_facts("{\"active\":true,\"count\":3}");
        assert_eq!(
            out,
            vec!["active: true".to_string(), "count: 3".to_string()]
        );
    }

    #[test]
    fn split_structured_fields_skips_empty_values() {
        // Empty-string value filtered; only the non-empty field survives.
        let out = split_structured_fields_to_facts("{\"a\":\"\",\"b\":\"x\"}");
        assert_eq!(out, vec!["b: x".to_string()]);
    }
}
