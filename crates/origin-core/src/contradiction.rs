// SPDX-License-Identifier: AGPL-3.0-only
//! Structured field comparison for semantic contradiction pre-filtering.
//!
//! These functions are cheap (no LLM, no embedding) and serve as a
//! fast pre-filter before queuing the full LLM contradiction check.

use serde_json::Value;

/// Check if two memories of the same type may contradict based on structured fields.
/// Returns true if key fields overlap but values differ — a candidate for LLM contradiction check.
/// Returns false on parse failure, missing fields, or non-overlapping contexts.
pub fn fields_may_contradict(memory_type: &str, existing_json: &str, new_json: &str) -> bool {
    let existing: serde_json::Map<String, Value> = match serde_json::from_str(existing_json) {
        Ok(m) => m,
        Err(_) => return false,
    };
    let new: serde_json::Map<String, Value> = match serde_json::from_str(new_json) {
        Ok(m) => m,
        Err(_) => return false,
    };

    match memory_type {
        "preference" => compare_with_context(&existing, &new, "preference", "applies_when"),
        "identity" => {
            let e_claim = get_str(&existing, "claim");
            let n_claim = get_str(&new, "claim");
            match (e_claim, n_claim) {
                (Some(e), Some(n)) => e != n && bigram_jaccard(e, n) > 0.3,
                _ => false,
            }
        }
        "fact" => {
            let e_claim = get_str(&existing, "claim");
            let n_claim = get_str(&new, "claim");
            let e_domain = get_str(&existing, "domain");
            let n_domain = get_str(&new, "domain");
            match (e_claim, n_claim) {
                (Some(e), Some(n)) => {
                    let same_domain = match (e_domain, n_domain) {
                        (Some(ed), Some(nd)) => ed.to_lowercase() == nd.to_lowercase(),
                        _ => true,
                    };
                    same_domain && e != n && bigram_jaccard(e, n) > 0.3
                }
                _ => false,
            }
        }
        "decision" => compare_with_context(&existing, &new, "decision", "context"),
        "goal" => {
            let e_obj = get_str(&existing, "objective");
            let n_obj = get_str(&new, "objective");
            let e_status = get_str(&existing, "status");
            let n_status = get_str(&new, "status");
            match (e_obj, n_obj) {
                (Some(e), Some(n)) => bigram_jaccard(e, n) > 0.5 && e_status != n_status,
                _ => false,
            }
        }
        _ => false,
    }
}

fn compare_with_context(
    existing: &serde_json::Map<String, Value>,
    new: &serde_json::Map<String, Value>,
    value_key: &str,
    context_key: &str,
) -> bool {
    let e_val = get_str(existing, value_key);
    let n_val = get_str(new, value_key);
    let e_ctx = get_str(existing, context_key);
    let n_ctx = get_str(new, context_key);

    match (e_val, n_val, e_ctx, n_ctx) {
        (Some(ev), Some(nv), Some(ec), Some(nc)) => {
            bigram_jaccard(ec, nc) > 0.5 && ev.to_lowercase() != nv.to_lowercase()
        }
        (Some(ev), Some(nv), _, _) => ev.to_lowercase() != nv.to_lowercase(),
        _ => false,
    }
}

fn get_str<'a>(map: &'a serde_json::Map<String, Value>, key: &str) -> Option<&'a str> {
    map.get(key).and_then(|v| v.as_str())
}

pub fn bigram_jaccard(a: &str, b: &str) -> f64 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_bigrams: std::collections::HashSet<(char, char)> =
        a_lower.chars().zip(a_lower.chars().skip(1)).collect();
    let b_bigrams: std::collections::HashSet<(char, char)> =
        b_lower.chars().zip(b_lower.chars().skip(1)).collect();
    if a_bigrams.is_empty() && b_bigrams.is_empty() {
        return 1.0;
    }
    let intersection = a_bigrams.intersection(&b_bigrams).count();
    let union = a_bigrams.union(&b_bigrams).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preference_contradiction() {
        let existing = r#"{"preference":"dark mode","applies_when":"editors"}"#;
        let new = r#"{"preference":"light mode","applies_when":"editors"}"#;
        assert!(fields_may_contradict("preference", existing, new));
    }

    #[test]
    fn test_preference_no_contradiction_different_context() {
        let existing = r#"{"preference":"dark mode","applies_when":"editors"}"#;
        let new = r#"{"preference":"light mode","applies_when":"reading"}"#;
        assert!(!fields_may_contradict("preference", existing, new));
    }

    #[test]
    fn test_identity_contradiction() {
        let existing = r#"{"claim":"I am a Python developer"}"#;
        let new = r#"{"claim":"I am a Rust developer"}"#;
        assert!(fields_may_contradict("identity", existing, new));
    }

    #[test]
    fn test_fact_contradiction() {
        let existing = r#"{"claim":"libSQL uses SQLite","domain":"databases"}"#;
        let new = r#"{"claim":"libSQL uses PostgreSQL","domain":"databases"}"#;
        assert!(fields_may_contradict("fact", existing, new));
    }

    #[test]
    fn test_goal_status_update() {
        let existing = r#"{"objective":"launch Origin","status":"in_progress"}"#;
        let new = r#"{"objective":"launch Origin","status":"completed"}"#;
        assert!(fields_may_contradict("goal", existing, new));
    }

    #[test]
    fn test_invalid_json_returns_false() {
        assert!(!fields_may_contradict(
            "preference",
            "not json",
            "also not json"
        ));
    }

    #[test]
    fn test_missing_fields_returns_false() {
        let existing = r#"{"preference":"dark mode"}"#;
        let new = r#"{"other_field":"something"}"#;
        assert!(!fields_may_contradict("preference", existing, new));
    }
}
