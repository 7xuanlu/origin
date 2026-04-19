// SPDX-License-Identifier: AGPL-3.0-only

/// Classification result for an incoming query, used to route retrieval strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryClassification {
    /// Optional space/project scope extracted from the query (placeholder — always None for now).
    pub space: Option<String>,
    /// Whether the query should trigger knowledge-graph augmentation.
    pub use_graph: bool,
    /// Whether the response should compose multiple memory sources (true for context calls).
    pub compose: bool,
    /// Trust level inherited from the calling agent.
    pub trust_level: String,
}

/// Temporal phrases that signal a need for graph-backed temporal traversal.
/// Uses multi-word phrases to reduce false positives (e.g. "before" alone is too broad).
const TEMPORAL_KEYWORDS: &[&str] = &[
    "recently",
    "what changed",
    "history of",
    "when did",
    "latest",
    "last week",
    "last month",
    "yesterday",
    "updated recently",
    "evolved",
    "used to be",
    "timeline",
    "how has",
    "changed recently",
    "changed since",
];

/// Relational phrases that signal a need for graph-backed entity traversal.
/// Uses multi-word phrases to avoid false positives on common words like "between".
const RELATIONAL_KEYWORDS: &[&str] = &[
    "relationship between",
    "how does.*relate",
    "who works",
    "who knows",
    "involved in",
    "connected to",
];

/// Classify a query to determine the optimal retrieval strategy.
///
/// # Parameters
/// - `query`: the raw query string from the caller
/// - `agent_name`: name of the agent making the request (unused in classification, kept for logging)
/// - `agent_trust`: trust level string from the agent record (e.g. "full", "review", "unknown")
/// - `is_context_call`: true when this is a `/api/chat-context` call that needs composition
pub fn classify_query(
    query: &str,
    _agent_name: &str,
    agent_trust: &str,
    is_context_call: bool,
) -> QueryClassification {
    let lower = query.to_lowercase();

    let use_graph = TEMPORAL_KEYWORDS.iter().any(|kw| lower.contains(kw))
        || RELATIONAL_KEYWORDS.iter().any(|kw| lower.contains(kw));

    QueryClassification {
        space: None, // space detection is a future feature
        use_graph,
        compose: is_context_call,
        trust_level: agent_trust.to_string(),
    }
}

/// Check whether an agent with the given trust level is allowed to access a retrieval tier.
///
/// Tiers:
/// - 1 (most sensitive): requires "full" trust
/// - 2 (standard):       requires "full" or "review" trust
/// - 3 (public/safe):    all trust levels allowed
pub fn tier_allowed(trust_level: &str, tier: u8) -> bool {
    match tier {
        1 => trust_level == "full",
        2 => matches!(trust_level, "full" | "review"),
        3 => true,
        _ => false,
    }
}

/// Estimate token count for a piece of text.
///
/// Uses the heuristic: word_count * 1.3, rounded up.
pub fn estimate_tokens(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    (word_count as f64 * 1.3).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_keywords_trigger_graph() {
        let cases = [
            "What changed recently?",
            "Show me the history of this decision",
            "When did we update the API?",
            "What happened last week?",
            "Give me the timeline of the project",
        ];
        for q in &cases {
            let c = classify_query(q, "agent", "full", false);
            assert!(c.use_graph, "expected use_graph=true for: {q}");
        }
    }

    #[test]
    fn relational_keywords_trigger_graph() {
        let cases = [
            "What is the relationship between Alice and Bob?",
            "Who works on the backend team?",
            "Who is involved in the launch?",
            "Who knows about the deployment process?",
            "Is this connected to the auth system?",
        ];
        for q in &cases {
            let c = classify_query(q, "agent", "full", false);
            assert!(c.use_graph, "expected use_graph=true for: {q}");
        }
    }

    #[test]
    fn common_words_no_false_positive() {
        // These should NOT trigger graph — common English usage, not temporal/relational
        let cases = [
            "What is the current database schema?",
            "What should I do before deploying?",
            "What is the difference between X and Y?",
            "How do I connect to the database?",
        ];
        for q in &cases {
            let c = classify_query(q, "agent", "full", false);
            assert!(!c.use_graph, "expected use_graph=false for: {q}");
        }
    }

    #[test]
    fn simple_query_no_graph() {
        let c = classify_query("What is the database password?", "agent", "full", false);
        assert!(
            !c.use_graph,
            "simple factual query should not trigger graph"
        );
    }

    #[test]
    fn context_call_always_composes() {
        let c_ctx = classify_query("summarize my work", "agent", "full", true);
        assert!(
            c_ctx.compose,
            "is_context_call=true should set compose=true"
        );

        let c_no = classify_query("summarize my work", "agent", "full", false);
        assert!(
            !c_no.compose,
            "is_context_call=false should set compose=false"
        );
    }

    #[test]
    fn tier_visibility_full_trust() {
        assert!(tier_allowed("full", 1), "full trust should access tier 1");
        assert!(tier_allowed("full", 2), "full trust should access tier 2");
        assert!(tier_allowed("full", 3), "full trust should access tier 3");
    }

    #[test]
    fn tier_visibility_review_trust() {
        assert!(
            !tier_allowed("review", 1),
            "review trust must not access tier 1"
        );
        assert!(
            tier_allowed("review", 2),
            "review trust should access tier 2"
        );
        assert!(
            tier_allowed("review", 3),
            "review trust should access tier 3"
        );
    }

    #[test]
    fn tier_visibility_unknown_trust() {
        assert!(
            !tier_allowed("unknown", 1),
            "unknown trust must not access tier 1"
        );
        assert!(
            !tier_allowed("unknown", 2),
            "unknown trust must not access tier 2"
        );
        assert!(
            tier_allowed("unknown", 3),
            "unknown trust should access tier 3"
        );
    }

    #[test]
    fn token_estimation() {
        // 4 words → ceil(4 * 1.3) = ceil(5.2) = 6
        assert_eq!(estimate_tokens("hello world foo bar"), 6);
        // 0 words → 0
        assert_eq!(estimate_tokens(""), 0);
        // 1 word → ceil(1.3) = 2
        assert_eq!(estimate_tokens("hello"), 2);
        // 10 words → ceil(13.0) = 13
        assert_eq!(
            estimate_tokens("one two three four five six seven eight nine ten"),
            13
        );
    }
}
