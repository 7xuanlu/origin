// SPDX-License-Identifier: AGPL-3.0-only
//! Pre-store quality gate — rule-based content checks that reject noise before
//! it reaches the memory database. Covers system prompts, heartbeats, credential
//! leaks, and trivially short content.

use crate::tuning::GateConfig;
use regex::Regex;
use std::sync::LazyLock;
use std::time::Instant;

// ── Types ──────────────────────────────────────────────────────────────────

/// Pre-store quality gate that applies rule-based checks to incoming content.
#[derive(Debug, Clone)]
pub struct QualityGate {
    config: GateConfig,
}

/// Result of running content through the quality gate.
#[derive(Debug)]
pub struct GateResult {
    pub admitted: bool,
    pub reason: Option<RejectionReason>,
    pub scores: GateScores,
}

/// Why a piece of content was rejected.
#[derive(Debug, Clone)]
pub enum RejectionReason {
    NoisePattern(String),
    TooShort(usize),
    NotNovel(f64),
    CredentialLeak(String),
}

impl RejectionReason {
    /// Category string for logging/metrics.
    pub fn as_str(&self) -> &str {
        match self {
            Self::NoisePattern(_) => "noise_pattern",
            Self::TooShort(_) => "too_short",
            Self::NotNovel(_) => "not_novel",
            Self::CredentialLeak(_) => "credential_leak",
        }
    }

    /// Human-readable explanation.
    pub fn detail(&self) -> String {
        match self {
            Self::NoisePattern(p) => format!("Matched noise pattern: {p}"),
            Self::TooShort(n) => format!("Only {n} meaningful words (below minimum)"),
            Self::NotNovel(score) => {
                format!("Similarity {score:.2} above threshold — too similar to existing memory")
            }
            Self::CredentialLeak(kind) => format!("Contains credential: {kind}"),
        }
    }
}

/// Numeric scores produced by the gate for observability.
#[derive(Debug)]
pub struct GateScores {
    pub content_type_pass: bool,
    pub novelty_score: Option<f64>,
    pub word_count: usize,
    pub pattern_matched: Option<String>,
    pub latency_ms: u64,
}

// ── Credential regexes (compiled once) ─────────────────────────────────────

static RE_SK_LIVE_TEST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"sk_(?:live|test)_[A-Za-z0-9]{20,}").unwrap());
static RE_OPENAI_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"sk-[A-Za-z0-9]{32,}").unwrap());
static RE_GITHUB_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"gh[ps]_[A-Za-z0-9]{36,}").unwrap());
static RE_GITLAB_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"glpat-[A-Za-z0-9\-]{20,}").unwrap());
// Minimum 20 chars of token-alphabet after "Bearer " — stops false
// positives on documentation prose like "no bearer field" or "bearer
// token is required" while still catching real JWTs (~100+ chars) and
// opaque tokens. Matches the length floor used by the other credential
// regexes in this file.
static RE_BEARER_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{20,}=*").unwrap());
static RE_API_KEY_ASSIGN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(?:api_key|apikey|api-key|secret_key|password)\s*[=:]\s*["']?[A-Za-z0-9\-_.]{16,}"#,
    )
    .unwrap()
});

// ── Timestamp regex ────────────────────────────────────────────────────────

static RE_TIMESTAMP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}").unwrap());

// ── Helpers ────────────────────────────────────────────────────────────────

/// Count "meaningful" words: length > 1, or single-char alphanumeric.
fn meaningful_word_count(text: &str) -> usize {
    text.split_whitespace()
        .filter(|w| {
            let trimmed: String = w.chars().filter(|c| !c.is_ascii_punctuation()).collect();
            if trimmed.is_empty() {
                return false;
            }
            if trimmed.len() == 1 {
                return trimmed.chars().next().is_some_and(|c| c.is_alphanumeric());
            }
            true
        })
        .count()
}

/// Instruction-density keywords.
const INSTRUCTION_KEYWORDS: &[&str] = &["must", "always", "never", "ensure", "shall", "do not"];

/// System-prompt preamble prefixes (lowercase).
const PREAMBLE_PREFIXES: &[&str] = &[
    "you are a ",
    "you are an ",
    "your role is ",
    "as an ai",
    "as a helpful",
];

/// System-prompt markers.
const SYS_MARKERS: &[&str] = &["<<sys>>", "[/inst]", "### instruction"];

/// Single-word acknowledgements (lowercase, exact match).
const ACK_WORDS: &[&str] = &[
    "ok",
    "done",
    "ack",
    "received",
    "noted",
    "yes",
    "no",
    "confirmed",
    "acknowledged",
    "roger",
    "copy",
];

/// Heartbeat exact matches (lowercase).
const HEARTBEAT_EXACT: &[&str] = &[
    "heartbeat",
    "health check",
    "ping",
    "pong",
    "status: ok",
    "alive",
];

/// Heartbeat prefixes (lowercase).
const HEARTBEAT_PREFIXES: &[&str] = &["heartbeat ", "health check "];

// ── Noise pattern checks ───────────────────────────────────────────────────

/// Check if content looks like a system prompt preamble.
fn is_system_prompt_preamble(lower: &str) -> bool {
    PREAMBLE_PREFIXES.iter().any(|p| lower.starts_with(p))
}

/// Check if content contains system prompt markers.
fn has_system_prompt_markers(lower: &str) -> bool {
    if SYS_MARKERS.iter().any(|m| lower.contains(m)) {
        return true;
    }
    // "System:" + short content + 3+ instruction keywords
    if lower.starts_with("system:") {
        let kw_count = INSTRUCTION_KEYWORDS
            .iter()
            .filter(|kw| lower.contains(**kw))
            .count();
        if kw_count >= 3 {
            return true;
        }
    }
    false
}

/// Check for high instruction density in the first 200 chars.
fn has_instruction_density(lower: &str) -> bool {
    let prefix: String = lower.chars().take(200).collect();
    let count = INSTRUCTION_KEYWORDS
        .iter()
        .filter(|kw| prefix.contains(**kw))
        .count();
    count >= 4
}

/// Check for heartbeat/health-check messages.
fn is_heartbeat(lower: &str) -> bool {
    if HEARTBEAT_EXACT.contains(&lower) {
        return true;
    }
    HEARTBEAT_PREFIXES.iter().any(|p| lower.starts_with(p))
}

/// Check if content is mostly timestamps with <3 non-timestamp words.
fn is_timestamp_only(text: &str) -> bool {
    let mut non_ts_words = 0;
    for token in text.split_whitespace() {
        if !RE_TIMESTAMP.is_match(token) {
            // Also treat bare date/time fragments as timestamp-like
            let is_ts_fragment = token.len() >= 4
                && token
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '-' || c == ':' || c == 'T' || c == 'Z');
            if !is_ts_fragment {
                non_ts_words += 1;
            }
        }
    }
    non_ts_words < 3
}

/// Check for single-word acknowledgement.
fn is_single_word_ack(lower: &str) -> bool {
    ACK_WORDS.contains(&lower)
}

// ── Credential checks ──────────────────────────────────────────────────────

/// Returns the name of the first credential pattern matched, or None.
fn check_credentials(text: &str) -> Option<String> {
    if RE_SK_LIVE_TEST.is_match(text) {
        return Some("sk_live/test".to_string());
    }
    if RE_OPENAI_KEY.is_match(text) {
        return Some("openai_key".to_string());
    }
    if RE_GITHUB_TOKEN.is_match(text) {
        return Some("github_token".to_string());
    }
    if RE_GITLAB_TOKEN.is_match(text) {
        return Some("gitlab_token".to_string());
    }
    if let Some(m) = RE_BEARER_TOKEN.find(text) {
        // Require at least one non-alphabetic char in the token portion.
        // The length floor alone still matches 20+ char all-alpha English
        // words like "authenticationrequired". Real tokens (JWTs, opaque
        // base64, hex) always contain digits, dashes, dots, or slashes.
        let token = m.as_str().split_whitespace().nth(1).unwrap_or("");
        if token.chars().any(|c| !c.is_ascii_alphabetic()) {
            return Some("bearer_token".to_string());
        }
    }
    if RE_API_KEY_ASSIGN.is_match(text) {
        return Some("api_key_assignment".to_string());
    }
    None
}

// ── QualityGate implementation ─────────────────────────────────────────────

impl QualityGate {
    pub fn new(config: GateConfig) -> Self {
        Self { config }
    }

    /// Rule-based content check. Pure function, no async, no DB access.
    pub fn check_content(&self, content: &str) -> GateResult {
        let start = Instant::now();
        let lower = content.trim().to_lowercase();
        let word_count = meaningful_word_count(content.trim());

        // Gate disabled → admit everything
        if !self.config.enabled {
            return GateResult {
                admitted: true,
                reason: None,
                scores: GateScores {
                    content_type_pass: true,
                    novelty_score: None,
                    word_count,
                    pattern_matched: None,
                    latency_ms: start.elapsed().as_millis() as u64,
                },
            };
        }

        // 1. Minimum word count
        if word_count < self.config.min_word_count {
            return GateResult {
                admitted: false,
                reason: Some(RejectionReason::TooShort(word_count)),
                scores: GateScores {
                    content_type_pass: false,
                    novelty_score: None,
                    word_count,
                    pattern_matched: None,
                    latency_ms: start.elapsed().as_millis() as u64,
                },
            };
        }

        // 2. Credential check
        if self.config.credential_check_enabled {
            if let Some(kind) = check_credentials(content) {
                return GateResult {
                    admitted: false,
                    reason: Some(RejectionReason::CredentialLeak(kind.clone())),
                    scores: GateScores {
                        content_type_pass: false,
                        novelty_score: None,
                        word_count,
                        pattern_matched: Some(kind),
                        latency_ms: start.elapsed().as_millis() as u64,
                    },
                };
            }
        }

        // 3. Noise pattern checks
        if self.config.noise_patterns_enabled {
            #[allow(clippy::type_complexity)]
            let noise_checks: &[(&str, fn(&str) -> bool)] = &[
                ("system_prompt_preamble", |l| is_system_prompt_preamble(l)),
                ("system_prompt_markers", |l| has_system_prompt_markers(l)),
                ("instruction_density", |l| has_instruction_density(l)),
                ("heartbeat", |l| is_heartbeat(l)),
                ("single_word_ack", |l| is_single_word_ack(l)),
            ];

            for (name, check) in noise_checks {
                if check(&lower) {
                    return GateResult {
                        admitted: false,
                        reason: Some(RejectionReason::NoisePattern(name.to_string())),
                        scores: GateScores {
                            content_type_pass: false,
                            novelty_score: None,
                            word_count,
                            pattern_matched: Some(name.to_string()),
                            latency_ms: start.elapsed().as_millis() as u64,
                        },
                    };
                }
            }

            // timestamp_only uses the original text (not lowercase) for regex
            if is_timestamp_only(content.trim()) {
                return GateResult {
                    admitted: false,
                    reason: Some(RejectionReason::NoisePattern("timestamp_only".to_string())),
                    scores: GateScores {
                        content_type_pass: false,
                        novelty_score: None,
                        word_count,
                        pattern_matched: Some("timestamp_only".to_string()),
                        latency_ms: start.elapsed().as_millis() as u64,
                    },
                };
            }
        }

        // Novelty check (NotNovel) requires DB access — handled by evaluate() in a later step.
        // check_content() is pure/sync; evaluate() is async and adds the novelty gate.

        // All checks passed — admit
        GateResult {
            admitted: true,
            reason: None,
            scores: GateScores {
                content_type_pass: true,
                novelty_score: None,
                word_count,
                pattern_matched: None,
                latency_ms: start.elapsed().as_millis() as u64,
            },
        }
    }

    /// Full gate check including novelty (needs DB for embedding search).
    /// Returns (GateResult, Option<similar_source_id>).
    /// Batch version of [`evaluate`] used by the ingest coalescer.
    ///
    /// Runs the fast content-shape checks per-doc first (pure CPU — no DB,
    /// no embedding), then issues ONE `check_novelty_batch` call for all
    /// docs that passed the shape check. This amortizes the FastEmbed
    /// invocation across concurrent stores — the main per-request cost
    /// (~50–100 ms for one embedding) becomes the per-BATCH cost.
    ///
    /// Returns a parallel `Vec<(GateResult, Option<String>)>` in the same
    /// order as `contents`.
    pub async fn evaluate_batch(
        &self,
        contents: &[&str],
        db: &crate::db::MemoryDB,
    ) -> Result<Vec<(GateResult, Option<String>)>, crate::error::OriginError> {
        // Fast path: gate disabled. Admit every doc.
        if !self.config.enabled {
            return Ok(contents
                .iter()
                .map(|c| {
                    (
                        GateResult {
                            admitted: true,
                            reason: None,
                            scores: GateScores {
                                content_type_pass: true,
                                novelty_score: None,
                                word_count: c.split_whitespace().count(),
                                pattern_matched: None,
                                latency_ms: 0,
                            },
                        },
                        None,
                    )
                })
                .collect());
        }

        // Step 1: content checks for every doc.
        let mut per_doc: Vec<(GateResult, Option<String>)> = contents
            .iter()
            .map(|c| (self.check_content(c), None))
            .collect();

        // Step 2: gather indices still eligible for novelty, batch-embed
        // + novelty-query them together.
        let mut survivor_indices: Vec<usize> = Vec::new();
        let mut survivor_contents: Vec<String> = Vec::new();
        for (i, (r, _)) in per_doc.iter().enumerate() {
            if r.admitted {
                survivor_indices.push(i);
                survivor_contents.push(contents[i].to_string());
            }
        }

        if survivor_contents.is_empty() {
            return Ok(per_doc);
        }

        let started = std::time::Instant::now();
        let novelty_results = db.check_novelty_batch(&survivor_contents).await?;
        let elapsed_ms = started.elapsed().as_millis() as u64;

        // Amortize the batch latency across survivors so each
        // per-doc `scores.latency_ms` reflects the shared cost.
        let per_doc_ms = if survivor_indices.is_empty() {
            0
        } else {
            elapsed_ms / survivor_indices.len() as u64
        };

        for (idx, novelty) in survivor_indices.iter().zip(novelty_results) {
            let (result, similar) = &mut per_doc[*idx];
            result.scores.latency_ms = result.scores.latency_ms.saturating_add(per_doc_ms);
            if let Some((source_id, similarity)) = novelty {
                result.scores.novelty_score = Some(similarity);
                if similarity >= self.config.novelty_threshold {
                    result.admitted = false;
                    result.reason = Some(RejectionReason::NotNovel(similarity));
                }
                *similar = Some(source_id);
            }
        }
        Ok(per_doc)
    }

    pub async fn evaluate(
        &self,
        content: &str,
        db: &crate::db::MemoryDB,
    ) -> Result<(GateResult, Option<String>), crate::error::OriginError> {
        // Early exit if gate is disabled
        if !self.config.enabled {
            return Ok((
                GateResult {
                    admitted: true,
                    reason: None,
                    scores: GateScores {
                        content_type_pass: true,
                        novelty_score: None,
                        word_count: content.split_whitespace().count(),
                        pattern_matched: None,
                        latency_ms: 0,
                    },
                },
                None,
            ));
        }

        // Run content checks first (fast, no DB)
        let mut result = self.check_content(content);
        if !result.admitted {
            return Ok((result, None));
        }

        // Novelty check (requires DB)
        let start = std::time::Instant::now();
        let similar_source_id = match db.check_novelty(content).await? {
            Some((source_id, similarity)) => {
                result.scores.novelty_score = Some(similarity);
                if similarity >= self.config.novelty_threshold {
                    result.admitted = false;
                    result.reason = Some(RejectionReason::NotNovel(similarity));
                }
                Some(source_id)
            }
            None => {
                result.scores.novelty_score = None;
                None
            }
        };
        result.scores.latency_ms += start.elapsed().as_millis() as u64;
        Ok((result, similar_source_id))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> QualityGate {
        QualityGate::new(GateConfig::default())
    }

    #[test]
    fn test_rejects_too_short() {
        let g = gate();
        let r = g.check_content("hi");
        assert!(!r.admitted);
        assert!(matches!(r.reason, Some(RejectionReason::TooShort(_))));
        assert_eq!(r.scores.word_count, 1);
    }

    #[test]
    fn test_admits_sufficient_length() {
        let g = gate();
        let r = g.check_content("The user prefers dark mode in all applications");
        assert!(r.admitted);
        assert!(r.reason.is_none());
        assert!(r.scores.word_count >= 5);
    }

    #[test]
    fn test_rejects_system_prompt_preamble() {
        let g = gate();

        let cases = [
            "You are a helpful assistant that answers questions",
            "You are an AI trained by Anthropic to assist users",
            "Your role is to help users with their questions",
            "As an AI language model I cannot do that task",
            "As a helpful assistant I will answer your question",
        ];
        for case in &cases {
            let r = g.check_content(case);
            assert!(!r.admitted, "should reject: {case}");
            assert!(
                matches!(&r.reason, Some(RejectionReason::NoisePattern(p)) if p == "system_prompt_preamble"),
                "wrong reason for: {case}"
            );
        }
    }

    #[test]
    fn test_rejects_inst_markers() {
        let g = gate();

        let cases = [
            "Some text with <<SYS>> system prompt here <</SYS>>",
            "The [INST] instruction block contains user prompt data [/INST]",
            "### Instruction\nPlease summarize the following document content",
            "System: you must always ensure the user never sees errors and shall do not fail",
        ];
        for case in &cases {
            let r = g.check_content(case);
            assert!(!r.admitted, "should reject: {case}");
            assert!(
                matches!(&r.reason, Some(RejectionReason::NoisePattern(p)) if p == "system_prompt_markers"),
                "wrong reason for: {case}"
            );
        }
    }

    #[test]
    fn test_rejects_instruction_density() {
        let g = gate();
        let r = g.check_content(
            "You must always ensure the output never contains errors. You shall do not include PII.",
        );
        assert!(!r.admitted);
        assert!(
            matches!(&r.reason, Some(RejectionReason::NoisePattern(p)) if p == "instruction_density")
        );
    }

    #[test]
    fn test_rejects_heartbeat() {
        let g = gate();

        for hb in &[
            "heartbeat",
            "health check",
            "ping",
            "pong",
            "status: ok",
            "alive",
            "heartbeat 2026-03-30T10:00:00Z",
            "health check interval 30s",
        ] {
            let r = g.check_content(hb);
            assert!(!r.admitted, "should reject heartbeat: {hb}");
        }
    }

    #[test]
    fn test_rejects_single_word_ack() {
        let g = gate();
        for ack in &[
            "ok",
            "done",
            "ack",
            "received",
            "noted",
            "yes",
            "no",
            "confirmed",
            "acknowledged",
            "roger",
            "copy",
        ] {
            let r = g.check_content(ack);
            assert!(!r.admitted, "should reject ack: {ack}");
        }
    }

    #[test]
    fn test_rejects_timestamp_only() {
        let g = gate();
        let r = g.check_content(
            "2026-03-30T10:00:00Z 2026-03-30T11:00:00Z 2026-03-30T12:00:00Z 2026-03-30T13:00:00Z 2026-03-30T14:00:00Z log",
        );
        assert!(!r.admitted);
        assert!(
            matches!(&r.reason, Some(RejectionReason::NoisePattern(p)) if p == "timestamp_only"),
            "expected timestamp_only but got: {:?}",
            r.reason
        );
    }

    #[test]
    fn test_rejects_openai_key() {
        let g = gate();
        let r = g.check_content(
            "My API key is sk-abcdefghijklmnopqrstuvwxyz123456 please remember that",
        );
        assert!(!r.admitted);
        assert!(matches!(
            &r.reason,
            Some(RejectionReason::CredentialLeak(k)) if k == "openai_key"
        ));
    }

    #[test]
    fn test_rejects_github_token() {
        let g = gate();
        let r = g.check_content(
            "Use this token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl for GitHub access",
        );
        assert!(!r.admitted);
        assert!(matches!(
            &r.reason,
            Some(RejectionReason::CredentialLeak(k)) if k == "github_token"
        ));
    }

    #[test]
    fn test_admits_bearer_in_prose_without_token() {
        // Regression: documentation and discussion that mentions "bearer"
        // (including long all-alpha follow-on words) must not be flagged
        // as a credential leak when no actual token-shaped value follows.
        let g = gate();
        let cases = [
            "Claude.ai MCP connectors only accept OAuth, no bearer field.",
            "The bearer token header is required but we omit it here for brevity.",
            "This server uses Bearer auth for remote access.",
            // 29-char all-alpha word after Bearer — defeats the bare length
            // floor that the first version of this fix shipped with.
            "Bearer authorizationheaderisrequired when hitting remote endpoints.",
            // 20 chars, all letters, camelCase — still prose.
            "Bearer HeaderAuthIsRequired is what the docs call it.",
        ];
        for text in cases {
            let r = g.check_content(text);
            assert!(
                !matches!(&r.reason, Some(RejectionReason::CredentialLeak(_))),
                "should not flag bearer-in-prose as credential leak: {text:?} -> {:?}",
                r.reason
            );
        }
    }

    #[test]
    fn test_rejects_bearer_token_with_single_digit() {
        // Boundary: a 20-char token that contains even one non-alpha char
        // (here a digit) MUST still trip the credential detector. This
        // proves the non-alpha post-check hasn't turned the detector off.
        let g = gate();
        let r = g.check_content("Authorization: Bearer abcdefghijklmnopqrs9 for the API call");
        assert!(
            matches!(&r.reason, Some(RejectionReason::CredentialLeak(k)) if k == "bearer_token"),
            "20-char token with a digit must still be detected; got: {:?}",
            r.reason
        );
    }

    #[test]
    fn test_rejects_bearer_token() {
        let g = gate();
        let r = g.check_content(
            "Please use this header for the request: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abcdef.ghijkl",
        );
        assert!(!r.admitted);
        assert!(matches!(
            &r.reason,
            Some(RejectionReason::CredentialLeak(k)) if k == "bearer_token"
        ));
    }

    #[test]
    fn test_rejects_sk_live_key() {
        let r = gate().check_content(
            "The Stripe key is sk_live_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef in production config",
        );
        assert!(!r.admitted);
        assert!(matches!(r.reason, Some(RejectionReason::CredentialLeak(_))));
    }

    #[test]
    fn test_rejects_gitlab_token() {
        let r = gate().check_content(
            "Use this GitLab token glpat-ABCDEFGHIJKLMNOPQRSTUVwxyz for CI pipeline access",
        );
        assert!(!r.admitted);
        assert!(matches!(r.reason, Some(RejectionReason::CredentialLeak(_))));
    }

    #[test]
    fn test_rejects_api_key_assignment() {
        let r = gate().check_content("Configuration has api_key = 'sk_production_key_12345678' in the environment variables file");
        assert!(!r.admitted);
        assert!(matches!(r.reason, Some(RejectionReason::CredentialLeak(_))));
    }

    #[test]
    fn test_admits_real_preference() {
        let g = gate();
        let r = g.check_content("Lucian prefers dark mode and uses Neovim as his primary editor");
        assert!(r.admitted);
    }

    #[test]
    fn test_admits_real_decision() {
        let g = gate();
        let r = g.check_content(
            "We decided to keep libSQL instead of switching to a dedicated graph database",
        );
        assert!(r.admitted);
    }

    #[test]
    fn test_admits_real_fact() {
        let g = gate();
        let r = g.check_content("Origin uses Tauri 2 with a Rust backend and React 19 frontend");
        assert!(r.admitted);
    }

    #[test]
    fn test_disabled_gate_admits_everything() {
        let config = GateConfig {
            enabled: false,
            ..GateConfig::default()
        };
        let g = QualityGate::new(config);

        // Even a heartbeat should be admitted when gate is disabled
        let r = g.check_content("ping");
        assert!(r.admitted);
        assert!(r.reason.is_none());
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_evaluate_admits_novel_content() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let g = QualityGate::new(GateConfig::default());
        let (r, _similar): (GateResult, Option<String>) = g
            .evaluate("User prefers Rust for all backend services", &db)
            .await
            .unwrap();
        assert!(r.admitted, "Novel content in empty DB should be admitted");
    }

    #[tokio::test]
    async fn test_evaluate_rejects_duplicate() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let g = QualityGate::new(GateConfig::default());

        // Seed a memory
        let doc = crate::sources::RawDocument {
            content: "User prefers dark mode for all IDEs".to_string(),
            source_id: "mem_existing".to_string(),
            source: "memory".to_string(),
            title: "Dark mode".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        // Exact duplicate should be rejected
        let (r, similar): (GateResult, Option<String>) = g
            .evaluate("User prefers dark mode for all IDEs", &db)
            .await
            .unwrap();
        assert!(!r.admitted, "Exact duplicate should be rejected");
        assert!(matches!(r.reason, Some(RejectionReason::NotNovel(_))));
        assert!(similar.is_some());
    }

    #[tokio::test]
    async fn test_evaluate_content_check_runs_first() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let g = QualityGate::new(GateConfig::default());
        // Short/noise content should be caught by content check before novelty
        let (r, _similar): (GateResult, Option<String>) = g.evaluate("You are a helpful AI assistant. Always respond politely and accurately to all queries.", &db).await.unwrap();
        assert!(!r.admitted);
        // Should be noise pattern, NOT NotNovel
        assert!(matches!(
            r.reason,
            Some(RejectionReason::NoisePattern(_)) | Some(RejectionReason::TooShort(_))
        ));
        // Novelty should not have run
        assert!(
            r.scores.novelty_score.is_none(),
            "Novelty should not run after content rejection"
        );
    }

    // FIXME: Flaky timing test — depends on system load. Threshold is 2000ms
    // but regularly exceeds it on loaded machines (e.g., during parallel builds).
    #[ignore]
    #[tokio::test]
    async fn test_evaluate_latency_under_500ms() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let g = QualityGate::new(GateConfig::default());

        let start = std::time::Instant::now();
        let (r, _similar): (GateResult, Option<String>) = g.evaluate(
            "A legitimate memory about project architecture decisions that should pass all checks",
            &db,
        ).await.unwrap();
        let elapsed = start.elapsed();

        assert!(r.admitted);
        assert!(
            elapsed.as_millis() < 2000,
            "Gate took {}ms, expected <2000ms",
            elapsed.as_millis()
        );
    }

    #[tokio::test]
    async fn test_rejection_logging_roundtrip() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let g = QualityGate::new(GateConfig::default());

        // Evaluate something that should be rejected by content check
        let (r, _) = g.evaluate("You are a helpful AI assistant. Your role is to answer all questions helpfully and accurately.", &db).await.unwrap();
        assert!(!r.admitted);

        // Log the rejection (simulating what handle_store_memory does)
        if let Some(ref reason) = r.reason {
            db.log_rejection(
                "rej_roundtrip_test",
                "You are a helpful AI assistant...",
                Some("test-agent"),
                reason.as_str(),
                Some(&reason.detail()),
                r.scores.novelty_score,
                None,
            )
            .await
            .unwrap();
        }

        // Verify the rejection was logged
        let rejections = db.get_rejections(10, None).await.unwrap();
        assert_eq!(rejections.len(), 1);
        assert_eq!(rejections[0].id, "rej_roundtrip_test");
        assert_eq!(rejections[0].rejection_reason, "noise_pattern");
        assert_eq!(rejections[0].source_agent.as_deref(), Some("test-agent"));
        assert!(rejections[0].rejection_detail.is_some());
    }

    /// Comprehensive eval: feed noise + legitimate content, measure rejection accuracy.
    /// This is the "show me the eval" test — real data, real measurements.
    #[test]
    fn test_eval_gate_precision() {
        let g = QualityGate::new(GateConfig::default());

        // ── NOISE: should ALL be rejected ──
        let noise: Vec<(&str, &str)> = vec![
            // System prompt restating (52.7% of Mem0 junk)
            ("You are a helpful AI coding assistant. Your role is to write clean, tested Rust code.", "sys_prompt"),
            ("You are an expert software engineer. Always follow best practices and write documentation.", "sys_prompt"),
            ("Your role is to assist developers with debugging and code review tasks accurately.", "sys_prompt"),
            ("As an AI assistant, you should provide accurate and helpful responses to all queries.", "sys_prompt"),
            ("[INST] Always respond in JSON format. Never include personal opinions. Ensure all outputs are valid. [/INST]", "sys_prompt"),
            ("You must always ensure the output is correct. Never skip validation. You shall always handle errors. Do not ignore edge cases.", "instruction_density"),
            ("As a helpful coding assistant I will review your pull request carefully.", "sys_prompt"),
            ("You are a specialized code review bot trained on open source repositories.", "sys_prompt"),
            ("Your role is to analyze code changes and provide constructive feedback.", "sys_prompt"),
            ("As an AI assistant I aim to provide thorough and accurate code analysis.", "sys_prompt"),
            ("You are an advanced reasoning engine trained on code and documentation.", "sys_prompt"),
            ("Your role is to generate high-quality test cases for the given codebase.", "sys_prompt"),
            ("As an AI model I am designed to assist with software engineering tasks.", "sys_prompt"),
            ("As a helpful assistant I will help you debug and fix the failing tests.", "sys_prompt"),
            ("You are a code generation assistant. Your role is to write idiomatic Rust.", "sys_prompt"),
            // Heartbeat / cron (11.5%)
            ("heartbeat", "heartbeat"),
            ("health check", "heartbeat"),
            ("ping", "heartbeat"),
            ("status: ok", "heartbeat"),
            ("alive", "heartbeat"),
            ("pong", "heartbeat"),
            ("heartbeat 2026-03-30T10:00:00Z connection stable", "heartbeat"),
            ("health check interval 30s response time normal", "heartbeat"),
            ("2026-03-30T14:22:00Z ok", "timestamp"),
            ("2026-03-30T10:00:00Z 2026-03-30T11:00:00Z 2026-03-30T12:00:00Z 2026-03-30T13:00:00Z", "timestamp"),
            // Single-word ack
            ("ok", "ack"),
            ("done", "ack"),
            ("acknowledged", "ack"),
            ("received", "ack"),
            ("noted", "ack"),
            ("yes", "ack"),
            ("no", "ack"),
            ("confirmed", "ack"),
            ("roger", "ack"),
            ("copy", "ack"),
            // Too short
            ("yes no", "too_short"),
            ("fix it", "too_short"),
            ("hmm ok", "too_short"),
            ("nope", "too_short"),
            ("ok sure", "too_short"),
            ("got it", "too_short"),
            // Credentials (2.1%)
            ("My API key is sk-abc123def456ghi789jkl012mno345pqr678stu901 for the production environment", "credential"),
            ("Use token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij to access the private repository resources", "credential"),
            ("The stripe key is sk_live_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef stored in the environment config file", "credential"),
            ("Set api_key = 'production_secret_key_12345678' in the configuration file for authentication purposes", "credential"),
            ("The GitLab personal access token is glpat-ABCDEFGHIJKLMNOPQRSTUVwxyz for CI pipeline deployment", "credential"),
            ("Use this bearer token for auth: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig in all requests", "credential"),
            ("The test Stripe key is sk_test_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef for sandbox transactions", "credential"),
            ("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature for API access", "credential"),
            ("Store secret_key = 'super_secret_production_value_999' in the vault for secure access", "credential"),
            // More system prompt variations
            ("You are a memory management system. Your role is to store and retrieve information.", "sys_prompt"),
            ("As a helpful AI I must always ensure accuracy and never make assumptions.", "sys_prompt"),
            ("Your role is to provide detailed answers to all user queries about code.", "sys_prompt"),
            ("You are an intelligent assistant trained to never refuse reasonable requests.", "sys_prompt"),
            ("As an AI language model I should always be transparent about my limitations.", "sys_prompt"),
            // More instruction density
            ("You must never skip tests. Always ensure coverage. You shall never ignore errors. Do not forget documentation.", "instruction_density"),
            ("Never deploy without review. You must always run linting. Ensure all checks pass. You shall never merge directly.", "instruction_density"),
            // More heartbeat variations
            ("heartbeat interval 60 seconds check result healthy", "heartbeat"),
            ("health check endpoint responding with 200 status code", "heartbeat"),
            // More too-short
            ("thanks", "too_short"),
            ("sure thing", "too_short"),
        ];

        // ── LEGITIMATE: should ALL be admitted ──
        let legit: Vec<&str> = vec![
            "User prefers dark mode for all code editors and terminal applications",
            "Decided to use libSQL instead of PostgreSQL because it supports embedded vector search natively",
            "The Origin app uses Tauri 2 with a React 19 frontend and Rust backend running on macOS",
            "Project deadline is April 15 for the v1 launch milestone with three features remaining",
            "User is a senior software engineer specializing in Rust and distributed systems",
            "Always run cargo test before committing to avoid breaking the CI pipeline checks",
            "The database migration from SQLite to libSQL was completed in February with zero data loss",
            "Lucian prefers TDD workflow with separate agents for test writing versus implementation code",
            "Origin uses FastEmbed bge-base-en-v1.5 for embeddings with 768 dimensions and DiskANN indexing",
            "The memory distillation pipeline clusters similar memories and generates consolidated summaries",
            "Knowledge graph uses entities relations and observations tables with FK cascade deletes enabled",
            "Remote access tunnel connects via Cloudflare Worker relay for stable MCP URLs to agents",
            "The refinery runs periodic steep cycles every 2 hours to process entity extraction and distillation",
            "Search uses hybrid retrieval combining vector similarity and FTS5 with reciprocal rank fusion",
            "Agent trust levels control which agents can write to the memory store with auto-registration enabled",
            "Prefers 4-space indentation over tabs in all Python files and projects",
            "Uses vim keybindings in VS Code for faster text navigation and editing",
            "The team standup happens at 9:30 AM Pacific every weekday morning",
            "Chose PostgreSQL over MySQL for the analytics database due to better JSON support",
            "Selected React Query over Redux for server state management to reduce boilerplate",
            "Works remotely from Austin Texas in the Central timezone year round",
            "Has a golden retriever named Max who needs walking at noon every day",
            "Born in Romania and moved to the United States for graduate school",
            "The API rate limit is 100 requests per minute per user in production",
            "The CI pipeline takes approximately 12 minutes to run all tests and linting",
            "The staging environment uses the same Docker image as production for parity",
            "Planning to migrate the frontend from JavaScript to TypeScript by Q3 this year",
            "Goal is to reach 1000 weekly active users before the Series A fundraise",
            "Wants to add end-to-end encryption for memory sync across devices by next quarter",
            "The chunking strategy uses 512 token windows with 64 token overlap for context",
            "PII redaction covers credit cards SSN numbers API keys and email addresses",
            "The toast window uses a non-activating panel so it never steals keyboard focus",
            "Cargo build cache is shared across git worktrees for faster incremental compilation",
            "The LLM formatter thread captures a tokio runtime handle for async database calls",
            "Config is persisted as JSON at the standard macOS Application Support directory path",
            "Meeting with the design team is scheduled for every Wednesday at two PM Pacific",
            "The project uses AGPL-3.0-only license for the main app and MIT for the MCP server",
            "Batch SQL operations should always be wrapped in BEGIN COMMIT transactions for atomicity",
            "The ambient capture system is disabled by default as part of the memory-layer pivot",
            "Frame comparison uses a 3-tier approach with downscaling hashing and Hellinger distance",
        ];

        let total_noise = noise.len();
        let total_legit = legit.len();

        // ── Run noise through gate ──
        let mut noise_rejected = 0;
        let mut noise_reasons: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut noise_false_negatives: Vec<String> = Vec::new();
        let mut total_noise_latency_us = 0u128;

        for (content, _category) in &noise {
            let start = Instant::now();
            let r = g.check_content(content);
            total_noise_latency_us += start.elapsed().as_micros();

            if !r.admitted {
                noise_rejected += 1;
                let reason_key = r
                    .reason
                    .as_ref()
                    .map(|r| r.as_str().to_string())
                    .unwrap_or_default();
                *noise_reasons.entry(reason_key).or_default() += 1;
            } else {
                noise_false_negatives.push(content.to_string());
            }
        }

        // ── Run legitimate through gate ──
        let mut legit_admitted = 0;
        let mut legit_false_positives: Vec<String> = Vec::new();
        let mut total_legit_latency_us = 0u128;

        for content in &legit {
            let start = Instant::now();
            let r = g.check_content(content);
            total_legit_latency_us += start.elapsed().as_micros();

            if r.admitted {
                legit_admitted += 1;
            } else {
                let reason = r.reason.as_ref().map(|r| r.detail()).unwrap_or_default();
                legit_false_positives.push(format!("{}: {}", content, reason));
            }
        }

        // ── Print results ──
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║          QUALITY GATE EVAL RESULTS                      ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!("║                                                          ║");
        println!("║  NOISE REJECTION                                         ║");
        println!(
            "║    Samples:     {:<4}                                     ║",
            total_noise
        );
        println!(
            "║    Rejected:    {:<4} ({:.1}%)                             ║",
            noise_rejected,
            (noise_rejected as f64 / total_noise as f64) * 100.0
        );
        println!(
            "║    Missed:      {:<4}                                     ║",
            total_noise - noise_rejected
        );
        println!("║                                                          ║");
        println!("║  Rejection breakdown:                                    ║");
        for (reason, count) in &noise_reasons {
            println!(
                "║    {:<20} {:<4} ({:.1}%)                       ║",
                reason,
                count,
                (*count as f64 / total_noise as f64) * 100.0
            );
        }
        println!("║                                                          ║");
        println!("║  LEGITIMATE ADMISSION                                    ║");
        println!(
            "║    Samples:     {:<4}                                     ║",
            total_legit
        );
        println!(
            "║    Admitted:    {:<4} ({:.1}%)                             ║",
            legit_admitted,
            (legit_admitted as f64 / total_legit as f64) * 100.0
        );
        println!(
            "║    False pos:   {:<4}                                     ║",
            legit_false_positives.len()
        );
        println!("║                                                          ║");
        println!("║  METRICS                                                 ║");
        println!(
            "║    Gate precision:   {:.1}%                                ║",
            if noise_rejected + legit_admitted > 0 {
                (noise_rejected as f64
                    / (noise_rejected as f64 + legit_false_positives.len() as f64))
                    * 100.0
            } else {
                0.0
            }
        );
        println!(
            "║    Gate recall:      {:.1}%                                ║",
            (noise_rejected as f64 / total_noise as f64) * 100.0
        );
        println!(
            "║    False pos rate:   {:.1}%                                ║",
            (legit_false_positives.len() as f64 / total_legit as f64) * 100.0
        );
        println!(
            "║    Noise latency:    {:.0}μs avg ({:.2}ms total)            ║",
            total_noise_latency_us as f64 / total_noise as f64,
            total_noise_latency_us as f64 / 1000.0
        );
        println!(
            "║    Legit latency:    {:.0}μs avg ({:.2}ms total)            ║",
            total_legit_latency_us as f64 / total_legit as f64,
            total_legit_latency_us as f64 / 1000.0
        );
        println!("║                                                          ║");
        println!("╚══════════════════════════════════════════════════════════╝");

        if !noise_false_negatives.is_empty() {
            println!("\n⚠ NOISE FALSE NEGATIVES (should have been rejected):");
            for s in &noise_false_negatives {
                println!("  - {}", s);
            }
        }
        if !legit_false_positives.is_empty() {
            println!("\n⚠ LEGITIMATE FALSE POSITIVES (should have been admitted):");
            for s in &legit_false_positives {
                println!("  - {}", s);
            }
        }

        // Hard assertions
        assert_eq!(
            noise_rejected, total_noise,
            "Not all noise was rejected. Missed: {:?}",
            noise_false_negatives
        );
        assert_eq!(
            legit_admitted, total_legit,
            "Some legitimate content was rejected. FP: {:?}",
            legit_false_positives
        );
        assert_eq!(legit_false_positives.len(), 0, "False positives detected");
    }

    /// Comprehensive eval: novelty gate catches semantic duplicates while admitting genuinely
    /// different content. 20 seeds × (2-3 paraphrases each) + 30 different samples.
    /// Prints threshold scan to expose the precision/recall tradeoff.
    #[tokio::test]
    async fn test_eval_novelty_gate() {
        let (db, _dir) = crate::db::tests::test_db().await;
        let g = QualityGate::new(GateConfig::default());
        let threshold = 0.75_f64;

        // ── Seed memories (20 diverse memories covering different types) ──
        let seeds: &[(&str, &str)] = &[
            // Programming preferences
            ("mem_dark",    "User prefers dark mode for all IDEs and terminal applications"),
            ("mem_rust",    "User prefers Rust for backend development and TypeScript for frontend code"),
            ("mem_vim",     "Primary editor is Neovim with LSP support configured for Rust and TypeScript"),
            ("mem_fmt",     "Auto-format on save is enabled using rustfmt for Rust and prettier for TypeScript"),
            // Architecture decisions
            ("mem_db",      "Decided to use libSQL for the embedded database because it supports vectors FTS5 and knowledge graph natively"),
            ("mem_tauri",   "Origin uses Tauri 2 with React 19 frontend and Rust backend on macOS"),
            ("mem_embed",   "Embeddings use FastEmbed bge-base-en-v1.5 model producing 768-dimensional vectors stored in DiskANN index"),
            ("mem_deploy",  "Application is distributed as a macOS .app bundle built via GitHub Actions CI pipeline"),
            // Personal facts
            ("mem_role",    "Lucian is a senior software engineer specializing in Rust distributed systems and developer tooling"),
            ("mem_tz",      "User is based in San Francisco and works Pacific Time hours starting around nine AM"),
            ("mem_team",    "Works on a two-person founding team with a focus on indie developer tooling products"),
            ("mem_schedule","Deep work blocks are scheduled in the morning and meetings are batched to afternoons"),
            // Project facts
            ("mem_tdd",     "Always write tests before implementation code following strict TDD workflow with separate agents"),
            ("mem_deadline","v1 launch target is end of April with ChatGPT integration as the highest priority feature"),
            ("mem_stack",   "Frontend uses React 19 with TanStack Query 5 Tailwind CSS 4 and Vite as the build tool"),
            ("mem_mcp",     "Origin exposes an MCP server on port 7878 that agents connect to for reading and writing memories"),
            // Identity facts
            ("mem_name",    "User full name is Lucian and preferred pronoun is he slash him in all written communications"),
            ("mem_style",   "Communication style is direct and concise with preference for bullet points over long prose"),
            ("mem_expert",  "Core expertise spans Rust systems programming database internals and local-first application architecture"),
            ("mem_lang",    "Speaks English as primary language and has conversational proficiency in Romanian"),
        ];

        for (id, content) in seeds {
            let doc = crate::sources::RawDocument {
                content: content.to_string(),
                source_id: id.to_string(),
                source: "memory".to_string(),
                title: id.to_string(),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        // ── Duplicates: 2-3 paraphrases per seed at varying semantic distances ──
        // Label: "near-exact" | "structural" | "semantic" | "partial"
        let duplicates: &[(&str, &str, &str)] = &[
            // mem_dark paraphrases
            ("User prefers dark mode for all IDEs and terminal applications",
             "near-exact", "mem_dark"),
            ("The user likes dark theme in all code editors and terminal emulators",
             "structural", "mem_dark"),
            ("Dark color schemes are strongly preferred for every development environment including the terminal",
             "semantic", "mem_dark"),

            // mem_rust paraphrases
            ("User prefers Rust for backend development and TypeScript for frontend code",
             "near-exact", "mem_rust"),
            ("For backend work Rust is the preferred language while frontend uses TypeScript",
             "structural", "mem_rust"),
            ("The team uses TypeScript on the frontend and Rust on the backend as their primary languages",
             "semantic", "mem_rust"),

            // mem_vim paraphrases
            ("Primary editor is Neovim with LSP support configured for Rust and TypeScript",
             "near-exact", "mem_vim"),
            ("Neovim is the main code editor with language server protocol set up for Rust and TypeScript",
             "structural", "mem_vim"),
            ("Development happens in Neovim using LSP-based completions and diagnostics for both languages",
             "semantic", "mem_vim"),

            // mem_db paraphrases
            ("We chose libSQL as our embedded database because it supports vectors full-text search and a knowledge graph",
             "structural", "mem_db"),
            ("libSQL was selected over alternatives because it natively handles vector similarity FTS5 and graph relations",
             "semantic", "mem_db"),

            // mem_tauri paraphrases
            ("Origin is built with Tauri 2 using a React 19 UI layer and a Rust backend targeting macOS",
             "structural", "mem_tauri"),
            ("The desktop app combines a Rust Tauri backend with a React frontend and runs on macOS",
             "semantic", "mem_tauri"),

            // mem_embed paraphrases
            ("Embedding model is FastEmbed bge-base-en-v1.5 outputting 768-dim vectors indexed with DiskANN",
             "near-exact", "mem_embed"),
            ("Vector embeddings are 768-dimensional produced by the bge-base-en-v1.5 model and stored in a DiskANN index",
             "structural", "mem_embed"),

            // mem_role paraphrases
            ("Lucian is a senior software engineer who specializes in Rust distributed systems and developer tools",
             "near-exact", "mem_role"),
            ("The user is a senior engineer with deep expertise in Rust and building distributed developer tooling",
             "structural", "mem_role"),

            // mem_tdd paraphrases
            ("Tests are always written before implementation following strict TDD with separate agents per task",
             "structural", "mem_tdd"),
            ("Strict test-driven development means tests come first and separate Claude agents handle tests versus code",
             "semantic", "mem_tdd"),

            // mem_deadline paraphrases
            ("v1 launch is targeted for the end of April with the ChatGPT feature as top priority",
             "near-exact", "mem_deadline"),
            ("The April deadline for v1 means ChatGPT integration must ship first as it has highest impact",
             "semantic", "mem_deadline"),

            // mem_mcp paraphrases
            ("Origin runs an MCP server on port 7878 that external agents use to store and retrieve memories",
             "structural", "mem_mcp"),
            ("Agents connect to the MCP server at port 7878 to read from and write memories into Origin",
             "semantic", "mem_mcp"),

            // mem_name paraphrases
            ("User name is Lucian and he slash him pronouns should be used in all written content",
             "near-exact", "mem_name"),
            ("Always refer to the user as Lucian using he and him as pronouns in any written communication",
             "semantic", "mem_name"),

            // mem_style paraphrases
            ("Communication is direct and concise preferring bullet points over lengthy paragraph prose",
             "structural", "mem_style"),
            ("The user values brevity and bullet-point formatting over long narrative explanations",
             "semantic", "mem_style"),

            // mem_expert paraphrases
            ("Core technical skills cover Rust systems work database internals and local-first app design",
             "structural", "mem_expert"),
            ("Deep expertise in systems programming with Rust storage engines and offline-first application patterns",
             "semantic", "mem_expert"),
        ];

        // ── Different: genuinely new facts related to the same topic areas ──
        let different: &[(&str, &str)] = &[
            // Same domain (editor prefs), different fact
            ("VS Code is used as a secondary editor when pair programming with colleagues remotely",
             "different editor fact"),
            ("Terminal color scheme is Catppuccin Mocha with JetBrains Mono as the monospace font",
             "different terminal customization"),

            // Same domain (languages), different fact
            ("Python is used for data science scripts and quick prototyping but not for production services",
             "different language use case"),
            ("Go is evaluated as a potential language for high-throughput networking utilities alongside Rust",
             "different language evaluation"),

            // Same domain (database), different fact
            ("Database migrations are managed manually using SQL scripts checked into the repository",
             "different DB fact"),
            ("The memories table stores document fragments with a 768-dimensional F32 blob vector column",
             "different schema fact"),
            ("Hybrid search combines cosine vector similarity with BM25 full-text scores using RRF fusion",
             "different search fact"),

            // Same domain (architecture), different fact
            ("The app listens on a Unix socket in addition to TCP port 7878 for local IPC connections",
             "different server fact"),
            ("Window management uses a multi-window Tauri setup with four separate window types for different views",
             "different window fact"),
            ("Axum 0.8 is the HTTP framework used to build the REST API inside the Tauri backend process",
             "different framework fact"),

            // Same domain (personal), different fact
            ("Primary timezone for scheduling is US Pacific and calendar is blocked from noon to one PM daily",
             "different schedule fact"),
            ("User holds a computer science degree and previously worked at two Bay Area startups before going indie",
             "different background fact"),

            // Same domain (project), different fact
            ("The app icon and marketing site are planned for completion two weeks before the April launch date",
             "different project timeline"),
            ("CI pipeline runs on GitHub Actions and caches Cargo build artifacts to speed up Rust compilation",
             "different CI fact"),
            ("User research interviews are scheduled weekly to validate memory pain points with target customers",
             "different process fact"),

            // Same domain (communication), different fact
            ("Meeting notes are captured in Obsidian and synced with the Origin memory store after each call",
             "different note-taking fact"),
            ("Status updates are shared as concise bullet-point summaries no longer than five lines per update",
             "different comms fact"),

            // Tangentially related via keyword overlap only
            ("The Rust compiler error messages are detailed enough to serve as documentation for new contributors",
             "different Rust observation"),
            ("DiskANN provides approximate nearest neighbor search with better recall than HNSW at high dimensions",
             "different vector index fact"),
            ("Tauri apps on macOS bundle a WebView2 equivalent through WKWebView provided by the operating system",
             "different Tauri detail"),
            ("bge-base-en-v1.5 was chosen over larger models because inference is fast enough to run synchronously",
             "different model choice rationale"),
            ("TypeScript strict mode is enabled globally and all any types must be justified with a comment",
             "different TS config"),
            ("The knowledge graph entity extraction phase runs every 30 minutes processing unlinked memories",
             "different pipeline timing"),
            ("Memory reconciliation merges overlapping facts from different agents into a single canonical record",
             "different pipeline stage"),
            ("Agent trust level controls whether memories auto-confirm or queue for human review before storage",
             "different trust model"),
            ("The MCP server supports API key authentication for remote agents and unauthenticated local connections",
             "different auth fact"),
            ("Memory confidence scores decay over time unless the fact is reinforced by additional observations",
             "different confidence model"),
            ("React Query cache TTL is set to five minutes for memory list queries to balance freshness and load",
             "different frontend cache fact"),
            ("The toast window is a non-activating panel that shows capture notifications without stealing focus",
             "different UI fact"),
            ("PII redaction strips credit card numbers SSNs API keys and email addresses before database storage",
             "different privacy fact"),
            ("Git hooks run cargo test on commit for fast checks and full coverage gate on push before merge",
             "different git hooks fact"),
        ];

        // ── Collect all similarity scores for distribution stats ──
        struct Sample {
            content: &'static str,
            label: &'static str, // "exact" | "paraphrase" | "different"
            sim: f64,
            admitted: bool,
        }

        let mut all_results: Vec<Sample> = Vec::new();

        // Evaluate duplicates
        for (content, kind, _seed) in duplicates {
            let (r, _similar) = g.evaluate(content, &db).await.unwrap();
            let sim = r.scores.novelty_score.unwrap_or(0.0);
            let label = if *kind == "near-exact" {
                "exact"
            } else {
                "paraphrase"
            };
            all_results.push(Sample {
                content,
                label,
                sim,
                admitted: r.admitted,
            });
        }

        // Evaluate different
        for (content, _desc) in different {
            let (r, _similar) = g.evaluate(content, &db).await.unwrap();
            let sim = r.scores.novelty_score.unwrap_or(0.0);
            all_results.push(Sample {
                content,
                label: "different",
                sim,
                admitted: r.admitted,
            });
        }

        // ── Compute distribution stats per category ──
        let stats = |label: &str| -> (f64, f64, f64) {
            let vals: Vec<f64> = all_results
                .iter()
                .filter(|s| s.label == label)
                .map(|s| s.sim)
                .collect();
            if vals.is_empty() {
                return (0.0, 0.0, 0.0);
            }
            let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            (min, max, mean)
        };

        let (exact_min, exact_max, exact_mean) = stats("exact");
        let (para_min, para_max, para_mean) = stats("paraphrase");
        let (diff_min, diff_max, diff_mean) = stats("different");

        let _n_exact = duplicates
            .iter()
            .filter(|(_, k, _)| *k == "near-exact")
            .count();

        // ── Results at target threshold ──
        let dup_rejected: usize = all_results
            .iter()
            .filter(|s| s.label != "different" && !s.admitted)
            .count();
        let diff_admitted: usize = all_results
            .iter()
            .filter(|s| s.label == "different" && s.admitted)
            .count();
        let false_positives: Vec<&Sample> = all_results
            .iter()
            .filter(|s| s.label == "different" && !s.admitted)
            .collect();

        // ── Threshold scan ──
        let thresholds = [0.60_f64, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.95];

        struct ThresholdResult {
            t: f64,
            dup_recall_pct: f64,
            fp: usize,
        }
        let mut scan: Vec<ThresholdResult> = Vec::new();
        for &t in &thresholds {
            let caught = all_results
                .iter()
                .filter(|s| s.label != "different" && s.sim > t)
                .count();
            let fp = all_results
                .iter()
                .filter(|s| s.label == "different" && s.sim > t)
                .count();
            let recall_pct = (caught as f64 / duplicates.len() as f64) * 100.0;
            scan.push(ThresholdResult {
                t,
                dup_recall_pct: recall_pct,
                fp,
            });
        }

        // ── Print summary table ──
        println!("\nNOVELTY GATE EVAL (threshold: {:.2})", threshold);
        println!(
            "Seeds: {} | Duplicates: {} | Different: {}",
            seeds.len(),
            duplicates.len(),
            different.len()
        );

        println!("\nSimilarity distribution:");
        println!(
            "  Exact dupes (expected >0.95):   min={:.2}  max={:.2}  mean={:.2}",
            exact_min, exact_max, exact_mean
        );
        println!(
            "  Paraphrases (expected 0.7-0.9): min={:.2}  max={:.2}  mean={:.2}",
            para_min, para_max, para_mean
        );
        println!(
            "  Different (expected <0.6):      min={:.2}  max={:.2}  mean={:.2}",
            diff_min, diff_max, diff_mean
        );

        println!("\nAt threshold {:.2}:", threshold);
        println!(
            "  Duplicate recall:  {}/{} ({:.0}%)",
            dup_rejected,
            duplicates.len(),
            (dup_rejected as f64 / duplicates.len() as f64) * 100.0
        );
        println!(
            "  Different admit:   {}/{} ({:.0}%)",
            diff_admitted,
            different.len(),
            (diff_admitted as f64 / different.len() as f64) * 100.0
        );
        println!("  False positives:   {}", false_positives.len());

        println!("\nOptimal threshold analysis:");
        for tr in &scan {
            println!(
                "  {:.2}: recall={:.0}% FP={}",
                tr.t, tr.dup_recall_pct, tr.fp
            );
        }

        if dup_rejected < duplicates.len() {
            let missed: Vec<&str> = all_results
                .iter()
                .filter(|s| s.label != "different" && s.admitted)
                .map(|s| s.content)
                .collect();
            println!(
                "\nWARN: {} duplicate(s) not caught at threshold {:.2}:",
                missed.len(),
                threshold
            );
            for m in &missed {
                println!("  - {}", m);
            }
        }

        if !false_positives.is_empty() {
            println!("\nFALSE POSITIVES (different content rejected):");
            for fp in &false_positives {
                println!("  - sim={:.3} {}", fp.sim, fp.content);
            }
        }

        // ── Edge zone: everything between 0.55 and 0.75 ──
        println!("\n═══ EDGE ZONE (sim 0.55 – 0.75) — the threshold decision area ═══");
        let mut edge_samples: Vec<&Sample> = all_results
            .iter()
            .filter(|s| s.sim >= 0.55 && s.sim <= 0.75)
            .collect();
        edge_samples.sort_by(|a, b| b.sim.partial_cmp(&a.sim).unwrap());

        if edge_samples.is_empty() {
            println!("  (no samples in this range)");
        } else {
            println!("  {:>4}  {:>10}  content", "sim", "type");
            println!("  ────  ──────────  ───────────────────────────────────");
            for s in &edge_samples {
                let truncated: String = s.content.chars().take(80).collect();
                println!(
                    "  {:.3} {:>10}  {}{}",
                    s.sim,
                    s.label,
                    truncated,
                    if s.content.len() > 80 { "..." } else { "" }
                );
            }
        }
        println!();

        // Soft assertion: warn but don't fail on missed paraphrases
        // Hard assertion: zero false positives on genuinely different content
        assert_eq!(
            false_positives.len(),
            0,
            "Genuinely different content was incorrectly rejected as duplicates: {:?}",
            false_positives
                .iter()
                .map(|s| s.content)
                .collect::<Vec<_>>()
        );
    }
}
