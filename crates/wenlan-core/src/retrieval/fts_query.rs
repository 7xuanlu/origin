// SPDX-License-Identifier: Apache-2.0
//! FTS5 query hardening helpers.
//!
//! Sanitizes user queries before passing them to the `memories_fts MATCH ?1`
//! clause, preventing FTS5 syntax errors from silently zeroing the FTS channel.
//!
//! All functions are pure (no I/O, no async, no DB). Flag gate lives here so
//! all call sites share the same env-var logic.

use std::collections::HashSet;
use std::sync::LazyLock;

/// FTS5 operator characters that trigger a syntax error when unquoted.
/// `+` and `-` are phrase operators; `*` is a prefix wildcard;
/// `(` `)` group subexpressions; `:` selects a column; `^` boosts a token;
/// `"` is a phrase delimiter; `{` `}` `[` `]` `~` `\` are reserved.
const FTS5_SPECIAL_CHARS: &[char] = &[
    '"', '*', '(', ')', ':', '^', '+', '-', '{', '}', '[', ']', '~', '\\',
];

/// Bare-keyword tokens that FTS5 interprets as operators.
const FTS5_KEYWORDS: &[&str] = &["AND", "OR", "NOT", "NEAR"];

/// ~40 common English stopwords used for relaxed-OR eligibility gating.
static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "a", "an", "the", "and", "or", "not", "in", "on", "at", "to", "of", "for", "with", "by",
        "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might", "shall", "it", "its",
        "this", "that", "as", "up", "if", "so", "but", "what", "which", "who", "how", "when",
        "where", "why", "near",
    ]
    .iter()
    .copied()
    .collect()
});

/// Returns the global stopword set for use in [`build_relaxed_or`].
pub fn stopwords() -> &'static HashSet<&'static str> {
    &STOPWORDS
}

/// True unless `WENLAN_ENABLE_FTS_HARDENING` is a falsy token (`0`, `false`,
/// `no`, `off`, case-insensitive). OPT-OUT, default ON.
///
/// When OFF, callers fall back to the raw, unsanitized query (pre-T12
/// behavior), which lets an FTS5 syntax error (e.g. a hyphenated token or a
/// bare date like `2026-12-15`) silently zero the FTS channel.
pub fn fts_recall_hardening_enabled() -> bool {
    match std::env::var("WENLAN_ENABLE_FTS_HARDENING") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
}

/// Sanitize a raw user query for safe use in an FTS5 `MATCH` expression.
///
/// Each whitespace-delimited token is examined. A token is "unsafe" when it
/// contains any FTS5-significant character (`"`, `*`, `(`, `)`, `:`, `^`,
/// `+`, `-`, `{`, `}`, `[`, `]`, `~`, `\\`) OR is a bare FTS5 keyword
/// (`AND`, `OR`, `NOT`, `NEAR`) case-sensitively. Unsafe tokens are wrapped
/// in double-quotes after escaping interior `"` as `""`, making FTS5 treat
/// the entire token as a literal phrase.
///
/// Safe tokens are emitted as-is. Tokens are joined with a single space,
/// preserving the implicit AND-matching default.
///
/// Iterates via `chars()` — never byte-slices — for UTF-8 safety per
/// .github/CONTRIBUTING.md "Code Conventions".
pub fn sanitize_fts_query(raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }

    raw.split_whitespace()
        .map(|token| {
            let needs_quoting = token.chars().any(|c| FTS5_SPECIAL_CHARS.contains(&c))
                || FTS5_KEYWORDS.contains(&token);
            if needs_quoting {
                // Escape interior double-quotes by doubling them, then wrap.
                let escaped: String = token
                    .chars()
                    .flat_map(|c| if c == '"' { vec!['"', '"'] } else { vec![c] })
                    .collect();
                format!("\"{}\"", escaped)
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Count whitespace-delimited tokens in `raw`.
pub fn query_token_count(raw: &str) -> usize {
    raw.split_whitespace().count()
}

/// Returns `true` when the query exceeds `max` tokens — i.e. the FTS branch
/// should be skipped entirely to avoid pathologically large MATCH expressions.
pub fn fts_length_exceeded(raw: &str, max: usize) -> bool {
    query_token_count(raw) > max
}

/// Build a relaxed OR query from `raw`, dropping stopwords and
/// gating on having at least two survivors before firing.
///
/// Steps:
/// 1. Split on whitespace.
/// 2. Drop tokens whose lowercase form appears in `stopwords`.
/// 3. If fewer than 2 survivors remain → `None` (OR is meaningless or empty).
/// 4. Sanitize each survivor via [`sanitize_fts_query`] (so `C++` becomes
///    `"C++"` in the OR list, not a bare operator).
/// 5. Join with ` OR ` → `Some(string)`.
pub fn build_relaxed_or(raw: &str, stopwords: &HashSet<&str>) -> Option<String> {
    let survivors: Vec<&str> = raw
        .split_whitespace()
        .filter(|t| !stopwords.contains(t.to_lowercase().as_str()))
        .collect();

    if survivors.len() < 2 {
        return None;
    }

    let parts: Vec<String> = survivors.into_iter().map(sanitize_fts_query).collect();

    Some(parts.join(" OR "))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── sanitize_fts_query ──────────────────────────────────────────────────

    #[test]
    fn sanitize_clean_query_unchanged() {
        assert_eq!(sanitize_fts_query("hello world"), "hello world");
    }

    #[test]
    fn sanitize_plus_token_quoted() {
        // "C++" contains a `+` character — must be quoted
        let out = sanitize_fts_query("C++ project");
        assert!(
            out.contains("\"C++\""),
            "C++ token should be double-quoted; got: {out}"
        );
    }

    #[test]
    fn sanitize_interior_quote_doubled() {
        // token `"big"` contains double-quotes: interior must be escaped as `""`
        let out = sanitize_fts_query(r#"what's the "big" plan"#);
        // The `"big"` token should appear as `"""big"""` in output
        // (outer wrap + doubled interior quotes)
        assert!(
            out.contains(r#""""big""""#),
            "Interior quotes must be doubled and token wrapped; got: {out}"
        );
    }

    #[test]
    fn sanitize_bare_booleans_quoted() {
        let out = sanitize_fts_query("a AND b OR c NOT d");
        // AND / OR / NOT must NOT appear as bare tokens (they must be quoted)
        let tokens: Vec<&str> = out.split_whitespace().collect();
        for tok in &["AND", "OR", "NOT"] {
            assert!(
                !tokens.contains(tok),
                "Bare keyword {tok} must not survive sanitization; got: {out}"
            );
        }
    }

    #[test]
    fn sanitize_star_paren_colon_caret() {
        for (input, bad_char) in [
            ("foo*", '*'),
            ("(bar)", '('),
            ("baz:qux", ':'),
            ("^lead", '^'),
        ] {
            let out = sanitize_fts_query(input);
            // The bad char should be inside a quoted phrase, not bare
            // Verify the token is quoted (starts and ends with `"`)
            // and bad_char still appears inside (not dropped)
            let token = out.split_whitespace().next().unwrap_or("");
            assert!(
                token.starts_with('"') && token.ends_with('"'),
                "Token with `{bad_char}` must be quoted; input={input:?} out={out:?}"
            );
            assert!(
                out.contains(bad_char),
                "Bad char must be preserved (not dropped); input={input:?} out={out:?}"
            );
        }
    }

    #[test]
    fn sanitize_empty_in_empty_out() {
        assert_eq!(sanitize_fts_query(""), "");
    }

    #[test]
    fn sanitize_injection_or_neutralized() {
        // An injection attempt using a bare double-quote followed by OR
        let out = sanitize_fts_query("\" OR 1=1 --");
        // The `"` token should be quoted/escaped, and OR must not be bare
        let tokens: Vec<&str> = out.split_whitespace().collect();
        assert!(
            !tokens.contains(&"OR"),
            "Bare OR must not survive injection attempt; got: {out}"
        );
    }

    #[test]
    fn sanitize_utf8_no_panic() {
        // Non-ASCII code points: must not byte-slice, must not panic
        let out = sanitize_fts_query("café ☕ naïve");
        // No special FTS chars in these tokens → returned as-is
        assert!(out.contains("café"), "café should be preserved; got: {out}");
        assert!(out.contains("☕"), "☕ should be preserved; got: {out}");
        assert!(
            out.contains("naïve"),
            "naïve should be preserved; got: {out}"
        );
    }

    // ── query_token_count ──────────────────────────────────────────────────

    #[test]
    fn token_count_three_words() {
        assert_eq!(query_token_count("one two three"), 3);
    }

    #[test]
    fn token_count_empty() {
        assert_eq!(query_token_count(""), 0);
    }

    // ── fts_length_exceeded ────────────────────────────────────────────────

    #[test]
    fn length_not_exceeded_small_query() {
        assert!(!fts_length_exceeded("a b c", 128));
    }

    #[test]
    fn length_exceeded_129_tokens() {
        let q: String = (0..129)
            .map(|i| format!("tok{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(fts_length_exceeded(&q, 128));
    }

    #[test]
    fn length_boundary_exactly_128_tokens() {
        let q: String = (0..128)
            .map(|i| format!("tok{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        // Exactly 128 tokens: NOT exceeded (strictly greater than)
        assert!(!fts_length_exceeded(&q, 128));
    }

    // ── build_relaxed_or ──────────────────────────────────────────────────

    #[test]
    fn relaxed_or_all_content_tokens_survive() {
        let sw = stopwords();
        let out = build_relaxed_or("vector embedding model", sw);
        assert_eq!(
            out.as_deref(),
            Some("vector OR embedding OR model"),
            "No stopwords -> all tokens survive"
        );
    }

    #[test]
    fn relaxed_or_single_survivor_returns_none() {
        // "what is the" are all stopwords; only "plan" survives -> <2 survivors -> None
        let sw = stopwords();
        let out = build_relaxed_or("what is the plan", sw);
        assert_eq!(out, None, "Only 1 survivor must return None");
    }

    #[test]
    fn relaxed_or_two_survivors_returns_some() {
        // "the" is a stopword; "database" and "schema" survive
        let sw = stopwords();
        let out = build_relaxed_or("the database schema", sw);
        assert_eq!(
            out.as_deref(),
            Some("database OR schema"),
            "2 survivors should return Some"
        );
    }

    #[test]
    fn relaxed_or_all_stopwords_returns_none() {
        let sw = stopwords();
        let out = build_relaxed_or("the of a an", sw);
        assert_eq!(out, None, "All stopwords must return None");
    }

    #[test]
    fn relaxed_or_single_token_returns_none() {
        let sw = stopwords();
        let out = build_relaxed_or("singleword", sw);
        assert_eq!(out, None, "Single token has no OR semantics");
    }

    #[test]
    fn relaxed_or_sanitizes_survivors() {
        // "foo*" contains a special char; it must be quoted in the OR output.
        // "AND" lowercases to "and" which IS a stopword, so it gets dropped.
        // Survivors: "foo*" and "bar" → "foo*" must be sanitized to "\"foo*\"".
        let sw = stopwords();
        let out = build_relaxed_or("foo* AND bar", sw);
        let s = out.expect("foo* and bar should yield two survivors");
        assert!(
            s.contains("\"foo*\""),
            "foo* survivor must be sanitized to \"foo*\"; got: {s}"
        );
        // "bar" has no special chars → emitted as-is
        assert!(
            s.contains("bar"),
            "bar survivor should appear in OR output; got: {s}"
        );
    }

    // ── fts_recall_hardening_enabled ────────────────────────────────────────

    #[test]
    fn hardening_enabled_by_default_and_for_truthy_values() {
        for val in [None, Some("1"), Some("true"), Some("yes"), Some("garbage")] {
            temp_env::with_var("WENLAN_ENABLE_FTS_HARDENING", val, || {
                assert!(
                    fts_recall_hardening_enabled(),
                    "Expected enabled for value={val:?}"
                );
            });
        }
    }

    #[test]
    fn hardening_disabled_for_falsy_values() {
        for val in ["0", "false", "no", "off", "FALSE", "OFF"] {
            temp_env::with_var("WENLAN_ENABLE_FTS_HARDENING", Some(val), || {
                assert!(
                    !fts_recall_hardening_enabled(),
                    "Expected disabled for value={val:?}"
                );
            });
        }
    }

    #[test]
    fn hardening_unset_sanitizes_hyphenated_query() {
        // Unset -> ON by default -> callers should sanitize a hyphenated token
        // instead of passing it raw to FTS5 (the soak-test regression).
        temp_env::with_var("WENLAN_ENABLE_FTS_HARDENING", None::<&str>, || {
            assert!(fts_recall_hardening_enabled());
            let out = sanitize_fts_query("2026-12-15");
            assert_eq!(
                out, "\"2026-12-15\"",
                "hyphenated date token must be quoted when hardening is on; got: {out}"
            );
        });
    }

    #[test]
    fn hardening_disabled_returns_raw_query() {
        // Explicit opt-out -> caller uses the raw, unsanitized query.
        temp_env::with_var("WENLAN_ENABLE_FTS_HARDENING", Some("0"), || {
            assert!(!fts_recall_hardening_enabled());
        });
    }
}
