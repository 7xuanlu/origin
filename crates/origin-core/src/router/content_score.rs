// SPDX-License-Identifier: AGPL-3.0-only
use std::collections::HashSet;

use crate::tuning::ScoringConfig;

/// Score the content value of OCR text. Returns 0.0–1.0.
///
/// Higher scores mean more "worth remembering":
/// - Sentences and paragraphs (articles, emails, chat) score high
/// - Code blocks score high
/// - Nav items, short fragments, loading screens score low
#[allow(dead_code)]
pub fn score_content(text: &str, cfg: &ScoringConfig) -> f32 {
    let trimmed = text.trim();
    if trimmed.len() < cfg.min_text_length {
        return 0.0;
    }

    let mut score: f32 = 0.0;

    // Check for sentences (>5 words ending in sentence punctuation)
    let has_sentences = trimmed.lines().any(|line| {
        let words: Vec<&str> = line.split_whitespace().collect();
        words.len() > 5 && line.ends_with(['.', '!', '?'])
    });
    if has_sentences {
        score += 0.4;
    }

    // Check for code patterns
    let has_code = trimmed.lines().any(|line| {
        let l = line.trim_start();
        // Indented lines with code-like characters
        (line.len() > l.len() && (l.contains('{') || l.contains('}') || l.contains(';') || l.contains("()") || l.contains("=>") || l.contains("->")))
        // Or common code patterns
        || l.starts_with("fn ") || l.starts_with("def ") || l.starts_with("func ")
        || l.starts_with("class ") || l.starts_with("import ") || l.starts_with("const ")
        || l.starts_with("let ") || l.starts_with("var ") || l.starts_with("pub ")
        || l.contains("//") || (l.starts_with("# ") && !l.starts_with("## ")) || l.starts_with("```")
    });
    if has_code {
        score += 0.4;
    }

    // Unique word count — vocabulary richness
    let words: HashSet<&str> = trimmed
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 1)
        .collect();
    if words.len() > 20 {
        score += 0.2;
    }

    // Text density penalty — ratio of non-whitespace chars to total
    let total_chars = trimmed.len() as f32;
    let content_chars = trimmed.chars().filter(|c| !c.is_whitespace()).count() as f32;
    let density = if total_chars > 0.0 {
        content_chars / total_chars
    } else {
        0.0
    };
    score *= density.max(0.4); // floor at 0.4 so single-signal content (0.4) can pass threshold

    score.min(1.0)
}

/// Score with focus bonus applied.
#[allow(dead_code)]
pub fn score_with_bonus(text: &str, is_focus: bool, cfg: &ScoringConfig) -> f32 {
    let base = score_content(text, cfg);
    if is_focus {
        (base + cfg.focus_bonus).min(1.0)
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ScoringConfig {
        ScoringConfig::default()
    }

    #[test]
    fn empty_text_scores_zero() {
        let c = cfg();
        assert_eq!(score_content("", &c), 0.0);
        assert_eq!(score_content("Loading...", &c), 0.0); // < 80 chars
    }

    #[test]
    fn short_nav_items_score_low() {
        let c = cfg();
        let nav = "Home\nAbout\nContact\nBlog\nPricing\nDocs\nAPI\nStatus\nTerms\nPrivacy\nHelp\nSettings\nAccount\nLogout";
        assert!(score_content(nav, &c) < c.score_threshold);
    }

    #[test]
    fn article_text_scores_high() {
        let c = cfg();
        let article = "The quick brown fox jumps over the lazy dog. This is a complete sentence with enough words to qualify. \
            It contains multiple paragraphs of real content that a user would want to remember and search for later. \
            The content is dense, meaningful, and represents the kind of text that ambient capture should preserve.";
        let s = score_content(article, &c);
        assert!(
            s >= c.score_threshold,
            "article scored {s}, expected >= {}",
            c.score_threshold
        );
    }

    #[test]
    fn code_scores_high() {
        let c = cfg();
        let code = "fn main() {\n    let x = 42;\n    println!(\"Hello, world!\");\n    for i in 0..10 {\n        process(i);\n    }\n}\n\nimpl Foo {\n    pub fn bar(&self) -> Result<(), Error> {\n        Ok(())\n    }\n}";
        let s = score_content(code, &c);
        assert!(
            s >= c.score_threshold,
            "code scored {s}, expected >= {}",
            c.score_threshold
        );
    }

    #[test]
    fn focus_bonus_applied() {
        let c = cfg();
        let text = "Home\nAbout\nContact\nBlog\nPricing\nDocs\nAPI\nStatus\nTerms\nPrivacy\nHelp\nSettings\nAccount\nLogout\nDashboard\nProfile";
        let base = score_content(text, &c);
        let with_bonus = score_with_bonus(text, true, &c);
        assert!(with_bonus > base);
    }

    #[test]
    fn window_headers_do_not_trigger_code_detection() {
        let c = cfg();
        // Simulates bundle_to_raw_document format with ## headers
        let bundled = "## Safari — GitHub\nHome About Contact Blog Pricing Docs API Status Terms Privacy Help Settings Account Logout Dashboard Profile";
        let s = score_content(bundled, &c);
        // ## headers should NOT count as code — this is pure nav content
        assert!(
            s < c.score_threshold,
            "bundled nav scored {s}, expected < {}",
            c.score_threshold
        );
    }
}
