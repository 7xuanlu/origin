// SPDX-License-Identifier: Apache-2.0
use std::collections::BTreeSet;

// =====================================================================
// Tag suggestion
// =====================================================================

/// Suggest tags for a document based on chunked content + title.
///
/// Returns an ordered list of candidate tag names extracted from:
/// 1. Content: top-N keywords by frequency (stop words removed, min count 2)
/// 2. Title: segments split on common separators (- | — · –)
///
/// Tags already assigned to the document (per `tag_store`) are excluded
/// from the result. Caller may enrich with additional sources (e.g. the
/// active app name for the document's time window).
///
/// This is the content+title portion of the original pre-split
/// `suggest_tags` Tauri command. Environment-dependent signals such as
/// the "what app was the user in at last_modified" hint live on the
/// caller side because activity data is tracked by the Tauri app.
pub fn suggest_tags_for_document(
    chunks: &[String],
    title: &str,
    existing_tags: &[String],
) -> Vec<String> {
    let mut suggestions: BTreeSet<String> = BTreeSet::new();

    // 1. Content keywords
    let full_text: String = chunks.join(" ");
    for kw in extract_keywords(&full_text).into_iter().take(5) {
        suggestions.insert(kw);
    }

    // 2. Title parts
    for part in extract_title_parts(title) {
        suggestions.insert(part);
    }

    // 3. Drop already-assigned tags
    for tag in existing_tags {
        suggestions.remove(tag);
    }

    suggestions.into_iter().collect()
}

/// Extract meaningful keywords from document content using word frequency.
///
/// Ports the pre-split `extract_keywords` helper: filters stop words,
/// short tokens (< 3 chars), and pure-numeric tokens; only keeps words
/// that appear at least twice; ranks by descending frequency.
fn extract_keywords(text: &str) -> Vec<String> {
    use std::collections::HashMap;

    const STOP_WORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can",
        "need", "dare", "ought", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below", "between", "out", "off",
        "over", "under", "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "and",
        "but", "or", "if", "while", "because", "until", "that", "which", "who", "whom", "this",
        "these", "those", "what", "just", "about", "also", "it", "its", "they", "them", "their",
        "we", "our", "you", "your", "he", "she", "him", "her", "his", "my", "me", "i", "up",
        "down", "new", "one", "two", "get", "got", "like", "make", "see", "use", "used", "using",
    ];

    let stop: std::collections::HashSet<&str> = STOP_WORDS.iter().copied().collect();

    let mut freq: HashMap<String, usize> = HashMap::new();
    for word in text.split(|c: char| !c.is_alphanumeric() && c != '-') {
        let w = word.trim().to_lowercase();
        if w.len() < 3 || stop.contains(w.as_str()) || w.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        *freq.entry(w).or_insert(0) += 1;
    }

    let mut ranked: Vec<(String, usize)> = freq.into_iter().filter(|(_, c)| *c >= 2).collect();
    ranked.sort_by_key(|item| std::cmp::Reverse(item.1));
    ranked.into_iter().map(|(w, _)| w).collect()
}

/// Extract meaningful parts from a document title.
///
/// Splits on common separators (-, |, —, ·, –) and returns cleaned
/// lowercase segments between 3 and 30 characters with no ellipsis.
fn extract_title_parts(title: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for segment in title.split(['-', '|', '—', '·', '–']) {
        let trimmed = segment.trim().to_lowercase();
        // Use char count, not byte length — CJK/emoji segments have
        // multi-byte chars, so `.len()` would under- or over-filter.
        let char_count = trimmed.chars().count();
        if (3..=30).contains(&char_count) && !trimmed.contains("...") {
            parts.push(trimmed);
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_keywords_ranks_by_frequency() {
        let text = "rust rust rust memory memory database libsql concurrent safe";
        let kws = extract_keywords(text);
        // rust appears 3 times, memory 2 times — both should appear, rust first
        assert!(kws.len() >= 2);
        assert_eq!(kws[0], "rust");
        assert!(kws.contains(&"memory".to_string()));
    }

    #[test]
    fn test_extract_keywords_drops_stop_words_short_and_numeric() {
        let text = "the the the fox fox jumped 42 42 a an in of on to";
        let kws = extract_keywords(text);
        // "the" is a stop word, "42" is numeric, "a/an/in/of/on/to" stop words
        // only "fox" survives (appears 2x)
        assert_eq!(kws, vec!["fox".to_string()]);
    }

    #[test]
    fn test_extract_keywords_requires_min_count_of_2() {
        // Single-occurrence words are dropped
        let text = "rust memory database libsql concurrent safe tokio axum";
        let kws = extract_keywords(text);
        assert!(kws.is_empty());
    }

    #[test]
    fn test_extract_title_parts_splits_on_separators() {
        let parts = extract_title_parts("main.rs - My Project | rust coding");
        assert!(parts.iter().any(|p| p == "my project"));
        assert!(parts.iter().any(|p| p == "rust coding"));
    }

    #[test]
    fn test_extract_title_parts_filters_length() {
        let parts =
            extract_title_parts("ab - fine - this title segment is way way way way too long");
        // "ab" is too short (< 3 chars), "fine" ok (4 chars),
        // too-long segment (> 30 chars) should drop
        assert!(!parts.iter().any(|p| p == "ab"));
        assert!(parts.iter().any(|p| p == "fine"));
        assert!(!parts.iter().any(|p| p.contains("way way way")));
    }

    #[test]
    fn test_extract_title_parts_drops_ellipsis() {
        let parts = extract_title_parts("some thing... - clean part");
        assert!(parts.iter().any(|p| p == "clean part"));
        assert!(!parts.iter().any(|p| p.contains("...")));
    }

    #[test]
    fn test_suggest_tags_combines_content_and_title() {
        let chunks = vec!["rust rust rust memory memory database libsql".to_string()];
        let title = "rust docs - memory layer";
        let existing: Vec<String> = vec![];
        let suggestions = suggest_tags_for_document(&chunks, title, &existing);
        // Should include keywords (rust, memory) and title parts (memory layer, rust docs)
        assert!(suggestions.contains(&"rust".to_string()));
        assert!(suggestions.contains(&"memory".to_string()));
        assert!(suggestions.contains(&"memory layer".to_string()));
    }

    #[test]
    fn test_suggest_tags_excludes_existing() {
        let chunks = vec!["rust rust rust memory memory".to_string()];
        let existing = vec!["rust".to_string()];
        let suggestions = suggest_tags_for_document(&chunks, "", &existing);
        // rust is already assigned, should be filtered out
        assert!(!suggestions.contains(&"rust".to_string()));
        assert!(suggestions.contains(&"memory".to_string()));
    }

    #[test]
    fn test_suggest_tags_empty_inputs() {
        let suggestions = suggest_tags_for_document(&[], "", &[]);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_extract_title_parts_accepts_cjk_five_chars() {
        // 5 CJK chars = 15 bytes. Well within the 3..=30 char range,
        // and verifies the char-count path keeps the segment. (Under
        // the old byte-length check 15 bytes also passed, so this
        // guards against regression in the other direction.)
        let title = "研究笔记汇 | notes";
        let parts = extract_title_parts(title);
        assert!(
            parts.iter().any(|p| p == "研究笔记汇"),
            "expected 5-char CJK segment to be kept, got {:?}",
            parts
        );
    }

    #[test]
    fn test_extract_title_parts_accepts_short_cjk_segment() {
        // 3 CJK chars = 9 bytes. The byte-length check `>= 3` would
        // accept it, but we want to ensure the char-count path also
        // accepts it (and that 2-char segments are rejected).
        let parts = extract_title_parts("研究所 - other");
        assert!(
            parts.iter().any(|p| p == "研究所"),
            "expected 3-char CJK segment to be kept, got {:?}",
            parts
        );
        // A 2-char CJK segment (6 bytes) would have passed the old
        // byte-length `>= 3` check, but must be rejected by char count.
        let parts2 = extract_title_parts("研究 - other");
        assert!(
            !parts2.iter().any(|p| p == "研究"),
            "expected 2-char CJK segment to be rejected, got {:?}",
            parts2
        );
    }

    #[test]
    fn test_extract_title_parts_rejects_long_cjk_segment() {
        // 31 CJK chars = 93 bytes. Under the old byte-length `<= 30`
        // check this would have been dropped even at 11 chars. Under
        // char-count it must still be dropped because 31 > 30.
        let long_segment: String = "研".repeat(31);
        let title = format!("{} | short", long_segment);
        let parts = extract_title_parts(&title);
        assert!(
            !parts.iter().any(|p| p == &long_segment),
            "expected 31-char CJK segment to be rejected, got {:?}",
            parts
        );
    }
}
