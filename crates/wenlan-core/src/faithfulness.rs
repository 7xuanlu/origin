// SPDX-License-Identifier: Apache-2.0
//! Page faithfulness scoring logic (shared between eval bench and prod verifier).

const STOPWORDS: &[&str] = &[
    "with", "from", "that", "this", "these", "those", "have", "been", "will", "would", "could",
    "should", "their", "there", "where", "when", "what", "which", "while", "about", "after",
    "before", "between", "into", "over", "under", "very", "more", "most", "some", "such", "than",
    "then", "they", "them", "your", "yours",
];

/// Byte spans of `body`'s sentences, in order. Splits on terminal punctuation
/// followed by whitespace; the final span may have no trailing whitespace.
///
/// This is the ONE definition of the sentence boundary — a `drift_guard` tooth
/// fails the build on a second copy, because the boundary decides where one
/// claim ends and the next begins, and M5 claim identity is content-addressed
/// over that text.
///
/// Empty spans are **retained**. A caller that maps recorded byte offsets onto
/// span indices (`citations::process_citation_output`, attributing citation
/// markers to sentences) needs indices aligned with the raw scan.
/// `split_sentences` drops them at its own layer instead.
pub fn sentence_spans(body: &str) -> Vec<(usize, usize)> {
    let re = regex::Regex::new(r"(?m)[.!?]+\s+").expect("static regex");
    let mut spans = Vec::new();
    let mut prev = 0;
    for m in re.find_iter(body) {
        spans.push((prev, m.start()));
        prev = m.end();
    }
    spans.push((prev, body.len()));
    spans
}

/// Split a page body into sentences, dropping blank ones. Boundaries come from
/// [`sentence_spans`]; this is the offset-free view for scoring callers.
pub fn split_sentences(body: &str) -> Vec<&str> {
    sentence_spans(body)
        .into_iter()
        .map(|(start, end)| &body[start..end])
        .filter(|s| !s.trim().is_empty())
        .collect()
}

/// Extract content-bearing tokens from a sentence: lowercase, length >= 4,
/// excluding stopwords. Used for faithfulness overlap scoring.
pub fn content_tokens(sentence: &str) -> Vec<String> {
    sentence
        .split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_ascii_lowercase())
        .filter(|t| t.len() >= 4 && !STOPWORDS.contains(&t.as_str()))
        .collect()
}

/// Fraction (0..=1) of the sentence's content tokens appearing as whole-word
/// matches in the source text. Zero content tokens => 1.0 (vacuously faithful).
pub fn overlap_fraction(sentence: &str, source: &str) -> f64 {
    let toks = content_tokens(sentence);
    if toks.is_empty() {
        return 1.0;
    }
    let lo_source = source.to_ascii_lowercase();
    let mut hits = 0usize;
    for t in &toks {
        let pattern = format!(r"\b{}\b", regex::escape(t));
        let found = regex::Regex::new(&pattern)
            .map(|re| re.is_match(&lo_source))
            .unwrap_or_else(|_| lo_source.contains(t.as_str()));
        if found {
            hits += 1;
        }
    }
    hits as f64 / toks.len() as f64
}

/// True if at least 50% of the sentence's content tokens appear in the source.
pub fn score_sentence_faithful(sentence: &str, source: &str) -> bool {
    overlap_fraction(sentence, source) >= 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sentences_basic_punctuation() {
        let s = split_sentences("First sentence. Second sentence! Third question? Final.");
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn sentence_spans_cover_the_body_and_match_split_sentences() {
        let body = "First sentence. Second sentence! Third question? Final.";
        let spans = sentence_spans(body);
        assert_eq!(spans.first().expect("nonempty").0, 0);
        assert_eq!(spans.last().expect("nonempty").1, body.len());
        let via_spans: Vec<&str> = spans
            .iter()
            .map(|&(s, e)| &body[s..e])
            .filter(|s| !s.trim().is_empty())
            .collect();
        assert_eq!(via_spans, split_sentences(body));
    }

    #[test]
    fn sentence_spans_retains_the_empty_spans_split_sentences_drops() {
        // Two adjacent delimiter matches leave an empty span between them.
        // `citations::process_citation_output` attributes each citation marker
        // by resolving its byte offset to a span INDEX, so dropping the empty
        // one here would shift every later sentence's attribution by one.
        let body = "A. . B";
        let spans = sentence_spans(body);
        assert_eq!(spans, vec![(0, 1), (3, 3), (5, 6)]);
        assert_eq!(split_sentences(body), vec!["A", "B"]);
    }

    #[test]
    fn content_tokens_strips_stopwords_and_short() {
        let toks = content_tokens("This is a Rust programming language with memory safety.");
        assert!(toks.contains(&"rust".to_string()));
        assert!(!toks.contains(&"this".to_string()));
    }

    #[test]
    fn overlap_fraction_exact_and_boundary() {
        assert_eq!(overlap_fraction("word", ""), 0.0);
        assert_eq!(overlap_fraction(".", "anything"), 1.0); // vacuous
                                                            // 2 of 5 content tokens present => 0.4 => below the 0.5 floor
        let sent = "Rust provides memory safety guarantees.";
        let src = "rust ... memory ..."; // hits: rust, memory; misses: provides, safety, guarantees
        let f = overlap_fraction(sent, src);
        assert!(f > 0.0 && f < 0.5);
        assert!(!score_sentence_faithful(sent, src));
    }

    #[test]
    fn score_sentence_faithful_majority_overlap() {
        let sentence = "Rust provides memory safety guarantees.";
        let all = "Rust provides memory safety guarantees";
        assert!(score_sentence_faithful(sentence, all));
        assert!(!score_sentence_faithful(sentence, "Rust is great"));
    }
}
