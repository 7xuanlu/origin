// SPDX-License-Identifier: AGPL-3.0-only
use crate::sensor::vision::WindowOcrResult;

/// Detected intent category for a context capture.
#[derive(Debug, Clone)]
pub enum Intent {
    Debug,
    Scheduling,
    Research,
    Communication,
    Unknown,
}

/// Result of intent classification with confidence score.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct IntentClassification {
    pub intent: Intent,
    pub confidence: f64,
}

/// Keyword sets for each intent category.
const DEBUG_KEYWORDS: &[&str] = &[
    "error",
    "stack trace",
    "panic",
    "exception",
    "breakpoint",
    "traceback",
    "undefined",
    "segfault",
    "fatal",
    "unhandled",
    "SIGABRT",
    "SIGSEGV",
    "NullPointerException",
    "TypeError",
    "SyntaxError",
    "RuntimeError",
    "failed",
    "crash",
    "bug",
    "debug",
];

const SCHEDULING_KEYWORDS: &[&str] = &[
    "meeting",
    "calendar",
    "deadline",
    "sprint",
    "standup",
    "schedule",
    "appointment",
    "agenda",
    "reminder",
    "due date",
    "milestone",
];

const RESEARCH_KEYWORDS: &[&str] = &[
    "documentation",
    "API",
    "tutorial",
    "stackoverflow",
    "how to",
    "guide",
    "reference",
    "docs",
    "MDN",
    "example",
    "specification",
    "RFC",
];

const COMMUNICATION_KEYWORDS: &[&str] = &[
    "email",
    "slack",
    "message",
    "reply",
    "thread",
    "chat",
    "inbox",
    "sent",
    "compose",
    "draft",
    "notification",
];

/// Classify the intent of OCR results using keyword scoring.
///
/// Score = keyword matches / total word count per category. Highest wins
/// if above a minimum threshold (5%). Falls back to `Intent::Unknown`.
pub fn classify(ocr_results: &[WindowOcrResult], min_threshold: f64) -> IntentClassification {
    let combined_text: String = ocr_results
        .iter()
        .map(|r| r.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let lower = combined_text.to_lowercase();
    let word_count = lower.split_whitespace().count();

    if word_count == 0 {
        return IntentClassification {
            intent: Intent::Unknown,
            confidence: 0.0,
        };
    }

    let debug_score = count_keyword_matches(&lower, DEBUG_KEYWORDS) as f64 / word_count as f64;
    let sched_score = count_keyword_matches(&lower, SCHEDULING_KEYWORDS) as f64 / word_count as f64;
    let research_score =
        count_keyword_matches(&lower, RESEARCH_KEYWORDS) as f64 / word_count as f64;
    let comms_score =
        count_keyword_matches(&lower, COMMUNICATION_KEYWORDS) as f64 / word_count as f64;

    let scores = [
        (Intent::Debug, debug_score),
        (Intent::Scheduling, sched_score),
        (Intent::Research, research_score),
        (Intent::Communication, comms_score),
    ];

    let (best_intent, best_score) = scores
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    if best_score >= min_threshold {
        IntentClassification {
            intent: best_intent,
            confidence: (best_score * 20.0).min(1.0), // Scale to 0-1 range
        }
    } else {
        IntentClassification {
            intent: Intent::Unknown,
            confidence: 0.0,
        }
    }
}

fn count_keyword_matches(text: &str, keywords: &[&str]) -> usize {
    keywords
        .iter()
        .filter(|kw| text.contains(&kw.to_lowercase()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ocr(text: &str) -> Vec<WindowOcrResult> {
        vec![WindowOcrResult {
            window_name: "Test".to_string(),
            app_name: "Test".to_string(),
            text: text.to_string(),
            observations: vec![],
            focused: true,
            confidence: 0.95,
        }]
    }

    #[test]
    fn test_debug_intent() {
        let result = classify(
            &make_ocr(
                "Unhandled exception: TypeError at line 42. Stack trace follows with fatal error.",
            ),
            0.05,
        );
        assert!(matches!(result.intent, Intent::Debug));
    }

    #[test]
    fn test_scheduling_intent() {
        let result = classify(
            &make_ocr(
                "Team standup meeting scheduled for tomorrow. Check the calendar for the agenda.",
            ),
            0.05,
        );
        assert!(matches!(result.intent, Intent::Scheduling));
    }

    #[test]
    fn test_unknown_intent() {
        let result = classify(
            &make_ocr("The quick brown fox jumps over the lazy dog."),
            0.05,
        );
        assert!(matches!(result.intent, Intent::Unknown));
    }

    #[test]
    fn test_empty_text() {
        let result = classify(&make_ocr(""), 0.05);
        assert!(matches!(result.intent, Intent::Unknown));
        assert_eq!(result.confidence, 0.0);
    }
}
