// SPDX-License-Identifier: AGPL-3.0-only
use crate::router::keywords::IntentClassification;
use crate::sensor::vision::{TextObservation, WindowOcrResult};
use crate::trigger::types::TriggerEvent;
use chrono::{DateTime, Utc};

/// A window snapshot within a context bundle.
#[derive(Debug, Clone)]
pub struct WindowSnapshot {
    pub app_name: String,
    pub window_title: String,
    pub text: String,
    #[allow(dead_code)]
    pub observations: Vec<TextObservation>,
    pub focused: bool,
    pub url: Option<String>,
}

/// A complete context capture bundle, ready for consumption.
#[derive(Debug, Clone)]
pub struct ContextBundle {
    pub trigger_type: String,
    pub timestamp: DateTime<Utc>,
    #[allow(dead_code)]
    pub intent: Option<IntentClassification>,
    pub windows: Vec<WindowSnapshot>,
    pub raw_text: Option<String>,
}

impl ContextBundle {
    /// Create a bundle from a QuickThought text (bypasses vision).
    pub fn from_text(text: String) -> Self {
        Self {
            trigger_type: "thought".to_string(),
            timestamp: Utc::now(),
            intent: None,
            windows: vec![],
            raw_text: Some(text),
        }
    }
}

/// Assemble a ContextBundle from OCR results (ambient — no intent).
#[allow(dead_code)]
pub fn assemble_bundle(ocr: Vec<WindowOcrResult>, event: &TriggerEvent) -> ContextBundle {
    let trigger_type = trigger_type_str(event);
    let windows = ocr_to_snapshots(ocr);

    ContextBundle {
        trigger_type,
        timestamp: Utc::now(),
        intent: None,
        windows,
        raw_text: None,
    }
}

/// Assemble a ContextBundle with intent classification (focus/hotkey/snip).
pub fn assemble_bundle_with_intent(
    ocr: Vec<WindowOcrResult>,
    event: &TriggerEvent,
    intent: IntentClassification,
) -> ContextBundle {
    let trigger_type = trigger_type_str(event);
    let windows = ocr_to_snapshots(ocr);

    ContextBundle {
        trigger_type,
        timestamp: Utc::now(),
        intent: Some(intent),
        windows,
        raw_text: None,
    }
}

/// Convert OCR results to window snapshots, carrying observations through.
fn ocr_to_snapshots(ocr: Vec<WindowOcrResult>) -> Vec<WindowSnapshot> {
    ocr.into_iter()
        .map(|r| WindowSnapshot {
            app_name: r.app_name,
            window_title: r.window_name,
            text: cleanup_ocr_text(&r.text),
            observations: r.observations,
            focused: r.focused,
            url: None,
        })
        .collect()
}

/// OCR text cleanup: dedup, remove UI chrome noise, collapse nav runs.
fn cleanup_ocr_text(text: &str) -> String {
    // Pass 1: basic line filtering
    let mut filtered: Vec<&str> = Vec::new();
    let mut prev = "";
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Strip leading bullets/icons that OCR picks up from UI
        let stripped = trimmed
            .trim_start_matches(|c: char| "•■›>▸▹◦–—·★☆♥♦▪▫⊕⊖✓✗✔✘←→↑↓".contains(c))
            .trim();
        // Skip noise: < 3 chars (unless all digits)
        if stripped.chars().count() < 3 && !stripped.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        // Skip exact duplicate of previous line
        if stripped == prev {
            continue;
        }
        prev = stripped;
        filtered.push(stripped);
    }

    // Pass 2: collapse runs of short nav-like lines.
    // A "nav line" is short, has no sentence punctuation, and doesn't look like code.
    // 4+ consecutive nav lines = sidebar/bookmark bar noise → collapse to single summary.
    let mut result: Vec<String> = Vec::new();
    let mut nav_run: Vec<&str> = Vec::new();

    for line in &filtered {
        let char_count = line.chars().count();
        let is_nav = char_count <= 25
            && !line.ends_with('.')
            && !line.ends_with('?')
            && !line.ends_with('!')
            && !line.ends_with(',')
            && !line.ends_with(';')
            && !line.ends_with('{')
            && !line.ends_with('}')
            && !line.ends_with(')')
            && !line.contains(". ")
            && !line.contains("::")
            && !line.contains("fn ")
            && !line.contains("pub ")
            && !line.contains("let ")
            && !line.contains("//");

        if is_nav {
            nav_run.push(line);
        } else {
            flush_nav_run(&mut result, &mut nav_run);
            result.push(line.to_string());
        }
    }
    flush_nav_run(&mut result, &mut nav_run);

    result.join("\n")
}

/// Flush a run of navigation-like lines: keep up to 3, summarize if more.
fn flush_nav_run(result: &mut Vec<String>, nav_run: &mut Vec<&str>) {
    if nav_run.is_empty() {
        return;
    }
    if nav_run.len() <= 3 {
        // Short run — keep as-is, likely meaningful
        for line in nav_run.iter() {
            result.push(line.to_string());
        }
    } else {
        // Long run — sidebar/bookmark noise, collapse
        let preview: Vec<&str> = nav_run.iter().take(3).copied().collect();
        result.push(format!(
            "[{} nav items: {}, ...]",
            nav_run.len(),
            preview.join(", ")
        ));
    }
    nav_run.clear();
}

/// Convert a TriggerEvent to its string type label.
fn trigger_type_str(event: &TriggerEvent) -> String {
    match event {
        TriggerEvent::ManualHotkey => "hotkey".to_string(),
        TriggerEvent::DragSnip { .. } => "snip".to_string(),
        TriggerEvent::QuickThought { .. } => "thought".to_string(),
        TriggerEvent::FocusChange { .. } => "focus_change".to_string(),
        TriggerEvent::TextSelected { .. } => "text_selection".to_string(),
        TriggerEvent::TextIcon { .. } => "text_icon".to_string(),
        TriggerEvent::HideIcon => "hide_icon".to_string(),
    }
}

// bundle_to_raw_document was removed — its logic was inlined into
// run_context_consumer in router/intent.rs during the thin-client conversion
// (commit 42f74160), and the function is no longer called. Kept a note here
// so future readers don't go hunting for it.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cleanup_ocr_dedup_lines() {
        let input = "Hello World\nHello World\nSomething else";
        assert_eq!(cleanup_ocr_text(input), "Hello World\nSomething else");
    }

    #[test]
    fn test_cleanup_ocr_removes_noise() {
        let input = "Good line\n..\n!!\nAnother good line";
        assert_eq!(cleanup_ocr_text(input), "Good line\nAnother good line");
    }

    #[test]
    fn test_cleanup_ocr_keeps_digits() {
        let input = "Line 1\n42\nLine 2";
        assert_eq!(cleanup_ocr_text(input), "Line 1\n42\nLine 2");
    }

    #[test]
    fn test_cleanup_ocr_strips_empty_and_whitespace() {
        let input = "First\n\n  \n\nSecond";
        assert_eq!(cleanup_ocr_text(input), "First\nSecond");
    }

    #[test]
    fn test_cleanup_ocr_strips_bullet_prefixes() {
        // Mix of bulleted and regular lines — bullets stripped, items kept
        let input = "• First item in a list of things to do.\n■ Second item that is a full sentence.\nNormal line that is long enough.";
        let result = cleanup_ocr_text(input);
        assert!(result.contains("First item"));
        assert!(result.contains("Normal line"));
        // Bullets should be stripped
        assert!(!result.contains("•"));
        assert!(!result.contains("■"));
    }

    #[test]
    fn test_cleanup_ocr_collapses_nav_runs() {
        // Simulates a bookmark bar / sidebar with many short items
        let input = "Jobs\nNews\nCareer\nDev\nCreate\nSeedwave\nGemini\nHome\nIntv\nMeta\nActual sentence content here.";
        let result = cleanup_ocr_text(input);
        // Should collapse the 10 nav items, not keep all 10 as separate lines
        assert!(
            result.contains("nav items"),
            "Expected nav collapse, got: {}",
            result
        );
        assert!(result.contains("Actual sentence content here."));
    }

    #[test]
    fn test_cleanup_ocr_keeps_short_runs() {
        // 3 or fewer short lines should be kept as-is
        let input = "Settings\nHelp\nThis is a full sentence.";
        let result = cleanup_ocr_text(input);
        assert!(result.contains("Settings"));
        assert!(result.contains("Help"));
        assert!(!result.contains("nav items"));
    }

    #[test]
    fn test_cleanup_ocr_real_mail_sidebar() {
        let input = "Favorites\nAll Inboxes\nVIPs\nFlagged\nAll Drafts\nAll Sent\nSmart Mailboxes\nToday\nExchange\nDrafts\nSent\nJunk\nTrash\nArchive\nActual email content follows here.";
        let result = cleanup_ocr_text(input);
        // 14 sidebar items should be collapsed
        assert!(
            result.contains("nav items"),
            "Expected nav collapse for mail sidebar, got: {}",
            result
        );
        assert!(result.contains("Actual email content follows here."));
    }
}
