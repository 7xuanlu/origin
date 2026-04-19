// SPDX-License-Identifier: AGPL-3.0-only

/// Extract person names and topic keywords from window context.
/// Used by the ambient monitor to build search queries.
/// Result of context extraction.
#[derive(Debug, Clone, Default)]
pub struct ExtractedContext {
    /// Person names detected (e.g. "Alice", "Bob Chen").
    pub persons: Vec<String>,
    /// Topic keywords detected (e.g. "Q3 budget", "CRM selection").
    pub topics: Vec<String>,
}

impl ExtractedContext {
    pub fn is_empty(&self) -> bool {
        self.persons.is_empty() && self.topics.is_empty()
    }

    /// Build search queries from extracted context.
    /// Returns person queries first (higher priority), then topic queries.
    pub fn to_queries(&self) -> Vec<String> {
        let mut queries = Vec::new();
        for person in &self.persons {
            queries.push(person.clone());
        }
        for topic in &self.topics {
            queries.push(topic.clone());
        }
        queries
    }
}

/// Split a window title on common separators, returning the first segment.
/// Handles " | ", " - ", " — ", " · " as separators (Slack uses |, most apps use -).
pub fn first_title_segment(title: &str) -> &str {
    for sep in [" | ", " - ", " — ", " · "] {
        if let Some(pos) = title.find(sep) {
            return title[..pos].trim();
        }
    }
    title.trim()
}

/// Words that indicate a generic/uninformative title segment — skip these as topics.
const SKIP_TOPICS: &[&str] = &[
    "new tab",
    "untitled",
    "home",
    "inbox",
    "drafts",
    "sent",
    "trash",
    "settings",
    "preferences",
    "loading",
    "sign in",
    "log in",
    "error",
];

/// Extract person names and topics from a window title.
///
/// Heuristics (in priority order):
/// - Email app or email site in browser title → email context
/// - Chat app or chat site in browser title → chat context
/// - Calendar/meeting app → meeting context
/// - Any app: first title segment as topic fallback (catches docs, browser tabs, etc.)
pub fn extract_from_window_title(app_name: &str, title: &str) -> ExtractedContext {
    let mut ctx = ExtractedContext::default();
    let app = app_name.to_lowercase();
    let title_lower = title.to_lowercase();

    // Email: native app or browser tab showing a mail site
    if app.contains("mail")
        || app.contains("gmail")
        || app.contains("outlook")
        || (is_browser(&app)
            && (title_lower.contains("gmail")
                || title_lower.contains("outlook")
                || title_lower.ends_with("mail")))
    {
        extract_email_context(title, &mut ctx);
    }
    // Chat: native app or browser tab showing a chat site
    else if app.contains("slack")
        || app.contains("teams")
        || app.contains("discord")
        || (is_browser(&app)
            && (title_lower.contains("slack")
                || title_lower.contains("teams")
                || title_lower.contains("discord")))
    {
        extract_chat_context(title, &mut ctx);
    }
    // Meeting/calendar
    else if app.contains("calendar") || app.contains("zoom") || app.contains("meet") {
        extract_meeting_context(title, &mut ctx);
    }

    // Generic fallback: use first segment of title as a topic for any app.
    // This catches VS Code files, browser tabs, Google Docs, GitHub PRs, etc.
    // Filtered: must be 3–80 chars and not a known generic word.
    if ctx.is_empty() {
        let segment = first_title_segment(title);
        let seg_lower = segment.to_lowercase();
        if segment.len() >= 3 && segment.len() <= 80 && !SKIP_TOPICS.iter().any(|s| seg_lower == *s)
        {
            ctx.topics.push(segment.to_string());
        }
    }

    ctx
}

fn is_browser(app: &str) -> bool {
    app.contains("chrome")
        || app.contains("safari")
        || app.contains("firefox")
        || app.contains("edge")
        || app.contains("brave")
        || app.contains("arc")
}

/// Extract person names and topics from OCR text.
/// Looks for:
/// - Email headers: "To: Name", "From: Name"
/// - Chat @mentions: "@alice"
/// - Subject lines: "Subject: ..."
/// - Slack-style: "Name  10:30 AM" (name at start of line before time)
pub fn extract_from_ocr(text: &str) -> ExtractedContext {
    let mut ctx = ExtractedContext::default();

    for line in text.lines() {
        let trimmed = line.trim();

        // Email headers: "To: Alice Chen <alice@co.com>" or "From: Bob"
        if let Some(rest) = trimmed
            .strip_prefix("To:")
            .or_else(|| trimmed.strip_prefix("From:"))
            .or_else(|| trimmed.strip_prefix("Cc:"))
        {
            let name = rest.trim().split('<').next().unwrap_or("").trim();
            // Drop if it's just an email address (no display name)
            if !name.is_empty() && name.len() < 60 && !name.contains('@') {
                ctx.persons.push(name.to_string());
            }
        }

        // Subject line → topic
        if let Some(rest) = trimmed.strip_prefix("Subject:") {
            let subject = rest
                .trim()
                .trim_start_matches("Re: ")
                .trim_start_matches("Fwd: ");
            if !subject.is_empty() && subject.len() < 80 {
                ctx.topics.push(subject.to_string());
            }
        }

        // "@alice" mentions in Slack/Discord chat
        for word in trimmed.split_whitespace() {
            if let Some(handle) = word.strip_prefix('@') {
                let clean = handle.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                if !clean.is_empty() && clean.len() < 30 {
                    ctx.persons.push(clean.to_string());
                }
            }
        }

        // Slack message author: line that's just a name (1-3 title-case words, no punctuation),
        // optionally followed by a timestamp. E.g. "Alice Chen  10:30 AM"
        {
            // Strip trailing timestamp-like suffix (digits, colon, AM/PM, spaces)
            let without_time = trimmed
                .trim_end_matches(|c: char| c.is_ascii_digit() || c == ':' || c == ' ')
                .trim_end_matches("AM")
                .trim_end_matches("PM")
                .trim();
            let words: Vec<&str> = without_time.split_whitespace().collect();
            if !words.is_empty()
                && words.len() <= 3
                && words.iter().all(|w| {
                    w.len() >= 2
                        && w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                        && w.chars()
                            .all(|c| c.is_alphabetic() || c == '-' || c == '\'')
                })
            {
                let name = words.join(" ");
                if name.len() < 40 {
                    ctx.persons.push(name);
                }
            }
        }
    }

    ctx.persons.dedup();
    ctx.topics.dedup();
    ctx
}

fn extract_email_context(title: &str, ctx: &mut ExtractedContext) {
    // Pattern: "Re: Subject - name@domain.com - Gmail"
    // or "Subject - name@domain.com - Gmail"
    let parts: Vec<&str> = title.splitn(3, " - ").collect();
    if parts.len() >= 2 {
        // First part is subject
        let subject = parts[0]
            .trim_start_matches("Re: ")
            .trim_start_matches("Fwd: ")
            .trim();
        if !subject.is_empty() {
            ctx.topics.push(subject.to_string());
        }
        // Second part may be email address
        let maybe_email = parts[1].trim();
        if maybe_email.contains('@') {
            // Extract name from email: "alice.chen@company.com" -> "Alice Chen"
            if let Some(local) = maybe_email.split('@').next() {
                let name = local
                    .replace(['.', '_'], " ")
                    .split_whitespace()
                    .map(|w| {
                        let mut c = w.chars();
                        match c.next() {
                            None => String::new(),
                            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                if !name.is_empty() {
                    ctx.persons.push(name);
                }
            }
        } else {
            // It's a name directly
            ctx.persons.push(maybe_email.to_string());
        }
    }
}

fn extract_chat_context(title: &str, ctx: &mut ExtractedContext) {
    // Slack native: "General | workspace-name | Slack" or "Alice Chen | Direct Message | Slack"
    // Slack web/older: "#channel-name - Slack" or "Person Name - Slack"
    // Teams, Discord: similar patterns with " - " or " | "
    let name = first_title_segment(title);
    if name.starts_with('#') {
        // Channel -> topic
        let channel = name.trim_start_matches('#').replace('-', " ");
        if !channel.is_empty() {
            ctx.topics.push(channel);
        }
    } else if !name.is_empty() && name.len() < 60 {
        // Could be a DM (person name) or a channel without # prefix
        // Heuristic: if it's 1-3 words with title-case, treat as person
        let word_count = name.split_whitespace().count();
        let title_case = name
            .split_whitespace()
            .all(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false));
        if word_count <= 3 && title_case {
            ctx.persons.push(name.to_string());
        } else {
            ctx.topics.push(name.to_string());
        }
    }
}

fn extract_meeting_context(title: &str, ctx: &mut ExtractedContext) {
    // Pattern: "Meeting with Alice Chen" or "1:1 with Bob"
    let lower = title.to_lowercase();
    for prefix in ["meeting with ", "1:1 with ", "call with ", "sync with "] {
        if let Some(pos) = lower.find(prefix) {
            let start = pos + prefix.len();
            // Use get() for safe UTF-8 boundary check instead of byte indexing
            if let Some(rest) = title.get(start..) {
                let name = rest.split(&['-', '|', '('][..]).next().unwrap_or("").trim();
                if !name.is_empty() && name.len() < 60 {
                    ctx.persons.push(name.to_string());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn email_with_address() {
        let ctx =
            extract_from_window_title("Gmail", "Re: Q3 Budget - alice.chen@company.com - Gmail");
        assert_eq!(ctx.topics, vec!["Q3 Budget"]);
        assert_eq!(ctx.persons, vec!["Alice Chen"]);
    }

    #[test]
    fn email_with_name() {
        let ctx = extract_from_window_title("Mail", "Project Update - Bob Smith - Mail");
        assert_eq!(ctx.topics, vec!["Project Update"]);
        assert_eq!(ctx.persons, vec!["Bob Smith"]);
    }

    #[test]
    fn slack_channel() {
        let ctx = extract_from_window_title("Slack", "#project-alpha - Slack");
        assert!(ctx.persons.is_empty());
        assert_eq!(ctx.topics, vec!["project alpha"]);
    }

    #[test]
    fn slack_dm() {
        let ctx = extract_from_window_title("Slack", "Alice Chen - Slack");
        assert_eq!(ctx.persons, vec!["Alice Chen"]);
        assert!(ctx.topics.is_empty());
    }

    #[test]
    fn zoom_meeting() {
        let ctx = extract_from_window_title("zoom.us", "Meeting with Bob Johnson");
        assert_eq!(ctx.persons, vec!["Bob Johnson"]);
    }

    #[test]
    fn calendar_event() {
        let ctx = extract_from_window_title("Calendar", "1:1 with Carol");
        assert_eq!(ctx.persons, vec!["Carol"]);
    }

    #[test]
    fn ocr_to_from() {
        let ctx = extract_from_ocr("To: Alice Chen <alice@co.com>\nFrom: Bob\nSubject: Q3");
        assert!(ctx.persons.contains(&"Alice Chen".to_string()));
        assert!(ctx.persons.contains(&"Bob".to_string()));
    }

    #[test]
    fn ocr_at_mentions() {
        let ctx = extract_from_ocr("Hey @alice_chen can you review this? cc @bob");
        assert!(ctx.persons.contains(&"alice_chen".to_string()));
        assert!(ctx.persons.contains(&"bob".to_string()));
    }

    // Generic fallback: VS Code extracts the filename as a topic
    #[test]
    fn vscode_extracts_filename_as_topic() {
        let ctx = extract_from_window_title("VS Code", "main.rs - origin - Visual Studio Code");
        assert_eq!(ctx.topics, vec!["main.rs"]);
        assert!(ctx.persons.is_empty());
    }

    // Generic fallback: skip short/generic title segments
    #[test]
    fn generic_skip_short_title() {
        let ctx = extract_from_window_title("Finder", "Go");
        assert!(ctx.is_empty());
    }

    // Slack with pipe separator (native Slack app format)
    #[test]
    fn slack_pipe_channel() {
        let ctx = extract_from_window_title("Slack", "#general | workspace | Slack");
        assert!(ctx.persons.is_empty());
        assert_eq!(ctx.topics, vec!["general"]);
    }

    #[test]
    fn slack_pipe_dm() {
        let ctx = extract_from_window_title("Slack", "Alice Chen | Direct Message | Slack");
        assert_eq!(ctx.persons, vec!["Alice Chen"]);
        assert!(ctx.topics.is_empty());
    }

    // Browser showing Gmail tab
    #[test]
    fn chrome_gmail_tab() {
        let ctx =
            extract_from_window_title("Google Chrome", "Re: Q3 Budget - alice@company.com - Gmail");
        assert_eq!(ctx.topics, vec!["Q3 Budget"]);
        assert_eq!(ctx.persons, vec!["Alice"]);
    }

    // Browser showing Slack tab
    #[test]
    fn chrome_slack_tab() {
        let ctx = extract_from_window_title("Google Chrome", "Alice Chen | Direct Message | Slack");
        assert_eq!(ctx.persons, vec!["Alice Chen"]);
    }

    // OCR: Subject line extraction
    #[test]
    fn ocr_subject_line() {
        let ctx = extract_from_ocr("From: Alice\nSubject: Q3 Budget Review\nBody text here");
        assert!(ctx.topics.contains(&"Q3 Budget Review".to_string()));
    }

    // OCR: Slack-style name at start of message line
    #[test]
    fn ocr_slack_author_name() {
        let ctx = extract_from_ocr("Alice Chen\nHey can you check the PR?\nBob\nSure!");
        assert!(ctx.persons.contains(&"Alice Chen".to_string()));
        assert!(ctx.persons.contains(&"Bob".to_string()));
    }

    #[test]
    fn empty_title() {
        let ctx = extract_from_window_title("Gmail", "");
        assert!(ctx.is_empty());
    }

    #[test]
    fn queries_persons_first() {
        let mut ctx = ExtractedContext::default();
        ctx.topics.push("Budget".into());
        ctx.persons.push("Alice".into());
        let queries = ctx.to_queries();
        assert_eq!(queries, vec!["Alice", "Budget"]);
    }
}
