// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::OriginError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

/// Default categories shipped with the app.
const DEFAULT_CATEGORIES: &[&str] = &[
    "code",
    "communication",
    "research",
    "writing",
    "design",
    "other",
];

/// Persistent tag storage, loaded from / saved to tags.json.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TagStore {
    /// The user's tag library (all known tag names), sorted.
    pub tags: BTreeSet<String>,
    /// Per-document tag assignments, keyed by "{source}::{source_id}".
    pub document_tags: HashMap<String, BTreeSet<String>>,
    /// Ordered list of category names.
    #[serde(default)]
    pub categories: Vec<String>,
    /// Per-document category assignment, keyed by "{source}::{source_id}".
    #[serde(default)]
    pub document_categories: HashMap<String, String>,
}

impl TagStore {
    pub fn doc_key(source: &str, source_id: &str) -> String {
        format!("{}::{}", source, source_id)
    }

    /// Ensure default categories exist. Called after loading from disk.
    pub fn ensure_default_categories(&mut self) {
        if self.categories.is_empty() {
            self.categories = DEFAULT_CATEGORIES.iter().map(|s| s.to_string()).collect();
        }
    }

    /// Set the tags for a document. Adds any new tags to the library.
    /// Returns the final set of tags for the document.
    pub fn set_document_tags(
        &mut self,
        source: &str,
        source_id: &str,
        tags: Vec<String>,
    ) -> Vec<String> {
        let key = Self::doc_key(source, source_id);
        let tag_set: BTreeSet<String> = tags
            .into_iter()
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();

        // Add all tags to the library
        for tag in &tag_set {
            self.tags.insert(tag.clone());
        }

        if tag_set.is_empty() {
            self.document_tags.remove(&key);
        } else {
            self.document_tags.insert(key, tag_set.clone());
        }

        tag_set.into_iter().collect()
    }

    /// Get tags for a specific document.
    pub fn get_document_tags(&self, source: &str, source_id: &str) -> Vec<String> {
        let key = Self::doc_key(source, source_id);
        self.document_tags
            .get(&key)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Delete a tag from the library and from all document assignments.
    pub fn delete_tag(&mut self, name: &str) {
        let normalized = name.trim().to_lowercase();
        self.tags.remove(&normalized);
        for tags in self.document_tags.values_mut() {
            tags.remove(&normalized);
        }
        // Remove entries that are now empty
        self.document_tags.retain(|_, v| !v.is_empty());
    }

    /// Remove all tag assignments for a document (called on document deletion).
    pub fn remove_document(&mut self, source: &str, source_id: &str) {
        let key = Self::doc_key(source, source_id);
        self.document_tags.remove(&key);
        self.document_categories.remove(&key);
    }

    /// Get all tags in the library as a sorted vector.
    pub fn list_all_tags(&self) -> Vec<String> {
        self.tags.iter().cloned().collect()
    }

    /// Get all document tags as a HashMap (for bulk frontend fetch).
    pub fn all_document_tags(&self) -> HashMap<String, Vec<String>> {
        self.document_tags
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect()
    }

    // ── Category methods ────────────────────────────────────────────

    /// Assign a category to a document.
    pub fn set_document_category(&mut self, source: &str, source_id: &str, category: &str) {
        let key = Self::doc_key(source, source_id);
        let normalized = category.trim().to_lowercase();
        // Only set if the category exists in the user's list
        if self.categories.iter().any(|c| c == &normalized) {
            self.document_categories.insert(key, normalized);
        }
    }

    /// Get the category for a specific document.
    #[allow(dead_code)]
    pub fn get_document_category(&self, source: &str, source_id: &str) -> Option<String> {
        let key = Self::doc_key(source, source_id);
        self.document_categories.get(&key).cloned()
    }

    /// Get all document categories as a HashMap (for bulk frontend fetch).
    pub fn all_document_categories(&self) -> HashMap<String, String> {
        self.document_categories.clone()
    }

    /// Add a new category to the list.
    pub fn add_category(&mut self, name: &str) {
        let normalized = name.trim().to_lowercase();
        if !normalized.is_empty() && !self.categories.contains(&normalized) {
            self.categories.push(normalized);
        }
    }

    /// Remove a category. Reassigns affected documents to "Other".
    pub fn remove_category(&mut self, name: &str) {
        let normalized = name.trim().to_lowercase();
        // Don't allow removing "other" — it's the fallback
        if normalized == "other" {
            return;
        }
        self.categories.retain(|c| c != &normalized);
        // Reassign affected docs to "other"
        for cat in self.document_categories.values_mut() {
            if *cat == normalized {
                *cat = "other".to_string();
            }
        }
    }
}

// ── Classification ──────────────────────────────────────────────────

/// Rule-based document classification. Returns the best-matching category name.
pub fn classify_document(
    source: &str,
    title: &str,
    content: &str,
    app_name: Option<&str>,
    categories: &[String],
) -> String {
    let result = classify_inner(source, title, content, app_name);
    // Only return if the category exists in the user's list
    if categories.iter().any(|c| c == &result) {
        result
    } else if categories.iter().any(|c| c == "other") {
        "other".to_string()
    } else {
        // User deleted "other" too — use first available
        categories
            .first()
            .cloned()
            .unwrap_or_else(|| "other".to_string())
    }
}

fn classify_inner(source: &str, title: &str, content: &str, app_name: Option<&str>) -> String {
    // Phase 1: App name matching (strongest signal for screen captures)
    if let Some(app) = app_name {
        let app_lower = app.to_lowercase();
        if let Some(cat) = classify_by_app_name(&app_lower) {
            return cat.to_string();
        }
    }

    // Phase 2: Content/title heuristics
    let title_lower = title.to_lowercase();

    // File extensions in title
    if has_code_extension(&title_lower) {
        return "code".to_string();
    }

    // Code patterns in content (UTF-8 safe: find valid char boundary)
    let sample_end = content
        .char_indices()
        .take_while(|(i, _)| *i < 2000)
        .last()
        .map_or(0, |(i, c)| i + c.len_utf8());
    let content_sample = &content[..sample_end];
    if has_code_patterns(content_sample) {
        return "code".to_string();
    }

    // URL patterns
    if content_sample.contains("http://") || content_sample.contains("https://") {
        return "research".to_string();
    }

    // Phase 3: Source type fallback
    match source {
        "local_files" => {
            if has_code_extension(&title_lower) {
                return "code".to_string();
            }
            // Most local indexed files are code
            return "code".to_string();
        }
        "clipboard" => {
            let trimmed = content.trim();
            if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                return "research".to_string();
            }
        }
        _ => {}
    }

    // Phase 4: Final fallback
    "other".to_string()
}

fn classify_by_app_name(app_lower: &str) -> Option<&'static str> {
    // Communication apps
    const COMMUNICATION_APPS: &[&str] = &[
        "slack",
        "discord",
        "mail",
        "messages",
        "teams",
        "zoom",
        "telegram",
        "whatsapp",
        "signal",
        "skype",
        "microsoft teams",
        "facetime",
        "webex",
    ];
    for &app in COMMUNICATION_APPS {
        if app_lower.contains(app) {
            return Some("communication");
        }
    }

    // Code editors / terminals
    const CODE_APPS: &[&str] = &[
        "visual studio code",
        "vs code",
        "code",
        "terminal",
        "iterm",
        "xcode",
        "jetbrains",
        "intellij",
        "webstorm",
        "pycharm",
        "rustrover",
        "clion",
        "goland",
        "cursor",
        "zed",
        "sublime",
        "vim",
        "neovim",
        "emacs",
        "warp",
        "alacritty",
        "kitty",
        "hyper",
        "datagrip",
        "android studio",
    ];
    for &app in CODE_APPS {
        if app_lower.contains(app) {
            return Some("code");
        }
    }

    // Design tools
    const DESIGN_APPS: &[&str] = &[
        "figma",
        "sketch",
        "photoshop",
        "illustrator",
        "affinity",
        "canva",
        "invision",
        "adobe xd",
        "framer",
    ];
    for &app in DESIGN_APPS {
        if app_lower.contains(app) {
            return Some("design");
        }
    }

    // Writing / notes apps
    const WRITING_APPS: &[&str] = &[
        "obsidian",
        "notion",
        "word",
        "pages",
        "bear",
        "typora",
        "ulysses",
        "ia writer",
        "scrivener",
        "google docs",
        "notes",
        "evernote",
        "craft",
    ];
    for &app in WRITING_APPS {
        if app_lower.contains(app) {
            return Some("writing");
        }
    }

    // Browsers → research
    const BROWSER_APPS: &[&str] = &[
        "chrome", "safari", "arc", "firefox", "brave", "edge", "opera", "vivaldi", "orion",
    ];
    for &app in BROWSER_APPS {
        if app_lower.contains(app) {
            return Some("research");
        }
    }

    None
}

fn has_code_extension(title: &str) -> bool {
    const CODE_EXTENSIONS: &[&str] = &[
        ".rs", ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".java", ".kt", ".swift", ".c", ".cpp",
        ".h", ".hpp", ".cs", ".rb", ".php", ".scala", ".clj", ".ex", ".exs", ".zig", ".lua", ".sh",
        ".bash", ".zsh", ".toml", ".yaml", ".yml", ".json", ".xml", ".html", ".css", ".scss",
        ".sql", ".graphql",
    ];
    CODE_EXTENSIONS.iter().any(|ext| title.ends_with(ext))
}

fn has_code_patterns(content: &str) -> bool {
    const CODE_PATTERNS: &[&str] = &[
        "fn ",
        "function ",
        "class ",
        "import ",
        "def ",
        "pub fn",
        "async fn",
        "const ",
        "let ",
        "var ",
        "=> {",
        "-> {",
        "impl ",
        "struct ",
        "enum ",
        "#include",
        "package ",
        "module ",
    ];
    let matches = CODE_PATTERNS
        .iter()
        .filter(|p| content.contains(**p))
        .count();
    // Require at least 2 code patterns to classify as code
    matches >= 2
}

fn tags_path() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("origin")
        .join("tags.json")
}

pub fn load_tags() -> TagStore {
    let path = tags_path();
    let mut store: TagStore = match std::fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => TagStore::default(),
    };
    store.ensure_default_categories();
    store
}

pub fn save_tags(store: &TagStore) -> Result<(), OriginError> {
    let path = tags_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(store)?;
    std::fs::write(&path, json)?;
    Ok(())
}

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

    fn empty_store() -> TagStore {
        let mut store = TagStore::default();
        store.ensure_default_categories();
        store
    }

    #[test]
    fn test_default_categories_populated() {
        let store = empty_store();
        assert!(store.categories.contains(&"code".to_string()));
        assert!(store.categories.contains(&"other".to_string()));
        assert_eq!(store.categories.len(), 6);
    }

    #[test]
    fn test_ensure_default_categories_idempotent() {
        let mut store = empty_store();
        let before = store.categories.clone();
        store.ensure_default_categories();
        assert_eq!(store.categories, before);
    }

    #[test]
    fn test_set_document_tags_normalizes() {
        let mut store = empty_store();
        let result = store.set_document_tags("local", "id1", vec!["Rust ".into(), " CODE".into()]);
        assert_eq!(result, vec!["code".to_string(), "rust".to_string()]);
    }

    #[test]
    fn test_set_document_tags_adds_to_library() {
        let mut store = empty_store();
        store.set_document_tags("s", "id", vec!["newtag".into()]);
        assert!(store.tags.contains("newtag"));
    }

    #[test]
    fn test_set_empty_tags_removes_entry() {
        let mut store = empty_store();
        store.set_document_tags("s", "id", vec!["tag1".into()]);
        store.set_document_tags("s", "id", vec![]);
        assert_eq!(store.get_document_tags("s", "id"), Vec::<String>::new());
    }

    #[test]
    fn test_delete_tag_removes_from_documents() {
        let mut store = empty_store();
        store.set_document_tags("s", "id1", vec!["a".into(), "b".into()]);
        store.set_document_tags("s", "id2", vec!["b".into(), "c".into()]);
        store.delete_tag("b");
        assert_eq!(store.get_document_tags("s", "id1"), vec!["a".to_string()]);
        assert_eq!(store.get_document_tags("s", "id2"), vec!["c".to_string()]);
        assert!(!store.tags.contains("b"));
    }

    #[test]
    fn test_remove_document() {
        let mut store = empty_store();
        store.set_document_tags("s", "id1", vec!["a".into()]);
        store.set_document_category("s", "id1", "code");
        store.remove_document("s", "id1");
        assert_eq!(store.get_document_tags("s", "id1"), Vec::<String>::new());
        assert_eq!(store.get_document_category("s", "id1"), None);
    }

    #[test]
    fn test_set_document_category_validates() {
        let mut store = empty_store();
        store.set_document_category("s", "id", "code");
        assert_eq!(
            store.get_document_category("s", "id"),
            Some("code".to_string())
        );
        store.set_document_category("s", "id2", "nonexistent");
        assert_eq!(store.get_document_category("s", "id2"), None);
    }

    #[test]
    fn test_add_category_dedup() {
        let mut store = empty_store();
        let before = store.categories.len();
        store.add_category("code");
        assert_eq!(store.categories.len(), before);
        store.add_category("  NewCat  ");
        assert!(store.categories.contains(&"newcat".to_string()));
    }

    #[test]
    fn test_remove_category_reassigns_to_other() {
        let mut store = empty_store();
        store.set_document_category("s", "id", "code");
        store.remove_category("code");
        assert_eq!(
            store.get_document_category("s", "id"),
            Some("other".to_string())
        );
        assert!(!store.categories.contains(&"code".to_string()));
    }

    #[test]
    fn test_remove_other_is_noop() {
        let mut store = empty_store();
        let before = store.categories.len();
        store.remove_category("other");
        assert_eq!(store.categories.len(), before);
    }

    #[test]
    fn test_classify_code_by_extension() {
        let cats: Vec<String> = vec!["code".into(), "other".into()];
        assert_eq!(
            classify_document("local_files", "main.rs", "", None, &cats),
            "code"
        );
    }

    #[test]
    fn test_classify_communication_by_app() {
        let cats: Vec<String> = vec!["communication".into(), "other".into()];
        assert_eq!(
            classify_document("screen", "Chat", "", Some("Slack"), &cats),
            "communication"
        );
    }

    #[test]
    fn test_classify_research_by_url() {
        let cats: Vec<String> = vec!["research".into(), "other".into()];
        assert_eq!(
            classify_document(
                "clip",
                "t",
                "see https://example.com for details",
                None,
                &cats
            ),
            "research"
        );
    }

    #[test]
    fn test_classify_falls_back_to_other() {
        let cats: Vec<String> = vec!["code".into(), "other".into()];
        assert_eq!(
            classify_document("unknown", "untitled", "hello world", None, &cats),
            "other"
        );
    }

    #[test]
    fn test_classify_with_unicode_content_no_panic() {
        // Regression: content_sample used byte-indexing which panics on multi-byte chars
        let cats: Vec<String> = vec!["code".into(), "other".into()];
        // Create content with multi-byte chars near the 2000-char boundary
        let content = "日本語テスト ".repeat(400); // 6 chars × 400 = 2400 chars, multi-byte
                                                   // Should not panic
        let result = classify_document("test", "unicode.txt", &content, None, &cats);
        assert!(!result.is_empty());
    }

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
