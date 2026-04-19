// SPDX-License-Identifier: AGPL-3.0-only

use regex::Regex;
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use crate::chunker::ChunkingEngine;
use crate::sources::{MemoryType, RawDocument};

// ---------------------------------------------------------------------------
// Pre-compiled regexes (compiled once, reused across all calls)
// ---------------------------------------------------------------------------

static WIKILINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(!?)\[\[([^\]|#]+)(?:#([^\]|]+))?(?:\|([^\]]+))?\]\]").unwrap());

static INLINE_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|[ \t])#([a-zA-Z][a-zA-Z0-9_/\-]*)").unwrap());

static DATE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap());

static WIKILINK_DETECT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[\[").unwrap());

// ---------------------------------------------------------------------------
// Frontmatter
// ---------------------------------------------------------------------------

/// Parsed YAML frontmatter from an Obsidian note.
#[derive(Debug, Clone, Default)]
pub struct NoteFrontmatter {
    pub fields: HashMap<String, serde_yaml::Value>,
}

impl NoteFrontmatter {
    /// Extract tags from the `tags` field (list of strings).
    pub fn tags(&self) -> Vec<String> {
        self.fields
            .get("tags")
            .and_then(|v| v.as_sequence())
            .map(|seq| {
                seq.iter()
                    .filter_map(|item| item.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Extract aliases from the `aliases` field (list of strings).
    pub fn aliases(&self) -> Vec<String> {
        self.fields
            .get("aliases")
            .and_then(|v| v.as_sequence())
            .map(|seq| {
                seq.iter()
                    .filter_map(|item| item.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get a string value by key.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(|v| v.as_str())
    }
}

/// Parse YAML frontmatter delimited by `---`. Returns the parsed frontmatter
/// and a reference to the body text after the closing `---`.
pub fn extract_frontmatter(content: &str) -> (NoteFrontmatter, &str) {
    // Frontmatter must start at the very beginning of the file
    let after_open = match content.strip_prefix("---") {
        Some(rest) => rest,
        None => return (NoteFrontmatter::default(), content),
    };

    // Find the closing "\n---" delimiter
    match after_open.find("\n---") {
        Some(pos) => {
            let yaml_str = &after_open[..pos];
            // "\n---" is exactly 4 ASCII bytes -- safe to index at find() boundary
            let body_after = &after_open[pos + 4..];
            let body = body_after.trim_start_matches('\n');

            let fields: HashMap<String, serde_yaml::Value> =
                serde_yaml::from_str(yaml_str).unwrap_or_default();

            (NoteFrontmatter { fields }, body)
        }
        None => (NoteFrontmatter::default(), content),
    }
}

// ---------------------------------------------------------------------------
// Code block ranges
// ---------------------------------------------------------------------------

/// Find byte ranges of fenced code blocks (``` or ~~~) so they can be skipped
/// during wikilink/tag extraction.
pub fn code_block_ranges(content: &str) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut offset = 0;
    let mut in_block = false;
    let mut block_start = 0;
    let mut fence_char: u8 = 0;

    for line in content.split('\n') {
        let trimmed = line.trim_start();
        if !in_block {
            if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
                in_block = true;
                block_start = offset;
                fence_char = trimmed.as_bytes()[0];
            }
        } else {
            // Closing fence must use the same character
            let closing = if fence_char == b'`' { "```" } else { "~~~" };
            if trimmed.starts_with(closing) {
                let block_end = offset + line.len();
                ranges.push(block_start..block_end);
                in_block = false;
            }
        }
        // +1 for the '\n' delimiter (split consumes it)
        offset += line.len() + 1;
    }
    // Unclosed code block extends to end
    if in_block {
        ranges.push(block_start..content.len());
    }
    ranges
}

/// Check whether a byte offset falls inside any code block range.
fn in_code_block(offset: usize, ranges: &[Range<usize>]) -> bool {
    ranges.iter().any(|r| r.contains(&offset))
}

// ---------------------------------------------------------------------------
// Wikilinks
// ---------------------------------------------------------------------------

/// A parsed `[[wikilink]]` from an Obsidian note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wikilink {
    pub target: String,
    pub heading: Option<String>,
    pub display: Option<String>,
    pub is_embed: bool,
}

/// Extract all `[[wikilinks]]` from content, skipping those inside code blocks.
pub fn extract_wikilinks(content: &str) -> Vec<Wikilink> {
    let code_ranges = code_block_ranges(content);

    WIKILINK_RE
        .captures_iter(content)
        .filter(|cap| {
            let m = cap.get(0).unwrap();
            !in_code_block(m.start(), &code_ranges)
        })
        .map(|cap| Wikilink {
            is_embed: &cap[1] == "!",
            target: cap[2].to_string(),
            heading: cap.get(3).map(|m| m.as_str().to_string()),
            display: cap.get(4).map(|m| m.as_str().to_string()),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Inline tags
// ---------------------------------------------------------------------------

/// Extract inline `#tag` references, skipping code blocks.
/// Supports nested tags like `#rust/async`.
pub fn extract_inline_tags(content: &str) -> Vec<String> {
    let code_ranges = code_block_ranges(content);

    INLINE_TAG_RE
        .captures_iter(content)
        .filter(|cap| {
            let m = cap.get(0).unwrap();
            !in_code_block(m.start(), &code_ranges)
        })
        .map(|cap| cap[1].to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Vault detection & scanning
// ---------------------------------------------------------------------------

/// Check if a directory is an Obsidian vault (contains `.obsidian/` directory).
pub fn is_obsidian_vault(path: &Path) -> bool {
    path.join(".obsidian").is_dir()
}

/// Classification of an Obsidian note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteType {
    Standard,
    Daily,
    Moc,
}

/// Detect the type of an Obsidian note based on path and content.
///
/// - **Daily**: path contains `daily/` or `journal/`, or filename matches `YYYY-MM-DD`
/// - **MOC** (Map of Content): >50% of non-empty lines contain `[[wikilinks]]`
/// - **Standard**: everything else
pub fn detect_note_type(path: &Path, content: &str) -> NoteType {
    // Check path components for daily/journal directories
    let path_str = path.to_string_lossy();
    for component in path.components() {
        let name = component.as_os_str().to_string_lossy();
        if name == "daily" || name == "journal" {
            return NoteType::Daily;
        }
    }

    // Check filename for YYYY-MM-DD pattern
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        if DATE_RE.is_match(stem) {
            return NoteType::Daily;
        }
    }

    // Ignore path_str lint — it was used for the component check above
    let _ = path_str;

    // Check for MOC: >50% of non-empty lines contain wikilinks
    let non_empty_lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    if !non_empty_lines.is_empty() {
        let link_lines = non_empty_lines
            .iter()
            .filter(|l| WIKILINK_DETECT_RE.is_match(l))
            .count();
        let ratio = link_lines as f64 / non_empty_lines.len() as f64;
        if ratio > 0.5 {
            return NoteType::Moc;
        }
    }

    NoteType::Standard
}

/// Directories to skip when scanning a vault.
const SKIP_DIRS: &[&str] = &[".obsidian", ".trash", ".git", "templates"];

/// Check if a path should be skipped (any component matches a skip directory).
pub fn should_skip(path: &Path) -> bool {
    path.components().any(|c| {
        let name = c.as_os_str().to_string_lossy();
        SKIP_DIRS.contains(&name.as_ref())
    })
}

/// Recursively scan an Obsidian vault for `.md` files, skipping excluded directories.
pub fn scan_vault(root: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    scan_dir_recursive(root, root, &mut results);
    results
}

fn scan_dir_recursive(root: &Path, dir: &Path, results: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        // Compute relative path for skip checking
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if should_skip(relative) {
            continue;
        }

        if path.is_dir() {
            scan_dir_recursive(root, &path, results);
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            results.push(path);
        }
    }
}

/// Short-circuit check: returns `true` as soon as any `.md` file is found in
/// `root` or any (non-skipped) subdirectory. Used at source registration time
/// so we don't need to walk the whole vault (which can be slow on large
/// knowledge bases) just to validate "has at least one markdown file".
pub fn has_any_markdown(root: &Path) -> bool {
    has_any_markdown_recursive(root, root)
}

fn has_any_markdown_recursive(root: &Path, dir: &Path) -> bool {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return false,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if should_skip(relative) {
            continue;
        }

        if path.is_dir() {
            if has_any_markdown_recursive(root, &path) {
                return true;
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Note → RawDocument conversion
// ---------------------------------------------------------------------------

/// Convert an Obsidian note into one or more `RawDocument` structs, using the
/// `ChunkingEngine` for header-based splitting. MOC notes are skipped (they are
/// structural, not content).
///
/// TODO(schema): The memory row schema overloads `source` (kind of entry) and
/// `source_agent` (who created it). We set `source = "memory"` so these rows
/// appear in the main Memory view (list_memories filters WHERE source = 'memory'),
/// and `source_agent = "obsidian"` to preserve origin for UI differentiation.
/// A future refactor should rename: source → kind, source_agent → source,
/// source_id → external_id. Until then, new source connectors should follow
/// this pattern: `source = "memory"` + distinctive `source_agent`.
pub fn note_to_documents(
    source_id: &str,
    path: &Path,
    content: &str,
    mtime: i64,
) -> Vec<RawDocument> {
    // MOCs are structural index pages — skip them
    if detect_note_type(path, content) == NoteType::Moc {
        return Vec::new();
    }

    let (frontmatter, body) = extract_frontmatter(content);

    // Map frontmatter to memory fields
    let domain = frontmatter.tags().into_iter().next();
    let memory_type = frontmatter
        .get_str("type")
        .and_then(|t| t.parse::<MemoryType>().ok())
        .map(|mt| mt.to_string());

    let mut metadata = HashMap::new();
    let tags = frontmatter.tags();
    if !tags.is_empty() {
        metadata.insert("tags".to_string(), tags.join(","));
    }
    let aliases = frontmatter.aliases();
    if !aliases.is_empty() {
        metadata.insert("aliases".to_string(), aliases.join(","));
    }
    metadata.insert("extension".to_string(), "md".to_string());

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled")
        .to_string();
    let path_str = path.to_string_lossy();

    // Chunk the body using the engine
    let engine = ChunkingEngine::new();
    let chunks = engine.chunk(body, &stem, &path_str, &metadata);

    // If body is empty or chunker produced nothing, create a single document
    if chunks.is_empty() {
        let source_id_full = format!("{}::{}::0", source_id, path_str);
        return vec![RawDocument {
            source: "memory".to_string(),
            source_id: source_id_full,
            title: stem,
            content: body.to_string(),
            last_modified: mtime,
            metadata,
            domain,
            memory_type,
            source_agent: Some("obsidian".to_string()),
            ..Default::default()
        }];
    }

    // Build a RawDocument per chunk. Note: noise/quality filtering happens in
    // `sync::sync_obsidian_vault` via the shared `QualityGate` — this function
    // stays pure and just maps chunks to documents.
    chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            let title = if chunks.len() == 1 {
                stem.clone()
            } else {
                match &chunk.semantic_unit {
                    Some(unit) => format!("{} ({})", stem, unit),
                    None => format!("{} (chunk {})", stem, i),
                }
            };
            let source_id_full = format!("{}::{}::{}", source_id, path_str, i);
            RawDocument {
                source: "memory".to_string(),
                source_id: source_id_full,
                title,
                content: chunk.content.clone(),
                last_modified: mtime,
                metadata: metadata.clone(),
                domain: domain.clone(),
                memory_type: memory_type.clone(),
                source_agent: Some("obsidian".to_string()),
                ..Default::default()
            }
        })
        .collect()
}

// NOTE: HTML comment stripping was considered but dropped. The text-splitter
// chunker merges short sections until they reach MIN_CHARS (800), so template
// placeholders like `<!-- Process with /capture -->` no longer exist as
// standalone chunks — they get merged into larger chunks with real content,
// and the QualityGate filters the combined result. Preserving comments also
// keeps legitimate user notes (`<!-- TODO: research auth -->`) intact.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Frontmatter tests
    #[test]
    fn test_extract_frontmatter_basic() {
        let content = "---\ntitle: My Note\ntags:\n  - work\n  - rust\n---\n\n# Hello\nBody text.";
        let (fm, body) = extract_frontmatter(content);
        assert_eq!(fm.get_str("title"), Some("My Note"));
        assert_eq!(fm.tags(), vec!["work", "rust"]);
        assert!(body.starts_with("# Hello"));
    }

    #[test]
    fn test_extract_frontmatter_none() {
        let content = "# No frontmatter\nJust text.";
        let (fm, body) = extract_frontmatter(content);
        assert!(fm.fields.is_empty());
        assert_eq!(body, content);
    }

    #[test]
    fn test_extract_frontmatter_aliases() {
        let content = "---\naliases:\n  - Project X\n  - PX\n---\nBody";
        let (fm, _) = extract_frontmatter(content);
        assert_eq!(fm.aliases(), vec!["Project X", "PX"]);
    }

    // Wikilink tests
    #[test]
    fn test_extract_wikilinks_basic() {
        let content = "See [[Project Alpha]] and [[Beta#heading|display text]].";
        let links = extract_wikilinks(content);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].target, "Project Alpha");
        assert_eq!(links[0].heading, None);
        assert!(!links[0].is_embed);
        assert_eq!(links[1].target, "Beta");
        assert_eq!(links[1].heading, Some("heading".to_string()));
        assert_eq!(links[1].display, Some("display text".to_string()));
    }

    #[test]
    fn test_extract_wikilinks_embed() {
        let content = "![[image.png]] and ![[note#section]]";
        let links = extract_wikilinks(content);
        assert_eq!(links.len(), 2);
        assert!(links[0].is_embed);
        assert_eq!(links[0].target, "image.png");
        assert!(links[1].is_embed);
    }

    #[test]
    fn test_extract_wikilinks_skips_code_blocks() {
        let content = "Real [[link]] here.\n\n```\n[[not a link]]\n```\n\nAnother [[real link]].";
        let links = extract_wikilinks(content);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].target, "link");
        assert_eq!(links[1].target, "real link");
    }

    #[test]
    fn test_extract_inline_tags() {
        let content = "This is #work and #rust/async stuff.\n```\n#not-a-tag\n```";
        let tags = extract_inline_tags(content);
        assert_eq!(tags, vec!["work", "rust/async"]);
    }

    #[test]
    fn test_code_block_ranges_tilde() {
        let content = "before\n~~~\ncode\n~~~\nafter";
        let ranges = code_block_ranges(content);
        assert_eq!(ranges.len(), 1);
    }

    // Vault scanning tests
    #[test]
    fn test_is_obsidian_vault() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(!is_obsidian_vault(dir.path()));
        std::fs::create_dir(dir.path().join(".obsidian")).unwrap();
        assert!(is_obsidian_vault(dir.path()));
    }

    #[test]
    fn test_detect_note_type_daily() {
        assert_eq!(
            detect_note_type(Path::new("daily/2026-04-09.md"), "# Morning\nNotes"),
            NoteType::Daily
        );
        assert_eq!(
            detect_note_type(Path::new("journal/entry.md"), "stuff"),
            NoteType::Daily
        );
        assert_eq!(
            detect_note_type(Path::new("2026-04-09.md"), "stuff"),
            NoteType::Daily
        );
    }

    #[test]
    fn test_detect_note_type_moc() {
        let content =
            "# Index\n- [[Note A]]\n- [[Note B]]\n- [[Note C]]\n- [[Note D]]\n- [[Note E]]\nOne line of prose.";
        assert_eq!(
            detect_note_type(Path::new("index.md"), content),
            NoteType::Moc
        );
    }

    #[test]
    fn test_detect_note_type_standard() {
        let content = "# My Thoughts\n\nThis is a regular note about something interesting.\n\nIt has [[one link]] but mostly prose.";
        assert_eq!(
            detect_note_type(Path::new("thoughts.md"), content),
            NoteType::Standard
        );
    }

    #[test]
    fn test_should_skip_path() {
        assert!(should_skip(Path::new(".obsidian/plugins/foo.json")));
        assert!(should_skip(Path::new(".trash/deleted.md")));
        assert!(should_skip(Path::new(".git/config")));
        assert!(should_skip(Path::new("templates/daily.md")));
        assert!(!should_skip(Path::new("notes/my-note.md")));
    }

    #[test]
    fn test_scan_vault_files() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join(".obsidian")).unwrap();
        std::fs::create_dir(dir.path().join(".trash")).unwrap();
        std::fs::create_dir(dir.path().join("notes")).unwrap();

        std::fs::write(dir.path().join("root.md"), "# Root").unwrap();
        std::fs::write(dir.path().join("notes/sub.md"), "# Sub").unwrap();
        std::fs::write(dir.path().join(".obsidian/app.json"), "{}").unwrap();
        std::fs::write(dir.path().join(".trash/deleted.md"), "gone").unwrap();
        std::fs::write(dir.path().join("image.png"), "binary").unwrap();

        let files = scan_vault(dir.path());
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|p| p.ends_with("root.md")));
        assert!(files.iter().any(|p| p.ends_with("sub.md")));
    }

    // note_to_documents tests
    #[test]
    fn test_note_to_raw_documents() {
        let content = "---\ntags:\n  - work\ntype: decision\n---\n\n# Heading\n\nFirst section.\n\n## Details\n\nSecond section with [[Other Note]].";
        let path = Path::new("/vault/my-note.md");
        let source_id = "obsidian-main";

        let docs = note_to_documents(source_id, path, content, 1712678400);
        assert!(!docs.is_empty());
        // Obsidian imports are stored as curated memory (source = "memory"),
        // with origin identified via source_agent = "obsidian".
        assert_eq!(docs[0].source, "memory");
        assert_eq!(docs[0].source_agent, Some("obsidian".to_string()));
        assert_eq!(docs[0].domain, Some("work".to_string()));
        assert_eq!(docs[0].memory_type, Some("decision".to_string()));
        assert!(docs[0].source_id.starts_with("obsidian-main::"));
    }

    #[test]
    fn test_note_to_raw_documents_no_frontmatter() {
        let content = "Just plain text without any YAML.";
        let path = Path::new("/vault/plain.md");
        let docs = note_to_documents("src1", path, content, 1712678400);
        assert_eq!(docs.len(), 1);
        assert!(docs[0].domain.is_none());
        assert!(docs[0].memory_type.is_none());
    }

    #[test]
    fn test_moc_produces_no_documents() {
        let content = "# Index\n- [[A]]\n- [[B]]\n- [[C]]\n- [[D]]\n- [[E]]";
        let path = Path::new("/vault/moc.md");
        let docs = note_to_documents("src1", path, content, 1712678400);
        assert!(docs.is_empty());
    }

    #[test]
    fn test_note_to_documents_metadata_fields() {
        let content = "---\ntags:\n  - work\n  - rust\naliases:\n  - Project X\n  - PX\n---\n\nSome body content.";
        let path = Path::new("/vault/project.md");
        let docs = note_to_documents("obs1", path, content, 1712678400);
        assert!(!docs.is_empty());
        assert_eq!(docs[0].metadata.get("tags").unwrap(), "work,rust");
        assert_eq!(docs[0].metadata.get("aliases").unwrap(), "Project X,PX");
        assert_eq!(docs[0].metadata.get("extension").unwrap(), "md");
    }

    #[test]
    fn test_note_to_documents_invalid_memory_type() {
        let content = "---\ntype: bogus\n---\n\nBody text.";
        let path = Path::new("/vault/bad-type.md");
        let docs = note_to_documents("src1", path, content, 1712678400);
        assert_eq!(docs.len(), 1);
        // Invalid type should result in None, not a crash
        assert!(docs[0].memory_type.is_none());
    }

    #[test]
    fn test_note_to_documents_multi_chunk_titles() {
        let content = "---\ntags:\n  - dev\n---\n\n# Architecture\n\nThis section describes the architecture of the system in detail with enough text to matter.\n\n## Database Layer\n\nThe database layer uses SQLite with vector extensions for hybrid search capability.\n\n## API Layer\n\nThe API layer exposes REST endpoints for external tools to integrate with the system.";
        let path = Path::new("/vault/architecture.md");
        let docs = note_to_documents("obs1", path, content, 1712678400);
        assert!(!docs.is_empty());
        for doc in &docs {
            assert_eq!(doc.source, "memory");
            assert_eq!(doc.source_agent, Some("obsidian".to_string()));
            assert_eq!(doc.domain, Some("dev".to_string()));
        }
        if docs.len() > 1 {
            assert!(docs[1].title.contains("architecture"));
        }
    }

    #[test]
    fn test_note_to_documents_last_modified() {
        let content = "# Simple note\n\nWith content.";
        let path = Path::new("/vault/simple.md");
        let docs = note_to_documents("src1", path, content, 1712678400);
        assert_eq!(docs[0].last_modified, 1712678400);
    }

    #[test]
    fn test_obsidian_full_import_flow() {
        let vault_dir = tempfile::TempDir::new().unwrap();
        std::fs::create_dir(vault_dir.path().join(".obsidian")).unwrap();

        // Standard note with frontmatter
        std::fs::write(
            vault_dir.path().join("rust-basics.md"),
            "---\ntags:\n  - rust\ntype: fact\n---\n\n# Rust Basics\n\nOwnership is core.\n\n## Borrowing\n\nReferences don't own data.",
        )
        .unwrap();

        // Daily note
        std::fs::create_dir(vault_dir.path().join("daily")).unwrap();
        std::fs::write(
            vault_dir.path().join("daily/2026-04-09.md"),
            "# Morning\nHad coffee.\n\n## Afternoon\nWrote code.",
        )
        .unwrap();

        // MOC (should be skipped)
        std::fs::write(
            vault_dir.path().join("index.md"),
            "# Index\n- [[Rust Basics]]\n- [[Other]]\n- [[Another]]\n- [[More]]\n- [[Even More]]",
        )
        .unwrap();

        // Non-md file (should be ignored by scan)
        std::fs::write(vault_dir.path().join("image.png"), "binary").unwrap();

        // Scan vault
        let files = scan_vault(vault_dir.path());
        assert_eq!(files.len(), 3); // rust-basics, daily note, index

        // Convert all to documents
        let mut all_docs = Vec::new();
        for file in &files {
            let content = std::fs::read_to_string(file).unwrap();
            let docs = note_to_documents("test-vault", file, &content, 1712678400);
            all_docs.extend(docs);
        }

        // MOC should produce no documents
        // Non-MOC files should produce at least one chunk each (2 files)
        assert!(all_docs.len() >= 2);
        assert!(all_docs.iter().all(|d| d.source == "memory"));
        assert!(all_docs
            .iter()
            .all(|d| d.source_agent == Some("obsidian".to_string())));
        assert!(all_docs
            .iter()
            .any(|d| d.domain == Some("rust".to_string())));
        assert!(all_docs
            .iter()
            .any(|d| d.memory_type == Some("fact".to_string())));

        // Verify no MOC content leaked through
        assert!(!all_docs
            .iter()
            .any(|d| d.title.contains("Index") || d.title.contains("index")));
    }
}
