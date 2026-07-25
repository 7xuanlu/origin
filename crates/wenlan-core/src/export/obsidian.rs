// SPDX-License-Identifier: Apache-2.0
//! Obsidian vault exporter for pages.

use crate::error::WenlanError;
use crate::export::{ExportResult, ExportStats, PageExporter};
use crate::pages::Page;
use std::path::PathBuf;

pub struct ObsidianExporter {
    vault_path: PathBuf,
}

impl ObsidianExporter {
    pub fn new(vault_path: PathBuf) -> Self {
        Self { vault_path }
    }
}

impl PageExporter for ObsidianExporter {
    fn export(&self, page: &Page) -> Result<ExportResult, WenlanError> {
        std::fs::create_dir_all(&self.vault_path)?;

        let filename = format!("{}.md", slugify(&page.title));
        let path = self.vault_path.join(&filename);

        let frontmatter = build_frontmatter(page);
        let body = convert_links_to_wikilinks(&page.content);
        let output = format!("---\n{}---\n\n{}\n", frontmatter, body);

        std::fs::write(&path, &output)?;

        Ok(ExportResult {
            concept_id: page.id.clone(),
            path: path.to_string_lossy().to_string(),
        })
    }

    fn export_all(&self, pages: &[Page]) -> Result<ExportStats, WenlanError> {
        let mut stats = ExportStats::default();
        for page in pages {
            match self.export(page) {
                Ok(_) => stats.exported += 1,
                Err(e) => {
                    log::warn!("[obsidian] export failed for '{}': {}", page.title, e);
                    stats.failed += 1;
                }
            }
        }
        Ok(stats)
    }
}

fn build_frontmatter(page: &Page) -> String {
    let mut fm = String::new();
    fm.push_str(&format!("title: {}\n", page.title));
    if let Some(ref space) = page.space {
        fm.push_str(&format!("tags:\n  - {}\n", space));
    }
    if let Some(ref summary) = page.summary {
        fm.push_str(&format!("aliases:\n  - {}\n", summary));
    }
    fm.push_str(&format!("origin_id: {}\n", page.id));
    fm.push_str(&format!("origin_version: {}\n", page.version));
    // Use chars().take(10) for date extraction — safe for ISO dates
    let created_date: String = page.created_at.chars().take(10).collect();
    let modified_date: String = page.last_modified.chars().take(10).collect();
    fm.push_str(&format!("created: {}\n", created_date));
    fm.push_str(&format!("modified: {}\n", modified_date));
    fm
}

/// Convert `[Title](concept_id)` markdown links to `[[Title]]` wikilinks.
pub fn convert_links_to_wikilinks(content: &str) -> String {
    let re = regex::Regex::new(r"\[([^\]]+)\]\(concept_[a-zA-Z0-9\-]+\)").unwrap();
    re.replace_all(content, "[[${1}]]").to_string()
}

/// Slugify a title for use as a filename.
pub fn slugify(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else if c == ' ' {
                '-'
            } else {
                '\0' // will be filtered
            }
        })
        .filter(|&c| c != '\0')
        .collect::<String>()
        .split("--")
        .collect::<Vec<_>>()
        .join("-")
        .trim_matches('-')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pages::Page;

    fn test_concept() -> Page {
        Page {
            id: "concept_abc".to_string(),
            title: "libSQL Architecture".to_string(),
            summary: Some("Core database layer".to_string()),
            content: "## Key Facts\n- Stores vectors\n\n## Related Concepts\n- [Wenlan Architecture](concept_xyz789)\n- [Embedding Pipeline](concept_pqr012)".to_string(),
            entity_id: Some("entity_libsql".to_string()),
            space: Some("architecture".to_string()),
            source_memory_ids: vec!["m1".to_string(), "m2".to_string()],
            version: 2,
            status: "active".to_string(),
            created_at: "2026-04-01T00:00:00+00:00".to_string(),
            last_compiled: "2026-04-07T00:00:00+00:00".to_string(),
            last_modified: "2026-04-07T00:00:00+00:00".to_string(),
            sources_updated_count: 0,
            stale_reason: None,
            pending_rebuild: None,
            user_edited: false,
            relevance_score: 0.0,
            last_edited_by: None,
            last_edited_at: None,
            last_delta_summary: None,
            changelog: None,
            creation_kind: "distilled".to_string(),
            review_status: "confirmed".to_string(),
            workspace: None,
            citations: Vec::new(),
            kind: "concept".to_string(),
        }
    }

    #[test]
    fn test_obsidian_export_creates_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let exporter = ObsidianExporter::new(dir.path().to_path_buf());
        let page = test_concept();

        let result = exporter.export(&page).unwrap();
        let file_content = std::fs::read_to_string(&result.path).unwrap();

        // Check frontmatter
        assert!(file_content.starts_with("---\n"));
        assert!(file_content.contains("title: libSQL Architecture"));
        assert!(file_content.contains("origin_id: concept_abc"));
        assert!(file_content.contains("origin_version: 2"));
        assert!(file_content.contains("created: 2026-04-01"));
        assert!(file_content.contains("modified: 2026-04-07"));

        // Check wikilinks conversion
        assert!(file_content.contains("[[Wenlan Architecture]]"));
        assert!(file_content.contains("[[Embedding Pipeline]]"));
        assert!(!file_content.contains("(concept_xyz789)"));
        assert!(!file_content.contains("(concept_pqr012)"));

        // Check file name
        assert!(result.path.ends_with("libsql-architecture.md"));
    }

    #[test]
    fn test_export_all() {
        let dir = tempfile::TempDir::new().unwrap();
        let exporter = ObsidianExporter::new(dir.path().to_path_buf());
        let pages = vec![test_concept()];

        let stats = exporter.export_all(&pages).unwrap();
        assert_eq!(stats.exported, 1);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_slugify_title() {
        assert_eq!(slugify("libSQL Architecture"), "libsql-architecture");
        assert_eq!(slugify("Wenlan's MCP (v2)"), "wenlans-mcp-v2");
        assert_eq!(slugify("  Spaces  "), "spaces");
        assert_eq!(slugify("Hello World"), "hello-world");
    }

    #[test]
    fn test_convert_links_to_wikilinks() {
        let input = "See [Wenlan Architecture](concept_abc-123-def) for details.";
        let output = convert_links_to_wikilinks(input);
        assert_eq!(output, "See [[Wenlan Architecture]] for details.");
    }

    #[test]
    fn test_convert_links_preserves_non_concept_links() {
        let input = "See [Google](https://google.com) for details.";
        let output = convert_links_to_wikilinks(input);
        assert_eq!(output, "See [Google](https://google.com) for details.");
    }
}
