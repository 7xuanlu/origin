// SPDX-License-Identifier: AGPL-3.0-only
//! Knowledge writer — exports concepts as `.md` files with state tracking.

use crate::concepts::Concept;
use crate::error::OriginError;
use crate::export::obsidian::{convert_links_to_wikilinks, slugify};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Default, Serialize, Deserialize)]
struct KnowledgeState {
    concepts: HashMap<String, ConceptFileState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConceptFileState {
    file: String,
    version: i64,
    last_written: String,
}

pub struct KnowledgeWriter {
    path: PathBuf,
}

impl KnowledgeWriter {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn write_concept(&self, concept: &Concept) -> Result<String, OriginError> {
        let origin_dir = self.path.join(".origin");
        std::fs::create_dir_all(&origin_dir)?;

        let filename = format!("{}.md", slugify(&concept.title));
        let file_path = self.path.join(&filename);

        let content = render_markdown(concept);
        std::fs::write(&file_path, &content)?;

        let mut state = self.load_state();
        state.concepts.insert(
            concept.id.clone(),
            ConceptFileState {
                file: filename,
                version: concept.version,
                last_written: concept.last_modified.clone(),
            },
        );
        self.save_state(&state)?;

        Ok(file_path.to_string_lossy().to_string())
    }

    pub fn remove_concept(&self, concept_id: &str) -> Result<(), OriginError> {
        let mut state = self.load_state();

        if let Some(entry) = state.concepts.remove(concept_id) {
            let file_path = self.path.join(&entry.file);
            if file_path.exists() {
                std::fs::remove_file(&file_path)?;
            }
            self.save_state(&state)?;
        }

        Ok(())
    }

    fn load_state(&self) -> KnowledgeState {
        let state_path = self.path.join(".origin/state.json");
        match std::fs::read_to_string(&state_path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => KnowledgeState::default(),
        }
    }

    fn save_state(&self, state: &KnowledgeState) -> Result<(), OriginError> {
        let state_path = self.path.join(".origin/state.json");
        let data = serde_json::to_string_pretty(state)?;
        std::fs::write(&state_path, data)?;
        Ok(())
    }
}

fn render_markdown(concept: &Concept) -> String {
    let mut out = String::new();

    // Frontmatter
    out.push_str("---\n");
    out.push_str(&format!("title: \"{}\"\n", concept.title));
    if let Some(ref domain) = concept.domain {
        out.push_str(&format!("domain: {}\n", domain));
    }
    out.push_str(&format!("origin_id: {}\n", concept.id));
    out.push_str(&format!("origin_version: {}\n", concept.version));
    let created_date: String = concept.created_at.chars().take(10).collect();
    let modified_date: String = concept.last_modified.chars().take(10).collect();
    out.push_str(&format!("created: {}\n", created_date));
    out.push_str(&format!("modified: {}\n", modified_date));
    out.push_str("---\n\n");

    // Body with wikilinks
    out.push_str(&convert_links_to_wikilinks(&concept.content));
    out.push('\n');

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concepts::Concept;

    fn test_concept() -> Concept {
        Concept {
            id: "concept_test123".to_string(),
            title: "Rust Ownership".to_string(),
            summary: Some("Memory safety without GC".to_string()),
            content: "## Overview\nRust uses ownership for memory safety.\n\n## Related\n- [Borrowing](concept_borrow1)".to_string(),
            entity_id: None,
            domain: Some("rust".to_string()),
            source_memory_ids: vec!["m1".to_string()],
            version: 2,
            status: "active".to_string(),
            created_at: "2026-04-01T00:00:00+00:00".to_string(),
            last_compiled: "2026-04-09T00:00:00+00:00".to_string(),
            last_modified: "2026-04-09T00:00:00+00:00".to_string(),
        }
    }

    #[test]
    fn test_write_concept_creates_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new(dir.path().to_path_buf());
        let concept = test_concept();

        let path = writer.write_concept(&concept).unwrap();
        assert!(path.ends_with("rust-ownership.md"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("---\n"));
        assert!(content.contains("title: \"Rust Ownership\""));
        assert!(content.contains("domain: rust"));
        assert!(content.contains("origin_id: concept_test123"));
        assert!(content.contains("origin_version: 2"));
        // Wikilinks converted
        assert!(content.contains("[[Borrowing]]"));
        assert!(!content.contains("(concept_borrow1)"));
    }

    #[test]
    fn test_write_updates_state() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new(dir.path().to_path_buf());

        writer.write_concept(&test_concept()).unwrap();

        let state = writer.load_state();
        assert!(state.concepts.contains_key("concept_test123"));
        assert_eq!(state.concepts["concept_test123"].file, "rust-ownership.md");
        assert_eq!(state.concepts["concept_test123"].version, 2);
    }

    #[test]
    fn test_remove_concept_deletes_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new(dir.path().to_path_buf());

        let path = writer.write_concept(&test_concept()).unwrap();
        assert!(std::path::Path::new(&path).exists());

        writer.remove_concept("concept_test123").unwrap();
        assert!(!std::path::Path::new(&path).exists());

        let state = writer.load_state();
        assert!(!state.concepts.contains_key("concept_test123"));
    }

    #[test]
    fn test_remove_nonexistent_concept_noop() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new(dir.path().to_path_buf());
        writer.remove_concept("nonexistent").unwrap();
    }

    #[test]
    fn test_write_multiple_concepts() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new(dir.path().to_path_buf());

        let c1 = Concept {
            id: "concept_a".to_string(),
            title: "Alpha".to_string(),
            ..test_concept()
        };
        let c2 = Concept {
            id: "concept_b".to_string(),
            title: "Beta".to_string(),
            ..test_concept()
        };

        writer.write_concept(&c1).unwrap();
        writer.write_concept(&c2).unwrap();

        assert!(dir.path().join("alpha.md").exists());
        assert!(dir.path().join("beta.md").exists());

        let state = writer.load_state();
        assert_eq!(state.concepts.len(), 2);
    }

    #[test]
    fn test_knowledge_writer_overwrite_on_version_change() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new(dir.path().to_path_buf());

        let mut concept = test_concept();
        writer.write_concept(&concept).unwrap();

        let v1 = std::fs::read_to_string(dir.path().join("rust-ownership.md")).unwrap();
        assert!(v1.contains("origin_version: 2"));

        // Update version and content
        concept.version = 3;
        concept.content = "## Updated\nNew content.".to_string();
        writer.write_concept(&concept).unwrap();

        let v2 = std::fs::read_to_string(dir.path().join("rust-ownership.md")).unwrap();
        assert!(v2.contains("origin_version: 3"));
        assert!(v2.contains("## Updated"));
        assert!(!v2.contains("memory safety")); // old content replaced

        // State reflects new version
        let state = writer.load_state();
        assert_eq!(state.concepts["concept_test123"].version, 3);
    }

    #[test]
    fn test_knowledge_writer_no_domain() {
        let dir = tempfile::TempDir::new().unwrap();
        let writer = KnowledgeWriter::new(dir.path().to_path_buf());

        let concept = Concept {
            domain: None,
            ..test_concept()
        };
        let path = writer.write_concept(&concept).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(!content.contains("domain:"));
    }
}
