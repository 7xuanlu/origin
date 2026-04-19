// SPDX-License-Identifier: AGPL-3.0-only
//! PromptRegistry — holds all LLM prompt templates used by the intelligence pipeline.
//! Loaded from override files with compiled-in defaults as fallback.
pub(crate) mod defaults;

use std::path::{Path, PathBuf};

/// Holds all LLM prompt templates used by the intelligence pipeline.
/// Loaded from files with compiled-in defaults as fallback.
#[derive(Debug, Clone)]
pub struct PromptRegistry {
    pub classify_memory: String,
    pub classify_memory_quality: String,
    pub classify_memory_quality_strict: String,
    pub classify_screen: String,
    pub merge_memories: String,
    pub detect_contradiction: String,
    pub summarize_decisions: String,
    pub detect_pattern: String,
    pub narrative: String,
    pub briefing_topic: String,
    pub rerank_results: String,
    pub summarize_activity_system: String,
    pub summarize_activity_user: String,
    pub batch_classify: String,
    pub extract_knowledge_graph: String,
    pub format_ocr_text: String,
    pub extract_structured_fields: String, // template with {memory_type}, {fields_json}, {required}, {optional}
    pub correct_memory: String,            // template with {original}, {correction}
    pub distill_concept: String,
    pub update_concept: String,
    pub assign_orphans: String,
    pub global_concept_review: String,
    pub refine_clusters: String,
}

impl Default for PromptRegistry {
    fn default() -> Self {
        Self {
            classify_memory: defaults::CLASSIFY_MEMORY.to_string(),
            classify_memory_quality: defaults::CLASSIFY_MEMORY_QUALITY.to_string(),
            classify_memory_quality_strict: defaults::CLASSIFY_MEMORY_QUALITY_STRICT.to_string(),
            classify_screen: defaults::CLASSIFY_SCREEN.to_string(),
            merge_memories: defaults::MERGE_MEMORIES.to_string(),
            detect_contradiction: defaults::DETECT_CONTRADICTION.to_string(),
            summarize_decisions: defaults::SUMMARIZE_DECISIONS.to_string(),
            detect_pattern: defaults::DETECT_PATTERN.to_string(),
            narrative: defaults::NARRATIVE.to_string(),
            briefing_topic: defaults::BRIEFING_TOPIC.to_string(),
            rerank_results: defaults::RERANK_RESULTS.to_string(),
            summarize_activity_system: defaults::SUMMARIZE_ACTIVITY_SYSTEM.to_string(),
            summarize_activity_user: defaults::SUMMARIZE_ACTIVITY_USER.to_string(),
            batch_classify: defaults::BATCH_CLASSIFY.to_string(),
            extract_knowledge_graph: defaults::EXTRACT_KNOWLEDGE_GRAPH.to_string(),
            format_ocr_text: defaults::FORMAT_OCR_TEXT.to_string(),
            extract_structured_fields: defaults::EXTRACT_STRUCTURED_FIELDS.to_string(),
            correct_memory: defaults::CORRECT_MEMORY.to_string(),
            distill_concept: defaults::DISTILL_CONCEPT.to_string(),
            update_concept: defaults::UPDATE_CONCEPT.to_string(),
            assign_orphans: defaults::ASSIGN_ORPHANS.to_string(),
            global_concept_review: defaults::GLOBAL_CONCEPT_REVIEW.to_string(),
            refine_clusters: defaults::REFINE_CLUSTERS.to_string(),
        }
    }
}

impl PromptRegistry {
    /// Load prompts from an override directory, falling back to defaults.
    /// Each prompt can be overridden by placing a file named `<field_name>.txt`
    /// in the override directory (e.g., `classify_memory.txt`).
    pub fn load(override_dir: &Path) -> Self {
        let mut reg = Self::default();
        let fields: Vec<(&str, &mut String)> = vec![
            ("classify_memory", &mut reg.classify_memory),
            ("classify_memory_quality", &mut reg.classify_memory_quality),
            (
                "classify_memory_quality_strict",
                &mut reg.classify_memory_quality_strict,
            ),
            ("classify_screen", &mut reg.classify_screen),
            ("merge_memories", &mut reg.merge_memories),
            ("detect_contradiction", &mut reg.detect_contradiction),
            ("summarize_decisions", &mut reg.summarize_decisions),
            ("detect_pattern", &mut reg.detect_pattern),
            ("narrative", &mut reg.narrative),
            ("briefing_topic", &mut reg.briefing_topic),
            ("rerank_results", &mut reg.rerank_results),
            (
                "summarize_activity_system",
                &mut reg.summarize_activity_system,
            ),
            ("summarize_activity_user", &mut reg.summarize_activity_user),
            ("batch_classify", &mut reg.batch_classify),
            ("extract_knowledge_graph", &mut reg.extract_knowledge_graph),
            ("format_ocr_text", &mut reg.format_ocr_text),
            (
                "extract_structured_fields",
                &mut reg.extract_structured_fields,
            ),
            ("correct_memory", &mut reg.correct_memory),
            ("distill_concept", &mut reg.distill_concept),
            ("update_concept", &mut reg.update_concept),
            ("assign_orphans", &mut reg.assign_orphans),
            ("global_concept_review", &mut reg.global_concept_review),
            ("refine_clusters", &mut reg.refine_clusters),
        ];
        for (name, value) in fields {
            let path = override_dir.join(format!("{name}.txt"));
            if let Ok(content) = std::fs::read_to_string(&path) {
                let trimmed = content.trim().to_string();
                if !trimmed.is_empty() {
                    log::info!("[prompts] loaded override: {name}");
                    *value = trimmed;
                }
            }
        }
        reg
    }

    /// Returns the prompt override directory path.
    pub fn override_dir() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("origin")
            .join("prompts")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_default_registry_has_all_prompts() {
        let reg = PromptRegistry::default();
        assert!(!reg.classify_memory.is_empty());
        assert!(!reg.classify_memory_quality.is_empty());
        assert!(!reg.classify_memory_quality_strict.is_empty());
        assert!(!reg.classify_screen.is_empty());
        assert!(!reg.merge_memories.is_empty());
        assert!(!reg.detect_contradiction.is_empty());
        assert!(!reg.summarize_decisions.is_empty());
        assert!(!reg.detect_pattern.is_empty());
        assert!(!reg.narrative.is_empty());
        assert!(!reg.briefing_topic.is_empty());
        assert!(!reg.rerank_results.is_empty());
        assert!(!reg.summarize_activity_system.is_empty());
        assert!(!reg.summarize_activity_user.is_empty());
        assert!(!reg.batch_classify.is_empty());
        assert!(!reg.extract_knowledge_graph.is_empty());
        assert!(!reg.format_ocr_text.is_empty());
        assert!(!reg.extract_structured_fields.is_empty());
        assert!(!reg.correct_memory.is_empty());
        assert!(!reg.distill_concept.is_empty());
        assert!(!reg.update_concept.is_empty());
        assert!(!reg.assign_orphans.is_empty());
        assert!(!reg.global_concept_review.is_empty());
    }

    #[test]
    fn test_load_overrides_from_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("narrative.txt"), "Custom narrative prompt").unwrap();

        let reg = PromptRegistry::load(dir.path());
        assert_eq!(reg.narrative, "Custom narrative prompt");
        // Non-overridden prompts should still have defaults
        assert_eq!(reg.briefing_topic, defaults::BRIEFING_TOPIC);
    }

    #[test]
    fn test_load_nonexistent_dir_returns_defaults() {
        let reg = PromptRegistry::load(Path::new("/nonexistent/path"));
        assert_eq!(reg.classify_memory, defaults::CLASSIFY_MEMORY);
    }

    #[test]
    fn test_empty_override_file_keeps_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("narrative.txt"), "   \n  ").unwrap();

        let reg = PromptRegistry::load(dir.path());
        assert_eq!(reg.narrative, defaults::NARRATIVE);
    }
}
