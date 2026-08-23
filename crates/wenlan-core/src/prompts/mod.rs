// SPDX-License-Identifier: Apache-2.0
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
    pub detect_contradiction: String,
    pub resolve_dual_pool: String,
    pub doc_reconcile: String,
    pub grounding_entailment: String,
    pub summarize_decisions: String,
    pub detect_pattern: String,
    pub narrative: String,
    pub briefing_topic: String,
    pub rerank_results: String,
    pub summarize_activity_system: String,
    pub summarize_activity_user: String,
    pub extract_knowledge_graph: String,
    pub extract_structured_fields: String, // template with {memory_type}, {fields_json}, {required}, {optional}
    pub correct_memory: String,            // template with {original}, {correction}
    pub distill_page: String,
    pub overview_summary: String,
    pub update_page: String,
    pub annotate_citations: String,
    pub refine_clusters: String,
    pub compress_context: String,
}

impl Default for PromptRegistry {
    fn default() -> Self {
        Self {
            classify_memory: defaults::CLASSIFY_MEMORY.to_string(),
            classify_memory_quality: defaults::CLASSIFY_MEMORY_QUALITY.to_string(),
            detect_contradiction: defaults::DETECT_CONTRADICTION.to_string(),
            resolve_dual_pool: defaults::RESOLVE_DUAL_POOL.to_string(),
            doc_reconcile: defaults::DOC_RECONCILE.to_string(),
            grounding_entailment: defaults::GROUNDING_ENTAILMENT.to_string(),
            summarize_decisions: defaults::SUMMARIZE_DECISIONS.to_string(),
            detect_pattern: defaults::DETECT_PATTERN.to_string(),
            narrative: defaults::NARRATIVE.to_string(),
            briefing_topic: defaults::BRIEFING_TOPIC.to_string(),
            rerank_results: defaults::RERANK_RESULTS.to_string(),
            summarize_activity_system: defaults::SUMMARIZE_ACTIVITY_SYSTEM.to_string(),
            summarize_activity_user: defaults::SUMMARIZE_ACTIVITY_USER.to_string(),
            extract_knowledge_graph: defaults::EXTRACT_KNOWLEDGE_GRAPH.to_string(),
            extract_structured_fields: defaults::EXTRACT_STRUCTURED_FIELDS.to_string(),
            correct_memory: defaults::CORRECT_MEMORY.to_string(),
            distill_page: defaults::DISTILL_PAGE.to_string(),
            overview_summary: defaults::OVERVIEW_SUMMARY.to_string(),
            update_page: defaults::UPDATE_PAGE.to_string(),
            annotate_citations: defaults::ANNOTATE_CITATIONS.to_string(),
            refine_clusters: defaults::REFINE_CLUSTERS.to_string(),
            compress_context: defaults::COMPRESS_CONTEXT.to_string(),
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
            ("detect_contradiction", &mut reg.detect_contradiction),
            ("resolve_dual_pool", &mut reg.resolve_dual_pool),
            ("doc_reconcile", &mut reg.doc_reconcile),
            ("grounding_entailment", &mut reg.grounding_entailment),
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
            ("extract_knowledge_graph", &mut reg.extract_knowledge_graph),
            (
                "extract_structured_fields",
                &mut reg.extract_structured_fields,
            ),
            ("correct_memory", &mut reg.correct_memory),
            ("distill_page", &mut reg.distill_page),
            ("overview_summary", &mut reg.overview_summary),
            ("update_page", &mut reg.update_page),
            ("annotate_citations", &mut reg.annotate_citations),
            ("refine_clusters", &mut reg.refine_clusters),
            ("compress_context", &mut reg.compress_context),
        ];
        // Legacy filename fallbacks for the Phase 0 (Page) taxonomy refactor.
        // Existing user prompt overrides are likely still under the old names;
        // load them if the new-name file is absent. Drop in next minor release.
        const LEGACY_ALIASES: &[(&str, &[&str])] = &[
            ("distill_page", &["distill_concept"]),
            ("update_page", &["update_concept"]),
        ];
        for (name, value) in fields {
            let canonical = override_dir.join(format!("{name}.txt"));
            let mut loaded = false;
            if let Ok(content) = std::fs::read_to_string(&canonical) {
                let trimmed = content.trim().to_string();
                if !trimmed.is_empty() {
                    log::info!("[prompts] loaded override: {name}");
                    *value = trimmed;
                    loaded = true;
                }
            }
            if !loaded {
                let aliases: &[&str] = LEGACY_ALIASES
                    .iter()
                    .find_map(|(canon, aliases)| if *canon == name { Some(*aliases) } else { None })
                    .unwrap_or(&[]);
                for alias in aliases {
                    let alt = override_dir.join(format!("{alias}.txt"));
                    if let Ok(content) = std::fs::read_to_string(&alt) {
                        let trimmed = content.trim().to_string();
                        if !trimmed.is_empty() {
                            log::warn!(
                                "[prompts] using legacy override file '{alias}.txt'; \
                                 rename to '{name}.txt' (legacy support drops next release)"
                            );
                            *value = trimmed;
                            break;
                        }
                    }
                }
            }
        }
        reg
    }

    /// Returns the prompt override directory path.
    pub fn override_dir() -> PathBuf {
        crate::config::data_root().join("prompts")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// `override_dir` must resolve under `WENLAN_DATA_DIR` like the rest of
    /// the data-root readers, not a hand-rolled `dirs::data_local_dir()` copy
    /// that ignores `--data-dir` isolation.
    #[test]
    fn override_dir_respects_wenlan_data_dir() {
        temp_env::with_var("WENLAN_DATA_DIR", Some("/tmp/wenlan-test-scratch"), || {
            assert_eq!(
                PromptRegistry::override_dir(),
                PathBuf::from("/tmp/wenlan-test-scratch").join("prompts")
            );
        });
    }

    #[test]
    fn test_default_registry_has_all_prompts() {
        let reg = PromptRegistry::default();
        assert!(!reg.classify_memory.is_empty());
        assert!(!reg.classify_memory_quality.is_empty());
        assert!(!reg.detect_contradiction.is_empty());
        assert!(!reg.resolve_dual_pool.is_empty());
        assert!(!reg.summarize_decisions.is_empty());
        assert!(!reg.detect_pattern.is_empty());
        assert!(!reg.narrative.is_empty());
        assert!(!reg.briefing_topic.is_empty());
        assert!(!reg.rerank_results.is_empty());
        assert!(!reg.summarize_activity_system.is_empty());
        assert!(!reg.summarize_activity_user.is_empty());
        assert!(!reg.extract_knowledge_graph.is_empty());
        assert!(!reg.extract_structured_fields.is_empty());
        assert!(!reg.correct_memory.is_empty());
        assert!(!reg.distill_page.is_empty());
        assert!(!reg.overview_summary.is_empty());
        assert!(!reg.update_page.is_empty());
        assert!(!reg.annotate_citations.is_empty());
        assert!(!reg.compress_context.is_empty());
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

    // -- T10: context-compression prompt registry --
    #[test]
    fn registry_default_has_compress_context() {
        let reg = PromptRegistry::default();
        assert!(!reg.compress_context.is_empty());
    }

    #[test]
    fn registry_loads_compress_context_override() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("compress_context.txt"),
            "Custom compress prompt",
        )
        .unwrap();
        let reg = PromptRegistry::load(dir.path());
        assert_eq!(reg.compress_context, "Custom compress prompt");
        // Non-overridden prompts keep defaults.
        assert_eq!(reg.narrative, defaults::NARRATIVE);
    }
}
