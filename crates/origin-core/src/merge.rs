// SPDX-License-Identifier: AGPL-3.0-only
//! Memory merging, pattern extraction, and contradiction detection.
//!
//! These operations compose multiple memories into higher-level insights:
//! deduplicating near-duplicates ([`LlmEngine::merge_memories`]), extracting
//! durable patterns from clusters ([`LlmEngine::extract_pattern`]), and
//! detecting whether a new memory contradicts or supersedes an existing one
//! ([`LlmEngine::detect_contradiction`]).

use crate::engine::{LlmEngine, CTX_SIZE};

/// Outcome of comparing a new memory against an existing one.
#[derive(Debug, Clone, PartialEq)]
pub enum ContradictionResult {
    Consistent,
    Contradicts { explanation: String },
    Supersedes { merged_content: String },
}

#[allow(dead_code)] // Wired via refinery / contradiction detection pipelines
impl LlmEngine {
    /// Merge near-duplicate memories into one consolidated version.
    pub fn merge_memories(&self, contents: &[String]) -> Option<String> {
        let combined = contents
            .iter()
            .enumerate()
            .map(|(i, c)| format!("Memory {}: {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {combined}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
            sys = self.prompts().merge_memories,
        );

        self.run_inference(&prompt, 512, 0.1, CTX_SIZE, Some("merge"))
    }

    /// Extract a durable pattern (preference, habit, identity trait) from a cluster of related memories.
    /// Returns None if no clear pattern exists.
    pub fn extract_pattern(&self, cluster: &[String], domain: Option<&str>) -> Option<String> {
        let combined = cluster
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n");

        let domain_hint = domain.unwrap_or("general");
        let sys = self
            .prompts()
            .detect_pattern
            .replace("{domain_hint}", domain_hint);
        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {combined}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
        );

        let result = self.run_inference(&prompt, 128, 0.1, CTX_SIZE, Some("merge"))?;
        if result.to_lowercase().trim() == "null" {
            None
        } else {
            Some(result)
        }
    }

    /// Detect if two memories contradict each other, are consistent, or if one supersedes the other.
    pub fn detect_contradiction(
        &self,
        new_content: &str,
        existing_content: &str,
    ) -> ContradictionResult {
        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             Existing: {existing_content}\n\
             New: {new_content}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
            sys = self.prompts().detect_contradiction,
        );

        match self.run_inference(&prompt, 256, 0.1, CTX_SIZE, Some("merge")) {
            Some(r) if r.starts_with("CONTRADICTS:") => ContradictionResult::Contradicts {
                explanation: r
                    .strip_prefix("CONTRADICTS:")
                    .unwrap_or("")
                    .trim()
                    .to_string(),
            },
            Some(r) if r.starts_with("SUPERSEDES:") => ContradictionResult::Supersedes {
                merged_content: r
                    .strip_prefix("SUPERSEDES:")
                    .unwrap_or("")
                    .trim()
                    .to_string(),
            },
            _ => ContradictionResult::Consistent,
        }
    }
}
