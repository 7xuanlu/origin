// SPDX-License-Identifier: AGPL-3.0-only
//! Profile narrative generation — LLM-assisted prose about the user.

use crate::db::MemoryDB;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::prompts::PromptRegistry;
use crate::tuning::NarrativeConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeResponse {
    pub content: String,
    pub generated_at: i64,
    pub is_stale: bool,
    pub memory_count: u64,
}

/// A memory row loaded for narrative assembly.
#[derive(Debug, Clone)]
pub struct NarrativeMemory {
    pub source_id: String,
    pub title: String,
    pub content: String,
    pub memory_type: String,
}

/// Returns true if the cached narrative is stale (older than configured threshold).
pub fn is_cache_stale(generated_at: i64, stale_secs: i64) -> bool {
    let now = chrono::Utc::now().timestamp();
    now - generated_at > stale_secs
}

/// Template-based narrative assembly — builds a flowing paragraph without LLM.
///
/// Groups memories by type and weaves them into connected sentences.
pub fn assemble_narrative_template(memories: &[NarrativeMemory]) -> String {
    if memories.is_empty() {
        return String::new();
    }

    // Only profile types belong in the narrative brief
    let memories: Vec<&NarrativeMemory> = memories
        .iter()
        .filter(|m| m.memory_type != "decision" && m.memory_type != "fact")
        .collect();

    // Group by type
    let mut identities = Vec::new();
    let mut preferences = Vec::new();
    let mut goals = Vec::new();

    for mem in &memories {
        let text = if !mem.title.is_empty() {
            &mem.title
        } else {
            &mem.content
        };
        let clean = text.trim().trim_end_matches('.').to_string();
        if clean.is_empty() {
            continue;
        }
        match mem.memory_type.as_str() {
            "identity" => identities.push(clean),
            "preference" => preferences.push(clean),
            "goal" => goals.push(clean),
            _ => {}
        }
    }

    let mut parts = Vec::new();

    // Identity: "You're X and Y."
    if !identities.is_empty() {
        let joined = join_naturally(&identities, 2);
        parts.push(format!("You're {}.", lowercase_first(&joined)));
    }

    // Preferences: "You prefer X and Y."
    if !preferences.is_empty() {
        let joined = join_naturally(&preferences, 2);
        parts.push(format!("You prefer {}.", lowercase_first(&joined)));
    }

    // Goals: "You're currently focused on X."
    if !goals.is_empty() {
        let joined = join_naturally(&goals, 2);
        parts.push(format!(
            "You're currently focused on {}.",
            lowercase_first(&joined)
        ));
    }

    parts.join(" ")
}

/// Join items naturally: "X", "X and Y", or "X, Y, and Z". Caps at `max` items.
fn join_naturally(items: &[String], max: usize) -> String {
    let items: Vec<&str> = items.iter().take(max).map(|s| s.as_str()).collect();
    match items.len() {
        0 => String::new(),
        1 => items[0].to_string(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => {
            let last = items[items.len() - 1];
            let rest = &items[..items.len() - 1];
            format!("{}, and {}", rest.join(", "), last)
        }
    }
}

/// Lowercase the first character of a string, preserving the rest.
fn lowercase_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Build the LLM prompt from memories.
fn build_narrative_prompt(memories: &[NarrativeMemory]) -> String {
    memories
        .iter()
        .map(|m| {
            let content = if !m.title.is_empty() {
                &m.title
            } else {
                &m.content
            };
            format!("- ({}) {}", m.memory_type, content.trim())
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate a profile narrative using template + optional LLM enhancement.
pub async fn generate_narrative(
    db: &MemoryDB,
    llm: Option<&dyn LlmProvider>,
    prompts: &PromptRegistry,
    tuning: &NarrativeConfig,
) -> Result<NarrativeResponse, crate::error::OriginError> {
    let now = chrono::Utc::now().timestamp();

    let memories = db.get_memories_for_narrative(tuning.max_memories).await?;

    if memories.is_empty() {
        return Ok(NarrativeResponse {
            content: String::new(),
            generated_at: now,
            is_stale: false,
            memory_count: 0,
        });
    }

    // Try LLM synthesis — returns a flowing paragraph string
    let content = if let Some(llm) = llm {
        if llm.is_available() {
            let prompt = build_narrative_prompt(&memories);
            match llm
                .generate(LlmRequest {
                    system_prompt: Some(prompts.narrative.clone()),
                    user_prompt: prompt,
                    max_tokens: 200,
                    temperature: 0.3,
                    label: None,
                })
                .await
            {
                Ok(raw) => {
                    let cleaned = crate::llm_provider::strip_think_tags(&raw);
                    let cleaned = cleaned.trim().to_string();
                    // If LLM output is reasonable length, use it; otherwise fallback
                    if cleaned.chars().count() > 20 && cleaned.chars().count() < 500 {
                        cleaned
                    } else {
                        assemble_narrative_template(&memories)
                    }
                }
                Err(e) => {
                    log::warn!("[narrative] LLM synthesis failed: {e}");
                    assemble_narrative_template(&memories)
                }
            }
        } else {
            assemble_narrative_template(&memories)
        }
    } else {
        assemble_narrative_template(&memories)
    };

    // Cache
    let memory_count = db.get_narrative_memory_count().await?;
    db.upsert_narrative_cache(&content, memory_count).await?;

    Ok(NarrativeResponse {
        content,
        generated_at: now,
        is_stale: false,
        memory_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(title: &str, memory_type: &str) -> NarrativeMemory {
        NarrativeMemory {
            source_id: "test".into(),
            title: title.into(),
            content: title.into(),
            memory_type: memory_type.into(),
        }
    }

    #[test]
    fn test_template_empty() {
        assert_eq!(assemble_narrative_template(&[]), "");
    }

    #[test]
    fn test_template_single_identity() {
        let memories = vec![mem("A software engineer", "identity")];
        let result = assemble_narrative_template(&memories);
        assert_eq!(result, "You're a software engineer.");
    }

    #[test]
    fn test_template_flowing_paragraph() {
        let memories = vec![
            mem("A Rust developer", "identity"),
            mem("Dark mode for all UIs", "preference"),
            mem("To use libSQL over SQLite", "decision"),
            mem("Ship the MVP by March", "goal"),
        ];
        let result = assemble_narrative_template(&memories);
        // Decision types are filtered out — narrative only includes profile types
        assert_eq!(
            result,
            "You're a Rust developer. You prefer dark mode for all UIs. You're currently focused on ship the MVP by March."
        );
    }

    #[test]
    fn test_template_multiple_same_type() {
        let memories = vec![
            mem("A Rust developer", "identity"),
            mem("Full-stack engineer", "identity"),
        ];
        let result = assemble_narrative_template(&memories);
        assert_eq!(result, "You're a Rust developer and Full-stack engineer.");
    }

    #[test]
    fn test_template_skips_empty() {
        let memories = vec![mem("", "identity"), mem("A developer", "identity")];
        let result = assemble_narrative_template(&memories);
        assert_eq!(result, "You're a developer.");
    }

    #[test]
    fn test_template_strips_trailing_period() {
        let memories = vec![mem("A software engineer.", "identity")];
        let result = assemble_narrative_template(&memories);
        assert_eq!(result, "You're a software engineer.");
    }

    #[test]
    fn test_join_naturally() {
        assert_eq!(join_naturally(&[], 3), "");
        assert_eq!(join_naturally(&["A".into()], 3), "A");
        assert_eq!(join_naturally(&["A".into(), "B".into()], 3), "A and B");
        assert_eq!(
            join_naturally(&["A".into(), "B".into(), "C".into()], 3),
            "A, B, and C"
        );
    }

    #[test]
    fn test_join_naturally_caps() {
        let items = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        assert_eq!(join_naturally(&items, 2), "A and B");
    }

    #[test]
    fn test_lowercase_first() {
        assert_eq!(lowercase_first("Hello"), "hello");
        assert_eq!(lowercase_first("hello"), "hello");
        assert_eq!(lowercase_first(""), "");
    }

    #[test]
    fn test_build_prompt() {
        let memories = vec![
            mem("A Rust developer", "identity"),
            mem("Dark mode", "preference"),
        ];
        let prompt = build_narrative_prompt(&memories);
        assert!(prompt.contains("- (identity) A Rust developer"));
        assert!(prompt.contains("- (preference) Dark mode"));
    }

    #[test]
    fn test_assemble_narrative_excludes_decision() {
        let memories = vec![
            mem("Rust engineer", "identity"),
            mem("Prefers TDD", "preference"),
            mem("Used libSQL over Neo4j", "decision"),
            mem("Ship wiki this weekend", "goal"),
        ];
        let result = assemble_narrative_template(&memories);
        assert!(
            !result.contains("libSQL"),
            "Narrative should not contain decision content, got: {}",
            result
        );
        assert!(
            !result.contains("Neo4j"),
            "Narrative should not contain decision content"
        );
    }

    #[test]
    fn test_is_cache_stale() {
        let now = chrono::Utc::now().timestamp();
        assert!(!is_cache_stale(now - 3600, 86400)); // 1h old — fresh
        assert!(!is_cache_stale(now - 12 * 3600, 86400)); // 12h old — fresh
        assert!(is_cache_stale(now - 25 * 3600, 86400)); // 25h old — stale
    }
}
