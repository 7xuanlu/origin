// SPDX-License-Identifier: AGPL-3.0-only
//! Daily briefing generation — LLM-assisted activity summaries.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::prompts::PromptRegistry;
use crate::tuning::BriefingConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefingResponse {
    pub content: String,
    pub new_today: u64,
    pub primary_agent: Option<String>,
    pub generated_at: i64,
    pub is_stale: bool,
}

/// A memory row loaded for briefing assembly.
#[derive(Debug, Clone, Serialize)]
pub struct BriefingMemory {
    pub title: String,
    pub content: String,
    pub memory_type: String,
    pub domain: Option<String>,
    pub last_modified: i64,
}

/// Stats for template assembly — computed from DB, no LLM needed.
#[derive(Debug, Clone)]
pub struct BriefingStats {
    pub dominant_domain: Option<String>,
    pub primary_agent: Option<String>,
    pub new_today: u64,
}

/// A pending contradiction surfaced by the refinement queue.
#[derive(Debug, Clone, Serialize)]
pub struct ContradictionItem {
    pub id: String,
    pub new_content: String,
    pub existing_content: String,
    pub new_source_id: String,
    pub existing_source_id: String,
}

/// Returns true if the cached briefing is stale.
pub fn is_cache_stale(
    generated_at: i64,
    cached_count: u64,
    current_count: u64,
    tuning: &BriefingConfig,
) -> bool {
    let now = chrono::Utc::now().timestamp();
    now - generated_at > tuning.stale_secs
        || current_count >= cached_count + tuning.stale_memory_delta
}

/// Build the topic extraction prompt from recent memory titles.
fn extract_topics_prompt(memories: &[BriefingMemory], tuning: &BriefingConfig) -> String {
    memories
        .iter()
        .take(tuning.max_topic_memories)
        .map(|m| {
            format!(
                "- {}",
                m.title
                    .chars()
                    .take(tuning.max_memory_chars)
                    .collect::<String>()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Fallback: format recent titles as a newline-separated list for structured display.
fn titles_fallback(memories: &[BriefingMemory]) -> Option<String> {
    let titles: Vec<String> = memories
        .iter()
        .take(3)
        .map(|m| {
            // Use title, truncate to keep scannable
            let t = m.title.trim();
            if t.is_empty() {
                m.content.chars().take(60).collect::<String>()
            } else {
                t.chars().take(60).collect::<String>()
            }
        })
        .filter(|t| !t.is_empty())
        .collect();
    if titles.is_empty() {
        None
    } else {
        Some(titles.join("\n"))
    }
}

/// Assemble the briefing body (content only — stats go to separate fields).
/// Returns either an LLM-generated sentence or newline-separated titles (for list rendering).
fn assemble_briefing(topics: Option<&str>, fallback_memories: &[BriefingMemory]) -> String {
    match topics {
        Some(t) if !t.is_empty() => {
            let clean = t.trim().trim_end_matches('.');
            format!("{}.", clean)
        }
        _ => {
            // Fallback: newline-separated titles (frontend renders as bullet list)
            titles_fallback(fallback_memories).unwrap_or_default()
        }
    }
}

/// Generate a briefing using template + optional LLM topic extraction.
pub async fn generate_briefing(
    db: &MemoryDB,
    llm: Option<&dyn LlmProvider>,
    prompts: &PromptRegistry,
    tuning: &BriefingConfig,
) -> Result<BriefingResponse, OriginError> {
    let now = chrono::Utc::now().timestamp();
    let cutoff = now - 48 * 3600;

    // 1. Fetch stats + recent memories
    let stats = db.get_briefing_stats(cutoff).await?;
    let memories = db
        .get_recent_memories_for_briefing(cutoff, tuning.max_topic_memories)
        .await?;

    // 2. Try LLM summarization from titles
    let topics = if let Some(llm) = llm {
        if llm.is_available() && !memories.is_empty() {
            let prompt = extract_topics_prompt(&memories, tuning);
            match llm
                .generate(LlmRequest {
                    system_prompt: Some(prompts.briefing_topic.clone()),
                    user_prompt: prompt,
                    max_tokens: 60,
                    temperature: 0.1,
                    label: None,
                })
                .await
            {
                Ok(t) => {
                    let cleaned = crate::llm_provider::strip_think_tags(&t);
                    let cleaned = cleaned.trim().to_string();
                    if cleaned.chars().count() < 100 {
                        Some(cleaned)
                    } else {
                        None
                    }
                }
                Err(e) => {
                    log::warn!("[briefing] topic extraction failed: {e}");
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // 3. Assemble — body only (stats go to separate fields)
    let content = assemble_briefing(topics.as_deref(), &memories);

    // 4. Cache
    let memory_count = db.get_memory_count().await?;
    db.upsert_briefing_cache(&content, memory_count).await?;

    Ok(BriefingResponse {
        content,
        new_today: stats.new_today,
        primary_agent: stats.primary_agent,
        generated_at: now,
        is_stale: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cache_stale_fresh() {
        let now = chrono::Utc::now().timestamp();
        let tuning = BriefingConfig::default();
        assert!(!is_cache_stale(now - 3600, 10, 10, &tuning));
        assert!(!is_cache_stale(now - 3600, 10, 14, &tuning));
    }

    #[test]
    fn test_is_cache_stale_by_time() {
        let now = chrono::Utc::now().timestamp();
        let tuning = BriefingConfig::default();
        assert!(is_cache_stale(now - 7 * 3600, 10, 10, &tuning));
    }

    #[test]
    fn test_is_cache_stale_by_count() {
        let now = chrono::Utc::now().timestamp();
        let tuning = BriefingConfig::default();
        assert!(is_cache_stale(now - 3600, 10, 15, &tuning));
        assert!(!is_cache_stale(now - 3600, 10, 14, &tuning));
    }

    fn mem(title: &str) -> BriefingMemory {
        BriefingMemory {
            title: title.into(),
            content: title.into(),
            memory_type: "fact".into(),
            domain: Some("engineering".into()),
            last_modified: 1000,
        }
    }

    #[test]
    fn test_assemble_with_llm_topics() {
        let memories = vec![mem("x")];
        let result = assemble_briefing(
            Some("You worked on the homepage redesign and LLM provider migration"),
            &memories,
        );
        assert_eq!(
            result,
            "You worked on the homepage redesign and LLM provider migration."
        );
    }

    #[test]
    fn test_assemble_fallback_uses_titles_as_list() {
        let memories = vec![
            mem("LLM provider migration"),
            mem("Setup wizard onboarding"),
            mem("MCP transport decision"),
        ];
        let result = assemble_briefing(None, &memories);
        assert_eq!(
            result,
            "LLM provider migration\nSetup wizard onboarding\nMCP transport decision"
        );
    }

    #[test]
    fn test_assemble_empty() {
        let result = assemble_briefing(None, &[]);
        assert_eq!(result, "");
    }

    #[test]
    fn test_extract_topics_uses_titles() {
        let memories = vec![
            mem("Homepage redesign"),
            mem("LLM provider trait migration"),
        ];
        let prompt = extract_topics_prompt(&memories, &BriefingConfig::default());
        assert!(prompt.contains("- Homepage redesign"));
        assert!(prompt.contains("- LLM provider trait migration"));
    }

    #[test]
    fn test_topics_strips_trailing_period() {
        let memories = vec![mem("x")];
        let result = assemble_briefing(Some("homepage redesign."), &memories);
        assert_eq!(result, "homepage redesign.");
    }
}
