// SPDX-License-Identifier: AGPL-3.0-only
pub mod code;
pub mod detection;
pub mod fixed_size;
pub mod markdown;
pub mod traits;

use detection::{detect_content_type, ContentType};
use std::collections::HashMap;
use traits::{ChunkContext, ChunkInfo, ChunkingStrategy};

/// Main chunking engine that coordinates different strategies
pub struct ChunkingEngine {
    strategies: HashMap<String, Box<dyn ChunkingStrategy>>,
}

impl ChunkingEngine {
    /// Creates a new ChunkingEngine with all available strategies
    pub fn new() -> Self {
        let mut strategies: HashMap<String, Box<dyn ChunkingStrategy>> = HashMap::new();

        strategies.insert(
            "markdown".to_string(),
            Box::new(markdown::MarkdownStrategy::new()),
        );
        strategies.insert("code".to_string(), Box::new(code::CodeStrategy::new()));
        strategies.insert(
            "text".to_string(),
            Box::new(fixed_size::FixedSizeStrategy::new()),
        );

        Self { strategies }
    }

    /// Chunks content using the appropriate strategy based on file path
    pub fn chunk(
        &self,
        content: &str,
        title: &str,
        file_path: &str,
        metadata: &HashMap<String, String>,
    ) -> Vec<ChunkInfo> {
        let content_type = detect_content_type(file_path);

        let strategy_name = match content_type {
            ContentType::Markdown => "markdown",
            ContentType::Code(_) => "code",
            ContentType::PlainText => "text",
        };

        // Screen captures: LLM-formatted uses markdown strategy (has ### headers),
        // raw screen captures use per-window chunking (each ## section = 1 chunk).
        let is_llm_formatted = metadata.get("llm_formatted").is_some_and(|v| v == "true");
        let is_screen = metadata.get("screen_capture").is_some_and(|v| v == "true")
            || file_path.starts_with("screen_");
        let is_raw_screen = is_screen && !is_llm_formatted;

        if is_raw_screen {
            let context = ChunkContext {
                content,
                title,
                file_path,
                metadata,
            };
            return split_window_chunks(context);
        }

        let strategy_name = if is_llm_formatted {
            "markdown"
        } else {
            strategy_name
        };

        let context = ChunkContext {
            content,
            title,
            file_path,
            metadata,
        };

        self.strategies
            .get(strategy_name)
            .expect("Strategy should exist")
            .chunk(context)
    }
}

/// Split screen capture content into per-window chunks.
/// Each `## ` section becomes exactly one chunk — no subdivision.
/// Pass 1 prioritizes 1-window-per-chunk; the LLM pass will re-chunk later.
fn split_window_chunks(context: ChunkContext) -> Vec<ChunkInfo> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut byte_offset: usize = 0;

    for line in context.content.lines() {
        if line.starts_with("## ") && !current.is_empty() {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                chunks.push(ChunkInfo {
                    content: trimmed.to_string(),
                    chunk_type: "markdown".to_string(),
                    language: None,
                    byte_range: Some((byte_offset, byte_offset + current.len())),
                    semantic_unit: Some("window".to_string()),
                });
            }
            byte_offset += current.len() + 1;
            current.clear();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    // Flush last section
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        chunks.push(ChunkInfo {
            content: trimmed.to_string(),
            chunk_type: "markdown".to_string(),
            language: None,
            byte_range: Some((byte_offset, byte_offset + current.len())),
            semantic_unit: Some("window".to_string()),
        });
    }
    // Fallback: if no ## headers were found, emit the whole content as one chunk
    if chunks.is_empty() && !context.content.trim().is_empty() {
        chunks.push(ChunkInfo {
            content: context.content.trim().to_string(),
            chunk_type: "markdown".to_string(),
            language: None,
            byte_range: Some((0, context.content.len())),
            semantic_unit: Some("window".to_string()),
        });
    }
    chunks
}

impl Default for ChunkingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = ChunkingEngine::new();
        assert_eq!(engine.strategies.len(), 3);
    }

    #[test]
    fn test_markdown_routing() {
        let engine = ChunkingEngine::new();
        let metadata = HashMap::new();

        let chunks = engine.chunk("# Title\nContent", "Test", "test.md", &metadata);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "markdown");
    }

    #[test]
    fn test_code_routing() {
        let engine = ChunkingEngine::new();
        let metadata = HashMap::new();

        let chunks = engine.chunk("fn main() {}", "Test", "main.rs", &metadata);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "code");
    }

    #[test]
    fn test_text_routing() {
        let engine = ChunkingEngine::new();
        let metadata = HashMap::new();

        let chunks = engine.chunk("Plain text content", "Test", "file.txt", &metadata);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "text");
    }

    #[test]
    fn test_llm_formatted_uses_markdown_strategy() {
        let engine = ChunkingEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert("llm_formatted".to_string(), "true".to_string());

        // source_id with no extension would normally route to "text"
        let chunks = engine.chunk(
            "# Screen Capture\n\nSome formatted content\n\n## Section Two\n\nMore content here",
            "Test",
            "screen_sustained_focus_abc123",
            &metadata,
        );

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "markdown");
    }

    #[test]
    fn test_non_screen_stays_text() {
        let engine = ChunkingEngine::new();
        let metadata = HashMap::new();

        let chunks = engine.chunk("Plain text content", "Test", "file.txt", &metadata);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "text");
    }

    #[test]
    fn test_raw_screen_capture_per_window_chunking() {
        let engine = ChunkingEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert("screen_capture".to_string(), "true".to_string());

        // 3 windows — should produce 3 chunks (one per window)
        let content = "## VS Code — main.rs\nfn main() { println!(\"hello\"); }\n\n## Chrome — Google\nSearch results for Rust\n\n## Finder — Documents\nfile1.txt file2.txt";
        let chunks = engine.chunk(content, "Test", "ctx_1740000000", &metadata);

        assert_eq!(
            chunks.len(),
            3,
            "Expected 1 chunk per window, got {}",
            chunks.len()
        );
        assert!(chunks[0].content.contains("VS Code"));
        assert!(chunks[1].content.contains("Chrome"));
        assert!(chunks[2].content.contains("Finder"));
        assert_eq!(chunks[0].semantic_unit, Some("window".to_string()));
    }

    #[test]
    fn test_raw_screen_capture_single_window() {
        let engine = ChunkingEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert("screen_capture".to_string(), "true".to_string());

        let content = "## VS Code — main.rs\nfn main() {}";
        let chunks = engine.chunk(content, "Test", "ctx_1740000000", &metadata);

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("VS Code"));
    }

    #[test]
    fn test_raw_screen_no_headers_fallback() {
        let engine = ChunkingEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert("screen_capture".to_string(), "true".to_string());

        let content = "Just some plain text without any headers";
        let chunks = engine.chunk(content, "Test", "ctx_1740000000", &metadata);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, content);
    }

    #[test]
    fn test_screen_source_id_uses_window_chunking() {
        let engine = ChunkingEngine::new();
        let metadata = HashMap::new();

        // Raw screen capture (no llm_formatted) with screen_ prefix → per-window chunks
        let chunks = engine.chunk(
            "## App Title\nSome OCR content from the screen capture that was structured.",
            "Test",
            "screen_sustained_focus_abc123",
            &metadata,
        );

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, "markdown");
        assert_eq!(chunks[0].semantic_unit, Some("window".to_string()));
    }
}
