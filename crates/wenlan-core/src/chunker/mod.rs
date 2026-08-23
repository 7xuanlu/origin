// SPDX-License-Identifier: Apache-2.0
pub mod code;
pub mod detection;
pub mod fixed_size;
pub mod markdown;
pub mod traits;

use detection::{detect_content_type, ContentType};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;
use traits::{ChunkContext, ChunkInfo, ChunkingStrategy};

/// Maximum number of *content* tokens allowed in an embedded chunk.
///
/// BGE-Base-EN-v1.5-Q accepts 512 tokens including the two special tokens
/// (`[CLS]` … `[SEP]`) that the embedder adds at encode time. Reserving those
/// two leaves 510 tokens of content, so a chunk sized to `MAX_CONTENT_TOKENS`
/// never triggers silent truncation during embedding.
pub const MAX_CONTENT_TOKENS: usize = 510;

/// Load the BGE-Base-EN-v1.5-Q `tokenizer.json` from a resolved FastEmbed cache
/// directory, returning a truncation/padding-free [`Tokenizer`] suitable for
/// token-aware chunk sizing.
///
/// FastEmbed stores the model under the HuggingFace hub layout
/// `models--Qdrant--bge-base-en-v1.5-onnx-Q/snapshots/<hash>/tokenizer.json`.
/// Truncation and padding are stripped so the tokenizer reports the *true*
/// token count of a chunk (a truncating tokenizer would cap reported sizes and
/// defeat the budget check). Returns `None` when the file is absent or fails to
/// parse — callers fall back to character-based sizing.
pub fn load_bge_tokenizer(cache_dir: &Path) -> Option<Tokenizer> {
    // HF repo for FastEmbed's `EmbeddingModel::BGEBaseENV15Q`.
    let snapshots = cache_dir
        .join("models--Qdrant--bge-base-en-v1.5-onnx-Q")
        .join("snapshots");
    let tokenizer_path = std::fs::read_dir(&snapshots)
        .ok()?
        .flatten()
        .find_map(|e| {
            let candidate = e.path().join("tokenizer.json");
            candidate.is_file().then_some(candidate)
        })?;

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).ok()?;
    // Strip truncation/padding so `size()` counts real tokens, not a capped count.
    let _ = tokenizer.with_truncation(None);
    tokenizer.with_padding(None);
    Some(tokenizer)
}

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

    /// Creates a `ChunkingEngine` whose markdown and text strategies size chunks
    /// by BGE token count (max [`MAX_CONTENT_TOKENS`]) rather than characters, so
    /// no embedded chunk is silently truncated. The code strategy is unchanged
    /// (it splits along syntax boundaries, not size). Falls back to
    /// [`ChunkingEngine::new`] semantics for any strategy that stays char-based.
    pub fn with_tokenizer(tokenizer: Tokenizer) -> Self {
        let mut strategies: HashMap<String, Box<dyn ChunkingStrategy>> = HashMap::new();

        strategies.insert(
            "markdown".to_string(),
            Box::new(markdown::MarkdownStrategy::with_tokenizer(
                tokenizer.clone(),
            )),
        );
        strategies.insert("code".to_string(), Box::new(code::CodeStrategy::new()));
        strategies.insert(
            "text".to_string(),
            Box::new(fixed_size::FixedSizeStrategy::with_tokenizer(tokenizer)),
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
    fn test_non_screen_stays_text() {
        let engine = ChunkingEngine::new();
        let metadata = HashMap::new();

        let chunks = engine.chunk("Plain text content", "Test", "file.txt", &metadata);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "text");
    }
}
