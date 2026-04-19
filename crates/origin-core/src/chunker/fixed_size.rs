// SPDX-License-Identifier: AGPL-3.0-only
use super::traits::{floor_char_boundary, ChunkContext, ChunkInfo, ChunkingStrategy};

/// Fixed-size chunking strategy with intelligent split points
///
/// Uses 512 character chunks with 64 character overlap.
/// Split priority: paragraph boundary > sentence boundary > word boundary > hard split.
pub struct FixedSizeStrategy {
    chunk_size: usize,
    overlap: usize,
}

impl FixedSizeStrategy {
    pub fn new() -> Self {
        Self {
            chunk_size: 512,
            overlap: 64,
        }
    }

    /// Find the best split point near `end` within the range.
    /// Preference: paragraph > sentence > word > hard split.
    fn find_split_point(&self, text: &str, start: usize, end: usize) -> usize {
        let search_region = &text[start..end];

        // Look for paragraph boundary (double newline) in the last 25% of the chunk
        let search_start = floor_char_boundary(search_region, search_region.len() * 3 / 4);
        if let Some(pos) = search_region[search_start..].rfind("\n\n") {
            return start + search_start + pos + 2;
        }

        // Look for sentence boundary in the last 25%
        if let Some(pos) = search_region[search_start..].rfind(". ") {
            return start + search_start + pos + 2;
        }

        // Look for any whitespace near the end
        if let Some(pos) = search_region[search_start..].rfind(' ') {
            return start + search_start + pos + 1;
        }

        // Hard split
        end
    }
}

impl Default for FixedSizeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkingStrategy for FixedSizeStrategy {
    fn chunk(&self, context: ChunkContext) -> Vec<ChunkInfo> {
        if context.content.is_empty() {
            return vec![];
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let text_len = context.content.len();

        while start < text_len {
            let end = floor_char_boundary(context.content, (start + self.chunk_size).min(text_len));

            // Find a good split point
            let split_at = if end >= text_len {
                text_len
            } else {
                self.find_split_point(context.content, start, end)
            };

            let chunk_text = &context.content[start..split_at];
            let chunk_text = chunk_text.trim();

            if !chunk_text.is_empty() {
                chunks.push(ChunkInfo {
                    content: chunk_text.to_string(),
                    chunk_type: "text".to_string(),
                    language: None,
                    byte_range: Some((start, split_at)),
                    semantic_unit: Some("paragraph".to_string()),
                });
            }

            // Move start forward, with overlap
            if split_at >= text_len {
                break;
            }
            start = floor_char_boundary(
                context.content,
                if split_at > self.overlap {
                    split_at - self.overlap
                } else {
                    split_at
                },
            );
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_context<'a>(
        content: &'a str,
        title: &'a str,
        metadata: &'a HashMap<String, String>,
    ) -> ChunkContext<'a> {
        ChunkContext {
            content,
            title,
            file_path: "test.txt",
            metadata,
        }
    }

    #[test]
    fn test_empty_text() {
        let strategy = FixedSizeStrategy::new();
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context("", "", &metadata));
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_short_text() {
        let strategy = FixedSizeStrategy::new();
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context("Hello, world!", "", &metadata));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hello, world!");
        assert_eq!(chunks[0].chunk_type, "text");
    }

    #[test]
    fn test_title_not_prefixed() {
        let strategy = FixedSizeStrategy::new();
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context("Some content here", "README.md", &metadata));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Some content here");
    }

    #[test]
    fn test_long_text_chunking() {
        let strategy = FixedSizeStrategy::new();
        let metadata = HashMap::new();
        let text = "word ".repeat(200); // ~1000 chars
        let chunks = strategy.chunk(make_context(&text, "", &metadata));
        assert!(chunks.len() > 1);

        // Each chunk should be roughly 512 chars or less
        for chunk in &chunks {
            assert!(
                chunk.content.len() <= 600,
                "Chunk too long: {} chars",
                chunk.content.len()
            );
        }

        // Check byte ranges
        for chunk in &chunks {
            assert!(chunk.byte_range.is_some());
        }
    }

    #[test]
    fn test_paragraph_split() {
        let strategy = FixedSizeStrategy::new();
        let metadata = HashMap::new();
        let para1 = "a".repeat(400);
        let para2 = "b".repeat(200);
        let text = format!("{}\n\n{}", para1, para2);
        let chunks = strategy.chunk(make_context(&text, "", &metadata));
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_semantic_unit() {
        let strategy = FixedSizeStrategy::new();
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context("Test content", "", &metadata));
        assert_eq!(chunks[0].semantic_unit, Some("paragraph".to_string()));
    }
}
