// SPDX-License-Identifier: AGPL-3.0-only
use std::collections::HashMap;

/// Context passed to chunking strategies
#[allow(dead_code)]
pub struct ChunkContext<'a> {
    pub content: &'a str,
    pub title: &'a str,
    pub file_path: &'a str,
    pub metadata: &'a HashMap<String, String>,
}

/// Information about a single chunk produced by a strategy
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    pub content: String,
    pub chunk_type: String,       // "markdown", "code", "text"
    pub language: Option<String>, // "rust", "python", etc.
    pub byte_range: Option<(usize, usize)>,
    pub semantic_unit: Option<String>, // "section_h1", "function", "paragraph"
}

/// Trait for implementing different chunking strategies
pub trait ChunkingStrategy: Send + Sync {
    /// Chunks the content according to this strategy
    fn chunk(&self, context: ChunkContext) -> Vec<ChunkInfo>;
}

/// Snap a byte index down to the nearest valid UTF-8 char boundary.
/// Prevents panics when slicing strings with multi-byte characters (curly quotes, em dashes, etc.).
pub fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut i = index;
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}
