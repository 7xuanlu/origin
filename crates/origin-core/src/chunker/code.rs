// SPDX-License-Identifier: AGPL-3.0-only
use super::detection::{detect_content_type, CodeLanguage, ContentType};
use super::traits::{ChunkContext, ChunkInfo, ChunkingStrategy};
use regex::Regex;

/// Code chunking strategy that splits by function/class boundaries
///
/// MVP approach: Regex-based function detection
/// Fallback: Line-based chunking (100 lines per chunk, 10 line overlap)
pub struct CodeStrategy {
    rust_pattern: Regex,
    python_pattern: Regex,
    javascript_pattern: Regex,
    typescript_pattern: Regex,
    go_pattern: Regex,
    java_pattern: Regex,
    c_pattern: Regex,
    max_chunk_lines: usize,
    overlap_lines: usize,
}

impl CodeStrategy {
    pub fn new() -> Self {
        Self {
            // Rust: functions, structs, enums, impls
            rust_pattern: Regex::new(
                r"(?m)^(pub\s+)?(async\s+)?(fn|struct|enum|impl|trait)\s+\w+"
            ).unwrap(),

            // Python: functions and classes
            python_pattern: Regex::new(
                r"(?m)^(async\s+)?(def|class)\s+\w+"
            ).unwrap(),

            // JavaScript: functions, classes, exports
            javascript_pattern: Regex::new(
                r"(?m)^(export\s+)?(async\s+)?(function|class|const\s+\w+\s*=\s*(async\s+)?\(|const\s+\w+\s*=\s*(async\s+)?function)"
            ).unwrap(),

            // TypeScript: similar to JavaScript but with type annotations
            typescript_pattern: Regex::new(
                r"(?m)^(export\s+)?(async\s+)?(function|class|interface|type\s+\w+\s*=|const\s+\w+:\s*\w+\s*=|const\s+\w+\s*=)"
            ).unwrap(),

            // Go: functions, structs, interfaces
            go_pattern: Regex::new(
                r"(?m)^func\s+(\(\w+\s+\*?\w+\)\s+)?\w+|^type\s+\w+\s+(struct|interface)"
            ).unwrap(),

            // Java: classes, methods, interfaces
            java_pattern: Regex::new(
                r"(?m)^(public|private|protected)?\s*(static\s+)?(class|interface|enum)\s+\w+|^\s*(public|private|protected)\s+.*\s+\w+\s*\("
            ).unwrap(),

            // C/C++: functions
            c_pattern: Regex::new(
                r"(?m)^\w+\s+\w+\s*\([^)]*\)\s*\{"
            ).unwrap(),

            max_chunk_lines: 100,
            overlap_lines: 10,
        }
    }

    /// Get the appropriate regex pattern for a language
    fn get_pattern(&self, language: &CodeLanguage) -> Option<&Regex> {
        match language {
            CodeLanguage::Rust => Some(&self.rust_pattern),
            CodeLanguage::Python => Some(&self.python_pattern),
            CodeLanguage::JavaScript => Some(&self.javascript_pattern),
            CodeLanguage::TypeScript => Some(&self.typescript_pattern),
            CodeLanguage::Go => Some(&self.go_pattern),
            CodeLanguage::Java => Some(&self.java_pattern),
            CodeLanguage::C | CodeLanguage::Cpp => Some(&self.c_pattern),
            _ => None,
        }
    }

    /// Extract functions/classes using regex
    fn extract_semantic_units(
        &self,
        content: &str,
        pattern: &Regex,
    ) -> Vec<(usize, usize, String)> {
        let lines: Vec<&str> = content.lines().collect();
        let mut units = Vec::new();
        let mut current_start: Option<usize> = None;

        for (idx, line) in lines.iter().enumerate() {
            if pattern.is_match(line) {
                // Found a new function/class
                if let Some(start) = current_start {
                    // Save the previous unit
                    units.push((start, idx, "function".to_string()));
                }
                current_start = Some(idx);
            }
        }

        // Save the last unit
        if let Some(start) = current_start {
            units.push((start, lines.len(), "function".to_string()));
        }

        units
    }

    /// Fallback: line-based chunking
    fn chunk_by_lines(&self, content: &str, language: Option<String>) -> Vec<ChunkInfo> {
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < lines.len() {
            let end = (start + self.max_chunk_lines).min(lines.len());
            let chunk_lines = &lines[start..end];
            let chunk_content = chunk_lines.join("\n");

            if !chunk_content.trim().is_empty() {
                chunks.push(ChunkInfo {
                    content: chunk_content,
                    chunk_type: "code".to_string(),
                    language: language.clone(),
                    byte_range: None, // Would need to calculate byte offsets
                    semantic_unit: Some("block".to_string()),
                });
            }

            if end >= lines.len() {
                break;
            }
            start = if end > self.overlap_lines {
                end - self.overlap_lines
            } else {
                end
            };
        }

        chunks
    }
}

impl Default for CodeStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkingStrategy for CodeStrategy {
    fn chunk(&self, context: ChunkContext) -> Vec<ChunkInfo> {
        // Detect language from file path
        let language = match detect_content_type(context.file_path) {
            ContentType::Code(lang) => lang,
            _ => CodeLanguage::Unknown,
        };

        let language_str = language.as_str().to_string();

        // Try to extract semantic units if we have a pattern
        if let Some(pattern) = self.get_pattern(&language) {
            let units = self.extract_semantic_units(context.content, pattern);

            if !units.is_empty() {
                let lines: Vec<&str> = context.content.lines().collect();
                let mut chunks = Vec::new();

                for (start_line, end_line, semantic_type) in units {
                    let chunk_lines = &lines[start_line..end_line];
                    let chunk_content = chunk_lines.join("\n");

                    if !chunk_content.trim().is_empty() {
                        chunks.push(ChunkInfo {
                            content: chunk_content,
                            chunk_type: "code".to_string(),
                            language: Some(language_str.clone()),
                            byte_range: None, // Could calculate if needed
                            semantic_unit: Some(semantic_type),
                        });
                    }
                }

                if !chunks.is_empty() {
                    return chunks;
                }
            }
        }

        // Fallback to line-based chunking
        self.chunk_by_lines(context.content, Some(language_str))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_context<'a>(
        content: &'a str,
        file_path: &'a str,
        metadata: &'a HashMap<String, String>,
    ) -> ChunkContext<'a> {
        ChunkContext {
            content,
            title: "Test",
            file_path,
            metadata,
        }
    }

    #[test]
    fn test_rust_functions() {
        let strategy = CodeStrategy::new();
        let code = r#"
fn foo() {
    println!("foo");
}

pub fn bar() -> i32 {
    42
}

pub async fn baz() {
    // async code
}
"#;
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context(code, "test.rs", &metadata));

        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].chunk_type, "code");
        assert_eq!(chunks[0].language, Some("rust".to_string()));
        assert_eq!(chunks[0].semantic_unit, Some("function".to_string()));
    }

    #[test]
    fn test_python_functions() {
        let strategy = CodeStrategy::new();
        let code = r#"
def hello():
    print("hello")

async def world():
    await something()

class MyClass:
    def method(self):
        pass
"#;
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context(code, "test.py", &metadata));

        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].language, Some("python".to_string()));
    }

    #[test]
    fn test_javascript_functions() {
        let strategy = CodeStrategy::new();
        let code = r#"
function hello() {
    console.log("hello");
}

const world = () => {
    console.log("world");
}

export async function test() {
    return true;
}
"#;
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context(code, "test.js", &metadata));

        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].language, Some("javascript".to_string()));
    }

    #[test]
    fn test_typescript_code() {
        let strategy = CodeStrategy::new();
        let code = r#"
interface User {
    name: string;
}

type ID = string | number;

export function getUser(): User {
    return { name: "test" };
}
"#;
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context(code, "test.ts", &metadata));

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].language, Some("typescript".to_string()));
    }

    #[test]
    fn test_line_based_fallback() {
        let strategy = CodeStrategy::new();
        let code = "a\n".repeat(150); // 150 lines
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context(&code, "test.rs", &metadata));

        // Should split into multiple chunks
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].semantic_unit, Some("block".to_string()));
    }

    #[test]
    fn test_unknown_language() {
        let strategy = CodeStrategy::new();
        let code = "some code";
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context(code, "test.unknown", &metadata));

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chunk_type, "code");
    }

    #[test]
    fn test_empty_code() {
        let strategy = CodeStrategy::new();
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context("", "test.rs", &metadata));
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_go_functions() {
        let strategy = CodeStrategy::new();
        let code = r#"
func Hello() {
    fmt.Println("hello")
}

type User struct {
    Name string
}

func (u User) GetName() string {
    return u.Name
}
"#;
        let metadata = HashMap::new();
        let chunks = strategy.chunk(make_context(code, "test.go", &metadata));

        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].language, Some("go".to_string()));
    }
}
