// SPDX-License-Identifier: AGPL-3.0-only
use std::path::Path;

/// Detected content type for a file
#[derive(Debug, Clone, PartialEq)]
pub enum ContentType {
    Markdown,
    Code(CodeLanguage),
    PlainText,
}

/// Supported programming languages
#[derive(Debug, Clone, PartialEq)]
pub enum CodeLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
    Ruby,
    Php,
    Swift,
    Kotlin,
    Unknown,
}

impl CodeLanguage {
    pub fn as_str(&self) -> &str {
        match self {
            CodeLanguage::Rust => "rust",
            CodeLanguage::Python => "python",
            CodeLanguage::JavaScript => "javascript",
            CodeLanguage::TypeScript => "typescript",
            CodeLanguage::Go => "go",
            CodeLanguage::Java => "java",
            CodeLanguage::C => "c",
            CodeLanguage::Cpp => "cpp",
            CodeLanguage::Ruby => "ruby",
            CodeLanguage::Php => "php",
            CodeLanguage::Swift => "swift",
            CodeLanguage::Kotlin => "kotlin",
            CodeLanguage::Unknown => "unknown",
        }
    }
}

/// Detects the content type based on file extension
pub fn detect_content_type(file_path: &str) -> ContentType {
    let extension = Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match extension.to_lowercase().as_str() {
        // Markdown
        "md" | "markdown" => ContentType::Markdown,

        // Rust
        "rs" => ContentType::Code(CodeLanguage::Rust),

        // Python
        "py" | "pyw" | "pyi" => ContentType::Code(CodeLanguage::Python),

        // JavaScript
        "js" | "mjs" | "cjs" | "jsx" => ContentType::Code(CodeLanguage::JavaScript),

        // TypeScript
        "ts" | "tsx" | "mts" | "cts" => ContentType::Code(CodeLanguage::TypeScript),

        // Go
        "go" => ContentType::Code(CodeLanguage::Go),

        // Java
        "java" => ContentType::Code(CodeLanguage::Java),

        // C
        "c" | "h" => ContentType::Code(CodeLanguage::C),

        // C++
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => ContentType::Code(CodeLanguage::Cpp),

        // Ruby
        "rb" | "rake" => ContentType::Code(CodeLanguage::Ruby),

        // PHP
        "php" | "phtml" => ContentType::Code(CodeLanguage::Php),

        // Swift
        "swift" => ContentType::Code(CodeLanguage::Swift),

        // Kotlin
        "kt" | "kts" => ContentType::Code(CodeLanguage::Kotlin),

        // Unknown - treat as plain text
        _ => ContentType::PlainText,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_markdown() {
        assert_eq!(detect_content_type("README.md"), ContentType::Markdown);
        assert_eq!(
            detect_content_type("/path/to/doc.markdown"),
            ContentType::Markdown
        );
    }

    #[test]
    fn test_detect_rust() {
        assert_eq!(
            detect_content_type("main.rs"),
            ContentType::Code(CodeLanguage::Rust)
        );
    }

    #[test]
    fn test_detect_python() {
        assert_eq!(
            detect_content_type("script.py"),
            ContentType::Code(CodeLanguage::Python)
        );
    }

    #[test]
    fn test_detect_javascript() {
        assert_eq!(
            detect_content_type("app.js"),
            ContentType::Code(CodeLanguage::JavaScript)
        );
        assert_eq!(
            detect_content_type("module.mjs"),
            ContentType::Code(CodeLanguage::JavaScript)
        );
    }

    #[test]
    fn test_detect_typescript() {
        assert_eq!(
            detect_content_type("component.tsx"),
            ContentType::Code(CodeLanguage::TypeScript)
        );
    }

    #[test]
    fn test_detect_plain_text() {
        assert_eq!(detect_content_type("file.txt"), ContentType::PlainText);
        assert_eq!(detect_content_type("no_extension"), ContentType::PlainText);
    }
}
