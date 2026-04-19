// SPDX-License-Identifier: AGPL-3.0-only
use crate::error::OriginError;
use crate::sources::{DataSource, RawDocument, SourceStatus};
use async_trait::async_trait;
use std::any::Any;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const DOCUMENT_EXTENSIONS: &[&str] = &["txt", "md", "csv", "log", "pdf", "docx", "rtf"];

const SOURCE_CODE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "tsx", "jsx", "json", "toml", "yaml", "yml", "go", "java", "c", "cpp",
    "h", "css", "html", "sh", "rb", "php", "swift", "kt", "scala",
];

const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".cache",
];

fn max_file_size(ext: &str) -> u64 {
    match ext {
        "pdf" | "docx" => 10_485_760, // 10 MB
        _ => 1_048_576,               // 1 MB
    }
}

#[derive(Default)]
pub struct LocalFilesSource {
    watch_paths: Vec<PathBuf>,
    last_sync: Option<i64>,
    document_count: u64,
}

impl LocalFilesSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_watch_path(&mut self, path: PathBuf) {
        if !self.watch_paths.contains(&path) {
            self.watch_paths.push(path);
        }
    }

    pub fn remove_watch_path(&mut self, path: &Path) {
        self.watch_paths.retain(|p| p != path);
    }

    /// Scan a directory recursively and collect all indexable files.
    pub fn scan_directory(dir: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let skip_dirs: HashSet<&str> = SKIP_DIRS.iter().copied().collect();

        Self::scan_recursive(dir, &skip_dirs, &mut files);
        files
    }

    fn scan_recursive(dir: &Path, skip_dirs: &HashSet<&str>, files: &mut Vec<PathBuf>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            // Skip dotfiles and excluded directories
            if name.starts_with('.') {
                continue;
            }

            if path.is_dir() {
                if !skip_dirs.contains(name.as_ref()) {
                    Self::scan_recursive(&path, skip_dirs, files);
                }
                continue;
            }

            // Check if the file is indexable
            if Self::is_indexable(&path) {
                files.push(path);
            }
        }
    }

    pub fn is_indexable(path: &Path) -> bool {
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_lowercase(),
            None => return false,
        };

        let supported = DOCUMENT_EXTENSIONS
            .iter()
            .chain(SOURCE_CODE_EXTENSIONS.iter())
            .any(|&e| e == ext);

        if !supported {
            return false;
        }

        // Check file size with format-aware limit
        match std::fs::metadata(path) {
            Ok(meta) => meta.len() <= max_file_size(&ext),
            Err(_) => false,
        }
    }

    /// Read a file and create a RawDocument, dispatching on extension for binary formats.
    pub fn read_file(path: &Path) -> Result<RawDocument, OriginError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        let content = match ext.as_str() {
            "pdf" => {
                let bytes = std::fs::read(path).map_err(|e| OriginError::Source {
                    source_name: "local_files".to_string(),
                    message: format!("Failed to read {}: {}", path.display(), e),
                })?;
                extract_pdf_text(&bytes, path)?
            }
            "docx" => {
                let bytes = std::fs::read(path).map_err(|e| OriginError::Source {
                    source_name: "local_files".to_string(),
                    message: format!("Failed to read {}: {}", path.display(), e),
                })?;
                extract_docx_text(&bytes, path)?
            }
            "rtf" => {
                let bytes = std::fs::read(path).map_err(|e| OriginError::Source {
                    source_name: "local_files".to_string(),
                    message: format!("Failed to read {}: {}", path.display(), e),
                })?;
                extract_rtf_text(&bytes)
            }
            _ => std::fs::read_to_string(path).map_err(|e| OriginError::Source {
                source_name: "local_files".to_string(),
                message: format!("Failed to read {}: {}", path.display(), e),
            })?,
        };

        if content.is_empty() {
            return Err(OriginError::Source {
                source_name: "local_files".to_string(),
                message: format!("No text content extracted from {}", path.display()),
            });
        }

        let metadata = std::fs::metadata(path)?;
        let last_modified = metadata
            .modified()
            .map(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64
            })
            .unwrap_or(0);

        let title = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());

        // Add file extension metadata for semantic chunking
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_string())
            .unwrap_or_default();

        let mut metadata = std::collections::HashMap::new();
        if !ext.is_empty() {
            metadata.insert("extension".to_string(), ext);
        }

        Ok(RawDocument {
            source: "local_files".to_string(),
            source_id: path.to_string_lossy().to_string(),
            title,
            summary: None,
            content,
            url: Some(format!("file://{}", path.display())),
            last_modified,
            metadata,
            memory_type: None,
            source_agent: None,
            domain: None,
            confidence: None,
            confirmed: None,
            supersedes: None,
            pending_revision: false,
            ..Default::default()
        })
    }
}

/// Extract text from a PDF using pdf-extract.
fn extract_pdf_text(bytes: &[u8], path: &Path) -> Result<String, OriginError> {
    match pdf_extract::extract_text_from_mem(bytes) {
        Ok(text) => {
            let trimmed = text.trim().to_string();
            if trimmed.is_empty() {
                log::warn!(
                    "PDF has no extractable text (image-only?): {}",
                    path.display()
                );
            }
            Ok(trimmed)
        }
        Err(e) => {
            log::warn!("Failed to extract text from PDF {}: {}", path.display(), e);
            Ok(String::new())
        }
    }
}

/// Extract text from a DOCX file (ZIP of XML).
fn extract_docx_text(bytes: &[u8], path: &Path) -> Result<String, OriginError> {
    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| OriginError::Source {
        source_name: "local_files".to_string(),
        message: format!("Failed to read DOCX {}: {}", path.display(), e),
    })?;

    let mut xml = String::new();
    {
        let mut file = archive
            .by_name("word/document.xml")
            .map_err(|e| OriginError::Source {
                source_name: "local_files".to_string(),
                message: format!("No document.xml in DOCX {}: {}", path.display(), e),
            })?;
        std::io::Read::read_to_string(&mut file, &mut xml).map_err(|e| OriginError::Source {
            source_name: "local_files".to_string(),
            message: format!("Failed to read document.xml from {}: {}", path.display(), e),
        })?;
    }

    // Split on paragraph boundaries and extract text from <w:t> tags
    let mut paragraphs = Vec::new();
    for para_xml in xml.split("</w:p>") {
        let mut para_text = String::new();
        for segment in para_xml.split("<w:t") {
            if let Some(gt_pos) = segment.find('>') {
                let after_gt = &segment[gt_pos + 1..];
                if let Some(end_pos) = after_gt.find("</w:t>") {
                    para_text.push_str(&after_gt[..end_pos]);
                }
            }
        }
        if !para_text.is_empty() {
            paragraphs.push(para_text);
        }
    }

    Ok(paragraphs.join("\n"))
}

/// Extract text from an RTF file using a simple state-machine stripper.
fn extract_rtf_text(bytes: &[u8]) -> String {
    let input = String::from_utf8_lossy(bytes);
    let mut result = String::new();
    let mut chars = input.chars().peekable();
    let mut depth: i32 = 0;
    let mut skip_depth: Option<i32> = None;

    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                depth += 1;
            }
            '}' => {
                if skip_depth == Some(depth) {
                    skip_depth = None;
                }
                depth -= 1;
                if depth < 0 {
                    depth = 0;
                }
            }
            _ if skip_depth.is_some() => continue,
            '\\' => {
                if let Some(&next) = chars.peek() {
                    if next.is_ascii_alphabetic() {
                        let mut word = String::new();
                        while let Some(&c) = chars.peek() {
                            if c.is_ascii_alphabetic() {
                                word.push(c);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        // Skip optional numeric parameter
                        if let Some(&c) = chars.peek() {
                            if c == '-' || c.is_ascii_digit() {
                                chars.next();
                                while let Some(&d) = chars.peek() {
                                    if d.is_ascii_digit() {
                                        chars.next();
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                        // Consume delimiter space
                        if chars.peek() == Some(&' ') {
                            chars.next();
                        }
                        // Handle known groups to skip
                        match word.as_str() {
                            "fonttbl" | "colortbl" | "stylesheet" | "info" | "pict" | "header"
                            | "footer" | "headerl" | "headerr" | "footerl" | "footerr" => {
                                skip_depth = Some(depth);
                            }
                            "par" | "line" => result.push('\n'),
                            "tab" => result.push('\t'),
                            _ => {}
                        }
                    } else if next == '\'' {
                        // Hex escape \'XX
                        chars.next();
                        let h1 = chars.next().unwrap_or('0');
                        let h2 = chars.next().unwrap_or('0');
                        if let Ok(byte) = u8::from_str_radix(&format!("{}{}", h1, h2), 16) {
                            if byte >= 0x20 {
                                result.push(byte as char);
                            }
                        }
                    } else if next == '\\' || next == '{' || next == '}' {
                        result.push(next);
                        chars.next();
                    } else {
                        chars.next();
                    }
                }
            }
            '\n' | '\r' => {}
            _ => {
                if depth >= 1 {
                    result.push(ch);
                }
            }
        }
    }

    result.trim().to_string()
}

#[async_trait]
impl DataSource for LocalFilesSource {
    fn name(&self) -> &str {
        "local_files"
    }

    fn requires_auth(&self) -> bool {
        false
    }

    async fn is_connected(&self) -> bool {
        !self.watch_paths.is_empty()
    }

    async fn connect(&mut self) -> Result<(), OriginError> {
        // Local files don't need authentication
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<(), OriginError> {
        self.watch_paths.clear();
        Ok(())
    }

    async fn fetch_updates(&mut self) -> Result<Vec<RawDocument>, OriginError> {
        // For local files, fetch_updates does a full scan
        // The file watcher handles incremental updates separately
        self.full_sync().await
    }

    async fn full_sync(&mut self) -> Result<Vec<RawDocument>, OriginError> {
        let mut docs = Vec::new();
        for watch_path in &self.watch_paths {
            let files = Self::scan_directory(watch_path);
            for file_path in files {
                match Self::read_file(&file_path) {
                    Ok(doc) => docs.push(doc),
                    Err(e) => {
                        log::warn!("Skipping file {}: {}", file_path.display(), e);
                    }
                }
            }
        }
        Ok(docs)
    }

    async fn status(&self) -> SourceStatus {
        SourceStatus {
            name: "local_files".to_string(),
            connected: !self.watch_paths.is_empty(),
            requires_auth: false,
            last_sync: self.last_sync,
            document_count: self.document_count,
            error: None,
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
