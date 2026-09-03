// SPDX-License-Identifier: Apache-2.0
//! Directory source connector — walk a local filesystem tree and collect
//! indexable files (md, txt, pdf) with size and symlink guards.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use sha2::{Digest, Sha256};

use crate::quality_gate::QualityGate;
use crate::sources::obsidian::note_to_documents;
use crate::sources::RawDocument;
use crate::tuning::GateConfig;

/// Maximum file size for text/markdown files (1 MB).
const MAX_TEXT_SIZE: u64 = 1024 * 1024;

/// Maximum file size for PDF files (10 MB).
const MAX_PDF_SIZE: u64 = 10 * 1024 * 1024;

/// Minimum extracted text required before a parsed file is worth ingesting.
/// Guards against image-only / garbage PDFs (no OCR in v1) and near-empty files.
const MIN_EXTRACTED_WORDS: usize = 5;
const MIN_EXTRACTED_NON_WS_CHARS: usize = 20;

/// Directories to skip during traversal (relative path components).
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".obsidian",
    "node_modules",
    ".cache",
    ".venv",
    "venv",
];

/// Scan a directory tree and return all indexable files (md, txt, pdf).
///
/// # Behavior
/// - Walks recursively, following directory structure
/// - Does NOT follow symlinks (prevents cycles)
/// - Skips hidden directories and files (leading '.')
/// - Skips known non-indexable directories (`.git`, `.obsidian`, etc.)
/// - Filters to extensions: `.md`, `.txt`, `.pdf` only
/// - Applies per-file size caps (1MB for text/md, 10MB for pdf)
/// - Single-file root: if root is a file, yields exactly it (no validation)
///
/// # Returns
/// A vector of `PathBuf` to all indexable files found, sorted.
pub fn scan_directory(root: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();

    // If root is a file, yield it as-is.
    if root.is_file() {
        results.push(root.to_path_buf());
        return results;
    }

    // If root is not a directory, return empty.
    if !root.is_dir() {
        return results;
    }

    walk_recursive(root, &mut results);
    results.sort();
    results
}

/// Recursively walk a directory, collecting indexable files.
fn walk_recursive(dir: &Path, results: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        // Skip hidden files/directories (leading '.').
        if let Some(name) = path.file_name() {
            if let Some(name_str) = name.to_str() {
                if name_str.starts_with('.') {
                    continue;
                }
            }
        }

        // Skip symlinks entirely — they don't follow links at all.
        if fs::symlink_metadata(&path)
            .map(|m| m.is_symlink())
            .unwrap_or(false)
        {
            continue;
        }

        // If it's a directory, recurse (but skip known non-indexable dirs).
        if path.is_dir() {
            if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
                if SKIP_DIRS.contains(&dir_name) {
                    continue;
                }
            }
            walk_recursive(&path, results);
        }

        // If it's a file, check if it's indexable.
        if path.is_file() && is_indexable(&path) {
            results.push(path);
        }
    }
}

/// Check if a file is indexable: right extension and within size limits.
fn is_indexable(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let size_limit = match ext.as_str() {
        "md" | "txt" => MAX_TEXT_SIZE,
        "pdf" => MAX_PDF_SIZE,
        _ => return false,
    };

    // Check file size.
    match fs::metadata(path) {
        Ok(m) => m.len() <= size_limit,
        Err(_) => false,
    }
}

/// Guard against scanning the reserved pages directory.
///
/// Checks if the given path is the pages directory (~/.wenlan/pages),
/// which should never be registered as an ingest source (prevents recursion).
pub fn is_reserved_ingest_root(path: &Path, knowledge_path: &Path) -> bool {
    // Normalize both paths to check for exact match.
    if let (Ok(p1), Ok(p2)) = (path.canonicalize(), knowledge_path.canonicalize()) {
        p1 == p2
    } else {
        // If canonicalization fails, do a simple comparison.
        path == knowledge_path
    }
}

/// Per-file parse result for directory ingest. One bad file maps to a single
/// `Skipped` or `Error` outcome so a folder sync can continue with the rest.
#[derive(Debug)]
pub enum FileOutcome {
    /// File parsed into one or more provenance-stamped documents.
    Ingested(Vec<RawDocument>),
    /// File was in-scope but produced nothing worth ingesting (image-only PDF,
    /// too-short text, quality-gate rejection). Carries a human reason.
    Skipped(String),
    /// File could not be parsed (read failure, malformed PDF). Carries detail.
    Error(String),
}

/// Convert an in-scope file into folder-provenance-stamped `RawDocument`s.
///
/// Dispatch by extension: `.md` goes through the Obsidian note parser
/// (frontmatter + wikilinks, safe on plain markdown); `.txt` is read and
/// decoded (non-UTF-8 via BOM/heuristic detection); `.pdf` is text-extracted.
/// Every produced document is stamped `source="memory"`, `source_agent="folder"`,
/// `source_id="{source_id}::{path}"`, and the file's SHA-256 `content_hash`.
///
/// This is a **synchronous, CPU-bound** helper. PDF extraction in particular can
/// be heavy; callers on an async request path MUST run it inside
/// `spawn_blocking` — never inline on an HTTP handler.
///
/// One bad file never aborts: a read/parse failure returns `Error`, an
/// out-of-scope or empty file returns `Skipped`, and a panicking PDF parser is
/// caught and mapped to `Error`. This function does not panic on bad input.
pub fn file_to_documents(
    source_id: &str,
    path: &Path,
    knowledge_path: Option<&Path>,
) -> FileOutcome {
    let ext = match file_extension(path) {
        Some(ext) => ext,
        None => return FileOutcome::Skipped("unsupported file (no extension)".to_string()),
    };

    let size_limit = match ext.as_str() {
        "md" | "txt" => MAX_TEXT_SIZE,
        "pdf" => MAX_PDF_SIZE,
        _ => return FileOutcome::Skipped(format!("unsupported file extension: .{ext}")),
    };

    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(err) => return FileOutcome::Error(format!("metadata read failed: {err}")),
    };
    if !metadata.is_file() {
        return FileOutcome::Skipped("path is not a regular file".to_string());
    }
    if metadata.len() > size_limit {
        return FileOutcome::Skipped(format!(
            "oversized .{ext} file: {} bytes exceeds {size_limit} bytes",
            metadata.len()
        ));
    }

    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) => return FileOutcome::Error(format!("read failed: {err}")),
    };

    let content_hash = sha256_hex(&bytes);
    let mtime = modified_unix_seconds(&metadata);
    let provenance = provenance_path(path, knowledge_path);

    let docs = match ext.as_str() {
        "md" => {
            let content = decode_text_bytes(&bytes);
            // note_to_documents handles frontmatter + wikilinks additively; on
            // plain (non-vault) markdown it just yields the body as one doc.
            note_to_documents(source_id, Path::new(&provenance), &content, mtime)
        }
        "txt" => {
            let content = decode_text_bytes(&bytes);
            vec![raw_file_document(
                source_id,
                &provenance,
                path,
                &ext,
                content,
                mtime,
            )]
        }
        "pdf" => match extract_pdf_text(&bytes) {
            Ok(content) => vec![raw_file_document(
                source_id,
                &provenance,
                path,
                &ext,
                content,
                mtime,
            )],
            Err(detail) => return FileOutcome::Error(detail),
        },
        _ => unreachable!("extension filtered by size_limit match above"),
    };

    finalize_file_documents(docs, &content_hash, &ext)
}

/// Stamp folder provenance onto every doc, then admit each through the
/// min-text heuristic + quality gate. Returns `Ingested` if any doc survives,
/// otherwise `Skipped` with the collected rejection reasons.
fn finalize_file_documents(
    docs: Vec<RawDocument>,
    content_hash: &str,
    extension: &str,
) -> FileOutcome {
    if docs.is_empty() {
        return FileOutcome::Skipped(format!(".{extension} produced no ingestable content"));
    }

    let gate = QualityGate::new(GateConfig::default());
    let mut admitted = Vec::new();
    let mut rejected = Vec::new();

    for mut doc in docs {
        doc.source = "memory".to_string();
        doc.source_agent = Some("folder".to_string());
        doc.content_hash = Some(content_hash.to_string());
        doc.metadata
            .entry("extension".to_string())
            .or_insert_with(|| extension.to_string());

        // Min-text heuristic first: catches image-only / garbage PDFs before the
        // quality gate so the skip reason is specific (no OCR in v1).
        if let Some(detail) = min_text_rejection_detail(&doc.content) {
            rejected.push(detail);
            continue;
        }

        let result = gate.check_content(&doc.content);
        if result.admitted {
            admitted.push(doc);
        } else {
            let detail = result
                .reason
                .map(|reason| reason.detail())
                .unwrap_or_else(|| "rejected by quality gate".to_string());
            rejected.push(detail);
        }
    }

    if admitted.is_empty() {
        let detail = if rejected.is_empty() {
            "no documents admitted".to_string()
        } else {
            rejected.join("; ")
        };
        FileOutcome::Skipped(detail)
    } else {
        FileOutcome::Ingested(admitted)
    }
}

/// Build a single `RawDocument` from a plain text/pdf file's extracted content.
/// Folder provenance (source/source_agent/content_hash) is stamped later in
/// `finalize_file_documents`, shared with the markdown path.
fn raw_file_document(
    source_id: &str,
    provenance: &str,
    path: &Path,
    extension: &str,
    content: String,
    mtime: i64,
) -> RawDocument {
    let mut metadata = HashMap::new();
    metadata.insert("extension".to_string(), extension.to_string());
    metadata.insert("path".to_string(), provenance.to_string());

    RawDocument {
        source: "memory".to_string(),
        source_id: format!("{source_id}::{provenance}"),
        title: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("untitled")
            .to_string(),
        content,
        last_modified: mtime,
        metadata,
        source_agent: Some("folder".to_string()),
        ..Default::default()
    }
}

/// Lowercased file extension, or `None` when the path has no extension.
fn file_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
}

/// Provenance path: relative to `knowledge_path` when the file lives under it,
/// otherwise the full path. Used in `source_id` and metadata.
///
/// `pub(crate)` so `document_enrichment` can recompute the canonical document
/// `source_id` (`{source_id}::{provenance}`) WITHOUT re-parsing the file on a
/// resumed run — the same mapping `file_to_documents` stamps at parse time.
pub(crate) fn provenance_path(path: &Path, knowledge_path: Option<&Path>) -> String {
    if let Some(root) = knowledge_path {
        if let Ok(relative) = path.strip_prefix(root) {
            return relative.to_string_lossy().to_string();
        }
    }
    path.to_string_lossy().to_string()
}

/// The canonical document `source_id` under which a file's chunks live in
/// `memories`: `{source_id}::{provenance}`.
///
/// This is the ONE authority for that key derivation. The enrichment worker
/// (write side, `document_enrichment::run_document_enrichment`) and folder-sync
/// deletion propagation (delete side, the daemon's `handle_sync_source`) both
/// call it, so the write-vs-delete key can never drift out of sync -- deletion
/// must target exactly the id the write side stamped.
pub fn document_source_id(
    source_id: &str,
    file_path: &Path,
    knowledge_path: Option<&Path>,
) -> String {
    format!(
        "{}::{}",
        source_id,
        provenance_path(file_path, knowledge_path)
    )
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn modified_unix_seconds(metadata: &fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|mtime| mtime.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

/// Extract text from PDF bytes, isolating any parser panic as an `Error`.
///
/// `pdf_extract` (via `lopdf`) can panic on malformed/truncated input; wrapping
/// in `catch_unwind` upholds the "one bad file never aborts" contract. This is
/// a sync CPU function — callers on async paths wrap it in `spawn_blocking`.
///
/// That guard covers panics only, and only panics. A stack overflow aborts the
/// process instead of unwinding, so it escapes `catch_unwind` entirely — which
/// is exactly RUSTSEC-2026-0187, where a ~21 KB PDF holding a 10,000-deep
/// nested array `SIGABRT`s any caller of `lopdf`. The lopdf `>= 0.42` floor is
/// what fixes that, not this wrapper. Keep the depth limit in the dependency,
/// and do not read this `catch_unwind` as making arbitrary PDF input safe.
pub fn extract_pdf_text(bytes: &[u8]) -> Result<String, String> {
    let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text_from_mem(bytes)
    }));
    match parsed {
        Ok(Ok(text)) => Ok(normalize_extracted_text(&text)),
        Ok(Err(err)) => Err(format!("pdf parse failed: {err}")),
        Err(_) => Err("pdf parse failed: parser panicked on malformed input".to_string()),
    }
}

/// Decode raw file bytes to a `String`, tolerating non-UTF-8 encodings.
/// BOM detection first (UTF-8/16/32), then valid-UTF-8 fast path, then a
/// UTF-16 heuristic (null-byte density), finally a Windows-1252 (latin1) decode
/// which is lossless for single-byte inputs.
fn decode_text_bytes(bytes: &[u8]) -> String {
    if let Some((encoding, bom_len)) = encoding_rs::Encoding::for_bom(bytes) {
        let (text, _, _) = encoding.decode(&bytes[bom_len..]);
        return text.into_owned();
    }

    if let Ok(text) = std::str::from_utf8(bytes) {
        return text.to_string();
    }

    if looks_like_utf16_le(bytes) {
        let (text, _, _) = encoding_rs::UTF_16LE.decode(bytes);
        return text.into_owned();
    }
    if looks_like_utf16_be(bytes) {
        let (text, _, _) = encoding_rs::UTF_16BE.decode(bytes);
        return text.into_owned();
    }

    let (text, _, _) = encoding_rs::WINDOWS_1252.decode(bytes);
    text.into_owned()
}

fn looks_like_utf16_le(bytes: &[u8]) -> bool {
    looks_like_utf16(bytes, 1)
}

fn looks_like_utf16_be(bytes: &[u8]) -> bool {
    looks_like_utf16(bytes, 0)
}

/// Heuristic: BOM-less UTF-16 text has a null byte in most code-unit halves
/// (the high byte of Latin/ASCII characters). `zero_offset` selects the half
/// (0 = big-endian high byte first, 1 = little-endian).
fn looks_like_utf16(bytes: &[u8], zero_offset: usize) -> bool {
    let sample_len = bytes.len().min(64);
    if sample_len < 4 {
        return false;
    }
    let pairs = sample_len / 2;
    let zero_count = (0..pairs)
        .filter(|i| bytes[i * 2 + zero_offset] == 0)
        .count();
    zero_count * 2 >= pairs
}

/// Collapse extracted PDF whitespace runs into single spaces so downstream
/// word/char heuristics and chunking see clean text.
fn normalize_extracted_text(content: &str) -> String {
    content.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Reject content with too little real text (image-only/garbage PDFs, near-empty
/// files). Returns a human reason when below floor, `None` when it passes.
fn min_text_rejection_detail(content: &str) -> Option<String> {
    let words = content
        .split_whitespace()
        .filter(|word| word.chars().any(|c| c.is_alphanumeric()))
        .count();
    let non_ws_chars = content.chars().filter(|c| !c.is_whitespace()).count();

    if words == 0 || non_ws_chars == 0 {
        return Some("no extractable text (no OCR in v1)".to_string());
    }
    if words < MIN_EXTRACTED_WORDS || non_ws_chars < MIN_EXTRACTED_NON_WS_CHARS {
        return Some(format!(
            "extracted text too short: {words} words, {non_ws_chars} non-whitespace chars"
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    const MAX_TEXT_SIZE: u64 = 1024 * 1024;
    const MAX_PDF_SIZE: u64 = 10 * 1024 * 1024;

    #[test]
    fn test_scan_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_scan_single_file() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("test.md");
        fs::File::create(&file_path)
            .unwrap()
            .write_all(b"# Test")
            .unwrap();

        let results = scan_directory(&file_path);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], file_path);
    }

    #[test]
    fn test_scan_filters_extensions() {
        let tmp = TempDir::new().unwrap();

        create_file(tmp.path(), "test.md", b"# Markdown");
        create_file(tmp.path(), "test.txt", b"Text content");
        create_file(tmp.path(), "test.pdf", b"PDF");
        create_file(tmp.path(), "test.png", b"PNG");
        create_file(tmp.path(), "test.doc", b"DOC");

        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 3);

        let names: Vec<_> = results
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
            .collect();
        assert!(names.contains(&"test.md"));
        assert!(names.contains(&"test.txt"));
        assert!(names.contains(&"test.pdf"));
        assert!(!names.contains(&"test.png"));
        assert!(!names.contains(&"test.doc"));
    }

    #[test]
    fn test_scan_skips_hidden_files() {
        let tmp = TempDir::new().unwrap();

        create_file(tmp.path(), "visible.md", b"# Visible");
        create_file(tmp.path(), ".hidden.md", b"# Hidden");

        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 1);
        assert!(results[0].file_name().unwrap().to_str().unwrap() == "visible.md");
    }

    #[test]
    fn test_scan_skips_hidden_directories() {
        let tmp = TempDir::new().unwrap();

        fs::create_dir(tmp.path().join("visible")).unwrap();
        fs::create_dir(tmp.path().join(".hidden")).unwrap();

        create_file(&tmp.path().join("visible"), "file.md", b"# Visible");
        create_file(&tmp.path().join(".hidden"), "file.md", b"# Hidden");

        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 1);
        assert!(results[0].components().any(|c| c.as_os_str() == "visible"));
    }

    #[test]
    fn test_scan_skips_known_dirs() {
        let tmp = TempDir::new().unwrap();

        fs::create_dir(tmp.path().join(".git")).unwrap();
        fs::create_dir(tmp.path().join(".obsidian")).unwrap();
        fs::create_dir(tmp.path().join("node_modules")).unwrap();

        create_file(&tmp.path().join(".git"), "file.md", b"# Git");
        create_file(&tmp.path().join(".obsidian"), "file.md", b"# Obsidian");
        create_file(
            &tmp.path().join("node_modules"),
            "file.md",
            b"# NodeModules",
        );

        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_scan_respects_text_size_limit() {
        let tmp = TempDir::new().unwrap();

        let ok_content = vec![b'a'; (MAX_TEXT_SIZE - 1) as usize];
        let mut ok_file = fs::File::create(tmp.path().join("ok.txt")).unwrap();
        ok_file.write_all(&ok_content).unwrap();

        let oversized_content = vec![b'b'; (MAX_TEXT_SIZE + 1) as usize];
        let mut oversized_file = fs::File::create(tmp.path().join("oversized.txt")).unwrap();
        oversized_file.write_all(&oversized_content).unwrap();

        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 1);
        assert!(results[0].file_name().unwrap().to_str().unwrap() == "ok.txt");
    }

    #[test]
    fn test_scan_respects_pdf_size_limit() {
        let tmp = TempDir::new().unwrap();

        let ok_content = vec![b'a'; (MAX_PDF_SIZE - 1) as usize];
        let mut ok_file = fs::File::create(tmp.path().join("ok.pdf")).unwrap();
        ok_file.write_all(&ok_content).unwrap();

        let oversized_content = vec![b'b'; (MAX_PDF_SIZE + 1) as usize];
        let mut oversized_file = fs::File::create(tmp.path().join("oversized.pdf")).unwrap();
        oversized_file.write_all(&oversized_content).unwrap();

        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 1);
        assert!(results[0].file_name().unwrap().to_str().unwrap() == "ok.pdf");
    }

    #[test]
    fn test_scan_recursive() {
        let tmp = TempDir::new().unwrap();

        fs::create_dir(tmp.path().join("subdir")).unwrap();
        fs::create_dir(tmp.path().join("subdir/nested")).unwrap();

        create_file(tmp.path(), "root.md", b"# Root");
        create_file(&tmp.path().join("subdir"), "sub.md", b"# Sub");
        create_file(&tmp.path().join("subdir/nested"), "nested.md", b"# Nested");

        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_scan_ignores_symlink_cycles() {
        let tmp = TempDir::new().unwrap();

        create_file(tmp.path(), "file.md", b"# File");

        let symlink_path = tmp.path().join("cycle_link");
        #[cfg(unix)]
        {
            use std::os::unix::fs as unix_fs;
            let _ = unix_fs::symlink(tmp.path(), &symlink_path);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs as win_fs;
            let _ = win_fs::symlink_dir(tmp.path(), &symlink_path);
        }

        let results = scan_directory(tmp.path());
        assert_eq!(results.len(), 1);
        assert!(results[0].file_name().unwrap().to_str().unwrap() == "file.md");
    }

    #[test]
    fn test_scan_ignores_file_symlinks() {
        let tmp = TempDir::new().unwrap();
        let external_tmp = TempDir::new().unwrap();

        // Create a file OUTSIDE the scan root
        let external_file = external_tmp.path().join("external.md");
        fs::File::create(&external_file)
            .unwrap()
            .write_all(b"# External")
            .unwrap();

        // Create a symlink INSIDE the scan root pointing to the external file
        let symlink_path = tmp.path().join("symlink.md");
        #[cfg(unix)]
        {
            use std::os::unix::fs as unix_fs;
            let _ = unix_fs::symlink(&external_file, &symlink_path);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs as win_fs;
            let _ = win_fs::symlink_file(&external_file, &symlink_path);
        }

        // Scan should NOT include the symlink to the external file
        let results = scan_directory(tmp.path());
        assert_eq!(
            results.len(),
            0,
            "symlink to external file should not be scanned"
        );
    }

    #[test]
    fn test_is_reserved_ingest_root_true() {
        let tmp = TempDir::new().unwrap();
        let pages_path = tmp.path().join("pages");
        fs::create_dir(&pages_path).unwrap();

        assert!(is_reserved_ingest_root(&pages_path, &pages_path));
    }

    #[test]
    fn test_is_reserved_ingest_root_false() {
        let tmp = TempDir::new().unwrap();
        let pages_path = tmp.path().join("pages");
        let other_path = tmp.path().join("other");

        fs::create_dir(&pages_path).unwrap();
        fs::create_dir(&other_path).unwrap();

        assert!(!is_reserved_ingest_root(&other_path, &pages_path));
    }

    #[test]
    fn file_to_documents_markdown_stamps_folder_provenance_and_hash() {
        let tmp = TempDir::new().unwrap();
        let path = create_file(
            tmp.path(),
            "linked.md",
            b"---\ntags:\n  - project\n---\n\n# Linked\n\nThis note points at [[Other Note]] and contains enough useful text for ingestion.",
        );

        let outcome = file_to_documents("src1", &path, Some(tmp.path()));
        let docs = match outcome {
            FileOutcome::Ingested(docs) => docs,
            other => panic!("expected markdown ingest, got {other:?}"),
        };

        assert!(!docs.is_empty());
        let hash = docs[0].content_hash.clone().expect("content_hash");
        assert_eq!(hash.len(), 64);
        for doc in docs {
            assert_eq!(doc.source, "memory");
            assert_eq!(doc.source_agent.as_deref(), Some("folder"));
            assert_eq!(doc.content_hash.as_deref(), Some(hash.as_str()));
            assert!(doc.source_id.starts_with("src1::"));
            assert!(doc.source_id.contains("linked.md"));
            assert_eq!(
                doc.metadata.get("extension").map(String::as_str),
                Some("md")
            );
        }
    }

    #[test]
    fn file_to_documents_text_decodes_utf16_and_latin1_without_loss() {
        let tmp = TempDir::new().unwrap();
        let utf16_path = tmp.path().join("utf16.txt");
        let mut utf16_bytes = vec![0xFF, 0xFE];
        for unit in "Café résumé from UTF-16 text with enough meaningful words for ingestion."
            .encode_utf16()
        {
            utf16_bytes.extend_from_slice(&unit.to_le_bytes());
        }
        fs::write(&utf16_path, utf16_bytes).unwrap();

        let latin1_path = create_file(
            tmp.path(),
            "latin1.txt",
            b"Caf\xe9 r\xe9sum\xe9 from latin1 text with enough meaningful words for ingestion.",
        );

        for path in [&utf16_path, &latin1_path] {
            let docs = match file_to_documents("src1", path, None) {
                FileOutcome::Ingested(docs) => docs,
                other => panic!("expected text ingest for {path:?}, got {other:?}"),
            };
            assert_eq!(docs.len(), 1);
            assert!(docs[0].content.contains("Café résumé"));
            assert_eq!(docs[0].source_agent.as_deref(), Some("folder"));
            assert!(docs[0].content_hash.is_some());
            assert_eq!(
                docs[0].metadata.get("extension").map(String::as_str),
                Some("txt")
            );
        }
    }

    #[test]
    fn file_to_documents_pdf_extracts_text_from_valid_tiny_pdf() {
        let tmp = TempDir::new().unwrap();
        let path = create_file(tmp.path(), "tiny.pdf", &tiny_text_pdf());

        let docs = match file_to_documents("src1", &path, None) {
            FileOutcome::Ingested(docs) => docs,
            other => panic!("expected pdf ingest, got {other:?}"),
        };

        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("Wenlan tiny PDF text"));
        assert_eq!(docs[0].source, "memory");
        assert_eq!(docs[0].source_agent.as_deref(), Some("folder"));
        assert!(docs[0].content_hash.is_some());
        assert_eq!(
            docs[0].metadata.get("extension").map(String::as_str),
            Some("pdf")
        );
    }

    #[test]
    fn file_to_documents_image_only_pdf_is_skipped_with_detail() {
        let tmp = TempDir::new().unwrap();
        let path = create_file(tmp.path(), "image-only.pdf", &image_only_pdf());

        match file_to_documents("src1", &path, None) {
            FileOutcome::Skipped(reason) => {
                assert!(reason.contains("no extractable text") || reason.contains("too short"));
            }
            other => panic!("expected skipped image-only pdf, got {other:?}"),
        }
    }

    #[test]
    fn file_to_documents_truncated_pdf_returns_error() {
        let tmp = TempDir::new().unwrap();
        let path = create_file(
            tmp.path(),
            "truncated.pdf",
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog",
        );

        match file_to_documents("src1", &path, None) {
            FileOutcome::Error(detail) => assert!(detail.contains("pdf")),
            other => panic!("expected pdf error, got {other:?}"),
        }
    }

    /// A valid one-page PDF whose content stream draws the given text via `Tj`.
    /// With `None`, the page has an empty content stream (no text operators) —
    /// the "image-only / no extractable text" case (no OCR in v1). Generated
    /// with lopdf so the xref/trailer are byte-correct and pdf_extract parses it.
    fn build_pdf(text: Option<&str>) -> Vec<u8> {
        use lopdf::content::{Content, Operation};
        use lopdf::{dictionary, Document, Object, Stream};

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();

        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });

        let operations = match text {
            Some(t) => vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 24.into()]),
                Operation::new("Td", vec![20.into(), 100.into()]),
                Operation::new("Tj", vec![Object::string_literal(t)]),
                Operation::new("ET", vec![]),
            ],
            None => Vec::new(),
        };
        let content = Content { operations };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 300.into(), 144.into()],
        });

        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        };
        doc.objects.insert(pages_id, Object::Dictionary(pages));

        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        let mut buf = Vec::new();
        doc.save_to(&mut buf).unwrap();
        buf
    }

    fn tiny_text_pdf() -> Vec<u8> {
        build_pdf(Some(
            "Wenlan tiny PDF text has enough useful words for folder ingestion.",
        ))
    }

    fn image_only_pdf() -> Vec<u8> {
        build_pdf(None)
    }

    fn create_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(content).unwrap();
        path
    }
}
