// SPDX-License-Identifier: Apache-2.0
//! `wenlan pages [query]` — browse distilled pages, or open one in your editor.
//!
//! Pure-local: reads the page markdown files the daemon writes under
//! `knowledge_path_or_default()` (default `~/.wenlan/pages/`) and hands the
//! match to the OS default `.md` app via `open`/`xdg-open`/`start`. No daemon
//! round-trip and no agent turn, so it stays instant and works offline. This
//! reads the user's own local page files, not daemon DB state, so it does not
//! cross the "CLI talks HTTP only" boundary.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;

use crate::output::{print_json, ResolvedFormat};

/// One distilled page on disk.
#[derive(Debug, Clone)]
pub struct PageEntry {
    pub path: PathBuf,
    pub title: String,
    pub page_id: Option<String>,
    pub modified: SystemTime,
}

/// Outcome of matching a query against the page list.
pub enum QueryMatch<'a> {
    None,
    One(&'a PageEntry),
    Many(Vec<&'a PageEntry>),
}

/// Pull the frontmatter `title:` (quotes stripped); fall back to the filename
/// stem. Only the leading `--- ... ---` block is scanned, so a `title:` line in
/// the body can't be mistaken for the page title.
fn extract_frontmatter_value(content: &str, key: &str) -> Option<String> {
    let mut in_fm = false;
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if i == 0 {
            if trimmed == "---" {
                in_fm = true;
                continue;
            }
            break; // no frontmatter -> use stem
        }
        if in_fm && trimmed == "---" {
            break; // end of frontmatter
        }
        if let Some(rest) = trimmed.strip_prefix(key) {
            let value = rest.trim().trim_matches('"').trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn extract_title(content: &str, fallback_stem: &str) -> String {
    extract_frontmatter_value(content, "title:").unwrap_or_else(|| fallback_stem.to_string())
}

/// Newest-first by mtime.
fn sort_newest_first(pages: &mut [PageEntry]) {
    pages.sort_by_key(|p| std::cmp::Reverse(p.modified));
}

/// Case-insensitive substring match against title OR filename stem.
fn match_query<'a>(pages: &'a [PageEntry], query: &str) -> QueryMatch<'a> {
    let q = query.to_lowercase();
    let exact_filename_hits: Vec<&PageEntry> = pages
        .iter()
        .filter(|p| {
            p.path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.to_lowercase() == q)
        })
        .collect();
    match exact_filename_hits.len() {
        1 => return QueryMatch::One(exact_filename_hits[0]),
        2.. => return QueryMatch::Many(exact_filename_hits),
        _ => {}
    }

    let hits: Vec<&PageEntry> = pages
        .iter()
        .filter(|p| {
            p.title.to_lowercase().contains(&q)
                || p.path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_lowercase().contains(&q))
                    .unwrap_or(false)
        })
        .collect();
    match hits.len() {
        0 => QueryMatch::None,
        1 => QueryMatch::One(hits[0]),
        _ => QueryMatch::Many(hits),
    }
}

/// Read every top-level `*.md` under `dir` into a newest-first list. A missing
/// dir yields an empty list (no pages distilled yet). The `.wenlan/` state dir
/// holds no `*.md`, so it's skipped by the extension filter.
fn read_pages(dir: &Path) -> Vec<PageEntry> {
    let mut entries = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return entries;
    };
    for e in rd.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("md") {
            continue;
        }
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let title = extract_title(&content, &stem);
        let page_id = extract_frontmatter_value(&content, "origin_id:");
        let modified = e
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        entries.push(PageEntry {
            path,
            title,
            page_id,
            modified,
        });
    }
    sort_newest_first(&mut entries);
    entries
}

/// Open a path in the OS default app for its type. Returns whether a launcher
/// actually ran (false -> caller prints the path so the user opens it).
fn open_in_editor(path: &Path) -> bool {
    use std::process::Command;
    let status = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(path)
            .status()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(path).status()
    } else {
        Command::new("xdg-open").arg(path).status()
    };
    status.map(|s| s.success()).unwrap_or(false)
}

/// Collapse same-title pages into `(title, count)`, newest-first by first
/// occurrence — so a topic distilled into many revisions shows as one row.
fn collapse_by_title(pages: &[PageEntry]) -> Vec<(String, usize)> {
    let mut order: Vec<String> = Vec::new();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for p in pages {
        if !counts.contains_key(p.title.as_str()) {
            order.push(p.title.clone());
        }
        *counts.entry(p.title.as_str()).or_insert(0) += 1;
    }
    order
        .iter()
        .map(|t| (t.clone(), counts[t.as_str()]))
        .collect()
}

pub fn run(
    format: ResolvedFormat,
    quiet: bool,
    query: Option<String>,
    limit: usize,
    resolve_id: bool,
) -> Result<()> {
    // An empty/whitespace query (e.g. a skill passing "" for no-arg) means "list".
    let query = query.filter(|q| !q.trim().is_empty());
    let dir = wenlan_core::config::load_config().knowledge_path_or_default();
    let pages = read_pages(&dir);

    match query {
        // No arg -> list distinct topics (same-title revisions collapsed),
        // newest-first, capped at `limit` topics (0 = all).
        None => {
            if quiet {
                return Ok(());
            }
            let total_pages = pages.len();
            let groups = collapse_by_title(&pages);
            let total_titles = groups.len();
            let shown: &[(String, usize)] = if limit == 0 {
                &groups
            } else {
                &groups[..total_titles.min(limit)]
            };
            match format {
                ResolvedFormat::Json => {
                    let items: Vec<_> = shown
                        .iter()
                        .map(|(title, count)| serde_json::json!({ "title": title, "count": count }))
                        .collect();
                    print_json(&serde_json::json!({
                        "pages": items,
                        "total_pages": total_pages,
                        "total_titles": total_titles,
                    }))?;
                }
                ResolvedFormat::Table => print_list(shown, total_titles, total_pages, &dir),
            }
        }
        // Query -> open the match.
        Some(q) => match match_query(&pages, &q) {
            QueryMatch::None => {
                eprintln!("no page matches: {q}");
                eprintln!("run `wenlan pages` to list, or `/distill {q}` to synthesize one");
            }
            QueryMatch::One(p) => {
                if resolve_id {
                    let page_id = p.page_id.as_deref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "page projection {} has no origin_id frontmatter",
                            p.path.display()
                        )
                    })?;
                    if !quiet {
                        println!("{page_id}");
                    }
                    return Ok(());
                }
                let launched = open_in_editor(&p.path);
                if !quiet {
                    if launched {
                        println!("Opened {}", p.path.display());
                    } else {
                        // Headless / no handler: print the path to open by hand.
                        println!("{}", p.path.display());
                    }
                }
            }
            QueryMatch::Many(hits) => {
                if !quiet {
                    // Titles can collide (many distilled revisions share a name),
                    // so show the filename too — that's the unique handle to refine on.
                    println!(
                        "{} matches for \"{q}\" — refine with a filename:",
                        hits.len()
                    );
                    for p in &hits {
                        let stem = p.path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                        println!("  {}  ·  {}", p.title, stem);
                    }
                }
            }
        },
    }
    Ok(())
}

/// Render a page's per-claim citations for terminal display: verified
/// citations show plain, unverified ones carry the exact spec-mandated badge
/// text (never "false" — see spec §5). Empty input renders empty (no
/// dangling "Citations:" header for uncited pages).
///
/// Reference implementation: `wenlan pages` currently reads pages from local
/// markdown files (no daemon round-trip, by design — see module doc), so
/// citation data (which lives on the DB-backed `Page`/`PageCitation` served
/// via `GET /api/pages/{id}`) isn't wired into this command's print path yet.
/// This helper demonstrates the rendering contract for that future caller.
#[allow(dead_code)] // reference implementation — not yet wired to a live fetch path, see doc above
fn render_citations(citations: &[wenlan_types::pages::PageCitation]) -> String {
    if citations.is_empty() {
        return String::new();
    }
    let mut out = String::from("\nCitations:\n");
    for c in citations {
        if c.status == "verified" {
            out.push_str(&format!("  [{}] {}\n", c.marker, c.locator));
        } else {
            out.push_str(&format!(
                "  [{}] {} — ⚠ not directly traceable to a source\n",
                c.marker, c.locator
            ));
        }
    }
    out
}

fn print_list(shown: &[(String, usize)], total_titles: usize, total_pages: usize, dir: &Path) {
    if total_pages == 0 {
        println!("(no pages in {})", dir.display());
        println!(
            "  Pages are written by /distill in the Claude Code plugin once 3 or more related memories exist in a space."
        );
        return;
    }
    if shown.len() < total_titles {
        println!(
            "{total_pages} pages, {total_titles} topics (newest {} shown — `--limit 0` for all):",
            shown.len()
        );
    } else {
        println!(
            "{total_pages} pages, {total_titles} topic{}:",
            if total_titles == 1 { "" } else { "s" }
        );
    }
    for (title, count) in shown {
        if *count > 1 {
            println!("  {title}  (×{count})");
        } else {
            println!("  {title}");
        }
    }
    println!("open one: wenlan pages <title-or-filename>");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    fn mk(title: &str, file: &str, secs: u64) -> PageEntry {
        PageEntry {
            path: PathBuf::from(file),
            title: title.into(),
            page_id: None,
            modified: UNIX_EPOCH + Duration::from_secs(secs),
        }
    }

    #[test]
    fn title_from_quoted_frontmatter() {
        let md = "---\ntitle: \"Rust Systems Language\"\norigin_id: x\n---\nbody\n";
        assert_eq!(extract_title(md, "stem"), "Rust Systems Language");
    }

    #[test]
    fn title_unquoted() {
        let md = "---\ntitle: Plain Title\n---\n";
        assert_eq!(extract_title(md, "stem"), "Plain Title");
    }

    #[test]
    fn title_falls_back_to_stem_when_no_frontmatter() {
        // a `title:` in the body must NOT be picked up
        let md = "no frontmatter\ntitle: body-title-should-not-match\n";
        assert_eq!(extract_title(md, "my-file"), "my-file");
    }

    #[test]
    fn title_falls_back_when_empty() {
        let md = "---\ntitle:\n---\n";
        assert_eq!(extract_title(md, "stem"), "stem");
    }

    #[test]
    fn query_unique_match_by_title() {
        let pages = vec![
            mk("Rust Systems", "a.md", 1),
            mk("Cooking Pasta", "b.md", 2),
        ];
        match match_query(&pages, "RUST") {
            QueryMatch::One(p) => assert_eq!(p.title, "Rust Systems"),
            _ => panic!("expected one"),
        }
    }

    #[test]
    fn query_multiple_matches() {
        let pages = vec![mk("Rust Systems", "a.md", 1), mk("Rust Async", "b.md", 2)];
        match match_query(&pages, "rust") {
            QueryMatch::Many(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected many"),
        }
    }

    #[test]
    fn query_no_match() {
        let pages = vec![mk("Rust", "a.md", 1)];
        assert!(matches!(match_query(&pages, "zzz"), QueryMatch::None));
    }

    #[test]
    fn query_matches_filename_stem() {
        let pages = vec![mk("Some Title", "2026-03-18-notes.md", 1)];
        match match_query(&pages, "2026-03") {
            QueryMatch::One(p) => assert_eq!(p.title, "Some Title"),
            _ => panic!("expected one (stem match)"),
        }
    }

    #[test]
    fn exact_filename_stem_wins_over_broader_substring_matches() {
        let pages = vec![
            mk("Atlas", "atlas.md", 1),
            mk("Atlas Notes", "atlas-notes.md", 2),
        ];
        match match_query(&pages, "atlas") {
            QueryMatch::One(p) => assert_eq!(p.path, PathBuf::from("atlas.md")),
            _ => panic!("expected the exact filename stem"),
        }
    }

    #[test]
    fn collapse_groups_same_title_newest_first() {
        // newest-first input; "Rust" appears non-contiguously -> still one row, count 3
        let pages = vec![
            mk("Rust", "r3.md", 9),
            mk("Rust", "r2.md", 8),
            mk("Cooking", "c.md", 7),
            mk("Rust", "r1.md", 6),
        ];
        let g = collapse_by_title(&pages);
        assert_eq!(g, vec![("Rust".to_string(), 3), ("Cooking".to_string(), 1)]);
    }

    #[test]
    fn sorts_newest_first() {
        let mut v = vec![
            mk("Old", "a.md", 1),
            mk("New", "b.md", 9),
            mk("Mid", "c.md", 5),
        ];
        sort_newest_first(&mut v);
        let order: Vec<&str> = v.iter().map(|p| p.title.as_str()).collect();
        assert_eq!(order, ["New", "Mid", "Old"]);
    }

    #[test]
    fn render_citations_verified_and_badged() {
        use wenlan_types::pages::PageCitation;
        let cites = vec![
            PageCitation {
                occurrence: 1,
                marker: 1,
                source_kind: "memory".into(),
                locator: "mem_a".into(),
                score: 0.9,
                status: "verified".into(),
                scope: "sentence".into(),
            },
            PageCitation {
                occurrence: 2,
                marker: 2,
                source_kind: "memory".into(),
                locator: "mem_b".into(),
                score: 0.2,
                status: "unverified".into(),
                scope: "paragraph".into(),
            },
        ];
        let out = render_citations(&cites);
        assert!(out.contains("[1] mem_a"));
        assert!(out.contains("⚠ not directly traceable"));
        assert!(!render_citations(&[]).contains("Citations"));
    }

    #[test]
    fn read_pages_reads_md_skips_non_md_and_missing_dir() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("one.md"), "---\ntitle: One\n---\n").unwrap();
        std::fs::write(tmp.path().join("two.md"), "---\ntitle: Two\n---\n").unwrap();
        std::fs::write(tmp.path().join("notes.txt"), "not a page").unwrap();
        // a .wenlan/state.json sibling must be ignored (non-recursive, non-md)
        std::fs::create_dir_all(tmp.path().join(".wenlan")).unwrap();
        std::fs::write(tmp.path().join(".wenlan/state.json"), "{}").unwrap();

        let pages = read_pages(tmp.path());
        assert_eq!(pages.len(), 2, "only the two .md files");
        let mut titles: Vec<&str> = pages.iter().map(|p| p.title.as_str()).collect();
        titles.sort();
        assert_eq!(titles, ["One", "Two"]);

        // missing dir -> empty, no panic
        let empty = read_pages(&tmp.path().join("does-not-exist"));
        assert!(empty.is_empty());
    }
}
