// SPDX-License-Identifier: Apache-2.0
//! `wenlan search <query>` — POST /api/search.

use anyhow::Result;
use wenlan_types::responses::SearchResponse;

use crate::client::WenlanClient;
use crate::output::{print_json, ResolvedFormat};

pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    query: String,
    limit: usize,
) -> Result<()> {
    let resp = client.search(query, limit).await?;
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&resp)?,
        ResolvedFormat::Table => print_table(&resp),
    }
    Ok(())
}

fn print_table(resp: &SearchResponse) {
    print!("{}", format_table(resp));
}

fn format_table(resp: &SearchResponse) -> String {
    if resp.results.is_empty() {
        return "(no results)\n".to_string();
    }
    let mut output = format!(
        "{} result(s) in {:.0}ms\n",
        resp.results.len(),
        resp.took_ms
    );
    for r in &resp.results {
        // Use title if non-empty, otherwise fall back to first content line.
        let title: &str = if r.title.is_empty() {
            r.content.lines().next().unwrap_or("(no title)")
        } else {
            &r.title
        };
        // Truncate to 60 chars.
        let title_disp = if title.chars().count() > 60 {
            format!("{}...", title.chars().take(57).collect::<String>())
        } else {
            title.to_string()
        };
        output.push_str(&format!(
            "  [{:.3}] {} ({})\n",
            r.score, title_disp, r.source_id
        ));
    }
    if let Some(pages) = &resp.supplemental_pages {
        if !pages.is_empty() {
            output.push_str("Compiled pages:\n");
            for p in pages {
                let t = if p.title.is_empty() {
                    p.content.lines().next().unwrap_or("(page)")
                } else {
                    &p.title
                };
                output.push_str(&format!("  - {} ({})\n", t, p.source_id));
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use wenlan_types::memory::SearchResult;

    fn search_result(id: &str, title: &str, source_id: &str) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: "First page line\nSecond page line".to_string(),
            source: "page".to_string(),
            source_id: source_id.to_string(),
            title: title.to_string(),
            url: None,
            chunk_index: 0,
            last_modified: 0,
            score: 0.9,
            chunk_type: None,
            language: None,
            semantic_unit: None,
            memory_type: None,
            space: None,
            source_agent: None,
            confidence: None,
            confirmed: None,
            stability: None,
            supersedes: None,
            summary: None,
            entity_id: None,
            entity_name: None,
            quality: None,
            importance: None,
            event_date: None,
            is_archived: false,
            is_recap: false,
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            content_hash: None,
            raw_score: 0.0,
            version: 0,
            pending_revision: false,
            merged_from: None,
            last_delta_summary: None,
        }
    }

    #[test]
    fn format_table_renders_supplemental_pages_section() {
        let resp = SearchResponse {
            results: vec![search_result("r1", "Memory result", "memory-1")],
            took_ms: 12.0,
            supplemental_pages: Some(vec![search_result("p1", "Compiled page", "page-source-1")]),
        };

        let output = format_table(&resp);

        assert!(output.contains("Compiled pages:"));
        assert!(output.contains("page-source-1"));
    }
}
