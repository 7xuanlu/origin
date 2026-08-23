// SPDX-License-Identifier: Apache-2.0
//! `wenlan recall <query>` — POST /api/memory/search.

use anyhow::Result;
use wenlan_types::responses::SearchMemoryResponse;

use crate::client::WenlanClient;
use crate::output::{print_json, ResolvedFormat};

const DEFAULT_RECALL_LIMIT: usize = 20;

pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    query: String,
) -> Result<()> {
    let response = client.recall(query, DEFAULT_RECALL_LIMIT).await?;
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&response)?,
        ResolvedFormat::Table => print_table(&response),
    }
    Ok(())
}

fn print_table(response: &SearchMemoryResponse) {
    if response.results.is_empty() {
        println!("(no recalled memories)");
        return;
    }
    println!(
        "{} recalled memor{} in {:.0}ms",
        response.results.len(),
        if response.results.len() == 1 {
            "y"
        } else {
            "ies"
        },
        response.took_ms
    );
    for result in response.results.iter().take(10) {
        let title = if result.title.is_empty() {
            result.content.lines().next().unwrap_or("(no title)")
        } else {
            &result.title
        };
        let title = if title.chars().count() > 70 {
            format!("{}...", title.chars().take(67).collect::<String>())
        } else {
            title.to_string()
        };
        println!("  [{:.3}] {} ({})", result.score, title, result.source_id);
    }
}
