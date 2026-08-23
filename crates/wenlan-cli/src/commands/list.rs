// SPDX-License-Identifier: Apache-2.0
//! `wenlan memories [--limit N] [--type X] [--pending]` — POST /api/memory/list.

use anyhow::Result;
use wenlan_types::responses::ListMemoriesResponse;

use crate::client::WenlanClient;
use crate::output::{print_json, ResolvedFormat};

pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    limit: usize,
    memory_type: Option<String>,
    pending: bool,
) -> Result<()> {
    let resp = client
        .list(Some(limit), memory_type, pending.then_some(false))
        .await?;
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&resp)?,
        ResolvedFormat::Table => print_table(&resp),
    }
    Ok(())
}

fn print_table(resp: &ListMemoriesResponse) {
    if resp.memories.is_empty() {
        println!("(no memories)");
        return;
    }
    println!(
        "{} memor{}",
        resp.memories.len(),
        if resp.memories.len() == 1 { "y" } else { "ies" }
    );
    for m in &resp.memories {
        let title: &str = if m.title.is_empty() {
            "(no title)"
        } else {
            &m.title
        };
        let title_disp = if title.chars().count() > 60 {
            format!("{}...", title.chars().take(57).collect::<String>())
        } else {
            title.to_string()
        };
        let mtype = m.memory_type.as_deref().unwrap_or("-");
        println!("  {} [{}] {}", m.source_id, mtype, title_disp);
    }
}
