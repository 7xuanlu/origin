// SPDX-License-Identifier: Apache-2.0
//! `wenlan entities merge/alias` — knowledge-graph entity identity surface.

use anyhow::Result;
use clap::Subcommand;

use crate::client::WenlanClient;
use crate::output::OutputFormat;

#[derive(Subcommand)]
pub enum EntitiesCmd {
    /// Merge one entity into another. The loser is deleted; its observations,
    /// memory links, edges, and aliases move to the canonical.
    Merge {
        /// The entity to merge away (id or exact name) — the loser.
        loser: String,
        /// The entity to merge into (id or exact name) — the canonical.
        #[arg(long)]
        into: String,
        /// Preview the merge counts without mutating anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Declare an additional name for an entity.
    Alias {
        /// Entity id or exact name.
        entity: String,
        /// The alias to add.
        alias: String,
    },
}

pub async fn run(
    client: &WenlanClient,
    format: OutputFormat,
    quiet: bool,
    cmd: EntitiesCmd,
) -> Result<()> {
    match cmd {
        EntitiesCmd::Merge {
            loser,
            into,
            dry_run,
        } => merge(client, format, quiet, &loser, &into, dry_run).await,
        EntitiesCmd::Alias { entity, alias } => {
            alias_cmd(client, format, quiet, &entity, &alias).await
        }
    }
}

/// Resolve `id_or_name` to an entity's (id, name): a direct id lookup first,
/// falling back to an exact case-insensitive name match against
/// `/api/memory/entities/search` (a vector-similarity endpoint, not an exact
/// lookup — so ranked results are filtered client-side down to exact name
/// matches). Errors when zero or more than one entity matches the name.
async fn resolve_entity(client: &WenlanClient, id_or_name: &str) -> Result<(String, String)> {
    if let Some(detail) = client.get_entity(id_or_name).await? {
        return Ok((detail.entity.id, detail.entity.name));
    }
    let response = client.search_entities(id_or_name.to_string(), 50).await?;
    let mut matches = response
        .results
        .into_iter()
        .filter(|r| r.entity.name.eq_ignore_ascii_case(id_or_name));
    let first = matches
        .next()
        .ok_or_else(|| anyhow::anyhow!("no entity found with id or exact name '{}'", id_or_name))?;
    let mut candidates = vec![format!("{} ({})", first.entity.id, first.entity.name)];
    for extra in matches {
        candidates.push(format!("{} ({})", extra.entity.id, extra.entity.name));
    }
    if candidates.len() > 1 {
        anyhow::bail!(
            "multiple entities match name '{}': {}",
            id_or_name,
            candidates.join(", ")
        );
    }
    Ok((first.entity.id, first.entity.name))
}

async fn merge(
    client: &WenlanClient,
    format: OutputFormat,
    quiet: bool,
    loser: &str,
    into: &str,
    dry_run: bool,
) -> Result<()> {
    let (loser_id, _) = resolve_entity(client, loser).await?;
    let (canonical_id, _) = resolve_entity(client, into).await?;
    let response = client
        .merge_entity(&loser_id, canonical_id, dry_run)
        .await?;
    if quiet {
        return Ok(());
    }
    match format {
        OutputFormat::Json => crate::output::print_json(&response)?,
        OutputFormat::Table => {
            let verb = if response.applied {
                "Merged"
            } else {
                "Would merge"
            };
            println!(
                "{verb} '{}' into '{}' ({} memory links, {} observations, {} edges).",
                response.loser_name,
                response.canonical_name,
                response.memory_links,
                response.observations,
                response.edges,
            );
            if !response.aliases_added.is_empty() {
                println!("Aliases added: {}", response.aliases_added.join(", "));
            }
        }
        OutputFormat::Auto => unreachable!("Auto resolved by main before dispatch"),
    }
    Ok(())
}

async fn alias_cmd(
    client: &WenlanClient,
    format: OutputFormat,
    quiet: bool,
    entity: &str,
    alias: &str,
) -> Result<()> {
    let (entity_id, entity_name) = resolve_entity(client, entity).await?;
    let response = client
        .add_entity_alias(&entity_id, alias.to_string())
        .await?;
    if quiet {
        return Ok(());
    }
    match format {
        OutputFormat::Json => crate::output::print_json(&response)?,
        OutputFormat::Table => {
            println!("'{}' aliases: {}", entity_name, response.aliases.join(", "));
        }
        OutputFormat::Auto => unreachable!("Auto resolved by main before dispatch"),
    }
    Ok(())
}
