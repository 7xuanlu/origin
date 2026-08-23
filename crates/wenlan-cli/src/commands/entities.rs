// SPDX-License-Identifier: Apache-2.0
//! `wenlan entities merge/alias` — knowledge-graph entity identity surface.

use anyhow::Result;
use clap::Subcommand;

use wenlan_types::responses::ListEntitiesResponse;

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
/// `/api/memory/entities/list` (every live entity in scope, no by-name
/// filter on the route, so results are filtered client-side down to exact
/// name matches). Errors when zero or more than one entity matches the name.
/// `list_cache` lets callers that resolve two arguments reuse one
/// `/api/memory/entities/list` response instead of fetching it twice.
async fn resolve_entity_with(
    client: &WenlanClient,
    id_or_name: &str,
    list_cache: &mut Option<ListEntitiesResponse>,
) -> Result<(String, String)> {
    if let Some(detail) = client.get_entity(id_or_name).await? {
        return Ok((detail.entity.id, detail.entity.name));
    }
    if list_cache.is_none() {
        *list_cache = Some(client.list_entities().await?);
    }
    let name_lower = id_or_name.to_lowercase();
    let matches: Vec<_> = list_cache
        .as_ref()
        .expect("just populated above")
        .entities
        .iter()
        .filter(|e| e.name.to_lowercase() == name_lower)
        .collect();
    match matches.as_slice() {
        [] => anyhow::bail!("no entity found with id or exact name '{}'", id_or_name),
        [only] => Ok((only.id.clone(), only.name.clone())),
        many => {
            let candidates = many
                .iter()
                .map(|e| format!("{} ({})", e.id, e.name))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "multiple entities match name '{}': {}",
                id_or_name,
                candidates
            )
        }
    }
}

async fn merge(
    client: &WenlanClient,
    format: OutputFormat,
    quiet: bool,
    loser: &str,
    into: &str,
    dry_run: bool,
) -> Result<()> {
    let mut list_cache = None;
    let (loser_id, _) = resolve_entity_with(client, loser, &mut list_cache).await?;
    let (canonical_id, _) = resolve_entity_with(client, into, &mut list_cache).await?;
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
    let (entity_id, entity_name) = resolve_entity_with(client, entity, &mut None).await?;
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
