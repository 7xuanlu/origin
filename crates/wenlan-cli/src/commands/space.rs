// SPDX-License-Identifier: Apache-2.0
//! `wenlan spaces list/add/default/move/show` — manage memory spaces.

use anyhow::Result;
use clap::Subcommand;

use crate::client::WenlanClient;
use crate::output::OutputFormat;

#[derive(Subcommand)]
pub enum SpaceCmd {
    /// List all registered spaces.
    List,
    /// Register a new space.
    Add {
        /// Space name (e.g. "career", "health", "ideas").
        name: String,
        /// Also set this space as the default.
        #[arg(long)]
        default: bool,
    },
    /// Get or set the default space.
    Default {
        /// Space name to set as default. Omit to print the current default.
        #[arg(conflicts_with = "clear")]
        name: Option<String>,
        /// Clear the daemon-owned default save space.
        #[arg(long)]
        clear: bool,
    },
    /// Bulk-reassign all memories from one space to another.
    Move {
        /// Source space.
        from: String,
        /// Destination space.
        to: String,
    },
    /// Show detail for a space — name, description, memory count, entity count, default + starred flags.
    Show {
        /// Space name.
        name: String,
    },
}

pub async fn run(
    client: &WenlanClient,
    format: OutputFormat,
    quiet: bool,
    cmd: SpaceCmd,
) -> Result<()> {
    match cmd {
        SpaceCmd::List => list(client, format, quiet).await,
        SpaceCmd::Add { name, default } => add(client, format, quiet, &name, default).await,
        SpaceCmd::Default { name, clear } => {
            default_cmd(client, format, quiet, name.as_deref(), clear).await
        }
        SpaceCmd::Move { from, to } => move_cmd(client, format, quiet, &from, &to).await,
        SpaceCmd::Show { name } => show(client, format, quiet, &name).await,
    }
}

async fn list(client: &WenlanClient, format: OutputFormat, quiet: bool) -> Result<()> {
    let spaces = client.list_spaces().await?;
    if quiet {
        return Ok(());
    }
    match format {
        OutputFormat::Json => crate::output::print_json(&spaces)?,
        OutputFormat::Table => {
            if spaces.is_empty() {
                println!("(no spaces registered)");
                return Ok(());
            }
            println!(
                "{:<20} {:<10} {:<10} {:<8}",
                "NAME", "MEMORIES", "ENTITIES", "DEFAULT?"
            );
            for s in &spaces {
                println!(
                    "{:<20} {:<10} {:<10} {:<8}",
                    s.name,
                    s.memory_count,
                    s.entity_count,
                    if s.is_default { "yes" } else { "" }
                );
            }
        }
        OutputFormat::Auto => unreachable!("Auto resolved by main before dispatch"),
    }
    Ok(())
}
async fn add(
    client: &WenlanClient,
    _format: OutputFormat,
    quiet: bool,
    name: &str,
    set_default: bool,
) -> Result<()> {
    match client.create_space(name).await {
        Ok(()) => {
            if !quiet {
                println!("Registered space '{}'.", name);
            }
        }
        Err(e) => {
            // Best-effort detection of "already exists" via error string match.
            // The daemon currently returns 500 on UNIQUE constraint failure;
            // surface as a non-fatal warning so the --default flow can proceed.
            let s = e.to_string().to_lowercase();
            let already_exists = s.contains("unique constraint")
                || s.contains("already exists")
                || s.contains("duplicate");
            if already_exists {
                if !quiet {
                    println!("Space '{}' already registered (no-op).", name);
                }
            } else {
                return Err(e);
            }
        }
    }
    if set_default {
        let space = client.get_space(name).await?;
        client.set_default_space(space.id).await?;
        if !quiet {
            println!("Set '{}' as the Default save space.", name);
        }
    }
    Ok(())
}
async fn default_cmd(
    client: &WenlanClient,
    format: OutputFormat,
    quiet: bool,
    name: Option<&str>,
    clear: bool,
) -> Result<()> {
    if clear {
        client.clear_default_space().await?;
        if !quiet {
            println!("Cleared the Default save space.");
        }
        return Ok(());
    }

    let response = match name {
        Some(name) => {
            let space = client.get_space(name).await?;
            client.set_default_space(space.id).await?
        }
        None => client.get_default_space().await?,
    };
    if quiet {
        return Ok(());
    }
    match format {
        OutputFormat::Json => crate::output::print_json(&response)?,
        OutputFormat::Table => match response.space {
            Some(space) => {
                if name.is_some() {
                    println!("Set Default save space to '{}'.", space.name);
                } else {
                    println!("{}", space.name);
                }
            }
            None => {
                println!("(no Default save space set; new writes use Uncategorized)");
            }
        },
        OutputFormat::Auto => unreachable!("Auto resolved by main before dispatch"),
    }
    Ok(())
}
async fn move_cmd(
    client: &WenlanClient,
    _format: OutputFormat,
    quiet: bool,
    from: &str,
    to: &str,
) -> Result<()> {
    let n = client.move_space(from, to).await?;
    if !quiet {
        println!(
            "Moved {} memory rows from '{}' to '{}'; pages and entities were cascaded.",
            n, from, to
        );
    }
    Ok(())
}
async fn show(client: &WenlanClient, format: OutputFormat, quiet: bool, name: &str) -> Result<()> {
    let space = client.get_space(name).await?;
    if quiet {
        return Ok(());
    }
    match format {
        OutputFormat::Json => crate::output::print_json(&space)?,
        OutputFormat::Table => {
            println!("Name:           {}", space.name);
            if let Some(desc) = &space.description {
                println!("Description:    {}", desc);
            }
            println!("Memory count:   {}", space.memory_count);
            println!("Entity count:   {}", space.entity_count);
            if space.starred {
                println!("Starred:        yes");
            }
            if space.is_default {
                println!("Default:        yes");
            }
        }
        OutputFormat::Auto => unreachable!("Auto resolved by main before dispatch"),
    }
    Ok(())
}
