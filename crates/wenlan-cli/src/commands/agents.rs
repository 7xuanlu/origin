// SPDX-License-Identifier: Apache-2.0
//! `wenlan agents list/show/edit` — manage registered agents.

use anyhow::Result;
use clap::Subcommand;
use wenlan_types::requests::UpdateAgentRequest;

use crate::client::WenlanClient;
use crate::output::{print_json, ResolvedFormat};

#[derive(Subcommand)]
pub enum AgentsCmd {
    /// List all registered agents.
    List,
    /// Show a single agent's full details.
    Show {
        /// Agent name (e.g. "claude-code", "cursor").
        name: String,
    },
    /// Update an agent's trust level, enabled state, or metadata.
    Edit {
        /// Agent name.
        name: String,
        /// New trust level: "full", "review", or "unknown". A store with
        /// `supersedes` from an agent below "full" stages for human review
        /// instead of taking effect.
        #[arg(long)]
        trust: Option<String>,
        /// Enable or disable the agent.
        #[arg(long)]
        enabled: Option<bool>,
        /// New display name (use empty string "" to clear).
        #[arg(long = "display-name")]
        display_name: Option<String>,
        /// New description.
        #[arg(long)]
        description: Option<String>,
    },
}

pub async fn run(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    cmd: AgentsCmd,
) -> Result<()> {
    match cmd {
        AgentsCmd::List => list(client, format, quiet).await,
        AgentsCmd::Show { name } => show(client, format, quiet, &name).await,
        AgentsCmd::Edit {
            name,
            trust,
            enabled,
            display_name,
            description,
        } => {
            edit(
                client,
                format,
                quiet,
                &name,
                trust,
                enabled,
                display_name,
                description,
            )
            .await
        }
    }
}

async fn list(client: &WenlanClient, format: ResolvedFormat, quiet: bool) -> Result<()> {
    let agents = client.list_agents().await?;
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&agents)?,
        ResolvedFormat::Table => {
            if agents.is_empty() {
                println!("(no agents registered)");
                return Ok(());
            }
            println!("{} agent(s)", agents.len());
            for a in &agents {
                let display = a.display_name.as_deref().unwrap_or(&a.name);
                let status = if a.enabled { "enabled" } else { "disabled" };
                println!(
                    "  {:24} [{}] trust={} memories={}",
                    display, status, a.trust_level, a.memory_count
                );
            }
        }
    }
    Ok(())
}

async fn show(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    name: &str,
) -> Result<()> {
    let agent = client.get_agent(name).await?;
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&agent)?,
        ResolvedFormat::Table => {
            println!("Agent: {}", agent.name);
            println!("  ID:            {}", agent.id);
            println!(
                "  Display:       {}",
                agent.display_name.as_deref().unwrap_or("-")
            );
            println!("  Type:          {}", agent.agent_type);
            println!(
                "  Description:   {}",
                agent.description.as_deref().unwrap_or("-")
            );
            println!("  Enabled:       {}", agent.enabled);
            println!("  Trust:         {}", agent.trust_level);
            println!("  Memory count:  {}", agent.memory_count);
            println!(
                "  Last seen:     {}",
                agent
                    .last_seen_at
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn edit(
    client: &WenlanClient,
    format: ResolvedFormat,
    quiet: bool,
    name: &str,
    trust: Option<String>,
    enabled: Option<bool>,
    display_name: Option<String>,
    description: Option<String>,
) -> Result<()> {
    if trust.is_none() && enabled.is_none() && display_name.is_none() && description.is_none() {
        anyhow::bail!(
            "No fields to update. Provide at least --trust, --enabled, --display-name, or --description."
        );
    }
    let req = UpdateAgentRequest {
        agent_type: None,
        description,
        enabled,
        trust_level: trust,
        display_name,
    };
    let updated = client.update_agent(name, req).await?;
    if quiet {
        return Ok(());
    }
    match format {
        ResolvedFormat::Json => print_json(&updated)?,
        ResolvedFormat::Table => {
            println!("Updated agent {}", updated.name);
            println!("  Trust:         {}", updated.trust_level);
            println!("  Enabled:       {}", updated.enabled);
        }
    }
    Ok(())
}
