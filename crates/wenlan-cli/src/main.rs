// SPDX-License-Identifier: Apache-2.0
use clap::{Parser, Subcommand};
use output::OutputFormat;
use std::path::PathBuf;
use std::process::ExitCode;
use wenlan_cli::space_context::{
    resolve_agent_name, resolve_cli_space, resolve_cli_space_offline, resolve_native_read_space,
    CliSpaceOperation,
};
use wenlan_cli::{client, commands, output};
use wenlan_types::lint::LintProfile;

#[derive(Parser)]
#[command(
    name = "wenlan",
    version,
    about = "Wenlan CLI. Set up and use the local Wenlan runtime."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format. Auto-detects JSON when piped, table on TTY.
    #[arg(long, value_enum, default_value_t = OutputFormat::Auto, global = true)]
    format: OutputFormat,

    /// Suppress all non-error output. Useful for scripts.
    #[arg(long, short, global = true)]
    quiet: bool,

    /// Identify the caller in X-Agent-Name audit metadata.
    #[arg(long, global = true)]
    agent_name: Option<String>,

    /// Use one registered Space for scope-aware reads and writes.
    #[arg(long, global = true, conflicts_with = "all_spaces")]
    space: Option<String>,

    /// Read across every Space. Invalid for writes and strict WENLAN_SPACE pins.
    #[arg(long, global = true)]
    all_spaces: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Show background process, model, API key, and memory state.
    Status,
    /// Guided setup for local memory, a local model, or an Anthropic key.
    Setup {
        /// Set up without a model or API key.
        #[arg(long)]
        basic: bool,
        /// Download and select a local model, for example qwen3-4b.
        #[arg(long, value_name = "MODEL_ID")]
        model: Option<String>,
        /// Read an Anthropic key from this environment variable.
        #[arg(long = "anthropic-api-key-env", value_name = "ENV_VAR")]
        anthropic_api_key_env: Option<String>,
        /// Skip confirmation prompts where possible.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Control whether Wenlan keeps running in the background.
    Background {
        #[command(subcommand)]
        command: commands::service::BackgroundCommand,
    },
    /// Restart the Wenlan background process. Required after an update.
    Restart,
    /// Diagnose runtime, model, and API key setup.
    Doctor,
    /// Check memory, Pages, runtime, and operation health through the daemon.
    Lint {
        #[arg(long)]
        profile: Option<LintProfile>,
        /// Permit a deep semantic pass to use an already configured external provider.
        #[arg(long)]
        allow_external: bool,
        /// Return bounded high-recall semantic candidates for the calling agent.
        #[arg(long)]
        agent_assist: bool,
        /// Submit typed agent verdicts produced from a prior prepare report.
        #[arg(long, value_name = "JSON_FILE")]
        agent_submission: Option<PathBuf>,
    },
    /// Manage local models.
    Models {
        #[command(subcommand)]
        command: commands::setup::ModelCommand,
    },
    /// Manage provider API keys.
    Keys {
        #[command(subcommand)]
        command: commands::setup::KeyCommand,
    },
    /// Configure, inspect, or disable model-backed background enrichment.
    Enrichment {
        #[command(subcommand)]
        command: commands::setup::EnrichmentCommand,
    },
    /// Connect Wenlan to a supported agent or editor.
    Connect(commands::mcp::ConnectArgs),
    /// Search memories by query (vector + FTS hybrid).
    Search {
        /// Search query.
        query: String,
        /// Max results (default 10).
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
    },
    /// Recall memories relevant to a query.
    Recall {
        /// Query to recall memories for.
        query: String,
    },
    /// Read the current Space Brief, optionally with related context.
    Brief(commands::brief::BriefArgs),
    /// Browse distilled pages, or open one in your editor by title query.
    Pages {
        /// Title/filename substring. Omit to list pages newest-first.
        query: Option<String>,
        /// Max pages to list (newest-first). 0 = all. Ignored when a query opens a page.
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
        /// Print the matched page's stable internal id instead of opening it.
        #[arg(long)]
        resolve_id: bool,
    },
    /// Manage folders and files Wenlan should learn from.
    Sources {
        #[command(subcommand)]
        command: commands::ingest::SourcesCommand,
    },
    /// Capture a memory. Provide text positionally, or use --file, or pipe via stdin.
    Capture {
        /// Content text. If omitted and --file unset, read from stdin.
        #[arg(conflicts_with = "file")]
        text: Option<String>,
        /// Read content from a file.
        #[arg(short, long)]
        file: Option<std::path::PathBuf>,
        /// Memory type (e.g. fact, decision).
        #[arg(short = 't', long = "type")]
        memory_type: Option<String>,
    },
    /// List recent memories.
    Memories {
        /// Max results.
        #[arg(short, long, default_value_t = 20)]
        limit: usize,
        /// Filter by memory type.
        #[arg(short = 't', long = "type")]
        memory_type: Option<String>,
        /// Only unconfirmed memories (what the MCP list_pending tool returns).
        #[arg(long)]
        pending: bool,
    },
    /// Walk pending revisions (conflicts / merges) awaiting your accept or dismiss.
    Curate {
        #[command(subcommand)]
        action: Option<commands::curate::CurateAction>,
    },
    /// Manage registered agents (list / show / edit).
    Agents {
        #[command(subcommand)]
        cmd: commands::agents::AgentsCmd,
    },
    /// Manage memory spaces (list, add, default, move, show).
    Spaces {
        #[command(subcommand)]
        cmd: commands::space::SpaceCmd,
    },
    /// Inspect or drain writes queued while the daemon was unreachable.
    Outbox {
        #[command(subcommand)]
        command: commands::outbox::OutboxCommand,
    },
    /// Merge or alias knowledge-graph entities (id or exact name).
    Entities {
        #[command(subcommand)]
        cmd: commands::entities::EntitiesCmd,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    if cli.all_spaces && matches!(&cli.command, Commands::Brief(_)) {
        anyhow::bail!("Brief is owned by one Space; --all-spaces is not supported");
    }
    let environment_agent = std::env::var("WENLAN_AGENT_NAME").ok();
    let agent_name = resolve_agent_name(cli.agent_name.as_deref(), environment_agent.as_deref());
    let recovery_enabled = !matches!(
        &cli.command,
        Commands::Status | Commands::Background { .. } | Commands::Restart
    );
    let base_client = client::WenlanClient::from_env_with_context(agent_name.as_deref(), None)?
        .with_recovery(recovery_enabled);
    let is_brief_update = matches!(
        &cli.command,
        Commands::Brief(args) if args.command.is_some()
    );
    let is_outbox = matches!(&cli.command, Commands::Outbox { .. });
    let operation = match &cli.command {
        Commands::Search { .. }
        | Commands::Recall { .. }
        | Commands::Brief(commands::brief::BriefArgs { command: None, .. })
        | Commands::Memories { .. } => Some(CliSpaceOperation::Read),
        Commands::Capture { .. } => Some(CliSpaceOperation::Write),
        _ => None,
    };
    let is_lint = matches!(&cli.command, Commands::Lint { .. });
    let mut effective_cli_space = cli.space.clone();
    let client = if is_outbox {
        if cli.space.is_some() || cli.all_spaces {
            anyhow::bail!("--space/--all-spaces are not supported by outbox commands");
        }
        effective_cli_space = None;
        base_client
    } else if let Some(operation) = operation {
        let context = match base_client.list_spaces().await {
            Ok(spaces) => resolve_cli_space(
                cli.space.clone(),
                cli.all_spaces,
                std::env::current_dir().ok(),
                operation,
                &spaces.into_iter().map(|space| space.name).collect(),
            )?,
            Err(error)
                if matches!(&cli.command, Commands::Capture { .. })
                    && base_client.is_local()
                    && wenlan_cli::outbox::is_daemon_unreachable(&error) =>
            {
                if cli.all_spaces {
                    anyhow::bail!("--all-spaces is valid only for read commands");
                }
                let context = resolve_cli_space_offline(
                    cli.space.clone(),
                    cli.all_spaces,
                    std::env::current_dir().ok(),
                    operation,
                )?;
                if let Some(space) = context.space.as_deref() {
                    eprintln!(
                        "wenlan: daemon unreachable — queued for Space '{space}' (not validated against the registry)"
                    );
                }
                context
            }
            Err(error) => return Err(error),
        };
        effective_cli_space = context.space.clone();
        client::WenlanClient::from_env_with_context(
            agent_name.as_deref(),
            context.space.as_deref(),
        )?
        .with_recovery(recovery_enabled)
    } else if is_brief_update {
        let strict_space = std::env::var("WENLAN_SPACE").ok();
        effective_cli_space =
            resolve_native_read_space(strict_space.as_deref(), cli.space.as_deref(), false)?;
        client::WenlanClient::from_env_with_context(
            agent_name.as_deref(),
            effective_cli_space.as_deref(),
        )?
        .with_recovery(recovery_enabled)
    } else if is_lint {
        let strict_space = std::env::var("WENLAN_SPACE").ok();
        effective_cli_space = resolve_native_read_space(
            strict_space.as_deref(),
            cli.space.as_deref(),
            cli.all_spaces,
        )?;
        base_client
    } else {
        if cli.space.is_some() || cli.all_spaces {
            anyhow::bail!("--space/--all-spaces are not supported by this command");
        }
        base_client
    };
    // Resolve Auto once based on stdout TTY state. Subcommands receive Json or Table only.
    let format = cli.format.resolve();
    match cli.command {
        Commands::Status => commands::status::run(&client, format, cli.quiet).await?,
        Commands::Setup {
            basic,
            model,
            anthropic_api_key_env,
            yes,
        } => {
            commands::setup::run_setup(commands::setup::SetupArgs {
                basic,
                model,
                anthropic_api_key_env,
                yes,
            })
            .await?
        }
        Commands::Background { command } => commands::service::run_background(command).await?,
        Commands::Restart => commands::service::restart()?,
        Commands::Doctor => commands::setup::run_doctor().await?,
        Commands::Lint {
            profile,
            allow_external,
            agent_assist,
            agent_submission,
        } => {
            return Ok(commands::lint::run(
                &client,
                format,
                cli.quiet,
                profile,
                effective_cli_space,
                allow_external,
                agent_assist,
                agent_submission,
            )
            .await)
        }
        Commands::Models { command } => commands::setup::run_model(command).await?,
        Commands::Keys { command } => commands::setup::run_key(command).await?,
        Commands::Enrichment { command } => commands::setup::run_enrichment(command).await?,
        Commands::Connect(args) => commands::mcp::run_connect(args, cli.quiet)?,
        Commands::Search { query, limit } => {
            commands::search::run(&client, format, cli.quiet, query, limit).await?
        }
        Commands::Recall { query } => {
            commands::recall::run(&client, format, cli.quiet, query).await?
        }
        Commands::Brief(args) => {
            commands::brief::run(
                &client,
                format,
                cli.quiet,
                args,
                effective_cli_space,
                agent_name.as_deref(),
            )
            .await?
        }
        Commands::Pages {
            query,
            limit,
            resolve_id,
        } => commands::pages::run(format, cli.quiet, query, limit, resolve_id)?,
        Commands::Sources { command } => {
            commands::ingest::run_sources(&client, format, cli.quiet, command).await?
        }
        Commands::Capture {
            text,
            file,
            memory_type,
        } => {
            commands::store::run(
                &client,
                format,
                cli.quiet,
                text,
                file,
                memory_type,
                effective_cli_space,
                agent_name.as_deref(),
            )
            .await?
        }
        Commands::Memories {
            limit,
            memory_type,
            pending,
        } => commands::list::run(&client, format, cli.quiet, limit, memory_type, pending).await?,
        Commands::Curate { action } => {
            commands::curate::run(&client, format, cli.quiet, action).await?
        }
        Commands::Agents { cmd } => commands::agents::run(&client, format, cli.quiet, cmd).await?,
        Commands::Spaces { cmd } => commands::space::run(&client, format, cli.quiet, cmd).await?,
        Commands::Outbox { command } => {
            commands::outbox::run(&client, format, cli.quiet, command).await?
        }
        Commands::Entities { cmd } => {
            commands::entities::run(&client, format, cli.quiet, cmd).await?
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod catalog_tests {
    use super::Cli;
    use clap::CommandFactory;
    use std::collections::BTreeSet;

    /// Every top-level clap subcommand must have a row in the M5 truth
    /// manifest, and the manifest must not list a command that no longer
    /// exists. Clap is the source of truth; the manifest is the catalog.
    #[test]
    fn truth_manifest_cli_rows_match_clap_subcommands() {
        let clap_names: BTreeSet<String> = Cli::command()
            .get_subcommands()
            .map(|c| format!("wenlan {}", c.get_name()))
            .filter(|name| name != "wenlan help")
            .collect();
        let manifest: BTreeSet<String> = wenlan_core::truth_manifest::CLI_READERS
            .iter()
            .map(|r| r.subcommand.to_string())
            .collect();
        assert_eq!(
            clap_names, manifest,
            "CLI_READERS in wenlan-core/src/truth_manifest.rs drifted from the clap `Commands` enum"
        );
    }
}
