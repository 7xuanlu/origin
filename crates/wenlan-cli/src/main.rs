// SPDX-License-Identifier: Apache-2.0
use clap::{Parser, Subcommand};
use output::OutputFormat;
use std::path::PathBuf;
use std::process::ExitCode;
use wenlan_cli::space_context::{
    resolve_agent_name, resolve_cli_space, resolve_native_read_space, CliSpaceOperation,
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
    /// Recall the working memory bundle for a query.
    Recall {
        /// Query to recall context for.
        query: String,
    },
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
        /// Memory type (e.g. fact, task, decision).
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
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let environment_agent = std::env::var("WENLAN_AGENT_NAME").ok();
    let agent_name = resolve_agent_name(cli.agent_name.as_deref(), environment_agent.as_deref());
    let base_client = client::WenlanClient::from_env_with_context(agent_name.as_deref(), None)?;
    let operation = match &cli.command {
        Commands::Search { .. } | Commands::Recall { .. } | Commands::Memories { .. } => {
            Some(CliSpaceOperation::Read)
        }
        Commands::Capture { .. } => Some(CliSpaceOperation::Write),
        _ => None,
    };
    let is_lint = matches!(&cli.command, Commands::Lint { .. });
    let mut effective_cli_space = cli.space.clone();
    let client = if let Some(operation) = operation {
        let registered = base_client
            .list_spaces()
            .await?
            .into_iter()
            .map(|space| space.name)
            .collect();
        let context = resolve_cli_space(
            cli.space.clone(),
            cli.all_spaces,
            std::env::current_dir().ok(),
            operation,
            &registered,
        )?;
        effective_cli_space = context.space.clone();
        client::WenlanClient::from_env_with_context(
            agent_name.as_deref(),
            context.space.as_deref(),
        )?
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
        } => commands::store::run(&client, format, cli.quiet, text, file, memory_type).await?,
        Commands::Memories { limit, memory_type } => {
            commands::list::run(&client, format, cli.quiet, limit, memory_type).await?
        }
        Commands::Curate { action } => {
            commands::curate::run(&client, format, cli.quiet, action).await?
        }
        Commands::Agents { cmd } => commands::agents::run(&client, format, cli.quiet, cmd).await?,
        Commands::Spaces { cmd } => commands::space::run(&client, format, cli.quiet, cmd).await?,
    }
    Ok(ExitCode::SUCCESS)
}
