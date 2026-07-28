use std::net::IpAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use rmcp::{transport::stdio, ServiceExt};
use wenlan_mcp::client::{discover_origin_url, WenlanClient};
use wenlan_mcp::tools::{TransportMode, WenlanMcpServer};
use wenlan_mcp::{serve, token};

#[derive(Parser)]
#[command(
    name = "wenlan-mcp",
    about = "MCP server for Wenlan — personal agent memory layer",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Wenlan server URL (e.g. http://127.0.0.1:7878). Auto-discovers if not set.
    #[arg(long, global = true)]
    origin_url: Option<String>,

    /// Agent name for source_agent on writes.
    #[arg(long, global = true)]
    agent_name: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start Streamable HTTP MCP server for remote clients (claude.ai, ChatGPT)
    Serve(ServeArgs),
    /// Manage bearer tokens for authentication
    Token(TokenArgs),
}

#[derive(Parser)]
struct ServeArgs {
    /// Port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Bearer token for authentication
    #[arg(long)]
    token: Option<String>,

    /// Path to file containing the bearer token
    #[arg(long)]
    token_file: Option<PathBuf>,

    /// Disable authentication (only allowed on loopback)
    #[arg(long)]
    no_auth: bool,

    /// User ID for multi-user/cross-device support (optional)
    #[arg(long)]
    user_id: Option<String>,

    /// Comma-separated list of allowed Wenlan headers
    #[arg(long, default_value = "https://claude.ai,https://chatgpt.com")]
    allowed_origins: String,
}

#[derive(Parser)]
struct TokenArgs {
    #[command(subcommand)]
    action: TokenAction,
}

#[derive(Subcommand)]
enum TokenAction {
    /// Generate a new bearer token
    Generate {
        /// Output file path
        #[arg(long, default_value = "~/.config/wenlan-mcp/token")]
        output: String,
    },
}

fn default_token_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("wenlan-mcp")
        .join("token")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("wenlan_mcp=info".parse()?),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    wenlan_mcp::lock_state::init_from_env();
    if let Some(space) = wenlan_mcp::lock_state::locked_space() {
        eprintln!(
            "wenlan-mcp: WENLAN_SPACE strict pin active, space=\"{}\"",
            space
        );
    } else if let Some(space) = wenlan_mcp::lock_state::default_space() {
        eprintln!(
            "wenlan-mcp: WENLAN_DEFAULT_SPACE fallback active, space=\"{}\"",
            space
        );
    } else {
        eprintln!("wenlan-mcp: no process Space context");
    }

    match cli.command {
        None => run_stdio(cli.origin_url, effective_agent_name(cli.agent_name, None)).await,
        Some(Commands::Serve(args)) => {
            let agent_name = effective_agent_name(cli.agent_name, Some(&args));
            run_serve(args, cli.origin_url, agent_name).await
        }
        Some(Commands::Token(args)) => run_token(args),
    }
}

fn effective_agent_name(agent_name: Option<String>, serve_args: Option<&ServeArgs>) -> String {
    agent_name.unwrap_or_else(|| {
        if serve_args.is_some() {
            "remote-mcp".into()
        } else {
            "claude-code".into()
        }
    })
}

async fn run_stdio(origin_url: Option<String>, agent_name: String) -> anyhow::Result<()> {
    let base_url = discover_origin_url(origin_url);
    tracing::info!("Connecting to Wenlan at {}", base_url);

    let client = WenlanClient::new(base_url).with_agent_name(agent_name.clone());

    if let Some(msg) = client.version_handshake().await {
        tracing::warn!("{msg}");
        eprintln!("Warning: {msg}");
    }

    tokio::spawn(async {
        if let Some(msg) = wenlan_mcp::self_update_check::check().await {
            tracing::info!("{msg}");
            eprintln!("Note: {msg}");
        }
    });

    let server = WenlanMcpServer::new(client, TransportMode::Stdio, agent_name, None);
    let service = server
        .serve(stdio())
        .await
        .inspect_err(|e| tracing::error!("Failed to start MCP server: {}", e))?;

    tracing::info!("wenlan-mcp server running on stdio");
    service.waiting().await?;
    Ok(())
}

async fn run_serve(
    args: ServeArgs,
    origin_url: Option<String>,
    agent_name: String,
) -> anyhow::Result<()> {
    let resolved_token = resolve_token(&args)?;

    if resolved_token.is_none() && !args.no_auth {
        anyhow::bail!(
            "Authentication required. Use --token, --token-file, or --no-auth.\n\
             Generate a token with: wenlan-mcp token generate"
        );
    }
    if args.no_auth {
        let host_addr: IpAddr = args
            .host
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid host address: {}", args.host))?;
        if !host_addr.is_loopback() {
            anyhow::bail!(
                "--no-auth is only allowed on loopback addresses (127.0.0.1 or ::1), not {}",
                args.host
            );
        }
    }

    let base_url = discover_origin_url(origin_url);
    tracing::info!("Connecting to Wenlan at {}", base_url);

    if let Some(msg) = WenlanClient::new(base_url.clone())
        .version_handshake()
        .await
    {
        tracing::warn!("{msg}");
        eprintln!("Warning: {msg}");
    }

    tokio::spawn(async {
        if let Some(msg) = wenlan_mcp::self_update_check::check().await {
            tracing::info!("{msg}");
            eprintln!("Note: {msg}");
        }
    });

    let allowed_origins: Vec<String> = args
        .allowed_origins
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    let config = serve::ServeConfig {
        port: args.port,
        host: args.host,
        origin_url: base_url,
        token: resolved_token,
        agent_name,
        user_id: args.user_id,
        allowed_origins,
    };

    serve::run_serve(config).await
}

fn resolve_token(args: &ServeArgs) -> anyhow::Result<Option<String>> {
    if let Some(ref t) = args.token {
        return Ok(Some(t.clone()));
    }
    if let Some(ref path) = args.token_file {
        let t = token::read_token(path)?;
        return Ok(Some(t));
    }
    if args.no_auth {
        return Ok(None);
    }
    let default_path = default_token_path();
    if default_path.exists() {
        let t = token::read_token(&default_path)?;
        tracing::info!("Using token from {}", default_path.display());
        return Ok(Some(t));
    }
    Ok(None)
}

fn run_token(args: TokenArgs) -> anyhow::Result<()> {
    match args.action {
        TokenAction::Generate { output } => {
            let path = if let Some(rest) = output.strip_prefix("~/") {
                let home = dirs::home_dir()
                    .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
                home.join(rest)
            } else {
                PathBuf::from(&output)
            };

            let new_token = token::generate_token();
            token::write_token(&path, &new_token)?;

            eprintln!("Token saved to {}", path.display());
            eprintln!("Token: {}", new_token);
            eprintln!();
            eprintln!("Use with: wenlan-mcp serve --token-file {}", path.display());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_stdio_agent_name_override() {
        let cli = Cli::try_parse_from(["wenlan-mcp", "--agent-name", "codex"])
            .expect("parse stdio agent override");

        assert_eq!(cli.agent_name.as_deref(), Some("codex"));
        assert!(cli.command.is_none());
    }

    #[test]
    fn serve_default_agent_name_stays_remote_mcp() {
        let cli = Cli::try_parse_from(["wenlan-mcp", "serve", "--no-auth"])
            .expect("parse serve defaults");

        let Some(Commands::Serve(args)) = cli.command else {
            panic!("expected serve command");
        };

        assert_eq!(
            effective_agent_name(cli.agent_name, Some(&args)),
            "remote-mcp"
        );
    }

    #[test]
    fn serve_accepts_global_agent_name_override() {
        let cli = Cli::try_parse_from([
            "wenlan-mcp",
            "serve",
            "--no-auth",
            "--agent-name",
            "chatgpt",
        ])
        .expect("parse serve agent override");

        let Some(Commands::Serve(args)) = cli.command else {
            panic!("expected serve command");
        };

        assert_eq!(effective_agent_name(cli.agent_name, Some(&args)), "chatgpt");
    }
}
