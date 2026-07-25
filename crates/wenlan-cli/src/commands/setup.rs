// SPDX-License-Identifier: Apache-2.0
//! Human-facing setup/status commands for the Wenlan runtime.

use clap::{Subcommand, ValueEnum};
use std::io::{self, Write};
use wenlan_core::{config, on_device_models};

use crate::client::origin_host_from_env;

#[derive(Clone, Debug)]
pub struct SetupArgs {
    pub basic: bool,
    pub model: Option<String>,
    pub anthropic_api_key_env: Option<String>,
    pub yes: bool,
}

#[derive(Subcommand)]
pub enum ModelCommand {
    /// List local models Wenlan can download and run.
    List,
    /// Show selected/downloaded local model state.
    Status,
    /// Download and select a local model.
    Install {
        /// Model id, for example qwen3-4b.
        model_id: Option<String>,
        /// Skip confirmation before downloading.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Set the cross-encoder reranker mode (off/lite/full).
    Reranker {
        /// Mode to persist; the daemon reads it at startup.
        mode: RerankerModeArg,
    },
}

#[derive(Subcommand)]
pub enum KeyCommand {
    /// Show API key status.
    Status,
    /// Store an API key.
    Set {
        /// Provider to configure.
        provider: KeyProvider,
        /// Read the key from this environment variable instead of prompting.
        #[arg(long = "env", value_name = "ENV_VAR")]
        env_var: Option<String>,
    },
    /// Clear a stored API key.
    Clear {
        /// Provider to clear.
        provider: KeyProvider,
    },
}

#[derive(Subcommand)]
pub enum EnrichmentCommand {
    /// Show the effective source and ready/paused/off state for each job group.
    Status,
    /// Confirm exact sources and enable model-backed background work.
    Configure {
        /// Source for background organization.
        #[arg(long, value_enum)]
        everyday: EnrichmentSource,
        /// Source for page synthesis.
        #[arg(long, value_enum)]
        synthesis: EnrichmentSource,
        /// Confirm the mapping supplied on this command line.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Turn off all model-backed background work without removing providers.
    Disable,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum EnrichmentSource {
    Anthropic,
    External,
    OnDevice,
}

impl EnrichmentSource {
    fn as_pin(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::External => "external",
            Self::OnDevice => "on_device",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic",
            Self::External => "configured external provider",
            Self::OnDevice => "on-device model",
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
pub enum KeyProvider {
    Anthropic,
}

/// Cross-encoder reranker mode, persisted to config and read by the daemon at
/// startup. Mirrors the core `RerankerMode` (off/lite/full).
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum RerankerModeArg {
    /// No cross-encoder on any path (default).
    Off,
    /// Turbo CE (jina-turbo, ~146MB) on quick + context + explicit deep rerank.
    Lite,
    /// Turbo on the light paths; heavy bge-base (~1.1GB) on the deep rerank.
    Full,
}

impl RerankerModeArg {
    fn as_str(self) -> &'static str {
        match self {
            RerankerModeArg::Off => "off",
            RerankerModeArg::Lite => "lite",
            RerankerModeArg::Full => "full",
        }
    }
}

pub async fn run_setup(args: SetupArgs) -> anyhow::Result<()> {
    if args.basic {
        configure_basic_memory()?;
        println!("Wenlan is set up for local memory.");
        println!("Storage, search, recall, and MCP memory work without a local model or API key.");
        println!("Model-backed background enrichment is off.");
        return Ok(());
    }

    if let Some(model_id) = args.model {
        install_model(&model_id, args.yes).await?;
        mark_setup_completed()?;
        print_enrichment_opt_in("on-device");
        return Ok(());
    }

    if let Some(env_name) = args.anthropic_api_key_env {
        let key = std::env::var(&env_name)
            .map_err(|_| anyhow::anyhow!("environment variable {} is not set", env_name))?;
        set_anthropic_key(key).await?;
        mark_setup_completed()?;
        print_enrichment_opt_in("Anthropic");
        return Ok(());
    }

    interactive_setup().await
}

pub async fn run_model(command: ModelCommand) -> anyhow::Result<()> {
    match command {
        ModelCommand::List => {
            print_model_list();
            Ok(())
        }
        ModelCommand::Status => {
            print_model_status();
            Ok(())
        }
        ModelCommand::Install { model_id, yes } => {
            let id = model_id.unwrap_or_else(|| on_device_models::get_default_model().id.into());
            install_model(&id, yes).await?;
            print_enrichment_opt_in("on-device");
            Ok(())
        }
        ModelCommand::Reranker { mode } => run_reranker(mode).await,
    }
}

pub async fn run_key(command: KeyCommand) -> anyhow::Result<()> {
    match command {
        KeyCommand::Status => {
            print_key_status();
            Ok(())
        }
        KeyCommand::Set {
            provider: KeyProvider::Anthropic,
            env_var,
        } => {
            let key = match env_var {
                Some(name) => std::env::var(&name)
                    .map_err(|_| anyhow::anyhow!("environment variable {} is not set", name))?,
                None => prompt_secret("Anthropic API key: ")?,
            };
            set_anthropic_key(key).await?;
            print_enrichment_opt_in("Anthropic");
            Ok(())
        }
        KeyCommand::Clear {
            provider: KeyProvider::Anthropic,
        } => clear_anthropic_key().await,
    }
}

pub async fn run_enrichment(command: EnrichmentCommand) -> anyhow::Result<()> {
    match command {
        EnrichmentCommand::Status => {
            print_enrichment_status().await;
            Ok(())
        }
        EnrichmentCommand::Configure {
            everyday,
            synthesis,
            yes,
        } => configure_enrichment(everyday, synthesis, yes).await,
        EnrichmentCommand::Disable => disable_enrichment().await,
    }
}

/// Persist the cross-encoder reranker mode to config. The daemon reads it at
/// startup via `reranker_mode_resolved` (the `WENLAN_RERANKER_MODE` env var
/// still overrides it). Model weights download lazily on first use after restart
/// — `full`'s heavy bge-base loads in the background so startup never blocks.
pub async fn run_reranker(mode: RerankerModeArg) -> anyhow::Result<()> {
    let mode_str = mode.as_str();
    let mut cfg = config::load_config();
    cfg.reranker_mode = Some(mode_str.to_string());
    config::save_config(&cfg)?;
    println!("Reranker mode set to '{mode_str}'.");
    match mode {
        RerankerModeArg::Off => println!("Cross-encoder rerank is disabled."),
        RerankerModeArg::Lite => {
            println!(
                "Turbo cross-encoder (~146MB) on quick + context paths; downloads on first query."
            );
        }
        RerankerModeArg::Full => {
            println!("Turbo (~146MB) on light paths + heavy bge-base (~1.1GB) on deep rerank.");
            println!("The deep model downloads in the background after restart.");
        }
    }
    println!("Run `wenlan restart` to apply.");
    Ok(())
}

pub async fn run_doctor() -> anyhow::Result<()> {
    println!("Wenlan doctor");
    println!();
    print_daemon_health().await;
    #[cfg(target_os = "macos")]
    print_daemon_log_paths();
    print_key_status();
    print_model_status();
    print_reranker_status().await;

    let cfg = config::load_config();
    let has_key = cfg
        .anthropic_api_key
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    let has_cached_model = configured_model()
        .map(on_device_models::is_cached)
        .unwrap_or(false);

    println!();
    if has_key || has_cached_model {
        println!("Model provider: available for explicit foreground use.");
        println!("  Provider availability does not authorize background inference.");
    } else {
        println!("Model provider: none configured.");
        println!("  Run: wenlan models install");
        println!("  Or:  wenlan keys set anthropic");
    }
    print_enrichment_status().await;

    let cwd = std::env::current_dir()?;
    print_space_resolution(&cwd);

    Ok(())
}

#[cfg(target_os = "macos")]
fn print_daemon_log_paths() {
    let data_root = std::env::var_os("WENLAN_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("wenlan")
        });
    let fallback_root = dirs::home_dir()
        .map(|home| home.join("Library/Logs/com.wenlan.server-fallback"))
        .unwrap_or_else(|| std::env::temp_dir().join("wenlan-server-fallback"));
    println!(
        "Daemon log: {}",
        data_root.join("logs/wenlan-server.log").display()
    );
    println!(
        "Bootstrap log: {}",
        data_root.join("logs/wenlan-server.bootstrap.log").display()
    );
    println!(
        "Bootstrap fallback log: {}",
        fallback_root
            .join("logs/wenlan-server.bootstrap.log")
            .display()
    );
    println!(
        "Daemon fallback log: {}",
        fallback_root.join("logs/wenlan-server.log").display()
    );
}

pub async fn print_runtime_status() -> anyhow::Result<()> {
    print_key_status();
    print_model_status();
    print_reranker_status().await;
    print_enrichment_status().await;
    Ok(())
}

async fn interactive_setup() -> anyhow::Result<()> {
    println!("Set up Wenlan");
    println!();
    println!("1) Local Memory");
    println!("   Store, search, recall, and MCP memory. No local model or API key.");
    println!("2) Local Model");
    println!("   Download a local model for private distill cycles.");
    println!("3) Anthropic Key");
    println!("   Use your Anthropic API key for stronger distill cycles. Memory stays local.");
    println!();

    let choice = prompt_line("Choose 1, 2, or 3 [1]: ")?;
    match choice.trim() {
        "" | "1" => {
            configure_basic_memory()?;
            println!("Wenlan is set up for local memory.");
            Ok(())
        }
        "2" => {
            let default = on_device_models::get_default_model();
            print_model_list();
            let input = prompt_line(&format!("Model id [{}]: ", default.id))?;
            let model_id = if input.trim().is_empty() {
                default.id
            } else {
                input.trim()
            };
            install_model(model_id, false).await?;
            print_enrichment_opt_in("on-device");
            mark_setup_completed()
        }
        "3" => {
            let key = prompt_secret("Anthropic API key: ")?;
            set_anthropic_key(key).await?;
            print_enrichment_opt_in("Anthropic");
            mark_setup_completed()
        }
        other => Err(anyhow::anyhow!("unknown setup choice: {}", other)),
    }
}

fn print_model_list() {
    let cfg = config::load_config();
    let selected = cfg
        .on_device_model
        .as_deref()
        .map(|id| on_device_models::resolve_or_default(Some(id)))
        .map(|model| model.id);
    for model in on_device_models::MODELS {
        let cached = if on_device_models::is_cached(model) {
            "downloaded"
        } else {
            "not downloaded"
        };
        let marker = if Some(model.id) == selected { "*" } else { " " };
        println!(
            "{} {} ({}, {:.1}GB download, needs {:.0}GB RAM) - {}",
            marker, model.id, model.display_name, model.file_size_gb, model.ram_required_gb, cached
        );
    }
}

fn print_model_status() {
    let Some(selected) = configured_model() else {
        println!("Local model: not selected");
        return;
    };
    let cached = on_device_models::is_cached(selected);
    println!(
        "Local model: {} ({})",
        selected.id,
        if cached {
            "downloaded"
        } else {
            "not downloaded"
        }
    );
}

async fn install_model(model_id: &str, yes: bool) -> anyhow::Result<()> {
    let model = on_device_models::get_model(model_id)
        .ok_or_else(|| anyhow::anyhow!("unknown model id: {}", model_id))?;

    if !on_device_models::is_cached(model) && !yes {
        println!(
            "{} is a {:.1}GB download and needs about {:.0}GB RAM.",
            model.display_name, model.file_size_gb, model.ram_required_gb
        );
        let answer = prompt_line("Download now? [y/N]: ")?;
        if !matches!(answer.trim(), "y" | "Y" | "yes" | "YES") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let body = serde_json::json!({ "model_id": model.id });
    match post_json("/api/on-device-model/download", &body).await {
        Ok(_) => {
            println!("Local model downloaded and loaded: {}", model.id);
            Ok(())
        }
        Err(http_err) => {
            println!("Daemon not available for hot-load ({}).", http_err);
            println!("Downloading directly, then the daemon will load it on next start.");
            tokio::task::spawn_blocking(move || {
                wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some(model.id))
            })
            .await??;
            let mut cfg = config::load_config();
            cfg.setup_completed = true;
            cfg.on_device_model = Some(model.id.to_string());
            config::save_config(&cfg)?;
            println!("Local model ready: {}", model.id);
            Ok(())
        }
    }
}

fn print_key_status() {
    let cfg = config::load_config();
    let configured = cfg
        .anthropic_api_key
        .as_deref()
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    println!(
        "Anthropic key: {}",
        if configured {
            "configured"
        } else {
            "not configured"
        }
    );
}

async fn set_anthropic_key(key: String) -> anyhow::Result<()> {
    let key = key.trim().to_string();
    if key.is_empty() {
        return Err(anyhow::anyhow!("API key cannot be empty"));
    }

    let body = serde_json::json!({ "api_key": key });
    match put_json("/api/setup/anthropic-key", &body).await {
        Ok(_) => println!("Anthropic key saved and active in the running daemon."),
        Err(_) => {
            let mut cfg = config::load_config();
            cfg.setup_completed = true;
            cfg.anthropic_api_key = Some(body["api_key"].as_str().unwrap().to_string());
            config::save_config(&cfg)?;
            println!("Anthropic key saved. Start or restart the daemon to activate it.");
        }
    }
    Ok(())
}

async fn clear_anthropic_key() -> anyhow::Result<()> {
    match delete("/api/setup/anthropic-key").await {
        Ok(_) => println!("Anthropic key cleared from the running daemon."),
        Err(_) => {
            let mut cfg = config::load_config();
            cfg.anthropic_api_key = None;
            config::save_config(&cfg)?;
            println!("Anthropic key cleared. Start or restart the daemon to apply the change.");
        }
    }
    Ok(())
}

fn mark_setup_completed() -> anyhow::Result<()> {
    let mut cfg = config::load_config();
    cfg.setup_completed = true;
    config::save_config(&cfg)?;
    Ok(())
}

fn configure_basic_memory() -> anyhow::Result<()> {
    let mut cfg = config::load_config();
    cfg.setup_completed = true;
    cfg.on_device_model = None;
    cfg.anthropic_api_key = None;
    cfg.everyday_source = None;
    cfg.synthesis_source = None;
    config::save_config(&cfg)?;
    Ok(())
}

fn print_enrichment_opt_in(source: &str) {
    println!("{source} is available, but model-backed background enrichment is still off.");
    println!(
        "To review the exact task mapping and enable it, run `wenlan enrichment configure --help`."
    );
}

async fn configure_enrichment(
    everyday: EnrichmentSource,
    synthesis: EnrichmentSource,
    yes: bool,
) -> anyhow::Result<()> {
    let routing = get_json("/api/config/routing").await.map_err(|err| {
        anyhow::anyhow!(
            "the running Wenlan runtime does not support background consent status: {err}"
        )
    })?;
    require_configured_source(&routing, everyday)?;
    require_configured_source(&routing, synthesis)?;

    println!("Enable model-backed background enrichment?");
    println!(
        "  Everyday organization: {} (classify, structure, entities, links, titles, citations, and page inputs)",
        everyday.label()
    );
    println!(
        "  Page synthesis:       {} (draft and update synthesized wiki pages)",
        synthesis.label()
    );
    if matches!(everyday, EnrichmentSource::OnDevice)
        || matches!(synthesis, EnrichmentSource::OnDevice)
    {
        println!(
            "  On-device work uses CPU/GPU/RAM and may briefly warm the machine. Wenlan admits one bounded inference at a time and pauses new work when the host is busy."
        );
    }
    if matches!(everyday, EnrichmentSource::Anthropic)
        || matches!(synthesis, EnrichmentSource::Anthropic)
    {
        println!(
            "  Anthropic receives relevant memory content and may charge your provider account."
        );
    }
    if matches!(everyday, EnrichmentSource::External)
        || matches!(synthesis, EnrichmentSource::External)
    {
        println!(
            "  The configured external endpoint receives relevant memory content; remote providers may charge your account."
        );
    }

    if !yes {
        let answer =
            prompt_line("Write these two hard pins and enable background enrichment? [y/N]: ")?;
        if !matches!(answer.trim(), "y" | "Y" | "yes" | "YES") {
            println!("Background enrichment remains off.");
            return Ok(());
        }
    }

    let body = serde_json::json!({
        "everyday_source": everyday.as_pin(),
        "synthesis_source": synthesis.as_pin(),
    });
    put_json("/api/config", &body).await?;
    println!("Background enrichment consent saved.");
    print_enrichment_status().await;
    Ok(())
}

async fn disable_enrichment() -> anyhow::Result<()> {
    let body = serde_json::json!({
        "everyday_source": "",
        "synthesis_source": "",
    });
    match put_json("/api/config", &body).await {
        Ok(_) => {
            println!("Background enrichment disabled. Providers and downloaded models were kept.");
        }
        Err(err)
            if err
                .downcast_ref::<reqwest::Error>()
                .is_some_and(reqwest::Error::is_connect) =>
        {
            let mut cfg = config::load_config();
            cfg.everyday_source = None;
            cfg.synthesis_source = None;
            config::save_config(&cfg)?;
            println!(
                "Background enrichment disabled in local config. Restart Wenlan if an older daemon is running."
            );
        }
        Err(err) => {
            return Err(anyhow::anyhow!(
                "the running Wenlan daemon rejected the disable request; no local pins were changed: {err}"
            ));
        }
    }
    Ok(())
}

fn require_configured_source(
    routing: &serde_json::Value,
    source: EnrichmentSource,
) -> anyhow::Result<()> {
    let pool = &routing["pool"];
    let configured = match source {
        EnrichmentSource::Anthropic => pool["anthropic"]["configured"].as_bool() == Some(true),
        EnrichmentSource::External => !pool["external"].is_null(),
        EnrichmentSource::OnDevice => !pool["on_device"].is_null(),
    };
    anyhow::ensure!(
        configured,
        "{} is not configured; install or connect it first. No background pins were changed",
        source.label()
    );
    Ok(())
}

pub async fn print_enrichment_status() {
    match get_json("/api/config/routing").await {
        Ok(routing) => {
            println!("Background enrichment:");
            print_job_route("Everyday organization", &routing["everyday"]);
            print_job_route("Page synthesis", &routing["synthesis"]);
        }
        Err(err) => {
            let cfg = config::load_config();
            if cfg.everyday_source.is_some() || cfg.synthesis_source.is_some() {
                println!("Background enrichment: status unavailable ({err})");
                println!("  Pins exist on disk, but the running daemon could not verify them.");
            } else {
                println!("Background enrichment: off (effective status unavailable: {err})");
            }
        }
    }
}

fn print_job_route(label: &str, route: &serde_json::Value) {
    let mode = route["mode"].as_str().unwrap_or("unsupported");
    let source = route["pin"]
        .as_str()
        .or_else(|| route["source"].as_str())
        .unwrap_or("none");
    let state = match mode {
        "pinned" => "ready",
        "pinned_unavailable" => "paused (exact source unavailable; no fallback)",
        "unconfigured" => "off",
        _ => "unsupported daemon state",
    };
    println!("  {label}: {state} [{source}]");
}

fn configured_model() -> Option<&'static on_device_models::OnDeviceModel> {
    let cfg = config::load_config();
    cfg.on_device_model
        .as_deref()
        .map(|id| on_device_models::resolve_or_default(Some(id)))
}

async fn print_daemon_health() {
    let url = origin_url("/api/health");
    match reqwest::get(&url).await {
        Ok(resp) if resp.status().is_success() => println!("Daemon: running on {}", url),
        Ok(resp) => println!("Daemon: unhealthy ({})", resp.status()),
        Err(_) => println!("Daemon: not reachable on {}", url),
    }
}

/// Fetch `/api/status` and print the per-path reranker summary (mode + deep + light).
/// Silent on network error — the daemon being down is already surfaced by the
/// preceding health check.
async fn print_reranker_status() {
    use wenlan_types::responses::RerankerStatus;
    let url = origin_url("/api/status");
    let Ok(resp) = reqwest::get(&url).await else {
        return;
    };
    let Ok(status) = resp.json::<wenlan_types::responses::StatusResponse>().await else {
        return;
    };
    let fmt = |s: &RerankerStatus| match s {
        RerankerStatus::Disabled => "disabled".to_string(),
        RerankerStatus::Active { model_id } => format!("active ({model_id})"),
        RerankerStatus::Failed { reason } => format!("failed ({reason})"),
    };
    let mode = if status.reranker_mode.is_empty() {
        "off"
    } else {
        status.reranker_mode.as_str()
    };
    println!("Reranker mode: {mode}");
    println!(
        "  deep  (/api/memory/search rerank=true): {}",
        fmt(&status.reranker)
    );
    println!(
        "  light (/api/search + /api/context): {}",
        fmt(&status.reranker_light)
    );
    if mode == "off" {
        println!("  (set WENLAN_RERANKER_MODE=lite|full to enable cross-encoder rerank)");
    }
}

fn origin_url(path: &str) -> String {
    format!("{}{}", origin_host_from_env(), path)
}

async fn get_json(path: &str) -> anyhow::Result<serde_json::Value> {
    let resp = reqwest::Client::new().get(origin_url(path)).send().await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }
    Ok(resp.json().await?)
}

async fn post_json(path: &str, body: &serde_json::Value) -> anyhow::Result<serde_json::Value> {
    let resp = reqwest::Client::new()
        .post(origin_url(path))
        .json(body)
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }
    Ok(resp.json().await?)
}

async fn put_json(path: &str, body: &serde_json::Value) -> anyhow::Result<serde_json::Value> {
    let resp = reqwest::Client::new()
        .put(origin_url(path))
        .json(body)
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }
    Ok(resp.json().await?)
}

async fn delete(path: &str) -> anyhow::Result<()> {
    let resp = reqwest::Client::new()
        .delete(origin_url(path))
        .send()
        .await?;
    if !resp.status().is_success() {
        return Err(anyhow::anyhow!("HTTP {}", resp.status()));
    }
    Ok(())
}

fn print_space_resolution(cwd: &std::path::Path) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "\n--- Space resolution ---");

    let env = std::env::var("WENLAN_SPACE").ok().filter(|s| !s.is_empty());
    let _ = writeln!(
        out,
        "WENLAN_SPACE env:      {}",
        env.as_deref().unwrap_or("(unset)")
    );

    let cfg = dirs::home_dir().map(|h| h.join(".wenlan/spaces.toml"));
    let cfg_exists = cfg.as_ref().map(|p| p.exists()).unwrap_or(false);
    let _ = writeln!(
        out,
        "~/.wenlan/spaces.toml: {}",
        if cfg_exists { "present" } else { "missing" }
    );

    let _ = writeln!(out, "cwd:                   {}", cwd.display());

    let plugin_resolver = std::env::var("CLAUDE_PLUGIN_ROOT")
        .ok()
        .map(|p| format!("{}/bin/resolve-space.sh", p));
    if let Some(p) = plugin_resolver {
        if std::path::Path::new(&p).exists() {
            let _ = writeln!(out, "Plugin resolver:       {}", p);
            let output = std::process::Command::new(&p)
                .arg("--cwd")
                .arg(cwd)
                .output();
            if let Ok(o) = output {
                let s = String::from_utf8_lossy(&o.stdout);
                let s = s.trim().replace('\t', " (from ");
                let s = if s.contains(" (from ") {
                    format!("{})", s)
                } else {
                    s.to_string()
                };
                let _ = writeln!(out, "Resolved:              {}", s);
            }
        } else {
            let _ = writeln!(out, "Plugin resolver:       not found at {}", p);
        }
    } else {
        let _ = writeln!(
            out,
            "Plugin resolver:       CLAUDE_PLUGIN_ROOT not set (running outside Claude Code)"
        );
    }
}

fn prompt_line(prompt: &str) -> anyhow::Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut value = String::new();
    io::stdin().read_line(&mut value)?;
    Ok(value)
}

fn prompt_secret(prompt: &str) -> anyhow::Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let _ = std::process::Command::new("stty").arg("-echo").status();
    let mut value = String::new();
    let read = io::stdin().read_line(&mut value);
    let _ = std::process::Command::new("stty").arg("echo").status();
    println!();
    read?;
    Ok(value)
}
