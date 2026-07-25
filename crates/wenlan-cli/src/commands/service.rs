// SPDX-License-Identifier: Apache-2.0
//! Cross-platform service registration for the Wenlan daemon.
//!
//! - macOS: launchd LaunchAgent via the `service-manager` crate.
//! - Linux: systemd --user unit via the `service-manager` crate.
//! - Windows: per-user ONLOGON Task Scheduler entry via `schtasks.exe`.
//!   We bypass `service-manager`'s `ScServiceManager` because wenlan-server
//!   is a plain console app and does not implement the Windows Service
//!   Control Protocol (`sc start` would time out at 30s with error 1053).

use anyhow::{Context, Result};
use service_manager::{
    ServiceInstallCtx, ServiceLabel, ServiceLevel, ServiceManager, ServiceStartCtx, ServiceStopCtx,
};
use std::path::{Path, PathBuf};

use crate::client::origin_host_from_env;

pub const SERVICE_LABEL: &str = "com.wenlan.server";
const DEFAULT_LOCAL_BIND_ADDR: &str = "127.0.0.1:7878";
const SHUTDOWN_PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);
const SHUTDOWN_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);
const SHUTDOWN_STABILITY_WINDOW: std::time::Duration = std::time::Duration::from_secs(1);
const SHUTDOWN_VERIFY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);
const DAEMON_PROCESS_ID_HEADER: &str = "x-wenlan-process-id";

#[derive(Clone, Copy, Debug)]
struct DaemonProcessIdentity {
    pid: sysinfo::Pid,
    started_at: Option<u64>,
}

impl DaemonProcessIdentity {
    fn capture(raw_pid: u32) -> Self {
        let pid = sysinfo::Pid::from_u32(raw_pid);
        let mut system = sysinfo::System::new();
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
        Self {
            pid,
            started_at: system.process(pid).map(sysinfo::Process::start_time),
        }
    }

    fn is_running(self, system: &mut sysinfo::System) -> bool {
        system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[self.pid]), true);
        match (self.started_at, system.process(self.pid)) {
            (Some(started_at), Some(process)) => process.start_time() == started_at,
            // The process had already exited by the time the CLI captured its
            // identity. Do not let PID reuse turn that completed exit back
            // into a live match.
            (None, _) | (_, None) => false,
        }
    }
}

enum DaemonShutdownRequest {
    NotRunning,
    Requested(Option<DaemonProcessIdentity>),
}

/// Windows Task Scheduler does not love dots in task names. The macOS launchd
/// and systemd-user paths still use the canonical reverse-DNS `SERVICE_LABEL`.
#[cfg(target_os = "windows")]
pub const WINDOWS_TASK_NAME: &str = "WenlanServer";

#[derive(clap::Subcommand)]
pub enum BackgroundCommand {
    /// Start Wenlan now and keep it running in the background after login.
    On,
    /// Stop Wenlan now while keeping its background registration.
    Off,
}

pub async fn run_background(command: BackgroundCommand) -> Result<()> {
    match command {
        BackgroundCommand::On => install(),
        BackgroundCommand::Off => stop().await,
    }
}

fn label() -> Result<ServiceLabel> {
    SERVICE_LABEL.parse().context("invalid service label")
}

#[cfg(target_os = "windows")]
fn run_schtasks(args: &[&str], action: &str) -> Result<std::process::Output> {
    let output = std::process::Command::new("schtasks.exe")
        .args(args)
        .output()
        .with_context(|| format!("spawn schtasks.exe ({action})"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::bail!(
            "schtasks.exe {} failed (exit {}): {}{}",
            action,
            output.status.code().unwrap_or(-1),
            stderr.trim(),
            if stdout.trim().is_empty() {
                String::new()
            } else {
                format!("\nstdout: {}", stdout.trim())
            }
        );
    }
    Ok(output)
}

fn manager() -> Result<Box<dyn ServiceManager>> {
    // macOS + Linux only. Windows install/stop short-circuit before
    // calling this and drive schtasks.exe directly (see install/stop).
    let mut m = <dyn ServiceManager>::native().context("detect native service manager")?;
    let _ = m.set_level(ServiceLevel::User);
    Ok(m)
}

/// Resolves the platform-specific path to the Wenlan service unit file.
///
/// Mirrors the on-disk path that `service-manager` 0.11 actually writes:
/// - macOS (launchd): `~/Library/LaunchAgents/<qualified_name>.plist`
///   (`to_qualified_name()` keeps the qualifier, e.g. `com.wenlan.server.plist`).
/// - Linux (systemd-user): `<config_dir>/systemd/user/<script_name>.service`
///   (`ServiceLabel::to_script_name()` joins org+app with `-` and DROPS the
///   qualifier, so `com.wenlan.server` becomes `wenlan-server.service`).
/// - Windows: no on-disk unit file. The scheduled task lives in the Task
///   Scheduler database — see `is_installed()` for the schtasks-based probe.
#[cfg(not(target_os = "windows"))]
pub fn service_unit_path() -> Result<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        Ok(dirs::home_dir()
            .context("HOME not set")?
            .join("Library/LaunchAgents")
            .join(format!("{}.plist", SERVICE_LABEL)))
    }
    #[cfg(target_os = "linux")]
    {
        let label = label()?;
        Ok(dirs::config_dir()
            .context("XDG_CONFIG_HOME not set")?
            .join("systemd/user")
            .join(format!("{}.service", label.to_script_name())))
    }
}

fn current_server_path() -> Result<PathBuf> {
    let origin_exe = std::env::current_exe().context("cannot determine origin CLI path")?;
    let mut server = origin_exe
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("wenlan-server");
    if cfg!(target_os = "windows") {
        server.set_extension("exe");
    }
    if !server.exists() {
        anyhow::bail!(
            "wenlan-server not found next to origin at {}. Re-run the Wenlan installer.",
            server.display()
        );
    }
    Ok(server)
}

/// Resolves the data root the daemon will use at runtime. Mirrors
/// `crates/wenlan-server/src/main.rs` so launchd log paths line up with the
/// on-disk layout the daemon owns. macOS-only because launchd is the only
/// service backend that consumes pre-rendered log paths today.
#[cfg(target_os = "macos")]
fn origin_data_root() -> PathBuf {
    std::env::var_os("WENLAN_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("wenlan")
        })
}

/// Builds a launchd plist that mirrors `service-manager`'s default output for
/// `OnFailure` restart + user-level + autostart, with the extra keys the old
/// embedded `com.wenlan.server.plist` template carried: stdout/stderr routing,
/// `EnvironmentVariables.RUST_LOG`, and the canonical `WENLAN_DATA_DIR`
/// ownership marker consumed by the desktop app. The daemon owns bounded file
/// logging, so launchd routes its raw streams to `/dev/null`.
///
/// `LaunchdInstallConfig` in service-manager 0.11 only exposes `keep_alive`;
/// stdout/stderr paths must come through `ServiceInstallCtx.contents` as a
/// pre-rendered plist string. This function is the minimal stand-in for the
/// crate's internal `make_plist`.
#[cfg(target_os = "macos")]
fn build_launchd_plist(
    program: &Path,
    stdout_path: &Path,
    stderr_path: &Path,
    rust_log: &str,
    data_root: &Path,
) -> String {
    let mut buf = String::new();
    buf.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    buf.push_str(
        "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n",
    );
    buf.push_str("<plist version=\"1.0\">\n<dict>\n");
    buf.push_str("\t<key>Label</key>\n");
    buf.push_str(&format!("\t<string>{}</string>\n", SERVICE_LABEL));
    buf.push_str("\t<key>ProgramArguments</key>\n");
    buf.push_str("\t<array>\n");
    buf.push_str(&format!(
        "\t\t<string>{}</string>\n",
        program.to_string_lossy()
    ));
    buf.push_str("\t</array>\n");
    // Mirrors service-manager's RestartPolicy::OnFailure rendering: KeepAlive
    // dict with SuccessfulExit=false. The matching `Disabled` key keeps the
    // service from auto-loading until start() removes it (cross-platform parity).
    buf.push_str("\t<key>KeepAlive</key>\n");
    buf.push_str("\t<dict>\n");
    buf.push_str("\t\t<key>SuccessfulExit</key>\n");
    buf.push_str("\t\t<false/>\n");
    buf.push_str("\t</dict>\n");
    buf.push_str("\t<key>RunAtLoad</key>\n\t<true/>\n");
    buf.push_str("\t<key>Disabled</key>\n\t<true/>\n");
    buf.push_str("\t<key>StandardOutPath</key>\n");
    buf.push_str(&format!(
        "\t<string>{}</string>\n",
        stdout_path.to_string_lossy()
    ));
    buf.push_str("\t<key>StandardErrorPath</key>\n");
    buf.push_str(&format!(
        "\t<string>{}</string>\n",
        stderr_path.to_string_lossy()
    ));
    buf.push_str("\t<key>EnvironmentVariables</key>\n");
    buf.push_str("\t<dict>\n");
    buf.push_str("\t\t<key>RUST_LOG</key>\n");
    buf.push_str(&format!("\t\t<string>{}</string>\n", rust_log));
    buf.push_str("\t\t<key>WENLAN_DATA_DIR</key>\n");
    buf.push_str(&format!(
        "\t\t<string>{}</string>\n",
        data_root.to_string_lossy()
    ));
    buf.push_str("\t</dict>\n");
    buf.push_str("</dict>\n</plist>\n");
    buf
}

pub fn install() -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        // wenlan-server is a plain console app and does not speak the Windows
        // Service Control Protocol, so sc.exe install + start would time out
        // at 30s with error 1053. Use Task Scheduler instead: register a
        // per-user ONLOGON task and trigger it immediately. Matches the
        // user-scope semantics of launchd LaunchAgent (macOS) and
        // systemd --user (Linux), without needing a service dispatcher in
        // wenlan-server.
        let program = current_server_path()?;
        let program_str = program.to_string_lossy();
        let _ = std::process::Command::new("schtasks.exe")
            .args(["/end", "/tn", WINDOWS_TASK_NAME])
            .output();
        run_schtasks(
            &[
                "/create",
                "/tn",
                WINDOWS_TASK_NAME,
                "/sc",
                "ONLOGON",
                "/tr",
                &program_str,
                "/f",
            ],
            "create scheduled task",
        )?;
        run_schtasks(&["/run", "/tn", WINDOWS_TASK_NAME], "run scheduled task")?;
        println!(
            "Installed and started Windows scheduled task '{}' (wenlan-server).",
            WINDOWS_TASK_NAME
        );
        return Ok(());
    }

    #[cfg_attr(target_os = "windows", allow(unreachable_code))]
    let label_value = label()?;
    let program = current_server_path()?;
    let m = manager()?;

    // Stop any daemon already running under this label so the reinstall swaps
    // the binary. Without this, the freshly-installed binary detects the
    // healthy incumbent on port 7878 and exits, leaving the OLD daemon running
    // (wenlan-server/src/main.rs:582-615). Best-effort: errors if not running.
    let _ = m.stop(ServiceStopCtx {
        label: label_value.clone(),
    });

    // Apply RUST_LOG=info to every platform. launchd consumes
    // `EnvironmentVariables`, systemd-user consumes `Environment=`. winsw +
    // sc.exe ignore the field (Windows daemons still rely on `RUST_LOG`
    // exported in the user environment).
    let environment = Some(vec![("RUST_LOG".to_string(), "info".to_string())]);

    // macOS: hand-roll the plist so launchd does not append an unbounded
    // duplicate of the daemon-owned rotating log. service-manager 0.11 has no
    // struct field for these keys, so the only honest knob is
    // `ServiceInstallCtx.contents`.
    let contents = {
        #[cfg(target_os = "macos")]
        {
            let data_root = origin_data_root();
            Some(build_launchd_plist(
                &program,
                Path::new("/dev/null"),
                Path::new("/dev/null"),
                "info",
                &data_root,
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    };

    m.install(ServiceInstallCtx {
        label: label_value.clone(),
        program,
        args: vec![],
        contents,
        username: None,
        working_directory: None,
        environment,
        autostart: true,
        restart_policy: service_manager::RestartPolicy::OnFailure {
            delay_secs: None,
            max_retries: None,
            reset_after_secs: None,
        },
    })
    .context("install service")?;

    m.start(ServiceStartCtx { label: label_value })
        .context("start service")?;
    println!("Installed and started {}.", SERVICE_LABEL);
    Ok(())
}

#[cfg(target_os = "macos")]
fn current_user_id() -> Result<String> {
    let output = std::process::Command::new("id")
        .arg("-u")
        .output()
        .context("run id -u for launchd user domain")?;
    if !output.status.success() {
        anyhow::bail!("id -u failed (exit {})", output.status.code().unwrap_or(-1));
    }
    let uid = std::str::from_utf8(&output.stdout)
        .context("id -u returned non-UTF-8 output")?
        .trim();
    if uid.is_empty() || !uid.bytes().all(|byte| byte.is_ascii_digit()) {
        anyhow::bail!("id -u returned invalid user id: {uid:?}");
    }
    Ok(uid.to_owned())
}

fn stop_registered_service() -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        // /end returns nonzero when the registered task is not currently
        // running. Preserve idempotence and, critically, never /delete it.
        let _ = std::process::Command::new("schtasks.exe")
            .args(["/end", "/tn", WINDOWS_TASK_NAME])
            .output()
            .context("spawn schtasks.exe (end scheduled task)")?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    {
        let uid = current_user_id()?;
        let domain = format!("gui/{uid}");
        let plist = service_unit_path()?;
        let bootout = std::process::Command::new("launchctl")
            .arg("bootout")
            .arg(&domain)
            .arg(&plist)
            .output()
            .context("spawn launchctl bootout")?;

        if !bootout.status.success() {
            let target = format!("{domain}/{SERVICE_LABEL}");
            let status = std::process::Command::new("launchctl")
                .args(["print", &target])
                .output()
                .context("spawn launchctl print after failed bootout")?;
            if status.status.code() != Some(113) {
                let stderr = String::from_utf8_lossy(&bootout.stderr);
                let stdout = String::from_utf8_lossy(&bootout.stdout);
                let details = if stderr.trim().is_empty() {
                    stdout.trim()
                } else {
                    stderr.trim()
                };
                anyhow::bail!(
                    "launchctl bootout failed (exit {}): {}",
                    bootout.status.code().unwrap_or(-1),
                    details
                );
            }
        }

        Ok(())
    }

    #[cfg(target_os = "linux")]
    {
        let label_value = label()?;
        let m = manager()?;
        m.stop(ServiceStopCtx { label: label_value })
            .context("stop service")?;
        Ok(())
    }
}

async fn request_daemon_shutdown() -> Result<DaemonShutdownRequest> {
    let base_url = local_daemon_base_url()?;
    let shutdown_url = format!("{base_url}/api/shutdown");
    let shutdown_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .pool_max_idle_per_host(0)
        .build()
        .context("build daemon shutdown client")?;

    let response = match build_shutdown_request(&shutdown_client, &shutdown_url)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) if error.is_connect() => return Ok(DaemonShutdownRequest::NotRunning),
        Err(error) => {
            return Err(error).with_context(|| format!("POST {shutdown_url} failed"));
        }
    };
    let response = response
        .error_for_status()
        .with_context(|| format!("daemon returned an error for {shutdown_url}"))?;
    let process = response
        .headers()
        .get(DAEMON_PROCESS_ID_HEADER)
        .map(|value| {
            value
                .to_str()
                .context("daemon shutdown process id header is not UTF-8")?
                .parse::<u32>()
                .context("daemon shutdown process id header is not a valid PID")
        })
        .transpose()?
        .map(DaemonProcessIdentity::capture);
    response
        .bytes()
        .await
        .with_context(|| format!("read daemon shutdown response from {shutdown_url}"))?;
    drop(shutdown_client);
    Ok(DaemonShutdownRequest::Requested(process))
}

fn build_shutdown_request(client: &reqwest::Client, shutdown_url: &str) -> reqwest::RequestBuilder {
    // This is a server-visible shutdown contract, not merely a client pool
    // preference. Hyper graceful shutdown waits for accepted HTTP/1.1
    // connections to close, so the shutdown request must not leave its
    // connection alive for the health verifier to reuse.
    client
        .post(shutdown_url)
        .header(reqwest::header::CONNECTION, "close")
}

fn local_daemon_base_url() -> Result<String> {
    let raw =
        std::env::var("WENLAN_BIND_ADDR").unwrap_or_else(|_| DEFAULT_LOCAL_BIND_ADDR.to_string());
    let mut address: std::net::SocketAddr = raw
        .parse()
        .with_context(|| format!("invalid local WENLAN_BIND_ADDR {raw:?}"))?;
    if address.ip().is_unspecified() {
        address.set_ip(if address.is_ipv4() {
            std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
        } else {
            std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
        });
    } else if !address.ip().is_loopback() {
        anyhow::bail!(
            "refusing background lifecycle control through non-loopback WENLAN_BIND_ADDR {raw:?}"
        );
    }
    Ok(format!("http://{address}"))
}

async fn verify_daemon_unreachable(
    client: &reqwest::Client,
    health_url: &str,
    expected_process: Option<DaemonProcessIdentity>,
) -> Result<()> {
    let deadline = std::time::Instant::now() + SHUTDOWN_VERIFY_TIMEOUT;
    let mut unreachable_since = None;
    let mut process_system = expected_process.map(|_| sysinfo::System::new());
    loop {
        tokio::time::sleep(SHUTDOWN_PROBE_INTERVAL).await;
        let expected_process_running = match (expected_process, process_system.as_mut()) {
            (Some(process), Some(system)) => process.is_running(system),
            _ => false,
        };
        match client
            .get(health_url)
            .timeout(SHUTDOWN_PROBE_TIMEOUT)
            .send()
            .await
        {
            // During cooperative shutdown the socket can still accept a
            // connection after the HTTP service has stopped answering. A
            // timeout is neither proof of exit nor a terminal verification
            // error: keep probing until the bounded overall deadline. Reset
            // the refusal window because a listening-but-hung socket is not
            // yet a confirmed stop.
            Err(error) if error.is_timeout() => {
                if expected_process.is_some() && !expected_process_running {
                    unreachable_since.get_or_insert_with(std::time::Instant::now);
                } else {
                    unreachable_since = None;
                }
            }
            Err(error) if is_shutdown_disconnect(&error) => {
                if expected_process.is_none() || !expected_process_running {
                    unreachable_since.get_or_insert_with(std::time::Instant::now);
                } else {
                    unreachable_since = None;
                }
            }
            Ok(_) => {
                if !expected_process_running {
                    if let Some(process) = expected_process {
                        anyhow::bail!(
                            "a different daemon remained reachable at {health_url} after process {} exited",
                            process.pid
                        );
                    }
                }
                if unreachable_since.is_some() {
                    anyhow::bail!("daemon remained reachable at {health_url} after disconnecting");
                }
            }
            Err(error) => {
                return Err(error).with_context(|| format!("verify shutdown via {health_url}"));
            }
        }
        if unreachable_since.is_some_and(|since| since.elapsed() >= SHUTDOWN_STABILITY_WINDOW) {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            if expected_process_running {
                if let Some(process) = expected_process {
                    anyhow::bail!("daemon process {} remained running", process.pid);
                }
            }
            anyhow::bail!("daemon remained reachable at {health_url}");
        }
    }
}

fn is_shutdown_disconnect(error: &reqwest::Error) -> bool {
    if error.is_connect() {
        return true;
    }
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(current) = cause {
        if let Some(io_error) = current.downcast_ref::<std::io::Error>() {
            return matches!(
                io_error.kind(),
                std::io::ErrorKind::ConnectionRefused
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::NotConnected
            );
        }
        cause = current.source();
    }
    false
}

async fn verify_local_daemon_shutdown(
    expected_process: Option<DaemonProcessIdentity>,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .pool_max_idle_per_host(0)
        .build()
        .context("build daemon shutdown verification client")?;
    let health_url = format!("{}/api/health", local_daemon_base_url()?);
    verify_daemon_unreachable(&client, &health_url, expected_process).await
}

async fn stop() -> Result<()> {
    let registration_present = is_installed();
    let mut expected_process = None;
    let graceful_result = match request_daemon_shutdown().await {
        Ok(DaemonShutdownRequest::Requested(process)) => {
            expected_process = process;
            verify_local_daemon_shutdown(process).await.map(|()| true)
        }
        Ok(DaemonShutdownRequest::NotRunning) => Ok(false),
        Err(error) => Err(error),
    };
    let shutdown_requested = match graceful_result {
        Ok(true) => true,
        Ok(false) if registration_present => {
            // Connection refusal is ambiguous while a registered manager job
            // may still be starting or respawning. Stop the supervisor too;
            // otherwise `background off` can report success before the hot
            // daemon appears on its port.
            stop_registered_service().context("daemon was unreachable; service fallback failed")?;
            verify_local_daemon_shutdown(None)
                .await
                .context("service fallback did not keep the daemon stopped")?;
            true
        }
        Ok(false) => false,
        Err(graceful_error) if registration_present => {
            stop_registered_service().with_context(|| {
                format!(
                    "graceful daemon shutdown failed ({graceful_error:#}); service fallback failed"
                )
            })?;
            verify_local_daemon_shutdown(expected_process)
                .await
                .with_context(|| format!("graceful daemon shutdown failed: {graceful_error:#}"))?;
            true
        }
        Err(error) => return Err(error),
    };
    if registration_present {
        println!("Stopped {}. Background registration kept.", SERVICE_LABEL);
    } else if shutdown_requested {
        println!(
            "Stopped {}. No background registration found.",
            SERVICE_LABEL
        );
    } else {
        println!("Wenlan background process is already stopped; no registration found.");
    }
    Ok(())
}

/// Restart the Wenlan daemon: stop the running process, then start the freshly
/// registered binary. Required after an upgrade — installing a new binary does
/// not replace an already-running daemon (the new process detects the healthy
/// incumbent on port 7878 and exits). See wenlan-server/src/main.rs:582-615.
pub fn restart() -> Result<()> {
    if !is_installed() {
        anyhow::bail!("Wenlan background process is not set up. Run `wenlan background on` first.");
    }

    #[cfg(target_os = "windows")]
    {
        // No service-manager on Windows: drive Task Scheduler directly,
        // mirroring stop()'s /end and install()'s /run.
        let _ = std::process::Command::new("schtasks.exe")
            .args(["/end", "/tn", WINDOWS_TASK_NAME])
            .output();
        run_schtasks(&["/run", "/tn", WINDOWS_TASK_NAME], "run scheduled task")?;
        println!("Restarted Windows scheduled task '{}'.", WINDOWS_TASK_NAME);
        return Ok(());
    }

    #[cfg_attr(target_os = "windows", allow(unreachable_code))]
    let label_value = label()?;
    let m = manager()?;
    // Best-effort stop: errors if not currently running, which is fine.
    let _ = m.stop(ServiceStopCtx {
        label: label_value.clone(),
    });
    m.start(ServiceStartCtx { label: label_value })
        .context("start service")?;
    println!("Restarted {}.", SERVICE_LABEL);
    Ok(())
}

pub fn is_installed() -> bool {
    #[cfg(target_os = "windows")]
    {
        // `schtasks /query /tn <name>` exits 0 when the task exists, 1 when
        // it does not. No admin rights needed for the read-only query.
        std::process::Command::new("schtasks.exe")
            .args(["/query", "/tn", WINDOWS_TASK_NAME])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    #[cfg(not(target_os = "windows"))]
    {
        service_unit_path().map(|p| p.exists()).unwrap_or(false)
    }
}

pub async fn print_status() -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        if is_installed() {
            println!(
                "Service: scheduled task '{}' (registered)",
                WINDOWS_TASK_NAME
            );
        } else {
            println!(
                "Service: scheduled task '{}' (not installed)",
                WINDOWS_TASK_NAME
            );
        }
    }
    #[cfg(not(target_os = "windows"))]
    match service_unit_path() {
        Ok(path) if path.exists() => println!("Service unit: {} (installed)", path.display()),
        Ok(path) => println!("Service unit: {} (not installed)", path.display()),
        Err(e) => println!("Service unit: unable to resolve ({})", e),
    }

    let url = format!("{}/api/health", origin_host_from_env());
    match reqwest::get(&url).await {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.text().await.unwrap_or_default();
            println!("Health: ok ({})", url);
            println!("{}", body);
        }
        Ok(resp) => {
            println!("Health: unhealthy (status {})", resp.status());
        }
        Err(e) => {
            println!("Health: not reachable ({})", e);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_process_identity_observes_child_exit() {
        #[cfg(target_os = "windows")]
        let mut child = std::process::Command::new("cmd.exe")
            .args(["/C", "ping -n 30 127.0.0.1 >NUL"])
            .spawn()
            .expect("spawn process identity test child");
        #[cfg(not(target_os = "windows"))]
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn process identity test child");

        let identity = DaemonProcessIdentity::capture(child.id());
        let mut system = sysinfo::System::new();
        assert!(identity.is_running(&mut system));

        child.kill().expect("kill process identity test child");
        child.wait().expect("reap process identity test child");
        assert!(!identity.is_running(&mut system));
    }

    #[test]
    fn shutdown_request_declares_connection_close() {
        let client = reqwest::Client::builder()
            .build()
            .expect("build shutdown request test client");
        let request = build_shutdown_request(&client, "http://127.0.0.1:7878/api/shutdown")
            .build()
            .expect("build shutdown request");

        assert_eq!(
            request.headers().get(reqwest::header::CONNECTION),
            Some(&reqwest::header::HeaderValue::from_static("close"))
        );
    }
}
