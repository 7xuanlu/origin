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
use crate::client::recovery::poll_health;

pub const SERVICE_LABEL: &str = "com.wenlan.server";
const DEFAULT_LOCAL_BIND_ADDR: &str = "127.0.0.1:7878";
const SHUTDOWN_PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);
const SHUTDOWN_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);
const SHUTDOWN_STABILITY_WINDOW: std::time::Duration = std::time::Duration::from_secs(1);
const SHUTDOWN_VERIFY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);
const DAEMON_PROCESS_ID_HEADER: &str = "x-wenlan-process-id";
const AUTOSTART_OFF_MARKER: &str = "autostart.off";
/// How long `background on` waits for `/api/health` before reporting.
const FIRST_HEALTH_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

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
        BackgroundCommand::On => install().await,
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
            "wenlan-server not found next to wenlan at {}. If you installed with Homebrew, \
             run `brew upgrade 7xuanlu/tap/wenlan` (the current formula ships the daemon); \
             otherwise re-run the installer with `npx wenlan setup`.",
            server.display()
        );
    }
    Ok(server)
}

fn autostart_off_marker_path() -> PathBuf {
    wenlan_core::config::data_root().join(AUTOSTART_OFF_MARKER)
}

pub(crate) fn autostart_off_marker_exists() -> bool {
    autostart_off_marker_path().is_file()
}

fn write_autostart_off_marker() -> Result<PathBuf> {
    let path = autostart_off_marker_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create Wenlan data root for {}", parent.display()))?;
    }
    std::fs::write(&path, b"")
        .with_context(|| format!("write autostart marker {}", path.display()))?;
    Ok(path)
}

fn remove_autostart_off_marker() -> Result<()> {
    let path = autostart_off_marker_path();
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("remove autostart marker {}", path.display()))
        }
    }
}

#[cfg(target_os = "macos")]
fn origin_data_root() -> PathBuf {
    wenlan_core::config::data_root()
}

/// Builds a launchd plist that mirrors `service-manager`'s default output for
/// `OnFailure` restart + user-level + autostart, with the extra keys the old
/// embedded `com.wenlan.server.plist` template carried: stdout/stderr routing,
/// `EnvironmentVariables.RUST_LOG`, and the canonical `WENLAN_DATA_DIR`
/// ownership marker consumed by the desktop app. The daemon owns bounded file
/// logging, so launchd's stdout goes to `/dev/null`; stderr goes to
/// `<data root>/logs/launchd-stderr.log` so a crash before file logging is up
/// (a missing library, an abort) is never silent. launchd does not rotate
/// that file: the daemon writes only bootstrap failures there (progress bars
/// are hidden off a terminal), so it grows a couple of lines per attempt only
/// while launchd keeps retrying a daemon that cannot start.
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

/// The data root this CLI run was told to use, if any: `WENLAN_DATA_DIR` (or
/// its legacy `ORIGIN_DATA_DIR` spelling) in the CLI's own environment, which
/// is also how the desktop app hands over its selected store.
#[cfg(target_os = "macos")]
fn explicit_data_root() -> Option<PathBuf> {
    wenlan_core::env_compat::var_compat("WENLAN_DATA_DIR").map(PathBuf::from)
}

/// The plist `background on` registers: a fresh one when none exists, else
/// the existing file with the CLI-owned keys brought up to date.
///
/// Ownership rule for an existing `com.wenlan.server.plist` (re-registering
/// over one is the upgrade path, and the user or the desktop app may have
/// edited it since):
/// - The CLI owns the launch mechanics and rewrites them every time: `Label`,
///   `ProgramArguments`, `KeepAlive`, `RunAtLoad`, `Disabled`,
///   `StandardOutPath`, `StandardErrorPath`.
/// - Every other top-level key and every `EnvironmentVariables` entry belongs
///   to whoever put it there and is kept as is (a full rewrite on 2026-08-16
///   dropped the user's `WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE=1`).
/// - `RUST_LOG` is seeded to `info` only when absent.
/// - `WENLAN_DATA_DIR` is written only when this run was given one explicitly
///   (see [`explicit_data_root`]); otherwise an existing value stays and a
///   missing one stays missing, so a plist that relies on the platform
///   default keeps doing so instead of gaining a pinned copy of it.
///
/// A file that does not parse as a plist dictionary carries nothing worth
/// keeping and is replaced by a fresh one; the CLI says so on stderr.
#[cfg(target_os = "macos")]
fn launchd_plist_contents(
    existing: Option<&str>,
    program: &Path,
    stderr_log: &Path,
    data_root: &Path,
    explicit_data_root: Option<&Path>,
) -> String {
    let stdout = Path::new("/dev/null");
    let fresh = || build_launchd_plist(program, stdout, stderr_log, "info", data_root);
    let Some(existing) = existing else {
        return fresh();
    };
    match merge_launchd_plist(existing, program, stdout, stderr_log, explicit_data_root) {
        Ok(merged) => merged,
        Err(error) => {
            eprintln!(
                "The existing {SERVICE_LABEL} plist could not be read ({error:#}); writing a fresh one."
            );
            fresh()
        }
    }
}

/// Applies the ownership rule in [`launchd_plist_contents`] to an existing
/// plist and renders the result.
#[cfg(target_os = "macos")]
fn merge_launchd_plist(
    existing: &str,
    program: &Path,
    stdout_path: &Path,
    stderr_path: &Path,
    explicit_data_root: Option<&Path>,
) -> Result<String> {
    use plist::{Dictionary, Value};

    let path_value = |path: &Path| Value::String(path.to_string_lossy().into_owned());
    let mut root = match Value::from_reader_xml(existing.as_bytes()).context("parse plist")? {
        Value::Dictionary(root) => root,
        _ => anyhow::bail!("plist root is not a dictionary"),
    };
    root.insert("Label".into(), Value::String(SERVICE_LABEL.into()));
    root.insert(
        "ProgramArguments".into(),
        Value::Array(vec![path_value(program)]),
    );
    let mut keep_alive = Dictionary::new();
    keep_alive.insert("SuccessfulExit".into(), Value::Boolean(false));
    root.insert("KeepAlive".into(), Value::Dictionary(keep_alive));
    root.insert("RunAtLoad".into(), Value::Boolean(true));
    root.insert("Disabled".into(), Value::Boolean(true));
    root.insert("StandardOutPath".into(), path_value(stdout_path));
    root.insert("StandardErrorPath".into(), path_value(stderr_path));

    if root
        .get("EnvironmentVariables")
        .and_then(Value::as_dictionary)
        .is_none()
    {
        root.insert(
            "EnvironmentVariables".into(),
            Value::Dictionary(Dictionary::new()),
        );
    }
    let env = root
        .get_mut("EnvironmentVariables")
        .and_then(Value::as_dictionary_mut)
        .context("EnvironmentVariables is not a dictionary")?;
    if !env.contains_key("RUST_LOG") {
        env.insert("RUST_LOG".into(), Value::String("info".into()));
    }
    if let Some(data_root) = explicit_data_root {
        env.insert("WENLAN_DATA_DIR".into(), path_value(data_root));
    }

    let mut rendered = Vec::new();
    Value::Dictionary(root)
        .to_writer_xml(&mut rendered)
        .context("render plist")?;
    String::from_utf8(rendered).context("rendered plist is not UTF-8")
}

pub async fn install() -> Result<()> {
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
        let log_marks = DaemonLogMarks::capture();
        run_schtasks(&["/run", "/tn", WINDOWS_TASK_NAME], "run scheduled task")?;
        remove_autostart_off_marker()?;
        return wait_for_first_health(
            &format!("Windows scheduled task '{WINDOWS_TASK_NAME}' (wenlan-server)"),
            &log_marks,
        )
        .await;
    }

    #[cfg_attr(target_os = "windows", allow(unreachable_code))]
    let label_value = label()?;
    let program = current_server_path()?;
    let m = manager()?;

    // Gracefully stop any daemon that is currently answering, and wait for it
    // to fully exit, before reinstalling. Swapping the binary under a live
    // process would otherwise leave the OLD daemon running until something
    // else kills it (see restart()'s doc comment for the full story).
    match request_daemon_shutdown().await {
        Ok(DaemonShutdownRequest::Requested(process)) => {
            verify_local_daemon_shutdown(process)
                .await
                .context("graceful daemon shutdown before reinstall did not complete")?;
        }
        Ok(DaemonShutdownRequest::NotRunning) => {}
        Err(error) => {
            return Err(error).context("request graceful daemon shutdown before reinstall")
        }
    }

    // Stop any daemon still running under this label so the reinstall swaps
    // the binary. Without this, the freshly-installed binary detects the
    // healthy incumbent on port 7878 and exits, leaving the OLD daemon running
    // (wenlan-server/src/main.rs:582-615). Best-effort: errors if not running
    // (e.g. the graceful shutdown above already stopped it).
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
            let stderr_log = launchd_stderr_log_path(&data_root);
            if let Some(log_dir) = stderr_log.parent() {
                std::fs::create_dir_all(log_dir).with_context(|| {
                    format!("create the daemon log directory {}", log_dir.display())
                })?;
            }
            let plist_path = service_unit_path()?;
            let existing = match std::fs::read_to_string(&plist_path) {
                Ok(existing) => Some(existing),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => {
                    return Err(error).with_context(|| format!("read {}", plist_path.display()))
                }
            };
            Some(launchd_plist_contents(
                existing.as_deref(),
                &program,
                &stderr_log,
                &data_root,
                explicit_data_root().as_deref(),
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    };

    let log_marks = DaemonLogMarks::capture();
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
    remove_autostart_off_marker()?;
    wait_for_first_health(SERVICE_LABEL, &log_marks).await
}

/// Where launchd writes the daemon's raw stderr (see `build_launchd_plist`).
#[cfg(target_os = "macos")]
pub(crate) fn launchd_stderr_log_path(data_root: &Path) -> PathBuf {
    data_root.join("logs/launchd-stderr.log")
}

/// Where the daemon's logs ended before this command started the daemon, so
/// an error reported afterwards is one this start produced and not a stale
/// line from an earlier run (the logs outlive the daemon).
struct DaemonLogMarks {
    #[cfg(target_os = "macos")]
    daemon_log: (PathBuf, u64),
    #[cfg(target_os = "macos")]
    stderr_log: (PathBuf, u64),
}

impl DaemonLogMarks {
    fn capture() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::at(
                crate::commands::setup::daemon_log_path(),
                launchd_stderr_log_path(&origin_data_root()),
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self {}
        }
    }

    #[cfg(target_os = "macos")]
    fn at(daemon_log: PathBuf, stderr_log: PathBuf) -> Self {
        let len = |path: &Path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        Self {
            daemon_log: (daemon_log.clone(), len(&daemon_log)),
            stderr_log: (stderr_log.clone(), len(&stderr_log)),
        }
    }

    /// The error this start produced, if any: the daemon log's last `ERROR`
    /// line written after the mark, else the last line launchd captured on
    /// stderr since then (a crash before file logging is up, such as a
    /// missing library).
    #[cfg(target_os = "macos")]
    fn error_since(&self) -> Option<(PathBuf, String)> {
        let (daemon_log, daemon_mark) = &self.daemon_log;
        if let Some(line) =
            crate::commands::setup::last_daemon_error_since(daemon_log, *daemon_mark)
        {
            return Some((daemon_log.clone(), line));
        }
        let (stderr_log, stderr_mark) = &self.stderr_log;
        let bytes = std::fs::read(stderr_log).ok()?;
        let start = usize::try_from(*stderr_mark)
            .unwrap_or(usize::MAX)
            .min(bytes.len());
        let line = String::from_utf8_lossy(&bytes[start..])
            .lines()
            .map(str::trim)
            .rfind(|line| !line.is_empty())?
            .chars()
            .take(400)
            .collect();
        Some((stderr_log.clone(), line))
    }
}

/// `background on` used to return as soon as the service manager accepted the
/// job, so a daemon that crash-looped under launchd still printed "Installed
/// and started" (first-run gauntlet finding F4). Wait for the daemon to
/// answer before claiming it started; when it does not, tell a first start
/// that is still downloading the embedding model apart from a daemon that
/// failed during this start.
async fn wait_for_first_health(installed: &str, marks: &DaemonLogMarks) -> Result<()> {
    // A routable `WENLAN_BIND_ADDR` has no local URL to poll; the install
    // itself succeeded, so there is nothing more to report.
    let Ok(base_url) = local_daemon_base_url() else {
        println!("Installed and started {installed}.");
        return Ok(());
    };
    let health_url = format!("{base_url}/api/health");
    if poll_health(&health_url, FIRST_HEALTH_WAIT).await.is_ok() {
        println!("Installed and started {installed}; daemon healthy at {base_url}.");
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    if let Some((log_path, line)) = marks.error_since() {
        anyhow::bail!(
            "installed {installed}, but the daemon stopped with an error ({}):\n  {line}\nFix the cause above, then run `wenlan background on` again.",
            log_path.display()
        );
    }
    #[cfg(not(target_os = "macos"))]
    let _ = marks;
    println!("Installed {installed}; the daemon is still starting.");
    println!("{}", first_health_pending_note(&health_url));
    Ok(())
}

/// The message for a daemon that has not answered yet but has not failed
/// either, as far as this command can see.
fn first_health_pending_note(health_url: &str) -> String {
    let mut note = format!(
        "Daemon not answering yet at {health_url} after {}s. A first start downloads the 210 MB embedding model; check `wenlan status` in a minute.",
        FIRST_HEALTH_WAIT.as_secs()
    );
    #[cfg(target_os = "macos")]
    note.push_str(&format!(
        "\nIf it stays down, `wenlan doctor` shows the daemon's last error (log: {}).",
        crate::commands::setup::daemon_log_path().display()
    ));
    #[cfg(target_os = "linux")]
    note.push_str("\nIf it stays down, `journalctl --user -u wenlan-server` says why.");
    #[cfg(target_os = "windows")]
    note.push_str("\nIf it stays down, run `wenlan doctor`.");
    note
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

#[cfg(target_os = "macos")]
fn kickstart_registered_service() -> Result<()> {
    let uid = current_user_id()?;
    let target = format!("gui/{uid}/{SERVICE_LABEL}");
    let output = std::process::Command::new("launchctl")
        .args(["kickstart", "-k", &target])
        .output()
        .context("spawn launchctl kickstart")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let details = if stderr.trim().is_empty() {
            stdout.trim()
        } else {
            stderr.trim()
        };
        anyhow::bail!(
            "launchctl kickstart failed (exit {}): {}",
            output.status.code().unwrap_or(-1),
            details
        );
    }
    Ok(())
}

/// Best-effort `launchctl print` dump for the restart failure path, so a
/// timed-out health poll leaves the user something to look at besides the URL
/// that never answered.
#[cfg(target_os = "macos")]
fn print_service_diagnostics() {
    let Ok(uid) = current_user_id() else {
        return;
    };
    let target = format!("gui/{uid}/{SERVICE_LABEL}");
    if let Ok(output) = std::process::Command::new("launchctl")
        .args(["print", &target])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            eprintln!("{}", stdout.trim());
        }
    }
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

// restart() starts the daemon back up after its graceful stop using whatever
// fallback each platform's service manager needs, so each platform gets its
// own definition rather than one function with unused parameters in the
// branches that do not need them (macOS's `kickstart -k` folds its fallback
// into the start itself and ignores both arguments).
#[cfg(target_os = "macos")]
async fn start_after_graceful_stop(
    _graceful_failed: bool,
    _expected_process: Option<DaemonProcessIdentity>,
) -> Result<()> {
    kickstart_registered_service()
        .context("if you ran `wenlan background off`, run `wenlan background on`")
}

#[cfg(target_os = "linux")]
async fn start_after_graceful_stop(
    graceful_failed: bool,
    expected_process: Option<DaemonProcessIdentity>,
) -> Result<()> {
    if graceful_failed {
        stop_registered_service()
            .context("graceful daemon shutdown failed; service fallback failed")?;
        verify_local_daemon_shutdown(expected_process)
            .await
            .context("service fallback did not stop the daemon before restart")?;
    }
    let label_value = label()?;
    let m = manager()?;
    m.start(ServiceStartCtx { label: label_value })
        .context("start service; if you ran `wenlan background off`, run `wenlan background on`")
}

#[cfg(target_os = "windows")]
async fn start_after_graceful_stop(
    graceful_failed: bool,
    expected_process: Option<DaemonProcessIdentity>,
) -> Result<()> {
    if graceful_failed {
        let _ = std::process::Command::new("schtasks.exe")
            .args(["/end", "/tn", WINDOWS_TASK_NAME])
            .output();
        verify_local_daemon_shutdown(expected_process)
            .await
            .context("service fallback did not stop the daemon before restart")?;
    }
    run_schtasks(&["/run", "/tn", WINDOWS_TASK_NAME], "run scheduled task")
        .context("if you ran `wenlan background off`, run `wenlan background on`")?;
    Ok(())
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
    let marker = write_autostart_off_marker()?;
    if registration_present {
        println!(
            "Stopped {}. Background registration kept. Autostart disabled via {}.",
            SERVICE_LABEL,
            marker.display()
        );
    } else if shutdown_requested {
        println!(
            "Stopped {}. No background registration found. Autostart disabled via {}.",
            SERVICE_LABEL,
            marker.display()
        );
    } else {
        println!(
            "Wenlan background process is already stopped; no registration found. Autostart disabled via {}.",
            marker.display()
        );
    }
    Ok(())
}

/// Restart the Wenlan daemon: gracefully stop the running process, wait for
/// it to fully exit, then start the freshly registered binary and verify
/// `/api/health` answers before returning. A bare stop-then-start with no
/// wait races the old process's exit: launchd's `KeepAlive
/// {SuccessfulExit=false}` never relaunches an exit-0 process, so `start`
/// issued while the old process is still alive is a silent no-op. Restarting
/// is also required after an upgrade — installing a new binary does not
/// replace an already-running daemon (the new process detects the healthy
/// incumbent on port 7878 and exits). See wenlan-server/src/main.rs:582-615.
pub async fn restart() -> Result<()> {
    if !is_installed() {
        anyhow::bail!("Wenlan background process is not set up. Run `wenlan background on` first.");
    }

    // Ask the daemon to shut down cooperatively and wait for it to actually
    // exit — the same graceful path stop() uses. A failure here does not
    // abort restart(); the OS-specific start step below force-stops as a
    // fallback where one is needed.
    let mut expected_process = None;
    let graceful_failed = match request_daemon_shutdown().await {
        Ok(DaemonShutdownRequest::Requested(process)) => {
            expected_process = process;
            verify_local_daemon_shutdown(process).await.is_err()
        }
        Ok(DaemonShutdownRequest::NotRunning) => false,
        Err(_) => true,
    };

    start_after_graceful_stop(graceful_failed, expected_process).await?;

    let base_url = local_daemon_base_url()?;
    let health_url = format!("{base_url}/api/health");
    if let Err(error) = poll_health(&health_url, std::time::Duration::from_secs(30)).await {
        #[cfg(target_os = "macos")]
        print_service_diagnostics();
        anyhow::bail!(
            "daemon at {health_url} did not become healthy after restart ({error:#}); the daemon may still be starting; re-check with `wenlan status`"
        );
    }

    remove_autostart_off_marker()?;
    println!(
        "Restarted {}; daemon healthy at {}.",
        SERVICE_LABEL, base_url
    );
    Ok(())
}

/// Start the already-registered Wenlan service without stopping or replacing it.
pub fn start_registered(explicit_user_command: bool) -> Result<()> {
    if !is_installed() {
        anyhow::bail!("Wenlan background process is not set up. Run `wenlan background on` first.");
    }

    #[cfg(target_os = "windows")]
    {
        run_schtasks(&["/run", "/tn", WINDOWS_TASK_NAME], "run scheduled task")?;
        if explicit_user_command {
            remove_autostart_off_marker()?;
        }
        return Ok(());
    }

    #[cfg_attr(target_os = "windows", allow(unreachable_code))]
    let label_value = label()?;
    let m = manager()?;
    // The service manager may unload/load when the plist carries Disabled: true.
    m.start(ServiceStartCtx { label: label_value })
        .context("start service")?;
    if explicit_user_command {
        remove_autostart_off_marker()?;
    }
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
        // Windows can keep a terminated process visible in the enumeration
        // snapshot briefly after wait(); poll instead of racing one check.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while identity.is_running(&mut system) {
            assert!(
                std::time::Instant::now() < deadline,
                "child process still visible 10s after exit"
            );
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn launchd_plist_keeps_stderr_in_the_data_root() {
        let data_root = Path::new("/tmp/wenlan-plist-test-root");
        let stderr_log = launchd_stderr_log_path(data_root);
        assert_eq!(
            stderr_log,
            Path::new("/tmp/wenlan-plist-test-root/logs/launchd-stderr.log")
        );
        let plist = build_launchd_plist(
            Path::new("/opt/wenlan/wenlan-server"),
            Path::new("/dev/null"),
            &stderr_log,
            "info",
            data_root,
        );
        assert!(plist.contains(
            "<key>StandardErrorPath</key>\n\t<string>/tmp/wenlan-plist-test-root/logs/launchd-stderr.log</string>"
        ));
        assert!(plist.contains("<key>StandardOutPath</key>\n\t<string>/dev/null</string>"));
        assert!(plist.contains(
            "<key>WENLAN_DATA_DIR</key>\n\t\t<string>/tmp/wenlan-plist-test-root</string>"
        ));
    }

    /// A plist as a user leaves it: an older binary path and stderr path, a
    /// tuning key the CLI knows nothing about, a `RUST_LOG` they changed, and
    /// an extra environment variable — with no `WENLAN_DATA_DIR`.
    #[cfg(target_os = "macos")]
    const USER_EDITED_PLIST: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>com.wenlan.server</string>
	<key>ProgramArguments</key>
	<array>
		<string>/old/bin/wenlan-server</string>
	</array>
	<key>StandardErrorPath</key>
	<string>/old/root/logs/launchd-stderr.log</string>
	<key>ThrottleInterval</key>
	<integer>30</integer>
	<key>EnvironmentVariables</key>
	<dict>
		<key>RUST_LOG</key>
		<string>debug</string>
		<key>WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE</key>
		<string>1</string>
	</dict>
</dict>
</plist>
"#;

    #[cfg(target_os = "macos")]
    fn plist_dict(rendered: &str) -> plist::Dictionary {
        match plist::Value::from_reader_xml(rendered.as_bytes()).expect("rendered plist parses") {
            plist::Value::Dictionary(root) => root,
            other => panic!("plist root is not a dictionary: {other:?}"),
        }
    }

    #[cfg(target_os = "macos")]
    fn plist_env(root: &plist::Dictionary) -> &plist::Dictionary {
        root.get("EnvironmentVariables")
            .and_then(plist::Value::as_dictionary)
            .expect("EnvironmentVariables dictionary")
    }

    #[cfg(target_os = "macos")]
    fn plist_string<'a>(dict: &'a plist::Dictionary, key: &str) -> Option<&'a str> {
        dict.get(key).and_then(plist::Value::as_string)
    }

    #[cfg(target_os = "macos")]
    fn merged_user_plist(explicit_data_root: Option<&Path>) -> plist::Dictionary {
        plist_dict(&launchd_plist_contents(
            Some(USER_EDITED_PLIST),
            Path::new("/new/bin/wenlan-server"),
            Path::new("/new/root/logs/launchd-stderr.log"),
            Path::new("/new/root"),
            explicit_data_root,
        ))
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn launchd_plist_without_an_existing_file_is_the_fresh_template() {
        let program = Path::new("/opt/wenlan/wenlan-server");
        let stderr_log = Path::new("/data/logs/launchd-stderr.log");
        let data_root = Path::new("/data");
        assert_eq!(
            launchd_plist_contents(None, program, stderr_log, data_root, Some(data_root)),
            build_launchd_plist(
                program,
                Path::new("/dev/null"),
                stderr_log,
                "info",
                data_root
            )
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn launchd_plist_merge_keeps_user_keys_and_updates_cli_owned_ones() {
        let root = merged_user_plist(None);
        let env = plist_env(&root);
        assert_eq!(
            plist_string(env, "WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE"),
            Some("1"),
            "user environment variables survive re-registration"
        );
        assert_eq!(
            plist_string(env, "RUST_LOG"),
            Some("debug"),
            "a RUST_LOG the user changed is theirs"
        );
        assert_eq!(
            env.get("WENLAN_DATA_DIR"),
            None,
            "no default WENLAN_DATA_DIR is added when the CLI was not given one"
        );
        assert_eq!(
            root.get("ThrottleInterval")
                .and_then(plist::Value::as_signed_integer),
            Some(30),
            "unknown top-level keys survive"
        );
        assert_eq!(
            root.get("ProgramArguments")
                .and_then(plist::Value::as_array)
                .and_then(|args| args.first())
                .and_then(plist::Value::as_string),
            Some("/new/bin/wenlan-server"),
            "the program path is CLI-owned and moves with the binary"
        );
        assert_eq!(
            plist_string(&root, "StandardErrorPath"),
            Some("/new/root/logs/launchd-stderr.log")
        );
        assert_eq!(plist_string(&root, "StandardOutPath"), Some("/dev/null"));
        assert_eq!(plist_string(&root, "Label"), Some(SERVICE_LABEL));
        assert_eq!(
            root.get("KeepAlive")
                .and_then(plist::Value::as_dictionary)
                .and_then(|keep_alive| keep_alive.get("SuccessfulExit"))
                .and_then(plist::Value::as_boolean),
            Some(false)
        );
        assert_eq!(
            root.get("RunAtLoad").and_then(plist::Value::as_boolean),
            Some(true)
        );
        assert_eq!(
            root.get("Disabled").and_then(plist::Value::as_boolean),
            Some(true)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn launchd_plist_merge_writes_only_an_explicit_data_root() {
        let root = merged_user_plist(Some(Path::new("/explicit/root")));
        assert_eq!(
            plist_string(plist_env(&root), "WENLAN_DATA_DIR"),
            Some("/explicit/root")
        );
        assert_eq!(
            plist_string(plist_env(&root), "WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE"),
            Some("1")
        );

        let pinned = USER_EDITED_PLIST.replace(
            "\t\t<key>RUST_LOG</key>",
            "\t\t<key>WENLAN_DATA_DIR</key>\n\t\t<string>/private/tmp/scratch</string>\n\t\t<key>RUST_LOG</key>",
        );
        let program = Path::new("/new/bin/wenlan-server");
        let stderr_log = Path::new("/new/root/logs/launchd-stderr.log");
        let data_root = Path::new("/new/root");
        let kept = plist_dict(&launchd_plist_contents(
            Some(&pinned),
            program,
            stderr_log,
            data_root,
            None,
        ));
        assert_eq!(
            plist_string(plist_env(&kept), "WENLAN_DATA_DIR"),
            Some("/private/tmp/scratch"),
            "an existing WENLAN_DATA_DIR is kept when the CLI was not given one"
        );
        let replaced = plist_dict(&launchd_plist_contents(
            Some(&pinned),
            program,
            stderr_log,
            data_root,
            Some(data_root),
        ));
        assert_eq!(
            plist_string(plist_env(&replaced), "WENLAN_DATA_DIR"),
            Some("/new/root"),
            "an explicit data root replaces the existing one"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn launchd_plist_merge_seeds_rust_log_and_replaces_an_unreadable_file() {
        let no_env = USER_EDITED_PLIST.replace(
            "\t<key>EnvironmentVariables</key>\n\t<dict>\n\t\t<key>RUST_LOG</key>\n\t\t<string>debug</string>\n\t\t<key>WENLAN_ENABLE_EDGE_GROUNDING_PROMOTE</key>\n\t\t<string>1</string>\n\t</dict>\n",
            "",
        );
        assert!(!no_env.contains("EnvironmentVariables"), "{no_env}");
        let program = Path::new("/new/bin/wenlan-server");
        let stderr_log = Path::new("/new/root/logs/launchd-stderr.log");
        let data_root = Path::new("/new/root");
        let seeded = plist_dict(&launchd_plist_contents(
            Some(&no_env),
            program,
            stderr_log,
            data_root,
            None,
        ));
        assert_eq!(plist_string(plist_env(&seeded), "RUST_LOG"), Some("info"));
        assert_eq!(plist_env(&seeded).get("WENLAN_DATA_DIR"), None);

        assert_eq!(
            launchd_plist_contents(
                Some("fake registered launch agent"),
                program,
                stderr_log,
                data_root,
                None
            ),
            build_launchd_plist(
                program,
                Path::new("/dev/null"),
                stderr_log,
                "info",
                data_root
            ),
            "a file that is not a plist has nothing to keep"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn install_reports_only_errors_written_after_it_started_the_daemon() {
        let dir = tempfile::tempdir().expect("tempdir");
        let daemon_log = dir.path().join("wenlan-server.log");
        let stderr_log = dir.path().join("launchd-stderr.log");
        std::fs::write(
            &daemon_log,
            "INFO wenlan-server v0.17.0\nERROR wenlan_server: stale failure\n",
        )
        .expect("write daemon log");
        let marks = DaemonLogMarks::at(daemon_log.clone(), stderr_log.clone());
        assert_eq!(
            marks.error_since(),
            None,
            "an error from an earlier run must not be blamed on this start"
        );
        std::fs::write(
            &stderr_log,
            "dyld[123]: Library not loaded: libonnxruntime.dylib\n",
        )
        .expect("write stderr log");
        assert_eq!(
            marks.error_since(),
            Some((
                stderr_log.clone(),
                "dyld[123]: Library not loaded: libonnxruntime.dylib".to_string()
            ))
        );
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&daemon_log)
            .expect("open daemon log");
        std::io::Write::write_all(
            &mut file,
            b"INFO wenlan-server v0.17.1\nERROR wenlan_server: fresh failure\n",
        )
        .expect("append");
        assert_eq!(
            marks.error_since(),
            Some((daemon_log, "ERROR wenlan_server: fresh failure".to_string())),
            "the daemon log's own error wins over launchd's stderr"
        );
    }

    #[test]
    fn first_health_note_says_how_long_it_waited_and_what_to_do() {
        let note = first_health_pending_note("http://127.0.0.1:7878/api/health");
        assert!(note.contains("after 10s"), "{note}");
        assert!(note.contains("210 MB embedding model"), "{note}");
        assert!(note.contains("`wenlan status`"), "{note}");
        assert!(note.contains("If it stays down"), "{note}");
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
