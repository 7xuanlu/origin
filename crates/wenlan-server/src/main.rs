// SPDX-License-Identifier: Apache-2.0
//! Wenlan headless daemon — runs the memory server without Tauri.

mod cmd_backfill;

struct DaemonDataLock {
    _file: std::fs::File,
}

impl DaemonDataLock {
    fn acquire(root: &std::path::Path, require_existing: bool) -> anyhow::Result<Self> {
        use sha2::Digest as _;

        let absolute_root = if root.is_absolute() {
            root.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|error| anyhow::anyhow!("resolve current directory: {error}"))?
                .join(root)
        };
        if require_existing && !absolute_root.is_dir() {
            anyhow::bail!(
                "repair-only startup requires an existing Wenlan data root: {}",
                root.display()
            );
        }
        if absolute_root.exists() && !absolute_root.is_dir() {
            anyhow::bail!("Wenlan data root is not a directory: {}", root.display());
        }

        let canonical_root = if absolute_root.is_dir() {
            std::fs::canonicalize(&absolute_root).map_err(|error| {
                anyhow::anyhow!("resolve Wenlan data root {}: {error}", root.display())
            })?
        } else {
            let parent = absolute_root.parent().ok_or_else(|| {
                anyhow::anyhow!("Wenlan data root has no parent: {}", root.display())
            })?;
            std::fs::create_dir_all(parent).map_err(|error| {
                anyhow::anyhow!(
                    "create Wenlan data-root parent {}: {error}",
                    parent.display()
                )
            })?;
            let canonical_parent = std::fs::canonicalize(parent).map_err(|error| {
                anyhow::anyhow!(
                    "resolve Wenlan data-root parent {}: {error}",
                    parent.display()
                )
            })?;
            canonical_parent.join(absolute_root.file_name().ok_or_else(|| {
                anyhow::anyhow!("Wenlan data root has no name: {}", root.display())
            })?)
        };
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"wenlan-daemon-data-root-lock-v1\0");
        #[cfg(windows)]
        hasher.update(canonical_root.to_string_lossy().to_lowercase().as_bytes());
        #[cfg(not(windows))]
        hasher.update(canonical_root.as_os_str().as_encoded_bytes());
        let lock_key = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        // Keep operational lock state in the canonical root's stable parent,
        // not in process-dependent TMPDIR and not inside the data being
        // verified. Lock files are intentionally never unlinked: removing one
        // can split contenders across two inodes.
        let lock_path = canonical_root
            .parent()
            .ok_or_else(|| {
                anyhow::anyhow!("Wenlan data root has no lock parent: {}", root.display())
            })?
            .join(format!(".wenlan-daemon-{lock_key}.lock"));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|error| {
                anyhow::anyhow!(
                    "open Wenlan data-root lock {}: {error}",
                    lock_path.display()
                )
            })?;
        fs2::FileExt::try_lock_exclusive(&file).map_err(|error| {
            anyhow::anyhow!(
                "Wenlan data root {} is already owned by another process: {error}",
                canonical_root.display()
            )
        })?;
        Ok(Self { _file: file })
    }
}

/// Resolve the bind address. Honors the `WENLAN_BIND_ADDR` env var when set
/// (e.g. inside Docker where the daemon must listen on `0.0.0.0`). Falls back
/// to the localhost-only address used by the macOS/native install path.
fn resolve_bind_addr(port: u16) -> String {
    wenlan_core::env_compat::var_compat("WENLAN_BIND_ADDR")
        .and_then(|v| v.into_string().ok())
        .unwrap_or_else(|| format!("127.0.0.1:{}", port))
}

fn resolve_startup_bind_addr(port: u16, startup_repair_claimed: bool) -> String {
    if startup_repair_claimed {
        format!("127.0.0.1:{port}")
    } else {
        resolve_bind_addr(port)
    }
}

fn resolve_startup_port(configured_port: u16, startup_repair_claimed: bool) -> anyhow::Result<u16> {
    if startup_repair_claimed && configured_port != 7878 {
        anyhow::bail!("repair-only startup requires canonical port 7878");
    }
    Ok(configured_port)
}

#[cfg(target_os = "macos")]
const SERVER_LOG_MAX_BYTES: usize = 10 * 1024 * 1024;
#[cfg(target_os = "macos")]
const SERVER_LOG_BACKUPS: usize = 5;
#[cfg(any(target_os = "macos", test))]
const BOOTSTRAP_LOG_MAX_BYTES: usize = 256 * 1024;
#[cfg(any(target_os = "macos", test))]
const BOOTSTRAP_LOG_BACKUPS: usize = 1;

fn resolve_wenlan_root() -> std::path::PathBuf {
    wenlan_core::env_compat::var_compat("WENLAN_DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join("wenlan")
        })
}

#[cfg(any(target_os = "macos", test))]
fn preflight_rotating_log_path(path: &std::path::Path) -> std::io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "log path has no parent")
    })?;
    std::fs::create_dir_all(parent)?;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::other("log path is not a regular file"));
    }
    Ok(())
}

#[cfg(any(target_os = "macos", test))]
fn new_server_log_writer(
    wenlan_root: &std::path::Path,
    max_bytes: usize,
    backups: usize,
) -> std::io::Result<file_rotate::FileRotate<file_rotate::suffix::AppendCount>> {
    let path = wenlan_root.join("logs/wenlan-server.log");
    preflight_rotating_log_path(&path)?;
    Ok(file_rotate::FileRotate::new(
        path,
        file_rotate::suffix::AppendCount::new(backups),
        file_rotate::ContentLimit::Bytes(max_bytes),
        file_rotate::compression::Compression::None,
        None,
    ))
}

#[cfg(any(target_os = "macos", test))]
fn new_bootstrap_log_writer(
    wenlan_root: &std::path::Path,
    max_bytes: usize,
    backups: usize,
) -> std::io::Result<file_rotate::FileRotate<file_rotate::suffix::AppendCount>> {
    let path = wenlan_root.join("logs/wenlan-server.bootstrap.log");
    preflight_rotating_log_path(&path)?;
    Ok(file_rotate::FileRotate::new(
        path,
        file_rotate::suffix::AppendCount::new(backups),
        file_rotate::ContentLimit::Bytes(max_bytes),
        file_rotate::compression::Compression::None,
        None,
    ))
}

#[cfg(any(target_os = "macos", test))]
fn write_bootstrap_message(
    wenlan_root: &std::path::Path,
    fallback_root: &std::path::Path,
    message: &str,
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write as _;

    let (mut writer, path) =
        match new_bootstrap_log_writer(wenlan_root, BOOTSTRAP_LOG_MAX_BYTES, BOOTSTRAP_LOG_BACKUPS)
        {
            Ok(writer) => (writer, wenlan_root.join("logs/wenlan-server.bootstrap.log")),
            Err(primary_error) => {
                eprintln!(
                    "Primary bootstrap log unavailable at {}: {primary_error}",
                    wenlan_root.display()
                );
                (
                    new_bootstrap_log_writer(
                        fallback_root,
                        BOOTSTRAP_LOG_MAX_BYTES,
                        BOOTSTRAP_LOG_BACKUPS,
                    )?,
                    fallback_root.join("logs/wenlan-server.bootstrap.log"),
                )
            }
        };
    writeln!(writer, "{message}")?;
    writer.flush()?;
    Ok(path)
}

fn report_bootstrap_error(wenlan_root: &std::path::Path, message: &str) {
    #[cfg(not(target_os = "macos"))]
    let _ = wenlan_root;

    eprintln!("{message}");
    tracing::error!("{message}");

    #[cfg(target_os = "macos")]
    if std::env::var_os("XPC_SERVICE_NAME").is_some() {
        let fallback_root = fallback_log_root();
        let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            write_bootstrap_message(wenlan_root, &fallback_root, message)
        }));
        if let Err(error) = write_result.unwrap_or_else(|_| {
            Err(std::io::Error::other(
                "bootstrap log writer panicked while reporting an error",
            ))
        }) {
            eprintln!("Failed to write bootstrap log: {error}");
        }
    }
}

fn install_bootstrap_panic_hook(wenlan_root: std::path::PathBuf) {
    std::panic::set_hook(Box::new(move |panic| {
        report_bootstrap_error(
            &wenlan_root,
            &format!("panic during daemon bootstrap: {panic}"),
        );
    }));
}

fn new_server_log_rate_limit() -> tracing_throttle::TracingRateLimitLayer {
    tracing_throttle::TracingRateLimitLayer::new()
}

#[cfg(target_os = "macos")]
fn fallback_log_root() -> std::path::PathBuf {
    dirs::home_dir()
        .map(|home| home.join("Library/Logs/com.wenlan.server-fallback"))
        .unwrap_or_else(|| std::env::temp_dir().join("wenlan-server-fallback"))
}

fn init_logging(wenlan_root: &std::path::Path) -> anyhow::Result<()> {
    use tracing_subscriber::prelude::*;

    #[cfg(not(target_os = "macos"))]
    let _ = wenlan_root;

    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,wenlan_core=info,wenlan_server=info".into());

    #[cfg(target_os = "macos")]
    {
        let writer = match new_server_log_writer(
            wenlan_root,
            SERVER_LOG_MAX_BYTES,
            SERVER_LOG_BACKUPS,
        ) {
            Ok(writer) => writer,
            Err(primary_error) => {
                let fallback_root = fallback_log_root();
                eprintln!(
                    "Primary daemon log unavailable at {}: {primary_error}; using {}",
                    wenlan_root.display(),
                    fallback_root.display()
                );
                new_server_log_writer(
                    &fallback_root,
                    SERVER_LOG_MAX_BYTES,
                    SERVER_LOG_BACKUPS,
                )
                .map_err(|fallback_error| {
                    anyhow::anyhow!(
                        "initialize rotating file logging: primary={primary_error}; fallback={fallback_error}"
                    )
                })?
            }
        };
        let writer = std::sync::Mutex::new(writer);
        let fmt = tracing_subscriber::fmt::layer()
            .with_writer(writer)
            .with_filter(new_server_log_rate_limit());
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt)
            .try_init()
            .map_err(|error| anyhow::anyhow!("initialize rotating file logging: {error}"))
    }

    #[cfg(not(target_os = "macos"))]
    {
        let fmt = tracing_subscriber::fmt::layer().with_filter(new_server_log_rate_limit());
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt)
            .try_init()
            .map_err(|error| anyhow::anyhow!("initialize console logging: {error}"))
    }
}

fn startup_projection_writes_allowed(repair_recovery_pending: bool) -> bool {
    !repair_recovery_pending
}

#[derive(Debug, Clone)]
struct StartupRepairClaim {
    manifest_id: String,
    manifest_digest: wenlan_types::repair::RepairDigest,
}

impl StartupRepairClaim {
    fn try_new(
        manifest_id: Option<String>,
        manifest_digest: Option<String>,
    ) -> anyhow::Result<Option<Self>> {
        match (manifest_id, manifest_digest) {
            (None, None) => Ok(None),
            (Some(manifest_id), Some(manifest_digest)) => {
                let manifest_digest = wenlan_types::repair::RepairDigest::parse(&manifest_digest)
                    .map_err(|error| {
                    anyhow::anyhow!("invalid startup repair digest: {error}")
                })?;
                Ok(Some(Self {
                    manifest_id,
                    manifest_digest,
                }))
            }
            _ => anyhow::bail!(
                "startup repair requires both --repair-manifest-id and --repair-manifest-digest"
            ),
        }
    }

    fn manifest_id(&self) -> &str {
        &self.manifest_id
    }

    fn apply_request(&self) -> anyhow::Result<wenlan_types::repair::ApplyRepairRequest> {
        let approval = format!(
            "apply repair {} {}",
            self.manifest_id,
            self.manifest_digest.as_str()
        );
        wenlan_types::repair::ApplyRepairRequest::try_new(
            self.manifest_id.clone(),
            self.manifest_digest.clone(),
            approval,
        )
        .map_err(|error| anyhow::anyhow!("invalid startup repair claim: {error}"))
    }
}

fn validate_startup_repair_claim(
    store: &wenlan_core::repair::RepairArtifactStore,
    claim: &StartupRepairClaim,
) -> anyhow::Result<()> {
    let manifest = store
        .load_manifest(claim.manifest_id())
        .map_err(|error| anyhow::anyhow!("load startup repair manifest: {error}"))?;
    if manifest.manifest_digest() != &claim.manifest_digest {
        anyhow::bail!("startup repair manifest digest mismatch");
    }
    Ok(())
}

fn stored_repair_apply_request(
    store: &wenlan_core::repair::RepairArtifactStore,
    manifest_id: &str,
) -> anyhow::Result<wenlan_types::repair::ApplyRepairRequest> {
    let manifest = store
        .load_manifest(manifest_id)
        .map_err(|error| anyhow::anyhow!("load pending repair manifest: {error}"))?;
    let digest = manifest.manifest_digest().clone();
    let approval = format!("apply repair {manifest_id} {}", digest.as_str());
    wenlan_types::repair::ApplyRepairRequest::try_new(manifest_id.to_string(), digest, approval)
        .map_err(|error| anyhow::anyhow!("invalid pending repair authority: {error}"))
}

fn select_startup_repair_fence(
    pending_manifest_ids: &[String],
    claim: Option<&StartupRepairClaim>,
) -> anyhow::Result<Option<String>> {
    let mut manifest_ids = std::collections::BTreeSet::new();
    manifest_ids.extend(pending_manifest_ids.iter().cloned());
    if let Some(claim) = claim {
        manifest_ids.insert(claim.manifest_id().to_string());
    }
    match manifest_ids.len() {
        0 => Ok(None),
        1 => Ok(manifest_ids.into_iter().next()),
        _ => anyhow::bail!(
            "multiple pending repairs require operator resolution before startup: {}",
            manifest_ids.into_iter().collect::<Vec<_>>().join(", ")
        ),
    }
}

fn optional_runtime_workers_allowed(startup_repair_claimed: bool) -> bool {
    !startup_repair_claimed
}

fn on_device_model_working_set_bytes(model: &wenlan_core::on_device_models::OnDeviceModel) -> u64 {
    (model.ram_required_gb * 1024.0 * 1024.0 * 1024.0).ceil() as u64
}

struct StartupModelLoadReservation(Arc<std::sync::atomic::AtomicBool>);

impl Drop for StartupModelLoadReservation {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Release);
    }
}

fn existing_daemon_may_satisfy_startup(startup_repair_claimed: bool) -> bool {
    !startup_repair_claimed
}

#[cfg(test)]
mod bind_addr_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    #[cfg(unix)]
    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
        fn _exit(status: std::ffi::c_int) -> !;
    }

    static TEST_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn env_lock() -> &'static Mutex<()> {
        TEST_ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(unix)]
    extern "C" fn fail_if_c_exit_handlers_run() {
        // SAFETY: This callback runs only in the dedicated child below. `_exit`
        // terminates that child immediately and cannot return into the handler.
        unsafe { _exit(71) }
    }

    #[cfg(unix)]
    #[test]
    fn daemon_exit_skips_c_exit_handlers() {
        const CHILD_ENV: &str = "WENLAN_TEST_DAEMON_EXIT_CHILD";
        if std::env::var_os(CHILD_ENV).is_some() {
            // SAFETY: The callback has the required C ABI, no captured state,
            // and remains valid for the lifetime of this dedicated child.
            assert_eq!(unsafe { atexit(fail_if_c_exit_handlers_run) }, 0);
            exit_daemon(0);
        }

        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "bind_addr_tests::daemon_exit_skips_c_exit_handlers",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .status()
            .unwrap();

        assert_eq!(
            status.code(),
            Some(0),
            "daemon exit ran a C exit handler instead of terminating directly: {status}"
        );
    }

    #[test]
    fn default_when_env_unset() {
        let _guard = env_lock().lock().unwrap();
        std::env::remove_var("WENLAN_BIND_ADDR");
        assert_eq!(resolve_bind_addr(7878), "127.0.0.1:7878");
    }

    #[test]
    fn server_log_writer_rotates_at_byte_cap_and_bounds_retention() {
        use std::io::Write as _;

        let root = tempfile::tempdir().unwrap();
        let mut writer = new_server_log_writer(root.path(), 64, 2).unwrap();
        for index in 0..20 {
            writeln!(writer, "bounded log line {index:02}").unwrap();
        }
        drop(writer);

        let log_dir = root.path().join("logs");
        let mut logs = std::fs::read_dir(&log_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("wenlan-server.log"))
            })
            .collect::<Vec<_>>();
        logs.sort();

        assert_eq!(
            logs.len(),
            3,
            "current log plus exactly two retained rotations: {logs:?}"
        );
        assert!(
            logs.iter().all(|path| path.metadata().unwrap().len() <= 64),
            "a rotated log exceeded the byte cap: {logs:?}"
        );
    }

    #[test]
    fn bootstrap_log_writer_rotates_at_byte_cap_and_bounds_retention() {
        use std::io::Write as _;

        let root = tempfile::tempdir().unwrap();
        let mut writer = new_bootstrap_log_writer(root.path(), 64, 1).unwrap();
        for index in 0..20 {
            writeln!(writer, "bootstrap failure line {index:02}").unwrap();
        }
        drop(writer);

        let log_dir = root.path().join("logs");
        let mut logs = std::fs::read_dir(&log_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("wenlan-server.bootstrap.log"))
            })
            .collect::<Vec<_>>();
        logs.sort();

        assert_eq!(logs.len(), 2, "current log plus one rotation: {logs:?}");
        assert!(
            logs.iter().all(|path| path.metadata().unwrap().len() <= 64),
            "a bootstrap log exceeded the byte cap: {logs:?}"
        );
    }

    #[test]
    fn server_log_rate_limit_suppresses_duplicate_bursts() {
        use tracing_subscriber::prelude::*;

        let rate_limit = new_server_log_rate_limit();
        let metrics_layer = rate_limit.clone();
        let metrics = metrics_layer.metrics();
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::sync::Mutex::new(Vec::<u8>::new()))
                .with_filter(rate_limit),
        );

        tracing::subscriber::with_default(subscriber, || {
            for _ in 0..100 {
                tracing::warn!("identical repeatable failure");
            }
        });

        assert!(
            metrics.events_suppressed() > 0,
            "an identical 100-event burst must be throttled"
        );
    }

    #[test]
    fn rotating_log_construction_reports_an_unusable_directory_without_panicking() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("logs"), "not a directory").unwrap();

        let result = std::panic::catch_unwind(|| new_server_log_writer(root.path(), 64, 1));

        assert!(result.is_ok(), "logger construction must not panic");
        assert!(
            result.unwrap().is_err(),
            "unusable log paths must fail loud"
        );
    }

    #[test]
    fn bootstrap_logging_falls_back_when_the_data_root_is_unwritable() {
        let primary = tempfile::tempdir().unwrap();
        let fallback = tempfile::tempdir().unwrap();
        std::fs::write(primary.path().join("logs"), "not a directory").unwrap();

        let path =
            write_bootstrap_message(primary.path(), fallback.path(), "bootstrap sentinel").unwrap();

        assert!(path.starts_with(fallback.path()));
        assert!(std::fs::read_to_string(path)
            .unwrap()
            .contains("bootstrap sentinel"));
    }

    #[test]
    fn honors_env_when_set() {
        let _guard = env_lock().lock().unwrap();
        std::env::set_var("WENLAN_BIND_ADDR", "0.0.0.0:9090");
        assert_eq!(resolve_bind_addr(7878), "0.0.0.0:9090");
        std::env::remove_var("WENLAN_BIND_ADDR");
    }

    #[test]
    fn applied_unverified_repair_blocks_startup_projection_writers() {
        assert!(!startup_projection_writes_allowed(true));
        assert!(startup_projection_writes_allowed(false));
    }

    #[test]
    fn startup_repair_claim_requires_the_complete_exact_pair() {
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        assert!(
            Cli::try_parse_from(["wenlan-server", "--repair-manifest-id", manifest_id,]).is_err()
        );
        assert!(Cli::try_parse_from([
            "wenlan-server",
            "--repair-manifest-digest",
            &"a".repeat(64),
        ])
        .is_err());
    }

    #[test]
    fn startup_repair_claim_validates_the_stored_manifest_digest() {
        let root = tempfile::tempdir().unwrap();
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        let manifest_dir = root.path().join(manifest_id);
        std::fs::create_dir_all(&manifest_dir).unwrap();
        std::fs::write(
            manifest_dir.join("manifest.json"),
            include_bytes!("../../wenlan-types/testdata/repair/v1/manifest.json"),
        )
        .unwrap();
        let store = wenlan_core::repair::RepairArtifactStore::new(root.path().to_path_buf());
        let claim = StartupRepairClaim::try_new(
            Some(manifest_id.to_string()),
            Some("6d79617ffac084a9668025d2a870aa569b5381ea62513c4fa57d9f1a1620bf34".to_string()),
        )
        .unwrap()
        .unwrap();

        validate_startup_repair_claim(&store, &claim).unwrap();
        let wrong =
            StartupRepairClaim::try_new(Some(manifest_id.to_string()), Some("a".repeat(64)))
                .unwrap()
                .unwrap();
        assert!(validate_startup_repair_claim(&store, &wrong).is_err());
    }

    #[test]
    fn startup_repair_claim_selects_one_fence_and_rejects_a_different_pending_repair() {
        let claim = StartupRepairClaim::try_new(
            Some("repair_550e8400-e29b-41d4-a716-446655440000".to_string()),
            Some("a".repeat(64)),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            select_startup_repair_fence(&[], Some(&claim)).unwrap(),
            Some(claim.manifest_id().to_string())
        );
        assert_eq!(
            select_startup_repair_fence(&[claim.manifest_id().to_string()], Some(&claim)).unwrap(),
            Some(claim.manifest_id().to_string())
        );
        assert!(select_startup_repair_fence(&["repair_other".to_string()], Some(&claim)).is_err());
    }

    #[test]
    fn startup_repair_claim_constructs_the_exact_approved_apply() {
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        let digest = "a".repeat(64);
        let claim =
            StartupRepairClaim::try_new(Some(manifest_id.to_string()), Some(digest.clone()))
                .unwrap()
                .unwrap();

        let request = claim.apply_request().unwrap();
        assert_eq!(request.manifest_id(), manifest_id);
        assert_eq!(request.approved_manifest_digest().as_str(), digest);
        assert_eq!(
            request.approval(),
            format!("apply repair {manifest_id} {digest}")
        );
    }

    #[test]
    fn stored_pending_repair_reconstructs_exact_startup_authority() {
        let root = tempfile::tempdir().unwrap();
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        let manifest_dir = root.path().join(manifest_id);
        std::fs::create_dir_all(&manifest_dir).unwrap();
        std::fs::write(
            manifest_dir.join("manifest.json"),
            include_bytes!("../../wenlan-types/testdata/repair/v1/manifest.json"),
        )
        .unwrap();
        let store = wenlan_core::repair::RepairArtifactStore::new(root.path().to_path_buf());

        let request = stored_repair_apply_request(&store, manifest_id).unwrap();
        assert_eq!(request.manifest_id(), manifest_id);
        assert_eq!(
            request.approved_manifest_digest().as_str(),
            "6d79617ffac084a9668025d2a870aa569b5381ea62513c4fa57d9f1a1620bf34"
        );
        assert_eq!(
            request.approval(),
            format!(
                "apply repair {manifest_id} {}",
                request.approved_manifest_digest().as_str()
            )
        );
    }

    #[test]
    fn startup_repair_claim_disables_optional_runtime_workers() {
        assert!(!optional_runtime_workers_allowed(true));
        assert!(optional_runtime_workers_allowed(false));
    }

    #[test]
    fn startup_model_working_set_uses_the_registry_ram_requirement() {
        let model = wenlan_core::on_device_models::get_model("qwen3-4b").unwrap();
        assert_eq!(
            on_device_model_working_set_bytes(model),
            3 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn startup_repair_claim_cannot_succeed_via_an_existing_daemon() {
        assert!(!existing_daemon_may_satisfy_startup(true));
        assert!(existing_daemon_may_satisfy_startup(false));
    }

    #[test]
    fn startup_repair_claim_forces_loopback_bind() {
        let _guard = env_lock().lock().unwrap();
        std::env::set_var("WENLAN_BIND_ADDR", "0.0.0.0:9090");
        assert_eq!(resolve_startup_bind_addr(7878, true), "127.0.0.1:7878");
        assert_eq!(resolve_startup_bind_addr(7878, false), "0.0.0.0:9090");
        std::env::remove_var("WENLAN_BIND_ADDR");
    }

    #[test]
    fn startup_repair_claim_requires_the_canonical_daemon_port() {
        assert_eq!(resolve_startup_port(7878, true).unwrap(), 7878);
        assert!(resolve_startup_port(7879, true).is_err());
        assert_eq!(resolve_startup_port(7879, false).unwrap(), 7879);
    }

    #[test]
    fn data_root_lock_excludes_a_second_daemon_for_the_same_root() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("wenlan");
        let first = DaemonDataLock::acquire(&root, false).unwrap();

        assert!(
            !root.exists(),
            "normal lock acquisition must not create the data root"
        );
        assert!(DaemonDataLock::acquire(&root, false).is_err());
        drop(first);
        DaemonDataLock::acquire(&root, false)
            .expect("dropping the owner releases the data-root lock");
    }

    #[test]
    fn repair_data_root_lock_refuses_to_create_a_missing_root() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("missing-wenlan");

        assert!(DaemonDataLock::acquire(&root, true).is_err());
        assert!(!root.exists());
    }

    #[test]
    fn normal_data_root_lock_does_not_suppress_legacy_migration() {
        let parent = tempfile::tempdir().unwrap();
        let legacy = parent.path().join("origin");
        let root = parent.path().join("wenlan");
        std::fs::create_dir_all(legacy.join("memorydb")).unwrap();
        std::fs::write(legacy.join("memorydb/origin_memory.db"), b"legacy-db").unwrap();

        let _lock = DaemonDataLock::acquire(&root, false).unwrap();
        assert!(!root.exists());
        assert_eq!(
            wenlan_core::migrate_rename::migrate_dir(&legacy, &root).unwrap(),
            wenlan_core::migrate_rename::MigrationOutcome::Migrated
        );
        assert_eq!(
            std::fs::read(root.join("memorydb/origin_memory.db")).unwrap(),
            b"legacy-db"
        );
    }

    #[test]
    fn data_root_lock_child_process_holds_lock() {
        let Some(root) = std::env::var_os("WENLAN_DATA_LOCK_CHILD_ROOT") else {
            return;
        };
        let ready =
            std::path::PathBuf::from(std::env::var_os("WENLAN_DATA_LOCK_CHILD_READY").unwrap());
        let release =
            std::path::PathBuf::from(std::env::var_os("WENLAN_DATA_LOCK_CHILD_RELEASE").unwrap());
        let _lock = DaemonDataLock::acquire(std::path::Path::new(&root), true).unwrap();
        std::fs::write(&ready, b"ready").unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !release.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(release.exists(), "parent did not release child lock test");
    }

    #[test]
    fn data_root_lock_excludes_another_process_with_a_different_temp_dir() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("wenlan");
        let child_tmp = parent.path().join("other-tmp");
        let ready = parent.path().join("child-ready");
        let release = parent.path().join("child-release");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&child_tmp).unwrap();

        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "bind_addr_tests::data_root_lock_child_process_holds_lock",
                "--nocapture",
            ])
            .env("WENLAN_DATA_LOCK_CHILD_ROOT", &root)
            .env("WENLAN_DATA_LOCK_CHILD_READY", &ready)
            .env("WENLAN_DATA_LOCK_CHILD_RELEASE", &release)
            .env("TMPDIR", &child_tmp)
            .env("TMP", &child_tmp)
            .env("TEMP", &child_tmp)
            .spawn()
            .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !ready.exists() && std::time::Instant::now() < deadline {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("lock-holder child exited early with {status}");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(ready.exists(), "lock-holder child did not become ready");
        assert!(DaemonDataLock::acquire(&root, true).is_err());

        std::fs::write(&release, b"release").unwrap();
        assert!(child.wait().unwrap().success());
    }
}

// All other modules live in the library target (src/lib.rs) so that
// integration tests in tests/ can reference them as wenlan_server::<mod>.
use wenlan_server::{
    ingest_batcher, lifecycle, router, scheduler,
    state::{ServerState, SharedState},
};

use clap::{Parser, Subcommand};
use std::{future::IntoFuture, io::Write, sync::Arc};
use tokio::sync::RwLock;

const SHUTDOWN_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(1_500);

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "_exit"]
    fn unix_process_exit(status: std::ffi::c_int) -> !;
}

fn exit_daemon(code: i32) -> ! {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(unix)]
    {
        // SAFETY: The daemon-owned HTTP and scheduler tasks have already
        // drained. `_exit` terminates the process without running C `atexit`
        // handlers, which could otherwise tear down Metal globals while a
        // deliberately detached blocking inference worker still owns them.
        unsafe { unix_process_exit(code) }
    }
    #[cfg(not(unix))]
    {
        std::process::exit(code)
    }
}

#[cfg(unix)]
struct TerminationSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

#[cfg(unix)]
fn install_termination_signals() -> std::io::Result<TerminationSignals> {
    Ok(TerminationSignals {
        interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?,
        terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
    })
}

#[cfg(unix)]
impl TerminationSignals {
    async fn wait(mut self) {
        tokio::select! {
            _ = self.interrupt.recv() => {}
            _ = self.terminate.recv() => {}
        }
    }
}

#[cfg(windows)]
struct TerminationSignals {
    ctrl_c: tokio::signal::windows::CtrlC,
}

#[cfg(windows)]
fn install_termination_signals() -> std::io::Result<TerminationSignals> {
    Ok(TerminationSignals {
        ctrl_c: tokio::signal::windows::ctrl_c()?,
    })
}

#[cfg(windows)]
impl TerminationSignals {
    async fn wait(mut self) {
        let _ = self.ctrl_c.recv().await;
    }
}

#[cfg(not(any(unix, windows)))]
struct TerminationSignals;

#[cfg(not(any(unix, windows)))]
fn install_termination_signals() -> std::io::Result<TerminationSignals> {
    Ok(TerminationSignals)
}

#[cfg(not(any(unix, windows)))]
impl TerminationSignals {
    async fn wait(self) {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(debug_assertions)]
async fn wait_at_startup_signal_test_barrier() -> anyhow::Result<()> {
    let Some(root) = std::env::var_os("WENLAN_TEST_STARTUP_SIGNAL_BARRIER") else {
        return Ok(());
    };
    let root = std::path::PathBuf::from(root);
    let ready = root.join("ready");
    let release = root.join("release");
    std::fs::write(&ready, b"ready")?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !release.exists() {
        if std::time::Instant::now() >= deadline {
            anyhow::bail!("startup signal test barrier timed out");
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    Ok(())
}

/// Wenlan memory daemon — headless HTTP server.
#[derive(Parser)]
#[command(
    name = "wenlan-server",
    bin_name = "wenlan-server",
    version,
    about = "Wenlan headless HTTP daemon."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Override the data directory (for isolated dev/demo runs).
    /// When set, the daemon reads/writes the DB at `<dir>/memorydb/origin_memory.db`
    /// and config at `<dir>/config.json` instead of the default
    /// the platform data directory under `dirs::data_local_dir().join("wenlan/")`.
    /// macOS: `~/Library/Application Support/wenlan/`. Linux: `~/.local/share/wenlan/`. Windows: `%LOCALAPPDATA%\origin\`. Also honored via `WENLAN_DATA_DIR` env.
    #[arg(long, global = true)]
    data_dir: Option<std::path::PathBuf>,

    /// Override the HTTP port (default 7878). Useful when running a scratch
    /// daemon alongside the main one. Also honored via `WENLAN_PORT` env.
    #[arg(long, global = true)]
    port: Option<u16>,

    /// Internal repair-only startup claim. Both exact fields are required.
    #[arg(long, global = true, hide = true, requires = "repair_manifest_digest")]
    repair_manifest_id: Option<String>,

    /// Approved digest for the exact repair-only startup claim.
    #[arg(long, global = true, hide = true, requires = "repair_manifest_id")]
    repair_manifest_digest: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Internal maintenance: delete archived stale pages. Daemon must be stopped first.
    #[command(name = "backfill-stale-pages", hide = true)]
    BackfillStalePages {
        /// Print candidates without modifying the database.
        #[arg(long)]
        dry_run: bool,
    },
}

async fn run_daemon(startup_repair_claim: Option<StartupRepairClaim>) -> anyhow::Result<()> {
    let startup_repair_claimed = startup_repair_claim.is_some();
    // Register with the OS before binding or touching durable state. Creating
    // Tokio's platform signal streams installs the handlers synchronously, so
    // SIGTERM/CTRL_C received during startup is retained for the waiter below
    // instead of taking the process down through the platform default path.
    let termination_signals = install_termination_signals()
        .map_err(|error| anyhow::anyhow!("install termination signal handlers: {error}"))?;
    let wenlan_root = resolve_wenlan_root();

    // Port (clap `--port`/`WENLAN_PORT` → env var set by main(); read here)
    let configured_port: u16 = wenlan_core::env_compat::var_compat("WENLAN_PORT")
        .and_then(|v| v.into_string().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(7878);
    let port = resolve_startup_port(configured_port, startup_repair_claimed)?;

    // Bind BEFORE touching the data dir. Losing the port race must be free:
    // under launchd KeepAlive, a retry loop that first runs full MemoryDB init
    // (schema/FTS writes + embedder load) hammers the live daemon's SQLite
    // file every ~10s — enough lock/CPU pressure to wedge the daemon that
    // actually owns the port.
    let addr = resolve_startup_bind_addr(port, startup_repair_claimed);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            if !existing_daemon_may_satisfy_startup(startup_repair_claimed) {
                return Err(anyhow::anyhow!(
                    "repair-only daemon failed to acquire {}: {}",
                    addr,
                    e
                ));
            }
            // Check if existing daemon is healthy
            eprintln!("Failed to bind {addr}: {e}");
            let url = format!("http://127.0.0.1:{}/api/health", port);
            // Bounded probe: a mute port-holder (accepts, never responds)
            // must not hang this process forever — under launchd KeepAlive
            // a hung loser also blocks the retry that would recover things.
            let probe = reqwest::Client::new()
                .get(&url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await;
            match probe {
                Ok(resp) if resp.status().is_success() => {
                    // Port already taken by a healthy daemon. If launchd is the
                    // parent (XPC_SERVICE_NAME set), exit non-zero so launchd
                    // retries after ThrottleInterval — otherwise launchd marks
                    // this attempt as a clean exit and refuses to respawn even
                    // after the winning daemon dies (KeepAlive.SuccessfulExit
                    // = false treats exit-0 as success). For sidecar invocation
                    // by the app, exit 0 is the right answer.
                    if std::env::var_os("XPC_SERVICE_NAME").is_some() {
                        report_bootstrap_error(
                            &wenlan_root,
                            &format!(
                                "Existing healthy daemon on port {port} — exiting 75 (launchd retry)"
                            ),
                        );
                        std::process::exit(75);
                    }
                    eprintln!("Existing healthy daemon on port {port} — exiting cleanly");
                    return Ok(());
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Port {} in use and no healthy daemon",
                        port
                    ));
                }
            }
        }
    };

    init_logging(&wenlan_root)?;
    std::panic::set_hook(Box::new(|panic| {
        tracing::error!("panic: {panic}");
    }));
    tracing::info!("wenlan-server v{}", wenlan_core::version());

    #[cfg(debug_assertions)]
    wait_at_startup_signal_test_barrier().await?;
    // Data directory. `WENLAN_DATA_DIR` (set by `--data-dir` flag) overrides the
    // default, enabling isolated dev/demo runs (e.g. `--data-dir /tmp/wenlan-demo`).
    let data_dir = wenlan_root.join("memorydb");
    tracing::info!("Wenlan data root: {}", wenlan_root.display());
    let _data_root_lock = DaemonDataLock::acquire(&wenlan_root, startup_repair_claimed)?;

    let repair_store = wenlan_core::repair::RepairArtifactStore::new(wenlan_root.join("repairs"));
    if let Some(claim) = startup_repair_claim.as_ref() {
        validate_startup_repair_claim(&repair_store, claim)?;
        tracing::warn!(
            manifest_id = claim.manifest_id(),
            "validated exact repair-only startup claim before opening the database"
        );
    }

    // Inspect durable repair state before opening the database. A prepared
    // exact claim and any applied-unverified receipt must identify the same
    // manifest; otherwise startup fails without touching canonical data.
    let pending_repairs = repair_store
        .pending_verification_manifest_ids()
        .map_err(|error| anyhow::anyhow!("inspect durable repair state: {error}"))?;
    let startup_repair_fence =
        select_startup_repair_fence(&pending_repairs, startup_repair_claim.as_ref())?;
    let repair_recovery_pending = startup_repair_fence.is_some();

    // One-time origin -> wenlan migration is an ordinary startup write. Run it
    // only after durable repair inspection has proved no repair fence exists.
    if !repair_recovery_pending && wenlan_core::env_compat::var_compat("WENLAN_DATA_DIR").is_none()
    {
        if let Some(dl) = dirs::data_local_dir() {
            wenlan_core::migrate_rename::migrate_and_log(&dl.join("origin"), &dl.join("wenlan"));
        }
    }
    if !repair_recovery_pending {
        if let Some(home) = dirs::home_dir() {
            wenlan_core::migrate_rename::migrate_and_log(
                &home.join(".origin"),
                &home.join(".wenlan"),
            );
        }
    }

    // Build state and restore the process-local fence while recovery is still
    // sealed. No background acquisition can start before `finish_recovery`.
    let mut server_state = ServerState::new();
    server_state.optional_runtime_workers_suspended = repair_recovery_pending;
    server_state.repair_root = Some(repair_store.root().to_path_buf());
    let startup_repair_authority = match startup_repair_claim.as_ref() {
        Some(claim) => Some(claim.apply_request()?),
        None => startup_repair_fence
            .as_deref()
            .map(|manifest_id| stored_repair_apply_request(&repair_store, manifest_id))
            .transpose()?,
    };
    if let Some(request) = startup_repair_authority {
        let manifest_id = request.manifest_id().to_string();
        server_state
            .maintenance_coordinator
            .rearm_approved_repair(request)
            .map_err(|error| anyhow::anyhow!("restore exact repair writer fence: {error}"))?;
        tracing::warn!(
            manifest_id,
            prepared_claim = startup_repair_claimed,
            "restored exact repair authority before startup writers"
        );
    }
    // Repair mode refuses schema drift and skips every ordinary constructor
    // side effect (schema/migrations/profile bootstrap/embedder load). Normal
    // startup retains the existing fully initialized path.
    let db = if repair_recovery_pending {
        tracing::warn!("Opening current database in side-effect-free repair mode");
        wenlan_core::db::MemoryDB::open_for_repair(&data_dir).await?
    } else {
        let emitter: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::NoopEmitter);
        tracing::info!("Initializing MemoryDB at {}", data_dir.display());
        wenlan_core::db::MemoryDB::new(&data_dir, emitter).await?
    };
    let db_arc = Arc::new(db);
    server_state.db = Some(db_arc.clone());

    // Run migration-55 backfill (event_date regex Pass A + memory_entities Pass B)
    // before the HTTP listener binds so no ingest races the backfill. Idempotent.
    if repair_recovery_pending {
        tracing::warn!("skipping first-boot data backfill until repair verification completes");
    } else {
        tracing::info!(
            "Running first-boot data backfill (event dates + knowledge-graph links); \
             this can take a moment on large databases…"
        );
        let m55 = db_arc.run_migration_55().await.map_err(|e| {
            anyhow::anyhow!("running migration 55 (event_date + memory_entities backfill): {e}")
        })?;
        tracing::info!(
            "First-boot backfill complete: scanned {} memories for dates, inserted {} entity links",
            m55.event_dates_scanned,
            m55.entity_links_inserted
        );
    }

    // Requeue any document-enrichment rows left `in_progress` by a previous run
    // (a crash / restart mid-enrichment). Their per-chunk checkpoint is
    // preserved, so the scheduler resumes them from where they stopped rather
    // than re-analyzing from scratch — restart-from-checkpoint, no manual step.
    if !repair_recovery_pending {
        match db_arc.reset_in_progress_documents().await {
            Ok(0) => {}
            Ok(n) => tracing::info!("[doc-enrich] requeued {n} in-progress document(s) for resume"),
            Err(e) => tracing::warn!("[doc-enrich] reset_in_progress_documents failed: {e}"),
        }
    }

    // Consolidate user-facing assets under ~/.wenlan/.
    // - Ensure ~/.wenlan/{pages, sessions, sessions/_status} exist
    // - Symlink ~/.wenlan/db -> <data_dir> (cosmetic alias; DB stays at
    //   the platform data directory (resolved via `dirs::data_local_dir()` per OS)
    //   under `wenlan/memorydb/`, to avoid moving live SQLite/WAL files mid-flight).
    // - Migrate legacy ~/Origin/knowledge/ md files into ~/.wenlan/pages/ if
    //   the new dir is empty. Never deletes the old dir; user can clean up
    //   manually after verifying.
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        if let Some(home) = dirs::home_dir() {
            let wenlan_dot = home.join(".wenlan");
            for sub in ["pages", "sessions", "sessions/_status"] {
                if let Err(e) = std::fs::create_dir_all(wenlan_dot.join(sub)) {
                    tracing::warn!("[wenlan-dir] create {} failed: {}", sub, e);
                }
            }

            let db_link = wenlan_dot.join("db");
            let link_target_already_correct = std::fs::read_link(&db_link)
                .map(|t| t == data_dir)
                .unwrap_or(false);
            if !link_target_already_correct && !db_link.exists() {
                #[cfg(unix)]
                if let Err(e) = std::os::unix::fs::symlink(&data_dir, &db_link) {
                    tracing::warn!(
                        "[wenlan-dir] symlink {} -> {} failed: {}",
                        db_link.display(),
                        data_dir.display(),
                        e
                    );
                }
                #[cfg(windows)]
                {
                    tracing::info!(
                        "Database at {} (no shortcut created; Windows symlinks require admin).",
                        data_dir.display()
                    );
                }
            }

            let legacy_pages = home.join("Origin/knowledge");
            let new_pages = wenlan_dot.join("pages");
            let legacy_has_md = std::fs::read_dir(&legacy_pages)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
                })
                .unwrap_or(false);
            let new_is_empty = std::fs::read_dir(&new_pages)
                .map(|entries| {
                    !entries
                        .filter_map(|e| e.ok())
                        .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("md"))
                })
                .unwrap_or(true);
            if startup_projection_writes_allowed(repair_recovery_pending)
                && legacy_has_md
                && new_is_empty
            {
                tracing::info!(
                    "[migrate] copying md files from {} to {}",
                    legacy_pages.display(),
                    new_pages.display()
                );
                if let Ok(entries) = std::fs::read_dir(&legacy_pages) {
                    let mut copied = 0usize;
                    for entry in entries.filter_map(|e| e.ok()) {
                        let src = entry.path();
                        if src.extension().and_then(|s| s.to_str()) != Some("md") {
                            continue;
                        }
                        if let Some(name) = src.file_name() {
                            let dst = new_pages.join(name);
                            if dst.exists() {
                                continue;
                            }
                            match std::fs::copy(&src, &dst) {
                                Ok(_) => copied += 1,
                                Err(e) => tracing::warn!(
                                    "[migrate] copy {} -> {} failed: {}",
                                    src.display(),
                                    dst.display(),
                                    e
                                ),
                            }
                        }
                    }
                    tracing::info!("[migrate] copied {} md files from legacy path", copied);
                }
            }

            // Initialize ~/.wenlan/ as a git repo so users get version history
            // of pages + sessions for free. Defensive — silent skip if git is
            // missing or any step fails. Skills (/handoff, /distill, /forget)
            // commit per logical batch; daemon only does the initial bring-up
            // here.
            let dot_git = wenlan_dot.join(".git");
            let git_available = std::process::Command::new("git")
                .arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !dot_git.exists() && git_available {
                let gitignore = wenlan_dot.join(".gitignore");
                if !gitignore.exists() {
                    // No trailing slash on `db` / `bin` — those entries are
                    // symlinks in the consolidated layout, and pattern `db/`
                    // would only match real directories.
                    let _ = std::fs::write(
                        &gitignore,
                        "db\nbin\nlogs/\nsessions/_status/handoff-*.json\n",
                    );
                }
                let run = |args: &[&str]| {
                    std::process::Command::new("git")
                        .args(args)
                        .current_dir(&wenlan_dot)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status()
                        .ok()
                        .filter(|s| s.success())
                };
                if run(&["init", "--quiet"]).is_some() {
                    let _ = run(&[
                        "-c",
                        "user.name=Wenlan",
                        "-c",
                        "user.email=daemon@origin.local",
                        "commit",
                        "--allow-empty",
                        "--quiet",
                        "-m",
                        "Wenlan initialized",
                    ]);
                    let _ = run(&["add", "-A"]);
                    let _ = run(&[
                        "-c",
                        "user.name=Wenlan",
                        "-c",
                        "user.email=daemon@origin.local",
                        "commit",
                        "--quiet",
                        "-m",
                        "backfill: initial pages from DB",
                    ]);
                    tracing::info!("[wenlan-dir] git init complete at {}", wenlan_dot.display());
                }
            }
        }
    }

    // One-time backfill: if the knowledge directory is empty but the DB has
    // active pages, write them all to disk. Handles the case where pages were
    // created before KnowledgeWriter was wired up, or via a code path that
    // bypasses the writer.
    //
    // We gate on a `.origin/.backfill-attempted` marker file (created on
    // first attempt regardless of outcome) so this block only runs once per
    // daemon install. Without the marker, a persistent write_page
    // failure — e.g. permission error on the destination directory — would
    // re-trigger a full DB scan + write attempt on every single startup.
    {
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
        let marker_path = knowledge_path.join(".wenlan").join(".backfill-attempted");

        let already_attempted = marker_path.exists();
        let has_md_files = !already_attempted
            && knowledge_path.exists()
            && std::fs::read_dir(&knowledge_path)
                .map(|entries| {
                    entries.filter_map(|e| e.ok()).any(|e| {
                        e.path()
                            .extension()
                            .and_then(|s| s.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("md"))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false);

        if startup_projection_writes_allowed(repair_recovery_pending)
            && !already_attempted
            && !has_md_files
        {
            match db_arc.list_pages("active", 10_000, 0).await {
                Ok(pages) if !pages.is_empty() => {
                    tracing::info!(
                        "[backfill] knowledge dir empty; writing {} pages to {}",
                        pages.len(),
                        knowledge_path.display()
                    );
                    let projection = wenlan_core::export::knowledge::KnowledgeProjectionWrite::new(
                        knowledge_path.clone(),
                        &db_arc,
                    );
                    let mut written = 0usize;
                    let mut failed = 0usize;
                    for page in &pages {
                        match projection.write_page(page) {
                            Ok(_) => written += 1,
                            Err(e) => {
                                tracing::warn!(
                                    "[backfill] write_page failed for {}: {}",
                                    page.id,
                                    e
                                );
                                failed += 1;
                            }
                        }
                    }
                    tracing::info!("[backfill] wrote {} pages, {} failed", written, failed);

                    // Create the marker file so we don't re-run the
                    // backfill on every subsequent startup — even if every
                    // write_page above failed. The user can delete
                    // `.origin/.backfill-attempted` to force a retry.
                    if let Some(parent) = marker_path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    if let Err(e) = std::fs::write(&marker_path, "") {
                        tracing::warn!(
                            "[backfill] failed to write marker {}: {}",
                            marker_path.display(),
                            e
                        );
                    }
                }
                Ok(_) => {
                    // DB has no pages yet — nothing to backfill. Don't create
                    // the marker; the next startup after pages exist should retry.
                }
                Err(e) => {
                    tracing::warn!("[backfill] list_pages failed: {}", e);
                }
            }
        }
    }

    // Startup reconcile: repair the markdown projection from the DB.
    //
    // `write_page` renames a temp file over the target without an fsync — that
    // buys readers atomicity, not crash durability. So a crash can leave a
    // page's file missing, holding the previous version's bytes, or
    // zero-length, plus `.tmp` orphans from a write that died mid-rename.
    // This is the pass that makes "the md is a repairable projection" true.
    //
    // Runs synchronously, before `axum::serve`, for the same reason the
    // backfill above does: no HTTP write and no scheduler tick can race the
    // repair, so the pass needs no locking. The listener is already bound
    // (see the bind-first block up top), so a slow pass on a large corpus
    // delays serving, never the port handoff.
    //
    // ponytail: same 10k page ceiling as the backfill, and one pass reads
    // every projected file. If a corpus ever outgrows that, page the scan or
    // move it behind the listener — do NOT background it naively, since a
    // concurrent page write would race the repair.
    {
        let knowledge_path = wenlan_core::config::load_config().knowledge_path_or_default();
        if knowledge_path.exists() {
            match db_arc.list_pages("active", 10_000, 0).await {
                Ok(pages) => {
                    let projection = wenlan_core::export::knowledge::KnowledgeProjectionWrite::new(
                        knowledge_path.clone(),
                        &db_arc,
                    );
                    match projection.reconcile(&pages) {
                        Ok(stats)
                            if stats.rewritten > 0
                                || stats.temp_files_removed > 0
                                || stats.errors > 0 =>
                        {
                            tracing::info!(
                                "[reconcile] projection repaired: {} checked, {} rewritten, \
                                 {} temp leftover(s) swept, {} failed",
                                stats.checked,
                                stats.rewritten,
                                stats.temp_files_removed,
                                stats.errors
                            );
                        }
                        Ok(stats) => {
                            tracing::debug!(
                                "[reconcile] {} page(s) checked, all clean",
                                stats.checked
                            );
                        }
                        Err(e) => tracing::warn!("[reconcile] pass failed: {e}"),
                    }
                }
                Err(e) => tracing::warn!("[reconcile] list_pages failed: {e}"),
            }
        }
    }

    // Load intelligence config
    server_state.prompts = wenlan_core::prompts::PromptRegistry::load(
        &wenlan_core::prompts::PromptRegistry::override_dir(),
    );
    server_state.tuning =
        wenlan_core::tuning::TuningConfig::load(&wenlan_core::tuning::TuningConfig::config_path());
    server_state.quality_gate =
        wenlan_core::quality_gate::QualityGate::new(server_state.tuning.gate.clone());

    // Load API LLM providers if configured
    let config = wenlan_core::config::load_config();
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        if let Some(ref key) = config.anthropic_api_key {
            if !key.is_empty() {
                let routine_model = config.routine_model.clone().unwrap_or_else(|| {
                    wenlan_core::llm_provider::DEFAULT_ROUTINE_MODEL.to_string()
                });
                let provider =
                    wenlan_core::llm_provider::ApiProvider::new(key.clone(), routine_model);
                server_state.api_llm = Some(Arc::new(provider));
                tracing::info!("API LLM provider initialized (routine)");

                let synthesis_model = config
                    .synthesis_model
                    .clone()
                    .unwrap_or_else(|| "claude-sonnet-4-6".to_string());
                let provider =
                    wenlan_core::llm_provider::ApiProvider::new(key.clone(), synthesis_model);
                server_state.synthesis_llm = Some(Arc::new(provider));
                tracing::info!("Synthesis LLM provider initialized");
            }
        }

        // Load external LLM provider if configured
        if let (Some(ref endpoint), Some(ref model)) =
            (&config.external_llm_endpoint, &config.external_llm_model)
        {
            if !endpoint.is_empty() && !model.is_empty() {
                let provider = wenlan_core::llm_provider::OpenAICompatibleProvider::new_with_key(
                    endpoint.clone(),
                    model.clone(),
                    config.external_llm_api_key.clone(),
                );
                server_state.external_llm = Some(Arc::new(provider));
                tracing::info!("External LLM provider initialized from config");
            }
        }
    }

    // Cross-encoder reranker wiring. `WENLAN_RERANKER_MODE = off|lite|full` (default
    // off) selects which retrieval paths get a CE and which model; the legacy
    // `WENLAN_RERANKER_ENABLED=1` (with MODE unset) maps to deep-only CE using the
    // configured model — exactly the pre-mode behavior. First construction downloads
    // weights (turbo ~146MB, bge-base ~1.1GB) into the shared FastEmbed cache;
    // failure is non-fatal (the affected path falls back to embedding+FTS ordering).
    let reranker_cache_dir = wenlan_core::db::resolve_fastembed_cache_dir(&data_dir);
    let mut deep_bgebase_pending = false;
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        use wenlan_core::reranker::{RerankerMode, RerankerPick};
        use wenlan_types::responses::RerankerStatus;
        let mode = wenlan_core::reranker::reranker_mode_resolved(&config);
        let legacy_enabled = std::env::var("WENLAN_RERANKER_ENABLED").as_deref() == Ok("1");
        let plan = wenlan_core::reranker::resolve_reranker_plan(mode, legacy_enabled);
        server_state.reranker_mode = match mode {
            RerankerMode::Off => "off",
            RerankerMode::Lite => "lite",
            RerankerMode::Full => "full",
        }
        .to_string();
        tracing::info!(
            "[reranker] mode={} (legacy_enabled={legacy_enabled}); light={:?} deep={:?}",
            server_state.reranker_mode,
            plan.light,
            plan.deep
        );

        // Light paths (quick `/api/search` + context `/api/context`): turbo
        // (~146MB), eager-load — small enough not to meaningfully block startup.
        let mut light_reranker: Option<Arc<dyn wenlan_core::reranker::Reranker>> = None;
        if let Some(pick) = plan.light {
            let cache = reranker_cache_dir.clone();
            match tokio::task::spawn_blocking(move || {
                wenlan_core::reranker::init_cross_encoder_reranker_pick(pick, cache)
            })
            .await
            {
                Ok(Ok(r)) => {
                    let model_id = r.model_id().to_string();
                    tracing::info!("[reranker] light paths active (model={model_id})");
                    server_state.reranker_light_status = RerankerStatus::Active { model_id };
                    light_reranker = Some(r.clone());
                    server_state.reranker_light = Some(r);
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        "[reranker] light init failed; quick + context fall back to plain hybrid: {e}"
                    );
                    server_state.reranker_light_status = RerankerStatus::Failed {
                        reason: e.to_string(),
                    };
                }
                Err(e) => {
                    tracing::warn!("[reranker] light init join failed: {e}");
                    server_state.reranker_light_status = RerankerStatus::Failed {
                        reason: e.to_string(),
                    };
                }
            }
        }

        // Deep path (`/api/memory/search` with rerank=true).
        match plan.deep {
            // Back-compat: ENABLED=1 + mode unset -> eager-load the configured model
            // (+ BYO via WENLAN_RERANKER_ONNX_DIR), blocking startup, exactly as before.
            Some(RerankerPick::Configured) => {
                tracing::info!(
                    "[reranker] deep path (legacy WENLAN_RERANKER_ENABLED); first run downloads \
                     weights (~1.1GB). The daemon finishes starting once the model is ready\u{2026}"
                );
                let cache = reranker_cache_dir.clone();
                match tokio::task::spawn_blocking(move || {
                    wenlan_core::reranker::init_cross_encoder_reranker(cache)
                })
                .await
                {
                    Ok(Ok(r)) => {
                        let model_id = r.model_id().to_string();
                        tracing::info!("[reranker] deep path active (model={model_id})");
                        server_state.reranker_status = RerankerStatus::Active { model_id };
                        server_state.reranker = Some(r);
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(
                            "[reranker] deep init failed; rerank=true falls back to plain hybrid: {e}"
                        );
                        server_state.reranker_status = RerankerStatus::Failed {
                            reason: e.to_string(),
                        };
                    }
                    Err(e) => {
                        tracing::warn!("[reranker] deep init join failed: {e}");
                        server_state.reranker_status = RerankerStatus::Failed {
                            reason: e.to_string(),
                        };
                    }
                }
            }
            // lite: the deep path reuses the already-loaded turbo (no second load).
            // Mirror the light status either way so a FAILED turbo load surfaces as
            // deep=failed (not a misleading deep=disabled) on /api/status; the missing
            // Arc still makes rerank=true fall back to plain hybrid. (review fix)
            Some(RerankerPick::Turbo) => {
                server_state.reranker_status = server_state.reranker_light_status.clone();
                if let Some(r) = light_reranker.clone() {
                    server_state.reranker = Some(r);
                }
            }
            // full: heavy bge-base. Council fix #3 — do NOT block startup; load it in
            // the background after the state is shared (rerank=true falls back to plain
            // until ready). Status stays Disabled until the background load completes.
            Some(RerankerPick::BgeBase) => {
                deep_bgebase_pending = true;
            }
            None => {}
        }
    }

    // Import any legacy tag data from the pre-PR-B2 spaces.db file.
    if !repair_recovery_pending {
        match wenlan_core::spaces::import_legacy_tags(&db_arc).await {
            Ok(n) if n > 0 => {
                tracing::info!("[startup] imported {} legacy tag triples from spaces.db", n)
            }
            Ok(_) => {}
            Err(e) => tracing::warn!("[startup] legacy tags import failed: {e}"),
        }
    }

    // Spawn the ingest coalescer. HTTP `/api/memory/store` handlers submit
    // fully-built RawDocuments + pre-computed chunk counts; the coalescer
    // runs the full ingest pipeline (batched quality gate, partition,
    // upsert survivors) per flush window. This amortizes both the FastEmbed
    // invocation (one batched call per flush for gate's novelty check) AND
    // the libSQL transaction (one per flush for the survivors) across
    // concurrent writes.
    //
    // See `crates/wenlan-server/src/ingest_batcher.rs` for the design and
    // contract tests.
    {
        let db_for_batcher = db_arc.clone();
        let gate_for_batcher = server_state.quality_gate.clone();
        let maintenance_for_batcher = server_state.maintenance_coordinator.clone();
        let process: ingest_batcher::BatchProcessFn = Arc::new(
            move |items: Vec<(wenlan_core::sources::RawDocument, usize)>| {
                let db = db_for_batcher.clone();
                let gate = gate_for_batcher.clone();
                let maintenance = maintenance_for_batcher.clone();
                Box::pin(async move {
                    let _maintenance_guard = maintenance.begin_background().await;
                    ingest_batch_process(db, gate, items).await
                })
            },
        );
        server_state.ingest_batcher = Some(ingest_batcher::IngestBatcher::spawn(
            process,
            ingest_batcher::BatcherConfig::default(),
        ));
    }

    server_state.maintenance_coordinator.finish_recovery();

    let shared: SharedState = Arc::new(RwLock::new(server_state));

    // full mode: load the heavy deep bge-base in the background so startup never
    // blocks on the ~1.1GB download (council fix #3). rerank=true uses plain hybrid
    // until this completes; deep status flips to Active/Failed when the load resolves.
    if optional_runtime_workers_allowed(repair_recovery_pending) && deep_bgebase_pending {
        let shared_for_deep = shared.clone();
        let cache = reranker_cache_dir.clone();
        tokio::spawn(async move {
            use wenlan_types::responses::RerankerStatus;
            tracing::info!(
                "[reranker] full mode: loading deep bge-base in background (~1.1GB first run); \
                 rerank=true uses plain hybrid until ready\u{2026}"
            );
            let loaded = tokio::task::spawn_blocking(move || {
                wenlan_core::reranker::init_cross_encoder_reranker_pick(
                    wenlan_core::reranker::RerankerPick::BgeBase,
                    cache,
                )
            })
            .await;
            match loaded {
                Ok(Ok(r)) => {
                    let model_id = r.model_id().to_string();
                    let mut st = shared_for_deep.write().await;
                    st.reranker_status = RerankerStatus::Active {
                        model_id: model_id.clone(),
                    };
                    st.reranker = Some(r);
                    tracing::info!("[reranker] deep bge-base loaded and active (model={model_id})");
                }
                Ok(Err(e)) => {
                    let mut st = shared_for_deep.write().await;
                    st.reranker_status = RerankerStatus::Failed {
                        reason: e.to_string(),
                    };
                    tracing::warn!(
                        "[reranker] deep bge-base load failed; rerank=true stays on plain hybrid: {e}"
                    );
                }
                Err(e) => {
                    let mut st = shared_for_deep.write().await;
                    st.reranker_status = RerankerStatus::Failed {
                        reason: e.to_string(),
                    };
                    tracing::warn!("[reranker] deep bge-base load task panicked: {e}");
                }
            }
        });
    }

    // Initialize an explicitly selected, already-cached on-device LLM without
    // making daemon restart itself a foreground-heavy event. Selection still
    // supports explicit routes even when background source pins are absent, so
    // we keep preload semantics; the load now waits for two quiet CPU samples
    // and enough free memory for the registry working set *above* the normal
    // scheduler reserve. A reservation also prevents an automatic turn from
    // racing the load after observing the same quiet window.
    //
    // This intentionally does NOT trigger a download — users opt in explicitly
    // via the settings UI (POST /api/on-device-model/download).
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        let selected_model = config
            .on_device_model
            .as_deref()
            .map(|id| wenlan_core::on_device_models::resolve_or_default(Some(id)));
        match selected_model {
            None => tracing::info!(
                "[on-device] no local model selected, skipping init (run `wenlan models install` to enable)"
            ),
            Some(model) if !wenlan_core::on_device_models::is_cached(model) => tracing::info!(
                "[on-device] model {} not cached, skipping init (use settings to download)",
                model.id
            ),
            Some(model) => {
                let shared_for_llm = shared.clone();
                let reservation = {
                    let state = shared.read().await;
                    state
                        .startup_model_load_reserved
                        .store(true, std::sync::atomic::Ordering::Release);
                    state.startup_model_load_reserved.clone()
                };
                let mut load_shutdown = {
                    let state = shared.read().await;
                    state.shutdown.subscribe()
                };
                let working_set_bytes = on_device_model_working_set_bytes(model);
                tokio::spawn(async move {
                    let _reservation = StartupModelLoadReservation(reservation);
                    if !scheduler::wait_for_startup_model_admission(
                        working_set_bytes,
                        &mut load_shutdown,
                    )
                    .await
                    {
                        tracing::info!(
                            "[on-device] shutdown requested before startup load admission"
                        );
                        return;
                    }
                    let model_id = model.id;
                    let result = tokio::task::spawn_blocking(move || {
                        let provider =
                            wenlan_core::llm_provider::OnDeviceProvider::new_with_model(Some(
                                model_id,
                            ))?;
                        let arc: Arc<dyn wenlan_core::llm_provider::LlmProvider> =
                            Arc::new(provider);
                        Ok::<_, wenlan_core::error::WenlanError>((
                            arc,
                            model_id.to_string(),
                        ))
                    })
                    .await;

                    match result {
                        Ok(Ok((provider, model_id))) => {
                            let mut state = shared_for_llm.write().await;
                            state.llm = Some(provider);
                            state.loaded_on_device_model = Some(model_id.clone());
                            tracing::info!("[on-device] model {} loaded and available", model_id);
                        }
                        Ok(Err(e)) => tracing::error!("[on-device] init failed: {}", e),
                        Err(e) => tracing::error!("[on-device] init task panicked: {}", e),
                    }
                });
            }
        }
    }

    // Register the LLM-readiness hook so that the `intelligence-ready`
    // onboarding milestone fires the first time any LLM provider successfully
    // serves traffic. `mark_llm_ready` is a one-shot per process, so this hook
    // runs at most once regardless of which provider fires first.
    //
    // The on-device `llm-provider-worker` (`crates/wenlan-core/src/llm_provider.rs:142`)
    // runs on a `std::thread`, not a Tokio task — GPU inference is blocking
    // and would starve the async runtime. When it calls `mark_llm_ready()`
    // from that thread, our hook fires synchronously on a thread with no
    // Tokio reactor in thread-local context. Bare `tokio::spawn(...)` would
    // then panic: "there is no reactor running, must be called from the
    // context of a Tokio 1.x runtime" — exactly what killed the worker on
    // 2026-04-16. Capture a `Handle` here (we are inside `#[tokio::main]`)
    // and use `handle.spawn(...)` from the closure instead.
    if optional_runtime_workers_allowed(repair_recovery_pending) {
        let db_for_ready = db_arc.clone();
        let maintenance_for_ready = {
            let state = shared.read().await;
            state.maintenance_coordinator.clone()
        };
        let emitter_for_ready: Arc<dyn wenlan_core::events::EventEmitter> =
            Arc::new(wenlan_core::events::NoopEmitter);
        let handle = tokio::runtime::Handle::current();
        let hook: wenlan_core::llm_provider::ReadinessHook = Arc::new(move || {
            let db = db_for_ready.clone();
            let emitter = emitter_for_ready.clone();
            let maintenance = maintenance_for_ready.clone();
            handle.spawn(async move {
                let _maintenance_guard = maintenance.begin_background().await;
                let ev = wenlan_core::onboarding::MilestoneEvaluator::new(&db, emitter);
                if let Err(e) = ev.check_after_llm_ready().await {
                    tracing::warn!(?e, "onboarding: check_after_llm_ready failed");
                }
            });
        });
        let _ = wenlan_core::llm_provider::LLM_READINESS_HOOK.set(hook);
    }

    let shutdown = { shared.read().await.shutdown.clone() };
    let signal_shutdown = shutdown.clone();
    tokio::spawn(async move {
        termination_signals.wait().await;
        tracing::info!("termination signal received");
        signal_shutdown.request();
    });

    // Spawn the event-driven steep scheduler.
    // See docs/superpowers/specs/2026-04-12-event-driven-steep-triggers-design.md
    let scheduler_task = if optional_runtime_workers_allowed(repair_recovery_pending) {
        let write_signal = {
            let s = shared.read().await;
            s.write_signal.clone()
        };
        scheduler::spawn_scheduler(shared.clone(), write_signal, shutdown.subscribe())
    } else {
        tokio::spawn(async {})
    };

    if repair_recovery_pending {
        tracing::warn!("repair-only startup: optional runtime workers are disabled");
    } else if wenlan_core::db::entity_sweep_enabled() {
        tracing::info!(
            "Ambient entity enrichment is ON: the shared quiet/cooldown-gated scheduler \
             backfills knowledge-graph links over existing memories. Set \
             WENLAN_ENABLE_ENTITY_SWEEP=0 to disable."
        );
    } else {
        tracing::info!("Ambient entity enrichment is OFF (WENLAN_ENABLE_ENTITY_SWEEP).");
    }

    // Build router
    let app = if repair_recovery_pending {
        router::build_repair_router(shared)
    } else {
        router::build_router_with_shutdown(shared, shutdown.clone())
    };

    // Advertise the bound port before accepting requests.
    // `addr` may be `127.0.0.1:0`; `local_addr()` gives the real ephemeral port.
    let local_addr = listener.local_addr()?;
    tracing::info!("Listening on http://{}", local_addr);

    // Eval harness reads this stdout line to discover the bound port even when
    // WENLAN_BIND_ADDR=127.0.0.1:0. Format MUST stay stable — see
    // crates/wenlan-core/src/eval/http_harness.rs in the P2 plan.
    println!("WENLAN_LISTENING_ON={}", local_addr);
    let _ = std::io::stdout().flush();

    // Alternate signal: write the port to a file if WENLAN_PORT_FILE is set.
    // Eval harness uses this when stdout is captured by tracing-appender.
    if let Ok(port_file) = std::env::var("WENLAN_PORT_FILE") {
        if let Err(e) = std::fs::write(&port_file, local_addr.port().to_string()) {
            tracing::error!("failed to write WENLAN_PORT_FILE={}: {}", port_file, e);
            return Err(anyhow::anyhow!("WENLAN_PORT_FILE write failed: {}", e));
        }
    }

    // Serve until HTTP shutdown or an OS termination signal. Axum stops
    // accepting new connections and drains in-flight requests; the scheduler
    // finishes its currently awaited item without starting another. A hard
    // deadline remains necessary because Tokio cannot cancel arbitrary
    // spawn_blocking work during runtime drop.
    let server = axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(lifecycle::wait_for_shutdown(shutdown.subscribe()))
        .into_future();
    tokio::pin!(server);
    let server_completed = tokio::select! {
        result = &mut server => Some(result),
        _ = lifecycle::wait_for_shutdown(shutdown.subscribe()) => None,
    };

    if let Some(result) = server_completed {
        shutdown.request();
        match tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, scheduler_task).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!("scheduler join failed: {error}"),
            Err(_) => tracing::warn!("scheduler did not stop within the drain deadline"),
        }
        match result {
            Ok(()) => {
                tracing::info!("HTTP server stopped; daemon lifecycle complete");
                exit_daemon(0);
            }
            Err(error) => {
                tracing::error!("HTTP server failed: {error}");
                exit_daemon(1);
            }
        }
    }

    tracing::info!(
        "shutdown requested — draining for at most {}ms",
        SHUTDOWN_DRAIN_TIMEOUT.as_millis()
    );
    let drained = tokio::time::timeout(SHUTDOWN_DRAIN_TIMEOUT, async {
        let server_result = (&mut server).await;
        let scheduler_result = scheduler_task.await;
        (server_result, scheduler_result)
    })
    .await;
    match drained {
        Ok((server_result, scheduler_result)) => {
            if let Err(error) = scheduler_result {
                tracing::warn!("scheduler join failed during shutdown: {error}");
            }
            server_result?;
            // `#[tokio::main]` waits indefinitely for already-started
            // `spawn_blocking` work while dropping the runtime. The HTTP
            // server and scheduler above are the daemon-owned drain boundary;
            // exit explicitly once both have stopped so shutdown remains
            // bounded even if an inference worker is still blocked.
            tracing::info!("graceful shutdown complete");
            exit_daemon(0);
        }
        Err(_) => {
            tracing::warn!(
                "graceful shutdown exceeded {}ms — forcing clean exit",
                SHUTDOWN_DRAIN_TIMEOUT.as_millis()
            );
            exit_daemon(0);
        }
    }
}

/// Batch processor invoked by the ingest coalescer per flush. Runs the
/// full per-request ingest pipeline — quality gate evaluate (batched so
/// one FastEmbed call covers every survivor's novelty check) → partition
/// admitted vs rejected → upsert survivors in a single transaction →
/// emit per-doc outcomes in input order.
///
/// Fail-open policy on gate infrastructure failure: if the batched gate
/// evaluator itself returns an error (DB unreachable, embedding panicked
/// inside FastEmbed, etc.), every doc is admitted rather than rejected —
/// matches `QualityGate::evaluate`'s per-doc behavior, which also fails
/// open rather than wedging stores behind the gate.
async fn ingest_batch_process(
    db: std::sync::Arc<wenlan_core::db::MemoryDB>,
    gate: wenlan_core::quality_gate::QualityGate,
    items: Vec<(wenlan_core::sources::RawDocument, usize)>,
) -> Vec<ingest_batcher::StoreOutcome> {
    use ingest_batcher::StoreOutcome;
    use wenlan_core::quality_gate::{GateResult, GateScores};

    if items.is_empty() {
        return vec![];
    }

    // Batch gate evaluate. One FastEmbed call, N vector queries, one pass.
    let contents: Vec<&str> = items.iter().map(|(d, _)| d.content.as_str()).collect();
    let gate_results = match gate.evaluate_batch(&contents, &db).await {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("[ingest_batch_process] gate batch evaluate failed (fail closed), rejecting all: {e}");
            contents
                .iter()
                .map(|c| {
                    (
                        GateResult {
                            admitted: false,
                            reason: Some(
                                wenlan_core::quality_gate::RejectionReason::EmbeddingUnavailable(
                                    e.to_string(),
                                ),
                            ),
                            scores: GateScores {
                                content_type_pass: true,
                                novelty_score: None,
                                word_count: c.split_whitespace().count(),
                                pattern_matched: Some("embedding_unavailable".to_string()),
                                latency_ms: 0,
                            },
                        },
                        None,
                    )
                })
                .collect()
        }
    };

    let n = items.len();
    let mut outcomes: Vec<Option<StoreOutcome>> = (0..n).map(|_| None).collect();
    // (original_position, doc, chunks_predicted) for every admitted doc.
    let mut survivors: Vec<(usize, wenlan_core::sources::RawDocument, usize)> = Vec::new();

    for (i, ((doc, chunks), (gate_result, similar_id))) in
        items.into_iter().zip(gate_results).enumerate()
    {
        if gate_result.admitted {
            survivors.push((i, doc, chunks));
        } else {
            let (reason_str, detail_str) = gate_result
                .reason
                .as_ref()
                .map(|r| (r.as_str().to_string(), r.detail()))
                .unwrap_or_else(|| ("unknown".to_string(), "rejected".to_string()));
            outcomes[i] = Some(StoreOutcome::GateRejected {
                reason: reason_str,
                detail: detail_str,
                similar_to: similar_id,
            });
        }
    }

    if !survivors.is_empty() {
        let docs: Vec<wenlan_core::sources::RawDocument> =
            survivors.iter().map(|(_, d, _)| d.clone()).collect();
        match db.upsert_documents(docs).await {
            Ok(_total) => {
                for (pos, _, chunks) in &survivors {
                    outcomes[*pos] = Some(StoreOutcome::Stored {
                        chunks_created: *chunks,
                    });
                }
            }
            Err(e) => {
                let msg = e.to_string();
                for (pos, _, _) in &survivors {
                    outcomes[*pos] = Some(StoreOutcome::UpsertFailed(msg.clone()));
                }
            }
        }
    }

    // Any `None` slot means the item was neither admitted nor rejected —
    // shouldn't happen, but backfill defensively.
    outcomes
        .into_iter()
        .map(|o| o.unwrap_or(StoreOutcome::UpsertFailed("missing outcome slot".into())))
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Propagate flags through env vars so both wenlan-server's own path logic
    // and wenlan-core's config loader (`wenlan_core::config::config_path`) see
    // the same values without plumbing a parameter through every call site.
    if let Some(ref dir) = cli.data_dir {
        std::env::set_var("WENLAN_DATA_DIR", dir);
    }
    if let Some(port) = cli.port {
        std::env::set_var("WENLAN_PORT", port.to_string());
    }

    // Resolving the path is read-only. Before rotating tracing is available,
    // a bounded bootstrap file keeps launchd failures and panics observable
    // even though the plist intentionally redirects stdout/stderr to /dev/null.
    let wenlan_root = resolve_wenlan_root();
    install_bootstrap_panic_hook(wenlan_root.clone());

    let result = async {
        let startup_repair_claim = StartupRepairClaim::try_new(
            cli.repair_manifest_id.clone(),
            cli.repair_manifest_digest.clone(),
        )?;

        if cli.command.is_some() && startup_repair_claim.is_some() {
            anyhow::bail!("startup repair claim is only valid when running the daemon");
        }

        match cli.command {
            Some(Command::BackfillStalePages { dry_run }) => cmd_backfill::run(dry_run).await,
            None => run_daemon(startup_repair_claim).await,
        }
    }
    .await;

    if let Err(error) = &result {
        report_bootstrap_error(
            &wenlan_root,
            &format!("wenlan-server terminated with an error: {error:#}"),
        );
    }
    result
}
