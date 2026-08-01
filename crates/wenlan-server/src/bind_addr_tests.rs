use super::*;
use std::sync::{Mutex, OnceLock};

#[cfg(unix)]
unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
    fn _exit(status: std::ffi::c_int) -> !;
}

static TEST_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static TEST_SUBPROCESS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn env_lock() -> &'static Mutex<()> {
    TEST_ENV_LOCK.get_or_init(|| Mutex::new(()))
}

fn subprocess_lock() -> &'static Mutex<()> {
    TEST_SUBPROCESS_LOCK.get_or_init(|| Mutex::new(()))
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

    let _guard = subprocess_lock().lock().unwrap();
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
fn brief_status_root_stays_inside_isolated_data_dir() {
    let _guard = env_lock().lock().unwrap();
    let root = tempfile::tempdir().unwrap();
    temp_env::with_var("WENLAN_DATA_DIR", Some(root.path()), || {
        assert_eq!(
            resolve_brief_status_root(root.path()),
            root.path().join("sessions/_status")
        );
    });
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
    assert!(Cli::try_parse_from(["wenlan-server", "--repair-manifest-id", manifest_id,]).is_err());
    assert!(
        Cli::try_parse_from(["wenlan-server", "--repair-manifest-digest", &"a".repeat(64),])
            .is_err()
    );
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
    let wrong = StartupRepairClaim::try_new(Some(manifest_id.to_string()), Some("a".repeat(64)))
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
    let claim = StartupRepairClaim::try_new(Some(manifest_id.to_string()), Some(digest.clone()))
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
    // A concurrently spawned test child inherits open Unix file descriptors,
    // including this lock. Keep the drop-and-reacquire proof isolated from
    // every test in this module that launches the current test executable.
    let _guard = subprocess_lock().lock().unwrap();
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("wenlan");
    let first = DaemonDataLock::acquire(&root, false).unwrap();

    assert!(
        !root.exists(),
        "normal lock acquisition must not create the data root"
    );
    assert!(DaemonDataLock::acquire(&root, false).is_err());
    drop(first);
    DaemonDataLock::acquire(&root, false).expect("dropping the owner releases the data-root lock");
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
    let ready = std::path::PathBuf::from(std::env::var_os("WENLAN_DATA_LOCK_CHILD_READY").unwrap());
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
    let _guard = subprocess_lock().lock().unwrap();
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
