// SPDX-License-Identifier: Apache-2.0
//! Ignored, target-Mac process proof for the production ambient scheduler.
//!
//! The test intentionally keeps the real 60-second startup delay, 30-second
//! poll, resource admission, and 2-minute minimum thermal recovery. Run it only
//! through `scripts/profile-ambient-rb01.sh daemon`, which owns host admission,
//! cooldown, provenance, and artifact persistence.

#![cfg(target_os = "macos")]

mod ambient_live_daemon {
    use std::collections::{BTreeMap, VecDeque};
    use std::io::{BufRead, BufReader, Read, Write};
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::sync::mpsc::{self, Receiver};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    const OPT_IN_ENV: &str = "WENLAN_RB01_DAEMON_PROFILE";
    const MODEL_ID: &str = "qwen3-4b";
    const ON_DEVICE_INFERENCE_HEADROOM_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    const MIN_PRODUCTION_COOLDOWN: Duration = Duration::from_secs(120);
    const SAFETY_SAMPLE_INTERVAL: Duration = Duration::from_secs(30);
    const RESIDENCY_CHECK_INTERVAL: Duration = Duration::from_secs(60);
    const MIN_MID_COOLDOWN_RESIDENCY_CHECKS: usize = 1;
    const MAX_RECENT_RELEVANT_LINES: usize = 8;
    const MAX_RELEVANT_LINE_CHARS: usize = 512;

    struct ChildGuard(Child);

    impl Drop for ChildGuard {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[derive(Debug)]
    struct SelectedTurn {
        job: String,
        started: Instant,
        completed: Instant,
        llm_calls: usize,
        elapsed_ms: u64,
        next_eligible_ms: u64,
    }

    #[derive(Debug, Default, serde::Serialize)]
    struct AdmissionBlockSummary {
        count: u64,
        last_cpu_percent: Option<f64>,
        last_available_memory_mb: Option<u64>,
    }

    struct LiveObserver {
        active: Option<(String, Instant)>,
        next_start_allowed_at: Option<Instant>,
        model_loads: usize,
        selected_turns: Vec<SelectedTurn>,
        peak_rss_bytes: u64,
        last_rss_sample: Option<Instant>,
        memory_floor_mb: u64,
        completed_turns: usize,
        admission_blocks: BTreeMap<String, AdmissionBlockSummary>,
        recent_relevant_lines: VecDeque<String>,
    }

    struct ObservedLine {
        line: String,
        observed_at: Instant,
    }

    #[derive(Debug, Clone, Copy)]
    struct SafetySample {
        thermal_state: u8,
        total_memory_bytes: u64,
        available_memory_bytes: u64,
    }

    struct SafetyWatchdog {
        started_at: Instant,
        next_sample_at: Instant,
        samples: Vec<(u64, SafetySample)>,
        thermal_helper: std::path::PathBuf,
    }

    impl SafetyWatchdog {
        fn new(now: Instant, thermal_helper: std::path::PathBuf) -> Self {
            Self {
                started_at: now,
                next_sample_at: now,
                samples: Vec::new(),
                thermal_helper,
            }
        }
    }

    fn emit_recovery_report(observer: &LiveObserver, reason: &str) -> u64 {
        let report_elapsed_ms = observer
            .selected_turns
            .iter()
            .map(|turn| turn.elapsed_ms)
            .max()
            .unwrap_or(0);
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "rb01_recovery_known": {
                    "reason": reason,
                    "selected_turns": observer.selected_turns.len(),
                    "admission_blocks": &observer.admission_blocks
                },
                "report_elapsed_ms": report_elapsed_ms
            }))
            .expect("serialize recovery-known marker")
        );
        std::io::stdout()
            .flush()
            .expect("flush recovery-known marker");
        report_elapsed_ms
    }

    impl LiveObserver {
        fn with_memory_floor_mb(memory_floor_mb: u64) -> Self {
            Self {
                active: None,
                next_start_allowed_at: None,
                model_loads: 0,
                selected_turns: Vec::new(),
                peak_rss_bytes: 0,
                last_rss_sample: None,
                memory_floor_mb,
                completed_turns: 0,
                admission_blocks: BTreeMap::new(),
                recent_relevant_lines: VecDeque::with_capacity(MAX_RECENT_RELEVANT_LINES),
            }
        }

        fn observe_line(&mut self, line: &str, observed_at: Instant) {
            if line.contains("[scheduler]")
                || line.contains("[on-device] model qwen3-4b loaded and available")
            {
                if self.recent_relevant_lines.len() == MAX_RECENT_RELEVANT_LINES {
                    self.recent_relevant_lines.pop_front();
                }
                self.recent_relevant_lines
                    .push_back(line.chars().take(MAX_RELEVANT_LINE_CHARS).collect());
            }

            if line.contains("[on-device] model qwen3-4b loaded and available") {
                self.model_loads += 1;
            }

            if let Some(rest) = line.split("[scheduler] heavy work deferred reason=").nth(1) {
                let reason = rest
                    .split_ascii_whitespace()
                    .next()
                    .expect("deferred scheduler log carries a reason");
                let summary = self.admission_blocks.entry(reason.to_owned()).or_default();
                summary.count += 1;
                summary.last_cpu_percent = parse_optional_f64_field(rest, "cpu_percent=");
                summary.last_available_memory_mb =
                    parse_optional_u64_field(rest, "available_memory_mb=");
                return;
            }

            if let Some(rest) = line.split("[scheduler] automatic trigger=").nth(1) {
                let selected = parse_bool_field(rest, "selected=");
                let llm_calls = parse_u64_field(rest, "llm_calls=");
                let panicked = parse_bool_field(rest, "panicked=");
                assert!(
                    !selected && llm_calls == 0 && !panicked,
                    "isolated daemon fixture unexpectedly ran heavy automatic work: {rest}"
                );
                return;
            }

            if let Some(rest) = line.split("[scheduler] ambient turn started job=").nth(1) {
                let job = rest
                    .split_ascii_whitespace()
                    .next()
                    .expect("ambient start log carries a job")
                    .to_string();
                let cpu_percent = parse_some_f64_field(rest, "cpu_percent=");
                let available_memory_mb = parse_some_u64_field(rest, "available_memory_mb=");
                assert!(
                    cpu_percent <= 20.0,
                    "ambient start escaped CPU admission: {cpu_percent}%"
                );
                assert!(
                    available_memory_mb >= self.memory_floor_mb,
                    "ambient start escaped memory admission: {available_memory_mb} MiB < {} MiB",
                    self.memory_floor_mb
                );
                if let Some(deadline) = self.next_start_allowed_at {
                    assert!(
                        observed_at >= deadline,
                        "ambient start occurred {:?} before the published deadline",
                        deadline.saturating_duration_since(observed_at)
                    );
                }
                assert!(
                    self.active.is_none(),
                    "overlapping ambient turns: active={:?}, next={job}",
                    self.active
                );
                self.active = Some((job, observed_at));
                return;
            }

            let Some(rest) = line.split("[scheduler] ambient job=").nth(1) else {
                return;
            };
            if !rest.contains(" selected=") {
                return;
            }

            let job = rest
                .split_ascii_whitespace()
                .next()
                .expect("ambient completion log carries a job")
                .to_string();
            let (started_job, started) = self
                .active
                .take()
                .expect("ambient completion must follow a start event");
            assert_eq!(
                started_job, job,
                "ambient completion must match its active start"
            );

            let selected = parse_bool_field(rest, "selected=");
            let llm_calls = parse_u64_field(rest, "llm_calls=") as usize;
            let panicked = parse_bool_field(rest, "panicked=");
            let elapsed_ms = parse_u64_field(rest, "elapsed_ms=");
            let next_eligible_ms = parse_u64_field(rest, "next_eligible_ms=");
            assert!(llm_calls <= 1, "ambient turn forwarded {llm_calls} calls");
            self.next_start_allowed_at =
                Some(observed_at + Duration::from_millis(next_eligible_ms));
            self.completed_turns += 1;

            if selected {
                assert!(!panicked, "selected ambient turn panicked: {job}");
                assert_eq!(
                    llm_calls, 1,
                    "the persistent-provider proof requires one real inference in each selected turn"
                );
                assert!(
                    next_eligible_ms >= MIN_PRODUCTION_COOLDOWN.as_millis() as u64,
                    "selected turn did not publish the production cooldown: {next_eligible_ms}ms"
                );
                self.selected_turns.push(SelectedTurn {
                    job,
                    started,
                    completed: observed_at,
                    llm_calls,
                    elapsed_ms,
                    next_eligible_ms,
                });
            }
        }

        fn diagnostic_summary(&self) -> String {
            let selected_jobs = self
                .selected_turns
                .iter()
                .map(|turn| turn.job.as_str())
                .collect::<Vec<_>>();
            let admission_blocks = self
                .admission_blocks
                .iter()
                .map(|(reason, summary)| {
                    format!(
                        "{reason}: count={} last_cpu_percent={:?} last_available_memory_mb={:?}",
                        summary.count, summary.last_cpu_percent, summary.last_available_memory_mb,
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "completed_turns={} selected_jobs={selected_jobs:?} active={:?} model_loads={} admission_blocks=[{admission_blocks}] recent_relevant_lines={:?}",
                self.completed_turns,
                self.active,
                self.model_loads,
                self.recent_relevant_lines
            )
        }

        fn sample_rss_if_due(&mut self, child_pid: u32, now: Instant) {
            if self
                .last_rss_sample
                .is_some_and(|last| now.saturating_duration_since(last) < Duration::from_secs(5))
            {
                return;
            }
            self.last_rss_sample = Some(now);
            if let Some(rss) = process_rss_bytes(child_pid) {
                self.peak_rss_bytes = self.peak_rss_bytes.max(rss);
            }
        }
    }

    fn parse_bool_field(line: &str, prefix: &str) -> bool {
        match field(line, prefix) {
            "true" => true,
            "false" => false,
            value => panic!("invalid {prefix} value in ambient log: {value}"),
        }
    }

    fn parse_u64_field(line: &str, prefix: &str) -> u64 {
        field(line, prefix)
            .parse()
            .unwrap_or_else(|error| panic!("invalid {prefix} field: {error}"))
    }

    fn parse_some_f64_field(line: &str, prefix: &str) -> f64 {
        parse_some_field(line, prefix)
            .parse()
            .unwrap_or_else(|error| panic!("invalid {prefix} field: {error}"))
    }

    fn parse_some_u64_field(line: &str, prefix: &str) -> u64 {
        parse_some_field(line, prefix)
            .parse()
            .unwrap_or_else(|error| panic!("invalid {prefix} field: {error}"))
    }

    fn parse_optional_f64_field(line: &str, prefix: &str) -> Option<f64> {
        parse_optional_field(line, prefix).map(|value| {
            value
                .parse()
                .unwrap_or_else(|error| panic!("invalid {prefix} field: {error}"))
        })
    }

    fn parse_optional_u64_field(line: &str, prefix: &str) -> Option<u64> {
        parse_optional_field(line, prefix).map(|value| {
            value
                .parse()
                .unwrap_or_else(|error| panic!("invalid {prefix} field: {error}"))
        })
    }

    fn parse_optional_field<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
        let value = field(line, prefix);
        match value {
            "None" => None,
            _ => Some(
                value
                    .strip_prefix("Some(")
                    .and_then(|value| value.strip_suffix(')'))
                    .unwrap_or_else(|| {
                        panic!("invalid optional {prefix} telemetry in ambient log: {line}")
                    }),
            ),
        }
    }

    fn parse_some_field<'a>(line: &'a str, prefix: &str) -> &'a str {
        field(line, prefix)
            .strip_prefix("Some(")
            .and_then(|value| value.strip_suffix(')'))
            .unwrap_or_else(|| panic!("missing admitted {prefix} telemetry in ambient log: {line}"))
    }

    fn field<'a>(line: &'a str, prefix: &str) -> &'a str {
        line.split_ascii_whitespace()
            .find_map(|token| token.strip_prefix(prefix))
            .unwrap_or_else(|| panic!("missing {prefix} in ambient log: {line}"))
    }

    fn binary_path() -> std::path::PathBuf {
        let cargo_built_binary = env!("CARGO_BIN_EXE_wenlan-server");
        let path = std::path::PathBuf::from(
            std::env::var_os("WENLAN_RB01_DAEMON_BINARY").unwrap_or_else(|| {
                panic!(
                    "live daemon profile requires the frozen daemon binary path; Cargo produced {cargo_built_binary}"
                )
            }),
        );
        let expected = std::env::var("WENLAN_RB01_DAEMON_SHA256")
            .expect("live daemon profile requires the frozen daemon binary hash");
        let output = Command::new("/usr/bin/shasum")
            .args(["-a", "256"])
            .arg(&path)
            .output()
            .expect("hash frozen daemon binary before spawn");
        assert!(output.status.success(), "frozen daemon hash command failed");
        let actual = String::from_utf8(output.stdout)
            .expect("frozen daemon hash UTF-8")
            .split_ascii_whitespace()
            .next()
            .expect("frozen daemon hash value")
            .to_owned();
        assert_eq!(
            actual, expected,
            "frozen daemon bytes changed before the harness spawned them"
        );
        path
    }

    fn thermal_helper_path() -> std::path::PathBuf {
        let path = std::path::PathBuf::from(
            std::env::var_os("WENLAN_RB01_THERMAL_HELPER")
                .expect("live daemon profile requires the frozen thermal helper path"),
        );
        let expected = std::env::var("WENLAN_RB01_THERMAL_HELPER_SHA256")
            .expect("live daemon profile requires the frozen thermal helper hash");
        let output = Command::new("/usr/bin/shasum")
            .args(["-a", "256"])
            .arg(&path)
            .output()
            .expect("hash frozen thermal helper before use");
        assert!(
            output.status.success(),
            "frozen thermal helper hash command failed"
        );
        let actual = String::from_utf8(output.stdout)
            .expect("frozen thermal helper hash UTF-8")
            .split_ascii_whitespace()
            .next()
            .expect("frozen thermal helper hash value")
            .to_owned();
        assert_eq!(
            actual, expected,
            "frozen thermal helper bytes changed before the harness used them"
        );
        path
    }

    fn spawn_log_reader<R>(stream: R, sender: mpsc::Sender<ObservedLine>) -> JoinHandle<()>
    where
        R: Read + Send + 'static,
    {
        std::thread::spawn(move || {
            for line in BufReader::new(stream).lines() {
                let line = line.expect("read daemon log line");
                eprintln!("{line}");
                if sender
                    .send(ObservedLine {
                        line,
                        observed_at: Instant::now(),
                    })
                    .is_err()
                {
                    return;
                }
            }
        })
    }

    fn spawn_observed_child(
        command: &mut Command,
    ) -> (Child, Receiver<ObservedLine>, Vec<JoinHandle<()>>) {
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn observed child");
        let stdout = child.stdout.take().expect("captured child stdout");
        let stderr = child.stderr.take().expect("captured child stderr");
        let (sender, receiver) = mpsc::channel();
        let readers = vec![
            spawn_log_reader(stdout, sender.clone()),
            spawn_log_reader(stderr, sender),
        ];
        (child, receiver, readers)
    }

    fn join_log_readers_and_drain(
        readers: Vec<JoinHandle<()>>,
        receiver: &Receiver<ObservedLine>,
        observer: &mut LiveObserver,
    ) {
        for reader in readers {
            reader.join().expect("observed child log reader joins");
        }
        drain_logs(receiver, observer);
    }

    fn process_rss_bytes(pid: u32) -> Option<u64> {
        let output = Command::new("/bin/ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let kib = String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        Some(kib.saturating_mul(1024))
    }

    fn system_total_memory_bytes() -> u64 {
        let output = Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .expect("read total system memory");
        assert!(output.status.success(), "sysctl hw.memsize failed");
        String::from_utf8(output.stdout)
            .expect("sysctl hw.memsize UTF-8")
            .trim()
            .parse()
            .expect("sysctl hw.memsize integer")
    }

    fn safety_violation(
        thermal_state: u8,
        total_memory_bytes: u64,
        available_memory_bytes: u64,
    ) -> Option<&'static str> {
        if thermal_state != 0 {
            return Some("thermal_not_nominal");
        }
        if total_memory_bytes == 0 || available_memory_bytes == 0 {
            return Some("memory_telemetry_unavailable");
        }
        let memory_floor =
            (total_memory_bytes.saturating_mul(15) / 100).max(2 * 1024 * 1024 * 1024);
        if available_memory_bytes < memory_floor {
            return Some("memory_pressure");
        }
        None
    }

    fn sample_safety(thermal_helper: &std::path::Path) -> Result<SafetySample, String> {
        let thermal = Command::new(thermal_helper)
            .output()
            .map_err(|error| format!("spawn thermal probe: {error}"))?;
        if !thermal.status.success() {
            return Err(format!("thermal probe exited {}", thermal.status));
        }
        let thermal_state = String::from_utf8(thermal.stdout)
            .map_err(|error| format!("thermal probe UTF-8: {error}"))?
            .trim()
            .parse::<u8>()
            .map_err(|error| format!("thermal probe value: {error}"))?;

        let refreshes = sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram());
        let mut system = sysinfo::System::new_with_specifics(refreshes);
        system.refresh_memory();
        Ok(SafetySample {
            thermal_state,
            total_memory_bytes: system.total_memory(),
            available_memory_bytes: system.available_memory(),
        })
    }

    async fn fail_safety_watchdog(
        child: &mut ChildGuard,
        client: &reqwest::Client,
        port: u16,
        reason: &str,
    ) -> ! {
        eprintln!("[rb01-watchdog] refusing continued live proof: {reason}");
        let _ = client
            .post(format!("http://127.0.0.1:{port}/api/shutdown"))
            .send()
            .await;
        let _ = wait_for_exit(child, Duration::from_secs(5)).await;
        panic!("RB-01 safety watchdog stopped the daemon: {reason}");
    }

    async fn enforce_safety_watchdog(
        child: &mut ChildGuard,
        client: &reqwest::Client,
        port: u16,
        thermal_helper: std::path::PathBuf,
    ) -> SafetySample {
        let sample = match tokio::task::spawn_blocking(move || sample_safety(&thermal_helper)).await
        {
            Ok(Ok(sample)) => sample,
            Ok(Err(error)) => {
                fail_safety_watchdog(child, client, port, &error).await;
            }
            Err(error) => {
                fail_safety_watchdog(
                    child,
                    client,
                    port,
                    &format!("safety probe task failed: {error}"),
                )
                .await;
            }
        };
        if let Some(reason) = safety_violation(
            sample.thermal_state,
            sample.total_memory_bytes,
            sample.available_memory_bytes,
        ) {
            fail_safety_watchdog(child, client, port, reason).await;
        }
        sample
    }

    async fn sample_safety_if_due(
        child: &mut ChildGuard,
        client: &reqwest::Client,
        port: u16,
        watchdog: &mut SafetyWatchdog,
    ) {
        if Instant::now() < watchdog.next_sample_at {
            return;
        }
        let sample =
            enforce_safety_watchdog(child, client, port, watchdog.thermal_helper.clone()).await;
        let observed_at = Instant::now();
        watchdog.samples.push((
            observed_at
                .saturating_duration_since(watchdog.started_at)
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
            sample,
        ));
        while watchdog.next_sample_at <= observed_at {
            watchdog.next_sample_at += SAFETY_SAMPLE_INTERVAL;
        }
    }

    fn drain_logs(receiver: &Receiver<ObservedLine>, observer: &mut LiveObserver) {
        while let Ok(observed) = receiver.try_recv() {
            observer.observe_line(&observed.line, observed.observed_at);
        }
    }

    fn assert_child_running(child: &mut ChildGuard, context: &str) {
        if let Some(status) = child.0.try_wait().expect("poll daemon child") {
            panic!("daemon exited {context}: {status}");
        }
    }

    async fn wait_for_port(child: &mut ChildGuard, port_file: &std::path::Path) -> u16 {
        tokio::time::timeout(Duration::from_secs(60), async {
            loop {
                assert_child_running(child, "before port discovery");
                if let Ok(contents) = std::fs::read_to_string(port_file) {
                    break contents.trim().parse::<u16>().expect("valid daemon port");
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .expect("daemon port discovery timed out")
    }

    async fn setup_status(client: &reqwest::Client, port: u16) -> serde_json::Value {
        client
            .get(format!("http://127.0.0.1:{port}/api/setup/status"))
            .send()
            .await
            .expect("request setup status")
            .error_for_status()
            .expect("setup status success")
            .json()
            .await
            .expect("setup status JSON")
    }

    async fn assert_model_loaded(client: &reqwest::Client, port: u16) {
        let status = setup_status(client, port).await;
        assert_eq!(
            status["local_model_loaded"], MODEL_ID,
            "the same on-device provider must remain resident between turns"
        );
    }

    async fn wait_for_model_load(
        child: &mut ChildGuard,
        receiver: &Receiver<ObservedLine>,
        observer: &mut LiveObserver,
        client: &reqwest::Client,
        port: u16,
        watchdog: &mut SafetyWatchdog,
    ) {
        let deadline = Instant::now() + Duration::from_secs(8 * 60);
        loop {
            drain_logs(receiver, observer);
            let now = Instant::now();
            observer.sample_rss_if_due(child.0.id(), now);
            assert_child_running(child, "while waiting for model load");
            sample_safety_if_due(child, client, port, watchdog).await;
            let status = setup_status(client, port).await;
            if status["local_model_loaded"] == MODEL_ID {
                return;
            }
            assert!(
                now < deadline,
                "on-device model did not reach loaded state within 8 minutes"
            );
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    async fn wait_for_exit(child: &mut ChildGuard, bound: Duration) -> ExitStatus {
        tokio::time::timeout(bound, async {
            loop {
                if let Some(status) = child.0.try_wait().expect("poll daemon exit") {
                    break status;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("daemon did not exit within the cooperative shutdown bound")
    }

    async fn current_ok_receipts(db_path: &std::path::Path, source_id: &str) -> Vec<(String, i64)> {
        let database = libsql::Builder::new_local(
            db_path
                .to_str()
                .expect("temp database path must be valid UTF-8"),
        )
        .build()
        .await
        .expect("open stopped daemon database");
        let connection = database.connect().expect("connect stopped daemon database");
        let mut rows = connection
            .query(
                "SELECT step_name, input_version
                   FROM enrichment_steps
                  WHERE source_id = ?1
                    AND status = 'ok'
                    AND input_version = (
                        SELECT version
                          FROM memories
                         WHERE source_id = ?1 AND chunk_index = 0
                    )
                  ORDER BY updated_at, step_name",
                libsql::params![source_id],
            )
            .await
            .expect("query durable enrichment receipts");
        let mut receipts = Vec::new();
        while let Some(row) = rows.next().await.expect("read durable enrichment receipt") {
            receipts.push((
                row.get::<String>(0).expect("receipt step name"),
                row.get::<i64>(1).expect("receipt input version"),
            ));
        }
        receipts
    }

    #[test]
    fn observer_rejects_start_before_the_published_deadline() {
        let base = Instant::now();
        let mut observer = LiveObserver::with_memory_floor_mb(2_500);
        observer.observe_line(
            "[scheduler] ambient turn started job=Classification cpu_percent=Some(5.0) available_memory_mb=Some(8000)",
            base,
        );
        observer.observe_line(
            "[scheduler] ambient job=Classification selected=true llm_calls=1 panicked=false elapsed_ms=100 next_eligible_ms=120000",
            base + Duration::from_millis(100),
        );

        let early = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            observer.observe_line(
                "[scheduler] ambient turn started job=StructuredExtract cpu_percent=Some(5.0) available_memory_mb=Some(8000)",
                base + Duration::from_secs(110),
            );
        }));
        assert!(
            early.is_err(),
            "110-second restart must not satisfy 120 seconds"
        );
    }

    #[test]
    fn observed_child_captures_stdout_and_stderr() {
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg("printf 'scheduler-stdout\\n'; printf 'model-stderr\\n' >&2");
        let (mut child, receiver, readers) = spawn_observed_child(&mut command);

        let status = child.wait().expect("wait for observed child");
        assert!(status.success(), "observed child status: {status}");
        for reader in readers {
            reader.join().expect("observed child log reader joins");
        }

        let lines = receiver
            .try_iter()
            .map(|observed| observed.line)
            .collect::<Vec<_>>();
        assert!(
            lines.iter().any(|line| line == "scheduler-stdout"),
            "stdout must reach the live observer: {lines:?}"
        );
        assert!(
            lines.iter().any(|line| line == "model-stderr"),
            "stderr must reach the live observer: {lines:?}"
        );
    }

    #[test]
    fn observed_child_drains_both_full_pipes_without_deadlock() {
        let mut command = Command::new("/usr/bin/perl");
        command.args([
            "-e",
            r#"
                for (1..1024) {
                    print STDOUT ("o" x 128), "\n";
                    print STDERR ("e" x 128), "\n";
                }
                print STDOUT "stdout-sentinel\n";
                print STDERR "stderr-sentinel\n";
            "#,
        ]);
        let (mut child, receiver, readers) = spawn_observed_child(&mut command);

        let deadline = Instant::now() + Duration::from_secs(10);
        let status = loop {
            if let Some(status) = child.try_wait().expect("poll saturated child") {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                for reader in readers {
                    let _ = reader.join();
                }
                panic!("concurrent stdout/stderr capture deadlocked on full pipes");
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(status.success(), "saturated child status: {status}");
        for reader in readers {
            reader.join().expect("saturated child log reader joins");
        }

        let lines = receiver
            .try_iter()
            .map(|observed| observed.line)
            .collect::<Vec<_>>();
        assert!(
            lines.iter().any(|line| line == "stdout-sentinel"),
            "stdout sentinel must survive pipe saturation"
        );
        assert!(
            lines.iter().any(|line| line == "stderr-sentinel"),
            "stderr sentinel must survive pipe saturation"
        );
    }

    #[test]
    fn joined_readers_deliver_final_scheduler_completion() {
        let mut command = Command::new("/bin/sh");
        command.arg("-c").arg(
            "printf '[scheduler] ambient turn started job=Classification cpu_percent=Some(5.0) available_memory_mb=Some(8000)\\n'; \
             printf 'native-stderr-noise\\n' >&2; \
             printf '[scheduler] ambient job=Classification selected=true llm_calls=1 panicked=false elapsed_ms=100 next_eligible_ms=120000\\n'",
        );
        let (mut child, receiver, readers) = spawn_observed_child(&mut command);
        let status = child.wait().expect("wait for scheduler-log child");
        assert!(status.success(), "scheduler-log child status: {status}");

        let mut observer = LiveObserver::with_memory_floor_mb(2_500);
        join_log_readers_and_drain(readers, &receiver, &mut observer);
        assert_eq!(
            observer.selected_turns.len(),
            1,
            "final completion must be drained after both readers join"
        );
        assert_eq!(observer.selected_turns[0].job, "Classification");
    }

    #[test]
    fn timeout_diagnostics_are_bounded_and_actionable() {
        let base = Instant::now();
        let mut observer = LiveObserver::with_memory_floor_mb(2_500);
        observer.observe_line("[on-device] model qwen3-4b loaded and available", base);
        observer.observe_line(
            "[scheduler] heavy work deferred reason=CpuBusy cpu_percent=Some(23.5) available_memory_mb=Some(7000)",
            base,
        );
        observer.observe_line(
            "[scheduler] heavy work deferred reason=CpuBusy cpu_percent=Some(24.5) available_memory_mb=Some(6900)",
            base,
        );
        observer.observe_line(
            "[scheduler] heavy work deferred reason=MemoryPressure cpu_percent=Some(8.0) available_memory_mb=Some(2100)",
            base,
        );
        observer.observe_line(
            "[scheduler] ambient turn started job=Classification cpu_percent=Some(5.0) available_memory_mb=Some(8000)",
            base + Duration::from_secs(1),
        );
        observer.observe_line(
            "[scheduler] ambient job=Classification selected=true llm_calls=1 panicked=false elapsed_ms=100 next_eligible_ms=120000",
            base + Duration::from_secs(2),
        );

        let diagnostics = observer.diagnostic_summary();
        assert!(diagnostics.contains("completed_turns=1"), "{diagnostics}");
        assert!(
            diagnostics.contains("selected_jobs=[\"Classification\"]"),
            "{diagnostics}"
        );
        assert!(diagnostics.contains("active=None"), "{diagnostics}");
        assert!(diagnostics.contains("model_loads=1"), "{diagnostics}");
        assert!(
            diagnostics.contains(
                "CpuBusy: count=2 last_cpu_percent=Some(24.5) last_available_memory_mb=Some(6900)"
            ),
            "{diagnostics}"
        );
        assert!(
            diagnostics.contains("MemoryPressure: count=1 last_cpu_percent=Some(8.0) last_available_memory_mb=Some(2100)"),
            "{diagnostics}"
        );
        assert!(
            diagnostics.contains("ambient turn started job=Classification"),
            "{diagnostics}"
        );
        assert!(
            diagnostics.contains("ambient job=Classification selected=true"),
            "{diagnostics}"
        );
        assert!(
            diagnostics.len() < 4_096,
            "diagnostics must stay bounded: {} bytes",
            diagnostics.len()
        );
    }

    #[test]
    fn observer_rejects_heavy_automatic_work_in_the_isolated_fixture() {
        let mut observer = LiveObserver::with_memory_floor_mb(2_500);
        let heavy = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            observer.observe_line(
                "[scheduler] automatic trigger=Idle selected=true llm_calls=0 panicked=false elapsed_ms=10 next_eligible_ms=120000",
                Instant::now(),
            );
        }));
        assert!(
            heavy.is_err(),
            "isolated fixture must not silently extend the production deadline"
        );
    }

    #[test]
    fn observer_rejects_start_telemetry_below_the_memory_floor() {
        let mut observer = LiveObserver::with_memory_floor_mb(2_500);
        let low_memory = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            observer.observe_line(
                "[scheduler] ambient turn started job=Classification cpu_percent=Some(5.0) available_memory_mb=Some(2499)",
                Instant::now(),
            );
        }));
        assert!(
            low_memory.is_err(),
            "ambient start must carry admitted CPU and memory telemetry"
        );
    }

    #[test]
    fn watchdog_fails_closed_for_thermal_memory_and_unavailable_telemetry() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(
            safety_violation(1, 16 * gib, 8 * gib),
            Some("thermal_not_nominal")
        );
        assert_eq!(
            safety_violation(0, 16 * gib, 2 * gib),
            Some("memory_pressure")
        );
        assert_eq!(
            safety_violation(0, 0, 0),
            Some("memory_telemetry_unavailable")
        );
        assert_eq!(safety_violation(0, 16 * gib, 8 * gib), None);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "real cached model + production scheduler timing; run via profile-ambient-rb01.sh daemon"]
    async fn persistent_provider_respects_production_cooldown() {
        assert_eq!(
            std::env::var(OPT_IN_ENV).ok().as_deref(),
            Some("1"),
            "real daemon profile requires explicit opt-in"
        );
        let fastembed_cache = std::env::var("WENLAN_TEST_FASTEMBED_CACHE")
            .expect("real daemon profile requires a preflighted FastEmbed cache");
        assert!(
            !fastembed_cache.trim().is_empty(),
            "real daemon profile requires a non-empty FastEmbed cache path"
        );

        let run_root = std::path::PathBuf::from(
            std::env::var("WENLAN_RB01_RUN_DIR")
                .expect("real daemon profile requires an external evidence directory"),
        );
        assert!(
            !run_root.exists(),
            "live daemon evidence directory must be unique per run: {}",
            run_root.display()
        );
        std::fs::create_dir_all(&run_root).expect("create persistent daemon evidence root");
        let data_dir = run_root.join("data");
        let knowledge_path = run_root.join("pages");
        std::fs::create_dir_all(&data_dir).expect("create temp data dir");
        std::fs::create_dir_all(&knowledge_path).expect("create temp knowledge dir");
        std::fs::write(
            data_dir.join("config.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "setup_completed": true,
                "knowledge_path": knowledge_path,
                "on_device_model": MODEL_ID,
                "everyday_source": "on_device",
                "synthesis_source": "on_device",
                "reranker_mode": "off",
                "page_map_auto_suggest": false
            }))
            .expect("serialize isolated config"),
        )
        .expect("write isolated config");

        let port_file = run_root.join("port");
        let home = std::env::var_os("HOME").expect("profile HOME");
        let path = std::env::var_os("PATH").expect("profile PATH");
        let tmpdir =
            std::env::var_os("TMPDIR").unwrap_or_else(|| std::ffi::OsString::from("/private/tmp"));
        let mut command = Command::new(binary_path());
        command
            .env_clear()
            .env("HOME", home)
            .env("PATH", path)
            .env("TMPDIR", tmpdir)
            .env("WENLAN_BIND_ADDR", "127.0.0.1:0")
            .env("WENLAN_DATA_DIR", &data_dir)
            .env("WENLAN_PORT_FILE", &port_file)
            .env(
                "RUST_LOG",
                "info,hyper=warn,reqwest=warn,wenlan_server::scheduler=debug",
            )
            .env("NO_COLOR", "1")
            .env("WENLAN_TEST_FASTEMBED_CACHE", fastembed_cache)
            .env("HF_ENDPOINT", "http://127.0.0.1:9");
        let (child, receiver, readers) = spawn_observed_child(&mut command);
        let mut child = ChildGuard(child);
        let total_memory_bytes = system_total_memory_bytes();
        let memory_floor_bytes = (total_memory_bytes.saturating_mul(15) / 100)
            .max(2 * 1024 * 1024 * 1024)
            .saturating_add(ON_DEVICE_INFERENCE_HEADROOM_BYTES);
        let memory_floor_mb = memory_floor_bytes.div_ceil(1024 * 1024);
        let mut observer = LiveObserver::with_memory_floor_mb(memory_floor_mb);
        let port = wait_for_port(&mut child, &port_file).await;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("build local HTTP client");

        let mut safety_watchdog = SafetyWatchdog::new(Instant::now(), thermal_helper_path());
        wait_for_model_load(
            &mut child,
            &receiver,
            &mut observer,
            &client,
            port,
            &mut safety_watchdog,
        )
        .await;

        let store = client
            .post(format!("http://127.0.0.1:{port}/api/memory/store"))
            .header("x-agent-name", "rb01-live-harness")
            .json(&serde_json::json!({
                "content": "The RB-01 scheduler must enrich one durable item per admitted turn while preserving foreground CPU and memory headroom.",
                "memory_type": "fact",
                "space": "rb01-live"
            }))
            .send()
            .await
            .expect("store synthetic memory")
            .error_for_status()
            .expect("synthetic memory store succeeds");
        let store: serde_json::Value = store.json().await.expect("store response JSON");
        let source_id = store["source_id"]
            .as_str()
            .expect("store response source_id")
            .to_owned();

        let deadline = Instant::now() + Duration::from_secs(27 * 60);
        let mut turn_model_checks = 0usize;
        let mut mid_cooldown_residency_checks = Vec::new();
        let mut next_residency_check_at = None;
        while observer.selected_turns.len() < 2 {
            let before = observer.selected_turns.len();
            drain_logs(&receiver, &mut observer);
            let now = Instant::now();
            observer.sample_rss_if_due(child.0.id(), now);
            assert_child_running(&mut child, "while waiting for ambient turns");

            sample_safety_if_due(&mut child, &client, port, &mut safety_watchdog).await;

            if observer.selected_turns.len() > before {
                assert_model_loaded(&client, port).await;
                turn_model_checks += observer.selected_turns.len() - before;
                if before == 0 {
                    next_residency_check_at = Some(Instant::now() + RESIDENCY_CHECK_INTERVAL);
                }
            }
            if observer.selected_turns.len() == 1
                && next_residency_check_at.is_some_and(|due| Instant::now() >= due)
            {
                assert_model_loaded(&client, port).await;
                let checked_at = Instant::now();
                mid_cooldown_residency_checks.push(
                    checked_at
                        .saturating_duration_since(safety_watchdog.started_at)
                        .as_millis()
                        .try_into()
                        .unwrap_or(u64::MAX),
                );
                let mut next = next_residency_check_at.expect("residency deadline exists");
                while next <= checked_at {
                    next += RESIDENCY_CHECK_INTERVAL;
                }
                next_residency_check_at = Some(next);
            }
            if now >= deadline {
                let diagnostic = observer.diagnostic_summary();
                let _ = client
                    .post(format!("http://127.0.0.1:{port}/api/shutdown"))
                    .send()
                    .await;
                let status = wait_for_exit(&mut child, Duration::from_secs(5)).await;
                emit_recovery_report(&observer, "deadline_without_safety_violation");
                panic!(
                    "two selected production-timing ambient turns did not complete within 27 minutes; cooperative shutdown status={status}: {diagnostic}"
                );
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        let thermal_work_max_elapsed_ms = emit_recovery_report(&observer, "two_turns_completed");

        let shutdown = client
            .post(format!("http://127.0.0.1:{port}/api/shutdown"))
            .send()
            .await
            .expect("request cooperative shutdown");
        assert!(shutdown.status().is_success());
        let status = wait_for_exit(&mut child, Duration::from_secs(5)).await;
        assert!(status.success(), "cooperative shutdown status: {status}");
        join_log_readers_and_drain(readers, &receiver, &mut observer);

        assert!(
            observer.active.is_none(),
            "daemon exited with an ambient turn still active"
        );
        assert_eq!(
            observer.model_loads, 1,
            "persistent provider must load exactly once"
        );
        assert_eq!(
            turn_model_checks, 2,
            "model residency checked after each selected turn"
        );
        assert!(
            mid_cooldown_residency_checks.len() >= MIN_MID_COOLDOWN_RESIDENCY_CHECKS,
            "expected at least {MIN_MID_COOLDOWN_RESIDENCY_CHECKS} provider checks between selected turns, observed {}",
            mid_cooldown_residency_checks.len()
        );
        let first = &observer.selected_turns[0];
        let second = &observer.selected_turns[1];
        assert_eq!(
            [first.job.as_str(), second.job.as_str()],
            ["Classification", "StructuredExtract"],
            "the live proof must observe the classification -> structured dependency"
        );
        let cooldown_gap = second.started.saturating_duration_since(first.completed);
        assert!(
            cooldown_gap >= MIN_PRODUCTION_COOLDOWN,
            "selected turns started only {cooldown_gap:?} apart"
        );

        let receipts = current_ok_receipts(
            &data_dir.join("memorydb").join("origin_memory.db"),
            &source_id,
        )
        .await;
        assert!(
            receipts
                .iter()
                .any(|(step, version)| step == "classify" && *version == 1),
            "classification turn must leave a current receipt: {receipts:?}"
        );
        assert!(
            receipts
                .iter()
                .any(|(step, version)| step == "structured_extract" && *version == 1),
            "structured turn must leave a current receipt: {receipts:?}"
        );

        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "rb01_live_daemon": {
                    "model": MODEL_ID,
                    "model_loads": observer.model_loads,
                    "selected_jobs": [
                        first.job.as_str(),
                        second.job.as_str()
                    ],
                    "llm_calls": [
                        first.llm_calls,
                        second.llm_calls
                    ],
                    "cooldown_gap_ms": cooldown_gap.as_millis(),
                    "next_eligible_ms": [
                        first.next_eligible_ms,
                        second.next_eligible_ms
                    ],
                    "turn_model_residency_checks": turn_model_checks,
                    "mid_cooldown_model_residency_checks_ms": mid_cooldown_residency_checks,
                    "safety_sample_interval_ms": SAFETY_SAMPLE_INTERVAL.as_millis(),
                    "safety_samples": safety_watchdog.samples.iter().map(|(observed_ms, sample)| {
                        serde_json::json!({
                            "observed_ms": observed_ms,
                            "thermal_state": sample.thermal_state,
                            "total_memory_bytes": sample.total_memory_bytes,
                            "available_memory_bytes": sample.available_memory_bytes
                        })
                    }).collect::<Vec<_>>(),
                    "admission_blocks": &observer.admission_blocks,
                    "durable_current_ok_receipts": receipts,
                    "peak_daemon_rss_bytes": observer.peak_rss_bytes
                },
                "report_elapsed_ms": thermal_work_max_elapsed_ms
            }))
            .expect("serialize live daemon evidence")
        );
    }
}
