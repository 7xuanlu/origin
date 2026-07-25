// SPDX-License-Identifier: Apache-2.0
//! Integration tests for the origin CLI. Offline (no daemon required).

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn cli() -> Command {
    Command::cargo_bin("wenlan").expect("origin binary built")
}

fn cli_with_isolated_runtime(runtime: &IsolatedRuntime) -> Command {
    let mut cmd = cli();
    // Prepend fake_bin to the existing PATH using the platform separator
    // (`:` on Unix, `;` on Windows). `std::env::join_paths` handles both.
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    let mut entries: Vec<PathBuf> = vec![runtime.fake_bin.path().to_path_buf()];
    entries.extend(std::env::split_paths(&path_var));
    let joined = std::env::join_paths(entries).expect("join PATH entries");
    cmd.env("HOME", runtime.home.path())
        .env("USERPROFILE", runtime.home.path())
        .env("WENLAN_DATA_DIR", runtime.data.path())
        .env("WENLAN_HOST", "http://127.0.0.1:9")
        .env("WENLAN_BIND_ADDR", "127.0.0.1:9")
        .env("PATH", &joined);
    cmd
}

struct IsolatedRuntime {
    root: TempDir,
    home: TempDir,
    data: TempDir,
    fake_bin: TempDir,
}

impl IsolatedRuntime {
    fn new() -> Self {
        let root = tempfile::tempdir().expect("temp root");
        let home = tempfile::tempdir_in(root.path()).expect("temp home");
        let data = tempfile::tempdir_in(root.path()).expect("temp data");
        let fake_bin = tempfile::tempdir_in(root.path()).expect("temp fake bin");
        #[cfg(target_os = "macos")]
        write_fake_launchctl(fake_bin.path());
        ensure_sibling_origin_server();
        ensure_sibling_origin_mcp();
        Self {
            root,
            home,
            data,
            fake_bin,
        }
    }

    #[cfg(target_os = "macos")]
    fn service_unit_path(&self) -> PathBuf {
        self.home
            .path()
            .join("Library/LaunchAgents/com.wenlan.server.plist")
    }

    #[cfg(target_os = "macos")]
    fn config_path(&self) -> PathBuf {
        self.data.path().join("config.json")
    }

    #[cfg(target_os = "macos")]
    fn root_exists(&self) -> bool {
        self.root.path().exists()
    }
}

#[cfg(target_os = "macos")]
fn write_fake_launchctl(fake_bin: &Path) {
    let path = fake_bin.join("launchctl");
    fs::write(
        &path,
        "#!/bin/sh\nif [ -n \"$WENLAN_TEST_LAUNCHCTL_LOG\" ]; then echo \"$@\" >> \"$WENLAN_TEST_LAUNCHCTL_LOG\"; fi\ncase \"$1\" in\n  list) exit 0 ;;\n  load|unload) echo \"fake launchctl $1 $2\"; exit 0 ;;\n  bootout) echo \"fake launchctl $@\"; exit \"${WENLAN_TEST_LAUNCHCTL_BOOTOUT_EXIT:-0}\" ;;\n  print) echo \"fake launchctl $@\"; exit \"${WENLAN_TEST_LAUNCHCTL_PRINT_EXIT:-0}\" ;;\n  *) echo \"fake launchctl $@\"; exit 0 ;;\nesac\n",
    )
    .expect("write fake launchctl");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path)
            .expect("fake launchctl metadata")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("chmod fake launchctl");
    }
}

#[cfg(target_os = "macos")]
fn spawn_shutdown_stub() -> (String, std::sync::mpsc::Receiver<String>) {
    spawn_shutdown_stub_response("200 OK", "shutting down")
}

#[cfg(target_os = "macos")]
fn spawn_shutdown_failure_stub() -> (String, std::sync::mpsc::Receiver<String>) {
    spawn_shutdown_stub_response("500 Internal Server Error", "")
}

#[cfg(target_os = "macos")]
fn spawn_respawning_shutdown_stub() -> String {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind respawning shutdown stub");
    let address = listener.local_addr().expect("respawning stub address");
    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept shutdown request");
        let mut reader = BufReader::new(stream);
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .expect("read shutdown request line");
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("read shutdown header");
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        reader
            .get_mut()
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\nConnection: close\r\n\r\nshutting down",
            )
            .expect("write shutdown response");
        drop(reader);
        drop(listener);

        thread::sleep(std::time::Duration::from_millis(400));
        let restarted = TcpListener::bind(address).expect("rebind respawned daemon stub");
        let (health, _) = restarted
            .accept()
            .expect("accept health probe after respawn");
        let mut health_reader = BufReader::new(health);
        let mut request_line = String::new();
        health_reader
            .read_line(&mut request_line)
            .expect("read respawned health request line");
        loop {
            let mut line = String::new();
            health_reader
                .read_line(&mut line)
                .expect("read respawned health header");
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        health_reader
            .get_mut()
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}",
            )
            .expect("write respawned health response");
    });
    format!("http://{address}")
}

#[cfg(target_os = "macos")]
fn spawn_hanging_health_probe_stub() -> String {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind slow-closing shutdown stub");
    let address = listener.local_addr().expect("slow-closing stub address");
    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept shutdown request");
        let mut reader = BufReader::new(stream);
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .expect("read shutdown request line");
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("read shutdown header");
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        reader
            .get_mut()
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\nConnection: close\r\n\r\nshutting down",
            )
            .expect("write shutdown response");
        drop(reader);

        // A real server can stop servicing health requests before its socket
        // is fully closed. Hold one accepted probe open long enough for the
        // client's per-request timeout, then close the listener for real.
        let (health, _) = listener.accept().expect("accept transient health probe");
        thread::sleep(std::time::Duration::from_millis(2_200));
        drop(health);
        drop(listener);
    });
    format!("http://{address}")
}

#[cfg(unix)]
fn spawn_keepalive_shutdown_stub() -> (String, std::sync::mpsc::Receiver<bool>) {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind keepalive shutdown stub");
    let address = listener.local_addr().expect("keepalive stub address");
    let (close_header_tx, close_header_rx) = mpsc::channel();
    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept shutdown request");
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(12)))
            .expect("set keepalive stub timeout");
        let mut reader = BufReader::new(stream);

        let read_request =
            |reader: &mut BufReader<std::net::TcpStream>| -> Option<(String, Vec<String>)> {
                let mut request_line = String::new();
                match reader.read_line(&mut request_line) {
                    Ok(0) => return None,
                    Ok(_) => {}
                    Err(error)
                        if matches!(
                            error.kind(),
                            std::io::ErrorKind::ConnectionReset
                                | std::io::ErrorKind::ConnectionAborted
                                | std::io::ErrorKind::UnexpectedEof
                        ) =>
                    {
                        return None;
                    }
                    Err(error) => panic!("read keepalive request line: {error}"),
                }
                let mut headers = Vec::new();
                loop {
                    let mut line = String::new();
                    reader
                        .read_line(&mut line)
                        .expect("read keepalive request header");
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    headers.push(line);
                }
                Some((request_line, headers))
            };

        let (shutdown, shutdown_headers) = read_request(&mut reader).expect("shutdown request");
        assert_eq!(shutdown, "POST /api/shutdown HTTP/1.1\r\n");
        let close_requested = shutdown_headers.iter().any(|header| {
            header.split_once(':').is_some_and(|(name, value)| {
                name.eq_ignore_ascii_case("connection")
                    && value.trim().eq_ignore_ascii_case("close")
            })
        });
        close_header_tx
            .send(close_requested)
            .expect("report shutdown connection header");

        if close_requested {
            // Hyper 1.x marks an HTTP/1.1 connection non-persistent when the
            // request carries `Connection: close`, then closes it after the
            // response. Model that server contract instead of leaving a raw
            // TCP socket alive in a state Hyper would never expose.
            reader
                .get_mut()
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
                .expect("write closing shutdown response");
            drop(reader);
            drop(listener);
            return;
        }

        reader
            .get_mut()
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .expect("write shutdown response");

        // Model hyper graceful shutdown: stop accepting fresh connections,
        // but keep the already-open HTTP/1.1 connection alive until its client
        // releases it. Reusing this connection for health probes prevents the
        // server from ever reaching its process-exit path.
        drop(listener);
        while let Some((request, _)) = read_request(&mut reader) {
            assert_eq!(request, "GET /api/health HTTP/1.1\r\n");
            reader
                .get_mut()
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}")
                .expect("write keepalive health response");
        }
    });
    (format!("http://{address}"), close_header_rx)
}

#[cfg(target_os = "macos")]
fn spawn_shutdown_stub_response(
    status: &'static str,
    body: &'static str,
) -> (String, std::sync::mpsc::Receiver<String>) {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind shutdown stub");
    let base = format!(
        "http://{}",
        listener.local_addr().expect("shutdown stub address")
    );
    let (sent, received) = mpsc::channel();
    thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept shutdown request");
        let mut reader = BufReader::new(stream);
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .expect("read shutdown request line");
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("read shutdown header");
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        sent.send(request_line).expect("record shutdown request");
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        reader
            .get_mut()
            .write_all(response.as_bytes())
            .expect("write shutdown response");
    });
    (base, received)
}

fn bind_addr_from_stub_host(host: &str) -> &str {
    host.strip_prefix("http://")
        .expect("shutdown stub host must use http")
}

fn write_fake_command(fake_bin: &Path, name: &str) {
    #[cfg(unix)]
    {
        let path = fake_bin.join(name);
        fs::write(
            &path,
            "#!/bin/sh\nprintf '%s' \"${0##*/}\" >> \"$WENLAN_TEST_CLI_LOG\"\nfor arg in \"$@\"; do printf '\\t%s' \"$arg\" >> \"$WENLAN_TEST_CLI_LOG\"; done\nprintf '\\n' >> \"$WENLAN_TEST_CLI_LOG\"\nexit 0\n",
        )
        .expect("write fake command");
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&path)
            .expect("fake command metadata")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("chmod fake command");
    }
    #[cfg(windows)]
    {
        // Windows PATH lookup honors PATHEXT; .cmd is in the default list.
        // The .cmd file appends `name<TAB>arg1<TAB>arg2...<LF>` to the log so
        // the Unix-side regression assertion keeps comparing the same shape.
        let path = fake_bin.join(format!("{name}.cmd"));
        // %~n0 = batch script basename (without extension). Loop %%i over args.
        let script = format!(
            "@echo off\r\n\
             setlocal enableextensions enabledelayedexpansion\r\n\
             set \"LINE={name}\"\r\n\
             :loop\r\n\
             if \"%~1\"==\"\" goto done\r\n\
             set \"LINE=!LINE!\t%~1\"\r\n\
             shift\r\n\
             goto loop\r\n\
             :done\r\n\
             >>\"%WENLAN_TEST_CLI_LOG%\" echo(!LINE!\r\n\
             exit /b 0\r\n"
        );
        fs::write(&path, script).expect("write fake .cmd");
    }
}

fn ensure_sibling_origin_server() {
    let origin_bin = assert_cmd::cargo::cargo_bin("wenlan");
    let server = origin_bin
        .parent()
        .expect("origin binary has parent")
        .join("wenlan-server");
    if !server.exists() {
        fs::write(&server, "#!/bin/sh\nexit 0\n").expect("write fake origin-server sibling");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&server)
                .expect("fake origin-server metadata")
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(server, perms).expect("chmod fake origin-server");
        }
    }
}

fn ensure_sibling_origin_mcp() {
    let path = origin_mcp_sibling();
    if !path.exists() {
        fs::write(&path, "#!/bin/sh\nexit 0\n").expect("write fake origin-mcp sibling");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&path)
                .expect("fake origin-mcp metadata")
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).expect("chmod fake origin-mcp");
        }
    }
}

fn origin_mcp_sibling() -> PathBuf {
    let origin_bin = assert_cmd::cargo::cargo_bin("wenlan");
    origin_bin
        .parent()
        .expect("origin binary has parent")
        .join("wenlan-mcp")
}

fn origin_mcp_sibling_arg() -> String {
    origin_mcp_sibling().display().to_string()
}

#[test]
fn top_level_help() {
    cli()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Wenlan CLI"));
}

#[test]
fn version_flag() {
    cli()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("wenlan"));
}

#[test]
fn each_subcommand_has_help() {
    for sub in [
        "status",
        "search",
        "recall",
        "capture",
        "memories",
        "agents",
        "setup",
        "background",
        "restart",
        "doctor",
        "models",
        "keys",
        "connect",
        "sources",
    ] {
        cli().args([sub, "--help"]).assert().success();
    }
}

#[test]
fn removed_top_level_commands_are_not_advertised() {
    let output = cli()
        .arg("--help")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let help = String::from_utf8(output).expect("help is utf8");
    for removed in [
        "install",
        "uninstall",
        "mcp",
        "store",
        "list",
        "space",
        "model",
        "key",
        "reranker",
        "ingest",
    ] {
        let advertised: Vec<&str> = help
            .lines()
            .filter_map(|line| line.strip_prefix("  "))
            .filter_map(|line| line.split_whitespace().next())
            .collect();
        assert!(
            !advertised.contains(&removed),
            "removed command advertised: {removed}\n{help}"
        );
    }
}

#[test]
fn connect_command_has_help() {
    for args in [
        &["connect", "--help"][..],
        &["connect", "claude-code", "--help"][..],
    ] {
        cli().args(args).assert().success();
    }
}

#[test]
fn connect_claude_code_dry_run_explains_tools_only() {
    let runtime = IsolatedRuntime::new();

    cli_with_isolated_runtime(&runtime)
        .args(["connect", "claude-code", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "claude mcp add -s user wenlan -- {}",
            origin_mcp_sibling_arg()
        )))
        .stdout(predicate::str::contains("MCP tools only"))
        .stdout(predicate::str::contains("/brief"))
        .stdout(predicate::str::contains("/handoff"))
        .stdout(predicate::str::contains("/distill"))
        .stdout(predicate::str::contains("/setup"));
}

#[test]
fn connect_native_clients_run_add_without_destructive_remove() {
    let wenlan_mcp = origin_mcp_sibling_arg();
    let cases = [
        (
            "claude-code",
            "claude",
            format!("claude\tmcp\tadd\t-s\tuser\twenlan\t--\t{wenlan_mcp}\n"),
        ),
        (
            "codex",
            "codex",
            format!("codex\tmcp\tadd\twenlan\t--\t{wenlan_mcp}\n"),
        ),
        (
            "gemini",
            "gemini",
            format!("gemini\tmcp\tadd\t-s\tuser\twenlan\t{wenlan_mcp}\n"),
        ),
    ];

    for (client, binary, expected_log) in cases {
        let runtime = IsolatedRuntime::new();
        write_fake_command(runtime.fake_bin.path(), binary);
        let log = runtime.root.path().join(format!("{client}.log"));

        cli_with_isolated_runtime(&runtime)
            .env("WENLAN_TEST_CLI_LOG", &log)
            .args(["connect", client])
            .assert()
            .success()
            .stdout(predicate::str::contains("Configured Wenlan MCP"));

        // .cmd shells on Windows write CRLF line endings; the Unix shell
        // script writes LF. Normalize before the byte-for-byte compare.
        let actual = fs::read_to_string(log)
            .expect("fake client log")
            .replace("\r\n", "\n");
        assert_eq!(actual, expected_log.as_str());
    }
}

#[test]
fn connect_cursor_preserves_existing_servers_and_backs_up_changed_wenlan() {
    let runtime = IsolatedRuntime::new();
    let config_path = runtime.home.path().join(".cursor/mcp.json");
    fs::create_dir_all(config_path.parent().expect("cursor config parent")).unwrap();
    fs::write(
        &config_path,
        r#"{"mcpServers":{"wenlan":{"command":"old-wenlan"},"other":{"command":"other-cmd"}}}"#,
    )
    .unwrap();

    cli_with_isolated_runtime(&runtime)
        .args(["connect", "cursor"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Updated"));

    let updated = fs::read_to_string(&config_path).expect("updated cursor config");
    assert!(updated.contains(r#""other""#), "{updated}");
    // serde_json escapes backslashes in path strings (Windows `\` → `\\`).
    // Mirror the same transform on the expected fragment.
    let expected_command_json = origin_mcp_sibling_arg().replace('\\', "\\\\");
    assert!(
        updated.contains(&format!(r#""command": "{expected_command_json}""#)),
        "{updated}"
    );
    assert!(updated.contains("wenlan-mcp"), "{updated}");

    let backups: Vec<_> = fs::read_dir(config_path.parent().unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("mcp.json.bak.")
        })
        .collect();
    assert_eq!(backups.len(), 1, "expected one backup");
    let backup = fs::read_to_string(backups[0].path()).expect("backup content");
    assert!(backup.contains("old-wenlan"), "{backup}");
}

#[test]
fn connect_cursor_dry_run_prints_only_wenlan_block() {
    let runtime = IsolatedRuntime::new();
    let config_path = runtime.home.path().join(".cursor/mcp.json");
    fs::create_dir_all(config_path.parent().expect("cursor config parent")).unwrap();
    fs::write(
        &config_path,
        r#"{"mcpServers":{"private":{"command":"secret-tool","env":{"API_KEY":"SECRET_TOKEN"}}}}"#,
    )
    .unwrap();

    cli_with_isolated_runtime(&runtime)
        .args(["connect", "cursor", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("mcpServers.wenlan"))
        .stdout(predicate::str::contains("wenlan-mcp"))
        .stdout(predicate::str::contains("SECRET_TOKEN").not())
        .stdout(predicate::str::contains("secret-tool").not());

    let unchanged = fs::read_to_string(&config_path).expect("cursor config unchanged");
    assert!(unchanged.contains("SECRET_TOKEN"), "{unchanged}");
}

#[cfg(target_os = "macos")]
#[test]
fn connect_json_clients_write_expected_config_shapes() {
    let runtime = IsolatedRuntime::new();

    cli_with_isolated_runtime(&runtime)
        .args(["connect", "claude-desktop"])
        .assert()
        .success();
    let claude = fs::read_to_string(
        runtime
            .home
            .path()
            .join("Library/Application Support/Claude/claude_desktop_config.json"),
    )
    .expect("claude desktop config");
    assert!(claude.contains(r#""mcpServers""#), "{claude}");
    assert!(claude.contains("wenlan-mcp"), "{claude}");

    cli_with_isolated_runtime(&runtime)
        .current_dir(runtime.root.path())
        .args(["connect", "vscode"])
        .assert()
        .success();
    let vscode = fs::read_to_string(runtime.root.path().join(".vscode/mcp.json"))
        .expect("vscode workspace config");
    assert!(vscode.contains(r#""servers""#), "{vscode}");
    assert!(vscode.contains("wenlan-mcp"), "{vscode}");
}

#[test]
fn connect_invalid_json_fails_without_modifying_file() {
    let runtime = IsolatedRuntime::new();
    let config_path = runtime.home.path().join(".cursor/mcp.json");
    fs::create_dir_all(config_path.parent().expect("cursor config parent")).unwrap();
    fs::write(&config_path, "{not json").unwrap();

    cli_with_isolated_runtime(&runtime)
        .args(["connect", "cursor"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid JSON"));

    assert_eq!(
        fs::read_to_string(&config_path).expect("cursor config still exists"),
        "{not json"
    );
}

#[test]
fn setup_subcommands_have_help() {
    for args in [
        &["setup", "--help"][..],
        &["models", "list", "--help"][..],
        &["models", "install", "--help"][..],
        &["models", "status", "--help"][..],
        &["models", "reranker", "--help"][..],
        &["keys", "status", "--help"][..],
        &["keys", "set", "--help"][..],
        &["keys", "clear", "--help"][..],
        &["enrichment", "status", "--help"][..],
        &["enrichment", "configure", "--help"][..],
        &["enrichment", "disable", "--help"][..],
    ] {
        cli().args(args).assert().success();
    }
}

#[test]
fn invalid_subcommand_fails() {
    cli().arg("nonexistent-command").assert().failure();
}

#[test]
fn capture_text_and_file_conflict_bails() {
    // text=Some, file=Some -> bail at runtime (mutual exclusion)
    cli()
        .args(["capture", "some text", "--file", "/dev/null"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("either"));
}

#[test]
fn agents_edit_no_flags_bails() {
    cli()
        .args(["agents", "edit", "dummy-agent-name"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("No fields to update"));
}

#[test]
fn status_json_succeeds_when_daemon_is_unreachable() {
    cli()
        .env("WENLAN_HOST", "http://127.0.0.1:9")
        .args(["status", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"status\": \"unreachable\""));
}

#[test]
fn status_table_uses_origin_host_for_health_probe() {
    let runtime = IsolatedRuntime::new();

    cli_with_isolated_runtime(&runtime)
        .args(["status", "--format", "table"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Health: not reachable"))
        .stdout(predicate::str::contains("http://127.0.0.1:9/api/health"));
}

#[test]
fn doctor_uses_origin_host_for_health_probe() {
    cli()
        .env("WENLAN_HOST", "http://127.0.0.1:9")
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Daemon: not reachable on http://127.0.0.1:9/api/health",
        ));
}

#[cfg(not(target_os = "windows"))]
#[test]
fn service_unit_path_resolves_per_os() {
    let path =
        wenlan_cli::commands::service_unit_path().expect("service_unit_path should not fail");

    // The path must match the on-disk file `service-manager` 0.11 actually
    // writes. See `crates/origin-cli/src/commands/service.rs` for the rules
    // we mirror.
    #[cfg(target_os = "macos")]
    assert!(path
        .to_string_lossy()
        .ends_with("Library/LaunchAgents/com.wenlan.server.plist"));

    #[cfg(target_os = "linux")]
    assert!(path
        .to_string_lossy()
        .ends_with(".config/systemd/user/wenlan-server.service"));
}

#[cfg(target_os = "macos")]
#[test]
fn restart_after_install_succeeds_isolated() {
    let runtime = IsolatedRuntime::new();

    // Not installed yet → restart should fail with a helpful hint.
    cli_with_isolated_runtime(&runtime)
        .arg("restart")
        .assert()
        .failure()
        .stderr(predicate::str::contains("background process is not set up"));

    cli_with_isolated_runtime(&runtime)
        .args(["background", "on"])
        .assert()
        .success();

    // Installed → restart stops then starts the service and reports it.
    cli_with_isolated_runtime(&runtime)
        .arg("restart")
        .assert()
        .success()
        .stdout(predicate::str::contains("Restarted com.wenlan.server"));
}

#[cfg(target_os = "macos")]
#[test]
fn background_on_over_running_daemon_stops_first_isolated() {
    let runtime = IsolatedRuntime::new();
    let log = runtime.data.path().join("launchctl.log");

    // First background-on registers + starts the service.
    cli_with_isolated_runtime(&runtime)
        .args(["background", "on"])
        .assert()
        .success();

    // Second background-on (the upgrade case): clear the log, re-register, and assert
    // a stop-class launchctl call happened before the new start.
    let _ = fs::remove_file(&log);
    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_TEST_LAUNCHCTL_LOG", &log)
        .args(["background", "on"])
        .assert()
        .success();

    let calls = fs::read_to_string(&log).unwrap_or_default();
    // We assert specifically for "stop" — our explicit m.stop() emits
    // `launchctl stop com.wenlan.server`. service-manager's internal reinstall
    // logic emits `launchctl remove`, which is a different verb and does NOT
    // terminate a running process the same way. The test must discriminate
    // between our explicit stop and the library's internal unload-before-reload.
    assert!(
        calls.lines().any(|l| l.starts_with("stop ")),
        "second background-on must explicitly call `launchctl stop` before re-registering; launchctl calls were:\n{calls}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_boots_out_keepalive_job_without_stop_restart_race() {
    let runtime = IsolatedRuntime::new();
    let log = runtime.data.path().join("launchctl.log");

    cli_with_isolated_runtime(&runtime)
        .args(["background", "on"])
        .assert()
        .success();

    let _ = fs::remove_file(&log);
    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_TEST_LAUNCHCTL_LOG", &log)
        .args(["background", "off"])
        .assert()
        .success();

    let calls = fs::read_to_string(&log).unwrap_or_default();
    assert!(
        calls.lines().any(|line| line.starts_with("bootout ")),
        "background off must boot out the KeepAlive job while preserving its plist; launchctl calls were:\n{calls}"
    );
    assert!(
        !calls.lines().any(|line| line.starts_with("stop ")),
        "stopping a KeepAlive job before removal can spawn an orphan replacement; launchctl calls were:\n{calls}"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn setup_background_status_roundtrip_isolated() {
    let runtime = IsolatedRuntime::new();

    cli_with_isolated_runtime(&runtime)
        .args(["setup", "--basic"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Wenlan is set up for local memory",
        ))
        .stdout(predicate::str::contains(
            "Model-backed background enrichment is off",
        ));

    let config = fs::read_to_string(runtime.config_path()).expect("config written");
    assert!(config.contains(r#""setup_completed": true"#));
    assert!(config.contains(r#""on_device_model": null"#));
    assert!(config.contains(r#""anthropic_api_key": null"#));

    cli_with_isolated_runtime(&runtime)
        .args(["background", "on"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Installed and started com.wenlan.server",
        ));

    let plist = fs::read_to_string(runtime.service_unit_path()).expect("plist written");
    assert!(plist.contains("<string>com.wenlan.server</string>"));
    assert!(plist.contains("wenlan-server"));
    assert!(!plist.contains("origin</string>"));
    // The daemon owns bounded file logging. launchd must not append an
    // unbounded second copy of stdout/stderr across every restart.
    assert!(
        plist.contains("<key>StandardOutPath</key>"),
        "missing StandardOutPath in plist: {plist}"
    );
    assert!(
        plist.contains("<string>/dev/null</string>"),
        "launchd stdout/stderr must be discarded after daemon-owned logging: {plist}"
    );
    assert!(
        plist.contains("<key>StandardErrorPath</key>"),
        "missing StandardErrorPath in plist: {plist}"
    );
    assert!(
        !plist.contains("wenlan-server.stdout.log") && !plist.contains("wenlan-server.stderr.log"),
        "unbounded legacy launchd log paths remain in plist: {plist}"
    );
    assert!(
        plist.contains("<key>EnvironmentVariables</key>"),
        "missing EnvironmentVariables in plist: {plist}"
    );
    assert!(
        plist.contains("<key>RUST_LOG</key>"),
        "RUST_LOG not propagated through launchd plist: {plist}"
    );
    assert!(
        plist.contains("<key>WENLAN_DATA_DIR</key>"),
        "WENLAN_DATA_DIR ownership marker missing from launchd plist: {plist}"
    );
    assert!(
        plist.contains(&format!(
            "<string>{}</string>",
            runtime.data.path().display()
        )),
        "selected data root not propagated through launchd plist: {plist}"
    );

    cli_with_isolated_runtime(&runtime)
        .args(["status", "--format", "json"])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"status\": \"unreachable\""));

    let launchctl_log = runtime.data.path().join("background-off-launchctl.log");
    let (host, shutdown_requests) = spawn_shutdown_stub();
    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_TEST_LAUNCHCTL_LOG", &launchctl_log)
        .env("WENLAN_BIND_ADDR", bind_addr_from_stub_host(&host))
        .args(["background", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stopped com.wenlan.server"))
        .stdout(predicate::str::contains("Uninstalled").not());

    assert!(
        runtime.service_unit_path().exists(),
        "background off preserves the launchd plist"
    );
    let calls = fs::read_to_string(&launchctl_log).unwrap_or_default();
    assert!(
        !calls.lines().any(|line| line.starts_with("bootout ")),
        "graceful shutdown must not boot out the registered LaunchAgent; calls were:\n{calls}"
    );
    assert_eq!(
        shutdown_requests
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("CLI must request graceful daemon shutdown"),
        "POST /api/shutdown HTTP/1.1\r\n"
    );
    assert!(
        runtime.root_exists(),
        "temp runtime remains alive until test end"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_without_registration_is_a_successful_noop() {
    let runtime = IsolatedRuntime::new();

    cli_with_isolated_runtime(&runtime)
        .args(["background", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "already stopped; no registration found",
        ))
        .stdout(predicate::str::contains("Uninstalled").not());

    assert!(!runtime.service_unit_path().exists());
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_with_registration_and_unreachable_daemon_stops_manager_but_keeps_registration() {
    let runtime = IsolatedRuntime::new();
    let plist = runtime.service_unit_path();
    fs::create_dir_all(plist.parent().expect("launch agent directory"))
        .expect("create fake launch agent directory");
    fs::write(&plist, "fake registered launch agent").expect("write fake registration");

    // Reserve an ephemeral address, then release it immediately so the shutdown
    // request deterministically takes the connection-refused path. A registered
    // job may be between supervisor attempts here; `background off` must still
    // stop the manager job rather than merely report success.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve unused port");
    let bind_addr = listener.local_addr().expect("reserved loopback address");
    drop(listener);

    let launchctl_log = runtime.data.path().join("unreachable-background-off.log");
    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_TEST_LAUNCHCTL_LOG", &launchctl_log)
        .env("WENLAN_BIND_ADDR", bind_addr.to_string())
        .args(["background", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stopped com.wenlan.server"))
        .stdout(predicate::str::contains("Uninstalled").not());

    let calls = fs::read_to_string(&launchctl_log).unwrap_or_default();
    assert!(
        calls.lines().any(|line| line.starts_with("bootout ")),
        "an unreachable registered daemon may still be starting; background off must stop its manager job; calls were:\n{calls}"
    );
    assert!(
        plist.exists(),
        "manager fallback must stop the daemon without deleting its registration"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_without_registration_stops_reachable_daemon() {
    let runtime = IsolatedRuntime::new();
    let (host, requests) = spawn_shutdown_stub();

    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_BIND_ADDR", bind_addr_from_stub_host(&host))
        .args(["background", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stopped com.wenlan.server"))
        .stdout(predicate::str::contains("No background registration found"));

    assert_eq!(
        requests
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("CLI must request daemon shutdown"),
        "POST /api/shutdown HTTP/1.1\r\n"
    );
    assert!(!runtime.service_unit_path().exists());
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_rejects_daemon_that_respawns_after_initial_disconnect() {
    let runtime = IsolatedRuntime::new();
    let host = spawn_respawning_shutdown_stub();

    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_BIND_ADDR", bind_addr_from_stub_host(&host))
        .args(["background", "off"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("daemon remained reachable"));
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_tolerates_transient_probe_timeout_before_socket_closes() {
    let runtime = IsolatedRuntime::new();
    let host = spawn_hanging_health_probe_stub();

    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_BIND_ADDR", bind_addr_from_stub_host(&host))
        .args(["background", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Stopped com.wenlan.server. No background registration found.",
        ));
}

#[cfg(unix)]
#[test]
fn background_off_does_not_keep_graceful_shutdown_connection_alive() {
    let runtime = IsolatedRuntime::new();
    let (host, close_header) = spawn_keepalive_shutdown_stub();

    let assertion = cli_with_isolated_runtime(&runtime)
        .env("WENLAN_BIND_ADDR", bind_addr_from_stub_host(&host))
        .args(["background", "off"])
        .assert();
    assert!(
        close_header
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("shutdown stub must report the request header"),
        "shutdown POST must request connection close"
    );
    assertion.success().stdout(predicate::str::contains(
        "Stopped com.wenlan.server. No background registration found.",
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_ignores_remote_wenlan_host_for_local_lifecycle() {
    let runtime = IsolatedRuntime::new();
    let (local_host, local_requests) = spawn_shutdown_stub();
    let (remote_host, remote_requests) = spawn_shutdown_stub();

    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_HOST", remote_host)
        .env("WENLAN_BIND_ADDR", bind_addr_from_stub_host(&local_host))
        .args(["background", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stopped com.wenlan.server"));

    assert_eq!(
        local_requests
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("CLI must shut down the local bind address"),
        "POST /api/shutdown HTTP/1.1\r\n"
    );
    assert!(
        remote_requests
            .recv_timeout(std::time::Duration::from_millis(100))
            .is_err(),
        "background lifecycle must never target remote WENLAN_HOST"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_succeeds_when_launchd_job_is_already_absent() {
    let runtime = IsolatedRuntime::new();
    let log = runtime.data.path().join("already-absent-launchctl.log");
    let (host, _requests) = spawn_shutdown_failure_stub();

    cli_with_isolated_runtime(&runtime)
        .args(["background", "on"])
        .assert()
        .success();

    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_TEST_LAUNCHCTL_LOG", &log)
        .env("WENLAN_BIND_ADDR", bind_addr_from_stub_host(&host))
        .env("WENLAN_TEST_LAUNCHCTL_BOOTOUT_EXIT", "3")
        .env("WENLAN_TEST_LAUNCHCTL_PRINT_EXIT", "113")
        .args(["background", "off"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stopped com.wenlan.server"));

    let calls = fs::read_to_string(&log).expect("already-absent launchctl log");
    assert!(calls.lines().any(|line| line.starts_with("print gui/")));
    assert!(runtime.service_unit_path().exists());
}

#[cfg(target_os = "macos")]
#[test]
fn background_off_surfaces_unconfirmed_launchd_failure() {
    let runtime = IsolatedRuntime::new();
    let (host, _requests) = spawn_shutdown_failure_stub();

    cli_with_isolated_runtime(&runtime)
        .args(["background", "on"])
        .assert()
        .success();

    cli_with_isolated_runtime(&runtime)
        .env("WENLAN_TEST_LAUNCHCTL_BOOTOUT_EXIT", "5")
        .env("WENLAN_TEST_LAUNCHCTL_PRINT_EXIT", "0")
        .env("WENLAN_BIND_ADDR", bind_addr_from_stub_host(&host))
        .args(["background", "off"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("launchctl bootout failed"));

    assert!(runtime.service_unit_path().exists());
}
