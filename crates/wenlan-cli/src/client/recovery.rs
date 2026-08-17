// SPDX-License-Identifier: Apache-2.0
//! Connection recovery for the CLI's registered daemon service.

use anyhow::{Context, Result};
use std::time::{Duration, Instant};

use crate::commands::service;

pub(crate) const NO_SERVICE_HINT: &str =
    "daemon not reachable and no background service is registered — run `wenlan background on`";

pub(crate) fn autostart_allowed(env_no_autostart: Option<&str>, recovery_enabled: bool) -> bool {
    recovery_enabled && matches!(env_no_autostart, None | Some(""))
}

pub(crate) fn autostart_allowed_from_env(recovery_enabled: bool) -> bool {
    let env_no_autostart = std::env::var_os("WENLAN_NO_AUTOSTART")
        .filter(|value| !value.is_empty())
        .map(|_| "set");
    autostart_allowed(env_no_autostart, recovery_enabled)
}

pub(crate) fn is_local_daemon_url(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };
    matches!(
        url.host_str(),
        Some("127.0.0.1" | "localhost" | "::1" | "[::1]")
    )
}

pub(crate) async fn recover(base_url: &str) -> Result<()> {
    if !is_local_daemon_url(base_url) {
        anyhow::bail!("recovery is only supported for a loopback daemon");
    }
    if service::autostart_off_marker_exists() {
        anyhow::bail!(
            "daemon stopped by `wenlan background off` — run `wenlan background on` to enable it again"
        );
    }
    if !service::is_installed() {
        anyhow::bail!(NO_SERVICE_HINT);
    }

    eprintln!(
        "wenlan: daemon not reachable — starting {}…",
        service::SERVICE_LABEL
    );
    service::start_registered(false).context("start registered daemon service")?;

    let health_url = format!("{base_url}/api/health");
    poll_health(&health_url, Duration::from_secs(10)).await
}

pub(crate) async fn poll_health(url: &str, deadline: Duration) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .context("building daemon recovery client")?;
    let deadline_at = Instant::now() + deadline;
    loop {
        if let Ok(response) = client.get(url).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }
        let now = Instant::now();
        if now >= deadline_at {
            break;
        }
        tokio::time::sleep((deadline_at - now).min(Duration::from_millis(500))).await;
    }

    anyhow::bail!("daemon did not become healthy within {deadline:?} after start");
}

#[cfg(test)]
mod tests {
    use super::{autostart_allowed, is_local_daemon_url, poll_health, NO_SERVICE_HINT};
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn autostart_requires_recovery_and_no_non_empty_opt_out() {
        assert!(autostart_allowed(None, true));
        assert!(autostart_allowed(Some(""), true));
        assert!(!autostart_allowed(Some("1"), true));
        assert!(!autostart_allowed(None, false));
    }

    #[test]
    fn no_service_hint_is_actionable() {
        assert_eq!(
            NO_SERVICE_HINT,
            "daemon not reachable and no background service is registered — run `wenlan background on`"
        );
    }

    #[test]
    fn local_daemon_url_accepts_only_loopback_hosts() {
        assert!(is_local_daemon_url("http://127.0.0.1:7878"));
        assert!(is_local_daemon_url("https://localhost/api/health"));
        assert!(is_local_daemon_url("http://[::1]:7878"));
        assert!(!is_local_daemon_url("http://example.com:7878"));
        assert!(!is_local_daemon_url("http://127.0.0.1.example.com:7878"));
        assert!(!is_local_daemon_url("not a URL"));
    }

    #[tokio::test]
    async fn poll_health_respects_deadline_for_unhealthy_server() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind health stub");
        listener
            .set_nonblocking(true)
            .expect("make health stub nonblocking");
        let url = format!(
            "http://{}/api/health",
            listener.local_addr().expect("health stub address")
        );
        let stop = Arc::new(AtomicBool::new(false));
        let stop_server = Arc::clone(&stop);
        let server = thread::spawn(move || {
            while !stop_server.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream
                            .write_all(
                                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                            )
                            .expect("write unhealthy response");
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("health stub accept failed: {error}"),
                }
            }
        });

        let deadline = Duration::from_secs(1);
        let started = Instant::now();
        let result = poll_health(&url, deadline).await;
        let elapsed = started.elapsed();
        stop.store(true, Ordering::Relaxed);
        server.join().expect("health stub thread");

        assert!(result.is_err(), "503 health responses must not pass");
        assert!(
            elapsed < deadline + Duration::from_secs(1),
            "poll_health exceeded its bound: {elapsed:?}"
        );
    }
}
