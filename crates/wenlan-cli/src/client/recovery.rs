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

pub(crate) async fn recover(base_url: &str) -> Result<()> {
    if !service::is_installed() {
        anyhow::bail!(NO_SERVICE_HINT);
    }

    eprintln!(
        "wenlan: daemon not reachable — starting {}…",
        service::SERVICE_LABEL
    );
    service::start_registered().context("start registered daemon service")?;

    let health_url = format!("{base_url}/api/health");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .context("building daemon recovery client")?;
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(response) = client.get(&health_url).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        tokio::time::sleep((deadline - now).min(Duration::from_millis(500))).await;
    }

    anyhow::bail!("daemon did not become healthy within 10s after start");
}

#[cfg(test)]
mod tests {
    use super::{autostart_allowed, NO_SERVICE_HINT};

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
}
