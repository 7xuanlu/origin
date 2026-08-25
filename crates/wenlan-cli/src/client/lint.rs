// SPDX-License-Identifier: Apache-2.0
use super::{WenlanClient, DEFAULT_HOST};
use anyhow::{bail, Context, Result};
use wenlan_types::lint::{
    LintAgentSubmission, LintErrorResponse, LintProfile, LintQuery, LintReport, LintRequestQuery,
};

const MAX_LINT_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

pub fn origin_host_from_env() -> String {
    normalize_origin_host(
        &std::env::var("WENLAN_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string()),
    )
}

/// `WENLAN_HOST` is a full URL, but the first thing people try is a bare
/// port (`17917`) or `host:port`; reqwest then fails with "relative URL
/// without a base", which the CLI used to report as "is the daemon
/// running?". Complete the obvious shorthand instead: digits become a
/// loopback URL, a schemeless host gets `http://`, and an empty value means
/// the default. A value with a scheme is only trimmed. A loopback shorthand
/// takes part in autostart exactly like the full loopback URL would; digits
/// that are not a valid non-zero port are left alone so the error names them.
pub fn normalize_origin_host(raw: &str) -> String {
    let value = raw.trim().trim_end_matches('/');
    if value.is_empty() {
        return DEFAULT_HOST.to_string();
    }
    if value.contains("://") {
        return value.to_string();
    }
    if value.bytes().all(|b| b.is_ascii_digit()) {
        return match value.parse::<u16>() {
            Ok(port) if port > 0 => format!("http://127.0.0.1:{port}"),
            _ => value.to_string(),
        };
    }
    format!("http://{value}")
}

impl WenlanClient {
    pub async fn lint(
        &self,
        profile: Option<LintProfile>,
        space: Option<String>,
        external_egress: bool,
        agent_assist: bool,
        submission: Option<&LintAgentSubmission>,
    ) -> Result<LintReport> {
        let url = format!("{}/api/lint", self.base_url);
        let query = LintRequestQuery::new(
            LintQuery { profile, space },
            external_egress,
            agent_assist || submission.is_some(),
        );
        let request = match submission {
            Some(submission) => self.http.post(&url).query(&query).json(submission),
            None => self.http.get(&url).query(&query),
        };
        let response = self
            .send(
                request,
                &format!(
                    "{} {url} failed",
                    if submission.is_some() { "POST" } else { "GET" }
                ),
            )
            .await?;
        let status = response.status();
        let body = read_lint_body(response, &url).await?;
        if !status.is_success() {
            if status == reqwest::StatusCode::UNPROCESSABLE_ENTITY {
                if let Ok(error) = serde_json::from_slice::<LintErrorResponse>(&body) {
                    if matches!(
                        error.error(),
                        "invalid_scope"
                            | "external_egress_requires_deep"
                            | "agent_assist_requires_deep"
                            | "agent_assist_required_for_submission"
                    ) {
                        bail!(error.error().to_string());
                    }
                }
            }
            bail!("daemon returned HTTP {status} for {url}");
        }
        serde_json::from_slice(&body).context("parsing /api/lint response")
    }
}

async fn read_lint_body(mut response: reqwest::Response, url: &str) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_LINT_RESPONSE_BYTES as u64)
    {
        bail!("lint response exceeds {MAX_LINT_RESPONSE_BYTES} bytes");
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .with_context(|| format!("reading daemon response for {url}"))?
    {
        if body.len().saturating_add(chunk.len()) > MAX_LINT_RESPONSE_BYTES {
            bail!("lint response exceeds {MAX_LINT_RESPONSE_BYTES} bytes");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[cfg(test)]
mod origin_host_tests {
    use super::{normalize_origin_host, DEFAULT_HOST};

    #[test]
    fn shorthand_hosts_become_full_urls() {
        assert_eq!(normalize_origin_host("17917"), "http://127.0.0.1:17917");
        assert_eq!(
            normalize_origin_host("127.0.0.1:17917"),
            "http://127.0.0.1:17917"
        );
        assert_eq!(
            normalize_origin_host("localhost:7878/"),
            "http://localhost:7878"
        );
        assert_eq!(normalize_origin_host(""), DEFAULT_HOST);
        assert_eq!(normalize_origin_host("0"), "0");
        assert_eq!(normalize_origin_host("70000"), "70000");
        assert_eq!(
            normalize_origin_host("http://127.0.0.1:17917/"),
            "http://127.0.0.1:17917"
        );
        assert_eq!(
            normalize_origin_host("https://wenlan.example:8443"),
            "https://wenlan.example:8443"
        );
    }
}
