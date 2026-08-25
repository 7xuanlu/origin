// SPDX-License-Identifier: Apache-2.0
//! HTTP client for talking to the origin daemon.
//!
//! Mirrors the relevant slice of `app/src/api.rs::WenlanClient` but uses
//! `anyhow::Result` (this is a CLI binary, not a Tauri command surface),
//! and reads `WENLAN_HOST` (full URL) instead of `WENLAN_PORT` so users can
//! point the CLI at a remote daemon over a tunnel.
//!
//! The methods exposed here are the subset the CLI subcommands need:
//! status, ping, search, recall, brief, store, list, agents.

use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use wenlan_types::{
    requests::{
        AddEntityAliasRequest, ListEntitiesRequest, ListMemoriesRequest, MergeEntityRequest,
        SearchMemoryRequest, SearchRequest, SetDefaultSpaceRequest, StoreMemoryRequest,
        UpdateAgentRequest,
    },
    responses::{
        AgentResponse, DefaultSpaceResponse, EntityAliasesResponse, HealthResponse,
        ListEntitiesResponse, ListMemoriesResponse, MemoryDetailResponse, MergeEntityResponse,
        PendingRevisionItem, RevisionAcceptResponse, RevisionDismissResponse, SearchMemoryResponse,
        SearchResponse, StoreMemoryResponse,
    },
    sources::Source,
    BriefReadRequest, BriefReadResponse, BriefUpdateReceipt, BriefUpdateRequest, EntityDetail,
    OutboxDrainReport,
};

mod lint;
pub(crate) mod recovery;
pub use lint::origin_host_from_env;

const DEFAULT_HOST: &str = "http://127.0.0.1:7878";

/// Local mirror of the daemon's `SyncStatsResponse` (defined in `wenlan-server`,
/// which the CLI must not depend on). Typed so envelope drift fails loud rather
/// than silently deserializing into `serde_json::Value`. The trailing optional
/// fields carry `#[serde(default)]` so older/leaner daemon responses parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStats {
    pub files_found: usize,
    pub ingested: usize,
    pub skipped: usize,
    pub errors: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<String>,
}

/// Local mirror of the daemon's `AmbientSweepReport`/`AmbientJobSweepResult`
/// (defined in `wenlan-server`, which the CLI must not depend on). Typed so
/// envelope drift fails loud rather than silently deserializing into
/// `serde_json::Value`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientSweepReport {
    pub phases: Vec<AmbientJobSweepResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientJobSweepResult {
    pub job: String,
    pub attempted: bool,
    pub selected: bool,
    pub llm_calls: usize,
    pub panicked: bool,
    pub elapsed_ms: u128,
}

pub struct WenlanClient {
    base_url: String,
    http: reqwest::Client,
    recovery_enabled: bool,
}

impl WenlanClient {
    /// Create a client using `WENLAN_HOST` env var, or default to `http://127.0.0.1:7878`.
    pub fn from_env() -> Self {
        Self::from_env_with_context(None, None)
            .expect("empty Wenlan client context always builds valid headers")
    }

    /// Create a client that consistently identifies the caller and supplies
    /// client-local Space context through the common request headers.
    pub fn from_env_with_context(agent_name: Option<&str>, space: Option<&str>) -> Result<Self> {
        let base_url = origin_host_from_env();
        let mut headers = HeaderMap::new();
        if let Some(agent_name) = agent_name {
            headers.insert(
                "x-agent-name",
                HeaderValue::from_str(agent_name).context("invalid --agent-name header value")?,
            );
        }
        if let Some(space) = space {
            headers.insert(
                "x-wenlan-space",
                HeaderValue::from_str(space).context("invalid Space header value")?,
            );
        }
        Ok(Self {
            base_url,
            http: reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .context("building Wenlan HTTP client")?,
            recovery_enabled: true,
        })
    }

    /// Enable or disable recovery by starting the registered daemon service
    /// after a connection failure. Environment opt-out still applies when
    /// recovery is enabled here.
    pub fn with_recovery(mut self, enabled: bool) -> Self {
        self.recovery_enabled = enabled;
        self
    }

    /// True when this client's base URL is a loopback daemon. Callers must
    /// gate offline fallbacks (outbox queueing, offline Space resolution) on
    /// this — a connect failure against a remote host is a real error, never
    /// grounds to queue locally or guess a Space.
    pub fn is_local(&self) -> bool {
        recovery::is_local_daemon_url(&self.base_url)
    }

    async fn send(&self, req: reqwest::RequestBuilder, what: &str) -> Result<reqwest::Response> {
        let retry = if recovery::autostart_allowed_from_env(self.recovery_enabled)
            && recovery::is_local_daemon_url(&self.base_url)
        {
            req.try_clone()
        } else {
            None
        };
        let response = req.send().await;
        match response {
            Ok(response) => Ok(response),
            Err(error) if error.is_connect() => {
                let Some(retry) = retry else {
                    return Err(error)
                        .with_context(|| what.to_owned())
                        .with_context(|| recovery::connect_failure_hint(&self.base_url));
                };
                let original = anyhow::Error::new(error).context(what.to_owned());
                if let Err(recovery_error) = recovery::recover(&self.base_url).await {
                    return Err(original.context(format!("{recovery_error:#}")).context(
                        "the daemon did not come up on its own — run `wenlan doctor` to see why",
                    ));
                }
                retry
                    .send()
                    .await
                    .with_context(|| what.to_owned())
                    .with_context(|| {
                        format!(
                            "the daemon was started but {} still does not answer — run `wenlan doctor`",
                            self.base_url
                        )
                    })
            }
            Err(error) => Err(error).with_context(|| what.to_owned()),
        }
    }

    // ===== Generic request helpers =====
    //
    // Every endpoint below is build-url / send / error_for_status / deserialize,
    // differing only in verb, path, request body, and response type. These
    // helpers hold that sequence once; each public method is a path expression
    // plus one delegating call. `hint_daemon_running` on `get_json` preserves
    // the "(is the daemon running?)" hint that health/list_sources/
    // list_pending_revisions/get_memory_detail carried before this existed —
    // the other GETs did not have it and still don't. The `.context("parsing
    // ... response")` text below is now generic ("parsing {path} response")
    // rather than each endpoint's previous literal wording; nothing in
    // crates/wenlan-cli/tests asserts the old text.

    /// GET `path` (e.g. "/api/health") and deserialize the JSON response.
    async fn get_json<R: DeserializeOwned>(
        &self,
        path: &str,
        hint_daemon_running: bool,
    ) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        let what = if hint_daemon_running {
            format!("GET {} failed (is the daemon running?)", url)
        } else {
            format!("GET {} failed", url)
        };
        let resp = self.send(self.http.get(&url), &what).await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .with_context(|| format!("parsing {} response", path))
    }

    /// POST `path` with a JSON `body` and deserialize the JSON response.
    async fn post_json<B: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .send(
                self.http.post(&url).json(body),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .with_context(|| format!("parsing {} response", path))
    }

    /// POST `path` with no body and deserialize the JSON response.
    async fn post_empty_json<R: DeserializeOwned>(&self, path: &str) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .send(self.http.post(&url), &format!("POST {} failed", url))
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .with_context(|| format!("parsing {} response", path))
    }

    /// PUT `path` with a JSON `body` and deserialize the JSON response.
    async fn put_json<B: Serialize, R: DeserializeOwned>(&self, path: &str, body: &B) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .send(
                self.http.put(&url).json(body),
                &format!("PUT {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .with_context(|| format!("parsing {} response", path))
    }

    /// PATCH `path` with a JSON `body` and deserialize the JSON response.
    async fn patch_json<B: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .send(
                self.http.patch(&url).json(body),
                &format!("PATCH {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .with_context(|| format!("parsing {} response", path))
    }

    /// POST `path` with a JSON `body`, treating any 2xx as success and
    /// discarding the response body.
    async fn post_ok<B: Serialize>(&self, path: &str, body: &B) -> Result<()> {
        let url = format!("{}{}", self.base_url, path);
        self.send(
            self.http.post(&url).json(body),
            &format!("POST {} failed", url),
        )
        .await?
        .error_for_status()
        .with_context(|| format!("daemon returned error for {}", url))?;
        Ok(())
    }

    /// DELETE `path`, treating any 2xx as success and discarding the response body.
    async fn delete_ok(&self, path: &str) -> Result<()> {
        let url = format!("{}{}", self.base_url, path);
        self.send(self.http.delete(&url), &format!("DELETE {} failed", url))
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        Ok(())
    }

    /// GET /api/health — daemon liveness + version.
    pub async fn health(&self) -> Result<HealthResponse> {
        self.get_json("/api/health", true).await
    }

    /// POST /api/search — hybrid memory search.
    pub async fn search(&self, query: String, limit: usize) -> Result<SearchResponse> {
        let req = SearchRequest {
            query,
            limit,
            source_filter: None,
            space: None,
        };
        self.post_json("/api/search", &req).await
    }

    /// POST /api/memory/search — semantic recall for one query.
    pub async fn recall(&self, query: String, limit: usize) -> Result<SearchMemoryResponse> {
        let request = SearchMemoryRequest {
            query,
            limit,
            memory_type: None,
            space: None,
            source_agent: None,
            rerank: false,
        };
        self.post_json("/api/memory/search", &request).await
    }

    /// POST /api/brief — complete Brief plus optional same-Space recall.
    pub async fn brief(&self, request: &BriefReadRequest) -> Result<BriefReadResponse> {
        self.post_json("/api/brief", request).await
    }

    /// PATCH /api/brief — item-level handoff deltas.
    pub async fn update_brief(&self, request: &BriefUpdateRequest) -> Result<BriefUpdateReceipt> {
        self.patch_json("/api/brief", request).await
    }

    /// POST /api/memory/store — write a memory.
    pub async fn store(
        &self,
        content: String,
        memory_type: Option<String>,
    ) -> Result<StoreMemoryResponse> {
        let req = Self::store_request(content, memory_type);
        self.post_json("/api/memory/store", &req).await
    }

    pub fn store_request(content: String, memory_type: Option<String>) -> StoreMemoryRequest {
        StoreMemoryRequest {
            content,
            memory_type,
            space: (None).into(),
            source_agent: None,
            title: None,
            confidence: None,
            supersedes: None,
            entity: None,
            entity_id: None,
            structured_fields: None,
            retrieval_cue: None,
        }
    }

    /// POST /api/outbox/drain — ask the daemon to replay queued envelopes.
    pub async fn drain_outbox(&self) -> Result<OutboxDrainReport> {
        self.post_empty_json("/api/outbox/drain").await
    }

    /// POST /api/ambient/sweep — force one bounded pass over every ambient
    /// job type, bypassing the idle/resource gate (each job's own per-call
    /// slice bound is unchanged). Document and Citation phases can call the
    /// on-device LLM, so a full lap can legitimately take minutes; this
    /// request gets its own generous timeout rather than the client's
    /// default, which is sized for ordinary sub-second daemon calls.
    pub async fn sweep_ambient(&self) -> Result<AmbientSweepReport> {
        const SWEEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);
        let url = format!("{}/api/ambient/sweep", self.base_url);
        let resp = self
            .send(
                self.http.post(&url).timeout(SWEEP_TIMEOUT),
                &format!("POST {url} failed"),
            )
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {url}"))?;
        resp.json()
            .await
            .context("parsing /api/ambient/sweep response")
    }

    /// GET /api/sources — list registered sources.
    pub async fn list_sources(&self) -> Result<Vec<Source>> {
        self.get_json("/api/sources", true).await
    }

    /// POST /api/sources — register a source. Returns the created `Source`.
    pub async fn add_source(&self, source_type: &str, path: &str) -> Result<Source> {
        let req = serde_json::json!({ "source_type": source_type, "path": path });
        self.post_json("/api/sources", &req).await
    }

    /// POST /api/sources/{id}/sync — sync a registered source, returning stats.
    pub async fn sync_source(&self, id: &str) -> Result<SyncStats> {
        self.post_empty_json(&format!("/api/sources/{}/sync", id))
            .await
    }

    /// POST /api/memory/list — list memories with optional filters.
    pub async fn list(
        &self,
        limit: Option<usize>,
        memory_type: Option<String>,
        confirmed: Option<bool>,
    ) -> Result<ListMemoriesResponse> {
        let req = build_list_request(limit, memory_type, confirmed);
        self.post_json("/api/memory/list", &req).await
    }

    /// GET /api/agents — list registered agents.
    pub async fn list_agents(&self) -> Result<Vec<AgentResponse>> {
        self.get_json("/api/agents", false).await
    }

    /// GET /api/agents/{name} — fetch a single agent.
    pub async fn get_agent(&self, name: &str) -> Result<AgentResponse> {
        self.get_json(&format!("/api/agents/{}", name), false).await
    }

    /// POST /api/spaces — register a new space.
    pub async fn create_space(&self, name: &str) -> Result<()> {
        self.post_ok("/api/spaces", &serde_json::json!({"name": name}))
            .await
    }

    /// POST /api/spaces/{from}/move-to/{to} — bulk reassign memories from one space to another.
    pub async fn move_space(&self, from: &str, to: &str) -> Result<usize> {
        let url = format!("{}/api/spaces/{}/move-to/{}", self.base_url, from, to);
        let resp = self
            .send(self.http.post(&url), &format!("POST {} failed", url))
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        let json: serde_json::Value = resp.json().await.context("parsing move-to response")?;
        Ok(json["affected"].as_u64().unwrap_or(0) as usize)
    }

    /// GET /api/spaces — fetch a single space by name (filters from list).
    pub async fn get_space(&self, name: &str) -> Result<wenlan_types::Space> {
        let spaces = self.list_spaces().await?;
        spaces
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| anyhow::anyhow!("space '{}' not found", name))
    }

    /// GET /api/spaces — list all spaces.
    pub async fn list_spaces(&self) -> Result<Vec<wenlan_types::Space>> {
        self.get_json("/api/spaces", false).await
    }

    /// GET /api/spaces/default — fetch the daemon-owned default save Space.
    pub async fn get_default_space(&self) -> Result<DefaultSpaceResponse> {
        self.get_json("/api/spaces/default", false).await
    }

    /// PUT /api/spaces/default — replace the daemon-owned default save Space.
    pub async fn set_default_space(&self, space_id: String) -> Result<DefaultSpaceResponse> {
        self.put_json("/api/spaces/default", &SetDefaultSpaceRequest { space_id })
            .await
    }

    /// DELETE /api/spaces/default — clear the daemon-owned default save Space.
    pub async fn clear_default_space(&self) -> Result<()> {
        self.delete_ok("/api/spaces/default").await
    }

    /// PUT /api/agents/{name} — update an agent's metadata.
    pub async fn update_agent(&self, name: &str, req: UpdateAgentRequest) -> Result<AgentResponse> {
        self.put_json(&format!("/api/agents/{}", name), &req).await
    }

    /// GET /api/memory/pending-revisions — staged revisions awaiting human accept/dismiss.
    pub async fn list_pending_revisions(&self, limit: usize) -> Result<Vec<PendingRevisionItem>> {
        self.get_json(
            &format!("/api/memory/pending-revisions?limit={}", limit),
            true,
        )
        .await
    }

    /// POST /api/memory/revision/{id}/accept — replace the original with the revision.
    /// `id` is the revision's own `source_id` (the daemon also accepts a target id, legacy).
    pub async fn accept_revision(&self, id: &str) -> Result<RevisionAcceptResponse> {
        self.post_empty_json(&format!("/api/memory/revision/{}/accept", id))
            .await
    }

    /// POST /api/memory/revision/{id}/dismiss — unstage the false revision: keep BOTH it and the original.
    /// `id` is the revision's own `source_id` (the daemon also accepts a target id, legacy).
    pub async fn dismiss_revision(&self, id: &str) -> Result<RevisionDismissResponse> {
        self.post_empty_json(&format!("/api/memory/revision/{}/dismiss", id))
            .await
    }

    /// GET /api/memory/{id}/detail — the assembled (chunks-joined) memory by source_id.
    /// Used by `wenlan curate` to fetch the ORIGINAL a revision would replace, so the
    /// card can show an original->revision diff.
    pub async fn get_memory_detail(&self, source_id: &str) -> Result<MemoryDetailResponse> {
        self.get_json(&format!("/api/memory/{}/detail", source_id), true)
            .await
    }

    /// GET /api/memory/entities/{id} — full entity detail by id. `Ok(None)` on 404
    /// (used by the CLI's id-or-name resolver to fall back to a name search).
    pub async fn get_entity(&self, id: &str) -> Result<Option<EntityDetail>> {
        let url = format!("{}/api/memory/entities/{}", self.base_url, id);
        let resp = self
            .send(
                self.http.get(&url),
                &format!("GET {} failed (is the daemon running?)", url),
            )
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .map(Some)
            .context("parsing /api/memory/entities/{id} response")
    }

    /// POST /api/memory/entities/list — every live entity in scope, unfiltered
    /// by name. Used for exact-name lookup: the daemon has no by-name filter
    /// on this route, so callers filter the returned list client-side.
    pub async fn list_entities(&self) -> Result<ListEntitiesResponse> {
        let url = format!("{}/api/memory/entities/list", self.base_url);
        let req = ListEntitiesRequest {
            entity_type: None,
            space: None,
        };
        let resp = self
            .send(
                self.http.post(&url).json(&req),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = ensure_daemon_success(resp, &url).await?;
        resp.json()
            .await
            .context("parsing /api/memory/entities/list response")
    }

    /// POST /api/memory/entities/{id}/merge — merge `loser_id` into `into`.
    pub async fn merge_entity(
        &self,
        loser_id: &str,
        into: String,
        dry_run: bool,
    ) -> Result<MergeEntityResponse> {
        let url = format!("{}/api/memory/entities/{}/merge", self.base_url, loser_id);
        let req = MergeEntityRequest { into, dry_run };
        let resp = self
            .send(
                self.http.post(&url).json(&req),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = ensure_daemon_success(resp, &url).await?;
        resp.json()
            .await
            .context("parsing /api/memory/entities/{id}/merge response")
    }

    /// POST /api/memory/entities/{id}/aliases — declare `alias` as an additional name.
    pub async fn add_entity_alias(&self, id: &str, alias: String) -> Result<EntityAliasesResponse> {
        let url = format!("{}/api/memory/entities/{}/aliases", self.base_url, id);
        let req = AddEntityAliasRequest { alias };
        let resp = self
            .send(
                self.http.post(&url).json(&req),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = ensure_daemon_success(resp, &url).await?;
        resp.json()
            .await
            .context("parsing /api/memory/entities/{id}/aliases response")
    }
}

/// Non-success response -> `anyhow::Error` carrying the daemon's own error
/// message, not just the status line `error_for_status()` alone gives.
/// Used only by `list_entities`, `merge_entity`, `add_entity_alias`; every
/// other client method keeps `error_for_status()` as-is.
async fn ensure_daemon_success(resp: reqwest::Response, url: &str) -> Result<reqwest::Response> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    let msg = daemon_error_message(&body);
    anyhow::bail!("daemon returned {status} for {url}: {msg}");
}

/// Pull the daemon's message out of an error response body: the `error`
/// string when the body is a `{"error": "..."}` JSON object (the shape
/// every `ServerError` response uses), else the raw body trimmed.
fn daemon_error_message(body: &str) -> String {
    #[derive(Deserialize)]
    struct ErrorEnvelope {
        error: String,
    }
    serde_json::from_str::<ErrorEnvelope>(body)
        .map(|envelope| envelope.error)
        .unwrap_or_else(|_| body.trim().to_string())
}

fn build_list_request(
    limit: Option<usize>,
    memory_type: Option<String>,
    confirmed: Option<bool>,
) -> ListMemoriesRequest {
    ListMemoriesRequest {
        memory_type,
        space: None,
        limit: limit.unwrap_or(100),
        confirmed,
    }
}

#[cfg(test)]
mod tests {
    use super::{build_list_request, daemon_error_message, WenlanClient};

    #[test]
    fn daemon_error_message_reads_error_envelope() {
        let body =
            r#"{"error":"alias \"origin-core\" is the name of live entity e1; use merge instead"}"#;
        assert_eq!(
            daemon_error_message(body),
            "alias \"origin-core\" is the name of live entity e1; use merge instead"
        );
    }

    #[test]
    fn daemon_error_message_falls_back_to_raw_text() {
        assert_eq!(
            daemon_error_message("  Internal Server Error  "),
            "Internal Server Error"
        );
    }

    #[test]
    fn list_request_can_filter_unconfirmed_memories() {
        let request = build_list_request(Some(20), None, Some(false));
        let json = serde_json::to_value(request).expect("serialize list request");
        assert_eq!(json["confirmed"], false);
    }

    #[test]
    fn list_request_omits_confirmation_filter_by_default() {
        let request = build_list_request(Some(20), None, None);
        let json = serde_json::to_value(request).expect("serialize list request");
        assert!(json.get("confirmed").is_none());
    }

    #[tokio::test]
    async fn connect_failure_says_what_to_do_next() {
        // An ephemeral port that was just released: nothing listens there.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        drop(listener);
        let base = format!("http://{addr}");
        let client = client_for(&base).with_recovery(false);
        let error = tokio::time::timeout(std::time::Duration::from_secs(10), client.health())
            .await
            .expect("connect failure within 10s")
            .expect_err("nothing listens on the released port");
        let text = format!("{error:#}");
        assert!(
            text.contains(&format!("no Wenlan daemon is listening at {base}")),
            "{text}"
        );
        assert!(
            text.contains(&format!("GET {base}/api/health failed")),
            "{text}"
        );
    }

    fn client_for(base_url: &str) -> WenlanClient {
        WenlanClient {
            base_url: base_url.to_string(),
            http: reqwest::Client::new(),
            recovery_enabled: true,
        }
    }

    // F1 regression: a client pointed at a remote host must never be treated
    // as merely offline. The integration-level version of this ("capture
    // against an unreachable *remote* WENLAN_HOST must never queue") is not
    // exercised end-to-end here because the sandboxed test environment
    // denies egress to arbitrary hosts (including a nonexistent
    // `*.invalid` one), so a live connection attempt cannot be made to fail
    // fast and deterministically the way `reqwest::Error::is_connect()`
    // would need. `is_local()` is exactly the gate `store`/`brief`/`main`
    // check before queueing, so pinning its behavior on both loopback and
    // remote URLs covers the same guarantee without depending on sandbox or
    // real-world DNS timing.
    #[test]
    fn is_local_accepts_only_loopback_hosts() {
        assert!(client_for("http://127.0.0.1:7878").is_local());
        assert!(client_for("http://localhost:7878").is_local());
        assert!(client_for("http://[::1]:7878").is_local());
        assert!(!client_for("http://wenlan-outbox-nonlocal.invalid:7878").is_local());
        assert!(!client_for("http://example.com:7878").is_local());
    }
}
