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
use serde::{Deserialize, Serialize};
use wenlan_types::{
    requests::{
        AddEntityAliasRequest, ListMemoriesRequest, MergeEntityRequest, SearchEntitiesRequest,
        SearchMemoryRequest, SearchRequest, SetDefaultSpaceRequest, StoreMemoryRequest,
        UpdateAgentRequest,
    },
    responses::{
        AgentResponse, DefaultSpaceResponse, EntityAliasesResponse, HealthResponse,
        ListMemoriesResponse, MemoryDetailResponse, MergeEntityResponse, PendingRevisionItem,
        RevisionAcceptResponse, RevisionDismissResponse, SearchEntitiesResponse,
        SearchMemoryResponse, SearchResponse, StoreMemoryResponse,
    },
    sources::Source,
    BriefReadRequest, BriefReadResponse, BriefUpdateReceipt, BriefUpdateRequest, EntityDetail,
    OutboxDrainReport,
};

mod lint;
mod recovery;
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
                    return Err(error).with_context(|| what.to_owned());
                };
                let original = anyhow::Error::new(error).context(what.to_owned());
                if let Err(recovery_error) = recovery::recover(&self.base_url).await {
                    return Err(original.context(format!("{recovery_error:#}")));
                }
                retry.send().await.with_context(|| what.to_owned())
            }
            Err(error) => Err(error).with_context(|| what.to_owned()),
        }
    }

    /// GET /api/health — daemon liveness + version.
    pub async fn health(&self) -> Result<HealthResponse> {
        let url = format!("{}/api/health", self.base_url);
        let resp = self
            .send(
                self.http.get(&url),
                &format!("GET {} failed (is the daemon running?)", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json().await.context("parsing /api/health response")
    }

    /// POST /api/search — hybrid memory search.
    pub async fn search(&self, query: String, limit: usize) -> Result<SearchResponse> {
        let url = format!("{}/api/search", self.base_url);
        let req = SearchRequest {
            query,
            limit,
            source_filter: None,
            space: None,
        };
        let resp = self
            .send(
                self.http.post(&url).json(&req),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json().await.context("parsing /api/search response")
    }

    /// POST /api/memory/search — semantic recall for one query.
    pub async fn recall(&self, query: String, limit: usize) -> Result<SearchMemoryResponse> {
        let url = format!("{}/api/memory/search", self.base_url);
        let request = SearchMemoryRequest {
            query,
            limit,
            memory_type: None,
            space: None,
            source_agent: None,
            rerank: false,
        };
        let response = self
            .send(
                self.http.post(&url).json(&request),
                &format!("POST {url} failed"),
            )
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {url}"))?;
        response
            .json()
            .await
            .context("parsing /api/memory/search response")
    }

    /// POST /api/brief — complete Brief plus optional same-Space recall.
    pub async fn brief(&self, request: &BriefReadRequest) -> Result<BriefReadResponse> {
        let url = format!("{}/api/brief", self.base_url);
        let response = self
            .send(
                self.http.post(&url).json(request),
                &format!("POST {url} failed"),
            )
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {url}"))?;
        response.json().await.context("parsing /api/brief response")
    }

    /// PATCH /api/brief — item-level handoff deltas.
    pub async fn update_brief(&self, request: &BriefUpdateRequest) -> Result<BriefUpdateReceipt> {
        let url = format!("{}/api/brief", self.base_url);
        let response = self
            .send(
                self.http.patch(&url).json(request),
                &format!("PATCH {url} failed"),
            )
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {url}"))?;
        response
            .json()
            .await
            .context("parsing PATCH /api/brief response")
    }

    /// POST /api/memory/store — write a memory.
    pub async fn store(
        &self,
        content: String,
        memory_type: Option<String>,
    ) -> Result<StoreMemoryResponse> {
        let url = format!("{}/api/memory/store", self.base_url);
        let req = Self::store_request(content, memory_type);
        let resp = self
            .send(
                self.http.post(&url).json(&req),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/memory/store response")
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
        let url = format!("{}/api/outbox/drain", self.base_url);
        let response = self
            .send(self.http.post(&url), &format!("POST {url} failed"))
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {url}"))?;
        response
            .json()
            .await
            .context("parsing /api/outbox/drain response")
    }

    /// GET /api/sources — list registered sources.
    pub async fn list_sources(&self) -> Result<Vec<Source>> {
        let url = format!("{}/api/sources", self.base_url);
        let resp = self
            .send(
                self.http.get(&url),
                &format!("GET {} failed (is the daemon running?)", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json().await.context("parsing /api/sources response")
    }

    /// POST /api/sources — register a source. Returns the created `Source`.
    pub async fn add_source(&self, source_type: &str, path: &str) -> Result<Source> {
        let url = format!("{}/api/sources", self.base_url);
        let req = serde_json::json!({ "source_type": source_type, "path": path });
        let resp = self
            .send(
                self.http.post(&url).json(&req),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing POST /api/sources response")
    }

    /// POST /api/sources/{id}/sync — sync a registered source, returning stats.
    pub async fn sync_source(&self, id: &str) -> Result<SyncStats> {
        let url = format!("{}/api/sources/{}/sync", self.base_url, id);
        let resp = self
            .send(self.http.post(&url), &format!("POST {} failed", url))
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/sources/{id}/sync response")
    }

    /// POST /api/memory/list — list memories with optional filters.
    pub async fn list(
        &self,
        limit: Option<usize>,
        memory_type: Option<String>,
        confirmed: Option<bool>,
    ) -> Result<ListMemoriesResponse> {
        let url = format!("{}/api/memory/list", self.base_url);
        let req = build_list_request(limit, memory_type, confirmed);
        let resp = self
            .send(
                self.http.post(&url).json(&req),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/memory/list response")
    }

    /// GET /api/agents — list registered agents.
    pub async fn list_agents(&self) -> Result<Vec<AgentResponse>> {
        let url = format!("{}/api/agents", self.base_url);
        let resp = self
            .send(self.http.get(&url), &format!("GET {} failed", url))
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json().await.context("parsing /api/agents response")
    }

    /// GET /api/agents/{name} — fetch a single agent.
    pub async fn get_agent(&self, name: &str) -> Result<AgentResponse> {
        let url = format!("{}/api/agents/{}", self.base_url, name);
        let resp = self
            .send(self.http.get(&url), &format!("GET {} failed", url))
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/agents/{name} response")
    }

    /// POST /api/spaces — register a new space.
    pub async fn create_space(&self, name: &str) -> Result<()> {
        let url = format!("{}/api/spaces", self.base_url);
        self.send(
            self.http
                .post(&url)
                .json(&serde_json::json!({"name": name})),
            &format!("POST {} failed", url),
        )
        .await?
        .error_for_status()
        .with_context(|| format!("daemon returned error for {}", url))?;
        Ok(())
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
        let url = format!("{}/api/spaces", self.base_url);
        let resp = self
            .send(self.http.get(&url), &format!("GET {} failed", url))
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json().await.context("parsing /api/spaces response")
    }

    /// GET /api/spaces/default — fetch the daemon-owned default save Space.
    pub async fn get_default_space(&self) -> Result<DefaultSpaceResponse> {
        let url = format!("{}/api/spaces/default", self.base_url);
        let resp = self
            .send(self.http.get(&url), &format!("GET {} failed", url))
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/spaces/default response")
    }

    /// PUT /api/spaces/default — replace the daemon-owned default save Space.
    pub async fn set_default_space(&self, space_id: String) -> Result<DefaultSpaceResponse> {
        let url = format!("{}/api/spaces/default", self.base_url);
        let resp = self
            .send(
                self.http
                    .put(&url)
                    .json(&SetDefaultSpaceRequest { space_id }),
                &format!("PUT {} failed", url),
            )
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/spaces/default response")
    }

    /// DELETE /api/spaces/default — clear the daemon-owned default save Space.
    pub async fn clear_default_space(&self) -> Result<()> {
        let url = format!("{}/api/spaces/default", self.base_url);
        self.send(self.http.delete(&url), &format!("DELETE {} failed", url))
            .await?
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        Ok(())
    }

    /// PUT /api/agents/{name} — update an agent's metadata.
    pub async fn update_agent(&self, name: &str, req: UpdateAgentRequest) -> Result<AgentResponse> {
        let url = format!("{}/api/agents/{}", self.base_url, name);
        let resp = self
            .send(
                self.http.put(&url).json(&req),
                &format!("PUT {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/agents/{name} update response")
    }

    /// GET /api/memory/pending-revisions — staged revisions awaiting human accept/dismiss.
    pub async fn list_pending_revisions(&self, limit: usize) -> Result<Vec<PendingRevisionItem>> {
        let url = format!(
            "{}/api/memory/pending-revisions?limit={}",
            self.base_url, limit
        );
        let resp = self
            .send(
                self.http.get(&url),
                &format!("GET {} failed (is the daemon running?)", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/memory/pending-revisions response")
    }

    /// POST /api/memory/revision/{id}/accept — replace the original with the revision.
    /// `id` is the revision's own `source_id` (the daemon also accepts a target id, legacy).
    pub async fn accept_revision(&self, id: &str) -> Result<RevisionAcceptResponse> {
        let url = format!("{}/api/memory/revision/{}/accept", self.base_url, id);
        let resp = self
            .send(self.http.post(&url), &format!("POST {} failed", url))
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing accept-revision response")
    }

    /// POST /api/memory/revision/{id}/dismiss — unstage the false revision: keep BOTH it and the original.
    /// `id` is the revision's own `source_id` (the daemon also accepts a target id, legacy).
    pub async fn dismiss_revision(&self, id: &str) -> Result<RevisionDismissResponse> {
        let url = format!("{}/api/memory/revision/{}/dismiss", self.base_url, id);
        let resp = self
            .send(self.http.post(&url), &format!("POST {} failed", url))
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing dismiss-revision response")
    }

    /// GET /api/memory/{id}/detail — the assembled (chunks-joined) memory by source_id.
    /// Used by `wenlan curate` to fetch the ORIGINAL a revision would replace, so the
    /// card can show an original->revision diff.
    pub async fn get_memory_detail(&self, source_id: &str) -> Result<MemoryDetailResponse> {
        let url = format!("{}/api/memory/{}/detail", self.base_url, source_id);
        let resp = self
            .send(
                self.http.get(&url),
                &format!("GET {} failed (is the daemon running?)", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/memory/{id}/detail response")
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

    /// POST /api/memory/entities/search — vector similarity search over entity names.
    pub async fn search_entities(
        &self,
        query: String,
        limit: usize,
    ) -> Result<SearchEntitiesResponse> {
        let url = format!("{}/api/memory/entities/search", self.base_url);
        let req = SearchEntitiesRequest { query, limit };
        let resp = self
            .send(
                self.http.post(&url).json(&req),
                &format!("POST {} failed", url),
            )
            .await?;
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/memory/entities/search response")
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
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
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
        let resp = resp
            .error_for_status()
            .with_context(|| format!("daemon returned error for {}", url))?;
        resp.json()
            .await
            .context("parsing /api/memory/entities/{id}/aliases response")
    }
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
    use super::{build_list_request, WenlanClient};

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
