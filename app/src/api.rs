// SPDX-License-Identifier: AGPL-3.0-only
//! HTTP client for the Wenlan daemon (wenlan-server).
//!
//! Thin wrapper around `reqwest::Client` that maps each daemon endpoint
//! to a typed method. The Tauri app uses this instead of direct DB access.

use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::time::Duration;
use wenlan_types::responses::HealthResponse;

/// A wedged daemon (accept backlog full) otherwise leaves connects hanging;
/// reqwest's default client has no timeouts at all.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Generous total backstop: a single ingest of a book-sized document
/// legitimately chunks + embeds synchronously before responding.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// The startup health loop polls a daemon that may accept connections but
/// never respond (wedged, or bound-but-still-initializing); the probe must
/// not inherit the ingest-sized backstop above.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(5);

fn build_http_client(connect_timeout: Duration, request_timeout: Duration) -> Client {
    Client::builder()
        .connect_timeout(connect_timeout)
        .timeout(request_timeout)
        .build()
        .expect("static reqwest client config cannot fail")
}

/// HTTP client that proxies requests to the Wenlan daemon.
#[derive(Clone)]
pub struct WenlanClient {
    client: Client,
    base_url: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SetupStatusResponse {
    pub setup_completed: bool,
    pub mode: String,
    pub anthropic_key_configured: bool,
    pub local_model_selected: Option<String>,
    pub local_model_loaded: Option<String>,
    pub local_model_cached: bool,
}

#[derive(Debug, Clone, serde::Deserialize, PartialEq, Eq)]
struct CaptureStatsResponse {
    total_chunks: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineStatusResponse {
    pub enrichment: BTreeMap<String, u64>,
    pub entity_linking: PipelineEntityLinkingStatus,
    pub refinement_queue: Vec<PipelineQueueEntry>,
    pub recaps: u64,
    pub types: BTreeMap<String, u64>,
    pub quality: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineEntityLinkingStatus {
    pub linked: u64,
    pub unlinked: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipelineQueueEntry {
    pub action: String,
    pub status: String,
    pub count: u64,
}

#[derive(Debug, Clone, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct TagInventoryResponse {
    pub(crate) tags: Vec<String>,
    #[serde(default)]
    pub(crate) document_tags: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq)]
pub struct OnDeviceModelEntry {
    pub id: String,
    pub display_name: String,
    pub param_count: String,
    pub ram_required_gb: f64,
    pub file_size_gb: f64,
    pub cached: bool,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq)]
pub struct OnDeviceModelResponse {
    pub loaded: Option<String>,
    pub selected: Option<String>,
    pub models: Vec<OnDeviceModelEntry>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
pub struct MoveSpaceResponse {
    pub affected: usize,
}

// ── Resolved routing (daemon ≥ PR #357; GET /api/config/routing) ─────────
// Local DTOs, NOT bumped into wenlan-types (the endpoint is unreleased). Shapes
// mirror `wenlan-server::config_routes` exactly; no `deny_unknown_fields` so a
// newer daemon can add fields without breaking this old app. Never carries key
// material.

/// Resolved route for one job class: which source serves it, the model, and how
/// it was chosen. `mode`: "pinned" | "pinned_degraded" | "auto". `source`
/// (everyday): "anthropic"|"external"|"on_device"|"basic"; (synthesis):
/// "anthropic"|"external"|"on_device"|"none". `pin`: the raw configured source
/// pin, or `None` when unpinned — distinct from `source` (the RESOLVED
/// source): on a `pinned_degraded` result the two differ, letting the app say
/// "Pinned to X — using Y". `#[serde(default)]` so a daemon that predates the
/// field (pre-#357) still deserializes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobRoute {
    pub source: String,
    pub model: Option<String>,
    pub mode: String,
    #[serde(default)]
    pub pin: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicPool {
    pub configured: bool,
    pub everyday_model: Option<String>,
    pub synthesis_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalPool {
    pub endpoint: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OnDevicePool {
    pub selected: Option<String>,
    pub loaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingPool {
    pub anthropic: AnthropicPool,
    pub external: Option<ExternalPool>,
    pub on_device: Option<OnDevicePool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResolvedRouting {
    pub everyday: JobRoute,
    pub synthesis: JobRoute,
    pub pool: RoutingPool,
}

#[derive(Debug, Clone, Serialize)]
struct DistillReviewRequest {}

#[derive(Debug, Clone, Serialize)]
struct PageRedistillRequest {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistillReviewResponse {
    pub pages_created: usize,
    pub scoped: bool,
    pub created_ids: Vec<String>,
    pub pending: Vec<DistillPendingCluster>,
    pub stale_pages: Vec<DistillStalePage>,
    pub stale_truncated: bool,
    pub orphan_topics: Vec<DistillOrphanTopic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistillPendingCluster {
    pub source_ids: Vec<String>,
    pub contents: Vec<String>,
    pub entity_id: Option<String>,
    pub entity_name: Option<String>,
    pub space: Option<String>,
    pub estimated_tokens: usize,
    pub centroid_embedding: Option<Vec<f32>>,
    pub existing_page_id: Option<String>,
    pub existing_page_title: Option<String>,
    pub new_memory_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DistillStalePage {
    pub page_id: String,
    pub title: String,
    pub summary: Option<String>,
    pub source_memory_ids: Vec<String>,
    pub sources_updated_count: Option<usize>,
    pub stale_reason: Option<String>,
    pub user_edited: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DistillOrphanTopic {
    pub label: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PageRedistillResponse {
    pub status: String,
    pub updated: bool,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct OnDeviceModelRequest {
    model_id: String,
}

impl Default for WenlanClient {
    fn default() -> Self {
        Self::new()
    }
}

impl WenlanClient {
    pub fn new() -> Self {
        let port = daemon_port();
        Self {
            client: build_http_client(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
            base_url: format!("http://127.0.0.1:{}", port),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// The base URL a reachability check runs against — exposed for
    /// `wire_state`, which reports it verbatim in `DaemonWire`.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Points a client at an arbitrary `base_url` instead of the real daemon
    /// port. Test-only: lets `wire_state`'s tests exercise an unreachable
    /// daemon without touching the process-global `WENLAN_PORT`/`ORIGIN_PORT`
    /// env vars `WenlanClient::new()` reads (shared with every other test in
    /// this binary — see `env_lock` below).
    #[cfg(test)]
    pub(crate) fn with_base_url(base_url: String) -> Self {
        Self {
            client: build_http_client(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
            base_url,
        }
    }

    // ── Generic helpers ─────────────────────────────────────────────

    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, String> {
        let resp = self
            .client
            .get(self.url(path))
            .send()
            .await
            .map_err(|e| format!("HTTP GET {}: {}", path, e))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP GET {} returned {}", path, resp.status()));
        }
        resp.json()
            .await
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    pub async fn get_optional_json<T: DeserializeOwned>(
        &self,
        path: &str,
    ) -> Result<Option<T>, String> {
        let resp = self
            .client
            .get(self.url(path))
            .send()
            .await
            .map_err(|e| format!("HTTP GET {}: {}", path, e))?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            return Err(format!("HTTP GET {} returned {}", path, resp.status()));
        }
        resp.json()
            .await
            .map(Some)
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    pub async fn post_json<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        path: &str,
        body: &Req,
    ) -> Result<Resp, String> {
        let resp = self
            .client
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(|e| format!("HTTP POST {}: {}", path, e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP POST {} returned {}: {}", path, status, text));
        }
        resp.json()
            .await
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    pub async fn put_json<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        path: &str,
        body: &Req,
    ) -> Result<Resp, String> {
        let resp = self
            .client
            .put(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(|e| format!("HTTP PUT {}: {}", path, e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP PUT {} returned {}: {}", path, status, text));
        }
        resp.json()
            .await
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    pub async fn delete_path<Resp: DeserializeOwned>(&self, path: &str) -> Result<Resp, String> {
        let resp = self
            .client
            .delete(self.url(path))
            .send()
            .await
            .map_err(|e| format!("HTTP DELETE {}: {}", path, e))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP DELETE {} returned {}", path, resp.status()));
        }
        resp.json()
            .await
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    async fn delete_empty(&self, path: &str) -> Result<(), String> {
        let resp = self
            .client
            .delete(self.url(path))
            .send()
            .await
            .map_err(|e| format!("HTTP DELETE {}: {}", path, e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "HTTP DELETE {} returned {}: {}",
                path, status, text
            ));
        }
        Ok(())
    }

    pub async fn delete_json<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        path: &str,
        body: &Req,
    ) -> Result<Resp, String> {
        let resp = self
            .client
            .delete(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(|e| format!("HTTP DELETE {}: {}", path, e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "HTTP DELETE {} returned {}: {}",
                path, status, text
            ));
        }
        resp.json()
            .await
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    pub async fn post_empty<Resp: DeserializeOwned>(&self, path: &str) -> Result<Resp, String> {
        let resp = self
            .client
            .post(self.url(path))
            .send()
            .await
            .map_err(|e| format!("HTTP POST {}: {}", path, e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP POST {} returned {}: {}", path, status, text));
        }
        resp.json()
            .await
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    // ── Health ──────────────────────────────────────────────────────

    pub async fn health(&self) -> Result<HealthResponse, String> {
        let path = "/api/health";
        let resp = self
            .client
            .get(self.url(path))
            .timeout(HEALTH_TIMEOUT)
            .send()
            .await
            .map_err(|e| format!("HTTP GET {}: {}", path, e))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP GET {} returned {}", path, resp.status()));
        }
        resp.json()
            .await
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    pub async fn status(&self) -> Result<wenlan_types::responses::StatusResponse, String> {
        self.get_json("/api/status").await
    }

    // ── Capture stats ────────────────────────────────────────────────

    pub async fn get_capture_stats(&self) -> Result<HashMap<String, u64>, String> {
        let stats: CaptureStatsResponse = self.get_json("/api/capture-stats").await?;
        let mut counts = HashMap::new();
        counts.insert("total".to_string(), stats.total_chunks);
        Ok(counts)
    }

    pub async fn pipeline_status(&self) -> Result<PipelineStatusResponse, String> {
        self.get_json("/api/debug/pipeline").await
    }

    pub async fn list_tags(&self) -> Result<Vec<String>, String> {
        Ok(self.list_tag_inventory().await?.tags)
    }

    pub(crate) async fn list_tag_inventory(&self) -> Result<TagInventoryResponse, String> {
        self.get_json("/api/tags").await
    }

    // ── Ingest ───────────────────────────────────────────────────────

    pub async fn ingest_webpage(
        &self,
        req: wenlan_types::requests::IngestWebpageRequest,
    ) -> Result<wenlan_types::responses::IngestResponse, String> {
        self.post_json("/api/ingest/webpage", &req).await
    }

    pub async fn distill_review(&self) -> Result<DistillReviewResponse, String> {
        self.post_json("/api/distill", &DistillReviewRequest {})
            .await
    }

    pub async fn redistill_page(&self, page_id: &str) -> Result<PageRedistillResponse, String> {
        let path = format!("/api/distill/{}", percent_encode_path_segment(page_id));
        self.post_json(&path, &PageRedistillRequest {}).await
    }

    pub async fn move_space(&self, from: &str, to: &str) -> Result<MoveSpaceResponse, String> {
        let path = format!(
            "/api/spaces/{}/move-to/{}",
            percent_encode_path_segment(from),
            percent_encode_path_segment(to)
        );
        self.post_empty(&path).await
    }

    // ── Chat export import ─────────────────────────────────────────

    pub async fn import_chat_export(
        &self,
        path: &str,
    ) -> Result<wenlan_types::import::ImportChatExportResponse, String> {
        let req = wenlan_types::import::ImportChatExportRequest {
            path: path.to_string(),
        };
        self.post_json("/api/import/chat-export", &req).await
    }

    pub async fn list_pending_imports(
        &self,
    ) -> Result<Vec<wenlan_types::import::PendingImport>, String> {
        self.get_json("/api/import/state").await
    }

    // ── Source registry ──────────────────────────────────────────────

    pub async fn list_sources(&self) -> Result<Vec<wenlan_types::sources::Source>, String> {
        self.get_json("/api/sources").await
    }

    pub async fn add_source(
        &self,
        source_type: String,
        path: String,
    ) -> Result<wenlan_types::sources::Source, String> {
        let req = wenlan_types::requests::AddSourceRequest { source_type, path };
        self.post_json("/api/sources", &req).await
    }

    pub async fn remove_source(&self, id: &str) -> Result<(), String> {
        let path = format!("/api/sources/{}", id);
        self.delete_empty(&path).await
    }

    pub async fn sync_source(
        &self,
        id: &str,
    ) -> Result<wenlan_types::responses::SyncStatsResponse, String> {
        let path = format!("/api/sources/{}/sync", id);
        self.post_empty(&path).await
    }

    // ── Onboarding milestones ──────────────────────────────────────

    pub async fn list_onboarding_milestones(
        &self,
    ) -> Result<Vec<wenlan_types::onboarding::MilestoneRecord>, String> {
        self.get_json("/api/onboarding/milestones").await
    }

    pub async fn acknowledge_onboarding_milestone(&self, id: &str) -> Result<(), String> {
        let path = format!("/api/onboarding/milestones/{}/acknowledge", id);
        let resp = self
            .client
            .post(self.url(&path))
            .send()
            .await
            .map_err(|e| format!("HTTP POST {}: {}", path, e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP POST {} returned {}: {}", path, status, text));
        }
        Ok(())
    }

    pub async fn reset_onboarding_milestones(&self) -> Result<(), String> {
        let path = "/api/onboarding/reset";
        let resp = self
            .client
            .post(self.url(path))
            .send()
            .await
            .map_err(|e| format!("HTTP POST {}: {}", path, e))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("HTTP POST {} returned {}: {}", path, status, text));
        }
        Ok(())
    }

    // ── Home delta feed ────────────────────────────────────────────────

    pub async fn list_recent_retrievals(
        &self,
        limit: i64,
    ) -> Result<Vec<wenlan_types::RetrievalEvent>, String> {
        let path = format!("/api/retrievals/recent?limit={}", limit);
        self.get_json(&path).await
    }

    pub async fn list_recent_changes(
        &self,
        limit: i64,
    ) -> Result<Vec<wenlan_types::PageChange>, String> {
        let path = format!("/api/pages/recent-changes?limit={}", limit);
        self.get_json(&path).await
    }

    pub async fn list_recent_memories(
        &self,
        limit: i64,
        since_ms: Option<i64>,
    ) -> Result<Vec<wenlan_types::RecentActivityItem>, String> {
        let path = match since_ms {
            Some(ms) => format!("/api/memory/recent?limit={}&since_ms={}", limit, ms),
            None => format!("/api/memory/recent?limit={}", limit),
        };
        self.get_json(&path).await
    }

    pub async fn get_enrichment_status(
        &self,
        source_id: &str,
    ) -> Result<wenlan_types::EnrichmentStatusResponse, String> {
        let path = format!("/api/memory/{}/enrichment-status", source_id);
        self.get_json(&path).await
    }

    pub async fn get_memory_revisions(
        &self,
        source_id: &str,
    ) -> Result<wenlan_types::responses::ListMemoryRevisionsResponse, String> {
        let path = format!("/api/memory/{}/revisions", source_id);
        self.get_json(&path).await
    }

    pub async fn list_unconfirmed_memories(
        &self,
        limit: i64,
    ) -> Result<Vec<wenlan_types::RecentActivityItem>, String> {
        let path = format!("/api/memory/unconfirmed?limit={}", limit);
        self.get_json(&path).await
    }

    pub async fn list_recent_pages(
        &self,
        limit: i64,
        since_ms: Option<i64>,
    ) -> Result<Vec<wenlan_types::RecentActivityItem>, String> {
        let path = match since_ms {
            Some(ms) => format!("/api/pages/recent?limit={}&since_ms={}", limit, ms),
            None => format!("/api/pages/recent?limit={}", limit),
        };
        self.get_json(&path).await
    }

    pub async fn list_recent_relations(
        &self,
        limit: Option<usize>,
        since_ms: Option<i64>,
    ) -> Result<Vec<wenlan_types::RecentRelation>, String> {
        let mut path = format!(
            "/api/knowledge/recent-relations?limit={}",
            limit.unwrap_or(10)
        );
        if let Some(ms) = since_ms {
            path.push_str(&format!("&since_ms={}", ms));
        }
        self.get_json(&path).await
    }

    pub async fn get_page_sources(
        &self,
        page_id: &str,
    ) -> Result<Vec<wenlan_types::PageSourceWithMemory>, String> {
        let path = format!("/api/pages/{}/sources", page_id);
        self.get_json(&path).await
    }

    pub async fn get_page_links(
        &self,
        page_id: &str,
    ) -> Result<wenlan_types::responses::PageLinksResponse, String> {
        let path = format!("/api/pages/{}/links", page_id);
        self.get_json(&path).await
    }

    pub async fn get_page_revisions(&self, page_id: &str) -> Result<serde_json::Value, String> {
        let path = format!("/api/pages/{}/revisions", page_id);
        self.get_json(&path).await
    }

    pub async fn list_orphan_links(
        &self,
        min_count: Option<usize>,
    ) -> Result<wenlan_types::responses::OrphanLinksResponse, String> {
        let path = match min_count {
            Some(min_count) => format!("/api/pages/orphan-links?min_count={}", min_count),
            None => "/api/pages/orphan-links".to_string(),
        };
        self.get_json(&path).await
    }

    pub async fn test_llm(
        &self,
        endpoint: String,
        model: String,
        api_key: Option<String>,
    ) -> Result<wenlan_types::requests::TestLlmResponse, String> {
        // Hand-built body: pinned wenlan-types 0.12 TestLlmRequest has no
        // api_key field; the daemon (≥0.13) reads it, 0.12 never receives it
        // because the UI omits the key below 0.13.
        let mut body = serde_json::json!({ "endpoint": endpoint, "model": model });
        if let Some(key) = api_key.filter(|k| !k.is_empty()) {
            body["api_key"] = serde_json::Value::String(key);
        }
        self.post_json("/api/llm/test", &body).await
    }

    pub async fn get_on_device_model(&self) -> Result<OnDeviceModelResponse, String> {
        self.get_json("/api/on-device-model").await
    }

    pub async fn download_on_device_model(&self, model_id: String) -> Result<(), String> {
        let req = OnDeviceModelRequest { model_id };
        let _resp: wenlan_types::responses::SuccessResponse = self
            .post_json("/api/on-device-model/download", &req)
            .await?;
        Ok(())
    }

    // ── Refinery queue ─────────────────────────────────────────────────────

    pub async fn list_refinements(
        &self,
        limit: Option<usize>,
    ) -> Result<wenlan_types::responses::ListRefinementsResponse, String> {
        let mut path = "/api/refinery/queue".to_string();
        if let Some(limit) = limit {
            path.push_str(&format!("?limit={limit}"));
        }
        // The daemon can run ahead of our pinned wenlan-types and emit proposal
        // actions this build doesn't know yet. Parse proposals one-by-one so an
        // unknown variant degrades to a skip instead of emptying the whole queue.
        let raw: serde_json::Value = self.get_json(&path).await?;
        let all = raw
            .get("proposals")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let proposals: Vec<wenlan_types::responses::RefinementProposalSummary> = raw
            .get("proposals")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|p| serde_json::from_value(p.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();
        if proposals.len() < all {
            tracing::warn!(
                skipped = all - proposals.len(),
                "list_refinements: skipped refinement proposals with unknown action variants"
            );
        }
        Ok(wenlan_types::responses::ListRefinementsResponse { proposals })
    }

    pub async fn accept_refinement(
        &self,
        id: &str,
    ) -> Result<wenlan_types::responses::AcceptRefinementResponse, String> {
        self.post_empty(&format!("/api/refinery/queue/{id}/accept"))
            .await
    }

    pub async fn reject_refinement(
        &self,
        id: &str,
    ) -> Result<wenlan_types::responses::RejectRefinementResponse, String> {
        self.post_empty(&format!("/api/refinery/queue/{id}/reject"))
            .await
    }

    // ── Config ─────────────────────────────────────────────────────────────

    /// GET /api/config — return the daemon's current config.
    pub async fn get_config(&self) -> Result<wenlan_types::responses::ConfigResponse, String> {
        self.get_json("/api/config").await
    }

    /// PUT /api/config — update one or more fields and return the new config.
    /// Pass `Option<T>` fields; `None` leaves a field unchanged.
    pub async fn update_config(
        &self,
        req: wenlan_types::requests::UpdateConfigRequest,
    ) -> Result<wenlan_types::responses::ConfigResponse, String> {
        let body = sparse_update_config(req)?;
        self.put_json("/api/config", &body).await
    }

    /// GET /api/setup/status — return daemon-owned setup/model/key state.
    pub async fn get_setup_status(&self) -> Result<SetupStatusResponse, String> {
        self.get_json("/api/setup/status").await
    }

    /// Mark setup complete/incomplete through the daemon config endpoint.
    pub async fn set_setup_completed(&self, completed: bool) -> Result<(), String> {
        self.update_config(empty_update().with_setup_completed(completed))
            .await
            .map(|_| ())
    }

    pub async fn get_model_choice(&self) -> Result<(Option<String>, Option<String>), String> {
        let cfg = self.get_config().await?;
        Ok((cfg.routine_model, cfg.synthesis_model))
    }

    /// Patch daemon model selection. `None` preserves the existing daemon value.
    pub async fn set_model_choice(
        &self,
        routine_model: Option<String>,
        synthesis_model: Option<String>,
    ) -> Result<(), String> {
        self.update_config(empty_update().with_model_choice(routine_model, synthesis_model))
            .await
            .map(|_| ())
    }

    pub async fn get_external_llm(&self) -> Result<(Option<String>, Option<String>), String> {
        let cfg = self.get_config().await?;
        Ok((cfg.external_llm_endpoint, cfg.external_llm_model))
    }

    /// Presence flag from daemon ≥ 0.13 (`external_llm_api_key_configured`,
    /// spec §7.2). Reads raw JSON because pinned 0.12 ConfigResponse lacks
    /// the field; absent ⇒ false.
    pub async fn external_llm_key_configured(&self) -> Result<bool, String> {
        let cfg: serde_json::Value = self.get_json("/api/config").await?;
        Ok(cfg
            .get("external_llm_api_key_configured")
            .and_then(|b| b.as_bool())
            .unwrap_or(false))
    }

    /// Patch daemon external LLM config. `None` endpoint/model preserves the
    /// existing daemon value. `api_key`: `None` = omit (preserve stored key),
    /// `Some("")` = clear, `Some(key)` = replace (spec §7.2 tri-state).
    pub async fn set_external_llm(
        &self,
        endpoint: Option<String>,
        model: Option<String>,
        api_key: Option<String>,
    ) -> Result<(), String> {
        let mut body = sparse_update_config(empty_update().with_external_llm(endpoint, model))?;
        if let Some(key) = api_key {
            body["external_llm_api_key"] = serde_json::Value::String(key);
        }
        let _resp: wenlan_types::responses::ConfigResponse =
            self.put_json("/api/config", &body).await?;
        Ok(())
    }

    pub async fn set_remote_access_enabled(
        &self,
        enabled: bool,
    ) -> Result<wenlan_types::responses::ConfigResponse, String> {
        self.update_config(empty_update().with_remote_access_enabled(enabled))
            .await
    }

    /// GET /api/config/routing — resolved per-job routing (daemon ≥ PR #357).
    /// Feature-detection point: a daemon without the endpoint answers 404, which
    /// maps to `Ok(None)` (LEGACY mode) rather than an error. Any other non-2xx
    /// is a real failure and surfaces as `Err`.
    pub async fn get_resolved_routing(&self) -> Result<Option<ResolvedRouting>, String> {
        let path = "/api/config/routing";
        let resp = self
            .client
            .get(self.url(path))
            .send()
            .await
            .map_err(|e| format!("HTTP GET {}: {}", path, e))?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            return Err(format!("HTTP GET {} returned {}", path, resp.status()));
        }
        resp.json()
            .await
            .map(Some)
            .map_err(|e| format!("Parse {}: {}", path, e))
    }

    /// PATCH the per-job source pins (daemon ≥ PR #357). Each arg: `None` leaves
    /// the pin untouched, `Some("")` clears it (auto chain), `Some(source)` pins.
    /// Written as raw JSON because the pinned `UpdateConfigRequest` predates these
    /// fields (mirrors `set_external_llm`). Only call once the routing endpoint is
    /// known present — never PATCH unknown fields at an old daemon.
    pub async fn set_source_pin(
        &self,
        everyday_source: Option<String>,
        synthesis_source: Option<String>,
    ) -> Result<(), String> {
        let mut body = sparse_update_config(empty_update())?;
        if let Some(v) = everyday_source {
            body["everyday_source"] = serde_json::Value::String(v);
        }
        if let Some(v) = synthesis_source {
            body["synthesis_source"] = serde_json::Value::String(v);
        }
        let _resp: serde_json::Value = self.put_json("/api/config", &body).await?;
        Ok(())
    }
}

fn daemon_port() -> u16 {
    std::env::var("WENLAN_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            std::env::var("ORIGIN_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(7878)
}

fn percent_encode_path_segment(value: &str) -> String {
    value
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                vec![byte as char]
            }
            _ => format!("%{byte:02X}").chars().collect(),
        })
        .collect()
}

// `UpdateConfigRequest` does not derive `Default` in wenlan-types, so
// build a baseline with every field set to `None` here. When/if wenlan-types
// adds the derive, these helpers can be deleted in favor of
// `UpdateConfigRequest { skip_apps: Some(...), ..Default::default() }`.
fn empty_update() -> wenlan_types::requests::UpdateConfigRequest {
    wenlan_types::requests::UpdateConfigRequest {
        skip_apps: None,
        skip_title_patterns: None,
        private_browsing_detection: None,
        setup_completed: None,
        clipboard_enabled: None,
        screen_capture_enabled: None,
        remote_access_enabled: None,
        routine_model: None,
        synthesis_model: None,
        external_llm_endpoint: None,
        external_llm_model: None,
        everyday_source: None,
        synthesis_source: None,
        page_map_auto_suggest: None,
        // Tri-state (Option<Option<_>>): outer None = omit from JSON =
        // preserve the stored key. Note sparse_update_config strips nulls,
        // so a future caller passing Some(None) to clear the key must bypass
        // that helper.
        external_llm_api_key: None,
    }
}

fn sparse_update_config(
    req: wenlan_types::requests::UpdateConfigRequest,
) -> Result<serde_json::Value, String> {
    let mut value =
        serde_json::to_value(req).map_err(|e| format!("Serialize config update: {}", e))?;
    if let serde_json::Value::Object(ref mut map) = value {
        map.retain(|_, value| !value.is_null());
    }
    Ok(value)
}

trait UpdateConfigBuilder {
    fn with_remote_access_enabled(self, v: bool) -> Self;
    fn with_setup_completed(self, v: bool) -> Self;
    fn with_model_choice(
        self,
        routine_model: Option<String>,
        synthesis_model: Option<String>,
    ) -> Self;
    fn with_external_llm(self, endpoint: Option<String>, model: Option<String>) -> Self;
}

impl UpdateConfigBuilder for wenlan_types::requests::UpdateConfigRequest {
    fn with_remote_access_enabled(mut self, v: bool) -> Self {
        self.remote_access_enabled = Some(v);
        self
    }
    fn with_setup_completed(mut self, v: bool) -> Self {
        self.setup_completed = Some(v);
        self
    }
    fn with_model_choice(
        mut self,
        routine_model: Option<String>,
        synthesis_model: Option<String>,
    ) -> Self {
        self.routine_model = routine_model;
        self.synthesis_model = synthesis_model;
        self
    }
    fn with_external_llm(mut self, endpoint: Option<String>, model: Option<String>) -> Self {
        self.external_llm_endpoint = endpoint;
        self.external_llm_model = model;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    struct PortEnvGuard {
        wenlan_port: Option<std::ffi::OsString>,
        origin_port: Option<std::ffi::OsString>,
    }

    impl PortEnvGuard {
        fn capture() -> Self {
            Self {
                wenlan_port: std::env::var_os("WENLAN_PORT"),
                origin_port: std::env::var_os("ORIGIN_PORT"),
            }
        }
    }

    impl Drop for PortEnvGuard {
        fn drop(&mut self) {
            match &self.wenlan_port {
                Some(value) => std::env::set_var("WENLAN_PORT", value),
                None => std::env::remove_var("WENLAN_PORT"),
            }
            match &self.origin_port {
                Some(value) => std::env::set_var("ORIGIN_PORT", value),
                None => std::env::remove_var("ORIGIN_PORT"),
            }
        }
    }

    async fn serve_json_once(body: &'static str) -> (String, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 2048];
            let n = stream.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });
        (format!("http://{}", addr), handle)
    }

    async fn serve_status_once(
        status: u16,
        reason: &'static str,
        body: &'static str,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 2048];
            let n = stream.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });
        (format!("http://{}", addr), handle)
    }

    #[tokio::test]
    async fn optional_get_maps_only_not_found_to_none() {
        let (base_url, request) =
            serve_status_once(404, "Not Found", r#"{"error":"missing"}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let value: Option<serde_json::Value> = client
            .get_optional_json("/api/memory/missing/detail")
            .await
            .unwrap();

        assert_eq!(value, None);
        assert_eq!(
            request.await.unwrap().lines().next().unwrap_or_default(),
            "GET /api/memory/missing/detail HTTP/1.1"
        );
    }

    #[tokio::test]
    async fn optional_get_preserves_other_http_failures() {
        let (base_url, _request) =
            serve_status_once(500, "Internal Server Error", r#"{"error":"failed"}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let error = client
            .get_optional_json::<serde_json::Value>("/api/memory/id/detail")
            .await
            .unwrap_err();

        assert!(error.contains("returned 500"), "{error}");
    }

    #[tokio::test]
    async fn optional_get_preserves_malformed_success_failures() {
        let (base_url, _request) = serve_status_once(200, "OK", "not-json").await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let error = client
            .get_optional_json::<serde_json::Value>("/api/memory/id/detail")
            .await
            .unwrap_err();

        assert!(error.contains("Parse /api/memory/id/detail"), "{error}");
    }

    #[tokio::test]
    async fn optional_get_deserializes_present_values() {
        let (base_url, _request) = serve_status_once(
            200,
            "OK",
            r#"{"memory":{"source_id":"memory-present","title":"Present memory","content":"body","confirmed":true,"pinned":false,"last_modified":1,"chunk_count":1,"enrichment_status":"enriched","supersede_mode":"hide","access_count":0}}"#,
        )
        .await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let value = client
            .get_optional_json::<wenlan_types::responses::MemoryDetailResponse>(
                "/api/memory/id/detail",
            )
            .await
            .unwrap()
            .and_then(|response| response.memory);

        assert_eq!(
            value.map(|memory| memory.source_id),
            Some("memory-present".to_string())
        );
    }

    /// Accepts one connection, reads the request, then never responds.
    async fn serve_hang_once() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0_u8; 2048];
            let _ = stream.read(&mut buf).await;
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            drop(stream);
        });
        format!("http://{}", addr)
    }

    #[tokio::test]
    async fn requests_error_instead_of_hanging_when_daemon_never_responds() {
        let base_url = serve_hang_once().await;
        let client = WenlanClient {
            client: build_http_client(
                std::time::Duration::from_millis(200),
                std::time::Duration::from_millis(300),
            ),
            base_url,
        };

        let started = std::time::Instant::now();
        let result: Result<HealthResponse, String> = client.get_json("/api/health").await;

        let err = result.unwrap_err();
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "request did not fail fast: {:?}",
            started.elapsed()
        );
        assert!(
            err.contains("HTTP GET /api/health"),
            "unexpected error shape: {err}"
        );
    }

    #[tokio::test]
    async fn health_probe_fails_fast_even_with_production_timeouts() {
        let base_url = serve_hang_once().await;
        let client = WenlanClient {
            client: build_http_client(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
            base_url,
        };

        let started = std::time::Instant::now();
        let err = client.health().await.unwrap_err();

        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "health probe waited on the ingest-sized backstop: {:?}",
            started.elapsed()
        );
        assert!(
            err.contains("HTTP GET /api/health"),
            "unexpected error shape: {err}"
        );
    }

    #[test]
    fn update_config_builder_can_set_setup_completed() {
        let req = empty_update().with_setup_completed(true);

        assert_eq!(req.setup_completed, Some(true));
        assert_eq!(req.skip_apps, None);
        assert_eq!(req.skip_title_patterns, None);
    }

    #[test]
    fn update_config_builder_patches_remote_access_without_touching_other_config() {
        let remote_req = empty_update().with_remote_access_enabled(true);
        assert_eq!(remote_req.remote_access_enabled, Some(true));
        assert_eq!(remote_req.setup_completed, None);
        assert_eq!(remote_req.skip_apps, None);
    }

    fn request_body(request: &str) -> serde_json::Value {
        let (_, body) = request.split_once("\r\n\r\n").unwrap();
        serde_json::from_str(body).unwrap()
    }

    #[tokio::test]
    async fn remote_access_config_field_uses_daemon_patch_endpoint() {
        let config_body = r#"{"skip_apps":[],"skip_title_patterns":[],"private_browsing_detection":true,"setup_completed":true,"clipboard_enabled":true,"screen_capture_enabled":false,"remote_access_enabled":true}"#;
        let (base_url, request) = serve_json_once(config_body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let config = client.set_remote_access_enabled(true).await.unwrap();

        assert!(config.remote_access_enabled);
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "PUT /api/config HTTP/1.1"
        );
        assert_eq!(
            request_body(&request),
            serde_json::json!({"remote_access_enabled": true})
        );
    }

    #[test]
    fn update_config_builder_patches_model_fields_without_touching_other_config() {
        let req = empty_update().with_model_choice(Some("claude-haiku-4-5-20251001".into()), None);

        assert_eq!(
            req.routine_model,
            Some("claude-haiku-4-5-20251001".to_string())
        );
        assert_eq!(req.synthesis_model, None);
        assert_eq!(req.external_llm_endpoint, None);
        assert_eq!(req.external_llm_model, None);
    }

    #[test]
    fn update_config_builder_patches_external_llm_fields_without_touching_models() {
        let req = empty_update().with_external_llm(None, Some("qwen3".into()));

        assert_eq!(req.external_llm_endpoint, None);
        assert_eq!(req.external_llm_model, Some("qwen3".to_string()));
        assert_eq!(req.routine_model, None);
        assert_eq!(req.synthesis_model, None);
    }

    #[tokio::test]
    async fn test_llm_omits_api_key_field_when_none_or_empty() {
        let (base_url, request) = serve_json_once(r#"{"response":"ok"}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        client
            .test_llm("http://localhost:11434/v1".into(), "qwen3".into(), None)
            .await
            .unwrap();

        let request = request.await.unwrap();
        assert_eq!(
            request_body(&request),
            serde_json::json!({"endpoint": "http://localhost:11434/v1", "model": "qwen3"})
        );

        let (base_url, request) = serve_json_once(r#"{"response":"ok"}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        client
            .test_llm(
                "http://localhost:11434/v1".into(),
                "qwen3".into(),
                Some(String::new()),
            )
            .await
            .unwrap();

        let request = request.await.unwrap();
        assert_eq!(
            request_body(&request),
            serde_json::json!({"endpoint": "http://localhost:11434/v1", "model": "qwen3"})
        );
    }

    #[tokio::test]
    async fn test_llm_includes_api_key_field_when_supplied() {
        let (base_url, request) = serve_json_once(r#"{"response":"ok"}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        client
            .test_llm(
                "http://localhost:11434/v1".into(),
                "qwen3".into(),
                Some("sk-secret".into()),
            )
            .await
            .unwrap();

        let request = request.await.unwrap();
        assert_eq!(
            request_body(&request),
            serde_json::json!({
                "endpoint": "http://localhost:11434/v1",
                "model": "qwen3",
                "api_key": "sk-secret"
            })
        );
    }

    #[tokio::test]
    async fn set_external_llm_omits_api_key_field_when_none() {
        let config_body = r#"{"skip_apps":[],"skip_title_patterns":[],"private_browsing_detection":true,"setup_completed":true,"clipboard_enabled":true,"screen_capture_enabled":false,"remote_access_enabled":false}"#;
        let (base_url, request) = serve_json_once(config_body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        client
            .set_external_llm(Some("http://localhost:11434/v1".into()), None, None)
            .await
            .unwrap();

        let request = request.await.unwrap();
        assert_eq!(
            request_body(&request),
            serde_json::json!({"external_llm_endpoint": "http://localhost:11434/v1"})
        );
    }

    #[tokio::test]
    async fn set_external_llm_clears_key_with_empty_string() {
        let config_body = r#"{"skip_apps":[],"skip_title_patterns":[],"private_browsing_detection":true,"setup_completed":true,"clipboard_enabled":true,"screen_capture_enabled":false,"remote_access_enabled":false}"#;
        let (base_url, request) = serve_json_once(config_body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        client
            .set_external_llm(None, None, Some(String::new()))
            .await
            .unwrap();

        let request = request.await.unwrap();
        assert_eq!(
            request_body(&request),
            serde_json::json!({"external_llm_api_key": ""})
        );
    }

    #[tokio::test]
    async fn set_external_llm_replaces_key_when_supplied() {
        let config_body = r#"{"skip_apps":[],"skip_title_patterns":[],"private_browsing_detection":true,"setup_completed":true,"clipboard_enabled":true,"screen_capture_enabled":false,"remote_access_enabled":false}"#;
        let (base_url, request) = serve_json_once(config_body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        client
            .set_external_llm(None, None, Some("sk-secret".into()))
            .await
            .unwrap();

        let request = request.await.unwrap();
        assert_eq!(
            request_body(&request),
            serde_json::json!({"external_llm_api_key": "sk-secret"})
        );
    }

    #[tokio::test]
    async fn get_resolved_routing_maps_404_to_none() {
        // Feature detection: an old daemon without the endpoint 404s, and that
        // must read as LEGACY mode (Ok(None)), never as an error.
        let (base_url, _request) = serve_status_once(404, "Not Found", "not found").await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        assert_eq!(client.get_resolved_routing().await.unwrap(), None);
    }

    #[tokio::test]
    async fn get_resolved_routing_parses_200_body() {
        let body = r#"{"everyday":{"source":"on_device","model":"qwen3-4b","mode":"pinned","pin":"on_device"},"synthesis":{"source":"external","model":"gpt-5.2","mode":"auto"},"pool":{"anthropic":{"configured":false,"everyday_model":null,"synthesis_model":null},"external":{"endpoint":"https://api.openai.com/v1","model":"gpt-5.2"},"on_device":{"selected":"qwen3-4b","loaded":true}}}"#;
        let (base_url, _request) = serve_json_once(body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let routing = client.get_resolved_routing().await.unwrap().unwrap();
        assert_eq!(routing.everyday.source, "on_device");
        assert_eq!(routing.everyday.mode, "pinned");
        assert_eq!(routing.everyday.pin.as_deref(), Some("on_device"));
        assert_eq!(routing.synthesis.model.as_deref(), Some("gpt-5.2"));
        // synthesis's JSON omits "pin" entirely — a daemon that predates
        // #357's pin field must still deserialize, defaulting to None.
        assert_eq!(routing.synthesis.pin, None);
        assert!(!routing.pool.anthropic.configured);
        assert!(routing.pool.on_device.unwrap().loaded);
    }

    #[tokio::test]
    async fn set_source_pin_sends_only_named_pins() {
        let config_body = r#"{"skip_apps":[],"skip_title_patterns":[],"private_browsing_detection":true,"setup_completed":true,"clipboard_enabled":true,"screen_capture_enabled":false,"remote_access_enabled":false}"#;
        let (base_url, request) = serve_json_once(config_body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        client
            .set_source_pin(Some("external".into()), None)
            .await
            .unwrap();

        let request = request.await.unwrap();
        assert_eq!(
            request_body(&request),
            serde_json::json!({"everyday_source": "external"})
        );
    }

    #[tokio::test]
    async fn external_llm_key_configured_reads_presence_flag_from_raw_json() {
        let config_body = r#"{"skip_apps":[],"skip_title_patterns":[],"private_browsing_detection":true,"setup_completed":true,"clipboard_enabled":true,"screen_capture_enabled":false,"remote_access_enabled":false,"external_llm_api_key_configured":true}"#;
        let (base_url, _request) = serve_json_once(config_body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        assert!(client.external_llm_key_configured().await.unwrap());
    }

    #[tokio::test]
    async fn external_llm_key_configured_defaults_false_when_field_absent() {
        // 0.12 daemon shape: no `external_llm_api_key_configured` field at all.
        let config_body = r#"{"skip_apps":[],"skip_title_patterns":[],"private_browsing_detection":true,"setup_completed":true,"clipboard_enabled":true,"screen_capture_enabled":false,"remote_access_enabled":false}"#;
        let (base_url, _request) = serve_json_once(config_body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        assert!(!client.external_llm_key_configured().await.unwrap());
    }

    #[test]
    fn setup_status_response_deserializes_daemon_payload() {
        let status: SetupStatusResponse = serde_json::from_value(serde_json::json!({
            "setup_completed": false,
            "mode": "basic-memory",
            "anthropic_key_configured": false,
            "local_model_selected": null,
            "local_model_loaded": null,
            "local_model_cached": false
        }))
        .expect("daemon setup status payload should deserialize");

        assert!(!status.setup_completed);
        assert_eq!(status.mode, "basic-memory");
        assert!(!status.anthropic_key_configured);
        assert_eq!(status.local_model_selected, None);
        assert_eq!(status.local_model_loaded, None);
        assert!(!status.local_model_cached);
    }

    #[test]
    fn wenlan_client_prefers_wenlan_port_over_legacy_origin_port() {
        let _guard = env_lock();
        let _env = PortEnvGuard::capture();
        std::env::set_var("WENLAN_PORT", "8787");
        std::env::set_var("ORIGIN_PORT", "9898");

        let client = WenlanClient::new();

        assert_eq!(
            client.url("/api/health"),
            "http://127.0.0.1:8787/api/health"
        );
    }

    #[test]
    fn wenlan_client_falls_back_to_legacy_origin_port() {
        let _guard = env_lock();
        let _env = PortEnvGuard::capture();
        std::env::remove_var("WENLAN_PORT");
        std::env::set_var("ORIGIN_PORT", "9898");

        let client = WenlanClient::new();

        assert_eq!(
            client.url("/api/health"),
            "http://127.0.0.1:9898/api/health"
        );
    }

    #[tokio::test]
    async fn capture_stats_uses_daemon_capture_stats_endpoint() {
        let (base_url, request) = serve_json_once(r#"{"total_chunks":42}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let stats = client.get_capture_stats().await.unwrap();

        assert_eq!(stats.get("total"), Some(&42));
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "GET /api/capture-stats HTTP/1.1"
        );
    }

    #[tokio::test]
    async fn pipeline_status_uses_daemon_debug_pipeline_endpoint() {
        let body = r#"{"enrichment":{"done":3,"pending":1},"entity_linking":{"linked":7,"unlinked":2},"refinement_queue":[{"action":"merge","status":"pending","count":4}],"recaps":5,"types":{"fact":6},"quality":{"high":2}}"#;
        let (base_url, request) = serve_json_once(body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let status = client.pipeline_status().await.unwrap();

        assert_eq!(status.enrichment.get("done"), Some(&3));
        assert_eq!(status.entity_linking.linked, 7);
        assert_eq!(status.entity_linking.unlinked, 2);
        assert_eq!(status.refinement_queue[0].action, "merge");
        assert_eq!(status.refinement_queue[0].status, "pending");
        assert_eq!(status.refinement_queue[0].count, 4);
        assert_eq!(status.recaps, 5);
        assert_eq!(status.types.get("fact"), Some(&6));
        assert_eq!(status.quality.get("high"), Some(&2));
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "GET /api/debug/pipeline HTTP/1.1"
        );
    }

    #[test]
    fn pipeline_status_response_deserializes_daemon_payload() {
        let status: PipelineStatusResponse = serde_json::from_value(serde_json::json!({
            "enrichment": {"raw": 1, "classified": 2},
            "entity_linking": {"linked": 3, "unlinked": 4},
            "refinement_queue": [
                {"action": "merge", "status": "pending", "count": 5}
            ],
            "recaps": 6,
            "types": {"fact": 7},
            "quality": {"trusted": 8},
            "future_additive_key": {"ignored": true}
        }))
        .expect("daemon pipeline status payload should deserialize");

        assert_eq!(status.enrichment.get("classified"), Some(&2));
        assert_eq!(status.entity_linking.linked, 3);
        assert_eq!(status.entity_linking.unlinked, 4);
        assert_eq!(status.refinement_queue.len(), 1);
        assert_eq!(status.recaps, 6);
        assert_eq!(status.types.get("fact"), Some(&7));
        assert_eq!(status.quality.get("trusted"), Some(&8));
    }

    #[tokio::test]
    async fn list_tags_uses_daemon_tags_endpoint() {
        let (base_url, request) = serve_json_once(r#"{"tags":["ai","rust"]}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let tags = client.list_tags().await.unwrap();

        assert_eq!(tags, vec!["ai".to_string(), "rust".to_string()]);
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "GET /api/tags HTTP/1.1"
        );
    }

    #[tokio::test]
    async fn list_tag_inventory_preserves_document_tag_map() {
        let (base_url, request) = serve_json_once(
            r#"{"tags":["ai","rust"],"document_tags":{"memory::mem1":["ai"],"page::page1":["rust"]}}"#,
        )
        .await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let inventory = client.list_tag_inventory().await.unwrap();

        assert_eq!(inventory.tags, vec!["ai".to_string(), "rust".to_string()]);
        assert_eq!(
            inventory.document_tags.get("memory::mem1"),
            Some(&vec!["ai".to_string()])
        );
        assert_eq!(
            inventory.document_tags.get("page::page1"),
            Some(&vec!["rust".to_string()])
        );
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "GET /api/tags HTTP/1.1"
        );
    }

    #[tokio::test]
    async fn ingest_webpage_uses_daemon_webpage_ingest_endpoint() {
        let (base_url, request) =
            serve_json_once(r#"{"chunks_created":3,"document_id":"https://example.com/post"}"#)
                .await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), "manual-url".to_string());
        let req = wenlan_types::requests::IngestWebpageRequest {
            url: "https://example.com/post".to_string(),
            title: "Example Post".to_string(),
            content: "A durable article body.".to_string(),
            metadata: Some(metadata),
        };

        let resp = client.ingest_webpage(req).await.unwrap();

        assert_eq!(resp.chunks_created, 3);
        assert_eq!(resp.document_id, "https://example.com/post");
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "POST /api/ingest/webpage HTTP/1.1"
        );
        assert_eq!(
            request_body(&request),
            serde_json::json!({
                "url": "https://example.com/post",
                "title": "Example Post",
                "content": "A durable article body.",
                "metadata": {"source": "manual-url"}
            })
        );
    }

    #[tokio::test]
    async fn move_space_uses_daemon_space_move_endpoint() {
        let (base_url, request) = serve_json_once(r#"{"affected":7}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let resp = client.move_space("Inbox", "Archive").await.unwrap();

        assert_eq!(resp.affected, 7);
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "POST /api/spaces/Inbox/move-to/Archive HTTP/1.1"
        );
    }

    #[tokio::test]
    async fn move_space_percent_encodes_space_names_as_path_segments() {
        let (base_url, request) = serve_json_once(r#"{"affected":2}"#).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let resp = client
            .move_space("Work/Clients?old=true", "Archive#2026")
            .await
            .unwrap();

        assert_eq!(resp.affected, 2);
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "POST /api/spaces/Work%2FClients%3Fold%3Dtrue/move-to/Archive%232026 HTTP/1.1"
        );
    }

    #[tokio::test]
    async fn distill_review_posts_empty_global_request_to_daemon() {
        let body = r#"{"pages_created":0,"scoped":false,"created_ids":[],"pending":[],"stale_pages":[],"stale_truncated":false,"orphan_topics":[]}"#;
        let (base_url, request) = serve_json_once(body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let resp = client.distill_review().await.unwrap();

        assert_eq!(resp.pages_created, 0);
        assert!(!resp.scoped);
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "POST /api/distill HTTP/1.1"
        );
        assert_eq!(request_body(&request), serde_json::json!({}));
    }

    #[tokio::test]
    async fn redistill_page_posts_empty_page_specific_request_to_daemon() {
        let body = r#"{"status":"skipped","updated":false,"hint":"page re-distill needs an LLM in the daemon"}"#;
        let (base_url, request) = serve_json_once(body).await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let resp = client.redistill_page("page_refresh").await.unwrap();

        assert_eq!(resp.status, "skipped");
        assert!(!resp.updated);
        assert_eq!(
            resp.hint.as_deref(),
            Some("page re-distill needs an LLM in the daemon")
        );
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "POST /api/distill/page_refresh HTTP/1.1"
        );
        assert_eq!(request_body(&request), serde_json::json!({}));
    }

    #[test]
    fn distill_review_deserializes_daemon_review_payload_with_centroid_embedding() {
        let payload = serde_json::json!({
            "pages_created": 1,
            "scoped": false,
            "created_ids": ["page_new"],
            "pending": [{
                "source_ids": ["mem_1", "mem_2"],
                "contents": ["First source", "Second source"],
                "entity_id": "entity_rust",
                "entity_name": "Rust",
                "space": "Engineering",
                "estimated_tokens": 220,
                "centroid_embedding": [0.1, 0.2],
                "existing_page_id": "page_rust",
                "existing_page_title": "Rust notes",
                "new_memory_count": 1
            }],
            "stale_pages": [{
                "page_id": "page_old",
                "title": "Old page",
                "summary": "Needs source review",
                "source_memory_ids": ["mem_old"],
                "sources_updated_count": 3,
                "stale_reason": "source_updated",
                "user_edited": false
            }],
            "stale_truncated": true,
            "orphan_topics": [{"label": "Vector clocks", "count": 4}]
        });

        let resp: DistillReviewResponse = serde_json::from_value(payload).unwrap();

        assert_eq!(resp.created_ids, vec!["page_new"]);
        assert_eq!(resp.pending[0].centroid_embedding, Some(vec![0.1, 0.2]));
        assert_eq!(
            resp.pending[0].existing_page_title.as_deref(),
            Some("Rust notes")
        );
        assert_eq!(resp.pending[0].new_memory_count, Some(1));
        assert_eq!(resp.stale_pages[0].page_id, "page_old");
        assert_eq!(resp.orphan_topics[0].label, "Vector clocks");
        assert!(resp.stale_truncated);
    }

    #[test]
    fn distill_review_rejects_unresolved_target_hint_shape() {
        let payload = serde_json::json!({
            "pages_created": 0,
            "pages_updated": 0,
            "unresolved": "unknown target",
            "hint": "target must be a page id"
        });

        let err = serde_json::from_value::<DistillReviewResponse>(payload).unwrap_err();

        assert!(
            err.to_string().contains("missing field") || err.to_string().contains("unknown field")
        );
    }

    #[tokio::test]
    async fn status_uses_daemon_status_endpoint() {
        let (base_url, request) = serve_json_once(
            r#"{"is_running":true,"files_indexed":42,"files_total":0,"sources_connected":[],"reranker":{"state":"disabled"},"reranker_light":{"state":"active","model_id":"bge"},"reranker_mode":"lite"}"#,
        )
        .await;
        let client = WenlanClient {
            client: reqwest::Client::new(),
            base_url,
        };

        let status = client.status().await.unwrap();

        assert_eq!(status.files_indexed, 42);
        assert_eq!(status.reranker_mode, "lite");
        let request = request.await.unwrap();
        assert_eq!(
            request.lines().next().unwrap_or_default(),
            "GET /api/status HTTP/1.1"
        );
    }

    #[test]
    fn wenlan_client_exposes_enrichment_status_method() {
        let _get_enrichment_status = WenlanClient::get_enrichment_status;
    }

    #[test]
    fn enrichment_status_response_deserializes_daemon_payload() {
        let status: wenlan_types::EnrichmentStatusResponse =
            serde_json::from_value(serde_json::json!({
                "source_id": "mem-1",
                "summary": "complete",
                "steps": [
                    { "step": "classify", "status": "done", "error": null, "attempts": 1 }
                ]
            }))
            .unwrap();

        assert_eq!(status.source_id, "mem-1");
        assert_eq!(status.steps[0].step, "classify");
    }

    #[test]
    fn wenlan_client_exposes_refinery_queue_methods() {
        let _list = WenlanClient::list_refinements;
        let _accept = WenlanClient::accept_refinement;
        let _reject = WenlanClient::reject_refinement;
    }

    #[test]
    fn wenlan_client_exposes_daemon_model_config_methods() {
        let _get_model_choice = WenlanClient::get_model_choice;
        let _set_model_choice = WenlanClient::set_model_choice;
        let _get_external_llm = WenlanClient::get_external_llm;
        let _set_external_llm = WenlanClient::set_external_llm;
        let _external_llm_key_configured = WenlanClient::external_llm_key_configured;
    }

    #[allow(dead_code)]
    async fn test_llm_uses_daemon_response_envelope(client: WenlanClient) {
        let _: Result<wenlan_types::requests::TestLlmResponse, String> =
            client.test_llm(String::new(), String::new(), None).await;
    }

    #[test]
    fn test_llm_response_type_is_checked() {}

    #[allow(dead_code)]
    async fn on_device_model_uses_typed_response(client: WenlanClient) {
        let _: Result<OnDeviceModelResponse, String> = client.get_on_device_model().await;
    }

    #[allow(dead_code)]
    async fn download_on_device_model_uses_typed_request(client: WenlanClient) {
        let _: Result<(), String> = client.download_on_device_model(String::new()).await;
    }

    #[test]
    fn on_device_model_response_type_is_checked() {}

    #[test]
    fn wenlan_client_exposes_source_registry_methods() {
        let _list = WenlanClient::list_sources;
        let _add = WenlanClient::add_source;
        let _remove = WenlanClient::remove_source;
        let _sync = WenlanClient::sync_source;
    }

    #[test]
    fn wenlan_client_exposes_page_link_methods() {
        let _get = WenlanClient::get_page_links;
        let _list = WenlanClient::list_orphan_links;
    }

    #[test]
    fn wenlan_client_exposes_revision_history_methods() {
        let _memory = WenlanClient::get_memory_revisions;
        let _page = WenlanClient::get_page_revisions;
    }

    #[test]
    fn revision_history_responses_deserialize_daemon_payloads() {
        let memory: wenlan_types::responses::ListMemoryRevisionsResponse =
            serde_json::from_value(serde_json::json!({
                "current_source_id": "mem-1",
                "chain_depth": 1,
                "entries": [
                    {
                        "source_id": "mem-1",
                        "depth": 0,
                        "title": "Current",
                        "content_preview": "Current version",
                        "last_modified": 10,
                        "source_agent": "claude-code",
                        "supersede_mode": "protected_revision",
                        "delta_summary": "Clarified wording"
                    }
                ]
            }))
            .unwrap();
        assert_eq!(memory.current_source_id, "mem-1");
        assert_eq!(
            memory.entries[0].delta_summary.as_deref(),
            Some("Clarified wording")
        );

        let page: wenlan_types::responses::ListPageRevisionsResponse =
            serde_json::from_value(serde_json::json!({
                "page_id": "page-1",
                "current_version": 2,
                "user_edited": false,
                "stale_reason": null,
                "entries": [
                    {
                        "version": 2,
                        "at": 1782490000000i64,
                        "edited_by": "distill",
                        "delta_summary": "Added backlinks",
                        "incoming_source_ids": ["mem-1"]
                    }
                ]
            }))
            .unwrap();
        assert_eq!(page.page_id, "page-1");
        assert_eq!(
            page.entries[0].incoming_source_ids.as_ref().unwrap(),
            &vec!["mem-1".to_string()]
        );
    }
}
