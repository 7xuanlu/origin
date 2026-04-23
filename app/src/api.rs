// SPDX-License-Identifier: AGPL-3.0-only
//! HTTP client for the Origin daemon (origin-server).
//!
//! Thin wrapper around `reqwest::Client` that maps each daemon endpoint
//! to a typed method. The Tauri app uses this instead of direct DB access.

use origin_types::responses::HealthResponse;
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// HTTP client that proxies requests to the origin-server daemon.
#[derive(Clone)]
pub struct OriginClient {
    client: Client,
    base_url: String,
}

impl Default for OriginClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OriginClient {
    pub fn new() -> Self {
        let port: u16 = std::env::var("ORIGIN_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7878);
        Self {
            client: Client::new(),
            base_url: format!("http://127.0.0.1:{}", port),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
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
        self.get_json("/api/health").await
    }

    // ── Chat export import ─────────────────────────────────────────

    pub async fn import_chat_export(
        &self,
        path: &str,
    ) -> Result<origin_types::import::ImportChatExportResponse, String> {
        let req = origin_types::import::ImportChatExportRequest {
            path: path.to_string(),
        };
        self.post_json("/api/import/chat-export", &req).await
    }

    pub async fn list_pending_imports(
        &self,
    ) -> Result<Vec<origin_types::import::PendingImport>, String> {
        self.get_json("/api/import/state").await
    }

    // ── Onboarding milestones ──────────────────────────────────────

    pub async fn list_onboarding_milestones(
        &self,
    ) -> Result<Vec<origin_core::onboarding::MilestoneRecord>, String> {
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
    ) -> Result<Vec<origin_types::RetrievalEvent>, String> {
        let path = format!("/api/retrievals/recent?limit={}", limit);
        self.get_json(&path).await
    }

    pub async fn list_recent_changes(
        &self,
        limit: i64,
    ) -> Result<Vec<origin_types::ConceptChange>, String> {
        let path = format!("/api/concepts/recent-changes?limit={}", limit);
        self.get_json(&path).await
    }

    pub async fn list_recent_memories(
        &self,
        limit: i64,
        since_ms: Option<i64>,
    ) -> Result<Vec<origin_types::RecentActivityItem>, String> {
        let path = match since_ms {
            Some(ms) => format!("/api/memory/recent?limit={}&since_ms={}", limit, ms),
            None => format!("/api/memory/recent?limit={}", limit),
        };
        self.get_json(&path).await
    }

    pub async fn list_unconfirmed_memories(
        &self,
        limit: i64,
    ) -> Result<Vec<origin_types::RecentActivityItem>, String> {
        let path = format!("/api/memory/unconfirmed?limit={}", limit);
        self.get_json(&path).await
    }

    pub async fn list_recent_concepts(
        &self,
        limit: i64,
        since_ms: Option<i64>,
    ) -> Result<Vec<origin_types::RecentActivityItem>, String> {
        let path = match since_ms {
            Some(ms) => format!("/api/concepts/recent?limit={}&since_ms={}", limit, ms),
            None => format!("/api/concepts/recent?limit={}", limit),
        };
        self.get_json(&path).await
    }

    pub async fn list_recent_relations(
        &self,
        limit: Option<usize>,
        since_ms: Option<i64>,
    ) -> Result<Vec<origin_types::RecentRelation>, String> {
        let mut path = format!(
            "/api/knowledge/recent-relations?limit={}",
            limit.unwrap_or(10)
        );
        if let Some(ms) = since_ms {
            path.push_str(&format!("&since_ms={}", ms));
        }
        self.get_json(&path).await
    }

    pub async fn get_concept_sources(
        &self,
        concept_id: &str,
    ) -> Result<Vec<origin_types::ConceptSourceWithMemory>, String> {
        let path = format!("/api/concepts/{}/sources", concept_id);
        self.get_json(&path).await
    }
}
