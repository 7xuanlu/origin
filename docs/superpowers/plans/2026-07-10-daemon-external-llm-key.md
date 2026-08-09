# Daemon External-LLM API Key Implementation Plan (PR 3 of 3, repo `7xuanlu/wenlan`)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Daemon 0.13 support for keyed OpenAI-compatible endpoints: `external_llm_api_key` config (Bearer auth), a strict key-lifecycle contract, `PUT /api/config` hot-swap of the external provider, testable keys, and honest external-slot status (spec §7).

**Architecture:** Three-slot provider model unchanged. The external slot (`ServerState.external_llm`, `state.rs:38`) gains an optional Bearer key, gets rebuilt on config PUT (mirroring `apply_anthropic_provider`, `config_routes.rs:129`), and reports `{configured, loaded}` in setup status. The key value never leaves the daemon.

**Tech Stack:** Rust workspace (`wenlan-types`, `wenlan-core`, `wenlan-server`), axum 0.7, reqwest, serde. No new dependencies.

**Repo/branch:** `/Users/lucian/Repos/wenlan`, new branch `feat/external-llm-key` from `origin/main`. Versioning is release-please-managed (`Cargo.toml` `version = "0.12.0"  # x-release-please-version`) — do NOT bump versions by hand; use `feat:`-prefixed conventional commits so the next release cuts **0.13.0**.

**Spec:** `wenlan-app` repo, `docs/superpowers/specs/2026-07-10-settings-onboarding-redesign-design.md` §7.

## Global Constraints

- **Key lifecycle contract (spec §7.2):** the key VALUE is never serialized in any response. `GET /api/config` exposes only `external_llm_api_key_configured: bool`. `PUT /api/config`: field omitted ⇒ preserve; `null` or `""` ⇒ clear; non-empty ⇒ replace.
- Config file perms: 0600 whenever ANY key is stored (`anthropic_api_key` OR `external_llm_api_key`) — `wenlan-core/src/config.rs` `save_config`.
- All changes additive; 0.12 clients (the pinned app) keep working — new response fields have serde defaults, new request fields are optional.
- The working tree is on branch `codex/docs-eval-ci` — branch from `origin/main`, not from it. The git stash stack is shared with other sessions: NEVER use bare `git stash`/`git stash pop`.
- Gates before PR: `cargo fmt --check --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
- Every commit: conventional `feat(server):`/`feat(core):` style + the session Co-Authored-By/Claude-Session trailer.

## Interfaces produced (consumed by the app-features plan)

| Surface | Shape |
| --- | --- |
| `GET /api/config` | + `"external_llm_api_key_configured": bool` |
| `PUT /api/config` | + optional `"external_llm_api_key"` (tri-state per contract); `""` for `external_llm_endpoint`/`external_llm_model` now clears them; PUT hot-swaps the external slot |
| `POST /api/llm/test` | + optional `"api_key"` (used for the probe only, not persisted) |
| `GET /api/setup/status` | + `"external_llm": {"configured": bool, "loaded": bool}` |

---

### Task 1: `external_llm_api_key` config field + 0600 condition

**Files:**
- Modify: `crates/wenlan-core/src/config.rs` (Config struct ~line 34; `save_config` ~line 164; tests mod)

**Interfaces:**
- Produces: `Config.external_llm_api_key: Option<String>` (serde default); `fn stores_secret(config: &Config) -> bool` used by `save_config`.

- [ ] **Step 1: Write the failing tests** (append to the existing `mod tests` in `config.rs`)

```rust
#[test]
fn test_external_llm_api_key_roundtrip_and_default() {
    let cfg: Config = serde_json::from_str("{}").unwrap();
    assert!(cfg.external_llm_api_key.is_none());
    let cfg = Config {
        external_llm_api_key: Some("sk-test".into()),
        ..Config::default()
    };
    let restored: Config = serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
    assert_eq!(restored.external_llm_api_key.as_deref(), Some("sk-test"));
}

#[test]
fn test_stores_secret_covers_both_keys() {
    assert!(!stores_secret(&Config::default()));
    assert!(stores_secret(&Config {
        anthropic_api_key: Some("k".into()),
        ..Config::default()
    }));
    assert!(stores_secret(&Config {
        external_llm_api_key: Some("k".into()),
        ..Config::default()
    }));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p wenlan-core config::tests` → FAIL: no field `external_llm_api_key`, `stores_secret` not found.

- [ ] **Step 3: Implement**

In `Config` (after `external_llm_model`):

```rust
/// Bearer key for the external OpenAI-compatible endpoint. Never returned
/// by any API response — see the key-lifecycle contract in the design spec.
#[serde(default)]
pub external_llm_api_key: Option<String>,
```

Above `save_config`:

```rust
/// True when the config holds any credential — used to tighten file perms.
fn stores_secret(config: &Config) -> bool {
    config.anthropic_api_key.is_some() || config.external_llm_api_key.is_some()
}
```

In `save_config`, replace `if config.anthropic_api_key.is_some() {` with `if stores_secret(config) {`.

- [ ] **Step 4: Verify** — `cargo test -p wenlan-core` → PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(core): add external_llm_api_key config field, 0600 for any stored key"`

---

### Task 2: wenlan-types request/response fields (tri-state key)

**Files:**
- Modify: `crates/wenlan-types/src/responses.rs` (`ConfigResponse`, ~line 402)
- Modify: `crates/wenlan-types/src/requests.rs` (`UpdateConfigRequest`, `TestLlmRequest`)

**Interfaces:**
- Produces: `ConfigResponse.external_llm_api_key_configured: bool`; `UpdateConfigRequest.external_llm_api_key: Option<Option<String>>` (outer `None` = omitted, `Some(None)` = explicit null, `Some(Some(s))` = value); `TestLlmRequest.api_key: Option<String>`.

- [ ] **Step 1: Write the failing tests** (in `requests.rs` tests mod; create `#[cfg(test)] mod tests` if absent)

```rust
#[test]
fn update_config_request_external_key_tristate() {
    let r: UpdateConfigRequest = serde_json::from_str("{}").unwrap();
    assert_eq!(r.external_llm_api_key, None); // omitted
    let r: UpdateConfigRequest =
        serde_json::from_str(r#"{"external_llm_api_key":null}"#).unwrap();
    assert_eq!(r.external_llm_api_key, Some(None)); // explicit null
    let r: UpdateConfigRequest =
        serde_json::from_str(r#"{"external_llm_api_key":"sk-x"}"#).unwrap();
    assert_eq!(r.external_llm_api_key, Some(Some("sk-x".to_string())));
}

#[test]
fn test_llm_request_api_key_optional() {
    let r: TestLlmRequest =
        serde_json::from_str(r#"{"endpoint":"http://x","model":"m"}"#).unwrap();
    assert!(r.api_key.is_none());
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p wenlan-types` → FAIL (fields missing).

- [ ] **Step 3: Implement**

`requests.rs` — helper (file scope) + fields:

```rust
/// Distinguishes an omitted JSON field (outer None) from an explicit `null`
/// (Some(None)). Used for tri-state secret updates.
fn double_option<'de, T, D>(de: D) -> Result<Option<Option<T>>, D::Error>
where
    T: serde::Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    serde::Deserialize::deserialize(de).map(Some)
}
```

In `UpdateConfigRequest` (after `external_llm_model`):

```rust
/// API key for the external endpoint. Tri-state: omitted = preserve stored
/// key; `null` or `""` = clear; non-empty = replace. Never echoed back.
#[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
pub external_llm_api_key: Option<Option<String>>,
```

In `TestLlmRequest`:

```rust
/// Optional bearer key for this probe only — not persisted.
#[serde(default, skip_serializing_if = "Option::is_none")]
pub api_key: Option<String>,
```

`responses.rs` — in `ConfigResponse` (after `external_llm_model`):

```rust
/// Whether an external-LLM API key is stored. The key value itself is
/// never serialized anywhere.
#[serde(default)]
pub external_llm_api_key_configured: bool,
```

- [ ] **Step 4: Verify** — `cargo test -p wenlan-types` → PASS. Then `cargo build --workspace` — expect FAIL in `wenlan-server` (`config_to_response` misses the new field); add `external_llm_api_key_configured: false,` placeholder in `crates/wenlan-server/src/config_routes.rs::config_to_response` (Task 4 replaces it) so the workspace builds.

- [ ] **Step 5: Commit** — `git commit -am "feat(types): tri-state external_llm_api_key request field + configured flag"`

---

### Task 3: Bearer header in `OpenAICompatibleProvider` + testable key in `/api/llm/test`

**Files:**
- Modify: `crates/wenlan-core/src/llm_provider.rs` (`OpenAICompatibleProvider`, ~line 1196)
- Modify: `crates/wenlan-server/src/routes.rs` (`handle_test_llm`, ~line 1186; new test mod)

**Interfaces:**
- Produces: `OpenAICompatibleProvider::new_with_key(endpoint: String, model: String, api_key: Option<String>) -> Self`; existing `new(endpoint, model)` delegates with `None` (call sites at `main.rs:484`, `routes.rs:1191`, provider unit tests stay compiling).

- [ ] **Step 1: Write the failing test** (new `#[cfg(test)] mod test_llm_bearer_tests` in `crates/wenlan-server/src/routes.rs`, alongside the existing test mods)

```rust
#[cfg(test)]
mod test_llm_bearer_tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::{Arc, Mutex};
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    use crate::state::ServerState;

    /// Mock OpenAI-compatible server capturing the Authorization header.
    async fn spawn_mock() -> (std::net::SocketAddr, Arc<Mutex<Vec<Option<String>>>>) {
        let captured: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let cap = captured.clone();
        let app = axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(move |headers: axum::http::HeaderMap| {
                let cap = cap.clone();
                async move {
                    cap.lock().unwrap().push(
                        headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .map(String::from),
                    );
                    axum::Json(serde_json::json!({
                        "choices": [{"message": {"content": "hello"}}]
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (addr, captured)
    }

    async fn probe(addr: std::net::SocketAddr, body: serde_json::Value) -> StatusCode {
        let state = Arc::new(RwLock::new(ServerState::default()));
        let router = crate::router::build_router(state);
        let mut body = body;
        body["endpoint"] = serde_json::json!(format!("http://{addr}"));
        body["model"] = serde_json::json!("test-model");
        router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/llm/test")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn test_llm_forwards_bearer_key() {
        let (addr, captured) = spawn_mock().await;
        let status = probe(addr, serde_json::json!({"api_key": "sk-test"})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            captured.lock().unwrap().as_slice(),
            &[Some("Bearer sk-test".to_string())]
        );
    }

    #[tokio::test]
    async fn test_llm_sends_no_auth_header_without_key() {
        let (addr, captured) = spawn_mock().await;
        let status = probe(addr, serde_json::json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(captured.lock().unwrap().as_slice(), &[None]);
    }
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p wenlan-server test_llm_bearer` → FAIL: `test_llm_forwards_bearer_key` gets `[None]` (no header support yet). (If the mod doesn't compile because `api_key` is unused in the handler, that's the same missing implementation.)

- [ ] **Step 3: Implement the provider key**

`llm_provider.rs` — add field + constructor; keep `new` delegating:

```rust
pub struct OpenAICompatibleProvider {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl OpenAICompatibleProvider {
    pub fn new(endpoint: String, model: String) -> Self {
        Self::new_with_key(endpoint, model, None)
    }

    pub fn new_with_key(endpoint: String, model: String, api_key: Option<String>) -> Self {
        // Ensure endpoint doesn't have trailing slash
        let endpoint = endpoint.trim_end_matches('/').to_string();
        Self {
            endpoint,
            model,
            api_key,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60)) // longer timeout for local models
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
    // endpoint() / model() unchanged
}
```

In `generate()`, replace the fixed request chain with:

```rust
let url = format!("{}/chat/completions", self.endpoint);
let mut request = self
    .client
    .post(&url)
    .header("content-type", "application/json")
    .json(&body);
if let Some(key) = self.api_key.as_deref().filter(|k| !k.trim().is_empty()) {
    request = request.header("authorization", format!("Bearer {key}"));
}
let resp = request
    .send()
    .await
    .map_err(|e| LlmError::InferenceFailed(format!("Request failed: {}", e)))?;
```

`routes.rs::handle_test_llm` — replace the provider construction:

```rust
let provider = OpenAICompatibleProvider::new_with_key(req.endpoint, req.model, req.api_key);
```

- [ ] **Step 4: Verify** — `cargo test -p wenlan-server test_llm_bearer && cargo test -p wenlan-core llm_provider` → PASS (existing endpoint-trimming tests still green).

- [ ] **Step 5: Commit** — `git commit -am "feat(core,server): Bearer auth for OpenAI-compatible provider, testable via /api/llm/test"`

---

### Task 4: Key lifecycle + hot-swap in `PUT /api/config`

**Files:**
- Modify: `crates/wenlan-server/src/config_routes.rs` (`config_to_response` :15, `handle_update_config` :38, new `apply_external_provider`, new test mod)
- Router registration (`crates/wenlan-server/src/router.rs:285`) needs no change — adding a `State` extractor to the handler is transparent.

**Interfaces:**
- Produces: `fn apply_external_provider(state: &mut crate::state::ServerState, cfg: &config::Config)` — rebuilds or clears `state.external_llm`; reused by Task 5's status test and mirrored by `main.rs` startup (left as-is; startup path already builds the provider — extend it with the key, see Step 3).

- [ ] **Step 1: Write the failing tests** (new `#[cfg(test)] mod external_llm_lifecycle_tests` in `config_routes.rs`, copying the `DataDirGuard` + `TEST_DATA_DIR_LOCK` pattern from the existing `config_model_fields_tests` mod verbatim)

```rust
    async fn put_config(
        app: &axum::Router,
        body: serde_json::Value,
    ) -> (StatusCode, Value) {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        (status, response_json(resp).await)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn external_key_lifecycle_and_hot_swap() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = DataDirGuard::new();
        let state = std::sync::Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state.clone());

        // 1. Set endpoint + model + key: hot-swap ON, flag true, value never echoed.
        let (status, body) = put_config(
            &app,
            serde_json::json!({
                "external_llm_endpoint": "http://localhost:11434/v1",
                "external_llm_model": "llama3",
                "external_llm_api_key": "sk-secret-123"
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["external_llm_api_key_configured"], true);
        assert!(
            !body.to_string().contains("sk-secret-123"),
            "key value must never be serialized"
        );
        assert!(state.read().await.external_llm.is_some(), "hot-swap must load the slot");

        // 2. Omitted field preserves the stored key.
        let (_, body) = put_config(&app, serde_json::json!({"clipboard_enabled": true})).await;
        assert_eq!(body["external_llm_api_key_configured"], true);
        assert_eq!(
            config::load_config().external_llm_api_key.as_deref(),
            Some("sk-secret-123")
        );

        // 3. Explicit null clears the key (endpoint+model remain -> slot stays loaded, keyless).
        let (_, body) = put_config(&app, serde_json::json!({"external_llm_api_key": null})).await;
        assert_eq!(body["external_llm_api_key_configured"], false);
        assert!(config::load_config().external_llm_api_key.is_none());
        assert!(state.read().await.external_llm.is_some());

        // 4. Empty string also clears.
        put_config(&app, serde_json::json!({"external_llm_api_key": "sk-2"})).await;
        let (_, body) = put_config(&app, serde_json::json!({"external_llm_api_key": ""})).await;
        assert_eq!(body["external_llm_api_key_configured"], false);

        // 5. Clearing the endpoint clears the slot.
        let (_, body) = put_config(&app, serde_json::json!({"external_llm_endpoint": ""})).await;
        assert_eq!(body["external_llm_endpoint"], Value::Null);
        assert!(state.read().await.external_llm.is_none());
    }
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p wenlan-server external_llm_lifecycle` → FAIL (compile error: handler has no `State`; or assertion 1 fails on hot-swap).

- [ ] **Step 3: Implement**

`config_to_response` — replace the Task 2 placeholder:

```rust
external_llm_api_key_configured: cfg
    .external_llm_api_key
    .as_deref()
    .map(|k| !k.trim().is_empty())
    .unwrap_or(false),
```

New helper (below `apply_anthropic_provider`):

```rust
/// (Re)build or clear the external OpenAI-compatible provider from config.
/// Mirrors `apply_anthropic_provider` so `PUT /api/config` hot-swaps the slot.
fn apply_external_provider(state: &mut crate::state::ServerState, cfg: &config::Config) {
    match (&cfg.external_llm_endpoint, &cfg.external_llm_model) {
        (Some(endpoint), Some(model)) if !endpoint.is_empty() && !model.is_empty() => {
            state.external_llm = Some(Arc::new(
                wenlan_core::llm_provider::OpenAICompatibleProvider::new_with_key(
                    endpoint.clone(),
                    model.clone(),
                    cfg.external_llm_api_key.clone(),
                ),
            ));
        }
        _ => {
            state.external_llm = None;
        }
    }
}
```

`handle_update_config` — new signature and external-field semantics:

```rust
pub async fn handle_update_config(
    State(state): State<SharedState>,
    Json(req): Json<UpdateConfigRequest>,
) -> Result<Json<ConfigResponse>, ServerError> {
    let mut cfg = config::load_config();
    let external_touched = req.external_llm_endpoint.is_some()
        || req.external_llm_model.is_some()
        || req.external_llm_api_key.is_some();
    // ... existing field applications unchanged, EXCEPT the two external fields:
    if let Some(v) = req.external_llm_endpoint {
        cfg.external_llm_endpoint = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.external_llm_model {
        cfg.external_llm_model = if v.is_empty() { None } else { Some(v) };
    }
    // Key lifecycle contract: omitted = preserve; null/"" = clear; value = replace.
    match req.external_llm_api_key {
        None => {}
        Some(None) => cfg.external_llm_api_key = None,
        Some(Some(v)) => {
            cfg.external_llm_api_key = if v.trim().is_empty() { None } else { Some(v) };
        }
    }
    config::save_config(&cfg).map_err(|e| ServerError::Internal(e.to_string()))?;
    if external_touched {
        let mut s = state.write().await;
        apply_external_provider(&mut s, &cfg);
    }
    Ok(Json(config_to_response(&cfg)))
}
```

`main.rs:479-491` startup block — pass the key through:

```rust
let provider = wenlan_core::llm_provider::OpenAICompatibleProvider::new_with_key(
    endpoint.clone(),
    model.clone(),
    config.external_llm_api_key.clone(),
);
```

- [ ] **Step 4: Verify** — `cargo test -p wenlan-server` → PASS, including the pre-existing `put_config_round_trips_model_fields` test (its handler call gains the State extractor transparently through the router).

- [ ] **Step 5: Commit** — `git commit -am "feat(server): external-LLM key lifecycle + hot-swap on PUT /api/config"`

---

### Task 5: `external_llm` status in `GET /api/setup/status`

**Files:**
- Modify: `crates/wenlan-server/src/config_routes.rs` (`SetupStatusResponse` :108, `handle_get_setup_status` :156, existing `setup_status_tests`)

**Interfaces:**
- Produces (additive JSON): `"external_llm": {"configured": bool, "loaded": bool}` — `configured` = endpoint+model non-empty in config; `loaded` = `state.external_llm.is_some()`. Consumed by the app's active-intelligence strip (spec §1).

- [ ] **Step 1: Write the failing test** (extend `setup_status_tests`)

```rust
    #[tokio::test(flavor = "current_thread")]
    async fn setup_status_reports_external_llm_state() {
        let _lock = crate::TEST_DATA_DIR_LOCK
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await;
        let _env = WenlanDataDirGuard::new();
        let state = Arc::new(RwLock::new(ServerState::default()));
        let app = crate::router::build_router(state.clone());

        let resp = app
            .clone()
            .oneshot(Request::builder().method("GET").uri("/api/setup/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = response_json(resp).await;
        assert_eq!(body["external_llm"]["configured"], false);
        assert_eq!(body["external_llm"]["loaded"], false);

        // Configure via PUT /api/config -> hot-swap makes it configured AND loaded.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/api/config")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"external_llm_endpoint":"http://localhost:11434/v1","external_llm_model":"llama3"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = app
            .oneshot(Request::builder().method("GET").uri("/api/setup/status").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = response_json(resp).await;
        assert_eq!(body["external_llm"]["configured"], true);
        assert_eq!(body["external_llm"]["loaded"], true);
    }
```

Also extend `setup_status_defaults_to_basic_memory` with the two `false` assertions.

- [ ] **Step 2: Run to verify failure** — `cargo test -p wenlan-server setup_status` → FAIL (`external_llm` is null).

- [ ] **Step 3: Implement**

```rust
#[derive(Debug, Serialize)]
pub struct ExternalLlmStatus {
    pub configured: bool,
    pub loaded: bool,
}
```

Add `pub external_llm: ExternalLlmStatus,` to `SetupStatusResponse`. In `handle_get_setup_status`, inside the existing state read block also capture `let external_loaded = s.external_llm.is_some();`, then:

```rust
let external_configured = matches!(
    (&cfg.external_llm_endpoint, &cfg.external_llm_model),
    (Some(e), Some(m)) if !e.is_empty() && !m.is_empty()
);
```

and set `external_llm: ExternalLlmStatus { configured: external_configured, loaded: external_loaded },` in the response.

- [ ] **Step 4: Verify** — `cargo test -p wenlan-server setup_status` → PASS.

- [ ] **Step 5: Commit** — `git commit -am "feat(server): report external_llm {configured,loaded} in setup status"`

---

### Task 6: Workspace gates + draft PR

- [ ] **Step 1: Full gates**

```bash
cargo fmt --all && cargo fmt --check --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: all green. Fix warnings properly — never `#[allow]` them away to pass the gate.

- [ ] **Step 2: Push + draft PR** (run `gh` with the sandbox disabled)

```bash
git push -u origin feat/external-llm-key
gh pr create --draft --base main --head feat/external-llm-key \
  --title "feat: external-LLM API key (Bearer), config hot-swap, external status" \
  --body "<summary of the four surfaces + key-lifecycle contract; link the wenlan-app spec; note release-please will cut 0.13.0; standard generated-with footer.>"
```

Expected: PR URL. Independently shippable; the app's keyed presets stay version-gated until the app's backend pin bumps to 0.13 (explicitly out of scope).
