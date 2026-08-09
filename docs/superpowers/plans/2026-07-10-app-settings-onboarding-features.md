# App Settings + Onboarding Features Implementation Plan (PR 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the daemon's multi-provider intelligence (Any-provider preset card, version-gated API keys), 3-way wizard intelligence choice, Obsidian vault connect with a real recursive check, and a connect-everywhere platform matrix (web + apps/CLIs incl. Codex TOML).

**Architecture:** Stacked on the `settings-decomposition` branch (PR 1 of 3, plan `2026-07-10-settings-decomposition.md`). All UI work happens in the post-decomposition section files. New Rust surface is three small additions to the app crate (`list_external_models`, api-key passthrough, MCP registry extension); vendor presets are frontend data. Everything must be fully functional against pinned daemon **0.12** — daemon 0.13 features (API key, hot-swap, serving status) light up via `useDaemonVersion()`.

**Tech Stack:** React 19 + Vite 6 + Tailwind v4 (CSS-first, `--mem-*` tokens, inline styles), TanStack Query, react-i18next, Vitest + jsdom; Tauri 2 Rust app crate (`app/`), reqwest 0.12, `toml_edit` (new dep, council change d).

**Spec:** `docs/superpowers/specs/2026-07-10-settings-onboarding-redesign-design.md` §1, §2, §2a, §3, §5, §6, §8.

## Global Constraints

- Branch `settings-onboarding-features`, cut from `settings-decomposition`. Draft PR bases on `settings-decomposition`, not main.
- **Daemon 0.12 is the runtime floor**: keyless local presets (Ollama `http://localhost:11434/v1`, LM Studio `http://localhost:1234/v1`) fully work; keyed cloud presets are disabled-with-explanation below daemon 0.13; save shows "Restart Wenlan to apply" below 0.13 (no hot-swap).
- Version gate: `supportsExternalKey` / `supportsHotSwap` = daemon semver ≥ `0.13.0` from `GET /api/health`. Fetch failure ⇒ conservative (treat as < 0.13). The app **never sends** `api_key` / `external_llm_api_key` fields to a daemon < 0.13.
- Design tokens only: `--mem-*` CSS vars, `var(--mem-font-heading)` (Fraunces) for headings, `var(--mem-font-body)` (Instrument Sans) for body, `var(--mem-font-mono)` (JetBrains Mono) for paths/keys/URLs. No new tokens, no `★`, no gradients. Light + dark both work automatically when only tokens are used.
- **i18n:** every user-facing string goes through `t()` with keys in `src/i18n/resources.ts` — in **all three locales (en, zh-Hans, zh-Hant) in the same commit** so `pnpm test:i18n` stays green at every task boundary. Vendor names (OpenAI, Ollama…), URLs, and model IDs are technical literals and stay hardcoded.
- react-jsx transform: **never** add `import React from "react"`. Type positions may use the global `React.` namespace (existing style, see `SetupWizard.tsx:300`).
- All frontend IPC goes through `src/lib/tauri.ts` wrappers — never call `invoke` elsewhere.
- Gates per task: named test file passes. Gates before PR (Task 12): `pnpm build`, `pnpm test`, `pnpm test:i18n`, and in `app/`: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test`.
- Commit after every task (`git add <files> && git commit`).

## File Structure (new/modified)

```text
src/
|-- hooks/useDaemonVersion.ts                          (new, Task 1)
|-- hooks/useDaemonVersion.test.ts                     (new, Task 1)
|-- lib/tauri.ts                                       (extend, Tasks 1-3)
|-- lib/vaultDetection.ts                              (new, Task 5)
|-- lib/vaultDetection.test.ts                         (new, Task 5)
|-- components/intelligence/providerPresets.ts         (new, Task 7)
|-- components/intelligence/AnyProviderCard.tsx        (new, Task 7)
|-- components/intelligence/AnyProviderCard.test.tsx   (new, Task 7)
|-- components/intelligence/ActiveIntelligenceStrip.tsx (new, Task 8)
|-- components/memory/settings/sections/IntelligenceSection.tsx (modify, Task 8)
|-- components/memory/settings/sections/AgentsSection.tsx       (modify, Task 11)
|-- components/memory/sources/VaultConnectCard.tsx     (new, Task 6)
|-- components/memory/sources/AddSourceDialog.tsx      (modify, Task 6)
|-- components/connect/WebPlatformCards.tsx            (new, Task 11)
|-- components/connect/ClientSetupList.tsx             (new, Task 11)
|-- components/SetupWizard.tsx                         (modify, Tasks 9-11)
|-- i18n/resources.ts                                  (extend, Tasks 6-11)
app/
|-- Cargo.toml                                         (add toml_edit, Task 4)
|-- src/search.rs                                      (extend, Tasks 2-4)
|-- src/api.rs                                         (extend, Task 3)
|-- src/mcp_config.rs                                  (extend, Task 4)
|-- src/lib.rs                                         (register commands, Tasks 2-3)
```

---

### Task 1: Branch + `useDaemonVersion()` hook + SetupStatus type extension

**Files:**
- Create: `src/hooks/useDaemonVersion.ts`
- Test: `src/hooks/useDaemonVersion.test.ts`
- Modify: `src/lib/tauri.ts` (SetupStatus interface, ~line 1917)

**Interfaces:**
- Consumes: existing `getDaemonVersion()` (`src/lib/tauri.ts:180`) and `daemonMeetsFloor(version, floor)` (`src/lib/tauri.ts:185`).
- Produces: `useDaemonVersion(): { version: string | null; supportsExternalKey: boolean; supportsHotSwap: boolean }` — used by Tasks 7, 8. `SetupStatus.external_llm?: { configured: boolean; loaded: boolean } | null` — used by Task 8.

- [ ] **Step 1: Create the branch (stacked on PR 1)**

```bash
git checkout settings-decomposition
git checkout -b settings-onboarding-features
```

If `settings-decomposition` does not exist yet (PR 1 not started), stop and report — this plan depends on it.

- [ ] **Step 2: Write the failing test**

`src/hooks/useDaemonVersion.test.ts`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

const mocks = vi.hoisted(() => ({ getDaemonVersion: vi.fn() }));
vi.mock("../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/tauri")>();
  return { ...actual, getDaemonVersion: mocks.getDaemonVersion };
});

import { useDaemonVersion } from "./useDaemonVersion";

function wrapper({ children }: { children: ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe("useDaemonVersion", () => {
  beforeEach(() => mocks.getDaemonVersion.mockReset());

  it("reports 0.13+ as supporting external key and hot-swap", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    const { result } = renderHook(() => useDaemonVersion(), { wrapper });
    await waitFor(() => expect(result.current.version).toBe("0.13.0"));
    expect(result.current.supportsExternalKey).toBe(true);
    expect(result.current.supportsHotSwap).toBe(true);
  });

  it("reports 0.12 as not supporting either", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.12.0");
    const { result } = renderHook(() => useDaemonVersion(), { wrapper });
    await waitFor(() => expect(result.current.version).toBe("0.12.0"));
    expect(result.current.supportsExternalKey).toBe(false);
    expect(result.current.supportsHotSwap).toBe(false);
  });

  it("is conservative when the health fetch fails", async () => {
    mocks.getDaemonVersion.mockRejectedValue(new Error("daemon down"));
    const { result } = renderHook(() => useDaemonVersion(), { wrapper });
    await waitFor(() => expect(mocks.getDaemonVersion).toHaveBeenCalled());
    expect(result.current.version).toBeNull();
    expect(result.current.supportsExternalKey).toBe(false);
    expect(result.current.supportsHotSwap).toBe(false);
  });
});
```

Note the file must be `.test.tsx` if JSX in wrapper complains under the project's Vitest config — check sibling tests (`src/hooks/useSearch.test.ts` is plain `.ts`); the wrapper above uses JSX, so name the file `useDaemonVersion.test.tsx`.

- [ ] **Step 3: Run test to verify it fails**

Run: `pnpm vitest run src/hooks/useDaemonVersion.test.tsx`
Expected: FAIL — `Cannot find module './useDaemonVersion'`

- [ ] **Step 4: Write the hook**

`src/hooks/useDaemonVersion.ts`:

```ts
import { useQuery } from "@tanstack/react-query";
import { getDaemonVersion, daemonMeetsFloor } from "../lib/tauri";

export interface DaemonVersionInfo {
  version: string | null;
  /** Daemon accepts `external_llm_api_key` / `api_key` fields (≥ 0.13). */
  supportsExternalKey: boolean;
  /** `PUT /api/config` hot-swaps the external provider (≥ 0.13). */
  supportsHotSwap: boolean;
}

/** Version gate for daemon-0.13 features (spec §8). Conservative on failure:
 *  an unreachable daemon reports both capabilities as false. */
export function useDaemonVersion(): DaemonVersionInfo {
  const { data } = useQuery({
    queryKey: ["daemon-version"],
    queryFn: getDaemonVersion,
    staleTime: 60_000,
    retry: 1,
  });
  const version = data ?? null;
  const atLeast013 = version !== null && daemonMeetsFloor(version, "0.13.0");
  return { version, supportsExternalKey: atLeast013, supportsHotSwap: atLeast013 };
}
```

- [ ] **Step 5: Extend `SetupStatus` in `src/lib/tauri.ts`** — add one optional field to the existing interface (near line 1917):

```ts
export interface SetupStatus {
  setup_completed: boolean;
  mode: "basic-memory" | "local-model" | "anthropic-key" | string;
  anthropic_key_configured: boolean;
  local_model_selected: string | null;
  local_model_loaded: string | null;
  local_model_cached: boolean;
  /** Daemon ≥ 0.13 only (additive, spec §7.6); absent on 0.12. */
  external_llm?: { configured: boolean; loaded: boolean } | null;
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pnpm vitest run src/hooks/useDaemonVersion.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```bash
git add src/hooks/useDaemonVersion.ts src/hooks/useDaemonVersion.test.tsx src/lib/tauri.ts
git commit -m "feat: useDaemonVersion gate for daemon-0.13 features"
```

---

### Task 2: `list_external_models` Tauri command (model auto-discovery)

**Files:**
- Modify: `app/src/search.rs` (add after `test_external_llm`, ~line 3800)
- Modify: `app/src/lib.rs` (register after `search::test_external_llm,` line 1008)
- Modify: `src/lib/tauri.ts` (wrapper next to `testExternalLlm`, ~line 418)

**Interfaces:**
- Produces: Rust `list_external_models(endpoint: String, api_key: Option<String>) -> Result<Vec<String>, String>`; TS `listExternalModels(endpoint: string, apiKey?: string | null): Promise<string[]>` — used by Task 7's AnyProviderCard.
- `parse_models_response(&serde_json::Value) -> Vec<String>` — pure, unit-tested.

- [ ] **Step 1: Write the failing tests** — in `app/src/search.rs`, add at the end of the file (or into an existing `#[cfg(test)]` module in that file if one is adjacent):

```rust
#[cfg(test)]
mod list_external_models_tests {
    use super::*;

    #[test]
    fn parses_openai_models_shape() {
        let body = serde_json::json!({
            "object": "list",
            "data": [
                {"id": "llama3.2:3b", "object": "model"},
                {"id": "qwen2.5-coder", "object": "model"}
            ]
        });
        assert_eq!(
            parse_models_response(&body),
            vec!["llama3.2:3b".to_string(), "qwen2.5-coder".to_string()]
        );
    }

    #[test]
    fn missing_or_malformed_data_yields_empty() {
        assert!(parse_models_response(&serde_json::json!({})).is_empty());
        assert!(parse_models_response(&serde_json::json!({"data": "nope"})).is_empty());
        assert!(parse_models_response(&serde_json::json!({"data": [{"name": "no-id"}]})).is_empty());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd app && cargo test list_external_models_tests`
Expected: FAIL — `cannot find function parse_models_response`

- [ ] **Step 3: Implement** — in `app/src/search.rs`, directly after the `test_external_llm` command (~line 3800):

```rust
/// Parse an OpenAI-compatible `GET {endpoint}/models` body into model IDs.
pub(crate) fn parse_models_response(body: &serde_json::Value) -> Vec<String> {
    body.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Model auto-discovery for the Any-provider card (spec §1, §6). Talks to the
/// provider directly (not the daemon) so discovery works before saving.
#[tauri::command]
pub async fn list_external_models(
    endpoint: String,
    api_key: Option<String>,
) -> Result<Vec<String>, String> {
    let base = endpoint.trim_end_matches('/');
    if !(base.starts_with("http://") || base.starts_with("https://")) {
        return Err("Endpoint must start with http:// or https://".to_string());
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut req = client.get(format!("{base}/models"));
    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("{} from {base}/models", resp.status()));
    }
    let body: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(parse_models_response(&body))
}
```

Register in `app/src/lib.rs` — after `search::test_external_llm,` (line 1008):

```rust
            search::list_external_models,
```

- [ ] **Step 4: Run tests + clippy**

Run: `cd app && cargo test list_external_models_tests && cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS, no warnings

- [ ] **Step 5: Add the TS wrapper** — `src/lib/tauri.ts`, after `testExternalLlm` (~line 418):

```ts
/** Model auto-discovery: GET {endpoint}/models (OpenAI `{data:[{id}]}` shape),
 *  Bearer auth when a key is given, 5s timeout. Spec §1. */
export async function listExternalModels(
  endpoint: string,
  apiKey?: string | null
): Promise<string[]> {
  return invoke("list_external_models", { endpoint, apiKey: apiKey ?? null });
}
```

(Tauri 2 maps camelCase invoke args to snake_case Rust params — same convention as `set_model_choice` above it.)

- [ ] **Step 6: Type-check and commit**

Run: `pnpm exec tsc -b`
Expected: clean

```bash
git add app/src/search.rs app/src/lib.rs src/lib/tauri.ts
git commit -m "feat: list_external_models command for provider model discovery"
```

---

### Task 3: API-key passthrough — `set_external_llm` / `test_external_llm` / presence flag

The app pins wenlan-types 0.12, whose `UpdateConfigRequest`/`TestLlmRequest` lack the new key fields — so the key rides in hand-built `serde_json` bodies, added **only when the caller supplies a key**. The frontend only supplies a key when `supportsExternalKey` (spec §6: "harmless extra field against 0.12 is NOT assumed").

Key semantics over this boundary (spec §7.2): `None`/omitted ⇒ preserve stored key; `Some("")` ⇒ clear; `Some(key)` ⇒ replace.

**Files:**
- Modify: `app/src/api.rs` (`test_llm` ~line 598, `set_external_llm` ~line 703, new `external_llm_key_configured`)
- Modify: `app/src/search.rs` (`set_external_llm` ~line 3774, `test_external_llm` ~line 3789, new command)
- Modify: `app/src/lib.rs` (register new command after line 1008)
- Modify: `src/lib/tauri.ts` (`setExternalLlm` ~line 399, `testExternalLlm` ~line 410, new wrapper)

**Interfaces:**
- Produces (TS, used by Task 7):
  - `setExternalLlm(endpoint: string | null, model: string | null, apiKey?: string | null): Promise<void>`
  - `testExternalLlm(endpoint: string, model: string, apiKey?: string | null): Promise<TestLlmResponse>`
  - `getExternalLlmKeyConfigured(): Promise<boolean>` — false on 0.12 (field absent).

- [ ] **Step 1: Update `WenlanClient::test_llm`** (`app/src/api.rs:598`) — replace the whole method:

```rust
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
```

- [ ] **Step 2: Update `WenlanClient::set_external_llm`** (`app/src/api.rs:703`) — replace the whole method:

```rust
    /// Patch daemon external LLM config. `None` endpoint/model preserves the
    /// existing daemon value. `api_key`: `None` = omit (preserve stored key),
    /// `Some("")` = clear, `Some(key)` = replace (spec §7.2 tri-state).
    pub async fn set_external_llm(
        &self,
        endpoint: Option<String>,
        model: Option<String>,
        api_key: Option<String>,
    ) -> Result<(), String> {
        let mut body =
            sparse_update_config(empty_update().with_external_llm(endpoint, model))?;
        if let Some(key) = api_key {
            body["external_llm_api_key"] = serde_json::Value::String(key);
        }
        let _resp: wenlan_types::responses::ConfigResponse =
            self.put_json("/api/config", &body).await?;
        Ok(())
    }
```

- [ ] **Step 3: Add `WenlanClient::external_llm_key_configured`** — next to `get_external_llm` (~line 697):

```rust
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
```

- [ ] **Step 4: Update the Tauri commands** in `app/src/search.rs` — replace `set_external_llm` (~3774) and `test_external_llm` (~3789), add the new command after them:

```rust
#[tauri::command]
pub async fn set_external_llm(
    state: tauri::State<'_, State>,
    endpoint: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
) -> Result<(), String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.set_external_llm(endpoint, model, api_key).await?;
    log::info!("[settings] External LLM config updated");
    Ok(())
}

#[tauri::command]
pub async fn test_external_llm(
    state: tauri::State<'_, State>,
    endpoint: String,
    model: String,
    api_key: Option<String>,
) -> Result<requests::TestLlmResponse, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.test_llm(endpoint, model, api_key).await
}

#[tauri::command]
pub async fn get_external_llm_key_configured(
    state: tauri::State<'_, State>,
) -> Result<bool, String> {
    let client = {
        let s = state.read().await;
        s.client.clone()
    };
    client.external_llm_key_configured().await
}
```

The existing `external_llm_command_type_tests` module (~line 3802) calls `test_external_llm(state, String::new(), String::new())` — update it to `test_external_llm(state, String::new(), String::new(), None)`.

Register in `app/src/lib.rs` after `search::list_external_models,`:

```rust
            search::get_external_llm_key_configured,
```

- [ ] **Step 5: Compile + test the app crate**

Run: `cd app && cargo test && cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS (existing api.rs tests confirm sparse-body behavior still holds), no warnings

- [ ] **Step 6: Update TS wrappers** in `src/lib/tauri.ts` — replace `setExternalLlm` (~399) and `testExternalLlm` (~410), add the new wrapper after them:

```ts
// The daemon config endpoint is patch-based: null endpoint/model preserves the
// current value. apiKey tri-state (spec §7.2): undefined/null = preserve stored
// key, "" = clear, non-empty = replace. Only pass a key when
// useDaemonVersion().supportsExternalKey — 0.12 daemons must never see it.
export async function setExternalLlm(
  endpoint: string | null,
  model: string | null,
  apiKey?: string | null
): Promise<void> {
  return invoke("set_external_llm", { endpoint, model, apiKey: apiKey ?? null });
}

export interface TestLlmResponse {
  response: string;
}

export async function testExternalLlm(
  endpoint: string,
  model: string,
  apiKey?: string | null
): Promise<TestLlmResponse> {
  return invoke("test_external_llm", { endpoint, model, apiKey: apiKey ?? null });
}

/** True when the daemon (≥ 0.13) has a stored external-provider API key.
 *  Always false on 0.12. The key value itself is never readable. */
export async function getExternalLlmKeyConfigured(): Promise<boolean> {
  return invoke("get_external_llm_key_configured");
}
```

(Keep the existing `TestLlmResponse` interface — shown for placement; don't duplicate it.)

- [ ] **Step 7: Type-check, run frontend tests, commit**

Run: `pnpm exec tsc -b && pnpm vitest run src/lib`
Expected: clean / PASS

```bash
git add app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts
git commit -m "feat: external LLM api_key passthrough + key presence flag (0.13-gated)"
```

---

### Task 4: MCP client registry extension — Claude Desktop, Gemini CLI, Codex CLI (TOML)

Claude Desktop and Gemini CLI reuse the existing JSON `mcpServers` writer. Codex CLI needs a new **format-preserving** TOML writer via `toml_edit` (council change d): comments, key order, and unrelated tables survive byte-for-byte.

**Files:**
- Modify: `app/Cargo.toml` (add `toml_edit`)
- Modify: `app/src/mcp_config.rs` (paths :22, detection :53, new TOML fns, tests :170+)
- Modify: `app/src/search.rs` (`write_mcp_config` routing)

**Interfaces:**
- Consumes: `wenlan_mcp_entry()` (`mcp_config.rs:110`), `MCP_SERVER_KEY = "wenlan"`, `LEGACY_MCP_SERVER_KEY = "origin"`.
- Produces: `client_config_path` supports `"gemini_cli"`, `"codex_cli"`; `detect_mcp_clients()` returns 5 clients; `write_wenlan_entry_toml(&Path) -> Result<(), AppError>`; `has_configured_entry_toml(&str) -> bool`. Frontend picks all of this up automatically through existing `detectMcpClients()`/`writeMcpConfig()` wrappers — no TS changes.

- [ ] **Step 1: Add the dependency** — in `app/Cargo.toml` `[dependencies]`, after `toml = "0.8"` (line 41):

```toml
toml_edit = "0.22"
```

- [ ] **Step 2: Write the failing tests** — append inside `mod tests` in `app/src/mcp_config.rs`:

```rust
    #[test]
    fn test_client_config_path_gemini_cli() {
        let path = client_config_path("gemini_cli").unwrap();
        assert!(path.to_string_lossy().ends_with(".gemini/settings.json"));
    }

    #[test]
    fn test_client_config_path_codex_cli() {
        let path = client_config_path("codex_cli").unwrap();
        assert!(path.to_string_lossy().ends_with(".codex/config.toml"));
    }

    #[test]
    fn test_detect_includes_new_clients() {
        let types: Vec<String> = detect_mcp_clients()
            .into_iter()
            .map(|c| c.client_type)
            .collect();
        for expected in ["cursor", "claude_code", "claude_desktop", "gemini_cli", "codex_cli"] {
            assert!(types.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn test_has_configured_entry_toml() {
        assert!(has_configured_entry_toml(
            "[mcp_servers.wenlan]\ncommand = \"npx\"\nargs = [\"-y\", \"wenlan-mcp\"]\n"
        ));
        assert!(has_configured_entry_toml(
            "[mcp_servers.origin]\ncommand = \"npx\"\n"
        ));
        assert!(!has_configured_entry_toml("[mcp_servers.other]\ncommand = \"x\"\n"));
        assert!(!has_configured_entry_toml("model = \"gpt-5.5\"\n"));
        assert!(!has_configured_entry_toml("not toml ["));
    }

    #[test]
    fn test_write_wenlan_entry_toml_creates_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        write_wenlan_entry_toml(&config_path).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        assert!(has_configured_entry_toml(&contents));
        let parsed: toml::Value = toml::from_str(&contents).unwrap();
        let wenlan = &parsed["mcp_servers"]["wenlan"];
        assert!(wenlan.get("command").is_some());
    }

    #[test]
    fn test_write_wenlan_entry_toml_preserves_formatting_byte_for_byte() {
        // Council change (d): a user's hand-edited config must survive the
        // upsert byte-for-byte — comments, spacing, key order, other tables.
        let fixture = r#"# my codex config — do not touch
model = "gpt-5.5"   # inline comment

[profiles.fast]
model   = "gpt-5.5-mini"

[mcp_servers.other]
command = "other-cmd"  # keep me
args = ["--flag"]
"#;
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(&config_path, fixture).unwrap();
        write_wenlan_entry_toml(&config_path).unwrap();
        let contents = std::fs::read_to_string(&config_path).unwrap();
        // Everything that existed before is preserved verbatim; the wenlan
        // table is appended after it.
        assert!(
            contents.starts_with(fixture),
            "existing content was reformatted:\n{contents}"
        );
        assert!(has_configured_entry_toml(&contents));
    }

    #[test]
    fn test_write_wenlan_entry_toml_upsert_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        write_wenlan_entry_toml(&config_path).unwrap();
        let first = std::fs::read_to_string(&config_path).unwrap();
        write_wenlan_entry_toml(&config_path).unwrap();
        let second = std::fs::read_to_string(&config_path).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn test_write_wenlan_entry_toml_creates_backup() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(&config_path, "model = \"gpt-5.5\"\n").unwrap();
        write_wenlan_entry_toml(&config_path).unwrap();
        let backup = tmp.path().join("config.toml.bak");
        assert!(backup.exists());
        assert!(std::fs::read_to_string(&backup).unwrap().contains("gpt-5.5"));
    }

    #[test]
    fn test_write_wenlan_entry_toml_errors_on_invalid_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let config_path = tmp.path().join("config.toml");
        std::fs::write(&config_path, "not toml [").unwrap();
        assert!(write_wenlan_entry_toml(&config_path).is_err());
    }
```

- [ ] **Step 3: Run to verify failure**

Run: `cd app && cargo test mcp_config`
Expected: FAIL — `cannot find function has_configured_entry_toml` / `write_wenlan_entry_toml`, path tests fail with `None`

- [ ] **Step 4: Implement.** In `app/src/mcp_config.rs`:

Replace the `client_config_path` match arms (lines 24-31) with:

```rust
    match client_type {
        "claude_desktop" => {
            // macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
            dirs::config_dir().map(|d| d.join("Claude").join("claude_desktop_config.json"))
        }
        "cursor" => Some(home.join(".cursor").join("mcp.json")),
        "claude_code" => Some(home.join(".claude.json")),
        "gemini_cli" => Some(home.join(".gemini").join("settings.json")),
        "codex_cli" => Some(home.join(".codex").join("config.toml")),
        _ => None,
    }
```

Replace `detect_mcp_clients` (lines 52-94) with:

```rust
/// Detect installed MCP-compatible tools and whether Wenlan is already configured.
pub fn detect_mcp_clients() -> Vec<McpClient> {
    let clients = [
        ("Cursor", "cursor"),
        ("Claude Code", "claude_code"),
        ("Claude Desktop", "claude_desktop"),
        ("Gemini CLI", "gemini_cli"),
        ("Codex CLI", "codex_cli"),
    ];

    clients
        .iter()
        .filter_map(|(name, client_type)| {
            let config_path = client_config_path(client_type)?;
            let config_path_str = config_path.to_string_lossy().to_string();

            let is_toml = *client_type == "codex_cli";
            let config_has_entry = || {
                config_path.exists()
                    && std::fs::read_to_string(&config_path)
                        .map(|s| {
                            if is_toml {
                                has_configured_entry_toml(&s)
                            } else {
                                has_configured_entry(&s)
                            }
                        })
                        .unwrap_or(false)
            };

            let (detected, already_configured) = if client_type == &"cursor" {
                // Cursor: detect by app bundle, not config file
                let app_exists = std::path::Path::new("/Applications/Cursor.app").exists()
                    || dirs::home_dir()
                        .map(|h| h.join("Applications/Cursor.app").exists())
                        .unwrap_or(false);
                (app_exists, config_has_entry())
            } else {
                // Everything else: detect by config file existence
                (config_path.exists(), config_has_entry())
            };

            Some(McpClient {
                name: name.to_string(),
                client_type: client_type.to_string(),
                config_path: config_path_str,
                detected,
                already_configured,
            })
        })
        .collect()
}
```

Add after `has_configured_entry` (line 50):

```rust
/// TOML variant for Codex CLI (`[mcp_servers.*]` tables).
fn has_configured_entry_toml(toml_str: &str) -> bool {
    toml_str
        .parse::<toml_edit::DocumentMut>()
        .ok()
        .and_then(|doc| {
            let servers = doc.get("mcp_servers")?;
            Some(
                servers.get(MCP_SERVER_KEY).is_some()
                    || servers.get(LEGACY_MCP_SERVER_KEY).is_some(),
            )
        })
        .unwrap_or(false)
}
```

Add after `write_wenlan_entry` (line 167):

```rust
/// Upsert the Wenlan entry into a Codex CLI `config.toml` — format-preserving:
/// user comments, key order, and unrelated tables survive byte-for-byte
/// (toml_edit round-trips everything it didn't touch).
pub fn write_wenlan_entry_toml(config_path: &std::path::Path) -> Result<(), AppError> {
    use toml_edit::{DocumentMut, Item, Table};

    let mut doc: DocumentMut = if config_path.exists() {
        let backup_path = config_path.with_extension("toml.bak");
        std::fs::copy(config_path, &backup_path)?;
        let contents = std::fs::read_to_string(config_path)?;
        contents.parse().map_err(|e| {
            AppError::Generic(format!("Invalid TOML in {}: {}", config_path.display(), e))
        })?
    } else {
        DocumentMut::new()
    };

    if doc.get("mcp_servers").is_none() {
        let mut parent = Table::new();
        parent.set_implicit(true); // render only [mcp_servers.wenlan], no bare [mcp_servers]
        doc.insert("mcp_servers", Item::Table(parent));
    }

    let entry = wenlan_mcp_entry();
    let mut server = Table::new();
    server.insert("command", toml_edit::value(entry.command));
    let mut args = toml_edit::Array::new();
    for a in entry.args {
        args.push(a);
    }
    server.insert("args", toml_edit::value(args));
    doc["mcp_servers"][MCP_SERVER_KEY] = Item::Table(server);

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(config_path, doc.to_string())?;
    Ok(())
}
```

- [ ] **Step 5: Route TOML clients in `write_mcp_config`** — in `app/src/search.rs`, replace the command body:

```rust
pub async fn write_mcp_config(client_type: String) -> Result<(), String> {
    let config_path = crate::mcp_config::client_config_path(&client_type)
        .ok_or(format!("Unknown client type: {}", client_type))?;
    if client_type == "codex_cli" {
        return crate::mcp_config::write_wenlan_entry_toml(&config_path)
            .map_err(|e| e.to_string());
    }
    let is_claude_code = client_type == "claude_code";
    crate::mcp_config::write_wenlan_entry(&config_path, is_claude_code).map_err(|e| e.to_string())
}
```

- [ ] **Step 6: Run tests + gates**

Run: `cd app && cargo test mcp_config && cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all mcp_config tests PASS (old + new), no warnings

- [ ] **Step 7: Commit**

```bash
git add app/Cargo.toml app/Cargo.lock app/src/mcp_config.rs app/src/search.rs
git commit -m "feat: MCP registry adds Claude Desktop, Gemini CLI, Codex CLI (format-preserving TOML)"
```

Note: `Cargo.lock` lives at the workspace root (`Cargo.toml` is a workspace wrapper) — `git status` will show the real path; add whichever lockfile changed.

---

### Task 5: `vaultDetection.ts` — recursive vault/notes detection

Fixes the shallow-scan bug (spec §3): recursive walk, depth ≤ 6, ≤ 5,000 entries, dot-directories skipped, validity per **source type** (obsidian ⇒ `.md` only; directory ⇒ `.md`/`.txt`/`.pdf`).

**Files:**
- Create: `src/lib/vaultDetection.ts`
- Test: `src/lib/vaultDetection.test.ts`

**Interfaces:**
- Consumes: `readDir` from `@tauri-apps/plugin-fs` (entries have `name: string`, `isDirectory: boolean`).
- Produces (used by Task 6):

```ts
interface VaultDetection {
  isVault: boolean;                       // `.obsidian/` at root
  sourceType: "obsidian" | "directory";
  docCount: number;                       // files valid for sourceType
  countCapped: boolean;                   // walk hit MAX_ENTRIES → show "5,000+"
  hasValidDoc: boolean;
  unreadable: boolean;                    // root readDir failed — submit stays allowed
}
detectVault(path: string): Promise<VaultDetection>
```

- [ ] **Step 1: Write the failing test** — `src/lib/vaultDetection.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

const mocks = vi.hoisted(() => ({ readDir: vi.fn() }));
vi.mock("@tauri-apps/plugin-fs", () => ({ readDir: mocks.readDir }));

import { detectVault, MAX_ENTRIES } from "./vaultDetection";

type Entry = { name: string; isDirectory: boolean };
const file = (name: string): Entry => ({ name, isDirectory: false });
const dir = (name: string): Entry => ({ name, isDirectory: true });

/** Wire a fake filesystem: map of absolute path → entries. */
function fakeFs(tree: Record<string, Entry[]>) {
  mocks.readDir.mockImplementation(async (p: string) => {
    if (!(p in tree)) throw new Error(`ENOENT: ${p}`);
    return tree[p];
  });
}

describe("detectVault", () => {
  beforeEach(() => mocks.readDir.mockReset());

  it("counts markdown at the top level of an Obsidian vault", async () => {
    fakeFs({ "/v": [dir(".obsidian"), file("a.md"), file("b.md"), file("c.pdf")] });
    const d = await detectVault("/v");
    expect(d.isVault).toBe(true);
    expect(d.sourceType).toBe("obsidian");
    // obsidian sources are markdown-only (daemon has_any_markdown) — c.pdf doesn't count
    expect(d.docCount).toBe(2);
    expect(d.hasValidDoc).toBe(true);
  });

  it("finds notes only in subfolders (the shallow-scan bug)", async () => {
    fakeFs({
      "/v": [dir(".obsidian"), dir("daily")],
      "/v/daily": [file("2026-07-10.md")],
    });
    const d = await detectVault("/v");
    expect(d.docCount).toBe(1);
    expect(d.hasValidDoc).toBe(true);
  });

  it("plain directory counts md/txt/pdf", async () => {
    fakeFs({ "/n": [file("a.md"), file("b.txt"), file("c.pdf"), file("d.docx")] });
    const d = await detectVault("/n");
    expect(d.isVault).toBe(false);
    expect(d.sourceType).toBe("directory");
    expect(d.docCount).toBe(3);
  });

  it("obsidian vault with only txt files has no valid doc", async () => {
    fakeFs({ "/v": [dir(".obsidian"), file("notes.txt")] });
    const d = await detectVault("/v");
    expect(d.sourceType).toBe("obsidian");
    expect(d.hasValidDoc).toBe(false);
    expect(d.docCount).toBe(0);
  });

  it("skips dot-directories and dot-files", async () => {
    fakeFs({
      "/n": [dir(".git"), file(".hidden.md"), file("real.md")],
      // /n/.git is never listed — walking into it would throw
    });
    const d = await detectVault("/n");
    expect(d.docCount).toBe(1);
  });

  it("stops descending beyond depth 6", async () => {
    const tree: Record<string, Entry[]> = {};
    let p = "/r";
    // depth 1 = root; build dirs to depth 8, each with one md
    tree[p] = [dir("d"), file("f1.md")];
    for (let i = 2; i <= 8; i++) {
      p = `${p}/d`;
      tree[p] = [dir("d"), file(`f${i}.md`)];
    }
    fakeFs(tree);
    const d = await detectVault("/r");
    // files at depth 1..6 counted; deeper dirs never entered
    expect(d.docCount).toBe(6);
  });

  it("caps at MAX_ENTRIES and reports countCapped", async () => {
    const many = Array.from({ length: MAX_ENTRIES + 100 }, (_, i) => file(`f${i}.md`));
    fakeFs({ "/big": many });
    const d = await detectVault("/big");
    expect(d.countCapped).toBe(true);
    expect(d.docCount).toBeLessThanOrEqual(MAX_ENTRIES);
  });

  it("unreadable root allows submit (daemon is the authority)", async () => {
    mocks.readDir.mockRejectedValue(new Error("EACCES"));
    const d = await detectVault("/locked");
    expect(d.unreadable).toBe(true);
    expect(d.hasValidDoc).toBe(false);
  });

  it("unreadable subdirectory is skipped, walk continues", async () => {
    fakeFs({ "/n": [dir("locked"), file("ok.md")] }); // /n/locked missing → throws
    const d = await detectVault("/n");
    expect(d.docCount).toBe(1);
    expect(d.unreadable).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `pnpm vitest run src/lib/vaultDetection.test.ts`
Expected: FAIL — `Cannot find module './vaultDetection'`

- [ ] **Step 3: Implement** — `src/lib/vaultDetection.ts`:

```ts
import { readDir } from "@tauri-apps/plugin-fs";

export const MAX_DEPTH = 6;
export const MAX_ENTRIES = 5000;

// Validity follows the SOURCE TYPE the daemon will enforce (spec §3, council
// change e): obsidian sources index markdown only (has_any_markdown); plain
// directory sources ingest md/txt/pdf (daemon directory.rs filter).
const OBSIDIAN_EXTENSIONS = [".md"];
const DIRECTORY_EXTENSIONS = [".md", ".txt", ".pdf"];

export interface VaultDetection {
  isVault: boolean;
  sourceType: "obsidian" | "directory";
  docCount: number;
  countCapped: boolean;
  hasValidDoc: boolean;
  unreadable: boolean;
}

/** Recursive, bounded walk. Never a reason to block submit — the daemon's
 *  POST /api/sources validation is the authority. */
export async function detectVault(path: string): Promise<VaultDetection> {
  let rootEntries: Awaited<ReturnType<typeof readDir>>;
  try {
    rootEntries = await readDir(path);
  } catch {
    return {
      isVault: false,
      sourceType: "directory",
      docCount: 0,
      countCapped: false,
      hasValidDoc: false,
      unreadable: true,
    };
  }

  const isVault = rootEntries.some((e) => e.name === ".obsidian" && e.isDirectory);
  const sourceType = isVault ? ("obsidian" as const) : ("directory" as const);
  const extensions = isVault ? OBSIDIAN_EXTENSIONS : DIRECTORY_EXTENSIONS;

  let docCount = 0;
  let entriesVisited = 0;
  let capped = false;

  const queue: Array<{ dir: string; entries: typeof rootEntries; depth: number }> = [
    { dir: path, entries: rootEntries, depth: 1 },
  ];

  while (queue.length > 0 && !capped) {
    const { dir, entries, depth } = queue.shift()!;
    for (const entry of entries) {
      if (entriesVisited >= MAX_ENTRIES) {
        capped = true;
        break;
      }
      entriesVisited += 1;
      const name = entry.name ?? "";
      if (name.startsWith(".")) continue; // dot files and dot dirs (.obsidian, .git…)
      if (entry.isDirectory) {
        if (depth < MAX_DEPTH) {
          try {
            const children = await readDir(`${dir}/${name}`);
            queue.push({ dir: `${dir}/${name}`, entries: children, depth: depth + 1 });
          } catch {
            // unreadable subdir: skip, keep walking
          }
        }
      } else if (extensions.some((ext) => name.toLowerCase().endsWith(ext))) {
        docCount += 1;
      }
    }
  }

  return {
    isVault,
    sourceType,
    docCount,
    countCapped: capped,
    hasValidDoc: docCount > 0,
    unreadable: false,
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pnpm vitest run src/lib/vaultDetection.test.ts`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/lib/vaultDetection.ts src/lib/vaultDetection.test.ts
git commit -m "feat: recursive bounded vault detection (fixes shallow-scan bug)"
```

---

### Task 6: `VaultConnectCard` + AddSourceDialog refactor

One shared card for "connect a vault/notes folder", used by the sources dialog and (Task 10) the wizard import step. Zero-count **never blocks submit** (council change e); post-connect it polls `listRegisteredSources` and shows "Indexed N files · M memories".

**Files:**
- Create: `src/components/memory/sources/VaultConnectCard.tsx`
- Modify: `src/components/memory/sources/AddSourceDialog.tsx`
- Modify: `src/i18n/resources.ts` (new `sources.vault.*` keys, all 3 locales)
- Test: `src/components/memory/sources/VaultConnectCard.test.tsx`

**Interfaces:**
- Consumes: `detectVault` (Task 5); `addSource`, `syncRegisteredSource`, `listRegisteredSources`, `RegisteredSource` from `src/lib/tauri` (`RegisteredSource.file_count` / `.memory_count` feed the indexed line); `open as openDialog` from `@tauri-apps/plugin-dialog`.
- Produces: `<VaultConnectCard variant="dialog" | "wizard" onConnected?={(src: RegisteredSource) => void} />`. In `dialog` variant the surrounding dialog closes on connect; in `wizard` variant the card stays and shows indexing progress inline.

- [ ] **Step 1: Add i18n keys** — in `src/i18n/resources.ts`, inside the `en` object's `settings.sources` block (~line 312), add a sibling `vault` block **under `sources` at the top level of `settings`**… no — put it in its own top-level `sources` namespace? Follow the existing structure: `settings.sources` holds settings-page copy. Add a new top-level `vaultConnect` namespace after `setup` (keeps wizard + dialog shared copy in one place):

```ts
  vaultConnect: {
    title: "Connect a notes folder",
    description:
      "Wenlan watches the folder and indexes your notes — Obsidian vaults, plain folders of .md, .txt, and .pdf files.",
    browse: "Browse…",
    placeholder: "Select a folder…",
    scanning: "Scanning folder…",
    detectedVault: "✓ Detected .obsidian/ — Obsidian vault",
    filesFound_one: "{{count}} supported file found",
    filesFound_other: "{{count}} supported files found",
    filesFoundCapped: "5,000+ supported files found",
    noneFound: "No notes found — Wenlan will verify on connect",
    vaultMarkdownOnly: "Obsidian vaults index Markdown notes (.md).",
    connect: "Connect",
    connecting: "Connecting…",
    indexing: "Indexing…",
    indexed: "Indexed {{files}} files · {{memories}} memories",
  },
```

zh-Hans (same keys, in the `zhHans` object at the same position):

```ts
  vaultConnect: {
    title: "连接笔记文件夹",
    description:
      "Wenlan 会监视该文件夹并索引你的笔记 — Obsidian 仓库或包含 .md、.txt、.pdf 文件的普通文件夹。",
    browse: "浏览…",
    placeholder: "选择文件夹…",
    scanning: "正在扫描文件夹…",
    detectedVault: "✓ 检测到 .obsidian/ — Obsidian 仓库",
    filesFound_one: "找到 {{count}} 个支持的文件",
    filesFound_other: "找到 {{count}} 个支持的文件",
    filesFoundCapped: "找到 5,000+ 个支持的文件",
    noneFound: "未找到笔记 — 连接时由 Wenlan 验证",
    vaultMarkdownOnly: "Obsidian 仓库仅索引 Markdown 笔记（.md）。",
    connect: "连接",
    connecting: "连接中…",
    indexing: "索引中…",
    indexed: "已索引 {{files}} 个文件 · {{memories}} 条记忆",
  },
```

zh-Hant:

```ts
  vaultConnect: {
    title: "連接筆記資料夾",
    description:
      "Wenlan 會監看該資料夾並索引你的筆記 — Obsidian 儲存庫或包含 .md、.txt、.pdf 檔案的一般資料夾。",
    browse: "瀏覽…",
    placeholder: "選擇資料夾…",
    scanning: "正在掃描資料夾…",
    detectedVault: "✓ 偵測到 .obsidian/ — Obsidian 儲存庫",
    filesFound_one: "找到 {{count}} 個支援的檔案",
    filesFound_other: "找到 {{count}} 個支援的檔案",
    filesFoundCapped: "找到 5,000+ 個支援的檔案",
    noneFound: "未找到筆記 — 連接時由 Wenlan 驗證",
    vaultMarkdownOnly: "Obsidian 儲存庫僅索引 Markdown 筆記（.md）。",
    connect: "連接",
    connecting: "連接中…",
    indexing: "索引中…",
    indexed: "已索引 {{files}} 個檔案 · {{memories}} 條記憶",
  },
```

(Exact insertion point: each locale object has the same top-level key order — add `vaultConnect` right after the `setup` block in all three.)

- [ ] **Step 2: Write the failing test** — `src/components/memory/sources/VaultConnectCard.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../../i18n";

const mocks = vi.hoisted(() => ({
  openDialog: vi.fn(),
  detectVault: vi.fn(),
  addSource: vi.fn(),
  syncRegisteredSource: vi.fn(),
  listRegisteredSources: vi.fn(),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: mocks.openDialog }));
vi.mock("../../../lib/vaultDetection", () => ({ detectVault: mocks.detectVault }));
vi.mock("../../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../lib/tauri")>();
  return {
    ...actual,
    addSource: mocks.addSource,
    syncRegisteredSource: mocks.syncRegisteredSource,
    listRegisteredSources: mocks.listRegisteredSources,
  };
});

import VaultConnectCard from "./VaultConnectCard";

function renderCard(props: Partial<React.ComponentProps<typeof VaultConnectCard>> = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <VaultConnectCard variant="wizard" {...props} />
    </QueryClientProvider>
  );
}

const SOURCE = {
  id: "s1",
  source_type: "obsidian",
  path: "/v",
  status: "Active",
  last_sync: null,
  file_count: 12,
  memory_count: 3,
};

describe("VaultConnectCard", () => {
  beforeEach(() => {
    Object.values(mocks).forEach((m) => m.mockReset());
    mocks.openDialog.mockResolvedValue("/v");
    mocks.addSource.mockResolvedValue(SOURCE);
    mocks.syncRegisteredSource.mockResolvedValue({
      files_found: 12, ingested: 12, skipped: 0, errors: 0,
    });
    mocks.listRegisteredSources.mockResolvedValue([SOURCE]);
  });

  it("zero-count detection warns but does NOT block submit (council change e)", async () => {
    mocks.detectVault.mockResolvedValue({
      isVault: false, sourceType: "directory", docCount: 0,
      countCapped: false, hasValidDoc: false, unreadable: false,
    });
    renderCard();
    await userEvent.click(screen.getByText("Browse…"));
    await waitFor(() =>
      expect(screen.getByText(/No notes found/)).toBeInTheDocument()
    );
    const connect = screen.getByRole("button", { name: "Connect" });
    expect(connect).toBeEnabled();
  });

  it("connects an obsidian vault: addSource + one-shot sync + indexed line", async () => {
    mocks.detectVault.mockResolvedValue({
      isVault: true, sourceType: "obsidian", docCount: 12,
      countCapped: false, hasValidDoc: true, unreadable: false,
    });
    renderCard();
    await userEvent.click(screen.getByText("Browse…"));
    await waitFor(() => expect(screen.getByText(/Obsidian vault/)).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Connect" }));
    await waitFor(() => expect(mocks.addSource).toHaveBeenCalledWith("obsidian", "/v"));
    await waitFor(() => expect(mocks.syncRegisteredSource).toHaveBeenCalledWith("s1"));
    await waitFor(() =>
      expect(screen.getByText(/Indexed 12 files/)).toBeInTheDocument()
    );
  });

  it("surfaces daemon 4xx verbatim", async () => {
    mocks.detectVault.mockResolvedValue({
      isVault: false, sourceType: "directory", docCount: 3,
      countCapped: false, hasValidDoc: true, unreadable: false,
    });
    mocks.addSource.mockRejectedValue(new Error("path does not exist: /v"));
    renderCard();
    await userEvent.click(screen.getByText("Browse…"));
    await waitFor(() => expect(screen.getByText(/3 supported files/)).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Connect" }));
    await waitFor(() =>
      expect(screen.getByText(/path does not exist: \/v/)).toBeInTheDocument()
    );
  });
});
```

(If `../../../i18n` is not how other component tests initialize i18n, copy the import used by an existing test such as `SettingsPage.language.test.tsx` — the i18n side-effect import must match the project's pattern.)

- [ ] **Step 3: Run to verify failure**

Run: `pnpm vitest run src/components/memory/sources/VaultConnectCard.test.tsx`
Expected: FAIL — `Cannot find module './VaultConnectCard'`

- [ ] **Step 4: Implement** — `src/components/memory/sources/VaultConnectCard.tsx`:

```tsx
import { useCallback, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import {
  addSource,
  syncRegisteredSource,
  listRegisteredSources,
  type RegisteredSource,
} from "../../../lib/tauri";
import { detectVault, type VaultDetection } from "../../../lib/vaultDetection";

interface Props {
  variant: "dialog" | "wizard";
  onConnected?: (source: RegisteredSource) => void;
}

export default function VaultConnectCard({ variant, onConnected }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [path, setPath] = useState("");
  const [detection, setDetection] = useState<VaultDetection | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [connectedId, setConnectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Post-connect: poll the registered source until it reports counts
  // ("Indexed N files · M memories", spec §3). Wizard variant only — the
  // dialog closes into the sources list which already polls.
  const { data: connectedSource } = useQuery({
    queryKey: ["vault-connect-progress", connectedId],
    queryFn: async () => {
      const sources = await listRegisteredSources();
      return sources.find((s) => s.id === connectedId) ?? null;
    },
    enabled: variant === "wizard" && connectedId !== null,
    refetchInterval: 2000,
  });

  const handleBrowse = useCallback(async () => {
    const selected = await openDialog({ directory: true, multiple: false });
    if (!selected || typeof selected !== "string") return;
    setPath(selected);
    setError(null);
    setDetection(null);
    setDetecting(true);
    setDetection(await detectVault(selected));
    setDetecting(false);
  }, []);

  const handleConnect = useCallback(async () => {
    setError(null);
    setConnecting(true);
    try {
      const source = await addSource(detection?.sourceType ?? "directory", path);
      queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
      // Obsidian vaults are not on the daemon's 30s directory scheduler —
      // kick a one-shot first index (same rationale as AddSourceDialog).
      if (source.source_type === "obsidian") {
        syncRegisteredSource(source.id).then(() => {
          queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
          queryClient.invalidateQueries({ queryKey: ["vault-connect-progress", source.id] });
        });
      }
      setConnectedId(source.id);
      onConnected?.(source);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setConnecting(false);
    }
  }, [detection, path, queryClient, onConnected]);

  // Submit is never blocked by a zero count (council change e): the daemon's
  // POST /api/sources validation is the authority; its 4xx surfaces verbatim.
  const canSubmit = path.length > 0 && !detecting && !connecting && connectedId === null;

  const indexed = connectedSource && connectedSource.file_count > 0;

  return (
    <div
      className="rounded-xl p-4 flex flex-col"
      style={{ border: "1px solid var(--mem-border)", backgroundColor: "var(--mem-surface)", gap: "12px" }}
    >
      <div>
        <h3 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "15px", fontWeight: 500, color: "var(--mem-text)" }}>
          {t("vaultConnect.title")}
        </h3>
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", lineHeight: 1.5, marginTop: "4px" }}>
          {t("vaultConnect.description")}
        </p>
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={path}
          readOnly
          placeholder={t("vaultConnect.placeholder")}
          className="flex-1 rounded-md px-3 py-2 text-sm"
          style={{
            border: "1px solid var(--mem-border)",
            backgroundColor: "var(--mem-bg)",
            color: "var(--mem-text)",
            fontFamily: "var(--mem-font-mono)",
            fontSize: "12px",
          }}
        />
        <button
          onClick={handleBrowse}
          className="rounded-md px-3 py-2 text-sm"
          style={{ border: "1px solid var(--mem-border)", color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}
        >
          {t("vaultConnect.browse")}
        </button>
      </div>

      {detecting && (
        <p style={{ fontSize: "12px", color: "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)" }}>
          {t("vaultConnect.scanning")}
        </p>
      )}

      {detection && !detecting && (
        <div style={{ fontSize: "12px", fontFamily: "var(--mem-font-body)", display: "flex", flexDirection: "column", gap: "2px" }}>
          {detection.isVault && (
            <span style={{ color: "var(--mem-accent-indigo)" }}>{t("vaultConnect.detectedVault")}</span>
          )}
          {detection.docCount > 0 ? (
            <span style={{ color: "var(--mem-text-secondary)" }}>
              {detection.countCapped
                ? t("vaultConnect.filesFoundCapped")
                : t("vaultConnect.filesFound", { count: detection.docCount })}
            </span>
          ) : (
            <span style={{ color: "var(--mem-accent-amber)" }}>{t("vaultConnect.noneFound")}</span>
          )}
          {detection.isVault && !detection.hasValidDoc && (
            <span style={{ color: "var(--mem-text-tertiary)" }}>{t("vaultConnect.vaultMarkdownOnly")}</span>
          )}
        </div>
      )}

      {error && (
        <p style={{ fontSize: "12px", color: "var(--mem-danger, #ef4444)", fontFamily: "var(--mem-font-mono)" }}>{error}</p>
      )}

      {connectedId === null ? (
        <button
          onClick={handleConnect}
          disabled={!canSubmit}
          className="self-end rounded-md px-4 py-2 text-sm font-medium disabled:opacity-50"
          style={{ backgroundColor: "var(--mem-accent-indigo)", color: "white", fontFamily: "var(--mem-font-body)" }}
        >
          {connecting ? t("vaultConnect.connecting") : t("vaultConnect.connect")}
        </button>
      ) : (
        <p style={{ fontSize: "12px", color: "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)" }}>
          {indexed
            ? t("vaultConnect.indexed", {
                files: connectedSource.file_count,
                memories: connectedSource.memory_count,
              })
            : t("vaultConnect.indexing")}
        </p>
      )}
    </div>
  );
}
```

Check `--mem-danger` exists in `src/index.css`; if not, use the literal the codebase already uses for errors (AddSourceDialog uses Tailwind `text-red-500` — match that idiom instead of inventing a token).

- [ ] **Step 5: Run test to verify it passes**

Run: `pnpm vitest run src/components/memory/sources/VaultConnectCard.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 6: Refactor `AddSourceDialog` to the shared pieces.** In `src/components/memory/sources/AddSourceDialog.tsx`:

1. Replace the inline detection block (lines 57-71) with `detectVault`:

```tsx
      const d = await detectVault(selectedPath);
      setDetection(d);
```

   and change the local `Detection` interface usage to `VaultDetection` (import from `../../../lib/vaultDetection`). Delete the now-unused `SUPPORTED_EXTENSIONS` const and `readDir` import.

2. `handleSubmit` uses the detected source type directly:

```tsx
    addMutation.mutate({
      sourceType: detection?.sourceType ?? "directory",
      sourcePath: path,
    });
```

3. **Remove the zero-count gate** — `canSubmit` (lines 97-101) becomes:

```tsx
  const canSubmit = path.length > 0 && !detecting && !addMutation.isPending;
```

4. Zero-count messaging (lines 147-151) becomes a warning, not a blocker — replace the red `No supported files…` span with:

```tsx
              <span className="text-[var(--mem-accent-amber)]">
                {t("vaultConnect.noneFound")}
              </span>
```

   Add `useTranslation` to the component (`const { t } = useTranslation();`, import from `react-i18next`). Also show `5,000+` when capped: replace the count line (lines 143-146) with:

```tsx
            ) : detection.docCount > 0 ? (
              <span className="text-[var(--mem-text-secondary)]">
                {detection.countCapped
                  ? t("vaultConnect.filesFoundCapped")
                  : t("vaultConnect.filesFound", { count: detection.docCount })}
              </span>
            ) : (
```

Leave the rest of the dialog's existing (English-literal) copy untouched — converting it wholesale is out of scope; only the strings this task changes go through i18n.

- [ ] **Step 7: Run the sources tests + type-check**

Run: `pnpm exec tsc -b && pnpm vitest run src/components/memory/sources`
Expected: clean / PASS

- [ ] **Step 8: Commit**

```bash
git add src/components/memory/sources/VaultConnectCard.tsx src/components/memory/sources/VaultConnectCard.test.tsx src/components/memory/sources/AddSourceDialog.tsx src/i18n/resources.ts
git commit -m "feat: shared VaultConnectCard; AddSourceDialog recursive check, no zero-count block"
```

---

### Task 7: Provider presets + `AnyProviderCard`

The vendor-agnostic external-provider card (spec §1): preset picker → endpoint quick-fill, model auto-discovery with free-text fallback, Test button, Save, precedence warning, and daemon-0.13 gating for keyed presets.

**Files:**
- Create: `src/components/intelligence/providerPresets.ts`
- Create: `src/components/intelligence/AnyProviderCard.tsx`
- Test: `src/components/intelligence/AnyProviderCard.test.tsx`
- Modify: `src/i18n/resources.ts` (`externalProvider.*` keys, 3 locales)

**Interfaces:**
- Consumes: `useDaemonVersion` (Task 1), `listExternalModels` (Task 2), `getExternalLlm`, `setExternalLlm`, `testExternalLlm`, `getExternalLlmKeyConfigured` (Task 3), `useApiKeyStatus` (existing, `src/components/intelligence/IntelligenceSetup.tsx:29`).
- Produces:
  - `PROVIDER_PRESETS: ProviderPreset[]`, `presetForEndpoint(endpoint: string | null): ProviderPreset`, `type PresetGroup = "cloud" | "local" | "custom"`.
  - `<AnyProviderCard groups?: PresetGroup[] initialPresetId?: string hidePresetPicker?: boolean />` — used by Tasks 8, 9.

- [ ] **Step 1: Create the preset table** — `src/components/intelligence/providerPresets.ts`:

```ts
export type PresetGroup = "cloud" | "local" | "custom";

export interface ProviderPreset {
  id: string;
  /** Vendor names are proper nouns — not translated. */
  name: string;
  endpoint: string;
  keyRequired: boolean;
  group: PresetGroup;
}

// Spec §1 preset table. Every keyed preset's /models + /chat/completions
// compatibility is validated live during implementation (council dissent
// note); a preset whose /models shape drifts ships free-text-only (the card
// already falls back to free text when discovery fails, so no code change —
// just note the finding in the PR description).
export const PROVIDER_PRESETS: ProviderPreset[] = [
  { id: "openai", name: "OpenAI", endpoint: "https://api.openai.com/v1", keyRequired: true, group: "cloud" },
  { id: "gemini", name: "Google Gemini", endpoint: "https://generativelanguage.googleapis.com/v1beta/openai", keyRequired: true, group: "cloud" },
  { id: "groq", name: "Groq", endpoint: "https://api.groq.com/openai/v1", keyRequired: true, group: "cloud" },
  { id: "openrouter", name: "OpenRouter", endpoint: "https://openrouter.ai/api/v1", keyRequired: true, group: "cloud" },
  { id: "mistral", name: "Mistral", endpoint: "https://api.mistral.ai/v1", keyRequired: true, group: "cloud" },
  { id: "deepseek", name: "DeepSeek", endpoint: "https://api.deepseek.com/v1", keyRequired: true, group: "cloud" },
  { id: "xai", name: "xAI", endpoint: "https://api.x.ai/v1", keyRequired: true, group: "cloud" },
  { id: "ollama", name: "Ollama (local)", endpoint: "http://localhost:11434/v1", keyRequired: false, group: "local" },
  { id: "lmstudio", name: "LM Studio (local)", endpoint: "http://localhost:1234/v1", keyRequired: false, group: "local" },
  { id: "custom", name: "Custom…", endpoint: "", keyRequired: false, group: "custom" },
];

/** Match a saved endpoint back to its preset ("custom" when no match). */
export function presetForEndpoint(endpoint: string | null): ProviderPreset {
  const norm = (endpoint ?? "").replace(/\/+$/, "");
  return (
    PROVIDER_PRESETS.find((p) => p.endpoint !== "" && p.endpoint === norm) ??
    PROVIDER_PRESETS[PROVIDER_PRESETS.length - 1]
  );
}
```

- [ ] **Step 2: Add i18n keys** — top-level `externalProvider` namespace after `vaultConnect`, all three locales:

en:

```ts
  externalProvider: {
    title: "Any provider",
    description:
      "Route synthesis through any OpenAI-compatible endpoint — cloud vendors or a local server.",
    presetLabel: "Provider",
    endpointLabel: "Endpoint URL",
    modelLabel: "Model",
    modelPlaceholder: "Model name (e.g. llama3.2)",
    modelDiscoveryFailed: "Couldn't list models — type a model name.",
    apiKeyLabel: "API key",
    apiKeyConfiguredPlaceholder: "••••••••  (key stored)",
    keyNeedsUpgrade:
      "API keys for cloud providers need Wenlan daemon 0.13+. This preset stays disabled until the runtime updates.",
    test: "Test",
    testing: "Testing…",
    testOk: "Response: {{response}}",
    save: "Save",
    saving: "Saving…",
    savedApplied: "Applied — this provider is live.",
    savedRestart: "Saved. Restart Wenlan to apply.",
    anthropicPrecedence:
      "Anthropic takes precedence while its key is set — this provider serves only when the Anthropic key is removed.",
  },
```

zh-Hans:

```ts
  externalProvider: {
    title: "任意服务商",
    description: "通过任意 OpenAI 兼容端点进行合成 — 云服务商或本地服务器。",
    presetLabel: "服务商",
    endpointLabel: "端点 URL",
    modelLabel: "模型",
    modelPlaceholder: "模型名称（如 llama3.2）",
    modelDiscoveryFailed: "无法列出模型 — 请手动输入模型名称。",
    apiKeyLabel: "API 密钥",
    apiKeyConfiguredPlaceholder: "••••••••（已保存密钥）",
    keyNeedsUpgrade:
      "云服务商的 API 密钥需要 Wenlan 守护进程 0.13+。运行时更新前此预设保持禁用。",
    test: "测试",
    testing: "测试中…",
    testOk: "响应：{{response}}",
    save: "保存",
    saving: "保存中…",
    savedApplied: "已生效 — 此服务商已启用。",
    savedRestart: "已保存。重启 Wenlan 后生效。",
    anthropicPrecedence:
      "设置了 Anthropic 密钥时优先使用 Anthropic — 移除该密钥后此服务商才会生效。",
  },
```

zh-Hant:

```ts
  externalProvider: {
    title: "任意服務商",
    description: "透過任意 OpenAI 相容端點進行合成 — 雲端服務商或本機伺服器。",
    presetLabel: "服務商",
    endpointLabel: "端點 URL",
    modelLabel: "模型",
    modelPlaceholder: "模型名稱（如 llama3.2）",
    modelDiscoveryFailed: "無法列出模型 — 請手動輸入模型名稱。",
    apiKeyLabel: "API 金鑰",
    apiKeyConfiguredPlaceholder: "••••••••（已儲存金鑰）",
    keyNeedsUpgrade:
      "雲端服務商的 API 金鑰需要 Wenlan 守護程序 0.13+。執行階段更新前此預設保持停用。",
    test: "測試",
    testing: "測試中…",
    testOk: "回應：{{response}}",
    save: "儲存",
    saving: "儲存中…",
    savedApplied: "已生效 — 此服務商已啟用。",
    savedRestart: "已儲存。重新啟動 Wenlan 後生效。",
    anthropicPrecedence:
      "設定了 Anthropic 金鑰時優先使用 Anthropic — 移除該金鑰後此服務商才會生效。",
  },
```

- [ ] **Step 3: Write the failing test** — `src/components/intelligence/AnyProviderCard.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../i18n";

const mocks = vi.hoisted(() => ({
  getDaemonVersion: vi.fn(),
  getExternalLlm: vi.fn(),
  setExternalLlm: vi.fn(),
  testExternalLlm: vi.fn(),
  listExternalModels: vi.fn(),
  getExternalLlmKeyConfigured: vi.fn(),
  getSetupStatus: vi.fn(),
}));
vi.mock("../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/tauri")>();
  return { ...actual, ...mocks };
});

import AnyProviderCard from "./AnyProviderCard";

function renderCard(props: Partial<React.ComponentProps<typeof AnyProviderCard>> = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <AnyProviderCard {...props} />
    </QueryClientProvider>
  );
}

describe("AnyProviderCard", () => {
  beforeEach(() => {
    Object.values(mocks).forEach((m) => m.mockReset());
    mocks.getDaemonVersion.mockResolvedValue("0.12.0");
    mocks.getExternalLlm.mockResolvedValue([null, null]);
    mocks.getExternalLlmKeyConfigured.mockResolvedValue(false);
    mocks.getSetupStatus.mockResolvedValue({
      setup_completed: true, mode: "basic-memory", anthropic_key_configured: false,
      local_model_selected: null, local_model_loaded: null, local_model_cached: false,
    });
    mocks.listExternalModels.mockResolvedValue(["llama3.2:3b"]);
    mocks.setExternalLlm.mockResolvedValue(undefined);
    mocks.testExternalLlm.mockResolvedValue({ response: "pong" });
  });

  it("preset fills the endpoint and discovers models", async () => {
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    expect(screen.getByLabelText("Endpoint URL")).toHaveValue("http://localhost:11434/v1");
    await waitFor(() =>
      expect(mocks.listExternalModels).toHaveBeenCalledWith("http://localhost:11434/v1", null)
    );
  });

  it("keyed presets are disabled with explanation below daemon 0.13", async () => {
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "openai");
    expect(await screen.findByText(/needs Wenlan daemon 0.13\+/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
    // key field hidden below 0.13
    expect(screen.queryByLabelText("API key")).not.toBeInTheDocument();
  });

  it("keyed preset works on 0.13: key field shown, save passes key, Applied note", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "openai");
    const keyField = await screen.findByLabelText("API key");
    await userEvent.type(keyField, "sk-test");
    await userEvent.type(screen.getByLabelText("Model"), "gpt-4o-mini");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(mocks.setExternalLlm).toHaveBeenCalledWith(
        "https://api.openai.com/v1", "gpt-4o-mini", "sk-test"
      )
    );
    expect(await screen.findByText(/Applied/)).toBeInTheDocument();
  });

  it("keyless save on 0.12 omits the key and shows restart note", async () => {
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    await userEvent.type(screen.getByLabelText("Model"), "llama3.2:3b");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(mocks.setExternalLlm).toHaveBeenCalledWith(
        "http://localhost:11434/v1", "llama3.2:3b", undefined
      )
    );
    expect(await screen.findByText(/Restart Wenlan to apply/)).toBeInTheDocument();
  });

  it("discovery failure falls back to free-text model entry with hint", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    expect(await screen.findByText(/type a model name/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Model")).toBeEnabled();
  });

  it("test button shows verbatim daemon error", async () => {
    mocks.testExternalLlm.mockRejectedValue(new Error("LLM request failed: 401 Unauthorized"));
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    await userEvent.type(screen.getByLabelText("Model"), "llama3.2:3b");
    await userEvent.click(screen.getByRole("button", { name: "Test" }));
    expect(await screen.findByText(/401 Unauthorized/)).toBeInTheDocument();
  });

  it("shows the Anthropic precedence warning when an Anthropic key is configured", async () => {
    mocks.getSetupStatus.mockResolvedValue({
      setup_completed: true, mode: "anthropic-key", anthropic_key_configured: true,
      local_model_selected: null, local_model_loaded: null, local_model_cached: false,
    });
    renderCard();
    expect(await screen.findByText(/Anthropic takes precedence/)).toBeInTheDocument();
  });
});
```

Note: `useApiKeyStatus` internally calls a tauri wrapper — read `IntelligenceSetup.tsx:29-39` to confirm which one (it uses `getSetupStatus` or `getApiKey`); mock THAT function in the test (the block above assumes `getSetupStatus`; adjust to what the hook actually calls).

- [ ] **Step 4: Run to verify failure**

Run: `pnpm vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: FAIL — `Cannot find module './AnyProviderCard'`

- [ ] **Step 5: Implement** — `src/components/intelligence/AnyProviderCard.tsx`:

```tsx
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  getExternalLlm,
  setExternalLlm,
  testExternalLlm,
  listExternalModels,
  getExternalLlmKeyConfigured,
} from "../../lib/tauri";
import { useDaemonVersion } from "../../hooks/useDaemonVersion";
import { useApiKeyStatus } from "./IntelligenceSetup";
import {
  PROVIDER_PRESETS,
  presetForEndpoint,
  type PresetGroup,
} from "./providerPresets";

interface Props {
  /** Filter the preset picker (wizard: cloud-only / local-only). */
  groups?: PresetGroup[];
  initialPresetId?: string;
  hidePresetPicker?: boolean;
}

const fieldStyle: React.CSSProperties = {
  width: "100%",
  padding: "8px 10px",
  borderRadius: "8px",
  border: "1px solid var(--mem-border)",
  backgroundColor: "var(--mem-bg)",
  color: "var(--mem-text)",
  fontFamily: "var(--mem-font-mono)",
  fontSize: "12px",
};

const labelStyle: React.CSSProperties = {
  fontFamily: "var(--mem-font-body)",
  fontSize: "12px",
  fontWeight: 500,
  color: "var(--mem-text-secondary)",
};

export default function AnyProviderCard({ groups, initialPresetId, hidePresetPicker }: Props) {
  const { t } = useTranslation();
  const { supportsExternalKey, supportsHotSwap } = useDaemonVersion();
  const anthropic = useApiKeyStatus();

  const presets = useMemo(
    () =>
      PROVIDER_PRESETS.filter(
        (p) => !groups || groups.includes(p.group) || p.group === "custom"
      ),
    [groups]
  );

  const [presetId, setPresetId] = useState(initialPresetId ?? presets[0].id);
  const preset = presets.find((p) => p.id === presetId) ?? presets[presets.length - 1];
  const [endpoint, setEndpoint] = useState(preset.endpoint);
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [testState, setTestState] = useState<
    | { kind: "idle" }
    | { kind: "testing" }
    | { kind: "ok"; response: string }
    | { kind: "error"; message: string }
  >({ kind: "idle" });
  const [saveState, setSaveState] = useState<
    "idle" | "saving" | "applied" | "restart" | `error:${string}`
  >("idle");

  // Prefill from the daemon's current external slot.
  const { data: current } = useQuery({ queryKey: ["external-llm"], queryFn: getExternalLlm });
  const { data: keyConfigured } = useQuery({
    queryKey: ["external-llm-key-configured"],
    queryFn: getExternalLlmKeyConfigured,
  });
  useEffect(() => {
    if (!current) return;
    const [savedEndpoint, savedModel] = current;
    if (savedEndpoint) {
      setEndpoint(savedEndpoint);
      setPresetId(presetForEndpoint(savedEndpoint).id);
    }
    if (savedModel) setModel(savedModel);
    // Run once when the saved config arrives; later edits are user-driven.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current]);

  // Keyed cloud presets stay dark below daemon 0.13 (spec §8).
  const lockedByVersion = preset.keyRequired && !supportsExternalKey;

  const trimmedEndpoint = endpoint.trim().replace(/\/+$/, "");
  const endpointValid = /^https?:\/\//.test(trimmedEndpoint);

  // Model auto-discovery; silent fallback to free text on failure (spec §1).
  const discovery = useQuery({
    queryKey: ["external-models", trimmedEndpoint, apiKey],
    queryFn: () => listExternalModels(trimmedEndpoint, apiKey || null),
    enabled: endpointValid && !lockedByVersion,
    retry: false,
    staleTime: 30_000,
  });
  const models = discovery.data ?? [];

  const selectPreset = (id: string) => {
    setPresetId(id);
    const next = PROVIDER_PRESETS.find((p) => p.id === id);
    if (next && next.endpoint) setEndpoint(next.endpoint);
    if (next && !next.endpoint) setEndpoint("");
    setModel("");
    setTestState({ kind: "idle" });
    setSaveState("idle");
  };

  const keyToSend = (): string | undefined =>
    supportsExternalKey && apiKey !== "" ? apiKey : undefined;

  const handleTest = async () => {
    setTestState({ kind: "testing" });
    try {
      const resp = await testExternalLlm(trimmedEndpoint, model, keyToSend() ?? null);
      setTestState({ kind: "ok", response: resp.response });
    } catch (err) {
      // Verbatim daemon error (spec: Error handling).
      setTestState({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  const handleSave = async () => {
    setSaveState("saving");
    try {
      await setExternalLlm(trimmedEndpoint, model, keyToSend());
      setSaveState(supportsHotSwap ? "applied" : "restart");
    } catch (err) {
      setSaveState(`error:${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const canAct = endpointValid && model.trim() !== "" && !lockedByVersion;

  return (
    <div
      className="rounded-xl p-4 flex flex-col"
      style={{ border: "1px solid var(--mem-border)", backgroundColor: "var(--mem-surface)", gap: "12px" }}
    >
      <div>
        <h3 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "15px", fontWeight: 500, color: "var(--mem-text)" }}>
          {t("externalProvider.title")}
        </h3>
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", lineHeight: 1.5, marginTop: "4px" }}>
          {t("externalProvider.description")}
        </p>
      </div>

      {anthropic.configured && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-accent-amber)", lineHeight: 1.5 }}>
          {t("externalProvider.anthropicPrecedence")}
        </p>
      )}

      {!hidePresetPicker && (
        <label className="flex flex-col gap-1">
          <span style={labelStyle}>{t("externalProvider.presetLabel")}</span>
          <select value={presetId} onChange={(e) => selectPreset(e.target.value)} style={fieldStyle}>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </label>
      )}

      {lockedByVersion ? (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", lineHeight: 1.5 }}>
          {t("externalProvider.keyNeedsUpgrade")}
        </p>
      ) : (
        <>
          <label className="flex flex-col gap-1">
            <span style={labelStyle}>{t("externalProvider.endpointLabel")}</span>
            <input type="text" value={endpoint} onChange={(e) => setEndpoint(e.target.value)} style={fieldStyle} />
          </label>

          <label className="flex flex-col gap-1">
            <span style={labelStyle}>{t("externalProvider.modelLabel")}</span>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder={t("externalProvider.modelPlaceholder")}
              list="any-provider-models"
              style={fieldStyle}
            />
            <datalist id="any-provider-models">
              {models.map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
            {discovery.isError && (
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
                {t("externalProvider.modelDiscoveryFailed")}
              </span>
            )}
          </label>

          {supportsExternalKey && (
            <label className="flex flex-col gap-1">
              <span style={labelStyle}>{t("externalProvider.apiKeyLabel")}</span>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={keyConfigured ? t("externalProvider.apiKeyConfiguredPlaceholder") : ""}
                style={fieldStyle}
              />
            </label>
          )}
        </>
      )}

      <div className="flex items-center gap-2">
        <button
          onClick={handleTest}
          disabled={!canAct || testState.kind === "testing"}
          className="rounded-md px-3 py-1.5 text-xs disabled:opacity-50"
          style={{ border: "1px solid var(--mem-border)", color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}
        >
          {testState.kind === "testing" ? t("externalProvider.testing") : t("externalProvider.test")}
        </button>
        <button
          onClick={handleSave}
          disabled={!canAct || saveState === "saving"}
          className="rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-50"
          style={{ backgroundColor: "var(--mem-accent-indigo)", color: "white", fontFamily: "var(--mem-font-body)" }}
        >
          {saveState === "saving" ? t("externalProvider.saving") : t("externalProvider.save")}
        </button>
      </div>

      {testState.kind === "ok" && (
        <p style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", color: "var(--mem-text-secondary)" }}>
          {t("externalProvider.testOk", { response: testState.response })}
        </p>
      )}
      {testState.kind === "error" && (
        <p className="text-red-500" style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px" }}>
          {testState.message}
        </p>
      )}
      {saveState === "applied" && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-accent-sage)" }}>
          {t("externalProvider.savedApplied")}
        </p>
      )}
      {saveState === "restart" && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>
          {t("externalProvider.savedRestart")}
        </p>
      )}
      {saveState.startsWith("error:") && (
        <p className="text-red-500" style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px" }}>
          {saveState.slice("error:".length)}
        </p>
      )}
    </div>
  );
}
```

Adjust the `useApiKeyStatus` return-shape usage to the real hook (read `IntelligenceSetup.tsx:29-39`; if it returns e.g. `{ hasKey }` use that instead of `.configured`). Wire `<label>` → control association: the JSX above nests controls inside `<label>`, which `getByLabelText` resolves.

**Live preset validation (council dissent note):** while this task is under review, run the card against a real local Ollama if one is running (`pnpm dev:all`) and — for any cloud vendor the implementer has a key for — confirm `GET /models` parses. Record per-preset findings in the PR description. No key ⇒ note "unvalidated, free-text fallback covers drift".

- [ ] **Step 6: Run tests to verify they pass**

Run: `pnpm vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: PASS (7 tests)

- [ ] **Step 7: Commit**

```bash
git add src/components/intelligence/providerPresets.ts src/components/intelligence/AnyProviderCard.tsx src/components/intelligence/AnyProviderCard.test.tsx src/i18n/resources.ts
git commit -m "feat: Any-provider preset card with discovery, test, and 0.13 key gating"
```

---

### Task 8: Active-intelligence strip + Intelligence section assembly

The strip states the daemon priority chain and distinguishes **configured vs serving vs restart-pending** (council change c) — it never claims "serving" from config alone.

**Files:**
- Create: `src/components/intelligence/ActiveIntelligenceStrip.tsx`
- Modify: `src/components/memory/settings/sections/IntelligenceSection.tsx` (post-decomposition file)
- Modify: `src/i18n/resources.ts` (`intelligenceStrip.*`, 3 locales)
- Test: `src/components/intelligence/ActiveIntelligenceStrip.test.tsx`

**Interfaces:**
- Consumes: `getSetupStatus` (with Task 1's optional `external_llm` field), `getExternalLlm`, `useDaemonVersion`.
- Produces: `<ActiveIntelligenceStrip />` — also used by nothing else yet (settings only; the wizard shows per-card state instead).

- [ ] **Step 1: Add i18n keys** (3 locales, top-level `intelligenceStrip` after `externalProvider`):

en:

```ts
  intelligenceStrip: {
    chain: "Priority: Anthropic → external endpoint → on-device → basic memory",
    servingAnthropic: "Serving: Anthropic",
    servingExternal: "Serving: external endpoint",
    servingOnDevice: "Serving: on-device model",
    servingBasic: "Basic memory — no model configured",
    externalRestartPending: "External endpoint configured — restart pending",
    externalUnverified: "External endpoint configured (unverified)",
  },
```

zh-Hans:

```ts
  intelligenceStrip: {
    chain: "优先级：Anthropic → 外部端点 → 本机模型 → 基础记忆",
    servingAnthropic: "当前使用：Anthropic",
    servingExternal: "当前使用：外部端点",
    servingOnDevice: "当前使用：本机模型",
    servingBasic: "基础记忆 — 未配置模型",
    externalRestartPending: "外部端点已配置 — 等待重启",
    externalUnverified: "外部端点已配置（未验证）",
  },
```

zh-Hant:

```ts
  intelligenceStrip: {
    chain: "優先順序:Anthropic → 外部端點 → 本機模型 → 基礎記憶",
    servingAnthropic: "目前使用:Anthropic",
    servingExternal: "目前使用:外部端點",
    servingOnDevice: "目前使用:本機模型",
    servingBasic: "基礎記憶 — 未設定模型",
    externalRestartPending: "外部端點已設定 — 等待重新啟動",
    externalUnverified: "外部端點已設定(未驗證)",
  },
```

- [ ] **Step 2: Write the failing test** — `src/components/intelligence/ActiveIntelligenceStrip.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../i18n";

const mocks = vi.hoisted(() => ({
  getSetupStatus: vi.fn(),
  getExternalLlm: vi.fn(),
  getDaemonVersion: vi.fn(),
}));
vi.mock("../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/tauri")>();
  return { ...actual, ...mocks };
});

import ActiveIntelligenceStrip from "./ActiveIntelligenceStrip";

const BASE_STATUS = {
  setup_completed: true,
  mode: "basic-memory",
  anthropic_key_configured: false,
  local_model_selected: null,
  local_model_loaded: null,
  local_model_cached: false,
};

function renderStrip() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <ActiveIntelligenceStrip />
    </QueryClientProvider>
  );
}

describe("ActiveIntelligenceStrip", () => {
  beforeEach(() => {
    Object.values(mocks).forEach((m) => m.mockReset());
    mocks.getDaemonVersion.mockResolvedValue("0.12.0");
    mocks.getSetupStatus.mockResolvedValue(BASE_STATUS);
    mocks.getExternalLlm.mockResolvedValue([null, null]);
  });

  it("anthropic key configured tops the chain", async () => {
    mocks.getSetupStatus.mockResolvedValue({ ...BASE_STATUS, anthropic_key_configured: true });
    renderStrip();
    expect(await screen.findByText("Serving: Anthropic")).toBeInTheDocument();
  });

  it("0.12 external config shows configured (unverified), never serving", async () => {
    mocks.getExternalLlm.mockResolvedValue(["http://localhost:11434/v1", "llama3.2"]);
    renderStrip();
    expect(await screen.findByText(/configured \(unverified\)/)).toBeInTheDocument();
    expect(screen.queryByText("Serving: external endpoint")).not.toBeInTheDocument();
  });

  it("0.13 external loaded shows serving", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    mocks.getSetupStatus.mockResolvedValue({
      ...BASE_STATUS,
      external_llm: { configured: true, loaded: true },
    });
    mocks.getExternalLlm.mockResolvedValue(["https://api.openai.com/v1", "gpt-4o-mini"]);
    renderStrip();
    expect(await screen.findByText("Serving: external endpoint")).toBeInTheDocument();
  });

  it("0.13 configured-but-not-loaded shows restart pending", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    mocks.getSetupStatus.mockResolvedValue({
      ...BASE_STATUS,
      external_llm: { configured: true, loaded: false },
    });
    mocks.getExternalLlm.mockResolvedValue(["https://api.openai.com/v1", "gpt-4o-mini"]);
    renderStrip();
    expect(await screen.findByText(/restart pending/)).toBeInTheDocument();
  });

  it("on-device model loaded shows serving on-device", async () => {
    mocks.getSetupStatus.mockResolvedValue({ ...BASE_STATUS, local_model_loaded: "qwen3-4b" });
    renderStrip();
    expect(await screen.findByText("Serving: on-device model")).toBeInTheDocument();
  });

  it("nothing configured shows basic memory", async () => {
    renderStrip();
    expect(await screen.findByText(/Basic memory/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run to verify failure**

Run: `pnpm vitest run src/components/intelligence/ActiveIntelligenceStrip.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 4: Implement** — `src/components/intelligence/ActiveIntelligenceStrip.tsx`:

```tsx
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { getSetupStatus, getExternalLlm } from "../../lib/tauri";
import { useDaemonVersion } from "../../hooks/useDaemonVersion";

/** One-line honest status of the daemon priority chain (spec §1, council
 *  change c): "serving" only from daemon runtime state, never from config. */
export default function ActiveIntelligenceStrip() {
  const { t } = useTranslation();
  const { supportsHotSwap } = useDaemonVersion(); // ≥0.13 ⇔ setup status carries external_llm
  const { data: status } = useQuery({ queryKey: ["setup-status"], queryFn: getSetupStatus });
  const { data: external } = useQuery({ queryKey: ["external-llm"], queryFn: getExternalLlm });

  const externalConfigured = Boolean(external?.[0]);

  // Top of the chain first: Anthropic → external → on-device → basic.
  let topLine: string;
  if (status?.anthropic_key_configured) {
    topLine = t("intelligenceStrip.servingAnthropic");
  } else if (supportsHotSwap && status?.external_llm) {
    // Daemon ≥0.13 reports runtime state (spec §7.6).
    topLine = status.external_llm.loaded
      ? t("intelligenceStrip.servingExternal")
      : status.external_llm.configured
        ? t("intelligenceStrip.externalRestartPending")
        : status?.local_model_loaded
          ? t("intelligenceStrip.servingOnDevice")
          : t("intelligenceStrip.servingBasic");
  } else if (externalConfigured) {
    // 0.12: config is all we can see — never claim serving.
    topLine = t("intelligenceStrip.externalUnverified");
  } else if (status?.local_model_loaded) {
    topLine = t("intelligenceStrip.servingOnDevice");
  } else {
    topLine = t("intelligenceStrip.servingBasic");
  }

  return (
    <div
      className="rounded-lg px-4 py-3 flex flex-col"
      style={{ backgroundColor: "var(--mem-hover)", border: "1px solid var(--mem-border)", gap: "2px" }}
    >
      <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 500, color: "var(--mem-text)" }}>
        {topLine}
      </span>
      <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
        {t("intelligenceStrip.chain")}
      </span>
    </div>
  );
}
```

- [ ] **Step 5: Assemble the section** — in `src/components/memory/settings/sections/IntelligenceSection.tsx` (created by PR 1), render, in order: `<ActiveIntelligenceStrip />`, then the three provider cards — existing `OnDeviceModelCard`, existing Anthropic block (`ApiKeyCard` + `ModelChoiceSection` as the section already renders today), then `<AnyProviderCard />` (no `groups` filter — settings shows all presets). Keep whatever wrapper/heading the decomposition produced; only the children change:

```tsx
import ActiveIntelligenceStrip from "../../../intelligence/ActiveIntelligenceStrip";
import AnyProviderCard from "../../../intelligence/AnyProviderCard";
// ...existing imports from the decomposition stay...
```

Order inside the section's container:

```tsx
      <ActiveIntelligenceStrip />
      {/* existing on-device + Anthropic content exactly as the decomposition
          left it (OnDeviceModelCard / ApiKeyCard / ModelChoiceSection) */}
      <AnyProviderCard />
```

(Exact surrounding JSX depends on the decomposition output — keep its heading/delay/animation wrapper untouched and match relative import depth to the real file location.)

- [ ] **Step 6: Run tests + type-check**

Run: `pnpm exec tsc -b && pnpm vitest run src/components/intelligence`
Expected: clean / PASS

- [ ] **Step 7: Commit**

```bash
git add src/components/intelligence/ActiveIntelligenceStrip.tsx src/components/intelligence/ActiveIntelligenceStrip.test.tsx src/components/memory/settings/sections/IntelligenceSection.tsx src/i18n/resources.ts
git commit -m "feat: active-intelligence strip (configured vs serving) + Any-provider in settings"
```

---

### Task 9: Wizard intelligence step — 3-way choice

`IntelligenceChoiceStep` (`src/components/SetupWizard.tsx:152-294`) becomes *On this device* / *Cloud API* / *Local server* (spec §2). Cloud pane: vendor pills, Anthropic first — Anthropic pill renders the native `ApiKeyCard`, every other pill renders `AnyProviderCard` locked to that preset. Local pane: `AnyProviderCard` with local presets.

**Files:**
- Modify: `src/components/SetupWizard.tsx` (IntelligenceChoiceStep only)
- Modify: `src/i18n/resources.ts` (`setup.intelligence.*` additions, 3 locales)
- Test: extend `src/components/SetupWizard.test.tsx` (or create `src/components/SetupWizard.intelligence.test.tsx` if per-concern test files are the pattern — check what PR 1 left)

**Interfaces:**
- Consumes: `AnyProviderCard` (Task 7), existing `ApiKeyCard`, `OnDeviceModelCard`, `PROVIDER_PRESETS`.

- [ ] **Step 1: Add i18n keys** — inside `setup.intelligence` in all three locales:

en (add to the existing block):

```ts
      deviceOption: "On this device",        // replaces "On-device model"
      cloudOption: "Cloud API",
      localOption: "Local server",
      cloudNote:
        "Bring a key from any provider. Anthropic connects natively; every other vendor runs through the OpenAI-compatible endpoint.",
      localNote:
        "Point Wenlan at a local server like Ollama or LM Studio. No key, no cloud — inference stays on your machine.",
```

zh-Hans:

```ts
      deviceOption: "在本机运行",
      cloudOption: "云端 API",
      localOption: "本地服务器",
      cloudNote:
        "使用任意服务商的密钥。Anthropic 原生接入；其他服务商通过 OpenAI 兼容端点运行。",
      localNote:
        "将 Wenlan 指向 Ollama 或 LM Studio 等本地服务器。无需密钥、不上云 — 推理保留在你的设备上。",
```

zh-Hant:

```ts
      deviceOption: "在本機執行",
      cloudOption: "雲端 API",
      localOption: "本機伺服器",
      cloudNote:
        "使用任意服務商的金鑰。Anthropic 原生接入;其他服務商透過 OpenAI 相容端點執行。",
      localNote:
        "將 Wenlan 指向 Ollama 或 LM Studio 等本機伺服器。無需金鑰、不上雲 — 推理保留在你的裝置上。",
```

Keep the existing `apiOption` key (other locales reference it) but it becomes unused by this component — delete it from all three locales **only if** `git grep -n "intelligence.apiOption" src/` shows no other user.

- [ ] **Step 2: Replace `IntelligenceChoiceStep`** (lines 152-294) with:

```tsx
function IntelligenceChoiceStep({
  onNext,
  onBack,
}: {
  onNext: () => void;
  onBack: () => void;
}) {
  const { t } = useTranslation();
  const [mode, setMode] = useState<"device" | "cloud" | "local">("device");
  // Cloud vendor pills: Anthropic (native slot) first, then cloud presets
  // (external slot) — spec §2.
  const [cloudVendor, setCloudVendor] = useState<string>("anthropic");
  const cloudPresets = PROVIDER_PRESETS.filter((p) => p.group === "cloud");

  const choiceButtonStyle = (active: boolean): React.CSSProperties => ({
    flex: 1,
    padding: "10px 12px",
    borderRadius: "10px",
    border: "1px solid var(--mem-border)",
    backgroundColor: active ? "rgba(99, 102, 241, 0.12)" : "var(--mem-surface)",
    color: active ? "var(--mem-accent-indigo)" : "var(--mem-text-secondary)",
    fontFamily: "var(--mem-font-body)",
    fontSize: "13px",
    fontWeight: 500,
    textAlign: "left",
  });

  const pillStyle = (active: boolean): React.CSSProperties => ({
    padding: "6px 12px",
    borderRadius: "999px",
    border: "1px solid var(--mem-border)",
    backgroundColor: active ? "rgba(99, 102, 241, 0.12)" : "var(--mem-surface)",
    color: active ? "var(--mem-accent-indigo)" : "var(--mem-text-secondary)",
    fontFamily: "var(--mem-font-body)",
    fontSize: "12px",
    fontWeight: 500,
  });

  const note = (key: string) => (
    <div
      className="rounded-xl px-4 py-3"
      style={{
        backgroundColor: "var(--mem-hover)",
        border: "1px solid var(--mem-border)",
        fontFamily: "var(--mem-font-body)",
        fontSize: "12px",
        color: "var(--mem-text-secondary)",
        lineHeight: 1.6,
      }}
    >
      {t(key)}
    </div>
  );

  return (
    <div
      className="flex flex-col max-w-xl mx-auto"
      style={{ gap: "24px", paddingTop: "24px" }}
    >
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 self-start transition-colors duration-150"
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "13px",
          color: "var(--mem-text-secondary)",
        }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M19 12H5M12 19l-7-7 7-7" />
        </svg>
        {t("setup.back")}
      </button>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        <h1
          style={{
            fontFamily: "var(--mem-font-heading)",
            fontSize: "20px",
            fontWeight: 500,
            color: "var(--mem-text)",
          }}
        >
          {t("setup.intelligence.title")}
        </h1>
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: "var(--mem-text-secondary)",
            lineHeight: "1.5",
          }}
        >
          {t("setup.intelligence.description")}
        </p>
      </div>

      <div className="flex gap-3">
        <button onClick={() => setMode("device")} style={choiceButtonStyle(mode === "device")}>
          {t("setup.intelligence.deviceOption")}
        </button>
        <button onClick={() => setMode("cloud")} style={choiceButtonStyle(mode === "cloud")}>
          {t("setup.intelligence.cloudOption")}
        </button>
        <button onClick={() => setMode("local")} style={choiceButtonStyle(mode === "local")}>
          {t("setup.intelligence.localOption")}
        </button>
      </div>

      {mode === "device" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <OnDeviceModelCard />
          {note("setup.intelligence.deviceNote")}
        </div>
      )}

      {mode === "cloud" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div className="flex flex-wrap gap-2">
            <button onClick={() => setCloudVendor("anthropic")} style={pillStyle(cloudVendor === "anthropic")}>
              Anthropic
            </button>
            {cloudPresets.map((p) => (
              <button key={p.id} onClick={() => setCloudVendor(p.id)} style={pillStyle(cloudVendor === p.id)}>
                {p.name}
              </button>
            ))}
          </div>
          {cloudVendor === "anthropic" ? (
            <ApiKeyCard showNoKeyGuidance={false} />
          ) : (
            <AnyProviderCard
              key={cloudVendor}
              groups={["cloud"]}
              initialPresetId={cloudVendor}
              hidePresetPicker
            />
          )}
          {note("setup.intelligence.cloudNote")}
        </div>
      )}

      {mode === "local" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <AnyProviderCard groups={["local"]} />
          {note("setup.intelligence.localNote")}
        </div>
      )}

      <div
        className="flex items-center gap-3"
        style={{
          paddingTop: "16px",
          borderTop: "1px solid var(--mem-border)",
        }}
      >
        <button
          onClick={onNext}
          className="ml-auto transition-colors duration-150"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: "var(--mem-text-tertiary)",
            background: "none",
            border: "none",
            cursor: "pointer",
          }}
        >
          {t("setup.skip")}
        </button>
        <button
          onClick={onNext}
          className="px-5 py-2 rounded-lg text-sm font-medium transition-colors duration-150"
          style={{
            fontFamily: "var(--mem-font-body)",
            backgroundColor: "var(--mem-accent-indigo)",
            color: "white",
          }}
        >
          {t("setup.continue")}
        </button>
      </div>
    </div>
  );
}
```

Add the imports at the top of `SetupWizard.tsx`:

```tsx
import AnyProviderCard from "./intelligence/AnyProviderCard";
import { PROVIDER_PRESETS } from "./intelligence/providerPresets";
```

- [ ] **Step 3: Add/extend the wizard test.** Follow the existing `SetupWizard.test.tsx` mocking pattern (it exists per PR 1's constraints — read it first, reuse its mock setup verbatim). Add:

```tsx
  it("intelligence step offers device, cloud, and local server", async () => {
    // ...render wizard, navigate to intelligence step per existing helpers...
    expect(screen.getByText("On this device")).toBeInTheDocument();
    expect(screen.getByText("Cloud API")).toBeInTheDocument();
    expect(screen.getByText("Local server")).toBeInTheDocument();
  });

  it("cloud pane lists Anthropic first and routes non-Anthropic vendors to the external card", async () => {
    // ...navigate, click "Cloud API"...
    const pills = screen.getAllByRole("button");
    // Anthropic pill renders the native key card
    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    // pick OpenAI → AnyProviderCard appears (its title)
    await userEvent.click(screen.getByText("OpenAI"));
    expect(await screen.findByText("Any provider")).toBeInTheDocument();
  });
```

(Adapt to the file's actual render/navigation helpers; assert on translated strings via the same i18n setup the file already uses.)

- [ ] **Step 4: Run the wizard tests**

Run: `pnpm vitest run src/components/SetupWizard.test.tsx`
Expected: PASS — existing tests still green (the step keeps `setup.intelligence.title`, skip, and continue semantics), new tests pass

- [ ] **Step 5: Commit**

```bash
git add src/components/SetupWizard.tsx src/components/SetupWizard.test.tsx src/i18n/resources.ts
git commit -m "feat: 3-way wizard intelligence choice (device / cloud / local server)"
```

---

### Task 10: Wizard import step — dual path (chat history + vault)

The import step (spec §2) offers *Chat history* and *Obsidian vault / notes folder* side by side. Chat history opens the existing `ImportView` flow; vault renders `VaultConnectCard` inline with its "Indexed N files · M memories" check before Continue.

**Files:**
- Modify: `src/components/SetupWizard.tsx` (new `ImportStep` component + the `step === "import"` branch, currently lines 1369-1385)
- Modify: `src/i18n/resources.ts` (`setup.import.*` additions, 3 locales)

**Interfaces:**
- Consumes: `ImportView` (existing props: `onBack`, `wizardMode`, `wizardHint`, `onPhaseChange`, `onSkip`, `onComplete`), `VaultConnectCard` (Task 6).

- [ ] **Step 1: Add i18n keys** — in `setup.import` (all 3 locales):

en:

```ts
    import: {
      laterHint:
        "You can import ChatGPT or Claude chat history later from <strong>Settings > Sources</strong>.",
      title: "Bring what you already know",
      description: "Import past AI conversations, connect your notes, or both.",
      chatPathTitle: "Chat history",
      chatPathDescription: "Import ChatGPT or Claude conversation exports.",
      chatPathCta: "Import chat history",
      vaultPathTitle: "Obsidian vault / notes folder",
    },
```

zh-Hans:

```ts
    import: {
      laterHint:
        "之后可以在<strong>设置 > 来源</strong>中导入 ChatGPT 或 Claude 聊天记录。",
      title: "带上你已有的知识",
      description: "导入过去的 AI 对话、连接你的笔记，或两者都做。",
      chatPathTitle: "聊天记录",
      chatPathDescription: "导入 ChatGPT 或 Claude 的对话导出文件。",
      chatPathCta: "导入聊天记录",
      vaultPathTitle: "Obsidian 仓库 / 笔记文件夹",
    },
```

zh-Hant:

```ts
    import: {
      laterHint:
        "之後可以在<strong>設定 > 來源</strong>中匯入 ChatGPT 或 Claude 聊天記錄。",
      title: "帶上你已有的知識",
      description: "匯入過去的 AI 對話、連接你的筆記,或兩者都做。",
      chatPathTitle: "聊天記錄",
      chatPathDescription: "匯入 ChatGPT 或 Claude 的對話匯出檔案。",
      chatPathCta: "匯入聊天記錄",
      vaultPathTitle: "Obsidian 儲存庫 / 筆記資料夾",
    },
```

(`laterHint` already exists — keep the existing translation, add the new siblings.)

- [ ] **Step 2: Add the `ImportStep` component** — in `SetupWizard.tsx`, after `IntelligenceChoiceStep`:

```tsx
// ── Import Step (dual path: chat history | vault) ───────────────────────

function ImportStep({
  onBack,
  onSkip,
  onComplete,
  onPhaseChange,
  importHint,
}: {
  onBack: () => void;
  onSkip: () => void;
  onComplete: (source: string, result: ImportResult) => void;
  onPhaseChange: (phase: string) => void;
  importHint: React.ReactNode;
}) {
  const { t } = useTranslation();
  const [pathChoice, setPathChoice] = useState<"none" | "chat">("none");

  if (pathChoice === "chat") {
    return (
      <ImportView
        onBack={() => setPathChoice("none")}
        wizardMode
        wizardHint={importHint}
        onPhaseChange={onPhaseChange}
        onSkip={onSkip}
        onComplete={onComplete}
      />
    );
  }

  return (
    <div className="flex flex-col max-w-xl mx-auto" style={{ gap: "24px", paddingTop: "24px" }}>
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 self-start transition-colors duration-150"
        style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text-secondary)" }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M19 12H5M12 19l-7-7 7-7" />
        </svg>
        {t("setup.back")}
      </button>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        <h1 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "20px", fontWeight: 500, color: "var(--mem-text)" }}>
          {t("setup.import.title")}
        </h1>
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text-secondary)", lineHeight: "1.5" }}>
          {t("setup.import.description")}
        </p>
      </div>

      <div
        className="rounded-xl p-4 flex items-center justify-between"
        style={{ border: "1px solid var(--mem-border)", backgroundColor: "var(--mem-surface)", gap: "12px" }}
      >
        <div>
          <h3 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "15px", fontWeight: 500, color: "var(--mem-text)" }}>
            {t("setup.import.chatPathTitle")}
          </h3>
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", marginTop: "4px" }}>
            {t("setup.import.chatPathDescription")}
          </p>
        </div>
        <button
          onClick={() => setPathChoice("chat")}
          className="rounded-md px-4 py-2 text-sm font-medium shrink-0"
          style={{ backgroundColor: "var(--mem-accent-indigo)", color: "white", fontFamily: "var(--mem-font-body)" }}
        >
          {t("setup.import.chatPathCta")}
        </button>
      </div>

      <VaultConnectCard variant="wizard" />

      <div className="flex items-center" style={{ paddingTop: "16px", borderTop: "1px solid var(--mem-border)" }}>
        <button
          onClick={onSkip}
          className="ml-auto transition-colors duration-150"
          style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer" }}
        >
          {t("setup.skip")}
        </button>
        <button
          onClick={onSkip}
          className="px-5 py-2 rounded-lg text-sm font-medium transition-colors duration-150 ml-3"
          style={{ fontFamily: "var(--mem-font-body)", backgroundColor: "var(--mem-accent-indigo)", color: "white" }}
        >
          {t("setup.continue")}
        </button>
      </div>
    </div>
  );
}
```

Add the import: `import VaultConnectCard from "./memory/sources/VaultConnectCard";`

- [ ] **Step 3: Rewire the `step === "import"` branch** (currently lines 1369-1385) to:

```tsx
        {step === "import" && (
          <ImportStep
            onBack={() => setStep("intelligence-choice")}
            importHint={(
              <Trans
                i18nKey="setup.import.laterHint"
                components={{ strong: <strong /> }}
              />
            )}
            onPhaseChange={setImportPhase}
            onSkip={() => setStep("connect")}
            onComplete={(_source, result) => {
              setImportResult(result);
              setStep("connect");
            }}
          />
        )}
```

- [ ] **Step 4: Run wizard tests + type-check**

Run: `pnpm exec tsc -b && pnpm vitest run src/components/SetupWizard.test.tsx`
Expected: clean / PASS. If an existing test drove the import step directly through `ImportView`, update it to click "Import chat history" first (that is a real behavior change the spec mandates — do not weaken the assertion, route through the new card).

- [ ] **Step 5: Commit**

```bash
git add src/components/SetupWizard.tsx src/i18n/resources.ts
git commit -m "feat: wizard import step offers chat history and vault side by side"
```

---

### Task 11: Connect matrix — web platform cards shared by wizard + settings

Web cards for Claude.ai and ChatGPT.com (numbered connector instructions, tunnel-URL copy, **no-auth boundary warning** — council change f), plus a per-card "Set up" list for apps/CLIs in Settings. The wizard's existing client checkboxes already pick up the Task 4 registry additions automatically; this task adds the web cards to both surfaces and the client setup list to Settings.

**Files:**
- Create: `src/components/connect/WebPlatformCards.tsx`
- Create: `src/components/connect/ClientSetupList.tsx`
- Test: `src/components/connect/WebPlatformCards.test.tsx`
- Modify: `src/components/SetupWizard.tsx` (ConnectStep web section, ~line 632)
- Modify: `src/components/memory/settings/sections/AgentsSection.tsx`
- Modify: `src/i18n/resources.ts` (`connectMatrix.*`, 3 locales)

**Interfaces:**
- Consumes: `getRemoteAccessStatus` (`src/lib/tauri.ts:2068`, returns `{status:"connected"; tunnel_url; relay_url}` when up), `listAgents` (`:1768`), `detectMcpClients`, `writeMcpConfig`, `RemoteAccessPanel` (`mode="compact"|"full"`).
- Produces: `<WebPlatformCards />` (self-contained; owns its polling), `<ClientSetupList />` (self-contained).

- [ ] **Step 1: Add i18n keys** — top-level `connectMatrix` (3 locales):

en:

```ts
  connectMatrix: {
    claudeTitle: "Claude.ai",
    chatgptTitle: "ChatGPT.com",
    claudeStep1: "Open Claude.ai → Settings → Connectors",
    claudeStep2: "Choose \"Add custom connector\"",
    claudeStep3: "Paste your Remote Access URL below",
    chatgptStep1: "Open ChatGPT.com → Settings → Connectors",
    chatgptStep2: "Enable Advanced / developer mode",
    chatgptStep3: "Add the Remote Access URL below as an MCP connector",
    copyUrl: "Copy URL",
    copied: "Copied",
    tunnelOff: "Turn on Remote Access above to get your connection URL.",
    noAuthWarning:
      "Anyone with this URL can read and write your memories — treat it like a password.",
    connectedHint: "Connected — a new agent appeared",
    setUp: "Set up",
    settingUp: "Setting up…",
    configured: "Configured",
    notDetected: "Not installed",
    appsTitle: "Apps & CLIs",
    webTitle: "Web — Claude.ai & ChatGPT",
    manualTitle: "Manual / anything else",
  },
```

zh-Hans:

```ts
  connectMatrix: {
    claudeTitle: "Claude.ai",
    chatgptTitle: "ChatGPT.com",
    claudeStep1: "打开 Claude.ai → 设置 → 连接器",
    claudeStep2: "选择「添加自定义连接器」",
    claudeStep3: "粘贴下方的远程访问 URL",
    chatgptStep1: "打开 ChatGPT.com → 设置 → 连接器",
    chatgptStep2: "启用高级 / 开发者模式",
    chatgptStep3: "将下方的远程访问 URL 添加为 MCP 连接器",
    copyUrl: "复制 URL",
    copied: "已复制",
    tunnelOff: "先在上方开启远程访问以获取连接 URL。",
    noAuthWarning: "任何拥有此 URL 的人都能读写你的记忆 — 请像密码一样保管。",
    connectedHint: "已连接 — 检测到新代理",
    setUp: "设置",
    settingUp: "设置中…",
    configured: "已配置",
    notDetected: "未安装",
    appsTitle: "应用与命令行工具",
    webTitle: "网页 — Claude.ai 与 ChatGPT",
    manualTitle: "手动 / 其他工具",
  },
```

zh-Hant:

```ts
  connectMatrix: {
    claudeTitle: "Claude.ai",
    chatgptTitle: "ChatGPT.com",
    claudeStep1: "開啟 Claude.ai → 設定 → 連接器",
    claudeStep2: "選擇「新增自訂連接器」",
    claudeStep3: "貼上下方的遠端存取 URL",
    chatgptStep1: "開啟 ChatGPT.com → 設定 → 連接器",
    chatgptStep2: "啟用進階 / 開發者模式",
    chatgptStep3: "將下方的遠端存取 URL 新增為 MCP 連接器",
    copyUrl: "複製 URL",
    copied: "已複製",
    tunnelOff: "先在上方開啟遠端存取以取得連接 URL。",
    noAuthWarning: "任何擁有此 URL 的人都能讀寫你的記憶 — 請像密碼一樣保管。",
    connectedHint: "已連接 — 偵測到新代理",
    setUp: "設定",
    settingUp: "設定中…",
    configured: "已設定",
    notDetected: "未安裝",
    appsTitle: "應用程式與命令列工具",
    webTitle: "網頁 — Claude.ai 與 ChatGPT",
    manualTitle: "手動 / 其他工具",
  },
```

- [ ] **Step 2: Write the failing test** — `src/components/connect/WebPlatformCards.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../i18n";

const mocks = vi.hoisted(() => ({
  getRemoteAccessStatus: vi.fn(),
  listAgents: vi.fn(),
}));
vi.mock("../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/tauri")>();
  return { ...actual, ...mocks };
});

import WebPlatformCards from "./WebPlatformCards";

function renderCards() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <WebPlatformCards />
    </QueryClientProvider>
  );
}

describe("WebPlatformCards", () => {
  beforeEach(() => {
    Object.values(mocks).forEach((m) => m.mockReset());
    mocks.listAgents.mockResolvedValue([]);
  });

  it("both web cards carry the no-auth boundary warning (council change f)", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({
      status: "connected", tunnel_url: "https://x.trycloudflare.com", token: "t", relay_url: null,
    });
    renderCards();
    expect(await screen.findByText("Claude.ai")).toBeInTheDocument();
    expect(screen.getByText("ChatGPT.com")).toBeInTheDocument();
    expect(screen.getAllByText(/treat it like a password/)).toHaveLength(2);
  });

  it("shows the connection URL when the tunnel is up", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({
      status: "connected", tunnel_url: "https://x.trycloudflare.com", token: "t",
      relay_url: "https://relay.example/abc",
    });
    renderCards();
    // relay URL preferred (what users hand to Claude.ai/ChatGPT)
    expect((await screen.findAllByText("https://relay.example/abc")).length).toBeGreaterThan(0);
  });

  it("prompts to enable Remote Access when the tunnel is off", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({ status: "off" });
    renderCards();
    expect((await screen.findAllByText(/Turn on Remote Access/)).length).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 3: Run to verify failure**

Run: `pnpm vitest run src/components/connect/WebPlatformCards.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 4: Implement `WebPlatformCards`** — `src/components/connect/WebPlatformCards.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { getRemoteAccessStatus, listAgents } from "../../lib/tauri";

/** Per-platform web connect cards (spec §2a group 1). Verification is the
 *  existing listAgents delta poll — best-effort attribution: a new agent in
 *  the poll window flips the last-copied card (hint, not proof). */
export default function WebPlatformCards() {
  const { t } = useTranslation();
  const [copiedPlatform, setCopiedPlatform] = useState<string | null>(null);
  const [connectedPlatform, setConnectedPlatform] = useState<string | null>(null);
  const baselineCount = useRef<number | null>(null);

  const { data: remote } = useQuery({
    queryKey: ["remote-access-status"],
    queryFn: getRemoteAccessStatus,
    refetchInterval: 3000,
  });
  const url =
    remote?.status === "connected" ? (remote.relay_url ?? remote.tunnel_url) : null;

  const { data: agents } = useQuery({
    queryKey: ["web-connect-agents"],
    queryFn: listAgents,
    refetchInterval: copiedPlatform !== null && connectedPlatform === null ? 3000 : false,
  });
  useEffect(() => {
    if (!agents) return;
    if (baselineCount.current === null) {
      baselineCount.current = agents.length;
      return;
    }
    if (copiedPlatform !== null && agents.length > baselineCount.current) {
      setConnectedPlatform(copiedPlatform);
    }
  }, [agents, copiedPlatform]);

  const copy = async (platform: string) => {
    if (!url) return;
    await navigator.clipboard.writeText(url);
    baselineCount.current = agents?.length ?? baselineCount.current;
    setCopiedPlatform(platform);
  };

  const card = (platform: "claude" | "chatgpt", title: string, steps: string[]) => (
    <div
      className="rounded-xl p-4 flex flex-col"
      style={{ border: "1px solid var(--mem-border)", backgroundColor: "var(--mem-surface)", gap: "10px" }}
    >
      <div className="flex items-center justify-between">
        <h3 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "15px", fontWeight: 500, color: "var(--mem-text)" }}>
          {title}
        </h3>
        {connectedPlatform === platform && (
          <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-accent-sage)" }}>
            {t("connectMatrix.connectedHint")}
          </span>
        )}
      </div>
      <ol style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", lineHeight: 1.7, paddingLeft: "18px", listStyle: "decimal", margin: 0 }}>
        {steps.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ol>
      {url ? (
        <div className="flex items-center gap-2">
          <code
            className="flex-1 truncate rounded-md px-2 py-1.5"
            style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", backgroundColor: "var(--mem-bg)", border: "1px solid var(--mem-border)", color: "var(--mem-text)" }}
          >
            {url}
          </code>
          <button
            onClick={() => copy(platform)}
            className="rounded-md px-3 py-1.5 text-xs shrink-0"
            style={{ border: "1px solid var(--mem-border)", color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}
          >
            {copiedPlatform === platform ? t("connectMatrix.copied") : t("connectMatrix.copyUrl")}
          </button>
        </div>
      ) : (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-tertiary)" }}>
          {t("connectMatrix.tunnelOff")}
        </p>
      )}
      {/* No-auth boundary (council change f, commit 3a272d0): always visible. */}
      <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-accent-amber)", lineHeight: 1.5 }}>
        {t("connectMatrix.noAuthWarning")}
      </p>
    </div>
  );

  return (
    <div className="flex flex-col" style={{ gap: "12px" }}>
      {card("claude", t("connectMatrix.claudeTitle"), [
        t("connectMatrix.claudeStep1"),
        t("connectMatrix.claudeStep2"),
        t("connectMatrix.claudeStep3"),
      ])}
      {card("chatgpt", t("connectMatrix.chatgptTitle"), [
        t("connectMatrix.chatgptStep1"),
        t("connectMatrix.chatgptStep2"),
        t("connectMatrix.chatgptStep3"),
      ])}
    </div>
  );
}
```

- [ ] **Step 5: Implement `ClientSetupList`** — `src/components/connect/ClientSetupList.tsx` (settings-side per-card one-click setup; the wizard keeps its batch flow):

```tsx
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { detectMcpClients, writeMcpConfig } from "../../lib/tauri";

/** Apps & CLIs group (spec §2a group 2): one row per registry client,
 *  "Set up" writes the MCP config, path in mono, errors verbatim. */
export default function ClientSetupList() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [busy, setBusy] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const { data: clients } = useQuery({ queryKey: ["mcp-clients"], queryFn: detectMcpClients });

  const setUp = async (clientType: string) => {
    setBusy(clientType);
    setErrors((prev) => ({ ...prev, [clientType]: "" }));
    try {
      await writeMcpConfig(clientType);
      queryClient.invalidateQueries({ queryKey: ["mcp-clients"] });
    } catch (err) {
      setErrors((prev) => ({ ...prev, [clientType]: String(err) }));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col" style={{ gap: "8px" }}>
      {(clients ?? []).map((client) => (
        <div
          key={client.client_type}
          className="rounded-lg px-3 py-2.5 flex items-center gap-3"
          style={{ border: "1px solid var(--mem-border)", backgroundColor: "var(--mem-surface)" }}
        >
          <div className="flex-1 min-w-0">
            <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 500, color: "var(--mem-text)", margin: 0 }}>
              {client.name}
            </p>
            <p className="truncate" style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)", margin: 0 }}>
              {client.config_path}
            </p>
            {errors[client.client_type] && (
              <p className="text-red-500" style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", margin: 0 }}>
                {errors[client.client_type]}
              </p>
            )}
          </div>
          {client.already_configured ? (
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-accent-sage)" }}>
              {t("connectMatrix.configured")}
            </span>
          ) : client.detected ? (
            <button
              onClick={() => setUp(client.client_type)}
              disabled={busy === client.client_type}
              className="rounded-md px-3 py-1.5 text-xs disabled:opacity-50 shrink-0"
              style={{ backgroundColor: "var(--mem-accent-indigo)", color: "white", fontFamily: "var(--mem-font-body)" }}
            >
              {busy === client.client_type ? t("connectMatrix.settingUp") : t("connectMatrix.setUp")}
            </button>
          ) : (
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
              {t("connectMatrix.notDetected")}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 6: Wire into the wizard.** In `SetupWizard.tsx` ConnectStep, replace the web-tools section (the block under `<SectionLabel>{t("setup.connect.webTools")}</SectionLabel>`, ~line 632 — currently a `RemoteAccessPanel mode="compact"` + prose) with:

```tsx
        <SectionLabel>{t("setup.connect.webTools")}</SectionLabel>
        <RemoteAccessPanel mode="compact" />
        <WebPlatformCards />
```

(Keep `RemoteAccessPanel` — it owns the tunnel on/off lifecycle; the cards read status only.) Add `import WebPlatformCards from "./connect/WebPlatformCards";`.

- [ ] **Step 7: Wire into settings.** In `src/components/memory/settings/sections/AgentsSection.tsx` (post-decomposition), add under the existing Remote Access block (which renders `RemoteAccessPanel mode="full"`):

```tsx
      <SectionHeader title={t("connectMatrix.webTitle")} />
      <WebPlatformCards />
      <SectionHeader title={t("connectMatrix.appsTitle")} />
      <ClientSetupList />
```

using the section's existing `SectionHeader` primitive (from `../primitives`) and matching its layout wrappers/spacing; imports:

```tsx
import WebPlatformCards from "../../../connect/WebPlatformCards";
import ClientSetupList from "../../../connect/ClientSetupList";
```

(Adjust relative depth to the real file location. The existing manual MCP-snippet block in the section stays — it is spec §2a group 3.)

- [ ] **Step 8: Run tests + type-check**

Run: `pnpm exec tsc -b && pnpm vitest run src/components/connect src/components/SetupWizard.test.tsx`
Expected: clean / PASS

- [ ] **Step 9: Commit**

```bash
git add src/components/connect src/components/SetupWizard.tsx src/components/memory/settings/sections/AgentsSection.tsx src/i18n/resources.ts
git commit -m "feat: connect matrix — web platform cards + client setup list in wizard and settings"
```

---

### Task 12: Full gates + draft PR

- [ ] **Step 1: Run every gate**

```bash
pnpm build          # tsc -b && vite build
pnpm test           # full vitest suite — decomposition tests must still pass
pnpm test:i18n      # locale parity + hardcodedCopyGuard
cd app && cargo fmt --check --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test && cd ..
```

Expected: all green. Fix forward anything red before proceeding (no assertion weakening; no skipping tests).

- [ ] **Step 2: Push and open the draft PR** (stacked on PR 1's branch)

```bash
git push -u origin settings-onboarding-features
gh pr create --draft --base settings-decomposition \
  --title "Settings + onboarding redesign: multi-provider models, vault connect, connect matrix" \
  --body "$(cat <<'EOF'
PR 2 of 3 (spec docs/superpowers/specs/2026-07-10-settings-onboarding-redesign-design.md, §8):
stacked on #<PR1-number> (settings decomposition). Fully functional against pinned daemon 0.12;
API-key/hot-swap/serving-status features light up via useDaemonVersion() when the backend pin
reaches 0.13 (companion daemon PR: 7xuanlu/wenlan feat/external-llm-key).

- §1 Models hub: active-intelligence strip (configured vs serving vs restart-pending),
  Any-provider preset card (OpenAI/Gemini/Groq/OpenRouter/Mistral/DeepSeek/xAI/Ollama/LM Studio/custom),
  model auto-discovery, Test button, precedence warning, 0.13 key gating
- §2 Wizard: 3-way intelligence choice (device / cloud / local server), dual-path import
- §2a Connect matrix: Claude.ai + ChatGPT.com cards with no-auth boundary warning,
  MCP registry adds Claude Desktop, Gemini CLI, Codex CLI (format-preserving TOML via toml_edit)
- §3 Vault connect: recursive bounded detection, per-source-type validity, zero-count never blocks
  submit, post-connect "Indexed N files · M memories"

Per-preset /models validation findings: <fill in from Task 7 live validation>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

(`gh` must run with the sandbox disabled. Fill `<PR1-number>` from the decomposition PR.)

- [ ] **Step 3: Report** the PR URL, gate results, and the per-preset validation findings.

---

## Self-Review Notes

- **Spec coverage:** §1 → Tasks 2, 3, 7, 8; §2 → Tasks 9, 10; §2a → Tasks 4, 11; §3 → Tasks 5, 6; §5 → constraints applied in every UI task; §6 → Tasks 2, 3, 4 (+Task 1 hook); §8 → Task 1 gating + Task 12 PR stacking. §4 and §7 live in their own plans (`2026-07-10-settings-decomposition.md`, `2026-07-10-daemon-external-llm-key.md`).
- **Known judgment points for implementers** (call out in task reports rather than improvising silently): the exact `useApiKeyStatus` return shape (Task 7 Step 5 note), the i18n side-effect import path in component tests (Task 6 Step 2 note), and the decomposition's final section-file JSX (Tasks 8, 11 integration steps).
