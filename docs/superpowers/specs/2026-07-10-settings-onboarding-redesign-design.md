# Settings + Onboarding Redesign — Design Spec

- **Date:** 2026-07-10
- **Status:** Approved with changes — `/boule:debate` council 2026-07-10, unanimous
  3/3 approve-with-changes (claude main-loop, gpt-5.6-sol xhigh, Gemini 3.1 Pro);
  this revision folds in the judge's six required changes (see "Council review")
- **Repos:** `wenlan-app` (primary PR) + `7xuanlu/wenlan` (small companion daemon PR)
- **Driver:** Surface the multi-provider intelligence the daemon already supports (avoid
  Anthropic lock-in), bring Obsidian import into onboarding with a real check, and
  decompose/refresh the settings surface.

## Context

The daemon already runs three LLM backends — on-device GGUF, Anthropic BYOK, and any
OpenAI-compatible endpoint (Ollama, LM Studio, vLLM) — but the app UI only exposes the
first two. Obsidian vault import already ships in Settings → Sources but is invisible
during onboarding, and its client-side check only scans the top level of the folder.
`SettingsPage.tsx` is a single 1,111-line file.

### Current state (verified 2026-07-10)

| Fact | Evidence |
| --- | --- |
| Settings = 1 file, 6 prop-switched groups | `src/components/memory/SettingsPage.tsx` (1111 lines); groups in `src/components/memory/settings/SettingsSidebar.tsx:52` |
| Intelligence UI is Anthropic-only + on-device | `src/components/intelligence/IntelligenceSetup.tsx` — `ApiKeyCard:41`, hardcoded `ANTHROPIC_MODELS:19`, `OnDeviceModelCard:246` |
| External-endpoint plumbing exists, **no UI** | `getExternalLlm`/`setExternalLlm`/`testExternalLlm` `src/lib/tauri.ts:395-414`; app `app/src/config.rs:77`; app cmds `app/src/search.rs:3774+` |
| Daemon: OpenAI-compatible provider, **no auth header** | `wenlan/crates/wenlan-core/src/llm_provider.rs:1196` — POSTs `{endpoint}/chat/completions`, 60s timeout, no api_key field |
| Daemon loads external provider **only at startup** | `wenlan/crates/wenlan-server/src/main.rs:479-491`; `PUT /api/config` persists but does not hot-swap (`config_routes.rs:38`); Anthropic key DOES hot-load (`apply_anthropic_provider`, `config_routes.rs:129`) |
| Provider priority (maintenance + steep paths) | `synthesis_llm → api_llm → external_llm → llm(on-device)` — `wenlan/crates/wenlan-server/src/scheduler.rs:325-328`; steep path verified same order during council defense (`wenlan/crates/wenlan-core/src/refinery/mod.rs:536`) |
| Endpoint test endpoint exists | `POST /api/llm/test` `wenlan/crates/wenlan-server/src/routes.rs:1186` — OpenAI-compatible only |
| Obsidian connect ships in Settings → Sources | `src/components/memory/sources/AddSourceDialog.tsx` — browse, `.obsidian` badge, `addSource("obsidian"|"directory")`, one-shot sync kick |
| Vault check is shallow | `AddSourceDialog.tsx` uses non-recursive `readDir`; vault with notes only in subfolders shows `docCount: 0` and **cannot be submitted** (`canSubmit` gate) |
| Daemon vault validation | `POST /api/sources` — path exists + is_dir + ≥1 markdown anywhere (`has_any_markdown`); `.obsidian/` cosmetic only |
| Wizard = 5 steps, binary intelligence choice | `src/components/SetupWizard.tsx` — `welcome → intelligence-choice → import → connect → verify`; `IntelligenceChoiceStep` offers device/api only |
| Version surface | `GET /api/health` → `{status, db_initialized, version}`; app reads it (`src/lib/tauri.ts:179`). App pins backend **0.12** |
| Embedding model | Hardcoded fastembed BGE-base-en-v1.5 (`wenlan/crates/wenlan-core/src/db.rs:2027`) — not configurable |

## Goals

1. **Vendor-agnostic intelligence UI** — one "Models" hub used by both Settings and
   the setup wizard: On-device / Anthropic (native) / **Any provider** — a preset-driven
   OpenAI-compatible card covering cloud vendors (OpenAI, Gemini, Groq, OpenRouter,
   Mistral, DeepSeek, xAI) and local servers (Ollama, LM Studio, vLLM) through one
   mechanism — with model auto-discovery, a Test button, and honest "which provider
   is actually serving" status. Anthropic is one option among many, not the default
   framing.
2. **API keys for any cloud vendor** — companion daemon change adds
   `external_llm_api_key` (`Authorization: Bearer`), which is what makes the keyed
   cloud presets work. Against pinned 0.12, Anthropic (native) + keyless local
   servers still work.
3. **Obsidian in onboarding, with a real check** — vault connect card in the wizard's
   import step; recursive client-side detection; post-connect sync verification
   ("Indexed N files · M memories").
4. **Connect everywhere** — the wizard connect step and Settings → Connected Agents
   present the same platform matrix: web (Claude.ai, ChatGPT.com via the Remote
   Access tunnel), desktop apps, and CLIs (Claude Code, Cursor today; + Claude
   Desktop, Codex CLI, Gemini CLI added), each with one-click config write or
   copy-paste instructions and live verification.
5. **Settings decomposition + visual refresh** — split the 1,111-line file into
   per-section modules; consistent row/card idiom; wizard + settings visual polish
   inside the existing `--mem-*` token system.

## Non-goals

- No provider registry / N-provider routing rework in the daemon (3 slots stay).
- No embedding or reranker model configuration UI.
- No native OpenAI/Gemini SDKs — OpenAI-compatible `/chat/completions` is the
  contract; cloud vendors are presets over it, not new provider implementations.
- No OAuth / account-linking for Claude.ai or ChatGPT.com — the Remote Access tunnel
  URL + platform connector settings is the connection mechanism.
- No backend-pin bump in this work (separate existing workflow); the app PR must be
  fully functional against pinned daemon 0.12.
- No sidebar IA change; hidden "capture" group stays hidden.
- No new fonts or color tokens (any new token must land in both light and dark blocks
  of `src/index.css` — avoid unless forced).

## Design

### 1. Models hub (Settings → Intelligence section, id unchanged)

Replace `IntelligenceSection` content with:

- **Active-intelligence strip** — one line stating the daemon priority chain
  (Anthropic → external endpoint → on-device → basic memory) and which provider the
  strip believes is on top. The strip distinguishes **configured vs serving** — it
  never claims "serving" from config alone (council change c):
  - *Serving: X* — only when the daemon confirms runtime state (≥ 0.13:
    `/api/setup/status` gains `external_llm: {configured, loaded}`, §7; Anthropic
    and on-device state are already reported today).
  - *Configured — restart pending* — external slot changed but not loaded (always
    on 0.12, where `PUT /api/config` persists without hot-swap; on ≥ 0.13 only in
    the window before hot-swap confirms).
  - *Configured (unverified)* — derived from config where the daemon exposes no
    runtime signal (external slot on 0.12). Copy states the chain explicitly so
    multi-configured setups aren't mysterious.
- **Three provider cards** (not mutually exclusive — mirrors daemon semantics):
  1. **On-device** — existing `OnDeviceModelCard` behavior unchanged (GGUF
     `qwen3-4b`/`qwen3.5-9b`, RAM gating, download progress).
  2. **Anthropic** — existing key + routine/synthesis model selects; hot-loads via
     `PUT /api/setup/anthropic-key` as today.
  3. **Any provider (new UI)** — one preset-driven OpenAI-compatible card. A vendor
     preset picker drives the fields; "Custom…" leaves everything editable:

     | Preset | Endpoint quick-fill | Key |
     | --- | --- | --- |
     | OpenAI | `https://api.openai.com/v1` | required |
     | Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | required `[verify compat /models at impl]` |
     | Groq | `https://api.groq.com/openai/v1` | required |
     | OpenRouter | `https://openrouter.ai/api/v1` | required |
     | Mistral | `https://api.mistral.ai/v1` | required |
     | DeepSeek | `https://api.deepseek.com/v1` | required |
     | xAI | `https://api.x.ai/v1` | required |
     | Ollama (local) | `http://localhost:11434/v1` | none |
     | LM Studio (local) | `http://localhost:1234/v1` | none |
     | Custom… | free | optional |

     - Every preset's `/models` + `/chat/completions` compatibility is validated
       live at implementation time, not assumed from docs (council dissent note);
       a preset whose `/models` shape drifts ships with free-text model entry only.
     - Endpoint URL (http/https only), editable regardless of preset.
     - Model — combobox: auto-populated via **new app-crate command**
       `list_external_models(endpoint, api_key?) → Vec<String>` (GET
       `{endpoint}/models`, OpenAI `{data:[{id}]}` shape, Bearer header when a key
       is entered, 5s timeout); free-text fallback when discovery fails.
     - API key — **keyed presets are disabled (with explanation) when daemon < 0.13**
       (see §7/§8); keyless local presets work on 0.12.
     - **Test** button → existing `POST /api/llm/test` (gains optional `api_key`
       in the daemon PR); verbatim error on failure.
     - Save → existing `setExternalLlm` path. Status note: "Applied" when daemon
       hot-swaps (≥ 0.13), else "Restart Wenlan to apply".
     - **Precedence warning**: when an Anthropic key is also configured, the card
       shows "Anthropic takes precedence while its key is set" (daemon priority
       chain; the active strip is the single source of truth).

The daemon keeps exactly two remote slots (native Anthropic + one OpenAI-compatible
external), so the UI never pretends more than one non-Anthropic vendor can be active
at once — the preset picker configures *the* external slot.

Cards live in `src/components/intelligence/` (existing home) and are shared verbatim
by the wizard.

### 2. Onboarding wizard

Step skeleton unchanged (`welcome → intelligence → import → connect → verify`), all
five steps get the visual refresh; three steps change functionally:

- **Intelligence step** — 3-way choice: *On this device* / *Cloud API* / *Local
  server*. "Cloud API" renders the Any-provider card filtered to cloud presets
  (Anthropic listed first among OpenAI, Gemini, Groq, … — picking Anthropic routes
  to the native card/slot, everything else to the external slot); "Local server"
  renders the same card filtered to Ollama / LM Studio / Custom. Skip stays
  available (basic-memory mode).
- **Import step** — two paths side by side:
  - *Chat history* — existing `ImportView`/`ImportFlow` (ChatGPT/Claude exports).
  - *Obsidian vault / notes folder* — new shared `VaultConnectCard` (§3).
- **Connect step** — restructured into an explicit platform matrix (shared with
  Settings → Connected Agents, see §2a).

### 2a. Connect surfaces (wizard connect step + Settings → Connected Agents)

Today `detect_mcp_clients()` (`app/src/mcp_config.rs:53`) supports only Cursor +
Claude Code; Claude Desktop is deliberately skipped; web is a `RemoteAccessPanel`
plus a manual JSON snippet. Redesign into three groups, same cards in wizard and
settings:

1. **Web — Claude.ai & ChatGPT.com** — per-platform cards with numbered connector
   instructions (Claude.ai: Settings → Connectors → Add custom connector;
   ChatGPT.com: Settings → Connectors → Advanced/developer mode), a copy button for
   the Remote Access tunnel URL, and the existing tunnel lifecycle from
   `RemoteAccessPanel` (`src/components/memory/RemoteAccessPanel.tsx`). Each web
   card carries the one-line no-auth boundary warning (council change f, echoing
   commit 3a272d0): anyone with the tunnel URL can read/write memories — treat the
   URL as a secret. Verification: the existing `listAgents` poll — card flips to
   "Connected" when a new agent appears. This is delta-attribution and best-effort:
   two platforms connecting in the same poll window can be misattributed, so the
   flip is a hint, not proof of *that* platform (council dissent note). No OAuth
   flows; the tunnel-URL connector is the mechanism (non-goal otherwise).
2. **Apps & CLIs — one-click config write** — extend the client registry in
   `app/src/mcp_config.rs`:
   - existing: Cursor (`~/.cursor/mcp.json`), Claude Code (`~/.claude.json`);
   - added: Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`
     — path helper already exists at `mcp_config.rs:25`; revisit the "remote-only"
     skip comment), Gemini CLI (`~/.gemini/settings.json`, same `mcpServers` JSON
     shape), Codex CLI (`~/.codex/config.toml` — **TOML, `mcp_servers` table**;
     a **format-preserving** merge via `toml_edit` (council change d): user
     comments, key order, and unrelated tables survive the upsert byte-for-byte —
     a parse-and-reserialize writer is not acceptable for a user's hand-edited
     config).
   - Each card: detected / configured state (existing struct fields), "Set up"
     writes config, path shown in mono, error verbatim.
3. **Manual / anything else** — the existing copyable `wenlan` MCP JSON snippet
   (`getWenlanMcpEntry`), kept as the escape hatch.

The wizard's verify step keeps its `listAgents` 3s poll + auto-advance, now shared
as the per-card verification state.

### 3. Obsidian vault connect + check

- Extract detection from `AddSourceDialog` into
  `src/components/memory/sources/vaultDetection.ts` + a shared `VaultConnectCard`
  used by the dialog and the wizard.
- **Recursive detection** (fixes the shallow-scan bug): walk subdirectories with
  caps — depth ≤ 6, ≤ 5,000 entries, skip dot-directories; display count capped as
  "5,000+". The validity rule follows the **source type**, which is what the daemon
  actually enforces (council change e):
  - `.obsidian/` at root ⇒ source_type **obsidian** — validity short-circuits on the
    first `.md` anywhere (daemon `has_any_markdown` checks markdown only; `.txt`/`.pdf`
    don't count for obsidian-typed sources).
  - otherwise ⇒ source_type **directory** — validity short-circuits on the first
    `.md`/`.txt`/`.pdf` (daemon directory ingest filter, `directory.rs:45`).
- **Submit is never blocked by a zero count** (council change e; removes the current
  `canSubmit` hard gate): a readable walk that finds nothing shows a warning
  ("No notes found — Wenlan will verify on connect") but allows submit; the daemon's
  `POST /api/sources` validation is the authority and its 4xx is surfaced verbatim.
- **Post-connect check**: after `addSource` + the existing one-shot sync kick for
  obsidian sources, poll `listRegisteredSources` and surface progress →
  "Indexed N files · M memories" (wizard shows it inline before Continue; dialog
  closes into the sources list which already polls every 10s).
- Daemon-side validation errors (`POST /api/sources` 4xx) surface verbatim.

### 4. Settings decomposition — its own behavior-preserving PR, landed first

Council change (a): the decomposition ships as a **separate, strictly
behavior-preserving PR** that lands before any feature work stacks on it — a pure
mechanical refactor is reviewable in minutes and de-risks every later diff.

- `SettingsPage.tsx` becomes a thin section switch (< ~150 lines). Sections move to
  `src/components/memory/settings/sections/{General,Intelligence,Capture,Sources,Agents,Diagnostics}Section.tsx`.
- Shared primitives (`Toggle`, `SettingRow`, `SectionHeader`) move to
  `src/components/memory/settings/primitives.tsx`.
- Public behavior unchanged; existing tests (`SettingsPage.language.test.tsx`,
  `SettingsSidebar.test.tsx`, `SetupWizard.test.tsx`) must stay green without
  loosening assertions. No copy, layout, or i18n changes in this PR.
- The feature branch is cut from (or rebased onto) the decomposition branch; all
  §1–§3 UI work happens in the new section files.

### 5. Visual design constraints (applied via frontend-design + taste skills)

- Token system only: `--mem-*` variables, Fraunces headings, Instrument Sans body,
  JetBrains Mono for values/keys. Light + dark both required.
- One card/row idiom across settings sections and wizard steps (today each section
  improvises).
- Provider cards get a clear configured/active state; masked key display (existing
  pattern in `ApiKeyCard`).
- No `★` decorations, no gradient noise; density matches the existing memory UI.

### 6. New app-crate surface (wenlan-app PR)

| Addition | Shape |
| --- | --- |
| `list_external_models` Tauri command | `(endpoint: String, api_key: Option<String>) → Result<Vec<String>, String>`; reqwest GET `{endpoint}/models` (+ Bearer when key given), 5s timeout, http/https scheme check; thin — JSON parsing in a unit-testable fn |
| `useDaemonVersion()` (frontend) | reads existing health wrapper; exposes `supportsExternalKey` / `supportsHotSwap` (semver ≥ 0.13) |
| `set_external_llm` / `test_external_llm` extension | existing command + `WenlanClient` chain gains optional `api_key`, forwarded to `PUT /api/config` / `POST /api/llm/test` only when set (harmless extra field against 0.12 is NOT assumed — the app omits it below 0.13) |
| MCP client registry extension | `mcp_config.rs`: add Claude Desktop + Gemini CLI (JSON `mcpServers` writers, existing pattern) and Codex CLI (TOML `mcp_servers` upsert via **`toml_edit`** — format-preserving: comments, key order, and unrelated tables survive byte-for-byte; council change d) |

No other new Rust commands: sources, import, config, test all exist. Vendor presets
are frontend data (one const table), not Rust.

### 7. Companion daemon PR (`7xuanlu/wenlan`)

Small, mechanical, independently shippable:

1. `Config.external_llm_api_key: Option<String>` (`wenlan-core/src/config.rs`) —
   serde default; extend the 0600-chmod condition to any stored key.
2. **Key lifecycle contract** (council change b — resolves the "never returned"
   vs "masking" ambiguity): the key **value** is never returned by any endpoint.
   `GET /api/config` returns a presence flag instead
   (`external_llm_api_key_configured: bool`). `PUT /api/config` semantics: field
   **omitted** ⇒ stored key preserved; field present as `""` or `null` ⇒ key
   **cleared**; non-empty ⇒ replaced. The UI renders the flag as a masked
   placeholder and never round-trips the value.
3. `OpenAICompatibleProvider` — optional key ⇒ `Authorization: Bearer {key}` header.
4. Hot-swap: `PUT /api/config` rebuilds/clears `state.external_llm` when external
   fields change (mirror `apply_anthropic_provider`); removing the endpoint clears
   the slot.
5. `TestLlmRequest` gains optional `api_key` so a key can be tested before save.
6. `SetupStatusResponse` gains `external_llm: {configured: bool, loaded: bool}` so
   the active strip can report serving state honestly (council change c);
   additive-only, 0.12 clients ignore it.
7. Ships in the next minor (0.13.0). No other daemon changes.

### 8. Version gating / rollout

- App PRs are complete against daemon **0.12**: custom endpoints work keyless
  (localhost/LAN), key field hidden, "Restart Wenlan to apply" notice shown.
- When a future backend pin bump lands ≥ 0.13, the key field and "Applied"
  hot-swap notice light up via `useDaemonVersion()` — no app re-release logic needed
  beyond the flag.
- **Sequencing risk, stated openly** (council dissent note): against pinned 0.12,
  7 of the 10 presets (all keyed cloud vendors) ship disabled-with-explanation —
  the feature launches "mostly dark" until the pin bump. This is accepted: the
  alternative (blocking the UI on the pin bump) couples two release trains, and
  the disabled cards double as discoverable roadmap. The pin bump remains its own
  existing workflow, out of scope here.
- **Three draft PRs**, in landing order: (1) app — settings decomposition,
  behavior-preserving (§4); (2) app — redesign features, stacked on 1;
  (3) daemon — 0.13 external-key/hot-swap/status (§7), independently shippable.

## Error handling

- Model discovery failure → silent fallback to free-text model entry + hint line.
- `POST /api/llm/test` failure → verbatim daemon error string in the card.
- Vault path unreadable client-side → allow submit; daemon validation is the
  authority and its error is surfaced.
- Health/version fetch failure → conservative flags (treat as < 0.13).

## Testing

- **Vitest:** `vaultDetection` (recursion, caps, short-circuit, dot-dir skip — mocked
  `plugin-fs`); Any-provider card (preset fill-in, discovery success/failure, version
  gating incl. keyed-preset lockout below 0.13, precedence warning, test button
  states — mocked invoke); wizard 3-way intelligence step; import step showing both
  paths; connect platform cards (detected/configured/connected states); settings
  sections render per group.
- **Rust (app crate, mcp_config):** unit tests per new client writer — especially the
  Codex TOML upsert: existing unrelated tables preserved, `mcp_servers.wenlan` upsert
  idempotent, and **comments + key order + formatting survive byte-for-byte** on a
  fixture config containing comments and custom spacing (council change d).
- **i18n:** all new copy via `resources.ts` in en / zh-Hans / zh-Hant;
  `hardcodedCopyGuard.test.ts` and `pnpm test:i18n` green.
- **Rust (app crate):** unit test for the `/models` response parsing fn; `cargo fmt`,
  `clippy -D warnings`, `cargo test` in `app/`.
- **Rust (daemon):** bearer-header construction; hot-swap handler add/replace/clear;
  key lifecycle contract — value never serialized in any response, presence flag
  correct, PUT omit-preserves / empty-or-null-clears / non-empty-replaces (§7.2);
  `SetupStatusResponse.external_llm` states; workspace fmt/clippy/test.
- **Gates before PR:** `pnpm build` (tsc -b + vite), `pnpm test`, `pnpm test:i18n`,
  both cargo suites.

## Implementation process

Plan via `superpowers:writing-plans`, execute via subagent-driven development with
model routing: mechanical, well-specified tasks → Sonnet subagents; per-task spec/code
review gates and the final integrated review → strongest available model. Task order
follows the PR order in §8: decomposition first (behavior-preserving, its own PR),
then features, with the daemon PR parallelizable. Worktree:
`worktree-settings-onboarding-redesign` (app repo); daemon work gets its own branch in
`7xuanlu/wenlan`.

## Council review (2026-07-10)

`/boule:debate` — 3 labs, anonymized cross-attack, stake-free swap-averaged judge.
Verdict: **unanimous approve-with-changes** (claude main-loop medium confidence,
gpt-5.6-sol xhigh high, Gemini 3.1 Pro high); judge approve-with-changes, medium
confidence, position stable. 6 attacks, 15 conceded points, 14 contested, 0 verdict
revisions. Required changes, all folded into this revision:

- (a) settings decomposition = its own behavior-preserving PR, landed first (§4, §8)
- (b) explicit key-lifecycle contract: presence flag + omit-preserves /
  explicit-clear PUT semantics (§7.2, Testing)
- (c) active strip distinguishes configured vs serving vs restart-pending; daemon
  0.13 exposes `external_llm: {configured, loaded}` in setup status (§1, §7.6)
- (d) Codex CLI TOML writer must be format-preserving (`toml_edit`) with
  comment/format-survival tests (§2a, §6, Testing)
- (e) vault check follows daemon per-source-type rules (obsidian ⇒ md-only;
  directory ⇒ md/txt/pdf) and never hard-blocks submit on a zero count (§3)
- (f) no-auth boundary warning on tunnel-URL web connect cards (§2a)

Dissent notes folded without design change: `listAgents` delta-attribution is
best-effort (§2a); per-preset `/models` compatibility validated at implementation
(§1); 0.12 "mostly dark" launch sequencing acknowledged as accepted risk (§8).

## Resolved questions

- **Why not Ollama's native `/api/tags`?** `{endpoint}/models` is part of the
  OpenAI-compatible surface Ollama/LM Studio/vLLM all serve — one code path, no
  vendor special-casing.
- **Radio-exclusive providers?** No — daemon keeps all configured slots and picks by
  priority; UI mirrors reality instead of inventing an exclusive mode.
- **Recursive scan cost on huge vaults?** Caps (depth 6 / 5,000 entries) bound the
  walk; validity short-circuits on the first supported file.

## §9 Feedback round 2 — user review of the live tour (2026-07-11)

User-approved scope (AskUserQuestion 2026-07-11): lands as **additional commits on
PR #82** (`settings-onboarding-features`); local probe = **both servers on pane
load**; plugin scope = **UI now + `.mcpb`/`.codex-plugin` as a follow-up daemon-repo
PR** (Gemini CLI extension repo = backlog). Amends §1, §2, §2a. All claims below
re-validated 2026-07-11 against primary sources (two research passes); items marked
`[unverified]` must not be load-bearing.

### 9.1 Per-provider key UX (amends §1 Any-provider card + wizard cloud pane)

Extend `ProviderPreset` (`src/components/intelligence/providerPresets.ts`) with:

```ts
keyPlaceholder?: string;   // provider-shaped example shown when no key stored
keyPrefixes?: string[];    // soft-check prefixes; absent = no format check
getKeyUrl?: string;        // provider console, opened in system browser
```

| preset id | new `name` | keyPlaceholder | keyPrefixes | getKeyUrl |
| --- | --- | --- | --- | --- |
| (native Anthropic card) | Anthropic | `sk-ant-api03-…` | `sk-ant-` | https://console.anthropic.com/settings/keys |
| openai | OpenAI | `sk-proj-…` | `sk-` | https://platform.openai.com/api-keys |
| gemini | **Gemini** (renamed from "Google Gemini") | `AIzaSy… or AQ.…` | `AIzaSy`, `AQ.` | https://aistudio.google.com/apikey |
| groq | Groq | `gsk_…` | `gsk_` | https://console.groq.com/keys |
| openrouter | OpenRouter | `sk-or-v1-…` | `sk-or-` | https://openrouter.ai/keys |
| mistral | Mistral | (none — opaque token) | (none) | https://console.mistral.ai |
| deepseek | DeepSeek | `sk-…` | `sk-` | https://platform.deepseek.com/api_keys |
| xai | **xAI (Grok)** (renamed from "xAI") | `xai-…` | `xai-` | https://console.x.ai |

Behavior:

- Key input `placeholder` = preset's `keyPlaceholder` (falls back to the existing
  configured-mask behavior when a key is already stored).
- **Soft validation only**: non-empty key not matching any `keyPrefixes` shows an
  amber, non-blocking hint ("This doesn't look like a {vendor} key — expected to
  start with {prefix}"). Save/Test are never blocked by format. Rationale: Gemini
  is mid-migration between two live formats (`AIzaSy` legacy, `AQ.` 2026 AI Studio
  default) and Mistral has no documented prefix.
- "Get a key →" link on every keyed preset, opens `getKeyUrl` in the system
  browser via the app's existing external-link mechanism (implementer verifies
  which opener the codebase already uses; do not add a new plugin for this).
- **Naming decision (validated)**: Groq (groq.com, LPU inference company) and Grok
  (xAI's model family, api.x.ai) are unrelated near-homophones — both presets stay.
  The `xai` rename to "xAI (Grok)" exists precisely to kill this confusion. Preset
  `id`s never change (endpoint matching is id-independent, but ids are stored
  nowhere — keep them stable anyway).

### 9.2 Local-server connected signal + model dropdown (amends §1/§2 local pane)

- **Probe both on load**: when the wizard "Local server" pane (or the settings card
  with a local preset selected) mounts, fire the existing `list_external_models`
  discovery against BOTH local presets (Ollama `http://localhost:11434/v1`,
  LM Studio `http://localhost:1234/v1`; existing 5s timeout). No new Rust.
- **Status on the pills**: each local-provider pill shows ● connected / ○ not
  detected. If exactly one server responds, auto-select it. Card body shows the
  full chip: "● Connected to Ollama — N models" / "○ Not detected at
  localhost:11434 — is Ollama running?".
- **Dropdown, not datalist**: when discovery returns ≥ 1 model, the model field is
  a real `<select>` over the discovered ids. Free-text input remains only when
  discovery fails or preset = Custom (existing fallback path, §1). No hand-typed
  model names in the happy path.
- Cloud presets keep the existing datalist/free-text behavior (discovery against
  cloud vendors requires a key and is already wired; unchanged here).

### 9.3 Plugin-first connect matrix v2 (amends §2a; validated per-surface)

Artifact facts (validated): Claude Code plugin (`.claude-plugin/plugin.json`),
Claude Desktop `.mcpb` (zip: `manifest.json` + one MCP server, NO skills), and
Codex plugin (`.codex-plugin/plugin.json`: skills + `.mcp.json` + hooks) are
**three distinct, non-interoperable formats**. One GitHub repo can host
`.claude-plugin/` and `.codex-plugin/` side by side. On claude.ai, a plugin
installed from the Directory activates its **skills** in web chat, and **Cowork
gets full plugin support including MCP connectors** (claude.com/docs/plugins/overview
availability table; Cowork is part of claude.ai for paid users since 2026-07-07).
The daemon connection in plain web chat still requires the custom connector,
because a public plugin cannot bundle a user-specific tunnel URL.

Per-surface cards, primary path first (user rule: recommend plugin everywhere
except ChatGPT.com):

| surface | primary (this round) | fallback / advanced |
| --- | --- | --- |
| Claude Code | plugin commands (exist today): `claude plugin marketplace add 7xuanlu/wenlan` then `claude plugin install wenlan@7xuanlu-wenlan` | "Copy setup prompt"; one-click config write moves under Advanced |
| Codex (CLI + ChatGPT-desktop Codex mode) | `codex mcp add wenlan -- <same command+args as getWenlanMcpEntry>` (works today, no marketplace needed); upgrade copy to `codex plugin marketplace add 7xuanlu/wenlan` once `.codex-plugin/` ships in the daemon repo | "Copy setup prompt"; Codex TOML one-click write under Advanced |
| Claude Desktop | one-click config write (existing registry entry) until the `.mcpb` bundle ships in the follow-up daemon-repo PR; card copy must not reference `.mcpb` before it exists | manual JSON |
| Gemini CLI | one-click config write (existing); `gemini extensions install` copy lands when the backlog extension repo exists | manual JSON |
| claude.ai (web) | **plugin-first, actionable today** (user-corrected twice 2026-07-11; screenshot-verified): the claude.ai Directory → Plugins tab has an **Add marketplace** dialog accepting "a GitHub `owner/repo` or a git repository URL" (Anthropic/Partners/**Personal** tabs) — individuals can sync `7xuanlu/wenlan` directly. Card step 1: install the Wenlan plugin (Directory → Plugins → + Add marketplace → `7xuanlu/wenlan` → Sync → install; plugin **skills** activate in chat, **Cowork gets the full plugin** — claude.com/docs/plugins/overview). Card step 2 (memory access in chat): custom connector — copy tunnel URL + deep link `https://claude.ai/settings/connectors?modal=add-custom-connector` (opens modal; does NOT prefill URL) — because a public plugin cannot carry a user-specific tunnel URL. Backlog (nice-to-have, no longer the unlock): submit to the Anthropic directory (claude.com/docs/plugins/submit) to remove the add-marketplace step | existing no-auth tunnel warning stays (council change f) |
| ChatGPT.com | connector only (Developer Mode paste-URL) — per user's own call, "the only MCP" surface | existing warning stays |
| Cursor | unchanged (one-click config write) | manual JSON |

**"Copy setup prompt"** (user idea, validated as supported plumbing): each CLI
client card gets a button copying a prompt the user pastes into that agent, e.g.
Claude Code: "Install the Wenlan plugin: run `claude plugin marketplace add
7xuanlu/wenlan`, then `claude plugin install wenlan@7xuanlu-wenlan`, then tell me to run
`/reload-plugins` or restart." Codex: same pattern over `codex mcp add wenlan --
<command+args>`. Prompt text must mention: the agent uses the non-interactive
shell verbs (the `/plugin` TUI is not agent-drivable), a reload/restart is needed
after install, and normal permission prompts will appear. Exact prompt strings are
i18n resources like all other copy.

Sequencing honesty: this round ships UI + copy that is true at ship time. The
`.mcpb` bundle and `.codex-plugin/` directory are a **follow-up PR in the
`7xuanlu/wenlan` repo** (user-approved "UI now + .mcpb next"); the two card-copy
upgrades above land with that PR, not this one.

### 9.4 Testing additions (extends Testing)

- Vitest: preset table carries the exact placeholders/prefixes/URLs above
  (data-shape test); soft-hint logic (match → no hint, mismatch → hint, empty →
  no hint, Mistral → never hints, Gemini both prefixes accepted); local pane
  dual-probe rendering (both up / one up + auto-select / none up), dropdown vs
  free-text switch; connect cards render primary-path copy per client;
  setup-prompt copy button writes the full prompt to clipboard (existing
  clipboardWrite mock pattern).
- i18n: every new string (hints, chips, prompts, card copy) in en / zh-Hans /
  zh-Hant; `pnpm test:i18n` green.
- Existing gates unchanged: `pnpm build`, `pnpm test`, `pnpm test:i18n`; no Rust
  changes expected this round (probe reuses `list_external_models`).
