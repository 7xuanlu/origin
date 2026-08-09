# Onboarding Feedback Round 2 (§9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship spec §9 (feedback round 2) on the onboarding/settings surfaces: per-provider API-key UX (placeholders, soft-validation hints, "Get a key" links), a local-server connected signal with model dropdown, and a plugin-first connect matrix v2 — UI + copy only, no Rust.

**Architecture:** Extend the existing `ProviderPreset` data table and reuse the already-wired external-link opener (`@tauri-apps/plugin-shell` `open`), clipboard (`clipboardWrite`), and model-discovery command (`list_external_models`). The Any-provider card gains local-aware behavior on both §9.2 surfaces: the wizard local pane (`groups === ["local"]`) gets dual-probe status pills + auto-select (T4/T5), and the settings all-groups card gets single-probe chip + model `<select>` when a local preset is selected (T5b). The connect matrix's CLI primary path (plugin commands + "Copy setup prompt") is extracted into a shared `CliPrimaryPath` component consumed by both the settings `ClientSetupList` (T6) and the wizard ConnectStep rows (T6b); GUI clients keep one-click.

**Tech Stack:** React 19, Vite 6, Tailwind v4 (CSS-first `--mem-*` tokens only), @tanstack/react-query 5, react-i18next, Vitest 4 + jsdom, @testing-library/react + userEvent.

## Global Constraints

_Every task's requirements implicitly include this section. Values copied verbatim from spec §9 and the round-2 dispatch._

- **Branch / PR:** work lands as additional commits on branch `settings-onboarding-features` (PR #82). Do NOT switch branches; do NOT rebase.
- **Commit trailer:** every commit message ends with these two lines exactly:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53
  ```
- **No Rust / no new deps:** no changes under `app/`; no new npm packages. The local probe reuses `list_external_models`. The external-link opener is the existing `import { open as shellOpen } from "@tauri-apps/plugin-shell"` (globally mocked in `src/test/setup.ts`). The clipboard is the existing `clipboardWrite` from `src/lib/tauri.ts`.
- **All user-facing copy via i18n:** every new string (hints, chips, prompts, card copy, link labels) added to `en`, `zh-Hans`, and `zh-Hant` in `src/i18n/resources.ts`. `src/i18n/resources.test.ts` enforces **exact key-set parity across all three locales** — a key missing from any locale fails the suite. `pnpm test:i18n` must stay green.
- **Design tokens:** styling uses `--mem-*` CSS variables only (no `tailwind.config`, no raw hex except the existing `text-red-500`/rgb error colors already in these files).
- **Preset renames (ids never change):** `gemini` display name → `Gemini` (was "Google Gemini"); `xai` display name → `xAI (Grok)` (was "xAI"). Preset `id`s stay `gemini` / `xai`. Endpoints unchanged.
- **Soft validation only:** a non-empty key that matches no `keyPrefixes` shows a non-blocking amber hint; Save/Test are NEVER disabled by key format. Empty key → no hint. Mistral (no prefixes) → never hints. Gemini accepts either `AIzaSy` or `AQ.`.
- **Local probe:** probe BOTH local presets (Ollama `http://localhost:11434/v1`, LM Studio `http://localhost:1234/v1`) on the wizard Local-server pane mount via `list_external_models` (existing 5s timeout); auto-select the single responder. The settings card with a local preset selected probes that single preset's endpoint and shows the same connected/not-detected chip (spec §9.2 "the wizard 'Local server' pane (or the settings card with a local preset selected)"). On both surfaces the model field becomes a real `<select>` when discovery returns ≥1 model; free-text only on discovery failure or Custom. Pills are wizard-only; the settings Provider `<select>` stays.
- **Connect matrix (§9.3):** primary path first, on BOTH surfaces (settings Apps & CLIs list AND the wizard connect step — shared components, same i18n keys). Shipped copy must NOT reference `.mcpb` or `.codex-plugin` (those ship in a later daemon-repo PR). Shipped copy MAY — and the Claude card DOES — reference the claude.ai plugin install: it is actionable today via Directory → Plugins → + Add marketplace → `7xuanlu/wenlan` (screenshot-verified 2026-07-11; spec §9.3 claude.ai row, re-read it from disk). "Copy setup prompt" prompts must mention: the agent uses non-interactive shell commands (the `/plugin` TUI is not agent-drivable), a reload/restart is needed after install, and permission prompts will appear.
- **Gates:** `pnpm exec tsc -b`, `pnpm test` (`vitest run`), and `pnpm test:i18n` all green. Run a single test file with `pnpm exec vitest run <path>`.

---

## File Structure

| File | Responsibility | Tasks |
| --- | --- | --- |
| `src/components/intelligence/providerPresets.ts` | Preset data table + soft-check helper | T1 |
| `src/components/intelligence/providerPresets.test.ts` (new) | Data-shape + soft-hint logic tests | T1 |
| `src/components/intelligence/AnyProviderCard.tsx` | Any-provider card: hint/link (§9.1) + local behavior in wizard AND settings (§9.2) | T2, T4, T5, T5b |
| `src/components/intelligence/AnyProviderCard.test.tsx` | Card behavior tests | T2, T4, T5, T5b |
| `src/components/intelligence/IntelligenceSetup.tsx` | Native Anthropic `ApiKeyCard`: placeholder + get-key link (§9.1) | T3 |
| `src/components/intelligence/IntelligenceSetup.test.tsx` | ApiKeyCard test | T3 |
| `src/components/connect/CliPrimaryPath.tsx` (new) | Shared CLI primary path: commands + Copy setup prompt (§9.3, both surfaces) | T6, T6b |
| `src/components/connect/ClientSetupList.tsx` | Apps & CLIs cards, plugin-first (§9.3, settings) | T6 |
| `src/components/connect/ClientSetupList.test.tsx` (new) | Connect card copy + setup-prompt tests | T6 |
| `src/components/SetupWizard.tsx` | Wizard ConnectStep rows consume the shared CLI primary path (§9.3, wizard) | T6b |
| `src/components/SetupWizard.test.tsx` | Wizard connect-step plugin-first tests | T6b |
| `src/components/connect/WebPlatformCards.tsx` | claude.ai plugin-first card + connector deep-link (§9.3) | T7 |
| `src/components/connect/WebPlatformCards.test.tsx` | Plugin-first card + deep-link tests | T7 |
| `src/i18n/resources.ts` | All new copy in en / zh-Hans / zh-Hant | T2–T7 (T5b/T6b reuse earlier keys; no new keys) |

**Reference — exact current i18n anchors** (blocks repeat once per locale; add the same keys to all three):
- `externalProvider`: en L110–132 (ends at `anthropicPrecedence`), zh-Hans L1074–1095, zh-Hant L2019–2040. Insert new keys before the closing `},` of each block.
- `connectMatrix`: en L947–969 (ends at `manualTitle`), zh-Hans L1893–1914, zh-Hant L2838–2859. Insert new keys before the closing `},` of each block.

---

## Task 1: Extend ProviderPreset data + soft-check helper

**Files:**
- Modify: `src/components/intelligence/providerPresets.ts`
- Create: `src/components/intelligence/providerPresets.test.ts`

**Interfaces:**
- Produces: `ProviderPreset` now has optional `keyPlaceholder?: string`, `keyPrefixes?: string[]`, `getKeyUrl?: string`. New exported pure function `keyPrefixMismatch(preset: ProviderPreset, key: string): boolean` — `true` only when `key.trim()` is non-empty, the preset has ≥1 prefix, and none match. Later tasks (T2, T3) consume both.

- [ ] **Step 1: Write the failing test**

Create `src/components/intelligence/providerPresets.test.ts`:

```ts
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect } from "vitest";
import {
  PROVIDER_PRESETS,
  keyPrefixMismatch,
  type ProviderPreset,
} from "./providerPresets";

const byId = (id: string): ProviderPreset => {
  const p = PROVIDER_PRESETS.find((x) => x.id === id);
  if (!p) throw new Error(`preset ${id} missing`);
  return p;
};

describe("providerPresets — §9.1 data shape", () => {
  it("carries the exact per-provider key metadata", () => {
    expect(byId("openai")).toMatchObject({
      name: "OpenAI",
      keyPlaceholder: "sk-proj-...",
      keyPrefixes: ["sk-"],
      getKeyUrl: "https://platform.openai.com/api-keys",
    });
    expect(byId("gemini")).toMatchObject({
      name: "Gemini",
      keyPlaceholder: "AIzaSy... or AQ....",
      keyPrefixes: ["AIzaSy", "AQ."],
      getKeyUrl: "https://aistudio.google.com/apikey",
    });
    expect(byId("groq")).toMatchObject({
      name: "Groq",
      keyPrefixes: ["gsk_"],
      getKeyUrl: "https://console.groq.com/keys",
    });
    expect(byId("openrouter")).toMatchObject({
      name: "OpenRouter",
      keyPrefixes: ["sk-or-"],
      getKeyUrl: "https://openrouter.ai/keys",
    });
    expect(byId("deepseek")).toMatchObject({
      name: "DeepSeek",
      keyPrefixes: ["sk-"],
      getKeyUrl: "https://platform.deepseek.com/api_keys",
    });
    expect(byId("xai")).toMatchObject({
      name: "xAI (Grok)",
      keyPlaceholder: "xai-...",
      keyPrefixes: ["xai-"],
      getKeyUrl: "https://console.x.ai",
    });
  });

  it("Mistral is keyed but opaque — a getKeyUrl, no prefixes", () => {
    const mistral = byId("mistral");
    expect(mistral.name).toBe("Mistral");
    expect(mistral.getKeyUrl).toBe("https://console.mistral.ai");
    expect(mistral.keyPrefixes).toBeUndefined();
  });

  it("keeps ids stable and endpoints unchanged after the renames", () => {
    expect(byId("gemini").endpoint).toBe(
      "https://generativelanguage.googleapis.com/v1beta/openai",
    );
    expect(byId("xai").endpoint).toBe("https://api.x.ai/v1");
  });
});

describe("keyPrefixMismatch — §9.1 soft validation", () => {
  it("no hint for an empty key", () => {
    expect(keyPrefixMismatch(byId("openai"), "")).toBe(false);
    expect(keyPrefixMismatch(byId("openai"), "   ")).toBe(false);
  });
  it("no hint when a prefix matches", () => {
    expect(keyPrefixMismatch(byId("openai"), "sk-proj-abc")).toBe(false);
    expect(keyPrefixMismatch(byId("xai"), "xai-abc")).toBe(false);
  });
  it("hint when no prefix matches", () => {
    expect(keyPrefixMismatch(byId("openai"), "nope-123")).toBe(true);
    expect(keyPrefixMismatch(byId("groq"), "sk-123")).toBe(true);
  });
  it("Gemini accepts either live format — neither mismatches", () => {
    expect(keyPrefixMismatch(byId("gemini"), "AIzaSyABC")).toBe(false);
    expect(keyPrefixMismatch(byId("gemini"), "AQ.abc")).toBe(false);
    expect(keyPrefixMismatch(byId("gemini"), "xyz")).toBe(true);
  });
  it("Mistral never hints (no documented prefix)", () => {
    expect(keyPrefixMismatch(byId("mistral"), "anything")).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec vitest run src/components/intelligence/providerPresets.test.ts`
Expected: FAIL — `keyPrefixMismatch` is not exported; `keyPlaceholder`/`keyPrefixes`/`getKeyUrl` undefined on presets.

- [ ] **Step 3: Write minimal implementation**

In `src/components/intelligence/providerPresets.ts`, extend the `ProviderPreset` interface (add the three optional fields) and the preset rows. Replace the interface and the `PROVIDER_PRESETS` array with:

```ts
export interface ProviderPreset {
  id: string;
  name: string;
  endpoint: string;
  keyRequired: boolean;
  group: PresetGroup;
  /** Provider-shaped example shown when no key is stored (§9.1). */
  keyPlaceholder?: string;
  /** Soft-check prefixes; absent = no format check (§9.1). */
  keyPrefixes?: string[];
  /** Provider console, opened in the system browser (§9.1). */
  getKeyUrl?: string;
}

export const PROVIDER_PRESETS: ProviderPreset[] = [
  {
    id: "openai",
    name: "OpenAI",
    endpoint: "https://api.openai.com/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "sk-proj-...",
    keyPrefixes: ["sk-"],
    getKeyUrl: "https://platform.openai.com/api-keys",
  },
  {
    id: "gemini",
    name: "Gemini",
    endpoint: "https://generativelanguage.googleapis.com/v1beta/openai",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "AIzaSy... or AQ....",
    keyPrefixes: ["AIzaSy", "AQ."],
    getKeyUrl: "https://aistudio.google.com/apikey",
  },
  {
    id: "groq",
    name: "Groq",
    endpoint: "https://api.groq.com/openai/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "gsk_...",
    keyPrefixes: ["gsk_"],
    getKeyUrl: "https://console.groq.com/keys",
  },
  {
    id: "openrouter",
    name: "OpenRouter",
    endpoint: "https://openrouter.ai/api/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "sk-or-v1-...",
    keyPrefixes: ["sk-or-"],
    getKeyUrl: "https://openrouter.ai/keys",
  },
  {
    id: "mistral",
    name: "Mistral",
    endpoint: "https://api.mistral.ai/v1",
    keyRequired: true,
    group: "cloud",
    getKeyUrl: "https://console.mistral.ai",
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    endpoint: "https://api.deepseek.com/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "sk-...",
    keyPrefixes: ["sk-"],
    getKeyUrl: "https://platform.deepseek.com/api_keys",
  },
  {
    id: "xai",
    name: "xAI (Grok)",
    endpoint: "https://api.x.ai/v1",
    keyRequired: true,
    group: "cloud",
    keyPlaceholder: "xai-...",
    keyPrefixes: ["xai-"],
    getKeyUrl: "https://console.x.ai",
  },
  { id: "ollama", name: "Ollama (local)", endpoint: "http://localhost:11434/v1", keyRequired: false, group: "local" },
  { id: "lmstudio", name: "LM Studio (local)", endpoint: "http://localhost:1234/v1", keyRequired: false, group: "local" },
  { id: "custom", name: "Custom…", endpoint: "", keyRequired: false, group: "custom" },
];
```

Then append the helper at the end of the file (after `presetForEndpoint`):

```ts
/** Soft key-format check (§9.1): true only when a non-empty key matches none
 *  of the preset's prefixes. Presets without prefixes (e.g. Mistral) never
 *  mismatch. Never used to block Save/Test — hint only. */
export function keyPrefixMismatch(preset: ProviderPreset, key: string): boolean {
  const k = key.trim();
  if (k === "" || !preset.keyPrefixes || preset.keyPrefixes.length === 0) return false;
  return !preset.keyPrefixes.some((prefix) => k.startsWith(prefix));
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/intelligence/providerPresets.test.ts`
Expected: PASS (all 8 cases).

- [ ] **Step 5: Run gates**

Run: `pnpm exec tsc -b`
Expected: no errors.
Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx src/components/SetupWizard.test.tsx`
Expected: PASS — the renames do not break existing suites (no test asserts the old "Google Gemini"/"xAI" strings; presets are selected by `id`).

- [ ] **Step 6: Commit**

```bash
git add src/components/intelligence/providerPresets.ts src/components/intelligence/providerPresets.test.ts
git commit -m "feat(intelligence): extend ProviderPreset with key metadata + soft-check helper (§9.1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Task 2: Any-provider key hint + "Get a key" link (§9.1)

**Files:**
- Modify: `src/components/intelligence/AnyProviderCard.tsx`
- Modify: `src/components/intelligence/AnyProviderCard.test.tsx`
- Modify: `src/i18n/resources.ts` (3 locales)

**Interfaces:**
- Consumes: `keyPrefixMismatch`, `ProviderPreset.keyPlaceholder`, `.getKeyUrl` (Task 1); `import { open as shellOpen } from "@tauri-apps/plugin-shell"`.
- Produces: two i18n keys `externalProvider.getKeyLink`, `externalProvider.keyHint` (reused by T3).

- [ ] **Step 1: Write the failing test**

Add these cases inside the existing `describe("AnyProviderCard", …)` block in `src/components/intelligence/AnyProviderCard.test.tsx`. Also add this import near the top (below the existing imports):

```ts
import { open as shellOpen } from "@tauri-apps/plugin-shell";
```

```ts
  it("shows the provider-shaped key placeholder on 0.13", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "openai");
    expect(await screen.findByLabelText("API key")).toHaveAttribute("placeholder", "sk-proj-...");
  });

  it("shows an amber soft hint for a key that matches no prefix, without blocking Save", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "openai");
    const keyField = await screen.findByLabelText("API key");
    await userEvent.type(keyField, "nope-123");
    await userEvent.type(screen.getByLabelText("Model"), "gpt-4o-mini");
    expect(screen.getByText(/doesn't look like an? OpenAI key/i)).toBeInTheDocument();
    // Soft only — Save stays enabled.
    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
  });

  it("shows no hint once the key matches a prefix", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "openai");
    await userEvent.type(await screen.findByLabelText("API key"), "sk-proj-abc");
    expect(screen.queryByText(/doesn't look like/i)).not.toBeInTheDocument();
  });

  it("opens the provider console via the system browser from Get a key", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    vi.mocked(shellOpen).mockClear();
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "openai");
    await userEvent.click(await screen.findByRole("button", { name: /Get a key/ }));
    expect(shellOpen).toHaveBeenCalledWith("https://platform.openai.com/api-keys");
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: FAIL — placeholder is empty, no hint text, no "Get a key" button.

- [ ] **Step 3: Write minimal implementation**

In `src/components/intelligence/AnyProviderCard.tsx`:

3a. Add the shell-opener import and the helper import. Change the top imports so `keyPrefixMismatch` is imported and add the plugin-shell line:

```ts
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import {
  PROVIDER_PRESETS,
  presetForEndpoint,
  keyPrefixMismatch,
  type PresetGroup,
} from "./providerPresets";
```

3b. Replace the API-key `{supportsExternalKey && (…)}` block (currently lines ~215–226) with this version — placeholder now prefers the preset shape, plus the amber hint and the get-key link:

```tsx
          {supportsExternalKey && (
            <label className="flex flex-col gap-1">
              <span style={labelStyle}>{t("externalProvider.apiKeyLabel")}</span>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={
                  keyConfigured
                    ? t("externalProvider.apiKeyConfiguredPlaceholder")
                    : (preset.keyPlaceholder ?? "")
                }
                style={fieldStyle}
              />
              {keyPrefixMismatch(preset, apiKey) && (
                <span
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "11px",
                    color: "var(--mem-accent-amber)",
                  }}
                >
                  {t("externalProvider.keyHint", {
                    vendor: preset.name,
                    prefix: (preset.keyPrefixes ?? []).join(" or "),
                  })}
                </span>
              )}
              {preset.getKeyUrl && (
                <button
                  type="button"
                  onClick={() => shellOpen(preset.getKeyUrl!)}
                  className="self-start text-xs"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    color: "var(--mem-accent-indigo)",
                    background: "none",
                    border: "none",
                    padding: 0,
                    cursor: "pointer",
                  }}
                >
                  {t("externalProvider.getKeyLink")}
                </button>
              )}
            </label>
          )}
```

- [ ] **Step 4: Add i18n keys (all three locales)**

In `src/i18n/resources.ts`, add these two keys to the `externalProvider` block in EACH locale (before the closing `},` — i.e. after `anthropicPrecedence`). Add a trailing comma to `anthropicPrecedence`'s value if needed.

en (block at L110):
```ts
    getKeyLink: "Get a key →",
    keyHint:
      "This doesn't look like a {{vendor}} key — expected to start with {{prefix}}.",
```
zh-Hans (block at L1074):
```ts
    getKeyLink: "获取密钥 →",
    keyHint: "这看起来不像 {{vendor}} 密钥 — 应以 {{prefix}} 开头。",
```
zh-Hant (block at L2019):
```ts
    getKeyLink: "取得金鑰 →",
    keyHint: "這看起來不像 {{vendor}} 金鑰 — 應以 {{prefix}} 開頭。",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: PASS (new + all existing cases).
Run: `pnpm test:i18n`
Expected: PASS (key parity holds across locales).

- [ ] **Step 6: Run gates + Commit**

Run: `pnpm exec tsc -b`
Expected: no errors.
```bash
git add src/components/intelligence/AnyProviderCard.tsx src/components/intelligence/AnyProviderCard.test.tsx src/i18n/resources.ts
git commit -m "feat(intelligence): per-provider key placeholder, soft hint, Get-a-key link (§9.1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Task 3: Native Anthropic card — placeholder + "Get a key" link (§9.1)

_Ambiguity resolved (flagged to coordinator): the §9.1 table's "(native Anthropic card)" row is NOT a `ProviderPreset` — it is the standalone `ApiKeyCard` in `IntelligenceSetup.tsx`. This task applies the row's `keyPlaceholder` and `getKeyUrl` there. The Anthropic key is a single fixed format, so no soft-hint is added (a hint would only ever fire on typos of one format); if the coordinator wants prefix-hint parity here, it is a one-line add._

**Files:**
- Modify: `src/components/intelligence/IntelligenceSetup.tsx`
- Modify: `src/components/intelligence/IntelligenceSetup.test.tsx`

**Interfaces:**
- Consumes: `externalProvider.getKeyLink` i18n key (Task 2); `import { open as shellOpen } from "@tauri-apps/plugin-shell"`.

- [ ] **Step 1: Write the failing test**

Add to `src/components/intelligence/IntelligenceSetup.test.tsx`. If the file does not already import `shellOpen`, add:

```ts
import { open as shellOpen } from "@tauri-apps/plugin-shell";
```

Add a test that renders `ApiKeyCard` in the un-configured state (the existing suite already mocks `getApiKey`; ensure it resolves `null` for this case so the input branch renders):

```ts
  it("offers a Get-a-key link to the Anthropic console when no key is set", async () => {
    // getApiKey → null renders the password-input branch (not the masked state).
    vi.mocked(shellOpen).mockClear();
    render(
      <QueryClientProvider client={new QueryClient()}>
        <ApiKeyCard />
      </QueryClientProvider>,
    );
    await userEvent.click(await screen.findByRole("button", { name: /Get a key/ }));
    expect(shellOpen).toHaveBeenCalledWith(
      "https://console.anthropic.com/settings/keys",
    );
  });
```

_Note: match the file's existing render/mocks setup (it already imports `ApiKeyCard`, `QueryClient`, `render`, `screen`, `userEvent`). If `getApiKey` is mocked globally to a masked value in this file, override it to `null` for this test via the file's existing mock handle before rendering._

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec vitest run src/components/intelligence/IntelligenceSetup.test.tsx`
Expected: FAIL — no "Get a key" button in `ApiKeyCard`.

- [ ] **Step 3: Write minimal implementation**

In `src/components/intelligence/IntelligenceSetup.tsx`:

3a. Add the shell-opener import at the top:
```ts
import { open as shellOpen } from "@tauri-apps/plugin-shell";
```

3b. Change the password input placeholder (line ~141) from `placeholder="sk-ant-..."` to the fuller shape:
```tsx
              placeholder="sk-ant-api03-..."
```

3c. In the `!isConfigured` input branch, add the get-key link directly after the closing `</div>` of the input+button row (i.e. after line ~161, still inside the `!isConfigured` block, before the `{error && …}` block). Insert:

```tsx
        {!isConfigured && (
          <button
            type="button"
            onClick={() => shellOpen("https://console.anthropic.com/settings/keys")}
            className="mt-2 text-xs"
            style={{
              fontFamily: "var(--mem-font-body)",
              color: "var(--mem-accent-indigo)",
              background: "none",
              border: "none",
              padding: 0,
              cursor: "pointer",
            }}
          >
            {t("externalProvider.getKeyLink")}
          </button>
        )}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/intelligence/IntelligenceSetup.test.tsx`
Expected: PASS.

- [ ] **Step 5: Run gates + Commit**

Run: `pnpm exec tsc -b` → no errors. `pnpm test:i18n` → PASS (no new keys, reuses `externalProvider.getKeyLink`).
```bash
git add src/components/intelligence/IntelligenceSetup.tsx src/components/intelligence/IntelligenceSetup.test.tsx
git commit -m "feat(intelligence): Anthropic card placeholder + Get-a-key link (§9.1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Task 4: Local-server dual probe, status pills, connected chip, auto-select (§9.2)

_Architecture note: the dual-probe + pills UI is gated on `isLocalOnly` (the card was rendered with `groups={["local"]}` — the wizard "Local server" pane). The settings all-groups card is untouched by THIS task so its existing tests stay green here; Task 5b then brings the settings card to §9.2 parity (single probe, chip, model `<select>`) and rewrites the affected tests. Do the tasks in order._

**Files:**
- Modify: `src/components/intelligence/AnyProviderCard.tsx`
- Modify: `src/components/intelligence/AnyProviderCard.test.tsx`
- Modify: `src/i18n/resources.ts` (3 locales)

**Interfaces:**
- Produces: derived values `isLocalOnly`, `probeFor(id)`, `selectedProbe`, `localModels`, and helper `localLabel(name)` — consumed by Task 5's model field. Three i18n keys: `externalProvider.localConnectedChip`, `.localNotDetectedChip`, `.localProbing`.

- [ ] **Step 1: Write the failing test**

Add these cases to `src/components/intelligence/AnyProviderCard.test.tsx`:

```ts
  it("local pane: both servers up → both pills connected, no auto-switch", async () => {
    // default mock resolves for any endpoint → both probes succeed.
    renderCard({ groups: ["local"] });
    expect(await screen.findByText(/Connected to Ollama/)).toBeInTheDocument();
    // both local pills present
    expect(screen.getByRole("button", { name: /Ollama/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /LM Studio/ })).toBeInTheDocument();
  });

  it("local pane: exactly one server up → auto-selects it", async () => {
    mocks.listExternalModels.mockImplementation((ep: string) =>
      ep.includes("1234")
        ? Promise.resolve(["qwen2.5:7b"])
        : Promise.reject(new Error("ECONNREFUSED")),
    );
    renderCard({ groups: ["local"] });
    // LM Studio (1234) is the sole responder → its chip is shown.
    expect(await screen.findByText(/Connected to LM Studio/)).toBeInTheDocument();
    expect(screen.getByLabelText("Endpoint URL")).toHaveValue("http://localhost:1234/v1");
  });

  it("local pane: no server up → not-detected chip for the selected pill", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard({ groups: ["local"] });
    expect(
      await screen.findByText(/Not detected at localhost:11434 — is Ollama running\?/),
    ).toBeInTheDocument();
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: FAIL — no pills, no chip; `groups: ["local"]` still renders the `<select>` picker.

- [ ] **Step 3: Write minimal implementation**

In `src/components/intelligence/AnyProviderCard.tsx`:

3a. Add `useRef` to the React import:
```ts
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
```

3b. Add module-level constants (below the imports, above `fieldStyle`):
```ts
const OLLAMA_ENDPOINT = "http://localhost:11434/v1";
const LMSTUDIO_ENDPOINT = "http://localhost:1234/v1";
const hostOf = (ep: string) => ep.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
const localLabel = (name: string) => name.replace(/\s*\(local\)$/i, "");
```

3c. After the `presets` `useMemo` (line ~57), add:
```ts
  const isLocalOnly = groups?.length === 1 && groups[0] === "local";
```

3d. Change the generic `discovery` query's `enabled` (line ~102) so it does not double-fetch local endpoints:
```ts
    enabled: endpointValid && !lockedByVersion && !isLocalOnly,
```

3e. Immediately after the `discovery` query + `const models = discovery.data ?? [];` (line ~106), add the two local probes and their derived state:
```ts
  // §9.2: probe BOTH local servers on the Local-server pane mount.
  const ollamaProbe = useQuery({
    queryKey: ["local-probe", OLLAMA_ENDPOINT],
    queryFn: () => listExternalModels(OLLAMA_ENDPOINT, null),
    enabled: !!isLocalOnly,
    retry: false,
    staleTime: 30_000,
  });
  const lmStudioProbe = useQuery({
    queryKey: ["local-probe", LMSTUDIO_ENDPOINT],
    queryFn: () => listExternalModels(LMSTUDIO_ENDPOINT, null),
    enabled: !!isLocalOnly,
    retry: false,
    staleTime: 30_000,
  });
  const probeFor = (id: string) =>
    id === "ollama" ? ollamaProbe : id === "lmstudio" ? lmStudioProbe : null;
  const selectedProbe = probeFor(presetId);
  const localModels = selectedProbe?.data ?? [];

  // Auto-select the single responder, once both probes have settled.
  const autoSelectedRef = useRef(false);
  useEffect(() => {
    if (!isLocalOnly || autoSelectedRef.current) return;
    if (ollamaProbe.isLoading || lmStudioProbe.isLoading) return;
    autoSelectedRef.current = true; // decide exactly once
    // Respect a previously-saved local endpoint over auto-selection.
    if (current && current[0]) return;
    const up = [
      ["ollama", ollamaProbe.isSuccess] as const,
      ["lmstudio", lmStudioProbe.isSuccess] as const,
    ].filter(([, ok]) => ok);
    if (up.length === 1) {
      const id = up[0][0];
      setPresetId(id);
      setEndpoint(id === "ollama" ? OLLAMA_ENDPOINT : LMSTUDIO_ENDPOINT);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLocalOnly, ollamaProbe.isLoading, ollamaProbe.isSuccess, lmStudioProbe.isLoading, lmStudioProbe.isSuccess]);
```

3f. Replace the preset-picker block (currently the `{!hidePresetPicker && (…)}` label, lines ~171–180) with a branch that shows pills in local-only mode and the chip:

```tsx
      {isLocalOnly ? (
        <>
          <div className="flex flex-wrap gap-2">
            {presets.map((p) => {
              const probe = probeFor(p.id);
              const status = !probe
                ? null
                : probe.isLoading
                ? "probing"
                : probe.isSuccess
                ? "connected"
                : "notDetected";
              const dot = status === "connected" ? "●" : status === "notDetected" ? "○" : "…";
              const selected = p.id === presetId;
              return (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => selectPreset(p.id)}
                  className="rounded-full px-3 py-1.5 text-xs"
                  style={{
                    border: `1px solid ${selected ? "var(--mem-accent-indigo)" : "var(--mem-border)"}`,
                    backgroundColor: selected ? "var(--mem-accent-indigo)" : "var(--mem-surface)",
                    color: selected ? "white" : "var(--mem-text)",
                    fontFamily: "var(--mem-font-body)",
                  }}
                >
                  {probe && (
                    <span
                      style={{
                        marginRight: "6px",
                        color:
                          status === "connected" && !selected
                            ? "var(--mem-accent-sage)"
                            : "inherit",
                      }}
                    >
                      {dot}
                    </span>
                  )}
                  {localLabel(p.name)}
                </button>
              );
            })}
          </div>
          {selectedProbe && (
            <p
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                lineHeight: 1.5,
                color: selectedProbe.isSuccess
                  ? "var(--mem-accent-sage)"
                  : "var(--mem-text-secondary)",
              }}
            >
              {selectedProbe.isLoading
                ? t("externalProvider.localProbing", { name: localLabel(preset.name) })
                : selectedProbe.isSuccess
                ? t("externalProvider.localConnectedChip", {
                    name: localLabel(preset.name),
                    modelCount: localModels.length,
                  })
                : t("externalProvider.localNotDetectedChip", {
                    name: localLabel(preset.name),
                    host: hostOf(preset.endpoint),
                  })}
            </p>
          )}
        </>
      ) : !hidePresetPicker ? (
        <label className="flex flex-col gap-1">
          <span style={labelStyle}>{t("externalProvider.presetLabel")}</span>
          <select value={presetId} onChange={(e) => selectPreset(e.target.value)} style={fieldStyle}>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </label>
      ) : null}
```

- [ ] **Step 4: Add i18n keys (all three locales)**

Add to the `externalProvider` block in each locale (alongside the Task 2 keys). The interpolation variable is deliberately `modelCount`, NOT `count` — i18next reserves `count` for plural-suffix resolution (`_one`/`_other`), which these single-form keys do not define.

en:
```ts
    localConnectedChip: "Connected to {{name}} — {{modelCount}} models",
    localNotDetectedChip: "Not detected at {{host}} — is {{name}} running?",
    localProbing: "Checking {{name}}…",
```
zh-Hans:
```ts
    localConnectedChip: "已连接到 {{name}} — {{modelCount}} 个模型",
    localNotDetectedChip: "未在 {{host}} 检测到 — {{name}} 是否正在运行？",
    localProbing: "正在检查 {{name}}…",
```
zh-Hant:
```ts
    localConnectedChip: "已連接到 {{name}} — {{modelCount}} 個模型",
    localNotDetectedChip: "未在 {{host}} 偵測到 — {{name}} 是否正在執行？",
    localProbing: "正在檢查 {{name}}…",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: PASS — new local cases + all existing all-groups cases (probes are disabled when `isLocalOnly` is false, so existing call assertions are unaffected).
Run: `pnpm test:i18n`
Expected: PASS.

- [ ] **Step 6: Run gates + Commit**

Run: `pnpm exec tsc -b` → no errors.
```bash
git add src/components/intelligence/AnyProviderCard.tsx src/components/intelligence/AnyProviderCard.test.tsx src/i18n/resources.ts
git commit -m "feat(intelligence): local-server dual probe, status pills, auto-select (§9.2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Task 5: Local model dropdown vs free-text (§9.2)

**Files:**
- Modify: `src/components/intelligence/AnyProviderCard.tsx`
- Modify: `src/components/intelligence/AnyProviderCard.test.tsx`
- Modify: `src/i18n/resources.ts` (3 locales)

**Interfaces:**
- Consumes: `isLocalOnly`, `selectedProbe`, `localModels` (Task 4).
- Produces: i18n key `externalProvider.modelSelectPlaceholder`.

- [ ] **Step 1: Write the failing test**

Add to `src/components/intelligence/AnyProviderCard.test.tsx`:

```ts
  it("local pane: discovered models render as a <select>, not free text", async () => {
    mocks.listExternalModels.mockResolvedValue(["qwen2.5:7b", "llama3.2:3b"]);
    renderCard({ groups: ["local"] });
    await screen.findByText(/Connected to Ollama/);
    const modelField = await screen.findByLabelText("Model");
    expect(modelField.tagName).toBe("SELECT");
    await userEvent.selectOptions(modelField, "llama3.2:3b");
    expect(modelField).toHaveValue("llama3.2:3b");
  });

  it("local pane: discovery failure keeps free-text model entry with hint", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard({ groups: ["local"] });
    await screen.findByText(/Not detected at localhost:11434/);
    const modelField = await screen.findByLabelText("Model");
    expect(modelField.tagName).toBe("INPUT");
    expect(screen.getByText(/type a model name/i)).toBeInTheDocument();
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: FAIL — the Model field is always an `<input>` in local mode; no `<select>`.

- [ ] **Step 3: Write minimal implementation**

In `src/components/intelligence/AnyProviderCard.tsx`, replace the Model `<label>` block and the discovery-error hint (currently lines ~193–213) with:

```tsx
          <label className="flex flex-col gap-1">
            <span style={labelStyle}>{t("externalProvider.modelLabel")}</span>
            {isLocalOnly && selectedProbe && localModels.length >= 1 ? (
              <select value={model} onChange={(e) => setModel(e.target.value)} style={fieldStyle}>
                <option value="">{t("externalProvider.modelSelectPlaceholder")}</option>
                {localModels.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            ) : (
              <>
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
              </>
            )}
          </label>
          {(discovery.isError || (isLocalOnly && selectedProbe?.isError)) && (
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
              {t("externalProvider.modelDiscoveryFailed")}
            </span>
          )}
```

- [ ] **Step 4: Add i18n key (all three locales)**

Add to the `externalProvider` block in each locale:

en: `modelSelectPlaceholder: "Select a model",`
zh-Hans: `modelSelectPlaceholder: "选择模型",`
zh-Hant: `modelSelectPlaceholder: "選擇模型",`

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: PASS — including the existing all-groups "discovery failure falls back to free-text" and "keyless save on 0.12" tests (both render with no `groups`, so `isLocalOnly` is false and the Model field stays an `<input>`).
Run: `pnpm test:i18n` → PASS.

- [ ] **Step 6: Run gates + Commit**

Run: `pnpm exec tsc -b` → no errors.
```bash
git add src/components/intelligence/AnyProviderCard.tsx src/components/intelligence/AnyProviderCard.test.tsx src/i18n/resources.ts
git commit -m "feat(intelligence): local model <select> over discovered ids (§9.2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Task 5b: Settings-card local parity — single probe, chip, model select (§9.2)

_Coordinator adjudication 2026-07-11: spec §9.2 covers "the wizard 'Local server' pane (or the settings card with a local preset selected)" — the settings surface is IN scope. In all-groups mode, when the SELECTED preset is local (`ollama`/`lmstudio`): the card probes that single preset's endpoint (the existing generic `discovery` query already does this — no second query needed), shows the connected/not-detected chip, and switches the model field to a `<select>` when discovery returns ≥1 model. No pills in settings — the Provider `<select>` stays. Dual-probe + pills remain wizard-local-pane-only (`isLocalOnly`). This task generalizes the chip + model-field code introduced in T4/T5 and rewrites the existing tests that typed into the ollama Model input._

**Files:**
- Modify: `src/components/intelligence/AnyProviderCard.tsx`
- Modify: `src/components/intelligence/AnyProviderCard.test.tsx`
- i18n: NO new keys — reuses `externalProvider.localConnectedChip` / `.localNotDetectedChip` / `.localProbing` (T4) and `.modelSelectPlaceholder` (T5).

**Interfaces:**
- Consumes: `isLocalOnly`, `selectedProbe`, `ollamaProbe`/`lmStudioProbe`, `localLabel`, `hostOf` (Task 4); the model `<select>` markup shape (Task 5); the generic `discovery` query (pre-existing).
- Produces: derived values `isLocalPreset` (selected preset's `group === "local"`), `localQuery` (the active local status query: wizard → fixed-endpoint probe, settings → `discovery`), `localQueryModels`. These REPLACE Task 4/5's direct uses of `selectedProbe`/`localModels` in the chip and model-field JSX.

- [ ] **Step 1: Write the failing tests (new settings-mode cases)**

Add these cases to `src/components/intelligence/AnyProviderCard.test.tsx`:

```ts
  it("settings card: selecting a local preset shows the connected chip and a model select", async () => {
    renderCard(); // all groups — provider <select>, no pills
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    expect(await screen.findByText(/Connected to Ollama — 1 models/)).toBeInTheDocument();
    // No pills in settings: the Provider select is still the picker.
    expect(screen.getByLabelText("Provider")).toBeInTheDocument();
    expect(screen.getByLabelText("Model").tagName).toBe("SELECT");
  });

  it("settings card: local preset with discovery failure keeps free text and shows the not-detected chip", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    expect(
      await screen.findByText(/Not detected at localhost:11434 — is Ollama running\?/),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Model").tagName).toBe("INPUT");
  });

  it("settings card: cloud presets keep the datalist input and never show a local chip", async () => {
    mocks.getDaemonVersion.mockResolvedValue("0.13.0");
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "openai");
    await waitFor(() => expect(mocks.listExternalModels).toHaveBeenCalled());
    expect(screen.getByLabelText("Model").tagName).toBe("INPUT");
    expect(screen.queryByText(/Connected to/)).not.toBeInTheDocument();
  });
```

- [ ] **Step 2: Rewrite the three existing ollama tests that the parity change breaks/strengthens**

In the same file, REPLACE these three existing tests in full. ("keyless save on 0.12" and "test button shows verbatim daemon error" break outright — the default `listExternalModels` mock resolves `["llama3.2:3b"]`, so under parity the Model field is a `<select>` and `userEvent.type` no longer applies. "discovery failure falls back to free-text" still passes but is strengthened to pin the element kind.)

Replace `it("keyless save on 0.12 omits the key and shows restart note", ...)` with:

```ts
  it("keyless save on 0.12 omits the key and shows restart note", async () => {
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    // Discovery succeeds → the model field is a <select> over discovered ids (§9.2).
    await screen.findByText(/Connected to Ollama/);
    await userEvent.selectOptions(screen.getByLabelText("Model"), "llama3.2:3b");
    await userEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(mocks.setExternalLlm).toHaveBeenCalledWith(
        "http://localhost:11434/v1", "llama3.2:3b", undefined
      )
    );
    expect(await screen.findByText(/Restart Wenlan to apply/)).toBeInTheDocument();
  });
```

Replace `it("discovery failure falls back to free-text model entry with hint", ...)` with:

```ts
  it("discovery failure falls back to free-text model entry with hint", async () => {
    mocks.listExternalModels.mockRejectedValue(new Error("ECONNREFUSED"));
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    expect(await screen.findByText(/type a model name/i)).toBeInTheDocument();
    const modelField = screen.getByLabelText("Model");
    expect(modelField.tagName).toBe("INPUT");
    expect(modelField).toBeEnabled();
  });
```

Replace `it("test button shows verbatim daemon error", ...)` with:

```ts
  it("test button shows verbatim daemon error", async () => {
    mocks.testExternalLlm.mockRejectedValue(new Error("LLM request failed: 401 Unauthorized"));
    renderCard();
    await userEvent.selectOptions(await screen.findByLabelText("Provider"), "ollama");
    await screen.findByText(/Connected to Ollama/);
    await userEvent.selectOptions(screen.getByLabelText("Model"), "llama3.2:3b");
    await userEvent.click(screen.getByRole("button", { name: "Test" }));
    expect(await screen.findByText(/401 Unauthorized/)).toBeInTheDocument();
  });
```

- [ ] **Step 3: Run tests to verify the new/rewritten cases fail**

Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: FAIL — settings mode has no chip and the Model field is an `<input>` (so `selectOptions` on it and the `SELECT` tagName assertions fail); the cloud-preset case passes vacuously last.

- [ ] **Step 4: Write minimal implementation**

In `src/components/intelligence/AnyProviderCard.tsx` (state as of end of Task 5):

4a. Replace the derived-probe lines introduced in Task 4 —

```ts
  const probeFor = (id: string) =>
    id === "ollama" ? ollamaProbe : id === "lmstudio" ? lmStudioProbe : null;
  const selectedProbe = probeFor(presetId);
  const localModels = selectedProbe?.data ?? [];
```

— with the generalized versions:

```ts
  const probeFor = (id: string) =>
    id === "ollama" ? ollamaProbe : id === "lmstudio" ? lmStudioProbe : null;
  const selectedProbe = probeFor(presetId);
  // §9.2 parity: the wizard local pane reads the fixed-endpoint probes; the
  // settings card (all groups) reuses the generic discovery query when a
  // local preset is selected. `localQuery` drives the chip and the <select>.
  const isLocalPreset = preset.group === "local";
  const localQuery = isLocalOnly ? selectedProbe : isLocalPreset ? discovery : null;
  const localQueryModels = localQuery?.data ?? [];
```

4b. In the Task 4 pills branch, the chip `<p>` currently lives INSIDE the `isLocalOnly ? (...)` fragment. Remove it from there and render it for both surfaces: replace the whole picker region (the `{isLocalOnly ? (<>…pills…{selectedProbe && (…chip…)}</>) : !hidePresetPicker ? (…select…) : null}` block) with:

```tsx
      {isLocalOnly ? (
        <div className="flex flex-wrap gap-2">
          {presets.map((p) => {
            const probe = probeFor(p.id);
            const status = !probe
              ? null
              : probe.isLoading
              ? "probing"
              : probe.isSuccess
              ? "connected"
              : "notDetected";
            const dot = status === "connected" ? "●" : status === "notDetected" ? "○" : "…";
            const selected = p.id === presetId;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => selectPreset(p.id)}
                className="rounded-full px-3 py-1.5 text-xs"
                style={{
                  border: `1px solid ${selected ? "var(--mem-accent-indigo)" : "var(--mem-border)"}`,
                  backgroundColor: selected ? "var(--mem-accent-indigo)" : "var(--mem-surface)",
                  color: selected ? "white" : "var(--mem-text)",
                  fontFamily: "var(--mem-font-body)",
                }}
              >
                {probe && (
                  <span
                    style={{
                      marginRight: "6px",
                      color:
                        status === "connected" && !selected
                          ? "var(--mem-accent-sage)"
                          : "inherit",
                    }}
                  >
                    {dot}
                  </span>
                )}
                {localLabel(p.name)}
              </button>
            );
          })}
        </div>
      ) : !hidePresetPicker ? (
        <label className="flex flex-col gap-1">
          <span style={labelStyle}>{t("externalProvider.presetLabel")}</span>
          <select value={presetId} onChange={(e) => selectPreset(e.target.value)} style={fieldStyle}>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </label>
      ) : null}

      {localQuery && (isLocalOnly || endpointValid) && (
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "12px",
            lineHeight: 1.5,
            color: localQuery.isSuccess
              ? "var(--mem-accent-sage)"
              : "var(--mem-text-secondary)",
          }}
        >
          {localQuery.isLoading
            ? t("externalProvider.localProbing", { name: localLabel(preset.name) })
            : localQuery.isSuccess
            ? t("externalProvider.localConnectedChip", {
                name: localLabel(preset.name),
                modelCount: localQueryModels.length,
              })
            : t("externalProvider.localNotDetectedChip", {
                name: localLabel(preset.name),
                host: hostOf(isLocalOnly ? preset.endpoint : trimmedEndpoint),
              })}
        </p>
      )}
```

4c. Generalize the Task 5 model field: replace its condition and options source —

```tsx
            {isLocalOnly && selectedProbe && localModels.length >= 1 ? (
              <select value={model} onChange={(e) => setModel(e.target.value)} style={fieldStyle}>
                <option value="">{t("externalProvider.modelSelectPlaceholder")}</option>
                {localModels.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
```

— with:

```tsx
            {localQuery && localQueryModels.length >= 1 ? (
              <select value={model} onChange={(e) => setModel(e.target.value)} style={fieldStyle}>
                <option value="">{t("externalProvider.modelSelectPlaceholder")}</option>
                {localQueryModels.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
```

4d. Generalize the Task 5 discovery-failed hint condition: replace

```tsx
          {(discovery.isError || (isLocalOnly && selectedProbe?.isError)) && (
```

with

```tsx
          {(discovery.isError || localQuery?.isError) && (
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/intelligence/AnyProviderCard.test.tsx`
Expected: PASS — new settings cases, rewritten ollama cases, all wizard-local cases from T4/T5 (in wizard mode `localQuery === selectedProbe`, so behavior there is unchanged), and all cloud cases (`isLocalPreset` false → no chip, datalist preserved).

- [ ] **Step 6: Run gates + Commit**

Run: `pnpm exec tsc -b` → no errors. `pnpm test:i18n` → PASS (no new keys).
```bash
git add src/components/intelligence/AnyProviderCard.tsx src/components/intelligence/AnyProviderCard.test.tsx
git commit -m "feat(intelligence): settings-card local parity — probe chip + model select (§9.2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Task 6: Plugin-first connect matrix v2 — CLI cards + Copy setup prompt (§9.3)

**Files:**
- Create: `src/components/connect/CliPrimaryPath.tsx`
- Modify: `src/components/connect/ClientSetupList.tsx`
- Create: `src/components/connect/ClientSetupList.test.tsx`
- Modify: `src/i18n/resources.ts` (3 locales)

**Interfaces:**
- Consumes: `detectMcpClients()` → `McpClient[]` (`{ name, client_type, config_path, detected, already_configured }`), `writeMcpConfig(clientType)`, `getWenlanMcpEntry()` → `{ command: string; args: string[] }`, `clipboardWrite(text)` — all from `src/lib/tauri.ts`.
- Produces (consumed by Task 6b): `CliPrimaryPath` default export, props `{ clientType: CliClientType }` where `export type CliClientType = "claude_code" | "codex_cli"`; named export `CLI_PRIMARY_CLIENTS: Set<string>`. The component is fully self-contained (fetches the MCP entry, builds the prompt, owns the copied state) so BOTH surfaces (settings list here, wizard rows in T6b) render identical primary-path copy from the same i18n keys.
- CLI clients (`client_type` in `claude_code`, `codex_cli`) lead with terminal commands + "Copy setup prompt"; the one-click config write moves under an `Advanced` `<details>`. GUI clients (`cursor`, `claude_desktop`, `gemini_cli`) keep the existing one-click "Set up" as primary.

- [ ] **Step 1: Write the failing test**

Create `src/components/connect/ClientSetupList.test.tsx`:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "../../i18n";

const mocks = vi.hoisted(() => ({
  detectMcpClients: vi.fn(),
  writeMcpConfig: vi.fn(),
  getWenlanMcpEntry: vi.fn(),
  clipboardWrite: vi.fn(),
}));
vi.mock("../../lib/tauri", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/tauri")>();
  return { ...actual, ...mocks };
});

import ClientSetupList from "./ClientSetupList";

const CLIENTS = [
  { name: "Claude Code", client_type: "claude_code", config_path: "~/.claude.json", detected: true, already_configured: false },
  { name: "Codex CLI", client_type: "codex_cli", config_path: "~/.codex/config.toml", detected: true, already_configured: false },
  { name: "Cursor", client_type: "cursor", config_path: "~/.cursor/mcp.json", detected: true, already_configured: false },
  { name: "Claude Desktop", client_type: "claude_desktop", config_path: "~/Library/.../config.json", detected: true, already_configured: false },
  { name: "Gemini CLI", client_type: "gemini_cli", config_path: "~/.gemini/settings.json", detected: true, already_configured: false },
];

function renderList() {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <ClientSetupList />
    </QueryClientProvider>,
  );
}

describe("ClientSetupList — §9.3 plugin-first matrix", () => {
  afterEach(() => Object.values(mocks).forEach((m) => m.mockReset()));
  beforeEach(() => {
    mocks.detectMcpClients.mockResolvedValue(CLIENTS);
    mocks.getWenlanMcpEntry.mockResolvedValue({ command: "npx", args: ["-y", "wenlan-mcp"] });
    mocks.writeMcpConfig.mockResolvedValue(undefined);
    mocks.clipboardWrite.mockResolvedValue(undefined);
  });

  it("Claude Code leads with the plugin commands", async () => {
    renderList();
    expect(await screen.findByText("claude plugin marketplace add 7xuanlu/wenlan")).toBeInTheDocument();
    expect(screen.getByText("claude plugin install wenlan@7xuanlu-wenlan")).toBeInTheDocument();
  });

  it("Codex leads with codex mcp add using the real command+args", async () => {
    renderList();
    expect(await screen.findByText("codex mcp add wenlan -- npx -y wenlan-mcp")).toBeInTheDocument();
  });

  it("Copy setup prompt writes the full agent prompt to the clipboard", async () => {
    renderList();
    const buttons = await screen.findAllByRole("button", { name: /Copy setup prompt/ });
    await userEvent.click(buttons[0]); // Claude Code card
    expect(mocks.clipboardWrite).toHaveBeenCalledTimes(1);
    expect(mocks.clipboardWrite.mock.calls[0][0]).toContain("claude plugin install wenlan@7xuanlu-wenlan");
  });

  it("GUI clients keep the one-click Set up as their primary action", async () => {
    renderList();
    // Cursor / Claude Desktop / Gemini CLI → 3 primary "Set up" buttons.
    const setUps = await screen.findAllByRole("button", { name: "Set up" });
    expect(setUps.length).toBeGreaterThanOrEqual(3);
  });

  it("shipped copy never references .mcpb or .codex-plugin", async () => {
    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <ClientSetupList />
      </QueryClientProvider>,
    );
    await screen.findByText("claude plugin install wenlan@7xuanlu-wenlan");
    expect(container.textContent).not.toContain(".mcpb");
    expect(container.textContent).not.toContain(".codex-plugin");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pnpm exec vitest run src/components/connect/ClientSetupList.test.tsx`
Expected: FAIL — current component renders a uniform "Set up" row for every client; no plugin commands, no "Copy setup prompt".

- [ ] **Step 3: Write minimal implementation**

3a. Create `src/components/connect/CliPrimaryPath.tsx` — the shared §9.3 primary path for CLI clients. It is fully self-contained (fetches the MCP entry, builds the i18n prompt, owns the copied state) so the wizard rows (Task 6b) render the identical block from the same i18n keys:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { clipboardWrite, getWenlanMcpEntry } from "../../lib/tauri";

export type CliClientType = "claude_code" | "codex_cli";

/** Clients whose §9.3 primary path is terminal commands, not one-click. */
export const CLI_PRIMARY_CLIENTS: Set<string> = new Set(["claude_code", "codex_cli"]);

/** §9.3 primary path for CLI clients, shared by ClientSetupList (settings)
 *  and the wizard ConnectStep rows: lead line, terminal command(s), reload
 *  note, and a "Copy setup prompt" button. */
export default function CliPrimaryPath({ clientType }: { clientType: CliClientType }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const { data: mcpEntry } = useQuery({
    queryKey: ["wenlan-mcp-entry"],
    queryFn: getWenlanMcpEntry,
    staleTime: Infinity,
  });
  const cmd = mcpEntry ? `${mcpEntry.command} ${mcpEntry.args.join(" ")}` : "";

  const isClaudeCode = clientType === "claude_code";
  const lead = isClaudeCode
    ? t("connectMatrix.claudeCodePrimary")
    : t("connectMatrix.codexPrimary");
  const commands = isClaudeCode
    ? [t("connectMatrix.claudeCodeCommand1"), t("connectMatrix.claudeCodeCommand2")]
    : [t("connectMatrix.codexCommand", { cmd })];
  const reload = isClaudeCode
    ? t("connectMatrix.claudeCodeReload")
    : t("connectMatrix.codexReload");
  const prompt = isClaudeCode
    ? t("connectMatrix.claudeCodePrompt")
    : t("connectMatrix.codexPrompt", { cmd });

  const copyPrompt = async () => {
    try {
      await clipboardWrite(prompt);
      setCopied(true);
    } catch {
      /* clipboard denial is non-fatal */
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", margin: 0 }}>
        {lead}
      </p>
      {commands.map((c) => (
        <code
          key={c}
          className="block truncate rounded-md px-2 py-1.5"
          style={{ fontFamily: "var(--mem-font-mono)", fontSize: "11px", backgroundColor: "var(--mem-bg)", border: "1px solid var(--mem-border)", color: "var(--mem-text)" }}
        >
          {c}
        </code>
      ))}
      <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)", margin: 0 }}>
        {reload}
      </p>
      <button
        type="button"
        onClick={copyPrompt}
        className="self-start rounded-md px-3 py-1.5 text-xs"
        style={{ border: "1px solid var(--mem-border)", color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}
      >
        {copied ? t("connectMatrix.promptCopied") : t("connectMatrix.copySetupPrompt")}
      </button>
    </div>
  );
}
```

3b. Replace the entire body of `src/components/connect/ClientSetupList.tsx` with:

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useState, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { detectMcpClients, writeMcpConfig, type McpClient } from "../../lib/tauri";
import CliPrimaryPath, { CLI_PRIMARY_CLIENTS, type CliClientType } from "./CliPrimaryPath";

/** Apps & CLIs group (spec §2a / §9.3). CLI clients lead with their primary
 *  plugin path (CliPrimaryPath: terminal commands + "Copy setup prompt");
 *  the one-click config write moves under Advanced. GUI clients keep the
 *  one-click "Set up". */
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

  const rowShell = (client: McpClient, children: ReactNode) => (
    <div
      key={client.client_type}
      className="rounded-lg px-3 py-2.5 flex flex-col gap-2"
      style={{ border: "1px solid var(--mem-border)", backgroundColor: "var(--mem-surface)" }}
    >
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", fontWeight: 500, color: "var(--mem-text)", margin: 0 }}>
            {client.name}
          </p>
          <p className="truncate" style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)", margin: 0 }}>
            {client.config_path}
          </p>
        </div>
        {client.already_configured && (
          <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-accent-sage)" }}>
            {t("connectMatrix.configured")}
          </span>
        )}
      </div>
      {children}
      {errors[client.client_type] && (
        <p className="text-red-500" style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", margin: 0 }}>
          {errors[client.client_type]}
        </p>
      )}
    </div>
  );

  const advancedSetUp = (client: McpClient) => (
    <details>
      <summary style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)", cursor: "pointer" }}>
        {t("connectMatrix.advanced")}
      </summary>
      <button
        onClick={() => setUp(client.client_type)}
        disabled={busy === client.client_type}
        className="mt-2 rounded-md px-3 py-1.5 text-xs disabled:opacity-50"
        style={{ border: "1px solid var(--mem-border)", color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}
      >
        {busy === client.client_type ? t("connectMatrix.settingUp") : t("connectMatrix.oneClickAdvanced")}
      </button>
    </details>
  );

  const guiPrimary = (client: McpClient) =>
    client.already_configured ? null : client.detected ? (
      <button
        onClick={() => setUp(client.client_type)}
        disabled={busy === client.client_type}
        className="self-start rounded-md px-3 py-1.5 text-xs disabled:opacity-50"
        style={{ backgroundColor: "var(--mem-accent-indigo)", color: "white", fontFamily: "var(--mem-font-body)" }}
      >
        {busy === client.client_type ? t("connectMatrix.settingUp") : t("connectMatrix.setUp")}
      </button>
    ) : (
      <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)" }}>
        {t("connectMatrix.notDetected")}
      </span>
    );

  return (
    <div className="flex flex-col" style={{ gap: "8px" }}>
      {(clients ?? []).map((client) =>
        CLI_PRIMARY_CLIENTS.has(client.client_type)
          ? rowShell(
              client,
              <>
                <CliPrimaryPath clientType={client.client_type as CliClientType} />
                {advancedSetUp(client)}
              </>,
            )
          : rowShell(client, guiPrimary(client)),
      )}
    </div>
  );
}
```

- [ ] **Step 4: Add i18n keys (all three locales)**

Add to the `connectMatrix` block in each locale (before the closing `},`, after `manualTitle`). Command keys are identical across locales (they are literal shell commands); `codexCommand` interpolates `{{cmd}}`.

en (block at L947):
```ts
    advanced: "Advanced",
    oneClickAdvanced: "Or write the config for me",
    copySetupPrompt: "Copy setup prompt",
    promptCopied: "Prompt copied",
    claudeCodePrimary: "Install the Wenlan plugin from your terminal:",
    claudeCodeCommand1: "claude plugin marketplace add 7xuanlu/wenlan",
    claudeCodeCommand2: "claude plugin install wenlan@7xuanlu-wenlan",
    claudeCodeReload: "Then run /reload-plugins or restart Claude Code.",
    claudeCodePrompt:
      "Install the Wenlan plugin for me. Run these shell commands directly (the /plugin menu can't be driven by an agent, so use the non-interactive commands): `claude plugin marketplace add 7xuanlu/wenlan` then `claude plugin install wenlan@7xuanlu-wenlan`. Approve any permission prompts that appear. When it's done, run /reload-plugins or ask me to restart Claude Code.",
    codexPrimary: "Add Wenlan to Codex from your terminal:",
    codexCommand: "codex mcp add wenlan -- {{cmd}}",
    codexReload: "Then restart Codex.",
    codexPrompt:
      "Add the Wenlan MCP server for me. Run this shell command directly (use the non-interactive command, not a menu): `codex mcp add wenlan -- {{cmd}}`. Approve any permission prompts that appear. When it's done, restart Codex.",
```
zh-Hans (block at L1893):
```ts
    advanced: "高级",
    oneClickAdvanced: "或替我写入配置",
    copySetupPrompt: "复制配置提示词",
    promptCopied: "提示词已复制",
    claudeCodePrimary: "在终端中安装 Wenlan 插件：",
    claudeCodeCommand1: "claude plugin marketplace add 7xuanlu/wenlan",
    claudeCodeCommand2: "claude plugin install wenlan@7xuanlu-wenlan",
    claudeCodeReload: "然后运行 /reload-plugins 或重启 Claude Code。",
    claudeCodePrompt:
      "帮我安装 Wenlan 插件。直接运行这些 shell 命令（/plugin 菜单无法由代理操作，请使用非交互式命令）：`claude plugin marketplace add 7xuanlu/wenlan`，然后 `claude plugin install wenlan@7xuanlu-wenlan`。批准出现的任何权限提示。完成后运行 /reload-plugins 或让我重启 Claude Code。",
    codexPrimary: "在终端中将 Wenlan 添加到 Codex：",
    codexCommand: "codex mcp add wenlan -- {{cmd}}",
    codexReload: "然后重启 Codex。",
    codexPrompt:
      "帮我添加 Wenlan MCP 服务器。直接运行这条 shell 命令（使用非交互式命令，而非菜单）：`codex mcp add wenlan -- {{cmd}}`。批准出现的任何权限提示。完成后重启 Codex。",
```
zh-Hant (block at L2838):
```ts
    advanced: "進階",
    oneClickAdvanced: "或替我寫入設定",
    copySetupPrompt: "複製設定提示詞",
    promptCopied: "提示詞已複製",
    claudeCodePrimary: "在終端機中安裝 Wenlan 外掛：",
    claudeCodeCommand1: "claude plugin marketplace add 7xuanlu/wenlan",
    claudeCodeCommand2: "claude plugin install wenlan@7xuanlu-wenlan",
    claudeCodeReload: "然後執行 /reload-plugins 或重新啟動 Claude Code。",
    claudeCodePrompt:
      "幫我安裝 Wenlan 外掛。直接執行這些 shell 命令（/plugin 選單無法由代理操作，請使用非互動式命令）：`claude plugin marketplace add 7xuanlu/wenlan`，然後 `claude plugin install wenlan@7xuanlu-wenlan`。核准出現的任何權限提示。完成後執行 /reload-plugins 或讓我重新啟動 Claude Code。",
    codexPrimary: "在終端機中將 Wenlan 新增到 Codex：",
    codexCommand: "codex mcp add wenlan -- {{cmd}}",
    codexReload: "然後重新啟動 Codex。",
    codexPrompt:
      "幫我新增 Wenlan MCP 伺服器。直接執行這條 shell 命令（使用非互動式命令，而非選單）：`codex mcp add wenlan -- {{cmd}}`。核准出現的任何權限提示。完成後重新啟動 Codex。",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/connect/ClientSetupList.test.tsx`
Expected: PASS (5 cases).
Run: `pnpm test:i18n`
Expected: PASS.

- [ ] **Step 6: Run gates + Commit**

Run: `pnpm exec tsc -b` → no errors.
```bash
git add src/components/connect/CliPrimaryPath.tsx src/components/connect/ClientSetupList.tsx src/components/connect/ClientSetupList.test.tsx src/i18n/resources.ts
git commit -m "feat(connect): plugin-first CLI cards with Copy setup prompt (§9.3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Task 6b: Wizard connect-step plugin-first parity (§9.3)

_Coordinator adjudication 2026-07-11: the §9.3 connect matrix applies to BOTH surfaces (wizard + settings, shared components); the user's feedback targeted the wizard screen specifically. The wizard's `renderClientList` (`src/components/SetupWizard.tsx`, rows rendered from the detected/supported sections at ~L772–780) keeps its visual idiom — checkbox rows, batch "Continue" — but detected CLI clients (Claude Code, Codex) now lead with the shared `CliPrimaryPath` block (Task 6), and the one-click config write is demoted to secondary: their checkbox defaults to UNCHECKED and is labeled by the existing `connectMatrix.oneClickAdvanced` line. GUI clients (Cursor, Claude Desktop, Gemini CLI) are unchanged. No new i18n keys._

**Files:**
- Modify: `src/components/SetupWizard.tsx` (ConnectStep: default-selection effect ~L492–503; `renderClientList` row body ~L660–685)
- Modify: `src/components/SetupWizard.test.tsx`

**Interfaces:**
- Consumes: `CliPrimaryPath` (default export, props `{ clientType: CliClientType }`) and `CLI_PRIMARY_CLIENTS: Set<string>` from `src/components/connect/CliPrimaryPath.tsx` (Task 6); i18n key `connectMatrix.oneClickAdvanced` (Task 6).

- [ ] **Step 1: Write the failing tests**

In `src/components/SetupWizard.test.tsx`, first extend the existing tauri import block (currently `import { detectMcpClients, writeMcpConfig, listAgents, setApiKey } from "../lib/tauri";`) to also pull the clipboard mock:

```ts
import {
  detectMcpClients,
  writeMcpConfig,
  listAgents,
  setApiKey,
  clipboardWrite,
} from "../lib/tauri";
```

Then add these tests inside `describe("SetupWizard", ...)` (the file's `vi.mock("../lib/tauri", ...)` factory already stubs `clipboardWrite` and `getWenlanMcpEntry` → `{ command: "npx", args: ["-y", "wenlan-mcp"] }`):

```ts
  it("connect step leads detected CLI clients with the plugin path and unchecks their one-click default", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Claude Code",
        client_type: "claude_code",
        config_path: "/path/to/claude.json",
        detected: true,
        already_configured: false,
      },
      {
        name: "Cursor",
        client_type: "cursor",
        config_path: "/path/to/cursor",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    // Primary path: the plugin commands render inside the wizard row.
    expect(
      await screen.findByText("claude plugin marketplace add 7xuanlu/wenlan"),
    ).toBeInTheDocument();
    expect(screen.getByText("claude plugin install wenlan@7xuanlu-wenlan")).toBeInTheDocument();

    // One-click demoted for CLI clients: checkbox defaults OFF; GUI stays ON.
    const cursorCheckbox = screen.getByRole("checkbox", { name: "Cursor" });
    await waitFor(() => expect(cursorCheckbox).toBeChecked());
    expect(screen.getByRole("checkbox", { name: "Claude Code" })).not.toBeChecked();
    expect(screen.getByText("Or write the config for me")).toBeInTheDocument();
  });

  it("connect step Copy setup prompt copies the Codex prompt with the real command", async () => {
    (detectMcpClients as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        name: "Codex CLI",
        client_type: "codex_cli",
        config_path: "/path/to/config.toml",
        detected: true,
        already_configured: false,
      },
    ]);

    renderWizard({ initialStep: "connect" });

    expect(
      await screen.findByText("codex mcp add wenlan -- npx -y wenlan-mcp"),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Copy setup prompt" }));
    await waitFor(() => expect(clipboardWrite).toHaveBeenCalledTimes(1));
    expect(
      (clipboardWrite as ReturnType<typeof vi.fn>).mock.calls[0][0],
    ).toContain("codex mcp add wenlan -- npx -y wenlan-mcp");
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm exec vitest run src/components/SetupWizard.test.tsx`
Expected: FAIL — the wizard rows render only the generic description; no plugin commands, no "Copy setup prompt"; the Claude Code checkbox defaults to checked.

- [ ] **Step 3: Write minimal implementation**

In `src/components/SetupWizard.tsx`:

3a. Add the import (next to the other `./connect/` imports at the top of the file):

```ts
import CliPrimaryPath, { CLI_PRIMARY_CLIENTS, type CliClientType } from "./connect/CliPrimaryPath";
```

3b. Demote the one-click default for CLI clients. In ConnectStep's selection-seeding effect, change the line

```ts
          next[client.client_type] = client.detected && !client.already_configured;
```

to

```ts
          // §9.3: CLI clients lead with the plugin path; the one-click batch
          // write is opt-in for them, so their checkbox starts unchecked.
          next[client.client_type] =
            client.detected &&
            !client.already_configured &&
            !CLI_PRIMARY_CLIENTS.has(client.client_type);
```

3c. In `renderClientList`, replace the description paragraph —

```tsx
              <p
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  color: "var(--mem-text-secondary)",
                  lineHeight: 1.5,
                  margin: 0,
                }}
              >
                {client.detected
                  ? t("setup.connect.detectedDescription")
                  : t("setup.connect.supportedDescription")}
              </p>
```

— with a branch that renders the shared primary path for detected, not-yet-connected CLI clients and keeps the existing copy everywhere else:

```tsx
              {CLI_PRIMARY_CLIENTS.has(client.client_type) && client.detected && !isConnected ? (
                <>
                  <CliPrimaryPath clientType={client.client_type as CliClientType} />
                  <p
                    style={{
                      fontFamily: "var(--mem-font-body)",
                      fontSize: "11px",
                      color: "var(--mem-text-tertiary)",
                      lineHeight: 1.5,
                      margin: 0,
                    }}
                  >
                    {t("connectMatrix.oneClickAdvanced")}
                  </p>
                </>
              ) : (
                <p
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "12px",
                    color: "var(--mem-text-secondary)",
                    lineHeight: 1.5,
                    margin: 0,
                  }}
                >
                  {client.detected
                    ? t("setup.connect.detectedDescription")
                    : t("setup.connect.supportedDescription")}
                </p>
              )}
```

(The `oneClickAdvanced` line — "Or write the config for me" — sits directly above the row's checkbox semantics: ticking the box and pressing Continue performs the demoted one-click write, exactly as before.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/SetupWizard.test.tsx`
Expected: PASS — the two new cases plus all existing connect-step cases ("separates detected and supported safe tools" uses an UNdetected Claude Code, which keeps the generic supported copy; "connects selected detected tools on continue" and "stays on connect step when MCP setup fails" use Cursor, whose default-checked one-click flow is unchanged).

- [ ] **Step 5: Run gates + Commit**

Run: `pnpm exec tsc -b` → no errors. `pnpm test:i18n` → PASS (no new keys).
```bash
git add src/components/SetupWizard.tsx src/components/SetupWizard.test.tsx
git commit -m "feat(wizard): connect step leads CLI clients with the shared plugin path (§9.3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Task 7: claude.ai plugin-first card + connector deep-link (§9.3)

_Coordinator adjudication #2, 2026-07-11 (screenshot-verified by the user; spec §9.3 claude.ai row updated on disk — re-read it): claude.ai's Directory → Plugins tab has an **Add marketplace** dialog accepting "a GitHub owner/repo or a git repository URL" (Anthropic / Partners / Personal tabs), so individuals CAN sync `7xuanlu/wenlan` today. The Claude card is therefore plugin-first and actionable now: **step 1** installs the Wenlan plugin (Directory → Plugins → + Add marketplace → `7xuanlu/wenlan` → Sync → install; skills activate in chat, Cowork gets the full plugin); **step 2** wires memory access via the custom connector (copy tunnel URL + deep link) — a public plugin cannot carry a user-specific tunnel URL, so chat still needs the connector. The existing no-auth tunnel warning stays untouched. ChatGPT card unchanged (connector only)._

**Files:**
- Modify: `src/components/connect/WebPlatformCards.tsx` (full-body restructure shown below)
- Modify: `src/components/connect/WebPlatformCards.test.tsx`
- Modify: `src/i18n/resources.ts` (3 locales)

**Interfaces:**
- Consumes: `import { open as shellOpen } from "@tauri-apps/plugin-shell"` (globally mocked in `src/test/setup.ts`).
- Produces: i18n keys `connectMatrix.claudePluginStepTitle`, `.claudePluginStep1`, `.claudePluginStep2`, `.claudePluginStep3`, `.claudePluginNote`, `.claudeConnectorStepTitle`, `.openConnectorSettings`.

- [ ] **Step 1: Write the failing tests**

In `src/components/connect/WebPlatformCards.test.tsx`, add the shell-opener import below the existing imports (the global mock in `src/test/setup.ts` provides `open: vi.fn()`):

```ts
import { open as shellOpen } from "@tauri-apps/plugin-shell";
```

Then add these cases inside the existing `describe("WebPlatformCards", ...)` block (this file uses `fireEvent` + the hoisted `mocks` object + `renderCards()` — reuse them; exact-string `getByText` is deliberate: it matches only the leaf element AND throws if the copy leaks onto the ChatGPT card):

```ts
  it("Claude card leads with the plugin install steps (Claude card only)", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({
      status: "connected", tunnel_url: "https://x.trycloudflare.com", token: "t", relay_url: null,
    });
    renderCards();
    expect(
      await screen.findByText("Step 1 — Install the Wenlan plugin"),
    ).toBeInTheDocument();
    // Marketplace repo string, exact li copy.
    expect(
      screen.getByText("Enter the marketplace repo 7xuanlu/wenlan and choose Sync"),
    ).toBeInTheDocument();
    // Honesty note: skills in chat, full plugin in Cowork.
    expect(
      screen.getByText(
        "The plugin's skills work in chat; Cowork gets the full plugin, including MCP connectors.",
      ),
    ).toBeInTheDocument();
    // Step 2 framing of the existing connector flow.
    expect(screen.getByText("Step 2 — Connect your memory")).toBeInTheDocument();
  });

  it("plugin steps render even when the tunnel is off (install is independent of the connector)", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({ status: "off" });
    renderCards();
    expect(
      await screen.findByText("Step 1 — Install the Wenlan plugin"),
    ).toBeInTheDocument();
    expect((await screen.findAllByText(/Turn on Remote Access/)).length).toBeGreaterThan(0);
  });

  it("opens the Claude connector settings via the deep link (step 2)", async () => {
    mocks.getRemoteAccessStatus.mockResolvedValue({
      status: "connected", tunnel_url: "https://x.trycloudflare.com", token: "t", relay_url: null,
    });
    vi.mocked(shellOpen).mockClear();
    renderCards();
    fireEvent.click(await screen.findByRole("button", { name: "Open connector settings" }));
    expect(shellOpen).toHaveBeenCalledWith(
      "https://claude.ai/settings/connectors?modal=add-custom-connector",
    );
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pnpm exec vitest run src/components/connect/WebPlatformCards.test.tsx`
Expected: FAIL — no plugin-step copy, no "Open connector settings" button. The four pre-existing cases still pass.

- [ ] **Step 3: Write minimal implementation**

Replace the entire body of `src/components/connect/WebPlatformCards.tsx` with (the state, queries, `copy` flow, URL row, connected hint, and no-auth warning are byte-identical to the current file — the restructure splits the old `card()` helper into `cardShell`/`stepList`/`urlRow` so the Claude card can carry two step sections):

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import { clipboardWrite, getRemoteAccessStatus, listAgents } from "../../lib/tauri";

const CLAUDE_CONNECTOR_URL = "https://claude.ai/settings/connectors?modal=add-custom-connector";

/** Per-platform web connect cards (spec §2a group 1; §9.3 round 2: the
 *  Claude card is plugin-first — step 1 installs the Wenlan plugin via the
 *  Directory's Add-marketplace dialog, step 2 wires memory access via the
 *  custom connector). Verification is the existing listAgents delta poll —
 *  best-effort attribution: a new agent in the poll window flips the
 *  last-copied card (hint, not proof). */
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
    remote?.status === "connected" ? (remote.relay_url ?? `${remote.tunnel_url}/mcp`) : null;

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
    try {
      await clipboardWrite(url);
    } catch {
      return;
    }
    baselineCount.current = agents?.length ?? baselineCount.current;
    setCopiedPlatform(platform);
  };

  const stepHeading = (text: string) => (
    <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", fontWeight: 600, color: "var(--mem-text)", margin: 0 }}>
      {text}
    </p>
  );

  const stepList = (steps: string[]) => (
    <ol style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", lineHeight: 1.7, paddingLeft: "18px", listStyle: "decimal", margin: 0 }}>
      {steps.map((s) => (
        <li key={s}>{s}</li>
      ))}
    </ol>
  );

  const urlRow = (platform: string) =>
    url ? (
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
    );

  /* No-auth boundary (council change f, commit 3a272d0): always visible. */
  const noAuthWarning = (
    <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-accent-amber)", lineHeight: 1.5 }}>
      {t("connectMatrix.noAuthWarning")}
    </p>
  );

  const cardShell = (platform: "claude" | "chatgpt", title: string, children: ReactNode) => (
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
      {children}
    </div>
  );

  return (
    <div className="flex flex-col" style={{ gap: "12px" }}>
      {cardShell(
        "claude",
        t("connectMatrix.claudeTitle"),
        <>
          {/* Step 1 — install the Wenlan plugin (§9.3: actionable today via
              Directory → Plugins → + Add marketplace, Personal tab). */}
          {stepHeading(t("connectMatrix.claudePluginStepTitle"))}
          {stepList([
            t("connectMatrix.claudePluginStep1"),
            t("connectMatrix.claudePluginStep2"),
            t("connectMatrix.claudePluginStep3"),
          ])}
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)", lineHeight: 1.5, margin: 0 }}>
            {t("connectMatrix.claudePluginNote")}
          </p>
          {/* Step 2 — memory access: a public plugin cannot carry a
              user-specific tunnel URL, so chat still needs the connector. */}
          {stepHeading(t("connectMatrix.claudeConnectorStepTitle"))}
          {stepList([
            t("connectMatrix.claudeStep1"),
            t("connectMatrix.claudeStep2"),
            t("connectMatrix.claudeStep3"),
          ])}
          {urlRow("claude")}
          {url && (
            <button
              type="button"
              onClick={() => shellOpen(CLAUDE_CONNECTOR_URL)}
              className="self-start rounded-md px-3 py-1.5 text-xs"
              style={{ border: "1px solid var(--mem-border)", color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}
            >
              {t("connectMatrix.openConnectorSettings")}
            </button>
          )}
          {noAuthWarning}
        </>,
      )}
      {cardShell(
        "chatgpt",
        t("connectMatrix.chatgptTitle"),
        <>
          {stepList([
            t("connectMatrix.chatgptStep1"),
            t("connectMatrix.chatgptStep2"),
            t("connectMatrix.chatgptStep3"),
          ])}
          {urlRow("chatgpt")}
          {noAuthWarning}
        </>,
      )}
    </div>
  );
}
```

- [ ] **Step 4: Add i18n keys (all three locales)**

Add to the `connectMatrix` block in each locale (before the closing `},`):

en:
```ts
    claudePluginStepTitle: "Step 1 — Install the Wenlan plugin",
    claudePluginStep1: "Open Directory → Plugins → + Add marketplace",
    claudePluginStep2: "Enter the marketplace repo 7xuanlu/wenlan and choose Sync",
    claudePluginStep3: "Install Wenlan from the marketplace",
    claudePluginNote:
      "The plugin's skills work in chat; Cowork gets the full plugin, including MCP connectors.",
    claudeConnectorStepTitle: "Step 2 — Connect your memory",
    openConnectorSettings: "Open connector settings",
```
zh-Hans:
```ts
    claudePluginStepTitle: "第 1 步 — 安装 Wenlan 插件",
    claudePluginStep1: "打开 Directory → 插件 → 「+ 添加市场」",
    claudePluginStep2: "输入市场仓库 7xuanlu/wenlan，然后选择同步",
    claudePluginStep3: "从市场安装 Wenlan",
    claudePluginNote: "插件的技能可在聊天中使用；Cowork 支持完整插件（含 MCP 连接器）。",
    claudeConnectorStepTitle: "第 2 步 — 连接你的记忆",
    openConnectorSettings: "打开连接器设置",
```
zh-Hant:
```ts
    claudePluginStepTitle: "第 1 步 — 安裝 Wenlan 外掛",
    claudePluginStep1: "開啟 Directory → 外掛 → 「+ 新增市集」",
    claudePluginStep2: "輸入市集儲存庫 7xuanlu/wenlan，然後選擇同步",
    claudePluginStep3: "從市集安裝 Wenlan",
    claudePluginNote: "外掛的技能可在聊天中使用；Cowork 支援完整外掛（含 MCP 連接器）。",
    claudeConnectorStepTitle: "第 2 步 — 連接你的記憶",
    openConnectorSettings: "開啟連接器設定",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pnpm exec vitest run src/components/connect/WebPlatformCards.test.tsx`
Expected: PASS — three new cases plus the four pre-existing ones (no-auth warning ×2, URL shown, tunnel-off prompt, `/mcp`-suffix copy — all preserved by the byte-identical `urlRow`/warning markup).
Run: `pnpm exec vitest run src/components/SetupWizard.test.tsx`
Expected: PASS — the wizard renders WebPlatformCards on its connect step; its assertions (exact command-string matches, checkbox names) do not collide with the new card copy.
Run: `pnpm test:i18n` → PASS.

- [ ] **Step 6: Run gates + Commit**

Run: `pnpm exec tsc -b` → no errors.
```bash
git add src/components/connect/WebPlatformCards.tsx src/components/connect/WebPlatformCards.test.tsx src/i18n/resources.ts
git commit -m "feat(connect): claude.ai plugin-first card + connector deep link (§9.3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01QTSrd2Uwd6Lg1QqDbX6P53"
```

---

## Final verification (run after Task 7)

- [ ] `pnpm exec tsc -b` → no errors
- [ ] `pnpm test` → all suites green
- [ ] `pnpm test:i18n` → green
- [ ] `pnpm build` → succeeds

---

## Self-Review

**1. Spec coverage:**
- §9.1 preset extension + renames → T1. Placeholder → T2 (AnyProviderCard), T3 (Anthropic). Soft hint → T1 (logic) + T2 (UI). Get-a-key link → T2 + T3. ✅
- §9.2 BOTH surfaces: wizard pane probe-both + status pills + auto-select + chip → T4; wizard dropdown vs free-text → T5; settings card with local preset selected: single probe + chip + dropdown, Provider `<select>` stays, existing ollama tests rewritten → T5b. Cloud presets unchanged → datalist branch kept in T5/T5b (asserted by T5b's cloud-preset test). ✅
- §9.3 BOTH surfaces: shared `CliPrimaryPath` (Claude Code plugin commands / Codex `codex mcp add` + Copy setup prompt) → T6; settings list with one-click demoted to Advanced → T6; wizard ConnectStep rows lead with the same component, CLI checkbox default off → T6b. Claude Desktop / Gemini CLI / Cursor one-click → T6 (GUI primary), wizard unchanged for them (T6b). claude.ai plugin-first two-step card (step 1: Add-marketplace plugin install, actionable today; step 2: connector — copy URL + deep link; no-auth warning untouched) → T7. ChatGPT connector-only + no-auth warning → unchanged (no `.mcpb`/`.codex-plugin` copy anywhere — asserted in T6). ✅
- §9.4 tests → data-shape/soft-hint (T1), placeholder/hint/link (T2/T3), dual-probe render + auto-select (T4), dropdown vs free-text wizard (T5) and settings (T5b), primary-path copy + setup-prompt clipboard in settings (T6) and wizard (T6b), claude.ai plugin-step copy + repo string + deep-link + tunnel-off independence (T7), i18n parity via `pnpm test:i18n` in every copy-touching task. No Rust. ✅

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N"/"write tests for the above" — every code and test step carries complete code, including T5b's full rewrites of the three pre-existing ollama tests and T6b's wizard row branch. ✅

**3. Type consistency:** `keyPrefixMismatch(preset, key)` (T1) is consumed with the same signature in T2. `isLocalOnly`/`selectedProbe`/`ollamaProbe`/`lmStudioProbe`/`localLabel`/`hostOf` are defined in T4, consumed in T5, and generalized in T5b to `isLocalPreset`/`localQuery`/`localQueryModels` (T5b step 4 shows the exact before→after blocks it replaces). `CliPrimaryPath` props `{ clientType: CliClientType }` and `CLI_PRIMARY_CLIENTS` are defined in T6 and consumed with identical names in T6b. `getWenlanMcpEntry()` returns `{ command, args }` used to build `cmd` in T6's `CliPrimaryPath` matching `src/lib/tauri.ts:1991`. `McpClient` fields (`client_type`, `already_configured`, `detected`, `config_path`, `name`) match `src/lib/tauri.ts:1948`. i18n keys referenced in components (`externalProvider.getKeyLink`, `.keyHint`, `.localConnectedChip`, `.localNotDetectedChip`, `.localProbing`, `.modelSelectPlaceholder`; `connectMatrix.advanced`, `.oneClickAdvanced`, `.copySetupPrompt`, `.promptCopied`, `.claudeCodePrimary`, `.claudeCodeCommand1/2`, `.claudeCodeReload`, `.claudeCodePrompt`, `.codexPrimary`, `.codexCommand`, `.codexReload`, `.codexPrompt`, `.claudePluginStepTitle`, `.claudePluginStep1/2/3`, `.claudePluginNote`, `.claudeConnectorStepTitle`, `.openConnectorSettings`) are all added in en/zh-Hans/zh-Hant; T5b and T6b add none; T7's card reuses the pre-existing `.claudeTitle`/`.claudeStep1-3`/`.copyUrl`/`.copied`/`.tunnelOff`/`.noAuthWarning`/`.connectedHint` unchanged. ✅

## Execution Handoff

Plan complete. Two execution options: (1) Subagent-driven (fresh subagent per task, review between — recommended); (2) Inline execution with checkpoints. Tasks are ordered by dependency: T1 → T2/T3 (parallelizable) → T4 → T5 → T5b → T6 → T6b → T7.
