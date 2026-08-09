# Wenlan App Interface Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add English, Simplified Chinese, and Traditional Chinese interface localization to the Wenlan desktop app.

**Architecture:** The frontend owns UI localization with `i18next` and `react-i18next`, using static bundled resource objects loaded before React renders. A small app-specific locale resolver maps system/browser language tags into `en`, `zh-Hans`, or `zh-Hant`, persists explicit choices in `localStorage`, and leaves user content, daemon output, MCP responses, and LLM text untranslated.

**Tech Stack:** React 19, Vite 6, TypeScript, Vitest + Testing Library, `i18next`, `react-i18next`, Tauri 2.

---

## Scope Boundaries

- Locales: `en`, `zh-Hans`, `zh-Hant`.
- Settings selector: `System`, `English`, `简体中文`, `繁體中文`.
- Persistence key: `wenlan-locale`.
- Fallback language: `en`.
- Translate fixed app interface only: setup wizard, main navigation/header/search placeholder, settings, home, quick capture UI, toast labels, updater, import, and intelligence setup.
- Do not translate user-generated memory/page/source/entity content, daemon/backend/MCP output, LLM-generated content, or add RTL support.
- Keep the Rust-side `Quick Capture` window title English unless the frontend title change is added with an explicit Tauri Window API test boundary.

## Task 1: Locale Resolver and Resource Contract

**Files:**
- Create: `src/i18n/locales.ts`
- Create: `src/i18n/resources.ts`
- Create: `src/i18n/locales.test.ts`
- Create: `src/i18n/resources.test.ts`

- [ ] Write failing tests proving script-aware locale mapping:
  - `zh`, `zh-CN`, `zh-SG`, and `zh-Hans-*` resolve to `zh-Hans`.
  - `zh-TW`, `zh-HK`, `zh-MO`, and `zh-Hant-*` resolve to `zh-Hant`.
  - unsupported or empty values resolve to `en`.
  - explicit stored choices accept only `en`, `zh-Hans`, `zh-Hant`, or `system`.
- [ ] Write a failing key parity test that flattens all resource trees and requires `zh-Hans` and `zh-Hant` to contain exactly the English key set.
- [ ] Implement the resolver and initial resource objects with enough keys to satisfy the first translated surfaces.
- [ ] Run: `pnpm test -- src/i18n/locales.test.ts src/i18n/resources.test.ts`.

## Task 2: i18next Runtime Wiring

**Files:**
- Create: `src/i18n/index.ts`
- Create: `src/i18n/types.d.ts`
- Modify: `src/main.tsx`
- Modify: `src/test/setup.ts`
- Modify: `package.json`
- Modify: `pnpm-lock.yaml`

- [ ] Add `i18next` and `react-i18next`.
- [ ] Initialize i18next synchronously with static `resources`, `supportedLngs: ["en", "zh-Hans", "zh-Hant"]`, `fallbackLng: "en"`, `load: "currentOnly"`, and `react: { useSuspense: false }`.
- [ ] Use `initReactI18next` and `interpolation.escapeValue=false`.
- [ ] Initialize the same module in Vitest setup so existing English assertions keep using real English strings.
- [ ] Add TypeScript module augmentation for the default namespace and resource shape.
- [ ] Run: `pnpm exec tsc -b`.

## Task 3: Settings Language Selector

**Files:**
- Modify: `src/components/memory/SettingsPage.tsx`
- Modify: `src/components/memory/settings/SettingsSidebar.tsx`
- Test: add or extend a settings render test under `src/components/memory/`

- [ ] Write a failing render test showing the Settings language selector displays `System`, `English`, `简体中文`, and `繁體中文`.
- [ ] Write a failing persistence test showing a selector change writes `wenlan-locale` and calls i18n language change through the resolver.
- [ ] Implement the selector in General settings without changing unrelated config behavior.
- [ ] Translate Settings sidebar labels and fixed Settings page copy.
- [ ] Run the targeted settings tests.

## Task 4: Primary Interface Translation Pass

**Files:**
- Modify: `src/components/SetupWizard.tsx`
- Modify: `src/components/memory/Main.tsx`
- Modify: `src/components/memory/Sidebar.tsx`
- Modify: `src/components/memory/HomePage.tsx`
- Modify: `src/components/QuickCapture.tsx`
- Modify: `src/components/QuickCaptureWindow.tsx`
- Modify: `src/components/ToastOverlay.tsx`
- Modify: `src/components/UpdaterDialog.tsx`
- Modify: `src/components/ChatImport/*.tsx`
- Modify: `src/components/intelligence/IntelligenceSetup.tsx`

- [ ] Convert fixed interface strings on the named surfaces to `t(...)` keys.
- [ ] Use i18next interpolation and pluralization for count-bearing strings.
- [ ] Keep user content and daemon/LLM text raw.
- [ ] Add English, Simplified Chinese, and Traditional Chinese resource entries for every new key.
- [ ] Fix `Main.tsx` search focus logic so it uses a ref or data attribute instead of querying by visible placeholder text.
- [ ] Keep the Rust Quick Capture title in English and document that full Rust-side i18n is out of scope for this pass.
- [ ] Run targeted render tests that cover setup, main search, home, import, and updater surfaces.

## Task 5: Verification and Audit

**Files:**
- All changed files.

- [ ] Run: `pnpm exec tsc -b`.
- [ ] Run: `pnpm test`.
- [ ] Inspect `rg` results for remaining hardcoded visible English in the in-scope primary surfaces.
- [ ] Confirm `.gitignore` remains untouched.
- [ ] If Rust/Tauri title behavior is changed, run `cd app && cargo test`; otherwise record that Rust was not touched for the title.
