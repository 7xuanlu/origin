# Settings Decomposition Implementation Plan (PR 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 1,111-line `SettingsPage.tsx` into per-section modules with zero behavior change — a strictly behavior-preserving refactor PR that lands before the redesign features stack on it (spec §4, council change a).

**Architecture:** `SettingsPage.tsx` becomes a thin section switch (< ~150 lines). Each settings group moves to `src/components/memory/settings/sections/<Name>Section.tsx`, owning its own hooks and JSX. Shared UI helpers move to `src/components/memory/settings/primitives.tsx`.

**Tech Stack:** React 19, TypeScript strict, Vitest + jsdom, TanStack Query, react-i18next. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-10-settings-onboarding-redesign-design.md` §4.

## Global Constraints

- **Zero behavior change**: no copy, layout, i18n, or logic edits. Cut-paste moves only.
- Existing tests must stay green **without loosening assertions**: `SettingsPage.language.test.tsx`, `SettingsSidebar.test.tsx`, `SetupWizard.test.tsx`.
- TypeScript strict, no-emit: `pnpm exec tsc -b` must pass after every task.
- The project does NOT import React for JSX (`react-jsx` transform); `React.ReactNode` in type positions works without an import — copy this idiom, don't add `import React`.
- Branch: `settings-decomposition`, cut from `origin/main`. PR base: `main`. Never push to main.
- Every commit message ends with the Co-Authored-By + Claude-Session trailer used in this session.

## File Structure (end state)

```text
src/components/memory/
|-- SettingsPage.tsx                  # thin switch, < ~150 lines (modified)
`-- settings/
    |-- SettingsSidebar.tsx           # unchanged
    |-- primitives.tsx                # NEW: Toggle, SettingRow, SectionHeader
    `-- sections/
        |-- GeneralSection.tsx        # NEW: profile + app + theme + language + rerun-setup
        |-- CaptureSection.tsx        # NEW: clipboard + screen capture (hidden group, kept)
        |-- SourcesSection.tsx        # NEW: import memories + chat history + sources list
        |-- AgentsSection.tsx         # NEW: connected agents + remote access
        |-- IntelligenceSection.tsx   # NEW: ApiKeyCard + OnDeviceModelCard wrapper
        `-- DiagnosticsSection.tsx    # MOVED from settings/DiagnosticsSection.tsx
```

Line references below are to `SettingsPage.tsx` at commit `e77923f` (current
`origin/main`); verify with `git blame`-free reading before each cut — if the
file drifted, locate the same code by its section comments (`{/* ── General … */}` etc.).

---

### Task 1: Branch + primitives.tsx

**Files:**
- Create: `src/components/memory/settings/primitives.tsx`
- Modify: `src/components/memory/SettingsPage.tsx` (delete local helper definitions, import from primitives)

**Interfaces:**
- Produces: `export function Toggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void })`; `export function SettingRow({ title, description, enabled, onToggle, statusLine?, warning?, error? })`; `export function SectionHeader({ icon, label }: { icon: React.ReactNode; label: string })` — exact bodies copied from `SettingsPage.tsx:77-153`.

- [ ] **Step 1: Create the branch**

```bash
git fetch origin
git checkout -b settings-decomposition origin/main
```

Note: this worktree currently has the spec branch checked out; the spec commits stay on that branch — this PR is code-only.

- [ ] **Step 2: Baseline — run the existing suite (must be green before any change)**

Run: `pnpm exec tsc -b && pnpm test`
Expected: PASS (this is the refactor's oracle; record any pre-existing failure and STOP if the baseline is red).

- [ ] **Step 3: Create `primitives.tsx`**

File content: SPDX header line (`// SPDX-License-Identifier: AGPL-3.0-only`), then the three components cut verbatim from `SettingsPage.tsx` lines 77–153 (`Toggle`, `SettingRow`, `SectionHeader`), each prefixed with `export`. No other imports are needed (JSX transform + UMD React types). `SettingRow` references `Toggle` — same file, no import.

- [ ] **Step 4: Rewire `SettingsPage.tsx`**

Delete lines 75–153 (the `// ── Helpers ──` block) and add to the import block:

```tsx
import { Toggle, SettingRow, SectionHeader } from "./settings/primitives";
```

- [ ] **Step 5: Verify**

Run: `pnpm exec tsc -b && pnpm test`
Expected: PASS, same test counts as baseline.

- [ ] **Step 6: Commit**

```bash
git add src/components/memory/settings/primitives.tsx src/components/memory/SettingsPage.tsx
git commit -m "refactor(settings): extract Toggle/SettingRow/SectionHeader to primitives.tsx"
```

---

### Task 2: GeneralSection

**Files:**
- Create: `src/components/memory/settings/sections/GeneralSection.tsx`
- Modify: `src/components/memory/SettingsPage.tsx`

**Interfaces:**
- Consumes: `SectionHeader`, `SettingRow` from `../primitives`.
- Produces: `export default function GeneralSection(): JSX element` — no props; owns theme, language, run-at-login, rerun-setup state and the profile block.

- [ ] **Step 1: Create `GeneralSection.tsx`**

Skeleton (moved code is cut verbatim from `SettingsPage.tsx`; ranges given):

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { open } from "@tauri-apps/plugin-dialog";
import { useTranslation } from "react-i18next";
import {
  getProfile,
  updateProfile,
  setAvatar,
  removeAvatar,
  setSetupCompleted,
  isRunAtLoginEnabled,
  setRunAtLogin,
} from "../../../../lib/tauri";
import { type Theme, useTheme } from "../../../../lib/theme";
import {
  readStoredLocalePreference,
  setLocalePreference,
  type StoredLocale,
} from "../../../../i18n";
import { SectionHeader, SettingRow } from "../primitives";
import ProfileAvatar from "../../ProfileAvatar";

// [PASTE from SettingsPage.tsx]
// - ThemeLabelKey type + THEME_OPTIONS const        (lines 40-73)
// - formatProfileMonth + ProfileUpdateFields         (lines 155-167)
// - ProfileSettingsBlock                             (lines 169-325)

export default function GeneralSection() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [theme, setThemeValue] = useTheme();
  const [languagePreference, setLanguagePreference] = useState<StoredLocale>(
    () => readStoredLocalePreference(),
  );
  // [PASTE run-at-login hooks: SettingsPage.tsx lines 388-395]
  return (
    <>
      {/* [PASTE the `section === "general"` JSX body: lines 455-569 —
          everything between `<>` and `</>` inside the gate, i.e.
          <ProfileSettingsBlock /> + the general <section>] */}
    </>
  );
}
```

- [ ] **Step 2: Rewire `SettingsPage.tsx`**

Replace the `{section === "general" && (<>…</>)}` block with:

```tsx
{section === "general" && <GeneralSection />}
```

Add `import GeneralSection from "./settings/sections/GeneralSection";`. Then delete from `SettingsPage.tsx`: `THEME_OPTIONS`/`ThemeLabelKey`, `formatProfileMonth`, `ProfileUpdateFields`, `ProfileSettingsBlock`, the theme/language/run-at-login hooks, and every import that `tsc` now flags as unused (`open`, `getProfile`, `updateProfile`, `setAvatar`, `removeAvatar`, `isRunAtLoginEnabled`, `setRunAtLogin`, `useTheme`, `Theme`, i18n locale helpers, `ProfileAvatar`, `useEffect` if unused).

- [ ] **Step 3: Verify**

Run: `pnpm exec tsc -b && pnpm test`
Expected: PASS. `SettingsPage.language.test.tsx` exercises the language select — it must pass unmodified.

- [ ] **Step 4: Commit**

```bash
git add -A src/components/memory
git commit -m "refactor(settings): extract GeneralSection"
```

---

### Task 3: CaptureSection

**Files:**
- Create: `src/components/memory/settings/sections/CaptureSection.tsx`
- Modify: `src/components/memory/SettingsPage.tsx`

**Interfaces:**
- Produces: `export default function CaptureSection()` — no props. (Group is currently hidden from the sidebar but the code path stays, spec non-goal "hidden capture group stays hidden".)

- [ ] **Step 1: Create `CaptureSection.tsx`**

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  getClipboardEnabled,
  setClipboardEnabled,
  getScreenCaptureEnabled,
  setScreenCaptureEnabled,
  checkScreenPermission,
  requestScreenPermission,
  getCaptureStats,
} from "../../../../lib/tauri";
import { SectionHeader, SettingRow, Toggle } from "../primitives";

export default function CaptureSection() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  // [PASTE clipboard + screen-capture + captureStats hooks: SettingsPage.tsx lines 354-385]
  return (
    // [PASTE the `section === "capture"` <section> JSX: lines 575-655]
  );
}
```

- [ ] **Step 2: Rewire `SettingsPage.tsx`** — `{section === "capture" && <CaptureSection />}`, add the import, delete the moved hooks and now-unused imports (`getClipboardEnabled`, `setClipboardEnabled`, `getScreenCaptureEnabled`, `setScreenCaptureEnabled`, `checkScreenPermission`, `requestScreenPermission`, `getCaptureStats`).

- [ ] **Step 3: Verify** — `pnpm exec tsc -b && pnpm test` → PASS.

- [ ] **Step 4: Commit** — `git add -A src/components/memory && git commit -m "refactor(settings): extract CaptureSection"`

---

### Task 4: SourcesSection

**Files:**
- Create: `src/components/memory/settings/sections/SourcesSection.tsx`
- Modify: `src/components/memory/SettingsPage.tsx`

**Interfaces:**
- Produces: `export default function SourcesSection({ onImport }: { onImport?: () => void })`.
- Consumes: the existing sources list component `src/components/memory/sources/SourcesSection.tsx` — import it renamed to avoid self-shadowing: `import SourcesList from "../../sources/SourcesSection";` and use `<SourcesList />` where the old code had `<SourcesSection />`.

- [ ] **Step 1: Create the file**

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useTranslation } from "react-i18next";
import SourcesList from "../../sources/SourcesSection";
import { ImportFlow } from "../../../ChatImport/ImportFlow";
import { SectionHeader } from "../primitives";

export default function SourcesSection({ onImport }: { onImport?: () => void }) {
  const { t } = useTranslation();
  return (
    <>
      {/* [PASTE the `section === "sources"` JSX body: lines 660-712 —
          three <section> blocks; replace `<SourcesSection />` with `<SourcesList />`] */}
    </>
  );
}
```

- [ ] **Step 2: Rewire `SettingsPage.tsx`** — `{section === "sources" && <SourcesSection onImport={onImport} />}`; delete unused imports (`SourcesSection` from sources/, `ImportFlow`).

- [ ] **Step 3: Verify** — `pnpm exec tsc -b && pnpm test` → PASS.

- [ ] **Step 4: Commit** — `git commit -am "refactor(settings): extract SourcesSection"`

---

### Task 5: AgentsSection

**Files:**
- Create: `src/components/memory/settings/sections/AgentsSection.tsx`
- Modify: `src/components/memory/SettingsPage.tsx`

**Interfaces:**
- Produces: `export default function AgentsSection({ onSetupAgent }: { onSetupAgent?: () => void })` — owns the `agents`/`mcp-clients` queries, update/delete mutations, `deletingAgent` state, trust-level UI, and the Remote Access panel.

- [ ] **Step 1: Create the file**

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  listAgents,
  updateAgent,
  deleteAgent,
  detectMcpClients,
} from "../../../../lib/tauri";
import { describeTrustLevel, resolveAgentDisplayName, TRUST_LEVELS } from "../../../../lib/agents";
import { RemoteAccessPanel } from "../../RemoteAccessPanel";
import { SectionHeader, Toggle } from "../primitives";

export default function AgentsSection({ onSetupAgent }: { onSetupAgent?: () => void }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  // [PASTE agents/mcpClients queries + pendingClients derivation +
  //  updateAgentMut/deleteAgentMut + deletingAgent state: lines 400-428]
  return (
    <>
      {/* [PASTE the `section === "agents"` JSX body: lines 717-1054 —
          Connected Agents <section> + Remote Access <section>] */}
    </>
  );
}
```

- [ ] **Step 2: Rewire `SettingsPage.tsx`** — `{section === "agents" && <AgentsSection onSetupAgent={onSetupAgent} />}`; delete moved hooks/state and unused imports (`listAgents`, `updateAgent`, `deleteAgent`, `detectMcpClients`, agents lib helpers, `RemoteAccessPanel`).

- [ ] **Step 3: Verify** — `pnpm exec tsc -b && pnpm test` → PASS.

- [ ] **Step 4: Commit** — `git commit -am "refactor(settings): extract AgentsSection"`

---

### Task 6: IntelligenceSection + DiagnosticsSection move + final slim-down

**Files:**
- Create: `src/components/memory/settings/sections/IntelligenceSection.tsx`
- Move: `src/components/memory/settings/DiagnosticsSection.tsx` → `src/components/memory/settings/sections/DiagnosticsSection.tsx`
- Modify: `src/components/memory/SettingsPage.tsx`

**Interfaces:**
- Produces: `export default function IntelligenceSection({ delay }: { delay: number })` — exact move of `SettingsPage.tsx:1086-1110`. PR 2 of the redesign replaces this file's content; keep the signature.

- [ ] **Step 1: Create `IntelligenceSection.tsx`**

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useTranslation } from "react-i18next";
import { ApiKeyCard, OnDeviceModelCard, useApiKeyStatus } from "../../../intelligence/IntelligenceSetup";
import { SectionHeader } from "../primitives";

// [PASTE function IntelligenceSection from SettingsPage.tsx:1086-1110,
//  prefixed `export default`]
```

- [ ] **Step 2: Move DiagnosticsSection**

```bash
git mv src/components/memory/settings/DiagnosticsSection.tsx src/components/memory/settings/sections/DiagnosticsSection.tsx
```

Then open the moved file and deepen its relative imports by one level (`../` → `../../` etc.) until `tsc` is clean.

- [ ] **Step 3: Slim `SettingsPage.tsx`**

End state (complete file structure — header/footer JSX kept verbatim from the original):

```tsx
// SPDX-License-Identifier: AGPL-3.0-only
import { useTranslation } from "react-i18next";
import type { SettingsSection } from "./settings/SettingsSidebar";
import { SETTINGS_GROUPS } from "./settings/SettingsSidebar";
import GeneralSection from "./settings/sections/GeneralSection";
import CaptureSection from "./settings/sections/CaptureSection";
import SourcesSection from "./settings/sections/SourcesSection";
import AgentsSection from "./settings/sections/AgentsSection";
import IntelligenceSection from "./settings/sections/IntelligenceSection";
import DiagnosticsSection from "./settings/sections/DiagnosticsSection";

interface SettingsPageProps {
  /** Which group to display. Driven by the Settings sidebar in Main.tsx. */
  section?: SettingsSection;
  onBack: () => void;
  onSetupAgent?: () => void;
  onImport?: () => void;
}

export default function SettingsPage({
  section = "general",
  onBack,
  onSetupAgent,
  onImport,
}: SettingsPageProps) {
  const { t } = useTranslation();
  const activeGroup = SETTINGS_GROUPS.find((g) => g.id === section);
  return (
    <div className="flex flex-col gap-6 max-w-2xl mx-auto py-4">
      {/* [KEEP verbatim: back button + heading block, original lines 439-449] */}
      {section === "general" && <GeneralSection />}
      {section === "capture" && <CaptureSection />}
      {section === "sources" && <SourcesSection onImport={onImport} />}
      {section === "agents" && <AgentsSection onSetupAgent={onSetupAgent} />}
      {section === "intelligence" && <IntelligenceSection delay={0} />}
      {section === "diagnostics" && <DiagnosticsSection />}
      {/* [KEEP verbatim: privacy-note footer, original lines 1070-1080] */}
    </div>
  );
}
```

- [ ] **Step 4: Verify size + full gates**

Run: `wc -l src/components/memory/SettingsPage.tsx` → expect < 150.
Run: `pnpm build && pnpm test && pnpm test:i18n`
Expected: all PASS (`pnpm build` = `tsc -b && vite build`).

- [ ] **Step 5: Commit**

```bash
git add -A src/components/memory
git commit -m "refactor(settings): SettingsPage becomes a thin section switch"
```

---

### Task 7: Push + draft PR

- [ ] **Step 1: Push** — `git push -u origin settings-decomposition` (run `gh`/network commands with the sandbox disabled per machine rules).

- [ ] **Step 2: Draft PR**

```bash
gh pr create --draft --base main --head settings-decomposition \
  --title "refactor(settings): decompose SettingsPage into per-section modules" \
  --body "<summary: behavior-preserving split per spec §4; zero copy/i18n/logic changes; existing tests unmodified. Link the spec file. Include the standard generated-with footer.>"
```

Expected: PR URL printed. Do NOT merge; the redesign feature branch (PR 2) rebases onto this branch.
