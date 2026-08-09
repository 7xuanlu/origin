# Onboarding + Settings — one design system

**Date:** 2026-07-12 · **Branch:** `redesign-onboarding-settings`
**Input:** `docs/superpowers/reviews/2026-07-12-onboarding-settings-design-critique.md` (Tier 2)
**Scope:** SetupWizard (all 6 steps), all settings sections, `src/components/intelligence/*`, and the shared components those surfaces render (`connect/*`, `ChatImport/*`, `sources/VaultConnectCard`, `RemoteAccessPanel`, `AddMemoryForm`, `ActivityFeed`'s select).

The user's mandate: "make onboarding and setting both state of the art modern. now it's
messy and inconsistent." The mess is not taste — it is the same element speaking three
dialects on one screen. This spec picks ONE dialect per element and lists every deviation
by file and line. It is implementable without having read the critique or seen the app:
every rule names its tokens, every work item names its lines.

Binding constraints (violating any voids the change):

- `--mem-*` tokens only. No raw hex, no px outside the scales in `src/index.css:128–326`.
- No new dependencies.
- All user-visible copy through `src/i18n/resources.ts`, exact key parity en / zh-Hans / zh-Hant.
- `src/i18n/hardcodedCopyBaseline.tsv` only shrinks (443 rows today; this spec removes ≥26).
- Chip-never-lies is preserved **at the type level** (§1.1, §2.2).
- Do not touch `settings.footer`, `settings.localOnly`, `setup.privacyBody` — that copy is
  being rewritten under separate sign-off. Every layout below keeps the footer slot where it is.
- No copy is reworded by this spec. Every i18n migration in §4 moves the literal string as-is.

---

## 1. THE RULES

### R1 — Chips: two kinds, two grammars, zero overlap

A pill is either a **probe result** or a **static label**. Nothing else may be a pill.

| | Probe result | Static label |
|---|---|---|
| Component | `StatusChip` (exists, unchanged) | `Tag` (new, §2.2) |
| Answers | "what did the app just observe?" | "what kind of thing is this?" |
| Face | `--mem-font-mono`, `--mem-text-2xs`, UPPERCASE, tracking `--mem-tracking-caps` | `--mem-font-body`, `--mem-text-xs`, Sentence case, no tracking |
| Leading dot | always (state-colored, `aria-hidden`) | never |
| Color source | `ProbeState` only (`primitives.tsx:339` — "the ONLY color input") | closed `TagTone` union: `neutral \| accent` |
| Palette | status triplets (success/danger) + tertiary | `--mem-text-secondary`/`--mem-border` or `--mem-indigo-bg`/`--mem-accent-indigo` |
| `aria-live` | `polite` (it changes at runtime) | none (it never changes) |
| Geometry | shared: `inline-flex items-center rounded-full border py-[2px] px-[8px]`, line-height 1.2 | same |

**How to judge:** does the text report something the app *checked* (probe, filesystem
detection, permission, tunnel status)? → `StatusChip`, state from the check. Does it
*classify or recommend* (trust level, "Recommended", content type)? → `Tag`. A `Tag` can
never render green or red — the tone union has no such member — so a static label cannot
impersonate a probe, and `StatusChip` accepts no free color, so a probe cannot be faked.
That is the type-level enforcement; §2.2 adds `@ts-expect-error` pins.

**Corollary — identifiers are never pills.** A machine string (`claude-code`,
`agent_type`, a config path) is mono *text* (tier T7, §R3), not a chip. Pills carry human
vocabulary only.

Disposition of every pill in scope:

| Today | Where | Becomes |
|---|---|---|
| `CONFIGURED` mono chip | `SetupWizard.tsx:655–657`, `AgentsSection.tsx:153` | `StatusChip` — already correct, stays |
| `Detected` / `Install first` sans pill | `SetupWizard.tsx:645–654` | **deleted.** The label is per-*list*, not per-row (`SetupWizard.tsx:876` stamps one string on every row); the group's `SectionLabel` directly above already says it. Pure duplication. Keys `setup.connect.detected` / `installFirst` (`resources.ts:48–49`) become orphans → remove in all three locales |
| `Recommended` raw-styled pill | `SetupWizard.tsx:325–337` | `Tag tone="accent"` (key `setup.intelligence.recommended` unchanged) |
| Trust legend badges (mono, per-level accent) | `AgentsSection.tsx:86–113` | `Tag tone="neutral"`; label from existing `settings.agents.trust.*`, summary from new `settings.agents.trustSummary.*` (§4.1). Per-level accent color drops — it appears nowhere else after R2 removes the colored select, so it carried no durable meaning |
| `agent_type` mono chip | `AgentsSection.tsx:183–185` | folded into the row's mono subtitle line (`id · type`) per the corollary |
| Done-step type-breakdown `color-mix` pills + agent pills + `+N` overflow | `SetupWizard.tsx` DoneStep | `Tag tone="neutral"` (counts), `Tag tone="accent"` (agent names), `Tag tone="neutral"` (overflow). The `color-mix` per-type colors go — they are the only place that palette exists |
| `CONNECTED TO OLLAMA — 0 MODELS` | `AnyProviderCard.tsx` | stays `StatusChip` `up` — the probe really succeeded; the tone-of-label question is a separate open thread (§6) |

### R2 — Selects: one primitive, styled-native

Every `<select>` in scope becomes the new `Select` primitive (§2.1):
`GeneralSection.tsx:300`, `AgentsSection.tsx:232`, `IntelligenceSetup.tsx:209, 227, 340`,
`AnyProviderCard.tsx:320` (delete the local `selectClassName` at `:56`),
`AddMemoryForm.tsx:70`, `ActivityFeed.tsx:352`.

**Decision: styled native (`appearance: none` + custom chevron), not a hand-rolled
listbox** — with no headless-UI dependency allowed, a listbox means reimplementing
type-ahead, Home/End, wheel, and VoiceOver item enumeration from scratch for zero product
win; the native control keeps the OS keyboard/screen-reader contract for free, and the
only un-themed part is the transient open popup, which macOS renders acceptably.
Precedent already in-tree: `AgentsSection.tsx:232` does `appearance:none` + chevron ad hoc,
and `AnyProviderCard.tsx:56` hand-copies `Input`'s class string — the primitive just makes
the existing best answer the only answer.

The trust-level select (`AgentsSection.tsx:232`) loses its per-level accent border/text
and becomes a standard `Select size="sm"` — one select appearance everywhere; the level's
meaning lives in the legend (R1).

### R3 — Typography: one ladder, three faces with fixed jobs

**The face rule:** Fraunces (serif, `--mem-font-heading`) is the *display voice* — it may
appear only at `--mem-text-xl` and above (page/step titles, stat numerals), exactly as the
scale itself declares ("where Fraunces starts earning its keep", `index.css:161–173`).
Instrument Sans (`--mem-font-body`) carries every operable surface below the display band.
JetBrains Mono (`--mem-font-mono`) is reserved for what a machine said or will read —
commands, ids, endpoints, probe chips, eyebrows. One glance tells you whether you are
reading a destination, a control, or telemetry.

| Tier | Token | Face / weight / case | Color | Used for |
|---|---|---|---|---|
| T1 step title | `--mem-text-2xl` (26) | serif 500 | `--mem-text` | every wizard step `h1` |
| T1 page title | `--mem-text-xl` (20) | serif 500 | `--mem-text` | settings page `h1`; stat numerals (add `font-variant-numeric: tabular-nums`) |
| T2 eyebrow | `--mem-text-2xs` (10) | mono 500 UPPERCASE, tracking `--mem-tracking-eyebrow` | `--mem-text-tertiary` | **`SectionHeader` is the only section label in the product** (settings groups, wizard connect groups, import groups) |
| T3 card title | `--mem-text-lg` (16) | sans 600 | `--mem-text` | every `Card`/panel heading — per the token's own comment, `index.css:171` |
| T4 row title | `--mem-text-md` (14) | sans 500 | `--mem-text` | `SettingRow` titles, client-row names (`SetupWizard.tsx:638` already correct) |
| T5 control text | `--mem-text-base` (13) | sans 400–500 | `--mem-text` | inputs, buttons, selects, step descriptions |
| T5 description | `--mem-text-sm` (12) | sans 400, lh 1.5 | `--mem-text-secondary` | card/row descriptions |
| T6 caption | `--mem-text-xs` (11) | sans 400 | `--mem-text-tertiary` | metadata, hints |
| T7 machine | `--mem-text-sm` blocks / `--mem-text-xs` inline | mono | `--mem-text` / secondary | command blocks (sm), inline ids & paths (xs), chip/eyebrow text (2xs caps) |

**Card titles are sans, not serif.** The serif card titles (`Claude.ai`,
`WebPlatformCards.tsx:107`; `Chat history`, `SetupWizard.tsx:441`; vault card,
`VaultConnectCard.tsx:86` — all raw 15px, a size that exists in no scale) read as content
headlines on what are actually control panels. The WHY of the serif is destination-marking;
a card you operate is not a destination. All become T3.

**Section labels:** the wizard's private `SectionLabel` (sans-caps 11px/0.06em,
`SetupWizard.tsx:472–488`) is deleted; its four call sites (`:836, :875, :881, :887`) use
`SectionHeader` (icon now optional, §2.3). Diagnostics' hand-rolled header
(`DiagnosticsSection.tsx:122`) becomes `SectionHeader` with an icon like its siblings and
the Refresh button (`:125–127`) in the new `action` slot. The import step's outer serif
heading `setup.import.vaultPathTitle` (`SetupWizard.tsx:458–460`) is deleted —
`VaultConnectCard` already carries its own title; the key (`resources.ts:38, 1065, 2065`)
is removed in all three locales.

Known off-scale values this tier sweep eliminates: welcome title 28px
(`SetupWizard.tsx:178`), import title raw `"20px"` (`:428`), settings `h1` raw `"20px"`
(`SettingsPage.tsx:38`), card titles 15px (above), choice-button radius `10px`
(`SetupWizard.tsx:251` → `--mem-radius-md`), `SettingRow` raw `"14px"/"12px"`
(`primitives.tsx:59, 62` → `--mem-text-md`/`--mem-text-sm`).

### R4 — Buttons: one primary per screen; the bar owns the wizard's

- **R4.1** A screen (one wizard step; one settings section) renders at most ONE
  `variant="primary"` Button. In the wizard the `StepShell` action bar owns it
  (`primaryAction`, e.g. `SetupWizard.tsx:290, 424`); **step content never renders
  primary.**
- **R4.2** Primary marks the action that *commits state* (Save, Connect, an import that
  executes). Actions that open/reveal/copy (file pickers, dialogs, "Copy URL") are
  `secondary`; escape hatches and navigation are `ghost`; destructive is `danger` behind a
  confirm.
- **R4.3** Everything that looks like a button IS the `Button` primitive. Raw
  `<button style={{backgroundColor: var(--mem-accent-indigo), color: "white"}}>` is banned
  outright — `"white"` on indigo is the documented 2.10:1 dark-theme contrast failure;
  `Button` primary already uses `--mem-text-on-accent` (`primitives.tsx:132`). Offenders:
  `SetupWizard.tsx:448–454` ("Import chat history" → `secondary`; it opens the chat-import
  flow, and the bar owns Continue), `ClientSetupList.tsx:87` ("Set up" → `secondary`, wizard
  content), `DropZone.tsx:129–138` ("Choose file" → `secondary`, it opens a picker),
  `WebPlatformCards.tsx:80, 147–149` (copy/open → `ghost size="sm"` / `secondary size="sm"`),
  `VaultConnectCard.tsx` Browse → `secondary`, Connect →
  `variant={variant === "wizard" ? "secondary" : "primary"}` (commits state, but R4.1 wins
  inside the wizard).
- **R4.4** The escape hatch "Or write the config for me" becomes a real control: `Button
  variant="ghost" size="sm"` toggling the manual-setup list. The existing key
  `connectMatrix.oneClickAdvanced` (`resources.ts:999`) serves both call sites unchanged —
  once both are buttons, the one-key-two-jobs problem dissolves.
- The "Import Memories" card gets its control back: `onImport` becomes **required**
  (`SourcesSection.tsx:7`), the `{onImport && …}` guard (`:26–29`) is removed, and the
  button is `secondary` (it opens the import dialog; the commit happens inside).
- The intelligence choice tiles (`SetupWizard.tsx:320–341`) are toggles and must carry
  `aria-pressed={mode === …}` (a11y floor; they currently signal selection by color only).

### R5 — Cards: one card per subsystem

Under a `SectionHeader`, settings that configure the **same subsystem** (same daemon
config family / same query-invalidation domain) share ONE `Card padding="rows"`, one
`SettingRow` each, separated by the Card's own dividers. A control gets a private card
only when it is its subsystem's sole control. Never one-card-per-toggle.

Application: General's APP group currently splits launch-at-login / theme / language into
three cards (`GeneralSection.tsx:236, 246, 286`) → ONE `Card padding="rows"` with three
rows. Theme and language are not toggles, so `SettingRow` grows a `control` slot (§2.4):
the segmented theme control and the language `Select` render as the row's right-hand
control. Profile keeps its own card (own header, `GeneralSection.tsx:131, 140`).

**CaptureSection: delete, don't restore.** The sidebar comment records the product
decision — "ambient capture is disabled as part of the memory-layer pivot"
(`SettingsSidebar.tsx:109–110`) — and the section is reachable only by forcing the route.
Unreachable UI cannot be reviewed, and renovating it (raw card div `CaptureSection.tsx:59`,
hand-rolled divider `:75`, duplicate toggle row `:89`, raw 10px "Grant access" button
`:120`) would be unreviewable work. Remove: `CaptureSection.tsx`, the `capture` branch in
`SettingsPage.tsx`, `"capture"` from the section-id type, the commented sidebar block
(`SettingsSidebar.tsx:109–121`), and `settings.capture.*` keys in all three locales. Git
history preserves it for a real re-enable.

---

## 2. NEW / CHANGED PRIMITIVES (`src/components/memory/settings/primitives.tsx`)

### 2.1 `Select` — new

```tsx
export interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "size"> {
  size?: "sm" | "md"; // md (default): h-32px, radius-md — Input/Button parity
                      // sm: h-26px, radius-sm — Button sm parity
  mono?: boolean;     // model ids, machine values (mirrors Input.mono)
  invalid?: boolean;  // aria-invalid + danger border (mirrors Input.invalid)
}
```

`Omit<…, "size">` because the native `size` attribute (visible-row count) collides with
the variant prop; nothing in scope uses the native meaning.

Rendering: a `relative inline-flex w-full` wrapper `<span>`; inside it the native
`<select>` carrying `Input`'s exact class recipe (`primitives.tsx:309–315`) plus
`appearance-none cursor-pointer w-full pr-[28px]` (sm: `pr-[24px]`), and a chevron:

```tsx
<span aria-hidden="true" className="pointer-events-none absolute right-[10px] top-1/2
  -translate-y-1/2 text-[var(--mem-text-tertiary)]">
  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
    <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5"
      strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
</span>
```

Sizes: md `h-[32px] pl-[10px] rounded-[var(--mem-radius-md)]` text `--mem-text-base`;
sm `h-[26px] pl-[8px] rounded-[var(--mem-radius-sm)]` text `--mem-text-sm`. `mono` swaps
font to `--mem-font-mono` at `--mem-text-sm` (Input parity, `primitives.tsx:321–322`).
`className`/`style`/`id`/`aria-*` spread onto the native `<select>` (not the wrapper), so
`Field`'s `cloneElement` id + `aria-describedby` wiring works unchanged.

**Deliberately absent:** no `options` prop (children `<option>` stay JSX — call sites keep
their option-building logic); no `label` prop (labeling is `Field`'s job); no open-state
styling hooks (the popup is the OS's — that is the point of the decision in R2).

### 2.2 `Tag` — new (the static-label chip)

```tsx
export type TagTone = "neutral" | "accent";

export interface TagProps {
  tone?: TagTone; // default "neutral"
  children: React.ReactNode;
}
```

Classes: `inline-flex items-center rounded-full border py-[2px] px-[8px]` (StatusChip's
geometry, `primitives.tsx:393`, so the two align on a shared row). Style: `--mem-font-body`,
`--mem-text-xs`, weight 500, line-height 1.2, no `text-transform`. Tones:

- `neutral`: `bg-transparent text-[var(--mem-text-secondary)] border-[var(--mem-border)]`
- `accent`: `bg-[var(--mem-indigo-bg)] text-[var(--mem-accent-indigo)] border-transparent`

Both pairings are pre-existing token pairs already carrying text at these sizes
(`SetupWizard.tsx:649–650, 253–254`) — no new contrast surface.

**Deliberately absent, and why they are the invariant:** no `state` (a Tag can never
consume probe plumbing), no free `color`/`intent`/`variant` beyond the two-member union
(no member maps to the success/danger triplets, so a Tag *cannot render* the probe
palette), no dot slot, no `aria-live`. Pin it in `primitives.test.tsx`:

```tsx
// @ts-expect-error — Tag has no probe vocabulary
<Tag tone="up">Configured</Tag>;
// @ts-expect-error — StatusChip color cannot be faked with a string
<StatusChip state="recommended" label="x" />;
```

`StatusChip` itself is **unchanged** — API, tones, tests all stand. The only edits are
mechanical: its raw `letterSpacing: "0.08em"` (`primitives.tsx:403`) reads the new
`--mem-tracking-caps` token.

### 2.3 `SectionHeader` — changed

```tsx
export function SectionHeader({ icon, label, action }: {
  icon?: React.ReactNode;    // was required (primitives.tsx:96); wizard groups have none
  label: string;
  action?: React.ReactNode;  // right-aligned slot — e.g. Diagnostics' Refresh button
})
```

Layout gains `justify-between`: `[icon? label] …spacer… [action?]`. Raw
`letterSpacing: "0.14em"` (`primitives.tsx:108`) reads `--mem-tracking-eyebrow`.
Existing call sites compile unchanged.

### 2.4 `SettingRow` — changed (discriminated union)

```tsx
type SettingRowProps = {
  title: string;
  description: string;
  statusLine?: React.ReactNode;
  warning?: string | null;
  error?: string | null;
} & (
  | { enabled: boolean; onToggle: () => void; control?: never } // toggle row (today's API)
  | { control: React.ReactNode; enabled?: never; onToggle?: never } // custom-control row
);
```

The toggle arm keeps today's exact rendering and aria-describedby wiring
(`primitives.tsx:49–68`); the `control` arm renders the given node in the Toggle's slot
(theme segmented control, language `Select`). Raw `"14px"/"12px"` (`:59, 62`) become
`var(--mem-text-md)`/`var(--mem-text-sm)`. Existing call sites compile unchanged.

**Deliberately unchanged primitives:** `StatusChip` (§2.2), `Toggle`, `Button`, `Card`,
`Field`, `Input` — their contracts are pinned by `primitives.test.tsx` and nothing in the
five problems requires touching them. `RemoteAccessPanel`'s hand-rolled `role="switch"`
duplicate is replaced BY `Toggle`, not the other way around: one toggle, one ARIA idiom
(`aria-pressed`).

---

## 3. NEW TOKENS (`src/index.css`)

| Token | Light | Dark | Notes |
|---|---|---|---|
| `--mem-tracking-caps` | `0.08em` | same | uppercase chip text (StatusChip, today raw at `primitives.tsx:403`) |
| `--mem-tracking-eyebrow` | `0.14em` | same | SectionHeader eyebrows (today raw at `primitives.tsx:108`) |

**Zero new color tokens.** Every color this spec assigns already exists; carriers of text
keep their already-documented ratios (`--mem-text-on-accent`: 8.8:1 dark, 5.7:1 light, per
the token block's own comments). Raw values eliminated by this spec: `#ef4444`
(`ImportFlow.tsx:130, 134, 148, 202–203`, `DropZone.tsx:147`, `RemoteAccessPanel.tsx:294,
459`) → `--mem-status-danger-text` / danger triplet; `bg-red-500`
(`RemoteAccessPanel.tsx:454`) → `--mem-status-danger-text` dot;
`rgba(251,191,36,…)` / `rgba(123,123,232,…)` (`RemoteAccessPanel.tsx:~212–214`) →
`--mem-status-warning-bg` / `--mem-indigo-bg`; `color: "white"` on indigo (R4.3 list) →
`Button` primary's `--mem-text-on-accent`.

---

## 4. i18n — new, moved, deleted keys

All migrations are literal — the English value is the string currently hardcoded,
character-for-character. Ratchet effect: −5 rows (AddMemoryForm, baseline rows 20–24),
−21 rows (RemoteAccessPanel, rows 244–264), plus any `CaptureSection` rows deleted with
the file. Regenerate the baseline after each slice; it may only shrink.

### 4.1 Trust levels — out of `src/lib/agents.ts:113–137`, into resources

The guard only scans `src/components/**/*.tsx` (`hardcodedCopyGuard.test.ts:7`), so these
English strings ship untranslated in zh today. `TRUST_LEVELS` shrinks to
`Record<TrustLevel, { accent: string }>`; labels stay on the existing
`settings.agents.trust.*` keys; summaries move to new keys. The `detail` field renders
nowhere (grep: only unrelated `item.detail` in ActivityFeed) — delete it, do not translate it.

| Key (`settings.agents.`) | en | zh-Hans | zh-Hant |
|---|---|---|---|
| `trustSummary.full` | Sees everything: identity, preferences, lessons, gotchas, decisions, pages, and search results. | 可见全部内容:身份、偏好、经验、注意事项、决策、页面与搜索结果。 | 可見全部內容:身份、偏好、經驗、注意事項、決策、頁面與搜尋結果。 |
| `trustSummary.review` | Sees lessons, gotchas, decisions, corrections, pages, and search — but not identity or preferences. | 可见经验、注意事项、决策、更正、页面与搜索——但不含身份与偏好。 | 可見經驗、注意事項、決策、更正、頁面與搜尋——但不含身份與偏好。 |
| `trustSummary.unknown` | Sees search results only. No identity or preferences. | 仅可见搜索结果。不含身份与偏好。 | 僅可見搜尋結果。不含身份與偏好。 |

### 4.2 `remoteAccess.*` — RemoteAccessPanel, currently 100% hardcoded

The no-auth boundary sentence is recently shipped trust-surface copy (commit `3a272d0`) —
keyed verbatim, translated faithfully, not reworded. The two third-party menu paths are
quoted UI: identical value in all three locales.

| Key (`remoteAccess.`) | en (verbatim from file) | zh-Hans | zh-Hant |
|---|---|---|---|
| `title` | Share with web-based AI tools | 与网页版 AI 工具共享 | 與網頁版 AI 工具共享 |
| `noAuthWarning` | Creates a public HTTPS URL with no authentication for Claude.ai and ChatGPT. Anyone with the URL can access Wenlan; turn Remote Access off when unused. | 将为 Claude.ai 与 ChatGPT 创建一个无需身份验证的公开 HTTPS 地址。任何拥有该地址的人都能访问 Wenlan;不使用时请关闭远程访问。 | 將為 Claude.ai 與 ChatGPT 建立一個無需身份驗證的公開 HTTPS 位址。任何擁有該位址的人都能存取 Wenlan;不使用時請關閉遠端存取。 |
| `urlLabel` | Your MCP URL | 你的 MCP 地址 | 你的 MCP 位址 |
| `urlLabelStable` | Your MCP URL (stable) | 你的 MCP 地址(稳定) | 你的 MCP 位址(穩定) |
| `copyUrl` | Copy URL | 复制地址 | 複製位址 |
| `copied` | Copied! | 已复制! | 已複製! |
| `testConnection` | Test connection | 测试连接 | 測試連線 |
| `testing` | Testing… | 正在测试… | 正在測試… |
| `reconnect` | Reconnect | 重新连接 | 重新連線 |
| `retry` | Retry | 重试 | 重試 |
| `statusOff` | Off | 关闭 | 關閉 |
| `statusConnecting` | Connecting… | 正在连接… | 正在連線… |
| `statusConnected` | Connected | 已连接 | 已連線 |
| `statusConnectedLatency` | Connected ({{ms}} ms) | 已连接({{ms}} 毫秒) | 已連線({{ms}} 毫秒) |
| `howTo` | How to connect Claude.ai and ChatGPT | 如何连接 Claude.ai 与 ChatGPT | 如何連接 Claude.ai 與 ChatGPT |
| `claudeAi` | Claude.ai | Claude.ai | Claude.ai |
| `chatGpt` | ChatGPT | ChatGPT | ChatGPT |
| `claudeSteps` | Settings → Connectors → Add Custom Connector → Paste URL | *(same)* | *(same)* |
| `chatgptSteps` | Settings → Apps → Advanced settings → Enable Developer mode → Back → Create app → Paste URL (No Auth) | *(same)* | *(same)* |
| `tunnelChangesNote` | This tunnel URL changes when your Mac sleeps or restarts. Enable a stable relay in Settings → Agents to avoid reconnecting. | 此隧道地址会在 Mac 休眠或重启后变化。可在“设置 → Agents”中启用稳定中继,免去重新连接。 | 此隧道位址會在 Mac 休眠或重啟後變化。可在「設定 → Agents」中啟用穩定中繼,免去重新連線。 |
| `stableNote` | This URL is stable — it won't change when your Mac sleeps or restarts. | 此地址是稳定的——不会在 Mac 休眠或重启后变化。 | 此位址是穩定的——不會在 Mac 休眠或重啟後變化。 |

The implementer keys every remaining literal in the file the same way (the baseline's 21
rows for this file are the checklist); the guard fails the build on any straggler.

### 4.3 `addMemory.*` — AddMemoryForm

| Key (`addMemory.`) | en | zh-Hans | zh-Hant |
|---|---|---|---|
| `placeholder` | What do you want to remember? | 你想记住什么? | 你想記住什麼? |
| `spaceLabel` | Space: | 空间: | 空間: |
| `noSpace` | No space | 无空间 | 無空間 |
| `cancel` | Cancel | 取消 | 取消 |
| `save` | Save | 保存 | 儲存 |

### 4.4 Deleted keys (all three locales, same commit as their last consumer)

- `setup.connect.detected`, `setup.connect.installFirst` (`resources.ts:48–49, 1075–1076, 2075–2076`) — pill deleted in R1.
- `setup.import.vaultPathTitle` (`resources.ts:38, 1065, 2065`) — heading deleted in R3.
- `settings.capture.*` — section deleted in R5.

No other key changes. `connectMatrix.oneClickAdvanced` is reused as-is (R4.4);
`setup.intelligence.recommended`, `setup.connect.configured`, `settings.agents.trust.*`,
`settings.sources.import` all keep their jobs.

---

## 5. WORK LIST — eight independently shippable slices

“M” = mechanical (this spec fully determines the diff); “J” = judgment (implementer
classifies against a rule). Every slice leaves tests + ratchet green on its own.

### S1 — Foundation (primitives + tokens) — blocks all others

| File | Change | Rule |
|---|---|---|
| `src/index.css` | add `--mem-tracking-caps: 0.08em`, `--mem-tracking-eyebrow: 0.14em` | §3 · M |
| `src/components/memory/settings/primitives.tsx` | add `Select` (§2.1), `Tag` (§2.2); `SectionHeader` icon optional + `action` slot (§2.3); `SettingRow` control union (§2.4); swap raw `letterSpacing`/`14px`/`12px` for tokens (`:403, :108, :59, :62`) | R1–R3, R5 · M |
| `src/components/memory/settings/primitives.test.tsx` | pin: Select native semantics + Field wiring + chevron `aria-hidden`; Tag tones + `@ts-expect-error` invariant pair; SettingRow control arm | §2 · M |

### S2 — Select sweep

| File | Change | Rule |
|---|---|---|
| `GeneralSection.tsx:300` | language picker → `Select` (as `SettingRow` control after S6, standalone until then) | R2 · M |
| `AgentsSection.tsx:232` | trust picker → `Select size="sm"`; drop per-level border/text color | R2 · M |
| `IntelligenceSetup.tsx:209, 227, 340` | → `Select` (`mono` where options are model ids) | R2 · M |
| `AnyProviderCard.tsx:320` | → `Select`; delete `selectClassName` (`:56`) | R2 · M |
| `AddMemoryForm.tsx:70` | → `Select size="sm"` | R2 · M |
| `ActivityFeed.tsx:352` | FilterSelect → `Select size="sm"` (keep its `aria-label`) | R2 · M |

### S3 — Chip vocabulary

| File | Change | Rule |
|---|---|---|
| `SetupWizard.tsx:645–654` | delete the group pill | R1 · M |
| `src/i18n/resources.ts` | delete `setup.connect.detected` / `installFirst` ×3 locales | §4.4 · M |
| `SetupWizard.tsx:325–337` | Recommended → `Tag tone="accent"` | R1 · M |
| `SetupWizard.tsx` DoneStep pills | → `Tag` per R1 disposition table | R1 · J (tone choice per pill) |
| `AgentsSection.tsx:86–113` | legend badges → `Tag tone="neutral"`; summaries via `t()` | R1, §4.1 · M |
| `AgentsSection.tsx:183–185` | agent_type pill → mono subtitle (`id · type`) | R1 corollary · M |
| `src/lib/agents.ts:113–137` | `TRUST_LEVELS` → `{ accent }` only; delete `detail`; `describeTrustLevel` returns accent + level | §4.1 · M |
| `src/i18n/resources.ts` | add `settings.agents.trustSummary.*` ×3 locales | §4.1 · M |

### S4 — Typography ladder

| File | Change | Rule |
|---|---|---|
| `SetupWizard.tsx:178, 300, 428` + verify/done titles | all step `h1` → T1 2xl serif | R3 · M |
| `SetupWizard.tsx:441` | chat card title serif 15px → T3 | R3 · M |
| `SetupWizard.tsx:458–460` | delete outer vault heading; drop `vaultPathTitle` key ×3 | R3, §4.4 · M |
| `SetupWizard.tsx:472–488, 836, 875, 881, 887` | delete `SectionLabel`; use `SectionHeader` (no icon) | R3 · M |
| `SetupWizard.tsx:251` | choice-tile radius 10px → `--mem-radius-md`; add `aria-pressed` (`:320–341`) | R3, R4 · M |
| `SettingsPage.tsx:38` | `h1` raw 20px → `var(--mem-text-xl)` | R3 · M |
| `WebPlatformCards.tsx:107`, `VaultConnectCard.tsx:86` | serif 15px titles → T3 | R3 · M |
| `IntelligenceSetup.tsx` (ApiKeyCard / OnDeviceModelCard headings) | classify each heading by role: card title → T3, field label → T5; e.g. `:206, :224` are field labels | R3 · J |
| `DiagnosticsSection.tsx:122, 125–127` | header → `SectionHeader` (icon matching siblings, Refresh in `action`); stat numerals stay T1-xl (`:64, :143` already correct) | R3 · M |
| `SetupWizard.tsx` DoneStep `Stat` | numeral raw 20px → `var(--mem-text-xl)` + tabular-nums | R3 · M |

### S5 — Button hierarchy

| File | Change | Rule |
|---|---|---|
| `SetupWizard.tsx:448–454` | raw indigo CTA → `Button variant="secondary"` | R4.1/4.3 · M |
| `SetupWizard.tsx` connect step | escape hatch caption → `Button variant="ghost" size="sm"`, toggles manual list; key reused | R4.4 · M |
| `ClientSetupList.tsx:87` | raw "Set up" → `Button variant="secondary"` | R4.3 · M |
| `DropZone.tsx:129–138, 147` | raw CTA → `Button variant="secondary"`; `#ef4444` → danger tokens | R4.3, §3 · M |
| `ImportFlow.tsx:130, 134, 148, 202–203` | `#ef4444` → `--mem-status-danger-*` | §3 · M |
| `WebPlatformCards.tsx:80, 147–149` | raw buttons → `Button ghost/secondary size="sm"` | R4.3 · M |
| `VaultConnectCard.tsx` | Browse → secondary; Connect → primary in settings, secondary in wizard variant | R4.1/4.2 · J |
| `SourcesSection.tsx:7, 26–29` | `onImport` required; button unconditional, `secondary`; audit the section for exactly ≤1 primary | R4.2 · J |

### S6 — Card grouping (Capture removal CUT — see override)

> **COORDINATOR OVERRIDE — do not delete `CaptureSection`.** The spec's argument
> (unreachable UI can't be reviewed) is sound, but `SettingsSidebar.tsx:109` records an
> explicit intent to bring it back: *"Capture is hidden for now — ambient capture is
> disabled as part of the memory-layer pivot. Re-enable when capture features are ready
> for users."* Deleting the component, its toggles, its config-boundary test, and the
> Rust commands behind them would reverse a documented product decision on an agent's
> initiative. Whether ambient capture returns is the user's call, not ours, and the cost
> of leaving it hidden is zero. It stays until they say otherwise. Escalated to the user.

| File | Change | Rule |
|---|---|---|
| `GeneralSection.tsx:236, 246, 286` | merge three APP cards → one `Card padding="rows"`; theme + language become `SettingRow control={…}` rows | R5 · M |
| ~~`CaptureSection.tsx`~~ | ~~delete file~~ — **cut, see override** | — |
| ~~`SettingsPage.tsx`~~ | ~~remove capture branch~~ — **cut** | — |
| ~~`SettingsSidebar.tsx:109–121`~~ | ~~remove commented block~~ — **cut** | — |
| ~~`src/i18n/resources.ts`~~ | ~~delete `settings.capture.*`~~ — **cut** | — |

### S7 — RemoteAccessPanel conformance (`src/components/memory/RemoteAccessPanel.tsx`)

| Change | Rule |
|---|---|
| hand-rolled `role="switch"` → `Toggle` primitive (`aria-pressed`) | §2.4 note · M |
| all raw buttons → `Button` (`Copy URL`/`Test connection` secondary sm; `Retry`/`Reconnect` secondary sm) | R4.3 · M |
| StatusRow dot+text → `StatusChip`: daemon tunnel status IS a probe — `connected` → `up` (detail = `{{ms}} ms`), `connecting` → `probing`, `error` → `down` (verbatim daemon error as detail), `off` → no chip (the Toggle already says off; an idle chip would be noise) | R1 · J |
| raw colors (`:212–214, 294, 454, 459`) → tokens per §3 | §3 · M |
| all copy → `remoteAccess.*` (§4.2); −21 baseline rows | §4.2 · M |
| panel title → T3; instruction steps body → T5; URL value mono T7 | R3 · M |

### S8 — AddMemoryForm i18n

| Change | Rule |
|---|---|
| 5 literals (`:54, 68, 80, 95, 108`) → `addMemory.*` (§4.3); −5 baseline rows | §4.3 · M |

Dependency shape: S1 → everything; S2–S8 are mutually independent and land in any order.

---

## 6. WHAT I DELIBERATELY DID NOT CHANGE

- **The privacy footer** (`settings.footer`, `settings.localOnly`, `setup.privacyBody`) —
  false copy, being rewritten under the user's personal sign-off. Every layout here keeps
  the footer slot exactly where it is so the rewrite drops in without layout work.
- **`StatusChip`'s API and tests** — the chip-never-lies anchor. `Tag` was designed around
  it, not instead of it.
- **Wizard flow**: the verify/done double-ending, the ~500px dead space on
  welcome/verify/done, and the on-device-default vs cloud-"Recommended" contradiction
  (`SetupWizard.tsx:246` vs `:336`) are product/composition calls (critique Tier 3), not
  vocabulary — they belong to team-lead, and none of the five rules turns on them.
- **The Ollama `0 MODELS` success-tone label** — already fixed on this branch (`362f50e`,
  "stop reading a zero-model local server as success"); nothing left for this spec to rule on.
- **`.memory-chip` on the memory pages** (`src/index.css`) — a different surface's chip
  grammar; unifying it with `Tag` is a follow-up once this vocabulary has landed.
- **`normalizeEndpoint` localhost/127.0.0.1 aliasing** — open engineering thread in the
  critique, not a design-system concern.
- **`ImportView` internals and the preview harness** — out of scope by brief; the harness
  is type-check-excluded on purpose (critique, "Gate blind spots").
- **Copy rewording of any kind** — every string this spec touches moves verbatim; the
  memory-first → wiki-pages-first copy pass is a separate, whole-surface edit that should
  not be entangled with a mechanical design-system sweep.
