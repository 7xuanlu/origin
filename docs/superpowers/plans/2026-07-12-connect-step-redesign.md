# Connect Step Redesign — Design Spec

**Date:** 2026-07-12
**Author:** design-connect (design lead)
**Scope:** `ConnectStep` in `src/components/SetupWizard.tsx:466-923`, plus the minimum
Settings-side changes needed to keep the no-auth warning single-sourced.
**Status:** Spec for implementation. No code changed by this document.
**Skills applied:** `frontend-design:frontend-design` (studio-lead framing, restraint,
copy-as-design-material).

---

## 1. TL;DR

The wizard's connect step becomes one thing: **the list of AI tools Wenlan already
found on this Mac, pre-checked, with one button that connects them.** Everything
else — Claude.ai/ChatGPT web connectors, Remote Access, the undetected-tools
catalog — leaves onboarding and lives where it already exists in
Settings → Agents. The no-auth warning renders exactly once, at the Remote
Access toggle (the decision moment), upgraded from 12px tertiary text to a real
warning treatment. The wizard's duplicate client-row renderer is deleted; both
surfaces compose one shared `ClientRow`.

Measured before → target after (5 detected clients):

| Metric | Before | After |
| --- | --- | --- |
| Words before any disclosure | ~370 | ~65 |
| Clickable controls (no scroll) | 18–20 | 11 (3 checkboxes, 2 copy-prompt, 2 advanced, 1 manual disclosure, Back/Skip/CTA) |
| Parallel paths presented | 8 | 2 (checkbox-batch for GUI, copy-prompt for CLI) + 1 quiet manual fallback |
| No-auth warning renderings | 3 | 1 (Settings, at the toggle) |
| Decisions required in the common case | many | 1 (accept the pre-checked defaults, click "Connect N tools") |

---

## 2. Design rationale

### The thesis: detection is the hero

`detect_mcp_clients_cmd` works and is reliable — on a real machine it found
Cursor, Claude Code, Claude Desktop, Gemini CLI, and Codex CLI by reading their
actual config files. The app already *knows* the answer to the question this
screen currently asks the user in 343 words. A screen whose subject is "the
machine already knows" should open with what it knows, not with a menu.

So the signature element of this screen is the **found list + a CTA that counts**:
the primary button reads `Connect 3 tools` and its number tracks the checkboxes
live. The screen's state *is* the copy. No paragraph explains what checking a
box does — the button label does it in two words. This is the one place the
design spends its boldness; everything around it is quiet.

### Why the web path leaves onboarding (question 3, argued)

The Claude.ai/ChatGPT path is cut from the wizard entirely. Not demoted behind a
disclosure — cut, with a one-line pointer to Settings → Agents. Four reasons:

1. **It is rarely used.** Live daemon data: ChatGPT connector last used 69 days
   ago with 9 memories total, vs Claude Code used today with 2,764 memories.
   The local path is where the product's value is; onboarding should optimize
   for it.
2. **It is the only genuine security hazard on the screen.** Remote Access
   exposes a public HTTPS URL with NO authentication (`wenlan-mcp serve
   --no-auth`, confirmed in code); anyone with the URL can read and write the
   entire store. First run is the worst moment to ask a user to make that
   call — they have no mental model of what's in the store yet, and wizard
   momentum ("keep clicking to finish") is exactly the psychology that gets
   dangerous toggles flipped. Moving the decision to Settings means it is made
   deliberately, on a surface with status, test-connection, and reconnect
   affordances around it.
3. **It silently breaks.** The tunnel URL is ephemeral — it changes on
   sleep/restart, so a URL pasted into Claude.ai during onboarding goes dead
   later with no signal. Shipping a path in onboarding that predictably decays
   teaches the new user that Wenlan is flaky.
4. **It costs ~200 words and 8+ controls** — more than half the screen's
   entire reading load — to serve the rare case.

Counter-case considered: a user whose *only* AI usage is web (zero local
tools). For them the wizard shows the empty state, which carries the same
Settings → Agents pointer plus the manual config fallback. This user is rare
for a desktop app whose install story is "your local AI tooling writes into
it," and they lose nothing — the web path is fully available in Settings,
where it is actually better served.

### Why the warning consolidates at the toggle

The warning currently renders three times (`RemoteAccessPanel.tsx:147`,
`WebPlatformCards.tsx:94` ×2). Repetition is not emphasis — three ambient
copies read as boilerplate and train the eye to skip all of them. The exposure
is created at exactly one moment: flipping the Remote Access toggle. The
warning therefore renders exactly once, adjacent to that toggle, and gets a
visual upgrade: today it is 12px `--mem-text-tertiary` body text (i.e. styled
as a *description*), which soft-pedals the sharpest fact in the app. It becomes
a true warning row (amber icon + `--mem-status-warning-text`, the same pattern
`SettingRow`'s `warning` slot already implements). The string itself is not
softened — see §6, the text is kept verbatim.

### Why checkboxes + one batch button, not per-row "Set up" (question 1)

Per-row buttons (the Settings model) cost N+1 clicks for N tools and give the
user N separate decisions. Pre-checked checkboxes + one counting CTA cost one
click and one decision ("do I accept the defaults?"), with per-row opt-out for
the user who doesn't want Wenlan touching a particular tool's config. Settings
keeps per-row buttons — it's a management surface where you act on one tool at
a time; onboarding is a batch surface where you act on all of them at once.
Both compose the same `ClientRow` (§7), so the row anatomy cannot drift again.

The wizard never writes config files without an explicit click: pre-checking is
a default, the write happens only on the CTA press. Auto-writing other apps'
config files on mount would be over the line.

### CLI vs GUI in one list (question 4)

One list, one row anatomy. The split is expressed per-row in the action
affordance only — never as separate sections the user must read in parallel:

- **GUI clients** (config writable programmatically): checkbox, pre-checked
  when `detected && !already_configured`. No description line — the name, the
  chip, and the counting CTA carry the full meaning.
- **CLI clients** (Claude Code, Codex — `isCliPrimaryClient`): **no checkbox**.
  A checkbox that the Continue button doesn't act on is a lie (the current
  code half-admits this by starting them unchecked). Instead: one lead line +
  one primary button, **Copy setup prompt** — the agent-native path where the
  user pastes one prompt into the tool and the tool installs Wenlan itself.
  The raw terminal commands move behind a per-row `Show terminal commands`
  disclosure, and "Or write the config for me" stays as the collapsed Advanced
  action — which, via the shared component, now *genuinely writes the config*
  (fixing the current wizard bug where that button merely toggles the manual
  JSON disclosure, `SetupWizard.tsx:678-685`).

Judgment call, flagged: prompt-primary (vs commands-primary) for CLI rows. I
chose prompt-primary because it is one paste instead of two commands + a
restart incantation, and §9.3 already introduced the affordance. If real usage
shows terminal users prefer raw commands, swapping the button and the
disclosure is a two-line change; the structure doesn't move.

### What gets promoted / demoted / cut (question 2)

| Item | Verdict | Where it goes |
| --- | --- | --- |
| Detected clients list | **Promoted** — the whole screen | Wizard (and Settings, shared rows) |
| CLI copy-setup-prompt | Promoted to the CLI row's single action | Wizard + Settings |
| CLI terminal commands | Demoted behind `Show terminal commands` | Per-row disclosure |
| "Or write the config for me" | Demoted (unchanged position), now truthful | Per-row Advanced |
| Manual JSON snippet | Demoted to one quiet disclosure line at the bottom; auto-expands on detection failure | Wizard fallback |
| "Supported tools" (undetected catalog) | **Cut from wizard** — you cannot connect what isn't installed; it's a catalog, not a task | Settings → Agents (`ClientSetupList` already shows them) |
| Remote Access toggle + URL + test | **Cut from wizard** | Settings → Agents (already there, `AgentsSection.tsx:295`) |
| Claude.ai / ChatGPT connector cards | **Cut from wizard** | Settings → Agents (already there, `AgentsSection.tsx:308`) |
| No-auth warning ×3 | Consolidated to ×1, upgraded treatment | `RemoteAccessPanel`, at the toggle |
| Per-row description sentences (`detectedDescription` etc.) | **Cut entirely** — they narrate what the chip and checkbox already show | Deleted |

---

## 3. Before / after structure

```
BEFORE (343+ words, 18-20 controls)          AFTER (~65 words, 11 controls)

┌─ Choose tools to connect ────────┐         ┌─ Connect your AI tools ──────────┐
│  2-line description              │         │  1-line description              │
│                                  │         │                                  │
│  DETECTED ON YOUR MAC            │         │  FOUND ON THIS MAC               │
│  [✓] Cursor + description        │         │  [✓] Cursor            ● config'd│
│  [✓] Claude Desktop + descr.     │         │  [✓] Claude Desktop              │
│  [ ] Claude Code                 │         │  [✓] Gemini CLI                  │
│      cmds inline ×2 + reload     │         │      Claude Code                 │
│      + copy prompt + advanced    │         │      1 lead line                 │
│  [ ] Codex — same again          │         │      [Copy setup prompt]         │
│                                  │         │      ▸ Show terminal commands    │
│  SUPPORTED TOOLS                 │         │      ▸ Advanced                  │
│  [ ] Windsurf + description      │         │      Codex — same shape          │
│  [ ] Zed + description           │         │                                  │
│                                  │         │  Claude.ai, ChatGPT, and more →  │
│  CLAUDE.AI & CHATGPT (WEB)       │         │  Settings → Agents (plain text)  │
│  Remote Access panel             │         │                                  │
│    toggle + ⚠ warning #1         │         │  ▸ Using another MCP client?     │
│  Claude.ai card                  │         │    Show config                   │
│    5 steps + URL + ⚠ warning #2  │         │                                  │
│  ChatGPT card                    │         └──────────────────────────────────┘
│    3 steps + URL + ⚠ warning #3  │         [Back] [Skip]     ● ● ● [Connect 3 tools]
│                                  │
│  MANUAL SETUP                    │         Warning lives ONCE, in Settings →
│  ▸ Show config snippet           │         Agents, at the Remote Access toggle,
└──────────────────────────────────┘         with amber warning treatment.
[Back] [Skip]  ● ● ●  [Continue]
```

States:

- **Loading:** existing skeleton rows (keep as-is).
- **Detected ≥ 1:** layout above. GUI rows sorted first, then CLI rows, then
  already-configured rows of either kind (they need no action).
- **Detected = 0 (scan succeeded):** empty-state line (`emptyTitle`) + the
  Settings pointer + the manual disclosure **auto-expanded**. CTA reads
  `Continue`.
- **Detection error:** `detectionFailed` line + manual disclosure
  auto-expanded (current behavior, new copy). CTA reads `Continue`.
- **Partial write failure on CTA press:** stay on the step; failed rows show
  their verbatim error (existing `role="alert"` mono line); succeeded rows flip
  to the Configured chip; CTA label recounts remaining checked rows.

CTA behavior: label is `Connect {{count}} tool(s)` where count = checked GUI
rows not yet configured; when count is 0 the label is `Continue`. Press writes
configs sequentially (existing `handleContinue` logic), then advances to
`verify` only if no write errored. `Skip` always available.

---

## 4. Component structure (question 6)

### New shared primitive: `src/components/connect/ClientRow.tsx`

One renderer for row anatomy, composed by both surfaces. This is the piece
whose duplication caused the drift; sharing the *row*, not the whole list,
keeps each surface's action semantics honest without a mode-flag component
that lies about being shared.

```tsx
interface ClientRowProps {
  client: McpClient;
  /** Settings shows the mono config path under the name; the wizard hides it
   *  (noise for a first-run user). */
  showConfigPath?: boolean;
  /** Left slot in the title row (wizard checkbox) OR right slot action
   *  (settings "Set up" button). Exactly one of leading / trailing. */
  leading?: ReactNode;
  trailing?: ReactNode;
  /** Body under the title row: CliPrimaryPath, advanced details, etc. */
  children?: ReactNode;
  error?: string | null;
  configured: boolean; // renders StatusChip state={{kind:"up"}} label=Configured
}
```

Rendering rules:

- Title row: `[leading?] name [StatusChip if configured] [trailing?]`.
- Configured indicator is `StatusChip` with `state={{ kind: "up" }}` and label
  `connectMatrix.configured` — in BOTH surfaces. `ClientSetupList`'s current
  raw sage-colored `<span>` (`ClientSetupList.tsx:50-53`) is replaced: a
  hand-colored span is a soft impersonation of the chip vocabulary. The config
  file was actually read (`already_configured` comes from parsing it), so
  `kind:"up"` is an honest probe result. The label stays "Configured" — NOT
  "Connected" — because a config entry was observed, not a live handshake.
- Error slot: existing `role="alert"` mono danger line, verbatim error text.
- Card shell: `--mem-surface` bg, `--mem-border`; wizard checkbox rows keep the
  `--mem-accent-indigo-border` highlight when checked.
- When `leading` is a checkbox, ClientRow wraps checkbox + name in a `<label>`
  and the body (`children`) stays a **sibling** wired via `aria-describedby` —
  never inside the label (preserves the current careful pattern from
  `SetupWizard.tsx:644-657`). In the new design CLI rows have no checkbox, so
  the button-inside-label hazard is structurally gone, but the rule stands for
  any future row that combines both.

### `CliPrimaryPath` (modified, shared, no fork)

```
CliPrimaryPath
├─ lead line (1 sentence, per-client key)
├─ [Copy setup prompt]  ← primary button
└─ <details> "Show terminal commands"
    ├─ command row(s) with per-command Copy buttons (existing markup)
    └─ reload note (existing keys, now inside the disclosure)
```

Same component everywhere — Settings and wizard render identical CLI bodies.
This inverts today's layout (commands inline, prompt last) in both surfaces at
once, so there is still exactly one CLI-path renderer.

### Wizard `ConnectStep` (rewritten body)

```
ConnectStep
├─ StepShell (existing; primaryAction = counting CTA)
├─ header: h1 (setup.connect.title) + p (setup.connect.description)
├─ SectionHeader (setup.connect.detectedOnMac)   ← only when list nonempty
├─ detected clients, mapped to ClientRow:
│   ├─ GUI row:  leading=<checkbox>, no children
│   └─ CLI row:  no checkbox; children = <CliPrimaryPath/> + Advanced
│                (<details> summary connectMatrix.advanced →
│                 Button connectMatrix.oneClickAdvanced → writeMcpConfig)
├─ p (setup.connect.settingsPointer)             ← plain text, not a control
└─ manual fallback:
    ├─ ghost disclosure button (setup.connect.showConfigSnippet)
    └─ expanded: addConfigSnippet line + <pre> snippet + prod/dev footnote
        (existing markup, unchanged; auto-expands on isError or zero detected)
```

Deleted from ConnectStep: `renderClientList` (the duplicate renderer,
`SetupWizard.tsx:594-752`), the `supportedClients` section, the
`RemoteAccessPanel` + `WebPlatformCards` imports and section, the
`setup.connect.webTools` SectionHeader.

State kept: `selectedClients` (GUI only — never seeded for CLI types),
`connectedClients`, `connectErrors`, `isConnectingAll`, `manualExpanded`.
`onConnected` wiring to the wizard parent is unchanged; `verify` step is
untouched.

### `ClientSetupList` (Settings, thinned)

Becomes a composition over `ClientRow`: fetches clients, maps each to
`ClientRow` with `showConfigPath`, `trailing` = "Set up" button (GUI,
detected), `children` = `CliPrimaryPath` + Advanced (CLI, detected), or
"Not installed" text (undetected). Behavior unchanged; only the row markup
moves into the shared component.

### `RemoteAccessPanel` (Settings only now)

- The `mode: "compact" | "full"` prop is **deleted** — the wizard was the only
  compact consumer. All full-mode content becomes unconditional.
- The warning moves out of the title-description slot: rendered directly below
  the toggle row as a warning block — amber triangle icon (`aria-hidden`) +
  text in `--mem-status-warning-text`, 12px, the exact pattern of
  `SettingRow`'s `warning` slot (`primitives.tsx:93-105`). Prefer refactoring
  the panel's header to use `SettingRow` with `warning={t("remoteAccess.noAuthWarning")}`
  so the treatment comes from the primitive, not bespoke markup.
- The warning is **always visible whenever the panel renders** — on or off,
  never behind a disclosure. Its `id` is referenced by the Toggle's
  `aria-describedby` so a screen-reader user hears the boundary at the moment
  of toggling (SettingRow already wires this).
- The ephemeral-URL fact stays adjacent to the URL: `tunnelChangesNote` /
  `stableNote` render next to the URL row (they already do in full mode).

### `WebPlatformCards` (Settings only now)

- Both `noAuthWarning` renderings (`WebPlatformCards.tsx:92-96,154,167`)
  are **deleted**. The cards sit directly below the Remote Access panel in
  `AgentsSection`, whose toggle now carries the single, louder warning.
- Everything else unchanged (steps, URL row, `tunnelOff` hint, connected
  attribution hint).

---

## 5. Copy table (question 5)

English strings below; zh-Hans and zh-Hant must be added/removed with exact
key parity (`pnpm test:i18n` enforces this). Keys not listed are unchanged.

### `setup.connect.*`

| Key | Status | en string |
| --- | --- | --- |
| `title` | changed | `Connect your AI tools` |
| `description` | changed | `Connected tools read and write your knowledge base as they work.` |
| `detectedOnMac` | changed | `Found on this Mac` |
| `connectCta_one` | **new** | `Connect {{count}} tool` |
| `connectCta_other` | **new** | `Connect {{count}} tools` |
| `connecting` | kept | `Connecting...` |
| `configured` | kept | `Configured` |
| `settingsPointer` | **new** | `Claude.ai, ChatGPT, and more tools can be connected any time in Settings → Agents.` |
| `emptyTitle` | **new** | `No AI tools found on this Mac.` |
| `detectionFailed` | changed | `Couldn't scan for AI tools. Add Wenlan to any MCP client with the config below.` |
| `showConfigSnippet` | changed | `Using another MCP client? Show config` |
| `addConfigSnippet` | kept | `Add this to your MCP client's configuration file:` |
| `productionDefault` | kept | (unchanged) |
| `developmentPath` | kept | (unchanged) |
| `webTools` | **removed** | — |
| `supportedTools` | **removed** | — |
| `manualSetup` | **removed** | — (the disclosure button is its own header now) |
| `detectedDescription` | **removed** | — |
| `supportedDescription` | **removed** | — |
| `connectedDescription` | **removed** | — |

### `connectMatrix.*`

| Key | Status | en string |
| --- | --- | --- |
| `claudeCodePrimary` | changed | `Copy the setup prompt and paste it into Claude Code — it sets itself up.` |
| `codexPrimary` | changed | `Copy the setup prompt and paste it into Codex — it sets itself up.` |
| `showCommands` | **new** | `Show terminal commands` |
| `copySetupPrompt` | kept | `Copy setup prompt` |
| `promptCopied` | kept | `Prompt copied` |
| `claudeCodeCommand1/2`, `codexCommand`, `*Reload`, `*Prompt`, `copyCommand*` | kept | (unchanged, now inside the disclosure) |
| `advanced` | kept | `Advanced` |
| `oneClickAdvanced` | kept | `Or write the config for me` (now truthful in the wizard) |
| `noAuthWarning` | **removed** | — (single-sourced at `remoteAccess.noAuthWarning`) |
| all other card/URL keys | kept | (Settings-only now) |

### `remoteAccess.*`

| Key | Status | en string |
| --- | --- | --- |
| `noAuthWarning` | kept **verbatim** | `Creates a public HTTPS URL with no authentication for Claude.ai and ChatGPT. Anyone with the URL can access Wenlan; turn Remote Access off when unused.` |
| everything else | kept | (unchanged) |

Word-count check for the common case (5 detected: Cursor, Claude Desktop,
Gemini CLI, Claude Code, Codex): title 4 + description 11 + section header 4 +
3 GUI row names ≈ 6 + 2 CLI rows (name + 13-word lead + 3-word button +
"Show terminal commands" + "Advanced") ≈ 42 + pointer 14 + manual line 6 +
action bar 4 ≈ **~90 words total, ~65 before the CLI leads** — versus ~370
today. Every remaining sentence is load-bearing: one explains the screen, one
per CLI row explains its single action, one points to Settings, one is the
escape hatch.

Copy register notes (per the frontend-design skill): active voice; the CTA
names the exact outcome (`Connect 3 tools`, never `Submit`/`Continue` when it
writes files); "Configured" is used consistently for the config-file state in
both surfaces; no sentence describes what a control already shows. Never call
anything "a plugin" in prose; the literal `claude plugin install …` command
strings are verbatim terminal content and stay as-is.

---

## 6. Security invariants (implementer MUST NOT break)

1. **The warning text is verbatim.** `remoteAccess.noAuthWarning` keeps its
   exact current wording. No softening, no summarizing, no splitting.
2. **Exactly one rendering.** After this change the string (and any equivalent
   phrasing) appears exactly once in the app: in `RemoteAccessPanel`, adjacent
   to the toggle. Add a test: render `AgentsSection` → the warning text occurs
   exactly once; render the wizard connect step → it occurs zero times (the
   wizard has no Remote Access surface at all).
3. **At the decision, with force.** The warning sits directly under the toggle
   row, always visible when the panel is visible (never inside a disclosure),
   with the amber warning icon + `--mem-status-warning-text` treatment, and is
   referenced by the toggle's `aria-describedby`.
4. **The wizard never enables Remote Access** — there is no code path from the
   connect step that calls `toggleRemoteAccess`.
5. **Chip never lies.** `StatusChip`'s `state` prop remains the only color
   input. `kind:"up"` on client rows is backed by the actual config-file read
   (`already_configured`); the label stays "Configured", never "Connected".
   No `Tag` or raw span may use success/danger palette colors to imitate a
   probe (this includes deleting `ClientSetupList`'s sage "Configured" span in
   favor of the chip).
6. **Config writes are explicit.** Pre-checking is a default; `writeMcpConfig`
   fires only on the CTA press (or the per-row Advanced button). Never on
   mount, never on detection.
7. **Errors verbatim.** Write failures render the raw error string in the row,
   `role="alert"`, never a friendly paraphrase.

## 7. Accessibility invariants

1. Decorative glyphs (chevrons, warning triangle, spinner) carry
   `aria-hidden="true"`.
2. Status/hint text is wired via `aria-describedby`, never placed inside a
   `<label>`. GUI rows: `<label>` wraps only checkbox + name (+ chip text).
   Any row body containing a real `<button>` is a sibling of the label.
3. Checkboxes keep `aria-label={client.name}`; when a row body exists it gets
   an `id` referenced by the checkbox's `aria-describedby`.
4. The Remote Access `Toggle` keeps `aria-pressed` (via the primitive) and
   `aria-label` = panel title, with `aria-describedby` → the warning's id.
5. Disclosure controls (`Show terminal commands`, `Advanced`, `Using another
   MCP client? Show config`) expose `aria-expanded` — native `<details>` gets
   this for free; the manual-snippet ghost Button must set it explicitly.
6. The counting CTA's label change (`Continue` ↔ `Connect N tools`) is the
   action-bar Button's text node — no `aria-live` region needed (it's the
   focused/adjacent control), but don't move the count into a separate
   decorative element.
7. Focus-visible outlines come from the primitives — do not introduce raw
   `<button>`s without the `focus-visible:outline` classes (`CliPrimaryPath`'s
   bespoke buttons should migrate to the `Button` primitive while it's being
   edited).
8. `StatusChip` keeps `aria-live="polite"` (from the primitive) so a row
   flipping to Configured after the batch write is announced.

## 8. What this deletes (inventory for the implementer)

- `SetupWizard.tsx`: `renderClientList` (lines 594–752), the web-tools section
  (lines 843–847), the supported-tools section (836–841), `manualSetup`
  SectionHeader, imports of `RemoteAccessPanel` / `WebPlatformCards`.
- `WebPlatformCards.tsx`: `noAuthWarning` const and both usages.
- `RemoteAccessPanel.tsx`: the `mode` prop and both `mode === "full"` guards
  (content becomes unconditional); warning moves to the warning slot.
- `ClientSetupList.tsx`: `rowShell` (replaced by `ClientRow`), the sage
  configured span.
- i18n: the six removed `setup.connect.*` keys + `connectMatrix.noAuthWarning`,
  in **all three locales**.

## 9. Test impact

Existing suites touching this surface: `SetupWizard.test.tsx`,
`ClientSetupList.test.tsx`, `WebPlatformCards.test.tsx`,
`RemoteAccessPanel.test.tsx`, `AgentsSection.test.tsx`,
`SettingsPrivacyClaim.test.tsx` (check whether it asserts on the warning —
it must keep passing with the single rendering).

New assertions worth writing (mutation-proof them — break the product code and
watch them fail):

1. Warning-once differential: `AgentsSection` renders the no-auth warning text
   exactly once; wizard connect step renders it zero times.
2. CLI rows render no checkbox; GUI detected+unconfigured rows render a
   pre-checked checkbox.
3. CTA label counts checked rows and falls back to `Continue` at zero.
4. Wizard "Or write the config for me" calls `writeMcpConfig` (the drift bug's
   regression test).
5. Zero-detected and detection-error states auto-expand the manual snippet.

## 10. Uncertainties (explicit, not hand-waved)

- **Prompt-primary vs commands-primary for CLI rows** (§2, flagged): chosen on
  reasoning, not data. Cheap to flip later; structure is unaffected.
- **Manual JSON snippet has no Settings home.** Today it exists only in the
  wizard; `connectMatrix.manualTitle` is unused in `AgentsSection`. I kept the
  wizard fallback so onboarding stays self-sufficient, and recommend a
  follow-up (out of scope here) adding the same snippet block to
  Settings → Agents so `settingsPointer` eventually covers the manual path too.
  The pointer's current wording deliberately does not promise manual setup in
  Settings.
- **zh plural forms for `connectCta`**: Chinese has no plural split; both
  locales use a single `connectCta` form (`连接 {{count}} 个工具` shape) — the
  translator should confirm the counter word.
- **Preview fixtures**: the wizard preview (`?mode=wizard&step=connect`)
  currently mocks zero detected clients, so the redesigned common case can't
  be pixel-reviewed. Recommend adding a fixture with 5 detected clients
  (3 GUI / 2 CLI, one already configured) to `preview/fixtures.ts` — optional,
  non-blocking.
- **Row ordering** (GUI first, then CLI, configured last) is my call for scan
  order (actionable-by-checkbox first); detection order would also be
  defensible. Implementer should not agonize — take the spec's order.

---

## 12. Adjudication (coordinator, 2026-07-12) — BINDING, overrides §2 and §4 where they conflict

The spec's thesis (detection is the hero; one counting CTA; web path leaves
onboarding; one shared `ClientRow`) is **approved as written**. Four corrections
and three additions below are binding on the implementer.

### 12.1 CORRECTION — the CLI/GUI split rests on a false premise

§2 (question 4) assumes CLI clients cannot have their config written
programmatically. **They can.** `app/src/mcp_config.rs:22-34` (`client_config_path`)
covers `claude_code` → `~/.claude.json` and `codex_cli` → `~/.codex/config.toml`,
and both `write_wenlan_entry` (JSON) and `write_wenlan_entry_toml` (TOML) are
implemented and unit-tested. `write_mcp_config` works for all five clients today.

So "CLI vs GUI" is not the real axis. There is exactly **one** special case.

### 12.2 The real axis: Claude Code would be registered TWICE

Writing `~/.claude.json` for Claude Code **duplicates the MCP server the Wenlan
plugin already registers.** Proven live on the maintainer's machine:

```
~/.claude.json                     mcpServers.wenlan → .../wenlan-mcp
plugin .mcp.json                   mcpServers.wenlan → ${CLAUDE_PLUGIN_ROOT}/bin/wenlan-mcp-runner.sh
```

Both load: a Claude Code session on this machine exposes every Wenlan tool twice
(`mcp__wenlan__capture` **and** `mcp__plugin_wenlan_wenlan__capture`). If the
wizard pre-checks Claude Code and batch-writes its config, it manufactures this
double registration for every plugin user.

**Binding rules:**

| Client | Wizard row | Rationale |
| --- | --- | --- |
| `cursor`, `claude_desktop`, `gemini_cli`, **`codex_cli`** | checkbox, pre-checked, batch write | writable; no plugin to collide with. The live `~/.codex/config.toml` already uses the MCP path (`command = "/Users/lucian/.wenlan/bin/wenlan-mcp"`). |
| **`claude_code`** | **no checkbox** — plugin path only | an MCP write collides with the plugin's own `wenlan` server |

Codex therefore **moves out of the CLI-row treatment and into the checkbox
batch** (overriding §2 and the §3 sketch, which show it as a CLI row). Claude
Code keeps the no-checkbox row exactly as §2 designs it — the design was right,
the stated reason was not. The wizard **must never write `~/.claude.json`**.

Consequence for §4: `isCliPrimaryClient` is no longer a category. It reduces to
`client_type === "claude_code"`. Do not keep a two-member list.

### 12.3 CORRECTION — the shipped commands are DEAD; here are the live ones

The wizard currently tells users to run
`claude plugin marketplace add 7xuanlu/wenlan` and install
`wenlan@7xuanlu-wenlan`. **`.claude-plugin/marketplace.json` was deleted from the
`wenlan` repo on 2026-06-15** (commit `048d77a8`, "retire self-marketplace"), so
that marketplace no longer exists and the command fails. The backend README at
HEAD is the authority:

```
/plugin marketplace add 7xuanlu/claude-plugins
/plugin install wenlan@7xuanlu
/setup
```

These strings are also embedded in the **"Copy setup prompt"** text — fix them
there too, or the redesign ships a prompt that instructs an agent to run a dead
command.

**Copy-rule note (adjudicated):** the project rule "never announce that something
is a plugin" governs *prose*. A command quoted verbatim in a `<code>` block is a
command, not a claim. Keep the commands exact; keep the surrounding prose free of
"install the Wenlan plugin" phrasing.

### 12.4 ADDITION — detect the plugin, or the wizard lies to configured users

`has_configured_entry` only inspects `mcpServers` in the client's own config, so a
user who installed via the plugin reads as **not configured** and is told to set up
again. Claude Code's `already_configured` must additionally be true when the
plugin is enabled:

`~/.claude/settings.json` → `enabledPlugins` contains a **`wenlan@*`** key set to
`true`. (On this machine: `"wenlan@7xuanlu-wenlan": true`. A fresh install yields
`wenlan@7xuanlu` — match the `wenlan@` prefix, **never a literal marketplace
name**, or the check breaks for exactly one of the two populations.)

Missing/malformed `settings.json` → treat as "no plugin", never an error.

### 12.5 ADDITION (BLOCKER) — `find_wenlan_mcp_binary()` misses the real binary

The batch write is now the default path, which makes this load-bearing.
`app/src/mcp_config.rs:120-127` looks in `~/.cargo/bin`,
`~/Repos/wenlan/target/{release,debug}` (a **personal dev-checkout path hardcoded
in shipped product code**), and `/usr/local/bin`. It does **not** look in
`~/.wenlan/bin/wenlan-mcp` — the canonical install location
(`install.sh:14-22,112-115` sets `BIN_DIR="${HOME}/.wenlan/bin"`; the plugin's
runner script hardcodes the same path; the live `~/.codex/config.toml` uses it).

So every real user's config gets `npx -y wenlan-mcp`. That does resolve (published;
`dist-tags.latest = 0.12.1`) but it is **unpinned** — it can drift away from the
daemon version the app is running.

New resolution order:

1. existing dev paths (`~/.cargo/bin`, `~/Repos/wenlan/target/{release,debug}`) — unchanged, so dev boxes keep their current behavior
2. `~/.wenlan/bin/wenlan-mcp` — canonical install
3. **sibling of `current_exe()`** — the app bundles `wenlan-mcp` as a sidecar in `Contents/MacOS/`, so a `.dmg`-only user (who never ran `install.sh` and has no `~/.wenlan/bin`) still gets a real, version-matched absolute path
4. `npx -y wenlan-mcp` — last resort

### 12.6 Two small fixes that ride along (both touch files this work already opens)

- **`src/i18n/resources.ts:12`** — `setup.privacyTitle` = "Everything stays on your
  device", a badge on the wizard's Welcome step. It is the third instance of the
  categorical privacy claim already corrected in `7ad4b30`, and the wizard offers a
  cloud model on the very next screen. Replace with **"Your memories live on this
  machine."** (all three locales). Do not delete the badge.
- **`src/components/memory/SettingsPage.tsx:63`** — the decorative lock `<svg>` is
  missing `aria-hidden="true"`.

### 12.7 Mutation-proof requirements (non-negotiable)

This project has shipped tests that stayed green while the feature they pinned was
deleted. Every load-bearing test below must be proven by breaking the product code,
watching it fail, pasting the failure, and reverting with a **targeted edit**
(never `git checkout HEAD -- <file>`):

1. The wizard **never** writes `~/.claude.json` (assert `writeMcpConfig` is never
   called with `claude_code`, even when its row is present and every box is checked).
2. "Or write the config for me" **actually calls** `writeMcpConfig` (the current
   wizard bug, `SetupWizard.tsx:678-685`).
3. The no-auth warning renders **exactly once** across the composed Settings screen
   (`getByText`, which throws on multiple matches — same shape as
   `SettingsPrivacyClaim.test.tsx`).
4. The counting CTA label tracks the checkboxes.
5. Claude Code reads as **configured** when the plugin is enabled.

A test you cannot make fail is not evidence.
