// SPDX-License-Identifier: AGPL-3.0-only
// Browser preview harness: page-detail citations, the review queue
// (DistillReviewPanel + ReviewDialog), the first-run wizard, settings, and
// the knowledge graph (AtlasView + EntityDetail/FocusGraph).
//
// The wizard and settings modes exist so pixel review of those surfaces is
// cheap. Reviewing them used to mean building the Tauri app and clicking
// through a real first run, which is why a whole redesign round once shipped
// green tests and zero reviewed pixels. Every step and section is directly
// addressable here.
import { StrictMode, Suspense, lazy, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import PageDetail from "../src/components/memory/PageDetail";
import EntityDetail from "../src/components/memory/EntityDetail";
import DistillReviewPanel from "../src/components/memory/DistillReviewPanel";
import { SetupWizard, STEP_ORDER, type WizardStep } from "../src/components/SetupWizard";
import SettingsPage from "../src/components/memory/SettingsPage";
import SettingsSidebar, {
  SETTINGS_GROUPS,
  type SettingsSection,
} from "../src/components/memory/settings/SettingsSidebar";
import { initializeI18n } from "../src/i18n";
import { resetReviewFixtures, REVIEW_FAIL } from "./fixtures";
import BakeoffHarness from "./bakeoff/BakeoffHarness";
import type { BakeoffRenderer } from "./bakeoff/bakeoffResult";
import "../src/index.css";

// Lazy so sigma lands in its own vite chunk, same as the bake-off adapters
// (see BakeoffHarness) — that chunk stays once the bake-off scaffolding is
// deleted in a later round.
const AtlasView = lazy(() => import("../src/components/memory/AtlasView"));

const VARIANTS = [
  { id: "page-cited", label: "Cited (all kinds)" },
  { id: "page-cleared", label: "Edit-cleared" },
  { id: "page-mismatch", label: "Mismatch" },
  { id: "page-plain", label: "No citations" },
];

// Imported, not copied. This WAS a hand-kept literal on the theory that drift
// would show up as "a visibly wrong tab, not a silent bug". It did the
// opposite: when the wizard renamed "verify" to "setting-up", the stale literal
// silently made that step unreachable here — ?step=setting-up failed validation
// and fell back, so the newest step in the wizard was the one nobody could
// review. Sharing the array means a rename can't desync again.
const WIZARD_STEPS: WizardStep[] = STEP_ORDER;

// Mirrors reviewSuppression.ts's STORAGE_KEY/HiddenReviewEntry shape so the
// "Seed hidden" button demoes the review panel's hidden-items footer without
// clicking through two real Hide actions. Keys reference real fixture items
// (the page-cited stale entry, the "Preview harness" orphan topic) so
// Restore visibly brings them back into the queue.
const HIDDEN_STORAGE_KEY = "wenlan.review.hidden.v1";
function seedHiddenEntries() {
  localStorage.setItem(
    HIDDEN_STORAGE_KEY,
    JSON.stringify([
      {
        key: "stale:page-cited",
        label: "Wenlan Daemon Architecture",
        kind: "stale_page",
        at: Date.now() - 3_600_000,
      },
      {
        key: "topic:Preview harness",
        label: "Preview harness",
        kind: "topic",
        at: Date.now() - 1_800_000,
      },
    ]),
  );
}

const client = new QueryClient({
  defaultOptions: { queries: { retry: false, staleTime: 0, gcTime: 0 } },
});

type Mode = "page" | "review" | "wizard" | "settings" | "atlas" | "entity" | "bakeoff";

// Quick-select entities for the "entity" tab: gk-wenlan is the highest-degree
// hub, gk-lucian shows a person node with mixed in/out edges, gk-remote-office
// is the fixture's zero-relation entity (empty-focus case).
const ENTITY_VARIANTS = [
  { id: "gk-wenlan", label: "Wenlan (hub)" },
  { id: "gk-lucian", label: "Lucian (person)" },
  { id: "gk-remote-office", label: "Remote Office (empty)" },
];

// Deep links: ?mode=wizard&step=connect, ?mode=settings&section=intelligence,
// ?mode=graph, ?mode=atlas, ?mode=entity&entity=gk-lucian, ?theme=light. Without these
// every surface would only be reachable by clicking, so a screenshot pass
// couldn't address one — which is the whole point of these modes existing.
const params = new URLSearchParams(window.location.search);
const param = <T extends string>(key: string, allowed: readonly T[], fallback: T): T => {
  const value = params.get(key) as T | null;
  return value && allowed.includes(value) ? value : fallback;
};

// Applied here, not in Harness: the toggle sets this attribute imperatively, so
// initial state alone would leave ?theme=light rendering dark.
const INITIAL_THEME = param("theme", ["dark", "light"] as const, "dark");
document.documentElement.setAttribute("data-theme", INITIAL_THEME);

// ?bar=0 drops the harness toolbar. The wizard lays out against 100vh, so the
// toolbar pushes its footer (the Continue button) off-screen — a screenshot
// taken with the bar showing would report a clipped button the app doesn't
// have. Screenshots use bar=0; humans clicking around leave it on.
const SHOW_BAR = params.get("bar") !== "0";
const BAR_H = SHOW_BAR ? 41 : 0;

function Harness() {
  const [mode, setMode] = useState<Mode>(
    param("mode", ["page", "review", "wizard", "settings", "atlas", "entity", "bakeoff"] as const, "review"),
  );
  const [pageId, setPageId] = useState(params.get("page") ?? "page-cited");
  const [entityId, setEntityId] = useState(params.get("entity") ?? "gk-wenlan");
  const [theme, setTheme] = useState(INITIAL_THEME);
  const [reviewRun, setReviewRun] = useState(0);
  const [failing, setFailing] = useState(false);
  const [wizardStep, setWizardStep] = useState<WizardStep>(
    param("step", WIZARD_STEPS, "welcome"),
  );
  const [section, setSection] = useState<SettingsSection>(
    param(
      "section",
      SETTINGS_GROUPS.map((g) => g.id),
      "general",
    ),
  );
  const [renderer, setRenderer] = useState<BakeoffRenderer>(
    param("renderer", ["cytoscape", "sigma", "g6"] as const, "cytoscape"),
  );
  const bakeoffN = Number(params.get("n")) || 1000;

  const applyTheme = (next: string) => {
    document.documentElement.setAttribute("data-theme", next);
    setTheme(next);
  };

  const tab = (active: boolean) => ({
    padding: "3px 10px",
    borderRadius: 6,
    border: "1px solid var(--mem-border)",
    background: active ? "var(--mem-accent, #6366f1)" : "transparent",
    color: active ? "#fff" : "inherit",
    cursor: "pointer",
  });

  return (
    <div style={{ minHeight: "100vh", background: "var(--mem-bg)" }}>
      {SHOW_BAR && (
      <div
        style={{
          display: "flex",
          gap: 8,
          padding: "10px 16px",
          borderBottom: "1px solid var(--mem-border)",
          alignItems: "center",
          fontFamily: "var(--mem-font-mono)",
          fontSize: 12,
          color: "var(--mem-text-secondary)",
        }}
      >
        <span style={{ fontWeight: 600 }}>PREVIEW</span>
        <button onClick={() => setMode("review")} style={tab(mode === "review")}>
          Review queue
        </button>
        <button onClick={() => setMode("page")} style={tab(mode === "page")}>
          Page detail
        </button>
        <button onClick={() => setMode("wizard")} style={tab(mode === "wizard")}>
          Wizard
        </button>
        <button onClick={() => setMode("settings")} style={tab(mode === "settings")}>
          Settings
        </button>
        <button onClick={() => setMode("atlas")} style={tab(mode === "atlas")}>
          Atlas
        </button>
        <button onClick={() => setMode("entity")} style={tab(mode === "entity")}>
          Entity
        </button>
        <button onClick={() => setMode("bakeoff")} style={tab(mode === "bakeoff")}>
          Bakeoff
        </button>
        <span style={{ opacity: 0.4 }}>|</span>
        {mode === "wizard" ? (
          WIZARD_STEPS.map((s) => (
            <button key={s} onClick={() => setWizardStep(s)} style={tab(wizardStep === s)}>
              {s}
            </button>
          ))
        ) : mode === "settings" ? (
          SETTINGS_GROUPS.map((g) => (
            <button key={g.id} onClick={() => setSection(g.id)} style={tab(section === g.id)}>
              {g.id}
            </button>
          ))
        ) : mode === "page" ? (
          VARIANTS.map((v) => (
            <button key={v.id} onClick={() => setPageId(v.id)} style={tab(pageId === v.id)}>
              {v.label}
            </button>
          ))
        ) : mode === "entity" ? (
          ENTITY_VARIANTS.map((v) => (
            <button key={v.id} onClick={() => setEntityId(v.id)} style={tab(entityId === v.id)}>
              {v.label}
            </button>
          ))
        ) : mode === "bakeoff" ? (
          (["cytoscape", "sigma", "g6"] as const).map((r) => (
            <button key={r} onClick={() => setRenderer(r)} style={tab(renderer === r)}>
              {r}
            </button>
          ))
        ) : mode === "atlas" ? null : (
          <>
            <button
              onClick={() => {
                resetReviewFixtures();
                client.clear();
                setReviewRun((n) => n + 1);
              }}
              style={tab(false)}
            >
              Reset queue
            </button>
            <button
              onClick={() => {
                REVIEW_FAIL.queue = !REVIEW_FAIL.queue;
                setFailing(REVIEW_FAIL.queue);
                client.clear();
                setReviewRun((n) => n + 1);
              }}
              style={tab(failing)}
            >
              {failing ? "Fail queue: on" : "Fail queue"}
            </button>
            <button
              onClick={() => {
                seedHiddenEntries();
                client.clear();
                setReviewRun((n) => n + 1);
              }}
              style={tab(false)}
            >
              Seed hidden
            </button>
          </>
        )}
        <button
          onClick={() => applyTheme(theme === "dark" ? "light" : "dark")}
          style={{ ...tab(false), marginLeft: "auto" }}
        >
          {theme === "dark" ? "☀ light" : "☾ dark"}
        </button>
      </div>
      )}
      {mode === "wizard" ? (
        // Full-bleed: the wizard owns the whole window in the real app.
        // initialStep only when ?step= asked for one: the wizard sets
        // hideDots = !!initialStep, so always passing it would hide the step
        // dots and misreport a real first run as having no progress indicator.
        // No ?step= → a natural run from welcome, dots and all.
        // ?model=<id> and ?import=<folder name> seed the picks the setting-up
        // step's model/import rows are conditional on. Without them, entering
        // at ?step=setting-up renders neither row — so the two slowest, least
        // reviewed rows in the wizard were the only ones the harness couldn't
        // show. ?model=stall pairs with the live-invoke stub that never reports
        // the model as loaded, which is what a dead download actually looks like.
        <SetupWizard
          key={wizardStep}
          initialStep={params.get("step") ? wizardStep : undefined}
          initialPendingModelId={params.get("model")}
          initialPendingImportPick={
            params.get("import")
              ? {
                  sourceType: "obsidian",
                  path: `/Users/preview/${params.get("import")}`,
                  label: params.get("import")!,
                }
              : null
          }
          onComplete={() => console.log("[preview] onComplete")}
        />
      ) : mode === "settings" ? (
        // Mirrors Main.tsx's composition — sidebar beside the page, not the
        // page alone. The sidebar is half the settings design.
        <div style={{ display: "flex", height: `calc(100vh - ${BAR_H}px)` }}>
          <SettingsSidebar
            collapsed={false}
            active={section}
            onSelect={setSection}
            onNavigateHome={() => console.log("[preview] onNavigateHome")}
          />
          <div style={{ flex: 1, overflowY: "auto", padding: "24px 16px" }}>
            <SettingsPage
              key={section}
              section={section}
              onBack={() => console.log("[preview] onBack")}
              onSetupAgent={() => console.log("[preview] onSetupAgent")}
              onImport={() => console.log("[preview] onImport")}
            />
          </div>
        </div>
      ) : mode === "atlas" ? (
        // Full-bleed like graph/wizard. sigma lands in its own lazy chunk
        // (see the AtlasView import above) so Suspense covers the load.
        <div style={{ height: `calc(100vh - ${BAR_H}px)` }}>
          <Suspense fallback={<div style={{ padding: 16 }}>Loading atlas…</div>}>
            <AtlasView
              onNodeClick={(id: string) => {
                setEntityId(id);
                setMode("entity");
              }}
              // Main.tsx passes navigateBack here; a logging stub keeps the
              // previewed toolbar at Main parity.
              onBack={() => console.log("[preview] onBack")}
            />
          </Suspense>
        </div>
      ) : mode === "bakeoff" ? (
        // Full-bleed like graph/wizard. renderer/n are URL-driven
        // (?mode=bakeoff&renderer=cytoscape&n=1000); the sub-tab row above
        // only switches renderer at a fixed n.
        <div style={{ height: `calc(100vh - ${BAR_H}px)` }}>
          <BakeoffHarness renderer={renderer} n={bakeoffN} />
        </div>
      ) : (
        <div style={{ maxWidth: 860, margin: "0 auto", padding: "24px 16px" }}>
          {mode === "page" ? (
            <PageDetail
              key={pageId}
              pageId={pageId}
              onBack={() => console.log("[preview] onBack")}
              onMemoryClick={(id: string) => console.log("[preview] onMemoryClick:", id)}
              onPageClick={(id: string) => {
                console.log("[preview] onPageClick:", id);
                setPageId(id);
              }}
            />
          ) : mode === "entity" ? (
            <EntityDetail
              key={entityId}
              entityId={entityId}
              onBack={() => setMode("atlas")}
              onEntityClick={(id: string) => setEntityId(id)}
              onMemoryClick={(id: string) => console.log("[preview] onMemoryClick:", id)}
            />
          ) : (
            <DistillReviewPanel
              key={reviewRun}
              onBack={() => console.log("[preview] onBack")}
              onPageClick={(id: string) => {
                console.log("[preview] onPageClick:", id);
                setMode("page");
                setPageId(id);
              }}
              onMemoryClick={(id: string) => console.log("[preview] onMemoryClick:", id)}
            />
          )}
        </div>
      )}
    </div>
  );
}

void initializeI18n().then(() => {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <QueryClientProvider client={client}>
        <Harness />
      </QueryClientProvider>
    </StrictMode>,
  );
});
