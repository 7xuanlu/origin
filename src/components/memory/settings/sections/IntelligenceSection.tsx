// SPDX-License-Identifier: AGPL-3.0-only
import { useId, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  downloadOnDeviceModel,
  getExternalLlm,
  getModelChoice,
  getOnDeviceModel,
  getResolvedRouting,
  listExternalModels,
  setExternalLlm,
  setSourcePin,
} from "../../../../lib/tauri";
import AnyProviderCard from "../../../intelligence/AnyProviderCard";
import {
  ANTHROPIC_MODELS,
  OnDeviceModelCard,
  RoutineModelSelect,
  SynthesisModelSelect,
  useApiKeyStatus,
} from "../../../intelligence/IntelligenceSetup";
import { presetForEndpoint, type PresetGroup } from "../../../intelligence/providerPresets";
import { Card, Select, SectionHeader, StatusChip, type ProbeState } from "../primitives";

// Module-level so each is a stable reference across renders — AnyProviderCard
// memoizes its preset list on these (same convention as SetupWizard's
// identical CLOUD_GROUPS/LOCAL_GROUPS, whose steps these rows mirror).
const CLOUD_GROUPS: PresetGroup[] = ["cloud"];
const LOCAL_GROUPS: PresetGroup[] = ["local", "custom"];

// ModelChoiceSection's own <Select> defaults — mirrored here so the summary
// meta names a model even before the user has ever touched the picker.
const DEFAULT_ROUTINE_MODEL = "claude-haiku-4-5-20251001";
const DEFAULT_SYNTHESIS_MODEL = "claude-sonnet-4-6";

function anthropicModelLabel(id: string): string {
  return ANTHROPIC_MODELS.find((m) => m.id === id)?.label ?? id;
}

type JobRowId = "everyday" | "synthesis";
type SourceRowId = "cloud" | "local" | "onDevice";
/** A pinnable provider source. The daemon also reports the "no source" resolved
 *  states ("basic" for everyday, "none" for synthesis), handled separately. */
type PinSource = "anthropic" | "external" | "on_device";

const labelStyle = { fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-base)", fontWeight: 500, color: "var(--mem-text)" } as const;
const metaLineStyle = { fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", lineHeight: 1.5 } as const;
const captionStyle = { fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)", marginTop: "4px", lineHeight: 1.5 } as const;
const amberStyle = { fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-warning-text)", marginTop: "4px", lineHeight: 1.5 } as const;

/** A ⓘ affordance that discloses "what a configured model adds" on hover/focus.
 *  Pure CSS reveal (.mem-infotip in index.css) — no popover lib, no JS state.
 *  Stops propagation so clicking/pressing it never toggles the parent row. */
function InfoTip({ title, body }: { title: string; body: string }) {
  const popId = useId();
  return (
    <span
      className="mem-infotip"
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <button type="button" className="mem-infotip-btn" aria-label={title} aria-describedby={popId}>
        i
      </button>
      <span id={popId} className="mem-infotip-pop" role="tooltip">
        <span className="mem-infotip-h">{title}</span>
        <span className="mem-infotip-b">{body}</span>
      </span>
    </span>
  );
}

/** One disclosure row inside a Models/Providers card: header (name / optional
 *  hint / meta / optional ⓘ / status chip / chevron) plus an expanded body slot.
 *  The name column is the keyboard toggle <button> (aria-expanded); the whole
 *  header row also toggles on mouse click. The ⓘ is a sibling, not nested — a
 *  button inside a button is invalid — so it keeps its own hover/focus tooltip. */
function ProviderRow({
  name,
  hint,
  meta,
  tip,
  chipState,
  chipLabel,
  expanded,
  onToggle,
  children,
}: {
  name: string;
  /** State-aware value line above the meta. Job rows pass their "what this does
   *  for you" summary when a model serves the job (omitted when none does — the
   *  meta carries the no-model status instead); among source rows only On-device
   *  passes one. */
  hint?: string;
  meta: string;
  /** Optional ⓘ detail affordance (title + body) rendered before the chip. Job
   *  rows use it to disclose what a configured model adds. */
  tip?: { title: string; body: string };
  chipState: ProbeState;
  chipLabel: string;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div>
      {/* Mouse click anywhere on the header toggles; keyboard flows through the
          name <button>. No eslint here, and the a11y contract is unchanged from
          when the whole row was one button (chip/chevron were never focusable). */}
      <div className="w-full flex items-center gap-3 px-5 py-4" onClick={onToggle}>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onToggle();
          }}
          aria-expanded={expanded}
          className="min-w-0 flex-1 text-left"
        >
          <div style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
            {name}
          </div>
          {hint && (
            <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)", marginTop: "2px" }}>
              {hint}
            </p>
          )}
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", marginTop: "2px", lineHeight: 1.5 }}>
            {meta}
          </p>
        </button>
        {tip && <InfoTip title={tip.title} body={tip.body} />}
        <StatusChip state={chipState} label={chipLabel} />
        <svg
          className="w-3 h-3 shrink-0 transition-transform"
          style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)", color: "var(--mem-text-tertiary)" }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
      {expanded && <div className="px-5 pb-4">{children}</div>}
    </div>
  );
}

/** The external provider's single model, chosen in a job row. The daemon holds
 *  ONE external slot, so both jobs that pin external share this model (surfaced
 *  in a caption by the caller). Options come from live discovery against the
 *  endpoint (works for local servers; a cloud provider whose key lives on the
 *  provider row may return only the saved model — still selectable). */
function ExternalModelSelect({ endpoint, model }: { endpoint: string; model: string }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: discovered } = useQuery({
    queryKey: ["externalModels", endpoint],
    queryFn: () => listExternalModels(endpoint),
    retry: false,
  });
  const options = Array.from(new Set([...(discovered ?? []), model].filter(Boolean)));

  return (
    <div className="flex items-center justify-between">
      <div style={labelStyle}>{t("intelligence.modelLabel")}</div>
      <div className="shrink-0 w-fit">
        <Select
          size="sm"
          mono
          aria-label={t("intelligence.chooseProviderModel")}
          value={model}
          onChange={async (e) => {
            await setExternalLlm(endpoint, e.target.value);
            queryClient.invalidateQueries({ queryKey: ["external-llm"] });
            queryClient.invalidateQueries({ queryKey: ["resolvedRouting"] });
          }}
        >
          {options.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </Select>
      </div>
    </div>
  );
}

/** The on-device model chosen in a job row. Reuses OnDeviceModelCard's exact
 *  switch mechanism — there is no dedicated "select" command, so loading a
 *  cached model is `downloadOnDeviceModel(id)` (it loads if cached, downloads
 *  if not) followed by invalidating ["onDeviceModel"]. Cached models are
 *  selectable; uncached ones are disabled with a pointer to the provider row. */
function OnDeviceJobModelSelect() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: modelData } = useQuery({ queryKey: ["onDeviceModel"], queryFn: getOnDeviceModel });
  const models = modelData?.models ?? [];
  const loadedId = modelData?.loaded ?? null;
  const selectedId = modelData?.selected ?? null;
  const currentId = loadedId ?? selectedId ?? models[0]?.id ?? null;
  const hasUncached = models.some((m) => !m.cached);

  if (models.length === 0) {
    return <p style={metaLineStyle}>{t("intelligence.modelCatalogUnavailable")}</p>;
  }

  return (
    <div>
      <div className="flex items-center justify-between">
        <div style={labelStyle}>{t("intelligence.modelLabel")}</div>
        <div className="shrink-0 w-fit">
          <Select
            size="sm"
            mono
            aria-label={t("intelligence.chooseOnDeviceModel")}
            value={currentId ?? ""}
            onChange={async (e) => {
              await downloadOnDeviceModel(e.target.value);
              queryClient.invalidateQueries({ queryKey: ["onDeviceModel"] });
              queryClient.invalidateQueries({ queryKey: ["resolvedRouting"] });
            }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id} disabled={!m.cached}>{m.display_name}</option>
            ))}
          </Select>
        </div>
      </div>
      {hasUncached && <p style={captionStyle}>{t("intelligence.onDeviceDownloadHint")}</p>}
    </div>
  );
}

interface SourceOption {
  source: PinSource;
  label: string;
  available: boolean;
}

interface JobView {
  job: JobRowId;
  /** Resolved/derived source: a PinSource, or "basic"/"none" (nothing serving). */
  source: string;
  /** Human label of `source`, reused for the legacy read-only line, the
   *  degraded-fallback hint, and the collapsed meta. */
  sourceDisplay: string;
  meta: string;
  connected: boolean;
  degraded: boolean;
  /** Human label of the raw configured pin (via `sourceLabel`), or `null`
   *  when unpinned or on a daemon that predates the pin field. Distinct from
   *  `sourceDisplay`: on a degraded route the two differ, naming what the
   *  pin was versus what's actually serving. */
  pinnedDisplay: string | null;
  sourceOptions: SourceOption[];
  external: { endpoint: string; model: string } | null;
}

/** The expanded body of a job row: source picker (interactive in PINNED mode,
 *  read-only in LEGACY), the model picker for the chosen source, and the
 *  resolved-state honesty hints. */
function JobPicker({
  view,
  isPinned,
  onPickSource,
}: {
  view: JobView;
  isPinned: boolean;
  onPickSource: (source: PinSource) => void;
}) {
  const { t } = useTranslation();
  const { job, source, sourceDisplay, degraded, pinnedDisplay, sourceOptions, external } = view;
  const isProviderSource = source === "anthropic" || source === "external" || source === "on_device";

  return (
    <div className="flex flex-col gap-3">
      {isPinned ? (
        // PINNED mode always renders the source picker — even when nothing is
        // configured (source resolves to "basic"/"none") a fresh user must SEE
        // what can be chosen. A disabled placeholder holds the empty slot.
        <div className="flex items-center justify-between">
          <div style={labelStyle}>{t("intelligence.sourceLabel")}</div>
          <div className="shrink-0 w-fit">
            <Select
              size="sm"
              aria-label={t("intelligence.chooseSource")}
              value={isProviderSource ? source : ""}
              onChange={(e) => onPickSource(e.target.value as PinSource)}
            >
              {!isProviderSource && (
                <option value="" disabled>{t("intelligence.chooseSource")}</option>
              )}
              {sourceOptions.map((o) => (
                <option key={o.source} value={o.source} disabled={!o.available}>{o.label}</option>
              ))}
            </Select>
          </div>
        </div>
      ) : (
        // LEGACY mode: a read-only source line only makes sense for a real
        // provider source. For "basic"/"none" the guidance caption below
        // carries the row instead.
        isProviderSource && (
          <div>
            <div className="flex items-center justify-between">
              <div style={labelStyle}>{t("intelligence.sourceLabel")}</div>
              <div style={metaLineStyle}>{sourceDisplay}</div>
            </div>
            <p style={captionStyle}>{t("intelligence.sourcePinLegacyHint")}</p>
          </div>
        )
      )}

      {source === "anthropic" && (job === "everyday" ? <RoutineModelSelect /> : <SynthesisModelSelect />)}
      {source === "external" && external && <ExternalModelSelect endpoint={external.endpoint} model={external.model} />}
      {source === "on_device" && <OnDeviceJobModelSelect />}
      {/* "basic"/"none" carry their state in the collapsed meta already; the
          body gives an actionable next step instead of repeating it. In PINNED
          mode a source select sits above (on-device included when a model is
          loaded), so point there too; LEGACY has no select and can't route
          synthesis to on-device, so it only points below. */}
      {(source === "basic" || source === "none") && (
        <p style={captionStyle}>
          {t(isPinned ? "intelligence.chooseSourceOrConnectHint" : "intelligence.connectProviderBelowHint")}
        </p>
      )}

      {source === "external" && <p style={captionStyle}>{t("intelligence.sharedSlotCaption")}</p>}
      {isPinned && degraded && (
        pinnedDisplay != null ? (
          <p style={amberStyle}>
            {t("intelligence.pinnedDegradedHintPinned", { pinned: pinnedDisplay, fallback: sourceDisplay })}
          </p>
        ) : (
          <p style={amberStyle}>{t("intelligence.pinnedDegradedHint", { fallback: sourceDisplay })}</p>
        )
      )}
    </div>
  );
}

export default function IntelligenceSection({ delay }: { delay: number }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const anthropic = useApiKeyStatus();
  const { data: external } = useQuery({ queryKey: ["external-llm"], queryFn: getExternalLlm });
  const { data: modelChoice } = useQuery({
    queryKey: ["modelChoice"],
    queryFn: getModelChoice,
    enabled: anthropic.isConfigured,
  });
  const { data: onDevice } = useQuery({ queryKey: ["onDeviceModel"], queryFn: getOnDeviceModel });
  // Feature detection: null ⇒ the daemon has no routing endpoint (LEGACY mode);
  // an object ⇒ per-job pins are live (PINNED mode). undefined while loading —
  // treated as legacy, which is also what the live 0.13.2 daemon returns.
  const { data: routing } = useQuery({ queryKey: ["resolvedRouting"], queryFn: getResolvedRouting });
  const isPinned = routing != null;

  const [expandedJob, setExpandedJob] = useState<JobRowId | null>(null);
  const toggleJob = (row: JobRowId) => setExpandedJob((cur) => (cur === row ? null : row));
  const [expandedSource, setExpandedSource] = useState<SourceRowId | null>(null);
  const toggleSource = (row: SourceRowId) => setExpandedSource((cur) => (cur === row ? null : row));

  // The daemon holds exactly one external-llm slot — a saved local server and
  // a saved cloud vendor overwrite each other — so the saved slot is
  // attributed to exactly one row, via its own preset's group.
  const [externalEndpoint, externalModel] = external ?? [null, null];
  const externalPreset = externalEndpoint ? presetForEndpoint(externalEndpoint) : null;
  const externalConfigured = externalPreset !== null;
  const cloudConnected = anthropic.isConfigured || externalPreset?.group === "cloud";
  const localConnected = externalPreset?.group === "local" || externalPreset?.group === "custom";

  const [routineId, synthesisId] = modelChoice ?? [null, null];

  // Cloud provider row — connection-only meta (provider name + masked key style,
  // per mockup). Per-job models live in the job rows now, never repeated here.
  const cloudMeta = anthropic.isConfigured
    ? `Anthropic · ${anthropic.maskedKey}`
    : externalPreset?.group === "cloud"
      ? `${externalPreset.name} · ${externalModel}`
      : t("intelligence.cloudRowHint");

  const localMeta = localConnected && externalPreset
    ? `${externalPreset.name} · ${externalModel}`
    : t("intelligence.localRowHint");

  const models = onDevice?.models ?? [];
  const loadedId = onDevice?.loaded ?? null;
  const selectedId = onDevice?.selected ?? null;
  const currentId = loadedId ?? selectedId ?? models[0]?.id ?? null;
  const current = currentId ? models.find((m) => m.id === currentId) : null;
  const onDeviceLoaded = !!current && loadedId === current.id;
  const onDeviceLabel = current?.display_name ?? null;

  const onDeviceMeta = !current
    ? t("intelligence.modelCatalogUnavailable")
    : onDeviceLoaded
      ? `${current.display_name} · ${t("intelligence.running")}`
      : current.cached
        ? `${current.display_name}${t("intelligence.downloadedNotLoaded")}`
        : `${current.display_name}${t("intelligence.notDownloaded")}`;

  // Provider slot the app can attribute an external source to, in either mode.
  const routedExternal = isPinned
    ? routing.pool.external
    : externalEndpoint && externalModel
      ? { endpoint: externalEndpoint, model: externalModel }
      : null;
  const providerName = routedExternal ? presetForEndpoint(routedExternal.endpoint)?.name ?? null : null;

  // What each provider source can currently serve — drives option enable/disable
  // in the pinned source select and the legacy derived chain.
  const avail = isPinned
    ? {
        anthropic: routing.pool.anthropic.configured,
        external: routing.pool.external != null,
        on_device: routing.pool.on_device != null,
      }
    : { anthropic: anthropic.isConfigured, external: externalConfigured, on_device: onDeviceLoaded };

  const sourceLabel = (source: string): string => {
    switch (source) {
      case "anthropic":
        return t("intelligence.sourceAnthropic");
      case "external":
        return providerName ?? t("intelligence.sourceConnectedProvider");
      case "on_device":
        return onDeviceLabel ?? t("intelligence.sourceOnDevice");
      case "basic":
        return t("intelligence.servingBasic");
      default:
        return t("intelligence.synthesisNoSourceHint");
    }
  };

  // Collapsed meta for a resolved (source, model). Shared by both modes so the
  // row reads identically whether the source came from the routing endpoint or
  // the legacy derived chain.
  const routeMeta = (job: JobRowId, source: string, model: string | null): string => {
    const fallbackModel = job === "everyday" ? DEFAULT_ROUTINE_MODEL : DEFAULT_SYNTHESIS_MODEL;
    switch (source) {
      case "anthropic":
        return `Anthropic · ${anthropicModelLabel(model ?? fallbackModel)}`;
      case "external":
        return `${providerName ?? t("intelligence.sourceConnectedProvider")} · ${model ?? routedExternal?.model ?? ""}`;
      case "on_device":
        return onDeviceLabel ?? model ?? t("intelligence.sourceOnDevice");
      default:
        return sourceLabel(source);
    }
  };

  const buildOptions = (job: JobRowId): SourceOption[] => {
    // No vendor privilege: everyday leads with the recommended on-device option;
    // synthesis lists cloud/server first (better synthesis quality) then offers
    // on-device last — pinnable once the daemon accepts synthesis=on_device.
    const order: PinSource[] = job === "everyday" ? ["on_device", "anthropic", "external"] : ["anthropic", "external", "on_device"];
    const suffix = t("intelligence.sourceUnavailableSuffix");
    return order.map((s) => {
      const available = avail[s];
      let base =
        s === "anthropic"
          ? t("intelligence.sourceAnthropic")
          : s === "on_device"
            ? t("intelligence.sourceOnDevice")
            : providerName ?? t("intelligence.sourceConnectedProvider");
      if (s === "on_device" && job === "everyday" && available) {
        base = `${base} · ${t("intelligence.sourceRecommended")}`;
      }
      return { source: s, label: available ? base : `${base}${suffix}`, available };
    });
  };

  const buildView = (job: JobRowId): JobView => {
    const source = isPinned
      ? (job === "everyday" ? routing.everyday.source : routing.synthesis.source)
      : job === "everyday"
        ? anthropic.isConfigured
          ? "anthropic"
          : onDeviceLoaded
            ? "on_device"
            : "basic"
        : anthropic.isConfigured
          ? "anthropic"
          : externalConfigured
            ? "external"
            : "none";
    const model = isPinned
      ? (job === "everyday" ? routing.everyday.model : routing.synthesis.model)
      : source === "anthropic"
        ? (job === "everyday" ? routineId : synthesisId)
        : source === "external"
          ? externalModel
          : null;
    const mode = isPinned ? (job === "everyday" ? routing.everyday.mode : routing.synthesis.mode) : "legacy";
    const pin = isPinned ? (job === "everyday" ? routing.everyday.pin : routing.synthesis.pin) : null;
    return {
      job,
      source,
      sourceDisplay: sourceLabel(source),
      meta: routeMeta(job, source, model),
      connected: source !== "basic" && source !== "none",
      degraded: mode === "pinned_degraded",
      pinnedDisplay: pin ? sourceLabel(pin) : null,
      sourceOptions: buildOptions(job),
      external: routedExternal,
    };
  };

  const everydayView = buildView("everyday");
  const synthesisView = buildView("synthesis");

  const pickSource = async (job: JobRowId, source: PinSource) => {
    await setSourcePin(job === "everyday" ? source : null, job === "synthesis" ? source : null);
    queryClient.invalidateQueries({ queryKey: ["resolvedRouting"] });
  };

  // With a model, the row leads with the everyday-value summary and its ⓘ carries
  // the "how it works, human-gated" detail. Without one, the summary is dropped
  // (the meta already states the no-model reality) and the ⓘ becomes "what a model
  // adds". buildView's `connected` is the switch.
  const jobRow = (view: JobView, id: JobRowId, name: string, summary: string) => {
    const on = view.connected;
    const tip = on
      ? id === "everyday"
        ? { title: t("intelligence.routineModelTipTitle"), body: t("intelligence.routineModelTipBody") }
        : { title: t("intelligence.synthesisModelTipTitle"), body: t("intelligence.synthesisModelTipBody") }
      : {
          title: t("intelligence.noModelTipTitle"),
          body: id === "everyday" ? t("intelligence.routineNoModelTipBody") : t("intelligence.synthesisNoModelTipBody"),
        };
    return (
      <ProviderRow
        name={name}
        hint={on ? summary : undefined}
        meta={view.meta}
        tip={tip}
        chipState={view.connected ? { kind: "up" } : { kind: "idle" }}
        chipLabel={view.connected ? t("intelligence.connected") : t("intelligence.notConfigured")}
        expanded={expandedJob === id}
        onToggle={() => toggleJob(id)}
      >
        <JobPicker view={view} isPinned={isPinned} onPickSource={(s) => pickSource(id, s)} />
      </ProviderRow>
    );
  };

  return (
    <section className="mem-fade-up" style={{ animationDelay: `${delay}ms` }}>
      <div className="flex flex-col gap-3">
        <div className="mem-fade-up" style={{ animationDelay: `${delay + 30}ms` }}>
          <SectionHeader label={t("intelligence.modelsTitle")} />
          <Card padding="rows">
            {jobRow(everydayView, "everyday", t("intelligence.routineModel"), t("intelligence.routineModelDescription"))}
            {jobRow(synthesisView, "synthesis", t("intelligence.synthesisModel"), t("intelligence.synthesisModelDescription"))}
          </Card>
        </div>

        <div className="mem-fade-up" style={{ animationDelay: `${delay + 60}ms` }}>
          <SectionHeader label={t("intelligence.providersTitle")} />
          <Card padding="rows">
            <ProviderRow
              name={t("intelligence.cloudRow")}
              meta={cloudMeta}
              chipState={cloudConnected ? { kind: "up" } : { kind: "idle" }}
              chipLabel={cloudConnected ? t("intelligence.connected") : t("intelligence.notConfigured")}
              expanded={expandedSource === "cloud"}
              onToggle={() => toggleSource("cloud")}
            >
              <AnyProviderCard bare groups={CLOUD_GROUPS} />
            </ProviderRow>
            <ProviderRow
              name={t("intelligence.localRow")}
              meta={localMeta}
              chipState={localConnected ? { kind: "up" } : { kind: "idle" }}
              chipLabel={localConnected ? t("intelligence.connected") : t("intelligence.notConfigured")}
              expanded={expandedSource === "local"}
              onToggle={() => toggleSource("local")}
            >
              <AnyProviderCard bare groups={LOCAL_GROUPS} />
            </ProviderRow>
            <ProviderRow
              name={t("intelligence.onDeviceModel")}
              hint={t("intelligence.onDeviceRowHint")}
              meta={onDeviceMeta}
              chipState={onDeviceLoaded ? { kind: "up" } : { kind: "idle" }}
              chipLabel={onDeviceLoaded ? t("intelligence.running") : t("intelligence.notLoaded")}
              expanded={expandedSource === "onDevice"}
              onToggle={() => toggleSource("onDevice")}
            >
              <OnDeviceModelCard bare />
            </ProviderRow>
          </Card>
        </div>
      </div>
    </section>
  );
}
