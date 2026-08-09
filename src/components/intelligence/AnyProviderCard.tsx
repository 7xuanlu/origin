// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import {
  getExternalLlm,
  setExternalLlm,
  testExternalLlm,
  listExternalModels,
  getExternalLlmKeyConfigured,
} from "../../lib/tauri";
import { useDaemonVersion } from "../../hooks/useDaemonVersion";
import { AnthropicFields, useApiKeyStatus } from "./IntelligenceSetup";
import {
  PROVIDER_PRESETS,
  presetForEndpoint,
  normalizeEndpoint,
  keyPrefixMismatch,
  visiblePresets,
  type PresetGroup,
  type ProviderPreset,
} from "./providerPresets";
import { Card, Field, Input, Button, Select, StatusChip, type ProbeState } from "../memory/settings/primitives";

// The preset table is the single source of truth for local endpoints — look
// them up by id rather than restating the URLs here, so a future table edit
// can't drift the probed endpoint away from the host shown in the UI.
function presetEndpoint(id: string): string {
  const preset = PROVIDER_PRESETS.find((p) => p.id === id);
  if (!preset) throw new Error(`Unknown preset id: ${id}`);
  return preset.endpoint;
}
const OLLAMA_ENDPOINT = presetEndpoint("ollama");
const LMSTUDIO_ENDPOINT = presetEndpoint("lmstudio");
const hostOf = (ep: string) => ep.replace(/^https?:\/\//, "").replace(/\/.*$/, "");
const localLabel = (name: string) => name.replace(/\s*\(local\)$/i, "");

// Thread #5: the discovery query is keyed on live user input, so every
// keystroke used to fire its own fetch. Delay only the value fed into that
// query key/queryFn — form responsiveness (endpointValid, canAct) stays on
// the live value.
function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  const isFirstRender = useRef(true);
  useEffect(() => {
    // Skip the mount run — there is nothing to debounce yet, and scheduling
    // a timer that just resets the value to itself is wasted work.
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

/** Card-scope-aware default: prefer Ollama (never a paid vendor) if this
 *  card's scope includes local presets; else Anthropic (native, no billing
 *  surprise) if in scope; else whatever the scope's first preset is. Runs
 *  once, as a useState lazy initializer — never re-evaluated on rerender,
 *  so a gate flipping open after mount can't move an already-made
 *  selection. */
function pickDefaultPresetId(list: ProviderPreset[]): string {
  if (list.some((p) => p.id === "ollama")) return "ollama";
  if (list.some((p) => p.id === "anthropic")) return "anthropic";
  return list[0]?.id ?? "custom";
}

export default function AnyProviderCard({
  groups,
  bare = false,
}: {
  groups?: PresetGroup[];
  /** Skip the <Card> wrapper and the title/description block — the host
   *  (a disclosure row) already supplies both. Default false keeps every
   *  existing call site (the wizard's cloud/local steps) unchanged. */
  bare?: boolean;
}) {
  const { t } = useTranslation();
  const { supportsHotSwap, supportsExternalKey } = useDaemonVersion();
  const anthropic = useApiKeyStatus();
  const queryClient = useQueryClient();

  // `groups` is commonly passed as an inline array literal, a fresh
  // reference every render — key the memo on the joined string, not on
  // `groups` identity, or it recomputes (and churns everything downstream)
  // every render regardless of whether the scope actually changed.
  const groupsKey = groups?.join(",");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const presets = useMemo(() => visiblePresets(supportsExternalKey, groups), [supportsExternalKey, groupsKey]);

  const [presetId, setPresetId] = useState(() => pickDefaultPresetId(presets));
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
    // Only adopt the saved endpoint when its preset is actually in THIS
    // card's scope — a cloud-scoped card must not jump to "custom" (or
    // show a foreign endpoint string) just because the user previously
    // saved an Ollama endpoint from a different, local-scoped card.
    if (savedEndpoint && presets.some((p) => p.id === presetForEndpoint(savedEndpoint).id)) {
      setEndpoint(savedEndpoint);
      setPresetId(presetForEndpoint(savedEndpoint).id);
      if (savedModel) setModel(savedModel);
    }
    // Run once when the saved config arrives; later edits are user-driven.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current]);

  const trimmedEndpoint = normalizeEndpoint(endpoint);
  const endpointValid = /^https?:\/\//.test(trimmedEndpoint);

  // §9.2: probe BOTH local servers on pane mount — but only when this card's
  // scope can actually show them. A cloud- or Anthropic-only card has no
  // Ollama/LM Studio chip to attach a probe result to, so skip the network
  // calls entirely rather than firing them for a dot no one will ever see.
  const ollamaInScope = presets.some((p) => p.id === "ollama");
  const lmStudioInScope = presets.some((p) => p.id === "lmstudio");
  const ollamaProbe = useQuery({
    queryKey: ["local-probe", OLLAMA_ENDPOINT],
    queryFn: () => listExternalModels(OLLAMA_ENDPOINT, null),
    enabled: ollamaInScope,
    retry: false,
    staleTime: 30_000,
  });
  const lmStudioProbe = useQuery({
    queryKey: ["local-probe", LMSTUDIO_ENDPOINT],
    queryFn: () => listExternalModels(LMSTUDIO_ENDPOINT, null),
    enabled: lmStudioInScope,
    retry: false,
    staleTime: 30_000,
  });
  const probeFor = (id: string) =>
    id === "ollama" ? ollamaProbe : id === "lmstudio" ? lmStudioProbe : null;
  // A probe is only "the" probe for the selected preset while the live
  // (trimmed) endpoint still matches that preset's endpoint — once the user
  // hand-edits Endpoint away from e.g. Ollama's default, the probed models
  // belong to a different server than the one now in the field, so the
  // association must drop (chip-never-lies: never show a stale chip or a
  // dropdown of the wrong server's models).
  const probedEndpointFor = (id: string) =>
    id === "ollama" ? OLLAMA_ENDPOINT : id === "lmstudio" ? LMSTUDIO_ENDPOINT : null;
  const selectedProbe =
    probedEndpointFor(presetId) === trimmedEndpoint ? probeFor(presetId) : null;

  // Model auto-discovery; silent fallback to free text on failure. Ollama/LM
  // Studio already have a dedicated probe above — reusing this query for them
  // would duplicate the fetch, so it stays off whenever their probe is the
  // active one. A hand-typed or "Custom…" endpoint has no dedicated probe, so
  // this query is the only discovery source for it.
  // Endpoint and key are debounced 400ms so a fetch fires once a typing
  // burst settles, not once per keystroke. `enabled` also requires the
  // debounced values to have caught up to the live ones (not just
  // endpointValid/selectedProbe, which react instantly) — otherwise a live
  // condition flipping true mid-burst would fire a fetch against the
  // still-stale debounced key.
  const debouncedEndpoint = useDebouncedValue(trimmedEndpoint, 400);
  const debouncedApiKey = useDebouncedValue(apiKey, 400);
  const discoverySettled = debouncedEndpoint === trimmedEndpoint && debouncedApiKey === apiKey;
  const discovery = useQuery({
    queryKey: ["external-models", debouncedEndpoint, debouncedApiKey],
    queryFn: () => listExternalModels(debouncedEndpoint, debouncedApiKey || null),
    enabled: endpointValid && !selectedProbe && discoverySettled,
    retry: false,
    staleTime: 30_000,
  });
  const models = discovery.data ?? [];
  const localQuery = selectedProbe;
  const localQueryModels = localQuery?.data ?? [];
  // A keyed cloud vendor's discovery results feed the same polished <Select>
  // that local presets get, once a key actually works — "paste key → model
  // list appears → pick one." Custom keeps the old free-text + datalist
  // experience: it never has a key step, so there's no equivalent moment
  // where the dropdown should just fill itself in.
  const showModelSelect = localQuery
    ? localQueryModels.length >= 1
    : preset.keyRequired && models.length >= 1;
  const effectiveModels = localQuery ? localQueryModels : models;
  // The daemon never exposes a stored key's VALUE to the frontend (security
  // posture, not a gap) — only a presence flag, `keyConfigured`, scoped to
  // the single external-llm slot rather than to any one vendor. So only a
  // key TYPED into the field right now can actually drive model discovery;
  // a stored key cannot. `keyConfigured === true` must never be read as "we
  // have a key to discover with" — it may belong to a different vendor
  // entirely (paste an OpenAI key, click the Groq chip, and `keyConfigured`
  // still reads true).
  const typedKey = apiKey.trim() !== "";
  // Which vendor the daemon's currently-stored key/endpoint actually
  // belongs to — `keyConfigured` alone can't say that, so cross-reference
  // it against the saved endpoint's preset.
  const savedPresetId = presetForEndpoint(current?.[0] ?? null).id;
  // True only when the stored key belongs to THIS card's selected vendor
  // and the user hasn't typed a fresh key over it.
  const storedKeyForThisVendor = keyConfigured === true && preset.id === savedPresetId && !typedKey;

  // Auto-select, once both in-scope probes and the key queries have settled.
  // Precedence: 1. a saved endpoint already adopted by the prefill effect
  // above wins outright. 2. else a configured, in-scope Anthropic key —
  // never auto-select a paid non-native vendor over a key the user already
  // set up. 3. else the single local responder (unchanged from before).
  const autoSelectedRef = useRef(false);
  useEffect(() => {
    if (autoSelectedRef.current) return;
    if (ollamaInScope && ollamaProbe.isLoading) return;
    if (lmStudioInScope && lmStudioProbe.isLoading) return;
    // Wait for the saved-config query to settle before latching a decision —
    // otherwise a fast probe can auto-select while `current` is still
    // undefined, and the prefill effect then re-applies the saved endpoint
    // right after, causing a visible double-set of the endpoint field.
    if (current === undefined) return;
    // Wait for the Anthropic key query too — its status is undefined until
    // it resolves, and precedence rule 2 below needs a real answer.
    if (anthropic.maskedKey === undefined) return;
    autoSelectedRef.current = true; // decide exactly once

    // 1. A saved endpoint whose preset is in scope was already adopted by
    // the prefill effect above.
    if (current[0] && presets.some((p) => p.id === presetForEndpoint(current[0]).id)) return;

    // 2. A configured Anthropic key, visible in this scope.
    if (anthropic.isConfigured && presets.some((p) => p.id === "anthropic")) {
      setPresetId("anthropic");
      return;
    }

    // 3. The single in-scope local responder.
    const up = (
      [
        ["ollama", ollamaInScope && ollamaProbe.isSuccess] as const,
        ["lmstudio", lmStudioInScope && lmStudioProbe.isSuccess] as const,
      ] as const
    ).filter(([, ok]) => ok);
    if (up.length === 1) {
      const id = up[0][0];
      setPresetId(id);
      setEndpoint(id === "ollama" ? OLLAMA_ENDPOINT : LMSTUDIO_ENDPOINT);
    }
  }, [
    ollamaInScope,
    ollamaProbe.isLoading,
    ollamaProbe.isSuccess,
    lmStudioInScope,
    lmStudioProbe.isLoading,
    lmStudioProbe.isSuccess,
    current,
    anthropic.maskedKey,
    anthropic.isConfigured,
    presets,
  ]);

  const selectPreset = (id: string) => {
    setPresetId(id);
    const next = presets.find((p) => p.id === id);
    if (next && next.endpoint) setEndpoint(next.endpoint);
    if (next && !next.endpoint) setEndpoint("");
    setModel("");
    setApiKey("");
    setTestState({ kind: "idle" });
    setSaveState("idle");
  };

  const keyToSend = (): string | undefined => (apiKey !== "" ? apiKey : undefined);

  const handleTest = async () => {
    setTestState({ kind: "testing" });
    try {
      const resp = await testExternalLlm(trimmedEndpoint, model, keyToSend() ?? null);
      setTestState({ kind: "ok", response: resp.response });
    } catch (err) {
      // Verbatim daemon error.
      setTestState({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  const handleSave = async () => {
    setSaveState("saving");
    try {
      await setExternalLlm(trimmedEndpoint, model, keyToSend());
      setSaveState(supportsHotSwap ? "applied" : "restart");
      // The daemon may hot-load this config immediately, so the routing
      // queries the job summary rows read from must refresh alongside our
      // own prefill query — otherwise the rows keep showing stale state.
      queryClient.invalidateQueries({ queryKey: ["external-llm"] });
      queryClient.invalidateQueries({ queryKey: ["external-llm-key-configured"] });
    } catch (err) {
      setSaveState(`error:${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const canAct = endpointValid && model.trim() !== "";

  // A vendor with a fixed, well-known endpoint (the 7 keyed cloud presets)
  // has nothing for the user to decide here — the endpoint is a constant
  // from the preset table, not a user choice. custom has no fixed endpoint
  // (that's the whole point of it), and ollama/lmstudio are genuinely
  // re-hostable, so both keep the field.
  const knownCloudEndpoint = preset.keyRequired && !preset.native && preset.group === "cloud";

  // Dropdown-only Model field for the keyed cloud vendors (user report: "I
  // thought even the model don't need to type, only need to select from
  // dropdown"). Every keyRequired preset that reaches this branch IS a cloud
  // vendor (knownCloudEndpoint) — local/custom never set keyRequired, so
  // this never touches their free-text-plus-datalist experience.
  // "loading" covers both an in-flight discovery fetch AND the debounce
  // window right after a keystroke, before discoverySettled catches up —
  // otherwise a stale settled state could flash the free-text fallback for
  // a moment between "key just typed" and "fetch actually started". Gated
  // on `typedKey`, not a stored key — a stored key never drives discovery
  // (see the comment above), so it must never land here either.
  const cloudModelsPending =
    knownCloudEndpoint && typedKey && !showModelSelect && (!discoverySettled || discovery.isFetching);
  // Priority: a stored key for the vendor actually selected wins first —
  // the one case where a real model can be shown with no live discovery at
  // all. Then "no usable key yet" (covers both no-key and wrong-vendor-key,
  // and a stored key with no saved model to show). Then the ordinary
  // discovery states. `storedKeyForThisVendor` already implies `!typedKey`,
  // so it can never race with "loading"/"select" below — typing a fresh key
  // flips it false and falls through to them on the next render.
  const modelFieldMode: "needsKey" | "storedKey" | "loading" | "select" | "input" =
    knownCloudEndpoint && storedKeyForThisVendor && model.trim() !== ""
      ? "storedKey"
      : knownCloudEndpoint && !typedKey
        ? "needsKey"
        : cloudModelsPending
          ? "loading"
          : showModelSelect
            ? "select"
            : "input";

  // Scope drives both copy and the Anthropic-fields no-key-guidance prop:
  // cloud-only is the wizard's Cloud model tile, which already shows its own
  // note text below the card, so AnthropicFields' guidance block would be
  // redundant there. The all-scope card (Settings, `groups` undefined) is
  // Anthropic's only entry point outside the wizard, so it keeps guidance on.
  const isCloudOnly = groups?.length === 1 && groups[0] === "cloud";
  const isLocalOnly = groups !== undefined && groups.every((g) => g !== "cloud");
  const titleKey = isCloudOnly
    ? "externalProvider.cloudTitle"
    : isLocalOnly
      ? "externalProvider.title"
      : "externalProvider.titleWithCloud";
  const descriptionKey = isCloudOnly
    ? "externalProvider.cloudDescription"
    : isLocalOnly
      ? "externalProvider.description"
      : "externalProvider.descriptionWithCloud";

  const chipState: ProbeState | null = !localQuery
    ? null
    : localQuery.isLoading
      ? { kind: "probing" }
      : localQuery.isSuccess
        ? { kind: "up" }
        : { kind: "down" };
  const chipLabel = !localQuery
    ? ""
    : localQuery.isLoading
      ? t("externalProvider.localProbing", { name: localLabel(preset.name) })
      : localQuery.isSuccess
        ? localQueryModels.length === 0
          ? t("externalProvider.localNoModelsChip", { name: localLabel(preset.name) })
          : t("externalProvider.localConnectedChip", {
              name: localLabel(preset.name),
              count: localQueryModels.length,
            })
        : t("externalProvider.localNotDetectedChip", {
            name: localLabel(preset.name),
            host: hostOf(preset.endpoint),
          });

  const body = (
    <div className="flex flex-col" style={{ gap: "12px" }}>
        {!bare && (
          <div>
            <h3
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "var(--mem-text-lg)",
                fontWeight: 600,
                color: "var(--mem-text)",
              }}
            >
              {t(titleKey)}
            </h3>
            <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", lineHeight: 1.5, marginTop: "4px" }}>
              {t(descriptionKey)}
            </p>
          </div>
        )}

        {anthropic.isConfigured && !preset.native && (
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-accent-amber)", lineHeight: 1.5 }}>
            {t("externalProvider.anthropicPrecedence")}
          </p>
        )}

        <div className="flex flex-wrap gap-2" role="group" aria-label={t("externalProvider.presetLabel")}>
          {presets.map((p) => {
            const probe = probeFor(p.id);
            const status = !probe ? null : probe.isLoading ? "probing" : probe.isSuccess ? "connected" : "notDetected";
            const dotClass =
              status === "connected"
                ? "bg-[var(--mem-status-success-text)] rounded-full"
                : status === "notDetected"
                  ? "border border-[var(--mem-text-tertiary)] rounded-full"
                  : "bg-[var(--mem-text-tertiary)] rounded-full animate-pulse";
            const selected = p.id === presetId;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => selectPreset(p.id)}
                aria-pressed={selected}
                className="rounded-full px-3 py-1.5 text-xs"
                style={{
                  border: `1px solid ${selected ? "var(--mem-accent-indigo)" : "var(--mem-border)"}`,
                  backgroundColor: selected ? "var(--mem-accent-indigo)" : "var(--mem-surface)",
                  color: selected ? "var(--mem-text-on-accent)" : "var(--mem-text)",
                  fontFamily: "var(--mem-font-body)",
                }}
              >
                {probe && (
                  <span aria-hidden="true" className={`inline-block w-1.5 h-1.5 mr-1.5 shrink-0 ${dotClass}`} />
                )}
                {localLabel(p.name)}
              </button>
            );
          })}
        </div>

        {preset.native ? (
          <AnthropicFields showNoKeyGuidance={!isCloudOnly} />
        ) : (
          <>
            {chipState && <StatusChip state={chipState} label={chipLabel} />}

            {!knownCloudEndpoint && (
              <Field label={t("externalProvider.endpointLabel")} htmlFor="any-provider-endpoint">
                <Input mono value={endpoint} onChange={(e) => setEndpoint(e.target.value)} />
              </Field>
            )}

            {/* Key precedes Model for the keyed cloud vendors: the key is
                the precondition for model discovery, so the form reads in
                the order the user actually fills it — paste key → models
                load → pick one. (preset.keyRequired is true only for cloud
                vendors here — see knownCloudEndpoint above — so this never
                reorders the local/custom presets, which have no key field.) */}
            {preset.keyRequired && (
              <Field
                label={t("externalProvider.apiKeyLabel")}
                htmlFor="any-provider-key"
                description={keyPrefixMismatch(preset, apiKey)
                  ? t("externalProvider.keyHint", { vendor: preset.name, prefix: (preset.keyPrefixes ?? []).join(" or ") })
                  : undefined}
              >
                <Input
                  type="password"
                  mono
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder={keyConfigured ? t("externalProvider.apiKeyConfiguredPlaceholder") : preset.keyPlaceholder ?? ""}
                />
              </Field>
            )}
            {preset.keyRequired && preset.getKeyUrl && (
              <Button variant="ghost" size="sm" onClick={() => shellOpen(preset.getKeyUrl!)} className="self-start">
                {t("externalProvider.getKeyLink")}
              </Button>
            )}

            <Field
              label={t("externalProvider.modelLabel")}
              htmlFor="any-provider-model"
              description={
                modelFieldMode === "storedKey" ? t("externalProvider.modelSelectStoredKeyHint") : undefined
              }
            >
              {modelFieldMode === "select" ? (
                <Select mono value={model} onChange={(e) => setModel(e.target.value)}>
                  <option value="">{t("externalProvider.modelSelectPlaceholder")}</option>
                  {/* A saved model that's no longer among the discovered ids
                      (e.g. removed from Ollama since it was saved) must still
                      render as the visibly-selected value — never a blank select
                      while Save/Test remain enabled on a value the user can't see. */}
                  {model !== "" && !effectiveModels.includes(model) && <option value={model}>{model}</option>}
                  {effectiveModels.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </Select>
              ) : modelFieldMode === "needsKey" ? (
                // Cloud vendor, no key yet: dropdown-only (user report) — no
                // free-text input to type a model id into before there's a
                // key to discover one with.
                <Select mono disabled value="">
                  <option value="">{t("externalProvider.modelSelectNeedsKey")}</option>
                </Select>
              ) : modelFieldMode === "storedKey" ? (
                // Honestly disabled: the daemon never hands the stored key's
                // value back to the frontend, so re-listing models here is
                // genuinely impossible without the user retyping it — this
                // shows the last-saved model rather than pretending to offer
                // a dropdown it can't populate. The Field description above
                // (modelSelectStoredKeyHint) tells the user how to get an
                // editable one back.
                <Select mono disabled value={model}>
                  <option value={model}>{model}</option>
                </Select>
              ) : modelFieldMode === "loading" ? (
                <Select mono disabled value="">
                  <option value="">{t("externalProvider.modelSelectLoading")}</option>
                </Select>
              ) : (
                // Escape hatch: a keyed cloud vendor whose /models call
                // errored (or returned nothing) falls back to free text —
                // see the preset-table comment on why a dropdown-only UI
                // with no fallback would permanently brick a vendor whose
                // /models response shape drifts. Local/custom presets always
                // land here too; they never get dropdown-only treatment.
                <Input
                  mono
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder={preset.modelPlaceholder ?? t("externalProvider.modelPlaceholder")}
                  list="any-provider-models"
                />
              )}
            </Field>
            {modelFieldMode === "input" && (
              <datalist id="any-provider-models">
                {models.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
            )}
            {/* Discovery with no typed key is a guaranteed 401 for a cloud
                vendor — expected, not worth surfacing. Local probes
                (localQuery) keep their own error path unconditionally: they
                have no key requirement at all, so there's no "expected
                failure" case to suppress. */}
            {(((!knownCloudEndpoint || typedKey) && discovery.isError) || localQuery?.isError) && (
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
                {t("externalProvider.modelDiscoveryFailed")}
              </span>
            )}

            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={handleTest}
                disabled={!canAct}
                loading={testState.kind === "testing"}
              >
                {t("externalProvider.test")}
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={handleSave}
                disabled={!canAct}
                loading={saveState === "saving"}
              >
                {t("externalProvider.save")}
              </Button>
            </div>

            {testState.kind === "ok" && (
              <p style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-secondary)" }}>
                {t("externalProvider.testOk", { response: testState.response })}
              </p>
            )}
            {testState.kind === "error" && (
              <p style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)" }}>
                {testState.message}
              </p>
            )}
            {saveState === "applied" && (
              <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-accent-sage)" }}>
                {t("externalProvider.savedApplied")}
              </p>
            )}
            {saveState === "restart" && (
              <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)" }}>
                {t("externalProvider.savedRestart")}
              </p>
            )}
            {saveState.startsWith("error:") && (
              <p style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)" }}>
                {saveState.slice("error:".length)}
              </p>
            )}
          </>
        )}
    </div>
  );

  return bare ? body : <Card padding="card">{body}</Card>;
}
