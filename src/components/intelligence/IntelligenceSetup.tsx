import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { open as shellOpen } from "@tauri-apps/plugin-shell";
import {
  downloadOnDeviceModel,
  getApiKey,
  getModelChoice,
  getOnDeviceModel,
  getSystemInfo,
  setApiKey,
  setModelChoice,
} from "../../lib/tauri";
import { Card, Field, Input, Button, Select, StatusChip } from "../memory/settings/primitives";

type AnthropicModelDescriptionKey =
  | "intelligence.modelDescriptions.fastAffordable"
  | "intelligence.modelDescriptions.balancedQuality"
  | "intelligence.modelDescriptions.maximumQuality";

export const ANTHROPIC_MODELS = [
  { id: "claude-haiku-4-5-20251001", label: "Haiku 4.5", descKey: "intelligence.modelDescriptions.fastAffordable" },
  { id: "claude-sonnet-4-6", label: "Sonnet 4.6", descKey: "intelligence.modelDescriptions.balancedQuality" },
  { id: "claude-opus-4-6", label: "Opus 4.6", descKey: "intelligence.modelDescriptions.maximumQuality" },
] satisfies Array<{
  id: string;
  label: string;
  descKey: AnthropicModelDescriptionKey;
}>;

export function useApiKeyStatus() {
  const { data: maskedKey } = useQuery({
    queryKey: ["apiKey"],
    queryFn: getApiKey,
  });

  return {
    maskedKey,
    isConfigured: !!maskedKey,
  };
}

/** The native Anthropic key form: StatusChip, masked-key/clear or key-input
 *  branch, get-key link, error line, no-key guidance, and (once configured)
 *  the routine/synthesis model pickers. No <Card> wrapper and no <h3> title
 *  — every host (the Settings card, AnyProviderCard's Anthropic chip)
 *  supplies its own title, so this is just the body. */
export function AnthropicFields({
  showModelChoice = true,
  showNoKeyGuidance = true,
}: {
  showModelChoice?: boolean;
  showNoKeyGuidance?: boolean;
}) {
  const { t } = useTranslation();
  const [keyInput, setKeyInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const { maskedKey, isConfigured } = useApiKeyStatus();

  // A save/clear here changes what the daemon is actually serving (the
  // Anthropic key is hot-loaded even on 0.12), so the routing queries the
  // job summary rows read from must be invalidated alongside the key
  // itself — otherwise the rows keep showing stale state until remount.
  const invalidateIntelligenceQueries = () => {
    queryClient.invalidateQueries({ queryKey: ["apiKey"] });
    queryClient.invalidateQueries({ queryKey: ["external-llm"] });
    queryClient.invalidateQueries({ queryKey: ["external-llm-key-configured"] });
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await setApiKey(keyInput);
      setKeyInput("");
      invalidateIntelligenceQueries();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    setSaving(true);
    try {
      await setApiKey("");
      invalidateIntelligenceQueries();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col" style={{ gap: "10px" }}>
      <div className="flex items-center justify-end">
        <StatusChip
          state={isConfigured ? { kind: "up" } : { kind: "idle" }}
          label={isConfigured ? t("intelligence.connected") : t("intelligence.notConfigured")}
        />
      </div>
      <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
        {t("intelligence.apiKeyDescription")}
      </p>

      {isConfigured ? (
        <div className="flex items-center gap-2">
          <span
            className="flex-1 px-3 py-1.5 rounded-md"
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "var(--mem-text-sm)",
              color: "var(--mem-text-secondary)",
              backgroundColor: "var(--mem-hover)",
            }}
          >
            {maskedKey}
          </span>
          <Button variant="ghost" size="sm" onClick={handleClear} disabled={saving}>
            {t("intelligence.clear")}
          </Button>
        </div>
      ) : (
        <div className="flex items-end gap-2">
          <div className="flex-1">
            <Field label={t("externalProvider.apiKeyLabel")} htmlFor="anthropic-api-key">
              <Input
                type="password"
                mono
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && keyInput) handleSave(); }}
                placeholder="sk-ant-api03-..."
              />
            </Field>
          </div>
          <Button variant="primary" size="sm" onClick={handleSave} disabled={saving || !keyInput} loading={saving}>
            {t("intelligence.save")}
          </Button>
        </div>
      )}

      {!isConfigured && (
        <Button variant="ghost" size="sm" onClick={() => shellOpen("https://console.anthropic.com/settings/keys")} className="self-start">
          {t("externalProvider.getKeyLink")}
        </Button>
      )}

      {error && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)" }}>
          {error}
        </p>
      )}

      {!isConfigured && showNoKeyGuidance && (
        <div
          className="rounded-lg"
          style={{
            padding: "10px 14px",
            backgroundColor: "var(--mem-hover)",
            fontSize: "var(--mem-text-sm)",
            color: "var(--mem-text-secondary)",
            lineHeight: 1.6,
          }}
        >
          <div style={{ fontWeight: 500, color: "var(--mem-text)", marginBottom: 2, fontSize: "var(--mem-text-sm)" }}>
            {t("intelligence.synthesisCloudRecommendation")}
          </div>
          <div>{t("intelligence.memorySafe")}</div>
          <div style={{ marginTop: 6 }}>{t("intelligence.addApiKey")}</div>
        </div>
      )}

      {isConfigured && showModelChoice && <ModelChoiceSection />}
    </div>
  );
}

/** Just the routine-model half of the picker — extracted so the Intelligence
 *  section's Everyday-model job row can embed exactly this Select in its
 *  disclosure body, without forking the markup ModelChoiceSection also
 *  renders below the Anthropic key form. */
export function RoutineModelSelect() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: modelChoice } = useQuery({
    queryKey: ["modelChoice"],
    queryFn: getModelChoice,
  });
  const [routineModel, synthesisModel] = modelChoice ?? [null, null];

  return (
    <div className="flex items-center justify-between">
      <div>
        <div style={{ fontSize: "var(--mem-text-base)", fontWeight: 500, color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}>{t("intelligence.routineModel")}</div>
        <div style={{ fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)", fontFamily: "var(--mem-font-body)" }}>{t("intelligence.routineModelDescription")}</div>
      </div>
      <div className="shrink-0 w-fit">
        <Select
          size="sm"
          mono
          aria-label={t("intelligence.chooseRoutineModel")}
          value={routineModel ?? "claude-haiku-4-5-20251001"}
          onChange={async (e) => {
            await setModelChoice(e.target.value, synthesisModel);
            queryClient.invalidateQueries({ queryKey: ["modelChoice"] });
            queryClient.invalidateQueries({ queryKey: ["resolvedRouting"] });
          }}
        >
          {ANTHROPIC_MODELS.map((m) => (
            <option key={m.id} value={m.id}>{m.label} - {t(m.descKey)}</option>
          ))}
        </Select>
      </div>
    </div>
  );
}

/** Synthesis-model half — see RoutineModelSelect's doc comment. */
export function SynthesisModelSelect() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: modelChoice } = useQuery({
    queryKey: ["modelChoice"],
    queryFn: getModelChoice,
  });
  const [routineModel, synthesisModel] = modelChoice ?? [null, null];

  return (
    <div className="flex items-center justify-between">
      <div>
        <div style={{ fontSize: "var(--mem-text-base)", fontWeight: 500, color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}>{t("intelligence.synthesisModel")}</div>
        <div style={{ fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)", fontFamily: "var(--mem-font-body)" }}>{t("intelligence.synthesisModelDescription")}</div>
      </div>
      <div className="shrink-0 w-fit">
        <Select
          size="sm"
          mono
          aria-label={t("intelligence.chooseSynthesisModel")}
          value={synthesisModel ?? "claude-sonnet-4-6"}
          onChange={async (e) => {
            await setModelChoice(routineModel, e.target.value);
            queryClient.invalidateQueries({ queryKey: ["modelChoice"] });
            queryClient.invalidateQueries({ queryKey: ["resolvedRouting"] });
          }}
        >
          {ANTHROPIC_MODELS.map((m) => (
            <option key={m.id} value={m.id}>{m.label} - {t(m.descKey)}</option>
          ))}
        </Select>
      </div>
    </div>
  );
}

export function ModelChoiceSection() {
  return (
    <div className="mt-4 pt-3" style={{ borderTop: "1px solid var(--mem-border)" }}>
      <div className="mb-3">
        <RoutineModelSelect />
      </div>
      <SynthesisModelSelect />
    </div>
  );
}

export function OnDeviceModelCard({
  deferDownload = false,
  onModelChosen,
  bare = false,
}: {
  /** Wizard step 2: record the choice, don't download here — step 5 does
   *  the download and proves it landed (`loaded === modelId`), not just
   *  that the request was sent. Settings (no props) keeps the immediate
   *  download this card has always done. */
  deferDownload?: boolean;
  onModelChosen?: (modelId: string | null) => void;
  /** Skip the <Card> wrapper and the title/status-chip header — the host
   *  (a disclosure row) already shows both. Default false keeps every
   *  existing call site unchanged. */
  bare?: boolean;
} = {}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [pickedId, setPickedId] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: modelData } = useQuery({
    queryKey: ["onDeviceModel"],
    queryFn: getOnDeviceModel,
  });
  const { data: systemInfo } = useQuery({
    queryKey: ["systemInfo"],
    queryFn: getSystemInfo,
  });

  const models = modelData?.models ?? [];
  const loadedId = modelData?.loaded ?? null;
  const selectedId = modelData?.selected ?? null;
  const currentId = pickedId ?? loadedId ?? selectedId ?? models[0]?.id ?? null;
  const current = currentId ? models.find((m) => m.id === currentId) : null;

  const isLoaded = !!current && loadedId === current.id;
  const needsDownload = !!current && !current.cached;
  const canLoad = !!current && current.cached && !isLoaded;
  const ramOk = systemInfo ? systemInfo.total_ram_gb + 0.5 >= (current?.ram_required_gb ?? 0) : true;

  // Report the current pick upward whenever it changes — mount (once the
  // catalog loads) and every dropdown change. A ref keeps this from
  // re-firing on every parent render just because it passed a fresh
  // closure.
  const onModelChosenRef = useRef(onModelChosen);
  useEffect(() => {
    onModelChosenRef.current = onModelChosen;
  }, [onModelChosen]);
  useEffect(() => {
    if (deferDownload) onModelChosenRef.current?.(currentId);
  }, [deferDownload, currentId]);

  const handleDownload = async () => {
    if (!current) return;
    setDownloading(true);
    setError(null);
    try {
      await downloadOnDeviceModel(current.id);
      queryClient.invalidateQueries({ queryKey: ["onDeviceModel"] });
      setPickedId(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setDownloading(false);
    }
  };

  const statusBadge = isLoaded ? t("intelligence.running") : t("intelligence.notLoaded");

  const body = (
      <div className="flex flex-col" style={{ gap: "10px" }}>
        {!bare && (
          <div className="flex items-center justify-between">
            <h3 style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-lg)", fontWeight: 600, color: "var(--mem-text)" }}>
              {t("intelligence.onDeviceModel")}
            </h3>
            <StatusChip state={isLoaded ? { kind: "up" } : { kind: "idle" }} label={statusBadge} />
          </div>
        )}
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "var(--mem-text-xs)",
            color: "var(--mem-text-tertiary)",
          }}
        >
          {t("intelligence.localModelDescription")}
          {systemInfo && (
            <>
              {" "}
              {t("intelligence.systemInfo", {
                ram: systemInfo.total_ram_gb.toFixed(0),
                accelerators: [
                  systemInfo.has_metal ? " · Metal GPU" : "",
                  systemInfo.has_cuda ? " · CUDA GPU" : "",
                ].join(""),
              })}
            </>
          )}
        </p>

        <div className="flex items-start gap-2">
          {current && (
            <span
              data-testid="on-device-model-spec"
              className="flex-1 min-w-0"
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "var(--mem-text-xs)",
                color: "var(--mem-text-secondary)",
              }}
            >
              {t("intelligence.modelDetails", {
                params: current.param_count,
                downloadSize: current.file_size_gb.toFixed(1),
                ram: current.ram_required_gb.toFixed(0),
              })}
              {current.cached && !isLoaded && t("intelligence.downloadedNotLoaded")}
              {!current.cached && t("intelligence.notDownloaded")}
            </span>
          )}

          {models.length > 0 ? (
            <div className="shrink-0 w-fit">
              <Select
                size="sm"
                mono
                aria-label={t("intelligence.chooseOnDeviceModel")}
                value={currentId ?? ""}
                onChange={(e) => setPickedId(e.target.value)}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name}
                  </option>
                ))}
              </Select>
            </div>
          ) : (
            <span
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "var(--mem-text-xs)",
                color: "var(--mem-text-tertiary)",
              }}
            >
              {t("intelligence.modelCatalogUnavailable")}
            </span>
          )}

          {needsDownload && !deferDownload && (
            <Button
              variant="primary"
              size="sm"
              onClick={handleDownload}
              disabled={downloading || !ramOk}
              loading={downloading}
              className="whitespace-nowrap shrink-0"
              title={!ramOk ? t("intelligence.notEnoughRamTitle") : undefined}
            >
              {t("intelligence.downloadSize", { size: current?.file_size_gb.toFixed(1) })}
            </Button>
          )}
          {canLoad && !deferDownload && (
            <Button
              variant="primary"
              size="sm"
              onClick={handleDownload}
              disabled={downloading}
              loading={downloading}
              className="whitespace-nowrap shrink-0"
            >
              {t("intelligence.load")}
            </Button>
          )}
        </div>

        {deferDownload && !isLoaded && current && (
          <p
            data-testid="on-device-model-deferred-note"
            style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}
          >
            {t("intelligence.willDownloadOnSetup")}
          </p>
        )}
        {downloading && !deferDownload && (
          <p
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "var(--mem-text-xs)",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {t("intelligence.downloadMayTake", { size: current?.file_size_gb.toFixed(1) })}
          </p>
        )}
        {error && (
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)" }}>
            {error}
          </p>
        )}
        {!ramOk && current && !downloading && (
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-warning-text)" }}>
            {t("intelligence.ramWarning", {
              model: current.display_name,
              required: current.ram_required_gb.toFixed(0),
              available: systemInfo?.total_ram_gb.toFixed(0),
            })}
          </p>
        )}
      </div>
  );

  return bare ? body : <Card padding="card">{body}</Card>;
}
