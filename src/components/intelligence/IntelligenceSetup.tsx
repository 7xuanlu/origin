import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  downloadOnDeviceModel,
  getApiKey,
  getModelChoice,
  getOnDeviceModel,
  getSystemInfo,
  setApiKey,
  setModelChoice,
} from "../../lib/tauri";

export const ANTHROPIC_MODELS = [
  { id: "claude-haiku-4-5-20251001", label: "Haiku 4.5", desc: "Fast, affordable" },
  { id: "claude-sonnet-4-6", label: "Sonnet 4.6", desc: "Balanced quality" },
  { id: "claude-opus-4-6", label: "Opus 4.6", desc: "Maximum quality" },
];

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

export function ApiKeyCard({
  showModelChoice = true,
  showNoKeyGuidance = true,
}: {
  showModelChoice?: boolean;
  showNoKeyGuidance?: boolean;
}) {
  const [keyInput, setKeyInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const { maskedKey, isConfigured } = useApiKeyStatus();

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await setApiKey(keyInput);
      setKeyInput("");
      queryClient.invalidateQueries({ queryKey: ["apiKey"] });
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
      queryClient.invalidateQueries({ queryKey: ["apiKey"] });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-[var(--mem-surface)] rounded-xl overflow-hidden border border-[var(--mem-border)]">
      <div className="px-5 py-4">
        <div className="flex items-center justify-between mb-1">
          <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text)" }}>
            Anthropic API Key
          </span>
          <span
            className="px-1.5 py-0.5 rounded text-[10px] font-medium"
            style={{
              fontFamily: "var(--mem-font-mono)",
              backgroundColor: isConfigured ? "rgba(34,197,94,0.1)" : "rgba(156,163,175,0.1)",
              color: isConfigured ? "rgb(34,197,94)" : "var(--mem-text-tertiary)",
            }}
          >
            {isConfigured ? "Connected" : "Not configured"}
          </span>
        </div>
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)", marginBottom: "10px" }}>
          Enables Claude Haiku for better memory titles and larger cluster distillation.
        </p>

        {isConfigured ? (
          <div className="flex items-center gap-2">
            <span
              className="flex-1 px-3 py-1.5 rounded-md"
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "12px",
                color: "var(--mem-text-secondary)",
                backgroundColor: "var(--mem-hover)",
              }}
            >
              {maskedKey}
            </span>
            <button
              onClick={handleClear}
              disabled={saving}
              className="px-2.5 py-1.5 rounded-md text-xs transition-colors hover:bg-[var(--mem-hover-strong)]"
              style={{ fontFamily: "var(--mem-font-body)", color: "var(--mem-text-tertiary)" }}
            >
              Clear
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <input
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && keyInput) handleSave(); }}
              placeholder="sk-ant-..."
              className="flex-1 px-3 py-1.5 rounded-md outline-none text-xs"
              style={{
                fontFamily: "var(--mem-font-mono)",
                backgroundColor: "var(--mem-hover)",
                border: "1px solid var(--mem-border)",
                color: "var(--mem-text)",
              }}
            />
            <button
              onClick={handleSave}
              disabled={saving || !keyInput}
              className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
              style={{
                backgroundColor: keyInput ? "var(--mem-accent-indigo)" : "var(--mem-hover)",
                color: keyInput ? "white" : "var(--mem-text-tertiary)",
              }}
            >
              {saving ? "..." : "Save"}
            </button>
          </div>
        )}

        {error && (
          <p className="mt-2" style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "rgb(239,68,68)" }}>
            {error}
          </p>
        )}

        {!isConfigured && showNoKeyGuidance && (
          <div
            className="mt-3 rounded-lg"
            style={{
              padding: "10px 14px",
              backgroundColor: "var(--mem-hover)",
              fontSize: "12px",
              color: "var(--mem-text-secondary)",
              lineHeight: 1.6,
            }}
          >
            <div style={{ fontWeight: 500, color: "var(--mem-text)", marginBottom: 2, fontSize: "12px" }}>
              Concept synthesis requires a cloud model
            </div>
            <div>Your memories are safe — search, recall, and entity linking work on-device.</div>
            <div style={{ marginTop: 6 }}>Add an Anthropic API key above to enable concept distillation.</div>
          </div>
        )}

        {isConfigured && showModelChoice && <ModelChoiceSection />}
      </div>
    </div>
  );
}

export function ModelChoiceSection() {
  const queryClient = useQueryClient();
  const { data: modelChoice } = useQuery({
    queryKey: ["modelChoice"],
    queryFn: getModelChoice,
  });
  const [routineModel, synthesisModel] = modelChoice ?? [null, null];

  const selectStyle: React.CSSProperties = {
    fontFamily: "var(--mem-font-mono)",
    fontSize: "11px",
    backgroundColor: "var(--mem-hover)",
    border: "1px solid var(--mem-border)",
    color: "var(--mem-text)",
    borderRadius: "6px",
    padding: "4px 8px",
    outline: "none",
  };

  return (
    <div className="mt-4 pt-3" style={{ borderTop: "1px solid var(--mem-border)" }}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <div style={{ fontSize: "13px", fontWeight: 500, color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}>Routine model</div>
          <div style={{ fontSize: "11px", color: "var(--mem-text-tertiary)", fontFamily: "var(--mem-font-body)" }}>Extraction, tagging, classification</div>
        </div>
        <select
          value={routineModel ?? "claude-haiku-4-5-20251001"}
          onChange={async (e) => {
            await setModelChoice(e.target.value, synthesisModel);
            queryClient.invalidateQueries({ queryKey: ["modelChoice"] });
          }}
          style={selectStyle}
        >
          {ANTHROPIC_MODELS.map((m) => (
            <option key={m.id} value={m.id}>{m.label} — {m.desc}</option>
          ))}
        </select>
      </div>
      <div className="flex items-center justify-between">
        <div>
          <div style={{ fontSize: "13px", fontWeight: 500, color: "var(--mem-text)", fontFamily: "var(--mem-font-body)" }}>Synthesis model</div>
          <div style={{ fontSize: "11px", color: "var(--mem-text-tertiary)", fontFamily: "var(--mem-font-body)" }}>Distillation, concepts, contradictions</div>
        </div>
        <select
          value={synthesisModel ?? "claude-sonnet-4-6"}
          onChange={async (e) => {
            await setModelChoice(routineModel, e.target.value);
            queryClient.invalidateQueries({ queryKey: ["modelChoice"] });
          }}
          style={selectStyle}
        >
          {ANTHROPIC_MODELS.map((m) => (
            <option key={m.id} value={m.id}>{m.label} — {m.desc}</option>
          ))}
        </select>
      </div>
    </div>
  );
}

export function OnDeviceModelCard() {
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

  const statusBadge = isLoaded ? "Running" : "Not loaded";
  const statusColor = isLoaded ? "rgb(34,197,94)" : "var(--mem-text-tertiary)";
  const statusBg = isLoaded ? "rgba(34,197,94,0.1)" : "rgba(156,163,175,0.1)";

  return (
    <section className="bg-[var(--mem-surface)] rounded-xl overflow-hidden border border-[var(--mem-border)]">
      <div className="px-5 py-4">
        <div className="flex items-center justify-between mb-1">
          <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "13px", color: "var(--mem-text)" }}>
            On-Device Model
          </span>
          <span
            className="px-1.5 py-0.5 rounded text-[10px] font-medium"
            style={{
              fontFamily: "var(--mem-font-mono)",
              backgroundColor: statusBg,
              color: statusColor,
            }}
          >
            {statusBadge}
          </span>
        </div>
        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "11px",
            color: "var(--mem-text-tertiary)",
            marginBottom: "10px",
          }}
        >
          Local LLM used for classification, entity extraction, and fallback distillation.
          {systemInfo && (
            <>
              {" "}Your system: {systemInfo.total_ram_gb.toFixed(0)}GB RAM
              {systemInfo.has_metal && " · Metal GPU"}
              {systemInfo.has_cuda && " · CUDA GPU"}.
            </>
          )}
        </p>

        <div className="flex items-center gap-2">
          {current && (
            <span
              className="flex-1 min-w-0 truncate"
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "11px",
                color: "var(--mem-text-secondary)",
              }}
            >
              {current.param_count} params · {current.file_size_gb.toFixed(1)}GB download · needs{" "}
              {current.ram_required_gb.toFixed(0)}GB RAM
              {current.cached && !isLoaded && " · downloaded (not loaded)"}
              {!current.cached && " · not downloaded"}
            </span>
          )}

          <select
            value={currentId ?? ""}
            onChange={(e) => setPickedId(e.target.value)}
            className="rounded-md outline-none"
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "11px",
              backgroundColor: "var(--mem-hover)",
              border: "1px solid var(--mem-border)",
              color: "var(--mem-text-secondary)",
              padding: "4px 8px",
              flexShrink: 0,
            }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.display_name}
              </option>
            ))}
          </select>

          {needsDownload && (
            <button
              onClick={handleDownload}
              disabled={downloading || !ramOk}
              className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
              style={{
                fontFamily: "var(--mem-font-body)",
                backgroundColor: ramOk ? "var(--mem-accent-indigo)" : "var(--mem-hover)",
                color: ramOk ? "white" : "var(--mem-text-tertiary)",
                whiteSpace: "nowrap",
                flexShrink: 0,
              }}
              title={!ramOk ? "Your system doesn't have enough RAM for this model" : undefined}
            >
              {downloading ? "Downloading..." : `Download ${current?.file_size_gb.toFixed(1)}GB`}
            </button>
          )}
          {canLoad && (
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
              style={{
                fontFamily: "var(--mem-font-body)",
                backgroundColor: "var(--mem-accent-indigo)",
                color: "white",
                whiteSpace: "nowrap",
                flexShrink: 0,
              }}
            >
              {downloading ? "Loading..." : "Load"}
            </button>
          )}
        </div>

        {downloading && (
          <p
            className="mt-2"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
            }}
          >
            This may take several minutes — the model is ~{current?.file_size_gb.toFixed(1)}GB.
          </p>
        )}
        {error && (
          <p
            className="mt-2"
            style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "rgb(239,68,68)" }}
          >
            {error}
          </p>
        )}
        {!ramOk && current && !downloading && (
          <p
            className="mt-2"
            style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "rgb(234,179,8)" }}
          >
            {current.display_name} needs {current.ram_required_gb.toFixed(0)}GB RAM, but your system has only{" "}
            {systemInfo?.total_ram_gb.toFixed(0)}GB.
          </p>
        )}
      </div>
    </section>
  );
}
