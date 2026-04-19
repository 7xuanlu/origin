import { useState, useEffect, useCallback, useRef } from "react";
import { DropZone } from "./DropZone";
import {
  importChatExport,
  saveTempFile,
  listPendingImports,
  importStageLabel,
  type ImportChatExportResponse,
  type PendingImport,
} from "../../lib/tauri";

/** What the user just triggered in this session. */
type LocalAction =
  | null
  | { kind: "reading" }
  | { kind: "done"; result: ImportChatExportResponse }
  | { kind: "error"; message: string };

const POLL_INTERVAL_MS = 5_000;

async function maybeNotify(title: string, body: string) {
  try {
    const { sendNotification, isPermissionGranted, requestPermission } =
      await import("@tauri-apps/plugin-notification");
    let granted = await isPermissionGranted();
    if (!granted) {
      const result = await requestPermission();
      granted = result === "granted";
    }
    if (granted) {
      await sendNotification({ title, body });
    }
  } catch {
    // Plugin unavailable or not initialized
  }
}

export function ImportFlow() {
  const [localAction, setLocalAction] = useState<LocalAction>(null);
  const [pending, setPending] = useState<PendingImport | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const prevPendingRef = useRef<PendingImport | null>(null);

  // Poll daemon for actual import state. Survives page switches because the
  // daemon is the source of truth, not React state.
  useEffect(() => {
    let alive = true;
    const poll = () => {
      listPendingImports()
        .then((imports) => {
          if (!alive) return;
          if (imports.length > 0) {
            setPending(imports[0]);
            setDismissed(false);
          } else if (prevPendingRef.current) {
            // Was pending, now done
            setPending(null);
            maybeNotify("Origin", "Import refinement complete");
          } else {
            setPending(null);
          }
          prevPendingRef.current = imports[0] ?? null;
        })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const runImport = useCallback(async (path: string) => {
    setLocalAction({ kind: "reading" });
    setDismissed(false);
    try {
      const result = await importChatExport(path);
      setLocalAction({ kind: "done", result });
      const msg = result.conversations_new > 0
        ? `Imported ${result.conversations_new} conversations (${result.memories_stored} memories) from ${result.vendor}`
        : `All ${result.conversations_total} conversations already imported`;
      maybeNotify("Origin", msg);
    } catch (e: any) {
      setLocalAction({ kind: "error", message: String(e) });
    }
  }, []);

  const handleFileSelected = useCallback(async (file: File) => {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const tempPath = await saveTempFile(bytes, file.name);
    await runImport(tempPath);
  }, [runImport]);

  const handlePathSelected = useCallback(async (path: string) => {
    await runImport(path);
  }, [runImport]);

  // Derive display state from local action + daemon state
  const isRefining = pending !== null;
  const showLocal = localAction !== null && !dismissed;
  const showRefining = isRefining && !dismissed;
  const showStrip = showLocal || showRefining;

  return (
    <div>
      <DropZone
        onFileSelected={handleFileSelected}
        onPathSelected={handlePathSelected}
      />

      {showStrip && (
        <div
          style={{
            marginTop: 10,
            borderRadius: 8,
            padding: "8px 12px",
            fontFamily: "var(--mem-font-body)",
            fontSize: 12,
            lineHeight: "1.5",
            display: "flex",
            alignItems: "center",
            gap: 8,
            background: localAction?.kind === "error"
              ? "color-mix(in srgb, #ef4444 8%, transparent)"
              : "color-mix(in srgb, var(--mem-accent-indigo) 8%, transparent)",
            border: `1px solid ${
              localAction?.kind === "error"
                ? "color-mix(in srgb, #ef4444 20%, transparent)"
                : "color-mix(in srgb, var(--mem-accent-indigo) 16%, transparent)"
            }`,
            transition: "all 0.2s ease",
          }}
        >
          <StatusIcon
            reading={localAction?.kind === "reading"}
            error={localAction?.kind === "error"}
            refining={isRefining}
          />

          <span style={{
            flex: 1,
            color: localAction?.kind === "error" ? "#ef4444" : "var(--mem-text-secondary)",
          }}>
            {localAction?.kind === "reading" && "Importing conversations..."}
            {localAction?.kind === "done" && formatDoneMessage(localAction.result, isRefining, pending)}
            {localAction?.kind === "error" && localAction.message}
            {!localAction && isRefining && formatRefiningMessage(pending)}
          </span>

          {localAction?.kind !== "reading" && (
            <button
              onClick={() => { setDismissed(true); setLocalAction(null); }}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                color: "var(--mem-text-tertiary)",
                padding: 2,
                lineHeight: 0,
                flexShrink: 0,
              }}
              aria-label="Dismiss"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M3 3L9 9M9 3L3 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </button>
          )}
        </div>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}

function StatusIcon({ reading, error, refining }: { reading?: boolean; error?: boolean; refining: boolean }) {
  if (reading || refining) {
    return (
      <span style={{
        width: 6, height: 6, borderRadius: "50%",
        background: "var(--mem-accent-indigo)",
        animation: "pulse 1.2s ease-in-out infinite",
        flexShrink: 0,
      }} />
    );
  }
  if (error) {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0 }}>
        <circle cx="7" cy="7" r="6" stroke="#ef4444" strokeWidth="1.5" />
        <path d="M5 5L9 9M9 5L5 9" stroke="#ef4444" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0 }}>
      <circle cx="7" cy="7" r="6" stroke="var(--mem-accent-sage)" strokeWidth="1.5" />
      <path d="M4.5 7L6.5 9L9.5 5" stroke="var(--mem-accent-sage)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function formatDoneMessage(r: ImportChatExportResponse, refining: boolean, pending: PendingImport | null): string {
  const stageSuffix = refining && pending
    ? ` \u00B7 ${importStageLabel(pending.stage).toLowerCase()} in background`
    : "";
  if (r.conversations_new > 0) {
    return `${r.conversations_new} conversations imported from ${r.vendor}${stageSuffix}`;
  }
  return `All ${r.conversations_total} conversations already imported${stageSuffix}`;
}

function formatRefiningMessage(p: PendingImport | null): string {
  if (!p) return "";
  const count = p.total_conversations ?? 0;
  const label = importStageLabel(p.stage).toLowerCase();
  return `${count} ${p.vendor} conversations \u00B7 ${label}`;
}
