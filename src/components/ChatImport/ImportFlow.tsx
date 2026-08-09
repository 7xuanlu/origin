import { useState, useEffect, useCallback, useRef } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { DropZone } from "./DropZone";
import {
  importChatExport,
  saveTempFile,
  listPendingImports,
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
  const { t } = useTranslation();
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
            maybeNotify("Wenlan", t("chatImport.importFlow.refinementComplete"));
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
  }, [t]);

  const runImport = useCallback(async (path: string) => {
    setLocalAction({ kind: "reading" });
    setDismissed(false);
    try {
      const result = await importChatExport(path);
      setLocalAction({ kind: "done", result });
      const msg = result.conversations_new > 0
        ? t("chatImport.importFlow.notificationImported", {
            conversations: result.conversations_new,
            memories: result.memories_stored,
            vendor: result.vendor,
          })
        : t("chatImport.importFlow.notificationAlreadyImported", {
            count: result.conversations_total,
          });
      maybeNotify("Wenlan", msg);
    } catch (e: any) {
      setLocalAction({ kind: "error", message: String(e) });
    }
  }, [t]);

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
            fontSize: "var(--mem-text-sm)",
            lineHeight: "1.5",
            display: "flex",
            alignItems: "center",
            gap: 8,
            background: localAction?.kind === "error"
              ? "var(--mem-status-danger-bg)"
              : "color-mix(in srgb, var(--mem-accent-indigo) 8%, transparent)",
            border: `1px solid ${
              localAction?.kind === "error"
                ? "var(--mem-status-danger-border)"
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
            color: localAction?.kind === "error" ? "var(--mem-status-danger-text)" : "var(--mem-text-secondary)",
          }}>
            {localAction?.kind === "reading" && t("chatImport.importFlow.importing")}
            {localAction?.kind === "done" && formatDoneMessage(t, localAction.result, isRefining, pending)}
            {localAction?.kind === "error" && localAction.message}
            {!localAction && isRefining && formatRefiningMessage(t, pending)}
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
              aria-label={t("chatImport.importFlow.dismiss")}
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                <path d="M3 3L9 9M9 3L3 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </button>
          )}
        </div>
      )}

    </div>
  );
}

function StatusIcon({ reading, error, refining }: { reading?: boolean; error?: boolean; refining: boolean }) {
  if (reading || refining) {
    return (
      <span
        className="mem-node-pulse"
        style={{
          width: 6, height: 6, borderRadius: "50%",
          background: "var(--mem-accent-indigo)",
          flexShrink: 0,
        }}
      />
    );
  }
  if (error) {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0 }} aria-hidden="true">
        <circle cx="7" cy="7" r="6" stroke="var(--mem-status-danger-text)" strokeWidth="1.5" />
        <path d="M5 5L9 9M9 5L5 9" stroke="var(--mem-status-danger-text)" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0 }} aria-hidden="true">
      <circle cx="7" cy="7" r="6" stroke="var(--mem-accent-sage)" strokeWidth="1.5" />
      <path d="M4.5 7L6.5 9L9.5 5" stroke="var(--mem-accent-sage)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function importStageText(t: TFunction, stage: string): string {
  switch (stage) {
    case "parsing":
      return t("chatImport.stage.parsing");
    case "stage_a":
      return t("chatImport.stage.stageA");
    case "stage_b":
      return t("chatImport.stage.stageB");
    case "done":
      return t("chatImport.stage.done");
    case "error":
      return t("chatImport.stage.error");
    default:
      return stage;
  }
}

function formatDoneMessage(
  t: TFunction,
  r: ImportChatExportResponse,
  refining: boolean,
  pending: PendingImport | null,
): string {
  const stageSuffix = refining && pending
    ? t("chatImport.importFlow.stageInBackground", {
        stage: importStageText(t, pending.stage),
      })
    : "";
  if (r.conversations_new > 0) {
    return t("chatImport.importFlow.conversationsImported", {
      count: r.conversations_new,
      vendor: r.vendor,
      stageSuffix,
    });
  }
  return t("chatImport.importFlow.allAlreadyImported", {
    count: r.conversations_total,
    stageSuffix,
  });
}

function formatRefiningMessage(t: TFunction, p: PendingImport | null): string {
  if (!p) return "";
  const count = p.total_conversations ?? 0;
  return t("chatImport.importFlow.refining", {
    count,
    vendor: p.vendor,
    stage: importStageText(t, p.stage),
  });
}
