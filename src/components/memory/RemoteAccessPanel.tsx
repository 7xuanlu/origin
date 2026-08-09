import { useEffect, useId, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { listen } from "@tauri-apps/api/event";
import {
  clipboardWrite,
  getRemoteAccessStatus,
  getWireState,
  installClientPlugin,
  testRemoteMcpConnection,
  toggleRemoteAccess,
  type RemoteAccessStatus,
  type RemoteConnectionTest,
} from "../../lib/tauri";
import { Button, Card, StatusChip, Tag, Toggle, WarningTriangleIcon } from "./settings/primitives";

const REMOTE_QUERY_KEY = ["remote-access-status"] as const;

/** Web access — the one state-aware surface for reaching memory from
 *  claude.ai and ChatGPT. Turning the toggle on IS the setup: the relay needs
 *  nothing configured. Below the toggle, one row per web platform reflects
 *  where each stands — Claude.ai is Ready once its connector is installed
 *  (a one-click Set up otherwise); ChatGPT gets its paste-in URL once web
 *  access is on. The always-visible no-auth warning is the one load-bearing
 *  boundary. Reads `RemoteAccessStatus` via React Query and invalidates on
 *  `remote-access-status` events. */
export function RemoteAccessPanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const warningId = `${useId()}-remote-access-warning`;

  const { data: status = { status: "off" } as RemoteAccessStatus } = useQuery({
    queryKey: REMOTE_QUERY_KEY,
    queryFn: getRemoteAccessStatus,
    staleTime: 30_000,
    refetchInterval: false,
  });

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listen<RemoteAccessStatus>("remote-access-status", (event) => {
      queryClient.setQueryData(REMOTE_QUERY_KEY, event.payload);
    }).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, [queryClient]);

  const toggleMut = useMutation({
    mutationFn: (enabled: boolean) => toggleRemoteAccess(enabled),
    onSuccess: (next) => {
      queryClient.setQueryData(REMOTE_QUERY_KEY, next);
    },
  });

  const [testResult, setTestResult] = useState<
    | { kind: "idle" }
    | { kind: "running" }
    | { kind: "ok"; latency_ms: number | null }
    | { kind: "err"; error: string }
  >({ kind: "idle" });

  const testOkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (testOkTimerRef.current) clearTimeout(testOkTimerRef.current);
    };
  }, []);

  const testMut = useMutation({
    mutationFn: testRemoteMcpConnection,
    onMutate: () => {
      setTestResult({ kind: "running" });
    },
    onSuccess: (result: RemoteConnectionTest) => {
      if (result.ok) {
        setTestResult({ kind: "ok", latency_ms: result.latency_ms });
        if (testOkTimerRef.current) clearTimeout(testOkTimerRef.current);
        testOkTimerRef.current = setTimeout(() => setTestResult({ kind: "idle" }), 2000);
      } else {
        setTestResult({ kind: "err", error: result.error ?? "Unknown error" });
      }
    },
    onError: (err) => {
      setTestResult({ kind: "err", error: String(err) });
    },
  });

  const isOn = status.status === "connected" || status.status === "starting";

  const handleReconnect = () => {
    toggleMut.mutate(false);
    setTimeout(() => toggleMut.mutate(true), 500);
  };

  return (
    <Card padding="none">
      {/* Toggle row — turning it on IS the setup; the relay needs nothing
          configured. */}
      <div className="px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "var(--mem-text-lg)",
              fontWeight: 600,
              color: "var(--mem-text)",
            }}
          >
            {t("remoteAccess.title")}
          </div>
          <div className="mt-0.5">
            <Toggle
              enabled={isOn}
              onToggle={() => toggleMut.mutate(!isOn)}
              aria-label={t("remoteAccess.title")}
              aria-describedby={warningId}
            />
          </div>
        </div>

        <p
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "var(--mem-text-sm)",
            color: "var(--mem-text-secondary)",
            lineHeight: "1.5",
            marginTop: "6px",
          }}
        >
          {t("remoteAccess.description")}
        </p>

        {/* No-auth warning — the single, louder surviving rendering (was 3
            across this panel + WebPlatformCards ×2, now exactly one). Always
            visible, never behind a disclosure, wired to the toggle via
            aria-describedby so a screen reader hears the boundary at the
            moment of toggling. */}
        <div className="flex items-start gap-2 mt-2">
          <WarningTriangleIcon className="w-3.5 h-3.5 text-[var(--mem-status-warning-text)] shrink-0 mt-px" />
          <p
            id={warningId}
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "var(--mem-text-sm)",
              color: "var(--mem-status-warning-text)",
              lineHeight: "1.5",
            }}
          >
            {t("remoteAccess.noAuthWarning")}
          </p>
        </div>
      </div>

      {/* Status row — chip, plus Test connection + Reconnect once connected.
          The relay URL is not here: its one home is the ChatGPT row below. */}
      <div className="px-5 pb-4">
        <StatusRow
          status={status}
          testResult={testResult}
          onTest={() => testMut.mutate()}
          onReconnect={handleReconnect}
          reconnecting={toggleMut.isPending}
        />
      </div>

      {/* Error / disabled reconnect */}
      {status.status === "error" && (
        <div className="px-5 pb-4 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <Button variant="secondary" size="sm" onClick={() => toggleMut.mutate(true)}>
              {t("remoteAccess.retry")}
            </Button>
            <Button variant="secondary" size="sm" onClick={handleReconnect}>
              {t("remoteAccess.reconnect")}
            </Button>
          </div>
        </div>
      )}

      {/* Claude.ai — Ready once the connector is installed, one-click Set up
          otherwise. Independent of the toggle (it's the connector, not the
          relay). */}
      <div className="border-t px-5 py-4" style={{ borderColor: "var(--mem-border)" }}>
        <ClaudeRow />
      </div>

      {/* ChatGPT — the one home for the paste-in relay URL, shown once web
          access is on. */}
      <div className="border-t px-5 py-4" style={{ borderColor: "var(--mem-border)" }}>
        <ChatgptRow status={status} />
      </div>
    </Card>
  );
}

type TestResult =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "ok"; latency_ms: number | null }
  | { kind: "err"; error: string };

function StatusRow({
  status,
  testResult,
  onTest,
  onReconnect,
  reconnecting,
}: {
  status: RemoteAccessStatus;
  testResult: TestResult;
  onTest: () => void;
  onReconnect: () => void;
  reconnecting: boolean;
}) {
  const { t } = useTranslation();
  // "off" is a setting the user chose, not something the app probed — the
  // chip-never-lies invariant means a chip's color may only come from an
  // observation, so off gets no chip at all (the Toggle already says it).
  if (status.status === "off") return null;
  if (status.status === "starting") {
    return <StatusChip state={{ kind: "probing" }} label={t("remoteAccess.statusConnecting")} />;
  }
  if (status.status === "error") {
    // The verbatim daemon error carries the chip; there is no honest constant
    // word to put beside it, so it stands alone as the label.
    return <StatusChip state={{ kind: "down" }} label={status.error} />;
  }
  // connected
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <StatusChip state={{ kind: "up" }} label={t("remoteAccess.statusConnected")} />
      <Button
        variant="secondary"
        size="sm"
        loading={testResult.kind === "running"}
        onClick={onTest}
      >
        {testResult.kind === "running"
          ? t("remoteAccess.testing")
          : t("remoteAccess.testConnection")}
      </Button>
      <Button variant="secondary" size="sm" loading={reconnecting} onClick={onReconnect}>
        {t("remoteAccess.reconnect")}
      </Button>
      {testResult.kind === "ok" && (
        <span
          className="inline-flex items-center gap-1"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "var(--mem-text-sm)",
            color: "var(--mem-accent-sage)",
          }}
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          {t("remoteAccess.statusConnectedLatency", { ms: testResult.latency_ms ?? "?" })}
        </span>
      )}
      {testResult.kind === "err" && (
        <span
          className="inline-flex items-center gap-1"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "var(--mem-text-sm)",
            color: "var(--mem-status-danger-text)",
          }}
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
          {testResult.error}
        </span>
      )}
    </div>
  );
}

const rowHeading = (text: string) => (
  <h3 style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-lg)", fontWeight: 600, color: "var(--mem-text)", margin: 0 }}>
    {text}
  </h3>
);

/** Claude.ai row: reads the real, resolved wiring. `has_plugin` true means the
 *  connector is installed and memory flows through the relay while web access
 *  is on — nothing to do. Otherwise a one-click Set up (idempotent plugin
 *  install) plus a manual fallback. When the wire query fails we can't tell
 *  whether the plugin exists, so we offer only the manual steps — never a
 *  one-click install against unknown state (it could double-register). */
function ClaudeRow() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState("");

  const { data: wire, isError } = useQuery({ queryKey: ["wireState"], queryFn: getWireState });
  const hasPlugin = wire?.clients.find((c) => c.client_type === "claude_code")?.has_plugin ?? false;

  const install = async () => {
    setInstalling(true);
    setError("");
    try {
      await installClientPlugin("claude_code");
      queryClient.invalidateQueries({ queryKey: ["wireState"] });
    } catch (err) {
      setError(String(err));
    } finally {
      setInstalling(false);
    }
  };

  if (hasPlugin) {
    return (
      <div className="flex flex-col" style={{ gap: "8px" }}>
        <div className="flex items-center justify-between gap-2">
          {rowHeading(t("connectMatrix.claudeTitle"))}
          <StatusChip state={{ kind: "up" }} label={t("connectMatrix.claudeReady")} />
        </div>
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", lineHeight: 1.5, margin: 0 }}>
          {t("connectMatrix.claudeReadyBody")}
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col" style={{ gap: "10px" }}>
      <div className="flex items-center justify-between gap-2">
        {rowHeading(t("connectMatrix.claudeTitle"))}
        <Tag tone="neutral">{t("intelligence.notConfigured")}</Tag>
      </div>
      {!isError && (
        <div className="flex flex-col gap-1.5">
          <div>
            <Button variant="secondary" size="sm" onClick={install} disabled={installing}>
              {installing ? t("connectMatrix.settingUp") : t("connectMatrix.setUp")}
            </Button>
          </div>
          {error && (
            <p role="alert" style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)", margin: 0 }}>
              {error}
            </p>
          )}
        </div>
      )}
      <details className="group">
        <summary
          className="inline-flex cursor-pointer list-none items-center gap-1.5 rounded-[var(--mem-radius-sm)] py-0.5 [&::-webkit-details-marker]:hidden focus-visible:outline-2 focus-visible:outline-[var(--mem-focus-ring)] focus-visible:outline-offset-2"
          style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)" }}
        >
          <svg
            aria-hidden="true"
            className="h-3 w-3 shrink-0 transition-transform duration-[var(--mem-dur-fast)] group-open:rotate-90"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          {t("connectMatrix.setUpManually")}
        </summary>
        <div className="flex flex-col gap-2 pt-2">
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", fontWeight: 600, color: "var(--mem-text)", margin: 0 }}>
            {t("connectMatrix.claudePluginStepTitle")}
          </p>
          <ol style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", lineHeight: 1.7, paddingLeft: "18px", listStyle: "decimal", margin: 0 }}>
            <li>{t("connectMatrix.claudePluginStep1")}</li>
            <li>{t("connectMatrix.claudePluginStep2")}</li>
            <li>{t("connectMatrix.claudePluginStep3")}</li>
          </ol>
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)", lineHeight: 1.5, margin: 0 }}>
            {t("connectMatrix.claudePluginNote")}
          </p>
        </div>
      </details>
    </div>
  );
}

/** ChatGPT row: the single home for the paste-in relay URL. Only meaningful
 *  once web access is connected; otherwise a one-line prompt to turn it on. */
function ChatgptRow({ status }: { status: RemoteAccessStatus }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  if (status.status !== "connected") {
    return (
      <div className="flex flex-col" style={{ gap: "8px" }}>
        {rowHeading(t("connectMatrix.chatgptTitle"))}
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)", margin: 0 }}>
          {t("connectMatrix.chatgptNeedsWebAccess")}
        </p>
      </div>
    );
  }

  const url = status.relay_url ?? `${status.tunnel_url}/mcp`;
  const copy = () => {
    clipboardWrite(url);
    setCopied(true);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col" style={{ gap: "10px" }}>
      {rowHeading(t("connectMatrix.chatgptTitle"))}
      <ol style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", lineHeight: 1.7, paddingLeft: "18px", listStyle: "decimal", margin: 0 }}>
        <li>{t("connectMatrix.chatgptStep1")}</li>
        <li>{t("connectMatrix.chatgptStep2")}</li>
        <li>{t("connectMatrix.chatgptStep3")}</li>
      </ol>
      <div className="flex items-center gap-2">
        <code
          className="flex-1 truncate rounded-md px-2 py-1.5"
          style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-xs)", backgroundColor: "var(--mem-bg)", border: "1px solid var(--mem-border)", color: "var(--mem-text)" }}
        >
          {url}
        </code>
        <Button type="button" variant="secondary" size="sm" onClick={copy} className="shrink-0">
          {copied ? t("connectMatrix.copied") : t("connectMatrix.copyUrl")}
        </Button>
      </div>
      <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)", lineHeight: 1.6, margin: 0 }}>
        {t("remoteAccess.tunnelChangesNote")}
      </p>
    </div>
  );
}
