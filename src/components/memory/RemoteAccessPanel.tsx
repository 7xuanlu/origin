import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import {
  clipboardWrite,
  getRemoteAccessStatus,
  rotateRemoteToken,
  testRemoteMcpConnection,
  toggleRemoteAccess,
  type RemoteAccessStatus,
  type RemoteConnectionTest,
} from "../../lib/tauri";

interface Props {
  mode: "compact" | "full";
}

const REMOTE_QUERY_KEY = ["remote-access-status"] as const;

/** Shared Remote Access control surface used by the onboarding wizard
 *  (compact mode) and the full Settings page (full mode). Owns the toggle,
 *  status indicator, URL copy affordance, test-connection probe, setup
 *  instructions, and — in full mode — the token section plus tunnel-behavior
 *  footnote. Reads `RemoteAccessStatus` via React Query and invalidates on
 *  `remote-access-status` events so both consumers stay in sync. */
export function RemoteAccessPanel({ mode }: Props) {
  const queryClient = useQueryClient();

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

  const rotateMut = useMutation({
    mutationFn: rotateRemoteToken,
    onSuccess: () => {
      queryClient.setQueryData(REMOTE_QUERY_KEY, {
        status: "starting",
      } as RemoteAccessStatus);
    },
  });

  const [urlCopied, setUrlCopied] = useState(false);
  const [tokenCopied, setTokenCopied] = useState(false);
  const [tokenVisible, setTokenVisible] = useState(false);
  const [confirmRotate, setConfirmRotate] = useState(false);
  const [showInstructions, setShowInstructions] = useState(false);
  const [testResult, setTestResult] = useState<
    | { kind: "idle" }
    | { kind: "running" }
    | { kind: "ok"; latency_ms: number | null }
    | { kind: "err"; error: string }
  >({ kind: "idle" });

  const testOkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const urlCopyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tokenCopyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rotateConfirmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (testOkTimerRef.current) clearTimeout(testOkTimerRef.current);
      if (urlCopyTimerRef.current) clearTimeout(urlCopyTimerRef.current);
      if (tokenCopyTimerRef.current) clearTimeout(tokenCopyTimerRef.current);
      if (rotateConfirmTimerRef.current) clearTimeout(rotateConfirmTimerRef.current);
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
        testOkTimerRef.current = setTimeout(
          () => setTestResult({ kind: "idle" }),
          2000,
        );
      } else {
        setTestResult({ kind: "err", error: result.error ?? "Unknown error" });
      }
    },
    onError: (err) => {
      setTestResult({ kind: "err", error: String(err) });
    },
  });

  const isOn = status.status === "connected" || status.status === "starting";
  const displayUrl =
    status.status === "connected"
      ? status.relay_url ?? `${status.tunnel_url}/mcp`
      : "";

  const handleCopyUrl = () => {
    clipboardWrite(displayUrl);
    setUrlCopied(true);
    if (urlCopyTimerRef.current) clearTimeout(urlCopyTimerRef.current);
    urlCopyTimerRef.current = setTimeout(() => setUrlCopied(false), 2000);
  };

  const handleCopyToken = (token: string) => {
    clipboardWrite(token);
    setTokenCopied(true);
    if (tokenCopyTimerRef.current) clearTimeout(tokenCopyTimerRef.current);
    tokenCopyTimerRef.current = setTimeout(() => setTokenCopied(false), 2000);
  };

  const handleRotate = () => {
    if (!confirmRotate) {
      setConfirmRotate(true);
      if (rotateConfirmTimerRef.current) clearTimeout(rotateConfirmTimerRef.current);
      rotateConfirmTimerRef.current = setTimeout(
        () => setConfirmRotate(false),
        3000,
      );
      return;
    }
    if (rotateConfirmTimerRef.current) clearTimeout(rotateConfirmTimerRef.current);
    setConfirmRotate(false);
    rotateMut.mutate();
  };

  const handleReconnect = () => {
    toggleMut.mutate(false);
    setTimeout(() => toggleMut.mutate(true), 500);
  };

  return (
    <div
      className="rounded-xl"
      style={{
        backgroundColor: "var(--mem-surface)",
        border: "1px solid var(--mem-border)",
      }}
    >
      {/* Toggle row */}
      <div className="px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "14px",
                fontWeight: 500,
                color: "var(--mem-text)",
              }}
            >
              Share with web-based AI tools
            </div>
            <p
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "12px",
                color: "var(--mem-text-tertiary)",
                marginTop: "2px",
                lineHeight: "1.5",
              }}
            >
              Let Claude.ai and ChatGPT access your memories via a secure tunnel.
            </p>
          </div>
          <div className="mt-0.5">
            <button
              role="switch"
              aria-checked={isOn}
              onClick={() => toggleMut.mutate(!isOn)}
              className={`relative w-11 h-[26px] rounded-full transition-colors shrink-0 ${
                isOn ? "bg-[var(--mem-accent-indigo)]" : "bg-[var(--mem-hover-strong)]"
              }`}
            >
              <span
                className={`absolute top-[3px] w-5 h-5 rounded-full bg-white shadow-sm transition-transform ${
                  isOn ? "left-[22px]" : "left-[3px]"
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Status row */}
      <div className="px-5 pb-4">
        <StatusRow status={status} />
      </div>

      {/* Connected: URL + test connection */}
      {status.status === "connected" && (
        <div className="px-5 pb-4 space-y-3">
          <div>
            <label
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "11px",
                fontWeight: 500,
                color: "var(--mem-text-tertiary)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              {status.relay_url ? "Your MCP URL (stable)" : "Your MCP URL"}
            </label>
            <div className="flex items-center gap-2 mt-1">
              <pre
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "12px",
                  color: "var(--mem-text)",
                  backgroundColor: "var(--mem-hover)",
                  padding: "6px 8px",
                  borderRadius: "6px",
                  flex: 1,
                  margin: 0,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                }}
              >
                {displayUrl}
              </pre>
              <button
                aria-label="Copy URL"
                onClick={handleCopyUrl}
                className="px-2 py-1 rounded text-xs font-medium transition-colors shrink-0"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  color: urlCopied ? "var(--mem-accent-warm)" : "var(--mem-accent-indigo)",
                  backgroundColor: urlCopied
                    ? "rgba(251, 191, 36, 0.15)"
                    : "rgba(123, 123, 232, 0.1)",
                }}
              >
                {urlCopied ? "Copied!" : "Copy URL"}
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => testMut.mutate()}
              disabled={testResult.kind === "running"}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors"
              style={{
                fontFamily: "var(--mem-font-body)",
                color: "var(--mem-text-secondary)",
                backgroundColor: "var(--mem-hover)",
              }}
            >
              {testResult.kind === "running" ? (
                <>
                  <svg
                    className="w-3 h-3 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <circle
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="3"
                      opacity="0.25"
                    />
                    <path
                      d="M4 12a8 8 0 018-8"
                      stroke="currentColor"
                      strokeWidth="3"
                      strokeLinecap="round"
                    />
                  </svg>
                  <span>Testing…</span>
                </>
              ) : (
                <>Test connection</>
              )}
            </button>
            {testResult.kind === "ok" && (
              <span
                className="inline-flex items-center gap-1"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  color: "var(--mem-accent-sage)",
                }}
              >
                <svg
                  className="w-3.5 h-3.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
                Connected ({testResult.latency_ms ?? "?"}ms)
              </span>
            )}
            {testResult.kind === "err" && (
              <span
                className="inline-flex items-center gap-1"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "12px",
                  color: "#ef4444",
                }}
              >
                <svg
                  className="w-3.5 h-3.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
                {testResult.error}
              </span>
            )}
          </div>

          {mode === "full" && (
            <TokenRow
              token={status.token}
              visible={tokenVisible}
              onToggleVisible={() => setTokenVisible((v) => !v)}
              copied={tokenCopied}
              onCopy={() => handleCopyToken(status.token)}
              confirmRotate={confirmRotate}
              onRotate={handleRotate}
            />
          )}

          <InstructionsBlock
            expanded={showInstructions}
            onToggle={() => setShowInstructions((v) => !v)}
          />

          {mode === "full" && (
            <div className="flex items-center gap-3 pt-1">
              <button
                onClick={handleReconnect}
                disabled={toggleMut.isPending}
                className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  color: "var(--mem-text-secondary)",
                  backgroundColor: "var(--mem-hover)",
                  opacity: toggleMut.isPending ? 0.5 : 1,
                }}
              >
                Reconnect
              </button>
              <p
                style={{
                  fontFamily: "var(--mem-font-body)",
                  fontSize: "11px",
                  color: "var(--mem-text-tertiary)",
                  lineHeight: "1.6",
                  margin: 0,
                }}
              >
                {status.relay_url
                  ? "This URL is stable — it won't change when your Mac sleeps or restarts."
                  : "This tunnel URL changes when your Mac sleeps or restarts. Enable a stable relay in Settings → Agents to avoid reconnecting."}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Error / disabled reconnect (full mode shows reconnect button) */}
      {status.status === "error" && (
        <div className="px-5 pb-4 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={() => toggleMut.mutate(true)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
              style={{
                fontFamily: "var(--mem-font-body)",
                backgroundColor: "var(--mem-accent-indigo)",
                color: "white",
              }}
            >
              Retry
            </button>
            {mode === "full" && (
              <button
                onClick={handleReconnect}
                className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
                style={{
                  fontFamily: "var(--mem-font-body)",
                  color: "var(--mem-text-secondary)",
                  backgroundColor: "var(--mem-hover)",
                }}
              >
                Reconnect
              </button>
            )}
          </div>
        </div>
      )}

      {mode === "full" && status.status === "off" && (
        <div className="px-5 pb-4">
          <button
            onClick={handleReconnect}
            className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
            style={{
              fontFamily: "var(--mem-font-body)",
              color: "var(--mem-text-secondary)",
              backgroundColor: "var(--mem-hover)",
            }}
          >
            Reconnect
          </button>
        </div>
      )}
    </div>
  );
}

function StatusRow({ status }: { status: RemoteAccessStatus }) {
  if (status.status === "off") {
    return (
      <span
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "13px",
          color: "var(--mem-text-secondary)",
        }}
      >
        Off
      </span>
    );
  }
  if (status.status === "starting") {
    return (
      <span className="inline-flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-[var(--mem-accent-amber)] animate-pulse" />
        <span
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: "var(--mem-text-secondary)",
          }}
        >
          Connecting…
        </span>
      </span>
    );
  }
  if (status.status === "connected") {
    return (
      <span className="inline-flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-[var(--mem-accent-sage)]" />
        <span
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            color: "var(--mem-accent-sage)",
          }}
        >
          Connected
        </span>
      </span>
    );
  }
  // error
  return (
    <span className="inline-flex items-center gap-2">
      <span className="w-2 h-2 rounded-full bg-red-500" />
      <span
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "13px",
          color: "#ef4444",
          wordBreak: "break-word",
        }}
      >
        {status.error}
      </span>
    </span>
  );
}

function TokenRow({
  token,
  visible,
  onToggleVisible,
  copied,
  onCopy,
  confirmRotate,
  onRotate,
}: {
  token: string;
  visible: boolean;
  onToggleVisible: () => void;
  copied: boolean;
  onCopy: () => void;
  confirmRotate: boolean;
  onRotate: () => void;
}) {
  return (
    <div>
      <label
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "11px",
          fontWeight: 500,
          color: "var(--mem-text-tertiary)",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        Token
      </label>
      <div className="flex items-center gap-2 mt-1">
        <code
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "12px",
            color: "var(--mem-text)",
            backgroundColor: "var(--mem-hover)",
            padding: "4px 8px",
            borderRadius: "6px",
            flex: 1,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {visible ? token : "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022"}
        </code>
        <button
          onClick={onToggleVisible}
          className="px-2 py-1 rounded text-xs font-medium transition-colors"
          style={{
            fontFamily: "var(--mem-font-body)",
            color: "var(--mem-text-secondary)",
            backgroundColor: "var(--mem-hover)",
          }}
        >
          {visible ? "Hide" : "Show"}
        </button>
        <button
          onClick={onCopy}
          className="px-2 py-1 rounded text-xs font-medium transition-colors"
          style={{
            fontFamily: "var(--mem-font-body)",
            color: copied ? "var(--mem-accent-warm)" : "var(--mem-accent-indigo)",
            backgroundColor: copied
              ? "rgba(251, 191, 36, 0.15)"
              : "rgba(123, 123, 232, 0.1)",
          }}
        >
          {copied ? "Copied!" : "Copy"}
        </button>
        <button
          onClick={onRotate}
          className="px-2 py-1 rounded text-xs font-medium transition-colors whitespace-nowrap"
          style={{
            fontFamily: "var(--mem-font-body)",
            color: confirmRotate ? "var(--mem-accent-warm)" : "var(--mem-text-tertiary)",
            backgroundColor: confirmRotate
              ? "rgba(251, 191, 36, 0.15)"
              : "var(--mem-hover)",
          }}
        >
          {confirmRotate ? "Click again to confirm" : "Rotate"}
        </button>
      </div>
    </div>
  );
}

function InstructionsBlock({
  expanded,
  onToggle,
}: {
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div>
      <button
        onClick={onToggle}
        aria-expanded={expanded}
        className="flex items-center gap-1 transition-colors"
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "12px",
          color: "var(--mem-text-tertiary)",
        }}
      >
        <svg
          className="w-3 h-3 transition-transform"
          style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)" }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 5l7 7-7 7"
          />
        </svg>
        How to connect Claude.ai and ChatGPT
      </button>
      {expanded && (
        <div
          className="mt-2 space-y-3 pl-4"
          style={{
            fontFamily: "var(--mem-font-body)",
            fontSize: "12px",
            color: "var(--mem-text-secondary)",
            lineHeight: "1.6",
          }}
        >
          <div>
            <div style={{ color: "var(--mem-text)", fontWeight: 500 }}>
              Claude.ai
            </div>
            <div style={{ color: "var(--mem-text-secondary)" }}>
              Settings &rarr; Connectors &rarr; Add Custom Connector &rarr; Paste URL
            </div>
          </div>
          <div>
            <div style={{ color: "var(--mem-text)", fontWeight: 500 }}>
              ChatGPT
            </div>
            <div style={{ color: "var(--mem-text-secondary)" }}>
              Settings &rarr; Apps &rarr; Advanced settings &rarr; Enable Developer mode &rarr; Back &rarr; Create app &rarr; Paste URL (No Auth)
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
