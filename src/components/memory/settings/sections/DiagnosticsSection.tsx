// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import {
  clipboardWrite,
  getPipelineStatus,
  getWireState,
  removeLegacyMcpEntry,
  removeRawMcpEntry,
  setSetupCompleted,
  startDaemonSidecar,
  type BinaryCandidate,
  type BinaryWire,
  type ClientWire,
  type DaemonWire,
  type JobBinding,
  type SidecarStopOutcome,
  type PipelineStatusResponse,
  type WireState,
} from "../../../../lib/tauri";
import { readingIsNo, readingIsYes, type Reading } from "../../../../lib/reading";
import { Button, Card, ConfirmActionButton, SectionHeader, Skeleton, StatusChip } from "../primitives";

function sortedEntries(values: Record<string, number>): [string, number][] {
  return Object.entries(values).sort(([leftKey, leftValue], [rightKey, rightValue]) => {
    if (rightValue !== leftValue) return rightValue - leftValue;
    return leftKey.localeCompare(rightKey);
  });
}

function isOldDaemonError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  const lowerMessage = message.toLowerCase();
  return (
    lowerMessage.includes("/api/debug/pipeline") &&
    (lowerMessage.includes("404") || lowerMessage.includes("not found"))
  );
}

// ── Wiring card ───────────────────────────────────────────────────────────
// What the pipeline card below can't tell you: whether the plumbing works at
// all. `getWireState()` never rejects on a down daemon (`daemon.reachable:
// false` instead), so isError here only means the IPC call itself failed.

/** `route` arrives from Rust as a bare tag (`plugin` | `config` | `skip`).
 *  It is never rendered raw: `plugin` is a word we don't say about Wenlan, and
 *  the i18n banned-word guard can't see a value that isn't in resources.ts. An
 *  unrecognised tag falls through to itself so a new route shows up rather than
 *  silently vanishing. */
const ROUTE_LABEL_KEYS = {
  plugin: "settings.diagnostics.wiring.routePlugin",
  config: "settings.diagnostics.wiring.routeConfig",
  skip: "settings.diagnostics.wiring.routeSkip",
  unknown: "settings.diagnostics.wiring.routeUnknown",
} as const;

function routeLabel(t: TFunction, route: string): string {
  const key = ROUTE_LABEL_KEYS[route as keyof typeof ROUTE_LABEL_KEYS];
  return key ? t(key) : route;
}

/** How one client detection reads on screen. THREE labels, because there are
 *  three readings: a look that failed is not a client that is absent, and
 *  "Not detected" said both. */
function detectionLabel(t: TFunction, detected: Reading): string {
  switch (detected.kind) {
    case "yes":
      return t("settings.diagnostics.wiring.clientDetected");
    case "no":
      return t("settings.diagnostics.wiring.clientNotDetected");
    case "unreadable":
      return t("settings.diagnostics.wiring.clientUnreadable");
  }
}

/** How one candidate reads on screen. "Missing" used to cover every non-`file`
 *  state, including the one where the OS refused to answer — the boolean
 *  collapse, rendered. A path that could not be looked at is not a missing
 *  path, and the user chasing a broken setup needs to know which it is. */
function candidateLabel(t: TFunction, state: BinaryCandidate["state"]): string {
  switch (state.kind) {
    case "file":
      return t("settings.diagnostics.wiring.candidateFound");
    case "unreadable":
      return t("settings.diagnostics.wiring.candidateUnreadable");
    case "not_a_file":
    case "not_executable":
      return t("settings.diagnostics.wiring.candidateUnusable");
    default:
      return t("settings.diagnostics.wiring.candidateMissing");
  }
}

/** C1.7. `sidecar_spawned_on_unknown_owner`, the job binding and the last
 *  stop outcome all reached `DaemonWire` and stopped there. `daemon_start`
 *  maps both `Spawn` and `SpawnOnUnknownOwner` to the same `Started` result,
 *  so the difference between "launchd was measured not to own the daemon" and
 *  "nobody could tell" was recorded and never rendered — which made the claim
 *  that the third value reaches a user-visible outcome false. These render it.
 *
 *  All three are about a daemon that outlives, or duplicates, the one the user
 *  thinks they have; none of them is visible anywhere else in the app. */
function sidecarBindingLabel(t: TFunction, binding: JobBinding): string {
  switch (binding.state) {
    case "bound":
      return t("settings.diagnostics.wiring.sidecarBound");
    case "unbound":
      return t("settings.diagnostics.wiring.sidecarUnbound");
    default:
      return t("settings.diagnostics.wiring.sidecarNotSupported");
  }
}

/** `null` for `no_sidecar`: nothing was stopped and nothing needed to be, so
 *  there is no outcome worth a row. Every other outcome gets one — including
 *  `could_not_measure`, which is the whole reason the field exists. */
function lastStopLabel(t: TFunction, stop: SidecarStopOutcome): string | null {
  switch (stop.outcome) {
    case "ended":
      return t("settings.diagnostics.wiring.lastStopEnded");
    case "still_running":
      return t("settings.diagnostics.wiring.lastStopStillRunning");
    case "could_not_measure":
      return t("settings.diagnostics.wiring.lastStopUnverified");
    default:
      return null;
  }
}

/** Lines for the three sidecar facts, shared by the card and the copyable
 *  report so they cannot drift apart. */
function sidecarReportLines(t: TFunction, daemon: DaemonWire): string[] {
  const lines: string[] = [];
  if (daemon.sidecar_job_binding) {
    lines.push(sidecarBindingLabel(t, daemon.sidecar_job_binding));
  }
  const stop = daemon.last_sidecar_stop ? lastStopLabel(t, daemon.last_sidecar_stop) : null;
  if (stop) lines.push(stop);
  if (daemon.sidecar_spawned_on_unknown_owner) {
    lines.push(t("settings.diagnostics.wiring.sidecarUnknownOwner"));
  }
  return lines;
}

/** Plain-text dump of the wire state, built from the same translated labels
 *  the card renders — what a user pastes into a bug report matches what they
 *  saw on screen. */
function buildWireReport(t: TFunction, wire: WireState): string {
  const lines: string[] = [t("settings.diagnostics.wiring.title")];
  lines.push("");

  const daemonState = wire.daemon.reachable
    ? t("settings.diagnostics.wiring.daemonReachable")
    : t("settings.diagnostics.wiring.daemonUnreachable");
  lines.push(`${t("settings.diagnostics.wiring.daemonTitle")}: ${daemonState}`);
  lines.push(`  ${wire.daemon.base_url}`);
  if (wire.daemon.version) {
    lines.push(`  ${t("settings.diagnostics.wiring.daemonVersion", { version: wire.daemon.version })}`);
  }
  if (!wire.daemon.reachable && wire.daemon.error) {
    lines.push(`  ${wire.daemon.error}`);
  }
  for (const line of sidecarReportLines(t, wire.daemon)) {
    lines.push(`  ${line}`);
  }
  lines.push("");

  lines.push(t("settings.diagnostics.wiring.mcpBinaryTitle"));
  lines.push(
    wire.mcp_binary.command === null
      ? `  ${wire.mcp_binary.unresolved?.message ?? t("settings.diagnostics.wiring.binaryUnresolved")}`
      : `  ${[wire.mcp_binary.command, ...wire.mcp_binary.args].join(" ")}`,
  );
  for (const candidate of wire.mcp_binary.candidates) {
    lines.push(`  [${candidateLabel(t, candidate.state)}] ${candidate.path} (${candidate.source})`);
  }
  // An input that could not be determined has NO candidate row — its paths
  // were never built — so a report that only listed candidates would show a
  // short list and no reason for it. Read off `mcp_binary`, not `unresolved`:
  // a pasted report from a machine where the binary WAS found must still name
  // the input that went unread (round 5, D4).
  for (const undetermined of wire.mcp_binary.undetermined ?? []) {
    lines.push(
      `  [${t("settings.diagnostics.wiring.candidateNotChecked")}] ${t(
        "settings.diagnostics.wiring.binaryUndetermined",
        { ...undetermined },
      )}`,
    );
  }
  lines.push("");

  lines.push(t("settings.diagnostics.wiring.clientsTitle"));
  if (wire.clients.length === 0) {
    lines.push(`  ${t("settings.diagnostics.wiring.clientsEmpty")}`);
  }
  for (const client of wire.clients) {
    const detected = detectionLabel(t, client.detected);
    const path = client.config_path ?? t("settings.diagnostics.wiring.clientPathUnknown");
    lines.push(`  ${client.name}: ${detected}, ${routeLabel(t, client.route)}, ${path}`);
    // A pasted report has to carry the REASON, not just the word "unknown" —
    // it is the only artefact the person debugging this ever sees.
    if (client.detected.kind === "unreadable") {
      lines.push(
        `    ! ${t("settings.diagnostics.wiring.clientUnreadableDetail", {
          error: client.detected.error,
        })}`,
      );
    }
    if (readingIsYes(client.has_plugin) && readingIsYes(client.has_raw_entry)) {
      lines.push(`    ! ${t("settings.diagnostics.wiring.doubleRegistrationBody", { name: client.name })}`);
    }
    // `readingIsNo`, not `!has_plugin`: a plugin state that could not be read
    // must not fire a warning whose whole premise is "this client has no
    // plugin", nor suppress the one above.
    if (readingIsNo(client.has_plugin) && readingIsYes(client.has_raw_duplicate)) {
      lines.push(`    ! ${t("settings.diagnostics.wiring.rawDuplicateBody", { name: client.name })}`);
    }
  }

  return lines.join("\n");
}

// ── Wiring rows: daemon, MCP binary, clients — three independent checks ──

/** Loading state: three skeleton rows matching the resolved row shape.
 *  `sr-only` text carries the load to assistive tech, since the Skeleton
 *  bars are decorative. */
function WiringSkeleton() {
  const { t } = useTranslation();
  return (
    <div aria-busy="true">
      <span className="sr-only">{t("settings.diagnostics.wiring.loading")}</span>
      {[0, 1, 2].map((index) => (
        <div key={index} className="px-5 py-4">
          <div className="flex flex-col gap-2">
            <Skeleton width="35%" height={14} />
            <Skeleton width="72%" height={10} />
          </div>
        </div>
      ))}
    </div>
  );
}

function WiringRows({ wire, onRetry }: { wire: WireState; onRetry: () => void }) {
  return (
    <>
      <div className="px-5 py-4">
        <DaemonStatus daemon={wire.daemon} onRetry={onRetry} />
      </div>
      <div className="px-5 py-4">
        <McpBinaryStatus mcpBinary={wire.mcp_binary} />
      </div>
      <div className="px-5 py-4">
        <ClientsWiring clients={wire.clients} />
      </div>
    </>
  );
}

function DaemonStatus({ daemon, onRetry }: { daemon: DaemonWire; onRetry: () => void }) {
  const { t } = useTranslation();
  return (
    <>
      <div className="flex items-center gap-2 flex-wrap mb-1">
        <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
          {t("settings.diagnostics.wiring.daemonTitle")}
        </span>
        <StatusChip
          state={daemon.reachable ? { kind: "up" } : { kind: "down" }}
          label={
            daemon.reachable
              ? t("settings.diagnostics.wiring.daemonReachable")
              : t("settings.diagnostics.wiring.daemonUnreachable")
          }
        />
      </div>
      <p style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", overflowWrap: "anywhere" }}>
        {daemon.base_url}
      </p>
      {daemon.version && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)", marginTop: 4 }}>
          {t("settings.diagnostics.wiring.daemonVersion", { version: daemon.version })}
        </p>
      )}
      <DaemonSidecarStatus daemon={daemon} />
      {!daemon.reachable && (
        <>
          {daemon.error && (
            <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-status-danger-text)", marginTop: 8, lineHeight: "1.5" }}>
              {daemon.error}
            </p>
          )}
          <div className="mt-3">
            <Button variant="secondary" size="sm" onClick={onRetry}>
              {t("settings.diagnostics.wiring.retry")}
            </Button>
          </div>
        </>
      )}
    </>
  );
}

/** See `sidecarBindingLabel`. Renders nothing at all when this app owns no
 *  sidecar and never stopped one — the ordinary case, where there is nothing
 *  to say and a row saying so would be noise. */
function DaemonSidecarStatus({ daemon }: { daemon: DaemonWire }) {
  const { t } = useTranslation();
  const binding = daemon.sidecar_job_binding ?? null;
  const lastStop = daemon.last_sidecar_stop ?? null;
  const lastStopText = lastStop ? lastStopLabel(t, lastStop) : null;
  if (!binding && !lastStopText && !daemon.sidecar_spawned_on_unknown_owner) return null;
  return (
    <div className="mt-3 flex flex-col gap-2">
      <div style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)" }}>
        {t("settings.diagnostics.wiring.daemonSidecarTitle")}
      </div>
      {binding && (
        <div className="flex items-center gap-2 flex-wrap">
          <StatusChip
            state={
              binding.state === "bound"
                ? { kind: "up" }
                : binding.state === "unbound"
                  ? { kind: "down" }
                  : { kind: "idle" }
            }
            label={sidecarBindingLabel(t, binding)}
          />
          {binding.state === "unbound" && (
            <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-2xs)", color: "var(--mem-text-tertiary)", overflowWrap: "anywhere" }}>
              {binding.reason}
            </span>
          )}
        </div>
      )}
      {lastStop && lastStopText && (
        <div className="flex items-center gap-2 flex-wrap">
          <StatusChip
            // Round 5, D5 residual. `could_not_measure` used to fall into the
            // same `down` as `still_running`, so the colour asserted a
            // negative stop nobody observed — the "unverified" wording was the
            // only thing keeping them apart, and colour is read first. Three
            // outcomes, three tones: ended is measured good, still_running is
            // measured bad, could_not_measure is not a measurement at all.
            state={
              lastStop.outcome === "ended"
                ? { kind: "up" }
                : lastStop.outcome === "could_not_measure"
                  ? { kind: "unknown" }
                  : { kind: "down" }
            }
            label={lastStopText}
          />
          {"reason" in lastStop && (
            <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-2xs)", color: "var(--mem-text-tertiary)", overflowWrap: "anywhere" }}>
              {lastStop.reason}
            </span>
          )}
        </div>
      )}
      {daemon.sidecar_spawned_on_unknown_owner && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-status-danger-text)", lineHeight: "1.5" }}>
          {t("settings.diagnostics.wiring.sidecarUnknownOwner")}
        </p>
      )}
    </div>
  );
}

function McpBinaryStatus({ mcpBinary }: { mcpBinary: BinaryWire }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  // The candidates list is a probe order — most paths are SUPPOSED to be
  // missing as long as one is found. Only flag red (down) when the binary
  // exists nowhere; otherwise a missing candidate is idle, not an alarm.
  const anyFound = mcpBinary.candidates.some((candidate) => candidate.state.kind === "file");
  // A candidate the OS would not answer about is NOT evidence the binary is
  // absent, so it must not trigger the "reinstall via setup" advice: the
  // install may be perfectly fine behind a permission problem, and reinstalling
  // is not the fix for that. Only a fully measured search offers the button.
  // ...and neither is an input that could not be determined: its candidate
  // paths were never built, so the search did not cover them and their absence
  // from the list below is not evidence of anything.
  // Read off `mcpBinary` itself, not off `unresolved`: an input that could not
  // be determined is a property of the search, and a search that FOUND
  // something can still have one (round 5, D4). Rendering it only when no
  // command was chosen is how it stayed invisible in the case where the
  // command beside it looks like a complete answer.
  const undetermined = mcpBinary.undetermined ?? [];
  const searchIncomplete =
    mcpBinary.candidates.some((candidate) => candidate.state.kind === "unreadable") ||
    undetermined.length > 0;
  return (
    <>
      <div className="mb-1" style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
        {t("settings.diagnostics.wiring.mcpBinaryTitle")}
      </div>
      <p style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text)", overflowWrap: "anywhere" }}>
        {mcpBinary.command === null
          ? (mcpBinary.unresolved?.message ?? t("settings.diagnostics.wiring.binaryUnresolved"))
          : [mcpBinary.command, ...mcpBinary.args].join(" ")}
      </p>
      <div className="mt-3 flex flex-col gap-2">
        {/* Keyed by SOURCE as well as path: two slots can name the same file
            (a dev override pointed at the installed binary), and they share one
            reading rather than being deduplicated away. */}
        {mcpBinary.candidates.map((candidate) => (
          <div
            key={`${candidate.source}:${candidate.path}`}
            className="flex items-center gap-2 flex-wrap"
          >
            <StatusChip
              // `unreadable` is a failed look, not a measured absence — same
              // reasoning as the sidecar stop above, so it gets the same
              // unknown tone rather than the red that would read as "this
              // path is definitely not the binary".
              state={
                candidate.state.kind === "file"
                  ? { kind: "up" }
                  : candidate.state.kind === "unreadable"
                    ? { kind: "unknown" }
                    : anyFound
                      ? { kind: "idle" }
                      : { kind: "down" }
              }
              label={candidateLabel(t, candidate.state)}
            />
            <span
              className="truncate flex-1 min-w-0"
              style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)" }}
            >
              {candidate.path}
            </span>
            <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-2xs)", color: "var(--mem-text-tertiary)" }}>
              {candidate.source}
            </span>
          </div>
        ))}
        {undetermined.map((input) => (
          <div key={input.input} className="flex items-center gap-2 flex-wrap">
            <StatusChip
              // An input that could not be determined is the purest case of
              // this: its candidate paths were never even built, so nothing
              // about them was measured either way.
              state={{ kind: "unknown" }}
              label={t("settings.diagnostics.wiring.candidateNotChecked")}
            />
            <span
              className="flex-1 min-w-0"
              style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", overflowWrap: "anywhere" }}
            >
              {t("settings.diagnostics.wiring.binaryUndetermined", { ...input })}
            </span>
          </div>
        ))}
      </div>
      {/* No candidate exists anywhere — the one actionable fix is to re-run
          setup, which reinstalls the binary. Mirrors General's re-run row:
          setup completion is cleared and the wizard is re-armed; data is
          preserved, so the confirm is a light guard, not a danger gate. */}
      {!anyFound && !searchIncomplete && (
        <div className="mt-3">
          <ConfirmActionButton
            variant="secondary"
            size="sm"
            confirmLabel={t("settings.agents.confirm")}
            cancelLabel={t("settings.agents.cancel")}
            onConfirm={async () => {
              await setSetupCompleted(false);
              queryClient.invalidateQueries({ queryKey: ["shouldShowWizard"] });
            }}
          >
            {t("settings.diagnostics.wiring.reinstallViaSetup")}
          </ConfirmActionButton>
        </div>
      )}
    </>
  );
}

/** The double-registration warnbox, now with its one-action fix: remove the
 *  raw MCP entry (`removeRawMcpEntry`). Its own component so the mutation's
 *  hooks stay at a component top level rather than inside the clients `.map`.
 *  On success the wire query is invalidated so the box re-renders against
 *  fresh state (and disappears once the duplicate is gone). A failure is
 *  surfaced verbatim — the same policy the daemon error uses. */
function DoubleRegistrationWarning({ client }: { client: ClientWire }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const removeMutation = useMutation({
    mutationFn: () => removeRawMcpEntry(client.client_type),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["wireState"] }),
  });
  const errorMessage =
    removeMutation.error instanceof Error
      ? removeMutation.error.message
      : removeMutation.error != null
        ? String(removeMutation.error)
        : null;
  return (
    <div
      className="flex items-start gap-2 mt-1"
      style={{
        background: "var(--mem-status-danger-bg)",
        border: "1px solid var(--mem-status-danger-border)",
        borderRadius: "var(--mem-radius-md)",
        padding: "8px 10px",
      }}
    >
      <svg aria-hidden="true" className="w-3.5 h-3.5 text-[var(--mem-status-danger-text)] shrink-0 mt-px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
      </svg>
      <div className="flex flex-col gap-2 min-w-0">
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)", lineHeight: "1.5" }}>
          {t("settings.diagnostics.wiring.doubleRegistrationBody", { name: client.name })}
        </p>
        <div>
          <Button
            variant="danger"
            size="sm"
            loading={removeMutation.isPending}
            onClick={() => removeMutation.mutate()}
          >
            {t("settings.diagnostics.wiring.removeDuplicate")}
          </Button>
        </div>
        {errorMessage && (
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)", lineHeight: "1.5", overflowWrap: "anywhere" }}>
            {errorMessage}
          </p>
        )}
      </div>
    </div>
  );
}

/** The raw+raw duplicate warnbox with its one-action fix: remove only the
 *  legacy `origin` entry, keeping the live `wenlan` one (`removeLegacyMcpEntry`).
 *  Its own component for the same reason as DoubleRegistrationWarning — the
 *  mutation's hooks stay at a component top level, not inside the clients
 *  `.map`. Distinct from that box: this fires only for a no-plugin client
 *  (Cursor, Gemini CLI), where both raw entries are the whole connection, so
 *  the fix must keep `wenlan` rather than drop both. */
function RawDuplicateWarning({ client }: { client: ClientWire }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const removeMutation = useMutation({
    mutationFn: () => removeLegacyMcpEntry(client.client_type),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["wireState"] }),
  });
  const errorMessage =
    removeMutation.error instanceof Error
      ? removeMutation.error.message
      : removeMutation.error != null
        ? String(removeMutation.error)
        : null;
  return (
    <div
      className="flex items-start gap-2 mt-1"
      style={{
        background: "var(--mem-status-danger-bg)",
        border: "1px solid var(--mem-status-danger-border)",
        borderRadius: "var(--mem-radius-md)",
        padding: "8px 10px",
      }}
    >
      <svg aria-hidden="true" className="w-3.5 h-3.5 text-[var(--mem-status-danger-text)] shrink-0 mt-px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
      </svg>
      <div className="flex flex-col gap-2 min-w-0">
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)", lineHeight: "1.5" }}>
          {t("settings.diagnostics.wiring.rawDuplicateBody", { name: client.name })}
        </p>
        <div>
          <Button
            variant="danger"
            size="sm"
            loading={removeMutation.isPending}
            onClick={() => removeMutation.mutate()}
          >
            {t("settings.diagnostics.wiring.removeLegacyEntry")}
          </Button>
        </div>
        {errorMessage && (
          <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-danger-text)", lineHeight: "1.5", overflowWrap: "anywhere" }}>
            {errorMessage}
          </p>
        )}
      </div>
    </div>
  );
}

function ClientsWiring({ clients }: { clients: ClientWire[] }) {
  const { t } = useTranslation();
  return (
    <>
      <div className="mb-1" style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
        {t("settings.diagnostics.wiring.clientsTitle")}
      </div>
      {clients.length === 0 ? (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)" }}>
          {t("settings.diagnostics.wiring.clientsEmpty")}
        </p>
      ) : (
        <div className="flex flex-col gap-3">
          {clients.map((client) => {
            // THE valuable finding this card exists to surface: Wenlan
            // registered twice for one client (plugin + a raw MCP entry).
            // Now carries its own one-action fix — see DoubleRegistrationWarning.
            // Both halves must be a MEASURED yes. A warning box that offers a
            // destructive one-click fix ("remove the duplicate entry") may
            // never be raised off a read that failed.
            const doubleRegistered =
              readingIsYes(client.has_plugin) && readingIsYes(client.has_raw_entry);
            // The raw+raw sibling: both `wenlan` and legacy `origin` raw
            // entries in a no-plugin client's config. Gated on a measured
            // `no` for the plugin — not `!has_plugin`, which fired this box on
            // an unread plugin state — so it never double-fires with
            // doubleRegistered (a plugin client is handled by the box above,
            // whose fix removes both raw entries).
            const rawDuplicate =
              readingIsNo(client.has_plugin) && readingIsYes(client.has_raw_duplicate);
            // A detection that could not be made gets the neutral chip, not
            // the "idle" one that means "we looked and it is not here".
            const detectedState: Parameters<typeof StatusChip>[0]["state"] =
              client.detected.kind === "yes"
                ? { kind: "up" }
                : client.detected.kind === "no"
                  ? { kind: "idle" }
                  : { kind: "unknown", detail: client.detected.error };
            return (
              <div key={client.client_type} className="flex flex-col gap-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", fontWeight: 500, color: "var(--mem-text)" }}>
                    {client.name}
                  </span>
                  <StatusChip state={detectedState} label={detectionLabel(t, client.detected)} />
                  <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-2xs)", color: "var(--mem-text-tertiary)" }}>
                    {routeLabel(t, client.route)}
                  </span>
                </div>
                <p
                  className="truncate"
                  style={{ fontFamily: "var(--mem-font-mono)", fontSize: "10px", color: "var(--mem-text-tertiary)" }}
                >
                  {client.config_path ?? t("settings.diagnostics.wiring.clientPathUnknown")}
                </p>
                {doubleRegistered && <DoubleRegistrationWarning client={client} />}
                {rawDuplicate && <RawDuplicateWarning client={client} />}
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}

function CopyReportButton({ wire }: { wire: WireState }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleCopy = () => {
    clipboardWrite(buildWireReport(t, wire));
    setCopied(true);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Button variant="secondary" size="sm" onClick={handleCopy}>
      {copied ? t("settings.diagnostics.wiring.copyReportCopied") : t("settings.diagnostics.wiring.copyReport")}
    </Button>
  );
}

function WiringError({ onRetry }: { onRetry: () => void }) {
  const { t } = useTranslation();
  const [failure, setFailure] = useState<string | null>(null);
  // Diagnostics is reached when the daemon is down, but Retry only re-probes —
  // it can't bring a dead daemon back. Start actually respawns the sidecar
  // (guarded app-side against double-spawn), then re-probes the wiring.
  const start = useMutation({
    mutationFn: startDaemonSidecar,
    onSuccess: (result) => {
      if (result.status === "failed") {
        setFailure(result.message);
        return;
      }
      // started / already_running / launchd_managed — re-probe the wiring.
      setFailure(null);
      onRetry();
    },
    onError: (err) => setFailure(err instanceof Error ? err.message : String(err)),
  });
  return (
    <div className="px-5 py-4">
      <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-status-danger-text)", lineHeight: "1.5" }}>
        {t("settings.diagnostics.wiring.unavailable")}
      </p>
      <div className="mt-3 flex items-center gap-2">
        <Button variant="secondary" size="sm" onClick={() => start.mutate()} disabled={start.isPending}>
          {start.isPending
            ? t("settings.diagnostics.wiring.starting")
            : t("settings.diagnostics.wiring.start")}
        </Button>
        <Button variant="secondary" size="sm" onClick={onRetry}>
          {t("settings.diagnostics.wiring.retry")}
        </Button>
      </div>
      {failure && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-status-danger-text)", lineHeight: "1.5", marginTop: "8px" }}>
          {t("settings.diagnostics.wiring.startFailed", { message: failure })}
        </p>
      )}
      {!failure && start.data?.status === "started" && (
        // The daemon needs a moment to bind the port after spawn; the immediate
        // re-probe can still see it down. Tell the user, don't leave them guessing.
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)", lineHeight: "1.5", marginTop: "8px" }}>
          {t("settings.diagnostics.wiring.startedHint")}
        </p>
      )}
    </div>
  );
}

// ── Pipeline card (unchanged) ───────────────────────────────────────────

function StatList({
  title,
  values,
  empty,
}: {
  title: string;
  values: Record<string, number>;
  empty: string;
}) {
  const entries = sortedEntries(values);
  return (
    <div className="px-5 py-4">
      <div className="mb-2" style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-base)", fontWeight: 600, color: "var(--mem-text)" }}>
        {title}
      </div>
      {entries.length === 0 ? (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)" }}>{empty}</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {entries.map(([key, count]) => (
            <div key={key} className="flex items-center justify-between gap-3">
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)" }}>{key}</span>
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text)" }}>{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EntityLinking({ data }: { data: PipelineStatusResponse }) {
  const { t } = useTranslation();
  const total = data.entity_linking.linked + data.entity_linking.unlinked;
  const percent = total === 0 ? null : Math.round((data.entity_linking.linked / total) * 100);
  return (
    <div className="px-5 py-4">
      <div className="mb-2" style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-base)", fontWeight: 600, color: "var(--mem-text)" }}>
        {t("settings.diagnostics.entityLinking")}
      </div>
      <div className="flex items-baseline gap-3">
        <span style={{ fontFamily: "var(--mem-font-heading)", fontSize: "var(--mem-text-xl)", color: "var(--mem-text)", fontVariantNumeric: "tabular-nums" }}>{data.entity_linking.linked}</span>
        <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)" }}>
          {t("settings.diagnostics.linkedUnlinked", { unlinked: data.entity_linking.unlinked })}
        </span>
      </div>
      {percent !== null && (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)", marginTop: 4 }}>
          {t("settings.diagnostics.percentLinked", { percent })}
        </p>
      )}
    </div>
  );
}

function RefineryQueue({ data }: { data: PipelineStatusResponse }) {
  const { t } = useTranslation();
  return (
    <div className="px-5 py-4">
      <div className="mb-2" style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-base)", fontWeight: 600, color: "var(--mem-text)" }}>
        {t("settings.diagnostics.refineryQueue")}
      </div>
      {data.refinement_queue.length === 0 ? (
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)" }}>{t("settings.diagnostics.refineryEmpty")}</p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {data.refinement_queue.map((entry) => (
            <div key={`${entry.action}:${entry.status}`} className="grid grid-cols-[1fr_auto_auto] items-center gap-3">
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)" }}>{entry.action}</span>
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)" }}>{entry.status}</span>
              <span style={{ fontFamily: "var(--mem-font-mono)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text)" }}>{entry.count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function DiagnosticsError({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  const { t } = useTranslation();
  // Version skew (an old daemon lacking the pipeline route) is not a failure —
  // it reads as a warning (amber), and retrying can't fix it, so no Retry.
  // A genuine unavailability keeps the danger tone and offers a Retry.
  const versionSkew = isOldDaemonError(error);
  return (
    <div className="px-5 py-4">
      <p
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "var(--mem-text-sm)",
          color: versionSkew ? "var(--mem-status-warning-text)" : "var(--mem-status-danger-text)",
          lineHeight: "1.5",
        }}
      >
        {versionSkew ? t("settings.diagnostics.needsNewerDaemon") : t("settings.diagnostics.unavailable")}
      </p>
      {!versionSkew && (
        <div className="mt-3">
          <Button variant="secondary" size="sm" onClick={onRetry}>
            {t("settings.diagnostics.wiring.retry")}
          </Button>
        </div>
      )}
    </div>
  );
}

export default function DiagnosticsSection() {
  const { t } = useTranslation();
  const wireQuery = useQuery({
    queryKey: ["wireState"],
    queryFn: getWireState,
    retry: false,
  });
  const pipelineQuery = useQuery({
    queryKey: ["pipelineStatus"],
    queryFn: getPipelineStatus,
    retry: false,
  });

  return (
    <>
      <section className="mem-fade-up" style={{ animationDelay: "0ms" }}>
        <SectionHeader
          icon={
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="7" width="12" height="10" rx="2" strokeWidth="1.5" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v4M15 3v4M9 21v-4M15 21v-4" />
            </svg>
          }
          label={t("settings.diagnostics.wiring.title")}
          action={wireQuery.data ? <CopyReportButton wire={wireQuery.data} /> : undefined}
        />
        <Card padding="rows">
          {wireQuery.isLoading && <WiringSkeleton />}
          {wireQuery.isError && <WiringError onRetry={() => wireQuery.refetch()} />}
          {wireQuery.data && <WiringRows wire={wireQuery.data} onRetry={() => wireQuery.refetch()} />}
        </Card>
      </section>

      <section className="mem-fade-up" style={{ animationDelay: "0ms" }}>
        <SectionHeader
          icon={
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 19V5" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 19h16" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 15l3-3 3 2 4-6" />
            </svg>
          }
          label={t("settings.diagnostics.pipelineTitle")}
          action={
            <Button variant="secondary" size="sm" onClick={() => pipelineQuery.refetch()}>
              {t("settings.diagnostics.refresh")}
            </Button>
          }
        />
        <Card padding="rows">
          {pipelineQuery.isLoading && (
            <p className="px-5 py-4" style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-secondary)" }}>
              {t("settings.diagnostics.loading")}
            </p>
          )}
          {pipelineQuery.isError && <DiagnosticsError error={pipelineQuery.error} onRetry={() => pipelineQuery.refetch()} />}
          {pipelineQuery.data && (
            <>
              <StatList title={t("settings.diagnostics.enrichment")} values={pipelineQuery.data.enrichment} empty={t("settings.diagnostics.enrichmentEmpty")} />
              <EntityLinking data={pipelineQuery.data} />
              <RefineryQueue data={pipelineQuery.data} />
              <div className="px-5 py-4">
                <div className="mb-1" style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-base)", fontWeight: 600, color: "var(--mem-text)" }}>{t("settings.diagnostics.recaps")}</div>
                <span style={{ fontFamily: "var(--mem-font-heading)", fontSize: "var(--mem-text-xl)", color: "var(--mem-text)", fontVariantNumeric: "tabular-nums" }}>{pipelineQuery.data.recaps}</span>
              </div>
              <StatList title={t("settings.diagnostics.memoryTypes")} values={pipelineQuery.data.types} empty={t("settings.diagnostics.memoryTypesEmpty")} />
              <StatList title={t("settings.diagnostics.quality")} values={pipelineQuery.data.quality} empty={t("settings.diagnostics.qualityEmpty")} />
            </>
          )}
        </Card>
      </section>
    </>
  );
}
