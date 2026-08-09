// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  listAgents,
  updateAgent,
  deleteAgent,
  detectMcpClients,
  type AgentConnection,
  type McpClient,
} from "../../../../lib/tauri";
import {
  clientTypeFamily,
  describeTrustLevel,
  familyDisplayName,
  resolveAgentDisplayName,
  toolFamilyOf,
  TRUST_LEVELS,
} from "../../../../lib/agents";
import { RemoteAccessPanel } from "../../RemoteAccessPanel";
import { Button, Card, ConfirmActionButton, SectionHeader, Select, Tag, Toggle } from "../primitives";
import ClientSetupList from "../../../connect/ClientSetupList";

type AgentUpdate = { enabled?: boolean; trustLevel?: string };

/** One coalesced tool row: every registered identity of a single physical
 *  tool (Codex's `codex` + `codex-mcp-client` + …) folded together. */
interface FamilyRow {
  family: string;
  identities: AgentConnection[];
  /** The wizard's internal connection probe — rendered muted, sorted last. */
  isProbe: boolean;
  /** A configured-but-not-yet-active client sharing this family: show the
   *  restart note in the row meta instead of as a separate pending row. */
  hasPendingNote: boolean;
  memoryCount: number;
  latestLastSeen: number | null;
  anyEnabled: boolean;
  /** The shared trust level when every identity agrees, else `null` (mixed). */
  uniformTrust: string | null;
}

export default function AgentsSection({ onSetupAgent }: { onSetupAgent?: () => void }) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const { data: agents = [] } = useQuery({
    queryKey: ["agents"],
    queryFn: listAgents,
  });

  const { data: mcpClients = [] } = useQuery({
    queryKey: ["mcp-clients"],
    queryFn: detectMcpClients,
  });

  const updateAgentMut = useMutation({
    mutationFn: ({ name, updates }: { name: string; updates: AgentUpdate }) =>
      updateAgent(name, updates),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["agents"] }),
  });

  const deleteAgentMut = useMutation({
    mutationFn: deleteAgent,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["agents"] }),
  });

  const updateOne = (name: string, updates: AgentUpdate) =>
    updateAgentMut.mutate({ name, updates });
  const updateAll = (identities: AgentConnection[], updates: AgentUpdate) =>
    identities.forEach((a) => updateAgentMut.mutate({ name: a.name, updates }));

  // ── Coalesce identities into one row per tool family ─────────────────
  const familyMap = new Map<string, AgentConnection[]>();
  for (const agent of agents) {
    const family = toolFamilyOf(agent);
    const list = familyMap.get(family);
    if (list) list.push(agent);
    else familyMap.set(family, [agent]);
  }
  const connectedFamilies = new Set(familyMap.keys());

  // A configured client with no connected identity in its family is a pending
  // tool row of its own; one whose family IS already connected folds in as a
  // "restart to activate" note on that family's row (mockup: Codex).
  const pendingNoteFamilies = new Set<string>();
  const pendingClients: McpClient[] = [];
  for (const client of mcpClients) {
    if (!client.already_configured) continue;
    const family = clientTypeFamily(client.client_type) || client.client_type;
    if (connectedFamilies.has(family)) pendingNoteFamilies.add(family);
    else pendingClients.push(client);
  }

  const familyRows: FamilyRow[] = [...familyMap.entries()].map(([family, identities]) => {
    const firstTrust = identities[0]?.trust_level ?? "unknown";
    return {
      family,
      identities,
      isProbe: family === "wenlan-setup",
      hasPendingNote: pendingNoteFamilies.has(family),
      memoryCount: identities.reduce((n, a) => n + a.memory_count, 0),
      latestLastSeen: identities.reduce<number | null>(
        (max, a) =>
          a.last_seen_at != null && (max === null || a.last_seen_at > max) ? a.last_seen_at : max,
        null,
      ),
      anyEnabled: identities.some((a) => a.enabled),
      uniformTrust: identities.every((a) => a.trust_level === firstTrust) ? firstTrust : null,
    };
  });

  // Sort: pending-only rows first, then connected families by latest activity
  // (desc), the wizard probe always last.
  const probeRows = familyRows.filter((r) => r.isProbe);
  const connectedRows = familyRows
    .filter((r) => !r.isProbe)
    .sort((a, b) => (b.latestLastSeen ?? 0) - (a.latestLastSeen ?? 0));

  const isEmpty = familyRows.length === 0 && pendingClients.length === 0;

  return (
    <>
      {/* ── Connected Agents ─────────────────────────────────────── */}
      <section className="mem-fade-up" style={{ animationDelay: "0ms" }}>
        <SectionHeader label={t("settings.agents.connectedAgents")} />
        <Card padding={isEmpty ? "none" : "rows"}>
          {isEmpty ? (
            <div className="px-5 py-6 text-center space-y-3">
              <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-base)", color: "var(--mem-text-tertiary)" }}>
                {t("settings.agents.noAgents")}
              </p>
              <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-sm)", color: "var(--mem-text-tertiary)", opacity: 0.7, lineHeight: "1.5" }}>
                {t("settings.agents.noAgentsDescription")}
              </p>
              {onSetupAgent && (
                <Button variant="primary" size="sm" onClick={onSetupAgent}>
                  {t("settings.agents.setupTool")}
                </Button>
              )}
            </div>
          ) : (
            <>
              {pendingClients.map((client) => (
                <PendingClientRow key={client.client_type} client={client} />
              ))}
              {connectedRows.map((row) => (
                <ToolFamilyRow
                  key={row.family}
                  row={row}
                  agents={agents}
                  expanded={!!expanded[row.family]}
                  onToggleExpand={() =>
                    setExpanded((prev) => ({ ...prev, [row.family]: !prev[row.family] }))
                  }
                  updateOne={updateOne}
                  updateAll={updateAll}
                  onDelete={(name) => deleteAgentMut.mutate(name)}
                />
              ))}
              {probeRows.map((row) => (
                <ToolFamilyRow
                  key={row.family}
                  row={row}
                  agents={agents}
                  expanded={!!expanded[row.family]}
                  onToggleExpand={() =>
                    setExpanded((prev) => ({ ...prev, [row.family]: !prev[row.family] }))
                  }
                  updateOne={updateOne}
                  updateAll={updateAll}
                  onDelete={(name) => deleteAgentMut.mutate(name)}
                />
              ))}
              {onSetupAgent && (
                <div className="px-3 py-2">
                  <Button variant="ghost" size="sm" onClick={onSetupAgent}>
                    {t("settings.agents.setupAnotherTool")}
                  </Button>
                </div>
              )}
            </>
          )}
        </Card>
        {/* Trust legend — collapsed by default; native <details> keeps the
            keyboard/screen-reader contract for free. */}
        <details className="group px-1 pt-2">
          <summary
            className="inline-flex cursor-pointer list-none items-center gap-1.5 rounded-[var(--mem-radius-sm)] px-1 py-0.5 [&::-webkit-details-marker]:hidden focus-visible:outline-2 focus-visible:outline-[var(--mem-focus-ring)] focus-visible:outline-offset-2"
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "var(--mem-text-xs)",
              color: "var(--mem-text-tertiary)",
            }}
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
            {t("settings.agents.trustLevels")}
          </summary>
          <div className="flex flex-col gap-1.5 px-1 pt-2">
            {(Object.keys(TRUST_LEVELS) as Array<keyof typeof TRUST_LEVELS>).map((level) => (
              <div key={level} className="flex items-start gap-2">
                <span className="shrink-0">
                  <Tag tone="neutral">{t(`settings.agents.trust.${level}`)}</Tag>
                </span>
                <span
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "var(--mem-text-sm)",
                    color: "var(--mem-text-secondary)",
                    lineHeight: 1.5,
                  }}
                >
                  {t(`settings.agents.trustSummary.${level}`)}
                </span>
              </div>
            ))}
          </div>
        </details>
      </section>

      {/* ── Add a tool ────────────────────────────────────────────── */}
      <section className="mem-fade-up" style={{ animationDelay: "30ms" }}>
        <SectionHeader label={t("connectMatrix.addToolTitle")} />
        <ClientSetupList connectedFamilies={connectedFamilies} />
      </section>

      {/* ── Web access ────────────────────────────────────────────── */}
      <section className="mem-fade-up" style={{ animationDelay: "60ms" }}>
        <SectionHeader label={t("connectMatrix.webRemoteTitle")} />
        <RemoteAccessPanel />
      </section>
    </>
  );
}

/** A tool that wrote its MCP config but hasn't sent a first memory yet, and
 *  whose family has no connected identity — a row of its own until it
 *  activates. */
function PendingClientRow({ client }: { client: McpClient }) {
  const { t } = useTranslation();
  return (
    <div className="px-5 py-3" style={{ opacity: 0.7 }}>
      <div className="flex items-center gap-2 flex-wrap">
        <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
          {client.name}
        </span>
        <Tag tone="neutral">{t("settings.agents.configured")}</Tag>
      </div>
      <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)", marginTop: "2px" }}>
        {t("settings.agents.restartToActivate", { name: client.name })}
      </p>
    </div>
  );
}

/** The coalesced tool row plus its per-identity disclosure. Aggregate
 *  controls (trust, enable) drive every identity at once; the disclosure
 *  exposes each identity's own controls unchanged from the v2 row. */
function ToolFamilyRow({
  row,
  agents,
  expanded,
  onToggleExpand,
  updateOne,
  updateAll,
  onDelete,
}: {
  row: FamilyRow;
  agents: AgentConnection[];
  expanded: boolean;
  onToggleExpand: () => void;
  updateOne: (name: string, updates: AgentUpdate) => void;
  updateAll: (identities: AgentConnection[], updates: AgentUpdate) => void;
  onDelete: (name: string) => void;
}) {
  const { t } = useTranslation();
  const { family, identities, isProbe, hasPendingNote, memoryCount, latestLastSeen, anyEnabled, uniformTrust } = row;
  const displayName = isProbe ? t("connections.internalProbe") : familyDisplayName(family);

  return (
    <div
      className="px-5 py-3 transition-opacity"
      style={{ opacity: isProbe ? 0.55 : anyEnabled ? 1 : 0.5 }}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-md)", fontWeight: 500, color: "var(--mem-text)" }}>
              {displayName}
            </span>
            {identities.length > 1 && (
              <Tag tone="neutral">{t("connections.identities", { count: identities.length })}</Tag>
            )}
            {hasPendingNote && <Tag tone="neutral">{t("settings.agents.configured")}</Tag>}
          </div>
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            {isProbe && (
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
                {t("connections.internalProbeHint")}
              </span>
            )}
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
              {t("settings.agents.memories", { count: memoryCount })}
            </span>
            {latestLastSeen != null && (
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
                {t("settings.agents.lastSeen", {
                  date: new Date(latestLastSeen * 1000).toLocaleDateString(),
                })}
              </span>
            )}
            {hasPendingNote && (
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
                {t("settings.agents.restartToActivate", { name: displayName })}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {/* Aggregate controls drive every identity; the probe row keeps its
              controls inside the disclosure only. */}
          {!isProbe && (
            <>
              <div
                className="w-fit shrink-0"
                title={
                  uniformTrust
                    ? t(`settings.agents.trustSummary.${describeTrustLevel(uniformTrust).level}`)
                    : undefined
                }
              >
                <Select
                  size="sm"
                  value={uniformTrust ?? ""}
                  onChange={(e) => {
                    if (e.target.value) updateAll(identities, { trustLevel: e.target.value });
                  }}
                >
                  {uniformTrust === null && (
                    <option value="" disabled>
                      {t("connections.mixed")}
                    </option>
                  )}
                  <option value="full">{t("settings.agents.trust.full")}</option>
                  <option value="review">{t("settings.agents.trust.review")}</option>
                  <option value="unknown">{t("settings.agents.trust.unknown")}</option>
                </Select>
              </div>
              <Toggle
                enabled={anyEnabled}
                onToggle={() => updateAll(identities, { enabled: !anyEnabled })}
                aria-label={displayName}
              />
            </>
          )}
          <button
            onClick={onToggleExpand}
            aria-expanded={expanded}
            aria-label={t("connections.showIdentities")}
            className="flex items-center justify-center rounded-[var(--mem-radius-sm)] p-1 transition-colors focus-visible:outline-2 focus-visible:outline-[var(--mem-focus-ring)] focus-visible:outline-offset-2"
            style={{ color: "var(--mem-text-tertiary)" }}
          >
            <svg
              className="w-3.5 h-3.5 transition-transform"
              style={{ transform: expanded ? "rotate(90deg)" : "rotate(0deg)" }}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>

      {expanded && (
        <div className="mt-2 -mx-5 border-t" style={{ borderColor: "var(--mem-border)" }}>
          {identities.map((agent) => (
            <IdentityRow
              key={agent.id}
              agent={agent}
              agents={agents}
              updateOne={updateOne}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** One identity inside a tool family's disclosure — the v2 per-agent row
 *  controls (trust Select, enable Toggle, delete), keyed to a single agent. */
function IdentityRow({
  agent,
  agents,
  updateOne,
  onDelete,
}: {
  agent: AgentConnection;
  agents: AgentConnection[];
  updateOne: (name: string, updates: AgentUpdate) => void;
  onDelete: (name: string) => void;
}) {
  const { t } = useTranslation();
  const trustLevel = describeTrustLevel(agent.trust_level).level;
  return (
    <div
      className="flex items-center justify-between gap-3 px-5 py-2.5 transition-opacity"
      style={{ opacity: agent.enabled ? 1 : 0.5 }}
    >
      <div className="min-w-0 flex-1">
        <span
          title={t("settings.agents.canonicalIdTitle")}
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "var(--mem-text-2xs)",
            color: "var(--mem-text-tertiary)",
            opacity: 0.75,
          }}
        >
          {agent.name} · {agent.agent_type}
        </span>
        <div className="flex items-center gap-3 mt-1 flex-wrap">
          <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
            {t("settings.agents.memories", { count: agent.memory_count })}
          </span>
          {agent.last_seen_at && (
            <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
              {t("settings.agents.lastSeen", {
                date: new Date(agent.last_seen_at * 1000).toLocaleDateString(),
              })}
            </span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <div className="w-fit shrink-0" title={t(`settings.agents.trustSummary.${trustLevel}`)}>
          <Select
            size="sm"
            value={agent.trust_level}
            onChange={(e) => updateOne(agent.name, { trustLevel: e.target.value })}
          >
            <option value="full">{t("settings.agents.trust.full")}</option>
            <option value="review">{t("settings.agents.trust.review")}</option>
            <option value="unknown">{t("settings.agents.trust.unknown")}</option>
          </Select>
        </div>
        <Toggle
          enabled={agent.enabled}
          onToggle={() => updateOne(agent.name, { enabled: !agent.enabled })}
          aria-label={agent.name}
        />
        <ConfirmActionButton
          variant="ghost"
          size="sm"
          aria-label={t("settings.agents.deleteAgent", {
            name: resolveAgentDisplayName(agent.name, agents),
          })}
          confirmLabel={t("settings.agents.confirm")}
          cancelLabel={t("settings.agents.cancel")}
          onConfirm={() => onDelete(agent.name)}
        >
          <svg aria-hidden="true" className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </ConfirmActionButton>
      </div>
    </div>
  );
}
