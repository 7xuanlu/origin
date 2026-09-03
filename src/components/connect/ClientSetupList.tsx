// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import {
  detectMcpClients,
  writeMcpConfig,
  installClientPlugin,
  type McpClient,
} from "../../lib/tauri";
import { readingIsYes } from "../../lib/reading";
import { isPluginClient } from "./pluginClients";
import { unreadPluginWriteRisk } from "./setupRisk";
import { clientTypeFamily } from "../../lib/agents";
import ClientRow, { clientRowDescId } from "./ClientRow";
import { Button } from "../memory/settings/primitives";

/** Apps & CLIs group. Every detected client has the same one-click "Set up" —
 *  what that button *does* differs by client, and `isPluginClient` is the only
 *  thing that decides: Claude Code and Codex get the Wenlan plugin (which
 *  registers the MCP server itself), everyone else gets an MCP config write.
 *  Writing a config for a plugin client would register Wenlan twice, so this
 *  surface obeys the same invariant the wizard does.
 *
 *  `connectedFamilies` (the tool families the roster above already shows as
 *  connected) is the single source of truth for what to hide here: a client
 *  whose family already has an identity is represented above, so re-listing
 *  it — even when its own config file looks unconfigured — is the duplication
 *  the user vetoed. */
export default function ClientSetupList({
  connectedFamilies,
}: {
  connectedFamilies?: Set<string>;
} = {}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [busy, setBusy] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});
  // Non-fatal notes from a write that SUCCEEDED. Round 6, D3's boundary
  // defect: `writeMcpConfig` now resolves with the resolver inputs it could not
  // determine, and a success that skipped candidates it never built must not
  // render as a plain success.
  const [warnings, setWarnings] = useState<Record<string, string>>({});

  const { data: clients } = useQuery({ queryKey: ["mcp-clients"], queryFn: detectMcpClients });
  // Hide a client that is already configured (nothing left to do) OR whose
  // tool family is already connected in the roster above.
  // `!readingIsYes`, not `!configured`: a client whose config could not be
  // READ is still actionable — hiding it would be "nothing left to do here"
  // stated from a look that failed. Its row carries the unknown chip.
  const actionable = (clients ?? []).filter(
    (client) =>
      !readingIsYes(client.already_configured) &&
      !(connectedFamilies?.has(clientTypeFamily(client.client_type)) ?? false),
  );

  const setUp = async (clientType: string) => {
    setBusy(clientType);
    setErrors((prev) => ({ ...prev, [clientType]: "" }));
    setWarnings((prev) => ({ ...prev, [clientType]: "" }));
    try {
      if (isPluginClient(clientType)) {
        await installClientPlugin(clientType);
      } else {
        const undetermined = await writeMcpConfig(clientType);
        // An empty array is a measurement: every resolver input was read.
        // A non-empty one is the round-5 D4 fact arriving at the pixel — the
        // entry was written by a search that never built some of its
        // candidates, and saying nothing would make that identical to a search
        // that read them all.
        if (undetermined.length > 0) {
          setWarnings((prev) => ({
            ...prev,
            [clientType]: undetermined
              .map((u) => t("connectMatrix.wroteWithUndeterminedInput", { ...u }))
              .join(" "),
          }));
        }
      }
      queryClient.invalidateQueries({ queryKey: ["mcp-clients"] });
    } catch (err) {
      setErrors((prev) => ({ ...prev, [clientType]: String(err) }));
    } finally {
      setBusy(null);
    }
  };

  const notInstalled = (
    <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
      {t("connectMatrix.notDetected")}
    </span>
  );

  const notChecked = (
    <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-status-warning-text)" }}>
      {t("connectMatrix.notChecked")}
    </span>
  );

  // Three branches, because there are three readings. Only a MEASURED `no`
  // says "Not installed"; a failed look says so, and still offers the button —
  // withholding the action would be the same false negative wearing a
  // different coat.
  const trailing = (client: McpClient) => {
    if (client.detected.kind === "no") return notInstalled;
    // Round 6, D6a. A write that could produce a duplicate registration must
    // not be offered as the SAME unqualified action a measured `no` gets. The
    // label changes to "Set up anyway", and the button points at the body line
    // that says what could not be read — so a screen reader hears the risk as
    // part of the button, not several nodes away from it.
    const duplicateRisk = unreadPluginWriteRisk(client) !== null;
    const button = (
      <Button
        type="button"
        variant="secondary"
        size="sm"
        onClick={() => setUp(client.client_type)}
        disabled={busy === client.client_type}
        aria-describedby={duplicateRisk ? clientRowDescId(client.client_type) : undefined}
      >
        {busy === client.client_type
          ? t("connectMatrix.settingUp")
          : duplicateRisk
            ? t("connectMatrix.setUpAnyway")
            : t("connectMatrix.setUp")}
      </Button>
    );
    if (client.detected.kind === "unreadable") {
      return (
        <div className="flex items-center gap-2">
          {notChecked}
          {button}
        </div>
      );
    }
    return button;
  };

  if (clients && actionable.length === 0) {
    return (
      <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
        {t("connectMatrix.allConnected")}
      </span>
    );
  }

  return (
    <div className="flex flex-col" style={{ gap: "8px" }}>
      {actionable.map((client) => {
        const duplicateRiskError = unreadPluginWriteRisk(client);
        return (
          <ClientRow
            key={client.client_type}
            client={client}
            configured={client.already_configured}
            error={errors[client.client_type]}
            warning={
              // Two independent non-fatal notes, and they can both be live:
              // the plugin state could not be read BEFORE the write, and the
              // binary search skipped an input DURING it. Joined, not ranked.
              [
                duplicateRiskError
                  ? t("connectMatrix.pluginStateUnknownBeforeWrite", { error: duplicateRiskError })
                  : null,
                warnings[client.client_type] || null,
              ]
                .filter(Boolean)
                .join(" ") || null
            }
            trailing={trailing(client)}
          />
        );
      })}
    </div>
  );
}
