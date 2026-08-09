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
import { isPluginClient } from "./pluginClients";
import { clientTypeFamily } from "../../lib/agents";
import ClientRow from "./ClientRow";
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

  const { data: clients } = useQuery({ queryKey: ["mcp-clients"], queryFn: detectMcpClients });
  // Hide a client that is already configured (nothing left to do) OR whose
  // tool family is already connected in the roster above.
  const actionable = (clients ?? []).filter(
    (client) =>
      !client.already_configured &&
      !(connectedFamilies?.has(clientTypeFamily(client.client_type)) ?? false),
  );

  const setUp = async (clientType: string) => {
    setBusy(clientType);
    setErrors((prev) => ({ ...prev, [clientType]: "" }));
    try {
      if (isPluginClient(clientType)) {
        await installClientPlugin(clientType);
      } else {
        await writeMcpConfig(clientType);
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

  const trailing = (client: McpClient) =>
    client.detected ? (
      <Button
        type="button"
        variant="secondary"
        size="sm"
        onClick={() => setUp(client.client_type)}
        disabled={busy === client.client_type}
      >
        {busy === client.client_type ? t("connectMatrix.settingUp") : t("connectMatrix.setUp")}
      </Button>
    ) : (
      notInstalled
    );

  if (clients && actionable.length === 0) {
    return (
      <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-xs)", color: "var(--mem-text-tertiary)" }}>
        {t("connectMatrix.allConnected")}
      </span>
    );
  }

  return (
    <div className="flex flex-col" style={{ gap: "8px" }}>
      {actionable.map((client) => (
        <ClientRow
          key={client.client_type}
          client={client}
          configured={client.already_configured}
          error={errors[client.client_type]}
          trailing={trailing(client)}
        />
      ))}
    </div>
  );
}
