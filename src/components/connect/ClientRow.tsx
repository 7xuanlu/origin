// SPDX-License-Identifier: AGPL-3.0-only
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { McpClient } from "../../lib/tauri";
import { StatusChip } from "../memory/settings/primitives";

interface ClientRowProps {
  client: McpClient;
  /** Settings shows the mono config path under the name; the wizard hides it
   *  (noise for a first-run user). */
  showConfigPath?: boolean;
  /** Left slot in the title row (wizard checkbox). Wrapped together with the
   *  name in a <label> so clicking the name also toggles the checkbox. */
  leading?: ReactNode;
  /** Right slot in the title row (Settings "Set up" button / "Not installed"
   *  text). Rendered outside the label — never nested inside it, so a click
   *  here can never also toggle a `leading` checkbox. */
  trailing?: ReactNode;
  /** Body under the title row (the wizard's "already set up" note, etc).
   *  Always a sibling of the label — never nested inside it — for the same
   *  reason. */
  children?: ReactNode;
  error?: string | null;
  /** Renders a StatusChip (`state: {kind: "up"}`, honest because
   *  `already_configured` came from actually parsing the config file). */
  configured: boolean;
  /** Highlights the card border — wizard rows use this when their checkbox
   *  is checked. */
  selected?: boolean;
}

/** Deterministic id for a row's body (config path, error, status note) so a
 *  caller-built `leading` checkbox can wire `aria-describedby` to it without a
 *  prop round-trip through this component. */
export function clientRowDescId(clientType: string): string {
  return `client-row-desc-${clientType}`;
}

/** One row renderer shared by the wizard's ConnectStep and Settings'
 *  ClientSetupList — sharing the row (not the whole list) is what the
 *  redesign spec (docs/superpowers/plans/2026-07-12-connect-step-redesign.md
 *  §4) uses to stop copy/behavior drifting between the two surfaces. */
export default function ClientRow({
  client,
  showConfigPath,
  leading,
  trailing,
  children,
  error,
  configured,
  selected,
}: ClientRowProps) {
  const { t } = useTranslation();
  // Deterministic (not useId-generated) so the caller's `leading` checkbox —
  // built independently, before this component ever renders it — can wire
  // aria-describedby to the same id without a prop round-trip.
  const descId = clientRowDescId(client.client_type);
  const hasBody = Boolean(showConfigPath || children || error);

  const nameBadges = (
    <div className="flex items-center gap-2 flex-wrap min-w-0">
      <span
        className="truncate"
        style={{
          fontFamily: "var(--mem-font-body)",
          fontSize: "var(--mem-text-md)",
          fontWeight: 500,
          color: "var(--mem-text)",
        }}
      >
        {client.name}
      </span>
      {configured && (
        <StatusChip state={{ kind: "up" }} label={t("connectMatrix.configured")} />
      )}
    </div>
  );

  return (
    <div
      className="flex flex-col gap-2 rounded-xl px-4 py-3"
      style={{
        backgroundColor: "var(--mem-surface)",
        border: `1px solid ${selected ? "var(--mem-accent-indigo-border)" : "var(--mem-border)"}`,
      }}
    >
      <div className="flex items-start justify-between gap-3">
        {leading ? (
          <label className="flex items-start gap-3 min-w-0" style={{ cursor: "pointer" }}>
            {leading}
            {nameBadges}
          </label>
        ) : (
          <div className="flex items-start gap-3 min-w-0">{nameBadges}</div>
        )}
        {trailing && <div className="shrink-0">{trailing}</div>}
      </div>

      {hasBody && (
        <div
          id={descId}
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "6px",
            paddingLeft: leading ? "28px" : 0,
          }}
        >
          {showConfigPath && (
            <p
              className="truncate"
              style={{
                fontFamily: "var(--mem-font-mono)",
                fontSize: "var(--mem-text-2xs)",
                color: "var(--mem-text-tertiary)",
                margin: 0,
              }}
            >
              {client.config_path}
            </p>
          )}
          {children}
          {error && (
            <p
              role="alert"
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "var(--mem-text-xs)",
                color: "var(--mem-status-danger-text)",
                margin: 0,
              }}
            >
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
