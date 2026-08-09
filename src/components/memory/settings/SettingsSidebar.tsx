// SPDX-License-Identifier: AGPL-3.0-only
//
// Settings sidebar. Replaces the default Spaces/Profile sidebar when the
// Settings page is active. It keeps the same width, background, border, and
// footer brand treatment as the main Sidebar so the transition feels like
// "the sidebar switched modes" rather than "a different layout loaded".

import { useEffect, useState } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { useTranslation } from "react-i18next";

export type SettingsSection =
  | "sources"
  | "agents"
  | "intelligence"
  | "diagnostics"
  | "general";

type SettingsGroupLabelKey =
  | "settings.groups.general.label"
  | "settings.groups.intelligence.label"
  | "settings.groups.diagnostics.label"
  | "settings.groups.agents.label"
  | "settings.groups.sources.label";

type SettingsGroupHintKey =
  | "settings.groups.general.hint"
  | "settings.groups.intelligence.hint"
  | "settings.groups.diagnostics.hint"
  | "settings.groups.agents.hint"
  | "settings.groups.sources.hint";

export interface SettingsGroup {
  id: SettingsSection;
  labelKey: SettingsGroupLabelKey;
  /** One-line caption shown beneath the label when the group is active. */
  hintKey: SettingsGroupHintKey;
  icon: React.ReactNode;
}

/**
 * The master list of settings groups. Two places it's consumed:
 *   1. The sidebar rendering (below) — icon + label per button.
 *   2. `SettingsPage.tsx` gating — each `<section>` checks `section === "..."`.
 *
 * Ordering is the display order in the sidebar. Groups that used to be
 * separate sections (Sources + Import; Connected Agents + Remote Access)
 * are consolidated into one group each — they were related enough that
 * stacking them as siblings was just noise.
 */
export const SETTINGS_GROUPS: SettingsGroup[] = [
  {
    id: "general",
    labelKey: "settings.groups.general.label",
    hintKey: "settings.groups.general.hint",
    icon: (
      <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
      </svg>
    ),
  },
  {
    id: "intelligence",
    labelKey: "settings.groups.intelligence.label",
    hintKey: "settings.groups.intelligence.hint",
    icon: (
      <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 3v18M3 12h18M5.6 5.6l12.8 12.8M18.4 5.6L5.6 18.4" />
      </svg>
    ),
  },
  {
    id: "diagnostics",
    labelKey: "settings.groups.diagnostics.label",
    hintKey: "settings.groups.diagnostics.hint",
    icon: (
      <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="M8 15l3-3 3 2 4-6" />
      </svg>
    ),
  },
  {
    id: "agents",
    labelKey: "settings.groups.agents.label",
    hintKey: "settings.groups.agents.hint",
    icon: (
      <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
  },
  {
    id: "sources",
    labelKey: "settings.groups.sources.label",
    hintKey: "settings.groups.sources.hint",
    icon: (
      <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5" />
        <path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3" />
      </svg>
    ),
  },
];

interface SettingsSidebarProps {
  collapsed: boolean;
  active: SettingsSection;
  onSelect: (section: SettingsSection) => void;
  onNavigateHome: () => void;
}

export default function SettingsSidebar({
  collapsed,
  active,
  onSelect,
  onNavigateHome,
}: SettingsSidebarProps) {
  const { t } = useTranslation();
  const [appVersion, setAppVersion] = useState<string>("");
  useEffect(() => {
    getVersion().then(setAppVersion).catch(() => setAppVersion(""));
  }, []);
  return (
    <aside
      className="flex-shrink-0 flex flex-col transition-[width] duration-200 ease-out"
      style={{
        width: collapsed ? 0 : 240,
        backgroundColor: "var(--mem-sidebar)",
        borderRight: collapsed ? "none" : "1px solid var(--mem-border)",
        overflow: "hidden",
      }}
    >
      <div
        className="flex flex-col h-full transition-opacity duration-150"
        style={{
          width: 240,
          opacity: collapsed ? 0 : 1,
          pointerEvents: collapsed ? "none" : "auto",
        }}
      >
        <div className="px-2 pt-3 pb-3">
          <button
            onClick={onNavigateHome}
            className="flex items-center gap-3 px-3 py-2 rounded-md transition-colors duration-150 text-left hover:bg-[var(--mem-hover)] w-full"
            style={{
              border: "none",
              background: "transparent",
              color: "var(--mem-text-secondary)",
              cursor: "pointer",
            }}
          >
            <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--mem-text-tertiary)" }}>
              <path d="M3 10.5L12 3l9 7.5" />
              <path d="M5 9.5V21h14V9.5" />
              <path d="M9.5 21v-6h5v6" />
            </svg>
            <span
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "var(--mem-text-base)",
                fontWeight: 400,
              }}
            >
              {t("settings.home")}
            </span>
          </button>
        </div>

        {/* Section caption */}
        <div className="px-4 pb-2">
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "var(--mem-text-2xs)",
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--mem-text-tertiary)",
            }}
          >
            {t("settings.title")}
          </span>
        </div>

        {/* Group list */}
        <nav className="flex flex-col gap-0.5 px-2 pb-4 flex-1 overflow-y-auto">
          {SETTINGS_GROUPS.map((group) => {
            const isActive = group.id === active;
            return (
              <button
                key={group.id}
                onClick={() => onSelect(group.id)}
                className={`group relative flex items-center gap-3 px-3 py-2 rounded-md transition-colors duration-150 text-left ${
                  isActive ? "" : "bg-transparent mem-row-hover"
                }`}
                style={{
                  backgroundColor: isActive ? "var(--mem-hover-strong)" : undefined,
                  color: isActive ? "var(--mem-text)" : "var(--mem-text-secondary)",
                  cursor: "pointer",
                  border: "none",
                }}
              >
                {/* Active accent bar */}
                {isActive && (
                  <span
                    aria-hidden
                    style={{
                      position: "absolute",
                      left: 0,
                      top: "20%",
                      bottom: "20%",
                      width: 2,
                      borderRadius: 2,
                      backgroundColor: "var(--mem-accent-indigo)",
                    }}
                  />
                )}
                <span
                  style={{
                    color: isActive
                      ? "var(--mem-accent-indigo)"
                      : "var(--mem-text-tertiary)",
                    display: "flex",
                    alignItems: "center",
                  }}
                >
                  {group.icon}
                </span>
                <span
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "var(--mem-text-base)",
                    fontWeight: isActive ? 500 : 400,
                  }}
                >
                  {t(group.labelKey)}
                </span>
              </button>
            );
          })}
        </nav>

        {/* Bottom brand strip — pinned. Mirrors main Sidebar pattern. The
            privacy note used to live here too, but it duplicated (and was
            worded more loosely than) the per-section footer in
            SettingsPage.tsx, so the claim now appears in exactly one place. */}
        <div
          className="px-4 pt-3 pb-4 flex-shrink-0"
          style={{ borderTop: "1px solid var(--mem-border)" }}
        >
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              onClick={onNavigateHome}
              className="rounded-sm transition-colors duration-150 hover:text-[var(--mem-text-secondary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--mem-accent-sage)]"
              style={{
                border: "none",
                background: "transparent",
                color: "var(--mem-text-tertiary)",
                cursor: "pointer",
                fontFamily: "var(--mem-font-body)",
                fontSize: "var(--mem-text-xs)",
                fontWeight: 600,
                lineHeight: 1,
                padding: 0,
              }}
            >
              Wenlan
            </button>
            {appVersion && (
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "var(--mem-text-2xs)",
                  color: "var(--mem-text-tertiary)",
                  letterSpacing: "0.02em",
                  opacity: 0.7,
                  whiteSpace: "nowrap",
                }}
              >
                v{appVersion}
              </span>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}
