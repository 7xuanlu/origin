// SPDX-License-Identifier: AGPL-3.0-only
//
// Settings sidebar. Replaces the default Spaces/Profile sidebar when the
// Settings page is active. Matches the main Sidebar chrome exactly — same
// width, same bg, same border, same Origin heading at the top — so the
// transition feels like "the sidebar switched modes" rather than "a
// different layout loaded".

export type SettingsSection =
  | "appearance"
  | "capture"
  | "sources"
  | "agents"
  | "intelligence";

export interface SettingsGroup {
  id: SettingsSection;
  label: string;
  /** One-line caption shown beneath the label when the group is active. */
  hint: string;
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
    id: "intelligence",
    label: "Intelligence",
    hint: "On-device and routed models",
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="9" />
        <path d="M12 3v18M3 12h18M5.6 5.6l12.8 12.8M18.4 5.6L5.6 18.4" />
      </svg>
    ),
  },
  {
    id: "agents",
    label: "Agents",
    hint: "Connected clients and remote access",
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
  },
  {
    id: "sources",
    label: "Sources",
    hint: "Connected files and imports",
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5" />
        <path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3" />
      </svg>
    ),
  },
  {
    id: "appearance",
    label: "Appearance",
    hint: "Theme and display",
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
      </svg>
    ),
  },
  // Capture is hidden for now — ambient capture is disabled as part of the
  // memory-layer pivot. Re-enable when capture features are ready for users.
  // {
  //   id: "capture",
  //   label: "Capture",
  //   hint: "Screen and clipboard",
  //   icon: (
  //     <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
  //       <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z" />
  //       <circle cx="12" cy="13" r="4" />
  //     </svg>
  //   ),
  // },
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
        {/* Origin heading — clickable to return home. Matches main Sidebar. */}
        <div className="px-4 pt-3 pb-3">
          <h1
            onClick={onNavigateHome}
            className="cursor-pointer"
            style={{
              fontFamily: "var(--mem-font-heading)",
              fontSize: "20px",
              fontWeight: 500,
              color: "var(--mem-text)",
            }}
          >
            Origin
          </h1>
        </div>

        {/* Section caption */}
        <div className="px-4 pb-2">
          <span
            style={{
              fontFamily: "var(--mem-font-mono)",
              fontSize: "10px",
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--mem-text-tertiary)",
            }}
          >
            Settings
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
                className="group relative flex items-center gap-3 px-3 py-2 rounded-md transition-colors duration-150 text-left"
                style={{
                  backgroundColor: isActive ? "var(--mem-hover-strong)" : "transparent",
                  color: isActive ? "var(--mem-text)" : "var(--mem-text-secondary)",
                  cursor: "pointer",
                  border: "none",
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    (e.currentTarget as HTMLElement).style.backgroundColor =
                      "var(--mem-hover)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    (e.currentTarget as HTMLElement).style.backgroundColor =
                      "transparent";
                  }
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
                    fontSize: "13px",
                    fontWeight: isActive ? 500 : 400,
                  }}
                >
                  {group.label}
                </span>
              </button>
            );
          })}
        </nav>

        {/* Bottom privacy reassurance — pinned. Mirrors main Sidebar pattern. */}
        <div
          className="px-4 pt-3 pb-4 flex-shrink-0"
          style={{ borderTop: "1px solid var(--mem-border)" }}
        >
          <div className="flex items-start gap-2">
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ color: "var(--mem-text-tertiary)", opacity: 0.7, marginTop: 1 }}
            >
              <rect x="3" y="11" width="18" height="11" rx="2" />
              <path d="M7 11V7a5 5 0 0110 0v4" />
            </svg>
            <span
              style={{
                fontFamily: "var(--mem-font-body)",
                fontSize: "10px",
                lineHeight: 1.5,
                color: "var(--mem-text-tertiary)",
                opacity: 0.7,
              }}
            >
              Local-only. Your data never leaves this machine.
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
