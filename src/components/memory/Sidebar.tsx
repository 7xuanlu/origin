// SPDX-License-Identifier: AGPL-3.0-only
import { useQuery } from "@tanstack/react-query";
import { getMemoryStats, listAgents } from "../../lib/tauri";
import IdentityCard from "./IdentityCard";
import SpaceList from "./SpaceList";
import EntitySuggestions from "./EntitySuggestions";

interface SidebarProps {
  collapsed: boolean;
  onSelectSpace: (spaceName: string | null) => void;
  onEntityClick: (entityId: string) => void;
  onNavigateLog?: () => void;
  onNavigateHome?: () => void;
  onNavigateDecisions?: () => void;
  onNavigateGraph?: () => void;
}


export default function Sidebar({
  collapsed,
  onSelectSpace,
  onEntityClick,
  onNavigateLog,
  onNavigateHome: _onNavigateHome,
  onNavigateDecisions,
  onNavigateGraph,
}: SidebarProps) {
  const { data: _stats } = useQuery({
    queryKey: ["memoryStats"],
    queryFn: getMemoryStats,
    refetchInterval: 10000,
  });

  const { data: agents = [] } = useQuery({
    queryKey: ["agents"],
    queryFn: listAgents,
    refetchInterval: 30000,
  });

  const activeAgents = agents.filter((a) => a.enabled);

  return (
    <aside
      className="flex-shrink-0 flex flex-col transition-[width] duration-200 ease-out overflow-x-hidden"
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
        <div className="flex flex-col gap-6 px-4 pt-2 pb-2">
          <IdentityCard onOpenDetail={onEntityClick} />

          <EntitySuggestions onEntityCreated={onEntityClick} />
        </div>

        <div className="flex-1 overflow-y-auto min-h-0 px-4 pb-4" style={{ overflowX: "hidden" }}>
          <SpaceList onSelectSpace={onSelectSpace} />

          {/* Nav links + agent count (inside scroll area) */}
          <div className="pt-4">
            {onNavigateLog && (
            <button
              onClick={onNavigateLog}
              className="flex items-center gap-2 px-1 py-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)] w-full"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--mem-text-tertiary)" }}>
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <polyline points="10 9 9 9 8 9" />
              </svg>
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>
                Memory Log
              </span>
            </button>
          )}

          {onNavigateDecisions && (
            <button
              onClick={onNavigateDecisions}
              className="flex items-center gap-2 px-1 py-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)] w-full"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--mem-text-tertiary)" }}>
                <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2" />
                <rect x="9" y="3" width="6" height="4" rx="1" />
                <path d="M9 14l2 2 4-4" />
              </svg>
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>
                Decisions
              </span>
            </button>
          )}

          {onNavigateGraph && (
            <button
              onClick={onNavigateGraph}
              className="flex items-center gap-2 px-1 py-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)] w-full"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--mem-text-tertiary)" }}>
                <circle cx="5" cy="6" r="2" />
                <circle cx="19" cy="6" r="2" />
                <circle cx="12" cy="18" r="2" />
                <line x1="7" y1="6" x2="17" y2="6" />
                <line x1="6" y1="8" x2="11" y2="16" />
                <line x1="18" y1="8" x2="13" y2="16" />
              </svg>
              <span style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)" }}>
                Graph
              </span>
            </button>
          )}

          <div style={{ paddingTop: "4px" }}>
            <div className="flex items-center gap-1.5">
              {activeAgents.length > 0 && (
                <span
                  className="w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: "var(--mem-accent-sage)" }}
                />
              )}
              <span
                style={{
                  fontFamily: "var(--mem-font-mono)",
                  fontSize: "10px",
                  color: "var(--mem-text-tertiary)",
                }}
              >
                {activeAgents.length > 0
                  ? `${activeAgents.length} agent${activeAgents.length === 1 ? "" : "s"} connected`
                  : "No agents connected"}
              </span>
            </div>
          </div>
          </div>
        </div>
      </div>
    </aside>
  );
}

/** Sidebar toggle button for use in the header */
export function SidebarToggleButton({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className="flex items-center justify-center rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
      style={{
        width: 28,
        height: 28,
        color: "var(--mem-text-tertiary)",
      }}
      title={collapsed ? "Show sidebar" : "Hide sidebar"}
    >
      <svg
        width="14"
        height="14"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ transition: "transform 200ms", transform: collapsed ? "scaleX(-1)" : "none" }}
      >
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
        <line x1="9" y1="3" x2="9" y2="21" />
      </svg>
    </button>
  );
}
