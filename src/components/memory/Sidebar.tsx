// SPDX-License-Identifier: AGPL-3.0-only
import { forwardRef, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { getMemoryStats, listSpaces, type Page, type Space } from "../../lib/tauri";
import { rankRecentPages, readRecentPageHistory } from "../../lib/recentPages";
import { rankRecentSpaces, readRecentSpaceHistory } from "../../lib/recentSpaces";
import IdentityCard from "./IdentityCard";
import EntitySuggestions from "./EntitySuggestions";
import { RecentPages } from "./RecentPages";
import { RecentSpaces } from "./RecentSpaces";
import { PrimaryNavigation } from "./navigation/PrimaryNavigation";
import { ReviewEnvironmentBadge } from "./navigation/ReviewEnvironmentBadge";
import type { GlobalNavigation } from "./navigation/viewState";
import { listAllActivePages } from "./pages/listAllPages";

interface SidebarProps {
  readonly activeNavigation?: GlobalNavigation | null;
  readonly collapsed: boolean;
  readonly currentPageId?: string | null;
  readonly currentSpaceId?: string | null;
  readonly onEntityClick: (entityId: string) => void;
  readonly onNavigateGraph?: () => void;
  onNavigateHome?: () => void;
  readonly onNavigateLog?: () => void;
  readonly onNavigatePages?: () => void;
  readonly onNavigateSettings?: () => void;
  onNavigateSources?: () => void;
  readonly onNavigateSpaces?: (create: boolean) => void;
  readonly onOpenAbout?: () => void;
  readonly onRequestClose?: () => void;
  readonly onSelectPage?: (page: Page) => void;
  readonly onSelectSpace: (space: Space) => void;
  readonly open?: boolean;
  readonly presentation?: "desktop" | "overlay";
  readonly recentPagesRevision?: number;
  readonly recentSpacesRevision?: number;
}

function closeAfterNavigation<Arguments extends readonly unknown[]>(
  navigate: (...arguments_: Arguments) => void,
  close: (() => void) | undefined,
): (...arguments_: Arguments) => void;
function closeAfterNavigation<Arguments extends readonly unknown[]>(
  navigate: ((...arguments_: Arguments) => void) | undefined,
  close: (() => void) | undefined,
): ((...arguments_: Arguments) => void) | undefined;
function closeAfterNavigation<Arguments extends readonly unknown[]>(
  navigate: ((...arguments_: Arguments) => void) | undefined,
  close: (() => void) | undefined,
): ((...arguments_: Arguments) => void) | undefined {
  if (navigate === undefined) return undefined;
  return (...arguments_: Arguments) => {
    navigate(...arguments_);
    close?.();
  };
}

export default function Sidebar({
  activeNavigation = null,
  collapsed,
  currentPageId = null,
  currentSpaceId = null,
  onEntityClick,
  onNavigateGraph,
  onNavigateHome,
  onNavigateLog,
  onNavigatePages,
  onNavigateSettings,
  onNavigateSources,
  onNavigateSpaces = () => {},
  onOpenAbout,
  onRequestClose,
  onSelectPage,
  onSelectSpace,
  open = !collapsed,
  presentation = "desktop",
  recentPagesRevision: _recentPagesRevision = 0,
  recentSpacesRevision: _recentSpacesRevision = 0,
}: SidebarProps) {
  const { t } = useTranslation();
  const asideRef = useRef<HTMLElement>(null);
  const { data: _stats } = useQuery({
    queryKey: ["memoryStats"],
    queryFn: getMemoryStats,
    refetchInterval: 10000,
  });
  const { data: pages = [] } = useQuery({
    queryKey: ["pages", "active"],
    queryFn: listAllActivePages,
  });
  const { data: spaces = [] } = useQuery({ queryKey: ["spaces"], queryFn: listSpaces });
  const pageHistory = readRecentPageHistory({ pages });
  const recentPages = rankRecentPages(pages, pageHistory, Date.now());
  const history = readRecentSpaceHistory({ spaces });
  const recentSpaces = rankRecentSpaces(spaces, history, Date.now());
  const overlay = presentation === "overlay";
  const closeOverlay = overlay ? onRequestClose : undefined;

  useEffect(() => {
    if (presentation !== "overlay" || !open) return;
    const first = asideRef.current?.querySelector<HTMLElement>("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])");
    first?.focus();
  }, [open, presentation]);

  const trapFocus = (event: React.KeyboardEvent<HTMLElement>) => {
    if (event.key !== "Tab" || presentation !== "overlay") return;
    const focusable = asideRef.current?.querySelectorAll<HTMLElement>("button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex='-1'])");
    if (!focusable || focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!first || !last) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <>
      {overlay && open && (
        <button
          aria-label={t("sidebar.close")}
          className="fixed inset-x-0 bottom-0 top-[52px] z-30 border-0 bg-black/40"
          onClick={onRequestClose}
          type="button"
        />
      )}
      <aside
        aria-hidden={!open}
        aria-label={t("sidebar.navigation")}
        className="memory-sidebar flex flex-shrink-0 flex-col overflow-x-hidden transition-[width,transform] duration-200 ease-out"
        inert={!open}
        onKeyDown={trapFocus}
        ref={asideRef}
        style={{
          backgroundColor: "var(--mem-sidebar)",
          borderRight: open ? "1px solid var(--mem-border)" : "none",
          bottom: overlay ? 0 : undefined,
          left: overlay ? 0 : undefined,
          overflow: "hidden",
          position: overlay ? "fixed" : "relative",
          top: overlay ? 52 : undefined,
          transform: overlay && !open ? "translateX(-100%)" : "translateX(0)",
          visibility: overlay && !open ? "hidden" : "visible",
          width: overlay ? 240 : collapsed ? 0 : 240,
          zIndex: overlay ? 40 : undefined,
        }}
      >
      <div
        className="memory-sidebar-content flex flex-col h-full transition-opacity duration-150"
        style={{
          width: 240,
          opacity: open ? 1 : 0,
          pointerEvents: open ? "auto" : "none",
        }}
      >
        <div className="flex flex-col gap-6 px-4 pt-2 pb-2">
          <EntitySuggestions />
        </div>

        <div className="flex-1 overflow-y-auto min-h-0 px-4 pb-4" style={{ overflowX: "hidden" }}>
          <PrimaryNavigation
            active={activeNavigation}
            labels={{
              graph: t("sidebar.graph"),
              home: t("sidebar.home"),
              memories: t("sidebar.memories"),
              navigation: t("sidebar.navigation"),
              pages: t("sidebar.pages"),
              sources: t("sidebar.sources"),
              spaces: t("sidebar.spaces"),
            }}
            onNavigateGraph={closeAfterNavigation(onNavigateGraph, closeOverlay)}
            onNavigateHome={closeAfterNavigation(onNavigateHome, closeOverlay)}
            onNavigateLog={closeAfterNavigation(onNavigateLog, closeOverlay)}
            onNavigatePages={closeAfterNavigation(onNavigatePages, closeOverlay)}
            onNavigateSources={closeAfterNavigation(onNavigateSources, closeOverlay)}
            onNavigateSpaces={closeAfterNavigation(onNavigateSpaces, closeOverlay)}
            recentPagesSection={onSelectPage !== undefined && recentPages.length > 0 ? (
              <section>
                <p className="mb-2 px-1" style={{ color: "var(--mem-text-tertiary)", fontFamily: "var(--mem-font-mono)", fontSize: "10px", fontWeight: 600, letterSpacing: "0.055em", textTransform: "uppercase" }}>
                  {t("sidebar.recentPages")}
                </p>
                <RecentPages
                  ariaLabel={t("sidebar.recentPages")}
                  currentPageId={currentPageId}
                  onSelectPage={closeAfterNavigation(onSelectPage, closeOverlay)}
                  pages={recentPages}
                />
              </section>
            ) : undefined}
            recentSpacesSection={recentSpaces.length > 0 ? (
              <section>
                <p className="mb-2 px-1" style={{ color: "var(--mem-text-tertiary)", fontFamily: "var(--mem-font-mono)", fontSize: "10px", fontWeight: 600, letterSpacing: "0.055em", textTransform: "uppercase" }}>
                  {t("sidebar.recentSpaces")}
                </p>
                <RecentSpaces
                  ariaLabel={t("sidebar.recentSpaces")}
                  currentSpaceId={currentSpaceId}
                  onSelectSpace={closeAfterNavigation(onSelectSpace, closeOverlay)}
                  spaces={recentSpaces}
                />
              </section>
            ) : undefined}
          />
        </div>

        <div className="px-4 pt-2 pb-3 flex-shrink-0">
          <ReviewEnvironmentBadge />
          <IdentityCard
            onOpenDetail={closeAfterNavigation(onEntityClick, closeOverlay)}
            onOpenSettings={closeAfterNavigation(onNavigateSettings, closeOverlay)}
            onOpenAbout={closeAfterNavigation(onOpenAbout, closeOverlay)}
          />
        </div>
      </div>
      </aside>
    </>
  );
}

export function SidebarHeaderDivider({ visible }: { readonly visible: boolean }) {
  if (!visible) return null;

  return (
    <span
      aria-hidden="true"
      data-sidebar-header-divider="true"
      style={{
        backgroundColor: "var(--mem-border)",
        height: 52,
        left: 239,
        pointerEvents: "none",
        position: "absolute",
        top: 0,
        width: 1,
        zIndex: 1,
      }}
    />
  );
}

/** Sidebar toggle button for use in the header */
export const SidebarToggleButton = forwardRef<HTMLButtonElement, { readonly collapsed: boolean; readonly onToggle: () => void }>(function SidebarToggleButton({ collapsed, onToggle }, ref) {
  const { t } = useTranslation();
  return (
    <button
      data-sidebar-toggle="true"
      ref={ref}
      onClick={onToggle}
      className="flex items-center justify-center rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover-strong)]"
      style={{
        width: 28,
        height: 28,
        color: "var(--mem-text-tertiary)",
      }}
      title={collapsed ? t("sidebar.show") : t("sidebar.hide")}
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
});
