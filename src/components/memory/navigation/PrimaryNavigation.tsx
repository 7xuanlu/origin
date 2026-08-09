import type { GlobalNavigation } from "./viewState";
import { SpaceMark } from "./SpaceMark";

type PrimaryNavigationLabels = {
  readonly graph: string;
  readonly home: string;
  readonly memories: string;
  readonly navigation: string;
  readonly pages: string;
  readonly sources: string;
  readonly spaces: string;
};

type PrimaryNavigationProps = {
  readonly active: GlobalNavigation | null;
  readonly labels: PrimaryNavigationLabels;
  readonly onNavigateGraph?: () => void;
  readonly onNavigateHome?: () => void;
  readonly onNavigateLog?: () => void;
  readonly onNavigatePages?: () => void;
  readonly onNavigateSources?: () => void;
  readonly onNavigateSpaces: (create: boolean) => void;
  readonly recentPagesSection?: React.ReactNode;
  readonly recentSpacesSection?: React.ReactNode;
};

type NavButtonProps = {
  readonly active: boolean;
  readonly children: React.ReactNode;
  readonly icon: React.ReactNode;
  readonly onClick?: () => void;
};

function NavButton({ active, children, icon, onClick }: NavButtonProps) {
  if (onClick === undefined) return null;
  return (
    <button
      aria-current={active ? "page" : undefined}
      className="relative flex w-full items-center gap-2 rounded-md px-3 py-1.5 transition-colors duration-150 hover:bg-[var(--mem-hover)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mem-accent-page)]"
      onClick={onClick}
      style={{
        backgroundColor: active ? "var(--mem-indigo-bg)" : "transparent",
        fontWeight: active ? 500 : 400,
      }}
      type="button"
    >
      {active && (
        <span
          aria-hidden="true"
          className="absolute inset-y-1 left-0 w-0.5 rounded-full bg-[var(--mem-accent-indigo)]"
          data-primary-navigation-active-marker="true"
        />
      )}
      {icon}
      <span style={{ color: active ? "var(--mem-text)" : "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)", fontSize: "13px" }}>
        {children}
      </span>
    </button>
  );
}

const iconStyle = { color: "var(--mem-text-tertiary)" } as const;

export function PrimaryNavigation({
  active,
  labels,
  onNavigateGraph,
  onNavigateHome,
  onNavigateLog,
  onNavigatePages,
  onNavigateSources,
  onNavigateSpaces,
  recentPagesSection,
  recentSpacesSection,
}: PrimaryNavigationProps) {
  return (
    <>
      <nav aria-label={labels.navigation} className="flex flex-col pb-4">
        <NavButton
          active={active === "home"}
          icon={<svg aria-hidden="true" height="14" style={iconStyle} viewBox="0 0 24 24" width="14"><path d="M3 10.5L12 3l9 7.5M5 9.5V21h14V9.5M9.5 21v-6h5v6" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" /></svg>}
          onClick={onNavigateHome}
        >
          {labels.home}
        </NavButton>
        <NavButton
          active={active === "pages"}
          icon={<svg aria-hidden="true" data-navigation-icon="wiki-page" height="14" style={{ color: active === "pages" ? "var(--mem-page-icon)" : "var(--mem-text-tertiary)" }} viewBox="0 0 24 24" width="14"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" /></svg>}
          onClick={onNavigatePages}
        >
          {labels.pages}
        </NavButton>
        <NavButton
          active={active === "spaces"}
          icon={<SpaceMark active={active === "spaces"} />}
          onClick={() => onNavigateSpaces(false)}
        >
          {labels.spaces}
        </NavButton>
        <NavButton
          active={active === "graph"}
          icon={<svg aria-hidden="true" data-navigation-icon="graph" height="14" style={iconStyle} viewBox="0 0 24 24" width="14"><circle cx="5" cy="6" fill="none" r="2" stroke="currentColor" strokeWidth="2" /><circle cx="19" cy="6" fill="none" r="2" stroke="currentColor" strokeWidth="2" /><circle cx="12" cy="18" fill="none" r="2" stroke="currentColor" strokeWidth="2" /><path d="M7 6h10M6 8l5 8M18 8l-5 8" fill="none" stroke="currentColor" strokeWidth="2" /></svg>}
          onClick={onNavigateGraph}
        >
          {labels.graph}
        </NavButton>
        <NavButton
          active={active === "memories"}
          icon={(
            <svg aria-hidden="true" data-navigation-icon="brain" fill="none" height="14" style={iconStyle} viewBox="0 0 24 24" width="14">
              {[
                "M15.5 13a3.5 3.5 0 0 0 -3.5 3.5v1a3.5 3.5 0 0 0 7 0v-1.8",
                "M8.5 13a3.5 3.5 0 0 1 3.5 3.5v1a3.5 3.5 0 0 1 -7 0v-1.8",
                "M17.5 16a3.5 3.5 0 0 0 0 -7h-.5",
                "M19 9.3v-2.8a3.5 3.5 0 0 0 -7 0",
                "M6.5 16a3.5 3.5 0 0 1 0 -7h.5",
                "M5 9.3v-2.8a3.5 3.5 0 0 1 7 0v10",
              ].map((geometry) => (
                <path
                  d={geometry}
                  key={geometry}
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="1.8"
                />
              ))}
            </svg>
          )}
          onClick={onNavigateLog}
        >
          {labels.memories}
        </NavButton>
        <NavButton
          active={active === "sources"}
          icon={(
            <svg
              aria-hidden="true"
              data-navigation-icon="sources-intake-tray"
              height="14"
              style={iconStyle}
              viewBox="0 0 24 24"
              width="14"
            >
              <path
                d="M7 4v6M12 4v6M17 4v6"
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeWidth="1.8"
              />
              <path
                d="M5 13l1.5 5h11l1.5-5"
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.8"
              />
            </svg>
          )}
          onClick={onNavigateSources}
        >
          {labels.sources}
        </NavButton>
      </nav>
      {(recentPagesSection !== undefined || recentSpacesSection !== undefined) && (
        <div className="mt-1 flex flex-col gap-5 border-t pt-5" style={{ borderColor: "var(--mem-border)" }}>
          {recentPagesSection}
          {recentSpacesSection}
        </div>
      )}
    </>
  );
}
