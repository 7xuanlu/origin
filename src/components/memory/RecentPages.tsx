// SPDX-License-Identifier: AGPL-3.0-only
import type { Page } from "../../lib/tauri";

export type RecentPagesProps = {
  readonly ariaLabel: string;
  readonly currentPageId: string | null;
  readonly onSelectPage: (page: Page) => void;
  readonly pages: readonly Page[];
};

export function RecentPages({
  ariaLabel,
  currentPageId,
  onSelectPage,
  pages,
}: RecentPagesProps) {
  const recent: Page[] = [];
  const seenIds = new Set<string>();
  for (const page of pages) {
    if (page.status !== "active" || seenIds.has(page.id)) continue;
    recent.push(page);
    seenIds.add(page.id);
    if (recent.length === 4) break;
  }

  return (
    <nav aria-label={ariaLabel} className="flex flex-col gap-0.5">
      {recent.map((page) => {
        const selected = currentPageId === page.id;
        return (
          <button
            aria-current={selected ? "page" : undefined}
            aria-label={page.title}
            className="flex w-full items-center rounded-md border-l-2 px-2 py-1.5 text-left transition-colors duration-150 hover:bg-[var(--mem-hover)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mem-accent-page)]"
            data-selected={selected ? "true" : "false"}
            key={page.id}
            onClick={() => onSelectPage(page)}
            style={{
              backgroundColor: selected ? "var(--mem-hover-strong)" : "transparent",
              borderLeftColor: selected ? "var(--mem-accent-indigo)" : "transparent",
              color: selected ? "var(--mem-text)" : "var(--mem-text-secondary)",
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
            }}
            type="button"
          >
            <span className="truncate pl-[18px]">{page.title}</span>
          </button>
        );
      })}
    </nav>
  );
}
