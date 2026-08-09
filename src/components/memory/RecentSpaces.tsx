// SPDX-License-Identifier: AGPL-3.0-only
import type { Space } from "../../lib/tauri";

export type RecentSpacesProps = {
  readonly ariaLabel: string;
  readonly spaces: readonly Space[];
  readonly currentSpaceId: string | null;
  readonly onSelectSpace: (space: Space) => void;
};

export function RecentSpaces({
  ariaLabel,
  spaces,
  currentSpaceId,
  onSelectSpace,
}: RecentSpacesProps) {
  const confirmed: Space[] = [];
  const seenIds = new Set<string>();
  for (const space of spaces) {
    if (space.suggested || seenIds.has(space.id)) continue;
    confirmed.push(space);
    seenIds.add(space.id);
    if (confirmed.length === 4) break;
  }

  return (
    <nav aria-label={ariaLabel} className="flex flex-col gap-0.5">
      {confirmed.map((space) => {
        const selected = currentSpaceId === space.id;
        return (
          <button
            key={space.id}
            type="button"
            aria-label={space.name}
            aria-current={selected ? "page" : undefined}
            data-selected={selected ? "true" : "false"}
            onClick={() => onSelectSpace(space)}
            className="flex w-full items-center justify-between rounded-md border-l-2 px-2 py-1.5 text-left transition-colors duration-150 hover:bg-[var(--mem-hover)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--mem-accent-page)]"
            style={{
              borderLeftColor: selected ? "var(--mem-accent-indigo)" : "transparent",
              backgroundColor: selected ? "var(--mem-hover-strong)" : "transparent",
              color: selected ? "var(--mem-text)" : "var(--mem-text-secondary)",
              fontFamily: "var(--mem-font-body)",
              fontSize: "13px",
            }}
          >
            <span className="flex min-w-0 items-center gap-1.5">
              <span
                aria-hidden="true"
                className="w-3 shrink-0 text-center"
                style={{ color: "var(--mem-accent-amber)", fontSize: "10px" }}
              >
                {space.starred ? "★" : ""}
              </span>
              <span className="truncate">{space.name}</span>
            </span>
          </button>
        );
      })}
    </nav>
  );
}
