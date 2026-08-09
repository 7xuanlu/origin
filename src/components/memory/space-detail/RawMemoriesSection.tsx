// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import type { MemoryItem } from "../../../lib/tauri";
import MemoryStream, { type SortMode } from "../MemoryStream";
import type { SpaceDetailCopy } from "./copy";
import { MEMORY_FETCH_LIMIT } from "./model";

type RawMemoriesSectionProps = {
  readonly copy: SpaceDetailCopy;
  readonly memories: readonly MemoryItem[];
  readonly onSelectMemory: (sourceId: string) => void;
  readonly totalMemoryCount: number;
};

export function RawMemoriesSection({
  copy,
  memories,
  onSelectMemory,
  totalMemoryCount,
}: RawMemoriesSectionProps) {
  const [expanded, setExpanded] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>("curated");
  const isTruncated = totalMemoryCount > memories.length
    && memories.length === MEMORY_FETCH_LIMIT;

  return (
    <section aria-label={copy.rawMemories} className="space-dossier-archive">
      <button
        aria-expanded={expanded}
        aria-label={`${copy.rawMemories} (${totalMemoryCount})`}
        className="space-dossier-archive-trigger"
        onClick={() => setExpanded((current) => !current)}
        type="button"
      >
        <svg aria-hidden="true" className={expanded ? "is-expanded" : ""} viewBox="0 0 24 24">
          <path d="m9 18 6-6-6-6" />
        </svg>
        <span>{copy.rawMemories}</span>
        <small>{new Intl.NumberFormat().format(totalMemoryCount)}</small>
      </button>

      {isTruncated && (
        <p className="space-dossier-archive-limit">
          {copy.showingLatest(MEMORY_FETCH_LIMIT, totalMemoryCount)}
        </p>
      )}

      {expanded && memories.length > 0 && (
        <MemoryStream
          cardVariant="insight"
          memories={[...memories]}
          onSelectMemory={onSelectMemory}
          onSortChange={setSortMode}
          selectedDomain={null}
          sortMode={sortMode}
        />
      )}
      {expanded && memories.length === 0 && (
        <p className="space-dossier-empty">{copy.noMemories}</p>
      )}
    </section>
  );
}
