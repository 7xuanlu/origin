// SPDX-License-Identifier: AGPL-3.0-only
import type { SearchResult } from "../lib/tauri";
import ResultCard from "./ResultCard";

interface ResultListProps {
  results: SearchResult[];
  selectedIndex: number;
  query: string;
  onSelect: (index: number) => void;
  onOpen: (index: number) => void;
  onCopy: (index: number) => void;
}

export default function ResultList({
  results,
  selectedIndex,
  query,
  onSelect,
  onOpen,
  onCopy,
}: ResultListProps) {
  if (results.length === 0) {
    return null;
  }

  return (
    <div className="divide-y divide-[var(--separator)]">
      {results.map((result, index) => (
        <ResultCard
          key={result.id}
          result={result}
          query={query}
          isSelected={index === selectedIndex}
          onSelect={() => onSelect(index)}
          onOpen={() => onOpen(index)}
          onCopy={() => onCopy(index)}
        />
      ))}
    </div>
  );
}
