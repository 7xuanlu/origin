// SPDX-License-Identifier: AGPL-3.0-only
import type { MemoryStats } from "../../lib/tauri";

interface MemoryStatsViewProps {
  stats: MemoryStats | undefined;
}

export default function MemoryStatsView({ stats }: MemoryStatsViewProps) {
  if (!stats) return null;

  const items = [
    { label: "memories", value: stats.total },
    { label: "new today", value: stats.new_today },
  ];

  return (
    <div className="flex flex-col gap-1.5">
      {items.map((item) => (
        <div
          key={item.label}
          className="flex items-center justify-between"
          style={{
            fontFamily: "var(--mem-font-mono)",
            fontSize: "11px",
            color: "var(--mem-text-tertiary)",
          }}
        >
          <span>{item.value}</span>
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  );
}
