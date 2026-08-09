// SPDX-License-Identifier: AGPL-3.0-only
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { MemoryItem } from "../../lib/tauri";

interface MemoryListSurfaceProps {
  toolbar: ReactNode;
  memories: readonly MemoryItem[];
  filteredToEmpty: boolean;
  renderMemory: (memory: MemoryItem, index: number) => ReactNode;
  undoPending: boolean;
  onUndo: () => void;
}

export default function MemoryListSurface({
  toolbar,
  memories,
  filteredToEmpty,
  renderMemory,
  undoPending,
  onUndo,
}: MemoryListSurfaceProps) {
  const { t } = useTranslation();

  return (
    <section className="memory-list-shell" aria-label={t("memoryList.label")}>
      {toolbar}

      {memories.length > 0 ? (
        <div className="memory-list-rows">
          {memories.map((memory, index) => renderMemory(memory, index))}
        </div>
      ) : filteredToEmpty ? (
        <div className="memory-list-empty">
          <span>{t("memoryList.noMemoriesMatch")}</span>
        </div>
      ) : (
        <div className="memory-list-empty">
          <span>{t("memoryList.noMemoriesYet")}</span>
        </div>
      )}

      {undoPending && (
        <div
          className="fixed bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-3 px-4 py-2.5 rounded-lg shadow-lg"
          style={{
            backgroundColor: "var(--mem-text)",
            color: "var(--mem-bg)",
            fontFamily: "var(--mem-font-body)",
            fontSize: "13px",
            animation: "mem-fade-up 300ms cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        >
          <span>{t("memoryList.memoryDeleted")}</span>
          <button
            onClick={onUndo}
            className="font-medium underline"
            style={{ color: "var(--mem-accent-glow)" }}
          >
            {t("memoryList.undo")}
          </button>
        </div>
      )}
    </section>
  );
}
