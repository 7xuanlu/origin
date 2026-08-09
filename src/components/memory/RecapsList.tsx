// SPDX-License-Identifier: AGPL-3.0-only
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { deleteFileChunks, listMemoriesRich } from "../../lib/tauri";
import MemoryCard from "./MemoryCard";

type RecapsListProps = {
  readonly onBack: () => void;
  readonly onNavigateMemory: (sourceId: string) => void;
};

export function RecapsList({ onBack, onNavigateMemory }: RecapsListProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { data: recaps = [] } = useQuery({
    queryKey: ["all-recaps"],
    queryFn: async () => {
      const all = await listMemoriesRich(undefined, undefined, undefined, 200);
      return all.filter((memory) => memory.is_recap === true);
    },
  });
  const deleteMutation = useMutation({
    mutationFn: (sourceId: string) => deleteFileChunks("memory", sourceId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["all-recaps"] }),
  });

  return (
    <div>
      <button onClick={onBack} className="p-1.5 -ml-1.5 rounded-md transition-colors duration-150 hover:bg-[var(--mem-hover)] mb-3" style={{ color: "var(--mem-text-tertiary)", background: "none", border: "none", cursor: "pointer", lineHeight: 0 }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
      </button>
      <h2 style={{ fontFamily: "var(--mem-font-heading)", fontSize: "20px", fontWeight: 400, color: "var(--mem-text)", margin: "0 0 16px 0" }}>{t("main.recaps")}</h2>
      <h2 className="mb-4" style={{ fontFamily: "var(--mem-font-mono)", fontSize: "12px", fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase" as const, color: "var(--mem-accent-indigo)" }}>
        {t("main.allRecaps", { count: recaps.length })}
      </h2>
      <div className="flex flex-col">
        {recaps.map((recap) => (
          <MemoryCard key={recap.source_id} memory={recap} onConfirm={() => {}} onDelete={(sourceId) => deleteMutation.mutate(sourceId)} expandedChain={false} onToggleChain={() => {}} versionChain={[]} onClick={onNavigateMemory} />
        ))}
      </div>
    </div>
  );
}
