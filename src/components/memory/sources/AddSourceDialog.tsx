import { useState, useEffect, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { addSource, syncRegisteredSource } from "../../../lib/tauri";
import { detectVault, type VaultDetection } from "../../../lib/vaultDetection";

interface Props {
  onClose: () => void;
  onSuccess: () => void;
}

export default function AddSourceDialog({ onClose, onSuccess }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [path, setPath] = useState("");
  const [detection, setDetection] = useState<VaultDetection | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addMutation = useMutation({
    mutationFn: ({ sourceType, sourcePath }: { sourceType: "obsidian" | "directory"; sourcePath: string }) =>
      addSource(sourceType, sourcePath),
    onSuccess: (newSource, { sourceType }) => {
      queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
      // Directory sources ride the daemon's 30s scheduler; Obsidian vaults are not
      // on it, so kick a one-shot first index or a new vault sits at "Indexing…"
      // forever (integration-review finding: scheduler filters to Directory only).
      if (sourceType === "obsidian") {
        syncRegisteredSource(newSource.id).then(() => {
          queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
        });
      }
      onSuccess();
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : String(err));
    },
  });

  const handleBrowse = useCallback(async () => {
    try {
      const selected = await openDialog({ directory: true, multiple: false });
      if (!selected) return;
      const selectedPath = typeof selected === "string" ? selected : selected;
      setPath(selectedPath);
      setDetecting(true);
      setError(null);
      setDetection(null);

      const d = await detectVault(selectedPath);
      setDetection(d);
      setDetecting(false);
    } catch {
      // Dialog cancelled
    }
  }, []);

  const handleSubmit = useCallback(() => {
    setError(null);
    addMutation.mutate({
      sourceType: detection?.sourceType ?? "directory",
      sourcePath: path,
    });
  }, [addMutation, detection, path]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const canSubmit = path.length > 0 && !detecting && !addMutation.isPending;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-[28rem] rounded-lg bg-[var(--mem-surface)] p-6 shadow-xl">
        <h3 className="text-sm font-medium text-[var(--mem-text)] mb-1">
          Add a folder
        </h3>
        <p className="text-xs text-[var(--mem-text-secondary)] mb-4">
          Wenlan reads .md, .txt, and .pdf files from the folder you choose —
          Obsidian vaults, note folders, books, and papers.
        </p>

        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="Select a folder…"
            readOnly
            className="flex-1 rounded-md border border-[var(--mem-border)] bg-[var(--mem-bg)] px-3 py-2 text-sm text-[var(--mem-text)]"
          />
          <button
            onClick={handleBrowse}
            className="rounded-md border border-[var(--mem-border)] px-3 py-2 text-sm text-[var(--mem-text)] hover:bg-[var(--mem-hover)]"
          >
            Browse…
          </button>
        </div>

        {detecting && (
          <p className="text-xs text-[var(--mem-text-secondary)] mb-3">
            Scanning folder…
          </p>
        )}

        {detection && !detecting && (
          <p className="text-xs mb-3">
            {detection.isVault ? (
              <span className="text-[var(--mem-accent-indigo)]">
                ✓ Detected .obsidian/ — Obsidian vault
              </span>
            ) : detection.docCount > 0 ? (
              <span className="text-[var(--mem-text-secondary)]">
                {detection.countCapped
                  ? t("vaultConnect.filesFoundCapped")
                  : t("vaultConnect.filesFound", { count: detection.docCount })}
              </span>
            ) : (
              <span className="text-[var(--mem-accent-amber)]">
                {t("vaultConnect.noneFound")}
              </span>
            )}
          </p>
        )}

        {error && (
          <p className="text-xs text-red-500 mb-3">{error}</p>
        )}

        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded-md px-3 py-1.5 text-xs text-[var(--mem-text-secondary)] border border-[var(--mem-border)] hover:bg-[var(--mem-hover)]"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="rounded-md bg-[var(--mem-accent-indigo)] px-3 py-1.5 text-xs text-white hover:opacity-90 disabled:opacity-50"
          >
            {addMutation.isPending ? "Adding\u2026" : "Add source"}
          </button>
        </div>
      </div>
    </div>
  );
}
