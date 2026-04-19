import { useState, useEffect, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { readDir } from "@tauri-apps/plugin-fs";
import { addSource, syncRegisteredSource } from "../../../lib/tauri";

interface Detection {
  isVault: boolean;
  mdCount: number;
}

interface Props {
  onClose: () => void;
  onSuccess: () => void;
}

export default function AddSourceDialog({ onClose, onSuccess }: Props) {
  const queryClient = useQueryClient();
  const [path, setPath] = useState("");
  const [detection, setDetection] = useState<Detection | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addMutation = useMutation({
    mutationFn: ({ sourceType, sourcePath }: { sourceType: string; sourcePath: string }) =>
      addSource(sourceType as "obsidian", sourcePath),
    onSuccess: (newSource) => {
      // Auto-trigger sync in background (don't await)
      syncRegisteredSource(newSource.id).then(() => {
        queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
      });
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

      try {
        const entries = await readDir(selectedPath);
        const isVault = entries.some(
          (e) => e.name === ".obsidian" && e.isDirectory,
        );
        const mdCount = entries.filter(
          (e) => e.name?.endsWith(".md") && !e.isDirectory,
        ).length;
        setDetection({ isVault, mdCount });
      } catch {
        // If readDir fails, still allow the path but show no detection
        setDetection({ isVault: false, mdCount: 0 });
      }
      setDetecting(false);
    } catch {
      // Dialog cancelled
    }
  }, []);

  const handleSubmit = useCallback(() => {
    setError(null);
    addMutation.mutate({ sourceType: "obsidian", sourcePath: path });
  }, [addMutation, path]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const canSubmit =
    path.length > 0 &&
    !detecting &&
    !addMutation.isPending &&
    (detection === null || detection.mdCount > 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-[28rem] rounded-lg bg-[var(--mem-surface)] p-6 shadow-xl">
        <h3 className="text-sm font-medium text-[var(--mem-text)] mb-1">
          Add a markdown folder
        </h3>
        <p className="text-xs text-[var(--mem-text-secondary)] mb-4">
          Origin reads .md files from the folder you choose. Obsidian vaults,
          plain markdown directories, or any folder of notes.
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
            ) : detection.mdCount > 0 ? (
              <span className="text-[var(--mem-text-secondary)]">
                {detection.mdCount} markdown files found
              </span>
            ) : (
              <span className="text-red-500">
                No markdown files found in this folder
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
