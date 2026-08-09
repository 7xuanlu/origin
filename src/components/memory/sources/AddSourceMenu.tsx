// SPDX-License-Identifier: AGPL-3.0-only
import { useState } from "react";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  uploadSourceFile,
  getDaemonVersion,
  daemonMeetsFloor,
  openFile,
} from "../../../lib/tauri";
import AddSourceDialog from "./AddSourceDialog";

export default function AddSourceMenu({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [showFolder, setShowFolder] = useState(false);

  const { data: version } = useQuery({ queryKey: ["daemonVersion"], queryFn: getDaemonVersion });
  // Optimistic until the version is known, so the menu never flickers a warning.
  const ready = version === undefined ? true : daemonMeetsFloor(version);

  const upload = useMutation({
    mutationFn: async () => {
      const picked = await openDialog({
        directory: false,
        multiple: false,
        filters: [{ name: "Documents", extensions: ["pdf", "md", "txt"] }],
      });
      if (!picked || typeof picked !== "string") return null;
      const name = picked.split("/").pop() ?? "file";
      await uploadSourceFile(picked);
      // Toast idiom: eyebrow (title) / heading + body (description).
      toast("Added", { description: `${name} is on the shelf. Indexing in the background.` });
      return name;
    },
    onSuccess: (name) => {
      if (!name) return;
      qc.invalidateQueries({ queryKey: ["registeredSources"] });
      // The uploaded file lands in the managed dir; refresh its on-disk listing
      // and the index cross-ref so the new file appears immediately instead of
      // waiting for a window-focus refetch.
      qc.invalidateQueries({ queryKey: ["sourceDir"] });
      qc.invalidateQueries({ queryKey: ["indexedFiles"] });
      onClose();
    },
    onError: (e) => {
      // Without this the button silently reverts to "Add files" and the user
      // never learns the upload failed (daemon down / IO error / bad file).
      toast("Couldn't add file", { description: String(e) });
    },
  });

  if (showFolder) {
    return <AddSourceDialog onClose={onClose} onSuccess={onClose} />;
  }

  const item = "w-full text-left rounded-md px-3 py-2 text-sm hover:bg-[var(--mem-hover)]";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="w-[22rem] rounded-lg bg-[var(--mem-surface)] p-4 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-sm font-medium text-[var(--mem-text)] mb-2" style={{ fontFamily: "var(--mem-font-heading)" }}>
          Add a source
        </h3>
        {!ready && (
          <div className="mb-3 rounded-md p-3 text-xs" style={{ background: "var(--mem-hover)", color: "var(--mem-text-secondary)" }}>
            <p className="mb-2">Your daemon needs an update to index files.</p>
            <button className="underline" style={{ color: "var(--mem-accent-indigo)" }} onClick={() => openFile("https://wenlan.app")}>
              Update Wenlan
            </button>
          </div>
        )}
        <button className={item} style={{ color: "var(--mem-text)" }} onClick={() => setShowFolder(true)}>
          Add a folder
        </button>
        <button
          className={item}
          style={{ color: "var(--mem-text)" }}
          disabled={upload.isPending}
          onClick={() => upload.mutate()}
        >
          {upload.isPending ? "Adding…" : "Add files"}
        </button>
      </div>
    </div>
  );
}
