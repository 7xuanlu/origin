// SPDX-License-Identifier: AGPL-3.0-only
import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import {
  addSource,
  syncRegisteredSource,
  listRegisteredSources,
  detectObsidianVaults,
  type RegisteredSource,
  type ObsidianVault,
  type SourceTypeStr,
} from "../../../lib/tauri";
import { detectVault, type VaultDetection } from "../../../lib/vaultDetection";
import { Button } from "../settings/primitives";

/** A recorded choice, not yet acted on. The wizard variant reports this
 *  upward on every change instead of calling addSource/syncRegisteredSource
 *  itself — those calls happen exactly once, on the "Setting up" step, so
 *  the user can watch them and get real SyncStats back. */
export interface VaultPick {
  sourceType: SourceTypeStr;
  path: string;
  label: string;
}

interface Props {
  variant: "dialog" | "wizard";
  onConnected?: (source: RegisteredSource) => void;
  /** Wizard variant only: called with the current pick every time it
   *  changes (a chip click or a browsed folder), so the wizard always
   *  knows what Continue should carry forward — never called for "dialog". */
  onPick?: (pick: VaultPick) => void;
}

export default function VaultConnectCard({ variant, onConnected, onPick }: Props) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [path, setPath] = useState("");
  const [detection, setDetection] = useState<VaultDetection | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [connectedId, setConnectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pickedVault, setPickedVault] = useState<ObsidianVault | null>(null);

  // Zero-friction detection: offer the user's real Obsidian vaults (read from
  // Obsidian's own registry, Rust-side) as one-tap chips. Empty on any
  // failure or when Obsidian isn't installed — the card just behaves as it
  // does today.
  const { data: obsidianVaults } = useQuery({
    queryKey: ["obsidian-vaults"],
    queryFn: detectObsidianVaults,
  });

  // Post-connect: poll the registered source until it reports counts
  // ("Indexed N files · M memories"). Dialog variant only — the wizard
  // variant never connects here at all (see onPick below); it hands the
  // pick to the "Setting up" step, which runs addSource/sync itself and
  // shows the real SyncStats it gets back.
  const { data: connectedSource } = useQuery({
    queryKey: ["vault-connect-progress", connectedId],
    queryFn: async () => {
      const sources = await listRegisteredSources();
      return sources.find((s) => s.id === connectedId) ?? null;
    },
    enabled: variant === "dialog" && connectedId !== null,
    refetchInterval: 2000,
  });

  // Wizard variant: report the current pick upward on every change instead
  // of connecting. A ref avoids re-firing just because the parent passed a
  // fresh onPick closure.
  const onPickRef = useRef(onPick);
  useEffect(() => {
    onPickRef.current = onPick;
  }, [onPick]);
  useEffect(() => {
    if (variant !== "wizard" || !path) return;
    const sourceType: SourceTypeStr = pickedVault ? "obsidian" : (detection?.sourceType ?? "directory");
    const label = pickedVault?.name ?? path.split("/").filter(Boolean).pop() ?? path;
    onPickRef.current?.({ sourceType, path, label });
  }, [variant, path, pickedVault, detection]);

  const handleBrowse = useCallback(async () => {
    const selected = await openDialog({ directory: true, multiple: false });
    if (!selected || typeof selected !== "string") return;
    setPath(selected);
    setError(null);
    setDetection(null);
    setPickedVault(null);
    setDetecting(true);
    setDetection(await detectVault(selected));
    setDetecting(false);
  }, []);

  // Chip-picked vaults are mutually exclusive with a browsed path: we can't
  // run detectVault() on a chip's path (the webview never picked it through
  // the dialog, so it's outside the fs scope), so there's no detection to
  // clear here — just the reverse direction (see handleBrowse above).
  const handlePickVault = useCallback((vault: ObsidianVault) => {
    setError(null);
    setDetection(null);
    setPath(vault.path);
    setPickedVault(vault);
  }, []);

  const handleConnect = useCallback(async () => {
    setError(null);
    setConnecting(true);
    try {
      const sourceType = pickedVault ? "obsidian" : (detection?.sourceType ?? "directory");
      const source = await addSource(sourceType, path);
      queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
      // Obsidian vaults are not on the daemon's 30s directory scheduler —
      // kick a one-shot first index (same rationale as AddSourceDialog).
      if (source.source_type === "obsidian") {
        syncRegisteredSource(source.id).then(() => {
          queryClient.invalidateQueries({ queryKey: ["registeredSources"] });
          queryClient.invalidateQueries({ queryKey: ["vault-connect-progress", source.id] });
        });
      }
      setConnectedId(source.id);
      onConnected?.(source);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setConnecting(false);
    }
  }, [detection, pickedVault, path, queryClient, onConnected]);

  // Submit is never blocked by a zero count (council change e): the daemon's
  // POST /api/sources validation is the authority; its 4xx surfaces verbatim.
  const canSubmit = path.length > 0 && !detecting && !connecting && connectedId === null;

  return (
    <div
      className="rounded-xl p-4 flex flex-col"
      style={{ border: "1px solid var(--mem-border)", backgroundColor: "var(--mem-surface)", gap: "12px" }}
    >
      <div>
        <h3 style={{ fontFamily: "var(--mem-font-body)", fontSize: "var(--mem-text-lg)", fontWeight: 600, color: "var(--mem-text)" }}>
          {t("vaultConnect.title")}
        </h3>
        <p style={{ fontFamily: "var(--mem-font-body)", fontSize: "12px", color: "var(--mem-text-secondary)", lineHeight: 1.5, marginTop: "4px" }}>
          {t("vaultConnect.description")}
        </p>
        {/* Obsidian support is stated ALWAYS, not only when a vault happens to be
            detected — the chip row below is conditional, so without this a user
            with Obsidian installed (but no registry entry we can read) has no way
            to know it's supported at all. The two lines also carry the real
            difference in what gets indexed, which used to appear only as a
            post-detection hint. */}
        <div style={{ display: "flex", flexDirection: "column", gap: "2px", marginTop: "8px" }}>
          {[t("vaultConnect.supportsObsidian"), t("vaultConnect.supportsFolder")].map((line) => (
            <span
              key={line}
              style={{ fontFamily: "var(--mem-font-body)", fontSize: "11px", color: "var(--mem-text-tertiary)", lineHeight: 1.5 }}
            >
              {line}
            </span>
          ))}
        </div>
      </div>

      {obsidianVaults && obsidianVaults.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          <span
            style={{
              fontFamily: "var(--mem-font-body)",
              fontSize: "11px",
              color: "var(--mem-text-tertiary)",
              textTransform: "uppercase",
              letterSpacing: "0.04em",
            }}
          >
            {t("vaultConnect.obsidianVaultsFound")}
          </span>
          <div className="flex flex-wrap gap-2">
            {obsidianVaults.map((vault) => {
              const picked = pickedVault?.path === vault.path;
              return (
                <button
                  key={vault.path}
                  type="button"
                  onClick={() => handlePickVault(vault)}
                  aria-label={vault.path}
                  className="rounded-full px-3 py-1"
                  style={{
                    fontFamily: "var(--mem-font-body)",
                    fontSize: "12px",
                    border: picked
                      ? "1px solid var(--mem-accent-indigo)"
                      : "1px solid var(--mem-border)",
                    backgroundColor: picked
                      ? "color-mix(in srgb, var(--mem-accent-indigo) 14%, transparent)"
                      : "var(--mem-bg)",
                    color: "var(--mem-text)",
                  }}
                >
                  {vault.name}
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="flex gap-2">
        <input
          type="text"
          value={path}
          readOnly
          placeholder={t("vaultConnect.placeholder")}
          className="flex-1 rounded-md px-3 py-2 text-sm"
          style={{
            border: "1px solid var(--mem-border)",
            backgroundColor: "var(--mem-bg)",
            color: "var(--mem-text)",
            fontFamily: "var(--mem-font-mono)",
            fontSize: "12px",
          }}
        />
        <Button type="button" variant="secondary" onClick={handleBrowse}>
          {t("vaultConnect.browse")}
        </Button>
      </div>

      {detecting && (
        <p style={{ fontSize: "12px", color: "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)" }}>
          {t("vaultConnect.scanning")}
        </p>
      )}

      {detection && !detecting && !pickedVault && (
        <div style={{ fontSize: "12px", fontFamily: "var(--mem-font-body)", display: "flex", flexDirection: "column", gap: "2px" }}>
          {detection.isVault && (
            <span style={{ color: "var(--mem-accent-indigo)" }}>{t("vaultConnect.detectedVault")}</span>
          )}
          {detection.docCount > 0 ? (
            <span style={{ color: "var(--mem-text-secondary)" }}>
              {detection.countCapped
                ? t("vaultConnect.filesFoundCapped")
                : t("vaultConnect.filesFound", { count: detection.docCount })}
            </span>
          ) : (
            <span style={{ color: "var(--mem-accent-amber)" }}>{t("vaultConnect.noneFound")}</span>
          )}
          {detection.isVault && !detection.hasValidDoc && (
            <span style={{ color: "var(--mem-text-tertiary)" }}>{t("vaultConnect.vaultMarkdownOnly")}</span>
          )}
        </div>
      )}

      {pickedVault && (
        <p style={{ fontSize: "12px", fontFamily: "var(--mem-font-body)", color: "var(--mem-accent-indigo)" }}>
          {t("vaultConnect.pickedVault", { name: pickedVault.name })}
        </p>
      )}

      {error && <p style={{ fontSize: "12px", fontFamily: "var(--mem-font-mono)", color: "var(--mem-status-danger-text)" }}>{error}</p>}

      {variant === "wizard" ? (
        path.length > 0 && (
          <p style={{ fontSize: "12px", fontFamily: "var(--mem-font-body)", color: "var(--mem-text-secondary)" }}>
            {t("vaultConnect.willImportOnSetup")}
          </p>
        )
      ) : connectedId === null ? (
        <Button type="button" variant="primary" onClick={handleConnect} disabled={!canSubmit} className="self-end">
          {connecting ? t("vaultConnect.connecting") : t("vaultConnect.connect")}
        </Button>
      ) : (
        <p style={{ fontSize: "12px", color: "var(--mem-text-secondary)", fontFamily: "var(--mem-font-body)" }}>
          {connectedSource && connectedSource.file_count > 0
            ? t("vaultConnect.indexed", {
                files: connectedSource.file_count,
                memories: connectedSource.memory_count,
              })
            : t("vaultConnect.indexing")}
        </p>
      )}
    </div>
  );
}
