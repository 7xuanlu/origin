import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  createPageDraft,
  discardPageDraft,
  updatePageDraft,
  type Page,
} from "../../../lib/tauri";

export type PageDraftSnapshot = {
  readonly title: string;
  readonly content: string;
  readonly space: string | null;
};

export type PageDraftAutosaveStatus =
  | "idle"
  | "saving"
  | "saved"
  | "error"
  | "conflict";

type RemoteDraft = {
  readonly draftId: string;
  readonly version: number;
  readonly snapshot: PageDraftSnapshot;
};

type PendingCreate = {
  readonly clientDraftId: string;
  readonly snapshot: PageDraftSnapshot;
};

type PendingDiscard = {
  readonly draftId: string;
  readonly version: number;
  readonly snapshot: PageDraftSnapshot;
};

type UsePageDraftAutosaveOptions = {
  readonly draftId?: string;
  readonly enabled?: boolean;
  readonly initial: PageDraftSnapshot;
  readonly initialVersion?: number;
  readonly onSpaceReconciled?: (space: string | null) => void;
  readonly snapshot: PageDraftSnapshot;
};

const INVENTORY_QUERY_KEYS = [
  ["pages"],
  ["pages", "active"],
  ["pages", "draft"],
  ["recent-concepts"],
  ["recent-pages"],
  ["space-pages"],
  ["spaces-page-counts"],
  ["sidebar-space-page-counts"],
] as const;

function sameSnapshot(left: PageDraftSnapshot, right: PageDraftSnapshot): boolean {
  return left.title === right.title
    && left.content === right.content
    && left.space === right.space;
}

function isMeaningful(snapshot: PageDraftSnapshot): boolean {
  return snapshot.title.trim().length > 0 || snapshot.content.trim().length > 0;
}

function newClientDraftId(): string {
  return `page_${crypto.randomUUID()}`;
}

function snapshotFromPage(page: Page): PageDraftSnapshot {
  return {
    title: page.title,
    content: page.content,
    space: page.space === undefined ? page.domain ?? null : page.space,
  };
}

function errorCode(error: unknown): string | null {
  if (!error || typeof error !== "object" || !("code" in error)) return null;
  const code = (error as { code?: unknown }).code;
  return typeof code === "string" ? code : null;
}

export function usePageDraftAutosave({
  draftId: initialDraftId,
  enabled = true,
  initial,
  initialVersion,
  onSpaceReconciled,
  snapshot,
}: UsePageDraftAutosaveOptions) {
  const queryClient = useQueryClient();
  const [draftId, setDraftId] = useState<string | null>(initialDraftId ?? null);
  const [version, setVersion] = useState<number | null>(initialVersion ?? null);
  const [status, setStatus] = useState<PageDraftAutosaveStatus>(
    initialDraftId ? "saved" : "idle",
  );
  const [error, setError] = useState<Error | null>(null);

  const mountedRef = useRef(true);
  const latestRef = useRef(snapshot);
  const persistedRef = useRef(initial);
  const draftIdRef = useRef<string | null>(initialDraftId ?? null);
  const clientDraftIdRef = useRef<string | null>(null);
  const versionRef = useRef<number | null>(initialVersion ?? null);
  const pendingCreateRef = useRef<PendingCreate | null>(null);
  const pendingDiscardRef = useRef<PendingDiscard | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loopRef = useRef<Promise<boolean> | null>(null);
  const conflictRef = useRef(false);

  latestRef.current = snapshot;
  if (clientDraftIdRef.current === null) {
    clientDraftIdRef.current = newClientDraftId();
  }

  const updateState = useCallback((
    nextStatus: PageDraftAutosaveStatus,
    nextError: Error | null = null,
  ) => {
    if (!mountedRef.current) return;
    setStatus(nextStatus);
    setError(nextError);
  }, []);

  const invalidateInventories = useCallback(async (pageId?: string | null) => {
    const invalidations = INVENTORY_QUERY_KEYS.map((queryKey) =>
      queryClient.invalidateQueries({ queryKey: [...queryKey] })
    );
    if (pageId) {
      invalidations.push(queryClient.invalidateQueries({ queryKey: ["page", pageId] }));
      invalidations.push(queryClient.invalidateQueries({ queryKey: ["page-draft", pageId] }));
    }
    await Promise.all(invalidations);
  }, [queryClient]);

  const saveLatest = useCallback(async (): Promise<boolean> => {
    if (conflictRef.current) return false;
    if (loopRef.current) return loopRef.current;

    const loop = (async () => {
      let idCollisionRetries = 0;
      while (enabled) {
        const next = latestRef.current;
        const currentId = draftIdRef.current;
        const pendingCreate = pendingCreateRef.current;
        const pendingDiscard = pendingDiscardRef.current;

        if (
          sameSnapshot(next, persistedRef.current)
          && !pendingCreate
          && !pendingDiscard
        ) {
          if (mountedRef.current && currentId) updateState("saved");
          return true;
        }

        if (!isMeaningful(next) && !currentId && !pendingCreate && !pendingDiscard) {
          // A Space selection alone is only local editor context.
          persistedRef.current = next;
          updateState("idle");
          return true;
        }

        updateState("saving");
        try {
          if (pendingDiscard) {
            try {
              await discardPageDraft({
                id: pendingDiscard.draftId,
                expectedVersion: pendingDiscard.version,
              });
            } catch (cause) {
              if (errorCode(cause) !== "page_draft_not_found") throw cause;
            }
            pendingDiscardRef.current = null;
            draftIdRef.current = null;
            clientDraftIdRef.current = newClientDraftId();
            versionRef.current = null;
            persistedRef.current = pendingDiscard.snapshot;
            if (mountedRef.current) {
              setDraftId(null);
              setVersion(null);
            }
            await invalidateInventories(pendingDiscard.draftId);
            updateState("idle");
            continue;
          }

          if (pendingCreate) {
            const saved = await createPageDraft({
              clientDraftId: pendingCreate.clientDraftId,
              ...pendingCreate.snapshot,
            });
            pendingCreateRef.current = null;
            draftIdRef.current = saved.id;
            versionRef.current = saved.version;
            const savedSnapshot = saved.id === pendingCreate.clientDraftId
              ? snapshotFromPage(saved)
              : pendingCreate.snapshot;
            if (
              saved.id === pendingCreate.clientDraftId
              && latestRef.current.space === pendingCreate.snapshot.space
              && savedSnapshot.space !== pendingCreate.snapshot.space
            ) {
              latestRef.current = {
                ...latestRef.current,
                space: savedSnapshot.space,
              };
              if (mountedRef.current) onSpaceReconciled?.(savedSnapshot.space);
            }
            persistedRef.current = savedSnapshot;
            if (mountedRef.current) {
              setDraftId(saved.id);
              setVersion(saved.version);
            }
            await invalidateInventories(saved.id);
            updateState("saved");
            continue;
          }

          if (!isMeaningful(next) && currentId) {
            const currentVersion = versionRef.current;
            if (currentVersion == null) throw new Error("Draft version is unavailable.");
            pendingDiscardRef.current = {
              draftId: currentId,
              version: currentVersion,
              snapshot: next,
            };
            continue;
          }

          let saved: Page;
          let savedSnapshot = next;
          if (!currentId) {
            const clientDraftId = clientDraftIdRef.current ?? newClientDraftId();
            clientDraftIdRef.current = clientDraftId;
            pendingCreateRef.current = {
              clientDraftId,
              snapshot: next,
            };
            saved = await createPageDraft({
              clientDraftId,
              ...next,
            });
            pendingCreateRef.current = null;
            if (saved.id === clientDraftId) {
              savedSnapshot = snapshotFromPage(saved);
            }
          } else {
            const currentVersion = versionRef.current;
            if (currentVersion == null) throw new Error("Draft version is unavailable.");
            saved = await updatePageDraft({
              id: currentId,
              expectedVersion: currentVersion,
              ...next,
            });
            savedSnapshot = snapshotFromPage(saved);
          }

          draftIdRef.current = saved.id;
          versionRef.current = saved.version;
          persistedRef.current = savedSnapshot;
          if (mountedRef.current) {
            setDraftId(saved.id);
            setVersion(saved.version);
          }
          await invalidateInventories(saved.id);
          updateState("saved");
        } catch (cause) {
          const nextError = cause instanceof Error ? cause : new Error(String(cause));
          const code = errorCode(cause);
          if (
            code === "page_draft_id_conflict"
            && !currentId
            && idCollisionRetries < 1
          ) {
            idCollisionRetries += 1;
            pendingCreateRef.current = null;
            clientDraftIdRef.current = newClientDraftId();
            continue;
          }
          if (code === "draft_version_conflict") {
            conflictRef.current = true;
            updateState("conflict", nextError);
          } else {
            updateState("error", nextError);
          }
          return false;
        }
      }
      return true;
    })();

    loopRef.current = loop;
    try {
      return await loop;
    } finally {
      if (loopRef.current === loop) loopRef.current = null;
    }
  }, [enabled, invalidateInventories, onSpaceReconciled, updateState]);

  const flush = useCallback(async (): Promise<boolean> => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    return saveLatest();
  }, [saveLatest]);

  const retry = useCallback(async (): Promise<boolean> => {
    if (conflictRef.current) return false;
    return flush();
  }, [flush]);

  const adoptRemote = useCallback((remote: RemoteDraft) => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    conflictRef.current = false;
    pendingCreateRef.current = null;
    pendingDiscardRef.current = null;
    draftIdRef.current = remote.draftId;
    versionRef.current = remote.version;
    persistedRef.current = remote.snapshot;
    latestRef.current = remote.snapshot;
    if (mountedRef.current) {
      setDraftId(remote.draftId);
      setVersion(remote.version);
      updateState("saved");
    }
  }, [updateState]);

  const getIdentity = useCallback(() => ({
    draftId: draftIdRef.current,
    version: versionRef.current,
  }), []);

  useEffect(() => {
    if (
      !enabled
      || conflictRef.current
      || (
        sameSnapshot(snapshot, persistedRef.current)
        && !pendingCreateRef.current
        && !pendingDiscardRef.current
      )
    ) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      void saveLatest();
    }, 700);
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [enabled, saveLatest, snapshot.content, snapshot.space, snapshot.title]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  return {
    adoptRemote,
    draftId,
    error,
    flush,
    getIdentity,
    retry,
    status,
    version,
  };
}
