// SPDX-License-Identifier: AGPL-3.0-only
/**
 * Module-level store tracking which source_ids are being AI-processed.
 * Persists across React component mounts/unmounts — survives navigation.
 * Fed by capture-event listeners; consumed by MemoryView and ChunkViewer.
 */

type Listener = () => void;

const processingIds = new Set<string>();
const listeners = new Set<Listener>();
let version = 0;

function notify() {
  version++;
  for (const fn of listeners) fn();
}

/** Snapshot for useSyncExternalStore — changes whenever the set changes. */
export function getSnapshot(): number {
  return version;
}

export function markProcessing(sourceId: string) {
  if (sourceId && !processingIds.has(sourceId)) {
    processingIds.add(sourceId);
    notify();
  }
}

export function clearProcessing(sourceId: string) {
  if (sourceId && processingIds.has(sourceId)) {
    processingIds.delete(sourceId);
    notify();
  }
}

export function isProcessing(sourceId: string): boolean {
  return processingIds.has(sourceId);
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
