// SPDX-License-Identifier: AGPL-3.0-only
/**
 * Module-level store tracking the most recent capture event.
 * Persists across React component mounts/unmounts — survives navigation.
 * Fed by capture-event listener in App.tsx; consumed by MemoryView heartbeat indicator.
 */

type Listener = () => void;

let lastCapture: { source: string; timestamp: number } | null = null;
const listeners = new Set<Listener>();

export function recordCapture(source: string) {
  lastCapture = { source, timestamp: Date.now() };
  for (const fn of listeners) fn();
}

export function getLastCapture() {
  return lastCapture;
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  return () => { listeners.delete(fn); };
}
