// SPDX-License-Identifier: AGPL-3.0-only
const STORAGE_KEY = "origin:collapsed-sections";
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

interface StoredEntry {
  id: string;
  collapsedAt: number;
}

function load(): Map<string, number> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return new Map();
    const entries: StoredEntry[] = JSON.parse(raw);
    const now = Date.now();
    const map = new Map<string, number>();
    for (const e of entries) {
      if (now - e.collapsedAt < MAX_AGE_MS) {
        map.set(e.id, e.collapsedAt);
      }
    }
    return map;
  } catch {
    return new Map();
  }
}

function save(map: Map<string, number>): void {
  const entries: StoredEntry[] = [];
  for (const [id, collapsedAt] of map) {
    entries.push({ id, collapsedAt });
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
}

/** Return the set of currently collapsed section IDs. */
export function getCollapsedSections(): Set<string> {
  return new Set(load().keys());
}

/** Toggle a section's collapsed state. Returns the updated collapsed set. */
export function toggleSection(id: string): Set<string> {
  const map = load();
  if (map.has(id)) {
    map.delete(id);
  } else {
    map.set(id, Date.now());
  }
  save(map);
  return new Set(map.keys());
}

/** Force-expand a section (remove from collapsed set). */
export function expandSection(id: string): void {
  const map = load();
  if (map.has(id)) {
    map.delete(id);
    save(map);
  }
}
