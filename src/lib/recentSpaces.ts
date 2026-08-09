// SPDX-License-Identifier: AGPL-3.0-only
import type { Space } from "./tauri";

export const RECENT_SPACES_STORAGE_KEY = "wenlan:recent-spaces:v1";

const HISTORY_VERSION = 1 as const;
const HISTORY_LIMIT = 50;
const RANKED_LIMIT = 4;
const VISIT_TTL_MS = 30 * 86_400_000;

export type RecentSpaceEntry = {
  readonly id: string;
  readonly name: string;
  readonly visitedAt: number;
};

export type RecentSpaceHistory = {
  readonly version: typeof HISTORY_VERSION;
  readonly entries: readonly RecentSpaceEntry[];
};

export type RecentSpacesStorage = {
  readonly getItem: (key: string) => string | null;
  readonly setItem: (key: string, value: string) => void;
};

export type RecentSpacesRuntime = {
  readonly storage?: RecentSpacesStorage | null;
  readonly spaces?: readonly Space[];
  readonly now?: number;
};

const EMPTY_HISTORY: RecentSpaceHistory = {
  version: HISTORY_VERSION,
  entries: [],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseEntry(value: unknown): RecentSpaceEntry | null {
  if (!isRecord(value)) return null;
  const id = value["id"];
  const name = value["name"];
  const visitedAt = value["visitedAt"];
  if (typeof id !== "string" || id.length === 0) return null;
  if (typeof name !== "string" || name.length === 0) return null;
  if (typeof visitedAt !== "number") return null;
  return { id, name, visitedAt };
}

function parseHistory(raw: string): RecentSpaceHistory {
  try {
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value) || value["version"] !== HISTORY_VERSION) return EMPTY_HISTORY;
    const rawEntries = value["entries"];
    if (!Array.isArray(rawEntries)) return EMPTY_HISTORY;
    const entries: RecentSpaceEntry[] = [];
    for (const rawEntry of rawEntries) {
      const entry = parseEntry(rawEntry);
      if (entry === null) return EMPTY_HISTORY;
      entries.push(entry);
    }
    return { version: HISTORY_VERSION, entries };
  } catch {
    return EMPTY_HISTORY;
  }
}

function resolveStorage(runtime?: RecentSpacesRuntime): RecentSpacesStorage | null {
  if (runtime?.storage !== undefined) return runtime.storage;
  try {
    return globalThis.localStorage;
  } catch {
    return null;
  }
}

function isValidVisit(visitedAt: number, now: number): boolean {
  return Number.isFinite(visitedAt)
    && visitedAt >= 0
    && visitedAt <= now
    && now - visitedAt < VISIT_TTL_MS;
}

function normalizeEntries(
  entries: readonly RecentSpaceEntry[],
  runtime?: RecentSpacesRuntime,
): readonly RecentSpaceEntry[] {
  const now = runtime?.now ?? Date.now();
  const currentById = runtime?.spaces === undefined
    ? null
    : new Map(
      runtime.spaces
        .filter(({ suggested }) => !suggested)
        .map((space) => [space.id, space] as const),
    );
  const newestById = new Map<string, RecentSpaceEntry>();

  for (const entry of entries) {
    if (!isValidVisit(entry.visitedAt, now)) continue;
    const current = currentById?.get(entry.id);
    if (currentById !== null && current === undefined) continue;
    const refreshed = current === undefined ? entry : { ...entry, name: current.name };
    const previous = newestById.get(entry.id);
    if (previous === undefined || refreshed.visitedAt > previous.visitedAt) {
      newestById.set(entry.id, refreshed);
    }
  }

  return [...newestById.values()]
    .sort((left, right) => {
      if (left.visitedAt !== right.visitedAt) return right.visitedAt - left.visitedAt;
      return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
    })
    .slice(0, HISTORY_LIMIT);
}

function writeHistory(
  entries: readonly RecentSpaceEntry[],
  runtime?: RecentSpacesRuntime,
): void {
  const storage = resolveStorage(runtime);
  if (storage === null) return;
  const history: RecentSpaceHistory = {
    version: HISTORY_VERSION,
    entries: normalizeEntries(entries, runtime),
  };
  try {
    storage.setItem(RECENT_SPACES_STORAGE_KEY, JSON.stringify(history));
  } catch {
    return;
  }
}

export function readRecentSpaceHistory(
  runtime?: RecentSpacesRuntime,
): RecentSpaceHistory {
  const storage = resolveStorage(runtime);
  if (storage === null) return EMPTY_HISTORY;
  try {
    const raw = storage.getItem(RECENT_SPACES_STORAGE_KEY);
    if (raw === null) return EMPTY_HISTORY;
    const parsed = parseHistory(raw);
    return {
      version: HISTORY_VERSION,
      entries: normalizeEntries(parsed.entries, runtime),
    };
  } catch {
    return EMPTY_HISTORY;
  }
}

export function recordRecentSpaceVisit(
  space: Pick<Space, "id" | "name" | "suggested">,
  runtime?: RecentSpacesRuntime,
): void {
  if (space.suggested) return;
  const now = runtime?.now ?? Date.now();
  const existing = readRecentSpaceHistory(runtime).entries
    .filter(({ id }) => id !== space.id);
  writeHistory([{ id: space.id, name: space.name, visitedAt: now }, ...existing], runtime);
}

export function renameRecentSpace(
  space: Pick<Space, "id" | "name">,
  runtime?: RecentSpacesRuntime,
): void {
  const entries = readRecentSpaceHistory(runtime).entries.map((entry) =>
    entry.id === space.id ? { ...entry, name: space.name } : entry,
  );
  writeHistory(entries, runtime);
}

export function deleteRecentSpace(
  id: string,
  runtime?: RecentSpacesRuntime,
): void {
  const entries = readRecentSpaceHistory(runtime).entries
    .filter((entry) => entry.id !== id);
  writeHistory(entries, runtime);
}

function compareNames(left: Space, right: Space): number {
  if (left.name !== right.name) return left.name < right.name ? -1 : 1;
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
}

export function rankRecentSpaces(
  spaces: readonly Space[],
  history: RecentSpaceHistory,
  now: number,
): readonly Space[] {
  const confirmedById = new Map<string, Space>();
  for (const space of spaces) {
    if (!space.suggested && !confirmedById.has(space.id)) confirmedById.set(space.id, space);
  }
  const visits = new Map(
    normalizeEntries(history.entries, { now }).map((entry) => [entry.id, entry] as const),
  );
  return [...confirmedById.values()]
    .filter(({ id }) => visits.has(id))
    .sort((left, right) => {
      const leftVisit = visits.get(left.id)?.visitedAt ?? 0;
      const rightVisit = visits.get(right.id)?.visitedAt ?? 0;
      return rightVisit - leftVisit || compareNames(left, right);
    })
    .slice(0, RANKED_LIMIT);
}
