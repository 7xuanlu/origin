// SPDX-License-Identifier: AGPL-3.0-only
import type { Page } from "./tauri";

export const RECENT_PAGES_STORAGE_KEY = "wenlan:recent-pages:v1";

const HISTORY_VERSION = 1 as const;
const HISTORY_LIMIT = 50;
const RANKED_LIMIT = 4;
const VISIT_TTL_MS = 30 * 86_400_000;

export type RecentPageEntry = {
  readonly id: string;
  readonly title: string;
  readonly visitedAt: number;
};

export type RecentPageHistory = {
  readonly version: typeof HISTORY_VERSION;
  readonly entries: readonly RecentPageEntry[];
};

export type RecentPagesStorage = {
  readonly getItem: (key: string) => string | null;
  readonly setItem: (key: string, value: string) => void;
};

export type RecentPagesRuntime = {
  readonly now?: number;
  readonly pages?: readonly Page[];
  readonly storage?: RecentPagesStorage | null;
};

const EMPTY_HISTORY: RecentPageHistory = {
  version: HISTORY_VERSION,
  entries: [],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseEntry(value: unknown): RecentPageEntry | null {
  if (!isRecord(value)) return null;
  const id = value["id"];
  const title = value["title"];
  const visitedAt = value["visitedAt"];
  if (typeof id !== "string" || id.length === 0) return null;
  if (typeof title !== "string" || title.length === 0) return null;
  if (typeof visitedAt !== "number") return null;
  return { id, title, visitedAt };
}

function parseHistory(raw: string): RecentPageHistory {
  try {
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value) || value["version"] !== HISTORY_VERSION) return EMPTY_HISTORY;
    const rawEntries = value["entries"];
    if (!Array.isArray(rawEntries)) return EMPTY_HISTORY;
    const entries: RecentPageEntry[] = [];
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

function resolveStorage(runtime?: RecentPagesRuntime): RecentPagesStorage | null {
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
  entries: readonly RecentPageEntry[],
  runtime?: RecentPagesRuntime,
): readonly RecentPageEntry[] {
  const now = runtime?.now ?? Date.now();
  const currentById = runtime?.pages === undefined
    ? null
    : new Map(
      runtime.pages
        .filter(({ status }) => status === "active")
        .map((page) => [page.id, page] as const),
    );
  const newestById = new Map<string, RecentPageEntry>();

  for (const entry of entries) {
    if (!isValidVisit(entry.visitedAt, now)) continue;
    const current = currentById?.get(entry.id);
    if (currentById !== null && current === undefined) continue;
    const refreshed = current === undefined ? entry : { ...entry, title: current.title };
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
  entries: readonly RecentPageEntry[],
  runtime?: RecentPagesRuntime,
): void {
  const storage = resolveStorage(runtime);
  if (storage === null) return;
  const history: RecentPageHistory = {
    version: HISTORY_VERSION,
    entries: normalizeEntries(entries, runtime),
  };
  try {
    storage.setItem(RECENT_PAGES_STORAGE_KEY, JSON.stringify(history));
  } catch {
    return;
  }
}

export function readRecentPageHistory(
  runtime?: RecentPagesRuntime,
): RecentPageHistory {
  const storage = resolveStorage(runtime);
  if (storage === null) return EMPTY_HISTORY;
  try {
    const raw = storage.getItem(RECENT_PAGES_STORAGE_KEY);
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

export function recordRecentPageVisit(
  page: Pick<Page, "id" | "status" | "title">,
  runtime?: RecentPagesRuntime,
): void {
  if (page.status !== "active") return;
  const now = runtime?.now ?? Date.now();
  const existing = readRecentPageHistory(runtime).entries
    .filter(({ id }) => id !== page.id);
  writeHistory([{ id: page.id, title: page.title, visitedAt: now }, ...existing], runtime);
}

function compareTitles(left: Page, right: Page): number {
  if (left.title !== right.title) return left.title < right.title ? -1 : 1;
  return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
}

export function rankRecentPages(
  pages: readonly Page[],
  history: RecentPageHistory,
  now: number,
): readonly Page[] {
  const activeById = new Map<string, Page>();
  for (const page of pages) {
    if (page.status === "active" && !activeById.has(page.id)) activeById.set(page.id, page);
  }
  const visits = new Map(
    normalizeEntries(history.entries, { now }).map((entry) => [entry.id, entry] as const),
  );
  return [...activeById.values()]
    .filter(({ id }) => visits.has(id))
    .sort((left, right) => {
      const leftVisit = visits.get(left.id)?.visitedAt ?? 0;
      const rightVisit = visits.get(right.id)?.visitedAt ?? 0;
      return rightVisit - leftVisit || compareTitles(left, right);
    })
    .slice(0, RANKED_LIMIT);
}
