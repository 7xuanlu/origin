// SPDX-License-Identifier: AGPL-3.0-only
import type { Entity, Page } from "../../../lib/tauri";

const RECENT_PAGE_LIMIT = 5;
const REVIEW_PAGE_LIMIT = 3;
export const KEY_ENTITY_LIMIT = 6;
export const PAGE_FETCH_LIMIT = 1_000;
export const MEMORY_FETCH_LIMIT = 200;

function pageTimestamp(page: Page): number | null {
  const timestamp = Date.parse(page.last_modified);
  return Number.isFinite(timestamp) ? timestamp : null;
}

function entityTimestamp(entity: Entity): number {
  return Number.isFinite(entity.updated_at) ? entity.updated_at : Number.NEGATIVE_INFINITY;
}

function comparePageRecency(left: Page, right: Page): number {
  const leftTimestamp = pageTimestamp(left) ?? Number.NEGATIVE_INFINITY;
  const rightTimestamp = pageTimestamp(right) ?? Number.NEGATIVE_INFINITY;
  const timestampDifference = rightTimestamp - leftTimestamp;
  return timestampDifference || left.title.localeCompare(right.title);
}

export function recentlyRefinedPages(pages: readonly Page[]): Page[] {
  return [...pages]
    .filter((page) => pageTimestamp(page) !== null)
    .sort(comparePageRecency)
    .slice(0, RECENT_PAGE_LIMIT);
}

export function pagesNeedingReview(pages: readonly Page[]): Page[] {
  return [...pages]
    .filter((page) => Boolean(page.stale_reason?.trim()))
    .sort((left, right) => {
      const leftPriority = left.stale_reason === "source_conflict" ? 0 : 1;
      const rightPriority = right.stale_reason === "source_conflict" ? 0 : 1;
      return leftPriority - rightPriority || comparePageRecency(left, right);
    })
    .slice(0, REVIEW_PAGE_LIMIT);
}

export function sortedKeyEntities(entities: readonly Entity[]): Entity[] {
  return [...entities].sort((left, right) => {
    if (left.confirmed !== right.confirmed) return left.confirmed ? -1 : 1;
    const confidenceDifference =
      (right.confidence ?? Number.NEGATIVE_INFINITY)
      - (left.confidence ?? Number.NEGATIVE_INFINITY);
    return confidenceDifference
      || entityTimestamp(right) - entityTimestamp(left)
      || left.name.localeCompare(right.name);
  });
}

export function latestDossierUpdate(
  spaceUpdatedAtSeconds: number,
  pages: readonly Page[],
): number | null {
  const spaceTimestamp = spaceUpdatedAtSeconds * 1_000;
  const validTimestamps = pages
    .map(pageTimestamp)
    .filter((timestamp): timestamp is number => timestamp !== null);
  if (Number.isFinite(spaceTimestamp)) validTimestamps.push(spaceTimestamp);
  return validTimestamps.length > 0 ? Math.max(...validTimestamps) : null;
}

export function formatLocalCalendarDate(
  timestamp: number,
  locale: string,
  timeZone?: string,
): string {
  const options: Intl.DateTimeFormatOptions = {
    day: "numeric",
    month: "short",
    year: "numeric",
    ...(timeZone ? { timeZone } : {}),
  };
  return new Intl.DateTimeFormat(locale, options).format(timestamp);
}

export function pageCountLabel(count: number, locale: string): string {
  const formatted = new Intl.NumberFormat(locale).format(Math.min(count, PAGE_FETCH_LIMIT));
  return count >= PAGE_FETCH_LIMIT ? `${formatted}+` : formatted;
}

export function reviewReasonName(
  page: Page,
): "needsRefresh" | "sourceConflict" | "sourceUpdated" {
  if (page.stale_reason === "source_conflict") return "sourceConflict";
  if (page.stale_reason === "source_updated") return "sourceUpdated";
  return "needsRefresh";
}
