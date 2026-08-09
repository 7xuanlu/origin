// SPDX-License-Identifier: AGPL-3.0-only
const ENTITY_CHARACTER = /[\p{L}\p{N}]/u;

function timestampDate(timestamp: number): Date | null {
  if (!Number.isFinite(timestamp)) return null;
  const date = new Date(timestamp * 1000);
  return Number.isFinite(date.getTime()) ? date : null;
}

export function entityMonogram(name: string): string {
  return name
    .trim()
    .split(/\s+/u)
    .map((word) => Array.from(word).find((character) => ENTITY_CHARACTER.test(character)))
    .filter((character): character is string => character !== undefined)
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export function formatAbsoluteTimestamp(timestamp: number, locale?: string): string | null {
  return timestampDate(timestamp)?.toLocaleString(locale) ?? null;
}

export function formatCalendarDate(timestamp: number, locale: string): string | null {
  return timestampDate(timestamp)?.toLocaleDateString(locale) ?? null;
}

export function formatRelativeEntityTime(
  timestamp: number,
  locale: string,
  nowSeconds = Date.now() / 1000,
): string | null {
  if (!Number.isFinite(timestamp) || !Number.isFinite(nowSeconds)) return null;
  const difference = Math.floor(nowSeconds - timestamp);
  const formatter = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  if (difference < 60) return formatter.format(-difference, "second");
  if (difference < 3600) return formatter.format(-Math.floor(difference / 60), "minute");
  if (difference < 86400) return formatter.format(-Math.floor(difference / 3600), "hour");
  if (difference < 604800) return formatter.format(-Math.floor(difference / 86400), "day");
  return formatCalendarDate(timestamp, locale);
}

export function formatConfidence(value: number | null, digits: number): string | null {
  if (value == null || !Number.isFinite(value) || value < 0 || value > 1) return null;
  return value.toFixed(digits);
}
