export type LocaleDateDisplay = {
  readonly label: string;
  readonly dateTime?: string;
};

/**
 * Formats an already-normalized date for visible inventory copy and semantic
 * `<time>` markup. Callers remain responsible for the source timestamp's unit.
 */
export function formatLocaleDate(
  date: Date,
  locales?: Intl.LocalesArgument,
): LocaleDateDisplay {
  if (!Number.isFinite(date.getTime())) return { label: "—" };
  return {
    label: date.toLocaleDateString(locales, {
      day: "numeric",
      month: "short",
      year: "numeric",
    }),
    dateTime: date.toISOString(),
  };
}
