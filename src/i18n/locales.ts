export const APP_LOCALE_STORAGE_KEY = "wenlan-locale";

export type AppLocale = "en" | "zh-Hans" | "zh-Hant";
export type StoredLocale = AppLocale | "system";

const STORED_LOCALES: readonly StoredLocale[] = [
  "system",
  "en",
  "zh-Hans",
  "zh-Hant",
];

export function isStoredLocale(value: unknown): value is StoredLocale {
  return (
    typeof value === "string" &&
    STORED_LOCALES.includes(value as StoredLocale)
  );
}

function normalizeLanguageTag(languageTag: string | undefined): string[] {
  return (languageTag ?? "")
    .trim()
    .replaceAll("_", "-")
    .split("-")
    .filter(Boolean)
    .map((part) => part.toLowerCase());
}

function resolveSystemLocale(languageTag: string | undefined): AppLocale | null {
  const parts = normalizeLanguageTag(languageTag);
  const [language, scriptOrRegion] = parts;

  if (language === "en") {
    return "en";
  }

  if (language !== "zh") {
    return null;
  }

  if (scriptOrRegion === "hant") {
    return "zh-Hant";
  }
  if (scriptOrRegion === "hans") {
    return "zh-Hans";
  }
  if (["tw", "hk", "mo"].includes(scriptOrRegion ?? "")) {
    return "zh-Hant";
  }

  return "zh-Hans";
}

export function resolveAppLocale(languageTag: string | undefined): AppLocale {
  return resolveSystemLocale(languageTag) ?? "en";
}

export function resolveInitialLocale(
  storedPreference: unknown,
  systemLanguages: readonly string[] = [],
): AppLocale {
  if (isStoredLocale(storedPreference) && storedPreference !== "system") {
    return storedPreference;
  }

  for (const language of systemLanguages) {
    const locale = resolveSystemLocale(language);
    if (locale) {
      return locale;
    }
  }

  return "en";
}
