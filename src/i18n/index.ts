import i18n, { type i18n as I18nInstance } from "i18next";
import { initReactI18next } from "react-i18next";

import {
  APP_LOCALE_STORAGE_KEY,
  type AppLocale,
  type StoredLocale,
  isStoredLocale,
  resolveInitialLocale,
} from "./locales";
import { resources, supportedAppLocales } from "./resources";

type LocaleStorage = Pick<Storage, "getItem" | "setItem">;

type I18nOptions = {
  storage?: LocaleStorage;
  systemLanguages?: readonly string[];
};

type SetLocaleOptions = I18nOptions & {
  instance?: I18nInstance;
};

function getBrowserStorage(): LocaleStorage | undefined {
  if (typeof window === "undefined") {
    return undefined;
  }

  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}

function getBrowserLanguages(): string[] {
  if (typeof navigator === "undefined") {
    return [];
  }

  if (navigator.languages.length > 0) {
    return [...navigator.languages];
  }

  return navigator.language ? [navigator.language] : [];
}

export function readStoredLocalePreference(
  storage: Pick<Storage, "getItem"> | undefined = getBrowserStorage(),
): StoredLocale {
  const storedValue = storage?.getItem(APP_LOCALE_STORAGE_KEY);
  return isStoredLocale(storedValue) ? storedValue : "system";
}

export function getInitialI18nLanguage(options: I18nOptions = {}): AppLocale {
  return resolveInitialLocale(
    readStoredLocalePreference(options.storage),
    options.systemLanguages ?? getBrowserLanguages(),
  );
}

export async function initializeI18n(
  instance: I18nInstance = i18n,
  options: I18nOptions = {},
): Promise<I18nInstance> {
  const lng = getInitialI18nLanguage(options);

  if (instance.isInitialized) {
    await instance.changeLanguage(lng);
    return instance;
  }

  instance.use(initReactI18next);
  await instance.init({
    resources,
    lng,
    supportedLngs: [...supportedAppLocales],
    fallbackLng: "en",
    load: "currentOnly",
    nonExplicitSupportedLngs: false,
    lowerCaseLng: false,
    cleanCode: false,
    initAsync: false,
    interpolation: {
      escapeValue: false,
    },
    react: {
      useSuspense: false,
    },
  });

  return instance;
}

export async function setLocalePreference(
  preference: StoredLocale,
  options: SetLocaleOptions = {},
): Promise<AppLocale> {
  const storage = options.storage ?? getBrowserStorage();
  storage?.setItem(APP_LOCALE_STORAGE_KEY, preference);

  const resolvedLanguage = resolveInitialLocale(
    preference,
    options.systemLanguages ?? getBrowserLanguages(),
  );
  await (options.instance ?? i18n).changeLanguage(resolvedLanguage);

  return resolvedLanguage;
}

export { i18n };
export type { AppLocale, StoredLocale };
