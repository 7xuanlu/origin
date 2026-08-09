// SPDX-License-Identifier: AGPL-3.0-only
import type { Page } from "@playwright/test";
import { APP_LOCALE_STORAGE_KEY, type AppLocale } from "../src/i18n/locales";
import type { MemoryItem } from "../src/lib/tauri";
import { createSpacesNavigationFixture } from "./fixtures/spacesNavigation";
import { TauriMockRuntime } from "./tauriMock/runtime";
import type { BrowserErrorCapture, InstallTauriMockOptions, MockCommandCall } from "./tauriMock/types";

export type TauriMockController = {
  readonly calls: () => readonly MockCommandCall[];
  readonly failNext: (command: string, message: string, times?: number) => void;
};

export function collectBrowserErrors(page: Page): BrowserErrorCapture {
  const capture: BrowserErrorCapture = { pageErrors: [], consoleErrors: [] };
  page.on("pageerror", (error) => capture.pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") capture.consoleErrors.push(message.text());
  });
  return capture;
}

export async function installTauriMock(
  page: Page,
  options: InstallTauriMockOptions,
): Promise<TauriMockController> {
  const defaults = createSpacesNavigationFixture();
  const fixture = { ...defaults, ...(options.fixture ?? {}), memories: options.memories ?? options.fixture?.memories ?? defaults.memories };
  const runtime = new TauriMockRuntime(fixture, options.failures, options.rawActions);
  await page.exposeBinding("__wenlanTauriInvoke", (_source, command: string, args?: unknown) => runtime.invoke(command, args));
  await page.addInitScript(({ locale, localeStorageKey, storageEntries }) => {
    window.localStorage.clear();
    window.localStorage.setItem(localeStorageKey, locale);
    window.localStorage.setItem("wenlan-sidebar-collapsed", "false");
    for (const [key, value] of storageEntries) window.localStorage.setItem(key, value);

    const callbacks = new Map<number, (...args: unknown[]) => unknown>();
    const eventListeners = new Map<string, Set<number>>();
    let nextCallbackId = 1;
    const emitEvent = (event: string, payload: unknown) => {
      for (const id of eventListeners.get(event) ?? []) callbacks.get(id)?.({ event, id, payload });
    };
    window.__TAURI_INTERNALS__ = {
      metadata: {
        currentWindow: { label: "main" },
        currentWebview: { label: "main" },
      },
      invoke: async (command: string, args?: unknown) => {
        if (command === "plugin:event|listen") {
          const event = typeof args === "object" && args !== null ? Reflect.get(args, "event") : undefined;
          const handler = typeof args === "object" && args !== null ? Reflect.get(args, "handler") : undefined;
          if (typeof event === "string" && typeof handler === "number") {
            const ids = eventListeners.get(event) ?? new Set<number>();
            ids.add(handler);
            eventListeners.set(event, ids);
            return handler;
          }
        }
        if (command === "plugin:event|emit") {
          const event = typeof args === "object" && args !== null ? Reflect.get(args, "event") : undefined;
          const payload = typeof args === "object" && args !== null ? Reflect.get(args, "payload") : undefined;
          if (typeof event === "string") emitEvent(event, payload);
          return null;
        }
        return window.__wenlanTauriInvoke(command, args);
      },
      transformCallback: (callback: (...args: unknown[]) => unknown) => {
        const id = nextCallbackId++;
        callbacks.set(id, callback);
        return id;
      },
      unregisterCallback: (id: number) => callbacks.delete(id),
    };
    window.__TAURI_EVENT_PLUGIN_INTERNALS__ = {
      unregisterListener: (event, eventId) => eventListeners.get(event)?.delete(eventId),
    };
  }, {
    locale: options.locale,
    localeStorageKey: APP_LOCALE_STORAGE_KEY,
    storageEntries: Object.entries(options.localStorage ?? {}),
  });
  return {
    calls: () => runtime.calls(),
    failNext: (command, message, times) => runtime.failNext(command, message, times),
  };

}

declare global {
  interface Window {
    __TAURI_INTERNALS__?: {
      metadata: {
        currentWindow: { label: string };
        currentWebview: { label: string };
      };
      invoke: (command: string, args?: unknown, options?: unknown) => Promise<unknown>;
      transformCallback: (callback: (...args: unknown[]) => unknown, once?: boolean) => number;
      unregisterCallback: (id: number) => void;
    };
    __TAURI_EVENT_PLUGIN_INTERNALS__?: {
      unregisterListener: (event: string, eventId: number) => void;
    };
    __wenlanTauriInvoke: (command: string, args?: unknown) => Promise<unknown>;
  }
}
