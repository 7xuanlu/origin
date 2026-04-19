// SPDX-License-Identifier: AGPL-3.0-only
import { useSyncExternalStore } from "react";
import { emit, listen } from "@tauri-apps/api/event";

export type Theme = "system" | "light" | "dark";

const STORAGE_KEY = "origin-theme";
const listeners = new Set<() => void>();
let current: Theme = (localStorage.getItem(STORAGE_KEY) as Theme) ?? "system";

function notify() {
  for (const fn of listeners) fn();
}

function resolveTheme(theme: Theme): "light" | "dark" {
  if (theme !== "system") return theme;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

/** Apply resolved theme to <html data-theme="..."> */
export function applyTheme(theme?: Theme) {
  const t = theme ?? current;
  const resolved = resolveTheme(t);
  document.documentElement.setAttribute("data-theme", resolved);
}

export function getTheme(): Theme {
  return current;
}

export function setTheme(theme: Theme) {
  current = theme;
  localStorage.setItem(STORAGE_KEY, theme);
  applyTheme(theme);
  notify();
  // Broadcast to all Tauri windows so quick capture, toast, etc. stay in sync
  emit("theme-changed", theme).catch(() => {});
}

// Listen for theme changes broadcast from other windows
listen<string>("theme-changed", (event) => {
  const incoming = event.payload as Theme;
  if (incoming && incoming !== current) {
    current = incoming;
    localStorage.setItem(STORAGE_KEY, incoming);
    applyTheme(incoming);
    notify();
  }
}).catch(() => {});

// Listen for OS color scheme changes — re-apply when in "system" mode
const mq = window.matchMedia("(prefers-color-scheme: dark)");
mq.addEventListener("change", () => {
  if (current === "system") applyTheme();
});

// React hook
function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function useTheme(): [Theme, (t: Theme) => void] {
  const theme = useSyncExternalStore(subscribe, getTheme, getTheme);
  return [theme, setTheme];
}
