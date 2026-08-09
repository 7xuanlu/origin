// SPDX-License-Identifier: AGPL-3.0-only
// No-op browser stands-ins for the remaining Tauri modules the app imports.
// One file, one export per aliased module entry (see vite.preview.config.ts).

// @tauri-apps/api/event
type EventPayload<T = unknown> = {
  event: string;
  payload: T;
};
type EventHandler<T = unknown> = (event: EventPayload<T>) => void;

const listeners = new Map<string, Set<EventHandler>>();

export async function listen<T = unknown>(
  event: string,
  handler: EventHandler<T>,
): Promise<() => void> {
  const handlers = listeners.get(event) ?? new Set<EventHandler>();
  handlers.add(handler as EventHandler);
  listeners.set(event, handlers);
  return () => {
    handlers.delete(handler as EventHandler);
    if (handlers.size === 0) listeners.delete(event);
  };
}
export async function once<T = unknown>(
  event: string,
  handler: EventHandler<T>,
): Promise<() => void> {
  const unlisten = await listen<T>(event, (payload) => {
    unlisten();
    handler(payload);
  });
  return unlisten;
}
export async function emit(event: string, payload?: unknown): Promise<void> {
  for (const handler of listeners.get(event) ?? []) {
    queueMicrotask(() => handler({ event, payload }));
  }
}
export const emitTo = emit;

declare global {
  interface Window {
    __TAURI_PREVIEW_EMIT__?: typeof emit;
  }
}

if (typeof window !== "undefined") {
  window.__TAURI_PREVIEW_EMIT__ = emit;
}

// @tauri-apps/api/app
export async function getVersion(): Promise<string> {
  return "browser-preview";
}
export async function getName(): Promise<string> {
  return "wenlan (preview)";
}

// @tauri-apps/api/window
export class LogicalSize {
  constructor(
    public width: number,
    public height: number,
  ) {}
}
export class LogicalPosition {
  constructor(
    public x: number,
    public y: number,
  ) {}
}
const windowStub = {
  setSize: async () => {},
  setPosition: async () => {},
  center: async () => {},
  innerSize: async () => new LogicalSize(window.innerWidth, window.innerHeight),
  outerPosition: async () => new LogicalPosition(0, 0),
  scaleFactor: async () => window.devicePixelRatio,
  setFocus: async () => {},
  setAlwaysOnTop: async () => {},
  show: async () => {},
  hide: async () => {},
  close: async () => {},
  listen: async () => () => {},
  once: async () => () => {},
  emit: async () => {},
  label: "main",
};
export function getCurrentWindow() {
  return windowStub;
}
export async function currentMonitor() {
  return {
    size: { width: window.screen.width, height: window.screen.height },
    position: { x: 0, y: 0 },
    scaleFactor: window.devicePixelRatio,
  };
}
export async function availableMonitors() {
  return [await currentMonitor()];
}

// @tauri-apps/api/webviewWindow
export class WebviewWindow {
  constructor(
    public label: string,
    _opts?: unknown,
  ) {}
  static async getByLabel(_label: string) {
    return null;
  }
  async once() {
    return () => {};
  }
  async listen() {
    return () => {};
  }
  async show() {}
  async setFocus() {}
  async close() {}
}
export function getCurrentWebviewWindow() {
  return windowStub;
}

// @tauri-apps/plugin-dialog
export async function open(_opts?: unknown): Promise<null> {
  console.warn("[preview] native file dialog unavailable in browser");
  return null;
}
export const save = open;

// @tauri-apps/plugin-fs
export async function readDir(_path: string): Promise<unknown[]> {
  return [];
}
export async function readTextFile(_path: string): Promise<string> {
  return "";
}
export async function exists(_path: string): Promise<boolean> {
  return false;
}

// tauri-plugin-clipboard-x-api
export async function startListening(): Promise<void> {}
export async function stopListening(): Promise<void> {}
export function onClipboardChange(_cb: unknown): () => void {
  return () => {};
}
export async function writeText(text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
}
export async function readText(): Promise<string> {
  return "";
}
