// SPDX-License-Identifier: AGPL-3.0-only
type EventPayload<T = unknown> = { event: string; payload: T };
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

export async function getVersion(): Promise<string> {
  return "review-fixtures";
}

export async function getName(): Promise<string> {
  return "Wenlan Review";
}

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
  label: "main",
  setSize: async () => {},
  setPosition: async () => {},
  center: async () => {},
  innerSize: async () => new LogicalSize(window.innerWidth, window.innerHeight),
  outerPosition: async () => new LogicalPosition(0, 0),
  scaleFactor: async () => window.devicePixelRatio,
  setFocus: async () => {},
  setAlwaysOnTop: async () => {},
  setIgnoreCursorEvents: async () => {},
  startDragging: async () => {},
  show: async () => {},
  hide: async () => {},
  close: async () => {},
  isVisible: async () => true,
  listen,
  once,
  emit,
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

export class WebviewWindow {
  constructor(
    public label: string,
    _options?: unknown,
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

  async show(): Promise<void> {}
  async setFocus(): Promise<void> {}
  async close(): Promise<void> {}
}

export function getCurrentWebviewWindow() {
  return windowStub;
}

export async function open(): Promise<null> {
  return null;
}

export const save = open;

export async function readDir(): Promise<unknown[]> {
  return [];
}

export async function readTextFile(): Promise<string> {
  return "";
}

export async function exists(): Promise<boolean> {
  return false;
}

export async function isPermissionGranted(): Promise<boolean> {
  return false;
}

export async function requestPermission(): Promise<"denied"> {
  return "denied";
}

export async function sendNotification(): Promise<void> {}

export async function startListening(): Promise<void> {}
export async function stopListening(): Promise<void> {}
export function onClipboardChange(): () => void {
  return () => {};
}
export async function writeText(): Promise<void> {}
export async function readText(): Promise<string> {
  return "";
}
