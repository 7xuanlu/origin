// SPDX-License-Identifier: AGPL-3.0-only
import { createSpacesNavigationFixture } from "../e2e/fixtures/spacesNavigation";
import { UnknownTauriCommandError } from "../e2e/tauriMock/errors";
import { TauriMockRuntime } from "../e2e/tauriMock/runtime";
import { isReviewCommand } from "./commandCapabilities";

export function createReviewRuntime(): TauriMockRuntime {
  return new TauriMockRuntime(createSpacesNavigationFixture());
}

let runtime = createReviewRuntime();

const REVIEW_COMMAND_FAILURES_KEY = "__WENLAN_REVIEW_COMMAND_FAILURES__";

function recordReviewCommandFailure(command: string, error: unknown): void {
  const current = Reflect.get(globalThis, REVIEW_COMMAND_FAILURES_KEY);
  const failures = Array.isArray(current) ? current : [];
  failures.push(`${command}: ${String(error)}`);
  Reflect.set(globalThis, REVIEW_COMMAND_FAILURES_KEY, failures);
}

export function resetReviewRuntime(): void {
  runtime = createReviewRuntime();
  Reflect.set(globalThis, REVIEW_COMMAND_FAILURES_KEY, []);
}

export async function invoke<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  try {
    if (!isReviewCommand(command)) {
      throw new UnknownTauriCommandError(command);
    }
    return await runtime.invoke(command, args) as T;
  } catch (error) {
    recordReviewCommandFailure(command, error);
    throw error;
  }
}

export function convertFileSrc(path: string): string {
  return `review-fixture://asset/${encodeURIComponent(path)}`;
}

export const isTauri = (): boolean => true;

export class Resource {
  close(): void {}
}

export class Channel<T = unknown> {
  onmessage: ((message: T) => void) | null = null;
}

export class PluginListener {
  unregister(): void {}
}

export async function addPluginListener(): Promise<PluginListener> {
  return new PluginListener();
}

export function transformCallback(): number {
  return 0;
}
