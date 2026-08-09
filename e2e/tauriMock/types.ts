// SPDX-License-Identifier: AGPL-3.0-only
import type { AppLocale } from "../../src/i18n/locales";
import type { MemoryItem } from "../../src/lib/tauri";
import type { SpacesNavigationFixture } from "../fixtures/spacesNavigation";

export type MockCommandCall = {
  readonly command: string;
  readonly args: unknown;
};

export type MockFailure = {
  readonly command: string;
  readonly message: string;
  readonly times?: number;
};

export type InstallTauriMockOptions = {
  readonly locale: AppLocale;
  readonly rawActions: readonly string[];
  readonly memories?: readonly MemoryItem[];
  readonly fixture?: SpacesNavigationFixture;
  readonly failures?: readonly MockFailure[];
  readonly localStorage?: Readonly<Record<string, string>>;
};

export type BrowserErrorCapture = {
  readonly pageErrors: string[];
  readonly consoleErrors: string[];
};
