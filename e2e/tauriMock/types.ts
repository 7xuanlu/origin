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

export type TauriMockPageScenario = {
  readonly daemonVersion?: string;
  readonly saveDaemonVersion?: string;
  readonly firstWriteRemoteMutation?: {
    readonly pageId: string;
    readonly content: string;
  };
};

export type InstallTauriMockOptions = {
  readonly locale: AppLocale;
  readonly rawActions: readonly string[];
  readonly memories?: readonly MemoryItem[];
  readonly fixture?: SpacesNavigationFixture;
  readonly failures?: readonly MockFailure[];
  /**
   * Milliseconds to hold a command before answering, keyed by command name.
   * A real library answers `get_page_map` in one to two seconds, and several
   * canvas defects only exist inside that window.
   */
  readonly delays?: Readonly<Record<string, number>>;
  readonly localStorage?: Readonly<Record<string, string>>;
  readonly pageScenario?: TauriMockPageScenario;
};

export type BrowserErrorCapture = {
  readonly pageErrors: string[];
  readonly consoleErrors: string[];
};
