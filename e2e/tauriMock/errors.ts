// SPDX-License-Identifier: AGPL-3.0-only
export class UnknownTauriCommandError extends Error {
  readonly name = "UnknownTauriCommandError";

  constructor(readonly command: string) {
    super(`Unknown Tauri command: ${command}`);
  }
}

export class TauriMockArgumentError extends Error {
  readonly name = "TauriMockArgumentError";

  constructor(readonly command: string, readonly key: string) {
    super(`Malformed ${command} arguments: ${key}`);
  }
}

export class ConfiguredTauriFailureError extends Error {
  readonly name = "ConfiguredTauriFailureError";

  constructor(readonly command: string, message: string) {
    super(`${command}: ${message}`);
  }
}
