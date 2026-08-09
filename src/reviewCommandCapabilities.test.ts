// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import { invoke, resetReviewRuntime } from "../review/tauri-core";
import { REVIEW_COMMAND_CAPABILITIES } from "../review/commandCapabilities";

describe("Review command capability contract", () => {
  it.each(
    Object.entries(REVIEW_COMMAND_CAPABILITIES).flatMap(([area, commands]) =>
      commands.map((command) => ({ area, command })),
    ),
  )("dispatches $area command $command instead of silently falling through", async ({ command }) => {
    resetReviewRuntime();

    try {
      await invoke(command);
    } catch (error) {
      expect(String(error)).not.toContain(`Unknown Tauri command: ${command}`);
    }
  });
});
