// SPDX-License-Identifier: AGPL-3.0-only
/**
 * The main window is born at its final geometry and never moved afterwards.
 *
 * The released 0.17.8 window appeared at 1294x798 outer, correctly centred by
 * Tauri, and then two to four seconds later shrank to 1294x758 and jumped from
 * (121,13) to (128,72). The cause was a mount `useEffect` in `src/App.tsx`
 * that called `resizeWindowCentered(1280, 720)`: the config said 760 inner
 * (hence the 40px shrink) and the helper centred the CLIENT rect against the
 * full MONITOR rect rather than the outer rect against the work area (hence
 * the offset). No window-state plugin exists, so it repeated every launch.
 *
 * Two halves, and both have to hold:
 *   - `app/tauri.conf.json` declares the final size and asks Tauri to centre.
 *   - Nothing in the frontend resizes or repositions the main window.
 *
 * This reads the real files rather than rendering, because the defect is a
 * launch-time side effect against a real native window — jsdom has no such
 * thing, so a render test would mock exactly the API whose absence is the
 * point.
 */
import { readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { readSourceText, repoRelativePath } from "../test/sourceText";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const TAURI_CONF_PATH = path.join(ROOT, "app/tauri.conf.json");
const APP_TSX_PATH = path.join(ROOT, "src/App.tsx");
const SRC_DIR = path.join(ROOT, "src");

/** Frontend calls that move or resize a native window. */
const GEOMETRY_CALLS = [
  "resizeWindowCentered",
  "setSize(",
  "setPosition(",
  "currentMonitor(",
] as const;

type TauriWindow = {
  label?: string;
  width?: number;
  height?: number;
  center?: boolean;
};

function mainWindow(): TauriWindow {
  const conf = JSON.parse(readSourceText(TAURI_CONF_PATH)) as {
    app?: { windows?: TauriWindow[] };
  };
  const windows = conf.app?.windows ?? [];
  const main = windows.find((window) => window.label === "main");
  expect(main, "app/tauri.conf.json should declare a window labelled 'main'").toBeDefined();
  return main!;
}

/** Every non-test TypeScript source under `src/`. */
function listSourceFiles(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "test" || entry.name === "__tests__") return [];
      return listSourceFiles(fullPath);
    }
    if (!/\.tsx?$/.test(entry.name)) return [];
    if (/\.test\.tsx?$/.test(entry.name)) return [];
    return [fullPath];
  });
}

describe("main window geometry", () => {
  it("declares the final size in tauri.conf.json and lets Tauri centre it", () => {
    const main = mainWindow();

    expect(main.width).toBe(1280);
    expect(main.height).toBe(720);
    expect(main.center).toBe(true);
  });

  it("does not resize or reposition the window from App's mount", () => {
    const source = readSourceText(APP_TSX_PATH);
    const found = GEOMETRY_CALLS.filter((call) => source.includes(call));

    expect(
      found,
      "src/App.tsx must leave the launch geometry to app/tauri.conf.json",
    ).toEqual([]);
  });

  it("keeps the resizeWindowCentered helper deleted across the frontend", () => {
    const offenders = listSourceFiles(SRC_DIR)
      .filter((file) => readSourceText(file).includes("resizeWindowCentered"))
      .map((file) => repoRelativePath(file, ROOT));

    expect(offenders).toEqual([]);
  });
});
