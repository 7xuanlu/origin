// SPDX-License-Identifier: AGPL-3.0-only
/**
 * The first-paint background in index.html.
 *
 * index.css keeps `html` and `body` transparent so the toast and quick-capture
 * webviews can float over the desktop. The main window inherits that, and
 * app/tauri.conf.json shows it from config on purpose, so between the window
 * appearing and React's first render it paints nothing. macOS hides this by
 * giving the window a native NSColor background (app/src/lib.rs); Windows and
 * Linux have no such call, so the window is blank for as long as the bundle,
 * the fonts and the daemon health check take -- tens of seconds on an upgrade
 * that runs database migrations.
 *
 * index.html closes that with two CSS rules and one classic inline script.
 * Those live outside the module graph, so nothing else in the suite covers
 * them: this file executes the real script from the real file, and then the
 * real script from the built file.
 *
 * What this file does NOT do: parse HTML in a real engine, apply a CSP, or
 * model browser scheduling. It is a behavioural test of one script.
 */
import { existsSync, readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SOURCE_HTML_PATH = path.join(ROOT, "index.html");
const DIST_HTML_PATH = path.join(ROOT, "dist/index.html");
const INDEX_HTML = readFileSync(SOURCE_HTML_PATH, "utf-8");
const INDEX_CSS = readFileSync(path.join(ROOT, "src/index.css"), "utf-8");

/** The one classic <script> in index.html: the module entry carries a src. */
function inlineScript(html: string): string {
  const matches = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
  expect(
    matches,
    "index.html should carry exactly one inline classic script",
  ).toHaveLength(1);
  return matches[0][1];
}

/**
 * The inline <style>'s rules, with comments removed and spacing flattened.
 *
 * Removing the comments is the point. That block explains itself by quoting
 * the very selectors it defines, so a bare `toContain` on the html is
 * satisfied by the prose alone and would still pass if the rules were deleted.
 */
function styleRules(html: string): string {
  const match = html.match(/<style>([\s\S]*?)<\/style>/);
  expect(match, "index.html should carry exactly one inline <style>").not.toBeNull();
  return match![1]
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\s+/g, " ")
    .replace(/\s*([{}:;])\s*/g, "$1")
    .trim()
    .toLowerCase();
}

/**
 * Run a first-paint script against a throwaway document, and report what it set.
 *
 * `storage` undefined means localStorage throws, which is what a webview with
 * site data disabled does. `label` undefined means Tauri injected no
 * internals, which is `pnpm dev` in a plain browser.
 */
function runFirstPaint(
  opts: {
    hash: string;
    label?: string;
    storage?: Record<string, string>;
    prefersDark?: boolean;
    matchMedia?: boolean;
  },
  script: string = inlineScript(INDEX_HTML),
): { window: string | null; theme: string | null } {
  const el: Record<string, string> = {};
  const documentElement = {
    setAttribute: (k: string, v: string) => {
      el[k] = v;
    },
  };
  const storage = opts.storage;
  const fakeWindow: Record<string, unknown> = {
    location: { hash: opts.hash },
    matchMedia:
      opts.matchMedia === false
        ? undefined
        : (query: string) => ({
            matches: query.includes("dark")
              ? !!opts.prefersDark
              : !opts.prefersDark,
          }),
  };
  if (opts.label !== undefined) {
    fakeWindow.__TAURI_INTERNALS__ = {
      metadata: { currentWindow: { label: opts.label } },
    };
  }
  const localStorage = {
    getItem: (k: string) => {
      if (!storage) throw new DOMException("denied", "SecurityError");
      return k in storage ? storage[k] : null;
    },
  };

  new Function(
    "window",
    "document",
    "localStorage",
    "DOMException",
    script,
  )(fakeWindow, { documentElement }, localStorage, DOMException);

  return { window: el["data-window"] ?? null, theme: el["data-theme"] ?? null };
}

describe("which window gets a background", () => {
  // The label is the window's real identity. Tauri prepends the currentWindow
  // metadata to the webview's initialization scripts and WebView2 runs those
  // before HTML parsing, so it is already there when this script runs.
  // Trusting it means a fragment added after parsing, or an overlay opened at
  // `index.html#` (whose location.hash is the empty string), can no longer be
  // mistaken for main.
  it("trusts the Tauri window label over the fragment", () => {
    expect(runFirstPaint({ hash: "", label: "main" }).window).toBe("main");
    expect(runFirstPaint({ hash: "#toast", label: "main" }).window).toBe("main");
  });

  it.each(["toast", "quick-capture"])(
    "leaves the %s window transparent even with no fragment",
    (label) => {
      const r = runFirstPaint({ hash: "", label });
      expect(r.window).toBeNull();
      expect(r.theme).toBeNull();
    },
  );

  // Without Tauri there are no overlay windows to confuse it with, so the
  // fragment is enough. This is `pnpm dev` and the preview harness.
  it("falls back to the fragment when Tauri injected no internals", () => {
    expect(runFirstPaint({ hash: "" }).window).toBe("main");
    expect(runFirstPaint({ hash: "#toast" }).window).toBeNull();
    expect(runFirstPaint({ hash: "#quick-capture" }).window).toBeNull();
  });
});

describe("the early theme resolves as theme.ts would", () => {
  // readPreference() returns the current key, falling back to the legacy key
  // only when the current one is absent; resolveTheme() sends "system" to the
  // media query and passes every other value through untouched. A value this
  // file does not recognise must therefore survive, or applyTheme() would
  // replace what was painted and the window would flash.
  const cases: Array<[string, Record<string, string>, boolean, string]> = [
    ["nothing stored, dark OS", {}, true, "dark"],
    ["nothing stored, light OS", {}, false, "light"],
    ["system, dark OS", { "wenlan-theme": "system" }, true, "dark"],
    ["system, light OS", { "wenlan-theme": "system" }, false, "light"],
    ["light beats a dark OS", { "wenlan-theme": "light" }, true, "light"],
    ["dark beats a light OS", { "wenlan-theme": "dark" }, false, "dark"],
    ["legacy key when current is absent", { "origin-theme": "light" }, true, "light"],
    [
      "current key wins over legacy",
      { "wenlan-theme": "dark", "origin-theme": "light" },
      false,
      "dark",
    ],
    ["an unrecognised value passes through", { "wenlan-theme": "bogus" }, false, "bogus"],
    ["an empty value passes through", { "wenlan-theme": "" }, true, ""],
  ];

  it.each(cases)("%s", (_name, storage, prefersDark, expected) => {
    expect(runFirstPaint({ hash: "", label: "main", storage, prefersDark }).theme).toBe(
      expected,
    );
  });

  // Two deliberate departures from theme.ts. Both exist because this script
  // runs before anything can catch it: an exception here leaves the window
  // transparent, which is the defect it was written to fix.
  describe("and departs from it deliberately, in two places", () => {
    // readPreference() copies a legacy value into the current key as it reads.
    // A first-paint script has no business writing to storage, and skipping
    // the write cannot change what is painted: the value returned is the same
    // either way, and applyTheme() performs the migration moments later.
    it("reads the legacy key without migrating it", () => {
      const storage = { "origin-theme": "light" };
      expect(runFirstPaint({ hash: "", label: "main", storage }).theme).toBe("light");
      expect(storage).toEqual({ "origin-theme": "light" });
    });

    // resolveTheme() calls window.matchMedia unconditionally and would throw
    // where it does not exist. Here that would abort the script before it set
    // anything, so the call is guarded and the light default stands in.
    it("treats a missing matchMedia as light instead of throwing", () => {
      const r = runFirstPaint({
        hash: "",
        label: "main",
        storage: {},
        prefersDark: true,
        matchMedia: false,
      });
      expect(r.window).toBe("main");
      expect(r.theme).toBe("light");
    });
  });

  it("still resolves a theme when localStorage throws", () => {
    const r = runFirstPaint({ hash: "", label: "main", prefersDark: false });
    expect(r.window).toBe("main");
    expect(r.theme).toBe("light");
  });
});

describe("first-paint colours track index.css", () => {
  let darkBg: string;
  let lightBg: string;

  beforeAll(() => {
    const dark = INDEX_CSS.match(/:root\s*\{[^}]*?--bg-primary:\s*(#[0-9a-fA-F]{6})/);
    const light = INDEX_CSS.match(
      /html\[data-theme="light"\]\s*\{[^}]*?--bg-primary:\s*(#[0-9a-fA-F]{6})/,
    );
    expect(dark, "index.css should declare a dark --bg-primary").not.toBeNull();
    expect(light, "index.css should declare a light --bg-primary").not.toBeNull();
    darkBg = dark![1].toLowerCase();
    lightBg = light![1].toLowerCase();
  });

  // index.html cannot use a CSS variable here: the variables live in
  // index.css, which the module graph loads well after this paints. The hex is
  // therefore duplicated on purpose, and this test is what keeps the duplicate
  // honest when the palette moves.
  it("uses the same dark background as --bg-primary", () => {
    expect(styleRules(INDEX_HTML)).toContain(
      `html[data-window="main"]{background:${darkBg};`,
    );
  });

  it("uses the same light background as --bg-primary", () => {
    expect(styleRules(INDEX_HTML)).toContain(
      `html[data-window="main"][data-theme="light"]{background:${lightBg};`,
    );
  });

  it("still keeps html and body transparent for the overlays", () => {
    expect(INDEX_CSS).toMatch(/html,\s*body\s*\{\s*background:\s*transparent/);
  });
});

/**
 * Everything above reads index.html. All of it passes if Vite drops the inline
 * script on its way to dist/, because none of it looks at what ships.
 *
 * dist/ is gitignored and `pnpm test` runs before any build, so on a fresh
 * checkout there is nothing here to read, and skipping quietly in that state
 * is how this block came to run nowhere at all. WENLAN_REQUIRE_DIST turns the
 * skip into a failure; the CI step that sets it builds immediately first, and
 * that build -- not the mtime below -- is what makes the artifact current
 * there.
 *
 * The mtime comparison is a convenience, not a proof: it catches the common
 * local mistake of editing index.html and forgetting to rebuild. A dist/
 * restored from a cache or another revision can carry a newer timestamp than
 * the source and would pass it. Do not read it as a guarantee that what is
 * checked is what this revision builds.
 */
function distStatus(): { ok: boolean; reason: string } {
  if (!existsSync(DIST_HTML_PATH)) {
    return { ok: false, reason: "dist/index.html does not exist; run pnpm build" };
  }
  if (statSync(DIST_HTML_PATH).mtimeMs < statSync(SOURCE_HTML_PATH).mtimeMs) {
    return { ok: false, reason: "dist/index.html is older than index.html; run pnpm build" };
  }
  return { ok: true, reason: "" };
}

const dist = distStatus();
const distIsMandatory = !!process.env.WENLAN_REQUIRE_DIST;

describe.skipIf(!dist.ok && !distIsMandatory)("survives the bundler", () => {
  let built: string;

  beforeAll(() => {
    // Only a WENLAN_REQUIRE_DIST run reaches this unsatisfied; without it the
    // block is skipped instead.
    if (!dist.ok) throw new Error(`cannot check the built output: ${dist.reason}`);
    built = readFileSync(DIST_HTML_PATH, "utf-8");
  });

  // The point of the whole block: not that the text survived, but that the
  // code did. Vite copies index.html's inline blocks through verbatim, so a
  // substring check is answered by the explanatory comments whether or not the
  // script still works. This runs the shipped script instead.
  it("still paints the main window and still leaves the overlays alone", () => {
    const script = inlineScript(built);
    expect(
      runFirstPaint({ hash: "", label: "main", storage: {}, prefersDark: true }, script),
    ).toEqual({ window: "main", theme: "dark" });
    expect(
      runFirstPaint({ hash: "", label: "main", storage: { "wenlan-theme": "light" } }, script),
    ).toEqual({ window: "main", theme: "light" });
    expect(runFirstPaint({ hash: "", label: "toast" }, script)).toEqual({
      window: null,
      theme: null,
    });
  });

  it("still carries both first-paint rules, and not just the comment", () => {
    const rules = styleRules(built);
    expect(rules).toContain('html[data-window="main"]{background:');
    expect(rules).toContain('html[data-window="main"][data-theme="light"]{background:');
    expect(rules).toBe(styleRules(INDEX_HTML));
  });

  // Vite hoists the module entry into <head>. Module scripts are deferred, so
  // ordering would hold either way, but asserting it keeps the guarantee from
  // resting on that one fact alone.
  it("keeps the classic script ahead of the module entry, and undeferred", () => {
    const doc = new DOMParser().parseFromString(built, "text/html");
    const scripts = [...doc.querySelectorAll("script")];
    const classicAt = scripts.findIndex((s) => !s.hasAttribute("src"));
    const moduleAt = scripts.findIndex((s) => s.getAttribute("type") === "module");
    expect(classicAt, "no inline classic script in dist/index.html").toBeGreaterThanOrEqual(0);
    expect(moduleAt, "no module entry in dist/index.html").toBeGreaterThanOrEqual(0);
    expect(classicAt).toBeLessThan(moduleAt);

    const classic = scripts[classicAt];
    expect(classic.hasAttribute("defer")).toBe(false);
    expect(classic.hasAttribute("async")).toBe(false);
    expect(classic.getAttribute("type")).toBeNull();
  });
});
