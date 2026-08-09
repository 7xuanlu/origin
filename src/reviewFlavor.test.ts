// SPDX-License-Identifier: AGPL-3.0-only
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function read(path: string): string {
  return readFileSync(resolve(root, path), "utf8");
}

describe("fixture-only native Review flavor", () => {
  it("has a distinct native identity and removes every production integration", () => {
    const review = JSON.parse(read("app/tauri.review.conf.json"));

    expect(review.productName).toBe("Wenlan Review");
    expect(review.identifier).toBe("com.wenlan.desktop.review");
    expect(review.mainBinaryName).toBe("wenlan-review");
    expect(review.build.frontendDist).toBe("../dist/review");
    expect(review.build.devUrl).toBe("http://localhost:1422");
    expect(review.app.windows).toEqual([
      expect.objectContaining({ label: "main", title: "Wenlan Review" }),
    ]);
    expect(review.app.trayIcon).toBeNull();
    expect(review.app.security.assetProtocol).toEqual({
      enable: true,
      scope: [],
    });
    expect(review.app.security.csp).toBe(
      "default-src 'self'; connect-src 'none'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self' data:; frame-src 'none'; object-src 'none'",
    );
    expect(review.app.security.capabilities).toEqual([
      {
        identifier: "review-shell",
        description: "Fixture-only Wenlan Review window",
        windows: ["main"],
        permissions: ["core:default"],
      },
    ]);
    expect(review.bundle.externalBin).toEqual([]);
    expect(review.bundle.createUpdaterArtifacts).toBe(false);
    expect(review.plugins.updater).toBeNull();
  });

  it("selects a separate minimal Rust entry point at compile time", () => {
    const cargo = read("app/Cargo.toml");
    const lib = read("app/src/lib.rs");
    const main = read("app/src/main.rs");
    const review = read("app/src/review.rs");

    expect(cargo).toContain("review-fixtures = []");
    expect(cargo).toContain(
      'features = ["protocol-asset", "tray-icon", "macos-private-api", "image-png"]',
    );
    expect(main).toContain('#[cfg(feature = "review-fixtures")]');
    expect(main).toContain("wenlan_lib::run_review()");
    expect(main).toContain('#[cfg(not(feature = "review-fixtures"))]');
    expect(main).toContain("wenlan_lib::run()");
    expect(lib).toContain(
      '#[cfg(not(feature = "review-fixtures"))]\n#[cfg_attr(mobile, tauri::mobile_entry_point)]\npub fn run()',
    );
    expect(review).toContain("tauri::Builder::default()");
    expect(review).toContain("tauri::generate_context!()");
    expect(review).not.toMatch(/\.plugin\(|\.manage\(|\.setup\(|\.invoke_handler\(/);
    expect(review).not.toMatch(/WenlanClient|launchd|sidecar|updater|tray|watcher|tunnel|mcp/i);
  });

  it("builds through fixture-only Vite aliases with no daemon proxy", () => {
    const vite = read("vite.review.config.ts");

    expect(vite).toContain('__WENLAN_REVIEW__: JSON.stringify(true)');
    expect(vite).toContain('"@tauri-apps/api/core"');
    expect(vite).toContain("./review/tauri-core.ts");
    expect(vite).toContain("port: 1422");
    expect(vite).toContain('outDir: "dist/review"');
    expect(vite).not.toContain("127.0.0.1:7878");
    expect(vite).not.toContain("proxy:");
    expect(vite).not.toContain("live-invoke");
  });

  it("exposes dedicated build and open commands without preparing sidecars", () => {
    const pkg = JSON.parse(read("package.json"));
    const scripts = pkg.scripts as Record<string, string>;
    const launcher = read("scripts/open-review-app.mjs");
    const verifier = read("scripts/verify-review-bundle.mjs");

    expect(scripts["build:review:web"]).toContain("vite.review.config.ts");
    expect(scripts["dev:review:web"]).toContain("vite.review.config.ts");
    expect(scripts["review:build"]).toContain("--features review-fixtures");
    expect(scripts["review:build"]).toContain("app/tauri.review.conf.json");
    expect(scripts["review:build"]).toContain("--bundles app");
    expect(scripts["review:build"]).not.toContain("prepare:sidecars");
    expect(scripts["review:build"]).not.toContain("clean:dev");
    expect(scripts["review:build"]).not.toContain("7878");
    expect(scripts["review:open"]).toBe("node scripts/open-review-app.mjs");
    expect(scripts["review:open"]).not.toContain("clean:dev");
    expect(scripts["review:open"]).not.toContain("7878");
    expect(launcher).toContain("Wenlan Review.app");
    expect(launcher).toContain("wenlan-review");
    expect(launcher).not.toContain("clean:dev");
    expect(launcher).not.toContain("7878");
    expect(scripts["review:verify"]).toContain("verify-review-bundle.mjs");
    expect(scripts.review).toBe(
      "pnpm review:build && pnpm review:verify && pnpm review:open",
    );
    expect(verifier).toContain("CFBundleIdentifier");
    expect(verifier).toContain("com.wenlan.desktop.review");
    expect(verifier).toContain("wenlan-review");
    expect(verifier).toContain("forbiddenBundleNames");
  });

  it("builds and verifies the fixture-only Review artifact in CI", () => {
    const ci = read(".github/workflows/ci.yml");
    const pkg = JSON.parse(read("package.json"));

    expect(ci).toContain("pnpm review:build && pnpm review:verify");
    expect(pkg.devDependencies["@tauri-apps/cli"]).toMatch(/^\^?2\./);
  });
});
