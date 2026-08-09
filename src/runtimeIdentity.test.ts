import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

describe("runtime product identity", () => {
  it("uses Wenlan app product identity", () => {
    const tauri = JSON.parse(
      readFileSync(resolve(root, "app/tauri.conf.json"), "utf8"),
    );

    expect(tauri.productName).toBe("Wenlan");
    expect(tauri.identifier).toBe("com.wenlan.desktop");
    expect(tauri.app.windows[0].title).toBe("Wenlan");
  });

  it("keeps the persistent native titlebar inset aligned in both app variants", () => {
    const production = JSON.parse(
      readFileSync(resolve(root, "app/tauri.conf.json"), "utf8"),
    );
    const review = JSON.parse(
      readFileSync(resolve(root, "app/tauri.review.conf.json"), "utf8"),
    );

    expect(production.app.windows[0].trafficLightPosition).toEqual({
      x: 16,
      y: 28,
    });
    expect(review.app.windows[0].trafficLightPosition).toEqual({
      x: 16,
      y: 28,
    });
  });

  it("uses Wenlan release artifact names", () => {
    const pkg = JSON.parse(
      readFileSync(resolve(root, "package.json"), "utf8"),
    );

    expect(pkg.name).toBe("wenlan-app");
    expect(pkg.repository.url).toBe("https://github.com/7xuanlu/wenlan-app");
    expect(pkg.scripts["release:dmg"]).toContain("Wenlan_");
    expect(pkg.scripts["release:dmg"]).toContain("-volname Wenlan");
    expect(pkg.scripts["clean:release"]).toContain("Wenlan_*.dmg");
    expect(pkg.scripts["clean:release"]).toContain("Origin_*.dmg");
  });

  it("uses Wenlan Rust package and executable identity", () => {
    const cargo = readFileSync(resolve(root, "app/Cargo.toml"), "utf8");

    expect(cargo).toContain('name = "wenlan-app"');
    expect(cargo).toContain('default-run = "wenlan-app"');
    expect(cargo).toContain('name = "wenlan_lib"');
  });

  it("uses Wenlan updater endpoint", () => {
    const tauri = JSON.parse(
      readFileSync(resolve(root, "app/tauri.conf.json"), "utf8"),
    );

    expect(tauri.plugins.updater.endpoints[0]).toBe(
      "https://github.com/7xuanlu/wenlan-app/releases/latest/download/latest.json",
    );
  });

  it("uses Wenlan web shell and capability identity", () => {
    const html = readFileSync(resolve(root, "index.html"), "utf8");
    const capability = JSON.parse(
      readFileSync(resolve(root, "app/capabilities/default.json"), "utf8"),
    );

    expect(html).toContain("<title>Wenlan</title>");
    expect(html).not.toContain("<title>Origin</title>");
    expect(capability.description).toBe("Default capability for Wenlan");
  });

  it("keeps the light sidebar brighter than the main workspace", () => {
    const css = readFileSync(resolve(root, "src/index.css"), "utf8");

    expect(css).toContain("--bg-primary: #FCFCFB;");
    expect(css).toContain("--bg-tertiary: #FFFFFF;");
    expect(css).toContain("--accent: #5E58C8;");
    expect(css).toContain("--mem-bg: #FCFCFB;");
    expect(css).toContain("--mem-sidebar: #FFFFFF;");
    expect(css).toContain("--mem-surface: #FFFFFF;");
    expect(css).toContain("--mem-account-card: #FCFCFB;");
    expect(css).toContain("--mem-popover: #FFFFFF;");
    expect(css).toContain("--mem-border: #E3E7EE;");
    expect(css).toContain("--mem-accent-warm: #B46A3A;");
    expect(css).toContain("--mem-accent-indigo: #5E58C8;");
    expect(css).toContain("--mem-accent-sage: #6F8F76;");
    expect(css).not.toContain("--mem-bg: #FBFCFD;");
    expect(css).not.toContain("--bg-primary: #FBFCFD;");
    expect(css).not.toContain("--mem-bg: #F7F8FA;");
    expect(css).not.toContain("--bg-primary: #F7F8FA;");
    expect(css).not.toContain("--mem-sidebar: #F2F4F7;");
    expect(css).not.toContain("--mem-account-card: #FFFFFF;");
    expect(css).not.toContain("--mem-bg: #FEFCF9;");
    expect(css).not.toContain("--mem-sidebar: #F5F0E8;");
    expect(css).not.toContain("--mem-surface: #F8F5F0;");
  });

  it("declares the main window visible at launch and keeps a backend reveal fallback", () => {
    const tauri = JSON.parse(
      readFileSync(resolve(root, "app/tauri.conf.json"), "utf8"),
    );
    const lib = readFileSync(resolve(root, "app/src/lib.rs"), "utf8");

    expect(tauri.app.windows[0].visible).toBe(true);
    expect(lib).toContain('handle.listen("app-ready"');
    expect(lib).toContain("set_activation_policy(activation_policy_for_main_window_visible(false))");
    expect(lib).toContain("startup_reveal_fallback_delay");
    expect(lib).toContain("app-ready did not reveal the main window");
    expect(lib).not.toContain("align_macos_traffic_lights");
    expect(lib).not.toContain("setFrameOrigin(button");
  });

  it("routes native and tray quit requests through the frontend persistence gate", () => {
    const lib = readFileSync(resolve(root, "app/src/lib.rs"), "utf8");

    expect(lib).toContain('emit("quit-requested"');
    expect(lib).toContain("RunEvent::ExitRequested");
    expect(lib).toContain("code: None");
    expect(lib).toContain("api.prevent_exit()");
    expect(lib).toContain("lifecycle::is_quitting()");
    expect(lib).toContain("fn cancel_guarded_quit_request()");
    expect(lib).toContain("cancel_guarded_quit_request,");
    expect(lib).not.toContain('handle.listen("quit-cancelled"');
    const exitBranch = lib.slice(lib.indexOf("RunEvent::ExitRequested"));
    expect(exitBranch.indexOf("request_full_quit(app)")).toBeLessThan(
      exitBranch.indexOf("api.prevent_exit()"),
    );
    expect(lib).not.toContain('"quit" => {\n                            let h = handle_for_menu.clone();\n                            tauri::async_runtime::spawn');
  });

  it("prepares sidecar binaries before Tauri validates external bins", () => {
    const tauri = JSON.parse(
      readFileSync(resolve(root, "app/tauri.conf.json"), "utf8"),
    );

    expect(tauri.bundle.externalBin).toEqual(
      expect.arrayContaining([
        "binaries/wenlan",
        "binaries/wenlan-server",
        "binaries/wenlan-mcp",
        "binaries/cloudflared",
      ]),
    );
    expect(tauri.build.beforeDevCommand).toContain("prepare:sidecars");
    expect(tauri.build.beforeBuildCommand).toContain("prepare:sidecars:tauri-build");
  });

  it("has a local app-bundle validation script that does not require updater signing keys", () => {
    const pkg = JSON.parse(
      readFileSync(resolve(root, "package.json"), "utf8"),
    );
    const tauri = JSON.parse(
      readFileSync(resolve(root, "app/tauri.conf.json"), "utf8"),
    );
    const localBundle = JSON.parse(
      readFileSync(resolve(root, "app/tauri.local-bundle.conf.json"), "utf8"),
    );

    expect(tauri.bundle.createUpdaterArtifacts).toBe(true);
    expect(localBundle.bundle.createUpdaterArtifacts).toBe(false);
    expect(pkg.scripts["build:app:local"]).toContain("tauri build --bundles app");
    expect(pkg.scripts["build:app:local"]).toContain(
      "--config app/tauri.local-bundle.conf.json",
    );
    expect(pkg.scripts["build:app:local"]).not.toContain("--no-bundle");

    const productionReleaseScripts = Object.entries(pkg.scripts).filter(
      ([name]) => name === "release" || name.startsWith("release:"),
    );
    expect(productionReleaseScripts.length).toBeGreaterThan(0);
    for (const [, script] of productionReleaseScripts) {
      expect(script).not.toContain("build:app:local");
      expect(script).not.toContain("tauri.local-bundle.conf.json");
    }
  });

  it("keeps product-owned visible copy on Wenlan", () => {
    const productOwnedFiles = [
      "src/components/memory/SettingsPage.tsx",
      "src/components/memory/AboutWenlanDialog.tsx",
      "src/components/memory/ConnectionsList.tsx",
      "src/components/memory/sources/AddSourceDialog.tsx",
      "src/components/memory/sources/SourcesSection.tsx",
      "src/components/memory/PageDetail.tsx",
      "src/components/memory/ImportView.tsx",
      "src/components/memory/SpaceDetail.tsx",
      "src/components/memory/WorthAGlanceScroll.tsx",
      "src/components/memory/settings/SettingsSidebar.tsx",
      "src/components/ChatImport/ImportFlow.tsx",
    ];

    for (const file of productOwnedFiles) {
      const content = readFileSync(resolve(root, file), "utf8");
      expect(content, file).not.toMatch(/\bOrigin\b/);
    }
  });
});
