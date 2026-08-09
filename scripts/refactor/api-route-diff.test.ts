import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { afterEach, describe, expect, it } from "vitest";

const tempRoots: string[] = [];
const scriptPath = resolve(import.meta.dirname, "api-route-diff.mjs");

afterEach(() => {
  for (const dir of tempRoots.splice(0)) {
    rmSync(dir, { force: true, recursive: true });
  }
});

function makeTempRoot(): string {
  const dir = mkdtempSync(resolve(tmpdir(), "wenlan-app-api-routes-"));
  tempRoots.push(dir);
  return dir;
}

function writeFixture(appRoot: string, backendRoot: string): void {
  mkdirSync(resolve(appRoot, "app/src"), { recursive: true });
  mkdirSync(resolve(appRoot, "src"), { recursive: true });
  mkdirSync(resolve(appRoot, "src/__tests__"), { recursive: true });
  mkdirSync(resolve(backendRoot, "crates/wenlan-server/src"), { recursive: true });

  writeFileSync(
    resolve(appRoot, "app/src/api.rs"),
    `pub async fn health(client: Client) {
  client.get_json("/api/health").await;
  let path = format!("/api/memory/{}/detail", "mem_1");
  client.get_json(&path).await;
  let label = "not a route";
  client.post_json("/api/pages/export", &()).await;
  let recent = format!("/api/pages/recent?limit={}", 5);
  client.get_json(&recent).await;
  client.post("http://127.0.0.1:7878/api/shutdown").send().await;
}
`,
  );
  writeFileSync(
    resolve(appRoot, "src/tauri.ts"),
    `import { invoke } from "@tauri-apps/api/core";
export const ignored = "not an api path";
`,
  );
  writeFileSync(
    resolve(appRoot, "src/__tests__/routes.test.ts"),
    `export const testOnlyRoute = "/api/pages/page-1/links";
`,
  );
  writeFileSync(
    resolve(backendRoot, "crates/wenlan-server/src/router.rs"),
    `Router::new()
  .route("/api/health", get(handle_health))
  .route(
    "/api/context",
    post(handle_context),
  )
  .route("/api/memory/{id}/detail", get(handle_detail))
  .route("/api/pages/export", post(handle_export))
  .route("/api/pages/recent", get(handle_recent))
  .route("/api/shutdown", post(handle_shutdown))
`,
  );
}

describe("api route diff", () => {
  it("normalizes backend and app route paths into a reproducible diff", () => {
    const root = makeTempRoot();
    const appRoot = resolve(root, "wenlan-app");
    const backendRoot = resolve(root, "wenlan");
    const outDir = resolve(root, "out");
    writeFixture(appRoot, backendRoot);

    const result = spawnSync(
      "node",
      [scriptPath, "--app", appRoot, "--backend", backendRoot, "--out", outDir, "--json"],
      { encoding: "utf8" },
    );

    expect(result.status, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      backendRoutes: 6,
      appSourceRoutes: 5,
      missingInApp: 1,
      classifiedMissingInApp: 0,
      unclassifiedMissingInApp: 1,
      appOnly: 0,
    });

    const markdown = readFileSync(resolve(outDir, "api-route-diff.md"), "utf8");
    const json = readFileSync(resolve(outDir, "api-route-diff.json"), "utf8");
    expect(JSON.parse(json).backendRoot).toBe("wenlan");
    expect(markdown).not.toContain(appRoot);
    expect(markdown).not.toContain(backendRoot);
    expect(json).not.toContain(appRoot);
    expect(json).not.toContain(backendRoot);
    expect(markdown).toContain("`/api/context`");
    expect(markdown).not.toContain("`/api/health`");
    expect(markdown).not.toContain("`/api/memory/{id}/detail`");
    expect(markdown).not.toContain("`/api/pages/export`");
    expect(markdown).not.toContain("`/api/pages/recent`");
    expect(markdown).not.toContain("`/api/shutdown`");
  });

  it("attaches route classifications and reports unclassified route gaps", () => {
    const root = makeTempRoot();
    const appRoot = resolve(root, "wenlan-app");
    const backendRoot = resolve(root, "wenlan");
    const outDir = resolve(root, "out");
    const classificationsPath = resolve(root, "api-route-classifications.json");
    writeFixture(appRoot, backendRoot);
    writeFileSync(
      classificationsPath,
      JSON.stringify(
        {
          routes: {
            "/api/context": {
              category: "agent_route",
              status: "intentional",
              rationale: "Context bundles are an MCP/agent contract, not a desktop command.",
              next_action: "Keep hidden | diagnostics panel must be designed.",
            },
          },
        },
        null,
        2,
      ),
    );

    const result = spawnSync(
      "node",
      [
        scriptPath,
        "--app",
        appRoot,
        "--backend",
        backendRoot,
        "--out",
        outDir,
        "--classifications",
        classificationsPath,
        "--json",
      ],
      { encoding: "utf8" },
    );

    expect(result.status, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      backendRoutes: 6,
      appSourceRoutes: 5,
      missingInApp: 1,
      classifiedMissingInApp: 1,
      unclassifiedMissingInApp: 0,
      appOnly: 0,
    });

    const markdown = readFileSync(resolve(outDir, "api-route-diff.md"), "utf8");
    const json = JSON.parse(readFileSync(resolve(outDir, "api-route-diff.json"), "utf8"));
    expect(json.missingInApp[0].classification).toEqual({
      category: "agent_route",
      status: "intentional",
      rationale: "Context bundles are an MCP/agent contract, not a desktop command.",
      next_action: "Keep hidden | diagnostics panel must be designed.",
    });
    expect(markdown).toContain("| `/api/context` | `agent_route` | intentional |");
    expect(markdown).toContain("Keep hidden \\| diagnostics panel must be designed.");
    expect(markdown).toContain("- none");
  });

  it("counts incomplete classifications as unclassified and matches normalized parameter keys", () => {
    const root = makeTempRoot();
    const appRoot = resolve(root, "wenlan-app");
    const backendRoot = resolve(root, "wenlan");
    const outDir = resolve(root, "out");
    const classificationsPath = resolve(root, "api-route-classifications.json");
    writeFixture(appRoot, backendRoot);
    writeFileSync(
      classificationsPath,
      JSON.stringify(
        {
          routes: {
            "/api/context": {},
            "/api/memory/{source_id}/detail": {
              category: "alternate_route",
              status: "intentional",
              rationale: "Parameter names should normalize before matching.",
              next_action: "No action.",
            },
          },
        },
        null,
        2,
      ),
    );

    const result = spawnSync(
      "node",
      [
        scriptPath,
        "--app",
        appRoot,
        "--backend",
        backendRoot,
        "--out",
        outDir,
        "--classifications",
        classificationsPath,
        "--json",
      ],
      { encoding: "utf8" },
    );

    expect(result.status, result.stderr).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      backendRoutes: 6,
      appSourceRoutes: 5,
      missingInApp: 1,
      classifiedMissingInApp: 0,
      unclassifiedMissingInApp: 1,
      appOnly: 0,
    });

    const markdown = readFileSync(resolve(outDir, "api-route-diff.md"), "utf8");
    const json = JSON.parse(readFileSync(resolve(outDir, "api-route-diff.json"), "utf8"));
    expect(json.missingInApp[0].classification).toBeUndefined();
    expect(markdown).toContain("| `/api/context` | `unclassified` |  |  |");
    expect(markdown).toContain("## Unclassified Backend Route Gaps\n\n- `/api/context`");
  });
});
