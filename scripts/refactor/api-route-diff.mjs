#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "../..");

function parseArgs(argv) {
  const args = {
    app: repoRoot,
    backend: process.env.WENLAN_BACKEND_DIR || null,
    out: null,
    classifications: null,
    json: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--app") {
      args.app = resolve(argv[++i]);
    } else if (arg === "--backend") {
      args.backend = resolve(argv[++i]);
    } else if (arg === "--out") {
      args.out = resolve(argv[++i]);
    } else if (arg === "--classifications") {
      args.classifications = resolve(argv[++i]);
    } else if (arg === "--json") {
      args.json = true;
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }
  return args;
}

function resolveBackend(appRoot, explicitBackend) {
  if (explicitBackend) {
    return resolve(appRoot, explicitBackend);
  }
  const resolver = resolve(appRoot, "scripts/resolve-backend-dir.sh");
  if (!existsSync(resolver)) {
    throw new Error("could not find scripts/resolve-backend-dir.sh; pass --backend");
  }
  return execFileSync("bash", [resolver, appRoot], { encoding: "utf8" }).trim();
}

function normalizeRoute(path) {
  const withoutQuery = path.split("?")[0];
  return withoutQuery
    .replace(/\{[^}/]+\}/g, "{}")
    .replace(/\/+$/g, "")
    .replace(/^$/, "/");
}

function readFiles(root, extensions) {
  const out = [];
  const stack = [root];
  while (stack.length > 0) {
    const dir = stack.pop();
    if (!dir || !existsSync(dir)) continue;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = resolve(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === "target" || entry.name === "node_modules") continue;
        stack.push(path);
      } else if (extensions.some((ext) => entry.name.endsWith(ext))) {
        out.push(path);
      }
    }
  }
  return out.sort();
}

function uniqueRoutes(routes) {
  const byNormalized = new Map();
  for (const route of routes) {
    const normalized = normalizeRoute(route.path);
    if (!byNormalized.has(normalized)) {
      byNormalized.set(normalized, { normalized, paths: new Set(), files: new Set() });
    }
    const current = byNormalized.get(normalized);
    current.paths.add(route.path);
    if (route.file) current.files.add(route.file);
  }
  return [...byNormalized.values()]
    .map((route) => ({
      normalized: route.normalized,
      paths: [...route.paths].sort(),
      files: [...route.files].sort(),
    }))
    .sort((a, b) => a.normalized.localeCompare(b.normalized));
}

function routePathFromLiteral(value) {
  if (value.startsWith("/api/") || value.startsWith("/ws/")) {
    return value;
  }
  const fullUrl = value.match(/^https?:\/\/[^/\s]+((?:\/api\/|\/ws\/)[^?#\s]*)/);
  return fullUrl ? fullUrl[1] : null;
}

function isTestSource(file) {
  return (
    file.includes("/__tests__/") ||
    file.includes("/src/test/") ||
    /\.(test|spec)\.[cm]?[jt]sx?$/.test(file)
  );
}

function extractBackendRoutes(backendRoot) {
  const routerPath = resolve(backendRoot, "crates/wenlan-server/src/router.rs");
  const source = readFileSync(routerPath, "utf8");
  const routes = [];
  const re = /\.route\(\s*"([^"]+)"/g;
  let match;
  while ((match = re.exec(source)) !== null) {
    routes.push({
      path: match[1],
      file: relative(backendRoot, routerPath),
    });
  }
  return uniqueRoutes(routes);
}

function extractAppRoutes(appRoot) {
  const files = [
    ...readFiles(resolve(appRoot, "app/src"), [".rs"]),
    ...readFiles(resolve(appRoot, "src"), [".ts", ".tsx"]),
  ].filter((file) => !isTestSource(file));
  const routes = [];
  const routeLiteral = /["'`]([^"'`\n]*(?:\/api\/|\/ws\/)[^"'`\n]*)["'`]/g;
  for (const file of files) {
    const source = readFileSync(file, "utf8");
    let match;
    while ((match = routeLiteral.exec(source)) !== null) {
      const path = routePathFromLiteral(match[1]);
      if (!path) continue;
      routes.push({
        path,
        file: relative(appRoot, file),
      });
    }
  }
  return uniqueRoutes(routes);
}

function readRouteClassifications(path) {
  if (!path || !existsSync(path)) {
    return new Map();
  }
  const parsed = JSON.parse(readFileSync(path, "utf8"));
  const routes = parsed.routes || {};
  const byNormalized = new Map();
  for (const [routePath, classification] of Object.entries(routes)) {
    byNormalized.set(normalizeRoute(routePath), classification);
  }
  return byNormalized;
}

function isCompleteClassification(classification) {
  return (
    classification &&
    typeof classification.category === "string" &&
    classification.category.trim() &&
    typeof classification.status === "string" &&
    classification.status.trim() &&
    typeof classification.next_action === "string" &&
    classification.next_action.trim()
  );
}

function withClassifications(routes, classifications) {
  return routes.map((route) => {
    const classification = classifications.get(route.normalized);
    return isCompleteClassification(classification) ? { ...route, classification } : route;
  });
}

function buildDiff(appRoot, backendRoot, classifications = new Map()) {
  const backendRoutes = extractBackendRoutes(backendRoot);
  const appRoutes = extractAppRoutes(appRoot);
  const appSet = new Set(appRoutes.map((route) => route.normalized));
  const backendSet = new Set(backendRoutes.map((route) => route.normalized));
  const missingInApp = withClassifications(
    backendRoutes.filter((route) => !appSet.has(route.normalized)),
    classifications,
  );
  const appOnly = appRoutes.filter((route) => !backendSet.has(route.normalized));
  const classifiedMissingInApp = missingInApp.filter((route) => route.classification).length;
  const unclassifiedMissingInApp = missingInApp.length - classifiedMissingInApp;
  return {
    appRoot: ".",
    backendRoot: basename(backendRoot),
    counts: {
      backendRoutes: backendRoutes.length,
      appSourceRoutes: appRoutes.length,
      missingInApp: missingInApp.length,
      classifiedMissingInApp,
      unclassifiedMissingInApp,
      appOnly: appOnly.length,
    },
    backendRoutes,
    appRoutes,
    missingInApp,
    appOnly,
  };
}

function renderMarkdown(diff) {
  const lines = [
    "# Wenlan App API Route Diff",
    "",
    `- **App root:** \`${diff.appRoot}\``,
    `- **Backend root:** \`${diff.backendRoot}\``,
    "",
    "## Counts",
    "",
    `- backend route paths: ${diff.counts.backendRoutes}`,
    `- app source route paths: ${diff.counts.appSourceRoutes}`,
    `- backend routes with no direct app source path: ${diff.counts.missingInApp}`,
    `- classified backend route gaps: ${diff.counts.classifiedMissingInApp}`,
    `- unclassified backend route gaps: ${diff.counts.unclassifiedMissingInApp}`,
    `- app source paths with no backend router path: ${diff.counts.appOnly}`,
    "",
    "## Backend Routes With No Direct App Source Path",
    "",
  ];
  if (diff.missingInApp.length === 0) {
    lines.push("- none");
  } else {
    lines.push("| Route | Category | Status | Next Action |");
    lines.push("|---|---|---|---|");
    for (const route of diff.missingInApp) {
      const classification = route.classification;
      lines.push(
        classification
          ? `| \`${route.paths[0]}\` | \`${escapeMarkdownTableCell(classification.category)}\` | ${escapeMarkdownTableCell(classification.status)} | ${escapeMarkdownTableCell(classification.next_action)} |`
          : `| \`${route.paths[0]}\` | \`unclassified\` |  |  |`,
      );
    }
  }
  lines.push("", "## Unclassified Backend Route Gaps", "");
  const unclassified = diff.missingInApp.filter((route) => !route.classification);
  if (unclassified.length === 0) {
    lines.push("- none");
  } else {
    for (const route of unclassified) {
      lines.push(`- \`${route.paths[0]}\``);
    }
  }
  lines.push("", "## App Source Paths With No Backend Router Path", "");
  if (diff.appOnly.length === 0) {
    lines.push("- none");
  } else {
    for (const route of diff.appOnly) {
      lines.push(`- \`${route.paths[0]}\` (${route.files.join(", ")})`);
    }
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function escapeMarkdownTableCell(value) {
  return String(value ?? "").replaceAll("|", "\\|");
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const appRoot = resolve(args.app);
  const backendRoot = resolveBackend(appRoot, args.backend);
  const outDir = args.out || resolve(appRoot, "docs/superpowers/refactor/wenlan-app-inventory");
  const classificationsPath =
    args.classifications || resolve(outDir, "api-route-classifications.json");
  const classifications = readRouteClassifications(classificationsPath);
  const diff = buildDiff(appRoot, backendRoot, classifications);

  mkdirSync(outDir, { recursive: true });
  writeFileSync(resolve(outDir, "api-route-diff.json"), `${JSON.stringify(diff, null, 2)}\n`);
  writeFileSync(resolve(outDir, "api-route-diff.md"), renderMarkdown(diff));

  if (args.json) {
    process.stdout.write(`${JSON.stringify(diff.counts)}\n`);
  } else {
    process.stdout.write(
      `Route diff written to ${outDir} (${diff.counts.backendRoutes} backend, ${diff.counts.appSourceRoutes} app, ${diff.counts.missingInApp} missing)\n`,
    );
  }
}

main();
