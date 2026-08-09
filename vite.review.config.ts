// SPDX-License-Identifier: AGPL-3.0-only
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

const local = (path: string) => fileURLToPath(new URL(path, import.meta.url));
const stubs = local("./review/tauri-stubs.ts");

export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  define: {
    __WENLAN_REVIEW__: JSON.stringify(true),
  },
  resolve: {
    alias: {
      "@tauri-apps/api/core": local("./review/tauri-core.ts"),
      "@tauri-apps/api/event": stubs,
      "@tauri-apps/api/app": stubs,
      "@tauri-apps/api/dpi": stubs,
      "@tauri-apps/api/window": stubs,
      "@tauri-apps/api/webviewWindow": stubs,
      "@tauri-apps/plugin-dialog": stubs,
      "@tauri-apps/plugin-fs": stubs,
      "@tauri-apps/plugin-notification": stubs,
      "@tauri-apps/plugin-shell": local("./review/shell.ts"),
      "tauri-plugin-clipboard-x-api": stubs,
    },
  },
  build: {
    outDir: "dist/review",
    emptyOutDir: true,
  },
  server: {
    port: 1422,
    strictPort: true,
    open: false,
    watch: {
      ignored: ["**/app/**"],
      usePolling: true,
      interval: 400,
    },
  },
});
