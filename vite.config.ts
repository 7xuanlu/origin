import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const host = process.env.TAURI_DEV_HOST;
const devPort = Number.parseInt(process.env.WENLAN_DEV_UI_PORT ?? "1420", 10);

export default defineConfig(async () => ({
  plugins: [react(), tailwindcss()],
  define: {
    __WENLAN_REVIEW__: "false",
  },
  clearScreen: false,
  server: {
    port: devPort,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: devPort + 1 } : undefined,
    watch: {
      ignored: ["**/app/**"],
    },
  },
}));
