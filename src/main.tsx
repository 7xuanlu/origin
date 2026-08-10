// SPDX-License-Identifier: AGPL-3.0-only
import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import ToastOverlay from "./components/ToastOverlay";
import QuickCaptureWindow from "./components/QuickCaptureWindow";
import { initializeI18n } from "./i18n";
import { applyTheme } from "./lib/theme";
import "./devTools";
import "./index.css";

// Apply theme before first render to prevent flash
applyTheme();

void bootstrap().catch((error: unknown) => {
  console.error("[bootstrap] failed to initialize the Wenlan UI", error);
  document.documentElement.dataset.bootstrapFailed = "true";
});

async function bootstrap() {
  await initializeI18n();

  const isToast = window.location.hash === "#toast";
  const isQuickCapture = window.location.hash === "#quick-capture";

  if (isToast) {
    // No StrictMode for the toast overlay — its double-mount behavior causes
    // Tauri event listener registration races in the hidden webview window.
    ReactDOM.createRoot(document.getElementById("root")!).render(
      <ToastOverlay />,
    );
  } else if (isQuickCapture) {
    ReactDOM.createRoot(document.getElementById("root")!).render(
      <React.StrictMode>
        <QuickCaptureWindow />
      </React.StrictMode>,
    );
  } else {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
          refetchOnWindowFocus: false,
        },
      },
    });

    ReactDOM.createRoot(document.getElementById("root")!).render(
      <React.StrictMode>
        <QueryClientProvider client={queryClient}>
          <App />
        </QueryClientProvider>
      </React.StrictMode>,
    );
  }
}
