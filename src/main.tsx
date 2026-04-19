// SPDX-License-Identifier: AGPL-3.0-only
import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import ToastOverlay from "./components/ToastOverlay";
import SnipOverlay from "./components/SnipOverlay";
import QuickCaptureWindow from "./components/QuickCaptureWindow";
import AmbientOverlay from "./components/AmbientOverlay";
import IconOverlay from "./components/IconOverlay";
import { applyTheme } from "./lib/theme";
import "./index.css";

// Apply theme before first render to prevent flash
applyTheme();


const isToast = window.location.hash === "#toast";
const isSnip = window.location.hash === "#snip";
const isQuickCapture = window.location.hash === "#quick-capture";
const isAmbient = window.location.hash === "#ambient";
const isIcon = window.location.hash === "#icon";

if (isToast) {
  // No StrictMode for the toast overlay — its double-mount behavior causes
  // Tauri event listener registration races in the hidden webview window.
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <ToastOverlay />,
  );
} else if (isSnip) {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <SnipOverlay />
    </React.StrictMode>,
  );
} else if (isQuickCapture) {
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <QuickCaptureWindow />
    </React.StrictMode>,
  );
} else if (isAmbient) {
  // No StrictMode — same reasoning as toast: avoid Tauri event listener races
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <AmbientOverlay />,
  );
} else if (isIcon) {
  // No StrictMode — avoid Tauri event listener races in hidden window
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <IconOverlay />,
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
