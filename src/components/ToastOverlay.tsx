// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { availableMonitors } from "@tauri-apps/api/window";
import { LogicalPosition } from "@tauri-apps/api/dpi";
import { Toaster, toast } from "sonner";

const SOURCE_CONFIG: Record<string, { label: string; color: string; silent?: boolean }> = {
  clipboard:      { label: "Clipboard", color: "#4ade80" },
  screen_capture: { label: "Screen",    color: "#fbbf24" },
  local_files:    { label: "Files",     color: "#60a5fa" },
  manual:         { label: "Captured",  color: "#c084fc" },
  ambient:        { label: "Ambient",   color: "#2dd4bf", silent: true },
  focus:          { label: "Focus",     color: "#fbbf24", silent: true },
  focus_capture:  { label: "Focus",     color: "#fbbf24", silent: true },
  hotkey:         { label: "Capture",   color: "#c084fc" },
  hotkey_capture: { label: "Capture",   color: "#c084fc" },
  snip:           { label: "Snip",      color: "#fb7185" },
  snip_capture:   { label: "Snip",      color: "#fb7185" },
  quick_thought:  { label: "Thought",   color: "#f472b6" },
  thought:        { label: "Thought",   color: "#f472b6" },
};

const pillStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  borderRadius: 9999,
  backgroundColor: "#14141c",
  border: "1px solid #2a2a3a",
  padding: "6px 12px",
};

const dotStyle = (color: string): React.CSSProperties => ({
  flexShrink: 0,
  height: 6,
  width: 6,
  borderRadius: 9999,
  backgroundColor: color,
});

const labelStyle: React.CSSProperties = {
  flexShrink: 0,
  fontSize: 11,
  fontWeight: 500,
  color: "rgba(255,255,255,0.9)",
};

const summaryStyle: React.CSSProperties = {
  fontSize: 11,
  color: "rgba(255,255,255,0.4)",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
  maxWidth: 180,
};

const WINDOW_WIDTH = 340;
const WINDOW_HEIGHT = 200;
const TOAST_DURATION = 3000;

async function positionAndShow(win: ReturnType<typeof getCurrentWindow>) {
  const monitors = await availableMonitors();
  const primary = monitors[0];
  if (primary) {
    const scale = primary.scaleFactor;
    await win.setPosition(
      new LogicalPosition(
        primary.size.width / scale - WINDOW_WIDTH,
        primary.size.height / scale - WINDOW_HEIGHT,
      ),
    );
  }
  await win.show();
}

interface CapturePayload {
  source: string;
  source_id: string;
  summary: string;
  chunks: number;
  processing: boolean;
}

export default function ToastOverlay() {
  const activeCount = useRef(0);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingCapture = useRef(false);

  useEffect(() => {
    const win = getCurrentWindow();
    let cancelled = false;
    let unlistenFn: (() => void) | null = null;

    listen<CapturePayload>(
      "capture-event",
      (event) => {
        const { source, source_id, summary, chunks } = event.payload;
        const config = SOURCE_CONFIG[source] ?? SOURCE_CONFIG.manual;

        // Silent sources (ambient, focus) never show toasts
        if (config.silent) return;

        // Capture-started (empty source_id): instant feedback on hotkey/snip
        if (!source_id) {
          pendingCapture.current = true;
          showToast(win, config, "");
          return;
        }

        // LLM completion / status updates → no toast (shown in MemoryView)
        if (chunks === 0) return;

        // Real capture: if a pending toast already showed, skip the second toast
        if (pendingCapture.current) {
          pendingCapture.current = false;
          return;
        }

        // Captures without a preceding started event (clipboard, file indexer)
        showToast(win, config, summary);
      },
    ).then((fn) => {
      if (cancelled) {
        fn();
      } else {
        unlistenFn = fn;
      }
    });

    function showToast(
      win: ReturnType<typeof getCurrentWindow>,
      config: { label: string; color: string },
      text: string,
    ) {
      positionAndShow(win).catch(() => {});
      activeCount.current = Math.max(0, activeCount.current) + 1;

      if (hideTimer.current) clearTimeout(hideTimer.current);
      hideTimer.current = setTimeout(() => {
        activeCount.current = 0;
        win.hide();
      }, TOAST_DURATION + 1000);

      const onClose = () => {
        activeCount.current--;
        if (activeCount.current <= 0) {
          activeCount.current = 0;
          if (hideTimer.current) clearTimeout(hideTimer.current);
          win.hide();
        }
      };

      toast.custom(
        () => (
          <div style={pillStyle}>
            <span style={dotStyle(config.color)} />
            <span style={labelStyle}>{config.label}</span>
            <span style={summaryStyle}>{text}</span>
          </div>
        ),
        {
          duration: TOAST_DURATION,
          onAutoClose: onClose,
          onDismiss: onClose,
        },
      );
    }

    return () => {
      cancelled = true;
      if (unlistenFn) unlistenFn();
      if (hideTimer.current) clearTimeout(hideTimer.current);
    };
  }, []);

  return (
    <>
      <Toaster
        position="bottom-right"
        visibleToasts={4}
        duration={TOAST_DURATION}
        gap={6}
        offset={0}
        toastOptions={{ unstyled: true }}
      />
    </>
  );
}
