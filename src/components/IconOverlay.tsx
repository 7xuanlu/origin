// SPDX-License-Identifier: AGPL-3.0-only

import { useEffect, useState, useRef, useCallback } from "react";
import { getCurrentWindow, primaryMonitor } from "@tauri-apps/api/window";
import { LogicalSize, LogicalPosition } from "@tauri-apps/api/dpi";
import { listen } from "@tauri-apps/api/event";
import {
  type ShowIconPayload,
  listenShowIcon,
  triggerIconClick,
} from "../lib/ambient";

/** Icon window dimensions — square */
const WIN_W = 30;
const WIN_H = 30;
/** Gap between cursor position and icon top edge */
const GAP = 8;
/** Screen edge padding */
const PADDING = 16;

export default function IconOverlay() {
  const [payload, setPayload] = useState<ShowIconPayload | null>(null);
  const [visible, setVisible] = useState(false);
  const payloadRef = useRef<ShowIconPayload | null>(null);

  // Keep ref in sync
  useEffect(() => { payloadRef.current = payload; }, [payload]);

  // Listen for show-icon events from backend
  useEffect(() => {
    const unlisten = listenShowIcon((p) => {
      setPayload(p);
      setVisible(true);
      positionWindow(p.x, p.y);
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  // Listen for hide-icon events (click/deselect/app switch)
  useEffect(() => {
    const unlisten = listen("hide-icon", () => {
      setVisible(false);
      setPayload(null);
    });
    return () => { unlisten.then((fn) => fn()); };
  }, []);

  // Auto-dismiss after 4s (covers Cmd+Tab and other non-click focus changes)
  useEffect(() => {
    if (!visible) return;
    const t = setTimeout(() => { setVisible(false); setPayload(null); }, 4000);
    return () => clearTimeout(t);
  }, [visible]);

  // Show/hide window based on visible state
  useEffect(() => {
    const win = getCurrentWindow();
    if (visible) {
      win.show();
    } else {
      win.hide();
    }
  }, [visible]);

  const handleClick = useCallback(async () => {
    const p = payloadRef.current;
    if (!p) return;
    setVisible(false);
    setPayload(null);
    getCurrentWindow().hide();
    await triggerIconClick(p.text, p.x, p.y);
  }, []);

  if (!visible || !payload) return null;

  return (
    <div
      onClick={handleClick}
      className="flex items-center justify-center w-full h-full cursor-pointer select-none"
      style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}
    >
      {/* Origin icon — 30×30 square, no shadow */}
      <div
        className="flex items-center justify-center rounded-md animate-[mem-fade-up_200ms_ease-out]"
        style={{
          width: 30,
          height: 30,
          background: "linear-gradient(135deg, var(--mem-accent-warm), var(--mem-accent-amber))",
        }}
      >
        <span style={{ color: "#fff", fontSize: 14, lineHeight: 1 }}>◈</span>
      </div>
    </div>
  );
}

async function positionWindow(macosX: number, macosY: number) {
  try {
    const monitor = await primaryMonitor();
    if (!monitor) return;
    const scale = monitor.scaleFactor;
    const screenW = monitor.size.width / scale;
    const screenH = monitor.size.height / scale;

    // CG coordinates and Tauri both use top-left origin — no Y flip needed.
    // (The old flip was wrong: it used monitor physical/scale height instead
    // of the CG point space height, causing progressive offset from center.)
    let x = macosX - WIN_W / 2;
    let y = macosY + GAP;

    // Clamp to screen edges
    x = Math.max(PADDING, Math.min(x, screenW - WIN_W - PADDING));
    y = Math.max(PADDING, Math.min(y, screenH - WIN_H - PADDING));

    console.warn(`[icon] input=(${macosX.toFixed(0)},${macosY.toFixed(0)}) screen=${screenW}x${screenH} scale=${scale} → pos=(${x.toFixed(0)},${y.toFixed(0)})`);

    const win = getCurrentWindow();
    await win.setSize(new LogicalSize(WIN_W, WIN_H));
    await win.setPosition(new LogicalPosition(x, y));
  } catch (e) {
    console.error("[icon] position error:", e);
  }
}
