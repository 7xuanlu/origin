// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect, useState, useCallback, useRef } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { availableMonitors } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export default function SnipOverlay() {
  const [dragging, setDragging] = useState(false);
  const [rect, setRect] = useState<Rect | null>(null);
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const monitorRef = useRef<{ x: number; y: number } | null>(null);

  // Fetch monitor offset on mount (position is physical px → convert to points)
  useEffect(() => {
    availableMonitors().then((monitors) => {
      const primary = monitors[0];
      if (primary) {
        monitorRef.current = {
          x: primary.position.x / primary.scaleFactor,
          y: primary.position.y / primary.scaleFactor,
        };
      }
    });
  }, []);

  // Escape to cancel
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setDragging(false);
        setRect(null);
        getCurrentWindow().hide();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    startRef.current = { x: e.clientX, y: e.clientY };
    setDragging(true);
    setRect({ x: e.clientX, y: e.clientY, width: 0, height: 0 });
  }, []);

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging || !startRef.current) return;
      const x = Math.min(startRef.current.x, e.clientX);
      const y = Math.min(startRef.current.y, e.clientY);
      const width = Math.abs(e.clientX - startRef.current.x);
      const height = Math.abs(e.clientY - startRef.current.y);
      setRect({ x, y, width, height });
    },
    [dragging],
  );

  const onMouseUp = useCallback(async () => {
    if (!rect || !dragging) return;
    setDragging(false);

    // Too small — treat as cancel click
    if (rect.width < 10 || rect.height < 10) {
      setRect(null);
      return;
    }

    const win = getCurrentWindow();
    const monitor = monitorRef.current;
    // Monitor position from Tauri is physical pixels; convert to logical (= macOS points)
    const offsetX = monitor?.x ?? 0;
    const offsetY = monitor?.y ?? 0;

    // CGWindowListCreateImage uses global display coordinate space in POINTS
    // (not pixels). CSS pixels in the webview = macOS points, so no scale needed.
    const screenX = rect.x + offsetX;
    const screenY = rect.y + offsetY;

    // Hide overlay first so it's not in the screenshot
    await win.hide();
    setRect(null);

    // Wait for window server to remove the overlay
    await new Promise((r) => setTimeout(r, 100));

    try {
      await invoke("capture_region", {
        x: screenX,
        y: screenY,
        width: rect.width,
        height: rect.height,
      });
    } catch (e) {
      console.error("capture_region failed:", e);
    }
  }, [rect, dragging]);

  return (
    <div
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      style={{
        position: "fixed",
        inset: 0,
        cursor: "crosshair",
        background: rect && rect.width > 0 ? "transparent" : "rgba(0,0,0,0.3)",
        userSelect: "none",
        WebkitUserSelect: "none",
      }}
    >
      {/* Selection rectangle with dimmed surround */}
      {rect && rect.width > 0 && rect.height > 0 && (
        <div
          style={{
            position: "absolute",
            left: rect.x,
            top: rect.y,
            width: rect.width,
            height: rect.height,
            border: "2px solid #60a5fa",
            borderRadius: 2,
            boxShadow: "0 0 0 9999px rgba(0,0,0,0.4)",
            pointerEvents: "none",
          }}
        >
          {/* Dimension label */}
          <div
            style={{
              position: "absolute",
              left: 0,
              top: rect.height + 6,
              fontSize: 11,
              fontFamily: "system-ui, sans-serif",
              color: "rgba(255,255,255,0.8)",
              background: "rgba(0,0,0,0.6)",
              padding: "2px 6px",
              borderRadius: 3,
              whiteSpace: "nowrap",
              pointerEvents: "none",
            }}
          >
            {Math.round(rect.width)} x {Math.round(rect.height)}
          </div>
        </div>
      )}

      {/* Hint text when idle */}
      {!dragging && (!rect || rect.width === 0) && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            fontSize: 14,
            fontFamily: "system-ui, sans-serif",
            color: "rgba(255,255,255,0.6)",
            pointerEvents: "none",
            textAlign: "center",
          }}
        >
          Drag to select · Esc to cancel
        </div>
      )}
    </div>
  );
}
