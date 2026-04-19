// SPDX-License-Identifier: AGPL-3.0-only
import { useEffect } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

export function useEscapeToHide() {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        getCurrentWindow().hide();
      }
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);
}
