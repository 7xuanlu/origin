// SPDX-License-Identifier: AGPL-3.0-only
import { getCurrentWindow, LogicalSize, LogicalPosition, currentMonitor } from "@tauri-apps/api/window";

/** Resize without moving — preserves current window position */
export async function resizeWindow(width: number, height: number): Promise<void> {
  const win = getCurrentWindow();
  await win.setSize(new LogicalSize(width, height));
}

/**
 * Position window for Spotlight mode — horizontally centered, ~22% from top.
 * When the window later resizes via resizeWindow(), it grows downward from this position.
 */
export async function resizeWindowSpotlight(width: number, height: number): Promise<void> {
  const win = getCurrentWindow();
  const monitor = await currentMonitor();
  const scaleFactor = await win.scaleFactor();

  await win.setSize(new LogicalSize(width, height));

  if (monitor) {
    const monitorX = monitor.position.x / scaleFactor;
    const monitorY = monitor.position.y / scaleFactor;
    const monitorW = monitor.size.width / scaleFactor;
    const monitorH = monitor.size.height / scaleFactor;

    const newX = monitorX + (monitorW - width) / 2;
    const newY = monitorY + monitorH * 0.22;

    await win.setPosition(new LogicalPosition(Math.max(monitorX, newX), Math.max(monitorY, newY)));
  }
}

export async function resizeWindowCentered(width: number, height: number): Promise<void> {
  const win = getCurrentWindow();
  const monitor = await currentMonitor();
  const scaleFactor = await win.scaleFactor();

  // Set size first, then position — avoids race where position is set
  // relative to the old size
  await win.setSize(new LogicalSize(width, height));

  if (monitor) {
    const monitorX = monitor.position.x / scaleFactor;
    const monitorY = monitor.position.y / scaleFactor;
    const monitorW = monitor.size.width / scaleFactor;
    const monitorH = monitor.size.height / scaleFactor;

    const newX = monitorX + (monitorW - width) / 2;
    const newY = monitorY + (monitorH - height) / 2;

    await win.setPosition(new LogicalPosition(Math.max(monitorX, newX), Math.max(monitorY, newY)));
  }
}
