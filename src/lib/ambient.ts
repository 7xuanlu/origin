// SPDX-License-Identifier: AGPL-3.0-only

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type AmbientCardKind = "person_context" | "decision_reminder";
export type AmbientMode = "proactive" | "on_demand" | "off";

export interface MemorySnippet {
  source: string;
  text: string;
}

export interface AmbientCard {
  card_id: string;
  kind: AmbientCardKind;
  title: string;
  topic: string;
  body: string;
  sources: string[];
  memory_count: number;
  primary_source_id: string;
  created_at: number;
  loading?: boolean;
  snippets?: MemorySnippet[];
}

export function listenAmbientCard(
  callback: (card: AmbientCard) => void,
): Promise<UnlistenFn> {
  return listen<AmbientCard>("ambient-card", (event) => {
    callback(event.payload);
  });
}

export async function dismissAmbientCard(query: string): Promise<void> {
  return invoke("dismiss_ambient_card", { query });
}

export interface AmbientTriggerResult {
  cards_emitted: number;
  context_summary: string;
  reason: string | null;
}

export async function triggerAmbient(): Promise<AmbientTriggerResult> {
  return invoke("trigger_ambient");
}

export async function getAmbientMode(): Promise<AmbientMode> {
  return invoke("get_ambient_mode");
}

export async function setAmbientMode(mode: AmbientMode): Promise<void> {
  return invoke("set_ambient_mode", { mode });
}

export interface SelectionCardPayload {
  card: AmbientCard;
  cursor_x: number;  // macOS logical coords (origin = bottom-left)
  cursor_y: number;
}

export function listenSelectionCard(
  callback: (payload: SelectionCardPayload) => void,
): Promise<UnlistenFn> {
  return listen<SelectionCardPayload>("selection-card", (event) => {
    callback(event.payload);
  });
}

export async function getSelectionCaptureEnabled(): Promise<boolean> {
  return invoke("get_selection_capture_enabled");
}

export async function setSelectionCaptureEnabled(enabled: boolean): Promise<void> {
  return invoke("set_selection_capture_enabled", { enabled });
}

export async function checkAccessibilityPermission(): Promise<boolean> {
  return invoke("check_accessibility_permission");
}

export async function requestAccessibilityPermission(): Promise<void> {
  return invoke("request_accessibility_permission");
}

export interface ShowIconPayload {
  text: string;
  x: number;  // macOS logical coords (origin = bottom-left)
  y: number;
}

export function listenShowIcon(
  callback: (payload: ShowIconPayload) => void,
): Promise<UnlistenFn> {
  return listen<ShowIconPayload>("show-icon", (event) => {
    callback(event.payload);
  });
}

export async function triggerIconClick(text: string, x: number, y: number): Promise<void> {
  return invoke("trigger_icon_click", { text, x, y });
}
