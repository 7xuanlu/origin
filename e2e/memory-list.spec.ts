// SPDX-License-Identifier: AGPL-3.0-only
import { writeFile } from "node:fs/promises";
import { expect, test, type Page } from "@playwright/test";
import { installTauriMock } from "./tauriMock";
import type { MemoryItem } from "../src/lib/tauri";

const evidenceDir = process.env.MEMORY_LIST_EVIDENCE_DIR;
test.describe.configure({ mode: "serial" });
const memoryFixtures: readonly MemoryItem[] = [
  {
    source_id: "mem-local-first",
    title: "Local-first decision",
    content: "Wenlan keeps the review workflow local-first so source context stays inspectable.",
    summary: "Local-first review workflow",
    memory_type: "decision",
    domain: "Wenlan",
    source_agent: "claude-code",
    confidence: 0.92,
    confirmed: true,
    pinned: true,
    supersedes: null,
    last_modified: 1_700_000_000,
    chunk_count: 2,
    access_count: 7,
    is_recap: false,
    stability: "confirmed",
  },
  {
    source_id: "mem-dark-mode",
    title: "Prefers dark mode",
    content: "Lucian prefers dark mode in editors and terminals.",
    summary: null,
    memory_type: "preference",
    domain: "Work",
    source_agent: "chatgpt",
    confidence: 0.84,
    confirmed: false,
    pinned: false,
    supersedes: null,
    last_modified: 1_699_999_800,
    chunk_count: 1,
    access_count: 3,
    is_recap: false,
    stability: "new",
  },
  {
    source_id: "mem-ci-fact",
    title: "CI placeholder sidecars are compile-time only",
    content: "The desktop app uses placeholder sidecars for compile checks; daemon runtime proof is separate.",
    summary: null,
    memory_type: "fact",
    domain: "Release",
    source_agent: "claude-code",
    confidence: 0.78,
    confirmed: true,
    pinned: false,
    supersedes: null,
    last_modified: 1_699_999_500,
    chunk_count: 1,
    access_count: 2,
    is_recap: false,
    stability: "confirmed",
  },
];

async function capture(page: Page, name: string): Promise<void> {
  if (!evidenceDir) return;
  await page.screenshot({ path: `${evidenceDir}/${name}.png`, fullPage: true });
}

async function writeConsoleEvidence(pageErrors: readonly string[], consoleErrors: readonly string[]): Promise<void> {
  if (!evidenceDir) return;
  await writeFile(
    `${evidenceDir}/console.json`,
    `${JSON.stringify({ pageErrors, consoleErrors }, null, 2)}\n`,
  );
}

test("opens a memory from the parent list and returns with Escape", async ({ page }) => {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });

  await page.setViewportSize({ width: 1280, height: 900 });
  await installTauriMock(page, { locale: "en", rawActions: [], memories: memoryFixtures });
  await page.goto("/");

  await page.getByRole("button", { name: "Memories" }).click();

  const list = page.getByRole("region", { name: "Memory list" });
  await expect(list).toBeVisible();
  const row = list.getByRole("article", { name: /Local-first decision/i });
  await expect(row).toBeVisible();
  await expect(row.getByText("Type")).toBeVisible();
  await expect(row.getByText("Space")).toBeVisible();
  await expect(row.getByText("Agent")).toBeVisible();
  await expect(row.getByText("Status")).toBeVisible();
  await expect(row.getByText("Updated")).toBeVisible();
  await capture(page, "memory-list-desktop");

  await page.setViewportSize({ width: 768, height: 900 });
  await capture(page, "memory-list-tablet");

  await page.setViewportSize({ width: 375, height: 812 });
  await expect(page.getByTitle("Show sidebar")).toBeVisible();
  await page.getByTitle("Show sidebar").click();
  await expect(page.getByTitle("Hide sidebar")).toBeVisible();
  await expect(page.locator("aside")).toHaveAttribute("aria-hidden", "false");
  await page.getByTitle("Hide sidebar").click();
  await expect(page.getByTitle("Show sidebar")).toBeVisible();
  await expect(page.locator("aside")).toHaveAttribute("aria-hidden", "true");
  await capture(page, "memory-list-mobile");

  await row.getByRole("button", { name: "Open memory" }).click();
  await expect(page.getByRole("main", { name: "Memory dossier" })).toBeVisible();
  await expect(page.getByText("Local-first decision")).toBeVisible();
  await capture(page, "memory-detail-clickthrough");

  await page.setViewportSize({ width: 1280, height: 900 });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("region", { name: "Memory list" })).toBeVisible();
  await capture(page, "memory-list-returned");

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
  await writeConsoleEvidence(pageErrors, consoleErrors);
});

test("renders the empty parent memory list", async ({ page }) => {
  const pageErrors: string[] = [];
  const consoleErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") {
      consoleErrors.push(message.text());
    }
  });

  await page.setViewportSize({ width: 1280, height: 900 });
  await installTauriMock(page, { locale: "en", rawActions: [], memories: [] });
  await page.goto("/");
  await page.getByRole("button", { name: "Memories" }).click();

  await expect(page.getByRole("region", { name: "Memory list" })).toBeVisible();
  await expect(page.getByText("No memories yet")).toBeVisible();
  await capture(page, "memory-list-empty");

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
  await writeConsoleEvidence(pageErrors, consoleErrors);
});
