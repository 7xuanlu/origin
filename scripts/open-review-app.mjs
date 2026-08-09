// SPDX-License-Identifier: AGPL-3.0-only
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const REVIEW_PROCESS = "wenlan-review";
const moduleFilePath = import.meta.url.startsWith("file:")
  ? fileURLToPath(import.meta.url)
  : "";
const REVIEW_APP_PATH = moduleFilePath
  ? resolve(dirname(moduleFilePath), "../target/debug/bundle/macos/Wenlan Review.app")
  : resolve(process.cwd(), "target/debug/bundle/macos/Wenlan Review.app");

function getReviewPids() {
  const result = spawnSync("pgrep", ["-x", REVIEW_PROCESS], {
    encoding: "utf8",
  });
  if (result.status === 1) return [];
  if (result.status !== 0) {
    throw new Error(result.stderr.trim() || "Unable to inspect the Review process.");
  }
  return result.stdout
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map(Number);
}

function terminateReviewProcess(pid) {
  try {
    process.kill(pid, "SIGTERM");
  } catch (error) {
    if (error?.code !== "ESRCH") throw error;
  }
}

function launchReviewBundle() {
  const result = spawnSync("open", [REVIEW_APP_PATH], { stdio: "inherit" });
  if (result.status !== 0) {
    throw new Error("Unable to open the Wenlan Review app.");
  }
}

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

export async function restartReviewApp({
  getPids = getReviewPids,
  terminate = terminateReviewProcess,
  sleep = delay,
  launch = launchReviewBundle,
  pollIntervalMs = 50,
  timeoutMs = 5_000,
} = {}) {
  for (const pid of getPids()) terminate(pid);

  const deadline = Date.now() + timeoutMs;
  while (getPids().length > 0) {
    if (Date.now() >= deadline) {
      throw new Error("The previous Wenlan Review process did not exit.");
    }
    await sleep(pollIntervalMs);
  }

  launch();
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === moduleFilePath) {
  await restartReviewApp();
}
