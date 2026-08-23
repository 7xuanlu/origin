// SPDX-License-Identifier: AGPL-3.0-only
import { readFile } from "node:fs/promises";

export async function pngDimensions(
  filePath: string,
): Promise<{ readonly height: number; readonly width: number }> {
  const bytes = await readFile(filePath);
  if (bytes.toString("ascii", 1, 4) !== "PNG") throw new Error(`Not a PNG: ${filePath}`);
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}
