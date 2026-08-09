// SPDX-License-Identifier: AGPL-3.0-only
import type { Page } from "@playwright/test";

export type ContrastProbe = {
  readonly backgroundSelector?: string;
  readonly foregroundProperty: "border-bottom-color" | "border-top-color" | "color" | "stroke";
  readonly label: string;
  readonly minimum: number;
  readonly selector: string;
};

export type ContrastResult = ContrastProbe & {
  readonly background: string;
  readonly foreground: string;
  readonly ratio: number;
};

export async function renderedContrast(page: Page, probes: readonly ContrastProbe[]): Promise<readonly ContrastResult[]> {
  return page.evaluate((input) => {
    type Color = { r: number; g: number; b: number; a: number };
    const parse = (value: string): Color => {
      const channels = value.match(/rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+))?\s*\)/);
      if (channels) {
        return { r: Number(channels[1]), g: Number(channels[2]), b: Number(channels[3]), a: Number(channels[4] ?? 1) };
      }
      const srgb = value.match(/color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+))?\s*\)/);
      if (!srgb) throw new Error(`Unsupported computed color: ${value}`);
      return { r: Number(srgb[1]) * 255, g: Number(srgb[2]) * 255, b: Number(srgb[3]) * 255, a: Number(srgb[4] ?? 1) };
    };
    const over = (foreground: Color, background: Color): Color => {
      const alpha = foreground.a + background.a * (1 - foreground.a);
      if (alpha === 0) return { r: 0, g: 0, b: 0, a: 0 };
      return {
        r: (foreground.r * foreground.a + background.r * background.a * (1 - foreground.a)) / alpha,
        g: (foreground.g * foreground.a + background.g * background.a * (1 - foreground.a)) / alpha,
        b: (foreground.b * foreground.a + background.b * background.a * (1 - foreground.a)) / alpha,
        a: alpha,
      };
    };
    const effectiveBackground = (start: Element): Color => {
      let color: Color = { r: 0, g: 0, b: 0, a: 0 };
      for (let node: Element | null = start; node; node = node.parentElement) {
        color = over(color, parse(getComputedStyle(node).backgroundColor));
        if (color.a >= 0.999) return color;
      }
      return over(color, { r: 255, g: 255, b: 255, a: 1 });
    };
    const luminance = (color: Color): number => {
      const channel = (value: number): number => {
        const normalized = value / 255;
        return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
      };
      return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
    };
    const ratio = (foreground: Color, background: Color): number => {
      const foregroundLuminance = luminance(over(foreground, background));
      const backgroundLuminance = luminance(background);
      return (Math.max(foregroundLuminance, backgroundLuminance) + 0.05)
        / (Math.min(foregroundLuminance, backgroundLuminance) + 0.05);
    };

    return input.map((probe) => {
      const element = document.querySelector(probe.selector);
      if (!element) throw new Error(`Missing contrast probe: ${probe.selector}`);
      const backgroundElement = probe.backgroundSelector
        ? document.querySelector(probe.backgroundSelector)
        : element;
      if (!backgroundElement) throw new Error(`Missing contrast background: ${probe.backgroundSelector}`);
      const foregroundValue = getComputedStyle(element).getPropertyValue(probe.foregroundProperty);
      const background = effectiveBackground(backgroundElement);
      const foreground = parse(foregroundValue);
      return {
        ...probe,
        background: `rgb(${Math.round(background.r)}, ${Math.round(background.g)}, ${Math.round(background.b)})`,
        foreground: foregroundValue,
        ratio: Number(ratio(foreground, background).toFixed(3)),
      };
    });
  }, probes);
}
