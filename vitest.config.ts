import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    passWithNoTests: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'lcov'],
      include: [
        // Only enforce coverage on modules that have tests.
        // Expand this list as component tests are added.
        'src/lib/tauri.ts',
        'src/lib/processingStore.ts',
        'src/lib/captureHeartbeat.ts',
        'src/hooks/useSearch.ts',
      ],
      exclude: [
        'src/test/**',
      ],
      thresholds: {
        lines: 90,
        functions: 90,
        branches: 85,
        statements: 90,
      },
    },
  },
});
