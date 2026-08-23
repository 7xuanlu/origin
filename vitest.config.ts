import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    __WENLAN_REVIEW__: "false",
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}', 'scripts/**/*.test.ts', 'preview/**/*.test.ts'],
    passWithNoTests: true,
    // The suite includes several jsdom-heavy CodeMirror and graph files.
    // Vitest otherwise uses every available core in run mode, which can
    // starve their async DOM work on developer laptops and Windows runners.
    maxWorkers: 4,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'lcov'],
      include: [
        // Only enforce coverage on modules that have tests.
        // Expand this list as component tests are added.
        'src/lib/tauri.ts',
        'src/lib/graph/**',
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
