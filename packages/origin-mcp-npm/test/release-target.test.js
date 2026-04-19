const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const {
  derivePrereleasePort,
  getPackageReleaseTag,
  resolveReleaseTarget,
  resolveRuntimeTarget,
  sanitizeTagForPath,
} = require('../scripts/release-target');

test('stable versions use latest release and shared bin dir', () => {
  const target = resolveReleaseTarget({
    env: {},
    version: '0.1.0',
    homeDir: '/tmp/origin-home',
  });

  assert.equal(target.tag, null);
  assert.equal(target.source, 'latest');
  assert.equal(target.apiUrl, 'https://api.github.com/repos/7xuanlu/origin/releases/latest');
  assert.equal(target.binDir, path.join('/tmp/origin-home', '.origin', 'bin'));
  assert.equal(target.runtimePort, '7878');
  assert.equal(target.runtimeDataDir, null);
});

test('prerelease package versions resolve to an exact release tag and isolated bin dir', () => {
  const target = resolveReleaseTarget({
    env: {},
    version: '0.2.0-alpha.1',
    homeDir: '/tmp/origin-home',
  });

  assert.equal(target.tag, 'v0.2.0-alpha.1');
  assert.equal(target.source, 'package-version');
  assert.equal(target.apiUrl, 'https://api.github.com/repos/7xuanlu/origin/releases/tags/v0.2.0-alpha.1');
  assert.equal(target.binDir, path.join('/tmp/origin-home', '.origin', 'releases', 'v0.2.0-alpha.1'));
  assert.equal(target.runtimePort, derivePrereleasePort('v0.2.0-alpha.1'));
  assert.equal(target.runtimeDataDir, path.join('/tmp/origin-home', 'Library', 'Application Support', 'origin', 'releases', 'v0.2.0-alpha.1'));
  assert.equal(target.usesIsolatedDefaults, true);
});

test('explicit release tags override the package version', () => {
  const target = resolveReleaseTarget({
    env: { ORIGIN_RELEASE_TAG: 'v0.3.0-beta.2' },
    version: '0.3.0-alpha.1',
    homeDir: '/tmp/origin-home',
  });

  assert.equal(target.tag, 'v0.3.0-beta.2');
  assert.equal(target.source, 'explicit-tag');
  assert.equal(target.apiUrl, 'https://api.github.com/repos/7xuanlu/origin/releases/tags/v0.3.0-beta.2');
  assert.equal(target.binDir, path.join('/tmp/origin-home', '.origin', 'releases', 'v0.3.0-beta.2'));
});

test('tag path sanitization strips unsafe path characters', () => {
  assert.equal(sanitizeTagForPath('v0.2.0-alpha/1'), 'v0.2.0-alpha_1');
});

test('package release tags are only derived for prerelease versions', () => {
  assert.equal(getPackageReleaseTag('0.1.0'), null);
  assert.equal(getPackageReleaseTag('0.2.0-beta.1'), 'v0.2.0-beta.1');
});

test('explicit runtime overrides win over prerelease defaults', () => {
  const runtime = resolveRuntimeTarget('v0.2.0-beta.1', {
    ORIGIN_PORT: '9123',
    ORIGIN_DATA_DIR: '/tmp/origin-beta',
  }, '/tmp/origin-home');

  assert.equal(runtime.runtimePort, '9123');
  assert.equal(runtime.runtimeDataDir, '/tmp/origin-beta');
  assert.equal(runtime.usesIsolatedDefaults, false);
});
