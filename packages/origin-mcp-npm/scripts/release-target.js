'use strict';

const os = require('os');
const path = require('path');
const { version: packageVersion } = require('../package.json');

const REPO = '7xuanlu/origin';

function derivePrereleasePort(tag) {
  let hash = 0;
  for (const char of tag) {
    hash = ((hash * 33) + char.charCodeAt(0)) >>> 0;
  }
  return String(8800 + (hash % 1000));
}

function sanitizeTagForPath(tag) {
  return tag.replace(/[^A-Za-z0-9._-]/g, '_');
}

function getExplicitReleaseTag(env = process.env) {
  const tag = env.ORIGIN_RELEASE_TAG || env.ORIGIN_TAG;
  return tag && tag.trim() ? tag.trim() : null;
}

function getPackageReleaseTag(version = packageVersion) {
  return version.includes('-') ? `v${version}` : null;
}

function buildReleaseApiUrl(repo, tag) {
  if (tag) {
    return `https://api.github.com/repos/${repo}/releases/tags/${encodeURIComponent(tag)}`;
  }
  return `https://api.github.com/repos/${repo}/releases/latest`;
}

function buildReleasePageUrl(repo, tag) {
  if (tag) {
    return `https://github.com/${repo}/releases/tag/${tag}`;
  }
  return `https://github.com/${repo}/releases`;
}

function resolveBinDir(tag, homeDir = os.homedir()) {
  if (tag) {
    return path.join(homeDir, '.origin', 'releases', sanitizeTagForPath(tag));
  }
  return path.join(homeDir, '.origin', 'bin');
}

function resolveRuntimeTarget(tag, env = process.env, homeDir = os.homedir()) {
  const runtimePort = env.ORIGIN_PORT || (tag ? derivePrereleasePort(tag) : '7878');
  const runtimeDataDir = env.ORIGIN_DATA_DIR
    || (tag ? path.join(homeDir, 'Library', 'Application Support', 'origin', 'releases', sanitizeTagForPath(tag)) : null);

  const runtimeEnv = {
    ...env,
    ORIGIN_PORT: runtimePort,
  };

  if (runtimeDataDir) {
    runtimeEnv.ORIGIN_DATA_DIR = runtimeDataDir;
  }

  return {
    runtimePort,
    runtimeDataDir,
    runtimeEnv,
    usesIsolatedDefaults: Boolean(tag && !env.ORIGIN_PORT && !env.ORIGIN_DATA_DIR),
  };
}

function resolveReleaseTarget({
  env = process.env,
  version = packageVersion,
  repo = REPO,
  homeDir = os.homedir(),
} = {}) {
  const explicitTag = getExplicitReleaseTag(env);
  const packageTag = getPackageReleaseTag(version);
  const tag = explicitTag || packageTag;
  const runtime = resolveRuntimeTarget(tag, env, homeDir);

  return {
    tag,
    repo,
    source: explicitTag ? 'explicit-tag' : packageTag ? 'package-version' : 'latest',
    apiUrl: buildReleaseApiUrl(repo, tag),
    pageUrl: buildReleasePageUrl(repo, tag),
    binDir: resolveBinDir(tag, homeDir),
    label: tag ? `release ${tag}` : 'latest stable release',
    ...runtime,
  };
}

module.exports = {
  REPO,
  buildReleaseApiUrl,
  buildReleasePageUrl,
  derivePrereleasePort,
  getExplicitReleaseTag,
  getPackageReleaseTag,
  resolveBinDir,
  resolveReleaseTarget,
  resolveRuntimeTarget,
  sanitizeTagForPath,
};
