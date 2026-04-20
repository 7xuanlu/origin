# Releasing Origin

## How release-please works

Merge conventional commits to `main` (e.g. `feat:`, `fix:`, `chore:`). The `release-please` workflow opens a "Release PR" automatically, bumping the version and updating `CHANGELOG.md`. Merge that PR to cut the release. Release-please then creates the git tag, which triggers the `release.yml` build workflow.

The `release-please-config.json` lists extra files that release-please also bumps on each release:

- `app/tauri.conf.json`
- `package.json`
- `packages/origin-mcp-npm/package.json`

The root `Cargo.toml` workspace version is the canonical source; `app/Cargo.toml` and the other crate `Cargo.toml` files are bumped by running `bash scripts/bump-version.sh` manually when needed outside of release-please.

## Manual override: bump-version.sh

If you need to cut a release without release-please (hotfix, first release, version correction):

```bash
bash scripts/bump-version.sh 0.2.0
```

This updates all version strings, regenerates `Cargo.lock`, and shows a diff summary. Review the diff, stage the files, and push. Then create and push the tag manually:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The `release.yml` workflow triggers on any `v*` tag push.

## Version consistency gate

The `release.yml` workflow validates that the pushed tag version matches all three files before building:

- `app/Cargo.toml`
- `app/tauri.conf.json`
- `package.json`

If any file is out of sync, the build fails immediately with instructions to run `bump-version.sh`.

## What the release workflow does

1. Validates version consistency (tag vs. Cargo.toml, tauri.conf.json, package.json).
2. Builds `origin-server` for `aarch64-apple-darwin`.
3. Fetches the `origin-mcp` binary from the separate repo and the `cloudflared` binary.
4. Builds the Tauri app with `CXXFLAGS="-std=c++17"` (required for llama.cpp on macOS 26.x).
5. Creates the GitHub release with DMG and standalone binaries attached.
6. After the release job succeeds, two parallel jobs run:
   - `publish-npm`: publishes `packages/origin-mcp-npm/` to npm.
   - `publish-crates`: publishes `origin-types` to crates.io if it changed since the previous tag.

## Cross-repo coordination with origin-mcp

`origin-mcp` lives in a separate repo (`~/Repos/origin-mcp`, MIT license). The release workflow pulls its binary directly from that repo via `cargo install --git`. There is no automated version pinning between the two repos. Steps for a coordinated release:

1. Release `origin-mcp` first (or ensure `main` is in a good state).
2. Tag and push `origin` as described above.
3. The workflow will install the latest `origin-mcp` from `main` of that repo.

If you need to pin to a specific `origin-mcp` commit or tag, edit the `cargo install --git` step in `release.yml`.

## Required secrets

Configure these in the repository settings (Settings, Secrets and variables, Actions):

| Secret | Purpose |
| ------ | ------- |
| `NPM_TOKEN` | Publish `origin-mcp` npm package. Create at npmjs.com under Access Tokens (Automation type). |
| `CARGO_REGISTRY_TOKEN` | Publish `origin-types` to crates.io. Create at crates.io under Account Settings, API Tokens. |
| `GITHUB_TOKEN` | Built-in. Used for GitHub release creation and release-please PR management. No setup needed. |
| `APPLE_CERTIFICATE` | Base64-encoded .p12 Developer ID Application certificate. Export from Keychain Access, then `base64 -i cert.p12`. |
| `APPLE_CERTIFICATE_PASSWORD` | Password used when exporting the .p12 file. |
| `APPLE_SIGNING_IDENTITY` | The identity string, e.g. `Developer ID Application: Your Name (TEAM_ID)`. Find via `security find-identity -v`. |
| `APPLE_ID` | Your Apple ID email. Used for notarization. |
| `APPLE_PASSWORD` | App-specific password for notarization. Generate at appleid.apple.com under Sign-In and Security. |
| `APPLE_TEAM_ID` | Your 10-character Apple Developer Team ID. Visible at developer.apple.com/account. |

## Code signing setup

Once you have an Apple Developer account:

1. Create a Developer ID Application certificate in Xcode (Settings, Accounts, Manage Certificates).
2. Export the certificate as .p12 from Keychain Access (right-click, Export).
3. Base64 encode it: `base64 -i DeveloperID.p12 | pbcopy`
4. Create an app-specific password at appleid.apple.com (Sign-In and Security, App-Specific Passwords).
5. Add all six APPLE_* secrets to the repo's GitHub Actions settings.
6. Update `app/tauri.conf.json`: change `signingIdentity` from `"-"` to your identity string.

The release workflow will automatically sign the .app, create a DMG, and submit for notarization. Users will no longer need `xattr -cr`.
