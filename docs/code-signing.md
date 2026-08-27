# Code signing and notarization

What ships today, and what it takes to make the installers trusted by the OS.

| Artifact | Today | What the OS shows | Fix |
| --- | --- | --- | --- |
| `Wenlan_<ver>_aarch64.dmg`, `Wenlan_aarch64.app.tar.gz` | ad-hoc signed (`"signingIdentity": "-"` in `app/tauri.conf.json`; `APPLE_SIGNING_IDENTITY` falls back to `-` in `release.yml`), not notarized | Gatekeeper: "Wenlan cannot be opened" (first-run gauntlet finding F11, `spctl --assess` → `rejected`) | Developer ID certificate + notarization, section 1 |
| `Wenlan_<ver>_x64-setup.exe` | unsigned | SmartScreen: "Windows protected your PC", unknown publisher | Authenticode certificate or Azure Trusted Signing, section 2 |
| `*.sig` next to the app bundles | signed with the Tauri updater key (`TAURI_SIGNING_PRIVATE_KEY`) | nothing; this is what the in-app updater verifies | already done; unrelated to Gatekeeper and SmartScreen |
| CLI tarballs, `wenlan-windows-x64.zip` | unsigned; `install.sh` strips quarantine and verifies `SHA256SUMS` | none for a terminal install | not needed for launch |

The Tauri updater signature protects updates after the first install. Gatekeeper and SmartScreen judge the first install, and only an OS-trusted certificate satisfies them.

## 1. macOS: Developer ID and notarization

The workflow is already wired: once the secrets below exist, the next release is signed and notarized with no code change. Without them the build stays ad-hoc and the verify step says so.

### One-time setup (about an hour, plus Apple's review of the membership)

1. **Join the Apple Developer Program** at <https://developer.apple.com/programs/enroll/> (USD 99 per year). After enrollment, note the ten-character **Team ID** under Membership details.
2. **Create a "Developer ID Application" certificate.** On any Mac: Keychain Access → Certificate Assistant → Request a Certificate From a Certificate Authority, save the request to disk. Then at <https://developer.apple.com/account/resources/certificates/add> pick **Developer ID Application**, upload the request, download the `.cer`, and double-click it so it lands in the login keychain. Check it:

   ```bash
   security find-identity -v -p codesigning
   # 1) ABC123…  "Developer ID Application: Your Name (TEAMID)"
   ```

   The quoted string is the signing identity you will paste into a secret.
3. **Export the certificate with its private key.** Keychain Access → My Certificates → right-click the Developer ID Application entry → Export → `.p12`, choose a password. Encode it:

   ```bash
   base64 -i DeveloperID.p12 | tr -d '\n' > DeveloperID.p12.b64
   ```

   The value must be a single line; the workflow strips stray whitespace from `APPLE_CERTIFICATE` but rejects any other line break.

4. **Create an app-specific password for notarization.** <https://account.apple.com/account/manage> → Sign-In and Security → App-Specific Passwords → generate one named `wenlan notarization`.
5. **Add six repository secrets** (Settings → Secrets and variables → Actions, or `gh secret set NAME < file`, e.g. `gh secret set APPLE_CERTIFICATE < DeveloperID.p12.b64`):

   | Secret | Value |
   | --- | --- |
   | `APPLE_CERTIFICATE` | contents of `DeveloperID.p12.b64` (one line) |
   | `APPLE_CERTIFICATE_PASSWORD` | the `.p12` export password |
   | `APPLE_SIGNING_IDENTITY` | `Developer ID Application: Your Name (TEAMID)` |
   | `APPLE_ID` | the Apple ID email of the developer account |
   | `APPLE_PASSWORD` | the app-specific password from step 4 |
   | `APPLE_TEAM_ID` | the Team ID from step 1 |

   These are the names the Tauri CLI reads (`APPLE_CERTIFICATE` and `APPLE_CERTIFICATE_PASSWORD` import the certificate into a temporary keychain on the runner; `APPLE_SIGNING_IDENTITY` overrides `signingIdentity` in `tauri.conf.json`; `APPLE_ID` with `APPLE_PASSWORD` and `APPLE_TEAM_ID` runs notarization after bundling).
6. **Cut a release** as usual (`RELEASING.md`). In the `app-bundle` job of `release.yml`, the step *Load Apple signing and notarization secrets* now runs, `pnpm tauri build` signs the app and every bundled binary (`wenlan`, `wenlan-server`, `wenlan-mcp`, `cloudflared`) with the hardened runtime and `app/Entitlements.plist`, submits the bundle to Apple, and staples the ticket. The step *Verify Apple signature and notarization* then fails the job unless `codesign --verify --deep --strict` passes and the signing authority is a Developer ID Application certificate, and — when the notarization secrets (`APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`) are set — `spctl --assess --type execute` and `xcrun stapler validate` on the `.app` also pass.
7. **Prove it from a user's seat.** Run the first-run gauntlet on the new tag (`gh workflow run first-run-gauntlet.yml --ref main -f release_tag=vX.Y.Z -f channels=macos-app`); the `dmg-gatekeeper` check flips from `rejected` to `accepted`. Then delete the "ad-hoc signed and not notarized" paragraph from the four READMEs (`README.md`, `README.es-ES.md`, `README.zh-Hans.md`, `README.zh-Hant.md`) and close F11 in the gauntlet report.

### Things to know

- Notarization adds 2–15 minutes to the `app-bundle` job; raise its `timeout-minutes` if Apple is slow on a given day.
- The entitlements `allow-jit` and `allow-unsigned-executable-memory` are ordinary hardened-runtime exceptions and pass notarization.
- A Developer ID certificate is valid for five years; apps notarized before it expires keep opening after.
- Keep the `.p12` and its password out of the repository and out of chat; the secrets are the only copy the workflow needs.
- Tauri notarizes and staples the `.app` only; the DMG is Developer ID signed but never notarized or stapled itself. If the gauntlet's `dmg-gatekeeper` check still says `rejected` after signing, add a step to the `app-bundle` job that runs `xcrun notarytool submit "$dmg" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APPLE_PASSWORD" --wait` followed by `xcrun stapler staple "$dmg"`.
- Developer ID signing without notarization still fails Gatekeeper, so set all six secrets together.

## 2. Windows: Authenticode or Azure Trusted Signing

Not wired yet; pick a provider first. Since 2023 code-signing private keys must live in hardware, so CI signs through a cloud signer rather than a `.pfx` file in a secret.

| Option | Cost and prerequisites | SmartScreen | How it plugs into the build |
| --- | --- | --- | --- |
| Azure Trusted Signing | about USD 10 per month; an Azure subscription and identity validation (individual or organization); not offered in every country | reputation from the first signed release | `trusted-signing-cli` as `bundle.windows.signCommand` in `app/tauri.conf.json`, with `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` secrets on the `app-bundle-windows` build step |
| OV certificate (Sectigo, DigiCert, SSL.com) | roughly USD 200–400 per year plus a cloud-signing service such as SSL.com eSigner or DigiCert KeyLocker | warning persists until download reputation builds | the vendor's CLI as `signCommand` |
| EV certificate | roughly USD 300–700 per year, stricter validation | immediate reputation | same as OV |

Prices are approximate; check the vendor page before buying.

When a provider is chosen: add a `scripts/windows-sign.ps1` that runs the vendor CLI on `%1` and exits 0 with a log line when its secrets are absent (so forks and unsigned builds still bundle), point `bundle.windows.signCommand` at it, pass the secrets to *Build Windows desktop app bundle* in `release.yml`, and add a verification step (`Get-AuthenticodeSignature` status must be `Valid`) mirroring the macOS one. The first-run gauntlet's `windows-nsis` leg then proves the installer from a clean Windows runner.

## 3. Related integrity checks already in place

- `SHA256SUMS` is published on every release by the `finalize-release` job, and `install.sh` verifies the tarball it downloads against it.
- `scripts/install-macos-app.sh` (the README one-liner for the app) verifies the DMG against the SHA-256 digest GitHub records for the asset.
- The Tauri updater verifies every update with the minisign public key in `app/tauri.conf.json`.
- Homebrew and npm packages carry their own registry checksums.
