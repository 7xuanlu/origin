# Code signing and notarization

What ships today, and what it takes to make the installers trusted by the OS.

| Artifact | Today | What the OS shows | Fix |
| --- | --- | --- | --- |
| `Wenlan_<ver>_aarch64.dmg`, `Wenlan_aarch64.app.tar.gz` | ad-hoc signed (`"signingIdentity": "-"` in `app/tauri.conf.json`; `APPLE_SIGNING_IDENTITY` falls back to `-` in `release.yml`), not notarized | Gatekeeper: "Wenlan cannot be opened" (first-run gauntlet finding F11, `spctl --assess` → `rejected`) | Developer ID certificate + notarization, section 1 |
| `Wenlan_<ver>_x64-setup.exe` | unsigned; the READMEs walk the user through the warning | SmartScreen: "Windows protected your PC", unknown publisher | sign it to establish a publisher identity, section 2; the warning itself fades only as that identity earns reputation |
| `*.sig` next to the app bundles | signed with the Tauri updater key (`TAURI_SIGNING_PRIVATE_KEY`) | nothing; this is what the in-app updater verifies | already done; unrelated to Gatekeeper and SmartScreen |
| CLI tarballs, `wenlan-windows-x64.zip` | unsigned; `install.sh` strips quarantine and verifies `SHA256SUMS` | none for a terminal install | not needed for launch |

The Tauri updater signature protects updates after the first install. Gatekeeper and SmartScreen judge the first install. A Developer ID certificate plus notarization satisfies Gatekeeper outright; SmartScreen also weighs how often an installer has been downloaded, so a certificate quiets it only as downloads accumulate.

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
6. **Cut a release** as usual (`RELEASING.md`). In the `app-bundle` job of `release.yml`, the step *Load Apple signing and notarization secrets* now runs, `pnpm tauri build` signs the app and every bundled binary (`wenlan`, `wenlan-server`, `wenlan-mcp`, `cloudflared`) with the hardened runtime and `app/Entitlements.plist`, submits the bundle to Apple, and staples the ticket. The step *Verify Apple signature and notarization* then fails the job unless `codesign --verify --deep --strict` passes and the signing authority is a Developer ID Application certificate, and — when the notarization secrets (`APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`) are set — `spctl --assess --type execute` and `xcrun stapler validate` pass on the `.app`, and `stapler validate` plus `spctl --assess --type open` pass on the `.dmg`. A separate step, *Notarize and staple the DMG*, runs between the build and that verification, because Tauri does not notarize the DMG itself.
7. **Prove it from a user's seat.** Run the first-run gauntlet on the new tag (`gh workflow run first-run-gauntlet.yml --ref main -f release_tag=vX.Y.Z -f channels=macos-app`). Its macOS leg stamps the quarantine attribute a browser would set, then requires `dmg-stapled`, `dmg-gatekeeper`, `dmg-developer-id`, `dmg-codesign-valid`, and `app-gatekeeper` (which must report `source=Notarized Developer ID`) to pass. Then delete the "ad-hoc signed and not notarized" paragraph from the four READMEs (`README.md`, `README.es-ES.md`, `README.zh-Hans.md`, `README.zh-Hant.md`) and close F11 in the gauntlet report.

### Things to know

- Notarization waits on Apple's queue twice, once for the app and once for the DMG, and the wait is unpredictable. A throwaway bundle submitted on 2026-08-29 sat `In Progress` for over 40 minutes while Apple's system status page reported the Notary Service healthy. The job allows 120 minutes for that reason; raise it rather than assume a hang.
- The entitlements `allow-jit` and `allow-unsigned-executable-memory` are ordinary hardened-runtime exceptions and pass notarization.
- A Developer ID certificate is valid for five years; apps notarized before it expires keep opening after.
- Keep the `.p12` and its password out of the repository and out of chat; the secrets are the only copy the workflow needs.
- Tauri notarizes and staples the `.app` but only signs the DMG ([`macos/dmg/mod.rs`](https://github.com/tauri-apps/tauri/blob/tauri-cli-v2.11.4/crates/tauri-bundler/src/bundle/macos/dmg/mod.rs) at the pinned CLI 2.11.4). The DMG is the outermost thing a browser hands a user, quarantine attribute and all, so `release.yml` submits it to the notary service and staples it in the step *Notarize and staple the DMG*. Do not remove that step on the assumption that notarizing the app inside is enough.
- Notarization removes Gatekeeper's block, not every dialog. A quarantined app still shows the one-time "downloaded from the Internet, are you sure you want to open it?" confirmation, which has an Open button. What goes away is the "cannot be opened because Apple cannot check it for malicious software" refusal and the trip to System Settings to approve it. The README one-liner installer strips the quarantine attribute, so that path shows nothing at all.
- Developer ID signing without notarization still fails Gatekeeper, so set all six secrets together.

## 2. Windows: Authenticode or Azure Artifact Signing

Not wired yet; pick a provider first. Since June 2023 code-signing private keys must live in hardware, so CI signs through a cloud signer rather than a `.pfx` file in a secret.

No certificate removes the warning on a first release. Microsoft withdrew the extended-validation certificate's instant SmartScreen bypass in 2024, so every tier now builds reputation the same way: release after release under one consistent publisher identity. Microsoft's own guidance is that paying the extended-validation premium purely to avoid SmartScreen is "no longer justified". Until reputation accumulates, the READMEs walk Windows users through "More info" and then "Run anyway", which is the same click path an unsigned installer needs.

Every price, country restriction, and SmartScreen behavior below is taken from [Microsoft's code signing options for Windows app developers](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options). Read that page before buying; these terms drift, and Microsoft's page wins wherever it disagrees with this table.

| Option | Cost and prerequisites | SmartScreen | How it plugs into the build |
| --- | --- | --- | --- |
| [Azure Artifact Signing](https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart) (called Trusted Signing until 2026) | about USD 10 per month, plus an Azure subscription and identity validation. Organizations are limited to a country list Microsoft publishes in the quickstart prerequisites; an individual developer must be in the United States or Canada. Validation takes a few business days. | warnings at first; reputation builds across releases | `trusted-signing-cli` as `bundle.windows.signCommand` in `app/tauri.conf.json`, with `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` secrets on the `app-bundle-windows` build step |
| [SignPath Foundation](https://signpath.org/) | free for qualifying open-source projects, signed through a managed pipeline. Confirm their eligibility rules and whose name the certificate carries before relying on it. | ordinary-certificate behavior | SignPath's own CI integration signs the artifact |
| Ordinary certificate (DigiCert, Sectigo, GlobalSign) | roughly USD 150–300 per year; the private key must sit on a hardware module or the vendor's cloud signer | same as Azure Artifact Signing | the vendor's CLI as `signCommand` |
| Extended-validation certificate | roughly USD 400 per year and up, stricter identity checks | same as an ordinary certificate since 2024 | same as an ordinary certificate |
| Microsoft Store, MSIX package | free; Microsoft re-signs the package | no warning at all | out of reach today: Tauri bundles NSIS and MSI, not MSIX, and a Store submission of an MSI or EXE installer must still be signed by the publisher |

When a provider is chosen: add a `scripts/windows-sign.ps1` that runs the vendor CLI on `%1` and exits 0 with a log line when its secrets are absent (so forks and unsigned builds still bundle), point `bundle.windows.signCommand` at it, and pass the secrets to *Build Windows desktop app bundle* in `release.yml`. Add a verification step (`Get-AuthenticodeSignature` status must be `Valid`) guarded the way the macOS one is guarded by `APPLE_SIGNING_CONFIGURED`, so it runs only when the signing secrets are present and an unsigned fork build still succeeds. The first-run gauntlet's `windows-nsis` leg then proves the installer from a clean Windows runner. Then finish the paperwork the way section 1 does for macOS: update the summary table at the top of this page, and replace the "not signed yet" sentence in `README.md`, `README.es-ES.md`, `README.zh-Hans.md`, and `README.zh-Hant.md`. <!-- drift-ok -->

## 3. Related integrity checks already in place

- `SHA256SUMS` is published on every release by the `finalize-release` job, and `install.sh` verifies the tarball it downloads against it.
- `scripts/install-macos-app.sh` (the README one-liner for the app) verifies the DMG against the SHA-256 digest GitHub records for the asset.
- The Tauri updater verifies every update with the minisign public key in `app/tauri.conf.json`.
- Homebrew and npm packages carry their own registry checksums.
