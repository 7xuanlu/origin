# Code signing and notarization

What ships today, and what it takes to make the installers trusted by the OS.

| Artifact | Today | What the OS shows | Fix |
| --- | --- | --- | --- |
| `Wenlan_<ver>_aarch64.dmg`, `Wenlan_aarch64.app.tar.gz` | ad-hoc signed (`"signingIdentity": "-"` in `app/tauri.conf.json`; `APPLE_SIGNING_IDENTITY` falls back to `-` in `release.yml`), not notarized | Gatekeeper: "Wenlan cannot be opened" (first-run gauntlet finding F11, `spctl --assess` → `rejected`) | Developer ID certificate + notarization, section 1 |
| `Wenlan_<ver>_x64-setup.exe` | unsigned; the READMEs walk the user through the warning | SmartScreen: "Windows protected your PC", unknown publisher | a free SignPath Foundation certificate, section 2; the warning itself fades only as that publisher identity earns reputation |
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

   These are the names the Tauri CLI reads (`APPLE_CERTIFICATE` and `APPLE_CERTIFICATE_PASSWORD` import the certificate into a temporary keychain on the runner; `APPLE_SIGNING_IDENTITY` overrides `signingIdentity` in `tauri.conf.json`; `APPLE_ID` with `APPLE_PASSWORD` and `APPLE_TEAM_ID` reach only the steps that talk to the notary service, never `pnpm tauri build` — see the outermost-container note below).
6. **Cut a release** as usual (`RELEASING.md`). In the `app-bundle` job of `release.yml` the steps run in Apple's prescribed order:
   1. *Load Apple signing and notarization secrets* exports the three signing secrets and deliberately withholds the three notarization ones.
   2. `pnpm tauri build` signs the app and every bundled binary (`wenlan`, `wenlan-server`, `wenlan-mcp`, `cloudflared`) with the hardened runtime and `app/Entitlements.plist`, signs the disk image, and — because it cannot see the notarization secrets — submits nothing.
   3. *Notarize and staple the DMG* makes the single submission and staples the ticket to the disk image.
   4. *Staple the app and rebuild the updater archive* staples the same ticket to the `.app` (a lookup, not a second submission), rebuilds `Wenlan_aarch64.app.tar.gz` from the stapled app, and re-signs it with `tauri signer sign`.
   5. *Verify Apple signature and notarization* fails the job unless `codesign --verify --deep --strict` passes and the signing authority is a Developer ID Application certificate, and — when the notarization secrets are set — `spctl --assess --type execute` and `xcrun stapler validate` pass on the `.app`, `stapler validate` plus `spctl --assess --type open` pass on the `.dmg`, and the app mounted from inside the disk image assesses as `source=Notarized Developer ID`.
7. **Prove it from a user's seat.** Run the first-run gauntlet on the new tag (`gh workflow run first-run-gauntlet.yml --ref main -f release_tag=vX.Y.Z -f channels=macos-app`). Its macOS leg stamps the quarantine attribute a browser would set, then requires `dmg-stapled`, `dmg-gatekeeper`, `dmg-developer-id`, `dmg-codesign-valid`, and `app-gatekeeper` (which must report `source=Notarized Developer ID`) to pass. Then delete the "ad-hoc signed and not notarized" paragraph from the four READMEs (`README.md`, `README.es-ES.md`, `README.zh-Hans.md`, `README.zh-Hant.md`) and close F11 in the gauntlet report.

### Things to know

- Notarization waits on Apple's queue once, and the wait is unpredictable. Two throwaway bundles submitted at 07:04 and 07:21 UTC on 2026-08-29 were still `In Progress` eleven hours later, while Apple's system status page reported the Notary Service healthy; the developer forums describe this for first-time accounts, whose early submissions get held for review. The step waits 90 minutes inside a job that allows 210, and the workspace compiles from scratch before it, so the two numbers move together — raise both rather than assume a hang. If it does time out, the step prints the submission id — wait on that id with `xcrun notarytool wait <id>` instead of rerunning the job, because a rerun submits the same disk image again and queues behind the first.
- The entitlements `allow-jit` and `allow-unsigned-executable-memory` are ordinary hardened-runtime exceptions and pass notarization.
- A Developer ID certificate is valid for five years; apps notarized before it expires keep opening after.
- Keep the `.p12` and its password out of the repository and out of chat; the secrets are the only copy the workflow needs.
- **Only the outermost container is notarized.** One submission of the disk image also covers the app inside it. [Customizing the notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow): *"if you submit a disk image that contains a signed installer package with an app bundle inside, the notarization service generates tickets for the disk image, installer package, and app bundle."* The disk image is what a browser hands a user, quarantine attribute and all, so that is the container we submit. tauri-bundler would submit the `.app` on its own as soon as it sees `APPLE_ID`, `APPLE_PASSWORD` and `APPLE_TEAM_ID` in the environment ([`macos/app.rs`](https://github.com/tauri-apps/tauri/blob/tauri-cli-v2.11.4/crates/tauri-bundler/src/bundle/macos/app.rs) at the pinned CLI 2.11.4), so *Load Apple signing and notarization secrets* keeps those three out of the job environment and hands them only to the steps that need them. Without them the bundler logs `skipping app notarization` and carries on. Do not add them back to the build step's environment: that silently restores a second submission and a second wait.
- **The app copied out of the disk image has no ticket of its own.** One submission covers it — a notarization ticket is a list of code directory hashes, and the app inside the image is in that list — but the copy inside the image was made before any ticket existed, so nothing is embedded in it. Gatekeeper resolves it against Apple on first launch, which needs a moment of network. Everything a person keeps is stapled: the disk image they downloaded, and the app the updater installs.
- **The updater archive is rebuilt, not the one Tauri produced.** A `.tar.gz` cannot hold a stapled ticket. Apple, on the equivalent ZIP case: *"While you can notarize a ZIP archive, you can't staple to it directly. Instead, run `stapler` against each item that you added to the archive. Then create a new ZIP file containing the stapled items for distribution."* Tauri writes `Wenlan_aarch64.app.tar.gz` before any ticket exists, so *Staple the app and rebuild the updater archive* staples the app, repacks it, and re-signs with `tauri signer sign`, which calls the same `sign_file` helper the bundler uses. The step unpacks the result and fails the job unless the app inside verifies and validates as stapled — a broken updater archive is the worst thing this job could ship.
- Notarization removes Gatekeeper's block, not every dialog. A quarantined app still shows the one-time "downloaded from the Internet, are you sure you want to open it?" confirmation, which has an Open button. What goes away is the "cannot be opened because Apple cannot check it for malicious software" refusal and the trip to System Settings to approve it. The README one-liner installer strips the quarantine attribute, so that path shows nothing at all.
- Developer ID signing without notarization still fails Gatekeeper, so set all six secrets together.

## 2. Windows: SignPath Foundation

The provider is chosen: [SignPath Foundation](https://signpath.org/), which signs qualifying open-source projects for free. Nothing is wired yet, because the workflow needs an organization id, a project slug, a signing-policy slug, and an API token that only exist once the Foundation accepts the application. Since June 2023 code-signing private keys must live in hardware, so CI signs through a cloud signer rather than a `.pfx` file in a secret; SignPath is that cloud signer.

Two consequences of choosing the Foundation tier, both irreversible in practice and worth knowing before applying:

- **The certificate is issued to SignPath Foundation, not to Wenlan.** Their terms say "The code signing certificate is issued to *SignPath Foundation*. This means that *SignPath Foundation* is the publisher of the OSS project." Windows will name SignPath Foundation as the verified publisher. The same certificate already signs several hundred projects, so it carries reputation a new certificate would not.
- **A human probably approves every release.** The Foundation's terms require that "each signing request must be approved by a team member", but their platform documentation lists the approval process as *available* for open-source signing rather than *required*, so the software itself permits an unattended policy. Ask during onboarding which applies. If approval stays on, the Windows job blocks until someone clicks approve and a tag release is no longer unattended.

### Before applying

1. **Turn on multi-factor authentication** for the GitHub account. Their terms require it: "All team members must use multi-factor authentication for both SignPath and source code repository access (e.g. GitHub)."
2. **Publish the code signing policy.** Done: the READMEs carry a `Code signing policy` section naming the Author, Reviewer and Approver roles and the required credit line. The Foundation checks the project's home page for it.
3. **Apply** at <https://signpath.org/apply>. Expect the repository URL, the license name, the release download URL, and a project description. Reported turnaround is a few days to a few weeks; no service level is published.

### After acceptance

Signing is a submission from inside the workflow, not a local `signtool` call: [`signpath/github-action-submit-signing-request`](https://github.com/SignPath/github-action-submit-signing-request) uploads the built installer, waits for the approval, and downloads the signed file. The pieces to add to `app-bundle-windows` in `release.yml`, in order:

1. Upload `*-setup.exe` as a GitHub artifact and submit it, guarded on a `SIGNPATH_CONFIGURED` flag the way the macOS steps are guarded by `APPLE_SIGNING_CONFIGURED`, so a fork without the secret still builds. Raise the action's `wait-for-completion-timeout-in-seconds` past its 600-second default, because it is waiting for a person.
2. **Re-sign the updater artifact.** The `.sig` beside the installer is computed over the unsigned bytes, so Authenticode signing invalidates it. Delete it and re-run `pnpm tauri signer sign`, the same repair the macOS path makes after stapling.
3. Take the SHA-256 checksums from the signed installer, not the unsigned one.
4. Verify with `Get-AuthenticodeSignature`, status must be `Valid`, guarded the same way.
5. Prove it on a clean machine through the first-run gauntlet's `windows-nsis` leg.

Two limits to plan around. SignPath has no artifact type for an NSIS container, so it signs the installer as an ordinary PE file and leaves the executables inside it unsigned; deep-signing them means splitting `pnpm tauri build` into separate compile and bundle phases, which is only worth doing if Smart App Control turns out to reject the inner binaries. And the bundled upstream binaries — cloudflared, the ONNX Runtime and Vulkan libraries — must stay out of the submission, because the terms cover a project signing its own output.

### Why not the alternatives

No certificate removes the warning on a first release. Microsoft withdrew the extended-validation certificate's instant SmartScreen bypass in 2024, so every tier now builds reputation the same way: release after release under one consistent publisher identity. Microsoft's own guidance is that paying the extended-validation premium purely to avoid SmartScreen is "no longer justified". Until reputation accumulates, the READMEs walk Windows users through "More info" and then "Run anyway", which is the same click path an unsigned installer needs.

Every price, country restriction, and SmartScreen behavior below is taken from [Microsoft's code signing options for Windows app developers](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options). Read that page before buying; these terms drift, and Microsoft's page wins wherever it disagrees with this table.

| Option | Cost and prerequisites | SmartScreen | How it plugs into the build |
| --- | --- | --- | --- |
| [Azure Artifact Signing](https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart) (called Trusted Signing until 2026) | about USD 10 per month, plus an Azure subscription and identity validation. Organizations are limited to a country list Microsoft publishes in the quickstart prerequisites; an individual developer must be in the United States or Canada. Validation takes a few business days. | warnings at first; reputation builds across releases | `trusted-signing-cli` as `bundle.windows.signCommand` in `app/tauri.windows.conf.json`, with `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` secrets on the `app-bundle-windows` build step |
| [SignPath Foundation](https://signpath.org/) — **chosen** | free. Requires an OSI-approved license with no commercial dual-licensing, no proprietary component, an actively maintained and already released project, multi-factor authentication on every maintainer account, a published code signing policy, and a build on a trusted build system (GitHub-hosted runners qualify). A maintainer approves every signing request. | ordinary-certificate behavior, on a certificate that already carries reputation from several hundred projects | their GitHub Action submits the built installer and returns it signed |
| Ordinary certificate (DigiCert, Sectigo, GlobalSign) | roughly USD 150–300 per year; the private key must sit on a hardware module or the vendor's cloud signer | same as Azure Artifact Signing | the vendor's CLI as `signCommand` |
| Extended-validation certificate | roughly USD 400 per year and up, stricter identity checks | same as an ordinary certificate since 2024 | same as an ordinary certificate |
| Microsoft Store, MSIX package | free; Microsoft re-signs the package | no warning at all | out of reach today: Tauri bundles NSIS and MSI, not MSIX, and a Store submission of an MSI or EXE installer must still be signed by the publisher |

Azure Artifact Signing is the fallback if the Foundation declines. It costs money rather than an approval click, and an individual developer must be resident in the United States or Canada, so check that first. A paid certificate from a commercial authority works anywhere but needs a hardware module or the vendor's cloud signer, and would be driven by a small wrapper script wired to `bundle.windows.signCommand` in `app/tauri.windows.conf.json` — that overlay file, not `app/tauri.conf.json`, is where the Windows bundle settings live.

Once the first signed installer ships, finish the paperwork the way section 1 does for macOS: update the summary table at the top of this page, and replace the "not signed yet" sentence in `README.md`, `README.es-ES.md`, `README.zh-Hans.md`, and `README.zh-Hant.md`. <!-- drift-ok -->

## 3. Related integrity checks already in place

- `SHA256SUMS` is published on every release by the `finalize-release` job, and `install.sh` verifies the tarball it downloads against it.
- `scripts/install-macos-app.sh` (the README one-liner for the app) verifies the DMG against the SHA-256 digest GitHub records for the asset.
- The Tauri updater verifies every update with the minisign public key in `app/tauri.conf.json`.
- Homebrew and npm packages carry their own registry checksums.
