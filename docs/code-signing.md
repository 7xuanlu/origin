# Code signing and notarization

What ships today, and what it takes to make the installers trusted by the OS.

| Artifact | Today | What the OS shows | Fix |
| --- | --- | --- | --- |
| `Wenlan_<ver>_aarch64.dmg`, `Wenlan_aarch64.app.tar.gz` | Developer ID signed and notarized, ticket stapled, from v0.17.4 | None. The first-run gauntlet on v0.17.4 reports `spctl --assess` as `accepted`, `source=Notarized Developer ID` | Done |
| `Wenlan_<ver>_x64-setup.exe` | unsigned; the READMEs walk the user through the warning | SmartScreen: "Windows protected your PC", unknown publisher | a free SignPath Foundation certificate, section 2; the warning itself fades only as that publisher identity earns reputation |
| `*.sig` next to the app bundles | signed with the Tauri updater key (`TAURI_SIGNING_PRIVATE_KEY`) | nothing; this is what the in-app updater verifies | already done; unrelated to Gatekeeper and SmartScreen |
| CLI tarballs, `wenlan-windows-x64.zip` | unsigned; `install.sh` strips quarantine and verifies `SHA256SUMS` | none for a terminal install | not needed for launch |

The Tauri updater signature protects updates after the first install. Gatekeeper and SmartScreen judge the first install. A Developer ID certificate plus notarization satisfies Gatekeeper outright; SmartScreen also weighs how often an installer has been downloaded, so a certificate quiets it only as downloads accumulate.

## 1. macOS: Developer ID and notarization

The workflow is already wired, and the secrets are in place: releases from v0.17.4 are signed and notarized. If the secrets are ever removed, the build falls back to ad-hoc signing and the verify step says so.

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
7. **Prove it from a user's seat.** Run the first-run gauntlet on the new tag (`gh workflow run first-run-gauntlet.yml --ref main -f release_tag=vX.Y.Z -f channels=macos-app`). Its macOS leg stamps the quarantine attribute a browser would set, then requires `dmg-stapled`, `dmg-gatekeeper`, `dmg-developer-id`, `dmg-codesign-valid`, and `app-gatekeeper` (which must report `source=Notarized Developer ID`) to pass. Then delete the "ad-hoc signed and not notarized" paragraph from the four READMEs (`README.md`, `README.es-ES.md`, `README.zh-Hans.md`, `README.zh-Hant.md`) and close F11 in the gauntlet report. Done for v0.17.4: run 33287491599 passed all five checks, with both Gatekeeper legs reporting `accepted`, `source=Notarized Developer ID`, `origin=Developer ID Application: Qi-Xuan Lu (TDFFZXRF3D)`.

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

Two consequences of choosing the Foundation tier, both worth knowing before applying. They hold for as long as Wenlan signs through the Foundation; leaving for a paid certificate later is possible, at the cost of restarting publisher reputation under a new identity.

- **The certificate is issued to SignPath Foundation, not to Wenlan.** Their terms say "The code signing certificate is issued to *SignPath Foundation*. This means that *SignPath Foundation* is the publisher of the OSS project." Windows will name SignPath Foundation as the verified publisher. Do not count on that shared identity to suppress the first warning: SmartScreen weighs the file's hash as well as the certificate, and [Microsoft says](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation) "Even when signed, a newly created binary could still show a SmartScreen warning until its hash or publisher certificate accumulates sufficient evidence of positive reputation."
- **A human probably approves every release.** The Foundation's terms require that "each signing request must be approved by a team member", but their platform documentation lists the approval process as *available* for open-source signing rather than *required*, so the software itself permits an unattended policy. Ask during onboarding which applies. If approval stays on, the Windows job blocks until someone clicks approve and a tag release is no longer unattended.

### Before applying

1. **Turn on multi-factor authentication** for the GitHub account. Their terms require it: "All team members must use multi-factor authentication for both SignPath and source code repository access (e.g. GitHub)."
2. **Publish the code signing policy.** Done: the READMEs carry a `Code signing policy` section naming the Author, Reviewer and Approver roles and the required credit line. The Foundation checks the project's home page for it.
3. **Apply** at <https://signpath.org/apply>. Expect the repository URL, the license name, the release download URL, and a project description. The application page states no turnaround time, so do not plan a release date around an acceptance date.

   The license answer is "Apache-2.0 and AGPL-3.0-only", not Apache-2.0 alone: `package.json` and `app/Cargo.toml` declare AGPL for the desktop app. Both are OSI-approved and neither part is sold, so the split meets "an OSI-approved Open Source license without commercial dual-licensing"; answering with one license would simply be wrong, and a reviewer who reads the manifests will ask.

### After acceptance

**Ask them one question before building any of this.** SignPath also offers a Windows Key Storage Provider, which would let the build sign the installer in place through Tauri's own `bundle.windows.signCommand` hook. That path skips steps 2, 3 and 4 below outright: nothing is uploaded and re-downloaded, so the updater signature and the checksums are computed once, over the already-signed bytes. The catch is that it produces no GitHub workflow artifact, and origin verification, which the free tier requires, appears to need one. SignPath has not ruled the combination out in writing, so ask; a yes removes most of the work on this page.

Signing is a submission from inside the workflow, not a local `signtool` call: [`signpath/github-action-submit-signing-request`](https://github.com/SignPath/github-action-submit-signing-request) uploads the built installer, waits for the approval, and downloads the signed file. The pieces to add to `app-bundle-windows` in `release.yml`, in order:

1. Upload `*-setup.exe` as a GitHub artifact and submit it, guarded on a `SIGNPATH_CONFIGURED` flag the way the macOS steps are guarded by `APPLE_SIGNING_CONFIGURED`, so a fork without the secret still builds. Raise the action's `wait-for-completion-timeout-in-seconds` past its 600-second default, because it is waiting for a person.
2. **Set `output-artifact-directory`, and overwrite the built installer with what comes back.** The input is optional and its absence is silent: the action's manifest describes it as "Path where the signed artifact will be saved. If not specified, the task will not download the signed artifact from SignPath". *Stage app bundle assets and checksums* then finds the original unsigned `*-setup.exe` under `target/x86_64-pc-windows-msvc/release/bundle/nsis` exactly as it does today, and a green run publishes the unsigned file. Write the returned file back over that same path — note `skip-decompress` defaults to `false`, so the action extracts the returned archive into the directory — and fail the job if the path is not newer than the submission.
3. **Re-sign the updater artifact, with the key in scope.** The `.sig` beside the installer is computed over the unsigned bytes, so Authenticode signing invalidates it. Delete it and re-run `pnpm tauri signer sign`, the same repair the macOS path makes after stapling. `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` are `env:` on *Build Windows desktop app bundle* alone, so the new step needs its own copy of both or it cannot sign at all. Regenerating the sidecar `.sig` does not disturb the installer's Authenticode signature, but nothing else may touch those bytes after signing.
4. Take the SHA-256 checksums from the signed installer, not the unsigned one.
5. Verify with `Get-AuthenticodeSignature`: status must be `Valid` **and** the subject must be SignPath Foundation, so a silently skipped download cannot pass as a signed build. Guard it the same way.
6. Prove it on a clean machine through the first-run gauntlet's `windows-nsis` leg.

Two limits to plan around.

**Signing the installer does not cover what it installs.** SignPath has no artifact type for an NSIS container, so it signs the installer as an ordinary PE file and leaves `Wenlan.exe`, the daemon, the CLI, the MCP server and the bundled libraries unsigned on disk. That is enough for the download warning and not enough for Smart App Control, which [Microsoft says](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation) "will block execution of unsigned files unless the file has a positive reputation" and whose "signature checks apply to all executable files, not just those downloaded from the Internet". Treat outer-only signing as a first step, not as finished Windows signing: test the installed binaries on a Windows 11 machine with Smart App Control in enforcement mode before claiming otherwise. Deep-signing them means splitting `pnpm tauri build` into separate compile and bundle phases so the executables can be signed before NSIS packs them, then signing the installer in a second round.

**Bundled upstream binaries stay in the installer.** Exclude cloudflared, the ONNX Runtime and the Vulkan loader from the signing directives, not from the package: the terms allow it — "You may include unsigned binaries of upstream OSS projects, e.g. DLL files, in your signed packages, e.g. MSI installers" — and stripping them would break the app at runtime. SignPath does reserve the right to require fully signed packages later.

### Why not the alternatives

A certificate does not guarantee that a first release avoids a warning. Microsoft withdrew the extended-validation certificate's instant SmartScreen bypass in 2024, so every tier now builds reputation the same way: release after release under one consistent publisher identity. Microsoft's own guidance is that paying the extended-validation premium purely to avoid SmartScreen is "no longer justified". Until reputation accumulates, the READMEs walk Windows users through "More info" and then "Run anyway", which is the same click path an unsigned installer needs.

Prices, country restrictions, and SmartScreen behavior below come from [Microsoft's code signing options for Windows app developers](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options), which lists SignPath Foundation itself as the open-source option — a second reason to try the Foundation before paying anyone. That page does not say who qualifies for it, though: it sends the reader to signpath.org, so the eligibility rules in the Foundation row come from the Foundation's own terms rather than from Microsoft. Read Microsoft's page before buying; these terms drift, and it wins wherever it disagrees with this table.

| Option | Cost and prerequisites | SmartScreen | How it plugs into the build |
| --- | --- | --- | --- |
| [Azure Artifact Signing](https://learn.microsoft.com/en-us/azure/artifact-signing/quickstart) (called Trusted Signing until 2026) | about USD 10 per month, plus an Azure subscription and identity validation. An individual developer must live in the United States or Canada. For organizations Microsoft's own two pages disagree: the comparison page says the United States, Canada, the EU and the UK, while the quickstart also lists Australia, New Zealand, Japan, South Korea, Singapore, Switzerland, Norway and Israel. Trust the quickstart, which is the page the service's own prerequisites live on. Identity validation takes 1 to 20 business days, not the few days the price suggests. | warnings at first; reputation builds across releases | `trusted-signing-cli` as `bundle.windows.signCommand` in `app/tauri.windows.conf.json`, with `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` secrets on the `app-bundle-windows` build step |
| [SignPath Foundation](https://signpath.org/) — **chosen** | free. Requires an OSI-approved license with no commercial dual-licensing, no proprietary component, an actively maintained and already released project, multi-factor authentication on every maintainer account, a published code signing policy, and a build on a trusted build system (GitHub-hosted runners qualify). A maintainer approves every signing request. | same as any ordinary certificate: warnings until reputation accumulates | their GitHub Action submits the built installer and returns it signed |
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
