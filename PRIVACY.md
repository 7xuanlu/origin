# Privacy Policy

Wenlan is a local-first personal memory system. This policy covers everything the project ships: the desktop app, the daemon, the CLI, the MCP server, and the Claude Code and Codex plugins.

## What data Wenlan stores

Only what you explicitly capture: decisions, lessons, observations, project context, and wiki pages synthesized from those memories. Wenlan does not monitor or scrape.

One thing is automatic, and you switch it on yourself. When you connect a folder as a source, Wenlan watches it and reads changed files without asking again for each one -- it syncs that folder once at startup and then whenever a file in it changes. Choosing the folder is the consent; each individual file is not a separate prompt.

## Where data is stored

All of your data stays on your machine:

- `~/.wenlan/pages/` -- wiki pages (Markdown)
- `~/.wenlan/sessions/` -- session logs (Markdown)
- `~/.wenlan/sources/` -- uploaded source documents
- `~/.wenlan/spaces.toml` -- Space mapping config
- `~/.wenlan/db/` -- symlink to the platform data directory (macOS: `~/Library/Application Support/wenlan/`; Linux: `~/.local/share/wenlan/`; Windows: `%LOCALAPPDATA%\wenlan\`), with the libSQL database at `<data dir>/memorydb/`
- `~/.wenlan/bin/` -- installed binaries

The daemon listens on `127.0.0.1:7878`, which is your own machine only. Nothing you capture -- no memory, page, source, or search -- is sent anywhere unless you turn on one of the features in "Sending your content somewhere" below.

## When Wenlan reaches the network

Wenlan is not fully offline. Three things happen on their own, with no setting to stop them, and each tells the other end your IP address. The first of the three is a download rather than a single request: it fetches the model file and its tokenizer and configuration files, so expect several requests, not one.

Separately, displaying a note that contains remote images reaches those image hosts. That is described under "Images in your notes reach their host" below.

| What | When | Where | What it sends |
| --- | --- | --- | --- |
| Search model download | Once, the first time the daemon starts | `https://huggingface.co`, repository `Qdrant/bge-base-en-v1.5-onnx-Q` | Downloads about 210 MB across several files. Sends no memory content. One caveat: the download library reads a Hugging Face access token from that tool's standard cache directory if you already have one saved on this machine, and attaches it, which identifies your Hugging Face account to them. This model is what makes local search work; without it the daemon cannot start. |
| Desktop app update check | 3 seconds after every launch of the app | `https://github.com/7xuanlu/wenlan/releases/latest/download/latest.json` | Nothing but the request itself, identified as `tauri-plugin-updater`. If a newer version exists the app asks you; it never installs without your click. |
| MCP server version check | Every time an AI client starts `wenlan-mcp`, at most once a day | `https://github.com/7xuanlu/wenlan/releases/latest` | Nothing but the request itself, identified as `wenlan-mcp/<version>`. It reads only where GitHub redirects, to compare version numbers. |

To avoid the model download, copy an existing model cache into the daemon's cache directory before first start. To avoid the two version checks, block `github.com` for those processes at your firewall; both fail quietly and Wenlan keeps working.

### Images in your notes reach their host

These requests deserve their own heading, because they are the ones that can reach a server nobody at Wenlan chose, and because a note can produce many of them.

Wenlan renders Markdown, and a Markdown image may point at any address. If a note contains `![something](https://example.com/picture.png)`, **displaying that note tries to load the picture from `example.com`**. Every remote image in a note is a separate load, so a note with ten of them reaches up to ten hosts. How many network requests that actually becomes depends on your browser cache and on any redirects those addresses follow. The desktop app sets no content security policy, so nothing blocks them.

A load that does reach the host tells it your IP address, the time, and the exact image address, which came out of your note. A uniquely generated image address therefore works as a read receipt on anyone who opens the note.

It can tell the host more than that. The reading view uses an ordinary image tag with no restrictions set, so the browser applies its defaults: it sends any cookies that host has already set in this app, and a `Referer` header naming the app's own address. The editor sets `referrerpolicy="no-referrer"`, which drops the referrer but does not stop the cookies. Wenlan adds nothing to the address, but anything already in it, including a query string, is sent as written.

Reading and editing differ. The reading view builds every image as soon as it renders the note. The editor builds images only for the part of the note on screen, and marks them to load lazily, so an image far below your cursor is not fetched until you scroll to it.

This matters most for content you did not write: a page imported from elsewhere, or a document someone sent you. There is no switch to turn it off.

Nor is there a way to point a note at a picture in your own notes folder. A relative address such as `![x](pictures/photo.png)` is passed through unchanged by the reading view, which resolves it against the app rather than your folder, so your file does not appear; the editor accepts only `http:` and `https:` addresses and skips it. The app can display local files through the mechanism it uses for avatars, which is limited to the folders it is configured for and reaches no network -- on Windows that mechanism produces an address beginning `http://asset.localhost/`, which looks remote but is not. Outside those folders, an image you can see in a note came over the network.

If a note worries you, remove the image link from it.

## Sending your content somewhere

These are off until you turn them on. Each one is the only way your captured content leaves your machine.

- **A cloud AI provider (bring your own key).** If you save an API key and pick that provider for enrichment, Wenlan sends the text of the memory or document being processed, plus its prompt, to that provider. Anthropic goes to `https://api.anthropic.com/v1/messages`. Any other provider goes to the endpoint you entered -- the app offers presets for OpenAI, Google, Groq, OpenRouter, Mistral, DeepSeek and xAI, and an endpoint on your own machine such as Ollama stays local. What that provider does with the text is governed by its own terms. Read them rather than assuming: several of these companies say their consumer privacy policy does not cover text submitted through their API, and point to a separate business or API agreement instead. Your key is stored in the local config and sent only to that provider. Turn it off with `wenlan enrichment disable`, or by clearing the key.
- **Testing a provider.** The "test endpoint" button sends one fixed sentence, `Say 'hello' and nothing else.`, and the model-list button asks the provider what models it offers. Neither sends anything you captured.
- **Remote Access (experimental, desktop app only).** When you turn it on, the app runs `cloudflared` to open a tunnel to the local MCP server so Claude.ai and ChatGPT can reach a stable address, and registers that address with Wenlan's relay at `https://origin-relay.originmemory.workers.dev/register`. The registration sends the tunnel address and a random 16-byte identifier that Wenlan generates once and keeps on disk; that identifier is used as both the account name and the shared secret. While it is on, the app checks the tunnel's health every 30 seconds. **The address has no login: anyone who has it can read and write your entire memory** until you turn Remote Access off. Nothing is sent while it is off.
- **On-device model download.** If you run `wenlan models install` or start the download from Settings, a Qwen model is fetched from `https://huggingface.co`. This is separate from the search model above and does not happen on its own. Once installed, enrichment runs on your machine and nothing leaves it.
- **Better search ranking.** If you turn on the reranker, its weights are downloaded from `https://huggingface.co` the next time the daemon starts, between roughly 146 MB and 1.1 GB depending on which one you choose. It is off unless you set it.

## Telemetry

None. Wenlan collects no usage analytics, no crash reports, and no diagnostics. There is no analytics library of any kind in the code, and the app's fonts are bundled rather than fetched. Opening a window contacts nothing by itself -- but see "Images in your notes reach their host" above, because the note you open can.

## The other companies involved

Wenlan reaches these services, so their policies govern what they do with the request. We have no agreement with any of them and receive nothing back from them about you.

| Service | Why Wenlan reaches it | Their policy |
| --- | --- | --- |
| GitHub | The two version checks, and downloading a release | [GitHub Privacy Statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement) |
| Hugging Face | Downloading the search model, and any optional model you install | [Hugging Face Privacy Policy](https://huggingface.co/privacy) |
| Cloudflare | Only if you turn on Remote Access, which runs `cloudflared` to open the tunnel | [Cloudflare Privacy Policy](https://www.cloudflare.com/privacypolicy/) |
| The host of a remote image in one of your notes | Displaying a note whose Markdown points at a remote image | Whoever runs that host. Wenlan cannot know who that is |

### The cloud AI providers Wenlan offers

None of these is contacted unless you save a key and choose that provider for enrichment. Wenlan ships a preset for each one, so each one's policy is named here.

| Provider | Their policy |
| --- | --- |
| Anthropic | [Anthropic Privacy Policy](https://www.anthropic.com/legal/privacy) |
| OpenAI | Published per region: [rest of world](https://openai.com/policies/row-privacy-policy/), [United States](https://openai.com/policies/us-privacy-policy/), [Europe](https://openai.com/policies/eu-privacy-policy/) |
| Google, for Gemini | [Google Privacy Policy](https://policies.google.com/privacy) |
| Groq | [Groq Privacy Policy](https://groq.com/privacy-policy) |
| OpenRouter | [OpenRouter Privacy Policy](https://openrouter.ai/privacy) |
| Mistral | [Mistral Privacy Policy](https://legal.mistral.ai/terms/privacy-policy) |
| DeepSeek | [DeepSeek Privacy Policy](https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html) |
| xAI | [xAI Privacy Policy](https://x.ai/legal/privacy-policy) |

Wenlan also offers presets for Ollama and LM Studio, which run on your own machine and reach no network outside it, and a custom option where you type the address yourself. For a custom address, the policy is whatever the operator of that address publishes.

Wenlan's own relay, at `origin-relay.originmemory.workers.dev`, is run by this project rather than a third party. It stores the tunnel address you register and the random identifier described above, and nothing else.

## Data deletion

- Delete individual memories: `/forget` skill.
- Delete everything: remove `~/.wenlan/` and your platform data directory (`~/Library/Application Support/wenlan/` on macOS, `~/.local/share/wenlan/` on Linux, `%LOCALAPPDATA%\wenlan\` on Windows). An install upgraded from Origin still holds a full copy of its data in `~/.origin/` and in the sibling `origin` data folder (`~/Library/Application Support/origin/` on macOS, `~/.local/share/origin/` on Linux, `%LOCALAPPDATA%\origin\` on Windows); delete or copy those two as well.
- Uninstall the daemon: run `wenlan background off` to stop it and disable autostart (this is a reversible runtime stop, not an uninstall), then remove the service registration for your platform -- `~/Library/LaunchAgents/com.wenlan.server.plist` (macOS), `~/.config/systemd/user/wenlan-server.service` (Linux), or the `WenlanServer` scheduled task (Windows) -- and delete `~/.wenlan/bin/`.
- **Uninstall the desktop app:** turn off *Run Wenlan in background at login* in Settings and quit the app (this removes `~/Library/LaunchAgents/com.wenlan.server.plist` and `com.wenlan.desktop.plist` on macOS), then delete `Wenlan.app`; on Windows run the uninstaller from Apps & features. The data folders above stay until you delete them.

## Contact

Questions or concerns: open an issue at https://github.com/7xuanlu/wenlan/issues.

Last updated: 2026-08-29.
