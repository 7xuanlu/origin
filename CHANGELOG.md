# Changelog

## [0.17.0](https://github.com/7xuanlu/wenlan/compare/v0.16.0...v0.17.0) (2026-08-23)


### Features

* **app:** redesign the Atlas default view — place names, islands, memory dust ([#572](https://github.com/7xuanlu/wenlan/issues/572)) ([7d345ae](https://github.com/7xuanlu/wenlan/commit/7d345ae2710b04a83a8b50f8376a06e982a8edc4))
* **cli,server:** merge entities and declare aliases ([#575](https://github.com/7xuanlu/wenlan/issues/575)) ([d1c92ea](https://github.com/7xuanlu/wenlan/commit/d1c92ea287325a4509a99e79734ad3368ad590b7))


### Bug Fixes

* **app:** count only knowledge pages in the Home pages metric ([#570](https://github.com/7xuanlu/wenlan/issues/570)) ([ec9619e](https://github.com/7xuanlu/wenlan/commit/ec9619e9a8c94a08f188abe9e0971bbed0aa2b0b))
* **app:** cut the Atlas insight rail ([#571](https://github.com/7xuanlu/wenlan/issues/571)) ([7a2755b](https://github.com/7xuanlu/wenlan/commit/7a2755b5e35de1ff2517bae77bdcb22fca528c14))
* **app:** keep atlas shelf clusters live under drag physics ([#569](https://github.com/7xuanlu/wenlan/issues/569)) ([5d42c7a](https://github.com/7xuanlu/wenlan/commit/5d42c7a0a09cde32e3d1cc3ebaea3dfc90e4bd29))
* **app:** stop the entry animation defeating the archived row fade ([#568](https://github.com/7xuanlu/wenlan/issues/568)) ([0ed56b2](https://github.com/7xuanlu/wenlan/commit/0ed56b2b23e2a9b06b2061a8912d8c03b61846b1))
* **core:** give observations an identity and clean duplicates ([#573](https://github.com/7xuanlu/wenlan/issues/573)) ([b5a5c54](https://github.com/7xuanlu/wenlan/commit/b5a5c546f2980007d01c4e390f3ae6f0e2330b44))
* keep archive-superseded memories visible and labelled in app lists ([#564](https://github.com/7xuanlu/wenlan/issues/564)) ([dbfb120](https://github.com/7xuanlu/wenlan/commit/dbfb1206d5a10c85488cda43e32a1ac38b43a327))
* **release:** auto-open the release-as cleanup PR and reject stale overrides ([#566](https://github.com/7xuanlu/wenlan/issues/566)) ([c99ade8](https://github.com/7xuanlu/wenlan/commit/c99ade8691f4be1c8c7069d4e05ece4923d883c9))
* **release:** make the recovery dispatch rerun-safe ([#560](https://github.com/7xuanlu/wenlan/issues/560)) ([f326b5a](https://github.com/7xuanlu/wenlan/commit/f326b5affc1e7bb344c69e4cfc04ec2fc55c21fb))
* **release:** put MSYS_NO_PATHCONV on the Windows tag-check step, not the macOS one ([#561](https://github.com/7xuanlu/wenlan/issues/561)) ([b7f13e1](https://github.com/7xuanlu/wenlan/commit/b7f13e13dc9f594185e9666e0cb5de7062c9184f))
* **release:** read Windows bundle checksums via stdin so sha256sum does not escape them ([#562](https://github.com/7xuanlu/wenlan/issues/562)) ([e57847d](https://github.com/7xuanlu/wenlan/commit/e57847d1fedb3c1b3c1b24ec09bbdbc263043ef7))
* **release:** stop Git Bash path conversion breaking the Windows bundle tag check ([#559](https://github.com/7xuanlu/wenlan/issues/559)) ([f05adc0](https://github.com/7xuanlu/wenlan/commit/f05adc09ee65eb84662db05e802df067bc94813d))
* use the shared superseder predicates in the community parity gate and summary buckets ([#565](https://github.com/7xuanlu/wenlan/issues/565)) ([7487c7e](https://github.com/7xuanlu/wenlan/commit/7487c7e3b66d5c3ca46f3e9a4c163e9053537d8c))


### Code Refactoring

* pre-launch audit — remove dead surface, fix correctness, trim deps ([#574](https://github.com/7xuanlu/wenlan/issues/574)) ([54c8a7e](https://github.com/7xuanlu/wenlan/commit/54c8a7eac1a58ec0e9e853e7b12e20fe682fe411))

## [0.16.0](https://github.com/7xuanlu/wenlan/compare/v0.15.8...v0.16.0) (2026-08-19)


### Features

* **app:** expose M5 truth axes as read-only badges on Pages and Page detail ([#526](https://github.com/7xuanlu/wenlan/issues/526)) ([cc25e35](https://github.com/7xuanlu/wenlan/commit/cc25e354947aef257dcb49b09dc1977f9345bb50))
* **app:** load the whole knowledge graph and lay it out in two zones with page, entity and memory layers ([#532](https://github.com/7xuanlu/wenlan/issues/532)) ([d28bd28](https://github.com/7xuanlu/wenlan/commit/d28bd283b50d075d8ee8d4327c0a3359bdc61bee))
* **app:** make the desktop app's runtime paths and tests platform-portable ([#525](https://github.com/7xuanlu/wenlan/issues/525)) ([39c3649](https://github.com/7xuanlu/wenlan/commit/39c36497d843a323c7f5c0796d9f232e2de81cf6))
* **app:** make the Windows desktop build installable and publishable ([#528](https://github.com/7xuanlu/wenlan/issues/528)) ([4b1c724](https://github.com/7xuanlu/wenlan/commit/4b1c724c427107f47f188c0fbb4d6ec2753ce19c))
* **app:** mark pages human-reviewed through a backend-minted presence capability ([#529](https://github.com/7xuanlu/wenlan/issues/529)) ([9bf51ee](https://github.com/7xuanlu/wenlan/commit/9bf51ee316f7fb1c21383ee605eddbe821f16191))
* **app:** restore dev-runtime isolation lost in the monorepo migration ([#519](https://github.com/7xuanlu/wenlan/issues/519)) ([bc9885f](https://github.com/7xuanlu/wenlan/commit/bc9885f107188db5414c452bfd064b00d6332e3a))
* **app:** restore the community-map overlay lost in the monorepo migration ([#524](https://github.com/7xuanlu/wenlan/issues/524)) ([03919ba](https://github.com/7xuanlu/wenlan/commit/03919ba843982169b7b534814a2817f8bcc168b8))
* **app:** restore the native CodeMirror page editor lost in the monorepo migration ([#521](https://github.com/7xuanlu/wenlan/issues/521)) ([687ca4c](https://github.com/7xuanlu/wenlan/commit/687ca4c46a714296ac997dbc40dd32b414084867))
* **app:** restore the Page Map canvas tab lost in the monorepo migration ([#522](https://github.com/7xuanlu/wenlan/issues/522)) ([de21f91](https://github.com/7xuanlu/wenlan/commit/de21f9134ad5f7a4c540a79a4803fdf1c0b33c2f))
* **cli:** auto-start the registered daemon service on connect failure; add memories --pending ([#544](https://github.com/7xuanlu/wenlan/issues/544)) ([7194a20](https://github.com/7xuanlu/wenlan/commit/7194a203e8affccd8c08868dfa5a138f36c27819))
* **outbox:** queue Brief updates and captures when the daemon is unreachable; daemon drains them over loopback ([#546](https://github.com/7xuanlu/wenlan/issues/546)) ([43be4c2](https://github.com/7xuanlu/wenlan/commit/43be4c246c64c19294c1cb5b378753c56360fb2d))
* **plugin:** handoff skill runs CLI-only, survives a down daemon; SessionStart hook reports outbox counts ([#545](https://github.com/7xuanlu/wenlan/issues/545)) ([c52888d](https://github.com/7xuanlu/wenlan/commit/c52888d3b66776eee8f7ffaf8637edfb8ae59248))


### Bug Fixes

* **app:** graph region count and week card over visible nodes, uncategorized community scope, order-independent edge dedupe, focus entity always drawn, page click on overlay, drop dead atlas exports ([#553](https://github.com/7xuanlu/wenlan/issues/553)) ([9ecad93](https://github.com/7xuanlu/wenlan/commit/9ecad930323152ec2135b7af8e50ebf95e395105))
* **communities:** count generation bumps only for assertion relates edges, supersede stale community proposals on republish, cover 422 paths in community routes ([#552](https://github.com/7xuanlu/wenlan/issues/552)) ([8f381c5](https://github.com/7xuanlu/wenlan/commit/8f381c5aed62af5fe258ef9f280079423841b6c5))
* **core:** close the page-history bypasses and cap poison-document retries ([#523](https://github.com/7xuanlu/wenlan/issues/523)) ([5156d27](https://github.com/7xuanlu/wenlan/commit/5156d2732783a8996d89881273c892439b7694be))
* **core:** let decisions into the distill pool and summary buckets ([#539](https://github.com/7xuanlu/wenlan/issues/539)) ([63a7bf0](https://github.com/7xuanlu/wenlan/commit/63a7bf0c3467919b315d6e113a68e5cf764f9289))
* **core:** reject junk entity names at the write seam and normalise entity_type ([#537](https://github.com/7xuanlu/wenlan/issues/537)) ([231794f](https://github.com/7xuanlu/wenlan/commit/231794f4e9cab8375f5152c7f2a2858249e6a711))
* **db:** roll back leaked transactions and detect them at phase boundaries ([#520](https://github.com/7xuanlu/wenlan/issues/520)) ([2548e1e](https://github.com/7xuanlu/wenlan/commit/2548e1ebfb87b065045e9bfa1ecc7d272e026621))
* **gate:** pre-push stays fast and non-compiling; heavy closure opt-in via WENLAN_PUSH_FULL; rebase-safe change detection ([#556](https://github.com/7xuanlu/wenlan/issues/556)) ([7f49bf9](https://github.com/7xuanlu/wenlan/commit/7f49bf973c3ebe21039d414b03d1a525f2e69d7d))
* **gate:** store near-duplicate memories with a soft flag and skip superseded rows in the novelty check ([82f02c9](https://github.com/7xuanlu/wenlan/commit/82f02c9874a6d9ba714d8e7c666d46ed49adc442))
* **harness:** hook wrappers allow outside any git repo, keep failing closed inside one ([#543](https://github.com/7xuanlu/wenlan/issues/543)) ([e2919fd](https://github.com/7xuanlu/wenlan/commit/e2919fd54ec57684b5488ad896eec9bce2306b37))
* keep isolated app launches off the shared daemon; deliver update prompts during setup ([#517](https://github.com/7xuanlu/wenlan/issues/517)) ([dac69e3](https://github.com/7xuanlu/wenlan/commit/dac69e376d5bc451e8a74333659802e5c3634941))
* **kg:** cascade the community/genesis substrate on space move and merge ([#555](https://github.com/7xuanlu/wenlan/issues/555)) ([246ec7e](https://github.com/7xuanlu/wenlan/commit/246ec7e52b4030c88b8aa9a121353d75c9d956fa))
* **kg:** core data-integrity repairs and migration 124 (m113/m114 columns, cross-space edge fence, orphan edges on archive/delete, claim locator and inventory retirement) ([#547](https://github.com/7xuanlu/wenlan/issues/547)) ([f4900ac](https://github.com/7xuanlu/wenlan/commit/f4900ac84b4359037f1d04971f6636b8b473ad1c))
* **kg:** keep provenance on folded edges, sentinel-aware uncategorized lint scope, decodable minhash merge proposals that stay dismissed, re-type entities when their vocabulary is accepted ([#551](https://github.com/7xuanlu/wenlan/issues/551)) ([4d610f6](https://github.com/7xuanlu/wenlan/commit/4d610f6598edeef37ee0a7f98d56a0639d63ddea))
* **kg:** principled short-sha name rule, reject self-loop relations, drop dead fresh-DB backup guard, make the eval harness compile in CI ([#549](https://github.com/7xuanlu/wenlan/issues/549)) ([dc7da9b](https://github.com/7xuanlu/wenlan/commit/dc7da9b0e9e6ae6032dfbac657dab2965f3a2f95))
* **kg:** read live edges instead of the frozen relations table in count_relations, stale-source diagnostics, and the quality pipeline test ([#548](https://github.com/7xuanlu/wenlan/issues/548)) ([0ea8270](https://github.com/7xuanlu/wenlan/commit/0ea8270394d5f3ec5f632d7a2af2424c31311530))
* list staged page revision cards and review them from the page itself ([#538](https://github.com/7xuanlu/wenlan/issues/538)) ([c164aec](https://github.com/7xuanlu/wenlan/commit/c164aecefaecf9b9926ce8ddd925d0f14612a4ef))
* make the repository buildable, testable, and pushable on Windows ([#534](https://github.com/7xuanlu/wenlan/issues/534)) ([6b3f664](https://github.com/7xuanlu/wenlan/commit/6b3f6640bd4326cc37eea2dc11159d02ab072e80))
* **server:** entity observation validation, memory-link union on graph readers, link-entity 404, delete dead KG surfaces ([#554](https://github.com/7xuanlu/wenlan/issues/554)) ([44b1284](https://github.com/7xuanlu/wenlan/commit/44b12845b0d4abde9f578db1bce6c09707f82aa5))

## [0.15.8](https://github.com/7xuanlu/wenlan/compare/v0.15.7...v0.15.8) (2026-08-09)


### Features

* point the app updater at the unified release manifest ([#514](https://github.com/7xuanlu/wenlan/issues/514)) ([350512d](https://github.com/7xuanlu/wenlan/commit/350512d925b3eebcb869ea4e3ccb7e09fc1f89e1))

## [0.15.7](https://github.com/7xuanlu/wenlan/compare/v0.15.6...v0.15.7) (2026-08-09)


### Features

* build and promote the macOS app bundle in the unified release workflow ([#509](https://github.com/7xuanlu/wenlan/issues/509)) ([56b6d7b](https://github.com/7xuanlu/wenlan/commit/56b6d7bdc4044b05046aa1c1419d27f3a7820e2d))
* lock the desktop app version to the workspace release train ([#510](https://github.com/7xuanlu/wenlan/issues/510)) ([f56d304](https://github.com/7xuanlu/wenlan/commit/f56d3043cef00e48d0f8844ad104ab92be192210))
* source app sidecars from the monorepo tree ([#507](https://github.com/7xuanlu/wenlan/issues/507)) ([e47c2b0](https://github.com/7xuanlu/wenlan/commit/e47c2b0d6cf08b496905721a5bb028f3f68b7df3))

## [0.15.6](https://github.com/7xuanlu/wenlan/compare/v0.15.5...v0.15.6) (2026-08-09)


### Features

* fold the wenlan-app desktop crate into the monorepo (app/), reversing the 2026-05-07 split. App-aware CI planner, app-check job, workspace lock with app subgraph. ([df73d30](https://github.com/7xuanlu/wenlan/commit/df73d304bb45c29a2b3c111c6704866ee7d00f6b))
* G6 Stage 3 — migration 123 retires the legacy entities/entity_aliases tables ([#499](https://github.com/7xuanlu/wenlan/issues/499)) ([0aaa695](https://github.com/7xuanlu/wenlan/commit/0aaa695eb4d02ca8458032710f8947281a1e3e9d))
* **kg:** edges carry their semantic payload — one source of truth (G6 Stage 1) ([#486](https://github.com/7xuanlu/wenlan/issues/486)) ([8100223](https://github.com/7xuanlu/wenlan/commit/8100223e6539d842c1e25de91aca36e108e1faab))
* **kg:** G6 Stage 1.1 — resolved page-link readers cut over to canonical links edges ([#488](https://github.com/7xuanlu/wenlan/issues/488)) ([f44ea66](https://github.com/7xuanlu/wenlan/commit/f44ea66cb386d21b771170d60443c4befc88b37b))
* **kg:** G6 Stage 1.2 — relations readers cut over to canonical relates edges ([#489](https://github.com/7xuanlu/wenlan/issues/489)) ([8105cac](https://github.com/7xuanlu/wenlan/commit/8105cac6f646ba8b9398764ad3178b2eb5a2c875))
* **kg:** G6 Stage 1.3 — page_sources/page_evidence readers cut over to canonical cites edges ([#490](https://github.com/7xuanlu/wenlan/issues/490)) ([9ac0fe3](https://github.com/7xuanlu/wenlan/commit/9ac0fe3f9b92aa1d9097bf7ab42149056d273691))
* **kg:** G6 Stage 1.5a — clean entity readers cut over to kind='entity' shadow pages ([#491](https://github.com/7xuanlu/wenlan/issues/491)) ([2a08ca6](https://github.com/7xuanlu/wenlan/commit/2a08ca640e113c7b62f96374a781809e5700f7bb))
* **kg:** G6 Stage 1.5b — scalar mirror extension + entity reader completion ([#492](https://github.com/7xuanlu/wenlan/issues/492)) ([ba89434](https://github.com/7xuanlu/wenlan/commit/ba894341b6d3f15ea1bd4111c62a00b4ffbc8bec))
* **kg:** G6 Stage 2 PR 2a — retire edges/entity-page parity-oracle machinery (m120) ([#494](https://github.com/7xuanlu/wenlan/issues/494)) ([9d5eb29](https://github.com/7xuanlu/wenlan/commit/9d5eb296b92580cb3483cf5e50b6298d1b65e521))
* **kg:** G6 Stage 2 PR 2c — entity writers and readers canonical-only; entity parity oracle retires ([#498](https://github.com/7xuanlu/wenlan/issues/498)) ([892e724](https://github.com/7xuanlu/wenlan/commit/892e72440f95f77ef0e98c055e2e2a0d659128fc))


### Bug Fixes

* **app:** adapt to wenlan-types HEAD — WriteSpaceTarget + StatusResponse fields ([7649f2f](https://github.com/7xuanlu/wenlan/commit/7649f2ff74d3c69f833061135c33e7faaac48d25))
* grant release finalization the pull-requests write it uses ([#483](https://github.com/7xuanlu/wenlan/issues/483)) ([573f45c](https://github.com/7xuanlu/wenlan/commit/573f45ce785d615283a0b81fea962fe073e0670a))
* **kg:** carry the label on rebind re-mints; survive the m81 pre-semantic_type crash window ([#487](https://github.com/7xuanlu/wenlan/issues/487)) ([29f51db](https://github.com/7xuanlu/wenlan/commit/29f51dbb3188fbe850e22f752a1b68f129b976c8))
* **kg:** edges-parity repair — doc-source-page over-retire + kind-derivation drift (m119) ([#493](https://github.com/7xuanlu/wenlan/issues/493)) ([6e55889](https://github.com/7xuanlu/wenlan/commit/6e5588949b3ef6494a05409d7e3847c719c9d7f7))
* **kg:** G6 Stage 0 part 1 — three writers re-drifting edges parity ([#481](https://github.com/7xuanlu/wenlan/issues/481)) ([b2c8280](https://github.com/7xuanlu/wenlan/commit/b2c828057dc1b284cc73174f87d9749af2b7528a))
* **kg:** G6 Stage 0 part 2 — five secondary writers re-drifting edges parity ([#482](https://github.com/7xuanlu/wenlan/issues/482)) ([1658e4d](https://github.com/7xuanlu/wenlan/commit/1658e4dc4ac8e367c7c5231a29b88f0ba26d66c0))
* **kg:** writers canonical-only for legacy KG stores; discovery-scan sweep (G6 Stage 2b) ([#495](https://github.com/7xuanlu/wenlan/issues/495)) ([7a8287f](https://github.com/7xuanlu/wenlan/commit/7a8287fe08e13c05c3beabec7d54f344ecbee34b))
* **test:** quarantine remaining high-risk flake patterns ([#497](https://github.com/7xuanlu/wenlan/issues/497)) ([dd0e617](https://github.com/7xuanlu/wenlan/commit/dd0e6177056652ec2d96cb331a40cd0b5d245b79))
* **test:** quarantine three flaky tests behind serialization and CI retries ([#496](https://github.com/7xuanlu/wenlan/issues/496)) ([71e9196](https://github.com/7xuanlu/wenlan/commit/71e9196d42e634566553e79f30e2563a668544d3))
* unify sysinfo on the workspace =0.38.3 pin ([007fc98](https://github.com/7xuanlu/wenlan/commit/007fc9816a105f5099f1225f3e7a2633797426ea))

## [0.15.5](https://github.com/7xuanlu/wenlan/compare/v0.15.4...v0.15.5) (2026-08-04)


### Features

* **m6:** gate the genesis shadow lane and fix its turn-driver livelock ([#469](https://github.com/7xuanlu/wenlan/issues/469)) ([7456f4f](https://github.com/7xuanlu/wenlan/commit/7456f4f6622b2b01415db268b90f9f2cbc8a8d59))


### Bug Fixes

* align the runtime image env contract with the nested data root ([#464](https://github.com/7xuanlu/wenlan/issues/464)) ([80a0d66](https://github.com/7xuanlu/wenlan/commit/80a0d66a3c63d502267b6c408ed109ba385d382b))
* give the release runtime image its OpenMP runtime ([#462](https://github.com/7xuanlu/wenlan/issues/462)) ([e8a9f88](https://github.com/7xuanlu/wenlan/commit/e8a9f88ea7e7bba5878d583c4bb0cc7919c15fbf))
* grant finalize-release the pull-requests scope it reads ([#465](https://github.com/7xuanlu/wenlan/issues/465)) ([1567515](https://github.com/7xuanlu/wenlan/commit/1567515f0e8b62003e7f14ac03aa4093a086bc95))
* **kg:** ambient enrichment dual-writes canonical edges; migration 111 repairs parity ([#473](https://github.com/7xuanlu/wenlan/issues/473)) ([d31c1eb](https://github.com/7xuanlu/wenlan/commit/d31c1eb58cd950428ad703a677199751ecdf7fbc))
* **kg:** ambient entity sweep dual-writes entity shadow pages; migration 113 repairs parity ([#479](https://github.com/7xuanlu/wenlan/issues/479)) ([c74ca3c](https://github.com/7xuanlu/wenlan/commit/c74ca3c9246787131a0980159b4321376ab01b46))
* **kg:** migration 114 re-syncs stale entity shadow pages ([#480](https://github.com/7xuanlu/wenlan/issues/480)) ([cfa972b](https://github.com/7xuanlu/wenlan/commit/cfa972bf593470c8fbff087f937b38378f9e7ac5))
* make release promotion and latest promotion fail closed ([#460](https://github.com/7xuanlu/wenlan/issues/460)) ([035e04e](https://github.com/7xuanlu/wenlan/commit/035e04ef0764f6138d0dba615b5b5b565445d4ee))
* nest the runtime data root inside its writable volume ([#463](https://github.com/7xuanlu/wenlan/issues/463)) ([ebfd9ca](https://github.com/7xuanlu/wenlan/commit/ebfd9ca34fbcaec1e1781382f2d5cca4e21f5740))
* repair the Homebrew and Docker publication lanes ([#461](https://github.com/7xuanlu/wenlan/issues/461)) ([e56c381](https://github.com/7xuanlu/wenlan/commit/e56c381e06290f88e858aa440176bc6f590f7136))
* skip the release candidate observer when its source CI run did not succeed ([#474](https://github.com/7xuanlu/wenlan/issues/474)) ([c6ef28c](https://github.com/7xuanlu/wenlan/commit/c6ef28c43dac7a4bc120d1602507b76c8cf702b4))
* **truth:** fence M5 claim edges out of the edges parity sweep ([#478](https://github.com/7xuanlu/wenlan/issues/478)) ([15909df](https://github.com/7xuanlu/wenlan/commit/15909df4518380f8e15fe67a991ac1f104e73ce6))
* **truth:** make memory origin daemon-authoritative ([#475](https://github.com/7xuanlu/wenlan/issues/475)) ([20bb938](https://github.com/7xuanlu/wenlan/commit/20bb938e132e41a5ec29da3531702c65fb223be4))

## [0.15.4](https://github.com/7xuanlu/wenlan/compare/v0.15.3...v0.15.4) (2026-08-03)


### Features

* add shadow claim support promoter ([#457](https://github.com/7xuanlu/wenlan/issues/457)) ([c620d7e](https://github.com/7xuanlu/wenlan/commit/c620d7e2a4977bb7b332851469314c022b52e283))
* complete M6 migration 109 substrate follow-up ([#437](https://github.com/7xuanlu/wenlan/issues/437)) ([247185b](https://github.com/7xuanlu/wenlan/commit/247185ba215d3426a4a1f2b97ad76d924fd1b30d))
* enforce truth-aware page exposure ([#429](https://github.com/7xuanlu/wenlan/issues/429)) ([5141f7b](https://github.com/7xuanlu/wenlan/commit/5141f7bf21586aee75a1204607651b1eba484c42))
* land M6 deterministic identity primitives and genesis substrate ([#433](https://github.com/7xuanlu/wenlan/issues/433)) ([aa77814](https://github.com/7xuanlu/wenlan/commit/aa77814c6a736e8ffef128e013ece56357f33481))


### Bug Fixes

* fence M5 derivation runs and bound reconciliation ([#439](https://github.com/7xuanlu/wenlan/issues/439)) ([f9b0789](https://github.com/7xuanlu/wenlan/commit/f9b07890d34697cc058e139d50bb250773fed50a))
* focus affected tests and harden release fallback ([#438](https://github.com/7xuanlu/wenlan/issues/438)) ([228576a](https://github.com/7xuanlu/wenlan/commit/228576a55d15bfc1c5af640302e4d0c4ab485add))
* keep release candidate tree current ([#436](https://github.com/7xuanlu/wenlan/issues/436)) ([9098693](https://github.com/7xuanlu/wenlan/commit/90986936e949416f7b0faa90324f084ff7a11602))
* publish tested release artifacts ([#431](https://github.com/7xuanlu/wenlan/issues/431)) ([096af3f](https://github.com/7xuanlu/wenlan/commit/096af3f9a9919162bd4966d82ba1ed6cfec18565))

## [0.15.3](https://github.com/7xuanlu/wenlan/compare/v0.15.2...v0.15.3) (2026-08-01)


### Features

* close the M5 daemon gaps — cutover readiness, page review endpoint, named-page truth axes ([#418](https://github.com/7xuanlu/wenlan/issues/418)) ([1c903be](https://github.com/7xuanlu/wenlan/commit/1c903becb99eb19653acc0c0974a546d0b0bfcbd))

## [0.15.2](https://github.com/7xuanlu/wenlan/compare/v0.15.1...v0.15.2) (2026-07-31)


### Bug Fixes

* skip the setup runtime repair when the daemon is newer than the plugin ([#415](https://github.com/7xuanlu/wenlan/issues/415)) ([22b401d](https://github.com/7xuanlu/wenlan/commit/22b401da132a975e58db047639f3eac35f3efd75))
* stop test builds resolving the developer's real config.json ([#411](https://github.com/7xuanlu/wenlan/issues/411)) ([07afba7](https://github.com/7xuanlu/wenlan/commit/07afba7df820ab4d9a6cd3e4a6f89b8506a23a43))

## [0.15.1](https://github.com/7xuanlu/wenlan/compare/v0.15.0...v0.15.1) (2026-07-29)


### Features

* add CLI Space Brief workflow ([e363754](https://github.com/7xuanlu/wenlan/commit/e363754a82e6dbc596a52934fbaf9df17506f8f5))
* add daemon Space Brief routes ([d4f842d](https://github.com/7xuanlu/wenlan/commit/d4f842de27de30148fce87b259da9e5c13d1d1a7))
* add Space Brief storage ([f00a5a6](https://github.com/7xuanlu/wenlan/commit/f00a5a676a44493491d8d7d0c8c50da3292b3100))
* expose Space Brief workflow ([1178e18](https://github.com/7xuanlu/wenlan/commit/1178e18448feee2f5eef66a13951bbc19f46f55e))
* replace session-start context with Space Briefs ([f416396](https://github.com/7xuanlu/wenlan/commit/f416396218bccca8b39b983274316d045329b1f3))


### Bug Fixes

* add daemon-owned default save space ([#399](https://github.com/7xuanlu/wenlan/issues/399)) ([79eadad](https://github.com/7xuanlu/wenlan/commit/79eadad0dc571457706a879e61be5a070d1ea3c9))
* consolidate MCP surface and grounded review flows ([#397](https://github.com/7xuanlu/wenlan/issues/397)) ([0534517](https://github.com/7xuanlu/wenlan/commit/053451772de103f96903b88585fbf7096d81d5af))
* harden Space Brief handoff ([82b63a0](https://github.com/7xuanlu/wenlan/commit/82b63a046fd8f3b38b8a3b06204061f6e852a1db))
* land the M4 communities rung — durable substrate and reader cutover ([#396](https://github.com/7xuanlu/wenlan/issues/396)) ([5ba8a3b](https://github.com/7xuanlu/wenlan/commit/5ba8a3b43b53267a7d973ef377275a86e93ad01e))
* land the M5 truth-exposure substrate, inert at generation 0 (PR-B) ([#408](https://github.com/7xuanlu/wenlan/issues/408)) ([f29c2c5](https://github.com/7xuanlu/wenlan/commit/f29c2c5496b2a60b1478e137641a084e29e53a3f))
* load release-please config (release-type input bypassed it) ([#410](https://github.com/7xuanlu/wenlan/issues/410)) ([48bf0a7](https://github.com/7xuanlu/wenlan/commit/48bf0a798d1931a9c28403862de4483aeb180ff1))
* M5 PR-A — durable claim identity substrate (schema + shadow derivation) ([#404](https://github.com/7xuanlu/wenlan/issues/404)) ([e4790ce](https://github.com/7xuanlu/wenlan/commit/e4790ce857056050a90a4adeef391375e8ce5f19))
* wire the M5 truth adapters — the protecting half of the exposure contract (PR-C) ([3932e3d](https://github.com/7xuanlu/wenlan/commit/3932e3d557ec0201324446c7d15b582c605dbe20))

## [0.15.0](https://github.com/7xuanlu/wenlan/compare/v0.14.1...v0.15.0) (2026-07-26)


### Features

* make background enrichment bounded and foreground-safe ([#379](https://github.com/7xuanlu/wenlan/issues/379)) ([9817e38](https://github.com/7xuanlu/wenlan/commit/9817e38661894fa3627580e8a09319ff44805c67))


### Bug Fixes

* accelerate Windows inference with Vulkan ([#382](https://github.com/7xuanlu/wenlan/issues/382)) ([6501345](https://github.com/7xuanlu/wenlan/commit/65013459b0909cbb1899e9fd1667f80bad981194))
* bound daemon logs and preserve bootstrap errors ([#385](https://github.com/7xuanlu/wenlan/issues/385)) ([93451bf](https://github.com/7xuanlu/wenlan/commit/93451bf0ef58399e08400e3b4ac613942adcfec8))
* defer observation provenance index to migrations ([#381](https://github.com/7xuanlu/wenlan/issues/381)) ([c8533a7](https://github.com/7xuanlu/wenlan/commit/c8533a7a825bab5830c73b0fe4998d2e7e6e0ec4))
* load bundled ONNX Runtime on Windows ([#365](https://github.com/7xuanlu/wenlan/issues/365)) ([c66f9d8](https://github.com/7xuanlu/wenlan/commit/c66f9d8e3e2edc991a540a89d3c5f60e2c109a99))
* M1 honest columns — migrate Page scope into unified `space` column ([#375](https://github.com/7xuanlu/wenlan/issues/375)) ([1b56c3a](https://github.com/7xuanlu/wenlan/commit/1b56c3a883c8f916fe3e604287a784898ec2167b))
* M2 PR-2 — edges reader cutover, parity reconciliation + §6.9 online backup ([#378](https://github.com/7xuanlu/wenlan/issues/378)) ([3246e18](https://github.com/7xuanlu/wenlan/commit/3246e180b3d5f196d8bf6be4079d53cf98aac850))
* M2 unified edges — schema expand + dual-write shadow (PR-1) ([#377](https://github.com/7xuanlu/wenlan/issues/377)) ([b256814](https://github.com/7xuanlu/wenlan/commit/b256814832be585e63888e997c6c7c6bf22fe119))
* M3 PR-1 — entity→page dual-write foundation (schema 89–93) ([#380](https://github.com/7xuanlu/wenlan/issues/380)) ([24c45d8](https://github.com/7xuanlu/wenlan/commit/24c45d8ab74713191fea501c35c3cd98b38e9737))
* M3 PR-2 — entity reader cutover, parity watermark + D6 + §6.5 soak ([#387](https://github.com/7xuanlu/wenlan/issues/387)) ([7f3bd7e](https://github.com/7xuanlu/wenlan/commit/7f3bd7ec3238acdaf2c5902bc424f033be264811))
* M3 wire freeze — adapter seam, freeze teeth, Q1 stub-page fence lifts ([#390](https://github.com/7xuanlu/wenlan/issues/390)) ([084e39d](https://github.com/7xuanlu/wenlan/commit/084e39dfcfd0091e5143a9acdf458a4f92775e2a))
* M3g edge-grounding promotion — span capture, default-OFF grounding sweep, gates made real ([#392](https://github.com/7xuanlu/wenlan/issues/392)) ([31e0fa6](https://github.com/7xuanlu/wenlan/commit/31e0fa6377a4537b0df7cb073b5fa403c5b118ac))
* start Page-Map Canvas blank — proactive suggestions now opt-in ([#371](https://github.com/7xuanlu/wenlan/issues/371)) ([34beb24](https://github.com/7xuanlu/wenlan/commit/34beb24663590c3e62e9391e3f138960ef229021))

## [0.14.1](https://github.com/7xuanlu/wenlan/compare/v0.14.0...v0.14.1) (2026-07-20)


### Bug Fixes

* preserve exact canonical page source ([c111ff1](https://github.com/7xuanlu/wenlan/commit/c111ff15333423551b062493b4f3ae44618f031b))

## [0.14.0](https://github.com/7xuanlu/wenlan/compare/v0.13.2...v0.14.0) (2026-07-20)


### Features

* M0 — route every Page write through one gate ([#369](https://github.com/7xuanlu/wenlan/issues/369)) ([7f70d2d](https://github.com/7xuanlu/wenlan/commit/7f70d2d0ba248d7fdd1df082da8b7c6e8a0e9353))
* per-job source pins for everyday/synthesis routing (no vendor privilege) + resolved-routing endpoint ([#357](https://github.com/7xuanlu/wenlan/issues/357)) ([d945347](https://github.com/7xuanlu/wenlan/commit/d945347f14b61eb9ae4e0a22c35cfe1455e3fd62))
* Tier 1 vocabulary auto-heal (deterministic folds auto-applied, semantic values queued for review) ([#362](https://github.com/7xuanlu/wenlan/issues/362)) ([88fc7df](https://github.com/7xuanlu/wenlan/commit/88fc7df1a242a1f2dbc33b6e76c2af4c39a1748a))


### Bug Fixes

* add core Page draft lifecycle ([#368](https://github.com/7xuanlu/wenlan/issues/368)) ([a989408](https://github.com/7xuanlu/wenlan/commit/a9894083a3f2e139f4318920a8cd1412f70b1015))
* complete approval-gated lint repair workflow ([#370](https://github.com/7xuanlu/wenlan/issues/370)) ([0c273c5](https://github.com/7xuanlu/wenlan/commit/0c273c52e7a215dae6184d5fb0df153cad5214b6))
* define Page draft wire contract ([#366](https://github.com/7xuanlu/wenlan/issues/366)) ([c4da0c8](https://github.com/7xuanlu/wenlan/commit/c4da0c8fa066803b71ab01f115e17d43c08626a7))
* dev-build version self-identification + plugin/daemon drift root-fixes ([#359](https://github.com/7xuanlu/wenlan/issues/359)) ([a10cf2a](https://github.com/7xuanlu/wenlan/commit/a10cf2a2f309058124cbec43f76aa480194c79af))
* drop the release-time marketplace notification ([#353](https://github.com/7xuanlu/wenlan/issues/353)) ([7d0fd95](https://github.com/7xuanlu/wenlan/commit/7d0fd959f3028f4ec053bac5a84f772139e54752))
* enforce read scope contracts ([#356](https://github.com/7xuanlu/wenlan/issues/356)) ([7b74c85](https://github.com/7xuanlu/wenlan/commit/7b74c85802ced0362b16ee1576efa8fb29fff823))
* isolate Page drafts from legacy mutators ([#367](https://github.com/7xuanlu/wenlan/issues/367)) ([0f7b8fe](https://github.com/7xuanlu/wenlan/commit/0f7b8fec7ea9e6dd7dca06f72d1c090e758b1e47))
* Page Map v1 daemon — schema, accessors, routes, improve pass ([#364](https://github.com/7xuanlu/wenlan/issues/364)) ([44bd6e8](https://github.com/7xuanlu/wenlan/commit/44bd6e8ded8517624260d5d3cdd52cad253097ea))

## [0.13.2](https://github.com/7xuanlu/wenlan/compare/v0.13.1...v0.13.2) (2026-07-14)


### Bug Fixes

* gate releases behind a prerelease, not a draft ([#340](https://github.com/7xuanlu/wenlan/issues/340)) ([ecdfd0a](https://github.com/7xuanlu/wenlan/commit/ecdfd0aec6dadd55c9a0c32f91d9fdf2c7a479cb))
* harden evidence-driven lint maintenance paths ([#351](https://github.com/7xuanlu/wenlan/issues/351)) ([92020e6](https://github.com/7xuanlu/wenlan/commit/92020e6e9417b5d0206ae32abf4f563672f2c63c))

## [0.13.1](https://github.com/7xuanlu/wenlan/compare/v0.13.0...v0.13.1) (2026-07-13)


### Bug Fixes

* add whole-system and deep semantic lint diagnostics ([#343](https://github.com/7xuanlu/wenlan/issues/343)) ([1c912fb](https://github.com/7xuanlu/wenlan/commit/1c912fb10986b600227be72da2b36abc5cc2e36a))
* harden lint diagnostics from live dogfood ([#349](https://github.com/7xuanlu/wenlan/issues/349)) ([f567f22](https://github.com/7xuanlu/wenlan/commit/f567f2243251d428e1de06fc1cf198814ff57f1c))
* keep daemon responsive during directory sync ([#350](https://github.com/7xuanlu/wenlan/issues/350)) ([54f3dd3](https://github.com/7xuanlu/wenlan/commit/54f3dd3903de0566be30becc1c241ed1b88fc887))

## [0.13.0](https://github.com/7xuanlu/wenlan/compare/v0.12.1...v0.13.0) (2026-07-12)


### Features

* external-LLM API key (Bearer), config hot-swap, external status ([#342](https://github.com/7xuanlu/wenlan/issues/342)) ([960095c](https://github.com/7xuanlu/wenlan/commit/960095c457a7992c69fc77a7d1b2f3d6b77d9f5e))

## [0.12.1](https://github.com/7xuanlu/wenlan/compare/v0.12.0...v0.12.1) (2026-07-12)


### Bug Fixes

* enforce release and public contract drift checks ([#341](https://github.com/7xuanlu/wenlan/issues/341)) ([9a59085](https://github.com/7xuanlu/wenlan/commit/9a590858f34336ece715a4b6ab537865b31df6cc))
* route document source pages through PageWrite ([a0861b2](https://github.com/7xuanlu/wenlan/commit/a0861b241767a58e46008bf2236c7dde54ca56d8))

## [0.12.0](https://github.com/7xuanlu/wenlan/compare/v0.11.0...v0.12.0) (2026-07-08)


### Features

* distill redesign - one refresh op, canonical PageWrite births, citation-gated synthesis ([#336](https://github.com/7xuanlu/wenlan/issues/336)) ([33a721c](https://github.com/7xuanlu/wenlan/commit/33a721c4544fdac7956087f8bfe647e484507c1f))
* per-claim verified citations for wiki pages ([#332](https://github.com/7xuanlu/wenlan/issues/332)) ([2f6ee4b](https://github.com/7xuanlu/wenlan/commit/2f6ee4bdc3db4c833f00de6991fc71907cd5fc92))


### Bug Fixes

* guard loopback daemon against cross-origin browser access; gate release publish on build success ([#335](https://github.com/7xuanlu/wenlan/issues/335)) ([4ee3ee5](https://github.com/7xuanlu/wenlan/commit/4ee3ee5299de123d6734c6fa2780cda8e42591e9))
* **server:** bind port before any data-dir work ([#334](https://github.com/7xuanlu/wenlan/issues/334)) ([046a1c9](https://github.com/7xuanlu/wenlan/commit/046a1c919dd84b040dc16163d39f55226a618dc6))

## [0.11.0](https://github.com/7xuanlu/wenlan/compare/v0.10.1...v0.11.0) (2026-07-03)


### Features

* doc-grounded revisions — documents propose rewrites to conflicting captures (L3) ([#330](https://github.com/7xuanlu/wenlan/issues/330)) ([7267699](https://github.com/7xuanlu/wenlan/commit/7267699ce5423a76875f149492fe839f843b1960))

## [0.10.1](https://github.com/7xuanlu/wenlan/compare/v0.10.0...v0.10.1) (2026-07-03)


### Bug Fixes

* clean up Wenlan command surface ([bcd56e8](https://github.com/7xuanlu/wenlan/commit/bcd56e8d51c02fd27f7fd90bc20a0a19634453e0))

## [0.10.0](https://github.com/7xuanlu/wenlan/compare/v0.9.6...v0.10.0) (2026-07-02)


### Features

* folder & multi-format document ingest (L1) ([#320](https://github.com/7xuanlu/wenlan/issues/320)) ([4cdc456](https://github.com/7xuanlu/wenlan/commit/4cdc45613fac42c51336b295c39295fdc859a569))


### Bug Fixes

* port pages skill to Codex plugin ([#318](https://github.com/7xuanlu/wenlan/issues/318)) ([1d869cd](https://github.com/7xuanlu/wenlan/commit/1d869cda53c14a4b14c38cafd6f0ffa1900cc897))

## [0.9.6](https://github.com/7xuanlu/wenlan/compare/v0.9.5...v0.9.6) (2026-07-01)


### Bug Fixes

* add Codex plugin install path ([eac308e](https://github.com/7xuanlu/wenlan/commit/eac308e62d3bd2c81207b846c2790c421eb0e571))
* add Wenlan plugin shared contract ([4e786e9](https://github.com/7xuanlu/wenlan/commit/4e786e91e1d122ba0b1db6ee8efda9519f72de68))
* stop space auto-create and cascade space moves ([993a039](https://github.com/7xuanlu/wenlan/commit/993a039267e4a6cd938fe3e0eaaf92a67dfd5cc5))

## [0.9.5](https://github.com/7xuanlu/wenlan/compare/v0.9.4...v0.9.5) (2026-06-30)


### Bug Fixes

* /curate revision queue overhaul — diff preview, dismiss=unstage, dead-code cleanup ([#311](https://github.com/7xuanlu/wenlan/issues/311)) ([d072a38](https://github.com/7xuanlu/wenlan/commit/d072a380c8fe819bc896176e4d5a6705d424cf49))

## [0.9.4](https://github.com/7xuanlu/wenlan/compare/v0.9.3...v0.9.4) (2026-06-30)


### Bug Fixes

* pages/curate CLI-first browse + Wenlan capture-guidance boundary ([#310](https://github.com/7xuanlu/wenlan/issues/310)) ([682ba36](https://github.com/7xuanlu/wenlan/commit/682ba360e304646dbec206b6616cb18a9f7eae32))
* preserve edited pages on skipped re-distill ([ab84bdd](https://github.com/7xuanlu/wenlan/commit/ab84bdd63080a4473b2d7d583a2ee1be35202ecc))

## [0.9.3](https://github.com/7xuanlu/wenlan/compare/v0.9.2...v0.9.3) (2026-06-29)


### Bug Fixes

* add `wenlan pages` CLI; /pages skill browses + opens via it ([#307](https://github.com/7xuanlu/wenlan/issues/307)) ([f29b5ff](https://github.com/7xuanlu/wenlan/commit/f29b5ff15c4795de6994a9561f452242d6657595))
* include document tag map in tags response ([#304](https://github.com/7xuanlu/wenlan/issues/304)) ([1fdd5f3](https://github.com/7xuanlu/wenlan/commit/1fdd5f300015b015ff63e6f39a617184cef5361e))
* share on-device model DTOs ([e401bc3](https://github.com/7xuanlu/wenlan/commit/e401bc3798bac804abf7d4069d8f7fc62ce78a6f))

## [0.9.2](https://github.com/7xuanlu/wenlan/compare/v0.9.1...v0.9.2) (2026-06-26)


### Bug Fixes

* repair stale wenlan plugin runtime ([963ab28](https://github.com/7xuanlu/wenlan/commit/963ab2824306ce9200fc7756aa0e2cdc50dfbe1f))
* surface distilled wiki pages across recall/search surfaces, space + tier gated ([#303](https://github.com/7xuanlu/wenlan/issues/303)) ([d6f770c](https://github.com/7xuanlu/wenlan/commit/d6f770c978e1885a1d8c881705ea4bb9c14e36bc))

## [0.9.1](https://github.com/7xuanlu/wenlan/compare/v0.9.0...v0.9.1) (2026-06-24)


### Bug Fixes

* complete Origin→Wenlan rebrand tail (docs, plugin, scripts) ([#292](https://github.com/7xuanlu/wenlan/issues/292)) ([3f80de3](https://github.com/7xuanlu/wenlan/commit/3f80de307f55a60dc8da248741cfa87eb0e53e7f))
* **dist:** complete origin→wenlan cleanup across all publish surfaces ([#289](https://github.com/7xuanlu/wenlan/issues/289)) ([8b1c6de](https://github.com/7xuanlu/wenlan/commit/8b1c6dedc83d667e0197652c6f64ac1bf182f187))
* rename CLI MCP server handle origin -&gt; wenlan; drop tracked ffmpeg2pass junk ([#294](https://github.com/7xuanlu/wenlan/issues/294)) ([cce076a](https://github.com/7xuanlu/wenlan/commit/cce076a80e7071d4e0339be7ecde0469ea0649f5))

## [0.9.0](https://github.com/7xuanlu/wenlan/compare/v0.8.6...v0.9.0) (2026-06-23)


### Features

* Phase-5 publish cutover — flip distribution identities to wenlan ([#288](https://github.com/7xuanlu/wenlan/issues/288)) ([3474f0c](https://github.com/7xuanlu/wenlan/commit/3474f0ccb9a648d2c18a86f3c7daefa2d7a64b04))
* rebrand to Wenlan (文瀾) ([#284](https://github.com/7xuanlu/wenlan/issues/284)) ([19441c8](https://github.com/7xuanlu/wenlan/commit/19441c8eadf047bb1dcd5c9d7821a59ca984b319))


### Bug Fixes

* config-backed WENLAN_RERANKER_MODE via `wenlan reranker <off|lite|full>` ([#285](https://github.com/7xuanlu/wenlan/issues/285)) ([33f89af](https://github.com/7xuanlu/wenlan/commit/33f89afd080540c0570b3d50b5699958d82ef8a5))

## [0.8.6](https://github.com/7xuanlu/origin/compare/v0.8.5...v0.8.6) (2026-06-22)


### Bug Fixes

* ORIGIN_RERANKER_MODE (off|lite|full) — reachable cross-encoder rerank ([#282](https://github.com/7xuanlu/origin/issues/282)) ([be88029](https://github.com/7xuanlu/origin/commit/be880296a4a33615c60aa5b0d9d16e57ddfeb76e))


### Performance Improvements

* continuous-batch slot backfill for on-device enrichment (default-OFF, 1.65x decode) ([#276](https://github.com/7xuanlu/origin/issues/276)) ([68f2f76](https://github.com/7xuanlu/origin/commit/68f2f765fe3fd90eaed0bc41032d04a75779839e))
* **engine:** prefill-side prefix-KV cache for on-device enrichment (default-OFF, ~1.9x prefill) ([#278](https://github.com/7xuanlu/origin/issues/278)) ([8274538](https://github.com/7xuanlu/origin/commit/827453837204b692bbf776c9dbedeeedb6cbc9e8))

## [0.8.5](https://github.com/7xuanlu/origin/compare/v0.8.4...v0.8.5) (2026-06-19)


### Bug Fixes

* **eval:** scenario concurrency for on-device LME CE A/B + batch-log instrumentation ([#270](https://github.com/7xuanlu/origin/issues/270)) ([b06872d](https://github.com/7xuanlu/origin/commit/b06872d49c7b3481db6b6c37a0a49aa3688cba88))
* guard migration 55 Pass B against orphaned entity_id ([#274](https://github.com/7xuanlu/origin/issues/274)) ([3273a51](https://github.com/7xuanlu/origin/commit/3273a51943acbb3a9d5e95fbf516baa6bd014e09))
* **llm:** no-truncation invariant for continuous-batch coalescer ([#272](https://github.com/7xuanlu/origin/issues/272)) ([22eae08](https://github.com/7xuanlu/origin/commit/22eae0843c5fcd534eef47d3a042036716db0ad8))

## [0.8.4](https://github.com/7xuanlu/origin/compare/v0.8.3...v0.8.4) (2026-06-14)


### Bug Fixes

* **eval:** CE answer-accuracy 2-arm gate (event_date + power) ([#263](https://github.com/7xuanlu/origin/issues/263)) ([a051807](https://github.com/7xuanlu/origin/commit/a05180793189b23050b43131d7fa65b586713fd1))
* reliable + observable daemon upgrades (restart, version handshake, reranker status, backfill/sweep notices) ([#265](https://github.com/7xuanlu/origin/issues/265)) ([14d2b07](https://github.com/7xuanlu/origin/commit/14d2b07916b3ae68cb4e4b44fd06e722b88cd12a))

## [0.8.3](https://github.com/7xuanlu/origin/compare/v0.8.2...v0.8.3) (2026-06-11)


### Bug Fixes

* **retrieval:** default cross-encoder model bge-reranker-base ([#261](https://github.com/7xuanlu/origin/issues/261)) ([16edb67](https://github.com/7xuanlu/origin/commit/16edb675465a0b2532d7990cb09725372fb7f937))

## [0.8.2](https://github.com/7xuanlu/origin/compare/v0.8.1...v0.8.2) (2026-06-11)


### Bug Fixes

* **eval:** channel_touched G3 attribution probes (de-star graph_stream + rerank arms) ([#258](https://github.com/7xuanlu/origin/issues/258)) ([7eff243](https://github.com/7xuanlu/origin/commit/7eff243a68cd7d28d5c9048af471952a89811eaa))
* **eval:** G3 A/A-floored verdict gate + page-substrate presence floor (Eval-Trust v3) ([#255](https://github.com/7xuanlu/origin/issues/255)) ([823fb15](https://github.com/7xuanlu/origin/commit/823fb15ff94c220bf133c23cc18dc70e6aab1703))
* **eval:** seed orchestrator fails loud on unavailable LLM / zero distilled pages ([#259](https://github.com/7xuanlu/origin/issues/259)) ([3fe652d](https://github.com/7xuanlu/origin/commit/3fe652de829f64850baca6f9867232ab68136c97))
* **provenance:** P3 workspace axis + consolidation lifecycle (demotion eval-gated PASS) ([#256](https://github.com/7xuanlu/origin/issues/256)) ([66a8ab4](https://github.com/7xuanlu/origin/commit/66a8ab4350814fdd3c9941865742408115a408af))
* **retrieval:** CE skip-preference bypass + rerank/graph-stack paired A/B arms ([#251](https://github.com/7xuanlu/origin/issues/251)) ([31ca621](https://github.com/7xuanlu/origin/commit/31ca62121590a8a74e1b8316e91f01a66a47f962))
* **retrieval:** graph memory stream default-on (quick path only) ([#257](https://github.com/7xuanlu/origin/issues/257)) ([df42f7e](https://github.com/7xuanlu/origin/commit/df42f7e18dd64bf18d464d132b30d34e45e75ea6))

## [0.8.1](https://github.com/7xuanlu/origin/compare/v0.8.0...v0.8.1) (2026-06-09)


### Bug Fixes

* **eval:** LME retrieval ingest-parity HOW + graph/intent apparatus ([#10](https://github.com/7xuanlu/origin/issues/10)/[#15](https://github.com/7xuanlu/origin/issues/15)) ([#247](https://github.com/7xuanlu/origin/issues/247)) ([92cb68d](https://github.com/7xuanlu/origin/commit/92cb68d71e0d9fcf920985caf041ba38a013421d))
* **provenance:** P2 typed page_evidence successor table, source-less pages, review_status trust gate ([#249](https://github.com/7xuanlu/origin/issues/249)) ([b245686](https://github.com/7xuanlu/origin/commit/b245686ca844303c7fd3c09495cb14e582e7de1e))

## [0.8.0](https://github.com/7xuanlu/origin/compare/v0.7.0...v0.8.0) (2026-06-08)


### Features

* **eval+server:** foundations P0b — comparable hash + path layout + daemon port discovery ([#190](https://github.com/7xuanlu/origin/issues/190)) ([032ce63](https://github.com/7xuanlu/origin/commit/032ce6316c8bdbcd944a22be3cb0145aa4374534))
* **eval:** foundations P0a — additive types + LatencySummary wiring ([#178](https://github.com/7xuanlu/origin/issues/178)) ([46a9703](https://github.com/7xuanlu/origin/commit/46a9703ad65a528f866682d145ccfa77b7b19f6e))
* **eval:** foundations P0c — cost caps + wall-clock watchdog + save guards + cleanup ([#191](https://github.com/7xuanlu/origin/issues/191)) ([ffbda8f](https://github.com/7xuanlu/origin/commit/ffbda8febe196b2d110e8923d9fdd2153d607315))
* **eval:** foundations P1 — L1 baseline plumbing + citation discipline ([#192](https://github.com/7xuanlu/origin/issues/192)) ([83e23b0](https://github.com/7xuanlu/origin/commit/83e23b05c257e5558e06383ea4cd51c409ce4fec))
* **retrieval:** Plan A foundation — temporal schema + signals + extraction ([#195](https://github.com/7xuanlu/origin/issues/195)) ([0e70cd3](https://github.com/7xuanlu/origin/commit/0e70cd3d7f36a0593669ff353206926b6b9121d0))


### Bug Fixes

* **ci:** poll crates.io sparse index, not v1 web API (was 5min false-fail) ([#184](https://github.com/7xuanlu/origin/issues/184)) ([542129b](https://github.com/7xuanlu/origin/commit/542129b0a18b7afcee210c6ecb93807e08f42929))
* **ci:** rename Homebrew origin CLI tarball + add release archive smoke test ([#197](https://github.com/7xuanlu/origin/issues/197)) ([63fcd4c](https://github.com/7xuanlu/origin/commit/63fcd4c76ecc51859090af65bb35049713aa5047))
* **eval:** assert page-faithfulness negative-controls are flagged ([#239](https://github.com/7xuanlu/origin/issues/239)) ([2b3601a](https://github.com/7xuanlu/origin/commit/2b3601a5f6791a664466accaf3365fa515d2563d))
* **ingest:** opt-in deterministic write-time temporal grounding (ORIGIN_ENABLE_TEMPORAL_GROUNDING) ([#225](https://github.com/7xuanlu/origin/issues/225)) ([0ea8107](https://github.com/7xuanlu/origin/commit/0ea8107f00aacc4993e84a84d501892978d907e5))
* **kg:** deterministic entity-resolution cascade - MinHash/LSH near-dedup (ORIGIN_ENABLE_ENTITY_MINHASH) ([#226](https://github.com/7xuanlu/origin/issues/226)) ([aa35a58](https://github.com/7xuanlu/origin/commit/aa35a58b19a50b07347ca51709cf028b3ec9744e))
* **memory:** two-pool dedup+contradiction resolution in one LLM call (ORIGIN_ENABLE_DUAL_POOL_RESOLVE) ([#228](https://github.com/7xuanlu/origin/issues/228)) ([b93f09b](https://github.com/7xuanlu/origin/commit/b93f09b90014c33634d29c5c35ac59685f8cb86f))
* **pages:** opt-in shrink-guard for LLM page rewrites (ORIGIN_MERGE_SHRINK_GUARD) ([#227](https://github.com/7xuanlu/origin/issues/227)) ([734ac52](https://github.com/7xuanlu/origin/commit/734ac52421d38d3c9dad1700b3e63114239220db))
* **plugin:** MCP runner prefers ~/.origin/bin/origin-mcp over npx ([#199](https://github.com/7xuanlu/origin/issues/199)) ([3f1be92](https://github.com/7xuanlu/origin/commit/3f1be9264f9c2c4edbd8cf56c633c0bc2d9b5f48))
* **provenance:** navigable source provenance in Obsidian projection (P1) ([#246](https://github.com/7xuanlu/origin/issues/246)) ([f93003c](https://github.com/7xuanlu/origin/commit/f93003c7b572af81114b7f82948f2aa116e9771d))
* **refinery:** opt-in archive-not-delete soft eviction (ORIGIN_ENABLE_EVICTION) ([#233](https://github.com/7xuanlu/origin/issues/233)) ([c1759de](https://github.com/7xuanlu/origin/commit/c1759dee078b5ebf0e590d08da92165cc5b3eea0))
* **rerank:** cross-encoder reranker via fastembed (P0 [#1](https://github.com/7xuanlu/origin/issues/1) retrieval lift) ([#187](https://github.com/7xuanlu/origin/issues/187)) ([c5a88b3](https://github.com/7xuanlu/origin/commit/c5a88b314ec32a14c77be642947cdb221b046004))
* **retrieval:** CoT iterative retrieve-reason-retrieve loop (opt-in ORIGIN_ENABLE_COT_RETRIEVAL) ([#235](https://github.com/7xuanlu/origin/issues/235)) ([a3bf3c6](https://github.com/7xuanlu/origin/commit/a3bf3c6785621919690ed1a5974377fedcdb6b76))
* **retrieval:** dual-granularity episode-channel (ORIGIN_ENABLE_EPISODE_CHANNEL) ([#224](https://github.com/7xuanlu/origin/issues/224)) ([395d1ae](https://github.com/7xuanlu/origin/commit/395d1ae58d78222d76efedc322350eb56301e3a6))
* **retrieval:** eval-neutral retrieval cleanup ([#30](https://github.com/7xuanlu/origin/issues/30) [#42](https://github.com/7xuanlu/origin/issues/42) [#48](https://github.com/7xuanlu/origin/issues/48)) ([#208](https://github.com/7xuanlu/origin/issues/208)) ([131421d](https://github.com/7xuanlu/origin/commit/131421df2024308e8395357d15e73557a30535c2))
* **retrieval:** LLM read-time strategy router (opt-in ORIGIN_LLM_ROUTE) ([#236](https://github.com/7xuanlu/origin/issues/236)) ([39c6e4d](https://github.com/7xuanlu/origin/commit/39c6e4d3f235e11728fe232b0bd29ad6b77c66b8))
* **retrieval:** multi-vector per-fact child indexing (ORIGIN_ENABLE_FACT_CHANNEL) ([#229](https://github.com/7xuanlu/origin/issues/229)) ([0b66d0c](https://github.com/7xuanlu/origin/commit/0b66d0c113f5939282864887e75c17ec9b2a7cd3))
* **retrieval:** opt-in FTS recall hardening (ORIGIN_ENABLE_FTS_HARDENING) ([#216](https://github.com/7xuanlu/origin/issues/216)) ([b76ea19](https://github.com/7xuanlu/origin/commit/b76ea19624c937802391813c56527d8925dda274))
* **retrieval:** opt-in graph-activation gate (ORIGIN_ENABLE_GRAPH_GATE) ([#213](https://github.com/7xuanlu/origin/issues/213)) ([fc8679a](https://github.com/7xuanlu/origin/commit/fc8679a07fe554a2781e4fdd4b20a410ca1a2b91))
* **retrieval:** opt-in hierarchical global-context prelude (ORIGIN_ENABLE_GLOBAL_PRELUDE) ([#232](https://github.com/7xuanlu/origin/issues/232)) ([1209879](https://github.com/7xuanlu/origin/commit/120987974c66bbd0c98152ffa4e039d2b4e59aaa))
* **retrieval:** opt-in k-hop entity graph traversal (ORIGIN_ENABLE_GRAPH_KHOP) ([#231](https://github.com/7xuanlu/origin/issues/231)) ([da5a966](https://github.com/7xuanlu/origin/commit/da5a966ccb900e55fd31779903f3b8746e6bdc27))
* **retrieval:** opt-in magnitude-preserving FTS score fusion (ORIGIN_MAGNITUDE_FUSION) ([#218](https://github.com/7xuanlu/origin/issues/218)) ([e60042d](https://github.com/7xuanlu/origin/commit/e60042dae953e2f9fade7058cbfffdfd1993cf7c))
* **retrieval:** opt-in per-session result diversification cap (ORIGIN_ENABLE_SESSION_DIVERSITY) ([#222](https://github.com/7xuanlu/origin/issues/222)) ([3e5cf5a](https://github.com/7xuanlu/origin/commit/3e5cf5aaa733598d713d8e7f660ae81f9cd179f3))
* **retrieval:** opt-in pseudo-relevance feedback retrieval (ORIGIN_PRF_ROUNDS) ([#230](https://github.com/7xuanlu/origin/issues/230)) ([b31a732](https://github.com/7xuanlu/origin/commit/b31a7325c861e21b37e5f9611e4ad16cecd8bff7))
* **retrieval:** opt-in query-adaptive RRF channel reweighting (ORIGIN_ENABLE_QUERY_INTENT) ([#221](https://github.com/7xuanlu/origin/issues/221)) ([d6c883e](https://github.com/7xuanlu/origin/commit/d6c883ed9c4e52f9ab7fb3c913cfdc97c1ab64f3))
* **retrieval:** opt-in wide-pool-seeded graph expansion (ORIGIN_ENABLE_GRAPH_SEED) ([#220](https://github.com/7xuanlu/origin/issues/220)) ([fbf42c4](https://github.com/7xuanlu/origin/commit/fbf42c445197b66c8e6963738e984c8c17cae1a4))
* **retrieval:** page-channel as 4th RRF stream in search_memory_with_reranker ([#203](https://github.com/7xuanlu/origin/issues/203)) ([7d16b41](https://github.com/7xuanlu/origin/commit/7d16b410672a1175c4d2e8d02a65a9e7d77209d3))
* **retrieval:** per-memory salience prior in ranking (ORIGIN_ENABLE_SALIENCE_PRIOR) ([#223](https://github.com/7xuanlu/origin/issues/223)) ([9cdabfe](https://github.com/7xuanlu/origin/commit/9cdabfeaf6d35969d02a40d6ce102eec1308b50f))
* **retrieval:** query decomposition into independent subqueries (search_memory_decomposed) ([#214](https://github.com/7xuanlu/origin/issues/214)) ([25e573a](https://github.com/7xuanlu/origin/commit/25e573ae051f4084fa812b3318fca098446b0063))
* **retrieval:** read-time context compression - opt-in ORIGIN_ENABLE_CONTEXT_COMPRESS ([#237](https://github.com/7xuanlu/origin/issues/237)) ([d2bc318](https://github.com/7xuanlu/origin/commit/d2bc318d346b950fe885b4048753548d7be6f58b))
* **retrieval:** retrieval/ namespace + cherry-pick hard_filters from [#200](https://github.com/7xuanlu/origin/issues/200) + ORDER BY confidence ([#202](https://github.com/7xuanlu/origin/issues/202)) ([95d3644](https://github.com/7xuanlu/origin/commit/95d36448858e9d0e4b7ee40c0c51f1c7c46163dc))
* **retrieval:** soft temporal proximity boost - opt-in ORIGIN_ENABLE_TEMPORAL_SOFT_BOOST ([#240](https://github.com/7xuanlu/origin/issues/240)) ([2b04a34](https://github.com/7xuanlu/origin/commit/2b04a346298783ccd51a126092c0268c0d38664f))
* **retrieval:** wire query-side temporal filter (ORIGIN_ENABLE_TEMPORAL_FILTER) ([#219](https://github.com/7xuanlu/origin/issues/219)) ([068d166](https://github.com/7xuanlu/origin/commit/068d166233c129ae45ae469963934d2d29013553))
* **seo:** crates.io metadata + drop stale darwin-x64 test assertion ([#177](https://github.com/7xuanlu/origin/issues/177)) ([7def4a0](https://github.com/7xuanlu/origin/commit/7def4a082f3804ef26c2ef97b310668983869786))
* **server:** opt-in debounced background reflection (ORIGIN_ENABLE_REFLECTION_DEBOUNCE) ([#234](https://github.com/7xuanlu/origin/issues/234)) ([f1b4539](https://github.com/7xuanlu/origin/commit/f1b4539528b468c937b82e17191143632a4db773))


### Performance Improvements

* **ci:** native ARM Docker build (drop 90min QEMU emulation) ([#185](https://github.com/7xuanlu/origin/issues/185)) ([eefa266](https://github.com/7xuanlu/origin/commit/eefa2668781a045c62b518f059fdf113911a6533))

## [0.7.0](https://github.com/7xuanlu/origin/compare/v0.6.1...v0.7.0) (2026-05-24)


### Features

* **cli:** implement Windows install via schtasks ([#162](https://github.com/7xuanlu/origin/issues/162)) ([ed9b96f](https://github.com/7xuanlu/origin/commit/ed9b96f6a76eaec4a7b2a32dbc6b7debfa9dd48b))
* **cli:** origin space subcommands + doctor resolver state (Plan C) ([#159](https://github.com/7xuanlu/origin/issues/159)) ([fd28fb2](https://github.com/7xuanlu/origin/commit/fd28fb2915364a631531bc8d3bb00fbd06881055))
* cross-platform Linux and Windows support ([#150](https://github.com/7xuanlu/origin/issues/150)) ([e732909](https://github.com/7xuanlu/origin/commit/e7329092884d063172d02a8898cf2b11ae81da29))
* **eval:** KG-faithfulness benchmark (Plan C-B) ([#149](https://github.com/7xuanlu/origin/issues/149)) ([93b9982](https://github.com/7xuanlu/origin/commit/93b998288db6ca76adb371d4adf99886a55374ce))
* **eval:** LLM judge for KG-faithfulness (Plan C-C) ([#152](https://github.com/7xuanlu/origin/issues/152)) ([f09fcf2](https://github.com/7xuanlu/origin/commit/f09fcf2878dea156e30d414082267efb6a5bab1e))
* **eval:** page-distillation faithfulness benchmark (Plan C-D) ([#151](https://github.com/7xuanlu/origin/issues/151)) ([eda861c](https://github.com/7xuanlu/origin/commit/eda861c11d3611d8e4872e3287d7e775bd64bbc2))
* **eval:** reproducibility foundations (Plan A) ([#145](https://github.com/7xuanlu/origin/issues/145)) ([a8424ef](https://github.com/7xuanlu/origin/commit/a8424ef9453dca62c3b4c0ed8bcd801a9e35cefe))
* **eval:** structured binary judge via tool_use ([#164](https://github.com/7xuanlu/origin/issues/164)) ([23dba48](https://github.com/7xuanlu/origin/commit/23dba48dec7bcaedd50f3ba65d00367b4c07d319))
* **plugin:** space resolver + 6-layer chain (Plan A) ([#153](https://github.com/7xuanlu/origin/issues/153)) ([0916e8c](https://github.com/7xuanlu/origin/commit/0916e8c5c83edd2c96b6d2e02d8124de2db12e95))
* **server, mcp:** X-Origin-Space header + tool schema gating (Plan B) ([#156](https://github.com/7xuanlu/origin/issues/156)) ([285d11a](https://github.com/7xuanlu/origin/commit/285d11a78406585e08169c92bc669ac6ee7bac4c))


### Bug Fixes

* **ci:** release.yml — publish-crates correctness + add origin CLI to Homebrew tap ([#163](https://github.com/7xuanlu/origin/issues/163)) ([6780986](https://github.com/7xuanlu/origin/commit/67809868c982cccd39ab20b4ed38bc51569e3ab4))
* **core:** serialize EVAL_MAX_USD env touches in eval_harness tests ([#160](https://github.com/7xuanlu/origin/issues/160)) ([ae9253c](https://github.com/7xuanlu/origin/commit/ae9253c87e72084cc2879362ccfc08ac7a60d93b))
* **docker:** switch daemon image base to Debian trixie ([#158](https://github.com/7xuanlu/origin/issues/158)) ([97642c2](https://github.com/7xuanlu/origin/commit/97642c20857f159d6c34a748de18cafda063cecd))
* **eval:** restore app/eval/fixtures to monorepo for L6 CI canary ([#148](https://github.com/7xuanlu/origin/issues/148)) ([379e2bc](https://github.com/7xuanlu/origin/commit/379e2bc9b961ef4f1a74dae32b2bb8f5831469e4))
* harden MCP setup distribution path ([d6fb5da](https://github.com/7xuanlu/origin/commit/d6fb5daada78a677d2d61dfe2906f19be3597fad))
* tighten version sync validation ([#139](https://github.com/7xuanlu/origin/issues/139)) ([80db565](https://github.com/7xuanlu/origin/commit/80db5652190a07b6fb79579c6fb01d62a116461b))

## [0.6.1](https://github.com/7xuanlu/origin/compare/v0.6.0...v0.6.1) (2026-05-16)


### Bug Fixes

* sync README to npm packages + republish v0.6.0+ with README content ([#137](https://github.com/7xuanlu/origin/issues/137)) ([43e4ce9](https://github.com/7xuanlu/origin/commit/43e4ce966be985ce5b6888a9ff23360e1cc685d9))

## [0.6.0](https://github.com/7xuanlu/origin/compare/v0.5.3...v0.6.0) (2026-05-16)


### Features

* BM-mode consumer-side accept dispatch ([#96](https://github.com/7xuanlu/origin/issues/96)) ([033ce55](https://github.com/7xuanlu/origin/commit/033ce5570f2e56f384c669e053dcfbcf661db822))
* BM-mode curation mutate MCPs (Spec C-2) ([#105](https://github.com/7xuanlu/origin/issues/105)) ([73aec7a](https://github.com/7xuanlu/origin/commit/73aec7a127f31d31a41b518a6ebd6a37c304ccfa))
* rename domain → space + complete e2e scoping (BREAKING CHANGE) ([#123](https://github.com/7xuanlu/origin/issues/123)) ([7281202](https://github.com/7xuanlu/origin/commit/72812025b0dbf73652bd8654f7016633fc2c76ad))


### Bug Fixes

* auto-supersede conflicting relations (last-write-wins) ([#111](https://github.com/7xuanlu/origin/issues/111)) ([eda6718](https://github.com/7xuanlu/origin/commit/eda67180ee8c390e6d5c4ed666c0e1dc24295936))
* bundle quick wins for CI noise + correctness ([#102](https://github.com/7xuanlu/origin/issues/102)) ([edde4c9](https://github.com/7xuanlu/origin/commit/edde4c9e45ebde81c6a607a7a795e54d1e684752))
* capture inline contradiction signal + surface bug fix ([#110](https://github.com/7xuanlu/origin/issues/110)) ([b35e843](https://github.com/7xuanlu/origin/commit/b35e843d0b15595d0ba17e867906e8e2762011dd))
* **ci:** main-canary filter eval::token_efficiency → eval::retrieval ([#124](https://github.com/7xuanlu/origin/issues/124)) ([ecfd386](https://github.com/7xuanlu/origin/commit/ecfd3867164f7cfebe0e384900532f1bd020249b))
* **ci:** split fmt/lint/test on ubuntu, pin toolchain + SHAs ([#117](https://github.com/7xuanlu/origin/issues/117)) ([f5cd75d](https://github.com/7xuanlu/origin/commit/f5cd75d51994725aa9d312711d13439724a45dbe))
* **core:** apply supersedes_exclusion to MemoryDB::search ([#130](https://github.com/7xuanlu/origin/issues/130)) ([0add226](https://github.com/7xuanlu/origin/commit/0add2268201f794bdf0e3f510f6d0d6a27b8b750))
* **core:** cross-process file lock around FastEmbed init ([#125](https://github.com/7xuanlu/origin/issues/125)) ([d7aaaab](https://github.com/7xuanlu/origin/commit/d7aaaab418ea1f0e5dcee084be3a1e89b5873e24))
* **core:** honor ORIGIN_DATA_DIR in spaces.legacy_db_path ([#135](https://github.com/7xuanlu/origin/issues/135)) ([a5c23ee](https://github.com/7xuanlu/origin/commit/a5c23eebb0b7e6cc1995b7fac7586faea1f41b5a))
* **distill:** respect user_edited + thread knowledge_path + add /distill rebuild ([#106](https://github.com/7xuanlu/origin/issues/106)) ([26a7345](https://github.com/7xuanlu/origin/commit/26a734549e3aed141543b1749deb4f025c89fe52))
* gate MCP curation wrappers to stdio transport ([#122](https://github.com/7xuanlu/origin/issues/122)) ([d874907](https://github.com/7xuanlu/origin/commit/d87490775e2d213148d42d1f8dabbcef2731dbe0))
* handoff pending-captures preview + list_pending plumbing (Spec C-3b) ([#114](https://github.com/7xuanlu/origin/issues/114)) ([4fe5fba](https://github.com/7xuanlu/origin/commit/4fe5fbae8a25f49a9f5f288a075cc43dfc41581a))
* handoff status file uses Active/Backlog two-tier split + date stamps ([#116](https://github.com/7xuanlu/origin/issues/116)) ([636e49a](https://github.com/7xuanlu/origin/commit/636e49aa4e62b4df10e0fe22f58940261b79cfa8))
* **kg:** coerce non-vocabulary relation types to related_to + prompt update ([#100](https://github.com/7xuanlu/origin/issues/100)) ([d6cd5d8](https://github.com/7xuanlu/origin/commit/d6cd5d8e9ce64617aa23e466cbea7640773d6979))
* make origin CLI own runtime setup ([#128](https://github.com/7xuanlu/origin/issues/128)) ([4f6d946](https://github.com/7xuanlu/origin/commit/4f6d946153691821c6c7ae13b3529f7f9e47d174))
* **mcp:** list_spaces tool + activate doc-path space filter ([#126](https://github.com/7xuanlu/origin/issues/126)) ([0ed205f](https://github.com/7xuanlu/origin/commit/0ed205fa21ff351ff38d7be6e1bb7a3d48c069ff))
* **mcp:** observation CRUD wrappers (PR-A of bm-mode extraction) ([#95](https://github.com/7xuanlu/origin/issues/95)) ([fda9b63](https://github.com/7xuanlu/origin/commit/fda9b631fc05a990b4d38b469a944654ce6d9fad))
* **refinery:** thread knowledge_path through re_distill_stale_pages ([#108](https://github.com/7xuanlu/origin/issues/108)) ([21a25a0](https://github.com/7xuanlu/origin/commit/21a25a09002373f120184798595227d11065b3be))
* remove /refinery skill (power-user MCPs stay) ([#109](https://github.com/7xuanlu/origin/issues/109)) ([083f458](https://github.com/7xuanlu/origin/commit/083f4580fe9576a35c49f4938463b17c6536f538))
* remove entity-suggestion mutate MCPs (dead scaffolding) ([#113](https://github.com/7xuanlu/origin/issues/113)) ([fe6fe18](https://github.com/7xuanlu/origin/commit/fe6fe182a77c12608f7375e7d4832ecfe375e972))
* **server:** clone Arc&lt;MemoryDB&gt; before await in 3 space-mutate handlers ([#129](https://github.com/7xuanlu/origin/issues/129)) ([226ae8d](https://github.com/7xuanlu/origin/commit/226ae8d8ed0669064320af43745d69b56c56b8ee))
* **server:** clone Arc&lt;MemoryDB&gt; before await in handle_list_memories ([#136](https://github.com/7xuanlu/origin/issues/136)) ([39a600d](https://github.com/7xuanlu/origin/commit/39a600d4bee3babb1a6b978d806e4b0d13ba93be))
* **server:** clone Arc&lt;MemoryDB&gt; before await in remaining handlers ([#131](https://github.com/7xuanlu/origin/issues/131)) ([7236eeb](https://github.com/7xuanlu/origin/commit/7236eeb4f5a2a728ddbfd4d0abff6e77c1d43e35))
* **skills:** /brief reads status file first + /review drops stale C-3b note ([#121](https://github.com/7xuanlu/origin/issues/121)) ([aa9899e](https://github.com/7xuanlu/origin/commit/aa9899e4bd3bd7a12df208ef14a61d02e0281cf0))
* soft-archive supersede_relation via activity payload ([#120](https://github.com/7xuanlu/origin/issues/120)) ([daf9bc2](https://github.com/7xuanlu/origin/commit/daf9bc20a3e5bd081ea0560ec6a569bb04519b94))
* stop emitting dedup_merge + detect_contradiction proposals ([#112](https://github.com/7xuanlu/origin/issues/112)) ([521498d](https://github.com/7xuanlu/origin/commit/521498d97c5df4984b1704abf97b592de0868e13))
* surface pending revisions in /brief + scoped /review walks (Spec C-3 Phase 1) ([#107](https://github.com/7xuanlu/origin/issues/107)) ([54b4e3b](https://github.com/7xuanlu/origin/commit/54b4e3b522416494a5c647e8991810a4b1f93a91))
* trust-tier auto-supersede for high-confidence contradictions ([#115](https://github.com/7xuanlu/origin/issues/115)) ([0c74271](https://github.com/7xuanlu/origin/commit/0c74271e0ba1ec42b1c1b1878db252a86cdfdb4a))

## [0.5.3](https://github.com/7xuanlu/origin/compare/v0.5.2...v0.5.3) (2026-05-13)


### Bug Fixes

* get_page_sources MCP tool + auto-commit retry (close skill ↔ MCP boundary) ([#85](https://github.com/7xuanlu/origin/issues/85)) ([101b595](https://github.com/7xuanlu/origin/commit/101b59535e8a14836801f8a9b5054af387510377))
* memory + page revision surfacing (Phase 1 of Task [#57](https://github.com/7xuanlu/origin/issues/57)) ([#91](https://github.com/7xuanlu/origin/issues/91)) ([02ddd43](https://github.com/7xuanlu/origin/commit/02ddd43d97f7b4a8d83af5ed24c4f15a437455df))
* **memory_routes:** drop silent topic-match upsert from write path ([#84](https://github.com/7xuanlu/origin/issues/84)) ([46175a0](https://github.com/7xuanlu/origin/commit/46175a0dfc433272994def751828daf8f77e72f7))
* **topic_match:** entity match must also satisfy similarity threshold ([#83](https://github.com/7xuanlu/origin/issues/83)) ([0670772](https://github.com/7xuanlu/origin/commit/067077225ac619d8e0c69deb54a9a0d3d4ec2a01))

## [0.5.2](https://github.com/7xuanlu/origin/compare/v0.5.1...v0.5.2) (2026-05-12)


### Bug Fixes

* handoff skill — categorized confirm output + git retry for index.lock ([01f87da](https://github.com/7xuanlu/origin/commit/01f87da9a37cf289c4cc1659c39504dc68f620f4))
* handoff skill — user-friendly labels mapped to daemon memory types ([9c14c2e](https://github.com/7xuanlu/origin/commit/9c14c2e957720293789246acfed4a4e594221ca2))
* handoff skill uses daemon's 6 canonical memory types ([9c74d34](https://github.com/7xuanlu/origin/commit/9c74d34baa291b68c2e2fe63d3b9de3acbeb7ee3))
* MCP wrappers for /api/pages/search + /api/pages/recent ([#77](https://github.com/7xuanlu/origin/issues/77)) ([6fab560](https://github.com/7xuanlu/origin/commit/6fab56012421a6f3b26b8acf601d009dfb53cdf6))
* **pages:** llm-wiki foundations — user_edited, cluster cap, refresh route, wikilink graph, fs watcher ([#78](https://github.com/7xuanlu/origin/issues/78)) ([a611ae1](https://github.com/7xuanlu/origin/commit/a611ae1c21dad56caacdbd93f5ed7b87fae52b72))
* plugin UX — ~/.origin consolidation, version pins, skill upgrades ([#73](https://github.com/7xuanlu/origin/issues/73)) ([4483dd6](https://github.com/7xuanlu/origin/commit/4483dd607ef1c8e3c9cdfd22a72e5ecc92ae606a))
* PR [#73](https://github.com/7xuanlu/origin/issues/73) follow-ups — daemon version hook + Basic Memory skill phases ([#75](https://github.com/7xuanlu/origin/issues/75)) ([b27c0ef](https://github.com/7xuanlu/origin/commit/b27c0efee67d40c5b70403aa90bad92671c799b8))
* reconcile README with PR [#72](https://github.com/7xuanlu/origin/issues/72) structure ([587b26c](https://github.com/7xuanlu/origin/commit/587b26cf2c532fd67898fec7b829148876634714))
* remove duplicate Repo Map section from README ([f5b946e](https://github.com/7xuanlu/origin/commit/f5b946e41f33d225dede004213ef5f78cd96791e))
* update release-please git-add paths for plugin/ subdir migration ([c711034](https://github.com/7xuanlu/origin/commit/c7110347ad46d6236703035c24575f5258c91799))

## [0.5.1](https://github.com/7xuanlu/origin/compare/v0.5.0...v0.5.1) (2026-05-10)


### Bug Fixes

* align README with monorepo runtime ([#72](https://github.com/7xuanlu/origin/issues/72)) ([ce44ceb](https://github.com/7xuanlu/origin/commit/ce44ceb1e6ac62027b0f5b4366b6d69fdab053da))
* **release:** drop origin-mcp from pre-flight dry-run ([a1b804b](https://github.com/7xuanlu/origin/commit/a1b804b4f4e9c1340cae06c3c26a049b85c35e6b))
* **release:** make origin-types publish idempotent ([afde654](https://github.com/7xuanlu/origin/commit/afde6548d0875ea88b620128831f5259a324159c))
* **release:** use RELEASE_TAG in npm + homebrew jobs ([f087f87](https://github.com/7xuanlu/origin/commit/f087f8772b585b187bed1cd9470b00471613ecf1))

## [0.5.0](https://github.com/7xuanlu/origin/compare/v0.3.1...v0.5.0) (2026-05-10)


### Features

* **mcp:** switch to workspace inheritance, Apache-2.0, path dep on origin-types ([52721f9](https://github.com/7xuanlu/origin/commit/52721f9490b921cb74a9088322fa61c0a6203dd5))
* merge origin-mcp + origin-plugin into monorepo (v0.5.0) ([bc95c84](https://github.com/7xuanlu/origin/commit/bc95c846d0a9b8f8381993a996ec26638e79895c))
* merge origin-mcp into monorepo as crates/origin-mcp/ ([c982ec7](https://github.com/7xuanlu/origin/commit/c982ec738930671f9c7e3eb1f24227fa86fab756))
* merge origin-plugin into monorepo (staging) ([0fdfe0e](https://github.com/7xuanlu/origin/commit/0fdfe0eeb7b683942bec75be6166dffef30b8c34))
* **plugin:** update manifest for monorepo (v0.5.0, repository=origin) ([647bab1](https://github.com/7xuanlu/origin/commit/647bab165e21c56437bc613cbf28839282943e89))
* **scripts:** add validate-versions.sh pre-flight check ([94587d2](https://github.com/7xuanlu/origin/commit/94587d2fc01befafcfe22c5e9dac79227b0abf2c))
* **scripts:** extend bump-version.sh to sync npm + plugin manifests ([cb26bc4](https://github.com/7xuanlu/origin/commit/cb26bc484c51d8a3aabb95647997c08b238f9803))


### Bug Fixes

* bump npm/package.json + Cargo to 0.4.1 (sync after v0.4.0 npm publish skip) ([a8f6a59](https://github.com/7xuanlu/origin/commit/a8f6a5920469d25498c2ca4ee39f63a4363e05b3))
* **ci,npm:** align npm syntax-check paths and metadata ([4c4e240](https://github.com/7xuanlu/origin/commit/4c4e2408515ff158c99572ff6b6ed7295052e9a4))
* **ci:** quote rust job if-expression to fix YAML parse ([553eed7](https://github.com/7xuanlu/origin/commit/553eed770a78d5733f57155bab8434e12f100308))
* **mcp:** suppress deprecated field warnings for include_goals + goals ([4dad838](https://github.com/7xuanlu/origin/commit/4dad83865732b069d853ca698a18504dd93933ef))
* replace placeholder skills with locked verb set (init/brief/capture/recall/distill/review/forget/handoff) ([196dc75](https://github.com/7xuanlu/origin/commit/196dc7594b19f9d1e3205df698ccbd3bd9d8929a))

## [0.3.1](https://github.com/7xuanlu/origin/compare/v0.3.0...v0.3.1) (2026-05-07)


### Bug Fixes

* /api/llm/test endpoint + app proxy (Phase 5 PR3 — recreated) ([#60](https://github.com/7xuanlu/origin/issues/60)) ([64805ee](https://github.com/7xuanlu/origin/commit/64805eedb1025177e0e43f095d67c854a0039b83))
* origin-cli crate with subcommands (Phase 3 PR2) ([#54](https://github.com/7xuanlu/origin/issues/54)) ([3c9a60f](https://github.com/7xuanlu/origin/commit/3c9a60f9efba6ab5f8b0355f6625a290948ecf09))
* Phase 5-D PR1 foundation — wire types + config fields + system_info inline ([#61](https://github.com/7xuanlu/origin/issues/61)) ([734920f](https://github.com/7xuanlu/origin/commit/734920f4a72a9548ca0db778d59abca770c2ad32))
* Phase 5-D PR2 — drop origin-core dep from app crate ([#62](https://github.com/7xuanlu/origin/issues/62)) ([7ffda88](https://github.com/7xuanlu/origin/commit/7ffda88fdb3ac03717590632f3961614c801ef5d))

## [0.3.0](https://github.com/7xuanlu/origin/compare/v0.2.1...v0.3.0) (2026-05-05)


### Features

* rename Concept → Page + expand MemoryType taxonomy ([4b91089](https://github.com/7xuanlu/origin/commit/4b91089f305d7e3f43ef49a4a0b8ddd44c4e8ab1))


### Bug Fixes

* add branded Origin setup flow ([dd4208a](https://github.com/7xuanlu/origin/commit/dd4208a85996d6487045463161aa9466e1c39e45))
* **updater:** avoid temp LaunchAgent paths ([#48](https://github.com/7xuanlu/origin/issues/48)) ([14af3e4](https://github.com/7xuanlu/origin/commit/14af3e4321948952376da767830c2bf4afca3041))

## [0.2.1](https://github.com/7xuanlu/origin/compare/v0.2.0...v0.2.1) (2026-05-03)


### Bug Fixes

* **updater:** emit release updater metadata ([#43](https://github.com/7xuanlu/origin/issues/43)) ([ceedde4](https://github.com/7xuanlu/origin/commit/ceedde4a82979838b0f56e586e77722e0c0b16f0))

## [0.2.0](https://github.com/7xuanlu/origin/compare/v0.1.4...v0.2.0) (2026-05-03)


### Features

* Tauri auto-updater + pnpm update-all + UX polish ([#30](https://github.com/7xuanlu/origin/issues/30)) ([d898d6b](https://github.com/7xuanlu/origin/commit/d898d6b901ee85c2adbb7d31dbc4624dd2f016a2))
* tray + lifecycle decoupling (LSUIElement + launchd) ([#39](https://github.com/7xuanlu/origin/issues/39)) ([b185967](https://github.com/7xuanlu/origin/commit/b185967cf17e1b1f49824aa93bc358f528616c03))


### Bug Fixes

* **ci:** use RELEASE_TOKEN for version-sync push in release-please ([#24](https://github.com/7xuanlu/origin/issues/24)) ([585b7b0](https://github.com/7xuanlu/origin/commit/585b7b02549ce4b3aeaf9f30bb62d0e3d4f72dd0))
* **db:** insert_concept dual-writes concept_sources at creation ([#37](https://github.com/7xuanlu/origin/issues/37)) ([239fe35](https://github.com/7xuanlu/origin/commit/239fe35bbc96782cbff3eb79d1cf28f529030111))
* **db:** migration 44 backfills concept_sources from source_memory_ids JSON ([#36](https://github.com/7xuanlu/origin/issues/36)) ([7791811](https://github.com/7xuanlu/origin/commit/77918111791f124569e0d11b974e209dd001ed20))
* eval module restructure + Batch API integration ([#27](https://github.com/7xuanlu/origin/issues/27)) ([2ca12c1](https://github.com/7xuanlu/origin/commit/2ca12c1d10a2880bcf8056a7afd3d9b8fcefdc6e))
* **eval:** default enrichment to on-device, drop dead token_efficiency.rs ([b23f0bd](https://github.com/7xuanlu/origin/commit/b23f0bda7fbebeffde5b4a49af92e0a30ba4cd17))
* **eval:** EVAL_BASELINES_DIR env var for worktree-agnostic cache sharing ([#33](https://github.com/7xuanlu/origin/issues/33)) ([f855932](https://github.com/7xuanlu/origin/commit/f855932a761858372d7242fa4007833173efadbc))
* **eval:** per-scenario DBs in full-pipeline eval (LoCoMo + LME) ([#32](https://github.com/7xuanlu/origin/issues/32)) ([2566c61](https://github.com/7xuanlu/origin/commit/2566c612e5eae5c3dbcd491c3ac9b44dd543b57f))
* full-pipeline eval + source overlap concept gate + 3 production bugs ([#29](https://github.com/7xuanlu/origin/issues/29)) ([e8923b7](https://github.com/7xuanlu/origin/commit/e8923b720750de434e257378f829ddcc0d16bf79))
* **hooks:** pre-push skips clippy + tests + coverage on docs-only changes ([#26](https://github.com/7xuanlu/origin/issues/26)) ([e3c0124](https://github.com/7xuanlu/origin/commit/e3c0124831ffd446121c942bb37dc40a275cc480))
* **updater:** in-app toast overlay + Settings version footer ([#40](https://github.com/7xuanlu/origin/issues/40)) ([b51d244](https://github.com/7xuanlu/origin/commit/b51d24477b5668f615c849149c4a663d5e4ffb20))

## [0.1.4](https://github.com/7xuanlu/origin/compare/v0.1.3...v0.1.4) (2026-04-26)


### Bug Fixes

* **ci:** skip CI on release-please merge commits ([7c74be7](https://github.com/7xuanlu/origin/commit/7c74be78ddbe56676a71eb0d4052f0234a8a1c84))
* **distill:** prevent generic-title and runaway-cluster concepts ([#23](https://github.com/7xuanlu/origin/issues/23)) ([c3ff292](https://github.com/7xuanlu/origin/commit/c3ff292859d25a7d877afade9be322128cf2d04d))
* enrichment status honesty -- per-step tracking + self-healing ([#9](https://github.com/7xuanlu/origin/issues/9)) ([1f18813](https://github.com/7xuanlu/origin/commit/1f1881392c08018e7c99579b1b7bbd8d4411894d))
* **hooks:** run targeted clippy in pre-commit, not just cargo check ([a99681c](https://github.com/7xuanlu/origin/commit/a99681c753a4f68ba3cb5785d50d2923f1b2c694))
* **quality-gate:** fail closed when embedding fails, not open ([8661a80](https://github.com/7xuanlu/origin/commit/8661a803cf0c4f269f1fe2366b961411ac088f42))
* remove useless format\! in refinery.rs ([4ae9195](https://github.com/7xuanlu/origin/commit/4ae9195a1f0b24d31317617230813a944f55c6a0))
* self-healing title re-enrichment for truncated titles ([#22](https://github.com/7xuanlu/origin/issues/22)) ([28b731c](https://github.com/7xuanlu/origin/commit/28b731cbbba0702cf9c55dc2caf562ea8deb6823))

## [0.1.3](https://github.com/7xuanlu/origin/compare/v0.1.2...v0.1.3) (2026-04-25)


### Bug Fixes

* **ci:** add workflow_dispatch to release.yml for manual re-trigger ([85c2842](https://github.com/7xuanlu/origin/commit/85c28420587076da33786fdbf2061abe51b0251c))
* **ci:** drop Origin prefix from release name, remove dead config ([aa25245](https://github.com/7xuanlu/origin/commit/aa25245dbe561df2d799f46664e277b4c4c3b953))
* **ci:** single release per version, consistent titles, changelog in body ([d30930b](https://github.com/7xuanlu/origin/commit/d30930b1ff98024737ca837725a38a481bcea028))
* **ci:** use env context for secrets check in workflow_dispatch ([cb84b95](https://github.com/7xuanlu/origin/commit/cb84b954a14af74b0f9184ebcf80bde3fb45c024))
* **ci:** use PAT in release-please so tag push triggers release build ([928ce65](https://github.com/7xuanlu/origin/commit/928ce6508b57e8257e05d41ba5c413280b7872b1))
* **eval:** token efficiency evaluation framework ([#3](https://github.com/7xuanlu/origin/issues/3)) ([311ceea](https://github.com/7xuanlu/origin/commit/311ceea4543f5c02864e03d9fe7d57fa3197ca61))

## [0.1.2](https://github.com/7xuanlu/origin/compare/v0.1.1...v0.1.2) (2026-04-24)


### Bug Fixes

* **app:** actually add fixture-gen feature gate ([aff3ffb](https://github.com/7xuanlu/origin/commit/aff3ffb75639ce70f8a788937a9c9c3d3900264a))
* **app:** gate fixture_gen dev binary behind opt-in feature ([ffa992e](https://github.com/7xuanlu/origin/commit/ffa992ee19b61115cf08628a01d6fe3bde9f16a8))
* **app:** spawn origin-server sidecar by bare name ([6e7f15d](https://github.com/7xuanlu/origin/commit/6e7f15de5c852cf4593fc920ab303d44970b91cc))
* **app:** tee logs to ~/Library/Logs so sidecar errors are visible ([045ebb8](https://github.com/7xuanlu/origin/commit/045ebb82963bff4d331f5df2f4e7ec177421486d))
* auto-format on commit and auto-activate git hooks ([57f6170](https://github.com/7xuanlu/origin/commit/57f617034c753792abe8105ce1559bb78b3a8daf))
* bump version to 0.1.2 ([33df942](https://github.com/7xuanlu/origin/commit/33df9420e72348b2c0a232257f9d449af3ca5950))
* cache FastEmbed ONNX model in CI to prevent flaky test failures ([003299d](https://github.com/7xuanlu/origin/commit/003299d5e04a68aac7f64249d9b60f840478ea16))
* cargo fmt on db.rs test formatting ([b6a6f32](https://github.com/7xuanlu/origin/commit/b6a6f32349aee720be5ede7f1719cf46c441a7bb))
* filter superseded source memories in concept re-distill ([30c90e5](https://github.com/7xuanlu/origin/commit/30c90e58fbd3850f01ce9acf0580e1abeabf4624))
* force next release-please version to 0.1.2 via release-as ([da8b62a](https://github.com/7xuanlu/origin/commit/da8b62a88a62c18d5da6668d67780cea573c8c74))
* force v0.1.2 release-as, document feat: bumps minor pre-1.0 ([7ca2c63](https://github.com/7xuanlu/origin/commit/7ca2c636beaadadad356221b3c841978ad0b4588))
* make feat: bump patch (not minor) while pre-1.0 ([52b147e](https://github.com/7xuanlu/origin/commit/52b147ec34d6b9cd7bf6d8cb284ffa2c5bc7e664))
* **quality-gate:** require 20+ token chars for bearer credential match ([a606636](https://github.com/7xuanlu/origin/commit/a6066360c384430565340a4a1c76411b45a8fd76))
* **quality-gate:** require non-alpha char in bearer token match ([0c3e9a6](https://github.com/7xuanlu/origin/commit/0c3e9a654b61bb0bb41adbb5ab7c8788eb126d0c))
* remove empty APPLE_ID/PASSWORD/TEAM_ID from tauri-action env ([3bc4a9a](https://github.com/7xuanlu/origin/commit/3bc4a9a76fcd7126138db617ece98925ec859d0d))
* skip crates.io publish when CARGO_REGISTRY_TOKEN not set ([1bb6ccc](https://github.com/7xuanlu/origin/commit/1bb6cccb7c4d170d590d49d92096dd39b757bacd))
* vector search for concepts (hybrid vector + FTS + RRF) ([#8](https://github.com/7xuanlu/origin/issues/8)) ([74c8287](https://github.com/7xuanlu/origin/commit/74c828776ba3d547195436328d07b41e1e25abcf))
* **workspace:** move fixture_gen to origin-core so Tauri doesn't bundle it ([8f076a7](https://github.com/7xuanlu/origin/commit/8f076a71846777869b5b10f45b7842d23f3fe397))

## [0.2.0](https://github.com/7xuanlu/origin/compare/v0.1.0...v0.2.0) (2026-04-23)


### Features

* automated release pipeline with release-please ([c9395ac](https://github.com/7xuanlu/origin/commit/c9395ac91601de680766eb13c2c9a89603fb5f45))
* code signing and notarization infrastructure ([f5614e8](https://github.com/7xuanlu/origin/commit/f5614e830f6338b8e2d76a41b5072030a72f24f9))
* **kg:** alias resolution and relation vocabulary query methods ([e5e5913](https://github.com/7xuanlu/origin/commit/e5e59138fcee0310ad806784748f3f20fe3fa727))
* **kg:** alias-based 4-step entity resolution ([69e08de](https://github.com/7xuanlu/origin/commit/69e08def7f55baeb51645cadbe89385b5a2a96ab))
* **kg:** migration 40 - alias table, relation vocabulary, dedup ([cd327ae](https://github.com/7xuanlu/origin/commit/cd327aeda28760bd0dcac216b8a77d174f1f7715))
* **kg:** migration 40 - alias table, relation vocabulary, dedup ([5ef6db4](https://github.com/7xuanlu/origin/commit/5ef6db4ac7e6a24b997d53e8e3bb869443dd5c38))
* **kg:** periodic rethink pass + integration test ([b73a498](https://github.com/7xuanlu/origin/commit/b73a4985bc1e6131ee56cf5b41713eda8dd86d94))
* **kg:** post-store verification checks for entities, concepts, relations ([d76a533](https://github.com/7xuanlu/origin/commit/d76a5337a136d369cfabe7eebd3479a5c859101a))
* **kg:** relation type normalization at ingest, source_memory_id tracking ([34c76cd](https://github.com/7xuanlu/origin/commit/34c76cd7d61f31ce8b2d09c47dc3207554d7941e))
* **kg:** self-healing entity backfill phase in refinery ([298a8d9](https://github.com/7xuanlu/origin/commit/298a8d968f07a21536106e3ae4a5a5a480e58a66))
* **kg:** structured extraction prompt with vocabulary and confidence ([2342841](https://github.com/7xuanlu/origin/commit/23428412324275ecb8f64141ab4984ad8ed271b3))
* knowledge graph quality - extraction, aliases, verification, rethink ([1beb6a3](https://github.com/7xuanlu/origin/commit/1beb6a3d5e3078be7d043a91768bdee7c01ef848))
* knowledge graph quality + chat template fix ([#5](https://github.com/7xuanlu/origin/issues/5)) ([1beb6a3](https://github.com/7xuanlu/origin/commit/1beb6a3d5e3078be7d043a91768bdee7c01ef848))
* topic-key upsert + concept source linking ([#4](https://github.com/7xuanlu/origin/issues/4)) ([84874c1](https://github.com/7xuanlu/origin/commit/84874c1b96644eec7366d934188c52771ac0b5f9))


### Bug Fixes

* apply Qwen chat template in OnDeviceProvider (entities never extracted via API) ([c8a3f84](https://github.com/7xuanlu/origin/commit/c8a3f84d354410ed39aa96022330746d29dbfd2f))
* filter concepts by domain in list endpoint ([ae06a76](https://github.com/7xuanlu/origin/commit/ae06a7686eb50c2d5e4f640392055a6c63a4da11))
* **kg:** critical review fixes - upsert, case-insensitive resolution, idempotent migration ([c42395d](https://github.com/7xuanlu/origin/commit/c42395dd6aa3e656f07b323e2ca841b0502d9523))
* **kg:** rename migration 40 refs to 41 + prevent orphaned aliases ([3d8ecaf](https://github.com/7xuanlu/origin/commit/3d8ecaf817a5185dac0392f8227d562584393e10))
* remove Cargo.toml from release-please extra-files ([cb054a0](https://github.com/7xuanlu/origin/commit/cb054a0fac7f92996dbb549b1a69c706ac3299bd))
* switch release-please to simple type with version markers ([ba46e0b](https://github.com/7xuanlu/origin/commit/ba46e0b0f96f4401a03b1ed4201737913d252de9))
* use node release-type for cargo workspace compatibility ([480c545](https://github.com/7xuanlu/origin/commit/480c545d9a21d34c52d66cba91dfa276d1756c25))
