# Changelog

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
