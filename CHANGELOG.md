# Changelog

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
