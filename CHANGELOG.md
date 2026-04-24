# Changelog

## [0.3.0](https://github.com/7xuanlu/origin/compare/v0.2.0...v0.3.0) (2026-04-24)


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
* CI lint failures + harden release pipeline and branch protection ([65411d2](https://github.com/7xuanlu/origin/commit/65411d2a21eeb6d3e960ae1e84b8fce4e2e872d9))
* filter concepts by domain in list endpoint ([ae06a76](https://github.com/7xuanlu/origin/commit/ae06a7686eb50c2d5e4f640392055a6c63a4da11))
* **kg:** critical review fixes - upsert, case-insensitive resolution, idempotent migration ([c42395d](https://github.com/7xuanlu/origin/commit/c42395dd6aa3e656f07b323e2ca841b0502d9523))
* **kg:** rename migration 40 refs to 41 + prevent orphaned aliases ([3d8ecaf](https://github.com/7xuanlu/origin/commit/3d8ecaf817a5185dac0392f8227d562584393e10))
* make feat: bump patch (not minor) while pre-1.0 ([54cb9f2](https://github.com/7xuanlu/origin/commit/54cb9f2dc5a01db9d5d39347c47ba1c431625bf0))
* remove Cargo.toml from release-please extra-files ([cb054a0](https://github.com/7xuanlu/origin/commit/cb054a0fac7f92996dbb549b1a69c706ac3299bd))
* remove empty APPLE_ID/PASSWORD/TEAM_ID from tauri-action env ([6b48dcc](https://github.com/7xuanlu/origin/commit/6b48dcc97d237f5008789816effcaac8ee9da624))
* reset version to 0.1.1 (v0.2.0 was premature) ([d421773](https://github.com/7xuanlu/origin/commit/d4217730fa9b9aa527577fe155e904b33bf0ee8e))
* skip crates.io publish when CARGO_REGISTRY_TOKEN not set ([26f458c](https://github.com/7xuanlu/origin/commit/26f458c5b56a9f209ef05d38efde1e15638c1f4c))
* switch release-please to simple type with version markers ([ba46e0b](https://github.com/7xuanlu/origin/commit/ba46e0b0f96f4401a03b1ed4201737913d252de9))
* sync versions to 0.2.0 and harden release version pipeline ([214fe1a](https://github.com/7xuanlu/origin/commit/214fe1adb77b4b6f359190aa23b728628400098f))
* use ad-hoc signing when no Apple Developer identity configured ([57143d7](https://github.com/7xuanlu/origin/commit/57143d7bc5fd4384619dd08a681b2a14fc10f3ad))
* use node release-type for cargo workspace compatibility ([480c545](https://github.com/7xuanlu/origin/commit/480c545d9a21d34c52d66cba91dfa276d1756c25))
* version validation sed regex matches trailing comment ([8891925](https://github.com/7xuanlu/origin/commit/8891925a5926e27da945eba979aa24a1bb9493b0))

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
