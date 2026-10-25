# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.5] - 2026-10-25

### Added
- `mlmf-gguf`: `parse_raw_imatrix`, `RawImatrix`, `RawImatrixEntry`, `RawImatrixError` read `llama-imatrix`'s raw positional layout (fuel row 42 slice E1). Pure, no I/O; every length is bounded against the remaining bytes before allocation. Values come back as stored: unlike fuel's reader this does not divide by `ncall`; `ncall`, the trailer's `last_call` and the dataset name are returned beside them. Checked against a real 1.3 MB file (`bartowski/Llama-3.2-1B-Instruct-GGUF`) with facts measured by an independent reader; the test announces `SKIPPED` when the file is absent.

## [0.7.4] - 2026-10-25

### Added
- `mlmf-safetensors`: `encode_safetensors`, `encode_header`, `TensorToWrite`, `EncodeError`, `name_of`. Lays tensors already encoded in their file dtype into a safetensors image (fuel row 42 slice F1, CireSnave 2026-10-25). Pure, no I/O: the caller writes the bytes. Tensors are ordered widest dtype first then by name so each starts aligned; length, duplicate-name, `__metadata__` and overflow errors refuse rather than write. Checked in `mlmf-conformance` against the upstream `safetensors` crate in both directions.

## [0.7.3] - 2026-10-25

### Changed
- `ROADMAP.md`: rows S-54..S-56 (ONNX writing, `.pt2`, StableHLO) from CireSnave's board 282 ruling. Docs only.

## [0.7.1] - 2026-10-25

### Changed
- `ROADMAP.md`: rows S-47..S-53 record CireSnave's ruling on fuel row 42 (slices E and F come to MLMF as file I/O; slice C and the i-quant decoders go to ml-convertant). Docs only.

## [0.7.0] - 2026-10-24

### Removed (breaking)
CireSnave, 2026-10-24: MLMF's code must add to its ability to read or write model files,
without interpreting or guessing about them. The rule is now in `CLAUDE.md`. Removed because
it does not:
- Multi-modal support: `multimodal`, `multimodal_loader`, `multimodal_processor` (cross-modal
  attention, fusion, `MultiModalModel::infer`), their exports and examples. Fixes #143.
- Distributed loading: `distributed`, `distributed_core`, `distributed_loader`, including
  `ShardingStrategy` and its `ModalitySpecific` variant, and the distributed examples.
- Post-training quantization: `quantization` (`QuantizationEngine`, calibration datasets,
  activation statistics) and its examples. The metadata types that DESCRIBE a file's
  quantization (`ModelQuantizationInfo`, `CalibrationMethod`) stay.
- Model caching: `cache`, `cached_loader` (`ModelCache`, `CachedModelLoader`).
- Model cards: `model_card` (`ModelCardGenerator` and its types), and its example.
- The Feature 6 and Feature 8 summary docs.

The removed code is at `ciresnave/mlmf` commit `bfc84ac`, for any project that wants it.
`impl Display for Architecture`, which lived in `model_card`, moved to `name_mapping`.

### Changed
- All inter-crate `version` requirements are 0.7.0.
- `ROADMAP.md`: the integrate-or-retire section is replaced by CireSnave's ruling that AWQ
  loading and saving stay in MLMF, with an inventory of what exists.

## [0.6.22] - 2026-10-24

### Documentation
- `ROADMAP.md`: spec-to-roadmap audit. 46 rows for work the specs, plans and summary docs
  describe and the roadmap did not list (25 need a CireSnave call), 13 rows for docs that
  contradict the code, and a list of items verified done. Docs only; nothing started.

## [0.6.21] - 2026-10-24

### Changed
- `ONNXLoader::infer_model_config` (private) drops its never-read `name_mapper`
  parameter. The `modality` parameters of `load_distributed_modality` and
  `load_single_modality`, `deploy_shard_to_node`'s `model_path` and `select_node`'s
  `request` are underscore-prefixed as planned call sites. No behaviour change.
  Resolves 5 of the remaining 6 `unused-parameters` findings; the sixth,
  `process_modality`'s `model`, is a live no-op filed as an issue.

## [0.6.20] - 2026-10-24

### Fixed
- `save_as_onnx` ignored its `export_options`: it built fresh writer options and never read
  the argument. `opset_version` and `custom_metadata` now reach the writer;
  `optimize_graph` and `preserve_metadata`, which the writer cannot honour, are refused
  with an error instead of accepted and dropped. Behaviour change: `convert` to ONNX with
  `preserve_metadata: true` now returns that error (the ONNX writer itself still refuses
  to write a graph, so no export that succeeded before fails now).

## [0.6.19] - 2026-10-24

### Fixed
- `save_as_gguf` wrote the model's tensors as an empty map, ignoring its `model`,
  and returned `Ok` with a tensor-less file. It now writes `model.raw_tensors`, and
  refuses with an error when there are none. Behaviour change: a call that used to
  return `Ok` on an empty model now returns `Err`.

## [0.6.18] - 2026-10-24

### Changed
- `distributed_loader.rs` (removed in 0.7.0): the nine parameters of the private `todo!()` stubs
  (`deploy_model`, `scale_cluster`, `NodeManager::new`, `ShardManager::new` and
  `migrate_shard`, `LoadBalancer::new`, `HealthMonitor::new`) are underscore-prefixed.
  Resolves 9 of the root crate's 17 `unused-parameters` findings; no behaviour change,
  not breaking. The other 8 are triaged in `ROADMAP.md`.

## [0.6.17] - 2026-10-24

### Added
- `crates/mlmf-exl2`, `mlmf-core`, `mlmf-gguf` and `mlmf-safetensors` each gain a
  `.coderipper.toml` declaring the file holding their flagged `reachability`
  items as published API (9 findings, verified clean with CodeRipper 0.9.14).
  `ROADMAP.md` now records the root crate's 17 `unused-parameters` and 1
  `unused-return-values` findings. Config and docs only; not breaking.

## [0.6.16] - 2026-10-24

### Fixed
- `.coderipper.toml`: the `mlmf-awq`, `mlmf-gptq` and `mlmf-hf-tokenizer` public-API
  declarations now live in each crate's own `.coderipper.toml`, which is the file a
  per-member check reads; 0.6.14's root-file entries had no effect. Repinned the
  `portfolio.yml` caller to d14a22b (CodeRipper 0.9.14). Config and CI only; not breaking.

## [0.6.15] - 2026-10-23

### Added
- `ROADMAP.md`: recorded that `mlmf-awq`, `mlmf-gptq` and `mlmf-hf-tokenizer`
  are each independently published to crates.io with real downloads but
  have no in-repo consumer (root `mlmf`'s `awq` feature is a stub; its
  `tokenizers` feature wraps the external crate). Facts and two options
  (integrate vs. retire) only, no decision — flagged to CireSnave's board
  per the PM. Documentation only; not breaking.

## [0.6.14] - 2026-10-23

### Added
- `.coderipper.toml`: declared `mlmf-awq`, `mlmf-gptq` and `mlmf-hf-tokenizer`'s
  public API under `[reachability] public_api` (7 files, 64 of the 73
  `reachability` findings from the per-member portfolio sweep). All three
  crates are independently published to crates.io; their entry points
  (`parse_standalone`/`parse_from_model_config`/`locate_layers`/`parse`) are
  exercised only by each crate's own unit tests and have no in-repo
  consumer yet (root `mlmf`'s `awq` feature is still a stub; its
  `tokenizers` feature wraps the external crate, not `mlmf-hf-tokenizer`).
  Configuration only; no code change, not breaking.

## [0.6.13] - 2026-10-23

### Fixed
- `ROADMAP.md`: a peer notice had reported 17 `unused-parameters` findings;
  recounting raw finding lines directly from the sweep job log on
  `origin/main` found 0 for both `unused-parameters` and
  `unused-return-values`. The PM traced the "17" to an earlier,
  different-method baseline (root-package-only, pre-workspace-split) and
  confirmed 0/0 for the current per-member sweep; removed as a backlog item.
  `reachability`'s count (73) was re-verified and is unchanged.

## [0.6.12] - 2026-10-23

### Added
- `ROADMAP.md`: a file-length row per Codacy finding on PR #132 (15
  files, biggest non-comment-LOC first, none started) — pulled from
  the check run's own annotations rather than a peer notice that had
  undercounted at 8.

## [0.6.11] - 2026-10-23

### Fixed
- `lint-suppression` baseline rows (ROADMAP.md): 12 of 14 `#[allow(...)]`
  attributes across src and tests now carry a `reason = "..."`; the other
  2 (`src/validation.rs`'s `unreachable_patterns` allows) had the
  attribute removed outright after `cargo check` confirmed, with the
  attribute removed, that neither produces a warning — the suppression
  was unneeded, not just unexplained.
- `licence` baseline row (ROADMAP.md), partial: every git-tracked `.rs`
  file (165) now carries `// SPDX-License-Identifier: MIT OR Apache-2.0`
  as its first line. `LICENSE-MIT`'s copyright-holder wording is
  unchanged, pending board 263.

## [0.6.10] - 2026-10-23

### Added
- `ROADMAP.md`: one row per failing rule from the first portfolio-wide
  CodeRipper sweep (PR #130) on `origin/main`, plus a note on the sweep's
  own `protoc`-related infrastructure defect (reported to the PM; fix
  belongs in `ciresnave/coderipper`, not here).

## [0.6.9] - 2026-10-23

### Added
- `.github/workflows/ci.yml`: an advisory-only, non-blocking `portfolio` job
  that runs CodeRipper's shared portfolio check (CireSnave, 2026-10-23:
  "Roll out advisory-only, non-blocking CodeRipper checks everywhere.").
  Not a required status check, and does not silence any existing rule.

## [0.6.8] - 2026-10-23

### Fixed
- `.github/workflows/ci.yml`'s `cargo metadata --locked` step moved to run
  first in `core-gates`, before any unlocked cargo invocation. It used to
  run fourth (after `cargo fmt`, two `cargo test -p mlmf-core` steps, and
  `scripts/check-deps.sh`), all of which resolve without `--locked` and so
  were free to silently rewrite the in-runner `Cargo.lock` before the
  dedicated check ever saw it — which is exactly how 0.6.7's own
  manifest/lockfile drift (fixed in #128) shipped on `main` with every
  `core-gates` leg green. Measured directly: restoring the committed
  drifted pair and running `cargo metadata --locked` first fails (exit
  101) as expected; running an unlocked `cargo test -p mlmf-core` first
  silently flips all 14 lockfile version lines to match the manifest, and
  the same `--locked` check then passes.
- Added a final `git diff --exit-code Cargo.lock` step at the end of
  `core-gates`, after every other cargo invocation in that job, as
  defense in depth: the moved check stops an existing drift from being
  masked, but nothing before stopped a later unlocked step in the same
  run from introducing a new one after the check passed.

## [0.6.7] - 2026-10-23

### Fixed
- `Cargo.lock` had drifted from `Cargo.toml`: every `mlmf-*` crate's
  recorded lock version still read `0.6.5` after the workspace version was
  bumped to `0.6.6`. `cargo update -p mlmf` resyncs it to this release's
  version.

## [0.6.6] - 2026-10-21

### Added
- `NOTICE` at the workspace root (also shipped inside `mlmf-ggml` and
  `mlmf-gguf`'s packages) crediting `ggml-org/llama.cpp` and `ggml-org/ggml`
  (MIT) for the type-geometry and required-key/version-support tables those
  two crates measure from and quote, with the upstream licence text copied
  from their LICENSE files and a README Acknowledgements section pointing
  to it.
- `crates/mlmf-ggml/src/types.rs` and `crates/mlmf-gguf/src/header.rs` now pin
  the exact upstream commit (`9d57ce456c94d241dde672b2db9cf18879766568`) their
  comments cite, instead of a file:line reference alone; `header.rs` also
  now names which parts of its header checks correspond to the GGUF spec's
  `## Version History` and which (the byte-swap heuristic) have no spec
  section at all. Not breaking: doc comments only, no behaviour change.

## [0.6.5] - 2026-10-21

### Security
- `rustls-webpki` 0.103.15 -> 0.103.16 (RUSTSEC-2026-0382: certificate
  verification may accept distrusted intermediates).
- `webpki-roots` 1.0.9 -> 1.0.10 (RUSTSEC-2026-0381: bundled roots trusted
  distrusted emSign intermediates).
- `rustls` 0.23.45 -> 0.23.46 (RUSTSEC-2026-0346, -0347, -0348, -0349).
  Dependency-only; no code change, not breaking.

## [0.6.4] - 2026-10-19

### Added
- `.coderipper.toml` declares the published API under `[reachability] public_api` (21 files). Configuration only; no code change, not breaking.

## [0.6.3] - 2026-10-18

### Added
- `GLOSSARY.md`: 53 terms the crates and docs use, each with a source pointer; linked from the README. Documentation only; not breaking.

## [0.6.2] - 2026-10-16

### Security
- CI hardening (Q14): GitHub Actions in `.github/workflows/ci.yml` are pinned to
  full commit SHAs, the workflow declares least-privilege `permissions: contents: read`,
  and `.github/CODEOWNERS` names a default owner. No code change; not breaking.

## [0.6.1] - 2026-10-08

### Fixed
- `TensorNameMapper` now recognizes qwen3's QK-Norm components -- GGUF
  `attn_q_norm.weight`/`attn_k_norm.weight` and HF/SafeTensors
  `self_attn.q_norm.*`/`self_attn.k_norm.*` (#96). Previously these hit the
  unrecognized-component refusal added in #97/#98; they are now ordinary
  recognized renames, consistent with the existing `q_proj`/`k_proj`/`v_proj`
  entries. Not a breaking change -- purely additive recognition.

## [0.6.0] - 2026-10-08

### Changed
- **Breaking:** `HFConfig`'s `vocab_size`, `hidden_size`, `num_attention_heads`
  and `num_hidden_layers` are now `Option<usize>` instead of `usize` (#118).
  An absent `vocab_size` previously decoded to `0`, indistinguishable from a
  file that declared `vocab_size: 0`; the other three were non-`Option`, so
  one absent key failed deserialization of the whole `config.json`.
  `HFConfig::to_model_config` now reports a missing field by name, and
  distinguishes "not declared" from "declared 0" with different error
  messages.

## [0.2.1] - 2024-11-12

### Added
- Added CHANGELOG.md to track version changes
- Added CONTRIBUTING.md with development guidelines
- Added comprehensive documentation for quantization API

### Fixed
- Updated README examples to use candlelight imports
- Fixed version specification in README examples
- Suppressed missing documentation warnings for cleaner builds

## [0.2.0] - 2024-11-12

### Added
- **Quantization API**: Complete post-training quantization (PTQ) support
  - Multiple quantization types: Int8, Int4, Mixed, Dynamic, Static
  - Calibration methods: minmax, percentile, entropy, KL divergence
  - Advanced features: block-wise quantization, symmetric quantization
  - Progress reporting and comprehensive configuration options
  - `advanced_quantization.rs` example (removed in 0.7.0) demonstrating the API
- **Candlelight Integration**: Migrated from direct candle dependencies to unified candlelight wrapper
  - Simplified dependency management
  - Consistent API across all Candle ecosystem crates
  - Better version compatibility

### Changed
- **BREAKING**: All `candle_core` imports must be changed to `candlelight`
- **BREAKING**: `LoadOptions` now includes `preserve_quantization` field
- **BREAKING**: `ModelConfig` now requires `raw_config` field
- Updated all examples to use candlelight imports
- Fixed all compilation errors after candlelight migration

### Fixed
- Resolved 51 compilation errors in quantization module
- Fixed doctest compilation issues across multiple files
- Corrected tensor name mapping for quantized models
- Fixed device validation and dtype checking

### Documentation
- Added comprehensive quantization API documentation
- Updated README with candlelight usage examples
- Fixed all doctest examples to use new API

## [0.1.0] - 2024-11-01

### Added
- Initial release of MLMF (Machine Learning Model Files)
- Support for multiple model formats: SafeTensors, GGUF, ONNX, PyTorch, AWQ
- Architecture detection for LLaMA, GPT-2, GPT-NeoX model families
- Dynamic tensor name mapping between formats
- Memory-efficient loading with memory mapping support
- Progress reporting system
- Comprehensive error handling and validation
- Model conversion capabilities
- Multi-modal support framework
