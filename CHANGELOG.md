# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  - `examples/advanced_quantization.rs` demonstrating the API
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
