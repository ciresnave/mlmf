# mlmf roadmap

Baseline findings from the portfolio-wide CodeRipper sweep (PR #130, first
run on `origin/main` at `c5ae1c2f`, 2026-10-23, run
https://github.com/ciresnave/mlmf/actions/runs/40008808856). Advisory-only
and non-blocking (CireSnave, 2026-10-23); nothing here gates a merge. One row
per failing rule — see that run's log for the full per-finding list.

A second sweep (after `ciresnave/coderipper`#127 repinned `portfolio.yml` to
sweep a cargo workspace member by member, 2026-10-23, run
https://github.com/ciresnave/mlmf/actions/runs/40024100747, on PR #132's
branch) surfaced `reachability` findings it could not see before (added as
its own row below) and confirmed the `lint-suppression` and most of the
`licence` fixes landed: "12 allow and 0 ignore attribute(s) carry a reason
and pass", "5 of 169 source files carry no SPDX-License-Identifier" (down
from 169 of 169).

| Rule | Findings | Status | Notes |
|---|---|---|---|
| `SUP-004` (file/dependency reach outside its own boundary) | 11 | open | 2 git deps (`candlelight`, `fuel-formats`) and 9 `../../` test-file reads outside their crate. |
| `lint-suppression` (allow/ignore with no reason) | 14 (corrected from an earlier miscount of 12 — recounted directly off the sweep log) | DONE | 12 of 14 got a `reason = "..."`; the other 2 (`src/validation.rs`, both `unreachable_patterns`) had their `#[allow(...)]` removed outright after `cargo check` with the attribute removed showed no warning either way — the suppression was already unnecessary, so a fabricated reason would have been worse than deleting it. Confirmed by the second sweep: 12 passing, 0 failing. |
| `licence` | 2 | SPDX done, copyright wording open | SPDX: every git-tracked `.rs` file (165, via `git ls-files`) plus the 3 tracked `.py` files and 2 tracked `.sh` files (CodeRipper's "source file" count of 169 includes those 5, which `git ls-files '*.rs'` alone does not) now carry an SPDX comment. The second sweep, run before the `.py`/`.sh` fix was pushed, still showed "5 of 169" missing — exactly those 5; not yet re-confirmed at 0. Copyright-holder wording in `LICENSE-MIT` waits on board 263 (CireSnave). |
| `reachability` (no caller found within its own crate) | 73 across `mlmf-awq` (23), `mlmf-gptq` (23), `mlmf-hf-tokenizer` (18), `mlmf-exl2` (4), `mlmf-core` (3), `mlmf-gguf` (1), `mlmf-safetensors` (1) | 73 of 73 DONE (declared) | `mlmf-awq`/`mlmf-gptq`/`mlmf-hf-tokenizer` (64 findings) all trace back to 1-2 public entry points each (`parse_standalone`/`parse_from_model_config`/`locate_layers`/`parse`) that have no in-repo caller — verified all three crates are independently published to crates.io (checked the crates.io API directly, 2026-10-23: `mlmf-awq` 0.5.11/30 downloads, `mlmf-gptq` 0.5.11/30, `mlmf-hf-tokenizer` 0.5.11/20) as standalone "bytes to structure" readers per this repo's own charter quote, each exercised only by its own `#[cfg(test)]` unit tests (which this check, by design, doesn't count as production reachability — unlike a separate `tests/*.rs` file, example or bench). None has an in-repo consumer: root `mlmf`'s `awq` feature is still `awq = []`, and its `tokenizers` feature wraps the external `tokenizers` crate, not `mlmf-hf-tokenizer`. **Declared in `.coderipper.toml`, not deleted** — the code is correct, documented and tested, just unintegrated, the same judgment call this row already called out ("a planned call site"). The remaining 9 (`mlmf-exl2` 4, `mlmf-core` 3, `mlmf-gguf` 1, `mlmf-safetensors` 1) are the same shape — `pub` items of published crates (`cargo search`, 2026-10-24: 0.5.11 each) that no in-repo non-test target calls — and are declared the same way, in each crate's own `crates/<name>/.coderipper.toml` (a per-member check reads only that file). Verified with `coderipper` 0.9.14 (the version CI installs): each of the four reports "no findings (n waived)". |
| `unused-parameters` / `unused-return-values` | **17 / 1** at `8481863` (was 0 / 0 before the protoc fix) | open | Counted off the sweep log of run 40068424658 (`coderipper` 0.9.14, root crate `mlmf` now analysable): 17 `unused-parameters` (`src/distributed_core.rs` 2, `src/distributed_loader.rs` 9, `src/formats/gguf_export.rs` 1, `onnx_export.rs` 1, `onnx_import.rs` 1, `src/multimodal_loader.rs` 3) and 1 `unused-return-values` (`complete`, `src/progress.rs:414`, discarded at 2 of 3 call sites). **Triage at `ad6669a`: 9 of the 17 (the `todo!()` stubs in `distributed_loader.rs`) fixed by underscore-prefix in 0.6.18. The remaining 8 are NOT renames:** `save_as_gguf`'s `model` is discarded and it writes an empty tensor map (`gguf_export.rs:787`, called from `conversion.rs:494`; the same shape as the ONNX discard fixed earlier, a LIVE section 6 defect); `save_as_onnx`'s `export_options` is a documented second discard; `infer_model_config`'s `name_mapper`, `load_distributed_modality`/`load_single_modality`'s `modality`, `process_modality`'s `model`, `deploy_shard_to_node`'s `model_path` and `select_node`'s `request` are unimplemented behaviour. **Update at `f8f6a00`: `save_as_gguf` (#141) and `save_as_onnx` (#142) fixed; 5 more resolved in 0.6.21; the last, `process_modality`'s `model`, is #143.** Each needs a look before any fix: a parameter of a stub that a planned call site will use is a different case from a dead one. *Superseded history — the 0 / 0 below was true only while the root crate could not run:* A peer notice reported 17 unused-parameters findings; recounting raw finding lines directly from the sweep job log on `origin/main` at `09d8321` (run 40034994212, job 120692157236) found **0** for both rules across all 13 workspace members that analysed successfully. The PM traced its "17" to an earlier, different-method baseline (root-package-only, pre-workspace-split) and confirmed 0/0 is correct for the current sweep; no fix work needed. |

## Decision needed (CireSnave): integrate or retire mlmf-awq/mlmf-gptq/mlmf-hf-tokenizer

Surfaced while disposing the `reachability` row above (#135/#136). The
facts, not a recommendation — this is a decision about a *published*
crate's lifecycle, not a code fix this lane can make unilaterally:

- All three crates (`mlmf-awq`, `mlmf-gptq`, `mlmf-hf-tokenizer`) are
  independently published to crates.io (verified via the crates.io API,
  2026-10-23: `mlmf-awq` 0.5.11/30 downloads, `mlmf-gptq` 0.5.11/30,
  `mlmf-hf-tokenizer` 0.5.11/20 — real external downloads, not zero).
- None has an in-repo consumer. The root `mlmf` crate's `awq` feature is
  still a stub (`awq = []`, no dependency on `mlmf-awq` at all). Its
  `tokenizers` feature wraps the external `tokenizers` crate, not
  `mlmf-hf-tokenizer`. No workspace member depends on `mlmf-gptq`.
- Each crate's own code is complete, documented, and covered by its own
  unit tests — this isn't abandoned or broken work, just unwired.
- **Option A — integrate**: wire each crate into the root `mlmf` crate's
  corresponding feature (`awq`, and a new or existing path for GPTQ/
  tokenizer support), making the published crates' existence match what
  `mlmf` actually does. Real engineering work, scope not yet sized.
- **Option B — retire**: yank/deprecate the published crates if they
  were speculative and integration isn't planned. Yanking a published
  crate is a decision this lane should not make unilaterally — flagged to
  CireSnave's board per the PM, 2026-10-23.
- Not actionable by this lane without a decision between A and B; parked
  here rather than left unrecorded.

## File-length (Codacy, on PR #132 at `e411340`, https://github.com/ciresnave/mlmf/pull/132)

Codacy flagged these files for length while PR #132 only touched them to add
an SPDX header and/or a lint-suppression reason — pre-existing, not caused by
that PR (PM disposition, 2026-10-23: "none of those files was lengthened
beyond that \[SPDX/reason\] line... pre-existing"). **15 files, not the 8 a
peer notice first named** — pulled the real count from the check run's own
annotations (`gh api repos/ciresnave/mlmf/check-runs/<id>/annotations`)
rather than the notice, which undercounted. One row per file, biggest
non-comment-LOC first, to split — none started.

| File | Non-comment LOC | Status |
|---|---|---|
| `src/loader.rs` | 1228 | open |
| `src/formats/gguf.rs` | 1091 | open |
| `src/lora.rs` | 1011 | open |
| `crates/mlmf-gguf/src/tensors.rs` | 822 | open |
| `src/model_card.rs` | 723 | open |
| `src/formats/gguf_export.rs` | 700 | open |
| `src/config.rs` | 645 | open |
| `src/formats/onnx_import.rs` | 644 | open |
| `crates/mlmf-core/tests/purity.rs` | 640 | open |
| `src/formats/onnx_export.rs` | 632 | open |
| `crates/mlmf-gguf/src/metadata.rs` | 616 | open |
| `src/metadata.rs` | 528 | open |
| `crates/mlmf-gguf/src/value.rs` | 515 | open |
| `src/distributed.rs` | 515 | open |
| `src/validation.rs` | 513 | open |

## Known infrastructure defect in the same run (not a code finding)

The sweep's `reachability`, `unused-return-values`, and
`unused-parameters` checks fail to run on the root `mlmf` crate specifically
(not failed *with* findings — failed to run at all, confirmed still true
across three separate sweep runs, most recently 40034994212 on `09d8321`):
its `build.rs` needs `protoc`, which the sweep's sandbox
(`ciresnave/coderipper/.github/workflows/portfolio.yml`) does not install.
This repo's own `core-gates`/`root-crate-tests` jobs install it
(`.github/workflows/ci.yml`, "install protoc" step) but that step isn't
available to a reusable workflow owned by another repo. Every other
workspace member (13/13, 14/14 before the file-length split) analysed
cleanly. Reported to the PM in #131; a concrete fix (install protoc via an
input, or report the crate as "not judged" instead of an unlabeled error
exit) was sent directly to the `coderipper` lane on 2026-10-23 — fix belongs
in `ciresnave/coderipper`, not here.

## Spec-to-roadmap audit (PM task, CireSnave's idea, 2026-10-24)

Read at `origin/main` `bfc84ac`: 30 of the 32 tracked markdown files (all but `CLAUDE.md` and this file; every `docs/` spec and plan, `README.md`, the root summary docs, `GLOSSARY.md`, `CONTRIBUTING.md`, and the planned / "not yet" mentions in `CHANGELOG.md`), diffed against this file. Reading was split across five read-only agents. **Re-checked by me against the code: the README ONNX claim, the GGUF quantizer claim, the absent `part_b_deferral.rs`, the dropped legacy-GGML ruling, `mlmf-meta` having no `Unreadable` / `index_complete` reference, the empty `tests/`.** The rest are the agents' findings. Two agents read the tails and headings of the long plans and keyword-scanned the middles, so a leftover item buried in a task body may be missing. `NOT VERIFIED` marks a status nobody checked. **Rows are recorded; none is started.**

Short names: SPEC = `2026-08-14-backend-agnostic-mlmf-design.md`, QSF = `2026-10-01-quantized-safetensors-formats-design.md`, GAP = `2026-10-03-fuel-gap-plan.md`, INV = `fuel-migration-inventory.md`.

### Work in a doc and not in this roadmap (46 rows, 25 need a CireSnave call)

| Row | Work | Where the doc says it | Status | CireSnave call |
|---|---|---|---|---|
| S-01 | Backend-agnostic writers for GGUF and safetensors in `mlmf-gguf` / `mlmf-safetensors` (the root crate's `saver.rs`, `gguf_export.rs` and `safetensors_export.rs` are the only writers) | SPEC §11 `gguf_export`/`safetensors_export` rows; INV §7.2.3, §8.6; gguf-metadata plan L3862 | open | CALL: new crate or module layout, and how it relates to the legacy crate |
| S-02 | GGUF quantizers beyond F32: F16, Q8_0, Q4_0, Q4_K_M and Q6_K all return `NOT IMPLEMENTED` (`gguf_export.rs:616`). `save_as_gguf` defaults to F16, so default options return an error since #141 | GGUF_EXPORT_SUMMARY.md:18-23,148 | open | CALL: implement the quantizers, or narrow the writer to F32 and drop the variants |
| S-03 | GGUF `preserve_metadata` writes only a placeholder `preserved="true"`, and `general.architecture` is written as `"unknown"` (section 6 fence) | `gguf_export.rs`, `save_as_gguf` placeholder comment | open | CALL: section 6 and the `Option<T>` policy |
| S-04 | GGUF calibration data support | GGUF_EXPORT_SUMMARY.md:149 | open | CALL: arguably interpreting content |
| S-05 | AWQ export (`awq_export.rs` returns `Err`); batch, streaming and compression export | GGUF_EXPORT_SUMMARY.md:154-157 | held (by the awq/gptq/hf-tokenizer decision above) | - |
| S-06 | GGUF v1 reading (`UnsupportedVersion`; detector `mlmf-gguf/tests/v1_deferral.rs`; the one corpus file is unmeasured) | SPEC §9 §7 L403; gguf plans L87, L1517 | held (accepted loss, CireSnave 2026-09-25) | CALL: build it or re-defer |
| S-07 | Non-native-endian GGUF: refused by name (`ByteSwapped`), no transcoder | gguf-metadata plan L3865 | open (by design) | CALL: low priority, no corpus file exhibits it |
| S-08 | Streaming, async and ranged sources: no `impl Read` source, no ranged GGUF `parse` | SPEC §9 §7 L404; gguf-metadata plan D4; source-file plan L818 | open | CALL: the `RangedSource` trait change (`&mut self` or interior mutability) is an API call |
| S-09 | Metadata-only GGUF entry point `metadata_only(bytes)` | INV §7.2.1 L295-302 | open | - |
| S-10 | `parse_from(&dyn ByteSource, ..)` constructors for gguf and safetensors | GAP Item 3 step 3, L250-253 | open | - |
| S-11 | Raw positional imatrix `.dat` reader (the GGUF-embedded half is done, #94) | SPEC OQ-3 L567; INV §1.4 | held (deferred, board 63a) | - |
| S-12 | GGUF architecture-family classifier, new crate `mlmf-gguf-arch` | GAP Item 1, L49-132 | open | CALL: GAP flags it as possibly outside the charter (SPEC §10 bans architecture inference from tensor names) |
| S-13 | `GgufMetadata::model_config()` convenience | GAP Item 2, L136-206 | open | CALL: shares Item 4's type or gets its own |
| S-14 | `mlmf-hf-config` crate (HfConfig / ResolvedConfig port) | GAP Item 4, L278-351 | open | CALL: ports from a legacy crate that SPEC §11 schedules for rewrite |
| S-15 | `mlmf-hf-layout` metadata half, remainder after #119: `tokenizer_config.json` and chat-template sidecars, `<filename>:<key>` keys, `BOS_TOKEN_TEXT`, the HF template object-list. Leaves `mlmf-meta`'s `Format::HuggingFace` vocab row dead | hf-layout-metadata-DEFERRED plan L8-12, L464; meta plan L111, L1737 | open | CALL: the plan says "NOT SCHEDULED, A RULING" |
| S-16 | `mlmf-meta` section 5 rows absent: `attention.head_count`, `attention.head_count_kv`, `rope.freq_base` | meta plan L1740 | open | CALL: the plan calls it a decision, and it risks interpreting content |
| S-17 | `mlmf-meta` extractors ignore `Declaration::Unreadable` and `index_complete()`; the blocker, a producing format crate, now exists | meta plan L1741-1742 | open | - |
| S-18 | `mlmf-meta` drops three malformed template-name shapes silently, with no loss entry | meta plan L1739 | open | - |
| S-19 | `mlmf-pickle` (port Fuel's 754-line VM with the AD-3 allowlist test); `load_zip_pickle` is still an unconditional `Err` | SPEC §9 §7 L405, §11 L479, §12 step 6; core plan L1885 | open (accepted loss, CireSnave 2026-09-25) | CALL: build order |
| S-20 | `mlmf-onnx`: move `onnx_import` and `onnx_export` out of the root crate | SPEC §11 L461, L472; ONNX_SUPPORT.md:1 | open | CALL: whether to extract |
| S-21 | `mlmf-oracle`, and removal of the oracle from the load path (`loader.rs:35`) | SPEC §10 L419, §12 step 7, OQ-4 L568 | open | CALL: same workspace or a separate repo (OQ-4) |
| S-22 | `mlmf` umbrella rewrite and legacy-crate lifecycle: retire the root crate, switch CI to `--workspace`, delete `ci_coverage.rs` | SPEC §11 L488, §12 step 7; gguf-metadata plan Step 6b | open | CALL: lifecycle of the legacy crate |
| S-23 | MLX support (QSF: resolve the packing order at implementation time) | QSF §3.4 L298-306, §1.5 | open | - |
| S-24 | EXL2 geometry (`locate_layers`); `mlmf-exl2` is detect-and-refuse (#108) | QSF §1.3 L153-157 | held (deferred by the PM; trigger: read the `q_groups` encoding from ExLlamaV2 source) | - |
| S-25 | EXL3 | QSF §1.4 L189-194 | held (trigger: a reachable EXL3 `.safetensors`) | - |
| S-26 | Delete-disposition legacy modules still present: `distributed*` (8 live `todo!()` in `distributed_loader.rs`, panics on construction), `multimodal*`, `model_card`, `quantization*`; `AlertManager` has no impl; feature 6 and 8 "next steps" | SPEC §10 L423, §11; FEATURE_6 L39-45, L215-219; FEATURE_8 L179-192 | open | CALL: timing of the deletion (scheduled, not executed) |
| S-27 | Undecided legacy modules: `cache.rs`, `cached_loader.rs`, `checkpoint.rs`, `saver.rs`, `progress.rs` home, `validation.rs` structural half | SPEC §11 L456-486; OQ-1 L558 | open | CALL: keep or delete each (OQ-1 is deferred until a second trainer appears) |
| S-28 | Retire `src/mmap_loader.rs` (475 lines) | core plan L1870; source-file plan L816 ("deleting the legacy implementation is CireSnave's") | held (CireSnave) | - |
| S-29 | Legacy `error.rs` (261 call sites) versus `mlmf-core::ErrorKind`: keep a loading-policy error type, or grow core | SPEC §11 `error.rs` row L487 | open | CALL: the row itself says "OPEN QUESTION, not work" |
| S-30 | Promote tensor-name mapping out of the root crate (`name_mapping.rs`, `smart_mapping.rs`) | INV §7.2.4, §8.7 (low priority, L375) | open | - |
| S-31 | ONNX: write real protobuf (`export_graph` returns `NOT IMPLEMENTED`), encoder / encoder-decoder / CNN export, LayerNorm and Attention operators; import Phase 2 (`candle-onnx`, operator support, `load_model(.onnx)`); Phase 3 (optimisation, custom operators, dynamic graph execution) | ONNX_SUPPORT.md:20-22, 40, 43, 105-118 | open | CALL: implement or delete the writer; adopt `candle-onnx`; Phase 3 is interpreting content |
| S-32 | PyTorch `.pth` / `.bin` loading and saving: `pytorch_loader.rs` is a stub, and `save_model` supports only safetensors, gguf and onnx | ONNX_SUPPORT.md:76-80, 92-95; `pytorch_loader.rs:257-262` | open | CALL: ties to S-19 |
| S-33 | Conversion with measured-error semantics (CV-1..5); `ConversionResult` has no measured-error field | SPEC §8, §11 L469 | open | - |
| S-34 | Provenance stamp (source hash, re-hash check, PV-1) | SPEC §8.2 L287-297 | open | - |
| S-35 | U-2 corpus currency: a test that fails on staleness against a recorded source URL and fetch date | core plan L1870; SPEC §7 L262-265 | open | CALL: sets the corpus lifecycle policy |
| S-36 | Corpus roots default to `C:/Models`; the two large files are local-only; nothing counts how often skips fire | safetensors plan, carried items | open | CALL: a self-hosted runner for the local-only files (portfolio rule on tests that cannot run on GitHub) |
| S-37 | Root crate has no integration tests (`tests/` is empty; 139 unit tests live in `src/`) | SPEC §11 L499-509 | open | - |
| S-38 | `load_safetensors_mmap` still forwards to regular loading (`loader.rs:1408`) | SPEC L497; README L18 | open | - |
| S-39 | AD-1 payload-identity half: tensor bytes are never compared | SPEC §9 §7 L393-395 | open | - |
| S-40 | Record the ruling on whether `MetaValue` is format-neutral (code pins `as_u64` as non-coercing; no ruling is recorded) | safetensors-scope plan Q1 | open | CALL: changes what the seam promises |
| S-41 | Lightbulb R5: "measure on the corpus and send the number" (`array_len` / `array_get` exist, the measurement is unverified) | lightbulb doc L195-196 | open (NOT VERIFIED) | - |
| S-42 | gguf-tensors plan Step 2b: classify multi-assert tests by tier and report the distribution; Step 3 liveness-control rate | gguf-tensors plan L1400-1500 | open (NOT VERIFIED; may be in a PR body) | - |
| S-43 | `mlmf-gptq` and `mlmf-awq` disagree on a non-string `quant_method` (gptq returns `Err`, awq records `malformed`) | gptq plan Review Focus vs `gptq/config.rs:157-166`, `awq/config.rs:405` | open | - |
| S-44 | OQ-2 naive round-to-nearest quantization | SPEC L559 | held (until an operator asks) | - |
| S-45 | `mlmf-source-hub`'s range implementation and Fuel's use of `ShardIndex` are unchecked | core plan L1886; shards plan L631 | open (NOT VERIFIED; the second is consumer-side) | - |
| S-46 | `MLMF_TEAM_BRIEFING.md`: rewrite or delete (its header warns it is stale, its body claims v0.1.0 production-ready) | MLMF_TEAM_BRIEFING.md L1-7 | open | CALL: a document with a named audience |

### Docs that contradict the code (13 rows)

Doc corrections: docs-only work that needs no decision. Recorded as rows, not edited in this PR.

| Row | Doc says | Code says | Status |
|---|---|---|---|
| D-01 | README.md:9 says ONNX `num_attention_heads` is GUESSED (`hidden_size / 64`) and `rope_theta`, `activation_function` and the dropouts are asserted constants | `onnx_import.rs` sets them all to `None` (#78); the paragraph is false. Re-checked. | open |
| D-02 | ONNX_SUPPORT.md:1, 9, 24-36 says export is "genuinely implemented" and `save_model(..., "my_model.onnx")` works; L43, 70 say import is "PLANNED" | `export_graph` returns `NOT IMPLEMENTED` (`onnx_export.rs:671`); import exists (`onnx_import.rs:99, 693`) | open |
| D-03 | GGUF_EXPORT_SUMMARY.md:12, 19-21, 32, 86, 159-168 says F16, Q8_0 and Q4_0 work, gives a size table and says "production-ready"; L15 claims streaming writes while L156 lists streaming as future | only F32 writes (`gguf_export.rs:616-620`); every other type, F16 included, returns `NOT IMPLEMENTED`. Re-checked. | open |
| D-04 | FEATURE_8_MULTIMODAL_SUMMARY.md:99-114 shows a working `model.infer(input)`; FEATURE_8_COMPILATION_SUCCESS.md:126, 139 says "production-ready" and ticks Feature 6 | `process_modality` ignores its model (#143); `DistributedModelLoader::new` panics through `todo!` (`distributed_loader.rs:811`) | open |
| D-05 | FEATURE_6_DISTRIBUTED_SUMMARY.md:223-236 says "COMPLETED ... production-ready" | 8 live `todo!()`; its own banner retracts this, but FEATURE_8's tick of Feature 6 does not | open |
| D-06 | hf-layout-metadata-DEFERRED plan L8-12 and `mlmf-hf-layout`'s `lib.rs` and `config.rs` cite `tests/part_b_deferral.rs` as existing and say the metadata half "has not been rewritten" | the file is absent on `origin/main` (re-checked); `ConfigJson` shipped in #119 (board #111) | open |
| D-07 | INV L188 says `ModelConfig` cannot represent an absent field (#48, open); L189 says `tokenizer.json` is "HAS"; L191-192 say `generation_config.json` and `special_tokens_map.json` are "MISSING"; L76 says mmap is "not yet wired" | fields are `Option<T>` (#77); GAP L359-364 retracts the tokenizer claim; both readers exist in `mlmf-hf-layout`; zero-copy is verified (#115) | open |
| D-08 | SPEC L441 says `mlmf-source-file` has not landed and `mlmf-hf-layout` does not exist; L483, 487 say the root crate has no `mlmf-*` dependency | both crates exist; `Cargo.toml` depends on `mlmf-gguf` and `mlmf-core` (#36); `gguf.rs:85` calls `mlmf_gguf::GgufMetadata::parse` | open |
| D-09 | README.md:31 `tag = "v0.2.1"`; RELEASE_NOTES_v0.2.0.md:1 "at 0.4.0"; ONNX_SUPPORT.md:126 `version = "0.1.0"`; MLMF_TEAM_BRIEFING.md "v0.1.0 production-ready" | the workspace is at 0.6.22 (proposed) | open |
| D-10 | `crates/mlmf-hf-layout/Cargo.toml` description says only "HuggingFace shard index: where each tensor lives" | the crate has four modules: `config`, `generation_config`, `shards`, `special_tokens_map` | open |
| D-11 | ggml plan L1752 still lists a future `mlmf-ggjt` crate | closed: ruled by CireSnave, board 63a, 2026-09-26 (SPEC L406, verbatim). Re-checked. | open |
| D-12 | gptq and awq plans describe `quant_method` handling and the awq struct differently from the code | `awq/config.rs:69` has a `quant_method: Option<String>` the plan's struct lacks (see S-43) | open |
| D-13 | safetensors plan says `mlmf-safetensors` is "a crate that does not exist"; the source-file plan says `src/` is "19,329 lines across 37 files" | `crates/mlmf-safetensors` exists; the line count is stale after #138-#144 | open |

### Verified done (the docs still list these as pending or planned)

| Item | Where |
|---|---|
| mmap zero-copy for gguf and safetensors | #115 |
| `tokenizer.json` reader (`mlmf-hf-tokenizer`) | #116 |
| `generation_config` and `special_tokens_map` readers | `mlmf-hf-layout` |
| `mlmf-hf-layout` `config.json` `MetadataSource` | #119 |
| `mlmf-gptq`, `mlmf-awq`, `mlmf-exl2` (detect-only) | #105, #110, #108 |
| `mlmf-source-file`, `mlmf-source-hub`, `mlmf-meta` | #103 and the crates |
| Embedded imatrix reader | #94 |
| `Option<T>` for absent fields (#48) | #77 |
| Legacy GGML (GGJT) container dropped | board 63a, 2026-09-26 |
| `save_as_gguf` writes the model's tensors | #141 |
| `save_as_onnx` honours or refuses its options | #142 |
| transitive-deps C1 and C2 gates | `mlmf-core/tests/transitive_deps.rs` |
| `errors.txt` removed | absent at the repo root |
| `Declaration`, `array_len`, `array_get`, `NotGguf` (lightbulb R2, R5, R7) | `mlmf-core`, `mlmf-gguf` |
