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
| `unused-parameters` / `unused-return-values` | **17 / 1** at `8481863` (was 0 / 0 before the protoc fix) | open | Counted off the sweep log of run 40068424658 (`coderipper` 0.9.14, root crate `mlmf` now analysable): 17 `unused-parameters` (`distributed_core` 2, `distributed_loader` 9 (both removed in 0.7.0), `src/formats/gguf_export.rs` 1, `onnx_export.rs` 1, `onnx_import.rs` 1, 3 in the multimodal loader, since removed with the rest of the multimodal code in 0.7.0) and 1 `unused-return-values` (`complete`, `src/progress.rs:414`, discarded at 2 of 3 call sites). **Triage at `ad6669a`: 9 of the 17 (the `todo!()` stubs in `distributed_loader.rs`) fixed by underscore-prefix in 0.6.18. The remaining 8 are NOT renames:** `save_as_gguf`'s `model` is discarded and it writes an empty tensor map (`gguf_export.rs:787`, called from `conversion.rs:494`; the same shape as the ONNX discard fixed earlier, a LIVE section 6 defect); `save_as_onnx`'s `export_options` is a documented second discard; `infer_model_config`'s `name_mapper`, `load_distributed_modality`/`load_single_modality`'s `modality`, `process_modality`'s `model`, `deploy_shard_to_node`'s `model_path` and `select_node`'s `request` are unimplemented behaviour. **Update at `f8f6a00`: `save_as_gguf` (#141) and `save_as_onnx` (#142) fixed; 5 more resolved in 0.6.21; the last, `process_modality`'s `model`, was #143 and is resolved by removing the multimodal code (0.7.0).** Each needs a look before any fix: a parameter of a stub that a planned call site will use is a different case from a dead one. *Superseded history — the 0 / 0 below was true only while the root crate could not run:* A peer notice reported 17 unused-parameters findings; recounting raw finding lines directly from the sweep job log on `origin/main` at `09d8321` (run 40034994212, job 120692157236) found **0** for both rules across all 13 workspace members that analysed successfully. The PM traced its "17" to an earlier, different-method baseline (root-package-only, pre-workspace-split) and confirmed 0/0 is correct for the current sweep; no fix work needed. |

## What belongs in MLMF (CireSnave, 2026-10-24; full quotes in `CLAUDE.md`)

Code stays if it adds to MLMF's ability to read or write model files without
interpreting or guessing about them. Detecting HOW a file is stored (architecture
detection, tensor-name mapping, the oracle) is in; supplying a value the file does
not declare is out. Applied in 0.7.0, which removed multimodal, distributed,
quantization, caching and model-card code.

Follow-ups from that ruling, none started:

- **Memory estimation** moves to Fuel; MLMF keeps what feeds it (shapes, dtypes, byte
  sizes, declared layer count, hidden size, KV heads, head dim, context length, all
  `Option`). The estimator code in the removed `cache` and `model_card` modules is at
  commit `bfc84ac`.
- **A conversion project** (proposed, PM to decide) for quantization and calibration,
  Q8_0 and other encoding, LoRA merging, and conversion with measured error. MLMF
  readers and writers should accept and return data already encoded in the file's
  dtype.
- **Offers of the removed code:** distributed loading to Fuel if it has nothing
  similar; quantization to Fuel or the conversion project; caching and model cards as
  possible small projects built on MLMF. All pending the PM.
- The design spec's section 10 still bans architecture inference and schedules the
  oracle for removal; CireSnave's ruling reverses that part and the spec needs
  amending.

## AWQ stays; gptq and hf-tokenizer undecided

CireSnave, 2026-10-24, verbatim: *"We are keeping AWQ loading and saving in MLMF."*
The "integrate or retire" decision this section used to pose (PR #137,
2026-10-23) was written by this lane while disposing of a `reachability`
finding, not asked for by CireSnave, and it never appeared on his decisions
list. It is withdrawn for AWQ.

What exists for AWQ, and what does not:

- `crates/mlmf-awq` (published, 0.5.11): config parsing (`parse_standalone`,
  `parse_from_model_config`) and `locate_layers`. It reads **no weights**.
- `src/formats/awq.rs` (root crate): loading. **Not implemented**; it refuses
  with `AWQ loading is NOT IMPLEMENTED`.
- `src/formats/awq_export.rs` (root crate): saving. **Not implemented**; it
  returns `Err`.
- The root `awq` feature is `awq = []` and is in `default`; it does not depend
  on `mlmf-awq`. Routing and detection around AWQ sit in `universal_loader.rs`,
  `loader.rs`, `validation.rs` and `conversion.rs`.

Work this implies, none started: real AWQ tensor loading, real AWQ saving, and
wiring the root `awq` feature to `mlmf-awq` (audit rows S-05 and the GGUF
export summary's AWQ item). `mlmf-gptq` and `mlmf-hf-tokenizer` have had no
ruling either way; their `reachability` declarations in `.coderipper.toml`
stand until one is made.

## File-length (Codacy, on PR #132 at `e411340`, https://github.com/ciresnave/mlmf/pull/132)

Codacy flagged these files for length while PR #132 only touched them to add
an SPDX header and/or a lint-suppression reason — pre-existing, not caused by
that PR (PM disposition, 2026-10-23: "none of those files was lengthened
beyond that \[SPDX/reason\] line... pre-existing"). **15 files (13 after 0.7.0 removed `model_card` and `distributed`), not the 8 a
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
| `src/formats/gguf_export.rs` | 700 | open |
| `src/config.rs` | 645 | open |
| `src/formats/onnx_import.rs` | 644 | open |
| `crates/mlmf-core/tests/purity.rs` | 640 | open |
| `src/formats/onnx_export.rs` | 632 | open |
| `crates/mlmf-gguf/src/metadata.rs` | 616 | open |
| `src/metadata.rs` | 528 | open |
| `crates/mlmf-gguf/src/value.rs` | 515 | open |
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
