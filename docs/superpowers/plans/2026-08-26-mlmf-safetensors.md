# `mlmf-safetensors` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second backend behind `mlmf-core`'s seam, so that "backend-agnostic" becomes a claim that can be falsified rather than a branch name.

**Architecture:** Two stages, separate by construction as in `mlmf-gguf`: an 8-byte length prefix and a JSON header, then a tensor directory derived from that header. `SafetensorsFile` implements both `MetadataSource` and `TensorContainer`. Offsets are rebased once at parse time, from safetensors' own base — the end of the JSON header — which is a *different base from GGUF's* and is the point.

**Tech Stack:** Rust 2024, `#![forbid(unsafe_code)]`, no I/O in `src/` (`&[u8]` in, structures out), `mlmf-core` plus `serde_json` (see D3).

**Spec:** `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md`. Scope and rulings: `docs/superpowers/plans/2026-08-26-mlmf-safetensors-scope.md`.

## Global Constraints

- Rust edition 2024, toolchain pinned to 1.98.0 by `rust-toolchain.toml`.
- `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`.
- **No file I/O anywhere in `src/`.** The core is about what is in the file, not how it is obtained. The C3 purity gate enforces this and the new crate must be added to it.
- **MLMF reads and writes model files. It is never an interpreter of their content.** A tensor name is an opaque key. No `config.json`, no architecture inference, no sharding.
- Every gate green by EXIT STATUS before every commit. **Never pipe a gate
  to another command**: piping to `tail` or `sed` makes the pipeline's exit
  status the *last* command's, and a check whose exit status is discarded is
  not a check. That has happened twice in this project.

  **Run what CI runs, per crate — not `--workspace`.** The two are not the
  same set and the difference is worth stating, because an earlier version of
  this line said `--workspace` and that is what a reader would have trusted:

  ```
  cargo test   -p mlmf-core -p mlmf-ggml -p mlmf-gguf -p mlmf-safetensors --no-fail-fast
  cargo test   -p mlmf-core --no-default-features
  cargo clippy -p <each> --all-targets -- -D warnings
  RUSTDOCFLAGS="-D warnings" cargo doc -p <each> --no-deps
  cargo fmt --all -- --check
  ```

  `cargo test --workspace` additionally builds the **legacy root `mlmf`
  package**, which CI excludes on the record: it needs `protoc` and a git
  dependency, and spec §11 schedules it for rewrite rather than repair. On a
  machine that has `protoc` it builds and passes, so `--workspace` is a
  SUPERSET of CI rather than a different or unrunnable set — which is the
  safe direction, and is why nothing was missed by using it. But a green from
  it is not the green CI computes, and only the per-crate set is reproducible
  on a runner.

  **Also run the gates once with `CARGO_TERM_COLOR=always`.** CI sets it and
  your shell does not, and it has produced a red in this repo that no plain
  local run reproduces — a dependency-snapshot gate that parsed coloured
  `cargo tree` output and reported crates as added that were already in the
  snapshot.

---

## Measured facts this plan is built on

From `docs/superpowers/plans/safetensors_recon.py`, against the two models present locally, 491 tensors. Reproduce before trusting.

**Layout, confirmed byte-exact on 2 of 2 files.** The falsification test: furthest `data_offsets[1]` + 8 + header length == file length, exactly.

```
8 bytes    u64 little-endian header length N
N bytes    JSON header
remainder  tensor data, offsets relative to the END OF THE HEADER

header:  name -> { dtype: str, shape: [int], data_offsets: [start, end] }
         plus an optional "__metadata__" key holding str -> str

  SmolLM2-360M    tensors=290  hlen=32664  data_start=32672   delta=0
  TinyLlama-1.1B  tensors=201  hlen=23088  data_start=23096   delta=0
                                             (2,200,119,864 bytes)
```

**`__metadata__` is `str -> str`.** Both files carry exactly `{'format': 'pt'}`.

**491 of 491 tensors are `BF16`.** One dtype. The corpus cannot vary the property most of this plan is about.

**Zero aliased ranges, zero overlaps** in either file.

## What the corpus cannot do, stated before anyone reads a green

Two files, one dtype, no aliasing, no overlaps, no non-UTF-8 names, no `F8` anything, no escaped characters in any tensor name. It confirmed the layout — byte-exactly, on 2.2 GB — and it can falsify almost nothing else.

**Authored fixtures are a prerequisite, not an optimisation.** This is the second time in this project a corpus has been measured and found unable to exercise the paths it appears to cover. It is not the corpus's fault: real files are samples of what writers actually emit, and the interesting inputs are the ones writers rarely emit.

## Design decisions

**D1 — Rulings carried in from the scope, both made rather than inherited.**

*A `MetaValue` variant reports how the format DECLARED a value, not what the value means.* Safetensors' `__metadata__` is `str -> str`, so every value from it is `MetaValue::String` and twelve variants never appear. `as_u64()` returning `None` there is **correct**: it means "this format did not declare a number here". Accessors widen losslessly within a family and never parse.

*Whether two tensors may share bytes is a FORMAT FACT.* GGUF collisions are malformed and `mlmf-gguf` reports them. **Safetensors tied weights — `lm_head` and `embed_tokens` on the same range — are a standard layout, and this crate must NOT report them.** Reporting one would blame a valid file.

**D2 — One type implements both traits.** GGUF splits `GgufMetadata` and `GgufTensors` because its metadata and tensor stages are separate byte regions and R1 requires that reading one cannot fail on the other. Safetensors has **one** header carrying both, so splitting would be cargo-culting GGUF's shape. The stage split that matters here is *parse the header* versus *resolve the tensors*, and it is enforced by returning a `Report` rather than by two entry points.

**D3 — `serde_json`, and this is the first external dependency in a format crate.** Both existing format crates have only workspace-internal deps, so this sets precedent and is recorded rather than slipped in.

A hand-rolled subset parser was considered and rejected. The header is JSON that a writer produced with a real serialiser, so it may legally contain `\uXXXX` escapes, surrogate pairs and exponential-form numbers. **A tensor name is a lookup key** — this project already ruled for GGUF that a key must be byte-exact or the parse fails — and a subset parser either mangles an escape, producing a name nobody can look up, or refuses a valid file, which is the failure mode D1's second ruling exists to avoid. Getting JSON right is not this crate's contribution.

`serde_json` is added to `crates/mlmf-safetensors/tests/direct-deps.allow` with this argument beside it. C1/C2 constrain `mlmf-core`, which is untouched.

**D4 — Rebase once, at parse time, from a different base.** `TensorDescriptor::bytes` is absolute into the slice. GGUF stores `data_start + info.offset` where `data_start` is a padded region boundary; safetensors stores `8 + header_len + data_offsets.0`. **The doc on that field argues at length that a consumer guessing the wrong base "would read plausible-looking floats, and only one of them would be right" — and until this crate exists that argument has been validated against exactly one base.**

**D5 — Dtype mapping is pinned arm by arm, never by width.** `F8_E4M3` and `F8_E5M2` are same-width, same-kind and mutually byte-incompatible. This project's notes have recorded that trap as dormant "until a format crate maps declared type strings onto `DType`". **This is that crate.** A test that checks widths would pass on a swapped mapping; every arm is pinned by identity.

---

## Before dispatching any task: who owns each file?

**For every file this plan names, which task's `Files:` line claims it?** A
file a plan schedules and no task owns is a file no per-task review looks
at — and it has now happened twice on the same filename. `mlmf-gguf`'s
`error.rs` still told consumers the tensor stage was unreachable after that
stage had landed, because plan 4 scheduled the file and each of its eight
tasks touched something else. This plan's Task 2 omitted
`crates/mlmf-safetensors/src/error.rs` for the same reason, caught by Task
1's implementer rather than by the plan.

Twice, same filename, same cause, is a pattern in the planning method rather
than two coincidences. So the check runs before dispatch: extract the files
the plan names, extract the files each task claims, and diff.

**And name every file the same way every time.** The check's own first run
reported `src/error.rs` owned by Task 1 and
`crates/mlmf-safetensors/src/error.rs` owned by Task 2 — **one file reading
as two, each apparently owned**. A genuine orphan can hide behind an
inconsistent spelling, which makes path-form consistency load-bearing rather
than cosmetic. Both were normalised to full paths.

It also found `tests/direct-deps.allow` genuinely orphaned: Task 0's prose
creates it, its `Files:` line did not claim it. No harm resulted because the
same task did the work, but the check does not know that and should not have
to.

---

## Task 0: Crate skeleton and the gates that must cover it

**Files:** `crates/mlmf-safetensors/Cargo.toml`,
`crates/mlmf-safetensors/src/lib.rs`,
`crates/mlmf-safetensors/tests/direct-deps.allow`,
`crates/mlmf-safetensors/tests/allowed-std.list`,
`.github/workflows/ci.yml`, `Cargo.toml`.

The gates are the deliverable, not scaffolding. `mlmf-gguf` existed for six tasks before CI named it, and every green in that window was honest and meaningless.

- [ ] **Step 1: Write the failing gate check first.** Before the crate exists, run `cargo test -p mlmf-core --test ci_coverage --no-fail-fast`. It passes. Create the crate directory with a stub `lib.rs`, re-run, and it must now FAIL naming `mlmf-safetensors` for all three of test/doc/clippy. **That red is the deliverable of this step** — it proves the gate covers a crate the moment the crate exists, which is the property `ci_coverage.rs` was written for.

- [ ] **Step 2: Add the crate to the workspace and to CI.** Three steps in `ci.yml` mirroring `mlmf-gguf`'s, including the `RUSTDOCFLAGS: -D warnings` env. Re-run `ci_coverage` and `module_registration`; both green.

- [ ] **Step 3: `direct-deps.allow` and `allowed-std.list`.** `mlmf-core` and `serde_json` in the first, with D3's argument as a comment. Start the second EMPTY and add permissions only in the commit that needs them, as `mlmf-gguf` did — a permission granted before its use is a permission nobody re-examines.

- [ ] **Step 4: Prove the purity gate covers the new crate.** Inject `use std::fs;` into the stub `lib.rs`, run `cargo test -p mlmf-core --test purity --no-fail-fast`, confirm it fails NAMING the crate. Remove. Then inject `bytemuck` into the manifest and confirm the deps gate fails naming the crate. **Both, separately** — they are different gates and one covering the crate does not imply the other does.

- [ ] **Step 5: Commit.**

## Task 1: The header — length prefix and JSON

**Files:** `crates/mlmf-safetensors/src/header.rs`, `crates/mlmf-safetensors/src/error.rs`.

- [ ] **Step 1: Write the failing tests.** Cases, each with a whole-value assertion: a well-formed minimal header; a length prefix larger than the file (truncated, not an allocation); a length prefix of 0; a header that is not valid UTF-8; a header that is valid UTF-8 and not valid JSON; a header whose top level is an array rather than an object. **Every "refused" case asserts the error's stage and kind, not merely that it is an error.**

- [ ] **Step 2: Run with a stub.** The crate will not compile with the functions missing, and a compile failure runs ZERO tests and prints no `test result:` line at all. Add a stub returning the trivial wrong answer, then confirm the tests fail at **named assertions** rather than at a compile error or a panic.

- [ ] **Step 3: Implement.** Bounds-check the declared length against the slice **before** allocating or slicing.

- [ ] **Step 4: Run — by exit status, `--no-fail-fast`.**

- [ ] **Step 5: Sabotage.** Name the expected kill set BEFORE each run; report the SHORTFALL. Run unfiltered. `git diff` each sabotage first — and `git add -N` any file the task itself created, because `git diff` prints nothing for an untracked file and empty output is exactly what an inert sabotage looks like.
  1. Drop the length bounds check. Predict whether it errors or aborts; record which you saw.
  2. Accept a non-object top level. Expect the array case red, and nothing else.
  3. Read the length as little-endian from the wrong offset. Expect several red; record which assertion fires first.

- [ ] **Step 6: Commit.**

## Task 2: The tensor directory, and the different base

**Files:** `crates/mlmf-safetensors/src/tensors.rs`, **and
`crates/mlmf-safetensors/src/error.rs`** — this task needs
`Stage::TensorDirectory` and a tensor-stage variant, and neither exists yet.

That second file was missing from this list until Task 1's implementer said
so. It is the same trap `mlmf-gguf/src/error.rs` already carries a comment
about: plan 4 scheduled a variant there, each of its eight tasks touched
something else, and the doc still told consumers the tensor stage was
unreachable after the stage had landed. **A file a plan schedules and no
task owns is a file no per-task review looks at.**

Task 1 deliberately did NOT pre-add the variant, which was right — a variant
added ahead of its use is one no review examines.

- [ ] **Step 1: Write the failing tests.**
  - A resolvable tensor becomes a descriptor **rebased onto the slice**: `8 + header_len + data_offsets.0`, asserted as a whole tuple against a fully-specified literal. This is D4 and it is the highest-value assertion in the plan.
  - `data_offsets` whose end is before its start.
  - `data_offsets` that run past the end of the file.
  - A shape whose element count times the dtype width does not equal `end - start` — **the file disagreeing with itself**, which the recon checks and no GGUF path has an analogue for.
  - **Tied weights: two tensors with identical `data_offsets` both resolve, both are readable, and the report is EMPTY.** This is D1's second ruling and it is the one that must not regress into GGUF's answer.
  - A tensor named `__metadata__`… is not a tensor. Assert it is treated as metadata and does not appear in `tensors()`.

- [ ] **Step 2: Stub, then confirm assertion-level failures.**

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run.**

- [ ] **Step 5: Sabotage.**
  1. Rebase from 0 instead of from `8 + header_len`. Expect the tuple assertion red; **record the two ranges** — the numbers are the argument, as they were in plan 4 where an equivalent defect was off by exactly a megabyte with every byte a real float from a different tensor.
  2. Report tied weights as overlapping. Expect the tied-weight test red. **This is the control for the ruling**, not for the code.
  3. Accept `end < start`. Expect that case red.

- [ ] **Step 6: Commit.**

## Task 3: Dtypes, pinned arm by arm

**Files:** `crates/mlmf-safetensors/src/dtype.rs`.

- [ ] **Step 1: Write the failing test.** Every safetensors dtype string mapped to its `DType`, **each arm asserted by identity**, plus a total-count assertion so a silently dropped arm fails. Include `F8_E4M3` and `F8_E5M2` explicitly and adjacently.

- [ ] **Step 2: Stub, confirm assertion-level failure.**

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run.**

- [ ] **Step 5: Sabotage — and this one is the reason the task exists.**
  1. **Swap `F8_E4M3` and `F8_E5M2`.** Same width, same kind, mutually byte-incompatible. Expect exactly two arms red. **A test asserting widths would stay green**, which is the whole point; confirm the assertion that fires is an identity one.
  2. Delete one arm. Expect the count assertion red — that is the control for the control.

- [ ] **Step 6: Commit.**

## Task 4: `MetadataSource`, and the ruling made testable

**Files:** `crates/mlmf-safetensors/src/lib.rs`.

- [ ] **Step 1: Write the failing tests.** `__metadata__` keys are readable; every value is `MetaValue::String`; `as_u64()` on a numeric-looking value returns `None`; `keys()` excludes tensor names; `declaration()` distinguishes absent from declared.

- [ ] **Step 2–4: Stub, implement, run.**

- [ ] **Step 5: Sabotage.**
  1. Parse numeric-looking strings into `U64`. Expect the `as_u64` test red. **This is the control for ruling 1** — that a variant reports the format's declaration and not a guess about meaning.
  2. Include tensor names in `keys()`. Expect the `keys()` test red.

- [ ] **Step 6: Commit.**

## Task 5: The cross-backend test, which is what this whole plan was for

**Files:** `crates/mlmf-safetensors/tests/cross_backend.rs`.

This is the only test in the project that can fail because an abstraction is wrong rather than because an implementation is.

- [ ] **Step 1: Write it.** Build a GGUF and a safetensors file that declare **the same logical model** — same tensor names, same shapes, same dtype, same one metadata key. Then assert, **through `&dyn MetadataSource` and `&dyn TensorContainer` only**, with no concrete type in scope:
  - `tensors()` yields the same names in the same order from both.
  - `tensor(name)`'s shape and encoding agree across backends.
  - `tensor_bytes()` returns byte-identical payloads.
  - **And the difference the seam permits, asserted as a difference:** the metadata key's `MetaValue` variant is `U32` from one and `String` from the other, and `as_u64()` answers `Some` and `None` respectively. **Pinning the divergence is the point** — it is ruling 1 stated as a test rather than as a doc comment, and it is unpinnable with one backend.

- [ ] **Step 2: Run.**

- [ ] **Step 3: Sabotage.** Change one backend's tensor ordering; change one's rebase. Both must redden. **And name what this test cannot do**: two backends is a better sample than one and it is still two.

- [ ] **Step 4: Commit.**

## Task 6: Corpus differential

**Files:** `crates/mlmf-safetensors/tests/corpus-safetensors.tsv`, `tests/corpus.rs`.

Extend `safetensors_recon.py` — **it must stay independent of the crate**, or an error shared between the parser and its expectations cannot be caught by comparing them. Emit per file: relative path, tensor count, header length, data_start, first tensor name/dtype/offsets, furthest end.

The differential must **enumerate** — exact row count, exact dtype distribution — not iterate over what it finds. And it must print a loud SKIPPED line **to the `std::io::stderr()` handle, not `eprintln!`**, which libtest captures for a passing test, and this test passes when it skips.

- [ ] Sabotage: corrupt one row; replace the loop body with `continue`; rename the corpus directory and confirm the SKIPPED line is visible under a plain `cargo test`.

## Task 7: Whole-branch review

**Not optional.** Every Important finding in plans 3 and 4 lived in the seams between tasks. The last pass found four, including a green that was true only of one machine's stale lockfile.

- [ ] Dispatch a review over the whole plan-5 diff with the priorities that worked twice: cross-file contradictions first; tests that cannot fail (**four known vacuity modes** — mutation never applied, filter excluded the test, earlier assert panicked first, defect named is impossible); claims of impossibility; charter violations; then whether the API reads as one thing.
- [ ] **Verify every Important finding yourself before fixing it.**
- [ ] Record the rate, and compare to plan 3's eleven and plan 4's two.

---

## Deliberately not in this plan

Each labelled by kind, because "we are not doing X" and "we measured and X is not available" look identical once written down.

- **Sharded models** (`model-00001-of-00002.safetensors` + index JSON) — **charter decision.** That is file discovery, and how a file is obtained is explicitly not this crate's business.
- **`config.json`** — **charter decision.** Interpreting a model's configuration is the consumer's job; this crate reads the file it is given.
- **Writing safetensors** — **design decision.** The authored builder emits malformed output on request, which is exactly what a writer must not do.
- **PyTorch pickle** — **design decision**, planned separately in the spec; it needs the borrow-or-own `Cow` path `tensor_bytes` already carries.
- **Proving the abstraction** — **measured premise.** Two backends is a better sample than one and it is still two. What this plan buys is the ability to be *wrong* in a way one backend cannot express.
