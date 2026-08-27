# `mlmf-safetensors` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second backend behind `mlmf-core`'s seam, so that "backend-agnostic" becomes a claim that can be falsified rather than a branch name.

**Architecture:** Two stages, separate by construction as in `mlmf-gguf`: an 8-byte length prefix and a JSON header, then a tensor directory derived from that header. `SafetensorsTensors` implements `TensorContainer` and `SafetensorsMetadata`
implements `MetadataSource` — **two types, not one**, mirroring
`mlmf-gguf`'s `GgufMetadata`/`GgufTensors` split. This line said
`SafetensorsFile` until Task 4's implementer pointed out no such type was
ever built; the split is better than the plan's original shape, because it
keeps the two stages separate BY CONSTRUCTION rather than by discipline,
which is D2's whole point. Offsets are rebased once at parse time, from safetensors' own base — the end of the JSON header — which is a *different base from GGUF's* and is the point.

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

  **Run `bash scripts/local-gates.sh`, which reads the workflow.** It
  extracts every `run:` line from `.github/workflows/ci.yml` and executes
  them in order, so it cannot drift from CI — it has no list of its own to
  drift. It sets `CARGO_TERM_COLOR=always` and `RUSTDOCFLAGS=-D warnings`,
  both of which CI sets and an interactive shell does not, and it refuses to
  report success if it extracted zero commands, because an empty job list is
  what a passing run looks like.

  It exists because a hand-written local set is a claim about CI's job list
  that decays silently every time CI gains a job. The gap was real here: CI
  runs `scripts/check-deps.sh` and my standing local set did not include it.
  Nothing was missed, and nothing would have told me if it had been.

  `cargo test --workspace` additionally builds the **legacy root `mlmf`
  package**, which CI excludes on the record — it needs `protoc` and a git
  dependency, and spec §11 schedules it for rewrite rather than repair. On a
  machine with `protoc` it builds and passes, so `--workspace` is a SUPERSET
  of CI rather than a different or unrunnable set. That is the safe
  direction, but a green from it is not the green CI computes.

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

## Task 2b: The two seam findings Task 2 produced

**This is what the plan was for.** Tasks 0 and 1 showed the seam could be
FITTED; Task 2 is the first that came back and said it is wrong. Both items
are `mlmf-core` changes and both are in scope for this task.

**Files:** `crates/mlmf-core/src/report.rs`,
`crates/mlmf-safetensors/src/tensors.rs`.

### 1. `TensorEncoding` cannot express a non-numeric declared type

`UnrecognizedKind::TensorEncoding { name, family, code: u32 }`. **Safetensors
declares a dtype as a STRING.** So the seam's own distinction — "cannot
resolve the encoding" versus "declined for another reason" — is
**inexpressible for a string-typed format**, and `mlmf-safetensors` currently
reports `TensorDeclined` for both, collapsing a distinction the seam says
matters.

The distinction is worth keeping: *this build does not know this dtype* and
*this tensor's range is bad* are different facts a consumer would act on
differently. So the field must admit both shapes. Change `code: u32` to a
small enum carrying either a numeric code or a declared name, pin both arms
by identity, and update `mlmf-gguf`'s three construction sites.

**Do not widen it to `String` and format the number into it.** That loses
the numeric identity GGUF needs and reproduces the defect this project
removed from `MetadataKey` — an explanation occupying a field documented as
the file's own data.

### 2. Both backends must agree on a range past the end of the file

**They currently do not, and the seam was silent, which is why.**
`mlmf-gguf` keeps the descriptor and fails at `tensor_bytes`;
`mlmf-safetensors` omits it and reports. Each read a different seam doc, and
both docs were right about what they said.

**Ruled: KEEP and report.** `TensorDescriptor::bytes` now says so, and the
`&blob[d.bytes]` licence that implied otherwise is corrected — a descriptor
records what the file DECLARES, including a range the file cannot honour, and
dropping it would be this crate deciding a declaration does not count.

So **`mlmf-safetensors` changes**: a tensor whose rebased range lies past the
end stays in `tensors()`, is named in the report, and fails at
`tensor_bytes`. Its existing test asserts omission and must be inverted, with
a sabotage proving the descriptor survives.

**Task 5's cross-backend test depends on this ruling.** Until both backends
answer the same way, it cannot assert agreement on the case at all.

### 3. `TensorDeclined`'s doc promises omission, and item 2 makes that false

**Found while briefing this task, not by Task 2.** `UnrecognizedKind::
TensorDeclined`'s doc opens *"Like `Self::TensorEncoding`, the tensor is
**omitted from the container's list**"* — a seam-level promise. Item 2 keeps
a past-EOF tensor in the list and reports it, so the promise breaks the
moment item 2 lands.

**This is the shape of defect this plan exists to catch:** a doc that was
true when one backend could reach it, and that no per-task review would look
at, because no task's file list names `report.rs` for a doc it does not
otherwise touch.

**Ruled: correct the doc; do NOT add a `retained: bool` field.**

The tempting fix is a field on the entry saying whether the descriptor
survived. Reject it. **The tensor list is the authoritative answer to
"is it in the list", and a field copying that answer can disagree with it.**
That is the same reasoning that keeps `TensorDescriptor::bytes` a single
rebased range rather than something each consumer recomputes, and the same
reasoning that removed explanations from `MetadataKey`'s `value`. A report
entry names a tensor this build has a complaint about; whether a descriptor
could still be built from the declaration is per-format, and `tensor(&name)`
answers it without a second copy that can rot.

**`TensorEncoding`'s omission promise STAYS**, and stays seam-level: an
unresolvable encoding means `TensorDescriptor::encoding` cannot be filled at
all, so no descriptor exists to keep. That is true of both backends and
remains true after item 1 — safetensors' unknown dtype moves to this kind
and is still omitted.

---

## Task 2c: GGUF implements half the keep-and-report ruling

**Found by Task 2b's implementer, in its own report, and it BLOCKS Task 5.**

The ruling is **keep and report**. After Task 2b the two backends *keep*
alike and *report* differently: `mlmf-safetensors` names a past-EOF tensor in
the report, `mlmf-gguf` says nothing. `mlmf-gguf::tensors::resolve` takes no
file length and there is no EOF sweep anywhere in its directory parse — only
`tensor_bytes` checks, at read time.

**So a consumer handed the same defect by two backends gets a diagnostic from
one and silence from the other**, which is the exact class of divergence
Task 2b existed to remove. A backend that keeps without reporting has
implemented half a ruling.

**Ruled: `mlmf-gguf` reports it too.**

Whether a declared range lies inside the file is a **fact about the bytes**,
not an interpretation of the model, so it is in scope under the charter.

**Files:** `crates/mlmf-gguf/src/tensors.rs`.

`parse_tensors` already has `bytes` in scope, so `bytes.len() as u64` reaches
`resolve` with a signature change and nothing else.

**Do not move the bound out of `tensor_bytes`.** This adds a report entry; it
does not remove a check. Both must hold.

**The vocab-only concern does not apply and the existing comment says why.**
`data_start` may legitimately point one alignment block past the end — 19 of
the 28 corpus files are that shape — which is why the *data region's* start
is deliberately unvalidated. A **per-tensor** bound cannot fire on those
files, because they declare zero tensors. Read that comment before touching
anything near it, and leave it intact.

**Test it with sabotage in both directions**: a past-EOF tensor is kept AND
named, and a normal file gains no entry. The second is not optional — a
check that reports every tensor is indistinguishable from a working one
until someone opens a healthy model.

### And a doc-only correction in the same task

`TensorDescriptor::validate()` returns `Ok` for a descriptor whose bytes are
not in the file: `71..83` is 12 bytes, `[2,3]` BF16 needs 12, and that is all
`validate` promises. **The doc is not lying; the NAME is louder than the
doc.** Say in `validate`'s doc what it does not check, and name `tensor_bytes`
as the only answer to "are these bytes present". This is the same defect
class as the `TensorDeclined` omission promise — prose or a name asserting a
guarantee nothing tests.

### Recorded, NOT scheduled: `family` has no registry

`UnrecognizedKind::TensorEncoding::family` is a bare `&'static str` and now
carries two different KINDS of name: `"ggml"` is a **type system** that is
not the container format, `"safetensors"` is the **container format**,
because its dtype-string space belongs to it. Both satisfy the field's doc.
A consumer matching on the string gets one name from each of two categories,
and nothing stops a third backend choosing a colliding one.

**Task 7 must rule on this: open string or closed set.** It is recorded here
so the whole-branch review cannot miss it, and deliberately not fixed inside
a task whose file list does not name the decision.

---

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

## Task 4b: `SafetensorsMetadata` moves to its own module

**Before Task 5, so Task 5's imports are written once.**

Task 4 put `SafetensorsMetadata` in `crates/mlmf-safetensors/src/lib.rs`
because its brief said so. **The brief was wrong**: the crate has
`dtype.rs`, `error.rs`, `header.rs`, `tensors.rs`, and `mlmf-gguf` has
`metadata.rs`. This is the only crate in the workspace where a public type
lives in `lib.rs` beside the module list.

**Files:** create `crates/mlmf-safetensors/src/metadata.rs`, modify
`crates/mlmf-safetensors/src/lib.rs`.

A rename plus a `mod` and a `pub use`. Its own commit and its own gate run:
**a move that silently drops a `pub use` changes the crate's public surface
with every test still green**, which is the exact hazard recorded for Task 7
under `dtype_of`'s reachability.

---

## Task 5 ruling: the test gets its own crate, `crates/mlmf-conformance`

**Task 5's implementer found the file cannot live in
`crates/mlmf-safetensors/tests/`.** It needs `mlmf-gguf`, which needs
`[dev-dependencies]`, which
`mlmf-core/tests/deps.rs::no_table_other_than_plain_dependencies_may_declare_an_edge`
rejects on the TABLE, so no allow-list entry can satisfy it. Committing the
test without the dependency fails to compile. Either way the tree goes red,
and it correctly committed neither.

**Ruled: option (b), a new gated member. NOT a dev-dependency exception.**

The implementer recommended (a), allowing `[dev-dependencies]` behind its own
allow-list, on the grounds that the gate's stated rationale covers C5/codegen
and platform-invisibility but says nothing about dev-deps. **That reading of
the gate is correct and the conclusion is still wrong**, for two reasons it
could not have measured:

**1. The architecture.** A dev-dependency from `mlmf-safetensors` to
`mlmf-gguf` puts an edge between two sibling backends. **Neither backend
should know the other exists** — that is what a seam is. It is also
arbitrary: nothing says why safetensors would be the one that knows about
GGUF rather than the reverse. **The cross-backend test is a CONSUMER test.**
It does what Fuel and Lightbulb do — reach both backends through
`mlmf-core` — so it belongs in a crate shaped like a consumer, depending on
all three through plain `[dependencies]`. Inside a backend it is a backend
testing its sibling, which is not a relationship the seam describes.

**2. Every risk (b) was rejected for is already gated.** The implementer's
stated cost was *"a new crate whose job is missing from ci.yml produces a
test that never runs while everything looks green."* **Measured:
`common::gated_members()` returns every directory under `crates/` holding a
`Cargo.toml` — there is no opt-in.** So a new member is automatically forced
into `ci_coverage.rs`'s CI-steps check (`crates/ contains members that CI
does not gate`), automatically panics in `deps.rs::allow_list` until it has
`tests/direct-deps.allow`, and automatically falls under the C3 I/O gate.
**The failure mode it was avoiding cannot occur here**, because
`local-gates.sh` and `ci_coverage.rs` exist precisely for it.

**And the gate stays as it is.** Its module header says *"an unanticipated
form fails loudly instead of being skipped"* — this IS that design working.
The right response to a loud failure is to stop needing the exception, not to
widen the gate until the failure stops.

**C7 is satisfied, not waived:** `version.workspace = true` and
`publish = false` are orthogonal fields. The crate declares the first (the
gate reads that field) and the second because it must never ship. Say so in
the manifest.

---

## Task 5: The cross-backend test, which is what this whole plan was for

**Files:** `crates/mlmf-safetensors/tests/cross_backend.rs`.

This is the only test in the project that can fail because an abstraction is wrong rather than because an implementation is.

- [ ] **Step 1: Write it.** Build a GGUF and a safetensors file that declare **the same logical model** — same tensor names, same shapes, same dtype, same one metadata key. Then assert, **through `&dyn MetadataSource` and `&dyn TensorContainer` only**, with no concrete type in scope:
  - `tensors()` yields the same names from both. **NOT the same order** —
    that assumption was written before either backend existed and Task 2
    measured it false: `mlmf-safetensors` yields lexicographic order,
    because `serde_json` parses the header into a `BTreeMap`, while
    `mlmf-gguf` yields declaration order from a forward walk. Compare sorted
    sets, and state in the test that order is per-format because the seam
    does not promise one.
  - **`keys()` compared as sorted sets too, for the same reason.** Task 4
    measured the identical divergence in a second method:
    `mlmf-safetensors` yields lexicographic order (`BTreeMap` again) and
    `mlmf-gguf` yields declaration order. **Order is now unpromised in TWO
    methods, not one** — say so in the test rather than letting the next
    reader rediscover it.
  - **`Declaration::Unreadable` from BOTH backends.** Task 4 built the
    third state for safetensors (a `__metadata__` value that is not a
    string); `mlmf-gguf` already had it (a value it cannot decode).
    Verified before scheduling: `mlmf-gguf/src/metadata.rs:393` overrides
    `declaration()` and returns `Unreadable`. **Assert that a key present
    but undecodable is distinguishable from an absent key in both** — that
    is the conflation `Declaration` exists to prevent, and until Task 4 it
    could only be shown in one backend.
  - **A tensor declared past the end of the file: KEPT and REPORTED by
    both.** Task 2c made this assertable. Before it, `mlmf-gguf` kept
    silently and `mlmf-safetensors` omitted, so the case could not be
    asserted at all.
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

### Three API-shape questions, deferred here on purpose

Each was found while doing a task whose file list did not name the decision.
They are recorded rather than fixed because **deciding a seam question inside
a task scoped to something else is exactly how `TensorDeclined`'s false
omission promise got written.**

- [ ] **`UnrecognizedKind::TensorEncoding::family` is a bare `&'static str`
      with no registry, and now carries two different KINDS of name.**
      `"ggml"` is a type system that is not the container format;
      `"safetensors"` is the container format, because its dtype-string space
      belongs to it. Both satisfy the field's doc. Nothing stops a third
      backend picking a colliding name. **Rule: open string or closed set.**
- [ ] **`dtype_of` is `pub` and nothing pins that it stays reachable.** Its
      tests are all `super::*` unit tests, so a change to the module's
      re-export would move the crate's public surface with the suite green.
- [ ] **The `UNSPELLED` reason in `mlmf-safetensors::dtype` is unconstrained.**
      The gate requires a `&str` beside the type and never reads it;
      `(DType::F16, "")` passes. **A field whose presence is checked and whose
      content is not** — the same shape as a report `reason`, one level down.
      A non-empty assertion is barely better than none; decide whether these
      reasons are for people or for machines before adding one.

### And a methodology note this plan earned

**Task 3's TDD steps were stale, and the failure mode is worth naming.** The
plan gave it "write the failing test / stub / implement", but Task 2 had
already built `dtype_of` complete. **Following those steps literally would
mean deleting working code to watch a test go red — sabotage wearing a TDD
label**, which proves the test can fail but tells you nothing about whether
the code was ever right.

**When a plan's TDD steps meet code that already exists, the honest shape is
write the tests, run them GREEN, then sabotage.** The red comes from the
mutation, not from an artificial absence. Task 4 was checked and does NOT
have this problem — `mlmf-safetensors` has no `MetadataSource` impl — so this
is a note about reading plans, not a systemic staleness.

---

## Deliberately not in this plan

Each labelled by kind, because "we are not doing X" and "we measured and X is not available" look identical once written down.

- **Sharded models** (`model-00001-of-00002.safetensors` + index JSON) — **charter decision.** That is file discovery, and how a file is obtained is explicitly not this crate's business.
- **`config.json`** — **charter decision.** Interpreting a model's configuration is the consumer's job; this crate reads the file it is given.
- **Writing safetensors** — **design decision.** The authored builder emits malformed output on request, which is exactly what a writer must not do.
- **PyTorch pickle** — **design decision**, planned separately in the spec; it needs the borrow-or-own `Cow` path `tensor_bytes` already carries.
- **Proving the abstraction** — **measured premise.** Two backends is a better sample than one and it is still two. What this plan buys is the ability to be *wrong* in a way one backend cannot express.
