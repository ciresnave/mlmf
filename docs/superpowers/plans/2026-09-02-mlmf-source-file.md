# `mlmf-source-file` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The first crate on the **source axis**, so that a consumer of this workspace can open a file without writing the I/O themselves.

**Architecture:** Spec §3.1 draws two orthogonal axes — format crates are `bytes → structure` and do no I/O; source crates are I/O only and know nothing about formats. Every crate built so far is on the format axis. This crate implements `mlmf-core`'s `ByteSource` and `RangedSource` over the filesystem, with `memmap2` behind a **default** feature and a plain-read path that must keep working without it.

**Tech Stack:** Rust 2024, `mlmf-core`, `memmap2` (default feature only).

**Spec:** `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md` — §3.1 (axes), §3.2 (layout/acquisition split), §3.4 (mmap), C3, C6, C7, §11.

> **This plan was audited before implementation and rewritten. Six blocking findings, all mine.** The most consequential: the first draft claimed both C3 gates "forbid `memmap2` and `std::fs`". **They do not forbid `std::fs`** — see the box below. Two more claims are corrected inline and marked. Read the corrections; they are the parts most likely to be wrong again.

---

## Global Constraints

- **C3 is scoped to the format axis** — verbatim: *"No crate **on the format axis** references `std::fs`, `memmap2`, or any network client."* §3.4: *"`memmap2` is a **default** feature of `mlmf-source-file`."*
- **C6.** *"CI builds **and runs** the full parser suite with `--no-default-features`, proving the mmap-free path is functional rather than merely compilable."*
- **C7.** `version.workspace = true`.
- **Sources depend on `mlmf-core` only, never on a format crate** (§3.1). This is what makes `any source × any format` compose.
- **No format knowledge.** This crate enumerates and reads; it does not interpret. §3.2 puts checkpoint structure in `mlmf-hf-layout`, which *"never enumerates a directory"*.
- Every gate green by exit status before every commit: `bash scripts/local-gates.sh`. **Never pipe a gate.** `cargo fmt --all` before every commit. Restore a sabotage with `cp`, never `git checkout --`.

---

## ⚠️ What actually blocks this crate — measured, and narrower than it looks

**Only one thing in the gates is axis-blind: the crate name `memmap2`.**

`std::fs`, `std::io` and `std::path` are **not forbidden by any gate.** `purity.rs::FORBIDDEN_CRATES` is fifteen *crate* names with no `std::` entry; `std` is checked against **each crate's own `tests/allowed-std.list`**. A crate gets `std::fs` by writing `fs` in its own list, exactly as `mlmf-safetensors` gets `iter` today. **There is nothing to relax and no gate to change for the `std` half.**

Measured in an isolated copy of the gates: a fake source crate using `std::fs`, `std::io`, `std::path` **and** `memmap2`, with `fs`/`io`/`path` in its allow-list, produced **two violations, both `memmap2`**. With the allow-list narrowed to `path` alone, the same scan produced six, naming `std::fs` and `std::io` — so the scanner would have found them had they been forbidden. **They are unlisted by choice, not forbidden.**

⚠️ **And the modules this crate actually needs are `fs`, `ops`, `path`, `ffi`
— not the `fs`/`io`/`path` triple the probe above used.** Measured by
running the real `purity.rs` over a realistic `src/` written to this plan's
own interfaces: with only `fs`/`path` allowed it reports `std::ops`, twice,
because `RangedSource::read_range(&self, range: Range<u64>, …)` forces
`use std::ops::Range;` — `traits.rs:8` does the same in core. `ffi` is added
by the `OsString` ruling in Task 4. **And `fmt` is required too, which the probe missed.** `Result::expect_err`
needs `T: Debug`, so the crate's own first error-path test does not compile
without a `Debug` impl on `FileSource`, and no spelling of one avoids
`std::fmt`. **Hand-write it** — a derived `Debug` formats every byte of a model
file into a panic message. *(Measured by Task 1's implementer. The probe that
produced `fs`/`ops`/`path`/`ffi` modelled a `src/` with no `Debug` impl, which
is a thing a real crate cannot be.)*

**`io` is not reached PROVIDED the error conversion leaves its
closure argument unannotated.** Measured both ways: `.map_err(|e| …Box::new(e))`
scans clean; `.map_err(|e: std::io::Error| …)` reports *"path names `std::io`"*.
Annotating a closure argument to make a conversion readable is an ordinary
thing to write, so this is a property of one spelling, not of the crate.
*(`std::error` is genuinely not needed — `Box::new(e)` coerces without naming
the trait, because the `Box<dyn …>` type is declared in `mlmf-core`.)*
**Do not pre-populate `allowed-std.list` from that probe** — it was chosen
to test the gate, not to model this crate.

**Three gates stand between this crate and a green run, not two** — the third
only if the implementer reaches for the obvious tool:

| Gate | Why | Fixed by |
|---|---|---|
| `purity.rs::every_gated_crate_performs_no_io` | `memmap2` in `src/` | Task 0 (axis) |
| `deps.rs::the_manifest_names_no_io_crate_anywhere` | `memmap2` anywhere in `Cargo.toml` — **including the `[features]` table**, since the scan is `line.contains()` over every non-comment line | Task 0 (axis) |
| `deps.rs::no_table_other_than_plain_dependencies_may_declare_an_edge` | **any** `[dev-dependencies]` | **Not fixed. Do not add one.** |

**That third gate is not axis-related and Task 0 will not relax it.** The obvious way to write Task 1's test — `tempfile` under `[dev-dependencies]`, which is what the *root* `mlmf` package does and therefore what an implementer will copy from inside this repo — **is rejected.** Use **`env!("CARGO_TARGET_TMPDIR")`**: a compile-time macro, available to integration tests under `tests/`, requiring no dependency and no new `std` path. `mlmf-conformance`'s manifest records the ruling that this gate refusing a dev-dependency is *"the design working"*.

---

## Every precondition a new gated crate must satisfy — fourteen today, fifteen after Task 0

Derived by reading every test that iterates `common::gated_members()`. **The first draft of this plan named six.**

| # | Precondition | Gate |
|---|---|---|
| 1 | `crates/<name>/Cargo.toml` exists | `gated_members()` — no opt-in; any dir with a manifest |
| 2 | `version.workspace = true` | `workspace.rs::every_package_carries_the_one_workspace_version` |
| 3 | **No `[dev-dependencies]`, `[build-dependencies]`, or `[target.*.dependencies]`** | `deps.rs::no_table_other_than_plain_dependencies_may_declare_an_edge` |
| 4 | No `build.rs` | `workspace.rs::only_allow_listed_packages_run_a_build_script` |
| 5 | No `FORBIDDEN_CRATES` name on any non-comment manifest line, **`[features]` included** | `deps.rs::the_manifest_names_no_io_crate_anywhere` |
| 6 | `tests/direct-deps.allow` exists **and equals the sorted dep list exactly** | `deps.rs::allow_list` + `direct_dependencies_match_allowlist` |
| 7 | `tests/allowed-std.list` exists; every `std::X` reached from `src/**` listed | `purity.rs::allowed_std` |
| 8 | `src/` exists with ≥1 `.rs` file | `purity.rs` (`assert!(!files.is_empty())`) |
| 9 | No `src/**` file names a `FORBIDDEN_CRATES` entry | `purity.rs::every_gated_crate_performs_no_io` |
| 10 | **Every `src/**/*.rs` except `lib`/`main`/`mod` named by a `mod` declaration** | `module_registration.rs` |
| 11 | Root `default-members` contains `crates/<name>` | `workspace.rs::every_gated_crate_is_reachable_from_a_bare_cargo_test` |
| 12 | ci.yml has `cargo test -p`, `cargo doc -p … --no-deps`, `cargo clippy -p … --all-targets` | `ci_coverage.rs::every_gated_crate_is_tested_documented_and_linted_by_ci` |
| 13 | **Exactly one `- name:` step per crate whose `run:` starts `cargo doc`, each carrying `RUSTDOCFLAGS: -D warnings` in its own `env:`** | `ci_coverage.rs::rustdoc_warnings_are_fatal_for_every_documented_crate` |
| 14 | Any test writing to `std::io::stderr()` names `NOTICE_TOKEN` **or contains the token literally** | `skip_notice.rs` |
| **15** | **`tests/axis` exists and reads exactly `format` or `source`** | **`common::axis()`, created by Task 0 — a loud panic without it** |

⚠️ **Row 15 is created by this plan and exists nowhere else in the ecosystem**,
so the next crate author will not know about it. It is in this table because
this table is the artefact they will read.

**Note on 7:** `purity.rs` scans `src/**` *including `#[cfg(test)]` modules*, and does **not** scan `tests/`. **This crate puts its tests in `tests/`**, so `allowed-std.list` records only what production code reaches. `mlmf-safetensors`' own list documents the opposite case for `iter`.

---

## `FileSource` is materialized. Decided here, once.

**The first draft left this open and three tasks contradicted each other about it.**

`FileSource::open` produces the whole file's bytes — a `Vec<u8>` without the `mmap` feature, a `Mmap` with it. `as_bytes()` returns that slice. **`read_range` is a bounds-checked copy out of that slice. There is no `File` handle, no `seek`, and no cursor.**

**Why, and it is not a shortcut.** `RangedSource::read_range` takes `&self`. A `seek`+`read_exact` implementation compiles — `std` has `impl Seek for &File` — but it mutates a **shared file cursor through a shared reference**, so two concurrent calls interleave their seeks and return the wrong bytes **with `Ok(())`**. The broken state is indistinguishable from the healthy one. A slice copy has no such state.

**And "materialized" is the wrong worry for mmap:** the OS pages a mapping lazily, so the default path does not hold the file in RAM. `RangedSource`'s doc explains it exists so an *HTTP range* or *IPC* source is expressible later. **That is a fact about other implementations, not a promise this one must keep by avoiding a slice.**

⚠️ **In TASK 3's crate doc — not before — state plainly that `RangedSource`
offers no memory benefit on the `--no-default-features` build.** *(Task 1
correctly declined this: at that point there is no `RangedSource` impl and no
features table, so both halves would be claims about code that is not there.
Task 1 states what is true then — every path materializes the whole file — and
this contrast lands with the feature that creates it.)* That path allocates the whole
file through `std::fs::read`. The lazy-paging argument above is true of the
default build only, and C6 is prominent enough in this plan that a reader
will otherwise carry the hedge across to the build C6 exists to protect.

---

## File Structure

- `crates/mlmf-core/tests/purity.rs`, `deps.rs`, `common/mod.rs`, `ci_coverage.rs` — gate changes.
- `crates/<each existing>/tests/axis` — five files, `format`.
- `crates/mlmf-source-file/` — `Cargo.toml`, `src/{lib,file,dir}.rs`, `tests/{axis,direct-deps.allow,allowed-std.list,bytes.rs,ranged.rs,mmap.rs,dir.rs}`.
- Root `Cargo.toml` (`default-members`), `.github/workflows/ci.yml`.

---

## Task 0: Give the two C3 gates the axis the spec already wrote

**Files:** `crates/mlmf-core/tests/{purity.rs, deps.rs, common/mod.rs}`, and `crates/{mlmf-core,mlmf-ggml,mlmf-gguf,mlmf-safetensors,mlmf-conformance}/tests/axis`.

**Interfaces produced:** `common::axis(dir) -> Axis`, `Axis::{Format, Source}`.

**Scope, corrected:** the relaxation is **`memmap2` only**. Do not touch the `std` allow-list machinery.

**`mlmf-conformance` is assigned `format`.** It is a consumer-shaped crate with no library code and §3.1's diagram does not contain it, so this is a ruling rather than a reading. It links two format crates and does no I/O; `format` is the stricter of the two and therefore the safe assignment.

- [ ] **Step 1: Make both scanners axis-aware, and they are NOT symmetric.**

      **`deps.rs` needs an extraction.** Its forbidden-crate scan is written
      inline in the `#[test]` body. Move it to `scan_manifest(label, text, axis)`.
      **Verified safe:** the extraction was performed in an isolated copy and
      `the_gate_can_fail` and `the_gate_does_not_cry_wolf` both stayed green,
      because they drive `parse_manifest`, which it does not touch.

      ⚠️ **`purity.rs` needs no extraction and is the side that breaks.** It
      already has `scan_text(label, src, allowed)`, but threading an axis
      through it touches **three existing call sites** — `purity.rs:545`,
      `:559`, `:588` — two inside `the_gate_can_fail` and one inside
      `the_gate_does_not_cry_wolf`. **Pin all three to `Axis::Format`.**

      **And `FORBIDDEN_CRATES` is consulted TWICE inside `scan_text`, not
      once** — in `check_pair`, and again in the `toks.windows(2)` loop that
      catches `extern crate memmap2;`. *(Found by Task 0's implementer, not by
      five audit passes.)* **Write the relaxation as one `is_forbidden(name,
      axis)` helper and route both through it.** Inlining
      `&& !(axis == Source && root == "memmap2")` into `check_pair` is the
      obvious minimal edit and leaves `extern crate memmap2;` rejected on the
      source axis — an inconsistency with no test against it. The helper is
      also what makes Step 6's first sabotage a clean single-site deletion;
      without it, "delete the relaxation branch" has two branches.

      **Passing `Axis::Source` at the self-test sites is the plausible mistake**,
      because the file's other call site passes the crate's own axis. Measured
      consequence:

          test the_gate_can_fail ... FAILED
          case 6 slipped through the C3 gate:
          use memmap2::Mmap;

      Cases **6** and **11** of `must_be_rejected` are memmap2-only, so they are
      the two the relaxation would silently disarm — **and the failure names the
      fixture, not the axis, so the diagnosis points away from the change.**

- [ ] **Step 2: Write the failing tests, in fixtures.** A `source`-axis manifest naming `memmap2` in `[dependencies]` **and** in `[features]` is accepted; the same text on `format` is rejected twice. A `source`-axis `src` naming `memmap2` is accepted; on `format`, rejected. **A `source`-axis manifest naming `reqwest` is still rejected** — the control that proves the relaxation is scoped.

- [ ] **Step 3: Run, confirm failure** because `axis` does not exist.

- [ ] **Step 4: Implement.** `common::axis()` reads `<crate>/tests/axis`, trims — **load-bearing, and measured rather than
  defensive: `core.autocrlf` is `true` in this checkout and `.gitattributes`
  has no rule for `tests/axis`, so the file arrives with CRLF. Verified green
  with CRLF written in.** *(Task 0's implementer declined to add a
  `.gitattributes` pin, reasoning it would make the trim look decorative. The
  repo carries prior line-ending scars, so Task 6's reviewer should confirm
  that call rather than inherit it.)* —, and accepts exactly `format` or `source`. **A missing file is a loud panic**, matching `deps.rs::allow_list`, whose doc says why. *(`purity.rs::allowed_std` panics identically but its doc does not say so — do not cite it as precedent.)*

  **Relax `memmap2` and nothing else.** Note there are **two** `FORBIDDEN_CRATES` constants and they do not match.
**Change both, and do not assume one list is the other.** The asymmetry,
measured with `comm` rather than described:

    only in purity.rs:  async_std  native_tls          (underscore forms)
    only in deps.rs:    async-std  native-tls  hf-hub  prost-build  protobuf-codegen

`smol`, `mio` and `libloading` are in **both** — an earlier revision named
them as purity-only, which would have sent someone editing `deps.rs`
looking for entries that were already there and missing the two that
differ only by a hyphen. `deps.rs` carries **both** `hf-hub` and `hf_hub`.

- [ ] **Step 5: Add the five `axis` files** (`format`) and run the full gate set.

- [ ] **Step 6: Sabotage.**
  1. ⚠️ **Delete the relaxation branch itself** — the `axis == Source && crate
     == "memmap2"` arm — and confirm the Step 2 source-axis fixtures redden.
     **Do NOT also test "`axis()` returns `Format` unconditionally" here — it
     is a no-op at Task 0.** Measured: baseline 4 passed; delete the branch →
     `step2_source_axis_fixture` FAILS; force `Format` → **4 passed, nothing
     moves.** Step 2's fixtures pass an `Axis` *value* straight into
     `scan_text`; `axis()` is the *file reader*, reachable only through
     `every_gated_crate_performs_no_io` over real crates — and at Task 0 no
     real crate names `memmap2`. **That mutation has no subject until Task 1
     exists, so it moves to Task 1 Step 5.**

     *(An earlier revision's item 1 was "flip a real crate's axis to `source`
     and confirm the scanner stops rejecting `memmap2`" — a no-op, since
     `grep -rn memmap2 crates/*/src/` returns nothing, so nothing is being
     rejected and the test is green either way. The fix removed the no-op and
     left the slot empty: items 2-4 sabotage the PARSER and the SCOPE, and
     nothing was left mutating the permission. Steps 2-3 already drive the
     fixtures; restating them under "Sabotage" is not a sabotage.)*
  2. Delete an `axis` file; confirm the panic names it.
  3. Write `Source` (wrong case); confirm it panics rather than silently reading as `format`.
  4. **Widen the arm to `crate_name == "memmap2" || crate_name == "reqwest"`**
     and confirm the scope control reddens on both scanners.

     *(An earlier revision wrote this as "`reqwest` on the `source` axis;
     confirm still rejected" — a restatement of a Step 2 assertion, not a
     mutation. That is the same shape item 1 of this list was fixed for, and
     the parenthetical explaining it sat two lines above. Found by Task 0's
     implementer, who executed it as a mutation because it was not executable
     as written.)*

- [ ] **Step 7: Commit.**

---

## Task 1: The crate, and the path that works without mmap

**Written before mmap deliberately.** C6 exists to prove the mmap-free path is *functional rather than merely compilable*, and a path written second is a path written to match the first.

**Files:** `crates/mlmf-source-file/{Cargo.toml, src/lib.rs, src/file.rs, tests/axis, tests/direct-deps.allow, tests/allowed-std.list, tests/bytes.rs}`, root `Cargo.toml`, `.github/workflows/ci.yml`.

**Interfaces produced:**
```rust
pub struct FileSource { /* Vec<u8>; becomes Vec<u8>|Mmap in Task 3 */ }

impl FileSource {
    /// Whole file, by whatever path this build was compiled for.
    pub fn open(path: &Path) -> mlmf_core::Result<Self>;
    /// Whole file, ALWAYS by plain read, regardless of features.
    ///
    /// **Deliberate permanent public API, not test scaffolding** — though a
    /// test is what forced the question. Task 3's equality assertion lives in
    /// `tests/`, so it can only reach `pub` items, and comparing the mmap path
    /// against a `std::fs::read` written in the test would compare the crate
    /// against a third implementation rather than against itself. It also has
    /// a real caller: forcing a plain read on a network mount, where mmap
    /// semantics are hostile.
    pub fn open_read(path: &Path) -> mlmf_core::Result<Self>;
}

impl mlmf_core::ByteSource for FileSource {
    fn as_bytes(&self) -> &[u8];
}
```

`mlmf_core::Result<T> = std::result::Result<T, mlmf_core::Error>`. Errors use
**`ErrorKind::Source`** with `Error::with_path`. *(The warrant is narrower than
the rule: `traits.rs:54-57` commits **`read_range`** to that variant and says
nothing about `open` or `read_dir`. Extending it to them is a decision made
here for consistency, not something the seam already ruled.)*

- [ ] **Step 1: Write the failing tests** in `tests/bytes.rs`. Known bytes round-trip exactly; a zero-length file gives an empty slice and **not** an error; a file with an embedded NUL is unchanged. Temp files via `env!("CARGO_TARGET_TMPDIR")` — **no `tempfile`, no dev-dependency** (precondition 3).

- [ ] **Step 2: Run, confirm failure.** **Expect a COMPILE error** —
      `E0432: unresolved import mlmf_source_file` — not a failing assertion.
      Worth saying because every later step in this plan distinguishes the two
      and this one did not. *(Also measured: `cargo` accepts a package whose
      only target is `tests/bytes.rs`, so this step works with no `src/` at
      all.)*

- [ ] **Step 3: Implement** with `std::fs::read`. No mmap, no feature yet. `tests/axis` = `source`. `tests/direct-deps.allow` = `mlmf-core`. `tests/allowed-std.list` = the `std` modules `src/` reaches, **one per line with a comment saying what forced it**, as every sibling list does.

- [ ] **Step 4: Wire the crate in, then run.** Add to root `default-members`, and add **four** CI steps: `cargo test -p`, `cargo test -p … --no-default-features` (C6, and Task 5 will gate it), `cargo doc -p … --no-deps` **in its own `- name:` block with `env: RUSTDOCFLAGS: -D warnings`**, and `cargo clippy -p … --all-targets`. **Precondition 13 counts doc steps per crate and checks each carries that env.** *(The first draft said "run, confirm pass, then add the CI steps" — impossible: the gates go red the moment the manifest exists.)*

- [ ] **Step 5: Sabotage.** Truncate the read by one byte → the exact-bytes test reddens. Return `Vec::new()` unconditionally → the non-empty test reddens **while the zero-length test stays green**, which is the control proving the empty case is not carrying the assertion.

- [ ] **Step 6: Commit.**

---

## Task 2: `RangedSource`, over the bytes already held

**Files:** `crates/mlmf-source-file/src/file.rs`, `tests/ranged.rs`,
`tests/allowed-std.list`.

**`FileSource` gains a `path: PathBuf` field here.** Task 1's block puts
`Error::with_path` in the constructor, where `open` is holding the path —
`read_range` receives only a range, and Task 1 correctly shipped
`FileSource { bytes }`. Without the field every out-of-range read says *"byte
range 248..257 is outside a 256-byte file"* **with no way to say which of a
model's shards it was about.** *(Found by executing Task 2; neither this Files
list nor the decision box mentioned it.)*

**Interfaces consumed:**
```rust
fn len(&self) -> Option<u64>;
fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> mlmf_core::Result<()>;
fn is_empty(&self) -> Option<bool>;   // DEFAULTED by the trait — do not implement,
                                      // and do not write a test that treats it as new.
```

- [ ] **Step 1: Write the failing tests.** A middle range; a range ending **exactly** at EOF; a range one byte past EOF (**an error, `ErrorKind::Source`, not a short read**); an inverted range — **`clippy::reversed_empty_ranges` is deny-by-default and
  rejects `read_range(96..64, …)` as anyone would first write it.** `cargo
  test` compiles it, so it survives Step 4 and dies only at Step 5's gate run.
  Build the range field-by-field. *(The lint is right about typed ranges and
  blind to the case that matters: nobody types an inverted range — it arrives
  as two computed offsets from a header this crate never reads.)*; `into` shorter than the range; `into` longer; `len()` for a real and a zero-length file.

  **The at-EOF / past-EOF pair is not optional.** Both corpus differentials in this repo caught a `>` vs `>=` off-by-one on real models — an 88,202,080-byte GGUF whose last tensor ends on the final byte, and a 723,674,912-byte safetensors. The last tensor of a well-formed file touches the last byte every time.

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement as a bounds-checked copy** out of `as_bytes()`. **No `seek`, no `File` handle** — see the decision box above. Do not clamp a range to the file's length: that converts a caller's arithmetic error into a short read they cannot see.

- [ ] **Step 4: Run.**

- [ ] **Step 5: Sabotage.** `>` → `>=` on the end bound: confirm **exactly** the at-EOF test reddens and past-EOF stays green — identify which assertion fires, not merely that one does. **Then clamp AND short-read** — `range.end.min(available)` alone reddens
      **nothing**, measured: 9 passed, 0 failed. The `into.len() == width`
      check catches the clamped read for the wrong reason — nine bytes asked
      for, eight after clamping, so the caller's nine-byte buffer no longer
      matches — and past-EOF still passes, on a fact that has nothing to do
      with the range. **A success criterion satisfiable without doing the
      work.**

      Task 2's implementer found it and added the test that removes that second
      line of defence: **a past-EOF range whose buffer is sized to what the
      file CAN give.** Against that, the pure clamp reddens alone; spelled as
      clamp-and-short-read it reddens three.

- [ ] **Step 6: Commit.**

---

## Task 3: mmap, behind the default feature

**Files:** `crates/mlmf-source-file/{Cargo.toml, src/lib.rs, src/file.rs, tests/mmap.rs, tests/direct-deps.allow}`.

`src/lib.rs` is in the list because Step 3 says *"say so in the crate doc"* —
the crate doc is `lib.rs`'s `//!` header.

⚠️ **`tests/direct-deps.allow` is in this list because `direct_dependencies_match_allowlist` is an exact `assert_eq!` against the sorted dependency list.** Adding `memmap2` reddens it until the file reads `memmap2` then `mlmf-core` — **in sorted position**, because the comparison is against a vector, not a set.

- [ ] **Step 1: Write the failing test** in `tests/mmap.rs`, opening `#![cfg(feature = "mmap")]`. Assert `FileSource::open(p).as_bytes() == FileSource::open_read(p).as_bytes()` for a multi-page file, a one-byte file, and a zero-length file. **Compare the crate's two paths against each other, not against a literal** — a literal lets both drift together, and a `std::fs::read` written in the test is a third implementation, not the crate's.

  ⚠️ **The reason given for that header — "so Task 5's `--no-default-features`
  run does not fail to compile" — is FALSE, measured.** Nothing in this file
  is feature-gated: the mapping lives behind a *private* constructor and this
  task adds no public API, so `open`, `open_read` and `as_bytes` all exist on
  both builds. With the header deleted, `cargo test -p mlmf-source-file
  --no-default-features --test mmap` **compiles and reports 4 passed.**

  **The header is still required, for a different reason.** On that build
  `open` *is* `open_read`, so those four greens assert `open_read ==
  open_read` — four tests named after a mapping, passing on a build with no
  mapping, inside the step C6 exists to make meaningful. It is a vacuity
  guard, not a compile guard. *(Found by Task 3's implementer. A reason that
  is wrong about which failure it prevents survives every review that agrees
  with the conclusion.)*

- [ ] **Step 2: Run, confirm failure.** ⚠️ **There is no failure to confirm,
      in either spelling, and Task 3's implementer measured both.** With the
      header and no feature declared yet, the file compiles to nothing —
      *"running 0 tests … ok"*, exit 0, one `unexpected_cfgs` warning. With
      the header removed, **all four tests run and pass**, because Task 1
      shipped `open` delegating to `open_read`: the two paths this file
      compares are one path until Step 3 splits them.

      **This test cannot be born red and the plan should have said so.** Its
      power is demonstrated at Step 5 and nowhere else, which is exactly the
      shape AD-2 exists to catch — so run the mutation before believing the
      file. Do not manufacture a red by asserting something else instead: an
      equality between two acquisition paths has no red state before the
      second path exists.

- [ ] **Step 3: Implement.** `memmap2` **optional**, `default = ["mmap"]`, `mmap = ["dep:memmap2"]`. `Mmap::map` is `unsafe` (verified: `memmap2 0.9.11`, `pub unsafe fn map`), so this crate **cannot** carry `#![forbid(unsafe_code)]`.

  **Say so in the crate doc with the reason.** *(The first draft justified this as "every sibling crate forbids it". Measured: **four of five** — `mlmf-core`, `mlmf-ggml`, `mlmf-gguf`, `mlmf-safetensors` do; `mlmf-conformance` has no crate-level attributes at all. And **nothing gates it**, so this is a convention worth honouring in prose, not a rule being broken.)* **Re-measured at Task 3 rather than copied: still four of five.**

  **Shipped as `#![deny(unsafe_code)]` plus one `#[expect(unsafe_code)]` on
  the private `open_mmap`**, which is not what "cannot carry the forbid"
  reads as. `forbid` cannot be overridden anywhere; `deny` can be, at exactly
  one site that has to name the lint and carry a reason. Every other `unsafe`
  in the crate stays a compile error, which is the half of the convention
  that was still available. **Checked, not asserted:** deleting the `expect`
  is a compile error — *"usage of an `unsafe` block … the lint level is
  defined here `#![deny(unsafe_code)]`"* — so the crate doc's claim that the
  deny is in force is a verified one.

- [ ] **Step 4: Run with default features and with `--no-default-features`.**

- [ ] **Step 5: Sabotage.** **First, the axis mutation — relocated here from
      Task 0, and then from Task 1, which was still one task too early.** Make
      `common::axis()` return `Format` unconditionally and confirm
      `every_gated_crate_performs_no_io` reddens naming `mlmf-source-file`.

      **This is the first moment it has a subject.** Task 0 had no crate naming
      `memmap2`; **Task 1 has the crate but still does not name `memmap2` in
      `src/`** — that arrives here, with the mmap arm. Measured at Task 1: the
      mutation left `purity` 4 passed and `deps` 6 passed, neither red nor a
      compile error. Task 1's implementer established it was a missing subject
      rather than a broken relaxation by previewing this task's `memmap2` line
      as text — `purity.rs` reads `src/**` without compiling it — at which
      point the same mutation reddened naming this crate.

      **Executed, and it reddens — on BOTH gates, not only the one named
      here.** `every_gated_crate_performs_no_io` fails with *"mlmf-source-file:
      src/file.rs: path names the I/O crate `memmap2`"* **twice** — the enum
      variant's field type and the `Mmap::map` call are two paths in the
      source text — and `the_manifest_names_no_io_crate_anywhere` fails naming
      the `[dependencies]` line **and** the `[features]` line, which is the
      `line.contains()` claim finally observed against a real manifest rather
      than a fixture. Both are test failures, not compile errors. **The
      relocation was right twice.**

      Then: map with an off-by-one length → the equality test reddens. **Measured:
      `MmapOptions::len(file_len - 1)` reddens NINE tests across three files** —
      `mmap.rs` 3 of 4, `bytes.rs` 2 of 5, `ranged.rs` 4 of 10, all assertion
      failures — because `bytes.rs` and `ranged.rs` call `open`, so they run
      against the mapping by default and against the owned buffer under the
      flag. The one-byte case fires on the **equality**, `left: [] right:
      [42]`, not on the length control beneath it.

      ⚠️ **The zero-length case stays GREEN under that mutation and cannot do
      otherwise** — a length off-by-one on a zero-length file saturates back to
      zero. It is in the file for a different question (whether `open` refuses
      a file neither platform can map: POSIX fails `mmap` at `len == 0`,
      `CreateFileMappingW` rejects one with `ERROR_FILE_INVALID`, and memmap2
      handles both itself), **not as a detector of length bugs.** Three of the
      four cases carry that assertion; say which in the test, or the green one
      reads as a fourth witness.

      Then confirm `--no-default-features` still compiles and Tasks 1–2 still pass — **C6's actual claim, observed rather than assumed.**

- [ ] **Step 6: Commit**, then confirm C1/C2 are unmoved by a sibling gaining a
      dependency. **The instrument is `cargo test -p mlmf-core --test
      transitive_deps` or `bash scripts/check-deps.sh`** — both already in the
      18-gate set. *(Not a bare `cargo tree -p mlmf-core`, which an earlier
      revision named: `transitive_deps::current()` runs it with
      `--edges normal,build --no-default-features --target all --prefix none
      --color never` and diffs the snapshot. A different query does not
      reproduce it.)*

---

## Task 4: Directory enumeration, with no format knowledge

§3.2: *"`mlmf-source-file` walks a local directory"*; `mlmf-hf-layout` *"never enumerates a directory"*.

**Files:** `crates/mlmf-source-file/{src/lib.rs, src/dir.rs, tests/dir.rs, tests/allowed-std.list}`.

⚠️ **`src/lib.rs` twice over:** `mod dir;` (precondition 10 — without it
`module_registration` reddens with *"these source files are not named by any
`mod` declaration, so they are never compiled and their tests never run"*),
and a `pub use dir::{DirEntry, read_dir};` for the unqualified paths in the
Interfaces block below.

**Interfaces produced:**
```rust
/// One entry, as the filesystem reports it. No interpretation.
pub struct DirEntry { pub name: std::ffi::OsString, pub is_dir: bool }

/// Immediate children only — NOT recursive — sorted by `name`.
pub fn read_dir(path: &Path) -> mlmf_core::Result<Vec<DirEntry>>;
```

⚠️ **`OsString`, not `String`, and this is spec §9 clause 2.1 one layer out.**
`std::fs::DirEntry::file_name()` returns `OsString`. A filename is not
guaranteed UTF-8 on Linux and can hold unpaired surrogates on Windows, so
`String` forces one of three silent choices: `to_string_lossy()` (a name
carrying U+FFFD that **cannot be passed back to `FileSource::open`** — a
listing whose entries cannot be opened), skipping the entry (the enumeration
omits a file and returns a healthy `Ok`), or failing the whole call because
one unrelated file has an odd name.

Clause 2.1 rules on exactly this class — *"round-trip **byte-exact**. No
Unicode normalization, case folding, trimming, or reordering — ever … the
failure is **silent**"* — and spec line 415 records why no corpus will catch
it: 4,686,500 strings scanned across 29 files, **zero non-UTF-8**. Task 4's
fixture is ASCII-only and is structurally blind the same way. `OsString` is
lossless, costs one allow-list line (`ffi`), and forces a consumer matching
against `index.json` to confront a name that cannot match rather than
silently matching a mangled one.

**Both fields are decided here**, because the first draft said "every file name" in one step and asked the test to assert a subdirectory appears in another — contradictory. `is_dir` is reported, not filtered on: §3.2's consumer is *"given a list of filenames"*, and deciding which entries count is the caller's job.

- [ ] **Step 1: Write the failing test.** A directory holding `model.safetensors`, `model.gguf`, `README.md` and a subdirectory `nested/`. Assert **all four** are returned, sorted, with `is_dir` true for exactly one. Assert a file inside `nested/` is **not** returned.

  ⚠️ **Two details of that fixture are load-bearing and the plan states
  neither. Both were measured by Task 4's implementer.**

  **The capital letters in `README.md` are what give "sorted" a red state.**
  An NTFS directory index returns entries in a case-folded order, so
  `README.md` comes back LAST from the OS and FIRST under the byte order
  `OsString: Ord` gives. Measured, by deleting the sort: the assertion
  reddens with `["model.gguf", "model.safetensors", "nested", "README.md"]`
  on the left. **Spell it `readme.md` and that sabotage reddens nothing** —
  measured on the same host, OS order and byte order become the same four
  names, so a `read_dir` that never sorts passes the assertion that says it
  does. The fixture is right; it is right by an accident of capitalisation
  that a tidy-up would undo.

  **And the recursion sabotage's only subject is the FILE inside `nested/`,
  not `nested/` itself.** "A subdirectory `nested/`" is satisfiable with an
  empty one, and against an empty one a `read_dir` that recurses returns the
  same four entries in the same order: measured, the all-four test stays
  **GREEN** under the recursion mutation and only the not-returned test
  reddens. The last sentence of this step is carrying the whole of Step 5's
  second sabotage.

  **And a name with NO `String` representation at all.** Not merely non-ASCII:
  `模型.safetensors` is valid UTF-8, so `to_string_lossy()` is the identity on
  it and **a `String`-typed field would round-trip it perfectly.** Measured —
  the intended sabotage came back `SABOTAGE DETECTED? : false`, `bytes equal :
  true`. Keep that name if you want a readability case, but it **is not the
  control**: it pins losslessness for a name that was never at risk, which is
  the vacuity mode this plan warns about, inside the fix for a finding about
  silent loss.

  **The control is an unpaired surrogate — U+D800 in the file name.**
  `#[cfg(windows)]`, three lines via `std::os::windows::ffi::OsStringExt::from_wide`;
  `std::os` is free here because `purity.rs` does not scan `tests/`. On Unix,
  `OsStrExt::from_bytes` with an invalid byte. **Measured on this host, so
  "platform-specific" does not mean impractical:** the file is creatable, the
  honest name reads back byte-exact, the lossy round-trip turns U+D800 into
  U+FFFD, and — the assertion worth writing — **reopening by the honest name
  succeeds while reopening by the lossy name is `NotFound`.** That demonstrates
  the OsString ruling's claim, *"a listing whose entries cannot be opened"*,
  rather than arguing it.

- [ ] **Step 2: Run, confirm failure.** **A COMPILE error, and worth naming
      because `EXIT=101` alone does not distinguish one from a red test:**
      `error[E0432]: unresolved import mlmf_source_file::read_dir` — *"no
      `read_dir` in the root"*. Not a failing assertion. Unlike Task 3's
      test, this one **does** have a real red state once it compiles; every
      sabotage at Step 5 produces one.

- [ ] **Step 3: Implement.** Names and a directory flag.

      ⚠️ **`is_dir` needs a symlink ruling and this plan does not make one.**
      *(Found by Task 4's implementer, who had to make it.)*
      `std::fs::DirEntry::file_type()` describes the ENTRY and does not
      follow a link; `std::fs::metadata(entry.path())` follows it. Shipped as
      `file_type()`, so a symlink pointing at a directory reports `false` —
      chosen for a consequence rather than for tidiness: resolving the target
      **fails on a dangling link**, so one broken symlink anywhere in a
      checkpoint directory would take the whole listing down with it. That is
      the third of the three losses the `OsString` ruling above rejects,
      arriving through the other field. The HF cache is built out of
      symlinks, so this is not a hypothetical shape. **Fully qualify
      `std::fs::read_dir` inside `src/dir.rs`** — this crate's own `read_dir`
      and `DirEntry` shadow the `std::fs` names, and an unqualified call
      recurses into itself. **No extension mapping, no sniffing, no guessing which file is a model** — that is interpretation, and the charter forbids it: *"MLMF is never intended to be an interpreter of the content of model files."*

- [ ] **Step 4: Run**, and update `allowed-std.list` if `read_dir` reaches a `std` module the crate did not already use.

- [ ] **Step 5: Sabotage.** Filter to `.safetensors` → the test names the three
      dropped entries. Recurse into `nested/` → the not-returned assertion
      reddens. **Replace `OsString` with `to_string_lossy().into_owned()` →
      the SURROGATE round-trip must redden, and the reopen-by-name assertion
      with it.** *(Against `模型.safetensors` this mutation is GREEN — measured.
      A sabotage that cannot fail against your fixture means the fixture is
      wrong, not that the code is right.)*

      **All three executed. The first two land as written** — the filter
      reddens two tests, the all-four assertion showing
      `left: ["model.safetensors"]` against a right-hand side naming all
      four; the recursion reddens one, listing
      `["buried.safetensors", "deeper", "model.safetensors", "nested"]`.
      Both are test failures, not compile errors.

      ⚠️ **The third is two different mutations and the sentence does not say
      which.** *(Found by Task 4's implementer.)* Read as a **type** change —
      `pub name: String`, which is what the whole ruling above is about — it
      is **nine compile errors in `tests/dir.rs`** (*"can't compare `String`
      with `OsString`"*), because the test was written against the honest
      type. Nothing reddens; there is no assertion left to redden. Read as a
      **value** change — the field stays `OsString`, constructed from
      `entry.file_name().to_string_lossy().into_owned()` — the surrogate test
      reddens alone, U+FFFD on the left against U+D800 on the right, and the
      counter-example test on `模型.safetensors` stays **green**, which is
      this plan's own falsification reproduced as a standing control in the
      suite rather than a measurement in a document. **Only the value form
      measures anything.**

      ⚠️ **And "the reopen-by-name assertion with it" cannot happen.** The
      byte-exact assertion fires first and a panic ends the test, so the
      assertion this plan calls *"the assertion worth writing"* is never
      reached by the mutation the plan pairs it with: present, never
      exercised. Shown to have a red state of its own by neutralising the two
      assertions above it and re-running the same mutation — *"the listed
      name must open the file: Error { kind: Source(Os { code: 2, kind:
      NotFound, … }), path: Some(…surrogate-U+FFFD.safetensors) }"*. Order
      the assertions knowing that, or the strongest one in the file is the
      one nothing proves.

      **A fourth sabotage, because "sorted" was a success criterion nothing
      mutated:** delete the sort → the all-four assertion reddens with the
      OS's own order. See Step 1 for why that only works because of a capital
      letter.

      **A fifth, because an allow-list entry is a permission and a permission
      needs a positive control:** remove `ffi` from `tests/allowed-std.list`
      → `purity.rs::every_gated_crate_performs_no_io` fails naming
      `src/dir.rs` **twice**, *"import names `std::ffi`"* and *"path names
      `std::ffi`"* — one `use` line seen by both of `scan_text`'s two scans.
      A test failure, not a compile error. The entry is forced rather than
      precautionary.

- [ ] **Step 6: Commit.**

---

## Task 5: C6 is under-enforced

**Measured:** `grep -n "no-default-features" .github/workflows/ci.yml` returns exactly two lines — the `- name:` and `run:` of **one** step, `cargo test -p mlmf-core --no-default-features`. **One crate.**

⚠️ **That measurement is stale as written, and only as a present tense.** It was taken before Task 1, which added a second step. Re-measured at Task 5 against `HEAD`: the grep returns **five** lines — the `- name:`/`run:` pairs of **two** steps, `mlmf-core` and `mlmf-source-file`, plus one comment line; **13** after this task. The **rule** in Step 2 was correct and its four names were exact. Left in place rather than rewritten, because "one crate" is the condition that motivated the task and the correction is the useful part. *(The first draft of this very correction said "eleven", by reading a count taken after the change into a sentence about before it — the same one-measurement-two-subjects error it is correcting, committed while correcting it.)*

Survivable while no crate had a meaningful default feature. **This crate's mmap is one.**

**Scope note, stated because the spec does not settle it:** C6 says *"the full parser suite"*. This task extends the flag to **every gated crate**, which includes `mlmf-conformance` and `mlmf-source-file` — neither of which is a parser. That is *more* than C6 asks for, which is the safe direction, but **"one crate is not the suite" does not by itself license "therefore all six"**. It is a ruling; record it as one.

**Files:** `.github/workflows/ci.yml`, `crates/mlmf-core/tests/ci_coverage.rs`.

- [ ] **Step 1: Write the failing test** — every gated crate has a `--no-default-features` test step, crate list derived from the filesystem as that file already does.
- [ ] **Step 2: Run; confirm it fails naming FOUR crates** — `mlmf-conformance`,
      `mlmf-ggml`, `mlmf-gguf`, `mlmf-safetensors`. **The rule, not the number: every gated crate that does
      not already have such a step.** Measured today that is the four above —
      `mlmf-core` has one, and **Task 1 Step 4 adds one for
      `mlmf-source-file`**. *(Stated as a rule because the count is right today
      and wrong after any CI edit; let Step 2 report the names.)* *(An earlier revision said "five, six once Task 1 lands". Neither
      is reachable — the number was introduced by a fix and never derived, and
      it sits inside the step that verifies the step. An implementer seeing
      four would go looking for a bug in their own test.)*
- [ ] **Step 3: Add the four steps.** Not six — do not add a second one for
      `mlmf-source-file`; Task 1 already did.

      ⚠️ **Adding them DISARMS the gate beside it, and no step says so.**
      *(Found by Task 5's implementer, by asking what the new steps break
      rather than what they add.)*
      `every_gated_crate_is_tested_documented_and_linted_by_ci` requires
      `cargo test -p <name>` via `workflow.contains()` — a **substring**,
      searched over the whole file. `cargo test -p mlmf-ggml` is a **prefix**
      of `cargo test -p mlmf-ggml --no-default-features`. **Measured**: with
      the four steps added and the plain `cargo test -p mlmf-ggml` step
      deleted outright, all four `ci_coverage` tests passed. The single
      deletion that gate exists to catch went invisible **for all six
      crates at once**, as a side effect of enforcing C6 properly, and Task
      5 as written ships that regression silently.

      The same file held an older hole of the same shape: every step writes
      its command **twice**, once after `- name:` and once after `run:`, so
      a whole-file substring search is satisfied by the *name* alone — a
      step whose `run:` was rewritten to anything at all still passed.
      `scripts/local-gates.sh` executes the `run:` lines and never reads a
      name, so that divergence had no reader anywhere in the repository.

      **Both fixed here rather than deferred to Task 6**, because Step 3 is
      what creates the first: the two gates now match against a
      `workflow_run_commands()` list — test and doc by **whole-string
      equality**, clippy by prefix since its run line carries
      `-- -D warnings`. Positive controls, measured: the same ggml deletion
      now reddens naming `cargo test -p mlmf-ggml`; gutting a doc step's
      `run:` while leaving its `- name:` intact now reddens **both** gates,
      where before it was caught only by adjacency, through
      `rustdoc_warnings_…`'s step count.

      ⚠️ **And a stale claim in `ci.yml`, fixed in passing.** Task 1's
      comment on the `mlmf-source-file --no-default-features` step read
      *"This crate declares no features yet, so today this step runs the
      same code as the one above it."* True when written; **false since
      Task 3**, which added `default = ["mmap"]` and made it the one step of
      six for which the flag subtracts anything — the comment now asserted
      the opposite of the fact. Task 3 owned `Cargo.toml`, `src/` and
      `tests/`, not `ci.yml`, so **no task owned the sentence its change
      falsified** — the shape Task 6 is pre-assigned to look for, arriving
      one task early.
- [ ] **Step 4: Run the full gate set.**
- [ ] **Step 5: Sabotage.** Delete one step; confirm the gate names that crate.

      **Lands exactly as written.** Deleting `cargo test -p mlmf-safetensors
      --no-default-features` reddens
      `every_gated_crate_is_run_with_default_features_off` **alone**, naming
      that crate and no other — a test failure, `3 passed; 1 failed`, not a
      compile error. Chosen from the middle of the list rather than the
      front, so that a gate naming a hardcoded crate would not have passed
      for it.

      **The three vacuity routes were PROBED, not reasoned about.** Hiding
      every `crates/*/Cargo.toml` panics inside `gated_members()` —
      *"expected at least two gated crates, found []"*; renaming the workflow
      away panics inside `workflow_without_comments()` — *"is readable: The
      system cannot find the file specified"*; truncating the workflow to
      zero bytes reddens naming **all six** crates. The gate enumerates the
      crates that must be covered rather than the steps that happen to be
      present, so unlike `rustdoc_warnings_are_fatal_for_every_documented_crate`
      it needs no separate count assertion to be non-vacuous.

      ⚠️ **What Step 5 cannot reach, and it is half of what C6 asks.**
      *(Found by Task 5's implementer.)* A step that EXISTS is not a step
      that RUNS anything. `tests/mmap.rs` opens `#![cfg(feature = "mmap")]`,
      so under the flag it compiles to zero tests — and a crate whose whole
      suite were gated that way would satisfy this gate with a step
      reporting *"running 0 tests … ok"*. C6's *"builds **and runs**"* is
      only half-enforceable from workflow text, and this gate is the half
      that is.

      **Measured, both configurations, per crate:** `mlmf-source-file` runs
      26 → 22 (the four absent are `mmap.rs`); `mlmf-core` 96 → 96,
      `mlmf-ggml` 28 → 28, `mlmf-gguf` 102 → 102, `mlmf-safetensors`
      49 → 49, `mlmf-conformance` 8 → 8. **Every crate but one is identical
      both ways**, because `src/file.rs` holds the only
      `#[cfg(feature = …)]` anywhere under `crates/*/src`. So five of the
      six steps are today an exact re-run of the plain step beside them.
      That is the ruling's price rather than a defect — but "C6 is now
      enforced" means the **policy** is enforced, not that five mmap-free
      paths were proven, and the plan's framing invites the stronger reading.
- [ ] **Step 6: Commit.**

---

## Task 6: Whole-branch review

- [ ] Dispatch a **fresh** reviewer over the whole diff. Priorities: cross-file contradictions first; tests that cannot fail (the four vacuity modes, plus *an assertion pinning a snapshot as a specification*); claims of impossibility; charter violations; then whether the API reads as one thing.
- [ ] **Tell the reviewer to read files the diff does not touch but whose claims it depends on.** Plan 5's sharpest finding was in a file with no diff hunk and no task owner.
- [ ] **Pre-assign one:** `crates/mlmf-core/tests/transitive_deps.rs`'s doc says its narrow scope *"becomes wrong at that moment"* if a member gains an external dependency. **Task 3 is that moment**, in a file no task owns.

      **Task 3 landed it and the numbers did not move**: `scripts/check-deps.sh`
      still reports *"C1/C2 OK: 8 transitive nodes, snapshot matches"*,
      `--test transitive_deps` is 3 passed, and `Cargo.lock` moved by one line
      because `memmap2 0.9.11` was already locked for the legacy root package.
      **What the reviewer should look at is the doc's second reason, not the
      snapshot.** It reads *"a format crate's dependencies are opt-in with the
      format"* — `memmap2` is a **default** feature of a **source** crate, so
      it is opt-**out**, and the sentence is now inexact about the case that
      actually arrived. The scope may still be right; the argument for it was
      written before this shape existed.
- [x] ~~Settle `read_range`'s error kind~~ — **SETTLED before Task 3, not
      deferred to review.** The doc's parenthetical bound ambiguously; the
      workspace had two answers.

      **`ErrorKind::Truncated { needed, available }` wins, and my plan's
      ruling of `Source` was wrong against more than the fake.** Measured:
      `mlmf-core`'s own reference `Fake::read_range` returns `Truncated`;
      **`mlmf-gguf` chose it citing that reference by name**; and
      `mlmf-safetensors` followed `mlmf-gguf`. Three implementations, a
      written rationale, and `FileSource` was the only one answering "this
      range is not in my bytes" a different way.

      **On the merits and not only on precedent:** a caller branches on two
      integers instead of parsing a sentence, and a parsed error message is a
      contract nobody wrote down and everybody depends on. The two `past_eof`
      tests now assert the numbers rather than a message substring.

      **The doc is fixed so it cannot bind two ways**, which is the half that
      stops it recurring — not just the implementation.

      **And the inverted-range and buffer-width cases keep `Source`, for a
      reason worth recording:** `mlmf-core` has typed `InvertedRange` and
      `SizeMismatch` variants and **neither fits, because both carry a tensor
      `name` and a source has no tensors.** They are format-axis errors;
      `RangedSource` is a source-axis trait. `Truncated` is the one variant
      with no tensor assumption, which is exactly why it is the one that fits.

- [ ] **Verify every Important finding yourself before fixing it.**
- [ ] **Record the review-findings rate**, and say which rate it is.

      ⚠️ **Do not compare it to "plan 3's eleven".** That number is a
      DIFFERENT MEASUREMENT: `2026-08-21-mlmf-gguf-tensors.md:81` defines it as
      *"eleven **controls could not reach the assertion they named**"* — a
      vacuity rate, ten of the eleven self-inflicted. Plan 5 then wrote
      *"compare to plan 3's eleven and plan 4's two"* under a whole-branch
      review heading, splicing a vacuity count onto a findings series, and an
      earlier revision of THIS plan inherited the splice and hardened it into
      "Plan 3 found eleven".

      **Both rates are worth recording. Neither is the other.** If you want a
      vacuity rate, count controls that could not reach their assertion, and
      keep the two series apart.

      **Do not carry a prior findings count into this plan at all.** No review
      artefacts are stored; the commit subjects say "review findings" with no
      severity band; and the nearest six — `79f4b1b`, *"six review findings from
      the final whole-branch pass"* — is dated 2026-08-15 over
      `mlmf-core`+`mlmf-ggml`, which is **plan 2, not plan 5**. **Record this
      plan's own number and cite the commit that carries it.**

      *(A previous revision also claimed "no review artefacts are stored in
      `docs/`". False — `grep -rn eleven docs/` finds **four** hits outside this plan: the
      definition (`2026-08-21…:81`), a restatement (`:1508`), one carrier
      (`2026-08-26…:572`), and one false positive (`2026-08-19…:1653`, "the
      eleven scalars"). Expect to discard one. A false absence-claim in a plan is worse than the silence it
      replaced.)*

---

## Deliberately not in this plan

- **`mlmf-source-hub`** — §3.1 makes it the only crate permitted a TLS edge. Different crate, different plan. The network crates stay forbidden on **both** axes here.
- **`mlmf-hf-layout`** — format axis, §12 step 5.
- **Retiring `src/mmap_loader.rs`** — §11 assigns those 475 lines here (verified: `wc -l` → 475), but deleting the legacy implementation is **CireSnave's**. **The legacy `src/` tree is 19,329 lines across 37 files; §11's table itemises 27 modules totalling 18,333.** *(The first draft called 19,329 "§11's" — it is the tree's total, and the table is a subset. Two real numbers, one wrong attribution.)*
- **Async or streaming sources** — §3.4 keeps the API shape open for them. Keeping it open is in scope; building one is not.
