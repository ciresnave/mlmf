# `mlmf-source-file` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The first crate on the **source axis**, so that something in this workspace can open a file.

**Architecture:** Spec §3.1 draws two orthogonal axes — format crates are `bytes → structure` and do no I/O; source crates are I/O only and know nothing about formats. Every crate built so far is on the format axis, which is why **nothing in the workspace can currently open a file** and why neither named consumer's dependency is complete. This crate implements `mlmf-core`'s `ByteSource` and `RangedSource` over the filesystem, with `memmap2` behind a **default** feature and a plain-read path that must work without it.

**Tech Stack:** Rust 2024, `mlmf-core`, `memmap2` (default feature only).

**Spec:** `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md` — §3.1 (axes), §3.2 (layout/acquisition split), §3.4 (mmap), C3, C6, C7, §11 (`mmap_loader.rs` 475 LOC → this crate).

---

## Global Constraints

- **C3 is scoped to the FORMAT axis.** Verbatim: *"No crate **on the format axis** references `std::fs`, `memmap2`, or any network client."* This crate is on the source axis and §3.4 says *"`memmap2` is a **default** feature of `mlmf-source-file`."*
- **C6.** *"CI builds **and runs** the full parser suite with `--no-default-features`, proving the mmap-free path is functional rather than merely compilable."*
- **C7.** One version across the workspace: `version.workspace = true`.
- **Sources depend on `mlmf-core` only and never on a format crate** (§3.1). Not a style preference — it is what makes `any source × any format` compose.
- **No format knowledge of any kind.** This crate must not know what a `.gguf` or a `.safetensors` is. §3.2 puts checkpoint structure in `mlmf-hf-layout`, which is on the *format* axis and never enumerates a directory. **This crate enumerates and does not interpret.**
- Every gate green by exit status before every commit: `bash scripts/local-gates.sh`. **Never pipe a gate.** `cargo fmt --all` before every commit.
- Restore a sabotage with `cp`, never `git checkout --`, whenever the file carries uncommitted work.

---

## ⚠️ Task 0 blocks everything else. Read it before writing any code.

**Two gates will reject this crate on its first commit, and they are wrong in the same way.**

`crates/mlmf-core/tests/purity.rs::every_gated_crate_performs_no_io` (source-level) and
`crates/mlmf-core/tests/deps.rs::the_manifest_names_no_io_crate_anywhere` (manifest-level) both iterate `common::gated_members()` — **every directory under `crates/` holding a `Cargo.toml`, with no axis distinction** — and both forbid `memmap2` and `std::fs`.

**The spec clause they implement is scoped and the gate is not.** `purity.rs`'s own header opens *"C3: every gated workspace member performs no I/O"*, while C3 says *"no crate **on the format axis**"*. **The qualifier was dropped when the clause became a gate.**

This has been harmless for five crates because all five are on the format axis. **`mlmf-source-file` is the first crate that makes it bite**, and it would bite as a red gate on a correct implementation — the shape where a check scores an implementation on how naively it spells the mechanism.

---

## File Structure

- `crates/mlmf-core/tests/purity.rs` — teach the C3 scanner the axis.
- `crates/mlmf-core/tests/deps.rs` — same, for the manifest scan.
- `crates/*/tests/axis` — one line per gated crate, `format` or `source`. Six files.
- `crates/mlmf-source-file/Cargo.toml`, `src/lib.rs`, `src/file.rs`, `src/dir.rs`, `tests/*`.
- `.github/workflows/ci.yml` — the new crate's three steps, plus C6.

---

## Task 0: Give the C3 gates the axis the spec already wrote

**Files:** `crates/mlmf-core/tests/purity.rs`, `crates/mlmf-core/tests/deps.rs`, `crates/mlmf-core/tests/common/mod.rs`, and `crates/<each>/tests/axis` (5 files: core, ggml, gguf, safetensors, conformance — all `format`).

**Interfaces produced:** `common::axis(dir) -> Axis`, where `Axis` is `Format` or `Source`.

- [ ] **Step 1: Write the failing test.** In `purity.rs`, a test asserting that a crate declaring `source` may name `memmap2` and `std::fs`, and one asserting a crate declaring `format` may not. Feed both through the existing scanner fixtures rather than creating real crates.

- [ ] **Step 2: Run it, confirm it fails** because `axis` does not exist.

- [ ] **Step 3: Implement.** `common::axis()` reads `<crate>/tests/axis`. **A missing file is a loud panic, not a default** — matching `allow_list()` and `allowed_std()`, whose docs say why: a silently-defaulted policy is the thing being avoided. Trim; accept exactly `format` or `source`; anything else panics naming the file.

  **The source axis still forbids the network crates.** `reqwest`, `ureq`, `tokio`, `hyper`, `rustls`, `native_tls`, `openssl`, `curl`, `socket2`, `hf_hub`, `async_std` stay forbidden on **both** axes — §3.1 says `mlmf-source-hub` is *"the only crate in the tree with a TLS edge"*, and it is not this crate. **Only the filesystem/mmap set is relaxed:** `memmap2`, `std::fs`, `std::io`, `std::path`.

- [ ] **Step 4: Add the five `axis` files**, each containing `format`, and run the full gate set.

- [ ] **Step 5: Sabotage, both directions.**
  1. Change one crate's `axis` to `source` and confirm the C3 scanner **stops** rejecting `memmap2` for it — proving the relaxation is real and not a no-op.
  2. Delete an `axis` file and confirm the loud panic names it.
  3. Put `reqwest` in a `source`-declared fixture and confirm it is **still rejected** — the control that proves the relaxation is scoped rather than a hole.
  4. Write `Source` (wrong case) into an `axis` file and confirm it panics rather than silently reading as `format`.

- [ ] **Step 6: Commit.**

---

## Task 1: The crate, and the path that works without mmap

**Do this before mmap, deliberately.** C6 exists to prove the mmap-free path is *functional rather than merely compilable*, and a path written second is a path written to match the first.

**Files:** `crates/mlmf-source-file/{Cargo.toml, src/lib.rs, src/file.rs, tests/axis, tests/direct-deps.allow, tests/allowed-std.list}`.

**Interfaces produced:**
```rust
pub struct FileSource { /* bytes: Vec<u8> or Mmap */ }
impl FileSource {
    pub fn open(path: &Path) -> Result<Self>;
}
impl mlmf_core::ByteSource for FileSource {
    fn as_bytes(&self) -> &[u8];
}
```

- [ ] **Step 1: Write the failing test.** Open a temp file with known bytes; assert `as_bytes()` returns them exactly, including a zero-length file (`as_bytes()` is empty, not an error) and a file with an embedded NUL.

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement** with `std::fs::read`, no mmap, no feature gate yet. `Cargo.toml` carries `version.workspace = true` (C7) and **no `memmap2` yet**. `tests/axis` contains `source`. Both allow-list files are required by the gates — `deps.rs::allow_list` and `purity.rs::allowed_std` panic without them.

- [ ] **Step 4: Run, confirm pass, and add the crate's three CI steps** to `.github/workflows/ci.yml`. `ci_coverage.rs` will demand them; **add them deliberately and let the gate confirm rather than discover.** Also add the crate to root `default-members`, which `workspace.rs` now gates.

- [ ] **Step 5: Sabotage.** Truncate the read by one byte and confirm the exact-bytes test reddens; return `Vec::new()` unconditionally and confirm the non-empty test reddens while the zero-length test stays green — **the control that proves the empty case is not carrying the whole assertion.**

- [ ] **Step 6: Commit.**

---

## Task 2: `RangedSource`, so a consumer need not materialize the file

**Files:** `crates/mlmf-source-file/src/file.rs`, `tests/ranged.rs`.

**Interfaces consumed:** `mlmf_core::RangedSource`:
```rust
fn len(&self) -> Option<u64>;
fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> Result<()>;
```

- [ ] **Step 1: Write the failing tests.** A middle range; a range ending exactly at EOF; a range one byte past EOF (**must be an error, not a short read**); an inverted range; `into` shorter than the range; `into` longer than the range; `len()` for a real file and for a zero-length file.

  **The past-EOF and exactly-at-EOF pair is not optional.** Both corpus differentials in this repo caught a `>` vs `>=` off-by-one on real models, in two formats. The last tensor of a well-formed file touches the last byte every time, so that boundary is exercised by ordinary inputs and is where this will be wrong if it is wrong.

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement** with `seek` + `read_exact`. **Do not silently truncate a range to the file's length** — that turns a caller's arithmetic error into a short read they cannot see.

- [ ] **Step 4: Run.**

- [ ] **Step 5: Sabotage.** `>` → `>=` on the EOF bound and confirm exactly the exactly-at-EOF test reddens and the past-EOF one does not — **identifying which assertion fires, not merely that one does.** Then clamp the range instead of erroring and confirm the past-EOF test reddens.

- [ ] **Step 6: Commit.**

---

## Task 3: mmap, behind the default feature

**Files:** `crates/mlmf-source-file/Cargo.toml`, `src/file.rs`, `tests/mmap.rs`.

- [ ] **Step 1: Write the failing test.** Under `--features mmap`, `as_bytes()` returns identical bytes to the plain-read path for the same file. **Assert equality against the plain path, not against a literal** — the two implementations checking each other is the point, and a literal would let both drift together.

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement.** `memmap2` as an **optional** dependency with `default = ["mmap"]`, `mmap = ["dep:memmap2"]`. `unsafe` is required by `Mmap::map`; this crate therefore **cannot** carry `#![forbid(unsafe_code)]` where the other crates do. **Say that in the crate doc with the reason**, since every sibling crate forbids it and a reader will assume an omission.

- [ ] **Step 4: Run both feature configurations.**

- [ ] **Step 5: Sabotage.** Map with an off-by-one length and confirm the cross-implementation equality test reddens. Then build with `--no-default-features` and confirm the crate still compiles and the Task 1 and 2 tests still pass — **that is C6's actual claim and it must be observed, not assumed.**

- [ ] **Step 6: Commit.**

---

## Task 4: Directory enumeration, with no format knowledge

§3.2: *"`mlmf-source-file` walks a local directory"*, while `mlmf-hf-layout` *"never enumerates a directory"*. The split is the point.

**Files:** `crates/mlmf-source-file/src/dir.rs`, `tests/dir.rs`.

- [ ] **Step 1: Write the failing test.** Enumerating a directory returns every file name, in sorted order, with no filtering by extension and no interpretation. Include a `.safetensors`, a `.gguf`, a `README.md` and a subdirectory, and assert **all four appear** — the test that a later "helpful" filter would break.

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement.** Names only. **No sniffing, no extension mapping, no guessing which file is a model.** That is interpretation and the charter forbids it: *"MLMF is never intended to be an interpreter of the content of model files."*

- [ ] **Step 4: Run.**

- [ ] **Step 5: Sabotage.** Filter to `.safetensors` only and confirm the test names the three files it dropped.

- [ ] **Step 6: Commit.**

---

## Task 5: C6 is under-enforced, and this crate is why it matters

**Measured 2026-09-02:** C6 says *"CI builds and runs the **full parser suite** with `--no-default-features`"*. CI has exactly one such step, `cargo test -p mlmf-core --no-default-features`. **One crate is not the suite.**

That was survivable while no crate had a meaningful default feature. **This crate's mmap is a default feature, so C6 is now load-bearing.**

**Files:** `.github/workflows/ci.yml`, `crates/mlmf-core/tests/ci_coverage.rs`.

- [ ] **Step 1: Write the failing test** in `ci_coverage.rs`: every gated crate has a `--no-default-features` test step. Derive the crate list from the filesystem as that file already does.

- [ ] **Step 2: Run, confirm it fails** naming the crates that lack one.

- [ ] **Step 3: Add the steps.**

- [ ] **Step 4: Run the full gate set.**

- [ ] **Step 5: Sabotage.** Delete one `--no-default-features` step and confirm the gate names that crate.

- [ ] **Step 6: Commit.**

---

## Task 6: Whole-branch review

**Not optional.** Plan 3's review found eleven Important findings, plan 4's two, plan 5's six — **and five of plan 5's six were cross-file contradictions created by the plan's own later tasks invalidating its own earlier prose.**

- [ ] Dispatch a **fresh** reviewer over the whole diff. Priorities: cross-file contradictions first; tests that cannot fail (the four vacuity modes, plus *an assertion pinning a snapshot as a specification*); claims of impossibility; charter violations; then whether the API reads as one thing.
- [ ] **Tell the reviewer to read files the diff does not touch but whose claims it depends on.** Plan 5's sharpest finding was in a file with no diff hunk and no task owner: a per-task review cannot see a file no task owns.
- [ ] **Verify every Important finding yourself before fixing it.**
- [ ] Record the rate and compare.

---

## Deliberately not in this plan

- **`mlmf-source-hub`** — §3.2 keeps acquisition separate from layout, and it is the only crate permitted a TLS edge. Different crate, different plan.
- **`mlmf-hf-layout`** — format axis, and it never enumerates a directory. §12 step 5.
- **Retiring `src/mmap_loader.rs`** — §11 assigns those 475 lines here, but deleting the legacy implementation is **CireSnave's**, alongside the rest of §11's 19,329 lines. This plan adds; it does not delete.
- **Async or streaming sources** — §3.4 keeps the API shape open for them deliberately. Keeping the shape open is in scope; building one is not.
