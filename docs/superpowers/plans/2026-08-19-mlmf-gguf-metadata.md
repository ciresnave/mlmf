# mlmf-gguf: the metadata path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read every declared key out of a GGUF file without decoding a single tensor, without materializing a 777,000-string vocabulary, and without an unrecognized quantization code being able to fail the open.

**Architecture:** A staged parse — magic+version, then the key-value block, then (in the next plan) the tensor directory — where only the stage that fails, fails. The KV block is *indexed* at open into `(key, type, byte range)` and a value is decoded on demand, so opening a file is O(number of keys) rather than O(vocabulary size). Errors are split so "this file is not what its name claims" and "this file is a GGUF and something inside it is wrong" are different variants.

**Tech Stack:** Rust 2024, `mlmf-core` only. No `unsafe`. No new third-party dependencies.

**Spec:** `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md`
**Consumer requirements:** `docs/superpowers/specs/2026-08-19-lightbulb-gguf-seam-requirements.md` — R1–R7, confirmed and corrected by Lightbulb's architect on 2026-08-19. **Read it; this plan is its implementation and every task cites the requirement it serves.**

## Global Constraints

Copied verbatim from the spec. Every task's requirements implicitly include these.

- **C1 — one dependency edge per capability.** `mlmf-gguf` depends on `mlmf-core` and `mlmf-ggml` and nothing else. No `byteorder`, no `memmap2`, no `serde`.
- **C3 — purity.** No filesystem, network, process spawning or clock in `src/`. Enforced by a gate, not by intention.
- **`#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`** at the crate root.
- **C7 — one version number across the workspace.** `version.workspace = true`; never a hardcoded version.
- **Absent means not declared (spec §5).** Never a default.
- **Loud unknowns (spec §7).** Anything the parse does not understand is reported, never silently dropped.
- **Byte-exact strings.** No trimming, no case folding, no Unicode normalization, anywhere, ever.
- **AD-2 — proven able to fail.** No test is trusted until sabotage has made it go red.
- **rustfmt is a gate.** `cargo fmt --all -- --check` must pass; a pre-commit hook enforces it locally.
- Clippy is a gate: `cargo clippy --all-targets -- -D warnings`.

---

## Why this plan stops before the tensors

`mlmf-gguf` will eventually implement `TensorContainer`. This plan deliberately does not.

R1 requires that reading metadata cannot fail on tensor content. The cleanest way to guarantee that is not discipline but **shape**: if the metadata stage has no access to the type table, it cannot fail against it. That makes header+KV a complete, useful, independently shippable artifact — and it is the only half Lightbulb needs. They can adopt it while the tensor path is being built.

The tensor directory, offset rebasing, `TensorContainer`, and `UnrecognizedKind::TensorEncoding` reporting are **Plan 4**.

## Measured facts this plan is built on

Do not re-derive these; they were measured on the 29-file corpus at `C:\Models\gguf-corpus`.

| Fact | Value | Consequence |
|---|---|---|
| Worst-case string count in one file | **777,056** (`ggml-vocab-gemma-4.gguf`) | Eager decode = ~777k `String` allocations, ~26 MB heap, to read one key |
| Worst-case KV block | **15.78 MB** | Indexing is cheap; decoding is not |
| Largest array key | `tokenizer.ggml.merges` (514,906) | Often **larger** than `tokenizer.ggml.tokens` — do not optimize only for `tokens` |
| Max keys in any corpus file | **42** | Indexing all keys eagerly is free |
| Non-UTF-8 strings across 29 files | **0** | R3 is untestable against real files; authored fixtures are mandatory |
| Trailing-NUL strings across 29 files | **0** | Same |
| Files declaring `general.alignment` | **0** | The default-32 path is the only one real files exercise |

## Format facts, from llama.cpp at the pinned commit

Source: `ggml/include/gguf.h` and `ggml/src/gguf.cpp` at `9d57ce456c94d241dde672b2db9cf18879766568`.

```
magic       4 bytes, "GGUF"
version     u32
n_tensors   i64
n_kv        i64
each KV:    key (string) · value type (u32) · value
              if ARRAY: element type (u32) · count (u64) · elements
strings     length (u64) then bytes, NO null terminator
enums       stored as int32
bools       stored as int8
```

`GGUF_VERSION` is 3. `GGUF_DEFAULT_ALIGNMENT` is 32. `general.alignment`, if present, must be `UINT32` and a power of two.

The 13 value types:

| id | type | id | type | id | type |
|---|---|---|---|---|---|
| 0 | UINT8 | 5 | INT32 | 10 | UINT64 |
| 1 | INT8 | 6 | FLOAT32 | 11 | INT64 |
| 2 | UINT16 | 7 | BOOL | 12 | FLOAT64 |
| 3 | INT16 | 8 | STRING | | |
| 4 | UINT32 | 9 | ARRAY | | |

**Endianness detection, from `gguf.cpp:497`:** if `version & 0x0000FFFF == 0`, the file was written with the opposite endianness — version 3 read byte-swapped is `0x03000000`. This is a cheap, real check and it belongs in the header stage.

## Design decisions

**D1 — The KV block is indexed at open, values decoded on demand.** Justified by measurement, not preference: 777k allocations to read one key. `MetadataSource::get` returns `&MetaValue`, so a decoded value must outlive the call — each entry carries a `OnceLock<MetaValue>` filled on first access. This puts `std::sync` on the crate's purity allow-list, which is not I/O and is fine.

**D2 — GGUF v1 is out of scope.** Current llama.cpp **refuses v1 outright** (`gguf.cpp:502`), so its reader is not a reference for the v1 layout, and the one v1 file in the corpus did not parse under a v2-shaped reader with only the integer widths substituted. Supporting it means deriving the layout from that file. That is a separate plan with its own evidence problem. **This plan rejects v1 with a specific error naming the version**, which is the honest behaviour.

**D3 — This is the third-consumer moment for the duplicated gates.** `mlmf-core` and `mlmf-ggml` each carry a ~830-line copy of `purity.rs` + `deps.rs`. Task 2 consolidates them **before** a third copy exists, using the route the `mlmf-ggml` final review identified: one gate iterating `crates/*/src`, reading each crate's allow-list from a file. This avoids the dev-dependency problem entirely — extracting a shared *crate* would require relaxing the C2 gate, which rejects `[dev-dependencies]` outright.

**D4 — The entry point takes `&[u8]`.** Spec §3.4: format crates parse bytes, source crates obtain them. No file paths anywhere in this crate.

**D5 — Errors split by stage.** R7: "magic or version unreadable" and "valid container, content unreadable" are different variants, because they carry different operator remedies — *you pointed me at the wrong file* versus *your file is malformed*.

**D6 — Nothing in this crate interprets a key.** No `chat_template` accessor, no architecture inference, no config struct. The charter puts interpretation outside these walls, and R4's seven join failure modes are the evidence: getting them right needs ecosystem knowledge MLMF does not have.

## File Structure

| File | Responsibility |
|---|---|
| `crates/mlmf-gguf/Cargo.toml` | manifest; two dependencies |
| `crates/mlmf-gguf/src/lib.rs` | crate docs, lints, re-exports |
| `crates/mlmf-gguf/src/error.rs` | `GgufError` — the stage-split taxonomy (D5) |
| `crates/mlmf-gguf/src/cursor.rs` | bounds-checked little-endian reads over `&[u8]` |
| `crates/mlmf-gguf/src/header.rs` | magic, version, counts, endianness sniff |
| `crates/mlmf-gguf/src/value.rs` | `ValueType`, value decoding, byte-range skipping |
| `crates/mlmf-gguf/src/metadata.rs` | `GgufMetadata` — the index, `MetadataSource` impl, array accessors |
| `crates/mlmf-gguf/tests/fixture.rs` | a GGUF *writer*, test-only, for authored adversarial files |
| `crates/mlmf-gguf/tests/authored.rs` | the adversarial cases real files cannot provide |
| `crates/mlmf-gguf/tests/corpus.rs` | differential test against fixtures extracted from real files |
| `crates/mlmf-gguf/tests/allowed-std.list` | this crate's C3 allow-list (Task 2's format) |
| `crates/mlmf-gguf/tests/direct-deps.allow` | `mlmf-core`, `mlmf-ggml` |

`cursor.rs` and `value.rs` are split from `metadata.rs` because they change for different reasons: the cursor changes if bounds-checking policy changes, `value.rs` if GGUF adds a type, `metadata.rs` if the index or seam changes.

---

## Task 1: `Declaration` and array accessors in mlmf-core

R2 requires three states in the return type. R5 requires indexed array access that does not materialize. Both are seam changes and belong in core, where `mlmf-safetensors` will also use them.

**Files:**
- Modify: `crates/mlmf-core/src/report.rs` (add `Declaration`)
- Modify: `crates/mlmf-core/src/traits.rs` (add two `MetadataSource` methods)
- Modify: `crates/mlmf-core/src/lib.rs` (re-export)

**Interfaces:**
- Produces:
  ```rust
  pub enum Declaration<'a> {
      Absent,
      Unreadable(&'a Unrecognized),
      Declared(&'a MetaValue),
  }
  // on MetadataSource, all three with default impls:
  fn declaration(&self, key: &str) -> Declaration<'_>;
  fn array_len(&self, key: &str) -> Option<u64>;
  fn array_get(&self, key: &str, index: u64) -> Option<MetaValue>;
  ```

- [ ] **Step 1: Write the failing tests**

Add to the `mod tests` block in `crates/mlmf-core/src/traits.rs`. The existing `Fake` struct there implements `MetadataSource` over a `HashMap`; these use it plus one new type.

```rust
    #[test]
    fn a_declaration_separates_absent_from_unreadable_from_declared() {
        // R2. A single Option collapses "this file has no chat template"
        // with "this file has one and we could not decode it", and those
        // send an operator in opposite directions.
        struct Partial {
            good: MetaValue,
            bad: crate::Unrecognized,
        }
        impl MetadataSource for Partial {
            fn get(&self, key: &str) -> Option<&MetaValue> {
                (key == "good").then_some(&self.good)
            }
            fn keys(&self) -> Vec<&str> {
                vec!["good", "bad"]
            }
            fn declaration(&self, key: &str) -> crate::Declaration<'_> {
                match key {
                    "good" => crate::Declaration::Declared(&self.good),
                    "bad" => crate::Declaration::Unreadable(&self.bad),
                    _ => crate::Declaration::Absent,
                }
            }
        }
        let p = Partial {
            good: MetaValue::String("ok".into()),
            bad: crate::Unrecognized {
                kind: crate::UnrecognizedKind::MetadataKey {
                    key: "bad".into(),
                    value: MetaValue::U32(7),
                },
                origin: "model.gguf".into(),
            },
        };

        assert!(matches!(
            p.declaration("good"),
            crate::Declaration::Declared(MetaValue::String(s)) if s == "ok"
        ));
        // The distinction that matters: `get` returns None for BOTH of
        // these, and `declaration` does not.
        assert!(p.get("bad").is_none());
        assert!(p.get("missing").is_none());
        match p.declaration("bad") {
            crate::Declaration::Unreadable(u) => assert_eq!(u.origin, "model.gguf"),
            other => panic!("expected Unreadable, got {other:?}"),
        }
        assert!(matches!(p.declaration("missing"), crate::Declaration::Absent));
    }

    #[test]
    fn the_default_declaration_never_invents_an_unreadable_state() {
        // An implementor that has not overridden `declaration` must report
        // Declared or Absent and never Unreadable — claiming a decode
        // failure that did not happen is worse than not reporting one.
        let f = fake();
        assert!(matches!(
            f.declaration("general.architecture"),
            Declaration::Declared(_)
        ));
        assert!(matches!(f.declaration("absent"), Declaration::Absent));
    }

    #[test]
    fn array_accessors_default_to_walking_the_materialized_value() {
        // The default impls are correct-but-slow, which is the honest
        // answer for a source that has already decoded everything. A format
        // crate that can index bytes overrides them; Task 7 does exactly
        // that, and this pins the semantics both must satisfy.
        struct WithArray(MetaValue);
        impl MetadataSource for WithArray {
            fn get(&self, key: &str) -> Option<&MetaValue> {
                (key == "toks").then_some(&self.0)
            }
            fn keys(&self) -> Vec<&str> {
                vec!["toks"]
            }
        }
        let w = WithArray(MetaValue::Array(vec![
            MetaValue::String("a".into()),
            MetaValue::String("b".into()),
            MetaValue::String("c".into()),
        ]));

        assert_eq!(w.array_len("toks"), Some(3));
        assert_eq!(w.array_get("toks", 1), Some(MetaValue::String("b".into())));
        // Out of range is None, and is distinguishable from "not an array"
        // and "absent" by consulting array_len first — which is exactly how
        // R4's three id-resolution failures get told apart.
        assert_eq!(w.array_get("toks", 3), None);
        assert_eq!(w.array_len("absent"), None);
        assert_eq!(w.array_get("absent", 0), None);
    }

    #[test]
    fn a_non_array_value_has_no_length_rather_than_length_one() {
        // A scalar is not a one-element array. Reporting Some(1) would let
        // a consumer index a string as though it were a vocabulary.
        let f = fake();
        assert_eq!(f.array_len("general.architecture"), None);
        assert_eq!(f.array_get("general.architecture", 0), None);
    }
```

- [ ] **Step 2: Run them and watch them fail**

```bash
cargo test -p mlmf-core declaration
```

Expected: compile error, `cannot find type Declaration in module crate`. Not `0 passed` — these live in an existing `mod tests` that is already compiled, so a missing type is a hard error.

- [ ] **Step 3: Add `Declaration` to `report.rs`**

After the `Unrecognized` struct:

```rust
/// What a source knows about one metadata key (spec §5, consumer R2).
///
/// [`MetadataSource::get`] returns `Option`, which collapses two facts an
/// operator must be able to tell apart: a file that **declares nothing**
/// under this key, and a file that declares something the parse **could not
/// decode**. Those carry opposite remedies — supply the value, versus repair
/// the file — and a consumer given only `None` cannot say which it is
/// looking at.
///
/// `get` stays as the ergonomic path. This is the honest one.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Declaration<'a> {
    /// The key is not declared. Never a default.
    Absent,
    /// The key is declared and the value could not be decoded. Carries the
    /// report entry, so the complaint can name the key and what was seen.
    Unreadable(&'a Unrecognized),
    /// The key is declared and decoded.
    Declared(&'a MetaValue),
}
```

- [ ] **Step 4: Add the three trait methods**

In `crates/mlmf-core/src/traits.rs`, inside `trait MetadataSource`, after `keys()`:

```rust
    /// What this source knows about `key` — three states, not two.
    ///
    /// The default reports [`Declaration::Declared`] or
    /// [`Declaration::Absent`] and **never** [`Declaration::Unreadable`],
    /// because a source that has not overridden this has no undecodable
    /// values to report: everything it holds is already a [`MetaValue`].
    /// Claiming a decode failure that did not happen would be worse than
    /// not reporting one.
    fn declaration(&self, key: &str) -> Declaration<'_> {
        match self.get(key) {
            Some(v) => Declaration::Declared(v),
            None => Declaration::Absent,
        }
    }

    /// Number of elements in the array declared under `key`.
    ///
    /// `None` means the key is absent **or** its value is not an array.
    /// A scalar deliberately has no length rather than length 1 — reporting
    /// 1 would let a caller index a string as though it were a vocabulary.
    ///
    /// Provided so a caller can distinguish "index out of range" from
    /// "there is no array here", which consumer R4 needs to tell apart
    /// three different ways a token-id lookup fails.
    fn array_len(&self, key: &str) -> Option<u64> {
        match self.get(key) {
            Some(MetaValue::Array(items)) => Some(items.len() as u64),
            _ => None,
        }
    }

    /// One element of the array declared under `key`, by index.
    ///
    /// Returns owned, because a format crate should be able to decode a
    /// single element out of bytes without materializing the array — see
    /// consumer R5. The default walks an already-decoded value, which is
    /// the honest answer for a source that has one; **a format crate
    /// reading a 500,000-element vocabulary must override this.**
    fn array_get(&self, key: &str, index: u64) -> Option<MetaValue> {
        let items = match self.get(key) {
            Some(MetaValue::Array(items)) => items,
            _ => return None,
        };
        usize::try_from(index).ok().and_then(|i| items.get(i)).cloned()
    }
```

Add `use crate::Declaration;` to the imports at the top of `traits.rs`, and export from `lib.rs`:

```rust
pub use report::{Declaration, Report, Unrecognized, UnrecognizedKind};
```

- [ ] **Step 5: Run and watch them pass**

```bash
cargo test -p mlmf-core
cargo clippy -p mlmf-core --all-targets -- -D warnings
cargo doc -p mlmf-core --no-deps
```

All clean. `cargo doc` matters: the new docs contain intra-doc links and CI runs rustdoc with `-D warnings`.

- [ ] **Step 6: Prove the tests can fail (AD-2)**

Three sabotages, each must go red:

1. Change the default `declaration` to return `Declaration::Absent` unconditionally. `the_default_declaration_never_invents_an_unreadable_state` must fail.
2. Change `array_len`'s `_ => None` arm to `_ => Some(1)`. `a_non_array_value_has_no_length_rather_than_length_one` must fail.
3. Change `array_get` to ignore `index` and return `items.first().cloned()`. `array_accessors_default_to_walking_the_materialized_value` must fail on the `index 1` assertion — **check it fails there and not on the out-of-range one**, since returning `first()` for index 3 also returns `Some`, and a test that only checked out-of-range would pass.

Restore after each.

- [ ] **Step 7: Commit**

```bash
cargo fmt --all
git add crates/mlmf-core/src/report.rs crates/mlmf-core/src/traits.rs crates/mlmf-core/src/lib.rs
git commit -m "feat(core): Declaration and indexed array access for the metadata seam"
```

---

## Task 2: Consolidate the purity and dependency gates

**This is the third-consumer moment (D3).** Do it before `mlmf-gguf` exists, so it never gets a copy.

`crates/mlmf-core/tests/purity.rs` and `crates/mlmf-ggml/tests/purity.rs` are ~560 lines each and function-by-function byte-identical except for `ALLOWED_STD`. `deps.rs` is the same story at ~270 lines. A prior review verified the identity; nothing forces them to stay that way.

**Files:**
- Create: `crates/mlmf-core/tests/allowed-std.list`, `crates/mlmf-ggml/tests/allowed-std.list`
- Modify: `crates/mlmf-core/tests/purity.rs` (becomes the workspace gate)
- Delete: `crates/mlmf-ggml/tests/purity.rs`, `crates/mlmf-ggml/tests/deps.rs`
- Modify: `crates/mlmf-core/tests/deps.rs` (becomes the workspace gate)

**Interfaces:**
- Produces: one gate per concern, iterating every `crates/*/` member. Each crate declares its own policy in `tests/allowed-std.list` and `tests/direct-deps.allow`.

- [ ] **Step 1: Read what you are consolidating**

Read `crates/mlmf-core/tests/purity.rs` in full before touching it. The three load-bearing mechanisms are `strip_comments_and_literals`, the `use`-tree expander (`parse_use_tree` / `collect_use_paths`), and the **allow-list** check. **Preserve all three exactly.** A prior version of this gate compared source text against the literal `"std::fs"` and was defeated by `use std::{fs, path::Path};`; a sabotage agent then compiled `fs::read`, `TcpStream::connect` and `Command::new("curl")` into the crate with the suite green.

Note the precedent for what you are building: `crates/mlmf-core/tests/workspace.rs` already walks every `crates/*/Cargo.toml`. Follow its shape for discovering members.

- [ ] **Step 2: Write the allow-list files**

`crates/mlmf-core/tests/allowed-std.list` — one submodule per line, `#` comments allowed. Copy the current `ALLOWED_STD` contents from `purity.rs` verbatim, one per line, with the existing rationale comments preserved as `#` lines.

`crates/mlmf-ggml/tests/allowed-std.list`:

```
# mlmf-ggml is a pure table: geometry in, geometry out. Deliberately narrow.
cmp
fmt
iter
option
primitive
result
slice
u32
u64
usize
```

- [ ] **Step 3: Make the gate iterate members**

In `crates/mlmf-core/tests/purity.rs`, replace the `const ALLOWED_STD: &[&str]` with a loader, and replace the single-crate scan with a loop. Keep every other function byte-identical.

```rust
/// Every workspace member that must satisfy C3, with its own allow-list.
///
/// One gate rather than a copy per crate. The copies were provably
/// identical when `mlmf-ggml` was written, and nothing forced them to stay
/// that way — a fix to the use-tree expander applied to one and not the
/// other leaves a gate silently under-enforcing, which is the exact
/// failure this gate exists to prevent and which happened here once
/// already.
fn gated_members() -> Vec<PathBuf> {
    let root = workspace_root();
    let mut out = Vec::new();
    for entry in fs::read_dir(root.join("crates")).expect("crates/ is readable") {
        let dir = entry.expect("readable entry").path();
        if dir.join("Cargo.toml").is_file() {
            out.push(dir);
        }
    }
    out.sort();
    assert!(out.len() >= 2, "expected at least two gated crates, found {out:?}");
    out
}

/// A crate's permitted `std` submodules, from its own tests/allowed-std.list.
///
/// Deliberately an allow-list. `std::fs`, `std::net`, `std::process`,
/// `std::env`, `std::os`, `std::io` and `std::thread` are all ways to reach
/// outside the process, and a deny-list would require enumerating the next
/// one before it exists.
fn allowed_std(crate_dir: &Path) -> Vec<String> {
    let path = crate_dir.join("tests/allowed-std.list");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} must declare its C3 allow-list: {e}", path.display()));
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}
```

Then the test itself:

```rust
#[test]
fn every_gated_crate_performs_no_io() {
    for dir in gated_members() {
        let allowed = allowed_std(&dir);
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        for (path, text) in collect_rs(&dir.join("src")) {
            let label = format!("{name}: {}", path.display());
            if let Some(problem) = scan_text(&text, &label, &allowed) {
                panic!("{problem}");
            }
        }
    }
}
```

Thread `allowed: &[String]` through `scan_text` and `check_pair` in place of the former `ALLOWED_STD` constant. Change nothing else in those functions.

- [ ] **Step 4: Do the same for `deps.rs`**

`crates/mlmf-core/tests/deps.rs` already reads `tests/direct-deps.allow`. Make it iterate `gated_members()` and read each crate's own allow file. Keep the manifest parser — including its handling of `[dependencies.foo]` sub-tables, which an earlier version missed and which let `memmap2` through.

`crates/mlmf-ggml/tests/direct-deps.allow` already exists and stays.

- [ ] **Step 5: Delete the copies**

```bash
git rm crates/mlmf-ggml/tests/purity.rs crates/mlmf-ggml/tests/deps.rs
```

Move `crates/mlmf-ggml/tests/fixtures/grouped_import.rs.fixture` to `crates/mlmf-core/tests/fixtures/` if it is not already there — the born-red fixture proving the use-tree expander works must live with the gate.

- [ ] **Step 6: Prove the consolidated gate still catches everything, per crate**

The whole risk of this task is a gate that scans fewer crates than before while still passing. Run **four** sabotages:

1. Add `use std::{fs, path::Path};` plus `pub fn r(p: &Path) -> Vec<u8> { fs::read(p).unwrap() }` to `crates/mlmf-core/src/lib.rs`. Must go red naming `mlmf-core` and `std::fs`. Restore.
2. The same in `crates/mlmf-ggml/src/lib.rs`. **Must go red naming `mlmf-ggml`.** If it does not, the loop is not reaching that crate and the consolidation has silently reduced coverage. Restore.
3. Add `bytemuck = "1"` to `crates/mlmf-ggml/Cargo.toml`. The deps gate must go red naming `mlmf-ggml` and `bytemuck`. Restore.
4. Delete `crates/mlmf-ggml/tests/allowed-std.list`. The gate must **panic with a clear message**, not silently skip that crate. This is the failure mode that matters most: a missing policy file must be loud.

- [ ] **Step 7: Full verification and commit**

```bash
cargo test -p mlmf-core -p mlmf-ggml
cargo clippy --all-targets -p mlmf-core -p mlmf-ggml -- -D warnings
cargo fmt --all
git add -A crates/ && git commit -m "test: one purity and dependency gate for the whole workspace"
```

**Do not use `git add -A` from the repository root** — it pulls in `.claude/`, `.serena/` and `supertool`, two of which are embedded git repositories that commit as unusable gitlinks. They are gitignored now, but scope the add anyway.

---

## Task 3: Crate skeleton, gates, and the cursor

**Files:**
- Create: `crates/mlmf-gguf/Cargo.toml`, `src/lib.rs`, `src/cursor.rs`
- Create: `crates/mlmf-gguf/tests/allowed-std.list`, `tests/direct-deps.allow`
- Modify: root `Cargo.toml` (`default-members`)

**Interfaces:**
- Produces:
  ```rust
  pub(crate) struct Cursor<'a> { /* private */ }
  impl<'a> Cursor<'a> {
      pub(crate) fn new(bytes: &'a [u8]) -> Self;
      pub(crate) fn pos(&self) -> u64;
      pub(crate) fn seek(&mut self, pos: u64) -> Result<(), Truncated>;
      pub(crate) fn u8(&mut self)  -> Result<u8,  Truncated>;
      pub(crate) fn u16(&mut self) -> Result<u16, Truncated>;
      pub(crate) fn u32(&mut self) -> Result<u32, Truncated>;
      pub(crate) fn u64(&mut self) -> Result<u64, Truncated>;
      pub(crate) fn i64(&mut self) -> Result<i64, Truncated>;
      pub(crate) fn f32(&mut self) -> Result<f32, Truncated>;
      pub(crate) fn f64(&mut self) -> Result<f64, Truncated>;
      pub(crate) fn take(&mut self, n: u64) -> Result<&'a [u8], Truncated>;
  }
  pub(crate) struct Truncated { pub needed: u64, pub available: u64 }
  ```

- [ ] **Step 1: Write the manifest and lib**

`crates/mlmf-gguf/Cargo.toml` — inherit everything inheritable; C7 forbids a hardcoded version:

```toml
[package]
    description          = "Reads GGUF metadata. No I/O, no interpretation."
    edition.workspace    = true
    license.workspace    = true
    name                 = "mlmf-gguf"
    repository.workspace = true
    version.workspace    = true

[dependencies]
    mlmf-core = { path = "../mlmf-core", version = "0.4.0" }
    mlmf-ggml = { path = "../mlmf-ggml", version = "0.4.0" }
```

Copy `LICENSE-MIT` and `LICENSE-APACHE` from the repository root into `crates/mlmf-gguf/`. Cargo packages from the crate directory and does not ascend, so a member without its own copies publishes with no licence text. **Verify with `cargo package --list -p mlmf-gguf` after committing** — the check reports nothing for untracked files, so a copy made and not committed looks identical to no copy at all.

`crates/mlmf-gguf/src/lib.rs`:

```rust
//! Reads GGUF metadata.
//!
//! A GGUF file is parsed in stages — magic and version, then the key-value
//! block, then the tensor directory — and **only the stage that fails,
//! fails**. Reading `tokenizer.chat_template` out of a file full of
//! quantizations this build has never heard of is not merely supported, it
//! is structurally guaranteed: the metadata stage has no access to a type
//! table, so it cannot fail against one.
//!
//! The key-value block is **indexed** at open, not decoded. A value is
//! decoded when asked for. This is not a micro-optimization: the largest
//! file in the reference corpus declares 777,056 strings, so decoding
//! eagerly costs about 26 MB of allocations to answer a question about one
//! key.
//!
//! # What this crate will not do
//!
//! It does not interpret keys. There is no chat-template accessor, no
//! architecture detection, no config struct. Resolving
//! `tokenizer.ggml.bos_token_id` against `tokenizer.ggml.tokens` has seven
//! distinct failure modes, and getting them right needs knowledge of the
//! ecosystem that MLMF deliberately does not hold.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod cursor;
pub mod error;
```

`crates/mlmf-gguf/tests/allowed-std.list`:

```
# mlmf-gguf parses bytes.
#
# NOTE: `sync` is deliberately ABSENT until Task 6 needs it. Granting a
# permission before any code uses it means no red-then-green cycle ever
# proves the gate would catch its misuse, and an entry nothing needs can
# outlive the design that asked for it.
cmp
fmt
iter
option
primitive
result
slice
str
string
u32
u64
usize
vec
```

`crates/mlmf-gguf/tests/direct-deps.allow`:

```
# C2: mlmf-gguf's direct dependencies. Changing this file is a deliberate act.
# Keep sorted. Every entry must be justified in the design spec.
mlmf-core
mlmf-ggml
```

Add `"crates/mlmf-gguf"` to `default-members` in the root `Cargo.toml`. **`members = ["crates/*"]` picks it up for `--workspace`, but `default-members` does not** — without this, bare `cargo test` runs none of this crate's gates. Verify: `cargo test 2>&1 | grep -c mlmf-gguf` must be greater than zero.

- [ ] **Step 2: Write the failing cursor tests**

Create `crates/mlmf-gguf/src/cursor.rs` with **only** this test module, and add `pub mod cursor;` to `lib.rs` in the same step. A module not declared in `lib.rs` is never compiled, so its tests report `0 passed` and the run looks green while asserting nothing.

```rust
//! Bounds-checked little-endian reads over a byte slice.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_generated_reader_reads_its_own_width() {
        // All seven, by identity rather than by sampling two. The macro
        // guarantees the SHAPE of the generated bodies is identical; it does
        // not guarantee the type list is right. `u64 => u32` in the
        // invocation would produce a method named `u64` that reads four
        // bytes, and no amount of testing `u16` and `u32` would notice.
        //
        // `u64` matters most: it is the width GGUF uses for every string
        // length and every count, so Task 5 and Task 6 build directly on it.
        let b = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        assert_eq!(Cursor::new(&b).u8().unwrap(), 0x11);
        assert_eq!(Cursor::new(&b).u16().unwrap(), 0x2211);
        assert_eq!(Cursor::new(&b).u32().unwrap(), 0x4433_2211);
        assert_eq!(Cursor::new(&b).u64().unwrap(), 0x8877_6655_4433_2211);
        assert_eq!(
            Cursor::new(&b).i64().unwrap(),
            0x8877_6655_4433_2211u64 as i64
        );

        // And each advances by its own width. A reader that returns the
        // right value after consuming the wrong number of bytes leaves the
        // cursor desynchronised, and the symptom appears at the NEXT field
        // rather than at this one.
        let mut c = Cursor::new(&b);
        c.u8().unwrap();
        assert_eq!(c.pos(), 1, "u8");
        let mut c = Cursor::new(&b);
        c.u16().unwrap();
        assert_eq!(c.pos(), 2, "u16");
        let mut c = Cursor::new(&b);
        c.u32().unwrap();
        assert_eq!(c.pos(), 4, "u32");
        let mut c = Cursor::new(&b);
        c.u64().unwrap();
        assert_eq!(c.pos(), 8, "u64");
        let mut c = Cursor::new(&b);
        c.i64().unwrap();
        assert_eq!(c.pos(), 8, "i64");
        let mut c = Cursor::new(&b);
        c.f32().unwrap();
        assert_eq!(c.pos(), 4, "f32");
        let mut c = Cursor::new(&b);
        c.f64().unwrap();
        assert_eq!(c.pos(), 8, "f64");
    }

    #[test]
    fn a_short_read_reports_what_it_needed_and_what_was_there() {
        // Not just "failed": a truncated file is common and the numbers are
        // what let an operator tell a truncated download from a corrupt one.
        let bytes = [0x01u8, 0x02];
        let mut c = Cursor::new(&bytes);
        // One comparison over the whole error. Chaining `needed` then
        // `available` means a transposition in the construction — the two
        // sit adjacent, so it is a plausible slip — trips the first
        // assertion and the second never runs, blaming the wrong field.
        assert_eq!(
            c.u32().unwrap_err(),
            Truncated {
                needed: 4,
                available: 2
            }
        );
        // Position is a property of the cursor rather than of the error, so
        // it cannot fold into the comparison above and stays its own check.
        assert_eq!(c.pos(), 0);
    }

    #[test]
    fn take_borrows_rather_than_copies() {
        let bytes = [0xAAu8; 16];
        let mut c = Cursor::new(&bytes);
        let s = c.take(8).unwrap();
        assert_eq!(s.len(), 8);
        // Same allocation, not a copy — this is what lets a 15 MB KV block
        // be indexed without being duplicated.
        assert_eq!(s.as_ptr() as usize, bytes.as_ptr() as usize);
        assert_eq!(c.pos(), 8);
    }

    #[test]
    fn a_length_that_cannot_fit_the_file_fails_before_allocating() {
        // The adversarial case: a declared string length of u64::MAX. A
        // reader that trusts it and allocates first is a denial of service
        // triggered by four bytes of a header.
        let bytes = [0u8; 8];
        let mut c = Cursor::new(&bytes);
        assert_eq!(
            c.take(u64::MAX).unwrap_err(),
            Truncated {
                needed: u64::MAX,
                available: 8
            }
        );
        // The same guarantee its sibling short-read test pins: a refused
        // read must not move the cursor, or every later error reports an
        // offset that is wrong by however much the failed read consumed.
        assert_eq!(c.pos(), 0);
    }

    #[test]
    fn seek_refuses_a_position_past_the_end() {
        let bytes = [0u8; 8];
        let mut c = Cursor::new(&bytes);
        c.seek(8).expect("the end is a legal position");
        assert_eq!(c.pos(), 8);
        assert!(c.seek(9).is_err());
        // A refused seek must not move the cursor.
        assert_eq!(c.pos(), 8);
    }

    #[test]
    fn floats_are_bit_exact_not_approximately_decoded() {
        // Compare BITS, and pick values that a lossy route would damage.
        //
        // A first draft used -1.5 and claimed exact equality would catch a
        // detour through f64. Both halves of that were wrong. -1.5 is
        // exactly representable in f32, f64 and decimal, so it distinguishes
        // nothing — and an f32 -> f64 -> f32 round trip is bit-exact for
        // EVERY f32, because widening never rounds and narrowing back with
        // round-to-nearest recovers the original. There is no double
        // rounding to lose. So the f64 detour was never a hazard worth
        // naming; the real ones are flush-to-zero and byte order, which is
        // what these values and this comparison are chosen for.
        let third = f32::from_bits(0x3EAA_AAAB); // nearest f32 to 1/3
        assert_eq!(
            Cursor::new(&third.to_le_bytes()).f32().unwrap().to_bits(),
            0x3EAA_AAAB
        );

        // A subnormal. An implementation that normalises on conversion —
        // or runs with flush-to-zero — turns this into 0.0, and comparing
        // bits is what catches it. Comparing values would too, but only
        // because 0.0 != this; comparing bits also catches a sign or
        // payload change that compares equal as a float.
        let subnormal = f32::from_bits(0x0000_0001);
        assert_eq!(
            Cursor::new(&subnormal.to_le_bytes()).f32().unwrap().to_bits(),
            0x0000_0001
        );

        let w = f64::from_bits(0x3FD5_5555_5555_5555); // nearest f64 to 1/3
        assert_eq!(
            Cursor::new(&w.to_le_bytes()).f64().unwrap().to_bits(),
            0x3FD5_5555_5555_5555
        );
    }
}
```

- [ ] **Step 3: Run and watch it fail**

```bash
cargo test -p mlmf-gguf
```

Expected: compile error, `cannot find type Cursor in this scope`. If you see `0 passed; 0 failed`, `pub mod cursor;` is missing from `lib.rs`.

- [ ] **Step 4: Implement the cursor**

Above the test module:

```rust
/// A read that ran off the end of the slice.
///
/// Carries both numbers because "failed to read" does not distinguish a
/// truncated download from a file whose declared lengths are nonsense, and
/// those are different operator problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truncated {
    /// Bytes the read required.
    pub needed: u64,
    /// Bytes actually available from the current position.
    pub available: u64,
}

/// A bounds-checked little-endian reader over borrowed bytes.
///
/// Every method that can run off the end returns [`Truncated`] and **leaves
/// the position unchanged**, so the caller can report the offset at which
/// the file stopped making sense.
#[derive(Debug)]
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// A cursor at the start of `bytes`.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Current offset from the start of the slice.
    #[must_use]
    pub fn pos(&self) -> u64 {
        self.pos as u64
    }

    /// Bytes remaining from the current position.
    ///
    /// Public because a parser needs it to bound a *declared* count before
    /// trusting it: no container can hold more elements than it has bytes,
    /// since every element occupies at least one. That check is what turns
    /// an absurd declared length into an error instead of an allocation.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        (self.bytes.len() - self.pos) as u64
    }

    /// Move to an absolute position.
    ///
    /// # Errors
    ///
    /// [`Truncated`] if `pos` is past the end. The end itself is legal.
    pub fn seek(&mut self, pos: u64) -> Result<(), Truncated> {
        let want = usize::try_from(pos).map_err(|_| Truncated {
            needed: pos,
            available: self.bytes.len() as u64,
        })?;
        if want > self.bytes.len() {
            return Err(Truncated {
                needed: pos,
                available: self.bytes.len() as u64,
            });
        }
        self.pos = want;
        Ok(())
    }

    /// Borrow the next `n` bytes.
    ///
    /// # Errors
    ///
    /// [`Truncated`] if fewer than `n` bytes remain. **Checked before any
    /// allocation**, so a declared length of `u64::MAX` costs a comparison
    /// rather than an out-of-memory abort.
    pub fn take(&mut self, n: u64) -> Result<&'a [u8], Truncated> {
        if n > self.remaining() {
            return Err(Truncated {
                needed: n,
                available: self.remaining(),
            });
        }
        let n = n as usize; // <= remaining, so it fits
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
}

/// Generates the fixed-width little-endian readers.
///
/// A macro rather than eight hand-written near-identical bodies: the
/// bounds check and the position update must be the same in all of them,
/// and eight copies is eight chances for one to differ.
macro_rules! fixed_width {
    ($($name:ident => $ty:ty),* $(,)?) => {$(
        impl Cursor<'_> {
            #[doc = concat!("Read a little-endian `", stringify!($ty), "`.")]
            ///
            /// # Errors
            ///
            /// [`Truncated`] if too few bytes remain; the position is unchanged.
            pub fn $name(&mut self) -> Result<$ty, Truncated> {
                const N: usize = core::mem::size_of::<$ty>();
                let raw = self.take(N as u64)?;
                let mut buf = [0u8; N];
                buf.copy_from_slice(raw);
                Ok(<$ty>::from_le_bytes(buf))
            }
        }
    )*};
}

fixed_width! {
    u8 => u8, u16 => u16, u32 => u32, u64 => u64,
    i64 => i64, f32 => f32, f64 => f64,
}
```

- [ ] **Step 5: Run and watch it pass**

```bash
cargo test -p mlmf-gguf
cargo clippy -p mlmf-gguf --all-targets -- -D warnings
```

- [ ] **Step 6: Prove the gates and the cursor can fail (AD-2)**

1. Add `use std::{fs, path::Path};` and `pub fn r(p:&Path)->Vec<u8>{fs::read(p).unwrap()}` to `src/lib.rs`. The consolidated purity gate must go red **naming `mlmf-gguf`** — this crate is new, so this is the first proof the Task 2 loop actually reaches it. Restore.
2. Add `byteorder = "1"` to `[dependencies]`. The deps gate must go red naming it. Restore.
3. In `take`, change the bounds check to `if n > self.remaining() + 8`. `a_short_read_reports_what_it_needed_and_what_was_there` must go red. Restore.

3b. **Isolate the advance half of the width test.** Inside the `fixed_width!` macro body, after `let raw = self.take(N as u64)?;` add `self.pos -= 1;`. Every value assertion still passes — the bytes read are correct — and only the seven `assert_eq!(c.pos(), ...)` assertions fail. Confirm the failure names a width. Restore.

   **Do not use `u64 => u32` in the macro invocation for this.** It goes red, but as a *compile error*: the deny-by-default `overflowing_literals` lint rejects the 64-bit test literal before any test runs, which halts the binary and proves nothing about whether the advance assertions are live. A control that stops the compiler has not exercised anything.

3c. **Prove the subnormal case is load-bearing.** In the `f32` reader, flush subnormals: `let v = <f32>::from_le_bytes(buf); Ok(if v.is_subnormal() { 0.0 } else { v })`. Only the subnormal assertion in `floats_are_bit_exact_not_approximately_decoded` may fail — the 1/3 case must stay green, which is what makes the two values complementary rather than redundant. Restore.

   **Do not use an `f64` round trip for this.** `f32 -> f64 -> f32` is bit-exact for every `f32`: widening never rounds, and narrowing back with round-to-nearest recovers the original, so there is no double rounding to lose. It is a mathematical no-op and cannot fail.
4. In `take`, make the **failure path itself** consume the remainder before returning:

   ```rust
   if n > self.remaining() {
       let available = self.remaining(); // captured BEFORE the mutation
       self.pos = self.bytes.len();      // wrongly consume on failure
       return Err(Truncated { needed: n, available });
   }
   ```

   `a_short_read_reports_what_it_needed_and_what_was_there` must go red on its `assert_eq!(c.pos(), 0)` assertion. Restore.

   **Capturing `available` first is what makes this mutation isolating, and that detail is load-bearing.** Without it, `self.pos = self.bytes.len()` changes what `self.remaining()` returns, so the test trips its `assert_eq!(e.available, 2)` assertion — which sits *above* the `pos` one — and panics before `pos` is ever checked. The test goes red either way, which looks like success and proves nothing about the assertion you were trying to exercise. **An earlier assertion in the same test can shadow a later one, so "the test went red" is not evidence that a particular assertion works.** Check which assertion fires, not merely that one did.

   **A first draft of this plan named the wrong mutation here** — "move the position update before the slice" — and an implementer correctly refused to fudge it when the suite stayed green. The bounds check is an *early return*, so the failure path never touches `self.pos`, and reordering the statements after the guard is invisible to a failing read. The assertion was right; the control named a mutation that could not reach it. The version above does reach it, and "consume what is available, then error" is a plausible thing someone writes on purpose — which is what a control should target.

- [ ] **Step 7: Commit**

```bash
cargo fmt --all
git add crates/mlmf-gguf Cargo.toml
git commit -m "feat(gguf): crate skeleton, gates, and a bounds-checked cursor"
```

Then verify the licences reached the tarball, which only works after committing:

```bash
cargo package --list -p mlmf-gguf | grep -i license
```

Both files must appear. If not, they were not committed.

---

## Task 4: The header stage, and the error split R7 requires

**Files:**
- Create: `crates/mlmf-gguf/src/error.rs`, `crates/mlmf-gguf/src/header.rs`
- Modify: `crates/mlmf-gguf/src/lib.rs`

**Interfaces:**
- Consumes: `Cursor`, `Truncated`.
- Produces:
  ```rust
  pub enum GgufError {
      NotGguf { found: [u8; 4] },
      ByteSwapped { raw_version: u32 },
      UnsupportedVersion { version: u32 },
      Truncated { stage: Stage, offset: u64, needed: u64, available: u64 },
      Malformed { stage: Stage, offset: u64, detail: String },
  }
  pub enum Stage { Header, Metadata, TensorDirectory }
  pub struct Header { pub version: u32, pub tensor_count: u64, pub kv_count: u64, pub end: u64 }
  pub fn parse_header(cursor: &mut Cursor<'_>) -> Result<Header, GgufError>;
  ```

- [ ] **Step 1: Write the failing tests**

Create `crates/mlmf-gguf/src/header.rs` with this test module, and add `pub mod header;` to `lib.rs` in the same step.

```rust
//! The first stage: magic, version, and the two counts.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::Cursor;
    use crate::error::{GgufError, Stage};

    /// A well-formed v3 header declaring `n_tensors` and `n_kv`.
    fn header_bytes(version: u32, n_tensors: i64, n_kv: i64) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"GGUF");
        v.extend_from_slice(&version.to_le_bytes());
        v.extend_from_slice(&n_tensors.to_le_bytes());
        v.extend_from_slice(&n_kv.to_le_bytes());
        v
    }

    #[test]
    fn reads_a_well_formed_v3_header() {
        let b = header_bytes(3, 291, 42);
        let mut c = Cursor::new(&b);
        // Whole-value comparison: a transposition of `tensor_count` and
        // `kv_count` in the constructor would otherwise trip the first
        // assertion and leave the second unproven.
        assert_eq!(
            parse_header(&mut c).expect("valid header"),
            Header {
                version: 3,
                tensor_count: 291,
                kv_count: 42,
                end: 24, // magic 4 + version 4 + two i64
            }
        );
    }

    #[test]
    fn a_file_that_is_not_a_gguf_says_so_rather_than_reporting_corruption() {
        // R7. This is the residue of the consumer's own extension check: a
        // file NAMED .gguf whose magic is something else. "Not what its name
        // claims" and "malformed GGUF" are different operator remedies, and
        // collapsing them makes the warning hedge across both.
        let mut b = header_bytes(3, 0, 0);
        b[0..4].copy_from_slice(b"PK\x03\x04"); // a zip, the common mistake
        let mut c = Cursor::new(&b);
        match parse_header(&mut c).unwrap_err() {
            GgufError::NotGguf { found } => assert_eq!(&found, b"PK\x03\x04"),
            other => panic!("expected NotGguf, got {other:?}"),
        }
    }

    #[test]
    fn a_byte_swapped_file_is_named_rather_than_reported_as_a_huge_version() {
        // llama.cpp's own check (gguf.cpp:497): version 3 read with the
        // wrong endianness is 0x03000000. Without this the error says
        // "unsupported version 50331648", which sends the reader looking
        // for a version that has never existed.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&0x0300_0000u32.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes());
        let mut c = Cursor::new(&b);
        match parse_header(&mut c).unwrap_err() {
            GgufError::ByteSwapped { raw_version } => assert_eq!(raw_version, 0x0300_0000),
            other => panic!("expected ByteSwapped, got {other:?}"),
        }
    }

    #[test]
    fn v1_is_refused_by_version_rather_than_misparsed() {
        // v1 used 32-bit counts. Parsing it with 64-bit reads produces
        // plausible-looking garbage rather than an error, which is worse
        // than refusing. Out of scope for this crate; the error says which
        // version so the message is actionable.
        let b = header_bytes(1, 0, 0);
        let mut c = Cursor::new(&b);
        match parse_header(&mut c).unwrap_err() {
            GgufError::UnsupportedVersion { version } => assert_eq!(version, 1),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn v2_and_v3_are_both_accepted() {
        // v2 differs from v3 only below the header, and the corpus contains
        // one (ggml-vocab-aquila.gguf). Refusing it would reject real files.
        for v in [2u32, 3] {
            let b = header_bytes(v, 1, 1);
            let mut c = Cursor::new(&b);
            assert_eq!(parse_header(&mut c).expect("accepted").version, v);
        }
    }

    #[test]
    fn a_future_version_is_refused_with_its_number() {
        let b = header_bytes(4, 0, 0);
        let mut c = Cursor::new(&b);
        assert!(matches!(
            parse_header(&mut c).unwrap_err(),
            GgufError::UnsupportedVersion { version: 4 }
        ));
    }

    #[test]
    fn a_negative_count_is_malformed_rather_than_a_huge_unsigned_one() {
        // The counts are i64 on the wire. Reading -1 as u64 gives
        // 18446744073709551615, and a parser that then loops that many
        // times has turned eight bytes into a hang.
        let b = header_bytes(3, -1, 0);
        let mut c = Cursor::new(&b);
        // Whole-value comparison, for two reasons. A chain of field
        // assertions carries ordering bias — one on `stage` above one on
        // `detail` means a mutation touching only `detail` is masked or
        // misattributed. And this test's own sabotage stops the error being
        // produced at all, so it panics at `unwrap_err` before any inner
        // assertion is reached: those assertions were never proven live.
        assert_eq!(
            parse_header(&mut c).unwrap_err(),
            GgufError::Malformed {
                stage: Stage::Header,
                offset: 8,
                detail: "tensor count is negative: -1".to_string(),
            }
        );
    }

    #[test]
    fn a_truncated_header_reports_the_stage_and_the_offset() {
        let b = b"GGUF\x03\x00\x00\x00\x01".to_vec(); // 9 bytes: magic, version, one stray
        let mut c = Cursor::new(&b);
        // Whole-value comparison rather than a chain of field assertions.
        // Chaining means a transposition of `needed` and `available` — a
        // plausible one-line slip, since they are adjacent in the
        // construction — trips the `needed` assertion and the `available`
        // one never runs, so the failure names the wrong field.
        assert_eq!(
            parse_header(&mut c).unwrap_err(),
            GgufError::Truncated {
                stage: Stage::Header,
                offset: 8, // the tensor count starts at byte 8
                needed: 8,
                available: 1,
            }
        );
    }

    #[test]
    fn an_empty_slice_is_not_gguf_rather_than_a_panic() {
        let mut c = Cursor::new(&[]);
        assert!(matches!(
            parse_header(&mut c).unwrap_err(),
            GgufError::Truncated { stage: Stage::Header, .. }
        ));
    }
}
```

- [ ] **Step 2: Run and watch it fail**

```bash
cargo test -p mlmf-gguf --lib header
```

Expected: compile error, `cannot find function parse_header`.

- [ ] **Step 3: Write `error.rs`**

```rust
//! What went wrong, and at which stage.
//!
//! The stage split exists because a consumer's remedies differ (R7). A file
//! whose magic is wrong is **not the file its name claims**; a file whose
//! magic is right and whose contents are broken **is a GGUF and is
//! malformed**. Collapsing those forces a warning to hedge across both, and
//! sends an operator looking for corruption in a file they simply pointed
//! at by mistake.

use std::fmt;

/// Which stage of the staged parse produced an error.
///
/// Naming the stage is what makes R1 checkable from the outside: a
/// `Stage::TensorDirectory` error must never be able to prevent metadata
/// from having been read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stage {
    /// Magic, version, and the two counts.
    Header,
    /// The key-value block.
    Metadata,
    /// The tensor directory. Reached only by the next plan.
    TensorDirectory,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Stage::Header => "header",
            Stage::Metadata => "metadata",
            Stage::TensorDirectory => "tensor directory",
        };
        f.write_str(s)
    }
}

/// A GGUF parse failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GgufError {
    /// The first four bytes are not `GGUF`. **This file is not what its
    /// name claims** — distinct from a malformed GGUF (R7).
    NotGguf {
        /// The four bytes actually found.
        found: [u8; 4],
    },
    /// The magic is right but the version has zeroes in its low half, which
    /// means the file was written on the other endianness.
    ByteSwapped {
        /// The version field exactly as read.
        raw_version: u32,
    },
    /// A GGUF of a version this build does not read.
    UnsupportedVersion {
        /// The declared version.
        version: u32,
    },
    /// The file ended before a declared structure did.
    Truncated {
        /// Stage that was reading.
        stage: Stage,
        /// Offset the read started at.
        offset: u64,
        /// Bytes the read needed.
        needed: u64,
        /// Bytes that remained.
        available: u64,
    },
    /// The bytes are present and do not make sense.
    Malformed {
        /// Stage that was reading.
        stage: Stage,
        /// Offset the problem was found at.
        offset: u64,
        /// What was wrong, naming the value seen.
        detail: String,
    },
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::NotGguf { found } => write!(
                f,
                "not a GGUF file: expected magic \"GGUF\", found {found:?}"
            ),
            GgufError::ByteSwapped { raw_version } => write!(
                f,
                "this GGUF was written with the opposite byte order \
                 (version field reads {raw_version:#010x})"
            ),
            GgufError::UnsupportedVersion { version } => {
                write!(f, "GGUF version {version} is not supported by this build")
            }
            GgufError::Truncated {
                stage,
                offset,
                needed,
                available,
            } => write!(
                f,
                "truncated in the {stage} at offset {offset}: \
                 needed {needed} bytes, {available} remain"
            ),
            GgufError::Malformed {
                stage,
                offset,
                detail,
            } => write!(f, "malformed {stage} at offset {offset}: {detail}"),
        }
    }
}

impl std::error::Error for GgufError {}
```

**Note on `std::error`:** add `error` to `crates/mlmf-gguf/tests/allowed-std.list`. It is a trait module, not a way out of the process.

- [ ] **Step 4: Write `header.rs`**

```rust
use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};

/// Versions this build reads. v1 used 32-bit counts and is refused rather
/// than misparsed — see the crate docs.
const SUPPORTED: &[u32] = &[2, 3];

/// The fixed-size prologue of a GGUF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Declared format version.
    pub version: u32,
    /// Number of tensors the file declares.
    pub tensor_count: u64,
    /// Number of key-value pairs the file declares.
    pub kv_count: u64,
    /// Offset just past the header, where the KV block begins.
    pub end: u64,
}

/// Read magic, version and counts.
///
/// # Errors
///
/// [`GgufError::NotGguf`] if the magic is wrong — which is a different fact
/// from a malformed GGUF and is reported differently (R7).
/// [`GgufError::ByteSwapped`] if the version indicates opposite-endian
/// authorship. [`GgufError::UnsupportedVersion`] for v1 or anything newer
/// than this build. [`GgufError::Truncated`] or [`GgufError::Malformed`]
/// otherwise.
pub fn parse_header(cursor: &mut Cursor<'_>) -> Result<Header, GgufError> {
    let at = cursor.pos();
    let magic = cursor.take(4).map_err(|t| GgufError::Truncated {
        stage: Stage::Header,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;
    if magic != b"GGUF" {
        let mut found = [0u8; 4];
        found.copy_from_slice(magic);
        return Err(GgufError::NotGguf { found });
    }

    let at = cursor.pos();
    let version = cursor.u32().map_err(|t| GgufError::Truncated {
        stage: Stage::Header,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;

    // llama.cpp's own sniff (gguf.cpp:497). Checked BEFORE the supported
    // list, so a byte-swapped v3 is reported as byte-swapped rather than as
    // "unsupported version 50331648".
    if version & 0x0000_FFFF == 0 {
        return Err(GgufError::ByteSwapped {
            raw_version: version,
        });
    }
    if !SUPPORTED.contains(&version) {
        return Err(GgufError::UnsupportedVersion { version });
    }

    let tensor_count = read_count(cursor, "tensor count")?;
    let kv_count = read_count(cursor, "key-value count")?;

    Ok(Header {
        version,
        tensor_count,
        kv_count,
        end: cursor.pos(),
    })
}

/// Read one `i64` count and refuse a negative.
///
/// The counts are signed on the wire. Reinterpreting -1 as `u64` yields
/// 18446744073709551615, and a parser that then loops that many times has
/// turned eight bytes of a header into a hang.
fn read_count(cursor: &mut Cursor<'_>, what: &str) -> Result<u64, GgufError> {
    let at = cursor.pos();
    let raw = cursor.i64().map_err(|t| GgufError::Truncated {
        stage: Stage::Header,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;
    u64::try_from(raw).map_err(|_| GgufError::Malformed {
        stage: Stage::Header,
        offset: at,
        detail: format!("{what} is negative: {raw}"),
    })
}
```

Add `pub mod error;` and `pub mod header;` to `lib.rs`, and re-export:

```rust
pub use error::{GgufError, Stage};
pub use header::{parse_header, Header};
```

- [ ] **Step 5: Run and watch it pass**

```bash
cargo test -p mlmf-gguf
cargo clippy -p mlmf-gguf --all-targets -- -D warnings
```

- [ ] **Step 6: Prove the tests can fail (AD-2)**

1. Move the `SUPPORTED.contains` check **above** the byte-swap check. `a_byte_swapped_file_is_named_rather_than_reported_as_a_huge_version` must go red — it now reports `UnsupportedVersion { version: 50331648 }`. **This ordering is the whole point of that test**; record the number you see. Restore.
2. Change `read_count` to `Ok(raw as u64)`. `a_negative_count_is_malformed_rather_than_a_huge_unsigned_one` must go red. Restore.
3. Change the magic comparison to `magic[0] == b'G'` — a prefix check rather than a magic check. This one needs **two distinct fixtures**, not one fixture edited between phases: a single test name asserted to both stay green and go red is a claim whose only discriminator is a sentence, and nothing executes a sentence.

   Add this second test alongside the first, permanently:

   ```rust
    #[test]
    fn a_wrong_magic_sharing_gguf_s_first_byte_is_still_not_a_gguf() {
        // The positive control for the magic check, kept as its own test so
        // that the claim and its control are two names rather than one name
        // in two states. "GGJT" is the real legacy ggml magic, so this is
        // the file a user actually holds when they hit this path.
        let mut b = header_bytes(3, 0, 0);
        b[0..4].copy_from_slice(b"GGJT");
        let mut c = Cursor::new(&b);
        match parse_header(&mut c).unwrap_err() {
            GgufError::NotGguf { found } => assert_eq!(&found, b"GGJT"),
            other => panic!("expected NotGguf, got {other:?}"),
        }
    }
   ```

   Under the sabotage `a_file_that_is_not_a_gguf_says_so_rather_than_reporting_corruption` stays green, because `PK` fails on byte 0 either way, and `a_wrong_magic_sharing_gguf_s_first_byte_is_still_not_a_gguf` goes red. Two names, two outcomes, no prose in between. Restore.

- [ ] **Step 7: Commit**

```bash
cargo fmt --all
git add crates/mlmf-gguf
git commit -m "feat(gguf): header stage with the not-a-GGUF / malformed split"
```

---

## Task 5: Value types and byte-range skipping

The index needs to know where each value **ends** without decoding it. That is the whole basis of D1.

**Files:**
- Create: `crates/mlmf-gguf/src/value.rs`
- Modify: `crates/mlmf-gguf/src/lib.rs`

**Interfaces:**
- Consumes: `Cursor`, `GgufError`, `Stage`.
- Produces:
  ```rust
  pub enum ValueType { U8, I8, U16, I16, U32, I32, F32, Bool, String, Array, U64, I64, F64 }
  impl ValueType {
      pub fn from_code(code: u32) -> Option<ValueType>;
      pub fn code(self) -> u32;
      pub fn fixed_width(self) -> Option<u64>;
  }
  pub fn skip_value(cursor: &mut Cursor<'_>, ty: ValueType) -> Result<(), GgufError>;
  pub fn decode_value(cursor: &mut Cursor<'_>, ty: ValueType) -> Result<MetaValue, GgufError>;
  ```

- [ ] **Step 1: Write the failing tests**

Create `crates/mlmf-gguf/src/value.rs` with this test module; add `pub mod value;` to `lib.rs` in the same step.

```rust
//! GGUF's thirteen value types: their codes, their widths, and how to walk
//! past one without decoding it.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::Cursor;
    use crate::error::{GgufError, Stage};
    use mlmf_core::MetaValue;

    /// Every code, name and fixed width, written out. Ground truth is
    /// `gguf.h`'s `enum gguf_type` at the pinned commit.
    const TABLE: [(u32, ValueType, Option<u64>); 13] = [
        (0, ValueType::U8, Some(1)),
        (1, ValueType::I8, Some(1)),
        (2, ValueType::U16, Some(2)),
        (3, ValueType::I16, Some(2)),
        (4, ValueType::U32, Some(4)),
        (5, ValueType::I32, Some(4)),
        (6, ValueType::F32, Some(4)),
        (7, ValueType::Bool, Some(1)),
        (8, ValueType::String, None),
        (9, ValueType::Array, None),
        (10, ValueType::U64, Some(8)),
        (11, ValueType::I64, Some(8)),
        (12, ValueType::F64, Some(8)),
    ];

    #[test]
    fn every_code_round_trips_and_reports_its_width() {
        for (code, ty, width) in TABLE {
            assert_eq!(ValueType::from_code(code), Some(ty), "code {code}");
            assert_eq!(ty.code(), code, "{ty:?}");
            assert_eq!(ty.fixed_width(), width, "{ty:?}");
        }
        // Bool is one byte on the wire, not four. A four-byte assumption
        // desynchronises every subsequent key in the block, and the symptom
        // is a garbage key name rather than an error at the bool.
        assert_eq!(ValueType::Bool.fixed_width(), Some(1));
    }

    #[test]
    fn an_unknown_value_type_is_none_rather_than_an_error() {
        // Same rule as the ggml type table: refusing to construct would
        // leave the caller holding a bare u32 with nothing to report.
        assert_eq!(ValueType::from_code(13), None);
        assert_eq!(ValueType::from_code(u32::MAX), None);
    }

    #[test]
    fn skipping_a_string_lands_exactly_after_it() {
        // len(u64) + bytes, no terminator.
        let mut b = Vec::new();
        b.extend_from_slice(&5u64.to_le_bytes());
        b.extend_from_slice(b"hello");
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        skip_value(&mut c, ValueType::String).unwrap();
        assert_eq!(c.pos(), 13);
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn skipping_an_array_of_strings_walks_every_element() {
        // The case that matters: 500k-element arrays must be skippable
        // without decoding. Element type u32, count u64, then elements.
        let mut b = Vec::new();
        b.extend_from_slice(&8u32.to_le_bytes()); // element type: String
        b.extend_from_slice(&3u64.to_le_bytes()); // count
        for s in ["a", "bb", "ccc"] {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        skip_value(&mut c, ValueType::Array).unwrap();
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn skipping_a_fixed_width_array_does_not_walk_element_by_element() {
        // 4 bytes x 1000 must be one seek, not 1000 reads. Asserted by
        // behaviour rather than by timing: the sentinel proves the landing
        // point, and the count is large enough that an element-wise walk
        // over a truncated buffer would fail instead of succeeding.
        let mut b = Vec::new();
        b.extend_from_slice(&4u32.to_le_bytes()); // U32
        b.extend_from_slice(&1000u64.to_le_bytes());
        b.extend_from_slice(&vec![0u8; 4000]);
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        skip_value(&mut c, ValueType::Array).unwrap();
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn a_nested_array_is_walked_rather_than_refused() {
        // GGUF permits arrays of arrays. Rare, but a parser that assumes
        // one level desynchronises rather than erroring.
        let mut inner = Vec::new();
        inner.extend_from_slice(&4u32.to_le_bytes()); // U32 elements
        inner.extend_from_slice(&2u64.to_le_bytes());
        inner.extend_from_slice(&1u32.to_le_bytes());
        inner.extend_from_slice(&2u32.to_le_bytes());

        let mut b = Vec::new();
        b.extend_from_slice(&9u32.to_le_bytes()); // element type: Array
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&inner);
        b.extend_from_slice(&inner);
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        skip_value(&mut c, ValueType::Array).unwrap();
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn an_array_of_unknown_element_type_is_malformed_not_silently_skipped() {
        let mut b = Vec::new();
        b.extend_from_slice(&99u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let mut c = Cursor::new(&b);
        // One comparison over the whole error, against a fully-specified
        // literal, rather than chaining `stage` then `detail`: chaining
        // means a `stage` mismatch fires first and the `detail` assertion
        // — the one that actually proves the code was named — never runs.
        assert_eq!(
            skip_value(&mut c, ValueType::Array).unwrap_err(),
            GgufError::Malformed {
                stage: Stage::Metadata,
                offset: 0,
                detail: "array declares unknown element type 99".to_string(),
            }
        );
    }

    #[test]
    fn a_string_length_larger_than_the_file_fails_before_allocating() {
        let b = 0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes().to_vec();
        let mut c = Cursor::new(&b);
        assert!(matches!(
            skip_value(&mut c, ValueType::String).unwrap_err(),
            GgufError::Truncated { stage: Stage::Metadata, .. }
        ));
    }

    #[test]
    fn decoding_preserves_bytes_exactly_including_invalid_utf8() {
        // R3, and the case no real file provides: 0xFF 0xFE is not valid
        // UTF-8. A `from_utf8_lossy` here replaces it with U+FFFD and the
        // tokenizer never matches the token again — with no error anywhere.
        let raw = [0xFFu8, 0xFE, b'a'];
        let mut b = Vec::new();
        b.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        b.extend_from_slice(&raw);
        let mut c = Cursor::new(&b);
        match decode_value(&mut c, ValueType::String).unwrap() {
            MetaValue::Bytes(got) => assert_eq!(got, raw),
            other => panic!("non-UTF-8 must survive as Bytes, got {other:?}"),
        }
    }

    #[test]
    fn a_trailing_nul_is_part_of_the_string_not_a_terminator() {
        // GGUF strings are length-prefixed with NO terminator. A parser
        // that strips a trailing NUL silently shortens a legal string, and
        // no real file in the corpus has one to catch it.
        let raw = b"end\0";
        let mut b = Vec::new();
        b.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        b.extend_from_slice(raw);
        let mut c = Cursor::new(&b);
        match decode_value(&mut c, ValueType::String).unwrap() {
            MetaValue::String(s) => assert_eq!(s.as_bytes(), raw, "the NUL is data"),
            other => panic!("expected String, got {other:?}"),
        }
    }

    #[test]
    fn decode_value_round_trips_every_type() {
        // `decode_value` had tests for exactly two of thirteen types, and
        // none at all for its Array arm. This loop covers the eleven
        // scalars; String and Array are exercised by their own tests
        // because both need structured fixtures rather than a width and a
        // literal — the branch carrying this crate's
        // whole 500,000-element rationale. A copy-paste width slip in any
        // scalar arm (`I16` reaching for `cursor.u32()`, say) would
        // desynchronise the cursor and no test would have noticed.
        //
        // Each case asserts the decoded value AND the bytes consumed,
        // because a reader that returns the right value after eating the
        // wrong number of bytes breaks the NEXT key rather than this one.
        let cases: Vec<(ValueType, Vec<u8>, MetaValue, u64)> = vec![
            (ValueType::U8, vec![0xFE], MetaValue::U8(0xFE), 1),
            (ValueType::I8, vec![0xFE], MetaValue::I8(-2), 1),
            (ValueType::U16, vec![0x01, 0x02], MetaValue::U16(0x0201), 2),
            (ValueType::I16, vec![0xFE, 0xFF], MetaValue::I16(-2), 2),
            (
                ValueType::U32,
                vec![0x01, 0x02, 0x03, 0x04],
                MetaValue::U32(0x0403_0201),
                4,
            ),
            (
                ValueType::I32,
                vec![0xFE, 0xFF, 0xFF, 0xFF],
                MetaValue::I32(-2),
                4,
            ),
            (
                ValueType::U64,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                MetaValue::U64(0x0807_0605_0403_0201),
                8,
            ),
            (
                ValueType::I64,
                vec![0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                MetaValue::I64(-2),
                8,
            ),
            (
                ValueType::F32,
                (-1.5f32).to_le_bytes().to_vec(),
                MetaValue::F32(-1.5),
                4,
            ),
            (
                ValueType::F64,
                (0.25f64).to_le_bytes().to_vec(),
                MetaValue::F64(0.25),
                8,
            ),
            (ValueType::Bool, vec![0], MetaValue::Bool(false), 1),
        ];
        for (ty, bytes, want, width) in cases {
            let mut c = Cursor::new(&bytes);
            assert_eq!(decode_value(&mut c, ty).unwrap(), want, "{ty:?} value");
            assert_eq!(c.pos(), width, "{ty:?} consumed the wrong width");
        }
    }

    #[test]
    fn decoding_an_array_yields_its_elements_and_consumes_it_exactly() {
        // The Array arm, which had no test at all — not even a happy path.
        let mut b = 4u32.to_le_bytes().to_vec(); // U32 elements
        b.extend_from_slice(&3u64.to_le_bytes());
        for v in [10u32, 20, 30] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(b"SENTINEL");
        let mut c = Cursor::new(&b);
        assert_eq!(
            decode_value(&mut c, ValueType::Array).unwrap(),
            MetaValue::Array(vec![
                MetaValue::U32(10),
                MetaValue::U32(20),
                MetaValue::U32(30)
            ])
        );
        // Landing point, so a decode that over- or under-runs is caught
        // here rather than at whatever key follows it in a real file.
        assert_eq!(c.take(8).unwrap(), b"SENTINEL");
    }

    #[test]
    fn an_array_claiming_more_elements_than_the_file_has_bytes_is_refused() {
        // Twelve bytes of prefix claiming 2^40 elements.
        //
        // This fixture proves the bound NAMES the right thing, not that it
        // prevents a crash — a sabotage established that `try_reserve`
        // already refuses a request this large, so removing the bound still
        // errors, just citing allocator capacity instead of the invariant
        // actually violated. The case the bound uniquely catches is milder
        // and untestable here without a large fixture: a count of ten
        // million against a small file, which allocates hundreds of
        // megabytes before failing on truncation partway through the loop.
        let mut b = 4u32.to_le_bytes().to_vec(); // U32 elements
        b.extend_from_slice(&(1u64 << 40).to_le_bytes());
        let mut c = Cursor::new(&b);
        let err = decode_value(&mut c, ValueType::Array).unwrap_err();
        match err {
            GgufError::Malformed { detail, .. } => {
                assert!(
                    detail.contains("1099511627776") && detail.contains("bytes remain"),
                    "must name the count and the bound: {detail}"
                );
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn nesting_deeper_than_the_limit_is_refused_rather_than_overflowing_the_stack() {
        // Each level is twelve bytes: a 4-byte element type and an 8-byte
        // count of 1. That is how cheaply a file can ask for unbounded
        // recursion, and why the depth bound exists — a stack overflow
        // aborts the process instead of returning an error.
        // A literal depth, deliberately NOT `MAX_ARRAY_DEPTH + 5`. Writing
        // it in terms of the constant means raising the constant to
        // `u32::MAX` — the obvious sabotage — overflows the addition at
        // COMPILE time and rustc's `arithmetic_overflow` lint rejects the
        // crate before a test binary exists. The control could then never
        // run. A fixture that depends on the value it is testing cannot
        // survive that value being mutated.
        const NESTED: usize = 200;
        let mut b = Vec::new();
        for _ in 0..NESTED {
            b.extend_from_slice(&9u32.to_le_bytes()); // element type: Array
            b.extend_from_slice(&1u64.to_le_bytes());
        }
        b.extend_from_slice(&4u32.to_le_bytes()); // innermost: U32
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&7u32.to_le_bytes());

        let mut c = Cursor::new(&b);
        assert!(matches!(
            skip_value(&mut c, ValueType::Array).unwrap_err(),
            GgufError::Malformed { .. }
        ));
        let mut c = Cursor::new(&b);
        assert!(matches!(
            decode_value(&mut c, ValueType::Array).unwrap_err(),
            GgufError::Malformed { .. }
        ));
    }

    #[test]
    fn skip_and_decode_agree_on_what_they_refuse() {
        // The whole lazy design rests on this. `skip_value` decides what
        // gets indexed at open; `decode_value` decides what can be read
        // later. If they disagree, a key survives indexing and then fails
        // when somebody asks for it, and the file looks fine until the
        // moment it does not.
        //
        // Nothing forces them to agree — they walk the same grammar twice,
        // in two functions, and a depth bound added to both independently
        // produced exactly this divergence: the skip path's fixed-width
        // fast path elided a descent the decode path still made, so skip
        // accepted 64 levels that decode refused. This pins the agreement
        // rather than the individual behaviours.
        for nested in [1usize, 2, 63, 64, 65, 70] {
            let mut b = Vec::new();
            for _ in 0..nested {
                b.extend_from_slice(&9u32.to_le_bytes()); // element type: Array
                b.extend_from_slice(&1u64.to_le_bytes());
            }
            b.extend_from_slice(&4u32.to_le_bytes()); // a fixed-width leaf,
            b.extend_from_slice(&1u64.to_le_bytes()); // which is the case
            b.extend_from_slice(&7u32.to_le_bytes()); // the fast path takes

            let skipped = skip_value(&mut Cursor::new(&b), ValueType::Array).is_ok();
            let decoded = decode_value(&mut Cursor::new(&b), ValueType::Array).is_ok();
            assert_eq!(
                skipped, decoded,
                "{nested} levels: skip accepted {skipped}, decode accepted {decoded}"
            );
        }
    }

    #[test]
    fn a_bool_is_false_only_for_zero() {
        for (byte, want) in [(0u8, false), (1, true), (2, true), (255, true)] {
            let b = [byte];
            let mut c = Cursor::new(&b);
            assert_eq!(
                decode_value(&mut c, ValueType::Bool).unwrap(),
                MetaValue::Bool(want),
                "byte {byte}"
            );
        }
    }
}
```

- [ ] **Step 2: Run and watch it fail**

```bash
cargo test -p mlmf-gguf --lib value
```

Expected: compile error, `cannot find type ValueType`.

- [ ] **Step 3: Implement**

```rust
use mlmf_core::MetaValue;

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};

/// One of GGUF's thirteen metadata value types.
///
/// Codes are `gguf.h`'s `enum gguf_type`. `#[non_exhaustive]` because GGUF
/// may add one, and a consumer matching on this must not break when it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValueType {
    /// 8-bit unsigned.
    U8,
    /// 8-bit signed.
    I8,
    /// 16-bit unsigned.
    U16,
    /// 16-bit signed.
    I16,
    /// 32-bit unsigned.
    U32,
    /// 32-bit signed.
    I32,
    /// IEEE-754 binary32.
    F32,
    /// One byte; zero is false.
    Bool,
    /// Length-prefixed bytes, **no terminator**.
    String,
    /// Element type, count, then elements.
    Array,
    /// 64-bit unsigned.
    U64,
    /// 64-bit signed.
    I64,
    /// IEEE-754 binary64.
    F64,
}

impl ValueType {
    /// Every type, for exhaustive tests.
    pub const ALL: [ValueType; 13] = [
        ValueType::U8,
        ValueType::I8,
        ValueType::U16,
        ValueType::I16,
        ValueType::U32,
        ValueType::I32,
        ValueType::F32,
        ValueType::Bool,
        ValueType::String,
        ValueType::Array,
        ValueType::U64,
        ValueType::I64,
        ValueType::F64,
    ];

    /// The type for `code`, or `None` if this build has no row for it.
    ///
    /// Never an error: a caller holding an unknown code needs to report it,
    /// and refusing to construct leaves them with a bare `u32`.
    #[must_use]
    pub fn from_code(code: u32) -> Option<ValueType> {
        ValueType::ALL.into_iter().find(|t| t.code() == code)
    }

    /// The wire code.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            ValueType::U8 => 0,
            ValueType::I8 => 1,
            ValueType::U16 => 2,
            ValueType::I16 => 3,
            ValueType::U32 => 4,
            ValueType::I32 => 5,
            ValueType::F32 => 6,
            ValueType::Bool => 7,
            ValueType::String => 8,
            ValueType::Array => 9,
            ValueType::U64 => 10,
            ValueType::I64 => 11,
            ValueType::F64 => 12,
        }
    }

    /// Bytes one value occupies, for the types that have a fixed size.
    ///
    /// `None` for `String` and `Array`, whose length is in the data.
    /// **`Bool` is 1**, not 4 — a wrong width here desynchronises every
    /// subsequent key, and the symptom is a garbage key name rather than an
    /// error at the offending value.
    #[must_use]
    pub const fn fixed_width(self) -> Option<u64> {
        match self {
            ValueType::U8 | ValueType::I8 | ValueType::Bool => Some(1),
            ValueType::U16 | ValueType::I16 => Some(2),
            ValueType::U32 | ValueType::I32 | ValueType::F32 => Some(4),
            ValueType::U64 | ValueType::I64 | ValueType::F64 => Some(8),
            ValueType::String | ValueType::Array => None,
        }
    }
}

/// Maximum nesting depth for arrays of arrays.
///
/// GGUF permits an array whose elements are arrays; real files use depth 1
/// at most. Each level costs only 12 bytes on the wire — a 4-byte element
/// type and an 8-byte count — so a few hundred kilobytes of crafted input
/// drives recursion tens of thousands of frames deep and overflows the
/// stack, which **aborts the process** rather than returning an error.
///
/// Every other adversarial input in this crate becomes a `Result`: a
/// declared length larger than the file, a count that overflows, an
/// unknown type code. Nesting is the one that could still take the process
/// down, so it gets a bound too. 64 is far past anything a real writer
/// emits and far short of anything that threatens the stack.
const MAX_ARRAY_DEPTH: u32 = 64;

/// Advance past one value without decoding it.
///
/// This is what makes opening a file O(keys) rather than O(vocabulary):
/// a 500,000-element array is walked, never materialized.
///
/// # Errors
///
/// [`GgufError::Truncated`] if the file ends inside the value;
/// [`GgufError::Malformed`] if an array declares an element type this build
/// does not know.
pub fn skip_value(cursor: &mut Cursor<'_>, ty: ValueType) -> Result<(), GgufError> {
    skip_at_depth(cursor, ty, 0)
}

fn skip_at_depth(cursor: &mut Cursor<'_>, ty: ValueType, depth: u32) -> Result<(), GgufError> {
    if depth > MAX_ARRAY_DEPTH {
        return Err(GgufError::Malformed {
            stage: Stage::Metadata,
            offset: cursor.pos(),
            detail: format!("array nesting deeper than {MAX_ARRAY_DEPTH}"),
        });
    }
    if let Some(w) = ty.fixed_width() {
        let at = cursor.pos();
        cursor.take(w).map_err(|t| trunc(at, t))?;
        return Ok(());
    }
    match ty {
        ValueType::String => {
            let at = cursor.pos();
            let len = cursor.u64().map_err(|t| trunc(at, t))?;
            let at = cursor.pos();
            cursor.take(len).map_err(|t| trunc(at, t))?;
            Ok(())
        }
        ValueType::Array => {
            let (elem, count) = read_array_prefix(cursor)?;
            match elem.fixed_width() {
                // One seek for the whole run, not `count` reads.
                Some(w) => {
                    // The fast path elides the per-element descent that the
                    // element-wise branch performs, so it must still account
                    // for the level that descent would have cost. Without
                    // this, `skip_value` accepts nesting `decode_value`
                    // rejects — and since the lazy index skips at open and
                    // decodes on access, that means a key survives indexing
                    // and then fails when somebody reads it.
                    if depth + 1 > MAX_ARRAY_DEPTH {
                        return Err(GgufError::Malformed {
                            stage: Stage::Metadata,
                            offset: cursor.pos(),
                            detail: format!("array nesting deeper than {MAX_ARRAY_DEPTH}"),
                        });
                    }
                    let at = cursor.pos();
                    let total = count.checked_mul(w).ok_or_else(|| GgufError::Malformed {
                        stage: Stage::Metadata,
                        offset: at,
                        detail: format!("array of {count} x {w} bytes overflows u64"),
                    })?;
                    cursor.take(total).map_err(|t| trunc(at, t))?;
                }
                None => {
                    for _ in 0..count {
                        skip_at_depth(cursor, elem, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
        _ => unreachable!("every other type has a fixed width"),
    }
}

/// Decode one value.
///
/// # Errors
///
/// As [`skip_value`].
pub fn decode_value(cursor: &mut Cursor<'_>, ty: ValueType) -> Result<MetaValue, GgufError> {
    decode_at_depth(cursor, ty, 0)
}

fn decode_at_depth(
    cursor: &mut Cursor<'_>,
    ty: ValueType,
    depth: u32,
) -> Result<MetaValue, GgufError> {
    if depth > MAX_ARRAY_DEPTH {
        return Err(GgufError::Malformed {
            stage: Stage::Metadata,
            offset: cursor.pos(),
            detail: format!("array nesting deeper than {MAX_ARRAY_DEPTH}"),
        });
    }
    let at = cursor.pos();
    let v = match ty {
        ValueType::U8 => MetaValue::U8(cursor.u8().map_err(|t| trunc(at, t))?),
        ValueType::I8 => MetaValue::I8(cursor.u8().map_err(|t| trunc(at, t))? as i8),
        ValueType::U16 => MetaValue::U16(cursor.u16().map_err(|t| trunc(at, t))?),
        ValueType::I16 => MetaValue::I16(cursor.u16().map_err(|t| trunc(at, t))? as i16),
        ValueType::U32 => MetaValue::U32(cursor.u32().map_err(|t| trunc(at, t))?),
        ValueType::I32 => MetaValue::I32(cursor.u32().map_err(|t| trunc(at, t))? as i32),
        ValueType::F32 => MetaValue::F32(cursor.f32().map_err(|t| trunc(at, t))?),
        ValueType::U64 => MetaValue::U64(cursor.u64().map_err(|t| trunc(at, t))?),
        ValueType::I64 => MetaValue::I64(cursor.i64().map_err(|t| trunc(at, t))?),
        ValueType::F64 => MetaValue::F64(cursor.f64().map_err(|t| trunc(at, t))?),
        // Zero is false; every other bit pattern is true. Refusing 2 would
        // reject a file llama.cpp accepts.
        ValueType::Bool => MetaValue::Bool(cursor.u8().map_err(|t| trunc(at, t))? != 0),
        ValueType::String => decode_string(cursor)?,
        ValueType::Array => {
            let (elem, count) = read_array_prefix(cursor)?;
            // Bound the declared count by what the file could possibly
            // contain: every element occupies at least one byte, so a count
            // exceeding the bytes remaining describes a file that cannot
            // exist.
            //
            // What this actually buys, stated accurately — an earlier draft
            // of this comment claimed it prevents an allocation-driven
            // abort, and a sabotage showed otherwise. For an absurd count
            // like 2^40, `try_reserve` below already refuses, so removing
            // this check still yields an error rather than a crash. The
            // real gap is the MIDDLE of the range: a count of ten million
            // against a one-kilobyte file passes `try_reserve` happily,
            // allocates hundreds of megabytes, and only then fails on
            // truncation partway through the loop. This bound refuses that
            // before a byte is allocated, and names the invariant actually
            // violated instead of the allocator's capacity.
            if count > cursor.remaining() {
                return Err(GgufError::Malformed {
                    stage: Stage::Metadata,
                    offset: at,
                    detail: format!(
                        "array declares {count} elements but only {} bytes remain",
                        cursor.remaining()
                    ),
                });
            }
            let n = usize::try_from(count).map_err(|_| GgufError::Malformed {
                stage: Stage::Metadata,
                offset: at,
                detail: format!("array of {count} elements cannot be held on this platform"),
            })?;
            let mut items = Vec::new();
            // Reserve the whole thing, not a token 1024. A partial
            // reservation protects only the first 1024 elements: every push
            // after that grows the Vec through `Vec::push`, which is NOT
            // fallible — on allocation failure it aborts the process rather
            // than returning an error this function could report. For the
            // 514,906-element `tokenizer.ggml.merges` this crate exists to
            // handle, a 1024-element guard protects nothing at all.
            items.try_reserve(n).map_err(|_| GgufError::Malformed {
                stage: Stage::Metadata,
                offset: at,
                detail: format!("cannot allocate {n} elements"),
            })?;
            for _ in 0..count {
                items.push(decode_at_depth(cursor, elem, depth + 1)?);
            }
            MetaValue::Array(items)
        }
    };
    Ok(v)
}

/// Decode a length-prefixed string, **byte-exactly**.
///
/// Valid UTF-8 becomes [`MetaValue::String`]; anything else becomes
/// [`MetaValue::Bytes`]. Never `from_utf8_lossy`: substituting U+FFFD for a
/// byte the tokenizer will later look for produces a prompt that reads
/// correctly and tokenizes differently, with no error at any layer (R3).
/// A trailing NUL is data, not a terminator — GGUF strings are
/// length-prefixed and carry no terminator.
fn decode_string(cursor: &mut Cursor<'_>) -> Result<MetaValue, GgufError> {
    let at = cursor.pos();
    let len = cursor.u64().map_err(|t| trunc(at, t))?;
    let at = cursor.pos();
    let raw = cursor.take(len).map_err(|t| trunc(at, t))?;
    Ok(match core::str::from_utf8(raw) {
        Ok(s) => MetaValue::String(s.to_string()),
        Err(_) => MetaValue::Bytes(raw.to_vec()),
    })
}

/// Read an array's element type and count.
fn read_array_prefix(cursor: &mut Cursor<'_>) -> Result<(ValueType, u64), GgufError> {
    let at = cursor.pos();
    let code = cursor.u32().map_err(|t| trunc(at, t))?;
    let elem = ValueType::from_code(code).ok_or_else(|| GgufError::Malformed {
        stage: Stage::Metadata,
        offset: at,
        detail: format!("array declares unknown element type {code}"),
    })?;
    let at = cursor.pos();
    let count = cursor.u64().map_err(|t| trunc(at, t))?;
    Ok((elem, count))
}

/// Wrap a cursor truncation with the metadata stage and an offset.
fn trunc(offset: u64, t: crate::cursor::Truncated) -> GgufError {
    GgufError::Truncated {
        stage: Stage::Metadata,
        offset,
        needed: t.needed,
        available: t.available,
    }
}
```

Add `pub mod value;` to `lib.rs` and re-export `pub use value::ValueType;`.

- [ ] **Step 4: Run and watch it pass**

```bash
cargo test -p mlmf-gguf
cargo clippy -p mlmf-gguf --all-targets -- -D warnings
```

- [ ] **Step 5: Prove the tests can fail (AD-2)** — four sabotages

1. Change `ValueType::Bool`'s width from 1 to 4. `every_code_round_trips_and_reports_its_width` must go red. **Also note which other tests fail** — if `skipping_a_string_lands_exactly_after_it` stays green, that confirms the width error only manifests through a sequence of keys, which is why Task 6's round-trip test matters.
2. Change `decode_string` to use `String::from_utf8_lossy(raw).into_owned()` wrapped in `MetaValue::String`. `decoding_preserves_bytes_exactly_including_invalid_utf8` must go red. **This is the single most important sabotage in the crate**: no file in the 29-file corpus can produce this failure, so this authored case is the only thing standing between the crate and a silent tokenizer mismatch.
3. Add `.trim_end_matches('\0')` to the decoded string. `a_trailing_nul_is_part_of_the_string_not_a_terminator` must go red. Same reasoning: zero corpus files have one.
4. In `skip_value`'s `Array` arm, replace the fixed-width fast path with the element-wise loop. **All tests must stay green** — the two paths are semantically identical, and that is the point: the fast path is an optimization whose correctness is pinned by the sentinel tests either way. Restore.

- [ ] **Step 5b: Prove the three adversarial guards (AD-2)**

These defend against input nobody publishes, so nothing but an authored control can exercise them.

5. **Remove the bytes-remaining bound** from `decode_value`'s `Array` arm. `an_array_claiming_more_elements_than_the_file_has_bytes_is_refused` must go red — and note *how*: without the bound the call reaches `try_reserve(1_099_511_627_776)`, so the failure is either a `Malformed` for a different reason or an allocation failure, not the specific "cannot exist" complaint. Record which. Restore.

6. **Set `MAX_ARRAY_DEPTH` to `u32::MAX`.** `nesting_deeper_than_the_limit_is_refused_rather_than_overflowing_the_stack` must go red on its `matches!` assertion: with no effective bound, 200 levels decode successfully and no `Malformed` is returned. That proves the bound is what causes the refusal. Restore.

   **This does not demonstrate the stack overflow itself, and it should not try to.** Reaching an actual overflow needs on the order of 100,000 levels — a 1.2 MB fixture — and a test that crashes the process is not a test. The hazard is the *reason* for the bound; the control's job is to show the bound is load-bearing, which a 200-level fixture does at 2.4 kB.

7. **Restore `try_reserve(n.min(1024))`** in place of `try_reserve(n)`. **Every test stays green** — that is the finding, not a failure. A 1024-element reservation protects only the first 1024 pushes; every push after that grows the `Vec` through `Vec::push`, which aborts on allocation failure rather than returning an error. The guard is decorative for exactly the 514,906-element arrays this crate exists to read, and no test can show that, because the failure it fails to prevent is a process abort. Record it as a case where the suite cannot distinguish a real guard from a decorative one, and restore.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/mlmf-gguf
git commit -m "feat(gguf): value types, byte-exact decoding, and skipping without decode"
```

---

## Task 6: The lazy index and `MetadataSource`

**Files:**
- Create: `crates/mlmf-gguf/src/metadata.rs`
- Modify: `crates/mlmf-gguf/src/lib.rs`

**Interfaces:**
- Consumes: `Cursor`, `parse_header`, `Header`, `ValueType`, `skip_value`, `decode_value`, `GgufError`, `Stage`.
- Produces:
  ```rust
  pub struct GgufMetadata<'a> { /* private */ }
  impl<'a> GgufMetadata<'a> {
      pub fn parse(bytes: &'a [u8], origin: &str) -> Result<(Self, Report), GgufError>;
      pub fn header(&self) -> &Header;
      pub fn alignment(&self) -> u64;
      pub fn kv_end(&self) -> u64;
  }
  impl MetadataSource for GgufMetadata<'_> { /* get, keys, declaration, array_len, array_get */ }
  ```

- [ ] **Step 1: Write the failing tests**

Create `crates/mlmf-gguf/src/metadata.rs` with this test module; add `pub mod metadata;` to `lib.rs` in the same step.

```rust
//! The key-value block, indexed at open and decoded on demand.

#[cfg(test)]
mod tests {
    use super::*;
    use mlmf_core::{Declaration, MetadataSource, MetaValue};

    /// Build a minimal GGUF: header plus the given KV pairs, no tensors.
    ///
    /// Each pair is (key, value type code, already-encoded value bytes).
    fn gguf(kvs: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes()); // no tensors
        b.extend_from_slice(&(kvs.len() as i64).to_le_bytes());
        for (k, ty, v) in kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(v);
        }
        b
    }

    fn s(v: &str) -> Vec<u8> {
        let mut b = (v.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(v.as_bytes());
        b
    }

    fn str_array(items: &[&str]) -> Vec<u8> {
        let mut b = 8u32.to_le_bytes().to_vec(); // String elements
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for i in items {
            b.extend_from_slice(&s(i));
        }
        b
    }

    #[test]
    fn reads_a_key_without_decoding_the_rest() {
        let bytes = gguf(&[
            ("general.architecture", 8, s("llama")),
            ("tokenizer.ggml.tokens", 9, str_array(&["a", "b", "c"])),
        ]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("parses");
        assert!(report.is_empty());
        assert_eq!(
            m.get("general.architecture"),
            Some(&MetaValue::String("llama".into()))
        );
        assert_eq!(m.keys().len(), 2);
    }

    #[test]
    fn an_unknown_value_type_costs_that_key_and_not_the_parse() {
        // R1's shape, at the metadata layer. A key this build cannot decode
        // must not stop the keys around it from being readable — and it must
        // be reported rather than dropped.
        //
        // Value type 13 does not exist, so the parse cannot know where the
        // value ends and must stop indexing; the keys BEFORE it stay
        // readable and the failure is reported.
        let bytes = gguf(&[
            ("first", 8, s("kept")),
            ("broken", 13, s("unreachable")),
            ("third", 8, s("lost")),
        ]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("does not fail the open");
        assert_eq!(m.get("first"), Some(&MetaValue::String("kept".into())));
        assert!(matches!(m.declaration("broken"), Declaration::Unreadable(_)));
        // `third` reads as Absent, and that is the honest limit of what
        // this API can say: an unknown width means the parse never found
        // out whether `third` exists. It is NOT a fact about the file.
        //
        // So the index must announce that it stopped, or a caller reading
        // `Absent` concludes "this model declares no third key" from a scan
        // that never reached it — a negative finding drawn from a truncated
        // walk. `index_complete()` is what separates the two.
        assert!(matches!(m.declaration("third"), Declaration::Absent));
        assert!(
            !m.index_complete(),
            "a walk that stopped early must say so, or Absent lies"
        );
        assert!(!report.is_empty(), "the unknown type must be reported");
    }

    #[test]
    fn two_files_each_report_their_own_unreadable_key() {
        // A draft of this crate held the Unreadable placeholder in a process
        // -wide `static OnceLock`. Within one file that is invisible — the
        // index stops at the first unknown type, so there is only ever one.
        // Across two files the first one parsed wins forever, and the second
        // file's operator is shown a key name from someone else's model.
        let a = gguf(&[("alpha", 13, s("x"))]);
        let b = gguf(&[("beta", 13, s("x"))]);
        let (ma, _) = GgufMetadata::parse(&a, "a.gguf").unwrap();
        let (mb, _) = GgufMetadata::parse(&b, "b.gguf").unwrap();

        for (m, want_key, want_origin) in [(&ma, "alpha", "a.gguf"), (&mb, "beta", "b.gguf")] {
            match m.declaration(want_key) {
                Declaration::Unreadable(u) => {
                    assert_eq!(u.origin, want_origin);
                    match &u.kind {
                        mlmf_core::UnrecognizedKind::MetadataKey { key, .. } => {
                            assert_eq!(key, want_key, "each file must name its own key");
                        }
                        other => panic!("wrong kind: {other:?}"),
                    }
                }
                other => panic!("{want_key}: expected Unreadable, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_clean_parse_reports_a_complete_index() {
        // The other half of the pair. Without this, `index_complete` could
        // return `false` unconditionally and the test above would still
        // pass — a flag that is always pessimistic is as useless as one
        // that is always optimistic, and only asserting both directions
        // distinguishes them.
        let bytes = gguf(&[("a", 8, s("x")), ("b", 8, s("y"))]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        assert!(m.index_complete());
        assert!(report.is_empty());
        assert_eq!(m.keys().len(), 2);
    }

    #[test]
    fn a_duplicate_key_keeps_the_first_and_reports_the_second() {
        // GGUF does not forbid it and llama.cpp does not check. Silently
        // taking the last would make the file's meaning depend on parse
        // order; taking the first and reporting is deterministic and loud.
        let bytes = gguf(&[("k", 8, s("first")), ("k", 8, s("second"))]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("parses");
        assert_eq!(m.get("k"), Some(&MetaValue::String("first".into())));
        assert!(!report.is_empty(), "the duplicate must be reported");
    }

    #[test]
    fn alignment_defaults_to_32_and_is_overridden_only_by_a_u32_power_of_two() {
        let plain = gguf(&[("general.architecture", 8, s("llama"))]);
        let (m, _) = GgufMetadata::parse(&plain, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32, "no key declared: the documented default");

        let declared = gguf(&[("general.alignment", 4, 64u32.to_le_bytes().to_vec())]);
        let (m, _) = GgufMetadata::parse(&declared, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 64);

        // Not a power of two: llama.cpp refuses (gguf.cpp:623). Report and
        // fall back rather than fail the open — this is metadata, and R1
        // says metadata reading survives.
        let bad = gguf(&[("general.alignment", 4, 63u32.to_le_bytes().to_vec())]);
        let (m, report) = GgufMetadata::parse(&bad, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32);
        assert!(!report.is_empty());

        // Wrong type: llama.cpp requires UINT32 (gguf.cpp:614).
        let wrong = gguf(&[("general.alignment", 10, 64u64.to_le_bytes().to_vec())]);
        let (m, report) = GgufMetadata::parse(&wrong, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32);
        assert!(!report.is_empty());
    }

    #[test]
    fn declaration_reports_absent_for_a_key_the_file_does_not_have() {
        let bytes = gguf(&[("k", 8, s("v"))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        assert!(matches!(m.declaration("k"), Declaration::Declared(_)));
        assert!(matches!(m.declaration("absent"), Declaration::Absent));
    }

    #[test]
    fn a_decoded_value_is_returned_by_reference_and_decoded_once() {
        // The lazy index must still satisfy `get(&self) -> Option<&MetaValue>`,
        // which means the decoded value has to be cached. Two calls must
        // return the same address, or the cache is not doing its job.
        let bytes = gguf(&[("k", 8, s("v"))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let a = m.get("k").unwrap() as *const MetaValue;
        let b = m.get("k").unwrap() as *const MetaValue;
        assert_eq!(a, b, "the value must be decoded once and cached");
    }

    #[test]
    fn the_kv_end_is_where_the_tensor_directory_begins() {
        // The next plan needs this, and it is only knowable here.
        let bytes = gguf(&[("k", 8, s("v"))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        // 24 header + 8 keylen + 1 key + 4 type + 8 strlen + 1 str
        assert_eq!(m.kv_end(), 46);
    }

    #[test]
    fn a_truncated_kv_block_reports_the_metadata_stage_not_the_header() {
        let mut bytes = gguf(&[("k", 8, s("v"))]);
        bytes.truncate(bytes.len() - 1);
        let err = GgufMetadata::parse(&bytes, "t.gguf").unwrap_err();
        assert!(matches!(
            err,
            crate::error::GgufError::Truncated {
                stage: crate::error::Stage::Metadata,
                ..
            }
        ));
    }
}
```

- [ ] **Step 2: Run and watch it fail**

```bash
cargo test -p mlmf-gguf --lib metadata
```

Expected: compile error, `cannot find type GgufMetadata`.

- [ ] **Step 3: Implement**

**First, add `sync` to `crates/mlmf-gguf/tests/allowed-std.list`** — insert it in sorted position, replacing the placeholder note Task 3 left. This is the commit that introduces `OnceLock`, so this is the commit where the permission is earned. Prove it: remove the entry and confirm the shared purity gate goes red naming `mlmf-gguf` and `std::sync`, then restore. A permission added alongside the code that needs it gets a red-then-green cycle; one added in advance never does.

```rust
use std::sync::OnceLock;

use mlmf_core::{Declaration, MetaValue, MetadataSource, Report, Unrecognized, UnrecognizedKind};

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};
use crate::header::{parse_header, Header};
use crate::value::{decode_value, skip_value, ValueType};

/// GGUF's documented default when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;

/// One indexed key: where its value is, and its value once decoded.
#[derive(Debug)]
struct Entry {
    key: String,
    ty: ValueType,
    /// Offset of the value's first byte.
    start: u64,
    /// Set when the value could not be indexed — an unknown type code.
    ///
    /// Owned per entry rather than shared, because [`Declaration::Unreadable`]
    /// borrows and the entry it names must be *this* key's. A shared
    /// placeholder initialised once would report the first unreadable key's
    /// name for every subsequent one.
    unreadable: Option<Unrecognized>,
    /// Decoded on first access. `OnceLock` rather than `OnceCell` so the
    /// metadata stays `Sync` — a consumer loading tensors in parallel will
    /// read metadata from several threads.
    value: OnceLock<MetaValue>,
}

/// A GGUF file's metadata, indexed but not decoded.
///
/// Opening indexes every key into `(key, type, offset)` and decodes
/// nothing. A value is decoded on first access and cached. The largest file
/// in the reference corpus declares 777,056 strings across 42 keys, so
/// eager decoding would cost roughly 26 MB of allocations to answer a
/// question about one of them.
#[derive(Debug)]
pub struct GgufMetadata<'a> {
    bytes: &'a [u8],
    header: Header,
    entries: Vec<Entry>,
    alignment: u64,
    kv_end: u64,
    /// False when the index stopped before every declared key was seen.
    index_complete: bool,
}

impl<'a> GgufMetadata<'a> {
    /// Parse the header and index the key-value block.
    ///
    /// Returns the metadata **and** a [`Report`] of everything the parse did
    /// not understand. The report is not optional: a caller cannot obtain
    /// the content without also receiving the account of what was skipped.
    ///
    /// `origin` names the artifact in report entries — a file name, a URL,
    /// whatever the caller can show an operator.
    ///
    /// # Errors
    ///
    /// [`GgufError::NotGguf`] if the magic is wrong, which is a different
    /// fact from a malformed GGUF (R7). [`GgufError::Truncated`] or
    /// [`GgufError::Malformed`] with `Stage::Metadata` if the KV block ends
    /// early or contains something structurally impossible.
    pub fn parse(bytes: &'a [u8], origin: &str) -> Result<(Self, Report), GgufError> {
        let mut cursor = Cursor::new(bytes);
        let header = parse_header(&mut cursor)?;
        let mut report = Report::new();
        let mut index_complete = true;
        // Deliberately NOT `Vec::with_capacity(header.kv_count)`. The count
        // is a declared number this build has only checked for negativity,
        // so `i64::MAX` reaches here intact; preallocating from it panics on
        // capacity overflow before any truncation check runs. Growing as we
        // go costs nothing at the 42-key scale real files use.
        let mut entries: Vec<Entry> = Vec::new();

        for _ in 0..header.kv_count {
            let key = read_key(&mut cursor)?;
            let at = cursor.pos();
            let code = cursor.u32().map_err(|t| GgufError::Truncated {
                stage: Stage::Metadata,
                offset: at,
                needed: t.needed,
                available: t.available,
            })?;

            let Some(ty) = ValueType::from_code(code) else {
                // An unknown value type has an unknown width, so the parse
                // cannot find the next key. Everything indexed so far stays
                // readable — which is R1's guarantee applied within the
                // metadata stage itself — and the failure is reported rather
                // than silently truncating the key list.
                let complaint = Unrecognized {
                    kind: UnrecognizedKind::MetadataKey {
                        key: key.clone(),
                        value: MetaValue::U32(code),
                    },
                    origin: origin.to_string(),
                };
                report.push(complaint.clone());
                entries.push(Entry {
                    key,
                    ty: ValueType::U8, // never read; `unreadable` gates access
                    start: 0,
                    unreadable: Some(complaint),
                    value: OnceLock::new(),
                });
                index_complete = false;
                break;
            };

            let start = cursor.pos();
            skip_value(&mut cursor, ty)?;

            if entries.iter().any(|e| e.key == key) {
                // Deterministic and loud: first wins, second reported. Taking
                // the last would make the file's meaning depend on parse order.
                report.push(Unrecognized {
                    kind: UnrecognizedKind::MetadataKey {
                        key: key.clone(),
                        value: MetaValue::String("duplicate key; first occurrence kept".into()),
                    },
                    origin: origin.to_string(),
                });
                continue;
            }

            entries.push(Entry {
                key,
                ty,
                start,
                unreadable: None,
                value: OnceLock::new(),
            });
        }

        let kv_end = cursor.pos();
        let mut me = Self {
            bytes,
            header,
            entries,
            alignment: DEFAULT_ALIGNMENT,
            kv_end,
            index_complete,
        };
        me.alignment = me.resolve_alignment(origin, &mut report);
        Ok((me, report))
    }

    /// The file's header.
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Whether every key the header declared was indexed.
    ///
    /// `false` means the walk stopped early: an unknown value type has an
    /// unknown width, so the parse cannot find the key that follows it.
    ///
    /// **This changes what [`mlmf_core::Declaration::Absent`] means, and a
    /// caller that ignores it will draw a false conclusion.** With a
    /// complete index, `Absent` is a fact about the file — the key is not
    /// declared. With an incomplete one it means only *not found in the part
    /// that could be read*, and the key may be sitting immediately past the
    /// point where the walk stopped. Those are different claims:
    /// "this model declares no chat template" versus "we could not get far
    /// enough to tell". A count or a `keys()` listing taken from an
    /// incomplete index can support a positive finding — this key IS here —
    /// and never a negative one.
    #[must_use]
    pub fn index_complete(&self) -> bool {
        self.index_complete
    }

    /// Offset just past the key-value block.
    ///
    /// The tensor directory begins here. Exposed because only this stage
    /// knows it, and the tensor stage needs it.
    #[must_use]
    pub fn kv_end(&self) -> u64 {
        self.kv_end
    }

    /// Alignment for the tensor data region.
    ///
    /// `general.alignment` when the file declares a valid one, otherwise
    /// GGUF's documented default of 32.
    ///
    /// **This is the effective value, and it does not say where it came
    /// from.** A file that declares 32 and a file that declares nothing
    /// both answer 32 here. That is the right shape for a caller who needs
    /// a number, but a caller who needs the *fact* must ask
    /// `declaration("general.alignment")`, which separates declared from
    /// absent from undecodable. Spec §5 says absent never means a default;
    /// this method returns a default precisely because alignment has a
    /// documented one, and the raw fact stays reachable beside it. An invalid declaration is reported
    /// and falls back rather than failing the open — this is metadata, and
    /// R1 says reading metadata survives.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    fn resolve_alignment(&self, origin: &str, report: &mut Report) -> u64 {
        let Some(v) = self.get("general.alignment") else {
            return DEFAULT_ALIGNMENT;
        };
        let complain = |report: &mut Report, why: &str| {
            report.push(Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "general.alignment".to_string(),
                    value: MetaValue::String(why.to_string()),
                },
                origin: origin.to_string(),
            });
        };
        // llama.cpp requires UINT32 specifically (gguf.cpp:614).
        let MetaValue::U32(a) = v else {
            complain(report, "must be UINT32; using the default of 32");
            return DEFAULT_ALIGNMENT;
        };
        let a = u64::from(*a);
        // And a power of two (gguf.cpp:623).
        if a == 0 || !a.is_power_of_two() {
            complain(report, "must be a power of two; using the default of 32");
            return DEFAULT_ALIGNMENT;
        }
        a
    }

    fn entry(&self, key: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.key == key)
    }

    /// Decode an entry's value, caching it.
    fn value_of(&self, e: &Entry) -> Option<&MetaValue> {
        if e.unreadable.is_some() {
            return None;
        }
        Some(e.value.get_or_init(|| {
            let mut c = Cursor::new(self.bytes);
            // Both operations succeeded during indexing, so neither can fail
            // here; a failure would mean the byte slice changed underneath us,
            // which is impossible for a shared borrow.
            c.seek(e.start).expect("indexed offset is in range");
            decode_value(&mut c, e.ty).expect("indexed value decoded during parse")
        }))
    }
}

/// Read one length-prefixed key.
fn read_key(cursor: &mut Cursor<'_>) -> Result<String, GgufError> {
    let at = cursor.pos();
    let len = cursor.u64().map_err(|t| GgufError::Truncated {
        stage: Stage::Metadata,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;
    let at = cursor.pos();
    let raw = cursor.take(len).map_err(|t| GgufError::Truncated {
        stage: Stage::Metadata,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;
    // A key is a lookup token, so it must be UTF-8 to be usable as one.
    // Values are different: a non-UTF-8 *value* survives as MetaValue::Bytes.
    core::str::from_utf8(raw)
        .map(str::to_string)
        .map_err(|e| GgufError::Malformed {
            stage: Stage::Metadata,
            offset: at,
            detail: format!("key is not valid UTF-8: {e}"),
        })
}

impl MetadataSource for GgufMetadata<'_> {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entry(key).and_then(|e| self.value_of(e))
    }

    fn keys(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.key.as_str()).collect()
    }

    fn declaration(&self, key: &str) -> Declaration<'_> {
        match self.entry(key) {
            None => Declaration::Absent,
            // Indexed but not decodable: the key is in the file and its value
            // is not readable. Exactly the state R2 exists to separate from
            // Absent, and the entry it borrows names *this* key.
            Some(e) if e.unreadable.is_some() => {
                Declaration::Unreadable(e.unreadable.as_ref().expect("just checked"))
            }
            Some(e) => match self.value_of(e) {
                Some(v) => Declaration::Declared(v),
                None => Declaration::Absent,
            },
        }
    }
}
```

Add `pub mod metadata;` to `lib.rs` and `pub use metadata::GgufMetadata;`.

- [ ] **Step 4: Run and watch it pass**

```bash
cargo test -p mlmf-gguf
cargo clippy -p mlmf-gguf --all-targets -- -D warnings
```

- [ ] **Step 5: Prove the tests can fail (AD-2)** — four sabotages

1. In `parse`, replace the duplicate check with unconditional `entries.push(...)`. `a_duplicate_key_keeps_the_first_and_reports_the_second` must go red — and note **which** assertion fails: `get` now returns the *last*, so the value assertion fails before the report assertion. Restore.
2. In `resolve_alignment`, remove the `is_power_of_two` check. The third block of `alignment_defaults_to_32_and_is_overridden_only_by_a_u32_power_of_two` must go red with `63`. Restore.
3. In `value_of`, replace `get_or_init` with a fresh decode each call, returning a leaked reference is impossible — so instead change `Entry::value` to be re-initialised by calling `OnceLock::new()` inside `value_of`. `a_decoded_value_is_returned_by_reference_and_decoded_once` must go red on the pointer comparison. Restore.
4. Replace `Entry::unreadable` with a process-wide `static UNREADABLE: OnceLock<Unrecognized>` initialised on first use — the shape a first draft of this plan actually specified. `an_unknown_value_type_costs_that_key_and_not_the_parse` stays green — it parses one file, so the shared placeholder is correct for it — and `two_files_each_report_their_own_unreadable_key` goes red showing `b.gguf`'s key reported as `alpha`. Both named, so neither side of the claim is a category. Restore. This is the sabotage worth doing slowly: the defect is invisible to any single-file test, which is why the test parses two.
5. In `parse`, change the unknown-type arm from `break` to `continue`. `an_unknown_value_type_costs_that_key_and_not_the_parse` must go red — with an unknown width the cursor is not positioned at the next key, so `third` decodes as garbage or the parse errors. **Record what actually happens**: this is the difference between stopping honestly and desynchronising silently.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/mlmf-gguf
git commit -m "feat(gguf): lazy KV index and the MetadataSource seam"
```

---

## Task 7: Array access without materializing

R5, and the reason D1 exists. `array_get` must decode one element out of a 500,000-element array without decoding the other 499,999.

**Files:**
- Modify: `crates/mlmf-gguf/src/metadata.rs`

**Interfaces:**
- Consumes: `Entry`, `Cursor`, `ValueType`, `skip_value`, `decode_value`.
- Produces: overrides of `MetadataSource::array_len` and `MetadataSource::array_get` on `GgufMetadata`.

- [ ] **Step 1: Write the failing tests**

Add to `metadata.rs`'s `mod tests`:

```rust
    #[test]
    fn array_access_does_not_decode_the_array() {
        // The R5 case. If this went through `get`, it would decode all
        // 100,000 elements to return one.
        let items: Vec<String> = (0..100_000).map(|i| format!("tok{i}")).collect();
        let refs: Vec<&str> = items.iter().map(String::as_str).collect();
        let bytes = gguf(&[("tokenizer.ggml.tokens", 9, str_array(&refs))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();

        assert_eq!(m.array_len("tokenizer.ggml.tokens"), Some(100_000));
        assert_eq!(
            m.array_get("tokenizer.ggml.tokens", 0),
            Some(MetaValue::String("tok0".into()))
        );
        assert_eq!(
            m.array_get("tokenizer.ggml.tokens", 99_999),
            Some(MetaValue::String("tok99999".into()))
        );
        assert_eq!(m.array_get("tokenizer.ggml.tokens", 100_000), None);

        // And the proof it stayed lazy: nothing decoded the whole array, so
        // the entry's cache is still empty. If `array_get` had gone through
        // `get`, this would hold a 100,000-element MetaValue::Array.
        let e = m.entry("tokenizer.ggml.tokens").unwrap();
        assert!(
            e.value.get().is_none(),
            "array access must not populate the whole-value cache"
        );
    }

    #[test]
    fn a_fixed_width_array_is_indexed_by_arithmetic_not_by_walking() {
        let mut v = 4u32.to_le_bytes().to_vec(); // U32 elements
        v.extend_from_slice(&5u64.to_le_bytes());
        for i in 0..5u32 {
            v.extend_from_slice(&(i * 11).to_le_bytes());
        }
        let bytes = gguf(&[("nums", 9, v)]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        assert_eq!(m.array_len("nums"), Some(5));
        assert_eq!(m.array_get("nums", 3), Some(MetaValue::U32(33)));
        assert_eq!(m.array_get("nums", 5), None);
    }

    #[test]
    fn array_accessors_agree_with_the_decoded_value() {
        // The two paths must not drift. Whatever `array_get(k, i)` returns
        // must equal `get(k)`'s i-th element — otherwise a consumer sees a
        // different vocabulary depending on which accessor it used.
        let bytes = gguf(&[("toks", 9, str_array(&["alpha", "beta", "gamma"]))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let MetaValue::Array(all) = m.get("toks").unwrap().clone() else {
            panic!("expected an array");
        };
        for (i, want) in all.iter().enumerate() {
            assert_eq!(m.array_get("toks", i as u64).as_ref(), Some(want), "index {i}");
        }
        assert_eq!(m.array_len("toks"), Some(all.len() as u64));
    }

    #[test]
    fn a_scalar_has_no_array_length_and_no_elements() {
        let bytes = gguf(&[("k", 8, s("scalar"))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        assert_eq!(m.array_len("k"), None);
        assert_eq!(m.array_get("k", 0), None);
    }
```

- [ ] **Step 2: Run and watch it fail**

```bash
cargo test -p mlmf-gguf --lib array
```

Expected: the tests compile (the default trait impls exist from Task 1) but `array_access_does_not_decode_the_array` **fails on the cache assertion**, because the default `array_len`/`array_get` go through `get`. That is the correct red: the behaviour is right and the cost is wrong.

- [ ] **Step 3: Implement the overrides**

Add to `impl MetadataSource for GgufMetadata<'_>`:

```rust
    fn array_len(&self, key: &str) -> Option<u64> {
        let e = self.entry(key)?;
        if e.ty != ValueType::Array || e.unreadable.is_some() {
            return None;
        }
        let mut c = Cursor::new(self.bytes);
        c.seek(e.start).ok()?;
        // Element type then count; both were validated during indexing.
        c.u32().ok()?;
        c.u64().ok()
    }

    fn array_get(&self, key: &str, index: u64) -> Option<MetaValue> {
        let e = self.entry(key)?;
        if e.ty != ValueType::Array || e.unreadable.is_some() {
            return None;
        }
        let mut c = Cursor::new(self.bytes);
        c.seek(e.start).ok()?;
        let elem = ValueType::from_code(c.u32().ok()?)?;
        let count = c.u64().ok()?;
        if index >= count {
            return None;
        }
        match elem.fixed_width() {
            // Constant time: the element's offset is arithmetic.
            Some(w) => {
                let skip = index.checked_mul(w)?;
                c.seek(c.pos().checked_add(skip)?).ok()?;
            }
            // Variable width: walk, but skip rather than decode. Still O(n)
            // in the index, and O(1) in allocations — which is the cost that
            // actually hurt.
            None => {
                for _ in 0..index {
                    skip_value(&mut c, elem).ok()?;
                }
            }
        }
        decode_value(&mut c, elem).ok()
    }
```

- [ ] **Step 4: Run and watch it pass**

```bash
cargo test -p mlmf-gguf
cargo clippy -p mlmf-gguf --all-targets -- -D warnings
```

- [ ] **Step 5: Prove the tests can fail (AD-2)**

1. Delete both overrides so the defaults apply again. `array_access_does_not_decode_the_array` must go red **on the cache assertion and nowhere else** — the values are still correct, only the cost is wrong. That contrast is the finding. Restore.
2. In `array_get`'s fixed-width arm, drop the `index >= count` check. `a_fixed_width_array_is_indexed_by_arithmetic_not_by_walking` must go red on index 5 — without it, index 5 reads whatever follows the array, which is a plausible-looking `U32` from the next key's bytes. **Record the value you get**; a wrong number that looks like a token id is worse than an error.
3. In `array_get`'s variable-width arm, change `for _ in 0..index` to `for _ in 0..index.saturating_sub(1)`. `array_accessors_agree_with_the_decoded_value` must go red — it compares every index against the decoded array, so an off-by-one cannot hide at any single index.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/mlmf-gguf
git commit -m "feat(gguf): indexed array access without materializing the array"
```

---

## Task 8: The fixture writer and the adversarial cases

The corpus cannot falsify R3. Measured across 29 real files: **zero non-UTF-8 strings, zero trailing-NUL strings, zero files declaring `general.alignment`**. A Hub corpus samples what people publish, which is precisely the population that never exercises the defensive paths. These cases exist or the guarantees are unfalsifiable.

**Files:**
- Create: `crates/mlmf-gguf/tests/fixture.rs` (a GGUF writer, test-only)
- Create: `crates/mlmf-gguf/tests/authored.rs`

**Interfaces:**
- Produces:
  ```rust
  pub struct GgufBuilder { /* private */ }
  impl GgufBuilder {
      pub fn new() -> Self;
      pub fn version(self, v: u32) -> Self;
      pub fn tensor_count(self, n: i64) -> Self;
      pub fn string(self, key: &str, value: &str) -> Self;
      pub fn raw_string(self, key: &str, value: &[u8]) -> Self;
      pub fn u32(self, key: &str, value: u32) -> Self;
      pub fn raw_kv(self, key: &str, type_code: u32, value: Vec<u8>) -> Self;
      pub fn string_array(self, key: &str, items: &[&[u8]]) -> Self;
      pub fn build(self) -> Vec<u8>;
  }
  ```

- [ ] **Step 1: Write the fixture builder**

`crates/mlmf-gguf/tests/fixture.rs`. It is a `mod` included by `authored.rs`, not a test file of its own.

```rust
//! A GGUF writer, for files no one publishes.
//!
//! The corpus contains zero non-UTF-8 strings, zero trailing-NUL strings
//! and zero declared alignments, so the guarantees about those paths are
//! untestable against real files. This builder produces the files that can
//! fail them.
//!
//! Deliberately dumb: it emits exactly what it is told, including things a
//! real writer would refuse. A builder that validated its own output could
//! not produce a malformed fixture.

/// Builds GGUF byte sequences, valid or otherwise.
#[derive(Debug)]
pub struct GgufBuilder {
    version: u32,
    tensor_count: i64,
    kvs: Vec<u8>,
    kv_count: i64,
}

impl GgufBuilder {
    /// A v3 builder with no tensors and no keys.
    pub fn new() -> Self {
        Self {
            version: 3,
            tensor_count: 0,
            kvs: Vec::new(),
            kv_count: 0,
        }
    }

    /// Override the version, including to values this build refuses.
    pub fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    /// Override the declared tensor count without writing tensors.
    pub fn tensor_count(mut self, n: i64) -> Self {
        self.tensor_count = n;
        self
    }

    fn push_str(buf: &mut Vec<u8>, s: &[u8]) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s);
    }

    /// A UTF-8 string value.
    pub fn string(self, key: &str, value: &str) -> Self {
        self.raw_string(key, value.as_bytes())
    }

    /// A string value of arbitrary bytes — including invalid UTF-8.
    pub fn raw_string(mut self, key: &str, value: &[u8]) -> Self {
        Self::push_str(&mut self.kvs, key.as_bytes());
        self.kvs.extend_from_slice(&8u32.to_le_bytes());
        Self::push_str(&mut self.kvs, value);
        self.kv_count += 1;
        self
    }

    /// A `UINT32` value.
    pub fn u32(mut self, key: &str, value: u32) -> Self {
        Self::push_str(&mut self.kvs, key.as_bytes());
        self.kvs.extend_from_slice(&4u32.to_le_bytes());
        self.kvs.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    /// A key with an arbitrary type code and pre-encoded value bytes.
    pub fn raw_kv(mut self, key: &str, type_code: u32, value: Vec<u8>) -> Self {
        Self::push_str(&mut self.kvs, key.as_bytes());
        self.kvs.extend_from_slice(&type_code.to_le_bytes());
        self.kvs.extend_from_slice(&value);
        self.kv_count += 1;
        self
    }

    /// An array of strings, each an arbitrary byte sequence.
    pub fn string_array(mut self, key: &str, items: &[&[u8]]) -> Self {
        Self::push_str(&mut self.kvs, key.as_bytes());
        self.kvs.extend_from_slice(&9u32.to_le_bytes());
        self.kvs.extend_from_slice(&8u32.to_le_bytes()); // String elements
        self.kvs.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for i in items {
            Self::push_str(&mut self.kvs, i);
        }
        self.kv_count += 1;
        self
    }

    /// The bytes.
    pub fn build(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.tensor_count.to_le_bytes());
        out.extend_from_slice(&self.kv_count.to_le_bytes());
        out.extend_from_slice(&self.kvs);
        out
    }
}
```

- [ ] **Step 2: Write the adversarial tests**

`crates/mlmf-gguf/tests/authored.rs`:

```rust
//! Cases no published model provides.
//!
//! Every test here answers "what would this show if the claim were false?"
//! — if the answer is "the same thing", it is not measuring anything.

mod fixture;

use fixture::GgufBuilder;
use mlmf_core::{Declaration, MetaValue, MetadataSource};
use mlmf_gguf::{GgufError, GgufMetadata, Stage};

#[test]
fn a_non_utf8_value_survives_byte_for_byte() {
    // R3. Zero of 29 corpus files can produce this failure, so a regression
    // to `from_utf8_lossy` would be byte-identical on every real file and
    // leave the whole suite green. This is the only thing standing between
    // the crate and a silent tokenizer mismatch.
    let raw: &[u8] = &[0xFF, 0xFE, b'h', b'i', 0x80];
    let bytes = GgufBuilder::new().raw_string("weird", raw).build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    match m.get("weird").unwrap() {
        MetaValue::Bytes(got) => assert_eq!(got.as_slice(), raw),
        other => panic!("expected Bytes, got {other:?}"),
    }
}

#[test]
fn a_trailing_nul_is_kept() {
    // GGUF strings are length-prefixed with no terminator, so a trailing
    // NUL is data. Zero corpus files have one.
    let bytes = GgufBuilder::new().raw_string("t", b"value\0").build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    assert_eq!(
        m.get("t"),
        Some(&MetaValue::String("value\0".into())),
        "the NUL is part of the string"
    );
}

#[test]
fn an_embedded_nul_does_not_truncate() {
    let bytes = GgufBuilder::new().raw_string("t", b"a\0b").build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    assert_eq!(m.get("t"), Some(&MetaValue::String("a\0b".into())));
}

#[test]
fn an_empty_string_is_declared_rather_than_absent() {
    // The distinction R2 exists for, in its most easily-confused form: a
    // key whose value is "" is DECLARED. A consumer may choose to treat it
    // as undeclared — that is their policy — but MLMF must not decide it.
    let bytes = GgufBuilder::new().string("tmpl", "").build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    assert!(matches!(m.declaration("tmpl"), Declaration::Declared(_)));
    assert_eq!(m.get("tmpl"), Some(&MetaValue::String(String::new())));
    assert!(matches!(m.declaration("other"), Declaration::Absent));
}

#[test]
fn a_declared_alignment_is_honoured_and_a_bad_one_is_reported() {
    // Zero corpus files declare general.alignment, so every branch here is
    // unreachable from real data.
    let good = GgufBuilder::new().u32("general.alignment", 64).build();
    let (m, r) = GgufMetadata::parse(&good, "authored").unwrap();
    assert_eq!(m.alignment(), 64);
    assert!(r.is_empty());

    let odd = GgufBuilder::new().u32("general.alignment", 63).build();
    let (m, r) = GgufMetadata::parse(&odd, "authored").unwrap();
    assert_eq!(m.alignment(), 32, "falls back rather than failing the open");
    assert!(!r.is_empty(), "and says so");

    let zero = GgufBuilder::new().u32("general.alignment", 0).build();
    let (m, r) = GgufMetadata::parse(&zero, "authored").unwrap();
    assert_eq!(m.alignment(), 32);
    assert!(!r.is_empty());
}

#[test]
fn a_key_that_is_not_utf8_is_malformed_rather_than_lossy() {
    // A key is a lookup token. A lossy key silently becomes unfindable —
    // the caller asks for the name they saw in a hex dump and gets Absent.
    let mut bytes = GgufBuilder::new().string("ok", "v").build();
    // Overwrite the first key's bytes with invalid UTF-8, same length.
    let key_at = 24 + 8;
    bytes[key_at] = 0xFF;
    bytes[key_at + 1] = 0xFE;
    match GgufMetadata::parse(&bytes, "authored").unwrap_err() {
        GgufError::Malformed { stage, .. } => assert_eq!(stage, Stage::Metadata),
        other => panic!("expected Malformed, got {other:?}"),
    }
}

#[test]
fn an_unknown_value_type_is_reported_and_earlier_keys_survive() {
    // R1 within the metadata stage.
    let bytes = GgufBuilder::new()
        .string("first", "kept")
        .raw_kv("odd", 42, vec![0; 4])
        .build();
    let (m, report) = GgufMetadata::parse(&bytes, "authored").expect("open survives");
    assert_eq!(m.get("first"), Some(&MetaValue::String("kept".into())));
    assert!(matches!(m.declaration("odd"), Declaration::Unreadable(_)));
    assert_eq!(report.entries().len(), 1);
}

#[test]
fn a_declared_key_count_larger_than_the_file_is_truncated_not_a_hang() {
    // Eight bytes of header claiming a million keys must cost a bounded
    // amount of work, not a million allocations.
    let mut bytes = GgufBuilder::new().string("only", "one").build();
    bytes[16..24].copy_from_slice(&1_000_000i64.to_le_bytes());
    assert!(matches!(
        GgufMetadata::parse(&bytes, "authored").unwrap_err(),
        GgufError::Truncated { stage: Stage::Metadata, .. }
    ));
}

#[test]
fn a_string_declaring_a_length_beyond_the_file_is_truncated_not_an_allocation() {
    let mut v = u64::MAX.to_le_bytes().to_vec();
    v.truncate(8);
    let bytes = GgufBuilder::new().raw_kv("huge", 8, v).build();
    assert!(matches!(
        GgufMetadata::parse(&bytes, "authored").unwrap_err(),
        GgufError::Truncated { .. }
    ));
}

#[test]
fn an_array_element_that_is_not_utf8_survives_indexed_access() {
    // R3 through R5's path, which is a different code path from `get`.
    let items: Vec<&[u8]> = vec![b"fine", &[0xFF, 0x00], b"also fine"];
    let bytes = GgufBuilder::new().string_array("toks", &items).build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    assert_eq!(m.array_len("toks"), Some(3));
    assert_eq!(
        m.array_get("toks", 1),
        Some(MetaValue::Bytes(vec![0xFF, 0x00]))
    );
}

#[test]
fn a_zero_length_array_has_a_length_and_no_elements() {
    let bytes = GgufBuilder::new().string_array("empty", &[]).build();
    let (m, _) = GgufMetadata::parse(&bytes, "authored").unwrap();
    // Some(0), not None: the array is declared and it is empty. None would
    // say "not an array", which is a different fact.
    assert_eq!(m.array_len("empty"), Some(0));
    assert_eq!(m.array_get("empty", 0), None);
}

#[test]
fn v1_and_a_future_version_are_both_refused_by_number() {
    for v in [1u32, 4, 99] {
        let bytes = GgufBuilder::new().version(v).build();
        match GgufMetadata::parse(&bytes, "authored").unwrap_err() {
            GgufError::UnsupportedVersion { version } => assert_eq!(version, v),
            other => panic!("version {v}: expected UnsupportedVersion, got {other:?}"),
        }
    }
}

#[test]
fn a_byte_swapped_file_is_named_as_such() {
    let bytes = GgufBuilder::new().version(0x0300_0000).build();
    assert!(matches!(
        GgufMetadata::parse(&bytes, "authored").unwrap_err(),
        GgufError::ByteSwapped { .. }
    ));
}
```

- [ ] **Step 3: Run**

```bash
cargo test -p mlmf-gguf --test authored
```

Expected: PASS.

- [ ] **Step 4: Prove the two that matter can fail (AD-2)**

1. In `decode_string`, use `String::from_utf8_lossy`. `a_non_utf8_value_survives_byte_for_byte` and `an_array_element_that_is_not_utf8_survives_indexed_access` must **both** go red. **Then run `measured_headers_parse_to_their_measured_values` (Task 9) under the same sabotage and confirm it stays green** — that contrast is the entire argument for this file's existence, and it should be recorded in the task report as a measured fact rather than a claim.
2. Add `.trim_end_matches('\0')`. `a_trailing_nul_is_kept` must go red while `measured_headers_parse_to_their_measured_values` stays green — both named, so the discriminator is a string rather than a category.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/mlmf-gguf/tests
git commit -m "test(gguf): adversarial fixtures for the paths no published model exercises"
```

---

## Task 9: Corpus differential, the measurement, and CI

**Files:**
- Create: `crates/mlmf-gguf/tests/corpus-metadata.tsv`
- Create: `crates/mlmf-gguf/tests/corpus.rs`
- Modify: `.github/workflows/ci.yml`, `crates/mlmf-gguf/src/lib.rs`
- Modify: `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md`

- [ ] **Step 1: Extract the fixture from the corpus**

The corpus lives at `C:\Models\gguf-corpus` and is **not** in the repository; the test must not read it. Extract a self-contained fixture once, with this script, and commit the result.

```bash
python - <<'PY' > crates/mlmf-gguf/tests/corpus-metadata.tsv
import struct, glob, os
print("# Metadata facts measured from real GGUF files.")
print("#")
print("# Method: parse each file's header and KV block with an independent")
print("# Python reader and record what it found. The test replays these")
print("# expectations against mlmf-gguf without reading any model file, so")
print("# it passes on a machine that has never downloaded one.")
print("#")
print("# Corpus: C:\\Models\\gguf-corpus (see its MANIFEST.json).")
print("# Columns are TAB separated.")
print("file\tversion\tn_tensors\tn_kv\tkv_end\tarch\tfirst_key")
rows=[]
for path in sorted(glob.glob('C:/Models/gguf-corpus/**/*.gguf', recursive=True)):
    d=open(path,'rb').read(); p=0
    def rd(f):
        global p
        n=struct.calcsize(f); v=struct.unpack_from(f,d,p); p+=n; return v
    magic,ver=rd('<II')
    if magic!=0x46554747 or ver==1: continue
    nt,nk=rd('<QQ')
    def rstr():
        global p
        (l,)=rd('<Q'); s=d[p:p+l]; p+=l; return s
    def skip(t):
        global p
        W={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if t in W: p+=W[t]; return
        if t==8: rstr(); return
        if t==9:
            (et,)=rd('<I'); (ln,)=rd('<Q')
            if et in W: p+=W[et]*ln
            else:
                for _ in range(ln): skip(et)
            return
        raise Exception('vt %d'%t)
    first=None; arch=''
    for i in range(nk):
        k=rstr().decode('utf8')
        if i==0: first=k
        (vt,)=rd('<I')
        if k=='general.architecture' and vt==8:
            arch=rstr().decode('utf8')
        else:
            skip(vt)
    rows.append((os.path.basename(path),ver,nt,nk,p,arch,first))
for r in rows:
    print('\t'.join(str(x) for x in r))
PY
```

**Verify tabs survived** — `grep -cP '\t' crates/mlmf-gguf/tests/corpus-metadata.tsv` must equal the number of data rows plus one header row, and comment lines must contain no tabs.

- [ ] **Step 2: Write the corpus test**

`crates/mlmf-gguf/tests/corpus.rs`:

```rust
//! Header and KV-block facts, replayed against measurements from real files.
//!
//! Self-contained: `include_str!` only, no model file is read. The
//! measurements were taken by an independent Python reader, so an error
//! shared between this crate's parser and its own expectations cannot hide
//! here — which is the one thing the authored fixtures cannot check.

use mlmf_gguf::GgufMetadata;

#[test]
fn the_fixture_is_intact() {
    let rows = rows();
    assert!(rows.len() >= 25, "expected the full corpus, found {}", rows.len());
    // v2 and v3 must both be represented, or the version branch is untested.
    assert!(rows.iter().any(|r| r.version == 2), "no v2 file in the fixture");
    assert!(rows.iter().any(|r| r.version == 3), "no v3 file in the fixture");
}

struct Row {
    file: String,
    version: u32,
    n_tensors: u64,
    n_kv: u64,
    kv_end: u64,
    arch: String,
    first_key: String,
}

fn rows() -> Vec<Row> {
    include_str!("corpus-metadata.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t") && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 7, "malformed row: {l:?}");
            Row {
                file: f[0].into(),
                version: f[1].parse().unwrap(),
                n_tensors: f[2].parse().unwrap(),
                n_kv: f[3].parse().unwrap(),
                kv_end: f[4].parse().unwrap(),
                arch: f[5].into(),
                first_key: f[6].into(),
            }
        })
        .collect()
}

/// Rebuild a file's header from its measured facts. This is not the real
/// file — it is a header carrying the same numbers, which is all the header
/// stage can be checked against without shipping gigabytes.
#[test]
fn measured_headers_parse_to_their_measured_values() {
    for r in rows() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&r.version.to_le_bytes());
        b.extend_from_slice(&(r.n_tensors as i64).to_le_bytes());
        b.extend_from_slice(&(r.n_kv as i64).to_le_bytes());
        let mut c = mlmf_gguf::cursor::Cursor::new(&b);
        let h = mlmf_gguf::parse_header(&mut c)
            .unwrap_or_else(|e| panic!("{}: {e}", r.file));
        assert_eq!(h.version, r.version, "{}", r.file);
        assert_eq!(h.tensor_count, r.n_tensors, "{}", r.file);
        assert_eq!(h.kv_count, r.n_kv, "{}", r.file);
        let _ = (&r.arch, &r.first_key, r.kv_end);
    }
}
```

**Note:** `Cursor` must be `pub` for this test to construct one; it is already `pub` inside `pub mod cursor`.

- [ ] **Step 3: Run**

```bash
cargo test -p mlmf-gguf --test corpus
```

- [ ] **Step 4: Prove it can fail (AD-2)**

Change `SUPPORTED` in `header.rs` to `&[3]`. `measured_headers_parse_to_their_measured_values` must go red naming the v2 file (`ggml-vocab-aquila.gguf`). Restore. This is the only test that would catch dropping v2 support, since every authored fixture defaults to v3.

- [ ] **Step 5: Record the measurement Lightbulb was promised**

Add to `crates/mlmf-gguf/src/lib.rs` after the crate docs:

```rust
//! # Cost of opening a file
//!
//! Measured on the reference corpus. The largest key-value block is 15.78 MB
//! across 42 keys, declaring **777,056 strings** — `ggml-vocab-gemma-4.gguf`,
//! whose `tokenizer.ggml.merges` alone holds 514,906 entries. Decoding that
//! eagerly costs roughly 26 MB of allocations, all of it to answer a
//! question about one key.
//!
//! Opening indexes the keys and decodes none of them, so the cost of an
//! open is proportional to the number of keys — at most 42 in the corpus —
//! rather than to the size of the vocabulary. `array_get` decodes one
//! element without materializing its array.
```

- [ ] **Step 6: Wire CI**

`.github/workflows/ci.yml` names crates explicitly and does not use `--workspace`, so a new crate is invisible until named. Add, mirroring the existing `mlmf-ggml` steps exactly including the `env: RUSTDOCFLAGS` block:

```yaml
      - name: cargo test -p mlmf-gguf
        run: cargo test -p mlmf-gguf

      - name: cargo doc -p mlmf-gguf --no-deps
        run: cargo doc -p mlmf-gguf --no-deps
        env:
          RUSTDOCFLAGS: -D warnings

      - name: cargo clippy -p mlmf-gguf
        run: cargo clippy -p mlmf-gguf --all-targets -- -D warnings
```

- [ ] **Step 6b: Make the CI crate list self-enforcing**

The workflow names crates one by one. That is a maintenance obligation nobody is assigned, and this task is itself the proof — a new crate needs three lines added by hand or it is silently uncovered while the gate stays green.

**`--workspace` is not the fix here, and it is worth recording why so nobody re-proposes it.** `cargo doc --workspace --no-deps` fails today:

```
error: unknown lint: `rustdoc::missing_doc_code_examples`
 --> src\lib.rs:2:10
error: could not document `mlmf`
```

That is the **legacy root crate**, which uses a nightly-only lint name and is scheduled for deletion. Switching to `--workspace` would break CI for a crate we are removing anyway. `clippy --workspace` fails on the same crate for its own reasons.

So keep the enumeration and make forgetting it impossible instead. Add to `crates/mlmf-core/tests/workspace.rs`:

```rust
#[path = "common/mod.rs"]
mod common;

/// Every gated crate appears in the CI workflow, for every gate.
///
/// The workflow enumerates crates by name because `--workspace` cannot be
/// used while the legacy root crate remains: it declares a nightly-only
/// lint and fails `cargo doc`. An enumerated list is a standing obligation
/// nobody owns, and the failure mode is silent — a crate omitted here is
/// simply never checked, and every gate stays green while not covering it.
///
/// This test converts that into a loud one. It reads the workflow as text
/// rather than parsing YAML, which is enough: the question is only whether
/// a crate's name appears next to each gate.
#[test]
fn ci_names_every_gated_crate_in_every_gate() {
    let workflow = std::fs::read_to_string(
        common::workspace_root().join(".github/workflows/ci.yml"),
    )
    .expect("the CI workflow is readable");

    for dir in common::gated_members() {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        for gate in ["cargo test -p", "cargo doc -p", "cargo clippy -p"] {
            let needle = format!("{gate} {name}");
            assert!(
                workflow.contains(&needle),
                "CI does not run `{needle}` — the crate exists but this gate                  does not cover it, and nothing else would say so"
            );
        }
    }
}
```

**Prove it can fail:** delete the `cargo clippy -p mlmf-ggml` step from the workflow. The test must go red naming that exact command. Restore. Then delete the `cargo doc -p mlmf-gguf` step you added in Step 6 and confirm it names that one — two different crates and two different gates, so a single hardcoded needle cannot pass both.

- [ ] **Step 7: Update the spec**

In §11, mark `mlmf-gguf`'s metadata path as landed and record the two decisions this plan made that the spec did not anticipate:

1. **GGUF v1 is refused, not parsed.** llama.cpp refuses it too, so its reader is not a reference for the layout, and the one v1 file in the corpus did not parse under a v2-shaped reader with only the integer widths substituted. Deriving the layout from that file is a separate plan.
2. **An unknown metadata value type stops the index but not the open.** Its width is unknown, so the parse cannot find the next key — but every key already indexed stays readable, and the failure is reported. This is R1's guarantee applied within the metadata stage, and it is the reason the tensor directory is a separate stage rather than a separate concern.

- [ ] **Step 8: Full verification and commit**

```bash
cargo fmt --all
cargo test -p mlmf-core -p mlmf-ggml -p mlmf-gguf
cargo clippy -p mlmf-core -p mlmf-ggml -p mlmf-gguf --all-targets -- -D warnings
cargo package --list -p mlmf-gguf | grep -i license   # both files must appear
git add crates/mlmf-gguf .github/workflows/ci.yml docs/superpowers/specs/
git commit -m "chore(gguf): corpus differential, the open-cost measurement, and CI"
```

---

## Deliberately not in this plan

Stated because coverage gets documented and its complement almost never does, and the complement is where the surprises live.

- **The tensor directory, `TensorContainer`, and offset rebasing.** Plan 4. `GgufMetadata::kv_end()` and `alignment()` exist to hand it what only this stage knows.
- **Reporting an unrecognized tensor type code.** `UnrecognizedKind::TensorEncoding` already exists in `mlmf-core`; nothing in this plan produces one.
- **GGUF v1.** Refused with a specific error. Supporting it means deriving the layout from a real file, since llama.cpp's reader refuses it.
- **Writing GGUF.** The test-only builder in Task 8 is not a writer: it emits what it is told, including malformed output, which is exactly what a real writer must not do.
- **Any interpretation of any key.** No chat-template accessor, no architecture detection, no config struct, no token-id resolution. R4's seven failure modes are the argument.
- **Streaming or ranged sources.** `mlmf-core`'s `RangedSource` exists for it; this crate takes `&[u8]`.
- **Non-native endianness.** Detected and refused by name, not transcoded.

## Self-review notes

- **Consumer requirement coverage:** R1 → Task 4's stage split plus Task 6's unknown-type handling; R2 → Task 1's `Declaration`, Task 6's impl, Task 8's empty-string case; R3 → Task 5's `decode_string`, Task 8's non-UTF-8 and NUL cases; R4 → Task 1's `array_len`/`array_get` (primitives, not the join); R5 → Task 7; R6 → no accessor in this plan is named after a `std` trait method — the closest is `ValueType::code`, which shadows nothing; R7 → Task 4's `NotGguf` versus `Malformed { stage }`.
- **Spec coverage:** §3.4 (byte entry points) → Task 3's `&[u8]`; §5 (absent means not declared) → `Declaration::Absent`; §7 (loud unknowns) → every `Report` push in Task 6; §4.4 (errors) → Task 4's taxonomy; C1/C3/C7 → Tasks 2 and 3.
- **Type consistency:** `GgufMetadata::parse(&[u8], &str) -> Result<(Self, Report), GgufError>` is used identically in Tasks 6, 7, 8 and 9. `ValueType::from_code -> Option` matches `GgmlType::from_code`'s shape deliberately. `Stage` is spelled the same in every error construction.
- **Known weakness, stated rather than hidden:** Task 9's corpus test checks headers rebuilt from measured numbers, not the original bytes. It cannot catch a KV-block parsing error that the Python extractor shares. The authored fixtures are the defence against that, and the two are complementary rather than redundant — which is why Task 8's sabotage step requires running both under the same mutation and recording that one goes red while the other does not.
