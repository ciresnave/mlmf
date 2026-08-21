# GGUF Tensor Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `mlmf-gguf` parses the GGUF tensor directory into `mlmf-core`'s `TensorContainer`, rebasing every offset once so a consumer writes `&blob[d.bytes]` with nothing added.

**Architecture:** A third stage, separate from the metadata stage by construction. `parse_tensors` takes the byte slice and a parsed `GgufMetadata`, because `kv_end()` and `alignment()` are the two facts only the metadata stage knows. Type codes resolve through `mlmf-ggml`, which this crate already depends on and has never used. A tensor whose code will not resolve is **omitted from the list and named in the report** — `TensorDescriptor` cannot say "length unknown", and fabricating a length would let a caller compute a byte range for a tensor whose extent is genuinely unknown.

**Tech Stack:** Rust 2024, `#![forbid(unsafe_code)]`, no I/O in the crate (`&[u8]` in, structures out), `mlmf-core` + `mlmf-ggml` as the only dependencies.

**Spec:** `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md`, and the consumer requirements in `docs/superpowers/specs/2026-08-19-lightbulb-gguf-seam-requirements.md`.

## Global Constraints

- Rust edition 2024. `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`.
- No file I/O anywhere in `crates/mlmf-gguf/src`. The core is about what is IN the file, not how it is obtained.
- **MLMF reads and writes model files. It is never an interpreter of their content.** No tensor-name parsing, no layer-index extraction, no architecture inference. A name is an opaque key.
- Every gate green before every commit: `cargo test --workspace`, `cargo clippy -p mlmf-gguf --all-targets -- -D warnings`, `RUSTDOCFLAGS=-D warnings cargo doc -p mlmf-gguf --no-deps`, `rustfmt --edition 2024 --check` on staged files.
- `crates/mlmf-core/src` is **out of scope** unless a task says otherwise in its Files list.

---

## Measured facts this plan is built on

Every number here was measured against the reference corpus at `C:/Models/gguf-corpus` (29 files, 1.13 GiB) with an independent Python reader, not read off a specification page. The script is `tensor_recon.py`; reproduce before trusting.

**The tensor-info record, confirmed byte-exact on 28 of 28 readable files:**

```
each tensor info:  name (u64 len + bytes, no terminator)
                   n_dims (u32)
                   dims (u64 x n_dims), declared order, first dimension first
                   type (u32, a ggml type code)
                   offset (u64, RELATIVE to the start of the data region)
```

The falsification test was whether the last tensor's computed end equals the file length. On all nine files that carry tensors it matched with **delta 0**, including a 270,885,952-byte f16 model. Zero contradictions. The one skipped file is GGUF v1, which this build refuses by design.

**Two facts that a specification page would not have given, both of which break a naive implementation:**

1. **A file with zero tensors has no data region and no padding.** It ends exactly at the end of the (empty) tensor directory. **18 of the 28 readable corpus files are this shape** — they are the `llamacpp-vocab/*` files, which is precisely what a metadata-only consumer opens. An implementation that computes `data_start = align_up(dir_end)` and bounds-checks it at parse time fails to open a majority of the corpus.

2. **Padding is `(align - dir_end % align) % align`, not a full block when already aligned.** `SmolLM2-135M-Instruct-f16.gguf` has `dir_end == data_start == 1785664`; every other quant file has `dir_end = 1785944` and `data_start = 1785952`, eight bytes of padding. A `dir_end + align - (dir_end % align)` formula adds a phantom 32 bytes when the directory happens to land on a boundary, and shifts every tensor in the file.

**Scale, for the cost arguments below:** the quant files declare **272 tensors** each. A 70B model declares on the order of a thousand. `TensorContainer::tensor`'s doc already says a format crate parsing a real model must override the default linear scan.

## Design decisions

**D1 — A separate call, not a field of `GgufMetadata`.** R1 requires that reading metadata cannot fail on tensor content. Plan 3 guaranteed that by *shape*: the metadata stage has no access to a type table, so it cannot fail against one. Keeping the tensor stage a separate function preserves that guarantee — a caller who never calls `parse_tensors` cannot be failed by it. Folding tensors into `GgufMetadata::parse` would make R1 a matter of discipline again.

**D2 — An unresolvable type code omits the tensor and reports it.** `mlmf-core`'s `TensorContainer::tensors` documents this already. The alternative — a descriptor with a guessed length — hands a consumer a byte range for a tensor whose extent is unknown, and a wrong range reads plausible floats.

**D3 — Rebase once, at parse time.** `TensorDescriptor::bytes` is absolute into the slice the container was opened over. GGUF's recorded offset is relative to the data region; the descriptor stores `data_start + info.offset`. This is `mlmf-core`'s CD-4 and its doc argues it at length.

**D4 — Overlap is reported, not refused.** Two tensors whose ranges overlap is a malformed file, but every tensor still has a well-defined range and a consumer may want the ones that do not overlap. Refusing the open would make one bad tensor cost the whole file, which is R1's argument one stage over.

**D5 — `u64 -> usize` failures are the consumer's platform, not the file.** `Shape::new` takes `usize`. A dimension that does not fit is a fact about a 32-bit target, not about the model, and it must not be reported as a malformed file. This distinction is recorded in the Lightbulb requirements as a constraint on `Declaration`; the same reasoning applies here.

## File Structure

- `crates/mlmf-gguf/src/tensors.rs` — **new.** The tensor directory: record parsing, padding, type resolution, rebasing, and the `TensorContainer` impl. One file because these are one grammar and one pass; splitting record-parsing from rebasing would put the offset base in two places, which is the mistake `TensorDescriptor::bytes` exists to prevent.
- `crates/mlmf-gguf/src/error.rs` — modified. `Stage::TensorDirectory` exists and is constructed nowhere; this plan is what makes it live.
- `crates/mlmf-gguf/src/lib.rs` — modified. `pub mod tensors;` and re-exports.
- `crates/mlmf-gguf/tests/fixture/mod.rs` — modified. The builder gains tensor support.
- `crates/mlmf-gguf/tests/authored.rs` — modified. Adversarial tensor cases.
- `crates/mlmf-gguf/tests/corpus.rs` — modified. The differential gains tensor columns.
- `crates/mlmf-gguf/tests/corpus-tensors.tsv` — **new.** Independently measured tensor facts.

## The baseline this plan is measured against

Plan 3 produced one number worth more than any single finding: **eleven controls could not reach the assertion they named, and ten of the eleven were mine.** That is a measured rate rather than an impression, and it is the only such rate in the portfolio.

Plan 3 added controls in response — expected-kill-sets, whole-value assertions, the standing rule about trailing keys. **If plan 4's rate does not improve, the controls are not the binding constraint and the next plan should try something else rather than adding more of the same.** Record the count in the ledger as each task closes. The whole-branch pass at the end (Task 8) is not optional for the same reason: every Important finding in plan 3 lived in the last third, in the seams between tasks that per-task review structurally cannot see.

---

## Task 0: A report entry for a tensor this build declined

**Files:**
- Modify: `crates/mlmf-core/src/report.rs` (**in scope for this task only**)

**This task exists because writing the rest of the plan produced two misuses
of an existing variant, and plan 3 ended by removing exactly that defect from
its sibling.**

`UnrecognizedKind::TensorEncoding { name, family, code }` means "this build
cannot resolve the declared encoding". Task 3 uses it correctly. Tasks 4 and
5 need to report a **duplicate tensor name** and an **overlapping byte
range**, and neither is an encoding failure: the duplicate resolved
perfectly well, and an overlap has no code to name. Reaching for
`TensorEncoding` there would put a meaningless `code: 0` in a field
documented as "the code exactly as declared" — which is the same lie, in the
same enum, that plan 3's final task spent a whole cycle removing from
`MetadataKey`.

Plan 3 already established the shape of the fix and it applies unchanged.

- [ ] **Step 1: Write the failing test**

In `crates/mlmf-core/src/report.rs`'s test module:

```rust
    #[test]
    fn a_declined_tensor_says_why_without_inventing_a_type_code() {
        // `TensorEncoding` answers "this build cannot resolve the encoding".
        // A duplicate name and an overlapping range are neither, and forcing
        // them into it means a `code` field holding a number the file never
        // declared. That is the defect this enum's MetadataKey sibling was
        // repaired for.
        let mut r = Report::new();
        r.push(Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: "blk.0.attn_q.weight".into(),
                reason: "declared more than once; the first occurrence is kept".into(),
            },
            origin: "model.gguf".into(),
        });
        r.push(Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: "blk.1.attn_k.weight".into(),
                reason: "byte range overlaps blk.0.attn_q.weight".into(),
            },
            origin: "model.gguf".into(),
        });

        // Whole values, in order. A chain of field assertions cannot see the
        // two entries swapped, and swapping them attributes each complaint
        // to the wrong tensor.
        assert_eq!(
            r.entries().iter().map(|e| e.kind.clone()).collect::<Vec<_>>(),
            vec![
                UnrecognizedKind::TensorDeclined {
                    name: "blk.0.attn_q.weight".into(),
                    reason: "declared more than once; the first occurrence is kept".into(),
                },
                UnrecognizedKind::TensorDeclined {
                    name: "blk.1.attn_k.weight".into(),
                    reason: "byte range overlaps blk.0.attn_q.weight".into(),
                },
            ]
        );
    }
```

- [ ] **Step 2: Run and watch it fail**

```bash
cargo test -p mlmf-core --lib a_declined_tensor
```

Expected: does not compile — no such variant.

- [ ] **Step 3: Add the variant**

In `crates/mlmf-core/src/report.rs`, alongside `TensorEncoding`:

```rust
    /// A tensor this build declined for a reason that is not its encoding.
    ///
    /// Like [`Self::TensorEncoding`], the tensor is **omitted from the
    /// container's list** — a consumer sees a shorter list and this entry is
    /// the only other signal. Unlike it, there is no type code involved: the
    /// encoding resolved fine, or the complaint is not about the encoding at
    /// all. A duplicate name and an overlapping byte range are both this.
    ///
    /// Separate from `TensorEncoding` rather than a `code: Option<u32>` added
    /// to it, because the two answer different questions and a reader
    /// branching on the kind should not have to check whether a field is
    /// meaningful before trusting it.
    TensorDeclined {
        /// Tensor name exactly as declared.
        name: String,
        /// Why, in terms a person can act on. Never a substitute for a
        /// field that exists — see the `MetadataKey` repair.
        reason: String,
    },
```

`UnrecognizedKind` is `#[non_exhaustive]`, so adding a variant is not a
breaking change for a downstream matcher. It IS a change every in-crate
exhaustive `match` must handle; find them with `cargo build` rather than by
grepping.

- [ ] **Step 4: Run and watch it pass**

```bash
cargo test --workspace
cargo clippy -p mlmf-core --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p mlmf-core --no-deps
```

- [ ] **Step 5: Prove it can fail (AD-2)**

1. Swap the two entries in the expected vector.
   Expect: red. This is the control for using a whole-value comparison
   rather than a field chain, and it is the only sabotage that distinguishes
   them.
2. Change `reason` to `String::new()` at the first push site.
   Expect: red on the same comparison, showing an empty reason. A
   `!r.is_empty()` assertion would not see it, which is the point.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/mlmf-core
git commit -m "feat(core): a report entry for a tensor declined for a reason that is not its encoding"
```

---

## Task 1: The tensor-info record

**Files:**
- Create: `crates/mlmf-gguf/src/tensors.rs`
- Modify: `crates/mlmf-gguf/src/lib.rs`

**Interfaces:**
- Consumes: `Cursor`, `GgufError`, `Stage`, `read_key`-style string reading.
- Produces: `struct RawInfo { name: String, dims: Vec<u64>, code: u32, offset: u64 }` and `fn read_info(cursor: &mut Cursor<'_>) -> Result<RawInfo, GgufError>`, both `pub(crate)`.

- [ ] **Step 1: Write the failing tests**

Create `crates/mlmf-gguf/src/tensors.rs` with a `mod tests` containing:

```rust
    /// Encode one tensor-info record.
    fn info(name: &str, dims: &[u64], code: u32, offset: u64) -> Vec<u8> {
        let mut b = (name.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&code.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b
    }

    #[test]
    fn reads_a_record_and_lands_exactly_after_it() {
        let b = info("blk.0.attn_q.weight", &[4096, 4096], 0, 1024);
        let mut c = Cursor::new(&b);
        let got = read_info(&mut c).expect("parses");
        // The WHOLE record in one comparison. A chain of field assertions
        // cannot see `code` and a dimension transposed, and both are u32
        // and u64 respectively sitting adjacent in the byte stream.
        assert_eq!(
            got,
            RawInfo {
                name: "blk.0.attn_q.weight".to_string(),
                dims: vec![4096, 4096],
                code: 0,
                offset: 1024,
            }
        );
        // Landing exactly after is what makes the NEXT record readable, and
        // no assertion above can see the cursor one byte off.
        assert_eq!(c.pos(), b.len() as u64, "must consume the record exactly");
    }

    #[test]
    fn a_rank_zero_tensor_is_a_record_not_an_error() {
        // GGUF does not forbid it and the reader must not either: `n_dims`
        // of 0 is a well-formed record with an empty dims list. Refusing it
        // here would be this crate deciding what a model may contain.
        let b = info("scalar", &[], 6, 0);
        let mut c = Cursor::new(&b);
        let got = read_info(&mut c).expect("parses");
        assert_eq!(got.dims, Vec::<u64>::new());
        assert_eq!(c.pos(), b.len() as u64);
    }

    #[test]
    fn a_dimension_count_larger_than_the_file_fails_before_allocating() {
        // `n_dims` is a declared u32. At 0xFFFF_FFFF, a `Vec::with_capacity`
        // from it asks for 34 GB before a single bounds check runs. The
        // count must be bounded by the bytes that remain.
        let mut b = 6u64.to_le_bytes().to_vec();
        b.extend_from_slice(b"scalar");
        b.extend_from_slice(&u32::MAX.to_le_bytes());
        let mut c = Cursor::new(&b);
        let err = read_info(&mut c).unwrap_err();
        assert!(matches!(
            err,
            GgufError::Truncated {
                stage: Stage::TensorDirectory,
                ..
            }
        ));
    }

    #[test]
    fn a_truncated_record_reports_the_tensor_stage_not_the_metadata_stage() {
        // R7's principle inside the crate: the stage tag is how a caller
        // tells "your metadata is malformed" from "your tensor directory
        // is". Every error out of this module carries TensorDirectory.
        let full = info("t", &[8], 0, 0);
        for cut in 1..full.len() {
            let mut c = Cursor::new(&full[..cut]);
            match read_info(&mut c) {
                Err(GgufError::Truncated { stage, .. }) => {
                    assert_eq!(stage, Stage::TensorDirectory, "cut at {cut}");
                }
                Err(other) => panic!("cut at {cut}: wrong error {other:?}"),
                Ok(v) => panic!("cut at {cut}: parsed {v:?} from a truncated record"),
            }
        }
    }

    #[test]
    fn a_name_that_is_not_utf8_is_malformed_rather_than_lossy() {
        // A tensor name is a lookup key, exactly as a metadata key is. A
        // lossy conversion produces a name no caller can look up and no
        // error anywhere. Values may be non-UTF-8; keys may not.
        let mut b = 2u64.to_le_bytes().to_vec();
        b.extend_from_slice(&[0xFF, 0xFE]);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let mut c = Cursor::new(&b);
        assert!(matches!(
            read_info(&mut c).unwrap_err(),
            GgufError::Malformed {
                stage: Stage::TensorDirectory,
                ..
            }
        ));
    }
```

- [ ] **Step 2: Run and watch them fail**

```bash
cargo test -p mlmf-gguf --lib tensors
```

Expected: does not compile — `tensors` is not a module yet. Add `pub mod tensors;` to `lib.rs`, then expect failures naming `read_info` and `RawInfo`.

- [ ] **Step 3: Implement**

```rust
//! The tensor directory: what tensors a file declares and where their bytes are.

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};

/// One tensor-info record, exactly as declared and not yet resolved.
///
/// `code` is a raw ggml type code, not a resolved encoding, and `offset` is
/// relative to the data region rather than to the file. Both stay raw here
/// so that record parsing has no opinion about type tables — the stage that
/// resolves them is the stage that can fail against a type table, and this
/// one must not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RawInfo {
    pub(crate) name: String,
    pub(crate) dims: Vec<u64>,
    pub(crate) code: u32,
    pub(crate) offset: u64,
}

fn trunc(at: u64, t: crate::cursor::Truncated) -> GgufError {
    GgufError::Truncated {
        stage: Stage::TensorDirectory,
        offset: at,
        needed: t.needed,
        available: t.available,
    }
}

/// Read one tensor-info record, leaving the cursor on the next.
pub(crate) fn read_info(cursor: &mut Cursor<'_>) -> Result<RawInfo, GgufError> {
    let at = cursor.pos();
    let len = cursor.u64().map_err(|t| trunc(at, t))?;
    let at = cursor.pos();
    let raw = cursor.take(len).map_err(|t| trunc(at, t))?;
    let name = core::str::from_utf8(raw)
        .map_err(|e| GgufError::Malformed {
            stage: Stage::TensorDirectory,
            offset: at,
            detail: format!("tensor name is not valid UTF-8: {e}"),
        })?
        .to_string();

    let at = cursor.pos();
    let n_dims = cursor.u32().map_err(|t| trunc(at, t))?;
    // Bound the declared count by what remains BEFORE allocating. `n_dims`
    // is a u32 from the file; at u32::MAX a `with_capacity` asks for 34 GB
    // and aborts the process before any bounds check runs.
    let need = u64::from(n_dims)
        .checked_mul(8)
        .ok_or_else(|| GgufError::Malformed {
            stage: Stage::TensorDirectory,
            offset: at,
            detail: format!("{n_dims} dimensions overflows a byte count"),
        })?;
    if need > cursor.remaining() {
        return Err(GgufError::Truncated {
            stage: Stage::TensorDirectory,
            offset: cursor.pos(),
            needed: need,
            available: cursor.remaining(),
        });
    }
    let mut dims = Vec::new();
    dims.try_reserve(n_dims as usize)
        .map_err(|_| GgufError::Malformed {
            stage: Stage::TensorDirectory,
            offset: at,
            detail: format!("cannot allocate {n_dims} dimensions"),
        })?;
    for _ in 0..n_dims {
        let at = cursor.pos();
        dims.push(cursor.u64().map_err(|t| trunc(at, t))?);
    }

    let at = cursor.pos();
    let code = cursor.u32().map_err(|t| trunc(at, t))?;
    let at = cursor.pos();
    let offset = cursor.u64().map_err(|t| trunc(at, t))?;

    Ok(RawInfo {
        name,
        dims,
        code,
        offset,
    })
}
```

Add to `lib.rs`:

```rust
pub mod tensors;
```

- [ ] **Step 4: Run and watch them pass**

```bash
cargo test -p mlmf-gguf
cargo clippy -p mlmf-gguf --all-targets -- -D warnings
```

- [ ] **Step 5: Prove they can fail (AD-2)**

**Name the expected kill set before each run. Report the SHORTFALL, not the count.** A sabotage that kills only some of what it should is the only signal separating a live suite from one whose fixtures carry no information.

1. Swap the `code` and `offset` reads (read u64 then u32).
   Expect: `reads_a_record_and_lands_exactly_after_it` red on the whole-record comparison; `a_rank_zero_tensor_is_a_record_not_an_error` red on the position assertion. **Record which assertion fires in the first test** — if it is the position rather than the value comparison, the value comparison is not the control you think it is.
2. Delete the `need > cursor.remaining()` bound.
   Expect: `a_dimension_count_larger_than_the_file_fails_before_allocating` red — or the test process aborts, which is itself the finding and must be reported rather than smoothed. Record which.
3. Replace the name's `from_utf8` with a lossy conversion.
   Expect: `a_name_that_is_not_utf8_is_malformed_rather_than_lossy` red, and nothing else. If another test also dies, the fixtures share a dependency they should not.
4. Change `Stage::TensorDirectory` to `Stage::Metadata` in `trunc`.
   Expect: `a_truncated_record_reports_the_tensor_stage_not_the_metadata_stage` red at a named cut position, and the truncation arm of test 3's sabotage unaffected.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/mlmf-gguf
git commit -m "feat(gguf): the tensor-info record, bounds-checked before allocating"
```

---

## Task 2: The data region, padding, and the zero-tensor file

**Files:**
- Modify: `crates/mlmf-gguf/src/tensors.rs`

**Interfaces:**
- Consumes: `GgufMetadata::kv_end`, `GgufMetadata::alignment`.
- Produces: `pub(crate) fn data_start(dir_end: u64, alignment: u64) -> Option<u64>`.

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn padding_is_zero_when_the_directory_already_lands_on_a_boundary() {
        // Measured, not assumed. `SmolLM2-135M-Instruct-f16.gguf` has
        // dir_end == data_start == 1785664 with alignment 32: the writer
        // emits NO padding when none is needed. A formula of
        // `dir_end + align - (dir_end % align)` adds a phantom 32 bytes
        // when the directory happens to land on a boundary, and shifts
        // every tensor in the file by one alignment block.
        assert_eq!(data_start(1785664, 32), Some(1785664));
        // And the same file's siblings, which do need padding:
        assert_eq!(data_start(1785944, 32), Some(1785952));
        // The whole table, so an off-by-one in either direction is visible:
        assert_eq!(
            (0..=8).map(|d| data_start(d, 4)).collect::<Vec<_>>(),
            vec![
                Some(0),
                Some(4),
                Some(4),
                Some(4),
                Some(4),
                Some(8),
                Some(8),
                Some(8),
                Some(8)
            ]
        );
    }

    #[test]
    fn a_data_start_that_overflows_is_none_rather_than_wrapping() {
        // `dir_end` comes from a walk over real bytes so it cannot be near
        // u64::MAX in practice, but the addition is still an addition and
        // a wrap would produce a data region BEFORE the directory.
        assert_eq!(data_start(u64::MAX, 32), None);
        assert_eq!(data_start(u64::MAX - 1, 2), None);
    }
```

- [ ] **Step 2: Run and watch it fail**

```bash
cargo test -p mlmf-gguf --lib data_start
```

Expected: FAIL, `data_start` not defined.

- [ ] **Step 3: Implement**

```rust
/// Where the tensor data region begins: `dir_end` rounded up to `alignment`.
///
/// `None` on overflow rather than a wrapped value, which would place the
/// data region BEFORE the directory that describes it.
///
/// **The padding is `(alignment - dir_end % alignment) % alignment`, and
/// the outer `%` is the part that matters.** A writer emits no padding at
/// all when the directory already ends on a boundary — measured on
/// `SmolLM2-135M-Instruct-f16.gguf`, where `dir_end == data_start ==
/// 1785664`. The naive `dir_end + alignment - (dir_end % alignment)` adds a
/// full block in that case and shifts every tensor in the file.
///
/// `alignment` is a power of two and at least 1: `GgufMetadata::alignment`
/// guarantees it, falling back to 32 for a file that declares otherwise.
pub(crate) fn data_start(dir_end: u64, alignment: u64) -> Option<u64> {
    debug_assert!(alignment.is_power_of_two(), "caller guarantees this");
    let pad = (alignment - dir_end % alignment) % alignment;
    dir_end.checked_add(pad)
}
```

- [ ] **Step 4: Run and watch it pass**

- [ ] **Step 5: Prove they can fail (AD-2)**

1. Drop the outer `% alignment`.
   Expect: `padding_is_zero_when_the_directory_already_lands_on_a_boundary` red on the FIRST assertion (`1785664`), because that is the aligned case. The table assertion would also catch it at `d == 0`, `4`, `8`. **If only the table fires and not the first assertion, the measured f16 case is not doing work and should be said so.**
2. Replace `checked_add` with `+`.
   Expect: `a_data_start_that_overflows_is_none_rather_than_wrapping` red — in a release-mode run it wraps silently, in debug it panics. Report which you saw, because "the test went red by panicking" and "the test went red by asserting" are different levels of evidence.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(gguf): the data region begins where the directory ends, rounded up"
```

---

## Task 3: Resolving type codes, and the tensors that cannot be

**Files:**
- Modify: `crates/mlmf-gguf/src/tensors.rs`

**Interfaces:**
- Consumes: `mlmf_ggml::GgmlType::from_code`, `GgmlType::encoding`, `GgmlType::nbytes`, `mlmf_core::{Report, Unrecognized, UnrecognizedKind, Shape, TensorDescriptor}`.
- Produces: `pub(crate) fn resolve(info: &RawInfo, data_start: u64, origin: &str, report: &mut Report) -> Option<TensorDescriptor>`.

**This is the task where the charter bites.** A code this build cannot resolve produces no descriptor. `mlmf-core`'s `TensorContainer::tensors` documents exactly this, and the reason is in `TensorDescriptor::bytes`: a byte range is the one fact a consumer cannot check for itself.

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn a_resolvable_tensor_becomes_a_descriptor_rebased_onto_the_slice() {
        // D3, and the reason `TensorDescriptor::bytes` is documented at
        // length: GGUF records an offset relative to the DATA REGION, and
        // the descriptor must carry an offset relative to the SLICE. A
        // consumer that guessed either base would read plausible floats and
        // only one guess would be right.
        let mut report = Report::new();
        let info = RawInfo {
            name: "blk.0.attn_q.weight".into(),
            dims: vec![64, 4],
            code: 0, // F32
            offset: 512,
        };
        let d = resolve(&info, 1_000_000, "t.gguf", &mut report).expect("resolves");
        assert_eq!(
            (d.name.as_str(), d.shape.dims(), d.bytes.clone()),
            (
                "blk.0.attn_q.weight",
                [64usize, 4].as_slice(),
                1_000_512..1_000_512 + 64 * 4 * 4
            )
        );
        assert_eq!(d.encoding, Encoding::Dense(DType::F32));
        assert!(report.is_empty(), "a resolvable tensor is not a finding");
    }

    #[test]
    fn an_unresolvable_code_omits_the_tensor_and_names_it_in_the_report() {
        // D2. There is no descriptor to produce: `TensorDescriptor` has no
        // way to say "length unknown", and inventing one would hand a
        // caller a byte range for a tensor whose extent is unknown. The
        // report is the only signal, and a consumer ignoring it sees a
        // shorter list with nothing else to notice.
        let mut report = Report::new();
        let info = RawInfo {
            name: "blk.0.future".into(),
            dims: vec![32],
            code: 9999,
            offset: 0,
        };
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        // The WHOLE entry. `!report.is_empty()` cannot see the wrong tensor
        // named, the wrong family, or the code silently defaulted.
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorEncoding {
                    name: "blk.0.future".into(),
                    family: "ggml",
                    code: 9999,
                },
                origin: "t.gguf".into(),
            }]
        );
    }

    #[test]
    fn a_retired_code_is_reported_like_any_other_unknown() {
        // ggml has eight retired slots. They are not "unknown to this
        // build" in the same sense — nothing will ever define them again —
        // but the consumer-visible outcome is identical: no descriptor, one
        // report entry. Distinguishing them would be interpretation.
        let mut report = Report::new();
        let info = RawInfo {
            name: "old".into(),
            dims: vec![32],
            code: 4, // Q4_2, retired
            offset: 0,
        };
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        assert_eq!(report.entries().len(), 1);
    }

    #[test]
    fn a_ragged_row_is_reported_rather_than_rounded() {
        // `GgmlType::nbytes` refuses a first dimension that is not a whole
        // number of blocks — the rule is stronger than whole-tensor
        // divisibility and mlmf-ggml owns it. This crate must not paper
        // over that by rounding: a rounded length is a byte range that
        // reads into the next tensor.
        let mut report = Report::new();
        let info = RawInfo {
            name: "ragged".into(),
            dims: vec![33], // Q4_0 blocks are 32 elements
            code: 2,
            offset: 0,
        };
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        assert_eq!(report.entries().len(), 1);
    }

    #[test]
    fn a_dimension_that_does_not_fit_the_platform_is_not_a_malformed_file() {
        // D5. `Shape::new` takes `usize`. On a 64-bit target every u64
        // dimension fits and this arm is unreachable; on a 32-bit one it is
        // not. The distinction is recorded because it is easy to report
        // this as a bad file, and it is not one — it is a fact about the
        // machine doing the reading.
        //
        // Asserted through the report's TEXT rather than by constructing a
        // 32-bit failure, which this test cannot do on the host it runs on.
        // The control below is what proves the arm exists.
        let mut report = Report::new();
        let info = RawInfo {
            name: "huge".into(),
            dims: vec![u64::MAX],
            code: 0,
            offset: 0,
        };
        // On a 64-bit host this fails in `nbytes` on overflow, not on the
        // `usize` conversion. Either way: no descriptor, one entry, and the
        // file is not called malformed.
        assert!(resolve(&info, 0, "t.gguf", &mut report).is_none());
        assert_eq!(report.entries().len(), 1);
    }
```

- [ ] **Step 2: Run and watch them fail**

- [ ] **Step 3: Implement**

```rust
use mlmf_core::{Encoding, Report, Shape, TensorDescriptor, Unrecognized, UnrecognizedKind};
use mlmf_ggml::GgmlType;

/// Turn a raw record into a descriptor, or report why it cannot be one.
///
/// `None` means the tensor is **omitted from the container's list** and the
/// report names it. Four things reach that outcome and all four look the
/// same to a consumer: a code this build does not know, a retired code, a
/// shape `mlmf-ggml` refuses as ragged, and a size that overflows. They are
/// deliberately not distinguished here — telling them apart is
/// interpretation, and the consumer-visible fact is identical.
pub(crate) fn resolve(
    info: &RawInfo,
    data_start: u64,
    origin: &str,
    report: &mut Report,
) -> Option<TensorDescriptor> {
    let complain = |report: &mut Report| {
        report.push(Unrecognized {
            kind: UnrecognizedKind::TensorEncoding {
                name: info.name.clone(),
                family: "ggml",
                code: info.code,
            },
            origin: origin.to_string(),
        });
        None::<TensorDescriptor>
    };

    let Some(ty) = GgmlType::from_code(info.code) else {
        return complain(report);
    };
    let Ok(nbytes) = ty.nbytes(&info.dims, &info.name) else {
        return complain(report);
    };
    let dims: Option<Vec<usize>> = info.dims.iter().map(|d| usize::try_from(*d).ok()).collect();
    let Some(dims) = dims else {
        return complain(report);
    };
    let start = data_start.checked_add(info.offset)?;
    let end = start.checked_add(nbytes)?;

    Some(TensorDescriptor {
        name: info.name.clone(),
        shape: Shape::new(dims),
        encoding: ty.encoding(),
        bytes: start..end,
    })
}
```

**Note for the implementer:** the two `checked_add`s return `None` WITHOUT pushing a report entry, which is inconsistent with the four paths above. Decide deliberately and say what you decided: either push there too (and add an assertion), or document why an offset that overflows the address space is a different category. Do not leave it as an accident.

- [ ] **Step 4: Run and watch them pass**

- [ ] **Step 5: Prove they can fail (AD-2)**

1. Return a descriptor with a guessed length instead of reporting, for an unknown code.
   Expect: `an_unresolvable_code_omits_the_tensor_and_names_it_in_the_report` red on the `is_none`, and the whole-entry comparison never reached. **That is the correct order** — record it, because if the entry comparison fires first the `is_none` is unproven.
2. Drop `data_start` from the rebase (`start = info.offset`).
   Expect: `a_resolvable_tensor_becomes_a_descriptor_rebased_onto_the_slice` red on the tuple comparison, showing `512..1536` against `1000512..1001536`. This is the single most consequential defect in the plan: it is off by a megabyte and every byte it reads is a real float from a real tensor.
3. Round the ragged row up instead of refusing.
   Expect: `a_ragged_row_is_reported_rather_than_rounded` red. Report the byte length the rounding produces — a length that reads into the next tensor is worse than an error, and the number is the argument.
4. Report `family: "gguf"` instead of `"ggml"`.
   Expect: the whole-entry comparison red. A membership or length check would not see it, and the family is what tells a consumer which code space the number lives in.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(gguf): resolve ggml codes, and omit-and-report the ones that will not"
```

---

## Task 4: `GgufTensors` and the `TensorContainer` seam

**Files:**
- Modify: `crates/mlmf-gguf/src/tensors.rs`, `crates/mlmf-gguf/src/lib.rs`

**Interfaces:**
- Produces: `pub struct GgufTensors<'a>`, `pub fn parse_tensors<'a>(bytes: &'a [u8], meta: &GgufMetadata<'a>, origin: &str) -> Result<(GgufTensors<'a>, Report), GgufError>`, and `impl TensorContainer for GgufTensors<'_>`.

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn a_file_with_no_tensors_opens_and_has_no_data_region() {
        // 18 of the 28 readable corpus files are this shape — every
        // `llamacpp-vocab/*` file — and they END at the tensor directory
        // with no padding and no data. An implementation that computes
        // `data_start` eagerly and bounds-checks it against the file length
        // refuses a MAJORITY OF THE CORPUS, and refuses precisely the files
        // a metadata-only consumer opens.
        let bytes = gguf_with_tensors(&[], &[]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("opens");
        assert_eq!(t.tensors(), &[]);
        assert!(report.is_empty());
        assert_eq!(bytes.len() as u64, m.kv_end(), "the file ends at kv_end");
    }

    #[test]
    fn tensor_bytes_returns_the_declared_range_and_borrows_it() {
        let payload: Vec<u8> = (0..64u8).collect();
        let bytes = gguf_with_tensors(&[("t", &[16], 0, 0)], &payload);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, _) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        let d = t.tensor("t").expect("declared");
        let got = t.tensor_bytes(d).expect("in range");
        assert_eq!(&*got, &payload[..64]);
        // GGUF is always borrowable. `Cow::Owned` here would mean MLMF
        // allocated a copy of a tensor, which is the invisible cost AL-3
        // exists to forbid — and the seam returns `Cow` precisely so a
        // caller can see it.
        assert!(matches!(got, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn a_declared_range_past_the_end_is_an_error_not_a_panic() {
        // A file may declare an offset its own bytes do not cover. That is
        // the file's fault and it must surface as an error, not a slice
        // panic — and not at parse time either, because the other tensors
        // are still readable. R1's shape, one stage over.
        let bytes = gguf_with_tensors(&[("t", &[16], 0, 1 << 40)], &[0u8; 64]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, _) = parse_tensors(&bytes, &m, "t.gguf").expect("the OPEN survives");
        let d = t.tensor("t").expect("still declared");
        assert!(t.tensor_bytes(d).is_err(), "but reading it fails");
    }

    #[test]
    fn lookup_is_indexed_rather_than_a_linear_scan() {
        // `TensorContainer::tensor`'s doc requires this of a format crate
        // parsing a real model: the corpus quants declare 272 tensors and a
        // 70B declares about a thousand, with consumers doing by-name
        // lookups inside per-layer loops.
        //
        // Asserted structurally rather than by timing: the index must hold
        // an entry for every tensor in the list, so a lookup cannot be
        // walking. A timing assertion would be flaky and would pass on a
        // fast linear scan.
        let specs: Vec<(String, Vec<u64>, u32, u64)> = (0..300)
            .map(|i| (format!("blk.{i}.w"), vec![32], 0, i as u64 * 128))
            .collect();
        let refs: Vec<(&str, &[u64], u32, u64)> = specs
            .iter()
            .map(|(n, d, c, o)| (n.as_str(), d.as_slice(), *c, *o))
            .collect();
        let bytes = gguf_with_tensors(&refs, &vec![0u8; 300 * 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, _) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert_eq!(t.index_len(), t.tensors().len(), "every tensor is indexed");
        assert_eq!(t.tensor("blk.299.w").map(|d| d.name.as_str()), Some("blk.299.w"));
        assert_eq!(t.tensor("blk.300.w"), None);
    }

    #[test]
    fn a_duplicate_tensor_name_keeps_the_first_and_reports_the_second() {
        // Same rule as a duplicate metadata key, for the same reason:
        // taking the last would make the file's meaning depend on parse
        // order. GGUF does not forbid it.
        let bytes = gguf_with_tensors(&[("t", &[16], 0, 0), ("t", &[16], 0, 64)], &[0u8; 128]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert_eq!(
            t.tensors().iter().map(|d| d.name.as_str()).collect::<Vec<_>>(),
            ["t"],
            "the second occurrence must not be indexed"
        );
        assert!(!report.is_empty(), "and it must be reported");
    }
```

The fixture helper, in the same `mod tests`:

```rust
    /// A GGUF with a tensor directory and a data region.
    ///
    /// `tensors` are (name, dims, ggml code, offset-within-data-region).
    fn gguf_with_tensors(tensors: &[(&str, &[u64], u32, u64)], data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes()); // no KV pairs
        for (n, d, c, o) in tensors {
            b.extend_from_slice(&info(n, d, *c, *o));
        }
        // Pad to 32, the default alignment, using the SAME rule the
        // implementation uses — and note that with no tensors and no data
        // this loop adds nothing, which is the corpus's own shape.
        if !data.is_empty() {
            while b.len() % 32 != 0 {
                b.push(0);
            }
        }
        b.extend_from_slice(data);
        b
    }
```

- [ ] **Step 2: Run and watch them fail**

- [ ] **Step 3: Implement**

```rust
/// A GGUF file's tensor directory, parsed and rebased.
#[derive(Debug)]
pub struct GgufTensors<'a> {
    bytes: &'a [u8],
    descriptors: Vec<TensorDescriptor>,
    index: std::collections::HashMap<String, usize>,
    data_start: u64,
}

impl<'a> GgufTensors<'a> {
    /// Where the tensor data region begins, absolute in the slice.
    ///
    /// For a file with no tensors this is a position that may lie one
    /// alignment block PAST the end of the file: the writer emits no
    /// padding when there is nothing to pad for. It is reported rather than
    /// validated for exactly that reason — see `parse_tensors`.
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.data_start
    }

    /// How many names the lookup index holds. Test surface for the
    /// structural assertion that lookup is not a scan.
    #[must_use]
    pub fn index_len(&self) -> usize {
        self.index.len()
    }
}

/// Parse the tensor directory that follows `meta`'s key-value block.
///
/// Separate from `GgufMetadata::parse` by construction, not by discipline:
/// R1 requires that reading metadata cannot fail on tensor content, and a
/// caller who never calls this function cannot be failed by it.
///
/// # Errors
///
/// [`GgufError::Truncated`] or [`GgufError::Malformed`] with
/// `Stage::TensorDirectory` if a record is unreadable. A tensor whose TYPE
/// this build cannot resolve is not an error — it is omitted and reported.
pub fn parse_tensors<'a>(
    bytes: &'a [u8],
    meta: &GgufMetadata<'a>,
    origin: &str,
) -> Result<(GgufTensors<'a>, Report), GgufError> {
    let mut cursor = Cursor::new(bytes);
    cursor.seek(meta.kv_end()).map_err(|t| GgufError::Truncated {
        stage: Stage::TensorDirectory,
        offset: meta.kv_end(),
        needed: t.needed,
        available: t.available,
    })?;

    let mut report = Report::new();
    let mut raws = Vec::new();
    for _ in 0..meta.header().tensor_count {
        raws.push(read_info(&mut cursor)?);
    }
    let dir_end = cursor.pos();

    // NOT validated against the file length. A file with no tensors ends at
    // `dir_end` and this value can point one alignment block past the end —
    // 18 of 28 corpus files are that shape. Validating here would refuse
    // every vocab-only GGUF, which is exactly what a metadata consumer
    // opens. The bound that matters is per-tensor, in `tensor_bytes`.
    let data_start = data_start(dir_end, meta.alignment()).ok_or(GgufError::Malformed {
        stage: Stage::TensorDirectory,
        offset: dir_end,
        detail: "the data region's start overflows".to_string(),
    })?;

    let mut descriptors = Vec::new();
    let mut index = std::collections::HashMap::new();
    for raw in &raws {
        let Some(d) = resolve(raw, data_start, origin, &mut report) else {
            continue;
        };
        if index.contains_key(&d.name) {
            // `TensorDeclined`, not `TensorEncoding`: this tensor's encoding
            // resolved perfectly well. See Task 0.
            report.push(Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: d.name.clone(),
                    reason: "declared more than once; the first occurrence is kept".to_string(),
                },
                origin: origin.to_string(),
            });
            continue;
        }
        index.insert(d.name.clone(), descriptors.len());
        descriptors.push(d);
    }

    Ok((
        GgufTensors {
            bytes,
            descriptors,
            index,
            data_start,
        },
        report,
    ))
}

impl TensorContainer for GgufTensors<'_> {
    fn tensors(&self) -> &[TensorDescriptor] {
        &self.descriptors
    }

    /// Indexed, not a scan. The corpus quants declare 272 tensors and a 70B
    /// declares about a thousand; `TensorContainer::tensor`'s own doc
    /// requires a format crate to override the default here.
    fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.index.get(name).map(|i| &self.descriptors[*i])
    }

    fn tensor_bytes(&self, descriptor: &TensorDescriptor) -> mlmf_core::Result<Cow<'_, [u8]>> {
        let start = usize::try_from(descriptor.bytes.start).map_err(|_| { /* see note */ })?;
        let end = usize::try_from(descriptor.bytes.end).map_err(|_| { /* see note */ })?;
        self.bytes
            .get(start..end)
            .map(Cow::Borrowed)
            .ok_or_else(|| { /* see note */ })
    }
}
```

**Note for the implementer, and this is a real gap you must close rather than guess at:** the three `/* see note */` arms need an `mlmf_core::Error`. Read `crates/mlmf-core/src/error.rs` and choose an existing `ErrorKind` variant that fits "a declared byte range lies outside the container's data", constructing it with `Error::from(ErrorKind::...)` as `mlmf-ggml`'s `geometry.rs` does. **If no variant fits, stop and report that** — inventing a poor fit here is worse than the plan being wrong, and adding a variant is an `mlmf-core` change this task does not have in scope.

- [ ] **Step 4: Run and watch them pass**

- [ ] **Step 5: Prove they can fail (AD-2)**

1. Validate `data_start` against `bytes.len()` at parse time.
   Expect: `a_file_with_no_tensors_opens_and_has_no_data_region` red. **This is the sabotage that reproduces a majority-of-corpus regression from one plausible line**, and it is the reason that test exists. Record it as such.
2. Replace the index lookup with `self.descriptors.iter().find(...)`.
   Expect: `lookup_is_indexed_rather_than_a_linear_scan` red on `index_len`. Note that the two `tensor()` assertions in that test STAY GREEN — a linear scan returns the right answers. The cost is the defect and only the structural assertion can see it.
3. Change `tensor_bytes` to return `Cow::Owned(self.bytes[start..end].to_vec())`.
   Expect: `tensor_bytes_returns_the_declared_range_and_borrows_it` red on the `matches!`, and the content assertion above it GREEN. Record which fires — the content is identical and only the borrow assertion distinguishes them.
4. Remove the duplicate-name check.
   Expect: `a_duplicate_tensor_name_keeps_the_first_and_reports_the_second` red on the whole-list comparison. A `tensor("t")` assertion alone would stay green, because the index keeps the first either way.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(gguf): GgufTensors, indexed lookup, and borrowed tensor bytes"
```

---

## Task 5: Overlap, and what a malformed directory costs

**Files:**
- Modify: `crates/mlmf-gguf/src/tensors.rs`

D4: overlap is reported, not refused. Two tensors sharing bytes is a malformed file, but each still has a well-defined range and the ones that do not overlap are still readable.

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn overlapping_tensors_are_reported_and_both_stay_readable() {
        // Refusing the open would make one bad tensor cost the whole file,
        // which is R1's argument one stage over. A consumer that cares can
        // read the report; a consumer that wants the other 271 tensors gets
        // them.
        let bytes = gguf_with_tensors(
            &[("a", &[16], 0, 0), ("b", &[16], 0, 32)],
            &[0u8; 128],
        );
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (t, report) = parse_tensors(&bytes, &m, "t.gguf").expect("the open survives");
        assert_eq!(t.tensors().len(), 2, "both are still declared");
        assert!(t.tensor_bytes(t.tensor("a").unwrap()).is_ok());
        assert!(t.tensor_bytes(t.tensor("b").unwrap()).is_ok());
        // The whole entry: `!report.is_empty()` cannot see the wrong
        // tensor blamed, and the overlap names TWO tensors of which only
        // one is the subject.
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: "b".into(),
                    reason: "byte range overlaps a".into(),
                },
                origin: "t.gguf".into(),
            }]
        );
    }

    #[test]
    fn adjacent_tensors_are_not_an_overlap() {
        // The off-by-one that would make this test necessary is the whole
        // reason it exists: `a` ends at 64 and `b` starts at 64, which is
        // the NORMAL layout of every real file. A `>=` where a `>` belongs
        // reports every tensor in the corpus as overlapping.
        let bytes = gguf_with_tensors(
            &[("a", &[16], 0, 0), ("b", &[16], 0, 64)],
            &[0u8; 128],
        );
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert!(report.is_empty(), "touching is not overlapping");
    }

    #[test]
    fn a_zero_length_tensor_never_overlaps() {
        // A rank-0 or empty tensor has start == end. Under a naive interval
        // test an empty range at the same offset as another tensor's start
        // reads as contained-in and reports a false overlap.
        let bytes = gguf_with_tensors(
            &[("empty", &[0], 0, 0), ("real", &[16], 0, 0)],
            &[0u8; 128],
        );
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let (_, report) = parse_tensors(&bytes, &m, "t.gguf").unwrap();
        assert!(report.is_empty(), "an empty range overlaps nothing");
    }
```

- [ ] **Step 2: Run and watch them fail**

- [ ] **Step 3: Implement**

Add after the descriptor loop in `parse_tensors`:

```rust
    // Sort by start once and compare neighbours, rather than comparing every
    // pair. At 272 tensors the quadratic form is 37,000 comparisons and at a
    // thousand it is half a million, for a check that runs on every open.
    let mut order: Vec<usize> = (0..descriptors.len()).collect();
    order.sort_unstable_by_key(|i| descriptors[*i].bytes.start);
    for w in order.windows(2) {
        let (a, b) = (&descriptors[w[0]], &descriptors[w[1]]);
        // Strictly greater: `a.end == b.start` is adjacency, which is how
        // every real file is laid out. An empty range has start == end and
        // cannot exceed the next start.
        if a.bytes.end > b.bytes.start {
            report.push(Unrecognized {
                kind: UnrecognizedKind::TensorDeclined {
                    name: b.name.clone(),
                    reason: format!("byte range overlaps {}", a.name),
                },
                origin: origin.to_string(),
            });
        }
    }
```

**Note:** the overlap is reported as `TensorDeclined`, not
`TensorEncoding`. An overlapping tensor's encoding resolved fine and there
is no code to name; Task 0 added the variant for exactly this and for the
duplicate case in Task 4. An earlier draft of this plan reached for
`TensorEncoding` with `code: 0` at both sites, which is the same lie plan 3
spent its final task removing from `MetadataKey` — caught by auditing this
plan before dispatching it rather than by an implementer afterwards.

- [ ] **Step 4: Run and watch them pass**

- [ ] **Step 5: Prove they can fail (AD-2)**

1. Change `a.bytes.end > b.bytes.start` to `>=`.
   Expect: `adjacent_tensors_are_not_an_overlap` red, and `a_zero_length_tensor_never_overlaps` red. **Both**, and if only one fires the other is not testing what it claims.
2. Remove the sort.
   Expect: `overlapping_tensors_are_reported_and_both_stay_readable` — predict before running whether it fires. The fixture declares them in start order already, so it may NOT. If it does not, that is a control that cannot reach its assertion and the fixture needs a reversed pair; add one and say so.
3. Report the overlap as an `Err` instead.
   Expect: `overlapping_tensors_are_reported_and_both_stay_readable` red at `.expect("the open survives")`.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(gguf): report overlapping tensor ranges without refusing the file"
```

---

## Task 6: Authored adversarial fixtures

**Files:**
- Modify: `crates/mlmf-gguf/tests/fixture/mod.rs`, `crates/mlmf-gguf/tests/authored.rs`

The corpus cannot falsify most of this. It contains no unknown type codes, no overlaps, no duplicate tensor names, no non-UTF-8 tensor names — measured, by the same reader that produced the layout facts. Authored fixtures are the only instrument that can produce a negative.

- [ ] **Step 1: Extend the builder**

`GgufBuilder` gains `tensor(name, dims, code, offset)` and `data(bytes)`, mirroring the metadata methods. Keep the same discipline: **it emits what it is told, including malformed output.** It is not a writer.

- [ ] **Step 2: Write the adversarial tests**

At minimum, each with the whole-value assertion form:

- A tensor name with an embedded NUL, kept byte-exactly (names are byte-exact for the same reason metadata strings are).
- A tensor count that exceeds the file's capacity, truncated rather than hung.
- A directory that is well-formed but whose declared data region starts past the end of the file — **must open**, per Task 4.
- A tensor with a code from ggml's retired range, reported and omitted.
- Two tensors with the same name, first kept.
- A tensor whose offset is not a multiple of its encoding's alignment, so `TensorDescriptor::offset_alignment` reports something a consumer can act on.

- [ ] **Step 3: Prove they can fail (AD-2)**

Name the expected kill set before each. **Every fixture that asserts an out-of-range or absent outcome must carry a subsequent tensor**, for exactly the reason plan 3 recorded three times: without one, the range ends at the file end and the cursor's bounds check supplies the answer the assertion attributes to the code under test.

- [ ] **Step 4: Commit**

---

## Task 7: The corpus differential, extended to tensors

**Files:**
- Create: `crates/mlmf-gguf/tests/corpus-tensors.tsv`
- Modify: `crates/mlmf-gguf/tests/corpus.rs`

- [ ] **Step 1: Extract the fixture**

Extend the independent Python reader to emit, per file: relative path, tensor count, `dir_end`, `data_start`, first tensor name, first tensor's code, first tensor's declared offset, and the computed end of the last tensor. **The reader must remain independent** — do not generate this with `mlmf-gguf`, or an error shared between parser and expectations cannot be caught.

Use corpus-root-relative paths, not basenames: the corpus is not flat, and plan 3 lost a task-day to exactly that.

- [ ] **Step 2: Extend the differential**

`the_corpus_agrees_or_says_it_was_not_there` gains tensor assertions, compared as whole tuples. Keep the loud SKIPPED notice on the `std::io::stderr()` handle — **not `eprintln!`**, which libtest captures for a passing test, measured in plan 3.

The row count assertion must be EXACT and the version mix must be exact, for the reason plan 3's Task 9 recorded.

- [ ] **Step 3: Prove they can fail (AD-2)**

1. Corrupt one `data_start` in the TSV. Expect red naming that file.
2. Replace the differential's loop body with `continue`. Expect red on the walked-every-row assertion.
3. Rename the corpus directory. Expect PASS with a visible SKIPPED line — read the output and confirm.

- [ ] **Step 4: Commit**

---

## Task 8: Whole-branch review

**Not optional, and not a formality.** Every Important finding in plan 3 lived in the last third of the plan, in the seams between tasks. Tasks 3-5 of that plan survived every mutation a reviewer could construct. Per-task review worked early and thinned as the surface grew.

- [ ] **Step 1: Dispatch a review over the whole plan-4 diff**, with the priorities plan 3's final pass used: cross-file contradictions first, then tests that cannot fail, then claims of impossibility, then charter violations, then whether the public API reads as one thing or as eight tasks stitched together.

- [ ] **Step 2: Verify every Important finding yourself before fixing it.** Two of plan 3's reviewer findings needed reproduction before they could be acted on, and one of my own "fixes" for a review finding was itself a control that could not fail.

- [ ] **Step 3: Record the rate.** Count the controls in this plan that could not reach the assertion they named. Plan 3's number was eleven, ten of them mine. **If plan 4's rate has not improved, say so plainly and say what that implies** — a rate that does not move when you add controls is telling you the controls are not the binding constraint.

- [ ] **Step 4: Commit and push.**

---

## Deliberately not in this plan

- **Writing GGUF.** The builder emits malformed output on request; a writer must not.
- **GGUF v1.** Still refused by number. Supporting it means deriving the layout from the one v1 file in the corpus, which is its own plan.
- **Any interpretation of tensor names.** No layer indices, no `blk.N` parsing, no architecture inference. A name is an opaque key.
- **Streaming or ranged reads.** `mlmf-core`'s `RangedSource` exists for it; this crate takes `&[u8]`.
- **Validating that a tensor's data is well-formed.** MLMF hands out a byte range. What the bytes mean is the consumer's.
- **`mlmf-safetensors`.** Its `F8_E4M3`/`F8_E5M2` mapping must be pinned arm-by-arm when it is written — same width, same kind, mutually byte-incompatible.
