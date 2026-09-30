# Fuel/Lightbulb → MLMF migration inventory

Written 2026-09-30. Answers CireSnave's standing rule (`CIRESNAVE-EXPECTATIONS.md`
§5.1-mlmf-owns-loading), verbatim: *"Everything directly relating to Machine
Learning Model File loading and saving should be in MLMF. That means Fuel
should be using MLMF for loading and saving any model-related files."*

**Refs measured against** (fetched today, all `origin/main`):

| repo | SHA |
|---|---|
| mlmf | `8b05acab1a0875b2ee774c12fc9b106ca0757bb5` (mlmf `0.5.0`) |
| fuel | `5bd79337710354e29567568e5ebcb9026005cdf8` |
| lightbulb | `285566da4c727d3b1540e2fbc48e1ba67eaabd11` |

All three read via `git -C <repo> show origin/main:<path>` / `git -C <repo>
ls-tree -r --name-only origin/main`, never a working tree — per portfolio
CLAUDE.md §2.

**Positive control for "nothing depends on mlmf"**: `git -C fuel grep -l
mlmf origin/main -- '*.toml'` → 0 hits; `git -C fuel grep -l safetensors
origin/main -- '*.toml'` → 9 hits (a dependency actually present). Same
control on lightbulb: `mlmf` → 0 hits, `candlelight` → 1 hit (its Cargo.toml).
The query works and finds zero, not "the query is broken." **Confirms
HANDOFF's claim: nothing in fuel or lightbulb depends on mlmf today.**

---

## 0. The boundary that shapes this whole doc

MLMF's charter (`CLAUDE.md`): *"MLMF's job is to read and write model files.
MLMF is never intended to be an interpreter of the content of model files.
That is for things like Fuel."* Everything below sorts capabilities by that
line — a format's byte layout is in scope; deciding what a weight or an
activation **means** (fuel-onnx's `lazy_eval*` compute graph, Lightbulb's
inference engine) is not, and stays where it is.

The §6 fence (`CLAUDE.md` §1–2) also binds any *proposed* mlmf API below: a
migrated config reader may supply a documented format default, never a
model's value it didn't read, and every field a format may omit must be
`Option<T>` — not `unwrap_or(hardcoded)`.

---

## 1. Format inventory

### 1.1 GGUF container + quantization types

| capability | mlmf | fuel/lightbulb |
|---|---|---|
| GGUF v2/v3 container parse | **HAS** — `crates/mlmf-gguf/src/{header,metadata,tensors,value}.rs` | **HAS** — `fuel-formats/src/gguf.rs` (both v2/v3); Lightbulb's own parser, `src/gguf/parser.rs` |
| GGUF v1 container parse | **MISSING** — deferred, `crates/mlmf-gguf/tests/v1_deferral.rs` | fuel: **HAS** (`fuel-formats/src/gguf.rs:50` reads `GgufV1`); Lightbulb: **MISSING** (`src/gguf/mod.rs:44-51` — its own parser refuses v1, falls back to candlelight which reads it) |
| GGML block-quant types (classic 15: F32/F16/BF16, Q4_0..Q8_1, Q2K..Q8K) | **HAS** — `crates/mlmf-ggml/src/types.rs` | **HAS** — `fuel-ir/src/quantized.rs:33-49 GgmlDType` |
| IQ*-family quant types (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M) | **HAS** — `crates/mlmf-ggml/src/types.rs:70-92` (all 9, plus geometry in `row()`) | **MISSING** — `fuel-ir/src/quantized.rs:33` doc comment: *"the classic 15 ... NOT the IQ\*/TQ\*/MXFP4/NVFP4 families"*; `from_u32` (`:52-71`) errors on every IQ code. This is **GAP-211**, open — see §3 |
| TQ1_0/TQ2_0 (ternary), MXFP4, NVFP4, Q1_0, Q2_0 | **HAS** — `crates/mlmf-ggml/src/types.rs:104-117` | **MISSING** — same enum, same gap |
| Malformed-file robustness (alignment=0, non-power-of-two, unbounded allocations from declared lengths) | **HAS** — `crates/mlmf-gguf/src/metadata.rs:235-251` reports-and-falls-back on non-power-of-two/zero alignment (never panics); `crates/mlmf-core/src/align.rs` has no unbounded-allocation-from-declared-length path (bounds checked against a `Cursor`) | fuel: **PARTIALLY FIXED** — this is **GAP-205**, `PANIC FIXED abf478e5`, `HARDENING DONE 44193322` (bounds now on declared lengths, not counts); 2 semantics items (see next row) still open by ruling |
| String decode fidelity (non-UTF-8 tensor/token names, trailing NULs) | **HAS** — `crates/mlmf-gguf/src/value.rs:307-320`: valid UTF-8 → `MetaValue::String`, invalid → `MetaValue::Bytes(Vec<u8>)`, **never `from_utf8_lossy`** (tested: `crates/mlmf-gguf/src/value.rs:496-506,517-522`) | fuel: **KNOWN GAP, documented-not-fixed by ruling** — GAP-205 row 2: `fuel-formats/src/gguf.rs:91-101` strips trailing NULs then `from_utf8_lossy`, corrupting non-UTF-8 tensor names. GAP-205's own text: *"MLMF HAS ALREADY SOLVED BOTH IN THE REPLACEMENT... `MetaValue::Bytes(Vec<u8>)` for (1)"* — confirmed true at this SHA |
| Duplicate metadata-key / tensor-name handling | **HAS** — `Vec<Entry>` not `HashMap`, first-wins with a reported finding: `crates/mlmf-gguf/src/metadata.rs:734` test `a_duplicate_key_keeps_the_first_and_reports_the_second` | fuel: **KNOWN GAP, documented-not-fixed by ruling** — GAP-205 row 2: `fuel-formats/src/gguf.rs:380-381` uses `HashMap`, silently last-wins, discards declared order |
| Unrecognized tensor dtype aborting the whole read vs. degrading gracefully | **HAS the structural fix** — `crates/mlmf-gguf/src/tensors.rs:214-216`: an unrecognized `GgmlType::from_code` **reports and continues**, does not abort; metadata (including `chat_template`) is fully readable even when a tensor dtype is unrecognized | fuel: **THIS IS GAP-211's structural half, still open** — `Content::read` hits `GgmlDType::from_u32(...)?` in the per-tensor loop and aborts the whole read, discarding already-decoded metadata. GAP-211 notes a metadata-only entry point is *assemblable* today from `pub` primitives (`VersionedMagic::read`, `read_string`, `ValueType::from_u32`, `Value::read`, all re-exported at `fuel-core/src/quantized/gguf_file.rs:19-21`) but none is *exposed* as one call |

**Note on write-path fidelity** (part of GAP-205, deferred): fuel's
`fuel-formats/src/gguf.rs:380` write path always emits u64 widths (nothing
written reads back as `GgufV1`), fabricates an empty array's element type as
`U32` (`:354`), and refuses mixed-type arrays outright (`:357-361`) — which
is exactly the shape a non-UTF-8 token in a 128k vocabulary produces. MLMF's
`mlmf-gguf` crate currently has **no write path at all** (see §2) so this
comparison has no current mlmf counterpart to check against; flagged here so
it is not lost when write support is built.

### 1.2 safetensors

| capability | mlmf | fuel |
|---|---|---|
| Header/metadata parse | **HAS** — `crates/mlmf-safetensors/src/{header,metadata}.rs` | **HAS** — `fuel-formats/src/safetensors.rs`, `fuel-core/src/safetensors.rs` |
| Tensor byte-range parse | **HAS** — `crates/mlmf-safetensors/src/tensors.rs` | **HAS** — `fuel-loaders/src/safetensors.rs` |
| mmap-backed zero-copy tensor access | **HAS the primitive, not yet wired into mlmf-safetensors** — `crates/mlmf-source-file/src/file.rs:101-190` (`SourceFile::open`/`open_mmap`, real `memmap2::Mmap`, feature `mmap` on by default) + `crates/mlmf-core/src/align.rs:72-86` (`try_as_slice::<T>` — fallible borrow, **deliberately no infallible accessor** — and `to_aligned_vec` as the named copy escape hatch). `mlmf-safetensors` itself does not yet call these; no `mmap`/`try_as_slice` reference found in `crates/mlmf-safetensors/src/tensors.rs` | **MISSING in practice — this is GAP-204.** `fuel-core/src/lazy.rs` (`load_tensor_as_f32`, `:10309`) eagerly byte-decodes every tensor into an owned `Vec`/`Arc<[f32]>`; the `Mmap` is a **local, dropped after decode**, so OS page-sharing never survives the load regardless of alignment. GAP-204 also measured BF16/F16/F64 tensors are upcast to f32 unconditionally — 2x RAM vs. disk for BF16 checkpoints, which dominates the memory story more than the copy itself |
| Endianness | safetensors is little-endian by spec; both sides correct (not at issue for this format) | same |

**Why this matters for the migration**: mlmf already has the exact primitive
GAP-204's fix needs (`try_as_slice` borrow-or-named-copy, endianness kept
separate from alignment per the design spec's OQ-3 discussion) sitting one
crate over in `mlmf-core`. `mlmf-safetensors` reading tensor bytes through
`try_as_slice` instead of a decode loop is a small, low-risk change that
would let Fuel adopt real zero-copy — but only if Fuel's own loader also
stops dropping the `Mmap` and stops force-upcasting to f32, which are
Fuel-side architecture decisions outside mlmf's charter to fix.

### 1.3 ONNX

| capability | mlmf | fuel |
|---|---|---|
| Container/protobuf parse | **HAS** — `src/formats/onnx_import.rs`, `src/formats/onnx_export.rs` (legacy root crate, see §2 caveat) | **HAS, more extensive** — `fuel-onnx/` is a dedicated crate with its own `.proto3`, `build.rs` codegen |
| Graph *evaluation* (lazy_eval, conv/norm ops) | **OUT OF SCOPE by charter** — MLMF reads/writes files, doesn't interpret content | **HAS** — `fuel-onnx/src/lazy_eval*.rs`. This is compute, not file I/O; stays in Fuel regardless of the migration rule |

ONNX is the clearest case where the charter boundary actually narrows the
migration: only the container/protobuf-parse half of `fuel-onnx` is a
migration candidate; the eval half is explicitly not model-file loading.

### 1.4 imatrix — two distinct wire formats, not one

Per HANDOFF, confirmed at this SHA:

- `crates/mlmf-gguf/src/imatrix.rs` (`#94`, `pub fn read` at `:222`) — the
  **GGUF-embedded** imatrix layout (an imatrix stored as ordinary GGUF KV
  pairs/tensors inside a `.gguf` container).
- `fuel-formats/src/imatrix.rs` — a **different, raw positional** layout:
  `i32` entry count, then per-entry `i32` name-length + UTF-8 name + `i32
  ncall` + `i32 nval` + `nval × f32`, no magic, no version field. This is
  bartowski's `.imatrix` sidecar format (real, currently-distributed files
  use it) and is unrelated to the GGUF-embedded layout mlmf reads.

**mlmf has no reader for the raw positional layout.** This is not a partial
overlap — it is two different file formats that happen to share a name.
Fuel's `fuel-formats/src/imatrix.rs` should stay Fuel's for this format
until/unless a decision is made to bring it into mlmf as a second, distinct
parser (not a merge with `mlmf-gguf::imatrix`).

---

## 2. The legacy-root-crate caveat (applies to §1's GGUF, ONNX, and all of §4)

MLMF is mid-split between a **legacy root crate** (`src/`, package name
`mlmf`) and a family of **backend-agnostic crates**
(`mlmf-core`/`mlmf-gguf`/`mlmf-safetensors`/`mlmf-ggml`/`mlmf-meta`/
`mlmf-hf-layout`/`mlmf-source-file`). The workspace `Cargo.toml` says this
plainly: the legacy package is a `default-members` entry kept alive
specifically because deleting it "would let it rot unobserved," and "spec
§11 schedules it for a REWRITE."

**Everything with write/save capability today lives in the legacy crate**,
not the tested backend-agnostic crates:

- `src/saver.rs` (`save_model`, `save_safetensors`, `save_gguf`)
- `src/formats/gguf_export.rs`, `onnx_export.rs`, `pytorch_export.rs`,
  `safetensors_export.rs`, `awq_export.rs`
- `src/name_mapping.rs` (`build_llama_map`, the #96/#97-fixed tensor-name
  translation) and `src/smart_mapping.rs` (`SmartTensorNameMapper`)

Positive control: `mlmf-gguf` and `mlmf-safetensors` (backend-agnostic) —
`git -C mlmf grep -n "pub fn write\|fn save" origin/main --
'crates/mlmf-gguf/src/*.rs' 'crates/mlmf-safetensors/src/*.rs'` → **0 hits**.
Same query against `src/saver.rs` → 3 hits (`save_model`, `save_safetensors`,
`save_gguf`). The query works; the backend-agnostic crates genuinely have no
write path yet.

**Implication for the migration plan**: a Fuel caller adopting mlmf's
*reading* capability today lands on crates this session has spent its
verification budget hardening (§4 of this repo's `CLAUDE.md`: sabotage
testing, corpus gating, mutation-verified guards). A Fuel caller adopting
mlmf's *saving* or *name-mapping* capability today lands on the legacy crate
that is explicitly scheduled for a rewrite and does not carry the same
verification discipline (no sabotage-tested guards found in `src/saver.rs`
or `src/name_mapping.rs`'s own test module beyond what #97/#98/#100 added
this session). Migrating Fuel onto code about to be rewritten means
migrating twice. This is why the migration plan (§6) sequences write/mapping
capability after the backend-agnostic crates gain it, not before.

---

## 3. Fuel's known loader bugs (GAP-204/205/206/211) — status against mlmf today

Source: `fuel/docs/gaps.md` at `origin/main` (fuel's internal gap tracker,
not GitHub issues — confirmed by `gh issue list` search returning nothing
for these strings and `docs/gaps.md` containing full entries).

| GAP | bug | fuel status at this SHA | does mlmf's equivalent path avoid it? |
|---|---|---|---|
| **GAP-204** | Safetensors loader copies every tensor element-by-element; mmap's page-sharing benefit is thrown away, and BF16/F16/F64 are eagerly upcast to f32 (2x RAM) | **OPEN** — root-caused (dominant cost is the f32 upcast + eager residency, not the copy itself); "highest-leverage lever is the dtype/residency question — needs a ruling, not a fix" | **Partially** — mlmf has the `try_as_slice`/mmap primitives (§1.2) that would fix the *alignment/copy* half if wired into `mlmf-safetensors`, but has no opinion on eager-upcast/residency (that's a caller/runtime decision, arguably out of MLMF's read/write charter) |
| **GAP-205** | `fuel-formats/src/gguf.rs` panics on `general.alignment = 0` (div-by-zero abort); 3 unbounded allocations from declared lengths; `from_utf8_lossy` string corruption; `HashMap` last-wins on duplicate keys; write path not a faithful inverse | **PARTIAL** — panic fixed (`abf478e5`), allocation hardening done (`44193322`); the two string/duplicate-key semantics items **documented-not-fixed by ruling**, deferred to mlmf; write-path fidelity out of scope | **Yes, for the 2 deferred items** — `mlmf-gguf` never panics on bad alignment (reports+falls back, §1.1), bounds declared-length reads, uses `Bytes` not lossy UTF-8, and uses order-preserving first-wins with a reported duplicate (§1.1). Write-path fidelity has no mlmf counterpart yet (§2) |
| **GAP-206** | `quantized-qwen3-moe` example computes `vocab_size` from the wrong tensor dimension (contradicts itself in its own comments 14 lines apart) | **OPEN** — live wrong value in a shipped example, independent of the MLMF port | Not directly comparable — this is a Fuel *example's* misreading of an already-correctly-parsed dimension order, not a container-parse bug. mlmf's `mlmf-core::shape` module uses named `DimOrder::{GgmlNe, RowMajor}` + `reorder()` rather than a bare `reverse_dims()`, specifically to avoid this class of "nearest comment wins" error — cited in GAP-206 itself as the argument for that API-naming choice |
| **GAP-211** | `GgmlDType::from_u32` rejects every IQ quantization; because tensor-type parsing runs after metadata and aborts the whole read on failure, an IQ-quantized GGUF cannot be opened **at all**, including its metadata | **OPEN** — not blocking Lightbulb today (its checkpoint is Q4_0); "the `read_metadata` split is the durable fix and should be sequenced with the MLMF seam" | **Yes, both halves** — `mlmf-ggml` has all IQ*/TQ*/MXFP4/NVFP4 types (§1.1), and `mlmf-gguf`'s tensor loop reports-and-continues on an unrecognized dtype rather than aborting metadata read (§1.1). GAP-211's own text already names this as the intended fix path |

**Reading this table**: mlmf already avoids 2 of GAP-205's semantics items
and both halves of GAP-211 — not by coincidence, but because those two GAPs
were raised *from* mlmf's own port reconnaissance and design work, per
`docs/gaps.md`'s own attribution ("FOUND 2026-08-14 by MLMF's port
reconnaissance", "REPORTED 2026-08-15 by the Lightbulb architect"). GAP-204
is only partially addressed (the primitive exists, isn't wired, and doesn't
touch the dominant residency/upcast cost). GAP-206 isn't an mlmf-shaped bug
at all.

---

## 4. Metadata files

| file | mlmf | fuel/lightbulb |
|---|---|---|
| `config.json` (`ModelConfig`) | **HAS** — `crates/mlmf-core/src/meta.rs`, consumed via `MetadataSource` trait; `crates/mlmf-hf-layout/src/shards.rs`. Known limitation tracked separately: `ModelConfig` cannot yet represent an absent field per the `Option<T>` standing policy — **#48**, open | fuel: `fuel-loaders/src/hf_config.rs`, `fuel-loaders/src/quantized/config_from_gguf.rs`; own `HfConfig` struct, own field derivation |
| `tokenizer.json` | **HAS** — `crates/mlmf-meta/src/vocab.rs` | fuel: `fuel-loaders/src/quantized/tokenizer.rs` |
| `tokenizer_config.json` + `chat_template` | **HAS** — `crates/mlmf-meta/src/template.rs` (positive control: `git grep tokenizer_config` → hits in `crates/mlmf-meta/src/vocab.rs` and its test file; the query is not silently missing files) | fuel: chat-template handling exists in `fuel-loaders/src/quantized/tokenizer.rs` |
| `generation_config.json` | **MISSING, confirmed** — `git -C mlmf grep -rn "generation_config" origin/main -- '*.rs'` → **0 hits**, same query for `tokenizer_config` → hits (positive control the query itself works) | not directly checked in fuel; not mlmf's job to have today |
| `special_tokens_map.json` | **MISSING, confirmed** — same grep, **0 hits** | not directly checked |

**§6-fence check on fuel's config reading**: `fuel-loaders/src/quantized/
config_from_gguf.rs:210` reads `rope_theta` as `m.f32("rope.freq_base")?` —
the `?` propagates an error when the key is absent; it does **not**
fall back to `10000.0`. The two `10000.0` literals in that file (`:319`,
`:366`) are inside test fixtures (`metadata.insert(...)`) constructing
synthetic GGUF metadata for a test, not a production fallback. **Fuel's own
GGUF config path already refuses rather than fabricates on this field** —
a good sign that migrating this reader onto mlmf's `MetadataSource` won't be
fighting an existing fabricated-default habit here. `fuel-loaders/src/
hf_config.rs:50,137,206` does use `unwrap_or` for `head_dim`/`num_kv_heads`
— but these derive from a **documented architectural convention** (`hidden
/ heads` when unspecified), which is the §6-permitted "format default" half,
not a model value.

---

## 5. Name mapping

mlmf's tensor-name → schema-field translation lives in the legacy crate
(§2): `src/name_mapping.rs` (`TensorNameMapper::build_llama_map`, fixed this
session by #96/#97 to hard-error on every unrecognized component rather than
silently discarding translation) and `src/smart_mapping.rs`
(`SmartTensorNameMapper`, oracle-based fuzzy mapping with
`ChatBasedOracle`).

Fuel's nearest equivalent is **architecture detection**, not per-tensor
schema mapping: `fuel-loaders/src/quantized/arch.rs` classifies a GGUF file
into a family (`Llama`, `Qwen2`, `Qwen3`, `Qwen3Moe`, `Phi`, `Phi3`, `Gemma`,
`Gemma3`, `Glm4`, `Lfm2`, `SmolLm3`, `Gpt2`, `GptNeoX`, `Unknown`) from
`general.architecture` or, failing that, tensor-name pattern-matching — used
to pick which of Fuel's ~10 `quantized_*` model loaders to hand the file to.
It answers "which loader?", not "what does tensor X become in my target
schema?" — a coarser question than mlmf's per-component map. **These are not
overlapping implementations of the same capability**; Fuel would gain, not
duplicate, by adopting mlmf's finer-grained map once it exists outside the
legacy crate.

---

## 6. Lightbulb's own GGUF module vs. `mlmf-gguf`

`lightbulb/src/gguf/mod.rs` (3396 lines) + `src/gguf/parser.rs`: a
from-scratch mmap-backed GGUF v2/v3 parser plus a `candlelight`-parsed
fallback content, kept side by side deliberately — its own module doc
explains that its parser and candle's `quantized::gguf_file` refuse
*different* files (candle reads v1, Lightbulb's parser doesn't), and running
them as an `AND` instead of an `OR` was previously discarding the union of
their coverage. Lightbulb also carries its own dtype census
(`tests/gguf_dtype_census.rs`) whose header comment states *"the mlmf lane
reached the same conclusion within the same ten minutes, through
`mlmf-gguf`'s own reader"* — i.e. Lightbulb and mlmf have already
cross-checked findings on the same corpus without Lightbulb depending on
mlmf's code.

This is a real overlap: Lightbulb's parser and `mlmf-gguf` solve the same
container-parse problem independently, with Lightbulb specifically adding
mmap zero-copy tensor slicing on top (a capability `mlmf-gguf` doesn't yet
expose — mlmf-source-file has the mmap primitive per §1.2, but `mlmf-gguf`'s
tensor-data API returns descriptors/offsets, not yet mmap-backed slices).
Lightbulb has **zero** `mlmf` dependency today (§ front-matter positive
control). Migrating Lightbulb's GGUF reading onto `mlmf-gguf` would delete
~3400 lines of parallel maintenance, provided `mlmf-gguf` gains a zero-copy
tensor-slice accessor first (see §7's proposed API).

---

## 7. Proposed mlmf API surface for Fuel/Lightbulb to call

Grouped by what already exists (Fuel/Lightbulb would just change their
caller) vs. what is genuinely new mlmf work.

### 7.1 Already exists — caller-side change only

```rust
// GGUF metadata + tensor directory read, no fabricated values,
// no panic on malformed alignment, no lossy string corruption,
// no whole-read abort on an unrecognized tensor dtype.
mlmf_gguf::GgufMetadata::parse(bytes: &[u8], origin: &str)
    -> Result<(GgufMetadata, Report), GgufError>;
mlmf_gguf::tensors::read_tensor_directory(...) -> (Vec<TensorDescriptor>, Report);

// The dtype/geometry table Fuel's GAP-211 needs (all 35 ggml codes incl. IQ*/TQ*/MXFP4/NVFP4)
mlmf_ggml::GgmlType::from_code(code: u32) -> Option<GgmlType>;
mlmf_ggml::GgmlType::row(self) -> Row; // elements/block, bytes/block, alignment

// The GAP-204 borrow-or-copy primitive
mlmf_core::align::try_as_slice::<T: Pod>(bytes: &[u8]) -> Result<&[T]>;
mlmf_core::align::to_aligned_vec::<T: Pod>(bytes: &[u8]) -> Result<Vec<T>>;

// Real mmap-backed file access
mlmf_source_file::SourceFile::open(path: &Path) -> Result<SourceFile>; // mmap by default

// safetensors header/tensor-range read
mlmf_safetensors::header::parse(...) -> Result<SafetensorsHeader>;

// GGUF-embedded imatrix (NOT the raw positional format Fuel already reads — see §1.4)
mlmf_gguf::imatrix::read(...) -> Result<Imatrix>;
```

### 7.2 New mlmf work required

1. **A metadata-only GGUF read entry point that never touches the tensor
   loop** — `mlmf-gguf` doesn't need this fix (§1.1: it already reports and
   continues on an unrecognized dtype rather than aborting), but Fuel's
   GAP-211 discussion frames this as the shape a consuming API should have;
   exposing `mlmf_gguf::metadata_only(bytes) -> Result<GgufMetadata>` as a
   named, blessed call (rather than "assemble it from four pub primitives",
   which is what Fuel does today) removes the temptation for a future
   caller to reintroduce the abort-on-tensor-loop shape.
2. **mmap-backed zero-copy tensor slicing in `mlmf-gguf`/`mlmf-safetensors`**
   — today `mlmf-source-file` has the mmap and `mlmf-core` has the aligned
   borrow, but nothing in `mlmf-gguf`/`mlmf-safetensors` wires a
   `TensorDescriptor` to `try_as_slice` over an open `SourceFile`. This is
   the blocking gap for both Lightbulb's zero-copy tensor access (§6) and a
   real fix for Fuel's GAP-204 alignment/copy half.
3. **Saving/writing, promoted out of the legacy crate** — `save_gguf`,
   `save_safetensors` exist only in `src/saver.rs` today (§2). Fuel adopting
   mlmf for *writing* model files (the second half of CireSnave's rule)
   needs this capability in a backend-agnostic, tested crate — plausibly a
   new `mlmf-gguf-write`/`mlmf-safetensors-write` module or crate, built to
   the same discipline (sabotage-tested guards, no fabricated values) as the
   read side, not a straight port of `src/saver.rs`.
4. **Tensor-name → schema mapping, promoted out of the legacy crate** —
   `TensorNameMapper`/`SmartTensorNameMapper` (§5) are the capability Fuel's
   `arch.rs` doesn't have (per-component mapping, not just family
   detection), but they live in the crate scheduled for a rewrite. This
   needs the same crate-promotion treatment as saving before Fuel should
   depend on it.
5. **`generation_config.json` and `special_tokens_map.json` readers** —
   confirmed missing (§4), needed if Fuel wants to retire its own
   equivalents (not directly checked in fuel this session — a follow-up
   grep on `fuel-loaders/src/hf_config.rs` and friends against these two
   filenames would confirm whether Fuel currently reads them at all before
   this is prioritized).
6. **imatrix raw positional format reader**, as a second parser distinct
   from `mlmf_gguf::imatrix` (§1.4) — only if/when a decision is made to
   bring `fuel-formats/src/imatrix.rs`'s format into mlmf; HANDOFF's stance
   is that format stays Fuel's for now.

---

## 8. Ordered migration plan

Ordered by **lowest risk / already-equivalent capability first**, and by
**not asking Fuel to depend on the legacy crate before the legacy crate's
capability is promoted**.

1. **GGUF metadata + dtype table (`mlmf-gguf` + `mlmf-ggml`) → Fuel's
   quantized-model config/arch path.** Zero new mlmf work required (§7.1);
   directly fixes GAP-211 (IQ* support, metadata survives an unrecognized
   dtype) and 2 of GAP-205's deferred semantics items for whichever Fuel
   code path is repointed. Lowest risk: mlmf's GGUF read side has this
   session's heaviest verification investment (sabotage-tested, corpus- and
   fixture-gated).
2. **Lightbulb's GGUF reader → `mlmf-gguf`**, gated on item 3 below
   (mmap-backed tensor slicing) landing first — otherwise Lightbulb takes a
   real perf regression (its current zero-copy path would become a copying
   one). Deletes ~3400 lines of parallel implementation (§6) once unblocked.
3. **mmap-backed zero-copy tensor slicing, new mlmf work (§7.2 item 2).**
   Prerequisite for item 2 and for a real GAP-204 fix. Bounded scope: wire
   existing `mlmf-source-file`/`mlmf-core` primitives into `mlmf-gguf` and
   `mlmf-safetensors`'s tensor-access API; no new format knowledge needed.
4. **safetensors tensor read via `try_as_slice` → Fuel's safetensors
   loader**, once item 3 lands. This only fixes GAP-204's alignment/copy
   half; the dominant BF16-upcast/eager-residency cost GAP-204 identified is
   a Fuel-side runtime/serving decision outside this migration's scope —
   flag it back to Fuel as still needing its own ruling.
5. **Metadata-file readers (`config.json`, `tokenizer.json`,
   `tokenizer_config.json`+template) → Fuel's `hf_config.rs`/`tokenizer.rs`**,
   gated on **#48** (`ModelConfig`'s `Option<T>` representation) resolving
   first — migrating Fuel onto a `ModelConfig` that still can't represent an
   absent field would carry the exact defect class MLMF has spent this
   session's history removing into a new consumer.
6. **Saving (write side), gated on crate promotion (§7.2 item 3).** Do not
   point Fuel's save path at `src/saver.rs` — it is scheduled for a rewrite
   (§2) and migrating onto it now means migrating again later. Sequence
   after the backend-agnostic crates gain write support.
7. **Name mapping, gated on crate promotion (§7.2 item 4).** Same reasoning
   as item 6 — `name_mapping.rs`/`smart_mapping.rs` are legacy-crate code
   today. Lowest priority of the ordered items because Fuel's `arch.rs`
   already covers its actual current need (loader dispatch), and the finer
   per-tensor mapping mlmf offers isn't yet something Fuel has asked for.
8. **`generation_config.json`/`special_tokens_map.json` readers** — new
   mlmf work with no confirmed Fuel consumer yet (§7.2 item 5); lowest
   priority until a caller need is measured, consistent with how this repo
   already treats #52 (deferred with a trigger, not built speculatively).

**Not migrating**: fuel-onnx's `lazy_eval*` compute graph (§1.3, out of
charter), `fuel-formats/src/imatrix.rs`'s raw positional format (§1.4,
different format, stays Fuel's per HANDOFF), GAP-206 (not an mlmf-shaped
bug, fixable in Fuel's example independent of this migration).
