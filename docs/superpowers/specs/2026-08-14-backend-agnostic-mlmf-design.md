# Backend-Agnostic MLMF — Design

**Date:** 2026-08-14
**Status:** Design approved section-by-section. **Rev 2** — revised after review by the Fuel architect: §4.6 (alignment) added, C1 demoted to a backstop, §9 §7 (adoption gates) added, OQ-1's premise corrected. Awaiting Eric's sign-off before an implementation plan is written.
**Participants:** Eric Evans (owner, charter); MLMF agent (author); Fuel architect and Lightbulb architect (consulted, constraints and measurements); Unpopped (boundary clearance, byte-exactness hazard).

Every number in this document was measured on 2026-08-14 unless attributed otherwise. Claims taken from another project are attributed inline, because several of them are load-bearing and their owners should be able to check them.

---

## 1. Charter

> **MLMF's job is to read and write model files. MLMF is never intended to be an interpreter of the content of model files. That is for things like Fuel, who have a far deeper understanding of the ecosystem and the intended use of the model. Any interpretation MLMF did would likely get in the way of projects like Fuel more than it would help.**
> — Eric Evans

Two corollaries that decide most of the hard cases below:

- **Extraction is in; inference is out.** Reading what a file *declares* is MLMF's job. Guessing what a file *means* is the consumer's.
- **What is in the file is core; how we obtain it is not.** Storage access — filesystem, mmap, network — is a necessary means, not part of MLMF's subject matter, and therefore does not belong in the crates that model file content.

---

## 2. Why: the measured problem

MLMF v0.3.0 compiles clean under `--all-features` in 37s (293 warnings, nearly all missing-docs from generated ONNX). The `errors.txt` at the repo root is stale and will be deleted. **The problem is shape, not rot.**

| | MLMF | `fuel-formats` |
|---|---:|---:|
| `cargo tree` nodes, default features | **168** | 45 |
| `cargo tree` nodes, **all features off** | **159** | 45 |
| Source LOC | 24,124 | 1,693 |

*(`fuel-formats` declares `default = []`, so both rows are the same measurement. Node counts include the crate itself and count duplicate major versions separately — Fuel independently measured 43 transitive dependencies for the same crate, and 11 direct.)*

The second row is the argument. Turning off *every* MLMF feature removes **nine** crates, because `tokio` (`features = ["full"]`), `sysinfo`, `uuid`, `chrono`, `time` and `regex` are unconditional dependencies of the core. Fuel's stated bar — *"if Fuel's dependency is `mlmf` with 8 features off, I'll take it; if it's `mlmf`, I won't"* — was therefore unreachable by configuration. The floor is 159 for a crate whose job is parsing bytes.

Three further findings shaped the design:

1. **The LLM oracle is on the load path**, not adjacent to it: `LoadOptions.smart_mapping_oracle: Option<Box<dyn NameMappingOracle>>`, consumed at `src/loader.rs:1189` mid-load. Both consumers refused it outright — a load path that can call an LLM has non-deterministic startup, which makes every downstream measurement unreproducible.
2. **MLMF's PyTorch support is a stub.** `load_zip_pickle` and `load_legacy_pickle` (`src/formats/pytorch_loader.rs`) both unconditionally return `Err`, advising the user to convert the file in Python. All 407 lines are detection, options and progress around two functions that never parse anything. The README nonetheless advertises *"Comprehensive support for SafeTensors, GGUF, ONNX, PyTorch, and AWQ."*
3. **MLMF was dropped from Lightbulb because `prost`/`protoc` was in the default build** (Lightbulb architect). `src/loaders/mlmf_wrapper.rs` remains in Lightbulb's tree, orphaned — never re-declared as a module, does not compile, and calls an MLMF API (`mlmf::prelude`, `mlmf::callbacks`, `with_progress_callback`) that no longer exists. Lightbulb owns that deletion.

### 2.1 Why features cannot solve this

Cargo features are **additive and unify across the whole dependency graph**; a feature can never subtract. Lightbulb depends on `fuel-core` unconditionally, so if Fuel enabled an `mlmf/tensors` feature, that feature would unify into every Lightbulb build regardless of Lightbulb's own `default-features = false`.

This argument (Lightbulb architect) is the load-bearing one for the entire structure, and it applies recursively: a single `mlmf-formats` crate with an `onnx` feature would put `prost` and a `protoc` build script into every consumer's graph the moment anyone in it enabled ONNX. **Only a dependency edge expresses capability.** Per-format crates are not gold-plating; they are the only mechanism that delivers the isolation both consumers require.

---

## 3. Architecture

### 3.1 Two orthogonal axes

Formats and sources **compose** — any source × any format. Sources therefore depend on `mlmf-core` only and never on a format crate; adding S3, IPC or an in-memory source later touches nothing that parses.

```
  ┌── FORMAT AXIS — pure: bytes → structure. No I/O of any kind. ─────────┐
  │                                                                       │
  │   mlmf-core        vocabulary + traits.  ~15 deps (projected).        │
  │      ▲                   ▲                    ▲                       │
  │   mlmf-ggml        mlmf-hf-layout        mlmf-onnx                    │
  │    block-quant      index.json,           prost + protoc,             │
  │    geometry,        config.json,          fully isolated              │
  │    ggml naming      tokenizer discovery                               │
  │      ▲              ▲       ▲       ▲                                 │
  │   mlmf-gguf   mlmf-safetensors  mlmf-pickle  mlmf-awq                 │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘
                                   ×
  ┌── SOURCE AXIS — I/O only. No format knowledge. ──────────────────────┐
  │                                                                      │
  │   mlmf-source-file    filesystem + mmap                              │
  │   mlmf-source-hub     HuggingFace Hub fetch + cache                  │
  │                       (the only crate in the tree with a TLS edge)   │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

  mlmf-meta      key vocabulary + extraction of declared facts, over the
                 traits. No format dependencies, no I/O. This is what
                 Fuel and Lightbulb depend on.
  mlmf           batteries-included convenience: every format × file source.
  mlmf-oracle    offline mapping-file emitter. Unreachable from any load path.
```

### 3.2 The HF layout / HF acquisition split

**`mlmf-hf-layout` and `mlmf-source-hub` must be different crates**, or reading a *local* HuggingFace checkpoint drags in the Hub client and its TLS stack — the `protoc` failure again in different clothing.

`mlmf-hf-layout` is pure: given a list of filenames plus the bytes of `model.safetensors.index.json` and `config.json`, it reports the checkpoint's structure and where each tensor lives. It never enumerates a directory. `mlmf-source-file` walks a local directory; `mlmf-source-hub` enumerates a repo and fetches. The same layout logic then serves a local folder, a Hub repo, an S3 prefix or a tarball.

### 3.3 Dependency contracts (enforced in CI, not by review)

- **C1 (backstop, not a target).** `mlmf-core` has **≤ 50 transitive dependency nodes** at `default-features = []`. This is Fuel's number and it is deliberately loose; at a projected actual of ~15 it leaves 35 nodes of room to drift into without ever failing. **C2 is the operative control; C1 only catches a catastrophe.** Once `mlmf-core` exists and is measured, C1 is reset to *measured + 5* and this loose value is retired. *(Fuel raised this against their own ceiling.)*
- **C2.** A **pinned dependency-set snapshot test** asserts `mlmf-core`'s exact transitive set. Any addition or removal fails the build and forces a human decision. This replaces an earlier proposed `wasm32-unknown-unknown` gate, which was withdrawn when Fuel dropped wasm (16,743 lines deleted, 2026-08-14) and Lightbulb was measured to have zero wasm references. The snapshot is a better instrument for the same purpose: it catches *any* new dependency, not only non-cross-compilable ones. *(Note: `wasm32-unknown-unknown` builds `std`, so `std::fs` compiles there and fails only at runtime — that gate would never have caught filesystem assumptions.)*
- **C3.** No crate on the format axis references `std::fs`, `memmap2`, or any network client. Enforced by a source-level check, not convention.
- **C4.** `mlmf` (umbrella) is the only crate permitted to depend on more than one format crate. `mlmf-meta` depends on no format crate.
- **C5.** No build-script codegen anywhere except `mlmf-onnx`.
- **C6.** CI builds **and runs** the full parser suite with `--no-default-features`, proving the mmap-free path is functional rather than merely compilable.
- **C7.** One version number across the workspace, released in lockstep. *(Precedent: `fuel-formats` and `fuel-ir` both sit at `v0.10.3`.)*

### 3.4 mmap

`memmap2` is a **default** feature of `mlmf-source-file`. Both consumers want mmap (Fuel loads through `MmapedSafetensors`; Lightbulb ranked mmap zero-copy first), and feature unification would enable it for everyone regardless, so making it opt-in would be ceremony. What is preserved is the **API shape**: `&[u8]` and `impl Read` are the primary entry points and mmap is one byte-source among them, which is what keeps streaming and IPC transports possible later. C6 is what makes that true rather than asserted.

---

## 4. The file model (`mlmf-core`)

Deliberately **not** called an IR: `fuel-ir` means something specific in this ecosystem (a compute IR with `Layout`, `StrideVec`, an operator vocabulary) and the two must not collide in conversation.

### 4.1 Encoding and size arithmetic

The single load-bearing abstraction. GGUF and safetensors are **siblings, not a stack** — two ecosystems answering the same question — so the shared substrate sits *underneath* both and is thin.

```rust
pub enum Encoding {
    Dense(DType),        // bytes = nelements × dtype.size()
    Blocked(BlockSpec),  // bytes = nelements / elements_per_block × bytes_per_block
}

pub struct BlockSpec {
    pub family: &'static str,       // "ggml"
    pub code: u32,                  // the file's OWN declared type id, passed through
    pub elements_per_block: usize,  // Q4_0: 32   K-quants: 256
    pub bytes_per_block: usize,     // Q4_0: 18   (2-byte f16 scale + 16 bytes of nibbles)
}
```

- **Size arithmetic is uniform and lives in core**, so no consumer re-derives block math and gets it subtly wrong. `mlmf-ggml` supplies the code→`BlockSpec` table; core stays format-agnostic.
- **`code` is the file's own declared type id, passed through untouched.** Fuel maps it back to `GgmlDType` to select a q4_0/q4_km kernel. MLMF never re-spells it.
- **Extend by data, not by enum variants.** A new format with a new block scheme adds a `BlockSpec` row. Otherwise a superset of N formats forces a widening that ripples through every consumer each time N grows.

*Measured basis:* `fuel-ir/src/quantized.rs` — `block_size()` returns 32 for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 and 256 for the K-quants; `type_size()` returns 18 for Q4_0.

### 4.2 TensorDescriptor and Shape

```rust
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Shape,        // dims AS DECLARED — never reordered
    pub encoding: Encoding,
    pub bytes: Range<u64>,
}
```

GGUF declares dims in the opposite order from HF state dicts. **The ggml reversal is an explicit call in `mlmf-ggml`, never an implicit courtesy.** A crate that silently "helpfully" reverses dims is the same class of defect as flattening rope scaling.

### 4.3 MetaValue

Core's `MetaValue` is **GGUF's 13 typed variants including nested `Array`, plus a 14th — `Bytes(Vec<u8>)`** — chosen deliberately because it is the strict superset.

**Why the 14th exists (added during implementation).** A GGUF string is a length-prefixed *byte array*, not a Rust `String`. A file carrying non-UTF-8 bytes in a token string is readable by the reference implementation, so refusing it would reject a valid file, and `String::from_utf8_lossy` substitutes U+FFFD — which silently changes what the model tokenizes to and violates §9 clause 2.1 outright. `Bytes` preserves such a value verbatim. Nothing is dropped and nothing is converted lossily. *Measured:* GGUF carries `U8 I8 U16 I16 U32 I32 U64 I64 F32 F64 Bool String Array(Vec<Value>)`; safetensors' `__metadata__` is `Option<HashMap<String, String>>` and can only ever produce `String`.

That asymmetry is *why* GGUF is one file rather than a directory: its metadata system is expressive enough to absorb what `config.json` and `tokenizer.json` hold. It is not file concatenation.

`mlmf-hf-layout` projects HF JSON into the same shape. **This is what makes the layer above format-agnostic**: because both GGUF in-file metadata and HF JSON sidecars satisfy one `MetadataSource` trait, config accessors and chat-template extraction are written once and serve both.

### 4.4 Errors

`mlmf-core` owns a plain `thiserror` enum with no foreign types, including path attribution. Fuel adds `impl From<mlmf_core::Error> for fuel_ir::Error` on their side (they own `fuel_ir::Error`, so no orphan-rule problem), keeping `?` working and confining churn to signatures rather than bodies.

*Verified with Fuel:* `fuel-formats` uses `fuel_ir::Context` at exactly **four** sites, every one `Option::context("static string")` and none of them path attribution; `bail!` at **34** sites, all formatted messages. No error quality is at risk in the transition.

### 4.5 Traits

`mlmf-core` defines the seam that format crates implement and `mlmf-meta` consumes: a tensor-container trait, a `MetadataSource` trait (typed key lookup), and a byte-source trait implemented by the source axis. Format crates depend on core alone; sources depend on core alone.

**`TensorContainer::tensor_bytes` returns `Cow<'_, [u8]>`, not `&[u8]` (corrected during implementation).** Two independent requirements force it. §11 records that `mlmf-pickle` must return borrowed-or-owned, because a deflated ZIP entry has to be inflated into a buffer and cannot be borrowed — with `&[u8]` that crate simply cannot implement the seam. And `Cow` is what makes **AL-3 hold *through* the seam**: a caller can test `matches!(bytes, Cow::Owned(_))` and see that MLMF allocated. With `&[u8]`, an inflate-into-self copy would be exactly the invisible cost AL-3 forbids, with no API surface capable of revealing it.

**Size arithmetic is checked, not wrapping (corrected during implementation).** `Shape::elem_count` and `Encoding::byte_size` both return `Result` and fail with `ShapeOverflow` / `SizeOverflow`. Unchecked `u64` multiplication over file-declared dimensions panics in debug and **wraps silently in release** — and a wrapped byte size then *validates as correct* against a wrapped range, which is a wrong answer that passes its own consistency check.

### 4.6 Alignment

**`TensorDescriptor.bytes` carries no blanket alignment guarantee, because the formats do not agree.** Stating this is load-bearing: a `&[u8]` at an arbitrary offset cannot be reinterpreted as `&[f32]` or `&[f16]` without either an alignment guarantee or a copy, and a copy at load time defeats the mmap decision in §3.4.

| Format | Guarantee | Basis |
|---|---|---|
| GGUF | Tensor data padded to `general.alignment` (default **32**) | Format specification |
| safetensors | **None.** `data_offsets` are cumulative byte offsets, so an F32 tensor following an odd-length U8 tensor is misaligned | Format specification |
| pickle / `.bin` | None; and a deflated ZIP entry cannot be borrowed at all (§11) | Container |

mmap yields a page-aligned *base*, which says nothing about per-tensor offsets. In practice most safetensors tensors land aligned because their sizes are large powers of two — which is precisely what makes this dangerous: it works until a checkpoint with an odd-length tensor appears, and then it is undefined behaviour rather than a wrong answer.

Therefore:

- **AL-1.** Every descriptor reports the **actual** alignment of its byte range as a fact. Consumers may branch on it.
- **AL-2.** Typed reinterpretation is **fallible and explicit**: `try_as_slice::<T>()` returns `Err` on misalignment (the `bytemuck::try_cast_slice` contract; `bytemuck` is already a dependency). There is no infallible typed accessor.
- **AL-3.** **MLMF never silently copies to satisfy alignment.** The realigning path is a separate, named call the consumer chooses. A silent copy is the same class of defect as §9 clause 3.1's promoting casts: correct output, invisible cost, discovered only by whoever profiles it later.
- **AL-4.** Writers emit naturally-aligned output where the target format permits it, and record the alignment they used.

Blocked encodings need less than dense ones: a `#[repr(C)]` Q4_0 block is `{ f16 scale, [u8; 16] }` — size 18, alignment 2. GGUF's 32-byte padding covers every encoding in this design; safetensors' worst case is F32's alignment of 4.

**AL-2 is a capability gain, not only a hazard avoided.** Checking Fuel's own tree for this hazard found the asymmetry reproduced exactly — their GGUF path reads `general.alignment` with a 32-byte default and pads; their safetensors path has *zero* alignment references. Fuel has no UB, because their safetensors decode is byte-wise (`chunks_exact(4)` → `from_le_bytes` → `push`), which is alignment-independent and endian-explicit. **But that means Fuel copies every tensor on load despite mmap: what AL-3 names as the separate realigning path is, in Fuel today, the unconditional one.** A fallible `try_as_slice::<T>()` is precisely the primitive they lack — it lets a consumer borrow when alignment permits and fall back to byte-wise decode when it does not, which is a choice they currently cannot make. *(Fuel has opened a measurement of what that copy costs; no number is asserted here.)*

**The failure shape is worth naming on its own: a latent UB masked by the common case.** It has no failing-test phase — it works, and works, and then a checkpoint with an odd-length U8 tensor arrives, and the failure is undefined behaviour rather than a wrong number. **A wrong answer gets investigated; UB gets a bug report about something else entirely.** AL-2's refusal to offer an infallible typed accessor is the response: it forces the common case and the rare case down the *same* code path, so the rare one cannot be reached only in production.

*Raised by the Fuel architect against the first draft, where "align" appeared once in 406 lines and not normatively. Their framing: silence here is discovered by whoever writes the first zero-copy backend hand-off, which is them.*

---

## 5. Canonical key vocabulary

What conversion and cross-format queries need is **canonical keys**, not a canonical struct.

```
mlmf key                     GGUF                              HF
attention.head_count    ←→   llama.attention.head_count   ←→   num_attention_heads | n_head
attention.head_count_kv ←→   llama.attention.head_count_kv←→   num_key_value_heads
rope.freq_base          ←→   llama.rope.freq_base         ←→   rope_theta
tokenizer.chat_template ←→   tokenizer.chat_template      ←→   tokenizer_config.json:chat_template
```

Declared, bidirectional, per-format. Values remain typed `MetaValue`s and are **never flattened into a canonical interpretation**. Agnosticism is delivered at the *query* layer, not the *storage* layer.

Four rules make this safe:

- **Enumerate what you understand; preserve what you don't.** Unknown keys survive verbatim. Dropping unrecognised keys is the worst kind of lossy — invisible.
- **Absent means `None`, never a default** (except under §6).
- **Mark lossiness per key, not per crate.** A crate-level caveat is documentation, and documentation loses that argument.
- **Extend by data, not by variants** (§4.1).

Structurally this is the move KISS made: a controlled vocabulary of declared tokens, matched exactly, with no inference logic.

---

## 6. Declared / Absent / Supplied

Writing output that is *exactly faithful* to a source but omits what the target format requires produces a file that is correct and will not run. Some formats have built-in assumptions that other formats require to be explicit. So the state model has three values:

- **Declared** — the source says it.
- **Absent** — the source does not.
- **Supplied** — MLMF filled it at write time, **attributed**.

The fence that keeps this from becoming interpretation:

> **MLMF may supply a *format's* documented default. MLMF may never supply a *model's* value.**

"The GGUF spec says `general.alignment` defaults to 32" is file-format knowledge. "This model probably wants `rope_theta` 10000" is interpretation and belongs to Fuel.

**The boundary is fuzzy and needs a fence, not just a principle.** HuggingFace's "format defaults" are really *transformers'* Python defaults — `num_key_value_heads` absent means MHA because the reference implementation says so, not because a specification does. That is a defensible default to supply, but the table will drift into opinion if we are casual. Therefore:

- **CD-1.** A default may only be supplied if it is **citable to a specification or a named reference implementation**, and the citation lives in the table beside the value. If you cannot cite it, you cannot supply it.
- **CD-2.** Every supplied value appears in the conversion report and in the output's provenance.
- **CD-3.** If a target format requires a value that is neither declared in the source nor a citable format default, **refuse the conversion and name the missing key.**

---

## 7. Unknown handling

A log line is ignorable by construction, so this lives in the type system: **every parse returns `(Content, Report)`.** The content cannot be obtained without also receiving the account of what was not understood.

**Two severity tiers, because not all unknowns are survivable:**

- **Fatal — refuse to parse.** Unknown format version, unrecognised block encoding, and an unknown type code **in a container that derives tensor offsets by accumulation** — each tensor's size feeding the next tensor's start, so one unreadable size makes every later offset unknowable. These make byte-size arithmetic unknowable, so proceeding means handing out *wrong* bytes rather than incomplete ones.
- **Loud — preserve verbatim and report.** Unknown metadata keys, unrecognised files in a checkpoint, unhandled feature flags, and an unknown type code **in a container that stores each tensor's offset explicitly** — GGUF is this case, so an unrecognised type code there costs exactly that one tensor's length, and metadata and every other tensor stay readable. Harmless to carry, dangerous to drop. The report names the key, its typed value, and its origin — not a count.

**The fatal/loud split for an unknown type code is a property of whether the unknown poisons other addressing, not a property of the unknown itself.** The same code is fatal in a format whose offsets accumulate and loud in a format whose offsets do not — §7 is not "type codes are always fatal," it is "an unknown is fatal exactly where continuing would compute wrong bytes for *something else*." (`mlmf-ggml`, §11.)

**Two independent obligations, and this distinction is the important part:**

- **U-1 (agreement).** CI asserts the report is empty across a corpus of known-good checkpoints. A new model with a new metadata key fails MLMF's build. This is how MLMF's developers learn about gaps rather than never finding out.
- **U-2 (currency).** **Corpus currency is tracked and can fail independently of corpus agreement.** Each entry records where it came from and when; staleness is its own failing check, never an inference from a passing one.

U-2 exists because of a measured incident in Fuel: their vendored corpus sat **seven releases behind upstream, green the whole time**, because the constant and the file were updated together — so the test proved the two *agreed*, not that either was *current*. Without U-2, a corpus frozen at today's checkpoints stays green in a year while MLMF silently fails to understand every metadata key shipped in the meantime. **A green agreement test with an absent currency test is not two-thirds of a guarantee; the second one's absence silently caps what the first can mean.**

Consumers get a policy knob for free: Lightbulb may refuse to serve a model with a non-empty report, Fuel may log it, a conversion tool may print it. MLMF states facts; consumers choose posture.

---

## 8. Conversion, measured error, and provenance

Conversion is **Lightbulb- and operator-facing**. Fuel loads and does not convert (~1 file in their tree matches conversion patterns), so it is scoped away from them.

### 8.1 Measured, not declared

Fuel's `PrecisionGuarantee` (`fuel-dispatch/src/fused.rs:117`; lowering rules in `fuel-dispatch/src/fkc/precision.rs`) carries `max_ulp` / `max_relative` / `max_absolute`, with the rule that **an absent bound means "no claim," not zero error** — the same principle as §5's "absent means `None`."

**The asymmetry that makes MLMF's version stronger** (Fuel architect): a kernel's error over all possible inputs cannot be computed up front, so Fuel's authors *declare* bounds and verify later. **A conversion's error is computable — at conversion time, over exactly the tensors being converted.** Therefore:

- **CV-1.** MLMF reports a **measured** maximum error. **"Declared" does not exist in the API.** The honest states are `measured` and `not computed`.
- **CV-2.** `bit_stable_on_same_hardware` is dropped as a field and held as an invariant: a file transform is deterministic unconditionally.
- **CV-3.** **Always measure and record** the error — unconditional, no opt-out.
- **CV-4.** **Refuse only when a budget was specified and is unmet**, with the measured value in the error rather than "exceeded."
- **CV-5.** **Never silently succeed with an unrecorded loss.** That is the case that outlives everyone.

### 8.2 Provenance, and the discriminator

A converted file carries: source content hash, source format, target format, measured error, converting-tool version. A **consumer asserts** these rather than reading them — *"a field that is only read is documentation"* (Fuel).

But a stamp proves **binding**, not **currency**. The general test, which applies to every provenance mechanism in this design:

> **PV-1. Can this assertion fail without a human changing a constant?** If not, it is a binding record wearing a currency check's clothes. Two stored constants compared against each other can only restate what someone wrote down — **the source must be a party to the check.**

Concretely: the source content hash must be verified by **re-hashing the actual source and comparing**, which is falsifiable; comparing two stored fields is not. Keep spec provenance separate from artifact commit, because conflating them leaves the report unfalsifiable.

The same correction applies to `mlmf-source-hub` (§9, HUB-1/HUB-2): keying a cache on repo + resolved revision proves what was fetched, never that it is current.

---

## 9. MFIS — normative semantic clauses

The shared crate is the standard for **mechanism**; these clauses are the standard for **semantics** — what code-sharing cannot enforce. Each clause is backed by a test in the relevant crate. Numbered so other projects' design documents can cite them.

**§1 Declaration and absence**
- **1.1** A blank `chat_template` is **UNDECLARED**, not declared-empty. Check `trim().is_empty()`, not `is_empty()`. *(A blank template parses fine and renders an empty prompt — the model receives a request to continue nothing. Shipped and caught in review at Lightbulb.)*
- **1.2** Absent means `None`, never a default, except under §6 (Supplied, attributed).
- **1.3** A single-file `.gguf` is a first-class checkpoint. Companion JSON lives **beside** the file, never inside a directory named after it — `foo.gguf/tokenizer_config.json` is a path that can never exist, and constructing it silently yields empty BOS/EOS. For GGUF the in-file metadata is authoritative regardless.
- **1.4** Chat-template extraction covers `tokenizer_config.json:chat_template` (string *or* a list of `{name, template}` where the `default` entry is wanted), the GGUF `tokenizer.chat_template` key, and the `chat_template.jinja` sidecar. Declared BOS/EOS likewise: `tokenizer_config.json`'s `bos_token`/`eos_token`; `config.json`'s `bos_token_id`/`eos_token_id` resolved through `tokenizer.json`'s `added_tokens`; GGUF's `tokenizer.ggml.{bos,eos}_token_id` indexed into `tokenizer.ggml.tokens`.

**§2 Byte-exactness**
- **2.1** Vocabulary, BPE merges, and token strings round-trip **byte-exact**. No Unicode normalization, case folding, trimming, or reordering — ever. *(Unpopped: their token grammar forbids prefix/subset/normalization logic because `cuda:sm90` and `cuda:sm90a` are different compilation targets that a prefix match collapses into one entry serving the wrong kernel. Tokenizer merges and token strings have the identical "looks normalizable, isn't" property, and the failure is silent.)*
- **2.2** Anything derived carries a marker rather than silently replacing the original.
- **2.3** Declared type codes and dim order pass through untouched (§4.1, §4.2).

**§3 Type policy**
- **3.1** Dtype coercion is **consumer policy, never a default**. "Preserve source dtype" and "coerce everything to X" both ship; neither is the default. *(Fuel's stock loader preserves bf16 for projection matrices while loading embeddings and norms as f32, yielding matmul keys `[F32, BF16, F32]` that no CPU kernel serves — the optimizer then inserts **155 promoting casts per realize on TinyLlama-1.1B**. Lightbulb's `loader_f32.rs` exists solely to force all-f32 and avoid that. If MLMF picks one, somebody silently eats those casts.)*
- **3.2** Rope scaling exposes the **variant** or the **raw**, never a flattened number. *(`linear` / `dynamic` / `llama3` / `yarn` flattened to a float reads fine and computes wrong on long context, with no error.)*
- **3.3** Lossiness is marked **per field**, not per crate.

**§4 Handoff obligations**
- **4.1** A rendered chat template must be tokenized with **`add_special_tokens = false`**, and that rule travels **with** the template as a field of the returned value, not as prose. *(Templates interpolate `bos_token` into prompt text; a tokenizer whose post-processor is `TemplateProcessing` then prepends BOS again, so a Llama-3 checkpoint receives `128000, 128000, 128006, …` — a pair it never saw in training. Invisible to every text-level assertion, because the render is byte-identical either way.)*
- **4.2** The Jinja dialect required to render real Hub templates is documented alongside extraction even though rendering is the consumer's job: `macros`, `loop_controls`, `json`/`tojson`, `strftime_now` registered as a **global**, and Python `str`/`dict` methods via minijinja-contrib's `pycompat`. A narrower list does not degrade rendering — it makes real checkpoint templates **fail to parse**, which falls through to a family guess and then reports success at a lower tier. *(8 of 9 real Hub templates measured at Lightbulb render only with all of it.)*

**§5 Derived artifacts**
- **5.1** A derived or cached artifact's validity key names everything it was baked against, or that thing must be unable to change for the artifact's lifetime. *(Fuel's rule, from four silent full-speed wrong-answer incidents in two days, every one an artifact outliving something it was baked against.)*
- **5.2** **PV-1** (§8.2) applies: an assertion that cannot fail without a human changing a constant is not a currency check.
- **5.3** Agreement and currency are separate obligations, each able to fail alone (§7 U-1/U-2).
- **5.4** Copy the obligation, not the granularity. *(Unpopped's stamp is deliberately coarse — crate version — which over-invalidates harmlessly for a generator, where a false invalidation costs a recompile. Fuel measured a **223× reuse factor** on a held decode plan, where version-granularity would fire every step and destroy the thing it protects. A model cache is closer to Fuel's situation than Unpopped's.)*

**§6 Hub access** *(`mlmf-source-hub` only)*
- **HUB-1** A revision must be pinned, or the resolved commit SHA recorded. Fetching `main` resolves differently over time, which makes downstream measurement unreproducible — the same objection both consumers raised against the LLM oracle, arriving from a different direction.
- **HUB-2** The cache keys on repo + resolved revision + filename, never repo alone, and **no API implies currency without an explicit network check**.

**§7 Adoption of existing parsers**

MLMF adopts Fuel's GGUF parser and pickle VM (§11, §12). **Removing I/O from a working parser is a behaviour-preserving refactor, which has no born-red state: "the tests still pass" is also what a no-op produces.** A port review cannot distinguish a correct port from a subtly broken one, so the gate is differential, not editorial.

- **AD-1.** Before Fuel switches its dependency, a **differential harness** parses a real corpus with both the origin implementation and the ported one and asserts **byte-identical** tensor descriptors *and* byte-identical tensor bytes. Not "equivalent" — identical.
- **AD-2.** The differential must be **proven able to fail**: deliberately sabotage the port (perturb an offset, drop a metadata key, flip a dim order) and confirm the harness catches each. An unfalsified differential is PV-1's failure wearing a test's clothes.
- **AD-3.** **Pickle's security boundary is its allowlist of permitted `REDUCE` targets, so that list carries a test that fails when it grows.** Without it the property silently weakens in its new home — a new entry is a one-line diff that looks like a feature and is a hole. Growth is permitted; growth *unnoticed* is not.

*(Raised by the Fuel architect. AD-1 and AD-2 protect the parser; AD-3 protects the property that makes §11's containment argument true.)*

---

## 10. Non-goals

Recorded with reasons so they are not relitigated from taste later.

- **A canonical semantic model config** (e.g. `TransformerConfig { rope_scaling: f32, … }`). Fuel has `LlamaFullConfig`, candlelight has `LlamaConfig`, Lightbulb has `Config`; a canonical fourth yields **N translations instead of N−1**. Lightbulb's orphaned `convert_mlmf_config_to_lightbulb()` is the receipt for this having been tried once. See §9 clause 3.2 for the failure mode.
- **A computation-graph IR.** ONNX's graph has no counterpart in GGUF or safetensors, and `fuel-ir` already exists. MLMF treats an ONNX graph as **opaque payload plus enumerable initializers** — tensors and bytes, no operators, no topology, no semantics.
- **Memory / KV-cache / GQA estimation.** Requires modelling what the model does. Fuel and Lightbulb each already have their own; two implementations in one binary is a divergence risk.
- **Architecture inference from tensor-name patterns.** A guess about semantics. Architecture *extraction* (`general.architecture`, `architectures[]`) is in — reading a declaration is not interpreting it.
- **Heuristic or LLM-assisted name mapping on any load path.** Non-deterministic startup. Moves to `mlmf-oracle`, which emits a mapping file offline that a user checks in. Declared translation tables remain in-tree; they are format knowledge and are needed to write a GGUF from an HF checkpoint.
- **Calibration-based quantization** (KL divergence, activation statistics, imatrix weighting). Not merely out of scope — **unimplementable**: it requires forward passes, and MLMF has no compute backend by design. *(This phrasing is deliberate. An out-of-scope decision invites relitigation; an unimplementable one does not.)*
- **Networking anywhere except `mlmf-source-hub`.** A network call on a load path is the same determinism problem as the LLM oracle, plus supply-chain surface. If model fetching grows, it grows in a crate that depends on MLMF, never the reverse.
- **LoRA application / weight merging.** Reading `adapter_model.safetensors` + `adapter_config.json` is in — those are model files. Merging weights is operating on the model.
- **Multimodal processing, distributed loading, model cards.** Not model-file work under any reading of the charter.

---

## 11. Disposition of existing MLMF code

The legacy crate at `src/` is **37 files and 19,329 lines**, measured at `e8c7616` with `find src -type f -name '*.rs' -exec wc -l {} +`. Every one of them has a row below and the LOC column sums to 19,329, so a file that goes missing from this table is a defect the sum will show.

*(This matters because the first version of this table was wrong in both directions at once. It listed 30 files and 18,333 lines: seven files had no row — they were present the day it was written, so this was an omission and not drift — and six rows had since been outgrown by their files. Neither error was visible, because a table with no total cannot be audited by reading it.)*

**What now exists**, merged to `main`, is the other half of what changed. The dispositions below were first written when no replacement crate existed and every one of them was an aspiration. Five can now be named as fact:

- **`mlmf-core`** (#1) — the vocabulary and traits of §4. No filesystem, no memory mapping, no networking; and no tensor type, no device, no backend trait.
- **`mlmf-ggml`** (#5) — the ggml type space and block geometry: 35 live codes plus 8 retired. It parses no container and does no I/O.
- **`mlmf-gguf`** (#3, #4) — the GGUF reader: header, metadata index, and tensor directory.
- **`mlmf-safetensors`** (#6) — the safetensors reader: header, metadata, tensor directory.
- **`mlmf-conformance`** — cross-backend conformance tests. No library code, never published.

**`mlmf-source-file` is in progress on a branch and has not landed** (§12 step 3). The work has not followed §12's order: steps 1 and 2 are done, step 3 is open, and `mlmf-safetensors` — half of step 5 — landed ahead of both step 3 and step 4. Its partner `mlmf-hf-layout` does not exist, so step 5 is half-done. Nothing from steps 6 or 7 has been started.

Two facts constrain how the table below should be read, and both are easy to get wrong from the crate names alone:

1. **All five merged crates are readers. Not one of them writes a file.** So every export row is unsatisfied regardless of how complete its format's reader is — `mlmf-gguf` landing does nothing for `formats/gguf_export.rs`.
2. **Nothing here has been deleted yet.** `src/` has not lost or gained a file since this spec was written; the only commit to touch it since is #5. "Delete" in the Status column is a disposition, never a report.

Status vocabulary, so "planned" and "done" cannot be confused:

- **Superseded** — a merged crate does this now. The file is redundant and awaiting deletion.
- **Planned** — the replacement is named, and does not exist yet.
- **Delete** — no replacement, by a decision recorded in §10 or in the row.
- **Undecided** — genuinely unsettled. Recorded as such rather than guessed, because a reader needs to know which rows are load-bearing questions.

| Module | LOC | Status | Disposition |
|---|---:|---|---|
| `loader.rs` | 1,526 | Planned | Rewritten across the format axis; `LoadedModel`'s candlelight `VarBuilder` dies with it |
| `quantization_simple.rs` | 1,111 | Delete | Calibration-based, unimplementable without a backend (§10) |
| `lora.rs` | 953 | Planned | Split: adapter **file** reading kept, weight merging deleted (§10) |
| `distributed.rs` | 923 | Delete | Not model-file work under any reading of the charter (§10) |
| `formats/onnx_export.rs` | 846 | Planned | → `mlmf-onnx` (§12 step 6, not started) |
| `distributed_loader.rs` | 842 | Delete | As `distributed.rs` |
| `formats/gguf_export.rs` | 818 | Planned | → `mlmf-gguf`, decoupled from candlelight. **Unsatisfied by the merge**: `mlmf-gguf` landed as a reader and has no writer |
| `metadata.rs` | 783 | Planned | Split: provenance kept and strengthened (§8.2), tensor statistics deleted |
| `model_card.rs` | 772 | Delete | Not model-file work (§10) |
| `validation.rs` | 761 | Planned | CUDA/device validation deleted — `mlmf-core` has no device type, so that half has no home by construction. Structural validation kept, not yet ported |
| `quantization.rs` | 656 | Delete | Calibration-based; see `quantization_simple.rs` |
| `distributed_core.rs` | 609 | Delete | As `distributed.rs` |
| `conversion.rs` | 599 | Planned | Rebuilt on §8 measured-error semantics |
| `config.rs` | 588 | Planned | → `mlmf-hf-layout` + `mlmf-meta` as **raw + typed accessors**, not a normalized struct (§10) |
| `cache.rs` | 584 | Undecided | Re-examine under §9 clause 5; delete rather than ship one that can serve a stale entry silently. **The re-examination has not happened**, so which of the two it is remains open |
| `formats/onnx_import.rs` | 574 | Planned | → `mlmf-onnx` (§12 step 6, not started) |
| `multimodal_processor.rs` | 565 | Delete | Not model-file work (§10) |
| `checkpoint.rs` | 543 | Undecided | Deferred (OQ-1) — but **not** for the reason this row first gave. §13 records "no consumer asked" as stale: Fuel trains and has its own checkpoint format over `LazyVarMap`. Deferred because the one project that trains has already solved it for itself |
| `name_mapping.rs` | 520 | Planned | Declared tables → `mlmf-meta`; inference → deleted (§10) |
| `multimodal_loader.rs` | 493 | Delete | Not model-file work (§10) |
| `progress.rs` | 477 | Planned, home undecided | Kept: optional, off by default. **No step in §12 names the crate it lands in** |
| `mmap_loader.rs` | 475 | Planned | → `mlmf-source-file`, in progress on a branch and **not landed** (§12 step 3) |
| `formats/pytorch_loader.rs` | 407 | Delete | **Delete the stub**; adopt Fuel's 753-line pickle VM as `mlmf-pickle` (§12 step 6). Confirmed a stub: `load_zip_pickle` and `load_legacy_pickle` both return errors, so this file never parses a pickle |
| `smart_mapping.rs` | 399 | Planned | → `mlmf-oracle`, off the load path entirely (§10; OQ-4 for where it lives) |
| `universal_loader.rs` | 362 | Planned | → `mlmf` umbrella, format detection (§12 step 7) |
| `multimodal.rs` | 358 | Delete | Not model-file work (§10) |
| `formats/gguf.rs` | 317 | **Superseded** | **by `mlmf-gguf`** — the replacement this row used to ask for now exists and is candlelight-free. What it replaces is a 317-line shim over `candlelight::quantized::gguf_file` that still imports `candlelight::VarBuilder` |
| `cached_loader.rs` | 270 | Undecided | Follows `cache.rs`, and is therefore open for the same reason |
| `formats/awq.rs` | 264 | Planned | → `mlmf-awq`; both consumers indicated it is dead weight to them (§12 step 6) |
| `saver.rs` | 221 | Undecided | Format-agnostic write dispatch — the write-side counterpart of `universal_loader.rs`, which no step in §12 provides a home for. Its only concrete saver, `SafeTensorsSaver::save_tensors`, returns `Err` unconditionally, and `save_model`/`save_safetensors` both route into it |
| `error.rs` | 161 | **Superseded** | **by `mlmf-core::error`** (§4.4). What it replaces is an enum whose variants wrap `candlelight::Error` and `safetensors::SafeTensorError` — dependency edges the split exists to remove |
| `lib.rs` | 158 | Planned | Rewritten as the `mlmf` umbrella (§12 step 7). `pub use candlelight::{DType, Device, VarBuilder}` at the crate root is the coupling the whole split removes. **This is the crate that `.` in `default-members` exists to keep building** — a rewrite presupposes it still compiles when someone starts, which is the reason recorded in `Cargo.toml` |
| `formats/safetensors_export.rs` | 134 | Planned | → a safetensors writer, which **does not exist**: `mlmf-safetensors` reads only. Worth stating what is being replaced — `save_as_safetensors` writes an 8-byte length prefix and a `__metadata__`-only header, appends no tensor data, and returns `Ok` |
| `formats/safetensors.rs` | 111 | **Superseded** | **by `mlmf-safetensors`** — the reader landed in #6. This file's `load_mmaped_safetensors` does not memory-map; it forwards to `load_regular_safetensors` |
| `formats/mod.rs` | 63 | Planned | Module declarations and glob re-exports (`pub use safetensors::*`, and so on) only. Dies with the modules it declares; nothing to port |
| `formats/awq_export.rs` | 50 | Planned | → `mlmf-awq` with `formats/awq.rs` |
| `formats/pytorch_export.rs` | 36 | Delete | **Delete the stub** — `save_as_pytorch` is an unconditional `Err`. Whether anything in MLMF ever *writes* pickle is undecided: §12 step 6 names `mlmf-pickle` without saying which direction it goes |

**Three rows carry Superseded, totalling 589 lines.** That is the honest measure of how much of `src/` five merged crates have actually retired, and it is small because the merged crates are readers of two formats while `src/` is mostly loading policy, deletion candidates, and export paths. The reader that replaced `formats/gguf.rs` is not smaller than it — `mlmf-gguf` is larger — because it does the job without candlelight and reports what it cannot read.

One file in `src/` is not Rust and has no row: `src/mlmf.code-workspace`, a VS Code multi-root workspace pointing at `..` and `../../lightbulb`, committed by an evacuation commit. It is editor configuration that landed under `src/` by accident, and it is noted here only so that "37 files" and a directory listing of 38 entries do not read as a contradiction.

**On pickle specifically.** Pickle is a stack-based virtual machine — the file is a program, and `GLOBAL`/`STACK_GLOBAL` plus `REDUCE` make arbitrary code execution expressible in a well-formed file. **No Python interpreter is required to read one safely:** Fuel's `fuel-formats/src/pickle.rs` implements the opcodes directly and honours `REDUCE` only for `torch._utils::_rebuild_tensor_v2`, `torch._utils::_rebuild_parameter`, `OrderedDict`/`defaultdict`, and a storage-name→dtype table. Everything else is refused. **The security model is: never be a general pickle interpreter.** MLMF's own implementation never parses anything, so there is no deliberate choice to make between the two.

This is also the strongest single justification for the per-format split: **the pickle VM can only enter a dependency graph through an explicit `mlmf-pickle` edge.** Feature-gating could never have promised that.

One wrinkle: mmap-slicing a `.bin` works only because `torch.save` writes ZIP entries **stored, not deflated**. A compressed entry must be decompressed into an owned buffer, so `mlmf-pickle` returns borrowed-or-owned, unlike GGUF and safetensors which are always borrowable.

### `mlmf-ggml` — landed

Implemented ahead of `mlmf-gguf` (§12 step 2), as the type-geometry half of that step. It is a type table and nothing else: it parses no container, so no row in the table above is retired by it. Two decisions this plan made that the spec above did not anticipate:

1. **The type space is 35 live codes plus 8 retired, not the ~15 a straight port of Fuel's table would have produced.** The corpus evidence for the wider space is concrete — 540 `IQ4_NL`, 30 `IQ3_S`, 30 `IQ4_XS` tensors in ordinary Hub downloads that a 15-type table cannot read at all.
2. **§7's fatal/loud rule is amended** (see §7 above): the split is a property of whether an unknown poisons other addressing, not a property of the unknown itself. This was found while building `mlmf-ggml`, because GGUF's explicit per-tensor offsets are the concrete case that falsifies "unknown type code ⇒ always fatal."

### `mlmf-gguf` — landed

§12 step 2's GGUF half. The metadata path landed first: magic, version, counts, and the key-value block, indexed at open and decoded on demand. **The tensor directory has since landed too**, so the read side of this step is complete; there is still no GGUF writer, which is why `formats/gguf_export.rs` above is Planned rather than Superseded. Two decisions this plan made that the spec above did not anticipate:

1. **GGUF v1 is refused, not parsed.** llama.cpp refuses it too, so its reader is not a reference for the layout, and the one v1 file in the corpus did not parse under a v2-shaped reader with only the integer widths substituted. Deriving the layout from that file is a separate plan. The consequence is visible in the corpus fixture: 29 `.gguf` files on disk produce 28 measured rows, and the missing one — `legacy/tinyllamas-stories-260k-f32.gguf` — is refused by version rather than misparsed into plausible-looking garbage.
2. **An unknown metadata value type stops the index but not the open.** Its width is unknown, so the parse cannot find the next key — but every key already indexed stays readable, and the failure is reported. This is R1's guarantee applied within the metadata stage, and it is the reason the tensor directory is a separate stage rather than a separate concern. `GgufMetadata::index_complete` is what makes the difference legible to a caller: with an incomplete index, `Declaration::Absent` means only *not found in the part that could be read*, so it can support a positive finding and never a negative one.

**On what the corpus can and cannot prove, measured rather than assumed.** The reference corpus is 1.13 GiB across 29 files. Scanning every key, string value and string array element in the 28 readable ones — **4,686,500 strings** — finds zero non-UTF-8, zero trailing-NUL and zero embedded-NUL strings. So the R3 byte-exactness guarantee is **not falsifiable against real files**: substituting `from_utf8_lossy` for the byte-exact decode is a no-op on every published model in the corpus. That is why `mlmf-gguf` carries authored fixtures alongside corpus-derived ones, and the split is not redundancy — it is the only instrument that can see the path. Verified by sabotage: lossy decoding turns three authored tests red and leaves all three corpus tests green.
---

## 12. Migration and sequencing

Both consumers are unblocked and neither is racing this work. Lightbulb's chat-template epic merged (PR #6); its **GGUF epic is in design with zero lines written**, and it is holding the file-layer-dependent half rather than racing. Fuel will keep `fuel-formats` as a **re-export shim retired on Fuel's schedule** — a big-bang cutover into a graph they cannot fully gate is how a latent break sits behind an unbuilt feature combination, which happened to them twice on 2026-08-14 alone. **This is not a courtesy and MLMF should be held to it:** on their own reading it is the only part of this plan that protects them from that failure mode, so MLMF must not make the shim awkward to keep, and must not treat its retirement as a milestone of MLMF's.

Suggested order, each step independently useful:

1. `mlmf-core` + the dependency snapshot test (C1/C2). Nothing else can be measured until the floor exists.
2. `mlmf-gguf` + `mlmf-ggml`, adopting Fuel's parser with I/O removed — **gated by AD-1/AD-2**, and the differential must be falsified before the port is believed. Highest-value, and it unblocks Lightbulb's GGUF epic.
3. `mlmf-source-file`. Makes 1–2 usable end to end.
4. `mlmf-meta`: key vocabulary, chat-template and BOS/EOS extraction (§9 §1, §4). This is Lightbulb's actual dependency.
5. `mlmf-hf-layout` + `mlmf-safetensors`. This is Fuel's actual dependency.
6. `mlmf-pickle` (Fuel's VM), `mlmf-awq`, `mlmf-onnx`.
7. `mlmf` umbrella, `mlmf-source-hub`, `mlmf-oracle`.

Housekeeping, independent of the above: delete the stale `errors.txt` (102KB, at the repo root, reads as current state and is not), and audit the README's format claims against what actually works before the spec's assertions are inherited.

---

## 13. Open questions

- **OQ-1.** Does `checkpoint.rs` (training checkpoints, optimizer state) come back? They are model files under the charter. **The original premise of this deferral — "no consumer asked" — is stale and must not be read as "nobody trains."** Fuel now has a training stack (`fuel-training`, `TrainState`, AdamW, a working MNIST example as of 2026-08-14) and its own checkpoint format over `LazyVarMap`, so Fuel does not need MLMF's. Still deferred, but now for the accurate reason: the one project that trains has already solved it for itself. Revisit if a second trainer appears. *(Correction supplied by the Fuel architect.)*
- **OQ-2.** Naive round-to-nearest quantization: deferred until an operator asks. If added, it ships with measured error (§8.1) and an explicit non-goal of matching llama.cpp's tuned quality.
- **OQ-3.** GGML-legacy and imatrix: Fuel keeps `imatrix` (an llama.cpp quantization-calibration artifact of narrow interest). Whether MLMF carries GGML-legacy at all is unresolved.
- **OQ-4.** Where `mlmf-oracle` lives — same workspace or a separate repo. Fuel's requirement is only that Fuel *cannot reach it*, which either satisfies.

---

## 14. Decision provenance

Recorded because several of these are load-bearing and their owners should be able to check them.

| Decision | Source |
|---|---|
| Charter: read/write, never interpret | Eric Evans |
| File operations out of core, source/format as separate axes | Eric Evans |
| Loud reporting of unknowns; conversion exception to "absent means None" | Eric Evans |
| Full layered crate granularity; semantic-clauses-only standard | Eric Evans |
| Feature unification defeats single-crate isolation | Lightbulb architect |
| Consumer builds tensors; no backend-generic accessor | Lightbulb architect |
| Dtype coercion as policy; 155 casts/realize | Lightbulb architect, measuring Fuel |
| Chat-template seam; §9 clauses 1.1, 1.3, 1.4, 4.1, 4.2 | Lightbulb architect |
| Raw + accessors over a normalized config; rope-scaling hazard | Lightbulb architect |
| ≤50 dependency ceiling; no build-script codegen | Fuel architect |
| 11-type seam surface (4 by naive grep); `Context`/`bail` measured harmless | Fuel architect |
| Measured-not-declared error; the three-way refuse/report split | Fuel architect |
| Binding vs currency; the PV-1 discriminator | Fuel architect, generalized in discussion |
| §9 clause 5.1 (validity keys) and 5.4 (granularity) | Fuel architect, relayed by Unpopped |
| §4.6 alignment is not uniform and must be stated | Fuel architect (spec review) |
| C1 is a backstop, not a target | Fuel architect (spec review, against their own number) |
| §9 §7 adoption gates AD-1/AD-2/AD-3 | Fuel architect (spec review) |
| OQ-1's premise is stale; Fuel now trains | Fuel architect (spec review) |
| §9 clause 2.1 (byte-exactness) | Unpopped |
| No vocabulary claim — `unpopped-vocab` is the kernel dispatch vocabulary | Unpopped |
