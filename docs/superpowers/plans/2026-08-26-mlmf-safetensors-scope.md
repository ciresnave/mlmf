# Plan 5 scope: `mlmf-safetensors`, and what a second backend is for

**Status: scope, not yet a plan.** The task decomposition follows once the
design questions below have answers, because two of them change what the
tasks are.

## Why this crate exists, and it is not "support safetensors"

The branch is named `backend-agnostic`. The abstraction has been proven
against exactly one backend:

```
impl MetadataSource for  GgufMetadata      <- one
impl TensorContainer for GgufTensors       <- one
Fake, Partial, WithArray, Inflating, Indexed
    — all in mlmf-core/src/traits.rs, all written alongside the traits
      they implement, by the author of those traits
```

Five implementations that look like a sample of six and are a sample of one.
The fakes cannot falsify anything: they are the same author's model of what
an instance looks like, written in the same file from the same assumptions,
so they can only confirm the seam is self-consistent.

**An abstraction validated against a single instance is certified at 100% by
choosing the sample.** The first real test of "agnostic" is the second
implementation. If `MetaValue` or `Encoding` has to change to admit
safetensors, that is the finding — and it is one no amount of further GGUF
work can produce.

## Measured facts this scope is built on

From `docs/superpowers/plans/safetensors_recon.py`, run against the two
models present locally (491 tensors). Reproduce before trusting.

**Layout, confirmed byte-exact on 2 of 2 files.** The falsification test is
that the furthest `data_offsets[1]`, plus the 8-byte length prefix, plus the
JSON header, equals the file length exactly.

```
8 bytes    u64 little-endian header length N
N bytes    JSON header
remainder  tensor data

header:  name -> { dtype: str, shape: [int], data_offsets: [start, end] }
         plus an optional "__metadata__" key holding str -> str

  SmolLM2-360M    tensors=290  hlen=32664  data_start=32672   delta=0
  TinyLlama-1.1B  tensors=201  hlen=23088  data_start=23096   delta=0
                                            (2,200,119,864 bytes)
```

**`__metadata__` is `str -> str`.** Both files carry exactly
`{'format': 'pt'}`.

**491 of 491 tensors are `BF16`.** The corpus cannot vary the dtype at all.

**Zero aliased ranges and zero overlaps** in either file.

## The design questions, and two of them are seam-level

### Q1 — Is `MetaValue` format-neutral, or GGUF's type system with a neutral name?

`MetaValue` has **fourteen** variants. GGUF has thirteen value types. The
thirteen are not a coincidence; the fourteenth, `Bytes`, is the one MLMF
added for a declared string whose bytes are not valid UTF-8.

*(This read "thirteen variants" and was the origin of a miscount that
reached three crates' doc comments. Counted 2026-08-27 in Task 7.)*

Safetensors metadata is `str -> str`, so a `MetadataSource` over it can only
ever emit `MetaValue::String`, and the other thirteen go unused. **The seam
accommodates that trivially, and that is the problem rather than the
reassurance.** A consumer writing `source.get(k).as_u64()` gets a number
from GGUF and a string from safetensors **for the same logical fact, and
nothing anywhere fails.** The interface did not break, so nothing signals.

An abstraction that survives its second instance by accepting anything has
not been validated; it has been shown to be uninformative about the thing it
claims to abstract.

Three answers, and this is a decision rather than a discovery:

- **Coerce in the seam** — `as_u64()` parses a `String`. Hides the
  difference, and hides it in the place a consumer trusts most.
- **Expose it** — the consumer branches on the backend, which is what the
  seam exists to prevent.
- **Say so in the trait** — document that `MetaValue`'s variant reflects how
  the FORMAT declared a value, not what the value means, and that a
  format-agnostic consumer must use the accessors rather than match on
  variants. Costs nothing at runtime and makes the difference legible.

Not decided here. It needs CireSnave's view because it changes what the
seam promises, and Lightbulb and Fuel are the ones who would live with it.

### Q2 — May two tensors share bytes, and who decides?

**Plan 4 made this decision implicitly, in one backend, and the seam
documents its consequence without owning its rule.**

`mlmf-core/src/report.rs` documents "an overlapping byte range" as a
`TensorDeclined` reason — that is in the SEAM. The sweep that produces it is
in `mlmf-gguf/src/tensors.rs` and nowhere else. `mlmf-core`'s
`TensorContainer` doc says nothing at all about whether two descriptors may
name overlapping ranges.

**In safetensors, sharing is a standard layout, not an anomaly.** Tied
weights — `lm_head` and `embed_tokens` referring to the same bytes — are
common in real models. Neither local file has them, so this corpus cannot
falsify anything here either, but the format permits it and writers use it.

So plan 5 must either:

- **implement the same rule**, and decline valid files with a verdict that
  reads as a defect in the file rather than in the reader; or
- **not implement it**, leaving two backends with different answers to the
  same question, both correct by their own lights, with the seam silent on
  which is right.

Neither is acceptable as an accident. The honest options are that overlap is
a **format-specific** fact (and the seam says so), or a **seam-level** one
(and `TensorContainer` states the rule and both backends follow it). What
must not happen is plan 5 inheriting plan 4's answer as an assumption
because plan 4 was written first.

### Q3 — Is `Encoding` format-neutral, or ggml's geometry with a general name?

GGUF resolves a numeric ggml code through `mlmf-ggml`. Safetensors declares
a dtype STRING and has no block-quantised types at all. Building `Encoding`
from `"BF16"` rather than from code 30 is the same question as Q1, one layer
down, and it is answerable by writing the mapping rather than by discussion.

**The `F8_E4M3` / `F8_E5M2` trap becomes live here.** Same width, same kind,
mutually byte-incompatible. This project's own notes say the exposure stays
dormant "until a format crate maps declared type strings onto `DType`" —
this is that crate, and the mapping must be pinned arm by arm rather than by
width. A recorded trap with no trigger is a claim nobody can check.

## What the corpus cannot do, stated before anyone reads a green

Two files, 491 tensors, **one dtype**, no aliasing, no overlaps, no
non-UTF-8 names, no `F8` anything. It can confirm the layout — and it did,
byte-exactly, on 2.2 GB — and it can falsify almost nothing else.

**Authored fixtures are a prerequisite, not an optimisation.** This is the
second time in this project a corpus has been measured and found unable to
exercise the paths it appears to cover; the first was the GGUF corpus and
byte-exactness. The pattern is not the corpus's fault. Real files are
samples of what writers actually emit, and the interesting inputs are the
ones writers rarely emit.

## Not in scope

- **Sharded models** (`model-00001-of-00002.safetensors` plus an index
  JSON). A different problem — file discovery — and file discovery is
  explicitly not this crate's business.
- **Writing safetensors.**
- **PyTorch pickle**, which the spec plans separately and which needs the
  `Cow` borrow-or-own path `TensorContainer::tensor_bytes` already carries.
