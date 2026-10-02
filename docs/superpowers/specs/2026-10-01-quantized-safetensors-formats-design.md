# GPTQ, AWQ, EXL2 (EXL3 deferred), and MLX — design

Written 2026-10-01. Answers the PM's task relaying CireSnave's survey: add
GPTQ, finish AWQ, add EXL2/EXL3, add MLX, and give fuel a shared
abstraction so it stops per-format-branching. Classified **architectural**
per `superpowers:brainstorming` (new crates, touches how consumers code
against multiple formats). Approaches and this design were reviewed and
approved by the PM before this file was written; see the mlmf lane's
session transcript for that exchange — not re-litigated here.

## 0. The corrected premise

The PM's survey said *"there's no common trait to code against"*. That is
half true, verified against `origin/main` before designing anything on top
of it:

`crates/mlmf-core/src/traits.rs` already declares `TensorContainer`
(name/shape/byte-range per tensor, declared-type-aware, overlap semantics
documented per-format) and `MetadataSource` (declared key/value access).
Both are **already implemented** by `mlmf-gguf` and `mlmf-safetensors`, and
composable with any source crate (`mlmf-source-file`, `mlmf-source-hub`) —
spec §3.1's "any source × any format" is real today for those two formats.

What is actually true:

- None of GPTQ/AWQ/EXL2/EXL3/MLX use this seam yet — there is no reader
  for any of them at all, except AWQ's detection-only scaffold (§1.2).
- The **legacy root crate**'s `src/formats/{onnx_import,pytorch_loader,
  awq}.rs` don't use it either — they're coupled directly to
  `candlelight::{DType,Device,Tensor}` and return a `LoadedModel` struct.
  That code is already scheduled for a rewrite onto this exact seam
  (design spec §11/§12), on its own timeline. **Ruled out of this task's
  scope by the PM**: new crates only, legacy untouched.

So the real task is narrower and better-understood than the survey framed
it: **extend an existing, working abstraction to five more formats**, not
invent one. This matters for the whole shape of what follows — there is no
new trait to design in `mlmf-core`. See §4.

## 1. What's actually on disk, per format — verified against real Hub files

Measured 2026-10-01 against: `TheBloke/dolphin-2.2.1-mistral-7B-GPTQ`,
`TheBloke/Llama-2-7B-Chat-AWQ`, a `LoneStriker` EXL2 repo, three
`turboderp` EXL3 repos, an `mlx-community` MLX repo. None gated — good for
a future test corpus. Source citations are the model repos themselves plus
each quantization library's own kernel source (AutoGPTQ, AutoAWQ),
read directly, not inferred from documentation.

### 1.1 GPTQ

Standard `.safetensors` container — **zero new container-parsing work**;
`mlmf-safetensors` already reads it. Per-layer tensors, confirmed from
AutoGPTQ's `qlinear_cuda.py`: `qweight`, `qzeros`, `scales`, `g_idx`,
optional `bias`. Packing: 8× int4 values per `int32`, **sequential**
bit order (`qweight[row] |= val << (4*(j-i))`).

Metadata is split across **two files that can disagree**: a standalone
`quantize_config.json` (6 fields: `bits`, `group_size`, `damp_percent`,
`desc_act`, `sym`, `true_sequential`) and `config.json`'s embedded
`quantization_config` (11 fields — adds `block_name_to_quantize`,
`disable_exllama`, `model_seqlen`, more). A reader needs both, or at least
needs to not assume either is the sole source.

### 1.2 AWQ

Standard `.safetensors`. Per-layer buffers, confirmed from AutoAWQ's
`gemm.py`: `qweight` `(in, out/8)` int32, `qzeros` `(in/group, out/8)`
int32, `scales` `(in/group, out)` fp16, optional `bias`. Packing: 8× int4
per `int32`, **interleaved** order `[0,2,4,6,1,3,5,7]` — **not GPTQ's
order**, confirmed from source rather than assumed from the shared "8
int4s in an int32" headline fact.

⚠️ **A real, currently-live gap in MLMF's own code, found while verifying
this plan rather than reported against it**: `src/formats/awq.rs`'s
`AWQConfig` (legacy crate) reads only `config.json`'s
`quantization_config{bits,group_size,quant_method,version,zero_point,
modules_to_not_convert}`. A real AWQ repo can instead ship a standalone
`quant_config.json` using **entirely different field names** —
`w_bit` not `bits`, `q_group_size` not `group_size`, the literal string
`"GEMM"` not `"gemm"`. A repo shipping only that second file reads today as
"no quantization config" with no error, because `AWQConfig`'s fields are
all `Option` and nothing requires either file to exist. **Low blast radius
right now** (`load_awq` already hard-refuses to load tensors regardless —
see `src/formats/awq.rs`'s own module doc), but it is a correctness gap in
code that ships, not a hypothetical. **Not fixed here** — the legacy crate
is out of this task's scope by the PM's own ruling (§0) — but `mlmf-awq`
(§3.2) must read both conventions from its first commit, or it inherits
the identical gap in a crate with no excuse for it.

### 1.3 EXL2

Standard `.safetensors`, sharded via `model.safetensors.index.json` —
`mlmf-hf-layout::ShardIndex` already reads this exact file, built this
session for an unrelated reason (#101's migration inventory) and reusable
here unchanged. `config.json`'s `quantization_config`: `{quant_method:
"exl2", version, bits: 8.0, head_bits, calibration:{rows,length,
dataset}}`. **`bits` is a real float in a real file** — 8.0, not a round
number dressed up — confirming the PM's survey on fractional bit-width
without needing to take it on faith.

### 1.4 EXL3 — deferred, not designed

Real name per the project's own README: *"a streamlined variant of QTIP
[...] using a fused Viterbi kernel"*. **"Trellis quantization" is the
survey's paraphrase, not exllamav3's own vocabulary** — correcting it here
so it doesn't ship as if quoted from the source.

**Distribution shape is a genuine forcing example for `mlmf-source-hub`**:
confirmed on three `turboderp` repos, EXL3 ships as **one repo per base
model with per-bpw git branches** (`2.05bpw_h4_ng4`, `3.05bpw_h5_ng5`, …),
`main` holding only a README. This is unlike EXL2 (separate repo per bpw)
and makes `Revision::Ref` branch-pinning (HUB-1) load-bearing rather than
a safety net — there is no way to fetch a specific EXL3 variant without
naming its branch.

**No real EXL3 tensor file was reachable in this verification pass** — the
sampled repos' `main` branches held no weights, and the per-bpw branches
were not fetched. So EXL3's actual on-disk tensor layout is **unverified**.
Given this repo's own standing discipline ("MLMF may never supply a value
it did not read" extends naturally to "this design must never describe
bytes it has not seen"), **EXL3 is explicitly out of this implementation
plan.**

**Named trigger to pick it back up**: a real, reachable EXL3 `.safetensors`
file (any bpw branch) becomes fetchable and its tensor names/dtypes can be
read with `mlmf-safetensors` as a first step, before any decode design is
attempted. Until then this stays a scheduling item, not a closed question
— the same posture this repo already uses for #52 and the GGML-legacy row
before it was superseded.

### 1.5 MLX — smaller than surveyed

Every real `mlx-community` repo checked ships standard, sharded
`.safetensors` + index. **`.npz` is not what current distributions use** —
drop it from scope entirely; it was the survey's assumption, not a
measurement. `config.json` carries `quantization_config: {group_size,
bits, mode: "affine"}`, duplicated verbatim under both a `quantization`
and a `quantization_config` top-level key in at least one real repo (not
yet confirmed as universal — flagged as an open detail for whoever
implements this, not asserted as a rule).

**MLX's actual int4 packing byte order was not verified** — the fork
confirmed the config fields but not the bit-packing order the way it did
for GPTQ and AWQ from their kernel source. §3.4 treats this as unresolved
rather than assuming it matches either GPTQ's or AWQ's order.

## 2. The shape of the problem GPTQ/AWQ/EXL2 share, and why it's not `BlockSpec`

`mlmf-core::{Encoding, BlockSpec}` already exists, already extends by data
("a new scheme is added as data, never as another `Encoding` variant" —
`encoding.rs`'s own doc), and is already how `mlmf-ggml` describes ggml's
block-quantized types to `mlmf-gguf`. The instinct is to add a `BlockSpec`
row per new format. **That doesn't fit, and the reason is structural, not
a missing row:**

- **ggml's quantization is intra-tensor.** One tensor holds interleaved
  scale-and-data blocks, self-contained. `BlockSpec` describes exactly
  that: elements per block, bytes per block, alignment — all facts about
  *one tensor's own bytes*.
- **GPTQ/AWQ/EXL2's quantization is inter-tensor.** A quantized linear
  layer is **three to four cooperating tensors** — `qweight`, `qzeros`,
  `scales`, optionally `g_idx` — each its own `TensorDescriptor` in the
  safetensors container, each independently a plain `Encoding::Dense`
  array (`I32`, `I32`, `F16` respectively) as far as `mlmf-safetensors`
  is concerned, because **that is what the container actually declares**.
  The packed/grouped *meaning* only exists across the set, keyed by a
  shared name prefix (`model.layers.3.self_attn.q_proj`) and interpreted
  against `bits`/`group_size` from a sidecar the container doesn't carry
  at all.

**Decision: `mlmf-core` gains no new type for this.** A `qweight` tensor
is correctly and completely described as `Encoding::Dense(DType::I32)` —
that is what the file declares, and declaring it as anything else would
be this crate inventing structure the format doesn't actually assert at
the container level (the §6 fence, restated one layer up: a format crate
may describe what is declared, never what a consumer will make of it).
The packed/grouped interpretation is **entirely a new crate's job**,
consuming `&dyn TensorContainer` the same way `mlmf-ggml` consumes
`GgmlType` codes — a geometry layer sitting *beside* the seam, not inside
it. This keeps `mlmf-core` exactly as stable as it is today: zero changes,
for the same reason `mlmf-ggml` required zero changes when it landed.

## 3. Proposed crates

Three real crates, one crate deferred to scope-shrink-or-thin status
pending implementation-time investigation, one format explicitly not
designed (§1.4).

### 3.1 `mlmf-gptq`

Given a `&dyn TensorContainer` and a base tensor name prefix, locate
`{prefix}.qweight`/`.qzeros`/`.scales`/optional `.g_idx`, validate their
shapes are mutually consistent for a declared `(bits, group_size)`, and
report a `PackedLinearLayer`-shaped description: logical in/out features,
the four component tensors' names, and the unpack order as a named constant
on the type rather than inline arithmetic a caller has to get right twice
(once for GPTQ, differently for AWQ in §3.2). Not a `BlockSpec` — §2
explains why the shape doesn't fit — but the same spirit `BlockSpec` itself
demonstrates: a format's specific layout is a fact the crate states, not
logic a consumer re-derives. No dequantization: same charter line
`mlmf-ggml` already draws ("turning blocks into floats... is squarely the
business of an inference engine").

Depends on `mlmf-core` and `mlmf-hf-layout` (for a new `quantize_config
.json` / `config.json`'s `quantization_config` reader, built the same way
as this session's `generation_config.rs`/`special_tokens_map.rs`: hand-
parsed `serde_json::Value`, absent-vs-malformed distinguished, every field
`Option<T>`). No new external dependency.

### 3.2 `mlmf-awq`

Same shape as `mlmf-gptq`, AWQ's own interleaved unpack order, and —
the one place this crate must do more than mirror GPTQ — a config reader
that tries **both** real conventions found in §1.2 (`config.json`'s
`quantization_config` and a standalone `quant_config.json` with disjoint
field names) and reports which it found, rather than silently preferring
one. This is the fix for §1.2's live gap, landing in new code rather than
patched into the legacy crate.

### 3.3 `mlmf-exl2`

Same shape again; the geometry math differs because `bits` is a float
(average bitrate across a per-layer calibration-driven mix, not one fixed
width) rather than GPTQ/AWQ's fixed integer bit-width. The group+scale
structure is still inter-tensor the same way. Lower confidence than §3.1/
§3.2: no hand-verification of a real tensor's byte layout was done this
pass (config-level fields only) — the implementation plan's first task
should be exactly that verification, the same way GPTQ/AWQ's bit orders
were confirmed from kernel source before this doc asserted them.

### 3.4 MLX — crate shape undecided, resolve at implementation time

Not yet clear whether MLX's `{group_size, bits, mode: "affine"}` packing
is byte-compatible with GPTQ's or AWQ's order, or is its own third order.
**Decision point for the implementation plan, not this doc**: if MLX's
packing matches an existing crate's order exactly, it is a few rows of
data in that crate (tensor-name convention + a `family` tag), not a new
crate; if it differs, it is a thin `mlmf-mlx` following §3.1's shape. No
`.npz` work in either case (§1.5).

### 3.5 EXL3 — not in this plan

§1.4's trigger stands: a real reachable `.safetensors` file first, a
design second.

## 4. What fuel actually gets

Nothing new to learn. A consumer that already composes `mlmf-gguf`/
`mlmf-safetensors` with `TensorContainer`/`MetadataSource` gets GPTQ/
AWQ/EXL2 support by adding `mlmf-gptq`/`mlmf-awq`/`mlmf-exl2` as
*companions* to its existing `mlmf-safetensors` dependency — same
container reader, same trait calls, a new crate supplying the
"which tensors form one quantized layer, and how is it packed" question
that `mlmf-safetensors` alone cannot answer (correctly: it isn't a
safetensors-container fact, it's format-on-top-of-a-container knowledge,
same layering `mlmf-ggml` already proves out). This is the sense in which
the task's "shared abstraction" framing was right about the *goal* and
wrong about the *gap*: the abstraction fuel needs already exists; what
was missing is formats that produce data shaped to use it.

## 5. Non-goals (recorded so they aren't relitigated)

- **Dequantization/decode kernels.** Same line `mlmf-ggml` already draws.
  These crates report where the bytes are and how they're grouped; turning
  them into floats is an inference engine's job.
- **EXL3 implementation**, until §1.4's trigger fires.
- **`.npz` support for MLX.** Not what real distributions use (§1.5).
- **Retrofitting the legacy crate's `onnx_import.rs`/`pytorch_loader.rs`/
  `awq.rs`.** Ruled out of scope by the PM (§0); already sequenced
  separately by design spec §12.
- **A canonical cross-format quantization-config struct.** Each format's
  `quantization_config` has disjoint fields beyond `bits`/`group_size`,
  and GPTQ/AWQ can't even agree on a key name for the same concept within
  one format. Forcing one shape here is the same mistake design spec §10
  already named and rejected for model configs generally ("N translations
  instead of N−1").

## 6. Open questions for the implementation plan, not this doc

- EXL2's real tensor byte order (§3.3) — first verification task before
  any code.
- MLX's packing order vs. GPTQ/AWQ (§3.4) — decides whether MLX gets its
  own crate.
- Whether `mlmf-hf-layout` is the right home for the three new
  `quantize_config.json`/`quant_config.json`/`quantization_config`-section
  readers, or whether each format crate should own its own — leaning
  `mlmf-hf-layout` for consistency with this session's `generation_config`/
  `special_tokens_map` precedent, not yet decided against a concrete
  second data point.
