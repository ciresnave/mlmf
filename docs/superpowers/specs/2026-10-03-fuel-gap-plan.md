# Fuel-unblocking gap plan: 5 items

**Status: PLAN ONLY, no code.** Written at PM's request after the
fuel-core-dissolution ASK response (2026-10-03), which found these five
pieces of fuel's `quantized`/`hf_config` capability missing or partial in
mlmf today. This document plans them; it does not build them.

**Spec this plan argues from:**
`docs/superpowers/specs/2026-10-01-quantized-safetensors-formats-design.md`
for the §6 fence and `Option<T>` policy that binds every item below (see
`CLAUDE.md` §1–2). The fuel-side context is
`fuel-core-dissolution-mlmf-amendment.md` (fuel `origin/main`) and mlmf's
own `docs/fuel-migration-inventory.md` (`#101`), corrected where this
document's own ASK response found it stale (tokenizer.json, #48).

**Method:** every API sketch below cites a real signature read directly
from `origin/main` at the time of writing (commit `aa8395f`), not a
description. Where a design question is genuinely open, it is named as
open rather than answered — a plan that quietly assigns a shape to an
unread question is worse than one that flags the gap.

---

## Ordering rationale

Ordered by **smallest, most self-contained first**, so Fuel can adopt
capabilities incrementally rather than waiting for all five. Each item
after #1 depends only on already-published primitives (`mlmf-core`,
`mlmf-gguf`, `mlmf-safetensors`, `mlmf-source-file` — all in the PM's
planned publish batch), not on each other, **except** #3 (mmap wiring)
touches the same two crates #1 and #2 read from, so it is sequenced after
them to avoid a merge-conflict-prone three-way race on the same files.

1. GGUF architecture-family classifier — new, small, zero new format
   knowledge (reads metadata mlmf-gguf already exposes).
2. Build-config-from-GGUF-metadata convenience — new, small, thin
   function over existing `mlmf-core`/`mlmf-gguf` primitives.
3. mmap wiring in `mlmf-gguf` + `mlmf-safetensors` — medium, touches
   existing crates' tensor-access path; sequenced after #1/#2 land to
   avoid three changes to the same files in flight at once.
4. Modular hf-config crate, promoted out of the legacy crate — medium,
   extraction + a sabotage-tested rewrite of logic the legacy crate
   already has but without this repo's verification discipline.
5. `tokenizer.json` vocab+merges reader — largest, genuinely new parser,
   no existing mlmf code to extract from.

---

## Item 1: GGUF architecture-family classifier

**Fuel's module:** `fuel-loaders/src/quantized/arch.rs`, 7,040 B.
Classifies a GGUF file into a family (`Llama`, `Qwen2`, `Qwen3`,
`Qwen3Moe`, `Phi`, `Phi3`, `Gemma`, `Gemma3`, `Glm4`, `Lfm2`, `SmolLm3`,
`Gpt2`, `GptNeoX`, `Unknown`) from `general.architecture`, falling back to
tensor-name pattern-matching. Answers "which of Fuel's ~10
`quantized_*` loaders handles this file?" — a coarser question than
mlmf's existing `src/name_mapping.rs::Architecture` (per-tensor schema
mapping, legacy crate, out of scope here).

**Why this is NOT `name_mapping.rs` reused:** confirmed by this plan's own
ASK response (#101 §5) — fuel's question is "which loader," mlmf's
existing enum answers "what does tensor X become," a different and finer
question. A new, smaller type is the right shape, not a repoint onto the
legacy enum.

**Target crate:** new `mlmf-gguf-arch` (format-axis, depends only on
`mlmf-core` + `mlmf-gguf`) — kept separate from `mlmf-gguf` itself because
family classification is architecture-interpretation knowledge layered on
top of a container parser, the same separation `mlmf-gptq`/`mlmf-awq` keep
from `mlmf-core`'s `TensorContainer` seam. Putting it in `mlmf-gguf`
directly would mix "what's in the file" with "what family does this file
belong to," which is closer to the interpretation `CLAUDE.md`'s charter
reserves for consumers — but naming a FAMILY from declared data (not
deciding what a weight MEANS) is squarely data-driven classification, not
interpretation, which is why this is proposed as an mlmf crate rather than
left to Fuel. Flagged as a judgment call for PM/CireSnave to confirm, not
asserted as obviously in-charter.

**API sketch** (built on `mlmf_core::traits::MetadataSource`, confirmed
real at `crates/mlmf-core/src/traits.rs:232`, and
`mlmf_gguf::metadata::GgufMetadata` implementing it at
`crates/mlmf-gguf/src/metadata.rs:379`):

```rust
/// A GGUF file's architecture family, as declared or as inferred from
/// tensor-name patterns when undeclared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Family {
    Llama, Qwen2, Qwen3, Qwen3Moe, Phi, Phi3, Gemma, Gemma3, Glm4, Lfm2,
    SmolLm3, Gpt2, GptNeoX,
}

/// How the family was determined -- a caller deciding whether to trust an
/// inferred family differently from a declared one needs this, same
/// reasoning as `AwqConfig::quant_method`'s `Option<String>` (the §1/§2
/// fence: a caller must be able to tell "the file said X" from "mlmf
/// guessed X").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    Declared,
    InferredFromTensorNames,
}

pub fn classify(
    metadata: &dyn mlmf_core::MetadataSource,
    tensors: &[mlmf_core::TensorDescriptor],
) -> Option<(Family, Basis)> {
    // metadata.get("general.architecture") first; tensor-name pattern
    // match over `tensors` as fallback. `None` when neither succeeds --
    // never a default family, per the §6 fence.
}
```

**Tests:** one fixture per `Family` variant (declared path, using
`FakeMetadataSource`/`FakeContainer` test doubles in the style
`mlmf-gptq`/`mlmf-awq` already use), at least 2 tensor-name-pattern
fallback cases (declared key absent), one "neither declared nor
pattern-matched" → `None` case, one malformed/empty-metadata case. No
corpus dependency needed — this is pure classification logic over
synthetic fixtures, same testing shape as `mlmf-gptq`/`mlmf-awq`'s
`FakeContainer`.

**Size estimate:** ~150–250 LOC production (a match table plus a handful
of tensor-name pattern checks), comparable to `mlmf-gptq`/`mlmf-awq`'s
`config.rs` in scope. Fuel's 7KB source is the ceiling, not the target —
mlmf's version needs no Fuel-specific loader-dispatch logic, only the
classification itself.

**Version impact:** new crate, additive — follows the `mlmf-awq`/`mlmf-gptq`
precedent (workspace version bump, `default-members` + CI wiring, no
feature flag).

---

## Item 2: build-config-from-GGUF-metadata convenience

**Fuel's module:** `fuel-loaders/src/quantized/config_from_gguf.rs`,
16,805 B — the largest file in fuel's `quantized/` directory, "config from
GGUF metadata." #101 §4 already read its `rope_theta` handling directly:
`m.f32("rope.freq_base")?` propagates absence via `?`, not a `10000.0`
fallback (the two `10000.0` literals nearby are test fixtures) — Fuel's
own GGUF config path already meets the §6-fence discipline, which is a
point in favor of this being a safe, mechanical port rather than a
rewrite that has to first un-learn a fabricated-default habit.

**What mlmf has today:** `mlmf-core::meta.rs` + `MetadataSource` give the
generic typed-value read surface GGUF metadata already projects into (per
#101 §4, confirmed still true). What's missing is a single named function
that walks the GGUF KV keys fuel's module reads and assembles them into a
config-shaped result — today a caller has to know which keys to read and
in what combination.

**Target crate:** `mlmf-gguf` itself (not a new crate) — this is reading
MORE of what `GgufMetadata` already exposes through the same
`MetadataSource` trait, not a new capability layer. Depends on Item 4's
crate existing first if the output type is meant to be the same
`ModelConfig`-shaped struct Item 4 produces (see Open question below);
otherwise it can land independently and be re-pointed later.

**API sketch:**

```rust
impl mlmf_gguf::GgufMetadata<'_> {
    /// Every standard inference-relevant key this reader knows to look
    /// for (hidden_size, num_attention_heads, num_key_value_heads,
    /// rope_theta, max_position_embeddings, ...), each `Option<T>` per
    /// the §2 standing policy -- absent is absent, never a plausible
    /// number. Cross-field defaults (num_key_value_heads <-
    /// num_attention_heads when the model is not GQA) are the ONE
    /// documented-convention exception §6 permits, and are applied the
    /// same way Item 4's hf-config crate applies them (see that item --
    /// this function and that crate should share the derivation logic,
    /// not each reimplement it).
    pub fn model_config(&self) -> GgufModelConfig { /* ... */ }
}
```

**Open question, not answered here:** should this return the SAME
`ModelConfig`-shaped type Item 4's hf-config crate produces (one shared
struct, two sources), or its own GGUF-flavored type? Sharing one type
would mean Item 2 depends on Item 4's crate existing first (reversing this
plan's stated order); keeping them separate avoids a cross-item
dependency but risks two slightly-divergent "model config" shapes in the
published surface. Flagged for a ruling at execution time, with a lean
toward "separate for now, unify later if a real caller needs both at
once" — unifying two not-yet-proven-identical shapes prematurely is the
kind of abstraction CLAUDE.md's YAGNI guidance warns against.

**Tests:** one real-GGUF-metadata fixture per field (using the same
synthetic-`MetaValue`-map fixture style `mlmf-gguf`'s own tests already
use, not a corpus file, so this doesn't inherit the corpus's
llama-architecture-only limitation per `CLAUDE.md` §5), one "key absent"
→ `None` test per field (not just the happy path — this is exactly the
defect class #37/#43/#44/#45/#50/#56/#58/#59/#62 already fixed elsewhere
in this repo), one cross-field-default test (num_key_value_heads absent →
defaults to num_attention_heads, mirroring the legacy crate's existing
`src/config.rs` test of the same behavior so the convention doesn't
silently drift between the two implementations).

**Size estimate:** ~200–300 LOC — a struct, a constructor function reading
~10–15 keys, and per-field tests. Fuel's 16.8KB module includes loader
glue mlmf doesn't need; this is the "read the metadata" slice of it only.

**Version impact:** feature addition to `mlmf-gguf`, additive patch per
CireSnave's versioning rule.

---

## Item 3: mmap wiring in `mlmf-gguf` + `mlmf-safetensors`

**Corrected mid-draft — the gap is smaller than first assumed, and the
first version of this section was wrong.** `mlmf-source-file::FileSource`
implements `mlmf_core::ByteSource`, whose trait method `as_bytes(&self) ->
&[u8]` IS public (`crates/mlmf-core/src/traits.rs:16-18`) and, for a
`FileSource` opened with the default `mmap` feature, returns a slice
directly over the live `memmap2::Mmap` (`crates/mlmf-source-file/src/
file.rs:206-209`: `fn as_bytes(&self) -> &[u8] { self.bytes.as_slice() }`).
The *private* method is `Bytes::as_slice` (the enum's own inherent method,
`file.rs:28`) — the trait-level `ByteSource::as_bytes` that wraps it is
not private. **This plan's first draft conflated the two and wrongly
concluded `FileSource` exposes no raw slice at all.** Caught and fixed
before sending, not after — the kind of claim `CLAUDE.md` §6 says needs a
positive control, and the control here is simply reading the trait impl,
which was skipped on the first pass.

So in principle, `mlmf_gguf::GgufMetadata::parse(file_source.as_bytes(),
origin)` already compiles and already borrows zero-copy through the
`Cow<'_, [u8]>` both crates' `tensor_bytes` already returns (confirmed:
`mlmf-gguf/src/tensors.rs:515`, `mlmf-safetensors/src/tensors.rs:548`) —
**with no source change to either crate.** This has not been exercised by
any test today (confirmed: zero references to `mmap`, `as_bytes`, or
`FileSource` in `crates/mlmf-gguf/src/*.rs` or
`crates/mlmf-safetensors/src/*.rs`), so "compiles in principle" is not
yet "verified to work" — the `Cow` could still come back `Owned` for a
reason not yet found (a defensive copy somewhere in the parse path), and
nobody has confirmed the composition end-to-end.

**Revised scope — mostly verification, not new wiring:**
1. Write the round-trip test first (`FileSource::open` on a real file →
   `GgufMetadata::parse(source.as_bytes(), ..)` → `tensor_bytes` →
   assert `Cow::Borrowed`, not just byte-equality). If it passes as-is,
   this item is a **test addition, not new production code** in
   `mlmf-gguf`/`mlmf-safetensors` — a materially different (and smaller)
   outcome than this plan's first draft assumed.
2. If the assertion fails (a copy happens somewhere), THEN find and
   remove it — at that point this becomes a real, small production
   change, scoped by whatever the failing test reveals rather than
   speculated here.
3. Either way, a convenience constructor is worth adding for ergonomics
   — e.g. `GgufMetadata::parse_from(source: &dyn ByteSource, origin: &str)`
   — so a caller doesn't have to know to call `.as_bytes()` themselves.
   Same shape for `mlmf-safetensors`'s header parse.

**Lesson for execution:** start this item with the test, per
`superpowers:test-driven-development` — the test's outcome decides
whether this is a one-file addition or a real bugfix, and guessing the
answer here (as the first draft did) produced a wrong, larger estimate.

**Target crates:** `mlmf-gguf` and `mlmf-safetensors` (existing) — no new
crate, and likely no change to `mlmf-source-file` at all.

**Tests:** the round-trip `Cow::Borrowed` assertion above, for both
crates, against a real file (not a synthetic fixture — the mmap behavior
is the thing under test, and a synthetic in-memory byte vec can't exercise
it).

**Size estimate:** small (1 test per crate) if step 1's test passes as
written; small-to-medium if it doesn't, bounded by whatever the failure
reveals — not the "unknown" this plan's first draft left it as.

**Version impact:** additive to `mlmf-gguf` and `mlmf-safetensors` if any
production code changes; a patch-level test-only change otherwise, per
CireSnave's "every pushed change bumps something" rule.

---

## Item 4: modular hf-config crate, promoted out of the legacy crate

**Fuel's module:** `fuel-loaders/src/hf_config.rs`, 13,538 B —
`config.json` cross-field-default resolution (`head_dim`,
`num_key_value_heads`), using `unwrap_or` for documented architectural
conventions (`hidden / heads` when unspecified), which #101 §4 already
confirmed is the §6-permitted "format default" half, not a fabricated
model value.

**What mlmf has today:** the legacy `mlmf` package's `src/config.rs`
(`HFConfig`/`ModelConfig`) already does exactly this — confirmed directly
this session: `num_key_value_heads` defaults to `num_attention_heads`
when absent (`src/config.rs:337`), `head_dim()` derives from
`hidden_size`/`num_attention_heads`, and **issue #48** ("`ModelConfig`
cannot represent an absent field") is **closed**, fixed by `#77` — so the
`Option<T>` gap the fuel-migration-inventory (`#101`, written against
mlmf `0.5.0`) flagged as blocking this exact migration item no longer
applies. The remaining blocker is purely structural: this logic lives in
the crate `CLAUDE.md` schedules for a REWRITE, not in the tested
`crates/mlmf-*` family, and per `docs/fuel-migration-inventory.md` §2,
"migrating Fuel onto code about to be rewritten means migrating twice."

**Target crate:** new `mlmf-hf-config` (format-axis: no I/O, reads bytes
already supplied, same shape as `mlmf-gptq`/`mlmf-awq`'s config readers).

**API sketch**, deliberately modeled on the fixed `HFConfig`/`ModelConfig`
split already proven in the legacy crate rather than redesigned from
scratch — porting working logic, not reinventing it:

```rust
/// config.json, read permissively -- every field an HF config may omit
/// is Option<T> (CLAUDE.md §2 standing policy), ported from the legacy
/// crate's already-#48-fixed HFConfig/ModelConfig split.
pub struct HfConfig { /* raw declared fields, Option<T> throughout */ }

impl HfConfig {
    pub fn parse(bytes: &[u8]) -> Result<Self, HfConfigError> { /* ... */ }
}

/// Cross-field-derived view: head_dim, kv_head_dim, kv_projection_size,
/// is_gated_ffn, ffn_hidden_size -- every one of these a documented
/// architectural convention (§6-permitted format default), never a
/// value this crate didn't read. num_key_value_heads defaulting to
/// num_attention_heads is the one such convention currently applied;
/// ported verbatim from src/config.rs, not re-derived.
pub struct ResolvedConfig { /* ... */ }

impl HfConfig {
    pub fn resolve(&self) -> ResolvedConfig { /* ... */ }
}
```

**Tests:** port the legacy crate's existing `src/config.rs` test module
(it already has real coverage: `head_dim()` cases, the
`num_key_value_heads` defaulting test, an "unknown head count yields no
head_dim" test) rather than writing from scratch — but each ported test
must be re-verified with this repo's sabotage discipline (`CLAUDE.md`
§4), since #101 §2 explicitly notes the legacy crate's tests predate that
discipline and "no sabotage-tested guards found in `src/saver.rs` or
`src/name_mapping.rs`'s own test module" — the same caveat likely applies
to `src/config.rs` and must be checked, not assumed clean by association
with #48's fix.

**Size estimate:** ~400–600 LOC including tests — largely a port, not new
design, since the shape (`HFConfig` → `ModelConfig`, `Option<T>`
throughout, one documented cross-field default) is already proven
correct by `#48`/`#77`'s fix and the existing legacy-crate test suite.

**Version impact:** new crate, additive. **Does not touch the legacy
`mlmf` package** — per the PM's publish-scope decision, that package
stays unpublished-beyond-0.2.0 and scheduled for its own rewrite; this
item ports its *logic*, not its *code*, into a new crate, leaving
`src/config.rs` as-is (future legacy-crate rewrite is out of this plan's
scope).

---

## Item 5: `tokenizer.json` vocab+merges reader

**Fuel's module:** `fuel-loaders/src/quantized/tokenizer.rs`, 12,071 B.

**Correction this plan's ASK response already made to `#101` §4:** mlmf
has **no** reader for `tokenizer.json`'s actual vocabulary (id↔token
mapping) or BPE merge rules today. `mlmf-meta::vocab` (the module `#101`
cited as "HAS") is a canonical **key-name spelling table** across formats
(e.g. the string `"tokenizer.chat_template"`), confirmed by reading all
110 lines — it contains no vocab/merges data structures at all. This is
the one item in this plan with no existing mlmf code to extract from or
port; it is genuinely new format-reading work.

**Target crate:** new `mlmf-hf-tokenizer` (format-axis: no I/O, parses
supplied bytes, consistent naming with `mlmf-hf-layout`).

**§6-fence note:** `tokenizer.json` is normally fully self-describing (no
documented "format default" analogous to AWQ/GPTQ's `bits`/`group_size`
defaults) — so this crate's risk profile is different from the
quantization-config readers: the main failure mode to guard against is
not fabricating values, but **silently dropping or miscounting entries**
on a malformed or unusually-shaped file (a duplicate token id, a merge
rule referencing a token not in the vocab, an `added_tokens` entry that
collides with the base vocab). The design should specify what happens on
each of those, not leave them as unstated.

**API sketch**, scoped to parsing only (no BPE *algorithm* — applying
merges to tokenize text is interpretation, Fuel's job per the charter,
not mlmf's):

```rust
/// One entry in tokenizer.json's `model.vocab` table, as declared.
pub struct VocabEntry {
    pub token: String,
    pub id: u64,
}

/// One BPE merge rule, as declared, in file order (merge PRIORITY is
/// positional -- order must be preserved, never re-sorted).
pub struct MergeRule {
    pub left: String,
    pub right: String,
}

pub struct TokenizerJson {
    pub vocab: Vec<VocabEntry>,
    pub merges: Vec<MergeRule>,
    pub added_tokens: Vec<AddedToken>, // id, content, special flags -- declared only
    pub malformed: Vec<String>,        // per-entry shape problems, never silently dropped
}

impl TokenizerJson {
    pub fn parse(bytes: &[u8]) -> Result<Self, TokenizerJsonError> { /* ... */ }
}
```

**Tests:** a real `tokenizer.json` fixture (verified against an actual
Hub file, same discipline as `mlmf-awq`/`mlmf-gptq`'s real-checkpoint
fixtures — this needs its own Hub verification pass, not reuse of the
AWQ/GPTQ fixtures, since a BPE tokenizer file is a different artifact),
a duplicate-vocab-id case, a merge rule referencing an unknown token, an
`added_tokens` entry colliding with the base vocab id, a malformed/empty
file, and a case confirming merge order survives round-trip (positional,
not re-sorted).

**Size estimate:** ~600–900 LOC including tests — the largest item here,
comparable to fuel's 12KB module, since it is genuinely new parsing
rather than a port or a thin convenience function.

**Version impact:** new crate, additive.

---

## Summary table

| # | Item | Crate | New or port | Size (LOC est.) | Depends on |
|---|---|---|---|---|---|
| 1 | GGUF arch classifier | `mlmf-gguf-arch` (new) | New | 150–250 | — |
| 2 | config-from-GGUF convenience | `mlmf-gguf` (existing) | New (thin) | 200–300 | Open Q: may want Item 4's type |
| 3 | mmap wiring | `mlmf-gguf` + `mlmf-safetensors` | Mostly test-only (verified mid-draft: `FileSource::as_bytes()` already public) | ~50–150 (test-first; bounded by what the test finds) | sequenced after 1/2 (same files) |
| 4 | hf-config crate | `mlmf-hf-config` (new) | Port + sabotage-retest | 400–600 | — |
| 5 | tokenizer.json reader | `mlmf-hf-tokenizer` (new) | New | 600–900 | — |

None of the five blocks any other — Fuel can adopt them as each lands,
in the order above or reordered by the PM/CireSnave's own priority once
this plan is reviewed.
