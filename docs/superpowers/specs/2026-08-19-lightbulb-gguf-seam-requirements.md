# Lightbulb's requirements for the `mlmf-gguf` metadata seam

**Status:** confirmed and corrected by Lightbulb's architect, 2026-08-19.
Six requirements confirmed complete (there was no seventh in the original set);
R7 below was added by their implementation experience since.
**Date recorded:** 2026-08-19. **Conversation date:** 2026-08-15.
**Consumer:** Lightbulb (`C:\Projects\lightbulb`), reading GGUF metadata to resolve chat templates.

## Why this document exists

These requirements were given to me in conversation and, until now, existed in
no artifact — not an issue, not a commit message, not a doc. The `mlmf-gguf`
plan incorporates them, so a dependency recorded nowhere would have been worse
than one recorded wrongly: there would be nothing to check.

**This is deliberately written as *my* understanding rather than as quotation.**
A restatement the other party merely recognises proves nothing — they would be
matching their own words against a paraphrase and finding them familiar. A
statement I commit to, which they then have to actively agree with, is a check
that can come back negative. Where I am unsure whether something was told to me
or inferred by me, it is marked **[INFERRED]** and needs the sharpest scrutiny.

---

## R1 — Reading metadata must not be able to fail on tensor content

**Priority: highest, by a distance.**

`container.get("tokenizer.chat_template")` must not require decoding a single
tensor dtype. If the container parser must walk the tensor directory at all —
for offsets or alignment — an unrecognized quantization code must degrade
*that tensor's entry*, never fail the open. A consumer who wants only metadata
must not inherit the type table's coverage as a liveness condition.

**The argument, in its corrected form.** Lightbulb originally justified this by
saying Fuel has *no metadata-only entry point*. They later retracted that: the
primitives (`VersionedMagic::read`, `read_string`, `ValueType::from_u32`,
`Value::read`) are all `pub` and re-exported, so a metadata-only reader is
assemblable today in ~25 lines. What is missing is a metadata-only
*convenience function*. The correct argument is therefore ergonomic, and it is
stronger for it:

> A shared reader whose users must bypass it to read a string has failed at its
> job even when the bypass is possible.

N consumers hand-rolling the same 25 lines from the same primitives is the
duplication `mlmf-gguf` exists to prevent, relocated from "a second GGUF
reader" into "a second GGUF *partial* reader" — which is worse, because it is
the one nobody reviews.

**What makes this testable rather than assertable:** reading metadata must be
the *ergonomic* path, not an escape hatch from it. If a future contributor can
make metadata-only reading possible-but-awkward and still pass the suite, the
suite is wrong.

**Concrete failure this prevents**, from Lightbulb's own tree: Fuel's
`Content::read` parses the KV block *first and completely*, then aborts on an
unknown tensor dtype in a later loop. For an IQ-quantized model the chat
template is fully decoded in memory when it is discarded, and the file takes a
"not a valid GGUF" branch — degrading to a family guess whose template has no
end-of-turn marker, which is the exact defect their epic exists to fix.

## R2 — Three distinguishable states, in the return type, not just a log

Key **absent** · key **present but unparseable or wrong-typed** · key **present
and parsed**. If these collapse into one `None`, an operator cannot be told
whether their file lacks a template or has a broken one, and those need
different remedies.

Lightbulb will treat a whitespace-only template as UNDECLARED — an empty
template renders an empty prompt, which reaches the model as "continue
nothing". **That is their policy, not mine.** MLMF hands over raw facts and
does not apply it.

**My commitment:** add to `mlmf-core`

```rust
pub enum Declaration<'a> {
    Absent,
    Unreadable(&'a Unrecognized),
    Declared(&'a MetaValue),
}
fn declaration(&self, key: &str) -> Declaration<'_>;
```

`get()` stays as the ergonomic path; `declaration()` is the honest one.

**Confirmed 2026-08-19.** I had flagged this `[INFERRED]`, suspecting I had
merged "wrong type" and "unparseable". I had not — Lightbulb's own phrasing was
`"key present but unparseable-or-wrong-type"` as a single state. The reading is
faithful and `Declaration` is not a state short.

**But see R4's addendum: the three states are correct *per key*, and the
*join* has seven.**

## R3 — Byte-exact strings. No normalization, ever

Not Unicode, not whitespace, not case. Special tokens are rendered into prompt
**text**, and the tokenizer must then recognise the identical bytes.
Normalization between "what the file declared" and "what we render" produces a
prompt that reads correctly and tokenizes differently, **with no error
anywhere**. If MLMF ever trims or NFC-normalizes a metadata string it must be
opt-in and loud.

**Status in MLMF:** already load-bearing. `MetaValue::Bytes` exists so a
non-UTF-8 declared string survives rather than being lossy-converted.

**The gap this requirement cannot close by itself:** measured across all 29
files in the corpus — **zero non-UTF-8 strings, zero trailing-NUL strings**. So
restoring a `from_utf8_lossy` produces byte-identical output on every real file
available and leaves the suite green. This requirement is untestable against
real files and needs authored adversarial fixtures.

## R4 — Token-id → token-string, with out-of-range reported rather than silently empty

Lightbulb resolves `tokenizer.ggml.bos_token_id` / `…eos_token_id` by indexing
`tokenizer.ggml.tokens`. Three ways it goes wrong — **id absent**, **`tokens`
array absent entirely**, **id out of range** — and all three must be
distinguishable from *resolved to the empty string*, which is itself a
legal-ish outcome they have to warn about.

**My position, agreed by Lightbulb on 2026-08-19 — and the provenance here was
wrong until they corrected it.** I originally recorded this as "my position,
which they accepted". They never accepted it: I proposed it and they replied
without addressing it, and I recorded silence as agreement. They do agree, so
the requirement is right and only its provenance was false — **which is the
more dangerous combination, because nothing downstream would ever have
surfaced it.** Had they disagreed, that sentence would have been the only
record that they were never asked.

**The generalisation, which is why this is written out rather than quietly
fixed:** my `[INFERRED]` marks caught the places I *knew* I was interpolating.
This was a place I did not, because **a proposal that draws no objection feels
like a proposal that was accepted.**

The position itself: MLMF supplies the primitives, not the join. Knowing that a particular integer key indexes a particular array key is
ecosystem knowledge Lightbulb has and MLMF must not fake — it is interpretation,
which the charter puts outside these walls. With `declaration()` plus R5's
accessors, all four outcomes are distinguishable by construction.

### Addendum — the join has seven failure modes, not three

From Lightbulb's implementation since 2026-08-15. Their reader distinguishes,
for BOS/EOS resolution alone:

1. id key absent
2. id present but not readable as an integer **of the type the accessor
   accepts** — their `to_u64` upcasts `U64/U8/U16/U32/Bool` and **bails on
   `I8/I16/I32/I64`**, so an `INT32`-typed id fails while looking like a
   perfectly ordinary id
3. id readable but **too large to index on this platform**
4. `tokens` array absent entirely
5. `tokens` present but **not an array**
6. id in range but the **element is not a string**
7. id genuinely out of range

They found (3) tonight, from an external reviewer, in code that had already
passed two audits, four task reviews and a whole-branch review — a single
silent `?` in a function whose entire purpose is discriminating failure modes.

**This strengthens the primitives-not-join position rather than complicating
it.** Had MLMF supplied the join, MLMF would have to enumerate those seven —
and the knowledge needed to get them right (that a `to_u64` rejects signed
types; that vocabulary indices can exceed `usize` on a 32-bit target) is
precisely the ecosystem knowledge the charter puts outside these walls.

## R5 — Indexed array access without materializing

`tokenizer.ggml.tokens` is the full vocabulary; Lightbulb needs two elements
from it. Reading BOS must not walk 32k values, and should not allocate them
either.

**My commitment:**

```rust
fn array_len(&self, key: &str) -> Option<u64>;            // None = absent or not an array
fn array_get(&self, key: &str, index: u64) -> Option<MetaValue>;
```

plus lazy per-key materialization: index the KV block at open as
`(key, type, byte range)` and decode a value only when asked. Open becomes
O(number of keys) rather than O(vocab size).

**I promised to measure this on the corpus and send the number rather than
claim it is fast.** Not yet done.

## R6 — No accessor may shadow a `std` trait method with different semantics

From a bug Lightbulb nearly shipped: Fuel's `Value::to_string()` returns
`Result<&String>`, not `String`. It reads as `ToString::to_string` at every call
site and is not. They caught it only by reading source.

**Status in MLMF:** `mlmf-core` already uses `as_str` / `as_u64` / `as_f64`
returning `Option`. Nothing is or will be called `to_string`.

---

## Design commitments I made in response

These are mine, not Lightbulb's requirements — recorded so the plan and the
promise cannot drift apart.

1. **Staged parse.** magic+version → KV block → tensor directory. Only the
   stage that fails, fails. The metadata stage has no access to the type table,
   so it cannot fail against it — R1 holds by *shape*, not by discipline.
2. **Unrecognized tensor handling.** Excluded from `tensors()`, reported via
   `UnrecognizedKind::TensorEncoding { name, family, code }`. **Landed in
   `mlmf-core` already** (final review of the `mlmf-ggml` plan), along with the
   contract on `TensorContainer::tensors()`.
3. **Type lookup is the last step, not the first.** The container reads name,
   dims and offset out of the tensor info *before* consulting the table, so a
   report entry is complete whether or not the code resolves — and no geometry
   is reconstructed anywhere.
4. **`CodeStatus` is three-state**, so a retired code is not reported as an
   unknown one. Eight ggml codes are retired; telling someone holding a 2023
   file to upgrade is a true-sounding message about the wrong cause.

## R7 — Distinguish "not the file it claims to be" from "malformed GGUF"

**Added 2026-08-19 from Lightbulb's implementation, not part of the original
six.**

They gate on `is_file()` plus a case-insensitive `.gguf` extension before ever
calling MLMF, so *"not a GGUF at all"* — the common, benign case of a directory
checkpoint — is answered before the seam sees it. **But that gate is a
heuristic, and its residue lands here:** a file *named* `.gguf` whose magic
bytes are not GGUF passes their check and arrives at the reader.

So what must be distinguishable at the MLMF boundary is:

- **magic or version unreadable** — this file is not what its name claims
- **valid container, content unreadable** — this is a GGUF and something inside
  it is wrong

Those carry different operator remedies: *"you pointed me at the wrong file"*
versus *"your file is malformed"*. Fuel collapses them today — both surface as
one `Err` from `from_path` — which forces Lightbulb's warning to hedge across
both with "if the file is otherwise sound".

**Cheap here, impossible there.** The distinction is free at parse time,
because the staged parse already separates magic+version from everything after
it, and unrecoverable afterwards.

## Corrections wanted

Reply against the numbers. In particular:

- **R2's [INFERRED] flag** — did you separate "wrong type" from "unparseable"?
- **Is anything here something I inferred rather than was told?** That is the
  most valuable thing this document could surface.
- **Is anything missing?** Six is my count, not necessarily yours.
- **Has your epic's implementation changed any of these** since 2026-08-15?
