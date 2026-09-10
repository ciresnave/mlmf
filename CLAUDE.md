# MLMF — agent working agreement

Inherits `C:\Projects\CLAUDE.md`. This file adds what is specific to **this** repo;
it may not relax anything above it.

**What MLMF is**, in CireSnave's words and the line every design question resolves
against:

> *"MLMF's job is to read and write model files. MLMF is never intended to be an
> interpreter of the content of model files. That is for things like Fuel."*

Two consequences that come up constantly: **file operations belong outside
`mlmf-core`** — the core is about what is *in* the file, not how it is obtained —
and **deciding what a weight or an activation MEANS is a consumer's job**, not
this crate's.

---

## 1. The §6 fence

> **MLMF may supply a format's documented default. MLMF may never supply a
> model's value.**

A *format default* is citable — the specification says "absent means 32". A
*model value* is a fact about one checkpoint. Supplying one MLMF did not read is
the defect class this repo has spent most of its recent history removing
(#37, #43, #44, #45, #50, #56, #58, #59, #62).

⚠️ **The tell is that the output is indistinguishable from a correct one.** A
fabricated `rope_theta: 10000.0` looks exactly like a declared one; a
`config.json` containing `null` looks exactly like a config file. **A caller
cannot distinguish "the model uses 10000" from "MLMF had nothing to say."**

---

## 2. ⚠️ STANDING POLICY: an absent field is `Option<T>`

**Ruled by CireSnave, 2026-09-10.** Two messages, and the second changed the
test — both are quoted because the difference is the whole scope:

> *"Lets make that a standing policy throughout MLMF: Make the fields a
> supported format may be unable to supply `Option<T>`."*

> *"Actually, I might even go one step further on that: Make the fields a
> supported format may not supply `Option<T>`. That way it doesn't matter if
> they are unable to supply them or someone previously writing the file chose
> not to write those fields to the file, either way MLMF just works."*

### The two tests are not the same, and the second is much wider

| test | asks | property of |
|---|---|---|
| *may be **unable** to supply* | does this FORMAT have a slot for the field? | the **specification** |
| *may **not** supply* ← **this one** | can the field be missing when reading a real FILE? | the **data** |

⚠️ **A format with a slot that a writer left empty is exactly as unreadable as a
format with no slot at all**, and only the second test catches it. Under the
first, GGUF was mostly in the clear — it *has* vocabulary for most of these keys.
Under the ruling actually given, **every key that is optional in practice is in
scope**, which reaches well past ONNX.

### What the policy means in code

- **`None` means "MLMF had nothing to read."** Nothing else.
- **A documented FORMAT default stays a concrete value** — that is §6's permitted
  half, and turning citable defaults into `None` would discard information the
  specification actually gives.
- ⚠️ **The acceptance criterion is his, and it is not "the type compiles":**
  *"either way MLMF just works."* **A caller must never have to know WHY a value
  is absent in order to handle it correctly.** If a consumer has to branch on
  format to interpret a `None`, the policy has not been met.

### Applies to every format, now and later

This is a policy, not a ticket. A new reader added next year is bound by it.

---

## 3. Where the reasoning lives

This file records **decisions**. The measurements behind them live in the issues
and PR bodies, which are durable and citable:

- **#48** — `ModelConfig` cannot represent an absent field. The construction and
  read-site census, and why `Resolution` from `mlmf-core` is the right *concept*
  but not a drop-in.
- **#72** — `save_with_metadata` writes `config.json` containing `null`.
- **#52**, **#53** — GGUF decode refusal, and the tensor-name architecture
  detector.

⚠️ **A ruling that lives only in a conversation evaporates.** #48 was decided
once, in chat, and neither the PM nor this lane could produce the plan
afterwards — it had to be re-taken from scratch. **That is why this file
exists**, and why the quotes above are verbatim rather than paraphrased: a
paraphrase in a decisions file inherits the decider's authority without their
words.

---

## 4. Verification habits this repo has paid for

Short list; the evidence is in the linked PRs.

- **Sabotage every guard.** A test is not trusted until a mutation makes it red,
  and the mutation must be *verified applied* — assert the occurrence count
  before mutating. A sabotage that fails to compile passes every check except
  the one that matters.
- ⚠️ **A test whose expected value is a constant may not notice the
  implementation being replaced by that constant.** Two cases whose answers
  differ is the cheapest fix (#68).
- ⚠️ **Corpus-gated tests must announce a skip, never assert one.** The corpus is
  absent in CI by design; failing there turns "we could not check" into "the code
  is broken". Print `SKIPPED` / `PARTIAL: n of m` so a partial run is
  distinguishable from a full one.
- ⚠️ **A `#[test]` in an `examples/` file does not run under the command CI
  uses.** Measured by planting one and running both forms:

  ```text
  cargo test -p mlmf-conformance              the planted test ran 0 times
  cargo test -p mlmf-conformance --examples   the planted test ran 1 time
  ```

  Every `cargo test` step in `.github/workflows/ci.yml` is the bare `-p <crate>`
  form, so **such a test reads as coverage and executes never**. It is not that
  cargo cannot run them — it is that nothing here asks it to.
- **Non-vacuity before any claim.** "Found no offences" and "walked no files" are
  byte-identical in the output. Assert the population, and cross-check the count
  against an independent enumeration where one exists.

---

## 5. The corpus

`C:/Models/gguf-corpus`, 29 GGUF files. Measured 2026-09-09:

- **9 carry tensors (272 each) and ALL declare `llama`.**
- **19 are vocab-only, carry ZERO tensors**, and hold 13 of the 14 architectures.
- 1 is GGUF v1 and `mlmf-gguf` does not parse it.

⚠️ **So the corpus cannot exhibit any defect that needs a file with tensors AND a
non-llama architecture.** State that limit when a corpus result is the evidence
for a claim — several findings here have turned on it.
