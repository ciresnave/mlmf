> # ⚠️ SUPERSEDED — DO NOT IMPLEMENT THIS FILE
>
> **This is the two-half plan, kept verbatim as the record of two audit rounds. It was RENAMED, not rewritten, and nothing below was changed.**
>
> **What replaced it:**
>
> - **The shard-index half** — `Task 4` below — was extracted, corrected and now lives in **`2026-09-05-mlmf-hf-layout-shards.md`**. ⚠️ **Do not build `src/shards.rs` or `tests/shards.rs` from Task 4 below.** Its interface returns `Result<Self, HfError>`, a shared error type that **does not compile**: private fields are module-scoped, so a second module cannot construct it. Part A uses a `ShardError` of its own. Task 4 also derives `Eq` on a type holding `MetaValue`, which has `F32`/`F64` and derives only `PartialEq`.
> - **The metadata half** — `HfLayout`, the sidecars, and `mlmf-meta`'s `BOS_TOKEN_TEXT` — **has not been rewritten yet.** When it is, it is written **fresh**, not patched from here.
>
> **Why it is kept.** Two audits found twelve blocking findings between them, and **four of the second round's six were created by the first round's fixes.** The revision table at the top of the plan and the findings in the body are the record of that, and deleting them would leave the next author to rediscover them. ⚠️ **In particular: `token_decl` below has four tests and three sabotages and is called by NOTHING** — the defect that forced the split, and the one a fresh part B has to avoid re-creating.

# `mlmf-hf-layout` — HuggingFace checkpoint layout (§12 step 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a checkpoint's filenames and the bytes of its JSON, report **where each tensor lives** and **what the checkpoint declared** — the two halves spec §5.5 and §5.7 give this crate.

**Architecture:** Format axis — **bytes in, structure out, no I/O, never enumerates a directory.** The caller supplies filenames and bytes (`mlmf-source-file` reads them). Metadata keys are spelled `<filename>:<key>`, which is **what spec §5's table already uses**, so `mlmf-meta`'s `Format::HuggingFace` rows apply unchanged.

```
mlmf-source-file (I/O)   ->  filenames + bytes
mlmf-hf-layout (format)  ->  ShardIndex   -- where each tensor lives   (Fuel)
                             HfLayout     -- a MetadataSource          (Lightbulb)
mlmf-meta (structure)    ->  TemplateSet / SpecialTokens
```

**Tech Stack:** Rust **edition 2024** (workspace, inherited), `mlmf-core` + `serde_json`, following `mlmf-safetensors`' precedent exactly.

**Spec:** `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md` — **§5.5 line 90** (this crate's definition), **line 157** (why it exists), §5 (vocabulary), §9 §1 including **1.4 as corrected by PR #14**, §9 3.1, §12 step 5.

---

## ⚠️ Revision note — this plan was audited once and rewritten

**Audited at `53de1ed`: SIX blocking findings.** They are recorded here because three of them are hazards the implementer can re-create.

| # | Finding | Fixed by |
|---|---|---|
| B1 | Six vocabulary rows were **inert** — `SpecialTokens::extract` never reads them. Measured: `text: None` on both real checkpoints. **The sabotage mutated `TABLE`, which only exercises `spelling()`** — AD-2 satisfied, feature dead, all tests green | Task 6 changes `extract` and sabotages **`extract`**, not the table |
| B2 | `filter_map` **silently emptied arrays of objects**. Measured: `added_tokens` came back `Array(len 0)` where the files declare 3 and 17 | Task 2: an array with any unrepresentable element is unrepresentable **as a whole**, and is recorded |
| B3 | The plan's **own clippy gate rejected its own tests** — an unused import, and three `ONE_INSTANCE` names under `-D warnings` | snake_case `_one_instance_*`; no unused imports |
| B4 | `pub fn token_decl(v: &serde_json::Value)` — a foreign type in the public API, **two lines after the constraint forbidding it**, caught by the plan's own Task 7 check | `pub(crate)` |
| B5 | **The branch was 3 commits behind `main`** — `git pull … 2>/dev/null` had failed on a stat-dirty `Cargo.lock` and the error was discarded. Task 4 modified a crate absent from its own tree | Rebased. ⚠️ **Never `2>/dev/null` a command whose success you then depend on** |
| B6 | Task 5 told the implementer to update a counting line **`main` had deliberately deleted one commit earlier**, for exactly this reason | Task 7 says leave that header alone |

**And the scope error, which is the largest:** the first draft implemented only the metadata half and called it "§12 step 5 → the whole plan." **Spec line 90 defines this crate by the SHARD-INDEX half, and §12 step 5 says it is *"Fuel's actual dependency."*** Both halves are in scope now.

---

## Global Constraints

- **Axis: `format`.** No I/O. **No `Path`, `PathBuf`, or `std::fs` in `src/`** — spec line 90: *"It never enumerates a directory."*
- **`[dev-dependencies]` is refused for every gated member** (`deps.rs`). Tests needing two crates in one binary go in `mlmf-conformance`.
- **`serde_json` never appears in a `pub` signature.** `mlmf-safetensors/src/header.rs:52` states the rule and the reason; that crate keeps `serde_json::Map` `pub(crate)`.
- **§5 rules.** 1: unknown keys survive; dropping them is *"the worst kind of lossy — invisible."* 2: absent means `None`, never a default. 3: lossiness **per key**.
- **§9 3.1 — dtype coercion is consumer policy.** Nothing here touches tensor **values**; the shard index reports **locations only**.
- **AD-2.** Verify each mutation with `cmp` against an **`mktemp`** copy — **never `git diff --stat`**, which prints nothing for an untracked file. Then read the **`failures:` block**, not the exit code: a build break also exits non-zero and changes the file.
- ⚠️ **Sabotage the CODE THAT RUNS, not a table it reads.** B1 is the whole reason: a mutation that perturbs an artefact the feature never consults produces a red that proves nothing about the feature.
- **Corpus rule — name the scan root beside every number.**
- **Unobserved shapes are labelled in the TEST NAME**, snake_case: `..._one_instance_llama2`.
- **Gate every commit on `local-gates.sh`'s EXIT CODE**, which is `exit "$failed"`:

```bash
bash scripts/local-gates.sh; rc=$?
if [ "$rc" -ne 0 ]; then echo "$rc gate(s) FAILED -- do not commit"; fi
```

  ⚠️ **Not `| tail -1`.** On a green run **that emitted a SKIPPED notice**, the last line is *"...but read the SKIPPED notice(s) above"*, so a `tail`-based match reports RED on a passing run. Task 6's corpus tests emit exactly that notice.

---

## Measured facts

### Half A — the shard index. Two instances, fetched 2026-09-05, **zero on this machine**

⚠️ **`find C:/Models ~/.cache/huggingface -name '*.index.json'` → 0**, positive control passing (it finds both `model.safetensors` that exist). **Both local checkpoints are single-shard, so every local test of this half is a fixture.** These two were fetched from the Hub because a shape derived from a fixture one wrote is not evidence.

| | Mistral-7B-Instruct-v0.2 | Qwen2.5-7B-Instruct |
|---|---|---|
| size / shards | 25,125 B / 3 | 27,752 B / 4 |
| top-level keys | `metadata`, `weight_map` — **in that order, and nothing else** (tail read separately) | same |
| `metadata.total_size` | `14483464192` | `15231233024` |
| `weight_map` | **flat `{tensor_name: filename}`**, string→string. No nesting, no arrays, no per-tensor objects | same |
| indentation | 4 spaces | 2 spaces |

⚠️ **BOTH `total_size` values exceed 2³². A `u32` wraps silently. Use `u64`.**

⚠️ **A SINGLE LAYER SPLITS ACROSS SHARDS.** Mistral layer 10, measured:

    model.layers.10.mlp.gate_proj.weight            -> model-00001-of-00003
    model.layers.10.self_attn.k_proj.weight         -> model-00001-of-00003
    model.layers.10.input_layernorm.weight          -> model-00002-of-00003
    model.layers.10.post_attention_layernorm.weight -> model-00002-of-00003

**Any implementation that assumes a layer lives in one file, or derives a shard from a layer index, is wrong on the first real multi-shard checkpoint.** The map is **per-tensor and only per-tensor**. `lm_head.weight` is in the *last* shard while `model.embed_tokens.weight` is in the *first*, so map order is lexical and carries **no locality**.

⚠️ **The tensor-name SET is architecture-dependent** — Qwen has `self_attn.{k,q,v}_proj.bias` entries Mistral lacks entirely. **No test may enumerate expected tensor names.**

⚠️ **"Exactly two top-level keys" is TWO INSTANCES, not a closed set.** The safetensors convention documents `metadata` as an open object. **Parse `metadata` permissively; do not assume `total_size` is its only member** — §5 rule 1 applies here precisely. **And do not encode indentation:** the two files differ, which is what a hand-written fixture would have hidden.

### Half B — the metadata sidecars. Six checkpoints

Local: `C:/Models/TinyLlama-1.1B-Chat-v1.0`, `C:/Models/SmolLM2-360M-Instruct`. Remote, fetched and re-verified from the bytes: `NousResearch/Hermes-3-Llama-3.1-8B`, `NousResearch/Llama-2-7b-chat-hf`, `openai/gpt-oss-20b`, `google/gemma-3-4b-it`.

| checkpoint | `bos_token` | `add_bos_token` | `chat_template` |
|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | string `<s>` | absent | string, **410 B** |
| SmolLM2-360M-Instruct | string `<\|im_start\|>` | absent | string, **368 B** |
| Hermes-3-Llama-3.1-8B | string | absent | **array** — `default` + `tool_use` |
| Llama-2-7b-chat-hf | **`AddedToken` object**, text in `content` | **`true`** | absent |
| openai/gpt-oss-20b | string | absent | absent — **`.jinja` sidecar is the only source** |
| google/gemma-3-4b-it | *not measured* | *not measured* | **`chat_template.json` sidecar** |

**Both local checkpoints, verified:** `config.json` `bos_token_id=1`, `eos_token_id=2`; `tokenizer.json`'s `added_tokens[1]`/`[2]` equal the strings `tokenizer_config.json` declares directly.

⚠️ **POPULATION — scan root `C:/Models`. FIVE shapes have exactly one instance:**

    added_tokens id -> text        10 files   ⚠️ but only 2 are NAMED tokenizer.json;
                                              the other 8 are hf-tokenizers/<name>.json.
                                              A scan for "tokenizer.json" finds 2, not 10.
    the two-path cross-check        2 checkpoints (3 files each)
    chat_template as an ARRAY       1   Hermes-3
    AddedToken OBJECT token         1   Llama-2-7b-chat-hf
    .jinja sidecar                  1   gpt-oss-20b
    .json sidecar                   1   gemma-3-4b-it
    add_bos_token PRESENT           1   Llama-2-7b-chat-hf

**One instance proves a shape EXISTS. It says nothing about frequency and cannot distinguish "the convention" from "this checkpoint is unusual."**

**The `add_bos_token` history is the caution:** absent in four consecutive checkpoints, and it was nearly recorded as GGUF-only. **One presence ended it, and it was the first key of the first file fetched to look for it.** *Spend the search on the disconfirming case.*

### The two list forms are opposite structures sharing a word

    HF     "chat_template": [ {"name":"default","template":"..."}, ... ]
           an array of OBJECTS CARRYING BODIES; the default is a NAMED entry
    GGUF   tokenizer.chat_templates = ["tool_use","rag"]   NAMES only
           tokenizer.chat_template  = the default          UNNAMED, not in the array

**`mlmf-meta` handles the GGUF form. It must not be reused for HF.**

---

## File structure

```
crates/mlmf-hf-layout/
  Cargo.toml · tests/{axis,allowed-std.list,direct-deps.allow}
  src/lib.rs        module registration ONLY -- no `///` on `pub mod` lines
  src/value.rs      Task 1  JSON -> MetaValue; the AddedToken shape
  src/source.rs     Task 2  HfLayout: a MetadataSource keyed "<file>:<key>"
  src/sidecar.rs    Task 3  both sidecar spellings
  src/shards.rs     Task 4  ShardIndex -- where each tensor lives
  tests/{value,source,sidecar,shards}.rs
crates/mlmf-meta/   Task 5-6  BOS_TOKEN_TEXT, and extract that actually reads it
crates/mlmf-conformance/tests/hf_corpus.rs   Task 7
```

⚠️ **No `///` above the `pub mod` lines.** An outer doc merges with the module's `//!` and the merged text resolves in the **parent's** scope, breaking that module's intra-doc links under `cargo doc -D warnings`. Cost one red commit in plan 7.

---

### Task 0: Crate skeleton, gated from its first commit

- [ ] **Step 1: Baseline.** `cargo test -p mlmf-core --test ci_coverage --test workspace --test purity --test deps` → 20 tests, 0 failures. ⚠️ `ci_coverage` reports **tests**, not members; on `main` it is 6 tests over 7 gated crates, so any equality is a coincidence.

- [ ] **Step 2: Create the crate.**

```toml
[package]
    description          = "HuggingFace checkpoint layout: where each tensor lives, and what the sidecars declared. Bytes to structure; no I/O."
    edition.workspace    = true
    license.workspace    = true
    name                 = "mlmf-hf-layout"
    repository.workspace = true
    version.workspace    = true

# NO [features]. mlmf-core has `default = []` and no `std` feature, so
# `std = ["mlmf-core/std"]` fails cargo at RESOLUTION for EVERY member.

[dependencies]
    mlmf-core  = { path = "../mlmf-core", version = "0.4.0" }
    serde_json = { version = "1.0", default-features = false, features = ["std"] }
```

```rust
// src/lib.rs
//! HuggingFace checkpoint layout.
//!
//! Two questions, both answered from bytes the caller supplies: **where
//! does each tensor live** ([`shards`]), and **what did the checkpoint
//! declare** ([`source`]). Finding the files belongs to `mlmf-source-file`;
//! reading the declarations belongs to `mlmf-meta`.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
```

`tests/axis` → `format`. `tests/direct-deps.allow` → header + `mlmf-core`, `serde_json` **sorted** (positional `assert_eq!`). `tests/allowed-std.list` → **must exist** (`purity.rs::allowed_std` panics otherwise); start empty and add only what the gate demands.

Root `Cargo.toml` `default-members`: **`crates/mlmf-hf-layout` sorts after `crates/mlmf-gguf`, before `crates/mlmf-meta`.**

- [ ] **Step 3: `ci_coverage` must FAIL naming `mlmf-hf-layout`.** Four modes: the intended red · a feature-resolution error (delete `[features]`) · `must declare its C3/C2 allow-list` (files missing) · **everything passes → stop and report.**

- [ ] **Step 4: Wire CI — four steps**, copying `mlmf-conformance`'s shape, including `RUSTDOCFLAGS: -D warnings` in `env:` on the doc step. **Four, not six** — the second-configuration test arms only on a non-empty `default`.

- [ ] **Step 5: `cargo fmt --all`, gate on the exit code, commit.**

---

### Task 1: JSON → `MetaValue`, and the `AddedToken` shape

**Files:** create `src/value.rs`, `tests/value.rs`.

**Interfaces produced** — all `pub(crate)`, because every one names `serde_json::Value`:
- `pub(crate) struct TokenDecl { pub text: String, pub was_object: bool }` — `#[derive(Debug, Clone, PartialEq, Eq)]`
- `pub(crate) fn token_decl(v: &serde_json::Value) -> Option<TokenDecl>`
- `pub(crate) fn meta_value(v: &serde_json::Value) -> Option<MetaValue>`

⚠️ **`TokenDecl` is re-exported publicly from `source.rs` as a plain struct.** `value.rs` is the seam that touches the foreign type; nothing crossing the crate boundary mentions it.

- [ ] **Step 1: Write the failing test.** (Unit tests, because the API is `pub(crate)`.)

```rust
// at the bottom of src/value.rs
#[cfg(test)]
mod tests {
    use super::*;

    fn json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).expect("test literal parses")
    }

    #[test]
    fn a_string_token_is_the_text_itself() {
        // 4 of 6 checkpoints, measured 2026-09-05.
        let d = token_decl(&json(r#""<s>""#)).expect("declared");
        assert_eq!(d.text, "<s>");
        assert!(!d.was_object);
    }

    #[test]
    fn an_added_token_object_yields_its_content_one_instance_llama2() {
        // POPULATION OF ONE: NousResearch/Llama-2-7b-chat-hf. The name says
        // so because a pass proves the shape is HANDLED, not that it is
        // common. §9 1.4 (corrected, PR #14): a string OR an AddedToken
        // object whose `content` holds the text. A reader accepting only a
        // string returns "not declared" for a checkpoint that DOES declare
        // it -- a fact about the reader reported as a fact about the file.
        let d = token_decl(&json(
            r#"{"__type":"AddedToken","content":"<s>","lstrip":false,
                "normalized":true,"rstrip":false,"single_word":false}"#,
        ))
        .expect("declared, one indirection deeper");
        assert_eq!(d.text, "<s>");
        assert!(d.was_object);
    }

    #[test]
    fn an_object_without_a_string_content_is_not_a_token() {
        assert!(token_decl(&json(r#"{"__type":"AddedToken"}"#)).is_none());
        assert!(token_decl(&json(r#"{"content":42}"#)).is_none());
    }

    #[test]
    fn a_number_or_null_is_not_a_token() {
        assert!(token_decl(&json("42")).is_none());
        assert!(token_decl(&json("null")).is_none());
    }

    #[test]
    fn a_json_string_that_looks_numeric_stays_a_string() {
        // §5: a format that did not declare a number did not declare one.
        assert_eq!(meta_value(&json(r#""32""#)), Some(MetaValue::String("32".into())));
    }

    #[test]
    fn an_array_with_an_unrepresentable_element_is_unrepresentable_whole() {
        // THE B2 DEFECT. An earlier draft used filter_map here, which
        // silently dropped objects and nulls. MEASURED against the real
        // files: tokenizer.json:added_tokens came back Array(len 0) while
        // the checkpoints declare 3 and 17 entries, and a mixed array
        // ["a",{"x":1},"b",null,"c"] came back as 3 elements with the
        // indices SHIFTED -- element 2 read "b" where the file said null.
        //
        // An array is all-or-nothing: partial is worse than absent, because
        // absent is visible and partial looks complete.
        assert_eq!(meta_value(&json(r#"["a","b"]"#)),
                   Some(MetaValue::Array(vec![MetaValue::String("a".into()),
                                              MetaValue::String("b".into())])));
        assert_eq!(meta_value(&json(r#"["a",{"x":1},"b"]"#)), None);
        assert_eq!(meta_value(&json(r#"["a",null]"#)), None);
        assert_eq!(meta_value(&json("[]")), Some(MetaValue::Array(vec![])));
    }

    #[test]
    fn a_big_integer_stays_exact() {
        // metadata.total_size is 14483464192 on Mistral -- above 2^32.
        assert_eq!(meta_value(&json("14483464192")), Some(MetaValue::U64(14_483_464_192)));
    }
}
```

- [ ] **Step 2: Run.** `cargo test -p mlmf-hf-layout --lib` → FAIL, no such module.

- [ ] **Step 3: Implement.**

```rust
// src/value.rs
//! JSON values, converted without interpretation.
//!
//! Everything here is `pub(crate)`: every signature names
//! `serde_json::Value`, and a foreign type in this crate's public API would
//! make its consumers depend on a version of `serde_json`. `mlmf-safetensors`
//! states the same rule at `src/header.rs:52`.

use mlmf_core::MetaValue;

/// A token declared in `tokenizer_config.json`, in either of its shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenDecl {
    /// The token text.
    pub text: String,
    /// Whether it arrived as an `AddedToken` object rather than a bare
    /// string. Reported rather than smoothed away: the two shapes are the
    /// same datum, and a consumer auditing a checkpoint may want to know
    /// which the file used.
    pub was_object: bool,
}

/// `bos_token` / `eos_token` / `unk_token`, in either shape (§9 1.4).
pub(crate) fn token_decl(v: &serde_json::Value) -> Option<TokenDecl> {
    if let Some(s) = v.as_str() {
        return Some(TokenDecl { text: s.to_string(), was_object: false });
    }
    let content = v.as_object()?.get("content")?.as_str()?;
    Some(TokenDecl { text: content.to_string(), was_object: true })
}

/// A JSON value as a [`MetaValue`], **without parsing strings**.
///
/// Returns `None` for anything with no faithful representation — an object,
/// a null, or **an array containing either**. The array case is
/// all-or-nothing deliberately: a partially-converted array reports a
/// length the file never declared and shifts every index after the drop,
/// which is worse than absence because absence is visible.
pub(crate) fn meta_value(v: &serde_json::Value) -> Option<MetaValue> {
    Some(match v {
        serde_json::Value::String(s) => MetaValue::String(s.clone()),
        serde_json::Value::Bool(b) => MetaValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                MetaValue::U64(u)
            } else if let Some(i) = n.as_i64() {
                MetaValue::I64(i)
            } else {
                MetaValue::F64(n.as_f64()?)
            }
        }
        serde_json::Value::Array(a) => {
            // `collect::<Option<Vec<_>>>()` is the all-or-nothing part.
            MetaValue::Array(a.iter().map(meta_value).collect::<Option<Vec<_>>>()?)
        }
        serde_json::Value::Null | serde_json::Value::Object(_) => return None,
    })
}
```

Add `mod value;` to `lib.rs` — **private, and no `///` above it.**

- [ ] **Step 4: Run.** Expected: **7 passed**.

- [ ] **Step 5: Sabotage (AD-2), three times.** Each mutates **the function under test**, not a table it reads.

```bash
# (a) Accept only the string shape -- the exact §9 1.4 defect.
BAK=$(mktemp); cp crates/mlmf-hf-layout/src/value.rs "$BAK"
perl -0pi -e 's/let content = v\.as_object\(\)\?\.get\("content"\)\?\.as_str\(\)\?;/return None;\n    #[allow(unreachable_code)] let content: \&str = "";/' crates/mlmf-hf-layout/src/value.rs
rc=0; cmp "$BAK" crates/mlmf-hf-layout/src/value.rs || rc=$?
case $rc in 0) echo '!! MUTATION DID NOT APPLY -- fix the pattern, do not proceed' ;;
             1) ;; *) echo '!! GUARD BROKEN -- no backup; re-run the cp' ;; esac
cargo test -p mlmf-hf-layout --lib 2>&1 | tail -14 || true   # exit code is NOT the signal
cp "$BAK" crates/mlmf-hf-layout/src/value.rs && rm -f "$BAK"
```
Expected: **FAIL** on `an_added_token_object_yields_its_content_one_instance_llama2`.

```bash
# (b) Restore the B2 defect: filter_map instead of all-or-nothing.
BAK=$(mktemp); cp crates/mlmf-hf-layout/src/value.rs "$BAK"
perl -0pi -e 's/a\.iter\(\)\.map\(meta_value\)\.collect::<Option<Vec<_>>>\(\)\?/a.iter().filter_map(meta_value).collect()/' crates/mlmf-hf-layout/src/value.rs
rc=0; cmp "$BAK" crates/mlmf-hf-layout/src/value.rs || rc=$?
case $rc in 0) echo '!! MUTATION DID NOT APPLY' ;; 1) ;; *) echo '!! GUARD BROKEN' ;; esac
cargo test -p mlmf-hf-layout --lib 2>&1 | tail -14 || true
cp "$BAK" crates/mlmf-hf-layout/src/value.rs && rm -f "$BAK"
```
Expected: **FAIL** on `an_array_with_an_unrepresentable_element_is_unrepresentable_whole` — `["a",{"x":1},"b"]` yields `Some(Array(["a","b"]))` where `None` is required.

```bash
# (c) Parse a numeric-looking string.
BAK=$(mktemp); cp crates/mlmf-hf-layout/src/value.rs "$BAK"
perl -0pi -e 's/serde_json::Value::String\(s\) => MetaValue::String\(s\.clone\(\)\),/serde_json::Value::String(s) => s.parse::<u64>().map_or_else(|_| MetaValue::String(s.clone()), MetaValue::U64),/' crates/mlmf-hf-layout/src/value.rs
rc=0; cmp "$BAK" crates/mlmf-hf-layout/src/value.rs || rc=$?
case $rc in 0) echo '!! MUTATION DID NOT APPLY' ;; 1) ;; *) echo '!! GUARD BROKEN' ;; esac
cargo test -p mlmf-hf-layout --lib 2>&1 | tail -14 || true
cp "$BAK" crates/mlmf-hf-layout/src/value.rs && rm -f "$BAK"
```
Expected: **FAIL** on `a_json_string_that_looks_numeric_stays_a_string`.

- [ ] **Step 6: `cargo fmt --all`; `cargo clippy -p mlmf-hf-layout --all-targets -- -D warnings`; commit.**

---

### Task 2: `HfLayout` — a `MetadataSource` keyed `<filename>:<key>`

**Files:** create `src/source.rs`, `tests/source.rs`.

**Interfaces produced** — the full signatures, so nothing is guessed:

```rust
pub struct HfLayout { /* private */ }
pub struct HfLayoutBuilder { /* private */ }
#[derive(Debug)] pub struct HfError { /* private */ }

impl HfLayout {
    pub fn builder() -> HfLayoutBuilder;
    /// Keys the checkpoint declared that have no `MetaValue` representation.
    /// Sorted. Also reachable through `MetadataSource::declaration`, which
    /// returns `Declaration::Unreadable` for exactly these.
    pub fn unrepresentable(&self) -> &[String];
}
impl HfLayoutBuilder {
    pub fn file(self, name: &str, bytes: &[u8]) -> Result<Self, HfError>;
    pub fn build(self) -> HfLayout;
}
impl core::fmt::Display for HfError { /* names the file and the parse error */ }
impl std::error::Error for HfError {}
impl mlmf_core::MetadataSource for HfLayout { /* get, keys, index_complete, declaration */ }
```

⚠️ **`file` takes BYTES, never a path.** Spec line 90: *"It never enumerates a directory."* A `Path` here is the axis violation C3 exists to catch.

⚠️ **`declaration()` is OVERRIDDEN, not defaulted.** `mlmf-core` provides `Declaration::Unreadable(&Unrecognized)` precisely for a key the file declared but this crate cannot represent. **The default impl returns `Absent`, and a consumer holding `&dyn MetadataSource` — which is this crate's entire purpose — could not otherwise tell "the checkpoint did not say" from "the checkpoint said something I cannot carry."**

- [ ] **Step 1: Write the failing test** covering: the `<file>:<key>` spelling; two files not colliding and unqualified keys **not** resolving; unknown keys surviving (§5 rule 1); a nested object appearing in `unrepresentable()` **and** as `Declaration::Unreadable` through the trait, with a control that a genuinely absent key is `Declaration::Absent`; malformed JSON naming its file; a valid non-object top level (`b"42"`, `b"[1,2]"`) being an error that names the file; and `index_complete() == true`.

⚠️ **`index_complete()`'s justification, corrected from the first draft:** it is `true` because **`serde_json` parses a document whole or fails** — so no key is missed for having stopped early. It is **not** because every declared key is retrievable via `get`: an unrepresentable key returns `None` from `get` while being `Unreadable` from `declaration`. **Those are different questions and the first draft conflated them.**

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** `Vec<(String, MetaValue)>` plus a sorted `Vec<String>` of unrepresentable keys; `get` is a linear scan (a handful of files, tens of keys — no map). `HfError` stores the filename and the `serde_json` message as a **`String`**.
- [ ] **Step 4: Run.**
- [ ] **Step 5: Sabotage, each on `source.rs`:** drop the `"{file}:"` prefix → the collision test reddens; make `declaration` fall back to the default → the `Unreadable`-vs-`Absent` test reddens; skip the `unrepresentable` push → the loss test reddens. **Each with the `mktemp`+`cmp` guard and the `failures:`-block read.**
- [ ] **Step 6: fmt, clippy, commit.**

---

### Task 3: The sidecar, in **both** spellings

**Files:** create `src/sidecar.rs`, `tests/sidecar.rs`.

**Interfaces:** `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum SidecarSpelling { Jinja, Json }`; `pub fn sidecar_template(filename: &str, bytes: &[u8]) -> Option<(SidecarSpelling, String)>`.

⚠️ **This is what PR #14 corrected.** `.jinja` holds **raw Jinja**; `.json` holds **a JSON object with one `chat_template` key**. A reader knowing only `.jinja` reports *"no sidecar"* on gemma-3.

- [ ] **Step 1: Tests** — `the_jinja_spelling_is_raw_template_text_one_instance_gptoss` (⚠️ **and on that checkpoint the sidecar is the ONLY source: its `tokenizer_config.json` carries no `chat_template` at all, whole-file read, all nine top-level keys enumerated — so ignoring it yields NO template, not a stale one**); `the_json_spelling_unwraps_one_key_one_instance_gemma3`; a `.json` sidecar without the key is not a template; an unrelated filename is never a sidecar; and **§9 1.1 — a blank sidecar is UNDECLARED**, `trim()` not `is_empty()`, checked here as well as in `mlmf-meta` **because this path never reaches `mlmf-meta`'s classifier**.
- [ ] **Steps 2–4:** implement (match a **set** of known names — the spelling varies with the `transformers` version that saved the checkpoint, **so a third will appear**), run.
- [ ] **Step 5: Sabotage — handle only `.jinja`** → the gemma-3 test reddens. ⚠️ **This protects PR #14's whole finding.** Then **`trim()` → `is_empty()`** → the blank test reddens **at the `"   "` iteration** while `""` still passes, which is why a test covering only `""` would ship the defect.
- [ ] **Step 6: fmt, clippy, commit.**

---

### Task 4: `ShardIndex` — where each tensor lives

**Files:** create `src/shards.rs`, `tests/shards.rs`.

**This is the half spec line 90 uses to DEFINE the crate, and §12 step 5 calls *"Fuel's actual dependency."***

**Interfaces:**

```rust
pub struct ShardIndex { /* private */ }
impl ShardIndex {
    /// Parse `model.safetensors.index.json`.
    pub fn parse(bytes: &[u8]) -> Result<Self, HfError>;
    /// The file holding `tensor`, or `None` if the index does not name it.
    pub fn shard_of(&self, tensor: &str) -> Option<&str>;
    /// Every shard filename, sorted and deduplicated.
    pub fn shards(&self) -> Vec<&str>;
    /// Every tensor name the index maps, sorted.
    pub fn tensors(&self) -> Vec<&str>;
    /// `metadata.total_size` in BYTES, if declared.
    pub fn total_size(&self) -> Option<u64>;
    /// Members of `metadata` other than `total_size`, preserved verbatim
    /// as `MetaValue`s (§5 rule 1: unknown keys survive).
    pub fn metadata_extras(&self) -> &[(String, MetaValue)];
}
```

⚠️ **`total_size` is `u64` and that is measured, not defensive.** Mistral `14483464192`, Qwen `15231233024` — **both exceed 2³², so a `u32` wraps silently.**

- [ ] **Step 1: Write the failing test.**

```rust
// crates/mlmf-hf-layout/tests/shards.rs
use mlmf_hf_layout::shards::ShardIndex;

/// A three-shard index in the shape measured on
/// mistralai/Mistral-7B-Instruct-v0.2, 2026-09-05.
const MISTRAL_SHAPED: &str = r#"{
    "metadata": { "total_size": 14483464192 },
    "weight_map": {
        "lm_head.weight": "model-00003-of-00003.safetensors",
        "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
        "model.layers.10.mlp.gate_proj.weight": "model-00001-of-00003.safetensors",
        "model.layers.10.self_attn.k_proj.weight": "model-00001-of-00003.safetensors",
        "model.layers.10.input_layernorm.weight": "model-00002-of-00003.safetensors",
        "model.layers.10.post_attention_layernorm.weight": "model-00002-of-00003.safetensors",
        "model.norm.weight": "model-00003-of-00003.safetensors"
    }
}"#;

#[test]
fn one_layer_may_span_two_shards() {
    // MEASURED on Mistral-7B-Instruct-v0.2, layer 10: gate_proj and
    // k_proj are in shard 1 while input_layernorm and
    // post_attention_layernorm are in shard 2.
    //
    // Any implementation that assumes a layer lives in one file, or
    // derives a shard from a layer index, is wrong on the first real
    // multi-shard checkpoint. The map is per-TENSOR and only per-tensor.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).expect("parses");
    assert_eq!(ix.shard_of("model.layers.10.mlp.gate_proj.weight"),
               Some("model-00001-of-00003.safetensors"));
    assert_eq!(ix.shard_of("model.layers.10.input_layernorm.weight"),
               Some("model-00002-of-00003.safetensors"));
}

#[test]
fn map_order_carries_no_locality() {
    // lm_head is in the LAST shard and embed_tokens in the FIRST, while
    // lm_head sorts first lexically. Order is lexical, not positional.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.shard_of("lm_head.weight"), Some("model-00003-of-00003.safetensors"));
    assert_eq!(ix.shard_of("model.embed_tokens.weight"), Some("model-00001-of-00003.safetensors"));
}

#[test]
fn total_size_exceeds_u32_and_must_not_wrap() {
    // Mistral 14483464192, Qwen 15231233024 -- both above 2^32. A u32
    // would wrap silently, which is why this is asserted rather than
    // assumed.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.total_size(), Some(14_483_464_192));
    assert!(ix.total_size().unwrap() > u64::from(u32::MAX));
}

#[test]
fn shards_are_deduplicated_and_tensors_are_not() {
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.shards().len(), 3, "seven tensors across three files");
    assert_eq!(ix.tensors().len(), 7);
}

#[test]
fn an_unknown_metadata_member_survives_verbatim() {
    // §5 rule 1. "Exactly two top-level keys" is TWO INSTANCES, not a
    // closed set, and the safetensors convention documents `metadata` as
    // an open object -- so an unrecognised member is preserved, never
    // dropped.
    let ix = ShardIndex::parse(
        br#"{"metadata":{"total_size":8,"format":"pt"},"weight_map":{"a":"s.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(ix.total_size(), Some(8));
    assert_eq!(ix.metadata_extras().len(), 1, "`format` survives");
}

#[test]
fn a_missing_tensor_is_none_rather_than_a_guess() {
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.shard_of("model.layers.99.nope"), None);
}

#[test]
fn indentation_is_not_part_of_the_contract() {
    // MEASURED: Mistral indents 4 spaces, Qwen indents 2. A hand-written
    // fixture would have encoded one of them.
    let two = r#"{
  "metadata": {
    "total_size": 8
  },
  "weight_map": {
    "a": "s.safetensors"
  }
}"#;
    assert_eq!(ShardIndex::parse(two.as_bytes()).unwrap().shard_of("a"),
               Some("s.safetensors"));
}

#[test]
fn a_non_string_weight_map_value_is_an_error_not_a_silent_skip() {
    assert!(ShardIndex::parse(br#"{"weight_map":{"a":42}}"#).is_err());
}
```

⚠️ **No test enumerates expected tensor names.** The set is architecture-dependent — Qwen carries `self_attn.{k,q,v}_proj.bias` entries Mistral lacks entirely — so a fixture from one model cannot validate a name list.

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement.** `weight_map` is flat `{string: string}`; a non-string value is an error naming the tensor. `metadata` is parsed permissively: `total_size` via `as_u64`, everything else through `value::meta_value` into `metadata_extras`.
- [ ] **Step 4: Run.** Expected: **8 passed**.
- [ ] **Step 5: Sabotage:** make `shard_of` derive a shard from the layer index → `one_layer_may_span_two_shards` reddens; narrow `total_size` to `u32` (`as_u64().and_then(|v| u32::try_from(v).ok()).map(u64::from)`) → the 2³² test reddens; `filter_map` the weight-map values → `a_non_string_weight_map_value_is_an_error_not_a_silent_skip` reddens.
- [ ] **Step 6: fmt, clippy, commit.**

---

### Task 5: `mlmf-meta` gains `BOS_TOKEN_TEXT` / `EOS_TOKEN_TEXT` and the measured HF rows

**Files:** modify `crates/mlmf-meta/src/vocab.rs`, `tests/vocab.rs`.

**New canonical keys:** `keys::BOS_TOKEN_TEXT = "tokenizer.bos_token_text"`, `keys::EOS_TOKEN_TEXT = "tokenizer.eos_token_text"`.

**New rows, every one measured 2026-09-05:**

```
BOS_TOKEN_ID    HuggingFace  "config.json:bos_token_id"
EOS_TOKEN_ID    HuggingFace  "config.json:eos_token_id"
BOS_TOKEN_TEXT  HuggingFace  "tokenizer_config.json:bos_token"
EOS_TOKEN_TEXT  HuggingFace  "tokenizer_config.json:eos_token"
ADD_BOS_TOKEN   HuggingFace  "tokenizer_config.json:add_bos_token"
ADD_EOS_TOKEN   HuggingFace  "tokenizer_config.json:add_eos_token"
```

⚠️ **`tokenizer_config.json:bos_token_id` must NEVER become a row.** It does not exist — the id is in `config.json`. An earlier draft of plan 7 invented it, and `vocab.rs`'s own doc comment already names this hazard.

⚠️ **The `*_TEXT` keys get an HF row and NO GGUF row**, because HF declares the token **text** while GGUF declares only an **id** resolved through its token table. **That direction is new; asymmetry itself is not** — six of seven canonical keys already have a GGUF row and no HF row, and `tests/vocab.rs`'s existing control covers presence asymmetry.

- [ ] **Step 1: Tests.** `spelling(BOS_TOKEN_TEXT, HuggingFace)` resolves; `spelling(BOS_TOKEN_TEXT, Gguf)` is `None`. ⚠️ **The existing "no rows at all" control is about `Safetensors`, which has ZERO rows. GGUF has SEVEN.** So the GGUF `None` needs its own control: assert GGUF's row count is **non-zero** while the specific lookup is `None` — otherwise the assertion is satisfiable by a format having no rows at all, which is not the situation being tested.
- [ ] **Steps 2–4:** add the two consts and six rows; the existing `the_table_is_bidirectional_for_every_row` covers them automatically.
- [ ] **Step 5: Sabotage — add a GGUF row for `BOS_TOKEN_TEXT`** → the asymmetry test reddens.
- [ ] **Step 6: fmt, clippy, commit.**

---

### Task 6: `SpecialTokens::extract` actually reads the text ⚠️ **THE B1 FIX**

**Files:** modify `crates/mlmf-meta/src/tokens.rs`, `tests/tokens.rs`.

⚠️ **This task exists because Task 5 alone does nothing.** Measured on the real crate with the six rows applied and `extract` untouched:

    bos = Some(SpecialToken { id: 1, text: None })     on BOTH local checkpoints

**`extract` resolves text only through `keys::TOKENS` and never looks up `BOS_TOKEN_TEXT`. Adding a row to a table does not make the extractor read it.**

**The change:** `extract`'s `token` closure resolves text by **two paths, in order**, and records which was used.

```rust
/// How a token's text was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextSource {
    /// Declared directly, as HF's `tokenizer_config.json` does.
    Declared,
    /// Resolved by indexing the id into the token table, as GGUF does.
    Indexed,
}
```

`SpecialToken` gains `pub text_source: Option<TextSource>` — `Some` exactly when `text` is `Some`.

⚠️ **`SpecialToken` derives `PartialEq, Eq` and is not `#[non_exhaustive]`, so adding a field is a breaking change for anyone constructing it literally. `mlmf-meta` is at `0.4.0` and unpublished; the only constructor is `extract`. Note it in the PR body rather than discovering it downstream.**

- [ ] **Step 1: Write the failing test.**

```rust
#[test]
fn text_comes_from_a_direct_declaration_when_the_format_has_one() {
    // The B1 REGRESSION TEST. Before this, extract resolved text ONLY
    // through keys::TOKENS, so an HF source -- which has no token table
    // and declares the text directly -- returned text: None on every real
    // checkpoint while every test passed.
    let f = Fake(vec![(
        "tokenizer_config.json:bos_token".into(),
        MetaValue::String("<s>".into()),
    )]);
    let t = SpecialTokens::extract(&f, Format::HuggingFace);
    let bos = t.bos.expect("declared by text, with no id anywhere");
    assert_eq!(bos.text.as_deref(), Some("<s>"));
    assert_eq!(bos.text_source, Some(TextSource::Declared));
}

#[test]
fn gguf_still_resolves_text_by_indexing_the_id() {
    let f = Fake(vec![
        ("tokenizer.ggml.tokens".into(), vocab_of(&["<unk>", "<s>"])),
        ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(1)),
    ]);
    let t = SpecialTokens::extract(&f, Format::Gguf);
    let bos = t.bos.unwrap();
    assert_eq!(bos.text.as_deref(), Some("<s>"));
    assert_eq!(bos.text_source, Some(TextSource::Indexed));
}
```

⚠️ **A token declared by TEXT may have no id at all**, so `SpecialToken.id` becomes `Option<u64>` — an HF checkpoint declaring only `tokenizer_config.json:bos_token` has a text and no id, and inventing `0` would be a supplied default §5 rule 2 forbids. **Update `an_id_past_the_vocabulary_keeps_the_id_and_reports_no_text` and the corpus differential accordingly.**

- [ ] **Step 2: Run → FAIL** (`text_source` does not exist; `text` is `None`).
- [ ] **Step 3: Implement** the two-path resolution.
- [ ] **Step 4: Run.**
- [ ] **Step 5: Sabotage — delete the `Declared` path from `extract`** (not from `TABLE`) → `text_comes_from_a_direct_declaration_when_the_format_has_one` reddens. ⚠️ **This is the sabotage the first draft got wrong: it must perturb the CODE THAT RUNS.**
- [ ] **Step 6: fmt, clippy, commit.**

---

### Task 7: The corpus differential, and branch review

**Files:** create `crates/mlmf-conformance/tests/hf_corpus.rs`; modify that crate's `Cargo.toml` and `tests/direct-deps.allow` (add `mlmf-hf-layout`, **sorted between `mlmf-gguf` and `mlmf-meta`**).

⚠️ **LEAVE THAT ALLOW-LIST'S HEADER ALONE.** `main` already replaced its counting line with a description, and says why: *"A count in a comment has to be EDITED to stay true, and the edit is the step that gets skipped."* **The first draft of this plan told the implementer to update a tally that no longer exists.**

**Scan root: `MLMF_HF_CHECKPOINTS`, default `C:/Models`. State it in the file header and beside every number.**

- [ ] **Assertions, existential floors rather than equalities:**
  - every directory with `config.json` **and** `tokenizer_config.json` yields a `SpecialTokens` whose `bos.text` is `Some` **with `text_source == Declared`** — ⚠️ **this is the assertion that would have caught B1 on real data**;
  - the id declared in `config.json` and the text declared in `tokenizer_config.json` **agree** with `tokenizer.json`'s `added_tokens[id]`, **and a disagreement fails printing BOTH**, because a disagreement is a finding about the checkpoint, not a reader bug;
  - ⚠️ **the `added_tokens` cross-check reads `tokenizer.json` through `HfLayout`, which is exactly what B2 broke** — it returned `Array(len 0)` silently. Assert the array's length is **non-zero** first, as a positive control on the layout itself;
  - at least one checkpoint declares `chat_template` as a string.
- [ ] ⚠️ **A scan for files literally named `tokenizer.json` finds TWO under `C:/Models`, not ten.** The other eight `added_tokens` files are `hf-tokenizers/<name>.json`. **Say which population the test walks.**
- [ ] **Skips:** `MLMF_CORPUS_REQUIRED` spelled `is_ok_and(|v| v != "0" && !v.is_empty())` as `mlmf-gguf/tests/corpus.rs:42` spells it; the stated reason must be true; assert **zero notice tokens** on the happy path by grepping, not by inferring "none skipped" from `ok`.
- [ ] **Sabotage:** regress `extract`'s `Declared` path → a **named** corpus test reddens rather than the suite skipping.

**Review:**
- [ ] `cargo fmt --all -- --check`; `bash scripts/local-gates.sh; rc=$?` — **gate on `$rc`.**
- [ ] `cargo tree -p mlmf-hf-layout --edges normal --depth 1` → `mlmf-core` and `serde_json`, nothing else from this workspace.
- [ ] **No `serde_json` in any `pub` signature**, with a positive control showing the pattern matches a real `pub` item: `grep -rnE "pub (fn|struct|enum)[^\n]*serde_json" crates/mlmf-hf-layout/src/` → nothing; `grep -rnE "^pub (fn|struct|enum)" crates/mlmf-hf-layout/src/` → several.
- [ ] **No `Path`/`PathBuf`/`fs::` in `src/`**, same control shape.
- [ ] **PR body:** the scan root behind every number; the **five** population-of-one shapes; that §9 1.4 is implemented **as corrected by PR #14**; that `SpecialToken` gained fields and `id` became `Option`; and that the shard half was verified against **two fetched instances and zero local ones**.

---

## Self-review

**Spec coverage.** Line 90 (where each tensor lives) → Task 4. Line 157 (HF JSON into one `MetadataSource`) → Tasks 1–3. §5 rules 1–3 → Tasks 1, 2, 4. §9 1.1 → Task 3. §9 1.4 as corrected → Tasks 1, 3, 5, 6. §9 3.1 → nothing touches tensor values. §12 step 5 → **both halves; `mlmf-safetensors`, the other named crate, already exists and this plan adds nothing to it.**

**Gaps left deliberately, each named:**
- **`tokenizer.json`'s `added_tokens` is not a first-class resolution path.** Task 6 resolves text by direct declaration or by the format's own token table; `added_tokens` is used **only** as the corpus differential's cross-check. Promoting it needs a rule for which source wins on disagreement, and **there is no measured disagreement to design against.**
- **Five shapes have a population of one**, and one half — the shard index — has **zero local instances**. Handled, labelled in test names, **not claimed to be conventions.**
- **`metadata`'s membership is open.** Two instances showed `total_size` alone; `metadata_extras` exists so a third does not lose data.
- **No test enumerates tensor names** — the set is architecture-dependent.
