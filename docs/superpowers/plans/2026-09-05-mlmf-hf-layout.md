# `mlmf-hf-layout` — the HuggingFace sidecar layer (§12 step 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the bytes of a HuggingFace checkpoint's JSON sidecars into a `MetadataSource`, so `mlmf-meta` extracts from an HF checkpoint through the code it already has.

**Architecture:** Format axis — **bytes in, structure out, no I/O.** The caller reads files with `mlmf-source-file`; this crate parses them. Keys are spelled `<filename>:<key>` (`tokenizer_config.json:chat_template`), which is **exactly what spec §5's table already uses**, so `mlmf-meta`'s existing `Format::HuggingFace` rows work unchanged.

```
mlmf-source-file (I/O)   ->  bytes
mlmf-hf-layout (format)  ->  MetadataSource keyed "<file>:<key>"
mlmf-meta (structure)    ->  TemplateSet / SpecialTokens
```

**Tech Stack:** Rust **edition 2024** (workspace, inherited), `mlmf-core` + `serde_json`. Precedent: `mlmf-safetensors` already depends on `serde_json = { version = "1.0", default-features = false, features = ["std"] }` and keeps `serde_json::Map` **`pub(crate)`** because it is *"a foreign type"*. Follow both.

**Spec:** `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md` — §5 (vocabulary), §9 §1 (esp. **1.4 as corrected in PR #14**, and 1.3), §9 3.1, §12 step 5.

> **This plan is written against §9 1.4 AS CORRECTED (PR #14, merged `60d8112`).** The pre-correction clause named one sidecar spelling and one token shape; both are narrower than the checkpoints. **Read 1.4 on `main`, not from memory.**

---

## Global Constraints

- **Axis: `format`.** No I/O. `purity.rs` scans `src/` only, so corpus tests that read files live in `mlmf-conformance`.
- **⚠️ `[dev-dependencies]` is refused for every gated member.** `deps.rs::no_table_other_than_plain_dependencies_may_declare_an_edge`. Any test needing two crates in one binary goes in `mlmf-conformance`.
- **`serde_json` types never appear in the public API.** `mlmf-safetensors/src/header.rs:52` states the rule and the reason; a `serde_json::Value` in a signature makes this crate's consumers depend on a version of a foreign crate.
- **§5 rule 1 — enumerate what you understand; preserve what you don't.** Unknown keys survive. Every drop lands in a named field.
- **§5 rule 2 — absent means `None`, never a default.**
- **§5 rule 3 — lossiness per key, not per crate.**
- **⚠️ §9 3.1 — dtype coercion is consumer policy, never a default.** This crate reads metadata and **never touches tensor values.** *(Named because `src/formats/onnx_import.rs:317,328` violates it today with `half::f16::from_bits(..).to_f32()` — scheduled for rewrite, and not a pattern to copy.)*
- **AD-2 — no test trusted until sabotage has made it go red.** Verify each mutation applied by `cmp` against an `mktemp` copy, **never `git diff --stat`** (silent on untracked files), and read the `failures:` block, **not the exit code** — a build break also exits non-zero.
- **Corpus rule — assert the corpus, and name where it came from**, including the **scan root**. A number without its subject becomes a claim about the world when repeated.
- **⚠️ Label unobserved shapes IN THE TEST NAME, not only in a comment.** A test passing on a shape with no local instance tests this code's structure, not the shape. **Three shapes here have exactly one instance each** — see the population table.
- **Skip-condition rule.** `mlmf_core::NOTICE_TOKEN`; `MLMF_CORPUS_REQUIRED` spelled `is_ok_and(|v| v != "0" && !v.is_empty())` as `mlmf-gguf/tests/corpus.rs:42` spells it; and the **stated skip reason must itself be true**.
- **`rustfmt` before every commit**, and **gate the commit on `local-gates.sh`'s SUMMARY line**, not its tail.

---

## Measured facts — six checkpoints, 2026-09-05

Local: `C:/Models/TinyLlama-1.1B-Chat-v1.0`, `C:/Models/SmolLM2-360M-Instruct`. Remote metadata fetched from the Hub 2026-09-03 and **re-verified from the bytes** on this side: `NousResearch/Hermes-3-Llama-3.1-8B`, `NousResearch/Llama-2-7b-chat-hf`, `openai/gpt-oss-20b`, `google/gemma-3-4b-it`.

| checkpoint | `bos_token` | `add_bos_token` | `chat_template` |
|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | string `<s>` | absent | string, 410 B |
| SmolLM2-360M-Instruct | string `<\|im_start\|>` | absent | string, 368 B |
| Hermes-3-Llama-3.1-8B | string `<\|begin_of_text\|>` | absent | **array** — `default` + `tool_use` |
| Llama-2-7b-chat-hf | **`AddedToken` object**, text in `content` | **`true`** | absent |
| openai/gpt-oss-20b | string `<\|startoftext\|>` | absent | absent — **`.jinja` sidecar only** |
| google/gemma-3-4b-it | *(not measured)* | *(not measured)* | **`chat_template.json` sidecar** |

**Both local checkpoints, verified:** `config.json` `bos_token_id=1`, `eos_token_id=2`; `tokenizer.json`'s `added_tokens` resolves both to the same strings `tokenizer_config.json` declares directly.

⚠️ **POPULATION, and it is thin where it matters most:**

    added_tokens id -> text          10 files   (2 checkpoints + 8 bare tokenizer.json)
    the TWO-PATH cross-check          2 files   <- both paths present only here
    chat_template as an ARRAY         1 file    (Hermes-3)
    AddedToken OBJECT token           1 file    (Llama-2-7b-chat-hf)
    .jinja sidecar                    1 file    (gpt-oss-20b)
    .json sidecar                     1 file    (gemma-3-4b-it)
    add_bos_token PRESENT             1 file    (Llama-2-7b-chat-hf)

**Compare GGUF: 29 files.** ⚠️ **Four shapes have a population of one.** One instance proves a shape **exists**; it says nothing about frequency, and it cannot distinguish "this is the convention" from "this checkpoint is unusual." **Every test touching a population-of-one shape says so in its name.**

**The `add_bos_token` history is the caution.** It was absent in four consecutive checkpoints and I nearly recorded "GGUF-only". **One presence ended it, and it was the first key of the first file fetched to look for it.** *Spend the search on the disconfirming case.*

---

## The two list forms are opposite structures sharing a word

    HF     "chat_template": [ {"name":"default","template":"..."},
                              {"name":"tool_use","template":"..."} ]
           an array of OBJECTS CARRYING BODIES; the default is a NAMED entry

    GGUF   tokenizer.chat_templates = ["tool_use","rag"]        NAMES only
           tokenizer.chat_template.<name> = body                bodies in siblings
           tokenizer.chat_template       = the default          UNNAMED, and NOT in the array

⚠️ **`mlmf-meta` handles the GGUF form. That is not the HF form and must not be reused.** Asserted on reasoning in plan 7; measured since.

---

## File structure

```
crates/mlmf-hf-layout/
  Cargo.toml                    mlmf-core + serde_json
  tests/axis                    "format"
  tests/allowed-std.list        the std submodules actually named
  tests/direct-deps.allow       mlmf-core, serde_json   (sorted)
  src/lib.rs                    module registration ONLY -- no `///` on `pub mod`
  src/value.rs                  Task 1: JSON -> MetaValue, and the AddedToken shape
  src/source.rs                 Task 2: HfLayout, a MetadataSource keyed "<file>:<key>"
  src/sidecar.rs                Task 3: both sidecar spellings
  tests/value.rs  tests/source.rs  tests/sidecar.rs

crates/mlmf-meta/
  src/vocab.rs                  Task 4: BOS_TOKEN_TEXT + the HF rows, now measured
  src/tokens.rs                 Task 4: report BOTH resolution paths

crates/mlmf-conformance/
  tests/hf_corpus.rs            Task 5: the differential over real checkpoints
```

⚠️ **`src/lib.rs` carries NO `///` doc on the `pub mod` lines.** An outer doc merges with the module's own `//!` and the merged text resolves in the **parent's** scope, breaking that module's intra-doc links and failing `cargo doc -D warnings` — a CI step. Measured in plan 7, cost one red commit.

---

### Task 0: Crate skeleton, gated from its first commit

**Files:** create `Cargo.toml`, `src/lib.rs`, `tests/{axis,allowed-std.list,direct-deps.allow}`; modify root `Cargo.toml` (**`default-members`**, sorted — `hf-layout` sorts after `gguf` and before `meta`), `.github/workflows/ci.yml`.

- [ ] **Step 1: Baseline.** `cargo test -p mlmf-core --test ci_coverage --test workspace --test purity --test deps` — all green. ⚠️ **`ci_coverage` prints no member count**; it reports *tests*, and the equality with the crate count is a coincidence.

- [ ] **Step 2: Create the crate.**

```toml
[package]
    description          = "HuggingFace checkpoint sidecars as an mlmf-core MetadataSource. Bytes to structure; no I/O."
    edition.workspace    = true
    license.workspace    = true
    name                 = "mlmf-hf-layout"
    repository.workspace = true
    version.workspace    = true

# NO [features]. `mlmf-core` has `default = []` and no `std` feature, so
# `std = ["mlmf-core/std"]` fails cargo at RESOLUTION for every member --
# not just this crate. Measured in plan 7.

[dependencies]
    mlmf-core  = { path = "../mlmf-core", version = "0.4.0" }
    # Same form as mlmf-safetensors, which parses the safetensors JSON header.
    # `default-features = false` keeps the dependency minimal; `std` is what
    # serde_json needs to build. serde_json types stay pub(crate) -- see
    # mlmf-safetensors/src/header.rs:52 for the rule and its reason.
    serde_json = { version = "1.0", default-features = false, features = ["std"] }
```

```rust
// src/lib.rs
//! HuggingFace checkpoint sidecars, as an [`mlmf_core::MetadataSource`].
//!
//! Bytes in, structure out. Finding the files is [`mlmf-source-file`]'s job
//! and reading their values is [`mlmf-meta`]'s; this crate is the layer
//! between, and it spells its keys `<filename>:<key>` so that §5's own
//! vocabulary table applies unchanged.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
```

`tests/axis` → `format\n`. `tests/direct-deps.allow` → a header plus `mlmf-core` then `serde_json`, **sorted** (positional `assert_eq!`). `tests/allowed-std.list` → the std submodules `src/` actually names; **start empty and add only what the gate demands** — it panics if the file is absent, so it must exist either way.

- [ ] **Step 3: Run `ci_coverage` and watch it FAIL naming `mlmf-hf-layout`.** Four failure modes to tell apart: the intended red · a feature-resolution error (delete the `[features]` block) · a `must declare its C3/C2 allow-list` panic (the three files are missing) · **everything passes → stop and report**, because the gate cannot see new members.

- [ ] **Step 4: Wire CI — four steps**, copying `mlmf-conformance`'s shape: `cargo test -p mlmf-hf-layout`, the same `--no-default-features`, `cargo doc -p mlmf-hf-layout --no-deps` **with `RUSTDOCFLAGS: -D warnings`**, and `cargo clippy -p mlmf-hf-layout --all-targets -- -D warnings`. **Four, not six** — the second-configuration test arms only on a non-empty `default` feature list.

```bash
cargo fmt --all
SUMMARY=$(bash scripts/local-gates.sh 2>&1 | tail -1); echo "$SUMMARY"
case "$SUMMARY" in *"pass locally"*) echo GREEN;; *) echo "RED -- do not commit"; exit 1;; esac
```

- [ ] **Step 5: Commit.**

---

### Task 1: JSON → `MetaValue`, and the `AddedToken` shape

**Files:** create `src/value.rs`, `tests/value.rs`.

**Interfaces produced:**
- `pub fn meta_value(v: &serde_json::Value) -> Option<MetaValue>` — **`pub(crate)`**, not public: it names a foreign type.
- `pub struct TokenDecl { pub text: String, pub was_object: bool }`
- `pub fn token_decl(v: &serde_json::Value) -> Option<TokenDecl>`

- [ ] **Step 1: Write the failing test.**

```rust
// crates/mlmf-hf-layout/tests/value.rs
use mlmf_hf_layout::value::{token_decl, TokenDecl};

fn json(s: &str) -> serde_json::Value {
    serde_json::from_str(s).expect("test literal parses")
}

#[test]
fn a_string_token_is_the_text_itself() {
    // TinyLlama, SmolLM2, Hermes-3, gpt-oss-20b -- 4 of 6 checkpoints,
    // measured 2026-09-05.
    let d = token_decl(&json(r#""<s>""#)).expect("declared");
    assert_eq!(d.text, "<s>");
    assert!(!d.was_object);
}

#[test]
fn an_added_token_object_yields_its_content_ONE_INSTANCE_llama2() {
    // POPULATION OF ONE: NousResearch/Llama-2-7b-chat-hf is the only
    // checkpoint measured carrying this shape. The name says so because a
    // pass here proves the shape is HANDLED, not that it is common.
    //
    // Spec §9 1.4 (corrected, PR #14): "a string, OR an AddedToken object
    // whose `content` key holds the text". A reader that only accepts a
    // string returns "not declared" for this checkpoint, which DOES
    // declare it -- a fact about the reader reported as a fact about the
    // file.
    let d = token_decl(&json(
        r#"{"__type":"AddedToken","content":"<s>","lstrip":false,
            "normalized":true,"rstrip":false,"single_word":false}"#,
    ))
    .expect("declared, one indirection deeper");
    assert_eq!(d.text, "<s>");
    assert!(d.was_object, "the caller may need to know which shape it was");
}

#[test]
fn an_object_without_content_is_not_a_token_declaration() {
    assert!(token_decl(&json(r#"{"__type":"AddedToken"}"#)).is_none());
    assert!(token_decl(&json(r#"{"content":42}"#)).is_none(), "content must be a string");
}

#[test]
fn a_number_or_null_is_not_a_token_declaration() {
    assert!(token_decl(&json("42")).is_none());
    assert!(token_decl(&json("null")).is_none());
}
```

- [ ] **Step 2: Run it. Expected: FAIL,** unresolved import.

- [ ] **Step 3: Implement.**

```rust
// crates/mlmf-hf-layout/src/value.rs
//! JSON values, converted without interpretation.

use mlmf_core::MetaValue;

/// A token declared in `tokenizer_config.json`, in either of its shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenDecl {
    /// The token text.
    pub text: String,
    /// Whether it arrived as an `AddedToken` object rather than a bare
    /// string. Reported rather than smoothed away: the two shapes are the
    /// same datum, and a consumer auditing a checkpoint may want to know
    /// which one the file used.
    pub was_object: bool,
}

/// Read `bos_token` / `eos_token` / `unk_token` in either shape.
///
/// Spec §9 1.4 as corrected: a **string**, or an **`AddedToken` object**
/// whose `content` holds the text. Returns `None` for anything else — a
/// number, a null, or an object with no string `content`.
#[must_use]
pub fn token_decl(v: &serde_json::Value) -> Option<TokenDecl> {
    if let Some(s) = v.as_str() {
        return Some(TokenDecl { text: s.to_string(), was_object: false });
    }
    let content = v.as_object()?.get("content")?.as_str()?;
    Some(TokenDecl { text: content.to_string(), was_object: true })
}

/// Convert a JSON scalar to a [`MetaValue`], **without parsing strings**.
///
/// §5: a format that did not declare a number did not declare a number.
/// A JSON string stays a `String` even if it looks numeric.
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
            MetaValue::Array(a.iter().filter_map(meta_value).collect())
        }
        // Null and Object have no MetaValue counterpart. Object is not a
        // loss in practice -- the two objects this layer cares about
        // (AddedToken, and a sidecar's wrapper) are read by name elsewhere.
        serde_json::Value::Null | serde_json::Value::Object(_) => return None,
    })
}
```

Add `pub mod value;` to `lib.rs` — **no `///` above it.**

- [ ] **Step 4: Run. Expected: 4 passed.**

- [ ] **Step 5: Sabotage (AD-2), twice.**

```bash
# (a) Accept only the string shape -- the exact §9 1.4 defect.
BAK=$(mktemp); cp crates/mlmf-hf-layout/src/value.rs "$BAK"
perl -0pi -e 's/let content = v\.as_object\(\)\?\.get\("content"\)\?\.as_str\(\)\?;/return None; #[allow(unreachable_code)] let content = "";/' crates/mlmf-hf-layout/src/value.rs
rc=0; cmp "$BAK" crates/mlmf-hf-layout/src/value.rs || rc=$?
case $rc in 0) echo '!! MUTATION DID NOT APPLY';; 1) ;; *) echo '!! GUARD BROKEN';; esac
cargo test -p mlmf-hf-layout --test value 2>&1 | tail -12 || true
cp "$BAK" crates/mlmf-hf-layout/src/value.rs && rm -f "$BAK"
```
Expected: **FAIL** on `an_added_token_object_yields_its_content_ONE_INSTANCE_llama2`.

```bash
# (b) Coerce a numeric `content` -- "never parse".
BAK=$(mktemp); cp crates/mlmf-hf-layout/src/value.rs "$BAK"
perl -0pi -e 's/\.get\("content"\)\?\.as_str\(\)\?/.get("content").map(|c| c.as_str().unwrap_or("42"))?/' crates/mlmf-hf-layout/src/value.rs
rc=0; cmp "$BAK" crates/mlmf-hf-layout/src/value.rs || rc=$?
case $rc in 0) echo '!! MUTATION DID NOT APPLY';; 1) ;; *) echo '!! GUARD BROKEN';; esac
cargo test -p mlmf-hf-layout --test value 2>&1 | tail -12 || true
cp "$BAK" crates/mlmf-hf-layout/src/value.rs && rm -f "$BAK"
```
Expected: **FAIL** on `an_object_without_content_is_not_a_token_declaration`.

⚠️ **Check each reddens by FAILING A NAMED TEST.** A build break also exits non-zero and also changes the file.

- [ ] **Step 6: Commit.**

---

### Task 2: `HfLayout` — a `MetadataSource` keyed `<filename>:<key>`

**Files:** create `src/source.rs`, `tests/source.rs`.

**Interfaces produced:**
- `pub struct HfLayout { /* private */ }`
- `pub fn HfLayout::builder() -> HfLayoutBuilder`; `HfLayoutBuilder::file(name: &str, bytes: &[u8]) -> Result<Self, HfError>`; `.build() -> HfLayout`
- `impl MetadataSource for HfLayout`
- `pub struct HfError`, with the offending filename

⚠️ **The builder takes BYTES, never a path.** This crate is on the format axis; `mlmf-source-file` finds and reads the files. **A `Path` in this signature is the axis violation C3 exists to catch.**

- [ ] **Step 1: Write the failing test.**

```rust
// crates/mlmf-hf-layout/tests/source.rs
use mlmf_core::MetadataSource;
use mlmf_hf_layout::source::HfLayout;

fn layout(files: &[(&str, &str)]) -> HfLayout {
    let mut b = HfLayout::builder();
    for (n, s) in files {
        b = b.file(n, s.as_bytes()).expect("test JSON parses");
    }
    b.build()
}

#[test]
fn keys_are_spelled_filename_colon_key_which_is_what_section_5_uses() {
    // Spec §5's table spells the HF side `tokenizer_config.json:chat_template`.
    // Producing exactly that is what lets mlmf-meta's existing vocabulary
    // rows work against this source with no change.
    let l = layout(&[("tokenizer_config.json", r#"{"chat_template":"{{ body }}"}"#)]);
    assert_eq!(
        l.get("tokenizer_config.json:chat_template")
            .and_then(mlmf_core::MetaValue::as_str)
            .map(String::as_str),
        Some("{{ body }}")
    );
    assert!(l.keys().contains(&"tokenizer_config.json:chat_template"));
}

#[test]
fn two_files_do_not_collide_even_on_the_same_key_name() {
    // Both local checkpoints declare bos-related keys in BOTH files, so the
    // qualification is load-bearing rather than cosmetic.
    let l = layout(&[
        ("config.json", r#"{"bos_token_id":1}"#),
        ("tokenizer_config.json", r#"{"bos_token":"<s>"}"#),
    ]);
    assert!(l.get("config.json:bos_token_id").is_some());
    assert!(l.get("tokenizer_config.json:bos_token").is_some());
    assert!(l.get("bos_token_id").is_none(), "unqualified keys must not resolve");
}

#[test]
fn every_declared_key_survives_even_unrecognised_ones() {
    // §5 rule 1: "Unknown keys survive verbatim. Dropping unrecognised keys
    // is the worst kind of lossy -- invisible."
    let l = layout(&[("config.json", r#"{"model_type":"llama","nonsense_key":7}"#)]);
    assert!(l.get("config.json:model_type").is_some());
    assert!(l.get("config.json:nonsense_key").is_some(), "unknown keys survive");
}

#[test]
fn a_nested_object_value_is_recorded_as_unreadable_not_dropped() {
    // MetaValue has no object variant, so a nested object cannot be
    // represented. §5 rule 3 says mark the loss per key rather than let the
    // key vanish.
    let l = layout(&[("config.json", r#"{"nested":{"a":1}}"#)]);
    assert!(l.get("config.json:nested").is_none(), "no MetaValue for an object");
    assert_eq!(l.unrepresentable(), &["config.json:nested".to_string()]);
}

#[test]
fn malformed_json_names_the_file_it_came_from() {
    let e = HfLayout::builder()
        .file("tokenizer_config.json", b"{ not json")
        .expect_err("malformed");
    assert!(
        format!("{e}").contains("tokenizer_config.json"),
        "the error must name the file: {e}"
    );
}

#[test]
fn index_complete_is_true_because_a_parsed_file_was_read_whole() {
    // Unlike a GGUF reader that may stop early, serde_json either parses
    // the whole document or fails. Saying so is what makes `None` mean
    // "not declared" rather than "not reached".
    assert!(layout(&[("config.json", r#"{"a":1}"#)]).index_complete());
}
```

- [ ] **Step 2: Run. Expected: FAIL.**

- [ ] **Step 3: Implement** `HfLayout` as a `Vec<(String, MetaValue)>` plus a `Vec<String>` of unrepresentable keys, built by iterating each file's top-level object and prefixing `"{file}:"`. `get` is a linear scan (a handful of files, tens of keys — do not reach for a map). `index_complete()` returns `true`, with the doc comment above as its justification. `HfError` carries the filename and the `serde_json` message as a **`String`**, not the foreign error type.

Add `pub mod source;` to `lib.rs`.

- [ ] **Step 4: Run. Expected: 6 passed.**

- [ ] **Step 5: Sabotage — drop the filename qualification**, expect `two_files_do_not_collide_even_on_the_same_key_name` to redden; and **drop the `unrepresentable` push**, expect `a_nested_object_value_is_recorded_as_unreadable_not_dropped` to redden.

- [ ] **Step 6: Commit.**

---

### Task 3: The sidecar, in **both** spellings

**Files:** create `src/sidecar.rs`, `tests/sidecar.rs`.

**Interfaces produced:** `pub enum SidecarSpelling { Jinja, Json }`, `pub fn sidecar_template(filename: &str, bytes: &[u8]) -> Option<(SidecarSpelling, String)>`.

⚠️ **This is the clause PR #14 corrected.** `chat_template.jinja` holds **raw Jinja**; `chat_template.json` holds a **JSON object with one `chat_template` key**. A reader that knows only `.jinja` reports *"no sidecar"* on gemma-3 — a fact about the reader delivered as a fact about the checkpoint.

- [ ] **Step 1: Write the failing test.**

```rust
// crates/mlmf-hf-layout/tests/sidecar.rs
use mlmf_hf_layout::sidecar::{sidecar_template, SidecarSpelling};

#[test]
fn the_jinja_spelling_is_raw_template_text_ONE_INSTANCE_gptoss() {
    // POPULATION OF ONE: openai/gpt-oss-20b. And on that checkpoint the
    // sidecar is the ONLY source -- its tokenizer_config.json carries no
    // chat_template at all (whole-file read, all nine top-level keys
    // enumerated), so ignoring the sidecar yields NO template rather than a
    // stale one.
    let (sp, body) = sidecar_template("chat_template.jinja", b"{{ bos_token }}hello")
        .expect("a .jinja sidecar is a template");
    assert_eq!(sp, SidecarSpelling::Jinja);
    assert_eq!(body, "{{ bos_token }}hello");
}

#[test]
fn the_json_spelling_unwraps_one_key_ONE_INSTANCE_gemma3() {
    // POPULATION OF ONE: google/gemma-3-4b-it. The shape §9 1.4 did NOT
    // name until PR #14; a reader looking only for `.jinja` finds nothing
    // here and reports "no sidecar".
    let (sp, body) = sidecar_template(
        "chat_template.json",
        br#"{"chat_template":"{{ bos_token }}\nhello"}"#,
    )
    .expect("a .json sidecar wraps the template");
    assert_eq!(sp, SidecarSpelling::Json);
    assert_eq!(body, "{{ bos_token }}\nhello");
}

#[test]
fn a_json_sidecar_without_the_key_is_not_a_template() {
    assert!(sidecar_template("chat_template.json", br#"{"something_else":1}"#).is_none());
}

#[test]
fn an_unrelated_filename_is_never_a_sidecar() {
    assert!(sidecar_template("tokenizer_config.json", br#"{"chat_template":"x"}"#).is_none());
    assert!(sidecar_template("README.md", b"{{ x }}").is_none());
}

#[test]
fn a_blank_sidecar_is_undeclared_per_clause_1_1() {
    // §9 1.1: a blank chat_template is UNDECLARED, not declared-empty.
    // `trim()`, not `is_empty()` -- checked here as well as in mlmf-meta,
    // because this path never passes through mlmf-meta's classifier.
    for blank in ["", "   ", "\n\t "] {
        assert!(
            sidecar_template("chat_template.jinja", blank.as_bytes()).is_none(),
            "{blank:?} must be UNDECLARED"
        );
    }
}
```

- [ ] **Step 2: Run. Expected: FAIL.**

- [ ] **Step 3: Implement.** Match on the exact filename; `.jinja` → UTF-8 decode; `.json` → parse and take a string `chat_template`; **both** then apply `trim().is_empty()` → `None`. Document that the spelling **varies with the `transformers` version that saved the checkpoint, so a third will appear** — and that this is why the function keys on a **set of known names** rather than a single constant.

Add `pub mod sidecar;` to `lib.rs`.

- [ ] **Step 4: Run. Expected: 5 passed.**

- [ ] **Step 5: Sabotage — handle only `.jinja`**, and confirm `the_json_spelling_unwraps_one_key_ONE_INSTANCE_gemma3` reddens. ⚠️ **This is the sabotage that protects PR #14's whole finding.** Then **regress `trim()` to `is_empty()`** and confirm the blank test reddens **at the `"   "` iteration** while `""` still passes — the reason a test covering only `""` would ship the defect.

- [ ] **Step 6: Commit.**

---

### Task 4: `mlmf-meta` gains `BOS_TOKEN_TEXT` and the measured HF rows

**Files:** modify `crates/mlmf-meta/src/vocab.rs`, `src/tokens.rs`, `tests/vocab.rs`, `tests/tokens.rs`.

⚠️ **This is the first time §5's "declared, bidirectional, per-format" is exercised by a datum ONE FORMAT HAS AND THE OTHER DOES NOT.** HF declares the token **text** directly; GGUF declares only an **id**, resolved through its token table. So `BOS_TOKEN_TEXT` gets an HF row and **no GGUF row** — and that asymmetry is the point, not an omission.

- [ ] **Step 1: Write the failing tests** — that `BOS_TOKEN_TEXT`/`EOS_TOKEN_TEXT` resolve for `Format::HuggingFace` and are `None` for `Format::Gguf`, **with the existing "no rows at all" control extended** so the `None` cannot be satisfied by GGUF having no rows; and that `SpecialTokens::extract` reports **both paths** when both are present, and flags a disagreement.

**New rows, every one measured (2026-09-05):**

```
BOS_TOKEN_ID    HuggingFace  "config.json:bos_token_id"
EOS_TOKEN_ID    HuggingFace  "config.json:eos_token_id"
BOS_TOKEN_TEXT  HuggingFace  "tokenizer_config.json:bos_token"
EOS_TOKEN_TEXT  HuggingFace  "tokenizer_config.json:eos_token"
ADD_BOS_TOKEN   HuggingFace  "tokenizer_config.json:add_bos_token"
ADD_EOS_TOKEN   HuggingFace  "tokenizer_config.json:add_eos_token"
```

⚠️ **`tokenizer_config.json:bos_token_id` is NOT a row and must never become one.** It does not exist; the id lives in `config.json`. An earlier draft of plan 7 invented it, and inventing a spelling in the table is worse than omitting one because the table is what a later reader trusts without checking.

- [ ] **Step 2–4:** implement, run, and confirm the extended control still distinguishes "this format has no row for this key" from "this format has no rows".

- [ ] **Step 5: Sabotage — make `BOS_TOKEN_TEXT` resolve for GGUF too**, and confirm the asymmetry test reddens.

- [ ] **Step 6: Commit.**

---

### Task 5: The corpus differential, in `mlmf-conformance`

**Files:** create `crates/mlmf-conformance/tests/hf_corpus.rs`; modify that crate's `Cargo.toml` and `direct-deps.allow` (add `mlmf-hf-layout`, **sorted between `mlmf-gguf` and `mlmf-meta`**, and **keep the header comment, updating the line that counts the entries**).

**Scan root: `MLMF_HF_CHECKPOINTS`, default `C:/Models`.** ⚠️ **State the root in the file header AND in any number this test reports.** A scan-root mismatch is how a 28-vs-30 disagreement happened with lightbulb; the number was right and its subject was missing.

- [ ] **Assertions, each an EXISTENTIAL floor rather than an equality**, so a larger checkpoint set does not fail them:
  - every checkpoint with both `config.json` and `tokenizer_config.json` resolves BOS by **both paths**, and they **agree** — ⚠️ **and a disagreement fails with both values, because a disagreement is a finding about the checkpoint, not a reader bug**;
  - at least one checkpoint declares `chat_template` as a **string**;
  - `index_complete()` is true for every layout built.
- [ ] **Skips:** `MLMF_CORPUS_REQUIRED` in this repo's spelling; the stated reason must be true; assert **zero notice tokens** on the happy path by grepping, rather than inferring "none skipped" from `ok`.
- [ ] **Sabotage:** regress `token_decl` to string-only and confirm a **named test** reddens rather than the suite skipping.

---

### Task 6: Branch review and PR

- [ ] `cargo fmt --all -- --check`; `local-gates.sh` **gated on its SUMMARY line**.
- [ ] **C3/C4 by reading, not assuming:** `cargo tree -p mlmf-hf-layout --edges normal --depth 1` shows `mlmf-core` and `serde_json` and nothing else from this workspace.
- [ ] **No `serde_json` in the public API:** `cargo doc` then grep the generated HTML — or, cheaper, `grep -rn "serde_json" crates/mlmf-hf-layout/src/ | grep "pub fn\|pub struct\|pub enum"` with a **positive control** showing the pattern can match a real `pub` item in those files.
- [ ] **No `Path`/`PathBuf`/`fs` anywhere in `src/`** — the axis claim, with a positive control.
- [ ] **PR body must state:** the scan root behind every number; the **population of one** for four shapes; that §9 1.4 is implemented **as corrected by PR #14**; and which shapes are handled-but-unobserved.

---

## Self-review

**Spec coverage.** §5 → Tasks 2 and 4 (the `<file>:<key>` spelling is §5's own; `BOS_TOKEN_TEXT` is the first asymmetric datum). §9 1.1 → Task 3's blank check, deliberately duplicated because the sidecar path never reaches `mlmf-meta`'s classifier. §9 1.4 as corrected → Tasks 1, 3, 4. §9 3.1 → nothing here touches tensor values. §12 step 5 → the whole plan.

**Gaps left deliberately, each named:**
- **`tokenizer.json`'s `added_tokens` is not consulted.** Path A resolves an id through it; this plan reports the id and the directly-declared text, and **the id→text resolution is Task 5's differential only**. Making it a first-class path needs a decision about which source wins on disagreement — and **there is no measured disagreement to design against.**
- **`mlmf-safetensors` is untouched.** §12 step 5 names it, but it already exists and this plan adds nothing to it; **the step is half done before it starts, and saying so is more useful than inventing work.**
- **Four shapes have a population of one.** Handled and labelled; **not claimed to be conventions.**
- **The `.jinja`/`.json` set will grow.** Keyed on a set of names, with the reason written beside it.
