# `mlmf-hf-layout` part A — the shard index (§12 step 5, Fuel's half)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse `model.safetensors.index.json` and answer **where does each tensor live** — the question spec line 90 uses to define this crate, and the one §12 step 5 calls *"Fuel's actual dependency."*

**Architecture:** Format axis — bytes in, structure out, **no I/O, never enumerates a directory** (spec line 90). The caller reads the file; this parses it.

**Tech Stack:** Rust **edition 2024** (workspace, inherited), `mlmf-core` + `serde_json`, following `mlmf-safetensors`' precedent.

**Spec:** `docs/superpowers/specs/2026-08-14-backend-agnostic-mlmf-design.md` — **line 90** (this crate's definition, in §3.2), §5 rules 1–3, §9 3.1, §12 step 5.

---

## ⚠️ Why this is half a plan, and where the other half went

**A single plan covering both halves of this crate was audited twice.** Round 1: six blocking. Round 2: six blocking, **four of them created by round 1's fixes**, and the worst was the round-1 defect recurring one layer down — a function with tests and three sabotages that **no production path called**, so every sabotage reddened a unit test of dead code.

**The two halves share almost nothing** — measured: one error type and one helper function, both removable. **All of round 2's entanglement is on the metadata side**, where a fix to where one function is called cascades into four tasks across two crates.

**So this plan is the shard index alone.** It is independently useful, it is the half a consumer is waiting on, and **its tests were built and run during the round-2 audit — all eight of the version audited passed, and this plan carries twelve, the four extra closing defects that audit found in the shard half itself.**

⚠️ **The metadata half is NOT yet written.** `2026-09-05-mlmf-hf-layout-metadata-DEFERRED.md` is the old two-half plan **renamed and not rewritten**, kept only as the record of the two audits; it carries a SUPERSEDED header saying so and warning that its Task 4 builds the same two files this plan builds, with an interface that does not compile. **Part B will be written fresh when it is written — that is a statement of intent, not of fact.**

**Carried from those two rounds, and from plan 7's seven:**

| Hazard | Guard |
|---|---|
| A feature with tests but **no caller** — twice now | ⚠️ **Task 2 is a reachability GATE with its own `the_gate_can_fail`.** Sabotage the code that RUNS |
| `git diff --stat` is **silent on an untracked file** | `cmp` against an `mktemp` copy |
| A build break exits non-zero and changes the file, exactly like a working sabotage | Read the **`failures:` block**, never the exit code |
| A `///` on a `pub mod` merges with the module's `//!` and resolves in the **parent's** scope | No outer docs on module declarations |
| **Intra-doc links to modules that do not exist yet** fail `cargo doc -D warnings` at the commit that introduces them | ⚠️ Task 0's `lib.rs` has **no links** |
| `local-gates.sh` prints a trailer after its success line | Gate on **`exit "$failed"`**, never `tail -1` |
| `git pull … 2>/dev/null` **hid a failure** and produced a 3-commit-stale base | Never discard stderr from a command whose success you depend on |

---

## Global Constraints

- **Axis `format`.** ⚠️ **No `Path`, `PathBuf` or `std::fs` in `src/`.** Spec line 90: *"It never enumerates a directory."*
- **`[dev-dependencies]` is refused for every gated member** (`deps.rs`).
- **`serde_json` never appears in a `pub` signature** — `mlmf-safetensors/src/header.rs:52` states the rule and its reason.
- **§5 rule 1** — unknown keys survive; dropping them is *"the worst kind of lossy — invisible."* **§5 rule 3** — lossiness **per key**.
- **§9 3.1** — this reports tensor **locations**. It never reads a tensor **value**.
- **AD-2**, with the guard shape above.
- **Gate every commit on the exit code:**

```bash
bash scripts/local-gates.sh; rc=$?
if [ "$rc" -ne 0 ]; then echo "$rc gate(s) FAILED -- do not commit"; fi
```

- ⚠️ **`Cargo.lock` on `main` does not list `mlmf-meta`.** The first `cargo` command dirties the tree. **Commit the lockfile update with Task 0** rather than letting it sit — a tree dirty for no semantic reason is what silently failed a `git pull` last round.

---

## Measured facts — two instances, **zero on this machine**

⚠️ **`find C:/Models ~/.cache/huggingface -name '*.index.json'` → 0**, positive control passing (`-name 'model.safetensors'` → 3). **Both local checkpoints are single-shard, so every local test here is a fixture.** These two were fetched from the Hub, and **re-fetched and re-verified independently during the round-2 audit** — because a shape derived from a fixture one wrote is not evidence.

| | `mistralai/Mistral-7B-Instruct-v0.2` | `Qwen/Qwen2.5-7B-Instruct` |
|---|---|---|
| size | 25,125 B | 27,752 B |
| shards / tensors | 3 / 291 | 4 / 339 |
| top-level keys | `metadata`, `weight_map` — **that order, nothing else** | same |
| `metadata` members | `total_size` only | `total_size` only |
| `total_size` | `14483464192` | `15231233024` |
| `weight_map` | flat `{tensor: filename}`, **all values strings** | same |
| indentation | **4 spaces** | **2 spaces** |

⚠️ **BOTH `total_size` values exceed 2³². A `u32` cast wraps silently. `u64`.**

⚠️ **A SINGLE LAYER SPLITS ACROSS SHARDS** — Mistral layer 10, measured twice:

    model.layers.10.mlp.gate_proj.weight            -> model-00001-of-00003
    model.layers.10.self_attn.k_proj.weight         -> model-00001-of-00003
    model.layers.10.input_layernorm.weight          -> model-00002-of-00003
    model.layers.10.post_attention_layernorm.weight -> model-00002-of-00003

**Any implementation assuming a layer lives in one file, or deriving a shard from a layer index, is wrong on the first real multi-shard checkpoint.** The map is **per-tensor and only per-tensor**. `lm_head.weight` is in the *last* shard, `model.embed_tokens.weight` in the *first*.

⚠️ **The tensor-name SET is architecture-dependent** — Qwen carries `self_attn.{k,q,v}_proj.bias` entries Mistral lacks entirely. **No test may enumerate expected tensor names.**

⚠️ **"Exactly two top-level keys" is TWO INSTANCES, not a closed set**, and the safetensors convention documents `metadata` as an open object. **Parse `metadata` permissively.** Do not encode indentation: the two files differ, which is what a hand-written fixture would have hidden.

⚠️ **Object key ORDER is unobservable here.** With `serde_json` at `default-features = false` (no `preserve_order`), objects land in a `BTreeMap`. Both real files happen to be lexically sorted, **but no test in this crate can check that**, so none claims to.

---

## File structure

```
crates/mlmf-hf-layout/
  Cargo.toml · tests/{axis,allowed-std.list,direct-deps.allow}
  src/lib.rs             module registration ONLY -- no `///`, no intra-doc links
  src/shards.rs          ShardIndex, ShardError                        Task 1
  tests/shards.rs        the 12 tests                                  Task 1
  tests/reachability.rs  the gate + the_gate_can_fail                  Task 2
```

⚠️ **THREE places name this crate's file set** — this block, Task 0's `tests/{…}` list, and each task's **Files:** line. **A split already desynchronised two of them once.** When you add a file, change all three.

---

### Task 0: Crate skeleton, gated from its first commit

- [ ] **Step 1: Baseline.** `cargo test -p mlmf-core --test ci_coverage --test workspace --test purity --test deps` → **20 tests, 0 failures**. ⚠️ `ci_coverage` reports **tests** (6) not members (7); any equality is coincidence. **This command dirties `Cargo.lock`** — expected, see Step 5.

- [ ] **Step 2: Create the crate.**

```toml
[package]
    description          = "HuggingFace shard index: where each tensor lives. Bytes to structure; no I/O."
    edition.workspace    = true
    license.workspace    = true
    name                 = "mlmf-hf-layout"
    repository.workspace = true
    version.workspace    = true

# NO [features]. mlmf-core has `default = []` and no `std` feature, so
# `std = ["mlmf-core/std"]` fails cargo at RESOLUTION for EVERY member --
# taking the whole workspace down, not just this crate.

[dependencies]
    mlmf-core  = { path = "../mlmf-core", version = "0.4.0" }
    serde_json = { version = "1.0", default-features = false, features = ["std"] }
```

```rust
// src/lib.rs
//! HuggingFace checkpoint layout.
//!
//! Answers one question from bytes the caller supplies: where does each
//! tensor live? Finding and reading the file belongs to `mlmf-source-file`;
//! spec line 90 is explicit that this crate never enumerates a directory.
#![forbid(unsafe_code)]
#![warn(missing_docs)]
```

⚠️ **NO `pub mod shards;` YET. Task 1 adds it, in Task 1.** Declaring a module before its file exists is `error[E0583]: file not found for module`, and it is not contained: **Step 4 below wires `cargo test -p mlmf-hf-layout` into `ci.yml`, and `local-gates.sh` runs every `run:` line it finds there**, so Step 5's "gate on the exit code" becomes unsatisfiable and the branch's first commit is red.

⚠️ **MEASURED DURING IMPLEMENTATION, after THREE audits missed it** — because every audit built the *finished* crate and none built this intermediate state. **An audit that only ever constructs the end state cannot see a defect that exists between two commits.**

⚠️ **No `[` `]` links in that doc comment.** An intra-doc link to a module fails `cargo doc -D warnings` — a CI step, run by `local-gates.sh` — at the commit that introduces it. ⚠️ **And no `///` above `pub mod shards;`**: an outer doc merges with the module's own `//!` and the merged text resolves in **this** module's scope, breaking the module's own links.

`tests/axis` → `format`. `tests/direct-deps.allow` → header + `mlmf-core`, `serde_json` **sorted** (positional `assert_eq!`). `tests/allowed-std.list` → **must exist**; `purity.rs::allowed_std` panics with `must declare its C3 allow-list` otherwise, and `deps.rs::allow_list` with `must declare its C2 allow-list`. Start empty; add only what the gate demands.

Root `Cargo.toml` `default-members`: **`crates/mlmf-hf-layout` sorts after `crates/mlmf-gguf`, before `crates/mlmf-meta`.**

- [ ] **Step 3: `ci_coverage` must FAIL naming `mlmf-hf-layout`.** Distinguish: the intended red · `failed to select a version … does not have that feature` (a `[features]` block crept in) · `must declare its C3/C2 allow-list` (a `tests/` file is missing) · **everything passes → stop and report**, because the gate cannot see new members.

- [ ] **Step 4: Wire CI — four steps**, copying `mlmf-conformance`'s shape exactly, with `RUSTDOCFLAGS: -D warnings` in `env:` on the doc step. **Four, not six**: the second-configuration test arms only on a non-empty `default`.

- [ ] **Step 5: `cargo fmt --all`, gate on the exit code, commit — including `Cargo.lock`.**

---

### Task 1: `ShardIndex`

**Files:** create `src/shards.rs`, `tests/shards.rs`; **modify `tests/allowed-std.list`.**

⚠️ **The allow-list is not optional here and Task 0 cannot predict it.** `impl std::error::Error for ShardError` names `std::error`, and with an empty list `purity.rs` fails: *"`mlmf-hf-layout`: …/src/shards.rs: path names `std::error`, which is not on the permitted-std allow-list (C3)"*. **Add `error` — and nothing else until a gate demands it.**

**Interfaces produced — full signatures, so nothing is guessed:**

```rust
/// A parsed `model.safetensors.index.json`.
///
/// ⚠️ **`PartialEq` but NOT `Eq`.** `metadata_extras` holds
/// [`mlmf_core::MetaValue`], which has `F32`/`F64` variants and therefore
/// derives only `PartialEq` (`mlmf-core/src/meta.rs:43`). Deriving `Eq`
/// here does not compile — and hand-writing `impl Eq for ShardIndex {}` to
/// get past that ships a type asserting total equality that is **not
/// reflexive**, because `MetaValue::F64(NAN) != MetaValue::F64(NAN)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardIndex { /* private */ }

/// Why an index could not be read. Owns its message; no foreign type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardError { /* private */ }
impl core::fmt::Display for ShardError {}
impl std::error::Error for ShardError {}

impl ShardIndex {
    /// Parse `model.safetensors.index.json`.
    pub fn parse(bytes: &[u8]) -> Result<Self, ShardError>;
    /// The file holding `tensor`, or `None` if the index does not name it.
    pub fn shard_of(&self, tensor: &str) -> Option<&str>;
    /// Every shard filename, sorted and deduplicated.
    pub fn shards(&self) -> Vec<&str>;
    /// Every tensor name, sorted.
    pub fn tensors(&self) -> Vec<&str>;
    /// `metadata.total_size` in BYTES, if declared **and readable as a
    /// `u64`**.
    ///
    /// ⚠️ **A `None` here is ambiguous on its own, and that is why
    /// `metadata_unrepresentable` covers this key too.** Six inputs
    /// produce `None` — absent, a string, a float, a negative, a
    /// non-object `metadata`, an absent `metadata` — and the plan's own
    /// argument for `a_missing_weight_map_is_an_error_not_an_empty_index`
    /// applies verbatim: a declared-but-unreadable value must not be
    /// indistinguishable from an undeclared one. **A `total_size` that was
    /// declared and could not be read appears in
    /// `metadata_unrepresentable`**, per §5 rule 3.
    pub fn total_size(&self) -> Option<u64>;
    /// `metadata` members other than `total_size`, preserved verbatim,
    /// sorted by key.
    pub fn metadata_extras(&self) -> &[(String, MetaValue)];
    /// `metadata` members with no `MetaValue` representation — an object,
    /// a null, or an array containing either. Sorted. §5 rule 3: the loss
    /// is named per key rather than left silent.
    pub fn metadata_unrepresentable(&self) -> &[String];
}
```

⚠️ **`ShardError` is this module's own type, not shared.** The two-half plan gave one `HfError` to two modules and it did not compile — **private fields are module-scoped, not crate-scoped**, so the other module could not construct it.

⚠️ **`parse` takes bytes and no filename**, so `ShardError` names the *problem*, never a file the function was not given.

- [ ] **Step 1: Write the failing test.**

```rust
// crates/mlmf-hf-layout/tests/shards.rs
//! Fixtures shaped from two real indices. ⚠️ THERE IS NO
//! `model.safetensors.index.json` ON THIS MACHINE -- both local
//! checkpoints are single-shard -- so every fixture here is a shape
//! FETCHED and verified from the Hub twice, not one I invented.
//! Population: TWO, measured 2026-09-05.

use mlmf_hf_layout::shards::ShardIndex;

/// Shaped from mistralai/Mistral-7B-Instruct-v0.2 (3 shards, 291 tensors).
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
fn one_layer_may_span_two_shards_two_instances_mistral_qwen() {
    // MEASURED on Mistral-7B-Instruct-v0.2 layer 10, twice: gate_proj and
    // k_proj in shard 1, input_layernorm and post_attention_layernorm in
    // shard 2.
    //
    // Any implementation that assumes a layer lives in one file, or
    // derives a shard from a layer index, is wrong on the first real
    // multi-shard checkpoint. The map is per-TENSOR and only per-tensor.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).expect("parses");
    assert_eq!(
        ix.shard_of("model.layers.10.mlp.gate_proj.weight"),
        Some("model-00001-of-00003.safetensors")
    );
    assert_eq!(
        ix.shard_of("model.layers.10.input_layernorm.weight"),
        Some("model-00002-of-00003.safetensors")
    );
}

#[test]
fn first_and_last_tensors_do_not_follow_map_position() {
    // lm_head is in the LAST shard while sorting first lexically;
    // embed_tokens is in the FIRST. Position carries no locality.
    //
    // NOTE ON WHAT THIS CANNOT SHOW: with serde_json at
    // default-features = false there is no `preserve_order`, so objects
    // land in a BTreeMap and the FILE's own key order is unobservable to
    // any test in this crate. This asserts the mapping, not the ordering.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.shard_of("lm_head.weight"), Some("model-00003-of-00003.safetensors"));
    assert_eq!(
        ix.shard_of("model.embed_tokens.weight"),
        Some("model-00001-of-00003.safetensors")
    );
}

#[test]
fn total_size_exceeds_u32_and_must_not_be_narrowed() {
    // Mistral 14483464192, Qwen 15231233024 -- BOTH above 2^32. A u32
    // cast wraps silently, which is why this is measured rather than
    // assumed defensive.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.total_size(), Some(14_483_464_192));
    assert!(ix.total_size().unwrap() > u64::from(u32::MAX));
}

#[test]
fn shards_are_sorted_and_deduplicated() {
    // Both halves are falsifiable: seven tensors map to three distinct
    // files, and the order is asserted rather than the count alone.
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(
        ix.shards(),
        vec![
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]
    );
}

#[test]
fn tensors_are_sorted() {
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    let got = ix.tensors();
    let mut want = got.clone();
    want.sort_unstable();
    assert_eq!(got, want, "tensors() promises sorted order");
    assert_eq!(got.len(), 7);
}

#[test]
fn an_unknown_metadata_member_survives_with_its_value() {
    // §5 rule 1. "Exactly two top-level keys" is TWO INSTANCES, not a
    // closed set, and the safetensors convention documents `metadata` as
    // an open object.
    //
    // ⚠️ The VALUE is asserted, not just the length: a length-only
    // assertion cannot see an implementation that stores a placeholder,
    // which is this repo's "assert position, not presence" defect.
    let ix = ShardIndex::parse(
        br#"{"metadata":{"total_size":8,"format":"pt"},"weight_map":{"a":"s.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(ix.total_size(), Some(8));
    assert_eq!(
        ix.metadata_extras(),
        &[("format".to_string(), mlmf_core::MetaValue::String("pt".into()))]
    );
}

#[test]
fn an_unrepresentable_metadata_member_is_named_not_dropped() {
    // §5 rule 3. MetaValue has no object variant, so a nested object
    // cannot be carried -- but vanishing is the invisible loss rule 1
    // forbids.
    let ix = ShardIndex::parse(
        br#"{"metadata":{"total_size":8,"nested":{"a":1}},"weight_map":{"a":"s.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(ix.metadata_extras(), &[]);
    assert_eq!(ix.metadata_unrepresentable(), &["nested".to_string()]);
}

#[test]
fn a_missing_tensor_is_none_rather_than_a_guess() {
    let ix = ShardIndex::parse(MISTRAL_SHAPED.as_bytes()).unwrap();
    assert_eq!(ix.shard_of("model.layers.99.nope"), None);
}

#[test]
fn indentation_is_not_part_of_the_contract() {
    // MEASURED: Mistral indents 4 spaces, Qwen 2. A hand-written fixture
    // would have encoded one of them.
    let two = "{\n  \"metadata\": {\n    \"total_size\": 8\n  },\n  \
               \"weight_map\": {\n    \"a\": \"s.safetensors\"\n  }\n}";
    assert_eq!(
        ShardIndex::parse(two.as_bytes()).unwrap().shard_of("a"),
        Some("s.safetensors")
    );
}

#[test]
fn a_non_string_weight_map_value_is_an_error_naming_the_tensor() {
    let e = ShardIndex::parse(br#"{"weight_map":{"a":42}}"#).expect_err("not a filename");
    assert!(format!("{e}").contains('a'), "the error must name the tensor: {e}");
}

#[test]
fn a_missing_weight_map_is_an_error_not_an_empty_index() {
    // ⚠️ An empty Ok is indistinguishable from a healthy index whose
    // tensor you did not ask for: shard_of() returns None either way, and
    // None is documented as "the index does not name it".
    assert!(ShardIndex::parse(br#"{"metadata":{"total_size":8}}"#).is_err());
}

#[test]
fn a_non_object_top_level_is_an_error() {
    assert!(ShardIndex::parse(b"42").is_err());
    assert!(ShardIndex::parse(b"[1,2]").is_err());
    assert!(ShardIndex::parse(b"{ not json").is_err());
}
```

⚠️ **No test enumerates expected tensor names.** The set is architecture-dependent; a fixture from one model cannot validate a name list.

**Add to `lib.rs`:** `pub mod shards;` — now that `src/shards.rs` exists.

- [ ] **Step 2: Run.** `cargo test -p mlmf-hf-layout --test shards` → **FAIL**, unresolved import.

- [ ] **Step 3: Implement.** `weight_map` is required and must be an object of strings; a non-string value is a `ShardError` naming the tensor. `metadata` is optional and parsed permissively: `total_size` via `as_u64`; every other member converted to `MetaValue` if it can be (string, bool, number, or an array of those **all-or-nothing** — a partially converted array reports a length the file never declared) and otherwise recorded in `metadata_unrepresentable`. Both vectors sorted before return.

- [ ] **Step 4: Run.** Expected: **12 passed**.

- [ ] **Step 5: Sabotage (AD-2), four times.** Each mutates `shards.rs` — **the code that runs.**

⚠️ **Sabotage (a) is a HAND EDIT, not a `perl`.** The first draft used a regex that prepended a discarded call and left the original expression intact — **a semantic no-op that `cmp` reported as applied and that left all 12 tests green.** There is no reliable one-line regex for "make `shard_of` wrong", so do it by hand and let the third guard state catch you if you get it wrong.

```bash
# (a) Make shard_of ignore the map -- the defect the real files refute.
#     HAND EDIT: change shard_of's body to
#         self.map.first().map(|(_, f)| f.as_str())
#     i.e. return the FIRST shard for every tensor, which is what
#     "derive the shard from the layer index" degenerates to.
BAK=$(mktemp); cp crates/mlmf-hf-layout/src/shards.rs "$BAK"
${EDITOR:-code} crates/mlmf-hf-layout/src/shards.rs   # make the edit above
rc=0; cmp "$BAK" crates/mlmf-hf-layout/src/shards.rs || rc=$?
case $rc in 0) echo '!! MUTATION DID NOT APPLY -- fix the pattern, do not proceed' ;;
             1) ;; *) echo '!! GUARD BROKEN -- no backup; re-run the cp' ;; esac
out=$(cargo test -p mlmf-hf-layout --test shards 2>&1); printf '%s\n' "$out" | tail -14
# ⚠️ THE THIRD STATE. Applied-and-nothing-reddened is a NO-OP MUTATION, and
# without this line it is indistinguishable from a working sabotage: `cmp`
# says applied, the guard above stays silent, and the suite prints ok.
printf '%s\n' "$out" | grep -q '^test result: FAILED' \
  || echo '!! APPLIED BUT NOTHING REDDENED -- the mutation is a no-op, not a passing sabotage'
cp "$BAK" crates/mlmf-hf-layout/src/shards.rs && rm -f "$BAK"
```

Expected: **FAIL on THREE tests** — `one_layer_may_span_two_shards_two_instances_mistral_qwen`, `first_and_last_tensors_do_not_follow_map_position`, and `a_missing_tensor_is_none_rather_than_a_guess`. Measured. ⚠️ **Three, not one:** returning the first shard for everything breaks every mapping claim, and a run reddening only one means the edit was narrower than intended.

**(b) Narrow `total_size` with a WRAPPING cast.** Replace the `as_u64` result with `(v as u32) as u64`.

⚠️ **NOT `try_from(v).ok()`.** That fails **loudly**, yields `None`, and reddens the test down a path the real defect never takes — **a sabotage that substitutes a loud failure for a silent one does not exercise the hazard**, and a test that survives it says nothing about the hazard. The measured values wrap to plausible numbers:

    14483464192 as u32 as u64  ->  1598562304    a believable byte count
    15231233024 as u32 as u64  ->  2346331136    a believable byte count

**Expect `total_size_exceeds_u32_and_must_not_be_narrowed` to fail with `left: Some(1598562304)`, not with `None`.** ⚠️ **If it fails with `None`, the wrong sabotage was applied — the equality assertion is what discriminates, and a test asserting only `is_some()` would have survived this.**

**(c) `filter_map` the metadata members** instead of recording the unrepresentable ones → `an_unrepresentable_metadata_member_is_named_not_dropped` reddens.

**(d) Return `Ok` with an empty index when `weight_map` is absent** → `a_missing_weight_map_is_an_error_not_an_empty_index` reddens.

- [ ] **Step 6: `cargo fmt --all`; `cargo clippy -p mlmf-hf-layout --all-targets -- -D warnings`; gate on the exit code; commit.**

---

### Task 2: Review, and the reachability check

⚠️ **This task exists because two audit rounds found the same defect: a feature with tests, docs and sabotages that no production path called.** Clippy caught it the second time (`function is never used`), and **the obvious repair — `#[allow(dead_code)]` — silences the only instrument that noticed.**

- [ ] **Step 1: A reachability GATE, as a test, with its own can-fail companion.**

⚠️ **Not a shell command run once.** A control run by hand proves the check worked **that day**, and the failure this guards against is an instrument that **silently stops discriminating**. `mlmf-core` already has the pattern in three places — `deps.rs:343`, `purity.rs:553`, `skip_notice.rs:144` all pair a gate with `the_gate_can_fail`, which exercises the **matcher** against synthetic inputs without editing any crate. Follow it.

```rust
// crates/mlmf-hf-layout/tests/reachability.rs
//! Every public function must be named by an integration test.
//!
//! This exists because TWO audit rounds of this crate's plan found the
//! same defect: a function with tests, docs and sabotages that **no
//! production path called**. Clippy caught the second one
//! (`function is never used`) — and the obvious repair,
//! `#[allow(dead_code)]`, silences the only instrument that noticed.
//!
//! **What this does NOT do**, said plainly so nobody reads more into a
//! pass than it earns: it does not check that a function is called by
//! anything a USER would run, only that this crate's own integration
//! tests name it. Integration tests can reach only the public API, which
//! is what makes that a real constraint and not a tautology — but the
//! strongest available claim here is still weaker than a real consumer.

use std::fs;

fn public_fns(src: &str) -> Vec<String> { /* lines matching `pub fn NAME` */ }

#[test]
fn every_public_fn_is_named_by_an_integration_test() {
    let mut sources = String::new();
    for e in fs::read_dir("tests").expect("tests dir") {
        let p = e.expect("entry").path();
        if p.extension().is_some_and(|x| x == "rs") {
            sources.push_str(&fs::read_to_string(&p).expect("readable"));
        }
    }

    let mut declared = Vec::new();
    for e in fs::read_dir("src").expect("src dir") {
        let p = e.expect("entry").path();
        if p.extension().is_some_and(|x| x == "rs") {
            declared.extend(public_fns(&fs::read_to_string(&p).expect("readable")));
        }
    }

    // ⚠️ A gate that finds NOTHING and a gate that CANNOT find anything
    // are the same output -- and so are a gate that finds SEVEN and one
    // that finds FIVE OF SEVEN. The interface block specifies exactly
    // seven public fns; an equality is what makes a silently-narrowed
    // matcher visible, and a floor is not.
    assert_eq!(
        declared.len(),
        7,
        "expected the seven public fns the plan specifies, found {:?} -- \
         a matcher that silently drops some is indistinguishable from a \
         clean tree",
        declared
    );

    // ⚠️ WORD-BOUNDED, not `contains`. A substring match passes any name
    // that happens to occur in unrelated test text: `metadata` appears in
    // every JSON fixture here and `shard` appears in `shard_of`,
    // `shards`, `shards.rs` and the string "3 shards". Both are plausible
    // next-API names, and both would be silently reported as reached.
    let named = |f: &str| {
        sources.split(|c: char| !c.is_alphanumeric() && c != '_').any(|w| w == f)
    };
    let unreached: Vec<_> = declared.iter().filter(|f| !named(f)).collect();
    assert!(
        unreached.is_empty(),
        "public functions no integration test names: {unreached:?}\n\n\
         A function with tests but no caller passed every other gate in \
         this repo, twice. If one of these is genuinely internal, make it \
         `pub(crate)`; do not add `#[allow(dead_code)]`, which silences \
         the instrument rather than the defect."
    );
}

#[test]
fn the_gate_can_fail() {
    // ⚠️ THE CONTROL, AND IT RUNS EVERY TIME. A reachability check that
    // has never fired is not known to work, and a control run once by
    // hand proves only that it worked that day.
    //
    // Exercises the MATCHER on synthetic sources, so it needs no
    // throwaway function in the real crate and cannot be forgotten.
    // ⚠️ `pub(crate) fn` is in this fixture DELIBERATELY. The failure
    // message above tells the implementer to demote a genuinely-internal
    // fn to `pub(crate)`; a loose matcher (`contains("pub") &&
    // contains("fn ")`) passes this control and then FLAGS the demoted
    // fn, leaving `#[allow(dead_code)]` -- which the same message forbids
    // -- as the only escape. The remedy the gate prescribes must be one
    // the gate accepts.
    let src = "pub fn reached() {}\n\
               pub fn unreached() {}\n\
               pub(crate) fn internal() {}\n\
               fn private() {}\n";
    let declared = public_fns(src);
    assert_eq!(
        declared,
        vec!["reached".to_string(), "unreached".to_string()],
        "only `pub fn` is the subject: `pub(crate)` and private are not"
    );

    // A SUBSTRING match would call `reach` reached; a word match does not.
    let tests = "assert!(reached()); let reachability = 1; let unreachable_x = 2;";
    let named = |f: &str| {
        tests.split(|c: char| !c.is_alphanumeric() && c != '_').any(|w| w == f)
    };
    let unreached: Vec<_> = declared.iter().filter(|f| !named(f)).collect();
    assert_eq!(
        unreached,
        vec![&"unreached".to_string()],
        "the matcher must NAME an unreached fn, and must not be fooled by \
         `unreachable_x` or `reachability` sharing a prefix"
    );
}
```

⚠️ **`the_gate_can_fail` is what makes the first test evidence rather than decoration**, and because it is a test it runs in CI on every commit — which is the point. **The two defects this catches passed every other gate in the repo.**

- [ ] **Step 2: `cargo clippy -p mlmf-hf-layout --all-targets -- -D warnings` with NO `#![allow(dead_code)]` anywhere.**

⚠️ **The `-p` is required and its absence is not a typo to shrug at.** Unscoped, the command reaches the legacy root package — **`default-members` includes `"."` deliberately** — which is **422 clippy errors on an untouched tree** and is excluded from CI on the record. An unscoped gate here is a definition-of-done that cannot be met before any work begins. **No command in `ci.yml` or `local-gates.sh` is unscoped; verified.**

```bash
grep -rn "allow(dead_code)" crates/mlmf-hf-layout/ && echo "!! dead_code silenced -- that is the defect, not the fix"
```

- [ ] **Step 3: Axis and dependency claims, each with a positive control.**

```bash
grep -rnE "\bPath\b|PathBuf|std::fs" crates/mlmf-hf-layout/src/ ; echo "exit=$?"   # expect 1, nothing
grep -rnE "\bstr\b" crates/mlmf-hf-layout/src/ | head -2                            # control: greps DO match here
# ⚠️ ANY `pub` item on a line with serde_json -- fn, struct, enum, TYPE ALIAS,
# or a public FIELD. The narrow `pub (fn|struct|enum)` form misses a public
# field (`pub raw: serde_json::Value`), which is exactly what
# mlmf-safetensors had to solve with `pub(crate) entries: serde_json::Map`,
# and misses any signature rustfmt wrapped across lines.
grep -rn "serde_json" crates/mlmf-hf-layout/src/ | grep -vE "^\s*[0-9]+:\s*//" \
  | grep -E "\bpub\b" ; echo "exit=$?"                              # expect 1, nothing
# ...and the multi-line case the line-oriented grep cannot see:
grep -rnA3 -E "^\s*pub fn " crates/mlmf-hf-layout/src/ | grep serde_json ; echo "exit=$?"
grep -rnE "^pub (fn|struct|enum)" crates/mlmf-hf-layout/src/ | head -3   # control: greps match here
cargo tree -p mlmf-hf-layout --edges normal --depth 1
```

- [ ] **Step 4: Full gates on the exit code**, then open the PR.

**PR body must state:** that the shard index was verified against **two fetched instances and ZERO local ones**, so every test is a fixture; the layer-split finding and why it forbids deriving a shard from a layer index; that `total_size` is `u64` **because both measured values exceed 2³²**; that `metadata`'s membership is open and two instances is not a closed set; that no test enumerates tensor names because the set is architecture-dependent; and that **part B — `HfLayout`, the sidecars, and `mlmf-meta`'s `BOS_TOKEN_TEXT` — is deliberately not in this PR.**

---

## Self-review

**Spec coverage.** Line 90's *"reports the checkpoint's structure and where each tensor lives"* → Task 1. §5 rule 1 → `metadata_extras`. §5 rule 3 → `metadata_unrepresentable`. §9 3.1 → locations only, never values. §12 step 5 → **this is the half of it Fuel needs; the metadata half is part B and `mlmf-safetensors`, the other crate the step names, already exists.**

**Gaps left deliberately, each named:**
- **Zero local instances.** Every test is a fixture shaped from two fetched files. Stated in the test module's header, not only here.
- **Object key order is unobservable** without `preserve_order`, so no test claims to check it, and `first_and_last_tensors_do_not_follow_map_position` says so in its own comment.
- **`metadata`'s membership is open.** Two instances showed `total_size` alone.
- **No name enumeration** — the tensor set is architecture-dependent.
- **Nothing consumes `ShardIndex` yet.** ⚠️ Fuel is the intended consumer and is not wired up here, so **Task 2's reachability check is against the crate's own tests, which is the strongest available claim and is weaker than a real consumer.** Named because the two prior rounds died on exactly this distinction.
