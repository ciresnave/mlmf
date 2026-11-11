# `mlmf-hf-config`: a `config.json` reader that fills only verified defaults

Design note, ROADMAP S-14. No code yet; this note comes first (PM, 2026-11-10).

## 1. The ruling and what it changes

CireSnave, 2026-11-10, relayed verbatim by the PM:

> *"S-14: A.  Filling in verified defaults we can be sure are always correct is not outside what MLMF should do.  It is not interpreting meaning if the parser fills in known values.  Rebuild the parser."*

This is the permitted half of the §6 fence in `CLAUDE.md`: *"MLMF may supply a format's documented default. MLMF may never supply a model's value."* It also changes one sentence of existing code. `crates/mlmf-hf-layout/src/config.rs` says a caller "maps aliases and applies defaults on top". That generic `ConfigJson` stays exactly as it is: it reads the keys as the file spells them. The new crate sits above it and is the one place a default may be filled.

## 2. The two defaults in the gap plan are not universal

The gap plan (`docs/superpowers/specs/2026-10-03-fuel-gap-plan.md`, item 4) names `num_key_value_heads = num_attention_heads` and `head_dim = hidden_size / num_attention_heads` as the defaults. Read in the transformers config classes at tag `v4.57.0` (`src/transformers/models/<m>/configuration_<m>.py`), they are not the rule for every family. What the class does when `config.json` omits the key:

| `model_type` | absent `num_key_value_heads` | absent `head_dim` |
|---|---|---|
| llama | `num_attention_heads` (L195-196: `if ... is None: ... = num_attention_heads`; constructor default `None`, L168) | `hidden_size // num_attention_heads` (L209) |
| phi3 | `num_attention_heads` (L157-158; constructor default `None`, L131) | not read here |
| mistral | **8** (constructor default, L123) | stored as `None` (L124, L146); the docstring (L56) says `hidden_size // num_attention_heads` |
| mixtral | **8** (L135) | stored as `None` (L136); docstring says `hidden_size // num_attention_heads` |
| qwen2 | **32** (L158) | no `head_dim` argument |
| qwen3 | **32** (L162) | **128** (L163) |
| gemma | **16** (L117) | **256** (L118) |
| gemma2 | **4** (L126) | **256** (L127) |

So "absent means the number of attention heads" is true for llama and phi3 and false for six others, and "absent means hidden over heads" is false for qwen3, gemma and gemma2 (fixed constants, not derivable from the file). A universal rule would write a wrong value for those families, and a wrong value looks exactly like a declared one (`CLAUDE.md` §1). The PM agreed on 2026-11-10 that the examples were not his or CireSnave's and are wrong as universal rules.

## 3. The reading that keeps the ruling true

A default is filled only if it is an entry in a table keyed on the declared `model_type`, and every entry:

1. names the transformers config class and the source line it comes from (the table above is the shape);
2. names the transformers version it was read at, and a test pins the cited line: the test reads a vendored copy of the one line (or its text hash) and fails if the entry and the citation disagree;
3. is admitted only after its value is checked to be **the same in every tagged release in which the class exists**, because a `config.json` can be written by any release. That check is a step of the implementation, not done yet; an entry that changed between releases is not admitted, since it would not be always correct.

For a `model_type` not in the table, or a file with no `model_type`, an absent field stays absent (`Option<T>`, `CLAUDE.md` §2). Nothing is guessed from a neighbouring family. A derived default (`hidden / heads`) is admitted only for a family whose class defines it, and only when both inputs are declared; an undeclared input leaves it absent.

### 3.4 The constancy check, run 2026-11-10

The check in 3.3 was run over the `.0` tag of every transformers minor release from `v4.30.0` to `v4.57.0` (the constructor default and the `if num_key_value_heads is None` branch, read from each `configuration_<m>.py`). Result, with the releases in which the class existed:

| `model_type` | field | default | releases | constant |
|---|---|---|---|---|
| llama | `num_key_value_heads` | absent means `num_attention_heads` | 4.31 to 4.57 | yes |
| phi3 | `num_key_value_heads` | absent means `num_attention_heads` | 4.41 to 4.57 | yes |
| mistral | `num_key_value_heads` | 8 | 4.34 to 4.57 | yes |
| mixtral | `num_key_value_heads` | 8 | 4.36 to 4.57 | yes |
| qwen2 | `num_key_value_heads` | 32 | 4.37 to 4.57 | yes |
| qwen3 | `num_key_value_heads` | 32 | 4.51 to 4.57 | yes |
| qwen3 | `head_dim` | 128 | 4.51 to 4.57 | yes |
| gemma | `num_key_value_heads` | 16 | 4.38 to 4.57 | yes |
| gemma | `head_dim` | 256 | 4.38 to 4.57 | yes |
| gemma2 | `head_dim` | 256 | 4.42 to 4.57 | yes |
| gemma2 | `num_key_value_heads` | **16 in 4.42 to 4.46, 4 in 4.47 to 4.57** | 4.42 to 4.57 | **no: not admitted** |

So gemma2's `num_key_value_heads` fails the criterion: a `config.json` written by a release before 4.47 with the key omitted means 16, and one from 4.47 on means 4, so no single value is always correct. It stays absent. This is the criterion doing its job, and it is why the table is checked rather than assumed.

Limits of this check, stated so it is not read as more: it read the `.0` tag of each minor release, not every patch release; it covers only the eight families above; and it reads the config class, not the model code that consumes the field. `head_dim` for llama, mistral and mixtral is not admitted: the class stores `None` (the derivation `hidden_size // num_attention_heads` lives in the docstring or in model code, and llama's `head_dim` argument only exists from 4.45). Those stay absent until a derived default is argued separately.

## 4. Shape

A new format-axis crate, no I/O, bytes in:

```rust
pub struct HfConfig { /* declared fields, Option<T> throughout */ }
impl HfConfig { pub fn parse(bytes: &[u8]) -> Result<Self, HfConfigError>; }

pub enum Source { Declared, ClassDefault { class: &'static str, transformers: &'static str } }
pub struct Resolved<T> { pub value: T, pub source: Source }

impl HfConfig {
    pub fn num_key_value_heads(&self) -> Option<Resolved<u64>>;
    pub fn head_dim(&self) -> Option<Resolved<u64>>;
}
```

`Source` is the part that is more than the gap plan's sketch. The standing policy asks that a caller never need to know why a value is absent; it says nothing against telling a caller why a value is present, and without it a filled default is indistinguishable from a declared one, which is the defect class `CLAUDE.md` §1 describes. A consumer that wants only declared values filters on `Source::Declared`.

## 5. Open questions

- **Dependencies.** Parsing on top of `mlmf-hf-layout`'s `ConfigJson` avoids a second JSON parser but adds a crate-to-crate edge that the dependency gates (`crates/mlmf-core/tests/deps.rs`, the per-crate allow-list) must be told about; reading with `serde_json` directly duplicates the parser. Recommendation: depend on `mlmf-hf-layout`, PM to confirm.
- **Which families first.** The ten entries marked constant in 3.4 (llama, phi3, mistral, mixtral, qwen2, qwen3, gemma, gemma2 head_dim). Everything else, and gemma2 `num_key_value_heads`, stays absent.
- **Wrong or missing `model_type`.** A file whose declared `model_type` names a family that does not use these fields leaves them absent; nothing is inferred from tensor names (that is architecture detection, a different crate's job).

## 6. Sources

Version-pinned to transformers `v4.57.0`, read 2026-11-10 from `raw.githubusercontent.com/huggingface/transformers/v4.57.0/src/transformers/models/<m>/configuration_<m>.py`. Line numbers are in that tag; they are the citation the pinning test of section 3.2 is built on.
