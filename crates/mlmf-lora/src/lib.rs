// SPDX-License-Identifier: MIT OR Apache-2.0
//! LoRA adapter files, as the file declares them (ROADMAP S-65, step 2).
//!
//! Design: `docs/superpowers/specs/2026-11-11-lora-adapter-files-design.md`.
//! Measured facts this reads against:
//! `crates/mlmf-test-support/data/lora-measurements.tsv`.
//!
//! # What this does, and refuses to do
//!
//! - **Describe, never interpret.** [`AdapterConfig`] reports the fields
//!   `adapter_config.json` declares. An absent `r` is `None`, never the PEFT
//!   class default of 8 (CLAUDE.md section 2: that is a default of a Python
//!   class, not of the file format). No scale (`lora_alpha / r`) is computed.
//! - **A declared value of an unexpected shape is reported, not dropped.**
//!   `use_rslora` is the bool `True` in one measured file and the string
//!   `'True'` in another, so [`Flag`] says which was declared and a reader
//!   never converts one into the other. A key that is present but not
//!   readable as its expected type lands in [`AdapterConfig::unreadable`],
//!   so "absent" and "declared oddly" stay distinguishable.
//! - **Pairing is a finding.** [`pair_tensors`] matches `lora_A` with
//!   `lora_B` by name; an unmatched one is an [`Orphan`], and a tensor that is
//!   neither (`embed_tokens.base_layer.weight`) is kept in
//!   [`Pairing::others`]. Nothing is dropped and nothing is completed.
//! - **No I/O, no merge, no tensor bytes.** Values are never touched.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use mlmf_core::{Declaration, MetaValue, MetadataSource, TensorDescriptor};

/// A boolean-looking field exactly as declared.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Flag {
    /// Declared as a boolean.
    Bool(bool),
    /// Declared as a string (measured: `use_rslora` as `'True'`). Not parsed.
    Text(String),
}

/// `target_modules` / `exclude_modules`, in the shape the file used.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModuleSelector {
    /// A list of module names.
    List(Vec<String>),
    /// A single string (PEFT treats it as a regex; that is not applied here).
    Single(String),
}

/// What `adapter_config.json` declares about a LoRA adapter.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct AdapterConfig {
    /// `peft_type`.
    pub peft_type: Option<String>,
    /// `base_model_name_or_path`.
    pub base_model_name_or_path: Option<String>,
    /// `r`, the declared rank.
    pub r: Option<u64>,
    /// `lora_alpha`, as declared (integers widen losslessly to `f64`).
    pub lora_alpha: Option<f64>,
    /// `use_rslora`.
    pub use_rslora: Option<Flag>,
    /// `fan_in_fan_out`.
    pub fan_in_fan_out: Option<Flag>,
    /// `bias`.
    pub bias: Option<String>,
    /// `target_modules`.
    pub target_modules: Option<ModuleSelector>,
    /// `exclude_modules`.
    pub exclude_modules: Option<ModuleSelector>,
    /// Entries of `rank_pattern` (`rank_pattern.<module>` keys), sorted by module.
    pub rank_pattern: Vec<(String, u64)>,
    /// Entries of `alpha_pattern` (`alpha_pattern.<module>` keys), sorted by module.
    pub alpha_pattern: Vec<(String, f64)>,
    /// Keys present in the source but not readable as the type this struct
    /// expects: a JSON `null`, an empty object (which the source cannot
    /// enumerate), or a wrong type. Sorted.
    pub unreadable: Vec<String>,
}

impl AdapterConfig {
    /// Read the declared fields from any [`MetadataSource`].
    ///
    /// Total: never fails. A key that is absent is `None`; a key that is
    /// declared but not of the expected type is `None` AND listed in
    /// [`Self::unreadable`].
    #[must_use]
    pub fn from_source(src: &dyn MetadataSource) -> Self {
        let mut bad: Vec<String> = Vec::new();
        let mut cfg = Self {
            peft_type: typed(src, "peft_type", string, &mut bad),
            base_model_name_or_path: typed(src, "base_model_name_or_path", string, &mut bad),
            bias: typed(src, "bias", string, &mut bad),
            r: typed(src, "r", MetaValue::as_u64, &mut bad),
            lora_alpha: typed(src, "lora_alpha", number, &mut bad),
            use_rslora: typed(src, "use_rslora", flag, &mut bad),
            fan_in_fan_out: typed(src, "fan_in_fan_out", flag, &mut bad),
            target_modules: typed(src, "target_modules", selector, &mut bad),
            exclude_modules: typed(src, "exclude_modules", selector, &mut bad),
            ..Self::default()
        };
        for key in src.keys() {
            if let Some(m) = key.strip_prefix("rank_pattern.") {
                match src.get(key).and_then(MetaValue::as_u64) {
                    Some(v) => cfg.rank_pattern.push((unescape(m), v)),
                    None => bad.push(key.to_owned()),
                }
            } else if let Some(m) = key.strip_prefix("alpha_pattern.") {
                match src.get(key).and_then(number) {
                    Some(v) => cfg.alpha_pattern.push((unescape(m), v)),
                    None => bad.push(key.to_owned()),
                }
            }
        }
        // Declared but unrepresentable keys that `keys()` does not list.
        for key in [
            "rank_pattern",
            "alpha_pattern",
            "modules_to_save",
            "layers_to_transform",
        ] {
            if matches!(src.declaration(key), Declaration::Unreadable(_)) {
                bad.push(key.to_owned());
            }
        }
        cfg.rank_pattern.sort_by(|a, b| a.0.cmp(&b.0));
        cfg.alpha_pattern.sort_by(|a, b| a.0.cmp(&b.0));
        bad.sort();
        bad.dedup();
        cfg.unreadable = bad;
        cfg
    }
}

/// Undo the flattening escape of a dotted key segment: a literal `.` or `\`
/// in a JSON key is written `\.` or `\\` by the sources that flatten nested
/// objects (`mlmf-hf-layout`'s `ConfigJson`), and PEFT module names are
/// full of dots.
fn unescape(seg: &str) -> String {
    let mut out = String::with_capacity(seg.len());
    let mut it = seg.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            if let Some(n) = it.next() {
                out.push(n);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// A declared number as `f64`: a float as is, an integer only when `f64` holds
/// it exactly (up to 2^53), so a widening never changes the declared value.
fn number(v: &MetaValue) -> Option<f64> {
    const EXACT: u64 = 1 << 53;
    if let Some(f) = v.as_f64() {
        return Some(f);
    }
    if let Some(u) = v.as_u64() {
        return (u <= EXACT).then_some(u as f64);
    }
    v.as_i64()
        .filter(|i| i.unsigned_abs() <= EXACT)
        .map(|i| i as f64)
}

fn string(v: &MetaValue) -> Option<String> {
    v.as_str().cloned()
}

fn flag(v: &MetaValue) -> Option<Flag> {
    match v {
        MetaValue::Bool(b) => Some(Flag::Bool(*b)),
        MetaValue::String(s) => Some(Flag::Text(s.clone())),
        _ => None,
    }
}

fn selector(v: &MetaValue) -> Option<ModuleSelector> {
    match v {
        MetaValue::String(s) => Some(ModuleSelector::Single(s.clone())),
        MetaValue::Array(items) => items
            .iter()
            .map(string)
            .collect::<Option<Vec<_>>>()
            .map(ModuleSelector::List),
        _ => None,
    }
}

/// Read `key` through `f`; a declared value `f` rejects is recorded in `bad`.
fn typed<T>(
    src: &dyn MetadataSource,
    key: &str,
    f: impl Fn(&MetaValue) -> Option<T>,
    bad: &mut Vec<String>,
) -> Option<T> {
    match src.declaration(key) {
        Declaration::Absent => None,
        Declaration::Declared(v) => {
            let out = f(v);
            if out.is_none() {
                bad.push(key.to_owned());
            }
            out
        }
        _ => {
            bad.push(key.to_owned());
            None
        }
    }
}

/// Which half of a pair a tensor is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Half {
    /// `lora_A` / `lora_embedding_A`.
    A,
    /// `lora_B` / `lora_embedding_B`.
    B,
}

/// The module path in front of a PEFT LoRA suffix, and which half it was.
fn classify(name: &str) -> Option<(&str, Half)> {
    const SUFFIXES: [(&str, Half); 4] = [
        (".lora_A.weight", Half::A),
        (".lora_B.weight", Half::B),
        (".lora_embedding_A", Half::A),
        (".lora_embedding_B", Half::B),
    ];
    SUFFIXES
        .iter()
        .find_map(|(s, h)| name.strip_suffix(s).map(|m| (m, *h)))
}

/// A matched `A`/`B` pair for one module.
#[derive(Debug, Clone, PartialEq)]
pub struct Pair {
    /// The tensor name with the LoRA suffix removed, verbatim (prefix kept).
    pub module: String,
    /// The `A` tensor.
    pub a: TensorDescriptor,
    /// The `B` tensor.
    pub b: TensorDescriptor,
}

/// A half with no partner.
#[derive(Debug, Clone, PartialEq)]
pub struct Orphan {
    /// Which half is present.
    pub half: Half,
    /// The tensor that has no partner.
    pub tensor: TensorDescriptor,
}

/// The result of [`pair_tensors`]: every input tensor lands in exactly one field.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Pairing {
    /// Matched pairs, sorted by module.
    pub pairs: Vec<Pair>,
    /// Unmatched halves, sorted by tensor name.
    pub orphans: Vec<Orphan>,
    /// Tensors that are not LoRA halves (e.g. `lm_head.base_layer.weight`).
    pub others: Vec<TensorDescriptor>,
}

/// Pair `lora_A` with `lora_B` tensors by name.
///
/// Every input lands in exactly one of `pairs` (two tensors), `orphans` or
/// `others`. A module whose same half appears twice keeps the first as the
/// pair member and reports the repeat as an orphan.
#[must_use]
pub fn pair_tensors(tensors: &[TensorDescriptor]) -> Pairing {
    let mut out = Pairing::default();
    let mut a_side: Vec<(String, TensorDescriptor)> = Vec::new();
    let mut b_side: Vec<(String, TensorDescriptor)> = Vec::new();
    for t in tensors {
        match classify(&t.name) {
            None => out.others.push(t.clone()),
            Some((m, Half::A)) => a_side.push((m.to_owned(), t.clone())),
            Some((m, Half::B)) => b_side.push((m.to_owned(), t.clone())),
        }
    }
    for (module, a) in a_side {
        match b_side.iter().position(|(m, _)| *m == module) {
            Some(i) => {
                let (_, b) = b_side.remove(i);
                out.pairs.push(Pair { module, a, b });
            }
            None => out.orphans.push(Orphan {
                half: Half::A,
                tensor: a,
            }),
        }
    }
    out.orphans.extend(b_side.into_iter().map(|(_, t)| Orphan {
        half: Half::B,
        tensor: t,
    }));
    out.pairs.sort_by(|x, y| x.module.cmp(&y.module));
    out.orphans
        .sort_by(|x, y| x.tensor.name.cmp(&y.tensor.name));
    out
}
