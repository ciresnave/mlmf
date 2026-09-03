//! Chat-template extraction.
//!
//! # What this returns, and why the shape is load-bearing
//!
//! Spec §9 clause 4.1, as corrected in PR #12: a rendered template must be
//! tokenized with `add_special_tokens` set to whether the RENDER opens with
//! BOS. **That is a property of the rendered string, which extraction never
//! produces**, so there is no `add_special_tokens` field here and no method
//! answering "should I add BOS?".
//!
//! What the consumer needs instead is the template plus the declared BOS
//! string, so it can do `text.starts_with(bos)` on its own render — see
//! `tokens::SpecialToken::text`, which exists for that check and must not
//! be optimised away.
//!
//! (Deliberately NOT an intra-doc link. `tokens` is registered one task
//! after this file lands, and a bracketed `crate::tokens::…` link here
//! makes `cargo doc -D warnings` — a CI step and a `local-gates.sh` step —
//! fail at that intermediate commit. The full linked reference lives on the
//! target itself, where it always resolves.)

use mlmf_core::{MetaValue, MetadataSource};

use crate::vocab::{Format, keys, spelling};

/// One template: the default (`name: None`) or a named variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateEntry {
    /// `None` for the default template, which is unnamed in every format
    /// that carries one.
    pub name: Option<String>,
    /// The template body, verbatim. Never parsed, never rendered.
    pub body: String,
}

/// Every template a source declared, plus every way it did not line up.
///
/// `blank` and `not_a_string` exist because §5 rule 3 requires lossiness
/// marked **per key**: *"a crate-level caveat is documentation, and
/// documentation loses that argument."* Without them, a file declaring a
/// blank template, a file declaring a number, and a file declaring nothing
/// all produce an identical empty result.
///
/// `named_but_absent` and `present_but_unnamed` exist for a different
/// reason: when the names array and the key set disagree, **the
/// disagreement is what the file says**, and reporting one side alone
/// asserts a consistency the file never claimed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TemplateSet {
    /// Templates found, default first when present.
    pub entries: Vec<TemplateEntry>,
    /// Keys a name in the names array pointed at that yielded no usable
    /// template. Spelled as the key it would have been — see
    /// [`Self::present_but_unnamed`] for why.
    ///
    /// **Three situations produce an entry here, and the other lists say
    /// which:** the key is genuinely absent (this list only); the key
    /// exists but is blank (also in [`Self::blank`]); the key exists but is
    /// not usable as text (also in [`Self::not_a_string`], whose doc notes
    /// that `Bytes` reaches it too). The pair is the discrimination — a
    /// consumer merging the lists for a report should deduplicate by key
    /// and keep the more specific reason.
    pub named_but_absent: Vec<String>,
    /// Body keys present but not listed in the names array. Their bodies
    /// ARE in `entries`; this list records that they were unannounced.
    ///
    /// **Like all four lists, this holds the full format-spelled key, not
    /// the bare name.** Sibling `Vec<String>` fields that switch vocabulary
    /// halfway are a trap for any consumer that concatenates them into one
    /// lossiness report.
    pub present_but_unnamed: Vec<String>,
    /// Keys whose value was a string but blank after `trim()`, and so are
    /// UNDECLARED per §9 clause 1.1 rather than declared-empty.
    pub blank: Vec<String>,
    /// Keys present whose value this crate cannot use as template text.
    /// Never coerced.
    ///
    /// **The name is a simplification and [`mlmf_core::MetaValue::Bytes`]
    /// is the exception.** `Bytes` is *"a declared string whose bytes are
    /// not valid UTF-8, preserved verbatim"*, so a file landing here may
    /// well have declared a string — just not one this crate will hand out
    /// as `&str`. The value is recorded rather than dropped, which is what
    /// §5 rule 3 requires; only the label is coarse, and it is coarse in
    /// the direction of over-reporting.
    pub not_a_string: Vec<String>,
}

impl TemplateSet {
    /// The default template's body, if the source declared a non-blank one.
    #[must_use]
    pub fn default_body(&self) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.name.is_none())
            .map(|e| e.body.as_str())
    }

    /// Read every template `source` declared under `format`'s spellings.
    ///
    /// Enumerates from the key set and cross-checks against the names
    /// array, rather than trusting either alone.
    #[must_use]
    pub fn extract<S: MetadataSource + ?Sized>(source: &S, format: Format) -> Self {
        let mut out = Self::default();

        let Some(default_key) = spelling(keys::CHAT_TEMPLATE, format) else {
            return out;
        };

        // Section 9 clause 1.1: blank is UNDECLARED. `trim()`, not
        // `is_empty()` -- a whitespace-only template parses fine and
        // renders an empty prompt.
        //
        // NOT `let mut`. This closure takes `out` as a PARAMETER rather
        // than capturing it, so it is `Fn` and the binding never needs to
        // be mutable -- and `unused_mut` is denied by the
        // `cargo clippy -p mlmf-meta --all-targets -- -D warnings` step in
        // ci.yml, so a `mut` here would be a build failure in this crate's
        // own gate rather than a style nit.
        let classify = |key: &str, out: &mut Self| -> Option<String> {
            let value = source.get(key)?;
            let Some(body) = value.as_str() else {
                out.not_a_string.push(key.to_string());
                return None;
            };
            if body.trim().is_empty() {
                out.blank.push(key.to_string());
                return None;
            }
            Some(body.clone())
        };

        if let Some(body) = classify(default_key, &mut out) {
            out.entries.push(TemplateEntry { name: None, body });
        }

        // Named bodies live under `<default_key>.<name>`. Enumerate them
        // from the keys actually present.
        //
        // The trailing dot is load-bearing: without it this prefix also
        // matches `tokenizer.chat_templates`, the names ARRAY. This repo's
        // gate history includes `strip_prefix("default")` silently matching
        // `default-tls`, which is the same defect one crate over. The
        // `!n.is_empty()` filter is the other half, rejecting the exact key
        // `tokenizer.chat_template.`.
        let prefix = format!("{default_key}.");
        let named: Vec<String> = source
            .keys()
            .into_iter()
            .filter_map(|k| k.strip_prefix(prefix.as_str()).map(ToString::to_string))
            .filter(|n| !n.is_empty())
            .collect();

        let mut found: Vec<String> = Vec::new();
        for name in named {
            let key = format!("{prefix}{name}");
            if let Some(body) = classify(&key, &mut out) {
                found.push(name.clone());
                out.entries.push(TemplateEntry {
                    name: Some(name),
                    body,
                });
            }
        }

        // Cross-check against the declared names, if the format has such a
        // key and the source declared it.
        let declared: Vec<String> = spelling(keys::CHAT_TEMPLATE_NAMES, format)
            .and_then(|k| source.get(k))
            .and_then(MetaValue::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(MetaValue::as_str)
                    .map(ToString::to_string)
                    .collect()
            })
            .unwrap_or_default();

        // Both push the FULL key, never the bare name -- see the field
        // docs. All four loss lists must speak one vocabulary.
        //
        // No `if !declared.is_empty()` guard on the second loop, and that
        // is deliberate: a file carrying `tokenizer.chat_template.orphan`
        // and no names array genuinely did leave it unannounced, and
        // suppressing that is the invisible loss §5 rule 1 forbids.
        for name in &declared {
            if !found.contains(name) {
                out.named_but_absent.push(format!("{prefix}{name}"));
            }
        }
        for name in &found {
            if !declared.contains(name) {
                out.present_but_unnamed.push(format!("{prefix}{name}"));
            }
        }

        out.named_but_absent.sort_unstable();
        out.present_but_unnamed.sort_unstable();
        out.blank.sort_unstable();
        out.not_a_string.sort_unstable();
        out
    }
}
