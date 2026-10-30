// SPDX-License-Identifier: MIT OR Apache-2.0
//! Special-token declarations.
//!
//! Every field here is something a file SAID. Nothing here is something a
//! runtime SHOULD DO — see [`crate::template`] for why that distinction is
//! load-bearing rather than stylistic.

use mlmf_core::{MetaValue, MetadataSource};

use crate::vocab::{Format, keys, read, spelling};

/// A declared special token: the id, and its text when the vocabulary
/// contains that index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialToken {
    /// The id exactly as declared.
    pub id: u64,
    /// The token string, or `None` when the id is not a valid index.
    /// **The id survives either way** — it is what the file said.
    ///
    /// **This field is load-bearing and must not be optimised away.** Spec
    /// §9 4.1 (corrected, PR #12) puts the `add_special_tokens` decision on
    /// the consumer, computed as `!bos.is_empty() && text.starts_with(bos)`
    /// against the consumer's own rendered string. **This string is that
    /// operand.** Reducing this type to just `id` would remove the
    /// consumer's ability to make the check at all, and no test in this
    /// crate would go red for it — which is why the reason is written here
    /// rather than left to be inferred.
    pub text: Option<String>,
}

/// What a source declared about its special tokens.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    /// The beginning-of-sequence token, if declared.
    pub bos: Option<SpecialToken>,
    /// The end-of-sequence token, if declared. May equal `bos` — **nine**
    /// corpus files declare the same id for both (measured 2026-09-03).
    pub eos: Option<SpecialToken>,
    /// What the file DECLARED about prepending BOS. `None` means the file
    /// did not say, which is not the same as saying `false`.
    ///
    /// **This is NOT `add_special_tokens`, and passing it as such
    /// reproduces the defect §9 4.1 describes.** That flag is a property of
    /// a *rendered string*; this is a key a *file* declared. The corpus
    /// refutes any unconditional rule in either direction: `add_bos_token`
    /// is true in 7 files and false in 9. Compute the flag from the render
    /// — see [`SpecialToken::text`].
    pub add_bos_declared: Option<bool>,
    /// What the file DECLARED about appending EOS. See
    /// [`Self::add_bos_declared`], including the warning.
    pub add_eos_declared: Option<bool>,
    /// Format-spelled keys this extraction reads that the file DECLARED but
    /// whose value could not be decoded. A field left `None` because its key
    /// is listed here is a file that needs repair, not a file that said
    /// nothing: the two have opposite remedies (S-17).
    pub unreadable: Vec<String>,
    /// `true` when the source did not see every key its container declared
    /// ([`MetadataSource::index_complete`] was `false`). Then a `None` field
    /// above means only *not found in the part that could be read*, not
    /// *not declared*.
    pub index_incomplete: bool,
}

impl SpecialTokens {
    /// Read the special-token declarations under `format`'s spellings.
    #[must_use]
    pub fn extract<S: MetadataSource + ?Sized>(source: &S, format: Format) -> Self {
        let mut unreadable: Vec<String> = Vec::new();
        let mut lookup = |canonical: &str| {
            spelling(canonical, format).and_then(|k| read(source, k, &mut unreadable))
        };

        let vocab: Option<&[MetaValue]> = lookup(keys::TOKENS).and_then(MetaValue::as_array);
        let bos_id = lookup(keys::BOS_TOKEN_ID).and_then(MetaValue::as_u64);
        let eos_id = lookup(keys::EOS_TOKEN_ID).and_then(MetaValue::as_u64);
        let add_bos_declared = lookup(keys::ADD_BOS_TOKEN).and_then(MetaValue::as_bool);
        let add_eos_declared = lookup(keys::ADD_EOS_TOKEN).and_then(MetaValue::as_bool);

        let token = |id: Option<u64>| -> Option<SpecialToken> {
            let id = id?;
            let text = vocab
                .and_then(|v| usize::try_from(id).ok().and_then(|i| v.get(i)))
                .and_then(MetaValue::as_str)
                .cloned();
            Some(SpecialToken { id, text })
        };

        let (bos, eos) = (token(bos_id), token(eos_id));
        unreadable.sort_unstable();
        Self {
            bos,
            eos,
            add_bos_declared,
            add_eos_declared,
            unreadable,
            index_incomplete: !source.index_complete(),
        }
    }
}
