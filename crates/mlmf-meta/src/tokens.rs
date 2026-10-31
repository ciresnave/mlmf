// SPDX-License-Identifier: MIT OR Apache-2.0
//! Special-token declarations.
//!
//! Every field here is something a file SAID. Nothing here is something a
//! runtime SHOULD DO -- see [`crate::template`] for why that distinction is
//! load-bearing rather than stylistic.

use mlmf_core::{MetaValue, MetadataSource};

use crate::vocab::{Format, keys, read, spelling};

/// How a token's text was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextSource {
    /// Declared directly, as HuggingFace's `tokenizer_config.json` does.
    Declared,
    /// Resolved by indexing the id into the token table, as GGUF does.
    Indexed,
}

/// A declared special token: its id and/or its text.
///
/// A format declares a token by id (GGUF), by text (HuggingFace's
/// `tokenizer_config.json`), or both (HuggingFace's `config.json` id next to
/// the `tokenizer_config.json` text). A token declared by text alone has no
/// id, so `id` is `Option`: inventing `0` would be a supplied default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialToken {
    /// The id exactly as declared, or `None` when the format declared the
    /// token by text and nothing declared an id.
    pub id: Option<u64>,
    /// The token string: declared directly, or found by indexing `id` into
    /// the token table. `None` when neither was possible.
    /// **The id survives either way** -- it is what the file said.
    ///
    /// **This field is load-bearing and must not be optimised away.** Spec
    /// §9 4.1 (corrected, PR #12) puts the `add_special_tokens` decision on
    /// the consumer, computed as `!bos.is_empty() && text.starts_with(bos)`
    /// against the consumer's own rendered string. **This string is that
    /// operand.** Reducing this type to just `id` would remove the
    /// consumer's ability to make the check at all, and no test in this
    /// crate would go red for it -- which is why the reason is written here
    /// rather than left to be inferred.
    pub text: Option<String>,
    /// Where `text` came from; `Some` exactly when `text` is `Some`. When a
    /// source offers both a direct declaration and a token table, the
    /// direct declaration wins and says so here.
    pub text_source: Option<TextSource>,
}

/// What a source declared about its special tokens.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    /// The beginning-of-sequence token, if declared.
    pub bos: Option<SpecialToken>,
    /// The end-of-sequence token, if declared. May equal `bos` -- **nine**
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
    /// -- see [`SpecialToken::text`].
    pub add_bos_declared: Option<bool>,
    /// What the file DECLARED about appending EOS. See
    /// [`Self::add_bos_declared`], including the warning.
    pub add_eos_declared: Option<bool>,
    /// Format-spelled keys this extraction reads that the file DECLARED but
    /// whose value could not be decoded. A field left `None` because its key
    /// is listed here is a file that needs repair, not a file that said
    /// nothing: the two have opposite remedies (S-17).
    pub unreadable: Vec<String>,
    /// Format-spelled keys the file declared and the source decoded, but
    /// with a shape this extraction does not read: a token id that is not a
    /// single integer (HuggingFace's `config.json:eos_token_id` may be a
    /// LIST of ids, as `NousResearch/Meta-Llama-3.1-8B-Instruct` declares),
    /// a text that is not a string, a flag that is not a boolean. Unlike
    /// [`Self::unreadable`] nothing is broken; the field is `None` because
    /// this type has no place for what was declared, not because the file
    /// said nothing.
    pub unusable: Vec<String>,
    /// `true` when the source did not see every key its container declared
    /// ([`MetadataSource::index_complete`] was `false`). Then a `None` field
    /// above means only *not found in the part that could be read*, not
    /// *not declared*.
    pub index_incomplete: bool,
}

/// Reads canonical keys through one format's spellings, recording the keys
/// that were declared but unreadable or unusable.
struct Reader<'a, S: MetadataSource + ?Sized> {
    source: &'a S,
    format: Format,
    unreadable: Vec<String>,
    unusable: Vec<String>,
}

impl<'a, S: MetadataSource + ?Sized> Reader<'a, S> {
    /// The declared value of `canonical`, with the key it was read from.
    fn value(&mut self, canonical: &str) -> Option<(&'static str, &'a MetaValue)> {
        let key = spelling(canonical, self.format)?;
        read(self.source, key, &mut self.unreadable).map(|v| (key, v))
    }

    /// `canonical` as `T`, or `None`; a declared value that is not a `T`
    /// is recorded as unusable.
    fn typed<T>(
        &mut self,
        canonical: &str,
        convert: impl FnOnce(&'a MetaValue) -> Option<T>,
    ) -> Option<T> {
        let (key, value) = self.value(canonical)?;
        let got = convert(value);
        if got.is_none() {
            self.unusable.push(key.to_string());
        }
        got
    }

    /// One token: its declared id, and its text by the two paths in order,
    /// a direct declaration first, then indexing the id into `vocab`.
    fn token(
        &mut self,
        id_key: &str,
        text_key: &str,
        vocab: Option<&[MetaValue]>,
    ) -> Option<SpecialToken> {
        let id = self.typed(id_key, MetaValue::as_u64);
        let declared = self.typed(text_key, |v| v.as_str().cloned());
        let (text, text_source) = resolve_text(declared, id, vocab);
        if id.is_none() && text.is_none() {
            return None;
        }
        Some(SpecialToken {
            id,
            text,
            text_source,
        })
    }
}

/// A token's text: a direct declaration first, then indexing `id` into
/// `vocab`; the source says which.
fn resolve_text(
    declared: Option<String>,
    id: Option<u64>,
    vocab: Option<&[MetaValue]>,
) -> (Option<String>, Option<TextSource>) {
    if let Some(t) = declared {
        return (Some(t), Some(TextSource::Declared));
    }
    let indexed = id
        .and_then(|i| usize::try_from(i).ok())
        .and_then(|at| vocab?.get(at)?.as_str().cloned());
    match indexed {
        Some(t) => (Some(t), Some(TextSource::Indexed)),
        None => (None, None),
    }
}

impl SpecialTokens {
    /// Read the special-token declarations under `format`'s spellings.
    #[must_use]
    pub fn extract<S: MetadataSource + ?Sized>(source: &S, format: Format) -> Self {
        let mut r = Reader {
            source,
            format,
            unreadable: Vec::new(),
            unusable: Vec::new(),
        };
        let vocab: Option<&[MetaValue]> = r.typed(keys::TOKENS, MetaValue::as_array);
        let bos = r.token(keys::BOS_TOKEN_ID, keys::BOS_TOKEN_TEXT, vocab);
        let eos = r.token(keys::EOS_TOKEN_ID, keys::EOS_TOKEN_TEXT, vocab);
        let add_bos_declared = r.typed(keys::ADD_BOS_TOKEN, MetaValue::as_bool);
        let add_eos_declared = r.typed(keys::ADD_EOS_TOKEN, MetaValue::as_bool);

        let Reader {
            mut unreadable,
            mut unusable,
            ..
        } = r;
        unreadable.sort_unstable();
        unusable.sort_unstable();
        Self {
            bos,
            eos,
            add_bos_declared,
            add_eos_declared,
            unreadable,
            unusable,
            index_incomplete: !source.index_complete(),
        }
    }
}
