//! The account of what a parse did not understand (spec §7).
//!
//! Every parse returns `(Content, Report)`: the content cannot be obtained
//! without also receiving this. A log line is ignorable by construction,
//! so the obligation lives in the type instead.
//!
//! Only **loud** unknowns appear here — things harmless to carry and
//! dangerous to drop. **Fatal** unknowns (an unrecognised container
//! version or block encoding) are [`crate::ErrorKind`] variants, because
//! they make byte-size arithmetic unknowable and continuing would hand out
//! wrong bytes rather than incomplete ones.
//!
//! An unrecognised **type code** is fatal or loud depending on the
//! container, not on the code itself: the split tracks whether the
//! unknown poisons *other* addressing. A format that derives tensor
//! offsets by accumulation — each tensor's size feeding the next tensor's
//! start — cannot survive one unreadable size, so there the code is fatal
//! for the same reason as an unrecognised version. GGUF stores each
//! tensor's offset explicitly, so an unrecognised type code there costs
//! exactly that one tensor's length; metadata and every other tensor stay
//! readable, so the code belongs here instead.

use crate::MetaValue;

/// Something a parse encountered and did not understand.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum UnrecognizedKind {
    /// A metadata key this build did not act on: one it has no canonical
    /// name for, one declared twice, or one whose value it recognised and
    /// refused. Dropping any of them silently is the worst kind of lossy,
    /// because it is invisible.
    MetadataKey {
        /// Key exactly as declared.
        key: String,
        /// Value exactly as declared, when the parse captured it.
        ///
        /// `None` when it deliberately did not. Reporting a duplicate key
        /// must not cost a value decode: the reference corpus's largest
        /// single key declares 514,906 strings, so decoding one to
        /// describe it would cost roughly 17 MB to say "this appeared
        /// twice" — that is the ONE key; the ~26 MB quoted elsewhere is the
        /// whole file's 777,056 strings, two quantities that were briefly
        /// given the same number. A reader that needs the value can seek it; a report
        /// entry exists to be cheap.
        value: Option<MetaValue>,
        /// Why this entry exists, when the key and value do not say.
        ///
        /// `None` for the plain unrecognised-key case, where the value IS
        /// the finding. `Some` when the parse recognised the key and
        /// declined it — a duplicate, or a declared value it refused —
        /// because nothing in the pair alone conveys that.
        ///
        /// This field exists because those reasons were previously stored
        /// in `value`, which is documented as the file's own bytes. An
        /// explanation sitting in a field documented as data is not a
        /// smaller problem than no explanation at all.
        reason: Option<String>,
    },
    /// A file present in a checkpoint whose role is unknown.
    File {
        /// File name as listed.
        name: String,
    },
    /// A container feature flag this build does not act on.
    FeatureFlag {
        /// Flag name.
        name: String,
        /// Raw value as declared.
        raw: u64,
    },
    /// A tensor whose declared encoding this build cannot resolve.
    ///
    /// The tensor is **omitted from the container's tensor list**, because
    /// [`crate::TensorDescriptor`] has no way to say "length unknown" — its
    /// `encoding` is not optional — and fabricating one would let a caller
    /// compute a byte range for a tensor whose extent is genuinely unknown.
    ///
    /// Omission is not silence: this entry names the tensor, the family
    /// that owns the code space, and the code itself, so a consumer can
    /// report exactly which tensor it cannot see and why. Whether an
    /// unresolvable code is merely loud (this) or fatal
    /// ([`crate::ErrorKind::UnknownTypeCode`]) is a property of the
    /// container — see this module's header.
    TensorEncoding {
        /// Tensor name exactly as declared.
        name: String,
        /// Family owning the code space, e.g. `"ggml"`.
        family: &'static str,
        /// The code exactly as declared.
        code: u32,
    },
    /// A tensor this build declined for a reason that is not its encoding.
    ///
    /// Like [`Self::TensorEncoding`], the tensor is **omitted from the
    /// container's list** — a consumer sees a shorter list and this entry is
    /// the only other signal. Unlike it, there is no type code involved: the
    /// encoding resolved fine, or the complaint is not about the encoding at
    /// all. A duplicate name and an overlapping byte range are both this.
    ///
    /// Separate from `TensorEncoding` rather than a `code: Option<u32>` added
    /// to it, because the two answer different questions and a reader
    /// branching on the kind should not have to check whether a field is
    /// meaningful before trusting it.
    ///
    /// **The overlapping-range example above is one format's rule, not this
    /// crate's.** [`crate::TensorContainer::tensors`] carries the ruling:
    /// whether two tensors may share bytes is a fact about the format —
    /// malformed in GGUF, a standard tied-weight layout in safetensors — so
    /// a format crate reports it only when its own format forbids it. This
    /// doc naming it as a reason, while the only sweep producing it lived in
    /// one backend, is what made a seam-level rule look decided when it had
    /// not been.
    TensorDeclined {
        /// Tensor name exactly as declared.
        name: String,
        /// Why, in terms a person can act on. Never a substitute for a
        /// field that exists — see the `MetadataKey` repair.
        reason: String,
    },
}

/// One unrecognised item plus where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct Unrecognized {
    /// What was not understood.
    pub kind: UnrecognizedKind,
    /// Artifact it came from, for attribution in messages.
    pub origin: String,
}

/// What a source knows about one metadata key (spec §5, consumer R2).
///
/// [`MetadataSource::get`] returns `Option`, which collapses two facts an
/// operator must be able to tell apart: a file that **declares nothing**
/// under this key, and a file that declares something the parse **could not
/// decode**. Those carry opposite remedies — supply the value, versus repair
/// the file — and a consumer given only `None` cannot say which it is
/// looking at.
///
/// `get` stays as the ergonomic path. This is the honest one.
///
/// [`MetadataSource::get`]: crate::MetadataSource::get
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Declaration<'a> {
    /// The key was not found. Never a default.
    ///
    /// **Whether this is a fact about the file depends on
    /// [`MetadataSource::index_complete`].** With a complete view it means
    /// the key is not declared. With an incomplete one it means only *not
    /// found in the part that could be read* — a walk that stopped early
    /// cannot distinguish "absent" from "past the point where we stopped",
    /// and a key may sit immediately beyond it.
    ///
    /// So a positive finding is always safe to act on and a negative one
    /// is not, until `index_complete()` says otherwise. Those are different
    /// claims: "this model declares no chat template" versus "we could not
    /// get far enough to tell".
    ///
    /// [`MetadataSource::index_complete`]: crate::MetadataSource::index_complete
    Absent,
    /// The key is declared and the value could not be decoded. Carries the
    /// report entry, so the complaint can name the key and what was seen.
    Unreadable(&'a Unrecognized),
    /// The key is declared and decoded.
    Declared(&'a MetaValue),
}

/// Everything a parse did not understand.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    entries: Vec<Unrecognized>,
}

impl Report {
    /// An empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an unrecognised item.
    pub fn push(&mut self, item: Unrecognized) {
        self.entries.push(item);
    }

    /// Absorb another report, for checkpoints spanning several files.
    pub fn merge(&mut self, other: Report) {
        self.entries.extend(other.entries);
    }

    /// Whether everything was understood.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Everything that was not understood.
    #[must_use]
    pub fn entries(&self) -> &[Unrecognized] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetaValue;

    #[test]
    fn a_fresh_report_is_empty() {
        assert!(Report::new().is_empty());
    }

    #[test]
    fn an_unknown_key_is_preserved_with_its_value_not_just_counted() {
        let mut r = Report::new();
        r.push(Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: "llama.future_thing".into(),
                value: Some(MetaValue::U32(42)),
                reason: None,
            },
            origin: "model.gguf".into(),
        });

        assert!(!r.is_empty());
        let entry = &r.entries()[0];
        assert_eq!(entry.origin, "model.gguf");
        match &entry.kind {
            UnrecognizedKind::MetadataKey { key, value, reason } => {
                assert_eq!(key, "llama.future_thing");
                assert_eq!(value.as_ref().and_then(MetaValue::as_u64), Some(42));
                // The plain unrecognised-key case: the value IS the
                // finding, so there is nothing for `reason` to add.
                assert_eq!(*reason, None);
            }
            other => panic!("wrong kind: {other:?}"),
        }
    }

    #[test]
    fn a_reason_and_a_value_are_different_fields_and_neither_impersonates_the_other() {
        // The defect this variant's shape exists to prevent: an explanation
        // sentence stored in a field documented as the file's own bytes. A
        // consumer reading `value` must always be reading the model, never
        // the parser's commentary.
        let mut r = Report::new();
        r.push(Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: "general.alignment".into(),
                value: Some(MetaValue::U64(64)),
                reason: Some("must be UINT32; using the default of 32".into()),
            },
            origin: "model.gguf".into(),
        });
        r.push(Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: "dup".into(),
                value: None,
                reason: Some("declared more than once".into()),
            },
            origin: "model.gguf".into(),
        });

        // Whole-value comparison of both entries, not a field at a time:
        // a chain of field assertions cannot see the two entries swapped.
        let got: Vec<_> = r.entries().iter().map(|e| e.kind.clone()).collect();
        assert_eq!(
            got,
            vec![
                UnrecognizedKind::MetadataKey {
                    key: "general.alignment".into(),
                    value: Some(MetaValue::U64(64)),
                    reason: Some("must be UINT32; using the default of 32".into()),
                },
                UnrecognizedKind::MetadataKey {
                    key: "dup".into(),
                    value: None,
                    reason: Some("declared more than once".into()),
                },
            ]
        );
    }

    #[test]
    fn a_tensor_encoding_entry_survives_a_merge_with_its_fields_intact() {
        let mut a = Report::new();
        a.push(Unrecognized {
            kind: UnrecognizedKind::TensorEncoding {
                name: "blk.0.attn_q.weight".into(),
                family: "ggml",
                code: 9999,
            },
            origin: "model.gguf".into(),
        });

        let mut b = Report::new();
        b.push(Unrecognized {
            kind: UnrecognizedKind::TensorEncoding {
                name: "blk.1.attn_q.weight".into(),
                family: "ggml",
                code: 9999,
            },
            origin: "shard-2.gguf".into(),
        });
        a.merge(b);

        assert_eq!(a.entries().len(), 2);
        match &a.entries()[0].kind {
            UnrecognizedKind::TensorEncoding { name, family, code } => {
                assert_eq!(name, "blk.0.attn_q.weight");
                assert_eq!(*family, "ggml");
                assert_eq!(*code, 9999);
            }
            other => panic!("wrong kind: {other:?}"),
        }
        assert_eq!(a.entries()[0].origin, "model.gguf");
        match &a.entries()[1].kind {
            UnrecognizedKind::TensorEncoding { name, family, code } => {
                assert_eq!(name, "blk.1.attn_q.weight");
                assert_eq!(*family, "ggml");
                assert_eq!(*code, 9999);
            }
            other => panic!("wrong kind: {other:?}"),
        }
        assert_eq!(a.entries()[1].origin, "shard-2.gguf");
    }

    fn key(k: &str, v: u32, origin: &str) -> Unrecognized {
        Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: k.into(),
                value: Some(MetaValue::U32(v)),
                reason: None,
            },
            origin: origin.into(),
        }
    }

    #[test]
    fn reports_merge_so_a_multi_file_checkpoint_yields_one_account() {
        // Identity and order, not arity. The previous version asserted only
        // `entries().len() == 2` over two fixtures of *different* kinds, so
        // a merge that deduplicated by kind and prepended — turning a
        // sharded checkpoint's 40 distinct unknown keys into a report of 1 —
        // passed. Spec §7 is explicit: "The report names the key, its typed
        // value, and its origin — not a count."
        let mut a = Report::new();
        a.push(key("llama.future_a", 1, "shard-1"));
        a.push(key("llama.future_b", 2, "shard-1"));

        let mut b = Report::new();
        b.push(key("llama.future_c", 3, "shard-2"));
        b.push(key("llama.future_d", 4, "shard-2"));

        a.merge(b);

        assert_eq!(
            a.entries(),
            &[
                key("llama.future_a", 1, "shard-1"),
                key("llama.future_b", 2, "shard-1"),
                key("llama.future_c", 3, "shard-2"),
                key("llama.future_d", 4, "shard-2"),
            ]
        );
    }

    #[test]
    fn merge_preserves_duplicate_kinds_from_different_origins() {
        // The same key unknown in every shard is four facts, not one: which
        // shards carried it is the thing an operator needs.
        let mut a = Report::new();
        a.push(key("llama.future_thing", 7, "shard-1"));
        let mut b = Report::new();
        b.push(key("llama.future_thing", 7, "shard-2"));

        a.merge(b);
        assert_eq!(a.entries().len(), 2);
        assert_eq!(a.entries()[0].origin, "shard-1");
        assert_eq!(a.entries()[1].origin, "shard-2");
    }

    #[test]
    fn merge_absorbs_every_kind() {
        let mut a = Report::new();
        a.push(Unrecognized {
            kind: UnrecognizedKind::File {
                name: "extra.bin".into(),
            },
            origin: "dir".into(),
        });
        let mut b = Report::new();
        b.push(Unrecognized {
            kind: UnrecognizedKind::FeatureFlag {
                name: "flag".into(),
                raw: 3,
            },
            origin: "shard-2".into(),
        });

        a.merge(b);
        assert_eq!(a.entries().len(), 2);
        assert!(matches!(a.entries()[0].kind, UnrecognizedKind::File { .. }));
        assert!(matches!(
            a.entries()[1].kind,
            UnrecognizedKind::FeatureFlag { .. }
        ));
    }

    #[test]
    fn a_declined_tensor_says_why_without_inventing_a_type_code() {
        // `TensorEncoding` answers "this build cannot resolve the encoding".
        // A duplicate name and an overlapping range are neither, and forcing
        // them into it means a `code` field holding a number the file never
        // declared. That is the defect this enum's MetadataKey sibling was
        // repaired for.
        let mut r = Report::new();
        r.push(Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: "blk.0.attn_q.weight".into(),
                reason: "declared more than once; the first occurrence is kept".into(),
            },
            origin: "model.gguf".into(),
        });
        r.push(Unrecognized {
            kind: UnrecognizedKind::TensorDeclined {
                name: "blk.1.attn_k.weight".into(),
                reason: "byte range overlaps blk.0.attn_q.weight".into(),
            },
            origin: "model.gguf".into(),
        });

        // Whole values, in order. A chain of field assertions cannot see the
        // two entries swapped, and swapping them attributes each complaint
        // to the wrong tensor.
        assert_eq!(
            r.entries()
                .iter()
                .map(|e| e.kind.clone())
                .collect::<Vec<_>>(),
            vec![
                UnrecognizedKind::TensorDeclined {
                    name: "blk.0.attn_q.weight".into(),
                    reason: "declared more than once; the first occurrence is kept".into(),
                },
                UnrecognizedKind::TensorDeclined {
                    name: "blk.1.attn_k.weight".into(),
                    reason: "byte range overlaps blk.0.attn_q.weight".into(),
                },
            ]
        );
    }
}
