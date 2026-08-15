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
    /// A metadata key this build has no canonical name for. The value is
    /// preserved verbatim — dropping unrecognised keys is the worst kind
    /// of lossy, because it is invisible.
    MetadataKey {
        /// Key exactly as declared.
        key: String,
        /// Value exactly as declared.
        value: MetaValue,
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
}

/// One unrecognised item plus where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct Unrecognized {
    /// What was not understood.
    pub kind: UnrecognizedKind,
    /// Artifact it came from, for attribution in messages.
    pub origin: String,
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
                value: MetaValue::U32(42),
            },
            origin: "model.gguf".into(),
        });

        assert!(!r.is_empty());
        let entry = &r.entries()[0];
        assert_eq!(entry.origin, "model.gguf");
        match &entry.kind {
            UnrecognizedKind::MetadataKey { key, value } => {
                assert_eq!(key, "llama.future_thing");
                assert_eq!(value.as_u64(), Some(42));
            }
            other => panic!("wrong kind: {other:?}"),
        }
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
                value: MetaValue::U32(v),
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
        assert!(matches!(
            a.entries()[0].kind,
            UnrecognizedKind::File { .. }
        ));
        assert!(matches!(
            a.entries()[1].kind,
            UnrecognizedKind::FeatureFlag { .. }
        ));
    }
}
