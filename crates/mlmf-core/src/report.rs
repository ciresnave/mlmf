//! The account of what a parse did not understand (spec §7).
//!
//! Every parse returns `(Content, Report)`: the content cannot be obtained
//! without also receiving this. A log line is ignorable by construction,
//! so the obligation lives in the type instead.
//!
//! Only **loud** unknowns appear here — things harmless to carry and
//! dangerous to drop. **Fatal** unknowns (an unrecognised type code,
//! version or encoding) are [`crate::ErrorKind`] variants, because they
//! make byte-size arithmetic unknowable and continuing would hand out
//! wrong bytes rather than incomplete ones.

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
    fn reports_merge_so_a_multi_file_checkpoint_yields_one_account() {
        let mut a = Report::new();
        a.push(Unrecognized {
            kind: UnrecognizedKind::File { name: "extra.bin".into() },
            origin: "dir".into(),
        });
        let mut b = Report::new();
        b.push(Unrecognized {
            kind: UnrecognizedKind::FeatureFlag { name: "flag".into(), raw: 3 },
            origin: "shard-2".into(),
        });

        a.merge(b);
        assert_eq!(a.entries().len(), 2);
    }
}
