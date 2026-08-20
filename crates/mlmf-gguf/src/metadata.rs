//! The key-value block, indexed at open and decoded on demand.

use std::sync::OnceLock;

use mlmf_core::{Declaration, MetaValue, MetadataSource, Report, Unrecognized, UnrecognizedKind};

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};
use crate::header::{Header, parse_header};
use crate::value::{ValueType, decode_value, skip_value};

/// GGUF's documented default when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;

/// One indexed key: where its value is, and its value once decoded.
#[derive(Debug)]
struct Entry {
    key: String,
    ty: ValueType,
    /// Offset of the value's first byte.
    start: u64,
    /// Set when the value could not be indexed — an unknown type code.
    ///
    /// Owned per entry rather than shared, because [`Declaration::Unreadable`]
    /// borrows and the entry it names must be *this* key's. A shared
    /// placeholder initialised once would report the first unreadable key's
    /// name for every subsequent one.
    unreadable: Option<Unrecognized>,
    /// Decoded on first access. `OnceLock` rather than `OnceCell` so the
    /// metadata stays `Sync` — a consumer loading tensors in parallel will
    /// read metadata from several threads.
    value: OnceLock<MetaValue>,
}

/// A GGUF file's metadata, indexed but not decoded.
///
/// Opening indexes every key into `(key, type, offset)` and decodes
/// nothing. A value is decoded on first access and cached. The largest file
/// in the reference corpus declares 777,056 strings across 42 keys, so
/// eager decoding would cost roughly 26 MB of allocations to answer a
/// question about one of them.
#[derive(Debug)]
pub struct GgufMetadata<'a> {
    bytes: &'a [u8],
    header: Header,
    entries: Vec<Entry>,
    alignment: u64,
    kv_end: u64,
    /// False when the index stopped before every declared key was seen.
    index_complete: bool,
}

impl<'a> GgufMetadata<'a> {
    /// Parse the header and index the key-value block.
    ///
    /// Returns the metadata **and** a [`Report`] of everything the parse did
    /// not understand. The report is not optional: a caller cannot obtain
    /// the content without also receiving the account of what was skipped.
    ///
    /// `origin` names the artifact in report entries — a file name, a URL,
    /// whatever the caller can show an operator.
    ///
    /// # Errors
    ///
    /// [`GgufError::NotGguf`] if the magic is wrong, which is a different
    /// fact from a malformed GGUF (R7). [`GgufError::Truncated`] or
    /// [`GgufError::Malformed`] with `Stage::Metadata` if the KV block ends
    /// early or contains something structurally impossible.
    pub fn parse(bytes: &'a [u8], origin: &str) -> Result<(Self, Report), GgufError> {
        let mut cursor = Cursor::new(bytes);
        let header = parse_header(&mut cursor)?;
        let mut report = Report::new();
        let mut index_complete = true;
        // Deliberately NOT `Vec::with_capacity(header.kv_count)`. The count
        // is a declared number this build has only checked for negativity,
        // so `i64::MAX` reaches here intact; preallocating from it panics on
        // capacity overflow before any truncation check runs. Growing as we
        // go costs nothing at the 42-key scale real files use.
        let mut entries: Vec<Entry> = Vec::new();

        for _ in 0..header.kv_count {
            let key = read_key(&mut cursor)?;
            let at = cursor.pos();
            let code = cursor.u32().map_err(|t| GgufError::Truncated {
                stage: Stage::Metadata,
                offset: at,
                needed: t.needed,
                available: t.available,
            })?;

            let Some(ty) = ValueType::from_code(code) else {
                // An unknown value type has an unknown width, so the parse
                // cannot find the next key. Everything indexed so far stays
                // readable — which is R1's guarantee applied within the
                // metadata stage itself — and the failure is reported rather
                // than silently truncating the key list.
                let complaint = Unrecognized {
                    kind: UnrecognizedKind::MetadataKey {
                        key: key.clone(),
                        value: MetaValue::U32(code),
                    },
                    origin: origin.to_string(),
                };
                report.push(complaint.clone());
                entries.push(Entry {
                    key,
                    ty: ValueType::U8, // never read; `unreadable` gates access
                    start: 0,
                    unreadable: Some(complaint),
                    value: OnceLock::new(),
                });
                index_complete = false;
                break;
            };

            let start = cursor.pos();
            skip_value(&mut cursor, ty)?;

            if entries.iter().any(|e| e.key == key) {
                // Deterministic and loud: first wins, second reported. Taking
                // the last would make the file's meaning depend on parse order.
                report.push(Unrecognized {
                    kind: UnrecognizedKind::MetadataKey {
                        key: key.clone(),
                        value: MetaValue::String("duplicate key; first occurrence kept".into()),
                    },
                    origin: origin.to_string(),
                });
                continue;
            }

            entries.push(Entry {
                key,
                ty,
                start,
                unreadable: None,
                value: OnceLock::new(),
            });
        }

        let kv_end = cursor.pos();
        let mut me = Self {
            bytes,
            header,
            entries,
            alignment: DEFAULT_ALIGNMENT,
            kv_end,
            index_complete,
        };
        me.alignment = me.resolve_alignment(origin, &mut report);
        Ok((me, report))
    }

    /// The file's header.
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Whether every key the header declared was indexed.
    ///
    /// `false` means the walk stopped early: an unknown value type has an
    /// unknown width, so the parse cannot find the key that follows it.
    ///
    /// **This changes what [`mlmf_core::Declaration::Absent`] means, and a
    /// caller that ignores it will draw a false conclusion.** With a
    /// complete index, `Absent` is a fact about the file — the key is not
    /// declared. With an incomplete one it means only *not found in the part
    /// that could be read*, and the key may be sitting immediately past the
    /// point where the walk stopped. Those are different claims:
    /// "this model declares no chat template" versus "we could not get far
    /// enough to tell". A count or a `keys()` listing taken from an
    /// incomplete index can support a positive finding — this key IS here —
    /// and never a negative one.
    #[must_use]
    pub fn index_complete(&self) -> bool {
        self.index_complete
    }

    /// Offset just past the key-value block.
    ///
    /// The tensor directory begins here. Exposed because only this stage
    /// knows it, and the tensor stage needs it.
    #[must_use]
    pub fn kv_end(&self) -> u64 {
        self.kv_end
    }

    /// Alignment for the tensor data region.
    ///
    /// `general.alignment` when the file declares a valid one, otherwise
    /// GGUF's documented default of 32.
    ///
    /// **This is the effective value, and it does not say where it came
    /// from.** A file that declares 32 and a file that declares nothing
    /// both answer 32 here. That is the right shape for a caller who needs
    /// a number, but a caller who needs the *fact* must ask
    /// `declaration("general.alignment")`, which separates declared from
    /// absent from undecodable. Spec §5 says absent never means a default;
    /// this method returns a default precisely because alignment has a
    /// documented one, and the raw fact stays reachable beside it. An invalid declaration is reported
    /// and falls back rather than failing the open — this is metadata, and
    /// R1 says reading metadata survives.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    fn resolve_alignment(&self, origin: &str, report: &mut Report) -> u64 {
        let Some(v) = self.get("general.alignment") else {
            return DEFAULT_ALIGNMENT;
        };
        let complain = |report: &mut Report, why: &str| {
            report.push(Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "general.alignment".to_string(),
                    value: MetaValue::String(why.to_string()),
                },
                origin: origin.to_string(),
            });
        };
        // llama.cpp requires UINT32 specifically (gguf.cpp:614).
        let MetaValue::U32(a) = v else {
            complain(report, "must be UINT32; using the default of 32");
            return DEFAULT_ALIGNMENT;
        };
        let a = u64::from(*a);
        // And a power of two (gguf.cpp:623).
        if a == 0 || !a.is_power_of_two() {
            complain(report, "must be a power of two; using the default of 32");
            return DEFAULT_ALIGNMENT;
        }
        a
    }

    fn entry(&self, key: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.key == key)
    }

    /// Decode an entry's value, caching it.
    fn value_of<'e>(&self, e: &'e Entry) -> Option<&'e MetaValue> {
        if e.unreadable.is_some() {
            return None;
        }
        Some(e.value.get_or_init(|| {
            let mut c = Cursor::new(self.bytes);
            // Both operations succeeded during indexing, so neither can fail
            // here; a failure would mean the byte slice changed underneath us,
            // which is impossible for a shared borrow.
            c.seek(e.start).expect("indexed offset is in range");
            decode_value(&mut c, e.ty).expect("indexed value decoded during parse")
        }))
    }
}

/// Read one length-prefixed key.
fn read_key(cursor: &mut Cursor<'_>) -> Result<String, GgufError> {
    let at = cursor.pos();
    let len = cursor.u64().map_err(|t| GgufError::Truncated {
        stage: Stage::Metadata,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;
    let at = cursor.pos();
    let raw = cursor.take(len).map_err(|t| GgufError::Truncated {
        stage: Stage::Metadata,
        offset: at,
        needed: t.needed,
        available: t.available,
    })?;
    // A key is a lookup token, so it must be UTF-8 to be usable as one.
    // Values are different: a non-UTF-8 *value* survives as MetaValue::Bytes.
    core::str::from_utf8(raw)
        .map(str::to_string)
        .map_err(|e| GgufError::Malformed {
            stage: Stage::Metadata,
            offset: at,
            detail: format!("key is not valid UTF-8: {e}"),
        })
}

impl MetadataSource for GgufMetadata<'_> {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entry(key).and_then(|e| self.value_of(e))
    }

    fn keys(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.key.as_str()).collect()
    }

    fn declaration(&self, key: &str) -> Declaration<'_> {
        match self.entry(key) {
            None => Declaration::Absent,
            // Indexed but not decodable: the key is in the file and its value
            // is not readable. Exactly the state R2 exists to separate from
            // Absent, and the entry it borrows names *this* key.
            Some(e) if e.unreadable.is_some() => {
                Declaration::Unreadable(e.unreadable.as_ref().expect("just checked"))
            }
            Some(e) => match self.value_of(e) {
                Some(v) => Declaration::Declared(v),
                None => Declaration::Absent,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlmf_core::{Declaration, MetaValue, MetadataSource};

    /// Build a minimal GGUF: header plus the given KV pairs, no tensors.
    ///
    /// Each pair is (key, value type code, already-encoded value bytes).
    fn gguf(kvs: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0i64.to_le_bytes()); // no tensors
        b.extend_from_slice(&(kvs.len() as i64).to_le_bytes());
        for (k, ty, v) in kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(v);
        }
        b
    }

    fn s(v: &str) -> Vec<u8> {
        let mut b = (v.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(v.as_bytes());
        b
    }

    fn str_array(items: &[&str]) -> Vec<u8> {
        let mut b = 8u32.to_le_bytes().to_vec(); // String elements
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for i in items {
            b.extend_from_slice(&s(i));
        }
        b
    }

    #[test]
    fn reads_a_key_without_decoding_the_rest() {
        let bytes = gguf(&[
            ("general.architecture", 8, s("llama")),
            ("tokenizer.ggml.tokens", 9, str_array(&["a", "b", "c"])),
        ]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("parses");
        assert!(report.is_empty());
        assert_eq!(
            m.get("general.architecture"),
            Some(&MetaValue::String("llama".into()))
        );
        assert_eq!(m.keys().len(), 2);
    }

    #[test]
    fn an_unknown_value_type_costs_that_key_and_not_the_parse() {
        // R1's shape, at the metadata layer. A key this build cannot decode
        // must not stop the keys around it from being readable — and it must
        // be reported rather than dropped.
        //
        // Value type 13 does not exist, so the parse cannot know where the
        // value ends and must stop indexing; the keys BEFORE it stay
        // readable and the failure is reported.
        let bytes = gguf(&[
            ("first", 8, s("kept")),
            ("broken", 13, s("unreachable")),
            ("third", 8, s("lost")),
        ]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("does not fail the open");
        assert_eq!(m.get("first"), Some(&MetaValue::String("kept".into())));
        assert!(matches!(
            m.declaration("broken"),
            Declaration::Unreadable(_)
        ));
        // `third` reads as Absent, and that is the honest limit of what
        // this API can say: an unknown width means the parse never found
        // out whether `third` exists. It is NOT a fact about the file.
        //
        // So the index must announce that it stopped, or a caller reading
        // `Absent` concludes "this model declares no third key" from a scan
        // that never reached it — a negative finding drawn from a truncated
        // walk. `index_complete()` is what separates the two.
        assert!(matches!(m.declaration("third"), Declaration::Absent));
        assert!(
            !m.index_complete(),
            "a walk that stopped early must say so, or Absent lies"
        );
        assert!(!report.is_empty(), "the unknown type must be reported");
    }

    #[test]
    fn two_files_each_report_their_own_unreadable_key() {
        // A draft of this crate held the Unreadable placeholder in a process
        // -wide `static OnceLock`. Within one file that is invisible — the
        // index stops at the first unknown type, so there is only ever one.
        // Across two files the first one parsed wins forever, and the second
        // file's operator is shown a key name from someone else's model.
        let a = gguf(&[("alpha", 13, s("x"))]);
        let b = gguf(&[("beta", 13, s("x"))]);
        let (ma, _) = GgufMetadata::parse(&a, "a.gguf").unwrap();
        let (mb, _) = GgufMetadata::parse(&b, "b.gguf").unwrap();

        for (m, want_key, want_origin) in [(&ma, "alpha", "a.gguf"), (&mb, "beta", "b.gguf")] {
            match m.declaration(want_key) {
                Declaration::Unreadable(u) => {
                    assert_eq!(u.origin, want_origin);
                    match &u.kind {
                        mlmf_core::UnrecognizedKind::MetadataKey { key, .. } => {
                            assert_eq!(key, want_key, "each file must name its own key");
                        }
                        other => panic!("wrong kind: {other:?}"),
                    }
                }
                other => panic!("{want_key}: expected Unreadable, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_clean_parse_reports_a_complete_index() {
        // The other half of the pair. Without this, `index_complete` could
        // return `false` unconditionally and the test above would still
        // pass — a flag that is always pessimistic is as useless as one
        // that is always optimistic, and only asserting both directions
        // distinguishes them.
        let bytes = gguf(&[("a", 8, s("x")), ("b", 8, s("y"))]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        assert!(m.index_complete());
        assert!(report.is_empty());
        assert_eq!(m.keys().len(), 2);
    }

    #[test]
    fn a_duplicate_key_keeps_the_first_and_reports_the_second() {
        // GGUF does not forbid it and llama.cpp does not check. Silently
        // taking the last would make the file's meaning depend on parse
        // order; taking the first and reporting is deterministic and loud.
        let bytes = gguf(&[("k", 8, s("first")), ("k", 8, s("second"))]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("parses");
        assert_eq!(m.get("k"), Some(&MetaValue::String("first".into())));
        assert!(!report.is_empty(), "the duplicate must be reported");
    }

    #[test]
    fn alignment_defaults_to_32_and_is_overridden_only_by_a_u32_power_of_two() {
        let plain = gguf(&[("general.architecture", 8, s("llama"))]);
        let (m, _) = GgufMetadata::parse(&plain, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32, "no key declared: the documented default");

        let declared = gguf(&[("general.alignment", 4, 64u32.to_le_bytes().to_vec())]);
        let (m, _) = GgufMetadata::parse(&declared, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 64);

        // Not a power of two: llama.cpp refuses (gguf.cpp:623). Report and
        // fall back rather than fail the open — this is metadata, and R1
        // says metadata reading survives.
        let bad = gguf(&[("general.alignment", 4, 63u32.to_le_bytes().to_vec())]);
        let (m, report) = GgufMetadata::parse(&bad, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32);
        assert!(!report.is_empty());

        // Wrong type: llama.cpp requires UINT32 (gguf.cpp:614).
        let wrong = gguf(&[("general.alignment", 10, 64u64.to_le_bytes().to_vec())]);
        let (m, report) = GgufMetadata::parse(&wrong, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32);
        assert!(!report.is_empty());
    }

    #[test]
    fn declaration_reports_absent_for_a_key_the_file_does_not_have() {
        let bytes = gguf(&[("k", 8, s("v"))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        assert!(matches!(m.declaration("k"), Declaration::Declared(_)));
        assert!(matches!(m.declaration("absent"), Declaration::Absent));
    }

    #[test]
    fn a_decoded_value_is_returned_by_reference_and_decoded_once() {
        // The lazy index must still satisfy `get(&self) -> Option<&MetaValue>`,
        // which means the decoded value has to be cached. Two calls must
        // return the same address, or the cache is not doing its job.
        let bytes = gguf(&[("k", 8, s("v"))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        let a = m.get("k").unwrap() as *const MetaValue;
        let b = m.get("k").unwrap() as *const MetaValue;
        assert_eq!(a, b, "the value must be decoded once and cached");
    }

    #[test]
    fn the_kv_end_is_where_the_tensor_directory_begins() {
        // The next plan needs this, and it is only knowable here.
        let bytes = gguf(&[("k", 8, s("v"))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        // 24 header + 8 keylen + 1 key + 4 type + 8 strlen + 1 str
        assert_eq!(m.kv_end(), 46);
    }

    #[test]
    fn a_truncated_kv_block_reports_the_metadata_stage_not_the_header() {
        let mut bytes = gguf(&[("k", 8, s("v"))]);
        bytes.truncate(bytes.len() - 1);
        let err = GgufMetadata::parse(&bytes, "t.gguf").unwrap_err();
        assert!(matches!(
            err,
            crate::error::GgufError::Truncated {
                stage: crate::error::Stage::Metadata,
                ..
            }
        ));
    }
}
