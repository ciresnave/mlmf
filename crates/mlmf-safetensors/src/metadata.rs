//! The third stage: `__metadata__`, projected onto [`MetadataSource`].
//!
//! Its own module rather than a block in `lib.rs`, matching `dtype`,
//! `error`, `header` and `tensors` — and matching `mlmf-gguf`, which splits
//! `metadata.rs` from `tensors.rs` for the same reason: the two stages are
//! separate by construction, so a caller who only wants `__metadata__`
//! cannot be failed by the tensor directory.

use std::collections::BTreeMap;

use mlmf_core::{Declaration, MetaValue, MetadataSource, Report, Unrecognized, UnrecognizedKind};

use crate::header::{Header, json_kind};
use crate::tensors::METADATA_KEY;

/// One `__metadata__` key: the value when this build could read it, and the
/// complaint when it could not.
///
/// Both, rather than an `Option<MetaValue>` alone, because
/// [`Declaration`] has three states and the middle one — declared, and not
/// decodable — needs something to point at. A key with neither is not
/// stored at all; that is [`Declaration::Absent`].
#[derive(Debug, Clone)]
struct Entry {
    value: Option<MetaValue>,
    unreadable: Option<Unrecognized>,
}

/// Safetensors' `__metadata__` block, projected onto [`MetadataSource`].
///
/// # Every value is a [`MetaValue::String`], and that is the ruling
///
/// Safetensors' `__metadata__` is `string -> string`. There is no type tag
/// anywhere in it, so **thirteen of `MetaValue`'s fourteen variants can
/// never come out of this type**, and a value that reads as `"32"` is the
/// string `"32"` — [`MetaValue::as_u64`] on it answers `None`.
///
/// Fourteen, not thirteen: thirteen is GGUF's value-type count, and
/// `MetaValue` is those thirteen plus `MetaValue::Bytes`. This paragraph
/// said twelve-of-thirteen while the one below said `Bytes` is "ALSO
/// unreachable" — twelve plus `Bytes` plus `String` is fourteen, so the two
/// contradicted each other sixteen lines apart.
///
/// That is [`MetadataSource::get`]'s contract and not this crate's
/// preference: a `MetaValue`'s variant reports **how the format declared
/// the value, not what the value means**. Deciding that `"32"` is thirty-two
/// is format knowledge, and a format that did not declare a number did not
/// declare one. A `None` from an accessor here means "this format did not
/// declare that kind of value", which is a fact about the file.
///
/// The ruling predates this type by two plans and **had never been able to
/// fail**: GGUF declares typed values, so with one backend nothing
/// distinguished "reports the declaration" from "reports the meaning". This
/// is the first code where the two give different answers.
///
/// [`MetaValue::Bytes`] is one of those thirteen, and it is unreachable for
/// a reason of its own rather than for the string-to-string one:
/// the block arrives through `serde_json`, which only produces `String`s
/// that are already valid UTF-8. The non-UTF-8 case `MetaValue::Bytes`
/// exists for is a GGUF fact — `gguf_string_t` is a length and raw bytes.
///
/// # Key order is lexicographic, and that is per-format
///
/// [`MetadataSource::keys`] documents its order as unspecified.
/// `serde_json` without `preserve_order` backs an object with a `BTreeMap`,
/// so this yields keys sorted by their UTF-8 bytes regardless of what the
/// header declared — the same measured divergence from `mlmf-gguf`'s
/// declaration order that [`crate::SafetensorsTensors`] has. Stated, not promised:
/// a caller holding `&dyn MetadataSource` may not rely on it.
#[derive(Debug, Clone)]
pub struct SafetensorsMetadata {
    entries: BTreeMap<String, Entry>,
    index_complete: bool,
}

/// Read the `__metadata__` block out of an already-parsed header.
///
/// A file with no `__metadata__` at all is **not an error and not rare** —
/// the key is optional and plenty of real files omit it. That yields a
/// source declaring nothing, with an empty [`Report`].
///
/// # Two things that are not the same, and this function keeps them apart
///
/// - A **value** that is not a JSON string is declared and not decodable.
///   The key stays in [`MetadataSource::keys`], [`MetadataSource::get`]
///   answers `None`, and [`MetadataSource::declaration`] answers
///   [`Declaration::Unreadable`]. Dropping it would make a declared key
///   look absent, which is the exact conflation `Declaration` exists to
///   prevent.
/// - A **`__metadata__` that is not an object** means no key could be
///   enumerated at all, so [`MetadataSource::index_complete`] answers
///   `false` — see this crate's derivation of that method below.
///
/// Neither is an error. Both cost exactly what they are about, and both are
/// named in the returned report, so nothing is dropped silently.
///
/// The report entries carry their explanation in `reason` and leave `value`
/// as `None`. A JSON number under `__metadata__` has no honest `MetaValue`
/// — turning `5` into `MetaValue::U64(5)` would be this crate deciding the
/// file declared a number in a block defined to hold strings, and putting
/// the sentence in `value` would repeat the defect `MetadataKey` was
/// repaired for.
#[must_use]
pub fn parse_metadata(header: &Header, origin: &str) -> (SafetensorsMetadata, Report) {
    let mut report = Report::new();
    let mut entries = BTreeMap::new();

    let Some(block) = header.entries.get(METADATA_KEY) else {
        // Absent, which is a fact about the file rather than a failure to
        // read one. The index is complete: there were no keys to miss.
        return (
            SafetensorsMetadata {
                entries,
                index_complete: true,
            },
            report,
        );
    };

    let serde_json::Value::Object(map) = block else {
        report.push(Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: METADATA_KEY.to_string(),
                value: None,
                reason: Some(format!(
                    "__metadata__ must be an object mapping strings to strings; \
                     this is a {}, so no metadata key could be enumerated",
                    json_kind(block)
                )),
            },
            origin: origin.to_string(),
        });
        return (
            SafetensorsMetadata {
                entries,
                index_complete: false,
            },
            report,
        );
    };

    for (key, value) in map {
        let serde_json::Value::String(text) = value else {
            let complaint = Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: key.clone(),
                    value: None,
                    reason: Some(format!(
                        "__metadata__ values are strings; this one is a {}",
                        json_kind(value)
                    )),
                },
                origin: origin.to_string(),
            };
            report.push(complaint.clone());
            // Kept in the map, not skipped. The key IS declared; only its
            // value is unreadable, and those are different answers.
            entries.insert(
                key.clone(),
                Entry {
                    value: None,
                    unreadable: Some(complaint),
                },
            );
            continue;
        };
        entries.insert(
            key.clone(),
            Entry {
                // `String`, always. Never parsed, never sniffed, never
                // widened — see this type's doc.
                value: Some(MetaValue::String(text.clone())),
                unreadable: None,
            },
        );
    }

    (
        SafetensorsMetadata {
            entries,
            index_complete: true,
        },
        report,
    )
}

impl MetadataSource for SafetensorsMetadata {
    /// **`true`, and derived rather than copied from `mlmf-gguf`.**
    ///
    /// [`crate::parse_header`] hands the whole header to `serde_json::from_str`,
    /// which either returns the complete top-level object or returns an
    /// error. There is no forward walk, nothing is skipped, and no value's
    /// width is unknown — so if a header parsed at all, every
    /// `__metadata__` key it declares is in hand.
    ///
    /// That makes a negative answer from this source a **positive fact
    /// about the file**, which is stronger than `mlmf-gguf` can offer:
    /// GGUF's key-value block is walked sequentially, so a value type that
    /// build does not know has an unknown width and the walk cannot find
    /// the key after it — it answers `false`, and its `Absent` then means
    /// only "not found in the part that could be read".
    ///
    /// **The one case that answers `false`** is a `__metadata__` present
    /// and not an object. Then no key could be enumerated, so this source
    /// cannot tell "the file declares no chat template" from "the block
    /// that would have declared one was unreadable" — and
    /// [`Declaration::Absent`]'s own doc says those are different claims
    /// and only the first is safe to act on.
    ///
    /// Note what this method is and is not about. It is about **enumerating
    /// keys**. A key whose VALUE could not be decoded does not make the
    /// index incomplete: that key was seen, it is in [`Self::keys`], and
    /// [`Self::declaration`] reports it as [`Declaration::Unreadable`].
    fn index_complete(&self) -> bool {
        self.index_complete
    }

    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entries.get(key).and_then(|e| e.value.as_ref())
    }

    /// Every declared key, including one whose value could not be decoded.
    ///
    /// Declared is declared. Omitting an undecodable key here would make
    /// the only complete list of what the file says disagree with
    /// [`Self::declaration`], which reports that same key as
    /// [`Declaration::Unreadable`]. `mlmf-gguf` answers the same way.
    ///
    /// Order is lexicographic — see this type's doc — and the trait does
    /// not promise one.
    fn keys(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    /// Three states, because two of them are otherwise indistinguishable.
    ///
    /// The trait's default can only ever answer `Declared` or `Absent`,
    /// which is right for a source whose every value is already a
    /// `MetaValue`. This one has a third case — a `__metadata__` value that
    /// is not a JSON string — so it overrides, and the entry it borrows
    /// names *this* key.
    fn declaration(&self, key: &str) -> Declaration<'_> {
        match self.entries.get(key) {
            None => Declaration::Absent,
            Some(e) => match (&e.value, &e.unreadable) {
                (Some(v), _) => Declaration::Declared(v),
                (None, Some(u)) => Declaration::Unreadable(u),
                // Not constructible: `parse_metadata` writes exactly one of
                // the two on every entry it inserts. Answered as `Absent`
                // rather than with a panic, because a report entry that
                // cannot be produced is not worth a crash in a reader.
                (None, None) => Declaration::Absent,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_header, parse_tensors};
    use mlmf_core::TensorContainer;

    const ORIGIN: &str = "model.safetensors";

    /// `__metadata__` carrying four values, three of which are a temptation
    /// to convert: a numeric-looking one, a boolean-looking one, and one
    /// holding a JSON document. Plus two real tensor records, because
    /// safetensors mixes `__metadata__` and tensors at the SAME level of
    /// one object and that is the trap `keys()` has to not fall into.
    const RICH: &[u8] = br#"{"__metadata__":{"cfg":"{\"a\":1}","format":"pt","tied":"true","total_size":"32"},"bias":{"dtype":"BF16","shape":[2,3],"data_offsets":[12,24]},"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;

    /// No `__metadata__` at all. Not a degenerate fixture: the key is
    /// optional and real files ship without it.
    const NO_METADATA: &[u8] =
        br#"{"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;

    /// A `__metadata__` value that is not a string. Well-formed JSON, and
    /// not what the format defines.
    const NON_STRING_VALUE: &[u8] = br#"{"__metadata__":{"format":"pt","n":5}}"#;

    /// `__metadata__` present and not an object at all.
    const NOT_AN_OBJECT: &[u8] =
        br#"{"__metadata__":5,"weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;

    /// A safetensors image: a truthful 8-byte length prefix, `json`, then
    /// `data_len` zero bytes.
    fn image(json: &[u8], data_len: usize) -> Vec<u8> {
        let mut v = (json.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(json);
        v.extend(std::iter::repeat_n(0u8, data_len));
        v
    }

    /// Both stages exactly as a consumer runs them.
    fn metadata(image: &[u8]) -> (SafetensorsMetadata, Report) {
        let header = parse_header(image).expect("the fixtures are well-formed headers");
        parse_metadata(&header, ORIGIN)
    }

    /// Every key with the value `get` answers, owned so it compares whole.
    fn pairs(m: &SafetensorsMetadata) -> Vec<(String, Option<MetaValue>)> {
        m.keys()
            .into_iter()
            .map(|k| (k.to_string(), m.get(k).cloned()))
            .collect()
    }

    // ---- the ruling ----------------------------------------------------

    #[test]
    fn every_value_arrives_as_a_string_whatever_it_looks_like() {
        // Three different temptations in one fixture rather than one
        // asserted once: `"32"` invites a number, `"true"` invites a
        // boolean, and `"{\"a\":1}"` invites a recursive parse into
        // `MetaValue::Array` or a nested map. All three are strings,
        // because the format declared strings.
        //
        // The whole key/value list in one comparison — a per-key assertion
        // chain cannot see two values swapped between keys, and swapping
        // `total_size` with `tied` is exactly the shape of that bug.
        //
        // Order is lexicographic, which is `serde_json`'s BTreeMap and not
        // a promise of the trait: `cfg` before `format` before `tied`
        // before `total_size`, while the fixture happens to declare them in
        // that order too.
        assert_eq!(
            pairs(&metadata(&image(RICH, 24)).0),
            vec![
                (
                    "cfg".to_string(),
                    Some(MetaValue::String("{\"a\":1}".into()))
                ),
                ("format".to_string(), Some(MetaValue::String("pt".into()))),
                ("tied".to_string(), Some(MetaValue::String("true".into()))),
                (
                    "total_size".to_string(),
                    Some(MetaValue::String("32".into()))
                ),
            ]
        );
    }

    #[test]
    fn a_numeric_looking_value_is_a_string_and_the_accessors_refuse_to_guess() {
        // THE test this task exists for, and the ruling it pins has never
        // been able to fail before: GGUF declares typed values, so with one
        // backend nothing distinguished "a MetaValue reports how the format
        // DECLARED a value" from "a MetaValue reports what the value
        // means". Here the two give different answers, and this asserts the
        // first.
        //
        // "32 is obviously a number" is the feeling this project treats as
        // a risk signal rather than a safety one. The file declared five
        // characters; `as_u64` answering `Some(32)` would be this crate
        // deciding what they mean, at a layer that does not know which
        // format it is reading.
        //
        // All six answers in one comparison, so a sabotage that makes
        // `as_u64` parse cannot hide behind an earlier assertion panicking.
        let (m, report) = metadata(&image(RICH, 24));
        let numeric = m.get("total_size").expect("declared");
        let boolean = m.get("tied").expect("declared");
        assert_eq!(
            (
                numeric,
                numeric.as_u64(),
                numeric.as_i64(),
                numeric.as_f64(),
                boolean,
                boolean.as_bool(),
            ),
            (
                &MetaValue::String("32".into()),
                None,
                None,
                None,
                &MetaValue::String("true".into()),
                None,
            )
        );
        // And nothing about any of that is a finding: a string-valued
        // format declaring strings is the format working.
        assert_eq!(report, Report::new());
    }

    // ---- the trap: one object holds metadata AND tensors ----------------

    #[test]
    fn keys_are_metadata_keys_and_the_tensor_names_come_out_of_the_other_view() {
        // Safetensors puts `__metadata__` and every tensor record in ONE
        // top-level object, so "the header's keys" and "the metadata keys"
        // are different sets that a careless implementation returns
        // interchangeably.
        //
        // Asserted from ONE header through BOTH seams, because that is the
        // only shape that can see a leak in either direction: a tensor name
        // appearing in `keys()`, or `__metadata__` appearing in
        // `tensors()`. Two separate tests over two fixtures could each pass
        // while the pair was wrong.
        let bytes = image(RICH, 24);
        let header = parse_header(&bytes).expect("well-formed");
        let (m, _) = parse_metadata(&header, ORIGIN);
        let (t, _) = parse_tensors(&bytes, &header, ORIGIN).expect("well-formed");

        assert_eq!(
            (
                m.keys(),
                t.tensors()
                    .iter()
                    .map(|d| d.name.as_str())
                    .collect::<Vec<_>>(),
            ),
            (
                vec!["cfg", "format", "tied", "total_size"],
                vec!["bias", "weight"],
            )
        );
        // The two sets are disjoint, and neither contains the container key
        // itself. `__metadata__` names the block; it is not a key IN it.
        assert_eq!((m.get("weight"), m.get(METADATA_KEY)), (None, None));
    }

    // ---- three states, not two ------------------------------------------

    #[test]
    fn declaration_separates_absent_from_declared() {
        let (m, _) = metadata(&image(RICH, 24));
        let pt = MetaValue::String("pt".into());
        assert_eq!(
            (m.declaration("format"), m.declaration("chat_template")),
            (Declaration::Declared(&pt), Declaration::Absent)
        );
    }

    #[test]
    fn a_value_that_is_not_a_string_is_declared_and_unreadable_never_absent() {
        // The middle state. A key whose value this build cannot decode is
        // NOT absent, and reporting it as absent would tell an operator the
        // file says nothing where in fact the file says something
        // unreadable — the exact conflation `Declaration` exists to
        // prevent, and the one a two-state answer cannot avoid.
        let (m, report) = metadata(&image(NON_STRING_VALUE, 0));
        let complaint = Unrecognized {
            kind: UnrecognizedKind::MetadataKey {
                key: "n".into(),
                value: None,
                reason: Some("__metadata__ values are strings; this one is a number".into()),
            },
            origin: ORIGIN.into(),
        };
        let pt = MetaValue::String("pt".into());

        // Five facts in one comparison, because it is their COMBINATION
        // that is the contract: the key is listed, `get` declines it, the
        // declaration names the complaint, the report carries the same
        // complaint, and the index is still complete — one undecodable
        // VALUE does not mean a KEY was missed.
        assert_eq!(
            (
                m.keys(),
                m.get("n"),
                m.declaration("n"),
                m.declaration("format"),
                report.entries(),
                m.index_complete(),
            ),
            (
                vec!["format", "n"],
                None,
                Declaration::Unreadable(&complaint),
                Declaration::Declared(&pt),
                [complaint.clone()].as_slice(),
                true,
            )
        );
    }

    // ---- absence, which is a real case ----------------------------------

    #[test]
    fn a_file_with_no_metadata_block_declares_nothing_and_is_not_an_error() {
        // Plenty of real files carry no `__metadata__`. An empty source and
        // an EMPTY REPORT: there is nothing unrecognised about a file that
        // omits an optional key, and reporting one would train a consumer
        // to ignore the report.
        let (m, report) = metadata(&image(NO_METADATA, 12));
        assert_eq!(
            (
                m.keys(),
                m.get("format"),
                m.declaration("format"),
                report,
                m.index_complete(),
            ),
            (Vec::new(), None, Declaration::Absent, Report::new(), true)
        );
    }

    // ---- index_complete, derived rather than assumed ---------------------

    #[test]
    fn the_index_is_complete_because_the_header_parses_whole_or_not_at_all() {
        // `parse_header` hands the entire header to `serde_json::from_str`,
        // which returns the complete object or an error. Nothing is walked
        // forward, nothing is skipped for unknown width — so every key the
        // file declares is in hand, and `Absent` from this source is a
        // POSITIVE FACT about the file rather than "not found in the part
        // that could be read".
        //
        // Stronger than `mlmf-gguf` can offer, which is why this is derived
        // here and not copied from there: GGUF walks its key-value block
        // sequentially and a value type it does not know has unknown width,
        // so its walk can stop early and its `Absent` is weaker.
        //
        // Asserted across three shapes so the answer is not accidentally
        // true of one fixture: metadata present, metadata absent, and a
        // value that could not be decoded.
        assert_eq!(
            (
                metadata(&image(RICH, 24)).0.index_complete(),
                metadata(&image(NO_METADATA, 12)).0.index_complete(),
                metadata(&image(NON_STRING_VALUE, 0)).0.index_complete(),
            ),
            (true, true, true)
        );
    }

    #[test]
    fn a_metadata_block_that_is_not_an_object_makes_every_absent_unsafe() {
        // The one case that answers `false`, and the reason the method
        // cannot just return a constant. No key could be enumerated, so
        // this source cannot tell "the file declares no chat template" from
        // "the block that would have declared one was unreadable" —
        // `Declaration::Absent`'s own doc says those are different claims
        // and only the first is safe to act on.
        //
        // The whole report entry, not a count: it must name `__metadata__`
        // and say what was found, because an operator holding it is about
        // to open the JSON and look.
        let (m, report) = metadata(&image(NOT_AN_OBJECT, 12));
        assert_eq!(
            (
                m.index_complete(),
                m.keys(),
                m.declaration("format"),
                report.entries(),
            ),
            (
                false,
                Vec::new(),
                Declaration::Absent,
                [Unrecognized {
                    kind: UnrecognizedKind::MetadataKey {
                        key: "__metadata__".into(),
                        value: None,
                        reason: Some(
                            "__metadata__ must be an object mapping strings to strings; \
                             this is a number, so no metadata key could be enumerated"
                                .into()
                        ),
                    },
                    origin: ORIGIN.into(),
                }]
                .as_slice(),
            )
        );
    }

    // ---- the seam, reached the way a consumer reaches it -----------------

    #[test]
    fn the_whole_thing_answers_through_dyn_metadata_source() {
        // Every test above holds a concrete type, and every consumer this
        // seam exists for holds `&dyn MetadataSource`. The two differ in a
        // way that matters here: an inherent method with the same name as a
        // trait method would satisfy every test above and be invisible
        // through the trait object, so the seam could be unimplemented in
        // the only shape that is used.
        //
        // Task 5 asserts across two backends through this same view.
        let (owned, _) = metadata(&image(RICH, 24));
        let m: &dyn MetadataSource = &owned;
        let numeric = m.get("total_size").expect("declared");
        assert_eq!(
            (
                m.index_complete(),
                m.keys(),
                numeric,
                numeric.as_u64(),
                m.array_len("total_size"),
            ),
            (
                true,
                vec!["cfg", "format", "tied", "total_size"],
                &MetaValue::String("32".into()),
                None,
                // A string is not an array of one. The trait's default
                // answers `None` for anything that is not `MetaValue::Array`
                // — reporting 1 would let a caller index a string as though
                // it were a vocabulary.
                None,
            )
        );
    }
}
