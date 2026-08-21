//! The key-value block, indexed at open and decoded on demand.

use std::sync::OnceLock;

use mlmf_core::{Declaration, MetaValue, MetadataSource, Report, Unrecognized, UnrecognizedKind};

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};
use crate::header::{Header, parse_header};
use crate::value::{ValueType, decode_value, read_array_prefix, skip_value};

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
    /// fact from a malformed GGUF (R7). [`GgufError::ByteSwapped`] if the
    /// magic matches but the version reads byte-reversed, and
    /// [`GgufError::UnsupportedVersion`] if it is a GGUF this build does
    /// not read — both reach a caller through this function, so both
    /// belong here. [`GgufError::Truncated`] or
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
                        // The code IS the finding here: an unrecognised
                        // value type has nothing else to say about itself.
                        value: Some(MetaValue::U32(code)),
                        reason: None,
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
                        // Deliberately not captured: decoding the value to
                        // say "this appeared twice" would cost roughly
                        // 26 MB for the corpus's largest single key.
                        value: None,
                        reason: Some(
                            "declared more than once; the first occurrence is kept".into(),
                        ),
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
        // Captured HERE, before the `u64::from` below shadows the `u32`
        // binding. A report built after that conversion would name a type
        // the file did not declare — the same lie in a new field.
        let declared = v.clone();
        let complain = |report: &mut Report, declared: MetaValue, why: &str| {
            report.push(Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "general.alignment".to_string(),
                    value: Some(declared),
                    reason: Some(why.to_string()),
                },
                origin: origin.to_string(),
            });
        };
        // llama.cpp requires UINT32 specifically (gguf.cpp:614).
        let MetaValue::U32(a) = v else {
            complain(report, declared, "must be UINT32; using the default of 32");
            return DEFAULT_ALIGNMENT;
        };
        let a = u64::from(*a);
        // And a power of two (gguf.cpp:623). No separate zero check: a
        // leading `a == 0 ||` stood here and is a no-op, because
        // `0u64.is_power_of_two()` is already false. Measured — deleting it
        // left every test green, including the authored alignment-0 case.
        // A clause that reads as a guard and guards nothing is worse than
        // none, because the next reader budgets trust for it.
        if !a.is_power_of_two() {
            complain(
                report,
                declared,
                "must be a power of two; using the default of 32",
            );
            return DEFAULT_ALIGNMENT;
        }
        a
    }

    fn entry(&self, key: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.key == key)
    }

    /// Position a cursor on an array's first element.
    ///
    /// Returns the element type, the declared count, and a cursor sitting
    /// at element 0. `None` when the key is absent, unreadable, or not an
    /// array.
    ///
    /// The prefix — element type code, then count — is read by
    /// `value::read_array_prefix`, the function `skip_value` and
    /// `decode_value` already use, instead of being re-read here. Task 5
    /// cost a day to the case where two functions walked one grammar and
    /// drifted apart; `array_len` and `array_get` would have been the third
    /// and fourth walkers of this particular grammar, so they go through
    /// the same door instead of each opening their own.
    fn array_at(&self, key: &str) -> Option<(ValueType, u64, Cursor<'a>)> {
        // THREE of the five early-outs below are defence in depth, not
        // coverage — each was mutated in isolation and the suite stayed
        // green, because none can be reached. `unreadable` is shadowed by
        // the type check: the only site that sets it hardcodes
        // `ty: ValueType::U8`, so an unreadable entry is never an Array.
        // `seek` cannot fail on an offset the parse walk produced, and
        // `read_array_prefix` cannot fail on a prefix `skip_value` already
        // accepted. They stay because the alternative is `unwrap`, and a
        // reachability argument is a worse thing to bet a panic on than a
        // `?`.
        //
        // The other TWO are live: `entry(key)?` and the `ty != Array`
        // guard. An earlier version of this comment said four and named
        // only the type check as live, which was wrong in the direction
        // that matters — replacing `entry(key)` with "find the first array
        // in this file" left all 63 tests green while `array_len` answered
        // `Some(3)` for a key that does not exist and `array_get` handed
        // back a plausible token from a different key. That is R4's
        // absent-vocabulary failure mode answering as though the array
        // were present. Both live guards are pinned in
        // `array_accessors_agree_with_the_decoded_value`.
        let e = self.entry(key)?;
        if e.ty != ValueType::Array || e.unreadable.is_some() {
            return None;
        }
        let mut c = Cursor::new(self.bytes);
        c.seek(e.start).ok()?;
        let (elem, count) = read_array_prefix(&mut c).ok()?;
        Some((elem, count, c))
    }

    /// Decode an entry's value, caching it.
    fn value_of<'e>(&self, e: &'e Entry) -> Option<&'e MetaValue> {
        if e.unreadable.is_some() {
            return None;
        }
        Some(e.value.get_or_init(|| {
            let mut c = Cursor::new(self.bytes);
            // Every *structural* check `decode_value` makes was already made
            // by `skip_value` at this offset during indexing, against these
            // same bytes — which a shared borrow cannot have changed. So no
            // truncation, unknown type code, depth or overflow failure can
            // appear here that did not appear then.
            //
            // One check is not structural. The array branch calls
            // `Vec::try_reserve`, and the allocator's answer on first access
            // is not the answer it gave at parse time, because parse never
            // allocated at all. Under genuine memory exhaustion this
            // `expect` can fire. That is not a new exposure: the string and
            // nested-vector allocations inside `decode_value` abort the
            // process on the same condition without passing through a
            // `Result` at any point.
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
    /// Delegates to the inherent [`GgufMetadata::index_complete`], whose
    /// doc carries the full argument. Present on the trait as well because
    /// a consumer holding `&dyn MetadataSource` is precisely the one who
    /// cannot reach an inherent method, and is precisely the one whose
    /// `Absent` results are worthless without it.
    fn index_complete(&self) -> bool {
        self.index_complete
    }

    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.entry(key).and_then(|e| self.value_of(e))
    }

    /// The declared keys, in the order the file declares them.
    ///
    /// [`MetadataSource::keys`] documents its order as unspecified, and it
    /// stays unspecified for backends that cannot cheaply do better. This
    /// implementation promises more because it costs nothing to: the index
    /// is one forward walk, so file order is simply what the walk produces.
    ///
    /// A caller holding a concrete `GgufMetadata` may rely on that. A
    /// caller holding `&dyn MetadataSource` may not, and should not.
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

    /// The number of elements the array DECLARES.
    ///
    /// Declared, not counted — nothing re-walks the elements here. The
    /// number is trustworthy for a different reason: indexing already ran
    /// `skip_value` over this value, which walks a variable-width array
    /// element by element and bounds a fixed-width one by
    /// `count * width <= bytes remaining`, so a count larger than the file
    /// can hold fails the OPEN. A file whose array claims nine elements and
    /// carries five is refused with `truncated in the metadata`, and never
    /// reaches this function.
    ///
    /// An understating count is not refused and is not a hazard: this
    /// under-reports and [`Self::array_get`]'s bound stays conservative.
    fn array_len(&self, key: &str) -> Option<u64> {
        self.array_at(key).map(|(_, count, _)| count)
    }

    /// Decode one element without materializing the array.
    ///
    /// **Cost, because a caller cannot guess it from the signature:**
    ///
    /// - Fixed-width elements: the offset is arithmetic, so one seek.
    /// - `String` elements: `O(index)` — each skip reads a length and
    ///   advances, so the walk is proportional to the index alone.
    /// - Nested `Array` elements: `O(elements walked before index)`, which
    ///   is NOT `O(index)`. Skipping one nested element costs that
    ///   element's whole subtree, so the total is the sum of the inner
    ///   counts before `index` and is unbounded in `index`.
    ///
    /// Every call also pays a linear scan of the key list to find the
    /// entry — immaterial at the 42 keys real files carry, but it means a
    /// loop over one array is `O(n * keys)` before the walking cost.
    ///
    /// So a loop calling this at every index of a 500,000-element token
    /// list is quadratic and will not finish. This method is for reading a
    /// few elements out of many. A caller who wants the whole array should
    /// call `get` once and pay for it once.
    fn array_get(&self, key: &str, index: u64) -> Option<MetaValue> {
        let (elem, count, mut c) = self.array_at(key)?;
        if index >= count {
            return None;
        }
        match elem.fixed_width() {
            // Constant time: the element's offset is arithmetic.
            Some(w) => {
                let skip = index.checked_mul(w)?;
                c.seek(c.pos().checked_add(skip)?).ok()?;
            }
            // Variable width: walk, but skip rather than decode. Still O(n)
            // in the index, and O(1) in allocations — which is the cost that
            // actually hurt.
            None => {
                for _ in 0..index {
                    skip_value(&mut c, elem).ok()?;
                }
            }
        }
        // Nesting depth restarts at 0 here, so a nested element is walked
        // with the full budget again rather than the remainder indexing had
        // left. This can never REJECT a subtree the index accepted, which
        // is the direction that matters: a key that survived the open must
        // stay readable.
        //
        // The reset is not what makes that true, and it would be wrong to
        // credit it. Safety comes from Task 5's `depth + 1` accounting in
        // `skip_at_depth`: indexing already refuses anything the decode
        // could refuse. A decode that CONTINUED from indexing's depth would
        // be equally safe. What the reset buys is slack — ONE level,
        // measured by shifting the decode's start depth: `nest(64)`
        // tolerates a shift of 1 and breaks at 2 — and slack is what hides
        // an accounting error rather
        // than what prevents one. `deep_nesting_is_readable_by_both_paths`
        // sits at exactly the depth where that slack runs out.
        decode_value(&mut c, elem).ok()
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

    /// Encode an array: element type code, count, then the element bytes.
    ///
    /// One builder for every array fixture in this file. The nested cases
    /// are built by feeding this function its own output, which is the only
    /// way a nested fixture is guaranteed to agree with a flat one.
    fn array_of(elem_code: u32, items: &[Vec<u8>]) -> Vec<u8> {
        let mut b = elem_code.to_le_bytes().to_vec();
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for i in items {
            b.extend_from_slice(i);
        }
        b
    }

    fn str_array(items: &[&str]) -> Vec<u8> {
        array_of(8, &items.iter().map(|i| s(i)).collect::<Vec<_>>())
    }

    /// `depth` levels of single-element array nesting around a string leaf.
    fn nest(depth: u32) -> Vec<u8> {
        let mut v = s("leaf");
        let mut code = 8u32; // the leaf is a String; every wrap above is an Array
        for _ in 0..depth {
            v = array_of(code, &[v]);
            code = 9;
        }
        v
    }

    /// A key that follows an array, so an out-of-range read has real bytes
    /// to land on instead of the end of the buffer.
    ///
    /// Without one, an out-of-range assertion proves `Cursor`'s bounds
    /// check rather than `array_get`'s: the array is last, the read hits
    /// EOF, and `None` comes back for the wrong reason. Measured, twice, in
    /// two different fixtures in this file.
    fn trailer() -> (&'static str, u32, Vec<u8>) {
        ("after", 4, 0xDEAD_BEEFu32.to_le_bytes().to_vec())
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
        assert_eq!(m.keys(), ["general.architecture", "tokenizer.ggml.tokens"]);
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
        // The WHOLE entry, not merely that the report is non-empty. This is
        // the one `MetadataKey` call site Task 10 left unasserted: swapping
        // it to `value: None, reason: Some(...)` — precisely the defect
        // Task 10 removed — left all 63 tests green. The doc on `reason`
        // singles this out as the `None` case because here the code IS the
        // finding, and that claim was unverified until this line.
        assert_eq!(
            report.entries(),
            [Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "broken".to_string(),
                    value: Some(MetaValue::U32(13)),
                    reason: None,
                },
                origin: "t.gguf".to_string(),
            }]
        );
        // `get` must refuse it too, not only `declaration`. Deleting
        // `value_of`'s `unreadable` guard left all 63 tests green while
        // `get("broken")` returned `Some(U8(71))` — 0x47, the `G` of the
        // file's own magic, read from the placeholder entry's `start: 0`
        // with its placeholder `ty: ValueType::U8`. Every existing test
        // reached this key through `declaration()`, which checks
        // `unreadable` first and so shadowed this path entirely.
        assert_eq!(
            m.get("broken"),
            None,
            "an unreadable key must not answer get() with fabricated bytes"
        );

        // The WHOLE key set, not membership of three chosen names.
        //
        // Every assertion above is satisfied by a parse that desynchronises
        // instead of stopping: changing `break` to `continue` leaves the
        // cursor inside the unreadable value, and the next read lands on
        // its tail — here producing a phantom key "unreachable" that is
        // valid UTF-8, carrying a garbage `I32(0)`. `first` still decodes,
        // `broken` is still Unreadable, `third` is still Absent, the index
        // still reports incomplete, the report is still non-empty. Five
        // assertions, none of which can see an invented key.
        //
        // Asserting the set closes it, and the principle is the collection
        // form of this crate's whole-value rule: check what IS there, not
        // that the things you thought of are among it.
        assert_eq!(m.keys(), ["first", "broken"]);
    }

    #[test]
    fn the_absent_caveat_reaches_a_consumer_through_the_trait_object() {
        // `index_complete` began as an inherent method on GgufMetadata, and
        // the whole seam is `&dyn MetadataSource` — so the one caller who
        // most needs it, the format-agnostic layer holding a trait object,
        // was the one caller who could not reach it. It got `Absent` with
        // no way to ask whether the walk had finished, while the core doc
        // for `Absent` said "the key is not declared" without qualification.
        //
        // This asserts the pairing a consumer actually depends on: a
        // negative answer AND the flag that says whether to believe it,
        // both obtained without knowing the concrete type.
        let stopped = gguf(&[("first", 8, s("kept")), ("broken", 13, s("x"))]);
        let clean = gguf(&[("first", 8, s("kept"))]);
        let (a, _) = GgufMetadata::parse(&stopped, "t.gguf").unwrap();
        let (b, _) = GgufMetadata::parse(&clean, "t.gguf").unwrap();

        for (src, complete) in [
            (&a as &dyn MetadataSource, false),
            (&b as &dyn MetadataSource, true),
        ] {
            // Same question, same answer, from sources where it means
            // opposite things.
            assert!(matches!(src.declaration("absent"), Declaration::Absent));
            assert_eq!(src.index_complete(), complete);
        }
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
        assert_eq!(m.keys(), ["a", "b"]);
    }

    #[test]
    fn a_duplicate_key_keeps_the_first_and_reports_the_second() {
        // GGUF does not forbid it and llama.cpp does not check. Silently
        // taking the last would make the file's meaning depend on parse
        // order; taking the first and reporting is deterministic and loud.
        let bytes = gguf(&[("k", 8, s("first")), ("k", 8, s("second"))]);
        let (m, report) = GgufMetadata::parse(&bytes, "t.gguf").expect("parses");
        assert_eq!(m.get("k"), Some(&MetaValue::String("first".into())));
        // The WHOLE entry, not `!report.is_empty()`. Non-emptiness cannot
        // see the wrong key named, the reason missing, or the explanation
        // sentence sitting back in `value` — which is the exact defect the
        // `reason` field was added to end. `value` is `None` on purpose
        // here: reporting a duplicate must not cost a value decode.
        assert_eq!(
            report.entries(),
            &[Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "k".into(),
                    value: None,
                    reason: Some("declared more than once; the first occurrence is kept".into()),
                },
                origin: "t.gguf".into(),
            }],
            "the duplicate must be reported, as a reason and not as a value"
        );
        // Dropping the `continue` after the report is a one-line regression
        // that reads like a tidy-up, and every assertion above survives it:
        // the duplicate is still reported, and `get` still answers "first"
        // because `entry()` returns the first match either way. The only
        // visible damage is a key set of ["k", "k"] — an EXTRA element,
        // which is precisely what a membership assertion cannot see.
        assert_eq!(m.keys(), ["k"], "the second occurrence must not be indexed");
    }

    #[test]
    fn alignment_defaults_to_32_and_is_overridden_only_by_a_u32_power_of_two() {
        let plain = gguf(&[("general.architecture", 8, s("llama"))]);
        let (m, _) = GgufMetadata::parse(&plain, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32, "no key declared: the documented default");

        let declared = gguf(&[("general.alignment", 4, 64u32.to_le_bytes().to_vec())]);
        let (m, report) = GgufMetadata::parse(&declared, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 64);
        assert!(report.is_empty(), "a valid declaration is not a finding");

        // Not a power of two: llama.cpp refuses (gguf.cpp:623). Report and
        // fall back rather than fail the open — this is metadata, and R1
        // says metadata reading survives.
        //
        // The whole entry, because `!report.is_empty()` cannot see the
        // parser's substituted default of 32 standing in `value` where the
        // file's declared 63 belongs.
        let bad = gguf(&[("general.alignment", 4, 63u32.to_le_bytes().to_vec())]);
        let (m, report) = GgufMetadata::parse(&bad, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32);
        assert_eq!(
            report.entries(),
            &[Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "general.alignment".into(),
                    value: Some(MetaValue::U32(63)),
                    reason: Some("must be a power of two; using the default of 32".into()),
                },
                origin: "t.gguf".into(),
            }]
        );

        // Wrong type: llama.cpp requires UINT32 (gguf.cpp:614).
        //
        // `U64(64)` is the point of the whole task: the file declared type
        // code 10, and the report must say so. `resolve_alignment` reaches
        // this branch before the `u64::from` conversion exists, but the
        // sibling branch above runs after it, so a value captured late
        // there would report a U64 for a file that declared a U32.
        let wrong = gguf(&[("general.alignment", 10, 64u64.to_le_bytes().to_vec())]);
        let (m, report) = GgufMetadata::parse(&wrong, "t.gguf").unwrap();
        assert_eq!(m.alignment(), 32);
        assert_eq!(
            report.entries(),
            &[Unrecognized {
                kind: UnrecognizedKind::MetadataKey {
                    key: "general.alignment".into(),
                    value: Some(MetaValue::U64(64)),
                    reason: Some("must be UINT32; using the default of 32".into()),
                },
                origin: "t.gguf".into(),
            }]
        );
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
    fn array_access_does_not_decode_the_array() {
        // The R5 case. If this went through `get`, it would decode all
        // 100,000 elements to return one.
        let items: Vec<String> = (0..100_000).map(|i| format!("tok{i}")).collect();
        let refs: Vec<&str> = items.iter().map(String::as_str).collect();
        // The trailing key is load-bearing for the out-of-range assertion
        // below — see `trailer`. Without it, deleting `array_get`'s
        // `index >= count` bound left THIS test green while the fixed-width
        // test three down went red, and the value a caller would have
        // received is worse here: `Some(String("after"))`, the next key's
        // NAME handed back as a vocabulary token. A string that looks like
        // a real token, for an id past the end of the vocabulary.
        let bytes = gguf(&[("tokenizer.ggml.tokens", 9, str_array(&refs)), trailer()]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();

        assert_eq!(m.array_len("tokenizer.ggml.tokens"), Some(100_000));
        assert_eq!(
            m.array_get("tokenizer.ggml.tokens", 0),
            Some(MetaValue::String("tok0".into()))
        );
        assert_eq!(
            m.array_get("tokenizer.ggml.tokens", 99_999),
            Some(MetaValue::String("tok99999".into()))
        );
        assert_eq!(m.array_get("tokenizer.ggml.tokens", 100_000), None);

        // And the proof it stayed lazy: nothing decoded the whole array, so
        // the entry's cache is still empty. If `array_get` had gone through
        // `get`, this would hold a 100,000-element MetaValue::Array.
        let e = m.entry("tokenizer.ggml.tokens").unwrap();
        assert!(
            e.value.get().is_none(),
            "array access must not populate the whole-value cache"
        );
    }

    #[test]
    fn a_fixed_width_array_is_indexed_by_arithmetic_not_by_walking() {
        let mut v = 4u32.to_le_bytes().to_vec(); // U32 elements
        v.extend_from_slice(&5u64.to_le_bytes());
        // `i * 11` gives 0, 11, 22, 33, 44. THE DISTINCTNESS IS LOAD-BEARING
        // and this is not a style preference. Measured, in a worktree:
        //
        //   these values          + off-by-one in the indexing arithmetic
        //                           -> 2 tests RED
        //   every element 0u32    + the same off-by-one
        //                           -> 47 passed, 0 failed
        //
        // `array_get(k, 3)` returning element 2 is invisible when elements
        // 2 and 3 are the same bytes. The assertions stay true, the tests
        // stay honest, and the suite reports success over data that cannot
        // tell a right answer from a wrong one. Reproduce before changing
        // these values: replace them with a constant, apply
        // `index.checked_mul(w)` -> `index.saturating_sub(1).checked_mul(w)`,
        // and watch two red tests turn green.
        //
        // The shape came from Fuel, where an integer cast of a [-0.5, 0.5)
        // fill made every integer probe tensor all zeros; every comparison
        // passed honestly against 0 vs 0.
        for i in 0..5u32 {
            v.extend_from_slice(&(i * 11).to_le_bytes());
        }
        // A second key AFTER the array, and it is load-bearing.
        //
        // The out-of-range assertion below is the control for `array_get`'s
        // `index >= count` bound. With the array last in the file — which is
        // how this fixture was first written — index 5 lands exactly on the
        // end of the buffer, `decode_value` fails on truncation, and
        // `array_get` answers `None`: the right answer for the wrong reason.
        // Deleting the bound left that assertion green (measured, not
        // reasoned), so it was proving `Cursor`'s bounds check rather than
        // this function's.
        //
        // With a key following, index 5 reads that key's bytes instead.
        // Measured with the bound deleted: `Some(U32(5))` — the low half of
        // the 8-byte length prefix of "after" — and `Some(U32(1702127201))`
        // at index 7, which is "afte" read as a little-endian integer. A
        // token id of 5 is a number a caller would have accepted without
        // blinking, which is the whole reason the bound is worth a control.
        let bytes = gguf(&[
            ("nums", 9, v),
            ("after", 4, 0xDEAD_BEEFu32.to_le_bytes().to_vec()),
        ]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        assert_eq!(m.array_len("nums"), Some(5));
        assert_eq!(m.array_get("nums", 3), Some(MetaValue::U32(33)));
        assert_eq!(m.array_get("nums", 5), None);
    }

    #[test]
    fn array_accessors_agree_with_the_decoded_value() {
        // The two paths must not drift. Whatever `array_get(k, i)` returns
        // must equal `get(k)`'s i-th element — otherwise a consumer sees a
        // different vocabulary depending on which accessor it used.
        // `array_get`'s own doc names three element shapes with different
        // costs: fixed-width, String, and nested Array. A flat string array
        // pins one of them. The fixed-width arm is a different branch —
        // arithmetic rather than walking — and the nested arm was exercised
        // by NOTHING: hardcoding `ValueType::String` in place of `elem` in
        // the variable-width walk passed all 46 tests, while genuinely
        // breaking nested access. Agreement has to cover every shape the
        // function branches on, or it is agreement about one branch.
        for (name, payload) in [
            (
                "fixed",
                array_of(
                    4,
                    &(0..5u32)
                        .map(|i| (i * 11).to_le_bytes().to_vec())
                        .collect::<Vec<_>>(),
                ),
            ),
            ("strings", str_array(&["alpha", "beta", "gamma"])),
            (
                "nested",
                array_of(
                    9,
                    &[
                        str_array(&["a", "b"]),
                        str_array(&["c"]),
                        str_array(&["d", "e", "f"]),
                    ],
                ),
            ),
        ] {
            let bytes = gguf(&[("toks", 9, payload), trailer()]);
            let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
            let MetaValue::Array(all) = m.get("toks").unwrap().clone() else {
                panic!("{name}: expected an array");
            };
            for (i, want) in all.iter().enumerate() {
                assert_eq!(
                    m.array_get("toks", i as u64).as_ref(),
                    Some(want),
                    "{name} index {i}"
                );
            }
            assert_eq!(m.array_len("toks"), Some(all.len() as u64), "{name} len");
            // The accessors must answer about THE KEY ASKED FOR. Replacing
            // `entry(key)` with a scan for the first array in the file left
            // every other assertion here green — this file has exactly one
            // array, so "the array" and "the array named toks" were
            // indistinguishable until these two lines. `after` is the
            // trailing U32 scalar, so it also pins the `ty != Array` guard
            // against a file that HAS an array, which the scalar-only test
            // cannot do.
            assert_eq!(m.array_len("nope"), None, "{name}: absent key");
            assert_eq!(m.array_get("nope", 0), None, "{name}: absent key");
            assert_eq!(m.array_len("after"), None, "{name}: scalar key");
            assert_eq!(m.array_get("after", 0), None, "{name}: scalar key");
            // Past the end, with a real key's bytes sitting there to be
            // misread. Inside the loop because each shape reaches it by a
            // different branch.
            assert_eq!(
                m.array_get("toks", all.len() as u64),
                None,
                "{name}: past the end"
            );
        }
    }

    #[test]
    fn deep_nesting_is_readable_by_both_paths_or_neither() {
        // `array_get` restarts the nesting-depth budget at 0, while indexing
        // spent one level reaching the same element. The comment on that
        // code claims the asymmetry can never make `array_get` REJECT a
        // subtree indexing accepted — the direction that matters, because a
        // key that survives the open must stay readable.
        //
        // Nothing pinned it, and all 46 tests passed with the decode started
        // at depth 2 instead of 0. This is the Task 5 lesson again: when two
        // paths walk one grammar, assert that THEY AGREE rather than that
        // each works alone.
        //
        // 64 is not an arbitrary depth, it is the ONLY one that works. The
        // reset buys slack, and the slack is what hides an accounting error,
        // so the test has to sit where there is least of it. Measured, by
        // starting the decode at each shift and finding the first that
        // fails:
        //
        //     nest(60) parses, no shift up to 5 breaks it
        //     nest(61) parses, breaks at shift 5
        //     nest(62) parses, breaks at shift 4
        //     nest(63) parses, breaks at shift 3
        //     nest(64) parses, breaks at shift 2   <-- here
        //     nest(65) REFUSED at parse
        //
        // This test was written at 63 first. The off-by-one regression it
        // exists to catch is a shift of 2, so at 63 it passed under the very
        // mutation it was written for — an eighth control in this plan that
        // could not reach the assertion it named. One level deeper and it
        // fires. Do not lower this number.
        let bytes = gguf(&[("deep", 9, nest(64)), trailer()]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").expect("64 levels parses");
        let MetaValue::Array(all) = m.get("deep").unwrap().clone() else {
            panic!("expected an array");
        };
        assert_eq!(all.len(), 1, "each level wraps exactly one element");
        assert_eq!(
            m.array_get("deep", 0).as_ref(),
            Some(&all[0]),
            "the deepest element indexing accepted must stay readable"
        );
    }

    #[test]
    fn a_scalar_has_no_array_length_and_no_elements() {
        let bytes = gguf(&[("k", 8, s("scalar"))]);
        let (m, _) = GgufMetadata::parse(&bytes, "t.gguf").unwrap();
        assert_eq!(m.array_len("k"), None);
        assert_eq!(m.array_get("k", 0), None);
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
