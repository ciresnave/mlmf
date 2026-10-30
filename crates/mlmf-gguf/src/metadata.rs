// SPDX-License-Identifier: MIT OR Apache-2.0
//! The key-value block, indexed at open and decoded on demand.

use std::sync::OnceLock;

use mlmf_core::{
    ByteSource, Declaration, MetaValue, MetadataSource, Report, Unrecognized, UnrecognizedKind,
};

use crate::cursor::Cursor;
use crate::error::{GgufError, Stage};
use crate::header::{Header, parse_header};
use crate::value::{ValueType, decode_value, read_array_prefix, skip_value};

/// GGUF's documented default when `general.alignment` is absent.
///
/// `u32` because the specification types the key as `uint32`. This is the
/// SINGLE source of truth for that 32: `crate::requirements` supplies it as
/// a CD-1 citable default, and a second literal there could have drifted
/// from this one silently.
pub(crate) const DEFAULT_ALIGNMENT_U32: u32 = 32;

/// The same 32, widened once for offset arithmetic.
///
/// Derived rather than written again — widening `u32` to `u64` is lossless,
/// and a derived constant cannot disagree with its source. (A test asserting
/// the two agree would be vacuous for exactly that reason, so there isn't
/// one.)
const DEFAULT_ALIGNMENT: u64 = DEFAULT_ALIGNMENT_U32 as u64;

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

    /// [`Self::parse`] over a [`ByteSource`], so a caller holding a source
    /// (a memory map, an owned buffer) need not know to call `as_bytes`
    /// itself. The metadata borrows from `source`, so a map stays zero-copy.
    ///
    /// # Errors
    ///
    /// Exactly those of [`Self::parse`].
    pub fn parse_from(
        source: &'a dyn ByteSource,
        origin: &str,
    ) -> Result<(Self, Report), GgufError> {
        Self::parse(source.as_bytes(), origin)
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

    /// Every declared key paired with its value, in ONE pass over the
    /// index this type already built at open -- not the trait default's
    /// `keys().len()` independent re-lookups through `entry`'s own
    /// linear scan.
    ///
    /// An entry whose value could not be decoded (the same
    /// [`Declaration::Unreadable`] case [`Self::declaration`] reports) is
    /// omitted here exactly as it is from [`MetadataSource::get`] -- this
    /// method answers "every key paired with its VALUE", and an
    /// undecodable entry has none to pair. [`Self::keys`] is still the
    /// complete key list including that entry; this is not a second source
    /// of truth about which keys exist, only about which pair cleanly.
    fn entries(&self) -> Vec<(&str, &MetaValue)> {
        self.entries
            .iter()
            .filter_map(|e| self.value_of(e).map(|v| (e.key.as_str(), v)))
            .collect()
    }
}

#[cfg(test)]
mod tests;
