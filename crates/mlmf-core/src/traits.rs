//! The seam between the format axis, the source axis, and everything above.
//!
//! Format crates implement [`TensorContainer`] and [`MetadataSource`].
//! Source crates implement [`ByteSource`]. Both depend on `mlmf-core`
//! alone, which is what lets any source compose with any format.

use std::borrow::Cow;
use std::ops::Range;

use crate::{Declaration, MetaValue, Result, TensorDescriptor};

/// A contiguous region of bytes obtained from somewhere.
///
/// Implemented by source crates over a memory map, an owned buffer, or a
/// downloaded file. `mlmf-core` performs no I/O and never implements this.
pub trait ByteSource {
    /// The bytes.
    fn as_bytes(&self) -> &[u8];
}

/// A source that can serve arbitrary byte ranges without materializing the
/// whole artifact.
///
/// Spec §3.4 says "`&[u8]` **and `impl Read`** are the primary entry points
/// and mmap is one byte-source among them, **which is what keeps streaming
/// and IPC transports possible later**". [`ByteSource`] is only the first
/// half: it demands one fully-materialized contiguous slice, so an HTTP
/// range-request source, an IPC or shared-memory transport, a tar or ZIP
/// member, or `mlmf-source-hub` streaming a 140 GB shard would each have to
/// buffer the entire artifact into memory before it could be a source at
/// all. That forecloses the very transports §3.1 sells the two-axis design
/// on ("adding S3, IPC or an in-memory source later touches nothing that
/// parses").
///
/// Expressed as a ranged read over a caller-supplied buffer rather than
/// `std::io::Read` so that `mlmf-core` need not name `std::io`, which keeps
/// the C3 allow-list of permitted `std` submodules tight.
pub trait RangedSource {
    /// Total size of the artifact in bytes, if the source knows it.
    fn len(&self) -> Option<u64>;

    /// Whether the artifact is known to be empty.
    ///
    /// `None` means the length is not known, which is not the same as empty.
    fn is_empty(&self) -> Option<bool> {
        self.len().map(|n| n == 0)
    }

    /// Fill `into` with the bytes in `range`.
    ///
    /// `into.len()` must equal the range's width; an implementation must not
    /// short-read.
    ///
    /// # Errors
    ///
    /// **[`crate::ErrorKind::Truncated`] if the range lies outside the
    /// artifact** — `needed` is the range's end, `available` the artifact's
    /// length. **[`crate::ErrorKind::Source`] if the underlying transport
    /// fails, if the range is inverted, or if `into.len()` does not equal
    /// the range's width.**
    ///
    /// Those last two are violations of the precondition above, and they are
    /// `Source` rather than a typed variant because
    /// [`crate::ErrorKind::InvertedRange`] and
    /// [`crate::ErrorKind::SizeMismatch`] both carry a tensor `name`. They
    /// are format-axis errors; this is a source-axis trait and a source has
    /// no tensors. **They were unbound until a review found the two
    /// implementations answering them differently** — and the reference below
    /// *panicked* on an inverted range rather than returning anything.
    ///
    /// Two sentences, deliberately. This used to read *"if the range lies
    /// outside the artifact, or the underlying transport fails
    /// (`ErrorKind::Source`)"*, and **the parenthetical bound ambiguously**:
    /// to the transport clause alone, or to both? The reference
    /// implementation below read it the first way and returned `Truncated`
    /// for out-of-range; `mlmf-gguf` cited that reference by name and
    /// followed it; `mlmf-safetensors` followed `mlmf-gguf`. **A fourth
    /// implementation read it the second way and returned `Source`**, which
    /// is how a workspace ends up with two answers to one question.
    ///
    /// `Truncated`'s numbers are the reason it wins on the merits and not
    /// only on precedent: a caller branches on two integers instead of
    /// parsing a sentence, **and a parsed error message is a contract nobody
    /// wrote down and everybody depends on.**
    fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> Result<()>;
}

/// Something that declares named tensors and can hand out their bytes.
///
/// There is no tensor type, no device and no backend trait: a consumer
/// builds whatever it wants from the slice. Alignment is not guaranteed —
/// see [`crate::align`].
pub trait TensorContainer {
    /// Every tensor this container declares **that this build can describe**.
    ///
    /// **Descriptors MAY name overlapping byte ranges, and what an overlap
    /// MEANS is a fact about the format rather than about this trait.**
    ///
    /// In GGUF every tensor carries its own explicit offset and a writer has
    /// no reason to make two of them collide, so `mlmf-gguf` treats an
    /// overlap as malformed and reports it. In safetensors, tied weights —
    /// `lm_head` and `embed_tokens` referring to the same bytes — are a
    /// standard layout that real models use, and reporting one would blame
    /// a valid file.
    ///
    /// The rule therefore belongs to each format crate, not here. Recorded
    /// because it was previously implicit: one backend's sweep existed and
    /// `crate::UnrecognizedKind::TensorDeclined`'s doc named an overlapping
    /// range as a reason, which together read as a seam-level rule that had
    /// never been decided.
    ///
    /// Two consequences for a consumer:
    ///
    /// - An empty report does **not** prove the ranges are disjoint. It
    ///   proves this format's reader found nothing worth reporting.
    /// - A reported overlap does **not** prove corruption unless you know
    ///   the format forbids sharing.
    ///
    /// A tensor whose declared encoding could not be resolved — an
    /// unrecognized ggml type code, or a safetensors dtype string this
    /// build does not know — is absent from this slice.
    /// [`TensorDescriptor`] has no way to say "length unknown," so there is
    /// no descriptor to put here; instead the parse's [`crate::Report`]
    /// gains a [`crate::UnrecognizedKind::TensorEncoding`] entry naming the
    /// tensor, the family, and the type it declares — a
    /// [`crate::DeclaredType::Code`] or a [`crate::DeclaredType::Name`],
    /// because formats declare types both ways. A consumer that ignores the
    /// report sees only a shorter list, with no other signal that anything
    /// is missing.
    ///
    /// **A tensor whose declared byte range runs past the end of the file
    /// is NOT absent.** It stays here with the range the file declares, and
    /// [`Self::tensor_bytes`] is where reading it fails. A descriptor
    /// records what the file DECLARES — [`TensorDescriptor::bytes`] carries
    /// that ruling and the argument for it — and dropping one would be this
    /// crate deciding a declaration does not count, which is interpretation.
    ///
    /// The line that separates the two, as far as the seam draws one:
    /// **a descriptor is kept when the declaration is internally consistent
    /// but unsatisfiable by the file's length, and dropped when there is no
    /// single consistent declaration to record.** An unresolvable encoding
    /// has no extent, so there is nothing to write down; a duplicate name
    /// has two declarations claiming one key and a container cannot hold
    /// both under it; a range past the end is one coherent declaration the
    /// file simply cannot honour, and that is a fact about the bytes rather
    /// than about the record.
    ///
    /// **That is a description, not a promise, and the exact set is
    /// per-format.** `crate::UnrecognizedKind::TensorDeclined`'s doc says
    /// the same thing from the other side. An earlier version of this
    /// paragraph claimed absence "means *this build cannot describe it*,
    /// never *this build cannot read it*", which reads as an absolute and is
    /// false in this very workspace: `mlmf-gguf` drops the second of two
    /// tensors declared under one name, and that tensor is perfectly
    /// describable — its shape, encoding and bytes are all known — and it is
    /// absent. **[`Self::tensor`] is the authoritative answer to whether a
    /// given name is in the list.**
    ///
    /// # Order is unspecified
    ///
    /// Exactly as [`MetadataSource::keys`] is, and for the same reason:
    /// a format crate yields whatever its own directory hands it, and the
    /// two that exist do not agree. `mlmf-gguf` yields declaration order
    /// from a forward walk of records; `mlmf-safetensors` yields
    /// lexicographic order, because `serde_json` without `preserve_order`
    /// backs a JSON object with a `BTreeMap`. **A consumer holding
    /// `&dyn TensorContainer` must compare sets, not sequences.**
    ///
    /// Written down because it was measured false: the assumption that both
    /// would agree predates either backend, and
    /// `mlmf-conformance`'s cross-backend test asserts the raw orders
    /// DIFFER, so the divergence is pinned outside this crate while the
    /// trait said nothing about it. A format crate may promise more about
    /// its own concrete type — `mlmf-gguf` does — and a caller holding a
    /// trait object may not rely on it.
    fn tensors(&self) -> &[TensorDescriptor];

    /// The tensor declared under `name`, if any.
    ///
    /// `None` means **not declared** — the same rule [`MetadataSource::get`]
    /// follows. It is not an error and not a default (spec §5).
    ///
    /// Consumers overwhelmingly want keyed access rather than iteration: a
    /// survey of Fuel's loader found 27 by-name lookups, several of them
    /// inside per-layer loops. Without this method each of those sites grows
    /// its own `tensors().iter().find(...)`, which puts the lookup strategy
    /// in consumer code where MLMF cannot improve it and where the quadratic
    /// walk is invisible.
    ///
    /// The default is that linear scan, which is the honest answer for a
    /// container holding a handful of tensors. **A format crate parsing a
    /// real model should override this with an index** — a 70B GGUF declares
    /// on the order of a thousand tensors, and a caller walking every layer
    /// turns the default into a million comparisons.
    fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors().iter().find(|d| d.name == name)
    }

    /// The bytes for one tensor, **borrowed or owned**.
    ///
    /// `Cow` rather than `&[u8]` because spec §11 requires it by name:
    /// "mmap-slicing a `.bin` works only because `torch.save` writes ZIP
    /// entries **stored, not deflated**. A compressed entry must be
    /// decompressed into an owned buffer, so **`mlmf-pickle` returns
    /// borrowed-or-owned**, unlike GGUF and safetensors which are always
    /// borrowable." A `&[u8]` return cannot carry that buffer, which left
    /// `mlmf-pickle` with three bad options: inflate every deflated entry
    /// eagerly at open time, reach for interior mutability plus unsafe
    /// lifetime extension in a crate that is `#![forbid(unsafe_code)]`, or
    /// not implement the seam at all — breaking §4.5's premise that
    /// `mlmf-meta` and the umbrella consume one seam.
    ///
    /// It is also what makes AL-3 hold *through* the seam. A caller can test
    /// `matches!(bytes, Cow::Owned(_))` and see that MLMF allocated. With
    /// `&[u8]` the inflate-into-self copy would be exactly the invisible
    /// cost AL-3 forbids, with no API surface able to reveal it.
    ///
    /// # Errors
    ///
    /// If the descriptor's range lies outside the container's data —
    /// **including a descriptor this very container produced.** A tensor
    /// declared with a range past the end of the file keeps its descriptor
    /// in [`Self::tensors`], and this method is where reading it fails.
    fn tensor_bytes(&self, descriptor: &TensorDescriptor) -> Result<Cow<'_, [u8]>>;
}

/// Something that declares typed metadata under string keys.
///
/// GGUF's in-file metadata and HuggingFace's JSON sidecars both project
/// into this, which is what makes the layer above format-agnostic:
/// config accessors and chat-template extraction are written once.
pub trait MetadataSource {
    /// The value declared under `key`, if any.
    ///
    /// **A [`MetaValue`]'s variant reports HOW THE FORMAT DECLARED the
    /// value, not what the value means.** GGUF declares typed values —
    /// `general.alignment` arrives as `U32`. Safetensors' `__metadata__` is
    /// `string -> string`, so every value from it arrives as `String`, and
    /// **thirteen of the fourteen** variants never appear. (Thirteen is
    /// GGUF's value-type count; [`MetaValue`] is those thirteen plus
    /// `Bytes`, which is not one. `MetaValue::kind` is the exhaustive match
    /// that keeps the number honest.)
    ///
    /// This is deliberate and it is the charter: MLMF extracts what a file
    /// says. Deciding that the string `"32"` is the number 32 — or that
    /// `"0x20"` is, or that `"true"` is a boolean — is format knowledge, and
    /// a format that did not declare a number did not declare a number.
    ///
    /// **So [`MetaValue::as_u64`] and its siblings widen losslessly within a
    /// family and NEVER parse.** `as_u64` accepts `U8`/`U16`/`U32`/`U64`
    /// and returns `None` for a `String`. A `None` from an accessor means
    /// **"this format did not declare that kind of value here"**, which is a
    /// fact about the file, not a failure of this crate.
    ///
    /// A consumer that wants a number out of a string-valued format must
    /// parse it deliberately, at a layer that knows which format it is
    /// reading. That layer is not this one.
    ///
    /// `None` means **not declared**. It never means a default (spec §5) —
    /// but see [`Self::index_complete`] for when "not declared" is a fact
    /// about the source and when it only means "not found in the part that
    /// could be read".
    fn get(&self, key: &str) -> Option<&MetaValue>;

    /// Every key declared, in unspecified order.
    fn keys(&self) -> Vec<&str>;

    /// Whether this source saw every key its container declared.
    ///
    /// **Required rather than defaulted, deliberately.** A default of
    /// `true` lets a source that CAN stop early claim a completeness it
    /// does not have, which is a false negative delivered silently — the
    /// exact failure this method exists to prevent. A default of `false`
    /// would make every eager source useless for negative findings. Having
    /// no default fails at compile time instead, which is the only one of
    /// the three that cannot be got wrong quietly.
    ///
    /// Return `true` if the source reads its whole input or fails: a JSON
    /// sidecar either parses or does not. Return `false` when a walk
    /// stopped before the end — GGUF's key-value block is walked
    /// sequentially, and a value type this build does not know has an
    /// unknown width, so the parse cannot find the key after it.
    ///
    /// This lives on the trait, not on a concrete type, because it is
    /// exactly the caller holding `&dyn MetadataSource` who cannot
    /// otherwise ask. It changes the meaning of every negative answer this
    /// trait gives — [`Declaration::Absent`] and `get`'s `None` alike —
    /// so it has to travel with them.
    fn index_complete(&self) -> bool;

    /// What this source knows about `key` — three states, not two.
    ///
    /// The default reports [`Declaration::Declared`] or
    /// [`Declaration::Absent`] and **never** [`Declaration::Unreadable`],
    /// because a source that has not overridden this has no undecodable
    /// values to report: everything it holds is already a [`MetaValue`].
    /// Claiming a decode failure that did not happen would be worse than
    /// not reporting one.
    fn declaration(&self, key: &str) -> Declaration<'_> {
        match self.get(key) {
            Some(v) => Declaration::Declared(v),
            None => Declaration::Absent,
        }
    }

    /// Number of elements in the array declared under `key`.
    ///
    /// `None` means the key is absent **or** its value is not an array.
    /// A scalar deliberately has no length rather than length 1 — reporting
    /// 1 would let a caller index a string as though it were a vocabulary.
    ///
    /// Provided so a caller can distinguish "index out of range" from
    /// "there is no array here", which consumer R4 needs to tell apart
    /// three different ways a token-id lookup fails.
    fn array_len(&self, key: &str) -> Option<u64> {
        match self.get(key) {
            Some(MetaValue::Array(items)) => Some(items.len() as u64),
            _ => None,
        }
    }

    /// One element of the array declared under `key`, by index.
    ///
    /// Returns owned, because a format crate should be able to decode a
    /// single element out of bytes without materializing the array — see
    /// consumer R5. The default walks an already-decoded value, which is
    /// the honest answer for a source that has one; **a format crate
    /// reading a 500,000-element vocabulary must override this.**
    fn array_get(&self, key: &str, index: u64) -> Option<MetaValue> {
        let items = match self.get(key) {
            Some(MetaValue::Array(items)) => items,
            _ => return None,
        };
        usize::try_from(index)
            .ok()
            .and_then(|i| items.get(i))
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Encoding, MetaValue, Shape, TensorDescriptor};
    use std::collections::HashMap;

    struct Fake {
        blob: Vec<u8>,
        tensors: Vec<TensorDescriptor>,
        meta: HashMap<String, MetaValue>,
    }

    impl ByteSource for Fake {
        fn as_bytes(&self) -> &[u8] {
            &self.blob
        }
    }

    impl TensorContainer for Fake {
        fn tensors(&self) -> &[TensorDescriptor] {
            &self.tensors
        }
        fn tensor_bytes(&self, d: &TensorDescriptor) -> crate::Result<Cow<'_, [u8]>> {
            let start = usize::try_from(d.bytes.start).expect("fits");
            let end = usize::try_from(d.bytes.end).expect("fits");
            Ok(Cow::Borrowed(&self.blob[start..end]))
        }
    }

    impl RangedSource for Fake {
        fn len(&self) -> Option<u64> {
            Some(self.blob.len() as u64)
        }
        fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> crate::Result<()> {
            // ORDER MATTERS. This guard is first because the width below
            // subtracts the endpoints, and `end - start` underflows on an
            // inverted range: a debug build panics, a release build computes
            // a width near `usize::MAX` and then indexes with it.
            //
            // The old `end > blob.len() || into.len() != end - start` did not
            // short-circuit away from it: for `24..8` on a 32-byte blob,
            // `8 > 32` is false, so `||` evaluated the subtraction and the
            // reference implementation of this trait panicked.
            if range.end < range.start {
                return Err(crate::Error::from(crate::ErrorKind::Source(
                    format!(
                        "byte range {}..{} ends before it starts",
                        range.start, range.end
                    )
                    .into(),
                )));
            }
            let start = usize::try_from(range.start).expect("fits");
            let end = usize::try_from(range.end).expect("fits");
            if end > self.blob.len() {
                return Err(crate::Error::from(crate::ErrorKind::Truncated {
                    needed: range.end,
                    available: self.blob.len() as u64,
                }));
            }
            if into.len() != end - start {
                // `Source`, not `Truncated`. A width mismatch is the caller's
                // buffer disagreeing with the caller's range; nothing is
                // truncated. The old code answered it with
                // `Truncated { needed: 8, available: 32 }` — two numbers
                // describing no truncation at all.
                return Err(crate::Error::from(crate::ErrorKind::Source(
                    format!(
                        "buffer of {} bytes does not match the {}-byte range {}..{}",
                        into.len(),
                        end - start,
                        range.start,
                        range.end
                    )
                    .into(),
                )));
            }
            into.copy_from_slice(&self.blob[start..end]);
            Ok(())
        }
    }

    /// A container that has to materialize — the shape spec §11 says
    /// `mlmf-pickle` takes for a deflated ZIP entry.
    struct Inflating {
        tensors: Vec<TensorDescriptor>,
    }

    impl TensorContainer for Inflating {
        fn tensors(&self) -> &[TensorDescriptor] {
            &self.tensors
        }
        fn tensor_bytes(&self, d: &TensorDescriptor) -> crate::Result<Cow<'_, [u8]>> {
            let n = usize::try_from(d.byte_len()).expect("fits");
            Ok(Cow::Owned(vec![0xAB; n]))
        }
    }

    impl MetadataSource for Fake {
        fn index_complete(&self) -> bool {
            true
        }

        fn get(&self, key: &str) -> Option<&MetaValue> {
            self.meta.get(key)
        }
        fn keys(&self) -> Vec<&str> {
            self.meta.keys().map(String::as_str).collect()
        }
    }

    fn fake() -> Fake {
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".to_string(),
            MetaValue::String("llama".into()),
        );
        Fake {
            blob: vec![0u8; 32],
            tensors: vec![TensorDescriptor {
                name: "blk.0.attn_q.weight".into(),
                shape: Shape::new([2usize, 4]),
                encoding: Encoding::Dense(DType::F32),
                bytes: 0..32,
            }],
            meta,
        }
    }

    #[test]
    fn a_container_hands_out_borrowed_bytes_not_owned_buffers() {
        let f = fake();
        let d = &f.tensors()[0];
        let bytes = f.tensor_bytes(d).expect("in range");
        assert!(
            matches!(bytes, Cow::Borrowed(_)),
            "a mappable container must not allocate"
        );
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes.as_ptr(), f.as_bytes().as_ptr());
    }

    #[test]
    fn a_caller_can_see_when_mlmf_had_to_allocate() {
        // AL-3 through the seam: the copy is a decision the caller can see.
        let c = Inflating {
            tensors: vec![TensorDescriptor {
                name: "deflated".into(),
                shape: Shape::new([2usize, 4]),
                encoding: Encoding::Dense(DType::F32),
                bytes: 0..32,
            }],
        };
        let bytes = c.tensor_bytes(&c.tensors()[0]).expect("inflates");
        assert!(
            matches!(bytes, Cow::Owned(_)),
            "an owned result must be distinguishable from a borrowed one"
        );
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn a_ranged_source_serves_a_window_without_materializing_the_whole() {
        // Spec §3.4's second entry point. A source that cannot produce one
        // contiguous slice — HTTP ranges, IPC, a ZIP member — still composes
        // with the format axis.
        let f = fake();
        assert_eq!(RangedSource::len(&f), Some(32));
        assert_eq!(f.is_empty(), Some(false));

        let mut buf = [0u8; 8];
        f.read_range(4..12, &mut buf).expect("in range");
        assert_eq!(buf, [0u8; 8]);

        let mut wrong = [0u8; 4];
        assert!(f.read_range(0..8, &mut wrong).is_err(), "no short reads");
        assert!(f.read_range(0..64, &mut [0u8; 64]).is_err(), "out of range");

        // An INVERTED range, which this test never covered. `end - start`
        // underflows for it, and the `end > blob.len()` check above does not
        // short-circuit: for 24..8 on a 32-byte blob, `8 > 32` is false, so
        // `||` evaluates the subtraction. Debug builds panic; release builds
        // compute a width near `usize::MAX` and then index.
        //
        // `mlmf-source-file` documents this exact hazard and orders its
        // checks to avoid it. The reference implementation the trait doc
        // points readers at did not.
        let mut small = [0u8; 4];
        // Built field-by-field, not written `24..8`. `clippy::
        // reversed_empty_ranges` is deny-by-default and rejects the literal
        // — measured, it failed this crate's own clippy gate. The lint is
        // right about typed ranges and blind to the case that matters:
        // nobody TYPES an inverted range, it arrives as two computed
        // offsets from a header the reader never validated.
        let inverted = std::ops::Range { start: 24, end: 8 };
        assert!(
            f.read_range(inverted, &mut small).is_err(),
            "an inverted range must be an error, not a panic"
        );
    }

    #[test]
    fn one_type_can_satisfy_both_seams() {
        // This is what makes the layer above format-agnostic: GGUF
        // metadata and HF JSON both project into MetadataSource, so
        // accessors are written once and serve both.
        let f = fake();
        assert_eq!(
            f.get("general.architecture").and_then(MetaValue::as_str),
            Some(&"llama".to_string())
        );
        assert!(f.get("absent").is_none());
        assert_eq!(f.keys().len(), 1);
    }

    #[test]
    fn traits_are_object_safe() {
        let f = fake();
        let m: &dyn MetadataSource = &f;
        assert!(m.get("general.architecture").is_some());
    }

    #[test]
    fn a_container_can_be_asked_for_one_tensor_by_name() {
        // Every consumer surveyed does keyed lookup, not iteration: Fuel
        // fetches by name at 27 sites, several inside per-layer loops. With
        // only `tensors()` in the seam each of those sites writes its own
        // `.iter().find()`, so the lookup strategy — and its cost — lands in
        // consumer code where MLMF cannot improve it.
        let f = fake();
        assert_eq!(
            f.tensor("blk.0.attn_q.weight").map(|d| d.name.as_str()),
            Some("blk.0.attn_q.weight")
        );
        // Absent means absent. It is not an error and not a default.
        assert!(f.tensor("blk.99.attn_q.weight").is_none());
    }

    #[test]
    fn a_container_with_many_tensors_can_replace_the_scan() {
        // The default is a linear scan, which is correct but quadratic when
        // a caller walks every layer. The point of putting the method in the
        // trait rather than leaving it to callers is that a format crate can
        // override it with an index; this asserts the override is actually
        // reachable through the seam rather than shadowed by the default.
        struct Indexed {
            tensors: Vec<TensorDescriptor>,
            hits: std::cell::Cell<u32>,
        }
        impl TensorContainer for Indexed {
            fn tensors(&self) -> &[TensorDescriptor] {
                &self.tensors
            }
            fn tensor_bytes(&self, _d: &TensorDescriptor) -> crate::Result<Cow<'_, [u8]>> {
                Ok(Cow::Borrowed(&[]))
            }
            fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
                self.hits.set(self.hits.get() + 1);
                self.tensors.iter().find(|d| d.name == name)
            }
        }
        let c = Indexed {
            tensors: fake().tensors,
            hits: std::cell::Cell::new(0),
        };
        let seam: &dyn TensorContainer = &c;
        assert!(seam.tensor("blk.0.attn_q.weight").is_some());
        assert_eq!(c.hits.get(), 1, "the override must win over the default");
    }

    #[test]
    fn descriptor_ranges_index_the_slice_the_container_was_given() {
        // CD-4. A GGUF file's tensor info records an offset relative to the
        // start of the *data region*, while a safetensors header records one
        // relative to the end of the header — two different bases, neither of
        // which is the start of the file. If the seam left the base
        // unstated, every consumer would guess, and a guess that is wrong by
        // exactly the header length still produces plausible-looking floats.
        //
        // The rule: `bytes` indexes the byte slice the container was opened
        // over, with no addend. Rebasing is the format crate's job, done once
        // at parse time, not the caller's job done 27 times.
        let header = 16usize;
        let mut blob = vec![0u8; header + 32];
        blob[header..].copy_from_slice(&[0xCD; 32]);
        let f = Fake {
            blob,
            tensors: vec![TensorDescriptor {
                name: "after.header".into(),
                shape: Shape::new([2usize, 4]),
                encoding: Encoding::Dense(DType::F32),
                bytes: (header as u64)..(header as u64 + 32),
            }],
            meta: HashMap::new(),
        };
        let d = f.tensor("after.header").expect("declared");
        d.validate().expect("consistent");
        let bytes = f.tensor_bytes(d).expect("in range");
        assert_eq!(
            bytes.as_ref(),
            &[0xCD; 32],
            "the range must land on the data, not the header"
        );
        // Stated as an address, so a container that helpfully re-added its
        // own base offset would fail here rather than silently reading 16
        // bytes of header as the first four weights.
        // Compared as addresses rather than via pointer arithmetic: the
        // crate is `#![forbid(unsafe_code)]`, and a test is not exempt.
        assert_eq!(bytes.as_ptr() as usize, f.blob.as_ptr() as usize + header);
    }

    #[test]
    fn a_declaration_separates_absent_from_unreadable_from_declared() {
        // R2. A single Option collapses "this file has no chat template"
        // with "this file has one and we could not decode it", and those
        // send an operator in opposite directions.
        struct Partial {
            good: MetaValue,
            bad: crate::Unrecognized,
        }
        impl MetadataSource for Partial {
            fn index_complete(&self) -> bool {
                true
            }

            fn get(&self, key: &str) -> Option<&MetaValue> {
                (key == "good").then_some(&self.good)
            }
            fn keys(&self) -> Vec<&str> {
                vec!["good", "bad"]
            }
            fn declaration(&self, key: &str) -> crate::Declaration<'_> {
                match key {
                    "good" => crate::Declaration::Declared(&self.good),
                    "bad" => crate::Declaration::Unreadable(&self.bad),
                    _ => crate::Declaration::Absent,
                }
            }
        }
        let p = Partial {
            good: MetaValue::String("ok".into()),
            bad: crate::Unrecognized {
                kind: crate::UnrecognizedKind::MetadataKey {
                    key: "bad".into(),
                    value: Some(MetaValue::U32(7)),
                    reason: None,
                },
                origin: "model.gguf".into(),
            },
        };

        assert!(matches!(
            p.declaration("good"),
            crate::Declaration::Declared(MetaValue::String(s)) if s == "ok"
        ));
        // The distinction that matters: `get` returns None for BOTH of
        // these, and `declaration` does not.
        assert!(p.get("bad").is_none());
        assert!(p.get("missing").is_none());
        match p.declaration("bad") {
            crate::Declaration::Unreadable(u) => assert_eq!(u.origin, "model.gguf"),
            other => panic!("expected Unreadable, got {other:?}"),
        }
        assert!(matches!(
            p.declaration("missing"),
            crate::Declaration::Absent
        ));
    }

    #[test]
    fn the_default_declaration_never_invents_an_unreadable_state() {
        // An implementor that has not overridden `declaration` must report
        // Declared or Absent and never Unreadable — claiming a decode
        // failure that did not happen is worse than not reporting one.
        let f = fake();
        assert!(matches!(
            f.declaration("general.architecture"),
            Declaration::Declared(_)
        ));
        assert!(matches!(f.declaration("absent"), Declaration::Absent));
    }

    #[test]
    fn array_accessors_default_to_walking_the_materialized_value() {
        // The default impls are correct-but-slow, which is the honest
        // answer for a source that has already decoded everything. A format
        // crate that can index bytes overrides them; Task 7 does exactly
        // that, and this pins the semantics both must satisfy.
        struct WithArray(MetaValue);
        impl MetadataSource for WithArray {
            fn index_complete(&self) -> bool {
                true
            }

            fn get(&self, key: &str) -> Option<&MetaValue> {
                (key == "toks").then_some(&self.0)
            }
            fn keys(&self) -> Vec<&str> {
                vec!["toks"]
            }
        }
        let w = WithArray(MetaValue::Array(vec![
            MetaValue::String("a".into()),
            MetaValue::String("b".into()),
            MetaValue::String("c".into()),
        ]));

        assert_eq!(w.array_len("toks"), Some(3));
        assert_eq!(w.array_get("toks", 1), Some(MetaValue::String("b".into())));
        // Out of range is None, and is distinguishable from "not an array"
        // and "absent" by consulting array_len first — which is exactly how
        // R4's three id-resolution failures get told apart.
        assert_eq!(w.array_get("toks", 3), None);
        assert_eq!(w.array_len("absent"), None);
        assert_eq!(w.array_get("absent", 0), None);
    }

    #[test]
    fn a_non_array_value_has_no_length_rather_than_length_one() {
        // A scalar is not a one-element array. Reporting Some(1) would let
        // a consumer index a string as though it were a vocabulary.
        let f = fake();
        assert_eq!(f.array_len("general.architecture"), None);
        assert_eq!(f.array_get("general.architecture", 0), None);
    }
}
