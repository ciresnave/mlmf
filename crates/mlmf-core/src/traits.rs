//! The seam between the format axis, the source axis, and everything above.
//!
//! Format crates implement [`TensorContainer`] and [`MetadataSource`].
//! Source crates implement [`ByteSource`]. Both depend on `mlmf-core`
//! alone, which is what lets any source compose with any format.

use std::borrow::Cow;
use std::ops::Range;

use crate::{MetaValue, Result, TensorDescriptor};

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
    /// If the range lies outside the artifact, or the underlying transport
    /// fails ([`crate::ErrorKind::Source`]).
    fn read_range(&self, range: Range<u64>, into: &mut [u8]) -> Result<()>;
}

/// Something that declares named tensors and can hand out their bytes.
///
/// There is no tensor type, no device and no backend trait: a consumer
/// builds whatever it wants from the slice. Alignment is not guaranteed —
/// see [`crate::align`].
pub trait TensorContainer {
    /// Every tensor this container declares.
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
    /// If the descriptor's range lies outside the container's data.
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
    /// Absent means **not declared**. It never means a default (spec §5).
    fn get(&self, key: &str) -> Option<&MetaValue>;

    /// Every key declared, in unspecified order.
    fn keys(&self) -> Vec<&str>;
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
            let start = usize::try_from(range.start).expect("fits");
            let end = usize::try_from(range.end).expect("fits");
            if end > self.blob.len() || into.len() != end - start {
                return Err(crate::Error::from(crate::ErrorKind::Truncated {
                    needed: range.end,
                    available: self.blob.len() as u64,
                }));
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
}
