//! What a model file declares about one tensor.

use std::ops::Range;

use crate::align::MAX_REPORTED_ALIGN;
use crate::{Encoding, Error, ErrorKind, Result, Shape};

/// One tensor as the file declares it: a name, a shape, an encoding, and
/// the byte range its data occupies.
///
/// There is no tensor type here and no device. A consumer takes the byte
/// range and builds whatever it wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    /// Name exactly as declared. Never normalized (spec §9 2.1).
    pub name: String,
    /// Dimensions in declared order (spec §4.2).
    pub shape: Shape,
    /// How the elements are laid out.
    pub encoding: Encoding,
    /// Byte range **into the slice the container was opened over**, absolute
    /// and with no addend (CD-4).
    ///
    /// This is the one fact in the descriptor a consumer cannot check for
    /// itself, so leaving the base unstated would be the most expensive kind
    /// of ambiguity: every format records offsets against a *different*
    /// origin, and none of them is the start of the file. GGUF's tensor info
    /// records an offset relative to the start of the data region, which
    /// begins after the header, the metadata KV block and alignment padding;
    /// safetensors' `data_offsets` are relative to the end of its JSON
    /// header. A consumer that guessed "from the start of the file" and a
    /// consumer that guessed "from the data region" would each read
    /// plausible-looking floats, and only one of them would be right.
    ///
    /// So the format crate rebases **once, at parse time** — GGUF stores
    /// `data_offset + info.offset`, safetensors stores `header_len +
    /// data_offsets.0` — and every consumer thereafter writes
    /// `&blob[d.bytes]` with nothing added. Rebasing is a parser's job done
    /// once, not a caller's job done at every use site.
    ///
    /// It follows that [`Self::offset_alignment`] means something: it reports
    /// the alignment of a real address within the mapping, not of a relative
    /// number whose base might itself be misaligned.
    pub bytes: Range<u64>,
}

impl TensorDescriptor {
    /// Width of the declared byte range.
    ///
    /// Saturates rather than panicking on an inverted range, so this is
    /// never a crash — but it is therefore not a fact [`Self::validate`] may
    /// trust, and `validate` checks the ordering itself first.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.bytes.end.saturating_sub(self.bytes.start)
    }

    /// The alignment of this tensor's **start offset within the container's
    /// data region** (AL-1).
    ///
    /// This is a different fact from [`crate::align::alignment_of`], which
    /// takes a materialized slice and therefore answers "where did the
    /// allocator put this buffer" — a property of the run, not of the model
    /// file. Spec §4.6's table is about *file offsets*: GGUF pads tensor
    /// data to `general.alignment` (default 32); safetensors guarantees
    /// nothing, because `data_offsets` are cumulative, so an F32 tensor
    /// following an odd-length U8 tensor starts at an odd offset.
    ///
    /// Reported as a fact so a consumer can plan **before** materializing —
    /// sizing buffers or choosing a decode strategy across
    /// `container.tensors()` without having called `tensor_bytes` yet.
    /// Capped at [`crate::align::MAX_REPORTED_ALIGN`]; an empty range
    /// reports the cap, because it constrains nothing.
    #[must_use]
    pub fn offset_alignment(&self) -> usize {
        if self.bytes.end <= self.bytes.start {
            return MAX_REPORTED_ALIGN;
        }
        if self.bytes.start == 0 {
            return MAX_REPORTED_ALIGN;
        }
        let tz = self
            .bytes
            .start
            .trailing_zeros()
            .min(MAX_REPORTED_ALIGN.trailing_zeros());
        1usize << tz
    }

    /// Whether this tensor's bytes can be borrowed as a typed slice when the
    /// container's data region starts at `base_alignment`.
    ///
    /// Combines the file's own offset (AL-1) with what the encoding needs.
    /// It states a fact; it does not perform the cast, which stays fallible
    /// and explicit (AL-2).
    #[must_use]
    pub fn is_borrowable_at(&self, base_alignment: usize) -> bool {
        let effective = base_alignment.min(self.offset_alignment());
        effective >= self.encoding.alignment()
    }

    /// Check that the declared byte range agrees with shape and encoding.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvertedRange`] if the range ends before it starts,
    /// [`ErrorKind::SizeMismatch`] if the width disagrees with shape and
    /// encoding, [`ErrorKind::RaggedBlock`] if the element count is not a
    /// whole number of blocks, or [`ErrorKind::ShapeOverflow`] /
    /// [`ErrorKind::SizeOverflow`] if the declared arithmetic does not fit
    /// in a `u64`.
    pub fn validate(&self) -> Result<()> {
        // Ordering first, and before `byte_len` gets to invent a number.
        // A zero-element tensor is legal in safetensors (shape [0, 768],
        // data_offsets [x, x]), so `4096..1024` is exactly the shape a real
        // parser produces from swapped or truncated offsets — and it used to
        // validate as Ok, after which a container does `&blob[4096..1024]`
        // and panics inside somebody else's crate.
        if self.bytes.end < self.bytes.start {
            return Err(Error::from(ErrorKind::InvertedRange {
                name: self.name.clone(),
                start: self.bytes.start,
                end: self.bytes.end,
            }));
        }
        let expected = self
            .encoding
            .byte_size(self.shape.elem_count()?, &self.name)?;
        let actual = self.byte_len();
        if expected != actual {
            return Err(Error::from(ErrorKind::SizeMismatch {
                name: self.name.clone(),
                expected,
                actual,
            }));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockSpec, DType, Encoding, ErrorKind, Shape};

    fn dense_f32(name: &str, dims: [usize; 2], start: u64, end: u64) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            shape: Shape::new(dims),
            encoding: Encoding::Dense(DType::F32),
            bytes: start..end,
        }
    }

    /// Q4_0: 32 weights per block, 18 bytes per block.
    fn q4_0() -> BlockSpec {
        BlockSpec {
            family: "ggml",
            code: 2,
            elements_per_block: 32,
            bytes_per_block: 18,
            alignment: 2,
        }
    }

    fn blocked(name: &str, dims: [usize; 1], start: u64, end: u64) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            shape: Shape::new(dims),
            encoding: Encoding::Blocked(q4_0()),
            bytes: start..end,
        }
    }

    #[test]
    fn byte_len_is_the_declared_range_width() {
        let d = dense_f32("t", [2, 3], 100, 124);
        assert_eq!(d.byte_len(), 24);
    }

    #[test]
    fn validate_accepts_a_consistent_descriptor() {
        // 2 x 3 f32 = 6 elements x 4 bytes = 24
        let d = dense_f32("t", [2, 3], 100, 124);
        d.validate().expect("consistent");
    }

    #[test]
    fn validate_rejects_a_range_that_disagrees_with_shape() {
        let d = dense_f32("model.embed_tokens.weight", [2, 3], 100, 120);
        let err = d.validate().unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::SizeMismatch { .. }));
        assert!(err.to_string().contains("model.embed_tokens.weight"));
    }

    #[test]
    fn validate_accepts_a_consistent_blocked_descriptor() {
        // 4096 weights = 128 blocks x 18 bytes = 2304. With only dense
        // fixtures, half of `Encoding` was untested through its main
        // consumer.
        blocked("blk.0.ffn_gate.weight", [4096], 0, 2304)
            .validate()
            .expect("consistent");
    }

    #[test]
    fn validate_propagates_ragged_block_from_a_quantized_tensor() {
        // `validate`'s own doc comment promised this propagation and no test
        // asserted it, so replacing `?` with `.unwrap_or(actual)` left the
        // suite green — downgrading a spec §7 FATAL unknown to silence. A
        // Q4_0 tensor whose shape is not a multiple of 32 would then pass
        // validation, and the consumer would read a byte range with no
        // defined block count.
        let d = blocked("blk.0.attn_q.weight", [33], 0, 18);
        let err = d.validate().unwrap_err();
        match err.kind() {
            ErrorKind::RaggedBlock {
                name,
                elem_count,
                elements_per_block,
            } => {
                assert_eq!(name, "blk.0.attn_q.weight");
                assert_eq!(*elem_count, 33);
                assert_eq!(*elements_per_block, 32);
            }
            other => panic!("wrong kind: {other:?}"),
        }
        assert!(err.to_string().contains("blk.0.attn_q.weight"), "{err}");
    }

    #[test]
    fn validate_rejects_an_inverted_range() {
        let d = dense_f32("real.weight", [2, 3], 124, 100);
        let err = d.validate().unwrap_err();
        match err.kind() {
            ErrorKind::InvertedRange { name, start, end } => {
                assert_eq!(name, "real.weight");
                assert_eq!(*start, 124);
                assert_eq!(*end, 100);
            }
            // The old behaviour: `saturating_sub` manufactured 0 and the
            // message said "byte range is 0 bytes", which is neither true
            // nor useful — it sends the reader after a missing tensor
            // rather than a swapped pair of offsets.
            other => panic!("wrong kind: {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_an_inverted_range_even_for_a_zero_element_tensor() {
        // The case that used to return Ok: elem_count is 0, byte_len
        // saturates to 0, so the two agreed and a structurally impossible
        // range validated.
        let d = dense_f32("empty.weight", [0, 768], 4096, 1024);
        assert!(matches!(
            d.validate().unwrap_err().kind(),
            ErrorKind::InvertedRange { .. }
        ));
    }

    #[test]
    fn validate_refuses_a_shape_that_overflows_rather_than_accepting_zero() {
        let d = TensorDescriptor {
            name: "huge.weight".into(),
            shape: Shape::new([1usize << 32, 1usize << 32]),
            encoding: Encoding::Dense(DType::F32),
            bytes: 0..0,
        };
        assert!(matches!(
            d.validate().unwrap_err().kind(),
            ErrorKind::ShapeOverflow { .. }
        ));
    }

    #[test]
    fn offset_alignment_is_a_fact_about_the_file_not_the_allocator() {
        // AL-1 at the descriptor level. Nothing in the crate looked at
        // `bytes.start`, which is the fact spec §4.6's whole table is about.
        // Exact values, not `>= 1` and `is_power_of_two()`: that shape of
        // assertion is satisfied by every constant a body could return.
        assert_eq!(dense_f32("t", [1, 1], 33, 37).offset_alignment(), 1);
        assert_eq!(dense_f32("t", [1, 1], 34, 38).offset_alignment(), 2);
        assert_eq!(dense_f32("t", [1, 1], 36, 40).offset_alignment(), 4);
        assert_eq!(dense_f32("t", [1, 1], 40, 44).offset_alignment(), 8);
        // GGUF pads tensor data to general.alignment, default 32.
        assert_eq!(dense_f32("t", [1, 1], 32, 36).offset_alignment(), 32);
        // Capped, and an empty or absent offset constrains nothing.
        assert_eq!(dense_f32("t", [1, 1], 4096, 4100).offset_alignment(), 64);
        assert_eq!(dense_f32("t", [0, 0], 33, 33).offset_alignment(), 64);
    }

    #[test]
    fn borrowability_combines_the_file_offset_with_the_encoding() {
        // The safetensors hazard, stated as a fact rather than discovered as
        // undefined behaviour: an F32 tensor following an odd-length U8
        // tensor cannot be borrowed however well-aligned the mapping is.
        let odd = dense_f32("after.u8.tensor", [1, 1], 33, 37);
        assert!(!odd.is_borrowable_at(4096));
        let even = dense_f32("aligned", [1, 1], 36, 40);
        assert!(even.is_borrowable_at(4096));
        // A page-aligned base does not rescue a bad offset, and a bad base
        // does not survive a good offset.
        assert!(!even.is_borrowable_at(2));
        // A Q4_0 block is { f16, [u8; 16] }: alignment 2, so it needs less.
        assert!(blocked("q", [32], 34, 52).is_borrowable_at(4096));
    }
}
