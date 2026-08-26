//! How a tensor's elements are laid out in bytes.
//!
//! GGUF and safetensors are **siblings, not a stack**: safetensors stores
//! dense arrays of a scalar type, GGUF stores sequences of quantization
//! blocks. Size arithmetic therefore differs in kind, not degree, and
//! lives here so no consumer re-derives it and gets it subtly wrong.

use crate::{DType, Error, ErrorKind, Result};

/// Geometry of one quantization block family member.
///
/// New schemes are added as **data** (another `BlockSpec` value), never as
/// another [`Encoding`] variant — otherwise a superset of N formats forces
/// a widening that ripples through every consumer (spec §4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockSpec {
    /// Family that defines the code space, e.g. `"ggml"`.
    pub family: &'static str,
    /// The file's **own declared type id**, passed through untouched so a
    /// consumer can map it back to its own type and pick a kernel.
    pub code: u32,
    /// Elements packed into one block, e.g. 32 for Q4_0, 256 for K-quants.
    pub elements_per_block: u64,
    /// Bytes one block occupies, e.g. 18 for Q4_0.
    pub bytes_per_block: u64,
    /// Alignment a consumer needs to reinterpret these blocks as structs.
    pub alignment: usize,
}

/// Dense scalars, or block-quantized data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// `bytes = elem_count * dtype.size()`
    Dense(DType),
    /// `bytes = elem_count / elements_per_block * bytes_per_block`
    Blocked(BlockSpec),
}

impl Encoding {
    /// Bytes occupied by `elem_count` elements in this encoding.
    ///
    /// `tensor_name` is used only for error attribution.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::RaggedBlock`] if a blocked encoding's element count is
    /// not a whole number of blocks. This is refused rather than rounded:
    /// a partial block has no defined byte length, so any answer would be
    /// invented.
    ///
    /// [`ErrorKind::SizeOverflow`] if the product exceeds `u64`. Both arms
    /// multiplied unchecked before, which wrapped silently in release and
    /// panicked in debug. The blocked arm's overflow is not hypothetical:
    /// it occurs whenever `bytes_per_block > elements_per_block`, which is
    /// true of real ggml types — Q8_0 (32/34), Q5_1 (32/24), Q8_K (256/292).
    /// Q4_0 (32/18) happens to be safe, so a suite that tests only Q4_0 —
    /// which this one did — cannot find it. A consumer that trusts a wrapped
    /// `byte_size` to size a slice or advance an offset gets exactly the
    /// subtle wrongness this module exists to prevent.
    pub fn byte_size(&self, elem_count: u64, tensor_name: &str) -> Result<u64> {
        let overflow = || {
            Error::from(ErrorKind::SizeOverflow {
                name: tensor_name.to_string(),
                elem_count,
            })
        };
        match self {
            Encoding::Dense(dt) => elem_count
                .checked_mul(dt.size() as u64)
                .ok_or_else(overflow),
            Encoding::Blocked(spec) => {
                if !elem_count.is_multiple_of(spec.elements_per_block) {
                    return Err(Error::from(ErrorKind::RaggedBlock {
                        name: tensor_name.to_string(),
                        elem_count,
                        elements_per_block: spec.elements_per_block,
                    }));
                }
                // The division stays before the multiply: that ordering is
                // already correct and load-bearing, and it reduces (though
                // it cannot eliminate) the overflow range.
                (elem_count / spec.elements_per_block)
                    .checked_mul(spec.bytes_per_block)
                    .ok_or_else(overflow)
            }
        }
    }

    /// Alignment a consumer needs to reinterpret bytes in this encoding.
    #[must_use]
    pub fn alignment(&self) -> usize {
        match self {
            Encoding::Dense(dt) => dt.alignment(),
            Encoding::Blocked(spec) => spec.alignment,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    /// Q4_0: 32 weights per block, 18 bytes per block
    /// (a 2-byte f16 scale plus 16 bytes of nibbles).
    fn q4_0() -> BlockSpec {
        BlockSpec {
            family: "ggml",
            code: 2,
            elements_per_block: 32,
            bytes_per_block: 18,
            alignment: 2,
        }
    }

    #[test]
    fn dense_size_is_a_simple_product() {
        let enc = Encoding::Dense(DType::F32);
        assert_eq!(enc.byte_size(1000, "t").unwrap(), 4000);
    }

    #[test]
    fn blocked_size_is_not_a_simple_product() {
        // 4096 weights = 128 blocks x 18 bytes = 2304 bytes.
        // A naive nelements * dtype_size would give a different answer,
        // which is the whole reason this lives in core.
        let enc = Encoding::Blocked(q4_0());
        assert_eq!(enc.byte_size(4096, "t").unwrap(), 2304);
    }

    #[test]
    fn a_ragged_element_count_is_an_error_not_a_rounding() {
        let enc = Encoding::Blocked(q4_0());
        let err = enc.byte_size(33, "blk.0.attn_q.weight").unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::RaggedBlock { .. }));
        assert!(err.to_string().contains("blk.0.attn_q.weight"));
    }

    #[test]
    fn zero_elements_is_zero_bytes_for_both_forms() {
        assert_eq!(Encoding::Dense(DType::F32).byte_size(0, "t").unwrap(), 0);
        assert_eq!(Encoding::Blocked(q4_0()).byte_size(0, "t").unwrap(), 0);
    }

    #[test]
    fn alignment_comes_from_the_dtype_or_the_block() {
        assert_eq!(Encoding::Dense(DType::F32).alignment(), 4);
        // A repr(C) Q4_0 block is { f16, [u8; 16] }: size 18, align 2.
        assert_eq!(Encoding::Blocked(q4_0()).alignment(), 2);
    }

    /// Q8_0: 32 weights per block, 34 bytes per block. Chosen deliberately
    /// for the overflow test because `bytes_per_block > elements_per_block`,
    /// which Q4_0 (32/18) does not satisfy — Q4_0's worst case is
    /// `u64::MAX/32 * 18 = 1.04e19 < u64::MAX`, so it cannot overflow this
    /// path and testing only Q4_0 could never find the defect.
    fn q8_0() -> BlockSpec {
        BlockSpec {
            family: "ggml",
            code: 8,
            elements_per_block: 32,
            bytes_per_block: 34,
            alignment: 2,
        }
    }

    #[test]
    fn a_dense_size_that_overflows_is_refused_not_wrapped() {
        let enc = Encoding::Dense(DType::F64);
        let err = enc.byte_size(u64::MAX / 4, "big.weight").unwrap_err();
        match err.kind() {
            ErrorKind::SizeOverflow { name, elem_count } => {
                assert_eq!(name, "big.weight");
                assert_eq!(*elem_count, u64::MAX / 4);
            }
            other => panic!("wrong kind: {other:?}"),
        }
        assert!(err.to_string().contains("big.weight"), "{err}");
    }

    #[test]
    fn a_blocked_size_that_overflows_is_refused_not_wrapped() {
        // (u64::MAX / 32) * 34 is 1.96e19; a wrap reported 1.15e18, a silent
        // 17x understatement of the tensor's byte length.
        let enc = Encoding::Blocked(q8_0());
        let elems = (u64::MAX / 32) * 32;
        let err = enc.byte_size(elems, "blk.0.ffn_down.weight").unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::SizeOverflow { .. }));
        assert!(err.to_string().contains("blk.0.ffn_down.weight"), "{err}");
    }

    #[test]
    fn q8_0_still_computes_ordinary_sizes() {
        // The overflow guard must not fire on a real tensor.
        assert_eq!(Encoding::Blocked(q8_0()).byte_size(4096, "t").unwrap(), 4352);
    }

    #[test]
    fn a_new_block_scheme_is_a_row_not_a_variant() {
        // Spec §4.1: extend by data. Adding a scheme must not require
        // touching this enum, so an unknown family still computes.
        let hypothetical = BlockSpec {
            family: "future",
            code: 999,
            elements_per_block: 64,
            bytes_per_block: 40,
            alignment: 4,
        };
        let enc = Encoding::Blocked(hypothetical);
        assert_eq!(enc.byte_size(128, "t").unwrap(), 80);
        assert_eq!(enc.alignment(), 4);
    }
}
