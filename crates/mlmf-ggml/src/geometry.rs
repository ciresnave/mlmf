//! Size arithmetic, with ggml's divisibility rule rather than the weaker
//! whole-tensor one.

use mlmf_core::{BlockSpec, DType, Encoding, Error, ErrorKind, Result};

use crate::GgmlType;

impl GgmlType {
    /// This type as `mlmf-core` describes encodings.
    ///
    /// Dense types become [`Encoding::Dense`]; everything else becomes a
    /// [`BlockSpec`] **carrying the ggml code**, which is the only thing
    /// separating types with identical geometry — `Q4_0` and `IQ4_NL` are
    /// both 32 elements in 18 bytes and are not interchangeable.
    #[must_use]
    pub fn encoding(self) -> Encoding {
        let dense = match self {
            GgmlType::F32 => Some(DType::F32),
            GgmlType::F16 => Some(DType::F16),
            GgmlType::BF16 => Some(DType::BF16),
            GgmlType::F64 => Some(DType::F64),
            GgmlType::I8 => Some(DType::I8),
            GgmlType::I16 => Some(DType::I16),
            GgmlType::I32 => Some(DType::I32),
            GgmlType::I64 => Some(DType::I64),
            _ => None,
        };
        match dense {
            Some(dt) => Encoding::Dense(dt),
            None => Encoding::Blocked(BlockSpec {
                family: "ggml",
                code: self.code(),
                elements_per_block: self.elements_per_block(),
                bytes_per_block: self.bytes_per_block(),
                alignment: self.alignment(),
            }),
        }
    }

    /// Bytes a tensor of shape `ne` occupies in this type.
    ///
    /// `ne` is in **declared order**, first dimension first — the same
    /// order GGUF writes it and the order `ggml_tensor::ne` uses. The
    /// first entry is the row length, and ggml requires it to be a whole
    /// number of blocks (`gguf.cpp`, tensor-info validation). That rule is
    /// stronger than whole-tensor divisibility and is checked here because
    /// `mlmf-core` cannot know an `Encoding` describes a row-major
    /// quantized layout.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::RaggedRow`] if the first dimension is not a whole
    /// number of blocks; [`ErrorKind::ShapeOverflow`] or
    /// [`ErrorKind::SizeOverflow`] if the arithmetic does not fit a `u64`.
    pub fn nbytes(self, ne: &[u64], name: &str) -> Result<u64> {
        let per_block = self.elements_per_block();
        if let Some(&row) = ne.first()
            && per_block > 1
            && !row.is_multiple_of(per_block)
        {
            return Err(Error::from(ErrorKind::RaggedRow {
                name: name.to_string(),
                row_len: row,
                elements_per_block: per_block,
            }));
        }

        let mut elems: u64 = 1;
        for &d in ne {
            elems = elems.checked_mul(d).ok_or_else(|| {
                Error::from(ErrorKind::ShapeOverflow {
                    // Core declares `dims: Vec<usize>`; GGUF declares
                    // dimensions as `u64`. On every 64-bit target the
                    // conversion is exact. It is saturated rather than
                    // unwrapped because this is already the error path and
                    // a panic here would replace a describable failure with
                    // an undescribable one.
                    dims: ne
                        .iter()
                        .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
                        .collect(),
                })
            })?;
        }

        self.encoding().byte_size(elems, name)
    }
}

#[cfg(test)]
mod tests {
    use crate::GgmlType;
    use mlmf_core::{DType, Encoding, ErrorKind};

    #[test]
    fn a_dense_type_projects_to_a_dense_encoding() {
        assert_eq!(GgmlType::F32.encoding(), Encoding::Dense(DType::F32));
        assert_eq!(GgmlType::BF16.encoding(), Encoding::Dense(DType::BF16));
        assert_eq!(GgmlType::I64.encoding(), Encoding::Dense(DType::I64));
    }

    #[test]
    fn a_quantized_type_projects_to_a_block_spec_carrying_its_code() {
        // The code must survive the projection: Q4_0 and IQ4_NL have
        // identical geometry, so a consumer picking a kernel has nothing
        // else to go on.
        let Encoding::Blocked(spec) = GgmlType::IQ4_NL.encoding() else {
            panic!("IQ4_NL is quantized");
        };
        assert_eq!(spec.family, "ggml");
        assert_eq!(spec.code, 20);
        assert_eq!(spec.elements_per_block, 32);
        assert_eq!(spec.bytes_per_block, 18);
        assert_eq!(spec.alignment, 2);

        let Encoding::Blocked(q4_0) = GgmlType::Q4_0.encoding() else {
            panic!("Q4_0 is quantized");
        };
        assert_eq!(q4_0.code, 2);
        assert_ne!(spec, q4_0, "identical geometry must not collapse");
    }

    #[test]
    fn nbytes_matches_the_block_formula() {
        // 576 x 192 = 110592 elements / 32 per block = 3456 blocks
        // x 18 bytes = 62208. Taken from a real file:
        // SmolLM2-135M-Instruct-Q4_0.gguf, blk.0.attn_k.weight.
        assert_eq!(
            GgmlType::Q4_0.nbytes(&[576, 192], "blk.0.attn_k.weight").unwrap(),
            62208
        );
        assert_eq!(GgmlType::F32.nbytes(&[576], "t").unwrap(), 2304);
    }

    #[test]
    fn a_row_that_is_not_a_whole_number_of_blocks_is_refused() {
        // THE point of this module. [16, 64] has 1024 elements — a clean 32
        // blocks — so a whole-tensor divisibility check accepts it. ggml
        // requires each ROW to divide, and 16 does not divide by 32. No
        // ggml writer produces this layout, so any byte count computed for
        // it describes a file that does not exist.
        let err = GgmlType::Q4_0
            .nbytes(&[16, 64], "blk.0.attn_q.weight")
            .unwrap_err();
        match err.kind() {
            ErrorKind::RaggedRow {
                name,
                row_len,
                elements_per_block,
            } => {
                assert_eq!(name, "blk.0.attn_q.weight");
                assert_eq!(*row_len, 16);
                assert_eq!(*elements_per_block, 32);
            }
            other => panic!("wrong kind: {other:?}"),
        }
    }

    #[test]
    fn the_row_rule_does_not_fire_on_dense_types() {
        // elements_per_block is 1, so every row divides. A guard that
        // applied unconditionally would reject every F32 tensor with an
        // odd first dimension, which is most of them.
        assert_eq!(GgmlType::F32.nbytes(&[17, 3], "t").unwrap(), 204);
    }

    #[test]
    fn a_scalar_and_an_empty_shape_are_handled_without_inventing_a_row() {
        // ne = [] is how a rank-0 tensor arrives. There is no first
        // dimension to check, and the element count is 1.
        assert_eq!(GgmlType::F32.nbytes(&[], "t").unwrap(), 4);
        // A zero-length dimension yields zero bytes, not an error: GGUF
        // permits it and the range is simply empty.
        assert_eq!(GgmlType::Q4_0.nbytes(&[0, 64], "t").unwrap(), 0);
    }

    #[test]
    fn an_overflowing_shape_is_refused_rather_than_wrapped() {
        // The row length must itself be a whole number of blocks, or the
        // RaggedRow guard fires first and this tests the wrong thing.
        // (u64::MAX / 32) * 32 is the largest multiple of 32 in a u64, so
        // it passes the guard and then overflows on the second dimension.
        let row = (u64::MAX / 32) * 32;
        assert_eq!(row % 32, 0, "the fixture must clear the row guard");
        let err = GgmlType::Q8_0.nbytes(&[row, 64], "big.weight").unwrap_err();
        assert!(
            matches!(err.kind(), ErrorKind::ShapeOverflow { .. }),
            "got {:?}",
            err.kind()
        );
    }

    #[test]
    fn every_live_type_computes_a_size_for_a_shape_it_permits() {
        // Coverage rather than spot-checks: a row whose elements_per_block
        // is wrong by a factor of 8 is caught here even if no fixture
        // names it.
        for t in GgmlType::ALL {
            let row = t.elements_per_block();
            let n = t.nbytes(&[row, 2], t.name()).unwrap_or_else(|e| {
                panic!("{} failed on its own block size: {e}", t.name())
            });
            assert_eq!(n, 2 * t.bytes_per_block(), "{}", t.name());
        }
    }
}
