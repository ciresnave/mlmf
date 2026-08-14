//! What a model file declares about one tensor.

use std::ops::Range;

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
    /// Byte range within the container's data region.
    pub bytes: Range<u64>,
}

impl TensorDescriptor {
    /// Width of the declared byte range.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.bytes.end.saturating_sub(self.bytes.start)
    }

    /// Check that the declared byte range agrees with shape and encoding.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::SizeMismatch`] if they disagree, or
    /// [`ErrorKind::RaggedBlock`] if the element count is not a whole
    /// number of blocks.
    pub fn validate(&self) -> Result<()> {
        let expected = self
            .encoding
            .byte_size(self.shape.elem_count(), &self.name)?;
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
    use crate::{DType, Encoding, ErrorKind, Shape};

    fn dense_f32(name: &str, dims: [usize; 2], start: u64, end: u64) -> TensorDescriptor {
        TensorDescriptor {
            name: name.to_string(),
            shape: Shape::new(dims),
            encoding: Encoding::Dense(DType::F32),
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
}
