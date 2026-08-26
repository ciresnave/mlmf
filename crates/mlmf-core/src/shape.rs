//! Tensor dimensions, stored exactly as the file declared them.

use smallvec::SmallVec;

use crate::{Error, ErrorKind, Result};

/// Dimensions in **declared order**.
///
/// GGUF declares dims in the opposite order from HuggingFace state dicts.
/// Core never reorders: a crate that silently "helpfully" reverses dims is
/// the same class of defect as flattening rope scaling (spec §4.2).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(SmallVec<[usize; 4]>);

impl Shape {
    /// Build a shape from dimensions in declared order.
    #[must_use]
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Self(dims.into_iter().collect())
    }

    /// Dimensions, in declared order.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Number of dimensions. A scalar has rank 0.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total element count. A rank-0 shape has exactly one element.
    ///
    /// Returns `u64` because an element count can exceed `usize` on 32-bit
    /// targets while still describing a valid file.
    ///
    /// **Fallible, and the fallible version owns the plain name.** The
    /// unchecked `product()` this replaced wrapped silently in release —
    /// `[2^32, 2^32]` reported `0`, `byte_size` then reported `Ok(0)`, and
    /// `TensorDescriptor::validate` accepted a tensor declaring 2^64
    /// elements as a zero-byte tensor — and panicked in debug from inside
    /// `core::iter::traits::accum`, so a fuzz backtrace did not even name
    /// this file. Both GGUF (`n_dims` × `u64`) and safetensors (a JSON
    /// number array) let an untrusted file declare it, and neither format
    /// bounds the product.
    ///
    /// Adding a separate `checked_elem_count()` and leaving this infallible
    /// was rejected for the reason AL-2 gives: that leaves the footgun
    /// holding the obvious name, so the common case and the rare case go
    /// down different code paths and the rare one is reached only in
    /// production.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::ShapeOverflow`] if the dimensions multiply out beyond
    /// `u64`. Spec §7's fatal tier: the byte size is genuinely unknowable,
    /// so proceeding would hand out wrong bytes rather than incomplete ones.
    pub fn elem_count(&self) -> Result<u64> {
        // `try_fold` with init 1 returns Ok(1) for an empty iterator, so the
        // rank-0 = 1 empty-product property is preserved exactly.
        self.0.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d as u64).ok_or_else(|| {
                Error::from(ErrorKind::ShapeOverflow {
                    dims: self.0.to_vec(),
                })
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn dims_are_preserved_exactly_as_declared() {
        // GGUF declares dims in the opposite order from HF state dicts.
        // Core stores what it was given; reversal is an explicit call
        // in mlmf-ggml, never an implicit courtesy (spec §4.2).
        let shape = Shape::new([4096usize, 11008]);
        assert_eq!(shape.dims(), &[4096, 11008]);
        assert_eq!(shape.rank(), 2);
    }

    #[test]
    fn elem_count_multiplies_all_dims() {
        assert_eq!(
            Shape::new([4096usize, 11008]).elem_count().unwrap(),
            45_088_768
        );
    }

    #[test]
    fn a_scalar_has_rank_zero_and_one_element() {
        let shape = Shape::new([]);
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.elem_count().unwrap(), 1);
    }

    #[test]
    fn a_zero_dim_yields_zero_elements() {
        assert_eq!(Shape::new([0usize, 128]).elem_count().unwrap(), 0);
    }

    #[test]
    fn a_dim_product_that_overflows_is_refused_not_wrapped() {
        // Release wrapped this to 0 and debug panicked from inside libcore.
        // Neither is acceptable for a library parsing files fetched from
        // public hubs: a wrap hands out a wrong byte size with core's
        // authority behind it, and a panic is a denial of service in
        // Lightbulb's serving path, which can refuse to serve on a non-empty
        // Report but cannot refuse a panic.
        let s = Shape::new([1usize << 32, 1usize << 32]);
        let err = s.elem_count().unwrap_err();
        match err.kind() {
            ErrorKind::ShapeOverflow { dims } => {
                assert_eq!(dims, &[1usize << 32, 1usize << 32]);
            }
            other => panic!("wrong kind: {other:?}"),
        }
    }

    #[test]
    fn overflow_is_not_reached_by_any_plausible_real_model() {
        // Measured headroom for a 70B f32 model is ~2.8e11 bytes against
        // u64::MAX = 1.8e19. Well-formedness is exactly what a parser cannot
        // assume, but the check must not fire on real files either.
        let seventy_b = Shape::new([70_000_000_000usize]);
        assert_eq!(seventy_b.elem_count().unwrap(), 70_000_000_000);
    }
}
