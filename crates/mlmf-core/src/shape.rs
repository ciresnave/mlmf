//! Tensor dimensions, stored exactly as the file declared them.

use smallvec::SmallVec;

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
    #[must_use]
    pub fn elem_count(&self) -> u64 {
        self.0.iter().map(|&d| d as u64).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(Shape::new([4096usize, 11008]).elem_count(), 45_088_768);
    }

    #[test]
    fn a_scalar_has_rank_zero_and_one_element() {
        let shape = Shape::new([]);
        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.elem_count(), 1);
    }

    #[test]
    fn a_zero_dim_yields_zero_elements() {
        assert_eq!(Shape::new([0usize, 128]).elem_count(), 0);
    }
}
