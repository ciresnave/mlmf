// SPDX-License-Identifier: MIT OR Apache-2.0
//! Validate that a set of GGUF files is exactly one complete `gguf-split`
//! model (ROADMAP S-63, slice 1).
//!
//! Pure: it takes facts the caller already read from each shard (its file
//! name, its [`SplitInfo`], its tensor count) and says whether they describe
//! one complete set, and in which order the shards belong. It opens nothing,
//! and it supplies nothing a shard did not declare: a missing
//! `split.tensors.count` stays unchecked rather than being guessed.
//!
//! Every refusal is a distinct [`ShardSetError`], never a repaired set: a
//! missing shard is reported as missing, not skipped.

use crate::split::{SplitFileName, SplitInfo};

/// What the caller read from one shard file.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ShardFacts {
    /// The shard's file name, parsed.
    pub name: SplitFileName,
    /// What the shard's own metadata declares about its place.
    pub info: SplitInfo,
    /// How many tensors this shard's directory lists.
    pub tensor_count: u64,
}

impl ShardFacts {
    /// Facts for one shard.
    #[must_use]
    pub fn new(name: SplitFileName, info: SplitInfo, tensor_count: u64) -> Self {
        Self {
            name,
            info,
            tensor_count,
        }
    }
}

/// Why a set of shards is not one complete split model.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShardSetError {
    /// No shards were given.
    Empty,
    /// Shard `index` (position in the input) names a different prefix from
    /// the first shard.
    PrefixDiffers {
        /// Position in the input.
        index: usize,
    },
    /// Shard `index` declares a `split.count` different from the first
    /// shard's.
    CountDiffers {
        /// Position in the input.
        index: usize,
        /// The first shard's `split.count`.
        expected: u16,
        /// This shard's `split.count`.
        found: u16,
    },
    /// Shard `index`'s file name disagrees with its own `split.no` /
    /// `split.count`.
    NameDisagreesWithMetadata {
        /// Position in the input.
        index: usize,
    },
    /// Two shards declare the same zero-based `split.no`.
    DuplicateNo {
        /// The repeated `split.no`.
        no: u16,
    },
    /// No shard declares this zero-based `split.no`.
    MissingShard {
        /// The absent `split.no`.
        no: u16,
    },
    /// A shard declares a `split.tensors.count` that is not the sum of the
    /// shards' tensor counts.
    TensorTotalDiffers {
        /// Position in the input of the declaring shard.
        index: usize,
        /// What it declared.
        declared: u32,
        /// The sum over the given shards.
        found: u64,
    },
}

impl std::fmt::Display for ShardSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("no shards given"),
            Self::PrefixDiffers { index } => {
                write!(f, "shard {index} has a different file-name prefix")
            }
            Self::CountDiffers {
                index,
                expected,
                found,
            } => write!(
                f,
                "shard {index} declares split.count {found}, the first shard declares {expected}"
            ),
            Self::NameDisagreesWithMetadata { index } => write!(
                f,
                "shard {index}: file name numbering disagrees with its split.no / split.count"
            ),
            Self::DuplicateNo { no } => write!(f, "two shards declare split.no {no}"),
            Self::MissingShard { no } => write!(f, "no shard declares split.no {no}"),
            Self::TensorTotalDiffers {
                index,
                declared,
                found,
            } => write!(
                f,
                "shard {index} declares split.tensors.count {declared}, the shards list {found}"
            ),
        }
    }
}

impl std::error::Error for ShardSetError {}

/// Check one shard against the first, and place it in its slot.
fn place(
    slots: &mut [Option<usize>],
    shards: &[ShardFacts],
    index: usize,
) -> Result<(), ShardSetError> {
    let (first, s) = (&shards[0], &shards[index]);
    if s.name.prefix != first.name.prefix {
        return Err(ShardSetError::PrefixDiffers { index });
    }
    if s.info.count != first.info.count {
        return Err(ShardSetError::CountDiffers {
            index,
            expected: first.info.count,
            found: s.info.count,
        });
    }
    if !s.info.matches_file_name(&s.name) {
        return Err(ShardSetError::NameDisagreesWithMetadata { index });
    }
    // `no < count` is guaranteed by `SplitInfo::from_source`, and `count`
    // is the slot length, but a hand-built `SplitInfo` is not so guarded.
    let slot = slots
        .get_mut(usize::from(s.info.no))
        .ok_or(ShardSetError::NameDisagreesWithMetadata { index })?;
    if slot.is_some() {
        return Err(ShardSetError::DuplicateNo { no: s.info.no });
    }
    *slot = Some(index);
    Ok(())
}

/// Every declared `split.tensors.count` must equal the sum of the shards'
/// own tensor counts. A shard that declares none is not checked.
fn check_total(shards: &[ShardFacts]) -> Result<(), ShardSetError> {
    let found: u64 = shards.iter().map(|s| s.tensor_count).sum();
    for (index, s) in shards.iter().enumerate() {
        if let Some(declared) = s.info.tensors_total
            && u64::from(declared) != found
        {
            return Err(ShardSetError::TensorTotalDiffers {
                index,
                declared,
                found,
            });
        }
    }
    Ok(())
}

/// Confirm `shards` are exactly one complete split model and return their
/// input positions in shard order: element `n` is the position of the shard
/// whose `split.no` is `n`.
///
/// # Errors
///
/// [`ShardSetError`]: the first inconsistency found, in the order prefix,
/// count, name numbering, duplicate, then (after every shard is placed)
/// missing shard, then tensor total.
pub fn validate_shard_set(shards: &[ShardFacts]) -> Result<Vec<usize>, ShardSetError> {
    let first = shards.first().ok_or(ShardSetError::Empty)?;
    let mut slots = vec![None; usize::from(first.info.count)];
    for index in 0..shards.len() {
        place(&mut slots, shards, index)?;
    }
    let order = slots
        .iter()
        .enumerate()
        .map(|(no, slot)| {
            slot.ok_or(ShardSetError::MissingShard {
                no: u16::try_from(no).unwrap_or(u16::MAX),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    check_total(shards)?;
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(prefix: &str, no: u16, count: u16, tensors: u64, total: Option<u32>) -> ShardFacts {
        ShardFacts::new(
            SplitFileName {
                prefix: prefix.to_string(),
                number: u32::from(no) + 1,
                count: u32::from(count),
            },
            SplitInfo {
                no,
                count,
                tensors_total: total,
            },
            tensors,
        )
    }

    fn three(total: Option<u32>) -> Vec<ShardFacts> {
        vec![
            facts("m", 0, 3, 10, total),
            facts("m", 1, 3, 20, total),
            facts("m", 2, 3, 5, total),
        ]
    }

    #[test]
    fn a_complete_set_in_order_validates() {
        assert_eq!(validate_shard_set(&three(Some(35))), Ok(vec![0, 1, 2]));
    }

    #[test]
    fn input_order_does_not_matter_and_the_result_says_where_each_shard_is() {
        let mut v = three(Some(35));
        v.swap(0, 2); // input is now no 2, 1, 0
        assert_eq!(validate_shard_set(&v), Ok(vec![2, 1, 0]));
    }

    #[test]
    fn an_empty_input_is_refused() {
        assert_eq!(validate_shard_set(&[]), Err(ShardSetError::Empty));
    }

    #[test]
    fn a_missing_shard_is_named_not_skipped() {
        let mut v = three(Some(35));
        v.remove(1);
        assert_eq!(
            validate_shard_set(&v),
            Err(ShardSetError::MissingShard { no: 1 })
        );
    }

    #[test]
    fn a_duplicate_no_is_refused() {
        let mut v = three(None);
        v[2] = facts("m", 1, 3, 5, None);
        assert_eq!(
            validate_shard_set(&v),
            Err(ShardSetError::DuplicateNo { no: 1 })
        );
    }

    #[test]
    fn a_different_prefix_is_refused() {
        let mut v = three(None);
        v[1] = facts("other", 1, 3, 20, None);
        assert_eq!(
            validate_shard_set(&v),
            Err(ShardSetError::PrefixDiffers { index: 1 })
        );
    }

    #[test]
    fn a_count_that_differs_from_the_first_shard_is_refused() {
        let mut v = three(None);
        v[2] = facts("m", 2, 4, 5, None);
        assert_eq!(
            validate_shard_set(&v),
            Err(ShardSetError::CountDiffers {
                index: 2,
                expected: 3,
                found: 4
            })
        );
    }

    #[test]
    fn a_name_that_disagrees_with_split_no_is_refused() {
        let mut v = three(None);
        v[1].name.number = 3; // name says shard 3, metadata says no 1
        assert_eq!(
            validate_shard_set(&v),
            Err(ShardSetError::NameDisagreesWithMetadata { index: 1 })
        );
    }

    #[test]
    fn a_tensor_total_that_is_not_the_sum_is_refused() {
        assert_eq!(
            validate_shard_set(&three(Some(36))),
            Err(ShardSetError::TensorTotalDiffers {
                index: 0,
                declared: 36,
                found: 35
            })
        );
    }

    #[test]
    fn an_undeclared_tensor_total_is_not_guessed() {
        // Control for the test above: the same shards with no declared
        // total pass, so the refusal there is about the declared value.
        assert_eq!(validate_shard_set(&three(None)), Ok(vec![0, 1, 2]));
    }

    #[test]
    fn a_hand_built_no_past_count_is_an_error_not_a_panic() {
        let mut v = three(None);
        v[2].info.no = 7;
        v[2].name.number = 8;
        assert_eq!(
            validate_shard_set(&v),
            Err(ShardSetError::NameDisagreesWithMetadata { index: 2 })
        );
    }
}
