// SPDX-License-Identifier: MIT OR Apache-2.0
//! Split (sharded) GGUF files: the file-name convention and the `split.*`
//! keys, as pure functions over strings and metadata. No I/O.
//!
//! llama.cpp's `gguf-split` tool writes a model as several `.gguf` files.
//! Reading one means knowing which files to open and checking each against
//! what it declares. This module supplies those facts and nothing else:
//! opening the files and merging their tensor tables is the caller's job.
//!
//! # Where the facts come from
//!
//! Checked 2026-10-30 against llama.cpp `master`
//! (`src/llama-arch.cpp`, `src/llama.cpp`, `src/llama-model-loader.cpp`,
//! `tools/gguf-split/gguf-split.cpp`):
//!
//! | fact | source |
//! |---|---|
//! | file name is `%s-%05d-of-%05d.gguf`, the number **one-based** | `llama_split_path` |
//! | `split.no` is a `u16`, **zero-based** (the loader requires `0` for the first file) | `gguf-split.cpp`, `llama-model-loader.cpp` |
//! | `split.count` is a `u16`; the loader treats `> 1` as a split model | `llama-model-loader.cpp` |
//! | `split.tensors.count` is an `i32`, the **total across all shards** | `gguf-split.cpp`, `llama-model-loader.cpp` |
//! | `gguf-split` writes `split.count = 0` as a placeholder until it finishes | `gguf-split.cpp` |
//!
//! ⚠️ `split.no` is zero-based and the number in the file name is one-based.
//! A check written as `1 <= no <= count` is wrong for the first file.
//!
//! # What this does not do
//!
//! No key here has a default. A file that does not declare `split.count` is
//! reported as not split, and a split file missing `split.no` is an error,
//! never an assumed `0`.

use mlmf_core::{Declaration, MetadataSource};

/// The key holding a shard's zero-based position.
pub const KEY_NO: &str = "split.no";
/// The key holding the number of shards.
pub const KEY_COUNT: &str = "split.count";
/// The key holding the total tensor count across all shards.
pub const KEY_TENSORS_COUNT: &str = "split.tensors.count";

/// The most shards the file-name convention can express (five digits).
pub const MAX_NAME_COUNT: u32 = 99_999;

const SUFFIX: &str = ".gguf";
/// `-NNNNN-of-MMMMM`
const TAIL_LEN: usize = 15;

/// A parsed `<prefix>-NNNNN-of-MMMMM.gguf` file name.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SplitFileName {
    /// Everything before the numbering: a plain name with no path
    /// separators.
    pub prefix: String,
    /// The shard's number as written in the name, **one-based**.
    pub number: u32,
    /// The shard count as written in the name.
    pub count: u32,
}

/// Why a split fact could not be established.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SplitError {
    /// A prefix is empty or contains a path separator, a drive colon or NUL,
    /// so a file name built from it could point outside the model directory.
    PrefixNotPlain,
    /// A shard count is zero or more than [`MAX_NAME_COUNT`].
    CountOutOfRange {
        /// The count given.
        count: u32,
    },
    /// A `split.*` key is declared with the wrong type or a value outside
    /// the range llama.cpp writes it in.
    WrongType {
        /// The key.
        key: &'static str,
        /// What was expected.
        expected: &'static str,
    },
    /// A `split.*` key is declared but its value could not be decoded.
    Unreadable {
        /// The key.
        key: &'static str,
    },
    /// `split.count` is `0`, the placeholder `gguf-split` writes until it
    /// finishes: a half-written split.
    CountZero,
    /// `split.count` is greater than one but `split.no` is not declared.
    MissingNo,
    /// `split.no` is not below `split.count`.
    NoOutOfRange {
        /// The declared `split.no`.
        no: u16,
        /// The declared `split.count`.
        count: u16,
    },
    /// `split.tensors.count` is negative.
    NegativeTensorCount,
}

impl std::fmt::Display for SplitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrefixNotPlain => write!(f, "split prefix is empty or not a plain file name"),
            Self::CountOutOfRange { count } => {
                write!(f, "split count {count} is outside 1..={MAX_NAME_COUNT}")
            }
            Self::WrongType { key, expected } => {
                write!(
                    f,
                    "{key} is declared with an unusable type, expected {expected}"
                )
            }
            Self::Unreadable { key } => write!(f, "{key} is declared but could not be decoded"),
            Self::CountZero => write!(f, "{KEY_COUNT} is 0, a half-written split"),
            Self::MissingNo => write!(f, "{KEY_NO} is not declared in a split file"),
            Self::NoOutOfRange { no, count } => {
                write!(f, "{KEY_NO} {no} is not below {KEY_COUNT} {count}")
            }
            Self::NegativeTensorCount => write!(f, "{KEY_TENSORS_COUNT} is negative"),
        }
    }
}

impl std::error::Error for SplitError {}

/// Whether `prefix` is usable as the front of a shard file name.
fn plain(prefix: &str) -> bool {
    !prefix.is_empty() && !prefix.contains(['/', '\\', ':', '\0'])
}

/// Five ASCII digits as a number, or `None`.
fn five_digits(s: &[u8]) -> Option<u32> {
    if s.len() == 5 && s.iter().all(u8::is_ascii_digit) {
        std::str::from_utf8(s).ok()?.parse().ok()
    } else {
        None
    }
}

/// Parse `<prefix>-NNNNN-of-MMMMM.gguf`, or `None` if `name` is not one.
///
/// `None` also for a number of `0` or above the count, and for a prefix that
/// is not a plain name: none of those is a valid shard name. The match is
/// exact, as llama.cpp's is: five ASCII digits each, lower-case `.gguf`.
#[must_use]
pub fn parse_split_file_name(name: &str) -> Option<SplitFileName> {
    let stem = name.strip_suffix(SUFFIX)?;
    let cut = stem.len().checked_sub(TAIL_LEN)?;
    if cut == 0 || !stem.is_char_boundary(cut) {
        return None;
    }
    let (prefix, tail) = stem.split_at(cut);
    let t = tail.as_bytes();
    if t[0] != b'-' || &t[6..10] != b"-of-" {
        return None;
    }
    let (number, count) = (five_digits(&t[1..6])?, five_digits(&t[10..15])?);
    if number == 0 || number > count || !plain(prefix) {
        return None;
    }
    Some(SplitFileName {
        prefix: prefix.to_string(),
        number,
        count,
    })
}

/// The file names of every shard of a split model, in order.
///
/// Plain names with no separators, so joining them to a model directory
/// cannot leave it.
///
/// # Errors
///
/// [`SplitError::PrefixNotPlain`] or [`SplitError::CountOutOfRange`].
pub fn shard_file_names(prefix: &str, count: u32) -> Result<Vec<String>, SplitError> {
    if !plain(prefix) {
        return Err(SplitError::PrefixNotPlain);
    }
    if count == 0 || count > MAX_NAME_COUNT {
        return Err(SplitError::CountOutOfRange { count });
    }
    Ok((1..=count)
        .map(|n| format!("{prefix}-{n:05}-of-{count:05}{SUFFIX}"))
        .collect())
}

/// What one shard declares about its place in a split model.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SplitInfo {
    /// `split.no`: this shard's position, **zero-based**.
    pub no: u16,
    /// `split.count`: the number of shards, always `> 1` here.
    pub count: u16,
    /// `split.tensors.count`: the total tensor count across all shards, or
    /// `None` when the file does not declare it.
    pub tensors_total: Option<u32>,
}

const WRONG_TENSORS: SplitError = SplitError::WrongType {
    key: KEY_TENSORS_COUNT,
    expected: "a signed integer that fits i32",
};

fn read_u16<S: MetadataSource + ?Sized>(
    src: &S,
    key: &'static str,
) -> Result<Option<u16>, SplitError> {
    match src.declaration(key) {
        Declaration::Absent => Ok(None),
        Declaration::Declared(v) => v
            .as_u64()
            .and_then(|n| u16::try_from(n).ok())
            .map(Some)
            .ok_or(SplitError::WrongType {
                key,
                expected: "an unsigned integer that fits u16",
            }),
        _ => Err(SplitError::Unreadable { key }),
    }
}

fn read_tensors_total<S: MetadataSource + ?Sized>(src: &S) -> Result<Option<u32>, SplitError> {
    match src.declaration(KEY_TENSORS_COUNT) {
        Declaration::Absent => Ok(None),
        Declaration::Declared(v) => {
            let n = v.as_i64().ok_or(WRONG_TENSORS)?;
            let n = i32::try_from(n).map_err(|_| WRONG_TENSORS)?;
            u32::try_from(n)
                .map(Some)
                .map_err(|_| SplitError::NegativeTensorCount)
        }
        _ => Err(SplitError::Unreadable {
            key: KEY_TENSORS_COUNT,
        }),
    }
}

impl SplitInfo {
    /// Read the `split.*` keys from `src`.
    ///
    /// `Ok(None)` when the file is not a multi-file split: `split.count` is
    /// not declared, or is `1`. llama.cpp's own test is `split.count > 1`.
    ///
    /// # Errors
    ///
    /// [`SplitError::CountZero`], [`SplitError::MissingNo`],
    /// [`SplitError::NoOutOfRange`], [`SplitError::NegativeTensorCount`],
    /// and [`SplitError::WrongType`] / [`SplitError::Unreadable`] for a key
    /// that is declared but not usable.
    pub fn from_source<S: MetadataSource + ?Sized>(src: &S) -> Result<Option<Self>, SplitError> {
        let Some(count) = read_u16(src, KEY_COUNT)? else {
            return Ok(None);
        };
        match count {
            0 => return Err(SplitError::CountZero),
            1 => return Ok(None),
            _ => {}
        }
        let no = read_u16(src, KEY_NO)?.ok_or(SplitError::MissingNo)?;
        if no >= count {
            return Err(SplitError::NoOutOfRange { no, count });
        }
        Ok(Some(Self {
            no,
            count,
            tensors_total: read_tensors_total(src)?,
        }))
    }

    /// Whether `name` agrees with what this shard declares: the one-based
    /// number is `no + 1` and the counts match.
    #[must_use]
    pub fn matches_file_name(&self, name: &SplitFileName) -> bool {
        name.number == u32::from(self.no) + 1 && name.count == u32::from(self.count)
    }
}

#[cfg(test)]
mod tests;
