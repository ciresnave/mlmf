//! What went wrong, and at which stage.
//!
//! The stage split exists for the same reason it does in `mlmf-gguf`: a
//! consumer's remedies differ (R7). Naming the stage is also what makes the
//! staged parse checkable from the outside — a header error and a tensor
//! error are different facts about a file, and collapsing them forces a
//! warning to hedge across both.
//!
//! Safetensors has no magic number, so there is no `NotSafetensors`
//! counterpart to `mlmf-gguf`'s `NotGguf`. The first eight bytes of any
//! file are a valid `u64`; what refuses a file that is not a safetensors
//! file is the bound check on that number, and it reports
//! [`SafetensorsError::Truncated`] — "the header this file declares does
//! not fit in this file" — which is the true statement available.

use std::fmt;

/// Which stage of the staged parse produced an error.
///
/// One variant today. The tensor directory is a second stage and gets its
/// own variant in the commit that reaches it, not before — a variant added
/// ahead of its use is one no review has reason to look at. `mlmf-gguf`'s
/// `Stage::TensorDirectory` carries a comment recording exactly that trap
/// springing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stage {
    /// The 8-byte little-endian length prefix and the JSON header it sizes.
    Header,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Stage::Header => "header",
        };
        f.write_str(s)
    }
}

/// A safetensors parse failure.
///
/// Every variant names the [`Stage`] that produced it and the offset — into
/// the slice as given, not into some sub-region — where the problem is.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SafetensorsError {
    /// The file ends before a declared structure does.
    ///
    /// This is the variant an 8-byte attacker-controlled length prefix
    /// lands in. `needed` is the number the FILE declared, reported
    /// verbatim so an operator sees the absurd value rather than a
    /// saturated one.
    Truncated {
        /// Stage that was reading.
        stage: Stage,
        /// Offset the read started at.
        offset: u64,
        /// Bytes the declared structure needs.
        needed: u64,
        /// Bytes that remain from `offset`.
        available: u64,
    },

    /// The bytes are present and do not make sense.
    Malformed {
        /// Stage that was reading.
        stage: Stage,
        /// Offset the problem was found at.
        offset: u64,
        /// What was wrong, naming the value seen.
        detail: String,
    },

    /// The header bytes are not valid UTF-8.
    ///
    /// Kept distinct from [`Self::Malformed`] because the remedies differ:
    /// a header that is not UTF-8 was written by something that did not
    /// emit JSON at all, or by something that emitted it in another
    /// encoding, and neither is "this JSON has a syntax error".
    NotUtf8 {
        /// Stage that was reading.
        stage: Stage,
        /// Offset of the first byte that is not valid UTF-8.
        offset: u64,
    },

    /// The header is valid JSON whose top level is not an object.
    ///
    /// Distinct from [`Self::Malformed`] for the same reason `mlmf-gguf`
    /// keeps `NotGguf` distinct: a well-formed JSON array is not a
    /// corrupt safetensors header, it is a different kind of file, and
    /// sending an operator to look for corruption wastes their time.
    NotAnObject {
        /// Stage that was reading.
        stage: Stage,
        /// Offset the JSON header starts at.
        offset: u64,
        /// The JSON kind actually found: `array`, `string`, `number`,
        /// `boolean` or `null`.
        found: &'static str,
    },
}

impl fmt::Display for SafetensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafetensorsError::Truncated {
                stage,
                offset,
                needed,
                available,
            } => write!(
                f,
                "truncated in the {stage} at offset {offset}: \
                 needed {needed} bytes, {available} remain"
            ),
            SafetensorsError::Malformed {
                stage,
                offset,
                detail,
            } => write!(f, "malformed {stage} at offset {offset}: {detail}"),
            SafetensorsError::NotUtf8 { stage, offset } => write!(
                f,
                "the {stage} is not valid UTF-8: first invalid byte at offset {offset}"
            ),
            SafetensorsError::NotAnObject {
                stage,
                offset,
                found,
            } => write!(
                f,
                "the {stage} at offset {offset} is a JSON {found}, not an object"
            ),
        }
    }
}

impl std::error::Error for SafetensorsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_truncation_names_the_stage_the_offset_and_both_byte_counts() {
        // Whole-string comparison. A chain of `contains` assertions passes
        // on a message that has transposed `needed` and `available` --
        // adjacent fields, a plausible one-line slip -- because both
        // numbers are still present somewhere in the string.
        assert_eq!(
            SafetensorsError::Truncated {
                stage: Stage::Header,
                offset: 8,
                needed: u64::MAX,
                available: 2,
            }
            .to_string(),
            "truncated in the header at offset 8: needed 18446744073709551615 bytes, 2 remain"
        );
    }

    #[test]
    fn a_non_object_top_level_says_what_it_found_instead() {
        assert_eq!(
            SafetensorsError::NotAnObject {
                stage: Stage::Header,
                offset: 8,
                found: "array",
            }
            .to_string(),
            "the header at offset 8 is a JSON array, not an object"
        );
    }

    #[test]
    fn a_non_utf8_header_points_at_the_offending_byte() {
        assert_eq!(
            SafetensorsError::NotUtf8 {
                stage: Stage::Header,
                offset: 12,
            }
            .to_string(),
            "the header is not valid UTF-8: first invalid byte at offset 12"
        );
    }

    #[test]
    fn a_malformed_header_carries_its_detail() {
        assert_eq!(
            SafetensorsError::Malformed {
                stage: Stage::Header,
                offset: 0,
                detail: "declared header length is zero".to_string(),
            }
            .to_string(),
            "malformed header at offset 0: declared header length is zero"
        );
    }
}
