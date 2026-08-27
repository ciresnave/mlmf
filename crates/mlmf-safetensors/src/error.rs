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
/// [`Stage::TensorDirectory`] landed with the tensor directory itself and
/// not before, which is the discipline `mlmf-gguf`'s `error.rs` records the
/// cost of skipping: a variant added ahead of its use is one no review has
/// reason to look at, and the doc beside it goes stale in the window where
/// nothing calls it. This doc used to say "one variant today"; that
/// sentence and the variant it described are the same commit's work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Stage {
    /// The 8-byte little-endian length prefix and the JSON header it sizes.
    Header,
    /// The tensor directory: the header object's entries read as tensor
    /// records, and their `data_offsets` rebased onto the slice.
    ///
    /// A **separate stage from the header even though it reads the same
    /// bytes.** In `mlmf-gguf` the two stages are two byte regions, so the
    /// split is physical; here one JSON object carries both the metadata
    /// and the tensor records (D2), so the split is between *parsing the
    /// header* and *deriving the tensors from it*. That is why a consumer
    /// who only wants `__metadata__` can get it from a file whose tensor
    /// records this build cannot read.
    TensorDirectory,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Stage::Header => "header",
            Stage::TensorDirectory => "tensor directory",
        };
        f.write_str(s)
    }
}

/// A safetensors parse failure.
///
/// Every variant names the [`Stage`] that produced it. Every variant that
/// **can** name a byte offset does, into the slice as given rather than into
/// some sub-region.
///
/// [`Self::MalformedEntry`] is the one that cannot, and it says so by
/// carrying a name instead: once the header has been handed to a JSON
/// parser there is no byte offset left to report, and inventing one would be
/// worse than admitting it. See that variant.
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

    /// An entry in the header object is not shaped like a tensor record.
    ///
    /// A tensor record is `{ "dtype": str, "shape": [int], "data_offsets":
    /// [int, int] }`. This is the entry that is missing one of those, or
    /// declares one with the wrong JSON kind, or declares `data_offsets`
    /// with a count other than two. It is **not** a record that parses and
    /// then makes a false claim — an inverted range, a width that disagrees
    /// with the shape, a dtype this build does not know — because those
    /// cost exactly one tensor and are reported through
    /// [`mlmf_core::Report`] with the tensor omitted, the way `mlmf-gguf`
    /// reports a type code it cannot resolve. Refusing the whole file for
    /// one bad tensor would make every other tensor and all of
    /// `__metadata__` unreachable.
    ///
    /// # Why this variant exists rather than [`Self::Malformed`]
    ///
    /// **It is located by name, because there is no offset to report.**
    /// Every other variant here points at a byte in the slice. By the time
    /// an entry is being read, the header has been through `serde_json`,
    /// which hands back a map and keeps no byte positions — so a
    /// `Malformed { offset }` here could only carry the offset of the
    /// header itself, which is the same number for every entry in the file
    /// and points at a `{` that is not the problem. A field documented as
    /// "where the problem is" holding a number that is not where the
    /// problem is would be the same defect as
    /// [`mlmf_core::UnrecognizedKind::MetadataKey`]'s repaired `value`
    /// field: an explanation living in a slot documented as data.
    ///
    /// A tensor name is a lookup key and is what the operator can search
    /// the header for, so it is the honest locator.
    MalformedEntry {
        /// Stage that was reading.
        stage: Stage,
        /// The header key this entry was declared under, exactly as
        /// declared.
        name: String,
        /// What was wrong, naming the JSON kind or value seen.
        detail: String,
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
            SafetensorsError::MalformedEntry {
                stage,
                name,
                detail,
            } => write!(
                f,
                // `{name:?}` rather than `{name}`: a tensor name is an
                // opaque key that this project has already ruled must be
                // byte-exact, and real headers carry names full of dots.
                // Quoting is what keeps `entry ` (a name that is one
                // space) from rendering as an empty gap the reader cannot
                // see, and what makes a trailing space visible at all.
                "malformed {stage} entry {name:?}: {detail}"
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
    fn a_malformed_entry_is_located_by_name_because_there_is_no_offset() {
        // The whole reason this variant is not `Malformed`: after
        // `serde_json` has parsed the header there is no byte position
        // left, so the locator is the key. Whole-string comparison, and
        // the name is quoted — see the `Display` impl for why.
        assert_eq!(
            SafetensorsError::MalformedEntry {
                stage: Stage::TensorDirectory,
                name: "model.embed_tokens.weight".to_string(),
                detail: "data_offsets has 3 elements, not 2".to_string(),
            }
            .to_string(),
            "malformed tensor directory entry \"model.embed_tokens.weight\": \
             data_offsets has 3 elements, not 2"
        );
    }

    #[test]
    fn a_name_whose_whitespace_would_be_invisible_is_quoted() {
        // A name that is one space renders as a gap the reader cannot see
        // without the quotes, and safetensors puts no constraint on a
        // tensor name beyond it being a JSON string. This is the assertion
        // that fails if `{name:?}` is relaxed to `{name}`.
        assert_eq!(
            SafetensorsError::MalformedEntry {
                stage: Stage::TensorDirectory,
                name: " ".to_string(),
                detail: "not an object".to_string(),
            }
            .to_string(),
            "malformed tensor directory entry \" \": not an object"
        );
    }

    #[test]
    fn every_stage_has_a_name() {
        // Pinned arm by arm, by identity. The header case is exercised by
        // every other test in this module and the tensor-directory case by
        // exactly one, so a `Display` that returned "header" for both would
        // otherwise be caught by a single assertion in a single test.
        assert_eq!(
            [
                Stage::Header.to_string(),
                Stage::TensorDirectory.to_string()
            ],
            ["header".to_string(), "tensor directory".to_string()]
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
