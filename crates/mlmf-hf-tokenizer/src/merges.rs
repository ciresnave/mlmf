//! `model.merges` reading, split out of `lib.rs` so each shape's guard
//! is its own function (Codacy complexity finding, mirroring mlmf#112's
//! fix for mlmf-awq/mlmf-gptq). No logic or message text changed from
//! the single-file version -- see the crate's module doc for the two
//! real shapes this validates against.

use crate::MergeRule;

/// `model.merges`: absent is an empty list; present-but-wrong-shaped is
/// `malformed`; each element must be one of this key's two real shapes
/// (see [`merge_rule_from_entry`]).
pub(crate) fn parse_merges(
    value: Option<&serde_json::Value>,
    malformed: &mut Vec<String>,
) -> Vec<MergeRule> {
    let Some(value) = value else {
        return Vec::new();
    };
    let Some(array) = value.as_array() else {
        malformed.push("merges".to_string());
        return Vec::new();
    };
    let mut out = Vec::with_capacity(array.len());
    for (i, entry) in array.iter().enumerate() {
        match merge_rule_from_entry(entry) {
            Some(rule) => out.push(rule),
            None => malformed.push(format!("merges[{i}]")),
        }
    }
    out
}

/// One `model.merges` entry, in either of its two verified real shapes
/// (see the module doc): a `"left right"` string, or a two-element
/// array. Each shape's own validation is its own function so this is a
/// straight-line dispatch, not a nest of conditions.
fn merge_rule_from_entry(entry: &serde_json::Value) -> Option<MergeRule> {
    if let Some(s) = entry.as_str() {
        return merge_rule_from_string(s);
    }
    if let Some(array) = entry.as_array() {
        return merge_rule_from_array(array);
    }
    None
}

/// The `"left right"` string shape: EXACTLY one space, two non-empty
/// pieces. `None` for no space, more than one space, or an empty piece.
///
/// A string like `"a b c"` could be read as `("a", "b c")` or `("a b",
/// "c")`; the file declares neither reading, and picking one would be
/// exactly the §6-fence violation (`CLAUDE.md` §1) a final review
/// (mlmf#116) found in this function's first draft, which split on the
/// first space unconditionally.
fn merge_rule_from_string(s: &str) -> Option<MergeRule> {
    let mut parts = s.split(' ');
    let left = parts.next()?;
    let right = parts.next()?;
    if parts.next().is_some() || left.is_empty() || right.is_empty() {
        return None;
    }
    Some(MergeRule {
        left: left.to_string(),
        right: right.to_string(),
    })
}

/// The two-element-array shape (`Qwen/Qwen3-0.6B`'s real
/// `tokenizer.json`, mlmf#116): exactly two elements, both non-empty
/// strings.
fn merge_rule_from_array(array: &[serde_json::Value]) -> Option<MergeRule> {
    let [left, right] = array else {
        return None;
    };
    let (Some(left), Some(right)) = (left.as_str(), right.as_str()) else {
        return None;
    };
    if left.is_empty() || right.is_empty() {
        return None;
    }
    Some(MergeRule {
        left: left.to_string(),
        right: right.to_string(),
    })
}
