//! Template extraction: the GGUF multi-template indirection, §9 1.1's
//! blank-is-undeclared rule, and the loss lists §5 rule 3 requires.

use mlmf_core::{MetaValue, MetadataSource};
use mlmf_meta::template::TemplateSet;
use mlmf_meta::vocab::Format;

/// A source whose contents the test states outright. Keys are the GGUF
/// spellings because that is what a GGUF-format source would carry.
struct Fake(Vec<(String, MetaValue)>);

impl MetadataSource for Fake {
    fn get(&self, key: &str) -> Option<&MetaValue> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn keys(&self) -> Vec<&str> {
        self.0.iter().map(|(k, _)| k.as_str()).collect()
    }
    fn index_complete(&self) -> bool {
        true
    }
}

fn s(v: &str) -> MetaValue {
    MetaValue::String(v.to_string())
}

#[test]
fn a_file_with_no_template_yields_an_empty_set_not_a_default() {
    let set = TemplateSet::extract(&Fake(vec![]), Format::Gguf);
    assert!(set.entries.is_empty());
    assert_eq!(
        set.default_body(),
        None,
        "absent means None, never a default"
    );
    assert!(set.blank.is_empty() && set.not_a_string.is_empty());
}

#[test]
fn the_lone_default_template_is_found_and_is_unnamed() {
    let f = Fake(vec![("tokenizer.chat_template".into(), s("{{ body }}"))]);
    let set = TemplateSet::extract(&f, Format::Gguf);
    assert_eq!(set.entries.len(), 1);
    assert_eq!(set.entries[0].name, None);
    assert_eq!(set.default_body(), Some("{{ body }}"));
}

#[test]
fn a_whitespace_only_template_is_undeclared_and_the_loss_is_named() {
    // Spec §9 clause 1.1: "A blank chat_template is UNDECLARED, not
    // declared-empty. Check trim().is_empty(), not is_empty()." A blank
    // template parses fine and renders an empty prompt -- the model gets a
    // request to continue nothing. Shipped once and caught in review at
    // Lightbulb.
    //
    // NOT OBSERVED in the corpus: 0 of the 16 STRING-VALUED chat_template
    // keys are whitespace-only as of 2026-09-03. The clause is normative
    // regardless, and nothing downstream would catch a regression.
    for blank in ["", " ", "\n", "  \t\n "] {
        let f = Fake(vec![("tokenizer.chat_template".into(), s(blank))]);
        let set = TemplateSet::extract(&f, Format::Gguf);
        assert!(
            set.entries.is_empty(),
            "{blank:?} must be UNDECLARED, not declared-empty"
        );
        assert_eq!(set.default_body(), None);
        // Section 5 rule 3: the loss is marked per key, not left silent.
        assert_eq!(set.blank, vec!["tokenizer.chat_template".to_string()]);
    }
}

#[test]
fn all_three_command_r_templates_are_returned_not_just_the_default() {
    // Shape taken from llamacpp-vocab/ggml-vocab-command-r.gguf, measured
    // 2026-09-03. The names array holds NAMES and excludes the default, so
    // a reader that follows only `tokenizer.chat_template` returns 1 of 3
    // and reports no loss. That is the failure this test exists to catch.
    let f = Fake(vec![
        ("tokenizer.chat_template".into(), s("DEFAULT")),
        ("tokenizer.chat_template.rag".into(), s("RAG")),
        ("tokenizer.chat_template.tool_use".into(), s("TOOLS")),
        (
            "tokenizer.chat_templates".into(),
            MetaValue::Array(vec![s("tool_use"), s("rag")]),
        ),
    ]);
    let set = TemplateSet::extract(&f, Format::Gguf);

    assert_eq!(set.entries.len(), 3, "got {:?}", set.entries);
    assert_eq!(set.default_body(), Some("DEFAULT"));

    let mut named: Vec<_> = set
        .entries
        .iter()
        .filter_map(|e| e.name.as_deref().map(|n| (n, e.body.as_str())))
        .collect();
    named.sort_unstable();
    assert_eq!(named, vec![("rag", "RAG"), ("tool_use", "TOOLS")]);

    assert!(set.named_but_absent.is_empty());
    assert!(set.present_but_unnamed.is_empty());
}

#[test]
fn a_name_with_no_body_is_reported_rather_than_dropped() {
    let f = Fake(vec![
        ("tokenizer.chat_template".into(), s("DEFAULT")),
        (
            "tokenizer.chat_templates".into(),
            MetaValue::Array(vec![s("ghost")]),
        ),
    ]);
    let set = TemplateSet::extract(&f, Format::Gguf);
    assert_eq!(set.entries.len(), 1);
    assert_eq!(
        set.named_but_absent,
        vec!["tokenizer.chat_template.ghost".to_string()],
        "the full key, not the bare name -- all four loss lists agree"
    );
}

#[test]
fn a_body_with_no_name_is_returned_and_flagged() {
    // NOT OBSERVED in the corpus as of 2026-09-03 -- every named body in
    // ggml-vocab-command-r.gguf is listed. This asserts the HANDLING, and
    // says so, rather than implying the shape exists in the wild.
    let f = Fake(vec![("tokenizer.chat_template.orphan".into(), s("ORPHAN"))]);
    let set = TemplateSet::extract(&f, Format::Gguf);
    assert_eq!(set.entries.len(), 1, "the body is preserved, not dropped");
    assert_eq!(
        set.present_but_unnamed,
        vec!["tokenizer.chat_template.orphan".to_string()]
    );
}

#[test]
fn a_non_string_template_value_is_not_coerced_and_the_loss_is_named() {
    let f = Fake(vec![("tokenizer.chat_template".into(), MetaValue::U32(7))]);
    let set = TemplateSet::extract(&f, Format::Gguf);
    assert!(
        set.entries.is_empty(),
        "a format that did not declare a string did not declare a template"
    );
    // Without this, a U32-valued key and an absent key are byte-identical
    // in the result -- section 5 rule 1's invisible loss.
    assert_eq!(
        set.not_a_string,
        vec!["tokenizer.chat_template".to_string()]
    );
}

#[test]
fn named_entries_come_back_sorted_whatever_order_the_source_offers() {
    // `MetadataSource::keys()` is documented "in unspecified order". This
    // Fake hands the SAME three templates back in two different orders; the
    // extraction must be identical both times, because a consumer diffing
    // two extractions -- a cross-reader disagreement harness, say -- would
    // otherwise read the source's iteration order as a disagreement about
    // the files.
    //
    // Without the sort in `extract`, this test fails on the second order
    // while the first still passes, which is why both are here.
    let bodies = [("zeta", "Z"), ("alpha", "A"), ("mid", "M")];

    let forward: Vec<(String, MetaValue)> = bodies
        .iter()
        .map(|(n, b)| (format!("tokenizer.chat_template.{n}"), s(b)))
        .collect();
    let mut reversed = forward.clone();
    reversed.reverse();

    let a = TemplateSet::extract(&Fake(forward), Format::Gguf);
    let b = TemplateSet::extract(&Fake(reversed), Format::Gguf);

    assert_eq!(a, b, "extraction must not depend on key iteration order");
    assert_eq!(
        a.entries
            .iter()
            .map(|e| e.name.as_deref().unwrap_or("<default>"))
            .collect::<Vec<_>>(),
        vec!["alpha", "mid", "zeta"],
        "named entries are sorted by name"
    );
}
