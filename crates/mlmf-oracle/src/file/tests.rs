// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;

const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn oracle_entry(source: &str, target: &str) -> Entry {
    Entry {
        source: source.into(),
        target: target.into(),
        evidence: Evidence::oracle("some-model", HASH, "2026-10-26"),
    }
}

fn plain(source: &str, target: &str, evidence: Evidence) -> Entry {
    Entry {
        source: source.into(),
        target: target.into(),
        evidence,
    }
}

#[test]
fn a_file_round_trips_through_json() {
    let f = MappingFile::new(
        Direction::HfToGguf,
        vec![
            oracle_entry("a.weight", "x.weight"),
            plain("b.weight", "y.weight", Evidence::human()),
            plain("c.weight", "z.weight", Evidence::table()),
        ],
    )
    .unwrap();
    let back = MappingFile::parse(f.to_json().as_bytes()).unwrap();
    assert_eq!(back, f);
    assert_eq!(back.target_of("b.weight"), Some("y.weight"));
    assert_eq!(back.target_of("missing"), None);
}

#[test]
fn names_are_compared_byte_exactly() {
    let f = MappingFile::new(Direction::GgufToHf, vec![oracle_entry("Blk.0.Attn_Q", "q")]).unwrap();
    assert_eq!(f.target_of("blk.0.attn_q"), None);
    assert_eq!(f.target_of("Blk.0.Attn_Q"), Some("q"));
    assert_eq!(f.target_of("Blk.0.Attn_Q "), None);
}

#[test]
fn a_duplicate_source_or_target_is_refused() {
    let dup_source = vec![oracle_entry("a", "x"), oracle_entry("a", "y")];
    let dup_target = vec![oracle_entry("a", "x"), oracle_entry("b", "x")];
    assert!(MappingFile::new(Direction::HfToGguf, dup_source).is_err());
    assert!(MappingFile::new(Direction::HfToGguf, dup_target).is_err());
}

#[test]
fn evidence_must_fit_its_kind() {
    let incomplete = Evidence {
        model: None,
        ..Evidence::oracle("m", HASH, "2026-10-26")
    };
    let human_with_model = Evidence {
        model: Some("m".into()),
        ..Evidence::human()
    };
    let bad_hash = Evidence::oracle("m", "ABC", "2026-10-26");
    let upper_hash = Evidence::oracle("m", &HASH.to_uppercase(), "2026-10-26");
    let bad_date = Evidence::oracle("m", HASH, "26-10-2026");
    let empty_model = Evidence::oracle("", HASH, "2026-10-26");
    for (what, ev) in [
        ("missing model", incomplete),
        ("human with model", human_with_model),
        ("short hash", bad_hash),
        ("uppercase hash", upper_hash),
        ("bad date", bad_date),
        ("empty model", empty_model),
    ] {
        let r = MappingFile::new(Direction::HfToGguf, vec![plain("a", "b", ev)]);
        assert!(r.is_err(), "{what} was accepted");
    }
}

#[test]
fn an_empty_name_is_refused() {
    assert!(MappingFile::new(Direction::HfToGguf, vec![oracle_entry("", "x")]).is_err());
    assert!(MappingFile::new(Direction::HfToGguf, vec![oracle_entry("a", "")]).is_err());
}

fn doc(format: &str, version: &str, direction: &str, entries: &str) -> String {
    format!(
        r#"{{"format":"{format}","version":{version},"direction":"{direction}","entries":{entries}}}"#
    )
}

#[test]
fn an_unknown_format_version_or_direction_is_a_read_error() {
    let ok = doc("mlmf-name-mapping", "1", "hf-to-gguf", "[]");
    assert!(MappingFile::parse(ok.as_bytes()).is_ok());
    for bad in [
        doc("other", "1", "hf-to-gguf", "[]"),
        doc("mlmf-name-mapping", "2", "hf-to-gguf", "[]"),
        doc("mlmf-name-mapping", "\"1\"", "hf-to-gguf", "[]"),
        doc("mlmf-name-mapping", "1", "sideways", "[]"),
    ] {
        assert!(MappingFile::parse(bad.as_bytes()).is_err(), "{bad}");
    }
}

#[test]
fn malformed_input_is_a_read_error_not_a_panic() {
    for bad in ["", "[]", "{", r#"{"format":"mlmf-name-mapping"}"#] {
        assert!(MappingFile::parse(bad.as_bytes()).is_err(), "{bad:?}");
    }
    let entry_not_object = doc("mlmf-name-mapping", "1", "hf-to-gguf", "[1]");
    let no_evidence = doc(
        "mlmf-name-mapping",
        "1",
        "hf-to-gguf",
        r#"[{"source":"a","target":"b"}]"#,
    );
    let wrong_type = doc(
        "mlmf-name-mapping",
        "1",
        "hf-to-gguf",
        r#"[{"source":"a","target":"b","evidence":{"kind":"human","model":3}}]"#,
    );
    let unknown_kind = doc(
        "mlmf-name-mapping",
        "1",
        "hf-to-gguf",
        r#"[{"source":"a","target":"b","evidence":{"kind":"guess"}}]"#,
    );
    for bad in [entry_not_object, no_evidence, wrong_type, unknown_kind] {
        assert!(MappingFile::parse(bad.as_bytes()).is_err(), "{bad}");
    }
}

#[test]
fn a_duplicate_source_in_a_file_is_a_read_error() {
    let two = r#"[{"source":"a","target":"x","evidence":{"kind":"human"}},
                  {"source":"a","target":"y","evidence":{"kind":"human"}}]"#;
    let d = doc("mlmf-name-mapping", "1", "hf-to-gguf", two);
    let e = MappingFile::parse(d.as_bytes()).unwrap_err().to_string();
    assert!(e.contains("appears twice"), "{e}");
}
