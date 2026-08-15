//! The table agrees with the committed ground truth, in both directions.

use mlmf_ggml::GgmlType;

/// One fixture row. Field order matches the TSV header.
#[derive(Debug, PartialEq)]
struct Row {
    code: u32,
    name: String,
    kind: String,
    elems: u64,
    bytes: u64,
    align: usize,
}

fn fixture() -> Vec<Row> {
    let raw = include_str!("ggml-types.tsv");
    raw.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("code\t") && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 6, "malformed fixture row: {l:?}");
            Row {
                code: f[0].parse().unwrap(),
                name: f[1].to_string(),
                kind: f[2].to_string(),
                elems: f[3].parse().unwrap(),
                bytes: f[4].parse().unwrap(),
                align: f[5].parse().unwrap(),
            }
        })
        .collect()
}

#[test]
fn the_fixture_itself_is_intact() {
    // If a merge mangles the TSV into three rows, every other test in this
    // file passes vacuously. Assert the fixture's own shape first.
    let rows = fixture();
    assert_eq!(rows.len(), 35, "expected 35 ground-truth rows");
    assert_eq!(rows.iter().filter(|r| r.kind == "dense").count(), 8);
    assert_eq!(rows.iter().filter(|r| r.kind == "blocked").count(), 27);
}

#[test]
fn every_ground_truth_row_is_in_the_table() {
    for row in fixture() {
        let t = GgmlType::from_code(row.code)
            .unwrap_or_else(|| panic!("code {} ({}) missing from table", row.code, row.name));
        assert_eq!(t.name(), row.name, "code {}", row.code);
        assert_eq!(t.elements_per_block(), row.elems, "{}", row.name);
        assert_eq!(t.bytes_per_block(), row.bytes, "{}", row.name);
        assert_eq!(t.alignment(), row.align, "{}", row.name);
        assert_eq!(t.is_quantized(), row.kind == "blocked", "{}", row.name);
    }
}

#[test]
fn the_table_declares_nothing_the_ground_truth_does_not() {
    // The other direction. Without this, an invented 36th type — a
    // plausible-looking row someone adds from memory — passes every check.
    let known: Vec<u32> = fixture().iter().map(|r| r.code).collect();
    for t in GgmlType::ALL {
        assert!(
            known.contains(&t.code()),
            "table declares {} (code {}) which is not in the ground truth",
            t.name(),
            t.code()
        );
    }
}
