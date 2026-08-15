//! Computed sizes match sizes measured from real model files.

use mlmf_ggml::GgmlType;

#[test]
fn computed_sizes_match_real_files() {
    let raw = include_str!("corpus-sizes.tsv");
    let mut checked = 0;
    for line in raw.lines() {
        if line.starts_with('#') || line.starts_with("code\t") || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 6, "malformed row: {line:?}");
        let code: u32 = f[0].parse().unwrap();
        let name = f[1];
        let ne: Vec<u64> = f[2].split(',').map(|d| d.parse().unwrap()).collect();
        let expected: u64 = f[3].parse().unwrap();

        let t = GgmlType::from_code(code).unwrap_or_else(|| panic!("code {code} missing"));
        assert_eq!(t.name(), name);
        let got = t.nbytes(&ne, f[5]).unwrap_or_else(|e| panic!("{name} {ne:?}: {e}"));
        assert_eq!(
            got, expected,
            "{name} {ne:?} from {} tensor {}",
            f[4], f[5]
        );
        checked += 1;
    }
    assert_eq!(checked, 20, "fixture lost rows");
}

#[test]
fn the_two_types_with_identical_geometry_are_both_present_and_agree() {
    // Q4_0 (code 2) and IQ4_NL (code 20) appear in the fixture at the same
    // shapes and the same byte counts — measured from different files. That
    // agreement is the evidence that identical geometry is a real property
    // of these formats and not a transcription error in one of the rows.
    let shape = [576u64, 192];
    assert_eq!(
        GgmlType::Q4_0.nbytes(&shape, "t").unwrap(),
        GgmlType::IQ4_NL.nbytes(&shape, "t").unwrap()
    );
}
