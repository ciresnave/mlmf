// SPDX-License-Identifier: MIT OR Apache-2.0
//! `parse_pth` against a real PyTorch checkpoint, checked against two
//! independent readers.
//!
//! The file is `hf-internal-testing/tiny-random-gpt2`'s `pytorch_model.bin`
//! (HF revision `71034c5d8bde858ff824298bdedc65515b97d2b9`, 3,561,811 bytes,
//! sha256 `4fab47c129967e0db58e8faf8494e4bd04f2ea79bbe287ac2f90c4183c0194be`;
//! the repository states no licence, so it is **not vendored**). The test
//! reads it from `MLMF_PTH_FILE` and prints `SKIPPED` when that is unset or
//! the file is absent. CI has no such file, so there it asserts nothing; a
//! skip is not a pass. `MLMF_CORPUS_REQUIRED=1` turns a skip red.
//!
//! `corpus-pth.tsv` (152 tensors) was produced 2026-10-25 by a Python reader
//! using `zipfile` and a restricted `pickle.Unpickler`, and **agrees
//! row for row** (name, shape, strides, offset, zip path, storage size) with
//! the output of Fuel's `read_pth_tensor_info` on the same file, measured
//! separately by the Fuel lane. The last column is the byte sum of each
//! storage as `zipfile` extracts it, which checks that `storage_ranges` land
//! on the right bytes and not merely the right length.

use mlmf_pickle::parse_pth;

use mlmf_test_support as armed;

#[test]
fn the_fixture_has_the_152_rows_it_was_measured_on() {
    let rows: Vec<&str> = include_str!("corpus-pth.tsv").lines().skip(1).collect();
    assert_eq!(rows.len(), 152);
    assert!(rows.iter().all(|r| r.split('\t').count() == 8));
}

fn join(v: &[usize]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

#[test]
fn the_real_checkpoint_reads_to_the_independently_measured_rows() {
    let Ok(path) = std::env::var("MLMF_PTH_FILE") else {
        assert!(
            !armed::armed("MLMF_CORPUS_REQUIRED"),
            "MLMF_CORPUS_REQUIRED is set but MLMF_PTH_FILE is not"
        );
        eprintln!("SKIPPED: MLMF_PTH_FILE is not set");
        return;
    };
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(bytes.len(), 3_561_811, "not the file these rows describe");
    let idx = parse_pth(&bytes, None).expect("a real torch.save checkpoint parses");
    let rows: Vec<Vec<&str>> = include_str!("corpus-pth.tsv")
        .lines()
        .skip(1)
        .map(|r| r.split('\t').collect())
        .collect();
    assert_eq!(idx.tensors.len(), rows.len());
    assert!(idx.skipped.is_empty());

    let mut by_name: Vec<usize> = (0..idx.tensors.len()).collect();
    by_name.sort_by(|&a, &b| idx.tensors[a].name.cmp(&idx.tensors[b].name));
    for (&i, row) in by_name.iter().zip(&rows) {
        let t = &idx.tensors[i];
        let got = [
            t.name.clone(),
            t.storage_class.clone(),
            join(&t.shape),
            join(&t.strides),
            t.storage_offset.to_string(),
            t.storage_path.clone(),
            t.storage_numel.to_string(),
        ];
        assert_eq!(got.as_slice(), &row[..7], "{}", t.name);
        let r = idx.storage_ranges[i].clone().expect("stored");
        let sum: u64 = bytes[r.start as usize..r.end as usize]
            .iter()
            .map(|&b| u64::from(b))
            .sum();
        assert_eq!(sum.to_string(), row[7], "{}: storage bytes", t.name);
        assert_eq!(
            r.end - r.start,
            t.storage_numel * t.dtype.size() as u64,
            "{}: storage length",
            t.name
        );
    }
    eprintln!("RAN: {} tensors from {path}", idx.tensors.len());
}
