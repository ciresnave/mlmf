// SPDX-License-Identifier: MIT OR Apache-2.0
//! `parse_raw_imatrix` against a real `llama-imatrix` output.
//!
//! The file is `bartowski/Llama-3.2-1B-Instruct-GGUF`'s
//! `Llama-3.2-1B-Instruct.imatrix` (HF commit
//! `067b946cf014b7c697f3654f621d577a3e3afd1c`, 1,314,426 bytes). It is NOT
//! vendored: its licence was not established, so the test reads it from the
//! path in `MLMF_IMATRIX_RAW_FILE` and announces `SKIPPED` when that is
//! unset or the file is absent. CI has no such file, so there this test
//! reports a skip and asserts nothing; a skip is not a pass.
//!
//! The expected numbers come from an independent reader (a ten-line Python
//! `struct` parse, run 2026-10-25 against that file), not from this crate:
//! 112 entries, 327,680 values, first entry `blk.15.attn_k.weight` with
//! `ncall` 125, `nval` 2048 and first value 13.612717628479004, trailer
//! `last_call` 125 and a dataset path.

use mlmf_gguf::parse_raw_imatrix;

#[test]
fn the_real_llama_3_2_1b_imatrix_parses_to_the_independently_measured_facts() {
    let Ok(path) = std::env::var("MLMF_IMATRIX_RAW_FILE") else {
        eprintln!("SKIPPED: MLMF_IMATRIX_RAW_FILE is not set");
        return;
    };
    let Ok(bytes) = std::fs::read(&path) else {
        eprintln!("SKIPPED: {path} is not readable");
        return;
    };
    assert_eq!(bytes.len(), 1_314_426, "not the file these facts describe");

    let m = parse_raw_imatrix(&bytes).expect("a real llama-imatrix file parses");
    assert_eq!(m.entries.len(), 112);
    assert_eq!(
        m.entries.iter().map(|e| e.values.len()).sum::<usize>(),
        327_680
    );
    let first = &m.entries[0];
    assert_eq!(first.name, "blk.15.attn_k.weight");
    assert_eq!((first.ncall, first.values.len()), (125, 2048));
    assert_eq!(f64::from(first.values[0]), 13.612717628479004);
    assert_eq!(m.last_call, Some(125));
    assert_eq!(
        m.dataset.as_deref(),
        Some("/training_dir/calibration_datav3.txt")
    );
    eprintln!("RAN: 112 entries, 327680 values from {path}");
}
