//! Types no ordinary file carries, with sizes derived by hand from the
//! block layouts in ggml-common.h rather than from the code under test.

use mlmf_ggml::GgmlType;

/// One block's worth, then two blocks' worth. `expected_one` is written as
/// the arithmetic that produces it, so a reader can check it against the
/// struct without trusting this file.
fn check(t: GgmlType, elems_per_block: u64, expected_one: u64) {
    assert_eq!(
        t.elements_per_block(),
        elems_per_block,
        "{} elements per block",
        t.name()
    );
    assert_eq!(
        t.nbytes(&[elems_per_block], t.name()).unwrap(),
        expected_one,
        "{} one block",
        t.name()
    );
    assert_eq!(
        t.nbytes(&[elems_per_block, 7], t.name()).unwrap(),
        expected_one * 7,
        "{} seven rows",
        t.name()
    );
}

#[test]
fn k_quants_and_i_quants_absent_from_the_corpus() {
    // block_q2_K: scales[256/16] + qs[256/4] + union{d,dmin | dm} (4)
    check(GgmlType::Q2_K, 256, 16 + 64 + 4);
    // block_q8_K: f32 d + qs[256] + bsums[256/16] i16
    check(GgmlType::Q8_K, 256, 4 + 256 + 32);
    // block_iq2_xxs: f16 d + qs[256/8] u16
    check(GgmlType::IQ2_XXS, 256, 2 + 64);
    // block_iq2_xs: f16 d + qs[256/8] u16 + scales[256/32]
    check(GgmlType::IQ2_XS, 256, 2 + 64 + 8);
    // block_iq2_s: f16 d + qs[256/4] + qh[256/32] + scales[256/32]
    check(GgmlType::IQ2_S, 256, 2 + 64 + 8 + 8);
    // block_iq3_xxs: f16 d + qs[3*256/8]
    check(GgmlType::IQ3_XXS, 256, 2 + 96);
    // block_iq1_s: f16 d + qs[256/8] + qh[256/32] u16
    check(GgmlType::IQ1_S, 256, 2 + 32 + 16);
    // block_iq1_m: qs[256/8] + qh[256/16] + scales[256/32]  (no scale field)
    check(GgmlType::IQ1_M, 256, 32 + 16 + 8);
}

#[test]
fn ternary_and_fp4_quants() {
    // block_tq1_0: qs[(256 - 4*256/64)/5] + qh[256/64] + f16 d
    check(GgmlType::TQ1_0, 256, 48 + 4 + 2);
    // block_tq2_0: qs[256/4] + f16 d
    check(GgmlType::TQ2_0, 256, 64 + 2);
    // block_mxfp4: u8 e + qs[32/2]  -- 17 bytes, an ODD block size
    check(GgmlType::MXFP4, 32, 1 + 16);
    // block_nvfp4: d[64/16] + qs[64/2]
    check(GgmlType::NVFP4, 64, 4 + 32);
}

#[test]
fn the_recent_low_bit_types() {
    // block_q1_0: f16 d + qs[128/8]
    check(GgmlType::Q1_0, 128, 2 + 16);
    // block_q2_0: f16 d + qs[64/4]
    check(GgmlType::Q2_0, 64, 2 + 16);
    // block_q8_1: union{d,s | ds} (4) + qs[32]
    check(GgmlType::Q8_1, 32, 4 + 32);
}

#[test]
fn integer_and_wide_dense_types() {
    // No blocks: one element, one width.
    check(GgmlType::I8, 1, 1);
    check(GgmlType::I16, 1, 2);
    check(GgmlType::I32, 1, 4);
    check(GgmlType::I64, 1, 8);
    check(GgmlType::F64, 1, 8);
    check(GgmlType::BF16, 1, 2);
}

#[test]
fn every_live_code_is_covered_by_a_test_somewhere() {
    // The completeness guard. Codes here are covered by the corpus
    // fixture; everything else must be named in this file. If someone adds
    // a 36th type and covers it nowhere, this fails and says which.
    const FROM_CORPUS: [u32; 14] = [0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 14, 20, 21, 23];
    const FROM_THIS_FILE: [u32; 21] = [
        10, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27, 28, 29, 34, 35, 39, 40, 41, 42, 9, 30,
    ];
    for t in GgmlType::ALL {
        let c = t.code();
        assert!(
            FROM_CORPUS.contains(&c) || FROM_THIS_FILE.contains(&c),
            "{} (code {c}) is in the table but covered by no size test",
            t.name()
        );
    }
    assert_eq!(FROM_CORPUS.len() + FROM_THIS_FILE.len(), 35);
}
