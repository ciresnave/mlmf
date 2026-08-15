//! Types no ordinary file carries, with sizes derived by hand from the
//! block layouts in ggml-common.h rather than from the code under test.

use mlmf_ggml::GgmlType;

/// One committed size claim: `(type, elements_per_block, expected bytes for
/// one block)`.
///
/// `expected` is written as the arithmetic that produces it — the
/// constituent fields of the block struct in ggml-common.h — so a reader
/// can check each row against that header without trusting this table.
/// Do not collapse these to their totals: the sum is the claim under test,
/// and the summands are the evidence for it.
///
/// This is the single source both the grouped tests below and the
/// completeness guard (`every_live_code_is_covered_by_a_test_somewhere`)
/// iterate. Deleting a row here removes both the coverage AND the guard's
/// record of it, which a hand-maintained list of claims about coverage
/// could not do.
const AUTHORED: [(GgmlType, u64, u64); 21] = [
    // block_q2_K: scales[256/16] + qs[256/4] + union{d,dmin | dm} (4)
    (GgmlType::Q2_K, 256, 16 + 64 + 4),
    // block_q8_K: f32 d + qs[256] + bsums[256/16] i16
    (GgmlType::Q8_K, 256, 4 + 256 + 32),
    // block_iq2_xxs: f16 d + qs[256/8] u16
    (GgmlType::IQ2_XXS, 256, 2 + 64),
    // block_iq2_xs: f16 d + qs[256/8] u16 + scales[256/32]
    (GgmlType::IQ2_XS, 256, 2 + 64 + 8),
    // block_iq2_s: f16 d + qs[256/4] + qh[256/32] + scales[256/32]
    (GgmlType::IQ2_S, 256, 2 + 64 + 8 + 8),
    // block_iq3_xxs: f16 d + qs[3*256/8]
    (GgmlType::IQ3_XXS, 256, 2 + 96),
    // block_iq1_s: f16 d + qs[256/8] + qh[256/32] u16
    (GgmlType::IQ1_S, 256, 2 + 32 + 16),
    // block_iq1_m: qs[256/8] + qh[256/16] + scales[256/32]  (no scale field)
    (GgmlType::IQ1_M, 256, 32 + 16 + 8),
    // block_tq1_0: qs[(256 - 4*256/64)/5] + qh[256/64] + f16 d
    (GgmlType::TQ1_0, 256, 48 + 4 + 2),
    // block_tq2_0: qs[256/4] + f16 d
    (GgmlType::TQ2_0, 256, 64 + 2),
    // block_mxfp4: u8 e + qs[32/2]  -- 17 bytes, an ODD block size
    (GgmlType::MXFP4, 32, 1 + 16),
    // block_nvfp4: d[64/16] + qs[64/2]
    (GgmlType::NVFP4, 64, 4 + 32),
    // block_q1_0: f16 d + qs[128/8]
    (GgmlType::Q1_0, 128, 2 + 16),
    // block_q2_0: f16 d + qs[64/4]
    (GgmlType::Q2_0, 64, 2 + 16),
    // block_q8_1: union{d,s | ds} (4) + qs[32]
    (GgmlType::Q8_1, 32, 4 + 32),
    // No blocks: one element, one width.
    (GgmlType::I8, 1, 1),
    (GgmlType::I16, 1, 2),
    (GgmlType::I32, 1, 4),
    (GgmlType::I64, 1, 8),
    (GgmlType::F64, 1, 8),
    (GgmlType::BF16, 1, 2),
];

/// One block's worth, then seven blocks' worth, checked against a row of
/// [`AUTHORED`].
fn check((t, elems_per_block, expected_one): (GgmlType, u64, u64)) {
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
    for row in AUTHORED[0..8].iter().copied() {
        check(row);
    }
}

#[test]
fn ternary_and_fp4_quants() {
    for row in AUTHORED[8..12].iter().copied() {
        check(row);
    }
}

#[test]
fn the_recent_low_bit_types() {
    for row in AUTHORED[12..15].iter().copied() {
        check(row);
    }
}

#[test]
fn integer_and_wide_dense_types() {
    for row in AUTHORED[15..21].iter().copied() {
        check(row);
    }
}

/// Distinct type codes named in `corpus-sizes.tsv`'s `code` column.
///
/// Derived from the fixture rather than retyped as a literal array: the
/// fixture is a correct hand-transcription today, but a second, independent
/// hand-transcription of the same facts is exactly the kind of duplicate
/// that drifts silently when one copy is updated and the other is not.
fn from_corpus() -> Vec<u32> {
    let mut codes: Vec<u32> = include_str!("corpus-sizes.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("code\t") && !l.trim().is_empty())
        .map(|l| {
            l.split('\t')
                .next()
                .expect("every row has at least one field")
                .parse()
                .expect("the code column is numeric")
        })
        .collect();
    codes.sort_unstable();
    codes.dedup();
    codes
}

#[test]
fn every_live_code_is_covered_by_a_test_somewhere() {
    // The completeness guard. Unlike the version this replaced, neither
    // side of the coverage set is a hand-maintained list of *claims* about
    // coverage: `from_corpus` is parsed out of the fixture the corpus test
    // actually reads, and `AUTHORED` is the same table the tests above
    // actually iterate. So deleting a row from either one removes coverage
    // and fails this guard in the same commit — there is no longer a
    // separate list that can go stale while the real coverage moves on.
    let from_corpus = from_corpus();
    let mut from_authored: Vec<u32> = AUTHORED.iter().map(|(t, _, _)| t.code()).collect();
    from_authored.sort_unstable();
    from_authored.dedup();

    for t in GgmlType::ALL {
        let c = t.code();
        assert!(
            from_corpus.contains(&c) || from_authored.contains(&c),
            "{} (code {c}) is in the table but covered by no size test",
            t.name()
        );
    }
    assert_eq!(
        from_corpus.len() + from_authored.len(),
        35,
        "the corpus and authored coverage sets overlap, or a code is \
         covered by neither"
    );
}
