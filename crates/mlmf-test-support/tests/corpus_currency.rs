//! U-2 (SPEC section 7): corpus CURRENCY, a check that fails on its own.
//!
//! `corpus-metadata.tsv` is replayed by U-1 and proves the parser AGREES with
//! it. It cannot prove the corpus is CURRENT: constant and file are updated
//! together, so it stays green while upstream ships keys nobody has looked at
//! (Fuel's vendored corpus sat seven releases behind, green throughout).
//! This test reads only `data/corpus-provenance.tsv`, never the metadata
//! table, so it can fail while U-1 passes and the reverse.
//!
//! Policy (S-35, stated in the table header): a recorded fetch is current for
//! `MAX_AGE_DAYS`; past that the test fails until the corpus is re-fetched and
//! the record updated. Offline by design: it measures age, not "upstream moved".

use mlmf_test_support::GGUF_CORPUS_METADATA_TSV;

const PROVENANCE: &str = include_str!("../data/corpus-provenance.tsv");

/// How long a recorded fetch stays current. The policy knob.
const MAX_AGE_DAYS: i64 = 180;

/// `(year, month, day)` of a `YYYY-MM-DD`, with month and day range-checked.
fn parse_ymd(s: &str) -> Option<(i64, i64, i64)> {
    let parts: Vec<i64> = s
        .split('-')
        .map(|p| p.parse().ok())
        .collect::<Option<_>>()?;
    let &[y, m, d] = parts.as_slice() else {
        return None;
    };
    ((1..=12).contains(&m) && (1..=31).contains(&d)).then_some((y, m, d))
}

/// Days since 1970-01-01 of a proleptic-Gregorian `YYYY-MM-DD`, or `None`.
fn days_from_ymd(s: &str) -> Option<i64> {
    let (y, m, d) = parse_ymd(s)?;
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

/// One data row: `(dir, fetched_utc)`. Header and `#` lines are skipped.
fn rows(table: &str) -> Vec<(String, String)> {
    table
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("dir\t") && !l.is_empty())
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            assert_eq!(c.len(), 5, "provenance row must have 5 columns: {l:?}");
            (c[0].to_string(), c[3].to_string())
        })
        .collect()
}

/// Every row that is stale, undated or malformed at `today`, as messages.
fn offences(table: &str, today: i64, max_age: i64) -> Vec<String> {
    rows(table)
        .into_iter()
        .filter_map(|(dir, date)| match days_from_ymd(&date) {
            None => Some(format!("{dir}: fetched_utc {date:?} is not YYYY-MM-DD")),
            Some(d) if today - d > max_age => Some(format!(
                "{dir}: fetched {date}, {} days ago (max {max_age}); re-fetch, re-run the census, record the new date",
                today - d
            )),
            Some(d) if d > today => Some(format!("{dir}: fetched {date} is in the future")),
            Some(_) => None,
        })
        .collect()
}

fn today() -> i64 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock before 1970")
        .as_secs();
    i64::try_from(secs / 86_400).expect("day count fits i64")
}

#[test]
fn the_corpus_is_current() {
    let stale = offences(PROVENANCE, today(), MAX_AGE_DAYS);
    assert!(
        !rows(PROVENANCE).is_empty(),
        "provenance table has no rows: the check is vacuous"
    );
    assert!(
        stale.is_empty(),
        "U-2 corpus currency FAILED:\n{}",
        stale.join("\n")
    );
}

/// Currency must cover the corpus U-1 replays: a tier with rows in the
/// metadata table and none here would be exempt from the check by omission.
#[test]
fn every_replayed_tier_has_a_provenance_row() {
    let recorded: Vec<String> = rows(PROVENANCE).into_iter().map(|(d, _)| d).collect();
    let mut tiers: Vec<&str> = GGUF_CORPUS_METADATA_TSV
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("file\t") && !l.is_empty())
        .filter_map(|l| l.split('\t').next()?.split('/').next())
        .collect();
    tiers.sort_unstable();
    tiers.dedup();
    assert!(
        tiers.len() >= 2,
        "found {tiers:?}: the tier enumeration is blind"
    );
    let missing: Vec<&&str> = tiers
        .iter()
        .filter(|t| !recorded.iter().any(|r| r == **t))
        .collect();
    assert!(
        missing.is_empty(),
        "tiers replayed by U-1 with no provenance row: {missing:?}"
    );
}

// --- the check can fail: pure-function controls, fixed clock ---

const T: &str = "dir\tsource\trevision\tfetched_utc\tbasis\nx\ts\t-\t2026-01-01\tmanifest\n";

#[test]
fn date_arithmetic_is_anchored() {
    assert_eq!(days_from_ymd("1970-01-01"), Some(0));
    assert_eq!(days_from_ymd("2000-03-01"), Some(11_017));
    assert_eq!(
        days_from_ymd("2026-01-31").unwrap() - days_from_ymd("2026-01-01").unwrap(),
        30
    );
    assert_eq!(days_from_ymd("2026-13-01"), None);
    assert_eq!(days_from_ymd("-"), None);
}

#[test]
fn a_fresh_row_passes_and_the_same_row_later_fails() {
    let d0 = days_from_ymd("2026-01-01").unwrap();
    assert!(
        offences(T, d0 + 180, 180).is_empty(),
        "exactly max age is still current"
    );
    let late = offences(T, d0 + 181, 180);
    assert_eq!(late.len(), 1, "one day past max age must fail: {late:?}");
}

#[test]
fn an_undated_or_future_row_fails() {
    let undated = "x\ts\t-\t-\t-\n";
    assert_eq!(offences(undated, 20_000, 180).len(), 1);
    let d0 = days_from_ymd("2026-01-01").unwrap();
    assert_eq!(
        offences(T, d0 - 1, 180).len(),
        1,
        "a future fetch date is not currency"
    );
}
