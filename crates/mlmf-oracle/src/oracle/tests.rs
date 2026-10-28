// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use mlmf_tensor_names::MappingFile;

const HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

fn p(source: &str, target: &str) -> Proposal {
    Proposal {
        source: source.into(),
        target: target.into(),
    }
}

fn ev() -> Evidence {
    Evidence::oracle("some-model", HASH, "2026-10-26")
}

fn request<'a>(unmapped: &'a [String], candidates: &'a [String]) -> Request<'a> {
    Request {
        direction: Direction::HfToGguf,
        unmapped,
        candidates,
    }
}

#[test]
fn a_valid_proposal_becomes_an_entry_with_the_evidence() {
    let (u, c) = (names(&["a", "b"]), names(&["x", "y", "z"]));
    let got = accept(&request(&u, &c), vec![p("a", "x"), p("b", "y")], &ev());
    assert_eq!(got.rejected, vec![]);
    assert_eq!(got.entries.len(), 2);
    assert_eq!(got.entries[0].evidence, ev());
    // What `accept` returns must be writable as a file.
    assert!(MappingFile::new(Direction::HfToGguf, got.entries).is_ok());
}

#[test]
fn each_rule_drops_with_its_own_reason() {
    let (u, c) = (names(&["a", "b", "d"]), names(&["x", "y"]));
    let got = accept(
        &request(&u, &c),
        vec![p("zzz", "x"), p("a", "nope"), p("b", "y")],
        &ev(),
    );
    let reasons: Vec<_> = got.rejected.iter().map(|r| r.reason).collect();
    assert_eq!(
        reasons,
        vec![
            RejectReason::SourceNotRequested,
            RejectReason::TargetNotCandidate
        ]
    );
    assert_eq!(got.entries.len(), 1);
    assert_eq!(got.entries[0].source, "b");
}

#[test]
fn two_claimants_for_one_target_are_both_dropped() {
    let (u, c) = (names(&["a", "b", "d"]), names(&["x", "y"]));
    let got = accept(
        &request(&u, &c),
        vec![p("a", "x"), p("b", "x"), p("d", "y")],
        &ev(),
    );
    assert_eq!(got.rejected.len(), 2);
    assert!(
        got.rejected
            .iter()
            .all(|r| r.reason == RejectReason::TargetClaimedTwice)
    );
    assert_eq!(got.entries.len(), 1);
    assert_eq!(got.entries[0].source, "d");
}

#[test]
fn a_source_proposed_twice_is_dropped_whole() {
    let (u, c) = (names(&["a"]), names(&["x", "y"]));
    let got = accept(&request(&u, &c), vec![p("a", "x"), p("a", "y")], &ev());
    assert!(got.entries.is_empty());
    assert_eq!(got.rejected.len(), 2);
    assert!(
        got.rejected
            .iter()
            .all(|r| r.reason == RejectReason::SourceProposedTwice)
    );
}

#[test]
fn matching_is_byte_exact() {
    let (u, c) = (names(&["Blk.0.Attn_Q"]), names(&["q_proj"]));
    let got = accept(
        &request(&u, &c),
        vec![p("blk.0.attn_q", "q_proj"), p("Blk.0.Attn_Q", "Q_proj")],
        &ev(),
    );
    assert!(got.entries.is_empty());
    assert_eq!(got.rejected[0].reason, RejectReason::SourceNotRequested);
    assert_eq!(got.rejected[1].reason, RejectReason::TargetNotCandidate);
}

struct Silent;

impl Oracle for Silent {
    fn propose(&self, _request: &Request<'_>) -> Result<Vec<Proposal>, OracleError> {
        Ok(vec![])
    }
}

#[test]
fn an_oracle_that_places_nothing_leaves_every_name_absent() {
    let (u, c) = (names(&["a"]), names(&["x"]));
    let r = request(&u, &c);
    let proposals = Silent.propose(&r).unwrap();
    let got = accept(&r, proposals, &ev());
    assert!(got.entries.is_empty() && got.rejected.is_empty());
}
