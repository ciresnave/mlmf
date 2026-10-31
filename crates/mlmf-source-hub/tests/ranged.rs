// SPDX-License-Identifier: MIT OR Apache-2.0
//! `HubSource::open_ranged` against a loopback server that counts the bytes
//! it serves (S-08 B). No network: the claim "reads only what it asks for"
//! is a claim about bytes on the wire, so the SERVER counts them.

use mlmf_core::{ErrorKind, RangedSource};
use mlmf_source_hub::{HubSource, Revision};
mod support;
use support::range_server::{Mode, RangeServer};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const REPO: &str = "org/model";
const FILE: &str = "model.safetensors";

fn blob(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn hub(server: &RangeServer) -> HubSource {
    HubSource::new("/unused-cache-dir").with_endpoint(server.endpoint())
}

fn open(server: &RangeServer) -> mlmf_source_hub::HubRangedFile {
    hub(server)
        .open_ranged(REPO, &Revision::Pinned(SHA.into()), FILE)
        .expect("opens")
}

#[test]
fn open_learns_the_length_from_one_byte_and_records_the_revision() {
    let server = RangeServer::start(blob(100_000), Mode::Honest, SHA);
    let file = open(&server);

    assert_eq!(file.len(), Some(100_000));
    assert_eq!(file.resolved_revision(), SHA);
    assert_eq!(server.bytes_served(), 1, "only the probe byte moved");
    let seen = server.requests();
    assert_eq!(
        seen.len(),
        1,
        "a pinned revision makes no API call: {seen:?}"
    );
    assert_eq!(seen[0].path, format!("/{REPO}/resolve/{SHA}/{FILE}"));
    assert_eq!(seen[0].range.as_deref(), Some("bytes=0-0"));
}

#[test]
fn a_read_returns_exactly_the_bytes_asked_and_moves_only_those() {
    let data = blob(100_000);
    let server = RangeServer::start(data.clone(), Mode::Honest, SHA);
    let file = open(&server);

    let mut buf = vec![0u8; 5000];
    file.read_range(40_000..45_000, &mut buf).expect("reads");
    assert_eq!(buf, data[40_000..45_000]);
    assert_eq!(server.bytes_served(), 1 + 5000);
    assert_eq!(
        server
            .requests()
            .last()
            .and_then(|s| s.range.clone())
            .as_deref(),
        Some("bytes=40000-44999")
    );
}

/// The shape of a safetensors header read, with no format crate: 8 bytes,
/// then the `N` they declare. The server's byte count is the proof.
#[test]
fn a_length_prefixed_header_read_moves_the_header_and_not_the_tensors() {
    let header_len: u64 = 300;
    let mut data = header_len.to_le_bytes().to_vec();
    data.extend(vec![b' '; header_len as usize]);
    data.extend(blob(2 << 20));
    let server = RangeServer::start(data.clone(), Mode::Honest, SHA);
    let file = open(&server);

    let mut prefix = [0u8; 8];
    file.read_range(0..8, &mut prefix).expect("prefix");
    let n = u64::from_le_bytes(prefix);
    let mut header = vec![0u8; n as usize];
    file.read_range(8..8 + n, &mut header).expect("header");

    assert_eq!(header, data[8..8 + n as usize]);
    assert_eq!(server.bytes_served(), 1 + 8 + n);
    assert!(
        server.bytes_served() * 1000 < data.len() as u64,
        "{} of {} bytes",
        server.bytes_served(),
        data.len()
    );
}

#[test]
fn a_range_past_the_end_is_truncated_without_a_request() {
    let server = RangeServer::start(blob(1000), Mode::Honest, SHA);
    let file = open(&server);
    let before = server.requests().len();

    let mut buf = vec![0u8; 10];
    let err = file.read_range(995..1005, &mut buf).unwrap_err();
    assert!(
        matches!(
            err.kind(),
            ErrorKind::Truncated {
                needed: 1005,
                available: 1000
            }
        ),
        "{err:?}"
    );
    assert_eq!(server.requests().len(), before, "refused before the wire");
}

#[test]
fn a_bad_call_is_a_source_error_without_a_request() {
    let server = RangeServer::start(blob(1000), Mode::Honest, SHA);
    let file = open(&server);
    let before = server.requests().len();

    let (start, end) = (10u64, 5u64); // built at run time: a literal 10..5 is a lint error
    let inverted = file.read_range(start..end, &mut []).unwrap_err();
    let mismatch = file.read_range(0..10, &mut [0u8; 4]).unwrap_err();
    assert!(
        matches!(inverted.kind(), ErrorKind::Source(_)),
        "{inverted:?}"
    );
    assert!(
        matches!(mismatch.kind(), ErrorKind::Source(_)),
        "{mismatch:?}"
    );
    assert_eq!(server.requests().len(), before);
}

#[test]
fn an_empty_range_is_ok_and_asks_for_nothing() {
    let server = RangeServer::start(blob(1000), Mode::Honest, SHA);
    let file = open(&server);
    let before = server.requests().len();
    file.read_range(500..500, &mut []).expect("empty range");
    assert_eq!(server.requests().len(), before);
}

#[test]
fn a_server_that_ignores_range_is_an_error_not_a_whole_file_download() {
    let server = RangeServer::start(blob(1000), Mode::IgnoresRange, SHA);
    let err = hub(&server)
        .open_ranged(REPO, &Revision::Pinned(SHA.into()), FILE)
        .unwrap_err();
    assert!(err.to_string().contains("does not honour Range"), "{err}");
}

#[test]
fn a_short_body_and_a_long_body_are_both_source_errors() {
    for mode in [Mode::ShortBody, Mode::LongBody] {
        let server = RangeServer::start(blob(1000), mode, SHA);
        // Opening reads only headers, so it succeeds; the reads must fail.
        let file = open(&server);
        let mut buf = vec![0u8; 100];
        let err = file.read_range(10..110, &mut buf).unwrap_err();
        assert!(
            matches!(err.kind(), ErrorKind::Source(_)),
            "{mode:?}: {err:?}"
        );
    }
}

#[test]
fn a_failing_server_is_a_source_error() {
    let server = RangeServer::start(blob(1000), Mode::Fails, SHA);
    let err = hub(&server)
        .open_ranged(REPO, &Revision::Pinned(SHA.into()), FILE)
        .unwrap_err();
    assert!(err.to_string().contains("opening"), "{err}");
}

#[test]
fn an_empty_file_opens_with_length_zero() {
    let server = RangeServer::start(Vec::new(), Mode::Honest, SHA);
    let file = open(&server);
    assert_eq!(file.len(), Some(0));
    file.read_range(0..0, &mut []).expect("empty read");
}

#[test]
fn a_ref_is_resolved_through_the_api_and_a_pinned_sha_is_not() {
    let server = RangeServer::start(blob(1000), Mode::Honest, SHA);
    let file = hub(&server)
        .open_ranged(REPO, &Revision::Ref("main".into()), FILE)
        .expect("opens by ref");
    assert_eq!(file.resolved_revision(), SHA);
    let paths: Vec<String> = server.requests().into_iter().map(|s| s.path).collect();
    assert_eq!(paths[0], format!("/api/models/{REPO}/revision/main"));
    assert_eq!(paths[1], format!("/{REPO}/resolve/{SHA}/{FILE}"));
}

#[test]
fn the_token_is_sent_on_range_requests() {
    let server = RangeServer::start(blob(1000), Mode::Honest, SHA);
    let file = hub(&server)
        .with_token("hf_secret")
        .open_ranged(REPO, &Revision::Pinned(SHA.into()), FILE)
        .expect("opens");
    let mut buf = [0u8; 4];
    file.read_range(0..4, &mut buf).expect("reads");
    let seen = server.requests();
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|s| s.authorized), "{seen:?}");
}
