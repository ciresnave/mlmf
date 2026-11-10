// SPDX-License-Identifier: MIT OR Apache-2.0
//! `RemoteHash::verify` checks a file on disk against the hash the Hub
//! reported (S-64 slice 2). The expected values are independent of the code
//! under test: `git hash-object` and `sha256sum` of the same bytes.

use std::fs;
use std::path::PathBuf;

use mlmf_source_hub::RemoteHash;

const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const ABC_BLOB: &str = "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f";
const EMPTY_BLOB: &str = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391";

fn file(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("verify");
    fs::create_dir_all(&dir).expect("dir");
    let p = dir.join(name);
    fs::write(&p, bytes).expect("write");
    p
}

#[test]
fn a_matching_sha256_verifies() {
    let p = file("abc-a", b"abc");
    RemoteHash::Sha256(ABC_SHA256.into())
        .verify(&p)
        .expect("matches");
}

#[test]
fn a_different_sha256_is_refused() {
    let p = file("abd", b"abd");
    let err = RemoteHash::Sha256(ABC_SHA256.into())
        .verify(&p)
        .expect_err("mismatch");
    assert!(err.to_string().contains("sha256"), "{err}");
}

#[test]
fn a_matching_git_blob_sha1_verifies() {
    let p = file("abc-b", b"abc");
    RemoteHash::GitBlobSha1(ABC_BLOB.into())
        .verify(&p)
        .expect("matches");
}

#[test]
fn the_blob_hash_covers_the_size_prefix_not_just_the_bytes() {
    // sha1("abc") alone, with no "blob 3\0" header, is not the blob id.
    let p = file("abc-c", b"abc");
    let plain_sha1 = "a9993e364706816aba3e25717850c26c9cd0d89d";
    RemoteHash::GitBlobSha1(plain_sha1.into())
        .verify(&p)
        .expect_err("a bare sha1 is not a blob id");
}

#[test]
fn an_empty_file_verifies_against_the_empty_blob() {
    let p = file("empty", b"");
    RemoteHash::GitBlobSha1(EMPTY_BLOB.into())
        .verify(&p)
        .expect("matches");
}

#[test]
fn upper_case_hex_from_the_hub_still_matches() {
    let p = file("abc-d", b"abc");
    RemoteHash::Sha256(ABC_SHA256.to_uppercase())
        .verify(&p)
        .expect("hex case is not content");
}

#[test]
fn a_missing_file_is_an_error_not_a_match() {
    let p = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("verify/none");
    RemoteHash::Sha256(ABC_SHA256.into())
        .verify(&p)
        .expect_err("no file");
}
