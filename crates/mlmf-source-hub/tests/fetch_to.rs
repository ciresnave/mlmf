// SPDX-License-Identifier: MIT OR Apache-2.0
//! `HubSource::fetch_to` against a loopback server (S-64): the bytes land at
//! the caller's path, the cache is not touched, and a failed download never
//! disturbs a file already at that path.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::thread;

use mlmf_source_hub::{HubSource, Revision};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const WANT: &str = "/org/model/resolve/0123456789abcdef0123456789abcdef01234567/w.bin";

/// Serves `body` for `WANT` (claiming `claimed` bytes, so a `claimed` larger
/// than `body` is a dropped download) and 404 for anything else.
fn endpoint(body: &'static [u8], claimed: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().expect("clone"));
                let mut target = String::new();
                let _ = reader.read_line(&mut target);
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                        break;
                    }
                }
                let mut out = stream;
                if target.split_whitespace().nth(1) == Some(WANT) {
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {claimed}\r\nConnection: close\r\n\r\n"
                    );
                    let _ = out.write_all(head.as_bytes());
                    let _ = out.write_all(body);
                } else {
                    let _ = out.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
            });
        }
    });
    endpoint
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("fetch-to-{name}"));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn pinned() -> Revision {
    Revision::Pinned(SHA.into())
}

#[test]
fn the_bytes_land_at_the_callers_path_and_the_cache_is_untouched() {
    let root = scratch("lands");
    let (cache, dest) = (root.join("cache"), root.join("out/deep/w.bin"));
    let hub = HubSource::new(&cache).with_endpoint(endpoint(b"hello", 5));
    let sha = hub
        .fetch_to("org/model", &pinned(), "w.bin", &dest)
        .expect("fetch_to");
    assert_eq!(sha, SHA);
    assert_eq!(fs::read(&dest).expect("dest"), b"hello");
    assert!(!cache.exists(), "the cache directory must not be created");
    assert!(
        !dest.with_file_name("w.bin.part").exists(),
        "no .part left after success"
    );
}

#[test]
fn a_dropped_download_leaves_an_existing_file_at_dest_untouched() {
    let root = scratch("dropped");
    fs::create_dir_all(&root).expect("mkdir");
    let dest = root.join("w.bin");
    fs::write(&dest, b"old").expect("seed");
    let hub = HubSource::new(root.join("cache")).with_endpoint(endpoint(b"abc", 10));
    let result = hub.fetch_to("org/model", &pinned(), "w.bin", &dest);
    assert!(result.is_err(), "a short body must be an error");
    assert_eq!(fs::read(&dest).expect("dest"), b"old");
}

#[test]
fn a_complete_download_replaces_an_existing_file() {
    let root = scratch("replace");
    fs::create_dir_all(&root).expect("mkdir");
    let dest = root.join("w.bin");
    fs::write(&dest, b"old-and-longer").expect("seed");
    let hub = HubSource::new(root.join("cache")).with_endpoint(endpoint(b"new", 3));
    hub.fetch_to("org/model", &pinned(), "w.bin", &dest)
        .expect("fetch_to");
    assert_eq!(fs::read(&dest).expect("dest"), b"new");
}

#[test]
fn a_missing_file_is_an_error_and_creates_nothing_at_dest() {
    let root = scratch("missing");
    let dest = root.join("nope.bin");
    let hub = HubSource::new(root.join("cache")).with_endpoint(endpoint(b"x", 1));
    let result = hub.fetch_to("org/model", &pinned(), "nope.bin", &dest);
    assert!(result.is_err(), "a 404 must be an error");
    assert!(!dest.exists(), "nothing may appear at dest");
}
