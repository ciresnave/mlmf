// SPDX-License-Identifier: MIT OR Apache-2.0
//! The temp file of a dropped download is named by APPENDING `.part` to the
//! full file name (S-64). `with_extension("part")` replaced the extension, so
//! `model.gguf` and `model.bin` both used `model.part` and could not be told
//! apart (or fetched concurrently) in one directory.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::Path;
use std::thread;

use mlmf_source_hub::{HubSource, Revision};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

/// Serves any request with a `Content-Length` of 10 and 3 bytes, then closes.
fn truncating_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            thread::spawn(move || {
                let mut reader = BufReader::new(stream.try_clone().expect("clone"));
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                        break;
                    }
                }
                let mut out = stream;
                let _ = out.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nabc",
                );
            });
        }
    });
    endpoint
}

fn dropped_fetch(cache: &Path, filename: &str) {
    let hub = HubSource::new(cache).with_endpoint(truncating_endpoint());
    let result = hub.fetch("org/model", &Revision::Pinned(SHA.into()), filename);
    assert!(result.is_err(), "a short body must be an error");
}

#[test]
fn a_dropped_download_leaves_the_full_name_plus_part() {
    let cache = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("mlmf-part-name-{}", std::process::id()));
    dropped_fetch(&cache, "model.gguf");
    dropped_fetch(&cache, "model.bin");
    let dir = cache.join("org/model").join(SHA);
    let present = |n: &str| dir.join(n).exists();
    assert!(present("model.gguf.part"), "model.gguf.part missing");
    assert!(present("model.bin.part"), "model.bin.part missing");
    assert!(!present("model.part"), "shared model.part must not exist");
    assert!(
        !present("model.gguf"),
        "a short file must never be at the real name"
    );
    let _ = std::fs::remove_dir_all(&cache);
}
