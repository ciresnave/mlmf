// SPDX-License-Identifier: MIT OR Apache-2.0
//! Resuming a dropped download with `Range` (S-64). A `.part` is resumed only
//! when its sidecar names the same repo, revision and file; a server that
//! ignores `Range` restarts the download; the result is byte-identical to the
//! file either way.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use mlmf_source_hub::{HubSource, Progress, Revision};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const WANT: &str = "/org/model/resolve/0123456789abcdef0123456789abcdef01234567/w.bin";
const LEN: usize = 200_000;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Answers a `Range` with 206.
    Honour,
    /// Answers every request with the whole file and a 200.
    Ignore,
    /// Honours `Range`, but the FIRST request is cut off after 50000 bytes.
    DropFirst,
    /// Answers a `Range` with a 206 that starts at byte 0, not where asked.
    WrongStart,
}

fn body() -> Vec<u8> {
    (0..LEN).map(|i| (i % 251) as u8).collect()
}

struct Server {
    endpoint: String,
    /// The `Range` header of each request, in order (`None` if absent).
    ranges: Arc<Mutex<Vec<Option<String>>>>,
}

fn start(mode: Mode) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
    let ranges = Arc::new(Mutex::new(Vec::new()));
    let (seen, count) = (Arc::clone(&ranges), Arc::new(AtomicUsize::new(0)));
    let all = Arc::new(body());
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let (seen, count, all) = (Arc::clone(&seen), Arc::clone(&count), Arc::clone(&all));
            thread::spawn(move || serve(stream, mode, &seen, &count, &all));
        }
    });
    Server { endpoint, ranges }
}

/// The request target and its `Range` header, if any.
fn read_request(stream: &std::net::TcpStream) -> (String, Option<String>) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut request_line = String::new();
    let _ = reader.read_line(&mut request_line);
    let mut range = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("range:") {
            range = Some(v.trim().to_string());
        }
    }
    let target = request_line.split_whitespace().nth(1).unwrap_or("");
    (target.to_string(), range)
}

/// The `N` of a `bytes=N-` header.
fn range_start(range: Option<&str>) -> Option<usize> {
    range?
        .strip_prefix("bytes=")?
        .strip_suffix('-')?
        .parse()
        .ok()
}

fn partial(first: usize, length: usize) -> String {
    format!(
        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {first}-{}/{LEN}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n",
        LEN - 1
    )
}

/// The reply head and the body bytes that follow it.
fn answer<'a>(mode: Mode, range: Option<&str>, all: &'a [u8]) -> (String, &'a [u8]) {
    if mode == Mode::WrongStart && range.is_some() {
        return (partial(0, LEN), all);
    }
    let start = if mode == Mode::Ignore {
        None
    } else {
        range_start(range)
    };
    match start {
        Some(s) => (partial(s, LEN - s), &all[s..]),
        None => (
            format!("HTTP/1.1 200 OK\r\nContent-Length: {LEN}\r\nConnection: close\r\n\r\n"),
            all,
        ),
    }
}

fn serve(
    stream: std::net::TcpStream,
    mode: Mode,
    seen: &Mutex<Vec<Option<String>>>,
    count: &AtomicUsize,
    all: &[u8],
) {
    let (target, range) = read_request(&stream);
    let mut out = stream;
    if target != WANT {
        let _ = out
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return;
    }
    seen.lock().expect("lock").push(range.clone());
    let nth = count.fetch_add(1, Ordering::SeqCst);
    let (head, tail) = answer(mode, range.as_deref(), all);
    let cut = if mode == Mode::DropFirst && nth == 0 {
        50_000
    } else {
        tail.len()
    };
    let _ = out.write_all(head.as_bytes());
    let _ = out.write_all(&tail[..cut]);
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("resume-{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir");
    dir
}

fn identity(sha: &str) -> String {
    format!("org/model\n{sha}\nw.bin\n")
}

/// A `.part` of the first `n` bytes with a sidecar naming `sha`.
fn seed(dest: &Path, n: usize, fill: Option<u8>, sha: &str) {
    let bytes: Vec<u8> = match fill {
        Some(b) => vec![b; n],
        None => body()[..n].to_vec(),
    };
    fs::write(dest.with_file_name("w.bin.part"), bytes).expect("seed part");
    fs::write(dest.with_file_name("w.bin.part.meta"), identity(sha)).expect("seed meta");
}

fn run(server: &Server, dest: &Path, progress: &mut Vec<Progress>) -> Result<String, String> {
    let hub = HubSource::new(dest.with_file_name("cache")).with_endpoint(server.endpoint.clone());
    hub.fetch_to_with_progress(
        "org/model",
        &Revision::Pinned(SHA.into()),
        "w.bin",
        dest,
        &mut |p| progress.push(p),
    )
    .map_err(|e| e.to_string())
}

fn ranges(server: &Server) -> Vec<Option<String>> {
    server.ranges.lock().expect("lock").clone()
}

#[test]
fn a_matching_part_is_resumed_and_only_the_tail_is_requested() {
    let dest = scratch("tail").join("w.bin");
    seed(&dest, 70_000, None, SHA);
    let server = start(Mode::Honour);
    let mut progress = Vec::new();
    run(&server, &dest, &mut progress).expect("resume");
    assert_eq!(fs::read(&dest).expect("dest"), body());
    assert_eq!(ranges(&server), vec![Some("bytes=70000-".to_string())]);
    assert!(progress.first().is_some_and(|p| p.downloaded > 70_000));
    assert_eq!(progress.last().map(|p| p.downloaded), Some(LEN as u64));
    assert!(progress.iter().all(|p| p.total == Some(LEN as u64)));
    assert!(!dest.with_file_name("w.bin.part.meta").exists());
}

#[test]
fn a_server_that_ignores_range_restarts_the_download() {
    let dest = scratch("ignore").join("w.bin");
    seed(&dest, 70_000, Some(0xFF), SHA);
    let server = start(Mode::Ignore);
    run(&server, &dest, &mut Vec::new()).expect("restart");
    assert_eq!(fs::read(&dest).expect("dest"), body());
}

#[test]
fn a_part_from_another_revision_is_discarded_not_resumed() {
    let dest = scratch("other-sha").join("w.bin");
    seed(
        &dest,
        70_000,
        Some(0xFF),
        "ffffffffffffffffffffffffffffffffffffffff",
    );
    let server = start(Mode::Honour);
    run(&server, &dest, &mut Vec::new()).expect("fresh");
    assert_eq!(fs::read(&dest).expect("dest"), body());
    assert_eq!(ranges(&server), vec![None], "no Range may be sent");
}

#[test]
fn a_part_with_no_sidecar_is_discarded_not_resumed() {
    let dest = scratch("no-meta").join("w.bin");
    seed(&dest, 70_000, Some(0xFF), SHA);
    fs::remove_file(dest.with_file_name("w.bin.part.meta")).expect("rm meta");
    let server = start(Mode::Honour);
    run(&server, &dest, &mut Vec::new()).expect("fresh");
    assert_eq!(fs::read(&dest).expect("dest"), body());
    assert_eq!(ranges(&server), vec![None]);
}

#[test]
fn a_dropped_download_is_resumed_by_the_next_call() {
    let dest = scratch("drop-then-resume").join("w.bin");
    let server = start(Mode::DropFirst);
    assert!(run(&server, &dest, &mut Vec::new()).is_err());
    assert!(!dest.exists(), "a short file must never be at dest");
    let part = fs::metadata(dest.with_file_name("w.bin.part")).expect("part");
    assert_eq!(part.len(), 50_000);
    run(&server, &dest, &mut Vec::new()).expect("resume");
    assert_eq!(fs::read(&dest).expect("dest"), body());
    assert_eq!(
        ranges(&server),
        vec![None, Some("bytes=50000-".to_string())]
    );
}

#[test]
fn a_206_that_starts_elsewhere_is_not_appended_to_the_part() {
    let dest = scratch("wrong-start").join("w.bin");
    seed(&dest, 70_000, None, SHA);
    let server = start(Mode::WrongStart);
    run(&server, &dest, &mut Vec::new()).expect("fresh");
    assert_eq!(fs::read(&dest).expect("dest"), body());
    assert_eq!(
        ranges(&server),
        vec![Some("bytes=70000-".to_string()), None],
        "the odd reply is abandoned for one plain request"
    );
}
