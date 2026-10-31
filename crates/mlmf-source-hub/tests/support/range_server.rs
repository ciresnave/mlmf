// SPDX-License-Identifier: MIT OR Apache-2.0
//! A loopback HTTP server that serves one blob the way the Hub serves a
//! file: `GET /{repo}/resolve/{sha}/{file}` with optional `Range`, and
//! `GET /api/models/{repo}/revision/{ref}` answering `{"sha": ...}`.
//!
//! Std only, one thread per connection, `Connection: close`. It exists so
//! the Hub range source can be tested without a network, and it COUNTS what
//! it serves, because "reads only the header" is a claim about bytes on the
//! wire, which only the server side can state.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// How the server answers a `Range` request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 206 with exactly the requested bytes.
    Honest,
    /// Ignores `Range` and sends the whole blob with 200.
    IgnoresRange,
    /// 206 whose Content-Length and body are one byte short of the range.
    ShortBody,
    /// 206 whose Content-Length and body are one byte past the range.
    LongBody,
    /// 500 for every request on the file path.
    Fails,
}

/// One request the server saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seen {
    /// The request path.
    pub path: String,
    /// The `Range` header, verbatim.
    pub range: Option<String>,
    /// Whether an `Authorization` header was sent.
    pub authorized: bool,
}

struct State {
    blob: Vec<u8>,
    mode: Mode,
    sha: String,
    served: AtomicU64,
    seen: Mutex<Vec<Seen>>,
}

/// A running server. Dropping it does not stop the thread; the process
/// ending does, which is all a test needs.
pub struct RangeServer {
    addr: String,
    state: Arc<State>,
}

impl RangeServer {
    /// Serve `blob` in `mode`, resolving every ref to `sha`.
    ///
    /// # Panics
    ///
    /// If no loopback port can be bound.
    #[must_use]
    pub fn start(blob: Vec<u8>, mode: Mode, sha: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = format!("http://{}", listener.local_addr().expect("local addr"));
        let state = Arc::new(State {
            blob,
            mode,
            sha: sha.to_string(),
            served: AtomicU64::new(0),
            seen: Mutex::new(Vec::new()),
        });
        let shared = Arc::clone(&state);
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let state = Arc::clone(&shared);
                thread::spawn(move || handle(stream, &state));
            }
        });
        Self { addr, state }
    }

    /// `http://127.0.0.1:PORT`, the value to use as the Hub endpoint.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.addr
    }

    /// Body bytes sent for file requests so far (API answers excluded).
    #[must_use]
    pub fn bytes_served(&self) -> u64 {
        self.state.served.load(Ordering::SeqCst)
    }

    /// Every request seen so far, in arrival order.
    ///
    /// # Panics
    ///
    /// If the log's lock was poisoned by a panicking handler.
    #[must_use]
    pub fn requests(&self) -> Vec<Seen> {
        self.state.seen.lock().expect("log").clone()
    }
}

fn handle(stream: TcpStream, state: &State) {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return;
    }
    let path = request_line.split(' ').nth(1).unwrap_or("").to_string();
    let (mut range, mut authorized) = (None, false);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("range:") {
            range = Some(v.trim().to_string());
        }
        if lower.starts_with("authorization:") {
            authorized = true;
        }
    }
    state.seen.lock().expect("log").push(Seen {
        path: path.clone(),
        range: range.clone(),
        authorized,
    });
    let mut stream = reader.into_inner();
    let _ = if path.starts_with("/api/") {
        let body = format!("{{\"sha\":\"{}\"}}", state.sha);
        respond(&mut stream, "200 OK", &[], body.as_bytes())
    } else {
        answer_file(&mut stream, state, range.as_deref())
    };
    let _ = stream.flush();
}

fn answer_file(stream: &mut TcpStream, state: &State, range: Option<&str>) -> std::io::Result<()> {
    let len = state.blob.len() as u64;
    if state.mode == Mode::Fails {
        return respond(stream, "500 Internal Server Error", &[], b"");
    }
    let wanted = range.and_then(parse_range);
    let (Some((start, end)), true) = (wanted, state.mode != Mode::IgnoresRange) else {
        state.served.fetch_add(len, Ordering::SeqCst);
        return respond(stream, "200 OK", &[], &state.blob);
    };
    if start >= len {
        let h = [format!("Content-Range: bytes */{len}")];
        return respond(stream, "416 Range Not Satisfiable", &h, b"");
    }
    let end = end.min(len - 1);
    let mut body = state.blob[start as usize..=end as usize].to_vec();
    let header = format!("Content-Range: bytes {start}-{end}/{len}");
    match state.mode {
        Mode::ShortBody => {
            body.pop();
        }
        Mode::LongBody => body.push(0),
        _ => {}
    }
    state.served.fetch_add(body.len() as u64, Ordering::SeqCst);
    respond(stream, "206 Partial Content", &[header], &body)
}

/// `bytes=a-b` as an inclusive pair.
fn parse_range(value: &str) -> Option<(u64, u64)> {
    let spec = value.trim().strip_prefix("bytes=")?;
    let (a, b) = spec.split_once('-')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    extra: &[String],
    body: &[u8],
) -> std::io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for h in extra {
        head.push_str(h);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)
}
