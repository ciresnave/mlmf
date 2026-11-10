// SPDX-License-Identifier: MIT OR Apache-2.0
//! `HubSource::list_files` against a loopback server (S-64 slice 1). No
//! network: pagination and "the token stays on the endpoint" are claims
//! about what goes over the wire, so the SERVER records each request.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use mlmf_source_hub::{HubSource, RemoteHash, Revision};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
const BLOB: &str = "6d34772f5ca361021038b404fb913ec8dc0b1a5a";
const REPO: &str = "org/model";
const FIRST: &str =
    "/api/models/org/model/tree/0123456789abcdef0123456789abcdef01234567?recursive=true";
const SECOND: &str =
    "/api/models/org/model/tree/0123456789abcdef0123456789abcdef01234567?recursive=true&cursor=2";

/// A canned reply: body and optional `Link` header.
type Route = (String, Option<String>);

struct Server {
    endpoint: String,
    /// (request target, had an Authorization header)
    seen: Arc<Mutex<Vec<(String, bool)>>>,
}

impl Server {
    /// `routes` maps a request target to its reply; it is built from the
    /// endpoint so a `Link` can name the server's own address.
    fn start(routes: impl FnOnce(&str) -> HashMap<String, Route>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}", listener.local_addr().expect("addr"));
        let routes = Arc::new(routes(&endpoint));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_in = Arc::clone(&seen);
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (routes, seen) = (Arc::clone(&routes), Arc::clone(&seen_in));
                thread::spawn(move || serve(stream, &routes, &seen));
            }
        });
        Self { endpoint, seen }
    }

    fn hub(&self) -> HubSource {
        HubSource::new("/unused-cache-dir")
            .with_endpoint(self.endpoint.clone())
            .with_token("secret-token")
    }

    fn requests(&self) -> Vec<(String, bool)> {
        self.seen.lock().expect("lock").clone()
    }
}

fn serve(stream: TcpStream, routes: &HashMap<String, Route>, seen: &Mutex<Vec<(String, bool)>>) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let target = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
        .to_string();
    let mut authorized = false;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        authorized |= line.to_ascii_lowercase().starts_with("authorization:");
    }
    seen.lock()
        .expect("lock")
        .push((target.clone(), authorized));
    let mut out = stream;
    let reply = match routes.get(&target) {
        Some((body, link)) => {
            let link = link
                .as_ref()
                .map_or(String::new(), |l| format!("Link: {l}\r\n"));
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n{link}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
        }
        None => "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
    };
    let _ = out.write_all(reply.as_bytes());
}

fn entry(path: &str) -> String {
    format!(r#"{{"type":"file","oid":"{BLOB}","size":5,"path":"{path}"}}"#)
}

fn page(paths: &[&str]) -> String {
    let rows: Vec<String> = paths.iter().map(|p| entry(p)).collect();
    format!("[{}]", rows.join(","))
}

fn names(server: &Server) -> Result<Vec<String>, String> {
    server
        .hub()
        .list_files(REPO, &Revision::Pinned(SHA.into()))
        .map(|l| l.files.into_iter().map(|f| f.path).collect())
        .map_err(|e| e.to_string())
}

fn two_pages(second_link: Option<&'static str>) -> Server {
    Server::start(|ep| {
        let next = format!("<{ep}{SECOND}>; rel=\"next\"");
        HashMap::from([
            (FIRST.to_string(), (page(&["a", "b"]), Some(next))),
            (
                SECOND.to_string(),
                (page(&["c"]), second_link.map(str::to_string)),
            ),
        ])
    })
}

#[test]
fn two_pages_are_followed_and_concatenated_in_order() {
    let server = two_pages(None);
    let listing = server
        .hub()
        .list_files(REPO, &Revision::Pinned(SHA.into()))
        .expect("lists");
    assert_eq!(listing.resolved_revision, SHA);
    let paths: Vec<_> = listing.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths, ["a", "b", "c"]);
    assert_eq!(listing.files[0].hash, RemoteHash::GitBlobSha1(BLOB.into()));
    let seen = server.requests();
    assert_eq!(
        seen.len(),
        2,
        "one request per page, none for the revision: {seen:?}"
    );
    assert!(
        seen.iter().all(|(_, auth)| *auth),
        "the token goes to the endpoint"
    );
}

#[test]
fn a_next_link_that_cannot_be_fetched_is_an_error_not_a_short_list() {
    // The control is the test above: with the second page served, the list
    // has three files. Here the link points at a 404, and the answer must
    // not be the two files of page one.
    let server = Server::start(|ep| {
        let next = format!("<{ep}/api/missing>; rel=\"next\"");
        HashMap::from([(FIRST.to_string(), (page(&["a", "b"]), Some(next)))])
    });
    assert!(names(&server).is_err());
}

#[test]
fn a_next_link_to_another_host_is_refused_and_the_token_stays() {
    let server = Server::start(|_| {
        let next = "<http://127.0.0.1:1/evil>; rel=\"next\"".to_string();
        HashMap::from([(FIRST.to_string(), (page(&["a"]), Some(next)))])
    });
    let err = names(&server).expect_err("refused");
    assert!(err.contains("not under"), "{err}");
    assert_eq!(
        server.requests().len(),
        1,
        "only the endpoint was contacted"
    );
}

#[test]
fn a_root_relative_next_link_is_followed_on_the_same_endpoint() {
    let server = Server::start(|_| {
        let next = format!("<{SECOND}>; rel=\"next\"");
        HashMap::from([
            (FIRST.to_string(), (page(&["a"]), Some(next))),
            (SECOND.to_string(), (page(&["b"]), None)),
        ])
    });
    assert_eq!(names(&server), Ok(vec!["a".to_string(), "b".to_string()]));
}

#[test]
fn a_malformed_entry_on_a_later_page_fails_the_whole_listing() {
    let server = Server::start(|ep| {
        let next = format!("<{ep}{SECOND}>; rel=\"next\"");
        HashMap::from([
            (FIRST.to_string(), (page(&["a"]), Some(next))),
            (
                SECOND.to_string(),
                (r#"[{"type":"submodule","path":"s"}]"#.to_string(), None),
            ),
        ])
    });
    assert!(names(&server).is_err());
}
