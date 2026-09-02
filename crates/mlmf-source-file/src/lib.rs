//! Opens local files as `mlmf-core` byte sources. I/O only, no format
//! knowledge.
//!
//! Spec §3.1 draws two orthogonal axes: format crates are `bytes ->
//! structure` and do no I/O, source crates are I/O only and know nothing
//! about formats. Every other crate in this workspace is on the format
//! axis. This is the first one on the source axis, and `tests/axis` says so
//! — the two C3 gates read that file, because C3 is scoped verbatim to
//! *"no crate **on the format axis**"* and the gates were not.
//!
//! A source crate depends on [`mlmf_core`] alone. It never depends on a
//! format crate, which is what makes *any source × any format* compose: the
//! consumer picks one of each and neither knows the other exists.
//!
//! # This crate materializes the whole file
//!
//! [`FileSource::open`] reads the file into an owned buffer and
//! [`FileSource::as_bytes`](mlmf_core::ByteSource::as_bytes) hands out a
//! slice of it, so **holding a `FileSource` costs the file's size in
//! memory.** There is no `File` handle, no cursor and no `seek` anywhere in
//! this crate. That is a decision rather than a shortcut:
//! [`mlmf_core::RangedSource::read_range`] takes `&self`, and a
//! `seek`+`read_exact` implementation of it — which compiles, because `std`
//! has `impl Seek for &File` — mutates a **shared file cursor through a
//! shared reference**, so two concurrent reads interleave their seeks and
//! return the wrong bytes with `Ok(())`. The broken state is
//! indistinguishable from the healthy one. A slice has no such state.
//!
//! [`mlmf_core::RangedSource`] exists so that an HTTP range request or an
//! IPC transport is expressible as a source later. That is a fact about
//! *other* implementations of the trait and not a promise this one has to
//! keep by avoiding a slice.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod file;

pub use file::FileSource;
