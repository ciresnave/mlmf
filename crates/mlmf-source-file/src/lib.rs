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
//! # Two acquisition paths, one of them default
//!
//! [`FileSource::open`] maps the file when the **default** `mmap` feature is
//! on and reads it into an owned buffer when it is not. Spec §3.4 asks for
//! exactly that — *"`memmap2` is a **default** feature of
//! `mlmf-source-file`"* — and C6 asks for the other half: CI *builds and
//! runs* this crate's tests with `--no-default-features`, so the mmap-free
//! path is proven functional rather than merely compilable. It was also
//! written first, one task earlier, because a path written second is a path
//! written to match the first.
//!
//! [`FileSource::open_read`] is the plain read on **either** build. It is
//! permanent public API; its own doc says why.
//!
//! # This crate materializes the whole file
//!
//! [`FileSource::as_bytes`](mlmf_core::ByteSource::as_bytes) hands out one
//! slice of the whole file's bytes. There is no `File` handle, no cursor and
//! no `seek` anywhere in this crate. That is a decision rather than a
//! shortcut:
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
//!
//! **What "materialized" costs is not the same on the two builds, and the
//! `--no-default-features` one is the expensive one.** A mapping is paged in
//! by the OS on demand, so the default build does not hold the file in RAM
//! and [`mlmf_core::RangedSource::read_range`] over it touches only the
//! pages the range covers. Without the feature, `open` allocates the entire
//! file through `std::fs::read` before anything is read out of it, and
//! `read_range` is then a copy out of an allocation that is already there:
//! **no memory benefit at all, on the build C6 exists to protect.** Said
//! here because the lazy-paging argument above is true of the default build
//! only, and a reader who carries it across to the other one will size a
//! process wrongly by the size of the model.
//!
//! # It also lists a directory, and that is the whole of what it knows
//!
//! [`read_dir`] returns the immediate children of a local directory, sorted
//! by name, each with a flag saying whether it is itself a directory.
//! Spec §3.2 assigns the walk here — *"`mlmf-source-file` walks a local
//! directory"* — and assigns the reading of it elsewhere: `mlmf-hf-layout`
//! *"never enumerates a directory"*. **The split is the point.** This crate
//! does not know what a checkpoint looks like, does not map an extension to
//! a format, and does not decide which of the files it found is a model.
//! Given a directory holding `model.safetensors`, `model.gguf` and a
//! `README.md`, it returns all three, because the charter says *"MLMF is
//! never intended to be an interpreter of the content of model files"* and
//! an enumerator that filtered would be interpreting.
//!
//! [`DirEntry::name`] is an [`std::ffi::OsString`], which is the one API
//! decision in that module worth arguing about; its own documentation makes
//! the argument.
//!
//! # Why this crate does not `forbid(unsafe_code)`
//!
//! `memmap2::Mmap::map` is an `unsafe fn` — a mapping is a window onto bytes
//! another process can truncate underneath it, and no argument this crate
//! can make changes that. So `#![forbid(unsafe_code)]`, which nothing can
//! override, would make the feature spec §3.4 requires unimplementable.
//!
//! Four of the five crates that existed before this one carry the forbid
//! (`mlmf-core`, `mlmf-ggml`, `mlmf-gguf`, `mlmf-safetensors`;
//! `mlmf-conformance` has no crate-level attributes at all) and **no gate
//! checks for it**, so this is a convention worth honouring rather than a
//! rule being broken. It is honoured as far as it goes: the attribute below
//! is `deny`, not `forbid`, which keeps every other `unsafe` in this crate a
//! compile error, and the one exception is an `#[expect(unsafe_code)]`
//! carrying its reason on the single private function that maps a file.
//! There is exactly one `unsafe` block in this crate, in `file.rs`, with the
//! argument for it written beside it.

#![deny(unsafe_code)]
#![warn(missing_docs)]

// Both modules are declared here and nowhere else. A `.rs` file under
// `src/` that no `mod` names is not compiled and its tests never run, with
// nothing going red — `mlmf-core/tests/module_registration.rs` is the gate
// that says so, in those words.
mod dir;
mod file;

// Re-exported flat, so a consumer writes `mlmf_source_file::read_dir` and
// not `mlmf_source_file::dir::read_dir`. The modules are an organisation of
// this crate's source and not part of its surface: `file` and `dir` are
// private, so moving an item between them is not a breaking change.
pub use dir::{DirEntry, read_dir};
pub use file::FileSource;
