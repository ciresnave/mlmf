//! Conformance tests for the `mlmf-core` seam, run against every backend.
//!
//! **This crate contains no library code and is never published.** Its
//! entire content is in `tests/`. It exists because of a linking fact with
//! an architectural reason behind it: a test that asserts two backends
//! answer one seam the same way needs both backends in one binary, and
//! there is no correct place inside either backend to put it.
//!
//! # Why not inside a backend
//!
//! The obvious home was `mlmf-safetensors/tests/`, with `mlmf-gguf` as a
//! dev-dependency. That was rejected on the architecture rather than on the
//! mechanism.
//!
//! **Neither backend should know the other exists — that is what a seam
//! is.** `mlmf-gguf` and `mlmf-safetensors` are siblings: both depend on
//! `mlmf-core` and neither depends on the other, so a consumer can take one
//! without the other and a third can be added without touching either. A
//! dev-dependency between them is an edge the seam does not describe, and
//! it is arbitrary in addition — nothing says why safetensors would be the
//! crate that knows about GGUF rather than the reverse.
//!
//! **A cross-backend test is a CONSUMER test.** It does exactly what Fuel
//! and Lightbulb do: hold `&dyn MetadataSource` and `&dyn TensorContainer`
//! and reach whichever backend produced them through `mlmf-core` alone. So
//! it belongs in a crate shaped like a consumer, depending on all three
//! through plain `[dependencies]`. Inside a backend it would be a backend
//! testing its sibling, which is a different thing that happens to compile.
//!
//! # What this crate is for as it grows
//!
//! Named for the job rather than for today's contents: a third backend —
//! spec §11 schedules `mlmf-pickle` — joins this suite by being added to
//! `Cargo.toml` and to the fixtures, with nothing renamed and no new crate.
//! `tests/cross_backend.rs` is the first member of the suite, not the
//! definition of it.
//!
//! # What a conformance suite here cannot do
//!
//! **Two backends is a better sample than one and it is still two.** An
//! assumption that every current backend happens to satisfy is exactly as
//! invisible here as it was with one backend; it is only less likely. The
//! individual test files say which assumptions they are and are not able to
//! see, and that habit is the point of the suite rather than a caveat on
//! it.
