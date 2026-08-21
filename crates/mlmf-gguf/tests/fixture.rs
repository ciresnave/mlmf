//! A GGUF writer, for files no one publishes.
//!
//! The corpus contains zero non-UTF-8 strings, zero trailing-NUL strings
//! and zero declared alignments, so the guarantees about those paths are
//! untestable against real files. This builder produces the files that can
//! fail them.
//!
//! Deliberately dumb: it emits exactly what it is told, including things a
//! real writer would refuse. A builder that validated its own output could
//! not produce a malformed fixture.

// The builder carries the whole surface the plan specifies, and no test in
// this task drives `tensor_count` — a declared tensor count is the tensor
// stage's business, and that stage does not exist yet. Suppressed rather
// than deleted: removing a method to satisfy a lint would mean the next
// task re-opens this file to add it back, and `cargo clippy --all-targets`
// is a gate this crate holds at `-D warnings`.
#![allow(dead_code)]

/// Builds GGUF byte sequences, valid or otherwise.
#[derive(Debug)]
pub struct GgufBuilder {
    version: u32,
    tensor_count: i64,
    kvs: Vec<u8>,
    kv_count: i64,
}

impl GgufBuilder {
    /// A v3 builder with no tensors and no keys.
    pub fn new() -> Self {
        Self {
            version: 3,
            tensor_count: 0,
            kvs: Vec::new(),
            kv_count: 0,
        }
    }

    /// Override the version, including to values this build refuses.
    pub fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    /// Override the declared tensor count without writing tensors.
    pub fn tensor_count(mut self, n: i64) -> Self {
        self.tensor_count = n;
        self
    }

    fn push_str(buf: &mut Vec<u8>, s: &[u8]) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s);
    }

    /// A UTF-8 string value.
    pub fn string(self, key: &str, value: &str) -> Self {
        self.raw_string(key, value.as_bytes())
    }

    /// A string value of arbitrary bytes — including invalid UTF-8.
    pub fn raw_string(mut self, key: &str, value: &[u8]) -> Self {
        Self::push_str(&mut self.kvs, key.as_bytes());
        self.kvs.extend_from_slice(&8u32.to_le_bytes());
        Self::push_str(&mut self.kvs, value);
        self.kv_count += 1;
        self
    }

    /// A `UINT32` value.
    pub fn u32(mut self, key: &str, value: u32) -> Self {
        Self::push_str(&mut self.kvs, key.as_bytes());
        self.kvs.extend_from_slice(&4u32.to_le_bytes());
        self.kvs.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    /// A key with an arbitrary type code and pre-encoded value bytes.
    pub fn raw_kv(mut self, key: &str, type_code: u32, value: Vec<u8>) -> Self {
        Self::push_str(&mut self.kvs, key.as_bytes());
        self.kvs.extend_from_slice(&type_code.to_le_bytes());
        self.kvs.extend_from_slice(&value);
        self.kv_count += 1;
        self
    }

    /// An array of strings, each an arbitrary byte sequence.
    pub fn string_array(mut self, key: &str, items: &[&[u8]]) -> Self {
        Self::push_str(&mut self.kvs, key.as_bytes());
        self.kvs.extend_from_slice(&9u32.to_le_bytes());
        self.kvs.extend_from_slice(&8u32.to_le_bytes()); // String elements
        self.kvs
            .extend_from_slice(&(items.len() as u64).to_le_bytes());
        for i in items {
            Self::push_str(&mut self.kvs, i);
        }
        self.kv_count += 1;
        self
    }

    /// The bytes.
    pub fn build(self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.tensor_count.to_le_bytes());
        out.extend_from_slice(&self.kv_count.to_le_bytes());
        out.extend_from_slice(&self.kvs);
        out
    }
}

/// Present because `cargo` compiles `tests/fixture.rs` twice — once as a
/// module of `authored`, once as an integration target of its own, where
/// `GgufBuilder` is a crate-root public type and
/// `clippy::new_without_default` fires. `new()` is the name the plan
/// specifies and the name every fixture reads better with, so the trait
/// gets added rather than the constructor renamed.
impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}
