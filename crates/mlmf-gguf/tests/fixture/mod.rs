//! A GGUF writer, for files no one publishes.
//!
//! The corpus contains zero non-UTF-8 strings, zero trailing-NUL strings,
//! zero declared alignments, zero unknown or retired ggml type codes, zero
//! overlapping tensor ranges, zero duplicate tensor names and zero
//! non-UTF-8 tensor names — measured, by the same reader that produced the
//! layout facts. Every guarantee about those paths is untestable against
//! real files. This builder produces the files that can fail them.
//!
//! Deliberately dumb: it emits exactly what it is told, including things a
//! real writer would refuse. A builder that validated its own output could
//! not produce a malformed fixture. Specifically it does NOT check a
//! declared tensor count against the records it holds, does NOT check a
//! tensor's declared offset against the data region it emits, does NOT
//! check two tensors' ranges against each other, and does NOT refuse a
//! type code — every one of those is a fixture this file exists to make.
//!
//! The single exception is documented on [`GgufBuilder::data`], and it is
//! placement rather than validation.

/// Builds GGUF byte sequences, valid or otherwise.
#[derive(Debug)]
pub struct GgufBuilder {
    version: u32,
    tensor_count: i64,
    infos: Vec<u8>,
    kvs: Vec<u8>,
    kv_count: i64,
    data: Vec<u8>,
}

impl GgufBuilder {
    /// A v3 builder with no tensors and no keys.
    pub fn new() -> Self {
        Self {
            version: 3,
            tensor_count: 0,
            infos: Vec::new(),
            kvs: Vec::new(),
            kv_count: 0,
            data: Vec::new(),
        }
    }

    /// Override the version, including to values this build refuses.
    pub fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }

    /// Override the declared tensor count.
    ///
    /// Applied when called, not at [`Self::build`] time, so a later
    /// [`Self::tensor`] increments from the override. To declare a count
    /// the file does not carry — the eight-bytes-claiming-a-million case —
    /// call this LAST.
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

    /// A tensor-info record with a UTF-8 name.
    ///
    /// `offset` is relative to the start of the data region, which is what
    /// the format records — not to the start of the file. Nothing here
    /// checks it against the bytes [`Self::data`] supplies, or against any
    /// other tensor's range.
    pub fn tensor(self, name: &str, dims: &[u64], code: u32, offset: u64) -> Self {
        self.raw_tensor(name.as_bytes(), dims, code, offset)
    }

    /// A tensor-info record whose name is an arbitrary byte sequence.
    ///
    /// The `raw_string`/`string` split, one stage over and for the same
    /// reason: tensor names are length-prefixed with no terminator, so a
    /// name may hold a NUL or bytes that are not UTF-8 at all, and a
    /// fixture asserting what the reader does with those cannot go through
    /// `&str`.
    pub fn raw_tensor(mut self, name: &[u8], dims: &[u64], code: u32, offset: u64) -> Self {
        Self::push_str(&mut self.infos, name);
        self.infos
            .extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            self.infos.extend_from_slice(&d.to_le_bytes());
        }
        self.infos.extend_from_slice(&code.to_le_bytes());
        self.infos.extend_from_slice(&offset.to_le_bytes());
        self.tensor_count += 1;
        self
    }

    /// The tensor data region's bytes, appended verbatim.
    ///
    /// **The one thing this builder computes**, and the exception the
    /// module doc points at: the padding between the directory's end and
    /// the start of the region, using the same `(a - end % a) % a` rule as
    /// `mlmf_gguf::tensors::data_start` with GGUF's default alignment of
    /// 32. It is computed rather than taken because a hand-written pad that
    /// is wrong moves EVERY tensor in the fixture and the file still
    /// parses, so the bug would surface as a mystery in the assertion
    /// rather than in the fixture.
    ///
    /// It is placement, not validation: no offset, length or overlap is
    /// checked against these bytes, and a fixture may declare a tensor
    /// that begins past their end.
    ///
    /// A fixture that declares `general.alignment` other than 32 gets 32
    /// here anyway and must account for the difference itself — which is
    /// also a usable fixture, since a data region that does not begin where
    /// the declared alignment says it does is a real file shape.
    ///
    /// Nothing is padded when the region is empty, because a real writer
    /// emits no padding for a region it is not writing: that is the shape
    /// of 19 of the 28 readable corpus files, and it is what puts
    /// `data_start` past the end of the file.
    pub fn data(mut self, bytes: &[u8]) -> Self {
        self.data.extend_from_slice(bytes);
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
        out.extend_from_slice(&self.infos);
        if !self.data.is_empty() {
            while out.len() % 32 != 0 {
                out.push(0);
            }
            out.extend_from_slice(&self.data);
        }
        out
    }
}
