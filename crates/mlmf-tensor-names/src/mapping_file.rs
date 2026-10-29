// SPDX-License-Identifier: MIT OR Apache-2.0
//! The mapping file: extra tensor-name pairs, each with its evidence.
//!
//! This crate reads and validates the file; `mlmf-oracle` is the offline tool
//! that writes proposals into one. Nothing here loads a file by itself.
//!
//! JSON, UTF-8, one document. `format` and `version` are required and an
//! unknown value of either is a read error: there is no best-effort read of a
//! newer file. Names are compared byte-exactly (ROADMAP S-60).

use std::collections::HashSet;
use std::fmt;

use crate::Direction;
use serde_json::{Map, Value, json};

const FORMAT: &str = "mlmf-name-mapping";
const VERSION: u64 = 1;

/// Why a mapping file could not be read, or a [`MappingFile`] built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileError {
    message: String,
}

impl FileError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FileError {}

/// Where an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceKind {
    /// Copied from a declared table.
    Table,
    /// Proposed by a model.
    Oracle,
    /// Written by a person. Never a response to a failed proposal (S-60).
    Human,
}

impl EvidenceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Oracle => "oracle",
            Self::Human => "human",
        }
    }

    fn parse(s: &str) -> Result<Self, FileError> {
        match s {
            "table" => Ok(Self::Table),
            "oracle" => Ok(Self::Oracle),
            "human" => Ok(Self::Human),
            other => Err(FileError::new(format!("unknown evidence kind {other:?}"))),
        }
    }
}

/// An entry's evidence. `model`, `prompt_sha256` and `date` are present for an
/// [`EvidenceKind::Oracle`] entry and absent for the other kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evidence {
    /// Where the entry came from.
    pub kind: EvidenceKind,
    /// The model identifier that made the proposal.
    pub model: Option<String>,
    /// SHA-256 of the exact prompt, 64 lowercase hex digits.
    pub prompt_sha256: Option<String>,
    /// The date of the proposal, `YYYY-MM-DD`.
    pub date: Option<String>,
}

impl Evidence {
    /// Evidence for an entry copied from a declared table.
    #[must_use]
    pub fn table() -> Self {
        Self::bare(EvidenceKind::Table)
    }

    /// Evidence for an entry a person wrote.
    #[must_use]
    pub fn human() -> Self {
        Self::bare(EvidenceKind::Human)
    }

    /// Evidence for a model's proposal; checked when the entry is built.
    #[must_use]
    pub fn oracle(model: &str, prompt_sha256: &str, date: &str) -> Self {
        Self {
            kind: EvidenceKind::Oracle,
            model: Some(model.to_string()),
            prompt_sha256: Some(prompt_sha256.to_string()),
            date: Some(date.to_string()),
        }
    }

    fn bare(kind: EvidenceKind) -> Self {
        Self {
            kind,
            model: None,
            prompt_sha256: None,
            date: None,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), FileError> {
        let fields = [&self.model, &self.prompt_sha256, &self.date];
        let present = fields.iter().filter(|f| f.is_some()).count();
        match (self.kind, present) {
            (EvidenceKind::Oracle, 3) => self.validate_oracle_fields(),
            (EvidenceKind::Oracle, _) => Err(FileError::new(
                "an oracle entry needs model, prompt_sha256 and date",
            )),
            (_, 0) => Ok(()),
            (kind, _) => Err(FileError::new(format!(
                "a {} entry must not carry model, prompt_sha256 or date",
                kind.as_str()
            ))),
        }
    }

    fn validate_oracle_fields(&self) -> Result<(), FileError> {
        let model = self.model.as_deref().unwrap_or("");
        if model.is_empty() {
            return Err(FileError::new("an oracle entry's model is empty"));
        }
        if !is_sha256_hex(self.prompt_sha256.as_deref().unwrap_or("")) {
            return Err(FileError::new(
                "prompt_sha256 must be 64 lowercase hex digits",
            ));
        }
        if !is_iso_date(self.date.as_deref().unwrap_or("")) {
            return Err(FileError::new("date must be YYYY-MM-DD"));
        }
        Ok(())
    }
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b.iter().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                *c == b'-'
            } else {
                c.is_ascii_digit()
            }
        })
}

/// One name pair and why it is believed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The name as the source spells it.
    pub source: String,
    /// The name as the target spells it.
    pub target: String,
    /// Where the pair came from.
    pub evidence: Evidence,
}

/// A validated mapping file.
///
/// Built only through [`MappingFile::new`] or [`MappingFile::parse`], so every
/// value of this type has unique sources, unique targets, non-empty names and
/// consistent evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappingFile {
    direction: Direction,
    entries: Vec<Entry>,
}

impl MappingFile {
    /// Build a file from entries, refusing any that break the file's rules.
    ///
    /// # Errors
    ///
    /// An empty name, a source or a target used twice, or evidence that does
    /// not fit its kind.
    pub fn new(direction: Direction, entries: Vec<Entry>) -> Result<Self, FileError> {
        let mut sources = HashSet::new();
        let mut targets = HashSet::new();
        for e in &entries {
            check_entry(e, &mut sources, &mut targets)?;
        }
        Ok(Self { direction, entries })
    }

    /// The direction every entry reads in.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// The entries, in file order.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The target recorded for `source`, byte-exact, or `None`.
    #[must_use]
    pub fn target_of(&self, source: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.source == source)
            .map(|e| e.target.as_str())
    }

    /// Serialize to the file's JSON text.
    #[must_use]
    pub fn to_json(&self) -> String {
        let entries: Vec<Value> = self.entries.iter().map(entry_json).collect();
        let doc = json!({
            "format": FORMAT,
            "version": VERSION,
            "direction": direction_str(self.direction),
            "entries": entries,
        });
        serde_json::to_string_pretty(&doc).unwrap_or_default()
    }

    /// Read a mapping file.
    ///
    /// # Errors
    ///
    /// Not JSON, not an object, an unknown `format`, `version` or `direction`,
    /// a malformed entry, or any rule [`MappingFile::new`] enforces.
    pub fn parse(bytes: &[u8]) -> Result<Self, FileError> {
        let root: Value = serde_json::from_slice(bytes)
            .map_err(|e| FileError::new(format!("not valid JSON: {e}")))?;
        let root = root
            .as_object()
            .ok_or_else(|| FileError::new("the top level is not a JSON object"))?;
        check_header(root)?;
        let direction = parse_direction(root)?;
        let list = root
            .get("entries")
            .and_then(Value::as_array)
            .ok_or_else(|| FileError::new("`entries` is missing or not an array"))?;
        let entries = list
            .iter()
            .map(parse_entry)
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(direction, entries)
    }
}

fn check_entry(
    e: &Entry,
    sources: &mut HashSet<String>,
    targets: &mut HashSet<String>,
) -> Result<(), FileError> {
    if e.source.is_empty() || e.target.is_empty() {
        return Err(FileError::new("an entry has an empty name"));
    }
    if !sources.insert(e.source.clone()) {
        return Err(FileError::new(format!(
            "source {:?} appears twice",
            e.source
        )));
    }
    if !targets.insert(e.target.clone()) {
        return Err(FileError::new(format!(
            "target {:?} appears twice",
            e.target
        )));
    }
    e.evidence.validate()
}

fn direction_str(d: Direction) -> &'static str {
    match d {
        Direction::HfToGguf => "hf-to-gguf",
        Direction::GgufToHf => "gguf-to-hf",
    }
}

fn entry_json(e: &Entry) -> Value {
    let mut ev = Map::new();
    ev.insert("kind".into(), json!(e.evidence.kind.as_str()));
    for (key, v) in [
        ("model", &e.evidence.model),
        ("prompt_sha256", &e.evidence.prompt_sha256),
        ("date", &e.evidence.date),
    ] {
        if let Some(v) = v {
            ev.insert(key.into(), json!(v));
        }
    }
    json!({ "source": e.source, "target": e.target, "evidence": ev })
}

fn check_header(root: &Map<String, Value>) -> Result<(), FileError> {
    if root.get("format").and_then(Value::as_str) != Some(FORMAT) {
        return Err(FileError::new(format!("`format` is not {FORMAT:?}")));
    }
    match root.get("version").and_then(Value::as_u64) {
        Some(VERSION) => Ok(()),
        other => Err(FileError::new(format!(
            "unsupported version {other:?}; this reader knows {VERSION}"
        ))),
    }
}

fn parse_direction(root: &Map<String, Value>) -> Result<Direction, FileError> {
    match root.get("direction").and_then(Value::as_str) {
        Some("hf-to-gguf") => Ok(Direction::HfToGguf),
        Some("gguf-to-hf") => Ok(Direction::GgufToHf),
        other => Err(FileError::new(format!("unknown direction {other:?}"))),
    }
}

fn str_field(obj: &Map<String, Value>, key: &str) -> Result<String, FileError> {
    obj.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| FileError::new(format!("`{key}` is missing or not a string")))
}

fn opt_str_field(obj: &Map<String, Value>, key: &str) -> Result<Option<String>, FileError> {
    match obj.get(key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(FileError::new(format!("`{key}` is not a string"))),
    }
}

fn parse_entry(v: &Value) -> Result<Entry, FileError> {
    let obj = v
        .as_object()
        .ok_or_else(|| FileError::new("an entry is not an object"))?;
    let ev = obj
        .get("evidence")
        .and_then(Value::as_object)
        .ok_or_else(|| FileError::new("an entry has no `evidence` object"))?;
    Ok(Entry {
        source: str_field(obj, "source")?,
        target: str_field(obj, "target")?,
        evidence: Evidence {
            kind: EvidenceKind::parse(&str_field(ev, "kind")?)?,
            model: opt_str_field(ev, "model")?,
            prompt_sha256: opt_str_field(ev, "prompt_sha256")?,
            date: opt_str_field(ev, "date")?,
        },
    })
}

#[cfg(test)]
mod tests;
