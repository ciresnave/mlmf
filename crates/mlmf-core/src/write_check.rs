//! CD-1/CD-2/CD-3 — what a target format requires, and what MLMF may supply.
//!
//! Spec §6 sorts every value a writer needs into three states: **Declared**
//! (the source says it), **Absent** (it does not), and **Supplied** (MLMF
//! filled it at write time, *attributed*). This module is the mechanism that
//! performs that sort and refuses when the third state is unavailable.
//!
//! # The fence
//!
//! > MLMF may supply a *format's* documented default. MLMF may never supply
//! > a *model's* value.
//!
//! [`Requirement`] therefore carries **two** citations, not one: why the key
//! is required, and — separately — where its default value comes from. They
//! are different claims, and a single specification sentence does not always
//! support both. CD-1 says the citation lives beside the value; this is that.
//!
//! # What lives here, and what deliberately does not
//!
//! ⚠️ **The mechanism is here; the TABLE is not.** Which keys a format
//! requires is knowledge *about that format*, and C4 forbids `mlmf-core`
//! from depending on a format crate. Each format crate owns its own
//! `&[Requirement]` and passes it in.
//!
//! That split also disposes of **conditional** requirements without teaching
//! core a single format condition. GGUF requires
//! `general.quantization_version` only *"if any tensors are quantized"* — so
//! `mlmf-gguf` puts that row in the slice only when it applies, and core
//! never asks why a row is present. A `Condition` enum in core would have
//! had to enumerate every format's conditions, which is the interpretation
//! this project does not do.

use crate::error::{Error, ErrorKind};
use crate::meta::MetaValue;
use crate::report::Declaration;
use crate::traits::MetadataSource;

/// A format's documented default, and the citation that licenses it.
///
/// CD-1: *"A default may only be supplied if it is citable to a
/// specification or a named reference implementation, and the citation lives
/// in the table beside the value. If you cannot cite it, you cannot supply
/// it."*
///
/// The citation is `&'static str` rather than a free-form `String` because
/// it must come from the table, which is a compile-time constant. A citation
/// assembled at run time is one that could be assembled *from the value it
/// is supposed to justify*.
#[derive(Debug, Clone, PartialEq)]
pub struct CitedDefault {
    /// The value MLMF supplies when the key is absent.
    pub value: MetaValue,
    /// Where the value comes from — quoted, not paraphrased.
    pub citation: &'static str,
}

/// One key a target format requires.
///
/// Construct these in the format crate that owns the format; see
/// [the module docs](self) for why they do not live here.
#[derive(Debug, Clone, PartialEq)]
pub struct Requirement {
    /// Canonical key name, exactly as the format spells it.
    pub key: &'static str,
    /// Why this key is required — quoted from the format's specification.
    ///
    /// ⚠️ **Separate from the default's citation.** For GGUF's
    /// `general.alignment`, one sentence establishes that it is required and
    /// a *different* one establishes that 32 is its default; a single field
    /// would have had to drop one of them.
    pub required_because: &'static str,
    /// The citable default, or `None` when the format documents none.
    ///
    /// `None` is what makes a key capable of triggering CD-3.
    pub default: Option<CitedDefault>,
}

/// What happened to one [`Requirement`] against one source.
///
/// Deliberately **not** `#[non_exhaustive]`: a fifth state would be a
/// genuine change to the §6 model, and a caller that stops matching
/// exhaustively should fail to compile rather than fall through a wildcard
/// into the wrong branch.
#[derive(Debug, Clone, PartialEq)]
pub enum Resolution {
    /// The source declares it. Nothing is supplied and nothing is refused.
    Declared,
    /// Absent, and the format documents a default — CD-1 supplies it.
    Supplied {
        /// The supplied value.
        value: MetaValue,
        /// Its citation, carried through so CD-2 can report it.
        citation: &'static str,
    },
    /// Absent, with no citable default. **CD-3 refuses.**
    Missing,
    /// Declared, but the declaration could not be read.
    ///
    /// ⚠️ **Not the same as [`Missing`](Self::Missing), and the difference
    /// is load-bearing.** The source *does* declare this key, so supplying a
    /// default over it would replace a value the file actually contains —
    /// exactly the "MLMF may never supply a model's value" violation the §6
    /// fence forbids. It refuses instead.
    ///
    /// Reachable only through a [`MetadataSource`] that overrides
    /// [`declaration`](MetadataSource::declaration); the trait's default
    /// implementation never reports it.
    Unreadable,
}

/// The result of checking a source against a target format's requirements.
///
/// Carries **both** halves of §6: the CD-2 report of what was supplied, and
/// the CD-3 verdict on what is missing.
#[derive(Debug, Clone, PartialEq)]
pub struct WriteCheck {
    rows: Vec<(String, Resolution)>,
}

impl WriteCheck {
    /// Resolve every requirement against `source`.
    ///
    /// ⚠️ **Reads [`MetadataSource::declaration`], never `get`.** `get`
    /// returns an `Option`, which collapses *absent* and *declared-but-
    /// unreadable* into one `None` — and those two take opposite actions
    /// here: the first may receive a default, the second must never. The
    /// three-state accessor exists precisely so this decision can be made
    /// correctly, and reaching for the ergonomic one would have been the bug.
    #[must_use]
    pub fn run<S: MetadataSource + ?Sized>(source: &S, required: &[Requirement]) -> Self {
        let rows = required
            .iter()
            .map(|req| {
                let resolution = match source.declaration(req.key) {
                    Declaration::Declared(_) => Resolution::Declared,
                    Declaration::Unreadable(_) => Resolution::Unreadable,
                    Declaration::Absent => match &req.default {
                        Some(d) => Resolution::Supplied {
                            value: d.value.clone(),
                            citation: d.citation,
                        },
                        None => Resolution::Missing,
                    },
                    // ⚠️ NO WILDCARD ARM, DELIBERATELY. `Declaration` is
                    // `#[non_exhaustive]`, but that attribute does nothing
                    // INSIDE the crate that defines it — and this module is
                    // in that crate. A `_` arm here compiles to dead code
                    // (measured: `unreachable_patterns`) and would silently
                    // absorb a state added later, choosing a behaviour for
                    // it that nobody decided.
                    //
                    // Matching exhaustively makes that future addition a
                    // BUILD FAILURE at this line, which is the loudest place
                    // it can surface. A consumer outside `mlmf-core` still
                    // needs the wildcard the attribute forces on them; we do
                    // not, and taking it anyway would trade a compile error
                    // for a wrong default in a written file.
                };
                (req.key.to_string(), resolution)
            })
            .collect();
        Self { rows }
    }

    /// Every requirement and how it resolved, in the table's own order.
    #[must_use]
    pub fn rows(&self) -> &[(String, Resolution)] {
        &self.rows
    }

    /// **CD-2**: the values MLMF supplied, each with its citation.
    ///
    /// *"Every supplied value appears in the conversion report and in the
    /// output's provenance."* This is the source of that report — a caller
    /// that writes a file without draining this is the CD-2 violation.
    #[must_use]
    pub fn supplied(&self) -> Vec<(&str, &MetaValue, &'static str)> {
        self.rows
            .iter()
            .filter_map(|(key, r)| match r {
                Resolution::Supplied { value, citation } => Some((key.as_str(), value, *citation)),
                _ => None,
            })
            .collect()
    }

    /// Required keys that are absent with no citable default. **CD-3.**
    #[must_use]
    pub fn missing(&self) -> Vec<&str> {
        self.keys_where(|r| matches!(r, Resolution::Missing))
    }

    /// Required keys whose declaration could not be read.
    ///
    /// Refuses like [`missing`](Self::missing), but for the opposite reason:
    /// the value is present and unusable, rather than not present at all.
    #[must_use]
    pub fn unreadable(&self) -> Vec<&str> {
        self.keys_where(|r| matches!(r, Resolution::Unreadable))
    }

    fn keys_where(&self, pred: impl Fn(&Resolution) -> bool) -> Vec<&str> {
        self.rows
            .iter()
            .filter(|(_, r)| pred(r))
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Whether CD-3 refuses this conversion.
    ///
    /// True when **either** a required key is missing **or** one is declared
    /// unreadably. Both mean the writer has no usable value that it is
    /// permitted to invent.
    #[must_use]
    pub fn refused(&self) -> bool {
        self.rows
            .iter()
            .any(|(_, r)| matches!(r, Resolution::Missing | Resolution::Unreadable))
    }

    /// One [`ErrorKind::MissingRequired`] per missing key, naming it.
    ///
    /// CD-3 says *"refuse the conversion and name the missing key"*; a
    /// caller that fixes one key, retries, and is refused again learns the
    /// set one round-trip at a time, so every missing key is named at once.
    ///
    /// ⚠️ **This can be EMPTY while [`refused`](Self::refused) is true.**
    /// `ErrorKind::MissingRequired` reads *"is not declared and has no
    /// citable default"*, which is false of an [`Unreadable`] row — that key
    /// *is* declared. Rather than attach a message that misdescribes the
    /// file, the unreadable set is exposed by
    /// [`unreadable`](Self::unreadable) and core's error vocabulary is left
    /// alone. **Branch on `refused()`, never on `errors().is_empty()`.**
    ///
    /// [`Unreadable`]: Resolution::Unreadable
    #[must_use]
    pub fn errors(&self) -> Vec<Error> {
        self.missing()
            .into_iter()
            .map(|key| {
                Error::from(ErrorKind::MissingRequired {
                    key: key.to_string(),
                })
            })
            .collect()
    }
}
