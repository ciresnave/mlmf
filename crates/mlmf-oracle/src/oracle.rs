// SPDX-License-Identifier: MIT OR Apache-2.0
//! The oracle seam and the check every proposal passes before it can become a
//! mapping-file entry.
//!
//! An [`Oracle`] is a model-backed tool a caller supplies. This crate ships no
//! network client and no model runtime. A proposal is never trusted: [`accept`]
//! checks it against the [`Request`] and drops, with a reason, anything that
//! fails. Names are matched byte-exactly, never normalized (ROADMAP S-60).

use std::collections::{HashMap, HashSet};
use std::fmt;

use mlmf_tensor_names::Direction;

use mlmf_tensor_names::mapping_file::{Entry, Evidence};

/// What an oracle is asked.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    /// Which spelling is the source and which the target.
    pub direction: Direction,
    /// Source names the declared table has no entry for.
    pub unmapped: &'a [String],
    /// The target names a proposal may choose from.
    pub candidates: &'a [String],
}

/// One proposed pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proposal {
    /// The source name.
    pub source: String,
    /// The proposed target name.
    pub target: String,
}

/// Why an oracle could not answer at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleError {
    message: String,
}

impl OracleError {
    /// An error with the given message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OracleError {}

/// A model-backed proposer of name pairs.
pub trait Oracle {
    /// Propose targets for the request's unmapped names. A name the oracle will
    /// not place is simply omitted: it stays absent.
    ///
    /// # Errors
    ///
    /// The oracle could not run (its own failure, not a refusal to place a name).
    fn propose(&self, request: &Request<'_>) -> Result<Vec<Proposal>, OracleError>;
}

/// Why a proposal was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The source was not among the request's unmapped names.
    SourceNotRequested,
    /// The target was not among the request's candidates.
    TargetNotCandidate,
    /// The source was proposed more than once; none of its proposals is kept.
    SourceProposedTwice,
    /// Another proposal claimed the same target; neither is kept.
    TargetClaimedTwice,
}

/// A dropped proposal and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejected {
    /// The proposal as the oracle made it.
    pub proposal: Proposal,
    /// Why it was dropped.
    pub reason: RejectReason,
}

/// The outcome of checking an oracle's answer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Accepted {
    /// Proposals that passed, as entries carrying the supplied evidence.
    pub entries: Vec<Entry>,
    /// Proposals that did not, each with its reason.
    pub rejected: Vec<Rejected>,
}

/// The sets a proposal is checked against.
struct Checks<'a> {
    requested: HashSet<&'a str>,
    candidates: HashSet<&'a str>,
    source_count: HashMap<&'a str, usize>,
    target_count: HashMap<&'a str, usize>,
}

impl Checks<'_> {
    fn reject_reason(&self, p: &Proposal) -> Option<RejectReason> {
        if !self.requested.contains(p.source.as_str()) {
            Some(RejectReason::SourceNotRequested)
        } else if !self.candidates.contains(p.target.as_str()) {
            Some(RejectReason::TargetNotCandidate)
        } else if self.source_count[p.source.as_str()] > 1 {
            Some(RejectReason::SourceProposedTwice)
        } else if self.target_count[p.target.as_str()] > 1 {
            Some(RejectReason::TargetClaimedTwice)
        } else {
            None
        }
    }
}

/// Check `proposals` against `request` and keep those that pass.
///
/// A source proposed twice, or a target claimed by two sources, is not settled
/// by order or by a score: every claimant is dropped.
#[must_use]
pub fn accept(request: &Request<'_>, proposals: Vec<Proposal>, evidence: &Evidence) -> Accepted {
    let checks = Checks {
        requested: request.unmapped.iter().map(String::as_str).collect(),
        candidates: request.candidates.iter().map(String::as_str).collect(),
        source_count: count_by(&proposals, |p| p.source.as_str()),
        target_count: count_by(&proposals, |p| p.target.as_str()),
    };
    let verdicts: Vec<Option<RejectReason>> =
        proposals.iter().map(|p| checks.reject_reason(p)).collect();
    drop(checks);

    let mut out = Accepted::default();
    for (p, verdict) in proposals.into_iter().zip(verdicts) {
        match verdict {
            Some(reason) => out.rejected.push(Rejected {
                proposal: p,
                reason,
            }),
            None => out.entries.push(Entry {
                source: p.source,
                target: p.target,
                evidence: evidence.clone(),
            }),
        }
    }
    out
}

fn count_by<'a>(
    proposals: &'a [Proposal],
    key: impl Fn(&'a Proposal) -> &'a str,
) -> HashMap<&'a str, usize> {
    let mut counts = HashMap::new();
    for p in proposals {
        *counts.entry(key(p)).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests;
