# `mlmf-oracle` — design (S-21)

Status: proposed. Ruled by the PM 2026-10-25: new crate in this workspace, offline only, never a dependency of
a load crate. Roadmap row S-21; parent spec `2026-08-14-backend-agnostic-mlmf-design.md` §10 and OQ-4.

## Purpose and fence

Some checkpoints use tensor names that `mlmf-tensor-names` does not list. Today those are reported
(`translate` returns `None`; `Unmapped` names each one). `mlmf-oracle` is the *offline* tool that proposes
entries for the names the table lacks and writes them to a **mapping file**. It finds where a value is stored;
it never supplies what a value should be (CLAUDE.md §0). A name the oracle cannot place stays absent.

- **Never on a load path.** No crate that reads a model file depends on `mlmf-oracle`. Enforced by
  `tests/direct-deps.allow` and the axis test, the same way Fuel is kept unable to reach it.
- **Names only.** Entries pair tensor names. They carry no shapes, dtypes, permutations or values.

## Mapping file

JSON, UTF-8, one document:

```json
{
  "format": "mlmf-name-mapping",
  "version": 1,
  "direction": "hf-to-gguf",
  "entries": [
    {
      "source": "model.layers.0.self_attn.q_proj.weight",
      "target": "blk.0.attn_q.weight",
      "evidence": {
        "kind": "oracle",
        "model": "<model id>",
        "prompt_sha256": "<hex>",
        "date": "2026-10-25"
      }
    }
  ]
}
```

- `format` and `version` are required. A reader rejects an unknown `format` or a `version` it does not know;
  there is no best-effort read of a newer file. A new field bumps `version`.
- `direction` is `hf-to-gguf` or `gguf-to-hf`, matching `mlmf-tensor-names::Direction`.
- `source` is unique within a file, and so is `target`. A duplicate of either is a read error, not last-wins. A name is never normalized before it is compared.
- `evidence.kind` is `table` (copied from a declared table), `oracle` (proposed by a model) or `human`
  (reviewed and entered by a person). `model`, `prompt_sha256` and `date` are required when `kind` is `oracle`.
- No confidence number is stored. The old `confidence_score -> f32` was a model's self-report; it is not
  evidence and would invite thresholding.

## Non-determinism

An oracle is a language model and may answer differently tomorrow. Each `oracle` entry therefore records the
**model identifier, the SHA-256 of the exact prompt, and the date**. The file is an artifact to be reviewed
and committed, not regenerated on demand: once written, loading it is deterministic. Re-running the oracle
produces a new file to diff, never an in-place update. The prompt text itself is written beside the file
(`<name>.prompts/`), so the hash can be checked.

## Oracle trait

```rust
pub trait Oracle {
    /// Propose targets for `unmapped` source names. A name the oracle will not place is omitted.
    fn propose(&self, request: &Request) -> Result<Vec<Proposal>, OracleError>;
}
```

`Request` holds the unmapped source names, the candidate target names, and the direction. The trait is
synchronous and does no I/O of its own; a caller supplies the implementation (the old `ChatBasedOracle` shape,
v0.7.11 `src/smart_mapping.rs`). `mlmf-oracle` ships no network client and no model runtime.

Two checks run on every proposal before it can enter a file, and a failure drops the entry and reports it:
the target must be in the candidate list, and no target may be proposed for two sources (a second claimant to
a target is an error, not a tie-break).

## How `mlmf-tensor-names` loads it

`mlmf-tensor-names` gains a loader for the **file only** (parse and validate; no oracle code, no new
dependency beyond a JSON parser). A caller that wants extra names builds a `Table` from the declared rows plus
a loaded file and passes it to `translate`. Lookup order is fixed: declared table first; a mapping-file entry
is consulted only when the table has none, and **never overrides a table row** (a conflict is a load error).
A name in neither stays `None` / `Unmapped`. Nothing loads a mapping file implicitly.

## Do not build (patent constraint, PM 2026-10-26)

Sweep Part E, H34 (US11556508B1) claims standardizing names, a model predicting the mapping, and a user's answer updating a dictionary when the model fails. Two rules follow, and they bind the crate and its loader:

- **Names compare byte-exactly.** No case-folding, no separator or underscore stripping, no other normalization of a source or candidate name before it is matched, proposed or looked up.
- **No model-fail-then-ask loop.** Nothing turns "the oracle could not place this name" into a prompt whose answer is stored. An unplaced name stays absent. A `human` entry is a person writing a row, not a response to a failed proposal.

If the claims read closer once the oracle is designed in more detail, stop and go to the PM. ROADMAP S-60.

## Crate checklist (CLAUDE.md conventions)

Workspace member and `default-members`; CI 4 steps; `tests/{axis,direct-deps.allow,allowed-std.list}`;
`.coderipper.toml`; SPDX on every `.rs`; README row; CHANGELOG; ROADMAP row.

## Before any publish

Patent sweep addendum for `mlmf-oracle` (PM ruling), as Part E of `docs/PATENT-SWEEP.md`. Not done here.

## Open points for review

1. JSON versus TOML for the file. Proposed JSON: no new parser style in the workspace.
2. Whether `human` entries need a reviewer field. Proposed no; git history carries it.

## Plan

1. This design PR (docs only).
2. `mlmf-oracle` crate: file type, validation, `Oracle` trait, tests including duplicate-source and
   target-not-in-candidates sabotage cases.
3. Loader in `mlmf-tensor-names`; conflict-with-table test.
4. Patent sweep Part E, then PM publish.
