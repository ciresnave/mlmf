"""Verify the safetensors layout, and measure what it asks of mlmf-core's seam.

Two questions, and the second is the one plan 5 exists for.

1. Is the layout what everyone says it is? Falsification test: the furthest
   `data_offsets[1]`, plus the 8-byte length prefix and the JSON header,
   must equal the file length exactly. If it does not, the layout is wrong.

2. Does `mlmf-core`'s seam fit a format it was not shaped by? `MetaValue`
   has thirteen variants and GGUF has thirteen value types, which is not a
   coincidence. This reports what safetensors metadata actually contains,
   and which `DType`s a real file needs, so the answer comes from files
   rather than from reading the trait.

Shares no code with mlmf-core, mlmf-ggml or mlmf-gguf, deliberately.

Run with `--tsv` to emit the fixture
`crates/mlmf-safetensors/tests/corpus-safetensors.tsv` on stdout instead of
the human report. That fixture is the expectation half of a differential, so
it must be produced by a reader that shares nothing with the crate under
test: an error present in BOTH the parser and its own expectations cannot be
caught by comparing them, which is the one failure authored fixtures cannot
see either.
"""

import glob
import json
import os
import struct
import sys

ROOTS = ["C:/Models"]

# Every dtype string safetensors defines, and the width each implies.
# Widths are asserted here so a file can contradict them.
WIDTH = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

files = []
for r in ROOTS:
    files += glob.glob(os.path.join(r, "**", "*.safetensors"), recursive=True)
files.sort()

TSV = "--tsv" in sys.argv

ok = bad = 0
dtypes_seen = {}
meta_kinds = {}
name_notes = []
rows = []

for path in files:
    rel = os.path.relpath(path, ROOTS[0]).replace("\\", "/")
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen).decode("utf-8"))

    data_start = 8 + hlen
    furthest = 0
    n = 0
    per_file_dtypes = {}
    meta_pairs = []
    first = None
    for name, v in header.items():
        if name == "__metadata__":
            for k, val in v.items():
                meta_kinds[type(val).__name__] = meta_kinds.get(type(val).__name__, 0) + 1
                # Recorded as `key=value` only when the value is a string,
                # which is all safetensors defines. A non-string would be
                # emitted as its Python type name so the fixture cannot
                # quietly launder it into a string.
                meta_pairs.append(
                    f"{k}={val}" if isinstance(val, str) else f"{k}=<{type(val).__name__}>"
                )
            continue
        n += 1
        dt = v["dtype"]
        dtypes_seen[dt] = dtypes_seen.get(dt, 0) + 1
        per_file_dtypes[dt] = per_file_dtypes.get(dt, 0) + 1
        if first is None:
            # DECLARATION order, which is the order the JSON object lists
            # and is NOT the order `mlmf-safetensors` yields -- serde_json
            # backs an object with a BTreeMap, so the crate returns
            # lexicographic order. The differential therefore looks this
            # tensor up BY NAME rather than by position, which sidesteps the
            # ordering divergence instead of encoding one side of it.
            lo0, hi0 = v["data_offsets"]
            # Absolute too, REBASED HERE rather than in the test. A test
            # body computing `data_start + lo` would be evaluating the
            # implementation's own expression a second time and would agree
            # with it however wrong both were.
            first = (name, dt, lo0, hi0, data_start + lo0, data_start + hi0)
        lo, hi = v["data_offsets"]
        furthest = max(furthest, hi)

        # Does the declared range match shape x width? A mismatch would mean
        # the seam cannot compute an extent from (shape, encoding) alone.
        elems = 1
        for d in v["shape"]:
            elems *= d
        want = elems * WIDTH[dt]
        if hi - lo != want:
            name_notes.append(f"{rel}:{name} range {hi - lo} != shape*width {want}")
        if "\t" in name or "\n" in name:
            name_notes.append(f"{rel}:{name!r} contains a tab or newline")

    end = data_start + furthest
    good = end == size
    ok, bad = (ok + 1, bad) if good else (ok, bad + 1)

    if first is None:
        name_notes.append(f"{rel}: declares no tensors; not emitted to the fixture")
    else:
        for field in (rel, first[0]):
            assert "\t" not in field and "\n" not in field, f"field breaks the TSV: {field!r}"
        rows.append(
            "\t".join(
                [
                    rel,
                    str(size),
                    str(hlen),
                    str(data_start),
                    str(n),
                    first[0],
                    first[1],
                    str(first[2]),
                    str(first[3]),
                    str(first[4]),
                    str(first[5]),
                    str(end),
                    ",".join(f"{k}={v}" for k, v in sorted(per_file_dtypes.items())),
                    ",".join(sorted(meta_pairs)) or "-",
                ]
            )
        )

    if TSV:
        continue
    print(
        f"  {'OK ' if good else 'BAD'} {rel}: tensors={n} hlen={hlen} "
        f"data_start={data_start} computed_end={end} file={size} delta={size - end}"
    )

if TSV:
    print("# Measured by docs/superpowers/plans/safetensors_recon.py --tsv.")
    print("# Regenerate deliberately; do not hand-edit.")
    print(
        "\t".join(
            [
                "file",
                "size",
                "header_len",
                "data_start",
                "n_tensors",
                "first_name",
                "first_dtype",
                "first_lo",
                "first_hi",
                "first_abs_lo",
                "first_abs_hi",
                "furthest_end",
                "dtypes",
                "metadata",
            ]
        )
    )
    for row in rows:
        print(row)
    sys.exit(1 if bad else 0)

print(f"\nlayout confirmed on {ok} files, contradicted on {bad}")
print(f"dtypes in use: {dict(sorted(dtypes_seen.items()))}")
print(f"__metadata__ value types: {meta_kinds or 'NONE — no __metadata__ key at all'}")
for note in name_notes:
    print(f"  NOTE {note}")
sys.exit(1 if bad else 0)
