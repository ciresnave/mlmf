"""Verify the GGUF tensor-directory layout against the real corpus, and emit
the tensor fixture that `crates/mlmf-gguf/tests/corpus.rs` replays.

The layout is asserted everywhere and recorded nowhere in this repo. If it
is wrong, the computed data region will not match the file size — that is
the falsification test, and it is why this reads whole files rather than
trusting a spec page.

**This reader must stay independent of `mlmf-gguf`.** It shares no code,
no type table and no arithmetic with the crate whose expectations it
produces, because an error shared between a parser and its own expectations
cannot be caught by comparing them. Nothing here may import, shell out to,
or copy from that crate.

Usage:

    python docs/superpowers/plans/tensor_recon.py
        Verify the layout. One line per file on stdout, exit 1 on any
        contradiction.

    python docs/superpowers/plans/tensor_recon.py --tsv > \\
        crates/mlmf-gguf/tests/corpus-tensors.tsv
        Same verification, on stderr this time, and the fixture on stdout.
        Refuses to emit anything if the layout was contradicted, so a
        fixture can never record a measurement this script disbelieves.
"""

import glob
import os
import struct
import sys

ROOT = "C:/Models/gguf-corpus"

# The sentinel the fixture uses for the four per-tensor columns of a file
# that declares no tensors. Nineteen of the twenty-eight readable corpus
# files are that shape, so this is the common case rather than an edge one.
# `emit` refuses to write a tensor name equal to it, which is what keeps it
# unambiguous.
NONE = "-"

# ggml block geometry: code -> (block elements, bytes per block)
# Dense types are (1, width).
GEOM = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 82),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 66),  # TQ1_0
    35: (256, 54),  # TQ2_0
    39: (32, 17),  # MXFP4
}


def rd(b, o, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack_from(fmt, b, o)[0], o + n


def rdstr(b, o):
    ln, o = rd(b, o, "<Q")
    return b[o : o + ln], o + ln


def skip_value(b, o, t):
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in fixed:
        return o + fixed[t]
    if t == 8:
        ln, o = rd(b, o, "<Q")
        return o + ln
    if t == 9:
        et, o = rd(b, o, "<I")
        n, o = rd(b, o, "<Q")
        if et in fixed:
            return o + fixed[et] * n
        for _ in range(n):
            o = skip_value(b, o, et)
        return o
    raise ValueError(f"unknown value type {t}")


def nbytes(ne, ty):
    """Bytes one tensor occupies, from block geometry alone.

    `None` for a type this table does not carry — the caller decides what
    an unknown type means, because the layout check and the fixture want
    different things from it.
    """
    if ty not in GEOM:
        return None
    blk, bys = GEOM[ty]
    elems = 1
    for d in ne:
        elems *= d
    return elems // blk * bys


def scan(path, rel):
    """Read one file's header, KV block and tensor directory.

    `None` when the file is not something this build measures — a bad magic
    or a version other than 2 or 3 — with the reason already printed.
    """
    b = open(path, "rb").read()
    if b[:4] != b"GGUF":
        return None, f"  SKIP {rel}: not GGUF"
    ver, o = rd(b, 4, "<I")
    if ver not in (2, 3):
        return None, f"  SKIP {rel}: version {ver}"
    nt, o = rd(b, o, "<q")
    nkv, o = rd(b, o, "<q")
    align = 32
    for _ in range(nkv):
        k, o = rdstr(b, o)
        t, o = rd(b, o, "<I")
        if k == b"general.alignment" and t == 4:
            v, _ = rd(b, o, "<I")
            align = v
        o = skip_value(b, o, t)
    kv_end = o

    # --- the layout under test ---
    infos = []
    for _ in range(nt):
        name, o = rdstr(b, o)
        ndim, o = rd(b, o, "<I")
        ne = []
        for _ in range(ndim):
            d, o = rd(b, o, "<Q")
            ne.append(d)
        ty, o = rd(b, o, "<I")
        off, o = rd(b, o, "<Q")
        infos.append((name, ne, ty, off))
    dir_end = o
    data_start = (dir_end + align - 1) // align * align

    return {
        "rel": rel,
        "version": ver,
        "n_tensors": nt,
        "align": align,
        "kv_end": kv_end,
        "dir_end": dir_end,
        "data_start": data_start,
        "infos": infos,
        "size": len(b),
    }, None


def verify(f):
    """Does the computed data region end exactly where the file does?

    Returns `(ok, line)`. A file with NO tensors has no data region and no
    padding: it ends at the end of the (empty) tensor directory. Padding to
    alignment exists only when there is data to align.
    """
    worst = 0
    unknown = []
    for name, ne, ty, off in f["infos"]:
        n = nbytes(ne, ty)
        if n is None:
            unknown.append(ty)
            continue
        worst = max(worst, off + n)
    end = f["dir_end"] if f["n_tensors"] == 0 else f["data_start"] + worst
    ok = end == f["size"]
    line = (
        f"  {'OK ' if ok else 'BAD'} {f['rel']}: tensors={f['n_tensors']} "
        f"align={f['align']} kv_end={f['kv_end']} dir_end={f['dir_end']} "
        f"data_start={f['data_start']} computed_end={end} file={f['size']} "
        f"delta={f['size'] - end}"
        + (f" UNKNOWN_TYPES={sorted(set(unknown))}" if unknown else "")
    )
    return ok, line


def row(f):
    """One fixture row, or a fatal complaint about why there cannot be one.

    Returns `(fields, None)` or `(None, complaint)`. Every complaint here is
    a case the fixture's SCHEMA cannot represent, not a case the crate
    cannot parse — a duplicate tensor name really is legal GGUF and
    `parse_tensors` really does keep the first, but then `n_tensors` stops
    being the length of the parsed list and the single count column would
    quietly mean two different things. The script refuses rather than
    guesses, and a human extends the schema.
    """
    rel, nt, infos = f["rel"], f["n_tensors"], f["infos"]

    if nt != len(infos):
        return None, f"{rel}: declared {nt} tensors and read {len(infos)}"
    if nt == 0:
        return [rel, "0", str(f["dir_end"]), str(f["data_start"])] + [NONE] * 4, None

    names = [n for n, _, _, _ in infos]
    if len(set(names)) != len(names):
        return None, (
            f"{rel}: declares the same tensor name twice, so `n_tensors` is "
            f"no longer the length of the parsed list -- extend the schema"
        )
    for n in names:
        if b"\t" in n or b"\n" in n:
            return None, f"{rel}: tensor name {n!r} contains a TSV separator"
    first_name, _, first_code, first_off = infos[0]
    try:
        first_name = first_name.decode("utf-8")
    except UnicodeDecodeError as e:
        return None, f"{rel}: first tensor name is not UTF-8: {e}"
    if first_name == NONE:
        return None, f"{rel}: first tensor is literally named {NONE!r}"

    ends = []
    for name, ne, ty, off in infos:
        n = nbytes(ne, ty)
        if n is None:
            return None, f"{rel}: type code {ty} is not in this script's GEOM"
        ends.append(off + n)
    last_end = f["data_start"] + ends[-1]

    # The claim the `last_end` column is worth carrying AT ALL: the last
    # tensor in DECLARATION order is also the furthest one, so its computed
    # end is the file's length. That makes the column a differential on the
    # crate's block geometry and on its rebase at once -- if `mlmf-ggml`
    # sized that tensor differently, or `parse_tensors` rebased it against
    # the wrong origin, the number moves. A corpus file that broke the claim
    # would make the column mean something much weaker without saying so, so
    # it is checked here rather than assumed in a comment.
    if last_end != max(ends) + f["data_start"]:
        return None, (
            f"{rel}: the last tensor declared is not the furthest one "
            f"({last_end} vs {max(ends) + f['data_start']})"
        )
    if last_end != f["size"]:
        return None, (
            f"{rel}: the last tensor ends at {last_end} and the file is "
            f"{f['size']} bytes"
        )

    return [
        rel,
        str(nt),
        str(f["dir_end"]),
        str(f["data_start"]),
        first_name,
        str(first_code),
        str(first_off),
        str(last_end),
    ], None


HEADER = """\
# Tensor-directory facts measured from real GGUF files.
#
# Method: `docs/superpowers/plans/tensor_recon.py --tsv`, an independent
# Python reader that shares no code, no type table and no arithmetic with
# mlmf-gguf. That independence is the whole point: an error shared between
# a parser and its own expectations cannot be caught by comparing them, so
# this file must never be regenerated from the crate it checks.
#
# Corpus: C:\\Models\\gguf-corpus. The `file` column is the path RELATIVE to
# that root, because the corpus is not flat: a bare basename cannot be
# reopened.
#
# 29 .gguf files were scanned and 28 rows are here, matching
# corpus-metadata.tsv row for row. The missing one is
# legacy/tinyllamas-stories-260k-f32.gguf, which is GGUF v1 and is refused
# by version rather than misparsed, so there is nothing to measure.
#
# Columns are TAB separated.
#
#   file          path relative to the corpus root
#   n_tensors     tensor-info records the header declares
#   dir_end       byte offset one past the last tensor-info record
#   data_start    dir_end rounded UP to the file's alignment. For a file
#                 with NO tensors this is past the end of the file: the
#                 writer emits no padding when there is nothing to pad for,
#                 and 19 of these 28 rows are that shape. Validating it
#                 against the file length would refuse every vocab-only
#                 GGUF, which is exactly what a metadata consumer opens.
#   first_name    name of the first tensor-info record
#   first_code    its raw ggml type code, as the file spells it
#   first_offset  its declared offset, RELATIVE to data_start
#   last_end      absolute end of the LAST tensor: data_start + its declared
#                 offset + its size from block geometry. Verified equal to
#                 the file's length on every row, which is what makes it a
#                 differential on geometry and rebase together.
#
# The last four columns are `-` for a file that declares no tensors.
"""

COLUMNS = [
    "file",
    "n_tensors",
    "dir_end",
    "data_start",
    "first_name",
    "first_code",
    "first_offset",
    "last_end",
]


def main(argv):
    want_tsv = "--tsv" in argv[1:]
    # The fixture goes to stdout so it can be redirected; when it does, the
    # human-readable verification must not be interleaved into it.
    log = sys.stderr if want_tsv else sys.stdout

    ok = bad = skipped = 0
    rows = []
    complaints = []
    for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.gguf"), recursive=True)):
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        f, why = scan(path, rel)
        if f is None:
            print(why, file=log)
            skipped += 1
            continue
        good, line = verify(f)
        print(line, file=log)
        if good:
            ok += 1
        else:
            bad += 1
        r, complaint = row(f)
        if complaint is not None:
            complaints.append(complaint)
        else:
            rows.append(r)

    print(
        f"\nlayout confirmed on {ok} files, contradicted on {bad}, "
        f"skipped {skipped}",
        file=log,
    )

    if not want_tsv:
        return 1 if bad else 0

    # A fixture is a set of expectations another test will trust. Emitting
    # one from a run that contradicted itself would launder a known-bad
    # measurement into an assertion, so nothing is written at all.
    if bad or complaints:
        for c in complaints:
            print(f"REFUSED: {c}", file=log)
        print("\nno fixture written", file=log)
        return 1

    sys.stdout.write(HEADER)
    sys.stdout.write("\t".join(COLUMNS) + "\n")
    for r in rows:
        sys.stdout.write("\t".join(r) + "\n")
    print(f"\n{len(rows)} rows written", file=log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
