"""Verify the GGUF tensor-directory layout against the real corpus.

The layout is asserted everywhere and recorded nowhere in this repo. If it
is wrong, the computed data region will not match the file size — that is
the falsification test, and it is why this reads whole files rather than
trusting a spec page.
"""

import glob
import os
import struct
import sys

ROOT = "C:/Models/gguf-corpus"

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


ok = bad = skipped = 0
for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.gguf"), recursive=True)):
    rel = os.path.relpath(path, ROOT).replace("\\", "/")
    b = open(path, "rb").read()
    if b[:4] != b"GGUF":
        print(f"  SKIP {rel}: not GGUF")
        skipped += 1
        continue
    ver, o = rd(b, 4, "<I")
    if ver not in (2, 3):
        print(f"  SKIP {rel}: version {ver}")
        skipped += 1
        continue
    nt, o = rd(b, o, "<q")
    nkv, o = rd(b, o, "<q")
    align = 32
    kvs = {}
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

    # Falsification: does the last tensor end exactly at the file end?
    worst = 0
    unknown = []
    for name, ne, ty, off in infos:
        if ty not in GEOM:
            unknown.append(ty)
            continue
        blk, bys = GEOM[ty]
        elems = 1
        for d in ne:
            elems *= d
        worst = max(worst, off + elems // blk * bys)
    # A file with NO tensors has no data region and no padding: it ends at
    # the end of the (empty) tensor directory. Padding to alignment exists
    # only when there is data to align.
    end = dir_end if nt == 0 else data_start + worst
    verdict = "OK " if end == len(b) else "BAD"
    if end == len(b):
        ok += 1
    else:
        bad += 1
    print(
        f"  {verdict} {rel}: tensors={nt} align={align} kv_end={kv_end} "
        f"dir_end={dir_end} data_start={data_start} computed_end={end} "
        f"file={len(b)} delta={len(b) - end}"
        + (f" UNKNOWN_TYPES={sorted(set(unknown))}" if unknown else "")
    )

print(f"\nlayout confirmed on {ok} files, contradicted on {bad}, skipped {skipped}")
sys.exit(1 if bad else 0)
