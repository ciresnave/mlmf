# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent reader for MLIR bytecode envelopes (stdlib only).

    python bytecode_recon.py fixtures/jax_export.mlirbc fixtures/mlir_resource.mlirbc > fixtures/expected.tsv

Prints, per file, one `file` row (version, producer, length) and one `section`
row each (id, alignment, data offset, data length). Written from the MLIR
bytecode specification; shares no code with mlmf-stablehlo.
"""
import os
import sys


def varint(b, pos):
    first = b[pos]
    if first == 0:
        return int.from_bytes(b[pos + 1 : pos + 9], "little"), pos + 9
    n = (first & -first).bit_length()
    return int.from_bytes(b[pos : pos + n], "little") >> n, pos + n


for path in sys.argv[1:]:
    b = open(path, "rb").read()
    if b[:4] != b"ML\xefR":
        sys.exit(path + ": not MLIR bytecode")
    ver, pos = varint(b, 4)
    end = b.index(0, pos)
    print("file", os.path.basename(path), ver, b[pos:end].decode(), len(b), sep="\t")
    pos = end + 1
    while pos < len(b):
        ida = b[pos]
        ln, pos = varint(b, pos + 1)
        al = ""
        if ida & 0x80:
            al, pos = varint(b, pos)
            while pos % al:
                if b[pos] != 0xCB:
                    sys.exit(path + ": bad padding byte")
                pos += 1
        print("section", os.path.basename(path), ida & 0x7F, al, pos, ln, sep="\t")
        pos += ln
