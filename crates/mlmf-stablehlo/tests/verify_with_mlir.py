# SPDX-License-Identifier: MIT OR Apache-2.0
"""Parse MLIR bytecode files with MLIR itself (via jaxlib) and compare.

    pip install jax jaxlib          # 0.12.0.1 when this was written
    python verify_with_mlir.py reference.mlirbc candidate.mlirbc

Parses both with the real MLIR bytecode reader, prints each module's text,
exits non-zero if the candidate does not parse or prints differently from the
reference. Also checks that a damaged copy of the candidate is REFUSED, so a
pass means the reader actually checked something.
"""
import sys

from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir


def text(path):
    with mlir.make_ir_context():
        return str(ir.Module.parse(open(path, "rb").read()))


a, b = text(sys.argv[1]), text(sys.argv[2])
print(b)
damaged = open(sys.argv[2], "rb").read()[:-5]
try:
    with mlir.make_ir_context():
        ir.Module.parse(damaged)
    print("DAMAGED COPY WAS ACCEPTED: the check proves nothing")
    sys.exit(2)
except Exception:
    print("damaged copy refused: ok")
sys.exit(0 if a == b else 1)
