# SPDX-License-Identifier: MIT OR Apache-2.0
"""Load .pt2 archives with torch.export.load and compare their outputs.

    python verify_with_torch.py reference.pt2 candidate.pt2

Runs both on the same input and prints the outputs; exits non-zero if the
candidate fails to load or differs. Used on the archive mlmf-pt2 writes
(fixtures/written_by_mlmf.pt2), against the one torch wrote.
"""
import sys

import torch

x = torch.ones(2, 4)
a = torch.export.load(sys.argv[1]).module()(x)
b = torch.export.load(sys.argv[2]).module()(x)
print("reference", a.tolist())
print("candidate", b.tolist())
sys.exit(0 if torch.equal(a, b) else 1)
