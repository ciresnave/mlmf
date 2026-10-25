# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate fixtures/torch_export.pt2 with PyTorch 2.14.1.

    pip install torch==2.14.1      # CPU build
    python make_fixture.py fixtures/torch_export.pt2

A tiny module with one Linear (two parameters), three buffers of different
dtypes (f32, i64, f16) and one lifted tensor constant, exported with
torch.export and saved with torch.export.save. Values are invented. Run
2026-10-25; `verify_with_torch.py` loads archives with torch.export.load.
"""
import sys

import torch


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)
        self.register_buffer("scale", torch.tensor([0.5, 2.0, 1.5]))
        self.register_buffer("idx", torch.tensor([2, 0, 1], dtype=torch.int64))
        self.register_buffer("h16", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))
        self.k = torch.tensor([1.0, 1.0, 1.0])  # plain attribute -> lifted constant
        with torch.no_grad():
            self.lin.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)
            self.lin.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))

    def forward(self, x):
        y = self.lin(x) * self.scale + self.h16.to(torch.float32) * self.k
        return y[:, self.idx]


ep = torch.export.export(M(), (torch.ones(2, 4),))
torch.export.save(ep, sys.argv[1])
