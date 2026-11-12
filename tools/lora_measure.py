# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent measurement of LoRA adapter files (ROADMAP S-65, step 1).

Shares no code with any mlmf crate, so an error shared between a parser and its
expectations cannot hide. Reads PEFT `adapter_model.safetensors` headers and GGUF
adapter files and prints one TSV row per tensor, then the GGUF `general.*` /
`adapter.*` keys. Usage: python -I tools/lora_measure.py <file>...
"""
import json
import struct
import sys

GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                10: "<Q", 11: "<q", 12: "<d"}
GGML_TYPES = {0: "F32", 1: "F16", 8: "Q8_0", 30: "BF16"}


def safetensors_rows(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return [(k, v["dtype"], v["shape"]) for k, v in sorted(header.items())]


def read_value(f, t):
    if t in GGUF_SCALARS:
        fmt = GGUF_SCALARS[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8")
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, et) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {t}")


def gguf(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        kv = {}
        for _ in range(n_kv):
            (kl,) = struct.unpack("<Q", f.read(8))
            key = f.read(kl).decode("utf-8")
            (t,) = struct.unpack("<I", f.read(4))
            kv[key] = read_value(f, t)
        rows = []
        for _ in range(n_tensors):
            (nl,) = struct.unpack("<Q", f.read(8))
            name = f.read(nl).decode("utf-8")
            (nd,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            (ty,) = struct.unpack("<I", f.read(4))
            f.read(8)
            rows.append((name, GGML_TYPES.get(ty, f"type{ty}"), dims))
    return version, kv, rows


def main(argv):
    for path in argv[1:]:
        print(f"## {path}")
        if path.endswith(".gguf"):
            version, kv, rows = gguf(path)
            print(f"# gguf v{version}, {len(rows)} tensors")
            for k, v in kv.items():
                if k.startswith(("general.", "adapter.")):
                    print(f"kv\t{k}\t{v!r}")
        else:
            rows = safetensors_rows(path)
            print(f"# safetensors, {len(rows)} tensors")
        for name, dt, shape in rows:
            print(f"tensor\t{name}\t{dt}\t{shape}")
    return 0


sys.exit(main(sys.argv))
