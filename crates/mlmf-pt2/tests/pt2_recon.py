# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent reader for fixtures/torch_export.pt2 (zipfile + json only).

    python pt2_recon.py fixtures/torch_export.pt2 > expected.tsv

Prints one row per tensor: kind, model, name, path, is_param, dtype code,
sizes, strides, storage_offset, requires_grad, byte length, byte sum. Shares
no code with mlmf-pt2.
"""
import json
import sys
import zipfile

z = zipfile.ZipFile(sys.argv[1])
root = z.namelist()[0].split("/")[0]
print("kind\tmodel\tname\tpath\tis_param\tdtype\tsizes\tstrides\toffset\trequires_grad\tbytes\tbyte_sum")
for kind, d in (("weight", "data/weights/"), ("constant", "data/constants/")):
    for n in sorted(z.namelist()):
        sfx = "_weights_config.json" if kind == "weight" else "_constants_config.json"
        if n.startswith(f"{root}/{d}") and n.endswith(sfx):
            model = n[len(f"{root}/{d}") : -len(sfx)]
            for name, c in json.loads(z.read(n))["config"].items():
                m = c["tensor_meta"]
                path = f"{root}/{d}{c['path_name']}"
                data = z.read(path)
                print("\t".join(map(str, [
                    kind, model, name, path, str(c["is_param"]).lower(), m["dtype"],
                    ",".join(str(s["as_int"]) for s in m["sizes"]),
                    ",".join(str(s["as_int"]) for s in m["strides"]),
                    m["storage_offset"]["as_int"], str(m["requires_grad"]).lower(),
                    len(data), sum(data),
                ])))
