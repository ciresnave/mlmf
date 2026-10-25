# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent readings of the ONNX files the reader tests parse.

    pip install onnx==1.23.2
    python make_read_fixtures.py variants.onnx read-golden.tsv read-conv1d.tsv \
        read-variants.tsv <path to onnx>/backend/test/data/pytorch-converted/test_Conv1d/model.onnx

The reference `onnx` package (protobuf) reads each file; this script prints
one row per fact, in the format `tests/read.rs` reproduces from
`mlmf_onnx::parse_model`. Nothing here shares code with the Rust reader.

`variants.onnx` is built here to carry what the other files lack: typed
(`float_data`, `int64_data`) elements, an external-data tensor, a tensor with
no name and no data, a packed `dims` run (hand-spliced into the file's
bytes: the reference serializer never writes one), a dimension with neither
value nor name, a negative `model_version` (a ten-byte varint) and a
dimension above 2**32.
Run 2026-10-25 with onnx 1.23.2.
"""
import sys

from onnx import ModelProto, TensorProto, helper


def varint(v):
    out = bytearray()
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def packed_tensor():
    """TensorProto { dims packed [2,3], data_type FLOAT, name "p", raw_data }."""
    dims = varint(2) + varint(3)
    raw = bytes(range(24))
    return (
        b"\x0a" + varint(len(dims)) + dims
        + b"\x10\x01"
        + b"\x42\x01p"
        + b"\x4a" + varint(len(raw)) + raw
    )


def variants():
    typed_f = helper.make_tensor("tf", TensorProto.FLOAT, [3], [1.5, 2.5, 3.5])
    typed_i = helper.make_tensor("ti", TensorProto.INT64, [2], [7, -9])
    ext = TensorProto()
    ext.name = "ext"
    ext.data_type = TensorProto.FLOAT
    ext.dims.extend([4, 4])
    ext.data_location = TensorProto.EXTERNAL
    for k, v in (("location", "w.bin"), ("offset", "64"), ("length", "64")):
        e = ext.external_data.add()
        e.key, e.value = k, v
    anon = TensorProto()
    anon.data_type = TensorProto.UINT8
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", None, 4])
    y = helper.make_empty_tensor_value_info("y")
    graph = helper.make_graph([], "v", [x], [y], initializer=[typed_f, typed_i, ext, anon])
    big = TensorProto()
    big.name = "big"
    big.data_type = TensorProto.FLOAT
    big.dims.extend([5_000_000_000])
    graph.initializer.append(big)
    model = helper.make_model(graph, producer_name="mlmf-read-test")
    del model.opset_import[:]
    model.ir_version = 9
    model.model_version = -5
    # The reference serializer writes `dims` unpacked, so splice the packed
    # tensor into the already-serialized graph by hand.
    tensor = packed_tensor()
    graph_bytes = graph.SerializeToString(deterministic=True)
    graph_bytes += b"\x2a" + varint(len(tensor)) + tensor
    model.ClearField("graph")
    head = model.SerializeToString(deterministic=True)
    return head + b"\x3a" + varint(len(graph_bytes)) + graph_bytes


def dim_text(d):
    which = d.WhichOneof("value")
    return "?" if which is None else str(d.dim_value) if which == "dim_value" else d.dim_param


def value_rows(kind, values):
    for v in values:
        t = v.type.tensor_type if v.HasField("type") else None
        elem = str(t.elem_type) if t and t.HasField("elem_type") else "-"
        shape = "-"
        if t and t.HasField("shape"):
            shape = "[" + ",".join(dim_text(d) for d in t.shape.dim) + "]"
        yield "\t".join((kind, v.name if v.HasField("name") else "-", elem, shape))


def data_text(t):
    if t.data_location == TensorProto.EXTERNAL:
        return "external:" + ";".join(f"{e.key}={e.value}" for e in t.external_data)
    if t.HasField("raw_data"):
        return f"raw:{len(t.raw_data)}:{t.raw_data[:8].hex()}"
    for f, _ in t.ListFields():
        if f.name.endswith("_data") and f.name != "external_data":
            return "typed:" + str(f.number)
    return "none"


def rows(path):
    m = ModelProto()
    with open(path, "rb") as fh:
        m.ParseFromString(fh.read())
    opsets = ",".join(
        (o.domain if o.HasField("domain") else "-")
        + ":"
        + (str(o.version) if o.HasField("version") else "-")
        for o in m.opset_import
    ) or "-"
    meta = ",".join(f"{p.key}={p.value}" for p in m.metadata_props) or "-"
    yield "\t".join((
        "model",
        str(m.ir_version) if m.HasField("ir_version") else "-",
        m.producer_name if m.HasField("producer_name") else "-",
        m.producer_version if m.HasField("producer_version") else "-",
        str(m.model_version) if m.HasField("model_version") else "-",
        m.doc_string if m.HasField("doc_string") else "-",
        opsets,
        meta,
    ))
    g = m.graph
    yield "\t".join(("graph", g.name if g.HasField("name") else "-", str(len(g.node))))
    for t in g.initializer:
        yield "\t".join((
            "init",
            t.name if t.HasField("name") else "-",
            str(t.data_type) if t.HasField("data_type") else "-",
            ",".join(str(d) for d in t.dims) or "-",
            data_text(t),
        ))
    yield from value_rows("input", g.input)
    yield from value_rows("output", g.output)


def main():
    variants_out, golden_tsv, conv_tsv, variants_tsv, conv_model = sys.argv[1:6]
    with open(variants_out, "wb") as fh:
        fh.write(variants())
    for src, dst in (("golden.onnx", golden_tsv), (conv_model, conv_tsv), (variants_out, variants_tsv)):
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(rows(src)) + "\n")


main()
