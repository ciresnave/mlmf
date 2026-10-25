# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate golden.onnx with the reference `onnx` package.

    pip install onnx==1.23.2
    python make_golden.py golden.onnx

The model below is mirrored field for field in tests/golden.rs. The bytes are
what the reference serializer writes (`deterministic=True`), so the Rust
writer's output is compared against another implementation, not against
itself. Run 2026-10-25 with onnx 1.23.2 and protobuf 7.36.2; the file also
passes `onnx.checker.check_model`.
"""
import sys

from onnx import TensorProto, checker, helper

F32_BYTES = (1.0, 2.0, 3.0, 4.0)


def f32_raw():
    import struct

    return struct.pack("<4f", *F32_BYTES)


def build():
    add = helper.make_node("Add", ["x", "w"], ["y"], name="add0")
    w = helper.make_tensor("w", TensorProto.FLOAT, [2, 2], f32_raw(), raw=True)
    b = helper.make_tensor(
        "b", TensorProto.INT64, [3], bytes(range(24)), raw=True
    )
    h = helper.make_tensor(
        "h", TensorProto.FLOAT16, [2], bytes([0, 60, 0, 64]), raw=True
    )
    bf = helper.make_tensor(
        "bf", TensorProto.BFLOAT16, [1], bytes([128, 63]), raw=True
    )
    scalar = helper.make_tensor("s", TensorProto.UINT8, [], bytes([7]), raw=True)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 2])
    graph = helper.make_graph(
        [add], "g", [x], [y], initializer=[w, b, h, bf, scalar]
    )
    model = helper.make_model(
        graph,
        producer_name="mlmf-test",
        producer_version="1",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.x", 3)],
    )
    model.ir_version = 8
    model.model_version = 5
    model.doc_string = "doc"
    for k, v in (("k", "v"), ("é", "ü")):
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    return model, add


def main():
    model, add = build()
    checker.check_model(model)
    data = model.SerializeToString(deterministic=True)
    open(sys.argv[1], "wb").write(data)
    print("model bytes", len(data))
    print("node bytes", add.SerializeToString().hex())


main()
