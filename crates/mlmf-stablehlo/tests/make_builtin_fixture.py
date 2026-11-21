# SPDX-License-Identifier: MIT OR Apache-2.0
"""Write fixtures/builtin_dense.mlirbc with MLIR itself (via jaxlib).

    pip install jax==0.12.1 jaxlib
    python make_builtin_fixture.py fixtures/builtin_dense.mlirbc

The module holds inline `dense<...>` constants of several element types (a
splat and a packed i1 among them) and two `dense_resource` constants, so the
reader's handle numbering and blob alignment are exercised. Prints the module
as MLIR parses it back: that text is the ground truth the Rust tests state.
"""
import sys

from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

TEXT = """
module {
  func.func @main() attributes {s = dense<[-1, 5]> : tensor<2xsi16>, w = dense<[1, 2]> : tensor<2xi12>, n = dense<[1, 2, 3]> : tensor<3xi4>} {
    %a = stablehlo.constant dense<[[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]> : tensor<2x3xf32>
    %b = stablehlo.constant dense<[1, -2, 3]> : tensor<3xi8>
    %c = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf64>
    %d = stablehlo.constant dense<[1.0, -2.0, 0.5]> : tensor<3xbf16>
    %e = stablehlo.constant dense<7> : tensor<4xi32>
    %f = stablehlo.constant dense<[true, false, true]> : tensor<3xi1>
    %g = stablehlo.constant dense<[10, 20, 30]> : tensor<3xui16>
    %r1 = stablehlo.constant dense_resource<first> : tensor<2xf32>
    %r2 = stablehlo.constant dense_resource<second> : tensor<3xi64>
    return
  }
}

{-#
  dialect_resources: {
    builtin: {
      first: "0x040000000000803F00000040",
      second: "0x080000000100000000000000020000000000000003000000000000000"
    }
  }
#-}
"""
# `second` must be 3 x i64: fix the literal here so the text above stays readable.
TEXT = TEXT.replace(
    '"0x080000000100000000000000020000000000000003000000000000000"',
    '"0x08000000010000000000000002000000000000000300000000000000"',
)

with mlir.make_ir_context():
    m = ir.Module.parse(TEXT)
    print(m)
    with open(sys.argv[1], "wb") as f:
        m.operation.write_bytecode(f)

# To check the file, parse it in a NEW process (a context that already saw
# `first` renames it `first_1`) and compare with the text printed above:
#     python -c "..parse fixtures/builtin_dense.mlirbc and print.."
