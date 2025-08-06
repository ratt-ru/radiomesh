import importlib.resources
import re
from argparse import Namespace
from contextlib import ExitStack
from typing import List, Literal, Tuple

import sympy
from sympy.physics.quantum import TensorProduct

import radiomesh

OUTPUT_FILENAME = "_stokes_expr.py"

PREAMBLE = """
from numpy import conjugate as conj

"""

POLARISATION_TYPES = ["linear", "circular"]

WEIGHT_FN_ARGUMENTS = [
  "gp00",
  "gp01",
  "gp10",
  "gp11",
  "gq00",
  "gq01",
  "gq10",
  "gq11",
  "w0",
  "w1",
  "w2",
  "w3",
]

VIS_FN_ARGUMENTS = WEIGHT_FN_ARGUMENTS + ["v00", "v01", "v10", "v11"]


def sympy_expressions(
  polarisations: str | Literal["linear", "circular"],
) -> Tuple[List[str], sympy.Expr, sympy.Expr]:
  """
  Returns
  -------
    A tuple of the form (schema, coherencies, weights) where
      1. schema is a list of stokes values
      2. coherencies is a sympy expression for generating stokes values
        from visibilities, weights and gains
      3. weights is a sympy expression for generating weights
        from gains and weights.
  """
  # set up symbolic expressions
  gp00, gp10, gp01, gp11 = sympy.symbols("gp00 gp10 gp01 gp11", real=False)
  gq00, gq10, gq01, gq11 = sympy.symbols("gq00 gq10 gq01 gq11", real=False)
  w0, w1, w2, w3 = sympy.symbols("w0 w1 w2 w3", real=True)
  v00, v10, v01, v11 = sympy.symbols("v00 v10 v01 v11", real=False)

  # Jones matrices
  Gp = sympy.Matrix([[gp00, gp01], [gp10, gp11]])
  Gq = sympy.Matrix([[gq00, gq01], [gq10, gq11]])

  # Mueller matrix (row major form)
  Mpq = TensorProduct(Gp, Gq.conjugate())
  Mpqinv = TensorProduct(Gp.inv(), Gq.conjugate().inv())

  # inverse noise covariance
  Sinv = sympy.Matrix([[w0, 0, 0, 0], [0, w1, 0, 0], [0, 0, w2, 0], [0, 0, 0, w3]])
  S = Sinv.inv()

  # visibilities
  Vpq = sympy.Matrix([[v00], [v01], [v10], [v11]])

  # Full Stokes to corr operator
  # Is this the only difference between linear and circular pol?
  # What about paralactic angle rotation?

  if polarisations == "linear":
    T = sympy.Matrix(
      [[1.0, 1.0, 0, 0], [0, 0, 1.0, 1.0j], [0, 0, 1.0, -1.0j], [1.0, -1.0, 0, 0]]
    )
  elif polarisations == "circular":
    T = sympy.Matrix(
      [[1.0, 0, 0, 1.0], [0, 1.0, 1.0j, 0], [0, 1.0, -1.0j, 0], [1.0, 0, 0, -1.0]]
    )
  else:
    raise ValueError(f"{polarisations} not in {'linear', 'circular'}")

  Tinv = T.inv()

  # Full Stokes weights
  W = T.H * Mpq.H * Sinv * Mpq * T
  Winv = Tinv * Mpqinv * S * Mpqinv.H * Tinv.H

  # Full Stokes coherencies
  schema = ["I", "Q", "U", "V"]
  C = Winv * (T.H * (Mpq.H * (Sinv * Vpq)))
  # Only keep diagonal of weights
  W = W.diagonal().T  # diagonal() returns row vector

  return schema, C, W


def subs_sympy(expr: sympy.Expr) -> str:
  """Simple string substitution on a sympy expression

  1. "I" -> "1j"
  2. "1.0*" -> ""
  3. "conjugate" -> "conj"
  """
  expr = re.sub(r"\bI\b", "1j", str(expr))
  expr = re.sub(r"\b1.0\*", "", str(expr))
  return re.sub(r"\bconjugate\b", "conj", expr)


def generate_expression(args: Namespace):
  with ExitStack() as stack:
    expr_path = stack.enter_context(
      importlib.resources.path(radiomesh, OUTPUT_FILENAME)
    )
    f = stack.enter_context(open(expr_path, "w"))

    f.write(PREAMBLE)

    for pol_type in POLARISATION_TYPES:
      stokes_schema, coherencies, weights = sympy_expressions(pol_type)
      assert coherencies.shape == (len(stokes_schema), 1)
      assert weights.shape == (len(stokes_schema), 1)
      coherencies = sympy.simplify(coherencies)
      weights = sympy.simplify(weights)

      for stokes, coherencies in zip(stokes_schema, coherencies):
        # for s, stokes in enumerate(STOKES):
        fn_name = f"{pol_type.upper()}_VIS_{stokes.upper()}"
        f.write(f"def {fn_name}({', '.join(VIS_FN_ARGUMENTS)}):\n")
        f.write(f"  return {subs_sympy(coherencies)}\n")
        f.write("\n")

      for stokes, weights in zip(stokes_schema, weights):
        fn_name = f"{pol_type.upper()}_WEIGHT_{stokes.upper()}"
        f.write(f"def {fn_name}({', '.join(WEIGHT_FN_ARGUMENTS)}):\n")
        f.write(f"  return ({subs_sympy(weights)})\n")
        f.write("\n")
