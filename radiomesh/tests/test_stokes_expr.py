import numpy as np
import pytest
import sympy

from radiomesh.generated._stokes_expr import CONVERT_FNS
from radiomesh.scripts.gen_expr import sympy_expressions

STOKES_SCHEMA = ["I", "Q", "U", "V"]

VIS_DIAG_ARGS = "v00 v01 v10 v11 jp00 jp11 jq00 jq11"
WEIGHT_DIAG_ARGS = "w00 w01 w10 w11 jp00 jp11 jq00 jq11"


def diag_expressions(pol_type):
  """Diag-jones oracle: the generator's symbolic derivation with
  off-diagonal jones terms substituted to zero."""
  schema, C, W, _, _ = sympy_expressions(pol_type)
  assert schema == STOKES_SCHEMA
  jp01, jp10, jq01, jq10 = sympy.symbols("jp01 jp10 jq01 jq10", real=False)
  diag = {jp01: 0, jp10: 0, jq01: 0, jq10: 0}
  return sympy.simplify(C.subs(diag)), sympy.simplify(W.subs(diag))


def random_args(names, rng):
  values = {}
  for name in names.split():
    if name.startswith("w"):
      values[name] = rng.uniform(0.1, 2.0)
    else:
      values[name] = rng.uniform(-1, 1) + rng.uniform(-1, 1) * 1j
  return values


@pytest.mark.parametrize("pol_type", ["linear", "circular"])
@pytest.mark.parametrize("stokes", STOKES_SCHEMA)
def test_vis_diagjones(pol_type, stokes):
  rng = np.random.default_rng(42)
  C_diag, _ = diag_expressions(pol_type)
  expr = C_diag[STOKES_SCHEMA.index(stokes)]
  oracle = sympy.lambdify(
    sympy.symbols(VIS_DIAG_ARGS), expr, modules=[{"conjugate": np.conjugate}]
  )
  fn = CONVERT_FNS[("VIS", pol_type.upper(), "DIAGJONES", stokes)]
  for _ in range(5):
    values = random_args(VIS_DIAG_ARGS, rng)
    np.testing.assert_allclose(fn(**values), oracle(**values), rtol=1e-12)


@pytest.mark.parametrize("pol_type", ["linear", "circular"])
@pytest.mark.parametrize("stokes", STOKES_SCHEMA)
def test_weight_diagjones(pol_type, stokes):
  rng = np.random.default_rng(43)
  _, W_diag = diag_expressions(pol_type)
  expr = W_diag[STOKES_SCHEMA.index(stokes)]
  oracle = sympy.lambdify(
    sympy.symbols(WEIGHT_DIAG_ARGS), expr, modules=[{"conjugate": np.conjugate}]
  )
  fn = CONVERT_FNS[("WEIGHT", pol_type.upper(), "DIAGJONES", stokes)]
  for _ in range(5):
    values = random_args(WEIGHT_DIAG_ARGS, rng)
    np.testing.assert_allclose(fn(**values), np.real(oracle(**values)), rtol=1e-12)


@pytest.mark.parametrize("pol_type", ["linear", "circular"])
@pytest.mark.parametrize("stokes", STOKES_SCHEMA)
def test_weight_minvar_diagjones(pol_type, stokes):
  """Minvar oracle mirrors pfb-imaging's stokes_funcs construction:
  4 * Min(*expand(element).args) with Min -> np.minimum."""
  rng = np.random.default_rng(44)
  _, W_diag = diag_expressions(pol_type)
  element = sympy.expand(W_diag[STOKES_SCHEMA.index(stokes)])
  expr = 4 * sympy.Min(
    *(element.args if isinstance(element, sympy.Add) else (element,))
  )
  oracle = sympy.lambdify(
    sympy.symbols(WEIGHT_DIAG_ARGS),
    expr,
    modules=[{"Min": np.minimum, "conjugate": np.conjugate}],
  )
  fn = CONVERT_FNS[("WEIGHT_MINVAR", pol_type.upper(), "DIAGJONES", stokes)]
  for _ in range(5):
    values = random_args(WEIGHT_DIAG_ARGS, rng)
    np.testing.assert_allclose(fn(**values), np.real(oracle(**values)), rtol=1e-12)
