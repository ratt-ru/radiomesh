import numba
import numpy as np
import pytest

from radiomesh.es_kernel import generate_poly_coeffs_numpy
from radiomesh.es_kernel_structref import ESKernelProxy, generate_poly_coeffs


@pytest.mark.parametrize(
  "support, beta, e0",
  [
    (5, 2.3, 0.5),
    (7, 2.3, 0.5),
    (4, 1.5, 0.75),
    (8, 3.0, 0.5),
    (6, 2.0, 1.0),
  ],
)
def test_generate_poly_coeffs_vs_numpy(support, beta, e0):
  ref = np.array(generate_poly_coeffs_numpy(support, beta, e0))
  result = generate_poly_coeffs(support, beta, e0)
  np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize(
  "support, beta, e0, rtol, atol",
  [
    (4, 1.5, 0.75, 1e-2, 1e-12),
    (5, 2.3, 0.5, 2e-2, 1e-12),
    (6, 2.0, 1.0, 2e-4, 1e-12),
    (7, 2.3, 0.5, 3e-3, 1e-12),
    (8, 3.0, 0.5, 1e-4, 1e-12),
  ],
)
def test_evaluate_analytic_vs_polynomial(support, beta, e0, rtol, atol):
  """Analytic and polynomial evaluate should agree at sample positions."""

  kw = {
    "epsilon": 2e-13,
    "oversampling": 2.0,
    "beta": beta,
    "e0": e0,
    "support": support,
    "single": False,
    "apply_w": True,
  }

  partial_analytic = ESKernelProxy(analytic=True, **kw)
  partial_poly = ESKernelProxy(analytic=False, **kw)
  full_analytic = ESKernelProxy.fully_specified(analytic=True, **kw)
  full_poly = ESKernelProxy.fully_specified(analytic=False, **kw)

  @numba.njit
  def eval_all(pak, ppk, fak, fpk, x):
    return pak.evaluate(x), ppk.evaluate(x), fak.evaluate(x), fpk.evaluate(x)

  half_support = support / 2.0
  positions = np.linspace(-half_support * 0.99, half_support * 0.99, 20)

  for pos in positions:
    pa_val, pp_val, fa_val, fp_val = eval_all(
      partial_analytic, partial_poly, full_analytic, full_poly, pos
    )
    np.testing.assert_allclose(pa_val, pp_val, rtol=rtol, atol=atol)
    np.testing.assert_allclose(fa_val, fp_val, rtol=rtol, atol=atol)
    np.testing.assert_allclose(pa_val, fa_val)
    np.testing.assert_allclose(pp_val, fp_val)


def test_evaluate_boundary():
  """Positions at or beyond ±half_support should return 0.0."""
  support = 7
  kernel = ESKernelProxy(
    epsilon=2e-13,
    oversampling=2.0,
    beta=2.3,
    e0=0.5,
    support=support,
    analytic=True,
    single=False,
    apply_w=True,
  )

  @numba.njit
  def eval_kernel(k, x):
    return k.evaluate(x)

  half_support = support / 2.0
  assert eval_kernel(kernel, half_support) == 0.0
  assert eval_kernel(kernel, -half_support) == 0.0
  assert eval_kernel(kernel, half_support + 1.0) == 0.0
  assert eval_kernel(kernel, -half_support - 1.0) == 0.0
