import numba
import numpy as np
import pytest

from radiomesh.es_kernel_structref import ESKernel, generate_poly_coeffs
from radiomesh.tests.test_polynomial_kernel import generate_poly_coeffs_numpy


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
  degree = support + 3
  ref = np.array(generate_poly_coeffs_numpy(support, beta, e0, degree))
  result = generate_poly_coeffs(support, beta, e0, degree)
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

  partial_analytic = ESKernel(analytic=True, **kw)
  partial_poly = ESKernel(analytic=False, **kw)
  full_analytic = ESKernel.fully_specified(analytic=True, **kw)
  full_poly = ESKernel.fully_specified(analytic=False, **kw)

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


@pytest.mark.parametrize("support", [4, 7, 12])
@pytest.mark.parametrize("single", [True, False])
def test_allocate_taps(support, single):
  """allocate_taps returns a 1-D array of length support.

  dtype follows ``single`` when it's a literal (fully_specified); otherwise
  falls back to float64.
  """
  kw = {
    "epsilon": 2e-13,
    "oversampling": 2.0,
    "beta": 2.3,
    "e0": 0.5,
    "support": support,
    "analytic": True,
    "single": single,
    "apply_w": True,
  }
  partial = ESKernel(**kw)
  full = ESKernel.fully_specified(**kw)

  @numba.njit
  def allocate(k):
    return k.allocate_taps()

  partial_taps = allocate(partial)
  assert partial_taps.shape == (support,)
  assert partial_taps.dtype == np.float64

  full_taps = allocate(full)
  assert full_taps.shape == (support,)
  assert full_taps.dtype == (np.float32 if single else np.float64)


def test_evaluate_boundary():
  """Positions at or beyond ±half_support should return 0.0."""
  support = 7
  kernel = ESKernel(
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
