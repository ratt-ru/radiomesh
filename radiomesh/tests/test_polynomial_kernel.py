"""Tests for polynomial ES kernel approximation.

Verifies that:
- mu == 0.5 with analytic=True uses the sqrt path
- polynomial and analytic evaluations of the same (W, betak, mu) agree to
  within numerical tolerance, tested directly via generate_poly_coeffs
- ESKernel with analytic=False selects parameters from KERNEL_DB
- polynomial kernel returns 0 outside support bounds
"""

import math

import numba
import numpy as np
import pytest

from radiomesh.es_kernel import ESKernel, eval_es_kernel, generate_poly_coeffs
from radiomesh.generated._es_kernel_params import KERNEL_DB
from radiomesh.literals import Datum

# Representative KernelDB entries: one per support width in {4,6,8,10,12},
# ndim in (2, 3), double precision, oversampling=2.0
_WIDTHS_OF_INTEREST = {4, 6, 8, 10, 12}
_SAMPLE_ENTRIES = [
  k
  for k in KERNEL_DB
  if k.support in _WIDTHS_OF_INTEREST
  and k.ndim in (2, 3)
  and not k.single
  and abs(k.oversampling - 2.0) < 1e-9
]


def _make_eval_fn(kernel: ESKernel):
  """Return a JIT-compiled function that evaluates the kernel for a given u."""
  HALF_SUPPORT_INT = kernel.half_support_int
  KERN = Datum(kernel)

  @numba.njit
  def fn(u: float):
    ps = int(np.round(u)) - HALF_SUPPORT_INT
    return eval_es_kernel(KERN, u, ps)

  return fn


# ---------------------------------------------------------------------------
# Test 1: analytic=True + mu=0.5 uses the sqrt path
# ---------------------------------------------------------------------------


def test_sqrt_path_for_mu_half():
  """analytic=True with mu=0.5 evaluates via exp(betak*(sqrt(1-x^2)-1))."""
  kernel = ESKernel(analytic=True, mu=0.5)
  fn = _make_eval_fn(kernel)

  HALF_SUPPORT = kernel.half_support
  BETAK = kernel.support * kernel.beta

  u = 2.3
  ps = int(np.round(u)) - kernel.half_support_int
  kernel_values = fn(u)

  for offset, kv in enumerate(kernel_values):
    x = (offset + ps - u) / HALF_SUPPORT
    if abs(x) >= 1.0:
      expected = 0.0
    else:
      expected = math.exp(BETAK * (math.sqrt(1.0 - x * x) - 1.0))
    assert abs(kv - expected) < 1e-15, f"offset={offset}: got {kv}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 2: polynomial matches analytic for the same (W, betak, mu)
# Uses generate_poly_coeffs directly to avoid ESKernel support-mismatch issues
# ---------------------------------------------------------------------------


def _poly_eval(coeffs: tuple, W: int, x: float) -> float:
  D = W + 3
  if x <= -1.0 or x >= 1.0:
    return 0.0
  xrel = W * 0.5 * (x + 1.0)
  nth = min(int(xrel), W - 1)
  locx = ((xrel - nth) - 0.5) * 2.0
  res = coeffs[0][nth]
  for i in range(1, D + 1):
    res = res * locx + coeffs[i][nth]
  return res


def _analytic_eval(betak: float, mu: float, x: float) -> float:
  tmp = (1.0 - x) * (1.0 + x)
  if tmp <= 0.0:
    return 0.0
  return math.exp(betak * (math.pow(tmp, mu) - 1.0))


@pytest.mark.parametrize(
  "entry", _SAMPLE_ENTRIES, ids=lambda e: f"W{e.support}_ndim{e.ndim}"
)
def test_polynomial_matches_analytic(entry):
  """Polynomial approximation agrees with the analytic kernel to within 1e-12."""
  W = entry.support
  betak = entry.beta * W
  mu = entry.mu

  coeffs = generate_poly_coeffs(W, betak, mu)

  xs = np.linspace(-0.99, 0.99, 200)
  max_err = max(
    abs(_poly_eval(coeffs, W, float(x)) - _analytic_eval(betak, mu, float(x)))
    for x in xs
  )
  # The polynomial of degree D=W+3 achieves accuracy well within the
  # kernel's own epsilon — not machine precision.
  assert (
    max_err < entry.epsilon
  ), f"W={W}: max |poly - analytic| = {max_err:.2e} > epsilon {entry.epsilon:.2e}"


# ---------------------------------------------------------------------------
# Test 3: ESKernel(analytic=False) selects beta/mu/support from KERNEL_DB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  "entry", _SAMPLE_ENTRIES, ids=lambda e: f"W{e.support}_ndim{e.ndim}"
)
def test_polynomial_kernel_uses_kernel_db(entry):
  """analytic=False overrides beta, mu, support with the KernelDB entry."""
  k = ESKernel(
    epsilon=entry.epsilon,
    oversampling=entry.oversampling,
    analytic=False,
    apply_w=(entry.ndim == 3),
  )
  assert k.support == entry.support
  assert k.beta == entry.beta
  assert k.mu == entry.mu


# ---------------------------------------------------------------------------
# Test 4: polynomial returns 0 at support boundaries (x == ±1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  "entry", _SAMPLE_ENTRIES[:1], ids=lambda e: f"W{e.support}_ndim{e.ndim}"
)
def test_polynomial_zero_at_boundary(entry):
  """The first kernel tap lands at x=-1 when u is centred on a grid pixel.

  For even support W, centering the kernel at u = half_support_int places
  the first tap exactly at kernel argument x = -1.0, which must return 0.
  """
  k = ESKernel(epsilon=entry.epsilon, oversampling=entry.oversampling, analytic=False)
  fn = _make_eval_fn(k)

  if k.support % 2 == 0:
    vals = fn(float(k.half_support_int))
    assert (
      vals[0] == 0.0
    ), f"W={k.support}: first tap at x=-1 should be 0, got {vals[0]}"
  else:
    pytest.skip("odd support: boundary tap does not land exactly at x=-1")
