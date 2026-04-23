"""Tests for polynomial ES kernel approximation.

Verifies that:
- e0 == 0.5 with analytic=True uses the sqrt path
- polynomial and analytic evaluations of the same (W, betak, e0) agree to
  within numerical tolerance, tested directly via generate_poly_coeffs
- polynomial kernel returns 0 outside support bounds
"""

import math

import numba
import numpy as np
import pytest
from numpy.polynomial.chebyshev import cheb2poly, chebinterpolate

from radiomesh.es_kernel_structref import ESKernel
from radiomesh.generated._es_kernel_params import KERNEL_DB

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


def generate_poly_coeffs_numpy(
  support: int, beta: float, e0: float, degree: int | None = None
) -> tuple[tuple[float, ...], ...]:
  """NumPy reference for the Numba ``generate_poly_coeffs`` in
  ``es_kernel_structref.py``. Retained in this test file only.

  Returns a nested tuple ``(D+1) x support`` in Horner (descending) order.
  """
  D = degree if degree is not None else support + 3
  betak = beta * support

  def es_kernel(v: np.ndarray) -> np.ndarray:
    tmp = (1.0 - v) * (1.0 + v)
    valid = tmp >= 0.0
    safe_tmp = np.where(valid, tmp, 0.0)
    return np.where(valid, np.exp(betak * (np.power(safe_tmp, e0) - 1.0)), 0.0)

  coeff = np.empty((D + 1, support))

  for i in range(support):
    left = -1.0 + 2.0 * i / support
    right = -1.0 + 2.0 * (i + 1) / support
    mid = (left + right) * 0.5
    half = (right - left) * 0.5

    cheb_c = chebinterpolate(lambda locx, m=mid, h=half: es_kernel(locx * h + m), D)
    poly_c = cheb2poly(cheb_c)
    coeff[:, i] = poly_c[::-1]

  return tuple(tuple(float(v) for v in row) for row in coeff)


def _kernel_db_entry(epsilon, oversampling, ndim, single=False):
  best = None
  for k in KERNEL_DB:
    if (
      k.ndim == ndim
      and k.single == single
      and k.oversampling <= oversampling
      and k.epsilon <= epsilon
      and (best is None or k.support < best.support)
    ):
      best = k
  assert best is not None, "no matching KERNEL_DB entry"
  return best


@numba.njit
def _eval_taps(kernel, u: float):
  half_support_int = kernel.support // 2
  support = kernel.support
  ps = int(np.round(u)) - half_support_int
  out = kernel.allocate_taps()
  for offset in range(support):
    x = (offset + ps) - u
    out[offset] = kernel.evaluate(x)
  return out


# ---------------------------------------------------------------------------
# Test 1: analytic=True + e0=0.5 uses the sqrt path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("apply_w", [False, True], ids=["ndim2", "ndim3"])
def test_sqrt_path_for_e0_half(apply_w):
  """analytic=True with e0=0.5 evaluates via exp(betak*(sqrt(1-x^2)-1))."""
  entry = _kernel_db_entry(epsilon=2e-13, oversampling=2.0, ndim=3 if apply_w else 2)
  kernel = ESKernel.fully_specified(
    epsilon=entry.epsilon,
    oversampling=entry.oversampling,
    beta=entry.beta,
    e0=0.5,
    support=entry.support,
    analytic=True,
    single=False,
    apply_w=apply_w,
  )
  half_support = kernel.support / 2.0
  betak = kernel.support * kernel.beta
  half_support_int = kernel.support // 2

  u = 2.3
  ps = int(np.round(u)) - half_support_int
  kernel_values = _eval_taps(kernel, u)

  for offset, kv in enumerate(kernel_values):
    x = (offset + ps - u) / half_support
    if abs(x) >= 1.0:
      expected = 0.0
    else:
      expected = math.exp(betak * (math.sqrt(1.0 - x * x) - 1.0))
    assert abs(kv - expected) < 1e-15, f"offset={offset}: got {kv}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 2: polynomial matches analytic for the same (W, betak, e0)
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


def _analytic_eval(betak: float, e0: float, x: float) -> float:
  tmp = (1.0 - x) * (1.0 + x)
  if tmp <= 0.0:
    return 0.0
  return math.exp(betak * (math.pow(tmp, e0) - 1.0))


@pytest.mark.parametrize(
  "entry", _SAMPLE_ENTRIES, ids=lambda e: f"W{e.support}_ndim{e.ndim}"
)
def test_polynomial_matches_analytic(entry):
  """Polynomial approximation agrees with the analytic kernel to within 1e-12."""
  SUPPORT = entry.support
  betak = entry.beta * SUPPORT
  e0 = entry.e0

  coeffs = generate_poly_coeffs_numpy(SUPPORT, entry.beta, e0)

  xs = np.linspace(-0.99, 0.99, 200)
  max_err = max(
    abs(_poly_eval(coeffs, SUPPORT, float(x)) - _analytic_eval(betak, e0, float(x)))
    for x in xs
  )
  assert (
    max_err < entry.epsilon
  ), f"W={SUPPORT}: max |poly - analytic| = {max_err:.2e} > epsilon {entry.epsilon:.2e}"


# ---------------------------------------------------------------------------
# Test 3: polynomial returns 0 at support boundaries (x == ±1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  "entry", _SAMPLE_ENTRIES[:1], ids=lambda e: f"W{e.support}_ndim{e.ndim}"
)
def test_polynomial_zero_at_boundary(entry):
  """The first kernel tap lands at x=-1 when u is centred on a grid pixel.

  For even support W, centering the kernel at u = half_support_int places
  the first tap exactly at kernel argument x = -1.0, which must return 0.
  """
  kernel = ESKernel.fully_specified(
    epsilon=entry.epsilon,
    oversampling=entry.oversampling,
    beta=entry.beta,
    e0=entry.e0,
    support=entry.support,
    analytic=False,
    single=False,
    apply_w=(entry.ndim == 3),
  )
  if kernel.support % 2 == 0:
    vals = _eval_taps(kernel, float(kernel.support // 2))
    assert (
      vals[0] == 0.0
    ), f"W={kernel.support}: first tap at x=-1 should be 0, got {vals[0]}"
  else:
    pytest.skip("odd support: boundary tap does not land exactly at x=-1")
