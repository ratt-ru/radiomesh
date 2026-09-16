import numba
import numpy as np
import pytest
from rarg_numba_patterns.literals import Datum

from radiomesh.es_kernel_structref import (
  ESKernel,
  TemplateESKernel,
  generate_poly_coeffs,
  polynomial_degree,
)
from radiomesh.intrinsics import stack_array
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
  degree = polynomial_degree(support)
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
  assert partial_taps.dtype == (np.float32 if single else np.float64)

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


def tap_positions(support, x):
  """Kernel positions of the ``support`` taps for local coordinate ``x``.

  Tap ``k`` covers grid cell ``k`` of the kernel's ``[-support/2, support/2]``
  footprint, and ``x`` in ``[-1, 1]`` spans that cell, so the tap sits at the
  cell centre displaced by ``x / 2``.
  """
  return np.array([-support / 2.0 + k + 0.5 + x / 2.0 for k in range(support)])


# float32 taps carry ~7 digits; the polynomial approximation error itself is
# common to both sides of every comparison here.
TEMPLATE_TOL = {True: 3e-6, False: 1e-11}


@pytest.mark.parametrize("support", [4, 6, 7, 15])
@pytest.mark.parametrize("single", [True, False], ids=["single", "double"])
def test_template_es_kernel(support, single):
  """eval2s reproduces ESKernel.evaluate at every tap position.

  This is what pins the Horner chains, the symmetry mirror and the zero
  padding: eval2s evaluates the same polynomials as the scalar evaluator, just
  all ``support`` of them at once, so the two must agree to round-off.

  ``ESKernel.evaluate`` is float64 regardless of ``single``, so it stays an
  accurate reference for both tap precisions.
  """
  es_kernel = ESKernel(support=support, analytic=False)
  template_es_kernel = TemplateESKernel(es_kernel, Datum(support), Datum(single))

  ntaps = template_es_kernel.ntaps
  dtype = np.float32 if single else np.float64

  # fastmath must be on the *caller*: eval2s is inline="always", so its body is
  # inlined at Numba IR level and the overload's own flags never reach codegen.
  #
  # The tap buffers are stack_array rather than np.empty: an alloca is provably
  # local, so LLVM can rule out the ku[k] store clobbering the next coeffs load
  # and pack the Horner chains. An NRT array does not give that -- NRT_Allocate
  # marks only the call's return value noalias, and the attribute is lost when
  # the data pointer is read back out of the meminfo.
  @numba.njit(fastmath=True)
  def call_template(template_es_kernel, x, y, z, nth, ku, kv):
    taps_u = stack_array((ntaps,), dtype)
    taps_v = stack_array((ntaps,), dtype)
    template_es_kernel.eval2s(x, y, z, nth, taps_u, taps_v)
    for i in range(ntaps):
      ku[i] = taps_u[i]
      kv[i] = taps_v[i]

  @numba.njit
  def evaluate(kernel, x):
    return kernel.evaluate(x)

  rng = np.random.default_rng(support)

  for _ in range(8):
    x, y = rng.uniform(-1.0, 1.0, 2)
    nth = int(rng.integers(0, support))
    # z is (w0 - w)/dw; 2*(z - nth) + (support - 1) puts the w coordinate into
    # the same [-1, 1] local frame as x and y.
    z = nth - (support / 2.0 - rng.uniform(0.0, 1.0))

    ku = np.full(ntaps, np.nan, dtype)
    kv = np.full(ntaps, np.nan, dtype)
    call_template(template_es_kernel, x, y, z, nth, ku, kv)

    # eval2s folds the single w tap into the u taps so the gridding loop never
    # multiplies by it again. Mirror eval2s' own reduction of the w coordinate.
    z_local = 2.0 * (z - nth) + (support - 1)
    w_nth = nth
    if w_nth >= template_es_kernel.row_stride:
      z_local, w_nth = -z_local, support - 1 - w_nth
    z_factor = evaluate(es_kernel, tap_positions(support, z_local)[w_nth])

    expected_v = np.array([evaluate(es_kernel, p) for p in tap_positions(support, y)])
    expected_u = np.array([evaluate(es_kernel, p) for p in tap_positions(support, x)])
    expected_u = expected_u * z_factor

    # The ES kernel is normalised to a peak of 1.0, so an *absolute* tolerance
    # is already a peak-relative one. That is the right measure here: a tap far
    # out on the kernel skirt is computed by a Horner chain whose intermediate
    # terms are O(1), so cancelling down to 1e-12 costs most of the mantissa
    # and its own relative accuracy is poor while its absolute error stays at
    # round-off. Measured worst absolute error is 1.3e-7 single, 3e-15 double.
    tol = TEMPLATE_TOL[single]
    np.testing.assert_allclose(kv[:support], expected_v, rtol=tol, atol=tol)
    np.testing.assert_allclose(ku[:support], expected_u, rtol=tol, atol=tol)

    # Taps past the support are padding and must be held at zero.
    assert np.all(ku[support:] == 0.0)
    assert np.all(kv[support:] == 0.0)


@pytest.mark.parametrize("support", [4, 6, 7, 15])
@pytest.mark.parametrize("single", [True, False], ids=["single", "double"])
def test_template_es_kernel_eval2(support, single):
  """eval2 reproduces ESKernel.evaluate at every tap position.

  eval2 is eval2s with the w axis removed, so the u taps must come out
  unscaled by any w factor and the v taps identical to eval2s'.
  """
  es_kernel = ESKernel(support=support, analytic=False)
  template_es_kernel = TemplateESKernel(es_kernel, Datum(support), Datum(single))

  ntaps = template_es_kernel.ntaps
  dtype = np.float32 if single else np.float64

  # fastmath (and stack_array) on the caller, for the reasons given in
  # test_template_es_kernel.
  @numba.njit(fastmath=True)
  def call_template(template_es_kernel, x, y, ku, kv):
    taps_u = stack_array((ntaps,), dtype)
    taps_v = stack_array((ntaps,), dtype)
    template_es_kernel.eval2(x, y, taps_u, taps_v)
    for i in range(ntaps):
      ku[i] = taps_u[i]
      kv[i] = taps_v[i]

  @numba.njit
  def evaluate(kernel, x):
    return kernel.evaluate(x)

  rng = np.random.default_rng(support)

  for _ in range(8):
    x, y = rng.uniform(-1.0, 1.0, 2)

    ku = np.full(ntaps, np.nan, dtype)
    kv = np.full(ntaps, np.nan, dtype)
    call_template(template_es_kernel, x, y, ku, kv)

    expected_u = np.array([evaluate(es_kernel, p) for p in tap_positions(support, x)])
    expected_v = np.array([evaluate(es_kernel, p) for p in tap_positions(support, y)])

    tol = TEMPLATE_TOL[single]
    np.testing.assert_allclose(ku[:support], expected_u, rtol=tol, atol=tol)
    np.testing.assert_allclose(kv[:support], expected_v, rtol=tol, atol=tol)

    # Taps past the support are padding and must be held at zero.
    assert np.all(ku[support:] == 0.0)
    assert np.all(kv[support:] == 0.0)


@pytest.mark.parametrize("support", [4, 6, 7, 15])
@pytest.mark.parametrize("single", [True, False], ids=["single", "double"])
def test_template_es_kernel_eval(support, single):
  """eval reproduces ESKernel.evaluate for a single tap.

  eval takes a position normalised to ``[-1, 1]``, ESKernel.evaluate one in
  grid pixels, so the reference position is scaled by ``support / 2``. This
  pins the mirroring of sub-intervals past ``row_stride``, which is the only
  part of the truncated table eval touches that the scalar evaluator doesn't.
  """
  es_kernel = ESKernel(support=support, analytic=False)
  template_es_kernel = TemplateESKernel(es_kernel, Datum(support), Datum(single))

  @numba.njit(fastmath=True)
  def call_eval(template_es_kernel, x):
    return template_es_kernel.eval(x)

  @numba.njit
  def evaluate(kernel, x):
    return kernel.evaluate(x)

  half_support = support / 2.0
  tol = TEMPLATE_TOL[single]

  # Cover every sub-interval, both those stored and those mirrored.
  for x in np.linspace(-0.999, 0.999, 8 * support):
    np.testing.assert_allclose(
      call_eval(template_es_kernel, x),
      evaluate(es_kernel, x * half_support),
      rtol=tol,
      atol=tol,
    )

  # Outside the footprint the kernel is zero.
  for x in (-1.0, 1.0, -1.5, 2.0):
    assert call_eval(template_es_kernel, x) == 0.0
