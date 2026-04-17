"""Tests that the numba overload of estimage_gridding_parameters
matches the pure-Python version."""

import pytest
from numba import njit

from radiomesh.parameters import (
  COMPLEX_FFT_PRIMES,
  REAL_FFT_PRIMES,
  _nm1_range,
  _smallest_smooth_number_above,
  estimate_gridding_parameters,
)

# ---------------------------------------------------------------------------
# Helper: call estimate_gridding_parameters from njit
# (keyword-only args must be positional)
# ---------------------------------------------------------------------------


@njit
def njit_estimate_gridding_parameters(
  nx,
  ny,
  pixsize_x,
  pixsize_y,
  epsilon,
  apply_w,
  single,
  nvis,
  nthreads,
  wmin_d,
  wmax_d,
  lshift,
  mshift,
  no_nshift,
  oversampling_min,
  oversampling_max,
  gridding,
):
  return estimate_gridding_parameters(
    nx,
    ny,
    pixsize_x,
    pixsize_y,
    epsilon,
    apply_w,
    single,
    nvis,
    nthreads,
    wmin_d,
    wmax_d,
    lshift,
    mshift,
    no_nshift,
    oversampling_min,
    oversampling_max,
    gridding,
  )


def _call_both(nx, ny, pixsize_x, pixsize_y, epsilon, **kwargs):
  """Call both Python and numba versions, return (py_result, nb_result)."""
  py = estimate_gridding_parameters(nx, ny, pixsize_x, pixsize_y, epsilon, **kwargs)
  # Build positional args for njit wrapper
  nb = njit_estimate_gridding_parameters(
    nx,
    ny,
    pixsize_x,
    pixsize_y,
    epsilon,
    kwargs.get("apply_w", False),
    kwargs.get("single", False),
    kwargs.get("nvis", 0),
    kwargs.get("nthreads", 1),
    kwargs.get("wmin_d", 0.0),
    kwargs.get("wmax_d", 0.0),
    kwargs.get("lshift", 0.0),
    kwargs.get("mshift", 0.0),
    kwargs.get("no_nshift", False),
    kwargs.get("oversampling_min", 1.1),
    kwargs.get("oversampling_max", 2.6),
    kwargs.get("gridding", True),
  )
  return py, nb


def _assert_grid_params_equal(py, nb):
  assert py.nu == nb.nu, f"nu: {py.nu} != {nb.nu}"
  assert py.nv == nb.nv, f"nv: {py.nv} != {nb.nv}"
  assert py.kernel.support == nb.kernel.support
  assert py.kernel.oversampling == nb.kernel.oversampling
  assert py.kernel.epsilon == nb.kernel.epsilon
  assert py.kernel.beta == nb.kernel.beta
  assert py.kernel.e0 == nb.kernel.e0
  assert py.kernel.apply_w is nb.kernel.apply_w
  assert py.kernel.single == nb.kernel.single
  assert py.nm1min == pytest.approx(nb.nm1min)
  assert py.nm1max == pytest.approx(nb.nm1max)
  assert py.nshift == pytest.approx(nb.nshift)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSmallestSmoothNumber:
  """Verify the numba and python version produce the same values."""

  @pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 15, 16, 17, 100, 1000, 2048, 4096])
  def test_complex_fft_primes(self, n):
    @njit
    def call(n, primes):
      return _smallest_smooth_number_above(n, primes)

    py = _smallest_smooth_number_above(n, COMPLEX_FFT_PRIMES)
    nb = call(n, COMPLEX_FFT_PRIMES)
    assert py == nb, f"n={n}: Python={py}, numba={nb}"

  @pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 15, 16, 17, 100, 1000, 2048, 4096])
  def test_real_fft_primes(self, n):
    @njit
    def call(n, primes):
      return _smallest_smooth_number_above(n, primes)

    py = _smallest_smooth_number_above(n, REAL_FFT_PRIMES)
    nb = call(n, REAL_FFT_PRIMES)
    assert py == nb, f"n={n}: Python={py}, numba={nb}"


class TestNm1Range:
  def test_basic(self):
    @njit
    def call(nxdirty, nydirty, psx, psy, ls, ms):
      return _nm1_range(nxdirty, nydirty, psx, psy, ls, ms)

    cases = [
      (256, 256, 1e-4, 1e-4, 0.0, 0.0),
      (512, 512, 2e-5, 2e-5, 0.0, 0.0),
      (128, 256, 1e-4, 1e-4, 0.001, -0.002),
      (64, 64, 5e-4, 5e-4, 0.0, 0.0),
    ]
    for args in cases:
      py = _nm1_range(*args)
      nb = call(*args)
      assert py[0] == pytest.approx(nb[0]), f"nm1min mismatch for {args}"
      assert py[1] == pytest.approx(nb[1]), f"nm1max mismatch for {args}"


class TestEstimatedGriddingParameters:
  def test_basic_2d(self):
    py, nb = _call_both(256, 256, 1e-4, 1e-4, 1e-4)
    _assert_grid_params_equal(py, nb)

  def test_basic_2d_single(self):
    py, nb = _call_both(256, 256, 1e-4, 1e-4, 1e-3, single=True)
    _assert_grid_params_equal(py, nb)

  def test_rectangular(self):
    py, nb = _call_both(512, 128, 2e-5, 8e-5, 1e-5)
    _assert_grid_params_equal(py, nb)

  def test_with_nvis_nthreads(self):
    py, nb = _call_both(
      256,
      256,
      1e-4,
      1e-4,
      1e-4,
      nvis=1_000_000,
      nthreads=4,
    )
    _assert_grid_params_equal(py, nb)

  def test_with_w_gridding(self):
    py, nb = _call_both(
      256,
      256,
      1e-4,
      1e-4,
      1e-4,
      apply_w=True,
      nvis=100_000,
      nthreads=2,
      wmin_d=-100.0,
      wmax_d=100.0,
    )
    _assert_grid_params_equal(py, nb)

  def test_with_lm_shift(self):
    py, nb = _call_both(
      256,
      256,
      1e-4,
      1e-4,
      1e-4,
      lshift=0.001,
      mshift=-0.002,
    )
    _assert_grid_params_equal(py, nb)

  def test_with_w_and_shift(self):
    py, nb = _call_both(
      128,
      128,
      2e-4,
      2e-4,
      1e-5,
      apply_w=True,
      nvis=500_000,
      nthreads=8,
      wmin_d=-50.0,
      wmax_d=50.0,
      lshift=0.0005,
      mshift=0.001,
    )
    _assert_grid_params_equal(py, nb)

  def test_no_nshift(self):
    py, nb = _call_both(
      256,
      256,
      1e-4,
      1e-4,
      1e-4,
      apply_w=True,
      no_nshift=True,
      wmin_d=-100.0,
      wmax_d=100.0,
    )
    _assert_grid_params_equal(py, nb)

  def test_degridding(self):
    py, nb = _call_both(
      256,
      256,
      1e-4,
      1e-4,
      1e-4,
      nvis=100_000,
      gridding=False,
    )
    _assert_grid_params_equal(py, nb)

  def test_tight_oversampling_range(self):
    py, nb = _call_both(
      256,
      256,
      1e-4,
      1e-4,
      1e-3,
      oversampling_min=1.5,
      oversampling_max=2.0,
    )
    _assert_grid_params_equal(py, nb)

  def test_large_epsilon(self):
    py, nb = _call_both(256, 256, 1e-4, 1e-4, 1e-2)
    _assert_grid_params_equal(py, nb)

  def test_return_type(self):
    """Verify the numba version returns a proper GridParameters NamedTuple."""
    _, nb = _call_both(256, 256, 1e-4, 1e-4, 1e-4)
    assert hasattr(nb, "nu")
    assert hasattr(nb, "nv")
    assert hasattr(nb, "kernel")
    assert hasattr(nb, "nm1min")
    assert hasattr(nb, "nm1max")
    assert hasattr(nb, "nshift")
