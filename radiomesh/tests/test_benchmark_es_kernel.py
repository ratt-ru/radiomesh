from typing import Any

import numba
import numpy as np
import pytest

from radiomesh.constants import LIGHTSPEED
from radiomesh.es_kernel_structref import ESKernel

SUPPORTS = [4, 7, 12, 15]

BASE_KW: dict[str, Any] = {
  "epsilon": 2e-13,
  "oversampling": 2.0,
  "beta": 2.3,
  "e0": 0.65,
  "single": False,
  "apply_w": True,
}

NX = 1024
NY = 1024
FOV = np.deg2rad(1.0)
PIXSIZEX = FOV / NX
PIXSIZEY = FOV / NY


def _make_gridding_loop(kernel):
  """Create and warm up a JIT-compiled gridding loop for a given kernel."""

  @numba.njit
  def gridding_loop(kernel, uvw, frequencies):
    ntime, nbl, _ = uvw.shape
    nchan = frequencies.shape[0]
    half_support = kernel.support // 2
    x_kernel = np.empty(kernel.support, dtype=np.float64)
    y_kernel = np.empty(kernel.support, dtype=np.float64)
    wavelengths = frequencies / LIGHTSPEED

    for t in range(ntime):
      for bl in range(nbl):
        u = uvw[t, bl, 0]
        v = uvw[t, bl, 1]
        for ch in range(nchan):
          u_scaled = u * wavelengths[ch] * PIXSIZEX
          v_scaled = v * wavelengths[ch] * PIXSIZEY

          u_grid = (u_scaled * NX) % NX
          v_grid = (v_scaled * NY) % NY

          u_pixel_start = int(np.round(u_grid)) - half_support
          v_pixel_start = int(np.round(v_grid)) - half_support

          kernel.evaluate_support(u_grid, u_pixel_start, x_kernel)
          kernel.evaluate_support(v_grid, v_pixel_start, y_kernel)

  # Warm up with small data
  warm_uvw = np.zeros((1, 1, 3), dtype=np.float64)
  warm_freq = np.array([1.0e9], dtype=np.float64)
  gridding_loop(kernel, warm_uvw, warm_freq)

  return gridding_loop


# Build kernels and compiled loops for each support size
_variants = {}

for _support in SUPPORTS:
  _kw = {**BASE_KW, "support": _support}
  _kernels = {
    "partial_analytic": ESKernel(analytic=True, **_kw),
    "partial_polynomial": ESKernel(analytic=False, **_kw),
    "full_analytic": ESKernel.fully_specified(analytic=True, **_kw),
    "full_polynomial": ESKernel.fully_specified(analytic=False, **_kw),
  }
  _loops = {name: _make_gridding_loop(k) for name, k in _kernels.items()}
  _variants[_support] = {"kernels": _kernels, "loops": _loops}


@pytest.mark.parametrize("support", SUPPORTS)
def test_benchmark_partial_analytic(benchmark, support, uvw_coordinates, frequencies):
  name = "partial_analytic"
  benchmark(
    _variants[support]["loops"][name],
    _variants[support]["kernels"][name],
    uvw_coordinates,
    frequencies,
  )


@pytest.mark.parametrize("support", SUPPORTS)
def test_benchmark_partial_polynomial(benchmark, support, uvw_coordinates, frequencies):
  name = "partial_polynomial"
  benchmark(
    _variants[support]["loops"][name],
    _variants[support]["kernels"][name],
    uvw_coordinates,
    frequencies,
  )


@pytest.mark.parametrize("support", SUPPORTS)
def test_benchmark_full_analytic(benchmark, support, uvw_coordinates, frequencies):
  name = "full_analytic"
  benchmark(
    _variants[support]["loops"][name],
    _variants[support]["kernels"][name],
    uvw_coordinates,
    frequencies,
  )


@pytest.mark.parametrize("support", SUPPORTS)
def test_benchmark_full_polynomial(benchmark, support, uvw_coordinates, frequencies):
  name = "full_polynomial"
  benchmark(
    _variants[support]["loops"][name],
    _variants[support]["kernels"][name],
    uvw_coordinates,
    frequencies,
  )
