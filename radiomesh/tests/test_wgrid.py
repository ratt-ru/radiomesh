import numpy as np
import pytest

from radiomesh.constants import LIGHTSPEED
from radiomesh.es_kernel import ESKernel
from radiomesh.gridding import WGridderParameters, wgrid
from radiomesh.literals import Datum
from radiomesh.utils import image_params


@pytest.mark.parametrize("nx", [1024])
@pytest.mark.parametrize("ny", [1024])
@pytest.mark.parametrize("fov", [1.0])
@pytest.mark.parametrize("oversampling", [2.0])
def test_numba_wgrid(nx, ny, fov, oversampling):
  """Smoke test. Call with NUMBA_DEBUG_CACHE=1 to ensure caching works"""
  rng = np.random.default_rng(42)

  shape = (100, 7 * 6 // 2, 64, 4)  # (ntime, nbl, nchan, npol)

  pixsize = fov * np.pi / 180.0 / nx

  # Simulate some frequencies and uvws
  # given some initial parameters above
  freqs = np.linspace(0.856e9, 2 * 0.856e9, shape[2])
  uvw = rng.random(shape[:2] + (3,)) - 0.5
  uvw /= pixsize * freqs[0] / LIGHTSPEED

  vis = rng.random(shape) + 0j
  vis += rng.random(shape) * 1j
  weights = rng.random(shape)
  flags = np.zeros_like(weights, np.uint8)

  # Now recompute these params
  nx, ny, pixsizex, pixsizey = image_params(uvw, freqs, fov, oversampling)

  wgrid_params = WGridderParameters(
    nx, ny, pixsizex, pixsizey, ESKernel(2e-13), schema="[XX,XY,YX,YY] -> [I,Q,U,V]"
  )

  vis_grid, weight_grid = wgrid(uvw, vis, weights, flags, freqs, Datum(wgrid_params))

  assert vis_grid.shape == (4, nx, ny)
  assert weight_grid.shape == (nx, ny)
  assert vis_grid.dtype == vis.real.dtype
  assert weight_grid.dtype == weights.dtype
