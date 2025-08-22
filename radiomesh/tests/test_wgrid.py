import numpy as np
import pytest

from radiomesh.constants import LIGHTSPEED
from radiomesh.es_kernel import ESKernel
from radiomesh.gridding import WGridderParameters, wgrid
from radiomesh.literals import Datum, Schema
from radiomesh.utils import image_params


@pytest.mark.parametrize("nx", [20])
@pytest.mark.parametrize("fov", [50.0])
@pytest.mark.parametrize("oversampling", [2])
@pytest.mark.parametrize("epsilon", [2e-13])
@pytest.mark.parametrize("apply_w", [True])
@pytest.mark.parametrize("apply_jones", [True, False])
def test_numba_wgrid(nx, epsilon, fov, oversampling, apply_w, apply_jones):
  """Smoke test. Call with NUMBA_DEBUG_CACHE=1 to ensure caching works"""
  rng = np.random.default_rng(42)

  na = 7
  ant1, ant2 = np.triu_indices(na, 1)
  antenna_pairs = np.stack([ant1, ant2], axis=1)
  shape = (100, ant1.size, 64, 4)  # (ntime, nbl, nchan, npol)
  ntime, nbl, nchan, npol = shape

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

  kernel = ESKernel(epsilon, apply_w=apply_w, oversampling=oversampling)

  # Now recompute these params
  nx, ny, nw, pixsizex, pixsizey, w0, dw = image_params(uvw, freqs, fov, kernel)

  wgrid_params = WGridderParameters(
    nx,
    ny,
    nw,
    pixsizex,
    pixsizey,
    w0,
    dw,
    kernel,
    pol_schema=Schema(("XX", "XY", "YX", "YY")),
    stokes_schema=Schema(("I", "Q", "U", "V")),
    apply_w=apply_w,
    jones_dir_sum=False,
  )

  ndir = 5
  jones = np.zeros((ntime, na, nchan, ndir, npol), vis.dtype)
  jones[..., 0] = 1.0 + 0j
  jones[..., -1] = 1.0 + 0j
  if apply_jones:
    jones += 0.05 * (rng.normal(size=jones.shape) + 1j * rng.normal(size=jones.shape))
    jones_params = (jones, antenna_pairs, wgrid_params.pol_schema)
  else:
    jones_params = None

  wgrid_params.jones_dir_sum = True
  vis_grid = wgrid(uvw, vis, weights, flags, freqs, Datum(wgrid_params), jones_params)

  expected_shape = (len(wgrid_params.stokes_schema),)
  expected_shape += ((nw,) if apply_w else ()) + (nx, ny)
  assert vis_grid.shape == expected_shape
  assert vis_grid.dtype == vis.dtype

  wgrid_params.jones_dir_sum = False
  vis_grid2 = wgrid(uvw, vis, weights, flags, freqs, Datum(wgrid_params), jones_params)

  np.testing.assert_allclose(vis_grid, vis_grid2)
