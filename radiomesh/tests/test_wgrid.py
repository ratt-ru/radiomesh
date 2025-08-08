import numpy as np
import pytest

from radiomesh.constants import LIGHTSPEED
from radiomesh.core import grid_data
from radiomesh.es_kernel import ESKernel
from radiomesh.gridding import WGridderParameters, wgrid
from radiomesh.literals import Datum
from radiomesh.stokes import stokes_funcs
from radiomesh.utils import image_params, wgridder_conventions


@pytest.mark.parametrize("nx", [1024])
@pytest.mark.parametrize("ny", [1024])
@pytest.mark.parametrize("fov", [1.0])
@pytest.mark.parametrize("oversampling", [2.0])
def test_numba_wgrid(nx, ny, fov, oversampling):
  """Smoke test. Call with NUMBA_DEBUG_CACHE=1 to ensure caching works"""
  rng = np.random.default_rng(42)

  shape = (100, 7 * 6 // 2, 64, 4)  # (ntime, nbl, nchan, npol)
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

  # Now recompute these params
  nx, ny, pixsizex, pixsizey = image_params(uvw, freqs, fov, oversampling)

  wgrid_params = WGridderParameters(
    nx, ny, pixsizex, pixsizey, ESKernel(2e-13), schema="[XX,XY,YX,YY] -> [I,Q,U,V]"
  )

  vis_grid, weight_grid = wgrid(uvw, vis, weights, flags, freqs, Datum(wgrid_params))

  assert vis_grid.shape == (4, nx, ny)
  assert weight_grid.shape == (4, nx, ny)
  assert vis_grid.dtype == vis.dtype
  assert weight_grid.dtype == weights.dtype

  # Set up identity jones for a single antenna
  # zero ant1 and ant2 refer to this
  ant1 = ant2 = np.zeros(shape[1], np.int32)
  na = 1
  jones = np.zeros((ntime, na, nchan, npol), np.complex64)
  jones[..., 0] = 1.0
  jones[..., -1] = 1.0

  jones_dims = (2, 2)
  assert npol == np.prod(jones_dims)
  ndir = 1
  jones = jones.reshape((ntime, na, nchan, ndir) + (jones_dims))
  vis_func, wgt_func = stokes_funcs(jones, "IQUV", "linear", npol)

  usign, vsign, _, _, _ = wgridder_conventions(0.0, 0.0)

  result = grid_data(
    uvw,
    freqs,
    vis,
    weights,
    flags,
    jones,
    ant1,
    ant2,
    wgrid_params.nx,
    wgrid_params.ny,
    wgrid_params.pixsizex,
    wgrid_params.pixsizey,
    npol,
    vis_func,
    wgt_func,
    alpha=wgrid_params.kernel.support,
    beta=wgrid_params.kernel.beta,
    mu=wgrid_params.kernel.mu,
    usign=usign,
    vsign=vsign,
  )

  np.testing.assert_allclose(vis_grid, result)
