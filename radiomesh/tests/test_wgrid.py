import numpy as np
import pytest

from radiomesh.constants import LIGHTSPEED
from radiomesh.core import wgrid_data
from radiomesh.es_kernel import ESKernel
from radiomesh.gridding import WGridderParameters, wgrid
from radiomesh.literals import Datum, Schema
from radiomesh.stokes import stokes_funcs
from radiomesh.utils import image_params, wgridder_conventions


@pytest.mark.parametrize("nx", [20])
@pytest.mark.parametrize("fov", [50.0])
@pytest.mark.parametrize("oversampling", [2])
@pytest.mark.parametrize("epsilon", [2e-13])
@pytest.mark.parametrize("apply_jones", [True, False])
def test_numba_wgrid(nx, epsilon, fov, oversampling, apply_jones):
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

  kernel = ESKernel(epsilon, apply_w=True, oversampling=oversampling)
  kernel.support = 10

  # Now recompute these params
  nx, ny, _, pixsizex, pixsizey, w0, dw = image_params(uvw, freqs, fov, kernel)
  nw = 11

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
  )

  ndir = 1
  jones = np.zeros((ntime, na, nchan, ndir, npol), vis.dtype)
  jones[..., 0] = 1.0 + 0j
  jones[..., -1] = 1.0 + 0j
  if apply_jones:
    jones += 0.05 * (rng.normal(size=jones.shape) + 1j * rng.normal(size=jones.shape))
    jones_params = (jones, antenna_pairs, wgrid_params.pol_schema)
  else:
    jones_params = None

  # vis_grid, weight_grid = wgrid(
  vis_grid = wgrid(uvw, vis, weights, flags, freqs, Datum(wgrid_params), jones_params)

  assert vis_grid.shape == (4, nw, nx, ny)
  # assert weight_grid.shape == (4, nx, ny)
  assert vis_grid.dtype == vis.dtype
  # assert weight_grid.dtype == weights.dtype

  # stokes_func wants a matrix form for jones
  jones_dims = (2, 2)
  assert npol == np.prod(jones_dims)
  jones = jones.reshape((ntime, na, nchan, ndir) + (jones_dims))
  vis_func, wgt_func = stokes_funcs(jones, "IQUV", "linear", npol)
  usign, vsign, _, _, _ = wgridder_conventions(0.0, 0.0)

  result = wgrid_data(
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
    w0,
    dw,
    nw,
    wgrid_params.kernel.support,
    wgrid_params.kernel.beta,
    wgrid_params.kernel.mu,
    usign,
    vsign,
  )

  # result = grid_data(
  #   uvw,
  #   freqs,
  #   vis,
  #   weights,
  #   flags,
  #   jones,
  #   ant1,
  #   ant2,
  #   wgrid_params.nx,
  #   wgrid_params.ny,
  #   wgrid_params.pixsizex,
  #   wgrid_params.pixsizey,
  #   npol,
  #   vis_func,
  #   wgt_func,
  #   alpha=wgrid_params.kernel.support,
  #   beta=wgrid_params.kernel.beta,
  #   mu=wgrid_params.kernel.mu,
  #   usign=usign,
  #   vsign=vsign,
  # )

  np.testing.assert_allclose(vis_grid, result)
