import numpy as np
import pytest

from radiomesh.constants import LIGHTSPEED
from radiomesh.core import grid_data, wgrid_data
from radiomesh.es_kernel_structref import ESKernel
from radiomesh.generated._es_kernel_params import KERNEL_DB
from radiomesh.gridding import wgrid
from radiomesh.literals import Datum, Schema
from radiomesh.parameters import WGridderParameters
from radiomesh.stokes import stokes_funcs
from radiomesh.utils import image_params, wgridder_conventions


@pytest.mark.parametrize("nx", [20])
@pytest.mark.parametrize("fov", [50.0])
@pytest.mark.parametrize("oversampling", [2])
@pytest.mark.parametrize("epsilon", [2e-13])
@pytest.mark.parametrize("apply_w", [True, False])
@pytest.mark.parametrize("apply_jones", [True, False])
@pytest.mark.parametrize("analytic", [False, True])
def test_numba_wgrid(nx, epsilon, fov, oversampling, apply_w, apply_jones, analytic):
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

  ndim = 3 if apply_w else 2
  best = None
  for entry in KERNEL_DB:
    if (
      entry.ndim == ndim
      and not entry.single
      and entry.oversampling <= oversampling
      and entry.epsilon <= epsilon
      and (best is None or entry.support < best.support)
    ):
      best = entry
  assert best is not None, "no matching KERNEL_DB entry"

  kernel = ESKernel.fully_specified(
    epsilon=epsilon,
    oversampling=best.oversampling,
    beta=best.beta,
    e0=best.e0,
    support=best.support,
    analytic=analytic,
    single=False,
    apply_w=apply_w,
  )

  # Now recompute these params
  nx, ny, nw, pixsizex, pixsizey, wmin, wmax, dw = image_params(uvw, freqs, fov, kernel)

  wgrid_params = WGridderParameters(
    nu=nx,
    nv=ny,
    kernel=kernel,
    wmin=wmin,
    wmax=wmax,
    nw=nw,
    nm1min=0.0,
    nm1max=0.0,
    nshift=0.0,
  )
  pol_schema = Schema(("XX", "XY", "YX", "YY"))
  stokes_schema = Schema(("I", "Q", "U", "V"))

  ndir = 5 if apply_jones else 1
  jones = np.zeros((ntime, na, nchan, npol), vis.dtype)
  jones[..., 0] = 1.0 + 0j
  jones[..., -1] = 1.0 + 0j
  if apply_jones:
    jones += 0.05 * (rng.normal(size=jones.shape) + 1j * rng.normal(size=jones.shape))
    # wgrid_data and grid_data only apply the first jones matrix
    jones = np.stack([jones] * ndir, axis=3)
    assert jones.shape == (ntime, na, nchan, ndir, npol)
    jones_params = (jones, antenna_pairs, pol_schema)
  else:
    jones_params = None

  vis_grid = wgrid(
    uvw,
    vis,
    weights,
    flags,
    freqs,
    wgrid_params,
    nx,
    ny,
    pixsizex,
    pixsizey,
    pol_schema,
    stokes_schema,
    Datum(apply_w),
    jones_params,
  )

  expected_shape = (len(stokes_schema), ndir)
  expected_shape += ((nw,) if apply_w else ()) + (nx, ny)
  assert vis_grid.shape == expected_shape
  assert vis_grid.dtype == vis.dtype

  # stokes_func wants a matrix form for jones
  jones_dims = (2, 2)
  assert npol == np.prod(jones_dims)
  jones = jones.reshape((ntime, na, nchan, ndir) + (jones_dims))
  vis_func, wgt_func = stokes_funcs(jones, "IQUV", "linear", npol)
  usign, vsign, _, _, _ = wgridder_conventions(0.0, 0.0)

  if apply_w:
    w0 = wmin - dw * (kernel.support / 2.0)
    result = wgrid_data(
      uvw,
      freqs,
      vis,
      weights,
      flags,
      jones,
      ant1,
      ant2,
      nx,
      ny,
      pixsizex,
      pixsizey,
      npol,
      vis_func,
      wgt_func,
      w0,
      dw,
      nw,
      kernel.support,
      kernel.beta,
      kernel.e0,
      usign,
      vsign,
    )
  else:
    result = grid_data(
      uvw,
      freqs,
      vis,
      weights,
      flags,
      jones,
      ant1,
      ant2,
      nx,
      ny,
      pixsizex,
      pixsizey,
      npol,
      vis_func,
      wgt_func,
      alpha=kernel.support,
      beta=kernel.beta,
      e0=kernel.e0,
      usign=usign,
      vsign=vsign,
    )

  # wgrid_data and grid_data always use the analytic kernel; when analytic=False
  # the polynomial approximation introduces small errors, so we relax tolerances.
  tol_kw = {} if analytic else {"rtol": 1e-6, "atol": 1e-15}

  # wgrid_data and grid_data only apply the first jones matrix
  # which we've stacked ndir times in the call to wgrid
  for d in range(ndir):
    np.testing.assert_allclose(vis_grid[:, d, ...], result, **tol_kw)
