from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from radiomesh.constants import LIGHTSPEED
from radiomesh.core import (
  _es_kernel,
  es_kernel,
  grid_corrector,
  vis2im,
  vis2im_wgrid,
)
from radiomesh.stokes import stokes_funcs

pmp = pytest.mark.parametrize


def explicit_gridder(
  uvw: npt.NDArray,
  freq: npt.NDArray,
  data: npt.NDArray,
  weight: npt.NDArray,
  flag: npt.NDArray,
  jones: npt.NDArray,
  ant1: npt.NDArray,
  ant2: npt.NDArray,
  nx: int,
  ny: int,
  cellx: float,
  celly: float,
  apply_w: bool,
  vis_func: Callable,
  wgt_func: Callable,
):
  x, y = np.meshgrid(*[-ss / 2 + np.arange(ss) for ss in [nx, ny]], indexing="ij")
  x *= cellx
  y *= celly
  eps = x**2 + y**2
  if apply_w:
    nm1 = -eps / (np.sqrt(1.0 - eps) + 1.0)
    n = (nm1 + 1)[None, :, :]
  else:
    nm1 = 0.0
    n = 1.0
  ntime, nbl, nchan, ncorr = data.shape
  res = np.zeros((ncorr, nx, ny))
  for t in range(ntime):
    for bl in range(nbl):
      p = ant1[bl]
      q = ant2[bl]
      gp = jones[t, p]
      gq = jones[t, q]
      for chan in range(nchan):
        if flag[t, bl, chan].any():
          continue
        # convert to corrected Stokes vis and weights
        vis = vis_func(gp[chan, 0], gq[chan, 0], weight[t, bl, chan], data[t, bl, chan])
        wgt = wgt_func(gp[chan, 0], gq[chan, 0], weight[t, bl, chan])
        u, v, w = uvw[t, bl]
        if w < 0:
          u *= -1
          v *= -1
          w *= -1
          vis = np.conjugate(vis)
        phase = freq[chan] / LIGHTSPEED * (x * u + y * v - w * nm1)
        cphase = np.exp(2j * np.pi * phase)
        for corr in range(ncorr):
          res[corr] += (vis[corr] * wgt[corr] * cphase).real
  return res / n


def taper_trapz(dom, alpha=5, beta=2.3, mu=0.5):
  npix = dom.size
  res = np.zeros(npix, dtype=np.float64)
  alphao2 = alpha / 2
  x = np.linspace(-alphao2, alphao2, npix)
  kern = np.zeros_like(x)
  betak = alpha * beta
  for i, k in enumerate(dom):
    kern[...] = 0.0
    tmp = np.cos(-2 * np.pi * k * x) * _es_kernel(x / alphao2, kern, betak, mu)
    res[i] = np.trapezoid(tmp, x)
  return res


def test_tapers(ms_name):
  """
  Compare Gauss-Legendre integration of kernel to trapz
  """
  fov = 1
  sigma = 2.0
  alpha = 10
  beta = 2.3
  mu = 0.5

  dt = xr.open_datatree(ms_name, engine="xarray-ms:msv2")
  dt_ms = dt[dt.groups[1]]
  freq = dt_ms.frequency.values
  uvw = dt_ms.UVW.values

  # Nyquist
  umax = np.abs(uvw[:, :, 0]).max()
  vmax = np.abs(uvw[:, :, 1]).max()
  uv_max = np.maximum(umax, vmax)
  cell = 1.0 / (2 * uv_max * freq.max() / LIGHTSPEED)
  nx = int(np.ceil(np.deg2rad(fov) / cell))
  if nx % 2:
    nx += 1
  ny = nx // 2
  x = (-(nx // 2) + np.arange(nx)) * cell
  y = (-(ny // 2) + np.arange(ny)) * cell

  xtaper = grid_corrector(x, alpha, beta, mu)
  xtaper_trapz = taper_trapz(x, alpha, beta, mu)
  xdiff = np.abs(xtaper - xtaper_trapz)
  ytaper = grid_corrector(y, alpha, beta, mu)
  ytaper_trapz = taper_trapz(y, alpha, beta, mu)
  ydiff = np.abs(ytaper - ytaper_trapz)

  assert np.allclose(1 + xdiff, 1, atol=1e-10, rtol=1e-10)
  assert np.allclose(1 + ydiff, 1, atol=1e-10, rtol=1e-10)

  # image grid coordinates
  x, y = np.meshgrid(*[-ss / 2 + np.arange(ss) for ss in (nx, ny)], indexing="ij")
  x *= cell
  y *= cell

  # get number of w grids
  w = uvw[:, :, -1].ravel() * freq.max() / LIGHTSPEED
  wabs = np.abs(w)
  wmax = wabs.max()
  wmin = -wmax
  # this tests eqn 10 i.e. the grid corrector in the w-direction
  eps = x**2 + y**2
  nm1 = -eps / (np.sqrt(1.0 - eps) + 1.0)
  dw = 1 / (2 * sigma * np.abs(nm1).max())
  nw = int(np.ceil((wmax - wmin) / dw)) + alpha
  w0 = wmin - dw * (alpha - 1) / 2
  wcorrector = grid_corrector(nm1 * dw, alpha, beta, mu)
  wgrid = w0 + np.arange(nw) * dw
  for ww in w[0::100]:
    res = np.exp(-2j * np.pi * ww * nm1)
    z = (wgrid - ww) / dw
    zkern = es_kernel(z, beta, mu, alpha)
    tmp = np.exp(-2j * np.pi * nm1[:, :, None] * wgrid[None, None, :])
    tmp2 = tmp * zkern[None, None, :]
    res2 = np.sum(tmp2, axis=-1)
    # this is the same as (since dc=dw)
    # res2 = np.trapezoid(tmp2/dw, wgrid, axis=-1)
    res2 /= wcorrector
    diff = np.abs(res - res2)
    assert_array_almost_equal(1 + diff, 1.0, decimal=7)


@pmp("fov", (1.0,))
@pmp("precision", ("single",))
def test_grid_data(fov, precision, ms_name):
  np.random.seed(420)
  if precision == "single":
    complex_type = np.complex64
  else:
    complex_type = np.complex128

  dt = xr.open_datatree(ms_name, engine="xarray-ms:msv2")
  dt_ms = dt[dt.groups[1]]
  dt_ant = dt[dt.groups[2]]
  vis = dt_ms.VISIBILITY.values
  vis[:, :, :, 0] = 1.0
  vis[:, :, :, -1] = 1.0
  wgt = dt_ms.WEIGHT.values
  flag = dt_ms.FLAG.values
  freq = dt_ms.frequency.values
  uvw = dt_ms.UVW.values
  ntime, nbl, nchan, ncorr = vis.shape
  nant = dt_ant.antenna_name.size
  jones = np.zeros((ntime, nant, nchan, 1, 2, 2), dtype=complex_type)
  jones[:, :, :, :, 0, 0] = 1.0
  jones[:, :, :, :, 1, 1] = 1.0
  _, ant1 = np.unique(dt_ms.baseline_antenna1_name.values, return_inverse=True)
  _, ant2 = np.unique(dt_ms.baseline_antenna2_name.values, return_inverse=True)
  pols = dt_ms.polarization.values
  if "XX" in pols or "YY" in pols:
    pol = "linear"
  elif "RR" in pols or "LL" in pols:
    pol = "circular"
  product = "IQUV"
  vis_func, wgt_func = stokes_funcs(jones, product, pol, ncorr)

  # if we don't grid at Nyquist some uv points may fall off the grid
  umax = np.abs(uvw[:, :, 0]).max()
  vmax = np.abs(uvw[:, :, 1]).max()
  uv_max = np.maximum(umax, vmax)
  cell = 1.0 / (2 * uv_max * freq.max() / LIGHTSPEED)
  nx = int(np.ceil(np.deg2rad(fov) / cell))
  if nx % 2:
    nx += 1
  ny = nx

  # we can probably speed up the tests by using the wgridder as reference
  dirty_dft = explicit_gridder(
    uvw,
    freq,
    vis,
    wgt,
    flag,
    jones,
    ant1,
    ant2,
    nx,
    ny,
    cell,
    cell,
    False,
    vis_func,
    wgt_func,
  )

  dirty = vis2im(
    uvw,
    freq,
    vis,
    wgt,
    flag,
    jones,
    ant1,
    ant2,
    nx,
    ny,
    cell,
    cell,
    pol,
    product,
    ncorr,
  )
  # we compare fractional differences because abs values can be very large
  diff = (dirty - dirty_dft) / dirty_dft.max()
  assert_array_almost_equal(1 + diff, 1.0, decimal=6)


@pmp("fov", (1.0,))
@pmp("precision", ("single",))
def test_wgrid_data(fov, precision, ms_name):
  np.random.seed(420)
  if precision == "single":
    # real_type = "f4"
    complex_type = "c8"
  else:
    # real_type = "f8"
    complex_type = "c16"

  dt = xr.open_datatree(ms_name, engine="xarray-ms:msv2")
  dt_ms = dt[dt.groups[1]]
  dt_ant = dt[dt.groups[2]]
  vis = dt_ms.VISIBILITY.values
  vis[:, :, :, 0] = 1.0
  vis[:, :, :, -1] = 1.0
  wgt = dt_ms.WEIGHT.values
  flag = dt_ms.FLAG.values
  freq = dt_ms.frequency.values
  uvw = dt_ms.UVW.values
  ntime, nbl, nchan, ncorr = vis.shape
  nant = dt_ant.antenna_name.size
  jones = np.zeros((ntime, nant, nchan, 1, 2, 2), dtype=complex_type)
  jones[:, :, :, :, 0, 0] = 1.0
  jones[:, :, :, :, 1, 1] = 1.0
  _, ant1 = np.unique(dt_ms.baseline_antenna1_name.values, return_inverse=True)
  _, ant2 = np.unique(dt_ms.baseline_antenna2_name.values, return_inverse=True)
  pols = dt_ms.polarization.values
  if "XX" in pols or "YY" in pols:
    pol = "linear"
  elif "RR" in pols or "LL" in pols:
    pol = "circular"
  product = "IQUV"
  vis_func, wgt_func = stokes_funcs(jones, product, pol, ncorr)

  # if we don't grid at Nyquist some uv points may fall off the grid
  umax = np.abs(uvw[:, :, 0]).max()
  vmax = np.abs(uvw[:, :, 1]).max()
  uv_max = np.maximum(umax, vmax)
  cell = 1.0 / (2 * uv_max * freq.max() / LIGHTSPEED)
  nx = int(np.ceil(np.deg2rad(fov) / cell))
  if nx % 2:
    nx += 1
  ny = nx

  dirty_dft = explicit_gridder(
    uvw,
    freq,
    vis,
    wgt,
    flag,
    jones,
    ant1,
    ant2,
    nx,
    ny,
    cell,
    cell,
    True,
    vis_func,
    wgt_func,
  )

  dirty = vis2im_wgrid(
    uvw,
    freq,
    vis,
    wgt,
    flag,
    jones,
    ant1,
    ant2,
    nx,
    ny,
    cell,
    cell,
    pol,
    product,
    ncorr,
  )

  # we compare fractional differences because abs values can be very large
  diff = (dirty - dirty_dft) / dirty_dft.max()
  assert_array_almost_equal(1 + diff, 1.0, decimal=6)
