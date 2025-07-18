from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

from radiomesh.constants import LIGHTSPEED
from radiomesh.core import vis2im
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
    n = nm1 + 1
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
        phase = (
          freq[chan]
          / LIGHTSPEED
          * (x * uvw[t, bl, 0] + y * uvw[t, bl, 1] - uvw[t, bl, 2] * nm1)
        )
        cphase = np.exp(2j * np.pi * phase)
        # convert to corrected Stokes vis and weights
        vis = vis_func(gp[chan, 0], gq[chan, 0], weight[t, bl, chan], data[t, bl, chan])
        wgt = wgt_func(gp[chan, 0], gq[chan, 0], weight[t, bl, chan])
        for corr in range(ncorr):
          res[corr] += (vis[corr] * wgt[corr] * cphase).real
  return res / n


@pmp("fov", (1.0,))
@pmp("precision", ("single",))
def test_grid_data_now(fov, precision, ms_name):
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
  assert np.allclose(1 + diff, 1, rtol=1e-4, atol=1e-4)
