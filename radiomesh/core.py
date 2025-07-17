from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt
from ducc0.fft import good_size
from numba import literally, njit
from numba.core import types
from numba.core.errors import RequireLiteralValue
from numba.extending import overload

from radiomesh.constants import LIGHTSPEED
from radiomesh.stokes import stokes_funcs
from radiomesh.utils import wgridder_conventions

Fs = partial(np.fft.fftshift, axes=(-2, -1))
iFs = partial(np.fft.ifftshift, axes=(-2, -1))

JIT_OPTIONS = {"nogil": True, "cache": True, "error_model": "numpy", "fastmath": False}


# hardcode 2D gridding params
# support, padding, epsilon, beta, alpha
# 8, 1.40, 3.1679034e-05, 1.83155364990234371, 0.516968027750650871  (single precision)
# 8, 1.40, 8.5117152e-06, 1.82943505181206612, 0.517185719807942368  (double precision)


def grid_corrector(ng, supp, beta, alpha, nroots=32):
  # even number of roots required to exploit even symmetry of integrand
  if nroots % 2:
    nroots += 1
  dom = np.linspace(-0.5, 0.5, ng, endpoint=False)
  q, wgt = np.polynomial.legendre.leggauss(nroots)
  idx = q > 0
  q = q[idx]
  wgt = wgt[idx]
  z = np.outer(dom, q)
  xkern = np.zeros(q.size)
  _es_kernel(q, xkern, supp * beta, alpha)
  tmp = np.sum(np.cos(np.pi * supp * z) * xkern[None, :] * wgt[None, :], axis=-1)
  return supp * tmp


@njit(**JIT_OPTIONS, inline="always")
def _es_kernel(x, kern, betak, alphak):
  # oneminxsq = (1 - x) * (1 + x)
  # mx = oneminxsq <= 1
  # kern[mx] = np.exp(betak * (np.pow(oneminxsq[mx], alphak) - 1))
  # kern[~mx] = 0.0
  kern[...] = 0.0
  mask = np.abs(x) <= 1
  kern[mask] = np.exp(betak * (np.sqrt(1 - x[mask] ** 2) - 1))
  return kern


@njit(**JIT_OPTIONS)
def grid_data(
  uvw: npt.NDArray,
  freq: npt.NDArray,
  data: npt.NDArray,
  weight: npt.NDArray,
  flag: npt.NDArray,
  jones: npt.NDArray,
  ant1: npt.NDArray,
  ant2: npt.NDArray,
  ngx: int,
  ngy: int,
  cellx: float,
  celly: float,
  pol: str,
  product: str,
  nc: int,
):
  grid = _grid_data_impl(
    uvw,
    freq,
    data,
    weight,
    flag,
    jones,
    ant1,
    ant2,
    ngx,
    ngy,
    cellx,
    celly,
    literally(pol),
    literally(product),
    literally(nc),
  )

  return grid


def _grid_data_impl(
  uvw: npt.NDArray,
  freq: npt.NDArray,
  data: npt.NDArray,
  weight: npt.NDArray,
  flag: npt.NDArray,
  jones: npt.NDArray,
  ant1: npt.NDArray,
  ant2: npt.NDArray,
  ngx: int,
  ngy: int,
  cellx: float,
  celly: float,
  pol: str,
  product: str,
  nc: int,
  padding=2.0,
  supp=8,
  beta=2.3,
  alpha=0.5,
):
  raise NotImplementedError


@overload(
  _grid_data_impl, prefer_literal=True, jit_options={**JIT_OPTIONS, "parallel": True}
)
def nb_grid_data_impl(
  uvw,
  freq,
  data,
  weight,
  flag,
  jones,
  ant1,
  ant2,
  ngx,
  ngy,
  cellx,
  celly,
  pol,
  product,
  nc,
  padding=2.0,
  supp=8,
  beta=2.3,
  alpha=0.5,
):
  if not isinstance(pol, types.StringLiteral):
    raise RequireLiteralValue(f"'pol' {pol} is not a str literal")
  if not isinstance(product, types.StringLiteral):
    raise RequireLiteralValue(f"'product' {product} is not a str literal")
  if not isinstance(nc, types.IntegerLiteral):
    raise RequireLiteralValue(f"'nc' {nc} is not a int literal")
  vis_func, wgt_func = stokes_funcs(
    jones, product.literal_value, pol.literal_value, nc.literal_value
  )
  ns = len(product.literal_value)
  usign, vsign, _, _, _ = wgridder_conventions(0.0, 0.0)

  def _impl(
    uvw,
    freq,
    data,
    weight,
    flag,
    jones,
    ant1,
    ant2,
    ngx,
    ngy,
    cellx,
    celly,
    pol,
    product,
    nc,
    padding=2.0,
    supp=8,
    beta=2.3,
    alpha=0.5,
  ):
    ntime, nbl, nchan, ncorr = data.shape
    vis_grid = np.zeros((ns, ngx, ngy), dtype=data.dtype)

    # ufreq
    u_cell = 1 / (ngx * cellx)
    umax = 1 / cellx / 2

    # vfreq
    v_cell = 1 / (ngy * celly)
    vmax = 1 / celly / 2

    normfreq = freq / LIGHTSPEED
    ko2 = supp / 2
    betak = beta * supp
    pos = (np.arange(supp) - ko2).astype(np.int64)
    xkern = np.zeros(supp)
    ykern = np.zeros(supp)
    for t in range(ntime):
      for bl in range(nbl):
        p = int(ant1[bl])
        q = int(ant2[bl])
        gp = jones[t, p, :, 0]
        gq = jones[t, q, :, 0]
        uvw_row = uvw[t, bl]
        wgt_row = weight[t, bl]
        vis_row = data[t, bl]
        for chan in range(nchan):
          if flag[t, bl, chan].any():
            continue
          wgt = wgt_func(gp[chan], gq[chan], wgt_row[chan])
          vis = vis_func(gp[chan], gq[chan], wgt_row[chan], vis_row[chan])

          # current uv coords
          chan_normfreq = normfreq[chan]
          u_tmp = uvw_row[0] * chan_normfreq * usign
          v_tmp = uvw_row[1] * chan_normfreq * vsign
          # pixel coordinates
          ug = (u_tmp + umax) / u_cell
          vg = (v_tmp + vmax) / v_cell
          # indices
          u_idx = int(np.round(ug))
          v_idx = int(np.round(vg))

          # the grid position is pos + u_idx on the un-fftshifted grid
          x_idx = pos + u_idx
          x = (x_idx - ug) / ko2
          # this finds the index on the fftshifted grid
          # x_idx = (x_idx + ngx//2) % ngx
          _es_kernel(x, xkern, betak, alpha)
          y_idx = pos + v_idx
          y = (y_idx - vg) / ko2
          # y_idx = (y_idx + ngy//2) % ngy
          _es_kernel(y, ykern, betak, alpha)

          for c in range(ncorr):
            wc = wgt[c]
            for i, xi in zip(x_idx, xkern):
              for j, yj in zip(y_idx, ykern):
                xyw = xi * yj * wc
                # wgt_grid[c, i, j] += xyw
                vis_grid[c, i, j] += xyw * vis[c]

    return vis_grid

  # _impl.returns = types.Tuple([types.Array(types.complex128, 3, 'C'),
  #                              types.Array(types.float64, 3, 'C')])
  return _impl


@njit(**JIT_OPTIONS)
def grid_data_np(
  uvw: npt.NDArray,
  freq: npt.NDArray,
  data: npt.NDArray,
  weight: npt.NDArray,
  flag: npt.NDArray,
  jones: npt.NDArray,
  ant1: npt.NDArray,
  ant2: npt.NDArray,
  ngx: int,
  ngy: int,
  cellx: float,
  celly: float,
  pol: str,
  product: str,
  nc: int,
  vis_func: Callable,
  wgt_func: Callable,
  padding=2.0,
  supp=10,
  beta=2.3,
  alpha=0.5,
  usign=1,
  vsign=-1,
):
  ntime, nbl, nchan, ncorr = data.shape
  vis_grid = np.zeros((nc, ngx, ngy), dtype=data.dtype)

  # ufreq
  u_cell = 1 / (ngx * cellx)
  umax = 1 / cellx / 2

  # vfreq
  v_cell = 1 / (ngy * celly)
  vmax = 1 / celly / 2

  normfreq = freq / LIGHTSPEED
  ko2 = supp / 2
  betak = beta * supp
  pos = (np.arange(supp) - ko2).astype(np.int64)
  xkern = np.zeros(supp)
  ykern = np.zeros(supp)
  for t in range(ntime):
    for bl in range(nbl):
      p = int(ant1[bl])
      q = int(ant2[bl])
      gp = jones[t, p, :, 0]
      gq = jones[t, q, :, 0]
      uvw_row = uvw[t, bl]
      wgt_row = weight[t, bl]
      vis_row = data[t, bl]
      for chan in range(nchan):
        if flag[t, bl, chan].any():
          continue
        wgt = wgt_func(gp[chan], gq[chan], wgt_row[chan])
        vis = vis_func(gp[chan], gq[chan], wgt_row[chan], vis_row[chan])

        # current uv coords
        chan_normfreq = normfreq[chan]
        u_tmp = uvw_row[0] * chan_normfreq * usign
        v_tmp = uvw_row[1] * chan_normfreq * vsign
        # pixel coordinates
        ug = (u_tmp + umax) / u_cell
        vg = (v_tmp + vmax) / v_cell
        # indices
        u_idx = int(np.round(ug))
        v_idx = int(np.round(vg))

        # the grid position is pos + u_idx on the un-fftshifted grid
        x_idx = pos + u_idx
        x = (x_idx - ug) / ko2
        # this finds the index on the fftshifted grid
        # x_idx = (x_idx + ngx//2) % ngx
        _es_kernel(x, xkern, betak, alpha)
        y_idx = pos + v_idx
        y = (y_idx - vg) / ko2
        # y_idx = (y_idx + ngy//2) % ngy
        _es_kernel(y, ykern, betak, alpha)

        for c in range(ncorr):
          wc = wgt[c]
          for i, xi in zip(x_idx, xkern):
            for j, yj in zip(y_idx, ykern):
              xyw = xi * yj * wc
              vis_grid[c, i, j] += xyw * vis[c]

  return vis_grid


def vis2im(
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
  pol: str,
  product: str,
  nc: int,
  padding=2.0,
  supp=10,
  beta=2.3,
  alpha=0.5,
):
  ngx = good_size(int(padding * nx))
  # make sure it is even and a good size for the FFT
  while ngx % 2:
    ngx = good_size(ngx + 1)
  xcorrector = grid_corrector(ngx, supp, beta, alpha)
  padxl = (ngx - nx) // 2
  padxr = ngx - nx - padxl
  slcx = slice(padxl, -padxr)
  ngy = good_size(int(padding * ny))
  # make sure it is even and a good size for the FFT
  while ngy % 2:
    ngy = good_size(ngy + 1)
  ycorrector = grid_corrector(ngy, supp, beta, alpha)
  padyl = (ngy - ny) // 2
  padyr = ngy - ny - padyl
  slcy = slice(padyl, -padyr)

  # taper
  corrector = xcorrector[None, :, None] * ycorrector[None, None, :]

  vis_func, wgt_func = stokes_funcs(jones, product, pol, nc)

  grid = grid_data_np(
    uvw,
    freq,
    data,
    weight,
    flag,
    jones,
    ant1,
    ant2,
    ngx,
    ngy,
    cellx,
    celly,
    pol,
    product,
    nc,
    vis_func,
    wgt_func,
    padding=2.0,
    supp=10,
    beta=2.3,
    alpha=0.5,
    usign=1,
    vsign=1,
  )

  # now the FFTs
  # the *ngx*ngy corrects for the FFT normalisation
  dirty = np.fft.ifft2(iFs(grid), axes=(-2, -1)) * ngx * ngy
  dirty = Fs(dirty.real)
  # apply taper
  dirty /= corrector
  return dirty[:, slcx, slcy]
