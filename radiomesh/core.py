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
# alpha (support), sigma (padding), epsilon, beta, mu
# 8, 1.40, 3.1679034e-05, 1.83155364990234371, 0.516968027750650871  (single precision)
# 8, 1.40, 8.5117152e-06, 1.82943505181206612, 0.517185719807942368  (double precision)


# TODO - the grid correctors could be further optimized
# see discussion below eq 3.10 in https://arxiv.org/pdf/1808.06736
def grid_corrector2D(dom, alpha, beta, mu, nroots=32):
  # even number of roots required to exploit even symmetry of integrand
  if nroots % 2:
    nroots += 1
  q, wgt = np.polynomial.legendre.leggauss(nroots)
  idx = q > 0
  q = q[idx]
  wgt = wgt[idx]
  z = np.einsum("ij,k->ijk", dom, q)
  xkern = es_kernel(q * alpha / 2, beta, mu, alpha)
  tmp = np.sum(
    np.cos(-np.pi * alpha * z) * xkern[None, None, :] * wgt[None, None, :], axis=-1
  )
  return alpha * tmp


def grid_corrector(dom, alpha, beta, mu, nroots=32):
  # even number of roots required to exploit even symmetry of integrand
  if nroots % 2:
    nroots += 1
  q, wgt = np.polynomial.legendre.leggauss(nroots)
  idx = q > 0
  q = q[idx]
  wgt = wgt[idx]
  z = np.outer(dom, q)
  xkern = np.zeros(q.size)
  _es_kernel(q, xkern, alpha * beta, mu)
  tmp = np.sum(np.cos(-np.pi * alpha * z) * xkern[None, :] * wgt[None, :], axis=-1)
  return alpha * tmp


@njit(**JIT_OPTIONS, inline="always")
def _es_kernel(x, kern, betak, mu):
  """
  Scaled version of the kernel

  exp(betak * (power(1 - x ** 2, mu) - 1))

  i.e. support squashed to [-1, 1] with betak = beta * alpha
  """
  kern[...] = 0.0
  mask = np.abs(x) <= 1
  kern[mask] = np.exp(betak * (np.power(1 - x[mask] ** 2, mu) - 1))
  return kern


@njit(**JIT_OPTIONS)
def es_kernel(x, beta, mu, alpha):
  """
  Unscaled version of the gridding

  exp(alpha * beta * (power(1 - (2*x/alpha) ** 2, mu) - 1))

  i.e. x is in pixel units
  """
  kern = np.zeros_like(x, dtype=x.dtype)
  _es_kernel(2 * x / alpha, kern, beta * alpha, mu)
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
  sigma=2.0,
  alpha=8,
  beta=2.3,
  mu=0.5,
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
  sigma=2.0,
  alpha=8,
  beta=2.3,
  mu=0.5,
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
    sigma=2.0,
    alpha=8,
    beta=2.3,
    mu=0.5,
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
    half_supp = alpha / 2
    betak = beta * alpha
    pos = (np.arange(alpha) - half_supp).astype(np.int64)
    xkern = np.zeros(alpha)
    ykern = np.zeros(alpha)
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
          x = (x_idx - ug) / half_supp
          # this finds the index on the fftshifted grid
          # x_idx = (x_idx + ngx//2) % ngx
          _es_kernel(x, xkern, betak, mu)
          y_idx = pos + v_idx
          y = (y_idx - vg) / half_supp
          # y_idx = (y_idx + ngy//2) % ngy
          _es_kernel(y, ykern, betak, mu)

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
  sigma=2.0,
  alpha=10,
  beta=2.3,
  mu=0.5,
  usign=1,
  vsign=-1,
):
  ntime, nbl, nchan, ncorr = data.shape
  # we only need half the
  vis_grid = np.zeros((nc, ngx, ngy), dtype=data.dtype)

  # ufreq
  u_cell = 1 / (ngx * cellx)
  umax = 1 / cellx / 2

  # vfreq
  v_cell = 1 / (ngy * celly)
  vmax = 1 / celly / 2

  normfreq = freq / LIGHTSPEED
  half_supp = alpha / 2
  betak = beta * alpha
  pos = (np.arange(alpha) - half_supp).astype(np.int64)
  xkern = np.zeros(alpha)
  ykern = np.zeros(alpha)
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
        x = (x_idx - ug) / half_supp
        # this finds the index on the fftshifted grid
        # x_idx = (x_idx + ngx//2) % ngx
        _es_kernel(x, xkern, betak, mu)
        y_idx = pos + v_idx
        y = (y_idx - vg) / half_supp
        # y_idx = (y_idx + ngy//2) % ngy
        _es_kernel(y, ykern, betak, mu)

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
  sigma=2.0,
  alpha=10,
  beta=2.3,
  mu=0.5,
):
  ngx = good_size(int(sigma * nx))
  # make sure it is even and a good size for the FFT
  while ngx % 2:
    ngx = good_size(ngx + 1)
  xcorrector = grid_corrector(ngx, alpha, beta, mu)
  padxl = (ngx - nx) // 2
  padxr = ngx - nx - padxl
  slcx = slice(padxl, -padxr)
  ngy = good_size(int(sigma * ny))
  # make sure it is even and a good size for the FFT
  while ngy % 2:
    ngy = good_size(ngy + 1)
  ycorrector = grid_corrector(ngy, alpha, beta, mu)
  padyl = (ngy - ny) // 2
  padyr = ngy - ny - padyl
  slcy = slice(padyl, -padyr)

  # taper
  corrector = xcorrector[slcx, None] * ycorrector[None, slcy]

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
    sigma=2.0,
    alpha=10,
    beta=2.3,
    mu=0.5,
    usign=1,
    vsign=1,
  )

  # now the FFTs
  # the *ngx*ngy corrects for the FFT normalisation
  dirty = np.fft.ifft2(iFs(grid), axes=(-2, -1)) * ngx * ngy
  dirty = Fs(dirty.real)[:, slcx, slcy]
  # apply taper
  dirty /= corrector
  return dirty


@njit(**JIT_OPTIONS)
def wgrid_data_np(
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
  w0: float,
  dw: float,
  nw: int,
  sigma=2.0,
  alpha=10,
  beta=2.3,
  mu=0.5,
  usign=1,
  vsign=1,
  wsign=-1,
):
  ntime, nbl, nchan, ncorr = data.shape
  # create a grid per wplane
  vis_grid = np.zeros((nc, nw, ngx, ngy), dtype=data.dtype)

  # ufreq
  u_cell = 1 / (ngx * cellx)
  umax = 1 / cellx / 2

  # vfreq
  v_cell = 1 / (ngy * celly)
  vmax = 1 / celly / 2

  # wfreq
  w_cell = dw
  wmax = w0 + (nw - 1) * dw
  wgrid = w0 + np.arange(nw) * dw

  normfreq = freq / LIGHTSPEED
  half_supp = alpha / 2
  betak = beta * alpha
  pos = (np.arange(alpha) - half_supp).astype(np.int64)
  xkern = np.zeros(alpha)
  ykern = np.zeros(alpha)
  zkern = np.zeros(alpha)
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
        w_tmp = uvw_row[1] * chan_normfreq * wsign
        # only use half the w grid due to Hermitian symmetry
        if w_tmp < 0:
          u_tmp = -u_tmp
          v_tmp = -v_tmp
          w_tmp = -w_tmp
          vis = np.conjugate(vis)

        # pixel coordinates
        ug = (u_tmp + umax) / u_cell
        vg = (v_tmp + vmax) / v_cell
        wg = (w_tmp + wmax) / w_cell

        # indices
        u_idx = int(np.round(ug))
        v_idx = int(np.round(vg))
        w_idx = int(np.round(wg))

        # the grid position is pos + u_idx on the un-fftshifted grid
        x_idx = pos + u_idx
        x = (x_idx - ug) / half_supp
        # this finds the index on the fftshifted grid
        # x_idx = (x_idx + ngx//2) % ngx
        _es_kernel(x, xkern, betak, mu)

        y_idx = pos + v_idx
        y = (y_idx - vg) / half_supp
        # y_idx = (y_idx + ngy//2) % ngy
        _es_kernel(y, ykern, betak, mu)

        w_diff = np.abs(wgrid - w_tmp)
        w_idx = np.nonzero(w_diff == w_diff.min())[0]
        z_idx = pos + w_idx
        # z = (z_idx - wg) / half_supp
        z = (wgrid - w_tmp) / dw / half_supp
        _es_kernel(z, zkern, betak, mu)

        # import ipdb; ipdb.set_trace()

        for c in range(ncorr):
          wc = wgt[c]
          for k, zk in zip(z_idx, zkern):
            for i, xi in zip(x_idx, xkern):
              for j, yj in zip(y_idx, ykern):
                xyzwc = xi * yj * zk * wc
                vis_grid[c, k, i, j] += xyzwc * vis[c]

  return vis_grid


def vis2im_wgrid(
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
  sigma=2,
  alpha=10,
  beta=2.3,
  mu=0.5,
):
  ngx = good_size(int(sigma * nx))
  # make sure it is even and a good size for the FFT
  while ngx % 2:
    ngx = good_size(ngx + 1)
  xcorrector = grid_corrector(ngx, alpha, beta, mu)
  padxl = (ngx - nx) // 2
  padxr = ngx - nx - padxl
  slcx = slice(padxl, -padxr)
  ngy = good_size(int(sigma * ny))
  while ngy % 2:
    ngy = good_size(ngy + 1)
  ycorrector = grid_corrector(ngy, alpha, beta, mu)
  padyl = (ngy - ny) // 2
  padyr = ngy - ny - padyl
  slcy = slice(padyl, -padyr)

  # xy taper
  corrector = xcorrector[slcx, None] * ycorrector[None, slcy]

  # get number of w grids
  wmin = np.abs(uvw[:, :, 2].ravel() * freq.min() / LIGHTSPEED).min()
  wmax = np.abs(uvw[:, :, 2].ravel() * freq.max() / LIGHTSPEED).max()
  x, y = np.meshgrid(*[-ss / 2 + np.arange(ss) for ss in (nx, ny)], indexing="ij")
  x *= cellx
  y *= celly
  eps = x**2 + y**2
  nm1 = -eps / (np.sqrt(1.0 - eps) + 1.0)
  # missing factor of a half compared to expression in the paper but
  # this gives the same w parameters as reported by the wgridder
  dw = 1 / (sigma * np.abs(nm1).max())
  nw = int(np.round((wmax - wmin) / dw)) + alpha
  # different from expression in paper?
  w0 = (wmin + wmax) / 2 - dw * (nw - 1) / 2
  wcorrector = grid_corrector2D(nm1 * dw, alpha, beta, mu) * (nm1 + 1)

  vis_func, wgt_func = stokes_funcs(jones, product, pol, nc)

  grid = wgrid_data_np(
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
    w0,
    dw,
    nw,
    sigma=sigma,
    alpha=alpha,
    beta=beta,
    mu=mu,
    usign=1,
    vsign=1,
    wsign=-1,
  )
  # 2D FFTs
  # the *ngx*ngy corrects for the FFT normalisation
  dirty = np.fft.ifft2(iFs(grid), axes=(-2, -1)) * ngx * ngy
  dirty = Fs(dirty)[:, :, slcx, slcy]

  # w-screens
  wgrid = w0 + np.arange(nw) * dw
  wscreens = np.exp(-2j * np.pi * nm1[None, :, :] * wgrid[:, None, None]).astype(
    data.dtype
  )
  dirty *= wscreens[None, :, :, :]
  # sum over w axis
  dirty = dirty.real.sum(axis=1)
  # xy tapers
  dirty /= corrector[None, :, :]
  # w-taper * n

  dirty /= wcorrector
  return dirty
