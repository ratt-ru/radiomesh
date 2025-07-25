from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt
from ducc0.fft import good_size
from numba import njit

from radiomesh.constants import LIGHTSPEED
from radiomesh.stokes import stokes_funcs

fftshift = partial(np.fft.fftshift, axes=(-2, -1))
ifftshift = partial(np.fft.ifftshift, axes=(-2, -1))

JIT_OPTIONS = {"nogil": True, "cache": True, "error_model": "numpy", "fastmath": False}


def grid_corrector(domain, alpha, beta, mu):
  """
  Use Gauss-Legendre quadrature to compute the grid corrector (taper).
  """
  # even number of roots required to exploit even symmetry of integrand
  nroots = 2 * alpha
  if nroots % 2:
    nroots += 1
  q, wgt = np.polynomial.legendre.leggauss(nroots)
  idx = q > 0
  q = q[idx]
  wgt = wgt[idx]
  shape = np.shape(domain)
  xkern = np.zeros(q.size)
  _es_kernel(q, xkern, alpha * beta, mu)
  if len(shape) > 1:
    z = np.einsum("ij,k->ijk", domain, q)
  else:
    z = np.outer(domain, q)
  # note the ndim dependent broadcast
  tmp = alpha * np.sum(np.cos(np.pi * alpha * z) * xkern * wgt, axis=-1)
  return np.reshape(tmp, shape)


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
  Unscaled version of the kernel

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
  nc: int,
  vis_func: Callable,
  wgt_func: Callable,
  alpha: float = 10,
  beta: float = 2.3,
  mu: float = 0.5,
  usign: float = 1,
  vsign: float = -1,
):
  ntime, nbl, nchan, ncorr = data.shape
  vis_grid = np.zeros((nc, ngx, ngy), dtype=data.dtype)
  normfreq = freq / LIGHTSPEED
  half_supp = alpha / 2
  betak = beta * alpha
  pos = np.arange(alpha)
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
        u_tmp = uvw_row[0] * chan_normfreq * usign * cellx
        v_tmp = uvw_row[1] * chan_normfreq * vsign * celly

        # uv coordinates on the FFT shifted grid
        ug = (ngx * u_tmp) % ngx
        vg = (ngy * v_tmp) % ngy

        # start indices
        xle = int(np.round(ug)) - alpha // 2
        yle = int(np.round(vg)) - alpha // 2

        # the grid indices on fftshifted grid
        x_idx = (xle + pos) % ngx
        # kernel coordinates
        x = (pos - ug + xle) / half_supp
        _es_kernel(x, xkern, betak, mu)

        y_idx = (yle + pos) % ngy
        y = (pos - vg + yle) / half_supp
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
  sigma: float = 2.0,
  alpha: float = 10,
  beta: float = 2.3,
  mu: float = 0.5,
  usign: float = 1,
  vsign: float = 1,
):
  ngx = good_size(int(sigma * nx))
  # make sure it is even and a good size for the FFT
  while ngx % 2:
    ngx = good_size(ngx + 1)
  dom = np.linspace(-0.5, 0.5, ngx, endpoint=False)
  xcorrector = grid_corrector(dom, alpha, beta, mu)
  padxl = (ngx - nx) // 2
  padxr = ngx - nx - padxl
  slcx = slice(padxl, -padxr)
  ngy = good_size(int(sigma * ny))
  while ngy % 2:
    ngy = good_size(ngy + 1)
  dom = np.linspace(-0.5, 0.5, ngy, endpoint=False)
  ycorrector = grid_corrector(dom, alpha, beta, mu)
  padyl = (ngy - ny) // 2
  padyr = ngy - ny - padyl
  slcy = slice(padyl, -padyr)

  # taper
  corrector = xcorrector[slcx, None] * ycorrector[None, slcy]

  vis_func, wgt_func = stokes_funcs(jones, product, pol, nc)

  grid = grid_data(
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
    nc,
    vis_func,
    wgt_func,
    alpha=alpha,
    beta=beta,
    mu=mu,
    usign=usign,
    vsign=vsign,
  )

  # the *ngx*ngy corrects for the FFT normalisation
  dirty = np.fft.ifft2(grid, axes=(-2, -1)) * ngx * ngy
  dirty = fftshift(dirty.real)[:, slcx, slcy]
  # apply taper
  dirty /= corrector
  return dirty


@njit(**JIT_OPTIONS)
def wgrid_data(
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
  nc: int,
  vis_func: Callable,
  wgt_func: Callable,
  w0: float,
  dw: float,
  nw: int,
  alpha: float = 5,
  beta: float = 2.3,
  mu: float = 0.5,
  usign: float = 1,
  vsign: float = 1,
):
  ntime, nbl, nchan, ncorr = data.shape
  # create a grid per wplane
  vis_grid = np.zeros((nc, nw, ngx, ngy), dtype=data.dtype)
  normfreq = freq / LIGHTSPEED
  half_supp = alpha / 2
  betak = beta * alpha
  pos = np.arange(alpha).astype(np.int64)
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
        u_tmp = uvw_row[0] * chan_normfreq * usign * cellx
        v_tmp = uvw_row[1] * chan_normfreq * vsign * celly
        w_tmp = uvw_row[2] * chan_normfreq
        # only use half the w grid due to Hermitian symmetry
        if w_tmp < 0:
          u_tmp = -u_tmp
          v_tmp = -v_tmp
          w_tmp = -w_tmp
          vis = np.conjugate(vis)

        # uv coordinates on the FFT shifted grid
        ug = (ngx * u_tmp) % ngx
        vg = (ngy * v_tmp) % ngy

        # w coordinate on the grid
        wg = (w_tmp - w0) / dw

        # start indices
        xle = int(np.round(ug)) - alpha // 2
        yle = int(np.round(vg)) - alpha // 2
        zle = int(np.round(wg)) - alpha // 2

        # the grid indices on fftshifted grid
        x_idx = (xle + pos) % ngx
        # kernel coordinates
        x = (pos - ug + xle) / half_supp
        _es_kernel(x, xkern, betak, mu)

        y_idx = (yle + pos) % ngy
        y = (pos - vg + yle) / half_supp
        _es_kernel(y, ykern, betak, mu)

        z_idx = zle + pos
        z = (pos - wg + zle) / half_supp
        # This is the same as
        # z_idx = np.arange(nw)
        # z = (wgrid - w_tmp) / dw / half_supp
        # but only within the support of the kernel
        _es_kernel(z, zkern, betak, mu)

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
  sigma: float = 2,
  alpha: float = 10,
  beta: float = 2.3,
  mu: float = 0.5,
  usign: float = 1,
  vsign: float = 1,
):
  # ngx = int(sigma * nx)
  ngx = good_size(int(sigma * nx))
  # make sure it is even and a good size for the FFT
  while ngx % 2:
    ngx = good_size(ngx + 1)
  dom = np.linspace(-0.5, 0.5, ngx, endpoint=False)
  xcorrector = grid_corrector(dom, alpha, beta, mu)
  padxl = (ngx - nx) // 2
  padxr = ngx - nx - padxl
  slcx = slice(padxl, -padxr)
  # ngy = int(sigma * ny)
  ngy = good_size(int(sigma * ny))
  while ngy % 2:
    ngy = good_size(ngy + 1)
  dom = np.linspace(-0.5, 0.5, ngy, endpoint=False)
  ycorrector = grid_corrector(dom, alpha, beta, mu)
  padyl = (ngy - ny) // 2
  padyr = ngy - ny - padyl
  slcy = slice(padyl, -padyr)

  # xy taper
  corrector = xcorrector[slcx, None] * ycorrector[None, slcy]

  # get number of w grids
  wmax = np.abs(uvw[:, :, 2].ravel() * freq.max() / LIGHTSPEED).max()
  wmin = np.abs(uvw[:, :, 2].ravel() * freq.min() / LIGHTSPEED).min()
  x, y = np.meshgrid(*[-ss / 2 + np.arange(ss) for ss in (nx, ny)], indexing="ij")
  x *= cellx
  y *= celly
  eps = x**2 + y**2
  nm1 = -eps / (np.sqrt(1.0 - eps) + 1.0)
  # removing the factor of a half compared to expression in the
  # paper gives the same w parameters as reported by the wgridder
  # but I can't seem to get that to agree with the DFT
  dw = 1.0 / (2 * sigma * np.abs(nm1).max())
  nw = int(np.ceil((wmax - wmin) / dw)) + alpha
  w0 = (wmin + wmax) / 2 - dw * (nw - 1) / 2
  wcorrector = grid_corrector(nm1 * dw, alpha, beta, mu) * (nm1 + 1)

  # this should be compiled in the overload
  vis_func, wgt_func = stokes_funcs(jones, product, pol, nc)

  grid = wgrid_data(
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
    nc,
    vis_func,
    wgt_func,
    w0,
    dw,
    nw,
    alpha=alpha,
    beta=beta,
    mu=mu,
    usign=usign,
    vsign=vsign,
  )
  # the *ngx*ngy corrects for the FFT normalisation
  dirty = np.fft.ifft2(grid, axes=(-2, -1)) * ngx * ngy
  dirty = fftshift(dirty)[:, :, slcx, slcy]

  # w-screens
  wgrid = w0 + np.arange(nw) * dw
  wscreens = np.exp(-2j * np.pi * nm1[None, :, :] * wgrid[:, None, None]).astype(
    data.dtype
  )
  dirty = dirty * wscreens[None, :, :, :]
  # sum over w axis
  dirty = dirty.real.sum(axis=1)
  # xy tapers
  dirty /= corrector[None, :, :]
  # w-taper * n
  dirty /= wcorrector[None, :, :]
  return dirty
