"""Heuristically select grid-parameters.

This module computes the oversampled grid dimensions ``(nu, nv)`` and
selects the optimal kernel from :data:`KERNEL_DB` by minimising an
estimated cost model that balances FFT and gridding work.
"""

from __future__ import annotations

import math

import numba
from numba.core import types
from numba.core.types import StructRef
from numba.experimental import structref
from numba.extending import (
  overload,
  register_jitable,
)
from numba.typed import List as TypedList

from radiomesh.errors import KernelSelectionError
from radiomesh.es_kernel_structref import ESKernelProxy
from radiomesh.generated._es_kernel_params import KERNEL_DB
from radiomesh.numba_utils import make_structref_property


def _make_parameter_list(exemplar):
  """Create a list. Overloaded to return a typed list in Numba."""
  return []


@overload(_make_parameter_list)
def overload_make_list(exemplar):
  EXEMPLAR_TYPE = exemplar
  return lambda exemplar: TypedList.empty_list(EXEMPLAR_TYPE)


@register_jitable
def _smallest_smooth_number_above(n, primes):
  """Return the smallest integer >= *n* whose prime factors are all in *primes*."""
  if n <= 1:
    return 1

  result = 2 * n
  stack = _make_parameter_list((type(n)(1), 0))
  stack.append((1, 0))

  while len(stack) > 0:
    value, idx = stack.pop()

    if value >= n:
      if value < result:
        result = value
      continue

    if idx >= len(primes):
      continue

    v = value
    while v < result:
      stack.append((v, idx + 1))
      v *= primes[idx]

    if n <= v < result:
      result = v

  return result


COMPLEX_FFT_PRIMES = (2, 3, 5, 7, 11)
REAL_FFT_PRIMES = (2, 3, 5)


@register_jitable
def optimal_complex_fft_size(n):
  """Return the smallest integer >= *n*
  whose prime factors are all in {2, 3, 5, 7, 11}."""
  return _smallest_smooth_number_above(n, COMPLEX_FFT_PRIMES)


@register_jitable
def optimal_real_fft_size(n):
  """Return the smallest integer >= *n*
  whose prime factors are all in {2, 3, 5}."""
  return _smallest_smooth_number_above(n, REAL_FFT_PRIMES)


@register_jitable
def _nm1_range(nx, ny, pixsize_x, pixsize_y, lshift, mshift):
  """Compute the range of *n - 1* over the image corners (and origin if enclosed).

  Returns ``(nm1min, nm1max)``.
  """
  xmin = lshift - 0.5 * nx * pixsize_x
  xmax = xmin + (nx - 1) * pixsize_x
  ymin = mshift - 0.5 * ny * pixsize_y
  ymax = ymin + (ny - 1) * pixsize_y

  # Up to 3 x-values, up to 3 y-values
  nx = 2
  ny = 2
  xext_0, xext_1, xext_2 = xmin, xmax, 0.0
  yext_0, yext_1, yext_2 = ymin, ymax, 0.0
  if xmin * xmax < 0:
    nx = 3
  if ymin * ymax < 0:
    ny = 3

  nm1min = math.inf
  nm1max = -math.inf
  for xi in range(nx):
    xc = xext_0 if xi == 0 else (xext_1 if xi == 1 else xext_2)
    for yi in range(ny):
      yc = yext_0 if yi == 0 else (yext_1 if yi == 1 else yext_2)
      tmp = xc * xc + yc * yc
      if tmp <= 1.0:
        nval = math.sqrt(1.0 - tmp) - 1.0
      else:
        nval = -math.sqrt(tmp - 1.0) - 1.0
      if nval < nm1min:
        nm1min = nval
      if nval > nm1max:
        nm1max = nval

  return nm1min, nm1max


@register_jitable
def sigmoid(x, m, s):
  x2 = x - 1.0
  m2 = m - 1.0
  if m2 == 0.0:
    return 1.0
  return 1.0 + x2 / (1.0 + (x2 / m2) ** s) ** (1.0 / s)


_MAX_SUPPORT = 32


@register_jitable
def _available_kernels(
  kernel_db, epsilon, ndim, single, oversampling_min, oversampling_max
):
  """Return candidate kernels — one per support width, with minimum oversampling."""
  best = _make_parameter_list((1e30, -1))
  for _ in range(_MAX_SUPPORT):
    best.append((1e30, -1))

  for i in range(len(kernel_db)):
    k = kernel_db[i]
    if (
      k.ndim == ndim
      and k.single == single
      and k.epsilon <= epsilon
      and oversampling_min <= k.oversampling <= oversampling_max
    ):
      w = k.support
      if w < _MAX_SUPPORT and k.oversampling < best[w][0]:
        best[w] = (k.oversampling, i)

  result = _make_parameter_list(kernel_db[0])
  for w in range(_MAX_SUPPORT):
    if best[w][1] >= 0:
      result.append(kernel_db[best[w][1]])

  return result


@structref.register
class WGridderParametersStructRef(StructRef):
  def preprocess_fields(self, fields):
    """Disallow literal types in field definitions"""
    return tuple((n, types.unliteral(t)) for n, t in fields)


class WGridderParameters(structref.StructRefProxy):
  def __new__(cls, nu, nv, kernel, nm1min, nm1max, nshift):
    return structref.StructRefProxy.__new__(cls, nu, nv, kernel, nm1min, nm1max, nshift)

  nu = make_structref_property("nu")
  nv = make_structref_property("nv")
  kernel = make_structref_property("kernel")
  nm1min = make_structref_property("nm1min")
  nm1max = make_structref_property("nm1max")
  nshift = make_structref_property("nshift")


structref.define_proxy(
  WGridderParameters,
  WGridderParametersStructRef,
  ["nu", "nv", "kernel", "nm1min", "nm1max", "nshift"],
)

# Reference FFT size and cost
REF_FFT_N = 2048.0
REF_FFT_COST = 0.0693
# Reference gridding cost
REF_GRIDDING_COST = 2.2e-10
# Sigmoid scaling parameters
MAX_FFT_SCALING = 6.0
SCALING_POWER = 2.0


@register_jitable
def estimate_gridding_parameters(
  nx,
  ny,
  pixsize_x,
  pixsize_y,
  epsilon,
  apply_w=False,
  single=False,
  nvis=0,
  nthreads=1,
  wmin_d=0.0,
  wmax_d=0.0,
  lshift=0.0,
  mshift=0.0,
  no_nshift=False,
  oversampling_min=1.1,
  oversampling_max=2.6,
  gridding=True,
):
  """Compute optimal grid dimensions and kernel"""
  # --- Phase 1: n-1 range and nshift ---
  nm1min, nm1max = _nm1_range(nx, ny, pixsize_x, pixsize_y, lshift, mshift)

  if no_nshift or not apply_w:
    nshift = 0.0
  else:
    nshift = -0.5 * (nm1max + nm1min)

  # --- Phase 2: enumerate candidate kernels ---
  ndim = 3 if apply_w else 2
  kernel_db = _get_kernel_db()
  candidates = _available_kernels(
    kernel_db, epsilon, ndim, single, oversampling_min, oversampling_max
  )

  if len(candidates) == 0:
    raise KernelSelectionError("No candidate kernels found")

  # Length of a simd register
  if gridding:
    # TODO: This should more correctly
    # be derived from the grid type
    vector_length = 8 if single else 4
  else:
    # TODO: This should be more correctly
    # derived from the kernel evaluation type
    vector_length = 8 if single else 4

  mincost = math.inf
  best_nu = 0
  best_nv = 0
  best_kernel = candidates[0]

  for ci in range(len(candidates)):
    kernel = candidates[ci]
    support = kernel.support
    oversampling = kernel.oversampling

    # Number of simd operations per kernel support
    vector_ops = (support + vector_length - 1) // vector_length
    # Polynomial degree
    degree = support + 3

    nu = 2 * optimal_complex_fft_size(int(nx * oversampling * 0.5) + 1)
    nv = 2 * optimal_complex_fft_size(int(ny * oversampling * 0.5) + 1)
    nu = max(nu, 16)
    nv = max(nv, 16)

    # Estimate the cost of a nu x nv FFT
    # by scaling it vs the reference FFT size and cost
    logterm = math.log(nu * nv) / math.log(REF_FFT_N * REF_FFT_N)
    fftcost = (nu / REF_FFT_N) * (nv / REF_FFT_N) * logterm * REF_FFT_COST

    gridcost = (
      REF_GRIDDING_COST
      * nvis
      * (
        # Grid accumulation cost: support x support, padded to simd boundaries
        support * vector_ops * vector_length
        # Kernel evaluation cost: support x support x degree
        + (2 * vector_ops + 1) * vector_length * degree
      )
    )

    if gridding:
      # TODO: scale the gridding cost by
      # accumulator type size / evaluation type size
      gridcost *= 1

    if apply_w:
      dw = 0.5 / oversampling / max(abs(nm1max + nshift), abs(nm1min + nshift))
      nplanes = int((wmax_d - wmin_d) / dw + support)
      fftcost *= nplanes
      gridcost *= support

    gridcost /= nthreads
    fftcost /= sigmoid(nthreads, MAX_FFT_SCALING, SCALING_POWER)

    cost = fftcost + gridcost
    if cost < mincost:
      mincost = cost
      best_nu = nu
      best_nv = nv
      best_kernel = kernel

  return WGridderParameters(
    nu=best_nu,
    nv=best_nv,
    kernel=ESKernelProxy(
      best_kernel.epsilon,
      best_kernel.oversampling,
      best_kernel.beta,
      best_kernel.e0,
      best_kernel.support,
      False,
      single,
      apply_w,
    ),
    nm1min=nm1min,
    nm1max=nm1max,
    nshift=nshift,
  )


# ---------------------------------------------------------------------------
# Numba support: KERNEL_DB access via objmode
# ---------------------------------------------------------------------------

_kernel_params_type = numba.typeof(KERNEL_DB[0])
_kernel_list_type = types.ListType(_kernel_params_type)

_CACHED_KERNEL_LIST: TypedList | None = None


def _get_kernel_list():
  global _CACHED_KERNEL_LIST
  if _CACHED_KERNEL_LIST is None:
    lst = TypedList()
    for k in KERNEL_DB:
      lst.append(k)
    _CACHED_KERNEL_LIST = lst
  return _CACHED_KERNEL_LIST


def _get_kernel_db():
  """Return KERNEL_DB. Overloaded in Numba to return a typed list via objmode."""
  return KERNEL_DB


@overload(_get_kernel_db)
def overload_get_kernel_db():
  def impl():
    with numba.objmode(kernel_list=_kernel_list_type):
      kernel_list = _get_kernel_list()
    return kernel_list

  return impl
