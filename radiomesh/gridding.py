from dataclasses import dataclass
from functools import reduce
from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt
from numba.core import types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import overload, register_jitable

from radiomesh.constants import LIGHTSPEED
from radiomesh.es_kernel import ESKernel, es_kernel_positions, eval_es_kernel
from radiomesh.intrinsics import (
  accumulate_data,
  apply_weights,
  check_args,
  load_data,
)
from radiomesh.jones_intrinsics import ApplyJones, maybe_apply_jones
from radiomesh.literals import Datum, DatumLiteral, Schema
from radiomesh.utils import wgridder_conventions

JIT_OPTIONS = {"parallel": False, "nogil": True, "cache": False, "fastmath": True}


@register_jitable
def maybe_conjugate(u, v, w, vis):
  if w < 0.0:
    u = -u
    v = -v
    w = -w

    for v, value in enumerate(numba.literal_unroll(vis)):
      vis = tuple_setitem(vis, v, np.conj(v))

  return u, v, w, vis


@dataclass(slots=True, eq=True, unsafe_hash=True)
class WGridderParameters:
  nx: int
  ny: int
  nw: int
  pixsizex: float
  pixsizey: float
  w0: float
  dw: float
  kernel: float | ESKernel
  pol_schema: Tuple[str, ...]
  stokes_schema: Tuple[str, ...]
  apply_w: bool = True
  apply_fftshift: bool = True

  def __post_init__(self):
    if isinstance(self.kernel, float):
      self.kernel = ESKernel(epsilon=self.kernel, apply_w=self.apply_w)


def wgrid_impl(
  uvw,
  visibilities,
  weights,
  flags,
  frequencies,
  wgrid_literal_params,
  jones_params,
):
  pass


any_flagged = numba.njit(**JIT_OPTIONS)(lambda a, f: a or f != 0)


@overload(wgrid_impl, jit_options=JIT_OPTIONS, prefer_literal=True)
def wgrid_overload(
  uvw,
  visibilities,
  weights,
  flags,
  frequencies,
  wgrid_literal_params,
  jones_params,
):
  if not isinstance(uvw, types.Array) or not isinstance(uvw.dtype, types.Float):
    raise TypingError(f"'uvw' {uvw} must be a Float Array")

  if not isinstance(visibilities, types.Array) or not isinstance(
    visibilities.dtype, types.Complex
  ):
    raise TypingError(f"'visibilities' {visibilities} must be a Complex Array")

  if not isinstance(frequencies, types.Array) or not isinstance(
    frequencies.dtype, types.Float
  ):
    raise TypingError(f"'frequencies' {frequencies} must be a Float Array")

  if not isinstance(weights, types.Array) or not isinstance(weights.dtype, types.Float):
    raise TypingError(f"'weights' {weights} must be a Float Array")

  if not isinstance(flags, types.Array) or not isinstance(
    flags.dtype, (types.Integer, types.Boolean)
  ):
    raise TypingError(f"'flags' {flags} must be a Integer or Boolean Array")

  if not isinstance(wgrid_literal_params, DatumLiteral) or not isinstance(
    (wgrid_params := wgrid_literal_params.datum_value), WGridderParameters
  ):
    raise RequireLiteralValue(
      f"'wgrid_literal_params' {wgrid_literal_params} "
      f"is not a DatumLiteral[WGridderParameters]"
    )

  # Currently this implementation only applies an fftshift
  if not (apply_fftshift := wgrid_params.apply_fftshift):
    raise NotImplementedError(f"wgrid_params.apply_fftshift={apply_fftshift}")

  KERNEL = Datum(wgrid_params.kernel)
  HALF_SUPPORT_INT = wgrid_params.kernel.half_support_int
  POL_SCHEMA_DATUM = Datum(wgrid_params.pol_schema)
  STOKES_SCHEMA_DATUM = Datum(wgrid_params.stokes_schema)
  NSTOKES = len(STOKES_SCHEMA_DATUM.value)
  NPOL = len(POL_SCHEMA_DATUM.value)
  NUVW = len(["U", "V", "W"])

  NX = wgrid_params.nx
  NY = wgrid_params.ny
  NW = wgrid_params.nw
  PIXSIZEX = wgrid_params.pixsizex
  PIXSIZEY = wgrid_params.pixsizey
  W0 = wgrid_params.w0
  DW = wgrid_params.dw
  U_SIGN, V_SIGN, _, _, _ = wgridder_conventions(0.0, 0.0)

  JONES_VIS_DATUM = Datum(
    ApplyJones("vis", POL_SCHEMA_DATUM.value, STOKES_SCHEMA_DATUM.value)
  )
  JONES_WGT_DATUM = Datum(
    ApplyJones("weight", POL_SCHEMA_DATUM.value, STOKES_SCHEMA_DATUM.value)
  )

  def impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    wgrid_literal_params,
    jones_params,
  ):
    check_args(uvw, visibilities, weights, flags, frequencies, NPOL)
    ntime, nbl, nchan, _ = visibilities.shape

    wavelengths = frequencies / LIGHTSPEED

    vis_grid = np.zeros((NSTOKES, NW, NX, NY), visibilities.dtype)
    # weight_grid = np.zeros((NSTOKES, NW, NX, NY), weights.dtype)

    vis_grid_view = vis_grid[:]
    # weight_grid_view = weight_grid[:]

    for t in numba.prange(ntime):
      for bl in range(nbl):
        u, v, w = load_data(uvw, (t, bl), NUVW, -1)
        for ch in range(nchan):
          idx = (t, bl, ch)  # Indexing tuple for use in intrinsics
          # Return early if any visibility is flagged
          vis_flag = load_data(flags, idx, NPOL, -1)
          if reduce(any_flagged, vis_flag, False):
            continue

          vis = load_data(visibilities, idx, NPOL, -1)
          wgt = load_data(weights, idx, NPOL, -1)
          vis = maybe_apply_jones(JONES_VIS_DATUM, jones_params, vis, idx)
          wgt = maybe_apply_jones(JONES_WGT_DATUM, jones_params, wgt, idx)
          vis = apply_weights(vis, wgt)

          # Scaled uv coordinates
          u_scaled = u * U_SIGN * wavelengths[ch] * PIXSIZEX
          v_scaled = v * V_SIGN * wavelengths[ch] * PIXSIZEY
          w_scaled = w * wavelengths[ch]

          # Use only half the w grid due to Hermitian symmetry
          u_scaled, v_scaled, w_scaled, vis = maybe_conjugate(
            u_scaled, v_scaled, w_scaled, vis
          )

          # UV coordinates on the FFT shifted grid
          u_grid = (u_scaled * NX) % NX
          v_grid = (v_scaled * NY) % NY
          w_grid = (w_scaled - W0) / DW

          # Pixel indices at the start of the kernel
          u_pixel_start = int(np.round(u_grid)) - HALF_SUPPORT_INT
          v_pixel_start = int(np.round(v_grid)) - HALF_SUPPORT_INT
          w_pixel_start = int(np.round(w_grid)) - HALF_SUPPORT_INT

          # Tuple of indices associated with each kernel value in X, Y and Z
          # Of length kernel.support
          x_indices = es_kernel_positions(KERNEL, NX, u_pixel_start)
          y_indices = es_kernel_positions(KERNEL, NY, v_pixel_start)
          z_indices = es_kernel_positions(KERNEL, NW, w_pixel_start)

          # Tuples of kernel values of length kernel.support
          x_kernel = eval_es_kernel(KERNEL, u_grid, u_pixel_start)
          y_kernel = eval_es_kernel(KERNEL, v_grid, v_pixel_start)
          z_kernel = eval_es_kernel(KERNEL, w_grid, w_pixel_start)

          for zfi, zk in zip(
            numba.literal_unroll(z_indices), numba.literal_unroll(z_kernel)
          ):
            zi = int(zfi)
            for xfi, xk in zip(
              numba.literal_unroll(x_indices), numba.literal_unroll(x_kernel)
            ):
              xi = int(xfi)
              for yfi, yk in zip(
                numba.literal_unroll(y_indices), numba.literal_unroll(y_kernel)
              ):
                pol_weight = xk * yk
                yi = int(yfi)
                weighted_stokes = apply_weights(vis, pol_weight)
                accumulate_data(weighted_stokes, vis_grid_view, (zi, xi, yi), 0)
                # weighted_weights = apply_weights(wgt, pol_weight)
                # accumulate_data(weighted_weights, weight_grid_view, (xi, yi), 0)

    return vis_grid  # , weight_grid

  return impl


@numba.njit(**{**JIT_OPTIONS, "parallel": False})
def wgrid(
  uvw: npt.NDArray[np.floating],
  visibilities: npt.NDArray[np.complexfloating],
  weights: npt.NDArray[np.floating],
  flags: npt.NDArray[np.integer],
  frequencies: npt.NDArray[np.floating],
  wgrid_literal_params: DatumLiteral[WGridderParameters],
  jones_params: Tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.integer], Schema]
  | None = None,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
  return wgrid_impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    numba.literally(wgrid_literal_params),
    jones_params,
  )
