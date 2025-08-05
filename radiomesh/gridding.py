from dataclasses import dataclass
from functools import reduce
from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt
from numba.core import types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.extending import overload

from radiomesh.constants import LIGHTSPEED
from radiomesh.es_kernel import ESKernel, es_kernel_positions, eval_es_kernel
from radiomesh.intrinsics import (
  accumulate_data,
  apply_flags,
  apply_weights,
  check_args,
  load_data,
  pol_to_stokes,
)
from radiomesh.literals import Datum, DatumLiteral
from radiomesh.utils import wgridder_conventions

JIT_OPTIONS = {"parallel": False, "nogil": True, "cache": True, "fastmath": True}


@dataclass(slots=True, eq=True, unsafe_hash=True)
class WGridderParameters:
  nx: int
  ny: int
  pixsizex: float
  pixsizey: float
  kernel: ESKernel
  schema: str
  apply_w: bool = True
  apply_fftshift: bool = True


def parse_schema(schema: str) -> Tuple[str, ...]:
  return tuple(s.strip().upper() for s in schema.lstrip("[ ").rstrip("] ").split(","))


def wgrid_impl(
  uvw,
  visibilities,
  weights,
  flags,
  frequencies,
  wgrid_literal_params,
):
  pass


flag_reduce = numba.njit(**JIT_OPTIONS)(lambda a, f: a or f != 0)


@overload(wgrid_impl, jit_options=JIT_OPTIONS, prefer_literal=True)
def wgrid_overload(
  uvw,
  visibilities,
  weights,
  flags,
  frequencies,
  wgrid_literal_params,
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

  if not isinstance(wgrid_literal_params, DatumLiteral):
    raise RequireLiteralValue(
      f"'wgrid_literal_params' {wgrid_literal_params} is not a DatumLiteral"
    )

  if not isinstance(
    (wgrid_params := wgrid_literal_params.datum_value), WGridderParameters
  ):
    raise TypingError(
      f"'wgrid_literal_params' {type(wgrid_params)} must be a WGridderParameters "
    )

  try:
    pol_str, stokes_str = wgrid_params.schema.split("->")
  except ValueError as e:
    raise ValueError(
      f"{wgrid_params.schema} should be of the form " f"[XX,XY,YX,YY] -> [I,Q,U,V]"
    ) from e

  # Currently this implementation only applies an fftshift
  if not (apply_fftshift := wgrid_params.apply_fftshift):
    raise NotImplementedError(f"wgrid_params.apply_fftshift={apply_fftshift}")

  KERNEL = Datum(wgrid_params.kernel)
  HALF_SUPPORT_INT = wgrid_params.kernel.half_support_int
  POL_SCHEMA_DATUM = Datum(parse_schema(pol_str))
  STOKES_SCHEMA_DATUM = Datum(parse_schema(stokes_str))
  NSTOKES = len(STOKES_SCHEMA_DATUM.value)
  NPOL = len(POL_SCHEMA_DATUM.value)
  NUVW = len(["U", "V", "W"])

  NX = wgrid_params.nx
  NY = wgrid_params.ny
  PIXSIZEX = wgrid_params.pixsizex
  PIXSIZEY = wgrid_params.pixsizey
  U_SIGN, V_SIGN, _, _, _ = wgridder_conventions(0.0, 0.0)

  def impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    wgrid_literal_params,
  ):
    check_args(uvw, visibilities, weights, flags, frequencies, NPOL)
    ntime, nbl, nchan, _ = visibilities.shape

    wavelengths = frequencies / LIGHTSPEED

    vis_grid = np.zeros((NSTOKES, NX, NY), visibilities.real.dtype)
    weight_grid = np.zeros((NX, NY), weights.dtype)

    vis_grid_view = vis_grid[:]
    weight_grid_view = weight_grid[:]

    for t in numba.prange(ntime):
      for bl in range(nbl):
        u, v, w = load_data(uvw, (t, bl), NUVW, -1)
        for ch in range(nchan):
          # Return early if entire visibility is flagged
          vis_flag = load_data(flags, (t, bl, ch), NPOL, -1)
          if reduce(flag_reduce, vis_flag, False):
            continue

          vis = load_data(visibilities, (t, bl, ch), NPOL, -1)
          wgt = load_data(weights, (t, bl, ch), NPOL, -1)
          wgt = apply_flags(wgt, vis_flag)
          vis = apply_weights(vis, wgt)
          stokes = pol_to_stokes(vis, POL_SCHEMA_DATUM, STOKES_SCHEMA_DATUM)

          # Scaled uv coordinates
          u_scaled = u * U_SIGN * wavelengths[ch] * PIXSIZEX
          v_scaled = v * V_SIGN * wavelengths[ch] * PIXSIZEY

          # UV coordinates on the FFT shifted grid
          u_grid = (u_scaled * NX) % NX
          v_grid = (v_scaled * NY) % NY

          # Pixel indices at the start of the kernel
          u_pixel_start = int(np.round(u_grid)) - HALF_SUPPORT_INT
          v_pixel_start = int(np.round(v_grid)) - HALF_SUPPORT_INT

          # Tuple of indices associated with each kernel value in X and Y
          # Of length kernel.support
          x_indices = es_kernel_positions(KERNEL, NX, u_pixel_start)
          y_indices = es_kernel_positions(KERNEL, NY, v_pixel_start)

          # Tuples of kernel values of length kernel.support
          x_kernel = eval_es_kernel(KERNEL, u_grid, u_pixel_start)
          y_kernel = eval_es_kernel(KERNEL, v_grid, v_pixel_start)

          for xfi, xk in zip(
            numba.literal_unroll(x_indices), numba.literal_unroll(x_kernel)
          ):
            xi = int(xfi)
            for yfi, yk in zip(
              numba.literal_unroll(y_indices), numba.literal_unroll(y_kernel)
            ):
              pol_weight = xk * yk
              yi = int(yfi)
              weighted_stokes = apply_weights(stokes, pol_weight)
              accumulate_data(weighted_stokes, vis_grid_view, (xi, yi), NSTOKES, 0)
              weight_grid_view[xi, yi] += pol_weight

    return vis_grid, weight_grid

  return impl


@numba.njit(**{**JIT_OPTIONS, "parallel": False})
def wgrid(
  uvw: npt.NDArray[np.floating],
  visibilities: npt.NDArray[np.complexfloating],
  weights: npt.NDArray[np.floating],
  flags: npt.NDArray[np.integer],
  frequencies: npt.NDArray[np.floating],
  wgrid_literal_params: DatumLiteral[WGridderParameters],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
  return wgrid_impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    numba.literally(wgrid_literal_params),
  )
