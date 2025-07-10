from functools import reduce
from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt
from numba.core import types
from numba.core.errors import RequireLiteralValue
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
from radiomesh.literals import Datum
from radiomesh.utils import wgridder_conventions

JIT_OPTIONS = {"parallel": False, "nogil": True, "cache": True, "fastmath": True}


def parse_schema(schema: str) -> Tuple[str, ...]:
  return tuple(s.strip().upper() for s in schema.lstrip("[ ").rstrip("] ").split(","))


def wgrid_impl(
  uvw,
  visibilities,
  weights,
  flags,
  frequencies,
  nx,
  ny,
  pixsizex,
  pixsizey,
  epsilon,
  schema,
):
  pass


flag_reduce = numba.njit(**JIT_OPTIONS)(lambda a, f: a and f != 0)


@overload(wgrid_impl, jit_options=JIT_OPTIONS)
def wgrid_overload(
  uvw,
  visibilities,
  weights,
  flags,
  frequencies,
  nx,
  ny,
  pixsizex,
  pixsizey,
  epsilon,
  schema,
):
  if not isinstance(uvw, types.Array) or not isinstance(uvw.dtype, types.Float):
    raise TypeError(f"'uvw' {uvw} must be a Float Array")

  if not isinstance(visibilities, types.Array) or not isinstance(
    visibilities.dtype, types.Complex
  ):
    raise TypeError(f"'visibilities' {visibilities} must be a Complex Array")

  if not isinstance(frequencies, types.Array) or not isinstance(
    frequencies.dtype, types.Float
  ):
    raise TypeError(f"'frequencies' {frequencies} must be a Float Array")

  if not isinstance(weights, types.Array) or not isinstance(weights.dtype, types.Float):
    raise TypeError(f"'weights' {weights} must be a Float Array")

  if not isinstance(flags, types.Array) or not isinstance(
    flags.dtype, (types.Integer, types.Boolean)
  ):
    raise TypeError(f"'flags' {flags} must be a Integer or Boolean Array")

  if not isinstance(schema, types.StringLiteral):
    raise RequireLiteralValue(f"'schema' {schema} is not a string literal")

  if not isinstance(nx, types.IntegerLiteral):
    raise RequireLiteralValue(f"'nx' {nx} is not a int literal")

  if not isinstance(ny, types.IntegerLiteral):
    raise RequireLiteralValue(f"'ny' {ny} is not a int literal")

  if not isinstance(pixsizex, types.StringLiteral):
    raise RequireLiteralValue(f"'pixsizex' {pixsizex} is not a string literal")

  if not isinstance(pixsizey, types.StringLiteral):
    raise RequireLiteralValue(f"'pixsizey' {pixsizey} is not a string literal")

  if not isinstance(epsilon, types.StringLiteral):
    raise RequireLiteralValue(f"'epsilon' {epsilon} is not a string literal")

  try:
    pol_str, stokes_str = schema.literal_value.split("->")
  except ValueError as e:
    raise ValueError(
      f"{schema} should be of the form " f"[XX,XY,YX,YY] -> [I,Q,U,V]"
    ) from e

  KERNEL = Datum(ESKernel(float(epsilon.literal_value)))
  POL_SCHEMA_DATUM = Datum(parse_schema(pol_str))
  STOKES_SCHEMA_DATUM = Datum(parse_schema(stokes_str))
  NSTOKES = len(STOKES_SCHEMA_DATUM.value)
  NPOL = len(POL_SCHEMA_DATUM.value)
  NUVW = len(["U", "V", "W"])

  NX = nx.literal_value
  NY = ny.literal_value
  PIXSIZEX = float(pixsizex.literal_value)
  PIXSIZEY = float(pixsizey.literal_value)
  U_CELL = 1.0 / (NX * PIXSIZEX)
  V_CELL = 1.0 / (NY * PIXSIZEY)
  U_MAX = 1.0 / PIXSIZEX / 2.0
  V_MAX = 1.0 / PIXSIZEY / 2.0
  U_SIGN, V_SIGN, _, _, _ = wgridder_conventions(0.0, 0.0)

  def impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    nx,
    ny,
    pixsizex,
    pixsizey,
    epsilon,
    schema,
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
          if reduce(flag_reduce, vis_flag, True):
            continue

          vis = load_data(visibilities, (t, bl, ch), NPOL, -1)
          wgt = load_data(weights, (t, bl, ch), NPOL, -1)
          wgt = apply_flags(wgt, vis_flag)
          vis = apply_weights(vis, wgt)
          stokes = pol_to_stokes(vis, POL_SCHEMA_DATUM, STOKES_SCHEMA_DATUM)

          # Pixel coordinates
          u_pixel = (U_SIGN * u * wavelengths[ch] + U_MAX) / U_CELL
          v_pixel = (V_SIGN * v * wavelengths[ch] + V_MAX) / V_CELL

          # Indices
          u_index = int(np.round(u_pixel))
          v_index = int(np.round(v_pixel))

          # Tuples of indices of length kernel.support
          # centred on {u,v}_index
          x_indices = es_kernel_positions(KERNEL, u_index)
          y_indices = es_kernel_positions(KERNEL, v_index)

          # Tuples of kernel values of length kernel.support
          # generated from (index - pixel)
          x_kernel = eval_es_kernel(KERNEL, x_indices, u_pixel)
          y_kernel = eval_es_kernel(KERNEL, y_indices, v_pixel)

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
  nx: int,
  ny: int,
  pixsizex: float,
  pixsizey: float,
  epsilon: float,
  schema: str,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
  return wgrid_impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    numba.literally(nx),
    numba.literally(ny),
    numba.literally(pixsizex),
    numba.literally(pixsizey),
    numba.literally(epsilon),
    numba.literally(schema),
  )
