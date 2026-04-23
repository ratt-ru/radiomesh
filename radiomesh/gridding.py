from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt
from numba import literal_unroll
from numba.core import types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import overload, register_jitable

from radiomesh.constants import LIGHTSPEED
from radiomesh.intrinsics import (
  accumulate_data,
  apply_weights,
  check_args,
  load_data,
)
from radiomesh.jones_intrinsics import ApplyJones, maybe_apply_jones, ndirections
from radiomesh.literals import Datum, DatumLiteral, Schema, SchemaLiteral
from radiomesh.parameters import WGridderParametersStructRef
from radiomesh.utils import wgridder_conventions

JIT_OPTIONS = {"parallel": False, "nogil": True, "cache": False, "fastmath": True}


@register_jitable
def maybe_conjugate(u, v, w, vis):
  """Invert uvw and conjugate visibilities if w < 0.0"""
  out_vis = vis

  if w < 0.0:
    u = -u
    v = -v
    w = -w

    for i, value in enumerate(literal_unroll(vis)):
      out_vis = tuple_setitem(out_vis, i, np.conj(value))

  return u, v, w, out_vis


@register_jitable
def any_flagged(flags):
  for flag in literal_unroll(flags):
    if flag != 0:
      return True

  return False


def wgrid_impl(
  uvw,
  visibilities,
  weights,
  flags,
  frequencies,
  wgrid_params,
  nx,
  ny,
  pixsizex,
  pixsizey,
  pol_schema,
  stokes_schema,
  apply_w,
  jones_params,
):
  pass


@overload(wgrid_impl, jit_options=JIT_OPTIONS, prefer_literal=True)
def wgrid_overload(
  uvw,
  visibilities,
  weights,
  flags,
  frequencies,
  wgrid_params,
  nx,
  ny,
  pixsizex,
  pixsizey,
  pol_schema,
  stokes_schema,
  apply_w,
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

  if not isinstance(wgrid_params, WGridderParametersStructRef):
    raise TypingError(
      f"'wgrid_params' {wgrid_params} must be a WGridderParameters StructRef"
    )

  if not isinstance(pol_schema, SchemaLiteral):
    raise RequireLiteralValue(f"'pol_schema' {pol_schema} must be a Schema literal")

  if not isinstance(stokes_schema, SchemaLiteral):
    raise RequireLiteralValue(
      f"'stokes_schema' {stokes_schema} must be a Schema literal"
    )

  if not isinstance(apply_w, DatumLiteral) or not isinstance(
    apply_w.literal_value, bool
  ):
    raise RequireLiteralValue(f"'apply_w' {apply_w} must be a Datum[bool]")

  POL_SCHEMA = pol_schema.literal_value
  STOKES_SCHEMA = stokes_schema.literal_value
  APPLY_W = apply_w.literal_value
  NSTOKES = len(STOKES_SCHEMA)
  NPOL = len(POL_SCHEMA)
  NUVW = len(["U", "V", "W"])

  U_SIGN, V_SIGN, W_SIGN, _, _ = wgridder_conventions(0.0, 0.0)

  JONES_VIS = Datum(ApplyJones("vis", POL_SCHEMA, STOKES_SCHEMA))
  JONES_WGT = Datum(ApplyJones("weight", POL_SCHEMA, STOKES_SCHEMA))

  def no_w_impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    wgrid_params,
    nx,
    ny,
    pixsizex,
    pixsizey,
    pol_schema,
    stokes_schema,
    apply_w,
    jones_params,
  ):
    check_args(uvw, visibilities, weights, flags, frequencies, NPOL)
    ntime, nbl, nchan, _ = visibilities.shape
    ndir = ndirections(jones_params)

    wavelengths = frequencies / LIGHTSPEED

    kernel = wgrid_params.kernel
    support = kernel.support
    half_support_int = support // 2
    x_taps = kernel.allocate_taps()
    y_taps = kernel.allocate_taps()

    vis_grid = np.zeros((NSTOKES, ndir, nx, ny), visibilities.dtype)

    for t in range(ntime):
      for bl in range(nbl):
        u, v, w = load_data(uvw, (t, bl), NUVW, -1)
        for ch in range(nchan):
          idx = (t, bl, ch)
          if any_flagged(load_data(flags, idx, NPOL, -1)):
            continue

          vis = load_data(visibilities, idx, NPOL, -1)
          wgt = load_data(weights, idx, NPOL, -1)

          wavelength = wavelengths[ch]
          u_scaled, v_scaled, _, vis = maybe_conjugate(
            U_SIGN * u * wavelength * pixsizex,
            V_SIGN * v * wavelength * pixsizey,
            W_SIGN * w * wavelength,
            vis,
          )

          u_grid = (u_scaled * nx) % nx
          v_grid = (v_scaled * ny) % ny

          u_pixel_start = int(np.round(u_grid)) - half_support_int
          v_pixel_start = int(np.round(v_grid)) - half_support_int

          kernel.evaluate_support(u_grid, u_pixel_start, x_taps)
          kernel.evaluate_support(v_grid, v_pixel_start, y_taps)

          for d in range(ndir):
            didx = idx + (d,)
            dir_vis = maybe_apply_jones(JONES_VIS, jones_params, vis, didx)
            dir_weight = maybe_apply_jones(JONES_WGT, jones_params, wgt, didx)
            dir_vis = apply_weights(dir_vis, dir_weight)

            for xo in range(support):
              xi = (u_pixel_start + xo) % nx
              xkw = x_taps[xo]
              for yo in range(support):
                yi = (v_pixel_start + yo) % ny
                ykw = y_taps[yo]
                weighted_stokes = apply_weights(dir_vis, xkw * ykw)
                accumulate_data(weighted_stokes, vis_grid, (d, xi, yi), 0)

    return vis_grid

  def apply_w_impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    wgrid_params,
    nx,
    ny,
    pixsizex,
    pixsizey,
    pol_schema,
    stokes_schema,
    apply_w,
    jones_params,
  ):
    check_args(uvw, visibilities, weights, flags, frequencies, NPOL)
    ntime, nbl, nchan, _ = visibilities.shape
    ndir = ndirections(jones_params)

    wavelengths = frequencies / LIGHTSPEED

    kernel = wgrid_params.kernel
    support = kernel.support
    half_support_int = support // 2
    x_taps = kernel.allocate_taps()
    y_taps = kernel.allocate_taps()
    z_taps = kernel.allocate_taps()

    nw = wgrid_params.nw
    # wmin lands at grid coord support/2; wmax at nw - support/2 (float).
    dw = (wgrid_params.wmax - wgrid_params.wmin) / (nw - support)
    w0 = wgrid_params.wmin - dw * (support / 2.0)

    vis_grid = np.zeros((NSTOKES, ndir, nw, nx, ny), visibilities.dtype)

    for t in range(ntime):
      for bl in range(nbl):
        u, v, w = load_data(uvw, (t, bl), NUVW, -1)
        for ch in range(nchan):
          idx = (t, bl, ch)
          if any_flagged(load_data(flags, idx, NPOL, -1)):
            continue

          vis = load_data(visibilities, idx, NPOL, -1)
          wgt = load_data(weights, idx, NPOL, -1)

          wavelength = wavelengths[ch]

          # Half w-grid only; Hermitian symmetry handled by maybe_conjugate.
          u_scaled, v_scaled, w_scaled, vis = maybe_conjugate(
            U_SIGN * u * wavelength * pixsizex,
            V_SIGN * v * wavelength * pixsizey,
            W_SIGN * w * wavelength,
            vis,
          )

          u_grid = (u_scaled * nx) % nx
          v_grid = (v_scaled * ny) % ny
          w_grid = (w_scaled - w0) / dw

          u_pixel_start = int(np.round(u_grid)) - half_support_int
          v_pixel_start = int(np.round(v_grid)) - half_support_int
          w_pixel_start = int(np.round(w_grid)) - half_support_int

          kernel.evaluate_support(u_grid, u_pixel_start, x_taps)
          kernel.evaluate_support(v_grid, v_pixel_start, y_taps)
          kernel.evaluate_support(w_grid, w_pixel_start, z_taps)

          for d in range(ndir):
            didx = idx + (d,)
            dir_vis = maybe_apply_jones(JONES_VIS, jones_params, vis, didx)
            dir_weight = maybe_apply_jones(JONES_WGT, jones_params, wgt, didx)
            dir_vis = apply_weights(dir_vis, dir_weight)

            for zo in range(support):
              zi = w_pixel_start + zo
              zkw = z_taps[zo]
              for xo in range(support):
                xi = (u_pixel_start + xo) % nx
                xkw = x_taps[xo]
                for yo in range(support):
                  yi = (v_pixel_start + yo) % ny
                  ykw = y_taps[yo]
                  weighted_stokes = apply_weights(dir_vis, xkw * ykw * zkw)
                  accumulate_data(weighted_stokes, vis_grid, (d, zi, xi, yi), 0)

    return vis_grid

  return apply_w_impl if APPLY_W else no_w_impl


@numba.njit(**{**JIT_OPTIONS, "parallel": False})
def wgrid(
  uvw: npt.NDArray[np.floating],
  visibilities: npt.NDArray[np.complexfloating],
  weights: npt.NDArray[np.floating],
  flags: npt.NDArray[np.integer],
  frequencies: npt.NDArray[np.floating],
  wgrid_params,
  nx: int,
  ny: int,
  pixsizex: float,
  pixsizey: float,
  pol_schema: Schema,
  stokes_schema: Schema,
  apply_w,
  jones_params: Tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.integer], Schema]
  | None = None,
) -> npt.NDArray[np.complexfloating]:
  return wgrid_impl(
    uvw,
    visibilities,
    weights,
    flags,
    frequencies,
    wgrid_params,
    nx,
    ny,
    pixsizex,
    pixsizey,
    pol_schema,
    stokes_schema,
    numba.literally(apply_w),
    jones_params,
  )
