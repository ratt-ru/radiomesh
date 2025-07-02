import math

import numba
from numba.extending import overload
from numba.core import types
import numpy as np
from typing import List

from radiomesh.es_kernel import es_kernel_factory
from radiomesh.product import (
  load_data_factory,
  store_data_factory,
  apply_weight_factory,
  pol_to_stokes_factory,
)

JIT_OPTIONS = {"parallel": False, "nogil": True, "cache": True, "fastmath": True}

def parse_schema(schema: str) -> List[str]:
  return [s.strip().upper() for s in schema.lstrip("[ ").rstrip("] ").split(",")]

# def load_data(schema, data, index):
#   pass

# @overload(load_data, prefer_literal=True)
# def load_data_overload(schema, data, index):
#   if not isinstance(schema, types.Literal):
#     return None

#   list_schema = parse_schema(schema.literal_value)
#   load_pol = load_data_factory(list_schema)
#   return lambda schema, data, index: load_pol(data, index)

# def store_data(schema, values, data, index):
#   pass

# @overload(store_data, prefer_literal=True)
# def store_data_overload(schema, values, data, index):
#   if not isinstance(schema, types.Literal):
#     return None

#   list_schema = parse_schema(schema.literal_value)
#   store_pol = store_data_factory(list_schema)
#   return lambda schema, values, data, index: store_pol(values, data, index)

# def apply_weight(schema, data, weight):
#   pass

# @overload(apply_weight, prefer_literal=True)
# def apply_weight_overload(schema, data, weight):
#   if not isinstance(schema, types.Literal):
#     return None

#   list_schema = parse_schema(schema.literal_value)
#   apply_weights = apply_weight_factory(list_schema)
#   return lambda schema, data, weight: apply_weights(data, weight)


# def pol_to_stokes(pol_schema, stokes_schema, data):
#   pass

# @overload(pol_to_stokes, prefer_literal=True)
# def pol_to_stokes_overload(pol_schema, stokes_schema, data):
#   p_to_s = pol_to_stokes_factory(pol_schema, stokes_schema)
#   return lambda pol_schema, stokes_schema, data: p_to_s(data)


def do_pol_test(visibilities, uvw, weights, frequencies, nx, ny, fov, support, schema):
  pass


@overload(do_pol_test, jit_options=JIT_OPTIONS)
def do_pol_test_overload(visibilities, uvw, weights, frequencies, nx, ny, fov, support, schema):
  if not isinstance(visibilities, types.Array) or not isinstance(visibilities.dtype, types.Complex):
    raise TypeError(f"'visibilities' {visibilities} must be a Complex Array")

  if not isinstance(uvw, types.Array) or not isinstance(uvw.dtype, types.Float):
    raise TypeError(f"'uvw' {uvw} must be a Float Array")

  if not isinstance(frequencies, types.Array) or not isinstance(frequencies.dtype, types.Float):
    raise TypeError(f"'frequencies' {frequencies} must be a Float Array")

  if not isinstance(weights, types.Array) or not isinstance(weights.dtype, types.Float):
    raise TypeError(f"'weights' {weights} must be a Float Array")

  if not isinstance(schema, types.Literal):
    return None

  if not isinstance(schema.literal_value, str):
    raise TypeError(f"schema '{schema.literal_value}' is not a string")

  if not isinstance(support, types.Literal):
    return None

  if not isinstance(support.literal_value, int):
    raise TypeError(f"support '{support}' is not an int")

  if not isinstance(nx, types.Literal):
    return None

  if not isinstance(nx.literal_value, int):
    raise TypeError(f"nx '{nx}' is not an int")

  if not isinstance(ny, types.Literal):
    return None

  if not isinstance(ny.literal_value, int):
    raise TypeError(f"ny '{ny}' is not an int")

  if not isinstance(fov, types.Literal):
    return None

  if not isinstance(fov.literal_value, str):
    raise TypeError(f"fov '{fov}' is not a str")

  try:
    pol_str, stokes_str = schema.literal_value.split("->")
  except ValueError:
    return ValueError(
      f"{schema} should be of the form "
      f"[XX,XY,YX,YY] -> [I,Q,U,V]"
    )

  pol_schema = parse_schema(pol_str)
  stokes_schema = parse_schema(stokes_str)
  NPOL = len(pol_schema)
  NSTOKES = len(stokes_schema)
  NX = nx.literal_value
  NY = ny.literal_value
  FOV = float(fov.literal_value)
  CELL_SIZE_X = FOV * math.pi / 180.0 / NX
  CELL_SIZE_Y = FOV * math.pi / 180.0 / NY
  U_CELL = 1.0 / (NX * CELL_SIZE_X)
  V_CELL = 1.0 / (NY * CELL_SIZE_Y)
  U_MAX = 1.0 / CELL_SIZE_X / 2.0
  V_MAX = 1.0 / CELL_SIZE_Y / 2.0
  LIGHTSPEED = 3e9    # temporary
  SUPPORT = support.literal_value
  HALF_SUPPORT = SUPPORT // 2
  BETA_K = 2.3 * HALF_SUPPORT
  KERNEL_POSITION = tuple(float(p - HALF_SUPPORT) for p in range(SUPPORT))

  # Generate intrinsics
  load_vis_data = load_data_factory(pol_schema)
  load_weight_data = load_data_factory(pol_schema)
  load_uvw_data = load_data_factory(["U", "V", "W"])
  apply_weights = apply_weight_factory(pol_schema)
  store_data = store_data_factory(stokes_schema, 0)
  pol_to_stokes = pol_to_stokes_factory(pol_schema, stokes_schema)
  es_kernel_pos, es_kernel = es_kernel_factory(BETA_K)

  def impl(visibilities, uvw, weights, frequencies, nx, ny, fov, support, schema):
    ntime, nbl, nchan, npol = visibilities.shape

    if npol != NPOL:
      raise ValueError(
        f"Number of schema {pol_str} ({npol}) "
        f"and visibility polarisations {NPOL} differ."
      )

    if frequencies.shape[0] != visibilities.shape[2]:
      raise ValueError(
        f"visibility {visibilities.shape[2]} and "
        f"frequency {frequencies.shape[0]} shapes differ"
      )

    wavelengths = frequencies / LIGHTSPEED

    vis_grid = np.zeros((NSTOKES, NX, NY), visibilities.real.dtype)
    weight_grid = np.zeros((NSTOKES, NX, NY), weights.dtype)

    for t in numba.prange(ntime):
      for bl in range(nbl):
        u, v, w = load_uvw_data(uvw, (t, bl))
        for ch in range(nchan):
          # Pixel coordinates
          u_grid = (u * wavelengths[ch] + U_MAX) / U_CELL
          v_grid = (v * wavelengths[ch] + V_MAX) / V_CELL

          # Indices
          u_index = int(np.round(u_grid))
          v_index = int(np.round(v_grid))

          x_idx = es_kernel_pos(KERNEL_POSITION, u_index)
          y_idx = es_kernel_pos(KERNEL_POSITION, v_index)

          x_kernel = es_kernel(x_idx, u_grid)
          y_kernel = es_kernel(y_idx, v_grid)

          vis = load_vis_data(visibilities, (t, bl, ch))
          wgt = load_weight_data(weights, (t, bl, ch))
          vis = apply_weights(vis, wgt)
          stokes = pol_to_stokes(vis)

          for xfi, xk in zip(numba.literal_unroll(x_idx), numba.literal_unroll(x_kernel)):
            xi = int(xfi)
            for yfi, yk in zip(numba.literal_unroll(y_idx), numba.literal_unroll(y_kernel)):
              weighted_stokes = apply_weights(stokes, xk * yk)
              store_data(weighted_stokes, vis_grid, (xi, int(yfi)))

    return vis_grid

  return impl


@numba.njit(**JIT_OPTIONS)
def do_pol_test_wrapper(visibilities, uvw, weights, frequencies, nx, ny, fov, support, schema):
  return do_pol_test(
    visibilities,
    uvw,
    weights,
    frequencies,
    numba.literally(nx),
    numba.literally(ny),
    numba.literally(fov),
    numba.literally(support),
    numba.literally(schema)
  )

def do_test_product():
  shape = (1024, 1024, 64, 4)
  nx = ny = 1024
  vis = np.random.random(shape) + 0j
  vis += np.random.random(shape) * 1j
  weights = np.random.random(shape)
  uvw = np.random.random(shape[:2] + (3,))
  freqs = np.linspace(.856e9, 2*.856e9, shape[2])
  print(vis.nbytes / (1024.**3))

  import time
  start = time.time()
  result = do_pol_test_wrapper(vis, uvw, weights, freqs, nx, ny, "5.0", 15, "[XX,XY,YX,YY] -> [I,Q,U,V]")
  print(f"Elapsed {time.time() - start}s")
  print(result.shape, result.dtype)
  with open("output.txt", "w") as f:
    for sig, asm in do_pol_test_wrapper.inspect_asm().items():
      print(sig, file=f)
      print(asm, file=f)



  #do_pol_test_wrapper.parallel_diagnostics(level=4)
  return

  from pprint import pprint


  import time
  start = time.time()
  print(to_stokes_array(vis, "[XX,XY,YX,YY] -> [I,Q]").shape)
  print(f"Elapsed {time.time() - start}s")
  with open("output.txt", "w") as f:
    for sig, asm in to_stokes_array.inspect_asm().items():
      print(sig, file=f)
      print(asm, file=f)

  # test_load_pol(vis, "[XX,XY,YX,YY]")

def test_grid_fourcorr(uvw, freq, vis, wgt, npix, fov, cell_rad,
                       epsilon=1e-4, precision='double', nthreads=1):

  import time
  start = time.time()
  result = do_pol_test_wrapper(vis, uvw, wgt, freq, npix, npix, str(fov), 6, "[XX,XY,YX,YY] -> [I,Q,U,V]")
  print(f"Elapsed {time.time() - start}s")
  return result

if __name__ == "__main__":
    import sys
    from casacore.tables import table
    from scipy.constants import c as lightspeed
    ms_name = sys.argv[1]
    ms = table(ms_name)
    uvw = ms.getcol('UVW')
    vis = ms.getcol('DATA')
    time = ms.getcol('TIME')
    try:
        wgt = ms.getcol('WEIGHT_SPECTRUM')
    except:
        wgt = np.ones(vis.shape, dtype='f4')
    print(f"Visibility size {vis.nbytes / 1024.**3}")
    ms.close()
    freq = table(f'{ms_name}::SPECTRAL_WINDOW').getcol('CHAN_FREQ')[0]

    utime = np.unique(time)
    vis = vis.reshape((utime.size, -1) + vis.shape[1:])
    wgt = wgt.reshape((utime.size, -1) + wgt.shape[1:])
    uvw = uvw.reshape((utime.size, -1) + uvw.shape[1:])
    uv_max = np.maximum(np.abs(uvw[:, 0]).max(), np.abs(uvw[:, 1]).max())
    max_freq = freq.max()
    cell_N = 1.0 / (2 * uv_max * max_freq / 3e9)  # max cell size
    cell_rad = cell_N/2.0  # oversample by a factor of two
    fov = 1.0  # field of view degrees
    # import ipdb; ipdb.set_trace()
    npix = int(fov/np.rad2deg(cell_rad))
    if npix % 2:
        npix += 1


    print(test_grid_fourcorr(uvw, freq, vis, wgt, npix, fov, cell_rad).shape)

  # shape = (1024, 1024, 64, 4)
  # vis = np.random.random(shape) + 0j
  # vis += np.random.random(shape) * 1j
  # uvw = np.random.random(shape[:2] + (3,))
  # print(vis.nbytes / (1024.**3))

  # # vis[:] = 2 + 1j
  # import time
  # start = time.time()
  # result = do_pol_test_wrapper(vis, uvw, 1024, 1024, "5.0", 15, "[XX,XY,YX,YY] -> [I,Q,U,V]")
  # print(f"Elapsed {time.time() - start}s")
  # print(result.shape, result.dtype)
  # with open("output.txt", "w") as f:
  #   for sig, asm in do_pol_test_wrapper.inspect_asm().items():
  #     print(sig, file=f)
  #     print(asm, file=f)
