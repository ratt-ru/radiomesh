from contextlib import nullcontext

import numba
import numpy as np
import pytest
from numba import typed
from numba.np.numpy_support import as_struct_dtype

import radiomesh.intrinsics  # noqa: F401
from radiomesh.constants import LIGHTSPEED
from radiomesh.gridding_types import (
  TILE_BITS,
  CachedAlignedCounter,
  ParallelWGridderImpl,
  WGridderImpl,
  index_from_uvw_tile,
  u_tile_from_index,
  uvw_tile_from_index,
  v_tile_from_index,
  w_tile_from_index,
)
from radiomesh.gridding_types import (
  TILE_MASK as MAX_TILE,
)


def test_record_assignment():
  @numba.njit(nogil=True, cache=True)
  def update(a):
    a[0].a = 1
    a[0].b = 2.0

  dtype = np.dtype([("a", np.int32), ("b", np.float64)])
  array = np.zeros(10, dtype=dtype)
  update(array)
  assert array[0]["a"] == 1
  assert array[0]["b"] == 2.0


def test_cache_aligned_counter():
  @numba.njit(nogil=True)
  def update(a, indices, op):
    literal_op = numba.literally(op)
    for i in indices:
      a.item_ptr(*i).field_ptr("count").atomic_rmw(literal_op, 1)

  A = np.zeros((5, 5), dtype=as_struct_dtype(CachedAlignedCounter))
  assert A.nbytes == np.prod(A.shape) * 64
  indices = typed.List([(0, 1), (1, 0), (4, 4)])
  update(A, indices, "add")
  for i in indices:
    assert A[i]["count"] == 1

  update(A, indices, "sub")
  update(A, indices, "sub")
  for i in indices:
    assert A[i]["count"] == -1


def test_uvw_tile_index():
  @numba.njit(nogil=True)
  def get_tiles(u, v, w):
    index = index_from_uvw_tile(u, v, w)
    return u_tile_from_index(index), v_tile_from_index(index), w_tile_from_index(index)

  @numba.njit(nogil=True)
  def get_tiles_tuple(u, v, w):
    return uvw_tile_from_index(index_from_uvw_tile(u, v, w))

  @numba.njit(nogil=True)
  def get_index(u, v, w):
    return index_from_uvw_tile(u, v, w)

  # Index is always a uint64 (checked via the pure-Python path; njit unboxes to int)
  assert isinstance(index_from_uvw_tile(0, 0, 0), np.uint64)

  # Zero values
  assert get_tiles(0, 0, 0) == (0, 0, 0)
  assert get_tiles_tuple(0, 0, 0) == (0, 0, 0)

  # Round-trip with distinct values per axis
  u, v, w = 1, 2, 3
  assert get_tiles(u, v, w) == (u, v, w)
  assert get_tiles_tuple(u, v, w) == (u, v, w)

  # Max tile values (all 21 bits set per field)
  assert get_tiles(MAX_TILE, MAX_TILE, MAX_TILE) == (MAX_TILE, MAX_TILE, MAX_TILE)
  assert get_tiles_tuple(MAX_TILE, MAX_TILE, MAX_TILE) == (
    MAX_TILE,
    MAX_TILE,
    MAX_TILE,
  )

  # Only u set — index should equal u (sits in lowest bits)
  assert get_index(7, 0, 0) == np.uint64(7)

  # Only v set — index should be v shifted by TILE_BITS
  assert get_index(0, 5, 0) == np.uint64(5 << TILE_BITS)

  # Only w set — index should be w shifted by TILE_BITS + TILE_BITS
  assert get_index(0, 0, 11) == np.uint64(11 << (TILE_BITS + TILE_BITS))

  # Verify fields are independent (setting one doesn't corrupt others)
  for u, v, w in [
    (MAX_TILE, 0, 0),
    (0, MAX_TILE, 0),
    (0, 0, MAX_TILE),
    (MAX_TILE, MAX_TILE, 0),
    (0, MAX_TILE, MAX_TILE),
  ]:
    assert get_tiles(u, v, w) == (u, v, w)
    assert get_tiles_tuple(u, v, w) == (u, v, w)


@numba.njit(parallel=True, nogil=True)
def parallel_fn(uvw, frequencies, vis, weight, flag):
  impl = ParallelWGridderImpl(uvw, frequencies, False, False, False)
  impl.scan_data(vis, weight, flag)
  return impl


@numba.njit(parallel=False, nogil=True)
def serial_fn(uvw, frequencies, vis, weight, flag):
  impl = WGridderImpl(uvw, frequencies, False, False, False)
  impl.scan_data(vis, weight, flag)
  return impl


@pytest.mark.filterwarnings(
  "ignore::numba.core.errors.NumbaPerformanceWarning",
  reason="The njit driver has no parallel directives, but the overloads do",
)
@pytest.mark.parametrize(
  "parallel, grid_fn, ctx",
  [
    (True, parallel_fn, nullcontext()),
    (
      False,
      serial_fn,
      pytest.raises(RuntimeError, match="self.setup has not been called"),
    ),
  ],
)
def test_wgridder_impl(parallel, grid_fn, ctx, uvw_coordinates, frequencies):
  rng = np.random.default_rng(seed=42)
  ntime, nbl, _ = uvw_coordinates.shape
  (nchan,) = frequencies.shape

  weight = rng.random((ntime, nbl, nchan))
  vis = rng.random(weight.shape) + rng.random(weight.shape) * 1j
  flag = rng.integers(0, 8, size=weight.shape, dtype=np.uint8)

  impl = grid_fn(uvw_coordinates, frequencies, vis, weight, flag)
  with ctx:
    grid_fn.parallel_diagnostics(level=4)

  ir = grid_fn.inspect_llvm()[grid_fn.signatures[0]]
  assert "parfor" in ir if parallel else "parfor" not in "ir"

  mask = (np.abs(vis) * weight * (flag != 0)) > 0.0
  np.testing.assert_array_equal(mask, impl.mask)
  assert np.count_nonzero(mask) == impl.nvis
  np.testing.assert_array_equal(frequencies / LIGHTSPEED, impl.wavelengths)
