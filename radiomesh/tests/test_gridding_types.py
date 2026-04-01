import numba
import numpy as np
from numba.np.numpy_support import as_struct_dtype

from radiomesh.gridding_types import (
  TILE_BITS,
  CachedAlignedCounter,
  StructUvwTile,
  UvwTile,
)
from radiomesh.gridding_types import TILE_MASK as MAX_TILE

def test_cache_aligned_counter():
  @numba.njit(nogil=True)
  def update(a):
    a.item_ptr(0).field_ptr("count").atomic_rmw("add", 1)
    a.item_ptr(1).field_ptr("count").atomic_rmw("sub", 1)

  A = np.zeros(100, dtype=as_struct_dtype(CachedAlignedCounter))
  update(A)
  assert A[0]["count"] == 1
  assert A[1]["count"] == -1


def test_record_uvw_tile():
  @numba.njit(nogil=True)
  def get_tiles(u, v, w):
    arr = np.zeros(1, dtype=UvwTile)
    arr[0].set_index(u, v, w)
    return arr[0].u_tile, arr[0].v_tile, arr[0].w_tile

  @numba.njit(nogil=True)
  def get_index(u, v, w):
    arr = np.zeros(1, dtype=UvwTile)
    arr[0].set_index(u, v, w)
    return arr[0]["index"]

  # Zero values
  assert get_tiles(0, 0, 0) == (0, 0, 0)

  # Round-trip with distinct values per axis
  u, v, w = 1, 2, 3
  assert get_tiles(u, v, w) == (u, v, w)

  # Max tile values (all 21 bits set per field)
  assert get_tiles(MAX_TILE, MAX_TILE, MAX_TILE) == (MAX_TILE, MAX_TILE, MAX_TILE)

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


def test_struct_uvw_tile():
  @numba.njit(nogil=True)
  def get_tiles(u, v, w):
    tile = StructUvwTile.from_uvw(u, v, w)
    return tile.u_tile, tile.v_tile, tile.w_tile

  @numba.njit(nogil=True)
  def get_index(u, v, w):
    tile = StructUvwTile.from_uvw(u, v, w)
    return tile.index

  # Zero values
  assert get_tiles(0, 0, 0) == (0, 0, 0)

  # Round-trip with distinct values per axis
  u, v, w = 1, 2, 3
  assert get_tiles(u, v, w) == (u, v, w)

  # Max tile values (all 21 bits set per field)
  assert get_tiles(MAX_TILE, MAX_TILE, MAX_TILE) == (MAX_TILE, MAX_TILE, MAX_TILE)

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
