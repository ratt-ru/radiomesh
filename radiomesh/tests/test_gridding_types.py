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
from radiomesh.parameters import estimate_gridding_parameters


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
def parallel_fn(uvw, frequencies, params, vis, weight, flag):
  impl = ParallelWGridderImpl(uvw, frequencies, params, False, False, False)
  impl.scan_data(vis, weight, flag)
  return impl


@numba.njit(parallel=False, nogil=True)
def serial_fn(uvw, frequencies, params, vis, weight, flag):
  impl = WGridderImpl(uvw, frequencies, params, False, False, False)
  impl.scan_data(vis, weight, flag)
  return impl


def _default_params(uvw, apply_w=False, nvis=0, wmin_d=0.0, wmax_d=0.0):
  """Minimal WGridderParameters for tests that don't exercise count_ranges."""
  nx, ny = 128, 128
  pixsize = 1.0 / 3600.0 * np.pi / 180.0
  return estimate_gridding_parameters(
    nx,
    ny,
    pixsize,
    pixsize,
    epsilon=1e-6,
    apply_w=apply_w,
    single=(uvw.dtype == np.float32),
    nvis=nvis,
    nthreads=1,
    wmin_d=wmin_d,
    wmax_d=wmax_d,
    gridding=True,
  )


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

  params = _default_params(uvw_coordinates)
  impl = grid_fn(uvw_coordinates, frequencies, params, vis, weight, flag)
  with ctx:
    grid_fn.parallel_diagnostics(level=4)

  ir = grid_fn.inspect_llvm()[grid_fn.signatures[0]]
  assert "parfor" in ir if parallel else "parfor" not in "ir"

  mask = (np.abs(vis) * weight * (flag != 0)) > 0.0
  np.testing.assert_array_equal(mask, impl.mask)
  assert np.count_nonzero(mask) == impl.nvis
  np.testing.assert_array_equal(frequencies / LIGHTSPEED, impl.wavelengths)


# ----------------------------------------------------------------------
# count_ranges tests
# ----------------------------------------------------------------------


@numba.njit(parallel=True, nogil=True)
def parallel_count_ranges(uvw, frequencies, vis, weight, flag, params, px, py):
  impl = ParallelWGridderImpl(uvw, frequencies, params, False, False, False)
  impl.scan_data(vis, weight, flag)
  impl.count_ranges(px, py)
  return impl


@numba.njit(parallel=False, nogil=True)
def serial_count_ranges(uvw, frequencies, vis, weight, flag, params, px, py):
  impl = WGridderImpl(uvw, frequencies, params, False, False, False)
  impl.scan_data(vis, weight, flag)
  impl.count_ranges(px, py)
  return impl


@numba.njit(nogil=True)
def _impl_get_uvw_tile_index(impl, u, v, w, ch):
  """Thin jit wrapper around `impl.uvw_tile_index` so Python test code can
  cross-check a fully-constructed impl's bucket indexing."""
  return impl.uvw_tile_index(u, v, w, ch)


def _build_gridded_impl(uvw, frequencies, apply_w, seed=42):
  """Construct a WGridderImpl, run scan_data + count_ranges with realistic params."""
  rng = np.random.default_rng(seed=seed)
  ntime, nbl, _ = uvw.shape
  (nchan,) = frequencies.shape
  weight = rng.random((ntime, nbl, nchan))
  vis = rng.random(weight.shape) + rng.random(weight.shape) * 1j
  flag = rng.integers(1, 4, size=weight.shape, dtype=np.uint8)

  # Mimic the pre-scan/post-scan flow: we need wmin/wmax from the scan.
  # Pre-compute a pre-scan estimate just to size params, then re-estimate
  # post-scan. For the test, since scan_data is cheap, do a quick pure-python
  # scan for wmin/wmax.
  wavelengths = frequencies / LIGHTSPEED
  mask_py = (np.abs(vis) * weight * (flag != 0)) > 0.0
  nvis = int(np.count_nonzero(mask_py))
  # uvw[t, bl, 2] is the w baseline; multiply each (t,bl) by each channel wavelength
  # to get effective |w| per (t, bl, ch).
  abs_w = np.abs(uvw[..., 2:3] * wavelengths[np.newaxis, np.newaxis, :])
  abs_w_masked = abs_w[mask_py]
  wmin_d = float(abs_w_masked.min()) if abs_w_masked.size else 0.0
  wmax_d = float(abs_w_masked.max()) if abs_w_masked.size else 0.0

  # Modest image so the grid is small.
  nx, ny = 128, 128
  pixsize_x = pixsize_y = 1.0 / 3600.0 * np.pi / 180.0  # 1 arcsec in radians
  params = estimate_gridding_parameters(
    nx,
    ny,
    pixsize_x,
    pixsize_y,
    epsilon=1e-6,
    apply_w=apply_w,
    single=(uvw.dtype == np.float32),
    nvis=nvis,
    nthreads=1,
    wmin_d=wmin_d,
    wmax_d=wmax_d,
    gridding=True,
  )
  return (
    uvw,
    frequencies,
    vis,
    weight,
    flag,
    params,
    nx,
    ny,
    pixsize_x,
    pixsize_y,
    wmin_d,
    wmax_d,
  )


@pytest.mark.filterwarnings(
  "ignore::numba.core.errors.NumbaPerformanceWarning",
)
@pytest.mark.parametrize("apply_w", [True, False])
@pytest.mark.parametrize("grid_fn", [parallel_count_ranges, serial_count_ranges])
def test_count_ranges_totals_and_invariants(
  grid_fn, apply_w, uvw_coordinates, frequencies
):
  """Total vis count matches scan_data.nvis, and every (uvw_tile, range) pair
  round-trips: every channel in every range re-derives the same uvw_tile."""
  (uvw, freqs, vis, weight, flag, params, nx, ny, px, py, _, _) = _build_gridded_impl(
    uvw_coordinates, frequencies, apply_w
  )

  impl = grid_fn(uvw, freqs, vis, weight, flag, params, px, py)

  blockstart = impl.blockstart
  ranges = impl.ranges
  # Totals: sum of (ch_end - ch_begin) over all ranges == nvis.
  vis_total = int(
    np.sum(ranges["ch_end"].astype(np.int64) - ranges["ch_begin"].astype(np.int64))
  )
  assert vis_total == impl.nvis, f"total vis {vis_total} != nvis {impl.nvis}"

  # Every range lies inside its bucket's contiguous slice. Since we subdivided
  # blockstart, the bucket for entry i is uvw_tile_i and its slice is
  # [offset_i, offset_{i+1}). We re-derive each channel's uvw_tile by calling the
  # jit method on the impl itself — it encapsulates all geometry.
  offsets = blockstart["offset"].astype(np.int64)
  uvw_tiles = blockstart["uvw_tile"].astype(np.uint64)
  for i in range(len(blockstart)):
    start = offsets[i]
    end = offsets[i + 1] if i + 1 < len(blockstart) else len(ranges)
    expected = uvw_tiles[i]
    for rng_row in ranges[start:end]:
      t = int(rng_row["time"])
      bl = int(rng_row["bl"])
      cb = int(rng_row["ch_begin"])
      ce = int(rng_row["ch_end"])
      u, v, w = uvw[t, bl]
      if w < 0.0:
        u, v, w = -u, -v, -w
      for ch in range(cb, ce):
        idx = _impl_get_uvw_tile_index(impl, float(u), float(v), float(w), ch)
        assert np.uint64(idx) == expected, (
          f"bucket mismatch at block {i} range [{cb},{ce}) ch={ch}: "
          f"got {int(idx):#x} expected {int(expected):#x}"
        )


@pytest.mark.filterwarnings(
  "ignore::numba.core.errors.NumbaPerformanceWarning",
)
@pytest.mark.parametrize("apply_w", [True, False])
def test_count_ranges_serial_parallel_equivalent(apply_w, uvw_coordinates, frequencies):
  """Serial and parallel must produce the same *set* of ranges (as a whole)
  and the same unique uvw_tile set. Pre-subdivision blockstart lengths need not
  match because the load-balancing step depends on `numba.get_num_threads()`."""
  (uvw, freqs, vis, weight, flag, params, nx, ny, px, py, _, _) = _build_gridded_impl(
    uvw_coordinates, frequencies, apply_w
  )
  impl_p = parallel_count_ranges(uvw, freqs, vis, weight, flag, params, px, py)
  impl_s = serial_count_ranges(uvw, freqs, vis, weight, flag, params, px, py)

  # Same number of RowchanRange objects.
  assert len(impl_p.ranges) == len(impl_s.ranges)

  # Same set of (time, bl, ch_begin, ch_end) quadruples — the atomic slot
  # allocator in Pass 2 reorders within buckets non-deterministically, so we
  # compare as a sorted multiset.
  def _canon(ranges):
    order = np.lexsort(
      (ranges["ch_end"], ranges["ch_begin"], ranges["bl"], ranges["time"])
    )
    return ranges[order]

  np.testing.assert_array_equal(_canon(impl_p.ranges), _canon(impl_s.ranges))

  # Same set of *unique* uvw_tile buckets (blockstart[] may be longer on one
  # side due to subdivision, but unique keys must match).
  np.testing.assert_array_equal(
    np.unique(impl_p.blockstart["uvw_tile"]),
    np.unique(impl_s.blockstart["uvw_tile"]),
  )

  # Same mask (pass-1 sets mask=2 at tile boundaries — this is deterministic).
  np.testing.assert_array_equal(impl_p.mask, impl_s.mask)

  # Same uranges/vranges (pure function of the bucket set).
  np.testing.assert_array_equal(impl_p.uranges_offsets, impl_s.uranges_offsets)
  np.testing.assert_array_equal(impl_p.uranges, impl_s.uranges)
  np.testing.assert_array_equal(impl_p.vranges_offsets, impl_s.vranges_offsets)
  np.testing.assert_array_equal(impl_p.vranges, impl_s.vranges)


@pytest.mark.filterwarnings(
  "ignore::numba.core.errors.NumbaPerformanceWarning",
)
def test_count_ranges_uranges_cover_blockstart(uvw_coordinates, frequencies):
  """Every blockstart tile's pixel extent must be covered by uranges/vranges
  for every w-plane the bucket contributes to."""
  (uvw, freqs, vis, weight, flag, params, nx, ny, px, py, _, _) = _build_gridded_impl(
    uvw_coordinates, frequencies, apply_w=True
  )
  impl = serial_count_ranges(uvw, freqs, vis, weight, flag, params, px, py)
  supp = int(params.kernel.support)
  log2tile = 5 if uvw.dtype == np.float32 else 4
  tilesize = 1 << log2tile
  nsafe = (supp + 1) // 2
  nu = int(impl.nu)
  nv = int(impl.nv)

  def covered(intervals, lo, hi, n):
    """True iff [lo, hi) mod n is fully covered by `intervals` (list of [a,b))."""
    # Build the set of pixels this range covers (with wrap).
    if lo < 0 or hi > n:
      # Wraps — split
      wanted = np.concatenate(
        [
          np.arange(lo % n, n),
          np.arange(0, hi % n),
        ]
      )
    else:
      wanted = np.arange(lo, hi)
    covered_set = np.zeros(n, dtype=bool)
    for a, b in intervals:
      covered_set[a:b] = True
    return bool(np.all(covered_set[wanted % n]))

  uranges_offsets = impl.uranges_offsets
  uranges = impl.uranges
  vranges_offsets = impl.vranges_offsets
  vranges = impl.vranges

  for entry in impl.blockstart:
    tu, tv, mp = uvw_tile_from_index(int(entry["uvw_tile"]))
    lo_u = int(tu) * tilesize - nsafe
    hi_u = (int(tu) + 1) * tilesize + nsafe
    lo_v = int(tv) * tilesize - nsafe
    hi_v = (int(tv) + 1) * tilesize + nsafe
    for k in range(supp):
      pl = int(mp) + k
      u_ivs = uranges[uranges_offsets[pl] : uranges_offsets[pl + 1]]
      v_ivs = vranges[vranges_offsets[pl] : vranges_offsets[pl + 1]]
      assert covered(u_ivs, lo_u, hi_u, nu), (
        f"uranges plane {pl} does not cover tile_u {tu}: need [{lo_u}, {hi_u}) "
        f"got {u_ivs.tolist()}"
      )
      assert covered(v_ivs, lo_v, hi_v, nv), (
        f"vranges plane {pl} does not cover tile_v {tv}: need [{lo_v}, {hi_v}) "
        f"got {v_ivs.tolist()}"
      )
