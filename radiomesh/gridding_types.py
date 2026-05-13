from dataclasses import dataclass
from typing import Any, Dict

import numba
import numpy as np
from numba import types
from numba.core.types import StructRef
from numba.experimental import structref
from numba.extending import (
  overload,
  overload_attribute,
  overload_method,
  register_jitable,
)
from numba.np.numpy_support import as_dtype, as_struct_dtype

from radiomesh.constants import CACHE_LINE_SIZE, LIGHTSPEED
from radiomesh.numba_utils import make_structref_property

# A 64 bit cache aligned integer counter
CachedAlignedCounter = types.Record(
  [("count", {"type": types.int64, "offset": 0})], size=CACHE_LINE_SIZE, aligned=True
)

# (time, bl, ch_begin, ch_end) — the unit of tile-sorted channel runs
SampleChanRange = types.Record(
  [
    ("time", {"type": types.uint32, "offset": 0}),
    ("bl", {"type": types.uint32, "offset": 4}),
    ("ch_begin", {"type": types.uint16, "offset": 8}),
    ("ch_end", {"type": types.uint16, "offset": 10}),
  ],
  size=12,
  aligned=True,
)

# (uvw_tile packed into uint64, offset into ranges[])
BlockStartEntry = types.Record(
  [
    ("uvw_tile", {"type": types.uint64, "offset": 0}),
    ("offset", {"type": types.int64, "offset": 8}),
  ],
  size=16,
  aligned=True,
)

COUNTER_DTYPE = as_struct_dtype(CachedAlignedCounter)
SAMPLE_CHANRANGE_DTYPE = as_struct_dtype(SampleChanRange)
BLOCK_START_DTYPE = as_struct_dtype(BlockStartEntry)


# Tile index takes 63 bits, we use 21 bits for each of the components
UVW_TILE_BITS = 63
TILE_BITS = 21
TILE_MASK = (1 << TILE_BITS) - 1

U_TILE_BIT_OFFSET = 0
V_TILE_BIT_OFFSET = TILE_BITS
W_TILE_BIT_OFFSET = 2 * TILE_BITS

U_TILE_MASK = TILE_MASK << U_TILE_BIT_OFFSET
V_TILE_MASK = TILE_MASK << V_TILE_BIT_OFFSET
W_TILE_MASK = TILE_MASK << W_TILE_BIT_OFFSET


# UVW Tile handling functions
@register_jitable
def index_from_uvw_tile(u_tile, v_tile, w_tile):
  index = np.uint64((u_tile & TILE_MASK) << U_TILE_BIT_OFFSET)
  index |= (v_tile & TILE_MASK) << V_TILE_BIT_OFFSET
  index |= (w_tile & TILE_MASK) << W_TILE_BIT_OFFSET
  return index


@register_jitable
def u_tile_from_index(index):
  return (index & U_TILE_MASK) >> U_TILE_BIT_OFFSET


@register_jitable
def v_tile_from_index(index):
  return (index & V_TILE_MASK) >> V_TILE_BIT_OFFSET


@register_jitable
def w_tile_from_index(index):
  return (index & W_TILE_MASK) >> W_TILE_BIT_OFFSET


@register_jitable
def uvw_tile_from_index(index):
  return (u_tile_from_index(index), v_tile_from_index(index), w_tile_from_index(index))


@register_jitable
def fix_w(u, v, w):
  """Conjugation-symmetric fix: if w < 0, negate all three UVW components.

  Returns (u, v, w, imflip) where imflip is +1 if no flip was applied,
  -1 otherwise. The caller must apply imflip to the imaginary part of the
  visibility (gridding) or of the gathered sample (degridding).
  """
  if w < 0.0:
    return -u, -v, -w, -1.0
  return u, v, w, 1.0


@register_jitable
def _count_wrapped(lo, hi, n):
  """Count how many sub-intervals [lo, hi) produces once clipped into [0, n)
  with periodic wraparound. Result is either 1 or 2."""
  if lo < 0 and hi > n:
    # Covers the whole grid; single [0, n) interval.
    return np.int64(1)
  if lo < 0:
    return np.int64(2)
  if hi > n:
    return np.int64(2)
  return np.int64(1)


@register_jitable
def _emit_wrapped(out, write, lo, hi, n):
  """Emit wrapped interval(s) for [lo, hi) into `out` starting at `write`.
  Returns the new write index."""
  if lo < 0 and hi > n:
    out[write, 0] = np.int32(0)
    out[write, 1] = n
    return write + 1
  if lo < 0:
    out[write, 0] = np.int32(0)
    out[write, 1] = hi
    out[write + 1, 0] = n + lo
    out[write + 1, 1] = n
    return write + 2
  if hi > n:
    out[write, 0] = lo
    out[write, 1] = n
    out[write + 1, 0] = np.int32(0)
    out[write + 1, 1] = hi - n
    return write + 2
  out[write, 0] = lo
  out[write, 1] = hi
  return write + 1


@structref.register
class WGridderImplStructRef(StructRef):
  def preprocess_fields(self, fields):
    """Disallow literal types in field definitions"""
    return tuple((n, types.unliteral(t)) for n, t in fields)


@structref.register
class ParallelWGridderImplStructRef(WGridderImplStructRef):
  pass


class WGridderImpl(structref.StructRefProxy):
  def __new__(
    cls, uvw, frequencies, wgrid_params, invert_u=False, invert_v=False, invert_w=False
  ):
    return structref.StructRefProxy.__new__(
      cls, uvw, frequencies, wgrid_params, invert_u, invert_v, invert_w
    )

  uvw = make_structref_property("uvw")
  wavelengths = make_structref_property("wavelengths")
  wgrid_params = make_structref_property("wgrid_params")
  u_min = make_structref_property("u_min")
  u_max = make_structref_property("u_max")
  w_min_d = make_structref_property("w_min_d")
  w_max_d = make_structref_property("w_max_d")
  mask = make_structref_property("mask")
  nvis = make_structref_property("nvis")
  # Delegated via overload_attribute to wgrid_params / kernel:
  nu = make_structref_property("nu")
  nv = make_structref_property("nv")
  nw = make_structref_property("nw")
  support = make_structref_property("support")
  nsafe = make_structref_property("nsafe")
  apply_w = make_structref_property("apply_w")
  # count_ranges-derived scalars stored directly on the struct:
  dw = make_structref_property("dw")
  dw_reciprocal = make_structref_property("dw_reciprocal")
  wshift = make_structref_property("wshift")
  ushift = make_structref_property("ushift")
  vshift = make_structref_property("vshift")
  maxiu0 = make_structref_property("maxiu0")
  maxiv0 = make_structref_property("maxiv0")
  pixsize_x = make_structref_property("pixsize_x")
  pixsize_y = make_structref_property("pixsize_y")
  bucket_count = make_structref_property("bucket_count")
  ranges = make_structref_property("ranges")
  blockstart = make_structref_property("blockstart")
  uranges_offsets = make_structref_property("uranges_offsets")
  uranges = make_structref_property("uranges")
  vranges_offsets = make_structref_property("vranges_offsets")
  vranges = make_structref_property("vranges")


class ParallelWGridderImpl(WGridderImpl):
  pass


structref.define_boxing(WGridderImplStructRef, WGridderImpl)
structref.define_boxing(ParallelWGridderImplStructRef, ParallelWGridderImpl)


@dataclass
class WGridderImplTemplate:
  """Template parameters for generating WGridderImpl types"""

  wgridder_structref: type
  wgridder_type: type
  parallel_jit_options: Dict[str, Any]

  @property
  def jit_options(self):
    """Base jit options, which exclude parallelisation directives"""
    return {"error_model": "numpy", "nogil": True, "fastmath": True}

  @property
  def full_jit_options(self):
    """Full jit options, which may include parallelisation directives"""
    return {**self.jit_options, **self.parallel_jit_options}


WGRIDDER_TEMPLATES = [
  WGridderImplTemplate(WGridderImplStructRef, WGridderImpl, {"parallel": False}),
  WGridderImplTemplate(
    ParallelWGridderImplStructRef, ParallelWGridderImpl, {"parallel": True}
  ),
]


def _register_wgridder_overloads(template):
  wgridder_type = template.wgridder_type
  wgridder_structref = template.wgridder_structref
  jit_options = template.jit_options
  full_jit_options = template.full_jit_options

  @overload(wgridder_type, jit_options=full_jit_options)
  def overload_gridding_impl(
    uvw, frequencies, wgrid_params, invert_u, invert_v, invert_w
  ):
    struct_type = wgridder_structref(
      [
        ("uvw", uvw),
        ("wavelengths", frequencies),
        ("wgrid_params", wgrid_params),
        ("u_max", uvw.dtype),
        ("v_max", uvw.dtype),
        ("w_min_d", uvw.dtype),
        ("w_max_d", uvw.dtype),
        ("mask", types.Array(numba.uint8, 3, "C")),
        ("nvis", numba.uint64),
        ("nsafe", wgrid_params.field_dict["kernel"].field_dict["support"]),
        # count_ranges-derived scalars
        ("pixsize_x", numba.float64),
        ("pixsize_y", numba.float64),
        ("dw", numba.float64),
        ("dw_reciprocal", numba.float64),
        ("wshift", numba.float64),
        ("ushift", numba.float64),
        ("vshift", numba.float64),
        ("maxiu0", numba.int32),
        ("maxiv0", numba.int32),
        # count_ranges-derived arrays
        ("bucket_count", types.Array(CachedAlignedCounter, 3, "C")),
        ("ranges", types.Array(SampleChanRange, 1, "C")),
        ("blockstart", types.Array(BlockStartEntry, 1, "C")),
        ("uranges_offsets", types.Array(numba.int64, 1, "C")),
        ("uranges", types.Array(numba.int32, 2, "C")),
        ("vranges_offsets", types.Array(numba.int64, 1, "C")),
        ("vranges", types.Array(numba.int32, 2, "C")),
      ]
    )

    def impl(uvw, frequencies, wgrid_params, invert_u, invert_v, invert_w):
      for i in range(frequencies.shape[0]):
        if frequencies[i] < 0.0:
          raise NotImplementedError("negative frequencies")

        if i > 0 and frequencies[i - 1] >= frequencies[i]:
          raise NotImplementedError("Frequencies that do not increase monotically")

      obj = structref.new(struct_type)
      obj.uvw = uvw.copy()
      obj.wavelengths = frequencies / LIGHTSPEED
      obj.wgrid_params = wgrid_params
      obj.nsafe = wgrid_params.kernel.nsafe

      u_sign = uvw.dtype.type(-1.0 if invert_u else 1.0)
      v_sign = uvw.dtype.type(-1.0 if invert_v else 1.0)
      w_sign = uvw.dtype.type(-1.0 if invert_w else 1.0)

      u_max = uvw.dtype.type(0.0)
      v_max = uvw.dtype.type(0.0)

      for t in numba.prange(uvw.shape[0]):
        for bl in range(uvw.shape[1]):
          obj.uvw[t, bl, 0] *= u_sign
          obj.uvw[t, bl, 1] *= v_sign
          obj.uvw[t, bl, 2] *= w_sign

          u_max = max(u_max, abs(obj.uvw[t, bl, 0]))
          v_max = max(v_max, abs(obj.uvw[t, bl, 1]))

      wavelength_max = obj.wavelengths.max()
      obj.u_max = u_max * wavelength_max
      obj.v_max = v_max * wavelength_max

      # Initialise count_ranges-derived fields to empty / zero so the struct
      # is well-formed before count_ranges() populates them.
      obj.pixsize_x = 0.0
      obj.pixsize_y = 0.0
      obj.dw = 0.0
      obj.dw_reciprocal = 0.0
      obj.wshift = 0.0
      obj.ushift = 0.0
      obj.vshift = 0.0
      obj.maxiu0 = numba.int32(0)
      obj.maxiv0 = numba.int32(0)
      obj.bucket_count = np.empty((0, 0, 0), dtype=COUNTER_DTYPE)
      obj.ranges = np.empty(0, dtype=SAMPLE_CHANRANGE_DTYPE)
      obj.blockstart = np.empty(0, dtype=BLOCK_START_DTYPE)
      obj.uranges_offsets = np.empty(0, dtype=np.int64)
      obj.uranges = np.empty((0, 2), dtype=np.int32)
      obj.vranges_offsets = np.empty(0, dtype=np.int64)
      obj.vranges = np.empty((0, 2), dtype=np.int32)

      return obj

    return impl

  # ------------------------------------------------------------------
  # Attributes delegated to wgrid_params / kernel
  # ------------------------------------------------------------------
  @overload_attribute(wgridder_structref, "nu", jit_options=jit_options)
  def overload_nu(self):
    return lambda self: self.wgrid_params.nu

  @overload_attribute(wgridder_structref, "nv", jit_options=jit_options)
  def overload_nv(self):
    return lambda self: self.wgrid_params.nv

  @overload_attribute(wgridder_structref, "nw", jit_options=jit_options)
  def overload_nw(self):
    return lambda self: self.wgrid_params.nw

  @overload_attribute(wgridder_structref, "support", jit_options=jit_options)
  def overload_support(self):
    return lambda self: self.wgrid_params.kernel.support

  @overload_attribute(wgridder_structref, "apply_w", jit_options=jit_options)
  def overload_apply_w(self):
    return lambda self: self.wgrid_params.kernel.apply_w

  # ------------------------------------------------------------------
  # Inline pixel / uvw tile helpers
  # ------------------------------------------------------------------
  @overload_method(
    wgridder_structref, "uv_pixels", inline="always", jit_options=jit_options
  )
  def overload_uv_pixels(self, u_in, v_in):
    def impl(self, u_in, v_in):
      """Map physical UV coordinates (wavelengths) to oversampled grid coords.

      Returns (ufrac, vfrac, iu0, iv0) where (iu0, iv0) is the left/bottom
      kernel anchor and (ufrac, vfrac) are sub-pixel offsets within [0, supp).
      """
      nu = self.nu
      nv = self.nv
      u_norm = u_in * self.pixsize_x
      u_norm -= np.floor(u_norm)
      u_grid = u_norm * nu
      iu0 = min(np.int32(u_grid + self.ushift) - np.int32(nu), self.maxiu0)
      ufrac = u_grid - iu0

      v_norm = v_in * self.pixsize_y
      v_norm -= np.floor(v_norm)
      v_grid = v_norm * nv
      iv0 = min(np.int32(v_grid + self.vshift) - np.int32(nv), self.maxiv0)
      vfrac = v_grid - iv0

      return ufrac, vfrac, iu0, iv0

    return impl

  @overload_method(
    wgridder_structref, "uvw_tile_index", inline="always", jit_options=jit_options
  )
  def overload_uvw_tile_index(self, u, v, w, ch):
    uvw_dtype = as_dtype(self.field_dict["uvw"].dtype)
    LOG2TILE = 5 if uvw_dtype == np.float32 else 4

    def impl(self, u, v, w, ch):
      """Pack (tile_u, tile_v, minplane) into a uint64 uvw_tile.

      u, v, w must already have been w-flipped (see ``fix_w``).
      ``ch`` is target channel.
      """
      wavelength = self.wavelengths[ch]
      u_scaled = u * wavelength
      v_scaled = v * wavelength
      _, _, iu0, iv0 = self.uv_pixels(u_scaled, v_scaled)
      tile_u = (iu0 + self.nsafe) >> LOG2TILE
      tile_v = (iv0 + self.nsafe) >> LOG2TILE

      if not self.apply_w:
        return index_from_uvw_tile(tile_u, tile_v, 0)

      w_scaled = w * wavelength
      minw = max(0, np.int64((w_scaled * self.wshift) * self.dw_reciprocal))
      return index_from_uvw_tile(tile_u, tile_v, minw)

    return impl

  # ------------------------------------------------------------------
  # count_ranges stages
  # ------------------------------------------------------------------
  @overload_method(wgridder_structref, "_setup_geometry", jit_options=full_jit_options)
  def overload_setup_geometry(self, pixsize_x, pixsize_y):
    def impl(self, pixsize_x, pixsize_y):
      """4.1: w-plane geometry + uv_pixels constants."""
      nu = self.nu
      nv = self.nv
      support = self.support
      nsafe = self.nsafe
      nw = self.nw

      self.pixsize_x = pixsize_x
      self.pixsize_y = pixsize_y

      # ushift / vshift place the support-wide kernel at its left/bottom edge.
      self.ushift = support * (-0.5) + 1 + nu
      self.vshift = support * (-0.5) + 1 + nv
      self.maxiu0 = numba.int32((nu + nsafe) - support)
      self.maxiv0 = numba.int32((nv + nsafe) - support)

      if self.apply_w:
        # dw centred on nw so that wmin maps to plane (support/2). Matches
        # parameters.py apply_w_impl: dw = (wmax - wmin) / (nw - support).
        dw = (self.wgrid_params.wmax - self.wgrid_params.wmin) / (nw - support)
        self.dw = dw
        self.dw_reciprocal = 1.0 / dw
        self.wshift = dw - 0.5 * support * dw - self.wgrid_params.wmin
      else:
        self.dw = 1.0
        self.dw_reciprocal = 1.0
        self.wshift = 0.0

    return impl

  @overload_method(wgridder_structref, "_count_buckets", jit_options=full_jit_options)
  def overload_count_buckets(self):
    uvw_dtype = as_dtype(self.field_dict["uvw"].dtype)
    LOG2TILE = 5 if uvw_dtype == np.float32 else 4

    def impl(self):
      """4.2: allocate bucket_count histogram + Pass 1 counting."""
      ntime = self.uvw.shape[0]
      nbl = self.uvw.shape[1]
      nchan = self.wavelengths.shape[0]
      nu = self.nu
      nv = self.nv
      support = self.support
      nw = self.nw
      apply_w = self.apply_w

      ntiles_u = (nu >> LOG2TILE) + 3
      ntiles_v = (nv >> LOG2TILE) + 3
      if apply_w:
        nwmin = nw - support + 3
      else:
        nwmin = 1

      # np.zeros + Record dtype trips parfors on the parallel variant; allocate
      # uninitialised and explicitly zero the count field instead.
      bucket_count = np.empty((ntiles_u, ntiles_v, nwmin), dtype=COUNTER_DTYPE)
      for tu in numba.prange(ntiles_u):
        for tv in numba.prange(ntiles_v):
          for mw in numba.prange(nwmin):
            bucket_count[tu, tv, mw].count = np.int64(0)
      self.bucket_count = bucket_count

      for t in numba.prange(ntime):
        # Per-row scratch for the iterative divide-and-conquer stack.
        # Each entry is (ch_lo, ch_hi, idx_lo_as_int64, idx_hi_as_int64).
        stack_buf = np.empty((64, 4), dtype=np.int64)
        for bl in range(nbl):
          u_raw = self.uvw[t, bl, 0]
          v_raw = self.uvw[t, bl, 1]
          w_raw = self.uvw[t, bl, 2]
          u, v, w, _ = fix_w(u_raw, v_raw, w_raw)

          ch0 = 0
          while ch0 < nchan:
            while ch0 < nchan and self.mask[t, bl, ch0] == 0:
              ch0 += 1
            if ch0 >= nchan:
              break
            ch1 = ch0 + 1
            while ch1 < nchan and self.mask[t, bl, ch1] != 0:
              ch1 += 1

            tile_index0 = self.uvw_tile_index(u, v, w, ch0)
            tu0, tv0, mw0 = uvw_tile_from_index(tile_index0)
            bucket_count.item_ptr(tu0, tv0, mw0).field_ptr("count").atomic_rmw("add", 1)

            if ch0 + 1 < ch1:
              tile_index_last = self.uvw_tile_index(u, v, w, ch1 - 1)
              stack_size = 0
              if tile_index0 != tile_index_last:
                stack_buf[0, 0] = ch0
                stack_buf[0, 1] = ch1 - 1
                stack_buf[0, 2] = tile_index0
                stack_buf[0, 3] = tile_index_last
                stack_size = 1
              while stack_size > 0:
                stack_size -= 1
                ch_lo = stack_buf[stack_size, 0]
                ch_hi = stack_buf[stack_size, 1]
                tile_index_lo = stack_buf[stack_size, 2]
                tile_index_hi = stack_buf[stack_size, 3]
                if ch_lo + 1 == ch_hi:
                  if tile_index_lo != tile_index_hi:
                    tu, tv, mw = uvw_tile_from_index(tile_index_hi)
                    bucket_count.item_ptr(tu, tv, mw).field_ptr("count").atomic_rmw(
                      "add", 1
                    )
                    self.mask[t, bl, ch_hi] = 2
                else:
                  ch_mid = ch_lo + (ch_hi - ch_lo) // 2
                  tile_index_mid = self.uvw_tile_index(u, v, w, ch_mid)
                  if tile_index_lo != tile_index_mid:
                    stack_buf[stack_size, 0] = ch_lo
                    stack_buf[stack_size, 1] = ch_mid
                    stack_buf[stack_size, 2] = tile_index_lo
                    stack_buf[stack_size, 3] = tile_index_mid
                    stack_size += 1
                  if tile_index_mid != tile_index_hi:
                    stack_buf[stack_size, 0] = ch_mid
                    stack_buf[stack_size, 1] = ch_hi
                    stack_buf[stack_size, 2] = tile_index_mid
                    stack_buf[stack_size, 3] = tile_index_hi
                    stack_size += 1

            ch0 = ch1

    return impl

  @overload_method(
    wgridder_structref, "_prefix_sum_buckets", jit_options=full_jit_options
  )
  def overload_prefix_sum_buckets(self):
    def impl(self):
      """4.3: prefix-sum bucket_count into blockstart + ranges allocation."""
      bucket_count = self.bucket_count
      ntiles_u = bucket_count.shape[0]
      ntiles_v = bucket_count.shape[1]
      nwmin = bucket_count.shape[2]

      n_nonempty = 0
      total_ranges = np.int64(0)
      for tu in numba.prange(ntiles_u):
        for tv in numba.prange(ntiles_v):
          for mw in range(nwmin):
            c = bucket_count[tu, tv, mw]["count"]
            n_nonempty += 1 * (c > 0)
            total_ranges += c * (c > 0)

      # np.empty for Record dtypes (parfors + np.zeros trips on record dtypes);
      # every entry is fully written before the first read, so no init needed.
      blockstart = np.empty(n_nonempty, dtype=BLOCK_START_DTYPE)
      ranges = np.empty(total_ranges, dtype=SAMPLE_CHANRANGE_DTYPE)

      acc = np.int64(0)
      bsi = 0
      for tu in range(ntiles_u):
        for tv in range(ntiles_v):
          for mw in range(nwmin):
            c = bucket_count[tu, tv, mw].count
            if c > 0:
              idx_packed = index_from_uvw_tile(tu, tv, mw)
              blockstart[bsi]["uvw_tile"] = idx_packed
              blockstart[bsi]["offset"] = acc
              bsi += 1
            # Replace count with start offset; bucket_count now acts as the
            # atomic slot-allocator for Pass 2.
            bucket_count[tu, tv, mw].count = acc
            acc += c

      self.blockstart = blockstart
      self.ranges = ranges

    return impl

  @overload_method(wgridder_structref, "_fill_ranges", jit_options=full_jit_options)
  def overload_fill_ranges(self):
    def impl(self):
      """4.4 Pass 2: fill ranges[] using bucket_count as atomic slot allocator."""
      ntime = self.uvw.shape[0]
      nbl = self.uvw.shape[1]
      nchan = self.wavelengths.shape[0]
      bucket_count = self.bucket_count
      ranges = self.ranges

      for t in numba.prange(ntime):
        interbuf = np.empty((nchan + 1, 2), dtype=np.int32)
        for bl in range(nbl):
          u_raw = self.uvw[t, bl, 0]
          v_raw = self.uvw[t, bl, 1]
          w_raw = self.uvw[t, bl, 2]
          u, v, w, _ = fix_w(u_raw, v_raw, w_raw)

          interbuf_size = 0
          current_tile_index = index_from_uvw_tile(0, 0, 0)
          has_current = False
          chan0 = 0
          active = False

          for ch in range(nchan):
            xmask = self.mask[t, bl, ch]
            if xmask != 0:
              if (not active) or xmask == 2:
                new_tile_index = self.uvw_tile_index(u, v, w, ch)
                if not active:
                  active = True
                  if has_current and current_tile_index != new_tile_index:
                    # flush pending entries for previous bucket
                    tu, tv, mw = uvw_tile_from_index(current_tile_index)
                    slot = (
                      bucket_count.item_ptr(tu, tv, mw)
                      .field_ptr("count")
                      .atomic_rmw("add", interbuf_size)
                    )
                    for i in range(interbuf_size):
                      ranges[slot + i].time = numba.uint32(t)
                      ranges[slot + i].bl = numba.uint32(bl)
                      ranges[slot + i].ch_begin = numba.uint16(interbuf[i, 0])
                      ranges[slot + i].ch_end = numba.uint16(interbuf[i, 1])
                    interbuf_size = 0
                  current_tile_index = new_tile_index
                  has_current = True
                  chan0 = ch
                elif current_tile_index != new_tile_index:
                  # tile boundary mid-run
                  interbuf[interbuf_size, 0] = chan0
                  interbuf[interbuf_size, 1] = ch
                  interbuf_size += 1
                  tu, tv, mw = uvw_tile_from_index(current_tile_index)
                  slot = (
                    bucket_count.item_ptr(tu, tv, mw)
                    .field_ptr("count")
                    .atomic_rmw("add", np.int64(interbuf_size))
                  )
                  for i in range(interbuf_size):
                    ranges[slot + i].time = numba.uint32(t)
                    ranges[slot + i].bl = numba.uint32(bl)
                    ranges[slot + i].ch_begin = numba.uint16(interbuf[i, 0])
                    ranges[slot + i].ch_end = numba.uint16(interbuf[i, 1])
                  interbuf_size = 0
                  current_tile_index = new_tile_index
                  chan0 = ch
            else:
              if active:
                interbuf[interbuf_size, 0] = chan0
                interbuf[interbuf_size, 1] = ch
                interbuf_size += 1
                active = False

          if active:
            interbuf[interbuf_size, 0] = chan0
            interbuf[interbuf_size, 1] = nchan
            interbuf_size += 1

          if interbuf_size > 0 and has_current:
            tu, tv, mw = uvw_tile_from_index(current_tile_index)
            slot = (
              bucket_count.item_ptr(tu, tv, mw)
              .field_ptr("count")
              .atomic_rmw("add", np.int64(interbuf_size))
            )
            for i in range(interbuf_size):
              ranges[slot + i].time = numba.uint32(t)
              ranges[slot + i].bl = numba.uint32(bl)
              ranges[slot + i].ch_begin = numba.uint16(interbuf[i, 0])
              ranges[slot + i].ch_end = numba.uint16(interbuf[i, 1])

    return impl

  @overload_method(
    wgridder_structref, "_subdivide_blockstart", jit_options=full_jit_options
  )
  def overload_subdivide_blockstart(self):
    def impl(self):
      """4.5: subdivide blockstart so no single block exceeds 1% of per-thread
      average work."""
      blockstart = self.blockstart
      ranges = self.ranges
      n_nonempty = len(blockstart)
      total_ranges = np.int64(len(ranges))

      support = self.support
      apply_w = self.apply_w
      nthreads = numba.get_num_threads()
      support_or_1 = support if apply_w else 1
      denom = max(1, support_or_1 * nthreads)
      max_allowed = max(1, self.nvis // (denom * 100))

      # Pass A: count new entries
      new_size = 0
      for i in range(n_nonempty):
        new_size += 1
        r_start = blockstart[i]["offset"]
        r_end = blockstart[i + 1]["offset"] if i + 1 < n_nonempty else total_ranges
        acc2 = np.int64(0)
        for j in range(r_start + 1, r_end):
          acc2 += ranges[j]["ch_end"] - ranges[j]["ch_begin"]
          if acc2 > max_allowed:
            new_size += 1
            acc2 = 0

      new_blockstart = np.empty(new_size, dtype=BLOCK_START_DTYPE)

      # Pass B: fill
      bsi2 = 0
      for i in range(n_nonempty):
        uvw_tile_i = blockstart[i]["uvw_tile"]
        r_start = blockstart[i]["offset"]
        new_blockstart[bsi2]["uvw_tile"] = uvw_tile_i
        new_blockstart[bsi2]["offset"] = r_start
        bsi2 += 1
        r_end = blockstart[i + 1]["offset"] if i + 1 < n_nonempty else total_ranges
        acc2 = np.int64(0)
        for j in range(r_start + 1, r_end):
          acc2 += ranges[j]["ch_end"] - ranges[j]["ch_begin"]
          if acc2 > max_allowed:
            new_blockstart[bsi2]["uvw_tile"] = uvw_tile_i
            new_blockstart[bsi2]["offset"] = j
            bsi2 += 1
            acc2 = 0

      self.blockstart = new_blockstart

    return impl

  @overload_method(
    wgridder_structref, "_compute_uvranges", jit_options=full_jit_options
  )
  def overload_compute_uvranges(self):
    uvw_dtype = as_dtype(self.field_dict["uvw"].dtype)
    LOG2TILE = 5 if uvw_dtype == np.float32 else 4

    def impl(self):
      """4.6: compute per-w-plane u/v pixel intervals covered by the tiles."""
      blockstart = self.blockstart
      new_size = len(blockstart)
      support = self.support
      apply_w = self.apply_w
      nu = self.nu
      nv = self.nv
      nw_i = self.nw
      ntiles_u = self.bucket_count.shape[0]
      ntiles_v = self.bucket_count.shape[1]

      tmpu = np.zeros((nw_i, ntiles_u + 1), dtype=np.bool_)
      tmpv = np.zeros((nw_i, ntiles_v + 1), dtype=np.bool_)

      for bi in range(new_size):
        tu, tv, mw = uvw_tile_from_index(blockstart[bi]["uvw_tile"])
        span = support if apply_w else np.int64(1)
        for k in range(span):
          tmpu[mw + k, tu] = True
          tmpv[mw + k, tv] = True

      tilesize = np.int32(1 << LOG2TILE)
      half_support = np.int32(support // 2)

      # u-intervals: pass A counts, pass B fills
      uranges_offsets = np.zeros(nw_i + 1, dtype=np.int64)
      for pl in range(nw_i):
        lo_prev = np.int32(0)
        hi_prev = np.int32(0)
        has_prev = False
        cnt = np.int64(0)
        for j in range(ntiles_u):
          if tmpu[pl, j]:
            lo = np.int32(j) * tilesize - half_support - 1
            hi = np.int32(j + 1) * tilesize + half_support + 1
            if has_prev and lo <= hi_prev:
              hi_prev = max(hi_prev, hi)
            else:
              if has_prev:
                cnt += _count_wrapped(lo_prev, hi_prev, np.int32(nu))
              lo_prev = lo
              hi_prev = hi
              has_prev = True
        if has_prev:
          cnt += _count_wrapped(lo_prev, hi_prev, np.int32(nu))
        uranges_offsets[pl + 1] = uranges_offsets[pl] + cnt

      uranges = np.zeros((uranges_offsets[nw_i], 2), dtype=np.int32)

      for pl in range(nw_i):
        write = uranges_offsets[pl]
        lo_prev = np.int32(0)
        hi_prev = np.int32(0)
        has_prev = False
        for j in range(ntiles_u):
          if tmpu[pl, j]:
            lo = np.int32(j) * tilesize - half_support - 1
            hi = np.int32(j + 1) * tilesize + half_support + 1
            if has_prev and lo <= hi_prev:
              hi_prev = max(hi_prev, hi)
            else:
              if has_prev:
                write = _emit_wrapped(uranges, write, lo_prev, hi_prev, np.int32(nu))
              lo_prev = lo
              hi_prev = hi
              has_prev = True
        if has_prev:
          write = _emit_wrapped(uranges, write, lo_prev, hi_prev, np.int32(nu))

      self.uranges_offsets = uranges_offsets
      self.uranges = uranges

      # v-intervals (same structure)
      vranges_offsets = np.zeros(nw_i + 1, dtype=np.int64)
      for pl in range(nw_i):
        lo_prev = np.int32(0)
        hi_prev = np.int32(0)
        has_prev = False
        cnt = np.int64(0)
        for j in range(ntiles_v):
          if tmpv[pl, j]:
            lo = np.int32(j) * tilesize - half_support - 1
            hi = np.int32(j + 1) * tilesize + half_support + 1
            if has_prev and lo <= hi_prev:
              hi_prev = max(hi_prev, hi)
            else:
              if has_prev:
                cnt += _count_wrapped(lo_prev, hi_prev, np.int32(nv))
              lo_prev = lo
              hi_prev = hi
              has_prev = True
        if has_prev:
          cnt += _count_wrapped(lo_prev, hi_prev, np.int32(nv))
        vranges_offsets[pl + 1] = vranges_offsets[pl] + cnt

      vranges = np.zeros((vranges_offsets[nw_i], 2), dtype=np.int32)

      for pl in range(nw_i):
        write = vranges_offsets[pl]
        lo_prev = np.int32(0)
        hi_prev = np.int32(0)
        has_prev = False
        for j in range(ntiles_v):
          if tmpv[pl, j]:
            lo = np.int32(j) * tilesize - half_support - 1
            hi = np.int32(j + 1) * tilesize + half_support + 1
            if has_prev and lo <= hi_prev:
              hi_prev = max(hi_prev, hi)
            else:
              if has_prev:
                write = _emit_wrapped(vranges, write, lo_prev, hi_prev, np.int32(nv))
              lo_prev = lo
              hi_prev = hi
              has_prev = True
        if has_prev:
          write = _emit_wrapped(vranges, write, lo_prev, hi_prev, np.int32(nv))

      self.vranges_offsets = vranges_offsets
      self.vranges = vranges

    return impl

  @overload_method(wgridder_structref, "count_ranges", jit_options=full_jit_options)
  def overload_count_ranges(self, pixsize_x, pixsize_y):
    def impl(self, pixsize_x, pixsize_y):
      self._setup_geometry(pixsize_x, pixsize_y)
      self._count_buckets()
      self._prefix_sum_buckets()
      self._fill_ranges()
      self._subdivide_blockstart()
      self._compute_uvranges()

    return impl

  @overload_method(wgridder_structref, "scan_data", jit_options=full_jit_options)
  def overload_scan_data(self, visibilities, weight, flag):
    uvw_dtype = as_dtype(self.field_dict["uvw"].dtype)
    uvw_finfo = np.finfo(uvw_dtype)
    UVW_MIN = uvw_finfo.min
    UVW_MAX = uvw_finfo.max

    def impl(self, visibilities, weight, flag):
      ntime = self.ntime
      nbl = self.nbl
      nchan = self.nchan

      mask = np.zeros((ntime, nbl, nchan), np.uint8)
      nvis = 0
      wmin_d = UVW_MAX
      wmax_d = UVW_MIN

      for t in numba.prange(ntime):
        for bl in numba.prange(nbl):
          for ch in range(nchan):
            v = visibilities[t, bl, ch]
            if (v.real**2 + v.imag**2) * weight[t, bl, ch] * (flag[t, bl, ch]) != 0:
              mask[t, bl, ch] = 1
              nvis += 1
              w = self.effective_abs_w(t, bl, ch)
              wmin_d = min(wmin_d, w)
              wmax_d = max(wmax_d, w)

      self.mask = mask
      self.nvis = nvis
      self.w_min_d = wmin_d
      self.w_max_d = wmax_d

    return impl

  @overload_method(wgridder_structref, "max_uv", jit_options=jit_options)
  def overload_max_uv(self):
    return lambda self: max(self.u_max, self.v_max)

  @overload_attribute(wgridder_structref, "ntime", jit_options=jit_options)
  def overload_ntime(self):
    return lambda self: self.uvw.shape[0]

  @overload_attribute(wgridder_structref, "nbl", jit_options=jit_options)
  def overload_nbl(self):
    return lambda self: self.uvw.shape[1]

  @overload_attribute(wgridder_structref, "nchan", jit_options=jit_options)
  def overload_nchan(self):
    return lambda self: self.wavelengths.shape[0]

  @overload_method(wgridder_structref, "effective_uvw", jit_options=jit_options)
  def overload_effective_uvw(self, t, bl, ch):
    def impl(self, t, bl, ch):
      return (
        self.uvw[t, bl, 0] * self.wavelengths[ch],
        self.uvw[t, bl, 1] * self.wavelengths[ch],
        self.uvw[t, bl, 2] * self.wavelengths[ch],
      )

    return impl

  @overload_method(wgridder_structref, "effective_abs_w", jit_options=jit_options)
  def overload_effective_abs_w(self, t, bl, ch):
    return lambda self, t, bl, ch: abs(self.uvw[t, bl, 2] * self.wavelengths[ch])

  @overload_method(wgridder_structref, "base_uvw", jit_options=jit_options)
  def overload_base_uvw(self, t, bl):
    def impl(self, t, bl):
      return self.uvw[t, bl, 0], self.uvw[t, bl, 1], self.uvw[t, bl, 2]

    return impl

  @overload_method(wgridder_structref, "wavelength", jit_options=jit_options)
  def overload_wavelength(self, ch):
    return lambda self, ch: self.wavelengths[ch]


for template in WGRIDDER_TEMPLATES:
  _register_wgridder_overloads(template)
