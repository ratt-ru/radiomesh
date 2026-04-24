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
from numba.np.numpy_support import as_dtype

from radiomesh.constants import CACHE_LINE_SIZE, LIGHTSPEED
from radiomesh.numba_utils import make_structref_property

# A 64 bit cache aligned integer counter
CachedAlignedCounter = types.Record(
  [("count", {"type": types.int64, "offset": 0})], size=CACHE_LINE_SIZE, aligned=True
)


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


@structref.register
class WGridderImplStructRef(StructRef):
  def preprocess_fields(self, fields):
    """Disallow literal types in field definitions"""
    return tuple((n, types.unliteral(t)) for n, t in fields)


@structref.register
class ParallelWGridderImplStructRef(WGridderImplStructRef):
  pass


class WGridderImpl(structref.StructRefProxy):
  def __new__(cls, uvw, frequencies, invert_u=False, invert_v=False, invert_w=False):
    return structref.StructRefProxy.__new__(
      cls, uvw, frequencies, invert_u, invert_v, invert_w
    )

  uvw = make_structref_property("uvw")
  wavelengths = make_structref_property("wavelengths")
  u_min = make_structref_property("u_min")
  u_max = make_structref_property("u_max")
  w_min_d = make_structref_property("w_min_d")
  w_max_d = make_structref_property("w_max_d")
  mask = make_structref_property("mask")
  nvis = make_structref_property("nvis")


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
    return {"nogil": True, "cache": True}

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
  def overload_gridding_impl(uvw, frequencies, invert_u, invert_v, invert_w):
    struct_type = wgridder_structref(
      [
        ("uvw", uvw),
        ("wavelengths", frequencies),
        ("u_max", uvw.dtype),
        ("v_max", uvw.dtype),
        ("w_min_d", uvw.dtype),
        ("w_max_d", uvw.dtype),
        ("mask", numba.types.Array(numba.uint8, 3, "C")),
        ("nvis", numba.uint64),
      ]
    )

    def impl(uvw, frequencies, invert_u, invert_v, invert_w):
      for i in range(frequencies.shape[0]):
        if frequencies[i] < 0.0:
          raise NotImplementedError("negative frequencies")

        if i > 0 and frequencies[i - 1] >= frequencies[i]:
          raise NotImplementedError("Frequencies that do not increase monotically")

      obj = structref.new(struct_type)
      obj.uvw = uvw.copy()
      obj.wavelengths = frequencies / LIGHTSPEED
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

      return obj

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
