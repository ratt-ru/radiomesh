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

from radiomesh.constants import CACHE_LINE_SIZE, LIGHTSPEED

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
class GriddingMetadataStructRef(StructRef):
  def preprocess_fields(self, fields):
    """Disallow literal types in field definitions"""
    return tuple((n, types.unliteral(t)) for n, t in fields)


class GriddingMetadata(structref.StructRefProxy):
  def __new__(cls, uvw, frequencies, invert_u=False, invert_v=False, invert_w=False):
    return structref.StructRefProxy.__new__(
      cls, uvw, frequencies, invert_u, invert_v, invert_w
    )


structref.define_boxing(GriddingMetadataStructRef, GriddingMetadata)


@overload(GriddingMetadata)
def overload_gridding_metadata(uvw, frequencies, invert_u, invert_v, invert_w):
  struct_type = GriddingMetadataStructRef(
    [
      ("uvw", uvw),
      ("wavelengths", frequencies),
      ("u_max", uvw.dtype),
      ("v_max", uvw.dtype),
    ]
  )

  def impl(uvw, frequencies, invert_u, invert_v, invert_w):
    for i in range(frequencies.shape[0]):
      if frequencies[i] < 0.0:
        raise NotImplementedError("negative frequencies")

      if i > 0 and frequencies[i - 1] >= frequencies[i]:
        raise NotImplementedError("Frequencies that do not increase monotically")

    obj = structref.new(struct_type)
    obj = structref.new(struct_type)
    obj.uvw = uvw.copy()
    obj.wavelengths = frequencies / LIGHTSPEED
    obj.u_max = 0.0
    obj.v_max = 0.0
    u_sign = uvw.dtype.type(-1.0 if invert_u else 1.0)
    v_sign = uvw.dtype.type(-1.0 if invert_v else 1.0)
    w_sign = uvw.dtype.type(-1.0 if invert_w else 1.0)

    for t in range(uvw.shape[0]):
      for bl in range(uvw.shape[1]):
        obj.uvw[t, bl, 0] *= u_sign
        obj.uvw[t, bl, 1] *= v_sign
        obj.uvw[t, bl, 2] *= w_sign

        obj.u_max = max(obj.u_max, abs(obj.uvw[t, bl, 0]))
        obj.v_max = max(obj.u_max, abs(obj.uvw[t, bl, 1]))

    wavelength_max = obj.wavelengths.max()
    obj.u_max *= wavelength_max
    obj.v_max *= wavelength_max

    return obj

  return impl


@overload_method(GriddingMetadataStructRef, "max_uv")
def overload_max_uv(self):
  return lambda self: max(self.u_max, self.v_max)


@overload_attribute(GriddingMetadataStructRef, "ntime")
def overload_ntime(self):
  return lambda self: self.uvw.shape[0]


@overload_attribute(GriddingMetadataStructRef, "nbl")
def overload_nbl(self):
  return lambda self: self.uvw.shape[1]


@overload_attribute(GriddingMetadataStructRef, "nchan")
def overload_nchan(self):
  return lambda self: self.wavelengths.shape[0]


@overload_method(GriddingMetadataStructRef, "effective_uvw")
def overload_effective_uvw(self, t, bl, ch):
  def impl(self, t, bl, ch):
    return (
      self.uvw[t, bl, 0] * self.wavelengths[ch],
      self.uvw[t, bl, 1] * self.wavelengths[ch],
      self.uvw[t, bl, 2] * self.wavelengths[ch],
    )

  return impl


@overload_method(GriddingMetadataStructRef, "effective_abs_w")
def overload_effective_abs_w(self, t, bl, ch):
  return lambda self, t, bl, ch: abs(self.uvw[t, bl, 2] * self.wavelengths[ch])


@overload_method(GriddingMetadataStructRef, "base_uvw")
def overload_base_uvw(self, t, bl):
  def impl(self, t, bl):
    return self.uvw[t, bl, 0], self.uvw[t, bl, 1], self.uvw[t, bl, 2]

  return impl


@overload_method(GriddingMetadataStructRef, "wavelength")
def overload_wavelength(self, ch):
  return lambda self, ch: self.wavelengths[ch]
