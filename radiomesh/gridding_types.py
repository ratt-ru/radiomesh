from numba import types
from numba.core.types import StructRef
from numba.experimental import structref
from numba.extending import (
  overload_attribute,
  overload_classmethod,
  overload_method,
)

from radiomesh.constants import CACHE_LINE_SIZE, LIGHTSPEED

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


# A 64 bit cache aligned integer counter
CachedAlignedCounter = types.Record(
  [("count", {"type": types.int64, "offset": 0})], size=CACHE_LINE_SIZE, aligned=True
)

# Align with cache line to avoid false sharing during atomic operations
UvwTile = types.Record(
  [("index", {"type": types.uint64, "offset": 0})], size=CACHE_LINE_SIZE, aligned=True
)


@overload_method(types.Record, "set_index")
def overload_set_index(self, u_tile, v_tile, w_tile):
  if self == UvwTile:

    def impl(self, u_tile, v_tile, w_tile):
      self.index = (u_tile & TILE_MASK) << U_TILE_BIT_OFFSET
      self.index |= (v_tile & TILE_MASK) << V_TILE_BIT_OFFSET
      self.index |= (w_tile & TILE_MASK) << W_TILE_BIT_OFFSET

    return impl


@overload_attribute(types.Record, "u_tile")
def overload_record_u_tile(self):
  if self == UvwTile:
    return lambda self: (self.index & U_TILE_MASK) >> U_TILE_BIT_OFFSET


@overload_attribute(types.Record, "v_tile")
def overload_record_v_tile(self):
  if self == UvwTile:
    return lambda self: (self["index"] & V_TILE_MASK) >> V_TILE_BIT_OFFSET


@overload_attribute(types.Record, "w_tile")
def overload_record_w_tile(self):
  if self == UvwTile:
    return lambda self: (self["index"] & W_TILE_MASK) >> W_TILE_BIT_OFFSET


# StructUvwTile
@structref.register
class StructUvwTile(StructRef):
  def preprocess_fields(self, fields):
    """Disallow literal types in field definitions"""
    return tuple((n, types.unliteral(t)) for n, t in fields)


StructUvwTileType = StructUvwTile(fields=[("index", types.uint64)])


@overload_classmethod(StructUvwTile, "from_uvw")
def overload_from_uvw(cls, u_tile, v_tile, w_tile):
  """Implement a from_uvw classmethod to avoid the overhead of creating
  a full constructor via a StructRefProxy"""

  def impl(cls, u_tile, v_tile, w_tile):
    obj = structref.new(StructUvwTileType)
    obj.index = (u_tile & TILE_MASK) << U_TILE_BIT_OFFSET
    obj.index |= (v_tile & TILE_MASK) << V_TILE_BIT_OFFSET
    obj.index |= (w_tile & TILE_MASK) << W_TILE_BIT_OFFSET
    return obj

  return impl


@overload_attribute(StructUvwTile, "u_tile")
def overload_u_tile(self):
  return lambda self: (self.index & U_TILE_MASK) >> U_TILE_BIT_OFFSET


@overload_attribute(StructUvwTile, "v_tile")
def overload_v_tile(self):
  return lambda self: (self.index & V_TILE_MASK) >> V_TILE_BIT_OFFSET


@overload_attribute(StructUvwTile, "w_tile")
def overload_w_tile(self):
  return lambda self: (self.index & W_TILE_MASK) >> W_TILE_BIT_OFFSET


@structref.register
class Baselines(StructRef):
  def preprocess_fields(self, fields):
    """Disallow literal types in field definitions"""
    return tuple((n, types.unliteral(t)) for n, t in fields)


BaselinesType = Baselines(
  fields=[
    ("uvw", types.float64[:, :, :]),
    ("wavelengths", types.float64[:]),
    ("u_max", types.float64),
    ("v_max", types.float64),
  ]
)


@overload_classmethod(Baselines, "from_args")
def overload_baseline_from_args(
  cls, uvw, frequencies, invert_u=False, invert_v=False, invert_w=False
):
  def impl(cls, uvw, frequencies, invert_u=False, invert_v=False, invert_w=False):
    for i in range(frequencies.shape[0]):
      if frequencies[i] < 0.0:
        raise NotImplementedError("negative frequencies")

      if i > 0 and frequencies[i - 1] >= frequencies[i]:
        raise NotImplementedError("Frequencies that do not increase monotically")

    obj = structref.new(BaselinesType)
    obj.uvw = uvw.copy()
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

    obj.wavelengths = frequencies / LIGHTSPEED
    wavelength_max = obj.wavelengths.max()
    obj.u_max *= wavelength_max
    obj.v_max *= wavelength_max

    return obj

  return impl


@overload_method(Baselines, "max_uv")
def overload_max_uv(self):
  return lambda self: max(self.u_max, self.v_max)


@overload_attribute(Baselines, "ntime")
def overload_ntime(self):
  return lambda self: self.uvw.shape[0]


@overload_attribute(Baselines, "nbl")
def overload_nbl(self):
  return lambda self: self.uvw.shape[1]


@overload_attribute(Baselines, "nchan")
def overload_nchan(self):
  return lambda self: self.wavelengths.shape[0]


@overload_method(Baselines, "effective_uvw")
def overload_effective_uvw(self, t, bl, ch):
  def impl(self, t, bl, ch):
    return (
      self.uvw[t, bl, 0] * self.wavelengths[ch],
      self.uvw[t, bl, 1] * self.wavelengths[ch],
      self.uvw[t, bl, 2] * self.wavelengths[ch],
    )

  return impl


@overload_method(Baselines, "effective_abs_w")
def overload_effective_abs_w(self, t, bl, ch):
  return lambda self, t, bl, ch: abs(self.uvw[t, bl, 2] * self.wavelengths[ch])


@overload_method(Baselines, "base_uvw")
def overload_base_uvw(self, t, bl):
  def impl(self, t, bl):
    return self.uvw[t, bl, 0], self.uvw[t, bl, 1], self.uvw[t, bl, 2]

  return impl


@overload_method(Baselines, "wavelength")
def overload_wavelength(self, ch):
  return lambda self, ch: self.wavelengths[ch]
