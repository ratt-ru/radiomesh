from numba import types
from numba.core.types import StructRef
from numba.experimental import structref
from numba.extending import overload_attribute, overload_classmethod, overload_method

from radiomesh.constants import CACHE_LINE_SIZE

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
