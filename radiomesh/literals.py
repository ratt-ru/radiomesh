from __future__ import annotations

from typing import Any, Callable, Tuple

from numba.core import types
from numba.core.datamodel.models import OpaqueModel, register_default
from numba.extending import (
  NativeValue,
  typeof_impl,
  unbox,
)


class Datum:
  """A simple class holding a value of any type"""

  __slots__ = ("value",)

  value: Any

  def __init__(self, value: Any):
    self.value = value

  def __eq__(self, other) -> bool:
    if isinstance(other, Datum):
      return self.value == other.value
    return NotImplemented

  def __hash__(self) -> int:
    return hash(self.value)

  def __reduce__(self) -> Tuple[Callable[[Datum], Any], Any]:
    return (Datum, (self.value,))

  def __str__(self) -> str:
    return str(self.value)

  def __repr__(self) -> str:
    return repr(self.value)


class DatumLiteral(types.Literal, types.Dummy):
  """Numba literal type holding an arbitrary Datum object"""

  def __init__(self, value: Datum):
    if not isinstance(value, Datum):
      raise TypeError(f"{value} of type {type(value)} should be a Datum")

    self._literal_init(value)
    super(types.Dummy, self).__init__(
      name=f"DatumLiteral[{type(value.value).__name__}]({value})"
    )


@unbox(DatumLiteral)
def unbox_datum_literal(typ, obj, c):
  """Convert a Python DatumLiteral to a Numba representation
  Here we can just the Python DatumLiteral itself"""
  return NativeValue(c.context.get_dummy_value())


@typeof_impl.register(Datum)
def typeof_index(val, c):
  """This is sufficient to use Datum within a numba.njit function"""
  return DatumLiteral(val)


# DatumLiteral is only implemented as a simple Literal and Dummy type
# in order to pass arbitrary Python objects through overload and intrinsic constructs.
# It's functionality is minimally exposed within the numba layer so we
# only register it with an OpaqueModel.
register_default(DatumLiteral)(OpaqueModel)

# This ensures numba.literally(Datum(...)) produces a DatumLiteral
types.Literal.ctor_map[Datum] = DatumLiteral
