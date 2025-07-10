from __future__ import annotations

from typing import Any, Callable, Hashable, Tuple

from numba.core import types
from numba.core.datamodel.models import OpaqueModel, register_default
from numba.extending import (
  NativeValue,
  typeof_impl,
  unbox,
)


class Datum:
  """A simple class holding an immutable value of any hashable type"""

  __slots__ = ("value", "hashvalue")

  value: Hashable

  def __init__(self, value: Hashable):
    self.value = value
    try:
      self.hashvalue = hash(value)
    except (ValueError, TypeError) as e:
      raise ValueError(f"{value} must be hashable") from e

  def __eq__(self, other) -> bool:
    if isinstance(other, Datum):
      return self.value == other.value
    return NotImplemented

  def __hash__(self) -> int:
    return self.hashvalue

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
      raise TypeError(f"{value} of type {type(value)} must be a Datum")

    name = f"DatumLiteral[{type(value.value).__name__}]({value})"
    super(types.Dummy, self).__init__(name=name)
    self._literal_init(value)

  def __reduce__(self):
    return (DatumLiteral, (self.literal_value,))

  def __eq__(self, other):
    if not isinstance(other, DatumLiteral):
      return NotImplemented

    return self.literal_value == other.literal_value

  def __hash__(self):
    return hash(self.literal_value)

  @property
  def datum_value(self) -> Any:
    """Returns the value wrapped in the underlying Datum object"""
    return self.literal_value.value


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
