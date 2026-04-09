from __future__ import annotations

from typing import Any, Callable, Generic, Hashable, Tuple, TypeVar

from numba.core import types
from numba.core.datamodel.models import OpaqueModel, register_default
from numba.core.errors import TypingError
from numba.extending import NativeValue, overload_attribute, typeof_impl, unbox


class Schema(tuple):
  """An extension of tuple that exists primarily to
  represent sequences of stokes/polarisations.

  This creates a unique type that numba can pass as a
  literal argument within numba functions"""

  pass


class SchemaLiteral(types.Literal, types.Dummy):
  """Literal type associated with Schema"""

  def __reduce__(self):
    return (SchemaLiteral, (self.literal_value,))


@unbox(SchemaLiteral)
def unbox_schema_literal(typ, obj, c):
  """Convert a Python SchemaLiteral to a Numba representation
  Here we can just the Python SchemaLiteral itself"""
  return NativeValue(c.context.get_dummy_value())


@typeof_impl.register(Schema)
def typeof_schema(val, c):
  """This is sufficient to use Schema within a numba.njit function"""
  return SchemaLiteral(val)


# SchemaLiteral is only implemented as a simple Literal and Dummy type
# in order to pass arbitrary Python objects through overload and intrinsic constructs.
# It's functionality is minimally exposed within the numba layer so we
# only register it with an OpaqueModel.
register_default(SchemaLiteral)(OpaqueModel)

# This ensures numba.literally(Schema(...)) produces a SchemaLiteral
types.Literal.ctor_map[Schema] = SchemaLiteral


H = TypeVar("H", bound=Hashable)


class Datum(Generic[H]):
  """A simple class holding an immutable value of any hashable type"""

  __slots__ = ("value", "hashvalue")

  value: H

  def __init__(self, value: H):
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


class DatumLiteral(Generic[H], types.Literal, types.Dummy):
  """Numba literal type holding an arbitrary object"""

  def __init__(self, value: H):
    name = f"DatumLiteral[{type(value).__name__}]({value})"
    super(types.Dummy, self).__init__(name=name)
    self._literal_init(value)

  @staticmethod
  def from_datum(datum: Datum[H]):
    return DatumLiteral(datum.value)


def is_datum_literal(obj, typ):
  """Return True if obj is a DatumLiteral holding a Datum of the given typ"""
  return isinstance(obj, DatumLiteral) and isinstance(obj.literal_value, typ)


@unbox(DatumLiteral)
def unbox_datum_literal(typ, obj, c):
  """Convert a Python DatumLiteral to a Numba representation
  Here we can just the Python DatumLiteral itself"""
  return NativeValue(c.context.get_dummy_value())


@typeof_impl.register(Datum)
def typeof_datum(val, c):
  """This is sufficient to use Datum within a numba.njit function"""
  return DatumLiteral(val.value)


# DatumLiteral is only implemented as a simple Literal and Dummy type
# in order to pass arbitrary Python objects through overload and intrinsic constructs.
# It's functionality is minimally exposed within the numba layer so we
# only register it with an OpaqueModel.
register_default(DatumLiteral)(OpaqueModel)

# This ensures numba.literally(Datum(...)) produces a DatumLiteral
types.Literal.ctor_map[Datum] = DatumLiteral.from_datum


@overload_attribute(DatumLiteral, "literal_value")
def overload_datum_value(self):
  if not isinstance(self, DatumLiteral):
    raise TypingError(f"{self} is not a DatumLiteral")

  """Returns the literal_value of a DatumLiteral"""
  VALUE = self.literal_value
  return lambda self: VALUE
