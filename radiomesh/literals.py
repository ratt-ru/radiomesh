from __future__ import annotations

from typing import Any, Callable, Generic, Hashable, Tuple, TypeVar

from numba.core import types
from numba.core.datamodel.models import OpaqueModel, register_default
from numba.core.imputils import lower_cast
from numba.core.types import StructRef
from numba.cpython.unicode import make_string_from_constant
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


class BooleanDatumLiteral(DatumLiteral):
  pass


class IntegerDatumLiteral(DatumLiteral):
  pass


class FloatDatumLiteral(DatumLiteral):
  pass


class StringDatumLiteral(DatumLiteral):
  pass


@lower_cast(IntegerDatumLiteral, types.Integer)
@lower_cast(FloatDatumLiteral, types.Float)
def number_datum_literal_to_constant(context, builder, fromty, toty, val):
  lit = context.get_constant_generic(builder, fromty.literal_type, fromty.literal_value)
  return context.cast(builder, lit, fromty.literal_type, toty)


@lower_cast(BooleanDatumLiteral, types.Boolean)
def boolean_datum_literal_to_constant(context, builder, fromty, toty, val):
  lit = context.get_constant_generic(builder, fromty.literal_type, fromty.literal_value)
  return context.is_true(builder, fromty.literal_type, lit)


@lower_cast(StringDatumLiteral, types.unicode_type)
def string_datum_literal_to_constant(context, builder, fromty, toty, val):
  return make_string_from_constant(context, builder, toty, fromty.literal_value)


def is_datum_literal(obj, typ):
  """Return True if obj is a DatumLiteral holding a Datum of the given typ"""
  return isinstance(obj, DatumLiteral) and isinstance(obj.literal_value, typ)


@unbox(DatumLiteral)
@unbox(FloatDatumLiteral)
@unbox(IntegerDatumLiteral)
@unbox(BooleanDatumLiteral)
@unbox(StringDatumLiteral)
def unbox_datum_literal(typ, obj, c):
  """Convert a Python DatumLiteral to a Numba representation"""
  return NativeValue(c.context.get_dummy_value())


def _from_datum(datum: Datum):
  """Converts from a Datum to a DatumLiteral"""
  value = datum.value

  if isinstance(value, float):
    return FloatDatumLiteral(value)
  elif isinstance(value, bool):
    return BooleanDatumLiteral(value)
  elif isinstance(value, int):
    return IntegerDatumLiteral(value)
  elif isinstance(value, str):
    return StringDatumLiteral(value)
  else:
    return DatumLiteral(value)


@typeof_impl.register(Datum)
def typeof_datum(datum: Datum, c):
  """This is sufficient to use Datum within a numba.njit function"""
  return _from_datum(datum)


# DatumLiteral is only implemented as a simple Literal and Dummy type
# in order to pass arbitrary Python objects through overload and intrinsic constructs.
# It's functionality is minimally exposed within the numba layer so we
# only register it with an OpaqueModel.
register_default(DatumLiteral)(OpaqueModel)
register_default(BooleanDatumLiteral)(OpaqueModel)
register_default(FloatDatumLiteral)(OpaqueModel)
register_default(IntegerDatumLiteral)(OpaqueModel)
register_default(StringDatumLiteral)(OpaqueModel)

# This ensures numba.literally(Datum(...)) produces a DatumLiteral
types.Literal.ctor_map[Datum] = _from_datum


@overload_attribute(DatumLiteral, "literal_value")
@overload_attribute(BooleanDatumLiteral, "literal_value")
@overload_attribute(IntegerDatumLiteral, "literal_value")
@overload_attribute(FloatDatumLiteral, "literal_value")
@overload_attribute(StringDatumLiteral, "literal_value")
def overload_datum_value(self):
  """Returns the literal_value of a DatumLiteral"""
  if isinstance(self, DatumLiteral):
    VALUE = self.literal_value
    return lambda self: VALUE


class LiteralStructRef(StructRef):
  """StructRef base that extracts Literal-typed fields into a
  ``_literal_values`` dict and mangles the type name to include them,
  enabling compile-time specialisation on literal values."""

  def __init__(self, fields):
    super().__init__(fields)
    literals = tuple(
      sorted(
        (name, typ.literal_value)
        for name, typ in fields
        if isinstance(typ, types.Literal)
      )
    )
    self.name = f"{self.name}{literals}"
    self._literal_values = dict(literals)

  def preprocess_fields(self, fields):
    return tuple((n, types.unliteral(t)) for n, t in fields)

  def get_literal(self, name: str, default: Any = None) -> Any:
    return self._literal_values.get(name, default)
