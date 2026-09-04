"""Exercises the "StructRef implementation strategy" example from
``design/data-model.md`` to confirm the documented pattern is accurate.

The implementation here is a faithful transcription of the design doc: a
``LiteralStructRef`` subclass, a ``StructRefProxy`` whose ``@numba.njit``
methods defer to overloaded attributes/methods, an ``@overload`` constructor
that specialises on a ``DatumLiteral`` field and derives a field type via
``cpu_target`` unification, plus ``@overload_attribute`` and
``@overload_method`` hooks (the latter constant-folding on a literal field).
"""

from numbers import Number

import numba
import pytest
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.experimental import structref
from numba.extending import overload, overload_attribute, overload_method

from radiomesh.literals import Datum, LiteralStructRef, is_datum_literal

# Prefer numpy error model and dropping the GIL
JIT_OPTIONS = {
  "error_model": "numpy",
  "nogil": True,
  # "parallel": True
}


@structref.register
class DataStructRef(LiteralStructRef):
  """The numba type.

  By inheriting from LiteralStructRef overloads can potentially
  specialise on any literals supplied as constructor arguments."""

  pass


class Data(structref.StructRefProxy):
  """Exposes the struct within Python"""

  def __new__(cls, a, b):
    a_datum = Datum(a)  # Example of specialising on a's type
    return structref.StructRefProxy.__new__(cls, a_datum, b)

  @numba.njit
  def d(self):
    """Defers to overloaded d attribute below"""
    return self.d

  @numba.njit
  def a_plus_x(self, x):
    """Defers to overloaded a_plus_x method below"""
    return self.a_plus_x(x)


# Required for the proxy to box a DataStructRef back to Python
structref.define_boxing(DataStructRef, Data)


@overload(Data, prefer_literal=True, jit_options=JIT_OPTIONS)
def overload_data_constructor(a, b):
  """Construct a Data struct within jitted code"""
  # a is expected to be a (float) DatumLiteral
  if not is_datum_literal(a, float):
    return None

  # field c type derived from a and b for cpu targets;
  # reject overload if a and b cannot be unified
  if (c := cpu_target.typing_context.unify_types(a, b)) is None:
    return None

  # Define struct fields. Remember these names as attributes.
  struct_type = DataStructRef([("a", a), ("b", b), ("c", c)])

  def impl(a, b):
    """Implement the Data constructor"""
    instance = structref.new(struct_type)
    instance.a = a
    instance.b = b
    instance.c = a + b
    return instance

  return impl


@overload_attribute(DataStructRef, "d", inline="always", jit_options=JIT_OPTIONS)
def overload_data_d(self):
  """Provide obj.d as an attribute on Data structs within jitted code"""
  return lambda self: self.a + 1  # Avoid defining impl for short functions


@overload_method(DataStructRef, "a_plus_x", jit_options=JIT_OPTIONS)
def overload_a_plus_x(self, x):
  """Provide obj.a_plus_x(x) as a method on Data structs within jitted code"""
  if isinstance(A := self.get_literal("a"), Number):
    # Provide a constant folded version if a is a literal
    # Prefer capitals when embedding constants in closures
    return lambda self, x: A + x
  # Fallback to variable
  return lambda self, x: self.a + x


@numba.njit
def get_c(data):
  """Read the unify-derived ``c`` field from within jit code."""
  return data.c


def test_data_struct_ref():
  """The documented DataStructRef pattern constructs and exposes its
  overloaded attribute/method through the proxy."""
  data = Data(2.0, 3.0)

  # d -> self.a + 1
  assert data.d() == 3.0
  # a_plus_x -> A + x, constant-folded because the proxy wraps a as a literal
  assert data.a_plus_x(10.0) == 12.0
  # c -> a + b, on a field whose type was derived via cpu_target unification
  assert get_c(data) == 5.0


def test_data_struct_ref_rejects_non_float_a():
  """The constructor overload only matches when ``a`` is a float
  DatumLiteral; a non-float ``a`` leaves no matching overload."""
  with pytest.raises(TypingError):
    Data("not a float", 3.0)
