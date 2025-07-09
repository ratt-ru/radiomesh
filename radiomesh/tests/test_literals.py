from functools import reduce

import numba
import pytest
from llvmlite import ir
from numba import types
from numba.core.errors import RequireLiteralValue
from numba.extending import intrinsic, overload

from radiomesh.literals import Datum, DatumLiteral


def test_datum_literal_name():
  assert str(DatumLiteral(Datum([2, 3, 4]))) == "DatumLiteral[list]([2, 3, 4])"


def test_datum_literal():
  """Test that Datum and DatumLiteral's can be
  passed through njit, overloads and intrinsics"""

  @intrinsic
  def add_datum_contents(typingctx, x, datum):
    if not isinstance(datum, DatumLiteral):
      raise RequireLiteralValue(f"{datum} is not a DatumLiteral")

    VALUE = datum.literal_value.value
    assert isinstance(VALUE, list)
    sig = x(x, datum)

    def codegen(context, builder, sig, args):
      x, _ = args
      x_type, _ = sig.args
      llvm_float_type = context.get_value_type(x_type)
      consts = [ir.Constant(llvm_float_type, v) for v in VALUE]
      return reduce(builder.fadd, consts, x)

    return sig, codegen

  def f_impl(x, datum):
    pass

  @overload(f_impl)
  def f_overload(x, datum):
    if not isinstance(datum, DatumLiteral):
      raise RequireLiteralValue(f"{datum} is not DatumLiteral")

    DATUM = datum.literal_value

    def impl(x, datum):
      return add_datum_contents(x, numba.literally(DATUM))

    return impl

  @numba.njit
  def f(x, value):
    return f_impl(x, numba.literally(value))

  assert f(1.0, Datum([2, 3, 4])) == 10.0


@pytest.fixture
def float_literal_cls():
  from radiomesh.literals import install_float_literal

  FloatLiteral = install_float_literal()

  yield FloatLiteral
  del types.Literal.ctor_map[float]


def test_float_literal_overload(float_literal_cls):
  """Test that a FloatLiteral is recognised by @overload"""
  # The type is registered with the base Literal constructor map
  assert types.Literal.ctor_map[float] is float_literal_cls

  float_value = 10.0

  def f_impl(value):
    pass

  @overload(f_impl)
  def f_overload(value):
    if not isinstance(value, float_literal_cls):
      raise RequireLiteralValue(f"value {value} must be a FloatLiteral")
    assert value.literal_value == float_value
    return lambda value: None

  @numba.njit
  def f(value):
    return f_impl(numba.literally(value))

  f(float_value)
