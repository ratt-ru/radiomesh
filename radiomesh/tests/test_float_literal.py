import numba
import pytest
from numba import types
from numba.core.errors import RequireLiteralValue
from numba.extending import overload


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
