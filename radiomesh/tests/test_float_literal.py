import numba
import pytest
from numba.core import types
from numba.core.boxing import NativeValue, unbox
from numba.core.datamodel.models import OpaqueModel, register_default
from numba.cpython.builtins import impl_ret_untracked, lower_builtin


class FloatLiteral(types.Literal, types.Dummy):
  pass


@unbox(FloatLiteral)
def unbox_float_literal(typ, obj, c):
  return NativeValue(c.context.get_dummy_value())


@lower_builtin(float, FloatLiteral)
def float_literal_impl(context, builder, sig, args):
  [ty] = sig.args
  res = context.get_constant(sig.return_type, float(ty.literal_value))
  return impl_ret_untracked(context, builder, sig.return_type, res)


register_default(FloatLiteral)(OpaqueModel)


@numba.njit
def f(value):
  print(numba.literally(value))


@pytest.fixture
def install_float_literal():
  types.Literal.ctor_map[float] = FloatLiteral
  yield
  del types.Literal.ctor_map[float]


@pytest.mark.xfail(reason="Needs more work for this to succeed")
def test_float_literal(with_float_literal):
  v = FloatLiteral(10.0)
  f(v)
