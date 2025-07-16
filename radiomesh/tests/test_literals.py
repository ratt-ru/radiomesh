import pickle
from functools import reduce

import numba
from llvmlite import ir
from numba.core.errors import RequireLiteralValue
from numba.extending import intrinsic, overload

from radiomesh.literals import Datum, DatumLiteral


def test_datum_literal_name():
  assert str(DatumLiteral(Datum((2, 3, 4)))) == "DatumLiteral[tuple]((2, 3, 4))"


def test_datum_literal_pickle():
  datum_literal = DatumLiteral(Datum((2, 3, 4)))
  assert pickle.loads(pickle.dumps(datum_literal)) == datum_literal


@intrinsic
def add_datum_contents(typingctx, x, datum):
  if not isinstance(datum, DatumLiteral):
    raise RequireLiteralValue(f"{datum} is not a DatumLiteral")

  VALUE = datum.literal_value.value
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
    return add_datum_contents(x, DATUM)

  return impl


@numba.njit(cache=True)
def f(x, value):
  return f_impl(x, numba.literally(value))


def test_datum_literal():
  """Test that Datum and DatumLiteral's can be
  passed through njit, overloads and intrinsics"""

  assert f(1.0, Datum((2, 3, 4))) == 10.0
