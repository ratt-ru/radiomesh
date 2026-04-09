import multiprocessing as mp
import pickle
from functools import reduce

import numba
from llvmlite import ir
from numba.core.errors import RequireLiteralValue
from numba.extending import intrinsic, overload

from radiomesh.literals import Datum, DatumLiteral, is_datum_literal
from radiomesh.tests.proc_utils import _init_numba_cache_debugging_with_capture


def test_is_datum_literal():
  assert is_datum_literal(DatumLiteral(4.0), float)


def test_datum_literal_name():
  assert str(DatumLiteral((2, 3, 4))) == "DatumLiteral[tuple]((2, 3, 4))"


def test_datum_literal_pickle():
  datum_literal = DatumLiteral((2, 3, 4))
  assert pickle.loads(pickle.dumps(datum_literal)) == datum_literal


@intrinsic
def add_datum_contents(typingctx, x, datum):
  if not isinstance(datum, DatumLiteral):
    raise RequireLiteralValue(f"{datum} is not a DatumLiteral")

  VALUE = datum.literal_value
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

  def impl(x, datum):
    return add_datum_contents(x, datum)

  return impl


@numba.njit(cache=True)
def f(x, value):
  return f_impl(x, numba.literally(value))


def test_datum_literal():
  """Test that Datum and DatumLiteral's can be
  passed through njit, overloads and intrinsics"""

  assert f(1.0, Datum((2, 3, 4))) == 10.0


def test_datum_literal_jit():
  value = 4.0
  datum = Datum(value)

  @numba.njit
  def fn():
    return datum.literal_value

  assert fn() == value


def _caching_worker(x, datum):
  @numba.njit(cache=True, nogil=True)
  def fn(x):
    return f_impl(x, datum)

  return fn(x)


def test_datum_caching(tmp_path):
  """Tests that Datum/DatumLiterals can be cached"""
  stdout_f = tmp_path / "stdout.txt"
  stderr_f = tmp_path / "stderr.txt"
  datum = Datum((1, 2, 3))

  with mp.get_context("spawn").Pool(
    1,
    initializer=_init_numba_cache_debugging_with_capture,
    initargs=(str(tmp_path), str(stdout_f), str(stderr_f)),
  ) as p:
    assert p.apply(_caching_worker, args=(0.5, datum)) == 6.5
    assert p.apply(_caching_worker, args=(0.5, datum)) == 6.5

  combined = stdout_f.read_text() + stderr_f.read_text()
  assert f"data saved to '{tmp_path}" in combined
  assert f"data loaded from '{tmp_path}" in combined
  assert f"index loaded from '{tmp_path}" in combined
