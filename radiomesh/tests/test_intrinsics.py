import numba
import numpy as np
import pytest
from numba import types
from numba.core.errors import RequireLiteralValue
from numba.extending import intrinsic

from radiomesh.intrinsics import (
  POL_CONVERSION,
  accumulate_data,
  apply_flags,
  apply_weights,
  load_data,
  pol_to_stokes,
)
from radiomesh.literals import Datum


@pytest.mark.parametrize(
  "data, weight",
  [
    [(1.0, 2.0), 3.0],
    [(1.0, 2.0, 3.0), (3.0, 4.0, 5.0)],
    [(1 + 2j, 3 + 4j), 4.0],
    [(1 + 2j, 3 + 4j), (5 + 6j, 7 + 8j)],
  ],
)
def test_apply_weights(data, weight):
  @numba.njit
  def weight_data(d, w):
    return apply_weights(d, w)

  if not isinstance(weight, tuple):
    expected = tuple(d * weight for d in data)
  else:
    expected = tuple(d * w for d, w in zip(data, weight))

  assert weight_data(data, weight) == expected


@pytest.mark.parametrize("data, flags", [[(1.0, 2.0), (0, 1)]])
def test_apply_flags(data, flags):
  @numba.njit
  def flag_data(d, f):
    return apply_flags(d, f)

  assert flag_data(data, flags) == tuple(
    d if f == 0.0 else 0.0 for d, f in zip(data, flags)
  )


def test_load_data():
  shape = (5, 4)

  @numba.njit
  def load(a, i):
    return load_data(a, (i,), shape[1], -1)

  data = np.arange(np.prod(shape)).reshape(shape)

  for i in range(shape[0]):
    assert load(data, i) == tuple(i * shape[1] + j for j in range(shape[1]))


def test_accumulate_data():
  shape = (5, 4)

  @numba.njit
  def accumulate(d, a, i):
    return accumulate_data(d, a, (i,), shape[1], -1)

  data = np.zeros(shape)

  for i in range(shape[0]):
    accumulate((i,) * shape[1], data, i)
    accumulate((i,) * shape[1], data, i)

  assert np.all(np.broadcast_to(np.arange(shape[0])[:, None], shape) * 2 == data)


@pytest.mark.parametrize(
  "pols,stokes",
  [
    (("XX", "XY", "YX", "YY"), ("I", "Q", "U", "V")),
    (("RR", "RL", "LR", "LL"), ("I", "Q", "U", "V")),
    (("XX", "YY"), ("I", "Q")),
    (("RR", "LL"), ("I", "V")),
  ],
)
def test_pol_conversion(pols, stokes):
  """Test that converting from polarisation to stokes works.
  This depends on correctness of the conversion routines in POL_CONVERSION"""
  POL_DATUM = Datum(pols)
  STOKES_DATUM = Datum(stokes)

  @numba.njit
  def convert(t):
    return pol_to_stokes(t, POL_DATUM, STOKES_DATUM)

  mapping = []

  for s in stokes:
    for (p1, p2), fn in POL_CONVERSION[s].items():
      try:
        p1i = pols.index(p1)
        p2i = pols.index(p2)
      except Exception:
        continue
      else:
        mapping.append((p1i, p2i, fn))

  values = np.random.random(len(pols)) + np.random.random(len(pols)) * 1j
  expected = tuple(fn(values[p1], values[p2]) for p1, p2, fn in mapping)
  assert convert(tuple(values)) == expected


@intrinsic
def add_intrinsic(typingctx, array, value):
  """Adds a literal value to an array"""
  if not isinstance(value, types.IntegerLiteral):
    raise RequireLiteralValue(f"{value}")

  if not isinstance(array, types.Array):
    raise TypeError(f"{array}")

  av = float(value.literal_value) + 0.1
  sig = array.copy(dtype=typingctx.unify_types(array.dtype, types.float64))(
    array, value
  )

  def codegen(context, builder, sig, args):
    return context.compile_internal(builder, lambda a, v: a + av, sig, args)

  return sig, codegen


@numba.njit(cache=True, nogil=True)
def jitted_intrinsic(a, v):
  """It's useful to search for any references to this function in the assembly output"""
  return add_intrinsic(a, numba.literally(v))


def test_intrinsic_caching():
  """This test case isn't that interesting in terms of comparing values,
  but is useful for testing intrinsic caching in conjunction with NUMBA_DEBUG_CACHE=1"""

  @numba.njit(cache=True, nogil=True)
  def g(a, v):
    return jitted_intrinsic(a, v)

  @numba.njit(cache=True, nogil=True)
  def h(a, v):
    return add_intrinsic(a, numba.literally(v))

  assert g(np.ones(1), 1).item() == 2.1

  np.testing.assert_array_almost_equal(g(np.ones(10), 1) + 1, h(np.ones(10), 2))
  np.testing.assert_array_almost_equal(h(np.ones(10), 5) - 3, h(np.ones(10), 2))


def test_none_type_intrinsic():
  @intrinsic
  def fintrinsic(typingctx, data, gains):
    sig = types.none(data, gains)
    print(sig, gains == types.none)

    def codegen(context, builder, signature, args):
      return None

    return sig, codegen

  @numba.njit(nogil=True)
  def f(a, b=None):
    return fintrinsic(a, b)

  f((1, 2, 3))
