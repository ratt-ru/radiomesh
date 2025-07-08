import numba
import numpy as np
import pytest
from numba import types
from numba.core.errors import RequireLiteralValue
from numba.extending import intrinsic

from radiomesh.intrinsics import (
  POL_CONVERSION,
  accumulate_data_factory,
  apply_weight_factory,
  load_data_factory,
  pol_to_stokes_factory,
)


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
  apply_weight_intrinsic = apply_weight_factory(len(data))

  @numba.njit
  def apply_weights(d, w):
    return apply_weight_intrinsic(d, w)

  if not isinstance(weight, tuple):
    expected = tuple(d * weight for d in data)
  else:
    expected = tuple(d * w for d, w in zip(data, weight))

  assert apply_weights(data, weight) == expected


@pytest.mark.parametrize("data, flags", [[(1.0, 2.0), (0, 1)]])
def test_apply_flags(data, flags):
  apply_flags_intrinsic = apply_weight_factory(len(data), flags=True)

  @numba.njit
  def apply_flags(d, f):
    return apply_flags_intrinsic(d, f)

  assert apply_flags(data, flags) == tuple(
    d if f == 0.0 else 0.0 for d, f in zip(data, flags)
  )


def test_load_data():
  shape = (5, 4)
  load_data = load_data_factory(shape[1])

  @numba.njit
  def load(a, i):
    return load_data(a, (i,))

  data = np.arange(np.prod(shape)).reshape(shape)

  for i in range(shape[0]):
    assert load(data, i) == tuple(i * shape[1] + j for j in range(shape[1]))


def test_accumulate_data():
  shape = (5, 4)
  accumulate_data = accumulate_data_factory(shape[1])

  @numba.njit
  def accumulate(d, a, i):
    return accumulate_data(d, a, (i,))

  data = np.zeros(shape)

  for i in range(shape[0]):
    accumulate((i,) * shape[1], data, i)
    accumulate((i,) * shape[1], data, i)

  assert np.all(np.broadcast_to(np.arange(shape[0])[:, None], shape) * 2 == data)


@pytest.mark.parametrize(
  "pols,stokes",
  [
    (["XX", "XY", "YX", "YY"], ["I", "Q", "U", "V"]),
    (["RR", "RL", "LR", "LL"], ["I", "Q", "U", "V"]),
    (["XX", "YY"], ["I", "Q"]),
    (["RR", "LL"], ["I", "V"]),
  ],
)
def test_pol_conversion(pols, stokes):
  """Test that converting from polarisation to stokes works.
  This depends on correctness of the conversion routines in POL_CONVERSION"""
  convert_intrinsic = pol_to_stokes_factory(pols, stokes)

  @numba.njit
  def convert(t):
    return convert_intrinsic(t)

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

  av = float(value.literal_value)

  def codegen(context, builder, sig, args):
    return context.compile_internal(builder, lambda a, v: a + av, sig, args)

  return array(array, value), codegen


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

  np.testing.assert_array_equal(g(np.ones(10), 1) + 1, h(np.ones(10), 2))
