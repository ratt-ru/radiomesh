import numba
import numpy as np
import pytest

from radiomesh.es_kernel import ESKernel, es_kernel_positions, eval_es_kernel
from radiomesh.literals import Datum


@pytest.mark.parametrize("modulo", [True, False])
def test_es_positions_intrinsic(modulo):
  """Test that position_intrinsic returns a tuple of floats
  around the given index"""
  KERNEL = Datum(ESKernel())
  assert KERNEL.value.support > 0

  N = 1024
  x = 2

  @numba.njit
  def fn(ps):
    return es_kernel_positions(KERNEL, N, ps, modulo)

  if modulo:
    assert fn(x) == tuple(((x + o) % N) for o in range(KERNEL.value.support))
  else:
    assert fn(x) == tuple((x + o for o in range(KERNEL.value.support)))


def test_es_kernel_intrinsic():
  """Test the ES kernel evaluation intrinsic"""
  KERNEL = Datum(ESKernel())
  HALF_SUPPORT_INT = KERNEL.value.half_support_int

  @numba.njit
  def fn(u):
    ps = int(np.round(u)) - HALF_SUPPORT_INT
    return eval_es_kernel(KERNEL, u, ps)

  u = 2.3
  kernel_values = fn(u)
  assert sum(int(kv > 0) for kv in kernel_values) == len(kernel_values) - 1
