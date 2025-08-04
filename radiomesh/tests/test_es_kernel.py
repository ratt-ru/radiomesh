import numba
import numpy as np

from radiomesh.es_kernel import ESKernel, es_kernel_positions, eval_es_kernel
from radiomesh.literals import Datum


def test_es_positions_intrinsic():
  """Test that position_intrinsic returns a tuple of floats
  around the given index"""
  KERNEL = Datum(ESKernel())
  assert KERNEL.value.support > 0

  N = 1024
  x = 2

  @numba.njit
  def fn(ps):
    return es_kernel_positions(KERNEL, N, ps)

  assert fn(x) == tuple(((x + o) % N) for o in range(KERNEL.value.support))


def test_es_kernel_intrinsic():
  """Test the ES kernel evaluation intrinsic"""
  KERNEL = Datum(ESKernel())
  HALF_SUPPORT_INT = KERNEL.value.half_support_int

  N = 1024

  @numba.njit
  def fn(u):
    ps = int(np.round(u)) - HALF_SUPPORT_INT
    k = es_kernel_positions(KERNEL, N, ps)
    return eval_es_kernel(KERNEL, k, u, ps)

  u = 2.3
  kernel_values = fn(u)
  assert sum(int(kv > 0) for kv in kernel_values) == 8
