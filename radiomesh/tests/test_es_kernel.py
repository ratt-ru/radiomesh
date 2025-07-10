import numba
import numpy as np

from radiomesh.es_kernel import ESKernel, es_kernel_positions, eval_es_kernel
from radiomesh.literals import Datum


def test_es_positions_intrinsic():
  """Test that position_intrinsic returns a tuple of floats
  around the given index"""
  KERNEL = Datum(ESKernel())
  assert KERNEL.value.support > 0
  assert KERNEL.value.offsets == tuple(
    float(p) - KERNEL.value.half_support for p in range(KERNEL.value.support)
  )

  @numba.njit
  def fn(i):
    return es_kernel_positions(KERNEL, i)

  assert fn(2) == tuple(o + 2.0 for o in KERNEL.value.offsets)


def test_es_kernel_intrinsic():
  """Test the ES kernel evaluation intrinsic"""
  KERNEL = Datum(ESKernel())

  @numba.njit
  def fn(u):
    i = int(np.round(u))
    p = es_kernel_positions(KERNEL, i)
    return p, eval_es_kernel(KERNEL, p, u)

  u = 2.3
  pos, kernel_values = fn(u)
  assert all(kv > 0 for kv in kernel_values)
  # assert kernel_values == tuple(map(kernel.kernel_fn, pos, [u] * len(pos)))
