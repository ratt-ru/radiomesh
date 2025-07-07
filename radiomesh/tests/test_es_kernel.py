import numba
import numpy as np

from radiomesh.es_kernel import ESKernel


def test_es_positions_intrinsic():
  """Test that position_intrinsic returns a tuple of floats
  around the given index"""
  kernel = ESKernel()
  es_pos = kernel.position_intrinsic
  assert kernel.support > 0
  assert kernel.offsets == tuple(
    float(p) - kernel.half_support for p in range(kernel.support)
  )

  @numba.njit
  def fn(i):
    return es_pos(i)

  assert fn(2) == tuple(o + 2.0 for o in kernel.offsets)


def test_es_kernel_intrinsic():
  """Test the ES kernel evaluation intrinsic"""
  kernel = ESKernel()
  es_pos = kernel.position_intrinsic
  eval_es_kernel = kernel.kernel_intrinsic

  @numba.njit
  def fn(u):
    i = int(np.round(u))
    p = es_pos(i)
    return p, eval_es_kernel(p, u)

  u = 2.3
  pos, kernel_values = fn(u)
  assert kernel_values == tuple(map(kernel.kernel_fn, pos, [u] * len(pos)))
