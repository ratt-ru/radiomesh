import numba


def test_register_jitable_literal():
  from numba.extending import overload, register_jitable

  @register_jitable(inline="always")
  def half(x):
    return x // 2

  def fimpl(x):
    pass

  @overload(fimpl, prefer_literal="True")
  def fimpl_overload(x):
    # assert isinstance(x, types.Literal)
    def impl(x):
      for i in numba.literal_unroll(range(100)):
        if i == x:
          HALF = half(i)
          acc = 0

          for j in numba.literal_unroll(range(HALF)):
            acc += j

          return acc

      return -1

    return impl

  @numba.njit
  def f(x):
    return fimpl(x)

  print(f(15))
  print(next(iter(f.inspect_llvm().values())))


def test_register_jitable_literal_side_effect():
  import numpy as np
  from numba.extending import overload, register_jitable

  @register_jitable(inline="always")
  def half(x):
    return x // 2

  def fimpl(x, out):
    pass

  @overload(fimpl, prefer_literal="True")
  def fimpl_overload(x, out):
    def impl(x, out):
      for i in numba.literal_unroll(range(100)):
        if i == x:
          HALF = half(i)
          acc = 0
          for j in numba.literal_unroll(range(HALF)):
            out[j] = j * 3 + 1  # opaque side effect
            acc += j
          return acc
      return -1

    return impl

  @numba.njit
  def f(x, out):
    return fimpl(x, out)

  out = np.zeros(64, dtype=np.int64)
  print(f(15, out))
  print(next(iter(f.inspect_llvm().values())))
