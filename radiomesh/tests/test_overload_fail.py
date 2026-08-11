from collections import namedtuple

import numba
import numpy as np
from numba.extending import overload

Data = namedtuple("Data", ["x", "y"])


def make_data(x, y):
  raise NotImplementedError


@overload(make_data)
def ol_make_data(x, y):
  return lambda x, y: Data(x, y)


@numba.njit(parallel=True, nogil=True)
def fn(a):
  data = make_data(np.arange(10), np.arange(20))

  accum = a.dtype.type(0)

  for i in numba.prange(a.shape[0]):
    accum += a[i]

  return accum


if __name__ == "__main__":
  fn(np.arange(10))
