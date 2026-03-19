"""pytest breaks on asserts in numba code without this: PYTEST_DONT_REWRITE"""

import numba
import numpy as np
from numba import types


def test_pointers():
  """Test atomic operations on arrays for integer primitives,
  as well as integers in records"""
  RecordType = types.Record(
    [
      ("count1", {"type": types.uint64, "offset": 0}),
      ("count2", {"type": types.uint64, "offset": types.uint64.bitwidth // 8}),
    ],
    size=64,
    aligned=True,
  )

  @numba.njit(nogil=True)
  def do_test():
    A = np.zeros((3, 3), dtype=np.uint64)
    assert A.item_ptr(1, 1).atomic_rmw("add", 1) == 0
    assert A[1, 1] == 1
    assert A.item_ptr(1, 1).atomic_rmw("sub", 1) == 1
    assert A[1, 1] == 0

    counters = np.zeros((3, 3), dtype=RecordType)
    assert counters.item_ptr(1, 2).field_ptr("count1").atomic_rmw("add", 1) == 0
    assert counters[1, 2]["count1"] == 1
    assert counters.item_ptr(1, 2).field_ptr("count1").atomic_rmw("sub", 1) == 1
    assert counters[1, 2]["count2"] == 0

  do_test()
