"""pytest breaks on asserts in numba code without this: PYTEST_DONT_REWRITE"""

import numba
import numpy as np
from numba import types


def test_pointers():
  RecordType = types.Record(
    [
      ("count1", {"type": types.uint64, "offset": 0}),
      ("count2", {"type": types.uint64, "offset": types.uint64.bitwidth // 8}),
    ],
    size=64,
    aligned=True,
  )

  @numba.njit
  def do_test():
    counters = np.zeros((3, 3), dtype=RecordType)
    A = np.zeros((3, 3), dtype=np.uint64)
    counters.item_ptr(1, 2).field_ptr("count1").atomic_inc()
    assert counters[1, 2]["count1"] == 1
    A.item_ptr(1, 1).atomic_inc()
    assert A[1, 1] == 1

    B = np.zeros((3, 3), dtype=np.float64)
    B.item_ptr(1, 1).atomic_inc()

  do_test()
