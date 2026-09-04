from typing import Mapping

import numba
import numpy as np
from numba.experimental import structref
from numba.extending import overload

from radiomesh.literals import LiteralStructRef

JIT_OPTIONS = {"nogil": True, "error_model": "numpy"}


class AbstractRimeTermStructRef(LiteralStructRef):
  pass


class AbstractRimeTermStructRefProxy(structref.StructRefProxy):
  pass


@structref.register
class CoreArgumentsStructRef(LiteralStructRef):
  pass


class CoreArguments(structref.StructRefProxy):
  def __new__(
    cls,
    time,
    uvw,
    frequency,
    baseline_antenna1_name,
    baseline_antenna2_name,
  ):
    antenna_names = np.concat((baseline_antenna1_name, baseline_antenna2_name))
    _, inv = np.unique(antenna_names, return_inverse=True)
    antenna1 = inv[: len(antenna_names) // 2]
    antenna2 = inv[len(antenna_names) // 2 :]

    return structref.StructRefProxy.__new__(
      cls,
      time,
      uvw,
      frequency,
      baseline_antenna1_name,
      baseline_antenna2_name,
      antenna1,
      antenna2,
    )

  @property
  @numba.njit
  def time(self):
    return self.time

  @property
  @numba.njit
  def antenna1(self):
    return self.antenna1

  @property
  @numba.njit
  def antenna2(self):
    return self.antenna2


structref.define_boxing(CoreArgumentsStructRef, CoreArguments)


@overload(CoreArguments, jit_options=JIT_OPTIONS)
def overload_core_args_constructor(
  time,
  uvw,
  frequency,
  baseline_antenna1_name,
  baseline_antenna2_name,
  antenna1,
  antenna2,
):
  struct_type = CoreArgumentsStructRef(
    [
      ("time", time),
      ("uvw", uvw),
      ("frequency", frequency),
      ("baseline_antenna1_name", baseline_antenna1_name),
      ("baseline_antenna2_name", baseline_antenna2_name),
      ("antenna1", antenna1),
      ("antenna2", antenna2),
    ]
  )

  def impl(
    time,
    uvw,
    frequency,
    baseline_antenna1_name,
    baseline_antenna2_name,
    antenna1,
    antenna2,
  ):
    obj = structref.new(struct_type)
    obj.time = time
    obj.uvw = uvw
    obj.frequency = frequency
    obj.baseline_antenna1_name = baseline_antenna1_name
    obj.baseline_antenna2_name = baseline_antenna2_name
    obj.antenna1 = antenna1
    obj.antenna2 = antenna2
    return obj

  return impl


def rime(specification: str, dataset: Mapping):
  time = dataset["time"]
  frequency = dataset["frequency"]
