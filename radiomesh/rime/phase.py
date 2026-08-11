import numba
import numpy as np
from numba.experimental import structref
from numba.extending import overload, overload_method
from numba.np.numpy_support import as_dtype

from radiomesh.constants import LIGHTSPEED
from radiomesh.rime.core import (
  AbstractRimeTermStructRef,
  AbstractRimeTermStructRefProxy,
)

JIT_OPTIONS = {"nogil": True, "error_model": "numpy"}


@structref.register
class PhaseTermStructRef(AbstractRimeTermStructRef):
  pass


class PhaseTerm(AbstractRimeTermStructRefProxy):
  def __new__(cls, phase_centre, radec):
    return AbstractRimeTermStructRefProxy.__new__(cls, phase_centre, radec)

  @property
  @numba.njit
  def lmnm1(self):
    return self.lmnm1

  @numba.njit
  def sample(self, core_args, s, t, bl, ch):
    return self.sample(core_args, s, t, bl, ch)


@overload(PhaseTerm, jit_options=JIT_OPTIONS)
def overload_phase_term(phase_centre, radec):
  struct_type = PhaseTermStructRef([("lmnm1", radec)])

  def impl(phase_centre, radec):
    obj = structref.new(struct_type)
    nsrc, _ = radec.shape
    obj.lmnm1 = np.empty((nsrc, 3), radec.dtype)

    zero = radec.dtype.type(0.0)
    one = radec.dtype.type(1.0)
    pc_ra, pc_dec = phase_centre
    sin_pc_dec = np.sin(pc_dec)
    cos_pc_dec = np.cos(pc_dec)

    for s in numba.prange(nsrc):
      da = radec[s, 0] - pc_ra
      sin_ra_delta = np.sin(da)
      cos_ra_delta = np.cos(da)
      sin_dec = np.sin(radec[s, 1])
      cos_dec = np.cos(radec[s, 1])
      obj.lmnm1[s, 0] = l = cos_dec * sin_ra_delta  # noqa
      obj.lmnm1[s, 1] = m = sin_dec * cos_pc_dec - cos_dec * sin_pc_dec * cos_ra_delta
      n = one - l**2 - m**2
      obj.lmnm1[s, 3] = np.sqrt(n * (n > zero)) - one

    return obj

  return impl


structref.define_boxing(PhaseTermStructRef, PhaseTerm)


@overload_method(PhaseTermStructRef, "sample", inline="always", jit_options=JIT_OPTIONS)
def overload_phase_term_sample(self, core_args, s, t, bl, ch):
  uvw_dtype = as_dtype(core_args.field_dict["uvw"].dtype)
  C = uvw_dtype.type(-2.0 * np.pi / LIGHTSPEED)

  def impl(self, core_args, s, t, bl, ch):
    real_phase = (
      C
      * core_args.frequency[ch]
      * (
        core_args.uvw[t, bl, 0] * self.lmnm1[s, 0]
        + core_args.uvw[t, bl, 1] * self.lmnm1[s, 1]
        + core_args.uvw[t, bl, 2] * self.lmnm1[s, 2]
      )
    )
    return np.cos(real_phase) + np.sin(real_phase) * 1j

  return impl
