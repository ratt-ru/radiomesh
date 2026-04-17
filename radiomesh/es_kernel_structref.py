import math

import numba
import numpy as np
from numba import types
from numba.experimental import structref
from numba.extending import overload, overload_method, register_jitable

from radiomesh.literals import Datum, LiteralStructRef, is_datum_literal
from radiomesh.numba_utils import make_structref_property


@register_jitable
def generate_poly_coeffs(support, beta, e0, degree):
  """Generate polynomial approximation coefficients for the ES kernel.

  The ES kernel ``exp(beta * support * ((1 - v^2)^e0 - 1))`` is approximated on
  ``[-1, 1]`` by ``support`` ``degree`` polynomials, one per sub-interval.

  Args:
    support: kernel support.
    beta: beta parameter.
    e0: exponent parameter.
    degree: polynomial degree.

  Returns:
    Array of shape ``(degree+1, support)``.
    ``coeffs[j, i]`` is the coefficient of ``x^(degree-j)`` for sub-interval ``i``
    (Horner order: index 0 is the leading / highest-power coefficient).
  """
  betak = beta * support

  # Chebyshev nodes on [-1, 1]
  i_arr = np.arange(degree + 1, dtype=np.float64)
  chebroot = np.cos((2.0 * i_arr + 1.0) * math.pi / (2.0 * degree + 2.0))

  # coeff[j, i] in output order (j = Horner step, i = sub-interval)
  coeff = np.zeros((degree + 1, support), dtype=np.float64)

  # Chebyshev-to-monomial conversion: C[j, k] = coeff of x^k in T_j(x)
  C = np.zeros((degree + 1, degree + 1), dtype=np.float64)
  C[0, 0] = 1.0
  if degree >= 1:
    C[1, 1] = 1.0
  for j in range(2, degree + 1):
    C[j, 0] = -C[j - 2, 0]
    for k in range(1, j + 1):
      C[j, k] = 2.0 * C[j - 1, k - 1] - C[j - 2, k]

  # Precompute cosine matrix for DCT: cos_mat[j, k] = cos(j*(2k+1)*pi/(2D+2))
  cos_mat = np.empty((degree + 1, degree + 1), dtype=np.float64)
  for j in range(degree + 1):
    for k in range(degree + 1):
      cos_mat[j, k] = math.cos(j * (2.0 * k + 1.0) * math.pi / (2.0 * degree + 2.0))

  for i in range(support):
    left = -1.0 + 2.0 * i / support
    right = -1.0 + 2.0 * (i + 1) / support

    # Function values at Chebyshev nodes mapped to [left, right]
    nodes = chebroot * (right - left) * 0.5 + (right + left) * 0.5
    y = np.empty(degree + 1, dtype=np.float64)
    for j in range(degree + 1):
      v = nodes[j]
      tmp = (1.0 - v) * (1.0 + v)
      if tmp < 0.0:
        y[j] = 0.0
      else:
        y[j] = math.exp(betak * (tmp**e0 - 1.0))

    avg = np.sum(y) / (degree + 1)
    y -= avg

    # Chebyshev coefficients via DCT-I-like sum
    lcf = np.empty(degree + 1, dtype=np.float64)
    for j in range(degree + 1):
      s = 0.0
      for k in range(degree + 1):
        s += cos_mat[j, k] * y[k]
      lcf[j] = (2.0 / (degree + 1)) * s
    lcf[0] *= 0.5

    # lcf2[k] = coefficient of x^k in the combined polynomial
    lcf2 = np.empty(degree + 1, dtype=np.float64)
    for k in range(degree + 1):
      s = 0.0
      for j in range(degree + 1):
        s += C[j, k] * lcf[j]
      lcf2[k] = s
    lcf2[0] += avg

    # Store in Horner order: coeff[j, i] = lcf2[D-j]
    for j in range(degree + 1):
      coeff[j, i] = lcf2[degree - j]

  return coeff


@structref.register
class ESKernelStructRef(LiteralStructRef):
  """ESKernel StructRef"""

  def literal_kernel_params(self):
    if (
      isinstance(support := self.get_literal("support"), int)
      and isinstance(beta := self.get_literal("beta"), float)
      and isinstance(e0 := self.get_literal("e0"), float)
    ):
      return (support, beta, e0)

    return False


class ESKernelProxy(structref.StructRefProxy):
  def __new__(
    cls,
    epsilon: float | Datum[float] = 2e-13,
    oversampling: float | Datum[float] = 2.0,
    beta: float | Datum[float] = 2.3,
    e0: float | Datum[float] = 0.5,
    support: int | Datum[int] = -1,
    analytic: bool | Datum[bool] = True,
    single: bool | Datum[bool] = True,
    apply_w: bool | Datum[bool] = True,
  ):
    return structref.StructRefProxy.__new__(
      cls,
      epsilon,
      oversampling,
      beta,
      e0,
      support,
      Datum(analytic) if not isinstance(analytic, Datum) else analytic,
      single,
      apply_w,
    )

  epsilon = make_structref_property("epsilon")
  oversampling = make_structref_property("oversampling")
  beta = make_structref_property("beta")
  e0 = make_structref_property("e0")
  support = make_structref_property("support")
  analytic = make_structref_property("analytic")
  single = make_structref_property("single")
  apply_w = make_structref_property("apply_w")

  @classmethod
  def fully_specified(
    cls,
    epsilon: float = 2e-13,
    oversampling: float = 2.0,
    beta: float = 2.3,
    e0: float = 0.5,
    support: int = -1,
    analytic=True,
    single=True,
    apply_w=True,
  ):
    return ESKernelProxy(
      Datum(epsilon),
      Datum(oversampling),
      Datum(beta),
      Datum(e0),
      Datum(support),
      Datum(analytic),
      Datum(single),
      Datum(apply_w),
    )


structref.define_boxing(ESKernelStructRef, ESKernelProxy)


@overload(ESKernelProxy, prefer_literal=True)
def overload_es_kernel(
  epsilon, oversampling, beta, e0, support, analytic, single, apply_w
):
  """Implement the ESKernel constructor"""
  ANALYTIC = is_datum_literal(analytic, bool) and analytic.literal_value is True

  fields = [
    ("epsilon", epsilon),
    ("oversampling", oversampling),
    ("beta", beta),
    ("e0", e0),
    ("support", support),
    ("analytic", analytic),
    ("single", single),
    ("apply_w", apply_w),
  ]

  if not ANALYTIC:
    fields.append(("coeffs", types.float64[:, :]))

  state_type = ESKernelStructRef(fields)

  def impl(epsilon, oversampling, beta, e0, support, analytic, single, apply_w):
    instance = structref.new(state_type)
    instance.epsilon = epsilon
    instance.oversampling = oversampling
    instance.beta = beta
    instance.e0 = e0
    instance.analytic = analytic
    instance.single = single
    instance.apply_w = apply_w

    if support <= 0:
      ndim = 3.0 if apply_w else 2.0
      instance.support = int(math.ceil(math.log10(ndim * 1.0 / epsilon))) + 1
    else:
      instance.support = support

    if not ANALYTIC:
      instance.coeffs = generate_poly_coeffs(support, beta, e0, support + 3)

    return instance

  return impl


@overload_method(ESKernelStructRef, "evaluate")
def overload_evaluate(self, x):
  if (kernel_params := self.literal_kernel_params()) is not False:
    SUPPORT, BETA, E0 = kernel_params
    HALF_SUPPORT = SUPPORT / 2.0
    BETAK = SUPPORT * BETA

    if self.get_literal("analytic") is True:
      if E0 == 0.5:

        def impl(self, x):
          x = x / HALF_SUPPORT
          tmp = 1.0 - x * x
          safe_tmp = max(tmp, 0.0)
          return math.exp(BETAK * (math.sqrt(safe_tmp) - 1.0)) * (tmp > 0.0)
      else:

        def impl(self, x):
          x = x / HALF_SUPPORT
          tmp = 1.0 - x * x
          safe_tmp = max(tmp, 0.0)
          return math.exp(BETAK * (math.pow(safe_tmp, E0) - 1.0)) * (tmp > 0.0)

    else:
      COEFFS = generate_poly_coeffs(SUPPORT, BETA, E0, SUPPORT + 3)
      NCOEFFS = len(COEFFS)

      def impl(self, x):
        x = x / HALF_SUPPORT
        if abs(x) >= 1:
          return 0.0

        xrel = SUPPORT * 0.5 * (x + 1.0)
        nth = min(int(xrel), SUPPORT - 1)
        locx = ((xrel - nth) - 0.5) * 2.0
        value = COEFFS[0][nth]
        for i in numba.literal_unroll(range(1, NCOEFFS)):
          value = value * locx + COEFFS[i][nth]
        return value

  else:
    if self.get_literal("analytic") is True:

      def impl(self, x):
        half_support = self.support / 2.0
        x = x / half_support
        tmp = 1.0 - x * x
        safe_tmp = max(tmp, 0.0)
        return math.exp(
          self.beta * self.support * (math.pow(safe_tmp, self.e0) - 1.0)
        ) * (tmp > 0.0)

    else:

      def impl(self, x):
        half_support = self.support / 2.0
        x = x / half_support
        if abs(x) >= 1:
          return 0.0
        xrel = self.support * 0.5 * (x + 1.0)
        nth = min(int(xrel), self.support - 1)
        locx = ((xrel - nth) - 0.5) * 2.0
        value = self.coeffs[0, nth]
        for i in range(1, self.coeffs.shape[0]):
          value = value * locx + self.coeffs[i, nth]
        return value

  return impl


@overload_method(ESKernelStructRef, "evaluate_support")
def overload_evaluate_support(self, grid, pixel_start, out):
  if (kernel_params := self.literal_kernel_params()) is not False:
    SUPPORT, BETA, E0 = kernel_params
    HALF_SUPPORT = SUPPORT / 2.0
    BETAK = SUPPORT * BETA

    if self.get_literal("analytic") is True:
      if E0 == 0.5:

        def impl(self, grid, pixel_start, out):
          for offset in range(self.support):
            x = (offset + pixel_start - grid) / HALF_SUPPORT
            tmp = 1.0 - x * x
            safe_tmp = max(tmp, 0.0)
            out[offset] = math.exp(BETAK * (math.sqrt(safe_tmp) - 1.0)) * (tmp > 0.0)

      else:

        def impl(self, grid, pixel_start, out):
          for offset in range(self.support):
            x = (offset + pixel_start - grid) / HALF_SUPPORT
            tmp = 1.0 - x * x
            safe_tmp = max(tmp, 0.0)
            out[offset] = math.exp(BETAK * (math.pow(safe_tmp, E0) - 1.0)) * (tmp > 0.0)

    else:
      COEFFS = generate_poly_coeffs(SUPPORT, BETA, E0, SUPPORT + 3)
      NCOEFFS = len(COEFFS)

      def impl(self, grid, pixel_start, out):
        for offset in range(self.support):
          x = (offset + pixel_start - grid) / HALF_SUPPORT
          if abs(x) >= 1:
            out[offset] = 0.0
          else:
            xrel = SUPPORT * 0.5 * (x + 1.0)
            nth = min(int(xrel), SUPPORT - 1)
            locx = ((xrel - nth) - 0.5) * 2.0
            value = COEFFS[0][nth]
            for i in range(1, NCOEFFS):
              value = value * locx + COEFFS[i][nth]
            out[offset] = value

  else:
    if self.get_literal("analytic") is True:

      def impl(self, grid, pixel_start, out):
        half_support = self.support / 2.0
        for offset in range(self.support):
          x = (offset + pixel_start - grid) / half_support
          tmp = 1.0 - x * x
          safe_tmp = max(tmp, 0.0)
          out[offset] = math.exp(
            self.beta * self.support * (math.pow(safe_tmp, self.e0) - 1.0)
          ) * (tmp > 0.0)

    else:

      def impl(self, grid, pixel_start, out):
        half_support = self.support / 2.0
        for offset in range(self.support):
          x = (offset + pixel_start - grid) / half_support
          if abs(x) >= 1:
            out[offset] = 0.0
          else:
            xrel = self.support * 0.5 * (x + 1.0)
            nth = min(int(xrel), self.support - 1)
            locx = ((xrel - nth) - 0.5) * 2.0
            value = self.coeffs[0, nth]
            for i in range(1, self.coeffs.shape[0]):
              value = value * locx + self.coeffs[i, nth]
            out[offset] = value

  return impl
