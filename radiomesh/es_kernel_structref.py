import math

import numba
import numpy as np
from numba import types
from numba.core.errors import RequireLiteralValue
from numba.core.types import StructRef
from numba.experimental import structref
from numba.extending import overload, overload_method, register_jitable


@register_jitable
def generate_poly_coeffs(support, beta, e0):
  """Generate polynomial approximation coefficients for the ES kernel.

  The ES kernel ``exp(beta * support * ((1 - v^2)^e0 - 1))`` is approximated on
  ``[-1, 1]`` by ``support`` degree-``D`` polynomials, one per sub-interval.

  Args:
    support: kernel support.
    beta: beta parameter.
    e0: exponent parameter.

  Returns:
    Array of shape ``(D+1, support)`` where ``D = support + 3``.
    ``coeffs[j, i]`` is the coefficient of ``x^(D-j)`` for sub-interval ``i``
    (Horner order: index 0 is the leading / highest-power coefficient).
  """
  D = support + 3
  betak = beta * support

  # Chebyshev nodes on [-1, 1]
  i_arr = np.arange(D + 1, dtype=np.float64)
  chebroot = np.cos((2.0 * i_arr + 1.0) * math.pi / (2.0 * D + 2.0))

  # coeff[j, i] in output order (j = Horner step, i = sub-interval)
  coeff = np.zeros((D + 1, support), dtype=np.float64)

  # Chebyshev-to-monomial conversion: C[j, k] = coeff of x^k in T_j(x)
  C = np.zeros((D + 1, D + 1), dtype=np.float64)
  C[0, 0] = 1.0
  if D >= 1:
    C[1, 1] = 1.0
  for j in range(2, D + 1):
    C[j, 0] = -C[j - 2, 0]
    for k in range(1, j + 1):
      C[j, k] = 2.0 * C[j - 1, k - 1] - C[j - 2, k]

  # Precompute cosine matrix for DCT: cos_mat[j, k] = cos(j*(2k+1)*pi/(2D+2))
  cos_mat = np.empty((D + 1, D + 1), dtype=np.float64)
  for j in range(D + 1):
    for k in range(D + 1):
      cos_mat[j, k] = math.cos(j * (2.0 * k + 1.0) * math.pi / (2.0 * D + 2.0))

  for i in range(support):
    left = -1.0 + 2.0 * i / support
    right = -1.0 + 2.0 * (i + 1) / support

    # Function values at Chebyshev nodes mapped to [left, right]
    nodes = chebroot * (right - left) * 0.5 + (right + left) * 0.5
    y = np.empty(D + 1, dtype=np.float64)
    for j in range(D + 1):
      v = nodes[j]
      tmp = (1.0 - v) * (1.0 + v)
      if tmp < 0.0:
        y[j] = 0.0
      else:
        y[j] = math.exp(betak * (tmp**e0 - 1.0))

    avg = np.sum(y) / (D + 1)
    y -= avg

    # Chebyshev coefficients via DCT-I-like sum
    lcf = np.empty(D + 1, dtype=np.float64)
    for j in range(D + 1):
      s = 0.0
      for k in range(D + 1):
        s += cos_mat[j, k] * y[k]
      lcf[j] = (2.0 / (D + 1)) * s
    lcf[0] *= 0.5

    # lcf2[k] = coefficient of x^k in the combined polynomial
    lcf2 = np.empty(D + 1, dtype=np.float64)
    for k in range(D + 1):
      s = 0.0
      for j in range(D + 1):
        s += C[j, k] * lcf[j]
      lcf2[k] = s
    lcf2[0] += avg

    # Store in Horner order: coeff[j, i] = lcf2[D-j]
    for j in range(D + 1):
      coeff[j, i] = lcf2[D - j]

  return coeff


@structref.register
class ESKernelStructRef(StructRef):
  """ESKernel StructRef"""

  def preprocess_fields(self, fields):
    """Disallow literal types in field definitions"""
    return fields
    return tuple((n, types.unliteral(t)) for n, t in fields)


@numba.njit
def es_kernel_ctor(epsilon, oversampling, beta, e0, support, analytic, single, apply_w):
  return ESKernelProxy(
    epsilon,
    oversampling,
    beta,
    e0,
    support,
    numba.literally(analytic),
    numba.literally(single),
    numba.literally(apply_w),
  )


class ESKernelProxy(structref.StructRefProxy):
  def __new__(
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
    return es_kernel_ctor(
      epsilon, oversampling, beta, e0, support, analytic, single, apply_w
    )


structref.define_boxing(ESKernelStructRef, ESKernelProxy)


@overload(ESKernelProxy, prefer_literal=True)
def overload_es_kernel(
  epsilon, oversampling, beta, e0, support, analytic, single, apply_w
):
  """Implement the ESKernel constructor"""
  if not isinstance(analytic, types.BooleanLiteral):
    raise RequireLiteralValue(f"analytic {analytic} must be a Boolean Literal")

  if not isinstance(single, types.BooleanLiteral):
    raise RequireLiteralValue(f"single {single} must be a Boolean Literal")

  if not isinstance(apply_w, types.BooleanLiteral):
    raise RequireLiteralValue(f"apply_w {apply_w} must be a Boolean Literal")

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

  if (ANALYTIC := analytic.literal_value) is False:
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
      support = int(math.ceil(math.log10(ndim * 1.0 / epsilon))) + 1

    instance.support = support

    if not ANALYTIC:
      instance.coeffs = generate_poly_coeffs(support, beta, e0)

    return instance

  return impl


@overload_method(ESKernelStructRef, "evaluate")
def overload_evaluate(self, x):
  if self.field_dict["analytic"].literal_value is True:
    # Define an analytic implementation
    def impl(self, x):
      pass
  else:
    # Define a polynomial implemenation
    def impl(self, x):
      pass

  return impl
