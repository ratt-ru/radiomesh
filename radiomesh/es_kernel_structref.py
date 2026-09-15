from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import numba
import numpy as np
from numba.core import errors, types
from numba.experimental import structref
from numba.extending import (
  overload,
  overload_attribute,
  overload_method,
  register_jitable,
)

from radiomesh.literals import Datum, LiteralStructRef, is_datum_literal
from radiomesh.numba_utils import make_structref_property
from radiomesh.simd import widest_simd_register_bits

if TYPE_CHECKING:
  import numpy.typing as npt


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


@register_jitable(inline="always")
def polynomial_degree(support: int) -> int:
  """Returns an even polynomial degree, given the kernel support"""
  return support + 3 + (support & 1)


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


class ESKernel(structref.StructRefProxy):
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
      Datum(single) if not isinstance(single, Datum) else single,
      Datum(apply_w) if not isinstance(apply_w, Datum) else apply_w,
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
    return ESKernel(
      Datum(epsilon),
      Datum(oversampling),
      Datum(beta),
      Datum(e0),
      Datum(support),
      Datum(analytic),
      Datum(single),
      Datum(apply_w),
    )


structref.define_boxing(ESKernelStructRef, ESKernel)


@overload(ESKernel, prefer_literal=True)
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
      degree = polynomial_degree(support)
      instance.coeffs = generate_poly_coeffs(support, beta, e0, degree)

    return instance

  return impl


@overload_method(ESKernelStructRef, "allocate_taps", inline="always")
def overload_allocate_taps(self):
  """Allocate a 1-D array of length ``support`` to hold kernel taps.

  dtype is float32 when ``single`` is a literal True, otherwise float64.
  """
  if isinstance(SINGLE := self.get_literal("single"), bool):
    dtype = np.float32 if SINGLE else np.float64
  else:
    dtype = np.float64

  if isinstance(SUPPORT := self.get_literal("support"), int):
    return lambda self: np.empty(SUPPORT, dtype)
  else:
    return lambda self: np.empty(self.support, dtype)


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
      COEFFS = generate_poly_coeffs(SUPPORT, BETA, E0, polynomial_degree(SUPPORT))
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
      COEFFS = generate_poly_coeffs(SUPPORT, BETA, E0, polynomial_degree(SUPPORT))
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


@overload_attribute(ESKernelStructRef, "nsafe")
def overload_es_kernel_nsafe(self):
  return lambda self: (self.support + 1) // 2


@structref.register
class TemplateESKernelStructRef(LiteralStructRef):
  """ESKernel StructRef"""

  @property
  def support(self) -> int:
    """Kernel support"""
    return self.get_literal("support")

  @property
  def degree(self) -> int:
    """Polynomial degree ``degree = support + 3 + (support & 1)``.
    Always odd, so that ``degree + 1``
    coefficients split evenly into the two Horner chains
    when evaluating kernels"""
    return polynomial_degree(self.support)

  @property
  def single(self) -> bool:
    """True if the kernel is represented by single precision floats,
    False if represented by double"""
    return self.get_literal("single")

  @property
  def dtype(self) -> npt.DTypeLike:
    return np.float32 if self.single else np.float64

  @property
  def vector_length(self) -> int:
    """SIMD lanes per register for the tap dtype"""
    if (simd_bits := widest_simd_register_bits()) == 0:
      return 1

    dt = np.dtype(np.float32 if self.single else np.float64)
    return simd_bits // (8 * dt.itemsize)

  @property
  def nvectors(self) -> int:
    """``ceil(support / vector_length)`` -- vectors spanning the full support"""
    vector_length = self.vector_length
    return (self.support + vector_length - 1) // vector_length

  @property
  def nevaluated_vectors(self) -> int:
    """``ceil(nvectors / 2)`` -- vectors actually evaluated"""
    return (self.nvectors + 1) // 2

  @property
  def ntaps(self) -> int:
    """``nvectors * vector_length`` -- number of kernel taps"""
    return self.nvectors * self.vector_length

  @property
  def row_stride(self) -> int:
    """``nevaluated_vectors * vector_length``` --
    row stride of the coefficient table, and the number of taps evaluated directly.
    """
    return self.nevaluated_vectors * self.vector_length

  @property
  def zero_padding_start(self) -> int:
    """``max(support, row_stride)`` --
    the first tap index at which zero padding starts"""
    return max(self.support, self.row_stride)

  @property
  def nmirror(self) -> int:
    """``max(support - row_stride)``` --"""
    return max(0, self.support - self.row_stride)

  @property
  def source_coeffs_shape(self) -> Tuple[int, int]:
    """Shape of the ``ESKernel`` table this kernel is built from."""
    return (self.degree + 1, self.support)

  @property
  def coeffs_shape(self) -> Tuple[int, int]:
    """Stored polynomial coefficient shape, ``(degree + 1, row_stride)``.

    Rows are padded out to a whole number of SIMD registers and truncated to
    the first ``row_stride`` sub-intervals; the rest are recovered from the
    kernel's symmetry. Note ``row_stride`` exceeds ``support`` whenever one
    register already spans the support -- the table must still be that wide,
    because the evaluators loop over ``range(row_stride)``.
    """
    return (self.degree + 1, self.row_stride)


class TemplateESKernel(structref.StructRefProxy):
  """SIMD-oriented re-layout of an :class:`ESKernel` polynomial coefficient table.

  A port of ducc0's ``TemplateKernel`` (``ducc0/math/gridding_kernel.h``).
  ``support`` and ``single`` are *template* parameters and must be
  compile-time :class:`Datum` literals: every loop bound, buffer length and
  the tap dtype derive from them and are baked into the generated code.
  ``beta`` and ``e0`` change only the values in the table, never its shape,
  so the same specialisation serves any kernel of that support.

  The table is stored as ``(degree + 1, row_stride)``, one column per
  sub-interval of ``[-1, 1]`` and rows in Horner order (row 0 is the
  highest power). It differs from ``ESKernel.coeffs`` in two ways:

  * Columns are truncated to ``row_stride``, a whole number of SIMD
    registers spanning half the support. The remaining ``nmirror``
    sub-intervals are recovered from the kernel's symmetry: sub-interval
    ``support - 1 - k`` is sub-interval ``k`` with the local coordinate
    negated, so the two share a pair of Horner chains and differ only in
    the sign of the odd part.
  * The dtype follows ``single``, rather than always being float64, so that
    the taps are produced in the precision the gridding loop consumes.

  The evaluators (``eval2s``, ``eval2``) write ``ntaps`` taps rather than
  ``support``: the tail past the support is padding held at zero, so that
  the gridding loop can process whole registers unconditionally. Tap buffers
  must be ``ntaps`` long -- ``ESKernel.allocate_taps()`` returns ``support``
  entries and is too short for them.

  Args:
    es_kernel: polynomial (``analytic=False``) kernel supplying the source
      ``(degree + 1, support)`` coefficient table.
    support: kernel support, as an integer literal.
    single: float32 taps if True, float64 otherwise, as a boolean literal.
  """

  def __new__(cls, es_kernel: ESKernel, support: int, single: bool):
    return structref.StructRefProxy.__new__(cls, es_kernel, support, single)

  @property
  @numba.njit
  def ntaps(self):
    """Expose ntaps within Python"""
    return self.ntaps

  @property
  @numba.njit
  def row_stride(self):
    """Expose row_stride within Python"""
    return self.row_stride


structref.define_boxing(TemplateESKernelStructRef, TemplateESKernel)


@overload(TemplateESKernel, prefer_literal=True)
def overload_template_es_kernel(es_kernel, support, single):
  """Implement the TemplateESKernel constructor"""
  if not is_datum_literal(support, int):
    raise errors.RequireLiteralValue(f"support {support} must be an IntegerLiteral")

  if not is_datum_literal(single, bool):
    raise errors.RequireLiteralValue(f"single {single} must be a BooleanLiteral")

  # The tap dtype follows ``single``, so the coefficient table must too. It
  # cannot be inherited from ``es_kernel``, whose table is always float64:
  # ``impl`` builds this field with the ``single``-derived dtype, and a float32
  # array will not store into a float64 field.
  fields = [
    ("es_kernel", es_kernel),
    ("support", support),
    ("single", single),
    ("coeffs", types.float32[:, :] if single.literal_value else types.float64[:, :]),
  ]

  state_type = TemplateESKernelStructRef(fields)

  NCOLUMNS = min(state_type.support, state_type.row_stride)
  DTYPE = state_type.dtype
  COEFFS_SHAPE = state_type.coeffs_shape
  SOURCE_SHAPE = state_type.source_coeffs_shape

  def impl(es_kernel, support, single):
    if es_kernel.coeffs.shape != SOURCE_SHAPE:
      raise ValueError(
        f"ESKernel.coeffs shape {es_kernel.coeffs.shape} != {SOURCE_SHAPE}"
      )

    instance = structref.new(state_type)
    instance.es_kernel
    instance.support = support
    instance.single = single
    instance.coeffs = np.zeros(COEFFS_SHAPE, DTYPE)
    instance.coeffs[:, :NCOLUMNS] = es_kernel.coeffs[:, :NCOLUMNS]
    return instance

  return impl


@overload_attribute(TemplateESKernelStructRef, "ntaps", inline="always")
def overload_template_es_kernel_ntaps(self):
  """Expose ntaps within numba"""
  NTAPS = self.ntaps
  return lambda self: NTAPS


@overload_attribute(TemplateESKernelStructRef, "row_stride", inline="always")
def overload_template_es_kernel_row_stride(self):
  """Expose row_stride within numba"""
  ROW_STRIDE = self.row_stride
  return lambda self: ROW_STRIDE


@overload_method(
  TemplateESKernelStructRef,
  "eval2s",
  prefer_literal=True,
  inline="always",
  fastmath=True,
)
def overload_eval2s(self, x, y, z, nth, ku, kv):
  """Evaluate the three-axis separable kernel for a single visibility.

  All ``support`` u taps and v taps are evaluated at once -- one Horner
  chain over the even powers of the local coordinate and one over the odd
  powers, which also lets each stored sub-interval yield its mirror image
  for free. The w axis contributes a *single* tap, because a visibility
  touches one w plane at a time; that tap is folded into the u taps as a
  scale factor, so the gridding loop never multiplies by it again.

  Args:
    x: u position of the visibility within the kernel footprint, normalised
      to ``[-1, 1]``, i.e. ``-2 * ufrac + (support - 1)``.
    y: as ``x``, for the v axis.
    z: w position of the visibility in w-plane units, ``(w0 - w) / dw``.
      Reduced internally, using ``nth``, to the same ``[-1, 1]`` frame.
    nth: index of the w plane being gridded, in ``[0, support)``.
    ku: output buffer of ``ntaps`` u taps, scaled by the w tap.
    kv: output buffer of ``ntaps`` v taps.

  Taps at indices ``[support, ntaps)`` are set to zero.
  """
  DTYPE = self.dtype
  ZERO = DTYPE(0.0)
  TWO = DTYPE(2.0)
  ROW_STRIDE = self.row_stride
  SUPPORT = self.support
  NMIRROR = self.nmirror
  DEGREE = self.degree

  # Iteration tuples for combination with literal_unroll
  # Bound is the polynomial DEGREE, not the support: the two Horner chains
  # between them consume all DEGREE + 1 coefficient rows. Using SUPPORT here
  # truncates the chains and silently drops the highest-order coefficients.
  COEFF_INDEX = tuple(range(2, DEGREE, 2))
  ZERO_PAD_INDEX = tuple(range(self.zero_padding_start, self.ntaps))
  ROW_STRIDE_INDEX = tuple(range(self.row_stride))
  HAS_PADDING = len(ZERO_PAD_INDEX) > 0

  def impl(self, x, y, z, nth, ku, kv):
    x = DTYPE(x)
    y = DTYPE(y)
    z = DTYPE(z - nth) * TWO + DTYPE(SUPPORT - 1)

    if nth >= ROW_STRIDE:
      z *= DTYPE(-1)
      nth = SUPPORT - 1 - nth

    x2 = DTYPE(x * x)
    y2 = DTYPE(y * y)
    z2 = DTYPE(z * z)

    # Evaluate the contribution of the z coordinate
    tap_value_z = self.coeffs[0, nth]
    tap_value_z2 = self.coeffs[1, nth]

    for j in numba.literal_unroll(COEFF_INDEX):
      tap_value_z = tap_value_z * z2 + self.coeffs[j, nth]
      tap_value_z2 = tap_value_z2 * z2 + self.coeffs[j + 1, nth]

    z_factor = tap_value_z * z + tap_value_z2

    if HAS_PADDING:
      for k in numba.literal_unroll(ZERO_PAD_INDEX):
        ku[k] = ZERO
        kv[k] = ZERO

    for k in numba.literal_unroll(ROW_STRIDE_INDEX):
      cj = self.coeffs[0, k]
      cj1 = self.coeffs[1, k]
      tap_value_x = cj
      tap_value_y = cj
      tap_value_x2 = cj1
      tap_value_y2 = cj1

      for j in numba.literal_unroll(COEFF_INDEX):
        cj = self.coeffs[j, k]
        cj1 = self.coeffs[j + 1, k]

        tap_value_x = tap_value_x * x2 + cj
        tap_value_y = tap_value_y * y2 + cj
        tap_value_x2 = tap_value_x2 * x2 + cj1
        tap_value_y2 = tap_value_y2 * y2 + cj1

      ku[k] = (tap_value_x * x + tap_value_x2) * z_factor
      kv[k] = tap_value_y * y + tap_value_y2

      if k < NMIRROR:
        k2 = SUPPORT - 1 - k
        ku[k2] = (tap_value_x2 - tap_value_x * x) * z_factor
        kv[k2] = tap_value_y2 - tap_value_y * y

  return impl


@overload_method(
  TemplateESKernelStructRef,
  "eval2",
  prefer_literal=True,
  inline="always",
  fastmath=True,
)
def overload_eval2(self, x, y, ku, kv):
  """Evaluate the two-axis separable kernel for a single visibility.

  ``eval2s`` without the w axis, used when w gridding is disabled: the same
  pair of Horner chains and the same symmetry mirror produce all ``support``
  u and v taps, but the u taps are left unscaled.

  Args:
    x: u position of the visibility within the kernel footprint, normalised
      to ``[-1, 1]``, i.e. ``-2 * ufrac + (support - 1)``.
    y: as ``x``, for the v axis.
    ku: output buffer of ``ntaps`` u taps.
    kv: output buffer of ``ntaps`` v taps.

  Taps at indices ``[support, ntaps)`` are set to zero.
  """
  DTYPE = self.dtype
  ZERO = DTYPE(0.0)
  SUPPORT = self.support
  NMIRROR = self.nmirror
  DEGREE = self.degree

  # Iteration tuples for combination with literal_unroll
  # Bound is the polynomial DEGREE, not the support: the two Horner chains
  # between them consume all DEGREE + 1 coefficient rows. Using SUPPORT here
  # truncates the chains and silently drops the highest-order coefficients.
  COEFF_INDEX = tuple(range(2, DEGREE, 2))
  ZERO_PAD_INDEX = tuple(range(self.zero_padding_start, self.ntaps))
  ROW_STRIDE_INDEX = tuple(range(self.row_stride))
  HAS_PADDING = len(ZERO_PAD_INDEX) > 0

  def impl(self, x, y, ku, kv):
    x = DTYPE(x)
    y = DTYPE(y)

    x2 = DTYPE(x * x)
    y2 = DTYPE(y * y)

    if HAS_PADDING:
      for k in numba.literal_unroll(ZERO_PAD_INDEX):
        ku[k] = ZERO
        kv[k] = ZERO

    for k in numba.literal_unroll(ROW_STRIDE_INDEX):
      cj = self.coeffs[0, k]
      cj1 = self.coeffs[1, k]
      tap_value_x = cj
      tap_value_y = cj
      tap_value_x2 = cj1
      tap_value_y2 = cj1

      for j in numba.literal_unroll(COEFF_INDEX):
        cj = self.coeffs[j, k]
        cj1 = self.coeffs[j + 1, k]

        tap_value_x = tap_value_x * x2 + cj
        tap_value_y = tap_value_y * y2 + cj
        tap_value_x2 = tap_value_x2 * x2 + cj1
        tap_value_y2 = tap_value_y2 * y2 + cj1

      ku[k] = tap_value_x * x + tap_value_x2
      kv[k] = tap_value_y * y + tap_value_y2

      if k < NMIRROR:
        k2 = SUPPORT - 1 - k
        ku[k2] = tap_value_x2 - tap_value_x * x
        kv[k2] = tap_value_y2 - tap_value_y * y

  return impl


@overload_method(
  TemplateESKernelStructRef,
  "eval",
  prefer_literal=True,
  inline="always",
  fastmath=True,
)
def overload_template_eval(self, x):
  """Evaluate the kernel at a single position.

  The scalar counterpart of ``eval2``/``eval2s``, and the equivalent of
  ``ESKernelStructRef.evaluate`` reading the truncated table: it locates the
  sub-interval containing ``x``, reflects it into the stored half of the
  table if need be, and runs a single Horner chain over all ``degree + 1``
  coefficient rows. Mostly useful for checking the table, since it evaluates
  one tap where the vector evaluators produce all of them for the same
  polynomial degree.

  Args:
    x: position within the kernel footprint, normalised to ``[-1, 1]``.
      Note this differs from ``ESKernelStructRef.evaluate``, which takes a
      position in grid pixels, i.e. in ``[-support / 2, support / 2]``.

  Returns:
    The kernel value, zero for ``abs(x) >= 1``.
  """
  DTYPE = self.dtype
  ZERO = DTYPE(0.0)
  SUPPORT = self.support
  ROW_STRIDE = self.row_stride
  # See the COEFF_INDEX note in eval2s: the chain runs over all DEGREE + 1
  # coefficient rows, which is not the same as the support.
  COEFF_INDEX = tuple(range(1, self.degree + 1))

  def impl(self, x):
    if abs(x) >= 1.0:
      return ZERO

    xrel = SUPPORT * 0.5 * (DTYPE(x) + DTYPE(1.0))
    nth = min(int(xrel), SUPPORT - 1)
    locx = (DTYPE(xrel - nth) - DTYPE(0.5)) * DTYPE(2.0)

    if nth >= ROW_STRIDE:
      locx = -locx
      nth = SUPPORT - 1 - nth

    value = self.coeffs[0, nth]

    for j in numba.literal_unroll(COEFF_INDEX):
      value = value * locx + self.coeffs[j, nth]

    return value

  return impl
