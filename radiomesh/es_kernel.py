from __future__ import annotations

import dataclasses
import math
from typing import Callable, Tuple

import numba
import numpy as np
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.core.typing.templates import Signature
from numba.extending import intrinsic
from numpy.polynomial.chebyshev import cheb2poly, chebinterpolate

from radiomesh.errors import KernelSelectionError
from radiomesh.generated._es_kernel_params import KERNEL_DB, KernelParams
from radiomesh.literals import DatumLiteral


def select_kernel_params(
  epsilon: float,
  oversampling: float,
  ndim: int,
  single: bool,
) -> KernelParams:
  """Return the KernelDB entry with the smallest support satisfying all constraints.

  Mirrors ``selectKernel<T>(ofactor, ndim, epsilon)`` from ducc0's
  ``gridding_kernel.h``.  Finds entries where ``ofactor <= oversampling``,
  ``epsilon <= target_epsilon``, ``ndim == ndim``, ``single == single``, then
  returns the one with minimum width (support).

  Args:
    epsilon: required accuracy.
    oversampling: target oversampling factor.
    ndim: dimensionality — 2 for 2D gridding, 3 for w-gridding.
    single: True for single (float32) precision, False for double.

  Raises:
    ValueError: if no matching entry exists in KERNEL_DB.
  """
  best: KernelParams | None = None
  for k in KERNEL_DB:
    if (
      k.ndim == ndim
      and k.single == single
      and k.oversampling <= oversampling
      and k.epsilon <= epsilon
      and (best is None or k.support < best.support)
    ):
      best = k
  if best is None:
    prec = "single" if single else "double"
    ndim_label = {2: "2-D gridding", 3: "w-gridding (3-D)"}.get(ndim, f"ndim={ndim}")
    raise KernelSelectionError(
      f"No KernelDB entry satisfies epsilon={epsilon:.2e}, "
      f"oversampling={oversampling}, {ndim_label}, {prec} precision.\n"
      f"Supported ranges for this (ndim={ndim}, single={single}) combination: "
      f"oversampling [1.20, 2.50], "
      f"epsilon >= minimum achievable at the requested oversampling.\n"
      f"Options: increase epsilon, increase oversampling toward 2.5"
      + (", or use single=False" if single else "")
      + ", or set analytic=True to bypass KernelDB selection."
    )
  return best


def generate_poly_coeffs(
  W: int, betak: float, mu: float
) -> tuple[tuple[float, ...], ...]:
  """Generate polynomial approximation coefficients for the ES kernel.

  Replicates ducc0's ``getCoeffs`` function (gridding_kernel.cc).

  The ES kernel ``exp(betak * ((1 - v²)^mu - 1))`` is approximated on
  ``[-1, 1]`` by ``W`` degree-``D`` polynomials, one per sub-interval.

  Args:
    W: kernel support (number of sub-intervals).
    betak: scaled beta parameter — ``beta * W`` (i.e. ``ESKernel.beta * support``).
    mu: exponent parameter; equivalent to ``e0`` in ducc0's KernelParams.

  Returns:
    Nested tuple of shape ``(D+1) x W`` where ``D = W + 3``.
    ``coeffs[j][i]`` is the coefficient of ``x^(D-j)`` for sub-interval ``i``
    (Horner order: index 0 is the leading / highest-power coefficient).
  """
  D = W + 3

  def es_kernel(v: float) -> float:
    tmp = (1.0 - v) * (1.0 + v)  # = 1 - v^2
    if tmp < 0.0:
      return 0.0
    return math.exp(betak * (math.pow(tmp, mu) - 1.0))

  # Chebyshev nodes on [-1, 1]
  chebroot = [math.cos((2 * i + 1) * math.pi / (2 * D + 2)) for i in range(D + 1)]

  # coeff[j][i] in output order (j = Horner step, i = sub-interval)
  coeff = [[0.0] * W for _ in range(D + 1)]

  for i in range(W):
    left = -1.0 + 2.0 * i / W
    right = -1.0 + 2.0 * (i + 1) / W

    # Function values at Chebyshev nodes mapped to [l, r]
    y = [
      es_kernel(chebroot[j] * (right - left) * 0.5 + (right + left) * 0.5)
      for j in range(D + 1)
    ]
    avg = sum(y) / (D + 1)
    y = [v - avg for v in y]

    # Chebyshev coefficients via DCT-I-like sum
    lcf = [
      sum(
        2.0 / (D + 1) * y[k] * math.cos(j * (2 * k + 1) * math.pi / (2 * D + 2))
        for k in range(D + 1)
      )
      for j in range(D + 1)
    ]
    lcf[0] *= 0.5

    # Chebyshev-to-monomial conversion: C[j][k] = coeff of x^k in T_j(x)
    C = [[0.0] * (D + 1) for _ in range(D + 1)]
    C[0][0] = 1.0
    if D >= 1:
      C[1][1] = 1.0
    for j in range(2, D + 1):
      C[j][0] = -C[j - 2][0]
      for k in range(1, j + 1):
        C[j][k] = 2.0 * C[j - 1][k - 1] - C[j - 2][k]

    # lcf2[k] = coefficient of x^k in the combined polynomial
    lcf2 = [sum(C[j][k] * lcf[j] for j in range(D + 1)) for k in range(D + 1)]
    lcf2[0] += avg

    # Store in Horner order: coeff[j][i] = lcf2[D-j]  (x^(D-j) term)
    for j in range(D + 1):
      coeff[j][i] = lcf2[D - j]

  return tuple(tuple(row) for row in coeff)


def generate_poly_coeffs_numpy(
  W: int, betak: float, mu: float
) -> tuple[tuple[float, ...], ...]:
  """Numpy alternative to :func:`generate_poly_coeffs`.

  Delegates Chebyshev interpolation to
  :func:`numpy.polynomial.chebyshev.chebinterpolate` and the
  Chebyshev-to-monomial conversion to
  :func:`numpy.polynomial.chebyshev.cheb2poly`, replacing the manual
  DCT sum and recurrence in the pure-Python version.

  Args:
    W: kernel support (number of sub-intervals).
    betak: scaled beta parameter — ``beta * W`` (i.e. ``ESKernel.beta * support``).
    mu: exponent parameter; equivalent to ``e0`` in ducc0's KernelParams.

  Returns:
    Same nested tuple of shape ``(D+1) x W`` as :func:`generate_poly_coeffs`.
  """
  D = W + 3

  # Vectorised ES kernel: chebinterpolate passes an array, not a scalar
  def es_kernel(v: np.ndarray) -> np.ndarray:
    tmp = (1.0 - v) * (1.0 + v)  # 1 - v^2; may be negative outside [-1,1]
    valid = tmp >= 0.0
    safe_tmp = np.where(valid, tmp, 0.0)  # avoid pow(negative, mu)
    return np.where(valid, np.exp(betak * (np.power(safe_tmp, mu) - 1.0)), 0.0)

  coeff = np.empty((D + 1, W))

  for i in range(W):
    left = -1.0 + 2.0 * i / W
    right = -1.0 + 2.0 * (i + 1) / W
    mid = (left + right) * 0.5
    half = (right - left) * 0.5

    # Exact Chebyshev interpolation at D+1 Type-I nodes on [-1, 1].
    # The function is expressed in the local sub-interval coordinate
    # (locx ∈ [-1, 1]) so no domain remapping is needed after this step.
    cheb_c = chebinterpolate(lambda locx, m=mid, h=half: es_kernel(locx * h + m), D)

    # Convert ascending Chebyshev coefficients to ascending power-basis
    # coefficients then store in Horner (descending) order:
    # coeff[j][i] = coeff of x^(D-j)
    poly_c = cheb2poly(cheb_c)  # ascending: poly_c[k] = coeff of x^k
    coeff[:, i] = poly_c[::-1]  # reverse to Horner order

  return tuple(tuple(float(v) for v in row) for row in coeff)


@intrinsic(prefer_literal=True)
def es_kernel_positions(
  typingctx,
  kernel_literal: DatumLiteral,
  grid_size: types.IntegerLiteral,
  pixel_start: types.Integer,
  fftshift_grid: types.BooleanLiteral,
) -> Tuple[Signature, Callable]:
  """Return a tuple of kernel
  :code:`(pixel_start + range(kernel.support)) % grid_size`
  positions.

  Args:
    kernel_literal: ES kernel object.
    grid_size: grid extent.
    pixel_start: u/v pixel start.
    fftshift_grid: flag indicating whether the position if fftshifted onto the grid.

  Returns:
    Tuple of kernel index positions
  """
  if not isinstance(kernel_literal, DatumLiteral):
    raise RequireLiteralValue(f"'kernel' {kernel_literal} must be a DatumLiteral")

  if not isinstance(grid_size, types.IntegerLiteral):
    raise RequireLiteralValue(f"'grid_size' {grid_size} must be an IntegerLiteral")

  if not isinstance(pixel_start, types.Integer):
    raise TypingError(f"'pixel_start' ({pixel_start}) must be an integer")

  if not isinstance(fftshift_grid, types.BooleanLiteral):
    raise RequireLiteralValue(
      f"'fftshift_grid' {fftshift_grid} must be a BooleanLiteral"
    )

  kernel = kernel_literal.datum_value
  SUPPORT = kernel.support
  N = grid_size.literal_value
  FFTSHIFT = fftshift_grid.literal_value
  return_type = types.Tuple([types.int64] * SUPPORT)
  sig = return_type(kernel_literal, grid_size, pixel_start, fftshift_grid)

  def codegen(context, builder, signature, args):
    _, _, pixel_start, _ = args
    _, _, pixel_start_type, _ = signature.args
    llvm_ret_type = context.get_value_type(signature.return_type)
    offset_type = signature.return_type.dtype
    llvm_offset_type = context.get_value_type(offset_type)
    pos_tuple = cgutils.get_null_value(llvm_ret_type)

    # Evaluate the possibly, fftshifted grid index for
    # each position in the kernel support
    for so in range(SUPPORT):
      ir_offset = ir.Constant(llvm_offset_type, so)
      fftshift_grid_index = context.compile_internal(
        builder,
        (lambda ps, o: (ps + o) % N) if FFTSHIFT else (lambda ps, o: ps + o),
        offset_type(pixel_start_type, offset_type),
        [pixel_start, ir_offset],
      )

      pos_tuple = builder.insert_value(pos_tuple, fftshift_grid_index, so)

    return pos_tuple

  return sig, codegen


@intrinsic(prefer_literal=True)
def eval_es_kernel(
  typingctx,
  kernel_literal: DatumLiteral[ESKernel],
  grid: types.Float,
  pixel_start: types.Integer,
) -> Tuple[Signature, Callable]:
  """Evaluates the es kernel at
  :code:`(range(kernel.support) + pixel_start - grid) / half_support`

  Args:
    kernel: ES Kernel object
    grid: UV grid coordinate
    pixel_start: u/v pixel coordinate at the start of the kernel

  Returns:
    Tuple of kernel values
  """
  if not isinstance(kernel_literal, DatumLiteral):
    raise RequireLiteralValue(f"'kernel' {kernel_literal} must be a DatumLiteral")

  if not isinstance(grid, types.Float):
    raise TypingError(f"'grid' ({grid}) must be a float")

  if not isinstance(pixel_start, types.Integer):
    raise TypingError(f"'pixel_start' ({pixel_start}) must be an integer")

  kernel = kernel_literal.datum_value

  SUPPORT = kernel.support
  HALF_SUPPORT = kernel.half_support
  BETAK = kernel.support * kernel.beta
  MU = kernel.mu
  ANALYTIC = kernel.analytic

  if ANALYTIC:
    if MU == 0.5:
      # mu == 0.5: fast sqrt variant
      def kernel_fn(kernel_offset: int, grid: float, pixel_start: int) -> float:
        x = (kernel_offset + pixel_start - grid) / HALF_SUPPORT
        value = np.exp(BETAK * (np.sqrt(1.0 - x * x) - 1.0))
        # Above is only defined for [-1.0, 1.0]
        # Zero after possible vectorisation (SIMD) of the above expression
        return value if -1.0 <= x <= 1.0 else 0.0
    else:
      # Full analytic version: exp(betak * ((1 - x^2)^mu - 1))
      def kernel_fn(kernel_offset: int, grid: float, pixel_start: int) -> float:
        x = (kernel_offset + pixel_start - grid) / HALF_SUPPORT
        value = np.exp(BETAK * (np.power(1.0 - x * x, MU) - 1.0))
        # Above is only defined for [-1.0, 1.0]
        # Zero after possible vectorisation (SIMD) of the above expression
        return value if -1.0 <= x <= 1.0 else 0.0
  else:
    # Polynomial approximation: W piecewise degree-(W+3) polynomials on [-1, 1].
    # Coefficients are computed at JIT-compile time via Chebyshev fitting
    # (replicates ducc0's getCoeffs / PolynomialKernel::eval).
    POLY_COEFFS = generate_poly_coeffs(SUPPORT, BETAK, MU)
    NCOEFFS = len(POLY_COEFFS)

    def kernel_fn(kernel_offset: int, grid: float, pixel_start: int) -> float:
      x = (kernel_offset + pixel_start - grid) / HALF_SUPPORT
      if x <= -1.0 or x >= 1.0:
        return 0.0
      xrel = SUPPORT * 0.5 * (x + 1.0)
      nth = int(xrel)
      if nth >= SUPPORT:
        nth = SUPPORT - 1
      locx = ((xrel - nth) - 0.5) * 2.0
      # Horner evaluation over coefficients for sub-interval nth
      res = POLY_COEFFS[0][nth]
      for i in numba.literal_unroll(range(1, NCOEFFS)):
        res = res * locx + POLY_COEFFS[i][nth]
      return res

  return_type = types.Tuple([types.float64] * kernel.support)
  sig = return_type(kernel_literal, grid, pixel_start)

  def codegen(context, builder, signature, args):
    _, grid, pixel_start = args
    _, grid_type, pixel_start_type = signature.args
    llvm_ret_type = context.get_value_type(signature.return_type)
    llvm_pixel_type = context.get_value_type(pixel_start_type)
    kernel_tuple = cgutils.get_null_value(llvm_ret_type)
    kernel_sig = signature.return_type.dtype(
      pixel_start_type, grid_type, pixel_start_type
    )

    for so in range(SUPPORT):
      ir_offset = ir.Constant(llvm_pixel_type, so)
      kernel_value = context.compile_internal(
        builder, kernel_fn, kernel_sig, [ir_offset, grid, pixel_start]
      )
      kernel_tuple = builder.insert_value(kernel_tuple, kernel_value, so)

    return kernel_tuple

  return sig, codegen


@dataclasses.dataclass(slots=True, eq=True, unsafe_hash=True)
class ESKernel:
  """Defines an ES Kernel of the form
  :code:`math.exp(beta * (math.pow(1.0 - x * x, mu) - 1.0))`

  Note: ``mu`` is equivalent to ``e0`` in ducc0's ``KernelParams``.
  """

  # Desired wgridder accuracy
  epsilon: float = 2e-13
  # Oversampling factor.
  # Corresponds to :code:`ofactor` within the ducc0 wgridder code base
  oversampling: float = 2.0
  # ES kernel parameters
  beta: float = 2.3
  # Exponent in (1 - x^2)^mu. Equivalent to e0 in ducc0's KernelParams.
  mu: float = 0.5
  # If True (default), evaluate analytically. mu == 0.5 uses the fast sqrt
  # variant; otherwise uses exp(pow(...)). If False, use polynomial approximation.
  analytic: bool = True
  # Single (float32) precision. Selects the appropriate KernelDB partition
  # when analytic=False.
  single: bool = False
  # Is w gridding enabled
  apply_w: dataclasses.InitVar[bool] = False
  # Kernel support. If None, computed from epsilon and apply_w via heuristic.
  support: int | None = None

  def __post_init__(self, apply_w):
    """Compute heuristic support from epsilon and apply_w if not provided."""
    if self.support is None:
      factor = 3.0 if apply_w else 2.0
      self.support = int(math.ceil(math.log10(factor * 1.0 / self.epsilon))) + 1

  @staticmethod
  def from_kernel_db(
    epsilon: float,
    oversampling: float = 2.0,
    apply_w: bool = False,
    single: bool = False,
    analytic: bool = False,
  ) -> ESKernel:
    """Construct an :class:`ESKernel` by selecting parameters from the KernelDB.

    Mirrors ``selectKernel`` from ducc0: finds the KernelDB entry with the
    smallest support satisfying all constraints, then returns a kernel
    configured with those parameters.

    Args:
      epsilon: required accuracy.
      oversampling: oversampling factor.
      apply_w: True for w-gridding (3-D), False for 2-D gridding.
      single: True for single (float32) precision, False for double.
      analytic: If False (default), use polynomial evaluation. If True,
        use analytic evaluation with the KernelDB beta/mu parameters.

    Raises:
      KernelSelectionError: if no matching entry exists in KERNEL_DB.
    """
    ndim = 3 if apply_w else 2
    entry = select_kernel_params(epsilon, oversampling, ndim, single)
    return ESKernel(
      epsilon=epsilon,
      oversampling=entry.oversampling,
      beta=entry.beta,
      mu=entry.mu,
      analytic=analytic,
      single=single,
      support=entry.support,
    )

  @property
  def half_support(self) -> float:
    """Return precise half-support"""
    return self.support / 2.0

  @property
  def half_support_int(self) -> int:
    """Returns half-support in integer coordinates"""
    return self.support // 2

  @property
  def offsets(self) -> Tuple[float, ...]:
    """Returns a tuple the size of the kernel support containing
    offsets from the centre of the kernel"""
    return tuple(float(p) - self.half_support for p in range(self.support))
