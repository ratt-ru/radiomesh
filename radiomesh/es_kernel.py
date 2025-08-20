from __future__ import annotations

import dataclasses
import math
from typing import Callable, Tuple

import numpy as np
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.core.typing.templates import Signature
from numba.extending import intrinsic

from radiomesh.literals import DatumLiteral


@intrinsic(prefer_literal=True)
def es_kernel_positions(
  typingctx,
  kernel_literal: DatumLiteral,
  grid_size: types.IntegerLiteral,
  pixel_start: types.Integer,
  modulo_grid_size: types.BooleanLiteral,
) -> Tuple[Signature, Callable]:
  """Return a tuple of kernel
  :code:`(pixel_start + range(kernel.support)) % grid_size`
  positions.

  Args:
    kernel_literal: ES kernel object
    grid_size: grid extent
    pixel_start: u/v pixel start.
    modulo_grid_size: flag indicating whether modulo grid size should be applied

  Returns:
    Tuple of kernel index positions
  """
  if not isinstance(kernel_literal, DatumLiteral):
    raise RequireLiteralValue(f"'kernel' {kernel_literal} must be a DatumLiteral")

  if not isinstance(grid_size, types.IntegerLiteral):
    raise RequireLiteralValue(f"'grid_size' {grid_size} must be an IntegerLiteral")

  if not isinstance(pixel_start, types.Integer):
    raise TypingError(f"'pixel_start' ({pixel_start}) must be an integer")

  if not isinstance(modulo_grid_size, types.BooleanLiteral):
    raise RequireLiteralValue(
      f"'modulo_grid_size' {modulo_grid_size} must be a BooleanLiteral"
    )

  kernel = kernel_literal.datum_value
  SUPPORT = kernel.support
  N = grid_size.literal_value
  MODULO = modulo_grid_size.literal_value
  return_type = types.Tuple([types.int64] * SUPPORT)
  sig = return_type(kernel_literal, grid_size, pixel_start, modulo_grid_size)

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
        (lambda ps, o: (ps + o) % N) if MODULO else (lambda ps, o: ps + o),
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

  if MU == 0.5:
    # Use the faster sqrt function
    def kernel_fn(kernel_offset: int, grid: float, pixel_start: int) -> float:
      x = (kernel_offset + pixel_start - grid) / HALF_SUPPORT
      value = np.exp(BETAK * (np.sqrt(1.0 - x * x) - 1.0))
      # Above is only defined for [-1.0, 1.0]
      # Zero after possible vectorisation (SIMD) of the above expression
      return value if -1.0 <= x <= 1.0 else 0.0
  else:

    def kernel_fn(kernel_offset: int, grid: float, pixel_start: int) -> float:
      x = (kernel_offset + pixel_start - grid) / HALF_SUPPORT
      value = np.exp(BETAK * (np.power(1.0 - x * x, MU) - 1.0))
      # Above is only defined for [-1.0, 1.0]
      # Zero after possible vectorisation (SIMD) of the above expression
      return value if -1.0 <= x <= 1.0 else 0.0

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
  :code:`math.exp(beta * (math.pow(1.0 - x * x, mu) - 1.0))`"""

  # Desired wgridder accuracy
  epsilon: float = 2e-13
  # Oversampling factor.
  # Corresponds to :code:`ofactor` within the ducc0 wgridder code base
  oversampling: int = 2
  # ES kernel parameters
  beta: float = 2.3
  mu: float = 0.5
  # Is w gridding enabled
  apply_w: dataclasses.InitVar[bool] = False
  # Kernel support
  support: int = dataclasses.field(init=False)

  def __post_init__(self, apply_w):
    """Determine the support given the other kernel parameters"""
    factor = 3.0 if apply_w else 2.0
    self.support = int(math.ceil(math.log10(factor * 1.0 / self.epsilon))) + 1

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
