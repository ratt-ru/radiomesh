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
) -> Tuple[Signature, Callable]:
  """Return a tuple of kernel :code:`offsets +  float(index)` positions.

  Args:
    kernel_literal: ES kernel object
    grid_size: grid extent
    pixel_start: u/v pixel start.

  Returns:
    Tuple of kernel index positions
  """
  if not isinstance(kernel_literal, DatumLiteral):
    raise RequireLiteralValue(f"'kernel' {kernel_literal} must be a DatumLiteral")

  if not isinstance(grid_size, types.IntegerLiteral):
    raise RequireLiteralValue(f"'grid_size' {grid_size} must be an IntegerLiteral")

  if not isinstance(pixel_start, types.Integer):
    raise TypingError(f"'index' ({pixel_start}) must be an integer")

  kernel = kernel_literal.datum_value
  SUPPORT = kernel.support
  N = grid_size.literal_value
  return_type = types.Tuple([types.int64] * SUPPORT)
  sig = return_type(kernel_literal, grid_size, pixel_start)

  def codegen(context, builder, signature, args):
    _, _, pixel_start = args
    _, _, pixel_start_type = signature.args
    llvm_ret_type = context.get_value_type(signature.return_type)
    offset_type = signature.return_type.dtype
    llvm_offset_type = context.get_value_type(offset_type)
    pos_tuple = cgutils.get_null_value(llvm_ret_type)

    # Evaluate the fftshifted grid index for
    # each position in the kernel support
    for so in range(SUPPORT):
      ir_offset = ir.Constant(llvm_offset_type, so)
      fftshift_grid_index = context.compile_internal(
        builder,
        lambda gi, o: (gi + o) % N,
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
  kernel_pos: types.UniTuple,
  grid: types.Float,
  pixel_start: types.Integer,
) -> Tuple[Signature, Callable]:
  """Evaluates the es kernel at :code:`(pos - grid + 0.5) / half_support`

  Args:
    kernel: ES Kernel object
    pos: Kernel evaluation coordinate
    grid: UV grid coordinate
    pixel_start: u/v pixel coordinate at the start of the kernel

  Returns:
    Tuple of kernel values
  """
  if not isinstance(kernel_literal, DatumLiteral):
    raise RequireLiteralValue(f"'kernel' {kernel_literal} must be a DatumLiteral")

  if not isinstance(kernel_pos, types.UniTuple) or not isinstance(
    kernel_pos.dtype, types.Integer
  ):
    raise TypingError(f"'x' ({kernel_pos}) must be a tuple of integers")

  if not isinstance(grid, types.Float):
    raise TypingError(f"'grid' ({grid}) must be a float")

  if not isinstance(pixel_start, types.Integer):
    raise TypingError(f"'pixel_start' ({pixel_start}) must be an integer")

  kernel = kernel_literal.datum_value

  HALF_SUPPORT = kernel.half_support
  BETAK = kernel.support * kernel.beta
  MU = kernel.mu

  if MU == 0.5:
    # Use the faster sqrt function
    def kernel_fn(kernel_pos: int, grid: float, pixel_start: int) -> float:
      x = (kernel_pos + pixel_start - grid) / HALF_SUPPORT
      value = np.exp(BETAK * (np.sqrt(1.0 - x * x) - 1.0))
      # Above is only defined for [-1.0, 1.0]
      # Zero after possible vectorisation (SIMD) of the above expression
      return value if -1.0 <= x <= 1.0 else 0.0
  else:

    def kernel_fn(kernel_pos: int, grid: float, pixel_start: int) -> float:
      x = (kernel_pos + pixel_start - grid) / HALF_SUPPORT
      value = np.exp(BETAK * (np.power(1.0 - x * x, MU) - 1.0))
      # Above is only defined for [-1.0, 1.0]
      # Zero after possible vectorisation (SIMD) of the above expression
      return value if -1.0 <= x <= 1.0 else 0.0

  return_type = types.Tuple([types.float64] * kernel.support)
  sig = return_type(kernel_literal, kernel_pos, grid, pixel_start)

  def codegen(context, builder, signature, args):
    _, pos, grid, pixel = args
    _, pos_type, grid_type, pixel_type = signature.args
    llvm_ret_type = context.get_value_type(signature.return_type)
    kernel_tuple = cgutils.get_null_value(llvm_ret_type)
    kernel_sig = signature.return_type.dtype(pos_type.dtype, grid_type, pixel_type)

    for i in range(len(pos_type)):
      x = builder.extract_value(pos, i)
      ktuple = context.compile_internal(
        builder, kernel_fn, kernel_sig, [x, grid, pixel]
      )
      kernel_tuple = builder.insert_value(kernel_tuple, ktuple, i)

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
  apply_w: bool = False
  # Kernel support
  support: int = dataclasses.field(init=False)

  def __post_init__(self):
    """Determine the support given the other kernel parameters"""
    factor = 3.0 if self.apply_w else 2.0
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
