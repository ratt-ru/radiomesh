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

from radiomesh.literals import DatumLiteral


@intrinsic(prefer_literal=True)
def es_kernel_positions(
  typingctx, kernel: DatumLiteral, index: types.Integer
) -> Tuple[Signature, Callable]:
  """Return a tuple of kernel :code:`offsets +  float(index)` positions.

  Args:
    kernel: ES kernel object
    index: uv grid index

  Returns:
    Tuple of kernel index positions
  """
  if not isinstance(kernel, DatumLiteral):
    raise RequireLiteralValue(f"'kernel' {kernel} must be a DatumLiteral")

  if not isinstance(index, types.Integer):
    raise TypingError(f"'index' ({index}) must be an integer")

  offsets = kernel.datum_value.offsets
  offset_type = numba.typeof(offsets)
  sig = offset_type(kernel, index)

  def codegen(context, builder, signature, args):
    _, index = args
    _, index_type = signature.args
    float_type = signature.return_type.dtype
    llvm_ret_type = context.get_value_type(signature.return_type)
    llvm_float_type = context.get_value_type(float_type)
    pos_tuple = cgutils.get_null_value(llvm_ret_type)
    sig = float_type(float_type, index_type)

    for o, offset in enumerate(offsets):
      ir_offset = ir.Constant(llvm_float_type, offset)
      value = context.compile_internal(
        builder, lambda o, i: o + float(i), sig, [ir_offset, index]
      )

      pos_tuple = builder.insert_value(pos_tuple, value, o)

    return pos_tuple

  return sig, codegen


@intrinsic(prefer_literal=True)
def eval_es_kernel(
  typingctx, kernel: DatumLiteral, pos: types.UniTuple, grid: types.Float
) -> Tuple[Signature, Callable]:
  """Evaluates the es kernel at :code:`(pos - grid + 0.5) / half_support`

  Args:
    kernel: ES Kernel object
    pos: Position at which to evaluate the kernel
    grid: uv grid position

  Returns:
    Tuple of kernel values
  """
  if not isinstance(kernel, DatumLiteral):
    raise RequireLiteralValue(f"'kernel' {kernel} must be a DatumLiteral")

  if not isinstance(pos, types.UniTuple) or not isinstance(pos.dtype, types.Float):
    raise TypingError(f"'x' ({pos}) must be a tuple of floats")

  if not isinstance(grid, types.Float):
    raise TypingError("'grid' must be a float")

  support = len(pos)
  return_type = types.Tuple([pos.dtype] * support)
  sig = return_type(kernel, pos, grid)

  HALF_SUPPORT = kernel.datum_value.half_support
  BETA = kernel.datum_value.beta
  E0 = kernel.datum_value.e0
  if E0 != 0.5:
    raise NotImplementedError(f"Polynomial kernels are required for e0 ({E0}) != 0.5")

  def kernel_fn(index: float, pixel: float) -> float:
    x = (index - pixel + 0.5) / HALF_SUPPORT
    value = np.exp(BETA * HALF_SUPPORT * (np.sqrt(1.0 - x * x) - 1.0))
    # Above is only defined for [-1.0, 1.0]
    # Zero after possible vectorisation (SIMD) of the above expression
    return value if -1.0 <= x <= 1.0 else 0.0

  def codegen(context, builder, signature, args):
    _, pos, grid = args
    _, pos_index_type, grid_type = signature.args
    llvm_ret_type = context.get_value_type(signature.return_type)
    kernel_tuple = cgutils.get_null_value(llvm_ret_type)
    kernel_sig = signature.return_type.dtype(pos_index_type.dtype, grid_type)

    for i in range(len(pos_index_type)):
      x = builder.extract_value(pos, i)
      ktuple = context.compile_internal(builder, kernel_fn, kernel_sig, [x, grid])
      kernel_tuple = builder.insert_value(kernel_tuple, ktuple, i)

    return kernel_tuple

  return sig, codegen


@dataclasses.dataclass(slots=True, eq=True, unsafe_hash=True)
class ESKernel:
  """Defines an ES Kernel of the form
  :code:`math.exp(beta * (math.pow(1.0 - x * x, e0) - 1.0))`"""

  # Desired wgridder accuracy
  epsilon: float = 2e-13
  # Oversampling factor.
  # Corresponds to :code:`ofactor` within the ducc0 wgridder code base
  oversampling: int = 2
  # ES kernel parameters
  beta: float = 2.3
  e0: float = 0.5
  # Is w gridding enabled
  wgridding: bool = False
  # Kernel support
  support: int = dataclasses.field(init=False)

  def __post_init__(self):
    """Determine the support given the other kernel parameters"""
    factor = 3.0 if self.wgridding else 2.0
    self.support = int(math.ceil(math.log10(factor * 1.0 / self.epsilon))) + 1

  @property
  def half_support(self) -> float:
    """Return precise half-support"""
    return self.support / 2.0

  @property
  def offsets(self) -> Tuple[float, ...]:
    """Returns a tuple the size of the kernel support containing
    offsets from the centre of the kernel"""
    return tuple(float(p) - self.half_support for p in range(self.support))
