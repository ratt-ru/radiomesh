import dataclasses
import math
from typing import Callable, Tuple

import numpy as np
from llvmlite import ir
from numba.core import cgutils, types
from numba.extending import intrinsic


def es_kernel_positions_factory(offsets: Tuple[float, ...]) -> Callable:
  """Returns an intrinsic that generates a tuple :code:`offsets + float(index)`"""
  if not isinstance(offsets, tuple) or not all(isinstance(o, float) for o in offsets):
    raise TypeError(f"{offsets} is not a tuple of floats")

  @intrinsic
  def es_kernel_positions(typingctx, index):
    """Return a tuple of kernel :code:`offsets +  float(index)` positions.

    Args:
      index: uv grid index

    Returns:
      Tuple of kernel index positions
    """

    if not isinstance(index, types.Integer):
      raise TypeError(f"'index' ({index}) must be an integer")

    return_type = types.Tuple([types.float64] * len(offsets))
    sig = return_type(index)

    def codegen(context, builder, signature, args):
      (index,) = args
      (index_type,) = signature.args
      llvm_ret_type = context.get_value_type(signature.return_type)
      llvm_float64_type = context.get_value_type(signature.return_type.dtype)
      pos_tuple = cgutils.get_null_value(llvm_ret_type)
      sig = types.float64(types.float64, index_type)

      for o, offset in enumerate(offsets):
        ir_offset = ir.Constant(llvm_float64_type, offset)
        value = context.compile_internal(
          builder, lambda o, i: o + float(i), sig, [ir_offset, index]
        )

        pos_tuple = builder.insert_value(pos_tuple, value, o)

      return pos_tuple

    return sig, codegen

  return es_kernel_positions


def es_kernel_factory(kernel_fn: Callable[[float, float], float]) -> Callable:
  @intrinsic
  def es_kernel(typingctx, pos, grid):
    """Evaluates the es kernel at :code:`(pos - grid + 0.5) / half_support`

    Args:
      pos: Position at which to evaluate the kernel
      grid: uv grid position

    Returns:
      Tuple of kernel values
    """
    if not isinstance(pos, types.UniTuple) or not isinstance(pos.dtype, types.Float):
      raise TypeError(f"'x' ({pos}) must be a tuple of floats")

    if not isinstance(grid, types.Float):
      raise TypeError("'grid' must be a float")

    support = len(pos)
    return_type = types.Tuple([pos.dtype] * support)
    sig = return_type(pos, grid)

    def codegen(context, builder, signature, args):
      pos, grid = args
      pos_index_type, grid_type = signature.args
      llvm_ret_type = context.get_value_type(signature.return_type)
      kernel_tuple = cgutils.get_null_value(llvm_ret_type)
      kernel_sig = signature.return_type.dtype(pos_index_type.dtype, grid_type)

      for i in range(len(pos_index_type)):
        x = builder.extract_value(pos, i)
        ktuple = context.compile_internal(builder, kernel_fn, kernel_sig, [x, grid])
        kernel_tuple = builder.insert_value(kernel_tuple, ktuple, i)

      return kernel_tuple

    return sig, codegen

  return es_kernel


@dataclasses.dataclass(slots=True)
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

  @property
  def kernel_fn(self) -> Callable[[float, float], float]:
    """Returns a function evaluating the ES kernel"""
    HALF_SUPPORT = self.half_support
    BETA = self.beta
    E0 = self.e0
    if E0 != 0.5:
      raise NotImplementedError(f"Polynomial kernels are required for e0 ({E0}) != 0.5")

    def kernel(index: float, pixel: float) -> float:
      x = (index - pixel + 0.5) / HALF_SUPPORT
      # kernel is only defined for [-1.0, 1.0]
      if x < -1.0 or x > 1.0:
        return 0.0

      return np.exp(BETA * HALF_SUPPORT * (np.sqrt(1.0 - x * x) - 1.0))

    return kernel

  @property
  def position_intrinsic(self) -> Callable[[Tuple[float]], float]:
    """Returns an intrinsic producing a tuple
    of length support positions at which to evaluate the kernel,
    relative to an integer UV grid index"""
    return es_kernel_positions_factory(self.offsets)

  @property
  def kernel_intrinsic(self) -> Callable:
    """Returns an intrinsic produceinga tuple
    of length support kernel values, given a tuple of
    positions at which to evaluate the kernel, relative
    to a UV grid position"""
    return es_kernel_factory(self.kernel_fn)
