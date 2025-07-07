import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from numba.core import cgutils, types
from numba.extending import intrinsic


def es_kernel_factory(betak: float) -> Tuple[Callable, Callable]:
  @intrinsic
  def es_kernel_positions(typingctx, position, index):
    """Return a tuple of kernel :code:`position - float(index)` position.

    Args:
      position: Tuple of kernel positions
      index: uv grid index

    Returns:
      Tuple of kernel index positions
    """
    if not isinstance(position, types.UniTuple) or not isinstance(
      position.dtype, types.Float
    ):
      raise TypeError(f"'x' ({position}) must be a tuple of floats")

    if not isinstance(index, types.Integer):
      raise TypeError(f"'index' ({index}) must be an integer")

    return_type = types.Tuple([position.dtype] * len(position))
    sig = return_type(position, index)

    def x_index_impl(position, index):
      return position + float(index)

    def codegen(context, builder, signature, args):
      position, index = args
      position_type, index_type = signature.args
      llvm_ret_type = context.get_value_type(signature.return_type)
      index_tuple = cgutils.get_null_value(llvm_ret_type)
      index_sig = signature.return_type.dtype(position_type.dtype, index_type)

      for i in range(len(position_type)):
        p_value = builder.extract_value(position, i)
        x_index = context.compile_internal(
          builder, x_index_impl, index_sig, [p_value, index]
        )
        index_tuple = builder.insert_value(index_tuple, x_index, i)

      return index_tuple

    return sig, codegen

  @intrinsic
  def es_kernel(typingctx, x_index, grid):
    """Evaluates the es kernel at :code:`(x_index - grid + 0.5) / half_support`

    Args:
      x_index: Tuple of values at which to evaluate the kernel
      grid: uv grid position

    Returns:
      Tuple of kernel values
    """
    if not isinstance(x_index, types.UniTuple) or not isinstance(
      x_index.dtype, types.Float
    ):
      raise TypeError(f"'x' ({x_index}) must be a tuple of floats")

    if not isinstance(grid, types.Float):
      raise TypeError("'grid' must be a float")

    support = len(x_index)
    half_support = support / 2.0
    return_type = types.Tuple([x_index.dtype] * support)
    sig = return_type(x_index, grid)

    def kernel_impl(x_index, grid):
      # Function is only defined for [-1.0, 1.0]
      # Potentially evaluating SIMD. Prefer vectorisation
      # followed by zeroing the result (if necessary)
      x = (x_index - grid + 0.5) / half_support
      undefined = -1.0 <= x <= 1.0
      x = 0.0 if undefined else x
      result = np.exp(betak * (np.sqrt(1.0 - x * x) - 1.0))
      return result if undefined else 0.0

    def codegen(context, builder, signature, args):
      x_index, grid = args
      x_index_type, grid_type = signature.args
      llvm_ret_type = context.get_value_type(signature.return_type)
      kernel_tuple = cgutils.get_null_value(llvm_ret_type)
      kernel_sig = signature.return_type.dtype(x_index_type.dtype, grid_type)

      for i in range(len(x_index_type)):
        x = builder.extract_value(x_index, i)
        ktuple = context.compile_internal(builder, kernel_impl, kernel_sig, [x, grid])
        kernel_tuple = builder.insert_value(kernel_tuple, ktuple, i)

      return kernel_tuple

    return sig, codegen

  return es_kernel_positions, es_kernel


DEFAULT_EPSILON: float = 2e-13
DEFAULT_OVERSAMPLING: int = 2
DEFAULT_BETA: float = 2.3
DEFAULT_E0: float = 0.5


@dataclass
class ESKernelParameters:
  """Dataclass holding parameters defining an ES kernel of the form
  :code:`math.exp(beta * (math.pow(1.0 - x * x, 0.5) - 1.0))`
  """

  # Desired wgridder accuracy
  epsilon: float
  # ES kernel parameters
  # Kernel support
  support: int
  # Oversampling factor.
  # Corresponds to :code:`ofactor` within the ducc0 wgridder code base
  oversampling: int
  beta: float
  e0: float

  def __init__(
    self,
    epsilon: float = DEFAULT_EPSILON,
    oversampling: int = DEFAULT_OVERSAMPLING,
    beta: float = DEFAULT_BETA,
    e0=DEFAULT_E0,
    wgridding: bool = False,
  ):
    self.epsilon = epsilon
    self.oversampling = oversampling
    wgrid_factor = 3.0 if wgridding else 2.0
    self.support = int(math.ceil(math.log10(wgrid_factor * 1.0 / epsilon))) + 1
    self.beta = beta
    self.e0 = e0

  @property
  def half_support(self, integer=False) -> int | float:
    return self.support // 2 if integer else self.support / 2.0
