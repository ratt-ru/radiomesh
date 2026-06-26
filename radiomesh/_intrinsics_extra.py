from typing import Callable, Tuple

from numba.core import cgutils, types
from numba.core.errors import TypingError
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, register_jitable


@intrinsic(prefer_literal=True)
def apply_weights(
  typingctx, data: types.UniTuple, weight: types.UniTuple | types.Float
) -> Tuple[Signature, Callable]:
  """Applies weight to a tuple of data"""

  if not isinstance(data, types.UniTuple):
    raise TypingError(f"'data' ({data}) must be a tuple")

  is_float_weight = isinstance(weight, types.Float)
  is_tuple_weight = isinstance(weight, types.UniTuple) and len(weight) == len(data)

  if not is_float_weight and not is_tuple_weight:
    raise TypingError(
      f"'weight' ({weight}) must be a float or "
      f"a tuple of values of length {len(data)}"
    )

  unified_type = typingctx.unify_types(
    data.dtype, weight if is_float_weight else weight.dtype
  )
  return_type = types.Tuple([unified_type] * len(data))
  sig = return_type(data, weight)

  def apply_weight_factory(p):
    if is_float_weight:
      return lambda d, w: d[p] * w
    else:
      return lambda d, w: d[p] * w[p]

  def codegen(context, builder, signature, args):
    data_type, weight_type = signature.args
    data, weight = args
    llvm_ret_type = context.get_value_type(signature.return_type)
    return_tuple = cgutils.get_null_value(llvm_ret_type)

    for p in range(len(data_type)):
      # Apply weights to data
      sig = unified_type(data_type, weight_type)
      value = context.compile_internal(
        builder, apply_weight_factory(p), sig, [data, weight]
      )
      return_tuple = builder.insert_value(return_tuple, value, p)

    return return_tuple

  return sig, codegen


@intrinsic(prefer_literal=True)
def apply_flags(
  typingctx, data: types.UniTuple, flags: types.UniTuple
) -> Tuple[Signature, Callable]:
  """Applies flags to a tuple of data"""

  if not isinstance(data, types.UniTuple):
    raise TypingError(f"'data' ({data}) must be a tuple")

  if not isinstance(flags, types.UniTuple) or len(flags) != len(data):
    raise TypingError(f"'flags' ({flags} must be a tuple of length {len(data)})")

  return_type = types.Tuple([data.dtype] * len(data))
  sig = return_type(data, flags)

  def codegen(context, builder, signature, args):
    data, flags = args
    data_type, flags_type = signature.args
    llvm_ret_type = context.get_value_type(signature.return_type)
    return_tuple = cgutils.get_null_value(llvm_ret_type)

    for p in range(len(data_type)):
      # Apply flags to data
      factory_sig = data_type.dtype(data_type, flags_type)
      value = context.compile_internal(
        builder, lambda d, f: 0 if f[p] != 0 else d[p], factory_sig, [data, flags]
      )
      return_tuple = builder.insert_value(return_tuple, value, p)

    return return_tuple

  return sig, codegen


@register_jitable
def check_args(uvw, visibilities, weights, flags, frequencies, nschema_pol):
  """Check gridder argument shapes"""
  if not (visibilities.shape == weights.shape == flags.shape):
    raise ValueError("Shapes of visibilities, weights and flags do not match")

  if uvw.shape[:-1] != visibilities.shape[:-2]:
    raise ValueError("uvw and visibility shapes do not match in the primary dimensions")

  if uvw.shape[-1] != 3:
    raise ValueError("The last axis of uvw should have 3 components")

  if visibilities.shape[-1] != nschema_pol:
    raise ValueError(
      f"Number of visibility polarisations {visibilities.shape[-1]}"
      f"does not match the number of schema polarisations {nschema_pol} "
    )

  if frequencies.shape[0] != visibilities.shape[-2]:
    raise ValueError(
      "Frequency shape does not match the visibility shape "
      "in the frequency dimension"
    )

  if not (visibilities.shape == weights.shape == flags.shape):
    raise ValueError("Shapes of visibilities, weights and flags do not match")
