from typing import Callable, Tuple

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, overload_method, register_jitable


@intrinsic(prefer_literal=True)
def load_data(
  typingctx,
  array: types.Array,
  index: types.UniTuple,
  ndata: types.IntegerLiteral,
  axis: types.IntegerLiteral,
) -> Tuple[Signature, Callable]:
  """An intrinsic that retrieves an `ndata` tuple of values
  from an array at a given `axis` and `index`:

  .. code-block:: python

    assert array.shape[axis] == ndata
    data = array[index[:axis] + (Ellipsis,) + index[axis:]]

  `index` should be a tuple of array.ndim - 1 integer values.
  `axis` references the dimension not referenced by `index` and
  should be of length ndata.
  """

  if not isinstance(ndata, types.IntegerLiteral):
    raise RequireLiteralValue(f"'ndata' ({ndata}) must be an IntegerLiteral")

  if not isinstance(axis, types.IntegerLiteral):
    raise RequireLiteralValue(f"'axis' ({axis}) must be an IntegerLiteral")

  if not isinstance(array, types.Array) or array.ndim != len(index) + 1:
    raise TypingError(f"'array' ({array}) should be a {len(index) + 1}D array")

  if not isinstance(index, types.BaseTuple) or not all(
    isinstance(i, types.Integer) for i in index
  ):
    raise TypingError(f"'index' {index} must be a tuple of integers")

  return_type = types.Tuple([array.dtype] * ndata.literal_value)
  sig = return_type(array, index, ndata, axis)
  ax = ndata.literal_value if axis.literal_value < 0 else axis.literal_value

  def index_factory(pol):
    """Index array with the first N-1 indices combined with pol"""
    return lambda array, index: array[index[:ax] + (pol,) + index[ax:]]

  def codegen(context, builder, signature, args):
    array_type, index_type, _, _ = signature.args
    array, index, _, _ = args
    llvm_ret_type = context.get_value_type(signature.return_type)
    pol_tuple = cgutils.get_null_value(llvm_ret_type)

    for p in range(ndata.literal_value):
      sig = array_type.dtype(array_type, index_type)
      value = context.compile_internal(builder, index_factory(p), sig, [array, index])
      pol_tuple = builder.insert_value(pol_tuple, value, p)

    return pol_tuple

  return sig, codegen


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


@intrinsic(prefer_literal=True)
def accumulate_data(
  typingctx,
  data: types.UniTuple,
  array: types.Array,
  index: types.UniTuple,
  axis: types.IntegerLiteral,
) -> Tuple[Signature, Callable]:
  """An intrinsic that accumulates a `data` tuple of values
  into an array at a given `axis` and `index`:

  .. code-block:: python

    assert len(data_tuple) == array.shape[axis]
    array[index[:axis] + (Ellipsis,) + index[axis:]] += data_tuple

  `index` should be a tuple of array.ndim - 1 integer values.
  `axis` references the dimension not referenced by `index` and
  should be of length ndata.
  """
  if not isinstance(axis, types.IntegerLiteral):
    raise RequireLiteralValue(f"'axis' ({axis}) must be an IntegerLiteral")

  if not isinstance(data, types.UniTuple):
    raise TypingError(f"'data' ({data}) should be a tuple")

  if not isinstance(array, types.Array) or array.ndim != len(index) + 1:
    raise TypingError(f"'array' ({array}) should be a {len(index) + 1}D array")

  if not isinstance(index, types.BaseTuple) or not all(
    isinstance(i, types.Integer) for i in index
  ):
    raise TypingError(f"'index' {index} must be a tuple of integers")

  sig = types.none(data, array, index, axis)
  # -1 signifies the axis should be at the end of the tuple
  ax = len(data) if axis.literal_value < 0 else axis.literal_value

  def assign_factory(pol):
    """Index array with the N-1 indices combined with pol"""

    def assign(value, array, index):
      array[index[:ax] + (pol,) + index[ax:]] += value[pol]

    return assign

  def codegen(context, builder, signature, args):
    data, array, index, _ = args
    data_type, array_type, index_type, _ = signature.args
    sig = types.none(data_type, array_type, index_type)

    for p in range(len(data_type)):
      context.compile_internal(builder, assign_factory(p), sig, [data, array, index])

    return None

  return sig, codegen


@intrinsic(prefer_literal=True)
def field_ptr(typingctx, record_ptr, field):
  """Return a pointer to the field within the supplied record"""
  if not isinstance(field, types.StringLiteral):
    raise RequireLiteralValue(f"{field} must be a StringLiteral")

  if not (
    isinstance(record_ptr, types.CPointer)
    and isinstance(record_dtype := record_ptr.dtype, types.Record)
    and field.literal_value in record_dtype.fields
  ):
    raise TypingError(
      f"record {record_ptr} must be a CPointer "
      f"to a Record containing a {field.literal_value} member"
    )

  field_type, field_offset, _, _ = record_ptr.dtype.fields[field.literal_value]
  field_ptr_type = types.CPointer(field_type)
  sig = field_ptr_type(record_ptr, field)

  def codegen(context, builder, signature, args):
    record_ptr, _ = args
    llvm_field_ptr_type = context.get_value_type(field_type).as_pointer()
    field_ptr = builder.gep(
      record_ptr,
      [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), field_offset)],
    )
    return builder.bitcast(field_ptr, llvm_field_ptr_type)

  return sig, codegen


@overload_method(types.CPointer, "field_ptr", prefer_literal=True)
def overload_field_ptr(ptr, field):
  return lambda ptr, field: field_ptr(ptr, field)


@intrinsic(prefer_literal=True)
def item_ptr(typingctx, array, index):
  """Return a pointer to the item in the array at the specified index"""
  if not isinstance(array, types.Array):
    raise TypingError(f"array {array} must be an Array")

  if not (
    isinstance(index, types.UniTuple)
    and isinstance(index.dtype, types.Integer)
    and array.ndim == len(index)
  ):
    raise TypingError(
      f"index {index} must be a Tuple of Integers of length {array.ndim}"
    )

  def codegen(context, builder, signature, args):
    array, index = args
    array_type, index_type = signature.args
    return cgutils.get_item_pointer(
      context,
      builder,
      array_type,
      context.make_array(array_type)(context, builder, array),
      [builder.extract_value(index, i) for i in range(len(index_type))],
    )

  return types.CPointer(array.dtype)(array, index), codegen


@overload_method(types.Array, "item_ptr", prefer_literal=True)
def overload_item_ptr(array, *index):
  return lambda array, *index: item_ptr(array, index)


@intrinsic(prefer_literal=True)
def atomic_rmw_intrinsic(typingctx, ptr, op, value, ordering):
  if not isinstance(op, types.StringLiteral):
    raise RequireLiteralValue(f"{op} is not a StringLiteral")

  if not isinstance(ordering, types.StringLiteral):
    raise RequireLiteralValue(f"{ordering} is not a StringLiteral")

  OP = op.literal_value
  ORDERING = ordering.literal_value

  def codegen(context, builder, signature, args):
    (ptr, _, value, _) = args
    return builder.atomic_rmw(OP, ptr, value, ORDERING)

  return ptr.dtype(ptr, op, value, ordering), codegen


@overload_method(types.CPointer, "atomic_rmw")
def overload_atomic_rmw(ptr, op, value, ordering=None):
  """This dispatches to llvmlite's atomic_rmw instruction
  https://llvmlite.readthedocs.io/en/latest/user-guide/ir/ir-builder.html#llvmlite.ir.atomic_rmw
  """

  def impl(ptr, op, value, ordering=None):
    if ordering is None:
      ordering = "acq_rel"

    return atomic_rmw_intrinsic(ptr, op, value, ordering)

  return impl


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
