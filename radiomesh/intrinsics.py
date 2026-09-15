"""Re-export shim: general-purpose intrinsics moved to rarg-numba-patterns,
radiomesh-specific ones remain in _intrinsics_extra."""

from numba.core import cgutils, errors, types
from numba.extending import intrinsic
from numba.np.arrayobj import populate_array
from rarg_numba_patterns.intrinsics import (  # noqa: F401
  accumulate_data,
  atomic_rmw_intrinsic,
  field_ptr,
  item_ptr,
  load_data,
  overload_atomic_rmw,
  overload_field_ptr,
  overload_item_ptr,
)

from radiomesh._intrinsics_extra import (  # noqa: F401
  apply_flags,
  apply_weights,
  check_args,
)


@intrinsic
def stack_array(typingctx, shape, dtype):
  """Allocate an uninitialised, C-contiguous array of the given
  `shape` and `dtype` on the stack (LLVM ``alloca``).

  The returned array is not reference counted and its storage
  is only valid until the enclosing function returns -- never
  store it in a longer-lived structure or return it.
  Allocations whose size is not compile-time constant are emitted
  in place, so avoid calling this in a loop with a dynamic shape.
  """
  if not isinstance(shape, types.BaseTuple) or not all(
    isinstance(e, types.Integer) for e in shape
  ):
    raise errors.TypingError(f"shape {shape} should be a tuple of integers")

  if not isinstance(dtype, types.NumberClass):
    raise errors.TypingError(f"dtype {dtype} should be a dtype")

  return_type = types.Array(dtype.instance_type, len(shape), "C")
  sig = return_type(shape, dtype)

  def codegen(context, builder, signature, args):
    shape, _ = args
    shape_type, _ = signature.args
    return_type = signature.return_type
    ndim = return_type.ndim

    llvm_dtype = context.get_data_type(return_type.dtype)
    itemsize = context.get_abi_sizeof(llvm_dtype)

    # Extract the dimensions, casting them to intp
    dims = [
      context.cast(builder, d, dt, types.intp)
      for d, dt in zip(cgutils.unpack_tuple(builder, shape, ndim), shape_type)
    ]

    # Number of elements and C-contiguous strides
    nelements = context.get_constant(types.intp, 1)
    strides = [None] * ndim
    stride = context.get_constant(types.intp, itemsize)

    for d in range(ndim - 1, -1, -1):
      strides[d] = stride
      stride = builder.mul(stride, dims[d])
      nelements = builder.mul(nelements, dims[d])

    data = builder.alloca(llvm_dtype, size=nelements, name="stack_array")

    array = context.make_array(return_type)(context, builder)
    populate_array(
      array,
      data=data,
      shape=dims,
      strides=strides,
      itemsize=context.get_constant(types.intp, itemsize),
      meminfo=None,
    )

    return array._getvalue()

  return sig, codegen
