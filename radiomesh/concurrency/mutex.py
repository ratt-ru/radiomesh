"""This module contains code for some simple atomic spinlocks

See the discussion here
https://numba.discourse.group/t/phi-node-error-when-creating-a-while-loop-for-an-atomic-spinlock/3046
"""

import ctypes

import numba
import numpy as np
from llvmlite import ir
from numba.core import types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.extending import intrinsic

try:
  libc = ctypes.CDLL("libc.so.6")
except OSError:
  libc = ctypes.CDLL("libc.so")

os_yield_fn = libc.sched_yield
os_yield_fn.argtypes = []


@intrinsic(prefer_literal=True)
def lock_op(
  typingctx, lock: types.Array, idx: types.UniTuple, operation: types.StringLiteral
):
  if not isinstance(operation, types.StringLiteral) or operation.literal_value not in {
    "lock",
    "unlock",
  }:
    raise RequireLiteralValue(
      f"'operation' {operation} must be a StringLiteral "
      f"set to either lock or unlock"
    )

  if not isinstance(lock, types.Array) or not isinstance(lock.dtype, types.Integer):
    raise TypingError(f"lock {lock} must be an Array of integers")

  if (
    not isinstance(idx, types.UniTuple)
    or not isinstance(idx.dtype, types.Integer)
    or len(idx) != lock.ndim
  ):
    raise TypingError(f"idx {idx} must be a Tuple of length {lock.ndim} integers")

  sig = types.bool(lock, idx, operation)

  def yield_wrapper_idx(i):
    os_yield_fn()

  def codegen(context, builder, signature, args):
    lock, idx, _ = args
    lock_type, idx_type, _ = signature.args
    llvm_lock_type = context.get_value_type(lock_type.dtype)
    lock_array = context.make_array(lock_type)(context, builder, lock)

    index_type = types.int64
    ll_index_type = context.get_value_type(index_type)
    native_idx = [builder.extract_value(idx, i) for i in range(len(idx_type))]
    out_ptr = builder.gep(lock_array.data, native_idx)

    loop_cond = builder.append_basic_block(name="lock.while.cond")
    loop_body = builder.append_basic_block(name="lock.while.body")
    loop_end = builder.append_basic_block(name="lock.while.end")

    # Save the starting block and branch to the conditional block
    start_block = builder.block
    builder.branch(loop_cond)

    if operation.literal_value == "lock":
      pre_xchg_value = ir.Constant(llvm_lock_type, 0)
      post_xchg_value = ir.Constant(llvm_lock_type, 1)
    elif operation.literal_value == "unlock":
      pre_xchg_value = ir.Constant(llvm_lock_type, 1)
      post_xchg_value = ir.Constant(llvm_lock_type, 0)
    else:
      raise ValueError(f"Invalid operation.literal_value " f"{operation.literal_value}")

    with builder.goto_block(loop_cond):
      count_phi = builder.phi(ll_index_type, name="lock.while.index")
      xchng_result = builder.cmpxchg(
        out_ptr,
        pre_xchg_value,
        post_xchg_value,
        ordering="acquire",
        failordering="monotonic",
      )
      success = builder.extract_value(xchng_result, 1)
      pred = builder.icmp_signed("==", success, success.type(1))
      builder.cbranch(pred, loop_end, loop_body)

    with builder.goto_block(loop_body):
      next_count = builder.add(count_phi, count_phi.type(1))
      context.compile_internal(
        builder, yield_wrapper_idx, types.none(index_type), [next_count]
      )
      branch_block = builder.block
      builder.branch(loop_cond)

    # Add incoming values to the PHI node
    count_phi.add_incoming(count_phi.type(0), start_block)  # From the entry block
    count_phi.add_incoming(next_count, branch_block)  # From the loop body block

    builder.position_at_end(loop_end)
    return ir.Constant(ir.IntType(1), 1)

  return sig, codegen


if __name__ == "__main__":
  """Test script"""

  @numba.njit(nogil=True)
  def lock_index(a, i):
    return lock_op(a, i, "lock")

  @numba.njit(nogil=True)
  def unlock_index(a, i):
    return lock_op(a, i, "unlock")

  locks = np.full(10, 0, np.int32)

  print(lock_index(locks, (0,)))
  print(lock_index(locks, (1,)))
  print(unlock_index(locks, (0,)))
  print(lock_index(locks, (0,)))
  print(unlock_index(locks, (1,)))
