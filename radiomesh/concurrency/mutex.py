"""This module contains code for some simple atomic spinlocks

See the discussion here
https://numba.discourse.group/t/phi-node-error-when-creating-a-while-loop-for-an-atomic-spinlock/3046
"""

import ctypes

import numba
import numpy as np
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.extending import intrinsic

try:
  libc = ctypes.CDLL("libc.so.6")
except OSError:
  libc = ctypes.CDLL("libc.so")

os_yield_fn = libc.sched_yield
os_yield_fn.argtypes = []


def _emit_lock_op(context, builder, operation, ptr, llvm_lock_type):
  """Emit IR for a lock or unlock operation on an already-resolved pointer."""

  def yield_wrapper_idx(i):
    os_yield_fn()

  index_type = types.int64
  ll_index_type = context.get_value_type(index_type)

  if operation.literal_value == "unlock":
    builder.store_atomic(
      ir.Constant(llvm_lock_type, 0),
      ptr,
      ordering="release",
      align=llvm_lock_type.width,
    )
    return ir.Constant(ir.IntType(1), 1)

  # Spin until the lock is acquired
  loop_cond = builder.append_basic_block(name="lock.while.cond")
  loop_body = builder.append_basic_block(name="lock.while.body")
  loop_end = builder.append_basic_block(name="lock.while.end")

  start_block = builder.block
  builder.branch(loop_cond)

  with builder.goto_block(loop_cond):
    count_phi = builder.phi(ll_index_type, name="lock.while.index")
    xchng_result = builder.cmpxchg(
      ptr,
      ir.Constant(llvm_lock_type, 0),
      ir.Constant(llvm_lock_type, 1),
      ordering="acquire",
      failordering="acquire",
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

  count_phi.add_incoming(count_phi.type(0), start_block)
  count_phi.add_incoming(next_count, branch_block)

  builder.position_at_end(loop_end)
  return ir.Constant(ir.IntType(1), 1)


@intrinsic(prefer_literal=True)
def atomic_lock(typingctx, lock_type):
  """Allocates an integer suitable for use as an atomic lock
  on the stack and returns a pointer to that integer"""

  if not isinstance(lock_type, (types.NumberClass, types.DType)):
    raise TypingError(f"{lock_type} is not a NumberClass or DType")

  def codegen(context, builder, signature, args):
    llvm_lock_type = context.get_value_type(lock_type.dtype)
    ptr = cgutils.alloca_once(builder, llvm_lock_type)
    builder.store(ir.Constant(llvm_lock_type, 0), ptr)
    return ptr

  return types.CPointer(lock_type.dtype)(lock_type), codegen


@intrinsic(prefer_literal=True)
def lock_int_op(typingctx, lock: types.CPointer, operation: types.StringLiteral):
  """Performs an atomic lock at a pointer location"""
  if not isinstance(operation, types.StringLiteral) or operation.literal_value not in {
    "lock",
    "unlock",
  }:
    raise RequireLiteralValue(
      f"'operation' {operation} must be a StringLiteral "
      f"set to either lock or unlock"
    )

  if not (isinstance(lock, types.CPointer) and isinstance(lock.dtype, types.Integer)):
    raise TypingError(f"lock {lock} must be an CPointer to an integer")

  sig = types.bool(lock, operation)

  def codegen(context, builder, signature, args):
    lock, _ = args
    lock_type, _ = signature.args
    llvm_lock_type = context.get_value_type(lock_type.dtype)
    return _emit_lock_op(context, builder, operation, lock, llvm_lock_type)

  return sig, codegen


@intrinsic(prefer_literal=True)
def lock_array_op(
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

  def codegen(context, builder, signature, args):
    lock, idx, _ = args
    lock_type, idx_type, _ = signature.args
    llvm_lock_type = context.get_value_type(lock_type.dtype)
    lock_array = context.make_array(lock_type)(context, builder, lock)
    native_idx = [builder.extract_value(idx, i) for i in range(len(idx_type))]
    out_ptr = builder.gep(lock_array.data, native_idx)
    return _emit_lock_op(context, builder, operation, out_ptr, llvm_lock_type)

  return sig, codegen


if __name__ == "__main__":
  """Test script"""

  @numba.njit(nogil=True)
  def lock_index():
    ll = numba.typed.List()
    ll.append(atomic_lock(np.dtype(np.uint8)))
    ll.append(atomic_lock(np.dtype(np.uint8)))
    ll.append(atomic_lock(np.dtype(np.uint8)))
    lock_int_op(ll[0], "lock")
    lock_int_op(ll[0], "unlock")

    return ll

  print(len(lock_index()))

  if True:

    @numba.njit(nogil=True)
    def lock_index(a, i):
      return lock_array_op(a, i, "lock")

    @numba.njit(nogil=True)
    def unlock_index(a, i):
      return lock_array_op(a, i, "unlock")

    locks = np.full(10, 0, np.int32)

    print(lock_index(locks, (0,)))
    print(lock_index(locks, (1,)))
    print(unlock_index(locks, (0,)))
    print(lock_index(locks, (0,)))
    print(unlock_index(locks, (1,)))
