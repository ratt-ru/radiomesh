import ctypes

import numba
import numpy as np
from llvmlite import ir
from numba.core import types
from numba.core.errors import TypingError
from numba.extending import intrinsic

try:
  libc = ctypes.CDLL("libc.so.6")
except OSError:
  libc = ctypes.CDLL("libc.so")

os_yield_fn = libc.sched_yield
os_yield_fn.argtypes = []


@intrinsic
def lock(typingctx, lock: types.Array, idx: types.UniTuple):
  if not isinstance(lock, types.Array) or not isinstance(lock.dtype, types.Integer):
    raise TypingError(f"lock {lock} must be an Array of integers")

  if (
    not isinstance(idx, types.UniTuple)
    or not isinstance(idx.dtype, types.Integer)
    or len(idx) != lock.ndim
  ):
    raise TypingError(f"idx {idx} must be a Tuple of length {lock.ndim} integers")

  sig = types.bool(lock, idx)

  def yield_wrapper():
    print("Yielding")
    os_yield_fn()

  def codegen(context, builder, signature, args):
    lock, idx = args
    lock_type, idx_type = signature.args
    llvm_lock_type = context.get_value_type(lock_type.dtype)
    lock_array = context.make_array(lock_type)(context, builder, lock)

    native_idx = [builder.extract_value(idx, i) for i in range(len(idx_type))]
    out_ptr = builder.gep(lock_array.data, native_idx)

    loop_cond = builder.append_basic_block(name="lock.while.cond")
    loop_body = builder.append_basic_block(name="lock.while.body")
    loop_end = builder.append_basic_block(name="lock.while.end")

    # Save the starting block and branch to the conditional block
    # start_block = builder.basic_block
    builder.branch(loop_cond)

    with builder.goto_block(loop_cond):
      # count = builder.phi(ir.IntType(64), name="lock.while.index")
      xchng_result = builder.cmpxchg(
        out_ptr,
        ir.Constant(llvm_lock_type, 0),
        ir.Constant(llvm_lock_type, 1),
        ordering="acquire",
        failordering="monotonic",
      )
      # old_value = builder.extract_value(xchng_result, 0)
      success = builder.extract_value(xchng_result, 1)
      pred = builder.icmp_signed("==", success, success.type(1))
      builder.cbranch(pred, loop_end, loop_body)

    with builder.goto_block(loop_body):
      # next_count = builder.add(count, count.type(1))
      context.compile_internal(builder, yield_wrapper, types.none(), [])
      builder.branch(loop_cond)

    # Add incoming values to the PHI node
    # count.add_incoming(count.type(0), start_block) # From the entry block
    # count.add_incoming(next_count, loop_body) # From the loop body block

    builder.position_at_end(loop_end)
    return ir.Constant(ir.IntType(1), 1)

  return sig, codegen


if __name__ == "__main__":

  @numba.njit(nogil=True)
  def f(a, i):
    return lock(a, i)

  print(f(np.full(10, 1, np.int32), (0,)))
