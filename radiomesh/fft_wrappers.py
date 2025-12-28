import ctypes
from dataclasses import dataclass
from typing import Dict, Tuple

import llvmlite.binding as llvm
import numba
import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.core.errors import RequireLiteralValue, TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import intrinsic, overload, register_jitable
from numba.np.numpy_support import as_dtype, from_dtype

from radiomesh.ufunc_minimal import MinimalPyUFuncObject


@dataclass(frozen=True, order=True, unsafe_hash=True)
class UFuncKey:
  """A key representing a specific ufunc and combination of argument types"""

  name: str
  arg_types: Tuple[int, ...]


FFT_UFUNC_TABLE_SYMBOL = "_radiomesh_pocketfft_ufunc_table"
FFT_UFUNCS = ["fft", "ifft", "rfft_n_even", "rfft_n_odd", "irfft"]
FFTUfuncTableType = MinimalPyUFuncObject * len(FFT_UFUNCS)
FFT_UFUNC_TABLE = None
"""(ufunc, arg_types) -> (ufunc_id, type_variant_id)"""
FFT_UFUNC_MAP: Dict[UFuncKey, Tuple[int, int]] = {}


def _install_ufunc_table(pfu):
  """Create the FFT ufunc table and make it visible as an LLVM symbol
  accessible from the numba intrinsic"""
  global FFT_UFUNC_TABLE
  global FFT_UFUNC_MAP

  ufuncs = []

  for u, ufunc_name in enumerate(FFT_UFUNCS):
    if (fft_ufunc := getattr(pfu, ufunc_name, False)) is False:
      raise ValueError(f"{ufunc_name} not found in pocketfft module")

    ufunc = MinimalPyUFuncObject.from_address(id(fft_ufunc))
    ufuncs.append(ufunc)

    for t in range(ufunc.ntypes):
      arg_type_ids = tuple(ufunc.types[t * ufunc.nargs + a] for a in range(ufunc.nargs))
      FFT_UFUNC_MAP[UFuncKey(ufunc_name, tuple(arg_type_ids))] = (u, t)

  FFT_UFUNC_TABLE = FFTUfuncTableType(*ufuncs)
  llvm.add_symbol(FFT_UFUNC_TABLE_SYMBOL, ctypes.addressof(FFT_UFUNC_TABLE))


try:
  from numpy.fft._pocketfft import pfu
except ImportError:
  raise ImportError("numpy does not appear to bundle pocketfft ufuncs")
else:
  _install_ufunc_table(pfu)


def intrinsic_fft_factory(ufunc_name):
  """Generate intrinsics for calling each fft ufunc variant"""

  def _invoke_fft_ufunc(typingctx, a, factor, out):
    a_dtype = as_dtype(a.dtype)
    factor_dtype = as_dtype(factor)
    out_dtype = as_dtype(out.dtype)

    key = UFuncKey(ufunc_name, (a_dtype.num, factor_dtype.num, out_dtype.num))

    try:
      ufunc_id, type_variant_id = FFT_UFUNC_MAP[key]
    except KeyError:
      raise KeyError(f"{key} ufunc type variant not registered in the fft ufunc table")

    sig = out(a, factor, out)

    def codegen(context, builder, signature, args):
      a, factor, out = args
      a_type, factor_type, out_type = signature.args
      out_type = signature.return_type

      # Establish structs over the inut and out arrays
      # so that we can access their internal data
      a_struct = cgutils.create_struct_proxy(a_type)(context, builder, a)
      out_struct = cgutils.create_struct_proxy(out_type)(context, builder, out)

      # Allocate stack space for factor to obtain a useable pointer
      float_ptr = builder.alloca(factor.type)
      builder.store(factor, float_ptr)

      # Get the index type used within numpy ufuncs
      # and allocate some indexing constants
      index_type = context.get_value_type(from_dtype(np.intp))
      ZERO = ir.Constant(index_type, 0)
      ONE = ir.Constant(index_type, 1)
      TWO = ir.Constant(index_type, 2)
      THREE = ir.Constant(index_type, 3)
      FOUR = ir.Constant(index_type, 4)

      # Allocate and populate args, dimensions and steps
      # See fft_loop in _pocketfft_umath.cpp

      # Allocate stack space for ufunc args
      # (ip, fp, op) in fft_loop
      # (a.data, factor_ptr, out.data)
      voidptr_t = cgutils.voidptr_t
      args_ptr = builder.alloca(voidptr_t, 3)
      builder.store(
        builder.bitcast(a_struct.data, voidptr_t), builder.gep(args_ptr, [ZERO])
      )
      builder.store(builder.bitcast(float_ptr, voidptr_t), builder.gep(args_ptr, [ONE]))
      builder.store(
        builder.bitcast(out_struct.data, voidptr_t), builder.gep(args_ptr, [TWO])
      )

      # Allocate stack space for ufunc dimensions
      # (n_outer, nin, nout) in fft_loop
      # (1, a.size, out.size)
      dims_ptr = builder.alloca(index_type, 3)
      builder.store(ONE, builder.gep(dims_ptr, [ZERO]))
      builder.store(a_struct.nitems, builder.gep(dims_ptr, [ONE]))
      builder.store(out_struct.nitems, builder.gep(dims_ptr, [TWO]))

      a_nbytes = builder.mul(a_struct.nitems, a_struct.itemsize)
      out_nbytes = builder.mul(out_struct.nitems, out_struct.itemsize)
      a_strides_0 = builder.extract_value(a_struct.strides, 0)
      out_strides_0 = builder.extract_value(out_struct.strides, 0)

      # Allocate stack space for ufunc steps
      # (si, sf, so, step_in, step_out) in fft_loop
      # (A.nbytes, 0, out.nbytes, A.strides[0], out.strides[0])
      step_ptr = builder.alloca(index_type, 5)
      builder.store(a_nbytes, builder.gep(step_ptr, [ZERO]))
      builder.store(ZERO, builder.gep(step_ptr, [ONE]))
      builder.store(out_nbytes, builder.gep(step_ptr, [TWO]))
      builder.store(a_strides_0, builder.gep(step_ptr, [THREE]))
      builder.store(out_strides_0, builder.gep(step_ptr, [FOUR]))

      # Build LLVM function type matching PyUFuncGenericFunction:
      # void(char**, npy_intp*, npy_intp*, void*)
      intp_ptr_t = ir.PointerType(index_type)
      char_pp_t = ir.PointerType(ir.PointerType(ir.IntType(8)))
      fn_llvm_type = ir.FunctionType(
        ir.VoidType(),
        [char_pp_t, intp_ptr_t, intp_ptr_t, voidptr_t],
      )

      # Load fn_ptr and data_ptr at runtime from the registered ufunc table.
      # ufunc_id and type_variant_id are compile-time constants from typing;
      # byte offsets are stable across processes (ctypes struct layout).
      i8_t = ir.IntType(8)
      i64_t = ir.IntType(64)
      ufunc_size = ctypes.sizeof(MinimalPyUFuncObject)
      FUNCTION_OFFSET = ir.Constant(
        i64_t, ufunc_id * ufunc_size + MinimalPyUFuncObject.functions.offset
      )
      DATA_OFFSET = ir.Constant(
        i64_t, ufunc_id * ufunc_size + MinimalPyUFuncObject.data.offset
      )
      TYPE_VARIANT_OFFSET = ir.Constant(i64_t, type_variant_id)

      # Declare (or retrieve) the external table global
      if FFT_UFUNC_TABLE_SYMBOL not in builder.module.globals:
        table_glob = ir.GlobalVariable(
          builder.module,
          ir.ArrayType(i8_t, ctypes.sizeof(FFT_UFUNC_TABLE)),
          FFT_UFUNC_TABLE_SYMBOL,
        )
        table_glob.linkage = "external"
      else:
        table_glob = builder.module.globals[FFT_UFUNC_TABLE_SYMBOL]

      # Load ufunc function pointer: ufunc.functions[type_variant_id]
      fns_field = builder.gep(table_glob, [ZERO, FUNCTION_OFFSET])
      fn_address = builder.load(builder.bitcast(fns_field, ir.PointerType(i64_t)))
      fn_address_ptr = builder.inttoptr(fn_address, ir.PointerType(i64_t))
      fn_int = builder.load(builder.gep(fn_address_ptr, [TYPE_VARIANT_OFFSET]))
      fn_ptr = builder.inttoptr(fn_int, ir.PointerType(fn_llvm_type))

      # Load ufunc data pointer if data is non-null: ufunc.data[type_variant_id]
      if bool(FFT_UFUNC_TABLE[ufunc_id].data):
        data_field = builder.gep(table_glob, [ZERO, DATA_OFFSET])
        data_address = builder.load(builder.bitcast(data_field, ir.PointerType(i64_t)))
        data_address_ptr = builder.inttoptr(data_address, ir.PointerType(i64_t))
        data_int = builder.load(builder.gep(data_address_ptr, [TYPE_VARIANT_OFFSET]))
        data_ptr = builder.inttoptr(data_int, voidptr_t)
      else:
        data_ptr = builder.inttoptr(ZERO, voidptr_t)

      # Call the ufunc inner loop
      builder.call(
        fn_ptr,
        [
          builder.bitcast(args_ptr, char_pp_t),
          builder.bitcast(dims_ptr, intp_ptr_t),
          builder.bitcast(step_ptr, intp_ptr_t),
          data_ptr,
        ],
      )

      # Increment out's reference count as we re-use
      # the argument for the return value
      context.nrt.incref(builder, out_type, out)
      return out

    return sig, codegen

  _invoke_fft_ufunc.__name__ = f"{ufunc_name}_intrinsic"
  return intrinsic(prefer_literal=True)(_invoke_fft_ufunc)


fft_intrinsic = intrinsic_fft_factory("fft")
ifft_intrinsic = intrinsic_fft_factory("ifft")
rfft_even_intrinsic = intrinsic_fft_factory("rfft_n_even")
rfft_odd_intrinsic = intrinsic_fft_factory("rfft_n_odd")
irfft_intrinsic = intrinsic_fft_factory("irfft")


@register_jitable
def normalize_axis_index(axis, ndim):
  """Normalise the axis argument against the number of array dimensions"""
  if axis < 0:
    axis += ndim

  if axis < 0 or axis >= ndim:
    raise ValueError(f"axis {axis} is out of bounds for array of dimension {ndim}")

  return axis


def raw_fft_impl(a, n, axis, is_real, is_forward, norm, out):
  pass


@overload(raw_fft_impl, prefer_literal=True)
def raw_fft_overload(a, n, axis, is_real, is_forward, norm, out):
  if not isinstance(is_real, types.BooleanLiteral):
    raise RequireLiteralValue(f"is_real {is_real} is not a BooleanLiteral")

  if not isinstance(is_forward, types.BooleanLiteral):
    raise RequireLiteralValue(f"is_forward {is_forward} is not a BooleanLiteral")

  if not isinstance(norm, types.StringLiteral):
    raise RequireLiteralValue(f"norm {norm} is not a StringLiteral")

  if not isinstance(axis, types.Integer):
    raise TypingError(f"axis {axis} should be an Integer")

  if not isinstance(a, types.Array):
    raise TypingError(f"a {a} is not an Array")

  if not (IS_N_NONE := isinstance(n, types.NoneType)) and not isinstance(
    n, types.Integer
  ):
    raise TypingError(f"n {n} must be None or an Integer")

  if not (IS_OUT_NONE := isinstance(out, types.NoneType)):
    if not (
      isinstance(out, types.Array) and a.dtype == out.dtype and a.ndim == out.ndim
    ):
      raise TypingError(f"Types of a {a} and {out} do not match")

  IS_REAL = is_real.literal_value
  IS_FORWARD = is_forward.literal_value
  IS_RFFT = IS_REAL and IS_FORWARD
  NORM_VALUE = norm.literal_value
  NORM_ERR_STR = f'norm {NORM_VALUE} should be "backward", "ortho" or "forward"'

  # Reverse the normalization value if the direction is backward
  if not IS_FORWARD:
    if NORM_VALUE == "ortho":
      pass
    elif NORM_VALUE == "forward":
      NORM_VALUE = "backward"
    elif NORM_VALUE == "backward":
      NORM_VALUE = "forward"
    else:
      raise ValueError(NORM_ERR_STR)

  # Infer numba data types
  dummy = as_dtype(a.dtype).type(0)

  # We're bypassing the standard ufunc type promotion functionality,
  # the developer must be precise about the inputs and the mode
  if IS_REAL:
    if IS_FORWARD:
      if np.iscomplexobj(dummy):
        raise NotImplementedError(f"rfft for complex inputs {a}")
    else:
      if np.isrealobj(dummy):
        raise NotImplementedError(f"irfft for real inputs {a}")
  else:
    if not np.iscomplexobj(dummy):
      raise NotImplementedError((f"fft/ifft transform for real inputs {a}"))

  REAL_DTYPE = from_dtype(np.result_type(dummy.real.dtype, 1.0))

  if IS_OUT_NONE:
    if IS_REAL and not IS_FORWARD:
      OUT_DTYPE = REAL_DTYPE
    else:
      OUT_DTYPE = from_dtype(np.result_type(dummy.dtype, 1j))
  else:
    OUT_DTYPE = REAL_DTYPE

  def impl(a, n, axis, is_real, is_forward, norm, out):
    axis = normalize_axis_index(axis, a.ndim)
    n_actual = a.shape[axis] if IS_N_NONE else n

    # Determine the normalization factor
    if NORM_VALUE == "backward":
      factor = REAL_DTYPE(1.0)
    elif NORM_VALUE == "ortho":
      factor = REAL_DTYPE(np.reciprocal(np.sqrt(np.float64(n_actual))))
    elif NORM_VALUE == "forward":
      factor = REAL_DTYPE(np.reciprocal((np.float64(n_actual))))
    else:
      raise ValueError(NORM_ERR_STR)

    n_out = n_actual // 2 + 1 if IS_RFFT else n_actual

    # Allocate or validate the output
    if IS_OUT_NONE:
      new_shape = a.shape
      for i, s in enumerate(numba.literal_unroll(a.shape)):
        new_shape = tuple_setitem(new_shape, i, n_out if i == axis else s)

      out = np.empty(shape=new_shape, dtype=OUT_DTYPE)
    else:
      if a.ndim != out.ndim or a.shape[axis] != n_out:
        raise ValueError("output array has incorrect shape")

    # Invoke ufunc
    if IS_REAL:
      if IS_FORWARD:
        if n_actual % 2 == 0:
          return rfft_even_intrinsic(a, factor, out)
        else:
          return rfft_odd_intrinsic(a, factor, out)
      else:
        return irfft_intrinsic(a, factor, out)
    else:
      if IS_FORWARD:
        return fft_intrinsic(a, factor, out)
      else:
        return ifft_intrinsic(a, factor, out)

  return impl


@overload(np.fft.fft)
def fft(a, n=None, axis=-1, norm="backward", out=None):
  def impl(a, n=None, axis=-1, norm="backward", out=None):
    return raw_fft_impl(a, n, axis, False, True, norm, out)

  return impl


@overload(np.fft.ifft)
def ifft(a, n=None, axis=-1, norm="backward", out=None):
  def impl(a, n=None, axis=-1, norm="backward", out=None):
    return raw_fft_impl(a, n, axis, False, False, norm, out)

  return impl


@overload(np.fft.rfft)
def rfft(a, n=None, axis=-1, norm="backward", out=None):
  def impl(a, n=None, axis=-1, norm="backward", out=None):
    return raw_fft_impl(a, n, axis, True, True, norm, out)

  return impl


@overload(np.fft.irfft)
def irfft(a, n=None, axis=-1, norm="backward", out=None):
  def impl(a, n=None, axis=-1, norm="backward", out=None):
    if n is None:
      n = (a.shape[axis] - 1) * 2
    return raw_fft_impl(a, n, axis, True, False, norm, out)

  return impl
