import ctypes
from enum import IntEnum

import numpy as np


class CTypesEnum(IntEnum):
  """A ctypes-compatible enumeration base class"""

  @classmethod
  def from_param(cls, obj) -> int:
    return int(obj)


NPY_SAME_VALUE_CASTING_FLAG = 64


class NPY_CASTING(CTypesEnum):
  _NPY_ERROR_OCCURRED_IN_CAST = -1
  NPY_NO_CASTING = 0
  NPY_EQUIV_CASTING = 1
  NPY_SAFE_CASTING = 2
  NPY_SAME_KIND_CASTING = 3
  NPY_UNSAFE_CASTING = 4
  NPY_SAME_VALUE_CASTING = NPY_UNSAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG


# Numpy types expressed as ctypes
_CTYPES_NPY_INTP = np.ctypeslib.as_ctypes_type(np.intp)
_CTYPES_NPY_UINT32 = np.ctypeslib.as_ctypes_type(np.int32)

# ctypes function signature for a UFunc
PyUFuncGenericFunction = ctypes.CFUNCTYPE(
  # return type: void
  None,
  # char ** args
  ctypes.POINTER(ctypes.c_char_p),
  # npy_intp * dimensions
  ctypes.POINTER(_CTYPES_NPY_INTP),
  # npy_intp * steps
  ctypes.POINTER(_CTYPES_NPY_INTP),
  # void * data
  ctypes.c_void_p,
)

# ctypes signature for TypeResolutionFunc
# TODO: some arguments depend on undefined types
PyUFunc_TypeResolutionFunc = ctypes.CFUNCTYPE(
  # return type
  ctypes.c_int,
  # ufunc
  ctypes.POINTER(PyUFuncGenericFunction),
  # casting: NPY_CASTING
  ctypes.c_int,
  # operands: PyArrayObject
  ctypes.POINTER(ctypes.c_void_p),
  # type_tup: PyObject
  ctypes.c_void_p,
  # out_dtype: PyArray_Descr
  ctypes.POINTER(ctypes.c_void_p),
)


class MinimalPyUFuncObject(ctypes.Structure):
  """This structure replicates a subset of the following numpy struct

  https://numpy.org/devdocs/reference/c-api/types-and-structures.html#c.PyUFuncObject
  https://github.com/numpy/numpy/blob/main/numpy/_core/include/numpy/ufuncobject.h#L102
  """

  _fields_ = [
    # PyObject members
    ("ob_refcnt", ctypes.c_ssize_t),
    ("ob_type", ctypes.c_void_p),
    # PyUfuncObject members follow
    ("nin", ctypes.c_int),
    ("nout", ctypes.c_int),
    ("nargs", ctypes.c_int),
    ("identity", ctypes.c_int),
    ("functions", ctypes.POINTER(PyUFuncGenericFunction)),
    ("data", ctypes.POINTER(ctypes.c_void_p)),
    ("ntypes", ctypes.c_int),
    ("reserved1", ctypes.c_int),
    ("name", ctypes.c_char_p),
    ("types", ctypes.c_char_p),
    ("doc", ctypes.c_char_p),
    ("ptr", ctypes.c_void_p),
    # PyObject
    ("obj", ctypes.c_void_p),
    # PyObject
    ("userloops", ctypes.c_void_p),
    # 0 for scalar ufunc; 1 for generalized ufunc
    ("core_enabled", ctypes.c_int),
    # Number of distinct dimension names in signature
    ("core_num_dim_ix", ctypes.c_int),
    # dimension indices of input/output argument k are stored in
    # core_dim_ixs[core_offsets[k]..core_offsets[k]+core_num_dims[k]-1]``
    #
    # number of core dimensions of each argument
    ("core_num_dims", ctypes.POINTER(ctypes.c_int)),
    # dimension indices in a flatted form; indices
    # are in the range of [0,core_num_dim_ix)
    ("core_dim_ixs", ctypes.POINTER(ctypes.c_int)),
    # positions of 1st core dimensions of each
    # argument in core_dim_ixs, equivalent to cumsum(core_num_dims)
    ("core_offsets", ctypes.POINTER(ctypes.c_int)),
    # signature string for printing purposes
    ("core_signature", ctypes.c_char_p),
    # A function which resolves the types and fills an array
    # with the dtypes for inputs and outputs.
    ("type_resolver", PyUFunc_TypeResolutionFunc),
    # A dictionary for monkey patching ufuncs
    ("dict", ctypes.c_void_p),
    # Unused inner loop selector
    ("vectorcall", ctypes.c_void_p),
    # Was previously the `PyUFunc_MaskedInnerLoopSelectionFunc`
    ("reserved3", ctypes.c_void_p),
    # List of flags for each operand when ufunc
    # is called by the nditer object
    ("op_flags", ctypes.POINTER(_CTYPES_NPY_UINT32)),
    # List of global flags used when ufunc is called by nditer object.
    ("iter_flags", _CTYPES_NPY_UINT32),
    ("core_dim_sizes", ctypes.POINTER(_CTYPES_NPY_INTP)),
    ("core_dim_flags", ctypes.POINTER(_CTYPES_NPY_UINT32)),
    # More PyUfuncObject members can follow, but
    # the above are sufficient for extracting ufunc
    # function pointers
  ]
