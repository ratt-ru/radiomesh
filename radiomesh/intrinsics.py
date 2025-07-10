from typing import Callable, Dict, Tuple

from numba.core import cgutils, types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, register_jitable

from radiomesh.literals import DatumLiteral

# A `P: {(S1, S2): FN}` mapping from stokes parameters to polarisations
# where S1 and S2 are the stokes parameters input to function FN
# to produce polarisation P
STOKES_CONVERSION: Dict[str, Dict[Tuple[str, str], Callable]] = {
  "RR": {("I", "V"): lambda i, v: i + v + 0j},
  "RL": {("Q", "U"): lambda q, u: q + u * 1j},
  "LR": {("Q", "U"): lambda q, u: q - u * 1j},
  "LL": {("I", "V"): lambda i, v: i - v + 0j},
  "XX": {("I", "Q"): lambda i, q: i + q + 0j},
  "XY": {("U", "V"): lambda u, v: u + v * 1j},
  "YX": {("U", "V"): lambda u, v: u - v * 1j},
  "YY": {("I", "Q"): lambda i, q: i - q + 0j},
}

# A `S: {(P1, P2): FN}` mapping from polarisatoins to stokes parameters
# where P1 and P2 are the polarisations input to function FN
# to produce stokes S.
POL_CONVERSION: Dict[str, Dict[Tuple[str, str], Callable]] = {
  "I": {
    ("XX", "YY"): lambda xx, yy: (xx.real + yy.real) / 2.0,
    ("RR", "LL"): lambda rr, ll: (rr.real + ll.real) / 2.0,
  },
  "Q": {
    ("XX", "YY"): lambda xx, yy: (xx.real - yy.real) / 2.0,
    ("RL", "LR"): lambda rl, lr: (rl.real + lr.real) / 2.0,
  },
  "U": {
    ("XY", "YX"): lambda xy, yx: (xy.real + yx.real) / 2.0,
    ("RL", "LR"): lambda rl, lr: (rl.imag - lr.imag) / 2.0,
  },
  "V": {
    ("XY", "YX"): lambda xy, yx: (xy.imag - yx.imag) / 2.0,
    ("RR", "LL"): lambda rr, ll: (rr.real - ll.real) / 2.0,
  },
}


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
  ndata: types.IntegerLiteral,
  axis: types.IntegerLiteral,
) -> Tuple[Signature, Callable]:
  """An intrinsic that accumulates an `ndata` tuple of values
  in an array at a given `axis` and `index`:

  .. code-block:: python

    assert array.shape[axis] == ndata
    assert len(data_tuple) == ndata
    array[index[:axis] + (Ellipsis,) + index[axis:]] += data_tuple

  `index` should be a tuple of array.ndim - 1 integer values.
  `axis` references the dimension not referenced by `index` and
  should be of length ndata.
  """
  if not isinstance(ndata, types.IntegerLiteral):
    raise RequireLiteralValue(f"'ndata' ({ndata}) must be an IntegerLiteral")

  if not isinstance(axis, types.IntegerLiteral):
    raise RequireLiteralValue(f"'axis' ({axis}) must be an IntegerLiteral")

  if not isinstance(data, types.UniTuple) or len(data) != ndata.literal_value:
    raise TypingError(
      f"'data' ({data}) should be a tuple of length {ndata.literal_value}"
    )

  if not isinstance(array, types.Array) or array.ndim != len(index) + 1:
    raise TypingError(f"'array' ({array}) should be a {len(index) + 1}D array")

  if not isinstance(index, types.BaseTuple) or not all(
    isinstance(i, types.Integer) for i in index
  ):
    raise TypingError(f"'index' {index} must be a tuple of integers")

  sig = types.none(data, array, index, ndata, axis)
  # -1 signifies the axis should be at the end of the tuple
  ax = ndata.literal_value if axis.literal_value < 0 else axis.literal_value

  def assign_factory(pol):
    """Index array with the N-1 indices combined with pol"""

    def assign(value, array, index):
      array[index[:ax] + (pol,) + index[ax:]] += value[pol]

    return assign

  def codegen(context, builder, signature, args):
    data, array, index, _, _ = args
    data_type, array_type, index_type, _, _ = signature.args
    sig = types.none(data_type, array_type, index_type)

    for p in range(ndata.literal_value):
      context.compile_internal(builder, assign_factory(p), sig, [data, array, index])

    return None

  return sig, codegen


@intrinsic(prefer_literal=True)
def pol_to_stokes(
  typingctx, data: types.UniTuple, pol_schema: DatumLiteral, stokes_schema: DatumLiteral
) -> Tuple[Signature, Callable]:
  """Converts `data` tuple described by `pol_schema` into a tuple
  described by `stokes_schema`"""
  if not isinstance(pol_schema, DatumLiteral):
    raise RequireLiteralValue(f"'pol_schema' ({pol_schema}) must be a DatumLiteral")

  if not isinstance(stokes_schema, DatumLiteral):
    raise RequireLiteralValue(f"'pol_schema' ({pol_schema}) must be a DatumLiteral")

  if not isinstance(data, types.UniTuple) or len(data) != len(pol_schema.datum_value):
    raise TypingError(
      f"data ({data}) should be a tuple of length {len(pol_schema.datum_value)}"
    )

  pol_schema_map = {c: i for i, c in enumerate(pol_schema.datum_value)}
  conv_map = {}

  for stokes in stokes_schema.datum_value:
    try:
      conv_schema = POL_CONVERSION[stokes]
    except KeyError:
      raise KeyError(
        f"No method for producing stokes {stokes} is registered. "
        f"The following targets are registered: {list(POL_CONVERSION.keys())}"
      )

    for (c1, c2), fn in conv_schema.items():
      try:
        i1 = pol_schema_map[c1]
        i2 = pol_schema_map[c2]
      except KeyError:
        continue
      else:
        conv_map[stokes] = (i1, i2, fn)

    if stokes not in conv_map:
      raise ValueError(
        f"No conversion to stokes {stokes} was possible. "
        f"The following correlations are available {pol_schema.datum_value} "
        f"but one of the following combinations {list(conv_schema.keys())} "
        f"is required to produces {stokes}."
      )

  float_type = data.dtype.underlying_float
  ret_type = types.Tuple([float_type] * len(stokes_schema.datum_value))
  sig = ret_type(data, pol_schema, stokes_schema)

  def codegen(context, builder, signature, args):
    data, _, _ = args
    data_type, _, _ = signature.args
    llvm_type = context.get_value_type(signature.return_type)
    stokes_tuple = cgutils.get_null_value(llvm_type)

    for s, (i1, i2, conv_fn) in enumerate(conv_map.values()):
      # Extract polarisations from the data tuple
      p1 = builder.extract_value(data, i1)
      p2 = builder.extract_value(data, i2)

      # Compute stokes from polarisations and insert into result tuple
      sig = signature.return_type[s](data_type.dtype, data_type.dtype)
      stokes = context.compile_internal(builder, conv_fn, sig, [p1, p2])
      stokes_tuple = builder.insert_value(stokes_tuple, stokes, s)

    return stokes_tuple

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
