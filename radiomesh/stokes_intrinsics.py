from enum import Enum
from typing import Any, Dict, Set, Tuple

from llvmlite import ir
from numba.core import cgutils, types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.extending import intrinsic

from radiomesh._stokes_expr import CONVERT_FNS
from radiomesh.literals import DatumLiteral

LINEAR_POLS = ("XX", "XY", "YX", "YY")
CIRCULAR_POLS = ("RR", "RL", "LR", "LL")
ALL_POLS = LINEAR_POLS + CIRCULAR_POLS
DIAGONALS = ("XX", "YY", "RR", "LL")

# Set of polarisations pairs that can
# construct a stokes parameter
STOKES_DEPENDENCIES: Dict[str, Set[Tuple[str, str]]] = {
  "I": {("XX", "YY"), ("RR", "LL")},
  "Q": {("XX", "YY"), ("RL", "LR")},
  "U": {("XY", "YX"), ("RL", "LR")},
  "V": {("XY", "YX"), ("RR", "LL")},
}

# Set of stokes parameters than can
# construct a polarisation
POL_DEPENDENCIES: Dict[str, Set[Tuple[str, str]]] = {
  "RR": {("I", "V")},
  "RL": {("Q", "U")},
  "LR": {("Q", "U")},
  "LL": {("I", "V")},
  "XX": {("I", "Q")},
  "XY": {("U", "V")},
  "YX": {("U", "V")},
  "YY": {("I", "Q")},
}


DataSource = Enum("DataSource", ["Index", "Default"])


def schema_pol_type(schema_type: str, pol_schema: Tuple[str, ...]) -> str:
  """Determine the linearity or circularity of the supplied polarisation schema.
  Mixed polarisation schemas are not allowed"""
  if set(pol_schema).issubset(set(LINEAR_POLS)):
    return "LINEAR"
  elif set(pol_schema).issubset(set(CIRCULAR_POLS)):
    return "CIRCULAR"
  else:
    raise ValueError(
      f"{schema_type} schema {pol_schema} must consist of either "
      f"linear {LINEAR_POLS} or circular {CIRCULAR_POLS} polarisations. "
      f"Mixed polarisations are not supported."
    )


def data_source_maps(
  data_type: str, data_schema: Tuple[str, ...], gain_schema: Tuple[str, ...]
) -> Tuple[Dict[str, Tuple[DataSource, Any]], Dict[str, Tuple[DataSource, Any]]]:
  """Identify data sources for the data and gains for all polarisations.

  Indices from the data and gains schemas are preferred,
  otherwise default values are provided.

  llvmlite creates complex constants from tuples of reals.
  """
  data_source: Dict[str, Tuple[DataSource, Any]] = {}
  gain_source: Dict[str, Tuple[DataSource, Any]] = {}

  for pol in ALL_POLS:
    try:
      data_source[pol] = (DataSource.Index, data_schema.index(pol))
    except ValueError:
      data_source[pol] = (DataSource.Default, (0, 0) if data_type == "vis" else 0)

    try:
      gain_source[pol] = (DataSource.Index, gain_schema.index(pol))
    except ValueError:
      # Fill in identity matrix
      gain_source[pol] = (DataSource.Default, (1, 0) if pol in DIAGONALS else (0, 0))

  return data_source, gain_source


def check_stokes_datasources(stokes_schema, data_schema, data_source_map):
  """Check that required polarisations exist in
  the data for the requested stokes parameters"""
  for stokes in stokes_schema:
    try:
      deps = STOKES_DEPENDENCIES[stokes]
    except KeyError:
      raise ValueError(
        f"{stokes} is not a valid "
        f"stokes parameter: {list(STOKES_DEPENDENCIES.keys())}"
      )

    index_maps = 0

    for p1, p2 in deps:
      s1 = data_source_map[p1]
      s2 = data_source_map[p2]

      if s1[0] is DataSource.Index and s2[0] is DataSource.Index:
        index_maps += 1

    if index_maps == 0:
      raise ValueError(
        f"Unable to derive {stokes} from the data {data_schema}. "
        f"One of the following combinations must be present "
        f"in the data {list(deps)}."
      )


@intrinsic(prefer_literal=True)
def data_conv_fn(
  typingctx,
  data: types.UniTuple,
  gains_p: types.UniTuple | types.NoneType,
  gains_q: types.UniTuple | types.NoneType,
  data_type_literal: types.StringLiteral,
  data_schema_literal: DatumLiteral[Tuple[str, ...]],
  gain_schema_literal: DatumLiteral[Tuple[str, ...]] | types.NoneType,
  stokes_schema_literal: DatumLiteral[Tuple[str, ...]],
):
  # Verify arguments
  have_gains = (
    gains_p != types.none
    and gains_q != types.none
    and gain_schema_literal != types.none
  )

  if not isinstance(data_type_literal, types.StringLiteral) or (
    data_type := data_type_literal.literal_value
  ) not in {"vis", "weight"}:
    raise RequireLiteralValue(
      f"'data_type_literal' {data_type_literal} must be a StringLiteral "
      f"equal to 'vis' or 'weight'"
    )

  if not isinstance(data_schema_literal, DatumLiteral):
    raise RequireLiteralValue(
      f"'data_schema_literal' {data_schema_literal} must be a DatumLiteral"
    )

  if have_gains and not isinstance(gain_schema_literal, DatumLiteral):
    raise RequireLiteralValue(
      f"'gain_schema_literal' {gain_schema_literal} must be a DatumLiteral"
    )

  if not isinstance(stokes_schema_literal, DatumLiteral):
    raise RequireLiteralValue(
      f"'stokes_schema_literal' {stokes_schema_literal} must be a DatumLiteral"
    )

  data_schema = tuple(v.upper() for v in data_schema_literal.datum_value)
  stokes_schema = tuple(s.upper() for s in stokes_schema_literal.datum_value)
  gain_schema = (
    tuple(g.upper() for g in gain_schema_literal.datum_value) if have_gains else ()
  )

  if (
    not isinstance(data, types.UniTuple)
    or not isinstance(data.dtype, (types.Float, types.Complex))
    or len(data) != len(data_schema)
  ):
    raise TypingError(
      f"'data' {data} must be a tuple of float/complex values "
      f"of len({data_schema}) == {len(data_schema)}"
    )

  if have_gains:
    if (
      not isinstance(gains_p, types.UniTuple)
      or not isinstance(gains_p.dtype, types.Complex)
      or len(gains_p) != len(gain_schema)
    ):
      raise TypingError(
        f"'gains_p' {gains_p} must be a tuple of complex values "
        f"of len({gain_schema}) == {len(gain_schema)}"
      )

    if (
      not isinstance(gains_q, types.UniTuple)
      or not isinstance(gains_q.dtype, types.Complex)
      or len(gains_q) != len(gain_schema)
    ):
      raise TypingError(
        f"'gains_q' {gains_q} must be a tuple of complex values "
        f"of len({gain_schema}) == {len(gain_schema)}"
      )

  data_pol_type = schema_pol_type("Visibility", data_schema)

  if have_gains and data_pol_type != (
    gain_pol_type := schema_pol_type("Gain", gain_schema)
  ):
    raise ValueError(
      f"Visibility polarisation type {data_pol_type.lower()} ({data_schema}) "
      f"does not match the "
      f"gain polarisation type {gain_pol_type.lower()} ({gain_schema})"
    )

  data_source_map, gain_source_map = data_source_maps(
    data_type, data_schema, gain_schema
  )
  check_stokes_datasources(stokes_schema, data_schema, data_source_map)

  # key for looking up per-stokes conversion functions
  base_key = (
    data_type.upper(),
    data_pol_type.upper(),
    "GAINS" if have_gains else "NOGAINS",
  )

  stokes_type = data.dtype
  return_type = types.Tuple([stokes_type] * len(stokes_schema))
  sig = return_type(
    data,
    gains_p,
    gains_q,
    data_type_literal,
    data_schema_literal,
    gain_schema_literal,
    stokes_schema_literal,
  )

  def codegen(context, builder, signature, args):
    data, gains_p, gains_q, _, _, _, _ = args
    data_type, gains_p_type, gains_q_type, _, _, _, _ = signature.args
    llvm_data_type = context.get_value_type(data_type.dtype)

    # Build conversion function arguments and argument types
    # When gains are present 12 arguments are needed
    # (4*vis/weight, 4*gain_p, 4*gain_q)
    # otherwise they only take the 4 visibilities/weights
    fn_args = []
    fn_arg_types = []

    # Conversion functions take all four polarisations for
    # data and both sets of gains (if gains are present)
    # Prefer extraction from data/gains, else use defaults
    data_pols = LINEAR_POLS if data_pol_type == "LINEAR" else CIRCULAR_POLS

    # Extract visibility arguments and types
    for pol in data_pols:
      src_type, src_value = data_source_map[pol]
      if src_type is DataSource.Index:
        arg_value = builder.extract_value(data, src_value)
      elif src_type is DataSource.Default:
        arg_value = ir.Constant(llvm_data_type, src_value)
      else:
        raise ValueError(f"Unhandled source type {src_type}")

      fn_args.append(arg_value)
      fn_arg_types.append(data_type.dtype)

    if have_gains:
      gain_pols = LINEAR_POLS if gain_pol_type == "LINEAR" else CIRCULAR_POLS
      llvm_gains_p_type = context.get_value_type(gains_p_type.dtype)
      llvm_gains_q_type = context.get_value_type(gains_q_type.dtype)

      # Extract gain p arguments and types
      for pol in gain_pols:
        src_type, src_value = gain_source_map[pol]
        if src_type is DataSource.Index:
          arg_value = builder.extract_value(gains_p, src_value)
        elif src_type is DataSource.Default:
          arg_value = ir.Constant(llvm_gains_p_type, src_value)
        else:
          raise ValueError(f"Unhandled DataSource type {src_type}")

        fn_args.append(arg_value)
        fn_arg_types.append(gains_p_type.dtype)

      # Extract gain q arguments and types
      for pol in gain_pols:
        src_type, src_value = gain_source_map[pol]
        if src_type is DataSource.Index:
          arg_value = builder.extract_value(gains_q, src_value)
        elif src_type is DataSource.Default:
          arg_value = ir.Constant(llvm_gains_q_type, src_value)
        else:
          raise ValueError(f"Unhandled DataSource type {src_type}")

        fn_args.append(arg_value)
        fn_arg_types.append(gains_q_type.dtype)

    # Signature is the same for all stokes parameters
    conv_fn_sig = stokes_type(*fn_arg_types)

    # Generate the tuple of stokes parameters
    llvm_ret_type = context.get_value_type(signature.return_type)
    stokes_tuple = cgutils.get_null_value(llvm_ret_type)

    for s, stokes in enumerate(stokes_schema):
      conv_fn = CONVERT_FNS[base_key + (stokes,)]
      value = context.compile_internal(builder, conv_fn, conv_fn_sig, fn_args)
      stokes_tuple = builder.insert_value(stokes_tuple, value, s)

    return stokes_tuple

  return sig, codegen
