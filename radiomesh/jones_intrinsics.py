from dataclasses import dataclass
from typing import Literal, Tuple

from numba.core import types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.extending import overload

from radiomesh.intrinsics import load_data
from radiomesh.literals import DatumLiteral, SchemaLiteral
from radiomesh.stokes_intrinsics import data_conv_fn


@dataclass(slots=True, unsafe_hash=True, eq=True)
class ApplyJones:
  data_type: Literal["vis", "weight"]
  pol_schema: Tuple[str, ...]
  stokes_schema: Tuple[str, ...]


def maybe_apply_jones(
  apply_jones_literal,
  jones_params,
  data,
  idx,
):
  pass


@overload(maybe_apply_jones, prefer_literal=True)
def maybe_apply_jones_overload(apply_jones_literal, jones_params, data, idx):
  if not isinstance(apply_jones_literal, DatumLiteral):
    raise RequireLiteralValue(
      f"'apply_jones_literal' {apply_jones_literal} is not a DatumLiteral"
    )

  HAVE_JONES_PARAMS = jones_params != types.none

  if not isinstance(data, types.UniTuple):
    raise TypingError(f"'data' {data} must be a UniTuple")

  if (
    not isinstance(idx, types.UniTuple)
    or len(idx) != 3
    or not all(isinstance(i, types.Integer) for i in idx)
  ):
    raise TypingError(f"'idx' {idx} must be a (time, baseline, channel) index tuple")

  apply_jones_params = apply_jones_literal.datum_value
  DATA_TYPE = apply_jones_params.data_type
  POL_SCHEMA = apply_jones_params.pol_schema
  STOKES_SCHEMA = apply_jones_params.stokes_schema

  if not HAVE_JONES_PARAMS:
    # Simple case

    def impl(apply_jones_literal, jones_params, data, idx):
      return data_conv_fn(data, None, None, DATA_TYPE, POL_SCHEMA, STOKES_SCHEMA, None)
  else:
    # Load in the jones term associated
    # with each baseline's antenna pair
    # and apply them to the data
    if (
      not isinstance(jones_params, types.Tuple)
      or len(jones_params) != 3
      or not all(
        isinstance(jp, types.Array)
        for jp in jones_params[:2] or not isinstance(jones_params[2], SchemaLiteral)
      )
    ):
      raise TypingError(
        f"'jones_params' {jones_params} must be None "
        f"or a (jones, antenna_pairs, schema) tuple of "
        f"types (Array, Array, Schema)"
      )

    JONES_SCHEMA = jones_params[2].literal_value
    NJONES = len(JONES_SCHEMA)

    def impl(apply_jones_literal, jones_params, data, idx):
      t, bl, ch = idx
      jones, antenna_pairs, _ = jones_params
      a1 = antenna_pairs[bl, 0]
      a2 = antenna_pairs[bl, 1]
      j1 = load_data(jones, (t, a1, ch, 0), NJONES, -1)
      j2 = load_data(jones, (t, a2, ch, 0), NJONES, -1)
      return data_conv_fn(
        data,
        j1,
        j2,
        DATA_TYPE,
        POL_SCHEMA,
        STOKES_SCHEMA,
        JONES_SCHEMA,
      )

  return impl
