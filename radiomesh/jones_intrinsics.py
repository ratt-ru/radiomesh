from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt
from numba.core import types
from numba.core.errors import RequireLiteralValue, TypingError
from numba.extending import overload, register_jitable

from radiomesh.intrinsics import load_data
from radiomesh.literals import DatumLiteral, Schema, SchemaLiteral
from radiomesh.stokes_intrinsics import data_conv_fn


@dataclass(slots=True, unsafe_hash=True, eq=True)
class ApplyJones:
  data_type: Literal["vis", "weight"]
  pol_schema: Tuple[str, ...]
  stokes_schema: Tuple[str, ...]


def check_jones_params(jones_params):
  """Check the jones_params numba type"""
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


def maybe_apply_jones(
  apply_jones_literal: DatumLiteral[ApplyJones],
  jones_params: Tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.integer], Schema]
  | None,
  data: npt.NDArray,
  idx: Tuple[int, int, int],
):
  pass


@overload(maybe_apply_jones, prefer_literal=True)
def maybe_apply_jones_overload(apply_jones_literal, jones_params, data, idx):
  if not isinstance(apply_jones_literal, DatumLiteral) or not isinstance(
    (apply_jones := apply_jones_literal.datum_value), ApplyJones
  ):
    raise RequireLiteralValue(
      f"'apply_jones_literal' {apply_jones_literal} is not a DatumLiteral[ApplyJones]"
    )

  HAVE_JONES_PARAMS = jones_params != types.none

  if not isinstance(data, types.UniTuple):
    raise TypingError(f"'data' {data} must be a UniTuple")

  if (
    not isinstance(idx, types.UniTuple)
    or len(idx) != 4
    or not all(isinstance(i, types.Integer) for i in idx)
  ):
    raise TypingError(
      f"'idx' {idx} must be a " f"(time, baseline, channel, direction) index tuple"
    )

  DATA_TYPE = apply_jones.data_type
  POL_SCHEMA = apply_jones.pol_schema
  STOKES_SCHEMA = apply_jones.stokes_schema

  if not HAVE_JONES_PARAMS:
    # Simple case

    def impl(apply_jones_literal, jones_params, data, idx):
      return data_conv_fn(data, None, None, DATA_TYPE, POL_SCHEMA, STOKES_SCHEMA, None)
  else:
    # Load in the jones term associated
    # with each baseline's antenna pair
    # and apply them to the data
    check_jones_params(jones_params)
    JONES_SCHEMA = jones_params[2].literal_value
    NJONES = len(JONES_SCHEMA)

    def impl(apply_jones_literal, jones_params, data, idx):
      t, bl, ch, d = idx
      jones, antenna_pairs, _ = jones_params
      a1 = antenna_pairs[bl, 0]
      a2 = antenna_pairs[bl, 1]
      j1 = load_data(jones, (t, a1, ch, d), NJONES, -1)
      j2 = load_data(jones, (t, a2, ch, d), NJONES, -1)
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


@register_jitable
def ndirections(jones_params):
  """Infer the number of directions from the jones_params.

  If jones_params is None, then 1 will be returned,
  otherwise the number of directions is derived from the
  jones array whose shape has the form
  :code:`(time, antenna, channel, direction, polarisation)`"""
  if jones_params is None:
    return 1
  else:
    jones = jones_params[0]
    assert len(jones.shape) == 5
    return jones.shape[3]
