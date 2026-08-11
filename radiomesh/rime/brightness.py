from typing import Callable, Dict, Tuple

import numba
import numpy as np
from numba.core import cgutils, errors, types
from numba.experimental import structref
from numba.extending import intrinsic, overload, overload_method
from numba.np.numpy_support import as_dtype, from_dtype
from rarg_numba_patterns.intrinsics import load_data
from rarg_numba_patterns.literals import Datum, is_datum_literal

from radiomesh.literals import DatumLiteral
from radiomesh.rime.core import (
  AbstractRimeTermStructRef,
  AbstractRimeTermStructRefProxy,
)

JIT_OPTIONS = {"nogil": True, "error_model": "numpy"}


STOKES_CONVERTERS: Dict[str, Dict[Tuple[str, str], Callable]] = {
  "RR": {("I", "V"): lambda i, v: i + v + 0j},
  "RL": {("Q", "U"): lambda q, u: q + u * 1j},
  "LR": {("Q", "U"): lambda q, u: q - u * 1j},
  "LL": {("I", "V"): lambda i, v: i - v + 0j},
  "XX": {("I", "Q"): lambda i, q: i + q + 0j},
  "XY": {("U", "V"): lambda u, v: u + v * 1j},
  "YX": {("U", "V"): lambda u, v: u - v * 1j},
  "YY": {("I", "Q"): lambda i, q: i - q + 0j},
}


@intrinsic(prefer_literal=True)
def stokes_to_pol(
  typingctx,
  stokes_schema: DatumLiteral[Tuple[str]],
  stokes: Tuple[types.Float],
  polarisation_schema: DatumLiteral[Tuple[str]],
):
  """Converts a tuple of stokes parameters into polarisations."""
  if not is_datum_literal(stokes_schema, tuple):
    raise errors.RequireLiteralValue(
      f"'stokes_schema' {stokes_schema} must be a DatumLiteral[tuple]"
    )

  if (NSTOKES := len(STOKES := stokes_schema.literal_value)) == 0:
    raise ValueError(
      "No stokes parameters were specified for conversion to polarisations"
    )

  if not is_datum_literal(polarisation_schema, tuple):
    raise errors.RequireLiteralValue(
      f"'polarisation_schema' {polarisation_schema} must be a DatumLiteral[tuple]"
    )

  if (NPOL := len(POLS := polarisation_schema.literal_value)) == 0:
    raise ValueError(
      "No polarisations were specified for conversion from stokes parameters"
    )

  if not isinstance(stokes, types.BaseTuple) or len(stokes) != NSTOKES:
    raise errors.TypingError(f"'stokes' {stokes} should be a tuple of length {NSTOKES}")

  stokes_map = {s: i for i, s in enumerate(STOKES)}
  conv_map = []

  for p in POLS:
    if (conv_schema := STOKES_CONVERTERS.get(p)) is None:
      raise ValueError(
        f"No converter registered for polarisation {p}. "
        f"{list(STOKES_CONVERTERS.keys())} available."
      )

    i1 = i2 = None

    for (s1, s2), fn in conv_schema.items():
      if not ((i1 := stokes_map.get(s1)) is None or (i2 := stokes_map.get(s2)) is None):
        break

    if i1 is None or i2 is None:
      raise ValueError(
        f"No conversion found for polarisation {p}. "
        f"{stokes_schema.literal_value} is available, but "
        f"some combination of {set(conv_schema.keys())} "
        f"is required for conversion to {p}"
      )

    conv_map.append((fn, i1, i2))

  CONV_FNS, S1_INDEX, S2_INDEX = zip(*conv_map)
  complex_type = typingctx.unify_types(stokes.dtype, types.complex64)
  return_type = types.Tuple([complex_type] * NPOL)
  sig = return_type(stokes_schema, stokes, polarisation_schema)

  def codegen(context, builder, signature, args):
    _, stokes_type, _ = signature.args
    _, stokes, _ = args
    return_type = signature.return_type
    llvm_ret_type = context.get_value_type(return_type)
    pol_tuple = cgutils.get_null_value(llvm_ret_type)

    for p, (conv_fn, i1, i2) in enumerate(zip(CONV_FNS, S1_INDEX, S2_INDEX)):
      s1 = builder.extract_value(stokes, i1)
      s2 = builder.extract_value(stokes, i2)
      sig = return_type.dtype(stokes_type.dtype, stokes_type.dtype)
      pol = context.compile_internal(builder, conv_fn, sig, [s1, s2])
      pol_tuple = builder.insert_value(pol_tuple, pol, p)

    return pol_tuple

  return sig, codegen


@structref.register
class BrightnessTermStructRef(AbstractRimeTermStructRef):
  pass


class BrightnessTerm(AbstractRimeTermStructRefProxy):
  def __new__(cls, locality, stokes_schema, stokes, polarisation_schema):
    return AbstractRimeTermStructRefProxy.__new__(
      cls, Datum(locality), Datum(stokes_schema), stokes, Datum(polarisation_schema)
    )

  @numba.njit
  def sample(self, core_args, s, t, bl, ch):
    return self.sample(core_args, s, t, bl, ch)


@overload(BrightnessTerm, prefer_literal=True, jit_options=JIT_OPTIONS)
def overload_brightness_term(locality, stokes_schema, stokes, polarisation_schema):
  if not is_datum_literal(locality, str) or locality.literal_value != "pq":
    raise errors.RequireLiteralValue("'locality' must be DatumLiteral[str]('pq')")

  if not isinstance(stokes, types.Array):
    return None

  STOKES_SCHEMA = Datum(stokes_schema.literal_value)
  POL_SCHEMA = Datum(polarisation_schema.literal_value)
  NSTOKES = len(stokes_schema.literal_value)
  NPOL = len(polarisation_schema.literal_value)

  pol_type = from_dtype(np.result_type(as_dtype(stokes.dtype), np.complex64))

  struct_type = BrightnessTermStructRef(
    [
      ("locality", locality),
      ("stokes", stokes),
      ("stokes_schema", stokes_schema),
      ("polarisation_schema", polarisation_schema),
    ]
  )

  def impl(locality, stokes_schema, stokes, polarisation_schema):
    obj = structref.new(struct_type)
    obj.stokes = stokes
    return obj

  return impl


structref.define_boxing(BrightnessTermStructRef, BrightnessTerm)


@overload_method(
  BrightnessTermStructRef, "sample", inline="always", jit_options=JIT_OPTIONS
)
def overload_phase_term_sample(self, core_args, s, t, bl, ch):
  STOKES_SCHEMA = Datum(self.get_literal("stokes_schema"))
  POL_SCHEMA = Datum(self.get_literal("polarisation_schema"))
  NSTOKES = len(STOKES_SCHEMA.value)

  def impl(self, core_args, s, t, bl, ch):
    stokes_tuple = load_data(self.stokes, (s,), NSTOKES, -1)
    return stokes_to_pol(STOKES_SCHEMA, stokes_tuple, POL_SCHEMA)

  return impl
