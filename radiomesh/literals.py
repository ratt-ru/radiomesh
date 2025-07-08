import warnings

from numba.core import types

FloatLiteral: type[types.Literal] | None = None


def install_float_literal() -> type[types.Literal]:
  try:
    # numba doesn't yet seem to have a FloatLiteral
    # but future-proof this just in case
    FloatLiteral = types.Literal.ctor_map[float]
    warnings.warn("Using numba FloatLiteral instead of radiomesh's custom FloatLiteral")
  except KeyError:
    # Create a basic FloatLiteral type that is sufficient to be recognised by @overload
    # Values of this type are probably unusable in jitted function implementations.
    from numba.core.boxing import NativeValue, unbox
    from numba.core.datamodel.models import OpaqueModel, register_default

    class RadioMeshFloatLiteral(types.Literal, types.Dummy):
      pass

    @unbox(RadioMeshFloatLiteral)
    def unbox_float_literal(typ, obj, c):
      return NativeValue(c.context.get_dummy_value())

    # Register model and literal
    register_default(RadioMeshFloatLiteral)(OpaqueModel)
    types.Literal.ctor_map[float] = RadioMeshFloatLiteral
    FloatLiteral = RadioMeshFloatLiteral

  return FloatLiteral
