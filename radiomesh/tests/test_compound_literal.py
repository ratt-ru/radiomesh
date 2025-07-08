import numba
from numba import types
from numba.core.boxing import NativeValue, unbox
from numba.core.datamodel.models import OpaqueModel, register_default
from numba.core.errors import RequireLiteralValue
from numba.extending import overload


class ImagingParameters:
  def __init__(self, nx, ny):
    self.nx = nx
    self.ny = ny


class ImagingParametersLiteral(types.Literal, types.Dummy):
  pass


@unbox(ImagingParametersLiteral)
def unbox_imaging_parameters_literal(typ, obj, c):
  return NativeValue(c.context.get_dummy_value())


# Register model and literal
register_default(ImagingParametersLiteral)(OpaqueModel)
types.Literal.ctor_map[ImagingParameters] = ImagingParametersLiteral


def test_compound_literal():
  img_params = ImagingParameters(1024.1, 1025.2)

  def f_impl(params):
    pass

  @overload(f_impl)
  def f_overload(params):
    if not isinstance(params, ImagingParametersLiteral):
      raise RequireLiteralValue(
        f"'params' {params} must be an ImagingParametersLiteral"
      )

    assert params.literal_value.nx == img_params.nx
    assert params.literal_value.nx == img_params.ny

    def impl(value):
      pass

    return impl

  @numba.njit
  def f(value):
    return numba.literally(value)

  f(img_params)
