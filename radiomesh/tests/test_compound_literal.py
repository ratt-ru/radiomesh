from dataclasses import dataclass

import numba
from numba import types
from numba.core.boxing import NativeValue, unbox
from numba.core.datamodel.models import OpaqueModel, register_default
from numba.core.errors import RequireLiteralValue
from numba.extending import overload


@dataclass
class ImagingParameters:
  nx: float
  ny: float


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
  executed = False

  def f_impl(params):
    pass

  @overload(f_impl)
  def f_overload(params):
    if not isinstance(params, ImagingParametersLiteral):
      raise RequireLiteralValue(
        f"'params' {params} must be an ImagingParametersLiteral"
      )

    assert params.literal_value.nx == img_params.nx
    assert params.literal_value.ny == img_params.ny
    nonlocal executed
    executed = True

    def impl(params):
      pass

    return impl

  @numba.njit
  def f(value):
    return f_impl(numba.literally(value))

  f(img_params)
  assert executed
