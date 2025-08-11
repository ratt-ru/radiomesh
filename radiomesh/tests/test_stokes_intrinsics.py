import numba
import numpy as np
import pytest

from radiomesh.intrinsics import accumulate_data, load_data
from radiomesh.literals import Schema
from radiomesh.stokes_intrinsics import data_conv_fn


@pytest.mark.parametrize(
  ("pol_schema", "stokes_schema", "jones_schema"),
  [
    (("XX", "YY"), ("I", "Q"), ("XX", "YY")),
    (("XY", "YX"), ("U", "V"), ("XY", "YX")),
    (("RR", "LL"), ("I", "V"), ("RR", "LL")),
    (("RL", "LR"), ("Q", "U"), ("RL", "LR")),
    (("XY", "YX"), ("U", "V"), ("XY", "YX")),
    (("XX", "XY", "YX", "YY"), ("I", "Q", "U", "V"), ("XX", "XY", "YX", "YY")),
    (("XX", "XY", "YX", "YY"), ("I", "Q"), ("XX", "XY", "YX", "YY")),
    # No jones
    (("XX", "XY", "YX", "YY"), ("I", "Q"), None),
    pytest.param(
      ("XX", "YY"),
      ("I",),
      (),
      marks=pytest.mark.xfail(reason="empty jones tuple not yet handled"),
    ),
    pytest.param(
      ("XX",), ("I",), ("XX",), marks=pytest.mark.xfail(reason="YY missing")
    ),
    pytest.param(
      ("XX", "YY"),
      ("I",),
      ("RR", "LL"),
      marks=pytest.mark.xfail(reason="pol_schema linear, jones_schema circular"),
    ),
  ],
)
@pytest.mark.parametrize("data_type", ["vis", "weight"])
def test_data_convert(data_type, pol_schema, stokes_schema, jones_schema):
  POL_SCHEMA = Schema(pol_schema)
  STOKES_SCHEMA = Schema(stokes_schema)
  NROW = 10
  NPOL = len(pol_schema)
  NSTOKES = len(stokes_schema)

  HAVE_JONES = jones_schema is not None
  JONES_SCHEMA = Schema(jones_schema) if HAVE_JONES else None
  NJONES = len(jones_schema) if HAVE_JONES else NPOL

  @numba.njit
  def convert(data, jones):
    nrow, _ = data.shape
    result = np.zeros((nrow, NSTOKES), data.dtype)

    for r in range(nrow):
      v = load_data(data, (r,), NPOL, -1)
      g = load_data(jones, (r,), NJONES, -1) if HAVE_JONES else None
      s = data_conv_fn(v, g, g, data_type, POL_SCHEMA, STOKES_SCHEMA, JONES_SCHEMA)
      accumulate_data(s, result, (r,), -1)

    return result

  data = np.random.random((NROW, NPOL))
  if data_type == "vis":
    data = data + np.random.random((NROW, NPOL)) * 1j

  jones = None
  if HAVE_JONES:
    jones = np.random.random((NROW, NJONES)) + np.random.random((NROW, NJONES)) * 1j

  conv_data = convert(data, jones)
  assert np.all(np.abs(conv_data) != 0.0)
  assert conv_data.shape == ((NROW, NSTOKES))
  assert conv_data.dtype == np.complex128 if data_type == "vis" else np.complex64
