import numba
import numpy as np
import pytest

from radiomesh.intrinsics import accumulate_data, load_data
from radiomesh.literals import Datum
from radiomesh.stokes_intrinsics import data_conv_fn


@pytest.mark.parametrize(
  ("pol_schema", "stokes_schema", "gain_schema"),
  [
    (("XX", "YY"), ("I", "Q"), ("XX", "YY")),
    (("XY", "YX"), ("U", "V"), ("XY", "YX")),
    (("RR", "LL"), ("I", "V"), ("RR", "LL")),
    (("RL", "LR"), ("Q", "U"), ("RL", "LR")),
    (("XY", "YX"), ("U", "V"), ("XY", "YX")),
    (("XX", "XY", "YX", "YY"), ("I", "Q", "U", "V"), ("XX", "XY", "YX", "YY")),
    (("XX", "XY", "YX", "YY"), ("I", "Q"), ("XX", "XY", "YX", "YY")),
    pytest.param(
      ("XX", "YY"),
      ("I",),
      (),
      marks=pytest.mark.xfail(reason="empty gains tuple not yet handled"),
    ),
    pytest.param(
      ("XX",), ("I",), ("XX",), marks=pytest.mark.xfail(reason="YY missing")
    ),
    pytest.param(
      ("XX", "YY"),
      ("I",),
      ("RR", "LL"),
      marks=pytest.mark.xfail(reason="pol schema != gain schema"),
    ),
  ],
)
@pytest.mark.parametrize("data_type", ["vis", "weight"])
def test_data_convert(data_type, pol_schema, stokes_schema, gain_schema):
  POL_SCHEMA = Datum(pol_schema)
  STOKES_SCHEMA = Datum(stokes_schema)
  GAIN_SCHEMA = Datum(gain_schema)
  NROW = 10
  NPOL = len(pol_schema)
  NGAINS = len(gain_schema)
  NSTOKES = len(stokes_schema)

  @numba.njit
  def convert(data, gains):
    nrow, _ = data.shape
    result = np.zeros((nrow, NSTOKES), data.dtype)

    for r in range(nrow):
      v = load_data(data, (r,), NPOL, -1)
      g = load_data(gains, (r,), NGAINS, -1)
      s = data_conv_fn(v, g, g, data_type, POL_SCHEMA, GAIN_SCHEMA, STOKES_SCHEMA)
      accumulate_data(s, result, (r,), NSTOKES, -1)

    return result

  data = np.random.random((NROW, NPOL))
  if data_type == "vis":
    data = data + np.random.random((NROW, NPOL)) * 1j
  gains = np.random.random((NROW, NGAINS)) + np.random.random((NROW, NGAINS)) * 1j

  conv_data = convert(data, gains)
  assert np.all(np.abs(conv_data) != 0.0)
  assert conv_data.shape == ((NROW, NSTOKES))
  assert conv_data.dtype == np.complex128 if data_type == "vis" else np.complex64
