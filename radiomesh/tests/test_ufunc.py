import numba
import numpy as np
import pytest

import radiomesh.fft_wrappers  # noqa: F401


@pytest.mark.parametrize("n", [2, 5, 10, 21, 25, 1025])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize(
  "np_function, dtype, rtol, atol",
  [
    (np.fft.fft, np.complex128, 1e-13, 0),
    (np.fft.ifft, np.complex128, 1e-13, 0),
    (np.fft.rfft, np.float64, 1e-13, 0),
    (np.fft.irfft, np.complex128, 1e-13, 0),
    # 1e-6 sometimes fails on single points
    (np.fft.fft, np.complex64, 1e-5, 1e-5),
    (np.fft.ifft, np.complex64, 1e-5, 1e-5),
    (np.fft.rfft, np.float32, 1e-5, 1e-5),
    (np.fft.irfft, np.complex64, 1e-5, 1e-5),
  ],
)
def test_numba_fft_call(n, norm, np_function, dtype, rtol, atol):
  @numba.njit(nogil=True, cache=True)
  def fn(a, norm):
    return np_function(a, norm=numba.literally(norm))

  rng = np.random.default_rng(42)

  if np.issubdtype(dtype, np.complexfloating):
    a = (rng.random(n) + rng.random(n) * 1j).astype(dtype)
  else:
    a = rng.random(n).astype(dtype)

  np_values = np_function(a, norm=norm)
  nb_values = fn(a, norm)

  np.testing.assert_allclose(np_values, nb_values, rtol=rtol, atol=atol)
