import multiprocessing as mp

import numba
import numpy as np
import pytest

from radiomesh.es_kernel import generate_poly_coeffs_numpy
from radiomesh.es_kernel_structref import ESKernelProxy, generate_poly_coeffs
from radiomesh.tests.proc_utils import _init_numba_cache_debugging_with_capture


@pytest.mark.parametrize(
  "support, beta, e0",
  [
    (5, 2.3, 0.5),
    (7, 2.3, 0.5),
    (4, 1.5, 0.75),
    (8, 3.0, 0.5),
    (6, 2.0, 1.0),
  ],
)
def test_generate_poly_coeffs_vs_numpy(support, beta, e0):
  ref = np.array(generate_poly_coeffs_numpy(support, beta, e0))
  result = generate_poly_coeffs(support, beta, e0)
  np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-12)


def _caching_worker(x):
  kernel = ESKernelProxy(2e-13, 2.0, 2.3, 0.5, -1, False, False, True)

  @numba.njit(cache=True, nogil=True)
  def fn(kernel, x):
    return kernel.evaluate(x)

  return fn(kernel, x)


def test_caching(tmp_path):
  stdout_f = tmp_path / "stdout.txt"
  stderr_f = tmp_path / "stderr.txt"

  with mp.get_context("spawn").Pool(
    1,
    initializer=_init_numba_cache_debugging_with_capture,
    initargs=(str(tmp_path), str(stdout_f), str(stderr_f)),
  ) as p:
    assert p.apply(_caching_worker, args=(0.5,)) == 2.3 * 0.5
    assert p.apply(_caching_worker, args=(0.5,)) == 2.3 * 0.5

  combined = stdout_f.read_text() + stderr_f.read_text()
  assert f"data saved to '{tmp_path}" in combined
  assert f"data loaded from '{tmp_path}" in combined
  assert f"index loaded from '{tmp_path}" in combined
