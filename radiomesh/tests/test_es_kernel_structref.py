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
  kernel = ESKernelProxy(2e-13, 2.0, 2.3, 0.5, -1, True, False, True)

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
    p.apply(_caching_worker, args=(0.5,))
    p.apply(_caching_worker, args=(0.5,))

  combined = stdout_f.read_text() + stderr_f.read_text()
  assert f"data saved to '{tmp_path}" in combined
  assert f"data loaded from '{tmp_path}" in combined
  assert f"index loaded from '{tmp_path}" in combined


@pytest.mark.parametrize(
  "support, beta, e0, rtol, atol",
  [
    (4, 1.5, 0.75, 1e-2, 1e-12),
    (5, 2.3, 0.5, 2e-2, 1e-12),
    (6, 2.0, 1.0, 2e-4, 1e-12),
    (7, 2.3, 0.5, 3e-3, 1e-12),
    (8, 3.0, 0.5, 1e-4, 1e-12),
  ],
)
def test_evaluate_analytic_vs_polynomial(support, beta, e0, rtol, atol):
  """Analytic and polynomial evaluate should agree at sample positions."""
  analytic_kernel = ESKernelProxy(
    epsilon=2e-13,
    oversampling=2.0,
    beta=beta,
    e0=e0,
    support=support,
    analytic=True,
    single=False,
    apply_w=True,
  )
  poly_kernel = ESKernelProxy(
    epsilon=2e-13,
    oversampling=2.0,
    beta=beta,
    e0=e0,
    support=support,
    analytic=False,
    single=False,
    apply_w=True,
  )

  @numba.njit
  def eval_both(ak, pk, x):
    return ak.evaluate(x), pk.evaluate(x)

  half_support = support / 2.0
  positions = np.linspace(-half_support * 0.99, half_support * 0.99, 20)

  for pos in positions:
    a_val, p_val = eval_both(analytic_kernel, poly_kernel, pos)
    np.testing.assert_allclose(p_val, a_val, rtol=rtol, atol=atol)


def test_evaluate_boundary():
  """Positions at or beyond ±half_support should return 0.0."""
  kernel = ESKernelProxy(
    epsilon=2e-13,
    oversampling=2.0,
    beta=2.3,
    e0=0.5,
    support=7,
    analytic=True,
    single=False,
    apply_w=True,
  )

  @numba.njit
  def eval_kernel(k, x):
    return k.evaluate(x)

  half_support = 7 / 2.0
  assert eval_kernel(kernel, half_support) == 0.0
  assert eval_kernel(kernel, -half_support) == 0.0
  assert eval_kernel(kernel, half_support + 1.0) == 0.0
  assert eval_kernel(kernel, -half_support - 1.0) == 0.0
