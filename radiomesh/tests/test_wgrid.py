import numpy as np

from radiomesh.gridding import wgrid


def test_wgrid():
  """Smoke test"""
  shape = (100, 7 * 6 // 2, 64, 4)
  nx = ny = 1024
  vis = np.random.random(shape) + 0j
  vis += np.random.random(shape) * 1j
  weights = np.random.random(shape)
  uvw = np.random.random(shape[:2] + (3,))
  freqs = np.linspace(0.856e9, 2 * 0.856e9, shape[2])
  flags = np.zeros_like(weights, np.uint8)
  vis_grid, weight_grid = wgrid(
    uvw, vis, weights, flags, freqs, nx, ny, "5.0", 15, "[XX,XY,YX,YY] -> [I,Q,U,V]"
  )
  assert vis_grid.shape == (4, nx, nx)
  assert weight_grid.shape == (nx, nx)
  assert vis_grid.dtype == vis.real.dtype
  assert weight_grid.dtype == weights.dtype
