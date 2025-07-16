import numpy as np
import pytest

from radiomesh.constants import LIGHTSPEED

pmp = pytest.mark.parametrize


def explicit_gridder(uvw, freq, ms, wgt, nxdirty, nydirty, xpixsize, ypixsize, apply_w):
  x, y = np.meshgrid(
    *[-ss / 2 + np.arange(ss) for ss in [nxdirty, nydirty]], indexing="ij"
  )
  x *= xpixsize
  y *= ypixsize
  res = np.zeros((nxdirty, nydirty))
  eps = x**2 + y**2
  if apply_w:
    nm1 = -eps / (np.sqrt(1.0 - eps) + 1.0)
    n = nm1 + 1
  else:
    nm1 = 0.0
    n = 1.0
  for row in range(ms.shape[0]):
    for chan in range(ms.shape[1]):
      phase = (
        freq[chan]
        / LIGHTSPEED
        * (x * uvw[row, 0] + y * uvw[row, 1] - uvw[row, 2] * nm1)
      )
      if wgt is None:
        res += (ms[row, chan] * np.exp(2j * np.pi * phase)).real
      else:
        res += (ms[row, chan] * wgt[row, chan] * np.exp(2j * np.pi * phase)).real
  return res / n


@pmp("nx", (16,))
@pmp("ny", (18, 64))
@pmp("fov", (5.0,))
@pmp("nrow", (1000,))
@pmp("nchan", (1, 7))
@pmp("precision", ("single", "double"))
def test_grid_data(nx, ny, fov, nrow, nchan, precision):
  if precision == "single":
    real_type = "f4"
    complex_type = "c8"
  else:
    real_type = "f8"
    complex_type = "c16"

  np.random.seed(420)
  cell = fov * np.pi / 180 / nx
  f0 = 1e9
  freq = f0 + np.arange(nchan) * (f0 / nchan)
  uvw = (np.random.rand(nrow, 3) - 0.5) / (cell * freq[-1] / LIGHTSPEED)
  vis = (
    np.random.rand(nrow, nchan) - 0.5 + 1j * (np.random.rand(nrow, nchan) - 0.5)
  ).astype(complex_type)
  wgt = np.random.rand(nrow, nchan).astype(real_type)

  explicit_gridder(uvw, freq, vis, wgt, nx, ny, cell, cell, False)
