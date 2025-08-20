from typing import Tuple

import numpy as np
import numpy.typing as npt

from radiomesh.constants import LIGHTSPEED
from radiomesh.es_kernel import ESKernel


def wgridder_conventions(
  l0: float, m0: float
) -> Tuple[float, float, float, float, float]:
  """
  Returns:
    (u_sign, v_sign, w_sign, x0, y0)
    according to the conventions documented
    `here <https://github.com/mreineck/ducc/issues/34>_`_
    in order to match the ducc0 wgridder.
  """
  return 1.0, -1.0, 1.0, -l0, -m0


def image_params(
  uvw: npt.NDArray[np.floating],
  frequencies: npt.NDArray[np.floating],
  fov: float,
  kernel: ESKernel,
) -> Tuple[int, int, int, float, float, float, float]:
  """Determine appropriate image and cell sizes
  given ``uvw`` coordinates, ``frequencies``, field of view ``fov``
  and ``kernel``.

  Args:
    uvw: uvw coordinates
    frequencies: frequencies
    fov: field of view
    kernel: gridding kernel

  Note:
    This function currently only returns even grids

  Returns:
    A tuple (nx, ny, nw, pixelsizex, pixelsizey, w0, dw)
    of imaging parameters describing the grid dimensions,
    the cell size in radians and the w plane starting value
    and increments.
  """
  u, v, _ = uvw.reshape((-1, 3)).T

  u_max = np.abs(u.max())
  v_max = np.abs(v.max())
  freq_max = frequencies.max()

  u_cell_n = 1.0 / (2.0 * u_max * freq_max / LIGHTSPEED)
  v_cell_n = 1.0 / (2.0 * v_max * freq_max / LIGHTSPEED)
  u_cell_rad = u_cell_n / float(kernel.oversampling)
  v_cell_rad = v_cell_n / float(kernel.oversampling)

  nx = int(fov / np.rad2deg(u_cell_rad))
  ny = int(fov / np.rad2deg(v_cell_rad))

  # Avoid odd grids
  nx += int(nx % 2)
  ny += int(ny % 2)

  x, y = np.meshgrid(*[-ss / 2.0 + np.arange(ss) for ss in (nx, ny)], indexing="ij")
  x *= u_cell_rad
  y *= v_cell_rad
  eps = x**2 + y**2
  nm1 = -eps / (np.sqrt(1.0 - eps) + 1.0)

  wmax = np.abs(uvw[..., -1] * frequencies.max() / LIGHTSPEED).max()
  wmin = np.abs(uvw[..., -1] * frequencies.min() / LIGHTSPEED).min()

  # removing the factor of a half compared to expression in the
  # paper gives the same w parameters as reported by the wgridder
  # but I can't seem to get that to agree with the DFT
  dw = 1.0 / (2.0 * kernel.oversampling * np.abs(nm1).max())
  nw = int(np.ceil((wmax - wmin) / dw)) + kernel.support
  w0 = (wmin + wmax) / 2.0 - dw * (nw - 1) / 2.0

  return (nx, ny, nw, u_cell_rad, v_cell_rad, w0, dw)
