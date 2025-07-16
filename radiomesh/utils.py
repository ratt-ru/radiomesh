from typing import Tuple

import numpy as np
import numpy.typing as npt

from radiomesh.constants import LIGHTSPEED


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
  oversampling: float,
) -> Tuple[int, int, float, float]:
  """Determine appropriate image and cell sizes
  given ``uvw`` coordinates, ``frequencies``, field of view ``fov``
  and ``oversampling`` factor.

  Args:
    uvw: uvw coordinates
    frequencies: frequencies
    fov: field of view
    oversampling: oversampling factor

  Note:
    This function currently only returns even grids

  Returns:
    A tuple (nx, ny, pixelsizex, pixelsizey) of imaging parameters
    describing the image size in pixels and cell size in radians.
  """
  u, v, _ = uvw.reshape((-1, 3)).T

  u_max = np.abs(u.max())
  v_max = np.abs(v.max())
  freq_max = frequencies.max()

  u_cell_n = 1.0 / (2.0 * u_max * freq_max / LIGHTSPEED)
  v_cell_n = 1.0 / (2.0 * v_max * freq_max / LIGHTSPEED)
  u_cell_rad = u_cell_n / float(oversampling)
  v_cell_rad = v_cell_n / float(oversampling)

  nx = int(fov / np.rad2deg(u_cell_rad))
  ny = int(fov / np.rad2deg(v_cell_rad))

  # Avoid odd grids
  nx += int(nx % 2)
  ny += int(ny % 2)

  return (nx, ny, u_cell_rad, v_cell_rad)
