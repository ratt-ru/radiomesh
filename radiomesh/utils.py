from typing import Tuple


def wgridder_conventions(
  l0: float, m0: float
) -> Tuple[float, float, float, float, float]:
  """
  Returns:
    u_sign, v_sign, w_sign, x0, y0

  according to the conventions documented here https://github.com/mreineck/ducc/issues/34

  These conventions are chosen to math the wgridder in ducc
  """
  return 1.0, -1.0, 1.0, -l0, -m0
