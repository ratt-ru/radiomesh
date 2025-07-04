import argparse
import time

import numpy as np

from radiomesh.constants import LIGHTSPEED
from radiomesh.gridding import wgrid


def do_ducc0_wgridding(
  uvw, freq, vis, wgt, npix, cell_rad, epsilon=1e-4, precision="double", nthreads=1
):
  try:
    from ducc0.wgridder import vis2dirty
  except ImportError as e:
    raise ImportError("pip install ducc0") from e

  if precision == "double":
    rtype = "f8"
    ctype = "c16"
  else:
    rtype = "f4"
    ctype = "c8"

  vis = vis.astype(ctype)
  wgt = wgt.astype(rtype)

  nrow, nchan, ncorr = vis.shape

  dirty = np.zeros((ncorr, npix, npix), dtype=rtype)

  start = time.time()
  for c in range(ncorr):
    vis2dirty(
      uvw=uvw,
      freq=freq,
      vis=vis[:, :, c],
      wgt=wgt[:, :, c],
      npix_x=npix,
      npix_y=npix,
      pixsize_x=cell_rad,
      pixsize_y=cell_rad,
      epsilon=epsilon,
      do_wgridding=False,
      nthreads=nthreads,
      verbosity=1,
      dirty=dirty[c],
    )

  print(
    f"Time taken to map ({nrow},{nchan},{ncorr}) to "
    f"({ncorr},{npix},{npix}) = {time.time()-start}s"
  )


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("ms")
  p.add_argument("--backend", choices=["radiomesh", "ducc0"], default="radiomesh")
  return p.parse_args()


if __name__ == "__main__":
  try:
    from casacore.tables import table
  except ImportError as e:
    raise ImportError("pip install python-casacore") from e

  args = parse_args()

  ms = table(args.ms)
  utime = np.unique(ms.getcol("TIME"))
  uvw = ms.getcol("UVW")
  vis = ms.getcol("DATA")
  flags = ms.getcol("FLAG")
  try:
    wgt = ms.getcol("WEIGHT_SPECTRUM")
  except Exception:
    wgt = np.ones(vis.shape, dtype="f4")
  print(f"Visibility size {vis.nbytes / 1024.**3:.2f}GB")
  ms.close()
  freq = table(f"{args.ms}::SPECTRAL_WINDOW").getcol("CHAN_FREQ")[0]
  uv_max = np.maximum(np.abs(uvw[:, 0]).max(), np.abs(uvw[:, 1]).max())
  max_freq = freq.max()
  cell_N = 1.0 / (2 * uv_max * max_freq / LIGHTSPEED)  # max cell size
  cell_rad = cell_N / 2.0  # oversample by a factor of two
  fov = 1.0  # field of view degrees
  # import ipdb; ipdb.set_trace()
  npix = int(fov / np.rad2deg(cell_rad))
  if npix % 2:
    npix += 1

  if args.backend == "ducc0":
    do_ducc0_wgridding(uvw, freq, vis, wgt, npix, cell_rad)
  elif args.backend == "radiomesh":
    (ntime,) = utime.shape
    # Reshape to (time, baseline, chan, corr)
    uvw = uvw.reshape((ntime, -1) + uvw.shape[1:])
    vis = vis.reshape((ntime, -1) + vis.shape[1:])
    wgt = wgt.reshape((ntime, -1) + wgt.shape[1:])
    flags = flags.reshape((ntime, -1) + flags.shape[1:])

    ntime, nbl, nchan, ncorr = vis.shape
    start = time.time()
    wgrid(
      uvw,
      vis,
      wgt,
      flags,
      freq,
      npix,
      npix,
      str(fov),
      7,
      "[XX,XY,YX,YY]->[I,Q,U,V]",
    )
    print(
      f"Time taken to map ({ntime},{nbl},{nchan},{ncorr}) "
      f"to ({ncorr},{npix},{npix}) = {time.time()-start}s"
    )

  else:
    raise NotImplementedError(args.backend)
