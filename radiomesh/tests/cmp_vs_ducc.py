import argparse
import time

import numpy as np

from radiomesh.gridding import WGridderParameters, wgrid
from radiomesh.literals import Datum, Schema
from radiomesh.utils import image_params


def do_ducc0_wgridding(
  uvw,
  freq,
  vis,
  wgt,
  nx,
  ny,
  pixsizex,
  pixsizey,
  epsilon=1e-4,
  precision="double",
  nthreads=1,
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

  dirty = np.zeros((ncorr, nx, ny), dtype=rtype)

  start = time.time()
  for c in range(ncorr):
    vis2dirty(
      uvw=uvw,
      freq=freq,
      vis=vis[:, :, c],
      wgt=wgt[:, :, c],
      npix_x=nx,
      npix_y=ny,
      pixsize_x=pixsizex,
      pixsize_y=pixsizey,
      epsilon=epsilon,
      do_wgridding=False,
      nthreads=nthreads,
      verbosity=1,
      dirty=dirty[c],
    )

  print(
    f"Time taken to map ({nrow},{nchan},{ncorr}) to "
    f"({ncorr},{nx},{ny}) = {time.time()-start}s"
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

  fov = 1.0
  oversampling = 2.0
  epsilon = 1e-4
  nx, ny, pixsizex, pixsizey = image_params(uvw, freq, fov, oversampling)

  if args.backend == "ducc0":
    do_ducc0_wgridding(uvw, freq, vis, wgt, nx, ny, pixsizex, pixsizey, epsilon)
  elif args.backend == "radiomesh":
    (ntime,) = utime.shape
    # Reshape to (time, baseline, chan, corr)
    uvw = uvw.reshape((ntime, -1) + uvw.shape[1:])
    vis = vis.reshape((ntime, -1) + vis.shape[1:])
    wgt = wgt.reshape((ntime, -1) + wgt.shape[1:])
    flags = flags.reshape((ntime, -1) + flags.shape[1:])

    wgrid_params = WGridderParameters(
      nx,
      ny,
      pixsizex,
      pixsizey,
      epsilon,
      Schema(("XX", "XY", "YX", "YY")),
      Schema(("I", "Q", "U", "V")),
    )

    ntime, nbl, nchan, ncorr = vis.shape
    start = time.time()
    wgrid(
      uvw,
      vis,
      wgt,
      flags,
      freq,
      Datum(wgrid_params),
    )
    print(
      f"Time taken to map ({ntime},{nbl},{nchan},{ncorr}) "
      f"to ({ncorr},{nx},{ny}) = {time.time()-start}s"
    )

  else:
    raise NotImplementedError(args.backend)
