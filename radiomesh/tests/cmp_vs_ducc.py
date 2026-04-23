import argparse
import time

import numpy as np

from radiomesh.es_kernel_structref import ESKernelProxy
from radiomesh.generated._es_kernel_params import KERNEL_DB
from radiomesh.gridding import wgrid
from radiomesh.literals import Datum, Schema
from radiomesh.parameters import WGridderParameters
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
  epsilon = 1e-4
  apply_w = False

  best = None
  for entry in KERNEL_DB:
    if (
      entry.ndim == (3 if apply_w else 2)
      and not entry.single
      and entry.oversampling <= 2.0
      and entry.epsilon <= epsilon
      and (best is None or entry.support < best.support)
    ):
      best = entry
  if best is None:
    raise RuntimeError("No matching KERNEL_DB entry")

  kernel = ESKernelProxy.fully_specified(
    epsilon=epsilon,
    oversampling=best.oversampling,
    beta=best.beta,
    e0=best.e0,
    support=best.support,
    analytic=True,
    single=False,
    apply_w=apply_w,
  )
  nx, ny, nw, pixsizex, pixsizey, wmin, wmax, dw = image_params(uvw, freq, fov, kernel)

  if args.backend == "ducc0":
    do_ducc0_wgridding(uvw, freq, vis, wgt, nx, ny, pixsizex, pixsizey, kernel.epsilon)
  elif args.backend == "radiomesh":
    (ntime,) = utime.shape
    # Reshape to (time, baseline, chan, corr)
    uvw = uvw.reshape((ntime, -1) + uvw.shape[1:])
    vis = vis.reshape((ntime, -1) + vis.shape[1:])
    wgt = wgt.reshape((ntime, -1) + wgt.shape[1:])
    flags = flags.reshape((ntime, -1) + flags.shape[1:])

    wgrid_params = WGridderParameters(
      nu=nx,
      nv=ny,
      kernel=kernel,
      wmin=wmin,
      wmax=wmax,
      nw=nw,
      nm1min=0.0,
      nm1max=0.0,
      nshift=0.0,
    )

    ntime, nbl, nchan, ncorr = vis.shape
    start = time.time()
    wgrid(
      uvw,
      vis,
      wgt,
      flags,
      freq,
      wgrid_params,
      nx,
      ny,
      pixsizex,
      pixsizey,
      Schema(("XX", "XY", "YX", "YY")),
      Schema(("I", "Q", "U", "V")),
      Datum(apply_w),
    )
    print(
      f"Time taken to map ({ntime},{nbl},{nchan},{ncorr}) "
      f"to ({ncorr},{nx},{ny}) = {time.time()-start}s"
    )

  else:
    raise NotImplementedError(args.backend)
