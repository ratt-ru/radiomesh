import numpy as np
import sympy as sm
from numba import literally, njit, types
from numba.extending import overload
from scipy.constants import c as lightspeed
from sympy.physics.quantum import TensorProduct
from sympy.utilities.lambdify import lambdify

JIT_OPTIONS = {"nogil": True, "cache": True, "error_model": "numpy", "fastmath": True}


def wgridder_conventions(l0, m0):
  """
  Returns

  flip_u, flip_v, flip_w, x0, y0

  according to the conventions documented here https://github.com/mreineck/ducc/issues/34

  These conventions are chosen to math the wgridder in ducc
  """
  return False, True, False, -l0, -m0


@njit(**JIT_OPTIONS, inline="always")
def _es_kernel(x, y, xkern, ykern, betak):
  for i in range(x.size):
    xkern[i] = np.exp(betak * (np.sqrt(1 - x[i] * x[i]) - 1))
    ykern[i] = np.exp(betak * (np.sqrt(1 - y[i] * y[i]) - 1))


@njit(**JIT_OPTIONS)
def grid_data(
  data,
  weight,
  flag,
  jones,
  tbin_idx,
  tbin_counts,
  ant1,
  ant2,
  nx,
  ny,
  nx_psf,
  ny_psf,
  pol,
  product,
  nc,
):
  vis, wgt = _grid_data_impl(
    data,
    weight,
    flag,
    jones,
    tbin_idx,
    tbin_counts,
    ant1,
    ant2,
    nx,
    ny,
    nx_psf,
    ny_psf,
    literally(pol),
    literally(product),
    literally(nc),
  )

  return vis, wgt


def _grid_data_impl(
  data,
  weight,
  flag,
  jones,
  tbin_idx,
  tbin_counts,
  ant1,
  ant2,
  nx,
  ny,
  nx_psf,
  ny_psf,
  pol,
  product,
  nc,
):
  raise NotImplementedError


@overload(
  _grid_data_impl, prefer_literal=True, jit_options={**JIT_OPTIONS, "parallel": True}
)
def nb_grid_data_impl(
  data,
  weight,
  flag,
  jones,
  tbin_idx,
  tbin_counts,
  ant1,
  ant2,
  nx,
  ny,
  nx_psf,
  ny_psf,
  k,
  pol,
  product,
  nc,
):
  vis_func, wgt_func = stokes_funcs(data, jones, product, pol, nc)

  if product.literal_value in ["I", "Q", "U", "V"]:
    ns = 1
  elif product.literal_value == "DS":
    ns = 2
  elif product.literal_value == "FS":
    ns = int(nc.literal_value)

  flip_u, flip_v, flip_w, x0, y0 = wgridder_conventions(x0, y0)  # noqa
  usign = -1.0 if flip_u else 1.0
  vsign = -1.0 if flip_v else 1.0

  def _impl(
    data,
    weight,
    flag,
    uvw,
    freq,
    jones,
    tbin_idx,
    tbin_counts,
    ant1,
    ant2,
    nx,
    ny,
    cell_size_x,
    cell_size_y,
    pol,
    product,
    nc,
    k=6,
  ):
    # for dask arrays we need to adjust the chunks to
    # start counting from zero
    tbin_idx -= tbin_idx.min()
    nt = np.shape(tbin_idx)[0]
    nrow, nchan, ncorr = data.shape
    vis_grid = np.zeros((nx, ny, ns), dtype=data.dtype)
    wgt_grid = np.zeros((nx, ny, ns), dtype=data.real.dtype)

    # ufreq
    u_cell = 1 / (nx * cell_size_x)
    # shifts fftfreq such that they start at zero
    # convenient to look up the pixel value
    umax = np.abs(-1 / cell_size_x / 2 - u_cell / 2)

    # vfreq
    v_cell = 1 / (ny * cell_size_y)
    vmax = np.abs(-1 / cell_size_y / 2 - v_cell / 2)

    normfreq = freq / lightspeed
    ko2 = k / 2
    betak = 2.3 * k
    pos = np.arange(k) - ko2
    xkern = np.zeros(k)
    ykern = np.zeros(k)
    for t in range(nt):
      for row in range(tbin_idx[t], tbin_idx[t] + tbin_counts[t]):
        p = int(ant1[row])
        q = int(ant2[row])
        gp = jones[t, p, :, 0]
        gq = jones[t, q, :, 0]
        uvw_row = uvw[row]
        wgt_row = weight[row]
        vis_row = data[row]
        for chan in range(nchan):
          if flag[row, chan]:
            continue
          wgt = wgt_func(gp[chan], gq[chan], wgt_row[chan])
          vis = vis_func(gp[chan], gq[chan], wgt_row[chan], vis_row[chan])

          # current uv coords
          chan_normfreq = normfreq[chan]
          u_tmp = uvw_row[0] * chan_normfreq * usign
          v_tmp = uvw_row[1] * chan_normfreq * vsign
          # pixel coordinates
          ug = (u_tmp + umax) / u_cell
          vg = (v_tmp + vmax) / v_cell
          # indices
          u_idx = int(np.round(ug))
          v_idx = int(np.round(vg))

          # the kernel is separable and only defined on [-1,1]
          # do we ever need to check these bounds?
          x_idx = pos + u_idx
          x = (x_idx - ug + 0.5) / ko2
          y_idx = pos + v_idx
          y = (y_idx - vg + 0.5) / ko2
          _es_kernel(x, y, xkern, ykern, betak)

          for c in range(ncorr):
            wc = wgt[c]
            for i, xi in zip(x_idx, xkern):
              for j, yj in zip(y_idx, ykern):
                xyw = xi * yj * wc
                wgt_grid[c, i, j] += xyw
                vis_grid[c, i, j] += xyw * vis[c]

    return (vis_grid, wgt_grid)

  _impl.returns = types.Tuple(
    [types.Array(types.complex128, 3, "C"), types.Array(types.float64, 3, "C")]
  )
  return _impl


def stokes_funcs(data, jones, product, pol, nc):
  # set up symbolic expressions
  gp00, gp10, gp01, gp11 = sm.symbols("gp00 gp10 gp01 gp11", real=False)
  gq00, gq10, gq01, gq11 = sm.symbols("gq00 gq10 gq01 gq11", real=False)
  w0, w1, w2, w3 = sm.symbols("W0 W1 W2 W3", real=True)
  v00, v10, v01, v11 = sm.symbols("v00 v10 v01 v11", real=False)

  # Jones matrices
  Gp = sm.Matrix([[gp00, gp01], [gp10, gp11]])
  Gq = sm.Matrix([[gq00, gq01], [gq10, gq11]])

  # Mueller matrix (row major form)
  Mpq = TensorProduct(Gp, Gq.conjugate())
  Mpqinv = TensorProduct(Gp.inv(), Gq.conjugate().inv())

  # inverse noise covariance
  Sinv = sm.Matrix([[w0, 0, 0, 0], [0, w1, 0, 0], [0, 0, w2, 0], [0, 0, 0, w3]])
  S = Sinv.inv()

  # visibilities
  Vpq = sm.Matrix([[v00], [v01], [v10], [v11]])

  # Full Stokes to corr operator
  # Is this the only difference between linear and circular pol?
  # What about paralactic angle rotation?
  if pol.literal_value == "linear":
    T = sm.Matrix(
      [[1.0, 1.0, 0, 0], [0, 0, 1.0, 1.0j], [0, 0, 1.0, -1.0j], [1.0, -1.0, 0, 0]]
    )
  elif pol.literal_value == "circular":
    T = sm.Matrix(
      [[1.0, 0, 0, 1.0], [0, 1.0, 1.0j, 0], [0, 1.0, -1.0j, 0], [1.0, 0, 0, -1.0]]
    )
  Tinv = T.inv()

  # Full Stokes weights
  W = T.H * Mpq.H * Sinv * Mpq * T
  Winv = Tinv * Mpqinv * S * Mpqinv.H * Tinv.H

  # Full Stokes coherencies
  C = Winv * (T.H * (Mpq.H * (Sinv * Vpq)))
  # Only keep diagonal of weights
  W = W.diagonal().T  # diagonal() returns row vector

  if product.literal_value == "I":
    i = (0,)
  elif product.literal_value == "Q":
    i = (1,)
  elif product.literal_value == "U":
    i = (2,)
  elif product.literal_value == "V":
    i = (3,)
  elif product.literal_value == "DS":
    if pol.literal_value == "linear":
      i = (0, 1)
    elif pol.literal_value == "circular":
      i = (0, -1)
  elif product.literal_value == "FS":
    if nc.literal_value == "2":
      if pol.literal_value == "linear":
        i = (0, 1)
      elif pol.literal_value == "circular":
        i = (0, -1)
    elif nc.literal_value == "4":
      i = (0, 1, 2, 3)
  else:
    raise ValueError(f"Unknown polarisation product {product}")

  if jones.ndim == 6:  # Full mode
    Wsymb = lambdify(
      (gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3),
      sm.simplify(W[i, 0]),
    )
    Wjfn = njit(nogil=True, inline="always")(Wsymb)

    Dsymb = lambdify(
      (
        gp00,
        gp01,
        gp10,
        gp11,
        gq00,
        gq01,
        gq10,
        gq11,
        w0,
        w1,
        w2,
        w3,
        v00,
        v01,
        v10,
        v11,
      ),
      sm.simplify(C[i, 0]),
    )
    Djfn = njit(nogil=True, inline="always")(Dsymb)

    @njit(nogil=True, inline="always")
    def wfunc(gp, gq, W):
      gp00 = gp[0, 0]
      gp01 = gp[0, 1]
      gp10 = gp[1, 0]
      gp11 = gp[1, 1]
      gq00 = gq[0, 0]
      gq01 = gq[0, 1]
      gq10 = gq[1, 0]
      gq11 = gq[1, 1]
      W00 = W[0]
      W01 = W[1]
      W10 = W[2]
      W11 = W[3]
      return Wjfn(
        gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, W00, W01, W10, W11
      ).real.ravel()

    @njit(nogil=True, inline="always")
    def vfunc(gp, gq, W, V):
      gp00 = gp[0, 0]
      gp01 = gp[0, 1]
      gp10 = gp[1, 0]
      gp11 = gp[1, 1]
      gq00 = gq[0, 0]
      gq01 = gq[0, 1]
      gq10 = gq[1, 0]
      gq11 = gq[1, 1]
      W00 = W[0]
      W01 = W[1]
      W10 = W[2]
      W11 = W[3]
      V00 = V[0]
      V01 = V[1]
      V10 = V[2]
      V11 = V[3]
      return Djfn(
        gp00,
        gp01,
        gp10,
        gp11,
        gq00,
        gq01,
        gq10,
        gq11,
        W00,
        W01,
        W10,
        W11,
        V00,
        V01,
        V10,
        V11,
      ).ravel()

  elif jones.ndim == 5:  # DIAG mode
    W = W.subs(gp10, 0)
    W = W.subs(gp01, 0)
    W = W.subs(gq10, 0)
    W = W.subs(gq01, 0)
    C = C.subs(gp10, 0)
    C = C.subs(gp01, 0)
    C = C.subs(gq10, 0)
    C = C.subs(gq01, 0)

    Wsymb = lambdify((gp00, gp11, gq00, gq11, w0, w1, w2, w3), sm.simplify(W[i, 0]))
    Wjfn = njit(nogil=True, inline="always")(Wsymb)

    Dsymb = lambdify(
      (gp00, gp11, gq00, gq11, w0, w1, w2, w3, v00, v01, v10, v11),
      sm.simplify(C[i, 0]),
    )
    Djfn = njit(nogil=True, inline="always")(Dsymb)

    if nc.literal_value == "4":

      @njit(nogil=True, inline="always")
      def wfunc(gp, gq, W):
        gp00 = gp[0]
        gp11 = gp[1]
        gq00 = gq[0]
        gq11 = gq[1]
        W00 = W[0]
        W01 = W[1]
        W10 = W[2]
        W11 = W[3]
        return Wjfn(gp00, gp11, gq00, gq11, W00, W01, W10, W11).real.ravel()

      @njit(nogil=True, inline="always")
      def vfunc(gp, gq, W, V):
        gp00 = gp[0]
        gp11 = gp[1]
        gq00 = gq[0]
        gq11 = gq[1]
        W00 = W[0]
        W01 = W[1]
        W10 = W[2]
        W11 = W[3]
        V00 = V[0]
        V01 = V[1]
        V10 = V[2]
        V11 = V[3]
        return Djfn(
          gp00, gp11, gq00, gq11, W00, W01, W10, W11, V00, V01, V10, V11
        ).ravel()
    elif nc.literal_value == "2":

      @njit(nogil=True, inline="always")
      def wfunc(gp, gq, W):
        gp00 = gp[0]
        gp11 = gp[1]
        gq00 = gq[0]
        gq11 = gq[1]
        W00 = W[0]
        W01 = 1.0
        W10 = 1.0
        W11 = W[-1]
        return Wjfn(gp00, gp11, gq00, gq11, W00, W01, W10, W11).real.ravel()

      @njit(nogil=True, inline="always")
      def vfunc(gp, gq, W, V):
        gp00 = gp[0]
        gp11 = gp[1]
        gq00 = gq[0]
        gq11 = gq[1]
        W00 = W[0]
        W01 = 1.0
        W10 = 1.0
        W11 = W[-1]
        V00 = V[0]
        V01 = 0j
        V10 = 0j
        V11 = V[-1]
        return Djfn(
          gp00, gp11, gq00, gq11, W00, W01, W10, W11, V00, V01, V10, V11
        ).ravel()
    else:
      raise ValueError(
        f"Selected product is only available from 2 or 4"
        f"correlation data while you have ncorr={nc}."
      )

  else:
    raise ValueError("Jones term has incorrect number of dimensions")

  return vfunc, wfunc
