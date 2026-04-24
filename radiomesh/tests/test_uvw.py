import numba
import numpy as np
import pytest

from radiomesh.gridding_types import WGridderImpl

# Synthetic Y-shaped antenna array near lat=-30°, lon=21°
# (MeerKAT-like, ~1 km baselines).
# Positions are ECEF geocentric XYZ in metres, derived from a Y-shaped layout
# with arm spacing of 200 m, centred at the reference geodetic point.
ANTENNA_POSITIONS = np.array(
  [
    [5163013.0, 1981582.0, -3170885.0],  # centre
    [5163106.0, 1981618.0, -3170712.0],  # N arm, 200 m
    [5163200.0, 1981654.0, -3170539.0],  # N arm, 400 m
    [5163293.0, 1981690.0, -3170365.0],  # N arm, 600 m
    [5163028.0, 1981402.0, -3170972.0],  # SW arm, 200 m
    [5163044.0, 1981223.0, -3171058.0],  # SW arm, 400 m
    [5163059.0, 1981043.0, -3171145.0],  # SW arm, 600 m
    [5162904.0, 1981726.0, -3170972.0],  # SE arm, 200 m
    [5162796.0, 1981870.0, -3171058.0],  # SE arm, 400 m
    [5162687.0, 1982013.0, -3171145.0],  # SE arm, 600 m
  ],
  dtype=np.float64,
)

MAX_ANTENNAS = len(ANTENNA_POSITIONS)

EPOCH = "2024-01-01T00:00:00"  # UTC reference epoch
INT_TIME_S = 8.0  # integration time in seconds

DEFAULT_PARAM = {"ntime": 10, "nant": 6}


@pytest.fixture(params=[DEFAULT_PARAM])
def uvw_coordinates(request):
  from astropy import units as u
  from astropy.coordinates import EarthLocation, SkyCoord
  from astropy.time import Time, TimeDelta

  ntime = request.param["ntime"]
  nant = request.param["nant"]
  if nant > MAX_ANTENNAS:
    raise ValueError(f"nant={nant} exceeds MAX_ANTENNAS={MAX_ANTENNAS}")

  ant_pos = ANTENNA_POSITIONS[:nant]

  # Observation times: ntime dumps spaced by INT_TIME_S seconds
  t0 = Time(EPOCH, format="isot", scale="utc")
  times = t0 + TimeDelta(np.arange(ntime) * INT_TIME_S, format="sec")

  # Source near transit for a lat=-30° array; RA=90° gives non-degenerate Hour Angles
  source = SkyCoord(ra=90.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")

  # Reference location for LST computation
  ref_loc = EarthLocation.from_geocentric(*ant_pos[0], unit=u.m)

  # ECEF XYZ baseline vectors (metres), one per unique antenna pair
  ant1_idx, ant2_idx = np.triu_indices(nant, 1)
  xyz_bl = ant_pos[ant2_idx] - ant_pos[ant1_idx]  # (nbl, 3)

  # Hour Angle per timestep via mean sidereal time (avoids IERS data download)
  lst = times.sidereal_time("mean", longitude=ref_loc.lon)
  H = lst.rad - source.ra.rad  # (ntime,) in radians
  delta = source.dec.rad  # scalar

  sinH = np.sin(H)
  cosH = np.cos(H)
  sindec = np.sin(delta)
  cosdec = np.cos(delta)

  # Standard radio-interferometric UVW rotation matrix, shape (ntime, 3, 3).
  # Rows give the u, v, w projections of an ECEF baseline vector.
  R = np.zeros((ntime, 3, 3))
  R[:, 0, 0] = sinH
  R[:, 0, 1] = cosH
  R[:, 1, 0] = -sindec * cosH
  R[:, 1, 1] = sindec * sinH
  R[:, 1, 2] = cosdec
  R[:, 2, 0] = cosdec * cosH
  R[:, 2, 1] = -cosdec * sinH
  R[:, 2, 2] = sindec

  # uvw[t, b, :] = R[t] @ xyz_bl[b]; shape (ntime, nbl, 3)
  return np.einsum("tij,bj->tbi", R, xyz_bl)


@pytest.fixture(params=[16])
def frequencies(request):
  return np.linspace(0.856e9, 2 * 0.856e9, request.param)


@numba.njit(nogil=True, parallel=True)
def scan_data(visibilities, weight, flag, uvw, frequencies, gridding=False):
  ntime, nbl, _ = uvw.shape
  (nchan,) = frequencies.shape

  grid_meta = WGridderImpl(uvw, frequencies, False, False, False)

  ntime = grid_meta.ntime
  nbl = grid_meta.nbl
  nchan = grid_meta.nchan

  lmask = np.zeros((ntime, nbl, nchan), np.uint8)
  nvis = 0
  wmin_d = 1e300
  wmax_d = -1e300

  for t in numba.prange(ntime):
    for bl in numba.prange(nbl):
      for ch in range(nchan):
        v = visibilities[t, bl, ch]
        if (v.real**2 + v.imag**2) * weight[t, bl, ch] * (flag[t, bl, ch]) != 0:
          lmask[t, bl, ch] = 1
          nvis += 1
          w = grid_meta.effective_abs_w(t, bl, ch)
          wmin_d = min(wmin_d, w)
          wmax_d = max(wmax_d, w)

  return nvis, wmin_d, wmax_d


def test_uvw_scan(uvw_coordinates, frequencies):
  rng = np.random.default_rng(seed=42)
  ntime, nbl, _ = uvw_coordinates.shape
  (nchan,) = frequencies.shape

  weight = rng.random((ntime, nbl, nchan))
  vis = rng.random(weight.shape) + rng.random(weight.shape) * 1j
  flag = rng.integers(0, 8, size=weight.shape, dtype=np.uint8)

  scan_data(vis, weight, flag, uvw_coordinates, frequencies)

  breakpoint()
