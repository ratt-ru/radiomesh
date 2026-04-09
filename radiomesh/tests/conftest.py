import tarfile
from pathlib import Path

import numpy as np
import pytest
import requests

MS_NAME = "test_ascii_1h60.0s.MS"
MS_TAR_NAME = f"{MS_NAME}.tar.gz"

# https://drive.google.com/file/d/1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT/view?usp=sharing

gdrive_id = "1rfGXGjjJ2XtF26LImlyJzCJMCNQZgEFT"
url = f"https://drive.google.com/uc?id={gdrive_id}"


def download_test_ms(path: Path) -> Path:
  ms_path = path / MS_NAME

  # Download and untar if the ms doesn't exist
  if not ms_path.exists():
    ms_tar_path = path / MS_TAR_NAME

    download = requests.get(url)
    with open(ms_tar_path, "wb") as f:
      f.write(download.content)

    with tarfile.open(ms_tar_path, "r:gz") as tar:
      tar.extractall(path=path, filter="data")

    ms_tar_path.unlink()

  return ms_path


@pytest.fixture(scope="session")
def ms_name():
  from appdirs import user_cache_dir

  cache_dir = Path(user_cache_dir("radiomesh")) / "test-data"
  cache_dir.mkdir(parents=True, exist_ok=True)
  return download_test_ms(cache_dir)


EPOCH = "2024-01-01T00:00:00"  # UTC reference epoch
INT_TIME_S = 8.0  # integration time in seconds
NTIME = 10

DEFAULT_TIME_PARAM = {"epoch": EPOCH, "integration_time": INT_TIME_S, "ntime": NTIME}


@pytest.fixture(params=[DEFAULT_TIME_PARAM])
def timesteps(request):
  ntime = request.param.get("ntime", NTIME)
  epoch = request.param.get("epoch", EPOCH)
  integration_time = request.param.get("integration_time", INT_TIME_S)
  from astropy.time import Time, TimeDelta

  # Observation times: ntime dumps spaced by INT_TIME_S seconds
  t0 = Time(epoch, format="isot", scale="utc")
  return t0 + TimeDelta(np.arange(ntime) * integration_time, format="sec")


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
NANT = 6
DEFAULT_UVW_PARAM = {"nant": NANT}


@pytest.fixture(params=[DEFAULT_UVW_PARAM])
def uvw_coordinates(request, timesteps):
  from astropy import units as u
  from astropy.coordinates import EarthLocation, SkyCoord

  nant = request.param.get("nant", NANT)
  if nant > MAX_ANTENNAS:
    raise ValueError(f"nant={nant} exceeds MAX_ANTENNAS={MAX_ANTENNAS}")

  ant_pos = ANTENNA_POSITIONS[:nant]

  # Source near transit for a lat=-30° array; RA=90° gives non-degenerate Hour Angles
  source = SkyCoord(ra=90.0 * u.deg, dec=-30.0 * u.deg, frame="icrs")

  # Reference location for LST computation
  ref_loc = EarthLocation.from_geocentric(*ant_pos[0], unit=u.m)

  # ECEF XYZ baseline vectors (metres), one per unique antenna pair
  ant1_idx, ant2_idx = np.triu_indices(nant, 1)
  xyz_bl = ant_pos[ant2_idx] - ant_pos[ant1_idx]  # (nbl, 3)

  # Hour Angle per timestep via mean sidereal time (avoids IERS data download)
  lst = timesteps.sidereal_time("mean", longitude=ref_loc.lon)
  H = lst.rad - source.ra.rad  # (ntime,) in radians
  delta = source.dec.rad  # scalar

  sinH = np.sin(H)
  cosH = np.cos(H)
  sindec = np.sin(delta)
  cosdec = np.cos(delta)

  # Standard radio-interferometric UVW rotation matrix, shape (ntime, 3, 3).
  # Rows give the u, v, w projections of an ECEF baseline vector.
  R = np.zeros((timesteps.shape[0], 3, 3))
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
