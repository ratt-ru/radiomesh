import numpy as np
import pytest

from radiomesh.rime.brightness import BrightnessTerm
from radiomesh.rime.core import CoreArguments
from radiomesh.rime.phase import PhaseTerm

NANT = 7
AUTO_CORRS = True
DEFAULT_CORE_ARG_PARAMS = {"nant": NANT, "auto_corrs": AUTO_CORRS}


@pytest.fixture(params=[DEFAULT_CORE_ARG_PARAMS])
def core_args(request, uvw_coordinates, frequencies):
  nant = request.param.get("nant", NANT)
  auto_corrs = request.param.get("auto_corrs", AUTO_CORRS)
  time = np.linspace(1.0, 100.0, 10)
  ant1, ant2 = np.triu_indices(nant, 1 if auto_corrs else 0)
  ant1_names = np.array([f"ANTENNA-{a + 1}" for a in ant1])
  ant2_names = np.array([f"ANTENNA-{a + 1}" for a in ant2])

  return CoreArguments(time, uvw_coordinates, frequencies, ant1_names, ant2_names)


@pytest.mark.parametrize(
  "uvw_coordinates", [{"nant": NANT, "auto_corrs": AUTO_CORRS}], indirect=True
)
def test_core_arguments(core_args):
  ant1, ant2 = np.triu_indices(NANT, 1 if AUTO_CORRS else 0)
  np.testing.assert_array_equal(core_args.antenna1, ant1)
  np.testing.assert_array_equal(core_args.antenna2, ant2)


def test_phase_term(core_args):
  phase_centre = np.zeros(2, np.float64)
  radec = np.array([[0.23, 0.11]], np.float64)
  phase = PhaseTerm(phase_centre, radec)
  print(phase.lmnm1)
  print(phase.sample(core_args, 0, 0, 0, 0))


def test_brightness_term(core_args):
  brightness = BrightnessTerm(
    "pq", ("I", "Q", "U", "V"), np.random.random((16, 4)), ("XX", "XY", "YX", "YY")
  )

  print(brightness.sample(core_args, 0, 0, 0, 0))
