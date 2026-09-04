import pytest

from radiomesh.rime.specification import RimeSpecification


@pytest.mark.parametrize(
  "rime_spec, die_terms, dde_terms, stokes, pols",
  [
    (
      "[Gp, (Ep, Lp, Kpq, Bpq, Lq, Eq), Gq]: [I, Q, U, V] -> [XX, XY, YX, YY]",
      [("G", "p"), ("G", "q")],
      [("E", "p"), ("L", "p"), ("K", "pq"), ("B", "pq"), ("L", "q"), ("E", "q")],
      ("I", "Q", "U", "V"),
      ("XX", "XY", "YX", "YY"),
    )
  ],
)
def test_rime_specification(rime_spec, die_terms, dde_terms, stokes, pols):
  spec = RimeSpecification(rime_spec)
  assert spec._pols == pols
  assert spec._stokes == stokes
  assert spec._die_terms == die_terms
  assert spec._dde_terms == dde_terms
  assert spec._dde_index == 1
