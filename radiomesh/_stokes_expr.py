from numpy import conjugate as conj


def LINEAR_VIS_I(
  gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11
):
  return (
    0.5
    * (
      -gp00 * v10 * conj(gq10)
      + gp00 * v11 * conj(gq00)
      - gp01 * v10 * conj(gq11)
      + gp01 * v11 * conj(gq01)
      + gp10 * v00 * conj(gq10)
      - gp10 * v01 * conj(gq00)
      + gp11 * v00 * conj(gq11)
      - gp11 * v01 * conj(gq01)
    )
    / (
      gp00 * gp11 * conj(gq00) * conj(gq11)
      - gp00 * gp11 * conj(gq01) * conj(gq10)
      - gp01 * gp10 * conj(gq00) * conj(gq11)
      + gp01 * gp10 * conj(gq01) * conj(gq10)
    )
  )


def LINEAR_VIS_Q(
  gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11
):
  return (
    0.5
    * (
      gp00 * v10 * conj(gq10)
      - gp00 * v11 * conj(gq00)
      - gp01 * v10 * conj(gq11)
      + gp01 * v11 * conj(gq01)
      - gp10 * v00 * conj(gq10)
      + gp10 * v01 * conj(gq00)
      + gp11 * v00 * conj(gq11)
      - gp11 * v01 * conj(gq01)
    )
    / (
      gp00 * gp11 * conj(gq00) * conj(gq11)
      - gp00 * gp11 * conj(gq01) * conj(gq10)
      - gp01 * gp10 * conj(gq00) * conj(gq11)
      + gp01 * gp10 * conj(gq01) * conj(gq10)
    )
  )


def LINEAR_VIS_U(
  gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11
):
  return (
    0.5
    * (
      gp00 * v10 * conj(gq11)
      - gp00 * v11 * conj(gq01)
      + gp01 * v10 * conj(gq10)
      - gp01 * v11 * conj(gq00)
      - gp10 * v00 * conj(gq11)
      + gp10 * v01 * conj(gq01)
      - gp11 * v00 * conj(gq10)
      + gp11 * v01 * conj(gq00)
    )
    / (
      gp00 * gp11 * conj(gq00) * conj(gq11)
      - gp00 * gp11 * conj(gq01) * conj(gq10)
      - gp01 * gp10 * conj(gq00) * conj(gq11)
      + gp01 * gp10 * conj(gq01) * conj(gq10)
    )
  )


def LINEAR_VIS_V(
  gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11
):
  return (
    0.5
    * 1j
    * (
      gp00 * v10 * conj(gq11)
      - gp00 * v11 * conj(gq01)
      - gp01 * v10 * conj(gq10)
      + gp01 * v11 * conj(gq00)
      - gp10 * v00 * conj(gq11)
      + gp10 * v01 * conj(gq01)
      + gp11 * v00 * conj(gq10)
      - gp11 * v01 * conj(gq00)
    )
    / (
      gp00 * gp11 * conj(gq00) * conj(gq11)
      - gp00 * gp11 * conj(gq01) * conj(gq10)
      - gp01 * gp10 * conj(gq00) * conj(gq11)
      + gp01 * gp10 * conj(gq01) * conj(gq10)
    )
  )


def LINEAR_WEIGHT_I(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3):
  return (
    gp00 * w0 * (gq00 * conj(gp00) + gq01 * conj(gp01)) * conj(gq00)
    + gp00 * w1 * (gq10 * conj(gp00) + gq11 * conj(gp01)) * conj(gq10)
    + gp01 * w0 * (gq00 * conj(gp00) + gq01 * conj(gp01)) * conj(gq01)
    + gp01 * w1 * (gq10 * conj(gp00) + gq11 * conj(gp01)) * conj(gq11)
    + gp10 * w2 * (gq00 * conj(gp10) + gq01 * conj(gp11)) * conj(gq00)
    + gp10 * w3 * (gq10 * conj(gp10) + gq11 * conj(gp11)) * conj(gq10)
    + gp11 * w2 * (gq00 * conj(gp10) + gq01 * conj(gp11)) * conj(gq01)
    + gp11 * w3 * (gq10 * conj(gp10) + gq11 * conj(gp11)) * conj(gq11)
  )


def LINEAR_WEIGHT_Q(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3):
  return (
    gp00 * w0 * (gq00 * conj(gp00) - gq01 * conj(gp01)) * conj(gq00)
    + gp00 * w1 * (gq10 * conj(gp00) - gq11 * conj(gp01)) * conj(gq10)
    - gp01 * w0 * (gq00 * conj(gp00) - gq01 * conj(gp01)) * conj(gq01)
    - gp01 * w1 * (gq10 * conj(gp00) - gq11 * conj(gp01)) * conj(gq11)
    + gp10 * w2 * (gq00 * conj(gp10) - gq01 * conj(gp11)) * conj(gq00)
    + gp10 * w3 * (gq10 * conj(gp10) - gq11 * conj(gp11)) * conj(gq10)
    - gp11 * w2 * (gq00 * conj(gp10) - gq01 * conj(gp11)) * conj(gq01)
    - gp11 * w3 * (gq10 * conj(gp10) - gq11 * conj(gp11)) * conj(gq11)
  )


def LINEAR_WEIGHT_U(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3):
  return (
    gp00 * w0 * (gq00 * conj(gp01) + gq01 * conj(gp00)) * conj(gq01)
    + gp00 * w1 * (gq10 * conj(gp01) + gq11 * conj(gp00)) * conj(gq11)
    + gp01 * w0 * (gq00 * conj(gp01) + gq01 * conj(gp00)) * conj(gq00)
    + gp01 * w1 * (gq10 * conj(gp01) + gq11 * conj(gp00)) * conj(gq10)
    + gp10 * w2 * (gq00 * conj(gp11) + gq01 * conj(gp10)) * conj(gq01)
    + gp10 * w3 * (gq10 * conj(gp11) + gq11 * conj(gp10)) * conj(gq11)
    + gp11 * w2 * (gq00 * conj(gp11) + gq01 * conj(gp10)) * conj(gq00)
    + gp11 * w3 * (gq10 * conj(gp11) + gq11 * conj(gp10)) * conj(gq10)
  )


def LINEAR_WEIGHT_V(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3):
  return (
    -gp00 * w0 * (gq00 * conj(gp01) - gq01 * conj(gp00)) * conj(gq01)
    - gp00 * w1 * (gq10 * conj(gp01) - gq11 * conj(gp00)) * conj(gq11)
    + gp01 * w0 * (gq00 * conj(gp01) - gq01 * conj(gp00)) * conj(gq00)
    + gp01 * w1 * (gq10 * conj(gp01) - gq11 * conj(gp00)) * conj(gq10)
    - gp10 * w2 * (gq00 * conj(gp11) - gq01 * conj(gp10)) * conj(gq01)
    - gp10 * w3 * (gq10 * conj(gp11) - gq11 * conj(gp10)) * conj(gq11)
    + gp11 * w2 * (gq00 * conj(gp11) - gq01 * conj(gp10)) * conj(gq00)
    + gp11 * w3 * (gq10 * conj(gp11) - gq11 * conj(gp10)) * conj(gq10)
  )


def CIRCULAR_VIS_I(
  gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11
):
  return (
    0.5
    * (
      -gp00 * v10 * conj(gq10)
      + gp00 * v11 * conj(gq00)
      - gp01 * v10 * conj(gq11)
      + gp01 * v11 * conj(gq01)
      + gp10 * v00 * conj(gq10)
      - gp10 * v01 * conj(gq00)
      + gp11 * v00 * conj(gq11)
      - gp11 * v01 * conj(gq01)
    )
    / (
      gp00 * gp11 * conj(gq00) * conj(gq11)
      - gp00 * gp11 * conj(gq01) * conj(gq10)
      - gp01 * gp10 * conj(gq00) * conj(gq11)
      + gp01 * gp10 * conj(gq01) * conj(gq10)
    )
  )


def CIRCULAR_VIS_Q(
  gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11
):
  return (
    0.5
    * (
      gp00 * v10 * conj(gq11)
      - gp00 * v11 * conj(gq01)
      + gp01 * v10 * conj(gq10)
      - gp01 * v11 * conj(gq00)
      - gp10 * v00 * conj(gq11)
      + gp10 * v01 * conj(gq01)
      - gp11 * v00 * conj(gq10)
      + gp11 * v01 * conj(gq00)
    )
    / (
      gp00 * gp11 * conj(gq00) * conj(gq11)
      - gp00 * gp11 * conj(gq01) * conj(gq10)
      - gp01 * gp10 * conj(gq00) * conj(gq11)
      + gp01 * gp10 * conj(gq01) * conj(gq10)
    )
  )


def CIRCULAR_VIS_U(
  gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11
):
  return (
    0.5
    * 1j
    * (
      gp00 * v10 * conj(gq11)
      - gp00 * v11 * conj(gq01)
      - gp01 * v10 * conj(gq10)
      + gp01 * v11 * conj(gq00)
      - gp10 * v00 * conj(gq11)
      + gp10 * v01 * conj(gq01)
      + gp11 * v00 * conj(gq10)
      - gp11 * v01 * conj(gq00)
    )
    / (
      gp00 * gp11 * conj(gq00) * conj(gq11)
      - gp00 * gp11 * conj(gq01) * conj(gq10)
      - gp01 * gp10 * conj(gq00) * conj(gq11)
      + gp01 * gp10 * conj(gq01) * conj(gq10)
    )
  )


def CIRCULAR_VIS_V(
  gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3, v00, v01, v10, v11
):
  return (
    0.5
    * (
      gp00 * v10 * conj(gq10)
      - gp00 * v11 * conj(gq00)
      - gp01 * v10 * conj(gq11)
      + gp01 * v11 * conj(gq01)
      - gp10 * v00 * conj(gq10)
      + gp10 * v01 * conj(gq00)
      + gp11 * v00 * conj(gq11)
      - gp11 * v01 * conj(gq01)
    )
    / (
      gp00 * gp11 * conj(gq00) * conj(gq11)
      - gp00 * gp11 * conj(gq01) * conj(gq10)
      - gp01 * gp10 * conj(gq00) * conj(gq11)
      + gp01 * gp10 * conj(gq01) * conj(gq10)
    )
  )


def CIRCULAR_WEIGHT_I(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3):
  return (
    gp00 * w0 * (gq00 * conj(gp00) + gq01 * conj(gp01)) * conj(gq00)
    + gp00 * w1 * (gq10 * conj(gp00) + gq11 * conj(gp01)) * conj(gq10)
    + gp01 * w0 * (gq00 * conj(gp00) + gq01 * conj(gp01)) * conj(gq01)
    + gp01 * w1 * (gq10 * conj(gp00) + gq11 * conj(gp01)) * conj(gq11)
    + gp10 * w2 * (gq00 * conj(gp10) + gq01 * conj(gp11)) * conj(gq00)
    + gp10 * w3 * (gq10 * conj(gp10) + gq11 * conj(gp11)) * conj(gq10)
    + gp11 * w2 * (gq00 * conj(gp10) + gq01 * conj(gp11)) * conj(gq01)
    + gp11 * w3 * (gq10 * conj(gp10) + gq11 * conj(gp11)) * conj(gq11)
  )


def CIRCULAR_WEIGHT_Q(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3):
  return (
    gp00 * w0 * (gq00 * conj(gp01) + gq01 * conj(gp00)) * conj(gq01)
    + gp00 * w1 * (gq10 * conj(gp01) + gq11 * conj(gp00)) * conj(gq11)
    + gp01 * w0 * (gq00 * conj(gp01) + gq01 * conj(gp00)) * conj(gq00)
    + gp01 * w1 * (gq10 * conj(gp01) + gq11 * conj(gp00)) * conj(gq10)
    + gp10 * w2 * (gq00 * conj(gp11) + gq01 * conj(gp10)) * conj(gq01)
    + gp10 * w3 * (gq10 * conj(gp11) + gq11 * conj(gp10)) * conj(gq11)
    + gp11 * w2 * (gq00 * conj(gp11) + gq01 * conj(gp10)) * conj(gq00)
    + gp11 * w3 * (gq10 * conj(gp11) + gq11 * conj(gp10)) * conj(gq10)
  )


def CIRCULAR_WEIGHT_U(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3):
  return (
    -gp00 * w0 * (gq00 * conj(gp01) - gq01 * conj(gp00)) * conj(gq01)
    - gp00 * w1 * (gq10 * conj(gp01) - gq11 * conj(gp00)) * conj(gq11)
    + gp01 * w0 * (gq00 * conj(gp01) - gq01 * conj(gp00)) * conj(gq00)
    + gp01 * w1 * (gq10 * conj(gp01) - gq11 * conj(gp00)) * conj(gq10)
    - gp10 * w2 * (gq00 * conj(gp11) - gq01 * conj(gp10)) * conj(gq01)
    - gp10 * w3 * (gq10 * conj(gp11) - gq11 * conj(gp10)) * conj(gq11)
    + gp11 * w2 * (gq00 * conj(gp11) - gq01 * conj(gp10)) * conj(gq00)
    + gp11 * w3 * (gq10 * conj(gp11) - gq11 * conj(gp10)) * conj(gq10)
  )


def CIRCULAR_WEIGHT_V(gp00, gp01, gp10, gp11, gq00, gq01, gq10, gq11, w0, w1, w2, w3):
  return (
    gp00 * w0 * (gq00 * conj(gp00) - gq01 * conj(gp01)) * conj(gq00)
    + gp00 * w1 * (gq10 * conj(gp00) - gq11 * conj(gp01)) * conj(gq10)
    - gp01 * w0 * (gq00 * conj(gp00) - gq01 * conj(gp01)) * conj(gq01)
    - gp01 * w1 * (gq10 * conj(gp00) - gq11 * conj(gp01)) * conj(gq11)
    + gp10 * w2 * (gq00 * conj(gp10) - gq01 * conj(gp11)) * conj(gq00)
    + gp10 * w3 * (gq10 * conj(gp10) - gq11 * conj(gp11)) * conj(gq10)
    - gp11 * w2 * (gq00 * conj(gp10) - gq01 * conj(gp11)) * conj(gq01)
    - gp11 * w3 * (gq10 * conj(gp10) - gq11 * conj(gp11)) * conj(gq11)
  )
