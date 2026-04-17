import numba


def make_structref_property(field):
  """Create a property that unboxes a structref field via a cached njit getter."""
  getter = numba.njit(lambda self: getattr(self, field))
  return property(lambda self: getter(self))
