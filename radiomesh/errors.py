from __future__ import annotations


class KernelSelectionError(ValueError):
  """Raised when no KernelDB entry satisfies the requested kernel parameters.

  This can occur when:
  - ``epsilon`` is smaller than the minimum achievable for the given
    ``(oversampling, ndim, single)`` combination.
  - ``oversampling`` is outside the supported range [1.20, 2.50].
  - ``single=True`` with a very small ``epsilon`` (single precision caps at W=8,
    limiting achievable accuracy).

  To resolve, consider:
  - Relaxing ``epsilon`` (increasing it).
  - Increasing ``oversampling`` toward 2.5.
  - Setting ``single=False`` if single precision was requested.
  - Setting ``analytic=True`` to bypass KernelDB selection entirely and use
    the analytic kernel with manually specified ``beta`` and ``e0``.
  """
