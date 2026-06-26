"""Re-export shim: general-purpose intrinsics moved to rarg-numba-patterns,
radiomesh-specific ones remain in _intrinsics_extra."""

from rarg_numba_patterns.intrinsics import (  # noqa: F401
  accumulate_data,
  atomic_rmw_intrinsic,
  field_ptr,
  item_ptr,
  load_data,
  overload_atomic_rmw,
  overload_field_ptr,
  overload_item_ptr,
)

from radiomesh._intrinsics_extra import (  # noqa: F401
  apply_flags,
  apply_weights,
  check_args,
)
