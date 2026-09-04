"""Quick prototype: can numba typed.List hold KernelParams NamedTuples?"""

import numba
import pytest
from numba import njit
from numba.typed import List

from radiomesh.generated._es_kernel_params import KERNEL_DB, KernelParams


def _make_kernel_list():
  """Build a typed.List from KERNEL_DB."""
  lst = List()
  for k in KERNEL_DB:
    lst.append(k)
  return lst


def test_typed_list_of_namedtuple():
  """Verify we can iterate a typed.List[KernelParams] inside njit."""
  kernel_list = _make_kernel_list()

  @njit
  def find_best_support(kl, target_epsilon, ndim, single):
    best_idx = -1
    best_oversampling = 1e30
    for i in range(len(kl)):
      k = kl[i]
      if (
        k.ndim == ndim
        and k.single == single
        and k.epsilon <= target_epsilon
        and k.oversampling < best_oversampling
      ):
        best_idx = i
        best_oversampling = k.oversampling
    return best_idx

  idx = find_best_support(kernel_list, 1e-4, 2, False)
  assert idx >= 0, "Should find at least one matching kernel"
  k = KERNEL_DB[idx]
  assert k.ndim == 2
  assert k.single is False
  assert k.epsilon <= 1e-4
  print(f"OK: found kernel at index {idx}: {k}")


def test_typed_list_bool_field():
  """Specifically test that the bool field round-trips correctly."""
  kernel_list = _make_kernel_list()

  @njit
  def check_bool_field(kl):
    n_single = 0
    n_double = 0
    for i in range(len(kl)):
      if kl[i].single:
        n_single += 1
      else:
        n_double += 1
    return n_single, n_double

  n_single, n_double = check_bool_field(kernel_list)
  expected_single = sum(1 for k in KERNEL_DB if k.single)
  expected_double = sum(1 for k in KERNEL_DB if not k.single)
  assert n_single == expected_single
  assert n_double == expected_double
  print(f"OK: single={n_single}, double={n_double}")


def test_typed_list_int_for_bool():
  """Fallback: use int instead of bool if bool causes issues."""
  from typing import NamedTuple

  class KernelParamsInt(NamedTuple):
    support: int
    oversampling: float
    epsilon: float
    beta: float
    e0: float
    ndim: int
    single: int  # 0 or 1 instead of bool

  lst = List()
  for k in KERNEL_DB:
    lst.append(
      KernelParamsInt(
        k.support,
        k.oversampling,
        k.epsilon,
        k.beta,
        k.e0,
        k.ndim,
        int(k.single),
      )
    )

  @njit
  def count_entries(kl):
    return len(kl)

  assert count_entries(lst) == len(KERNEL_DB)
  print(f"OK: int-for-bool list has {count_entries(lst)} entries")


@pytest.mark.xfail(reason="typed list can't be captured as a free variable")
def test_closure_capture():
  """Test that a typed.List[KernelParams] can be captured in an njit closure."""
  import time

  kernel_list = _make_kernel_list()

  t0 = time.perf_counter()

  @njit
  def find_best_closure(target_epsilon, ndim, single):
    # kernel_list is captured from the enclosing scope, not passed as arg
    best_idx = -1
    best_oversampling = 1e30
    for i in range(len(kernel_list)):
      k = kernel_list[i]
      if (
        k.ndim == ndim
        and k.single == single
        and k.epsilon <= target_epsilon
        and k.oversampling < best_oversampling
      ):
        best_idx = i
        best_oversampling = k.oversampling
    return best_idx

  # First call: triggers compilation
  idx = find_best_closure(1e-4, 2, False)
  t1 = time.perf_counter()
  compile_time = t1 - t0

  assert idx >= 0, "Should find at least one matching kernel"
  k = KERNEL_DB[idx]
  assert k.ndim == 2
  assert k.single is False
  assert k.epsilon <= 1e-4

  # Second call: already compiled, measure runtime
  t2 = time.perf_counter()
  idx2 = find_best_closure(1e-4, 2, False)
  t3 = time.perf_counter()
  run_time = t3 - t2

  assert idx2 == idx
  print(
    f"OK closure capture: index={idx}, "
    f"compile={compile_time:.3f}s, "
    f"run={run_time*1e6:.1f}us"
  )


@pytest.mark.xfail(reason="typed list can't be captured as a free variable")
def test_closure_capture_multiple_calls():
  """Verify the closure doesn't re-compile on repeated calls with different args."""
  import time

  kernel_list = _make_kernel_list()

  @njit
  def search_closure(target_epsilon, ndim, single):
    best_idx = -1
    best_oversampling = 1e30
    for i in range(len(kernel_list)):
      k = kernel_list[i]
      if (
        k.ndim == ndim
        and k.single == single
        and k.epsilon <= target_epsilon
        and k.oversampling < best_oversampling
      ):
        best_idx = i
        best_oversampling = k.oversampling
    return best_idx

  # Warm up
  search_closure(1e-4, 2, False)

  # Time several calls with different parameters
  times = []
  params = [
    (1e-2, 1, True),
    (1e-4, 2, False),
    (1e-6, 3, False),
    (1e-3, 2, True),
  ]
  for eps, nd, s in params:
    t0 = time.perf_counter()
    search_closure(eps, nd, s)
    times.append(time.perf_counter() - t0)

  avg_us = sum(times) / len(times) * 1e6
  print(f"OK multiple calls: avg={avg_us:.1f}us over {len(params)} calls")


def test_closure_capture_python_list():
  """Test capturing the plain Python KERNEL_DB tuple as a closure freevar."""
  import time

  # KERNEL_DB is a plain tuple of NamedTuples — not a typed.List
  kernel_db = KERNEL_DB

  t0 = time.perf_counter()

  @njit
  def find_best_freevar(target_epsilon, ndim, single):
    best_idx = -1
    best_oversampling = 1e30
    for i in range(len(kernel_db)):
      k = kernel_db[i]
      if (
        k.ndim == ndim
        and k.single == single
        and k.epsilon <= target_epsilon
        and k.oversampling < best_oversampling
      ):
        best_idx = i
        best_oversampling = k.oversampling
    return best_idx

  idx = find_best_freevar(1e-4, 2, False)
  t1 = time.perf_counter()
  compile_time = t1 - t0

  assert idx >= 0, "Should find at least one matching kernel"
  k = KERNEL_DB[idx]
  assert k.ndim == 2
  assert k.single is False
  assert k.epsilon <= 1e-4

  # Second call: already compiled
  t2 = time.perf_counter()
  idx2 = find_best_freevar(1e-4, 2, False)
  t3 = time.perf_counter()
  run_time = t3 - t2

  assert idx2 == idx
  print(
    f"OK python list freevar: index={idx}, "
    f"compile={compile_time:.3f}s, "
    f"run={run_time*1e6:.1f}us"
  )


# ---------------------------------------------------------------------------
# Intrinsic approach: build typed.List at compile time via objmode,
# then use it in nopython code.
# ---------------------------------------------------------------------------
from numba.core import types as nb_types
from numba.extending import intrinsic
from numba.typed import List as TypedList


def _build_kernel_typed_list():
  """Python helper — called from objmode to build the list once."""
  lst = TypedList()
  for k in KERNEL_DB:
    lst.append(k)
  return lst


# Cache so we only build once
_CACHED_KERNEL_LIST = None


def _get_kernel_list():
  global _CACHED_KERNEL_LIST
  if _CACHED_KERNEL_LIST is None:
    _CACHED_KERNEL_LIST = _build_kernel_typed_list()
  return _CACHED_KERNEL_LIST


# Determine the numba type for the list
_kernel_params_type = numba.typeof(KernelParams(0, 0.0, 0.0, 0.0, 0.0, 0, False))
_kernel_list_type = nb_types.ListType(_kernel_params_type)


@intrinsic
def _get_kernel_db(typingctx):
  """Intrinsic that unboxes the cached typed.List[KernelParams] into nopython."""
  sig = _kernel_list_type()

  def codegen(context, builder, signature, args):
    pyapi = context.get_python_api(builder)
    gil_state = pyapi.gil_ensure()

    # Serialize the callable into the LLVM module, then call it at runtime
    fn_obj = pyapi.unserialize(pyapi.serialize_object(_get_kernel_list))
    empty_tuple = pyapi.tuple_new(0)
    py_list = pyapi.call_function_objargs(fn_obj, ())

    # Unbox into native typed list
    native = pyapi.to_native_value(signature.return_type, py_list)

    pyapi.decref(fn_obj)
    pyapi.decref(empty_tuple)
    pyapi.decref(py_list)
    pyapi.gil_release(gil_state)

    return native.value

  return sig, codegen


def test_intrinsic_kernel_db():
  """Test the intrinsic that builds/returns typed.List inside njit."""
  import time

  t0 = time.perf_counter()

  @njit
  def find_best_intrinsic(target_epsilon, ndim, single):
    kernel_list = _get_kernel_db()
    best_idx = -1
    best_oversampling = 1e30
    for i in range(len(kernel_list)):
      k = kernel_list[i]
      if (
        k.ndim == ndim
        and k.single == single
        and k.epsilon <= target_epsilon
        and k.oversampling < best_oversampling
      ):
        best_idx = i
        best_oversampling = k.oversampling
    return best_idx

  idx = find_best_intrinsic(1e-4, 2, False)
  t1 = time.perf_counter()
  compile_time = t1 - t0

  assert idx >= 0
  k = KERNEL_DB[idx]
  assert k.ndim == 2
  assert k.single is False
  assert k.epsilon <= 1e-4

  # Second call
  t2 = time.perf_counter()
  idx2 = find_best_intrinsic(1e-4, 2, False)
  t3 = time.perf_counter()
  run_time = t3 - t2

  assert idx2 == idx
  print(
    f"OK intrinsic: index={idx}, "
    f"compile={compile_time:.3f}s, "
    f"run={run_time*1e6:.1f}us"
  )


# ---------------------------------------------------------------------------
# objmode approach: use numba.objmode() to call back into Python and
# retrieve the cached typed.List, then continue in nopython.
# ---------------------------------------------------------------------------


def test_objmode_kernel_db():
  """Test using numba.objmode to fetch typed.List[KernelParams] inline."""
  import time

  t0 = time.perf_counter()

  @njit
  def find_best_objmode(target_epsilon, ndim, single):
    with numba.objmode(kernel_list=_kernel_list_type):
      kernel_list = _get_kernel_list()

    best_idx = -1
    best_oversampling = 1e30
    for i in range(len(kernel_list)):
      k = kernel_list[i]
      if (
        k.ndim == ndim
        and k.single == single
        and k.epsilon <= target_epsilon
        and k.oversampling < best_oversampling
      ):
        best_idx = i
        best_oversampling = k.oversampling
    return best_idx

  idx = find_best_objmode(1e-4, 2, False)
  t1 = time.perf_counter()
  compile_time = t1 - t0

  assert idx >= 0
  k = KERNEL_DB[idx]
  assert k.ndim == 2
  assert k.single is False
  assert k.epsilon <= 1e-4

  # Second call
  t2 = time.perf_counter()
  idx2 = find_best_objmode(1e-4, 2, False)
  t3 = time.perf_counter()
  run_time = t3 - t2

  assert idx2 == idx
  print(
    f"OK objmode: index={idx}, "
    f"compile={compile_time:.3f}s, "
    f"run={run_time*1e6:.1f}us"
  )


# ---------------------------------------------------------------------------
# @register_jitable approach: wrap the objmode fetch in a reusable helper
# that other njit / @register_jitable code can call directly.
# ---------------------------------------------------------------------------
from numba.extending import register_jitable


@register_jitable
def get_kernel_db():
  """Return the cached typed.List[KernelParams] — callable from njit code."""
  with numba.objmode(kernel_list=_kernel_list_type):
    kernel_list = _get_kernel_list()
  return kernel_list


def test_register_jitable_kernel_db():
  """Test @register_jitable helper that returns typed.List[KernelParams]."""
  import time

  t0 = time.perf_counter()

  @njit
  def find_best_jitable(target_epsilon, ndim, single):
    kernel_list = get_kernel_db()
    best_idx = -1
    best_oversampling = 1e30
    for i in range(len(kernel_list)):
      k = kernel_list[i]
      if (
        k.ndim == ndim
        and k.single == single
        and k.epsilon <= target_epsilon
        and k.oversampling < best_oversampling
      ):
        best_idx = i
        best_oversampling = k.oversampling
    return best_idx

  idx = find_best_jitable(1e-4, 2, False)
  t1 = time.perf_counter()
  compile_time = t1 - t0

  assert idx >= 0
  k = KERNEL_DB[idx]
  assert k.ndim == 2
  assert k.single is False
  assert k.epsilon <= 1e-4

  # Second call
  t2 = time.perf_counter()
  idx2 = find_best_jitable(1e-4, 2, False)
  t3 = time.perf_counter()
  run_time = t3 - t2

  assert idx2 == idx
  print(
    f"OK register_jitable: index={idx}, "
    f"compile={compile_time:.3f}s, "
    f"run={run_time*1e6:.1f}us"
  )


def test_register_jitable_called_from_jitable():
  """Verify get_kernel_db() composes — callable from another @register_jitable."""
  import time

  @register_jitable
  def count_matching(target_epsilon, ndim, single):
    kernel_list = get_kernel_db()
    count = 0
    for i in range(len(kernel_list)):
      k = kernel_list[i]
      if k.ndim == ndim and k.single == single and k.epsilon <= target_epsilon:
        count += 1
    return count

  @njit
  def driver(target_epsilon, ndim, single):
    return count_matching(target_epsilon, ndim, single)

  t0 = time.perf_counter()
  n = driver(1e-4, 2, False)
  t1 = time.perf_counter()

  expected = sum(
    1 for k in KERNEL_DB if k.ndim == 2 and not k.single and k.epsilon <= 1e-4
  )
  assert n == expected
  print(f"OK jitable-from-jitable: {n} matching kernels, " f"compile={t1-t0:.3f}s")


# ---------------------------------------------------------------------------
# Fully-expanded @register_jitable: every KERNEL_DB entry is a literal
# KernelParams() construction inside a pure nopython function.
# ---------------------------------------------------------------------------


def _generate_kernel_db_source():
  """Generate Python source for a @register_jitable that builds the full list."""
  lines = [
    "from numba.extending import register_jitable",
    "from numba.typed import List as TypedList",
    "from radiomesh.generated._es_kernel_params import KernelParams",
    "",
    "@register_jitable",
    "def make_kernel_db():",
    "    lst = TypedList()",
  ]
  for k in KERNEL_DB:
    lines.append(
      f"    lst.append(KernelParams("
      f"{k.support}, {k.oversampling!r}, {k.epsilon!r}, "
      f"{k.beta!r}, {k.e0!r}, {k.ndim}, {k.single}))"
    )
  lines.append("    return lst")
  return "\n".join(lines)


def test_expanded_register_jitable():
  """Test a @register_jitable with all ~1458 KernelParams expanded as literals."""
  import time

  # Generate and exec the function
  src = _generate_kernel_db_source()
  ns = {}
  exec(src, ns)
  make_kernel_db = ns["make_kernel_db"]

  print(f"Generated function: {len(KERNEL_DB)} entries, " f"{len(src)} chars of source")

  t0 = time.perf_counter()

  @njit
  def find_best_expanded(target_epsilon, ndim, single):
    kernel_list = make_kernel_db()
    best_idx = -1
    best_oversampling = 1e30
    for i in range(len(kernel_list)):
      k = kernel_list[i]
      if (
        k.ndim == ndim
        and k.single == single
        and k.epsilon <= target_epsilon
        and k.oversampling < best_oversampling
      ):
        best_idx = i
        best_oversampling = k.oversampling
    return best_idx

  idx = find_best_expanded(1e-4, 2, False)
  t1 = time.perf_counter()
  compile_time = t1 - t0

  assert idx >= 0
  k = KERNEL_DB[idx]
  assert k.ndim == 2
  assert k.single is False
  assert k.epsilon <= 1e-4

  # Second call
  t2 = time.perf_counter()
  idx2 = find_best_expanded(1e-4, 2, False)
  t3 = time.perf_counter()
  run_time = t3 - t2

  assert idx2 == idx
  print(
    f"OK expanded jitable: index={idx}, "
    f"compile={compile_time:.3f}s, "
    f"run={run_time*1e6:.1f}us"
  )


if __name__ == "__main__":
  test_typed_list_of_namedtuple()
  test_typed_list_bool_field()
  test_typed_list_int_for_bool()
  # test_closure_capture()  # known to fail — typed.List as freevar
  # test_closure_capture_multiple_calls()  # same
  test_closure_capture_python_list()
  test_intrinsic_kernel_db()
  test_objmode_kernel_db()
  test_register_jitable_kernel_db()
  test_register_jitable_called_from_jitable()
  test_expanded_register_jitable()
  print("\nAll tests passed!")
