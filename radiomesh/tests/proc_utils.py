import os
import sys


def _init_numba_cache_debugging_with_capture(cache_dir, stdout_path, stderr_path):
  assert "numba" not in sys.modules
  os.environ["NUMBA_CACHE_DIR"] = cache_dir
  os.environ["NUMBA_DEBUG_CACHE"] = "1"

  sys.stdout = open(stdout_path, "a", buffering=1)
  sys.stderr = open(stderr_path, "a", buffering=1)
