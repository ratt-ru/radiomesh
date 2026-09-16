from __future__ import annotations

from typing import Tuple

from llvmlite import binding as llvm


def llvm_features() -> Tuple[str, llvm.FeatureMap]:
  """What numba/LLVM thinks the host is (this is exactly what numba -s prints)."""
  try:
    # llvmlite < 0.46 needs explicit init
    llvm.initialize()
    llvm.initialize_native_target()
  except Exception:
    # newer llvmlite initialises itself
    pass
  return llvm.get_host_cpu_name(), llvm.get_host_cpu_features()


def widest_simd_register_bits() -> int:
  """Widest usable vector register, in bits. LLVM spells them avx512f/sse2."""
  _, features = llvm_features()

  # intel
  if features.get("avx512f"):
    return 512
  if features.get("avx"):
    return 256
  if features.get("sse2"):
    return 128
  # arm
  if features.get("neon") or features.get("asimd"):
    return 128

  return 0
