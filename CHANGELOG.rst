:tocdepth: 1

Changelog
=========

X.Y.Z (DD-MM-YYYY)
------------------
* Pre-generate expressions for converting visibilities, weights and gains to stokes parameters (:pr:`39`)
* Treat warnings as errors in py.test (:pr:`40`)
* Use ``tarfile.extractall(.., filter="data")`` (:pr:`40`)
* Ignore xarray-ms ImputedMetadataWarnings in test cases (:pr:`40`)
* Speed up DFT tests by discarding channels and imaging at a lower frequency (:pr:`38`)
* Introduce a literal WGridderParameters argumen to to the numba wgridder (:pr:`36`)
* Remove comment describing threads from a previous attempt at parallelisation of explicit_gridde (:pr:`34`)
* Add full reference gridding and wgridding implementations (:pr:`32`)
* Remove 0.5 offset from the ES kernel evaluation (:pr:`31`)
* Parametrise eval_es_kernel kernel parameter (``DatumLiteral[ESKernel]``) (:pr:`30`)
* Fix ``pol_to_stokes`` exception message (:pr:`29`)
* Datum typing updates (:pr:`28`)
* Apply ``prefer_literal=True`` to overloads and intrinsics (:pr:`27`)
* Substitute use of Literals for factory functions in numba overloads and intrinsics
  to ensure numba cache hits (:pr:`26`)
* Remove FloatLiteral type (:pr:`25`)
* Remove compound literal tests (:pr:`25`)
* Add a DatumLiteral type (:pr:`24`)
* Add an intrinsic caching test case (:pr:`22`)
* Avoid hard-coding types in the kernel positions intrinsic (:pr:`21`)
* Add a compound literal test case (:pr:`19`, :pr:`20`)
* Apply flags in gridding kernel (:pr:`18`)
* Refine FloatLiteral implementation (:pr:`17`)
* Move kernel functionality into ESKernel class (:pr:`16`)
* Move gridder argument checks into a separate function (:pr:`15`)
* Align ducc0 and numba wgridder parameters (:pr:`14`)
* Return 0 for values where the ES kernel is undefined (:pr:`13`)
* Fix construction of U from LR and RL (:pr:`12`)
* Rename KERNEL_POSITION to KERNEL_OFFSET (:pr:`11`)
* Add Github Action Issue and Pull Request templates (:pr:`10`)
* Add changelog (:pr:`10`)
* Incorporate wgridder_conventions (:pr:`9`)
* Fix zeroing es kernels outside [-1.0, 0.0] (:pr:`8`)
* Remove scipy dependency (:pr:`6`)
* Test intrinsics (:pr:`5`)

0.1.0 (04-07-2025)
------------------

* Initial release
