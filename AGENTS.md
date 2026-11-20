# radiomesh: a numba w-stacking gridder

[ratt-ru/radiomesh](https://github.com/ratt-ru/radiomesh) is a numba-accelerated
convolutional gridder for radio interferometry, implementing the improved
w-stacking method (see the [README](README.md) for the papers). It is validated
against DUCC's wgridder. Correctness and numerical accuracy come first, and
performance is a hard requirement, not a nice-to-have.

## Hard constraints

- All hot-path code must compile in nopython mode (`@njit`). Never fall back to
  object mode; a forced object-mode fallback is a bug.
- numpy >= 2.0 only: no removed aliases (`np.float_`, `np.int0`), and respect 2.0
  copy/view semantics.
- No Python-level loops over visibilities in hot paths.
- Don't regress performance. Run the `pytest-benchmark` cases in
  [`test_benchmark_es_kernel.py`](radiomesh/tests/test_benchmark_es_kernel.py)
  before and after a kernel change and compare.
- Don't regress accuracy. Kernel tests compare against reference results with
  `np.testing.assert_allclose` (see [`test_wgrid.py`](radiomesh/tests/test_wgrid.py)),
  and [`cmp_vs_ducc.py`](radiomesh/tests/cmp_vs_ducc.py) is a script for checking
  the gridder against DUCC's wgridder.

## Numba conventions

- [`core.py`](radiomesh/core.py) and [`gridding.py`](radiomesh/gridding.py) set
  their numba options once, in a module-level `JIT_OPTIONS` dict passed to
  `@njit(**JIT_OPTIONS)`. In a module with such a dict, use it rather than
  adding per-function options.
- Stay inside nopython-supported Python: no mixed-type dicts, dynamic lists, or
  most of the stdlib inside kernels.
- Preallocate outputs, and keep allocations out of inner loops.
- Add a `pytest-benchmark` case for a new kernel.

## Python conventions

- ruff for formatting and linting: **2-space indent**, line length 88, imports
  sorted by the ruff `I` rules (`ruff check --fix --extend-select I`, then
  `ruff format`).
- Targets Python 3.11 to 3.13.
- Fully type public signatures, and document array dtypes and shapes in
  docstrings. Keep units explicit (metres vs wavelengths).
- Don't hand-edit `radiomesh/generated/`. Regenerate it with
  `uv run --group test radiomesh gen-expr` or
  `uv run --group test radiomesh gen-kernel-params`.

## Tests and test data

Tests live in [`radiomesh/tests/`](radiomesh/tests/). The `ms_name` fixture in
[`conftest.py`](radiomesh/tests/conftest.py) downloads a small test Measurement Set
on first use and caches it under the user cache directory (`radiomesh/test-data`),
so the first run needs network access. Other fixtures build synthetic
observations.

## Tooling

Use `uv` to interact with this package.

- Install: `uv sync`
- Test: `uv run --group test pytest radiomesh/tests/`. CI runs
  `pytest --benchmark-disable -Werror`, so a new warning fails CI.
- Pre-commit hooks (ruff, mypy and file hygiene) gate commits; run them before
  proposing a change as done: `uv run --group dev pre-commit run -a`
- Versioning: tbump (never hand-edit version strings): `uv run --group dev tbump <new-version>`.

Update [`CHANGELOG.rst`](CHANGELOG.rst) for user-visible changes.
