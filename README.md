# GILTT-Py 2.0

GILTT-Py 2.0 is a research software implementation of a verified spectral-Laplace framework for atmospheric advection-diffusion, deposition, gravitational settling, numerical verification, sensitivity analysis and uncertainty analysis.

The repository distinguishes two branches conceptually:

- the historical GILTT reconstruction/provenance track, retained for reproducibility and computational archaeology;
- the modern GILTT-Py 2.0 track, developed under explicit mathematical, physical, numerical and software QA gates.

The active governed package tree is `01_SRC/gilttpy`; tests are in `02_TESTS`. Scientific conclusions are restricted by the frozen validation claim envelope. Historical or observational concentration targets are not permitted to tune numerical methods, uncertainty distributions, sensitivity designs or model-form choices.

## Development status

Release candidate version: `2.0.0` (`v2.0.0`). The source-code license decision is BSD-3-Clause. Software authorship/ORCID metadata, the live repository record, external CI evidence and the Zenodo DOI remain controlled release-promotion fields and are not inferred or fabricated here.

## Runtime dependencies

The current runtime contract declares NumPy, SciPy and mpmath. Testing utilities are separate optional dependencies.

## Reproducibility

QA artifacts record source hashes, package hashes, deterministic seeds, test partitions, numerical limitations and explicit HOLD/PROHIBIT statements. A passing QA checkpoint does not extend claims beyond the tested domain.

## Compatibility evidence policy

The current declared compatibility target is CPython 3.11–3.14 (`>=3.11,<3.15`). Platform/Python combinations are treated as testable CI targets, not as passed environments merely because package metadata or upstream dependencies report support. QA-051 records the locally executed environment separately from cells that still require independent CI execution.

## Operational documentation

The QA-057 development snapshot adds an executable installation quickstart and a single governed QA-replay command. See `docs/INSTALLATION_QUICKSTART.md` and `docs/REPRODUCIBILITY.md`. Development change history is recorded in `docs/CHANGELOG.md`; these materials do not constitute a public release.
