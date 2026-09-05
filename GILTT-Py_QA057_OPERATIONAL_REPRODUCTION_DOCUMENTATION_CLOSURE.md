# GILTT-Py QA-057 — Operational reproduction documentation closure

Canonical candidate gate: `PASS_OPERATIONAL_REPRODUCTION_DOCUMENTATION_CLOSURE`

## Scope

QA-057 closes the operational documentation layer identified by QA-056: executable installation quickstart, one governed QA-replay command, machine-readable replay plan, reproducibility instructions, development changelog/release notes, sdist inclusion of operational artifacts, and scoped automated coverage evidence. It does not claim manuscript table/figure reproduction, complete scientific-run configuration coverage, full-package coverage, public-release metadata, software-license selection, data redistribution rights, Zenodo readiness, or cross-platform CI completion.

No scientific equation, physical closure, numerical tolerance, sensitivity/UQ design, model-form conclusion, historical branch, QA-045 claim envelope, or NO TARGET TUNING rule was changed.

## Inheritance and dedicated tests

- Governed Python inheritance from QA-056: **101/101 byte-identical**.
- New Python files: exactly 3 (`operational_reproduction.py`, validation campaign, QA-057 test).
- Dedicated QA-057 tests after final `MANIFEST.in`: **12/12 PASS**.
- Governed cumulative collection: **384 tests**.

## One-command reproduction

Canonical command:

`python -m gilttpy.engineering.operational_reproduction --project-root .`

The final source replay after the sdist-manifest correction completed **384/384 PASS**, exit code 0, with exact collection count 384. The replay plan is `reproduction/qa057_qa_replay.toml` and partitions every governed QA-029–QA-057 test file exactly once while enforcing single-thread/hash controls.

The installed-wheel replay imports `gilttpy` from `/mnt/data/qa057_installed_site/gilttpy/__init__.py`, outside the governed source tree, and the same one-command invocation completed **384/384 PASS**, exit code 0.

## Distribution reproducibility

Frozen local controls include `SOURCE_DATE_EPOCH=1788436800`, `PYTHONHASHSEED=0`, and single-thread BLAS/OpenMP execution.

- Two independently built wheels are bitwise identical.
- Canonical wheel SHA-256: `3c1f8b3c0df8830fce9a47ecf2df5ab45282ebdc05356a200904746a71976475`.
- Wheel size: 145821 bytes.
- Raw setuptools sdists retain controlled archive-container timestamp variability and are not represented as byte-identical.
- Two deterministically normalized sdists are bitwise identical.
- Canonical normalized sdist SHA-256: `af4c719184476cfd98010aeaf9fa752c3b174f0776da7b07cd3dc5585b7353a3`.
- Normalized sdist size: 162539 bytes.
- Rebuilding from the normalized sdist reproduces the canonical wheel exactly.
- The sdist contains the QA-057 quickstart, reproducibility document, changelog, development release notes, replay TOML, and QA-057 test.
- `python -m build` remains a local HOLD because the PyPA `build` frontend is not installed in this executor.

## Coverage evidence

Automated coverage was generated only for the new QA-057 operational layer: **54.3%** total statement coverage across the operational entrypoint and validation campaign. This percentage is engineering evidence only. Full-package coverage remains an explicit HOLD and no coverage percentage is used as scientific validation or a scientific promotion threshold.

## Frozen HOLDs

- `HOLD_MANUSCRIPT_TABLE_FIGURE_REPRODUCTION_ENTRYPOINT`
- `HOLD_MACHINE_READABLE_CONFIG_COVERAGE_FOR_ALL_CANONICAL_SCIENTIFIC_RUNS`
- `HOLD_DATA_AND_SOURCE_PROVENANCE_CATALOG_COMPLETENESS`
- `HOLD_FULL_PACKAGE_COVERAGE_REPORT`
- `HOLD_FINAL_PUBLIC_RELEASE_METADATA`
- `HOLD_SOFTWARE_LICENSE_SELECTION`
- `HOLD_ZENODO_ARCHIVAL_RECORD`
- `HOLD_EXTERNAL_CI_MATRIX_EXECUTION`
- `HOLD_STANDARD_PYPA_BUILD_FRONTEND_LOCAL_EXECUTION`

## Prohibitions

QA replay is not manuscript-output reproduction proof; coverage percentage is not scientific validation; unverified data redistribution rights may not be claimed; unresolved metadata may not be promoted to final release metadata; an engineering gate is not scientific validation; target tuning remains prohibited.

## Freeze criterion

The gate may be frozen only after deterministic ZIP sealing, internal SHA-256 verification, exact ZIP extraction, installation of the bundled wheel outside the extracted governed tree, exact one-command replay of all 384 tests, Drive upload/readback of the ZIP and critical active files, and append-only governance registration.
