# GILTT-Py QA-053 — Dependency Lock and Environment Resolution Contract

Gate candidate: `PASS_DEPENDENCY_LOCK_AND_ENVIRONMENT_RESOLUTION_CONTRACT`

QA-053 separates local dependency-resolution evidence from a true hermetic hash lock. A local installed-version snapshot is explicitly classified as `LOCAL_RESOLUTION`; it is neither cross-platform evidence nor a hash-locked environment proof.

## Verified evidence

- QA-053 dedicated tests: 12/12 PASS.
- Complete source regression QA029–QA053: 336/336 PASS.
- Installed-wheel replay outside the governed source tree: 336/336 PASS.
- Inherited Python files from QA-052: 89/89 byte-identical.
- Two independent QA-053 wheel builds are bitwise identical.
- Canonical wheel SHA-256: `6832bf42d3afd64066824eaf8ab7c51795d849bf58b0a58fcd04756e711b35a1`.
- Two raw sdists have identical semantic member content but backend timestamp variability.
- Deterministically normalized sdists are bitwise identical; canonical SHA-256: `f92df9369a4c13ee2c235999caec4a50acc180f1bac2f053c6b7bd24eb9e6c15`.
- Rebuilding the wheel from the normalized sdist produces the exact canonical wheel.
- Local runtime stack: CPython 3.13.5 / Linux x86-64; NumPy 2.3.5; SciPy 1.17.0; mpmath 1.3.0; pytest 9.0.2; threadpoolctl 3.6.0.
- Local resolution fingerprint: `6f97ac7e5448581dd9aea82535ebdab94d63c44c47e3c567e285ea5c03f04f31`.

## Controlled limitations / HOLDs

- cross-platform hash-lock generation remains HOLD;
- offline wheelhouse completeness remains HOLD;
- minimum-supported-dependency execution remains HOLD;
- external-CI lock replay remains HOLD;
- the standard `python -m build` frontend is specified for release CI but is not installed in this executor.

## Prohibitions

- do not treat `pip freeze` as a universal cross-platform lock;
- do not call unhashed requirements hermetic;
- do not promote host-extraneous packages into runtime requirements;
- do not promote an unexecuted lock target to PASS;
- engineering QA is not scientific validation;
- NO TARGET TUNING.

Status inside package: candidate pending exact replay of the final immutable ZIP and remote readback. Promotion to the canonical gate occurs only outside this package after those checks succeed.
