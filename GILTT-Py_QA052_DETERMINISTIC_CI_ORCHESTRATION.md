# GILTT-Py QA-052 — Deterministic CI Orchestration Specification

**Date:** 2026-09-03  
**Canonical gate:** `PASS_DETERMINISTIC_CI_ORCHESTRATION_SPECIFICATION`

## Scope

QA-052 freezes a provider-neutral CI evidence contract. It changes no scientific equation, physical closure, numerical tolerance, sensitivity/UQ design, model-form comparison, or historical branch. A CI matrix cell may be promoted only after the ordered evidence stages `metadata_contract -> distribution_build -> installed_artifact_import -> partitioned_test_replay -> artifact_integrity -> evidence_record` all pass. Unexecuted cells remain `CI_REQUIRED`.

## Deterministic test partition contract

The governed QA029-QA052 suite contains exactly 324 tests partitioned as 210 + 30 + 12 + 12 + 12 + 12 + 12 + 12 + 12. A partial partition set, a failed test, unexpected test count, source-tree import or artifact-hash mismatch blocks a cell PASS.

Canonical source-tree replay after staging correction: **324/324 PASS**.
Canonical installed-wheel replay on Linux x86-64 / CPython 3.13.5 with the QA-051 reference stack: **324/324 PASS**. The imported package path is outside the source tree.

## Build evidence and release-CI boundary

Two independent local wheel builds using `python -m pip wheel --no-deps --no-build-isolation .` with fixed `SOURCE_DATE_EPOCH` are bitwise identical. Wheel SHA-256: `902bb821922bd8b6fc488a75c8e96e97923417bd608a058e0a89640bec251e76`; size: 128938 bytes.

The canonical release-CI build command is specified as `python -m build`, producing both sdist and wheel. That frontend is not installed in this executor and was not silently substituted for purposes of closing the release-CI stage. Therefore `HOLD_STANDARD_PYPA_BUILD_FRONTEND_LOCAL_EXECUTION` remains active, and the local matrix cell is not promoted under the *full* QA-052 stage chain merely because the installed-wheel replay passed.

## Staging correction

The first aggregate staging attempt omitted the inherited `QA051_SCOPE_GOVERNANCE.md`, causing QA051 to return 11/12 by `FileNotFoundError`. The artifact was restored byte-identically from QA-051; no scientific source or test assertion changed. Canonical replay then passed. The preliminary log is preserved.

## Inheritance

QA-052 inherits **86/86 Python files** from QA-051 byte-identically (56 package sources and 30 tests). New Python files are only the CI orchestration module, validation campaign and QA-052 test.

## Critical hashes

- `ci_orchestration.py`: `763422ad4526b5f20eced6c3dc1e0f7ec0a394fa2b418a1691c7ce58d2ebba95`
- `ci_orchestration_campaign.py`: `49acae2832ce75c27e4678a4fdd39838b683fa2f46ae89c100e20ceb95ed7d1a`
- `test_qa052_ci_orchestration.py`: `d9be2bf4a7b10e27e4b6532447910d0df97176eb70a669fba54f0822a1718128`
- `qa052_ci_contract.json`: `63be45f2ad96c06c95e86a93aed37bad880e90645d9acd6715b35fe3c8443de3`
- wheel: `902bb821922bd8b6fc488a75c8e96e97923417bd608a058e0a89640bec251e76`

## Holds

- `HOLD_EXTERNAL_CI_MATRIX_EXECUTION`
- `HOLD_HERMETIC_HASH_LOCKED_DEPENDENCY_RESOLUTION`
- `HOLD_MINIMUM_DEPENDENCY_COMPATIBILITY_EXECUTION`
- `HOLD_STANDARD_PYPA_BUILD_FRONTEND_LOCAL_EXECUTION`
- `HOLD_RELEASE_CI_PROVIDER_ACTIVATION`
- `HOLD_PUBLIC_RELEASE_PUBLISHING`

## Prohibitions

- `PROHIBIT_PARTIAL_PARTITION_SET_AS_FULL_PASS`
- `PROHIBIT_SOURCE_TREE_IMPORT_IN_INSTALLED_ARTIFACT_REPLAY`
- `PROHIBIT_CROSS_PLATFORM_WHEEL_BYTE_IDENTITY_ASSUMPTION`
- `PROHIBIT_UNEXECUTED_MATRIX_CELL_AS_PASS`
- `PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION`
- `PROHIBIT_TARGET_TUNING`

## Gate decision

`PASS_DETERMINISTIC_CI_ORCHESTRATION_SPECIFICATION`

This gate validates the deterministic CI **specification and local installed-artifact replay**, not unexecuted cross-platform cells and not public release readiness.
