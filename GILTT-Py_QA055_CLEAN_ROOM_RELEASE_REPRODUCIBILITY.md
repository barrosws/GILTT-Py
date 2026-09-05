# GILTT-Py QA-055 — Clean-room release reproducibility

Candidate gate: `PASS_CLEAN_ROOM_RELEASE_REPRODUCIBILITY_CONTRACT`.

## Scope

QA-055 is an engineering/reproducibility gate only. It changes no frozen scientific equation, physics closure, numerical tolerance, uncertainty design, model-form comparison, historical benchmark lineage, or NO TARGET TUNING rule.

## Inheritance

All 95 Python files inherited from QA-054 are byte-identical. QA-055 adds exactly three Python files: `engineering/clean_room.py`, `validation/clean_room_campaign.py`, and `test_qa055_clean_room_release_reproducibility.py`.

## Clean-room contract

The required evidence stages are source integrity, distribution build, artifact integrity, isolated install, installed-import provenance, partitioned test replay, runtime provenance, and evidence record. Every stage and every frozen test partition must pass at its exact expected count.

A local clean-room PASS is evidence only for the executed Linux x86-64 / CPython 3.13.5 cell. It does not promote the other QA-051 matrix cells, prove a cross-platform hermetic lock, or close minimum-supported-dependency execution.

## Build evidence

The canonical release-CI command remains `python -m build`, but the local executor still lacks the PyPA `build` frontend. The local deterministic fallback therefore remains engineering evidence only. Two independently built wheels under frozen reproducibility controls are bitwise identical. Two normalized sdists are bitwise identical, and rebuilding from the normalized sdist reproduces the canonical wheel exactly.

## Test evidence

QA-055 dedicated tests: 12/12 PASS.

Complete source-tree regression: 360/360 PASS.

Fresh external installed-wheel clean-room replay: 360/360 PASS. `gilttpy` imported from `/mnt/data/qa055_clean_room_site/gilttpy/__init__.py`, outside the governed source tree.

Operational partitioning of QA-047/QA-048 was used only to stay inside external execution windows. Test definitions, package bytes, scientific parameters and tolerances were unchanged.

## Release-metadata status

QA-054 HOLDs remain binding. `CITATION.cff` is not final, public authorship/ORCIDs are not inferred, an SPDX license is not selected here, data redistribution rights are not asserted, no DOI or release date is invented, and `0.0.0.dev1` is not promoted to a public semantic release.

## Gate status before immutable-ZIP replay

Local source/build/install evidence satisfies the candidate QA-055 contract. Formal freeze requires deterministic final ZIP creation, internal checksum verification, exact-ZIP fresh install/replay, Drive readback and append-only governance registration.
