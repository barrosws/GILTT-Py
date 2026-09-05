# QA-055 clean-room release reproducibility — scope governance

Canonical candidate gate: `PASS_CLEAN_ROOM_RELEASE_REPRODUCIBILITY_CONTRACT`.

QA-055 changes no scientific equation, physical closure, numerical tolerance, sensitivity/UQ design, model-form result, historical benchmark lineage, or target-selection rule. It strengthens only engineering evidence for reproducibility of the locally executed reference environment.

A clean-room PASS requires all of the following on one immutable snapshot: source-integrity verification; deterministic distribution build evidence; SHA-256 artifact integrity; installation into a fresh location outside the governed project/source tree; proof that `gilttpy` imports from that installed location; complete partitioned replay at exact expected counts; runtime provenance; and an immutable evidence record.

A local clean-room PASS is not cross-platform evidence. The 11 unexecuted QA-051 platform/Python matrix cells remain `CI_REQUIRED`. QA-053 local dependency resolution remains non-hermetic and non-universal. Minimum-supported dependency execution remains a HOLD.

The canonical release-CI build command remains `python -m build` with both sdist and wheel. If the local executor lacks the PyPA `build` frontend, the existing deterministic `pip wheel --no-deps --no-build-isolation` plus normalized-sdist route may provide local engineering evidence only; it does not close the standard-build-frontend HOLD.

QA-054 metadata HOLDs remain binding. The presence of a successful clean-room replay must not promote placeholder citation metadata, choose software authors, infer ORCIDs, select an SPDX license, assert benchmark-data redistribution rights, mint a DOI, set a release date, or promote `0.0.0.dev1` to a public release version.

NO TARGET TUNING remains frozen.
