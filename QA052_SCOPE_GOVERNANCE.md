# QA-052 scope governance

QA-052 is an engineering orchestration gate. It specifies the evidence sequence required for an individual CI matrix cell to be promoted to PASS. It does not execute absent operating systems/Python versions, does not modify scientific equations, and does not extend the QA-045 claim envelope.

A cell can pass only after the full metadata/build/install/import-provenance/test-partition/artifact-integrity/evidence chain passes. Partial partitions, source-tree imports, artifact mismatches, or missing stages fail that cell. An unexecuted cell remains `CI_REQUIRED`.

The canonical release-CI build command is `python -m build`, consistent with current PyPA guidance. The `build` frontend is not installed in the present local executor, so local execution of that exact frontend remains HOLD. QA-052 may validate orchestration using the same deterministic setuptools wheel route already frozen by QA-050/QA-051, but it must not relabel the standard frontend as locally executed.

Cross-platform wheel byte identity is not required or assumed. Scientific equivalence is established by execution of the same governed test suite and evidence contract in each cell, not by assuming identical wheel container bytes across operating systems.

NO TARGET TUNING, historical/modern separation, and all QA-051 hermetic/minimum-dependency HOLDs remain unchanged.
