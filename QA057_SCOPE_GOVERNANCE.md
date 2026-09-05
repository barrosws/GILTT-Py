# QA-057 scope governance — operational reproduction documentation closure

QA-057 materializes operational documentation and a single machine-readable QA replay entrypoint. It is derived from QA-056 HOLDs but is deliberately narrower than manuscript/archival reproduction.

The canonical QA replay command is:

`python -m gilttpy.engineering.operational_reproduction --project-root .`

It must cover every governed QA test file from QA-029 through QA-057 exactly once, enforce the frozen single-thread/hash environment, verify partition collection counts before execution, fail closed on missing/extra/duplicated test-file coverage, and emit machine-readable evidence when requested.

QA-057 also materializes an executable installation quickstart, reproducibility instructions, a development CHANGELOG, development release notes, a machine-readable QA replay TOML plan, and automated code-coverage evidence focused on the new QA-057 operational layer. Full-package coverage remains an explicit HOLD because instrumenting the complete scientific suite is a distinct release-evidence task. Coverage is engineering evidence only; no coverage percentage may be interpreted as scientific validation.

QA-057 does **not** claim to reproduce manuscript tables or figures. Complete machine-readable configuration coverage for all final scientific runs remains HOLD until those runs are frozen. Complete data/source provenance and redistribution-rights adjudication remains HOLD. Software authorship, ORCIDs, license, DOI, public release version/date and Zenodo deposition remain under prior/future release governance and may not be inferred here.

No scientific equation, physical closure, numerical tolerance, sensitivity/UQ design, model-form conclusion, historical branch, QA-045 claim envelope or NO TARGET TUNING rule is modified.
