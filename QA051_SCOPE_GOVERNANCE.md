# QA-051 scope governance

QA-051 is an engineering/reproducibility gate. It does not modify or revalidate scientific equations, physics, numerical tolerances, sensitivity designs, uncertainty distributions, or model-form conclusions.

The declared compatibility target is CPython 3.11–3.14 on Linux, macOS, and Windows. A declared cell is not test evidence. In the present executor only Linux x86-64 / CPython 3.13.5 is executable; all other cells remain `CI_REQUIRED`.

The exact direct/test versions recorded for the local Python-3.13/Linux cell are a reference environment record, not a hermetic cross-platform lock. A hermetic claim requires platform-specific resolved artifacts (including transitive dependencies and hashes) and clean replay without inheriting host packages.

The package metadata is bounded to `requires-python >=3.11,<3.15` so that untested future Python series are not silently claimed. Cross-platform execution, minimum-dependency execution, hash-locked dependency resolution, standard PyPA `python -m build`, and release CI automation remain separate HOLDs.

NO TARGET TUNING and the QA-045 scientific claim envelope remain unchanged.
