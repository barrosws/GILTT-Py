"""QA-050 engineering/reproducibility campaign contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from gilttpy.engineering.reproducibility import canonical_runtime_snapshot, environment_fingerprint

QA050_GATE = "PASS_ENGINEERING_REPRODUCIBILITY_BASELINE"
QA050_HOLDS = (
    "HOLD_CROSS_PLATFORM_EQUIVALENCE_MATRIX_TO_QA051",
    "HOLD_HERMETIC_DEPENDENCY_RESOLUTION_TO_QA051",
    "HOLD_RELEASE_CI_AUTOMATION",
    "HOLD_PUBLIC_SEMANTIC_VERSION_AND_PYPI_READINESS",
    "HOLD_VCS_TAG_AND_ARCHIVAL_RELEASE_PROVENANCE",
)
QA050_PROHIBITIONS = (
    "PROHIBIT_ENGINEERING_GATE_AS_NEW_SCIENTIFIC_VALIDATION",
    "PROHIBIT_HOST_DEPENDENCY_INHERITANCE_AS_HERMETIC_LOCK_PROOF",
    "PROHIBIT_NORMALIZED_SDIST_AS_RAW_BACKEND_BYTE_IDENTITY",
    "PROHIBIT_TARGET_TUNING",
)

@dataclass(frozen=True)
class PackagingContract:
    package_dir: str
    test_dir: str
    runtime_dependencies: tuple[str, ...]
    version: str


def load_packaging_contract(pyproject: str | Path) -> PackagingContract:
    with Path(pyproject).open("rb") as handle:
        cfg = tomllib.load(handle)
    package_dir = cfg["tool"]["setuptools"]["package-dir"][""]
    test_dir = cfg["tool"]["pytest"]["ini_options"]["testpaths"][0]
    return PackagingContract(
        package_dir=package_dir,
        test_dir=test_dir,
        runtime_dependencies=tuple(cfg["project"]["dependencies"]),
        version=cfg["project"]["version"],
    )


def qa050_runtime_provenance():
    snapshot = canonical_runtime_snapshot()
    return snapshot, environment_fingerprint(snapshot)
