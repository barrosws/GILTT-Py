"""QA-051 environment/dependency/CI matrix governance.

This module separates *declared compatibility targets* from *executed evidence*.
A matrix cell is never promoted to tested merely because package metadata or an
upstream dependency claims support for that interpreter/platform.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import platform
import sys
from typing import Iterable

QA051_GATE = "PASS_REPRODUCIBLE_ENVIRONMENT_AND_CI_MATRIX_SPECIFICATION"
SUPPORTED_PYTHON_SERIES = ("3.11", "3.12", "3.13", "3.14")
SUPPORTED_PLATFORM_FAMILIES = ("linux", "macos", "windows")
REQUIRES_PYTHON = ">=3.11,<3.15"

QA051_HOLDS = (
    "HOLD_CROSS_PLATFORM_EXECUTION_EVIDENCE_TO_EXTERNAL_CI",
    "HOLD_PYTHON_311_312_314_EXECUTION_EVIDENCE_TO_EXTERNAL_CI",
    "HOLD_HERMETIC_HASH_LOCKED_DEPENDENCY_RESOLUTION",
    "HOLD_MINIMUM_DEPENDENCY_COMPATIBILITY_EXECUTION",
    "HOLD_STANDARD_PYPA_BUILD_FRONTEND_EXECUTION_TO_RELEASE_CI",
    "HOLD_RELEASE_CI_AUTOMATION",
)
QA051_PROHIBITIONS = (
    "PROHIBIT_UNEXECUTED_MATRIX_CELL_AS_PASS",
    "PROHIBIT_UPSTREAM_SUPPORT_METADATA_AS_GILTTPY_TEST_EVIDENCE",
    "PROHIBIT_REFERENCE_PINS_AS_CROSS_PLATFORM_HERMETIC_LOCK",
    "PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_TARGET_TUNING",
)

class MatrixEvidenceStatus(str, Enum):
    LOCAL_TESTED = "LOCAL_TESTED"
    CI_REQUIRED = "CI_REQUIRED"

@dataclass(frozen=True)
class CompatibilityTarget:
    platform_family: str
    python_series: str

    def __post_init__(self) -> None:
        if self.platform_family not in SUPPORTED_PLATFORM_FAMILIES:
            raise ValueError("unsupported platform family")
        if self.python_series not in SUPPORTED_PYTHON_SERIES:
            raise ValueError("unsupported Python series")

@dataclass(frozen=True)
class MatrixEvidence:
    target: CompatibilityTarget
    status: MatrixEvidenceStatus
    observed_python: str | None = None
    observed_platform: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["status"] = self.status.value
        return out


def declared_matrix() -> tuple[CompatibilityTarget, ...]:
    return tuple(
        CompatibilityTarget(os_name, py)
        for os_name in SUPPORTED_PLATFORM_FAMILIES
        for py in SUPPORTED_PYTHON_SERIES
    )


def runtime_platform_family(system: str | None = None) -> str:
    s = (platform.system() if system is None else system).strip().lower()
    if s == "linux": return "linux"
    if s == "darwin": return "macos"
    if s == "windows": return "windows"
    raise RuntimeError(f"runtime platform is outside QA051 matrix: {s!r}")


def runtime_python_series(version_info=None) -> str:
    v = sys.version_info if version_info is None else version_info
    return f"{int(v.major)}.{int(v.minor)}"


def evidence_matrix(
    *,
    platform_family: str | None = None,
    python_series: str | None = None,
    observed_python: str | None = None,
    observed_platform: str | None = None,
) -> tuple[MatrixEvidence, ...]:
    pf = runtime_platform_family() if platform_family is None else platform_family
    py = runtime_python_series() if python_series is None else python_series
    current = CompatibilityTarget(pf, py)
    targets = declared_matrix()
    if current not in targets:
        raise RuntimeError("current runtime is outside the declared QA051 matrix")
    return tuple(
        MatrixEvidence(
            target=t,
            status=(MatrixEvidenceStatus.LOCAL_TESTED if t == current else MatrixEvidenceStatus.CI_REQUIRED),
            observed_python=(observed_python if t == current else None),
            observed_platform=(observed_platform if t == current else None),
            note=(
                "executed in the QA051 local runner; does not imply another OS/Python cell"
                if t == current else
                "requires execution in an independent CI runner before PASS"
            ),
        )
        for t in targets
    )


def assert_no_false_pass(evidence: Iterable[MatrixEvidence]) -> None:
    items = tuple(evidence)
    tested = [e for e in items if e.status is MatrixEvidenceStatus.LOCAL_TESTED]
    if len(tested) != 1:
        raise ValueError("QA051 local evidence must identify exactly one executed cell")
    for e in items:
        if e.status not in (MatrixEvidenceStatus.LOCAL_TESTED, MatrixEvidenceStatus.CI_REQUIRED):
            raise ValueError("unknown matrix evidence status")
