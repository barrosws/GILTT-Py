"""QA-052 deterministic CI orchestration governance.

The contract is provider-neutral.  It defines what evidence an executed CI cell
must produce before that cell can be promoted to PASS; it does not itself claim
execution on operating systems or Python versions absent from the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

QA052_GATE = "PASS_DETERMINISTIC_CI_ORCHESTRATION_SPECIFICATION"
QA052_HOLDS = (
    "HOLD_EXTERNAL_CI_MATRIX_EXECUTION",
    "HOLD_HERMETIC_HASH_LOCKED_DEPENDENCY_RESOLUTION",
    "HOLD_MINIMUM_DEPENDENCY_COMPATIBILITY_EXECUTION",
    "HOLD_STANDARD_PYPA_BUILD_FRONTEND_LOCAL_EXECUTION",
    "HOLD_RELEASE_CI_PROVIDER_ACTIVATION",
    "HOLD_PUBLIC_RELEASE_PUBLISHING",
)
QA052_PROHIBITIONS = (
    "PROHIBIT_PARTIAL_PARTITION_SET_AS_FULL_PASS",
    "PROHIBIT_SOURCE_TREE_IMPORT_IN_INSTALLED_ARTIFACT_REPLAY",
    "PROHIBIT_CROSS_PLATFORM_WHEEL_BYTE_IDENTITY_ASSUMPTION",
    "PROHIBIT_UNEXECUTED_MATRIX_CELL_AS_PASS",
    "PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_TARGET_TUNING",
)

THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

REQUIRED_STAGE_ORDER = (
    "metadata_contract",
    "distribution_build",
    "installed_artifact_import",
    "partitioned_test_replay",
    "artifact_integrity",
    "evidence_record",
)


class CIStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class TestPartition:
    name: str
    qa_start: int
    qa_end: int
    expected_tests: int

    def __post_init__(self) -> None:
        if not self.name or self.qa_start > self.qa_end or self.expected_tests <= 0:
            raise ValueError("invalid CI partition")

    def matching_files(self, test_dir: str | Path) -> tuple[Path, ...]:
        root = Path(test_dir)
        files: list[Path] = []
        for qa in range(self.qa_start, self.qa_end + 1):
            files.extend(sorted(root.glob(f"test_qa{qa:03d}*.py")))
        return tuple(files)


@dataclass(frozen=True)
class StageEvidence:
    stage: str
    status: CIStatus
    note: str = ""

    def __post_init__(self) -> None:
        if self.stage not in REQUIRED_STAGE_ORDER:
            raise ValueError(f"unknown stage: {self.stage}")


@dataclass(frozen=True)
class PartitionEvidence:
    name: str
    status: CIStatus
    passed: int
    failed: int

    def __post_init__(self) -> None:
        if self.passed < 0 or self.failed < 0:
            raise ValueError("negative test count")


def canonical_partitions() -> tuple[TestPartition, ...]:
    return (
        TestPartition("qa029_qa042", 29, 42, 210),
        TestPartition("qa043_qa045", 43, 45, 30),
        TestPartition("qa046", 46, 46, 12),
        TestPartition("qa047", 47, 47, 12),
        TestPartition("qa048", 48, 48, 12),
        TestPartition("qa049", 49, 49, 12),
        TestPartition("qa050", 50, 50, 12),
        TestPartition("qa051", 51, 51, 12),
        TestPartition("qa052", 52, 52, 12),
    )


def expected_total_tests() -> int:
    return sum(p.expected_tests for p in canonical_partitions())


def expected_qa_test_files(test_dir: str | Path) -> tuple[Path, ...]:
    root = Path(test_dir)
    out: list[Path] = []
    for qa in range(29, 53):
        out.extend(sorted(root.glob(f"test_qa{qa:03d}*.py")))
    return tuple(out)


def assert_exact_partition_file_coverage(test_dir: str | Path) -> None:
    expected = expected_qa_test_files(test_dir)
    assigned = tuple(f for p in canonical_partitions() for f in p.matching_files(test_dir))
    if not expected:
        raise ValueError("no QA029-QA052 test files found")
    if len(assigned) != len(set(assigned)):
        raise ValueError("duplicate test file assignment across CI partitions")
    if set(assigned) != set(expected):
        missing = sorted(set(expected) - set(assigned))
        extra = sorted(set(assigned) - set(expected))
        raise ValueError(f"partition coverage mismatch: missing={missing}, extra={extra}")


def assert_installed_import_outside_source(imported_file: str | Path, source_tree: str | Path) -> None:
    imported = Path(imported_file).resolve()
    source = Path(source_tree).resolve()
    if imported == source or source in imported.parents:
        raise ValueError("installed-artifact replay imported gilttpy from the governed source tree")


def cell_can_pass(
    stages: Iterable[StageEvidence],
    partitions: Iterable[PartitionEvidence],
) -> bool:
    stage_map = {x.stage: x for x in stages}
    if set(stage_map) != set(REQUIRED_STAGE_ORDER):
        return False
    if any(stage_map[name].status is not CIStatus.PASS for name in REQUIRED_STAGE_ORDER):
        return False
    part_map = {x.name: x for x in partitions}
    expected = {p.name: p for p in canonical_partitions()}
    if set(part_map) != set(expected):
        return False
    for name, spec in expected.items():
        ev = part_map[name]
        if ev.status is not CIStatus.PASS or ev.failed != 0 or ev.passed != spec.expected_tests:
            return False
    return True
