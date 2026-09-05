"""QA-055 clean-room release reproducibility contract.

This engineering layer defines evidence required to call one local release
reproduction clean-room verified.  It does not promote unexecuted platform or
Python cells, unresolved public metadata, or scientific claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Iterable

QA055_GATE = "PASS_CLEAN_ROOM_RELEASE_REPRODUCIBILITY_CONTRACT"
QA055_HOLDS = (
    "HOLD_EXTERNAL_CI_MATRIX_EXECUTION",
    "HOLD_CROSS_PLATFORM_HERMETIC_LOCK",
    "HOLD_MINIMUM_SUPPORTED_DEPENDENCY_EXECUTION",
    "HOLD_STANDARD_PYPA_BUILD_FRONTEND_LOCAL_EXECUTION",
    "HOLD_FINAL_PUBLIC_RELEASE_METADATA",
    "HOLD_VCS_TAG_SIGNING_AND_PUBLICATION",
)
QA055_PROHIBITIONS = (
    "PROHIBIT_SOURCE_TREE_IMPORT_AS_CLEAN_ROOM_EVIDENCE",
    "PROHIBIT_UNEXECUTED_MATRIX_CELL_AS_PASS",
    "PROHIBIT_LOCAL_CLEAN_ROOM_AS_CROSS_PLATFORM_PROOF",
    "PROHIBIT_UNRESOLVED_METADATA_AS_FINAL_RELEASE_METADATA",
    "PROHIBIT_ENGINEERING_GATE_AS_SCIENTIFIC_VALIDATION",
    "PROHIBIT_TARGET_TUNING",
)

THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}

REQUIRED_CLEAN_ROOM_STAGES = (
    "source_integrity",
    "distribution_build",
    "artifact_integrity",
    "isolated_install",
    "installed_import_provenance",
    "partitioned_test_replay",
    "runtime_provenance",
    "evidence_record",
)


class EvidenceStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CleanRoomStageEvidence:
    name: str
    status: EvidenceStatus
    note: str = ""

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_CLEAN_ROOM_STAGES:
            raise ValueError(f"unknown clean-room stage: {self.name}")


@dataclass(frozen=True)
class CleanRoomPartition:
    name: str
    qa_start: int
    qa_end: int
    expected_tests: int

    def __post_init__(self) -> None:
        if not self.name or self.qa_start > self.qa_end or self.expected_tests <= 0:
            raise ValueError("invalid clean-room partition")

    def matching_files(self, test_dir: str | Path) -> tuple[Path, ...]:
        root = Path(test_dir)
        files: list[Path] = []
        for qa in range(self.qa_start, self.qa_end + 1):
            files.extend(sorted(root.glob(f"test_qa{qa:03d}*.py")))
        return tuple(files)


@dataclass(frozen=True)
class PartitionEvidence:
    name: str
    status: EvidenceStatus
    passed: int
    failed: int

    def __post_init__(self) -> None:
        if self.passed < 0 or self.failed < 0:
            raise ValueError("negative test count")


def canonical_partitions() -> tuple[CleanRoomPartition, ...]:
    return (
        CleanRoomPartition("qa029_qa042", 29, 42, 210),
        CleanRoomPartition("qa043_qa045", 43, 45, 30),
        CleanRoomPartition("qa046", 46, 46, 12),
        CleanRoomPartition("qa047", 47, 47, 12),
        CleanRoomPartition("qa048", 48, 48, 12),
        CleanRoomPartition("qa049", 49, 49, 12),
        CleanRoomPartition("qa050", 50, 50, 12),
        CleanRoomPartition("qa051", 51, 51, 12),
        CleanRoomPartition("qa052", 52, 52, 12),
        CleanRoomPartition("qa053", 53, 53, 12),
        CleanRoomPartition("qa054", 54, 54, 12),
        CleanRoomPartition("qa055", 55, 55, 12),
    )


def expected_total_tests() -> int:
    return sum(p.expected_tests for p in canonical_partitions())


def expected_test_files(test_dir: str | Path) -> tuple[Path, ...]:
    root = Path(test_dir)
    out: list[Path] = []
    for qa in range(29, 56):
        out.extend(sorted(root.glob(f"test_qa{qa:03d}*.py")))
    return tuple(out)


def assert_exact_partition_file_coverage(test_dir: str | Path) -> None:
    expected = expected_test_files(test_dir)
    assigned = tuple(f for part in canonical_partitions() for f in part.matching_files(test_dir))
    if not expected:
        raise ValueError("no QA029-QA055 test files found")
    if len(assigned) != len(set(assigned)):
        raise ValueError("duplicate test-file assignment across clean-room partitions")
    if set(assigned) != set(expected):
        missing = sorted(set(expected) - set(assigned))
        extra = sorted(set(assigned) - set(expected))
        raise ValueError(f"partition coverage mismatch: missing={missing}, extra={extra}")


def assert_import_outside_governed_tree(imported_file: str | Path, governed_root: str | Path) -> None:
    imported = Path(imported_file).resolve()
    governed = Path(governed_root).resolve()
    if imported == governed or governed in imported.parents:
        raise ValueError("clean-room import resolved inside the governed source/package tree")


def clean_room_can_pass(
    stages: Iterable[CleanRoomStageEvidence],
    partitions: Iterable[PartitionEvidence],
) -> bool:
    stage_map = {item.name: item for item in stages}
    if set(stage_map) != set(REQUIRED_CLEAN_ROOM_STAGES):
        return False
    if any(stage_map[name].status is not EvidenceStatus.PASS for name in REQUIRED_CLEAN_ROOM_STAGES):
        return False
    part_map = {item.name: item for item in partitions}
    expected = {p.name: p for p in canonical_partitions()}
    if set(part_map) != set(expected):
        return False
    for name, spec in expected.items():
        evidence = part_map[name]
        if evidence.status is not EvidenceStatus.PASS:
            return False
        if evidence.failed != 0 or evidence.passed != spec.expected_tests:
            return False
    return True


def runtime_fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
