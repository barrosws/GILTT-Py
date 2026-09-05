"""QA-052 provider-neutral CI orchestration evidence contract."""
from __future__ import annotations

from pathlib import Path
import json

from gilttpy.engineering.ci_orchestration import (
    QA052_GATE, QA052_HOLDS, QA052_PROHIBITIONS,
    REQUIRED_STAGE_ORDER, THREAD_ENVIRONMENT,
    canonical_partitions, expected_total_tests,
)
from gilttpy.engineering.environment_matrix import (
    SUPPORTED_PLATFORM_FAMILIES, SUPPORTED_PYTHON_SERIES,
)


def provider_neutral_contract() -> dict:
    return {
        "gate": QA052_GATE,
        "matrix": {
            "platforms": list(SUPPORTED_PLATFORM_FAMILIES),
            "python_series": list(SUPPORTED_PYTHON_SERIES),
            "promotion_semantics": "each matrix cell is independent evidence; no transitive PASS",
        },
        "stage_order": list(REQUIRED_STAGE_ORDER),
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "build": {
            "canonical_release_ci_command": ["python", "-m", "build"],
            "required_outputs": ["sdist", "wheel"],
            "local_qa052_execution_status": "HOLD_BUILD_FRONTEND_NOT_INSTALLED_IN_LOCAL_EXECUTOR",
        },
        "installed_replay": {
            "source_tree_import_allowed": False,
            "pytest_entrypoint": ["python", "-m", "pytest"],
            "test_execution": "deterministic partitions",
        },
        "partitions": [
            {
                "name": p.name,
                "qa_start": p.qa_start,
                "qa_end": p.qa_end,
                "expected_tests": p.expected_tests,
            }
            for p in canonical_partitions()
        ],
        "expected_total_tests": expected_total_tests(),
        "artifact_integrity": {
            "sha256_required": True,
            "cross_platform_wheel_byte_identity_required": False,
            "reason": "platform/Python cells are independently verified; cross-platform wheel byte identity is not assumed",
        },
        "failure_semantics": {
            "missing_stage": "FAIL_CELL",
            "failed_test": "FAIL_CELL",
            "unexpected_test_count": "FAIL_CELL",
            "source_tree_import": "FAIL_CELL",
            "artifact_hash_mismatch": "FAIL_CELL",
            "unexecuted_cell": "CI_REQUIRED",
        },
        "holds": list(QA052_HOLDS),
        "prohibitions": list(QA052_PROHIBITIONS),
    }


def write_contract(path: str | Path) -> None:
    Path(path).write_text(json.dumps(provider_neutral_contract(), indent=2, sort_keys=True) + "\n")
