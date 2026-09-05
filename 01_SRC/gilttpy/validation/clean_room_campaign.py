"""QA-055 provider-neutral clean-room release reproducibility campaign contract."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from gilttpy.engineering.clean_room import (
    QA055_GATE,
    QA055_HOLDS,
    QA055_PROHIBITIONS,
    REQUIRED_CLEAN_ROOM_STAGES,
    THREAD_ENVIRONMENT,
    canonical_partitions,
    expected_total_tests,
)
from gilttpy.engineering.release_metadata import load_state, public_release_ready


def standard_build_frontend_available() -> bool:
    return importlib.util.find_spec("build") is not None


def clean_room_contract(project_root: str | Path) -> dict:
    root = Path(project_root)
    metadata_state = load_state(root / "metadata/qa054_release_metadata_state.json")
    return {
        "gate": QA055_GATE,
        "scope": "local clean-room release reproducibility evidence; not public release readiness",
        "stage_order": list(REQUIRED_CLEAN_ROOM_STAGES),
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "build": {
            "canonical_release_ci_command": ["python", "-m", "build"],
            "local_frontend_available": standard_build_frontend_available(),
            "local_fallback_command": ["python", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "."],
            "fallback_role": "local deterministic engineering evidence only; does not resolve standard frontend HOLD",
            "required_release_outputs": ["sdist", "wheel"],
        },
        "installation": {
            "source_tree_import_allowed": False,
            "install_target": "fresh external site-packages directory",
            "dependency_semantics": "QA053 local resolution inherited; not a universal hermetic lock",
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
        "public_release_metadata_ready": public_release_ready(metadata_state),
        "matrix_semantics": "local clean-room PASS is evidence only for the executed local cell; no cross-platform promotion",
        "holds": list(QA055_HOLDS),
        "prohibitions": list(QA055_PROHIBITIONS),
    }


def write_contract(project_root: str | Path, output: str | Path) -> dict:
    payload = clean_room_contract(project_root)
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
