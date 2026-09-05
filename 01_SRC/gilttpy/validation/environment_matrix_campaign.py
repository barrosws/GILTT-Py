"""QA-051 provider-neutral environment/dependency matrix campaign."""
from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

from gilttpy.engineering.environment_matrix import (
    QA051_GATE, QA051_HOLDS, QA051_PROHIBITIONS, REQUIRES_PYTHON,
    SUPPORTED_PLATFORM_FAMILIES, SUPPORTED_PYTHON_SERIES,
    assert_no_false_pass, evidence_matrix,
)

QA051_REFERENCE_DIRECT_PINS = {
    "numpy": "2.3.5",
    "scipy": "1.17.0",
    "mpmath": "1.3.0",
    "pytest": "9.0.2",
    "threadpoolctl": "3.6.0",
}


def installed_direct_versions() -> dict[str, str]:
    return {name: md.version(name) for name in QA051_REFERENCE_DIRECT_PINS}


def local_reference_matches() -> bool:
    return installed_direct_versions() == QA051_REFERENCE_DIRECT_PINS


def qa051_runtime_evidence() -> dict:
    matrix = evidence_matrix(
        observed_python=platform.python_version(),
        observed_platform=f"{platform.system()} {platform.machine()}",
    )
    assert_no_false_pass(matrix)
    return {
        "gate": QA051_GATE,
        "requires_python": REQUIRES_PYTHON,
        "supported_python_series": list(SUPPORTED_PYTHON_SERIES),
        "supported_platform_families": list(SUPPORTED_PLATFORM_FAMILIES),
        "runtime": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "python_series": f"{sys.version_info.major}.{sys.version_info.minor}",
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "direct_versions": installed_direct_versions(),
        "reference_direct_pins": dict(QA051_REFERENCE_DIRECT_PINS),
        "reference_pins_match": local_reference_matches(),
        "matrix": [x.to_dict() for x in matrix],
        "holds": list(QA051_HOLDS),
        "prohibitions": list(QA051_PROHIBITIONS),
        "semantics": {
            "compatibility_target_is_test_evidence": False,
            "reference_pins_are_hermetic_lock": False,
            "upstream_support_implies_gilttpy_pass": False,
        },
    }


def write_runtime_evidence(path: str | Path) -> None:
    Path(path).write_text(json.dumps(qa051_runtime_evidence(), indent=2, sort_keys=True) + "\n")
