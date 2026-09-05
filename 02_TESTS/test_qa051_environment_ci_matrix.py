from __future__ import annotations

import json
from pathlib import Path
import tomllib

from gilttpy.engineering.environment_matrix import (
    QA051_GATE, QA051_HOLDS, QA051_PROHIBITIONS,
    MatrixEvidenceStatus, REQUIRES_PYTHON, SUPPORTED_PLATFORM_FAMILIES,
    SUPPORTED_PYTHON_SERIES, assert_no_false_pass, declared_matrix,
    evidence_matrix, runtime_platform_family, runtime_python_series,
)
from gilttpy.validation.environment_matrix_campaign import (
    QA051_REFERENCE_DIRECT_PINS, installed_direct_versions,
    local_reference_matches, qa051_runtime_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_qa051_01_gate_holds_and_prohibitions_are_explicit():
    assert QA051_GATE == "PASS_REPRODUCIBLE_ENVIRONMENT_AND_CI_MATRIX_SPECIFICATION"
    assert "HOLD_HERMETIC_HASH_LOCKED_DEPENDENCY_RESOLUTION" in QA051_HOLDS
    assert "PROHIBIT_UNEXECUTED_MATRIX_CELL_AS_PASS" in QA051_PROHIBITIONS
    assert "PROHIBIT_TARGET_TUNING" in QA051_PROHIBITIONS


def test_qa051_02_declared_matrix_is_exact_3x4_cartesian_product():
    m=declared_matrix()
    assert len(m)==12 and len(set(m))==12
    assert set(x.platform_family for x in m)==set(SUPPORTED_PLATFORM_FAMILIES)
    assert set(x.python_series for x in m)==set(SUPPORTED_PYTHON_SERIES)


def test_qa051_03_runtime_evidence_promotes_only_the_executed_cell():
    m=evidence_matrix(observed_python="runtime", observed_platform="runtime")
    assert_no_false_pass(m)
    tested=[x for x in m if x.status is MatrixEvidenceStatus.LOCAL_TESTED]
    assert len(tested)==1
    assert tested[0].target.platform_family==runtime_platform_family()
    assert tested[0].target.python_series==runtime_python_series()
    assert all(x.status is MatrixEvidenceStatus.CI_REQUIRED for x in m if x is not tested[0])


def test_qa051_04_pyproject_bounds_python_to_current_verified_series():
    data=tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert data['project']['requires-python']==REQUIRES_PYTHON==">=3.11,<3.15"


def test_qa051_05_runtime_dependencies_remain_the_qa050_direct_contract():
    data=tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert set(data['project']['dependencies'])=={"numpy>=1.26","scipy>=1.11","mpmath>=1.3"}
    assert set(data['project']['optional-dependencies']['test'])=={"pytest>=8","threadpoolctl>=3"}


def test_qa051_06_local_reference_versions_are_explicit_and_match_runtime():
    assert installed_direct_versions()==QA051_REFERENCE_DIRECT_PINS
    assert local_reference_matches()


def test_qa051_07_reference_pin_file_is_labeled_nonhermetic_and_matches_direct_versions():
    text=(ROOT/'requirements/qa051_reference_py313_linux_x86_64.txt').read_text()
    assert "NOT A CROSS-PLATFORM LOCK FILE" in text
    for name,version in QA051_REFERENCE_DIRECT_PINS.items():
        assert f"{name}=={version}" in text


def test_qa051_08_provider_neutral_ci_spec_covers_all_platform_python_targets():
    spec=json.loads((ROOT/'ci/qa051_ci_matrix.json').read_text())
    assert set(spec['python_series'])==set(SUPPORTED_PYTHON_SERIES)
    assert set(spec['platforms'])==set(SUPPORTED_PLATFORM_FAMILIES)
    assert "full frozen suite" in spec['pass_rule']


def test_qa051_09_ci_spec_freezes_single_thread_numerical_controls():
    spec=json.loads((ROOT/'ci/qa051_ci_matrix.json').read_text())
    env=spec['thread_environment']
    assert env['OPENBLAS_NUM_THREADS']==env['OMP_NUM_THREADS']==env['MKL_NUM_THREADS']=="1"
    assert env['PYTHONHASHSEED']=="0"


def test_qa051_10_hermetic_and_minimum_dependency_tracks_remain_holds():
    spec=json.loads((ROOT/'ci/qa051_ci_matrix.json').read_text())
    assert spec['dependency_tracks']['minimum_supported'].startswith('HOLD')
    assert spec['dependency_tracks']['hash_locked_hermetic'].startswith('HOLD')
    ev=qa051_runtime_evidence()
    assert ev['semantics']['reference_pins_are_hermetic_lock'] is False


def test_qa051_11_upstream_support_metadata_cannot_promote_gilttpy_cells():
    ev=qa051_runtime_evidence()
    assert ev['semantics']['upstream_support_implies_gilttpy_pass'] is False
    statuses=[x['status'] for x in ev['matrix']]
    assert statuses.count('LOCAL_TESTED')==1 and statuses.count('CI_REQUIRED')==11


def test_qa051_12_scope_governance_preserves_engineering_science_separation():
    text=(ROOT/'QA051_SCOPE_GOVERNANCE.md').read_text()
    assert "does not modify or revalidate scientific equations" in text
    assert "NO TARGET TUNING" in text
    assert "hermetic cross-platform lock" in text
