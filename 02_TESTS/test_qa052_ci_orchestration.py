from __future__ import annotations

import json
from pathlib import Path

import pytest

from gilttpy.engineering.ci_orchestration import (
    QA052_GATE, QA052_HOLDS, QA052_PROHIBITIONS,
    CIStatus, PartitionEvidence, REQUIRED_STAGE_ORDER, StageEvidence,
    THREAD_ENVIRONMENT, assert_exact_partition_file_coverage,
    assert_installed_import_outside_source, canonical_partitions,
    cell_can_pass, expected_total_tests,
)
from gilttpy.validation.ci_orchestration_campaign import provider_neutral_contract

ROOT = Path(__file__).resolve().parents[1]


def _pass_stages():
    return tuple(StageEvidence(name, CIStatus.PASS) for name in REQUIRED_STAGE_ORDER)


def _pass_partitions():
    return tuple(PartitionEvidence(p.name, CIStatus.PASS, p.expected_tests, 0) for p in canonical_partitions())


def test_qa052_01_gate_holds_prohibitions_explicit():
    assert QA052_GATE == "PASS_DETERMINISTIC_CI_ORCHESTRATION_SPECIFICATION"
    assert "HOLD_EXTERNAL_CI_MATRIX_EXECUTION" in QA052_HOLDS
    assert "PROHIBIT_PARTIAL_PARTITION_SET_AS_FULL_PASS" in QA052_PROHIBITIONS
    assert "PROHIBIT_TARGET_TUNING" in QA052_PROHIBITIONS


def test_qa052_02_required_stage_order_is_frozen():
    assert REQUIRED_STAGE_ORDER == (
        "metadata_contract", "distribution_build", "installed_artifact_import",
        "partitioned_test_replay", "artifact_integrity", "evidence_record",
    )


def test_qa052_03_partition_test_counts_are_explicit_and_total_324():
    parts=canonical_partitions()
    assert [p.name for p in parts] == [
        "qa029_qa042","qa043_qa045","qa046","qa047","qa048","qa049","qa050","qa051","qa052"
    ]
    assert expected_total_tests() == 324


def test_qa052_04_partition_file_coverage_is_exact_no_duplicate():
    assert_exact_partition_file_coverage(ROOT / "02_TESTS")


def test_qa052_05_full_pass_requires_every_stage_and_partition():
    assert cell_can_pass(_pass_stages(), _pass_partitions())
    assert not cell_can_pass(_pass_stages()[:-1], _pass_partitions())
    assert not cell_can_pass(_pass_stages(), _pass_partitions()[:-1])


def test_qa052_06_failed_or_miscounted_partition_blocks_cell_pass():
    parts=list(_pass_partitions())
    parts[-1]=PartitionEvidence("qa052", CIStatus.FAIL, 11, 1)
    assert not cell_can_pass(_pass_stages(), parts)
    parts[-1]=PartitionEvidence("qa052", CIStatus.PASS, 11, 0)
    assert not cell_can_pass(_pass_stages(), parts)


def test_qa052_07_failed_artifact_or_import_stage_blocks_pass():
    stages=list(_pass_stages())
    i=REQUIRED_STAGE_ORDER.index("installed_artifact_import")
    stages[i]=StageEvidence("installed_artifact_import", CIStatus.FAIL)
    assert not cell_can_pass(stages, _pass_partitions())
    stages=list(_pass_stages())
    i=REQUIRED_STAGE_ORDER.index("artifact_integrity")
    stages[i]=StageEvidence("artifact_integrity", CIStatus.FAIL)
    assert not cell_can_pass(stages, _pass_partitions())


def test_qa052_08_import_provenance_rejects_governed_source_tree(tmp_path):
    source=tmp_path/"project"/"01_SRC"/"gilttpy"; source.mkdir(parents=True)
    with pytest.raises(ValueError):
        assert_installed_import_outside_source(source/"__init__.py", tmp_path/"project")
    installed=tmp_path/"venv"/"site-packages"/"gilttpy"/"__init__.py"
    installed.parent.mkdir(parents=True)
    assert_installed_import_outside_source(installed, tmp_path/"project")


def test_qa052_09_numerical_thread_environment_is_exact():
    assert THREAD_ENVIRONMENT == {
        "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1",
        "MKL_NUM_THREADS":"1", "PYTHONHASHSEED":"0",
    }


def test_qa052_10_contract_is_provider_neutral_and_no_false_matrix_pass():
    c=provider_neutral_contract()
    assert c["matrix"]["promotion_semantics"].startswith("each matrix cell")
    assert c["failure_semantics"]["unexecuted_cell"] == "CI_REQUIRED"
    assert c["artifact_integrity"]["cross_platform_wheel_byte_identity_required"] is False


def test_qa052_11_standard_build_is_specified_but_local_status_stays_hold():
    c=provider_neutral_contract()
    assert c["build"]["canonical_release_ci_command"] == ["python","-m","build"]
    assert c["build"]["required_outputs"] == ["sdist","wheel"]
    assert c["build"]["local_qa052_execution_status"].startswith("HOLD")


def test_qa052_12_engineering_contract_contains_no_target_selection_logic():
    text=(ROOT/"01_SRC/gilttpy/engineering/ci_orchestration.py").read_text().lower()
    text += json.dumps(provider_neutral_contract()).lower()
    for forbidden in ("observed concentration", "historical concentration", "copenhagen", "hanford"):
        assert forbidden not in text
