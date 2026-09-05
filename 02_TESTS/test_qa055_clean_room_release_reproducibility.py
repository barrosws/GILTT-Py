from pathlib import Path
import pytest

from gilttpy.engineering.clean_room import (
    QA055_GATE,
    QA055_HOLDS,
    QA055_PROHIBITIONS,
    REQUIRED_CLEAN_ROOM_STAGES,
    THREAD_ENVIRONMENT,
    CleanRoomStageEvidence,
    EvidenceStatus,
    PartitionEvidence,
    assert_exact_partition_file_coverage,
    assert_import_outside_governed_tree,
    canonical_partitions,
    clean_room_can_pass,
    expected_total_tests,
    runtime_fingerprint,
)
from gilttpy.validation.clean_room_campaign import clean_room_contract

ROOT = Path(__file__).resolve().parents[1]


def _pass_stages():
    return tuple(CleanRoomStageEvidence(name, EvidenceStatus.PASS) for name in REQUIRED_CLEAN_ROOM_STAGES)


def _pass_partitions():
    return tuple(PartitionEvidence(p.name, EvidenceStatus.PASS, p.expected_tests, 0) for p in canonical_partitions())


def test_qa055_01_gate_holds_prohibitions_explicit():
    assert QA055_GATE == "PASS_CLEAN_ROOM_RELEASE_REPRODUCIBILITY_CONTRACT"
    assert "HOLD_EXTERNAL_CI_MATRIX_EXECUTION" in QA055_HOLDS
    assert "HOLD_FINAL_PUBLIC_RELEASE_METADATA" in QA055_HOLDS
    assert "PROHIBIT_SOURCE_TREE_IMPORT_AS_CLEAN_ROOM_EVIDENCE" in QA055_PROHIBITIONS
    assert "PROHIBIT_TARGET_TUNING" in QA055_PROHIBITIONS


def test_qa055_02_stage_order_is_exact():
    assert REQUIRED_CLEAN_ROOM_STAGES == (
        "source_integrity", "distribution_build", "artifact_integrity", "isolated_install",
        "installed_import_provenance", "partitioned_test_replay", "runtime_provenance", "evidence_record",
    )


def test_qa055_03_partition_counts_are_explicit_and_total_360():
    parts = canonical_partitions()
    assert parts[-2].name == "qa054" and parts[-2].expected_tests == 12
    assert parts[-1].name == "qa055" and parts[-1].expected_tests == 12
    assert expected_total_tests() == 360


def test_qa055_04_partition_file_coverage_is_exact():
    assert_exact_partition_file_coverage(ROOT / "02_TESTS")


def test_qa055_05_full_pass_requires_every_stage_and_partition():
    assert clean_room_can_pass(_pass_stages(), _pass_partitions())
    assert not clean_room_can_pass(_pass_stages()[:-1], _pass_partitions())
    assert not clean_room_can_pass(_pass_stages(), _pass_partitions()[:-1])


def test_qa055_06_failed_or_miscounted_evidence_blocks_pass():
    stages = list(_pass_stages())
    stages[3] = CleanRoomStageEvidence("isolated_install", EvidenceStatus.FAIL)
    assert not clean_room_can_pass(stages, _pass_partitions())
    parts = list(_pass_partitions())
    parts[-1] = PartitionEvidence("qa055", EvidenceStatus.PASS, 11, 0)
    assert not clean_room_can_pass(_pass_stages(), parts)


def test_qa055_07_import_provenance_rejects_governed_tree(tmp_path):
    governed = tmp_path / "governed"
    source = governed / "01_SRC/gilttpy/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("")
    with pytest.raises(ValueError):
        assert_import_outside_governed_tree(source, governed)
    installed = tmp_path / "clean/site-packages/gilttpy/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("")
    assert_import_outside_governed_tree(installed, governed)


def test_qa055_08_thread_environment_is_frozen():
    assert THREAD_ENVIRONMENT == {
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
    }


def test_qa055_09_contract_is_local_evidence_not_cross_platform_proof():
    c = clean_room_contract(ROOT)
    assert c["expected_total_tests"] == 360
    assert "no cross-platform promotion" in c["matrix_semantics"]
    assert c["installation"]["source_tree_import_allowed"] is False


def test_qa055_10_standard_build_is_canonical_and_fallback_is_not_promotion():
    c = clean_room_contract(ROOT)
    assert c["build"]["canonical_release_ci_command"] == ["python", "-m", "build"]
    assert c["build"]["required_release_outputs"] == ["sdist", "wheel"]
    assert "does not resolve standard frontend HOLD" in c["build"]["fallback_role"]


def test_qa055_11_unresolved_qa054_metadata_still_blocks_public_release():
    c = clean_room_contract(ROOT)
    assert c["public_release_metadata_ready"] is False
    assert not (ROOT / "CITATION.cff").exists()
    assert (ROOT / "CITATION.cff.template").exists()


def test_qa055_12_runtime_fingerprint_is_order_invariant_and_no_target_logic():
    assert runtime_fingerprint({"python": "3.13", "os": "linux"}) == runtime_fingerprint({"os": "linux", "python": "3.13"})
    text = (ROOT / "01_SRC/gilttpy/engineering/clean_room.py").read_text().lower()
    text += (ROOT / "01_SRC/gilttpy/validation/clean_room_campaign.py").read_text().lower()
    for forbidden in ("observed concentration", "historical concentration", "copenhagen", "hanford"):
        assert forbidden not in text
