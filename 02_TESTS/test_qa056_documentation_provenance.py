from pathlib import Path
import json

from gilttpy.engineering.provenance_catalog import (
    QA056_GATE, QA056_HOLDS, QA056_PROHIBITIONS, REQUIRED_ROLES,
    EvidenceStatus, audit_can_pass, audit_documentation, evidence_map,
    provenance_index, sha256_file,
)
from gilttpy.validation.documentation_provenance_campaign import qa056_documentation_evidence

ROOT = Path(__file__).resolve().parents[1]


def test_qa056_01_gate_holds_prohibitions_explicit():
    assert QA056_GATE == "PASS_DOCUMENTATION_PROVENANCE_COMPLETENESS_AUDIT"
    assert "HOLD_CHANGELOG_AND_RELEASE_NOTES" in QA056_HOLDS
    assert "HOLD_FINAL_PUBLIC_RELEASE_METADATA" in QA056_HOLDS
    assert "PROHIBIT_UNVERIFIED_DATA_REDISTRIBUTION_RIGHTS" in QA056_PROHIBITIONS
    assert "PROHIBIT_TARGET_TUNING" in QA056_PROHIBITIONS


def test_qa056_02_required_roles_are_unique_and_archival_scoped():
    keys = [r.key for r in REQUIRED_ROLES]
    assert len(keys) == len(set(keys)) == 13
    assert all(r.description for r in REQUIRED_ROLES)
    assert sum(r.required_for_archival_release for r in REQUIRED_ROLES) >= 10


def test_qa056_03_audit_explicitly_classifies_every_required_role():
    items = audit_documentation(ROOT)
    assert set(evidence_map(items)) == {r.key for r in REQUIRED_ROLES}
    assert audit_can_pass(items)


def test_qa056_04_existing_scope_checksums_tests_runtime_and_claim_governance_present():
    m = evidence_map(audit_documentation(ROOT))
    for key in ("readme_project_scope", "checksums", "test_evidence", "runtime_environment_provenance", "qa_claim_governance"):
        assert m[key].status is EvidenceStatus.PRESENT


def test_qa056_05_installation_quickstart_is_not_silently_claimed_complete():
    m = evidence_map(audit_documentation(ROOT))
    assert m["installation_quickstart"].status is EvidenceStatus.HOLD
    assert "README" in m["installation_quickstart"].paths[0]


def test_qa056_06_reproduction_entrypoint_and_changelog_are_explicit_holds():
    m = evidence_map(audit_documentation(ROOT))
    assert m["reproduction_entrypoint"].status is EvidenceStatus.HOLD
    assert m["changelog_release_notes"].status is EvidenceStatus.HOLD


def test_qa056_07_citation_and_license_remain_qa054_holds_not_fabricated():
    m = evidence_map(audit_documentation(ROOT))
    assert m["citation_metadata"].status is EvidenceStatus.HOLD
    assert m["software_license"].status is EvidenceStatus.HOLD
    assert not (ROOT / "CITATION.cff").exists()
    assert not (ROOT / "LICENSE").exists()


def test_qa056_08_data_rights_and_coverage_are_not_silently_promoted():
    m = evidence_map(audit_documentation(ROOT))
    assert m["data_rights_provenance"].status is EvidenceStatus.HOLD
    assert m["coverage_report"].status is EvidenceStatus.HOLD


def test_qa056_09_machine_readable_engineering_configs_exist_but_scope_note_is_conservative():
    m = evidence_map(audit_documentation(ROOT))
    assert m["machine_readable_configs"].status is EvidenceStatus.PRESENT
    assert len(m["machine_readable_configs"].paths) >= 3
    assert "coverage" in m["machine_readable_configs"].note.lower()


def test_qa056_10_provenance_index_hashes_critical_files_exactly():
    idx = provenance_index(ROOT)
    assert idx["archival_release_ready"] is False
    for rel, digest in idx["critical_hashes"].items():
        assert digest == sha256_file(ROOT / rel)


def test_qa056_11_campaign_exposes_present_and_hold_roles_without_release_promotion():
    ev = qa056_documentation_evidence(ROOT)
    assert ev["role_count"] == 13
    assert ev["archival_release_ready"] is False
    assert "checksums" in ev["present_roles"]
    assert "changelog_release_notes" in ev["hold_roles"]


def test_qa056_12_engineering_audit_has_no_transport_target_logic():
    text = (ROOT / "01_SRC/gilttpy/engineering/provenance_catalog.py").read_text().lower()
    text += (ROOT / "01_SRC/gilttpy/validation/documentation_provenance_campaign.py").read_text().lower()
    for forbidden in ("observed concentration", "historical concentration", "copenhagen", "hanford"):
        assert forbidden not in text
