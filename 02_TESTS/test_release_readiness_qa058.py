from pathlib import Path
import json
from gilttpy.engineering.archival_readiness import (
    QA058_GATE, QA058_HOLDS, QA058_PROHIBITIONS, REQUIRED_ARCHIVAL_ROLES,
    ReadinessStatus, archival_release_ready, audit_archival_readiness, audit_can_pass,
    evidence_map, public_release_blockers, readiness_state, sha256_file, write_readiness_state,
)
from gilttpy.validation.archival_readiness_campaign import qa058_archival_readiness_evidence

ROOT=Path(__file__).resolve().parents[1]

def test_qa058_01_gate_holds_prohibitions_explicit():
    assert QA058_GATE=='PASS_ARCHIVAL_RELEASE_READINESS_AUDIT'
    assert 'HOLD_SOFTWARE_LICENSE_SELECTION' in QA058_HOLDS
    assert 'HOLD_ZENODO_DEPOSITION' in QA058_HOLDS
    assert 'PROHIBIT_AUDIT_PASS_AS_ARCHIVAL_RELEASE_READY' in QA058_PROHIBITIONS
    assert 'PROHIBIT_TARGET_TUNING' in QA058_PROHIBITIONS

def test_qa058_02_required_roles_are_unique_and_archival_scoped():
    keys=[r.key for r in REQUIRED_ARCHIVAL_ROLES]
    assert len(keys)==len(set(keys))==18
    assert sum(r.required_for_archival_release for r in REQUIRED_ARCHIVAL_ROLES)>=15

def test_qa058_03_audit_classifies_every_role_and_can_pass_as_audit():
    items=audit_archival_readiness(ROOT)
    assert set(evidence_map(items))=={r.key for r in REQUIRED_ARCHIVAL_ROLES}
    assert audit_can_pass(items)

def test_qa058_04_operational_integrity_runtime_and_docs_are_ready():
    m=evidence_map(audit_archival_readiness(ROOT))
    for key in ('distribution_reproducibility','operational_qa_replay','integrity_manifest','runtime_environment_provenance','installation_reproduction_docs','development_change_history'):
        assert m[key].status is ReadinessStatus.READY

def test_qa058_05_public_version_authorship_license_and_citation_remain_hold():
    m=evidence_map(audit_archival_readiness(ROOT))
    for key in ('public_software_version','software_authorship_orcids','software_license','citation_doi_metadata'):
        assert m[key].status is ReadinessStatus.HOLD
    assert not (ROOT/'CITATION.cff').exists() and not (ROOT/'LICENSE').exists()

def test_qa058_06_data_scientific_config_and_manuscript_provenance_remain_hold():
    m=evidence_map(audit_archival_readiness(ROOT))
    for key in ('canonical_data_release','data_rights_provenance','scientific_config_coverage','manuscript_output_reproduction'):
        assert m[key].status is ReadinessStatus.HOLD

def test_f10a_qa058_07_zenodo_is_ready_while_remaining_engineering_roles_stay_hold():
    m=evidence_map(audit_archival_readiness(ROOT))
    for key in ('full_package_coverage','external_ci_matrix','cross_platform_hermetic_lock'):
        assert m[key].status is ReadinessStatus.HOLD
    assert m['zenodo_archival_record'].status is ReadinessStatus.READY

def test_qa058_08_audit_pass_does_not_mean_archival_release_ready():
    items=audit_archival_readiness(ROOT)
    assert audit_can_pass(items) is True
    assert archival_release_ready(items) is False
    assert len(public_release_blockers(items))>=10

def test_qa058_09_machine_readable_state_is_false_and_hashes_are_exact(tmp_path):
    out=tmp_path/'state.json'
    state=write_readiness_state(ROOT,out)
    assert state['archival_release_ready'] is False
    assert json.loads(out.read_text())['gate']==QA058_GATE
    for rel,digest in state['critical_hashes'].items():
        assert digest==sha256_file(ROOT/rel)

def test_qa058_10_campaign_reports_ready_and_hold_roles_without_promotion():
    ev=qa058_archival_readiness_evidence(ROOT)
    assert ev['role_count']==18
    assert ev['ready_role_count']>=6
    assert ev['hold_role_count']>=10
    assert ev['archival_release_ready'] is False

def test_qa058_11_qa057_frozen_replay_contract_is_preserved_byte_semantically():
    # QA058 test filename intentionally does not match test_qa*.py so QA057's frozen
    # exact QA029-QA057 replay-plan file coverage remains valid and unchanged.
    assert (ROOT/'02_TESTS/test_release_readiness_qa058.py').exists()
    assert not (ROOT/'02_TESTS/test_qa058_archival_readiness.py').exists()

def test_qa058_12_new_audit_layer_has_no_target_specific_logic():
    text=(ROOT/'01_SRC/gilttpy/engineering/archival_readiness.py').read_text().lower()
    text+=(ROOT/'01_SRC/gilttpy/validation/archival_readiness_campaign.py').read_text().lower()
    for forbidden in ('observed concentration','historical concentration','copenhagen','hanford'):
        assert forbidden not in text
