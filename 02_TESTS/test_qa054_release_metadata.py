from pathlib import Path
import json
import tomllib
from gilttpy.engineering.release_metadata import *
from gilttpy.validation.release_metadata_campaign import qa054_metadata_evidence
ROOT=Path(__file__).resolve().parents[1]

def test_qa054_01_gate_holds_prohibitions_explicit():
    assert QA054_GATE=="PASS_SCIENTIFIC_METADATA_CITATION_LICENSING_CONTRACT"
    assert "HOLD_SOFTWARE_LICENSE_SELECTION" in QA054_HOLDS
    assert "HOLD_PEP639_BACKEND_UPGRADE_UNTIL_LICENSE_ACTIVATION" in QA054_HOLDS
    assert "PROHIBIT_GUESSED_ORCID" in QA054_PROHIBITIONS
    assert "PROHIBIT_TARGET_TUNING" in QA054_PROHIBITIONS

def test_qa054_02_known_project_metadata_matches_pyproject():
    state=load_state(ROOT/'metadata/qa054_release_metadata_state.json')
    project=load_project_metadata(ROOT/'pyproject.toml')
    assert state['known']['package_name']==project['name']=='gilttpy'
    assert state['known']['development_version']==project['version']=='0.0.0.dev1'
    assert state['known']['requires_python']==project['requires-python']=='>=3.11,<3.15'

def test_qa054_03_release_critical_fields_are_explicit_holds():
    state=load_state(ROOT/'metadata/qa054_release_metadata_state.json')
    for name in RELEASE_CRITICAL_FIELDS:
        assert state['fields'][name]['status']=='HOLD' and state['fields'][name]['value'] is None
    assert public_release_ready(state) is False

def test_qa054_04_citation_template_targets_cff_1_2_0_and_software():
    text=(ROOT/'CITATION.cff.template').read_text()
    assert 'cff-version: 1.2.0' in text
    assert 'title: "GILTT-Py 2.0"' in text
    assert 'type: software' in text
    assert CFF_SCHEMA_VERSION=='1.2.0'

def test_qa054_05_template_placeholders_block_public_release_readiness():
    assert citation_template_has_placeholder(ROOT/'CITATION.cff.template')
    assert not (ROOT/'CITATION.cff').exists()
    assert_no_public_citation_with_placeholders(ROOT)

def test_qa054_06_pyproject_does_not_guess_license():
    project=load_project_metadata(ROOT/'pyproject.toml')
    assert 'license' not in project and 'license-files' not in project
    assert PEP639_MIN_SETUPTOOLS=='77.0.3'
    assert CORE_METADATA_LICENSE_VERSION=='2.4'

def test_qa054_07_pyproject_does_not_guess_software_authors():
    project=load_project_metadata(ROOT/'pyproject.toml')
    assert 'authors' not in project and 'maintainers' not in project

def test_qa054_08_license_policy_separates_software_data_and_manuscript():
    text=(ROOT/'LICENSE_POLICY.md').read_text().lower()
    assert 'software licensing' in text and 'benchmark-data redistribution rights' in text
    assert 'manuscript/supplement licensing' in text
    assert 'does not automatically license the software' in text

def test_qa054_09_source_authorship_cannot_be_promoted_by_inference():
    text=(ROOT/'QA054_SCOPE_GOVERNANCE.md').read_text().lower()
    assert 'does not by itself establish authorship of giltt-py software' in text
    assert 'no orcid may be guessed' in text

def test_qa054_10_metadata_evidence_reports_hold_not_release_ready():
    ev=qa054_metadata_evidence(ROOT)
    assert ev['license_declared_in_pyproject'] is False
    assert ev['authors_declared_in_pyproject'] is False
    assert ev['public_release_ready'] is False
    assert ev['cff_schema_target']=='1.2.0'
    assert ev['pep639_min_setuptools']=='77.0.3'
    assert ev['core_metadata_license_version']=='2.4'
    assert 'software_authorship' in ev['unresolved_fields']

def test_qa054_11_reproducibility_protocol_requirements_are_reflected():
    text=(ROOT/'QA054_SCOPE_GOVERNANCE.md').read_text()
    assert 'Data redistribution rights are audited separately' in text
    assert 'public release version, release date, DOI and repository URL are deferred' in text

def test_qa054_12_engineering_metadata_layer_has_no_transport_target_logic():
    text=(ROOT/'01_SRC/gilttpy/engineering/release_metadata.py').read_text().lower()
    for forbidden in ('copenhagen','hanford','observed concentration','historical concentration'):
        assert forbidden not in text
