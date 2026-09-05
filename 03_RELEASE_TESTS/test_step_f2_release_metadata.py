from pathlib import Path
import json, re, tomllib

ROOT = Path(__file__).resolve().parents[1]


def project():
    with (ROOT/'pyproject.toml').open('rb') as f:
        return tomllib.load(f)


def test_version_and_backend_are_release_activated():
    p = project()
    assert p['project']['version'] == '2.0.0'
    assert any(req.startswith('setuptools>=77.0.3') for req in p['build-system']['requires'])


def test_pep639_bsd3_license_is_active_and_materialized():
    p = project()['project']
    assert p['license'] == 'BSD-3-Clause'
    assert p['license-files'] == ['LICENSE']
    text = (ROOT/'LICENSE').read_text()
    assert 'Redistribution and use in source and binary forms' in text
    assert 'Neither the name of the copyright holder nor the names of its contributors' in text


def test_authorship_orcid_and_doi_are_not_inferred():
    p = project()['project']
    assert 'authors' not in p and 'maintainers' not in p
    state = json.loads((ROOT/'metadata/step_f2_release_state.json').read_text())
    assert state['software_authorship'] is None
    assert state['author_orcids'] is None
    assert state['zenodo_version_doi'] is None
    assert state['zenodo_concept_doi'] is None
    assert not (ROOT/'CITATION.cff').exists()


def test_raw_copenhagen_hanford_data_are_not_redistributed():
    state = json.loads((ROOT/'metadata/step_f2_release_state.json').read_text())
    assert state['raw_copenhagen_hanford_redistributed'] is False
    data_ext = {'.csv','.tsv','.xlsx','.xls','.parquet','.feather','.h5','.hdf5','.nc','.npz','.npy'}
    suspicious=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix.lower() in data_ext:
            n=p.name.lower()
            if 'copenhagen' in n or 'hanford' in n:
                suspicious.append(str(p.relative_to(ROOT)))
    assert suspicious == []


def test_ci_declares_exact_12_cell_matrix():
    text=(ROOT/'.github/workflows/release-ci.yml').read_text()
    for os_name in ('ubuntu-latest','macos-latest','windows-latest'):
        assert os_name in text
    for py in ('3.11','3.12','3.13','3.14'):
        assert f'"{py}"' in text


def test_release_state_does_not_promote_external_platform_evidence():
    state=json.loads((ROOT/'metadata/step_f2_release_state.json').read_text())
    assert state['repository_created'] is False
    assert state['external_ci_complete'] is False
    assert state['zenodo_deposition_complete'] is False
    assert state['status'] == 'PROMOTION_CANDIDATE_EXTERNAL_HOLDS_REMAIN'
