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


def test_cross_platform_ci_does_not_promote_local_reference_pins_to_universal_lock():
    ref=(ROOT/'requirements/qa051_reference_py313_linux_x86_64.txt').read_text()
    resolution=json.loads((ROOT/'requirements/qa053_local_resolution_py313_linux_x86_64.json').read_text())
    assert 'NOT A CROSS-PLATFORM LOCK FILE' in ref
    assert resolution['evidence'] == 'LOCAL_RESOLUTION'
    assert resolution['hermetic'] is False
    assert resolution['cross_platform'] is False


def test_cross_platform_ci_excludes_only_local_runtime_equality_checks():
    text=(ROOT/'.github/workflows/release-ci.yml').read_text()
    assert '--deselect=02_TESTS/test_qa051_environment_ci_matrix.py::test_qa051_06_local_reference_versions_are_explicit_and_match_runtime' in text
    assert '--deselect=02_TESTS/test_qa053_dependency_lock.py::test_qa053_08_local_resolution_matches_direct_runtime_stack' in text
    for exact_pin in ('numpy==2.3.5','scipy==1.17.0','mpmath==1.3.0','pytest==9.0.2'):
        assert exact_pin not in text
