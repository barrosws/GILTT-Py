from pathlib import Path
import json
import pytest
from gilttpy.engineering.dependency_lock import *
from gilttpy.validation.dependency_lock_campaign import installed_versions
ROOT=Path(__file__).resolve().parents[1]

def test_qa053_01_gate_holds_prohibitions():
    assert QA053_GATE=="PASS_DEPENDENCY_LOCK_AND_ENVIRONMENT_RESOLUTION_CONTRACT"
    assert "HOLD_CROSS_PLATFORM_HASH_LOCK_GENERATION" in QA053_HOLDS
    assert "PROHIBIT_PIP_FREEZE_AS_UNIVERSAL_CROSS_PLATFORM_LOCK" in QA053_PROHIBITIONS
    assert "PROHIBIT_TARGET_TUNING" in QA053_PROHIBITIONS

def test_qa053_02_canonical_resolution_order_and_normalization():
    r=canonical_resolution({'B_Pkg':'2','a-pkg':'1'}); assert [x.name for x in r]==['a-pkg','b-pkg']

def test_qa053_03_resolution_fingerprint_deterministic():
    a=canonical_resolution({'b':'2','a':'1'}); b=canonical_resolution({'a':'1','b':'2'}); assert resolution_fingerprint(a)==resolution_fingerprint(b)

def test_qa053_04_hash_lock_accepts_exact_hashed_entries():
    assert validate_hash_lock_lines(['a==1 --hash=sha256:'+'0'*64])

def test_qa053_05_hash_lock_rejects_unhashed_exact_pin():
    with pytest.raises(ValueError): validate_hash_lock_lines(['a==1'])

def test_qa053_06_hash_lock_rejects_range():
    with pytest.raises(ValueError): validate_hash_lock_lines(['a>=1 --hash=sha256:'+'0'*64])

def test_qa053_07_local_resolution_is_explicitly_nonhermetic():
    d=json.loads((ROOT/'requirements/qa053_local_resolution_py313_linux_x86_64.json').read_text()); assert d['evidence']=='LOCAL_RESOLUTION' and d['hermetic'] is False and d['cross_platform'] is False

def test_qa053_08_local_resolution_matches_direct_runtime_stack():
    d=json.loads((ROOT/'requirements/qa053_local_resolution_py313_linux_x86_64.json').read_text()); assert {x['name']:x['version'] for x in d['packages']}==installed_versions()

def test_qa053_09_template_is_not_executable_evidence():
    x=(ROOT/'requirements/qa053_hash_lock_TEMPLATE.txt').read_text(); assert 'TEMPLATE ONLY' in x and 'NOT AN EXECUTED LOCK' in x

def test_qa053_10_scope_rejects_pip_freeze_universal_semantics():
    x=(ROOT/'QA053_SCOPE_GOVERNANCE.md').read_text(); assert 'pip freeze' in x and 'not a universal lock' in x

def test_qa053_11_science_separation_is_explicit():
    x=(ROOT/'QA053_SCOPE_GOVERNANCE.md').read_text(); assert 'does not modify or revalidate scientific equations' in x and 'NO TARGET TUNING' in x

def test_qa053_12_engineering_module_has_no_transport_targets():
    x=(ROOT/'01_SRC/gilttpy/engineering/dependency_lock.py').read_text().lower(); assert 'copenhagen' not in x and 'hanford' not in x and 'observed concentration' not in x
